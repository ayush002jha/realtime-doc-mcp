import sys
sys.stdout.reconfigure(encoding='utf-8')
import asyncio
import aiohttp
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
from readability import Document
import html2text
import re
import os
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

# MASA Data API base URL 
MASA_BASE_URL = "https://data.dev.masalabs.ai"

MASA_API_KEY = os.environ.get("MASA_DATA_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if TAVILY_API_KEY is None:
    raise EnvironmentError("TAVILY_API_KEY not found in environment variables or .env file!")

if MASA_API_KEY is None:
    raise EnvironmentError("MASA_API_KEY not found in .env file or environment variables!")
    
# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the MCP server instance.
mcp = FastMCP("DocSearchContextServer")

# Initialize Tavily client
tavily = TavilyClient(TAVILY_API_KEY)

@mcp.tool()
async def get_latest_version_tech(technology: str) -> str:
    """
    Fetches the latest stable version for the specified technology using DuckDuckGo.
    Use this as the first step when you need the most up-to-date documentation.

    Parameters:
        technology (str): Any technology name
    
    Returns:
        str: The extracted version string, or a message if not found.
    """
    query = f"latest stable version of {technology}"
    try:
        response = tavily.search(query=query, include_answer="basic")
        answer = response.get("answer", "").strip()
        # Extract version using regex
        m = re.search(r"\b\d+(?:\.\d+){1,3}\b", answer)
        if m:
            return m.group(0)
        # Fallback to full answer
        return answer or "No version found."
    except Exception as e:
        return f"Error fetching version via Tavily: {e}"


@mcp.tool()
def fetch_relevant_doc_urls(technology: str, task: str, version: float = None) -> str:
    """
    Searches for documentation links using DuckDuckGo. You need to run doc_context to get the context/data of the documentation. This will only provide links.
    Call this after obtaining the latest version using `get_latest_version`.

    Parameters:
        technology (str): Any technology name 
        task (str): Any task e.g. 'create API route', 'write backend server', etc...
        version (float, optional): Version number e.g. 13, 15.4, etc... can be obtained from calling tool 'get_latest_version'
    
    Returns:
        str: A newline-delimited list of documentation links.
    """
    # Append version info if provided.
    version_str = str(version) if version is not None else ""
    query = f"{technology} {version_str} {task}".strip()
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=5):
            results.append(f"{result['title']}: {result['href']}")
    return "\n".join(results) if results else "No documentation links found."


async def masa_fetch_text(url: str) -> str:
    """
    Uses MASA /api/v1/search/live/web/scrape to get raw page text.
    Returns None on failure.
    """
    endpoint = f"{MASA_BASE_URL}/api/v1/search/live/web/scrape"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MASA_API_KEY}"
    }
    payload = {"url": url, "format": "text"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                data = await resp.json()
                if resp.status == 200:
                    return data.get("content", "")
    except:
        pass
    return None

async def lightweight_scrape(url: str) -> str:
    """
    Fetches HTML, extracts main body via readability, and converts to Markdown.
    Returns error message if fails.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                html = await resp.text()
    except Exception as e:
        return f"Error fetching {url}: {e}"
    try:
        doc = Document(html)
        content = doc.summary()
        h = html2text.HTML2Text()
        h.ignore_links = False
        return h.handle(content)
    except Exception as e:
        return f"Error processing {url}: {e}"

@mcp.tool()
async def scrape_multiple_urls_to_get_context(urls: list[str], query: str) -> str:
    """
    Scrape multiple URLs via MASA Data API, score them against `query`,
    and return the full scrape of the most relevant URL.
    Call this after getting the documentation URLs using `obtain_relevant_doc_urls`.

    Parameters:
      urls (list[str]): List of page URLs to scrape.
      query (str):     The user’s context query to rank pages.

    Returns:
      str: The complete scraped result (url and markdown content) of the best match,
           or an error if none succeed.
    """
    # 1. Fetch MASA text in parallel
    tasks = [masa_fetch_text(u) for u in urls]
    masa_texts = await asyncio.gather(*tasks)
    # 2. Build embeddings for successfully scraped pages
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    scored = []
    for url, text in zip(urls, masa_texts):
        if text:
            emb = embedding_model.encode(text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_emb, emb).item()
            scored.append((score, url))
    if not scored:
        return "All MASA scrapes failed or returned no content."
    # 3. Select best URL
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_url = scored[0]
    # 4. Return full Markdown 
    markdown = await lightweight_scrape(best_url)
    return (
        f"Best URL ({best_score:.3f} cosine match): {best_url}\n\n"
        f"{markdown}"
    )

if __name__ == "__main__":
    mcp.run()
