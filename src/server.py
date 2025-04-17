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
from dotenv import load_dotenv
load_dotenv()

# MASA Data API base URL 
MASA_BASE_URL = "https://data.dev.masalabs.ai"

MASA_API_KEY = os.environ.get("MASA_DATA_API_KEY")
if MASA_API_KEY is None:
    raise EnvironmentError("MASA_API_KEY not found in .env file or environment variables!")
    
# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the MCP server instance.
mcp = FastMCP("DocSearchContextServer")

@mcp.tool()
def obtain_relevant_doc_urls(technology: str, task: str, version: str = None) -> str:
    """
    Searches for documentation links using DuckDuckGo. You need to run doc_context to get the context/data of the documentation. This will only provide links.
    
    Parameters:
        technology (str): e.g. 'Next.js'
        task (str): e.g. 'API route'
        version (str, optional): e.g. '13'
    
    Returns:
        str: A newline-delimited list of documentation links.
    """
    # Append version info if provided.
    query = f"{technology} {version} {task}" if version else f"{technology} {task}"
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=5):
            results.append(f"{result['title']}: {result['href']}")
    return "\n".join(results) if results else "No documentation links found."

@mcp.tool()
def get_latest_version(technology: str) -> str:
    """
    Fetches the latest stable version for the specified technology using DuckDuckGo.
    
    Parameters:
        technology (str): e.g. 'Next.js'
    
    Returns:
        str: The extracted version string, or a message if not found.
    """
    query = f"latest stable version of {technology}"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=1))
        if results:
            version_info = results[0]['title']
            # Try to extract a version number using regex (e.g., 13, 13.1, etc.)
            match = re.search(r'(\d+(\.\d+)+)', version_info)
            if match:
                return match.group(0)
            return version_info
        return "No version found."

# @mcp.tool()
# def doc_search_with_latest(technology: str, task: str) -> str:
#     """
#     Searches for documentation links by automatically determining the latest stable
#     version for the technology and then calling doc_search.
    
#     Parameters:
#         technology (str): e.g. 'Next.js'
#         task (str): e.g. 'API route'
    
#     Returns:
#         str: The documentation search results including the version info.
#     """
#     version = get_latest_version(technology)
#     return obtain_relevant_doc_urls(technology, task, version)

async def lightweight_scrape(url: str) -> str:
    """
    Asynchronously fetches a web page using aiohttp, extracts its main content via readability‑lxml,
    and converts it to markdown using html2text.
    
    Returns:
         str: The untrimmed markdown text of the main content, or an error message.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                html = await response.text()
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"
    
    try:
        # Use readability to extract the main article content.
        doc = Document(html)
        content_html = doc.summary()
        # Convert HTML to markdown.
        h = html2text.HTML2Text()
        h.ignore_links = False  # Set True to remove links if desired.
        markdown = h.handle(content_html)
        return markdown
    except Exception as e:
        return f"Error processing {url}: {str(e)}"

def extract_relevant_context_from_texts(texts: list[str], query: str, top_k: int = 3) -> str:
    """
    Given a list of markdown texts and a query, compute vector embeddings and
    return the top_k most relevant texts.
    
    Parameters:
         texts (list[str]): List of markdown texts.
         query (str): The query to match.
         top_k (int): Number of top texts to return.
    
    Returns:
         str: The concatenated snippets with relevance scores.
    """
    if not texts:
        return "No texts available for context extraction."
    
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scored_texts = []
    for text in texts:
        if text:
            doc_embedding = embedding_model.encode(text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            scored_texts.append((score, text))
    
    scored_texts.sort(key=lambda x: x[0], reverse=True)
    top_texts = scored_texts[:top_k]
    
    output = []
    for score, snippet in top_texts:
        output.append(f"Score: {score:.3f}\n{snippet}\n")
    return "\n".join(output)

@mcp.tool()
async def masa_scrape(url: str, depth: int = 2, timeout_sec: int = 30) -> str:
    """
    Performs a complete real-time web scrape using the MASA Data API:
      1. Creates a scraping job for the given URL.
      2. Polls the job status until completion.
      3. Retrieves and returns the scraping results.
    
    Parameters:
        url (str): The URL to scrape.
        depth (int): The scraping depth (default: 2).
        timeout_sec (int): The timeout for scraping (default: 30 seconds).
        api_key (str): API key for MASA Data API authorization.
    
    Returns:
        str: The scraped results (formatted as a string), or an error message.
    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MASA_API_KEY}"
    }
    create_endpoint = f"{MASA_BASE_URL}/api/v1/search/live/web"
    payload = {"url": url, "depth": depth, "timeout": timeout_sec}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Create a scraping job.
            async with session.post(create_endpoint, json=payload, headers=headers) as response:
                json_response = await response.json()
                if "uuid" not in json_response:
                    return f"Error creating job: {json_response.get('error', 'Unknown error')}"
                job_uuid = json_response["uuid"]
    except Exception as e:
        return f"Exception during job creation: {str(e)}"
    
    # Step 2: Poll the job status.
    status_endpoint = f"{MASA_BASE_URL}/api/v1/search/live/web/status/{job_uuid}"
    max_attempts = 15  # e.g. poll for up to 30 seconds (15 * 2 seconds)
    for attempt in range(max_attempts):
        await asyncio.sleep(2)  # Wait for 2 seconds between polls.
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(status_endpoint, headers=headers) as resp:
                    status_json = await resp.json()
                    status = status_json.get("status", "").lower()
                    if status in ["completed", "finished", "done"]:
                        break
                    elif status in ["error", "failed"]:
                        return f"Job failed with status: {status}"
        except Exception as e:
            return f"Exception during job status polling: {str(e)}"
    else:
        return "Job did not complete within the expected time."
    
    # Step 3: Retrieve the job results.
    result_endpoint = f"{MASA_BASE_URL}/api/v1/search/live/web/result/{job_uuid}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(result_endpoint, headers=headers) as resp:
                result_json = await resp.json()
                if "results" in result_json:
                    results = result_json["results"]
                    formatted_results = "\n".join([
                        f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nSnippet: {item.get('snippet', '')}"
                        for item in results
                    ])
                    return formatted_results if formatted_results else "No results found."
                else:
                    return f"Error retrieving results: {result_json.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Exception during result retrieval: {str(e)}"

@mcp.tool()
async def obtain_doc_context_from_urls(urls: list[str], query: str) -> str:
    """
    Extracts and processes documentation context from a list of URLs:
      1. Asynchronously fetches each URL and converts its main content into Markdown.
      2. Evaluates and ranks the markdown texts against the provided query.
    
    Parameters:
         urls (list[str]): List of documentation URLs.
         query (str): The query to rank relevance.
    
    Returns:
         str: The most relevant markdown context based on the query.
    """
    markdown_texts = await asyncio.gather(*(lightweight_scrape(url) for url in urls))
    context = extract_relevant_context_from_texts(markdown_texts, query)
    return context

# if __name__ == "__main__":
#     mcp.run()
