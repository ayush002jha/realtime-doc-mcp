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
#     return doc_search(technology, task, version)

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
