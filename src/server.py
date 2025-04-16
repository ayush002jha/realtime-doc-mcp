import sys
sys.stdout.reconfigure(encoding='utf-8')
import asyncio
import aiohttp
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util

# Use readability to extract main content and html2text to convert HTML to Markdown.
from readability import Document
import html2text


# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the MCP server instance.
mcp = FastMCP("DocSearchContextServer")

@mcp.tool()
def doc_search(technology: str, task: str, version: str = None) -> str:
    """
    Searches for documentation links using DuckDuckGo.
    
    Parameters:
        technology (str): e.g. 'Next.js'
        task (str): e.g. 'API route'
        version (str, optional): e.g. '13'
    
    Returns:
        str: A newline-delimited list of documentation links.
    """
    query = f"{technology} {version} {task}" if version else f"{technology} {task}"
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=5):
            results.append(f"{result['title']}: {result['href']}")
    return "\n".join(results) if results else "No documentation links found."

async def lightweight_scrape(url: str) -> str:
    """
    Asynchronously fetches a web page using aiohttp, extracts the main content using
    readability-lxml, and converts it to markdown via html2text.
    
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
        # Convert the content HTML to Markdown using html2text.
        h = html2text.HTML2Text()
        h.ignore_links = False  # Set to True if you want to remove links.
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
         query (str): The query to match against.
         top_k (int): Number of top texts to return.
    
    Returns:
         str: Concatenated snippets with relevance scores.
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
async def doc_context(urls: list[str], query: str) -> str:
    """
    Extracts and processes documentation context from a list of URLs by:
      1. Asynchronously fetching each URL and converting its main content into Markdown.
      2. Evaluating the relevance of each Markdown document against the query.
    
    Parameters:
         urls (list[str]): List of documentation URLs.
         query (str): The query to rank content.
    
    Returns:
         str: The most relevant Markdown context based on the query.
    """
    # Fetch markdown content concurrently from all provided URLs.
    markdown_texts = await asyncio.gather(*(lightweight_scrape(url) for url in urls))
    
    # Process the extracted texts to determine relevance.
    context = extract_relevant_context_from_texts(markdown_texts, query)
    return context

# if __name__ == "__main__":
#     mcp.run()
