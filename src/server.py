import asyncio
import requests
from bs4 import BeautifulSoup

from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

# For vector embeddings
from sentence_transformers import SentenceTransformer, util

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the MCP server instance.
mcp = FastMCP("DocSearchContextServer")


@mcp.tool()
def doc_search(technology: str, task: str, version: str = None) -> str:
    """
    Tool to search for documentation links using DuckDuckGo.

    Parameters:
        technology (str): The name of the technology (e.g., 'Next.js').
        task (str): The specific task (e.g., 'API route').
        version (str, optional): The version of the technology (e.g., '13').

    Returns:
        str: A newline-delimited list of documentation links.
    """
    query = f"{technology} {version} {task}" if version else f"{technology} {task}"
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=5):
            results.append(f"{result['title']}: {result['href']}")
    return "\n".join(results) if results else "No documentation links found."


def simple_scrape(url: str) -> str:
    """
    A simple scraper that fetches the HTML content of a URL and extracts the text.
    In production, replace this with a dedicated scraper (e.g. Crawl4AI) that handles dynamic content.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements.
            for element in soup(["script", "style"]):
                element.decompose()
            text = soup.get_text(separator="\n")
            # Optionally, clean/shorten the text.
            return text.strip()
        else:
            return ""
    except Exception as e:
        return ""


def extract_relevant_context_from_texts(texts: list[str], query: str, top_k: int = 3) -> str:
    """
    Given a list of document texts and a query, compute embeddings and
    return the top_k most relevant text snippets.
    """
    if not texts:
        return "No texts available for context extraction."

    # Compute the embedding for the query.
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    scored_texts = []
    for text in texts:
        # For efficiency, you might want to chunk the document into smaller pieces.
        # Here, we compute an embedding over the full text.
        if text:
            doc_embedding = embedding_model.encode(text, convert_to_tensor=True)
            # Compute cosine similarity between the query and the document.
            score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            scored_texts.append((score, text))

    # Sort texts by relevance score (descending order) and take top_k.
    scored_texts.sort(key=lambda x: x[0], reverse=True)
    top_texts = scored_texts[:top_k]

    # For readability, return the top snippets with their score.
    output = []
    for score, snippet in top_texts:
        snippet_excerpt = snippet[:500].replace("\n", " ")  # first 500 characters
        output.append(f"Score: {score:.3f}\n{text_excerpt(snippet_excerpt)}\n")

    return "\n".join(output)


def text_excerpt(text: str, max_len: int = 500) -> str:
    """Helper to clean and trim text to a maximum length."""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


@mcp.tool()
def doc_context(urls: list[str], query: str) -> str:
    """
    Tool to extract and process documentation context from a list of URLs based on a query.
    
    Parameters:
        urls (list[str]): List of documentation URLs.
        query (str): The query for which context is desired.
    
    Returns:
        str: The most relevant contextual information extracted from the URLs.
    """
    # Step 1: Scrape each URL.
    scraped_texts = []
    for url in urls:
        text = simple_scrape(url)
        if text:
            scraped_texts.append(text)

    # Step 2: Process the extracted texts using vector embeddings to determine relevance.
    context = extract_relevant_context_from_texts(scraped_texts, query)
    return context

