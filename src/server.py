import sys
sys.stdout.reconfigure(encoding='utf-8')
import asyncio
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util

# Import Crawl4AI classes based on the updated docs
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the MCP server instance
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

async def crawl_scrape(url: str) -> str:
    """
    Uses Crawl4AI’s asynchronous crawler to fetch a URL and generate
    fit markdown (filtered markdown) from the webpage.
    
    The configuration below applies a PruningContentFilter with:
      - threshold=0.45,
      - threshold_type="dynamic",
      - min_word_threshold=5.
    
    Returns:
         The full fit_markdown output if the crawl is successful,
         or an error message.
    """
    # Configure Crawl4AI with a PruningContentFilter for fit markdown
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.45,
            threshold_type="dynamic",
            min_word_threshold=5
        )
    )
    config = CrawlerRunConfig(
        markdown_generator=md_generator
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)
        if result.success:
            # Return the filtered "fit_markdown" version
            return result.markdown.fit_markdown
        else:
            return f"Error scraping {url}: {result.error_message}"

def extract_relevant_context_from_texts(texts: list[str], query: str, top_k: int = 3) -> str:
    """
    Given a list of markdown texts and a query, computes vector embeddings and
    returns the top_k most relevant texts.
    
    Parameters:
         texts (list[str]): List of markdown texts.
         query (str): The query to determine relevance.
         top_k (int): Number of top relevant texts to return.
    
    Returns:
         str: The best matching snippets with relevance scores.
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
    Extracts and processes documentation context from a list of URLs
    by using Crawl4AI to generate fit markdown (filtered by PruningContentFilter)
    and then scoring the returned markdown against a query.
    
    Parameters:
         urls (list[str]): List of documentation URLs.
         query (str): The context query to rank the content.
    
    Returns:
         str: The most relevant markdown context based on the query.
    """
    # Fetch fit markdown content concurrently from all URLs
    markdown_texts = await asyncio.gather(*(crawl_scrape(url) for url in urls))
    
    # Process extracted markdown texts to determine relevance
    context = extract_relevant_context_from_texts(markdown_texts, query)
    return context

# if __name__ == "__main__":
#     mcp.run()
