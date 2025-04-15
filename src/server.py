import asyncio
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
import sys
print("Python executable:", sys.executable,file=sys.stderr)

# Create an MCP server instance with a descriptive name.
mcp = FastMCP("DocSearchServer")

@mcp.tool()
def doc_search(technology: str, task: str, version: str = None) -> str:
    """
    Search for documentation links using DuckDuckGo.

    Parameters:
        technology (str): The name of the technology (e.g., 'Next.js').
        task (str): The specific task (e.g., 'API route').
        version (str, optional): The version of the technology (e.g., '13').

    Returns:
        str: A list of documentation links or an informative message.
    """
    # Build the query string using the provided input.
    query = f"{technology} {version} {task}" if version else f"{technology} {task}"
    
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=5):
            results.append(f"{result['title']}: {result['href']}")
    
    return "\n".join(results) if results else "No documentation links found."

# if __name__ == "__main__":
#     mcp.run()
