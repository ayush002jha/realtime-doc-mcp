# Implementation Guide: RealtimeDocContext MCP

This guide details the implementation of the RealtimeDocContext MCP server and client examples.

## Architecture

The implementation follows a client-server architecture based on the Model Context Protocol (MCP).

1.  **Server (`server.py`, `server_sse.py`):**
    *   Built using Python and the `mcp-python` library (`FastMCP`).
    *   **Asynchronous:** Leverages `asyncio` and `aiohttp` for efficient handling of I/O-bound operations (network requests to MASA, Tavily, web scraping).
    *   **Tool-Based:** Exposes functionality as discrete, callable tools (`get_latest_version_tech`, `fetch_relevant_doc_urls`, `scrape_multiple_urls_to_get_context`).
    *   **Modular Transport:** Offers two transport options:
        *   `server.py`: Uses the default `FastMCP` transport, typically STDIO.
        *   `server_sse.py`: Integrates `FastMCP` with `SseServerTransport` and the Starlette ASGI framework to provide an HTTP-based Server-Sent Events interface.
    *   **External Dependencies:** Relies on external APIs (MASA, Tavily) and Python libraries for search (DDGS), scraping (Readability, html2text), and semantic analysis (Sentence Transformers).

2.  **Clients (`client_cli.py`, `client_sse_cli.py`, `app.py`):**
    *   Python-based examples demonstrating how to interact with the MCP server.
    *   Utilize `mcp-python` client libraries (`stdio_client`, `sse_client`) corresponding to the server transport.
    *   **LLM Integration:** Integrate with Google Gemini (`google-generativeai`) to drive the conversation and tool usage. The clients act as wrappers, mediating between the LLM and the MCP server.
    *   **Tool Adaptation:** Include logic (`convert_mcp_tools_to_gemini`, `clean_schema`) to adapt the MCP tool definitions for Gemini's function calling API.
    *   **User Interfaces:** Provide different interaction methods:
        *   Command-Line Interface (CLI) for both STDIO and SSE servers, featuring colored output and spinners (`colorama`, `threading`).
        *   Gradio Web UI (`app.py`) for a user-friendly chat experience with the SSE server.

## Components

*   **`server.py`:** Standalone MCP server implementation using STDIO transport. Contains tool definitions and logic for interacting with external services.
*   **`server_sse.py`:** MCP server implementation using SSE transport. Wraps the core tool logic from `server.py` within a Starlette application using `SseServerTransport`.
*   **`client_cli.py`:** Command-line client designed to connect to `server.py` via STDIO. Includes Gemini integration, UI enhancements, and tool adaptation logic.
*   **`client_sse_cli.py`:** Command-line client designed to connect to `server_sse.py` via its SSE URL. Shares much of the core logic and UI features with `client_cli.py`.
*   **`app.py`:** Gradio-based web application client connecting to `server_sse.py` via its SSE URL. Provides a chatbot interface, manages connection state, and displays tool usage status. Uses an `AsyncProcessor` class to handle asynchronous MCP operations within the Gradio framework.
*   **`pyproject.toml`:** Defines project metadata and dependencies for installation using tools like `uv` or `pip`.
*   **`.env` (Required):** File to store necessary API keys:
    *   `MASA_DATA_API_KEY`: For MASA API access (Server).
    *   `TAVILY_API_KEY`: For Tavily Search API access (Server).
    *   `GEMINI_API_KEY`: For Google Gemini API access (Clients).
    *   `MCP_SERVER_URL` (Optional for `app.py`): Default URL for the Gradio app.
*   **Key Libraries (to be listed in `pyproject.toml`):**
    *   `mcp-python`: Core MCP server/client functionality.
    *   `google-generativeai`: Gemini LLM interaction.
    *   `sentence-transformers`: Embeddings for semantic search.
    *   `torch` # Or `torch-cpu` depending on setup
    *   `tavily-python`: Tavily Search API client.
    *   `duckduckgo-search`: DuckDuckGo search client.
    *   `aiohttp`: Asynchronous HTTP requests.
    *   `readability-lxml`: HTML content extraction.
    *   `html2text`: HTML to Markdown conversion.
    *   `starlette`: ASGI framework for SSE.
    *   `uvicorn[standard]`: ASGI server for SSE.
    *   `gradio`: Web UI framework.
    *   `python-dotenv`: Loading environment variables.
    *   `colorama`: Colored terminal output (CLI clients).

## Setup

1.  **Install `uv`:** If you don't have `uv` installed, follow the official installation instructions: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)
    ```bash
    # Example using curl:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone Repository:** Obtain the project code.
    ```bash
    git clone https://github.com/ayush002jha/realtime-doc-mcp.git
    cd src
    ```
3.  **Create Virtual Environment (using `uv`):**
    ```bash
    uv venv
    ```
    This creates a `.venv` directory.
4.  **Activate Virtual Environment:**
    ```bash
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows (Command Prompt):
    .\.venv\Scripts\activate.bat
    # On Windows (PowerShell):
    .\.venv\Scripts\Activate.ps1
    ```
    *(Your prompt should now indicate you are in the virtual environment)*
5.  **Install Dependencies (using `uv` and `pyproject.toml`):**
    ```bash
    uv sync
    ```
    This command reads the `pyproject.toml` file and installs all specified dependencies into your virtual environment.
    *Note: Ensure all libraries listed under "Key Libraries" above are correctly specified in your `pyproject.toml` under `[project.dependencies]` or similar.*
6.  **Configure API Keys:** Create a file named `.env` in the project's root directory and add your API keys:
    ```dotenv
    MASA_DATA_API_KEY="your_masa_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    # Optional: Set default SSE URL for Gradio app
    # MCP_SERVER_URL="http://localhost:8000/sse"
    ```
7.  **Run the Server:** Choose the desired server:
    *   **STDIO Server:**
        ```bash
        # Ensure you are in the activated virtual environment
        python server.py
        ```
    *   **SSE Server:**
        ```bash
        # Ensure you are in the activated virtual environment
        # Host on 0.0.0.0, port 8000
        uvicorn server_sse:app --host 0.0.0.0 --port 8000
        ```
        *(You might need to adjust host/port if 8000 is occupied)*

## Usage

1.  **CLI Client (STDIO):**
    *   Ensure `server.py` is *not* running separately (the client will start it).
    *   Run the client, providing the path to `server.py`:
        ```bash
        python client_cli.py server.py
        ```
    *   Follow the on-screen prompts. Enter your queries. Type `quit` to exit.

2.  **CLI Client (SSE):**
    *   Ensure the SSE server (`server_sse.py`) is running (e.g., via `uvicorn`).
    *   Run the client, providing the *full SSE URL*:
        ```bash
        # If server is running locally on port 8000:
        python client_sse_cli.py http://localhost:8000/sse
        # If server is hosted remotely (e.g., via ngrok):
        # python client_sse_cli.py https://your-ngrok-url.ngrok-free.app/sse
        ```
    *   Follow the on-screen prompts. Enter your queries. Type `quit` to exit.

3.  **Gradio Web UI Client (SSE):**
    *   Ensure the SSE server (`server_sse.py`) is running.
    *   Run the Gradio app:
        ```bash
        python app.py
        ```
    *   Open the URL provided by Gradio (usually `http://127.0.0.1:7860`).
    *   Enter the SSE server URL in the "MCP Server URL" field (it might default correctly if `MCP_SERVER_URL` is set in `.env` or the server runs locally).
    *   Click "🔌 Connect".
    *   Once connected, use the chat interface to enter queries. Tool usage status will appear during processing.

4.  **Third-Party Clients (Cursor, Claude Desktop):**
    *   Configure the SSE server URL in the respective application's settings (`config.json`) as shown in the SPECIFICATION.md "Integration Guidelines" section. Ensure the SSE server is running and accessible from where the client application is running.
    **Example Configurations (SSE):**
    *   **Cursor:** Add the server to `settings.json` under `mcpServers`:
        ```json
        {
          "mcpServers": {
            "RealtimeDocContext": {
              "url": "YOUR_SSE_SERVER_URL/sse"
            }
          }
        }
        ```
    *   **Claude Desktop:** Use `mcp-remote` (install via `npm i -g mcp-remote`) in `settings.json`:
        ```json
        {
          "mcpServers": {
            "RealtimeDocContext": {
              "command": "npx",
              "args": [
                "mcp-remote",
                "YOUR_SSE_SERVER_URL/sse"
              ]
            }
          }
        }
        ```
    *   Replace `YOUR_SSE_SERVER_URL` with the actual URL where `server_sse.py` is hosted (e.g., `http://localhost:8000` or a public URL like the ngrok example).

5.  **Example Queries:**
    *   "What is the latest stable version of Node.js?" (Triggers `get_latest_version_tech`)
    *   "How do I implement authentication in FastAPI?" (Triggers `fetch_relevant_doc_urls`, then `scrape_multiple_urls_to_get_context`)
    *   "Show me the documentation for creating middleware in Express.js version 4." (Triggers `fetch_relevant_doc_urls` with version, then `scrape_multiple_urls_to_get_context`)

## Performance

*   **Latency:** The primary source of latency is network requests to external services (Tavily, MASA, website scraping) and the response time of the LLM (Gemini). The `scrape_multiple_urls_to_get_context` tool involves multiple fetches and local embedding/ranking, adding noticeable delay.
*   **Concurrency:** The use of `asyncio` in servers and clients allows for efficient handling of concurrent I/O operations, especially relevant for the SSE server which might handle multiple client connections.
*   **Resource Usage:**
    *   **CPU:**  CPU usage occurs during embedding generation by `sentence-transformers`.
    *   **Memory:** The embedding model consumes considerable RAM. Memory usage also depends on the size of scraped content being processed.
    *   **Network:** Dependent on the number and size of API calls and web pages scraped.
*   **Transport:**
    *   **STDIO:** Lowest overhead, suitable for local use. Does not scale to multiple clients easily.
    *   **SSE:** Introduces network latency but allows for remote access, web UIs, and persistent connections which can be more efficient than repeated HTTP requests for ongoing interactions.

## Testing

*   **Functional Testing:** The provided client examples (`client_cli.py`, `client_sse_cli.py`, `app.py`) serve as primary functional tests, verifying the end-to-end flow from user query to tool invocation and response generation via different transports.
*   **Manual Testing:** Execute various queries targeting different tools and scenarios (e.g., queries requiring version checks, simple URL fetching, complex scraping/ranking) using the clients to ensure correct behavior and tool chaining.
*   **Compatibility:** Tested primarily with Google Gemini and Claude Sonnet. Adapters (`convert_mcp_tools_to_gemini`, `clean_schema`) are included, suggesting potential compatibility with other LLMs supporting similar tool-calling mechanisms, but further testing would be required.
