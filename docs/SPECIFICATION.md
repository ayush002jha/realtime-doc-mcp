# Model Context Protocol Specification: RealtimeDocContext

## Protocol Overview

The **RealtimeDocContext** MCP provides AI models with access to fresh, relevant information extracted from online documentation and web sources in real-time. It addresses the limitation of static training data by enabling LLMs to dynamically fetch the latest technology versions, find relevant documentation URLs, and scrape/process the content of those URLs to answer user queries accurately.

This protocol is designed as an MCP plugin, exposing specific data retrieval and processing capabilities as standardized "tools". Clients (like LLMs integrated via wrappers, CLIs, or web UIs) connect to the RealtimeDocContext server using either **STDIO** (for local execution) or **Server-Sent Events (SSE)** (for network accessibility) transports.

The core workflow involves a multi-step process often orchestrated by the LLM client:
1.  **(Optional but Recommended)** Identify the latest stable version of a technology (`get_latest_version_tech`).
2.  Find relevant documentation URLs based on the technology, user task, and optionally, the version (`fetch_relevant_doc_urls`).
3.  Scrape content from multiple candidate URLs, semantically rank them against the user's query, and return the most relevant content (`scrape_multiple_urls_to_get_context`).

This protocol leverages external services like the MASA Data API, Tavily Search API, and local processing via Sentence Transformers for semantic relevance.

## Core Components

1.  **MCP Server (`server.py`, `server_sse.py`):**
    *   The central component implementing the `mcp-python` server logic (`FastMCP`).
    *   Exposes the documentation context tools.
    *   Manages communication over STDIO or SSE (using Starlette for the SSE variant).
    *   Handles interaction with external APIs and local processing libraries.

2.  **Transport Layers:**
    *   **STDIO:** Enables direct command-line interaction between a local client and the server (`server.py`).
    *   **SSE (Server-Sent Events):** Provides a persistent HTTP-based connection suitable for web clients or remote access (`server_sse.py`). Exposes endpoints like `/sse` for connection and `/messages/` for data transfer.

3.  **Context Tools:**
    *   `get_latest_version_tech`: Fetches the latest stable version number for a given technology using Tavily Search.
    *   `fetch_relevant_doc_urls`: Searches for relevant documentation page URLs using DuckDuckGo Search based on technology, task, and version.
    *   `scrape_multiple_urls_to_get_context`: Orchestrates scraping (using MASA API and a lightweight fallback), embedding generation (Sentence Transformers), semantic similarity scoring, and returns the content of the best-matching URL for a given query.

4.  **External Services & Libraries:**
    *   **MASA Data API:** Used for initial, robust web scraping.
    *   **Tavily Search API:** Used for targeted information retrieval (latest versions).
    *   **DuckDuckGo Search (`duckduckgo-search` library):** Used for finding relevant documentation URLs.
    *   **Sentence Transformers:** Used locally for generating embeddings to calculate semantic similarity between user query and scraped content.
    *   **Readability & HTML2Text:** Used for extracting main content and converting HTML to Markdown in the fallback scraping mechanism.

5.  **MCP Client (`client_cli.py`, `client_sse_cli.py`, `app.py`):**
    *   The entity that connects to the MCP Server and consumes the tools.
    *   Typically wraps an LLM (like Google Gemini) to interpret user queries and orchestrate tool calls.
    *   Handles MCP communication (`initialize`, `list_tools`, `call_tool`).
    *   Translates MCP tool schemas into formats understood by the LLM (e.g., Gemini FunctionDeclarations).

## Interfaces

1.  **MCP Core Interface:** Adheres to the standard MCP interactions:
    *   `initialize`: Establishes the session and capabilities.
    *   `list_tools`: Client requests available tools; Server responds with a list including name, description, and input schema.
    *   `call_tool`: Client requests execution of a specific tool with arguments matching its schema; Server executes the tool and returns the result (or error).
    *   Data is exchanged in a JSON-based format defined by the MCP standard.

2.  **Transport Interfaces:**
    *   **STDIO:** Communication occurs over the standard input/output streams of the server process. Requires the client to spawn and manage the server process.
    *   **SSE:**
        *   Connection Endpoint (e.g., `GET /sse`): Client establishes a persistent HTTP connection.
        *   Message Endpoint (e.g., `POST /messages/{client_id}`): Client sends messages (like `call_tool` requests) to the server via standard HTTP POST requests. Server pushes events (like tool results) over the established SSE connection.

3.  **Tool Interfaces (Input Schemas derived from Python type hints):**
    *   **`get_latest_version_tech`**:
        *   Input: `{"technology": "string"}` (Technology name)
        *   Output: `{"content": "string"}` (Version string or message)
    *   **`fetch_relevant_doc_urls`**:
        *   Input: `{"technology": "string", "task": "string", "version": "string | null"}` (Version is optional)
        *   Output: `{"content": "string"}` (Newline-separated list of 'Title: URL' strings, or message)
    *   **`scrape_multiple_urls_to_get_context`**:
        *   Input: `{"urls": ["string", ...], "query": "string"}` (List of URLs, user query for ranking)
        *   Output: `{"content": "string"}` (Formatted string containing best URL, score, and scraped Markdown content, or error message)

## Data Flow

This diagram illustrates the sequence of interactions between the user, client, LLM, MCP server, and external services.

sequenceDiagram
    participant User
    participant Client
    participant LLM
    participant MCP Server
    participant External APIs
    
    %% Connection & Initialization
    Client->>MCP Server: initialize
    MCP Server-->>Client: initialization response
    
    %% Tool Discovery
    Client->>MCP Server: list_tools
    MCP Server-->>Client: tools and schemas
    
    %% Query Processing
    User->>Client: Query (e.g., "What's new in React?")
    Client->>LLM: Forward query + available tools
    
    %% Tool Decision
    LLM->>Client: function_call (e.g., get_latest_version_tech)
    
    %% Tool Invocation
    Client->>MCP Server: call_tool request
    
    %% Tool Execution
    MCP Server->>External APIs: API calls (Tavily, DDGS, MASA)
    External APIs-->>MCP Server: API responses
    MCP Server->>MCP Server: Process results (embeddings, ranking)
    
    %% Tool Response
    MCP Server-->>Client: Tool result
    
    %% LLM Continuation - Multiple Tool Calls
    Client->>LLM: function_response
    LLM->>Client: function_call (e.g., fetch_relevant_doc_urls)
    Client->>MCP Server: call_tool request
    MCP Server->>External APIs: API calls
    External APIs-->>MCP Server: API responses
    MCP Server-->>Client: Tool result
    Client->>LLM: function_response
    
    %% Additional Tool Call
    LLM->>Client: function_call (e.g., scrape_multiple_urls_to_get_context)
    Client->>MCP Server: call_tool request
    MCP Server->>External APIs: API calls
    External APIs-->>MCP Server: API responses
    MCP Server-->>Client: Tool result
    Client->>LLM: function_response
    
    %% Final Output
    LLM->>Client: Final text response
    Client->>User: Display answer

## Context Management

Context is managed primarily through the **on-demand fetching and processing of external data**.

1.  **Situational Context:** The protocol provides context relevant to the user's *specific query* by fetching *current* information (latest versions, relevant docs, scraped content).
2.  **Relevance Filtering:** The `scrape_multiple_urls_to_get_context` tool employs semantic search (embeddings + cosine similarity) to filter and rank scraped content, ensuring only the most relevant information from multiple potential sources is passed back to the LLM, preventing context overload.
3.  **LLM Orchestration:** The client-side LLM manages the conversational context and decides *when* and *which* tools to call to augment its internal knowledge with the real-time external context provided by this MCP implementation.
4.  **State:** The MCP server itself is largely stateless between calls, relying on the client/LLM to maintain the conversational state and drive the multi-step data gathering process if needed.

## Integration Guidelines

1.  **Choose Transport:**
    *   **STDIO:** Suitable for local development or tightly integrated setups where the client manages the server process. Run the server using `python server.py`. The client needs to use an MCP library capable of STDIO communication (like `mcp.client.stdio.stdio_client`).
    *   **SSE:** Recommended for remote access, web UIs, or integrating with tools like Cursor/Claude Desktop. Run the server using `python server_sse.py` (typically via `uvicorn`). Clients connect to the specified URL (e.g., `http://localhost:8000/sse`) using an SSE-compatible MCP client (`mcp.client.sse.sse_client` or tools like `mcp-remote`).

2.  **API Keys:** Ensure necessary API keys are configured:
    *   **Server:** Requires `MASA_DATA_API_KEY` and `TAVILY_API_KEY` (e.g., in a `.env` file).
    *   **Client:** Requires `GEMINI_API_KEY` (or the key for the chosen LLM) for the LLM interaction part.

3.  **Client Implementation:**
    *   Use an MCP client library compatible with your chosen transport.
    *   After connecting and initializing, call `list_tools` to get the available tools.
    *   **LLM Integration:**
        *   Convert the MCP tool schemas (specifically `inputSchema`) into the format required by your LLM's function calling/tool use API (e.g., Google Gemini's `Tool` and `FunctionDeclaration` objects, potentially using a helper like the `convert_mcp_tools_to_gemini` function provided in the example clients). *Note: The `clean_schema` helper might be needed to remove incompatible fields like 'title' for some LLMs.*
        *   Pass the converted tool definitions to the LLM during inference.
        *   Handle function call requests from the LLM by sending corresponding `call_tool` requests via MCP.
        *   Send tool results back to the LLM.

4.  **Example Configurations (SSE):**
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