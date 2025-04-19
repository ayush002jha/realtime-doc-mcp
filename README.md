# RealtimeDocContext MCP Implementation for Masa Subnet 42 Challenge

## Overview

This repository contains the **RealtimeDocContext MCP implementation**, submitted for the **Masa Subnet 42 MCP Challenge**.

The goal of this project is to enhance Large Language Models (LLMs) by providing them with the ability to access and utilize fresh, relevant context from online documentation and web sources in real-time. It addresses the limitation of static training data by allowing models to dynamically:

1.  Fetch the latest stable version of a specified technology.
2.  Discover relevant documentation URLs based on the technology and a user's task.
3.  Scrape content from candidate URLs, semantically rank them against the user's query, and provide the most pertinent information back to the LLM.

This implementation leverages the Model Context Protocol (MCP) standard, offering both STDIO and Server-Sent Events (SSE) transports for flexibility. It includes server implementations, multiple client examples (CLI, Web UI), and demonstrates integration with LLMs like Google Gemini for tool orchestration. The use of the MASA Data API for scraping directly aligns with the goals of leveraging real-time data sources potentially relevant to the Masa Subnet 42 ecosystem.

## Features

*   **Real-time Context:** Provides LLMs with up-to-date information from the web.
*   **Targeted Information Retrieval:**
    *   Fetches latest stable software versions via Tavily Search (`get_latest_version_tech`).
    *   Finds relevant documentation URLs using DuckDuckGo Search (`fetch_relevant_doc_urls`).
*   **Intelligent Content Scraping & Ranking:**
    *   Scrapes content from multiple URLs using the MASA Data API (with a lightweight fallback).
    *   Uses Sentence Transformers for semantic embedding and ranking to return the most relevant scraped content based on the user's query (`scrape_multiple_urls_to_get_context`).
*   **Flexible Transports:**
    *   **STDIO Server (`server.py`):** For local execution and simple integration.
    *   **SSE Server (`server_sse.py`):** For network accessibility, web clients, and integration with tools like Cursor/Claude Desktop.
*   **Multiple Client Examples:**
    *   `client_cli.py`: CLI client for the STDIO server.
    *   `client_sse_cli.py`: CLI client for the SSE server.
    *   `app.py`: Gradio-based Web UI client for the SSE server.
*   **LLM Agnostic (in principle):** Demonstrated with Google Gemini, but adaptable to other LLMs supporting function calling/tool use via MCP.

## Repository Structure
```
.
├── assets
│ └── images
│ └── sequence_diagram.png # Image used in SPECIFICATION.md (optional)
├── docs
│ ├── IMPLEMENTATION.md # Detailed implementation guide
│ └── SPECIFICATION.md # Protocol specification & design
├── src
│ ├── app.py # Gradio Web UI Client (SSE)
│ ├── client_cli.py # CLI Client (STDIO)
│ ├── client_sse_cli.py # CLI Client (SSE)
│ ├── server.py # MCP Server (STDIO)
│ └── server_sse.py # MCP Server (SSE)
├── .gitignore
├── LICENSE # Project License (MIT)
├── pyproject.toml # Project dependencies and metadata (for uv/pip)
└── README.md # This file
```


## Architecture

The project follows a standard MCP client-server architecture:

*   **Server:** Implemented in Python using `mcp-python` (`FastMCP`). Handles tool execution, interacts with external APIs (MASA, Tavily, DDGS), and performs local processing (embeddings). Offers STDIO and SSE transports.
*   **Client:** Connects to the server via the chosen transport. Integrates with an LLM (Gemini) to interpret user queries, decide which tools to call, invoke them via MCP, and formulate responses based on the results.

For a detailed breakdown, see the [Protocol Specification](docs/SPECIFICATION.md).

## Getting Started

### Prerequisites

1.  **Python:** Version 3.9+ recommended.
2.  **`uv`:** This project uses `uv` for environment and package management. Install it if you haven't already: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)
    ```bash
    # Example using curl:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3.  **API Keys:** You will need API keys for:
    *   MASA Data API
    *   Tavily Search API
    *   Google Gemini API (or your chosen LLM)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ayush002jha/realtime-doc-mcp.git
    cd src
    ```
2.  **Create & Activate Virtual Environment (using `uv`):**
    ```bash
    uv venv
    source .venv/bin/activate # Linux/macOS
    # .\.venv\Scripts\activate.bat # Windows CMD
    # .\.venv\Scripts\Activate.ps1 # Windows PowerShell
    ```
3.  **Install Dependencies (using `uv`):**
    ```bash
    uv sync
    ```
    *(This reads `pyproject.toml` and installs necessary packages)*
4.  **Configure API Keys:**
    Create a `.env` file in the project root and add your keys:
    ```dotenv
    MASA_DATA_API_KEY="your_masa_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    GEMINI_API_KEY="your_gemini_api_key"

    # Optional: Set default SSE URL for Gradio app if hosted elsewhere
    # MCP_SERVER_URL="http://your_sse_server_host:8000/sse"
    ```

## Usage

*(Ensure your virtual environment is activated for all commands)*

### 1. Run the Server

Choose ONE server type to run:

*   **SSE Server (Recommended for Web UI / Remote Access):**
    ```bash
    uvicorn src.server_sse:app --host 0.0.0.0 --port 8000 --reload
    ```
    *(Note the URL it's running on, e.g., `http://localhost:8000`). `--reload` is optional for development.*

*   **STDIO Server (For local CLI use):**
    The `client_cli.py` script starts this server automatically. Do not run it separately if using `client_cli.py`.

### 2. Run a Client

*   **Gradio Web UI (Connects to SSE Server):**
    ```bash
    python src/app.py
    ```
    Open the provided URL (e.g., `http://127.0.0.1:7860`). Enter the SSE Server URL (e.g., `http://localhost:8000/sse`) and click Connect.

*   **CLI Client for SSE Server:**
    ```bash
    python src/client_sse_cli.py <your_sse_server_url>/sse
    # Example: python src/client_sse_cli.py http://localhost:8000/sse
    ```

*   **CLI Client for STDIO Server:**
    ```bash
    python src/client_cli.py src/server.py
    ```
    *(Note: provide the path to the server script within the src directory)*

### 3. Integration with Other Tools (Cursor, Claude Desktop)

You can connect the **SSE Server** to tools supporting MCP:

*   **Cursor:** Add to `settings.json` (`mcpServers`):
    ```json
    {
      "mcpServers": {
        "RealtimeDocContext": {
          "url": "YOUR_SSE_SERVER_URL/sse"
        }
      }
    }
    ```

*   **Claude Desktop:** Add to `settings.json` (`mcpServers`), using `mcp-remote` (`npm i -g mcp-remote`):
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
    *(Replace `YOUR_SSE_SERVER_URL` with the actual running URL of the SSE server)*

## Documentation

For more detailed information, please see:

*   **[Protocol Specification](docs/SPECIFICATION.md):** Describes the MCP design, components, data flow, and interfaces.
*   **[Implementation Guide](docs/IMPLEMENTATION.md):** Details the architecture, components, setup, usage, performance, and testing.

## Hackathon Context

This project was developed for the **MCP (Model Context Protocol) Challenge - Masa Subnet 42**. It aims to fulfill the challenge's objective by building an innovative MCP plugin that enhances AI agent capabilities through real-time data access and sophisticated context management, demonstrating potential integration points with the Bittensor ecosystem and Masa's data infrastructure.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.