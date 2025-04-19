#!/usr/bin/env python
"""
gradio_client.py

RealtimeDocContext MCP Client with Gradio UI
"""

import asyncio
import os
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from mcp import ClientSession
from mcp.client.sse import sse_client

from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration

import gradio as gr
from dotenv import load_dotenv
load_dotenv()
# Reuse your core logic classes
class AsyncProcessor:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.client = MCPClient()
        self.running = True
        self.active_tools: List[Dict] = []
        self.executor.submit(self._run_loop)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        while self.running:
            self.loop.run_forever()

    async def _connect(self, server_url: str):
        await self.client.connect_to_sse_server(server_url)
        return True

    async def _process_query(self, query: str):
        self.active_tools = []
        result = await self.client.process_query(query, self._update_tool_status)
        return result

    async def _cleanup(self):
        await self.client.cleanup()

    def _update_tool_status(self, tool_name: str, status: str):
        existing = next((t for t in self.active_tools if t["name"] == tool_name), None)
        if existing:
            existing["status"] = status
        else:
            self.active_tools.append({"name": tool_name, "status": status})

    def connect(self, server_url: str):
        future = asyncio.run_coroutine_threadsafe(self._connect(server_url), self.loop)
        return future.result()

    def process_query(self, query: str):
        future = asyncio.run_coroutine_threadsafe(self._process_query(query), self.loop)
        return future.result()

    def cleanup(self):
        future = asyncio.run_coroutine_threadsafe(self._cleanup(), self.loop)
        return future.result()

    def shutdown(self):
        self.running = False
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.executor.shutdown(wait=True)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        self.genai_client = None
        self.function_declarations = []
        self.tools = []

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.genai_client = genai.Client(api_key=gemini_api_key)

    async def connect_to_sse_server(self, server_url: str):
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()
        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()
        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = response.tools
        self.function_declarations = convert_mcp_tools_to_gemini(self.tools)

    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str, update_tool_status) -> str:
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text="Always use available tools over your internal knowledge. You must first fetch latest stable version of the tech, followed by fetching relevant urls related to query and finally scrape these urls to get context to produce accurate answer!  "+query)]
        )

        conversation = [user_prompt_content]
        final_text = []
        tool_calls = []

        while True:
            try:
                response = self.genai_client.models.generate_content(
                    model='gemini-2.0-flash-001',
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        tools=self.function_declarations,
                    ),
                )
            except Exception as e:
                return f"AI Model Error: {str(e)}"

            function_calls = []
            new_parts = []

            for candidate in response.candidates:
                if not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
                        new_parts.append(part)
                    elif part.text:
                        final_text.append(part.text)

            if not function_calls:
                break

            for call_part, call in zip(new_parts, function_calls):
                tool_name = call.name
                tool_args = call.args
                tool_calls.append({"name": tool_name, "status": "started"})
                update_tool_status(tool_name, "started")

                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    function_response = {"result": result.content}
                    update_tool_status(tool_name, "completed")
                except Exception as e:
                    function_response = {"error": f"Tool Error: {str(e)}"}
                    update_tool_status(tool_name, "failed")

                function_response_part = types.Part.from_function_response(
                    name=tool_name,
                    response=function_response
                )

                tool_response_content = types.Content(
                    role='tool',
                    parts=[function_response_part]
                )

                conversation.append(call_part)
                conversation.append(tool_response_content)

        response_lines = []
        for tool in tool_calls:
            status = "🔄 Processing..." if tool["status"] == "started" else \
                     "✅ Completed" if tool["status"] == "completed" else \
                     "❌ Failed"
            response_lines.append(f"- **{tool['name']}**: {status}")
        
        if final_text:
            response_lines.append("\n" + "\n".join(final_text).strip())
        
        return "\n".join(response_lines)

def clean_schema(schema):
    if isinstance(schema, dict):
        schema.pop("title", None)
        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                schema["properties"][key] = clean_schema(schema["properties"][key])
    return schema

def convert_mcp_tools_to_gemini(mcp_tools):
    gemini_tools = []
    for tool in mcp_tools:
        parameters = clean_schema(tool.inputSchema)
        function_declaration = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=parameters
        )
        gemini_tool = Tool(function_declarations=[function_declaration])
        gemini_tools.append(gemini_tool)
    return gemini_tools


# Update CSS for proper rendering
css = """
:root {
    --bg: #1e1e1e;
    --surface: #252526;
    --text: #d4d4d4;
    --primary: #007acc;
    --border: #3c3c3c;
}

.gradio-container {
    background: var(--bg) !important;
    font-family: 'Segoe UI', sans-serif;
}

.chatbot {
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    min-height: 400px;
}

.tool-call {
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    background: rgba(0,122,204,0.1);
    border-left: 3px solid var(--primary);
    border-radius: 4px;
}

.final-response {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
}

.loading-spinner {
    display: inline-block;
    width: 1rem;
    height: 1rem;
    border: 2px solid var(--primary);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
.tool-list {
    background: var(--surface);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}
"""

def create_app():
    with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="blue")) as app:
        # State management
        processor = gr.State()
        active_tools = gr.State([])
        available_tools = gr.State([])

        
        # Header
        gr.Markdown("# RealtimeDocContext MCP Client")
        
        # Connection UI
        with gr.Row():
            server_url = gr.Textbox(
                label="MCP Server URL",
                value=os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse"),
                interactive=True
            )
            connect_btn = gr.Button("🔌 Connect", variant="primary")
            status = gr.Markdown("Status: **Disconnected**")

        # Available tools display
        tools_panel = gr.Markdown("### Available Tools\n*Not connected*", elem_classes=["tool-list"])

        # Chat interface
        chatbot = gr.Chatbot(label="Document Context Session")
        msg = gr.Textbox(label="Input", placeholder="Enter your technical query...")
        
        # Event handlers
        def connect(server):
            try:
                proc = AsyncProcessor()
                proc.connect(server)
                tools = proc.client.tools
                tool_list = "\n".join([f"- `{tool.name}`" for tool in tools])
                return {
                    processor: proc,
                    available_tools: tools,
                    tools_panel: f"### Available Tools\n{tool_list}",
                    status: "Status: **Connected** ✅",
                    connect_btn: gr.update(interactive=False)
                }
            except Exception as e:
                return {status: f"Connection failed: {str(e)}"}
        
        def process_message(message, chat_history, processor):
            # Immediately show user message
            chat_history.append((message, None))
            yield chat_history, ""
            
            # Show processing states
            for state in [
                "⏳ Initializing response pipeline...",
                "🔍 Analyzing query structure...",
                "🛠️ Utilizing required tools..."
            ]:
                chat_history[-1] = (message, state)
                yield chat_history, ""
                time.sleep(0.5)
            
            try:
                # Process query
                response = processor.process_query(message)
                
                # Parse tool calls
                tool_updates = []
                for tool in processor.active_tools:
                    tool_updates.append(
                        f"<div class='tool-call'>🛠️ {tool['name']}: "
                        f"{'🔄 Processing...<div class=\"status-spinner\"></div>' if tool['status'] == 'started' else '✅ Completed'}"
                        f"</div>"
                    )
                
                # Show tool calls
                chat_history[-1] = (
                    message, 
                    "\n".join(tool_updates) + "\n\n**Final Response:**\n" + response
                )
                yield chat_history, ""
                
            except Exception as e:
                chat_history[-1] = (message, f"🚨 Error: {str(e)}")
                yield chat_history, ""

        # Wire up components
        connect_btn.click(
            connect,
            inputs=server_url,
            outputs=[processor,  available_tools, tools_panel, status, connect_btn]
        )
        
        msg.submit(
            process_message,
            [msg, chatbot, processor],
            [chatbot, msg],
            concurrency_limit=20
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True
    )