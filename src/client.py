import asyncio
import os
import sys
import json
import time
import shutil
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

def get_terminal_size():
    """Get terminal size and return columns, rows."""
    columns, rows = shutil.get_terminal_size()
    return columns, rows

def draw_section(text, width, color=Fore.CYAN, emoji=""):
    """
    Draw a section with only top and bottom horizontal lines.
    """
    horizontal_line = "━" * width
    
    # Process and wrap text to fit within the width
    wrapped_lines = []
    if isinstance(text, list):
        for line in text:
            wrapped_lines.extend(_wrap_text(line, width))
    else:
        wrapped_lines = _wrap_text(text, width)
    
    # Add emoji to first line if provided
    if emoji and wrapped_lines:
        wrapped_lines[0] = f"{emoji}  {wrapped_lines[0]}"
    
    # Create content with centered text
    content_lines = []
    for line in wrapped_lines:
        padding_needed = width - len(_strip_ansi(line))
        left_padding = padding_needed // 2
        content_lines.append(f"{' ' * left_padding}{line}")
    
    # Build the complete section with only top and bottom lines
    result = [
        f"{color}{horizontal_line}",
        *[f"{Style.RESET_ALL}{line}" for line in content_lines],
        f"{color}{horizontal_line}"
    ]
    
    return "\n".join(result)

def _wrap_text(text, width):
    """Wrap text to fit within specified width."""
    if not text:
        return [""]
    
    words = text.split(' ')
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        # Strip ANSI color codes for length calculation
        clean_word = _strip_ansi(word)
        if current_length + len(clean_word) + (1 if current_line else 0) <= width:
            current_line.append(word)
            current_length += len(clean_word) + (1 if current_length > 0 else 0)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(clean_word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else [""]

def _strip_ansi(text):
    """Strip ANSI color codes for correct length calculation."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

class MCPClient:
    def __init__(self):
        """Initialize the MCP client and configure the Gemini API."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")
        self.genai_client = genai.Client(api_key=gemini_api_key)
        self.tools = []
        self.terminal_width, self.terminal_height = get_terminal_size()
        self.box_width = min(100, self.terminal_width - 4)  # Leave some margin
        
        # Define consistent colors
        self.MAIN_COLOR = Fore.CYAN      # Main UI elements
        self.TOOL_COLOR = Fore.YELLOW    # Tool-related info
        self.INPUT_COLOR = Fore.CYAN     # User input prompts
        self.RESPONSE_COLOR = Fore.CYAN  # AI responses

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server and list available tools."""
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = response.tools
        self.function_declarations = convert_mcp_tools_to_gemini(self.tools)

    def display_welcome_screen(self):
        """Display a welcome screen with available tools."""
        clear_screen()
        title = f"{self.MAIN_COLOR}AI-Powered Tool Assistant{Style.RESET_ALL}"
        title_section = draw_section(title, self.box_width, self.MAIN_COLOR, emoji="🤖")
        
        # Create a list of formatted tool descriptions
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"{self.TOOL_COLOR}🛠️  {tool.name} {Style.RESET_ALL}- {tool.description}")
        
        tools_section = draw_section(tool_descriptions, self.box_width, self.MAIN_COLOR, emoji="🧰")
        
        instructions = [
            f"{self.MAIN_COLOR}• Type your query and press Enter to interact with the AI",
            f"{self.MAIN_COLOR}• The AI will use appropriate tools to help answer your questions",
            f"{self.MAIN_COLOR}• Type {Style.RESET_ALL}'quit'{self.MAIN_COLOR} to exit the application"
        ]
        
        instructions_section = draw_section(instructions, self.box_width, self.MAIN_COLOR, emoji="ℹ️")
        
        print("\n" + title_section)
        print("\n" + tools_section)
        print("\n" + instructions_section)

    async def process_query(self, query: str) -> str:
        """Process a query with Gemini and tools, updating the UI during execution."""
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )

        conversation = [user_prompt_content]
        final_text = []

        # Show thinking message
        thinking_message = f"AI is thinking..."
        print("\n" + draw_section(thinking_message, self.box_width, self.MAIN_COLOR, emoji="🔮"))

        while True:
            response = self.genai_client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=conversation,
                config=types.GenerateContentConfig(
                    tools=self.function_declarations,
                ),
            )

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
                
                # Show tool execution message
                tool_message = f"Using tool: '{tool_name}' with arguments: {json.dumps(tool_args, indent=2)}"
                print("\n" + draw_section(tool_message, self.box_width, self.TOOL_COLOR, emoji="🛠️"))
                
                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    function_response = {"result": str(result.content)}
                except Exception as e:
                    function_response = {"error": str(e)}

                # Show tool result
                result_message = f"Tool result: {json.dumps(function_response, indent=2)}"
                print("\n" + draw_section(result_message, self.box_width, self.TOOL_COLOR, emoji="📦"))

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

        return "\n".join(final_text).strip()

    async def chat_loop(self):
        """Run an interactive chat session with the user."""
        self.display_welcome_screen()
        
        while True:
            # Create input prompt
            input_prompt = "What would you like to ask?"
            input_section = draw_section(input_prompt, self.box_width, self.INPUT_COLOR, emoji="💬")
            print("\n" + input_section)
            
            # Get user input
            query = input(f"{self.INPUT_COLOR}>>> {Style.RESET_ALL}")
            if query.lower() == 'quit':
                # Show goodbye message
                goodbye_message = "Thank you for using the AI Tool Assistant. Goodbye!"
                goodbye_section = draw_section(goodbye_message, self.box_width, self.MAIN_COLOR, emoji="👋")
                print("\n" + goodbye_section + "\n")
                break

            # Process the query
            response = await self.process_query(query)
            
            # Display the response
            response_section = draw_section(response, self.box_width, self.RESPONSE_COLOR, emoji="💡")
            print("\n" + response_section)

    async def cleanup(self):
        """Clean up resources before exiting."""
        await self.exit_stack.aclose()

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

async def main():
    if len(sys.argv) < 2:
        # Create error message box
        terminal_width, _ = get_terminal_size()
        box_width = min(80, terminal_width - 4)
        error_message = f"{Fore.RED}Usage: python client.py <path_to_server_script>{Style.RESET_ALL}"
        error_section = draw_section(error_message, box_width, Fore.RED, emoji="❌")
        print("\n" + error_section + "\n")
        sys.exit(1)

    client = MCPClient()
    try:
        # Show loading message
        terminal_width, _ = get_terminal_size()
        box_width = min(80, terminal_width - 4)
        loading_message = "Connecting to server and initializing AI..."
        loading_section = draw_section(loading_message, box_width, Fore.CYAN, emoji="🔄")
        print("\n" + loading_section + "\n")
        
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())