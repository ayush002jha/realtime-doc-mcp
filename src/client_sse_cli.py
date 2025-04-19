
"""
client_sse.py

This file implements an MCP client that connects to an MCP server using SSE (Server-Sent Events) transport.
It features UI enhancements like colored output, spinners, and formatted sections, similar to client.py.
It supports multi-turn conversations and sequential tool calling based on Gemini's responses.
"""

import asyncio            # For asynchronous programming
import os                 # For accessing environment variables and terminal size
import sys                # For command-line argument handling
import json               # For JSON processing
import time               # For spinner delay
import shutil             # For terminal size detection
import threading          # For spinner thread
import itertools          # For spinner characters
from typing import Optional  # For type annotations
from contextlib import AsyncExitStack # For managing async contexts

# Import MCP components
from mcp import ClientSession
from mcp.client.sse import sse_client # Use SSE client

# Import Gemini components
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig

# Import utilities
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# --- Spinner Class (Copied from client.py) ---
class Spinner:
    """A simple terminal spinner using threading."""
    def __init__(self, message="Processing...", delay=0.1):
        self.spinner_chars = itertools.cycle(['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'])
        self.delay = delay
        self._busy = False
        self._spinner_visible = False
        self.message = message
        self._thread = None
        self._lock = threading.Lock() # To manage state changes safely

    def _spinner_task(self):
        """The task that runs in a separate thread to display the spinner."""
        while self._busy:
            char = next(self.spinner_chars)
            print(f'\r{self.message} {char}', end='', flush=True)
            self._spinner_visible = True
            time.sleep(self.delay)

    def start(self, message="Processing..."):
        """Start the spinner."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return # Avoid starting multiple threads if already running

            self.message = message
            self._busy = True
            self._thread = threading.Thread(target=self._spinner_task, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the spinner and clear the line."""
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                 if self._spinner_visible:
                    print('\r' + ' ' * (len(self.message) + 5) + '\r', end='', flush=True)
                    self._spinner_visible = False
                 self._busy = False
                 self._thread = None
                 return

            self._busy = False
            self._thread.join(timeout=self.delay * 3)
            if self._spinner_visible:
                print('\r' + ' ' * (len(self.message) + 5) + '\r', end='', flush=True)
                self._spinner_visible = False
            self._thread = None

# --- Utility Functions (Copied/Adapted from client.py) ---
def get_terminal_size():
    """Get terminal size and return columns, rows."""
    try:
        columns, rows = shutil.get_terminal_size()
    except OSError:
        columns, rows = 80, 24 # Default size
    return columns, rows

def draw_section(text, width, color=Fore.CYAN, emoji=""):
    """Draw a section with only top and bottom horizontal lines."""
    horizontal_line = "━" * width
    wrapped_lines = []
    if isinstance(text, list):
        for line in text:
            wrapped_lines.extend(_wrap_text(str(line), width - (4 if emoji else 2)))
    else:
        wrapped_lines = _wrap_text(str(text), width - (4 if emoji else 2))

    if emoji and wrapped_lines:
        if wrapped_lines[0].strip():
             wrapped_lines[0] = f"{emoji}  {wrapped_lines[0]}"
        else:
             found = False
             for i, line in enumerate(wrapped_lines):
                 if line.strip():
                     wrapped_lines[i] = f"{emoji}  {line}"
                     found = True
                     break
             if not found:
                 wrapped_lines.insert(0, f"{emoji}")

    content_lines = []
    left_margin = " "
    right_margin = " "
    for line in wrapped_lines:
        clean_line = _strip_ansi(line)
        available_width_for_text_and_padding = width - len(left_margin) - len(right_margin)
        padding_needed = max(0, available_width_for_text_and_padding - len(clean_line))
        right_padding = ' ' * padding_needed
        content_lines.append(f"{left_margin}{line}{right_padding}{right_margin}")

    result = [
        f"{color}{horizontal_line}",
        *[f"{Style.RESET_ALL}{line}" for line in content_lines],
        f"{color}{horizontal_line}"
    ]
    return "\n".join(result)

def _wrap_text(text, width):
    """Wrap text to fit within specified width, handling existing newlines."""
    if not text: return [""]
    if width <= 0: return [text]
    lines = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split(' ')
        current_line = []
        current_length = 0
        for word in words:
            clean_word = _strip_ansi(word)
            word_len = len(clean_word)
            if word_len > width:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                    current_length = 0
                start_index = 0
                while start_index < len(word):
                    end_index = start_index
                    char_count = 0
                    in_ansi = False
                    while end_index < len(word) and char_count < width:
                        if word[end_index] == '\x1b': in_ansi = True
                        elif in_ansi and word[end_index].isalpha(): in_ansi = False
                        if not in_ansi: char_count += 1
                        end_index += 1
                    lines.append(word[start_index:end_index])
                    start_index = end_index
                continue
            if current_length + word_len + (1 if current_line else 0) <= width:
                current_line.append(word)
                current_length += word_len + (1 if current_length > 0 else 0)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len
        if current_line:
            lines.append(' '.join(current_line))
    return lines if lines else [""]

def _strip_ansi(text):
    """Strip ANSI escape codes for correct length calculation."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- MCP Client Class (Updated) ---
class MCPClient:
    def __init__(self):
        """Initialize the MCP client with UI elements and Gemini."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack() # Use AsyncExitStack for cleanup
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")
        self.genai_client = genai.Client(api_key=gemini_api_key)
        self.tools = []
        self.function_declarations = []
        self.terminal_width, self.terminal_height = get_terminal_size()
        self.box_width = min(100, self.terminal_width - 4)
        self.spinner = Spinner() # Initialize the spinner

        # Define consistent colors
        self.MAIN_COLOR = Fore.MAGENTA   # Changed main color for SSE client distinction
        self.TOOL_COLOR = Fore.YELLOW
        self.INPUT_COLOR = Fore.GREEN
        self.RESPONSE_COLOR = Fore.CYAN

    async def connect_to_sse_server(self, server_url: str):
        """Connect to the MCP server via SSE and list available tools."""
        print(f"Attempting to connect to SSE server at: {server_url}")
        # Use AsyncExitStack to manage SSE client and MCP session contexts
        streams_context = sse_client(url=server_url)
        streams = await self.exit_stack.enter_async_context(streams_context)
        session_context = ClientSession(*streams)
        self.session = await self.exit_stack.enter_async_context(session_context)

        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = response.tools
        self.function_declarations = convert_mcp_tools_to_gemini(self.tools)
        print(f"{Fore.GREEN}Successfully connected to SSE server.{Style.RESET_ALL}")


    def display_welcome_screen(self):
        """Display a welcome screen with available tools."""
        clear_screen()
        title = f"{self.MAIN_COLOR}AI-Powered Tool Assistant (SSE Client){Style.RESET_ALL}"
        title_section = draw_section(title, self.box_width, self.MAIN_COLOR, emoji="📡") # SSE Emoji

        tool_descriptions = []
        if self.tools:
            for tool in self.tools:
                tool_descriptions.append(f"{self.TOOL_COLOR}{tool.name}{Style.RESET_ALL}: {tool.description}")
        else:
            tool_descriptions.append("No tools available from the server.")
        tools_section = draw_section(tool_descriptions, self.box_width, self.MAIN_COLOR, emoji="🧰")

        instructions = [
            f"• Type your query below and press Enter.",
            f"• The AI will use tools ({self.TOOL_COLOR}Yellow{Style.RESET_ALL}) if needed.",
            f"• Type {Style.BRIGHT}'quit'{Style.NORMAL} to exit."
        ]
        instructions_section = draw_section(instructions, self.box_width, self.MAIN_COLOR, emoji="ℹ️")

        print("\n" + title_section)
        print("\n" + tools_section)
        print("\n" + instructions_section)

    async def process_query(self, query: str) -> str:
        """Process a query with Gemini and tools, handling multiple tool calls."""
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text="Always use available tools over your internal knowledge. You must first fetch latest stable version of the tech, followed by fetching relevant urls related to query and finally scrape these urls to get context to produce accurate answer!  "+query)]
            # parts=[types.Part.from_text(text=query)] # Simplified prompt for testing
        )

        conversation = [user_prompt_content]
        final_text_parts = [] # Collect text parts separately

        # Use a loop similar to client.py to handle potential sequences of tool calls
        while True:
            self.spinner.start("🔮 AI is thinking...")
            try:
                response = self.genai_client.models.generate_content(
                    model='gemini-2.0-flash-001', # Using a capable model
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        tools=self.function_declarations,
                    ),
                )
            except Exception as e:
                self.spinner.stop()
                error_message = f"Error during AI generation: {e}"
                print("\n" + draw_section(error_message, self.box_width, Fore.RED, emoji="❌"))
                return "" # Return empty on error
            finally:
                self.spinner.stop()

            # Process response parts
            function_calls = []
            # Add the model's response (potentially containing function calls) to conversation history
            # Check for valid response structure first
            if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                 safety_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                 if finish_reason == types.FinishReason.STOP and not final_text_parts:
                     pass # Normal stop without text/calls
                 elif finish_reason != types.FinishReason.STOP:
                     block_message = f"AI response blocked. Reason: {safety_reason} / {finish_reason}"
                     print("\n" + draw_section(block_message, self.box_width, Fore.RED, emoji="🚫"))
                 if final_text_parts: # Return accumulated text if any
                     break
                 else: # No text, no function calls, nothing more to do
                     return ""

            model_response_content = response.candidates[0].content
            conversation.append(model_response_content) # Add model response to history

            for part in model_response_content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
                    # The part containing the function call is already added via model_response_content
                elif part.text:
                    final_text_parts.append(part.text) # Collect text

            # If no function calls in this turn, break the loop
            if not function_calls:
                break

            # Execute function calls
            tool_response_parts = []
            for call in function_calls:
                tool_name = call.name
                # Convert Struct to dict for MCP call_tool
                tool_args = {k: v for k, v in call.args.items()}

                tool_message_details = f"Preparing to use tool: '{tool_name}'\nArguments: {json.dumps(tool_args, indent=2)}"
                print("\n" + draw_section(tool_message_details, self.box_width, self.TOOL_COLOR, emoji="🛠️"))

                self.spinner.start(f"⚙️ Calling tool '{tool_name}'...")
                function_response = {}
                try:
                    # Call the MCP Tool via SSE session
                    result = await self.session.call_tool(tool_name, tool_args)
                    # Ensure result.content is serializable (usually string)
                    function_response = {"result": str(result.content)}
                except Exception as e:
                    function_response = {"error": f"Tool execution failed: {str(e)}"}
                    print(f"\n{Fore.RED}Error calling tool {tool_name}: {e}{Style.RESET_ALL}")
                finally:
                    self.spinner.stop()

                result_message = f"Tool '{tool_name}' result:\n{json.dumps(function_response, indent=2)}"
                result_color = Fore.RED if "error" in function_response else self.TOOL_COLOR
                print("\n" + draw_section(result_message, self.box_width, result_color, emoji="📦"))

                # Create the FunctionResponse part for Gemini
                tool_response_parts.append(types.Part.from_function_response(
                    name=tool_name,
                    response=function_response
                ))

            # Add the tool responses back to the conversation history for the next Gemini call
            if tool_response_parts:
                conversation.append(types.Content(role='tool', parts=tool_response_parts))
            # Loop continues: Gemini processes tool responses

        # Join collected text parts for the final response
        return "\n".join(final_text_parts).strip()


    async def chat_loop(self):
        """Run an interactive chat session with the user using the enhanced UI."""
        self.display_welcome_screen()

        while True:
            input_prompt = "Enter your query below (or type 'quit' to exit):"
            input_section = draw_section(input_prompt, self.box_width, self.INPUT_COLOR, emoji="💬")
            print("\n" + input_section)

            try:
                query = input(f"{self.INPUT_COLOR}>>> {Style.RESET_ALL}")
            except EOFError:
                query = 'quit'
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
                continue

            if query.lower() == 'quit':
                goodbye_message = "Thank you for using the AI Tool Assistant (SSE). Goodbye!"
                goodbye_section = draw_section(goodbye_message, self.box_width, self.MAIN_COLOR, emoji="👋")
                print("\n" + goodbye_section + "\n")
                break

            if not query.strip():
                continue

            response_text = await self.process_query(query)

            if response_text:
                response_section = draw_section(response_text, self.box_width, self.RESPONSE_COLOR, emoji="💡")
                print("\n" + response_section)
            # Tool call/result messages are printed within process_query

    async def cleanup(self):
        """Clean up resources using AsyncExitStack."""
        print("\nShutting down SSE client...")
        self.spinner.stop() # Ensure spinner is stopped
        await self.exit_stack.aclose() # Closes session and SSE connection
        print("Connection closed.")

# --- Helper Functions for Tool Conversion (Copied from client.py) ---
def clean_schema(schema):
    """Remove unsupported 'title' fields recursively from schema for Gemini."""
    if isinstance(schema, dict):
        schema.pop("title", None)
        for key, value in schema.items():
            if isinstance(value, dict):
                clean_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        clean_schema(item)
            if key == "properties" and isinstance(value, dict):
                 for prop_key in value:
                     clean_schema(value[prop_key])
            if key == "items" and isinstance(value, dict):
                clean_schema(value)
    return schema

def convert_mcp_tools_to_gemini(mcp_tools):
    """Convert MCP Tool definitions to Gemini Tool format."""
    gemini_tools = []
    for tool in mcp_tools:
        cleaned_parameters = clean_schema(tool.inputSchema or {})
        if not isinstance(cleaned_parameters, dict):
             print(f"Warning: Invalid schema format for tool '{tool.name}'. Skipping parameters.")
             cleaned_parameters = {"type": "object", "properties": {}}
        if "type" not in cleaned_parameters:
            cleaned_parameters["type"] = "object"
        if cleaned_parameters["type"] == "object" and "properties" not in cleaned_parameters:
            cleaned_parameters["properties"] = {}

        try:
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=cleaned_parameters
            )
            gemini_tool = Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)
        except Exception as e:
            print(f"{Fore.RED}Error converting tool '{tool.name}' to Gemini format: {e}{Style.RESET_ALL}")
            print(f"Schema causing error: {json.dumps(cleaned_parameters, indent=2)}")
            continue
    return gemini_tools

# --- Main Execution (Updated) ---
async def main():
    if len(sys.argv) < 2:
        terminal_width, _ = get_terminal_size()
        box_width = min(80, terminal_width - 4)
        error_message = f"{Fore.RED}Usage: python client_sse.py <sse_server_url>{Style.RESET_ALL}"
        error_section = draw_section(error_message, box_width, Fore.RED, emoji="❌")
        print("\n" + error_section + "\n")
        sys.exit(1)

    server_url = sys.argv[1]
    client = MCPClient()
    initial_spinner = Spinner() # Separate spinner for initial connection

    try:
        initial_spinner.start("🔄 Connecting to SSE server and initializing AI...")
        await client.connect_to_sse_server(server_url)
        initial_spinner.stop()

        await client.chat_loop()

    except ConnectionRefusedError: # Might occur if server isn't running
         initial_spinner.stop()
         terminal_width, _ = get_terminal_size()
         box_width = min(80, terminal_width - 4)
         error_msg = f"Connection Refused: Could not connect to the SSE server.\nEnsure the server at '{server_url}' is running and accessible."
         print("\n" + draw_section(error_msg, box_width, Fore.RED, emoji="❌") + "\n")
    except Exception as e: # Catch other potential errors during connection or chat
        initial_spinner.stop()
        terminal_width, _ = get_terminal_size()
        box_width = min(80, terminal_width - 4)
        # Display unexpected errors clearly
        error_msg = f"An unexpected error occurred:\n{type(e).__name__}: {e}"
        print("\n" + draw_section(error_msg, box_width, Fore.RED, emoji="🔥") + "\n")
        # import traceback # Uncomment for debugging
        # traceback.print_exc() # Uncomment for debugging
    finally:
        # Ensure cleanup runs even if errors occur during chat_loop
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nClient interrupted by user. Exiting.")
        sys.exit(0)
