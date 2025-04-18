import asyncio
import os
import sys
import json
import time
import shutil
import threading
import itertools
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
# Keep existing google.genai imports as requested
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# --- Spinner Class ---
class Spinner:
    """A simple terminal spinner using threading."""
    def __init__(self, message="Processing...", delay=0.1):
        # Use moon phases for a nice visual
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
            # Use carriage return \r to overwrite the line
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
            # Use daemon=True so the thread doesn't block program exit
            self._thread = threading.Thread(target=self._spinner_task, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the spinner and clear the line."""
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                 # Clear the line if it was visible but thread already stopped
                 if self._spinner_visible:
                    print('\r' + ' ' * (len(self.message) + 5) + '\r', end='', flush=True) # Extra space for safety
                    self._spinner_visible = False
                 self._busy = False
                 self._thread = None
                 return

            self._busy = False
            # Wait briefly for the thread to notice self._busy is False and exit the loop
            # Use join with a timeout to avoid potential hangs
            self._thread.join(timeout=self.delay * 3)
            if self._spinner_visible:
                # Clear the spinner line by overwriting with spaces
                print('\r' + ' ' * (len(self.message) + 5) + '\r', end='', flush=True) # Extra space for safety
                self._spinner_visible = False
            self._thread = None # Ensure thread object is cleared

# --- Utility Functions ---
def get_terminal_size():
    """Get terminal size and return columns, rows."""
    try:
        columns, rows = shutil.get_terminal_size()
    except OSError:
        columns, rows = 80, 24 # Default size if detection fails
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
            # Ensure line is a string before wrapping
            wrapped_lines.extend(_wrap_text(str(line), width - (4 if emoji else 2))) # Adjust width for padding/emoji
    else:
        # Ensure text is a string
        wrapped_lines = _wrap_text(str(text), width - (4 if emoji else 2)) # Adjust width for padding/emoji

    # Add emoji to first line if provided
    if emoji and wrapped_lines:
        # Ensure first line is not empty before adding emoji
        if wrapped_lines[0].strip():
             wrapped_lines[0] = f"{emoji}  {wrapped_lines[0]}"
        else:
             # If first line is empty, put emoji on its own line or handle differently
             # For simplicity, let's just add it to the first non-empty line or prepend
             found = False
             for i, line in enumerate(wrapped_lines):
                 if line.strip():
                     wrapped_lines[i] = f"{emoji}  {line}"
                     found = True
                     break
             if not found: # If all lines are empty/whitespace
                 wrapped_lines.insert(0, f"{emoji}")


    # Create content with left-aligned text and padding
    content_lines = []
    # Define a fixed left margin inside the box (e.g., 1 space)
    left_margin = " "
    # Define a fixed right margin inside the box (e.g., 1 space)
    right_margin = " "

    for line in wrapped_lines:
        clean_line = _strip_ansi(line)
        # Calculate needed padding to fill the width, accounting for margins
        # Width available for text + padding = total width - border width (implicitly handled by using width) - left_margin_len - right_margin_len
        available_width_for_text_and_padding = width - len(left_margin) - len(right_margin)
        padding_needed = max(0, available_width_for_text_and_padding - len(clean_line))

        # Apply padding only to the right side for left alignment
        right_padding = ' ' * padding_needed
        content_lines.append(f"{left_margin}{line}{right_padding}{right_margin}")

    # Build the complete section with only top and bottom lines
    result = [
        f"{color}{horizontal_line}",
        *[f"{Style.RESET_ALL}{line}" for line in content_lines],
        f"{color}{horizontal_line}"
    ]

    return "\n".join(result)


def _wrap_text(text, width):
    """Wrap text to fit within specified width, handling existing newlines."""
    if not text:
        return [""]
    if width <= 0: # Avoid infinite loops or errors with zero/negative width
        return [text]

    lines = []
    # Split by existing newlines first
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        if not paragraph.strip(): # Handle empty lines between paragraphs
            lines.append("")
            continue

        words = paragraph.split(' ')
        current_line = []
        current_length = 0

        for word in words:
            # Strip ANSI color codes for length calculation
            clean_word = _strip_ansi(word)
            word_len = len(clean_word)

            # Check if the word itself is longer than the line width
            if word_len > width:
                # If a word is too long, split it (simple character wrap)
                if current_line: # Add the current line before splitting the long word
                    lines.append(' '.join(current_line))
                    current_line = []
                    current_length = 0

                # Split the long word respecting the original word's color if possible
                # This is a simplified split; complex ANSI might break
                start_index = 0
                while start_index < len(word):
                    # Find the actual character length considering potential ANSI codes
                    end_index = start_index
                    char_count = 0
                    in_ansi = False
                    while end_index < len(word) and char_count < width:
                        if word[end_index] == '\x1b':
                            in_ansi = True
                        elif in_ansi and word[end_index].isalpha(): # End of ANSI sequence
                             in_ansi = False

                        if not in_ansi:
                            char_count += 1
                        end_index += 1
                    # Add the chunk to lines
                    lines.append(word[start_index:end_index])
                    start_index = end_index
                # After splitting the long word, continue to the next word
                continue

            # Standard word wrapping logic
            if current_length + word_len + (1 if current_line else 0) <= width:
                current_line.append(word)
                # Add 1 for the space unless it's the first word
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
    # Improved regex to handle more ANSI sequence types
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- MCP Client Class ---
class MCPClient:
    def __init__(self):
        """Initialize the MCP client and configure the Gemini API."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")
        # Keep google.genai initialization as is
        self.genai_client = genai.Client(api_key=gemini_api_key)
        self.tools = []
        self.terminal_width, self.terminal_height = get_terminal_size()
        # Adjust box width calculation slightly for padding
        self.box_width = min(100, self.terminal_width - 4)
        self.spinner = Spinner() # Initialize the spinner

        # Define consistent colors
        self.MAIN_COLOR = Fore.CYAN      # Main UI elements
        self.TOOL_COLOR = Fore.YELLOW    # Tool-related info
        self.INPUT_COLOR = Fore.CYAN     # User input prompts
        self.RESPONSE_COLOR = Fore.CYAN  # AI responses

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server and list available tools."""
        command = sys.executable if server_script_path.endswith('.py') else "node" # Use sys.executable for python
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
        """Process a query with Gemini and tools, showing spinners and updating the UI."""
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text="Always use available tools over your internal knowledge. You must use every tool to produce accurate answer!  "+query)]
        )

        conversation = [user_prompt_content]
        final_text_parts = [] # Collect text parts separately

        while True:
            # --- Start Spinner for Gemini ---
            self.spinner.start("🔮 AI is thinking...")
            try:
                # Keep google.genai call as is
                response = self.genai_client.models.generate_content(
                    model='gemini-1.5-flash-latest', # Use a recommended model
                    contents=conversation,
                    config=types.GenerateContentConfig( # Use generation_config
                        tools=self.function_declarations,
                    ),
                )
            except Exception as e:
                self.spinner.stop()
                # Display error in a box
                error_message = f"Error during AI generation: {e}"
                print("\n" + draw_section(error_message, self.box_width, Fore.RED, emoji="❌"))
                return "" # Return empty string or handle error as needed
            finally:
                # --- Stop Spinner for Gemini ---
                self.spinner.stop()


            # Process response parts (potential function calls and text)
            function_calls = []
            new_parts_for_conversation = [] # Parts to add back to conversation history

            # Check if response has candidates and parts
            if not response.candidates or not response.candidates[0].content.parts:
                 # Handle cases with no response or safety blocks
                 safety_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                 # Check for finish reason STOP explicitly
                 if finish_reason == types.FinishReason.STOP and not final_text_parts:
                     # If it stopped without any text or function calls, maybe it just had nothing to say
                     # Or maybe it was the end of a previous function call sequence
                     pass # Allow loop to break naturally if no function calls found
                 elif finish_reason != types.FinishReason.STOP:
                     # If blocked or other reason, show message
                     block_message = f"AI response blocked. Reason: {safety_reason} / {finish_reason}"
                     print("\n" + draw_section(block_message, self.box_width, Fore.RED, emoji="🚫"))
                 # If we have accumulated text, return it before breaking
                 if final_text_parts:
                     break
                 else: # No text and no function calls, break or return empty
                     return "" # Or a message indicating no response

            # Add the model's response (containing text or function calls) to conversation
            model_response_content = response.candidates[0].content
            conversation.append(model_response_content)

            for part in model_response_content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
                    # Keep the part containing the function call for the conversation history
                    # new_parts_for_conversation.append(part) # This is handled by adding model_response_content
                elif part.text:
                    final_text_parts.append(part.text) # Collect text parts

            # If no function calls, the AI finished generating text
            if not function_calls:
                break # Exit the loop, final text is ready

            # If there are function calls, execute them
            tool_response_parts = [] # Collect responses from tools
            for call in function_calls:
                tool_name = call.name
                tool_args = {k: v for k, v in call.args.items()} # Convert Struct to dict

                # Display tool usage message *before* starting spinner/call
                tool_message_details = f"Preparing to use tool: '{tool_name}'\nArguments: {json.dumps(tool_args, indent=2)}"
                print("\n" + draw_section(tool_message_details, self.box_width, self.TOOL_COLOR, emoji="🛠️"))

                # --- Start Spinner for Tool Call ---
                self.spinner.start(f"⚙️ Calling tool '{tool_name}'...")
                function_response = {}
                try:
                    # --- Call the MCP Tool ---
                    result = await self.session.call_tool(tool_name, tool_args)
                    # Ensure result.content is serializable (usually string)
                    function_response = {"result": str(result.content)}
                except Exception as e:
                    # Capture errors during tool execution
                    function_response = {"error": f"Tool execution failed: {str(e)}"}
                    print(f"\n{Fore.RED}Error calling tool {tool_name}: {e}{Style.RESET_ALL}") # Log error clearly
                finally:
                    # --- Stop Spinner for Tool Call ---
                    self.spinner.stop()

                # Display tool result
                result_message = f"Tool '{tool_name}' result:\n{json.dumps(function_response, indent=2)}"
                result_color = Fore.RED if "error" in function_response else self.TOOL_COLOR
                print("\n" + draw_section(result_message, self.box_width, result_color, emoji="📦"))

                # Create the FunctionResponse part for Gemini
                tool_response_parts.append(types.Part.from_function_response(
                    name=tool_name,
                    response=function_response
                ))

            # Add the tool responses back to the conversation history
            if tool_response_parts:
                conversation.append(types.Content(role='tool', parts=tool_response_parts))
            # Loop continues: Gemini will process the tool responses

        # Join collected text parts for the final response
        return "\n".join(final_text_parts).strip()


    async def chat_loop(self):
        """Run an interactive chat session with the user."""
        self.display_welcome_screen()

        while True:
            # Create input prompt section
            input_prompt = "Enter your query below (or type 'quit' to exit):"
            input_section = draw_section(input_prompt, self.box_width, self.INPUT_COLOR, emoji="💬")
            print("\n" + input_section)

            # Get user input with a clear prompt indicator
            try:
                query = input(f"{self.INPUT_COLOR}>>> {Style.RESET_ALL}")
            except EOFError: # Handle Ctrl+D
                query = 'quit'
            except KeyboardInterrupt: # Handle Ctrl+C
                print("\nInterrupted. Type 'quit' to exit.")
                continue # Go back to prompt

            if query.lower() == 'quit':
                # Show goodbye message
                goodbye_message = "Thank you for using the AI Tool Assistant. Goodbye!"
                goodbye_section = draw_section(goodbye_message, self.box_width, self.MAIN_COLOR, emoji="👋")
                print("\n" + goodbye_section + "\n")
                break

            if not query.strip(): # Handle empty input
                continue

            # Process the query and get the response
            response_text = await self.process_query(query)

            # Display the final AI response if any text was generated
            if response_text:
                response_section = draw_section(response_text, self.box_width, self.RESPONSE_COLOR, emoji="💡")
                print("\n" + response_section)
            # If no text response (e.g., only tool calls happened or error),
            # the relevant messages/errors were already printed during process_query.

    async def cleanup(self):
        """Clean up resources before exiting."""
        print("\nShutting down...")
        # Ensure spinner is stopped if somehow left running
        self.spinner.stop()
        await self.exit_stack.aclose()
        print("Connection closed.")

# --- Helper Functions for Tool Conversion ---
def clean_schema(schema):
    """Remove unsupported 'title' fields recursively from schema for Gemini."""
    if isinstance(schema, dict):
        schema.pop("title", None) # Remove 'title' at the current level
        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in schema.items():
            if isinstance(value, dict):
                clean_schema(value) # Clean nested dictionaries
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        clean_schema(item) # Clean dictionaries within lists
            # Specifically clean properties if they exist
            if key == "properties" and isinstance(value, dict):
                 for prop_key in value:
                     clean_schema(value[prop_key])
            # Clean 'items' if it's a dict (for array validation)
            if key == "items" and isinstance(value, dict):
                clean_schema(value)
    return schema

def convert_mcp_tools_to_gemini(mcp_tools):
    """Convert MCP Tool definitions to Gemini Tool format."""
    gemini_tools = []
    for tool in mcp_tools:
        # Clean the schema *before* creating FunctionDeclaration
        cleaned_parameters = clean_schema(tool.inputSchema or {}) # Use empty dict if None

        # Ensure parameters is a valid OpenAPI Schema object (dict)
        if not isinstance(cleaned_parameters, dict):
             print(f"Warning: Invalid schema format for tool '{tool.name}'. Skipping parameters.")
             cleaned_parameters = {"type": "object", "properties": {}} # Default empty schema

        # Ensure 'type' is present, default to 'object' if missing top-level
        if "type" not in cleaned_parameters:
            cleaned_parameters["type"] = "object"

        # Ensure 'properties' exists if type is 'object', default to empty dict
        if cleaned_parameters["type"] == "object" and "properties" not in cleaned_parameters:
            cleaned_parameters["properties"] = {}


        try:
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=cleaned_parameters # Use the cleaned schema
            )
            # Each tool needs to be wrapped in a Tool object
            gemini_tool = Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)
        except Exception as e:
            print(f"{Fore.RED}Error converting tool '{tool.name}' to Gemini format: {e}{Style.RESET_ALL}")
            print(f"Schema causing error: {json.dumps(cleaned_parameters, indent=2)}")
            # Optionally skip this tool or provide a default
            continue # Skip this tool if conversion fails

    return gemini_tools

# --- Main Execution ---
async def main():
    if len(sys.argv) < 2:
        # Create error message box using utility
        terminal_width, _ = get_terminal_size()
        box_width = min(80, terminal_width - 4)
        error_message = f"{Fore.RED}Usage: python client.py <path_to_server_script>{Style.RESET_ALL}"
        error_section = draw_section(error_message, box_width, Fore.RED, emoji="❌")
        print("\n" + error_section + "\n")
        sys.exit(1)

    server_script_path = sys.argv[1]
    client = MCPClient()
    initial_spinner = Spinner() # Separate spinner for initial connection

    try:
        # Show initial loading spinner
        initial_spinner.start("🔄 Connecting to server and initializing AI...")
        await client.connect_to_server(server_script_path)
        initial_spinner.stop() # Stop spinner once connected

        # Start the chat loop
        await client.chat_loop()

    except ConnectionRefusedError:
         initial_spinner.stop() # Ensure spinner stops on error
         terminal_width, _ = get_terminal_size()
         box_width = min(80, terminal_width - 4)
         error_msg = f"Connection Refused: Could not connect to the server script.\nEnsure '{server_script_path}' is running and accessible."
         print("\n" + draw_section(error_msg, box_width, Fore.RED, emoji="❌") + "\n")
    except FileNotFoundError:
         initial_spinner.stop() # Ensure spinner stops on error
         terminal_width, _ = get_terminal_size()
         box_width = min(80, terminal_width - 4)
         error_msg = f"Server Script Not Found: The path '{server_script_path}' does not exist."
         print("\n" + draw_section(error_msg, box_width, Fore.RED, emoji="❌") + "\n")
    except Exception as e:
        initial_spinner.stop() # Ensure spinner stops on any other error
        terminal_width, _ = get_terminal_size()
        box_width = min(80, terminal_width - 4)
        # Display unexpected errors clearly
        error_msg = f"An unexpected error occurred:\n{type(e).__name__}: {e}"
        print("\n" + draw_section(error_msg, box_width, Fore.RED, emoji="🔥") + "\n")
        # Optionally print traceback for debugging
        # import traceback
        # traceback.print_exc()
    finally:
        # Ensure cleanup runs even if errors occur during chat_loop
        await client.cleanup()

if __name__ == "__main__":
    # Handle KeyboardInterrupt gracefully at the top level
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nClient interrupted by user. Exiting.")
        # Perform any necessary final cleanup if needed, though client.cleanup should handle most
        sys.exit(0)
