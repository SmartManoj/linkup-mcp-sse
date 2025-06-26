import asyncio
import json
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from litellm import completion
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model = "gemini/gemini-2.5-flash"

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        try:
            # Initial Gemini API call
            response = completion(
                model=self.model,
                messages=messages,
                tools=available_tools,
                max_tokens=1000
            )

            # Process response and handle tool calls
            tool_results = []
            final_text = []

            for choice in response.choices:
                message = choice.message
                if message.content:
                    final_text.append(message.content)
                
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        # Execute tool call
                        result = await self.session.call_tool(tool_name, tool_args)
                        tool_results.append({"call": tool_name, "result": result})
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                        # Extract text content from MCP response
                        tool_response_text = ""
                        if hasattr(result, 'content') and result.content:
                            if hasattr(result.content[0], 'text'):
                                tool_response_text = result.content[0].text
                            else:
                                tool_response_text = str(result.content[0])
                        else:
                            tool_response_text = str(result.content)

                        # Always append the tool response text to the output
                        final_text.append(tool_response_text)

                        # Continue conversation with tool results (use plain text)
                        messages.append({
                            "role": "assistant",
                            "content": message.content or ""
                        })
                        messages.append({
                            "role": "user", 
                            "content": tool_response_text
                        })

                        # Get next response from Gemini
                        response = completion(
                            model=self.model,
                            messages=messages,
                            max_tokens=1000
                        )

                        if response.choices and response.choices[0].message.content:
                            final_text.append(response.choices[0].message.content)

            return "\n###\n".join(final_text)
        except Exception as e:
            return f"Error processing query: {str(e)}"
    

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    if len(sys.argv[1:]) > 0:
        server_url = sys.argv[1]
    else:
        server_url = "http://localhost:8080/sse"


    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=server_url)
        await client.chat_loop()
    finally:
        await client.cleanup()


async def test_single_query(server_url: str, query: str):
    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=server_url)
        response = await client.process_query(query)
        print("\nTest Query Response:\n" + response)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Usage: python client.py --test [server_url] [query]
        server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080/sse"
        query = sys.argv[3] if len(sys.argv) > 3 else "who is SmartManoj?"
        asyncio.run(test_single_query(server_url, query))
        sys.exit(0)
    else:
        asyncio.run(main())