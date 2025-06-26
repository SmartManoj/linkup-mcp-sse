import asyncio
import logging

import mcp.server.stdio
import mcp.types as types
from linkup import LinkupClient
from mcp.server import Server
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.responses import Response
from dotenv import load_dotenv

load_dotenv()

import uvicorn
server = Server("mcp-search-linkup")
logger = logging.getLogger("mcp-search-linkup")
logger.setLevel(logging.INFO)


@server.set_logging_level()  # type: ignore
async def set_logging_level(level: types.LoggingLevel) -> types.EmptyResult:
    logger.setLevel(level.upper())
    await server.request_context.session.send_log_message(
        level="info",
        data=f"Log level set to {level}",
        logger="mcp-search-linkup",
    )
    return types.EmptyResult()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available search tools."""
    return [
        types.Tool(
            name="search-web",
            description="Search the web in real time using Linkup. Use this tool whenever the user needs trusted facts, news, or source-backed information. Returns comprehensive content from the most relevant sources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query. Full questions work best, e.g., 'How does the new EU AI Act affect startups?'",
                    },
                    "depth": {
                        "type": "string",
                        "description": "The search depth to perform. Use 'standard' for "
                        "queries with likely direct answers. Use 'deep' for complex queries "
                        "requiring comprehensive analysis or multi-hop questions",
                        "enum": ["standard", "deep"],
                    },
                },
                "required": ["query", "depth"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict | None,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle search tool execution requests."""
    if name != "search-web":
        raise ValueError(f"Unknown tool: {name}")
    if not arguments:
        raise ValueError("Missing arguments")

    query = arguments.get("query")
    if not query:
        raise ValueError("Missing query")
    depth = arguments.get("depth")
    if not depth:
        raise ValueError("Missing depth")

    client = LinkupClient()
    search_response = client.search(
        query=query,
        depth=depth,
        output_type="searchResults",
    )

    return [
        types.TextContent(
            type="text",
            text=str(search_response),
        )
    ]

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )
        return Response()

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = server

    import argparse

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
