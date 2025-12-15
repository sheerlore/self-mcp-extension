import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field, create_model

BASE_DIR = Path(__file__).parent
GENERATED_TOOLS_DIR = BASE_DIR / "generated_tools"


def json_schema_to_pydantic_field(
    name: str,
    schema: dict[str, Any]
) -> tuple[type, Any]:
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    json_type = schema.get("type", "string")
    python_type = type_mapping.get(json_type, str)
    description = schema.get("description", "")
    default = schema.get("default", ...)

    return (python_type, Field(default=default, description=description))


def create_args_model(
    tool_name: str,
    input_schema: dict[str, Any]
) -> type[BaseModel]:
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    fields = {}
    for prop_name, prop_schema in properties.items():
        python_type, field = json_schema_to_pydantic_field(
            prop_name, prop_schema
        )
        # Set default to None if not required
        if prop_name not in required and field.default is ...:
            field = Field(default=None, description=field.description)
        fields[prop_name] = (python_type, field)

    model_name = f"{tool_name.title().replace('_', '')}Args"
    return create_model(model_name, **fields)  # type: ignore[call-overload]


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]
    tool_file: Path


async def call_mcp_tool(
    tool_file: Path,
    tool_name: str,
    arguments: dict[str, Any]
) -> str:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(tool_file)],
        cwd=str(tool_file.parent),
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

                if result.content:
                    texts = [
                        c.text for c in result.content if hasattr(c, "text")
                    ]
                    return "\n".join(texts) if texts else str(result.content)
                return str(result)
    except Exception as e:
        return f"Error calling tool {tool_name}: {e}"


async def scan_mcp_tools(tool_file: Path) -> list[MCPToolInfo]:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(tool_file)],
        cwd=str(tool_file.parent),
    )

    tools_info: list[MCPToolInfo] = []

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()

                for mcp_tool in result.tools:
                    tools_info.append(MCPToolInfo(
                        name=mcp_tool.name,
                        description=mcp_tool.description or "",
                        input_schema=mcp_tool.inputSchema or {},
                        tool_file=tool_file,
                    ))
    except Exception as e:
        print(f"Warning: Failed to scan tools from {tool_file}: {e}")

    return tools_info


class MCPManager:
    def __init__(self):
        self._tools_info_cache: list[MCPToolInfo] = []
        self._tools_cache: list[StructuredTool] = []
        self._initialized = False
        self._lock = asyncio.Lock()

    def scan_tool_files(self) -> list[Path]:
        if not GENERATED_TOOLS_DIR.exists():
            return []
        return list(GENERATED_TOOLS_DIR.glob("*.py"))

    async def _scan_all_tools(self) -> list[MCPToolInfo]:
        tool_files = self.scan_tool_files()
        all_tools_info: list[MCPToolInfo] = []

        for tool_file in tool_files:
            tools_info = await scan_mcp_tools(tool_file)
            all_tools_info.extend(tools_info)

        return all_tools_info

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            self._tools_info_cache = await self._scan_all_tools()
            self._initialized = True

    async def get_tools(self) -> list[StructuredTool]:
        await self.initialize()

        if self._tools_cache:
            return self._tools_cache

        tools: list[StructuredTool] = []

        for tool_info in self._tools_info_cache:
            args_model = create_args_model(
                tool_info.name, tool_info.input_schema
            )

            _tool_file = tool_info.tool_file
            _tool_name = tool_info.name

            async def async_run(
                __tool_file: Path = _tool_file,
                __tool_name: str = _tool_name,
                **kwargs: Any
            ) -> str:
                return await call_mcp_tool(__tool_file, __tool_name, kwargs)

            def sync_run(
                __tool_file: Path = _tool_file,
                __tool_name: str = _tool_name,
                **kwargs: Any
            ) -> str:
                return asyncio.run(
                    call_mcp_tool(__tool_file, __tool_name, kwargs)
                )

            langchain_tool = StructuredTool(
                name=tool_info.name,
                description=tool_info.description,
                args_schema=args_model,
                func=sync_run,
                coroutine=async_run,
            )
            tools.append(langchain_tool)

        self._tools_cache = tools
        return tools

    async def refresh_tools(self) -> list[StructuredTool]:
        async with self._lock:
            self._initialized = False
            self._tools_info_cache.clear()
            self._tools_cache.clear()

        return await self.get_tools()

    async def stop_all_servers(self) -> None:
        pass


# Singleton
mcp_manager = MCPManager()