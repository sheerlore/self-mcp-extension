"""
MCP Manager - MCP server processes and LangChain tool integration.

Note: Each tool invocation creates a fresh connection to avoid
async context issues with stdio_client across different tasks.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field, create_model

# ディレクトリパスの設定
BASE_DIR = Path(__file__).parent
GENERATED_TOOLS_DIR = BASE_DIR / "generated_tools"


def json_schema_to_pydantic_field(
    name: str,
    schema: dict[str, Any]
) -> tuple[type, Any]:
    """JSONスキーマからPydanticフィールドを作成"""
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
    """MCPツールの入力スキーマからPydanticモデルを動的に作成"""
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    fields = {}
    for prop_name, prop_schema in properties.items():
        python_type, field = json_schema_to_pydantic_field(
            prop_name, prop_schema
        )
        # 必須フィールドでない場合はデフォルト値をNoneに
        if prop_name not in required and field.default is ...:
            field = Field(default=None, description=field.description)
        fields[prop_name] = (python_type, field)

    model_name = f"{tool_name.title().replace('_', '')}Args"
    return create_model(model_name, **fields)  # type: ignore[call-overload]


@dataclass
class MCPToolInfo:
    """MCPツールの情報を保持"""
    name: str
    description: str
    input_schema: dict[str, Any]
    tool_file: Path


async def call_mcp_tool(
    tool_file: Path,
    tool_name: str,
    arguments: dict[str, Any]
) -> str:
    """
    MCPツールを呼び出す（毎回新しい接続を作成）

    Note: stdio_clientは同一タスク内でenter/exitする必要があるため、
    呼び出しごとに新しい接続を作成・終了する設計としている。
    """
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

                # 結果のcontentからテキストを抽出
                if result.content:
                    texts = [
                        c.text for c in result.content if hasattr(c, "text")
                    ]
                    return "\n".join(texts) if texts else str(result.content)
                return str(result)
    except Exception as e:
        return f"Error calling tool {tool_name}: {e}"


async def scan_mcp_tools(tool_file: Path) -> list[MCPToolInfo]:
    """
    MCPサーバーに接続してツール情報を取得

    Note: この関数は接続を作成して情報を取得後、すぐに閉じる。
    """
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
    """generated_tools/ 内のMCPサーバーを管理"""

    def __init__(self):
        self._tools_info_cache: list[MCPToolInfo] = []
        self._tools_cache: list[StructuredTool] = []
        self._initialized = False
        self._lock = asyncio.Lock()

    def scan_tool_files(self) -> list[Path]:
        """generated_tools/ 内の.pyファイルをスキャン"""
        if not GENERATED_TOOLS_DIR.exists():
            return []
        return list(GENERATED_TOOLS_DIR.glob("*.py"))

    async def _scan_all_tools(self) -> list[MCPToolInfo]:
        """全てのツールファイルをスキャンしてツール情報を取得"""
        tool_files = self.scan_tool_files()
        all_tools_info: list[MCPToolInfo] = []

        for tool_file in tool_files:
            tools_info = await scan_mcp_tools(tool_file)
            all_tools_info.extend(tools_info)

        return all_tools_info

    async def initialize(self) -> None:
        """全てのツールファイルをスキャンして情報をキャッシュ"""
        async with self._lock:
            if self._initialized:
                return

            self._tools_info_cache = await self._scan_all_tools()
            self._initialized = True

    async def get_tools(self) -> list[StructuredTool]:
        """全MCPツールをLangChain形式で返す"""
        await self.initialize()

        # キャッシュがあればそれを返す
        if self._tools_cache:
            return self._tools_cache

        tools: list[StructuredTool] = []

        for tool_info in self._tools_info_cache:
            # 入力スキーマからPydanticモデルを作成
            args_model = create_args_model(
                tool_info.name, tool_info.input_schema
            )

            # クロージャ用に変数をキャプチャ
            _tool_file = tool_info.tool_file
            _tool_name = tool_info.name

            async def async_run(
                __tool_file: Path = _tool_file,
                __tool_name: str = _tool_name,
                **kwargs: Any
            ) -> str:
                """非同期でMCPツールを呼び出す"""
                return await call_mcp_tool(__tool_file, __tool_name, kwargs)

            def sync_run(
                __tool_file: Path = _tool_file,
                __tool_name: str = _tool_name,
                **kwargs: Any
            ) -> str:
                """同期でMCPツールを呼び出す"""
                return asyncio.run(
                    call_mcp_tool(__tool_file, __tool_name, kwargs)
                )

            # StructuredToolを作成
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
        """ツールを再スキャンして更新"""
        async with self._lock:
            self._initialized = False
            self._tools_info_cache.clear()
            self._tools_cache.clear()

        return await self.get_tools()

    async def stop_all_servers(self) -> None:
        """
        互換性のために残しているメソッド

        Note: 新しい設計では永続的な接続を保持しないため、
        このメソッドは何もしない。
        """
        pass


# シングルトンインスタンス
mcp_manager = MCPManager()


async def main():
    """テスト実行"""
    print("=== MCPManager Test ===\n")

    print("Scanning tool files...")
    files = mcp_manager.scan_tool_files()
    print(f"Found {len(files)} tool files: {[f.name for f in files]}\n")

    if not files:
        print("No tool files found. Create some tools first!")
        return

    print("Getting LangChain tools...")
    tools = await mcp_manager.get_tools()
    print(f"Converted {len(tools)} tools:\n")

    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Args: {tool.args_schema.model_json_schema()}\n")

    # テスト実行
    if tools:
        print("Testing first tool...")
        test_tool = tools[0]
        print(f"Tool: {test_tool.name}")

        # XORツールがあればテスト
        if test_tool.name == "calculate_xor":
            result = await test_tool.ainvoke({
                "bits1": "1100",
                "bits2": "1010"
            })
            print(f"Result: {result}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
