"""
AI Agent - LangGraph based conversational agent
Using Google Generative AI (Gemini) as the LLM backend.
Capable of creating and using MCP server tools dynamically.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import tools_condition

from mcp_manager import MCPManager

# 環境変数を読み込み
load_dotenv()

# ディレクトリパスの設定
BASE_DIR = Path(__file__).parent
GENERATED_TOOLS_DIR = BASE_DIR / "generated_tools"
TEMPLATE_PATH = BASE_DIR / "templates" / "minimal_mcp.py"

# テンプレートの読み込み
MCP_TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")

# MCPManagerのグローバルインスタンス
mcp_manager = MCPManager()

# システムプロンプトのベース部分
SYSTEM_PROMPT_BASE = f"""あなたは自律的に行動するAIエージェントです。
ユーザーのリクエストを達成するために、ツールを使用します。

## 重要な行動原則（優先順位順）

1. **既存のMCPツールを最優先で使う**
   - 下記の「利用可能なMCPツール」に該当するツールがあれば、それを使う
   - 新しいツールを作成する前に、既存ツールで対応できないか確認する

2. **既存ツールで対応できない場合のみ、新しいツールを作成する**
   - `create_mcp_tool` で新しいMCPツールを作成
   - ユーザーに確認を求めずに即座に作成する

3. **作成したツールを使って結果を出す**
   - ツール作成後、そのツールで処理を実行する

## create_mcp_tool の使い方（新規ツール作成時のみ）

`create_mcp_tool(filename, code)` を呼び出すとMCPツールが作成されます。

### テンプレート:
```python
{MCP_TEMPLATE}
```

### コード作成のルール:
- ファイル名: `<機能名>_tool.py` (例: `calculator_tool.py`)
- サーバー名: `FastMCP("機能名-tool")`
- 関数には `@mcp.tool()` デコレータを付ける
- 型ヒントとdocstringを必ず付ける
- 最後に `if __name__ == "__main__": mcp.run()` を含める
"""


def build_system_prompt(mcp_tools: list[BaseTool]) -> str:
    """動的にシステムプロンプトを構築する"""
    prompt = SYSTEM_PROMPT_BASE

    # 利用可能なMCPツールのリストを追加
    prompt += "\n## 利用可能なMCPツール\n\n"

    if mcp_tools:
        prompt += "以下のツールが利用可能です。該当する機能があれば優先的に使用してください:\n\n"
        for t in mcp_tools:
            prompt += f"- **{t.name}**: {t.description}\n"
    else:
        prompt += "現在、作成済みのMCPツールはありません。\n"
        prompt += "必要に応じて `create_mcp_tool` で新しいツールを作成してください。\n"

    prompt += "\n## 常に利用可能なツール\n\n"
    prompt += "- **create_mcp_tool**: 新しいMCPツールを作成・保存する\n"

    return prompt


# LLMの初期化（ツールは動的にバインド）
# モデル選択肢:
# - "gemini-2.0-flash-exp": 最新の実験版（ツール呼び出しが不安定な場合あり）
# - "gemini-1.5-pro": 安定版（ツール呼び出しが確実）
# - "gemini-1.5-flash": 高速版
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # ツール呼び出しが安定
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)


@tool
def create_mcp_tool(filename: str, code: str) -> str:
    """新しいMCPツールを作成してファイルに保存します。

    このツールを使って、計算、変換、データ処理などの機能を持つ
    カスタムツールを作成できます。作成後、そのツールは自動的に
    利用可能になります。

    Args:
        filename: 保存するファイル名。必ず「_tool.py」で終わること。
                  例: "calculator_tool.py", "converter_tool.py"
        code: FastMCPを使ったPythonコード。テンプレートに従うこと。
              必須要素: FastMCPインポート、@mcp.tool()デコレータ、
              型ヒント、docstring、mcp.run()

    Returns:
        成功時: "Successfully saved {filename} to {path}"
        失敗時: エラーメッセージ
    """
    # ディレクトリが存在しない場合は作成
    GENERATED_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    # ファイルパスの構築
    file_path = GENERATED_TOOLS_DIR / filename

    # セキュリティチェック: ディレクトリトラバーサル防止
    if not file_path.resolve().is_relative_to(GENERATED_TOOLS_DIR.resolve()):
        return f"Error: Invalid filename '{filename}'"

    # ファイルに保存
    file_path.write_text(code, encoding="utf-8")

    return f"Successfully saved {filename} to {file_path}"


# 静的ツール（常に利用可能）
STATIC_TOOLS: list[BaseTool] = [create_mcp_tool]


async def get_all_tools() -> list[BaseTool]:
    """静的ツールとMCPツールを合わせた全ツールリストを取得"""
    try:
        mcp_tools = await mcp_manager.get_tools()
    except Exception as e:
        print(f"Warning: Failed to get MCP tools: {e}")
        mcp_tools = []

    all_tools: list[BaseTool] = list(STATIC_TOOLS)
    all_tools.extend(mcp_tools)
    return all_tools


async def call_model(state: MessagesState) -> MessagesState:
    """LLMを呼び出してメッセージに応答する（動的ツールバインド）"""
    messages = state["messages"]

    # 動的にツールを取得
    all_tools = await get_all_tools()

    # MCPツール（create_mcp_tool以外）を抽出
    mcp_tools = [t for t in all_tools if t.name != "create_mcp_tool"]

    # 動的にシステムプロンプトを構築
    system_prompt = build_system_prompt(mcp_tools)

    # システムプロンプトを先頭に追加（まだない場合）または更新
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + list(messages)
    else:
        # 既存のシステムメッセージを更新
        messages = [SystemMessage(content=system_prompt)] + list(messages[1:])

    # デバッグ: 利用可能なツール名を出力
    tool_names = [t.name for t in all_tools]
    print(f"[DEBUG] Available tools: {tool_names}")

    # ツールをバインド（tool_choice="auto"で自動選択を有効化）
    llm_with_tools = llm.bind_tools(all_tools, tool_choice="auto")

    # 非同期で呼び出し
    response = await llm_with_tools.ainvoke(messages)

    # デバッグ: ツール呼び出しの有無を確認
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"[DEBUG] Tool calls: {response.tool_calls}")
    else:
        print("[DEBUG] No tool calls in response")

    return {"messages": [response]}


async def run_tools(state: MessagesState) -> MessagesState:
    """ツールを実行するカスタムノード（動的ツール対応）"""
    messages = state["messages"]
    last_message = messages[-1]

    # ツール呼び出しがない場合は何もしない
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    # 現時点の全ツールを取得
    all_tools = await get_all_tools()

    # ツール名からツールオブジェクトへのマッピング
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in all_tools}

    # 各ツール呼び出しを実行
    tool_messages: list[AnyMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name in tools_by_name:
            tool_obj = tools_by_name[tool_name]
            try:
                # 非同期実行を試みる
                if hasattr(tool_obj, "ainvoke"):
                    result = await tool_obj.ainvoke(tool_args)
                else:
                    result = tool_obj.invoke(tool_args)
            except Exception as e:
                result = f"Error executing tool {tool_name}: {e}"
        else:
            result = f"Error: Tool '{tool_name}' not found"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    # ツール作成後はMCPサーバーをリフレッシュ
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "create_mcp_tool":
            try:
                # ファイル書き込み完了を少し待つ（サーバー起動時間を考慮）
                await asyncio.sleep(0.5)
                await mcp_manager.refresh_tools()
            except Exception as e:
                print(f"Warning: Failed to refresh MCP tools: {e}")
            break

    return {"messages": tool_messages}


# StateGraphの構築
workflow = StateGraph(MessagesState)

# ノードを追加
workflow.add_node("call_model", call_model)
workflow.add_node("tools", run_tools)

# エッジを追加
workflow.add_edge(START, "call_model")

# tools_conditionを使用: ツール呼び出しがあればtoolsノードへ、なければ終了
workflow.add_conditional_edges("call_model", tools_condition)

# toolsノードからはcall_modelに戻る
workflow.add_edge("tools", "call_model")

# グラフをコンパイル
graph = workflow.compile()


async def cleanup() -> None:
    """リソースのクリーンアップ"""
    await mcp_manager.stop_all_servers()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    async def main() -> None:
        print("=== テスト1: 通常の会話 ===")
        result = await graph.ainvoke(
            {  # type: ignore[arg-type]
                "messages": [HumanMessage(content="こんにちは！")]
            }
        )
        for message in result["messages"]:
            content = getattr(message, "content", "")
            if content:
                print(f"{message.type}: {content[:100]}...")

        print("\n=== テスト2: 利用可能なツール確認 ===")
        tools = await get_all_tools()
        print(f"利用可能なツール: {[t.name for t in tools]}")

        print("\n=== クリーンアップ ===")
        await cleanup()
        print("Done!")

    asyncio.run(main())
