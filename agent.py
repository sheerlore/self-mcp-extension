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

load_dotenv()

BASE_DIR = Path(__file__).parent
GENERATED_TOOLS_DIR = BASE_DIR / "generated_tools"
TEMPLATE_PATH = BASE_DIR / "templates" / "minimal_mcp.py"
MCP_TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")

mcp_manager = MCPManager()

SYSTEM_PROMPT_BASE = f"""You are an autonomous AI agent.
You utilize tools to fulfill user requests.

## Core Principles (In order of priority)

1. **Prioritize Existing MCP Tools**
   - If a tool listed in "Available MCP Tools" below is applicable, use it.
   - Check if existing tools can handle the request before creating a new one.

2. **Create New Tools Only When Necessary**
   - Use `create_mcp_tool` to create a new MCP tool only if existing tools are insufficient.
   - Create the tool immediately without asking the user for confirmation.

3. **Execute with Created Tools**
   - After creating a tool, execute the process using that tool.

## How to use create_mcp_tool (Only for new tool creation)

Call `create_mcp_tool(filename, code)` to create an MCP tool.

### Template:
```python
{MCP_TEMPLATE}
```

### Code Creation Rules:
- Filename: `<function_name>_tool.py` (e.g., `calculator_tool.py`)
- Server Name: `FastMCP("function-name-tool")`
- Decorate functions with `@mcp.tool()`
- MUST include type hints and docstrings
- MUST include `if __name__ == "__main__": mcp.run()` at the end
"""


def build_system_prompt(mcp_tools: list[BaseTool]) -> str:
    """Dynamically build system prompt"""
    prompt = SYSTEM_PROMPT_BASE

    prompt += "\n## Available MCP Tools\n\n"

    if mcp_tools:
        prompt += "The following tools are available. Prioritize using them if applicable:\n\n"
        for t in mcp_tools:
            prompt += f"- **{t.name}**: {t.description}\n"
    else:
        prompt += "No MCP tools created yet.\n"
        prompt += "Create new tools using `create_mcp_tool` as needed.\n"

    prompt += "\n## Always Available Tools\n\n"
    prompt += "- **create_mcp_tool**: Create and save a new MCP tool\n"

    return prompt


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


@tool
def create_mcp_tool(filename: str, code: str) -> str:
    """Creates a new MCP tool and saves it to a file.

    Use this tool to create custom tools for calculations, conversions,
    data processing, etc. Once created, the tool automatically becomes
    available for use.

    Args:
        filename: Filename to save. MUST end with "_tool.py".
                  e.g., "calculator_tool.py", "converter_tool.py"
        code: Python code using FastMCP. Must follow the template.
              Required elements: FastMCP import, @mcp.tool() decorator,
              type hints, docstrings, mcp.run()

    Returns:
        Success: "Successfully saved {filename} to {path}"
        Failure: Error message
    """
    # Create directory if it doesn't exist
    GENERATED_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    # Build file path
    file_path = GENERATED_TOOLS_DIR / filename

    # Security check: Prevent directory traversal
    if not file_path.resolve().is_relative_to(GENERATED_TOOLS_DIR.resolve()):
        return f"Error: Invalid filename '{filename}'"

    # Save to file
    file_path.write_text(code, encoding="utf-8")

    return f"Successfully saved {filename} to {file_path}"


# Static tools (Always available)
STATIC_TOOLS: list[BaseTool] = [create_mcp_tool]


async def get_all_tools() -> list[BaseTool]:
    """Get all tools including static and MCP tools"""
    try:
        mcp_tools = await mcp_manager.get_tools()
    except Exception as e:
        print(f"Warning: Failed to get MCP tools: {e}")
        mcp_tools = []

    all_tools: list[BaseTool] = list(STATIC_TOOLS)
    all_tools.extend(mcp_tools)
    return all_tools


async def call_model(state: MessagesState) -> MessagesState:
    """Call LLM to respond to messages (Dynamic tool binding)"""
    messages = state["messages"]

    all_tools = await get_all_tools()
    mcp_tools = [t for t in all_tools if t.name != "create_mcp_tool"]
    system_prompt = build_system_prompt(mcp_tools)

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + list(messages)
    else:
        messages = [SystemMessage(content=system_prompt)] + list(messages[1:])

    tool_names = [t.name for t in all_tools]
    print(f"[DEBUG] Available tools: {tool_names}")

    llm_with_tools = llm.bind_tools(all_tools, tool_choice="auto")
    response = await llm_with_tools.ainvoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"[DEBUG] Tool calls: {response.tool_calls}")
    else:
        print("[DEBUG] No tool calls in response")

    return {"messages": [response]}


async def run_tools(state: MessagesState) -> MessagesState:
    """Custom node to run tools (Dynamic tool support)"""
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    all_tools = await get_all_tools()

    tools_by_name: dict[str, BaseTool] = {t.name: t for t in all_tools}
    tool_messages: list[AnyMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name in tools_by_name:
            tool_obj = tools_by_name[tool_name]
            try:
                # Try asynchronous execution
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

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "create_mcp_tool":
            try:
                await asyncio.sleep(0.5)
                await mcp_manager.refresh_tools()
            except Exception as e:
                print(f"Warning: Failed to refresh MCP tools: {e}")
            break

    return {"messages": tool_messages}


# StateGraphの構築
# START -> [call_model <-> tools] -> END
workflow = StateGraph(MessagesState)
workflow.add_node("call_model", call_model)
workflow.add_node("tools", run_tools)
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", tools_condition)
workflow.add_edge("tools", "call_model")

graph = workflow.compile()

async def cleanup() -> None:
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
