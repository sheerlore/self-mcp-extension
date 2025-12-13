"""
AI Agent Experiment TUI - Self-Evolving MCP
A hacker-style terminal interface for AI agent interactions.
"""

import asyncio
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual.binding import Binding

from agent import graph, get_all_tools

# 環境変数を読み込み
load_dotenv()


class AgentTUI(App):
    """AI Agent実験用TUIアプリケーション"""

    CSS = """
    Screen {
        background: #0a0a0a;
    }

    Header {
        background: #1a1a2e;
        color: #00ff41;
        text-style: bold;
    }

    Footer {
        background: #1a1a2e;
        color: #00ff41;
    }

    #main-container {
        height: 100%;
    }

    .column {
        height: 100%;
        border: solid #00ff41;
        background: #0d0d0d;
    }

    #chat-column {
        width: 60%;
    }

    #system-column {
        width: 40%;
    }

    .panel-title {
        dock: top;
        height: 3;
        background: #16213e;
        color: #00ff41;
        text-align: center;
        text-style: bold;
        padding: 1;
        border-bottom: solid #00ff41;
    }

    #chat-log {
        height: 1fr;
        background: #0a0a0a;
        color: #e0e0e0;
        padding: 1;
        scrollbar-color: #00ff41;
        scrollbar-color-hover: #00ff88;
        scrollbar-color-active: #00ffaa;
    }

    #mcp-tools-panel {
        height: auto;
        max-height: 40%;
        background: #0d0d0d;
        border-bottom: solid #00ff41;
    }

    #mcp-tools-log {
        height: auto;
        max-height: 100%;
        min-height: 5;
        background: #0a0a0a;
        color: #00ccff;
        padding: 1;
        scrollbar-color: #00ccff;
    }

    #system-log {
        height: 1fr;
        background: #0a0a0a;
        color: #ffcc00;
        padding: 1;
        scrollbar-color: #ff6600;
        scrollbar-color-hover: #ff8833;
        scrollbar-color-active: #ffaa55;
    }

    #chat-input {
        dock: bottom;
        height: 3;
        background: #1a1a2e;
        color: #00ff41;
        border: solid #00ff41;
        padding: 0 1;
    }

    #chat-input:focus {
        border: solid #00ff88;
        background: #1e1e3e;
    }

    Input > .input--placeholder {
        color: #4a4a4a;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_logs", "Clear Logs", show=True),
        Binding("ctrl+r", "refresh_tools", "Refresh Tools", show=True),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    TITLE = "🤖 AI Agent Lab"
    SUB_TITLE = "Self-Evolving MCP Terminal"

    def compose(self) -> ComposeResult:
        """UIコンポーネントを構成"""
        yield Header()

        with Horizontal(id="main-container"):
            # 左側: メインチャット
            with Vertical(id="chat-column", classes="column"):
                yield Static("[ MAIN CHAT ]", classes="panel-title")
                yield RichLog(
                    id="chat-log", highlight=True, markup=True, wrap=True
                )
                yield Input(
                    placeholder=">>> メッセージを入力してEnter...",
                    id="chat-input"
                )

            # 右側: MCPツール + システムログ
            with Vertical(id="system-column", classes="column"):
                # MCPツールリストパネル
                with Vertical(id="mcp-tools-panel"):
                    yield Static(
                        "[ 🔧 MCP TOOLS ] (Ctrl+R: Refresh)",
                        classes="panel-title"
                    )
                    yield RichLog(
                        id="mcp-tools-log", highlight=True, markup=True
                    )

                # システムログパネル
                yield Static(
                    "[ SYSTEM LOG ]",
                    classes="panel-title"
                )
                yield RichLog(id="system-log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        """アプリ起動時の初期化"""
        self._log_system("[bold green]>>> システム起動[/]")
        self._log_system("[dim]AI Agent Lab v1.0 initialized[/]")
        self._log_system("[dim]LangGraph + Gemini backend ready[/]")
        self._log_system("[dim]Waiting for user input...[/]")
        self._log_system("")

        self._log_chat(
            "[bold cyan]🤖 AI:[/] こんにちは！AIエージェント実験環境へようこそ。"
        )
        self._log_chat(
            "[dim]    何か質問があれば、下の入力欄に入力してください。[/]"
        )
        self._log_chat("")

        # 入力フィールドにフォーカス
        self.query_one("#chat-input", Input).focus()

        # MCPツールリストを読み込み
        asyncio.create_task(self._refresh_mcp_tools())

    def _get_timestamp(self) -> str:
        """現在のタイムスタンプを取得"""
        return datetime.now().strftime("%H:%M:%S")

    def _log_chat(self, message: str) -> None:
        """チャットログにメッセージを追加"""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(message)

    def _log_system(self, message: str) -> None:
        """システムログにメッセージを追加"""
        system_log = self.query_one("#system-log", RichLog)
        timestamp = self._get_timestamp()
        system_log.write(f"[dim]{timestamp}[/] {message}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """メッセージ送信時の処理"""
        message = event.value.strip()
        if not message:
            return

        # 入力をクリア
        event.input.value = ""

        # ユーザーメッセージを表示
        self._log_chat(f"[bold magenta]👤 You:[/] {message}")
        self._log_system("[cyan]>>> User input received[/]")

        # 非同期でAI応答を処理
        asyncio.create_task(self._process_ai_response(message))

    async def _process_ai_response(self, user_message: str) -> None:
        """AIの応答を非同期で処理（LangGraph経由）"""
        import time
        start_time = time.time()

        self._log_system("[bold cyan]━━━ NEW REQUEST ━━━[/]")
        self._log_system(f"[cyan]📝 Input: {user_message[:50]}...[/]"
                         if len(user_message) > 50
                         else f"[cyan]📝 Input: {user_message}[/]")

        # 利用可能なツールを表示
        try:
            all_tools = await get_all_tools()
            mcp_tools = [t for t in all_tools if t.name != "create_mcp_tool"]
            if mcp_tools:
                self._log_system("[blue]🔧 Available MCP Tools:[/]")
                for t in mcp_tools:
                    desc = t.description[:40] + "..." \
                        if len(t.description) > 40 else t.description
                    self._log_system(f"[dim]   • {t.name}: {desc}[/]")
            else:
                self._log_system("[dim]🔧 No MCP tools available yet[/]")
        except Exception:
            self._log_system("[dim]🔧 Tools: loading...[/]")

        self._log_chat("[dim]    🤖 考え中...[/]")

        final_response = ""
        token_count = 0
        tool_call_count = 0

        try:
            # LangGraphのastream_eventsでストリーミング処理
            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                version="v2"
            ):
                event_type = event.get("event")
                event_name = event.get("name", "")

                # グラフノードの遷移を追跡
                if event_type == "on_chain_start":
                    if event_name in ["call_model", "tools"]:
                        icon = "🧠" if event_name == "call_model" else "🔧"
                        self._log_system(
                            f"[blue]▶ Node: {icon} {event_name}[/]"
                        )

                elif event_type == "on_chain_end":
                    if event_name in ["call_model", "tools"]:
                        elapsed = time.time() - start_time
                        self._log_system(
                            f"[dim]   └─ Completed ({elapsed:.2f}s)[/]"
                        )

                elif event_type == "on_chat_model_start":
                    data = event.get("data") or {}
                    model = "Gemini"
                    if isinstance(data, dict):
                        params = data.get("invocation_params")
                        if isinstance(params, dict):
                            model = str(params.get("model", "Gemini"))
                    self._log_system(f"[yellow]🧠 LLM: {model}[/]")

                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token_count += 1
                        if token_count % 20 == 0:
                            self._log_system(
                                f"[dim]   ├─ {token_count} tokens[/]"
                            )

                elif event_type == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    # ツール呼び出しの検出
                    if output and hasattr(output, "tool_calls"):
                        tool_calls = output.tool_calls
                        if tool_calls:
                            self._log_system(
                                f"[yellow]🔗 Tool calls detected: "
                                f"{len(tool_calls)}[/]"
                            )
                            for tc in tool_calls:
                                self._log_system(
                                    f"[yellow]   └─ {tc.get('name', '?')}[/]"
                                )
                        else:
                            self._log_system("[green]✓ Response ready[/]")
                    # 最終応答を取得
                    if output and hasattr(output, "content"):
                        if output.content:
                            final_response = output.content

                elif event_type == "on_tool_start":
                    tool_call_count += 1
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})

                    self._log_system("")
                    self._log_system(
                        f"[bold yellow]━━━ TOOL #{tool_call_count} ━━━[/]"
                    )
                    self._log_system(
                        f"[bold yellow]🔨 {tool_name}[/]"
                    )

                    # ツール固有の詳細表示
                    if tool_name == "create_mcp_tool":
                        filename = tool_input.get("filename", "?")
                        code = tool_input.get("code", "")
                        lines = code.count("\n") + 1 if code else 0
                        self._log_system(
                            f"[yellow]   📄 File: {filename}[/]"
                        )
                        self._log_system(
                            f"[yellow]   📏 Code: {lines} lines[/]"
                        )
                    else:
                        # その他のツール
                        for key, value in tool_input.items():
                            val_str = str(value)
                            if len(val_str) > 30:
                                val_str = val_str[:30] + "..."
                            self._log_system(
                                f"[yellow]   {key}: {val_str}[/]"
                            )

                elif event_type == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")
                    output_str = str(tool_output)
                    if len(output_str) > 60:
                        output_str = output_str[:60] + "..."

                    if "Successfully" in str(tool_output):
                        self._log_system(
                            "[bold green]✅ Success[/]"
                        )
                    elif "Error" in str(tool_output):
                        self._log_system(
                            "[bold red]❌ Failed[/]"
                        )
                    else:
                        self._log_system(
                            "[bold green]✓ Done[/]"
                        )
                    self._log_system(f"[dim]   Result: {output_str}[/]")
                    self._log_system(
                        "[bold yellow]━━━━━━━━━━━━━━━━━━━━[/]"
                    )

            # 処理完了
            elapsed = time.time() - start_time
            self._log_system("")
            self._log_system("[bold cyan]━━━ COMPLETED ━━━[/]")
            self._log_system(
                f"[cyan]⏱ Total time: {elapsed:.2f}s[/]"
            )
            self._log_system(
                f"[cyan]📊 Tokens: {token_count} | Tools: {tool_call_count}[/]"
            )
            self._log_system("")

            # ツールが使用された場合、MCPツールリストを更新
            if tool_call_count > 0:
                await self._refresh_mcp_tools()

            # AI応答を表示
            if final_response:
                self._log_chat(f"[bold cyan]🤖 AI:[/] {final_response}")
            else:
                self._log_chat(
                    "[bold red]🤖 AI:[/] 応答を取得できませんでした。"
                )
            self._log_chat("")

        except Exception as e:
            elapsed = time.time() - start_time
            self._log_system("[bold red]━━━ ERROR ━━━[/]")
            self._log_system(f"[red]❌ {type(e).__name__}: {e}[/]")
            self._log_system(f"[dim]   After {elapsed:.2f}s[/]")
            self._log_chat(f"[bold red]🤖 AI:[/] エラー: {e}")
            self._log_chat("")

    def action_clear_logs(self) -> None:
        """ログをクリア"""
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#system-log", RichLog).clear()
        self._log_system("[bold green]>>> ログをクリアしました[/]")

    def action_focus_input(self) -> None:
        """入力フィールドにフォーカス"""
        self.query_one("#chat-input", Input).focus()

    def action_refresh_tools(self) -> None:
        """MCPツールリストをリフレッシュ"""
        asyncio.create_task(self._refresh_mcp_tools())

    async def _refresh_mcp_tools(self) -> None:
        """MCPツールリストを更新"""
        mcp_log = self.query_one("#mcp-tools-log", RichLog)
        mcp_log.clear()
        mcp_log.write("[dim]Loading MCP tools...[/]")

        try:
            all_tools = await get_all_tools()
            mcp_tools = [t for t in all_tools if t.name != "create_mcp_tool"]

            mcp_log.clear()

            if mcp_tools:
                mcp_log.write(
                    f"[bold green]✓ {len(mcp_tools)} tool(s) available[/]"
                )
                mcp_log.write("")

                for tool in mcp_tools:
                    # ツール名
                    mcp_log.write(f"[bold cyan]📦 {tool.name}[/]")

                    # 説明
                    desc = tool.description or "No description"
                    # 説明を複数行に分割して表示
                    desc_lines = desc.split("\n")
                    for line in desc_lines[:3]:  # 最大3行
                        if line.strip():
                            mcp_log.write(f"[dim]   {line.strip()}[/]")

                    # 引数情報
                    if hasattr(tool, "args_schema") and tool.args_schema:
                        args_schema = tool.args_schema
                        if hasattr(args_schema, "model_json_schema"):
                            schema = args_schema.model_json_schema()
                            props = schema.get("properties", {})
                            if props:
                                args_str = ", ".join(props.keys())
                                mcp_log.write(
                                    f"[yellow]   Args: {args_str}[/]"
                                )
                    mcp_log.write("")
            else:
                mcp_log.write("[dim]No MCP tools available yet.[/]")
                mcp_log.write("")
                mcp_log.write("[dim]Ask AI to create a tool![/]")
                mcp_log.write('[dim]Example: "Create a calculator tool"[/]')

            self._log_system("[green]✓ MCP tools refreshed[/]")

        except Exception as e:
            mcp_log.clear()
            mcp_log.write(f"[red]Error loading tools: {e}[/]")
            self._log_system(f"[red]✗ Failed to refresh tools: {e}[/]")


def main():
    """アプリケーションのエントリーポイント"""
    app = AgentTUI()
    app.run()


if __name__ == "__main__":
    main()
