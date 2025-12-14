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

# Load environment variables
load_dotenv()


class AgentTUI(App):
    """AI Agent Experiment TUI Application"""

    CSS = """
    Screen {
        background: #050505;
        align: center middle;
    }

    Header {
        background: #002b00;
        color: #00ff41;
        text-style: bold;
        dock: top;
        height: 1;
    }

    Footer {
        background: #002b00;
        color: #00ff41;
        dock: bottom;
        height: 1;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    .column {
        height: 100%;
        border: solid #004400;
        background: #0c0c0c;
        margin: 0 1;
    }

    #chat-column {
        width: 2fr;
    }

    #system-column {
        width: 1fr;
    }

    .panel-title {
        height: 1;
        background: #003300;
        color: #00ff41;
        text-align: center;
        text-style: bold;
        border-bottom: solid #004400;
    }

    RichLog {
        height: 1fr;
        background: #080808;
        color: #e0e0e0;
        padding: 0 1;
        scrollbar-gutter: stable;
        scrollbar-color: #00ff41;
        border: none;
    }
    
    #chat-log {
        background: #0a0a0a;
    }

    #mcp-tools-panel {
        height: 35%;
        border-bottom: solid #004400;
        background: #0c0c0c;
    }

    #mcp-tools-log {
        background: #0a0a0a;
        color: #00ccff;
        scrollbar-color: #00ccff;
    }

    #system-log {
        background: #080808;
        color: #ffcc00;
        scrollbar-color: #ff6600;
    }

    #chat-input {
        height: 3;
        background: #050505;
        color: #00ff41;
        border: solid #004400;
        padding: 0 1;
        margin-top: 1;
    }

    #chat-input:focus {
        border: solid #00ff41;
        background: #001100;
    }

    Input > .input--placeholder {
        color: #336633;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_logs", "Clear Logs", show=True),
        Binding("ctrl+r", "refresh_tools", "Refresh Tools", show=True),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    TITLE = "⚡ SELF-MCP TERMINAL"
    SUB_TITLE = "v1.0 // DO NOT POWER OFF"

    def compose(self) -> ComposeResult:
        """Configure UI components"""
        yield Header()

        with Horizontal(id="main-container"):
            # Left: Main Chat
            with Vertical(id="chat-column", classes="column"):
                yield Static("[ COMMUNICATION CHANNEL ]", classes="panel-title")
                yield RichLog(
                    id="chat-log", highlight=True, markup=True, wrap=True
                )
                yield Input(
                    placeholder=">>> INPUT COMMAND SEQUENCE...",
                    id="chat-input"
                )

            # Right: MCP Tools + System Log
            with Vertical(id="system-column", classes="column"):
                # MCP Tools Panel
                with Vertical(id="mcp-tools-panel"):
                    yield Static(
                        "[ MODULE REGISTRY ] (CTRL+R: REFRESH)",
                        classes="panel-title"
                    )
                    yield RichLog(
                        id="mcp-tools-log", highlight=True, markup=True
                    )

                # System Log Panel
                yield Static(
                    "[ KERNEL LOG ]",
                    classes="panel-title"
                )
                yield RichLog(id="system-log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        """Initialize application on mount"""
        self._log_system("[bold green]>>> SYSTEM BOOT SEQUENCE INITIATED[/]")
        self._log_system("[dim]Kernel: Self-MCP Terminal v1.0[/]")
        self._log_system("[dim]Modules: LangGraph [ONLINE], Gemini [ONLINE][/]")
        self._log_system("[dim]Status: WAITING FOR COMMAND...[/]")
        self._log_system("")

        self._log_chat(
            "[bold cyan]🤖 AI:[/] SYSTEM ONLINE. READY TO EVOLVE."
        )
        self._log_chat(
            "[dim]    Enter command or request to begin adaptation protocol.[/]"
        )
        self._log_chat("")

        # Focus input field
        self.query_one("#chat-input", Input).focus()

        # Load MCP tools list
        asyncio.create_task(self._refresh_mcp_tools())

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%H:%M:%S")

    def _log_chat(self, message: str) -> None:
        """Add message to chat log"""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(message)

    def _log_system(self, message: str) -> None:
        """Add message to system log"""
        system_log = self.query_one("#system-log", RichLog)
        timestamp = self._get_timestamp()
        system_log.write(f"[dim]{timestamp}[/] {message}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle message submission"""
        message = event.value.strip()
        if not message:
            return

        # Clear input
        event.input.value = ""

        # Display user message
        self._log_chat(f"[bold magenta]👤 OPERATOR:[/] {message}")
        self._log_system("[cyan]>>> INPUT SEQUENCE RECEIVED[/]")

        # Process AI response asynchronously
        asyncio.create_task(self._process_ai_response(message))

    async def _process_ai_response(self, user_message: str) -> None:
        """Process AI response asynchronously (via LangGraph)"""
        import time
        start_time = time.time()

        self._log_system("[bold cyan]━━━ NEW REQUEST ━━━[/]")
        self._log_system(f"[cyan]📝 DATA PACKET: {user_message[:50]}...[/]"
                         if len(user_message) > 50
                         else f"[cyan]📝 DATA PACKET: {user_message}[/]")

        # Display available tools
        try:
            all_tools = await get_all_tools()
            mcp_tools = [t for t in all_tools if t.name != "create_mcp_tool"]
            if mcp_tools:
                self._log_system("[blue]🔧 MODULES AVAILABLE:[/]")
                for t in mcp_tools:
                    desc = t.description[:40] + "..." \
                        if len(t.description) > 40 else t.description
                    self._log_system(f"[dim]   • {t.name}: {desc}[/]")
            else:
                self._log_system("[dim]🔧 NO MODULES DETECTED[/]")
        except Exception:
            self._log_system("[dim]🔧 Tools: loading...[/]")

        self._log_chat("[dim]    🤖 PROCESSING...[/]")

        final_response = ""
        token_count = 0
        tool_call_count = 0

        try:
            # Stream events using LangGraph's astream_events
            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                version="v2"
            ):
                event_type = event.get("event")
                event_name = event.get("name", "")

                # Track graph node transitions
                if event_type == "on_chain_start":
                    if event_name in ["call_model", "tools"]:
                        icon = "🧠" if event_name == "call_model" else "🔧"
                        self._log_system(
                            f"[blue]▶ PROCESS NODE: {icon} {event_name.upper()}[/]"
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
                    self._log_system(f"[yellow]🧠 COGNITIVE MODEL: {model}[/]")

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
                    # Detect tool calls
                    if output and hasattr(output, "tool_calls"):
                        tool_calls = output.tool_calls
                        if tool_calls:
                            self._log_system(
                                f"[yellow]🔗 EXTERNAL CALLS DETECTED: "
                                f"{len(tool_calls)}[/]"
                            )
                            for tc in tool_calls:
                                self._log_system(
                                    f"[yellow]   └─ {tc.get('name', '?')}[/]"
                                )
                        else:
                            self._log_system("[green]✓ OUTPUT GENERATED[/]")
                    # Get final response
                    if output and hasattr(output, "content"):
                        if output.content:
                            final_response = output.content

                elif event_type == "on_tool_start":
                    tool_call_count += 1
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})

                    self._log_system("")
                    self._log_system(
                        f"[bold yellow]━━━ EXECUTING MODULE #{tool_call_count} ━━━[/]"
                    )
                    self._log_system(
                        f"[bold yellow]🔨 {tool_name}[/]"
                    )

                    # Display tool-specific details
                    if tool_name == "create_mcp_tool":
                        filename = tool_input.get("filename", "?")
                        code = tool_input.get("code", "")
                        lines = code.count("\n") + 1 if code else 0
                        self._log_system(
                            f"[yellow]   📄 [DATA_TARGET]: {filename}[/]"
                        )
                        self._log_system(
                            f"[yellow]   📏 [PAYLOAD_SIZE]: {lines} lines[/]"
                        )
                    else:
                        # Other tools
                        for key, value in tool_input.items():
                            val_str = str(value)
                            if len(val_str) > 30:
                                val_str = val_str[:30] + "..."
                            self._log_system(
                                f"[yellow]   [PARAM:{key}]: {val_str}[/]"
                            )

                elif event_type == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")
                    output_str = str(tool_output)
                    if len(output_str) > 60:
                        output_str = output_str[:60] + "..."

                    if "Successfully" in str(tool_output):
                        self._log_system(
                            "[bold green]✅ OPERATION SUCCESSFUL[/]"
                        )
                    elif "Error" in str(tool_output):
                        self._log_system(
                            "[bold red]❌ OPERATION FAILED[/]"
                        )
                    else:
                        self._log_system(
                            "[bold green]✓ SEQUENCE COMPLETE[/]"
                        )
                    self._log_system(f"[dim]   >> OUTPUT: {output_str}[/]")
                    self._log_system(
                        "[bold yellow]━━━━━━━━━━━━━━━━━━━━[/]"
                    )

            # Process complete
            elapsed = time.time() - start_time
            self._log_system("")
            self._log_system("[bold cyan]━━━ REQUEST COMPLETE ━━━[/]")
            self._log_system(
                f"[cyan]⏱ EXECUTION TIME: {elapsed:.2f}s[/]"
            )
            self._log_system(
                f"[cyan]📊 Tokens: {token_count} | Tools: {tool_call_count}[/]"
            )
            self._log_system("")

            # Update MCP tool list if tools were used
            if tool_call_count > 0:
                await self._refresh_mcp_tools()

            # Display AI response
            if final_response:
                self._log_chat(f"[bold cyan]🤖 AI:[/] {final_response}")
            else:
                self._log_chat(
                    "[bold red]🤖 AI:[/] NO RESPONSE RECEIVED FROM CORE."
                )
            self._log_chat("")

        except Exception as e:
            elapsed = time.time() - start_time
            self._log_system("[bold red]━━━ ERROR ━━━[/]")
            self._log_system(f"[red]❌ {type(e).__name__}: {e}[/]")
            self._log_system(f"[dim]   After {elapsed:.2f}s[/]")
            self._log_chat(f"[bold red]🤖 AI:[/] SYSTEM ERROR: {e}")
            self._log_chat("")

    def action_clear_logs(self) -> None:
        """Clear logs"""
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#system-log", RichLog).clear()
        self._log_system("[bold green]>>> LOG BUFFER CLEARED[/]")

    def action_focus_input(self) -> None:
        """Focus input field"""
        self.query_one("#chat-input", Input).focus()

    def action_refresh_tools(self) -> None:
        """Refresh MCP tools list"""
        asyncio.create_task(self._refresh_mcp_tools())

    async def _refresh_mcp_tools(self) -> None:
        """Update MCP tools list"""
        mcp_log = self.query_one("#mcp-tools-log", RichLog)
        mcp_log.clear()
        mcp_log.write("[dim]SCANNING FOR MODULES...[/]")

        try:
            all_tools = await get_all_tools()
            mcp_tools = [t for t in all_tools if t.name != "create_mcp_tool"]

            mcp_log.clear()

            if mcp_tools:
                mcp_log.write(
                    f"[bold green]✓ {len(mcp_tools)} MODULE(S) ONLINE[/]"
                )
                mcp_log.write("")

                for tool in mcp_tools:
                    # Tool name
                    mcp_log.write(f"[bold cyan]📦 {tool.name}[/]")

                    # Description
                    desc = tool.description or "No description"
                    # Split description into multiple lines
                    desc_lines = desc.split("\n")
                    for line in desc_lines[:3]:  # Max 3 lines
                        if line.strip():
                            mcp_log.write(f"[dim]   {line.strip()}[/]")

                    # Argument information
                    if hasattr(tool, "args_schema") and tool.args_schema:
                        args_schema = tool.args_schema
                        if hasattr(args_schema, "model_json_schema"):
                            schema = args_schema.model_json_schema()
                            props = schema.get("properties", {})
                            if props:
                                args_str = ", ".join(props.keys())
                                mcp_log.write(
                                    f"[yellow]   PARAMS: {args_str}[/]"
                                )
                    mcp_log.write("")
            else:
                mcp_log.write("[dim]No MCP tools available yet.[/]")
                mcp_log.write("")
                mcp_log.write("[dim]INITIATE TOOL GENERATION PROTOCOL[/]")
                mcp_log.write('[dim]EX: "Create quantum calculator module"[/]')

            self._log_system("[green]✓ MODULE REGISTRY UPDATED[/]")

        except Exception as e:
            mcp_log.clear()
            mcp_log.write(f"[red]MODULE SCAN ERROR: {e}[/]")
            self._log_system(f"[red]✗ REFRESH FAILURE: {e}[/]")


def main():
    """Application entry point"""
    app = AgentTUI()
    app.run()


if __name__ == "__main__":
    main()
