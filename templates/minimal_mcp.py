from mcp.server.fastmcp import FastMCP

# Define server name
mcp = FastMCP("my-tool")


@mcp.tool()
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


if __name__ == "__main__":
    mcp.run()
