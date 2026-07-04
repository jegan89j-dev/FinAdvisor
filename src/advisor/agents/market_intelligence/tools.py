"""Market Intelligence Agent tools: quotes, sector performance, FX."""

from typing import Dict, Any, List
from advisor.tools import alpha_vantage as av


def get_market_tools() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible tool schemas for market intelligence."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_stock_quote",
                "description": "Latest price, change, and volume for a stock symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_sector_performance",
                "description": "Real-time and trailing performance for the major US sectors.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_fx_rate",
                "description": "Realtime FX rate between two currencies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_currency": {"type": "string"},
                        "to_currency": {"type": "string"},
                    },
                    "required": ["from_currency", "to_currency"],
                },
            },
        },
    ]


def dispatch_market_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch and execute a market intelligence tool.
    
    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments
        
    Returns:
        Tool result dict
    """
    dispatch = {
        "get_stock_quote": lambda: av.get_quote(args.get("symbol")),
        "get_sector_performance": lambda: av.get_sector_performance(),
        "get_fx_rate": lambda: av.get_fx_rate(
            args.get("from_currency"),
            args.get("to_currency"),
        ),
    }

    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"unknown market intelligence tool: {tool_name}"}

    try:
        return fn()
    except TypeError as e:
        return {"error": f"bad arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
