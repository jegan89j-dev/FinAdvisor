"""Portfolio Analysis Agent tools: company overview, news sentiment, technicals."""

from typing import Dict, Any, List
from advisor.tools import alpha_vantage as av


def get_portfolio_tools() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible tool schemas for portfolio analysis."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_company_overview",
                "description": "Fundamentals for a stock: sector, P/E, market cap, dividend yield, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_news_sentiment",
                "description": "Recent news headlines and sentiment scores. Provide tickers (comma-separated) or topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {"type": "string", "description": "Comma-separated tickers, e.g. AAPL,MSFT"},
                        "topics": {"type": "string", "description": "AV topics, e.g. technology,ipo,economy_macro"},
                        "limit": {"type": "integer", "default": 10},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_technical_indicator",
                "description": "Compute a technical indicator (RSI, SMA, EMA, MACD) for a symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "indicator": {"type": "string", "enum": ["RSI", "SMA", "EMA", "MACD"]},
                        "interval": {"type": "string", "default": "daily"},
                        "time_period": {"type": "integer", "default": 14},
                    },
                    "required": ["symbol", "indicator"],
                },
            },
        },
    ]


def dispatch_portfolio_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch and execute a portfolio analysis tool.
    
    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments
        
    Returns:
        Tool result dict
    """
    dispatch = {
        "get_company_overview": lambda: av.get_company_overview(args.get("symbol")),
        "get_news_sentiment": lambda: av.get_news_sentiment(**args),
        "get_technical_indicator": lambda: av.get_technical(
            symbol=args.get("symbol"),
            indicator=args.get("indicator"),
            interval=args.get("interval", "daily"),
            time_period=args.get("time_period", 14),
        ),
    }

    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"unknown portfolio analysis tool: {tool_name}"}

    try:
        return fn()
    except TypeError as e:
        return {"error": f"bad arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
