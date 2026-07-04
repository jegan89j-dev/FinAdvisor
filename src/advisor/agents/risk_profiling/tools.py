"""Risk Profiling Agent tools: asset allocation, risk metrics."""

from typing import Dict, Any, List
from advisor.tools import calculators


def get_risk_tools() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible tool schemas for risk profiling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "asset_allocation",
                "description": "Suggested stocks/bonds/cash split given age and risk tolerance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                        "risk_tolerance": {
                            "type": "string",
                            "enum": ["low", "moderate", "high"],
                        },
                    },
                    "required": ["age", "risk_tolerance"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "emergency_fund",
                "description": "Recommended emergency fund target given monthly expenses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "monthly_expenses": {"type": "number"},
                        "months_target": {"type": "integer", "default": 6},
                    },
                    "required": ["monthly_expenses"],
                },
            },
        },
    ]


def dispatch_risk_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch and execute a risk profiling tool.
    
    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments
        
    Returns:
        Tool result dict
    """
    dispatch = {
        "asset_allocation": lambda: calculators.asset_allocation(
            age=args.get("age"),
            risk_tolerance=args.get("risk_tolerance"),
        ),
        "emergency_fund": lambda: calculators.emergency_fund(
            monthly_expenses=args.get("monthly_expenses"),
            months_target=args.get("months_target", 6),
        ),
    }

    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"unknown risk profiling tool: {tool_name}"}

    try:
        return fn()
    except TypeError as e:
        return {"error": f"bad arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
