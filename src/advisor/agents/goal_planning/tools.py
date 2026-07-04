"""Goal Planning Agent tools: retirement, savings, debt payoff."""

from typing import Dict, Any, List
from advisor.tools import calculators


def get_goal_tools() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible tool schemas for goal planning."""
    return [
        {
            "type": "function",
            "function": {
                "name": "retirement_projection",
                "description": "Project portfolio future value at retirement.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_age": {"type": "integer"},
                        "retire_age": {"type": "integer"},
                        "current_savings": {"type": "number"},
                        "monthly_contribution": {"type": "number"},
                        "annual_return": {"type": "number", "default": 0.07},
                    },
                    "required": ["current_age", "retire_age", "current_savings", "monthly_contribution"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "savings_goal",
                "description": "Compute required monthly contribution to hit a target value over N years.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "number"},
                        "years": {"type": "integer"},
                        "annual_return": {"type": "number", "default": 0.05},
                    },
                    "required": ["target", "years"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "debt_payoff",
                "description": "Months to pay off a debt and total interest paid given balance, APR, monthly payment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "balance": {"type": "number"},
                        "apr": {"type": "number", "description": "Annual rate as decimal, e.g. 0.18 for 18%"},
                        "monthly_payment": {"type": "number"},
                    },
                    "required": ["balance", "apr", "monthly_payment"],
                },
            },
        },
    ]


def dispatch_goal_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch and execute a goal planning tool.
    
    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments
        
    Returns:
        Tool result dict
    """
    dispatch = {
        "retirement_projection": lambda: calculators.retirement_projection(
            current_age=args.get("current_age"),
            retire_age=args.get("retire_age"),
            current_savings=args.get("current_savings"),
            monthly_contribution=args.get("monthly_contribution"),
            annual_return=args.get("annual_return", 0.07),
        ),
        "savings_goal": lambda: calculators.savings_goal(
            target=args.get("target"),
            years=args.get("years"),
            annual_return=args.get("annual_return", 0.05),
        ),
        "debt_payoff": lambda: calculators.debt_payoff(
            balance=args.get("balance"),
            apr=args.get("apr"),
            monthly_payment=args.get("monthly_payment"),
        ),
    }

    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"unknown goal planning tool: {tool_name}"}

    try:
        return fn()
    except TypeError as e:
        return {"error": f"bad arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
