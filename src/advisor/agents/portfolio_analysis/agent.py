"""Portfolio Analysis Agent implementation."""

from typing import Optional, Dict, Any, List

from advisor.agents.base import BaseAgent
from advisor.agents.portfolio_analysis.tools import (
    get_portfolio_tools,
    dispatch_portfolio_tool,
)
from advisor.agents.portfolio_analysis.prompts import get_portfolio_system_prompt
from advisor.llm.client import chat
from advisor.rag.retrieve import HybridRetriever, format_snippets
from advisor.agent.safety import post_process
import json


class PortfolioAnalysisAgent(BaseAgent):
    """Analyzes user's portfolio holdings and provides recommendations.
    
    Responsibilities:
    - Evaluate current holdings
    - Analyze portfolio concentration and diversification
    - Provide technical analysis on holdings
    - Assess news sentiment for held positions
    """

    def __init__(self):
        super().__init__(
            agent_name="portfolio_analysis",
            description="Analyzes portfolio holdings, technical indicators, and news sentiment",
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return portfolio analysis tools."""
        return get_portfolio_tools()

    def get_system_prompt(self, profile: Optional[Dict[str, Any]] = None) -> str:
        """Return portfolio evaluation system prompt."""
        return get_portfolio_system_prompt(profile)

    def run(
        self,
        user_msg: str,
        history: List[Dict[str, str]],
        profile: Optional[Dict[str, Any]] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> str:
        """Execute portfolio analysis agent."""
        # Retrieve context if available
        rag_block = ""
        if retriever is not None:
            snippets = retriever.search(user_msg, k=4)
            rag_block = format_snippets(snippets)

        # Build system prompt
        sys = self.get_system_prompt(profile)
        if rag_block:
            sys += f"\n\nRELEVANT CONTEXT (use if helpful, cite when used):\n{rag_block}"

        # Build message list
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys},
            *history,
            {"role": "user", "content": user_msg},
        ]

        # ReAct loop
        tools = self.get_tools()
        for step in range(self.max_steps):
            resp = chat(messages, tools=tools)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                return post_process(msg.content or "")

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": self._serialize_tool_calls(tool_calls),
                }
            )

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result = dispatch_portfolio_tool(tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(result, default=str)[
                            : self.tool_output_limit
                        ],
                    }
                )

        return post_process(
            "I couldn't complete the portfolio analysis within the step budget. "
            "Try focusing on a specific holding or use the Portfolio page."
        )

    @staticmethod
    def _serialize_tool_calls(tool_calls) -> List[Dict[str, Any]]:
        """Serialize tool calls to JSON-compatible format."""
        out = []
        for tc in tool_calls:
            try:
                out.append(tc.model_dump())
            except AttributeError:
                out.append(
                    {
                        "id": getattr(tc, "id", None),
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
        return out
