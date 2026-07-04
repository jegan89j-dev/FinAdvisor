"""Market Intelligence Agent implementation."""

from typing import Optional, Dict, Any, List

from advisor.agents.base import BaseAgent
from advisor.agents.market_intelligence.tools import (
    get_market_tools,
    dispatch_market_tool,
)
from advisor.agents.market_intelligence.prompts import get_market_system_prompt
from advisor.llm.client import chat
from advisor.rag.retrieve import HybridRetriever, format_snippets
from advisor.agent.safety import post_process
import json


class MarketIntelligenceAgent(BaseAgent):
    """Provides market data, trends, and sector analysis.
    
    Responsibilities:
    - Fetch live stock quotes and sector performance
    - Analyze market sentiment and news
    - Provide technical indicators
    - Currency and FX analysis
    """

    def __init__(self):
        super().__init__(
            agent_name="market_intelligence",
            description="Fetches live market data, news sentiment, and sector trends",
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return market intelligence tools."""
        return get_market_tools()

    def get_system_prompt(self, profile: Optional[Dict[str, Any]] = None) -> str:
        """Return market analysis system prompt."""
        return get_market_system_prompt(profile)

    def run(
        self,
        user_msg: str,
        history: List[Dict[str, str]],
        profile: Optional[Dict[str, Any]] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> str:
        """Execute market intelligence agent."""
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

                result = dispatch_market_tool(tc.function.name, args)
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
            "I couldn't complete the market analysis within the step budget. "
            "Try asking about a single ticker or narrower topic."
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
