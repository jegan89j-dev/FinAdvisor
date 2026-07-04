"""Risk Profiling Agent implementation."""

from typing import Optional, Dict, Any, List

from advisor.agents.base import BaseAgent
from advisor.agents.risk_profiling.tools import get_risk_tools, dispatch_risk_tool
from advisor.agents.risk_profiling.prompts import get_risk_system_prompt
from advisor.llm.client import chat
from advisor.rag.retrieve import HybridRetriever, format_snippets
from advisor.agent.safety import post_process
import json


class RiskProfilingAgent(BaseAgent):
    """Evaluates user risk profile and recommends asset allocation.
    
    Responsibilities:
    - Assess user risk tolerance
    - Recommend portfolio allocation (stocks/bonds/cash)
    - Evaluate portfolio diversification
    - Suggest rebalancing strategies
    """

    def __init__(self):
        super().__init__(
            agent_name="risk_profiling",
            description="Evaluates risk tolerance and recommends asset allocation",
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return risk profiling tools."""
        return get_risk_tools()

    def get_system_prompt(self, profile: Optional[Dict[str, Any]] = None) -> str:
        """Return risk assessment system prompt."""
        return get_risk_system_prompt(profile)

    def run(
        self,
        user_msg: str,
        history: List[Dict[str, str]],
        profile: Optional[Dict[str, Any]] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> str:
        """Execute risk profiling agent."""
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

        # ReAct loop: reason + act until no tool calls
        tools = self.get_tools()
        for step in range(self.max_steps):
            resp = chat(messages, tools=tools)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                # No more tool calls; return response
                return post_process(msg.content or "")

            # Append assistant response with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": self._serialize_tool_calls(tool_calls),
                }
            )

            # Execute tools and append results
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result = dispatch_risk_tool(tc.function.name, args)
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

        # Step budget exceeded
        return post_process(
            "I couldn't complete the risk analysis within the step budget. "
            "Try a narrower question or break it into parts."
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
