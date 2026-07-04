"""Specialized agent modules for FinAdvisor.

Each agent handles a specific domain:
- risk_profiling: Risk assessment and asset allocation
- market_intelligence: Market data, trends, and analysis
- portfolio_analysis: Portfolio evaluation and holdings analysis
- goal_planning: Goal-based financial planning
"""

from advisor.agents.risk_profiling.agent import RiskProfilingAgent
from advisor.agents.market_intelligence.agent import MarketIntelligenceAgent
from advisor.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
from advisor.agents.goal_planning.agent import GoalPlanningAgent

__all__ = [
    "RiskProfilingAgent",
    "MarketIntelligenceAgent",
    "PortfolioAnalysisAgent",
    "GoalPlanningAgent",
]
