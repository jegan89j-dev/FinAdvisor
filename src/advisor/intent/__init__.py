"""Intent classification module for routing queries to specialized agents.

Provides semantic understanding of user queries to determine which specialized
agents (risk profiling, market intelligence, portfolio analysis, goal planning)
should be invoked.
"""

from advisor.intent.classifier import IntentClassifier, IntentType

__all__ = ["IntentClassifier", "IntentType"]
