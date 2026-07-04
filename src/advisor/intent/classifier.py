"""Intent classifier: Maps user queries to specialized agents.

Uses keyword matching, semantic hints, and user context to determine which
agents should handle a given query. Supports multi-agent routing when appropriate.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set
import re


class IntentType(str, Enum):
    """Enumeration of specialized agent types."""

    RISK_PROFILING = "risk_profiling"
    MARKET_INTELLIGENCE = "market_intelligence"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    GOAL_PLANNING = "goal_planning"
    GENERAL = "general"  # Fallback: educational Q&A only


class IntentClassifier:
    """Classifies user queries to determine agent routing.

    Intent detection is based on:
    1. Keyword matching (high precision)
    2. Query length and complexity
    3. User profile context (if available)
    4. Historical query patterns
    """

    # Keywords associated with each intent type
    INTENT_KEYWORDS: Dict[IntentType, List[str]] = {
        IntentType.RISK_PROFILING: [
            "risk",
            "tolerance",
            "allocation",
            "diversif",
            "asset mix",
            "portfolio split",
            "rebalance",
            "conservative",
            "aggressive",
            "emergency fund",
            "how much should i save",
            "am i too concentrated",
            "what allocation",
        ],
        IntentType.MARKET_INTELLIGENCE: [
            "quote",
            "price",
            "stock",
            "sector",
            "market",
            "trending",
            "performance",
            "fx",
            "currency",
            "exchange rate",
            "how is",
            "what's the price",
            "market today",
            "sector news",
            "how is the market",
            "trading",
        ],
        IntentType.PORTFOLIO_ANALYSIS: [
            "portfolio",
            "holding",
            "position",
            "should i hold",
            "should i sell",
            "analyze",
            "technical",
            "rsi",
            "macd",
            "sma",
            "news sentiment",
            "company overview",
            "fundamentals",
            "valuation",
            "p/e ratio",
            "my positions",
            "my investments",
        ],
        IntentType.GOAL_PLANNING: [
            "retirement",
            "retire",
            "goal",
            "target",
            "how much",
            "how long",
            "save",
            "savings",
            "debt payoff",
            "debt",
            "student loan",
            "mortgage",
            "projection",
            "future",
            "plan",
            "when can i",
            "can i afford",
            "monthly payment",
        ],
    }

    # Patterns that strongly indicate a specific intent
    INTENT_PATTERNS: Dict[IntentType, List[str]] = {
        IntentType.RISK_PROFILING: [
            r"\b(asset allocation|risk tolerance|how.*allocate|diversif)\b",
            r"\b(stocks?|bonds?|cash)\s+(percent|%|split|mix)",
        ],
        IntentType.MARKET_INTELLIGENCE: [
            r"\b(price|quote)\b.*\$(\w+|\d+)",
            r"\b(sector|market)\b.*\b(performance|trend|today)\b",
            r"\b(\w+)\b\s+(up|down|trading|quote)",
        ],
        IntentType.PORTFOLIO_ANALYSIS: [
            r"\b(analyze|review|evaluate)\b.*\b(portfolio|holding|position)\b",
            r"\b(should i|should we|time to)\b.*(sell|buy|hold)",
            r"\b(rsi|macd|sma|technical)\b",
        ],
        IntentType.GOAL_PLANNING: [
            r"\b(retire|retirement)\b\s+(at|by|age)\s+\d+",
            r"\b(save|reach|accumulate)\s+\$(\d+|\d+k|\d+m)",
            r"\b(pay off|payoff)\b.*(debt|loan|mortgage)",
        ],
    }

    def __init__(self, verbose: bool = False):
        """Initialize the classifier.

        Args:
            verbose: If True, log reasoning during classification
        """
        self.verbose = verbose
        self._confidence_scores: Dict[IntentType, float] = {}

    def classify(
        self,
        user_msg: str,
        profile: Optional[Dict[str, Any]] = None,
        top_k: int = 1,
    ) -> List[IntentType]:
        """Classify a user query and return top-K intent types.

        Args:
            user_msg: User's input query
            profile: Optional user profile (age, goals, holdings, etc.)
            top_k: Number of intents to return (1=single, >1=multi-agent)

        Returns:
            List of IntentType in priority order. Always includes GENERAL as fallback.
        """
        self._confidence_scores = {}
        user_msg_lower = user_msg.lower()

        # Score each intent type
        for intent_type in IntentType:
            if intent_type == IntentType.GENERAL:
                continue  # GENERAL is fallback, not scored

            score = self._score_intent(user_msg_lower, intent_type)
            self._confidence_scores[intent_type] = score

            if self.verbose:
                print(f"[Intent] {intent_type.value}: {score:.2f}")

        # Boost scores based on profile context
        if profile:
            self._apply_profile_context(profile)

        # Get top-k intents
        sorted_intents = sorted(
            self._confidence_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Include intents with non-zero score
        result = [intent for intent, score in sorted_intents[:top_k] if score > 0]

        # Always fallback to GENERAL if no intent matched
        if not result:
            result = [IntentType.GENERAL]

        if self.verbose:
            print(f"[Intent] Selected: {[i.value for i in result]}")

        return result

    def _score_intent(self, user_msg_lower: str, intent_type: IntentType) -> float:
        """Score how likely a query matches a given intent.

        Uses keyword matching + pattern matching, normalized to [0, 1].

        Args:
            user_msg_lower: Lowercased user message
            intent_type: Intent type to score

        Returns:
            Confidence score in [0, 1]
        """
        score = 0.0

        # Keyword matching (base score)
        keywords = self.INTENT_KEYWORDS.get(intent_type, [])
        keyword_matches = sum(
            1 for kw in keywords if kw in user_msg_lower
        )
        if keywords:
            score += (keyword_matches / len(keywords)) * 0.6  # 60% from keywords

        # Pattern matching (bonus)
        patterns = self.INTENT_PATTERNS.get(intent_type, [])
        pattern_matches = sum(
            1 for pattern in patterns if re.search(pattern, user_msg_lower)
        )
        if patterns:
            score += (pattern_matches / len(patterns)) * 0.4  # 40% from patterns

        return min(score, 1.0)

    def _apply_profile_context(self, profile: Dict[str, Any]) -> None:
        """Boost intent scores based on user profile context.

        Args:
            profile: User profile dict
        """
        goals = profile.get("goals", [])
        holdings = profile.get("holdings", [])

        # If user has stated goals, boost goal planning
        if goals:
            self._confidence_scores[IntentType.GOAL_PLANNING] = min(
                self._confidence_scores.get(IntentType.GOAL_PLANNING, 0.0) + 0.1, 1.0
            )

        # If user has holdings, boost portfolio analysis
        if holdings:
            self._confidence_scores[IntentType.PORTFOLIO_ANALYSIS] = min(
                self._confidence_scores.get(IntentType.PORTFOLIO_ANALYSIS, 0.0) + 0.1, 1.0
            )

    def explain(self, user_msg: str, profile: Optional[Dict[str, Any]] = None) -> str:
        """Return a human-readable explanation of the classification.

        Args:
            user_msg: User's input query
            profile: Optional user profile

        Returns:
            Explanation string
        """
        intents = self.classify(user_msg, profile, top_k=3)
        scores = [
            (intent, self._confidence_scores.get(intent, 0.0)) for intent in intents
        ]

        lines = ["Intent Classification Explanation:", ""]
        for intent, score in scores:
            lines.append(f"- {intent.value}: {score:.2%}")

        return "\n".join(lines)

    def get_top_intent(
        self, user_msg: str, profile: Optional[Dict[str, Any]] = None
    ) -> IntentType:
        """Get the single highest-confidence intent.

        Args:
            user_msg: User's input query
            profile: Optional user profile

        Returns:
            Highest-scoring IntentType
        """
        intents = self.classify(user_msg, profile, top_k=1)
        return intents[0] if intents else IntentType.GENERAL

    def get_confidence(self, intent_type: IntentType) -> float:
        """Get the confidence score for a specific intent type.

        Must call classify() first.

        Args:
            intent_type: Intent type to query

        Returns:
            Confidence score in [0, 1]
        """
        return self._confidence_scores.get(intent_type, 0.0)
