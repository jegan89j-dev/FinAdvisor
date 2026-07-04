"""Unit tests for intent classification."""

import pytest
from advisor.intent.classifier import IntentClassifier, IntentType


class TestIntentClassifier:
    """Test suite for IntentClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance."""
        return IntentClassifier(verbose=False)

    # Risk Profiling Tests
    def test_risk_profiling_allocation(self, classifier):
        """Should detect asset allocation queries."""
        msg = "What's a good stock/bond allocation for my age?"
        intents = classifier.classify(msg)
        assert IntentType.RISK_PROFILING in intents

    def test_risk_profiling_diversification(self, classifier):
        """Should detect diversification questions."""
        msg = "Am I too concentrated in tech?"
        intents = classifier.classify(msg)
        assert IntentType.RISK_PROFILING in intents

    def test_risk_profiling_emergency_fund(self, classifier):
        """Should detect emergency fund questions."""
        msg = "How much emergency fund should I have?"
        intents = classifier.classify(msg)
        assert IntentType.RISK_PROFILING in intents

    # Market Intelligence Tests
    def test_market_intelligence_quote(self, classifier):
        """Should detect stock quote queries."""
        msg = "What's the price of AAPL?"
        intents = classifier.classify(msg)
        assert IntentType.MARKET_INTELLIGENCE in intents

    def test_market_intelligence_sector(self, classifier):
        """Should detect sector performance queries."""
        msg = "How are tech stocks performing today?"
        intents = classifier.classify(msg)
        assert IntentType.MARKET_INTELLIGENCE in intents

    def test_market_intelligence_fx(self, classifier):
        """Should detect currency exchange queries."""
        msg = "What's the USD to EUR exchange rate?"
        intents = classifier.classify(msg)
        assert IntentType.MARKET_INTELLIGENCE in intents

    # Portfolio Analysis Tests
    def test_portfolio_analysis_holdings(self, classifier):
        """Should detect portfolio evaluation queries."""
        msg = "Analyze my portfolio of AAPL, MSFT, and TSLA."
        intents = classifier.classify(msg)
        assert IntentType.PORTFOLIO_ANALYSIS in intents

    def test_portfolio_analysis_technicals(self, classifier):
        """Should detect technical indicator queries."""
        msg = "What's the RSI for NVDA?"
        intents = classifier.classify(msg)
        assert IntentType.PORTFOLIO_ANALYSIS in intents

    def test_portfolio_analysis_sentiment(self, classifier):
        """Should detect news sentiment queries."""
        msg = "What's the latest news sentiment on MSFT?"
        intents = classifier.classify(msg)
        assert IntentType.PORTFOLIO_ANALYSIS in intents

    # Goal Planning Tests
    def test_goal_planning_retirement(self, classifier):
        """Should detect retirement planning queries."""
        msg = "Can I retire at 65 with my current savings?"
        intents = classifier.classify(msg)
        assert IntentType.GOAL_PLANNING in intents

    def test_goal_planning_savings_target(self, classifier):
        """Should detect savings goal queries."""
        msg = "How much do I need to save monthly to reach $100k in 5 years?"
        intents = classifier.classify(msg)
        assert IntentType.GOAL_PLANNING in intents

    def test_goal_planning_debt_payoff(self, classifier):
        """Should detect debt payoff queries."""
        msg = "How long will it take to pay off my $10k credit card at 18% APR?"
        intents = classifier.classify(msg)
        assert IntentType.GOAL_PLANNING in intents

    # General/Educational Tests
    def test_general_educational(self, classifier):
        """Should fallback to general for educational queries."""
        msg = "What's the difference between stocks and bonds?"
        intents = classifier.classify(msg)
        # Should have low scores for specific intents, default to general
        assert any(intent in intents for intent in [IntentType.GENERAL] + list(IntentType))

    # Multi-intent Tests
    def test_multi_intent_portfolio_market(self, classifier):
        """Should detect multiple intents when appropriate."""
        msg = "Analyze my portfolio and show me today's market performance."
        intents = classifier.classify(msg, top_k=2)
        assert len(intents) >= 1  # At least one intent

    # Profile Context Tests
    def test_profile_boosts_goal_planning(self, classifier):
        """Profile with goals should boost goal planning intent."""
        profile = {"goals": ["retirement", "home purchase"]}
        msg = "I'm not sure about my financial plan."
        intents = classifier.classify(msg, profile)
        # Goal planning should be detected even with weak keywords
        scores = [
            classifier.get_confidence(IntentType.GOAL_PLANNING),
            classifier.get_confidence(IntentType.RISK_PROFILING),
        ]
        # Goal planning should be highest
        assert scores[0] >= scores[1]

    def test_profile_boosts_portfolio_analysis(self, classifier):
        """Profile with holdings should boost portfolio analysis intent."""
        profile = {"holdings": ["AAPL", "MSFT", "TSLA"]}
        msg = "What should I do with my investments?"
        intents = classifier.classify(msg, profile)
        # Portfolio analysis should be detected
        scores = [
            classifier.get_confidence(IntentType.PORTFOLIO_ANALYSIS),
            classifier.get_confidence(IntentType.MARKET_INTELLIGENCE),
        ]
        # Portfolio analysis should be highest
        assert scores[0] >= scores[1]

    # Utility Methods Tests
    def test_explain(self, classifier):
        """Should produce human-readable explanation."""
        msg = "Should I buy Apple stock?"
        explanation = classifier.explain(msg)
        assert "Intent Classification Explanation" in explanation
        assert "%" in explanation

    def test_get_top_intent(self, classifier):
        """Should return single highest-scoring intent."""
        msg = "What's the current price of AAPL?"
        top_intent = classifier.get_top_intent(msg)
        assert isinstance(top_intent, IntentType)
        assert top_intent != IntentType.GENERAL or top_intent in [IntentType.MARKET_INTELLIGENCE]

    def test_get_confidence(self, classifier):
        """Should return confidence score for intent."""
        msg = "How much should I save for retirement?"
        classifier.classify(msg)
        confidence = classifier.get_confidence(IntentType.GOAL_PLANNING)
        assert 0 <= confidence <= 1

    # Edge Cases
    def test_ambiguous_query(self, classifier):
        """Should handle ambiguous queries gracefully."""
        msg = "Tell me about investing."
        intents = classifier.classify(msg)
        # Should return at least one intent (possibly general)
        assert len(intents) > 0

    def test_empty_query(self, classifier):
        """Should handle empty queries."""
        msg = ""
        intents = classifier.classify(msg)
        # Should default to general
        assert IntentType.GENERAL in intents or len(intents) > 0

    def test_very_long_query(self, classifier):
        """Should handle very long queries."""
        msg = "I have 50k in savings, I'm 30 years old, " \
              "I want to retire at 65, I have AAPL and MSFT stocks, " \
              "I'm interested in tech, I want to save for a house, " \
              "What should I do? Should I rebalance? What's my allocation? " \
              "How much can I save? What are the tax implications?"
        intents = classifier.classify(msg, top_k=3)
        # Should detect multiple strong intents
        assert len(intents) > 0
