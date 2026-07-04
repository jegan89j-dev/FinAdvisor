# FinAdvisor — Personalized Financial Advisor using LLM

Capstone project: a personalized financial guidance assistant built on an open-weights LLM (Llama-3.1-8B-Instruct via Hugging Face Inference Providers), grounded with retrieval-augmented generation (RAG) and a **multi-agent architecture** for specialized financial domains.

## What it does

Three pillars wrapped in one chat experience:

1. **Conversational Q&A** — explains financial concepts (IRA vs 401k, dollar-cost averaging, ETFs), grounded in SEC/IRS/FINRA materials.
2. **Investment & portfolio analysis** — pulls live quotes, news sentiment, technical indicators and sector data from Alpha Vantage; reasons over your holdings.
3. **Goal-based planning** — retirement projection, savings target, debt payoff, asset-allocation suggestion via deterministic calculators the agent calls as tools.

Every advisory response carries a disclaimer and source/tool citations.

## Architecture

### High-Level Flow

```
User Query → Intent Classification → Specialized Agents (1 or more)
    ↓
[Risk Profiling | Market Intelligence | Portfolio Analysis | Goal Planning]
    ↓
RAG (Chroma + BM25) + LLM (Llama-3.1-8B) + Tools (Alpha Vantage + Calculators)
    ↓
Safety Guardrails → Response
```

### Multi-Agent System

FinAdvisor uses a **specialized multi-agent architecture** where different financial domains are handled by dedicated agents:

| Agent | Responsibility | Tools | Use Case |
|-------|---|---|---|
| **Risk Profiling** | Asset allocation, risk assessment, emergency funds | `asset_allocation`, `emergency_fund` | "What allocation for my age?" |
| **Market Intelligence** | Live quotes, sector trends, currency rates | `get_stock_quote`, `get_sector_performance`, `get_fx_rate` | "What's AAPL's price?" |
| **Portfolio Analysis** | Holdings evaluation, technicals, news sentiment | `get_company_overview`, `get_news_sentiment`, `get_technical_indicator` | "Analyze my portfolio" |
| **Goal Planning** | Retirement, savings targets, debt payoff | `retirement_projection`, `savings_goal`, `debt_payoff` | "Can I retire at 60?" |

**Intent Classification** (keyword + pattern matching) automatically routes queries to the appropriate agent(s). See [Architecture & Request Pipeline](Architecture%20&%20Request%20Pipeline.md) for details.

## Quickstart

```bash
# 1. Setup
make setup
source .venv/bin/activate
cp .env.example .env
# edit .env: set HF_TOKEN (huggingface.co/settings/tokens) and ALPHA_VANTAGE_KEY (alphavantage.co/support/#api-key)

# 2. Sanity-check the offline test suite (no API keys needed)
make test

# 3. Drop some PDFs / markdown into corpus/sec_education/, corpus/irs_pubs/, etc.
#    Or just rely on the bundled corpus/glossary.md to start.

# 4. Build the RAG index
make index

# 5. Run the app
make run
# opens http://localhost:8501
```

## Project layout

```
financial-advisor-llm/
├── src/advisor/
│   ├── agents/                    ← Specialized agent modules
│   │   ├── __init__.py
│   │   ├── base.py                (BaseAgent interface)
│   │   ├── risk_profiling/        (asset allocation, risk metrics)
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── tools.py
│   │   │   └── prompts.py
│   │   ├── market_intelligence/   (quotes, sectors, FX)
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── tools.py
│   │   │   └── prompts.py
│   │   ├── portfolio_analysis/    (holdings, technicals, sentiment)
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── tools.py
│   │   │   └── prompts.py
│   │   └── goal_planning/         (retirement, savings, debt payoff)
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── tools.py
│   │       └── prompts.py
│   ├── intent/                    ← Intent classifier for routing
│   │   ├── __init__.py
│   │   └── classifier.py          (keyword + pattern matching)
│   ├── config.py                  (Settings ← .env)
│   ├── llm/                       (LLM client + prompts)
│   ├── tools/                     (Generic tool implementations)
│   ├── rag/                       (RAG retrieval + storage)
│   ├── agent/                     (Orchestrator, memory, safety)
│   └── eval/                      (Evaluation framework)
├── app/                           (Streamlit UI + 4 pages)
├── corpus/                        (Source documents for RAG)
├── data/                          (Caches, vector DB, profile DB — gitignored)
├── notebooks/                     (Exploratory notebooks)
├── scripts/                       (CLI entry points)
└── tests/                         (pytest suite)
```

## Common commands

| Command | What it does |
|---|---|
| `make setup` | Create `.venv` and install deps |
| `make test` | Run pytest suite (includes intent classifier + agent tests) |
| `make test-intent` | Test intent classifier only |
| `make test-agents` | Test specialized agents only |
| `make index` | Build the Chroma vector store from `corpus/` |
| `make run` | Launch Streamlit |
| `make eval` | Run Financial PhraseBank eval (50 samples) |
| `make clean` | Remove generated data + caches |

## Configuration

Settings are read from `.env` (see `.env.example`):

### Required
- `HF_TOKEN` — Hugging Face access token (get from https://huggingface.co/settings/tokens)
- `ALPHA_VANTAGE_KEY` — Alpha Vantage API key for market data (free tier OK; 25 req/day, get from https://alphavantage.co/support/#api-key)

### Optional
- `LLM_MODEL_ID` — Model to use; defaults to `meta-llama/Llama-3.1-8B-Instruct`
- `LLM_PROVIDER` — HF Inference Provider backend (`together`, `fireworks-ai`, `hyperbolic`, `groq`, etc.)
- `LLM_TEMPERATURE` — Model temperature; default `0.2` (deterministic)
- `LLM_MAX_TOKENS` — Max tokens per response; default `1024`
- `EMBED_MODEL_ID` — Sentence-transformer for RAG embeddings; default `BAAI/bge-small-en-v1.5`
- `CHROMA_DIR` — Path to Chroma DB; default `./data/chroma`

## How It Works

### Step-by-Step Example

User asks: *"I'm 30 with $50k saved. Can I retire at 65? What's a good allocation?"*

1. **Intent Classification** → Detects `goal_planning` + `risk_profiling` intents
2. **Agent Routing** → Invokes both Goal Planning and Risk Profiling agents
3. **RAG Retrieval** → Finds relevant passages (retirement rules, asset allocation guides)
4. **Agent Execution** (ReAct loop per agent):
   - Goal Planning Agent calls `retirement_projection` tool → projects $X at 65
   - Risk Profiling Agent calls `asset_allocation` tool → recommends 70% stocks / 30% bonds
5. **Safety Post-Process** → Scrubs any "guaranteed" claims, adds disclaimer
6. **Response** → Returns structured guidance with tool citations

See [Architecture & Request Pipeline](Architecture%20&%20Request%20Pipeline.md) for a detailed walkthrough.

### Multi-Agent Execution Flow

```
User Message
    ↓
IntentClassifier.classify(msg, profile, top_k=2)
    ↓
Identifies: [goal_planning, risk_profiling]
    ↓
Orchestrator routes to:
  • GoalPlanningAgent.run(msg, history, profile, retriever)
  • RiskProfilingAgent.run(msg, history, profile, retriever)
    ↓
Each agent independently:
  1. Retrieves RAG context
  2. Builds system prompt (domain-specific)
  3. Runs ReAct loop (LLM calls tools as needed)
  4. Safety post-processes
    ↓
Responses aggregated → User
```

## Key Features

✅ **Multi-Agent Specialization** — Different agents handle different domains for better focus and accuracy

✅ **Intent-Based Routing** — Automatically selects the right agent(s) based on your question using keyword + pattern matching

✅ **Live Market Data** — Alpha Vantage integration for quotes, sentiment, technical indicators

✅ **Deterministic Calculators** — No hallucination on financial math (retirement projections, debt payoff, etc.)

✅ **RAG Grounding** — Responses cite SEC/IRS/FINRA materials from `corpus/`

✅ **Profile-Aware** — Personalizes advice based on age, risk tolerance, goals, holdings

✅ **Safety Guardrails** — Scrubs misleading claims ("guaranteed profit", "risk-free"), enforces disclaimers

✅ **Open-Source LLM** — Uses Llama-3.1 (not OpenAI/Anthropic), routes through Hugging Face Inference Providers

## Project Structure Details

### `src/advisor/agents/` — Specialized Agent Modules

Each specialized agent (risk_profiling, market_intelligence, portfolio_analysis, goal_planning) follows the same structure:

```
agents/agent_name/
├── __init__.py              # Public exports
├── agent.py                 # Agent class extending BaseAgent
├── tools.py                 # Tool schemas + dispatch functions
└── prompts.py               # Domain-specific system prompts
```

**Base Agent Interface** (`base.py`):
```python
class BaseAgent(ABC):
    def get_tools() -> List[Dict]           # Return tool JSON schemas
    def get_system_prompt() -> str          # Return domain-specific prompt
    def run(msg, history, profile) -> str   # Execute agent (ReAct loop)
```

**Example: Risk Profiling Agent**
```python
class RiskProfilingAgent(BaseAgent):
    def get_tools(self) -> List[Dict]:
        return [asset_allocation_schema, emergency_fund_schema]
    
    def get_system_prompt(self, profile) -> str:
        return "You are the Risk Profiling Agent. Your role is to..."
    
    def run(self, user_msg, history, profile, retriever) -> str:
        # 1. Retrieve RAG context
        # 2. Build system prompt with profile
        # 3. Run ReAct loop (LLM + tools)
        # 4. Post-process for safety
        return response
```

### `src/advisor/intent/` — Intent Classification

Maps user queries to agent types using:

- **Keyword Matching (60%)** — Does query contain domain keywords?
  - Example: "allocation" → risk_profiling
  - Example: "price" → market_intelligence
  
- **Regex Pattern Matching (40%)** — Do advanced patterns match?
  - Example: `retire.*age.*65` → goal_planning
  - Example: `stocks.*bonds.*%` → risk_profiling
  
- **Profile Context Boosting (+10%)** — If user has stated goals/holdings, boost relevant agents
  - User profile has "retirement" goal → boost goal_planning score
  - User profile has holdings list → boost portfolio_analysis score

**Supports multi-intent routing** (e.g., query can match 2+ agents):
```python
classifier = IntentClassifier()
intents = classifier.classify(msg, profile, top_k=2)
# Returns: [IntentType.PORTFOLIO_ANALYSIS, IntentType.MARKET_INTELLIGENCE]
```

### `src/advisor/tools/` — Generic Tool Layer

Unchanged from original; provides lowest-level implementations:

- `alpha_vantage.py` — API wrapper for Alpha Vantage market data
- `calculators.py` — Deterministic financial math functions
- `profile.py` — User profile read/write utilities

**Agents import these tools** and wrap them in their own dispatch tables:
```python
# Risk Profiling Agent
from advisor.tools import calculators
dispatch = {
    "asset_allocation": lambda: calculators.asset_allocation(...),
    "emergency_fund": lambda: calculators.emergency_fund(...),
}
```

### Supporting Modules

- **`src/advisor/agent/`** — Orchestrator (routes to agents), memory (SQLite profiles), safety guardrails
- **`src/advisor/rag/`** — Hybrid retrieval (BM25 + dense embeddings via Chroma)
- **`src/advisor/llm/`** — LLM client wrapper, centralized prompts
- **`src/advisor/config.py`** — Unified environment configuration (pydantic-settings)
- **`src/advisor/eval/`** — Evaluation framework (Financial PhraseBank tasks, LLM-as-judge)

## Disclaimers

⚠️ **Educational Use Only**

This software is for educational use only and does not constitute regulated financial advice. Outputs may be incomplete, outdated, or wrong. Always consult a licensed financial advisor before acting on any guidance from this system.

**Not Investment Advice** — We don't make buy/sell recommendations; we analyze and educate.

**No Liability** — Use at your own risk. Neither the developer nor this project assumes responsibility for financial losses.

## Development & Testing

### Running Tests

```bash
# All tests (unit + intent + agent integration)
make test

# Intent classifier only
make test-intent

# Specialized agents only
make test-agents

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_intent_classifier.py -v
```

### Test Coverage

- **Intent Classifier Tests** (`tests/test_intent_classifier.py`)
  - Risk profiling intent detection
  - Market intelligence intent detection
  - Portfolio analysis intent detection
  - Goal planning intent detection
  - Multi-intent routing
  - Profile context boosting
  - Edge cases (ambiguous, empty, very long queries)

- **Agent Tests** (forthcoming)
  - Individual agent execution
  - Tool dispatch correctness
  - ReAct loop termination
  - Safety post-processing

### Adding a New Specialized Agent

1. **Create agent module:**
   ```bash
   mkdir -p src/advisor/agents/my_domain
   touch src/advisor/agents/my_domain/{__init__.py,agent.py,tools.py,prompts.py}
   ```

2. **Implement agent.py** extending `BaseAgent`:
   ```python
   from advisor.agents.base import BaseAgent
   
   class MyDomainAgent(BaseAgent):
       def __init__(self):
           super().__init__("my_domain", "Description of agent role")
       
       def get_tools(self) -> List[Dict]:
           return my_tools()
       
       def get_system_prompt(self, profile) -> str:
           return my_system_prompt(profile)
       
       def run(self, user_msg, history, profile, retriever) -> str:
           # ReAct loop implementation
           pass
   ```

3. **Implement tools.py** with tool schemas and dispatch:
   ```python
   def get_my_tools() -> List[Dict]:
       return [{"type": "function", "function": {...}}, ...]
   
   def dispatch_my_tool(tool_name: str, args: Dict) -> Dict:
       # Tool execution logic
       pass
   ```

4. **Implement prompts.py** with domain-specific prompts:
   ```python
   def get_my_system_prompt(profile) -> str:
       return f"You are the My Domain Agent. ..."
   ```

5. **Update intent classifier** in `src/advisor/intent/classifier.py`:
   ```python
   INTENT_KEYWORDS = {
       IntentType.MY_DOMAIN: ["keyword1", "keyword2", ...],
   }
   ```

6. **Add tests** in `tests/test_my_agent.py`

7. **Update orchestrator** to include new agent type

See [Agent Development Guide](docs/AGENT_GUIDE.md) (forthcoming) for detailed examples.

### Evaluating Agent Quality

```bash
# Run Financial PhraseBank sentiment eval (50 samples)
make eval

# Custom eval on specific agent
python scripts/run_eval.py --agent portfolio_analysis --n 100

# Eval all agents
python scripts/run_eval.py --all
```

## References & Further Reading

### Core Papers

- Yang, Liu, Wang. *FinGPT: Open-Source Financial Large Language Models.* arXiv:2306.06031 (2023) — Inspired this capstone; we built the inference-side multi-agent path.
- Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023 — Agent-loop pattern used by each specialized agent.
- Cormack, Clarke, Buettcher. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009 — Why we use RRF for hybrid RAG retrieval.

### APIs & Libraries

- [Alpha Vantage API](https://www.alphavantage.co/documentation/) — Live market data source
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/) — LLM inference endpoint
- [Chroma](https://docs.trychroma.com/) — Vector DB for RAG storage
- [Streamlit](https://docs.streamlit.io/) — UI framework for dashboard
- [sentence-transformers](https://www.sbert.net/) — Embeddings for dense retrieval

### Documentation

- **[Architecture & Request Pipeline](Architecture%20&%20Request%20Pipeline.md)** — Detailed system walkthrough and component deep-dive
- **[Intent Classification Guide](docs/INTENT_CLASSIFICATION.md)** (forthcoming) — How queries are routed to agents
- **[Agent Development Guide](docs/AGENT_GUIDE.md)** (forthcoming) — Creating and extending agents
- **[Multi-Agent Design Patterns](docs/MULTI_AGENT_DESIGN.md)** (forthcoming) — System architecture principles

## Contributing

Contributions are welcome! Areas of particular interest:

- **New Specialized Agents** — Tax planning, insurance advisor, college savings, etc.
- **Improved Intent Classification** — Better pattern matching, semantic routing
- **Expanded Corpus** — More SEC/IRS/FINRA documents, international markets, etc.
- **Enhanced Safety** — ML-based moderation, refusal classifiers, jailbreak detection
- **Performance** — Caching strategies, parallel agent execution, response streaming
- **Testing** — More comprehensive test coverage, benchmark suites

See [CONTRIBUTING.md](CONTRIBUTING.md) (forthcoming) for guidelines and code standards.

## Future Roadmap

**Short-term (Q3-Q4 2024)**
- [ ] LoRA fine-tuning on Financial PhraseBank for domain-specific accuracy
- [ ] Orchestrator parallel agent execution (invoke multiple agents concurrently)
- [ ] Token streaming to UI for faster response rendering (`st.write_stream`)
- [ ] Expanded corpus with international markets (India: NSE/BSE, SEBI guides)

**Medium-term (2025)**
- [ ] Replace regex safety with ML-based moderation classifier
- [ ] Portfolio backtest tool (simulate allocation against historical data)
- [ ] Multi-tenant support (currently single-user demo)
- [ ] API endpoint (FastAPI) for programmatic access

**Long-term (2025+)**
- [ ] Mobile app (iOS/Android native)
- [ ] Integration with live brokerages (Alpaca, Interactive Brokers, etc.)
- [ ] Agentic workflows (e.g., "set up my Roth IRA and invest $X")
- [ ] Personalized learning (agent improves from feedback)

## License

MIT (Capstone project — educational use only)

## Questions & Support

- 📖 Read [Architecture & Request Pipeline](Architecture%20&%20Request%20Pipeline.md) for detailed explanations
- 🐛 Found a bug? Open an [issue](https://github.com/jegan89j-dev/FinAdvisor/issues)
- 💡 Have a suggestion? Start a [discussion](https://github.com/jegan89j-dev/FinAdvisor/discussions)
- 🤝 Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md) (forthcoming)

---

**Built with ❤️ for financial education and open-source AI.**
