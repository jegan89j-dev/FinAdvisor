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
FinAdvisor/
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
| `make index` | Build the Chroma vector store from `corpus/` |
| `make run` | Launch Streamlit |
| `make eval` | Run Financial PhraseBank eval (50 samples) |
| `make test` | Run pytest suite |
| `make clean` | Remove generated data + caches |

## Configuration

Settings are read from `.env` (see `.env.example`):

- `HF_TOKEN` — Hugging Face access token
- `ALPHA_VANTAGE_KEY` — Alpha Vantage API key (free tier OK; 25 req/day)
- `LLM_MODEL_ID` — defaults to `meta-llama/Llama-3.1-8B-Instruct`
- `LLM_PROVIDER` — HF Inference Provider (`together`, `fireworks-ai`, `hyperbolic`, ...)

## Disclaimers

This software is for educational use only and does not constitute regulated financial advice. Outputs may be incomplete, outdated, or wrong. Always consult a licensed advisor before acting.

## References

- Yang, Liu, Wang. *FinGPT: Open-Source Financial Large Language Models.* arXiv:2306.06031 (2023).
- Alpha Vantage API: https://www.alphavantage.co/documentation/
- Hugging Face Inference Providers: https://huggingface.co/docs/inference-providers/
