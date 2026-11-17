# ResearchAI — Multi-Agent Research Assistant

A Python research workflow with four specialized agents:
1. Search Agent — discovers sources using Tavily.
2. Reader Agent — extracts webpage text and creates evidence notes.
3. Writer Agent — writes a cited Markdown report.
4. Critic Agent — scores the report and can request one revision.

## Architecture

Streamlit UI → LangGraph state graph → Search → Read → Write → Critic
                                              ↑               |
                                              └── revision ──┘

Each node updates a shared typed state. The LLM calls use LangChain Expression Language (LCEL):
`ChatPromptTemplate | ChatGroq | StrOutputParser`.

## 1. Get API keys

- Groq: https://console.groq.com/keys
- Tavily: https://app.tavily.com/

Free tiers have rate/usage limits and can change. The app makes multiple LLM calls per report, so start with 3 sources.

## 2. Run locally (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add your API keys, then:

```powershell
streamlit run app.py
```

## 3. Deploy free on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose your repository, branch `main`, and file `app.py`.
4. Open **Advanced settings → Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
TAVILY_API_KEY = "your_tavily_api_key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

5. Deploy. Never commit `.env` or secrets to GitHub.

## Notes and limitations

- Search results and webpage extraction are best-effort. Some sites block automated requests or render content with JavaScript.
- The Reader falls back to Tavily's search snippet if a page cannot be scraped.
- The Critic score is an LLM-based editorial assessment, not a factuality guarantee.
- Verify important claims against the linked original sources.
- Free API quotas and model availability can change. If a model is unavailable, set `GROQ_MODEL` to an active model in the Groq model catalogue.

## Next engineering upgrades

- Add persistent run history (SQLite/PostgreSQL).
- Add URL/domain allowlists, retries, caching, request timeouts and per-user rate limits.
- Add structured outputs (Pydantic) and claim-to-source verification.
- Add PDF export and background jobs for long reports.
- Add observability/tracing and automated tests.
