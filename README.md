# Single-agent starter

Minimal Python starter using LangChain only.

## Run

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Set `LANGSMITH_API_KEY` to enable tracing.
4. Start the agent: `python agent.py`.

The graph uses a ReAct agent with a UTC time tool. Set `OPENAI_MODEL` to choose another compatible chat model.