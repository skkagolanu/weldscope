# Operational telemetry

Every completed query receives a random request ID. A sanitized JSONL event records status, route, stage/total timing, retrieval counts, generation model, reported token usage and answer-check status. Errors record the exception class and elapsed time, **not exception text**. Questions, answers, credentials and unknown provider-usage fields are excluded. Files rotate at 5 MB with three backups; the same sanitized events are written to the process log.

Default destination: `artifacts/telemetry/events.jsonl`, configurable through `TELEMETRY_PATH`. Setting an empty path disables the sink. Logging failure does not fail a user response. In-memory performance history remains bounded to 200 entries. These counters describe observations; they are not an uptime guarantee.

`cold=true` specifically means this request loaded the cross-encoder. Retriever-initialization and reranker-load timers are reported separately, including shared-resource cache lookup costs where incurred. A new browser session can reuse a warm process and is not automatically a cold-start trial. Model-provider token counts are recorded only when returned; unknown usage remains null.

With Langfuse configured, a parent observation and correlated retrieval/generation spans carry the request ID in metadata; the operational event includes the associated trace ID. With tracing disabled, the request ID supplies **local event correlation**, not evidence that cloud spans exist. Span inputs/outputs contain character counts by default; `TRACE_CONTENT=true` explicitly opts into raw trace content. A real SDK/in-memory-exporter test verifies parent/child correlation without sending user data to Langfuse Cloud.

**Persistence boundary:** JSONL is persistent on the machine's filesystem, and CI uploads sanitized test events with its benchmark artifact. Streamlit Community Cloud disk is ephemeral; these files and process logs do **not** establish durable retention across rebuilds/reboots. A configured external Langfuse account or another retained log sink is required for durable hosted telemetry. The owner has only confirmed Groq configuration; live external trace export/retention has not been verified. No credentials were requested in chat or committed.

The per-response Latency breakdown includes the current session's request ID, token usage and answer-check status. The public regression harness saves those diagnostics for its own known test turns only; it does not expose or download other users' event logs.
