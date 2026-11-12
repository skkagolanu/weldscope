# Hosting status and deployment

**13 November source update:** `quality-v3` removes HF runtime dependencies and model downloads, retains Groq generation, and loads a rebuilt current-PDF index. Hosted verification of this specific revision is recorded in [latest maintenance](maintenance-2026-11-13.md); older observations below do not establish its deployment. Remove any old `EMBEDDING_MODEL`, `RERANKER_MODEL`, or `VECTORSTORE_PATH` overrides to use the bundled defaults.

Public URL: https://weldscope.streamlit.app/

Source: https://github.com/skkagolanu/weldscope

## Observed status on 11 November 2026

The owner recovered the app and reported **HTTP 402 Payment Required** from the HF/Novita inference route. That is a generation billing failure, separate from the public URL.

An anonymous browser confirmed that Community Cloud embeds the running application in a frame. The deployment was still showing the old “FSW Defect Analysis RAG System” heading, rather than the current source's “WeldScope.” The original browser probe only searched the outer page; the checker now inspects app frames as well.

The updated source uses Groq directly for generation. Greetings and app/source explanations work without inference credentials. Live Groq inference remains to be verified after owner configuration. Runtime hosting logs and settings were not available to this maintenance session.

## Deploy the current source

1. In Community Cloud, verify the source is `skkagolanu/weldscope`, branch `main`, entrypoint `app.py`.
2. Use **Python 3.12**. If the existing deployment uses another Python version, Streamlit requires deleting and redeploying it to change the interpreter. Preserve its settings first, and reuse `weldscope` as the subdomain.
3. Add the root-level Groq settings from the [README](https://github.com/skkagolanu/weldscope#configure-groq-generation) in the app's Secrets settings. Add the key privately; no credentials need to be shared in chat.
4. Wait for dependency installation and bundled asset loading. Confirm the WeldScope header and a local reply to “hi.” Then submit a welding question, inspect its source passages, ask a follow-up, and clear the conversation.
5. If the deployment remains on the old heading, inspect its source coordinates and build logs before calling it updated. A successful GitHub workflow alone is not proof of the deployed version.

[Streamlit app settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings) · [Changing Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python) · [Changing GitHub coordinates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app)

## Verified locally

Python 3.12, compatible pinned dependencies, real local embeddings and reranking, and retrieval from the 2,165-chunk index. Streamlit tests cover chat/reruns, citations, local replies without a key, and clearing history. Provider request tests use a local transport. Trace correlation is checked using real SDK spans with an in-memory exporter.

Docker execution remains unverified because the local Docker daemon is unavailable.

## Hosted availability checks

The checker inspects the app frame, waits for a chat input, and verifies the new local greeting without sending a provider request. It fails on sleep pages, startup errors, stale UI, and broken greetings, and stores screenshot/text artifacts. `--startup-only` checks startup without sending a greeting.

These checks run daily, after successful CI, and manually. They measure startup and local conversation availability, not provider quotas or answer quality. Community Cloud hibernation still applies; this is not an always-on hosting guarantee.

## Verified after the 12 November owner reboot

The public `quality-v1` UI passed the seven-turn end-to-end browser regression: greeting, purpose, real Groq FSW answer, the exact ship/butter-chicken request, TMAZ, the HAZ follow-up and an unsupported future-standard question. Source identifiers referred only to current `fsw_doc*.pdf` files, citation ranges were valid, and reported usage was captured for all four research turns. [Dated raw evidence](benchmarks/hosted-quality-2026-11-12.json).

The exact recipe was refused locally with **0 retrieval/generation**, **no token usage**, **1 ms backend** and **610 ms browser**. Warm TMAZ/HAZ replies took **582/614 ms backend**, **1,222/1,263 ms browser**. The future-standard request correctly stated that the evidence contained no 2035 changes. Its then-serving abstention recognizer produced an unnecessary support warning; a focused regression now covers and corrects that wording in source. This final small correction has separate local verification rather than being silently attributed to the earlier hosted run.

**Cold startup is still slow:** the first research request took **10,014 ms backend / 12,004 ms browser**: retriever initialization 4,598 ms, reranker loading 4,663 ms, reranker prediction 104 ms, search 33 ms, generation 613 ms. These measurements confirm model/index initialization dominates this request. They do not isolate CPU deserialization from model-file fetching. The future-standard answer took 1,648 ms backend / 2,304 ms browser, including 1.346 s reported provider queue time. There is no universal two-second claim.

CI for `56b32ce` passed tests, blocking type/error lint, security and the actual retrieval regression gate: [executed workflow](https://github.com/skkagolanu/weldscope/actions/runs/43213547802). Its quality-evidence artifact preserves benchmark output and sanitized test telemetry. Live external Langfuse retention and reliable Community Cloud storage across rebuilds remain unverified.
