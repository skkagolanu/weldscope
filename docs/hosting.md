# Hosting status and deployment

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
4. Wait for dependency installation and model downloads. Confirm the WeldScope header and a local reply to “hi.” Then submit a welding question, inspect its source passages, ask a follow-up, and clear the conversation.
5. If the deployment remains on the old heading, inspect its source coordinates and build logs before calling it updated. A successful GitHub workflow alone is not proof of the deployed version.

[Streamlit app settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings) · [Changing Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python) · [Changing GitHub coordinates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app)

## Verified locally

Python 3.12, compatible pinned dependencies, real local embeddings and reranking, and retrieval from the 2,165-chunk index. Streamlit tests cover chat/reruns, citations, local replies without a key, and clearing history. Provider request tests use a local transport. Trace correlation is checked using real SDK spans with an in-memory exporter.

Docker execution remains unverified because the local Docker daemon is unavailable.

## Hosted availability checks

The checker inspects the app frame, waits for a chat input, and verifies the new local greeting without sending a provider request. It fails on sleep pages, startup errors, stale UI, and broken greetings, and stores screenshot/text artifacts. `--startup-only` checks startup without sending a greeting.

These checks run daily, after successful CI, and manually. They measure startup and local conversation availability, not provider quotas or answer quality. Community Cloud hibernation still applies; this is not an always-on hosting guarantee.
