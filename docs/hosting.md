# WeldScope hosting and recovery

## Addresses

- Source: https://github.com/skkagolanu/weldscope
- Existing hosted app: https://rag-demo-skullkrak7.streamlit.app/
- Browser health checks: https://github.com/skkagolanu/weldscope/actions/workflows/deploy.yml

The existing subdomain is intentionally retained until a replacement is deployed.
A repository rename does not automatically migrate a Community Cloud app.

## Inspection on 11 November 2026

The public app returned HTTP 200 but rendered “This app has gone to sleep due to
inactivity.” Its public wake button was clicked. Subsequent browser visits showed
the wake-up screen, including “taking longer than normal,” without a chat input.
The hosted app has **not** passed an end-to-end question/answer check. Its runtime
logs and deployment configuration were unavailable in this session, so the cause
of the failed wake-up is unconfirmed.

Local checks passed on Python 3.11:

- The bundled Chroma index loaded 2,165 chunks, and real embedding, hybrid retrieval,
  and cross-encoder reranking completed for a welding-defect question.
- 107 tests passed with 96.71% package coverage. The Streamlit test exercises chat,
  chunk delivery, citations including page one, and clearing history with mocked
  inference. It does not establish that a hosted HuggingFace account can generate.
- Editable package installation and formatting checks passed.
- The real Streamlit server started locally, loaded its components, and rendered
  a usable chat input in Chromium. A placeholder token was used for startup only;
  no external inference was attempted.

Docker execution was not verified because the local Docker daemon was unavailable.

## Recover or migrate the Community Cloud app

An owner must perform these steps in the Streamlit dashboard. The source changes
are ready for Python 3.11 or 3.12; select **3.11** for parity with local verification.

1. Open the app in https://share.streamlit.io/ and inspect its build/runtime logs.
2. Before changing GitHub coordinates, preserve the app settings through the
   dashboard, then delete the existing deployment. Community Cloud documents
   deleting and redeploying as the required repository-rename procedure.
3. Deploy `skkagolanu/weldscope`, branch `main`, entrypoint `app.py` with Python 3.11.
   Retain the existing app subdomain if available, or choose a WeldScope subdomain.
4. Restore the app's existing configuration through Streamlit settings. Leave
   optional Langfuse tracing disabled unless it is configured.
5. Wait for model downloads and index loading. Verify that the chat input appears,
   submit a welding question, confirm an answer and source passages, ask a follow-up,
   and clear the conversation.
6. If the public URL changes, update this page, the README, and the repository
   variable `STREAMLIT_APP_URL`. The browser checker accepts `--url` for ad hoc checks.

If the app is view-only or missing after a previous GitHub owner/repo change,
follow Streamlit's recovery guidance below. Do not delete an unrelated app or
rename the entire GitHub account to repair this one deployment.

- [Streamlit: changing GitHub coordinates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app)
- [Streamlit: app settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings)

## Uptime checks

The Hosted App Check workflow runs daily, after successful CI on main, and on
manual dispatch. It opens a browser and waits for the rendered chat input. A
sleep page, wake-up loop, initialization error, or HTTP error fails the check and
produces screenshot/text artifacts. It never converts failure into a green result.
This checks startup availability, not inference-provider availability or answer quality.

Run it locally after installing Playwright and its Chromium browser:

```bash
python scripts/check_hosted_app.py --url https://rag-demo-skullkrak7.streamlit.app/
```

The checker does not wake sleeping apps or submit paid inference requests.
Community Cloud hibernation is a hosting limitation; these checks detect downtime
but do not provide an always-on service guarantee.
