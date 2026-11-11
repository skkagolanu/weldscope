# Streamlit documentation assessment

Reviewed the official current documentation on 11 November 2026, including the
1.67.0 release notes. The changes are checked against the selected package pins
and tested in the actual installed runtime.

| Area | Assessment and resulting change |
| --- | --- |
| Resource cache | Global resources are shared between users and must be thread-safe. The pipeline and provider client use a session-scoped resource cache. Heavy retrieval/reranking resources remain global, with locks around mutable operations and copies of result metadata. |
| Conversation state | Messages and topic history stay in `st.session_state`, outside global caches. Only complete successful exchanges go to the model. Refreshing the browser can reset the session. |
| Streaming | The callback updates a single `st.empty` placeholder with incoming chunks, preserving final citations and tracing metadata. Streamlit also supports generator-based `st.write_stream`; no completed-string character simulation is used. |
| Provider settings | Root-level TOML secrets become environment variables, which the Pydantic settings loader already reads. Provider credentials are configured in Community Cloud settings; placeholders alone are committed. |
| URL | The custom app URL is edited in Community Cloud app settings. Repository links and browser checks now target `https://weldscope.streamlit.app/`. Changing the URL does not change provider billing. |
| Deployment | Source pushes are picked up automatically; dependency changes cause a full redeploy. Changing the Python interpreter requires deleting and redeploying, the project now requires Python 3.12, including Docker and CI. If the deployed app uses a different interpreter, redeploy it with 3.12. |
| Testing | AppTest covers reruns, chat state, local greetings without inference, citations and clearing history. A Chromium check covers the actual server and local greeting path. |

Sources:

- [Latest release notes](https://docs.streamlit.io/develop/quick-reference/release-notes)
- [Resource caching and scopes](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
- [Session State caveats](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [Replacing a placeholder](https://docs.streamlit.io/develop/api-reference/layout/st.empty)
- [Streaming text](https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream)
- [Secrets management](https://docs.streamlit.io/develop/concepts/connections/secrets-management)
- [Community Cloud app settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings)
- [Automatic source and dependency updates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Changing Python versions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)
