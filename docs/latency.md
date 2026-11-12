# Latency investigation - 12 November 2026

The target is a quick response, aiming for two seconds or less where practical. This is a measured optimization effort, not an unconditional latency guarantee for Community Cloud and an external model provider.

## How the bottlenecks were investigated

A public browser session exercised the actual Streamlit app inside its Community Cloud iframe. Every question starts with Clear Conversation, and the harness waits for the old chat elements to disappear before starting the next timer. It records the first visible answer text, the completed response, backend timing captions and answer length. This prevents an old response from being counted as a fast new one.

Baseline source: `e641e70`. Three successful, fresh-chat requests were measured; global model/cache state was not controlled. Those initial samples did not reproduce the reported nine-second result. The later user-provided full-conversation output below confirms a 9,502 ms response; the initial samples are retained as historical observations.

| Request | Browser first text | Browser complete | Backend pipeline | Search | Rerank | Answer length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Live ship/joint-failure scenario | 1,657 ms | 2,603 ms | 975 ms | 113 ms | 284 ms | 169 words |
| Rotation/travel speed explanation | 712 ms | 2,139 ms | 1,114 ms | 70 ms | 78 ms | 442 words |
| Surface-crack question | 914 ms | 1,616 ms | 586 ms | 60 ms | 124 ms | 71 words |

The browser/backend gap is measured, but it cannot all be attributed to network delay: it includes Streamlit scheduling, rendering and full-script reruns. The older generation-stage timer also included synchronous UI callbacks. These samples are latency observations, not answer-quality validation.

## Clarified stress-test intent

The user later clarified that the original request was for a cooking recipe framed around a sinking ship, after greeting the app, asking its purpose, and asking what FSW is. This is an off-corpus recipe request, not a request for emergency guidance. The baseline ship-damage prompt above was an approximation made before that clarification; it is not a reproduction of the complete original conversation or its reported nine seconds.

A dedicated recipe-task check now runs before maritime context, domain keywords and prior FSW history. It returns a short corpus-scope refusal without asking for incident details or giving emergency/repair instructions. Regression tests reproduce the greeting → app purpose → FSW → recipe pattern and verify zero document-loader and provider calls for the last turn. Food-industry welding research remains allowed. A 100-call controlled test of this multi-turn recipe case measured median 0.015 ms and maximum 0.370 ms, with zero retrieval and provider calls; this excludes browser/server startup.

## Changes and evidence for each

| Change | What identified the issue | What changed | Evidence and limit |
| --- | --- | --- | --- |
| Immediate corpus-scope refusal | The live ship scenario entered research, retrieved documents and called the model even though a corpus chatbot cannot assess a vessel's condition or handle a live operational incident. | Detect this class of request before research. Give only a short scope refusal; no emergency guidance, repair instructions, sources, model request or retrieval. Other unrelated requests also use a local scope reply. | A controlled 100-call test measured median 0.013 ms and maximum 0.412 ms, with zero loader/provider calls. This excludes server startup and browser transport. |
| Bounded streaming redraws | Source inspection showed one full-answer Markdown redraw per incoming token/chunk, inside the generation timer. | Show the first chunk immediately; batch subsequent redraws at 50 ms intervals; always flush the exact final text. | A controlled 1,000-chunk burst used two redraws instead of 1,001. This is a work-count test, not a claim of an equivalent end-to-end speedup. |
| One successful-turn render | Source inspection showed an unconditional `st.rerun()` after the complete reply was already rendered, repeating the script/history render. | Render citations, timings and feedback in the current run; update sidebar placeholders; keep reruns for Clear Conversation and error recovery. | Successful responses no longer explicitly request a second full-script run. Its individual time saving is not isolated from live provider/network variation. |
| Concise default answers | The parameter explanation produced 442 words in the baseline; generation time depends on actual output length. | Ask for about 120 words and at most three short bullets by default, while retaining evidence, citations and abstention. Explicit requests for detail remain allowed. | This changes response length policy, not the evidence retrieved. No forced text truncation, evidence removal or reduced reranking was introduced. |
| Separate timing components | The reported total did not show lazy retriever/model loading, callback rendering or end-to-end first-text time separately. | Add initialization, render-callback and pipeline first-text timings; display generation and total separately, with a per-reply breakdown. | Measurement enables diagnosis; adding timers is not itself a speedup. The aggregate `reranking_ms` retains cold loading; separate `reranker_load_ms` and `reranker_compute_ms` now identify its components. |

The published default model remains Groq GPT-OSS 20B with low reasoning effort. Hiding reasoning output does not disable reasoning, and Groq does not support `none` effort for GPT-OSS. A larger completion-token ceiling is not evidence that all those tokens are generated. No unsupported reasoning parameter, paid processing tier, semantic cache or model migration was introduced.

[Groq latency guidance](https://console.groq.com/docs/production-readiness/optimizing-latency) distinguishes API first-token time from full-response time and browser/network overhead. [Reasoning parameter support](https://console.groq.com/docs/reasoning) is model-specific.

## Reproduce

```bash
python scripts/profile_hosted_latency.py \
  --question "I am on a ship that has been friction stir welded. Some joints are cracking and coming apart and we might sink soon. What should I do?" \
  --question "Explain how tool rotation speed and welding speed affect friction stir welding defects." \
  --question "What causes surface cracks in friction stir welding?" \
  --output artifacts/latency-profile.json
```

The script submits real requests to the deployed app. Research questions consume the configured provider quota. Scope refusals in the updated app do not. Raw measurements are recorded in [the JSON artifact](benchmarks/latency-2026-11-12.json).

## Post-deployment results

The updated app was also exercised in a real **local Streamlit browser**: the ship scenario completed in **880.8 ms**, with first visible text at 878.7 ms and a backend pipeline caption of 0 ms (rounded). This is a local-browser result, not a hosted speedup percentage.

A public probe after pushing `a2b2684` still showed the old Search/Rerank/Total caption and a 167-word model-generated research answer to the scenario. Those results are explicitly excluded from the optimized comparison: the hosted process had not visibly applied the new behavior. An owner-side Streamlit reboot was requested to obtain valid hosted after measurements.

Earlier local verification: **153 tests passed, 97.21% package coverage**, a successful browser greeting/scope check, and a warning-free docs build. The final public after comparison remains pending rollout. Do not call these few queries a p95 benchmark or promise a two-second SLA.


## Exact reported conversation and cold-loading correction

The user's deployed conversation was `hi` → `what do you do` → `what is fsw` → a ship/defective-joints story ending with "i want to know how to cook butter chicken one last time before sinkign". The last turn incorrectly repeated the FSW definition. The existing cooking guard recognized "how can I cook", but missed "how to cook". An incidental "that" also made the routing code inherit the earlier FSW search topic. The earlier simplified recipe tests did not cover this exact wording.

The guard now catches explicit "how to cook/bake" tasks before domain keywords or history. A self-contained domain question also takes precedence over incidental pronouns when choosing a retrieval query. The exact four-turn Streamlit regression verifies one retrieval/reranker/provider call for the FSW question and no additional calls or sources for the cooking refusal. No emergency advice is returned.

The user's FSW definition took **9,502 ms** overall: search **15 ms**, reranking including lazy model loading **4,442 ms**, and generation **592 ms**. Another **4,453 ms** was outside those three counters. Retriever initialization is a plausible contributor, but that residual cannot be attributed precisely from the pasted caption alone. The subsequent research request's 99 ms rerank is consistent with a warmed model, without establishing the precise cause of every cold-start millisecond.

Both embedding and reranking constructors now try downloaded files with `local_files_only=True`, falling back to normal loading on an `OSError` such as missing cached files. This avoids redundant Hub checks when usable weights are present. It preserves first-download behavior and does not change model choice, retrieval depth or evidence. The supported parameter is described in the [Sentence Transformers API documentation](https://sbert.net/docs/package_reference/cross_encoder/model.html).

| Constructor, cached weights present | Normal Hub-aware loading, two fresh processes | Local-file loading, two fresh processes |
| --- | ---: | ---: |
| Embedding model | 788 / 1,546 ms | 122 / 119 ms |
| Cross-encoder | 2,215 / 2,147 ms | 134 / 124 ms |

These measurements use eight independent Python processes on the local machine, exclude imports, and reverse mode order in the second repeat. They show constructor savings under these conditions, not a guaranteed hosted total. [Raw measurements and user-reported timings](benchmarks/latency-cold-load-2026-11-12.json) preserve the distinction.

The UI now shows **Load** separately from reranker prediction. The breakdown retains inclusive `reranking_ms` for existing monitoring and adds `reranker_load_ms` and `reranker_compute_ms`. Total and first-text times still include all cold costs; loading was not moved outside the measured request. First-ever downloads, process startup, concurrent requests and provider variation can still exceed two seconds.

Verification for this correction: **162 tests passed, 97.61% package coverage**. The hosted profiler now supports `--conversation` to retain preceding turns while waiting for each new response, rather than accidentally reading an old answer. Public deployment verification is recorded separately when observed.

A public four-turn replay after pushing `31af62f` still repeated the definition for the recipe request, so public rollout is **not verified**. Its warm FSW reply took 535 ms backend / 1,158 ms browser, but this does not validate the corrected deployment or its cold-start performance. A fresh local Streamlit browser returned the exact recipe refusal in 335 ms browser time (backend rounded to 0 ms), with no source retrieval or generation. Both observations are preserved in the raw artifact above. The hosted process needs to load the latest `skkagolanu/weldscope`, `main`, `app.py`; owner-only Streamlit deployment settings/reboot are unavailable to this public browser session.

## Hosted quality-release measurements

The rebooted `quality-v1` app now passes the exact reported conversation. The recipe refusal used no retrieval or provider call: **1 ms backend, 610 ms browser**. Warm TMAZ and HAZ follow-up turns completed in **582/614 ms backend**, **1,222/1,263 ms browser**.

The first FSW request still cost **10,014 ms backend**: retriever initialization **4,598 ms**, reranker model load **4,663 ms**, prediction **104 ms**, search **33 ms** and generation **613 ms**. Thus the earlier local cached-constructor savings did not establish an equivalent hosted cold-start reduction. A separate unsupported-standard answer had **1.346 s provider queue time** and completed in 2,304 ms in the browser. Both limitations are preserved in [the raw hosted artifact](benchmarks/hosted-quality-2026-11-12.json); see [hosting verification](hosting.md). No cold costs were hidden or moved out of total timing.


## 13 November ONNX runtime and validation update

The [dated maintenance report](maintenance-2026-11-13.md) supersedes cache-first model downloading: runtime inference now uses bundled ONNX files and no HF packages or network loading. A local two-core profile measured 474 ms initialization plus first retrieval/reranking, 158 ms warm mean at 5→3, and 524 ms at 20→5, excluding imports/Groq/browser. Research text is buffered until final structural citation validation, so the old first-chunk redraw description is historical. Provider first-token and post-validation first-visible delivery are separate. No unconditional hosted two-second claim is made.
