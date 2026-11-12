# Latency investigation - 12 November 2026

The target is a quick response, aiming for two seconds or less where practical. This is a measured optimization effort, not an unconditional latency guarantee for Community Cloud and an external model provider.

## How the bottlenecks were investigated

A public browser session exercised the actual Streamlit app inside its Community Cloud iframe. Every question starts with Clear Conversation, and the harness waits for the old chat elements to disappear before starting the next timer. It records the first visible answer text, the completed response, backend timing captions and answer length. This prevents an old response from being counted as a fast new one.

Baseline source: `e641e70`. Three successful, fresh-chat requests were measured; global model/cache state was not controlled. The reported nine-second result was not reproduced and is not used as a before/after number.

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
| Separate timing components | The reported total did not show lazy retriever/model loading, callback rendering or end-to-end first-text time separately. | Add initialization, render-callback and pipeline first-text timings; display generation and total separately, with a per-reply breakdown. | Measurement enables diagnosis; adding timers is not itself a speedup. Cross-encoder cold loading remains part of reranking. |

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

Local verification: **153 tests passed, 97.21% package coverage**, a successful browser greeting/scope check, and a warning-free docs build. The final public after comparison remains pending rollout. Do not call these few queries a p95 benchmark or promise a two-second SLA.
