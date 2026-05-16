# AI Engineer Phase 2.5 — Tutor Notes: AI System Design

A 4-week deep dive into AI/LLM system design. Sits between Phase 2 (building your project) and Phase 3 (interviewing). The goal: turn the system design *thinking* you developed in Phase 2 into system design *performance* — fluent, structured, defensible answers in 45-60 minute whiteboard sessions.

**How this phase is different from the others:**
- Phase 1 = learn fundamentals
- ML Sprint = learn LLM internals
- Phase 2 = build a real project
- **Phase 2.5 = practice answering high-stakes design questions** (this one)
- Phase 3 = interview

You're not building anything new in this phase. You're practicing performance, building a library of canonical designs in your head, and developing the reflexes that separate someone who *can* design systems from someone who can *articulate* designs under pressure.

**Prerequisites:** Phase 1 + ML Sprint + Phase 2 substantially complete. You need real project experience to ground your answers — without it, this phase is theatrical and won't help you.

**How to use this doc:**
- Read all of Week 1 before starting. The framework comes first.
- Weeks 2-4 are practice. Don't skip the reps.
- Pair with your CLAUDE.md Phase 3 (the ML System Design Coach mode is built for this).
- Record yourself. Listening back is painful but the single highest-leverage thing you can do.

---

## Why System Design Performance Is Its Own Skill

You already think well about systems — you've built one. So why the dedicated phase?

Because **building** and **whiteboarding** are different performances:

| Building | Whiteboarding |
|----------|---------------|
| Weeks to design | 5 minutes to clarify scope |
| Iterate based on what breaks | Predict what'll break, fast |
| Test against reality | Defend against an interviewer's edge cases |
| Documentation as you go | Talk while drawing |
| Time to research tools | Name a tool + defend in 10 seconds |
| Tradeoffs emerge through pain | Tradeoffs articulated up front |

The translation between modes isn't automatic. Most engineers who've shipped real systems still blow their first 3-5 mock interviews because they can't compress 8 weeks of thinking into a 5-minute clear answer.

The good news: this is a learnable skill, and 30-40 reps is roughly all it takes.

---

## Week 1: The Framework

This week is theory + structure. By end of week you should be able to recite the 4-step framework in your sleep and have practiced it on 3-4 easy scenarios.

### The 4-step framework

For EVERY system design question, use this exact structure. Memorize it.

**1. Clarify (5-7 minutes)**
- Restate the problem in your own words
- Ask about users, scale, SLAs, budget, constraints
- Confirm you've understood before designing

**2. High-level design (10-15 minutes)**
- Draw boxes and arrows end-to-end
- Cover: input → processing → storage → retrieval → LLM → output → monitoring
- Don't go deep on any one component yet

**3. Deep dive on hard parts (15-20 minutes)**
- The interviewer picks 1-3 components
- Go deep: specific tools, specific tradeoffs, specific numbers
- This is where you signal seniority

**4. Scale, cost, failure modes (5-10 minutes)**
- Where does this break at 10x? 100x?
- How do costs scale?
- What happens when the LLM provider goes down?
- What happens when hallucinations slip through?

If you can't keep these four phases distinct in your head under pressure, you'll bleed into solutions during clarification, miss components in high-level design, never get to the deep dive, and forget failure modes entirely. The framework prevents this.

### Clarifying questions — the cheat sheet

You'll ask versions of these in every interview. Memorize the categories:

**Scope**
- Is this internal-facing or external?
- B2B or B2C?
- How many users / requests per second / documents?
- What's the geographic distribution? Compliance constraints (data residency, GDPR, HIPAA)?

**Quality requirements**
- What's the latency budget? P50, P95, P99?
- What's the cost budget per request?
- What's the accuracy bar? Tolerable error rate?
- What's the failure cost? (Wrong answer = annoying vs lawsuit)

**Inputs and outputs**
- What's the exact input format?
- What's the exact output format?
- Are there hard structural constraints (e.g., must be JSON conforming to schema)?
- What languages? Modalities (text, image, audio)?

**Constraints**
- Can we use any LLM provider? Any model?
- Are there cost constraints?
- Self-hosted requirements?
- Data privacy requirements?

**Timeline**
- MVP or production?
- How quickly does this need to be live?

Pick 4-6 questions per round. You're not interrogating — you're scoping. Most candidates skip this entirely and lose points immediately.

### The reference architecture

Every LLM system design problem reduces to some subset of this canonical architecture. Memorize it. Mental shortcut for any whiteboard:

```
[Client] 
   ↓
[API Gateway / Rate limiter]
   ↓
[Input validation / sanitization]
   ↓
[Cache check] → cached response
   ↓
[Retrieval layer (RAG)]
   ├─ [Embedding model]
   ├─ [Vector DB]
   ├─ [Re-ranker]
   └─ [Filter / metadata layer]
   ↓
[Prompt assembly]
   ↓
[LLM call(s)]
   ├─ [Primary provider]
   ├─ [Fallback provider]
   └─ [Local model fallback]
   ↓
[Tool calling / agent loop]
   ↓
[Output validation]
   ├─ [Schema validation]
   ├─ [Safety classifier]
   └─ [Faithfulness check]
   ↓
[Response to client]

Cross-cutting:
- Logging (every step, every call)
- Eval pipeline (offline + online)
- Monitoring + alerting
- Cost tracking
- A/B testing infra
```

For any system design question, your high-level design is a *subset* of this with arrows in the right places. Don't reinvent. Adapt.

### What to draw

Boxes and arrows. Label everything. Color-code if you have multiple colors:

- **Blue boxes**: data stores
- **Green boxes**: services / computation
- **Yellow boxes**: external APIs (LLM providers)
- **Red arrows**: failure paths / fallbacks
- **Dotted lines**: async / offline paths (eval, training, batch)

If you're doing virtual interviews, use Excalidraw or a simple drawing tool. Don't fumble with complicated software during the round.

### Numbers you should know cold

System design without numbers is hand-waving. Memorize these so you can speak quantitatively:

**LLM latency** (rough P50 estimates for 2026)
- Small models (Haiku, GPT-4o-mini): 500ms-1.5s for short responses
- Medium models (Sonnet, GPT-4o): 1-3s for short responses
- Large models (Opus, GPT-5): 2-5s for short responses
- Streaming first token: 200-800ms typically
- Long-context (>50K tokens): add 2-5x latency

**LLM cost** (per million tokens, ballpark 2026)
- Haiku-class: $1-3 input / $5-15 output
- Sonnet-class: $3-10 input / $15-30 output
- Opus-class: $15-30 input / $75-100 output
- Local quantized 20B: ~$0 marginal, compute-bound

**Embedding latency / cost**
- ~10-50ms per call (batched is much cheaper per item)
- ~$0.02-$0.13 per million tokens

**Vector DB scale rules**
- HNSW index: ~4-10ms search for millions of vectors
- 1M vectors × 1536 dim ≈ 6GB memory
- pgvector vs Qdrant: pgvector slower but simpler; Qdrant faster, more features

**Token-to-word ratios**
- English: ~1.3 tokens per word
- Code: ~1.5 tokens per "word"
- 4K tokens ≈ 3K words ≈ 6 pages
- 200K tokens ≈ 150K words ≈ a novel

When discussing scale, plug in these numbers. *"At 100 requests per second with average 3K input + 500 output tokens at Sonnet pricing, that's $X per day"*. This kind of concrete math signals seniority instantly.

### Week 1 practice scenarios

Do these in order. Self-scored or with the CLAUDE.md Phase 3 ML System Design Coach.

**Scenario 1 (warm-up): "Design a chatbot that answers questions about our company handbook."**
- Easy on purpose. Lets you focus on the framework.
- Goal: hit all 4 phases without missing components.
- Time-box: 30 minutes. Then review what you missed.

**Scenario 2: "Design a system that auto-categorizes customer support tickets."**
- Less obviously LLM. Forces you to think about when LLMs are even the right tool.
- Goal: clarify whether this needs LLMs at all. (Sometimes a classifier is enough!)

**Scenario 3: "Design an internal search system over our company's docs and Slack."**
- Hybrid: semantic + keyword search, multiple data sources.
- Goal: handle multiple ingestion sources, real-time vs batch indexing.

After each, do this debrief:
- What did I miss in clarification?
- Did I jump to solutions too early?
- Did I cover failure modes?
- Were my numbers concrete?

### Week 1 deliverables

By end of Week 1:
- Can recite the 4-step framework cold
- Know the reference architecture from memory
- Have memorized the key numbers
- Practiced 3 warm-up scenarios

---

## Week 2: Core RAG & Search Systems

Now you specialize. RAG systems are the single most common LLM system design question. Get deeply fluent here.

### What "fluent" looks like

For any RAG question, you should reflexively cover:

**Ingestion pipeline:**
- Data sources (where docs come from)
- Pre-processing (cleaning, deduplication)
- Chunking strategy (and why)
- Metadata extraction
- Embedding generation
- Indexing into vector store

**Retrieval:**
- Query understanding (any rewriting? translation? expansion?)
- Embedding the query
- Vector search (which vector DB, which index)
- Metadata filtering
- Hybrid search (dense + sparse)
- Re-ranking (when, with what model)
- Result formatting for the LLM

**Generation:**
- Prompt template
- LLM choice (and why this one)
- Output validation
- Citation handling

**Quality:**
- Retrieval evals (recall@K, MRR)
- Generation evals (faithfulness, completeness, citation accuracy)
- A/B testing infrastructure

**Operations:**
- Data freshness (how often re-index)
- Cache strategy
- Cost per query
- Latency budget per component

If any of these blocks is fuzzy in your head, you'll get caught when the interviewer asks. The strongest candidates have a default answer for each, and they're calibrated to the scenario.

### The key tradeoffs to internalize

Every RAG decision is one of these tradeoffs. Know them cold:

**Chunk size: small vs large**
- Small: precise retrieval, but context fragmented
- Large: more context, but signal diluted, more expensive

**Retrieval: top-K**
- Higher K: better recall, more cost, "lost in the middle"
- Lower K: cheaper, but might miss relevant chunks

**Re-ranking: yes vs no**
- Yes: better precision, adds ~100-300ms latency, adds cost
- No: faster, cheaper, less accurate

**Embedding model: cloud vs local**
- Cloud: better quality, costs money, network latency
- Local: free, private, slightly weaker, depends on hardware

**Index type: HNSW vs IVF vs flat**
- HNSW: fast search, slow indexing, memory-heavy
- IVF: balanced
- Flat: slow at scale, but exact

**Hybrid search: dense+sparse vs dense-only**
- Hybrid: better for keyword-heavy queries, more complex
- Dense-only: simpler, sometimes misses exact-term queries

**Streaming vs synchronous**
- Streaming: better UX, harder to validate output
- Sync: easier validation, worse perceived latency

For each, you should be able to say: "I'd pick X because Y, given the constraints we discussed."

### Week 2 practice scenarios

**Scenario 4: "Design a customer support RAG bot for an e-commerce company."**
- Classic. You'll hit this in some form in 50%+ of interviews.
- Key angles: multi-turn conversation, citing policies, escalation to human, multilingual support, data freshness for inventory/orders.

**Scenario 5: "Design a system for searching across legal contracts at a large law firm."**
- Domain: legal. High precision required. Long documents.
- Key angles: chunking strategy for long legal text, exact-term search (case names, dates), permissions/access control, audit logging.

**Scenario 6: "Design a 'talk to your data' system that lets non-technical users query a database in natural language."**
- Hybrid: LLM + SQL generation + execution + result interpretation.
- Key angles: schema retrieval, SQL safety (read-only? sanitization?), error recovery, result explanation, eval (was the query correct?).

**Scenario 7: "Design a personalized news recommendation system using LLM-enhanced features."**
- Hybrid: traditional recsys + LLM features.
- Key angles: when LLMs add value vs traditional features, cold start, cost at scale (per-user inference is expensive), feedback loops.

**Scenario 8: "Design a semantic search system over 100M product descriptions for an e-commerce site."**
- Scale challenge. Sub-100ms latency requirement.
- Key angles: indexing at scale, sharding, caching, hybrid search, query understanding (typo handling, synonyms), faceted filtering.

After each, list 3 things you'd improve about your answer. Then re-do the scenario the next day and hit those improvements.

### Common RAG curveballs (be ready for these)

The interviewer will throw one of these at you mid-design. Have an answer:

- "What if the user asks something off-topic?" → Topic classifier or LLM-based gate
- "What if retrieval returns no good results?" → Confidence threshold + fallback (admit you don't know)
- "What if the documents are updated daily?" → Incremental indexing, deduplication
- "What if a user's query is ambiguous?" → Clarification turn before retrieval
- "What if a user is from another country/language?" → Multi-language embedding model OR query translation
- "What if the LLM fabricates citations?" → Citation verification step, faithfulness eval
- "What if a user tries to extract sensitive data?" → Access control on retrieved chunks, output filtering
- "How do you handle PII in documents?" → PII detection at ingestion, redaction, access logging

### Week 2 deliverables

By end of Week 2:
- 5 RAG scenarios practiced (Scenarios 4-8)
- Can describe a RAG system end-to-end in 5 minutes without notes
- Have specific defenses for the 8 curveballs above
- Can quote concrete numbers (latency, cost per query) for at least 2 scenarios

---

## Week 3: Agentic Systems

RAG is your bread and butter. Agents are the rarer but higher-signal scenarios. Many AI engineer roles in 2026 specifically want agentic system experience — your Approval Agent project gives you a real story here.

### What makes agentic system design hard

Agents add complexity in three dimensions:

**1. Control flow becomes nondeterministic**
- The LLM decides what to do next
- Termination is harder to reason about
- Loops can run away

**2. Multi-step failures compound**
- Each step has a failure probability
- 5 steps at 95% success each = 77% end-to-end success
- Failures partway through leave partial state

**3. Cost and latency explode**
- A 5-tool-call workflow can be 20x the cost of a single LLM call
- Latency stacks linearly per step

The strongest interview answers acknowledge these explicitly. *"With agents we trade determinism for flexibility — here's how I'd mitigate that..."* sounds senior.

### The agent design framework

When designing an agent system, cover these components:

**The decision-making LLM**
- Which model? (Capability matters for tool selection)
- Temperature? (Usually low for agents — 0-0.3)
- Context window needed?

**Tools**
- What's the full tool inventory?
- Tool boundaries (one purpose each)
- Tool input/output schemas
- Tool failure handling

**Control flow**
- Simple loop, ReAct, plan-and-execute, state machine?
- Termination conditions
- Max iterations / cost ceiling
- When to escalate to human

**State management**
- What state persists across turns?
- How is conversation history compressed?
- Multi-session continuity?

**Safety**
- Action authorization (what can the agent actually do irreversibly?)
- Cost ceiling per session
- Tool-call validation (don't trust LLM-generated inputs blindly)

**Observability**
- Full trace per session (every step)
- Tool-call success rates
- Cost per session distribution
- Eval pipeline

### Patterns to know

**Single-agent with tools**: Simple. ~80% of agentic interviews.

**Hierarchical (orchestrator + specialists)**: One agent decides what to do, dispatches to specialist agents. Useful when sub-tasks need different models or tools.

**Parallel agents**: Multiple agents work concurrently on independent subtasks, results merged. Faster but harder to coordinate.

**Verifier pattern**: Agent does work; separate LLM verifies before committing. Reduces error rate, doubles cost.

**Human-in-the-loop**: Agent pauses at critical decisions for human approval. Common in high-stakes domains.

Know when each applies. Don't over-engineer; default to single-agent unless the problem actively demands more.

### Week 3 practice scenarios

**Scenario 9: "Design an AI agent that triages and assigns support tickets to the right team."**
- Simple agent territory. Few tools (classify, lookup team, assign, notify).
- Key angles: confidence thresholds, escalation when uncertain, training data for hard cases.

**Scenario 10: "Design a research agent that, given a question, browses the web, synthesizes findings, and produces a report."**
- Multi-step, longer-running.
- Key angles: tool design (search, fetch, summarize), citation handling, termination, cost management for long sessions.

**Scenario 11: "Design a code-review agent that runs on every PR."**
- Domain-specific tools (code search, diff analysis, test running).
- Key angles: false positive cost, integration with PR workflow, learning from human overrides.

**Scenario 12: "Design a multi-agent system that helps draft business proposals."**
- Hierarchical pattern: research agent + writer agent + reviewer agent.
- Key angles: coordination, when multi-agent is justified vs single agent, eval at the system level (not just per-agent).

**Scenario 13: "Design a personal assistant agent with access to email, calendar, and Slack."**
- Sensitive: authorization matters.
- Key angles: action authorization (read vs write), audit logging, user confirmation for irreversible actions, privacy.

**Scenario 14: "Design an autonomous data analysis agent — give it a dataset and a question, it explores and answers."**
- Open-ended, hardest to bound.
- Key angles: scope limiting, sandboxed code execution, hallucinated finding prevention, eval for analytical correctness.

### Agent curveballs

- "What if the agent gets stuck in a loop?" → Max iteration limit, repeated-action detection, escape hatch
- "What if a tool times out?" → Retry policy, fallback tool, fail-gracefully strategy
- "What if the agent takes a destructive action by mistake?" → Authorization layer, reversibility check, dry-run mode
- "What if costs spike?" → Per-session budget cap, model downgrade on retry, alerting
- "How do you debug a failed run?" → Full trace per step, replay capability
- "How do you evaluate agent quality?" → Per-step evals + end-to-end evals + human review on sample

### Week 3 deliverables

By end of Week 3:
- 6 agentic scenarios practiced (Scenarios 9-14)
- Can describe the 5 main agent patterns and when each fits
- Can defend agent vs no-agent decisions (your Power Platform edge!)
- Can talk through cost/latency math for a multi-step agent

---

## Week 4: Advanced Topics + Mock Loops

Final week. Harder scenarios, plus full mock-interview loops to build interview stamina.

### Harder scenarios

These add at least one twist that's harder than standard RAG or agent design.

**Scenario 15: "Design an LLM-based fraud detection system for a fintech."**
- High-stakes, low false-positive tolerance, regulatory requirements.
- Key angles: hybrid (rule-based + LLM), explainability, audit trails, calibration, adversarial robustness.

**Scenario 16: "Design a content moderation system at Reddit scale."**
- Massive scale (thousands of posts per second), multilingual, evolving policies.
- Key angles: tiered architecture (fast classifiers → LLM for borderline → human review), latency budget, cost at scale, policy change deployment.

**Scenario 17: "Design an eval / observability platform for LLM applications (meta question)."**
- Test of how deeply you understand the production lifecycle.
- Key angles: trace collection, eval orchestration, dataset management, A/B testing, drift detection, alerting.

**Scenario 18: "Design a fine-tuning pipeline for domain-specific models."**
- Training-side system design (not just inference).
- Key angles: data curation, distillation, eval before/after, deployment, rollback.

**Scenario 19: "Design a real-time translation chatbot with sub-1-second latency."**
- Tight latency constraint forces optimization.
- Key angles: streaming, smaller models, caching, model selection per language pair, quality vs speed.

**Scenario 20: "Design a multi-tenant LLM API gateway for an internal platform."**
- Platform engineering: serving many internal teams.
- Key angles: rate limiting per tenant, cost allocation, model routing, observability per tenant, versioning.

**Scenario 21: "Design an LLM-powered SQL generator over a 500-table data warehouse."**
- Schema complexity is the challenge.
- Key angles: schema retrieval (you can't fit 500 tables in a prompt), query validation, sandboxed execution, result explanation, eval for query correctness.

### Variations on familiar problems

Take Scenarios 4-14 and re-do them with one of these twists. This builds the "now imagine..." reflex:

- "Now do it at 100x scale"
- "Now do it with $500/month budget"
- "Now do it without any hosted LLM (self-hosted only)"
- "Now do it with 99.99% uptime requirement"
- "Now do it with sub-200ms P99 latency"
- "Now do it across 20 languages"
- "Now do it with GDPR + HIPAA compliance"
- "Now imagine LLM costs drop 10x. Does anything change?"

Practice at least 5 of these. The skill of adapting your default design to new constraints is what differentiates senior candidates.

### Full mock loops

In the last 3-4 days, do at least 3 full system design rounds with feedback.

**A full round looks like:**
- Interviewer gives you a problem (45-60 minutes)
- You drive the entire conversation
- Interviewer plays the role honestly — short responses, occasional confusion, probing follow-ups
- You go through clarify → design → deep dive → scale, managing the clock
- Debrief at the end: what worked, what didn't, hire/no-hire signal

**Use your CLAUDE.md Phase 3** in ML System Design Coach mode (or Mock Interviewer mode for stay-in-character rounds). Tell it: *"Give me a fresh scenario, play interviewer for 45 minutes, then debrief."*

After 3-4 full mock rounds, you'll feel the difference. The first round will be rough. The fourth will feel manageable.

### Common debrief findings (watch for these in yourself)

- **Spent too long clarifying** — 5-7 min max, then move on
- **Spent too long on high-level** — 10-15 min then go deeper
- **Never got to scale/failure** — manage the clock, leave 10 min at end
- **Hand-waved on tools** — name specific products + defend
- **No numbers** — quantify everything possible
- **Missed the eval discussion** — biggest miss for AI engineer specifically
- **Got defensive on pushback** — interviewers probe to test depth, not to attack
- **Drew badly** — practice the diagram. Boxes and arrows, clear labels.

### Week 4 deliverables

By end of Week 4:
- All 21 scenarios attempted (some twice)
- At least 3 full mock loops completed
- A "default architecture" you can reach for instantly
- Calibrated sense of timing per phase
- Specific weak areas identified for ongoing drill

---

## What "interview-ready" looks like

By the end of this phase you should be able to:

1. **Walk into any scenario and reach a defensible end-state in 45 minutes.** Not perfect, but coherent, with tradeoffs articulated.

2. **Quote concrete numbers without hesitation.** Latency budgets, cost estimates, scale calculations. *"At 100 RPS, this is ~$X/day."*

3. **Name and defend specific tools.** Not "a vector DB" — *"Qdrant for self-host or pgvector if we're already in Postgres."*

4. **Handle curveballs without panic.** *"Now do it at 100x scale"* doesn't faze you because you've practiced the adaptation reflex.

5. **Cover evals without prompting.** Most candidates need to be asked. Strong candidates volunteer the eval design as part of the answer.

6. **Acknowledge what you don't know.** *"I don't have direct experience with this part — here's how I'd approach learning it"* beats faking expertise.

---

## Interview question bank for system design

These map to Question Bank section 3 (Phase 3) and overlap with the 21 scenarios above. Use them as drill material — pick one at random and answer out loud in 45 minutes.

### Easy / warm-up

1. Design a chatbot for a company FAQ
2. Design a system to categorize incoming emails into folders
3. Design an internal documentation search system
4. Design a meeting summarizer

### Standard RAG

5. Design a customer support RAG bot
6. Design legal contract search
7. Design "talk to your data" / SQL generation
8. Design semantic product search at e-commerce scale
9. Design personalized recommendations with LLM features
10. Design news article search and summarization

### Agentic

11. Design a support ticket triage agent
12. Design a research agent that browses and synthesizes
13. Design a code-review agent
14. Design a multi-agent business proposal drafter
15. Design a personal assistant with email/calendar access
16. Design an autonomous data analysis agent

### Hard / advanced

17. Design an LLM-based fraud detection system
18. Design content moderation at Reddit scale
19. Design an eval / observability platform
20. Design a fine-tuning pipeline
21. Design a real-time translation chatbot
22. Design a multi-tenant LLM API gateway
23. Design a 500-table SQL generator
24. Design a streaming response system with mid-stream tool calls
25. Design a compliance review system for regulated industries

### Curveballs (apply to any of the above)

26. Now do it at 100x scale
27. Now do it on a $500/month budget
28. Now do it without hosted LLMs
29. Now do it with 99.99% uptime
30. Now do it across 20 languages
31. Now do it with strict data residency requirements
32. Now do it with 10x cheaper LLM costs in 2 years
33. Now do it where every decision must be human-auditable

---

## Resources

For deepening (skip the rest):

- **Eugene Yan's "Patterns for LLM Applications"** — best taxonomy of architectures
- **Hamel Husain's blog**, esp. evals + LLM patterns
- **Chip Huyen's "Designing Machine Learning Systems"** book — broader context
- **Anthropic's "Building effective agents" post** — agent patterns from the source
- **Lilian Weng's blog** (lilianweng.github.io) — deeper LLM systems writeups
- **Pragmatic Engineer** newsletter — system design at production companies

Avoid: most YouTube "system design" channels (they're mostly traditional system design, not LLM-specific). The LLM-specific resources above are higher signal.

---

## Closing notes

**Two warnings:**

**1. Don't skip the reps.** Reading about system design is comfortable; doing 21 scenarios is uncomfortable. The uncomfortable thing is the one that works.

**2. Don't memorize answers.** The whole point of practicing many scenarios is that you internalize the *pattern*, not the specific answers. Interviewers ask weird variants; you need to be able to invent a coherent answer on the spot, drawing from your library of building blocks.

**One encouragement:**

You're not designing systems for the first time. You spent 8 weeks of Phase 2 making real architecture decisions on the Approval Agent. Every tradeoff is something you've already lived. This phase converts that lived experience into fluent performance.

The first mock will be rough. By the tenth scenario you'll feel a step-change. By the twentieth, you'll be the candidate that interviewers remember as "really sharp on systems."

Then you start Phase 3 — and you're hard to refuse.
