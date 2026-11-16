# Question Bank: AI Agents, Tool Calling and Orchestration

> A revision bank for the agents module: every question below is one a senior AI/ML interviewer might ask. Try answering each one aloud first (aim for about 2 minutes), then open it to compare with the model answer and the key points. Source chapters: [What Is an AI Agent?](../09-agents/01-what-is-an-ai-agent.md), [Tool Calling in Depth](../09-agents/02-tool-calling-in-depth.md), [Agent Design Patterns](../09-agents/03-agent-design-patterns.md), [Orchestration with LangGraph](../09-agents/04-orchestration-with-langgraph-and-state-machines.md), [Multi-Agent Systems](../09-agents/05-multi-agent-systems.md), [Agent Memory](../09-agents/06-agent-memory.md), [Model Context Protocol](../09-agents/07-model-context-protocol-and-tool-ecosystems.md), [Reliability, Security and Prompt Injection](../09-agents/08-agent-reliability-security-and-prompt-injection.md), [Evaluating and Observing Agents](../09-agents/09-evaluating-and-observing-agents.md), [Case Study: Designing a Production Agent](../09-agents/10-case-study-designing-a-production-agent.md).

## Table of contents

- [Agent fundamentals: agents vs workflows](#agent-fundamentals-agents-vs-workflows)
- [Agent design patterns](#agent-design-patterns)
- [Tool calling and execution](#tool-calling-and-execution)
- [Authorization, human approval and prompt injection](#authorization-human-approval-and-prompt-injection)
- [Orchestration with LangGraph and state machines](#orchestration-with-langgraph-and-state-machines)
- [Multi-agent systems](#multi-agent-systems)
- [Agent memory](#agent-memory)
- [Model Context Protocol and tool ecosystems](#model-context-protocol-and-tool-ecosystems)
- [Production reliability](#production-reliability)
- [Evaluation and observability](#evaluation-and-observability)
- [Case study: designing a production agent](#case-study-designing-a-production-agent)
- [Rapid-fire round](#rapid-fire-round)
- [How to practice](#how-to-practice)

## Agent fundamentals: agents vs workflows

<details>
<summary><b>Q1. What is an AI agent, and how is it different from a conventional LLM application?</b></summary>

**Answer:** An AI agent is a system in which an LLM repeatedly chooses the next action, asks the system to run a tool, observes the result and loops until the goal is met or a stopping rule fires. In a conventional LLM application, the engineer's code fixes the control flow and the model transforms inputs at known points (classify, extract, summarize, answer over retrieved context). In an agent, the model decides part of the control flow at run time, inside limits that code enforces. Handing over control flow has consequences: variable cost and latency, many possible paths to test, a new attack surface (text can manipulate which actions are chosen) and the need for traces to debug. A single function call is not yet an agent; it becomes agentic when the model can pick an unplanned next step based on what it observed.

**Key points to mention:**
- Four ingredients: a loop with a variable number of iterations, model-chosen actions, observations fed back into context, and two exits (the model says "done", or code stops it).
- Autonomy is a dial, not a yes/no: single call, chain, router, workflow with LLM steps, single agent, multi-agent.
- Pitfall: equating "agent" with "chatbot" or "uses tools". Many tool-using systems are workflows.
- The model only requests actions; code executes them. This sets up every later security answer.

**Deep dive:** [A precise definition of an AI agent](../09-agents/01-what-is-an-ai-agent.md#a-precise-definition-of-an-ai-agent), [The spectrum](../09-agents/01-what-is-an-ai-agent.md#the-spectrum-from-one-llm-call-to-multi-agent-systems)

</details>

<details>
<summary><b>Q2. When should you use an AI agent instead of a deterministic workflow? When is a deterministic workflow better?</b></summary>

**Answer:** Use an agent when the task is open-ended and the sequence of steps cannot be predicted in advance, when the value per task justifies variable cost, and when there is a feedback signal (tests, query results, a validator) the agent can check progress against. Use a deterministic workflow when the steps are known, when the process is regulated and must be explainable, when volume is high and latency SLAs are tight, or when actions are irreversible. The reliability argument is simple: if each model-decided step succeeds with probability $p$, an $n$-step run succeeds with roughly $p^n$, so $0.95^{10} \approx 0.60$ (illustrative), and every decision moved into code removes a factor. The common production answer is "workflow outside, agent inside": deterministic intake, authorization and approval, with an agentic node only for the genuinely open-ended part.

**Key points to mention:**
- Workflow wins: known procedures (a standard refund), compliance, high volume, tight latency, irreversible actions.
- Agent wins: debugging, research, data investigation, navigating unfamiliar systems, highly variable step counts.
- Decide with evidence: build the simplest version, evaluate on real cases, add autonomy only where it fails.
- The move runs both ways: if traces show the same trajectory on most runs, freeze that path into a workflow.
- Failure mode of each: workflows grow a thicket of special-case branches; agents grow cost tails and hard-to-audit decisions.

**Deep dive:** [Workflows vs agents](../09-agents/01-what-is-an-ai-agent.md#workflows-vs-agents-choosing-the-right-amount-of-autonomy), [Workflow patterns vs agent patterns](../09-agents/03-agent-design-patterns.md#workflow-patterns-vs-agent-patterns)

</details>

<details>
<summary><b>Q3. What are tools, state, memory, planning and observation in an agent?</b></summary>

**Answer:** Tools are functions the model can ask the system to run, each described by a name, a description and a JSON Schema for its arguments; the model only proposes calls, and code validates and executes them. State is the exact, serializable data of the current run: messages, tool results, the current plan, step counters, budgets used and approval flags, owned by the orchestrator so the run can be checkpointed and resumed. Memory is information kept beyond the current context or run and recalled selectively: short-term (trimmed or summarized conversation) and long-term (profile facts and past episodes in a store). Planning is how the next actions are chosen: fixed by code, implicit and step by step (ReAct), or an explicit plan with re-planning. Observation is the result of an action fed back into context; it is the agent's only link to reality, so it enables self-correction but also brings context bloat and prompt-injection risk.

**Key points to mention:**
- Split tools into read-only (safe to retry and parallelize) and side-effecting (need validation, idempotency, approval).
- State answers "where am I in this task"; memory answers "what do I know that is not in front of me".
- Each block has its own failure mode: wrong tool or bad arguments; state lost on a crash; stale or leaked memories; stale plans and loops.
- Complete the picture with the model itself and the stopping criteria.

**Deep dive:** [The core components of an agent](../09-agents/01-what-is-an-ai-agent.md#the-core-components-of-an-agent), [The four building blocks](../09-agents/03-agent-design-patterns.md#the-four-building-blocks-tools-state-memory-and-planning)

</details>

<details>
<summary><b>Q4. Does every LLM application need an agent?</b></summary>

**Answer:** No. Most production value comes from single calls, RAG and workflows with LLM steps; agents add cost, latency, variance and attack surface, so they are justified only when the task genuinely requires the model to choose a variable sequence of actions. Classification, extraction and summarization are single calls; document Q&A is RAG; multi-stage processing with known stages is a workflow. For example, "refund a damaged item" looks agentic but is a known procedure: a workflow needs about two LLM calls and has rules an auditor can read. Climb the ladder of autonomy only when evaluations show the simpler design failing.

**Key points to mention:**
- Map task types to the simplest sufficient architecture before naming any framework.
- Signal to switch from workflow to agent: a growing share of cases hitting the fallback branch, and those cases needing varied investigation steps.
- An agent must beat a workflow baseline on success, cost per task and p95 latency on the same eval set.

**Deep dive:** [Does every LLM application need an agent?](../09-agents/01-what-is-an-ai-agent.md#does-every-llm-application-need-an-agent)

</details>

<details>
<summary><b>Q5. What makes an agent non-deterministic?</b></summary>

**Answer:** Several sources stack up: sampling randomness in the LLM; inference numerics that vary even at temperature 0 (floating-point addition is not associative, and results depend on how the server batches requests under load); tool outputs that change (search results, data updates, intermittent timeouts); long contexts where small differences shift attention; and silent updates to models, prompts or tool schemas. The agent loop amplifies all of these, because every output becomes the next input, so one different token early can lead to a completely different trajectory. Mitigations are pinned model versions, low temperature for tool selection, consistency-critical decisions moved into code, and record-and-replay of traces for debugging. You measure it with repeated runs per task, reporting consistency (pass^k) rather than one run per test.

**Key points to mention:**
- Temperature 0 removes sampling randomness only, not numeric or environmental variance.
- Models use mid-context information less reliably ("Lost in the Middle", Liu et al., 2023).
- Snapshot or mock environments in tests so failures are reproducible.
- Follow-up: "how many runs per task?" Several, with tasks (not runs) as the unit of resampling.

**Deep dive:** [Why agents are non-deterministic](../09-agents/01-what-is-an-ai-agent.md#why-agents-are-non-deterministic), [Sources of non-determinism](../09-agents/09-evaluating-and-observing-agents.md#sources-of-non-determinism)

</details>

<details>
<summary><b>Q6. How do you prevent an agent from entering an infinite loop?</b></summary>

**Answer:** With layered limits enforced in orchestration code, never in the prompt: a hard maximum number of iterations; repeated-action detection on a canonical hash of tool name plus normalized arguments; progress checks every few steps (new facts found, fewer failing tests, plan items closed); token, cost and tool-call budgets; per-tool and whole-run timeouts; and a clear finish signal checked by a validator. On the first repeat, inject a note ("you already called this with these arguments, the result was X"); on persistent repetition, stop. Every stop returns a structured status such as `loop`, `max_steps`, `timeout` or `token_budget`, plus the trace, and escalates to a human or the user. Loops usually come from failing tools being retried, the model not realizing it is done, oscillation between states or vague goals, so fix the root cause too (clearer error messages, tool descriptions, missing data).

**Key points to mention:**
- "Do not loop" in the prompt is not a control: a looping model is already not following instructions.
- Set `max_steps` from data, slightly above the p99 step count of successful runs.
- Monitor how often each limit fires; frequent hits are a product signal.
- Loops can live one level up: a scheduler that auto-relaunches failed runs. Cap job-level retries and keep side effects idempotent.

**Deep dive:** [Preventing infinite loops](../09-agents/01-what-is-an-ai-agent.md#preventing-infinite-loops), [Hard limits: step, token, time and cost budgets](../09-agents/08-agent-reliability-security-and-prompt-injection.md#hard-limits-step-token-time-and-cost-budgets)

</details>

<details>
<summary><b>Q7. How do you control token usage and execution time?</b></summary>

**Answer:** Each call resends the history, so input tokens for an $n$-step run grow roughly as $nP + s\,n(n-1)/2$, where $P$ is the fixed prompt and $s$ the tokens added per step: with $P = 1{,}500$ and $s = 400$, 5 steps cost 11,500 input tokens but 20 steps cost 106,000 (illustrative). The levers are to reduce $n$, $s$ and $P$: cap the run with step, token, cost and time budgets; shape observations (project fields, paginate, truncate, summarize old turns); route sub-steps such as extraction and classification to smaller models; cache stable prompt prefixes and read-only tool results; run independent tool calls in parallel; and move predictable steps into code. Cutting $s$ from 400 to 150 takes the 20-step run from 106,000 to 58,500 tokens. For time, latency is roughly the sum of LLM and tool time per step, so parallel calls, per-call timeouts, an overall deadline and streaming the final answer matter most.

**Key points to mention:**
- Enforce before spending: estimate the next prompt and set the maximum-output-tokens parameter on every call.
- Degrade gracefully: on budget exhaustion, one last call with tools disabled, or a partial result with a clear status.
- Track p50 and p95 cost and latency per successful task; budgets exist to cut the heavy tail.
- Tool definitions are a hidden fixed cost on every call; send only the relevant ones.

**Deep dive:** [Controlling token usage and execution time](../09-agents/01-what-is-an-ai-agent.md#controlling-token-usage-and-execution-time), [The hidden cost of ReAct](../09-agents/03-agent-design-patterns.md#the-hidden-cost-of-react-context-grows-every-step)

</details>

## Agent design patterns

<details>
<summary><b>Q8. Explain ReAct. Why does interleaving reasoning and acting help, and what are its failure modes?</b></summary>

**Answer:** ReAct (Yao et al., 2022) has the model alternate between a thought, an action (a tool call) and an observation from the environment, looping until it gives a final answer or a budget is hit. Reasoning makes tool choice deliberate; acting grounds the reasoning in fresh facts, which reduces hallucination compared with reasoning alone and lets the agent react to surprises. Example: "Can I return order A1234?" leads to `lookup_order` (delivered 2026-10-02), then `get_refund_policy` (30 days), then the answer with the deadline. The original paper parsed text such as "Action: search[query]"; modern systems use native tool calling, and the thought may be hidden reasoning tokens. Failure modes are loops, premature answers, wrong tools or arguments, drift on long tasks, being derailed by bad observations, injection through tool outputs and context that grows every step.

**Key points to mention:**
- What the paper showed: clearest wins on interactive tasks (ALFWorld, WebShop) and on grounding; on HotpotQA the best results combined ReAct with chain-of-thought self-consistency.
- Cost: $nP + s\,n(n-1)/2$ input tokens; with $P = 2{,}000$, $s = 500$, $n = 6$, that is 19,500 tokens.
- Controls: repeat detection, schema validation with actionable errors, compact observations, budgets.
- Alternatives: plan-and-execute and reflection.

**Deep dive:** [Agent pattern 1: ReAct](../09-agents/03-agent-design-patterns.md#agent-pattern-1-react-reason-and-act-interleaved), [The agent loop: ReAct in pseudocode](../09-agents/01-what-is-an-ai-agent.md#the-agent-loop-react-in-pseudocode)

</details>

<details>
<summary><b>Q9. When would you choose plan-and-execute over ReAct?</b></summary>

**Answer:** Choose plan-and-execute when the task is longer, has a mostly predictable structure, benefits from parallel steps, or needs a human to approve the intended actions before anything happens. A strong model plans once, cheaper models or plain code execute the steps, and a re-planner revises the remaining steps when a step fails or returns something surprising. Benefits are fewer strong-model calls, parallelism through a dependency graph (the LLMCompiler idea), better coherence on long tasks and an inspectable plan for approval and debugging. The risk is stale plans, so re-plan on errors and surprises with a re-plan budget (for example 2) and escalate when it is exceeded. ReAct stays better for short exploratory tasks in which each step depends heavily on the last observation.

**Key points to mention:**
- Hybrid: plan high-level steps, run each as a small bounded ReAct loop with a narrow tool set.
- An explicit plan is structured output that can be validated before execution.
- Related work: Plan-and-Solve, ReWOO, LLMCompiler.

**Deep dive:** [Agent pattern 2: plan-and-execute](../09-agents/03-agent-design-patterns.md#agent-pattern-2-plan-and-execute-with-re-planning), [ReAct vs plan-and-execute](../09-agents/03-agent-design-patterns.md#react-vs-plan-and-execute)

</details>

<details>
<summary><b>Q10. Does self-reflection actually improve LLM agents? When does it help and when does it hurt?</b></summary>

**Answer:** It helps when the critique is grounded in an external signal: failing tests, a SQL error, a schema violation or a retrieved document that contradicts a claim. Pure intrinsic self-correction, where the model rereads its own answer with no new information, is unreliable on reasoning and can turn right answers into wrong ones (Huang et al., 2023). Grounded variants such as CRITIC (tool-checked critique) and Reflexion (verbal lessons stored after externally detected failures) are the ones that show gains. Each reflection round adds 2 to 3 calls, so cap rounds at 1 to 3 and stop as soon as the check passes. Good fits are coding agents with tests, text-to-SQL with execution and extraction with schema validation; open-ended writing with no ground truth mostly gets style polishing.

**Key points to mention:**
- "Where does the feedback come from?" is the key question.
- A separate judge prompt with an explicit rubric beats "are you sure?", and should be calibrated against human labels.
- Always return an explicit status ("failed after max rounds") instead of silently shipping the last draft.

**Deep dive:** [Agent pattern 3: reflection and self-critique](../09-agents/03-agent-design-patterns.md#agent-pattern-3-reflection-and-self-critique)

</details>

<details>
<summary><b>Q11. How would you design and evaluate a router in front of several LLM handlers?</b></summary>

**Answer:** Layer it: deterministic rules for obvious and safety-critical cases, then an embedding classifier, trained classifier or LLM classifier with structured output, then a fallback to a general handler or a human when confidence is low. Keep an explicit "other" route so novel intents are not forced into a known bucket. The cost model is $\mathbb{E}[C] = c_{\text{router}} + p\,c_{\text{small}} + (1-p)\,c_{\text{large}}$; with illustrative prices $0.0002 + 0.7 \times 0.0005 + 0.3 \times 0.01 = 0.00355$ versus 0.01 for always using the large model, about 64.5% saved. Evaluate on a labeled set with per-route precision and recall and a confusion matrix, then measure the end-to-end effect on quality and cost. Misroutes are asymmetric: sending a hard query to a small model costs quality, sending an easy one to the large model only costs money, so tune the threshold on that trade-off.

**Key points to mention:**
- LLM-stated confidence is poorly calibrated; use log-probabilities or a calibrated classifier for thresholds.
- Monitor route distribution drift; a product launch creates intents the router has never seen.
- Routing is a workflow pattern, not an agent pattern: code still owns the control flow.

**Deep dive:** [Workflow pattern 2: routing](../09-agents/03-agent-design-patterns.md#workflow-pattern-2-routing)

</details>

<details>
<summary><b>Q12. Why do long-horizon agents fail more often, and what do you do about it?</b></summary>

**Answer:** Errors compound: if each step is right with probability $p$, an $n$-step task is fully right with roughly $p^n$, so $0.95^{10} \approx 0.60$ while $0.99^{10} \approx 0.90$ (illustrative). At the same time the context grows, so later decisions are made with a longer, noisier history in which mid-context facts are used less reliably. The fixes are fewer model-decided steps, verification checkpoints where code checks progress (tests pass, validator accepts, postconditions hold), and decomposition into sub-tasks with their own budgets. Keep context focused by summarizing completed sub-tasks and keeping the goal and key facts in structured state rather than buried in the transcript. Small per-step gains matter a lot because they are raised to the power $n$.

**Key points to mention:**
- Validation plus retry raises effective per-step reliability: $1 - (1-p)^k$, but only for independent, detectable errors.
- Deterministic steps for the predictable parts shrink $n$.
- Explicit plans and sub-agents with isolated context are tools here, with their own overhead.

**Deep dive:** [Reliability compounds multiplicatively](../09-agents/03-agent-design-patterns.md#reliability-compounds-multiplicatively), [Why small per-step errors become big task errors](../09-agents/08-agent-reliability-security-and-prompt-injection.md#why-small-per-step-errors-become-big-task-errors)

</details>

<details>
<summary><b>Q13. What is a code-as-action agent, and how would you sandbox LLM-generated code?</b></summary>

**Answer:** Instead of one JSON tool call per step, the model writes a short program that can call several tools, loop, branch and compute exactly; a sandbox runs it and the output becomes the observation. It often needs fewer LLM round trips and handles computation well (CodeAct, Wang et al., 2024), but it executes untrusted code. Sandbox it in a fresh, ephemeral, isolated environment, ideally a microVM (such as Firecracker) or a user-space kernel (gVisor), at minimum a hardened container: no network by default, no secrets, a read-only root filesystem with a small scratch area, a non-root user, dropped capabilities, CPU, memory, process and wall-clock limits, and output size caps. Destroy the sandbox after each task or tenant session. If network access is needed, route it through an egress proxy with a domain allowlist and logging.

**Key points to mention:**
- In-process restrictions or string filters on code are not a security boundary.
- Containers share the host kernel; prefer stronger isolation for multi-tenant workloads.
- Good fits: data analysis, reports, developer tooling. Poor fits: customer-facing actions with strict per-action permissions.
- Treat sandbox output as untrusted: validate and truncate it before it enters the context.

**Deep dive:** [Agent pattern 5: code-as-action agents](../09-agents/03-agent-design-patterns.md#agent-pattern-5-code-as-action-agents), [Sandboxing code execution](../09-agents/08-agent-reliability-security-and-prompt-injection.md#sandboxing-code-execution)

</details>

## Tool calling and execution

<details>
<summary><b>Q14. Explain function calling and tool execution.</b></summary>

**Answer:** You send the model tool definitions (name, description, JSON Schema for parameters), and a model fine-tuned for tool use may answer with a tool call: a tool name plus JSON arguments in a dedicated field. That is only a proposal. The application parses it, validates the arguments, authorizes the action for the authenticated user, optionally asks a human, executes the tool with timeouts and idempotency, and sends back a tool result linked by the call id. The model then continues, possibly calling more tools, until it returns a final answer, and the loop is bounded by step and time limits. Native function calling (fine-tuned format, server-side parsing, optional strict schemas) is far more reliable than asking for JSON in prose and parsing it with regex, but it ties you to a vendor's message format, so keep an abstraction layer over the loop.

**Key points to mention:**
- Every call id needs exactly one result message, including rejected or denied calls.
- Tool choice settings (`auto`, `none`, `required`, a named tool) steer whether and which tool is called.
- `arguments` arrives as a JSON string: parse, then validate.
- Pitfall: treating the model's text "I've done X" as execution.

**Deep dive:** [Recap: the tool-calling contract](../09-agents/02-tool-calling-in-depth.md#recap-the-tool-calling-contract), [The full execution trace](../09-agents/02-tool-calling-in-depth.md#the-full-execution-trace-message-by-message), [Structured Outputs and Function Calling](../05-llms/05-structured-outputs-and-function-calling.md)

</details>

<details>
<summary><b>Q15. How do you make tool calling reliable and secure?</b></summary>

**Answer:** Treat the model as an untrusted planner and build a deterministic execution layer around it. For reliability: narrow, well-described tools with typed schemas; strict validation with structured errors and capped self-correction; per-call timeouts and a turn deadline; retries with backoff and jitter only for idempotent operations; idempotency keys and a dedup table for writes; reconciliation of unknown outcomes; sagas with compensation; bounded loops and good tracing. For security: identity from the session, per-user tool filtering, a policy engine on every call, delegated least-privilege credentials, human approval for high-risk actions and layered prompt-injection defenses that cap the blast radius. Per-step reliability compounds ($0.98^{10} \approx 0.82$), so fewer, sturdier steps help.

**Key points to mention:**
- Keep tool outputs small (projection, pagination, handles) for cost, quality and a smaller injection surface.
- Test with fault injection: a timeout after the side effect committed must not cause a second effect.
- Measure end-to-end task success, hallucinated-action rate, double effects (target zero) and harmful actions actually executed under injection tests.

**Deep dive:** [Putting it together: a production-style tool executor](../09-agents/02-tool-calling-in-depth.md#putting-it-together-a-production-style-tool-executor)

</details>

<details>
<summary><b>Q16. How does the model select a tool?</b></summary>

**Answer:** By next-token prediction over a context that contains the tool names, descriptions and schemas, rendered into the prompt by the chat template. Models are fine-tuned on tool-use conversations, so they learn to map a request to the tool whose description fits best and to fill arguments from the conversation. Selection is probabilistic and only as good as the descriptions and the tool set you expose. Constrained decoding can guarantee a valid tool name by masking invalid tokens, but not the right one. Accuracy drops with many or overlapping tools, so use distinct names, "when not to use" guidance in descriptions, merged duplicates, and tool retrieval or a router that sends only the relevant few.

**Key points to mention:**
- Tool choice settings and which tools you send are the levers that constrain selection.
- Security angle: a tool not sent to the model cannot be selected, so filter by user permission first.
- Tool definitions cost input tokens on every call.
- Measure tool-selection accuracy on a labeled eval set.

**Deep dive:** [How a model selects a tool](../09-agents/02-tool-calling-in-depth.md#how-a-model-selects-a-tool)

</details>

<details>
<summary><b>Q17. Your agent has 150 tools (or must reach 300 internal APIs) and keeps picking the wrong one. What do you do?</b></summary>

**Answer:** Reduce what the model sees per request. Group raw APIs into a smaller set of business-level tools with clear, namespaced names (`billing_get_invoice`), then route the request to a domain and expose only that domain's tools, or retrieve the top-k tools by embedding similarity of their descriptions. Clean up the tools themselves: merge overlaps, give distinct names and descriptions that say when not to use them, and return actionable validation errors. Tool retrieval has its own recall problem: if the right tool is not retrieved, the agent cannot use it, so measure tool-retrieval recall@k on logged queries and always include a few core tools. Wrap every tool in the same executor (validation, policy, idempotency, timeouts) and compare tool-selection accuracy before and after the change.

**Key points to mention:**
- Context math: 200 tools at 150 tokens each is 30,000 tokens of definitions on every call (illustrative).
- Hierarchical design: a top-level agent with domain tools backed by sub-agents or workflows, with multi-agent overhead.
- A protocol such as MCP standardizes discovery but does not replace your authorization layer.
- Per-tool invalid-argument rates point to confusing definitions.

**Deep dive:** [When selection goes wrong, and tool retrieval](../09-agents/02-tool-calling-in-depth.md#when-selection-goes-wrong-and-tool-retrieval), [Tool-set design and the "giant tool list" problem](../09-agents/03-agent-design-patterns.md#tool-set-design-and-the-giant-tool-list-problem)

</details>

<details>
<summary><b>Q18. Who validates tool arguments, and how do you validate them?</b></summary>

**Answer:** The application does, always, because arguments are untrusted input that can be malformed, wrong or injected. Validate in layers: the tool exists and is enabled for this user; the JSON parses; schema and types pass a strict validator such as Pydantic in strict mode; semantic checks hold (ranges, formats, units); business rules hold against fresh state; the user is authorized on this specific resource; and a risk gate decides whether approval is needed. Failures go back to the model as structured errors it can act on. Schema constraints in the tool definition and strict-mode decoding help the model produce the right shape, but they guarantee shape, not truth or permission. Re-check business rules at execution time inside the transaction, because state can change between check and use, especially after an approval delay.

**Key points to mention:**
- Decide coercion per field: integer cents for money, canonical ids, real booleans, ISO 8601 times with zones.
- Never auto-fix ambiguous risky values such as "1.200".
- Trade-off: strict typing costs a self-correction turn; lax coercion can silently produce the wrong amount.
- Measure invalid-argument rate and self-correction success per tool; spikes after a model change point to schema or description issues.

**Deep dive:** [Validating arguments: who and how](../09-agents/02-tool-calling-in-depth.md#validating-arguments-who-and-how)

</details>

<details>
<summary><b>Q19. How do you handle invalid arguments, timeouts and retries?</b></summary>

**Answer:** Invalid arguments become a structured tool result (field, problem, a valid example, `retryable`) so the model can self-correct, with a cap of 2 or 3 attempts before asking the user or escalating. Every tool call gets a timeout and the turn gets a deadline. Transient failures (timeouts, 429, 503) are retried by the executor, not the model, with exponential backoff and jitter, and only for idempotent operations or writes protected by an idempotency key the backend honors; 400s and 403s are never blindly retried. A timed-out write is an unknown outcome, because cancelling the local wait does not cancel the remote operation, so retry it only with the same key or reconcile first. Worst-case latency is $A \cdot T$ plus the backoff waits: 3 attempts with 3 s timeouts and a 0.2 s base is about 9.6 s, which is why interactive tools use fewer, faster attempts.

**Key points to mention:**
- Retry amplification: $A^L$ requests across $L$ retrying layers (3 layers of 3 attempts is 27). Retry in one layer, with retry budgets and circuit breakers.
- Duplicates are handled with keys reused across attempts plus database invariants.
- Measure timeout rate, p95 and p99 per tool, retries per call and unknown-outcome rate.

**Deep dive:** [Handling invalid arguments](../09-agents/02-tool-calling-in-depth.md#handling-invalid-arguments-structured-errors-and-capped-retries), [Timeouts and retries](../09-agents/02-tool-calling-in-depth.md#timeouts-and-retries)

</details>

<details>
<summary><b>Q20. How do you prevent duplicate payments or other irreversible side effects?</b></summary>

**Answer:** Accept that exactly-once delivery is impossible over a network and design for at-least-once execution plus idempotency, which together give exactly-once effects. Each logical action gets an idempotency key derived from the action (tenant, user, conversation, tool and canonical arguments, or a business key such as the invoice id), claimed atomically in a dedup table with a unique constraint before executing, passed to the payment provider, and reused on every retry; a duplicate returns the stored result. Back it with a business invariant enforced in a transaction (amount already paid plus this amount must not exceed the invoice total). Put irreversible actions behind a two-phase prepare-and-confirm step with human approval, and order multi-step work so the irreversible step runs last. Duplicates come from model repeats, user double-clicks, executor retries and crash recovery, so the key must survive all four.

**Key points to mention:**
- Key choice: a per-attempt UUID is useless; `tool_call_id` misses model repeats; canonical argument hashes and business keys catch more.
- Canonicalize before hashing, or `INV-0042` and `inv-0042` produce two keys.
- Leases on in-progress records let another worker take over after a crash.
- Target zero double effects in fault-injection tests.

**Deep dive:** [Preventing duplicate payments](../09-agents/02-tool-calling-in-depth.md#preventing-duplicate-payments-and-other-irreversible-side-effects), [Compensation](../09-agents/02-tool-calling-in-depth.md#compensation-undoing-what-can-be-undone)

</details>

<details>
<summary><b>Q21. A payment tool call timed out. What exactly do you do?</b></summary>

**Answer:** Treat it as an unknown outcome, not a failure: the timeout cancelled your waiting, not the remote operation. If the provider honors idempotency keys, retry with the same key, which returns the original payment if it already landed. If it does not, do not retry: mark the record as unknown, tell the model to say the payment is being verified (never that it failed or succeeded), and run reconciliation that queries the provider by key or business reference, then notify the user of the real outcome. Look up the key before re-running business checks, or the already-applied payment will make the balance check fail confusingly. Track unknown-outcome rate and reconciliation lag as health metrics.

**Key points to mention:**
- Same-key retries only work inside the provider's key-retention window (often around 24 hours); after that, reconcile by business reference.
- Your own database invariant is the backstop when keys differ.
- The most valuable single test: "timeout after commit" must produce exactly one payment.

**Deep dive:** [Unknown outcomes and reconciliation](../09-agents/02-tool-calling-in-depth.md#unknown-outcomes-and-reconciliation)

</details>

<details>
<summary><b>Q22. How do you distinguish a model-generated tool request from successful tool execution?</b></summary>

**Answer:** A tool call in the model output is a request; nothing has happened yet. Execution is proven only by the result your executor produced after the real tool returned, with a status and a real external identifier (a payment id), recorded in an audit or idempotency table and ideally reconcilable with the downstream system of record. Model each call as a state machine: proposed, validated, denied, pending approval, executing, succeeded, failed, unknown. Return machine-readable statuses such as `pending_approval` or `unknown_outcome` with explicit "do not claim success" notes, and build user-facing confirmations from executor records, not from model prose. An optional claims check compares actions mentioned in the final answer against succeeded results in the trace.

**Key points to mention:**
- Model text like "I've paid it" is a claim, not evidence.
- Metric: hallucinated-action rate (final answers claiming actions with no matching succeeded result).
- Postconditions checked against the system of record catch these too.

**Deep dive:** [Proposed is not executed: the tool-call lifecycle](../09-agents/02-tool-calling-in-depth.md#proposed-is-not-executed-the-tool-call-lifecycle)

</details>

<details>
<summary><b>Q23. How do you keep large tool outputs from blowing up cost and quality?</b></summary>

**Answer:** Return only what the model needs: project fields, paginate, filter server side, truncate with an explicit marker, summarize in a sub-step, or return a handle to the full result plus a tool to query it. Results are resent on every later call, so total input tokens are $(n+1)P + T \cdot n(n+1)/2$ for base prompt $P$ and result size $T$ over $n$ tool calls ($n+1$ model calls). Illustratively, 5 tool calls on a 2,000-token base prompt cost 162,000 input tokens with 10,000-token results but only 19,500 with 500-token results. Long contexts also hurt recall of mid-context facts and raise prefill latency. Smaller outputs shrink the prompt-injection surface too.

**Key points to mention:**
- Design tools to return compact, structured results by default.
- Truncation must be visible to the model ("showing 20 of 340, use page=2").
- Summarize with a smaller model when a large document must be read.

**Deep dive:** [Managing large tool outputs](../09-agents/02-tool-calling-in-depth.md#managing-large-tool-outputs)

</details>

## Authorization, human approval and prompt injection

<details>
<summary><b>Q24. How do you enforce access control for tools, and why must authorization be enforced outside the LLM?</b></summary>

**Answer:** Tools execute with the end user's identity and scopes, taken from the authenticated session and injected by the server, never from model arguments (no `user_id` parameter the model can choose). Filter which tools the model sees per user, check every call with a policy engine against the specific resource (ownership, tenant, limits), and call downstream services with delegated, downscoped tokens (for example OAuth 2.0 Token Exchange, RFC 8693) so they enforce permissions too. Authorization must live outside the LLM because the model is probabilistic, can be manipulated by any text it reads, does not know the real identity, and cannot be unit-tested or audited like code. An agent on a powerful service account is a confused deputy waiting to happen: injected text can make it use its own privileges for a less privileged user. Policy engines such as OPA (Rego), Cedar or Zanzibar-style systems (OpenFGA, SpiceDB) make the decisions testable and loggable with a policy version.

**Key points to mention:**
- "Model proposes, code disposes": authorization questions have exact answers, so use a lookup, not a judgment.
- Trade-off: per-user tool filtering fragments prompt-cache hits; fixed role-based tool sets are a compromise.
- Unit-test every policy rule; a denied-call spike can signal an injection campaign.

**Deep dive:** [Access control: whose permissions does a tool use?](../09-agents/02-tool-calling-in-depth.md#access-control-whose-permissions-does-a-tool-use), [Why authorization must be enforced outside the LLM](../09-agents/02-tool-calling-in-depth.md#why-authorization-must-be-enforced-outside-the-llm), [Deterministic authorization and policy enforcement](../09-agents/08-agent-reliability-security-and-prompt-injection.md#deterministic-authorization-and-policy-enforcement-outside-the-model)

</details>

<details>
<summary><b>Q25. When should human approval be required, and how do you design it?</b></summary>

**Answer:** Require approval by risk tier, not by the model's mood: actions that are irreversible, above amount or volume thresholds, financial actions to new payees, communications outside the organization, changes to permissions or account security, operations with a large blast radius, and actions taken under low-confidence signals (untrusted content in context, ambiguous matches, repeated validation failures, out-of-pattern amounts). Read-only and small reversible actions run automatically, and some actions are denied outright with no approval path. Approval is enforced by the application, bound to the exact canonical arguments with a signed token that expires, and given by an authenticated human, never by the model. The approval card shows the exact action, the evidence (with untrusted content marked), why approval is needed and whether it is reversible; any change to the arguments requires re-approval, and a timeout defaults to reject. The run pauses durably and re-checks business rules when it resumes.

**Key points to mention:**
- Use observable signals, not the model's self-reported confidence, which is poorly calibrated.
- Approval fatigue: if most actions need approval, reviewers rubber-stamp. Track approve, edit and reject rates and review time.
- Separation of duties for high value; out-of-band verification for bank-detail changes.
- Roll out from "suggest only" and raise autonomy per action class as data proves it safe.

**Deep dive:** [When to require human approval](../09-agents/02-tool-calling-in-depth.md#when-to-require-human-approval), [Where human approval belongs](../09-agents/08-agent-reliability-security-and-prompt-injection.md#where-human-approval-belongs-and-how-to-design-it)

</details>

<details>
<summary><b>Q26. How do you protect against prompt injection, including injection from tools or retrieved documents?</b></summary>

**Answer:** Assume it will sometimes succeed at the model level, because instructions and data share one token stream and no prompt-only defense is complete, then design so a hijacked model cannot do damage. Indirect injection is the main agent threat: instructions hidden in emails, web pages, indexed documents, order notes, tool descriptions or API fields. Bound the damage with deterministic controls: authorization and business rules in code, least-privilege narrow tools, credentials scoped to the current user and task, human approval for irreversible or external actions, egress allowlists, and taint rules that raise the risk tier once untrusted content enters the context. For high-risk flows, use privilege-separated designs (dual LLM, CaMeL) so untrusted text cannot change which tools are called. On top, label untrusted content as data (spotlighting), run classifiers as signals, and measure attack success with red-team suites such as AgentDojo in CI.

**Key points to mention:**
- The lethal trifecta (Willison, 2025): private data plus untrusted content plus external communication. Break one leg per task.
- Block or proxy URLs in rendered output to stop markdown-image exfiltration.
- Trade-off: strict layers cost utility, so apply them where a run is tainted and holds high-risk tools.
- For RAG, enforce document permissions at retrieval as well.

**Deep dive:** [Prompt injection from tools and retrieved documents](../09-agents/08-agent-reliability-security-and-prompt-injection.md#prompt-injection-from-tools-and-retrieved-documents), [Prompt injection through tool results](../09-agents/02-tool-calling-in-depth.md#prompt-injection-through-tool-results), [Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md)

</details>

<details>
<summary><b>Q27. What is the difference between a jailbreak and indirect prompt injection, and why does it matter for agents?</b></summary>

**Answer:** A jailbreak is the user trying to make the model violate its safety training; indirect prompt injection is a third party planting instructions in content the agent reads, so that the agent works against its own user. The threat models are opposite: in a jailbreak the user is the adversary, in injection the user is the victim. For agents, injection is worse because the attacker needs no access to the conversation, and the payload can trigger real actions with the legitimate user's permissions. Defenses differ accordingly: safety training and output moderation for jailbreaks; privilege separation, scoped tokens, egress control and deterministic policy for injection. Injection can also persist (written into memory or an indexed document) and spread between agents.

**Key points to mention:**
- Reference: Greshake et al., 2023, on indirect prompt injection.
- Memory poisoning turns a one-time injection into a persistent compromise.
- In multi-agent systems, injection can hop from a low-privilege reader to a high-privilege actor.

**Deep dive:** [Prompt injection from tools and retrieved documents](../09-agents/08-agent-reliability-security-and-prompt-injection.md#prompt-injection-from-tools-and-retrieved-documents), [LLM Safety, Guardrails and Responsible AI](../05-llms/13-llm-safety-guardrails-and-responsible-ai.md)

</details>

<details>
<summary><b>Q28. Explain the dual-LLM pattern and its limitations.</b></summary>

**Answer:** Split the agent into a privileged LLM that plans and calls tools but never sees untrusted text, and a quarantined LLM that reads untrusted text with no tool access and returns results as opaque variables or tightly constrained values. Plain code moves those variables around and substitutes them only at execution or display time, so injected instructions cannot change which tools are called. CaMeL (Debenedetti et al., 2025) generalizes the idea with an interpreter and per-value capabilities that track provenance and allowed data flows. Limitations: the data can still be malicious where it lands (a phishing link in an emailed summary), so outputs still need validation and egress control; and utility drops when the planner genuinely needs to reason about untrusted content ("decide whether to refund based on this email"). Use it for high-risk flows that combine private data, untrusted content and external actions.

**Key points to mention:**
- It breaks the lethal trifecta by construction rather than by detection.
- Constrained outputs (enums, numbers, ids) from the quarantined model are the safe interface.
- Proposed by Simon Willison (2023).

**Deep dive:** [Layered defenses](../09-agents/08-agent-reliability-security-and-prompt-injection.md#layered-defenses)

</details>

## Orchestration with LangGraph and state machines

<details>
<summary><b>Q29. How would you implement a production agent using LangGraph or a similar framework?</b></summary>

**Answer:** Model the agent as an explicit state graph: a typed, minimal state schema; small nodes for classify, retrieve, plan, validate, approve, execute and respond; and conditional edges written in code. LLMs make only the judgement calls (intent, plan); validation, risk checks and routing are deterministic. Compile with a durable checkpointer (Postgres in production) keyed by one thread ID per ticket or job, use `interrupt()` in an approval node for risky actions, give tool nodes retry policies for transient errors, make side effects idempotent with keys derived from the thread ID, and trace every node, LLM call and tool call. Bound every cycle (for example at most 3 re-plans) and set a recursion limit and cost budget that route to an error node and a human. Gate releases on a scenario suite, launch in shadow mode or approval-for-everything, and canary new graph versions on a slice of new threads.

**Key points to mention:**
- State: typed, serializable, with reducers for multi-writer fields and references instead of large blobs.
- Deployment: stateless workers behind a queue, one active run per thread, streaming updates to the UI.
- Version the graph and state schema because paused threads outlive deploys.
- Pitfall: a free-form ReAct loop with every tool exposed and "please ask before refunding" in the prompt.

**Deep dive:** [Implementing a production agent: a complete example graph](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#10-implementing-a-production-agent-a-complete-example-graph), [Going deeper: running agent graphs in production](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#11-going-deeper-running-agent-graphs-in-production)

</details>

<details>
<summary><b>Q30. Compare LangGraph, a simple orchestration loop and multi-agent architectures.</b></summary>

**Answer:** They sit on different axes. A simple loop is a `while` loop around one LLM with tools: tiny, transparent and cheap, but hard to bound, resume or audit, with no persistence, approvals or structured branching. LangGraph is a framework for explicit state graphs: code defines nodes and allowed transitions, the model works inside nodes, and the engine adds checkpoints, interrupts, retries and time travel, at the cost of upfront design, framework concepts and API churn. Multi-agent is an architecture (several agents with a coordination scheme) that you can build with either; it buys context isolation, specialization and parallelism at the cost of tokens, latency, compounding errors and debugging effort. The progression I recommend: prototype with a loop, move to an explicit graph when you need durability, approvals and retries, and go multi-agent only when evals show a single agent's prompt or tool list is the bottleneck.

**Key points to mention:**
- Concrete trigger to leave the loop: a refund that waits a day for approval; the loop's in-memory messages die on restart, a checkpointed graph resumes.
- LangGraph is orchestration plumbing; a LangGraph app can be a single agent.
- Compare fairly: same scenario suite, task success, pass^k, policy compliance and cost per successful task.
- For a refund agent, pick a graph: approval, idempotency and audit are hard requirements.

**Deep dive:** [Three ways to orchestrate](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#1-three-ways-to-orchestrate-a-loop-a-chain-a-graph), [Simple loop vs LangGraph vs multi-agent](../09-agents/05-multi-agent-systems.md#simple-loop-vs-langgraph-vs-multi-agent)

</details>

<details>
<summary><b>Q31. When is a state graph preferable to a simple chain?</b></summary>

**Answer:** When the flow is not a straight line or must outlive one request: branching on intermediate results, cycles such as re-planning or retrieve-again loops, persistence across restarts or long waits, human approval in the middle, recovery from a failed step without redoing earlier ones, or a step-by-step audit trail. A support refund with approval, agentic RAG that may re-query, and document processing with human review are graph-shaped. RAG question answering, extraction and summarization are fixed pipelines that finish in one request and should stay chains. The graph costs more code, a state schema to maintain and a checkpoint write per step. A middle ground is a chain inside a single graph node, or a functional API for linear code that still needs persistence and interrupts.

**Key points to mention:**
- Structure reduces the number of model-decided steps, which raises $p^n$ reliability.
- Pitfall: a graph for a 3-step pipeline "to be future-proof".

**Deep dive:** [When a state graph beats a simple chain or loop](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#2-when-a-state-graph-beats-a-simple-chain-or-loop)

</details>

<details>
<summary><b>Q32. How do you manage state and checkpoints?</b></summary>

**Answer:** Define a small, typed state schema with explicit reducers, have nodes return partial updates, and attach a durable checkpointer so a checkpoint is written after every super-step under a thread ID per conversation or job. A checkpoint holds state values, the next nodes, metadata, a parent pointer and pending writes from parallel nodes that already succeeded. To resume after a crash or an approval, invoke the same thread (in LangGraph, passing `None` as input continues from the latest checkpoint); to debug, inspect or fork the checkpoint history. Node execution is at-least-once, so side effects need idempotency keys, and only one run should be active per thread (queue or lock). Keep state lean with references to large data, encrypt and expire checkpoints, and version the schema with additive optional fields and stable node names.

**Key points to mention:**
- Storage growth (illustrative): 20 KB per checkpoint, 15 super-steps, 100,000 runs a day is 30 GB a day; savers that store only changes grow more slowly.
- Long-term cross-thread memory lives in a separate store, not in thread state.
- Choose sync vs async checkpoint writes as a durability vs latency trade-off.

**Deep dive:** [Persistence: checkpointers, threads and crash recovery](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#5-persistence-checkpointers-threads-and-crash-recovery)

</details>

<details>
<summary><b>Q33. How do you model branching, retries, checkpoints, recovery and human approval in an agent graph?</b></summary>

**Answer:** Branching is conditional edges whose router functions are plain code reading state (often a label the model produced), with a default branch. Retries depend on the failure class: transient errors get a per-node retry policy with exponential backoff and jitter ($d_n = \min(d_{\max}, d_0 b^{n-1})$, so 0.5, 1, 2 s); invalid model output loops back to the planner with the validation errors in state, bounded by an attempt counter; permanent or business-rule failures go to an error node. Checkpoints after every super-step let a crashed run resume at the failed node, with idempotency keys making the re-run safe and pending writes preventing successful parallel branches from re-running. Human approval is a dedicated node that calls `interrupt()` with the exact actions, persisted by the checkpointer and resumed with the reviewer's decision (approve, edit or reject) via a resume command. Multi-step side effects that must be undone use saga-style compensations, and budget breaches route to a human.

**Key points to mention:**
- Retry at one layer only to avoid multiplicative storms (3 x 3 x 3 = 27 calls).
- Retries help independent transient failures ($0.95 \to 0.9975$ with one retry), not deterministic ones.
- Nothing side-effecting above `interrupt()`; bind the approval to a hash of the approved actions.
- Measure retry rate, re-plan attempts and approval wait time per node.

**Deep dive:** [Branching, retries, error nodes and recovery](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#7-branching-retries-error-nodes-and-recovery), [Human in the loop: interrupts, resuming and time travel](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#6-human-in-the-loop-interrupts-resuming-and-time-travel)

</details>

<details>
<summary><b>Q34. What happens to side effects when a node is retried or resumed after an interrupt, and how do you make that safe?</b></summary>

**Answer:** The node runs again from the top. Retries re-run the failed node, resuming after `interrupt()` re-runs the interrupted node, and crash recovery re-runs whatever node was in progress, so any side effect inside that node can happen more than once. Make side effects idempotent with keys derived from stable IDs (thread ID plus action), keep `interrupt()` above any side effects or isolate side effects in their own nodes, and keep one side effect per node where practical. At-least-once execution plus idempotency equals exactly-once effects. For external systems without idempotency support, use check-then-act with a status lookup, or a durable engine where each call is a separately recorded activity.

**Key points to mention:**
- For your own services, store processed keys in a table with a unique constraint.
- A node with three tool calls re-runs all three; finer-grained nodes shrink the blast radius.
- Use saga compensations when an effect must be undone.

**Deep dive:** [Retry policies and exponential backoff](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#73-retry-policies-and-exponential-backoff), [Compensation for multi-step side effects](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#75-compensation-for-multi-step-side-effects)

</details>

<details>
<summary><b>Q35. What is a reducer in LangGraph, and why is it needed?</b></summary>

**Answer:** A reducer is a per-field function `(old, new) -> merged` that defines how a node's update combines with the existing state; without one, a new value simply overwrites the old. Reducers are needed for accumulating fields such as message histories or tool results, and essential when parallel nodes in the same super-step write the same key: without a reducer, LangGraph raises an update-conflict error instead of silently dropping data. `Annotated[list, operator.add]` concatenates lists, and `add_messages` appends and replaces by message ID. Custom reducers can dedupe by key, keep a maximum or merge dicts. A common bug is an append reducer on a field you meant to reset each loop, which then grows forever.

**Key points to mention:**
- Do not rely on the order of writes from parallel nodes; sort by a key in the join node.
- Reducers are what make fan-out and fan-in (`Send` per item, then a join node) safe.

**Deep dive:** [Reducers: how updates merge](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#42-reducers-how-updates-merge)

</details>

<details>
<summary><b>Q36. How does LangGraph checkpointing compare with a durable execution engine such as Temporal?</b></summary>

**Answer:** LangGraph saves a state snapshot per super-step and resumes at the pending node, re-running it. Temporal records an event history of every activity result and replays deterministic workflow code, skipping completed activities, with first-class timeouts, retries, signals and durable timers. LangGraph gives the better developer experience for agent logic (typed state, messages, streaming, human-in-the-loop, time travel); Temporal gives stronger guarantees for long-running, multi-service business processes, at the cost of running a cluster or managed service and keeping workflow code deterministic. Granularity is the practical difference: a node with three tool calls re-runs all three, while in Temporal each can be its own activity. Neither gives exactly-once side effects for free, so idempotency keys are still required; they combine well, with Temporal for the business process and LangGraph for the reasoning step.

**Key points to mention:**
- Temporal constraint: all I/O lives in activities; workflow code must be deterministic.
- Other engines in the family: AWS Step Functions, Azure Durable Functions, Restate, DBOS, Inngest.

**Deep dive:** [Durable execution engines (Temporal-style)](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#8-durable-execution-engines-temporal-style-and-how-they-relate)

</details>

<details>
<summary><b>Q37. You must deploy a new graph version while hundreds of threads are paused waiting for approval. What do you do?</b></summary>

**Answer:** Treat the graph and its state schema like a database schema with live data. Prefer backward-compatible changes: new optional fields with defaults, new nodes, unchanged node names. Record the graph version in thread metadata and either keep the old version running until its threads drain (routing old threads to old workers at the API layer) or write an explicit migration that rewrites old checkpoints. Renaming or deleting a node that a paused checkpoint lists as next breaks resume, and changing a field's type or reducer can corrupt merges for in-flight threads. Test by resuming a corpus of real paused checkpoints from staging against the new code as a contract test.

**Key points to mention:**
- Version state schemas explicitly; resume-from-checkpoint success is a release gate.
- Canary new versions on new threads first.

**Deep dive:** [Versioning graphs with in-flight threads](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#113-versioning-graphs-with-in-flight-threads)

</details>

## Multi-agent systems

<details>
<summary><b>Q38. What are the disadvantages of multi-agent systems?</b></summary>

**Answer:** Multiple agents cost more tokens and money because each has its own prompt, tools and loop; in an illustrative example, a supervisor plus 3 workers cost about 2.7 times a single agent for the same request. They add latency through sequential handoffs and plan-and-synthesize calls, and errors compound because more steps must all succeed: with $p = 0.95$, 5 steps give 0.774, 10 give 0.599, 20 give 0.358. They introduce coordination failures (lost context in lossy summaries, duplicated work, ping-pong loops), conflicting decisions, and harder debugging and evaluation because there are many trajectories. The security surface grows: more credentials and endpoints, and injected text can propagate from a low-privilege agent to a high-privilege one. Always give the mitigation for each and say when the benefits (parallelism, context isolation, specialization) outweigh the costs.

**Key points to mention:**
- Failure taxonomy (Cemri et al., 2025): specification and design issues, inter-agent misalignment, verification and termination failures.
- A/B the design against a strong single-agent baseline with several trials per task.
- Compare cost per successful task and p95 latency, not just accuracy.

**Deep dive:** [The disadvantages, with numbers](../09-agents/05-multi-agent-systems.md#the-disadvantages-with-numbers)

</details>

<details>
<summary><b>Q39. How do you prevent agents from producing conflicting decisions?</b></summary>

**Answer:** Agents propose; exactly one owner decides. Give each decision type a single owner, ideally deterministic code, which is the only component holding the action tool ("parallel read, single write"). All agents read and write a shared, typed, versioned state that is the source of truth, and disagreements are resolved by arbitration rules written in advance as a deterministic merge function. For example, if billing proposes a refund up to 120, policy up to 80 and fraud approves: any hard block denies, missing proposals escalate, unanimous approval takes the minimum (80), amounts above the auto-limit escalate. Shared resources are protected by optimistic concurrency (stale writes rejected), leases with a TTL, and idempotency keys on the side effect so it executes at most once even if a lease expires.

**Key points to mention:**
- Why not an LLM judge? Fine for advisory synthesis; for money and permissions you want reproducible, testable rules plus human escalation.
- A lease can expire while a slow holder still works, so the lock alone is not enough.
- Verify with arbiter unit tests, race tests on the same case and audit-log checks for contradictory outbound messages.

**Deep dive:** [Preventing conflicting decisions](../09-agents/05-multi-agent-systems.md#preventing-conflicting-decisions)

</details>

<details>
<summary><b>Q40. Describe the main multi-agent architectures and when you would use each.</b></summary>

**Answer:** Supervisor or orchestrator-worker: one lead dispatches subtasks and synthesizes, best for parallel research, with the risks of a bottleneck and lossy summaries. Hierarchical teams: supervisors of supervisors for very large multi-domain tasks, where each layer adds plan and synthesize calls, so keep it shallow. Peer handoffs (swarm): the active agent transfers control, good for conversational routing such as support triage, but it needs hop limits and ping-pong detection, and global budgets are harder because no one is in charge. Debate and critique: several answers critiqued and judged, for verifying high-stakes outputs; with independent $p = 0.8$, a 3-way majority reaches 0.896, but real agents share errors so gains are smaller. Sequential pipelines and blackboard designs round out the set.

**Key points to mention:**
- Communication channels: a mesh has $n(n-1)/2$ (10 for 5 agents), a star has $n-1$ (4), which is why supervisors are easier to operate.
- Use structured worker outputs with source references to limit information loss.
- Supervisor and swarm patterns are commonly built on LangGraph, with agents as nodes or subgraphs.

**Deep dive:** [Multi-agent architectures](../09-agents/05-multi-agent-systems.md#multi-agent-architectures)

</details>

<details>
<summary><b>Q41. When would you choose a single agent with good tools over a multi-agent design?</b></summary>

**Answer:** By default. A single agent wins when the task is sequential, involves consistent writes or decisions, fits in one context window, has tight latency and cost budgets, or when the real problem is weak tools rather than missing specialists. One agent keeps the full context, so nothing is lost in handoffs and implicit decisions stay consistent, and fewer steps mean less compounding error. Often a better tool (one returning clean, filtered results) replaces a whole research sub-agent. Go multi-agent only for wide, independent subtasks where parallelism and context isolation pay off, or for deliberate security separation, and only with evals showing a gain over a strong single-agent baseline.

**Key points to mention:**
- Middle ground: sub-agents exposed as read-only tools to the main agent.
- Separation can itself be the reason: an agent that reads untrusted content and an agent that acts.
- Measure cost per successful task and p95 latency, not just accuracy.

**Deep dive:** [When one agent with good tools beats many agents](../09-agents/05-multi-agent-systems.md#when-one-agent-with-good-tools-beats-many-agents)

</details>

<details>
<summary><b>Q42. What new security risks does a multi-agent system introduce, and how do you mitigate them?</b></summary>

**Answer:** Prompt injection can propagate: a low-privilege agent reads malicious content and passes instructions in its output to a high-privilege agent, which becomes a confused deputy. There are also more credentials, more tool permissions and more inter-agent endpoints to secure. Mitigate with least privilege per agent, separation between agents that read untrusted content and agents that act, schema-validated messages treated as data (pass extracted typed fields, never a free-text summary into a privileged agent's instructions), authenticated agent-to-agent calls, deterministic policy checks and human approval for irreversible actions. Full traces across agents give the audit trail.

**Key points to mention:**
- For remote agents over A2A, authenticate with the declared scheme and verify the peer's identity; treat its outputs as untrusted.
- Trust problems multiply with each agent; the policy layer at the executor stays the boundary.

**Deep dive:** [Security practices specific to multi-agent](../09-agents/05-multi-agent-systems.md#security-practices-specific-to-multi-agent)

</details>

## Agent memory

<details>
<summary><b>Q43. How would you implement memory for an AI agent?</b></summary>

**Answer:** Separate short-term and long-term memory. Short-term is the session: recent turns verbatim in a sliding window plus a running summary, persisted per thread by a checkpointer. Long-term is an external store namespaced by tenant and user, with a structured key-value profile for stable facts and a vector store of atomic memories for fuzzy recall. On the read path, always inject the small profile and retrieve the top few memories by similarity plus recency plus importance, behind metadata filters and inside a token budget. On the write path, a background job uses an LLM to extract atomic facts from user turns only, compares them with existing memories (add, update, delete or no-op), asks for confirmation on sensitive facts, and stores provenance and timestamps. Around that: TTLs, deletion that follows lineage, isolation at the storage layer, and multi-session evaluation.

**Key points to mention:**
- Data model fields: tenant, user, kind, key, content, embedding, importance, source, sensitive, status, superseded_by, valid_from, expires_at.
- Keep source text so you can re-embed when the embedding model changes.
- Pitfalls: similarity-only retrieval, no conflict handling, extraction from untrusted content, no deletion path.
- Measure multi-session recall, knowledge updates, abstention, write precision and zero canary leaks.

**Deep dive:** [The types of agent memory](../09-agents/06-agent-memory.md#the-types-of-agent-memory), [How memory fits into the agent loop](../09-agents/06-agent-memory.md#how-memory-fits-into-the-agent-loop)

</details>

<details>
<summary><b>Q44. Compare conversation history, summarized memory and external long-term memory.</b></summary>

**Answer:** Conversation history keeps recent turns verbatim: perfect fidelity, scoped to one session, but a full buffer costs more every turn (quadratic in total). Summarized memory compresses older turns into a running summary: bounded size, keeps the gist, but lossy, and repeated re-summarization drifts and drops numbers and exact constraints. External long-term memory stores selected facts and episodes in a database across sessions: scales to months of history and only costs tokens for what you retrieve, but depends on a good write policy and retrieval and introduces staleness, isolation and privacy concerns. Illustratively, 50 turns of 300 tokens cost 382,500 input tokens with a full buffer versus about 156,500 with a 10-turn window plus a 500-token summary. In practice you combine all three.

**Key points to mention:**
- Frame by scope (session vs cross-session), fidelity (verbatim, lossy, selective) and cost (growing, bounded, pay-per-retrieval).
- Use structured summaries (decisions, constraints, open questions) and keep raw transcripts retrievable.
- Short chat needs a buffer; a long session needs window plus summary; recall across days needs long-term memory.

**Deep dive:** [Comparing the memory types](../09-agents/06-agent-memory.md#comparing-the-memory-types)

</details>

<details>
<summary><b>Q45. How do you select relevant historical information from memory?</b></summary>

**Answer:** Two stages. First, exact metadata filters: namespace (tenant and user), active status, not expired, and type or domain. Then a ranked search: embed a query built from recent turns (not just the last message), take top candidates by vector similarity (often plus keyword search), and rescore with $\text{score} = w_s \cdot \text{sim} + w_r \cdot 0.5^{\Delta t / h} + w_i \cdot \text{importance}$. With weights 0.6, 0.2, 0.2 and a 30-day half-life, "moved to Seattle" (similarity 0.75, 15 days old, importance 0.9) scores about 0.771 and beats "prefers transit hotels" (0.85, 30 days, 0.6) at 0.73. Keep only the top few above a threshold, dedupe and fit them into a token budget. Scoring does not remove false facts, though: the stale "lives in Boston" still scored 0.56, which is why status filters from conflict resolution matter.

**Key points to mention:**
- Tune weights and thresholds on labeled query-memory pairs (precision@k, recall@k).
- Options: cross-encoder reranking, query rewriting for "book that one", skipping retrieval when not needed.

**Deep dive:** [Selecting relevant historical information](../09-agents/06-agent-memory.md#selecting-relevant-historical-information)

</details>

<details>
<summary><b>Q46. How do you handle outdated or contradictory memory?</b></summary>

**Answer:** Make time and provenance first-class. Every memory has timestamps (including `valid_from`, when the fact became true), a source (user-stated, inferred, tool) and a status. Single-valued facts live in keyed slots with one active value per user, and a new value supersedes the old one, which is kept for audit, never overwritten. For free text, at write time retrieve similar memories and have an LLM classify the new fact as add, update, delete or no-op. Conflicts follow a priority order: system of record, then explicit user correction, then stated over inferred, then newer over older; if it is still ambiguous, ask the user. Recency decay per memory type (allergies never decay, current location expires in days) and periodic consolidation handle slow drift.

**Key points to mention:**
- Out-of-order writes: compare `valid_from`, not arrival order.
- For business data, reference the authoritative system rather than copying it into memory.
- Show dates in the prompt so the model can reason "as of March 2026".
- Test with knowledge-update cases where a fact changes mid-history.

**Deep dive:** [Handling outdated or contradictory memory](../09-agents/06-agent-memory.md#handling-outdated-or-contradictory-memory)

</details>

<details>
<summary><b>Q47. How do you enforce tenant isolation in agent memory?</b></summary>

**Answer:** Every memory carries `tenant_id` and `user_id`, taken from the authenticated request and injected by the server, never from model output or tool arguments. The filter is mandatory at the storage layer: row-level security in the database or a data-access wrapper that refuses unscoped queries and fails closed. For large or regulated tenants, use physically separate indexes or databases plus per-tenant encryption keys; caches are keyed by tenant and user, and shared organization memory is a separate namespace with its own ACL. Memory is especially risky because similarity search will happily return another user's near-identical memory if ownership is not a hard filter. Test continuously with canary memories and adversarial prompts, expecting zero leaks.

**Key points to mention:**
- Filtered ANN pitfall: post-filtering can return too few results for selective filters; use filter-aware search or partition by tenant.
- Hidden leak paths: semantic caches, cross-user summaries, logs, traces, analytics exports, fine-tuning sets.
- Shared vs per-tenant index: cost and simplicity vs blast radius and easy deletion.

**Deep dive:** [Going deeper: tenant and user isolation](../09-agents/06-agent-memory.md#going-deeper-tenant-and-user-isolation), [Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md)

</details>

<details>
<summary><b>Q48. How do you avoid unnecessary token consumption from memory?</b></summary>

**Answer:** Give the prompt an explicit token budget per component and assemble within it (illustrative: system 1.5k, tools 2k, profile 300, memories 800, summary 500, recent turns 4k, documents 4k, output reserve 2k). Use a sliding window plus a running summary instead of a full buffer, whose total cost is $t \cdot N(N+1)/2$, and compact old turns when history passes a threshold. Replace large tool outputs with short summaries and IDs, retrieve memories only above a relevance threshold with a small k, dedupe and format them compactly, and skip retrieval when the message does not need it. Order the prompt with stable content first so prefix caching works, and run summarization and extraction with a smaller model in the background. When over budget, trim the lowest-scored memories first, then documents, then fold turns into the summary, never the system prompt or the current message.

**Key points to mention:**
- Measure tokens per successful task, not per call.
- Over-compression causes re-asking and errors, which costs more overall.
- Load only relevant tool definitions; they are a large hidden cost.

**Deep dive:** [Going deeper: avoiding unnecessary token consumption](../09-agents/06-agent-memory.md#going-deeper-avoiding-unnecessary-token-consumption)

</details>

<details>
<summary><b>Q49. When should memory be forgotten or deleted?</b></summary>

**Answer:** Five cases: when it expires (a TTL for short-lived facts such as current location or an upcoming trip); when the user asks, in chat or in settings; when regulation or contract requires it (right to erasure, retention limits); when it is sensitive data that should never have been stored (credentials, card numbers); and when it is low value (never retrieved, decayed, duplicate). Deletion must be complete: rows including old versions, vector entries, derived summaries, caches, transcripts, logs and eventually backups, with an audit record that it happened. Lineage fields (`derived_from`) let you rebuild summaries without the deleted fact. Embeddings can leak text, so deleting text while keeping vectors is not deletion, and crypto-shredding (destroying per-user keys) handles backups.

**Key points to mention:**
- Soft delete, then hard delete within a defined window; vector tombstones need compaction.
- Personal data in fine-tuned weights is hard to remove, a strong argument for external memory.
- Prove it with canary audits: plant a fact, delete the test user, search every store.

**Deep dive:** [Going deeper: forgetting and deletion](../09-agents/06-agent-memory.md#going-deeper-forgetting-and-deletion)

</details>

<details>
<summary><b>Q50. How do you prevent memory poisoning?</b></summary>

**Answer:** Treat memory as untrusted data. Extract memories only from the user's own turns, never from tool outputs, web pages or documents, and store provenance on every memory. Reject candidates that look like instructions, require explicit confirmation for high-impact memories (anything that changes where data is sent or what is allowed), and review changes to procedural memory. In the prompt, present memories as quoted reference data, and keep tool-level guardrails so a poisoned memory cannot trigger a risky action on its own. Poisoning matters because it turns a one-time injection into a persistent compromise across sessions.

**Key points to mention:**
- Per-user isolation limits the blast radius.
- User-visible memory pages let users spot and delete strange entries.
- Alert on memories containing email addresses, URLs or imperative phrasing; test with injected documents.

**Deep dive:** [Going deeper: privacy, security and memory poisoning](../09-agents/06-agent-memory.md#going-deeper-privacy-security-and-memory-poisoning)

</details>

## Model Context Protocol and tool ecosystems

<details>
<summary><b>Q51. What problem does MCP solve, and how is it different from function calling?</b></summary>

**Answer:** Function calling is how a model expresses "call tool X with arguments Y". MCP (Model Context Protocol) is an open, model-agnostic protocol for how an LLM application discovers and invokes tools, data and prompts that live in separate processes or services. It turns the $M \times N$ integration problem (every app writing a connector for every system) into $M + N$: 5 apps and 8 systems need 40 custom connectors but only 13 protocol implementations. The host still uses the model's function calling; it just gets tool definitions from MCP servers (`tools/list`) and routes calls back to them (`tools/call`) over JSON-RPC 2.0. The senior part of the answer is the cost shift: you now have to trust, version and secure third-party servers.

**Key points to mention:**
- Layers: model API (function calling), integration protocol (MCP), agent logic (your loop).
- MCP adds dynamic discovery, capability negotiation, standard transports (stdio, streamable HTTP) and an auth model for remote servers.
- Skip MCP for one app with a few in-process tools; use it when tools are shared across hosts or owned by other teams.

**Deep dive:** [The integration problem](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#the-integration-problem), [What MCP is](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#what-mcp-is-hosts-clients-and-servers)

</details>

<details>
<summary><b>Q52. Explain MCP hosts, clients and servers, and the tools, resources and prompts primitives.</b></summary>

**Answer:** The host is the LLM application the user uses; it owns model calls, UI, consent and policy. Inside the host, each client keeps one connection to one server, which isolates state, capabilities and failures and means a server only sees its own traffic. Servers expose three primitives: tools are model-controlled executable functions (the model decides to call `search_tickets`), resources are application-controlled read-only data addressed by URI (the user attaches `ticket://4812`), and prompts are user-controlled templates (the user picks `/triage`). The host merges the tool catalogs, namespaces names, translates them into the model's format and routes calls back; the model never talks to a server directly. That is why approval, logging and redaction belong in the host or a gateway.

**Key points to mention:**
- Client-side features: sampling, roots and elicitation (servers asking the host for things).
- Host support for resources and prompts varies; put core value in tools.
- Isolation does not stop tool poisoning, because the model sees all descriptions together.

**Deep dive:** [The primitives: tools, resources and prompts](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#the-primitives-tools-resources-and-prompts)

</details>

<details>
<summary><b>Q53. What is tool poisoning, and how would you defend against it?</b></summary>

**Answer:** Tool poisoning is hiding malicious instructions in a tool's description or schema, which the model reads but the user usually does not see. For example, a weather tool whose description tells the model to read the user's SSH key with the filesystem tool and pass it as an argument. It is cross-server: the poisoned server cannot read your files, but it can get the model to do it with another server's tool and send it the data. Variants include rug pulls (definitions change after approval), tool shadowing and name collisions. Defend by vetting and allowlisting servers, showing full descriptions at approval, pinning definition hashes and re-approving on change, namespacing tool names and scanning descriptions, then layering capability limits (least-privilege credentials, approvals on outbound actions, egress control) so obedience does limited damage.

**Key points to mention:**
- Tool annotations come from the same untrusted server and do not help.
- Keep poisoned and mutated server fixtures in CI and assert the host flags changed fingerprints.

**Deep dive:** [Security: the MCP threat model](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#security-the-mcp-threat-model)

</details>

<details>
<summary><b>Q54. How should a remote MCP server handle authentication and authorization, and what is the confused deputy problem here?</b></summary>

**Answer:** A confused deputy is a component with legitimate authority that is tricked into using it for someone who lacks that authority, for example an MCP server on a powerful service account returning data the calling user should never see. The fix is to act with the user's identity: OAuth 2.1 with PKCE, the MCP server acting as a resource server that validates short-lived, audience-bound tokens, and a separate delegated token (for example via token exchange) for downstream APIs so the system of record enforces the user's permissions. Token passthrough, forwarding the client's token straight to downstream APIs, is forbidden by the spec because it breaks audience checks, policy enforcement and audit. Scopes give coarse authorization (`tickets:write`); row-level authorization must use the user's identity at the data source. The auth section has changed across spec revisions, so know the principles and check the current text.

**Key points to mention:**
- Flow: 401 with protected resource metadata, authorization-server discovery, authorization code with PKCE, resource indicators to bind the audience.
- At company scale, put an MCP gateway in front: auth, per-user catalog filtering, policy by risk tier, rate limits, audit.

**Deep dive:** [Authentication and authorization for remote servers](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#authentication-and-authorization-for-remote-servers)

</details>

<details>
<summary><b>Q55. MCP vs A2A: what is the difference and when would you use each?</b></summary>

**Answer:** MCP connects an LLM application to tools and data that its own model drives; the callee is a transparent capability with a schema. A2A (Agent2Agent) connects an agent to another autonomous agent; the callee is an opaque peer with its own reasoning, advertised through an Agent Card (skills, endpoint, auth), and work is modeled as tasks with a lifecycle that may run long, stream updates or request input. Use MCP for tools, and A2A for delegating outcomes to agents owned by other teams or vendors. They compose: an agent can expose itself over A2A and use MCP internally. A2A does not solve reliability, conflicting decisions or injection; you still need contracts, single owners, validation and trace propagation.

**Key points to mention:**
- A2A raises the same trust issues plus harder debugging, since the peer's reasoning is opaque.
- Many "multi-agent" problems are better as one agent with good tools.
- Specs evolve; name durable concepts, not field names.

**Deep dive:** [Beyond MCP: OpenAPI-based tools and agent-to-agent protocols](../09-agents/07-model-context-protocol-and-tool-ecosystems.md#beyond-mcp-openapi-based-tools-and-agent-to-agent-protocols), [Inter-agent protocols: A2A in brief](../09-agents/05-multi-agent-systems.md#inter-agent-protocols-a2a-in-brief)

</details>

## Production reliability

<details>
<summary><b>Q56. How do you make agents reliable enough for production?</b></summary>

**Answer:** Treat the LLM as an unreliable, manipulable planner and wrap it in deterministic engineering. Keep the path short and the tools narrow (a workflow with a few LLM decisions beats a free-form agent for well-defined tasks), orchestrate as an explicit state machine so the runtime owns control flow, retries and checkpoints, and enforce hard budgets with graceful termination. Validate state between steps against systems of record, make side effects idempotent with a durable step journal, put irreversible or high-value actions behind deterministic policy and human approval, sandbox code and restrict egress, and write everything to an audit trail. Measure end-state success and consistency (pass^k) on an evaluation set that grows with every incident. Roll out progressively: read-only, shadow mode, approvals on, then automate low-risk actions as data proves them safe.

**Key points to mention:**
- Name the failure modes and map each to a control: loops, wrong tool, bad args, hallucinated results, goal drift, premature termination, cost blowups, duplicate side effects, unauthorized actions.
- Metrics: end-state success, pass^k, unauthorized actions executed (must be 0), cost per successful task, p95 latency, escalation rate.

**Deep dive:** [How agents fail: a taxonomy of failure modes](../09-agents/08-agent-reliability-security-and-prompt-injection.md#how-agents-fail-a-taxonomy-of-failure-modes), [Putting it all together in production](../09-agents/08-agent-reliability-security-and-prompt-injection.md#going-deeper-putting-it-all-together-in-production)

</details>

<details>
<summary><b>Q57. How do you set iteration and token limits?</b></summary>

**Answer:** From data, not intuition. Run the evaluation set, look at the distribution of steps, tokens, wall-clock time and cost for successful runs, and set each limit a bit above the p99 for that task type; if successful runs have a p95 of 9 steps and a p99 of 12, a limit around 15 is reasonable, while 50 just makes failures expensive. The runtime enforces them before each model call, together with per-call output token caps, per-tool timeouts, per-tool counts (for example at most one `create_return` per run) and a loop detector. Several budgets catch different failures, and cost grows roughly quadratically with steps: with $P = 2{,}000$ and $k = 500$, 10 steps is 42,500 input tokens and 40 steps is 470,000 (illustrative). On breach, terminate gracefully: record the reason, summarize completed work from the action log, and escalate or ask the user.

**Key points to mention:**
- Trade-off: too tight cuts off legitimate long tasks (replay successful runs to measure false stops); too loose burns money.
- Limits live in config, vary by task type and tier, and are re-derived when model, prompt or tools change.
- A spike in budget stops often means a downstream tool is failing.

**Deep dive:** [Hard limits: step, token, time and cost budgets](../09-agents/08-agent-reliability-security-and-prompt-injection.md#hard-limits-step-token-time-and-cost-budgets)

</details>

<details>
<summary><b>Q58. How do you validate intermediate state?</b></summary>

**Answer:** At four layers. Schemas check shape and types (Pydantic or JSON Schema, optionally constrained decoding); preconditions before every action are checked against the system of record (the order exists, belongs to this user, the amount is refundable); invariants must always hold (refunds never exceed payments); and postconditions after actions confirm the effect (the refund actually shows as succeeded). For example, a proposed refund of 95.00 on an order with 80.00 paid is rejected with "amount 95.00 exceeds refundable 80.00". On failure, return a specific error to the model with a small retry cap, ask the user for missing information, or stop and escalate. Never validate against the model's own summary of the world: query the database or service, and re-check inside the executor's transaction to avoid time-of-check to time-of-use races.

**Key points to mention:**
- Valid JSON is not a valid action; constrained decoding solves only the first layer.
- Postconditions catch hallucinated results and premature "done".
- Validate tool outputs too (schema, size, status), since bad results poison later steps.
- Track validation failures per layer and tool as a regression signal.

**Deep dive:** [Validating intermediate state](../09-agents/08-agent-reliability-security-and-prompt-injection.md#validating-intermediate-state)

</details>

<details>
<summary><b>Q59. How do you implement retries without repeating side effects?</b></summary>

**Answer:** Make every side-effecting call idempotent with a key derived from runtime-controlled IDs (run ID plus step ID), never from model-generated text, which changes on regeneration. Record each step in a durable journal (started, then done with the result), checkpoint after every side effect, and on resume replay completed steps from the journal instead of re-executing them or re-asking the model. Steps caught mid-flight are retried with the same recorded arguments and key so the downstream service deduplicates. Retry only transient errors (timeouts, 429, 503) with exponential backoff and jitter, and treat a timeout after send as "maybe happened". For multi-step workflows use a saga: compensations run in reverse and are themselves idempotent, and irreversible steps go last, often behind approval.

**Key points to mention:**
- If a downstream API lacks idempotency: wrap it in your own idempotent service, check-then-act on a unique reference, or mark it no-auto-retry and escalate.
- Durable execution (Temporal, LangGraph checkpointers) implements the journal.
- Fault injection in CI (kill between call and checkpoint) plus daily reconciliation; duplicate count must be zero.

**Deep dive:** [Retries without repeating side effects](../09-agents/08-agent-reliability-security-and-prompt-injection.md#retries-without-repeating-side-effects)

</details>

## Evaluation and observability

<details>
<summary><b>Q60. How would you evaluate task completion and agent reliability?</b></summary>

**Answer:** Build a scenario suite in which each test seeds a sandboxed environment, runs the agent end to end (with a simulated user if it is conversational), and grades the final state with deterministic checks: the right database rows, files or API calls, and nothing extra. That gives task success rate, plus partial completion (fraction of subgoal checks passed) for debugging. For reliability, run each scenario several times and report pass^1 and pass^k (all k runs succeed): at per-run success 0.8, pass^4 is about 0.41, and the gap between pass^1 and pass^k separates capability gaps from flakiness. Add fault-injection slices (timeouts, empty results, malformed JSON) to measure error recovery, and adversarial slices for safety. Run a core slice on every PR, the full suite with pass^k nightly or before release, and turn every production failure into a regression scenario.

**Key points to mention:**
- Never trust the agent's own claim of success; check the world.
- Statistics: 50 tasks at 84% success is about plus or minus 10 points at 95%, so compare versions paired by task.
- Pitfalls: shared environments leaking state, mocks that never fail, overfitting to the suite (keep a held-out slice).
- Online, combine hard outcomes (ticket not reopened), escalations, feedback and sampled judges.

**Deep dive:** [Building an agent evaluation set](../09-agents/09-evaluating-and-observing-agents.md#building-an-agent-evaluation-set), [Evaluating task completion and agent reliability](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#13-evaluating-task-completion-and-agent-reliability)

</details>

<details>
<summary><b>Q61. What are the right evaluation metrics for agent success?</b></summary>

**Answer:** A small scorecard, not one number. Outcome: task success verified on final state, plus partial completion. Process: tool-selection and argument accuracy, error recovery rate, steps per task and the share of runs hitting the step limit. Efficiency: cost and latency per task, especially cost per successful task ($\overline{\text{cost}} / \hat{p}$: 0.09 dollars per run at 84% success is about 0.107 per resolved case, illustrative). Safety: policy violations (executed must be zero, attempted tracked as a leading indicator), injection attack success rate, and escalation precision and recall. Reliability: pass^k across repeated runs. Pair every metric with a counter-metric so it cannot be gamed: an agent that escalates everything has zero violations and is useless.

**Key points to mention:**
- If only one: task success on verified final state, measured as pass^k on a representative suite.
- Slice by intent, difficulty and customer tier.
- Tie offline metrics to business outcomes (resolution without reopen, CSAT) and check that they predict them.

**Deep dive:** [Agent metrics, from basic to advanced](../09-agents/09-evaluating-and-observing-agents.md#agent-metrics-from-basic-to-advanced), [Evaluation metrics for agent success](../09-agents/08-agent-reliability-security-and-prompt-injection.md#evaluation-metrics-for-agent-success)

</details>

<details>
<summary><b>Q62. What is the difference between pass@k and pass^k, and when do you use each?</b></summary>

**Answer:** pass@k is the probability that at least one of k attempts succeeds; pass^k is the probability that all k succeed. With independent attempts and per-attempt success $p$, pass@k $= 1 - (1-p)^k$ and pass^k $= p^k$: at $p = 0.8$ and $k = 4$, that is 0.9984 versus 0.4096. pass@k fits generate-and-verify settings, such as code with tests, where you can automatically pick a passing candidate. pass^k fits customer-facing agents, where every run must work, so it measures consistency. From $n$ trials with $c$ successes, the unbiased estimators are $1 - \binom{n-c}{k}/\binom{n}{k}$ and $\binom{c}{k}/\binom{n}{k}$, computed per task and averaged.

**Key points to mention:**
- Example: 4 trials, 3 successes gives pass^2 = 3/6 = 0.5 while pass@2 = 1.0.
- pass@k was popularized for code (Chen et al., 2021); pass^k was introduced for agents in tau-bench (Yao et al., 2024).
- Reporting pass@k for an agent with no production verifier overstates what users see.

**Deep dive:** [Reliability: pass@k vs pass^k](../09-agents/09-evaluating-and-observing-agents.md#reliability-passk-vs-passk)

</details>

<details>
<summary><b>Q63. When would you grade with deterministic checks versus an LLM judge?</b></summary>

**Answer:** Use deterministic checks whenever the outcome is observable in the world (database rows, files, API calls, test results), because they are exact, cheap and repeatable; they should be the source of truth for success and for CI gates. Use LLM judges for qualities code cannot check: tone, clarity, whether claims are supported by tool outputs, whether a clarifying question was sensible. A good judge uses binary criteria, sees the full trajectory including tool outputs, cites evidence (step numbers), returns structured JSON and ideally is a different strong model from the agent. Calibrate it on 50 to 200 human-labeled trajectories (agreement and Cohen's kappa) and re-calibrate when the judge model changes. Safety outcomes are always checked deterministically.

**Key points to mention:**
- Judge biases: verbosity, position, self-preference, confident text, missed errors deep in long traces.
- Deterministic pitfall: over-strict checks (exact string match on an email body) cause false failures.
- Hybrid: an agent-as-a-judge that queries the final state with tools.

**Deep dive:** [Grading: deterministic final-state checks vs LLM judges](../09-agents/09-evaluating-and-observing-agents.md#grading-deterministic-final-state-checks-vs-llm-judges), [LLM Evaluation Metrics and LLM-as-a-Judge](../05-llms/10-llm-evaluation-metrics-and-llm-as-a-judge.md)

</details>

<details>
<summary><b>Q64. How do you trace tool calls and model decisions across multiple model and tool calls?</b></summary>

**Answer:** Use distributed tracing, ideally OpenTelemetry with the GenAI semantic conventions. Each task is one trace with a root agent span; every LLM call, tool call, retrieval, guardrail check, graph node and handoff is a child span with a parent span ID, timing, status and attributes. Beyond latency and tokens, record decisions as structured attributes: model and version, prompt version, chosen tool and redacted arguments, finish reason, router label, conditional edge taken, validation errors, approvals and who gave them, plus the thread and checkpoint IDs so a bad trace links to replayable state. The trace ID propagates across processes with the W3C `traceparent` header (or queue message metadata) and is stored with feedback and tickets. Export asynchronously in batches to a platform such as Langfuse, LangSmith or Phoenix, with PII redacted before export, and never let a tracing failure fail the run.

**Key points to mention:**
- Keep tool outputs in traces, or you cannot tell model hallucination from bad tool data.
- Tail sampling: keep all errors, slow and negative-feedback traces, plus a random sample.
- Broken propagation (thread pools, queues) shows up as orphan spans; test tracing in CI.
- Pitfall: free-text logs only; you cannot query "runs where the router chose refund but intent was FAQ".

**Deep dive:** [Observability: tracing every decision](../09-agents/09-evaluating-and-observing-agents.md#observability-tracing-every-decision), [Tracing execution across agents, models and tools](../09-agents/05-multi-agent-systems.md#tracing-execution-across-agents-models-and-tools), [Tracing tool calls and model decisions](../09-agents/04-orchestration-with-langgraph-and-state-machines.md#9-tracing-tool-calls-and-model-decisions)

</details>

<details>
<summary><b>Q65. An agent works in the demo but fails 30% of the time in production. How do you debug it?</b></summary>

**Answer:** Pull full traces of failing runs and find the first bad decision in each. Classify failures: bad information from tools or retrieval, a bad decision by the model, bad execution by a tool, or a limit firing. Look at trajectory metrics (steps per run, wrong-tool rate, invalid-argument rate, which stop condition fired) and replay failing steps with recorded observations to test prompt or tool-description fixes in isolation. Common root causes are production inputs that differ from demo inputs, larger or messier tool outputs, too many tools, missing validators and a changed model version. Fix the most common class first, add those cases to a regression suite that runs each scenario several times, and compare success and cost per task before and after each fix so one fix does not break other cases.

**Key points to mention:**
- A failure taxonomy (wrong tool, wrong args, hallucinated result, premature stop, loop, policy violation) prioritizes work.
- Redact PII when turning traces into tests; confirm the test fails before the fix and passes after.
- A related trap: 95% on the eval set but users complain usually means a measurement gap (final message instead of end state, single runs, unrepresentative cases).

**Deep dive:** [How to debug an agent run](../09-agents/01-what-is-an-ai-agent.md#how-to-debug-an-agent-run), [Turning production failures into regression tests](../09-agents/09-evaluating-and-observing-agents.md#turning-production-failures-into-regression-tests)

</details>

<details>
<summary><b>Q66. After a model upgrade, success is unchanged but cost per task rose 60% and p95 latency doubled. How do you investigate?</b></summary>

**Answer:** Compare traces between versions on the same scenarios, breaking cost and latency down per span: LLM calls per task, tokens per call, tool calls per task and retries. Typical culprits are more steps (extra lookups, loops, re-planning), longer outputs or billed reasoning tokens, more retries from schema errors with the new model, or lost prompt caching because the prompt prefix changed. Diff the steps-per-task distribution, the share of runs hitting max steps, output tokens per call, tool error and retry counts and cache hit rate, and read a few flipped traces side by side. Fixes include adjusting prompts and tool descriptions for the new model, setting a reasoning-effort or output-length budget, adding a step budget and restoring a stable prefix for caching. Prevent recurrence by making cost and latency per task release-gate metrics alongside success.

**Key points to mention:**
- Watch for silent provider updates behind model aliases; pin versions.
- Cost per successful task is the comparison metric.

**Deep dive:** [Cost of evaluation and tracing at scale](../09-agents/09-evaluating-and-observing-agents.md#cost-of-evaluation-and-tracing-at-scale), [Speculative Decoding, Prompt Caching and Other Speedups](../06-llm-inference/05-speculative-decoding-caching-and-other-speedups.md)

</details>

## Case study: designing a production agent

<details>
<summary><b>Q67. Design an autonomous procurement or support agent.</b></summary>

**Answer:** Start with 5 minutes of scope: users and roles, autonomy level per action (assist, draft, act with approval, act within limits), systems touched, volume, compliance, and success metrics (task success, zero executed policy violations, escalation rate, cost per task, p95 completion time). The shaping principle is "the LLM proposes, deterministic code disposes": split the system along a trust boundary so the LLM understands the request, retrieves context and proposes a typed plan, while deterministic services authenticate the user, enforce policy, execute with idempotent sagas, route approvals and write an append-only audit trail. Run the agent inside a state graph (Intake, Retrieve, Plan, Clarify, PolicyCheck, AwaitApproval, Execute, Confirmed or Compensated) whose only edge into Execute comes from the policy check, which issues a signed grant bound to the exact plan. Then walk one request end to end, name failure modes, and finish with evaluation, monitoring and cost.

**Key points to mention:**
- The LLM appears only in planning steps and the LLM gateway; everything that can spend money is plain code.
- Trade-off: a state graph is safer and cheaper but less flexible than an open loop (odd requests go to Clarify or a human).
- Volunteer failure modes: duplicate orders on timeouts, injection via vendor text, two requests draining one budget, approval fatigue, silent change after a model upgrade.
- Pacing for 45 minutes: scope 5, architecture 5, one request walk-through 20, failure modes 10, evaluation and cost 5.

**Deep dive:** [Case Study: Designing a Production Agent](../09-agents/10-case-study-designing-a-production-agent.md), [The one principle that shapes everything](../09-agents/10-case-study-designing-a-production-agent.md#the-one-principle-that-shapes-everything)

</details>

<details>
<summary><b>Q68. In that design, how do you handle input validation and authentication?</b></summary>

**Answer:** Authentication proves who the user is; authorization decides what they may do, and they are separate steps. The user signs in through company SSO (for example OIDC); the gateway verifies the token and attaches a trusted identity (user id, roles, cost centers) that propagates through every service, often as short-lived on-behalf-of tokens (OAuth 2.0 Token Exchange), so tools see "the agent acting for Maya", not "the agent". Identity never comes from model output: a tool like `create_po(user_id=...)` lets an injection choose a different user. Input validation rejects garbage before it costs tokens: a typed schema that forbids unknown fields, length limits, attachment type and size allowlists with malware scanning, and per-user and per-cost-center rate limits. Injection screening with a classifier is a signal for monitoring, not a defense; the real defense is that the policy layer ignores what the text says.

**Key points to mention:**
- One authorization check already happens here: may this user charge this cost center at all? That is a lookup, not an LLM question.
- For a support agent, add step-up customer verification before account changes.

**Deep dive:** [Step 1: Input validation and authentication](../09-agents/10-case-study-designing-a-production-agent.md#step-1-input-validation-and-authentication)

</details>

<details>
<summary><b>Q69. How do you do context retrieval for an agent that acts, such as a procurement agent?</b></summary>

**Answer:** Different context needs different retrieval: policies are long documents (hybrid RAG with reranking, re-indexed on change); vendor catalogs need semantic search on descriptions to find candidates, then a live API call for price and stock; past orders ("same model as last time") are a SQL query scoped to the requester's cost centers; budgets are never retrieved for the LLM at all, because the policy engine reads them directly and transactionally. Retrieval must be permission-aware: filter by ACL metadata before ranking and truncation, never only in the prompt, or the agent leaks data through its answers even if it never places a bad order. Keep document ids and versions so they become evidence in the audit log. Two agent-specific rules: prices in retrieved text are hints, not facts (the executor re-fetches the live price), and retrieved text is data, not instructions.

**Key points to mention:**
- Not everything belongs in a vector database; candidates who say "embed everything" miss this.
- Filtering after top-k can leave too few results; pre-filter in the vector store.

**Deep dive:** [Step 2: Context retrieval with permission-aware RAG](../09-agents/10-case-study-designing-a-production-agent.md#step-2-context-retrieval-with-permission-aware-rag), [Secure Multi-Tenant RAG](../08-rag/13-secure-multi-tenant-rag.md)

</details>

<details>
<summary><b>Q70. How do you handle planning and tool selection safely in that agent?</b></summary>

**Answer:** Give the model as few tools as possible, each as narrow as possible (least privilege, the opposite of what OWASP calls "Excessive Agency"). The LLM gets read-only tools (`search_policies`, `search_catalog`, `get_past_orders` scoped by the server), `ask_user` for ambiguity, and `propose_plan`, which produces structured data with no side effects; the policy check is called by the orchestrator only, and write tools such as `create_po` live in the execution service and are never exposed to the model. The plan is typed structured output with SKUs, quantities, cost center, rationale and evidence ids, and deliberately no price and no user id: prices come from the live catalog, identity from the session. Planning runs inside a state graph with budgets (for example at most 5 planning iterations, a wall-clock limit and a token cost cap) that route to a human on breach. Generic tools such as `run_sql(query)` or `call_erp_api(endpoint, payload)` turn every injection into arbitrary actions.

**Key points to mention:**
- Fewer fields the LLM controls means fewer things that can go wrong.
- The graph makes it structurally impossible to skip the policy check.

**Deep dive:** [Step 3: Planning and tool selection with a state graph](../09-agents/10-case-study-designing-a-production-agent.md#step-3-planning-and-tool-selection-with-a-state-graph)

</details>

<details>
<summary><b>Q71. How do you implement business rules and permission checks? Why not put them in the system prompt?</b></summary>

**Answer:** In a deterministic policy layer (policy-as-code in Python or OPA/Rego) that takes the proposed plan plus trusted facts (who, how much, which vendor, which budget) and returns ALLOW, REQUIRE_APPROVAL or DENY with reasons and a policy version. Rules include approved vendors per category, blocked categories, remaining budget, role spending limits, split-order detection, separation of duties, a hard ceiling and price-freshness re-approval, and the engine fails closed when a fact is missing. Worked example: a manager with a 5,000 auto-limit orders 5 laptops at a live price of 1,450, so $5 \times 1{,}450 = 7{,}250$, which fits the 20,000 budget but exceeds the limit, giving REQUIRE_APPROVAL by her director. Rules about money and permissions need guarantees, and an LLM only gives probabilities: it can misread numbers, miss cumulative limits or be talked out of a rule by injected text. Code rules can be unit-tested at boundaries (exactly 5,000 vs 5,000.01), reviewed by finance and versioned per decision.

**Key points to mention:**
- The prompt may still describe the rules so plans pass more often, but enforcement lives in code; generate that summary from the policy config to avoid drift.
- Record the policy version with every decision to answer "which rule allowed this?" months later.

**Deep dive:** [Step 4: The deterministic policy layer](../09-agents/10-case-study-designing-a-production-agent.md#step-4-the-deterministic-policy-layer), [Model proposes, code disposes](../09-agents/08-agent-reliability-security-and-prompt-injection.md#model-proposes-code-disposes)

</details>

<details>
<summary><b>Q72. How do you handle execution and rollback or compensation for an agent that places orders?</b></summary>

**Answer:** The execution service is the only component holding credentials; it verifies a signed, plan-bound grant, then runs a durable saga. A saga is a sequence of local transactions across systems in which each reversible step has a compensating action, run in reverse on failure: reserve budget, create PO, submit to vendor, with compensations cancel PO and release budget. One big database transaction is impossible because the budget service, ERP and vendor APIs share no ACID transaction. Every write carries a stable idempotency key per logical step (for example `request_id:create_po`), so retries after a timeout return the original result instead of ordering twice, and compensations are idempotent too. Put the irreversible pivot step last and approvals before the saga starts; compensation is a new offsetting action, not erasure, so history remains for audit.

**Key points to mention:**
- Orchestrated sagas (one coordinator) are easier to audit than choreographed ones.
- Use a durable workflow engine so the saga resumes after crashes; a transactional outbox when publishing events with database writes.
- Compensation logic is deterministic, never improvised by the LLM.
- If an API lacks idempotency support, search by your reference number before writing and reconcile.

**Deep dive:** [Step 5: Execution, idempotency and the saga pattern](../09-agents/10-case-study-designing-a-production-agent.md#step-5-execution-idempotency-and-the-saga-pattern)

</details>

<details>
<summary><b>Q73. How do you set human-in-the-loop approval thresholds without overwhelming approvers?</b></summary>

**Answer:** Combine an expected-loss rule with risk flags. Require approval when the expected loss of acting alone, $p \times C$, exceeds the cost of a review $R$, which gives a break-even order value $C^* = R / p$; with $p = 0.02$ and $R = 10$ dollars, orders above about 500 deserve review, so a 400 order auto-approves ($8 < 10$) and a 7,250 order does not ($145 > 10$). If reviews catch only a fraction $q$ of mistakes, the break-even rises to $R / (q \times p)$. Add flags the formula ignores: irreversibility, new vendors or categories, injection signals, and hard DENY rules with no approval path. Then check capacity: 2,000 requests a day at 15% approval is 300 approvals, 6 per approver across 50 people, which is fine, while 60% would drown approvers into rubber-stamping.

**Key points to mention:**
- Approvals are bound to the plan hash, expire, and are re-checked against live price and budget.
- Fatigue signals: approval rate near 100% and review times of a few seconds.
- As measured error rates fall, thresholds can rise: that is how autonomy expands safely.

**Deep dive:** [Step 6: Human-in-the-loop approval](../09-agents/10-case-study-designing-a-production-agent.md#step-6-human-in-the-loop-approval)

</details>

<details>
<summary><b>Q74. What do you put in an agent's audit log, and how is it different from tracing?</b></summary>

**Answer:** For every proposed action: timestamps and run and step IDs; who (requester, approver, the service identity and the user it acts for); what (tool, validated arguments, result, external IDs such as the PO number); why (the plan, the policy decision with reasons and rule version, any approval record); which evidence (document ids with versions, because policies and catalogs change); and which versions (model, prompt, tool set, policy, code), all linked by one trace ID. The log is append-only, complete rather than sampled, access-controlled and tamper-evident through a hash chain anchored in a separate protected store. Tracing is debugging telemetry for engineers and can be sampled and redacted; the audit log is the record for accountability and investigations. Denials and blocked attempts are logged too, because they are an early warning of injection campaigns.

**Key points to mention:**
- Model and prompt versions let you find and replay every action a version took after an incident.
- Log structured rationale; free-form model reasoning is not the justification of record.
- Apply redaction, encryption and retention rules per legal requirements.

**Deep dive:** [Step 7: Audit logging and traceability](../09-agents/10-case-study-designing-a-production-agent.md#step-7-audit-logging-and-traceability), [Audit logging](../09-agents/08-agent-reliability-security-and-prompt-injection.md#audit-logging)

</details>

<details>
<summary><b>Q75. How do you handle evaluation, monitoring and cost control for that agent?</b></summary>

**Answer:** Before launch: unit tests for deterministic components (policy rules at boundaries, routers, validators), a sandboxed scenario suite that checks final system state (PO with the right SKU and quantity, budget reduced correctly), and an adversarial suite, each run several times per scenario; at 92% per-trial success, $0.92^3 \approx 0.78$, so consistency is lower than the average suggests. After launch: shadow mode, staged autonomy, and dashboards from traces plus ERP data for success, executed vs attempted violations (any executed violation pages on-call), escalation and approval rates, compensation rate, tool errors and latency, injection classifier hits, and tokens, steps and cost per task. Task cost is the sum over LLM calls of input tokens times input price plus output tokens times output price; routing two simple calls to a small model cut an illustrative 0.090 to 0.0735 dollars per task (only about 18%, because planning dominates), so better retrieval (fewer re-plans) and prompt caching are bigger levers. Add per-task budgets, per-user quotas and a kill switch that drops the whole system to draft-only mode.

**Key points to mention:**
- LLM-as-a-judge may grade explanation quality, never safety outcomes.
- Every model, prompt, tool or policy change reruns the suite as a regression gate.
- Human processing time usually dwarfs token cost; do not optimize cost at the expense of safety.

**Deep dive:** [Step 8: Evaluation](../09-agents/10-case-study-designing-a-production-agent.md#step-8-evaluation), [Step 9: Monitoring and cost control](../09-agents/10-case-study-designing-a-production-agent.md#step-9-monitoring-and-cost-control)

</details>

<details>
<summary><b>Q76. Senior-level challenge: an agent completes 92% of tasks successfully but occasionally executes an unauthorized action. Would you improve the prompt, add a validation agent, or change the architecture? Why?</b></summary>

**Answer:** Change the architecture. First separate the two problems: the 8% failures are a quality problem to improve iteratively, while unauthorized actions are a control problem whose acceptable rate is zero under adversarial pressure. A better prompt or a validator LLM only makes a bad action less likely, and a validator shares the planner's weaknesses (same context, same injections, correlated errors) while adding cost and latency. Illustratively, 10,000 tasks a day at 0.1% is 10 incidents a day; a 90%-effective validator still leaves about 1 a day, around 365 a year, and the survivors are the adversarially optimized ones. So: contain immediately (kill switch to draft-only or forced approval for the action class), root-cause from the audit log, then move authorization into deterministic code: identity from the session, least-privilege narrow tools, a policy engine at the executor that fails closed, signed plan-bound grants, user-scoped short-lived credentials, approvals for irreversible actions and egress restrictions. For every rule the policy encodes, the action becomes impossible, not just unlikely; prompt work and an optional reviewer model then serve as defense in depth for quality, never as the boundary.

**Key points to mention:**
- Root causes to classify: excessive privilege, rule only in the prompt, trusted LLM arguments (identity, price), injection, or a semantic mistake mislabeled as unauthorized.
- Cost of the fix: more engineering, less flexibility and maybe a first dip in success, because some of the 92% was success the system was never allowed to have.
- Proof: every incident becomes a regression and adversarial test; "unauthorized executed = 0" is a release gate; attempted violations are a leading indicator.
- If a rule cannot be codified: narrow the action, require approval for that class, or use a validator whose only power is to escalate.
- Nuance: policy blocks unauthorized actions, not authorized-but-wrong ones, which need preconditions, thresholds and reversibility.

**Deep dive:** [The senior challenge (reliability chapter)](../09-agents/08-agent-reliability-security-and-prompt-injection.md#the-senior-challenge-92-percent-success-with-occasional-unauthorized-actions), [The senior challenge (case study)](../09-agents/10-case-study-designing-a-production-agent.md#the-senior-challenge-92-success-with-occasional-unauthorized-actions)

</details>

## Rapid-fire round

| Question | Answer |
|---|---|
| What is the one-line definition of an agent? | An LLM in a loop that chooses actions, calls tools, observes results and continues until done or stopped by code. |
| Is one function call an agent? | No; it becomes agentic when the model picks unplanned next steps from observations. |
| $0.95^{10}$ is roughly? | 0.60, which is why fewer model-decided steps means higher reliability. |
| "Workflow outside, agent inside" means? | Deterministic intake, auth and approval in code, with an agentic node only for the open-ended part. |
| Does temperature 0 make an agent deterministic? | No; numerics, batching, tools and environment still vary. |
| Where must loop limits live? | In orchestration code, never only in the prompt. |
| How do you set `max_steps`? | A bit above the p99 step count of successful eval runs. |
| Input tokens for an $n$-step loop grow how? | Roughly $nP + s\,n(n-1)/2$, quadratic in steps. |
| What does ReAct interleave? | Thought, action (tool call) and observation. |
| When does reflection help? | When critique is grounded in an external signal such as tests or errors. |
| Is a tool call in model output an executed action? | No, it is a proposal until your executor returns a real result. |
| Who validates tool arguments? | The application, always, in layers from schema to authorization. |
| Does strict-mode constrained decoding replace server validation? | No; it guarantees shape, not truth or permission. |
| Which failures should be retried? | Transient ones (timeouts, 429, 503), with backoff and jitter, never blind 400s or 403s. |
| What is a timed-out write? | An unknown outcome: retry only with the same idempotency key or reconcile. |
| Is exactly-once delivery achievable? | No; at-least-once plus idempotency gives exactly-once effects. |
| Retry amplification with 3 layers of 3 attempts? | 27 requests; retry in one layer only. |
| Where does identity for a tool call come from? | The authenticated session, injected by the server, never model arguments. |
| What is the confused deputy problem? | A privileged component tricked into using its authority for someone who lacks it. |
| What is the lethal trifecta? | Private data plus untrusted content plus external communication in one agent. |
| Jailbreak vs indirect injection in one line? | Jailbreak: the user attacks the model; injection: a third party attacks the user via content. |
| Is spotlighting a security boundary? | No, it reduces attack success but is only a signal. |
| What should a human approval be bound to? | The exact canonical arguments, with a signature and an expiry. |
| What is a LangGraph reducer? | A per-field `(old, new) -> merged` function controlling how updates combine. |
| What does a checkpointer give you? | Durable state per super-step: crash recovery, interrupts, time travel. |
| What runs when you resume after `interrupt()`? | The interrupted node again from the top, so keep side effects below or idempotent. |
| LangGraph vs Temporal in one line? | Snapshot per step and re-run the node vs event history replay that skips completed activities. |
| Default choice: single agent or multi-agent? | Single agent with good tools, unless evals prove multi-agent wins. |
| Supervisor vs mesh channel counts for 5 agents? | Star has 4, mesh has 10. |
| How do you avoid conflicting agent decisions? | Agents propose, one owner decides with deterministic arbitration rules. |
| Which memory facts should be extracted? | Facts from the user's own turns, never from tool outputs or documents. |
| How is agent memory different from RAG? | Memory is written by the agent per user, so it needs write policy, updates, isolation and deletion. |
| What does MCP turn $M \times N$ integrations into? | $M + N$ protocol implementations. |
| MCP tools, resources, prompts: who controls each? | Model, application and user respectively. |
| Is MCP token passthrough allowed? | No; use audience-bound tokens and delegated tokens downstream. |
| MCP vs A2A in one line? | MCP connects an agent to tools; A2A connects an agent to another opaque agent. |
| pass@k vs pass^k? | At least one of k succeeds vs all k succeed; agents need pass^k. |
| Headline agent metric if you get only one? | Task success on verified final state, measured as pass^k. |
| Audit log vs trace? | Audit log is complete, append-only and for accountability; traces are sampled debugging telemetry. |
| Expected-loss approval break-even? | Order value $C^* = R / p$, for example 10 / 0.02 = 500. |

## How to practice

- **Answer out loud first.** Read only the question, close your notes and answer as if the interviewer is waiting. Open the answer only afterwards.
- **Time yourself to 2 minutes.** A strong answer covers definition, mechanism, trade-off and one concrete example or number in that time; if you run long, trim the definition, not the trade-off.
- **Explain it to a non-expert.** If you cannot explain idempotency keys or pass^k to a friend outside tech, you do not own the idea yet.
- **Then go deeper.** Use the key points as likely follow-ups and practice answering two levels down (for example: "and if the downstream API has no idempotency support?").
- **Drill the design questions end to end.** Practice Q67 to Q76 as one 45-minute mock with a whiteboard sketch of the trust boundary and the state graph.
- **Last week before the interview:** do the rapid-fire table daily, and re-read the deep-dive sections for any question you could not answer in 2 minutes.
