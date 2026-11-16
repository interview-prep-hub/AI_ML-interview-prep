# Orchestrating Agents with LangGraph and State Machines

> **Level:** Beginner to Advanced · **Study time:** ~100 min · **Prerequisites:** [What Is an AI Agent?](01-what-is-an-ai-agent.md), [Tool Calling in Depth](02-tool-calling-in-depth.md), [Agent Design Patterns](03-agent-design-patterns.md)

## Table of contents

- [Why this matters](#why-this-matters)
- [1. Three ways to orchestrate: a loop, a chain, a graph](#1-three-ways-to-orchestrate-a-loop-a-chain-a-graph)
- [2. When a state graph beats a simple chain or loop](#2-when-a-state-graph-beats-a-simple-chain-or-loop)
- [3. Build a tiny state machine engine from scratch](#3-build-a-tiny-state-machine-engine-from-scratch)
- [4. LangGraph core concepts: state, nodes, edges, reducers](#4-langgraph-core-concepts-state-nodes-edges-reducers)
- [5. Persistence: checkpointers, threads and crash recovery](#5-persistence-checkpointers-threads-and-crash-recovery)
- [6. Human in the loop: interrupts, resuming and time travel](#6-human-in-the-loop-interrupts-resuming-and-time-travel)
- [7. Branching, retries, error nodes and recovery](#7-branching-retries-error-nodes-and-recovery)
- [8. Durable execution engines (Temporal-style) and how they relate](#8-durable-execution-engines-temporal-style-and-how-they-relate)
- [9. Tracing tool calls and model decisions](#9-tracing-tool-calls-and-model-decisions)
- [10. Implementing a production agent: a complete example graph](#10-implementing-a-production-agent-a-complete-example-graph)
- [11. Going deeper: running agent graphs in production](#11-going-deeper-running-agent-graphs-in-production)
- [12. Other frameworks and how to evaluate one](#12-other-frameworks-and-how-to-evaluate-one)
- [13. Evaluating task completion and agent reliability](#13-evaluating-task-completion-and-agent-reliability)
- [Common mistakes and misconceptions](#common-mistakes-and-misconceptions)
- [Interview questions and model answers](#interview-questions-and-model-answers)
- [Key takeaways](#key-takeaways)
- [Further reading](#further-reading)
- [Next](#next)

## Why this matters

A demo agent is a `while` loop around a model call. A production agent has to survive a server restart halfway through a refund, wait two days for a manager to click "approve", retry a flaky payment API without paying twice, and explain to an auditor exactly why it did what it did. Those requirements are about **orchestration**, not about the model. Frameworks such as LangGraph exist because teams kept rebuilding the same machinery: explicit control flow, saved state, pause and resume, and step-level tracing. Interviewers for senior AI roles ask about this because it separates people who have shipped agents from people who have only prompted them.

---

## 1. Three ways to orchestrate: a loop, a chain, a graph

**Orchestration** is the code that decides *which step runs next*, *what data each step sees*, and *what happens when a step fails*. The model reasons; the orchestrator is the traffic controller.

> **Analogy:** Think of a hospital. Doctors (the models and tools) do the skilled work. The patient chart and the admission process (the orchestrator) decide who sees the patient next, record every decision, and make sure that if a doctor goes home mid-shift, the next doctor can pick up exactly where things stopped.

Throughout this chapter we use one running example: **a support agent for an online store** that answers policy questions and processes refunds. Our customer writes: *"Please refund order A1001, it arrived damaged."* Order A1001 cost $250, and our policy says refunds over $100 need a human approver. The policy lives in a tiny **4-document corpus**:

1. "Refunds: damaged items are refundable within 30 days of delivery."
2. "Refunds over $100 require approval by a support lead."
3. "Shipping: standard delivery takes 3-5 business days."
4. "Returns: unused items can be returned within 14 days."

### 1.1 Option A: the simple while-loop orchestrator

The simplest agent is a loop: call the model, if it asks for a tool, run the tool, feed the result back, repeat until it answers. This is the **ReAct** pattern from "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022), covered in [Agent Design Patterns](03-agent-design-patterns.md).

```python
# Conceptual: `llm` is any chat model with function calling, `tools` maps names to Python functions.
def run_agent(user_msg: str, llm, tools: dict, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_msg}]
    for _ in range(max_steps):                         # hard budget so it cannot loop forever
        reply = llm(messages, tools=list(tools))       # model decides: answer, or call tools
        messages.append(reply)
        if not reply.get("tool_calls"):
            return reply["content"]                    # no tool requested: this is the final answer
        for call in reply["tool_calls"]:
            result = tools[call["name"]](**call["args"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
    return "Stopped: step budget exhausted."
```

All the "control flow" lives inside the model's head. The only state is the `messages` list, held in RAM.

### 1.2 Option B: a chain

A **chain** is a fixed sequence of steps where each output feeds the next input, like a Unix pipe: `rewrite_query -> retrieve -> generate`. No step chooses what runs next.

```python
# A chain is just function composition. LangChain's "prompt | llm | parser" syntax is the same idea.
def answer_policy_question(question: str) -> str:
    query = rewrite_query(question)       # step 1: LLM rewrites the question for search
    docs = retrieve(query, k=2)           # step 2: search the 4-document corpus
    return generate(question, docs)       # step 3: LLM answers grounded in the docs
```

Chains are predictable, cheap and easy to test. They cannot loop back, branch on a result, or pause for a human.

### 1.3 Option C: an explicit state graph

A **state machine** (or **finite state machine, FSM**) is a system that is always in exactly one of a finite set of states and moves between them through defined **transitions**, triggered by conditions. A traffic light is the classic example: green, yellow, red, with fixed rules for moving between them.

A **state graph** is the agent version: **nodes** are steps (a function, a model call, a tool call), **edges** are the allowed transitions, and a shared **state** object (a typed dictionary of data) flows through the graph. Some edges are **conditional**: a small function (a **router**) reads the state and decides where to go next.

> **Senior insight:** Watch the vocabulary. In a textbook FSM, "state" means *where you are* (the red light). In LangGraph, "state" means *the data* (the chart), and "where you are" is the set of next nodes to run. Both views matter: you need the data to resume work and the position to know which step to resume.

```text
LOOP                      CHAIN                         STATE GRAPH
+--------------+          rewrite -> retrieve -> gen    classify --+--> retrieve --> plan --> validate
|  call LLM    |<--+                                               |                  ^          |
+------+-------+   |      (fixed order, no choices)                |                  +--invalid-+
       | tool?     |                                               |                             | valid
       v           |                                               +--> respond <-- execute <-- approve?
+--------------+   |                                                       |
|  run tool    |---+                                                      END
+--------------+
(model decides everything)  (code decides everything)   (code decides structure, model decides inside nodes)
```

| Approach | What it is | When to use | Pros | Cons |
|---|---|---|---|---|
| While-loop agent | Model picks next action each turn; tools in a loop | Open-ended tasks, prototypes, short sessions | Tiny code, very flexible | Hard to guarantee order, no persistence, hard to audit, failure mid-loop loses everything |
| Chain | Fixed pipeline of steps | Well-defined tasks: RAG Q&A, extraction, summarization | Predictable, cheap, easy to test | No branching, cycles, pauses or recovery |
| State graph | Typed state + nodes + (conditional) edges, compiled and run by an engine | Multi-step tasks with branches, loops, approvals, long waits | Explicit control, checkpoints, pause/resume, per-node retries and tracing | More code and concepts; framework learning curve |
| Durable workflow engine (Section 8) | Event-sourced workflow code with automatic replay | Long-running business processes, strict reliability needs | Strongest crash recovery, timeouts, retries | Operational weight, determinism rules |

A useful mental model: these are not rivals but points on a dial from "model decides" to "code decides". Good production agents are usually a graph whose *structure* is fixed by code, with loops of model autonomy *inside* specific nodes.

---

## 2. When a state graph beats a simple chain or loop

Reach for an explicit graph when you see one or more of these six needs:

| Need | Example in our support agent | Why loops and chains struggle |
|---|---|---|
| **Branching** | FAQ questions skip planning; refunds go through it | A chain has one path; a loop leaves routing to the model |
| **Cycles** | If the plan fails validation, go back and re-plan (max 3 times) | Chains cannot loop; a free loop has no structured exit condition |
| **Persistence** | Ticket handling spans hours; the server restarts | Loop state lives in RAM and is lost |
| **Human approval** | $250 refund needs a support lead | You cannot block a web request for two days |
| **Recovery** | Payment API times out after the order lookup succeeded | Restarting from scratch repeats work and side effects |
| **Auditability** | "Why did the agent refund this order?" | A loop's decisions are buried in a message list |

### 2.1 Why structure matters: reliability compounds

Here is the math behind "keep autonomy small". If a task needs $n$ steps and each succeeds independently with probability $p$, the whole task succeeds with probability:

$$P(\text{task succeeds}) = p^{\,n}$$

- $p$: probability one step succeeds (model picks the right tool, tool returns, output parses).
- $n$: number of sequential steps that all must succeed.

**Worked example** (illustrative numbers): $p = 0.95$, $n = 10$.

- $0.95^2 = 0.9025$
- $0.95^4 = 0.9025^2 = 0.81450625$
- $0.95^8 = 0.81450625^2 \approx 0.66342$
- $0.95^{10} = 0.95^8 \times 0.95^2 \approx 0.66342 \times 0.9025 \approx 0.5987$

So a step that looks "95% reliable" gives a 10-step task only about a **60%** success rate.

Now add **one retry** per step. If failures are independent and each attempt fails with probability $1 - p = 0.05$, a step fails only if both attempts fail:

$$p_{\text{retry}} = 1 - (1-p)^{k}$$

- $k$: number of attempts per step (here $k = 2$).

$p_{\text{retry}} = 1 - 0.05^2 = 1 - 0.0025 = 0.9975$, and $0.9975^{10} \approx 0.975$. The task success rate jumps from about 60% to about 97.5%.

> **Common mistake:** Assuming retries always help this much. The formula assumes failures are *independent*. A transient network error is independent; a prompt that always produces the wrong JSON is not, and retrying it identically just fails again. That is why good graphs retry transient errors blindly but send validation errors *back to the model with feedback* (a different input) or to an error node.

A graph lets you shrink $n$ for the model (deterministic nodes for routing, validation and formatting do not "fail" randomly), attach the right retry policy to each node, and resume from the failed step instead of restarting all $n$.

---

## 3. Build a tiny state machine engine from scratch

Before using a framework, build the mechanism in 60 lines of plain Python. It shows exactly what LangGraph does for you: nodes return partial updates, routers pick the next node, a checkpoint is written after every step, and a node can **interrupt** (pause) to wait for a human.

```python
# mini_graph.py: no dependencies, Python 3.10+
import json
import os
from typing import Callable

END = "__end__"


class Interrupt(Exception):
    """Raised by a node that needs a human decision before it can continue."""
    def __init__(self, payload: dict):
        self.payload = payload


class MiniGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[dict], dict]] = {}
        self.routers: dict[str, Callable[[dict], str]] = {}

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.routers[src] = lambda state: dst          # always go to dst

    def add_conditional_edge(self, src: str, router: Callable[[dict], str]) -> None:
        self.routers[src] = router                      # router reads state, returns next node

    def _save(self, path: str, state: dict, next_node: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"state": state, "next": next_node}, f)
        os.replace(tmp, path)                           # atomic: never a half-written checkpoint

    def run(self, path: str, state: dict | None = None, start: str = "",
            resume: dict | None = None, max_steps: int = 20) -> dict:
        if os.path.exists(path):                        # resume from the last checkpoint
            with open(path) as f:
                saved = json.load(f)
            state, current = saved["state"], saved["next"]
            state.update(resume or {})                  # e.g. the human's approval decision
        else:
            current = start
        for _ in range(max_steps):                      # hard step budget: no infinite loops
            if current == END:
                os.remove(path)                         # thread finished, clean up
                return {"status": "done", "state": state}
            try:
                update = self.nodes[current](state)
            except Interrupt as i:
                self._save(path, state, current)        # pause BEFORE the node, re-run it on resume
                return {"status": "interrupted", "payload": i.payload}
            state = {**state, **update}                 # merge the node's partial update
            current = self.routers[current](state)
            self._save(path, state, current)            # checkpoint after every step
        raise RuntimeError("step budget exhausted")


# ---- the running example: a refund request -----------------------------
def classify(s):
    return {"intent": "refund" if "refund" in s["request"].lower() else "faq"}

def plan_refund(s):
    return {"action": {"tool": "issue_refund", "order_id": "A1001", "amount": 250}}

def approve(s):
    if s["action"]["amount"] <= 100:
        return {"approved": True}                       # low risk: auto-approve
    if s.get("approved") is None:
        raise Interrupt({"question": "Approve refund?", "action": s["action"]})
    return {}

def execute(s):
    a = s["action"]
    return {"result": f"refunded {a['amount']} on {a['order_id']}"}

def respond(s):
    if s["intent"] == "faq":
        return {"answer": "See our FAQ."}
    if not s.get("approved"):
        return {"answer": "A human reviewer declined this refund."}
    return {"answer": f"Done: {s['result']}."}


g = MiniGraph()
for name, fn in [("classify", classify), ("plan", plan_refund), ("approve", approve),
                 ("execute", execute), ("respond", respond)]:
    g.add_node(name, fn)
g.add_conditional_edge("classify", lambda s: "plan" if s["intent"] == "refund" else "respond")
g.add_edge("plan", "approve")
g.add_conditional_edge("approve", lambda s: "execute" if s["approved"] else "respond")
g.add_edge("execute", "respond")
g.add_edge("respond", END)

ckpt = "thread-42.json"
print(g.run(ckpt, {"request": "Please refund order A1001"}, start="classify"))
# {'status': 'interrupted', 'payload': {'question': 'Approve refund?', 'action': {'tool': 'issue_refund', 'order_id': 'A1001', 'amount': 250}}}

# ... the process could crash or be restarted here; the JSON file survives ...
print(g.run(ckpt, resume={"approved": True})["state"]["answer"])
# Done: refunded 250 on A1001.
```

What this toy teaches, and what a real engine adds (the bold terms are defined in Sections 4 and 5):

| Mechanism in the toy | What a production engine adds |
|---|---|
| `state = {**state, **update}` (last write wins) | **Reducers** per field, so parallel nodes can safely write the same key |
| One file per conversation | A **checkpointer** backed by a database, keyed by **thread ID**, keeping full history |
| `Interrupt` exception, re-run the node on resume | `interrupt()` with a resume value passed back into the node |
| `max_steps` | A recursion limit, plus timeouts and cost budgets |
| Sequential only | Parallel nodes in **super-steps**, streaming, subgraphs |
| Nothing | Retry policies, tracing hooks, time travel, typed schemas |

Notice one subtle design choice we will see again: when `approve` interrupts, the checkpoint stores `next = "approve"`, so on resume **the whole node runs again from the top**. Any side effect placed before the interrupt would happen twice.

---

## 4. LangGraph core concepts: state, nodes, edges, reducers

**LangGraph** is an open-source library from the LangChain team for building stateful, graph-shaped LLM applications. It can be used without the rest of LangChain. As of 2026 it has a stable 1.x line, but details (parameter names, import paths, return shapes) have changed across versions, so treat the code here as *conceptually accurate and realistic*, and check the current docs before copying it into production.

```bash
pip install -U langgraph
```

### 4.1 The building blocks

| Concept | Plain English | In code |
|---|---|---|
| **State schema** | The typed "patient chart" every node reads and updates | A `TypedDict` (a dict with declared keys and value types), Pydantic model or dataclass, passed to `StateGraph(...)` |
| **Node** | One step of work | A Python function `state -> partial update dict` |
| **Edge** | "After A, always go to B" | `builder.add_edge("A", "B")` |
| **Conditional edge** | "After A, ask a router function where to go" | `builder.add_conditional_edges("A", router_fn)` |
| **START / END** | Entry and exit markers | `from langgraph.graph import START, END` |
| **Reducer** | Rule for merging a node's update into an existing field | `Annotated[list, operator.add]` |
| **Compiled graph** | The validated, runnable object | `app = builder.compile(checkpointer=...)` |
| **Checkpointer** | Saves state after each step | `InMemorySaver`, SQLite, Postgres savers |
| **Thread** | One conversation or job's history of checkpoints | `config={"configurable": {"thread_id": "ticket-42"}}` |

A **node returns only the fields it changed** (a partial update). LangGraph merges that into the state. This keeps nodes small, pure-ish and unit-testable: give a dict, get a dict.

**Compiling** (`builder.compile()`) checks the structure (for example, no edges pointing at missing nodes, there is an entry point) and returns a runnable object with `invoke`, `stream`, `get_state` and friends.

### 4.2 Reducers: how updates merge

A **reducer** is a function `(old_value, new_value) -> merged_value` attached to one field of the state. Without a reducer, a new value **overwrites** the old one.

> **Analogy:** A whiteboard field ("current intent") is overwritten each time someone writes on it. A logbook field ("tool results") is appended to. The reducer says which kind each field is.

**Worked example.** Field `notes` has reducer `operator.add` (list concatenation). Current state: `notes = ["order found"]`.

- Node X returns `{"notes": ["policy allows refund"]}`.
- Node Y returns `{"notes": ["amount over $100"]}`.
- After both: `["order found"] + ["policy allows refund"] + ["amount over $100"]`, a 3-item list.

Without the reducer, the field would just be replaced by whichever write came last. And if X and Y ran **in parallel** in the same step and both wrote `notes` with no reducer, LangGraph raises an error (`InvalidUpdateError`) rather than silently picking one, because it cannot know which write should win.

LangGraph ships a special reducer for chat histories, `add_messages`, which appends new messages and **replaces** a message that has the same ID as an existing one (useful for editing a message).

### 4.3 The execution model: super-steps

LangGraph's runtime is inspired by **Pregel** ("Pregel: A System for Large-Scale Graph Processing", Malewicz et al., 2010), which uses the **bulk synchronous parallel** (BSP) model: workers compute independently, then all wait at a barrier and exchange results before the next round starts. Execution proceeds in **super-steps** (one such round):

1. Run every node scheduled for this super-step (in parallel if there are several).
2. Collect all their updates and apply them to the state through the reducers.
3. Save a checkpoint (if a checkpointer is configured).
4. Evaluate outgoing edges to decide which nodes run in the next super-step.
5. Stop when nothing is scheduled (END reached) or the **recursion limit** (a maximum number of super-steps) is hit.

This is why reducers matter: parallel nodes in one super-step all read the *same* snapshot and their writes are merged together afterward.

### 4.4 A first LangGraph graph

```python
# pip install -U langgraph
import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict, total=False):
    request: str
    intent: str
    notes: Annotated[list[str], operator.add]   # reducer: parallel writes are concatenated
    answer: str


def classify(state: State) -> dict:
    return {"intent": "refund" if "refund" in state["request"].lower() else "faq"}

def check_order(state: State) -> dict:          # runs in parallel with check_policy
    return {"notes": ["order A1001: delivered, damaged, $250"]}

def check_policy(state: State) -> dict:
    return {"notes": ["policy: damaged items refundable within 30 days"]}

def respond(state: State) -> dict:
    return {"answer": f"intent={state['intent']}; evidence={state.get('notes', [])}"}

def route(state: State) -> list[str] | Literal["respond"]:
    # Returning a list fans out to several nodes in the same super-step.
    return ["check_order", "check_policy"] if state["intent"] == "refund" else "respond"


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("check_order", check_order)
builder.add_node("check_policy", check_policy)
builder.add_node("respond", respond)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, ["check_order", "check_policy", "respond"])
builder.add_edge(["check_order", "check_policy"], "respond")   # fan-in: wait for both
builder.add_edge("respond", END)
app = builder.compile()

print(app.invoke({"request": "Please refund order A1001"})["answer"])
```

Trace the super-steps for this refund request: first `classify` runs and writes `intent="refund"`; next `check_order` and `check_policy` run in parallel, both reading the same snapshot, and the `operator.add` reducer merges their two `notes` lists; finally `respond` runs, because both branches its fan-in edge waits for have finished, and the run reaches END. With a checkpointer attached, a checkpoint is saved at the end of each super-step.

The final `notes` list contains both entries. Do not rely on the order of entries written by parallel nodes; if order matters, sort or key them.

Two more primitives you should recognize by name:

- **`Command`**: a node can return `Command(goto="plan", update={...})` to update state *and* choose the next node in one step, instead of using a separate conditional edge. Handy for agent "handoffs".
- **`Send`**: a router can return `[Send("grade_doc", {"doc": d}) for d in docs]` to run the same node once per item with different inputs, a **map-reduce** (fan-out then fan-in) pattern useful for grading retrieved documents in parallel.

LangGraph also has a **functional API** (`@entrypoint` and `@task` decorators) that gives persistence and interrupts to ordinary Python control flow, without drawing a graph. Same engine, different authoring style.

---

## 5. Persistence: checkpointers, threads and crash recovery

### 5.1 Definitions

- A **checkpoint** is a saved snapshot of the graph at a super-step boundary: the state values, which nodes run next, and metadata (step number, which node wrote what, parent checkpoint).
- A **checkpointer** is the storage backend that writes and reads checkpoints.
- A **thread** is the sequence of checkpoints for one unit of work (one support ticket, one chat session), identified by a **thread ID** you choose.

> **Analogy:** A video game save system. Each checkpoint is a save file; the thread is your save slot. If the console crashes, you reload the latest save in your slot instead of starting the game over. And you can load an older save to try a different path (time travel, Section 6).

```python
from langgraph.checkpoint.memory import InMemorySaver

app = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ticket-42"}}

app.invoke({"request": "Please refund order A1001"}, config)
snapshot = app.get_state(config)      # latest checkpoint for this thread
print(snapshot.values)                # the state dict
print(snapshot.next)                  # tuple of nodes that would run next; empty when finished
```

### 5.2 Choosing a checkpointer backend

| Backend | Install (package names may change) | Use for | Watch out for |
|---|---|---|---|
| In-memory | built in | Tests, notebooks | Lost on restart; never for production |
| SQLite | `pip install langgraph-checkpoint-sqlite` | Single-process apps, local dev | One writer at a time, one machine |
| Postgres | `pip install langgraph-checkpoint-postgres` | Production with many workers | Table growth, retention, connection pooling |
| Others (Redis, MongoDB, cloud stores, custom) | Community or vendor packages | Existing infra | Maturity varies; test crash semantics |

```python
# Production-style persistence (API shape as of LangGraph 1.x; verify against current docs).
# pip install langgraph-checkpoint-postgres "psycopg[binary,pool]"
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://agent:secret@localhost:5432/agent"   # load from a secret store in real code
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()                                    # creates tables on first run
    app = builder.compile(checkpointer=checkpointer)
    app.invoke({"request": "Please refund order A1001"},
               {"configurable": {"thread_id": "ticket-42"}})
```

### 5.3 Recovering after a crash

Suppose the worker process dies while the `execute_tools` node is calling the payment API. Because a checkpoint was saved after the previous super-step, recovery is:

```python
# New process, same thread ID. Passing None as input means "continue from the latest checkpoint".
app.invoke(None, {"configurable": {"thread_id": "ticket-42"}})
```

The engine sees that `execute_tools` is still pending and runs it again. Two details make this safe or unsafe:

1. **Pending writes.** If several nodes run in one super-step and one fails, LangGraph stores the successful nodes' writes as *pending writes*, so on resume only the failed node re-runs.
2. **Re-execution of the interrupted node.** The node that was running when the crash happened runs *again from the start*. If it had already called the payment API, you could refund twice. The defense is **idempotency**: an operation is **idempotent** if doing it twice has the same effect as doing it once. Pass an **idempotency key** (for example `ticket-42:refund:A1001`) to the payment API so the second call returns the first call's result instead of paying again. Many payment APIs support exactly this.

> **Senior insight:** Checkpointing gives you **at-least-once** execution of nodes (a scheduled node runs to completion once the thread is resumed, but may run more than once), never **exactly-once** (runs one time no matter what fails). Exactly-once *effects* come from combining at-least-once execution with idempotent side effects. Say this sentence in the interview.

**Cheat sheet: what runs again, and when.** One rule (a node always restarts from its first line) shows up in four places:

| Event | What runs again | What can go wrong |
|---|---|---|
| Retry policy fires after a transient error (Section 7.3) | The failed node, from the top | Side effects before the error repeat |
| Resume after `interrupt()` (Section 6) | The interrupted node, from the top | Side effects above `interrupt()` repeat |
| Worker crash, then resume | The node in progress; successful parallel siblings are kept as pending writes | A half-finished side effect repeats |
| Replay or fork from an old checkpoint (Section 6.3) | Every node after that checkpoint | Real tool calls execute again |

The defense is the same in every row: idempotent side effects, and at most one side effect per node where practical.

Newer LangGraph versions also let you pick a **durability mode** per run (for example, save synchronously before continuing, asynchronously, or only at exit), trading write latency for safety. Check the current name of this option; the trade-off is the durable concept.

### 5.4 What belongs in state (and what does not)

- Keep state **small and serializable**: IDs, decisions, short texts, structured results.
- Store **references** to large blobs (documents, images, long tool outputs) rather than the blobs themselves; every super-step writes a checkpoint, so a 2 MB state written 20 times is up to 40 MB per run (illustrative; an upper bound, since some savers store only the fields that changed).
- Separate **short-term memory** (this thread's state) from **long-term memory** (facts about the user across threads). Long-term memory has its own store and is covered in [Memory for AI Agents](06-agent-memory.md).
- Treat checkpoints as **data with PII**: encrypt at rest, set retention, and support deletion requests.

---

## 6. Human in the loop: interrupts, resuming and time travel

**Human in the loop (HITL)** means a person reviews, edits or approves something before the agent continues. For a $250 refund, the support lead must approve.

### 6.1 Dynamic interrupts with `interrupt()`

An **interrupt** pauses the graph mid-run, saves a checkpoint, and returns control to your application. Later you **resume** the same thread with the human's answer. This only works with a checkpointer attached and a `thread_id` in the config, because resuming means loading that saved checkpoint.

```python
from langgraph.types import interrupt, Command

def human_approval(state):
    # Pauses here. The payload is what your UI shows the reviewer.
    decision = interrupt({"question": "Approve refund?", "actions": state["plan"]})
    # On resume, the node runs again from the top, and interrupt() returns the resume value.
    return {"approval": decision}

# 1) First call runs until the interrupt.
out = app.invoke({"request": "Please refund order A1001, it arrived damaged"}, config)
# Recent versions surface the pending interrupt in the result under "__interrupt__";
# you can always inspect app.get_state(config).next and .tasks.

# 2) Hours later, possibly from a different process, the reviewer clicks Approve:
out = app.invoke(Command(resume={"approved": True, "reviewer": "maria"}), config)
```

```mermaid
sequenceDiagram
    participant U as Customer UI
    participant G as Graph worker
    participant C as Checkpointer (Postgres)
    participant R as Reviewer UI
    U->>G: invoke(request, thread_id=ticket-42)
    G->>C: save checkpoint after each super-step
    G-->>U: paused at human_approval (interrupt payload)
    Note over G: worker is free; nothing is blocked
    R->>G: invoke(Command(resume=approved), thread_id=ticket-42)
    G->>C: load latest checkpoint
    G->>G: re-run human_approval, interrupt() returns the decision
    G->>C: save checkpoints for execute_tools and respond
    G-->>R: final answer
```

> **Common mistake:** Putting a side effect *before* `interrupt()` in the same node, such as sending a Slack message to the approver. On resume the node re-runs from the top and the message is sent twice. Put notifications in a separate node before the approval node, or make them idempotent.

Three standard HITL patterns, all built on the same mechanism:

| Pattern | Reviewer does | Resume value |
|---|---|---|
| Approve or reject | Clicks yes or no on a proposed action | `{"approved": True}` |
| Edit | Changes the tool arguments (refund $200 instead of $250) | `{"approved": True, "edited_args": {...}}` |
| Answer a question | Agent needs missing info ("which address?") | `{"answer": "..."}` |

There are also **static breakpoints**: `builder.compile(checkpointer=cp, interrupt_before=["execute_tools"])` pauses before a named node every time. They are mainly a debugging tool; dynamic `interrupt()` is preferred for business approvals because it can be conditional (only refunds over $100).

### 6.2 Inspecting and editing state

- `app.get_state(config)` returns the latest **state snapshot**: `values`, `next`, `config` (including the checkpoint ID), `metadata`, `tasks` (pending work and interrupts).
- `app.update_state(config, {"field": value})` writes a new checkpoint with your changes, as if a node had produced them. Reducers still apply. This is how an operator can fix a bad plan by hand before resuming.

### 6.3 Time travel: replay and fork

Because every super-step is a checkpoint, a thread is a full history you can walk.

```python
history = list(app.get_state_history(config))        # most recent first
for snap in history:
    print(snap.config["configurable"]["checkpoint_id"], snap.next)

# Pick the checkpoint just before planning, change the input, and branch from there.
before_plan = next(s for s in history if s.next == ("plan",))
fork_config = app.update_state(before_plan.config, {"request": "Refund only $200 of order A1001"})
app.invoke(None, fork_config)    # continues from the fork; the original history is kept
```

- **Replay**: invoking with a past checkpoint's config re-executes the steps *after* that checkpoint (earlier steps are not re-run).
- **Fork**: `update_state` on a past checkpoint creates a new branch of history.

Uses: debugging ("what if the classifier had said FAQ?"), reproducing a production failure locally from its exact state, and letting a user "undo" an agent step.

> **Interview tip:** Mention that replay re-runs real nodes, so tool calls after the fork point execute again. In debugging environments, point tools at sandboxes or mocks before replaying production threads.

---

## 7. Branching, retries, error nodes and recovery

### 7.1 Branching patterns

| Pattern | How it is modeled | Example |
|---|---|---|
| **Router** | Conditional edge after a classifier node | FAQ vs refund vs "other" |
| **Guarded loop** | Conditional edge back to an earlier node with a counter in state | Re-plan if validation fails, at most 3 times |
| **Fan-out / fan-in** | Router returns several nodes or `Send` objects; a join node waits | Check order and policy in parallel |
| **Escalation** | Branch to a human-handoff or error node | Unknown intent, repeated failure |
| **Subgraph** | A compiled graph used as a node | A "refund" subgraph reused by several agents |

Make routing **deterministic where possible**. A model can produce a label (`intent`), but the conditional edge should be ordinary code reading that label, with a default branch for unexpected values. That makes routes testable and auditable.

### 7.2 A taxonomy of failures

Different failures need different handling. This table is the heart of the "how do you handle retries and recovery" answer.

| Failure type | Example | Handling |
|---|---|---|
| **Transient** | Timeout, HTTP 429 (rate limited) or 503 (service unavailable) from a tool or model API | Automatic retry with exponential backoff and jitter (node retry policy) |
| **Model output invalid** | Plan JSON fails the schema, unknown tool name | Feed the validation error back to the model and re-plan, with a max attempt count |
| **Business rule violation** | Refund amount exceeds order total | Do not retry; route to error/escalation node, explain to user |
| **Permanent tool error** | Order does not exist (404) | No retry; branch to a response that says so |
| **Policy / risk** | High-value or irreversible action | Interrupt for human approval |
| **Infrastructure crash** | Worker killed mid-node | Resume thread from checkpoint; idempotent side effects |
| **Runaway behavior** | Model keeps looping | Recursion limit, step and cost budgets, then escalate |

### 7.3 Retry policies and exponential backoff

A **retry policy** says how many times to re-attempt a failing node and how long to wait between attempts. **Exponential backoff** doubles (or multiplies) the wait each time, so a struggling service gets breathing room; **jitter** adds randomness so a thousand clients do not all retry at the same instant.

$$d_n = \min\left(d_{\max},\; d_0 \cdot b^{\,n-1}\right)$$

- $d_n$: wait before the $n$-th retry (in seconds), before jitter.
- $d_0$: initial interval.
- $b$: backoff factor (multiplier per retry).
- $d_{\max}$: cap on any single wait.

**Worked example:** $d_0 = 0.5$ s, $b = 2$, $d_{\max} = 10$ s.

- Retry 1: $0.5 \times 2^0 = 0.5$ s
- Retry 2: $0.5 \times 2^1 = 1.0$ s
- Retry 3: $0.5 \times 2^2 = 2.0$ s
- Total waiting for 3 retries: $0.5 + 1.0 + 2.0 = 3.5$ s (plus jitter).

Jitter is applied on top of $d_n$. A popular variant, *full jitter*, waits a uniformly random time between 0 and $d_n$, so clients that failed together spread their retries across the whole window.

In LangGraph, a policy is attached per node (parameter named `retry_policy` in recent versions, `retry` in older ones):

```python
from langgraph.types import RetryPolicy

class TransientToolError(Exception):
    """Raise this from tool wrappers for timeouts, 429s and 5xx responses."""

builder.add_node(
    "execute_tools", execute_tools,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5,
                             backoff_factor=2.0, retry_on=TransientToolError),
)
```

`max_attempts=3` counts the first try, so there are at most 2 retries and 2 waits: $0.5 + 1.0 = 1.5$ s before jitter. (LangGraph's built-in jitter, on by default, adds a small random amount, up to about 1 s in current versions, to each wait rather than using full jitter; check your version.) Use `retry_on` to restrict retries to errors that are actually transient. Retrying a `ValueError` from a bad argument just burns time and money.

> **Senior insight:** Put retries at **one layer only**. If the HTTP client, the tool wrapper and the node each make up to 3 attempts, one outage triggers up to $3 \times 3 \times 3 = 27$ calls (and if each layer *retries* 3 times, that is 4 attempts per layer and $4^3 = 64$ calls) and multiplies latency. Pick the layer that knows whether the error is transient (usually the tool wrapper or node) and disable retries elsewhere.

### 7.4 Error nodes and validation loops

An **error node** is an ordinary node that the graph routes to when something cannot be fixed automatically. It typically writes a structured error to state, opens a ticket for a human, and produces a graceful user-facing message. Routing to an error node is better than letting an exception escape because the failure itself becomes **checkpointed, traceable state**.

The **validation loop** is the standard way to handle invalid model output:

```text
plan --> validate_plan --valid--> (approval or execute)
  ^            |
  +--invalid---+  (errors written to state; attempts < 3)
               |
               +--invalid, 3 attempts used--> handle_error
```

The errors are written into state so the next `plan` call sees them in its prompt ("your previous plan used unknown tool `refund_all`"). This changes the input, so unlike a blind retry, it can actually fix the problem.

### 7.5 Compensation for multi-step side effects

Sometimes step 3 fails after steps 1 and 2 already changed the world (an order was marked "refund pending", an email was sent). Checkpoints let you resume, but they cannot *undo* external effects. The classic answer is the **saga pattern** ("Sagas", Garcia-Molina and Salem, 1987): pair each side-effecting step with a **compensating action** (for example, "mark refund pending" is compensated by "clear refund pending") and run the compensations in reverse order when the overall operation must be abandoned. In a graph, compensation is just another branch leading from the error node.

---

## 8. Durable execution engines (Temporal-style) and how they relate

**Durable execution** means your workflow code keeps running correctly across crashes, deploys and long waits, as if the machine never failed. **Temporal** (which grew out of Uber's Cadence project) is the best-known engine; others in the same family include AWS Step Functions, Azure Durable Functions, Restate, DBOS and Inngest.

### 8.1 Temporal's core ideas

- **Workflow**: ordinary code (Python, Go, Java, TypeScript and others) describing the process. It must be **deterministic**: given the same history it makes the same decisions. No direct network calls, random numbers or clock reads inside workflow code (the SDKs provide deterministic replacements, such as a workflow-safe "now" and random source, whose values are recorded in the history).
- **Activity**: a function that does the real side-effecting work (call the LLM, call the payment API). Activities get timeouts and retry policies. Activities themselves run **at least once**: if a worker dies mid-activity, the activity is retried, so side-effecting activities still need idempotency keys.
- **Event history**: the engine records every step result (activity completed with value X, timer fired, signal received) in an append-only log. This is **event sourcing**.
- **Replay**: after a crash, a worker re-runs the workflow code from the beginning, but instead of re-executing completed activities it reads their results from the history. The code "fast-forwards" to where it stopped.
- **Signals**: external messages into a running workflow, such as "approved by maria". This is the HITL mechanism; the workflow can wait days for one without holding a thread or process.

### 8.2 LangGraph checkpointing vs a durable engine

| Aspect | LangGraph with a checkpointer | Temporal-style durable execution |
|---|---|---|
| What is saved | State snapshot per super-step | Event history of every activity result, timer and signal |
| Recovery | Resume from last checkpoint; the interrupted node re-runs | Replay workflow code; completed activities are not re-run |
| Granularity | Per node (a node with 3 tool calls re-runs all 3) | Per activity (each tool call can be its own activity) |
| Waiting for humans | `interrupt()` + resume via a new invoke | Signals, durable timers ("remind after 24 h") |
| Timeouts | Mostly your code | First-class: start-to-close, schedule-to-close, heartbeats |
| Programming constraints | Nodes can do anything | Workflow code must be deterministic |
| Agent-specific features | Typed state, message reducers, streaming tokens, time travel, LLM tracing integrations | None built in; you bring the agent logic |
| Operational cost | A database for checkpoints | A cluster (or managed service) plus workers |

**How they relate in practice:**

1. **LangGraph alone** is enough for most agents whose runs last seconds to days and whose side effects are few and idempotent.
2. **Durable engine alone**: write the agent loop as a Temporal workflow and each LLM or tool call as an activity. You get the strongest guarantees but must build agent conveniences yourself.
3. **Combined**: the business process (order lifecycle, multi-day onboarding) is a durable workflow, and one activity invokes a LangGraph agent for the reasoning-heavy part. Several agent frameworks now advertise integrations with durable engines; check the current state before designing around one.

> **Senior insight:** If you ever hear "our node re-ran and charged the customer twice", the fix is finer-grained durability (one side effect per node or activity) plus idempotency keys, not just "add more checkpoints".

---

## 9. Tracing tool calls and model decisions

### 9.1 Vocabulary

- A **trace** is the full record of one request or run, end to end.
- A **span** is one timed operation inside a trace (a node, an LLM call, a tool call, a retrieval), with a start time, duration, status and **attributes** (key-value metadata). Spans nest into a tree.
- **Observability** is being able to answer new questions about system behavior from this telemetry without shipping new code.

For an agent you need more than latency. You need **why**: what the model saw, what it decided, which branch the graph took, and what each tool returned.

```text
trace thread=ticket-42 run=7f3a                       (all numbers illustrative)
├─ node: classify                      0.41 s   intent=refund
│   └─ llm: router-model               0.39 s   in=312 tok, out=4 tok, prompt=classify@v7
├─ node: retrieve                      0.12 s   doc_ids=[1, 2]
├─ node: plan                          1.80 s   attempt=1
│   └─ llm: planner-model              1.77 s   in=1,140 tok, out=96 tok, prompt=plan@v12
├─ node: validate_plan                 0.01 s   errors=[]  route=human_approval
├─ node: human_approval                INTERRUPTED  checkpoint=1ef...a9
│        (resumed 2 h 14 min later, reviewer=maria, approved=true)
├─ node: execute_tools                 0.95 s
│   ├─ tool: lookup_order              0.20 s   status=ok
│   └─ tool: issue_refund              0.72 s   status=ok idempotency_key=ticket-42:refund:A1001
└─ node: respond                       0.60 s
```

### 9.2 What to record on each span

| Span type | Attributes worth recording |
|---|---|
| LLM call | model name and version, prompt template ID and version, input/output token counts, latency, cost estimate, finish reason, sampled inputs and outputs (with PII redaction) |
| Model decision | the structured decision (`intent=refund`, chosen tool, plan), confidence or scores if available, the route the graph took next |
| Tool call | tool name, validated arguments (redacted), status, latency, retry count, idempotency key, error class |
| Graph | thread ID, checkpoint ID, node name, super-step number, graph version, user/tenant ID (hashed), interrupts and who resumed them |

Linking **thread ID and checkpoint ID** to trace IDs is the bridge between "what happened" (trace) and "let me reproduce it" (time travel from that checkpoint).

### 9.3 Tools

| Tool | What it is | Notes |
|---|---|---|
| **LangSmith** | LangChain's hosted tracing, evaluation and prompt platform | Near-zero setup for LangGraph: set environment variables such as `LANGSMITH_TRACING=true` and an API key (older versions used `LANGCHAIN_` prefixed names) |
| **Langfuse** | Open-source LLM observability (self-host or cloud): traces, prompt management, evals, cost | Integrates with LangGraph via a callback handler passed in `config={"callbacks": [...]}`; import path differs between SDK major versions. See [LLMOps: Tracing, Prompt Management, Langfuse and RAGAS in Practice](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md) |
| **OpenTelemetry (OTel)** | Vendor-neutral standard for traces, metrics, logs | Has evolving **GenAI semantic conventions** (attribute names such as `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.tool.name`). Lets you send agent spans to the same backend as the rest of your services |

Vendor-neutral manual instrumentation with OTel looks like this. The `opentelemetry-api` calls are stable; the `gen_ai.*` attribute names follow conventions that are still marked as in development, so check the current spec.

```python
# pip install opentelemetry-api opentelemetry-sdk
import time
from opentelemetry import trace

tracer = trace.get_tracer("support-agent")

def traced_tool_call(name: str, fn, args: dict, thread_id: str):
    with tracer.start_as_current_span(f"execute_tool {name}") as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("agent.thread_id", thread_id)          # custom attribute
        start = time.perf_counter()
        try:
            result = fn(**args)
            span.set_attribute("agent.tool.status", "ok")
            return result
        except Exception as exc:
            span.record_exception(exc)                            # stack trace attached to span
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
        finally:
            span.set_attribute("agent.tool.latency_ms", (time.perf_counter() - start) * 1000)
```

> **Interview tip:** Say that you trace *decisions*, not just calls: log the router's label and the edge taken as span attributes. When a refund goes wrong, the question is rarely "how slow was it?" and usually "why did it take that branch?".

Production tracing concerns: **sampling** (keep 100% of errors and interrupts, a fraction of successes), **PII redaction** before export, **retention** policies, and **cost** (storing full prompts for millions of runs is expensive). Monitoring dashboards built on top of traces are covered in [Monitoring ML and LLM Systems in Production](../10-mlops/06-monitoring-ml-and-llm-systems-in-production.md).

---

## 10. Implementing a production agent: a complete example graph

Now we assemble everything into the support agent: **classify request, retrieve context, plan, validate tool calls, ask a human to approve risky actions, execute tools, respond**, with a validation loop, a retry policy, an error node and a checkpointer.

```mermaid
flowchart TD
    S([START]) --> C[classify]
    C -->|faq or refund| R[retrieve]
    C -->|other| RESP[respond]
    R -->|faq| RESP
    R -->|refund| P[plan]
    P --> V[validate_plan]
    V -->|invalid, attempts left| P
    V -->|invalid, 3 attempts used| ERR[handle_error]
    V -->|valid and risky| H[human_approval: interrupt]
    V -->|valid and low risk| X[execute_tools: retry policy]
    H -->|approved| X
    H -->|rejected| RESP
    X --> RESP
    RESP --> E([END])
    ERR --> E
```

Design decisions, before code:

- **The model only does what needs judgement**: classifying intent and proposing a plan. Retrieval here is plain search, validation is plain code, risk checks are plain code, and routing is plain code.
- **Tools are validated against schemas** (here Pydantic models: Python classes whose type hints are checked at runtime) before anything executes (see [Structured Outputs, JSON Mode and Function Calling](../05-llms/05-structured-outputs-and-function-calling.md) and [Tool Calling in Depth](02-tool-calling-in-depth.md)).
- **Risk is decided by code, not by the model**: any plan whose refunds total more than $100 requires approval regardless of what the model says. (Summing matters: checking each call separately would let a plan split one $250 refund into three $83.33 refunds and skip approval.)
- **Side effects are idempotent**: the refund call carries a key derived from the thread ID.

The LLM calls are stubbed so the file runs offline. In production, `llm_classify` and `llm_plan` would be model calls with structured output (for example a chat model's `.with_structured_output(Schema)` in LangChain, or your provider's JSON-schema mode).

```python
# support_agent.py
# pip install -U langgraph pydantic
# Written against the LangGraph 1.x API. Names such as `retry_policy`, the "__interrupt__"
# result key and checkpointer import paths have changed between versions: verify before use.
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig      # installed as a langgraph dependency
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, Field, ValidationError


# ---------- 1. Typed state ---------------------------------------------------
class ToolCall(TypedDict):
    tool: str
    args: dict

class SupportState(TypedDict, total=False):
    request: str                                         # customer's message
    intent: Literal["faq", "refund", "other"]
    context: list[str]                                   # retrieved policy snippets
    plan: list[ToolCall]                                 # proposed tool calls
    plan_errors: list[str]                               # validation feedback for re-planning
    plan_attempts: int
    approval: dict                                       # {"approved": bool, "reviewer": str}
    tool_results: Annotated[list[dict], operator.add]    # reducer: append, never overwrite
    answer: str


# ---------- 2. Tool schemas, risk rules and fake backends ---------------------
class LookupOrderArgs(BaseModel):
    order_id: str = Field(pattern=r"^[A-Z]\d{4}$")

class IssueRefundArgs(BaseModel):
    order_id: str = Field(pattern=r"^[A-Z]\d{4}$")
    amount: float = Field(gt=0, le=1000)

TOOL_SCHEMAS = {"lookup_order": LookupOrderArgs, "issue_refund": IssueRefundArgs}
APPROVAL_THRESHOLD = 100.0

ORDERS = {"A1001": {"total": 250.0, "status": "delivered", "damaged": True}}
POLICY_DOCS = [                                          # our 4-document corpus
    "Refunds: damaged items are refundable within 30 days of delivery.",
    "Refunds over $100 require approval by a support lead.",
    "Shipping: standard delivery takes 3-5 business days.",
    "Returns: unused items can be returned within 14 days.",
]

class TransientToolError(Exception):
    """Timeouts, 429s, 5xx: safe to retry."""

_refunds_by_key: dict[str, dict] = {}                    # stands in for the payment provider

def payment_api_refund(order_id: str, amount: float, idempotency_key: str) -> dict:
    if idempotency_key in _refunds_by_key:               # duplicate call: return original result
        return _refunds_by_key[idempotency_key]
    result = {"refund_id": f"R-{order_id}", "amount": amount}
    _refunds_by_key[idempotency_key] = result
    return result


# ---------- 3. LLM stand-ins (replace with structured-output model calls) -----
def llm_classify(text: str) -> str:
    t = text.lower()
    return "refund" if "refund" in t else "faq" if "?" in t else "other"

def llm_plan(request: str, context: list[str], errors: list[str]) -> list[ToolCall]:
    # A real planner sees the request, the policy context and any previous validation errors.
    return [{"tool": "lookup_order", "args": {"order_id": "A1001"}},
            {"tool": "issue_refund", "args": {"order_id": "A1001", "amount": 250.0}}]


# ---------- 4. Nodes ------------------------------------------------------------
def classify(state: SupportState) -> dict:
    return {"intent": llm_classify(state["request"])}

def retrieve(state: SupportState) -> dict:
    words = set(state["request"].lower().split())
    ranked = sorted(POLICY_DOCS, key=lambda d: -len(words & set(d.lower().split())))
    return {"context": ranked[:2]}                       # toy keyword overlap; use hybrid search in prod

def plan(state: SupportState) -> dict:
    steps = llm_plan(state["request"], state["context"], state.get("plan_errors", []))
    return {"plan": steps, "plan_attempts": state.get("plan_attempts", 0) + 1}

def validate_plan(state: SupportState) -> dict:
    errors = []
    for step in state["plan"]:
        schema = TOOL_SCHEMAS.get(step["tool"])
        if schema is None:                                # allowlist: unknown tools are rejected
            errors.append(f"unknown tool {step['tool']!r}")
            continue
        try:
            args = schema(**step["args"])
        except ValidationError as e:
            errors.append(f"{step['tool']}: {e.errors()[0]['msg']}")
            continue
        if step["tool"] == "issue_refund":                # business rule, checked in code
            order = ORDERS.get(args.order_id)
            if order is None or args.amount > order["total"]:
                errors.append("refund amount exceeds order total or order not found")
    return {"plan_errors": errors}

def human_approval(state: SupportState) -> dict:
    risky = [s for s in state["plan"] if s["tool"] == "issue_refund"]
    # Pauses the graph and checkpoints. On resume this node re-runs from the top and
    # interrupt() returns the reviewer's decision, so nothing side-effecting goes above it.
    decision = interrupt({"question": "Approve these actions?", "actions": risky,
                          "request": state["request"], "policy": state["context"]})
    return {"approval": decision}

def execute_tools(state: SupportState, config: RunnableConfig) -> dict:
    thread_id = config["configurable"]["thread_id"]
    results = []
    for step in state["plan"]:
        args = step["args"]
        if step["tool"] == "lookup_order":
            results.append({"tool": "lookup_order", "output": ORDERS[args["order_id"]]})
        elif step["tool"] == "issue_refund":
            key = f"{thread_id}:refund:{args['order_id']}"   # same key on every retry or resume
            out = payment_api_refund(args["order_id"], args["amount"], idempotency_key=key)
            results.append({"tool": "issue_refund", "output": out})
    return {"tool_results": results}

def respond(state: SupportState) -> dict:
    if state.get("intent") == "other":
        return {"answer": "I've passed your message to a human agent."}
    if state.get("intent") == "faq":
        return {"answer": f"From our policy: {state['context'][0]}"}
    if "approval" in state and not state["approval"].get("approved"):   # same test as the router
        return {"answer": "Your refund needs further review. A support lead will contact you."}
    refund = next(r["output"] for r in state["tool_results"] if r["tool"] == "issue_refund")
    return {"answer": f"Refund {refund['refund_id']} for ${refund['amount']:.2f} has been issued."}

def handle_error(state: SupportState) -> dict:
    # In production: open a ticket with the errors, alert if the error rate spikes.
    return {"answer": "Sorry, I couldn't complete this automatically. A human agent will follow up."}


# ---------- 5. Routers (plain, testable code) -----------------------------------
def route_after_classify(state: SupportState) -> Literal["retrieve", "respond"]:
    return "retrieve" if state["intent"] in ("faq", "refund") else "respond"

def route_after_retrieve(state: SupportState) -> Literal["plan", "respond"]:
    return "plan" if state["intent"] == "refund" else "respond"

def route_after_validation(state: SupportState) -> Literal[
        "plan", "handle_error", "human_approval", "execute_tools"]:
    if state["plan_errors"]:
        return "plan" if state["plan_attempts"] < 3 else "handle_error"
    refund_total = sum(s["args"]["amount"] for s in state["plan"] if s["tool"] == "issue_refund")
    risky = refund_total > APPROVAL_THRESHOLD          # total, so splitting a refund cannot dodge approval
    return "human_approval" if risky else "execute_tools"

def route_after_approval(state: SupportState) -> Literal["execute_tools", "respond"]:
    return "execute_tools" if state["approval"].get("approved") else "respond"


# ---------- 6. Build and compile -------------------------------------------------
def build_graph() -> StateGraph:
    b = StateGraph(SupportState)
    b.add_node("classify", classify)
    b.add_node("retrieve", retrieve)
    b.add_node("plan", plan)
    b.add_node("validate_plan", validate_plan)
    b.add_node("human_approval", human_approval)
    b.add_node("execute_tools", execute_tools,
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5,
                                        backoff_factor=2.0, retry_on=TransientToolError))
    b.add_node("respond", respond)
    b.add_node("handle_error", handle_error)

    b.add_edge(START, "classify")
    b.add_conditional_edges("classify", route_after_classify)
    b.add_conditional_edges("retrieve", route_after_retrieve)
    b.add_edge("plan", "validate_plan")
    b.add_conditional_edges("validate_plan", route_after_validation)
    b.add_conditional_edges("human_approval", route_after_approval)
    b.add_edge("execute_tools", "respond")
    b.add_edge("respond", END)
    b.add_edge("handle_error", END)
    return b


if __name__ == "__main__":
    app = build_graph().compile(checkpointer=InMemorySaver())   # PostgresSaver in production
    config = {"configurable": {"thread_id": "ticket-42"}, "recursion_limit": 25}

    out = app.invoke({"request": "Please refund order A1001, it arrived damaged"}, config)
    print(app.get_state(config).next)       # ('human_approval',): paused, waiting for a reviewer

    # Later, the reviewer approves (in production this arrives via an API call from the review UI).
    out = app.invoke(Command(resume={"approved": True, "reviewer": "maria"}), config)
    print(out["answer"])                    # Refund R-A1001 for $250.00 has been issued.

    # Audit trail: every super-step is a checkpoint you can list, inspect or fork.
    for snap in app.get_state_history(config):
        print(snap.metadata.get("step"), snap.next)
```

Things to notice:

- **Routers have `Literal` return types.** LangGraph uses them to know the possible targets for visualization (`app.get_graph().draw_mermaid()` renders the graph). You can instead pass an explicit list or mapping of targets to `add_conditional_edges`.
- **The guarded loop** is `validate_plan -> plan` with `plan_attempts < 3`. The counter lives in state, so it survives crashes.
- **The approval is enforced by the graph, not by the prompt.** A prompt injection cannot talk the router out of `human_approval`, because the route is decided by code reading the total refund `amount`. The model still controls the *arguments*, though, so the rule must look at the whole plan (here, the sum of refunds), and a hard per-order limit belongs in the payment service too. Prompt injection defenses are covered in [Making Agents Reliable and Secure for Production](08-agent-reliability-security-and-prompt-injection.md).
- **Streaming progress**: `for chunk in app.stream(inputs, config, stream_mode="updates"): ...` yields each node's update as it completes, which you can push to a UI ("Looking up your order...").
- **Tracing**: add LangSmith environment variables or a Langfuse callback handler in `config["callbacks"]`, and every node, LLM call and tool call appears as a nested span.

### 10.1 Testing the graph

Because nodes are functions from dict to dict, most tests need no LLM:

```python
# pytest test_support_agent.py
from support_agent import route_after_validation, validate_plan

def test_large_refund_requires_approval():
    state = {"plan": [{"tool": "issue_refund", "args": {"order_id": "A1001", "amount": 250.0}}],
             "plan_errors": [], "plan_attempts": 1}
    assert route_after_validation(state) == "human_approval"

def test_unknown_tool_is_rejected():
    state = {"plan": [{"tool": "refund_all", "args": {}}]}
    assert validate_plan(state)["plan_errors"] == ["unknown tool 'refund_all'"]

def test_third_invalid_plan_goes_to_error_node():
    state = {"plan": [], "plan_errors": ["bad"], "plan_attempts": 3}
    assert route_after_validation(state) == "handle_error"
```

Then add graph-level tests with a mocked LLM (does the refund path pause at approval? does rejection skip the tool?), crash tests (kill the worker during `execute_tools`, resume, assert exactly one refund), and evaluation suites (Section 13).

---

## 11. Going deeper: running agent graphs in production

### 11.1 Deployment architecture

```text
             +-------------------+        +--------------------+
 client ---> |  API (stateless)  | -----> |  job queue         |
  /review UI |  start / resume   |        |  (thread_id keyed) |
             +-------------------+        +---------+----------+
                      ^                             |
                      | results, interrupts         v
             +--------+----------+        +--------------------+      +-----------------+
             |  event stream /   | <----- |  graph workers     | ---> | tools / LLM APIs|
             |  webhooks         |        |  (stateless, N)    |      +-----------------+
             +-------------------+        +---------+----------+
                                                    |
                                          +---------v----------+      +-----------------+
                                          |  Postgres:         |      |  tracing backend|
                                          |  checkpoints       |      |  (OTel/Langfuse)|
                                          +--------------------+      +-----------------+
```

- **Workers are stateless.** All durable state is in the checkpointer, so any worker can resume any thread, and you can autoscale workers.
- **One run per thread at a time.** Two concurrent invocations on the same thread (a user double-clicks "approve") race on checkpoints. Serialize per thread with a queue partitioned by thread ID or a database lock, or reject the second request. Managed LangGraph servers offer policies such as reject or enqueue for this case.
- **Long waits cost nothing.** A thread paused at `human_approval` is just rows in Postgres; no process is blocked.

### 11.2 Budgets and guardrails on the orchestration itself

- **Recursion limit**: maximum super-steps per run (passed in config as `recursion_limit`; a default of 25 has been common, but check your version). Exceeding it raises an error you should route to escalation.
- **Cost budget**: track tokens per thread in state; route to an error node when exceeded.
- **Wall-clock timeouts** per node and per run, especially around tools.
- **Tool permissions**: the graph decides which tools are even reachable from which nodes; a planner node cannot call `issue_refund` directly, only propose it.

### 11.3 Versioning graphs with in-flight threads

Threads paused for approval may sit for days while you deploy a new graph version. If you renamed `human_approval` to `approval_gate`, the old checkpoint's `next` points at a node that no longer exists.

- Prefer **additive changes**: new optional state fields with defaults, new nodes, keep old node names.
- Store a **graph version** in thread metadata; keep the old version deployable until its threads drain, or write an explicit migration.
- Treat the state schema like a database schema: version it and review changes.

### 11.4 Cost and latency

- Every LLM node adds a model round trip: five sequential LLM nodes at about 1 s each (illustrative) means about 5 s. Fan out independent nodes, use small fast models for routing, and stream partial results.
- Checkpoint writes add a database round trip per super-step: small next to LLM latency, but noticeable for many tiny nodes. Merge trivial nodes or relax the durability mode for non-critical runs.
- Inference-level latency work is covered in [Optimizing and Benchmarking LLM Latency and Cost](../06-llm-inference/07-optimizing-and-benchmarking-llm-latency-and-cost.md).

### 11.5 Security and compliance

- **Approval binding**: the reviewer approves a specific set of actions. Store a hash of the approved actions with the decision and re-check it in `execute_tools`, so a later state edit cannot swap in a different refund amount (a time-of-check versus time-of-use bug).
- **Least privilege**: tool credentials are scoped per tenant and per tool, never placed in state or prompts.
- **Audit**: checkpoints plus traces give a replayable record of who approved what and why. For regulated workflows, export approval events to an append-only audit store.

---

## 12. Other frameworks and how to evaluate one

The landscape changes quickly; this is a snapshot of durable *design philosophies*, not a ranking. Verify current features before choosing.

| Framework | Core abstraction | Strengths | Weaknesses | Good fit |
|---|---|---|---|---|
| **LangGraph** | Explicit state graph, reducers, checkpointers, interrupts | Fine-grained control, persistence, HITL, time travel, streaming | More boilerplate; concepts to learn; API has evolved | Production agents that need control and auditability |
| **OpenAI Agents SDK** | Agents (instructions + tools), handoffs between agents, guardrails, sessions, built-in tracing | Small API, quick to start, handoff-style multi-agent | Control flow mostly implicit in the model loop; persistence and HITL patterns less explicit than a graph | Tool-using assistants, triage-and-handoff designs |
| **LlamaIndex Workflows** | Event-driven steps: functions that consume and emit typed events | Natural for RAG-heavy pipelines, async, close to LlamaIndex retrieval | Smaller ecosystem for agent ops than graph-first tools | Retrieval-centric agents and document workflows |
| **CrewAI** | Role-based agents (role, goal, backstory) collaborating on tasks in a "crew"; "Flows" for more deterministic control | Fast multi-agent prototypes, readable configs | Emergent conversations are harder to debug and bound | Content and research pipelines, demos |
| **AutoGen** (Microsoft) | Agents exchanging messages; newer versions use an asynchronous, event-driven actor model | Research-friendly multi-agent conversation patterns | Microsoft announced in 2025 a convergence of AutoGen and Semantic Kernel into the Microsoft Agent Framework; check current status | Experiments with multi-agent conversation |
| **Semantic Kernel** (Microsoft) | Kernel with plugins (functions) and AI services; C#, Python, Java | Enterprise and .NET integration, Azure ecosystem | Same convergence note as AutoGen | Enterprise teams on Microsoft stacks |
| **Plain code (+ durable engine)** | Your own loop or state machine, optionally on Temporal | Total control, no framework churn | You build persistence, HITL, tracing yourself | Teams with strong platform engineering, strict requirements |

### 12.1 How to evaluate a framework

Score each candidate against your requirements, then run a **bake-off**: implement the same one or two representative workflows (including an approval step and a failing tool) in each finalist.

1. **Control flow explicitness**: can you see and test every possible path?
2. **Persistence and recovery**: pluggable durable store? What exactly re-runs after a crash?
3. **Human in the loop**: pause for days, resume from another process, edit state?
4. **Observability**: native tracing, OpenTelemetry export, can you trace decisions and not just calls?
5. **Testability**: can nodes be unit-tested without an LLM? Can you mock tools?
6. **Model and vendor neutrality**: can you swap model providers easily?
7. **Concurrency and streaming**: async support, parallel branches, token streaming.
8. **Deployment story**: self-hosted, managed, both? What does it cost to operate?
9. **Maturity and API stability**: release cadence, breaking changes, migration guides.
10. **Escape hatches**: can you drop to plain Python when the abstraction does not fit?
11. **License and lock-in**: open source? Is the core usable without a paid platform?

> **Senior insight:** The most important evaluation test is the kill test. Start a run that includes a side effect, kill the worker mid-node, restart it, and check whether the side effect happened zero, one or two times. Many frameworks look identical on the happy path and differ completely here.

### 12.2 Loop vs graph vs multi-agent

A **multi-agent architecture** splits work across several agents, each with its own prompt, tools and often its own model. Common shapes are a **supervisor** (a router agent delegates to specialists and collects results) and **handoffs** or **swarms** (agents pass control to each other directly). In LangGraph, each sub-agent is often a subgraph and the supervisor is a router node.

| | Simple loop | State graph | Multi-agent |
|---|---|---|---|
| Who decides the next step | The model, every turn | Code (structure) plus model (inside nodes) | Supervisor model or agents handing off |
| Strength | Flexibility, minimal code | Control, persistence, testability | Separation of concerns, specialized prompts and tools, parallel work |
| Weakness | Unbounded, hard to audit or resume | Upfront design, less flexible | More LLM calls, cost, latency, coordination failures, harder evaluation |
| Use when | Short open-ended tasks | Known business processes with branches and approvals | Genuinely distinct sub-domains or tool sets too large for one prompt |

These compose: a multi-agent system is usually *implemented* as a graph. When multi-agent designs help and when they hurt is covered in [Multi-Agent Systems](05-multi-agent-systems.md).

---

## 13. Evaluating task completion and agent reliability

Evaluation gets its own chapter, [Evaluating and Observing Agents](09-evaluating-and-observing-agents.md). Here is what you need for orchestration questions.

### 13.1 What to measure

| Metric | Definition | Why it matters |
|---|---|---|
| **Task success rate** | Fraction of tasks where the final *world state* is correct (refund issued once, for the right amount) | The real goal; check outcomes, not just the text of the answer |
| **pass^k** | Probability the agent succeeds on *all* of $k$ repeated trials of the same task | Consistency; users hit the same task many times |
| **Trajectory quality** | Correct tools, valid arguments, no unnecessary steps, correct branch taken | Diagnoses *why* a task failed |
| **Policy compliance** | Risky actions always went through approval; no forbidden tools | Safety, often a hard gate |
| **Escalation / handoff rate** | Fraction routed to humans or error nodes | Too high: useless automation. Too low: possibly overconfident |
| **Recovery rate** | Fraction of injected failures (tool timeouts, worker kills) that still end correctly | Tests your retries, checkpoints and idempotency |
| **Cost and latency per successful task** | Tokens, dollars and seconds divided by successes | Efficiency that accounts for failures |

### 13.2 pass@k vs pass^k

These two are easy to confuse. **pass@k** asks "did *at least one* of $k$ attempts succeed?" (generous; fits code generation where you can pick a winner). **pass^k**, introduced for agents in "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" (Yao et al., 2024), asks "did *all* $k$ attempts succeed?" (strict; fits agents that must be reliable every time).

With $n$ trials of a task of which $c$ succeeded, unbiased per-task estimates are:

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \qquad \text{pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}}$$

- $n$: number of trials run for the task.
- $c$: number of successful trials.
- $k$: number of attempts being asked about ($k \le n$).
- $\binom{a}{b}$: "a choose b", the number of ways to pick $b$ items from $a$ (zero when $b > a$).

Average these over all tasks in the suite.

**Worked example:** our refund task is run $n = 4$ times and succeeds $c = 3$ times. Take $k = 2$.

- $\binom{4}{2} = \frac{4 \times 3}{2} = 6$
- $\binom{3}{2} = 3$, so $\text{pass}^2 = 3/6 = 0.5$
- $\binom{1}{2} = 0$ (cannot pick 2 failures from 1), so $\text{pass@}2 = 1 - 0/6 = 1.0$

Same data, very different stories: "always solvable in 2 tries" vs "only a coin flip that 2 runs both succeed". If trials are independent with success probability $p$, $\text{pass}^k \approx p^k$: with $p = 0.8$ and $k = 3$, $0.8^3 = 0.512$, while pass@3 would be $1 - 0.2^3 = 1 - 0.008 = 0.992$ (illustrative).

### 13.3 How to evaluate in practice

1. **Scenario suite** with a *checkable end state* ("exactly one $250 refund recorded, approval event present"), including adversarial cases, plus **simulated users** for multi-turn tasks, as in tau-bench.
2. **Node-level unit tests** for routers, validators and risk rules (Section 10.1).
3. **Trajectory assertions** from traces: "every `issue_refund` span over $100 is preceded by an approval".
4. **Fault injection**: random tool timeouts and worker kills; measure recovery rate and duplicate side effects.
5. **LLM-as-a-judge** for answer quality, calibrated against human labels (see [Evaluating LLMs: Metrics, Benchmarks and LLM-as-a-Judge](../05-llms/10-llm-evaluation-metrics-and-llm-as-a-judge.md)).
6. **Regression gates in CI** on every prompt, model or graph change (see [Building an LLM Evaluation Framework and Regression Tests](../05-llms/11-building-an-llm-evaluation-framework-and-regression-tests.md)), plus online monitoring of escalation rate, approval rejections and cost per task.

---

## Common mistakes and misconceptions

- **"A framework makes my agent reliable."** It gives you mechanisms (checkpoints, retries, interrupts). Reliability comes from how you use them: idempotent tools, typed validation, budgets and tests.
- **Letting the model choose risky routes.** If the prompt says "ask for approval for big refunds", an injected or confused model can skip it. Put risk gates in conditional edges written in code.
- **Side effects before `interrupt()`** in the same node. The node re-runs on resume, so the side effect repeats.
- **Assuming exactly-once execution.** Checkpointing is at-least-once per node. Without idempotency keys, crash recovery can duplicate payments, emails or tickets.
- **Retrying everything.** Retrying validation errors or permanent 404s wastes money; retrying at three layers multiplies calls. Classify errors and retry transient ones at one layer.
- **Stuffing huge objects into state.** Every super-step checkpoints the state; store references instead, and redact PII.
- **No reducer on fields written by parallel nodes.** You get an update conflict error or, with a sloppy custom design, silent data loss.
- **Ignoring graph versioning.** Renaming nodes breaks threads that are paused mid-flight.
- **Building a multi-agent system when one graph would do.** More agents mean more calls, more cost and more ways to fail; start with the simplest structure that works.
- **Tracing only latency.** Without the router decision and tool arguments on spans, you cannot answer "why did it do that?".
- **Evaluating only the final text.** A polite answer can hide a wrong or duplicated refund; check the end state of the world.

---

## Interview questions and model answers

<details>
<summary><b>Q1. How would you implement a production agent using LangGraph or a similar framework?</b></summary>

**Short answer:** I would model the agent as an explicit state graph: a typed state schema, small nodes for classify, retrieve, plan, validate, approve, execute and respond, and conditional edges written in code. LLMs only make the judgement calls (intent, plan); validation, risk checks and routing are deterministic. I compile it with a Postgres checkpointer keyed by a thread ID per ticket, use `interrupt()` for human approval of risky actions, give tool nodes retry policies for transient errors, make side effects idempotent, and trace every node, LLM call and tool call. Then I gate releases on an evaluation suite.

**Deeper answer and follow-ups:**
- **State**: minimal, typed, serializable; reducers (`operator.add`, `add_messages`) for fields written by multiple or parallel nodes; references instead of large blobs.
- **Tools**: schemas (Pydantic), allowlists per node, business rules in code (refund cannot exceed order total), scoped credentials, idempotency keys derived from thread ID.
- **Control**: guarded loops (re-plan at most 3 times), error node for unrecoverable cases, recursion limit and cost budget, escalation to humans.
- **HITL**: approval node with no side effects above `interrupt()`; resume via `Command(resume=...)` from a review API; bind approval to a hash of the approved actions.
- **Deployment**: stateless workers behind a queue, one active run per thread, streaming updates to the UI, graph and state schema versioning for in-flight threads.
- **Observability and eval**: LangSmith, Langfuse or OTel spans with decisions as attributes; scenario suite with end-state checks, pass^k, fault injection; CI regression gates.
- **Rollout**: launch in shadow mode or with every risky action requiring approval, relax approval thresholds only when measured task success, pass^k and policy compliance justify it, and canary new graph versions on a slice of new threads.
- **Pitfall**: building a free-form ReAct loop with all tools exposed and "please ask before refunding" in the prompt.

</details>

<details>
<summary><b>Q2. Compare LangGraph, a simple orchestration loop, and multi-agent architectures.</b></summary>

**Short answer:** A simple loop lets the model choose every next step; it is tiny and flexible but hard to bound, resume or audit. LangGraph makes the structure explicit: code defines nodes and allowed transitions, the model works inside nodes, and the engine adds checkpoints, interrupts, retries and time travel. Multi-agent architectures split work across specialized agents coordinated by a supervisor or handoffs; they help with distinct sub-domains or very large tool sets but add calls, latency, cost and coordination failures. They are not exclusive: multi-agent systems are often implemented as LangGraph graphs.

**Deeper answer and follow-ups:**
- **Loop**: best for short, open-ended tasks and prototypes. Reliability compounds badly: at 95% per step, 10 steps give about $0.95^{10} \approx 0.60$ success (illustrative).
- **Graph**: best for known business processes with branches, approvals and long waits. Cost: upfront design, framework concepts, API churn.
- **Multi-agent**: supervisor (central router, easier to control) vs handoff/swarm (decentralized, more flexible, harder to bound). Each extra agent hop is another LLM round trip.
- **Decision rule**: start with a workflow or graph; allow loop autonomy inside specific nodes; go multi-agent only when a single agent's prompt or tool list is demonstrably the bottleneck.
- **How to compare them fairly**: run every candidate on the same scenario suite and compare task success, pass^k, policy compliance, and cost and latency per successful task. A multi-agent design must beat the single-graph baseline by enough to pay for its extra calls.
- **Follow-up "which would you pick for a refund agent?"**: a graph, because approval, idempotency and audit are hard requirements.

</details>

<details>
<summary><b>Q3. When is a state graph preferable to a simple chain?</b></summary>

**Short answer:** When the flow is not a straight line or must outlive one request: you need branching based on intermediate results, cycles such as re-planning or retrieve-again loops, persistence across restarts or long waits, human approval in the middle, recovery from a failed step without redoing earlier ones, or a step-by-step audit trail. If the task is a fixed pipeline that finishes in one request, a chain is simpler and better.

**Deeper answer and follow-ups:**
- Chain examples that should stay chains: RAG question answering, extraction, summarization.
- Graph examples: support refunds with approval, agentic RAG that may re-query, document processing with human review.
- **Cost of the graph**: more code, state schema to maintain, checkpoint writes per step.
- **Middle ground**: a chain inside a single graph node; LangGraph's functional API for linear code that still needs persistence and interrupts.
- **Pitfall**: building a graph for a 3-step pipeline "to be future-proof"; you pay complexity with no benefit.

</details>

<details>
<summary><b>Q4. How do you manage state and checkpoints?</b></summary>

**Short answer:** I define a small, typed state schema with explicit reducers, have nodes return partial updates, and attach a durable checkpointer (Postgres in production) so a checkpoint is written after every super-step under a thread ID per conversation or job. To resume after a crash or an approval, I invoke the same thread; to debug, I inspect or fork the checkpoint history. I keep state lean, put large data behind references, encrypt and expire checkpoints, and version the schema.

**Deeper answer and follow-ups:**
- **Checkpoint contents**: state values, next nodes, metadata, parent pointer, pending writes from parallel nodes that already succeeded.
- **Semantics**: at-least-once node execution, so side effects need idempotency keys.
- **Concurrency**: one active run per thread (queue or lock).
- **Operations**: retention, PII encryption and deletion, durability mode (sync vs async writes); storage growth is at most checkpoint size times super-steps per run times runs per day (for example 20 KB x 15 x 100,000 = 30 GB per day, illustrative; savers that store only changed fields grow more slowly).
- **Schema evolution**: additive optional fields, stable node names, graph version in metadata.
- Long-term cross-thread memory lives in a separate store, not in thread state.

</details>

<details>
<summary><b>Q5. How do you handle branching, retries, and recovery? How do you model branching, retries, checkpoints, and human approval?</b></summary>

**Short answer:** Branching is conditional edges whose router functions are plain code reading state (often a label the model produced), with a default branch. Retries depend on the failure class: transient errors get a per-node retry policy with exponential backoff and jitter; invalid model output loops back to the planner with the validation errors in state, bounded by an attempt counter; permanent or business-rule failures go to an error node. Checkpoints after every super-step let a crashed run resume at the failed node, and idempotency keys make that safe. Human approval is a dedicated node that calls `interrupt()`, persisted by the checkpointer and resumed with the reviewer's decision.

**Deeper answer and follow-ups:**
- **Backoff**: $d_n = \min(d_{\max}, d_0 b^{n-1})$; with $d_0 = 0.5$ s and $b = 2$, waits are 0.5, 1, 2 s. Retry at one layer only to avoid multiplicative storms ($3 \times 3 \times 3 = 27$ calls).
- **Retry math**: with independent failures, $1 - (1-p)^k$ per step; 0.95 becomes 0.9975 with one retry. It does not help for deterministic failures.
- **Recovery**: resume the thread with `invoke(None, config)`; pending writes prevent re-running successful parallel nodes.
- **Compensation**: saga-style compensating actions for multi-step side effects that must be undone.
- **Approval**: conditional on risk computed in code; payload shows exact actions; supports approve, edit or reject; nothing side-effecting before `interrupt()`; approval bound to the actions' hash.
- **Escalation**: recursion limit and cost budget breaches route to a human.
- **Measure**: retry rate and retry success rate per node and error class, re-plan attempts per run, duplicate side effects under fault injection (target zero), and approval wait time. A rising retry rate is often the first sign of a degraded dependency.

</details>

<details>
<summary><b>Q6. How do you trace tool calls and model decisions?</b></summary>

**Short answer:** I instrument each run as a trace with nested spans for every node, LLM call and tool call, using LangSmith or Langfuse integrations or OpenTelemetry with the GenAI semantic conventions. Beyond latency and tokens, I record decisions as attributes: the classifier's label, the plan, the conditional edge taken, validation errors, approvals and who gave them. Each trace carries the thread ID and checkpoint IDs so I can jump from a bad trace to time-travel replay of that exact state.

**Deeper answer and follow-ups:**
- **LLM span**: model and version, prompt template version, tokens in and out, cost, finish reason, sampled redacted inputs and outputs.
- **Tool span**: name, validated (redacted) arguments, status, retries, idempotency key, error class.
- **Graph span**: node, super-step, graph version, interrupts and resumes.
- **Operations**: sample successes but keep all errors and interrupts; redact PII before export; set retention; watch tracing cost.
- **Overhead**: export spans asynchronously in batches (OpenTelemetry's `BatchSpanProcessor`) so tracing never blocks the agent, and never let a tracing failure fail the run.
- **Usage**: dashboards (success rate, escalations, cost per task), alerting, building eval datasets from production traces.
- **Pitfall**: logging only free-text prompts and outputs; without structured decision attributes you cannot query "all runs where the router chose refund but intent was FAQ".

</details>

<details>
<summary><b>Q7. How would you evaluate task completion and agent reliability?</b></summary>

**Short answer:** I build a scenario suite with checkable end states (was exactly one $250 refund recorded, with an approval?) and measure task success rate, consistency with pass^k over repeated trials, trajectory quality (right tools, valid arguments, correct branches), policy compliance, escalation rate, recovery rate under injected faults, and cost and latency per successful task. Node-level unit tests cover routers and validators, LLM-as-a-judge with human calibration covers answer quality, and the whole suite runs in CI as a regression gate, backed by online monitoring.

**Deeper answer and follow-ups:**
- **pass^k vs pass@k**: with 4 trials and 3 successes, $\text{pass}^2 = \binom{3}{2}/\binom{4}{2} = 3/6 = 0.5$ while $\text{pass@}2 = 1.0$. Agents need pass^k.
- **Simulated users** for multi-turn tasks; adversarial cases (prompt injection, missing info, contradictory requests).
- **Fault injection**: tool timeouts, 500s, worker kills mid-node; assert no duplicate side effects.
- **Trajectory checks from traces**: approvals always precede risky tool spans.
- **Variance**: run multiple seeds or trials; report confidence intervals; LLM outputs are stochastic.
- Pointer: full treatment in the agent evaluation chapter.

</details>

<details>
<summary><b>Q8. What is a reducer in LangGraph, and why is it needed?</b></summary>

**Short answer:** A reducer is a per-field function `(old, new) -> merged` that defines how a node's update is combined with the existing state. Without one, a new value overwrites the old. Reducers are needed for accumulating fields like message histories or tool results, and essential when parallel nodes in the same super-step write the same key; without a reducer that situation raises an update conflict error instead of silently dropping data.

**Deeper answer and follow-ups:**
- `Annotated[list, operator.add]` concatenates lists; `add_messages` appends and replaces by message ID.
- Custom reducers: dedupe by key, keep max, merge dicts.
- **Pitfall**: an append reducer on a field you meant to reset each loop; it grows forever. Use an overwrite field or a reducer that supports clearing.
- **Order**: do not rely on the order of writes from parallel nodes.

</details>

<details>
<summary><b>Q9. What happens to side effects when a node is retried or resumed after an interrupt, and how do you make that safe?</b></summary>

**Short answer:** The node runs again from the top. Retries re-run the failed node; resuming after `interrupt()` re-runs the interrupted node; crash recovery re-runs whatever node was in progress. So any side effect in that node can happen more than once. I make side effects idempotent with keys derived from stable IDs (thread ID plus action), keep `interrupt()` above any side effects or isolate side effects in their own nodes, and keep one side effect per node where practical.

**Deeper answer and follow-ups:**
- At-least-once execution plus idempotency equals exactly-once effects.
- Payment and messaging APIs often accept idempotency keys; for your own services, store processed keys in a table with a unique constraint.
- For non-idempotent external systems: check-then-act with a status lookup, or a durable engine where each call is a separately recorded activity.
- Saga compensations when you must undo.

</details>

<details>
<summary><b>Q10. How does LangGraph checkpointing compare with a durable execution engine like Temporal? When would you use each?</b></summary>

**Short answer:** LangGraph saves a state snapshot per super-step and resumes at the pending node, re-running it. Temporal records an event history of every activity result and replays deterministic workflow code, skipping completed activities, with first-class timeouts, retries, signals and durable timers. LangGraph is the better developer experience for agent logic (typed state, messages, streaming, HITL, time travel); Temporal gives stronger guarantees for long-running, multi-service business processes. They combine well: Temporal for the business process, LangGraph for the reasoning step.

**Deeper answer and follow-ups:**
- Temporal constraint: workflow code must be deterministic; all I/O lives in activities.
- Granularity matters: a LangGraph node with three tool calls re-runs all three; in Temporal each can be its own activity.
- Neither gives exactly-once side effects for free: a Temporal activity that crashes mid-call is retried (at-least-once), so idempotency keys are still required. Temporal shrinks the blast radius; it does not remove the need.
- Operational cost: Temporal cluster or managed service vs a Postgres table.
- Other engines in the same family: AWS Step Functions, Azure Durable Functions, Restate, DBOS, Inngest.

</details>

<details>
<summary><b>Q11. How do you prevent infinite loops and runaway cost in an agent graph?</b></summary>

**Short answer:** Bound every cycle with a counter in state and a conditional edge that exits after N attempts, set a recursion limit on super-steps, track token and dollar spend per thread and route to an error node when a budget is exceeded, add wall-clock timeouts per node and run, and alert on runs that approach limits. Exceeding a limit should end in a graceful escalation, not a crash.

**Deeper answer and follow-ups:**
- Detect no-progress loops: same tool with same arguments twice in a row, or identical plans across attempts.
- Make validation feedback specific so loops converge.
- Use cheaper models for routers and retries where possible.
- Monitor the distribution of steps per run; a shift often signals a prompt or model regression.

</details>

<details>
<summary><b>Q12. You need to deploy a new graph version while hundreds of threads are paused waiting for approval. What do you do?</b></summary>

**Short answer:** Treat the graph and its state schema like a database schema with live data. Prefer backward-compatible changes (new optional fields with defaults, new nodes, unchanged node names), record the graph version in thread metadata, and either keep the old version available until its threads drain or write an explicit migration that rewrites old checkpoints. Test by resuming real paused checkpoints from staging against the new code.

**Deeper answer and follow-ups:**
- Renaming or deleting a node that a paused checkpoint lists in `next` breaks resume.
- Changing a field's type or reducer can corrupt merges for in-flight threads.
- Route by version at the API layer: old threads to old workers, new threads to new workers.
- Add a contract test that loads a corpus of saved checkpoints and resumes each.

</details>

<details>
<summary><b>Q13. How do you choose an agent framework, or decide to build your own?</b></summary>

**Short answer:** Start from requirements (explicit control flow, recovery semantics, HITL, OpenTelemetry-friendly tracing, testability without an LLM, model neutrality, API stability, escape hatches, license), shortlist two or three, and run a bake-off on a real workflow with an approval step, a flaky tool and a kill test. Build your own only with strong platform engineering, usually on a durable engine.

**Deeper answer and follow-ups:**
- Framework value is mostly persistence, HITL and tracing; a basic loop is 20 lines.
- Limit lock-in by keeping tools, prompts and business rules framework-agnostic.

</details>

<details>
<summary><b>Q14. How do you run parallel branches in a graph, for example grading 10 retrieved documents at once?</b></summary>

**Short answer:** Use fan-out and fan-in. A router returns a list of nodes or, for a variable number of items, a list of `Send("grade_doc", {"doc": d})` objects so the same node runs once per document in a single super-step. Their results go into a state field with an append reducer, and a join node runs after all branches finish.

**Deeper answer and follow-ups:**
- Latency becomes roughly the slowest branch instead of the sum; cost is unchanged (same number of calls).
- Respect provider rate limits: cap concurrency or batch.
- If one branch fails, successful branches' writes are kept as pending writes; only the failed one re-runs.
- Do not depend on output order; sort by a key in the join node.

</details>

---

## Key takeaways

- **Orchestration** decides which step runs next, what it sees, and what happens on failure; it is separate from the model's reasoning.
- Loops (model decides), chains (fixed order) and state graphs (code-defined structure, model inside nodes) sit on a dial from flexibility to control.
- Prefer a state graph when you need **branching, cycles, persistence, human approval, recovery or auditability**; keep simple pipelines as chains.
- Reliability compounds: $p^n$ (0.95 per step over 10 steps is about 0.60), so keep model-driven steps few and add targeted retries.
- LangGraph = **typed state + nodes + edges + conditional edges + reducers**, compiled and run in **super-steps**; nodes return partial updates.
- **Reducers** define merges; parallel writes to one key need one.
- **Checkpointers** save state after each super-step under a **thread ID**; resume with the same thread; use Postgres-class storage in production.
- Execution is **at-least-once per node**; combine with **idempotency keys** to get exactly-once effects.
- **`interrupt()` + `Command(resume=...)`** implements human approval; the node re-runs on resume, so no side effects above the interrupt.
- **Time travel** (state history, replay, fork via `update_state`) is for debugging, reproduction and undo.
- Classify failures: transient (retry with backoff, one layer), invalid output (feedback loop with a cap), permanent or policy (error node or human).
- **Durable engines** (Temporal-style) replay event histories with first-class timeouts and signals; use them for long, multi-service processes, possibly wrapping a LangGraph agent.
- Trace **decisions**, not just calls: router labels, edges taken, tool arguments, approvals, with thread and checkpoint IDs.
- Evaluate end states, **pass^k**, trajectories, policy compliance and recovery under fault injection; gate releases in CI.

## Further reading

- LangGraph official documentation (concepts: low-level graph API, persistence, human-in-the-loop, time travel, durable execution).
- "Pregel: A System for Large-Scale Graph Processing", Malewicz et al., 2010 (SIGMOD). The super-step model behind LangGraph's runtime.
- "ReAct: Synergizing Reasoning and Acting in Language Models", Yao et al., 2022. The loop that graphs structure.
- "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains", Yao et al., 2024. Introduces pass^k for agent consistency.
- "Statecharts: A Visual Formalism for Complex Systems", David Harel, 1987. Hierarchical state machines, the ancestor of subgraphs.
- "Sagas", Hector Garcia-Molina and Kenneth Salem, 1987 (SIGMOD). Compensating actions for long-lived transactions.
- Temporal documentation (workflows, activities, event history, signals, retry policies).
- OpenTelemetry Semantic Conventions for Generative AI (GenAI spans and attributes).
- LangSmith and Langfuse documentation (tracing LangGraph applications).
- OpenAI Agents SDK, LlamaIndex Workflows, CrewAI, AutoGen and Semantic Kernel official documentation.
- *Designing Data-Intensive Applications*, Martin Kleppmann, 2017. Idempotency, exactly-once semantics, event sourcing.

## Next

Continue with **[Multi-Agent Systems: When They Help and When They Hurt](05-multi-agent-systems.md)**.

Related chapters:

- [Agent Design Patterns: ReAct, Plan-and-Execute, Reflection and Routing](03-agent-design-patterns.md)
- [Memory for AI Agents](06-agent-memory.md)
- [Making Agents Reliable and Secure for Production](08-agent-reliability-security-and-prompt-injection.md)
- [Evaluating and Observing Agents](09-evaluating-and-observing-agents.md)
- [Case Study: Designing a Production Procurement and Support Agent](10-case-study-designing-a-production-agent.md)
- [LLMOps: Tracing, Prompt Management, Langfuse and RAGAS in Practice](../10-mlops/08-llmops-tracing-prompt-management-and-langfuse.md)
- [Coding Exercises: Transformers, LLMs, RAG and Agents](../12-interview-practice/02-coding-exercises-llm-rag-and-agents.md)
