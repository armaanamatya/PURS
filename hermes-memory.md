# Hermes Agent — Memory System

Notes on how [NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent) stores, updates, and prompts around memory.

---

## 1. Storage layout

All persistent state lives under `~/.hermes/`:

| Path | Purpose | Cap |
|---|---|---|
| `~/.hermes/memories/MEMORY.md` | Agent's own durable notes (env facts, conventions, lessons learned) | ~2,200 chars / ~800 tokens |
| `~/.hermes/memories/USER.md` | User profile (name, GitHub handle, preferences, style) | ~1,375 chars / ~500 tokens |
| `~/.hermes/SOUL.md` | Agent persona / identity | n/a |
| `~/.hermes/state.db` | SQLite DB of all CLI + messaging sessions, indexed with **FTS5** | n/a |
| `skills/` | Procedural memory — autonomously generated reusable workflows | n/a |

Entries inside `MEMORY.md` / `USER.md` are separated by `§` (section sign) delimiters. Usage percentages are shown in the system prompt header.

---

## 2. Update mechanism

### Built-in memory (`MEMORY.md`, `USER.md`)
- Agent-curated via an **agent-level `memory` tool** (`add` / `replace` / `remove`).
- Intercepted in `run_agent.py` before normal `handle_function_call()` dispatch.
- Writes hit disk **immediately**, but the system prompt is a **frozen snapshot taken at session start** — mid-session edits only appear next session. This preserves prompt-cache stability.
- No `read` action — content is auto-injected into the system prompt, so the agent already sees it.
- When at capacity, the tool errors and returns existing entries; agent must consolidate. Best practice: prune at ~80% full.
- Cron sessions pass `skip_memory=True` — memory providers don't run on scheduled jobs.

### Session memory (FTS5 in `state.db`)
- Every turn of every session is written to SQLite.
- Cross-session recall: agent queries FTS5, results are **summarized by Gemini Flash** before being returned.

### Skills (procedural memory)
- After complex tasks, the agent autonomously writes a reusable skill file.
- Skills self-improve through repeated use.
- Follows the `agentskills.io` open standard.

### Honcho (optional plugin) — dialectic user model
Two independent update cadences:
- `contextCadence` (default **every turn**) — refreshes user representation + session summary.
- `dialecticCadence` (default **every 2 turns**) — runs multi-pass LLM reasoning to derive **conclusions** about user patterns/goals.

Honcho stores:
- User representation (auto-built, not manually curated)
- Session summaries
- Per-peer profiles (separate per agent — prevents cross-contamination)
- Conclusions (server-side pattern reasoning)
- Messages (optional, `saveMessages` flag)

Retrieved via:
- Base-context layer injected into the system prompt
- Explicit tools: `honcho_search` (semantic), `honcho_reasoning` (synthesized answers, depth-configurable)

---

## 3. Services & components used

| Component | Role |
|---|---|
| **SQLite + FTS5** | Session storage + full-text search |
| **Gemini Flash** | Summarizes FTS5 search hits for cross-session recall |
| **Honcho** | AI-native memory backend (dialectic reasoning, semantic search) |
| **Filesystem (`~/.hermes/`)** | Markdown-based durable storage for memory/user/persona |
| **`agent/memory_provider.py`** | ABC each backend implements |
| **`agent/memory_manager.py`** | Orchestrates providers |
| **`agent/prompt_builder.py`** | Assembles the system prompt |
| **`run_agent.py`** | Intercepts agent-level memory/todo tool calls |

Config lives under `memory:` in `config.yaml`. 8 external memory provider plugins ship (Honcho is the headline one).

---

## 4. Prompt assembly

`agent/prompt_builder.py` builds the cached system prompt in this order:

1. **Agent identity** — `SOUL.md` (or default fallback)
2. **Tool-aware behavior guidance** — memory + tool usage instructions
3. **Honcho static block** — if active (personality/context)
4. **Optional system message** — from config or API
5. **Frozen MEMORY snapshot** — `MEMORY.md` content
6. **Frozen USER profile snapshot** — `USER.md` content
7. **Skills index** — compact reference to available skills
8. **Project context file** — first match wins:
   - `.hermes.md` / `HERMES.md` (walks to git root)
   - `AGENTS.md` (cwd)
   - `CLAUDE.md` (cwd)
   - `.cursorrules` / `.cursor/rules/*.mdc` (cwd)
   - All security-scanned, truncated at 20,000 chars
9. **Timestamp + session ID**
10. **Platform hint** — CLI-specific rendering guidance

Tool schemas use **lazy loading**: only names + one-line descriptions (~300–500 tokens) ship by default; full schemas (3,500–5,000 tokens for 50+ tools) load on demand when a tool is selected.

### What the memory-tool prompt tells the agent
- Save durable facts only: user preferences, env details, tool quirks, stable conventions.
- Memory is injected every turn → keep compact, fact-dense.
- No read action — content is already visible in context.
- Consolidate proactively as capacity approaches the cap.

---

## 5. TL;DR

- **Built-in:** agent-curated markdown files with hard char caps, written via `memory` tool, frozen into the prompt at session start.
- **Sessions:** every turn → SQLite/FTS5, recalled via search + Gemini Flash summary.
- **Honcho:** automatic post-turn LLM reasoning, no caps, semantic + dialectic retrieval.
- **Skills:** autonomously generated procedural memory that self-improves.

---

## Sources

- [Hermes Agent README](https://github.com/nousresearch/hermes-agent)
- [Built-in memory docs](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/memory.md)
- [Honcho docs](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/honcho.md)
- [Prompt assembly docs](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/prompt-assembly.md)
- [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md)
- [Persistent Memory site](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)
- [Memory Providers site](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers)
