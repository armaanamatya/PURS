"use client";

import { useEffect, useState, useMemo, useRef } from "react";

interface Question {
  question: string;
  choices: string[];
  answer: string;
  task_type: string;
  prediction: string | null;
  correct: boolean | null;
  reasoning: string | null;
}

interface Entry {
  dataset: string;
  task_type: string;
  duration_s: number | null;
  video_url: string;
  questions: Question[];
}

interface DatasetStats { correct: number; total: number; }

interface Data {
  generated_at: string;
  stats: {
    total_correct: number;
    total_questions: number;
    accuracy: number;
    by_dataset: Record<string, DatasetStats>;
  };
  entries: Entry[];
}

const DATASET_COLORS: Record<string, string> = {
  "video-mme":      "bg-violet-500/20 text-violet-300 border-violet-500/30",
  "daily-omni":     "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
  "worldsense":     "bg-amber-500/20 text-amber-300 border-amber-500/30",
  "omnivideobench": "bg-rose-500/20 text-rose-300 border-rose-500/30",
  "shortvid-bench": "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
};
const DATASET_DOT: Record<string, string> = {
  "video-mme":      "bg-violet-400",
  "daily-omni":     "bg-cyan-400",
  "worldsense":     "bg-amber-400",
  "omnivideobench": "bg-rose-400",
  "shortvid-bench": "bg-emerald-400",
};

function pct(n: number) { return (n * 100).toFixed(1) + "%"; }

function AccuracyBar({ value }: { value: number }) {
  return (
    <div className="h-1 w-full rounded-full bg-white/10 overflow-hidden mt-1">
      <div className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-300"
        style={{ width: pct(value) }} />
    </div>
  );
}

// ── Question cell ─────────────────────────────────────────────────────────────

function QuestionCell({ q }: { q: Question }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex flex-col gap-2 min-w-[240px] max-w-[320px]">
      <p className="text-xs text-white/70 leading-snug break-words">{q.question}</p>
      <div className="flex flex-col gap-1">
        {q.choices.map((choice, ci) => {
          const letter = String.fromCharCode(65 + ci);
          const isAnswer = q.answer === letter;
          const isPred = q.prediction === letter;
          const isCorrectPred = isPred && q.correct;
          const isWrongPred = isPred && !q.correct;

          let cls = "flex items-start gap-1.5 px-2 py-1 rounded text-[11px] border ";
          if (isCorrectPred)            cls += "bg-emerald-500/20 border-emerald-500/40 text-emerald-200";
          else if (isWrongPred)         cls += "bg-red-500/20 border-red-500/40 text-red-300";
          else if (isAnswer && !isPred) cls += "bg-emerald-500/10 border-emerald-500/20 text-emerald-300/60";
          else                          cls += "border-white/6 text-white/35";

          return (
            <div key={ci} className={cls}>
              <span className="font-mono font-bold shrink-0 text-[10px] pt-0.5">{letter}</span>
              <span className="flex-1 min-w-0 break-words">{choice.replace(/^[A-D]\.\s*/, "")}</span>
              {isAnswer && !isPred && <span className="text-[9px] text-emerald-400 font-mono shrink-0">ANS</span>}
              {isCorrectPred && <span className="text-[9px] text-emerald-400 font-mono shrink-0">✓</span>}
              {isWrongPred && <span className="text-[9px] text-red-400 font-mono shrink-0">✗</span>}
            </div>
          );
        })}
      </div>
      {/* Per-question reasoning */}
      {q.reasoning && (
        <div>
          <button onClick={() => setOpen(!open)}
            className="text-[10px] text-white/25 hover:text-white/50 transition-colors flex items-center gap-1">
            <span className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>▶</span>
            reasoning
          </button>
          {open && (
            <pre className="mt-1 text-[10px] font-mono text-white/40 whitespace-pre-wrap break-words leading-relaxed bg-black/30 rounded p-2 border border-white/6 max-h-48 overflow-y-auto">
              {q.reasoning}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ── Table row ─────────────────────────────────────────────────────────────────

function TableRow({ entry, maxQ }: { entry: Entry; maxQ: number }) {
  const answered = entry.questions.filter((q) => q.prediction !== null);
  const correct = answered.filter((q) => q.correct).length;

  return (
    <tr className="border-b border-white/6 hover:bg-white/[0.015] transition-colors align-top">
      {/* Clip */}
      <td className="p-3 sticky left-0 bg-[#0a0a0f] z-10 border-r border-white/6 min-w-[200px]">
        <div className="rounded-lg overflow-hidden bg-black border border-white/8 aspect-video w-48">
          <video src={entry.video_url} controls preload="metadata"
            className="w-full h-full object-contain" />
        </div>
        <div className="mt-2 flex flex-col gap-1">
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border w-fit ${DATASET_COLORS[entry.dataset] ?? "bg-white/10 text-white/60 border-white/20"}`}>
            {entry.dataset}
          </span>
          <span className="text-[11px] text-white/60 leading-tight">{entry.task_type}</span>
          {entry.duration_s && <span className="text-[10px] font-mono text-white/25">{entry.duration_s}s</span>}
          {answered.length > 0 && (
            <span className={`text-[10px] font-mono font-bold ${correct === answered.length ? "text-emerald-400" : correct === 0 ? "text-red-400" : "text-amber-400"}`}>
              {correct}/{answered.length} correct
            </span>
          )}
        </div>
      </td>

      {/* Question columns */}
      {Array.from({ length: maxQ }).map((_, qi) => {
        const q = entry.questions[qi];
        return (
          <td key={qi} className="p-3 border-r border-white/6 align-top">
            {q ? <QuestionCell q={q} /> : <span className="text-white/15 text-xs">—</span>}
          </td>
        );
      })}

      {/* Prediction summary */}
      <td className="p-3 align-top min-w-[120px]">
        <div className="flex flex-col gap-1.5">
          {entry.questions.map((q, qi) => (
            <div key={qi} className="flex items-center gap-1.5">
              <span className="text-[10px] text-white/25 font-mono w-4">Q{qi + 1}</span>
              {q.prediction ? (
                <span className={`font-mono text-sm font-bold ${q.correct ? "text-emerald-400" : "text-red-400"}`}>
                  {q.prediction}
                </span>
              ) : (
                <span className="text-white/20 text-xs">—</span>
              )}
              <span className="text-white/20 text-[10px]">→ {q.answer}</span>
            </div>
          ))}
        </div>
      </td>
    </tr>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

const ENTRIES_BAR_HEIGHT = 40; // fallback; measured dynamically below

export default function Home() {
  const [data, setData] = useState<Data | null>(null);
  const [activeDataset, setActiveDataset] = useState<string | null>(null);
  const [activeTaskType, setActiveTaskType] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const headerRef = useRef<HTMLElement>(null);
  const entriesBarRef = useRef<HTMLDivElement>(null);
  const [headerHeight, setHeaderHeight] = useState(73);
  const [entriesBarHeight, setEntriesBarHeight] = useState(ENTRIES_BAR_HEIGHT);

  useEffect(() => {
    fetch("/data.json").then((r) => r.json()).then(setData);
  }, []);

  useEffect(() => {
    const el = headerRef.current;
    if (!el) return;
    const update = () => setHeaderHeight(el.getBoundingClientRect().height);
    const ro = new ResizeObserver(update);
    ro.observe(el);
    update();
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = entriesBarRef.current;
    if (!el) return;
    const update = () => setEntriesBarHeight(el.getBoundingClientRect().height);
    const ro = new ResizeObserver(update);
    ro.observe(el);
    update();
    return () => ro.disconnect();
  }, [data]);

  const datasets = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.entries.map((e) => e.dataset))].sort();
  }, [data]);

  const taskTypes = useMemo(() => {
    if (!data) return [];
    const src = activeDataset ? data.entries.filter((e) => e.dataset === activeDataset) : data.entries;
    return [...new Set(src.map((e) => e.task_type))].sort();
  }, [data, activeDataset]);

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.entries.filter((e) => {
      if (activeDataset && e.dataset !== activeDataset) return false;
      if (activeTaskType && e.task_type !== activeTaskType) return false;
      if (search) {
        const sq = search.toLowerCase();
        if (!e.task_type.toLowerCase().includes(sq) && !e.dataset.toLowerCase().includes(sq) &&
          !e.questions.some((q) => q.question.toLowerCase().includes(sq))) return false;
      }
      return true;
    });
  }, [data, activeDataset, activeTaskType, search]);

  const maxQ = useMemo(() => Math.max(1, ...filtered.map((e) => e.questions.length)), [filtered]);

  if (!data) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <div className="text-white/40 font-mono text-sm animate-pulse">Loading…</div>
      </div>
    );
  }

  const { stats } = data;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex flex-col">
      {/* Header */}
      <header ref={headerRef} className="border-b border-white/8 bg-[#0a0a0f]/95 backdrop-blur-sm sticky top-0 z-30">
        <div className="px-6 py-4 flex items-start justify-between gap-6 flex-wrap">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Qwen2.5-Omni Eval</h1>
            <p className="text-xs text-white/35 font-mono mt-0.5">
              {data.entries.length} clips · {stats.total_questions} questions · video + audio benchmark
            </p>
          </div>
          <div className="flex items-center gap-8 flex-wrap">
            <div>
              <p className="text-2xl font-mono font-bold">{pct(stats.accuracy)}</p>
              <p className="text-xs text-white/35 font-mono">{stats.total_correct}/{stats.total_questions} overall</p>
            </div>
            <div className="flex gap-5 flex-wrap">
              {Object.entries(stats.by_dataset).map(([ds, s]) => (
                <div key={ds} className="min-w-[72px]">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className={`w-2 h-2 rounded-full ${DATASET_DOT[ds] ?? "bg-white/30"}`} />
                    <span className="text-[10px] font-mono text-white/40">{ds}</span>
                  </div>
                  <p className="text-sm font-mono font-bold text-white/80">{pct(s.correct / s.total)}</p>
                  <AccuracyBar value={s.correct / s.total} />
                  <p className="text-[10px] font-mono text-white/30 mt-0.5">{s.correct}/{s.total}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        {/* Sidebar */}
        <aside
          className="w-56 shrink-0 border-r border-white/8 sticky overflow-y-auto py-4 px-3 pb-24 hidden md:block"
          style={{ top: headerHeight, height: `calc(100vh - ${headerHeight}px)` }}
        >
          <input type="text" placeholder="Search…" value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-white/25 focus:outline-none focus:border-white/25 mb-4 font-mono" />

          <p className="text-[10px] font-mono uppercase tracking-widest text-white/25 mb-2 px-1">Dataset</p>
          <div className="flex flex-col gap-0.5 mb-5">
            <button onClick={() => { setActiveDataset(null); setActiveTaskType(null); }}
              className={`text-left px-3 py-1.5 rounded-lg text-xs transition-colors ${!activeDataset ? "bg-white/10 text-white" : "text-white/45 hover:text-white/70 hover:bg-white/5"}`}>
              All
            </button>
            {datasets.map((ds) => {
              const s = stats.by_dataset[ds];
              return (
                <button key={ds} onClick={() => { setActiveDataset(ds); setActiveTaskType(null); }}
                  className={`text-left px-3 py-1.5 rounded-lg text-xs transition-colors flex items-center gap-2 ${activeDataset === ds ? "bg-white/10 text-white" : "text-white/45 hover:text-white/70 hover:bg-white/5"}`}>
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${DATASET_DOT[ds] ?? "bg-white/30"}`} />
                  <span className="flex-1 truncate">{ds}</span>
                  {s && <span className="text-[10px] font-mono text-white/25">{(s.correct / s.total * 100).toFixed(0)}%</span>}
                </button>
              );
            })}
          </div>

          <p className="text-[10px] font-mono uppercase tracking-widest text-white/25 mb-2 px-1">Task Type</p>
          <div className="flex flex-col gap-0.5 pb-8">
            <button onClick={() => setActiveTaskType(null)}
              className={`text-left px-3 py-1.5 rounded-lg text-xs transition-colors ${!activeTaskType ? "bg-white/10 text-white" : "text-white/45 hover:text-white/70 hover:bg-white/5"}`}>
              All
            </button>
            {taskTypes.map((tt) => (
              <button key={tt} onClick={() => setActiveTaskType(tt)}
                className={`text-left px-3 py-1.5 rounded-lg text-xs transition-colors break-words ${activeTaskType === tt ? "bg-white/10 text-white" : "text-white/45 hover:text-white/70 hover:bg-white/5"}`}>
                {tt}
              </button>
            ))}
          </div>
        </aside>

        {/* Table area - overflow-auto makes this the scroll container so sticky headers don't cover first row */}
        <main className="flex-1 min-h-0 overflow-auto">
          <div
            ref={entriesBarRef}
            className="px-4 py-3 border-b border-white/6 flex items-center gap-2 text-xs font-mono text-white/30 sticky top-0 bg-[#0a0a0f] z-20"
          >
            <span>{filtered.length} entries</span>
            {activeDataset && <><span>·</span><span className="text-white/50">{activeDataset}</span></>}
            {activeTaskType && <><span>·</span><span className="text-white/50">{activeTaskType}</span></>}
          </div>

          <table className="w-full border-separate border-spacing-0 text-sm">
            <thead>
              <tr
                className="border-b border-white/10 bg-[#0a0a0f] text-left sticky z-20"
                style={{ top: entriesBarHeight }}
              >
                <th className="px-3 py-2.5 text-[10px] font-mono uppercase tracking-widest text-white/30 sticky left-0 bg-[#0e0e14] border-r border-white/6 w-[212px]">
                  Clip
                </th>
                {Array.from({ length: maxQ }).map((_, i) => (
                  <th key={i} className="px-3 py-2.5 text-[10px] font-mono uppercase tracking-widest text-white/30 border-r border-white/6 min-w-[260px]">
                    Question {i + 1}
                  </th>
                ))}
                <th className="px-3 py-2.5 text-[10px] font-mono uppercase tracking-widest text-white/30 min-w-[120px]">
                  Prediction
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry, i) => (
                <TableRow key={`${entry.dataset}-${entry.task_type}-${i}`} entry={entry} maxQ={maxQ} />
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={maxQ + 2} className="text-center py-20 text-white/20 font-mono text-sm">
                    No entries match your filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </main>
      </div>
    </div>
  );
}
