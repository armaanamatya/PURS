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

function AccuracyBar({ value, color = "emerald" }: { value: number; color?: string }) {
  const gradient = color === "blue"
    ? "from-blue-500 to-blue-300"
    : "from-emerald-500 to-emerald-300";
  return (
    <div className="h-1 w-full rounded-full bg-white/10 overflow-hidden mt-1">
      <div className={`h-full rounded-full bg-gradient-to-r ${gradient}`}
        style={{ width: pct(value) }} />
    </div>
  );
}

// ── Choices panel for one model ───────────────────────────────────────────────

function ChoicesPanel({ q, label, labelColor }: { q: Question; label: string; labelColor: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className={`text-[9px] font-mono font-bold uppercase tracking-wider mb-0.5 ${labelColor}`}>
        {label}
      </span>
      {q.choices.map((choice, ci) => {
        const letter = String.fromCharCode(65 + ci);
        const isAnswer = q.answer === letter;
        const isPred = q.prediction === letter || q.prediction === "ERROR";
        const isError = q.prediction === "ERROR";
        const isCorrectPred = isPred && q.correct && !isError;
        const isWrongPred = (isPred && !q.correct) || isError;

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
            {isWrongPred && !isError && <span className="text-[9px] text-red-400 font-mono shrink-0">✗</span>}
            {isError && <span className="text-[9px] text-orange-400 font-mono shrink-0">ERR</span>}
          </div>
        );
      })}
      {q.prediction === "ERROR" && (
        <div className="text-[10px] text-orange-400/70 font-mono">codec error</div>
      )}
      {q.reasoning && (
        <div className="mt-0.5">
          <button onClick={() => setOpen(!open)}
            className="text-[10px] text-white/25 hover:text-white/50 transition-colors flex items-center gap-1">
            <span className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>▶</span>
            reasoning
          </button>
          {open && (
            <pre className="mt-1 text-[10px] font-mono text-white/40 whitespace-pre-wrap break-words leading-relaxed bg-black/30 rounded p-2 border border-white/6 max-h-40 overflow-y-auto">
              {q.reasoning}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ── Question cell (side-by-side) ──────────────────────────────────────────────

function QuestionCell({ q, qZip }: { q: Question; qZip: Question | null }) {
  return (
    <div className="flex flex-col gap-2 min-w-[460px] max-w-[600px]">
      <p className="text-xs text-white/70 leading-snug break-words">{q.question}</p>
      <div className="grid grid-cols-2 gap-2">
        <ChoicesPanel q={q} label="Qwen" labelColor="text-white/40" />
        {qZip
          ? <ChoicesPanel q={qZip} label="OmniZip" labelColor="text-blue-400/70" />
          : <div className="text-white/15 text-xs pt-4">—</div>
        }
      </div>
    </div>
  );
}

// ── Table row ─────────────────────────────────────────────────────────────────

function TableRow({ entry, entryZip, maxQ }: { entry: Entry; entryZip: Entry | null; maxQ: number }) {
  const baseAnswered = entry.questions.filter((q) => q.prediction !== null && q.prediction !== "ERROR");
  const baseCorrect = baseAnswered.filter((q) => q.correct).length;
  const zipAnswered = entryZip ? entryZip.questions.filter((q) => q.prediction !== null && q.prediction !== "ERROR") : [];
  const zipCorrect = zipAnswered.filter((q) => q.correct).length;

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
          {/* Per-model scores */}
          <div className="flex flex-col gap-0.5 mt-0.5">
            <span className={`text-[10px] font-mono ${baseCorrect === baseAnswered.length ? "text-emerald-400" : baseCorrect === 0 ? "text-red-400" : "text-amber-400"}`}>
              Qwen&nbsp;&nbsp;&nbsp;&nbsp;{baseCorrect}/{baseAnswered.length}
            </span>
            {entryZip && (
              <span className={`text-[10px] font-mono ${zipCorrect === zipAnswered.length ? "text-blue-400" : zipCorrect === 0 ? "text-red-400" : "text-amber-400"}`}>
                OmniZip&nbsp;{zipCorrect}/{zipAnswered.length}
              </span>
            )}
          </div>
        </div>
      </td>

      {/* Question columns */}
      {Array.from({ length: maxQ }).map((_, qi) => {
        const q = entry.questions[qi];
        const qZip = entryZip?.questions[qi] ?? null;
        return (
          <td key={qi} className="p-3 border-r border-white/6 align-top">
            {q
              ? <QuestionCell q={q} qZip={qZip} />
              : <span className="text-white/15 text-xs">—</span>
            }
          </td>
        );
      })}

      {/* Prediction summary */}
      <td className="p-3 align-top min-w-[160px]">
        <div className="flex flex-col gap-2">
          {entry.questions.map((q, qi) => {
            const qz = entryZip?.questions[qi];
            return (
              <div key={qi} className="flex flex-col gap-0.5">
                <span className="text-[9px] text-white/25 font-mono">Q{qi + 1} → {q.answer}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] text-white/30 w-10 font-mono">Qwen</span>
                  {q.prediction && q.prediction !== "ERROR" ? (
                    <span className={`font-mono text-sm font-bold ${q.correct ? "text-emerald-400" : "text-red-400"}`}>
                      {q.prediction}
                    </span>
                  ) : q.prediction === "ERROR" ? (
                    <span className="text-orange-400 text-xs font-mono">ERR</span>
                  ) : (
                    <span className="text-white/20 text-xs">—</span>
                  )}
                </div>
                {qz && (
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] text-blue-400/40 w-10 font-mono">Zip</span>
                    {qz.prediction && qz.prediction !== "ERROR" ? (
                      <span className={`font-mono text-sm font-bold ${qz.correct ? "text-blue-400" : "text-red-400"}`}>
                        {qz.prediction}
                      </span>
                    ) : qz.prediction === "ERROR" ? (
                      <span className="text-orange-400 text-xs font-mono">ERR</span>
                    ) : (
                      <span className="text-white/20 text-xs">—</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </td>
    </tr>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function Home() {
  const [data, setData] = useState<Data | null>(null);
  const [dataZip, setDataZip] = useState<Data | null>(null);
  const [activeDataset, setActiveDataset] = useState<string | null>(null);
  const [activeTaskType, setActiveTaskType] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const headerRef = useRef<HTMLElement>(null);
  const entriesBarRef = useRef<HTMLDivElement>(null);
  const [headerHeight, setHeaderHeight] = useState(73);
  const [entriesBarHeight, setEntriesBarHeight] = useState(40);

  useEffect(() => {
    fetch("/data.json").then((r) => r.json()).then(setData);
    fetch("/data_zip.json").then((r) => r.json()).then(setDataZip);
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

  // Build a lookup from video_url → zip entry
  const zipByUrl = useMemo(() => {
    if (!dataZip) return new Map<string, Entry>();
    return new Map(dataZip.entries.map((e) => [e.video_url, e]));
  }, [dataZip]);

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
  const zipStats = dataZip?.stats;

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

          {/* Model comparison cards */}
          <div className="flex items-start gap-6 flex-wrap">
            {/* Qwen */}
            <div className="border border-white/10 rounded-lg px-4 py-3 bg-white/[0.02]">
              <p className="text-[10px] font-mono text-white/35 uppercase tracking-widest mb-1">Qwen2.5-Omni</p>
              <p className="text-2xl font-mono font-bold">{pct(stats.accuracy)}</p>
              <p className="text-xs text-white/35 font-mono">{stats.total_correct}/{stats.total_questions}</p>
            </div>
            {/* OmniZip */}
            {zipStats && (
              <div className="border border-blue-500/30 rounded-lg px-4 py-3 bg-blue-500/[0.04]">
                <p className="text-[10px] font-mono text-blue-400/60 uppercase tracking-widest mb-1">+ OmniZip</p>
                <p className="text-2xl font-mono font-bold text-blue-300">{pct(zipStats.accuracy)}</p>
                <p className="text-xs text-blue-400/50 font-mono">{zipStats.total_correct}/{zipStats.total_questions}</p>
              </div>
            )}

            {/* Per-dataset split bars */}
            <div className="flex gap-5 flex-wrap items-start">
              {Object.entries(stats.by_dataset).map(([ds, s]) => {
                const zs = zipStats?.by_dataset[ds];
                return (
                  <div key={ds} className="min-w-[80px]">
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <span className={`w-2 h-2 rounded-full ${DATASET_DOT[ds] ?? "bg-white/30"}`} />
                      <span className="text-[10px] font-mono text-white/40">{ds}</span>
                    </div>
                    {/* Qwen bar */}
                    <p className="text-sm font-mono font-bold text-white/80">{pct(s.correct / s.total)}</p>
                    <AccuracyBar value={s.correct / s.total} />
                    <p className="text-[10px] font-mono text-white/30 mt-0.5">{s.correct}/{s.total}</p>
                    {/* OmniZip bar */}
                    {zs && (
                      <>
                        <p className="text-sm font-mono font-bold text-blue-300 mt-1">{pct(zs.correct / zs.total)}</p>
                        <AccuracyBar value={zs.correct / zs.total} color="blue" />
                        <p className="text-[10px] font-mono text-blue-400/40 mt-0.5">{zs.correct}/{zs.total}</p>
                      </>
                    )}
                  </div>
                );
              })}
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

        {/* Table */}
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
                  <th key={i} className="px-3 py-2.5 text-[10px] font-mono uppercase tracking-widest text-white/30 border-r border-white/6 min-w-[480px]">
                    <div className="flex items-center gap-3">
                      <span>Question {i + 1}</span>
                      <span className="text-white/20">·</span>
                      <span className="text-white/20">Qwen</span>
                      <span className="text-white/10">/</span>
                      <span className="text-blue-400/40">OmniZip</span>
                    </div>
                  </th>
                ))}
                <th className="px-3 py-2.5 text-[10px] font-mono uppercase tracking-widest text-white/30 min-w-[160px]">
                  Prediction
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry, i) => (
                <TableRow
                  key={`${entry.dataset}-${entry.task_type}-${i}`}
                  entry={entry}
                  entryZip={zipByUrl.get(entry.video_url) ?? null}
                  maxQ={maxQ}
                />
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
