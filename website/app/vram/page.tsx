"use client";

import { useEffect, useState, useMemo } from "react";

interface VramEntry {
  entry: string;
  task_type: string;
  duration_s: number;
  peak_vram_gb: number;
  after_vram_gb: number;
}

function fmt(n: number, d = 1) { return n.toFixed(d); }

// ── Simple SVG bar chart ───────────────────────────────────────────────────────

function BarChart({
  data,
  xKey,
  bars,
  height = 220,
}: {
  data: Record<string, number>[];
  xKey: string;
  bars: { key: string; color: string; label: string }[];
  height?: number;
}) {
  const W = 900;
  const H = height;
  const PAD = { top: 16, right: 16, bottom: 80, left: 52 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const allVals = data.flatMap((d) => bars.map((b) => d[b.key] as number));
  const maxVal = Math.max(...allVals, 0);
  const yMax = Math.ceil(maxVal / 20) * 20 || 10;

  const groupW = innerW / data.length;
  const barW = Math.min(groupW / bars.length - 2, 28);

  const yTicks = Array.from({ length: 5 }, (_, i) => (yMax / 4) * i);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
      {/* Grid lines */}
      {yTicks.map((t) => {
        const y = PAD.top + innerH - (t / yMax) * innerH;
        return (
          <g key={t}>
            <line x1={PAD.left} y1={y} x2={PAD.left + innerW} y2={y}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <text x={PAD.left - 6} y={y + 4} textAnchor="end"
              fontSize={9} fill="rgba(255,255,255,0.3)">{fmt(t, 0)}</text>
          </g>
        );
      })}

      {/* Bars */}
      {data.map((d, gi) => {
        const groupX = PAD.left + gi * groupW + groupW / 2;
        const totalBarW = bars.length * (barW + 2) - 2;
        const startX = groupX - totalBarW / 2;
        return (
          <g key={gi}>
            {bars.map((b, bi) => {
              const val = d[b.key] as number;
              const bH = (val / yMax) * innerH;
              const x = startX + bi * (barW + 2);
              const y = PAD.top + innerH - bH;
              return (
                <g key={b.key}>
                  <rect x={x} y={y} width={barW} height={bH}
                    fill={b.color} rx={2} opacity={0.85} />
                  {bH > 14 && (
                    <text x={x + barW / 2} y={y + 10} textAnchor="middle"
                      fontSize={7} fill="rgba(255,255,255,0.5)">{fmt(val)}</text>
                  )}
                </g>
              );
            })}
            {/* X label */}
            <text
              x={groupX}
              y={PAD.top + innerH + 12}
              textAnchor="end"
              fontSize={8}
              fill="rgba(255,255,255,0.4)"
              transform={`rotate(-35, ${groupX}, ${PAD.top + innerH + 12})`}
            >
              {(d[xKey] as string).length > 22 ? (d[xKey] as string).slice(0, 21) + "…" : d[xKey] as string}
            </text>
          </g>
        );
      })}

      {/* Axes */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH}
        stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
      <line x1={PAD.left} y1={PAD.top + innerH} x2={PAD.left + innerW} y2={PAD.top + innerH}
        stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
    </svg>
  );
}

// ── Scatter plot ───────────────────────────────────────────────────────────────

function ScatterPlot({ data }: { data: VramEntry[] }) {
  const [hover, setHover] = useState<VramEntry | null>(null);
  const W = 900;
  const H = 280;
  const PAD = { top: 16, right: 16, bottom: 40, left: 52 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const maxX = Math.ceil(Math.max(...data.map((d) => d.duration_s)) / 10) * 10 || 10;
  const maxY = Math.ceil(Math.max(...data.map((d) => d.peak_vram_gb)) / 20) * 20 || 20;

  const xTicks = Array.from({ length: 5 }, (_, i) => (maxX / 4) * i);
  const yTicks = Array.from({ length: 5 }, (_, i) => (maxY / 4) * i);

  const px = (v: number) => PAD.left + (v / maxX) * innerW;
  const py = (v: number) => PAD.top + innerH - (v / maxY) * innerH;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
        {/* Grid */}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={px(t)} y1={PAD.top} x2={px(t)} y2={PAD.top + innerH}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <text x={px(t)} y={PAD.top + innerH + 14} textAnchor="middle"
              fontSize={9} fill="rgba(255,255,255,0.3)">{fmt(t, 0)}s</text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.left} y1={py(t)} x2={PAD.left + innerW} y2={py(t)}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <text x={PAD.left - 6} y={py(t) + 4} textAnchor="end"
              fontSize={9} fill="rgba(255,255,255,0.3)">{fmt(t, 0)}</text>
          </g>
        ))}

        {/* Points */}
        {data.map((d, i) => (
          <circle
            key={i}
            cx={px(d.duration_s)} cy={py(d.peak_vram_gb)}
            r={5} fill="rgba(139,92,246,0.7)" stroke="rgba(139,92,246,0.4)" strokeWidth={1}
            className="cursor-pointer hover:fill-violet-300 transition-colors"
            onMouseEnter={() => setHover(d)}
            onMouseLeave={() => setHover(null)}
          />
        ))}

        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH}
          stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
        <line x1={PAD.left} y1={PAD.top + innerH} x2={PAD.left + innerW} y2={PAD.top + innerH}
          stroke="rgba(255,255,255,0.12)" strokeWidth={1} />

        {/* Axis labels */}
        <text x={PAD.left + innerW / 2} y={H - 2} textAnchor="middle"
          fontSize={9} fill="rgba(255,255,255,0.25)">duration (s)</text>
        <text x={10} y={PAD.top + innerH / 2} textAnchor="middle"
          fontSize={9} fill="rgba(255,255,255,0.25)"
          transform={`rotate(-90, 10, ${PAD.top + innerH / 2})`}>peak VRAM (GB)</text>
      </svg>

      {hover && (
        <div className="absolute top-4 right-4 bg-[#0e0e14] border border-white/10 rounded-lg px-3 py-2 text-xs font-mono pointer-events-none">
          <p className="text-white/70 mb-1">{hover.task_type}</p>
          <p className="text-violet-300">peak: {fmt(hover.peak_vram_gb)} GB</p>
          <p className="text-cyan-300">after: {fmt(hover.after_vram_gb)} GB</p>
          <p className="text-white/40">dur: {fmt(hover.duration_s)}s</p>
        </div>
      )}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function VramPage() {
  const [raw, setRaw] = useState<VramEntry[]>([]);

  useEffect(() => {
    fetch("/vram_log.jsonl")
      .then((r) => r.text())
      .then((text) => {
        const lines = text.trim().split("\n").filter(Boolean);
        const parsed: VramEntry[] = lines.map((l) => JSON.parse(l));
        // Deduplicate: keep one per task_type (take first occurrence)
        const seen = new Set<string>();
        const deduped = parsed.filter((e) => {
          if (seen.has(e.entry)) return false;
          seen.add(e.entry);
          return true;
        });
        setRaw(deduped);
      });
  }, []);

  const sorted = useMemo(
    () => [...raw].sort((a, b) => b.peak_vram_gb - a.peak_vram_gb),
    [raw]
  );

  const stats = useMemo(() => {
    if (!raw.length) return null;
    const peaks = raw.map((d) => d.peak_vram_gb);
    const afters = raw.map((d) => d.after_vram_gb);
    const durs = raw.map((d) => d.duration_s);
    const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
    return {
      avgPeak: avg(peaks),
      maxPeak: Math.max(...peaks),
      minPeak: Math.min(...peaks),
      avgAfter: avg(afters),
      avgDur: avg(durs),
      maxDur: Math.max(...durs),
      count: raw.length,
    };
  }, [raw]);

  const barData = useMemo(
    () =>
      sorted.map((d) => ({
        task_type: d.task_type,
        peak_vram_gb: d.peak_vram_gb,
        after_vram_gb: d.after_vram_gb,
      })),
    [sorted]
  );

  if (!raw.length) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <div className="text-white/40 font-mono text-sm animate-pulse">Loading…</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      {/* Header */}
      <header className="border-b border-white/8 px-6 py-4 sticky top-0 bg-[#0a0a0f]/95 backdrop-blur-sm z-30">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">VRAM Usage</h1>
            <p className="text-xs text-white/35 font-mono mt-0.5">
              {stats?.count} task types · peak / after / duration
            </p>
          </div>
          <a href="/" className="text-xs font-mono text-white/30 hover:text-white/60 transition-colors">
            ← eval results
          </a>
        </div>
      </header>

      <div className="px-6 py-6 flex flex-col gap-8 max-w-[1100px]">
        {/* Stat cards */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {[
              { label: "Avg Peak VRAM", value: fmt(stats.avgPeak) + " GB", color: "text-violet-300" },
              { label: "Max Peak VRAM", value: fmt(stats.maxPeak) + " GB", color: "text-rose-300" },
              { label: "Min Peak VRAM", value: fmt(stats.minPeak) + " GB", color: "text-emerald-300" },
              { label: "Avg After VRAM", value: fmt(stats.avgAfter) + " GB", color: "text-cyan-300" },
              { label: "Avg Duration", value: fmt(stats.avgDur) + " s", color: "text-amber-300" },
            ].map(({ label, value, color }) => (
              <div key={label} className="border border-white/10 rounded-lg px-4 py-3 bg-white/[0.02]">
                <p className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-1">{label}</p>
                <p className={`text-xl font-mono font-bold ${color}`}>{value}</p>
              </div>
            ))}
          </div>
        )}

        {/* Bar chart: peak vs after VRAM */}
        <div className="border border-white/8 rounded-xl bg-white/[0.01] p-4">
          <div className="flex items-center gap-4 mb-4">
            <p className="text-sm font-mono text-white/70">Peak vs After VRAM by Task Type</p>
            <div className="flex items-center gap-3 ml-auto text-[10px] font-mono text-white/40">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-violet-500/80 inline-block" /> peak
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-cyan-500/80 inline-block" /> after
              </span>
            </div>
          </div>
          <BarChart
            data={barData}
            xKey="task_type"
            bars={[
              { key: "peak_vram_gb", color: "rgba(139,92,246,0.8)", label: "Peak" },
              { key: "after_vram_gb", color: "rgba(6,182,212,0.7)", label: "After" },
            ]}
            height={260}
          />
        </div>

        {/* Scatter: duration vs peak VRAM */}
        <div className="border border-white/8 rounded-xl bg-white/[0.01] p-4">
          <p className="text-sm font-mono text-white/70 mb-4">Duration vs Peak VRAM</p>
          <ScatterPlot data={raw} />
        </div>

        {/* Table */}
        <div className="border border-white/8 rounded-xl bg-white/[0.01] overflow-hidden">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-white/8 text-white/30 uppercase text-[10px] tracking-widest">
                <th className="px-4 py-2.5 text-left">Task Type</th>
                <th className="px-4 py-2.5 text-right">Peak VRAM</th>
                <th className="px-4 py-2.5 text-right">After VRAM</th>
                <th className="px-4 py-2.5 text-right">VRAM Freed</th>
                <th className="px-4 py-2.5 text-right">Duration</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((d, i) => {
                const freed = d.peak_vram_gb - d.after_vram_gb;
                return (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/[0.015] transition-colors">
                    <td className="px-4 py-2 text-white/70">{d.task_type}</td>
                    <td className="px-4 py-2 text-right text-violet-300">{fmt(d.peak_vram_gb)} GB</td>
                    <td className="px-4 py-2 text-right text-cyan-300">{fmt(d.after_vram_gb)} GB</td>
                    <td className="px-4 py-2 text-right text-emerald-300">{fmt(freed)} GB</td>
                    <td className="px-4 py-2 text-right text-amber-300/80">{fmt(d.duration_s)}s</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
