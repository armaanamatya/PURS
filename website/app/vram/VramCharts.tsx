"use client";

import { useState, useMemo } from "react";
import type { VramEntry } from "./vramNormalize";

export type { VramEntry };

function fmt(n: number, d = 1) {
  return n.toFixed(d);
}

function BarChart({
  data,
  xKey,
  bars,
  height = 280,
}: {
  data: Record<string, number | string>[];
  xKey: string;
  bars: { key: string; color: string }[];
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
  const yTicks = Array.from({ length: 5 }, (_, i) => (yMax / 4) * i);

  const groupW = innerW / data.length;
  const barW = Math.min(groupW / bars.length - 1, bars.length > 3 ? 18 : 28);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
      {yTicks.map((t) => {
        const y = PAD.top + innerH - (t / yMax) * innerH;
        return (
          <g key={t}>
            <line
              x1={PAD.left}
              y1={y}
              x2={PAD.left + innerW}
              y2={y}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 6}
              y={y + 4}
              textAnchor="end"
              fontSize={9}
              fill="rgba(255,255,255,0.3)"
            >
              {fmt(t, 0)}
            </text>
          </g>
        );
      })}
      {data.map((d, gi) => {
        const groupX = PAD.left + gi * groupW + groupW / 2;
        const totalBarW = bars.length * (barW + 1) - 1;
        const startX = groupX - totalBarW / 2;
        return (
          <g key={gi}>
            {bars.map((b, bi) => {
              const val = d[b.key] as number;
              const bH = (val / yMax) * innerH;
              const x = startX + bi * (barW + 1);
              const y = PAD.top + innerH - bH;
              return (
                <g key={b.key}>
                  <rect
                    x={x}
                    y={y}
                    width={barW}
                    height={bH}
                    fill={b.color}
                    rx={2}
                    opacity={0.85}
                  />
                  {bH > 14 && (
                    <text
                      x={x + barW / 2}
                      y={y + 10}
                      textAnchor="middle"
                      fontSize={6}
                      fill="rgba(255,255,255,0.5)"
                    >
                      {fmt(val)}
                    </text>
                  )}
                </g>
              );
            })}
            <text
              x={groupX}
              y={PAD.top + innerH + 12}
              textAnchor="end"
              fontSize={8}
              fill="rgba(255,255,255,0.4)"
              transform={`rotate(-35, ${groupX}, ${PAD.top + innerH + 12})`}
            >
              {(d[xKey] as string).length > 22
                ? (d[xKey] as string).slice(0, 21) + "…"
                : (d[xKey] as string)}
            </text>
          </g>
        );
      })}
      <line
        x1={PAD.left}
        y1={PAD.top}
        x2={PAD.left}
        y2={PAD.top + innerH}
        stroke="rgba(255,255,255,0.12)"
        strokeWidth={1}
      />
      <line
        x1={PAD.left}
        y1={PAD.top + innerH}
        x2={PAD.left + innerW}
        y2={PAD.top + innerH}
        stroke="rgba(255,255,255,0.12)"
        strokeWidth={1}
      />
    </svg>
  );
}

type ScatterPoint = { entry: VramEntry; variant: "baseline" | "omnizip" };

function ScatterPlot({ points }: { points: ScatterPoint[] }) {
  const [hover, setHover] = useState<ScatterPoint | null>(null);
  const W = 900;
  const H = 280;
  const PAD = { top: 16, right: 16, bottom: 40, left: 52 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const maxX =
    points.length === 0
      ? 10
      : Math.ceil(
          Math.max(...points.map((p) => p.entry.duration_s), 0) / 10
        ) * 10 || 10;
  const maxY =
    points.length === 0
      ? 20
      : Math.ceil(
          Math.max(...points.map((p) => p.entry.peak_vram_gb), 0) / 20
        ) * 20 || 20;
  const xTicks = Array.from({ length: 5 }, (_, i) => (maxX / 4) * i);
  const yTicks = Array.from({ length: 5 }, (_, i) => (maxY / 4) * i);
  const px = (v: number) => PAD.left + (v / maxX) * innerW;
  const py = (v: number) => PAD.top + innerH - (v / maxY) * innerH;

  const fillFor = (v: "baseline" | "omnizip") =>
    v === "baseline"
      ? "rgba(139,92,246,0.75)"
      : "rgba(59,130,246,0.75)";
  const strokeFor = (v: "baseline" | "omnizip") =>
    v === "baseline"
      ? "rgba(139,92,246,0.45)"
      : "rgba(59,130,246,0.45)";

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line
              x1={px(t)}
              y1={PAD.top}
              x2={px(t)}
              y2={PAD.top + innerH}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
            <text
              x={px(t)}
              y={PAD.top + innerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="rgba(255,255,255,0.3)"
            >
              {fmt(t, 0)}s
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line
              x1={PAD.left}
              y1={py(t)}
              x2={PAD.left + innerW}
              y2={py(t)}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 6}
              y={py(t) + 4}
              textAnchor="end"
              fontSize={9}
              fill="rgba(255,255,255,0.3)"
            >
              {fmt(t, 0)}
            </text>
          </g>
        ))}
        {points.map((p, i) => (
          <circle
            key={`${p.variant}-${p.entry.entry}-${i}`}
            cx={px(p.entry.duration_s)}
            cy={py(p.entry.peak_vram_gb)}
            r={5}
            fill={fillFor(p.variant)}
            stroke={strokeFor(p.variant)}
            strokeWidth={1}
            className="cursor-pointer transition-colors hover:opacity-90"
            onMouseEnter={() => setHover(p)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        <line
          x1={PAD.left}
          y1={PAD.top}
          x2={PAD.left}
          y2={PAD.top + innerH}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={1}
        />
        <line
          x1={PAD.left}
          y1={PAD.top + innerH}
          x2={PAD.left + innerW}
          y2={PAD.top + innerH}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={1}
        />
        <text
          x={PAD.left + innerW / 2}
          y={H - 2}
          textAnchor="middle"
          fontSize={9}
          fill="rgba(255,255,255,0.25)"
        >
          duration (s)
        </text>
        <text
          x={10}
          y={PAD.top + innerH / 2}
          textAnchor="middle"
          fontSize={9}
          fill="rgba(255,255,255,0.25)"
          transform={`rotate(-90, 10, ${PAD.top + innerH / 2})`}
        >
          peak VRAM (GB)
        </text>
      </svg>
      {hover && (
        <div className="absolute top-4 right-4 bg-[#0e0e14] border border-white/10 rounded-lg px-3 py-2 text-xs font-mono pointer-events-none">
          <p className="text-white/50 mb-0.5">
            {hover.variant === "baseline" ? "Baseline" : "OmniZip"}
          </p>
          <p className="text-white/70 mb-1">{hover.entry.task_type}</p>
          <p className="text-violet-300">
            peak: {fmt(hover.entry.peak_vram_gb)} GB
          </p>
          <p className="text-cyan-300">
            after: {fmt(hover.entry.after_vram_gb)} GB
          </p>
          <p className="text-white/40">dur: {fmt(hover.entry.duration_s)}s</p>
        </div>
      )}
    </div>
  );
}

export interface VramChartsProps {
  baseline: VramEntry[];
  omnizip: VramEntry[];
}

export default function VramCharts({ baseline, omnizip }: VramChartsProps) {
  const merged = useMemo(() => {
    const baseMap = new Map(baseline.map((e) => [e.entry, e]));
    const zipMap = new Map(omnizip.map((e) => [e.entry, e]));
    const keys = [
      ...new Set([...baseMap.keys(), ...zipMap.keys()]),
    ].sort();
    return keys.map((k) => ({
      entryKey: k,
      task_type: baseMap.get(k)?.task_type ?? zipMap.get(k)!.task_type,
      baseline: baseMap.get(k) ?? null,
      omnizip: zipMap.get(k) ?? null,
    }));
  }, [baseline, omnizip]);

  const sortedBar = useMemo(() => {
    return [...merged].sort((a, b) => {
      const pb = Math.max(a.baseline?.peak_vram_gb ?? 0, a.omnizip?.peak_vram_gb ?? 0);
      const qb = Math.max(b.baseline?.peak_vram_gb ?? 0, b.omnizip?.peak_vram_gb ?? 0);
      return qb - pb;
    });
  }, [merged]);

  const stats = useMemo(() => {
    const avg = (arr: number[]) =>
      arr.length ? arr.reduce((x, y) => x + y, 0) / arr.length : 0;
    const basePeaks = baseline.map((d) => d.peak_vram_gb);
    const zipPeaks = omnizip.map((d) => d.peak_vram_gb);
    const baseAfters = baseline.map((d) => d.after_vram_gb);
    const zipAfters = omnizip.map((d) => d.after_vram_gb);
    const durs = baseline.map((d) => d.duration_s);
    if (!baseline.length && !omnizip.length) return null;
    return {
      baseline: {
        avgPeak: avg(basePeaks),
        maxPeak: basePeaks.length ? Math.max(...basePeaks) : 0,
        minPeak: basePeaks.length ? Math.min(...basePeaks) : 0,
        avgAfter: avg(baseAfters),
      },
      omnizip: {
        avgPeak: avg(zipPeaks),
        maxPeak: zipPeaks.length ? Math.max(...zipPeaks) : 0,
        minPeak: zipPeaks.length ? Math.min(...zipPeaks) : 0,
        avgAfter: avg(zipAfters),
      },
      avgDur: avg(durs.length ? durs : omnizip.map((d) => d.duration_s)),
    };
  }, [baseline, omnizip]);

  const barData = useMemo(
    () =>
      sortedBar.map((row) => ({
        task_type: row.task_type,
        baseline_peak: row.baseline?.peak_vram_gb ?? 0,
        omnizip_peak: row.omnizip?.peak_vram_gb ?? 0,
        baseline_after: row.baseline?.after_vram_gb ?? 0,
        omnizip_after: row.omnizip?.after_vram_gb ?? 0,
      })),
    [sortedBar]
  );

  const scatterPoints = useMemo((): ScatterPoint[] => {
    const out: ScatterPoint[] = [];
    for (const e of baseline) {
      out.push({ entry: e, variant: "baseline" });
    }
    for (const e of omnizip) {
      out.push({ entry: e, variant: "omnizip" });
    }
    return out;
  }, [baseline, omnizip]);

  if (!stats) return null;

  return (
    <div className="px-6 py-6 flex flex-col gap-8 max-w-[1100px]">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {[
          {
            label: "Baseline · Avg Peak",
            value: fmt(stats.baseline.avgPeak) + " GB",
            color: "text-violet-300",
          },
          {
            label: "OmniZip · Avg Peak",
            value: fmt(stats.omnizip.avgPeak) + " GB",
            color: "text-blue-300",
          },
          {
            label: "Baseline · Avg After",
            value: fmt(stats.baseline.avgAfter) + " GB",
            color: "text-cyan-300",
          },
          {
            label: "OmniZip · Avg After",
            value: fmt(stats.omnizip.avgAfter) + " GB",
            color: "text-sky-300",
          },
          {
            label: "Avg Duration",
            value: fmt(stats.avgDur) + " s",
            color: "text-amber-300",
          },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            className="border border-white/10 rounded-lg px-4 py-3 bg-white/[0.02]"
          >
            <p className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-1">
              {label}
            </p>
            <p className={`text-xl font-mono font-bold ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      <div className="border border-white/8 rounded-xl bg-white/[0.01] p-4">
        <div className="flex items-center gap-4 mb-4 flex-wrap">
          <p className="text-sm font-mono text-white/70">
            VRAM by task (baseline vs OmniZip)
          </p>
          <div className="flex items-center gap-3 ml-auto text-[10px] font-mono text-white/40 flex-wrap justify-end">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-violet-500/80 inline-block" />{" "}
              base peak
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-blue-500/80 inline-block" />{" "}
              zip peak
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-cyan-600/70 inline-block" />{" "}
              base after
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-sky-600/70 inline-block" />{" "}
              zip after
            </span>
          </div>
        </div>
        <BarChart
          data={barData}
          xKey="task_type"
          bars={[
            { key: "baseline_peak", color: "rgba(139,92,246,0.85)" },
            { key: "omnizip_peak", color: "rgba(59,130,246,0.85)" },
            { key: "baseline_after", color: "rgba(6,182,212,0.75)" },
            { key: "omnizip_after", color: "rgba(14,165,233,0.75)" },
          ]}
          height={300}
        />
      </div>

      <div className="border border-white/8 rounded-xl bg-white/[0.01] p-4">
        <div className="flex items-center gap-4 mb-4 flex-wrap">
          <p className="text-sm font-mono text-white/70">
            Duration vs Peak VRAM
          </p>
          <div className="flex items-center gap-3 ml-auto text-[10px] font-mono text-white/40">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-violet-500/80 inline-block" />{" "}
              baseline
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500/80 inline-block" />{" "}
              OmniZip
            </span>
          </div>
        </div>
        <ScatterPlot points={scatterPoints} />
      </div>

      <div className="border border-white/8 rounded-xl bg-white/[0.01] overflow-hidden overflow-x-auto">
        <table className="w-full text-xs font-mono min-w-[720px]">
          <thead>
            <tr className="border-b border-white/8 text-white/30 uppercase text-[10px] tracking-widest">
              <th className="px-4 py-2.5 text-left">Task Type</th>
              <th className="px-4 py-2.5 text-right text-violet-400/80">
                Base Peak
              </th>
              <th className="px-4 py-2.5 text-right text-blue-400/80">
                Zip Peak
              </th>
              <th className="px-4 py-2.5 text-right text-cyan-400/80">
                Base After
              </th>
              <th className="px-4 py-2.5 text-right text-sky-400/80">
                Zip After
              </th>
              <th className="px-4 py-2.5 text-right">Δ Peak</th>
              <th className="px-4 py-2.5 text-right">Duration</th>
            </tr>
          </thead>
          <tbody>
            {sortedBar.map((row) => {
              const bp = row.baseline?.peak_vram_gb;
              const zp = row.omnizip?.peak_vram_gb;
              const delta =
                bp !== undefined && zp !== undefined ? zp - bp : null;
              const dur =
                row.baseline?.duration_s ?? row.omnizip?.duration_s ?? 0;
              return (
                <tr
                  key={row.entryKey}
                  className="border-b border-white/5 hover:bg-white/[0.015] transition-colors"
                >
                  <td className="px-4 py-2 text-white/70">{row.task_type}</td>
                  <td className="px-4 py-2 text-right text-violet-300">
                    {row.baseline ? fmt(row.baseline.peak_vram_gb) + " GB" : "—"}
                  </td>
                  <td className="px-4 py-2 text-right text-blue-300">
                    {row.omnizip ? fmt(row.omnizip.peak_vram_gb) + " GB" : "—"}
                  </td>
                  <td className="px-4 py-2 text-right text-cyan-300">
                    {row.baseline
                      ? fmt(row.baseline.after_vram_gb) + " GB"
                      : "—"}
                  </td>
                  <td className="px-4 py-2 text-right text-sky-300">
                    {row.omnizip
                      ? fmt(row.omnizip.after_vram_gb) + " GB"
                      : "—"}
                  </td>
                  <td
                    className={`px-4 py-2 text-right font-bold ${
                      delta === null
                        ? "text-white/25"
                        : delta > 0
                          ? "text-rose-300"
                          : delta < 0
                            ? "text-emerald-300"
                            : "text-white/40"
                    }`}
                  >
                    {delta === null ? "—" : (delta > 0 ? "+" : "") + fmt(delta)}
                  </td>
                  <td className="px-4 py-2 text-right text-amber-300/80">
                    {fmt(dur)}s
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
