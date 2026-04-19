import { existsSync, readFileSync } from "fs";
import Link from "next/link";
import { join } from "path";

interface SummaryMetric {
  n: number;
  mean: number;
  stddev: number;
  variance: number;
  min: number;
  max: number;
}

interface ConfigSummary {
  key: string;
  method: string;
  temperature: number;
  runs_requested: number;
  runs_observed: number;
  runs_successful: number;
  runs_failed: number;
  stats: Record<string, SummaryMetric>;
}

interface RunMetric {
  method: string;
  temperature: number;
  repeat: number;
  seed: number;
  gpu: string;
  return_code: number;
  ok: boolean;
  out_dir: string;
  result_path: string;
  vram_path: string;
  total_questions: number;
  correct: number;
  accuracy: number;
  prefill_ms_mean: number;
  prefill_ms_max: number;
  e2e_ms_mean: number;
  e2e_ms_max: number;
  orig_frames_mean: number;
  orig_frames_max: number;
  used_frames_mean: number;
  used_frames_max: number;
  frame_keep_ratio_mean: number;
  model_loaded_alloc_gb: number;
  model_loaded_reserved_gb: number;
  peak_alloc_gb: number;
  peak_reserved_gb: number;
  inference_extra_peak_alloc_gb: number;
  after_alloc_gb: number;
  run_wall_s: number;
  gpu_process_peak_gb: number;
  gpu_total_peak_gb: number;
}

interface PlanArgs {
  qwen_model?: string | null;
  fps?: number;
  max_pixels?: number;
  max_new_tokens?: number;
  dtype?: string | null;
  parallel?: number;
  no_audio?: boolean;
  measure_prefill?: boolean;
  mixkv_budget?: number | null;
  mixkv_window_size?: number | null;
  mixkv_select_method?: string | null;
}

interface PlanData {
  created_at: string;
  output_root: string;
  gpus: string[];
  methods: string[];
  temperatures: number[];
  runs: number;
  total_jobs: number;
  args: PlanArgs;
}

const MATRIX_ROOT = join(
  process.cwd(),
  "..",
  "10x",
  "qwen25_matrix_gpu7_all7_snapkv",
);

function readJsonFile<T>(filename: string): T {
  return JSON.parse(readFileSync(join(MATRIX_ROOT, filename), "utf8")) as T;
}

function loadRunMetrics(): RunMetric[] {
  const raw = readFileSync(join(MATRIX_ROOT, "run_metrics.jsonl"), "utf8");
  return raw
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RunMetric);
}

function configId(method: string, temperature: number) {
  return `${method}|${temperature}`;
}

function configSlug(method: string, temperature: number) {
  return `${method}-temp-${String(temperature).replace(".", "p")}`;
}

function formatPercent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatMs(value: number, digits = 0) {
  return `${value.toFixed(digits)} ms`;
}

function formatSeconds(value: number, digits = 1) {
  return `${value.toFixed(digits)} s`;
}

function formatGb(value: number, digits = 2) {
  return `${value.toFixed(digits)} GB`;
}

function formatTemperature(value: number) {
  return value.toFixed(1);
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function basename(input: string | null | undefined) {
  if (!input) {
    return "n/a";
  }
  return input.split(/[\\/]/).filter(Boolean).at(-1) ?? input;
}

function loadMatrixData() {
  const required = [
    "plan.json",
    "summary_by_config.json",
    "run_metrics.jsonl",
  ];

  if (!required.every((filename) => existsSync(join(MATRIX_ROOT, filename)))) {
    return null;
  }

  const plan = readJsonFile<PlanData>("plan.json");
  const summaryMap = readJsonFile<Record<string, Omit<ConfigSummary, "key">>>(
    "summary_by_config.json",
  );
  const configs = Object.entries(summaryMap)
    .map(([key, value]) => ({ ...value, key }))
    .sort(
      (left, right) =>
        left.method.localeCompare(right.method) ||
        left.temperature - right.temperature,
    );
  const runs = loadRunMetrics().sort(
    (left, right) =>
      left.method.localeCompare(right.method) ||
      left.temperature - right.temperature ||
      left.repeat - right.repeat,
  );

  const runsByConfig = new Map<string, RunMetric[]>();
  for (const run of runs) {
    const id = configId(run.method, run.temperature);
    const bucket = runsByConfig.get(id);
    if (bucket) {
      bucket.push(run);
    } else {
      runsByConfig.set(id, [run]);
    }
  }

  const successfulRuns = runs.filter((run) => run.ok);
  const bestAccuracyConfig =
    configs.find(
      (config) =>
        config.stats.accuracy.mean ===
        Math.max(...configs.map((entry) => entry.stats.accuracy.mean)),
    ) ?? null;
  const fastestConfig =
    configs.find(
      (config) =>
        config.stats.prefill_ms_mean.mean ===
        Math.min(...configs.map((entry) => entry.stats.prefill_ms_mean.mean)),
    ) ?? null;
  const lowestPeakConfig =
    configs.find(
      (config) =>
        config.stats.gpu_process_peak_gb.mean ===
        Math.min(...configs.map((entry) => entry.stats.gpu_process_peak_gb.mean)),
    ) ?? null;

  return {
    plan,
    configs,
    runs,
    runsByConfig,
    overview: {
      totalRuns: runs.length,
      successfulRuns: successfulRuns.length,
      failedRuns: runs.length - successfulRuns.length,
      successRate: runs.length ? successfulRuns.length / runs.length : 0,
      avgAccuracy: average(successfulRuns.map((run) => run.accuracy)),
      avgWall: average(successfulRuns.map((run) => run.run_wall_s)),
      bestAccuracyConfig,
      fastestConfig,
      lowestPeakConfig,
    },
  };
}

function MetricCard({
  label,
  value,
  tone = "text-white",
  detail,
}: {
  label: string;
  value: string;
  tone?: string;
  detail?: string;
}) {
  return (
    <div className="border border-white/10 rounded-xl bg-white/[0.02] px-4 py-3">
      <p className="text-[10px] font-mono uppercase tracking-widest text-white/30 mb-1">
        {label}
      </p>
      <p className={`text-xl font-mono font-bold ${tone}`}>{value}</p>
      {detail ? (
        <p className="text-[11px] font-mono text-white/35 mt-1">{detail}</p>
      ) : null}
    </div>
  );
}

function StatBlock({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone: string;
}) {
  return (
    <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
      <p className="text-[10px] font-mono uppercase tracking-widest text-white/25">
        {label}
      </p>
      <p className={`text-lg font-mono font-bold mt-1 ${tone}`}>{value}</p>
      <p className="text-[11px] font-mono text-white/35 mt-1">{detail}</p>
    </div>
  );
}

export default function MatrixPage() {
  const data = loadMatrixData();

  if (!data) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] text-white">
        <header className="border-b border-white/8 px-6 py-4 sticky top-0 bg-[#0a0a0f]/95 backdrop-blur-sm z-30">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                Qwen2.5-Omni Benchmark Matrix
              </h1>
              <p className="text-xs text-white/35 font-mono mt-0.5">
                Expected summary files were not found in{" "}
                <span className="text-white/50">
                  10x/qwen25_matrix_gpu7_all7_snapkv
                </span>
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs font-mono text-white/30">
              <Link href="/" className="hover:text-white/60 transition-colors">
                eval
              </Link>
              <Link
                href="/vram"
                className="hover:text-white/60 transition-colors"
              >
                vram
              </Link>
            </div>
          </div>
        </header>
      </div>
    );
  }

  const { plan, configs, runsByConfig, overview } = data;
  const createdAt = new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(plan.created_at));

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <header className="border-b border-white/8 px-6 py-4 sticky top-0 bg-[#0a0a0f]/95 backdrop-blur-sm z-30">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div>
            <div className="flex items-baseline gap-4 flex-wrap">
              <h1 className="text-lg font-semibold tracking-tight">
                Qwen2.5-Omni Benchmark Matrix
              </h1>
              <span className="text-[11px] font-mono text-white/25">
                {configs.length} configs
              </span>
            </div>
            <p className="text-xs text-white/35 font-mono mt-0.5">
              {createdAt} · {overview.totalRuns} runs ·{" "}
              {overview.successfulRuns}/{overview.totalRuns} successful · GPU{" "}
              {plan.gpus.join(", ")}
            </p>
          </div>

          <div className="flex items-center gap-4 text-[11px] font-mono text-white/30">
            <Link href="/" className="hover:text-white/60 transition-colors">
              Eval
            </Link>
            <Link href="/vram" className="hover:text-white/60 transition-colors">
              VRAM
            </Link>
          </div>
        </div>
      </header>

      <main className="px-6 py-6 flex flex-col gap-6">
        <section className="grid grid-cols-2 lg:grid-cols-6 gap-3">
          <MetricCard
            label="Success Rate"
            value={formatPercent(overview.successRate, 0)}
            tone="text-emerald-300"
            detail={`${overview.successfulRuns} ok · ${overview.failedRuns} failed`}
          />
          <MetricCard
            label="Avg Run Accuracy"
            value={formatPercent(overview.avgAccuracy)}
            tone="text-cyan-300"
            detail="Across successful runs"
          />
          <MetricCard
            label="Avg Wall Time"
            value={formatSeconds(overview.avgWall)}
            tone="text-amber-300"
            detail="Successful runs only"
          />
          <MetricCard
            label="Best Accuracy"
            value={
              overview.bestAccuracyConfig
                ? formatPercent(overview.bestAccuracyConfig.stats.accuracy.mean)
                : "n/a"
            }
            tone="text-emerald-300"
            detail={
              overview.bestAccuracyConfig
                ? `${overview.bestAccuracyConfig.method} · temp ${formatTemperature(overview.bestAccuracyConfig.temperature)}`
                : "No configs"
            }
          />
          <MetricCard
            label="Fastest Prefill"
            value={
              overview.fastestConfig
                ? formatMs(overview.fastestConfig.stats.prefill_ms_mean.mean)
                : "n/a"
            }
            tone="text-sky-300"
            detail={
              overview.fastestConfig
                ? `${overview.fastestConfig.method} · temp ${formatTemperature(overview.fastestConfig.temperature)}`
                : "No configs"
            }
          />
          <MetricCard
            label="Lowest Proc Peak"
            value={
              overview.lowestPeakConfig
                ? formatGb(overview.lowestPeakConfig.stats.gpu_process_peak_gb.mean)
                : "n/a"
            }
            tone="text-violet-300"
            detail={
              overview.lowestPeakConfig
                ? `${overview.lowestPeakConfig.method} · temp ${formatTemperature(overview.lowestPeakConfig.temperature)}`
                : "No configs"
            }
          />
        </section>

        <section className="grid xl:grid-cols-[1.2fr,2fr] gap-4">
          <div className="border border-white/10 rounded-2xl bg-white/[0.02] p-4">
            <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
              <h2 className="text-sm font-mono text-white/70">Run Setup</h2>
              <span className="text-[10px] font-mono uppercase tracking-widest text-white/25">
                {plan.total_jobs} scheduled jobs
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                <p className="text-white/25 uppercase tracking-widest text-[10px]">
                  Model
                </p>
                <p className="text-white/70 mt-1 break-words">
                  {basename(plan.args.qwen_model)}
                </p>
              </div>
              <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                <p className="text-white/25 uppercase tracking-widest text-[10px]">
                  Output Root
                </p>
                <p className="text-white/70 mt-1 break-words">
                  {plan.output_root}
                </p>
              </div>
              <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                <p className="text-white/25 uppercase tracking-widest text-[10px]">
                  Sampling
                </p>
                <p className="text-white/70 mt-1">
                  FPS {plan.args.fps ?? "n/a"} · max_pixels {plan.args.max_pixels ?? "n/a"}
                </p>
              </div>
              <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                <p className="text-white/25 uppercase tracking-widest text-[10px]">
                  Decode
                </p>
                <p className="text-white/70 mt-1">
                  {plan.args.dtype ?? "n/a"} · max_new_tokens{" "}
                  {plan.args.max_new_tokens ?? "n/a"}
                </p>
              </div>
              <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                <p className="text-white/25 uppercase tracking-widest text-[10px]">
                  MixKV
                </p>
                <p className="text-white/70 mt-1">
                  {plan.args.mixkv_select_method ?? "n/a"} · budget{" "}
                  {plan.args.mixkv_budget ?? "n/a"} · window{" "}
                  {plan.args.mixkv_window_size ?? "n/a"}
                </p>
              </div>
              <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                <p className="text-white/25 uppercase tracking-widest text-[10px]">
                  Harness
                </p>
                <p className="text-white/70 mt-1">
                  parallel {plan.args.parallel ?? "n/a"} · prefill{" "}
                  {plan.args.measure_prefill ? "on" : "off"} · audio{" "}
                  {plan.args.no_audio ? "off" : "on"}
                </p>
              </div>
            </div>
          </div>

          <div className="border border-white/10 rounded-2xl bg-white/[0.02] p-4">
            <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
              <h2 className="text-sm font-mono text-white/70">Configs</h2>
              <div className="flex flex-wrap gap-2 text-[10px] font-mono">
                {configs.map((config) => (
                  <a
                    key={config.key}
                    href={`#${configSlug(config.method, config.temperature)}`}
                    className="px-2.5 py-1 rounded-full border border-white/10 bg-black/20 text-white/45 hover:text-white/75 hover:border-white/20 transition-colors"
                  >
                    {config.method} · t={formatTemperature(config.temperature)}
                  </a>
                ))}
              </div>
            </div>

            <div className="grid lg:grid-cols-2 gap-4">
              {configs.map((config) => {
                const accuracy = config.stats.accuracy;
                const prefill = config.stats.prefill_ms_mean;
                const e2e = config.stats.e2e_ms_mean;
                const peakAlloc = config.stats.peak_alloc_gb;
                const processPeak = config.stats.gpu_process_peak_gb;
                const keepRatio = config.stats.frame_keep_ratio_mean;

                return (
                  <div
                    key={config.key}
                    className="rounded-xl border border-white/8 bg-black/20 p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-[10px] font-mono uppercase tracking-widest text-white/25">
                          {config.method}
                        </p>
                        <h3 className="text-lg font-semibold mt-1">
                          temp {formatTemperature(config.temperature)}
                        </h3>
                      </div>
                      <span className="text-[11px] font-mono text-white/35">
                        {config.runs_successful}/{config.runs_requested} ok
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 mt-4">
                      <StatBlock
                        label="Accuracy"
                        value={formatPercent(accuracy.mean)}
                        detail={`std ${formatPercent(accuracy.stddev)} · range ${formatPercent(accuracy.min)}-${formatPercent(accuracy.max)}`}
                        tone="text-emerald-300"
                      />
                      <StatBlock
                        label="Prefill Mean"
                        value={formatMs(prefill.mean)}
                        detail={`range ${formatMs(prefill.min)}-${formatMs(prefill.max)}`}
                        tone="text-sky-300"
                      />
                      <StatBlock
                        label="E2E Mean"
                        value={formatMs(e2e.mean)}
                        detail={`range ${formatMs(e2e.min)}-${formatMs(e2e.max)}`}
                        tone="text-cyan-300"
                      />
                      <StatBlock
                        label="Proc Peak"
                        value={formatGb(processPeak.mean)}
                        detail={`alloc peak ${formatGb(peakAlloc.mean)}`}
                        tone="text-violet-300"
                      />
                      <StatBlock
                        label="Keep Ratio"
                        value={formatPercent(keepRatio.mean, 0)}
                        detail={`${config.stats.used_frames_mean.mean.toFixed(0)} used / ${config.stats.orig_frames_mean.mean.toFixed(0)} orig`}
                        tone="text-amber-300"
                      />
                      <StatBlock
                        label="Run Wall"
                        value={formatSeconds(config.stats.run_wall_s.mean)}
                        detail={`range ${formatSeconds(config.stats.run_wall_s.min)}-${formatSeconds(config.stats.run_wall_s.max)}`}
                        tone="text-rose-300"
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {configs.map((config) => {
          const runs = runsByConfig.get(configId(config.method, config.temperature)) ?? [];

          return (
            <section
              key={config.key}
              id={configSlug(config.method, config.temperature)}
              className="border border-white/10 rounded-2xl bg-white/[0.02] overflow-hidden scroll-mt-20"
            >
              <div className="px-4 py-3 border-b border-white/8 flex items-center justify-between gap-4 flex-wrap">
                <div>
                  <p className="text-[10px] font-mono uppercase tracking-widest text-white/25">
                    Run Details
                  </p>
                  <h2 className="text-base font-semibold mt-1">
                    {config.method} · temp {formatTemperature(config.temperature)}
                  </h2>
                </div>
                <div className="flex flex-wrap gap-2 text-[10px] font-mono text-white/35">
                  <span className="px-2.5 py-1 rounded-full border border-white/10 bg-black/20">
                    {runs.length} runs
                  </span>
                  <span className="px-2.5 py-1 rounded-full border border-white/10 bg-black/20">
                    {config.runs_successful} successful
                  </span>
                  <span className="px-2.5 py-1 rounded-full border border-white/10 bg-black/20">
                    {config.stats.accuracy.n} summarized
                  </span>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[1280px] text-xs font-mono">
                  <thead>
                    <tr className="border-b border-white/8 text-white/30 uppercase tracking-widest text-[10px]">
                      <th className="px-4 py-2.5 text-left">Run</th>
                      <th className="px-4 py-2.5 text-left">GPU</th>
                      <th className="px-4 py-2.5 text-left">Status</th>
                      <th className="px-4 py-2.5 text-right">Accuracy</th>
                      <th className="px-4 py-2.5 text-right">Score</th>
                      <th className="px-4 py-2.5 text-right">Prefill Mean</th>
                      <th className="px-4 py-2.5 text-right">Prefill Max</th>
                      <th className="px-4 py-2.5 text-right">E2E Mean</th>
                      <th className="px-4 py-2.5 text-right">E2E Max</th>
                      <th className="px-4 py-2.5 text-right">Frames</th>
                      <th className="px-4 py-2.5 text-right">Peak Alloc</th>
                      <th className="px-4 py-2.5 text-right">Proc Peak</th>
                      <th className="px-4 py-2.5 text-right">Wall</th>
                      <th className="px-4 py-2.5 text-left">Out Dir</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr
                        key={`${run.method}-${run.temperature}-${run.repeat}`}
                        className="border-b border-white/5 hover:bg-white/[0.015] transition-colors"
                      >
                        <td className="px-4 py-2 text-white/70">
                          r{String(run.repeat).padStart(2, "0")}
                          <span className="text-white/25"> · seed {run.seed}</span>
                        </td>
                        <td className="px-4 py-2 text-white/55">{run.gpu}</td>
                        <td className="px-4 py-2">
                          <span
                            className={
                              run.ok
                                ? "text-emerald-300"
                                : "text-red-300"
                            }
                          >
                            {run.ok ? "ok" : `rc=${run.return_code}`}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-right text-emerald-300">
                          {formatPercent(run.accuracy)}
                        </td>
                        <td className="px-4 py-2 text-right text-white/55">
                          {run.correct}/{run.total_questions}
                        </td>
                        <td className="px-4 py-2 text-right text-sky-300">
                          {formatMs(run.prefill_ms_mean)}
                        </td>
                        <td className="px-4 py-2 text-right text-sky-300/70">
                          {formatMs(run.prefill_ms_max)}
                        </td>
                        <td className="px-4 py-2 text-right text-cyan-300">
                          {formatMs(run.e2e_ms_mean)}
                        </td>
                        <td className="px-4 py-2 text-right text-cyan-300/70">
                          {formatMs(run.e2e_ms_max)}
                        </td>
                        <td className="px-4 py-2 text-right text-amber-300/80">
                          {run.used_frames_mean.toFixed(0)}
                        </td>
                        <td className="px-4 py-2 text-right text-violet-300">
                          {formatGb(run.peak_alloc_gb)}
                        </td>
                        <td className="px-4 py-2 text-right text-fuchsia-300/80">
                          {formatGb(run.gpu_process_peak_gb)}
                        </td>
                        <td className="px-4 py-2 text-right text-rose-300/80">
                          {formatSeconds(run.run_wall_s)}
                        </td>
                        <td className="px-4 py-2 text-white/35 max-w-[420px] truncate">
                          {run.out_dir}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          );
        })}
      </main>
    </div>
  );
}
