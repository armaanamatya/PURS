import { existsSync, readFileSync } from "fs";
import Link from "next/link";
import { dirname, join } from "path";

export const dynamic = "force-dynamic";

const RUN_DIR_NAME = "qwen25_matrix_gpu7_all7_snapkv";
const RUN_ROOT_CANDIDATES = [
  join("public", "matrix", RUN_DIR_NAME),
  join("website", "public", "matrix", RUN_DIR_NAME),
  join("10x", RUN_DIR_NAME),
];

function hasRunFiles(dir: string) {
  return [
    "plan.json",
    "summary_by_config.json",
    "run_metrics.jsonl",
  ].every((filename) => existsSync(join(dir, filename)));
}

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
  total_questions: number;
  correct: number;
  accuracy: number;
  prefill_ms_mean: number;
  e2e_ms_mean: number;
  orig_frames_mean: number;
  used_frames_mean: number;
  frame_keep_ratio_mean: number;
  model_loaded_alloc_gb: number;
  peak_alloc_gb: number;
  peak_reserved_gb: number;
  inference_extra_peak_alloc_gb: number;
  after_alloc_gb: number;
  run_wall_s: number;
  gpu_process_peak_gb: number;
  gpu_total_peak_gb: number;
  out_dir: string;
}

interface PlanData {
  created_at: string;
  output_root: string;
  gpus: string[];
  total_jobs: number;
}

const EMPTY_METRIC: SummaryMetric = {
  n: 0,
  mean: 0,
  stddev: 0,
  variance: 0,
  min: 0,
  max: 0,
};

function findRunRoot() {
  const starts = [
    process.cwd(),
    join(process.cwd(), ".."),
    join(process.cwd(), "..", ".."),
  ];
  const seen = new Set<string>();

  for (const start of starts) {
    let current = start;
    while (!seen.has(current)) {
      seen.add(current);
      for (const relativePath of RUN_ROOT_CANDIDATES) {
        const candidate = join(current, relativePath);
        if (hasRunFiles(candidate)) {
          return candidate;
        }
      }

      const parent = dirname(current);
      if (parent === current) {
        break;
      }
      current = parent;
    }
  }

  return join(process.cwd(), "public", "matrix", RUN_DIR_NAME);
}

const RUN_ROOT = findRunRoot();

function readJsonFile<T>(filename: string): T {
  return JSON.parse(readFileSync(join(RUN_ROOT, filename), "utf8")) as T;
}

function loadRunMetrics(): RunMetric[] {
  return readFileSync(join(RUN_ROOT, "run_metrics.jsonl"), "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RunMetric)
    .sort(
      (left, right) =>
        left.method.localeCompare(right.method) ||
        left.temperature - right.temperature ||
        left.repeat - right.repeat,
    );
}

function loadVramData() {
  const required = [
    "plan.json",
    "summary_by_config.json",
    "run_metrics.jsonl",
  ];

  if (!required.every((filename) => existsSync(join(RUN_ROOT, filename)))) {
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
  const runs = loadRunMetrics();
  const successfulRuns = runs.filter((run) => run.ok);

  return {
    plan,
    configs,
    runs,
    successfulRuns,
  };
}

function metric(config: ConfigSummary, key: string) {
  return config.stats[key] ?? EMPTY_METRIC;
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatDate(input: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(input));
}

function formatGb(value: number, digits = 2) {
  return `${value.toFixed(digits)} GB`;
}

function formatMs(value: number, digits = 0) {
  return `${value.toFixed(digits)} ms`;
}

function formatPercent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSeconds(value: number, digits = 1) {
  return `${value.toFixed(digits)} s`;
}

function formatTemperature(value: number) {
  return value.toFixed(1);
}

function configLabel(method: string, temperature: number) {
  return `${method} t=${formatTemperature(temperature)}`;
}

function MetricCard({
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
    <div className="border border-white/10 rounded-xl bg-white/[0.02] px-4 py-3">
      <p className="text-[10px] font-mono uppercase tracking-widest text-white/30 mb-1">
        {label}
      </p>
      <p className={`text-xl font-mono font-bold ${tone}`}>{value}</p>
      <p className="text-[11px] font-mono text-white/35 mt-1">{detail}</p>
    </div>
  );
}

function BarRow({
  label,
  value,
  max,
  tone,
  barClass,
}: {
  label: string;
  value: number;
  max: number;
  tone: string;
  barClass: string;
}) {
  const width = max > 0 ? Math.max((value / max) * 100, 2) : 0;

  return (
    <div>
      <div className="flex items-center justify-between gap-3 mb-1">
        <span className="text-[10px] font-mono uppercase tracking-widest text-white/25">
          {label}
        </span>
        <span className={`text-xs font-mono font-bold ${tone}`}>
          {formatGb(value)}
        </span>
      </div>
      <div className="h-2 rounded-full bg-white/8 overflow-hidden">
        <div
          className={`h-full rounded-full ${barClass}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <header className="border-b border-white/8 px-6 py-4 sticky top-0 bg-[#0a0a0f]/95 backdrop-blur-sm z-30">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              VRAM Usage
            </h1>
            <p className="text-xs text-white/35 font-mono mt-0.5">
              Could not find the new run metrics under 10x/{RUN_DIR_NAME}.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-white/30">
            <Link href="/matrix" className="hover:text-white/60 transition-colors">
              matrix
            </Link>
            <Link href="/" className="hover:text-white/60 transition-colors">
              {"<-"} eval results
            </Link>
          </div>
        </div>
      </header>
    </div>
  );
}

export default function VramPage() {
  const data = loadVramData();

  if (!data) {
    return <EmptyState />;
  }

  const { plan, configs, runs, successfulRuns } = data;
  const avgProcessPeak = average(
    successfulRuns.map((run) => run.gpu_process_peak_gb),
  );
  const avgAllocPeak = average(successfulRuns.map((run) => run.peak_alloc_gb));
  const avgExtraPeak = average(
    successfulRuns.map((run) => run.inference_extra_peak_alloc_gb),
  );
  const highestProcessRun =
    successfulRuns.reduce<RunMetric | null>(
      (best, run) =>
        !best || run.gpu_process_peak_gb > best.gpu_process_peak_gb
          ? run
          : best,
      null,
    );
  const maxMemory = Math.max(
    1,
    ...configs.flatMap((config) => [
      metric(config, "gpu_process_peak_gb").mean,
      metric(config, "peak_reserved_gb").mean,
      metric(config, "peak_alloc_gb").mean,
      metric(config, "model_loaded_alloc_gb").mean,
      metric(config, "inference_extra_peak_alloc_gb").mean,
    ]),
  );

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <header className="border-b border-white/8 px-6 py-4 sticky top-0 bg-[#0a0a0f]/95 backdrop-blur-sm z-30">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              VRAM Usage - 10x Matrix Runs
            </h1>
            <p className="text-xs text-white/35 font-mono mt-0.5">
              {formatDate(plan.created_at)} | {runs.length} runs |{" "}
              {successfulRuns.length}/{runs.length} successful | GPU{" "}
              {plan.gpus.join(", ")}
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-white/30">
            <Link href="/matrix" className="hover:text-white/60 transition-colors">
              matrix
            </Link>
            <Link href="/" className="hover:text-white/60 transition-colors">
              {"<-"} eval results
            </Link>
          </div>
        </div>
      </header>

      <main className="px-6 py-6 flex flex-col gap-6">
        <section className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          <MetricCard
            label="Avg Process Peak"
            value={formatGb(avgProcessPeak)}
            detail="nvidia-smi process peak"
            tone="text-fuchsia-300"
          />
          <MetricCard
            label="Avg Alloc Peak"
            value={formatGb(avgAllocPeak)}
            detail="torch peak allocated"
            tone="text-violet-300"
          />
          <MetricCard
            label="Avg Inference Extra"
            value={formatGb(avgExtraPeak)}
            detail="peak alloc minus before alloc"
            tone="text-sky-300"
          />
          <MetricCard
            label="Highest Process Peak"
            value={
              highestProcessRun
                ? formatGb(highestProcessRun.gpu_process_peak_gb)
                : "n/a"
            }
            detail={
              highestProcessRun
                ? `${configLabel(highestProcessRun.method, highestProcessRun.temperature)} r${highestProcessRun.repeat}`
                : "No successful runs"
            }
            tone="text-rose-300"
          />
          <MetricCard
            label="Configs"
            value={String(configs.length)}
            detail={`source: 10x/${RUN_DIR_NAME}`}
            tone="text-emerald-300"
          />
        </section>

        <section className="grid xl:grid-cols-2 gap-4">
          {configs.map((config) => {
            const processPeak = metric(config, "gpu_process_peak_gb");
            const reservedPeak = metric(config, "peak_reserved_gb");
            const allocPeak = metric(config, "peak_alloc_gb");
            const loadedAlloc = metric(config, "model_loaded_alloc_gb");
            const extraPeak = metric(config, "inference_extra_peak_alloc_gb");
            const runWall = metric(config, "run_wall_s");
            const prefill = metric(config, "prefill_ms_mean");

            return (
              <div
                key={config.key}
                className="border border-white/10 rounded-2xl bg-white/[0.02] p-4"
              >
                <div className="flex items-start justify-between gap-4 mb-5">
                  <div>
                    <p className="text-[10px] font-mono uppercase tracking-widest text-white/25">
                      {config.method}
                    </p>
                    <h2 className="text-lg font-semibold mt-1">
                      temp {formatTemperature(config.temperature)}
                    </h2>
                  </div>
                  <div className="text-right">
                    <p className="text-[11px] font-mono text-white/35">
                      {config.runs_successful}/{config.runs_requested} ok
                    </p>
                    <p className="text-[11px] font-mono text-white/25 mt-1">
                      {formatMs(prefill.mean)} prefill |{" "}
                      {formatSeconds(runWall.mean)} wall
                    </p>
                  </div>
                </div>

                <div className="flex flex-col gap-4">
                  <BarRow
                    label="Process Peak"
                    value={processPeak.mean}
                    max={maxMemory}
                    tone="text-fuchsia-300"
                    barClass="bg-fuchsia-500/80"
                  />
                  <BarRow
                    label="Reserved Peak"
                    value={reservedPeak.mean}
                    max={maxMemory}
                    tone="text-purple-300"
                    barClass="bg-purple-500/75"
                  />
                  <BarRow
                    label="Allocated Peak"
                    value={allocPeak.mean}
                    max={maxMemory}
                    tone="text-violet-300"
                    barClass="bg-violet-500/75"
                  />
                  <BarRow
                    label="Loaded Alloc"
                    value={loadedAlloc.mean}
                    max={maxMemory}
                    tone="text-cyan-300"
                    barClass="bg-cyan-500/75"
                  />
                  <BarRow
                    label="Inference Extra"
                    value={extraPeak.mean}
                    max={maxMemory}
                    tone="text-sky-300"
                    barClass="bg-sky-500/75"
                  />
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5 text-xs font-mono">
                  <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                    <p className="text-white/25 uppercase tracking-widest text-[10px]">
                      Proc Range
                    </p>
                    <p className="text-white/65 mt-1">
                      {formatGb(processPeak.min)}-{formatGb(processPeak.max)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                    <p className="text-white/25 uppercase tracking-widest text-[10px]">
                      Alloc Range
                    </p>
                    <p className="text-white/65 mt-1">
                      {formatGb(allocPeak.min)}-{formatGb(allocPeak.max)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                    <p className="text-white/25 uppercase tracking-widest text-[10px]">
                      Frames
                    </p>
                    <p className="text-white/65 mt-1">
                      {metric(config, "used_frames_mean").mean.toFixed(0)} used
                    </p>
                  </div>
                  <div className="rounded-lg border border-white/8 bg-black/20 px-3 py-2">
                    <p className="text-white/25 uppercase tracking-widest text-[10px]">
                      Keep Ratio
                    </p>
                    <p className="text-white/65 mt-1">
                      {formatPercent(metric(config, "frame_keep_ratio_mean").mean, 0)}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </section>

        <section className="border border-white/10 rounded-2xl bg-white/[0.02] overflow-hidden">
          <div className="px-4 py-3 border-b border-white/8 flex items-center justify-between gap-4 flex-wrap">
            <div>
              <p className="text-[10px] font-mono uppercase tracking-widest text-white/25">
                Per Run
              </p>
              <h2 className="text-base font-semibold mt-1">
                VRAM and runtime metrics
              </h2>
            </div>
            <p className="text-[11px] font-mono text-white/35">
              Process peak includes CUDA context, allocator reservation, and
              non-PyTorch allocations.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[1180px] text-xs font-mono">
              <thead>
                <tr className="border-b border-white/8 text-white/30 uppercase tracking-widest text-[10px]">
                  <th className="px-4 py-2.5 text-left">Config</th>
                  <th className="px-4 py-2.5 text-left">Run</th>
                  <th className="px-4 py-2.5 text-left">Status</th>
                  <th className="px-4 py-2.5 text-right">Process Peak</th>
                  <th className="px-4 py-2.5 text-right">Total Peak</th>
                  <th className="px-4 py-2.5 text-right">Alloc Peak</th>
                  <th className="px-4 py-2.5 text-right">Reserved Peak</th>
                  <th className="px-4 py-2.5 text-right">Inference Extra</th>
                  <th className="px-4 py-2.5 text-right">After Alloc</th>
                  <th className="px-4 py-2.5 text-right">Frames</th>
                  <th className="px-4 py-2.5 text-right">Wall</th>
                  <th className="px-4 py-2.5 text-right">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr
                    key={`${run.method}-${run.temperature}-${run.repeat}`}
                    className="border-b border-white/5 hover:bg-white/[0.015] transition-colors"
                  >
                    <td className="px-4 py-2 text-white/70">
                      {configLabel(run.method, run.temperature)}
                    </td>
                    <td className="px-4 py-2 text-white/45">
                      r{String(run.repeat).padStart(2, "0")} | seed {run.seed}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className={run.ok ? "text-emerald-300" : "text-red-300"}
                      >
                        {run.ok ? "ok" : `rc=${run.return_code}`}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right text-fuchsia-300">
                      {formatGb(run.gpu_process_peak_gb)}
                    </td>
                    <td className="px-4 py-2 text-right text-pink-300/80">
                      {formatGb(run.gpu_total_peak_gb)}
                    </td>
                    <td className="px-4 py-2 text-right text-violet-300">
                      {formatGb(run.peak_alloc_gb)}
                    </td>
                    <td className="px-4 py-2 text-right text-purple-300">
                      {formatGb(run.peak_reserved_gb)}
                    </td>
                    <td className="px-4 py-2 text-right text-sky-300">
                      {formatGb(run.inference_extra_peak_alloc_gb)}
                    </td>
                    <td className="px-4 py-2 text-right text-cyan-300/80">
                      {formatGb(run.after_alloc_gb)}
                    </td>
                    <td className="px-4 py-2 text-right text-amber-300/80">
                      {run.used_frames_mean.toFixed(0)}/
                      {run.orig_frames_mean.toFixed(0)}
                    </td>
                    <td className="px-4 py-2 text-right text-rose-300/80">
                      {formatSeconds(run.run_wall_s)}
                    </td>
                    <td className="px-4 py-2 text-right text-emerald-300">
                      {formatPercent(run.accuracy)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
