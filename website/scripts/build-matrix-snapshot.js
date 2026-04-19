/* eslint-disable @typescript-eslint/no-require-imports */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const RUN_DIR_NAME = "qwen25_matrix_gpu7_all7_snapkv";
const SOURCE_DIR = path.join(ROOT, "10x", RUN_DIR_NAME);
const OUT_DIR = path.join(
  __dirname,
  "..",
  "public",
  "matrix",
  RUN_DIR_NAME,
);

const SUMMARY_FIELDS = [
  "accuracy",
  "prefill_ms_mean",
  "prefill_ms_max",
  "e2e_ms_mean",
  "e2e_ms_max",
  "orig_frames_mean",
  "orig_frames_max",
  "used_frames_mean",
  "used_frames_max",
  "frame_keep_ratio_mean",
  "model_loaded_alloc_gb",
  "model_loaded_reserved_gb",
  "peak_alloc_gb",
  "peak_reserved_gb",
  "inference_extra_peak_alloc_gb",
  "after_alloc_gb",
  "gpu_process_peak_gb",
  "gpu_total_peak_gb",
  "run_wall_s",
];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function listDirectories(dirPath) {
  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
}

function stat(values) {
  const n = values.length;
  const mean = n ? values.reduce((sum, value) => sum + value, 0) / n : 0;
  const variance = n
    ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n
    : 0;
  return {
    n,
    mean,
    stddev: Math.sqrt(variance),
    variance,
    min: n ? Math.min(...values) : 0,
    max: n ? Math.max(...values) : 0,
  };
}

function tempSlug(temperature) {
  return String(temperature).replace(".", "p");
}

function loadRuns() {
  const runs = [];
  for (const methodDir of listDirectories(SOURCE_DIR)) {
    if (methodDir.includes("partial_interrupted")) {
      continue;
    }

    const methodPath = path.join(SOURCE_DIR, methodDir);
    for (const runDir of listDirectories(methodPath)) {
      const summaryPath = path.join(methodPath, runDir, "run_summary.json");
      if (!fs.existsSync(summaryPath)) {
        continue;
      }

      const run = readJson(summaryPath);
      runs.push({
        ...run,
        method: methodDir,
      });
    }
  }

  return runs.sort(
    (left, right) =>
      left.method.localeCompare(right.method) ||
      left.temperature - right.temperature ||
      left.repeat - right.repeat,
  );
}

function buildSummary(runs) {
  const requestedByConfig = new Map();
  for (const methodDir of listDirectories(SOURCE_DIR)) {
    if (methodDir.includes("partial_interrupted")) {
      continue;
    }

    const methodPath = path.join(SOURCE_DIR, methodDir);
    for (const runDir of listDirectories(methodPath)) {
      const match = runDir.match(/^temp_(.+)-run_/);
      if (!match) {
        continue;
      }
      const temperature = Number(match[1].replace("p", "."));
      const key = `${methodDir}/temp_${tempSlug(temperature)}`;
      requestedByConfig.set(key, (requestedByConfig.get(key) ?? 0) + 1);
    }
  }

  const runsByConfig = new Map();
  for (const run of runs) {
    const key = `${run.method}/temp_${tempSlug(run.temperature)}`;
    const bucket = runsByConfig.get(key);
    if (bucket) {
      bucket.push(run);
    } else {
      runsByConfig.set(key, [run]);
    }
  }

  const summary = {};
  for (const [key, configRuns] of [...runsByConfig.entries()].sort()) {
    const [method] = key.split("/");
    const temperature = configRuns[0].temperature;
    const successfulRuns = configRuns.filter((run) => run.ok);
    const stats = {};

    for (const field of SUMMARY_FIELDS) {
      stats[field] = stat(
        successfulRuns
          .map((run) => run[field])
          .filter((value) => typeof value === "number" && Number.isFinite(value)),
      );
    }

    summary[key] = {
      method,
      temperature,
      runs_requested: requestedByConfig.get(key) ?? configRuns.length,
      runs_observed: configRuns.length,
      runs_successful: successfulRuns.length,
      runs_failed: configRuns.length - successfulRuns.length,
      stats,
    };
  }

  return summary;
}

function writeSnapshot() {
  if (!fs.existsSync(SOURCE_DIR)) {
    throw new Error(`Missing source run directory: ${SOURCE_DIR}`);
  }

  const runs = loadRuns();
  const summary = buildSummary(runs);
  const plan = readJson(path.join(SOURCE_DIR, "plan.json"));
  const methods = [...new Set(runs.map((run) => run.method))].sort();
  const temperatures = [...new Set(runs.map((run) => run.temperature))].sort(
    (left, right) => left - right,
  );

  const planSnapshot = {
    ...plan,
    methods,
    temperatures,
    total_jobs: runs.length,
    runs:
      runs.length && methods.length && temperatures.length
        ? runs.length / methods.length / temperatures.length
        : plan.runs,
  };

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(
    path.join(OUT_DIR, "plan.json"),
    `${JSON.stringify(planSnapshot, null, 2)}\n`,
  );
  fs.writeFileSync(
    path.join(OUT_DIR, "summary_by_config.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
  );
  fs.writeFileSync(
    path.join(OUT_DIR, "run_metrics.jsonl"),
    `${runs.map((run) => JSON.stringify(run)).join("\n")}\n`,
  );

  console.log(
    `Wrote ${runs.length} runs across ${Object.keys(summary).length} configs to ${OUT_DIR}`,
  );
}

writeSnapshot();
