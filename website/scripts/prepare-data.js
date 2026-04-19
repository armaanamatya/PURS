/* eslint-disable @typescript-eslint/no-require-imports */
// scripts/prepare-data.js
// Merges ../videos/metadata.json + ../results.jsonl into public/data.json
// Also copies videos into public/videos/ (Windows-safe, no symlinks)
//
// Flags:
//   --results <path>   Override results file (default: ../../results.jsonl)
//   --output  <path>   Override output file  (default: public/data.json)
//   --no-videos        Skip copying videos

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const META_FILE = path.join(ROOT, "videos", "metadata.json");
const PUBLIC_VIDEOS = path.join(__dirname, "../public/videos");

// Parse CLI flags
const args = process.argv.slice(2);
function getArg(flag) {
  const i = args.indexOf(flag);
  return i !== -1 ? args[i + 1] : null;
}
const noVideos = args.includes("--no-videos");

const RESULTS_FILE = getArg("--results")
  ? path.resolve(getArg("--results"))
  : path.join(ROOT, "results.jsonl");
const OUT_FILE = getArg("--output")
  ? path.resolve(__dirname, "..", getArg("--output"))
  : path.join(__dirname, "../public/data.json");

// ── Load results.jsonl ────────────────────────────────────────────────────────
function loadResults() {
  if (!fs.existsSync(RESULTS_FILE)) {
    console.warn("WARNING: results.jsonl not found — no predictions will be shown");
    return [];
  }
  return fs
    .readFileSync(RESULTS_FILE, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

// ── Normalize video path → URL slug ──────────────────────────────────────────
function toVideoUrl(filePath) {
  // "videos\video-mme\action_reasoning\video.mp4" → "/videos/video-mme/action_reasoning/video.mp4"
  const normalized = filePath.replace(/\\/g, "/");
  const idx = normalized.indexOf("videos/");
  if (idx === -1) return "/" + normalized;
  return "/" + normalized.slice(idx);
}

// ── Copy videos to public/videos ─────────────────────────────────────────────
function copyVideosDir(srcDir, destDir) {
  if (!fs.existsSync(srcDir)) {
    console.warn(`Videos source not found: ${srcDir}`);
    return;
  }
  fs.mkdirSync(destDir, { recursive: true });
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const src = path.join(srcDir, entry.name);
    const dest = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      copyVideosDir(src, dest);
    } else if (!fs.existsSync(dest)) {
      fs.copyFileSync(src, dest);
      process.stdout.write(".");
    }
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
function main() {
  const metadata = JSON.parse(fs.readFileSync(META_FILE, "utf8"));
  const results = loadResults();

  // Positional matching: results.jsonl rows are in the same order as
  // flattened metadata questions. Key-based matching fails when worldsense
  // has duplicate dataset||task_type||question keys across different videos.
  let resultIdx = 0;

  // Merge
  const entries = metadata.map((entry) => {
    const videoUrl = toVideoUrl(entry.file);
    const enrichedQuestions = (entry.questions || []).map((q) => {
      const result = resultIdx < results.length ? results[resultIdx] : null;
      resultIdx++;
      return {
        question: q.question,
        choices: q.choices,
        answer: q.answer,
        task_type: q.task_type || entry.task_type,
        prediction: result?.prediction ?? null,
        correct: result?.correct ?? null,
        reasoning: result?.reasoning ?? null,
      };
    });

    return {
      dataset: entry.dataset,
      task_type: entry.task_type,
      duration_s: entry.duration_s,
      video_url: videoUrl,
      questions: enrichedQuestions,
    };
  });

  // Summary stats
  const allQs = entries.flatMap((e) => e.questions).filter((q) => q.prediction !== null);
  const totalCorrect = allQs.filter((q) => q.correct).length;
  const byDataset = {};
  for (const entry of entries) {
    for (const q of entry.questions) {
      if (q.prediction === null) continue;
      if (!byDataset[entry.dataset]) byDataset[entry.dataset] = { correct: 0, total: 0 };
      byDataset[entry.dataset].total++;
      if (q.correct) byDataset[entry.dataset].correct++;
    }
  }

  const output = {
    generated_at: new Date().toISOString(),
    stats: {
      total_correct: totalCorrect,
      total_questions: allQs.length,
      accuracy: allQs.length ? totalCorrect / allQs.length : 0,
      by_dataset: byDataset,
    },
    entries,
  };

  fs.writeFileSync(OUT_FILE, JSON.stringify(output, null, 2));
  console.log(`\nWrote ${OUT_FILE}`);

  // Copy videos
  if (!noVideos) {
    console.log("Copying videos to public/videos/ (skipping existing)...");
    copyVideosDir(path.join(ROOT, "videos"), PUBLIC_VIDEOS);
  }
  console.log("\nDone.");
}

main();
