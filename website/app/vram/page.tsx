import { readFileSync } from "fs";
import Link from "next/link";
import { join } from "path";
import VramCharts from "./VramCharts";
import { loadVramEntriesFromText } from "./vramNormalize";

function loadVramPair() {
  const basePath = join(process.cwd(), "public", "vram_baseline.jsonl");
  const zipPath = join(process.cwd(), "public", "vram_omnizip.jsonl");
  const baselineText = readFileSync(basePath, "utf-8");
  const omnizipText = readFileSync(zipPath, "utf-8");
  return {
    baseline: loadVramEntriesFromText(baselineText),
    omnizip: loadVramEntriesFromText(omnizipText),
  };
}

export default function VramPage() {
  const { baseline, omnizip } = loadVramPair();
  const n = Math.max(baseline.length, omnizip.length);

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <header className="border-b border-white/8 px-6 py-4 sticky top-0 bg-[#0a0a0f]/95 backdrop-blur-sm z-30">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">VRAM Usage</h1>
            <p className="text-xs text-white/35 font-mono mt-0.5">
              {n} task types | baseline vs OmniZip (run2) | peak / after /
              duration
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
      <VramCharts baseline={baseline} omnizip={omnizip} />
    </div>
  );
}
