import { readFileSync } from "fs";
import { join } from "path";
import VramCharts, { VramEntry } from "./VramCharts";

function loadEntries(): VramEntry[] {
  const filePath = join(process.cwd(), "public", "vram_log.jsonl");
  const text = readFileSync(filePath, "utf-8");
  const parsed: VramEntry[] = text
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
  // Deduplicate by entry key
  const seen = new Set<string>();
  return parsed.filter((e) => {
    if (seen.has(e.entry)) return false;
    seen.add(e.entry);
    return true;
  });
}

export default function VramPage() {
  const entries = loadEntries();

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <header className="border-b border-white/8 px-6 py-4 sticky top-0 bg-[#0a0a0f]/95 backdrop-blur-sm z-30">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">VRAM Usage</h1>
            <p className="text-xs text-white/35 font-mono mt-0.5">
              {entries.length} task types · peak / after / duration
            </p>
          </div>
          <a href="/" className="text-xs font-mono text-white/30 hover:text-white/60 transition-colors">
            ← eval results
          </a>
        </div>
      </header>
      <VramCharts entries={entries} />
    </div>
  );
}
