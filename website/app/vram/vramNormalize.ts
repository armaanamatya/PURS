export interface VramEntry {
  entry: string;
  task_type: string;
  duration_s: number;
  peak_vram_gb: number;
  after_vram_gb: number;
}

interface RawVramRecord {
  entry: string;
  task_type: string;
  duration_s: number;
  peak_vram_gb?: number;
  after_vram_gb?: number;
  peak_alloc_gb?: number;
  after_alloc_gb?: number;
}

function normalizeVramRecord(raw: RawVramRecord): VramEntry {
  const peak =
    raw.peak_vram_gb !== undefined ? raw.peak_vram_gb : raw.peak_alloc_gb;
  const after =
    raw.after_vram_gb !== undefined ? raw.after_vram_gb : raw.after_alloc_gb;
  if (peak === undefined || after === undefined) {
    throw new TypeError(
      `VRAM row missing peak/after fields: ${JSON.stringify(raw)}`
    );
  }
  return {
    entry: raw.entry,
    task_type: raw.task_type,
    duration_s: raw.duration_s,
    peak_vram_gb: peak,
    after_vram_gb: after,
  };
}

export function loadVramEntriesFromText(text: string): VramEntry[] {
  const lines = text.trim().split("\n").filter(Boolean);
  const parsed: VramEntry[] = [];
  const seen = new Set<string>();
  for (const line of lines) {
    const raw = JSON.parse(line) as RawVramRecord;
    const e = normalizeVramRecord(raw);
    if (seen.has(e.entry)) continue;
    seen.add(e.entry);
    parsed.push(e);
  }
  return parsed;
}
