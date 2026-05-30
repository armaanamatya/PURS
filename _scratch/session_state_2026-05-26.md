# Session-state synthesis (looked back to 2026-05-15)

Sources: 11 jsonl session transcripts in `~/.claude/projects/C--Users-Armaan-Desktop-PURS/` since 5/15; `docs/PROJECT_BRIEFING.md`; `docs/research_log.md` ("Key Results as of 5/07/2026"); `docs/videozip_plan.md`; git log on `docs` branch.

Note on date: user wrote "5/20/2025" but memory `currentDate=2026-05-26`. Treating the cutoff as **2026-05-15** to cover everything since the last batch of active sessions.

---

## What you are doing right now

**You are between phases — the last 5 sessions were all communication / writeups, not new experiments.**

Last 5 sessions (chronological):
1. **5/17–5/22 `18d1a9cb`** — Drafted the *PURS Program Impact* paragraph (omnimodal token-compression framing, identified gap, PhD pitch).
2. **5/19 `8b4128dd`** — Wrote `_scratch/PURS_research_report.md` for Chengming + self (you don't have Gmail authed in those sessions; never sent).
3. **5/19 `eff43b5a`** — YC Startup School 2026 poster-session form: answer file at `_scratch/yc_startup_school_poster_answer.md`. Pitched **VideoZip** as the headline.
4. **5/21–5/22 `bc917773`** — Iterated the same elevator-pitch paragraph (em-dashes removed, more "humanized"). Lead message: L6 reproduces OmniZip's keep mask on ~97% of tokens, matches 1.61× prefill, with a clean mechanistic story.
5. **5/22 `7e44a781`** — You asked Claude to *launch the past 10 sessions in 10 terminals* (`claude --resume` per session in Windows Terminal). All 10 windows opened; PURS-S1…S10.

Git: on branch `docs`, last commit `d18560ec docs` (2026-05-19). No commits between 5/19 and today.

**Bottom line:** you've been packaging the story (poster form, professor report, PURS impact) for ~2 weeks. The repo's last real *experiment* completed on 2026-05-07 (the 10× matrix + L6 sweep). Nothing in the experimental queue has advanced since.

---

## What you have tried (anchored by repo, not memory)

| Thread | State | Evidence |
|---|---|---|
| **10× repeat matrix** (baseline, omnizip, mixkv@256, divprune, rediprune) | **Done.** OmniZip = baseline on accuracy (0.312 vs 0.311, T=0.1). OmniZip's real win = **35% prefill, 38% VRAM** at parity acc. MixKV@256 is broken (-11 pts). ReDiPrune > DivPrune at matched compression. | `research_log.md` "Authoritative results — 10× repeat matrix"; `runs/qwen25_matrix_gpu7_all7_snapkv/` |
| **L6 cache layer probe** | **Done.** AUC 0.6528 at L6 vs 0.5253 at L14; within-video std 0.0013 (caching safe); L6 beats L14 on 58/59 questions; 114/118 exact-prediction agreement with OmniZip; 1.61× prefill. | `vizzing/omnizip_auc_l{6,14}_worldsense/`, `docs/method_l6_omnizip.md`, `docs/l6cache.md` |
| **Mechinterp Tier-1 depth curve** (28 layers, Gini/cross-modal/last-token attn) | **Done.** Canonical 5 PNGs in `mechinterp/outputs/depth_curve/`. Findings: L0 question-aware, L4-collapse-robust. | Session `94b42dc4`, chat `chats/mechinterp_tier1_depth_curve_session.md` |
| **VideoZip — design + partial implementation** | **Spec complete, code partial.** Ultra-plan §1–§11 in `docs/videozip_plan.md`. Files: `videozip/src/videozip.py`, `videozip/eval/eval_videozip.py`, `OmniZip-main/omnizip/videozip_units.py`, `demo_videozip.py`. Not benchmarked. | Sessions `86cec7df`, `5a33dc40`; chat `2026-05-01_videozip_vitsaliency_session.md` |
| **FastKV reproduction** | Attempted via `paper2code` skill; vendor + 3-file implementation + server. | Session `55a29d09`, chat `chats/reproducing.md` |
| **Crossmodal AUC analysis** vs OmniZip | Done; doc in `vizzing/crossmodal_spearman/`. | Chat `2026-04-30_crossmodal_auc_omnizip_analysis.md` |

---

## What you were going to do next (per `research_log.md` "Next planned experiment")

The repo plan is explicit and unambiguous — this is the gating decision:

1. **Audio-guidance-OFF ablation in OmniZip** ← **gating experiment**. Modify `eval_qwen_omni_zip.py` to replace audio-derived per-group video budget with random/uniform at fixed ρ_v=0.6, all else identical. Two outcomes, both decisive:
   - random ≈ OmniZip → audio guidance is *story* not contribution → pivot to adaptive ρ + diversity.
   - random ≈ baseline → audio signal carries the gain → pursue 3-signal fusion (L6 + encoder + ReDiPrune query relevance).
2. MixKV budget sweep (512/1024/2048) — to make MixKV usable as comparator/stacking partner.
3. ReDiPrune α-sweep at fixed subset=0.5.

**Queued behind those:** OmniZip×MixKV stacking; 3-signal submodular fusion with L6-variance routing; VideoZip benchmark; hybrid α·L6 + (1−α)·encoder; saliency-entropy-driven adaptive ρ; compression × speculative decoding.

---

## What should be done next (my call)

**Run the audio-guidance-OFF ablation now.** It is the cheapest experiment in the queue (one flag in `eval_qwen_omni_zip.py`, reuses the 10× harness) and it disambiguates the entire research narrative — including the poster you just promised YC.

Reasons it dominates the alternatives:
- **It's gating.** VideoZip, MixKV stacking, fusion — all assume audio guidance carries the OmniZip gain. If random matches OmniZip, your VideoZip pitch ("audio guides video, so video should guide audio") loses its empirical motivation; the project pivots to adaptive ρ + diversity instead.
- **The 10× harness is already in place.** Same `runs/qwen25_matrix_gpu7_all7_snapkv/` layout, same temperatures, same 1.2× variance noise floor you already know. Adding a `--guidance random` arm is ~50 LOC, hours not days.
- **It's safe regardless of outcome.** Both result branches have a clear next step pre-written.
- **It locks the poster claim before YC.** Right now the pitch leans on the L6→OmniZip-equivalence + 1.61× prefill. The ablation determines whether OmniZip itself is the right reference, or whether the more honest framing is *"L6 matches an audio-mask method that itself reduces to random — and the real finding is the cacheable layer signal, full stop."*

**Concrete first step (one session):**
1. Add `--guidance {audio,random,uniform}` flag to `eval_qwen_omni_zip.py`.
2. In the per-group video-budget computation, when `guidance != audio`, replace with uniform-random sampling at the same total budget (same RNG seed across runs).
3. Run 10× repeats × 2 temperatures × {baseline, omnizip-audio, omnizip-random, omnizip-uniform} on the same VideoMME/AIR-Bench/WorldSense slice.
4. Add the result rows to `research_log.md` "10× matrix" table. Update PROJECT_BRIEFING §6 with the verdict.

**Secondary (if any time left this week):** MixKV budget sweep at 1024 — single hyperparameter change, makes MixKV a viable comparator for any later stacking story.

**Communication side, while ablation runs:** Send the PURS report (`_scratch/PURS_research_report.md`) — it was drafted 5/19 and never sent. Either authenticate Gmail in a session or paste it manually.
