"""
Qwen2.5-Omni: Audio + Video Encoding Pipeline Walkthrough
Concrete example: 4-second video with audio
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── CONSTANTS (kept small for visual clarity) ───────────────────────────────
DURATION      = 4.0        # seconds
SAMPLE_RATE   = 16_000
N_MEL         = 128
HOP_MS        = 10         # ms
WIN_MS        = 25         # ms
FRAME_MS      = 40         # one audio token = 40 ms
BLOCK_S       = 2.0        # 2-second attention block
FPS           = 2          # video frames per second (kept low for legibility)
PATCH_GRID    = 4          # 4×4 patches per frame (represents 14×14 in real model)
MERGE         = 2          # 2×2 → 1 token merge
LLM_DIM       = 4096       # symbolic

n_audio_tokens = int(DURATION / (FRAME_MS / 1000))   # 100
n_blocks       = int(DURATION / BLOCK_S)               # 2
tokens_per_block = n_audio_tokens // n_blocks          # 50
n_frames       = int(DURATION * FPS)                   # 8
patches_per_frame = PATCH_GRID ** 2                    # 16
tokens_per_frame  = (PATCH_GRID // MERGE) ** 2        # 4
n_video_tokens    = n_frames * tokens_per_frame        # 32

# ─── SYNTHETIC DATA ──────────────────────────────────────────────────────────
t_audio = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
waveform = (0.4 * np.sin(2*np.pi*220*t_audio) +
            0.3 * np.sin(2*np.pi*440*t_audio) +
            0.15 * np.random.randn(len(t_audio)))

mel_frames = int(DURATION * 1000 / HOP_MS)  # ~400
mel = np.random.rand(N_MEL, mel_frames) * 0.4
for f in range(1, 5):
    mel += 0.15 * np.outer(np.exp(-np.linspace(0, 3, N_MEL)),
                           np.sin(2*np.pi*f*np.linspace(0, DURATION, mel_frames)))
mel = np.clip(mel, 0, 1)

# ─── COLORS ──────────────────────────────────────────────────────────────────
C_AUDIO  = '#4C9BE8'
C_VIDEO  = '#E87B4C'
C_TEXT   = '#6DBE6D'
C_MERGE  = '#B46DBE'
C_BLOCK0 = '#AED6F1'
C_BLOCK1 = '#A9DFBF'
C_TROPE  = '#F9E79F'
BG       = '#1A1A2E'
FG       = '#E0E0E0'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': '#0F0F1E',
    'text.color': FG, 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG,
    'axes.edgecolor': '#444466', 'grid.color': '#333355',
})

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor(BG)
gs_main = gridspec.GridSpec(5, 1, figure=fig, hspace=0.55,
                             left=0.06, right=0.97, top=0.96, bottom=0.03)

title = fig.text(0.5, 0.978,
    'Qwen2.5-Omni — 4-Second Video+Audio Pipeline Walkthrough',
    ha='center', va='top', fontsize=18, fontweight='bold', color='white')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — AUDIO ENCODER
# ══════════════════════════════════════════════════════════════════════════════
gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0],
                                        wspace=0.35, width_ratios=[2, 3, 2])

# 1a: Waveform
ax1a = fig.add_subplot(gs1[0])
t_plot = np.linspace(0, DURATION, 4000)
w_plot = (0.4*np.sin(2*np.pi*220*t_plot) + 0.3*np.sin(2*np.pi*440*t_plot)
          + 0.1*np.random.randn(4000))
ax1a.plot(t_plot, w_plot, color=C_AUDIO, lw=0.7, alpha=0.85)
ax1a.axvline(2.0, color='yellow', lw=1.5, ls='--', alpha=0.7, label='block boundary')
ax1a.set_xlim(0, DURATION); ax1a.set_ylim(-1.1, 1.1)
ax1a.set_xlabel('Time (s)'); ax1a.set_ylabel('Amplitude')
ax1a.set_title('① Raw Waveform\n16 kHz mono', fontsize=10, color=C_AUDIO)
ax1a.legend(fontsize=7, loc='upper right')
ax1a.fill_between(t_plot, w_plot, alpha=0.15, color=C_AUDIO)
ax1a.text(0.5, -0.95, 'Block 0 (0–2s)', fontsize=7, ha='center',
          color=C_BLOCK0, transform=ax1a.get_xaxis_transform())
ax1a.text(3.0, -0.95, 'Block 1 (2–4s)', fontsize=7, ha='center',
          color=C_BLOCK1, transform=ax1a.get_xaxis_transform())

# 1b: Mel-spectrogram
ax1b = fig.add_subplot(gs1[1])
cmap_mel = LinearSegmentedColormap.from_list('mel', ['#0F0F1E', '#1a4f8a', '#4C9BE8', '#ffd700'])
im = ax1b.imshow(mel, aspect='auto', origin='lower', cmap=cmap_mel,
                 extent=[0, DURATION, 0, N_MEL])
ax1b.axvline(2.0, color='yellow', lw=1.5, ls='--', alpha=0.8)
for i, (x, label) in enumerate([(0.5, 'Audio\nBlock 0'), (2.5, 'Audio\nBlock 1')]):
    color = C_BLOCK0 if i == 0 else C_BLOCK1
    ax1b.text(x, 115, label, ha='center', fontsize=8, color=color, fontweight='bold')
ax1b.set_xlabel('Time (s)'); ax1b.set_ylabel('Mel bin (0–127)')
ax1b.set_title('② 128-ch Mel-Spectrogram\n(25ms window, 10ms hop → ~400 frames)',
               fontsize=10, color=C_AUDIO)
plt.colorbar(im, ax=ax1b, fraction=0.03, pad=0.02)

# 1c: Audio tokens diagram
ax1c = fig.add_subplot(gs1[2])
ax1c.set_xlim(0, 10); ax1c.set_ylim(-1, 4.5)
ax1c.axis('off')
ax1c.set_title('③ Audio Tokens\n(1 token = 40 ms → 50 tokens/block)',
               fontsize=10, color=C_AUDIO)

for blk, (y_off, color, label) in enumerate([(2.5, C_BLOCK0, 'Block 0\n(0–2s)\n50 tokens'),
                                              (0.5, C_BLOCK1, 'Block 1\n(2–4s)\n50 tokens')]):
    for i in range(10):
        rect = FancyBboxPatch((i * 0.9 + 0.1, y_off), 0.75, 0.8,
                              boxstyle='round,pad=0.05',
                              fc=color, ec='white', alpha=0.7, lw=0.5)
        ax1c.add_patch(rect)
        if i == 0:
            ax1c.text(i*0.9+0.48, y_off+0.4, f't{blk*50}', ha='center',
                      va='center', fontsize=5.5, color='black', fontweight='bold')
        elif i == 9:
            ax1c.text(i*0.9+0.48, y_off+0.4, f't{blk*50+49}', ha='center',
                      va='center', fontsize=5.5, color='black', fontweight='bold')
        else:
            ax1c.text(i*0.9+0.48, y_off+0.4, '·', ha='center',
                      va='center', fontsize=8, color='black')
    ax1c.text(-0.1, y_off+0.4, label, ha='right', va='center',
              fontsize=7.5, color=color, fontweight='bold')

ax1c.text(4.6, 4.1, '→ proj → LLM dim (4096)', ha='center', fontsize=8,
          color=FG, style='italic')
ax1c.annotate('', xy=(4.6, 3.9), xytext=(4.6, 3.45),
              arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — VIDEO ENCODER
# ══════════════════════════════════════════════════════════════════════════════
gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1],
                                        wspace=0.4, width_ratios=[2, 2.5, 2, 2])

# 2a: Frame timeline
ax2a = fig.add_subplot(gs2[0])
ax2a.set_xlim(-0.5, DURATION+0.5); ax2a.set_ylim(-0.5, 1.5)
ax2a.axis('off')
ax2a.set_title('① Video Frame Sampling\n(2 fps → 8 frames over 4s)',
               fontsize=10, color=C_VIDEO)
frame_times = np.arange(n_frames) / FPS
for i, ft in enumerate(frame_times):
    color = C_BLOCK0 if ft < 2.0 else C_BLOCK1
    ax2a.add_patch(FancyBboxPatch((ft-0.15, 0.3), 0.3, 0.55,
                                   boxstyle='round,pad=0.05',
                                   fc=color, ec='white', lw=0.8, alpha=0.85))
    ax2a.text(ft, 0.575, f'F{i}', ha='center', va='center',
              fontsize=8, fontweight='bold', color='black')
    ax2a.text(ft, 0.15, f'{ft:.1f}s', ha='center', va='center',
              fontsize=6.5, color=FG)
ax2a.axvline(2.0, ymin=0.25, ymax=0.95, color='yellow', lw=1.5, ls='--', alpha=0.7)
ax2a.text(2.05, 1.1, 'block\nboundary', fontsize=6.5, color='yellow', va='top')
ax2a.text(1.0, 1.35, 'Block 0', ha='center', fontsize=8, color=C_BLOCK0, fontweight='bold')
ax2a.text(3.0, 1.35, 'Block 1', ha='center', fontsize=8, color=C_BLOCK1, fontweight='bold')

# 2b: Patch extraction for one frame
ax2b = fig.add_subplot(gs2[1])
ax2b.set_xlim(-0.5, 9); ax2b.set_ylim(-0.5, 5.5)
ax2b.axis('off')
ax2b.set_title(f'② ViT Patch Extraction (Frame 0)\n{PATCH_GRID}×{PATCH_GRID} patches (rep. 14×14px each)',
               fontsize=10, color=C_VIDEO)

# Draw a frame as a grid
for r in range(PATCH_GRID):
    for c in range(PATCH_GRID):
        idx = r * PATCH_GRID + c
        color = plt.cm.viridis(idx / patches_per_frame)
        rect = Rectangle((c*0.9, (PATCH_GRID-1-r)*0.9 + 0.5), 0.85, 0.85,
                          fc=color, ec='white', lw=0.5, alpha=0.8)
        ax2b.add_patch(rect)
        ax2b.text(c*0.9+0.42, (PATCH_GRID-1-r)*0.9+0.93,
                  f'p{idx:02d}', ha='center', va='center', fontsize=5, color='white')

ax2b.text(1.7, 0.1, f'Frame 0 → {patches_per_frame} patch tokens → ViT encoder',
          ha='center', fontsize=7.5, color=FG, style='italic')

# 2c: MLP 2x2 merge
ax2c = fig.add_subplot(gs2[2])
ax2c.set_xlim(-0.5, 5); ax2c.set_ylim(-0.5, 5.5)
ax2c.axis('off')
ax2c.set_title(f'③ MLP Token Merge (2×2→1)\n{patches_per_frame} patches → {tokens_per_frame} tokens/frame',
               fontsize=10, color=C_MERGE)

MERGED_GRID = PATCH_GRID // MERGE  # 2
for r in range(MERGED_GRID):
    for c in range(MERGED_GRID):
        idx = r * MERGED_GRID + c
        # Draw 4 source patches
        for dr in range(2):
            for dc in range(2):
                src_r = r*2 + dr
                src_c = c*2 + dc
                src_idx = src_r * PATCH_GRID + src_c
                color = plt.cm.viridis(src_idx / patches_per_frame)
                rect = Rectangle((c*2.1 + dc*0.85, (MERGED_GRID-1-r)*2.1 + dr*0.85 + 0.5),
                                  0.8, 0.8, fc=color, ec='white', lw=0.5, alpha=0.6)
                ax2c.add_patch(rect)
        # Draw merged token
        color = plt.cm.plasma(idx / tokens_per_frame)
        rect = FancyBboxPatch((c*2.1 + 3.1, (MERGED_GRID-1-r)*2.1 + 0.85 + 0.5),
                              1.1, 1.0, boxstyle='round,pad=0.08',
                              fc=color, ec='white', lw=1.5, alpha=0.9)
        ax2c.add_patch(rect)
        ax2c.text(c*2.1 + 3.65, (MERGED_GRID-1-r)*2.1 + 1.35 + 0.5,
                  f'v{idx}', ha='center', va='center', fontsize=9,
                  fontweight='bold', color='white')
        ax2c.annotate('', xy=(c*2.1+3.05, (MERGED_GRID-1-r)*2.1+1.35+0.5),
                      xytext=(c*2.1+1.85, (MERGED_GRID-1-r)*2.1+1.35+0.5),
                      arrowprops=dict(arrowstyle='->', color=C_MERGE, lw=1.5))

ax2c.text(2.2, 0.0, 'MLP compresses 4 patches → 1 token', ha='center',
          fontsize=7.5, color=C_MERGE, style='italic')

# 2d: All frames → token count
ax2d = fig.add_subplot(gs2[3])
ax2d.set_xlim(-0.5, 4.5); ax2d.set_ylim(-0.5, 5.5)
ax2d.axis('off')
ax2d.set_title(f'④ Total Video Tokens\n{n_frames} frames × {tokens_per_frame} tokens = {n_video_tokens}',
               fontsize=10, color=C_VIDEO)

for fi in range(n_frames):
    color = C_BLOCK0 if fi < n_frames//2 else C_BLOCK1
    row = fi // 2
    col = fi % 2
    ax2d.add_patch(FancyBboxPatch((col*1.9, (n_frames//2 - 1 - row)*0.85 + 0.3),
                                   1.7, 0.7, boxstyle='round,pad=0.05',
                                   fc=color, ec='white', lw=0.8, alpha=0.8))
    ax2d.text(col*1.9+0.85, (n_frames//2-1-row)*0.85+0.65,
              f'F{fi}: {tokens_per_frame} vtoks', ha='center', va='center',
              fontsize=7.5, fontweight='bold', color='black')

ax2d.text(1.75, -0.1, f'→ proj → LLM dim (4096)', ha='center',
          fontsize=8, color=FG, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — TMRoPE ID ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs_main[2])
ax3.axis('off')
ax3.set_title('TMRoPE — Time-aligned Multimodal RoPE: Concrete ID Assignments',
              fontsize=13, color=C_TROPE, pad=12, fontweight='bold')

# Build table data
rows = []
# Text tokens (first few)
for i in range(3):
    rows.append(('Text', f'tok_{i}', str(i), str(i), str(i), '—',
                 f'All 3 IDs same → standard 1D-RoPE'))
rows.append(('Text', '...', '...', '...', '...', '—', ''))

# Audio tokens
for i in range(0, 6):
    block = 'B0' if i < 3 else 'B1'
    t_id = i  # 40ms steps
    rows.append(('Audio', f'a_{i}', str(t_id), str(t_id), str(t_id), block,
                 f't={i*40}ms: Temporal={t_id}, H=W=same'))
rows.append(('Audio', '...', '...', '...', '...', '—', ''))

# Video tokens — Frame 0 (t=0s → temporal_id=0)
frame_t_ids = [0, 0, 12, 12, 25, 25, 37, 37]  # temporal IDs per frame (2fps → 500ms apart = 12.5 steps)
for fi in range(4):
    t_id = frame_t_ids[fi*2]
    block = 'B0' if fi < n_frames//2 else 'B1'
    for vi in range(min(2, tokens_per_frame)):
        h_id = vi // MERGED_GRID
        w_id = vi % MERGED_GRID
        rows.append(('Video', f'F{fi}_v{vi}', str(t_id), str(h_id), str(w_id), block,
                     f'Frame{fi} t={fi*500}ms: Temporal={t_id}, H={h_id}, W={w_id}'))
    rows.append(('Video', '...', '...', '...', '...', block, ''))

cols = ['Modality', 'Token', 'Temporal ID\n(40ms units)', 'Height ID', 'Width ID',
        'Block', 'Notes']
col_x  = [0.02, 0.13, 0.24, 0.35, 0.43, 0.51, 0.60]
col_w  = [0.10, 0.10, 0.10, 0.07, 0.07, 0.08, 0.39]
row_h  = 0.048
header_y = 0.93
n_show = min(len(rows), 18)

# Header
for j, (cx, cw, col) in enumerate(zip(col_x, col_w, cols)):
    ax3.text(cx + cw/2, header_y, col, ha='center', va='top',
             fontsize=8.5, fontweight='bold', color=C_TROPE,
             transform=ax3.transAxes)

ax3.plot([0.01, 0.99], [header_y - 0.03, header_y - 0.03],
         color=C_TROPE, lw=0.8, transform=ax3.transAxes)

mod_colors = {'Text': C_TEXT, 'Audio': C_AUDIO, 'Video': C_VIDEO}
for i, row in enumerate(rows[:n_show]):
    y = header_y - 0.04 - i * row_h
    mod = row[0]
    bg_alpha = 0.08 if i % 2 == 0 else 0.03
    ax3.add_patch(Rectangle((0.01, y - row_h*0.85), 0.98, row_h,
                             fc='white', alpha=bg_alpha,
                             transform=ax3.transAxes, clip_on=False))
    for j, (cx, cw, val) in enumerate(zip(col_x, col_w, row)):
        color = mod_colors.get(mod, FG) if j == 0 else FG
        fontw = 'bold' if j == 0 else 'normal'
        ax3.text(cx + cw/2, y - row_h*0.3, val, ha='center', va='top',
                 fontsize=7.5, color=color, fontweight=fontw,
                 transform=ax3.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — 2-SECOND INTERLEAVED SEQUENCE
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs_main[3])
ax4.axis('off')
ax4.set_title('2-Second Chunk Interleaving → Final Token Sequence into Thinker LLM',
              fontsize=13, color='white', pad=12, fontweight='bold')

# Build the actual interleaved sequence for 4s of content
# Each 2s block: video tokens (frames in that block) + audio tokens (50 per block)
sequence = []
for blk in range(n_blocks):
    frame_start = blk * (n_frames // n_blocks)
    frame_end   = frame_start + (n_frames // n_blocks)
    for fi in range(frame_start, frame_end):
        for vi in range(tokens_per_frame):
            sequence.append(('V', fi, vi, blk))
    for ai in range(tokens_per_block):
        abs_ai = blk * tokens_per_block + ai
        sequence.append(('A', abs_ai, 0, blk))

# Draw the sequence as colored boxes
n_seq = len(sequence)
box_w = 0.012
gap   = 0.001
total_w = n_seq * (box_w + gap)
start_x = (1.0 - total_w) / 2

y_seq = 0.75
ax4.text(0.5, 0.96, f'Total: {n_seq} tokens  '
         f'({n_video_tokens} video [orange] + {n_audio_tokens} audio [blue])',
         ha='center', va='top', fontsize=10, color=FG,
         transform=ax4.transAxes)

block_labels_done = [False, False]
for i, (mod, idx_a, idx_b, blk) in enumerate(sequence):
    x = start_x + i * (box_w + gap)
    color = C_VIDEO if mod == 'V' else C_AUDIO
    alpha = 0.85 if blk == 0 else 0.65
    ax4.add_patch(Rectangle((x, y_seq - 0.12), box_w, 0.22,
                             fc=color, ec='none', alpha=alpha,
                             transform=ax4.transAxes))

# Block boundary lines & labels
for blk in range(n_blocks):
    # find first index of this block
    first_i = next(i for i, s in enumerate(sequence) if s[3] == blk)
    last_i  = next((i for i, s in enumerate(sequence) if s[3] > blk), len(sequence))
    x_start = start_x + first_i * (box_w + gap)
    x_end   = start_x + last_i  * (box_w + gap) - gap
    mid_x   = (x_start + x_end) / 2
    bcolor  = C_BLOCK0 if blk == 0 else C_BLOCK1
    ax4.add_patch(Rectangle((x_start, y_seq - 0.18), x_end - x_start, 0.35,
                             fc='none', ec=bcolor, lw=2, alpha=0.8,
                             transform=ax4.transAxes))
    ax4.text(mid_x, y_seq + 0.22,
             f'Chunk {blk} (t={blk*2:.0f}–{blk*2+2:.0f}s)',
             ha='center', va='bottom', fontsize=9, color=bcolor, fontweight='bold',
             transform=ax4.transAxes)

    # Mark video/audio sub-sections
    v_indices = [i for i, s in enumerate(sequence) if s[3] == blk and s[0] == 'V']
    a_indices = [i for i, s in enumerate(sequence) if s[3] == blk and s[0] == 'A']
    if v_indices:
        vx = start_x + v_indices[0] * (box_w + gap)
        ax4.text(vx + len(v_indices)*(box_w+gap)/2, y_seq - 0.25,
                 f'Video ({len(v_indices)} tok)', ha='center', fontsize=8,
                 color=C_VIDEO, transform=ax4.transAxes, fontweight='bold')
    if a_indices:
        ax4.text(start_x + a_indices[0]*(box_w+gap) + len(a_indices)*(box_w+gap)/2,
                 y_seq - 0.25, f'Audio ({len(a_indices)} tok)',
                 ha='center', fontsize=8, color=C_AUDIO,
                 transform=ax4.transAxes, fontweight='bold')

ax4.text(0.5, 0.12,
         'Within each 2s chunk: Visual tokens FIRST → Audio tokens SECOND\n'
         'TMRoPE temporal IDs ensure audio @ t=1.2s and video frame @ t=1.0s '
         'carry matching temporal coordinates for cross-modal attention',
         ha='center', va='bottom', fontsize=9.5, color=FG, style='italic',
         transform=ax4.transAxes)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — ARCHITECTURE FLOW SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs_main[4])
ax5.set_xlim(0, 20); ax5.set_ylim(0, 5)
ax5.axis('off')
ax5.set_title('End-to-End Architecture Flow Summary',
              fontsize=13, color='white', pad=8, fontweight='bold')

def draw_box(ax, x, y, w, h, label, sublabel='', color='#4466aa', fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                                fc=color, ec='white', lw=1.5, alpha=0.85))
    ax.text(x+w/2, y+h/2 + (0.2 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white')
    if sublabel:
        ax.text(x+w/2, y+h/2 - 0.3, sublabel,
                ha='center', va='center', fontsize=7, color='#cccccc')

def draw_arrow(ax, x1, y1, x2, y2, color='white'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Row 1: encoders
draw_box(ax5, 0.3, 2.8, 3.5, 1.6, 'Audio Encoder', 'Qwen2-Audio\n128-mel → block-wise attn\n100 audio tokens', C_AUDIO)
draw_box(ax5, 0.3, 0.5, 3.5, 1.6, 'Video Encoder', 'ViT-675M (Qwen2.5-VL)\n14×14 patches + 2×2 MLP merge\n32 video tokens', C_VIDEO)

# Projections
draw_box(ax5, 5.0, 2.8, 2.2, 1.6, 'Linear\nProjection', '→ 4096 dim', '#666699')
draw_box(ax5, 5.0, 0.5, 2.2, 1.6, 'Linear\nProjection', '→ 4096 dim', '#666699')

draw_arrow(ax5, 3.8, 3.6, 5.0, 3.6)
draw_arrow(ax5, 3.8, 1.3, 5.0, 1.3)

# TMRoPE box
draw_box(ax5, 5.0, 5.3, 9.5, 0.9, 'TMRoPE: Temporal(40ms) + Height + Width → applied to ALL tokens',
         '', C_TROPE, fontsize=8)

# Interleaving
draw_box(ax5, 8.5, 1.5, 2.5, 2.0, '2s Chunk\nInterleaving',
         'Video first,\naudio second\nper chunk', '#886633')

draw_arrow(ax5, 7.2, 3.6, 8.5, 2.8)
draw_arrow(ax5, 7.2, 1.3, 8.5, 2.0)

# Thinker
draw_box(ax5, 12.0, 1.2, 4.0, 2.5, 'Thinker LLM\n(Transformer Decoder)',
         'Full self-attention over ALL tokens\n(text + audio + video)\nGenerates text + hidden states', '#3a7a3a', fontsize=9)

draw_arrow(ax5, 11.0, 2.5, 12.0, 2.5)

# Talker
draw_box(ax5, 17.2, 1.8, 2.5, 1.5, 'Talker', 'Dual-track\nauto-regressive\nspeech tokens', '#7a3a7a')
draw_arrow(ax5, 16.0, 2.5, 17.2, 2.5)

ax5.text(10.0, 0.2,
    f'Input: {n_video_tokens} video tokens + {n_audio_tokens} audio tokens '
    f'+ N text tokens → interleaved sequence of {n_video_tokens + n_audio_tokens}+ tokens with TMRoPE',
    ha='center', va='bottom', fontsize=8.5, color=FG, style='italic')

plt.savefig(r'C:\Users\Armaan\Desktop\PURS\qwen_omni_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: qwen_omni_pipeline.png")
plt.close()
