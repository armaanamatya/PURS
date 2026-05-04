"""
TMRoPE — Time-aligned Multimodal RoPE: Visual Deep Dive
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc, FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import warnings; warnings.filterwarnings('ignore')

np.random.seed(0)

BG = '#0D1117'; FG = '#E6EDF3'
CA = '#58A6FF'; CV = '#F78166'; CT = '#3FB950'; CLIN = '#BC8CFF'
CY = '#E3B341'; CDIM = '#8B949E'; CRED = '#FF7B72'
C_TEMP = '#FF9500'; C_H = '#00BFFF'; C_W = '#00E676'

plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': '#161B22',
    'text.color': FG, 'axes.labelcolor': FG, 'xtick.color': FG,
    'ytick.color': FG, 'axes.edgecolor': '#30363D', 'grid.alpha': 0.3})

fig = plt.figure(figsize=(24, 30))
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.988, 'TMRoPE — Time-aligned Multimodal Rotary Position Embedding',
         ha='center', fontsize=21, fontweight='bold', color='white')
fig.text(0.5, 0.982, 'How position is encoded via rotation, and how 3 axes align audio, video, and text in time',
         ha='center', fontsize=11, color=CDIM)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32,
                       left=0.05, right=0.97, top=0.978, bottom=0.02)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — What RoPE is: rotating a vector by position × frequency
# ══════════════════════════════════════════════════════════════════════════════
axA = fig.add_subplot(gs[0, 0])
axA.set_facecolor('#0D1117')
axA.set_xlim(-1.7, 1.7); axA.set_ylim(-1.7, 1.7)
axA.set_aspect('equal')
axA.set_title('① What is RoPE? — Rotating a dimension pair by angle θ·p',
              fontsize=11, color=CY, pad=8, fontweight='bold')
axA.axhline(0, color='#333355', lw=0.8); axA.axvline(0, color='#333355', lw=0.8)
axA.set_xlabel('dim 2i', fontsize=9); axA.set_ylabel('dim 2i+1', fontsize=9)

circle = plt.Circle((0, 0), 1.0, fc='none', ec='#333355', lw=1, ls='--')
axA.add_patch(circle)

# Show a base vector q, then rotated versions at different positions
q0 = np.array([0.85, 0.3])
q0 /= np.linalg.norm(q0)
theta_base = np.arctan2(q0[1], q0[0])

positions = [0, 2, 5, 9]
colors_p  = ['#888888', '#4488FF', '#FF8844', '#44FF88']
labels_p  = ['p=0', 'p=2', 'p=5', 'p=9']
freq = 0.35  # θ for this dim pair

for p, color, label in zip(positions, colors_p, labels_p):
    angle = theta_base + p * freq
    qp = np.array([np.cos(angle), np.sin(angle)])
    axA.annotate('', xy=qp, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    axA.text(qp[0]*1.18, qp[1]*1.18, label, ha='center', va='center',
             fontsize=9, color=color, fontweight='bold')

# Arc showing rotation from p=0 to p=2
a0 = theta_base; a2 = theta_base + 2*freq
arc_theta = np.linspace(a0, a2, 40)
axA.plot(np.cos(arc_theta)*0.6, np.sin(arc_theta)*0.6, color='yellow', lw=1.5, ls='--')
axA.text(np.cos((a0+a2)/2)*0.75, np.sin((a0+a2)/2)*0.75,
         '2·θ', color='yellow', fontsize=9, ha='center')

axA.text(0, -1.58,
    'q rotated = [q·cos(p·θ) − k·sin(p·θ),  q·sin(p·θ) + k·cos(p·θ)]\n'
    'θ = 1/10000^(2i/d)  — smaller for higher dims (lower frequency)\n'
    'Dot product of two rotated vectors only depends on (p_q − p_k)  → relative pos',
    ha='center', va='center', fontsize=8.5, color=FG, family='monospace',
    bbox=dict(fc='#1A1500', ec=CY, lw=1, boxstyle='round,pad=0.3'))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Frequency spectrum across dimensions
# ══════════════════════════════════════════════════════════════════════════════
axB = fig.add_subplot(gs[0, 1])
axB.set_facecolor('#0D1117')
axB.set_title('② Frequency per Dimension Pair — low i = high freq, high i = low freq',
              fontsize=11, color=CY, pad=8, fontweight='bold')
d = 128  # representative hidden dim per axis
i_vals = np.arange(d//2)
thetas = 1.0 / (10000 ** (2*i_vals / d))
axB.semilogy(i_vals, thetas, color=CY, lw=2.5)
axB.fill_between(i_vals, thetas, alpha=0.15, color=CY)
axB.set_xlabel('Dimension pair index i', fontsize=9)
axB.set_ylabel('θᵢ = 1/10000^(2i/d)', fontsize=9)
axB.axvline(d//6, color=C_TEMP, lw=1.5, ls='--', label='low i: fast rotation\n(captures fine position)')
axB.axvline(d//2-1, color=CLIN, lw=1.5, ls='--', label='high i: slow rotation\n(captures coarse position)')
axB.legend(fontsize=8, loc='upper right', framealpha=0.3)
axB.grid(True, alpha=0.2)
axB.text(5, thetas[5]*2.5, 'high freq\n(fine detail)', fontsize=8, color=C_TEMP)
axB.text(45, thetas[45]*3, 'low freq\n(coarse position)', fontsize=8, color=CLIN)
axB.text(0.5, 0.05,
    'Together, all dim-pairs form a unique "rotation fingerprint" for each position.\n'
    'Nearby positions → similar fingerprints → high attention scores.',
    transform=axB.transAxes, ha='center', fontsize=8.5, color=FG,
    bbox=dict(fc='#1A1500', ec=CY, lw=1, boxstyle='round,pad=0.3'))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — The 3-axis dimension split in TMRoPE
# ══════════════════════════════════════════════════════════════════════════════
axC = fig.add_subplot(gs[1, 0])
axC.set_xlim(0, 12); axC.set_ylim(0, 7)
axC.axis('off')
axC.set_title('③ TMRoPE — Splitting the Embedding into 3 Independent RoPE Axes',
              fontsize=11, color=CLIN, pad=8, fontweight='bold')

D = 3584  # Qwen2.5-Omni LLM hidden dim
d_each = D // 3  # each axis gets ~1/3 of dimensions

# Draw the embedding dimension bar
bar_y = 5.4; bar_h = 1.1
total_w = 11.0
axC.add_patch(FancyBboxPatch((0.5, bar_y), total_w/3, bar_h,
    boxstyle='square,pad=0', fc='#3A2000', ec=C_TEMP, lw=2.5, alpha=0.9))
axC.add_patch(FancyBboxPatch((0.5+total_w/3, bar_y), total_w/3, bar_h,
    boxstyle='square,pad=0', fc='#001A3A', ec=C_H, lw=2.5, alpha=0.9))
axC.add_patch(FancyBboxPatch((0.5+2*total_w/3, bar_y), total_w/3, bar_h,
    boxstyle='square,pad=0', fc='#00210F', ec=C_W, lw=2.5, alpha=0.9))

axC.text(0.5 + total_w/6,     bar_y+bar_h/2, f'Temporal\ndims 0–{d_each-1}',
         ha='center', va='center', fontsize=10, color=C_TEMP, fontweight='bold')
axC.text(0.5 + total_w/2,     bar_y+bar_h/2, f'Height\ndims {d_each}–{2*d_each-1}',
         ha='center', va='center', fontsize=10, color=C_H, fontweight='bold')
axC.text(0.5 + 5*total_w/6,   bar_y+bar_h/2, f'Width\ndims {2*d_each}–{D-1}',
         ha='center', va='center', fontsize=10, color=C_W, fontweight='bold')

axC.text(6.0, bar_y+bar_h+0.35,
         f'Full embedding  d={D}  →  each axis gets d/3 = {d_each} dims',
         ha='center', fontsize=9, color=FG)

# Per-modality ID assignment table
rows = [
    ('Text',          CT,   'seq_pos',          'seq_pos',          'seq_pos',          'All 3 axes same → collapses to standard 1D RoPE'),
    ('Audio token',   CA,   't/40ms',           'same as temporal', 'same as temporal', 'Temporal increments; H=W follow (only temporal matters)'),
    ('Image patch',   CLIN, 'const per image',  'patch_row',        'patch_col',        'Temporal frozen; H/W vary per patch position'),
    ('Video frame\n+ Audio', CV, 'frame_t/40ms', 'patch_row',       'patch_col',        'Temporal from timestamp; H/W from patch; audio locks to frame time'),
]

col_x = [0.0, 1.8, 3.6, 5.9, 8.2, 10.3]
headers = ['Modality', 'Temporal ID', 'Height ID', 'Width ID', 'Notes']
for j, (hx, hdr) in enumerate(zip(col_x[1:], headers)):
    axC.text(hx, 5.1, hdr, ha='left', fontsize=8.5, color=CDIM, fontweight='bold')
axC.plot([0.0, 11.5], [5.0, 5.0], color='#444466', lw=0.8)

for i, (mod, color, tid, hid, wid, note) in enumerate(rows):
    y = 4.3 - i * 1.02
    if i % 2 == 0:
        axC.add_patch(FancyBboxPatch((-0.1, y-0.2), 11.7, 0.85,
            boxstyle='square,pad=0', fc='white', alpha=0.04, ec='none'))
    axC.text(col_x[0], y+0.25, mod, ha='left', fontsize=8.5, color=color, fontweight='bold')
    axC.add_patch(FancyBboxPatch((col_x[1]-0.1, y+0.08), 2.0, 0.5,
        boxstyle='round,pad=0.05', fc='#3A2000', ec=C_TEMP, lw=1, alpha=0.8))
    axC.text(col_x[1]+0.9, y+0.33, tid, ha='center', va='center',
             fontsize=7.5, color=C_TEMP, family='monospace')
    axC.add_patch(FancyBboxPatch((col_x[2]-0.1, y+0.08), 2.1, 0.5,
        boxstyle='round,pad=0.05', fc='#001A3A', ec=C_H, lw=1, alpha=0.8))
    axC.text(col_x[2]+0.95, y+0.33, hid, ha='center', va='center',
             fontsize=7.5, color=C_H, family='monospace')
    axC.add_patch(FancyBboxPatch((col_x[3]-0.1, y+0.08), 2.1, 0.5,
        boxstyle='round,pad=0.05', fc='#00210F', ec=C_W, lw=1, alpha=0.8))
    axC.text(col_x[3]+0.95, y+0.33, wid, ha='center', va='center',
             fontsize=7.5, color=C_W, family='monospace')
    axC.text(col_x[4], y+0.25, note, ha='left', fontsize=7.5, color=CDIM)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — Concrete 6-second video example: ID assignment timeline
# ══════════════════════════════════════════════════════════════════════════════
axD = fig.add_subplot(gs[1, 1])
axD.set_xlim(-0.5, 12.5); axD.set_ylim(-1.5, 9.0)
axD.axis('off')
axD.set_title('④ Concrete Example: 6s Video — Temporal ID Assignment per Token',
              fontsize=11, color=C_TEMP, pad=8, fontweight='bold')

# Time axis
for t in range(7):
    axD.plot([t*2, t*2], [7.8, 8.1], color='#666688', lw=1)
    axD.text(t*2, 8.3, f'{t*0.5:.1f}s', ha='center', fontsize=7.5, color=CDIM)
axD.plot([0, 12], [7.9, 7.9], color='#666688', lw=1.5)
axD.text(-0.4, 7.9, 'time', va='center', fontsize=8, color=CDIM)

# 1 ID = 40ms → 25 IDs per second
fps = 5; duration = 1.5  # 1.5s for clarity
frame_times = np.arange(0, duration, 1/fps)  # 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2
audio_times = np.arange(0, duration, 0.04)   # every 40ms

# Draw audio tokens
for at in audio_times:
    tid = int(round(at / 0.04))
    ax_x = at * 8 / duration
    axD.add_patch(FancyBboxPatch((ax_x-0.13, 6.0), 0.26, 0.7,
        boxstyle='round,pad=0.05', fc=CA, ec='none', alpha=0.7))
    if tid % 5 == 0:
        axD.text(ax_x, 5.7, f't={tid}', ha='center', fontsize=6.5,
                 color=CA, fontweight='bold')

axD.text(-0.4, 6.35, 'Audio\ntokens', ha='right', va='center', fontsize=8.5, color=CA)
axD.text(-0.4, 5.65, '(Na total)', ha='right', va='center', fontsize=7, color=CDIM)

# Draw video frames
for ft in frame_times:
    tid = int(round(ft / 0.04))
    ax_x = ft * 8 / duration
    # Frame box (represents 4×4 patches, show 2×2 here)
    for pr in range(2):
        for pc in range(2):
            px = ax_x - 0.45 + pc*0.45
            py = 3.2 + pr*0.45
            axD.add_patch(FancyBboxPatch((px, py), 0.4, 0.4,
                boxstyle='square,pad=0', fc=CV, ec='white', lw=0.5, alpha=0.65))
            # Show IDs only for first and middle frame
            if ft in [0.0, 0.6]:
                axD.text(px+0.2, py+0.2, f'T={tid}\nH={pr}\nW={pc}',
                         ha='center', va='center', fontsize=5, color='white')
    if tid % 5 == 0 or ft == 0:
        axD.text(ax_x, 3.0, f't={tid}', ha='center', fontsize=6.5,
                 color=CV, fontweight='bold')

axD.text(-0.4, 3.85, 'Video\npatches', ha='right', va='center', fontsize=8.5, color=CV)
axD.text(-0.4, 3.15, '(2×2 shown)', ha='right', va='center', fontsize=7, color=CDIM)

# Alignment lines: audio token and video patch at same timestamp share Temporal ID
for ft in [0.0, 0.4, 0.8]:
    tid = int(round(ft / 0.04))
    ax_x = ft * 8 / duration
    axD.plot([ax_x, ax_x], [3.1, 5.9], color='yellow', lw=1.2, ls='--', alpha=0.7)
    axD.text(ax_x, 2.65, f'Temporal ID = {tid}\nshared!', ha='center', fontsize=7,
             color='yellow', fontweight='bold')

# Text tokens (independent sequential)
text_pos = [9.5, 10.3, 11.1, 11.9]
for i, tx in enumerate(text_pos):
    axD.add_patch(FancyBboxPatch((tx-0.28, 5.8), 0.56, 0.7,
        boxstyle='round,pad=0.06', fc=CT, ec='none', alpha=0.75))
    axD.text(tx, 6.15, f'T={i}', ha='center', fontsize=7.5, color='white', fontweight='bold')
    axD.add_patch(FancyBboxPatch((tx-0.28, 4.9), 0.56, 0.7,
        boxstyle='round,pad=0.06', fc=CT, ec='none', alpha=0.55))
    axD.text(tx, 5.25, f'H={i}', ha='center', fontsize=7.5, color='white')
    axD.add_patch(FancyBboxPatch((tx-0.28, 4.0), 0.56, 0.7,
        boxstyle='round,pad=0.06', fc=CT, ec='none', alpha=0.35))
    axD.text(tx, 4.35, f'W={i}', ha='center', fontsize=7.5, color='white')

axD.text(10.7, 7.0, 'Text tokens:\nT = H = W = seq_pos\n→ same as 1D RoPE', ha='center',
         fontsize=8, color=CT, bbox=dict(fc='#0A1A0A', ec=CT, lw=1, boxstyle='round,pad=0.3'))

axD.text(4.0, -0.5,
    'KEY INSIGHT: Audio token at t=0.4s and video patch from frame at t=0.4s '
    'both get Temporal ID = 10.\n'
    'Their Temporal RoPE rotations match → attention sees them as "at the same moment" '
    'regardless of their positions in the sequence.',
    ha='center', fontsize=9, color=FG,
    bbox=dict(fc='#1A1500', ec='yellow', lw=1.5, boxstyle='round,pad=0.4'))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL E — How attention uses TMRoPE: dot product breakdown
# ══════════════════════════════════════════════════════════════════════════════
axE = fig.add_subplot(gs[2, 0])
axE.set_xlim(0, 12); axE.set_ylim(0, 8.5)
axE.axis('off')
axE.set_title('⑤ How Attention Uses TMRoPE — Dot Product Decomposes into 3 Independent Terms',
              fontsize=11, color=CLIN, pad=8, fontweight='bold')

# Show that Q·K = Q_t·K_t + Q_h·K_h + Q_w·K_w
axE.text(6.0, 8.1, 'attention_score(q, k) = RoPE(q, ids_q) · RoPE(k, ids_k)',
         ha='center', fontsize=11, color=FG, fontweight='bold')

# 3 terms
for i, (name, color, desc, ids_q, ids_k, match) in enumerate([
    ('Temporal', C_TEMP, 'q_t · k_t  =  f(t_q − t_k)',
     'T_q = 10', 'T_k = 10', True),
    ('Height',   C_H,   'q_h · k_h  =  f(h_q − h_k)',
     'H_q = 1', 'H_k = 3', False),
    ('Width',    C_W,   'q_w · k_w  =  f(w_q − w_k)',
     'W_q = 2', 'W_k = 2', True),
]):
    bx = 0.5 + i*3.9; by = 5.5
    border_color = color
    axE.add_patch(FancyBboxPatch((bx, by), 3.5, 2.3,
        boxstyle='round,pad=0.15', fc='#161B22', ec=border_color, lw=2))
    axE.text(bx+1.75, by+2.1, f'{name} term', ha='center', fontsize=10,
             color=color, fontweight='bold')
    axE.text(bx+1.75, by+1.55, desc, ha='center', fontsize=9, color=FG, family='monospace')
    axE.add_patch(FancyBboxPatch((bx+0.2, by+0.6), 1.3, 0.7,
        boxstyle='round,pad=0.08', fc='#3A2000' if i==0 else '#001A3A' if i==1 else '#00210F',
        ec=color, lw=1))
    axE.text(bx+0.85, by+0.95, ids_q, ha='center', va='center', fontsize=8, color=color)
    axE.add_patch(FancyBboxPatch((bx+1.7, by+0.6), 1.3, 0.7,
        boxstyle='round,pad=0.08', fc='#3A2000' if i==0 else '#001A3A' if i==1 else '#00210F',
        ec=color, lw=1))
    axE.text(bx+2.35, by+0.95, ids_k, ha='center', va='center', fontsize=8, color=color)
    match_text = '✓ MATCH' if match else '✗ differ'
    match_color = '#44FF44' if match else '#FF4444'
    axE.text(bx+1.75, by+0.35, match_text, ha='center', fontsize=9,
             color=match_color, fontweight='bold')

# Plus signs
axE.text(4.2, 6.65, '+', ha='center', fontsize=20, color=FG)
axE.text(8.1, 6.65, '+', ha='center', fontsize=20, color=FG)

# Result
axE.text(6.0, 5.15, '= high (same time) + medium (diff row) + high (same col)',
         ha='center', fontsize=9.5, color=FG)

# Example: audio token attends to video patch at same time
axE.add_patch(FancyBboxPatch((0.3, 0.5), 5.2, 4.3,
    boxstyle='round,pad=0.15', fc='#0A1520', ec=CA, lw=1.5))
axE.text(2.9, 4.55, 'Audio token  a₁₀  (t=400ms)', ha='center', fontsize=9,
         color=CA, fontweight='bold')
axE.text(0.5, 4.1, 'Temporal ID: 10   ←→  same as video frame at t=400ms', fontsize=8.5, color=C_TEMP)
axE.text(0.5, 3.65, 'Height ID:   10   (H=W=T for audio)', fontsize=8.5, color=C_H)
axE.text(0.5, 3.2,  'Width ID:    10   (H=W=T for audio)', fontsize=8.5, color=C_W)
axE.text(0.5, 2.6,  'When attending to a video patch (T=10, H=2, W=3):', fontsize=8.5, color=FG)
axE.text(0.5, 2.1,  f'  Temporal term:  T_diff = 0   → max contribution ✓', fontsize=8.5, color=C_TEMP)
axE.text(0.5, 1.65, f'  Height term:    T_diff = 8   → small contribution', fontsize=8.5, color=C_H)
axE.text(0.5, 1.2,  f'  Width term:     T_diff = 7   → small contribution', fontsize=8.5, color=C_W)
axE.text(0.5, 0.65, '→ Total score dominated by temporal alignment\n'
                     '→ Audio naturally attends to video from the same moment',
         fontsize=8.5, color=FG, fontweight='bold')

# Compare same-time vs different-time
axE.add_patch(FancyBboxPatch((6.2, 0.5), 5.5, 4.3,
    boxstyle='round,pad=0.15', fc='#1A0A0A', ec=CRED, lw=1.5))
axE.text(8.95, 4.55, 'Why NOT attend to wrong time?', ha='center', fontsize=9,
         color=CRED, fontweight='bold')

t_vals = np.array([0, 5, 10, 15, 20])
score_same = np.exp(-0.0 * t_vals**2)  # T_diff=0: max
score_diff5 = np.exp(-0.05 * t_vals**2)

inner = axE.inset_axes([0.555, 0.09, 0.42, 0.75], transform=axE.transAxes)
inner.set_facecolor('#0D1117')
t_range = np.linspace(0, 25, 200)
inner.plot(t_range, np.exp(-0.035*t_range**2), color=C_TEMP, lw=2.5,
           label='temporal contribution')
inner.axvline(0, color='#44FF44', lw=1.5, ls='--', label='same time (T_diff=0)')
inner.axvline(10, color='#FF4444', lw=1.5, ls='--', label='10 steps apart (400ms)')
inner.set_xlabel('|T_q − T_k|', fontsize=7.5, color=FG)
inner.set_ylabel('attn score\ncontribution', fontsize=7.5, color=FG)
inner.tick_params(labelsize=7, colors=FG)
inner.legend(fontsize=6.5, framealpha=0.3)
for sp in inner.spines.values(): sp.set_edgecolor('#444466')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL F — 2-Second Interleaving + TMRoPE ID trace
# ══════════════════════════════════════════════════════════════════════════════
axF = fig.add_subplot(gs[2, 1])
axF.set_xlim(0, 12); axF.set_ylim(0, 8.5)
axF.axis('off')
axF.set_title('⑥ 2-Second Chunk Interleaving — TMRoPE IDs in the Actual Sequence',
              fontsize=11, color=CV, pad=8, fontweight='bold')

# Sequence layout: chunk 0 then chunk 1
chunk_specs = [
    # (label, color, temporal_id, h, w, x_start)
    # Chunk 0: 4 video patches (2 frames × 2 patches) then 5 audio tokens
    ('V F0\nT=0 H=0 W=0', CV, 0, 0, 0, 0.3),
    ('V F0\nT=0 H=0 W=1', CV, 0, 0, 1, 1.55),
    ('V F1\nT=5 H=0 W=0', CV, 5, 0, 0, 2.8),
    ('V F1\nT=5 H=0 W=1', CV, 5, 0, 1, 4.05),
    ('A t=0\nT=0', CA, 0, 0, 0, 5.3),
    ('A t=1\nT=1', CA, 1, 1, 1, 6.25),
    ('A t=2\nT=2', CA, 2, 2, 2, 7.2),
    ('A t=3\nT=3', CA, 3, 3, 3, 8.15),
    ('A t=4\nT=4', CA, 4, 4, 4, 9.1),
]

y_token = 6.0
for spec in chunk_specs:
    label, color, t_id, h_id, w_id, bx = spec
    axF.add_patch(FancyBboxPatch((bx, y_token), 1.1, 1.8,
        boxstyle='round,pad=0.08', fc=color, ec='white', lw=0.8, alpha=0.75))
    axF.text(bx+0.55, y_token+1.55, label.split('\n')[0], ha='center', va='top',
             fontsize=7.5, color='white', fontweight='bold')

    # Show 3 ID bars
    bar_h_px = 0.38
    for j, (val, col) in enumerate([(t_id, C_TEMP), (h_id, C_H), (w_id, C_W)]):
        by2 = y_token + 0.08 + j*0.41
        axF.add_patch(FancyBboxPatch((bx+0.05, by2), 1.0, bar_h_px,
            boxstyle='square,pad=0', fc=col, ec='none', alpha=0.6))
        axF.text(bx+0.55, by2+bar_h_px/2, str(val), ha='center', va='center',
                 fontsize=7, color='white', fontweight='bold')

# Chunk boundary
axF.add_patch(FancyBboxPatch((0.2, y_token-0.1), 10.1, 2.1,
    boxstyle='round,pad=0.1', fc='none', ec='yellow', lw=2, alpha=0.7))
axF.text(5.25, y_token+2.2, 'Chunk 0  (t=0–2s):  Video tokens first, then Audio tokens',
         ha='center', fontsize=9, color='yellow', fontweight='bold')

# Legend for bars
for j, (name, col) in enumerate([('T:', C_TEMP), ('H:', C_H), ('W:', C_W)]):
    axF.add_patch(FancyBboxPatch((0.3+j*1.5, 5.3), 0.3, 0.3, boxstyle='square,pad=0',
                                  fc=col, ec='none', alpha=0.7))
    axF.text(0.68+j*1.5, 5.45, name, ha='left', va='center', fontsize=8, color=col)

# Zoom into alignment: V F1 and A t=5 share T=5
axF.plot([3.4, 3.4], [y_token, y_token-0.4], color='yellow', lw=1.5, ls='--')

# Note about audio T=0..4 not matching F0 T=0 issue
axF.text(6.0, 4.9,
    'Note on audio H=W=T:\n'
    'Audio H/W IDs track the Temporal ID (they\'re always identical).\n'
    'This means audio tokens only carry meaningful position in the\n'
    'Temporal axis — Height/Width RoPE components cancel out\n'
    'for any two audio tokens (same delta in all axes → 1D behaviour).',
    ha='center', va='top', fontsize=8.5, color=FG,
    bbox=dict(fc='#0D1A2A', ec=CA, lw=1.2, boxstyle='round,pad=0.4'))

# Timeline showing out-of-order sequence vs in-order time
axF.text(0.2, 4.15,
    'Sequence order (left→right in LLM):  V F0 → V F0 → V F1 → V F1 → A t=0 → A t=1 → ...',
    ha='left', fontsize=8.5, color=FG, family='monospace')
axF.text(0.2, 3.65,
    'Temporal ID order:                    0      0      5      5      0      1      2',
    ha='left', fontsize=8.5, color=C_TEMP, family='monospace')
axF.text(0.2, 3.15,
    '→ Sequence position ≠ Temporal position!  TMRoPE decouples the two.',
    ha='left', fontsize=9.5, color='yellow', fontweight='bold')

# Show why this is powerful
axF.add_patch(FancyBboxPatch((0.2, 0.2), 11.5, 2.6,
    boxstyle='round,pad=0.2', fc='#0A1020', ec=CLIN, lw=1.5))
axF.text(6.0, 2.55, 'Why This Matters', ha='center', fontsize=10, color=CLIN, fontweight='bold')
axF.text(0.5, 2.1,
    '• V F1 tokens (T=5) appear at sequence positions 3,4  but audio token A t=5 (T=5) is at position 10.',
    ha='left', fontsize=8.5, color=FG)
axF.text(0.5, 1.65,
    '• Standard 1D RoPE: these tokens are "7 apart" → weak attention.',
    ha='left', fontsize=8.5, color=CRED)
axF.text(0.5, 1.2,
    '• TMRoPE Temporal: both have T=5 → Temporal diff = 0 → STRONG temporal attention.',
    ha='left', fontsize=8.5, color=CT)
axF.text(0.5, 0.7,
    '• The LLM can now ask: "what was happening visually when I heard this sound?"',
    ha='left', fontsize=8.5, color=FG, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL G — Image vs Video position encoding
# ══════════════════════════════════════════════════════════════════════════════
axG = fig.add_subplot(gs[3, 0])
axG.set_xlim(0, 12); axG.set_ylim(0, 8.0)
axG.axis('off')
axG.set_title('⑦ Image vs Video Position Encoding — Temporal ID Behaviour',
              fontsize=11, color=CLIN, pad=8, fontweight='bold')

# Image: 3×3 patches, all same temporal ID
axG.text(2.0, 7.6, 'IMAGE  (treated as 2 identical frames)', ha='center',
         fontsize=10, color=CLIN, fontweight='bold')
grid_n = 3; cell = 1.1
for r in range(grid_n):
    for c in range(grid_n):
        bx = 0.3 + c*cell; by = 4.5 + (grid_n-1-r)*cell
        axG.add_patch(FancyBboxPatch((bx, by), cell*0.9, cell*0.9,
            boxstyle='round,pad=0.08', fc='#1A1030', ec=CLIN, lw=1.5, alpha=0.85))
        axG.text(bx+cell*0.45, by+cell*0.62, f'T=42', ha='center',
                 fontsize=7, color=C_TEMP, fontweight='bold')
        axG.text(bx+cell*0.45, by+cell*0.35, f'H={r} W={c}', ha='center',
                 fontsize=6.5, color=FG)
axG.text(1.95, 4.2, 'All patches share T=42 (constant for whole image)',
         ha='center', fontsize=8, color=C_TEMP)
axG.text(1.95, 3.8, '→ Only H,W axes vary within an image', ha='center', fontsize=8, color=CDIM)

# Video: 3 frames, each with different temporal IDs
axG.text(8.5, 7.6, 'VIDEO  (3 frames shown)', ha='center',
         fontsize=10, color=CV, fontweight='bold')
frame_t_ids = [0, 5, 10]
for fi, ftid in enumerate(frame_t_ids):
    bx_base = 5.5 + fi*2.2
    for r in range(2):
        for c in range(2):
            bx = bx_base + c*0.95; by = 5.3 + (1-r)*0.95
            alpha = 0.5 + fi*0.15
            axG.add_patch(FancyBboxPatch((bx, by), 0.85, 0.85,
                boxstyle='round,pad=0.05', fc='#2A1000', ec=CV, lw=1, alpha=alpha))
            axG.text(bx+0.42, by+0.55, f'T={ftid}', ha='center',
                     fontsize=6.5, color=C_TEMP, fontweight='bold')
            axG.text(bx+0.42, by+0.25, f'H={r}W={c}', ha='center',
                     fontsize=5.5, color=FG)
    axG.text(bx_base+0.9, 5.1, f'Frame {fi}\nt={ftid*40}ms', ha='center',
             fontsize=8, color=CV)

# Show that T increases per frame
for fi in range(len(frame_t_ids)-1):
    x1 = 5.5 + fi*2.2 + 1.9; x2 = 5.5 + (fi+1)*2.2
    axG.annotate('', xy=(x2, 6.2), xytext=(x1, 6.2),
                 arrowprops=dict(arrowstyle='->', color=C_TEMP, lw=1.5))
    axG.text((x1+x2)/2, 6.42, '+5', ha='center', fontsize=8,
             color=C_TEMP, fontweight='bold')

axG.text(8.5, 4.6,
    'Each frame: T increments by (frame_duration / 40ms)\n'
    'H,W still vary per patch within each frame',
    ha='center', fontsize=8.5, color=FG,
    bbox=dict(fc='#1A0A00', ec=CV, lw=1, boxstyle='round,pad=0.3'))

# Key difference
axG.add_patch(FancyBboxPatch((0.2, 0.2), 11.5, 3.3,
    boxstyle='round,pad=0.2', fc='#100C1A', ec=CLIN, lw=1.5))
axG.text(6.0, 3.25, 'Image vs Video: The T axis difference', ha='center',
         fontsize=10, color=CLIN, fontweight='bold')
for i, (label, text, color) in enumerate([
    ('Image', 'T = CONSTANT across all patches  (one snapshot in time)\n'
              'RoPE sees all patches as "at the same time" → attends freely across space',
              CLIN),
    ('Video', 'T = VARIES per frame  (different moments)\n'
              'RoPE creates temporal structure → patches from same frame attend more strongly\n'
              'to audio tokens from the same timestamp',
              CV),
]):
    axG.text(0.5, 2.7 - i*1.45, f'{label}:', ha='left', fontsize=9.5,
             color=color, fontweight='bold')
    axG.text(0.5, 2.35 - i*1.45, text, ha='left', fontsize=8.5, color=FG)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL H — Summary: why TMRoPE is better than alternatives
# ══════════════════════════════════════════════════════════════════════════════
axH = fig.add_subplot(gs[3, 1])
axH.set_xlim(0, 12); axH.set_ylim(0, 8.0)
axH.axis('off')
axH.set_title('⑧ TMRoPE vs Alternatives — Why This Design?',
              fontsize=11, color=CY, pad=8, fontweight='bold')

options = [
    ('1D RoPE\n(text-only)', CRED,
     '• Assigns sequential positions 0,1,2,...\n'
     '• Audio at pos 100, video at pos 5\n'
     '  look "95 apart" even if same moment\n'
     '• Breaks cross-modal temporal attention\n'
     '• Used for TEXT only (T=H=W collapses it)',
     False),
    ('Absolute learned\n2D pos embed', '#F0A000',
     '• Fixed resolution: breaks for variable-\n'
     '  resolution inputs\n'
     '• No generalization to unseen resolutions\n'
     '• No temporal axis at all\n'
     '• Cannot align audio with video',
     False),
    ('M-RoPE\n(Qwen2.5-VL)', '#88AAFF',
     '• Height + Width RoPE for images ✓\n'
     '• Variable resolution ✓\n'
     '• No temporal axis ✗\n'
     '• Cannot sync audio ✗\n'
     '• Images only, no video-audio alignment',
     False),
    ('TMRoPE\n(Qwen2.5-Omni)', CT,
     '• 3D: Temporal + Height + Width ✓\n'
     '• 1 Temporal ID = 40ms (audio grain) ✓\n'
     '• Variable resolution video ✓\n'
     '• Audio↔Video temporal sync ✓\n'
     '• Text degenerates to 1D RoPE ✓',
     True),
]

col_w = 2.8
for i, (title, color, text, best) in enumerate(options):
    bx = 0.3 + i*2.9
    ec_color = color; lw_val = 2.5 if best else 1.5
    fc_color = '#0A2010' if best else '#161B22'
    axH.add_patch(FancyBboxPatch((bx, 1.5), col_w, 6.1,
        boxstyle='round,pad=0.15', fc=fc_color, ec=ec_color, lw=lw_val, alpha=0.9))
    axH.text(bx+col_w/2, 7.3, title, ha='center', fontsize=9.5, color=color,
             fontweight='bold')
    if best:
        axH.text(bx+col_w/2, 7.0, '★ USED', ha='center', fontsize=8.5,
                 color=CT, fontweight='bold')
    for j, line in enumerate(text.split('\n')):
        fc2 = '#88FF88' if '✓' in line else '#FF8888' if '✗' in line else FG
        axH.text(bx+0.15, 6.55-j*0.55, line, ha='left', fontsize=7.8,
                 color=fc2, va='top')

axH.add_patch(FancyBboxPatch((0.2, 0.1), 11.5, 1.1,
    boxstyle='round,pad=0.2', fc='#0A1500', ec=CT, lw=1.5))
axH.text(6.0, 0.85,
    'TMRoPE costs nothing extra at inference — just different position IDs fed into '
    'standard RoPE.\n'
    'The 3-axis decomposition is applied inside the QK rotation, fully compatible with '
    'Flash Attention.',
    ha='center', fontsize=8.5, color=FG)

plt.savefig(r'C:\Users\Armaan\Desktop\PURS\viz_tmrope.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: viz_tmrope.png")
plt.close()
