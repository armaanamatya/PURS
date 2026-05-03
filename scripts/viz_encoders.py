"""
Qwen2.5-Omni: Audio & Vision Encoder Architecture + Lineage
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np
import warnings; warnings.filterwarnings('ignore')

BG   = '#0D1117'; FG = '#E6EDF3'
CA   = '#58A6FF'; CV = '#F78166'; CW = '#3FB950'
CLIN = '#BC8CFF'; CDIM = '#8B949E'

plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': '#161B22',
    'text.color': FG, 'axes.labelcolor': FG, 'xtick.color': FG,
    'ytick.color': FG, 'axes.edgecolor': '#30363D'})

fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.975, 'Qwen2.5-Omni — Encoder Architectures & Lineage',
         ha='center', va='top', fontsize=20, fontweight='bold', color='white')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.10, wspace=0.08,
                       left=0.03, right=0.97, top=0.95, bottom=0.02)

# ─────────────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sub='', fc='#1C2128', ec='white', lw=1.5,
        fontsize=9, subfontsize=7.5, alpha=0.9, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                                fc=fc, ec=ec, lw=lw, alpha=alpha,
                                transform=ax.transData, clip_on=False))
    fw = 'bold' if bold else 'normal'
    ax.text(x+w/2, y+h/2 + (0.15 if sub else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight=fw,
            color='white', transform=ax.transData)
    if sub:
        ax.text(x+w/2, y+h/2 - 0.22, sub, ha='center', va='center',
                fontsize=subfontsize, color=CDIM, transform=ax.transData)

def arrow(ax, x1, y1, x2, y2, color='white', lw=1.8, label='', fontsize=7.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my, label, fontsize=fontsize, color=color, va='center')

def section_title(ax, x, y, label, color):
    ax.text(x, y, label, ha='center', va='center', fontsize=13,
            fontweight='bold', color=color, transform=ax.transData,
            bbox=dict(fc=BG, ec=color, lw=1.5, boxstyle='round,pad=0.3'))

# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEFT: Audio Encoder Lineage
# ══════════════════════════════════════════════════════════════════════════════
ax_al = fig.add_subplot(gs[0, 0])
ax_al.set_xlim(0, 12); ax_al.set_ylim(0, 14)
ax_al.axis('off')
ax_al.set_title('Audio Encoder — Lineage', fontsize=13, color=CA,
                pad=8, fontweight='bold')

# Whisper column (leftmost)
section_title(ax_al, 2.5, 13.3, 'Whisper Large-v2  (OpenAI, 2022)', CW)
blocks_w = [
    ('Log-Mel Spectrogram', '80 mel bins, 25ms win / 10ms hop'),
    ('Conv1d  k=3 s=1', '80 → 1280 ch  + GELU'),
    ('Conv1d  k=3 s=2', '1280 → 1280 ch  + GELU  (2× downsample)'),
    ('Add Sinusoidal\nPos Enc', '1D fixed sinusoidal, max 1500 frames'),
    ('× 32  Encoder Blocks', 'MHA (20 heads, d=1280)\n+ LayerNorm + FFN (1280→5120→1280)'),
    ('Final LayerNorm', 'd=1280  →  encoder output'),
]
y = 12.0
for label, sub in blocks_w:
    box(ax_al, 0.3, y-0.55, 4.4, 0.85, label, sub, fc='#1A2F1A', ec=CW, fontsize=8.5)
    y -= 1.1
    if y > 6.5:
        arrow(ax_al, 2.5, y+0.55, 2.5, y+0.25, CW)

# Qwen2-Audio column (middle)
section_title(ax_al, 7.5, 13.3, 'Qwen2-Audio  (Alibaba, 2024)', CA)
blocks_qa = [
    ('Log-Mel Spectrogram', '128 mel bins  ← expanded from 80'),
    ('Conv1d  k=3 s=1', '128 → 1280 ch  + GELU'),
    ('Conv1d  k=3 s=2', '1280 → 1280 ch  + GELU'),
    ('Add Sinusoidal\nPos Enc', '1D fixed, adapted for 128-mel input'),
    ('× 32  Encoder Blocks', 'MHA (20 heads, d=1280)\n+ LayerNorm + FFN (1280→5120→1280)'),
    ('Final LayerNorm\n+ Linear Projection', 'd=1280  →  LLM hidden dim'),
]
y = 12.0
for label, sub in blocks_qa:
    box(ax_al, 5.3, y-0.55, 4.4, 0.85, label, sub, fc='#1A1F3A', ec=CA, fontsize=8.5)
    y -= 1.1
    if y > 6.5:
        arrow(ax_al, 7.5, y+0.55, 7.5, y+0.25, CA)

# Arrow: Whisper → Qwen2-Audio
ax_al.annotate('', xy=(5.2, 9.8), xytext=(4.8, 9.8),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
ax_al.text(5.05, 10.05, 'adapted\n+128mel', ha='center', fontsize=7,
           color='yellow', fontweight='bold')

# Bottom: Qwen2.5-Omni modification
ax_al.text(5.5, 5.7, 'Qwen2.5-Omni Modification', ha='center', fontsize=10,
           color='white', fontweight='bold',
           bbox=dict(fc='#2D1B3D', ec=CLIN, lw=1.5, boxstyle='round,pad=0.3'))
arrow(ax_al, 7.5, 6.3, 7.5, 6.05, CLIN)

mod_text = (
    "Block-wise Attention  (new in Qwen2.5-Omni)\n"
    "─────────────────────────────────────────────────\n"
    "• Full self-attention  →  2-second windowed blocks\n"
    "• Each block: T_block = 2s / 10ms = 200 mel frames\n"
    "  → 200 / 2 (stride) = 100 encoder frames → 50 tokens\n"
    "• Tokens within a block attend freely; NO cross-block attn\n"
    "• Enables chunked prefill for streaming inference\n"
    "• 40 ms per token  (fundamental temporal atom for TMRoPE)"
)
ax_al.text(0.5, 5.4, mod_text, ha='left', va='top', fontsize=8.5,
           color=FG, family='monospace',
           bbox=dict(fc='#1C1030', ec=CLIN, lw=1.5, boxstyle='round,pad=0.4'))

# ══════════════════════════════════════════════════════════════════════════════
# TOP-RIGHT: Vision Encoder Lineage
# ══════════════════════════════════════════════════════════════════════════════
ax_vl = fig.add_subplot(gs[0, 1])
ax_vl.set_xlim(0, 12); ax_vl.set_ylim(0, 14)
ax_vl.axis('off')
ax_vl.set_title('Vision Encoder — Lineage', fontsize=13, color=CV,
                pad=8, fontweight='bold')

# ViT-H/14 (left column)
section_title(ax_vl, 2.5, 13.3, 'ViT-H/14  (Dosovitskiy 2020 + scaling)', '#F0A000')
blocks_vit = [
    ('Image Input', 'H × W × 3  (any resolution)'),
    ('Patch Embed\nConv2d  14×14', '→ (H/14) × (W/14) tokens\n  each token ∈ R^1280'),
    ('+ 2D Learned\nPos Embed', 'Absolute 2D pos for each patch'),
    ('× 32  ViT Blocks', 'MHA (16 heads, d=1280)\n+ LayerNorm + FFN\n≈ 632M total params'),
    ('CLS token /\nPool', 'Global image representation'),
]
y = 12.2
for label, sub in blocks_vit:
    box(ax_vl, 0.3, y-0.6, 4.4, 0.95, label, sub, fc='#2A1A00', ec='#F0A000', fontsize=8.5)
    y -= 1.15
    if y > 7.8:
        arrow(ax_vl, 2.5, y+0.6, 2.5, y+0.28, '#F0A000')

# Qwen2.5-VL (right column)
section_title(ax_vl, 7.5, 13.3, 'Qwen2.5-VL ViT  (Alibaba, 2025)', CV)
blocks_qvl = [
    ('Image / Video Frame', 'Any resolution, packed into sequences'),
    ('Patch Embed  14×14', '→ (H/14)×(W/14) tokens  ∈ R^1280'),
    ('M-RoPE 2D\n(no abs pos embed)', 'Height + Width RoPE axes replace\nlearned 2D pos embeddings'),
    ('× 32  ViT Blocks', 'MHA (16 heads, d=1280)\n+ RMSNorm + SwiGLU FFN\n≈ 675M total params'),
    ('2×2 MLP\nToken Merge', '[t_i, t_{i+1}, t_{j}, t_{j+1}]  →  MLP  →  1 token\n4 × 1280 = 5120 → 1280  (4:1 compression)'),
    ('Linear Projection', '1280  →  LLM hidden dim  (e.g. 3584)'),
]
y = 12.2
for label, sub in blocks_qvl:
    box(ax_vl, 5.3, y-0.6, 4.4, 0.95, label, sub, fc='#1A0F2A', ec=CV, fontsize=8.5)
    y -= 1.15
    if y > 5.4:
        arrow(ax_vl, 7.5, y+0.6, 7.5, y+0.28, CV)

# Arrow: ViT → Qwen2.5-VL
ax_vl.annotate('', xy=(5.2, 10.5), xytext=(4.8, 10.5),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
ax_vl.text(5.0, 10.75, 'M-RoPE\n+SwiGLU', ha='center', fontsize=7, color='yellow', fontweight='bold')

# Bottom modification box
ax_vl.text(5.5, 5.4, 'Qwen2.5-Omni Modifications', ha='center', fontsize=10,
           color='white', fontweight='bold',
           bbox=dict(fc='#2D1B3D', ec=CLIN, lw=1.5, boxstyle='round,pad=0.3'))
arrow(ax_vl, 7.5, 5.7, 7.5, 5.5, CLIN)
mod_text_v = (
    "TMRoPE  (replaces M-RoPE for video+audio sync)\n"
    "─────────────────────────────────────────────────────\n"
    "• 3D decomposition: Temporal + Height + Width axes\n"
    "• Video temporal IDs increment with frame timestamps\n"
    "  (1 temporal ID = 40 ms, matching audio granularity)\n"
    "• Dynamic FPS sampling → frames snapped to 40ms grid\n"
    "• Images = two identical frames (uniform processing)\n"
    "• 2×2 MLP merge  retained unchanged  from Qwen2.5-VL"
)
ax_vl.text(0.5, 5.1, mod_text_v, ha='left', va='top', fontsize=8.5,
           color=FG, family='monospace',
           bbox=dict(fc='#1C1030', ec=CLIN, lw=1.5, boxstyle='round,pad=0.4'))

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM-LEFT: Audio Encoder Internal Detail
# ══════════════════════════════════════════════════════════════════════════════
ax_ad = fig.add_subplot(gs[1, 0])
ax_ad.set_xlim(0, 12); ax_ad.set_ylim(0, 11)
ax_ad.axis('off')
ax_ad.set_title('Audio Encoder — Internal Block Detail (Qwen2-Audio / Whisper-style)',
                fontsize=11, color=CA, pad=6, fontweight='bold')

# Draw one transformer encoder block detail
enc_layers = [
    ('x  [T × 1280]', '#333', 'Input sequence'),
    ('LayerNorm', '#1A3A2A', ''),
    ('Multi-Head Attention\n20 heads, d_head=64,  d=1280', '#1A2A3A',
     'Q=K=V=x·W  (self-attn)\nBlock-wise mask: attend only within 2s window'),
    ('+ Residual', '#2A2A2A', ''),
    ('LayerNorm', '#1A3A2A', ''),
    ('FFN: Linear(1280→5120)\n+ GELU + Linear(5120→1280)', '#2A1A3A', ''),
    ('+ Residual', '#2A2A2A', ''),
    ('x\'  [T × 1280]', '#333', 'Output (same shape)'),
]
y = 10.2
x_left = 1.0
for i, item in enumerate(enc_layers):
    if len(item) == 3:
        label, fc, note = item
    else:
        label, fc = item; note = ''
    h = 0.75 if '\n' in label else 0.55
    box(ax_ad, x_left, y-h, 5.5, h, label, note, fc=fc, ec=CA, fontsize=8.5, subfontsize=7)
    if i < len(enc_layers)-1:
        arrow(ax_ad, x_left+2.75, y-h, x_left+2.75, y-h-0.1, CA)
    y -= h + 0.15

# Block-wise attention diagram on right
ax_ad.text(8.0, 10.4, 'Block-wise Attention Mask', ha='center', fontsize=10,
           color=CA, fontweight='bold')
N = 8
cell = 0.7
ox, oy = 6.4, 3.5
for i in range(N):
    for j in range(N):
        blk_i, blk_j = i // 4, j // 4
        in_block = (blk_i == blk_j)
        color = '#2060A0' if in_block else '#1A1A2A'
        ec_c = CA if in_block else '#333355'
        ax_ad.add_patch(FancyBboxPatch((ox+j*cell, oy+i*cell), cell*0.88, cell*0.88,
                                       boxstyle='square,pad=0.0', fc=color, ec=ec_c, lw=0.5))
ax_ad.text(ox + N*cell/2, oy + N*cell + 0.3, 'Attention matrix  [T×T]',
           ha='center', fontsize=8, color=FG)
ax_ad.text(ox - 0.5, oy + N*cell/2, 'Q\n(k\ney\n)', ha='center', va='center',
           fontsize=8, color=FG, rotation=90)
ax_ad.text(ox + N*cell/2, oy - 0.35, 'K (query)', ha='center', fontsize=8, color=FG)
ax_ad.add_patch(FancyBboxPatch((ox+0.05, oy+4*cell+0.05), 4*cell-0.1, 4*cell-0.1,
                               boxstyle='square,pad=0', fc='none', ec='yellow', lw=2))
ax_ad.add_patch(FancyBboxPatch((ox+4*cell+0.05, oy+0.05), 4*cell-0.1, 4*cell-0.1,
                               boxstyle='square,pad=0', fc='none', ec='yellow', lw=2))
ax_ad.text(ox+2*cell, oy+6.5*cell, 'Block 0\n(0–2s)', ha='center', fontsize=7.5, color='yellow')
ax_ad.text(ox+6*cell, oy+2.5*cell, 'Block 1\n(2–4s)', ha='center', fontsize=7.5, color='yellow')
ax_ad.text(ox+6*cell, oy+6.5*cell, '✗ masked', ha='center', fontsize=7, color='#666688')
ax_ad.text(ox+2*cell, oy+2.5*cell, '✗ masked', ha='center', fontsize=7, color='#666688')

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM-RIGHT: Vision Encoder Internal Detail
# ══════════════════════════════════════════════════════════════════════════════
ax_vd = fig.add_subplot(gs[1, 1])
ax_vd.set_xlim(0, 12); ax_vd.set_ylim(0, 11)
ax_vd.axis('off')
ax_vd.set_title('Vision Encoder — Internal Block Detail (Qwen2.5-VL ViT)',
                fontsize=11, color=CV, pad=6, fontweight='bold')

vit_layers = [
    ('x  [N_patches × 1280]', '#333', 'N = (H/14)×(W/14) per frame'),
    ('RMSNorm', '#1A3A2A', ''),
    ('Multi-Head Attention\n16 heads, d_head=80,  d=1280', '#1A2A3A',
     'Q,K,V = x·W  +  2D M-RoPE (H+W axes)\nFlash Attention for efficiency'),
    ('+ Residual', '#2A2A2A', ''),
    ('RMSNorm', '#1A3A2A', ''),
    ('SwiGLU FFN\nLinear(1280→3413×2) + SiLU gate\n→ Linear(3413→1280)', '#2A1A3A', ''),
    ('+ Residual', '#2A2A2A', ''),
    ('x\'  [N_patches × 1280]', '#333', 'Repeat ×32 blocks'),
]
y = 10.2
for i, item in enumerate(vit_layers):
    label, fc = item[0], item[1]
    note = item[2] if len(item) > 2 else ''
    h = 0.75 if '\n' in label else 0.55
    box(ax_vd, 0.5, y-h, 5.5, h, label, note, fc=fc, ec=CV, fontsize=8.5, subfontsize=7)
    if i < len(vit_layers)-1:
        arrow(ax_vd, 3.25, y-h, 3.25, y-h-0.1, CV)
    y -= h + 0.15

# 2×2 MLP Token Merge diagram
ax_vd.text(9.2, 10.4, '2×2 MLP Token Merge', ha='center', fontsize=10,
           color=CV, fontweight='bold')

# Draw 4 patches → concat → MLP → 1 token
patch_colors = ['#3A1060', '#1A3060', '#1A6030', '#603010']
patch_labels = ['t_{r,c}', 't_{r,c+1}', 't_{r+1,c}', 't_{r+1,c+1}']
positions = [(6.9, 7.8), (8.2, 7.8), (6.9, 6.5), (8.2, 6.5)]
for (px, py), pc, pl in zip(positions, patch_colors, patch_labels):
    ax_vd.add_patch(FancyBboxPatch((px, py), 1.1, 1.0,
                                   boxstyle='round,pad=0.08', fc=pc, ec='white', lw=1.2))
    ax_vd.text(px+0.55, py+0.5, pl, ha='center', va='center',
               fontsize=7.5, color='white', fontweight='bold')
    ax_vd.text(px+0.55, py+0.2, '1280-d', ha='center', va='center',
               fontsize=6.5, color=CDIM)

# concat arrow
ax_vd.annotate('', xy=(9.7, 7.15), xytext=(8.45, 7.15),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
ax_vd.text(9.0, 7.42, 'concat', ha='center', fontsize=7, color=FG)

# MLP box
box(ax_vd, 9.8, 6.6, 1.8, 1.1, 'MLP', 'Linear\n5120→1280', fc='#302060', ec=CLIN, fontsize=9)
ax_vd.annotate('', xy=(11.0, 7.15), xytext=(11.6, 7.15),
               arrowprops=dict(arrowstyle='->', color=CLIN, lw=1.5))
box(ax_vd, 11.65, 6.6, 1.2, 1.1, 'merged\ntoken', '1280-d\n(4:1)', fc='#3A0060', ec=CV, fontsize=8)

# shape annotation
ax_vd.text(8.5, 5.9,
           'Input:  4 × 1280 = 5120-dim concatenation\n'
           'Output: 1 × 1280  →  4× fewer video tokens\n'
           'Applied AFTER all 32 ViT blocks',
           ha='left', va='top', fontsize=8.5, color=FG, family='monospace',
           bbox=dict(fc='#1A1030', ec=CV, lw=1.2, boxstyle='round,pad=0.35'))

# MRoPE axes diagram
ax_vd.text(7.5, 5.1, 'M-RoPE / TMRoPE Axes per Token', ha='center',
           fontsize=9, color=CLIN, fontweight='bold')
axes_data = [('Temporal', '#804000', 'video: frame timestamp\nimage: constant\naudio: 40ms steps'),
             ('Height', '#004080', 'patch row index\n(within frame)'),
             ('Width', '#008040', 'patch col index\n(within frame)')]
for i, (name, fc, desc) in enumerate(axes_data):
    bx = 6.4 + i*1.9
    ax_vd.add_patch(FancyBboxPatch((bx, 3.5), 1.7, 1.3,
                                   boxstyle='round,pad=0.1', fc=fc, ec='white', lw=1.2, alpha=0.8))
    ax_vd.text(bx+0.85, 4.75, name, ha='center', va='center',
               fontsize=8.5, fontweight='bold', color='white')
    ax_vd.text(bx+0.85, 3.9, desc, ha='center', va='center',
               fontsize=6.5, color='#cccccc')
ax_vd.text(8.5, 3.25, '→ RoPE applied independently per axis; combined via concatenation in frequency space',
           ha='center', fontsize=8, color=CDIM)

plt.savefig(r'C:\Users\Armaan\Desktop\PURS\viz_encoders_arch.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: viz_encoders_arch.png")
plt.close()
