"""
OmniZip — Full Mathematical Pipeline Visualization
Every stage annotated with exact code operations from omnizip_units.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)

BG = '#0D1117'; FG = '#E6EDF3'
CA = '#58A6FF'; CV = '#F78166'; CT = '#3FB950'
CLIN = '#BC8CFF'; CDIM = '#8B949E'; CY = '#E3B341'
CRED = '#FF7B72'

plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': '#161B22',
    'text.color': FG, 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG, 'axes.edgecolor': '#30363D'})

fig = plt.figure(figsize=(26, 32))
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.987, 'OmniZip — Complete Mathematical Pipeline',
         ha='center', fontsize=22, fontweight='bold', color='white')
fig.text(0.5, 0.981, 'Training-free audio-guided token compression for Qwen2.5-Omni  |  omnizip_units.py',
         ha='center', fontsize=11, color=CDIM)

gs = gridspec.GridSpec(6, 1, figure=fig, hspace=0.55,
                       left=0.04, right=0.96, top=0.978, bottom=0.01)

# ─────────────────────────────────────────────────────────────────────────────
def cbox(ax, x, y, w, h, title, body='', title_color='white',
         fc='#1C2128', ec='white', lw=1.5, fontsize=9, bodyfontsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                                fc=fc, ec=ec, lw=lw, alpha=0.92,
                                transform=ax.transData, clip_on=False))
    ty = y + h - 0.15 if body else y + h/2
    va = 'top' if body else 'center'
    ax.text(x+w/2, ty, title, ha='center', va=va, fontsize=fontsize,
            fontweight='bold', color=title_color, transform=ax.transData)
    if body:
        ax.text(x+w/2, y+0.12, body, ha='center', va='bottom',
                fontsize=bodyfontsize, color=CDIM, transform=ax.transData,
                family='monospace')

def arr(ax, x1, y1, x2, y2, color='white', lw=2, label='', lfs=8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    if label:
        ax.text((x1+x2)/2+0.05, (y1+y2)/2, label, fontsize=lfs,
                color=color, va='center')

def code_box(ax, x, y, w, h, text, ec=CLIN):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                                fc='#1A1030', ec=ec, lw=1.2, alpha=0.95,
                                transform=ax.transData, clip_on=False))
    ax.text(x+0.1, y+h-0.07, text, ha='left', va='top', fontsize=7.5,
            color='#A8D8A8', family='monospace', transform=ax.transData)

def eq_box(ax, x, y, w, h, text, ec=CY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                                fc='#1A1500', ec=ec, lw=1.5, alpha=0.95,
                                transform=ax.transData, clip_on=False))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=9,
            color=CY, transform=ax.transData)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 0 — Input & Token Extraction
# ══════════════════════════════════════════════════════════════════════════════
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(0, 26); ax0.set_ylim(0, 4.5)
ax0.axis('off')
ax0.set_title('Stage 0 — Input Embeddings & Token Extraction  [omnizip(), lines 142–175]',
              fontsize=12, color=CT, pad=6, fontweight='bold', loc='left')

# Draw input sequence
seq_labels = ['txt','txt','txt','vid','vid','vid','vid','vid','vid','aud','aud','aud','aud','aud','aud','txt']
seq_colors = [CT]*3 + [CV]*6 + [CA]*6 + [CT]*1
sx = 1.0; sy = 2.5; sw = 1.3; sh = 0.9
for i, (sl, sc) in enumerate(zip(seq_labels, seq_colors)):
    ax0.add_patch(FancyBboxPatch((sx+i*1.4, sy), sw, sh,
                                 boxstyle='round,pad=0.08', fc=sc, ec='white',
                                 lw=0.8, alpha=0.75))
    ax0.text(sx+i*1.4+sw/2, sy+sh/2, sl, ha='center', va='center',
             fontsize=8, fontweight='bold', color='white')
ax0.text(sx + len(seq_labels)*1.4/2, sy+sh+0.25,
         'flat_embeds  [L × D]  with input_ids', ha='center', fontsize=9, color=FG)

# Extraction arrows
ax0.annotate('', xy=(7.3, sy), xytext=(7.3, sy-0.7),
             arrowprops=dict(arrowstyle='->', color=CV, lw=2))
ax0.text(7.3, sy-1.0, 'video_indices = nonzero(ids==video_id)\nvideo_feature = embeds[video_indices]  [Nv × D]',
         ha='center', fontsize=8, color=CV, family='monospace')

ax0.annotate('', xy=(13.3, sy), xytext=(13.3, sy-0.7),
             arrowprops=dict(arrowstyle='->', color=CA, lw=2))
ax0.text(13.3, sy-1.0, 'audio_indices = nonzero(ids==audio_id)\naudio_feature = embeds[audio_indices]  [Na × D]',
         ha='center', fontsize=8, color=CA, family='monospace')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Audio Selection (omnizip_audio_attn)
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[1])
ax1.set_xlim(0, 26); ax1.set_ylim(0, 6.5)
ax1.axis('off')
ax1.set_title('Stage 1 — Audio Token Selection  [omnizip_audio_attn(), lines 4–88]',
              fontsize=12, color=CA, pad=6, fontweight='bold', loc='left')

# Step 1: Attention importance
cbox(ax1, 0.3, 3.5, 5.5, 2.5, 'Step 1a: Attention Importance',
     fc='#0D2140', ec=CA, title_color=CA, fontsize=10)
eq_box(ax1, 0.5, 4.1, 5.1, 0.8,
       'importance = A.mean(dim=0).sum(dim=0)\nA ∈ R^{H × T × T}  →  [T]  per-token score')
ax1.text(0.5, 3.6, 'attn_logits [H,T,T]  →  avg over heads  →  sum incoming attn per token\n'
                    '"How much does each token get attended to?"',
         ha='left', fontsize=8, color=CDIM, family='monospace')

# Step 2: TopK dominant selection
arr(ax1, 6.0, 4.75, 7.3, 4.75, CA)
cbox(ax1, 7.3, 3.5, 4.5, 2.5, 'Step 1b: TopK Dominant',
     fc='#0D2140', ec=CA, title_color=CA, fontsize=10)
eq_box(ax1, 7.5, 4.1, 4.1, 0.8,
       'dominant_num = round((1−ρ_a) × N)\n_, topk = topk(importance, dominant_num)')
ax1.text(7.5, 3.6, 'keep_mask[topk] = True\nkeep top (1−ρ_a) fraction', ha='left',
         fontsize=8, color=CDIM, family='monospace')

# Step 3: Contextual anchors
arr(ax1, 11.9, 4.75, 13.0, 4.75, CA)
cbox(ax1, 13.0, 3.5, 5.2, 2.5, 'Step 1c: Contextual Anchors',
     fc='#0D2140', ec=CA, title_color=CA, fontsize=10)
ax1.text(13.2, 5.65, 'remaining = tokens NOT in keep_mask\n'
                      'contextual_num = round(0.03 × N)\n'
                      'anchors = evenly spaced from remaining\n'
                      'keep_mask[anchors] = True',
         ha='left', va='top', fontsize=8, color=CDIM, family='monospace')

# Step 4: Cross-modal merge assignment
arr(ax1, 18.3, 4.75, 19.5, 4.75, CA)
cbox(ax1, 19.5, 3.5, 6.0, 2.5, 'Step 1d: Cross-Modal Merge',
     fc='#0D2140', ec=CA, title_color=CA, fontsize=10)
eq_box(ax1, 19.7, 4.6, 5.6, 0.7,
       'â = a / ‖a‖,  v̂ = v / ‖v‖  (L2 norm, ε=1e-6)')
eq_box(ax1, 19.7, 3.85, 5.6, 0.7,
       'S_av = â[pool] @ v̂ᵀ  →  scores = S_av.max(dim=1)')
ax1.text(19.7, 3.6, 'Top-g scoring: which pooled tokens\nare most similar to ANY video token?',
         ha='left', fontsize=8, color=CDIM, family='monospace')

# Merge formula
cbox(ax1, 0.3, 0.3, 25.3, 2.8, '', fc='#0A1520', ec='#2A4A6A', lw=1)
ax1.text(0.5, 2.8, 'Step 1e: Weighted Merge (lines 195–208)', fontsize=10,
         fontweight='bold', color=CA)
eq_box(ax1, 0.5, 2.0, 7.5, 0.65,
       'scores_i = (â[merge_rel] @ v̂ᵀ).max(dim=1).values')
arr(ax1, 8.1, 2.3, 9.0, 2.3, 'white', lw=1.5)
eq_box(ax1, 9.1, 2.0, 4.5, 0.65,
       'w = softmax(scores_i)')
arr(ax1, 13.7, 2.3, 14.6, 2.3, 'white', lw=1.5)
eq_box(ax1, 14.7, 2.0, 10.5, 0.65,
       "new_anchor = (anchor + Σᵢ(audio[i]·wᵢ)) / (1 + Σwᵢ)  ← weighted avg")
ax1.text(0.5, 1.6,
    'Pruned audio tokens are NOT discarded — they are MERGED into nearby anchors,\n'
    'with weights based on cross-modal similarity to video. The anchor embedding absorbs their content.',
    ha='left', fontsize=8.5, color=FG)
ax1.text(0.5, 0.55,
    'Output: keep_mask [Na bool]  +  updated flat_embeds (anchor tokens modified in-place)',
    ha='left', fontsize=9, color=CA, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Audio Group Retention → Video Pruning Rate
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[2])
ax2.set_xlim(0, 26); ax2.set_ylim(0, 6.0)
ax2.axis('off')
ax2.set_title('Stage 2 — Audio Retention → Video Pruning Rate  [omnizip(), lines 214–265]',
              fontsize=12, color=CY, pad=6, fontweight='bold', loc='left')

cbox(ax2, 0.3, 3.2, 7.5, 2.5, 'Step 2a: Per-Group Audio Retention',
     fc='#1A1400', ec=CY, title_color=CY, fontsize=10)
ax2.text(0.5, 5.35, 'Group video+audio tokens into windows (4 frames = 1 group)\n'
                     'VIDEO_GROUP_SIZE = tokens_per_frame × 4\n'
                     'AUDIO_GROUP_SIZE = 50  (matches 2s block)',
         ha='left', va='top', fontsize=8.5, color=CDIM, family='monospace')
eq_box(ax2, 0.5, 3.3, 7.0, 0.75,
       'Sa(i) = audio_mask[group_i].float().mean()  ∈ [0, 1]')

arr(ax2, 7.9, 4.45, 9.5, 4.45, CY, label='per group', lfs=8)

cbox(ax2, 9.5, 3.2, 8.5, 2.5, 'Step 2b: Inverse Mapping to Video Rate',
     fc='#1A1400', ec=CY, title_color=CY, fontsize=10)
eq_box(ax2, 9.7, 4.05, 8.1, 0.85,
       'ρ_v(i) = ρ_max + (ρ_min − ρ_max) · Sa(i)\n      ρ_min=0.35,  ρ_max=0.75')
ax2.text(9.7, 3.3, 'High audio retention Sa→1  ⟹  ρ_v→0.35  (gentle video prune)\n'
                    'Low audio retention  Sa→0  ⟹  ρ_v→0.75  (aggressive video prune)',
         ha='left', fontsize=8, color=CDIM, family='monospace')

arr(ax2, 18.1, 4.45, 19.5, 4.45, CY)
cbox(ax2, 19.5, 3.2, 6.0, 2.5, 'Step 2c: Budget Normalization',
     fc='#1A1400', ec=CY, title_color=CY, fontsize=10)
ax2.text(19.7, 5.35, 'Fix min/max group ratios.\nScale remaining groups so:\n'
                      'mean(ρ_v) = global merging_ratio_v\n'
                      'Clamp each ρ_v ∈ [0.35, 0.75]',
         ha='left', va='top', fontsize=8.5, color=CDIM, family='monospace')

# Plot: Sa vs ρ_v
ax2_inner = ax2.inset_axes([0.02, 0.02, 0.35, 0.48],
                            transform=ax2.transAxes)
ax2_inner.set_facecolor('#0D1117')
sa = np.linspace(0, 1, 100)
rho_v = 0.75 + (0.35 - 0.75) * sa
ax2_inner.plot(sa, rho_v, color=CY, lw=2.5)
ax2_inner.axhline(0.75, color='red', lw=1, ls='--', alpha=0.5)
ax2_inner.axhline(0.35, color='green', lw=1, ls='--', alpha=0.5)
ax2_inner.set_xlabel('Sa(i): audio retention', fontsize=8, color=FG)
ax2_inner.set_ylabel('ρ_v(i): video prune rate', fontsize=8, color=FG)
ax2_inner.set_title('Inverse relationship', fontsize=8, color=CY)
ax2_inner.text(0.1, 0.72, 'ρ_max=0.75', fontsize=7, color='red', transform=ax2_inner.transAxes)
ax2_inner.text(0.1, 0.12, 'ρ_min=0.35', fontsize=7, color='green', transform=ax2_inner.transAxes)
ax2_inner.tick_params(labelsize=7, colors=FG)
for sp in ax2_inner.spines.values(): sp.set_edgecolor('#444466')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — ISTM Video Pruning (Spatial + Temporal)
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[3])
ax3.set_xlim(0, 26); ax3.set_ylim(0, 7.5)
ax3.axis('off')
ax3.set_title('Stage 3 — ISTM Video Token Pruning  [omnizip_istm(), lines 90–140]',
              fontsize=12, color=CV, pad=6, fontweight='bold', loc='left')

# 4 frames
frame_x = [0.3, 6.5, 12.7, 19.0]
frame_labels = ['Frame 0\n(even)', 'Frame 1\n(odd)', 'Frame 2\n(even)', 'Frame 3\n(odd)']
frame_types  = ['DPC-KNN\nSpatial', 'Cosine\nTemporal', 'DPC-KNN\nSpatial', 'Cosine\nTemporal']
frame_colors = [CLIN, CV, CLIN, CV]
for i, (fx, fl, ft, fc) in enumerate(zip(frame_x, frame_labels, frame_types, frame_colors)):
    # Frame box
    ax3.add_patch(FancyBboxPatch((fx, 5.5), 5.8, 1.7,
                                 boxstyle='round,pad=0.12', fc='#1A1130', ec=fc, lw=2))
    ax3.text(fx+2.9, 6.8, fl, ha='center', va='center', fontsize=9,
             fontweight='bold', color=FG)
    ax3.text(fx+2.9, 6.1, ft, ha='center', va='center', fontsize=9,
             color=fc, fontweight='bold')
    ax3.text(fx+2.9, 5.6, f'tokens: start={i}×N_per_frame,  end={i+1}×N_per_frame',
             ha='center', va='bottom', fontsize=7, color=CDIM)
    # ratio label
    ax3.text(fx+2.9, 5.3, f'ratio_id = {"0" if i<2 else "1"}  →  merging_ratio[{"0" if i<2 else "1"}]',
             ha='center', fontsize=7.5, color=CY)

# Connecting arrow pairs (odd frames look at prev)
for i in [1, 3]:
    fx_prev = frame_x[i-1]; fx_curr = frame_x[i]
    ax3.annotate('', xy=(fx_curr+0.3, 6.35), xytext=(fx_prev+5.5, 6.35),
                 arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5,
                                 connectionstyle='arc3,rad=-0.3'))
    ax3.text((fx_prev+5.5+fx_curr+0.3)/2, 7.2, 'compare\nprev frame', ha='center',
             fontsize=7, color='yellow')

# DPC-KNN math
cbox(ax3, 0.3, 0.2, 11.5, 4.8, '', fc='#100C20', ec=CLIN, lw=1.5)
ax3.text(0.5, 4.8, 'DPC-KNN Spatial Pruning  (even frames t%2==0)', fontsize=10,
         fontweight='bold', color=CLIN)
eq_box(ax3, 0.5, 4.05, 11.0, 0.65,
       'normed = F.normalize(tokens, dim=1)  →  normed ∈ R^{N × D}')
arr(ax3, 6.0, 3.8, 6.0, 3.55, CLIN, lw=1.5)
eq_box(ax3, 0.5, 3.2, 11.0, 0.65,
       'sim = normed @ normedᵀ  →  [N × N]  pairwise cosine similarity')
arr(ax3, 6.0, 2.95, 6.0, 2.7, CLIN, lw=1.5)
eq_box(ax3, 0.5, 2.35, 11.0, 0.65,
       'knn_dist = topk(sim, k=5, dim=1).mean(dim=1)  →  [N]  avg similarity to 5 nearest')
arr(ax3, 6.0, 2.1, 6.0, 1.85, CLIN, lw=1.5)
eq_box(ax3, 0.5, 1.5, 11.0, 0.65,
       'keep_idx = topk(−knn_dist, num_keep)  →  high knn_dist = outliers = PRUNE')
ax3.text(0.5, 1.3, 'Keep tokens with MOST similar neighbors (high local density = representative)',
         ha='left', fontsize=8.5, color=FG)
ax3.text(0.5, 0.85, 'Intuition: tokens in crowded regions are redundant with their neighbors.\n'
                     'Keep the ones that are most "central" to a patch cluster.',
         ha='left', fontsize=8, color=CDIM)

# Cosine temporal math
cbox(ax3, 12.7, 0.2, 12.8, 4.8, '', fc='#101520', ec=CV, lw=1.5)
ax3.text(12.9, 4.8, 'Cosine Temporal Pruning  (odd frames t%2==1)', fontsize=10,
         fontweight='bold', color=CV)
eq_box(ax3, 12.9, 4.05, 12.4, 0.65,
       'prev_norm = F.normalize(prev_tokens, p=2, dim=1)  [N × D]')
arr(ax3, 19.1, 3.8, 19.1, 3.55, CV, lw=1.5)
eq_box(ax3, 12.9, 3.2, 12.4, 0.65,
       'curr_norm = F.normalize(curr_tokens, p=2, dim=1)  [N × D]')
arr(ax3, 19.1, 2.95, 19.1, 2.7, CV, lw=1.5)
eq_box(ax3, 12.9, 2.35, 12.4, 0.65,
       'sim_i = cosine_similarity(curr_norm[i], prev_norm[i], dim=1)  →  [N]')
arr(ax3, 19.1, 2.1, 19.1, 1.85, CV, lw=1.5)
eq_box(ax3, 12.9, 1.5, 12.4, 0.65,
       'keep_idx = sim.topk(num_keep, largest=False).indices  ← LOWEST similarity!')
ax3.text(12.9, 1.3, 'Keep tokens most DIFFERENT from previous frame (= novel info)',
         ha='left', fontsize=8.5, color=FG)
ax3.text(12.9, 0.85, 'Intuition: if a patch looks the same as last frame, it carries no new info.\n'
                      'Keep patches that CHANGED the most between frames.',
         ha='left', fontsize=8, color=CDIM)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — Cross-Modal Similarity Deep Dive
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[4])
ax4.set_xlim(0, 26); ax4.set_ylim(0, 6.5)
ax4.axis('off')
ax4.set_title('Stage 4 — Cross-Modal Similarity: Where & How  [omnizip_audio_attn(), lines 55–86]',
              fontsize=12, color=CRED, pad=6, fontweight='bold', loc='left')

# Visual: audio token × video token similarity matrix
Na_show, Nv_show = 8, 6
cell = 0.55
ox, oy = 0.5, 1.0
sim_mat = np.random.rand(Na_show, Nv_show) * 0.6 + 0.1
sim_mat[2, 3] = 0.95; sim_mat[5, 1] = 0.89; sim_mat[1, 4] = 0.82
cmap = plt.cm.YlOrRd
for i in range(Na_show):
    for j in range(Nv_show):
        color = cmap(sim_mat[i, j])
        ax4.add_patch(FancyBboxPatch((ox+j*cell, oy+i*cell), cell*0.92, cell*0.92,
                                     boxstyle='square,pad=0', fc=color, ec='none', alpha=0.9))
        if sim_mat[i, j] > 0.8:
            ax4.text(ox+j*cell+cell/2, oy+i*cell+cell/2, f'{sim_mat[i,j]:.2f}',
                     ha='center', va='center', fontsize=6.5, color='black', fontweight='bold')

ax4.text(ox + Nv_show*cell/2, oy + Na_show*cell + 0.25,
         'S_av = â_audio @ v̂_videoᵀ\n[Na × Nv]  cosine similarity',
         ha='center', fontsize=9, color=CRED, fontweight='bold')
ax4.text(ox - 0.55, oy + Na_show*cell/2, 'audio\ntokens\n(pool)', ha='center',
         va='center', fontsize=8, color=CA, rotation=90)
ax4.text(ox + Nv_show*cell/2, oy - 0.35, 'video tokens', ha='center',
         fontsize=8, color=CV)

# Max per row
ax4.text(ox + Nv_show*cell + 0.2, oy + Na_show*cell/2,
         '→ max(dim=1)\n= score per\naudio token',
         ha='left', va='center', fontsize=8, color=CRED)
for i in range(Na_show):
    max_j = np.argmax(sim_mat[i])
    ax4.add_patch(FancyBboxPatch((ox+max_j*cell+0.02, oy+i*cell+0.02), cell*0.88, cell*0.88,
                                 boxstyle='square,pad=0', fc='none', ec='white', lw=1.5))

# Text explanation
ax4.text(5.0, 6.2,
         'Where cross-modal similarity is used:', ha='left', fontsize=11,
         color=CRED, fontweight='bold')
uses = [
    ('Merge target scoring', CA, 7.5,
     'sim_av = â[pool] @ v̂ᵀ\nscores = sim_av.max(dim=1).values\n'
     '"Which audio token is most semantically\n related to any video token?"'),
    ('Merge weights', CY, 13.5,
     'w = softmax(scores)\nnew_anchor = (anchor + Σ audio[i]·wᵢ) / (1+Σwᵢ)\n'
     '"How much of each pruned token\n should the anchor absorb?"'),
    ('Audio→Video rate coupling', CV, 19.5,
     'Sa(i) = group_mask.float().mean()\nρ_v(i) = 0.75 + (0.35−0.75)·Sa(i)\n'
     '"Where audio is retained, keep\n more video tokens too"'),
]
for title, color, tx, body in uses:
    ax4.add_patch(FancyBboxPatch((tx, 3.5), 5.5, 2.5,
                                 boxstyle='round,pad=0.15', fc='#1A1010', ec=color, lw=1.5))
    ax4.text(tx+2.75, 5.7, title, ha='center', fontsize=9, fontweight='bold', color=color)
    ax4.text(tx+0.2, 5.35, body, ha='left', va='top', fontsize=8,
             color=FG, family='monospace')

# L2 normalization
eq_box(ax4, 0.3, 0.2, 25.3, 1.0,
       'L2 Normalization (applied before ALL similarity ops):  '
       'â = a / (‖a‖₂ + ε),   v̂ = v / (‖v‖₂ + ε)   ε=1e-6\n'
       '⟹  dot product â·v̂ = cos(θ)  ∈ [−1, 1],  invariant to embedding magnitude')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Global Mask Assembly & Output
# ══════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[5])
ax5.set_xlim(0, 26); ax5.set_ylim(0, 5.5)
ax5.axis('off')
ax5.set_title('Stage 5 — Global Mask Assembly & Output  [omnizip(), lines 293–307]',
              fontsize=12, color=CT, pad=6, fontweight='bold', loc='left')

cbox(ax5, 0.3, 2.8, 5.5, 2.3, 'audio_mask\n[Na bool]',
     'topk dominant +\ncontextual anchors', CA, fc='#0D2140', ec=CA, fontsize=10)
cbox(ax5, 0.3, 0.3, 5.5, 2.2, 'video_mask\n[Nv bool]',
     'ISTM spatial + temporal\nper 4-frame groups', CV, fc='#2A1000', ec=CV, fontsize=10)

arr(ax5, 5.9, 3.9, 7.8, 3.1, CA, lw=2)
arr(ax5, 5.9, 1.4, 7.8, 2.5, CV, lw=2)

cbox(ax5, 7.8, 1.8, 6.5, 2.8, 'global_mask  [L bool]',
     'ones(L)\nglobal_mask[video_indices] = video_mask\nglobal_mask[audio_indices] = audio_mask\n(text tokens always kept = True)',
     CT, fc='#0D2010', ec=CT, fontsize=10, bodyfontsize=8)

arr(ax5, 14.4, 3.2, 16.5, 3.2, CT, lw=2)

cbox(ax5, 16.5, 1.8, 5.5, 2.8, 'compressed_embeds\n[L′ × D]',
     'input_embeds[global_mask]\nL′ ≪ L\n(text unchanged,\naudio merged,\nvideo pruned)',
     'white', fc='#1A1A1A', ec='white', fontsize=10, bodyfontsize=8)

arr(ax5, 22.1, 3.2, 23.3, 3.2, 'white', lw=2)
cbox(ax5, 23.3, 1.8, 2.3, 2.8, 'Thinker\nLLM',
     'Prefill with\n<50% tokens', 'white', fc='#1A3020', ec=CT, fontsize=10)

# Stats box
ax5.text(0.3, 0.15,
    'Typical compression:  Audio ~50% kept  |  Video ~30–65% kept per frame  |  '
    'Total: <50% of original tokens enter LLM  |  Training-free, post-hoc, compatible with FlashAttn',
    ha='left', fontsize=9, color=CY,
    bbox=dict(fc='#1A1500', ec=CY, lw=1, boxstyle='round,pad=0.3'))

plt.savefig(r'C:\Users\Armaan\Desktop\PURS\viz_omnizip_math.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: viz_omnizip_math.png")
plt.close()
