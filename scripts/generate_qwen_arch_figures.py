"""
Generate architecture diagrams for Qwen2.5-Omni layer-by-layer explanation.
Produces 7 figures saved to figures/ directory.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT, exist_ok=True)

# Color palette
C = {
    'audio': '#4A90D9',
    'vision': '#7B68EE',
    'thinker': '#E8833A',
    'talker': '#50C878',
    'dit': '#DC143C',
    'bigvgan': '#DAA520',
    'norm': '#B0C4DE',
    'attn': '#FFB347',
    'ffn': '#87CEEB',
    'residual': '#D3D3D3',
    'input': '#98FB98',
    'output': '#FFB6C1',
    'proj': '#DDA0DD',
    'bg': '#FAFAFA',
    'arrow': '#333333',
}

def box(ax, x, y, w, h, text, color, fontsize=9, bold=False, alpha=0.85):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.05", facecolor=color, edgecolor='#333',
                        linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(r)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=3, wrap=True)
    return r

def arrow(ax, x1, y1, x2, y2, color='#333', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2),
                zorder=1)

def skip_arrow(ax, x1, y1, x2, y2, label='+', color='#999'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                               connectionstyle='arc3,rad=0.3', linestyle='--'),
                zorder=1)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx + 0.15, my, label, fontsize=8, color=color, ha='center', va='center')


# ============================================================
# Figure 1: Full Model Pipeline
# ============================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1, 4.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('Qwen2.5-Omni: Full Model Pipeline', fontsize=14, fontweight='bold', pad=15)

    # Input row
    box(ax, 1.5, 3.5, 2.2, 0.7, 'Audio Input\n(Mel Spectrogram)', C['input'], fontsize=8)
    box(ax, 4.5, 3.5, 2.2, 0.7, 'Video/Image Input\n(RGB Frames)', C['input'], fontsize=8)
    box(ax, 7.5, 3.5, 2.2, 0.7, 'Text Input\n(Token IDs)', C['input'], fontsize=8)

    # Encoder row
    box(ax, 1.5, 2, 2.2, 0.8, 'Audio Encoder\n32 Layers\n(Whisper-style)', C['audio'], fontsize=8, bold=True)
    box(ax, 4.5, 2, 2.2, 0.8, 'Vision Encoder\n32 Layers\n(ViT)', C['vision'], fontsize=8, bold=True)
    box(ax, 7.5, 2, 2.2, 0.8, 'Token\nEmbedding', C['norm'], fontsize=8)

    # Thinker
    box(ax, 4.5, 0.5, 7.5, 0.9, 'Thinker (LLM Backbone) — 28 Decoder Layers — GQA + SwiGLU + TMRoPE',
        C['thinker'], fontsize=9, bold=True)

    # Talker
    box(ax, 10.5, 2, 2.4, 0.8, 'Talker\n4 Decoder Layers\n(Speech LM)', C['talker'], fontsize=8, bold=True)

    # Token2Wav
    box(ax, 13.2, 2, 2, 0.8, 'DiT Vocoder\n22 Layers\n(Mel Pred)', C['dit'], fontsize=8, bold=True)
    box(ax, 13.2, 3.5, 2, 0.7, 'BigVGAN\n~12 Conv Layers\n(Waveform)', C['bigvgan'], fontsize=8, bold=True)

    # Output
    box(ax, 7.5, -0.8, 2.2, 0.6, 'Text Output', C['output'], fontsize=9)
    box(ax, 13.2, -0.8, 2.2, 0.6, 'Speech Output\n(Waveform)', C['output'], fontsize=9)

    # Arrows: inputs to encoders
    arrow(ax, 1.5, 3.1, 1.5, 2.45)
    arrow(ax, 4.5, 3.1, 4.5, 2.45)
    arrow(ax, 7.5, 3.1, 7.5, 2.45)

    # Encoders to Thinker
    arrow(ax, 1.5, 1.55, 2.5, 1.0)
    arrow(ax, 4.5, 1.55, 4.5, 1.0)
    arrow(ax, 7.5, 1.55, 7.5, 1.0)

    # Thinker to text output
    arrow(ax, 7.5, 0.05, 7.5, -0.5)

    # Thinker to Talker
    arrow(ax, 8.25, 0.7, 10.5, 1.55, color=C['talker'])

    # Talker to DiT
    arrow(ax, 11.7, 2, 12.2, 2)

    # DiT to BigVGAN
    arrow(ax, 13.2, 2.45, 13.2, 3.1)

    # BigVGAN to output
    arrow(ax, 13.2, 1.55, 13.2, -0.5)

    # Annotation
    ax.text(9.3, 1.35, 'hidden\nstates', fontsize=7, color=C['talker'], ha='center', style='italic')
    ax.text(7.5, -0.25, 'lm_head', fontsize=7, color='gray', ha='center', style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'qwen_omni_fig1_pipeline.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Figure 1 saved.")


# ============================================================
# Figure 2: Audio Encoder Layer
# ============================================================
def fig2_audio_encoder():
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('Audio Encoder Layer (x32)', fontsize=13, fontweight='bold', pad=10)

    # Input
    box(ax, 2, 11, 3, 0.5, 'Mel Spectrogram (num_mel_bins channels)', C['input'], fontsize=8)
    box(ax, 2, 10, 3, 0.5, 'Conv1d(mel→d_model, k=3) + GELU', C['proj'], fontsize=8)
    box(ax, 2, 9, 3, 0.5, 'Conv1d(d_model→d_model, k=3, s=2) + GELU', C['proj'], fontsize=8)
    box(ax, 2, 8, 3, 0.5, 'Sinusoidal Position Embedding', C['norm'], fontsize=8)

    arrow(ax, 2, 10.75, 2, 10.3)
    arrow(ax, 2, 9.75, 2, 9.3)
    arrow(ax, 2, 8.75, 2, 8.3)

    # Layer block
    y0 = 7
    box(ax, 2, y0, 3.5, 0.45, 'LayerNorm (pre-norm)', C['norm'], fontsize=8)
    arrow(ax, 2, y0+0.5, 2, y0+0.25)
    box(ax, 2, y0-0.7, 3.5, 0.55, 'Multi-Head Self-Attention\n(windowed via n_window)', C['attn'], fontsize=8)
    arrow(ax, 2, y0-0.25, 2, y0-0.42)
    box(ax, 2, y0-1.5, 3.5, 0.4, 'Residual Add (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y0-1.0, 2, y0-1.28)

    # skip connection
    skip_arrow(ax, -0.5, y0+0.5, -0.5, y0-1.5, '+')
    ax.plot([-0.5, 0.2], [y0+0.5, y0+0.5], color='#999', lw=1, ls='--')
    ax.plot([-0.5, 0.2], [y0-1.5, y0-1.5], color='#999', lw=1, ls='--')

    y1 = y0 - 2.4
    box(ax, 2, y1, 3.5, 0.45, 'LayerNorm (pre-norm)', C['norm'], fontsize=8)
    arrow(ax, 2, y0-1.72, 2, y1+0.25)
    box(ax, 2, y1-0.65, 3.5, 0.45, 'FFN: fc1(d→4d) → GELU → fc2(4d→d)', C['ffn'], fontsize=8)
    arrow(ax, 2, y1-0.25, 2, y1-0.4)
    box(ax, 2, y1-1.4, 3.5, 0.4, 'Residual Add (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y1-0.9, 2, y1-1.18)

    skip_arrow(ax, -0.5, y1+0.5, -0.5, y1-1.4, '+')
    ax.plot([-0.5, 0.2], [y1+0.5, y1+0.5], color='#999', lw=1, ls='--')
    ax.plot([-0.5, 0.2], [y1-1.4, y1-1.4], color='#999', lw=1, ls='--')

    # Repeat annotation
    ax.annotate('× 32 layers', xy=(4.5, (y0+y1)/2 - 0.5), fontsize=10, color=C['audio'],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['audio'], lw=1.5))

    # Output
    yout = y1 - 2.2
    box(ax, 2, yout, 3, 0.45, 'LayerNorm (post)', C['norm'], fontsize=8)
    arrow(ax, 2, y1-1.62, 2, yout+0.25)
    box(ax, 2, yout-0.65, 3, 0.45, 'AvgPool1d(k=2, s=2)', C['proj'], fontsize=8)
    arrow(ax, 2, yout-0.25, 2, yout-0.4)
    box(ax, 2, yout-1.3, 3, 0.45, 'Linear(d_model → output_dim)', C['proj'], fontsize=8)
    arrow(ax, 2, yout-0.9, 2, yout-1.05)
    box(ax, 2, yout-1.95, 3, 0.45, 'Audio Embeddings + BOS/EOS', C['output'], fontsize=8)
    arrow(ax, 2, yout-1.55, 2, yout-1.7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'qwen_omni_fig2_audio_encoder.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Figure 2 saved.")


# ============================================================
# Figure 3: Vision Encoder Layer
# ============================================================
def fig3_vision_encoder():
    fig, ax = plt.subplots(figsize=(6, 9.5))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-1.5, 12)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('Vision Encoder Block (x32)', fontsize=13, fontweight='bold', pad=10)

    # Input
    box(ax, 2, 11, 3.2, 0.5, 'Video/Image Frames (RGB)', C['input'], fontsize=8)
    box(ax, 2, 10, 3.8, 0.55, 'Conv3d Patch Embed\n(3→embed, k=[t,p,p], s=[t,p,p])', C['proj'], fontsize=8)
    box(ax, 2, 9, 3.2, 0.5, 'Rotary Position Embedding\n(spatial H,W + temporal)', C['norm'], fontsize=8)
    arrow(ax, 2, 10.75, 2, 10.3)
    arrow(ax, 2, 9.72, 2, 9.28)

    # Block
    y0 = 7.8
    box(ax, 2, y0, 3.5, 0.45, 'RMSNorm (pre-norm, eps=1e-6)', C['norm'], fontsize=8)
    arrow(ax, 2, y0+0.95, 2, y0+0.25)
    box(ax, 2, y0-0.7, 3.5, 0.55, 'Multi-Head Attention + RoPE\n(full or windowed per layer)', C['attn'], fontsize=8)
    arrow(ax, 2, y0-0.25, 2, y0-0.4)
    box(ax, 2, y0-1.5, 3.5, 0.4, 'Residual Add (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y0-1.0, 2, y0-1.28)

    skip_arrow(ax, -0.5, y0+0.5, -0.5, y0-1.5, '+')
    ax.plot([-0.5, 0.2], [y0+0.5, y0+0.5], color='#999', lw=1, ls='--')
    ax.plot([-0.5, 0.2], [y0-1.5, y0-1.5], color='#999', lw=1, ls='--')

    y1 = y0 - 2.4
    box(ax, 2, y1, 3.5, 0.45, 'RMSNorm (pre-norm, eps=1e-6)', C['norm'], fontsize=8)
    arrow(ax, 2, y0-1.72, 2, y1+0.25)
    box(ax, 2, y1-0.7, 3.5, 0.55, 'Gated MLP (SwiGLU)\ndown(GELU(gate(x)) * up(x))', C['ffn'], fontsize=8)
    arrow(ax, 2, y1-0.25, 2, y1-0.4)
    box(ax, 2, y1-1.5, 3.5, 0.4, 'Residual Add (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y1-1.0, 2, y1-1.28)

    skip_arrow(ax, -0.5, y1+0.5, -0.5, y1-1.5, '+')
    ax.plot([-0.5, 0.2], [y1+0.5, y1+0.5], color='#999', lw=1, ls='--')
    ax.plot([-0.5, 0.2], [y1-1.5, y1-1.5], color='#999', lw=1, ls='--')

    ax.annotate('× 32 blocks', xy=(4.5, (y0+y1)/2 - 0.5), fontsize=10, color=C['vision'],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['vision'], lw=1.5))

    # PatchMerger
    yout = y1 - 2.3
    box(ax, 2, yout, 3.5, 0.5, 'PatchMerger: RMSNorm → reshape(2x2)\n→ MLP(GELU) → 4x reduction', C['proj'], fontsize=8)
    arrow(ax, 2, y1-1.72, 2, yout+0.28)
    box(ax, 2, yout-0.75, 3, 0.45, 'Vision Embeddings', C['output'], fontsize=8)
    arrow(ax, 2, yout-0.28, 2, yout-0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'qwen_omni_fig3_vision_encoder.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Figure 3 saved.")


# ============================================================
# Figure 4: Thinker Decoder Layer
# ============================================================
def fig4_thinker():
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(-2.5, 7)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('Thinker Decoder Layer (x28)', fontsize=13, fontweight='bold', pad=10)

    # Multimodal fusion
    box(ax, -0.5, 10, 2, 0.5, 'Audio\nEmbeddings', C['audio'], fontsize=7)
    box(ax, 2, 10, 2, 0.5, 'Vision\nEmbeddings', C['vision'], fontsize=7)
    box(ax, 4.5, 10, 2, 0.5, 'Text Token\nEmbeddings', C['norm'], fontsize=7)
    box(ax, 2, 9, 4.5, 0.55, 'Multimodal Embedding Fusion\n(replace tokens at marked positions)', C['proj'], fontsize=8)

    arrow(ax, -0.5, 9.72, 0.8, 9.3)
    arrow(ax, 2, 9.72, 2, 9.3)
    arrow(ax, 4.5, 9.72, 3.2, 9.3)

    # Decoder layer
    y0 = 7.5
    box(ax, 2, y0, 3.8, 0.45, 'RMSNorm (input_layernorm)', C['norm'], fontsize=8)
    arrow(ax, 2, 8.7, 2, y0+0.25)

    box(ax, 2, y0-0.8, 3.8, 0.65, 'Grouped-Query Attention (GQA)\n+ TMRoPE (time-aligned\nmultimodal rotary)', C['attn'], fontsize=8)
    arrow(ax, 2, y0-0.25, 2, y0-0.45)

    box(ax, 2, y0-1.7, 3.8, 0.4, 'Residual Add (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y0-1.15, 2, y0-1.48)

    skip_arrow(ax, -0.8, y0+0.5, -0.8, y0-1.7, '+')
    ax.plot([-0.8, 0.08], [y0+0.5, y0+0.5], color='#999', lw=1, ls='--')
    ax.plot([-0.8, 0.08], [y0-1.7, y0-1.7], color='#999', lw=1, ls='--')

    y1 = y0 - 2.7
    box(ax, 2, y1, 3.8, 0.45, 'RMSNorm (post_attention_layernorm)', C['norm'], fontsize=8)
    arrow(ax, 2, y0-1.92, 2, y1+0.25)

    box(ax, 2, y1-0.8, 3.8, 0.65, 'SwiGLU MLP\ndown_proj(SiLU(gate(x)) * up(x))\nhidden → 4h → hidden', C['ffn'], fontsize=8)
    arrow(ax, 2, y1-0.25, 2, y1-0.45)

    box(ax, 2, y1-1.7, 3.8, 0.4, 'Residual Add (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y1-1.15, 2, y1-1.48)

    skip_arrow(ax, -0.8, y1+0.5, -0.8, y1-1.7, '+')
    ax.plot([-0.8, 0.08], [y1+0.5, y1+0.5], color='#999', lw=1, ls='--')
    ax.plot([-0.8, 0.08], [y1-1.7, y1-1.7], color='#999', lw=1, ls='--')

    ax.annotate('× 28 layers', xy=(5, (y0+y1)/2 - 0.5), fontsize=10, color=C['thinker'],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['thinker'], lw=1.5))

    # Output
    yout = y1 - 2.5
    box(ax, 2, yout, 3, 0.45, 'RMSNorm (final)', C['norm'], fontsize=8)
    arrow(ax, 2, y1-1.92, 2, yout+0.25)
    box(ax, 0, yout-0.75, 2.2, 0.45, 'lm_head\n(→ vocab_size)', C['output'], fontsize=8)
    box(ax, 4, yout-0.75, 2.5, 0.45, 'Hidden States\n→ Talker', C['talker'], fontsize=8)
    arrow(ax, 1.2, yout-0.25, 0, yout-0.5)
    arrow(ax, 2.8, yout-0.25, 4, yout-0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'qwen_omni_fig4_thinker.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Figure 4 saved.")


# ============================================================
# Figure 5: Talker Architecture
# ============================================================
def fig5_talker():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-0.5, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('Talker (Speech Decoder, x4 layers)', fontsize=13, fontweight='bold', pad=10)

    # Inputs
    box(ax, 0.5, 9, 2.5, 0.55, 'Thinker Hidden\nStates', C['thinker'], fontsize=8)
    box(ax, 3.5, 9, 2, 0.55, 'Codec Token\nEmbedding', C['norm'], fontsize=8)

    box(ax, 0.5, 8, 2.5, 0.5, 'thinker_to_talker_proj\nLinear(emb→hidden)', C['proj'], fontsize=8)
    arrow(ax, 0.5, 8.7, 0.5, 8.28)

    box(ax, 2, 7, 3.5, 0.5, 'Concatenate / Fuse Inputs', C['proj'], fontsize=8)
    arrow(ax, 0.5, 7.72, 1.2, 7.28)
    arrow(ax, 3.5, 8.7, 3.5, 7.28)

    # Decoder layer
    y0 = 5.8
    box(ax, 2, y0, 3.5, 0.45, 'RMSNorm (input_layernorm)', C['norm'], fontsize=8)
    arrow(ax, 2, 6.72, 2, y0+0.25)

    box(ax, 2, y0-0.7, 3.5, 0.55, 'GQA Self-Attention\n+ RoPE + KV-Cache', C['attn'], fontsize=8)
    arrow(ax, 2, y0-0.25, 2, y0-0.4)

    box(ax, 2, y0-1.45, 3.5, 0.35, 'Residual (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y0-1.0, 2, y0-1.25)

    y1 = y0 - 2.2
    box(ax, 2, y1, 3.5, 0.45, 'RMSNorm (post_attn_layernorm)', C['norm'], fontsize=8)
    arrow(ax, 2, y0-1.65, 2, y1+0.25)

    box(ax, 2, y1-0.65, 3.5, 0.45, 'SwiGLU MLP', C['ffn'], fontsize=8)
    arrow(ax, 2, y1-0.25, 2, y1-0.4)

    box(ax, 2, y1-1.3, 3.5, 0.35, 'Residual (+)', C['residual'], fontsize=8)
    arrow(ax, 2, y1-0.9, 2, y1-1.1)

    ax.annotate('× 4 layers', xy=(4.8, (y0+y1)/2 - 0.3), fontsize=10, color=C['talker'],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['talker'], lw=1.5))

    # Output
    yout = y1 - 2.1
    box(ax, 2, yout, 3, 0.45, 'RMSNorm (final)', C['norm'], fontsize=8)
    arrow(ax, 2, y1-1.5, 2, yout+0.25)
    box(ax, 2, yout-0.7, 3, 0.5, 'codec_head\nLinear(hidden → codebook_size)', C['output'], fontsize=8)
    arrow(ax, 2, yout-0.25, 2, yout-0.42)
    box(ax, 2, yout-1.4, 2.5, 0.45, 'Speech Codec\nTokens', C['output'], fontsize=8, bold=True)
    arrow(ax, 2, yout-0.97, 2, yout-1.15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'qwen_omni_fig5_talker.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Figure 5 saved.")


# ============================================================
# Figure 6: DiT Decoder Layer
# ============================================================
def fig6_dit():
    fig, ax = plt.subplots(figsize=(6, 9.5))
    ax.set_xlim(-2, 6.5)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('DiT Vocoder Layer (x22)', fontsize=13, fontweight='bold', pad=10)

    # Inputs
    box(ax, 0, 11, 2, 0.5, 'Codec Tokens', C['input'], fontsize=8)
    box(ax, 3, 11, 2.5, 0.5, 'Noised Mel +\nSpeaker Emb', C['input'], fontsize=8)
    box(ax, 5.5, 11, 1.5, 0.5, 'Timestep t', C['input'], fontsize=8)

    box(ax, 0, 10, 2.2, 0.5, 'DiTCodecEmb\n(+ repeat)', C['proj'], fontsize=8)
    box(ax, 3, 10, 2.2, 0.5, 'DiTInputEmb\n(fuse cond)', C['proj'], fontsize=8)
    box(ax, 5.5, 10, 1.8, 0.5, 'TimestepEmb\nSin→MLP(SiLU)', C['proj'], fontsize=8)
    arrow(ax, 0, 10.72, 0, 10.28)
    arrow(ax, 3, 10.72, 3, 10.28)
    arrow(ax, 5.5, 10.72, 5.5, 10.28)

    box(ax, 2, 9, 4, 0.5, 'Combine Embeddings', C['proj'], fontsize=8)
    arrow(ax, 0, 9.72, 1, 9.28)
    arrow(ax, 3, 9.72, 2.5, 9.28)

    # Layer
    y0 = 7.7
    box(ax, 2, y0, 4, 0.55, 'AdaLayerNormZero\n(timestep-conditioned norm + gate)', C['norm'], fontsize=8)
    arrow(ax, 2, 8.72, 2, y0+0.3)
    arrow(ax, 5.5, 9.72, 5, y0+0.1, color=C['dit'])
    ax.text(5.3, 8.5, 't', fontsize=7, color=C['dit'], style='italic')

    box(ax, 2, y0-0.85, 4, 0.55, 'Block-Sparse Attention\n(look_ahead + look_backward)', C['attn'], fontsize=8)
    arrow(ax, 2, y0-0.3, 2, y0-0.55)

    box(ax, 2, y0-1.6, 4, 0.4, 'Gate * Attn + Residual', C['residual'], fontsize=8)
    arrow(ax, 2, y0-1.15, 2, y0-1.38)

    y1 = y0 - 2.5
    box(ax, 2, y1, 4, 0.55, 'AdaLayerNormZero\n(timestep-conditioned norm + gate)', C['norm'], fontsize=8)
    arrow(ax, 2, y0-1.82, 2, y1+0.3)

    box(ax, 2, y1-0.8, 4, 0.5, 'DiT MLP (Linear→GELU→Linear)', C['ffn'], fontsize=8)
    arrow(ax, 2, y1-0.3, 2, y1-0.52)

    box(ax, 2, y1-1.5, 4, 0.4, 'Gate * MLP + Residual', C['residual'], fontsize=8)
    arrow(ax, 2, y1-1.08, 2, y1-1.28)

    ax.annotate('× 22 layers', xy=(5.3, (y0+y1)/2 - 0.3), fontsize=10, color=C['dit'],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['dit'], lw=1.5))

    # Output
    yout = y1 - 2.3
    box(ax, 2, yout, 3.5, 0.5, 'AdaLayerNormZero_Final\n(timestep cond)', C['norm'], fontsize=8)
    arrow(ax, 2, y1-1.72, 2, yout+0.28)
    box(ax, 2, yout-0.7, 3, 0.45, 'Linear(hidden → mel_dim)', C['proj'], fontsize=8)
    arrow(ax, 2, yout-0.28, 2, yout-0.45)
    box(ax, 2, yout-1.4, 2.5, 0.45, 'Predicted Mel\nSpectrogram', C['output'], fontsize=8, bold=True)
    arrow(ax, 2, yout-0.95, 2, yout-1.15)

    ax.text(2, yout-2.1, 'Euler ODE Solver (iterative denoising)', fontsize=7,
            ha='center', style='italic', color='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'qwen_omni_fig6_dit.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Figure 6 saved.")


# ============================================================
# Figure 7: BigVGAN Vocoder
# ============================================================
def fig7_bigvgan():
    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.set_xlim(-1.5, 6)
    ax.set_ylim(-0.5, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('BigVGAN Vocoder (~12 conv layers)', fontsize=13, fontweight='bold', pad=10)

    box(ax, 2, 8.2, 3.5, 0.5, 'Predicted Mel Spectrogram\n(from DiT)', C['input'], fontsize=8)
    box(ax, 2, 7.2, 3.5, 0.5, 'conv_pre: Conv1d(mel → ch, k=7)', C['proj'], fontsize=8)
    arrow(ax, 2, 7.92, 2, 7.48)

    # Upsample block
    y = 6
    box(ax, 2, y, 4, 0.55, 'ConvTranspose1d Upsample\n(stride = upsample_rate[i])', C['proj'], fontsize=8)
    arrow(ax, 2, 6.92, 2, y+0.3)

    y -= 1.1
    box(ax, 2, y, 4, 0.8, 'AMPBlock (Residual)\n  SnakeBeta: x + (1/b)sin²(ax)\n  Dilated Conv1d (d=1,3,5)\n  Parallel paths → sum', C['ffn'], fontsize=8)
    arrow(ax, 2, y+1.0, 2, y+0.42)

    ax.annotate('× N upsample\n  stages', xy=(5, y+0.3), fontsize=9, color=C['bigvgan'],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['bigvgan'], lw=1.5))

    y -= 1.3
    box(ax, 2, y, 4, 0.5, 'SnakeBeta Activation', C['ffn'], fontsize=8)
    arrow(ax, 2, y+0.88, 2, y+0.28)

    y -= 0.85
    box(ax, 2, y, 3.5, 0.5, 'conv_post: Conv1d(ch → 1, k=7)', C['proj'], fontsize=8)
    arrow(ax, 2, y+0.62, 2, y+0.28)

    y -= 0.8
    box(ax, 2, y, 2.5, 0.45, 'tanh activation', C['proj'], fontsize=8)
    arrow(ax, 2, y+0.52, 2, y+0.25)

    y -= 0.75
    box(ax, 2, y, 3, 0.5, 'Waveform Output\n(1D audio signal)', C['output'], fontsize=8, bold=True)
    arrow(ax, 2, y+0.52, 2, y+0.28)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'qwen_omni_fig7_bigvgan.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Figure 7 saved.")


if __name__ == '__main__':
    fig1_pipeline()
    fig2_audio_encoder()
    fig3_vision_encoder()
    fig4_thinker()
    fig5_talker()
    fig6_dit()
    fig7_bigvgan()
    print("\nAll 7 figures generated in:", os.path.abspath(OUT))
