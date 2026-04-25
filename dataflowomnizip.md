# Dataflow OmniZip

This document compiles the findings from the earlier analysis of `Qwen3-Omni`, `Qwen2.5-Omni`, and `OmniZip-main/omnizip` in this workspace.

It focuses on:
- what is actually implemented locally
- how prompting and evaluation are wired
- how `Qwen3-Omni` differs from `Qwen2.5-Omni`
- how `OmniZip` differs from both
- the exact OmniZip dataflow from model forward pass to token pruning

## 1. What Exists in This Workspace

### `Qwen3-Omni`

Local `Qwen3-Omni` is mostly:
- `README.md`
- `web_demo.py`
- `web_demo_captioner.py`
- cookbooks
- assets and Docker bits

It does not include a local model implementation file. The actual model classes are imported from `transformers`:
- `Qwen3OmniMoeForConditionalGeneration`
- `Qwen3OmniMoeProcessor`

So in this workspace, `Qwen3-Omni` is mainly:
- usage docs
- prompt guidance
- demo wiring
- cookbook examples

### `Qwen2.5-Omni`

Local `Qwen2.5-Omni` includes:
- `README.md`
- `web_demo.py`
- cookbooks
- `qwen-omni-utils`

The actual model class normally comes from `transformers`, while `qwen-omni-utils` handles multimodal preprocessing such as:
- reading images
- decoding video frames
- extracting audio
- sampling video at a chosen fps

### `OmniZip-main/omnizip`

This is the actual patched implementation. It includes:
- `omnizip/modeling_qwen2_5_omni.py`
- `omnizip/omnizip_units.py`
- `lmms-eval/`

This is the important distinction:
- `Qwen3-Omni` and `Qwen2.5-Omni` local folders are mostly docs and demos
- `OmniZip-main/omnizip` contains real model code changes

## 2. Shared Runtime Pattern

Across Qwen2.5 and Qwen3, the runtime pattern is broadly:

```python
messages
-> processor.apply_chat_template(...)
-> process_mm_info(...)
-> processor(text=..., audio=..., images=..., videos=...)
-> model.generate(...)
```

This appears in:
- `Qwen3-Omni/README.md`
- `Qwen3-Omni/web_demo.py`
- `Qwen2.5-Omni/README.md`
- `Qwen2.5-Omni/web_demo.py`
- local eval scripts

The most important control variable is usually:

```python
use_audio_in_video=True or False
```

That flag must stay consistent across:
- `process_mm_info(...)`
- `processor(...)`
- `model.generate(...)`

## 3. Qwen3-Omni Prompt and Usage Structure

### Main usage pattern

Qwen3 examples use multimodal content followed by text inside a user turn:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "..."},
            {"type": "audio", "audio": "..."},
            {"type": "text", "text": "What can you see and hear?"},
        ],
    },
]
```

Then:

```python
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, ...)
```

### Demo prompt behavior

In `Qwen3-Omni/web_demo.py`:
- `default_system_prompt = ''`
- a system prompt is optional
- user media items are grouped before text in the final user content list

So the Qwen3 demo is relatively permissive:
- system prompt can be empty
- multimodal items are placed before text
- audio/video/image are grouped into a single user turn

### Qwen3 guidance for interactive AV use

Qwen3 explicitly recommends a specialized system prompt for audio-visual interaction when the video audio acts like part of the user query. The guidance emphasizes:
- short, conversational spoken responses
- no structured formatting
- no action descriptions
- answer the user's question rather than narrating the video directly
- reply in the user's language

### Qwen3 thinking model guidance

The `Thinking` model guidance says each turn should ideally contain an explicit textual instruction alongside the multimodal input, for example:

```python
[
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/path/to/audio.wav"},
            {"type": "image", "image": "/path/to/image.png"},
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "Analyze this audio, image, and video together."},
        ],
    }
]
```

## 4. Qwen2.5-Omni Prompt and Usage Structure

### Main usage pattern

Qwen2.5 examples usually include a system prompt by default:

```python
[
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "..."},
        ],
    },
]
```

### Audio output behavior

Qwen2.5 is stricter about audio output:
- if you want audio output, the "virtual human" system prompt is expected
- audio output is tied closely to that persona/system setting

Qwen2.5 also notes that if you need extra behavior control while using audio output, a workaround is to simulate instruction tuning inside the conversation history rather than relying on arbitrary system prompt customization.

### Demo prompt behavior

In `Qwen2.5-Omni/web_demo.py`:
- the system prompt is always included
- history formatting is simpler
- each user media item becomes its own content list
- `generate(...)` returns both text and audio in the demo path

So Qwen2.5's public-facing examples are more strongly centered on:
- a fixed system identity
- audio-capable assistant behavior
- predictable prompt scaffolding

## 5. Official Evaluation Guidance for Qwen3-Omni

The Qwen3 README gives explicit evaluation guidance:

- `Instruct` models should use greedy decoding
- `Thinking` models should use parameters from the checkpoint `generation_config.json`
- benchmark-specific formatting should be respected
- all video data should be evaluated at `fps=2`
- no system prompt should be set for evaluation
- unless a benchmark says otherwise, multimodal items should come before the text instruction in the user content

Qwen3 also lists default prompts for tasks missing their own prompt, including:
- Chinese ASR
- multilingual ASR
- speech-to-text translation
- song lyric transcription

Important consequence:
- official Qwen3 eval guidance says no system prompt
- local workspace eval scripts do not fully follow that rule

## 6. Local Evaluation in This Workspace

The workspace has two custom eval entry points:
- `eval_qwen_omni.py`
- `eval_qwen_omni_zip.py`

### `eval_qwen_omni.py`

This script can load either:
- `qwen2.5-omni`
- `qwen3-omni`

For Qwen3, it imports the model classes lazily from `transformers`.

### Important caveat

Even when running `--model_variant qwen3-omni`, the local script still uses the same custom prompt-builder logic used for Qwen2.5-style MCQ evaluation. So the local Qwen3 baseline is not a pure reproduction of the official Qwen3 evaluation recipe.

### Local prompt templates

The script defines dataset-specific user prompts:

- `Video-MME`
  - "Select the best answer to the following multiple-choice question based on the video and the subtitles."
  - ends with `"The best answer is:"`

- `WorldSense`
  - prepends:
    - "Carefully watch this video and pay attention to every detail."
    - "Based on your observations, select the best option that accurately addresses the question."
  - then:
    - "These are the frames of a video and the corresponding audio."
    - question
    - candidate answers

- `DailyOmni`
  - "Listen and watch the video carefully."
  - "Select the best answer..."
  - answer letter only

- default
  - generic video-based MCQ prompt

### Local message structure

The local eval script builds messages like:

```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DEFAULT}]},
    {"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
        {"type": "text", "text": prompt},
    ]},
]
```

Then it runs:

```python
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(messages, use_audio_in_video=effective_use_audio)
inputs = processor(...)
output = model.generate(..., temperature=0.0, do_sample=False, return_audio=False)
```

### Consequences

Local evaluation differs from official Qwen3 guidance because:
- it injects a system prompt
- it reuses Qwen2.5-style prompt scaffolding
- some runs reduce fps and max pixels for VRAM reasons

## 7. Upstream `lmms-eval` Prompt Wiring

In `OmniZip-main/lmms-eval`, the benchmark prompt definitions live in task utils such as:
- `lmms_eval/tasks/videomme/utils.py`
- `lmms_eval/tasks/worldsense/utils.py`

Examples:

### Video-MME

The prompt builder does:
- question
- options
- "Select the best answer..."
- `"The best answer is:"`

Subtitle variants prepend:
- "This video's subtitles are listed below:"

### WorldSense

The task utils define:
- `SYS`
- `FRAMES_TMPL_NOSUB`
- `FRAMES_TMPL_SUB`
- `FRAMES_TMPL_AUDIO`

The audio case is:

```text
Carefully watch this video and pay attention to every detail.
Based on your observations, select the best option that accurately addresses the question.
These are the frames of a video and the corresponding audio.
Select the best answer...
```

This matches the prompt pattern copied into the local eval scripts.

## 8. Where OmniZip Fits

OmniZip is not a new prompt format and not a separate model family.

It is:
- a patched Qwen2.5-Omni thinker
- plus wiring in `lmms-eval`
- plus compression parameters like:
  - `rho_audio`
  - `rho_video`
  - `g`
  - `contextual_ratio`

In the `lmms-eval` adapter:
- if `WRAPPER=OmniZip`, it imports the patched model class
- otherwise it imports the standard Qwen2.5 class

So prompt structure is mostly the same as Qwen2.5 eval, but the internal token flow changes.

## 9. High-Level Differences: Qwen3 vs Qwen2.5 vs OmniZip

## Qwen3-Omni

- model family: newer omni-modal generation model
- local repo content: mostly docs, demos, cookbooks
- actual model implementation: from `transformers`
- official eval guidance: no system prompt, benchmark-native formatting, `fps=2`
- interaction style: more flexible system prompting, stronger emphasis on explicit multimodal task text for thinking mode

## Qwen2.5-Omni

- model family: earlier Thinker-Talker omni model
- notable architecture note: TMRoPE time alignment
- local examples strongly use the "virtual human" system prompt
- audio output behavior is more tightly tied to that system prompt

## OmniZip

- not a new base model family
- a Qwen2.5-Omni patch that compresses multimodal tokens
- prompt and task wiring mostly stay the same
- internal sequence is shortened before the thinker processes it
- key value proposition: speed and memory savings with minimal quality loss

## 10. OmniZip: Integration Layer vs Compression Logic

There are two different code responsibilities:

### `omnizip/modeling_qwen2_5_omni.py`

This is the integration layer.

It is responsible for:
- collecting audio features and audio attention outputs from the model
- deciding when OmniZip should run
- loading OmniZip config values
- calling `omnizip(...)`
- applying the returned `global_mask` to:
  - `inputs_embeds`
  - `attention_mask`
  - `position_ids`

This file answers:
- when does OmniZip run?
- what tensors are passed into it?
- how is the compressed sequence fed back into the model?

### `omnizip/omnizip_units.py`

This is the compression algorithm.

It is responsible for:
- deciding which audio tokens to keep
- deciding which audio tokens to merge into anchors
- estimating temporal importance from audio retention
- converting that into video pruning ratios
- building `audio_mask`, `video_mask`, and `global_mask`

This file answers:
- which tokens survive?
- which dropped tokens get merged?
- how does audio guide video compression?

## 11. OmniZip Dataflow End-to-End

This is the key flow.

### Step 1: model forward gets audio tower outputs

Inside the patched thinker in `modeling_qwen2_5_omni.py`, the model gets:
- `audio_features`
- `attn_logits`

Then if video, audio, and attention outputs are present, it runs:

```python
inputs_embeds, global_mask = omnizip(...)
```

### Step 2: `omnizip(...)` isolates audio and video token positions

Inside `omnizip(...)`, it flattens the embeddings and ids if needed, then finds:
- `video_indices`
- `audio_indices`

It pulls:
- `video_feature = flat_embeds[video_indices]`
- `audio_feature = flat_embeds[audio_indices]`

So it now has:
- the audio token embeddings only
- the video token embeddings only
- the positions those tokens occupy in the full sequence

### Step 3: audio attention is reduced to audio token importance

`omnizip_audio_attn(...)` converts `attn_logits` to a 1D importance score:
- average over heads if needed
- sum over incoming attention
- align the resulting score length with audio token count

Then it keeps the top dominant audio tokens:

```python
dominant_num = round((1.0 - merging_ratio_audio) * N)
```

These become the first `keep_mask` entries set to `True`.

### Step 4: contextual anchors are added

Among the remaining audio tokens, the algorithm chooses a few contextual anchors based on:

```python
contextual_num = round(contextual_ratio * N)
```

These anchors are also forced to stay in the audio keep mask.

So at this point:
- some audio tokens are kept because they are high-importance
- some audio tokens are kept because they act as contextual anchors
- the rest are eligible to be merged and dropped

### Step 5: build `merge_plan`

The remaining non-kept audio tokens are assigned to anchors:
- audio-audio similarity is used to attach each leftover token to an anchor
- audio-video similarity is used to rank which leftovers are most important to merge

For each anchor, at most `g` leftover tokens are selected:

```python
merge_plan[anchor_idx] = [list of dropped tokens to fold into that anchor]
```

So `merge_plan` does not keep those dropped tokens as separate positions.
It tells the system which dropped audio content should be folded into each kept anchor embedding.

### Step 6: merge dropped audio into kept anchor embeddings

Back in `omnizip(...)`, the algorithm iterates through `merge_plan`.

For each anchor:
- it computes weights from cross-modal or audio similarity
- it computes a weighted sum of the dropped tokens assigned to that anchor
- it writes a new merged vector into the anchor slot

Conceptually:
- the anchor stays
- the merged tokens disappear as positions
- some of their information survives inside the anchor embedding

### Step 7: compute `audio_group_retention`

After audio keep/drop is decided, OmniZip computes retention by temporal group:

```python
audio_group_retention = kept_fraction_per_group
```

This is the critical bridge from audio compression to video compression.

Interpretation:
- high retention means the audio in that time region seems important
- low retention means the audio in that region seems less important

### Step 8: map audio retention to video merge ratios

OmniZip maps each audio retention value into a video merging ratio.

The code clamps ratios between:
- `min_ratio = 0.35`
- `max_ratio = 0.75`

Then it rescales them toward the requested overall target from `rho_video`.

Effectively:
- more important audio groups keep more visual detail
- less important audio groups can be pruned more aggressively

### Step 9: prune video tokens with `omnizip_istm(...)`

Video pruning happens groupwise.

`omnizip_istm(...)` processes video frames in alternating ways:
- on even frames/groups it does spatial token selection using a similarity-based `dpcknn` strategy
- on odd frames/groups it compares tokens to the previous frame and keeps the less redundant ones

The output is `video_mask`.

So video pruning is not random. It is structured and temporally aware.

### Step 10: build `global_mask`

At the end of `omnizip(...)`, the algorithm creates a full-sequence mask:

```python
global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool)
global_mask[video_indices] = video_mask
global_mask[audio_indices] = audio_mask
```

Important:
- text tokens are left untouched
- video tokens are replaced by `video_mask`
- audio tokens are replaced by `audio_mask`

So `global_mask` is the final full-sequence keep/drop decision.

### Step 11: physically shorten the sequence

Back in `modeling_qwen2_5_omni.py`, the thinker applies:

```python
inputs_embeds = inputs_embeds[:, global_mask, :]
attention_mask = attention_mask[:, global_mask]
position_ids = position_ids[:, :, global_mask]
```

This is where compression becomes real.

The model does not merely "ignore" dropped tokens.
It actually removes them from the sequence passed into the main transformer stack.

That is why OmniZip saves memory and compute.

## 12. Toy Example of OmniZip Dataflow

Imagine one flattened sequence with positions:

```text
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
T  T  V  V  V  V  V  V  V  V  A  A  A  A  A  A  A  A  T
```

So:
- text at `0,1,18`
- video at `2..9`
- audio at `10..17`

### Audio stage

Suppose audio importance says to keep:
- `a0, a2, a4, a7`

Initial relative audio mask:

```text
[1,0,1,0,1,0,0,1]
```

Then contextual anchors add:
- `a1, a5`

Updated audio mask:

```text
[1,1,1,0,1,1,0,1]
```

Suppose merge plan becomes:

```python
{
  1: [3],
  5: [6],
}
```

Meaning:
- merge dropped `a3` into anchor `a1`
- merge dropped `a6` into anchor `a5`

But positions `a3` and `a6` still disappear from the final sequence.

### Video stage

Suppose the algorithm decides:

```text
video_mask = [1,1,0,0,1,0,1,0]
```

### Final global mask

Then the final full-sequence mask becomes:

```text
pos:          0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
token:        T T V V V V V V V V  A  A  A  A  A  A  A  A  T
global_mask:  1 1 1 1 0 0 1 0 1 0  1  1  1  0  1  1  0  1  1
```

Meaning:
- all text stays
- only selected video tokens stay
- only selected audio tokens stay
- merged audio information survives in anchor embeddings, not as separate positions

Then the thinker sees only the compressed sequence.

## 13. What Is Different Between the Two OmniZip Files

This was an important question in the earlier discussion.

### `modeling_qwen2_5_omni.py`

This file:
- lives inside the model forward pass
- gets tensors from the Qwen architecture
- calls OmniZip
- applies the output mask

It is the hook point.

### `omnizip_units.py`

This file:
- implements the actual token compression policy
- turns audio attention into `audio_mask`
- turns audio retention into video pruning
- returns the compressed embedding state and `global_mask`

It is the decision-making layer.

Simplest mental model:

- `modeling_qwen2_5_omni.py` = where OmniZip is plugged in
- `omnizip_units.py` = how OmniZip decides what to keep

## 14. Practical Takeaways

- If you are studying prompt structures, most of that lives in:
  - `Qwen3-Omni/README.md`
  - `Qwen2.5-Omni/README.md`
  - local eval scripts
  - `lmms-eval` task utils

- If you are studying benchmark behavior, the most relevant files are:
  - `eval_qwen_omni.py`
  - `eval_qwen_omni_zip.py`
  - `OmniZip-main/lmms-eval/lmms_eval/tasks/videomme/utils.py`
  - `OmniZip-main/lmms-eval/lmms_eval/tasks/worldsense/utils.py`
  - `OmniZip-main/lmms-eval/lmms_eval/models/simple/qwen2_5_omni.py`

- If you are studying compression internals, the core files are:
  - `OmniZip-main/omnizip/modeling_qwen2_5_omni.py`
  - `OmniZip-main/omnizip/omnizip_units.py`

- If you are comparing official Qwen3 evaluation to local workspace evaluation, note that the local Qwen3 path uses a custom Qwen2.5-style harness and is not identical to the official README recipe.

## 15. Short Summary

- `Qwen3-Omni` here is mainly docs, demos, and cookbook guidance.
- `Qwen2.5-Omni` here is docs, demos, utils, and classic prompt scaffolding.
- `OmniZip` is a patched Qwen2.5 thinker that compresses audio and video token sequences before the main transformer processes them.
- Audio attention drives audio keep/merge decisions.
- Audio retention over time drives adaptive video pruning.
- The final result is a `global_mask` that physically shortens the multimodal input sequence.
