# Dataset Schemas

## 1. Video-MME (`lmms-lab/Video-MME`)
**2,700 test rows · YouTube videos · 12 task types · 6 domains**

| Field | Type | Description |
|---|---|---|
| `video_id` | string | 3-char unique video identifier |
| `url` | string | Full YouTube watch URL |
| `videoID` | string | YouTube video ID (11–13 chars), use for embeds |
| `duration` | string (3 classes) | short / medium / long |
| `domain` | string (6 values) | Knowledge, Literature & Art, Biology & Medicine, Finance & Commerce, etc. |
| `sub_category` | string (30 values) | Fine-grained topic (e.g. Humanity & History) |
| `question_id` | string | Unique question identifier |
| `task_type` | string (12 types) | See categories below |
| `question` | string (19–813 chars) | Question text |
| `options` | list[string] (4 items) | Multiple-choice options [A, B, C, D] |
| `answer` | string (A/B/C/D) | Correct answer letter |

**`task_type` categories (12):**
1. Counting Problem
2. Information Synopsis
3. Object Recognition
4. Action Recognition
5. Action Reasoning
6. Object Reasoning
7. Attribute Perception
8. Spatial Perception
9. Temporal Perception
10. Temporal Reasoning
11. OCR Problems
12. *(12th unconfirmed — HF viewer listed Temporal Perception twice)*

**Sample rows used:**
- Row 1: video_id=001, videoID=fFjv93ACGo8, question_id=001-1, task_type=Counting Problem
- Row 2: video_id=001, videoID=fFjv93ACGo8, question_id=001-2, task_type=Information Synopsis

---

## 2. Daily-Omni (`liarliar/Daily-Omni`)
**1,197 rows · YouTube videos · 6 question types · Audio-visual alignment · CC-BY-NC-SA-4.0**

| Field | Type | Description |
|---|---|---|
| `video_id` | string | YouTube video ID (11 chars), use for embeds |
| `Question` | string (40–204 chars) | Question text |
| `Choice` | list[string] (4 items) | Multiple-choice options [A, B, C, D] |
| `Answer` | string (A/B/C/D) | Correct answer letter |
| `Type` | string (6 types) | See categories below |
| `content_parent_category` | string (10 values) | Top-level content category (Lifestyle, Hobbies & Interests, etc.) |
| `content_fine_category` | string (95 values) | Fine-grained category (e.g. Skincare Routines) |
| `video_category` | string (15 values) | YouTube video category (e.g. Howto & Style) |
| `video_duration` | string | Clip length (e.g. "30s") |
| `Explaination` | string \| null | Optional reasoning explanation (note: typo in original field name) |

**`Type` categories (6):**
1. AV Event Alignment
2. Event Sequence
3. Inference
4. Reasoning
5. Context Understanding
6. Comparative Analysis

**Sample rows used:**
- Row 1: video_id=Ec_lQgZ9wlg, Type=Event Sequence, category=Lifestyle > Skincare Routines
- Row 2: video_id=XUWxQYmiBQY, Type=AV Event Alignment, category=Hobbies & Interests > Toy Collections

---

## 3. WorldSense (`honglyhly/WorldSense`)
**1,662 videos · 3,172 QA pairs · 26 task types · 8 domains · 67 subcategories**
**⚠️ HuggingFace viewer broken (video format error) — videos not publicly streamable**

| Field | Type | Description |
|---|---|---|
| `video_id` | string | Video filename without .mp4 (e.g. ws_0042) |
| `video_caption` | string | Human-annotated caption describing video content |
| `domain` | string (8 domains) | Daily Life / Sports / Film & TV / Music / Tech & Science / Culture & Politics / Performance / Games |
| `task_type` (= `problem_type`) | string (26 types) | See categories below |
| `task_domain` (= `data_type`) | string | Audio / Visual / Audio-Visual |
| `question` | string | Multiple-choice question text |
| `candidates` | list[string] (4 items) | Answer choices [A, B, C, D] |
| `answer` | string (A/B/C/D) | Correct answer letter |

**`task_type` categories (26 total — 8 confirmed, 18 unknown without downloading dataset):**
1. Audio Event Recognition
2. Temporal Localization
3. Causal Reasoning
4. Object Recognition
5. Emotional Recognition
6. Scene Understanding
7. Audio Counting
8. Spatial Reasoning
9–26. *(not enumerated in paper, README, or project page — requires downloading dataset)*

**Notes:**
- Original data format: `worldsense_qa.json` — dict keyed by video ID with nested `task1`, `task2`, ... sub-objects
- Schema confirmed from `/OmniZip-main/eval/eval.py` (lines 92–107)
- 80 expert annotators, 1,662 audio-visual synchronized videos

**Sample rows used:**
- Row 1: video_id=ws_0042, domain=Daily Life, task_type=Audio Event Recognition, task_domain=Audio
- Row 2: video_id=ws_0117, domain=Sports, task_type=Causal Reasoning, task_domain=Audio-Visual

---

## 4. OmniVideoBench (`NJU-LINK/OmniVideoBench`)
**1,000 QA pairs · 628 videos · 13 reasoning types · 8 video types**
**⚠️ Gated dataset — login + terms agreement required on HuggingFace**

| Field | Type | Description |
|---|---|---|
| `video` | string | Video identifier (e.g. "video_047") |
| `video_type` | string (8 types) | Vlog / News / Cartoon / Sports / Documentary / TV / Ego / Others |
| `duration` | string (MM:SS) | Video length — range: 4s to ~32 min (1,955s) |
| `audio_type` | string (3 types) | Speech / Sound / Music |
| `question_type` | string (13 types) | See categories below |
| `question` | string | Question text (avg. 14.68 words) |
| `options` | list[string] (4 items) | Multiple-choice options [A, B, C, D] |
| `correct_option` | string (A/B/C/D) | Correct answer letter |
| `answer` | string | Full answer text (avg. 4.92 words) |
| `reasoning_steps` | list[object] | Step-by-step multimodal reasoning chain with `modality` (vision/audio), `evidence`, `inference` sub-fields |

**`question_type` categories (13):**
1. Fine-grained Perception
2. Spatial Reasoning
3. Attribute Comparison
4. Background & Music Understanding
5. Counting
6. Temporal Understanding
7. Summarization
8. Sentiment Analysis
9. Causal Reasoning
10. Relationship Reasoning
11. Reference Reasoning
12. Ego Reasoning
13. Hypothetical Reasoning

**Sample rows used:**
- Row 1: video=video_047, video_type=TV, question_type=Spatial Reasoning, audio_type=Speech
- Row 2: video=video_010, video_type=Cartoon, question_type=Causal Reasoning, audio_type=Speech

---

## 5. ShortVid-Bench (`TencentARC/ShortVid-Bench`)
**1,000 QA pairs · Short user-generated videos · 6 comprehension dimensions · Apache-2.0**
**⚠️ HuggingFace viewer broken (video format error) — exact Parquet column names unconfirmed**

| Field | Type | Description |
|---|---|---|
| `video` | binary | Raw short video content (visual + audio) |
| `question` | string | Multiple-choice question text |
| `options` | list[string] (4 items) | Answer choices [A, B, C, D] with challenging distractors |
| `answer` | string (A/B/C/D) | Human-annotated correct answer letter |
| `dimension` | string (6 types) | See categories below |

**`dimension` categories (6):**
1. Temporal Reasoning and Localization
2. Affective Intent Classification
3. Creator Intent Taxonomy
4. Narrative Comprehension
5. Humor & Meme Deconstruction
6. Creative Innovation Analysis

**Notes:**
- MCQ format; questions go beyond descriptive captioning, targeting context, intent, and narrative understanding
- Both visual and audio cues required for answering
- Automated pipeline for question generation + human annotation
- Evaluation uses 1 FPS / 150 frames per video; original paper used 400-sample subset

**Sample rows used:**
- (viewer broken — no sample rows available)
