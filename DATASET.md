# Dataset

**Benchmark:** HotpotQA, *distractor* setting, **validation** split (Yang et al., EMNLP 2018). Public, CC BY-SA 4.0.

| Item | Link |
|---|---|
| Hugging Face | https://huggingface.co/datasets/hotpotqa/hotpot_qa |
| Original release | https://hotpotqa.github.io/ |
| Original JSON (dev, distractor) | http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json |

**Subset used in the experiments:** a random sample of the validation questions, seed 42, size `--n` given to
`scripts/prepare_data.py` (default 300; the evaluation then uses the first `--n` of them, default 100).
Created by `python scripts/prepare_data.py --n 300 --seed 42`, which writes
`data/hotpot_subset.json` (same JSON format as the original) and `data/DATA_CARD.json` (source, seed, question ids).
The retrieval corpus is the union of the paragraphs (gold + distractors) of the evaluated questions.

**Files in this archive**
- `data/hotpot_subset.json`, `data/DATA_CARD.json` - the exact evaluated subset (created by the script above; include them in the upload).
- `data/sample_mini.json` - tiny *synthetic* set used only for offline smoke tests. Not used for any reported result.
- `results/per_question.csv` - per-question predictions, metrics and cost for every system (created by `evaluate.py`).
- `results/llm_cache.jsonl` - cached LLM responses, so the experiment can be replayed without API calls.

**Optional own dataset:** `scripts/build_course_qa.py` builds a QA set from lecture PDFs (LLM-generated questions;
spot-check by hand before publishing it).
