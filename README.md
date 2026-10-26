# DASE7506 MP2: Pairwise Ranking with Arithmetic Evidence

Student ID: **3036780955**

Official full-test accuracy: **81.15631691648822% (379/467)**.

- Report: [PDF](REPORT.pdf) / [editable Markdown](REPORT.md).
- Predictor: `submission_arithmetic.py` (with `submission.py` and `arithmetic_evidence.py`).
- Complete inference checkpoint: `checkpoint.npz`.
- Official score: [`evaluation/official-score.json`](evaluation/official-score.json).
- Timing and frozen-version record: [`evaluation/measurement.json`](evaluation/measurement.json), [`evaluation/FROZEN.json`](evaluation/FROZEN.json).
- Experiment records and all model-selection results: `records/`.

## Setup

Use Python 3.12. The original environment used Python 3.12.13. NumPy is the only inference dependency; training additionally uses scikit-learn and its dependencies.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
git clone https://github.com/xudongwu-0/DASE7506-MP2.git teacher-source
git -C teacher-source checkout bb801f4f2e7629d270df765ab12c333b2dc616bb
.venv/bin/python restore_data.py --source teacher-source
.venv/bin/python upstream/code/check_data.py
```

The restore command verifies and copies the supplied teacher data. The included teacher parser, scorer, and manifest match this pinned release. No additional model downloads are needed.

## Reproduce the submitted score

```sh
.venv/bin/python evaluate.py --split test --output-dir output/test
```

This runs the official predictor and independent scorer in fresh CPU processes, with a total library thread limit of four. It writes predictions, `official-score.json`, and full-process timing under `output/test/`. Choose a new output directory for a repeat evaluation. The optional `--checkpoint` argument selects another explicitly supplied checkpoint.

Expected score: 379/467, accuracy `81.15631691648822`. Original local timing was 0.31483 seconds on an Apple M5. The course reference machine is an Intel Xeon Platinum 8457C. The checkpoint is 26,064 bytes (25,456 bytes of uncompressed entries).

Checkpoint SHA256:

```text
3efecdf40f8420a2b547e4ac49bb67b13cb3873e2a4fc87c73a4042dbe00fc66
```

## Reproduce final-model training

```sh
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -B train.py --output output/retrained.npz
.venv/bin/python evaluate.py --split validation --checkpoint output/retrained.npz --output-dir output/retrained-validation
.venv/bin/python -B -m unittest -v test_arithmetic_evidence test_contract
```

`train.py` reproduces the final C=0.01 model directly from the training bank: 1,032 features, train-only standardization, 6,516 correct/incorrect pairs, and equal total weight per informative question. It uses seed 17, L-BFGS, tolerance 1e-6, and a 2,000-iteration cap. Expected validation result: 738/798 overall and 295/355 on disagreement questions. The training command writes a parameter checkpoint and a training-time record alongside it.

## Reproduce the experimental progression

The original scripts are retained unchanged. To run them in a new directory:

```sh
.venv/bin/python reproduce_experiments.py --output output/experiments --include-mlp
```

This repeats six initial classifiers, the C=1 and C=0.01 pairwise models, the optional MLP, the train-only arithmetic diagnostic, and the final arithmetic model. It uses validation for evaluation and leaves test scoring to the explicit command above. `records/` contains the original configurations, convergence and cost records, parser audit, paired comparisons, and error examples. Historical local directory prefixes in these records are replaced by `$PROJECT_ROOT`; current executable commands are given here.

## Method and attribution

The predictor scores fixed candidate embeddings, four candidate-context features, and four arithmetic-evidence features. It selects the highest-scoring parseable answer. Ties use the lowest candidate index, and an all-unparseable group returns index 0. Numeric labels, parser behavior, and final scoring follow the teacher package.

The report cites verifier learning (Cobbe et al.), pairwise ranking (Burges et al.), and program execution for mathematical reasoning (Gao et al.). Training uses scikit-learn and inference uses NumPy. The supplied GSM8K material and Qwen3-0.6B candidate provenance are documented in `upstream/code/data/LICENSE.txt` and `NOTICE.txt`.

Codex assisted with literature review, planning, implementation, execution, analysis, and report drafting. The report and records describe the measured results and the submitted algorithm.
