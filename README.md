# Characterising Latent-Space Adversarial Training for LLM Safety

Code, results and logs for the Oxford MSc thesis and the NeurIPS 2026 (Agents in the Wild) workshop paper, both named _Characterising Latent-Space Adversarial Training for LLM Safety_. The work studies **continuous (embedding-space) adversarial training (CAT)** and **latent adversarial training (LAT)** of LLMs (Zephyr-7B-β, a Mistral-7B fine-tune, and Llama-3-8B-Instruct). It looks at robustness to jailbreak attacks, utility, and over-refusal of benign prompts.

**Trained models (LoRA adapters):** https://huggingface.co/collections/maria715/masters-thesis (also listed at https://huggingface.co/maria715). The models used in the thesis and paper are listed in [`commands/All_final_used_models.txt`](commands/All_final_used_models.txt); the collection also contains exploratory runs (`diag_*`).

---

## Repository map

The repository combines several public research codebases, used as starting points and modified, together with new analysis code. Each forked sub-folder keeps its **original authors' README**. The table below says what each folder is and what it is used for.

| Folder | Origin | What it is used for in the thesis |
|---|---|---|
| [`Continuous-AdvTrain/`](Continuous-AdvTrain) | Fork of [Xhonneux et al., 2024](https://arxiv.org/abs/2405.15589) | **Main CAT training code** (`train_CAT.sh`, `config/`, `src/`), with added attack-trace tracking, inner-loop analysis (`analyse_attack_traces.py`, `embedding_nn_analysis/`, `posthoc_bridging/`) and the run lists (`to_run*.csv`). Contains the training slurm logs. |
| [`Reachable-Continuous-AdvTrain/`](Reachable-Continuous-AdvTrain) | Variant of the above (Work In Progress) | **Reachable CAT (RCAT)**: CAT with nearest-neighbour projection onto the token-embedding set. Plots in `RCAT_plots/`, `meaningful_plots/`. |
| [`latent-adversarial-training/`](latent-adversarial-training) | Fork of [Sheshadri et al., 2024](https://arxiv.org/abs/2407.15549) | **LAT training** (`train_LAT.sh`). `tasks/` holds the datasets used by LAT. |
| [`MixAT/`](MixAT) | Fork of [MixAT (Dékány et al., 2025)](https://arxiv.org/abs/2505.16947) | **Evaluation harness**: HarmBench robustness (`robustness_eval_harmbench/`, `harmbench/`), harmless/over-refusal evaluation (`eval_harmless/`), utility benchmarks (`utility_eval/`, `lm-evaluation-harness/`), MT-Bench (`fastchat/`). |
| [`circuit-breakers-eval/`](circuit-breakers-eval) | Fork of [Schwinn & Geisler, 2024](https://arxiv.org/abs/2407.15902) ([original repo](https://github.com/SchwinnL/circuit-breakers-eval)) | **Embedding-space attacks** against the trained models; ε-vs-ASR perturbation studies (`epsilon_asr_perturbation*/`, `results/`). |
| [`XSTest/`](XSTest) | Fork of [XSTest](https://aclanthology.org/2024.naacl-long.301/) | **Over-refusal evaluation** (exaggerated safety). Completions in `model_completions/`, scores in `results/`. |
| [`OR-Bench/`](OR-Bench) | Scripts around [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) | Building the over-refusal train/eval subsets (`or-bench-from80kto1k_*.csv`) and generating reference answers. |
| [`AdversariaLLM/`](AdversariaLLM) | Fork of [AdversariaLLM](https://arxiv.org/abs/2511.04316) | Additional discrete attacks and judges. Hydra run outputs in `multirun/` and `outputs/`, organised by date. |
| [`I-GCG/`](I-GCG) | Fork of [I-GCG](https://arxiv.org/abs/2405.21018) | GCG-style discrete jailbreak attacks (`results/`). |
| [`representational_analysis/`](representational_analysis) | New | Extracting and comparing hidden representations of trained checkpoints. |
| [`experiments/`](experiments) | New | **Aggregated analyses** of the CAT attack traces / embedding-space geometry, produced by `Continuous-AdvTrain/analyse_attack_traces.py`. Most sub-folders have `tables/`, `plots/`, `per_run/` and a `report.txt`. |
| [`plots_and_charts/`](plots_and_charts) | New | **Final figures for the thesis**, each with the script that produced it: `Main/`, `Methods/`, `Inner-loop/`, `LAT/`, `OR/`, `Workshop_submission/`, `Workshop_poster/`. |
| [`commands/`](commands) | New | Commands used to launch training/evaluation jobs on the cluster (`commands*.txt`, `commands_lat.md`) and the list of final models used in the thesis (`All_final_used_models.txt`). |
| [`repo_management/`](repo_management) | New | Housekeeping scripts, not part of the experiments: uploading the trained adapters to Hugging Face and verifying them (`upload_models.py`, `upload_missing_logs.py`, `verify_uploads.py`), inventorying model folders (`inspect_folders.py`) and checking file sizes before pushing to GitHub (`git_size_report.py`). |

## Naming of runs and models

Run folders, slurm logs and Hugging Face models share the same names, for example `CAT_llama3b_likeZephyr_eps0150_42_relativelr_utility_500_NEW`:

- `CAT_` / `lat_` / `RCAT_`: training method; `original_lat_llama3` = LAT with its released configuration
- `zephyr`, `adv`: Zephyr-7B-β; `llama3b`: Llama-3-8B-Instruct. `github` = reproduction of the released Zephyr CAT configuration; `likeZephyr` / `likeLlama2` = Llama 3 with hyper-parameters copied from that setting (untagged Llama 3 runs use the Llama-2 ones, `otherparams` the Zephyr ones)
- `eps0150` / `0150`: CAT radius multiplier s = 0.150 (radius ε = s × mean token-embedding norm); LAT `eps6` … `eps48`: radius ε
- `42`, `123`, `456`, ...: random seed (42 if absent)
- `relativelr`: CAT attack step size proportional to ε (fixed otherwise); `rel10`: LAT inner learning rate ε/640, so the first step is ≈ 0.1·ε (otherwise scaled from the released 10⁻³ at ε = 6)
- `utility_N`: N UltraChat utility examples replaced by OR-Bench over-refusal examples (N = 1000 if absent; no `utility` = UltraChat only); `weightX`: loss weight on those examples (`weight00595` = 0.0595)
- `REFAIT` ("redone"), `NEW`, `correcteps`, `TEST`: re-runs that replace earlier versions
- `diag_*`: exploratory LAT runs, not used in the final results

## Logs and compressed files

Slurm job logs (`slurm-<jobid>.out` / `.err`) often contain the **only record of some results**, such as per-step attack metrics and evaluation scores. Most logs over 5 MB, and the few data files over 50 MB, are stored **gzip-compressed** with `.gz` added to the name (for example `slurm-96007.out.gz`). The content is identical. To read them:

```bash
zcat Continuous-AdvTrain/slurm-96007.out.gz | less        # Linux / macOS / Git Bash
gunzip -k file.out.gz                                      # decompress, keeping the .gz
```
```python
import pandas as pd; pd.read_csv("experiments/.../all_attack_level_chosen_vs_final.csv.gz")
```
On Windows, 7-Zip opens `.gz` files directly.

## Running the code

Most sub-projects have their own dependency files (usually `requirements.txt`; `MixAT/` uses `requirements/` and `environments/`, `AdversariaLLM/` uses `pyproject.toml`), and the forked ones keep their original README with setup instructions. Training and evaluation were run on a SLURM GPU cluster through the `*.sh` scripts in each folder. Paths in configs and scripts refer to that cluster (`/auto/users/...`) and need adapting.

**Note:** the committed `Continuous-AdvTrain/src/data.py` always mixes OR-Bench data into CAT training for Zephyr and Llama 3 (the "Adding OR data" block in `load_adversarial_training_data`). Runs without `utility` in their name were trained without it: to reproduce them, comment out that whole block so no OR-Bench data is included.
