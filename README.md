# MSc Thesis: Continuous Adversarial Training for LLMs: Robustness, Utility and Over-refusal

Code, results and logs for my Oxford MSc thesis named _Characterising Latent-Space Adversarial Training for LLM Safety_. The thesis studies **continuous (embedding-space) adversarial training (CAT)** and **latent adversarial training (LAT)** of LLMs (Zephyr-7B, Llama-3-8B, Mistral-7B). It looks at robustness to jailbreak attacks, utility, and over-refusal of benign prompts.

**Trained models (LoRA adapters):** https://huggingface.co/collections/maria715/masters-thesis (also listed at https://huggingface.co/maria715)

---

## Repository map

The repository combines several public research codebases, which I used as starting points and modified, with my own analysis code. Each sub-folder keeps its **original authors' README**. The table below says what each folder is and what I used it for.

| Folder | Origin | What it is used for in the thesis |
|---|---|---|
| [`Continuous-AdvTrain/`](Continuous-AdvTrain) | Fork of [Xhonneux et al., 2024](https://arxiv.org/abs/2405.15589) | **Main CAT training code** (`train_CAT.sh`, `config/`, `src/`), with my additions for attack-trace tracking, inner-loop analysis (`analyse_attack_traces.py`, `embedding_nn_analysis/`, `posthoc_bridging/`) and the run lists (`to_run*.csv`). Contains the training slurm logs. |
| [`Reachable-Continuous-AdvTrain/`](Reachable-Continuous-AdvTrain) | My variant of the above (Work In Progress) | **Reachable CAT (RCAT)**: CAT with nearest-neighbour projection onto the token-embedding set. Plots in `RCAT_plots/`, `meaningful_plots/`. |
| [`latent-adversarial-training/`](latent-adversarial-training) | Fork of [Sheshadri et al., 2024](https://arxiv.org/abs/2407.15549) | **LAT training** (`train_LAT.sh`). `tasks/` holds the datasets used by LAT. |
| [`MixAT/`](MixAT) | Fork of the MixAT codebase | **Evaluation harness**: HarmBench robustness (`robustness_eval_harmbench/`, `harmbench/`), harmless/over-refusal evaluation (`eval_harmless/`), utility benchmarks (`utility_eval/`, `lm-evaluation-harness/`), MT-Bench (`fastchat/`). |
| [`circuit-breakers-eval/`](circuit-breakers-eval) | Fork of an independent Circuit Breakers robustness evaluation | **Embedding-space attacks** against the trained models; ε-vs-ASR perturbation studies (`epsilon_asr_perturbation*/`, `results/`). |
| [`XSTest/`](XSTest) | Fork of [XSTest](https://aclanthology.org/2024.naacl-long.301/) | **Over-refusal evaluation** (exaggerated safety). Completions in `model_completions/`, scores in `results/`. |
| [`OR-Bench/`](OR-Bench) | My scripts around [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) | Building the over-refusal train/eval subsets (`or-bench-from80kto1k_*.csv`) and generating reference answers. |
| [`AdversariaLLM/`](AdversariaLLM) | Fork of [AdversariaLLM](https://arxiv.org/abs/2511.04316) | Additional discrete attacks and judges. Hydra run outputs in `multirun/` and `outputs/`, organised by date. |
| [`I-GCG/`](I-GCG) | Fork of [I-GCG](https://arxiv.org/abs/2405.21018) | GCG-style discrete jailbreak attacks (`results/`). |
| [`representational_analysis/`](representational_analysis) | Mine | Extracting and comparing hidden representations of trained checkpoints. |
| [`experiments/`](experiments) | Mine | **Aggregated analyses** of the CAT attack traces / embedding-space geometry. Each sub-folder has `tables/`, `plots/`, `per_run/` and a `report.txt`. |
| [`plots_and_charts/`](plots_and_charts) | Mine | **Final figures for the thesis**, each with the script that produced it: `Main/`, `Methods/`, `Inner-loop/`, `LAT/`, `OR/`, `Workshop_submission/`, `Workshop_poster/`. |
| [`commands/`](commands) | Mine | Commands used to launch training/evaluation jobs on the cluster (`commands*.txt`, `commands_lat.md`) and the list of final models used in the thesis (`All_final_used_models.txt`). |
| `upload_models.py`, `verify_uploads.py`, `upload_missing_logs.py` | Mine | Scripts used to upload the trained adapters to Hugging Face and check the uploads. |

## Naming of runs and models

Run folders, slurm logs and Hugging Face models share the same names, for example `CAT_llama3b_likeZephyr_eps0150_42_relativelr_utility_500_NEW`:

- `CAT_` / `lat_` / `RCAT`: training method
- `zephyr`, `llama3b` (Llama-3-8B), `mistral`: base model; `likeZephyr` / `likeLlama2` = hyper-parameters copied from that setting
- `eps0150` / `eps01`: perturbation budget ε (`0150` = 0.150, `01` = 0.1)
- `42`, `123`, `456`, ...: random seed
- `relativelr`: attack step size relative to ε
- `utility_500`, `utility_3000`: number of utility (benign) examples in the training mix; `weight...` = utility-loss weight
- `REFAIT` ("redone"), `NEW`: re-runs that replace earlier versions; use these

## Logs and compressed files

Slurm job logs (`slurm-<jobid>.out` / `.err`) often contain the **only record of some results**, such as per-step attack metrics and evaluation scores. Logs over 5 MB, and the few data files over 50 MB, are stored **gzip-compressed** with `.gz` added to the name (for example `slurm-96007.out.gz`). The content is identical. To read them:

```bash
zcat Continuous-AdvTrain/slurm-96007.out.gz | less        # Linux / macOS / Git Bash
gunzip -k file.out.gz                                      # decompress, keeping the .gz
```
```python
import pandas as pd; pd.read_csv("experiments/.../all_attack_level_chosen_vs_final.csv.gz")
```
On Windows, 7-Zip opens `.gz` files directly.

## Running the code

Each sub-project has its own `requirements.txt` (and the original README with setup instructions). Training and evaluation were run on a SLURM GPU cluster through the `*.sh` scripts in each folder. Paths in configs and scripts refer to that cluster (`/auto/users/...`) and need adapting.
