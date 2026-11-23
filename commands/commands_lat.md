### \# After training



rsync -azP --exclude '.venv' --exclude '\_\_pycache\_\_' --exclude '.git' --exclude 'archive' --exclude tasks ms25mds@oat0.cs.ox.ac.uk:\~/MSc\_Project/latent-adversarial-training/experiments/ ./Code/experiments



rsync -azP --exclude '.venv' --exclude '\_\_pycache\_\_' --exclude '.git' --exclude 'experiments' --exclude 'archive' --exclude tasks ms25mds@oat0.cs.ox.ac.uk:\~/MSc\_Project/latent-adversarial-training/slurm-\* ./Code/latent-adversarial-training



mv \* ../../experiments/

### 

### \# HuggingFace



sbatch run.sh "LLM-LAT/robust-llama3-8b-instruct" None "lat\_llama3\_huggingface" 0



sbatch ./../utility\_eval/lm-evaluation-harness/run\_utility\_test.sh "LLM-LAT/robust-llama3-8b-instruct" 0



sbatch xstest\_eval.sh --model\_path "LLM-LAT/robust-llama3-8b-instruct" --output "results/lat\_llama3\_huggingface.jsonl"



sbatch ./../robustness\_eval\_harmbench/generate\_samples.sh "GCG" "lat\_llama3\_huggingface" 0

(NEED 2 GPUs) sbatch ./../robustness\_eval\_harmbench/generate\_samples.sh "AutoDAN" "lat\_llama3\_huggingface" 0

sbatch ./../robustness\_eval\_harmbench/evaluate\_samples.sh "GCG" "lat\_llama3\_huggingface" 0

sbatch ./../robustness\_eval\_harmbench/evaluate\_samples.sh "AutoDAN" "lat\_llama3\_huggingface" 0



sbatch run\_embedding\_attack.sh --model "LLM-LAT/robust-llama3-8b-instruct" --logging\_path "results/lat\_llama3\_huggingface\_42" --seed 42







### \# GitHub versions



sbatch train\_LAT.sh --logging\_path ./experiments/original\_lat\_llama3\_seed42 --seed 42



sbatch run.sh "meta-llama/Meta-Llama-3-8B-Instruct" "../../latent-adversarial-training/experiments/original\_lat\_llama3\_seed42/final\_model" "original\_lat\_llama3\_seed42" 0

sbatch run.sh "meta-llama/Meta-Llama-3-8B-Instruct" "../../latent-adversarial-training/experiments/original\_lat\_llama3\_seed123/final\_model" "original\_lat\_llama3\_seed123" 0

sbatch run.sh "meta-llama/Meta-Llama-3-8B-Instruct" "../../latent-adversarial-training/experiments/original\_lat\_llama3\_seed456/final\_model" "original\_lat\_llama3\_seed456" 0



sbatch ./../utility\_eval/lm-evaluation-harness/run\_utility\_test\_peft.sh "meta-llama/Meta-Llama-3-8B-Instruct" "../../latent-adversarial-training/experiments/original\_lat\_llama3\_seed42/final\_model" 0

sbatch ./../utility\_eval/lm-evaluation-harness/run\_utility\_test\_peft.sh "meta-llama/Meta-Llama-3-8B-Instruct" "../../latent-adversarial-training/experiments/original\_lat\_llama3\_seed123/final\_model" 0

sbatch ./../utility\_eval/lm-evaluation-harness/run\_utility\_test\_peft.sh "meta-llama/Meta-Llama-3-8B-Instruct" "../../latent-adversarial-training/experiments/original\_lat\_llama3\_seed456/final\_model" 0



sbatch xstest\_eval.sh --model\_path "meta-llama/Meta-Llama-3-8B-Instruct" --adapter "../experiments/original\_lat\_llama3\_seed42/final\_model" --output "results/original\_lat\_llama3\_seed42.jsonl"

sbatch xstest\_eval.sh --model\_path "meta-llama/Meta-Llama-3-8B-Instruct" --adapter "../experiments/original\_lat\_llama3\_seed123/final\_model" --output "results/original\_lat\_llama3\_seed123.jsonl"

sbatch xstest\_eval.sh --model\_path "meta-llama/Meta-Llama-3-8B-Instruct" --adapter "../experiments/original\_lat\_llama3\_seed456/final\_model" --output "results/original\_lat\_llama3\_seed456.jsonl"



sbatch ./../robustness\_eval\_harmbench/generate\_samples\_newpeft.sh "GCG" "original\_lat\_llama3\_seed42" 0

sbatch ./../robustness\_eval\_harmbench/generate\_samples\_newpeft.sh "GCG" "original\_lat\_llama3\_seed123" 0

sbatch ./../robustness\_eval\_harmbench/generate\_samples\_newpeft.sh "GCG" "original\_lat\_llama3\_seed456" 0



(NEED 2 GPUs) sbatch ./../robustness\_eval\_harmbench/generate\_samples\_newpeft.sh "AutoDAN" "original\_lat\_llama3\_seed42" 0

(NEED 2 GPUs) sbatch ./../robustness\_eval\_harmbench/generate\_samples\_newpeft.sh "AutoDAN" "original\_lat\_llama3\_seed123" 0

(NEED 2 GPUs) sbatch ./../robustness\_eval\_harmbench/generate\_samples\_newpeft.sh "AutoDAN" "original\_lat\_llama3\_seed456" 0



sbatch ./../robustness\_eval\_harmbench/evaluate\_samples\_newpeft.sh "GCG" "original\_lat\_llama3\_seed42" 0

sbatch ./../robustness\_eval\_harmbench/evaluate\_samples\_newpeft.sh "GCG" "original\_lat\_llama3\_seed123" 0

sbatch ./../robustness\_eval\_harmbench/evaluate\_samples\_newpeft.sh "GCG" "original\_lat\_llama3\_seed456" 0



sbatch ./../robustness\_eval\_harmbench/evaluate\_samples\_newpeft.sh "AutoDAN" "original\_lat\_llama3\_seed42" 0

sbatch ./../robustness\_eval\_harmbench/evaluate\_samples\_newpeft.sh "AutoDAN" "original\_lat\_llama3\_seed123" 0

sbatch ./../robustness\_eval\_harmbench/evaluate\_samples\_newpeft.sh "AutoDAN" "original\_lat\_llama3\_seed456" 0



sbatch run\_embedding\_attack\_newpeft.sh --model "meta-llama/Meta-Llama-3-8B-Instruct" --adapter "../experiments/original\_lat\_llama3\_seed42/final\_model" --logging\_path "results/original\_lat\_llama3\_seed42\_42" --seed 42

sbatch run\_embedding\_attack\_newpeft.sh --model "meta-llama/Meta-Llama-3-8B-Instruct" --adapter "../experiments/original\_lat\_llama3\_seed123/final\_model" --logging\_path "results/original\_lat\_llama3\_seed123\_42" --seed 42

sbatch run\_embedding\_attack\_newpeft.sh --model "meta-llama/Meta-Llama-3-8B-Instruct" --adapter "../experiments/original\_lat\_llama3\_seed456/final\_model" --logging\_path "results/original\_lat\_llama3\_seed456\_42" --seed 42

