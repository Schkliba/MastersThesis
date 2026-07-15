# Masters Thesis: Fitness vs. Novelty
Goal of this thesis is 
##  Project Structure

```text
.
├── Archive/                        # Archived experiments and outputs (not relevant to the thesis)
├── Data/                           # Datasets and generated data
├── Docs/                           # The thesis latex files
├── libs/                           # Shared library for evolutionary algorithms
├── notebooks/                      # helper Jupyter notebooks
├── scripts/                        # Automation bash scripts
│
├── adaptive_gridsearch.py          # Incremental grid search implementation
├── diff_cont.py                    # standalone differential evaluation
├── final_examination.py            # Final evaluation script
├── generation_examination.py       # Generation evaluation script
├── lambda_cont.py                  # Lambda controller implementation
├── optuna_experiment.py            # Optuna experiment script, auxilary channel to notebooks
├── view_episodes.py                # Episode visualization utility (infrastructure)
├── visualisation.py                # Visualization functions (infrastructure)
│
├── optuna_cartpole.ipynb           # CartPole Optuna BS experiments
├── optuna_lunarlander.ipynb        # LunarLander Optuna BS experiments
├── special_optuna_lunar_lander.ipynb # auxilary Lunar Lander Optuna BS experiment
├── study_view.ipynb                # Study analysis and visualization
│
├── relevant_finals.json            # Hyperparameters used in final experiment
├── relevant_grid_search.json       # Used gridsearch folder name for each environment 
├── relevant_studies.json           # Study name used for each of the containers
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```


## Final Examination Generator

This script executes the final evaluation of an evolutionary algorithm on a selected environment and container configuration.

For each experiment, a sequence of random seeds is generated allowing the algorithm to be evaluated over multiple independent runs. The results are saved to output location for later analysis.

### Usage

```bash
python final_examination.py \
    --environment cartpole \
    --algorithm lambda \
    --container fitness \
    --name results/cartpole \
    --seeds 30
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-e`, `--environment` | Environment to evaluate. |
| `-a`, `--algorithm` | Evolutionary algorithm to execute. |
| `-c`, `--container` | Objective/container configuration. |
| `-o`, `--name` | Output file or directory for generated results. |
| `-s`, `--seeds` | Number of independent runs. Seeds are generated in the range `101` to `101 + seeds - 1`. Default: `30`. |
| `-R`, `--source` | Source from which the experiment configuration is loaded. Default: `list`. |

## Generation Examination Generator

This script evaluates an evolutionary algorithm at selected generations during training. It executes multiple independent runs using a sequence of random seeds and records the algorithm's performance at the specified generation intervals. 

The generations to evaluate are either predifined for the environemnt or defined by by a start value, end value, and step size depending on the type of output. Results are saved to the specified output location. Type decides whether the writes a new protocol or adds to existing one.

### Usage

```bash
python generation_examination.py \
    --environment cartpole \
    --algorithm lambda \
    --container fitness \
    --name results/cartpole \
    --seeds 5
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-e`, `--environment` | Environment to evaluate. |
| `-a`, `--algorithm` | Evolutionary algorithm to execute. |
| `-c`, `--container` | Objective/container configuration. |
| `-o`, `--name` | Output file or directory for generated results. |
| `-t`, `--type_out` | Type of generated output. Default: `base`. |
| `-l`, `--gen_start` | First generation to evaluate. |
| `-u`, `--gen_end` | Upper bound (exclusive) of the evaluated generation range. |
| `-p`, `--gen_step` | Step size between evaluated generations. |
| `-s`, `--seeds` | Number of independent runs. 
| `-R`, `--source` | Source from which the experiment configuration is loaded. Default: `optuna`. |
