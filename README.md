# Masters Thesis: Fitness vs. Novelty
## Abstract
This thesis investigates the use of novelty in evolutionary algorithms on reinforcement learning tasks. We compare novelty with fitness based selection, their archival variants and dynamic combinations of novelty and fitness on Gymnasium environments CartPole and Lunar Lander. For each environment we define an interpretable behavioural characteristic and examine how the choice of objective handling method affects the resulting fitness and population diversity.
The results show the fitness-based methods achieve best performance under the experiment conditions. The novelty increases population diversity but its benefit depends on specific task dynamics. 
The thesis also discusses sensitivity of novelty search to the choice of behavioural space and interactions between the environment and the evolutionary algorithm.

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
├── diff_cont.py                    # DE evaluation
├── final_examination.py            # Final evaluation 
├── generation_examination.py       # Generation evaluation
├── lambda_cont.py                  # ES evaluation
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


## Final Examination 

This script executes the final evaluation of an evolutionary algorithm on a selected environment and container configuration.

For each experiment, a sequence of random seeds is generated allowing the algorithm to be evaluated over multiple independent runs. The results are saved to output location for later analysis.

### Usage

```bash
python final_examination.py -e lunarlander -a lambda -c $i -s 45 -o $OUTPUT -R list
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

## Generation Examination 

This script evaluates an evolutionary algorithm at selected generations during training. It executes multiple independent runs using a sequence of random seeds and records the algorithm's performance of specified lenghts of the algorithm run. 

The generations to evaluate are either predifined for the environemnt or defined by by a start value, end value, and step size depending on the type of output. Results are saved to the specified output location. Type decides whether the writes a new protocol or adds to existing one.

### Usage

```bash
 python generation_examination.py -e cartpole -a lambda -c $i -o $OUTPUT -R grid_search 
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-e`, `--environment` | Environment to evaluate. |
| `-a`, `--algorithm` | Evolutionary algorithm to execute. |
| `-c`, `--container` | Objective/container configuration. |
| `-o`, `--name` | Output file or directory for generated results. |
| `-t`, `--type_out` | Type of generated output. |
| `-l`, `--gen_start` | First generation to evaluate. |
| `-u`, `--gen_end` | Upper bound (exclusive) of the evaluated generation range. |
| `-p`, `--gen_step` | Step size between evaluated generations. |
| `-s`, `--seeds` | Number of independent runs. 
| `-R`, `--source` | Source from which the experiment configuration is loaded. |

## Incremental Grid Search

This module peroforms incremental grid search as described in the text of the thesis.
The evaluation function returns the mean best fitness and mean behavioural diversity across all seeds.

### Main function

```python adaptive_gridsearch.py -e lunarlander -a diff -c novelty_archiving -o $OUTPUT -r bash_run -H 3

```

### Parameters

| Argument | Description |
|----------|-------------|
| `-e`, `--environment` | Environment in which the adaptive grid search is performed. |
| `-a`, `--algorithm` | Evolutionary algorithm to optimize. |
| `-r`, `--run-name` | Name of the grid search experiment. | 
| `-c`, `--container` | Objective/container configuration to evaluate. 
| `-H`, `--hops` | Number of adaptive grid search iterations (hops). |
| `-o`, `--out-path` | Directory where the search results are stored. | 
| `-S`, `--source` | Source from which the initial configuration is obtained. |

