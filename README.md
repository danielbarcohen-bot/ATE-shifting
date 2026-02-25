# Steering Causal Effect through Data Preparation Pipeline Search

## Overview

Data preparation is a fundamental stage in data science workflows, yet in observational causal analysis, even standard data preparation choices can substantially alter estimated effects. This project presents a general framework that automatically constructs data preparation pipelines capable of **steering an estimated causal effect** into a user-specified target range.

## Search Algorithms

The framework provides several strategies to navigate the exponential search space of pipelines:

### 1. Exact & Pruned Search
* **Brute Force (`brute`)**: Explores all possible combinations. **Guarantees the shortest sequence** to reach the target ATE, but has high computational complexity.
* **Pruned (`OE`)**: An optimized version of brute force. It leverages **Observational Equivalence** to prune sequences that produce identical data states, significantly reducing redundant computations while maintaining the shortest-path guarantee.

### 2. Probabilistic & LLM Search
* **Probe (`probe`)**: Uses probability-guided search to find a solution. While it does not guarantee the shortest path, it offers a **very fast search** suitable for large spaces.
* **LLM Variants**: Leverages Large Language Models to guide the synthesis via:
    * `llm_zero_shot`
    * `llm_few_shot`
    * `llm_cot` (Chain of Thought)

### 3. Baselines & Automl
* **Random (`random`)**: Explores sequences of a fixed length randomly.
* **Auto-Sklearn**: Includes both a vanilla version and a specialized **ATE-based score function** version.
---

## Repository Structure

* `main.py`: The primary entry point for executing searches.
* `experiments.py`: Contains all experiment definitions, target ATEs, and configurations.
* `data_loader.py`: Specialized classes for loading and preprocessing the four core datasets.
* `utils.py`: The library of available data transformation functions used during the search.
* `search_methods`: Holds the search method classes. each search method has function search that searches sequence that steers the ATE into given range, respecting max sequence length.
* `prompts.py`: Contains the different prompts to the LLM.
---

## Data & Transformations

### Supported Datasets
The framework includes loading classes for four real-world datasets in `data_loader.py`:
1.  **Twins**: Mortality data for twin births in the USA.
2.  **ACS**: American Community Survey data.
3.  **Lalonde**: Employment data from the National Supported Work program.
4.  **IHDP**: Infant Health and Development Program.

### Transformation Logic
Transformations are treated as the "building blocks" of the synthesized programs.
* **Location:** All functional logic for transformations resides in `utils.py`.
* **Definition:** Each experiment includes a dictionary mapping the **Transformation Name** to its corresponding **Function**.

---
### Experiment Configuration
Experiments are defined in `experiments.py`. Each experiment requires the following parameters:
* `df`: The input dataset.
* `transformations_dict`: Dictionary of available functions.
* `common_causes`: List of columns to treat as confounders.
* `target_ate`: The goal ATE.
* `epsilon`: The tolerance range around the target.
* `max_length`: Maximum number of transformations allowed.
* `sequence_length`: (Required only for `random` baseline experiments).


## Running Experiments

Experiments are configured in `experiments.py`. Each experiment definition includes the dataframe, the transformation dictionary, confounder columns, the **target ATE**, **epsilon** (tolerance), and **max_length** (search depth).

### Command Syntax
To run an experiment, provide the Experiment ID and the desired Algorithm Name to `main.py`:

```bash
python main.py <EXP_ID> <ALG_NAME>