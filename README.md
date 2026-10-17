# Steering Causal Effect through Data Preparation Pipeline Search

## Overview

Data preparation is a fundamental stage in data science workflows, yet in observational causal analysis, even standard data preparation choices can substantially alter estimated effects. This project presents a general framework that automatically constructs data preparation pipelines capable of **steering an estimated causal effect** into a user-specified target range. 

The framework utilizes probability-guided search abilities, allowing the search space to be navigated based on uniform or action-specific probability weights.

## Search Algorithms

The framework provides several strategies to navigate the exponential search space of pipelines:

### 1. Exact & Pruned Search
* **Brute Force (`brute`)**: Explores all possible combinations. Explores exact sequences to find a pipeline, optimizing strictly on **sequence length** (guaranteeing the shortest sequence to reach the target ATE).
* **Pruned (`OE`)**: An optimized version of brute force. It leverages **Observational Equivalence** to prune sequences that produce identical data states, significantly reducing redundant computations while maintaining the exact search guarantees.

### 2. Probabilistic Search (TAPS)
* **Probe (`probe`)**: A unified probability-guided search algorithm that navigates large search spaces efficiently. By tweaking its configuration flags (`use_restart`, `op_probs`, `is_brute`), it supports multiple operational modes:
    * **Standard TAPS**: Uses explicit action probability weights without a restart mechanism.
    * **TAPS-JIT**: Uses explicit action probability weights combined with a restart mechanism to escape local minima.
    * **TAPS-JIT-UNIFORM**: Uses a completely uniform probability distribution across all actions combined with a restart mechanism (`use_restart=True`, `op_probs=None`, `is_brute=False`).
    * **Probability-Based Brute**: Leverages the probe infrastructure with specialized configuration flags to perform an exact, probability-driven exploration guaranteeing the most probable sequence to reach the target ATE.

### 3. LLM Search
* **LLM Variants**: Leverages Large Language Models to guide the synthesis via:
    * `llm_zero_shot`
    * `llm_few_shot`
    * `llm_cot` (Chain of Thought)

### 4. Baselines & Automl
* **Greedy (`greedy`)**: A deterministic baseline that selects the single most probable action at each step based on the provided probability distribution.
* **Random (`random`)**: Explores sequences of a fixed length randomly.
* **Auto-Sklearn**: Includes both a vanilla version and a specialized **ATE-based score function** version.

---

## Repository Structure

* `main.py`: The primary entry point for executing searches.
* `experiments.py`: Contains all experiment definitions, target ATEs, weight matrices, and configurations.
* `data_loader.py`: Specialized classes for loading and preprocessing the four core datasets.
* `utils.py`: The library of available data transformation functions used during the search.
* `search_methods/`: Holds the search method classes. Each search method has a `search` function that searches for a sequence that steers the ATE into a given range, respecting max sequence length.
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
* `op_probs`: Dictionary of action probabilities (defaults to uniform if not provided).
* `whole_df_ops`: Operations applied at the table granularity level.
* `legal_ops_by_type`: Mapping defining which operations are applicable to each column type.
* `legal_fill_by_type`: Mapping defining validation rules for missing value imputation based on column type.

## Running Experiments

Experiments are configured in `experiments.py`. Each experiment definition includes the dataframe, the transformation dictionary, confounder columns, the **target ATE**, **epsilon** (tolerance), and **max_length** (search depth).

### Command Syntax
To run an experiment, provide the Experiment ID and the desired Algorithm Name to `main.py`:

```bash
python main.py <EXP_ID> <ALG_NAME>
```
