# README.md

# Semantic Divergence Metrics for LLM Hallucination Detection

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-e92063.svg?style=flat&logo=pydantic&logoColor=white)](https://pydantic-docs.helpmanual.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.10192-b31b1b.svg)](https://arxiv.org/abs/2508.10192)
[![Research](https://img.shields.io/badge/Research-LLM%20Evaluation-green)](https://github.com/chirindaopensource/llm_faithfulness_hallucination_misalignment_detection)
[![Discipline](https://img.shields.io/badge/Discipline-NLP%20%26%20Info.%20Theory-blue)](https://github.com/chirindaopensource/llm_faithfulness_hallucination_misalignment_detection)
[![Methodology](https://img.shields.io/badge/Methodology-Semantic%20Divergence-orange)](https://github.com/chirindaopensource/llm_faithfulness_hallucination_misalignment_detection)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/llm_faithfulness_hallucination_misalignment_detection)

**Repository:** `https://github.com/chirindaopensource/llm_faithfulness_hallucination_misalignment_detection`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models"** by:

*   Igor Halperin

The project provides a complete, end-to-end computational framework for detecting faithfulness hallucinations (confabulations) in Large Language Models (LLMs). It moves beyond traditional prompt-agnostic methods by introducing a prompt-aware, ensemble-based approach that measures the semantic consistency of LLM responses across multiple, semantically equivalent paraphrases of a user's query. The goal is to provide a transparent, robust, and computationally efficient toolkit for researchers and practitioners to replicate, validate, and apply the Semantic Divergence Metrics (SDM) framework.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: execute_sdm_analysis](#key-callable-execute_sdm_analysis)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models." The core of this repository is the iPython Notebook `faithfulness_hallucination_misalignment_detection_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial configuration validation to the final calculation of the SDM scores and a full suite of robustness checks.

Traditional hallucination detection methods often measure the diversity of answers to a single, fixed prompt. This can fail to distinguish between a healthy, multifaceted answer and a genuinely unstable, confabulatory one. This project implements the SDM framework, which introduces a more rigorous, prompt-aware methodology.

This codebase enables users to:
-   Rigorously validate and structure a complete experimental configuration using Pydantic.
-   Automatically generate a high-quality corpus of semantically equivalent prompt paraphrases.
-   Efficiently generate a matrix of LLM responses using fault-tolerant, asynchronous API calls.
-   Transform the raw text corpus into a shared semantic topic space via joint embedding and hierarchical clustering.
-   Calculate a full suite of information-theoretic (JSD, KL Divergence) and geometric (Wasserstein Distance) metrics.
-   Aggregate these metrics into the final, interpretable scores for **Semantic Instability ($S_H$)** and **Semantic Exploration (KL)**.
-   Execute a full suite of robustness checks to validate the stability of the framework itself.

## Theoretical Background

The implemented methods are grounded in information theory, statistics, and natural language processing, providing a quantitative framework for measuring the alignment between a prompt and a response.

**1. Ensemble-Based Testing:**
The core innovation is to test for a deeper form of arbitrariness. Instead of just generating $N$ answers to a single prompt $Q$, the framework first generates $M$ semantically equivalent paraphrases $\{Q_1, ..., Q_M\}$. Then, for each $Q_m$, it generates $N$ answers. This $M \times N$ response matrix allows for the measurement of consistency across both multiple answers *and* multiple prompt phrasings.

**2. Joint Semantic Clustering:**
All sentences from both the $M$ prompts and the $M \times N$ answers are embedded into a common high-dimensional vector space. A single clustering algorithm (Hierarchical Agglomerative Clustering with Ward's linkage) is applied to this joint set of embeddings. This creates a shared, discrete "topic space" where semantically similar sentences are assigned the same topic label, regardless of whether they came from a prompt or a response.

**3. Semantic Divergence Metrics:**
From the topic assignments, topic probability distributions are created for the prompts ($P$) and the answers ($A$). The divergence between these is quantified using:
-   **Jensen-Shannon Divergence ($D_{JS}$):** A symmetric, bounded measure of the dissimilarity between the prompt and answer topic distributions.
    $$ D_{JS}(P||A) = \frac{1}{2}(D_{KL}(P||M) + D_{KL}(A||M)), \quad M = \frac{1}{2}(P+A) $$
-   **Wasserstein Distance ($W_d$):** A measure of the geometric shift between the raw embedding clouds, capturing changes in meaning that might not be reflected in the topic distributions.
-   **Kullback-Leibler (KL) Divergence ($D_{KL}$):** An asymmetric measure of "surprise." The paper identifies $D_{KL}(A||P)$ as a powerful indicator of **Semantic Exploration**—the degree to which the LLM must introduce new concepts not present in the prompt.

**4. Final Aggregated Scores:**
These components are combined into the final, normalized scores:
-   **Semantic Instability ($S_H$):** The primary hallucination score.
    $$ S_H = \frac{w_{jsd} \cdot D_{JS}^{ens} + w_{wass} \cdot W_d}{H(P)} $$
-   **Semantic Exploration (KL Score):**
    $$ KL(\text{Answer| |Prompt}) = \frac{D_{KL}^{ens}(A || P)}{H(P)} $$

## Features

The provided iPython Notebook (`faithfulness_hallucination_misalignment_detection_draft.ipynb`) implements the full research pipeline, including:

-   **Configuration Pipeline:** A robust, Pydantic-based validation system for all experimental parameters.
-   **High-Performance Data Generation:** Asynchronous API calls for efficient generation of the paraphrase and response corpora, with built-in fault tolerance and retry logic.
-   **Rigorous Analytics:** Elite-grade, modular functions for each stage of the analysis, from embedding and clustering to the final metric calculations, leveraging optimized libraries like `scipy` and `scikit-learn`.
-   **Automated Orchestration:** A master function that runs the entire end-to-end workflow with a single call.
-   **Comprehensive Validation:** A full suite of robustness checks to analyze the framework's sensitivity to hyperparameters, model substitutions, and statistical noise.
-   **Full Research Lifecycle:** The codebase covers the entire research process from configuration to final, validated scores, providing a complete and transparent replication package.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Configuration Validation (Task 1):** The pipeline ingests a configuration dictionary and rigorously validates its schema, constraints, and content.
2.  **Environment Setup (Task 2):** It establishes a deterministic, reproducible computational environment and initializes all models and clients.
3.  **Paraphrase Generation (Task 3):** It generates and validates `M` semantically equivalent paraphrases of the original prompt.
4.  **Response Generation (Task 4):** It generates and validates an `M x N` matrix of responses.
5.  **Sentence Segmentation (Task 5):** It deconstructs all texts into a cataloged, sentence-level corpus.
6.  **Embedding Generation (Task 6):** It transforms the sentence corpus into a validated, high-dimensional vector space.
7.  **Clustering (Task 7):** It determines the optimal number of topics (`k*`) and partitions the embedding space into `k*` clusters.
8.  **Distribution Construction (Task 8):** It translates the discrete cluster labels into numerically stable probability distributions.
9.  **Metric Computation (Tasks 9-10):** It calculates the full suite of information-theoretic and geometric metrics.
10. **Score Aggregation (Task 11):** It synthesizes all intermediate metrics into the final, interpretable SDM scores and validates them against paper benchmarks.
11. **Orchestration & Robustness (Tasks 12-13):** Master functions orchestrate the main pipeline and the optional, full suite of robustness checks.

## Core Components (Notebook Structure)

The `faithfulness_hallucination_misalignment_detection_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 13 tasks.

## Key Callable: execute_sdm_analysis

The central function in this project is `execute_sdm_analysis`. It orchestrates the entire analytical workflow, providing a single entry point for either a standard analysis or a full robustness study.

```python
def execute_sdm_analysis(
    experiment_config: Dict[str, Any],
    perform_robustness_checks: bool = False
) -> Dict[str, Any]:
    """
    Executes the main SDM analysis pipeline and optionally a full suite of robustness checks.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.8+
-   An OpenAI API key set as an environment variable (`OPENAI_API_KEY`).
-   Core dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pydantic`, `openai`, `sentence-transformers`, `nltk`, `tenacity`, `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/llm_faithfulness_hallucination_misalignment_detection.git
    cd llm_faithfulness_hallucination_misalignment_detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy scikit-learn pydantic "openai>=1.0.0" sentence-transformers nltk tenacity tqdm
    ```

4.  **Set your OpenAI API Key:**
    ```sh
    export OPENAI_API_KEY='your_secret_api_key_here'
    ```

5.  **Download NLTK data:**
    Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Input Data Structure

The pipeline is controlled by a single, comprehensive Python dictionary, `experiment_config`. A fully specified example, `FusedExperimentInput`, is provided in the notebook. This dictionary defines everything from the prompt text and model choices to hyperparameters and validation thresholds.

## Usage

The `faithfulness_hallucination_misalignment_detection_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Define your `experiment_config` dictionary. A complete template is provided.
2.  **Execute Pipeline:** Call the master orchestrator function.

    **For a standard, single analysis:**
    ```python
    # Returns a dictionary with the results of the main run
    standard_results = execute_sdm_analysis(
        experiment_config=FusedExperimentInput,
        perform_robustness_checks=False
    )
    ```

    **For a full robustness study (computationally expensive):**
    ```python
    # Returns a dictionary with main run results and robustness reports
    full_study_results = execute_sdm_analysis(
        experiment_config=FusedExperimentInput,
        perform_robustness_checks=True
    )
    ```
3.  **Inspect Outputs:** Programmatically access any result from the returned dictionary. For example, to view the primary scores:
    ```python
    final_scores = full_study_results['main_run']['final_scores']
    print(final_scores)
    ```

## Output Structure

The `execute_sdm_analysis` function returns a single, comprehensive dictionary:
-   `main_run`: A dictionary containing the `SDMFullResult` object from the primary analysis. This includes the final scores, all intermediate diagnostic metrics, and the validation report against paper benchmarks.
-   `robustness_analysis` (optional): If `perform_robustness_checks=True`, this key will contain a dictionary of `pandas.DataFrame`s, with each DataFrame summarizing the results of a specific robustness test.

## Project Structure

```
llm_faithfulness_hallucination_misalignment_detection/
│
├── faithfulness_hallucination_misalignment_detection_draft.ipynb  # Main implementation notebook   
├── requirements.txt                                                 # Python package dependencies
├── LICENSE                                                          # MIT license file
└── README.md                                                        # This documentation file
```

## Customization

The pipeline is highly customizable via the master `experiment_config` dictionary. Users can easily modify:
-   The `original_prompt_text` to analyze any prompt.
-   The `system_components` to target different LLMs or embedding models.
-   All `hyperparameters`, including `M`, `N`, `temperature`, clustering settings, and final score weights.
-   All `validation_protocols` to tighten or loosen quality control thresholds.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{halperin2025prompt,
  title={Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models},
  author={Halperin, Igor},
  journal={arXiv preprint arXiv:2508.10192},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of "Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models". 
GitHub repository: https://github.com/chirindaopensource/llm_faithfulness_hallucination_misalignment_detection
```

## Acknowledgments

-   Credit to Igor Halperin for the insightful and clearly articulated research.
-   Thanks to the developers of the scientific Python ecosystem (`numpy`, `pandas`, `scipy`, `scikit-learn`, `pydantic`) that makes this work possible.

--

*This README was generated based on the structure and content of `faithfulness_hallucination_misalignment_detection_draft.ipynb` and follows best practices for research software documentation.*
