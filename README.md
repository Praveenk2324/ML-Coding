# ?? ML Coding — Machine Learning Study Repository

A comprehensive, hands-on Machine Learning and Deep Learning study repository covering classical ML, neural networks, computer vision, NLP, reinforcement learning, and more — built using scikit-learn, PyTorch, and Gymnasium.

---

## ?? Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Notebooks](#notebooks)
- [Practice Projects](#practice-projects)
- [ML Prep](#ml-prep)
- [Reinforcement Learning](#reinforcement-learning)
- [Datasets](#datasets)
- [Saved Models](#saved-models)
- [Getting Started](#getting-started)
- [Requirements](#requirements)

---

## Overview

This repository is a structured learning journal covering the full ML/DL spectrum — from fundamental supervised learning algorithms to advanced deep learning architectures and reinforcement learning agents. Each notebook is self-contained and progressively builds on prior concepts.

**Topics Covered:**
- Classical Machine Learning (classification, regression, ensembles)
- Deep Learning & Neural Networks (ANNs, CNNs, RNNs)
- Natural Language Processing
- Computer Vision
- Unsupervised Learning & Dimensionality Reduction
- Autoencoders
- Reinforcement Learning (Q-Learning)
- PyTorch from scratch

---

## Repository Structure

```
ML Coding/
¦
+-- ?? Core Notebooks           # Numbered topic-based study notebooks
¦
+-- practice/                   # Applied ML projects & experiments
¦   +-- Animal_Classification.ipynb
¦   +-- Audio_classification.ipynb
¦   +-- Bean_leaf_classifier.ipynb
¦   +-- cancer_detection.ipynb
¦   +-- diabetes_predictor.ipynb
¦   +-- rice_classification.ipynb
¦   +-- q_learning.ipynb
¦   +-- playground.ipynb
¦   +-- Video+3+-+Autograd.ipynb
¦
+-- mlprep/                     # Interview & exam preparation
¦   +-- from_scratch/
¦   ¦   +-- Logistic_Regression_Scratch.ipynb
¦   +-- interview_notebooks/
¦       +-- DSA_Practice_Guide.ipynb
¦       +-- ML_AI_MLOps_Interview_QA.ipynb
¦
+-- rl/                         # Standalone Reinforcement Learning scripts
¦   +-- Q_learning.py
¦
+-- datasets/                   # Local datasets used across notebooks
+-- models/                     # Saved trained model files
+-- pytorch/                    # PyTorch-specific experiments (WIP)
+-- sandbox.ipynb               # Free-form experimentation
+-- requirements.txt
```

---

## Notebooks

The core notebooks are numbered to reflect the recommended study order:

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `Pytorch_Practice[1].ipynb` | Introduction to PyTorch — tensors, operations, GPU basics |
| 2 | `Classification[2].ipynb` | Binary & multi-class classification with scikit-learn |
| 3 | `Training_models[3].ipynb` | Gradient descent, regularization, and model training strategies |
| 4 | `Decision_trees[4].ipynb` | Decision Trees — splitting criteria, pruning, visualization |
| 5 | `Random_Forests[5].ipynb` | Random Forests, Bagging, feature importance |
| 6 | `Dimensionality_Reduction[6].ipynb` | PCA, t-SNE, UMAP and other reduction methods |
| 7 | `Unsupervised_learning[7].ipynb` | Clustering (K-Means, DBSCAN), anomaly detection |
| 8 | `Intro_to_ANN[8].ipynb` | Artificial Neural Networks — architecture, backpropagation |
| 9 | `NNs_with_Pytorch[9].ipynb` | Building neural networks with PyTorch |
| 10 | `Deep_Neural_Networks[10].ipynb` | Deep networks — batch norm, dropout, advanced optimizers |
| 11 | `CNN[11].ipynb` | Convolutional Neural Networks for image tasks |
| 12 | `Sequences_RNN_CNN[12].ipynb` | Sequence modelling with RNNs and 1D CNNs |
| 13 | `NLP_with_RNNs[13].ipynb` | Natural Language Processing using RNNs |
| — | `Autoencoders.ipynb` | Autoencoders for unsupervised representation learning |
| — | `Reinforcement_Learning.ipynb` | Introduction to Reinforcement Learning concepts |
| — | `sandbox.ipynb` | Scratch space for quick experiments |

---

## Practice Projects

Located in the `practice/` folder, these are applied end-to-end ML projects:

| Project | Description |
|---------|-------------|
| `Animal_Classification.ipynb` | Image classification on the AFHQ (Animal Faces HQ) dataset |
| `Audio_classification.ipynb` | Audio feature extraction and classification |
| `Bean_leaf_classifier.ipynb` | Plant disease detection on bean leaf images using CNNs |
| `cancer_detection.ipynb` | Binary cancer classification with a PyTorch neural network |
| `diabetes_predictor.ipynb` | Diabetes risk prediction using classical ML |
| `rice_classification.ipynb` | Multi-class rice grain classification |
| `q_learning.ipynb` | Q-Learning agent exploration in notebook form |
| `Video+3+-+Autograd.ipynb` | Deep dive into PyTorch autograd mechanics |
| `playground.ipynb` | Miscellaneous experiments |

---

## ML Prep

The `mlprep/` folder contains interview and exam preparation material:

### `from_scratch/`
- **`Logistic_Regression_Scratch.ipynb`** — Logistic Regression implemented from the ground up using NumPy, without any ML libraries. Covers the math behind the algorithm step by step.

### `interview_notebooks/`
- **`DSA_Practice_Guide.ipynb`** — Data Structures & Algorithms practice in Python, covering common patterns and problems.
- **`ML_AI_MLOps_Interview_QA.ipynb`** — Curated ML, AI, and MLOps interview questions with detailed answers.

---

## Reinforcement Learning

### `rl/Q_learning.py`
A standalone Python script implementing a **Q-Learning agent** using the [Gymnasium](https://gymnasium.farama.org/) library on the `Taxi-v4` environment.

**Key parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.9 | Learning rate |
| `gamma` | 0.95 | Discount factor |
| `epsilon` | 1.0 ? 0.01 | e-greedy exploration rate |
| `epsilon_decay` | 0.9995 | Decay per episode |
| `num_episodes` | 10,000 | Training episodes |
| `max_steps` | 100 | Steps per episode |

After training, the agent is re-run with `render_mode='human'` to visually demonstrate the learned policy over 5 evaluation episodes.

---

## Datasets

Stored in the `datasets/` directory:

| Dataset | Description |
|---------|-------------|
| `riceClassification.csv` | Rice grain classification features |
| `force2020_data_unsupervised_learning.csv` | Unsupervised learning dataset |
| `CTA_-_Ridership_-_Daily_Boarding_Totals.csv` | Chicago Transit Authority ridership time-series data |
| `shakespeare.txt` | Shakespeare corpus for NLP/RNN text generation |
| `sample_audio.flac` | Sample audio file for audio classification |
| `housing/` + `housing.tgz` | California Housing dataset |
| `FashionMNIST/` | FashionMNIST image dataset |
| `afhq/` | Animal Faces HQ dataset for image classification |
| `bean_leaf/` | Bean leaf disease image dataset |

---

## Saved Models

Pre-trained model checkpoints stored in `models/`:

| File | Description |
|------|-------------|
| `bean_leaf_classifier.pth` | PyTorch CNN trained on bean leaf disease classification |
| `cancer_predictor.pth` | PyTorch neural network for cancer detection |
| `my_california_housing_model.pkl` | Scikit-learn model for California housing price prediction |
| `yolov9m.pt` | YOLOv9-M object detection model checkpoint |

---

## Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Lab or VS Code with Jupyter extension

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "ML Coding"
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch** (visit [pytorch.org](https://pytorch.org/get-started/locally/) for the appropriate command for your system):
   ```bash
   # Example: CPU only
   pip install torch torchvision torchaudio
   ```

5. **Install additional dependencies as needed:**
   ```bash
   pip install gymnasium numpy seaborn plotly opencv-python torchaudio
   ```

6. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

---

## Requirements

Core dependencies from `requirements.txt`:

```
pandas
scikit-learn
matplotlib
Pillow
jupyterlab
```

Additional libraries used across notebooks:
- `torch`, `torchvision`, `torchaudio` — Deep learning with PyTorch
- `numpy` — Numerical computing
- `gymnasium` — Reinforcement learning environments
- `seaborn` — Statistical data visualization
- `opencv-python` — Computer vision utilities

---

## ?? Learning Resources

A curated list of resources referenced during this project:

- [PyTorch in 1 Hour](https://youtu.be/r1bquDz5GGA?si=bGlGtfjOq1vE1-0b)
- [PyTorch Introduction — 0byte.io](https://0byte.io/articles/pytorch_introduction.html)
- [Learn PyTorch in 5 Projects – Tutorial](https://youtu.be/E0bwEAWmVEM?si=i4aJPkMyqMYpfJ50)

---

## ??? Study Roadmap

```
Classical ML  ?  Deep Learning  ?  Computer Vision  ?  NLP  ?  Reinforcement Learning
    ¦                ¦                   ¦               ¦              ¦
  [2–5]           [8–10]              [11–12]           [13]          [rl/]
```

> **Tip:** Follow the numbered notebook sequence for the most coherent learning path. Use the `practice/` projects to apply each concept to real datasets.

---

*Built with ?? as a personal ML learning journey.*
