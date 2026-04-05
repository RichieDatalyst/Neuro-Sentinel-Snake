# Neuro-Sentinel Snake
### An ML-Powered Benchmarking and Explainability Framework for Classical Pathfinding Agents

[![CI](https://github.com/RichieDatalyst/Neuro-Sentinel-Snake/actions/workflows/ci.yml/badge.svg)](https://github.com/RichieDatalyst/Neuro-Sentinel-Snake/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What This Project Does

Classical pathfinding algorithms —> A\*, BFS, Greedy Best-First Search are deterministic and optimal, but they are black boxes. They make no attempt to explain their decisions, predict their own failure, or adapt based on experience.

This project treats those algorithms as **agents under audit**. It runs them across 5 maze configurations of increasing complexity, collects 4.5 million step-level observations, and applies a complete ML analysis stack to answer questions classical AI cannot answer on its own:

- Can a neural network learn to imitate A\* from demonstrations alone?
- Do different search algorithms behave differently, or does maze complexity matter more?
- Which state features actually drive agent decisions?
- Can we predict dangerous states before they occur?
- Do agents show performance drift over time on harder mazes?

**Real-world parallel:** This mirrors how ML is applied in autonomous systems, robotics, and AIOps using data to audit, explain, and improve rule-based decision systems.

---

## Architecture

```
neuro-sentinel-snake/
│
├── core/                    # Game engine (Snake, Maze, A*, BFS, GBFS)
├── mazes/                   # 5 maze configurations (easy → dense)
├── ml/
│   ├── logger.py            # Headless simulation → CSV data collection
│   ├── features.py          # Shared feature engineering layer
│   ├── imitation.py         # Option 1: Behavioural Cloning (MLP)
│   ├── classifier.py        # Option 2: RF / GBM / LR comparison
│   ├── maze_difficulty.py   # Option 3: Difficulty regression
│   ├── clustering.py        # Option 4: K-Means + t-SNE
│   ├── failure_predictor.py # Option 5: High-danger state prediction
│   ├── anomaly_detector.py  # Option 6: Isolation Forest + drift detection
│   └── xai.py               # SHAP explainability across all models
├── dashboard/app.py         # 8-page Streamlit analytics dashboard
├── main.py                  # CLI entry point
├── config.py                # All hyperparameters in one place
└── requirements.txt
```

---

## ML Modules and Results

### Data Collection
**3,000 episodes × 4.56 million steps** logged across 3 agents × 5 mazes × 200 episodes each.

Each step records an 11-dimensional state vector:
```
[danger_up, danger_down, danger_left, danger_right,   # wall proximity (4 bits)
 food_up, food_down, food_left, food_right,           # food direction (4 bits)
 dir_x, dir_y,                                        # current heading (2 values)
 dist_to_food]                                        # normalised Manhattan distance
```

---

### Option 1: Imitation Learning (Behavioural Cloning)

**Concept:** Let A\* play 200 games per maze, record every `(state → action)` pair as labelled data, and train an MLP to replicate A\*'s decisions. At inference time the neural network decides no search at all.

**Results:**

| Split | Accuracy |
|---|---|
| Train (seen mazes) | 97.7% |
| Test (unseen mazes: Maze4, Maze5) | 92.9% |
| Covariate shift | 4.8% |

The 4.8% performance gap between seen and unseen mazes demonstrates **covariate shift** a fundamental ML concept where a model trained on one distribution degrades on another. Even though the MLP achieved 92.9% on unseen mazes, the remaining errors cluster in novel wall configurations that A\* handled deterministically but the MLP had never encountered.

**Feature importance (permutation-based):**

| Feature | Importance |
|---|---|
| dir_x (current X direction) | 25.3% |
| dir_y (current Y direction) | 15.7% |
| danger_down | 15.2% |
| danger_up | 13.8% |
| danger_right | 10.4% |
| danger_left | 9.8% |
| food signals | ~5% combined |

Direction signals dominate at 41% combined. Danger signals follow at 49%. Food direction accounts for only 5%. The imitation model learned A\*'s actual priority ordering: **avoid walls first, navigate toward food second** not the reverse, as one might naively expect.

---

### Option 2: Classical ML Classifier Comparison

**Concept:** Train Random Forest, Gradient Boosting, and Logistic Regression to predict the next action from the state vector. Tune hyperparameters on a 40k-row stratified sample (fast), refit on 3.6M rows (rigorous).

**Results:**

| Model | Train Acc | Test Acc | Time |
|---|---|---|---|
| Random Forest | 97.1% | 97.1% | 293s |
| Gradient Boosting | 97.1% | 97.1% | 4224s |
| Logistic Regression | 96.8% | 96.8% | 45s |

**Key finding:** Logistic Regression achieves 96.8% — only 0.3% behind Random Forest. This means the 11-dimensional state space is **almost perfectly linearly separable**. The binary danger/direction flags create clean hyperplanes that a linear model can exploit. Adding tree complexity (RF, GBM) provides marginal benefit at significant compute cost a meaningful model selection insight.

**Random Forest feature importances:**

| Feature | Importance |
|---|---|
| dir_x | 36.1% |
| dir_y | 28.3% |
| food signals (all) | 22% combined |
| danger signals (all) | 11% combined |

Food direction matters more in the multi-agent classifier (22%) than in imitation learning (5%). This reveals a behavioural difference: **GBFS and BFS weight food direction more heavily than A\*** in their path decisions, a difference the classifier captured by training on all three agents simultaneously.

---

### Option 3: Maze Difficulty Regression

**Concept:** Extract structural performance features per maze and train a regression model to predict a composite difficulty score. Difficulty is computed from steps-per-food (40%), dead-end encounters (30%), path optimality (20%), and death rate (10%).

**Difficulty scores (validated ordering):**

| Maze | Avg Steps/Food | Difficulty Score |
|---|---|---|
| Maze1 Easy | 36.4 | 0.200 |
| Maze2 Medium | 37.6 | 0.204 |
| Maze3 Medium | 51.8 | 0.248 |
| Maze4 Hard | 85.4 | 0.483 |
| Maze5 Dense | 165.6 | 0.800 |

Linear Regression achieves LOO R² = 1.000 on this dataset. Note: with 5 mazes, any regression model will overfit to zero error — the value here is the **validated difficulty ranking**, not the regression metrics. Adding more maze configurations would make this a genuine regression problem.

---

### Option 4: Behaviour Clustering (K-Means + t-SNE)

**Concept:** Extract 8 behavioural features per episode (steps, score, direction changes, etc.), cluster using K-Means, and visualise with t-SNE dimensionality reduction.

**Optimal k=3** (silhouette score = 0.625):

| Cluster | Avg Steps | Avg Score | Avg Foods | Interpretation |
|---|---|---|---|---|
| 0 | 1,992 | 475.8 | 47.6 | Long, successful games |
| 1 | 822 | 60.3 | 6.0 | Short games, less food |
| 2 | 0 | 0.0 | 0.0 | Immediate no-path episodes |

**Most important finding:** All three agents split across clusters in nearly identical proportions A\* (636/320/44), BFS (622/333/45), GBFS (630/320/50). The clustering found **no algorithmic separation**. Maze complexity, not algorithm choice, determines which cluster an episode falls into. A\* and BFS are behaviourally indistinguishable at the episode level they reach food with the same efficiency and the same failure patterns.

---

### Option 5: High-Danger State Prediction

**Concept:** Since well-designed mazes ensure agents never die, the failure label is redefined as "high-danger state" steps where the snake is surrounded on 2+ sides, or its current heading leads directly into a wall. This mirrors real-world predictive maintenance: flag pre-failure conditions before the event, not after.

**Dataset:** 526,600 high-danger steps (11.5%) from 4.56M total steps.

**Results:**

| Model | ROC-AUC | Avg Precision | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.9919 | 0.9930 | 95.96% |
| Random Forest | 1.0000 | 1.0000 | 99.99% |
| Gradient Boosting | 1.0000 | 1.0000 | 100.00% |

**Perfect AUC explained:** The danger label was defined using `danger_*` flags that are also input features. The model learned a deterministic rule already encoded in the feature vector confirming that the state representation is correctly engineered. A ROC-AUC of 1.0 here means "the features are sufficient to perfectly determine danger" rather than "the model overfit."

**Feature importances (danger prediction):**

| Feature | Importance |
|---|---|
| danger_down | 29.1% |
| danger_up | 24.5% |
| danger_right | 20.1% |
| danger_left | 14.5% |
| dir_y | 5.0% |
| dir_x | 3.9% |

Danger flags dominate at 88% combined. This contrasts sharply with the action classifier where direction dominated at 64%. The two models learned fundamentally different patterns from the same features: **what drives action choice** vs **what indicates danger** are different things.

---

### Option 6: Anomaly Detection + Drift Monitoring

**Concept:** Profile each agent's normal behavioral distribution using Isolation Forest. Flag anomalous episodes. Apply rolling-window drift detection across episode sequences.

**Anomaly counts:**

| Agent | Episodes | Anomalies | Rate |
|---|---|---|---|
| A\* | 1,000 | 9 | 0.9% |
| BreadthFirst | 1,000 | 13 | 1.3% |
| GreedyBestFirst | 1,000 | 23 | 2.3% |

GBFS has 2.5× more anomalies than A\*. This is explainable: GBFS uses a greedy heuristic sensitive to food placement — unusual food positions create anomalous behavioral patterns. A\* is more robust because it considers full path cost, not just proximity to food.

**Drift alerts detected:**

| Agent | Maze | Early Score | Late Score | Drop |
|---|---|---|---|---|
| A\* | Maze4_hard | 92.0 | 56.0 | **39.1%** |
| BreadthFirst | Maze4_hard | 107.0 | 53.0 | **50.5%** |

Both A\* and BFS show significant score degradation in later episodes on the hard maze. Both algorithms perform ~45% worse in the second half of 200 episodes than the first. The cause is random food placement as episodes progress, food spawns in increasingly difficult positions relative to the maze structure. This performance drift mirrors real-world model degradation under distribution shift.

---

### XAI — SHAP Explainability

**SHAP values across all three models reveal consistent but distinct patterns:**

**Random Forest (action classifier):**
Food direction features dominate (food_left=6.0%, food_down=5.0%), with danger signals secondary. The classifier learned that food direction is the primary action driver across all three agents.

**Failure Predictor (danger classifier):**
Danger flags are perfectly symmetric (danger_left=18.1%, danger_right=18.1%, danger_up=17.7%, danger_down=17.7%). All four directions contribute equally danger from any direction matters equally for flagging a high-risk state.

**Imitation MLP (A\* imitator):**
Food_left dominates (10.0%), followed by food_right (6.0%) and danger_up (5.6%). The MLP learned that lateral food position is the strongest signal for A\*'s left/right action choice — consistent with A\*'s horizontal-first path planning on these maze layouts.

**Cross-model insight:** Direction signals (`dir_x`, `dir_y`) matter most for imitation learning but least for danger prediction. Danger flags matter most for danger prediction but least for imitation. The same 11 features carry different information for different tasks, a validation of the feature engineering design.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Windows / Linux / macOS

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/neuro-sentinel-snake.git
cd neuro-sentinel-snake
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the Full Pipeline
```bash
# Step 1: Collect game data (headless, ~7 min)
python main.py --mode simulate

# Step 2: Train all 6 ML modules (~2 hours for GBM, rest is fast)
python main.py --mode train

# Step 3: Run SHAP explainability analysis (~10 min)
python main.py --mode analyze

# Step 4: Launch the dashboard
python main.py --mode dashboard
# Open http://localhost:8501

# Optional: Watch a live game
python main.py --mode play --agent AStar
python main.py --mode play --agent BreadthFirst
python main.py --mode play --agent GreedyBestFirst
```

### Run Tests
```bash
pip install pytest
python -m pytest tests/ -v
```

## GitHub Setup

```bash
# Initialise repository
git init
git add .
git commit -m "Initial commit: Neuro-Sentinel Snake ML framework"

# Push to GitHub (create repo first at github.com)
git remote add origin https://github.com/YOUR_USERNAME/neuro-sentinel-snake.git
git branch -M main
git push -u origin main
```

**Important — what NOT to push (handled by .gitignore):**
- `data/game_log.csv`  4.5M rows, too large for GitHub (>500MB)
- `data/episode_stats.csv`  regenerated by `--mode simulate`
- `models/`  all PKL files, regenerated by `--mode train`
- `.venv/`  local virtual environment
- `experiments/`  MLflow runs

**What IS committed:**
- All source code
- 5 maze `.txt` files
- Small derived data files (`shap_*.json`, `maze_features.csv`, `drift_report.csv`, `clustering_artifacts.json`)
- `requirements.txt`, `config.py`

### GitHub Actions CI
The CI pipeline at `.github/workflows/ci.yml` runs automatically on every push:
1. Installs dependencies
2. Runs a 2-episode smoke simulation
3. Trains all ML modules on small data
4. Runs pytest
5. Verifies all model files were created

---

## MLflow Experiment Tracking

```bash
# View MLflow UI
mlflow ui --backend-store-uri experiments/mlflow_runs

# Open http://localhost:5000
```

All training runs are tracked with hyperparameters, metrics, and model artifacts.

---

## Configuration

All hyperparameters live in `config.py`. Key settings:

```python
EPISODES_PER_AGENT_MAZE = 200    # increase for more data
CLASSIFIER_SAMPLE_SIZE  = 50_000 # GridSearchCV sample (increase for accuracy)
FAILURE_SAMPLE_SIZE     = 80_000 # Danger predictor sample
FAILURE_ALERT_THRESHOLD = 0.70   # Dashboard danger alert trigger
ANOMALY_DRIFT_DROP      = 0.30   # 30% score drop triggers drift alert
IMITATION_TEST_MAZES    = ["Maze4_hard.txt", "Maze5_dense.txt"]  # held-out
```



## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn (RF, GBM, LR, K-Means, Isolation Forest, MLP) |
| XAI | SHAP |
| Experiment tracking | MLflow |
| Dashboard | Streamlit + Plotly |
| Gameplay UI | Tkinter |
| CI/CD | GitHub Actions |
| Data | CSV pipeline (no database required) |

---

## Project Structure

```
neuro-sentinel-snake/
├── .github/workflows/ci.yml   # GitHub Actions CI
├── core/
│   ├── State.py               # Game environment + get_state(), get_reward()
│   ├── agentsnake.py          # Agent dispatcher
│   ├── astar.py               # A* search
│   ├── breadthfirst.py        # BFS
│   ├── greedybestfirst.py     # GBFS
│   └── view.py                # Tkinter viewer (single + side-by-side)
├── mazes/
│   ├── Maze1_easy.txt
│   ├── Maze2_medium.txt
│   ├── Maze3_medium.txt
│   ├── Maze4_hard.txt
│   └── Maze5_dense.txt
├── ml/
│   ├── logger.py              # Simulation engine + data collection
│   ├── features.py            # Shared feature engineering
│   ├── imitation.py           # Opt 1: Behavioural Cloning
│   ├── classifier.py          # Opt 2: Multi-model comparison
│   ├── maze_difficulty.py     # Opt 3: Difficulty regression
│   ├── clustering.py          # Opt 4: K-Means + t-SNE
│   ├── failure_predictor.py   # Opt 5: Danger prediction
│   ├── anomaly_detector.py    # Opt 6: Isolation Forest + drift
│   └── xai.py                 # SHAP across all models
├── dashboard/app.py           # Streamlit dashboard (8 pages)
├── tests/test_pipeline.py     # Pytest smoke tests
├── main.py                    # CLI entry point
├── config.py                  # All hyperparameters
├── requirements.txt
└── README.md
```

---

## Author

**RichieDatalyst**: Computer Science, NUCES FAST Lahore  
Built as a portfolio project demonstrating end-to-end ML engineering: data collection, supervised learning, unsupervised learning, explainability, experiment tracking, and deployment. This is the updated version of an assignment to convert it into a pure AI/ML project. Existing version of assignment can be seen at: "https://github.com/RichieDatalyst/Intelligent-Snake-Solvers"
