# Wildfire Time-to-Hit: A Survival Analysis Approach

Predicting when a wildfire will threaten an evacuation zone — using only the first five hours of data.

**Competition:** [WiDS Worldwide Global Dathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview)

---

## 📌 The Problem

When a wildfire ignites, emergency managers face high-stakes decisions before certainty is available: which communities to warn, when to warn them, and where to position scarce resources. This project frames that operational need as a **right-censored survival analysis problem**: given only the first 5 hours of a fire's perimeter dynamics, predict the probability it will come within 5km of an evacuation zone within **12, 24, 48, and 72 hours**.

Unlike a simple classification task, this requires two things at once:
- **Urgency ranking** — which fires are most dangerous, relative to each other
- **Calibrated probabilities** — numbers decision-makers can actually trust for threshold-based actions

## 📊 Dataset

- **221 wildfire events** in the training set, **95** in the test set
- **36 engineered features** covering fire growth, spread direction, and distance-to-evac-zone dynamics — all computed strictly from the first 5 hours after ignition
- **Target:** `time_to_hit_hours` (duration) and `event` (1 = hit within 72h, 0 = censored)
- **Censoring rate:** 68.8% — most fires in the dataset never reach an evac zone within the observation window

## 🔬 Methodology

### 1. Exploratory Analysis
Kaplan-Meier survival curves to understand the baseline hazard shape and censoring structure before any modeling.

### 2. Multicollinearity Reduction
The raw feature set was heavily redundant (many features were transforms or near-duplicates of each other). Reduced 32 predictors down to **15** using:
- Duplicate detection (exact and near-exact, including a literal `sin`/`cos`-style relationship and a perfect-negative pair)
- Hierarchical clustering on correlation distance to identify feature "families"
- **Variance Inflation Factor (VIF)** analysis — max VIF dropped from **infinite** (perfect duplicates) to **~6.24**

Every "champion" feature chosen from a correlated cluster was picked based on physical interpretability, not just statistical convenience — and each choice was verified empirically (e.g., a data-quality artifact feature was swapped for a physically meaningful one, confirmed via improved convergence and fewer proportional-hazards violations).

### 3. Separation Diagnosis
One feature (`dist_min_ci_0_5h`) showed **0% range overlap** between event/censored groups — a near-perfect proxy for the outcome that caused model convergence failures and artificially inflated concordance (0.93). Diagnosed systematically across all features and removed.

### 4. Model: Cox Proportional Hazards
Fit with `lifelines.CoxPHFitter`. Proportional hazards assumption checked formally (`check_assumptions()`); minor violations were investigated and a pragmatic decision made to retain a simpler, more interpretable model over a marginally "more correct" but less interpretable one.

### 5. Competition-Specific Evaluation
The competition's **Hybrid Score** (`0.3 × C-index + 0.7 × (1 − Weighted Brier Score)`) isn't a built-in metric in any survival analysis library. Custom evaluation functions were derived:
- `predict_hit_probabilities()` — converts survival curves into horizon-specific hit probabilities
- `censor_aware_brier()` — implements the competition's exact censoring rule (excluding fires whose outcome is genuinely unknown at a given horizon)
- Full **5-fold cross-validation** for honest, out-of-sample performance estimates (in-sample metrics were consistently and substantially inflated)

### 6. Model Comparison
A **Random Survival Forest** (`scikit-survival`) was benchmarked against the Cox model using identical evaluation code. Cox outperformed on this dataset size (~0.797 vs ~0.772 mean Hybrid Score), likely due to the small sample size (221 rows) favoring a regularized linear model over a tree ensemble.

### 7. Final Model
`CoxPHFitter(penalizer=0.20)` — regularization strength chosen via cross-validation, refit on the full training set for final test predictions.

## 📈 Results

| Metric | Value |
|---|---|
| Final feature count | 15 (reduced from 32) |
| Max VIF after reduction | 6.24 |
| Cross-validated C-index | ~0.76–0.79 |
| Cross-validated Weighted Brier Score | ~0.17 |
| **Cross-validated Hybrid Score** | **~0.797** |

## 🧠 Key Lessons

- **In-sample metrics lie.** An early model showed 0.93 concordance — which turned out to be a near-perfect proxy feature causing separation, not genuine skill. Cross-validation is what caught it.
- **VIF and separation are different problems.** A feature can have a perfectly healthy VIF (no multicollinearity with other predictors) while still being dangerously predictive of the outcome itself.
- **Statistical "correctness" and practical usefulness aren't always aligned.** A model that technically violates the proportional hazards assumption on a borderline feature isn't automatically worse for the task at hand.

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `lifelines` · `scikit-survival` · `scikit-learn` · `statsmodels` · `scipy` · `matplotlib`

## 📁 Repository Structure

```
├── wildfire_full_pipeline.ipynb   # End-to-end notebook: EDA → modeling → submission
├── train.csv                       # (not tracked — see .gitignore)
├── test.csv                        # (not tracked — see .gitignore)
├── submission.csv                  # Final competition predictions
├── requirements.txt
└── README.md
```

## 🚀 Running This Project

```bash
git clone https://github.com/phionahadhi/WildFire_Survival_Analysis.git
cd WildFire_Survival_Analysis
pip install -r requirements.txt
jupyter notebook WildFire_Survival_Analysis.ipynb
```

## 📬 Contact

*(Phionah Adhiambo, https://www.linkedin.com/in/phionah-adhiambo/)*

---

*This project was built as part of the WiDS Worldwide Global Dathon 2026. Dataset and competition framing © WiDS Worldwide.*
