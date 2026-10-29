![Titanic](kaggle.jpeg)

# Titanic: Machine Learning from Disaster

Binary classification on the [Kaggle Titanic](https://www.kaggle.com/competitions/titanic)
dataset: predict passenger survival from the training data and submit
predictions for the 418 test passengers.

A small Keras/TensorFlow MLP (4,481 parameters) on top of hand-built features,
trained on a single stratified 80/20 split.

<!-- TODO: update this badge URL if the repo is renamed -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ghavvvo/kaggle-titanic/blob/main/titanic.ipynb)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)

## Results

| Submission | Split | Validation accuracy | Public LB | Notes |
|---|---|---|---|---|
| #1 | stratified 80/20 | 0.8212 | **0.76794** | baseline MLP, threshold 0.5 |

The public leaderboard score sits close to the validation estimate, which is
consistent with a model of this size on 891 training rows.

## Structure

```
.
├── titanic.ipynb     # notebook: download → features → MLP → submission.csv
├── requirements.txt
├── kaggle.jpeg       # README banner
└── LICENSE
```

## Requirements

- Python 3.10 – 3.12
- A Kaggle account with API credentials

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then export your Kaggle token (never hardcode it in the notebook):

```bash
export KAGGLE_API_TOKEN="<your-kaggle-token>"
```

Create a token at [kaggle.com/settings](https://www.kaggle.com/settings) → *API*
→ *Create New Token*, and revoke it there if it ever leaks.

## Run

**Colab:** use the badge above, and store the token in *Secrets* (key icon in the
left sidebar) under the name `KAGGLE_API_TOKEN`.

**Locally:**

```bash
jupyter notebook titanic.ipynb
```

The dataset is downloaded automatically by `kagglehub` into its local cache;
`submission.csv` is written next to the notebook.

## Pipeline

1. **Download** — `kagglehub.competition_download("titanic")` → `train.csv` (891 rows) + `test.csv` (418 rows).
2. **Feature engineering** — `Title` from `Name`, `FamilySize`, `IsAlone`, `Cabin` first letter, `log1p(Fare)`.
3. **Preprocessing** — median imputation + `StandardScaler`, one-hot encoding, test columns reindexed to train (28 features).
4. **Model** — `64 → 32 → 16 → 1` dense layers, dropout 0.4/0.3/0.2, Adam(1e-3), `EarlyStopping` on `val_loss` (patience 20).
5. **Submit** — predict test, threshold 0.5, write `submission.csv` (418 rows).

## Notes

- Validation uses a stratified 80/20 hold-out split.
- Predictions use a 0.5 probability threshold.
- `Cabin` is reduced to its first letter; the field is absent for most passengers.

## Possible next steps

- 5-fold stratified cross-validation for a more stable estimate.
- Baseline comparison (logistic regression, random forest, gradient boosting).
- Threshold tuning, added as a sweep over validation predictions.

## License

MIT — see [LICENSE](LICENSE).
