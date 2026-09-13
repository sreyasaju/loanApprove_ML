# loanApprove_ML

loanApprove_ML is a machine learning classifier  to predict whether a loan application gets approved. It trains and compares two models on a 45,000-row synthetic loan dataset and includes a tinyy interface for testing your own applicant ✨
## Features

- **Data Quality Handling** : flags and excludes 7 records with ~~impossible~~ ages (up to 144 years) per the dataset author’s own documented warning, rather than ignoring them!
- **Exploratory Data Analysis** : visualizes target balance, credit score by loan status, and loan intent by loan status before any modelling happens
- **Two Classification Models** : trains a Logistic Regression baseline and a Random Forest model.
- **Full Evaluation** : accuracy, precision, recall, F1, and a confusion matrix for each model
- **Feature Importance** : ranks which features the Random Forest actually leaned on!
- **Interactive Prediction** : a notebook cell that takes a new applicant’s details as input and returns Approved/Rejected with a confidence score.

## How It Works

1. **Load & Clean** : pandas loads the CSV, strips whitespace from column headers, and drops the 7 rows with `person_age` over 100.
2. **Explore** : checks the approve/reject split (77.8% / 22.2%) and plots credit score and loan intent against loan status to see what actually separates the two groups
3. **Preprocess** : one-hot encodes the five categorical columns (`drop_first=True`), splits 80/20 with stratification on the target, and scales numeric features for Logistic Regression only, Random Forest doesn’t need it
4. **Train & Evaluate** : fits both models, then reports accuracy, precision, recall, F1, and a confusion matrix heatmap for each
5. **Interpret** : pulls Gini importance from the Random Forest to rank the top 10 features, and compares both models point-by-point.
6. **Predict** : runs a new applicant’s details through the trained Random Forest and prints back a plain Approved/Rejected call with a confidence percentage

## Results

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 89.48% | 77.18% | 74.75% | 75.95% |
| Random Forest | 92.93% | 89.74% | 77.00% | 82.88% |

Random Forest wins on every metric, but not evenly... Precision jumps by 12.56pp while recall only moves 2.25pp. 

Random Forest showed a larger improvement in precision than recall. This means it made fewer incorrect approval predictions, while the improvement in identifying approved applications was smaller!

## Dependencies

| Library | Purpose |
| --- | --- |
| `pandas` | Data loading, cleaning, and the one-hot encoded feature matrix |
| `numpy` | Numeric feature selection and array operations |
| `scikit-learn` | Train/test split, scaling, both models, and all evaluation metrics |
| `matplotlib` | Base plotting for every chart in the notebook |
| `seaborn` | Boxplots, countplots, and confusion matrix heatmaps |


## Installation

Clone the repository:

```bash
git clone https://github.com/sreyasaju/loanApprove_ML.git
cd loanApprove_ML
```

Set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
jupyter notebook loan_approval.ipynb
```

## Known Limitations

- The dataset is imbalanced (77.8% rejected vs. 22.2% approved), which is why accuracy alone isn’t trusted here; precision, recall, and F1 are tracked separately for that reason
- Feature importance shows what the Random Forest found useful, not what causes approval
- The interactive prediction cell has basic input handling but doesn’t validate every malformed entry ;)

## Notes
Built as a submission for CSA CINTEL’s technical recruitment. Feedback and suggestions welcome via issues. Feel free to star the repo! 🌟

## License

```
MIT License

Copyright (c) 2026 Sreya Saju

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### Dataset Attribution

The dataset is the Loan Approval Classification Dataset by Tawei Lo, licensed under Apache 2.0.
[Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data)


Copyright © 2026 Sreya Saju