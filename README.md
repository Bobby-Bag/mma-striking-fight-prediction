# UFC Fight Prediction: Striking Performance Classifier

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)]()

## 📌 Project Overview
The Ultimate Fighter Championship (UFC) requires athletes to be "complete fighters," excelling in striking, grappling, and conditioning. This project challenges widely accepted MMA notions by strictly isolating a single facet of competition: **striking**. 

The goal of this project is to build a binary classification model to determine if a fighter's historical striking performance alone is a sufficient indicator to predict the winner of a UFC bout (Red Corner vs. Blue Corner).

## 📂 Dataset & Scope
*   **Source:** Kaggle (UFC-Fight historical data from 1993 to 2021)
*   **Scope:** The dataset was filtered to focus exclusively on the "Modern Era" of the UFC (fights post-2015) to account for the evolution of the sport's meta.
*   **Data Strategy:** Removed features unrelated to striking, handled debuting fighters (NaN values) by engineering specific `is_debut` flags, and utilized `RobustScaler` to handle extreme outliers (star performers) without corrupting the dataset's bulk distribution.

## 🚀 Feature Engineering & Methodology
To capture the true dynamic of a fight, absolute striking statistics were converted into **relative performance metrics**. Key engineered features include:
*   `sig_striking_diff`: Difference in landed significant strikes between a fighter and their opponent.
*   `striking_acc_diff`: Difference in overall striking accuracy.
*   `opp_sig_striking_diff`: A defensive metric evaluating how many strikes a fighter absorbs relative to their opponent.

**Modeling:** 
I utilized a **Logistic Regression** model with an **L1 (Lasso) penalty**. Because striking metrics are highly correlated (e.g., strikes attempted vs. strikes landed), L1 regularization was specifically chosen to handle multicollinearity and perform automatic feature selection—shrinking redundant features to zero.

## 📊 Key Findings & Evaluation
The model's performance was evaluated against a **naive baseline of 56.4%** (which represents the accuracy of simply guessing the Red Corner—typically the favorite/champion—every time).

*   **Overall Accuracy:** The L1 Logistic Regression model achieved **62.4% accuracy**, successfully outperforming the naive baseline.
*   **Feature Importance:** The model identified Clinch Strikes Landed/Attempted and Total Strikes Landed as the most predictive metrics for the Red Corner. Defensive metrics (`opp_sig_striking_diff`) proved to be less predictive.
*   **Model Bias & Limitations:** The model achieved a strong F1-Score (0.70) and Recall (0.78) for the Red Corner, but struggled significantly to predict Blue Corner wins.

## 🛠️ How to Run the Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/Bobby-Bag/mma-striking-fight-prediction.git](https://github.com/Bobby-Bag/mma-striking-fight-prediction.git)
