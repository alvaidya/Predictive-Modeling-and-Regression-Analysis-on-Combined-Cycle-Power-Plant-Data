# Combined Cycle Power Plant (CCPP) Data Analysis

## Project Overview
This project focuses on analyzing the **Combined Cycle Power Plant (CCPP)** dataset. The dataset contains operational records of a power plant with environmental and operational parameters used to predict the net hourly electrical energy output (PE).  

Through this analysis, we explore exploratory data analysis (EDA), regression models (linear, multiple, polynomial), interaction effects, feature importance, and K-Nearest Neighbors (KNN) regression. The project also compares flexible vs. inflexible models based on dataset characteristics, aligning with statistical learning theory (ISLR exercises).

---

## 1. Dataset Information
- **Source**: CCPP dataset (Excel file: `Folds5x2_pp.xlsx`)  
- **Size**: 9,568 rows × 5 columns  
- **Features**:  
  - **AT** – Ambient Temperature  
  - **V** – Exhaust Vacuum Speed  
  - **AP** – Ambient Pressure  
  - **RH** – Relative Humidity  
- **Target**:  
  - **PE** – Net hourly electrical energy output of the plant  

### (a) Package Imports
Python libraries used:
- Data handling: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Regression/ML: `scikit-learn`, `statsmodels`
- Utilities: `StandardScaler`, `PolynomialFeatures`, `train_test_split`

### (b) Exploratory Data Analysis (EDA)
1. **Rows and Columns**: Verified dataset size (9,568 × 5).  
2. **Scatterplots**:  
   - AT and V show strong positive correlation.  
   - AT and PE are negatively correlated.  
   - V and PE are negatively correlated.  
3. **Summary Statistics**: Mean, median, quartiles, IQR, and ranges were computed for all features.  

---

## 2. Regression Models

### (c) Simple Linear Regression
- Built individual linear regression models for each predictor (`AT`, `V`, `AP`, `RH`) against target `PE`.  
- Identified outliers using residuals and IQR thresholds, highlighted in visual plots.  

### (d) Multiple Regression
- Modeled PE as a function of all predictors simultaneously.  
- All predictors were statistically significant (p-values < 0.05).  

### (e) Comparison of Simple vs. Multiple Regression
- Extracted coefficients from both models.  
- Found that combining predictors in multiple regression gave more stable coefficients compared to individual regressions.  

### (f) Nonlinear Association
- Fit cubic polynomial regression models.  
- Results:  
  - AT, AP, RH showed strong nonlinear associations with PE.  
  - V did not show significant nonlinear association.  

### (g) Interaction Terms
- Tested pairwise interactions between predictors.  
- Most interactions were statistically significant, except `AT:AP` (p > 0.1).  

### (h) Model Improvement
- Compared base linear regression with polynomial regression models.  
- Metrics used: Mean Squared Error (MSE) on training and testing datasets.  
- Refinement step removed statistically insignificant predictors.  
- Final model achieved lower test error, improving generalization.  

### (i) K-Nearest Neighbors (KNN)
- Applied KNN regression with raw and normalized features.  
- Normalization significantly improved results.  
- Best K was selected based on lowest test error.  

### (j) Model Comparison
| Model | Train Error (MSE) | Test Error (MSE) |
|-------|------------------|------------------|
| Initial Linear Regression | Moderate | Higher |
| Polynomial Regression (Quadratic) | Lower | Balanced |
| Refined Polynomial Model | Lowest | Lowest |
| KNN (Raw) | Higher | Higher |
| KNN (Normalized) | Competitive | Competitive |  

---

## 3. ISLR Conceptual Questions

### ISLR 2.4.1  
- **Large n, small p**: Flexible models perform better (risk of underfitting decreases).  
- **Small n, large p**: Inflexible models perform better (risk of overfitting decreases).  
- **Nonlinear relationships**: Flexible models perform better.  
- **High variance in errors**: Inflexible models perform better (less prone to overfitting noise).  

### ISLR 2.4.7  
- **K = 1**: Prediction based on single nearest neighbor → label is Green.  
- **K = 3**: Prediction based on 3 nearest neighbors → majority label is Red.  
- **Best K with nonlinear boundary**: Small K, since larger K smooths the boundary too much.  

---

## 4. Key Learnings
- Environmental features like temperature and pressure have strong predictive power for plant output.  
- Polynomial regression with feature selection offers a balance between interpretability and accuracy.  
- KNN requires normalization to be effective, and optimal K selection is critical.  
- Model comparisons show trade-offs between flexibility and generalization, aligning with ISLR theoretical principles.  

---

## 5. How to Run
1. Place dataset in `../data/CCPP/Folds5x2_pp.xlsx`.  
2. Install required packages:  
   ```bash
   pip install pandas numpy seaborn matplotlib statsmodels scikit-learn
   ```  
3. Run the Jupyter notebook or Python script step by step.  

---

## 6. Requirements
- Python 3.8+  
- Required libraries in `requirements.txt`:  
  ```txt
  pandas
  numpy
  seaborn
  matplotlib
  statsmodels
  scikit-learn
  ```

---

## 7. Conclusion
This project demonstrates how different regression approaches and KNN can be applied to predict energy output from a power plant. Insights align with statistical learning theory — showing when flexible vs. inflexible models perform best. The combination of EDA, regression analysis, nonlinear modeling, and KNN comparison provides a comprehensive understanding of the dataset and modeling trade-offs.
