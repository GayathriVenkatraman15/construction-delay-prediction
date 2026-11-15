# AI-Based Construction Project Delay Prediction and Risk Analysis System

This mini project predicts whether a construction project is likely to be delayed, gives the delay probability, and classifies the risk level:

| Delay probability | Risk level |
|---|---|
| Below 40% | Low Risk |
| 40% to 70% | Medium Risk |
| Above 70% | High Risk |

## Dataset
A simulated dataset of 400 Residential and Commercial construction projects with 17 attributes (200 delayed, 200 on time). The results demonstrate the approach and may not match real projects.

## Method
- Exploratory data analysis (including a correlation heatmap)
- Encoding of Project_Type, scaling with StandardScaler, 15 model features
- Stratified 80:20 train-test split (320 training, 80 test records)
- Four models compared on test accuracy:

| Model | Accuracy |
|---|---|
| Logistic Regression | 75.00% |
| SVM | 71.25% |
| Random Forest | 68.75% |
| Decision Tree | 62.50% |

Logistic Regression was selected as the final model.

## Streamlit application
The app validates the input (actual progress cannot exceed planned progress), predicts Delayed or On Time, and shows the delay probability and risk level. A Clear / Reset button restores the default values.

## Files
- `app.py` : Streamlit application
- `Construction_Delay_Prediction_Analysis.ipynb` : EDA, preprocessing, model training and evaluation
- `Mini_Project_Construction_Delay_Dataset.csv` : dataset
- `logistic_delay_model.pkl` : trained Logistic Regression model
- `delay_scaler.pkl` : fitted feature scaler
- `requirements.txt` : Python libraries needed

## How to run
```
pip install -r requirements.txt
streamlit run app.py
```
Keep `app.py`, `logistic_delay_model.pkl` and `delay_scaler.pkl` in the same folder.

## Author
Devanath Gayathri, MCA, SRM Institute of Science and Technology (Directorate of Online Education)
