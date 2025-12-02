# validata-predict-loan-approval
```
Data Source: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data
```
<pre>
This project builds machine learning models to predict whether a loan application will be approved or rejected
based on applicant and loan features.
  
Two models are implemented and compared:
	•	K-Nearest Neighbors (KNN)
	•	Decision Tree Classifier
</pre>
The project includes preprocessing, training, evaluation, hyperparameter tuning, feature importance analysis, and saving the cleaned dataset + trained model.


```
project/
│
├── data/
│   ├── loan_approval.csv               # Original dataset (input)
│   └── loan_approval_cleaned.csv       # Cleaned dataset (auto-generated)
│
├── loan_approval_knn_decision_tree.ipynb  
├── Loan Approval Prediction Report.pdf
├── README.md                                
```

```
Requirements:
	•	pandas
	•	numpy
	•	scikit-learn
	•	matplotlib
	•	seaborn
```
## How to Run the Project

```
jupyter notebook loan_approval_knn_decision_tree.ipynb

# In the notebook menu:
Cell > Run All
```
```
Running all cells will:
    •	Load and preprocess data
    •	Save cleaned dataset
    •	Split into train/test
    •	Train baseline KNN and Decision Tree models
    •	Perform hyperparameter tuning using GridSearchCV
    •	Evaluate model performance
    •	Show confusion matrices and feature importance
```

## Output & Results

After running the notebook, you will get:

Model Metrics:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score
	•	Classification Report
	•	Confusion Matrix

Best Model: The tuned Decision Tree performed best.

Top predictive features include:
	•	Credit Score
  •	Loan Term
	•	Loan Amount
	


