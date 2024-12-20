# Fraud-Detection-Analysis
# Overview
A large number of problems in data mining are related to fraud detection. Fraud is a common problem in
auto insurance claims, health insurance claims, credit card transactions, financial transaction and so on.
The data in this particular case comes from an actual auto insurance company. Each record represents an
insurance claim. The last column in the table tells you whether the claim was fraudulent or not. A number
of people have used this dataset and here are some observations from them:
• “This is an interesting data because the rules that most tools are coming up with do not make any
intuitive sense. I think a lot of the tools are overfitting the data set.”
• “The other systems are producing low error rates but the rules generated make no sense.”
• “It is OK to have a higher overall error rate with simple human understandable rules for a
business use case like this.”

#Key Features
1.	Data Loading and Preprocessing:
	•	The dataset includes Insurance Fraud - TRAIN-3000.csv and Insurance Fraud - TEST-12900.csv.
	•	Numerical features are scaled, and categorical features are one-hot encoded.
	•	Missing values are imputed (mean for numerical and most frequent for categorical).
2.	Pipeline and Modeling:
	•	Pipelines are defined for both classifiers:
	•	DecisionTreeClassifier
	•	RandomForestClassifier
	•	Used Pipeline from Scikit-Learn to integrate preprocessing and modeling.
3.	Hyperparameter Tuning:
	•	Hyperparameters tuned for Decision Tree:
	•	max_depth: Limits tree depth to prevent overfitting.
	•	min_samples_split: Minimum samples required to split a node.
	•	min_samples_leaf: Minimum samples required at leaf nodes.
	•	Hyperparameters tuned for Random Forest:
	•	n_estimators: Number of trees in the forest.
	•	max_depth: Limits tree depth.
	•	min_samples_split: Minimum samples required to split a node.
	•	Tuning methods:
	•	Grid Search (GridSearchCV): Exhaustive search over hyperparameters.
	•	Random Search (RandomizedSearchCV): Random combinations of parameters.
	•	Bayesian Search (BayesSearchCV): Efficient exploration of parameter space.
4.	Evaluation:
	•	Models were evaluated using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score
	•	Separate evaluation on test data for models from each tuning method.

# Technologies Used
1.	Programming Language:
	•	Python
2.	Libraries:
	•	Data Manipulation: pandas
	•	Preprocessing: scikit-learn (Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder)
	•	Modeling: DecisionTreeClassifier, RandomForestClassifier
	•	Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV, BayesSearchCV (from skopt)
	•	Evaluation Metrics: accuracy_score, precision_score, recall_score, f1_score

# Key Insights
1.	Hyperparameter Tuning:
	•	Decision Tree:
	•	Grid Search and Random Search produced similar best parameters.
	•	Bayesian Search explored a broader range and returned slightly different parameters.
	•	Random Forest:
	•	Bayesian Search provided the best results, likely due to its efficient exploration of the parameter space.
2.	Model Performance:
	•	Random Forest consistently outperformed Decision Tree across all metrics.
	•	Best Random Forest model achieved:
	•	Accuracy: 96.28%
	•	Precision: 97.47%
	•	Recall: 96.28%
	•	F1-score: 96.70%
	•	Decision Tree performance peaked at:
	•	Accuracy: 85.11%
	•	Precision: 93.92%
	•	Recall: 85.11%
	•	F1-score: 88.93%
3.	Comparison of Tuning Techniques:
	•	Bayesian Search yielded superior results compared to Grid and Random Search for Random Forest.
	•	For Decision Tree, all tuning methods performed similarly.
4.	Insights on Fraud Detection:
	•	A balanced approach to hyperparameter tuning can significantly improve model performance.
	•	Random Forest’s ensemble nature makes it better suited for high-dimensional problems like fraud detection.

# What Was Learned
	•	Efficient hyperparameter tuning can optimize model performance significantly.
	•	Bayesian Search outperforms traditional tuning methods by focusing on the most promising areas of the parameter space.
	•	The importance of pipelines in maintaining a consistent and repeatable workflow for preprocessing and modeling.

 ￼
