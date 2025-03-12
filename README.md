# IIMK-Capstone - Credit Risk Analysis

<h2>Introduction</h2>
The primary goal of credit risk analysis in loan disbursement is to minimize default rates while maximizing profitability. Traditionally, the process of loan approval has been quite manual wherein each application is carefully evaluated based on credit history, income, age etc. of the applicant. A borrower with a high credit score and stable income is more likely to receive loan approval based on the credit risk analysis. However, now using machine learning techniques, it is possible to significantly reduce manual work in evaluating loan applications. Moreover, manual investigation is time-consuming and error-prone. In this project, an attempt is made to train a machine learning model on a dataset consisting of different attributes of typical loan applications such as credit history, income etc. of the applicants and predict whether a new loan application should be accepted or rejected. 

The dataset has over 600 records and is analyzed to understand any correlation between the target variable LoanStatus and other variables or features in the dataset. For instance, we observe that most loan applications are from low income groups and most applications that are approved have a good credit score.

<h2>EDA</h2>
Other observations such as skewness of features, missing values and general distribution of values across the dataset helped in making further decisions in the exploratory analysis. For instance, a positive skewness was observed for some numerical features such as ApplicantIncome. This information helped in ensuring that such features are transformed using techniques such as Boxcox transformations before being fed into the machine learning pipeline. Similarly, for the features where values are missing, imputation is performed as part the preprocessing step in the pipeline.

<h2>ML pipeline</h2>
The machine learing pipeline consists of preprocessing steps such as feature transformations, imputation, data scaling, and encoding. It is also observed that the dataset is imbalanced because there were significantly more samples for approved loan applications(LoanStatus = Y) than rejected ones(LoanStatus = N). Therefore, several sampling techniques like SMOTE, ADASYN etc. were evaluated to ensure the dataset is balanced well for training.

The pipeline also trained and evaluated multiple classifier algorithms - Logistic Regression, KNN Classifier, Support Vector Machines, Decision Tree Classifier, Bagging Classifier, Gradient Boosting Classifier, AdaBoost Classifier and CatBoost Classifier. Each classifier was intially defined with default hyperparameters and the model was trained. To optimize the model performance, these hyperparameters were fine tuned to get the best performance.

<h2>Observations</h2>
Using GridSearch cross-validation technique, the best performing model and hyperparameters were identified as below:

`{'classifier': LogisticRegression(C=0.1, class_weight='balanced', l1_ratio=0.9, max_iter=1000,
                   penalty='elasticnet', solver='saga'), 'classifier__C': 0.1, 'classifier__class_weight': 'balanced', 'classifier__l1_ratio': 0.9, 'classifier__penalty': 'elasticnet', 'classifier__solver': 'saga', 'sampler': RandomUnderSampler()}`

The model trained on LogisticRegression was chosen as the best performing classifier in the GridSearch cross validation. It has demonstrated a mean accuracy of __73%__(test set) across k folds(deduced through stratification) which means that the model has learned patterns in the data well enough to generalize to unseen examples most of the time. The accuracy as per classification report is also high(__81%__). However, the f1-score and recall were biased towards class 1(LoanStatus = Y) because the dataset has more samples for LoanStatus = Y. Despite using sampling techniques and class weights(for some classifier algorithms), this bias was observed in the results. It is likely that more samples will be needed to give better, more reliable results.

The AUC-ROC score of __0.76__ suggests that the model has a reasonably good ability to separate the two classes (Class 0 and Class 1). From the learning curve plots, it was observed that the model has generalized well and not overfitting. When comparing various metrics such as Accuracy, f1 score, AUC-ROC between training set and test set, the results were similar suggesting that the model is not underfitting or overfitting and has generalized well.
