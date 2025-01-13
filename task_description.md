# candidate-data-scientist


## Technical Case for Data Scientist Position: Fraud Detection: 
 

Your task is to build a machine learning model to identify potentially fraudulent transactions from a large set of anonymized transaction data. The goal is to predict which transactions are suspicious and output a ranked list of these transactions based on fraud probability. 

In recent years, fraud detection has become a critical challenge for financial institutions and welfare systems worldwide. For example, Norway experienced a significant case known as the Olga fraud scheme, where individuals exploited the country's welfare system by falsely claiming benefits. The fraudsters used personal data and transactions that seemed legitimate but, upon closer examination, exhibited patterns indicating frauds.

This real-world scenario is an inspiration for this exercise, where your goal is to detect Olga fraud scheme fraudulent activities through data-driven insights. 

## Dataset Details: 

You are provided with three following anonymized files: 

### Transactions Datasets (output_nonfraud_0.csv ... output_nonfraud_9.csv) : 
each file contains non-fraudulent transactions (the rows) enriched with the following information (the columns): 
* id: Unique identifier for the transaction. 
* sender_id: ID of the sender of the transaction. 
* receiver_id: ID of the receiver of the transaction. 
* amount: The value of the transaction. 
* status: the status of the transaction. 
* flowname: encode name of the flow that the transaction was in 
* bankfee: the bank fee of the transaction. 
* timestamp: When the transaction occurred 
* sender_age: age of the sender age. 
* receiver_age: age of the receiver. 

### Fraudulent Transactions Dataset (output_fraud.csv): 

* the file contains a subset of confirmed fraudulent transactions with the same schema as above dataset 

## Task Breakdown: 

### Data Exploration and Feature Engineering: 
* Explore and analyze the provided datasets. 
* Engineer new features that may help distinguish between normal and fraudulent transactions.  

### Model Development: 

* Build a machine learning model to predict fraudulent transactions using the labeled fraud data. 
* if needed, handle class imbalance, as fraudulent transactions are expected to be rare compared to normal ones.
* Train the model

### Prediction: 

* Produce a list of suspected fraudulent transactions ranked by probability of fraud.
* Produce Confusion matrix, Sensitivity and Specificity metrics 
 
## Deliverables: 

### Deliverables before the interview: 
_Create a fork of this repo and push your solution to the master branch_

**Code:** The Python code used for feature engineering, model development, and prediction.

**Results:** 
* Result file containing the ranked list of predicted fraud transactions.
* The confusion matrix for your test datasets (as an image file).

**Documentation:** A short summary explaining the approach, model performance, and any observations or limitations.

### Deliverables during the interview:
* You will present your work and the rationale behind your solution.
* You will run your model inference on our test datasets (following the same split fraud vs nonfraud data); these test datasets have the same schema as the given datasets in the repo. Finally, generate confusion matrix and the other required metrics. 
