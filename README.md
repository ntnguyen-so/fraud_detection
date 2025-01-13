# Technical interview case for Data Scientist position at Vipps MobilePay AS: Fraud Detection.

## Task Description
Refer to [Task Description](task_description.md) for the original task description from the company

## Prerequisite
If you want to reproduce results, install necessary labraries.
```bash
pip install -r requirements.txt
```

## Summary
My work in this project is done in two steps:
- Explanatory Data Analysis and Feature Engineering (see [here](explanatory_data_analysis.ipynb))
- Model Development (see [here](model_development.ipynb))

### Explanatory Data Analysis
For the first step, you will see I perform the EDA and feature engineering pretty carefully because I believe if we incorporate business logic of the data well, then simple ML models can perform very well. My philosophy for this is simple > complex. The work in this project is inspired by my own experience and a case from Amazon (I hope the story was true):
- My experience: I have worked with anomaly detection on time series for almost 3 years. Till now, 100+ models have been proposed in literature and the people keep working on that and they want domain-anogstic models (i.e., just throw data into models and wait for the results). Tested on my data, those models were not useful at all and I developed another by myself and this is simple and it outperformed 40 SOTA. The secret is I incorporated the knowledge of the data I am working on to the decision-making process.
- For Amazon (see [this LinkedIn post](https://www.linkedin.com/posts/janaboerger_datainlogistics-logistics-machinelearning-activity-7275523299177107456-SDya/)): long story short, two interns were hired to work on the same project. One spent the whole time understanding business logic of the data, and the other was eager to jump into throwing the data to fancy deep learning models. The first guy just used simple ML models and outperformed DL models developed by the other by a large margin. Of course, an offer is awarded to the first guy.

While doing EDA, I also propose new features to be engineered. I believe development process should be transparent and transferrable. This is important as we will work in a team and our co-workers should understand what we have done. This saves costs for the company and also time for us. 

You can just look at the notebook [explanatory_data_analysis.ipynb](explanatory_data_analysis.ipynb) where I walked you through the whole process.

### Model development
You can just look at the notebook [model_development.ipynb](model_development.ipynb) where I walked you through the whole process.
- As what I can see from the EDA step, the data we are using is very skewed. Then, I thought of doing some data oversampling/undersampling strategies. Specifically, I tried SMOTE, which increases minority class (in this case, fraudulent transactions). I do not think it is a good idea to perform undersampling as the number of fraudulent samples is too small (97).

- The data is labelled. Natually, we can think of supervised techniques. However, I would like to see the performance of unsupervised and semi-supervised learning models. In short:
  + For supervised learning models, I tried Logistic Regression (LR), Decision Tree (DT), and XGBoosting (XGB). Those models perform very well, with almost perfect score! You will see later that LR and XGB have only 1 false negative and 0 false postive. DT is a bit worse than those, but it is still impressive with 1 false negative and 1 false positive. A note, in those models, I did not play with hyperparameters too much, but still got very good results! A major problem with supervised learning models is we need labelled data, which is sometimes not easy to obtain (not sure for the case with Vipps MobilePay).
  + To address that problem, we can try unsupervised and semi-supervised learning models.
  + For unsupervised learning models, I tried KMeans, IsolationForest (IF) and GaussianMixture (GM). They are popular models for anomaly detection and they do not need any label. However, the performance is very bad, but I can still report in anyway so that we can have some more points to discuss.
  + For semi-supervised learning models, I tried AutoEncoder (AE). I would say doing deep learning may be overkilling, but why not give it a try. A major problem with is there are many hyperparameters to optimize like weights, number of layers, etc.

- In this work, I want to focus on models that I have high interpretability as we should deal with GDPR. That is why I picked those models. Everything must be clearly explained to customers.

- I splitted the development data to 7/3, meaning that 30% of the data is reserved for test. To ensure fair evaluation, I test all models (listed below) on the same test data. As you will see below, the false negative install that LR, DT, and XGB made have pretty small impact to the system, so I think the process can be decleared as done. In terms of performance, I come with the following: supervised > semi-supervised > unsupervised. Of course, it comes with a major trade-off - labelling the data.

- I have been talking about performance and I mean accuracy. As I explain in the EDA part, here is my motivation. Distinguishing fraud and non-fraud transactions is basically a binary classification problem. There are a few we can think of the following: F1, AUC-ROC, or AUC-PR. I prefer to use AUC-PR and below are some reasonsings:
   + F1: this can give good scores to ML models even if anomalies are missed and we don't want it! In addition, it's senstive to thresholds. A small change in the threshold can result in significant difference. It's sometimes difficult to come up with a reasonable threshold and we have to frequently adjust thresholds due to concept drift (i.e., unexpected data distribution changes - one of my ML project is about this [3]).
   + AUC-ROC: although this metric is independent from thresholds (see F1). A major drawback of AUC-ROC is it can give good scores to ML models that have high TNR (True Negative Rate) and low TPR (True Positive Rate). *We clearly don't want it because we want to identify as many positives (fraud transactions) as possible!*
   + AUC-PR: same to AUC-ROC, this is independent from thresholds (+1). In addition, it focuses on positive class (in our case, fraud transactions).
   + For more details, you can look at [3] ([2] can be a good resource).

<u>References:</u>

[1] Wu, R. and Keogh, E.J., 2021. Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. IEEE transactions on knowledge and data engineering, 35(3), pp.2421-2429. https://arxiv.org/pdf/2009.13807 

[2] Sebastian Schmidl, Phillip Wenig, and Thorsten Papenbrock. Anomaly Detection in Time Series: A Comprehensive Evaluation. PVLDB, 15(9): 1779 - 1797, 2022. doi:10.14778/3538598.3538602. https://dl.acm.org/doi/10.14778/3538598.3538602 

[3] Nguyen, N.T., Heldal, R. and Pelliccione, P., 2024. Concept-drift-adaptive anomaly detector for marine sensor data streams. Internet of Things, p.101414. https://www.sciencedirect.com/science/article/pii/S254266052400355X

### Checking results
You can look at folder results/ where I store confusion matrixes of LR and XGB on the test data. In addition, you can also find the reports ranking fradulent probability of transactions from those models. These are the ones that yield the best results during the development. 

For the reports, I tried to reserve the original structure of the data. Then, I added three columns at the end: 
- fraud_predicted: predicted label of the transaction (1: fraud, 0: normal).
- fraud_predicted_proba: probability of the transaction being fraudulent.
- fraud_rank: the ranking of transactions being fraudulent w.r.t. to the test data.

Due to the big sizes of the prediction reports, please first extract it. 