# Business Report on Consumer Complaint Classification Project

*Created by Jai Kumar*

## Executive Summary
This research aims to classify consumer complaints into predefined categories or clusters using machine learning techniques. By leveraging the Consumer Complaint Database, we will transform text data into vectors and apply various classification algorithms to identify the most effective model. The results will help in categorizing future complaints, providing valuable insights for improving consumer services.

## Rationale
The importance of this research lies in its practical applications within our workplace. By utilizing text classification and clustering methods, we can effectively analyze reports from the ASIC implementation and test team. This analysis will help identify clusters that highlight issues, errors, deficiencies, and functional failures.

## Research Question
How can consumer complaints be classified into predefined categories to enable the categorization of future texts, and can these methods be extrapolated to any body of text to identify clusters?

## Data Sources
The data for this research will be sourced from the Consumer Complaint Database, which contains complaints about consumer financial products and services. This database is publicly accessible at Consumer Complaint Database.

This database has approximately 6 million rows and 18 columns.

### Input Variables
- **Date received**: Date and time
- **Product**: Categorical variable, plain text
- **Sub-product**: Categorical variable, plain text (Not all products have sub-products)
- **Issue**: Categorical variable, plain text
- **Sub-issue**: Categorical variable, plain text (Not all issues have sub-issues)
- **Consumer complaint narrative**: Customer description in natural language, plain text
- **Company public response**: Company's response in natural language, plain text
- **Company**: Categorical variable, plain text
- **State**: Categorical variable, plain text
- **ZIP code**: Five-digit USPS zip code, number
- **Tags**: Plain text
- **Consumer consent provided?**: Shows whether the consumer provided consent to publish their complaint narrative, plain text
- **Submitted via**: Categorical variable, plain text
- **Date sent to company**: Date and time
- **Company response to consumer**: Categorical variable, plain text
- **Timely response?**: Yes/No, plain text
- **Consumer disputed?**: Yes, No, N/A, plain text
- **Complaint ID**

## EDA Exploration on Data
1. Looked for the unique values in the dataset for each metric/column.
2. Looked for nulls and NAs in the dataset.

The dataset includes features that are not essential for our research on text processing, vectorization, and multi-classification. To simplify the text classification task, we will create a new dataframe that focuses only on the ‘Product’ and ‘Consumer complaint narrative’ columns, renaming the latter to ‘Consumer_Complaints’. This method ensures our analysis remains relevant and efficient.

- Created a new dataframe with the ‘Product’ and ‘Consumer complaint narrative’ columns.
- Removed missing values if present in the columns (we found 3.8M missing narratives).
- Renamed 'Consumer complaint narrative' to 'Consumer_Complaint'.

The shape of the DataFrame is: (2,050,110, 2). The percentage of non-null consumer complaints is: 34.6%.

Out of more than 6 million complaints, about 2 million (roughly 35% of the original dataset) contain text data. This significant subset offers a strong basis for category identification and classification tasks.

## Analysis and Category Restructuring
The analysis revealed 21 distinct product categories, though some overlap is present. For example, ‘Credit card’ and ‘Prepaid card’ are both included under the broader ‘Credit card or prepaid card’ category. This overlap can cause classification ambiguities and potentially impact model performance. To address this, we renamed and regrouped categories, reducing them to 14 unique ones. This restructuring aimed to ensure clearer distinctions and improve analysis and classification.

From the data after plotting and analysis, it is evident that the majority of customer complaints are related to credit reporting and repair, debt collection, and mortgages.

## Computational Challenges and Data Sampling
In the context of our recent computational analysis project, we encountered significant challenges related to the time-consuming nature of the computations, particularly in terms of CPU usage. To address this issue and ensure timely results, we implemented a data sampling strategy. The current dataset has close to 2M rows and 2 columns to address.

## Methodology
The text data on consumer complaints will be preprocessed using WordNetLemmatizer to generate tokens, which will then be transformed into vectors using Term Frequency-Inverse Document Frequency (TFIDF) weighting.

We will use basic classifiers for modeling with cross-validation to see the initial results:
* Logistic Regression
* K-Nearest Neighbors
* Linear SVC
* Decision Tree
* Multinomial NB
* Random Forest Classifier
* AdaBoost Classifier
* Gradient Boosting Classifier

Stratified KFold will be used to generate optimal cross-validation.

The baseline score for the classifier was obtained using the DummyClassifier with the training data. The score to meet or exceed is approximately 60%.

The best models will be selected based on cross-validation scores. We will enhance the efficiency of the models using hyperparameter tuning. Finally, we will compute the models' accuracy and use it on the test set to see if the classifier correctly identifies the product for the complaint.

## Cross-Validation Modeling Results
The research found that the best algorithm for this particular dataset was logistic regression.

| Model                           | Mean Accuracy | Standard Deviation |
|---------------------------------|---------------|--------------------|
| RandomForestClassifier          | 0.847444      | 0.014917           |
| LinearSVC                       | 0.988822      | 0.000568           |
| MultinomialNB                   | 0.988154      | 0.001117           |
| KNeighborsClassifier            | 0.941808      | 0.005136           |
| LogisticRegression              | 0.991013      | 0.001322           |
| DecisionTreeClassifier          | 0.957182      | 0.002776           |
| AdaBoostClassifier              | 0.425466      | 0.028551           |
| HistGradientBoostingClassifier  | 0.976493      | 0.000900           |

## Enhancing Model Performance through Hyperparameter Tuning
To improve the performance of our initial models, we will focus on hyperparameter tuning. The primary models under consideration are LinearSVC, Logistic Regression, and Multinomial Naive Bayes.

From hyperparameter tuning and applying GridSearchCV on the top three models, we see that the LogisticRegression model performs the best and has a good accuracy and prediction rate compared to the LinearSVC and MNB models.

Using the LogisticRegression model to predict some of the test complaints to see if they fall in the right cluster from the test dataset for validation:
1. Test Set Creation:
    * Created a CSV file for the test set.
2. Vectorization and Prediction:
    * Ran the fitted vectorizer.
    * Compared every complaint to see if there are matching products based on the prediction model.
3. Results Analysis:
    * Generated a new CSV and DataFrame to look at the matches and misses.
    * Achieved 83% accuracy, which is great in terms of business needs.
    * Noted 17% misses, mostly due to poor supervised labeling of the input dataset.

## Business Executive Summary
The implementation of the LogisticRegression model for predicting test complaints has yielded promising results. The model achieved an accuracy of 83%, which aligns well with our business requirements. This high level of accuracy indicates that the model is effective in correctly classifying the majority of complaints into their respective product categories.

However, there is a 17% miss rate, primarily attributed to poor supervised labeling in the input dataset. This highlights an area for potential improvement in data preprocessing and labeling accuracy. Addressing these issues could further enhance the model's performance and reliability.

Overall, the current model performance is satisfactory and provides a solid foundation for further refinement and optimization. The next steps should focus on improving data quality and exploring advanced techniques to reduce the miss rate.

## GitHub Locations
The main project and its files are located at GitHub Repository.

https://github.com/jaigetsback/BL_JAIK_CAPSTONE

The Data which is zipped is located in 
https://github.com/jaigetsback/BL_JAIK_CAPSTONE/blob/main/data/complaints.csv.zip

The main jupiter notebook is 
https://github.com/jaigetsback/BL_JAIK_CAPSTONE/blob/main/Final_capstone_jaik_TC.ipynb
