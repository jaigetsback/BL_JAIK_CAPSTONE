### Analyze and Bin data using Test based classifier models

**Jai Kumar**

#### Executive summary
This research aims to classify consumer complaints into predefined categories or clusters using machine learning techniques. By leveraging the Consumer Complaint Database, we will transform text data into vectors and apply various classification algorithms to identify the most effective model. The results will help in categorizing future complaints, providing valuable insights for improving consumer services.

#### Rationale
The importance of this research lies in its practical applications within our workplace. By utilizing text classification and clustering methods, we can effectively analyze reports from the ASIC implementation and test team. This analysis will help identify clusters that highlight issues, errors, deficiencies, and functional failures.

#### Research Question
What How can consumer complaints be classified into predefined categories to enable the categorization of future texts, and can these methods be extrapolated to any body of text to identify clusters?

#### Data Sources
The data for this research will be sourced from the Consumer Complaint Database, which contains complaints about consumer financial products and services. This database is publicly accessible at Consumer Complaint Database 
Links to an external site.

This database has ~6 million rows and 18 columns

Input variables:
# Date received - date and time
# Product - categorical variable, plain text 
# Sub-product - categorical variable, plain text
*  Not all products have sub products 
# Issue- categorical variable, plain text
# Sub-issue - categorical variable, plain text
*  Not all iussues have sub issues 
# Consumer complaint narrative - Customer description in natural language, plain text
# Company public response - Companies response in natural language, plain text
# Company - categorical variable, plain text
# State - categorical variable, plain text
# ZIP code - Five digit USPS zip code, number 
# Tags - plain text
# Consumer consent provided?- shows whether the consumer provided consent to publish their complaint narrative, plain text
# Submitted via - categorical variable, plain text
# Date sent to company - date and time
# Company response to consumer - categorical variable, plain text
# Timely response? - yes/no, plain text
# Consumer disputed?- yes, no, N/A, plain text
# Complaint ID 

#### EDA exploration on data
1. Looked for the unique values on the dataset for each metric/ column 
2. Looked for nulls and NA in the dataset 

The dataset includes features that are not essential for our research on text processing, vectorization, and multi-classification. To simplify the text classification task, we will create a new dataframe that focuses only on the ‘Product’ and ‘Consumer complaint narrative’ columns, renaming the latter to ‘Consumer_complaints’. This method ensures our analysis remains relevant and efficient.

a. Created a new dataframe with the  ‘Product’ and ‘Consumer complaint narrative’ columns
b. Removed missing values if present on the columns - We do see 3.8M missing narratives 
c. Renamed Consumer complaint narrative to "Consumer_Complaint"

The shape of the DataFrame is: (2050110, 2)
The percentage of non-null consumer complaints is: 34.6%

Out of more than 6 million complaints, about 2 million (roughly 35% of the original dataset) contain text data. This significant subset offers a strong basis for category identification and classification tasks.

The analysis revealed 21 distinct product categories, though some overlap is present. For example, ‘Credit card’ and ‘Prepaid card’ are both included under the broader ‘Credit card or prepaid card’ category. This overlap can cause classification ambiguities and potentially impact model performance. To address this, we can rename or regroup certain categories to ensure clearer distinctions, making analysis and classification easier.

We initially had 21 product categories, but overlaps like ‘Credit card’ and ‘Prepaid card’ under the broader ‘Credit card or prepaid card’ category caused classification ambiguities. To address this, we renamed and regrouped categories, reducing them to 14 unique ones. This restructuring aimed to ensure clearer distinctions and improve analysis and classification.

From the data after plotting and analysis, It is evident that the majority of customer complaints are related to credit reporting and repair, debt collection, and mortgages.

In the context of our recent computational analysis project, we encountered significant challenges related to the time-consuming nature of the computations, particularly in terms of CPU usage. To address this issue and ensure timely results, we implemented a data sampling strategy. The current dataset has close to 2M rows and 2 columns to address.

#### Methodology
The text data on consumer complaints will be preprocessed using WordNerLemmatizer to generate tokens which will be transformed into vectors using Term Frequency-Inverse Document Frequency (TFIDF) weighting. 

Using basic classifiers for modeling with cross validation - to see the initial result
* Logistic Regression
* K-Nearest Neighbors
* Linear SVC
* Decision Tree
* Multinomial NB
* Random Forest Classifier

Used Stratified Kfold for generating the optimal cross validation.

The baseline score for the classifier was obtained by using the DummyClassifier with the training data. The score to meet or exceed is ~60%

The best of the models will be picked as needed after the cross validation scores are alayzed. We will enhance the effeciency of the models using hyperparameter tuning. Finally we will compute the models accuracy and use it on the test set to see if the classfifer picks the right product for the complaint. 

#### Cross validation modeling Results
The research found that the best algorithm for this particular dataset was the logistic regression. 

                          Mean Accuracy  Standard Deviation
Model                                                    
RandomForestClassifier       0.847444            0.014917
LinearSVC                    0.988822            0.000568
MultinomialNB                0.988154            0.001117
KNeighborsClassifier         0.941808            0.005136
LogisticRegression           0.991013            0.001322
DecisionTreeClassifier       0.958482            0.002215


#### Next steps
To elevate the performance of our initial models, we will focus on hyperparameter tuning. The primary models under consideration are LinearSVC, Logistic Regression, and Multinomial Naive Bayes. These models have demonstrated comparable accuracy, and our goal is to determine if hyperparameter optimization can reveal a standout performer among them. This process aims to refine our predictive capabilities and select the most effective model for our needs.

Once the final model is selected, Accuracy scores for the same in terms of precision, recall, accurary and F1 scores will be computed. the confusion matrix will also be generated. 

The model will then be trained on the “test” set and we will look at the prediction of right products based on consumer complaints. 


