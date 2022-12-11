# CIS-4130 Term project- sentiment analysis on Amazon US Customer Reviews Dataset
## Project description


In this porject, I'm going to do sentiment analysis to Amazon customer reviews data to classify customers attitude toward the products they bought. I want to know how's customers' thoughts and feeling behind their reviews to see the feedback of the products on Amazon. I need to clean the data and do text preprocessing and make them into a pipeline to prepare to train the model. I select to use logistic regression to create the model to complete this project. And in the end, I would also need to do visualization about project results and model performance graph to evaluate how's our model doing in the project. 

## Data Acquisition

I used AWS Command Line Interface to extract data from Kaggle to Amazon S3 Bucket. I need to prepare our AWS Access Key Aand Secret Key ID. Then I define the region at "us-east-2" and output foramt choose "json". Once I download the data, it would be a zip file then I have to unzip it to our S3 Bucket.


## Data dictionary

1. marketplace- 2 letter country code of the marketplace where the review was written.
2. customer_id- Random identifier that can be used to aggregate reviews written by a single author.
3. review_id- The unique ID of the review.
4. productid- The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same productid.
5. product_parent- Random identifier that can be used to aggregate reviews for the same product.
6. product_title- Title of the product.
7. product_category- Broad product category that can be used to group reviews
8. star_rating- The 1-5 star rating of the review.
9. helpful_votes- Number of helpful votes.
10. total_votes- Number of total votes the review received.
11. vine- Review was written as part of the Vine program.
12. verified_purchase- The review is on a verified purchase.
13. review_headline- The title of the review.
14. review_body- The review text.
15. review_date- The date the review was written.

## Machine Learnging Modeling and Pipeline

In this project, I choose to use Logistic Regression as our model since our label was categorical variable which include only "Positve" which equal to 0, and "Negative" which equal to 1. As for the pipeline, I apply several of operation to our reviews data such as Tokenizer, StopWordsRemover, HashingTF, IDF, StringIdexer, LogicticRegression, then we put all these steps to our pipeline. Besides, in order to get more accurate results from this model, I also use grid search and cross validator to test more model and find the best hyperparameters to train our data.

## Sources:
1.	https://github.com/Kaggle/kaggle-api/issues/315
2.	Sentiment Analysis with PySpark. One of the tools I’m deeply interested… | by Ricky Kim | Towards Data Science
3.	Sentiment-Analysis-and-Text-Classification-Using-PySpark/Food Review.ipynb at master · shikha720/Sentiment-Analysis-and-Text-Classification-Using-PySpark · GitHub
4.	Confusion Matrix Visualization. How to add a label and percentage to a… | by Dennis T | Medium

