# CIS-4130 Term project- sentiment analysis on Amazon US Customer Reviews Dataset
## Project description


In this porject, I'm going to do sentiment analysis to Amazon customer reviews data to classify customers attitude toward the products they bought. We want to know how's customers' thoughts and feeling behind their reviews to see the feedback of the products on Amazon. We need to clean the data and do text preprocessing and make them into a pipeline to prepare to train the model. We select to use logistic regression to create the model to complete this project. And in the end, we would also need to do visualization about project results and model performance graph to evaluate how's our model doing in the project. 

## Data Acquisition

We used AWS Command Line Interface to extract data from Kaggle to Amazon S3 Bucket.


## data dictionary

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



