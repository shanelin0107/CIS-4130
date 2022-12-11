# CIS-4130 Term project- sentiment analysis on Amazon customers review data
# Project description


In this porject, I'm going to do sentiment analysis to Amazon customer reviews data to classify customers attitude toward the products they bought. We want to know how's customers' thoughts and feeling behind their reviews to see the feedback of the products on Amazon. We need to clean the data and do text preprocessing and make them into a pipeline to prepare to train the model. We select to use logistic regression to create the model to complete this project. And in the end, we would also need to do visualization about project results and model performance graph to evaluate how's our model doing in the project. 

#Appendix A: Code for downloading data
Region name – $us-east-2
Output format- $json
aws s3api create-bucket –mybucket4130 --region us-east-2 --create-bucket-configuration \ LocationConstraint=us-east-2

pip3 install kaggle

mkdir .kaggle
nano .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
kaggle datasets list
nano  ~/.local/lib/python3.7/sitepackages/kaggle/api/kaggle_api_extended.py

#change to these two lines
if not os.path.exists(outpath) and outpath != "-":

with open(outfile, 'wb') if outpath != "-" else os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as out:

kaggle datasets list  
kaggle datasets download
kaggle datasets download --quiet -d cynthiarempel/amazon-us-customer-reviews-dataset-p -  | aws s3 cp - s3://mybucket4130/amazonreview.zip

#Unzip the file
import zipfile
import boto3
from io import BytesIO
bucket="mybucket4130"   
zipfile_to_unzip="amazonreview.zip"   
s3_client = boto3.client('s3', use_ssl=False)
s3_resource = boto3.resource('s3')

zip_obj = s3_resource.Object(bucket_name=bucket, key=zipfile_to_unzip)
buffer = BytesIO(zip_obj.get()["Body"].read())
z = zipfile.ZipFile(buffer)
# Loop through all of the files contained in the Zip archive
for filename in z.namelist():
    print('Working on ' + filename)
    # Unzip the file and write it back to S3 in the same bucket
    s3_resource.meta.client.upload_fileobj(z.open(filename),
                 Bucket=bucket,Key= f'{filename}')

Appendix B: Carrying out descriptive statistics
#data cleaning and statistics of data
from pyspark.sql.functions import col, isnan, when, count, udf, to_date, year, month, date_format, size, split


#read the data from s3 bucket
df=spark.read.csv('s3n://mybucket4130/final_amz.csv',header=True)

#print all columns
df.columns

#print the summary of the dataframe
df.summary().show()

#check null values of star_rating and review_body column
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in 
["star_rating", "review_body"]] ).show()

#count how many records in the dataframe
df.count()

#drop null values
df = df.na.drop(subset=["star_rating", "review_body"])

#define function to drop the emoji from customer reviews
def ascii_only(mystring):
    if mystring:
        return mystring.encode('ascii', 'ignore').decode('ascii')
    else:
        return None
# assign function to udf
ascii_udf = udf(ascii_only)

#applied function to review body
df = df.withColumn("clean_review_body", ascii_udf('review_body'))

#save the cleaned data to csv
output_file_path="s3://mybucket4130/ cleaned_data.csv'"
df.write.options(header='True', delimiter=',').csv(output_file_path)

Appendix C:  ML pipeline (cleaning, feature extraction, model building)
#import all library we need
Import io
import pandas as pd
import s3fs
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, isnan, when, count, udf, to_date, year, month, date_format, size, split
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix
from pyspark.ml.feature import StopWordsRemover
# read the clean data
df=spark.read.csv('s3n://mybucket4130/cleaned_data.csv',sep='\t', header=True, inferSchema=True)
#drop the column we don't need
df=df.drop('_c0', 'marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category','helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date')

# check the null values in dataset
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in 
["clean_review_body","star_rating"]] ).show()

#remove any non numeric value in star rating
df = df.filter(~df.star_rating.rlike('\D+'))

# drop the rows with null values
df = df.na.drop(subset=["clean_review_body","star_rating"])

#convert the star_rating column from string to integer
df = df.withColumn("star_rating",df.star_rating.cast(IntegerType()))

#create a column to identify if the star rating >=3=1, otherwise is 0
df = df.withColumn("rating_converted", when(col("star_rating") > 3, 1).otherwise(0))
#split the dataset
train_set, test_set = df.randomSplit([0.7, 0.3], seed = 2000)

#convert the sentence to token
tokenizer = Tokenizer(inputCol="clean_review_body", outputCol="clean_review_words")
#remove stopwords
remover = StopWordsRemover(inputCol="clean_review_words", outputCol="remove_stopword")
#convert words to vectors
hashtf = HashingTF(numFeatures=2**16, inputCol="remove_stopword", outputCol='tf')
#create inverse document frequency
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) 
#create indexer for the rating_converted column
label_stringIdx = StringIndexer(inputCol = "rating_converted", outputCol = "label")
#create model
lr = LogisticRegression(maxIter=100)
#create pipeline
pipeline = Pipeline(stages=[tokenizer,remover, hashtf, idf, label_stringIdx, lr])

# Create a grid to hold hyperparameters 
grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, [0.0, 1.0])
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the parameter grid
grid = grid.build()

# How many models to be tested
print('Number of models to be tested: ', len(grid))

# Create a BinaryClassificationEvaluator to evaluate how well the model works
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the CrossValidator using the hyperparameter grid
cv = CrossValidator(estimator=pipeline,
estimatorParamMaps=grid,
evaluator=evaluator, 
numFolds=3,
seed=789
)
# Train the models
cv = cv.fit(train_set)

# Test the predictions
predictions = cv.transform(test_set)


# Calculate AUC
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print('AUC:', auc)

# Create the confusion matrix
predictions.groupby('label').pivot('prediction').count().fillna(0).show()
cm = predictions.groupby('label').pivot('prediction').count().fillna(0).collect()
def calculate_recall_precision(cm):
    fn = cm[0][1] # False Negative 
    tn = cm[0][2] # True Negative
    tp = cm[1][1] # True Positive 
    fp = cm[1][2] # false Positive 
    precision = tp / ( tp + fp ) 
    recall = tp / ( tp + fn )
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return accuracy, precision, recall, f1_score
print( calculate_recall_precision(cm) )
# Appendix D: Visualization
Save each graph through following code:
img_data=io.BytesIO()
plt.savefig(img_data,format='png')
img_data.seek(0)

s3=s3fs.S3FileSystem(anon=False)
with s3.open('s3://mybucket4130/file_name.png','wb') as f:
    f.write(img_data.getbuffer())
visualization 1: product category couunting
gg=df.groupby('product_category').count().sort('count',ascending=False).show()
pd=['Apparel','Toys','Sports','Shoes','Electronics','Pet Products','Office Products','Grocery','Camera','Tools','Furniture']
sb=gg.filter(gg.product_category.isin(pd))
sb.groupby('product_category').count().sort('count',ascending=False).show()
qty=sb.groupby('product_category').count().sort('count',ascending=False).toPandas()
grh=sns.barplot(x=qty['product_category'],y=qty['count'],color='c').set(title='product_counting')
grh.set_xticklabels(grh.get_xticklabels(),rotation=50)
plt.figure(figsize = (11,11))
plt.ticklabel_format(style='plain',axis='y')
-----------------------------------------------------------------------------------------------------
## Visualization 2: star rating counting
str=df.groupby('star_rating').count().sort('count',ascending=False).toPandas()
grh=sns.barplot(x=str['star_rating'],y=str['count'],color='c').set(title='star_counting')
grh.set_xticklabels(grh2.get_xticklabels(),rotation=0)
-----------------------------------------------------------------------------------------------------
## Visualization 3: confusion matrix
y_true = predictions.select("label")
y_true = y_true.toPandas()
y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()
cnf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

-----------------------------------------------------------------------------------------------------
## Visualization 4:ROC curve
#Look at the parameters for the best model that was evaluated from the grid
parammap = cv.bestModel.stages[5].extractParamMap()
for p, v in parammap.items():
    print(p, v)
#Grab the model from Stage 5 of the pipeline
mymodel = cv.bestModel.stages[5]

plt.figure(figsize=(5,5))
plt.plot(mymodel.summary.roc.select('FPR').collect(),
mymodel.summary.roc.select('TPR').collect())
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
--------------------------------------------------------------------------------------------------------
## Visualization 5: prediction result
qty=predictions.groupby('prediction').count().toPandas()
sns.barplot(x=qty['prediction'],y=qty['count'],color='c')
plt.ticklabel_format(style='plain',axis='y')
plt.title('prediction_result')
with s3.open('s3://mybucket4130/predict_result2.png','wb') as f:
	f.write(img_data.getbuffer())

# Sources:
1.	https://github.com/Kaggle/kaggle-api/issues/315
2.	Sentiment Analysis with PySpark. One of the tools I’m deeply interested… | by Ricky Kim | Towards Data Science
3.	Sentiment-Analysis-and-Text-Classification-Using-PySpark/Food Review.ipynb at master · shikha720/Sentiment-Analysis-and-Text-Classification-Using-PySpark · GitHub
4.	Confusion Matrix Visualization. How to add a label and percentage to a… | by Dennis T | Medium
