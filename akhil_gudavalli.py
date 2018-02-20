import pandas as pd 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import re
import numpy as np
import scipy.stats as c
from textblob import TextBlob

      
train = pd.read_excel("Swiggy.xlsx")

def review_to_words( raw_review ):
          
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words )) 

clean_train_reviews = []
for review in train["Review"]:                                                               
    clean_train_reviews.append( review_to_words( review ))


vectorizer = CountVectorizer(analyzer = "word",max_features=500) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

features_data=pd.DataFrame(train_data_features)

sentiment=[]
for review in clean_train_reviews:
    blob = TextBlob(review)
    sentiment.append(blob.sentiment.polarity)

features_data['sentiment']=sentiment

x_train, x_test, y_train, y_test = train_test_split(train_data_features, train['Score'], test_size=0.2,random_state=2231889)


regression=LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs')

model=regression.fit(x_train,y_train)

output=model.predict(x_test)

print("Accuracy of the Regression model1 is",100*np.mean(output==y_test))

print("MAE of the Regression model1 is",(abs(output-y_test).mean()))

print("Ranked Correlation coefficient of the Regression model1 is",c.spearmanr(output,y_test)[0])



x_train, x_test, y_train, y_test = train_test_split(features_data, train['Score'], test_size=0.2,random_state=2231889)


regression=LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs')

model=regression.fit(x_train,y_train)

output=model.predict(x_test)

print("Accuracy of the Regression model2 is",100*np.mean(output==y_test))

print("MAE of the Regression model2 is",(abs(output-y_test).mean()))

print("Ranked Correlation coefficient of the Regression model2 is",c.spearmanr(output,y_test)[0])

scores=[]
for score in train['Score']:
	scores.append(score)

count=0
for i in range(0,len(sentiment)):
	if sentiment[i]>0:
		if scores[i]<=3:
			count=count+1
	elif sentiment[i]<0:
		if scores[i]>3:
			count=count+1

print("Percent of inconsistent reviews is",100*count/len(scores))









