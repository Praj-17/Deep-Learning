import pandas as pd
import numpy as np
from pandas.core.indexes.period import PeriodIndex
messages = pd.read_csv('SMSSpamCollection', sep='\t', names= ["label", "message"])

#___________Data cleaning and pre-processing_____

import re
import nltk
nltk.download('stopwords')
corpus =[]
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
Y = pd.get_dummies(messages['label'])
Y = Y.iloc[:,1].values

for i in range(0, len(messages)):
    msg = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    msg= msg.lower()
    msg= msg.split()
    msg = [stemmer.stem(word) for word in msg if word not in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg)
corpus = np.array(corpus)

#___________Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
bag = cv.fit_transform(corpus).toarray()

#____________splitting the data as train and test_____
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(bag,Y,test_size= 0.2, random_state=42)

#____________Traing the model_____
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, Y_train)
y_pred = model.predict(X_test)



#__________Finding the accuracy and confusion matrix_______

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score

cm = confusion_matrix(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

print("_____________Confusion matrix______________")
print(cm)
print("______________Accuracy_______________")
print(accuracy)
print("______________F1_score")
print(f1)















