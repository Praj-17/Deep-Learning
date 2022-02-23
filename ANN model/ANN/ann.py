# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
from keras.backend import dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('E:\DATA SETS\Bank data\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
"""
Creating dummy variables means assigning the categorical data a number
for example if you have data like male and female 
then you can do like male = 1 and female = 0 
if you have n categories then it will create n-1 new columns
for example if you have france, spain and germany then it will create only 2 columns 
if the given value is neither france or spain then it must be germany this is the basic principle
"""
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)
## Concatenate the Data Frames
X=pd.concat([X,geography,gender],axis=1)

# ## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# # Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = False)

# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Part 2 - Now let's make the ANN!

# # Importing the Keras libraries and package
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU # Few Discussed Activation function
from keras.layers import Dropout #REgularization parameter


# # Initialising the ANN
classifier = Sequential() #the model

# # Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

"""
Here in the above line we are creating 11 input layers since we have 11 features
These 11 feautes will work as the 11 input neurons and the units ==6 is representing the first dense layer. The number of dense units can be choosen randomly but for an precise number we use hyper optimization function
for now the Neural Network looks like

11 features    randomly chosen
  🟡             6 units
  🟡
  🟡              
  🟡 ------------>🔴
  🟡 ------------>🔴
  🟡 ------------>🔴
  🟡 ------------>🔴 
  🟡 ------------>🔴
  🟡 ------------>🔴
  🟡
  🟡
Input layer   Hidden layer 1

"""
# # Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
"""
11 features    randomly chosen      randomly chosen
  🟡             6 units                6 units
  🟡
  🟡              
  🟡 ------------>🔴 ------------>🔴
  🟡 ------------>🔴 ------------>🔴
  🟡 ------------>🔴 ------------>🔴
  🟡 ------------>🔴 ------------>🔴
  🟡 ------------>🔴 ------------>🔴
  🟡 ------------>🔴 ------------>🔴
  🟡
  🟡
Input layer   Hidden layer 1           Hidden layer 2
"""
# # Adding the output layer
classifier.add(Dense(units = 1,kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

"""
11 feautres     hyperparametric       hyperparametric          output (sigmoid)
  🟡             6 units                6 units       (since we need only one type of o/p)
  🟡 
  🟡                
  🟡  ------------>🔴 -------------------->🔴                    
  🟡  ------------>🔴 -------------------->🔴
  🟡  ------------>🔴 -------------------->🔴------------------>🔵    
  🟡  ------------>🔴 -------------------->🔴                  (0/1)
  🟡  ------------>🔴 -------------------->🔴
  🟡  ------------>🔴 -------------------->🔴
  🟡
  🟡
Input layer   Hidden layer 1           Hidden layer 2        Output layer

"""

# # Compiling the ANN
#"Since we are only doing a binary classification we will use loss = 'binary_crossentropy' "
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# # Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33,shuffle=False, batch_size = 20, epochs = 100)
print(model_history)
# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#___________________________________________________________________________________
# # Part 3 - Making the predictions and evaluating the model

# # Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score)

#claculate the f1 score
from sklearn.metrics import f1_score
f1_score = f1_score(y_pred,y_test)
print(f1_score)



