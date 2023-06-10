
import numpy as np 
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

#Import Tensorflow modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

#%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

"""
Example data
-A new pandemic has arived, and a vaccine needs to be rolled out as soon as possible
-An experiemental vaccine was tested on individuals from ages 5 to 100 in a clinical trial
-The trial had 3600 participants. Half were under 70 years old, half were 70 years or older
-95% of patients 70 or older experienced side effects
-95% of patients under 70 experienced no side effects
"""

trainLabels = []
trainSamples = []
for i in range(90):
    #the 5% of younger individuals who did experience side effects
    randomYounger = randint(5,69) #ages
    trainSamples.append(randomYounger) #append ages of individuals
    trainLabels.append(1) #Append 1 means individual DID experience side effect

    #the 5% of older individuals who did not experience side effects
    randomOlder = randint(70,100)
    trainSamples.append(randomOlder)
    trainLabels.append(0) #Append 0 means individual DID NOT experience side effect

for i in range(1710):
    #the 95% of younger individuals who did not expereince side effects
    randomYounger = randint(5,69)
    trainSamples.append(randomYounger)
    trainLabels.append(0)

    #the 95% of older individuals who did experience side effects
    randomOlder = randint(70,100)
    trainSamples.append(randomOlder)
    trainLabels.append(1) 

trainLabels = np.array(trainLabels) #convert Lists into numpy arrays
trainSamples = np.array(trainSamples)
trainLabels, trainSamples = shuffle(trainLabels, trainSamples) # shuffle to remove any order

scaler = MinMaxScaler(feature_range=(0,1)) #scales from 13-100. we want to scale it from 0-1. this line specifies the range
scaledTrainSamples = scaler.fit_transform(trainSamples.reshape(-1,1)) # this will actually transform the scaling

#Build a Sequential Model.
model = Sequential([ 
    Dense(units=16, input_shape=(1,), activation='relu'), #input shape knows shape of the data in our case 1 dimension
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax') #2 outputs, Either patient experienced side effects or not
    #softmax will give us a probability distribution among the possible outputs
])
#print(model.summary()) quick visualization of our model

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) #prepare for training

model.fit(x=scaledTrainSamples, y=trainLabels, batch_size=10, epochs=38, shuffle=True, verbose=2)
#first paramater is inputdata, y is target data, batch size is how many samples are included in one batch to be processed
#epochs = model will train on all the data in the dataset 30 times before completing training process
#shuffle = data will be shuffled
#verbose = option to allow us to see output. 0, 1 or 2. 2 is the most verbose level(highest level of output)

### Validation set
# model.fit(x=scaledTrainSamples, y=trainLabels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

#### Now, the model has been trained, so now lets build the test set

testLabels = []
testSamples = []
for i in range(90):
    #the 5% of younger individuals who did experience side effects
    randomYounger = randint(5,69) #ages
    testSamples.append(randomYounger) #append ages of individuals
    testLabels.append(1) #Append 1 means individual DID experience side effect

    #the 5% of older individuals who did not experience side effects
    randomOlder = randint(70,100)
    testSamples.append(randomOlder)
    testLabels.append(0) #Append 0 means individual DID NOT experience side effect

for i in range(1710):
    #the 95% of younger individuals who did not expereince side effects
    randomYounger = randint(5,69)
    testSamples.append(randomYounger)
    testLabels.append(0)

    #the 95% of older individuals who did experience side effects
    randomOlder = randint(70,100)
    testSamples.append(randomOlder)
    testLabels.append(1) 

testLabels = np.array(testLabels)
testSamples = np.array(testSamples)
testLabels, testSamples = shuffle(testLabels, testSamples)

scaledTestSamples = scaler.fit_transform(testSamples.reshape(-1,1))
#predict on test data
#maps a probability to a patient experiencing side effect or not
predictions = model.predict( 
    x=scaledTestSamples,
    batch_size = 10,
    verbose = 0
)

roundedPredictions = np.argmax(predictions, axis=-1) #most probably prediction

#Now we have have the predictions, but we want to visually observe how well the model predicts on test data
cm = confusion_matrix(y_true=testLabels, y_pred=roundedPredictions)

#function from scikit-learn's website.
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cmPlotLabels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cmPlotLabels, title='Confusion Matrix')
plt.show() #print confusion matrix
#bottom left - no side effects when they had side effects (bad prediction)
#top left - no side effects when they did not have side effects (good prediction) ... and so on
