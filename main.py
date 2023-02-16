#im doing this project to solidify my knowledge and review the process on how to make a binary image classification model 
import matplotlib 
from matplotlib import pyplot as plt 
import numpy as np 
import tensorflow as tf 
import cv2 
import os 
import random 
import imghdr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import Precision , Recall, BinaryAccuracy 
from tensorflow.keras.models import load_model



labelNames = ['Ace', 'Shanks' ]
# #load data 
# # want to load the data into a tensorflow pipeline to more efficiently load and access our data on the fly 
# dataDirectory = 'data'
# #validImageExtensions = ['jpg','png', 'bmp', 'jpeg'] # we want to check all the images to see if they are in this list 
# #118 images in the ace directory 
# # #111 images in the shanks directory 
# # for folder in os.listdir(dataDirectory):   # for each folder in data directory
# #     for image in os.listdir(os.path.join(dataDirectory , folder )):  #for each image in the folder 
# #         imagePath = os.path.join(dataDirectory, folder , image)
# #         try : 
# #             img = cv2.imread(imagePath) #read in the image ; if opencv cannot read in the image then it will throw an error 
# #             extension = imghdr.what(imagePath )
# #             if extension not in validImageExtensions: 
# #                 #not a valid extension 
# #                 print("Image is not a valid format {}".format(imagePath))
# #                 os.remove(imagePath)    
# #         except Exception as e : 
# #             print("there is an issue with this image {}".format(imagePath))
        
        
        
# #now that we removed all the sketchy images we can use the tensorflow dataset api to make the data a pipline 
# data= tf.keras.utils.image_dataset_from_directory(dataDirectory) # this loads it into a default dataset : 32 per batch   , 256x256 pixels 

# #we want to now make an iterator so that we can generate data on the fly
# dataIterator = data.as_numpy_iterator(); 

# # batch = dataIterator.next(); 

# # #now that we have a batch of data lets get the first few so that we can verify with images correlate to which labels 
# # figure , ax = plt.subplots(ncols= 4 , figsize=(20,20))  #builds the main figure and 4 subplots 
# # for idx , img in enumerate( batch[0][:4]): 
# #     ax[idx].imshow(img.astype(int))
# #     ax[idx].title.set_text(batch[1][idx])

# # plt.show()    

# # 1 : shanks 
# # 0 : ace 



# #preprocess data 
# #scale the data to make it between 0 and 1 
# scaledData = data.map(lambda x, y: (x/255.0, y))

# #test that the data is now scaled correctly
# # scaledIterator = scaledData.as_numpy_iterator() 
# # scaledBatch = scaledIterator.next()

# # scaledBatch[0][0].min () : 0 


# #split data into train, val, and test sets 

# # print(len(scaledData))  # there are 7 batches of 32 

# trainSize = int(len(scaledData)*.7)
# print("Train Size: ", trainSize)
    
# valSize = int(len(scaledData)*.2) +1 
# print("Val Size: ", valSize)

# testSize = int(len(scaledData)*.1) +1
# print("Test Size: ", testSize)


# trainData = scaledData.take(trainSize)
# valData = scaledData.skip(trainSize).take(valSize)
# testData = scaledData.skip(trainSize+valSize).take(testSize)




# #build model 
# #now build the model since we have our data partitioned out 

# model = Sequential()
# #number of filters , kernel size, stride, activation , input shape : 256x256 pixels with 3 channels for each (rgb)
# model.add(Conv2D(16 , (3,3), 1 , activation='relu', input_shape = (256,256,3 )))
# model.add(MaxPooling2D())

# model.add(Conv2D(32 , (3,3), 1 , activation='relu'))
# model.add(MaxPooling2D())

# model.add(Conv2D(16 , (3,3), 1 , activation='relu'))
# model.add(MaxPooling2D())

# #flattens the result from the conv layers 
# model.add(Flatten())

# #16x16 = 256 ; 256 is going to be the result when the flatten layer flattens it 
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# #print(model.summary())


# #compile model 
# #add our optimizer , loss , and metrics to our model
# model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics = ['accuracy'])


# logDirectory = 'logs'
# tensorBoardCallback = tf.keras.callbacks.TensorBoard(log_dir= logDirectory)


# #train model 
# hist = model.fit(trainData, epochs=20, validation_data=valData, callbacks=[tensorBoardCallback])

# #evaulate model 
# lossFigure = plt.figure()
# plt.plot(hist.history['loss'], color='red', label = 'loss')
# plt.plot(hist.history['val_loss'], color='blue', label = 'valLoss')
# lossFigure.suptitle('Loss', fontsize=20 )
# plt.legend(loc="upper left")
# plt.show()  

# accFigure = plt.figure()
# plt.plot(hist.history['accuracy'], color = 'red', label = 'accuracy')
# plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'val_accuracy')    
# accFigure.suptitle('Accuracy', fontsize =20 )
# plt.legend(loc='upper left')
# plt.show()



# #evaluate model 
# #precision,recall , binacc

# precision = Precision()
# recall = Recall()
# binAcc = BinaryAccuracy()

# #use the test Dataset to test out metrics 
# for batch in testData.as_numpy_iterator():
#     X , y = batch
#     yhat = model.predict(X)
#     precision.update_state(y, yhat)
#     recall.update_state(y , yhat )
#     binAcc.update_state(y, yhat) 

# print("Precision {}".format(precision.result().numpy()),"Recall {}".format(recall.result().numpy()),"BinaryAccuracy {}".format(binAcc.result().numpy()))
    
    
# #***Save model****

# model.save(os.path.join('models', 'AceOrShanksBotV1.h5')) 



# now that we have trained and saved the model we can load and test different images on it 

bot =load_model(os.path.join('models', 'AceOrShanksBotV1.h5'))

# # get an image then resize it to the right dimensions then have the model predict on that input 
# image = cv2.cvtColor(cv2.imread(os.path.join('data','shanks', 'shanksarrives.jpg')),cv2.COLOR_BGR2RGB)

# resizedImage = tf.image.resize(image, (256,256))
# resizedImage = resizedImage.numpy().astype(int) # change it to a numpy array of ints 


# plt.imshow(resizedImage)
# plt.show(); 


# resizedImage = np.expand_dims(resizedImage / 255,0) #normalize image 



# prediction = bot.predict(resizedImage)
# print("Prediction Value: ", prediction)
# if prediction >.5: 
#     print("I am looking at shanks")
# else: 
#     print("I am looking at ace ")



dataDirectory= 'testData'
validImageExtensions = ['jpg', 'png', 'jpeg', 'bmp']
# for folder in os.listdir(dataDirectory):   # for each folder in data directory
#     for image in os.listdir(os.path.join(dataDirectory , folder )):  #for each image in the folder 
#         imagePath = os.path.join(dataDirectory, folder , image)
#         try : 
#             img = cv2.imread(imagePath) #read in the image ; if opencv cannot read in the image then it will throw an error 
#             extension = imghdr.what(imagePath )
#             if extension not in validImageExtensions: 
#                 #not a valid extension 
#                 print("Image is not a valid format {}".format(imagePath))
#                 os.remove(imagePath)    
#         except Exception as e : 
#             print("there is an issue with this image {}".format(imagePath))
data= tf.keras.utils.image_dataset_from_directory('testData')
scaledData = data.map(lambda x,y: (x/255,y)) 
testDataSize = int(len(scaledData)*.7); 
testData = scaledData.take(testDataSize)
batch = scaledData.as_numpy_iterator().next()

fig = plt.figure(figsize = (15, 9))

#setting values to rows and column 
numCols = 4 
numRows = 3 
count =0 

for i in range(numCols*numRows): 
    fig.add_subplot(numRows , numCols , i+1)
    plt.imshow(batch[0][i]) #shows the image 
    plt.axis('off')
    resized = tf.image.resize(batch[0][i], (256,256)) #resizes 
    expanded = np.expand_dims(resized,0) #expands
    prediction = bot.predict(expanded)
    prediction  = round(prediction.max()) #makes the prediction a scalar value 
    
    actual = labelNames[batch[1][i].max()]
    guess =labelNames[prediction]
    clr =''
    if(guess == actual ): 
        clr = 'green'  
        count +=1 
    else: 
        clr = 'red'


    plt.title("{} (Predicted {})  ".format(actual ,guess ), color=clr, weight ='bold')

fig.suptitle("Ai Was {}% right".format(round(count*100 /(numCols*numRows))), fontsize=20)

plt.show()