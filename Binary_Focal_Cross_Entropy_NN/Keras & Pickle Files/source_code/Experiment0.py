import pickle
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense , Input 


# loading data from serialized files


Xtrain = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\Training_NN.pickle","rb"))
Ytrain = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\Targets_NN.pickle","rb"))
bmi_train = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\bmi_train_NN.pickle","rb"))
race_train = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\race_train_NN.pickle","rb"))
nrelbc_train = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\nrelbc_train_NN.pickle","rb"))

# this gives us our alpha ration where our ratio is minority / majority

k = [x for x in Ytrain if x == 1] # minority
n = [y for y in Ytrain if y != 1] # majority


ration = len(k)/len(n) # -> alpha value

# reshaping data so we can put it in the neural network

import numpy as np

bmi = []
race = []
start = 0

for val in race_train:
    race.append(np.array([race_train[start][0] , nrelbc_train[start][0]]))
    bmi.append(np.array([bmi_train[start][0] , nrelbc_train[start][0]]))
    start += 1

race = np.array(race) # - > race
bmi = np.array(bmi) # - > bmi



# creating inputs for both models

three_input = tf.keras.Input(Xtrain[0].shape)

race_input = tf.keras.Input(race[0].shape) # race + nrelbc

bmi_input = tf.keras.Input(bmi[0].shape) # bmi + nrelbc


# race model

dense1 = tf.keras.layers.Dense(1,activation = "relu")(three_input) # this is the three age related inputs agefrst, menopause, agerp

dense2 = tf.keras.layers.Dense(1,activation = "sigmoid")(race_input) # this is the race + nrelbc

merged1 = tf.keras.layers.Concatenate()([dense1,dense2]) # final sum

race_model = tf.keras.layers.Dense(1,activation = "sigmoid")(merged1) # final model

race_model0 = keras.Model(inputs = [three_input,race_input] , outputs = race_model , name = 'RACE_EXPERIMENT')


#bmi model


dense4 = tf.keras.layers.Dense(1,activation = "relu")(three_input) # this is the three age related inputs agefrst, menopause, agerp

dense5 = tf.keras.layers.Dense(1,activation = "sigmoid")(bmi_input) # this is the bmi + nrelbc

merged2 = tf.keras.layers.Concatenate()([dense4,dense5]) # final sum

bmi_model = tf.keras.layers.Dense(1,activation = "sigmoid")(merged2) # final model

bmi_model0 = keras.Model(inputs = [three_input,bmi_input] , outputs = bmi_model , name = 'BMI_EXPERIMENT')


     
# model compilation
 
race_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=ration, gamma=3) ,
                   optimizer = "SGD" , metrics = ["accuracy"])
            
          
bmi_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=ration, gamma=3) ,
                   optimizer = "SGD", metrics = ["accuracy"])



# model trainig 

race_model0.fit([Xtrain,race] ,Ytrain , epochs = 750 , batch_size = 7000 , validation_split = 0.20)

bmi_model0.fit([Xtrain,bmi] , Ytrain , epochs = 750 , batch_size = 7000 , validation_split = 0.20)


# model file saving

race_model0.save(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\RACE_EXP0.keras")


bmi_model0.save(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\BMI_EXP0.keras")
















