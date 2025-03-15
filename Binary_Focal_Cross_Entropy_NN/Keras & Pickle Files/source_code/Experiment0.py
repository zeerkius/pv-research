import pickle
from keras.src.optimizers.schedules import learning_rate_schedule
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

ration = (len(k)/len(n))






# creating inputs for bmi model

three_input = tf.keras.Input(Xtrain[0].shape)

bmi_input = tf.keras.Input(bmi_train[0].shape)

race_input = tf.keras.Input(race_train[0].shape)

nrelbc_input = tf.keras.Input(nrelbc_train[0].shape)

# race model

dense1 = tf.keras.layers.Dense(1,activation = "relu")(three_input) # this is the three age related inputs agefrst, menopause, agerp

merged0 = tf.keras.layers.Concatenate()([dense1,nrelbc_input,race_input])

race_model = tf.keras.layers.Dense(1, activation = "relu")(merged0)

# bmi model


dense2 = tf.keras.layers.Dense(1,activation = "relu")(three_input)

merged1 = tf.keras.layers.Concatenate()([dense2,nrelbc_input,bmi_input])

bmi_model = tf.keras.layers.Dense(1 , activation = "relu")(merged1)

# model creation

bmi_model0 = keras.Model(inputs = [three_input,bmi_input,nrelbc_input] , outputs = bmi_model , name = "BMI_EXPERIMENT")

race_model0 = keras.Model(inputs = [three_input,race_input,nrelbc_input],outputs = race_model , name = "RACE_EXPERIMENT")

     
# model compilation
 
            
          
bmi_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=ration, gamma=2) ,
                   optimizer = "SGD", metrics = ["accuracy"])


# model trainig and model saving


bmi_model0.fit([Xtrain,bmi_train,nrelbc_train] ,Ytrain , epochs = 500 , batch_size = 1000 , validation_split = 0.30)


bmi_model0.save(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\BMI_EXP0.keras")

race_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=ration, gamma=2) ,
                   optimizer = "SGD" , metrics = ["accuracy"])


race_model0.fit([Xtrain,race_train,nrelbc_train] ,Ytrain , epochs = 500 , batch_size = 1000 , validation_split = 0.30)


race_model0.save(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\RACE_EXP0.keras")

















