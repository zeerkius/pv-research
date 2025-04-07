import pickle
from keras.src.optimizers.schedules import learning_rate_schedule
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense , Dropout , Input , Flatten


# loading data from serialized files


Xtrain = pickle.load(open(r"C:\Users\agboo\source\repos\BC_NN\BC_NN\Training_NN.pickle","rb"))
Ytrain = pickle.load(open(r"C:\Users\agboo\source\repos\BC_NN\BC_NN\Targets_NN.pickle","rb"))
bmi_train = pickle.load(open(r"C:\Users\agboo\source\repos\BC_NN\BC_NN\bmi_train_NN.pickle","rb"))
race_train = pickle.load(open(r"C:\Users\agboo\source\repos\BC_NN\BC_NN\race_train_NN.pickle","rb"))
nrelbc_train = pickle.load(open(r"C:\Users\agboo\source\repos\BC_NN\BC_NN\nrelbc_train_NN.pikcle","rb"))

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


# create Lr_schedule

# this employs something similar to learning rate decay using c = 10 , alpha_0 (c/c + tau) , where tau = epoch# , under SGD

class glide_sched(keras.optimizers.schedules.LearningRateSchedule): # super class we are calling
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
    def __call__(self,step):
        c = 100
        k = self.learning_rate * c / c + step
        return k
    # to  save the model we need to use get_config

    def get_config(self): 
        import numpy 
        # saving this as a float so when the resulting file is saved and we want to use new isnatnces we can then use it as it isnt saved merley as
        # a graphical represntation of the instruction and instead a float we can use in classification
        return {"learning_rate":float(self.learning_rate.numpy())}
        # tensorflow uses dictionaries to help utilize configirations to save from
    
    # this helps un-pack the dictionary to be saved in a file
    @classmethod # we need this a classmethod so it dosent work on new instances when called
    def from_config(cls,config):
        return cls(**config)
    
        
# model compilation
 
            
          
bmi_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0) ,
                   optimizer = tf.keras.optimizers.SGD(learning_rate= glide_sched(learning_rate = tf.cast(0.00005,tf.float64))) , metrics = ["accuracy"])


# model trainig and model saving


bmi_model0.fit([Xtrain,bmi_train,nrelbc_train] ,Ytrain , epochs = 500 , batch_size = 1000 , validation_split = 0.20)


bmi_model0.save("BMI_EXP0.keras")

race_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0) ,
                   optimizer = tf.keras.optimizers.SGD(learning_rate= glide_sched(learning_rate = tf.cast(0.00005,tf.float64))) , metrics = ["accuracy"])


race_model0.fit([Xtrain,race_train,nrelbc_train] ,Ytrain , epochs = 500 , batch_size = 1000 , validation_split = 0.20)


race_model0.save("RACE_EXP0.keras")

















