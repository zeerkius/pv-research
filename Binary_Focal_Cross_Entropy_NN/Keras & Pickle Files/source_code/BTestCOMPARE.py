import tensorflow as tf
import keras
import pickle

# load Neural Networks from keras Models



bmi_model0 = tf.keras.models.load_model(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\BMI_EXP0.keras")
race_model0 = tf.keras.models.load_model(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\RACE_EXP0.keras")




# load backtest_data

backtest_three = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\TrainingB.pickle","rb"))

backtest_nrelbc  = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\nrelbcBFL.pickle","rb"))

backtest_race = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\raceBFL.pickle","rb"))

backtest_bmi = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\bmiBFL.pickle","rb"))

Ttargets = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\TargetsB.pickle","rb"))

# class to create 

class Run:
    def __init__(self,bmi_model,race_model,targets):
        self.bmi_model = bmi_model
        self.race_model = race_model
        self.targets = targets
        
        
    # this will give us the metrics for our confusion matrix

    def activation(self,x):
        if x >= 0.1:
            return 1
        else:
            return 0

        
    def confusion_matrix_race(self):
        import matplotlib.pyplot as plt          
        import pandas as pd
        import seaborn as sn
        import numpy as np
        

        
        TP , FP , TN , FN = 0 , 0 , 0 , 0

        start = 0

        race = []

        for val in backtest_race:
            race.append(np.array([backtest_race[start][0],backtest_nrelbc[start][0]]))
            start += 1
        race = np.array(race)


        
        
        preidictionsrace = self.race_model.predict([backtest_three,race]) # race metrics

        print(preidictionsrace)

        
        print(min(preidictionsrace))
        print(max(preidictionsrace))

        for j in range(len(preidictionsrace)):
            if self.activation(*preidictionsrace[j]) == 1 and self.targets[j] == 1:
                TP += 1                
            if self.activation(*preidictionsrace[j]) == 1 and self.targets[j] == 0:
                FP += 1
            if self.activation(*preidictionsrace[j]) == 0 and self.targets[j] == 0:
                TN += 1
            if self.activation(*preidictionsrace[j]) == 0 and self.targets[j] == 1:             
                FN += 1

            
        k = [i for i in self.targets if i == 1]
        p = [i for i in self.targets if i != 1]
        print(len(k))
        print(len(p))
            
        cf = [[TP, FP] ,
              [TN , FN]]

        def MCC(TP,FP,TN,FN):
            import math
            top = (TP * TN) - (FP * FN)
            bottom = math.sqrt((TP+FP)*(TP+FN)*(FN+TP)*(TN+FN))
            if bottom == 0:
                return 0
            return top / bottom

        print(MCC(cf[0][0],cf[0][1],cf[1][0],cf[1][1]))

        import itertools

        print(list(itertools.chain(*cf)))



       
        
    def confusion_matrix_bmi(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sn
        import numpy as np

        TP , FP , TN , FN = 0 , 0 , 0 , 0


        start = 0

        bmi = []

        for val in backtest_race:
            bmi.append(np.array([backtest_bmi[start][0],backtest_nrelbc[start][0]]))
            start += 1
        bmi = np.array(bmi)
        
        predictionsbmi = self.bmi_model.predict([backtest_three,bmi]) # bmi metrics
        # loop through predicions

        print(predictionsbmi)

        print(min(predictionsbmi))
        print(max(predictionsbmi))

        for j in range(len(predictionsbmi)):
            if self.activation(*predictionsbmi[j]) == 1 and self.targets[j] == 1:
                TP += 1                
            if self.activation(*predictionsbmi[j]) == 1 and self.targets[j] == 0:
                FP += 1
            if self.activation(*predictionsbmi[j]) == 0 and self.targets[j] == 0:
                TN += 1
            if self.activation(*predictionsbmi[j]) == 0 and self.targets[j] == 1:             
                FN += 1

        cf = [[TP , FP] ,
              [TN , FN]]

        def MCC(TP,FP,TN,FN):
            import math
            top = (TP * TN) - (FP * FN)
            bottom = math.sqrt((TP+FP)*(TP+FN)*(FN+TP)*(TN+FN))
            if bottom == 0:
                return 0
            return top / bottom

        import itertools

        print(MCC(cf[0][0],cf[0][1],cf[1][0],cf[1][1]))

        print(list(itertools.chain(*cf)))



        
    

k = Run(bmi_model = bmi_model0 , race_model = race_model0 , targets = Ttargets)

k.confusion_matrix_race()

k.confusion_matrix_bmi()
        

        

        


        
        
        
        



