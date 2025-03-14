import tensorflow as tf
import keras
import pickle

# load Neural Networks from keras Models



bmi_model0 = tf.keras.models.load_model(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\BMIEXP0.keras", compile = False)
race_model0 = tf.keras.models.load_model(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_models\RACE_EXP0.keras" , compile = False)

bmi_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy( alpha = 50,gamma = 50) , optimizer = "SGD" , metrics = ["accuracy"])
race_model0.compile(loss = tf.keras.losses.BinaryFocalCrossentropy(alpha = 50,gamma = 50) , optimizer = "SGD" , metrics = ["accuracy"])


# load backtest_data

backtest_three = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\TrainingB.pickle","rb"))

backtest_nrelbc  = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\nrelbcBFL.pickle","rb"))

backtest_race = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\raceBFL.pickle","rb"))

backtest_bmi = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\bmiBFL.pickle","rb"))

targets = pickle.load(open(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\Keras & Pickle Files\serialized_data\TargetsB.pickle","rb"))

# class to create 

class Run:
    def __init__(self,bmi_model,race_model,targets):
        self.bmi_model = bmi_model
        self.race_model = race_model
        self.targets = targets
        
        
    # this will give us the metrics for our confusion matrix

    def activation(self,x):
        if x >= 0.5:
            return 1
        else:
            return 0
        
    def confusion_matrix_race(self):
        import matplotlib.pyplot as plt          
        import pandas as pd
        import seaborn as sn
        

        
        TP , FP , TN , FN = 0 , 0 , 0 , 0
        
        
        preidictionsrace = self.race_model.predict([backtest_three,backtest_race,backtest_nrelbc]) # race metrics
        
        # loop through predictions and 
        for j in range(len(preidictionsrace)):
            if self.activation(preidictionsrace[j][0]) == 1.0 and self.targets[j] == 1.0:
                TP += 1                
            elif self.activation(preidictionsrace[j][0]) == 1.0 and self.targets[j] == 0:
                FP += 1
            elif self.activation(preidictionsrace[j][0]) == 0 and self.targets[j] == 0:
                TN += 1
            elif self.activation(preidictionsrace[j][0]) == 0 and self.targets[j] == 1.0:             
                FN += 1
            else:
                pass
                
        cf = [[TP/len(self.targets) , FP/len(self.targets)] ,
              [TN/len(self.targets) , FN/len(self.targets)]]
        df = pd.DataFrame(cf, index = [i for i in "TF"] , columns = [ j for j in "PN"])
        plt.figure(figsize = (10,7))       
        sn.heatmap(df , annot = True)        
        plt.savefig("heatmap_race.png")
        
        
        
    def confusion_matrix_bmi(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sn
        TP , FP , TN , FN = 0 , 0 , 0 , 0
        
        predictionsbmi = self.bmi_model.predict([backtest_three,backtest_bmi,backtest_nrelbc]) # bmi metrics
        # loop through predicions

        for j in range(len(predictionsbmi)):
            if self.activation(predictionsbmi[j][0]) == 1 and self.targets[j] == 1:
                TP += 1                
            elif self.activation(predictionsbmi[j][0]) == 1 and self.targets[j] == 0:
                FP += 1
            elif self.activation(predictionsbmi[j][0]) == 0 and self.targets[j] == 0:
                TN += 1
            elif self.activation(predictionsbmi[j][0]) == 0 and self.targets[j] == 1:             
                FN += 1
            else:
                pass
            
        cf = [[TP/len(self.targets) , FP/len(self.targets)] ,
              [TN/len(self.targets) , FN/len(self.targets)]]
        df = pd.DataFrame(cf, index = [i for i in "TF"] , columns = [ j for j in "PN"])
        plt.figure(figsize = (10,7))       
        sn.heatmap(df , annot = True)
        plt.savefig("heatmap_bmi.png")
        

        
    

k = Run(bmi_model = bmi_model0 , race_model = race_model0 , targets = targets)

k.confusion_matrix_race()

k.confusion_matrix_bmi()
        

        

        


        
        
        
        



