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
        if x > 0.0:
            return 1
        else:
            return 0

        
    def confusion_matrix_race(self):
        import matplotlib.pyplot as plt          
        import pandas as pd
        import seaborn as sn
        

        
        TP , FP , TN , FN = 0 , 0 , 0 , 0
        
        
        preidictionsrace = self.race_model.predict([backtest_three,backtest_race,backtest_nrelbc]) # race metrics

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
            
        cf = [[TP/len(k), FP/len(k)] ,
              [TN/len(p) , FN/len(p)]]
        


        df = pd.DataFrame(cf, index = [i for i in "TF"] , columns = [j for j in "PN"])
        plt.figure(figsize = (10,7))       
        sn.heatmap(df , annot = True)
        plt.title("> 0.0 Threshold" , fontsize = 16 , fontweight="bold")
        plt.savefig(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\NN_Data_Visualizations\heatmap_race06.png")
        
        
        
    def confusion_matrix_bmi(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sn
        TP , FP , TN , FN = 0 , 0 , 0 , 0
        
        predictionsbmi = self.bmi_model.predict([backtest_three,backtest_bmi,backtest_nrelbc]) # bmi metrics
        # loop through predicions

        for j in range(len(predictionsbmi)):
            if self.activation(*predictionsbmi[j]) == 1 and self.targets[j] == 1:
                TP += 1                
            if self.activation(*predictionsbmi[j]) == 1 and self.targets[j] == 0:
                FP += 1
            if self.activation(*predictionsbmi[j]) == 0 and self.targets[j] == 0:
                TN += 1
            if self.activation(*predictionsbmi[j]) == 0 and self.targets[j] == 1:             
                FN += 1

            
        k = [i for i in self.targets if i == 1]
        p = [i for i in self.targets if i != 1]
        print(len(k))
        print(len(p))
            
        cf = [[TP/len(k) , FP/len(k)] ,
              [TN/len(p) , FN/len(p)]]
        df = pd.DataFrame(cf, index = [i for i in "TF"] , columns = [ j for j in "PN"])
        plt.figure(figsize = (10,7))    
        plt.title("> 0.0 Threshold" , fontsize = 16 , fontweight="bold")
        sn.heatmap(df , annot = True)
        plt.savefig(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\NN_Data_Visualizations\heatmap_bmi06.png")
        

        
    

k = Run(bmi_model = bmi_model0 , race_model = race_model0 , targets = Ttargets)

k.confusion_matrix_race()

k.confusion_matrix_bmi()
        

        

        


        
        
        
        



