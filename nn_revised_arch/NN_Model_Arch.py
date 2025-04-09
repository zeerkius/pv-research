import tensorflow
import keras

class NN:
    def __init__(self,path,fit_race = True):
        self.path = path
        self.fit_race = fit_race
    def preprocessbmi(self):
        import numpy as np
        import pandas as pd
        df = pd.read_csv(self.path)
        X = []
        Y = []
        index = 0
        for val in df.menoapause:
            X.append(df.iloc[index][0:2].tolist() + df.iloc[index][3:6].tolist())
            Y.append([df.iloc[index][6].tolist()])
            index += 1
        X = np.array(X)
        Y = np.array(Y)
        return [X,Y]
    def preprocessrace(self):
        import pandas as pd
        import numpy as np
        df = pd.read_csv(self.path)
        X = []
        Y = []
        index = 0
        for val in df.menoapause:
            X.append(df.iloc[index][0:3].tolist() + df.iloc[index][4:6].tolist())
            Y.append([df.iloc[index][6].tolist()])
            index += 1
        X = np.array(X)
        Y = np.array(Y)
        return [X,Y]
    def create_model(self ,model_name):
        if self.fit_race == True:
            feat = self.preprocessrace()[0]
            targets = self.preprocessrace()[1]
        else:
            feat = self.preprocessbmi()[0]
            targets = self.preprocessbmi()[1]

        import tensorflow
        import keras
        from tensorflow.keras.layers import Dense , Input
        import numpy as np
        # create shape

        majority = len([i for i in targets if targets[i] == 0])
        minority = len(targets) - majority
        ratio = minority/majority

        input_shape = tensorflow.keras.Input(feat[0].shape)

        dense1 = tensorflow.keras.layers.Dense(3 , activation = "relu")(input_shape)

        dense2 = tensorflow.keras.layers.Dense(3 , activation = "relu")(dense1)

        dense3 = tensorflow.keras.layers.Dense(3 , activation = "relu")(dense2)

        final = tensorflow.keras.layers.Dense(1 , activation = "sigmoid")(dense3)


        model1 = keras.Model(inputs = input_shape , outputs =  final , name = '9631_model')

        model1.compile(loss = tensorflow.keras.losses.CosineSimilarity(
    axis=-1,
    reduction='sum_over_batch_size',
    name='cosine_similarity'
) , optimizer= "SGD" , metrics = ['accuracy'])

        model1.fit(feat ,targets , batch_size = 700 , validation_split = 0.30 , epochs = 750)

        model_name += ".keras"

        model1.save(model_name)

        return model_name 

class Testing:
    def __init__(self,path ,fit_race = True):
        self.path = path
        self.fit_race = fit_race

    def btest_bmi(self):
        import pandas as pd
        import numpy as np
        df = pd.read_csv(self.path)
        X = []
        Y = []
        index = 0

        for val in df.menoapause:
            X.append(df.iloc[index][0:2].tolist() + df.iloc[index][3:6].tolist())
            Y.append(df.iloc[index][6])
            index += 1
        X = np.array(X)
        Y = np.array(Y)
        return [X,Y]

    def btest_race(self):
        import pandas as pd
        import numpy as np
        df = pd.read_csv(self.path)
        X = []
        Y = []
        index = 0
        for val in df.menoapause:
            X.append(df.iloc[index][0:3].tolist() + df.iloc[index][4:6].tolist())
            Y.append(df.iloc[index][6])
            index += 1
        X = np.array(X)
        Y = np.array(Y)
        return [X,Y]
    def activation(self,x):
        if x >= 0.5:
            return 1
        else:
            return 0
    def predict(self ,model):
        if self.fit_race == True:
            feat = self.btest_race()[0]
            targets = self.btest_race()[1]
        else:
            feat = self.btest_bmi()[0]
            targets = self.btest_bmi()[1]
        

        predictions = model.predict(feat)
        print(predictions)

        TP , FP , TN , FN = 0 , 0 , 0 , 0
        index = 0
        for val in predictions:
            if self.activation(val[0]) == 1 and targets[index] == 1:
                TP += 1
            if self.activation(val[0]) == 1 and targets[index] == 1:
                FP += 1
            if self.activation(val[0]) == 0 and targets[index] == 0:
                TN += 1
            if self.activation(val[0]) == 0 and targets[index] == 1:
                FN += 1
            index += 1
        x = [TP , FP , TN , FN]

        print(x)
           
        return x
            
       

race = NN(path = r"C:\Users\agboo\nn_revised_arch\bcbs_risk.csv")

file_name  = race.create_model(model_name = "race_model_nn_revised")


b_race = Testing(r"C:\Users\agboo\nn_revised_arch\Logr_backtest.csv")

f = tensorflow.keras.models.load_model(file_name)


p = b_race.predict(f)











        








