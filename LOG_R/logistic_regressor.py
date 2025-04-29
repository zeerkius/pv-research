class Regressor: 
    def __init__(self,path,fit_race = True):
        self.path = path
        self.fit_race = fit_race
    # if you look carefully bothe race and bmi are preprocessed with a extra 1 so we can perform the dot product 
    # this allows us to have a bias vector
    # features are all scaled 1 - 0 based on sub category range  c[i] in [0,5] 0.2 , 0.4 , 0.6 , 0.8  , 1
    def preprocessbmi(self):
        import numpy as np
        import pandas as pd
        df = pd.read_csv(self.path)
        X = []
        Y = []
        index = 0
        for val in df.menoapause:
            X.append(df.iloc[index][0:2].tolist() + df.iloc[index][3:6].tolist() + [1])
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
            X.append(df.iloc[index][0:3].tolist() + df.iloc[index][4:6].tolist() + [1])
            Y.append([df.iloc[index][6].tolist()])
            index += 1
        X = np.array(X)
        Y = np.array(Y)
        return [X,Y]
    def MCC(self, TP , FP , TN , FN):
        import math
        top = (TP * TN) - (FP * FN)
        n = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        bottom = math.sqrt(n)
        if bottom == 0:
            return 0
        else:
            mcc = top / bottom
            return mcc
    def acc(self,TP,FP,TN,FN):
        top = (TP + TN)
        bottom = (TP + TN + FP + FN)
        acc = top / bottom
        return acc
    def precision(self,TP,FP,TN,FN):
        top = TP
        bottom = (TP + FP)
        if bottom == 0:
            return 0
        prec = top / bottom
        return prec
    def recall(self,TP,FP,TN,FN):
        top = TP
        bottom = (TP + FN)
        if bottom == 0:
            return 0
        recall = top / bottom
        return recall

    def f1score(self,TP,FP,TN,FN):
        top = 2 * self.precision(TP,FP,TN,FN) * self.recall(TP,FP,TN,FN)
        bottom = self.precision(TP,FP,TN,FN) + self.recall(TP,FP,TN,FN)
        if bottom == 0:
            return 0
        f1 = top / bottom
        return f1

    def activation(self,x):
        if x >= 0.5:
            return 1
        else:
            return 0

    def sigmoid(self,x):
        import numpy as np
        s = (1 / (1 + np.exp(-x)))
        return s

    def bce_grad(self,y,z,var):
        delta = (y - z) * var
        return delta


    def learning_rate_decay(self,alpha , c = 10 , tau = 0):
        # alphanew = alpha * c / (c + t)
        top = (c / (c + tau))
        new = alpha * top
        return new

    def fit(self ,epochs =  50 , batch_size = 14638 , learning_rate =  0.00000005 , decay = True , beta = 0.01):
        # get data
        if self.fit_race == True:
            feat = self.preprocessrace()[0]
            targ = self.preprocessrace()[1]
        else:
            feat = self.preprocessbmi()[0]
            targ = self.preprocessbmi()[1]

        import numpy as np
        import sys



        Tau = 0
        n = 0
        velocity = 0
        stop = len(feat) * epochs
        weights = [0.5 for x in range(len(feat[0])-1)]
        weights.append(-0.15)
        error_cache = [[] for x in range(len(feat[0]))]


        for k in range(epochs):
            for i in range(len(feat)):
                dot_product = np.dot(weights,feat[i])
                s = self.sigmoid(dot_product)
                y_hat = self.activation(s)
                if y_hat == targ[i][0]:
                    continue
                elif n == batch_size:
                    if decay == True:
                        Tau += 1
                        for j in range(len(weights)):
                            velocity = (velocity * beta) + ((1-beta) * sum(error_cache[j]))
                            weights[j] -= (velocity * self.learning_rate_decay(alpha = learning_rate, tau = Tau))
                    else:
                        for j in range(len(weights)):
                            velocity = (velocity * beta) + ((1-beta) * sum(error_cache[j]))
                            weights[j] -= (velocity * learning_rate)
                    print(" New Weights " + str(weights) , end = "\n\n")
                    error_cache = [[] for j in range(len(feat[0]))] # clear the error cache
                    n = 0
                else:
                    feature_index = 0
                    for k in range(len(error_cache)):
                        if dot_product <= 0:
                            error_cache[k].append(0)
                        else:
                            error_cache[k].append(self.bce_grad(y = targ[i][0]  , z  = y_hat  , var = feat[i][feature_index]))
                            feature_index += 1
                n += 1
                print(n , end = "\r")
        return weights



class Testing:
    def __init__(self,path ,fit_race = True):
        self.path = path
        self.fit_race = fit_race

    # we preprocess race and bmi the same way so we can ensure the model computes properly

    def btest_bmi(self):
        import pandas as pd
        import numpy as np
        df = pd.read_csv(self.path)
        X = []
        Y = []
        index = 0

        for val in df.menoapause:
            X.append(df.iloc[index][0:2].tolist() + df.iloc[index][3:6].tolist() + [1])
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
            X.append(df.iloc[index][0:3].tolist() + df.iloc[index][4:6].tolist() + [1])
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



        import numpy as np # do product computation
        import random

        trained_weights = model.fit()

        TP , FP , TN , FN = 0 , 0 , 0 , 0

        rTP , rFP , rTN , rFN = 0 , 0 , 0 , 0

        r = [random.randint(0,1) for x in range(len(feat))] # random guessed used for chi
        

        index = 0
        for vec in feat:
            guess = np.dot(trained_weights,vec)
            actual_guess = self.activation(guess)
            if actual_guess == 1 and targets[index] == 1:
                TP += 1
            if r[index] == 1 and targets[index] == 1:
                rTP += 1
            if actual_guess == 1 and targets[index] == 0:
                FP += 1
            if r[index] == 1 and targets[index] == 0:
                rFP += 1
            if actual_guess == 0 and targets[index] == 0:
                TN += 1
            if r[index] == 0 and targets[index] == 0:
                rTN += 1
            if actual_guess == 0 and targets[index] == 1:
                FN += 1
            if r[index] == 0 and targets[index] == 1:
                rFN += 1
            index += 1

        x = [TP , FP , TN , FN]
        y = [rTP , rFP , rTN , rFN]

        measured = model.acc(x[0],x[1],x[2],x[3])
        rad = model.acc(y[0],y[1],y[2],y[3])

        

        from scipy.stats import chi2_contingency

        table = [ [TP + TN , FP + FN] , 
                 [rTP + rTN , rFP + rFN]]


        chi2_value, p_value, dof, expected = chi2_contingency(table)


        print(" Random Model Accuracy " + str(rad))

        print(f" Chi2: {chi2_value}" , end = "\n\n")
        print(f" p-value: {p_value}" , end = "\n\n")
        print(f" Degrees of Freedom: {dof}" , end = "\n\n")
        print(f" Expected Counts:\n{expected}" , end = "\n\n")

        print(" Confusion Matrix [TP , FP , TN , FN] " + str(x) , end = "\n\n")

        print(" Total Accuracy " + str(model.acc(x[0],x[1],x[2],x[3])) , end = "\n\n")

        print(" Recall " + str(model.recall(x[0],x[1],x[2],x[3])),  end = "\n\n")
        print(" Precision " + str(model.precision(x[0],x[1],x[2],x[3])) ,end = "\n\n")
        print(" F1-Score " + str(model.f1score(x[0],x[1],x[2],x[3])), end = "\n\n")
        print(" Matthews Correlation Co-effecient " + str(model.MCC(x[0],x[1],x[2],x[3])) ,end = "\n\n")
           
        return x
            
       

race = Regressor(path = r"C:\Users\agboo\LOG_R\bcbs_risk.csv")


race_perf = Testing(path = r"C:\Users\agboo\LOG_R\Logr_backtest.csv")

m = race_perf.predict(race)




# checking both models


bmi  = Regressor(path = r"C:\Users\agboo\LOG_R\bcbs_risk.csv" , fit_race=False)

bmi_perf = Testing(path = r"C:\Users\agboo\LOG_R\Logr_backtest.csv" , fit_race=False)

n = bmi_perf.predict(bmi)


print(m , end = "\n\n")
print(n , end = "\n\n")

