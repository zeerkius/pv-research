
from pandas.tseries.offsets import Second


class Regressor: # can regress linearly or with a skewed gaussian distribution
    def __init__(self,path,fit_race = True):
        self.path = path
        self.fit_race = fit_race
    # if you look carefully bothe race and bmi are preprocessed with a extra 1 so we can perform the dot product 
    # this allows us to have a bias vector
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
        return 0

    def grad(self,y,y_hat,var):
        delta = (var * ((-2 * y) - 1)) / y_hat
        return delta
    def sse(self,x,y):
        err = (x - y) ** 2
        return err

        
    def learning_rate_decay(self,alpha , c = 100 , tau = 0):
        # alphanew = alpha * c / (c + t)
        top = (c / (c + tau))
        new = alpha * top
        return new


    def fit(self ,epochs = 0 , batch_size = 10000 , learning_rate = 0.0000005):
        # get data
        if self.fit_race == True:
            feat = self.preprocessrace()[0]
            targ = self.preprocessrace()[1]
        else:
            feat = self.preprocessbmi()[0]
            targ = self.preprocessbmi()[1]

        # x1 * w1 + ..... xn * wn + b
        weights = [0.05 for x in range(len(feat[0]) - 1)]
        weights.append(0.05) # bias initilized as 2
        # store deltas in error


        def train(weights , Tau):
            import numpy as np
            import itertools
            error_cache = [[]for x in range(len(feat[0]))]

            n = 0  # tick for batch_size 

            sse = []

            for i in range(len(feat)): # there are more ones at then end 
                T = np.dot(feat[i],weights)
                prediction = self.activation(T)
                if n == batch_size:
                    Tau += 1
                    for m in range(len(weights)):
                        weights[m] -= (sum(error_cache[m]) * self.learning_rate_decay(alpha=learning_rate , tau = Tau))
                    alpha = self.learning_rate_decay(alpha=learning_rate , tau = Tau)
                    error_cache = [[] for x in range(len(feat[0]))] # clear cache
                    print(" New Weights " + str(weights) , end = "\n\n")
                    n = 0
                elif n == batch_size - 1:
                    print(" Batch Sum of Squared Loss " + str(sum(sse) /len(sse)) , end = "\n\n")
                    sse = []
                else:
                    if prediction == targ[i]:
                        for j in error_cache:
                            j.append(0)
                    else:
                        sse.append(self.sse(T,prediction))
                        for j in range(len(error_cache)):
                            gt = targ[i][0] # - > avoids it being stored as list
                            xi = feat[i][j]
                            f = self.grad(y = gt , y_hat = T , var = xi)
                            error_cache[j].append(f)
                n += 1
            weights = weights
            weights = [weights , Tau]
            return weights

        frst = train(weights , Tau = 0)
        for val in range(epochs): # global tau for decay
            frst = train(frst[0] , Tau = frst[1])
        return frst[0]




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

        trained_weights = model.fit()

        TP , FP , TN , FN = 0 , 0 , 0 , 0

        index = 0
        for vec in feat:
            guess = np.dot(trained_weights,vec)
            actual_guess = self.activation(guess)
            if actual_guess == 1 and targets[index] == 1:
                TP += 1
            if actual_guess == 1 and targets[index] == 0:
                FP += 1
            if actual_guess == 0 and targets[index] == 0:
                TN += 1
            if actual_guess == 0 and targets[index] == 1:
                FN += 1
            index += 1

        x = [TP , FP , TN , FN]

        print(" Confusion Matrix [TP , FP , TN , FN] " + str(x) , end = "\n\n")

        print(" Total Accuracy " + str(model.acc(x[0],x[1],x[2],x[3])) , end = "\n\n")

        print(" Recall " + str(model.recall(x[0],x[1],x[2],x[3])),  end = "\n\n")
        print(" Precision " + str(model.precision(x[0],x[1],x[2],x[3])) ,end = "\n\n")
        print(" F1-Score " + str(model.f1score(x[0],x[1],x[2],x[3])), end = "\n\n")
        print(" Matthews Correlation Co-effecient " + str(model.MCC(x[0],x[1],x[2],x[3])) ,end = "\n\n")
           
        return x
            
       

race = Regressor(path = r"C:\Users\agboo\nn_revised_arch\bcbs_risk.csv")


race_perf = Testing(path = r"C:\Users\agboo\nn_revised_arch\Logr_backtest.csv")

m = race_perf.predict(race)




# checking both models


bmi  = Regressor(path = r"C:\Users\agboo\nn_revised_arch\bcbs_risk.csv" , fit_race=False)

bmi_perf = Testing(path = r"C:\Users\agboo\nn_revised_arch\Logr_backtest.csv" , fit_race=False)

n = bmi_perf.predict(bmi)


print(m , end = "\n\n")
print(n , end = "\n\n")












        








