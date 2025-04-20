
class LinearRegressor:
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

    def sum_of_squared_error(self,y,y_prime):
        loss = (y - y_prime) ** 2
        return float(loss)

    def sse_gradient(self,y,y_prime,var):
        # introduce k = 1/2 
        frst = (y - y_prime)
        frst = frst * (-1)
        delta = frst * var
        return delta

    def activation(self,y_prime):
        if y_prime >= 0.5:
            return 1
        else:
            return 0

    def learning_rate_decay(self,alpha , c = 10 , tau = 0):
        # alphanew = alpha * c / (c + t)
        # or alpha * 1/2 , 1/3 , 1/4 ....
        top = (c / (c + tau))
        new = alpha * top
        return new


    def fit(self , batch_size = 1000 , learning_rate = 0.0005):
        import numpy as np
        import itertools
        # get data
        if self.fit_race == True:
            feat = self.preprocessrace()[0]
            targ = self.preprocessrace()[1]
        else:
            feat = self.preprocessbmi()[0]
            targ = self.preprocessbmi()[1]

        # x1 * w1 + ..... xn * wn + b
        weights = [0.5 for x in range(len(feat[0]) - 1)]
        weights.append(2) # bias initilized as 2
        # store deltas in error
        error_cache = [[]for x in range(len(feat[0]))]

        n = 0  # tick for batch_size 
        Tau = 0 # tick for learning rate decay
        for i in range(len(feat)):
            guess = np.dot(feat[i],weights)
            true_guess = self.activation(guess)
            if n == batch_size:
                Tau += 1
                for m in range(len(weights)):
                    weights[m] -= sum(error_cache[m]) * self.learning_rate_decay(alpha=learning_rate , tau = Tau)
                alpha = self.learning_rate_decay(alpha=learning_rate , tau = Tau)
                error_cache = [[] for x in range(len(feat[0]))] # clear cache
                print(" New Weights " + str(list(itertools.chain(*weights))) , end = "\n\n")
                n = 0
            else:
                if true_guess == targ[i]:
                    for j in error_cache:
                        j.append(0)
                else:
                    for j in range(len(error_cache)):
                        error_cache[j].append(self.sse_gradient(targ[i],guess,feat[i][j]))
            n += 1
        weights = list(itertools.chain(*weights))
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

        trained_weights = model.fit()

        TP , FP , TN , FN = 0 , 0 , 0 , 0

        index = 0
        for vec in feat:
            guess = np.dot(trained_weights,vec)
            actual_guess = self.activation(guess)
            if actual_guess == 1 and targets[index] == 1:
                TP += 1
            if actual_guess == 1 and targets[index] == 1:
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
            
       

race = LinearRegressor(path = r"C:\Users\agboo\nn_revised_arch\bcbs_risk.csv")


race_perf = Testing()










        








