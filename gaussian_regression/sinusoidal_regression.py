
class Regressor: 
    import numpy as np
    import numba
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

    def sinusoidal(self,x):
        import math
        s = math.sin(x)
        return s

    def sin_grad(self,y,y_prime,T,var):
        import math
        grad = ((y - y_prime) * (math.cos(T) * var))
        return grad

    def learning_rate_decay(self,alpha , c = 1000 , tau = 0):
        # alphanew = alpha * c / (c + t)
        top = (c / (c + tau))
        new = alpha * top
        return new

    def avg(self,arr):
        if not arr:
            return 0
        else:
            num = sum(arr)
            den = len(arr)
            avg = num / den
            return avg

    def fit(self ,epochs = 55 , batch_size = 14361, learning_rate =  0.000005 , decay = False , beta = 0.98 , velocity = 0):
        # get data
        if self.fit_race == True:
            feat = self.preprocessrace()[0]
            targ = self.preprocessrace()[1]
        else:
            feat = self.preprocessbmi()[0]
            targ = self.preprocessbmi()[1]

        ##############################
        import numpy as np
        import statistics

        weight_vector = [1 for x in range(len(feat[0]) - 1)]
        bias = 1
        weight_vector += [bias]
        batch_counter = 0
        tau = 0

        error = [[] for err in range(len(feat[0]))]

        for i in range(epochs):
            query_index = 0
            for query in feat:
                dot_product = np.dot(query,weight_vector)
                sin_output = self.sinusoidal(dot_product)
                y_hat = self.activation(sin_output)
                if y_hat == targ[query_index][0]:
                    continue
                else:
                    if batch_counter == batch_size:
                        batch_counter = 0
                        tau += 1
                        if decay == True:
                            velocity +=  (1 - beta) * (self.learning_rate_decay(tau = tau))
                            for e in range(len(error)):
                                weight_vector[e] -= (weight_vector[e] * sum(error[e]) * velocity)
                        else:
                            velocity += ((1 - beta) * (learning_rate))
                            for e in range(len(error)):
                                weight_vector[e] -= (weight_vector[e] * sum(error[e]) * velocity)
                        print("New Weights " + str(weight_vector) , end = "\n\n")
                        error = [[] for err in range(len(feat[0]))]
                    else:
                        v = 0
                        for err in error:
                            err.append(float(self.sin_grad(y = targ[query_index][0] , y_prime = sin_output , T = dot_product , var = feat[query_index][v])))
                            v += 1
                query_index += 1
                batch_counter += 1
                print(batch_counter , end = "\r")
        return weight_vector
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

    def sinusoidal(self,x):
        import math
        s = math.sin(x)
        return s

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
            guess = self.sinusoidal(guess)
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
            
       

race = Regressor(path = r"C:\Users\agboo\gaussian_regression\bcbs_risk.csv")


race_perf = Testing(path = r"C:\Users\agboo\gaussian_regression\Logr_backtest.csv")

m = race_perf.predict(race)




# checking both models


bmi  = Regressor(path = r"C:\Users\agboo\gaussian_regression\bcbs_risk.csv" , fit_race=False)

bmi_perf = Testing(path = r"C:\Users\agboo\gaussian_regression\Logr_backtest.csv" , fit_race=False)

n = bmi_perf.predict(bmi)


print(m , end = "\n\n")
print(n , end = "\n\n")
