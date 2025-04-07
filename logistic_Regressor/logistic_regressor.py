class LogisticRegressor:
    import numba
    import functools
    def __init__(self,path , fit_race):
        self.path = path
        self.fit_race = fit_race

    def preprocessbmi(self):
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
            Y.append(df.iloc[index][6])
            index += 1
        return [X,Y]
        
    def binary_cross_entropy(self,y,y_prime):
        # sigmoid_model = y'
        # bce(y,y`) = - [y * log_2(y') + (1 - y) * log_2(1 - y')]
        import math
        f = y * math.log(y_prime)
        n = (1 - y) * math.log(1-y_prime)
        tot = - (f + n)
        error = tot
        print("                                  ")
        print(" Distribution Difference " + str(error))
        print("                                   ")
        return error

    def log_loss_entropy(self,y,y_prime):
        # sigmoid_model = y'
        # bce(y,y`) = - [y * log_e(y') + (1 - y) * log_e(1 - y')]
        import math
        f = y * math.log(y_prime,2)
        n = (1 - y) * math.log(1-y_prime,2)
        tot = - (f + n)
        error = tot
        print("                                  ")
        print(" Distribution Difference - LogLoss " + str(error))
        print("                                   ")
        return error


    def sigmoid(self,x):
        import numpy as np
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    @numba.njit
    def activation_prediction(p):
        if p >= 0.5:
            return 1
        else:
            return 0

    @staticmethod
    @numba.njit # speeding up learning schedule
    def learning_rate_decay(alpha, c = 10 , tau = 0):
        # alpha_new = alpha * c / c + tau 
        # tau grows from [0, len(D1) % batch_size] , where D1 is the length of the dataset
        top = alpha * c 
        bottom = c + tau
        new = top / bottom
        return new

    def loading_bar(self,stop_event):
        import time
        import sys
        chars = ["-" * x + "@" for x in range(30)] # bars in loading screen + @
        begin = 0
        while not stop_event.is_set():
            sys.stdout.write("\r" + chars[begin % len([chars])])
            sys.stdout.flush()
            begin += 1
            time.sleep(0.01) # animation should go quite fast

    def gradient(self,sig,ground_truth,var):
        diff = sig - ground_truth
        diff = diff * var
        return diff

    def predict(self,backtest,weights,targets):
        # returns confusion matrix
        import numpy as np
        p = 0
        TP , TN , FP , FN = 0 , 0 , 0 , 0
        for arr in backtest:
            values = np.dot(weights,arr) # model
            values = self.sigmoid(values) # sigmoid
            values = self.activation_prediction(values)
            if values == 1 and targets[p] == 1:
                TP += 1
            if values ==  0 and targets[p] == 0:
                TN += 1
            if values == 1 and targets[p] == 0:
                FP += 1
            if values == 0 and targets[p] == 1:
                FN += 1
            p += 1
        return [[TP , FP] , [TN, FN]]

    def MCC(self, TP , FP , TN , FN):
        import math
        top = (TP * TN) - (FP * FN)
        bottom = math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FN))
        if bottom == 0:
            return 0
        mcc = top / bottom
        return mcc

    def fit(self , batch_size = 500 , alpha = 0.000005):
        if self.fit_race == True:
            features = self.preprocessrace()[0]
            targets = self.preprocessrace()[1]
        else:
            features = self.preprocessrace()[0]
            targets = self.preprocessrace()[1]

        ################################

        import numpy as np
        weights = [0.5 for x in range(len(features[0]))] # initializing weights at 0.5
        weights += [1] # bias initialzied as 1
        import numpy as np
        # we have u = [u1,u2....uN] , v = [v1,v2 .... vN] where u is our weight vector and v is our variables
        # we can compute the dot product of these values to get the final value
        n = 0
        t = 1
        error_cache = [[] for x in range(len(features[0]) + 1)] # we will add all these values to there respective error cache to perform gradient calc
        # avoids multiple references
        for val in range(len(targets)):
            if n == batch_size:
                print("             ")
                print(" Batch Limit ")
                print(" Update Weights")
                print("              ")
                for m in range(len(error_cache)):
                    weights[m] = weights[m] - sum(error_cache[m]) * self.learning_rate_decay(alpha = alpha ,tau = t) # minimize loss
                    error_cache[m] = []
                t += 1
                n = 0
                print(" Updated Weight Vector " +  str(weights))
            else:
                features[val].append(1) # this way we can add the x0 == 1 and perform it on our bias so x0*b
                model = np.dot(weights,features[val])
                guess = self.sigmoid(model)
                act_guess = self.activation_prediction(guess)
                self.binary_cross_entropy(targets[val],guess)
                self.log_loss_entropy(targets[val],guess)
                if act_guess == targets[val]:
                    for val in error_cache:
                        val.append(0)
                else:
                    for j in range(len(error_cache)):
                        m = self.gradient(guess,targets[val],features[val][j])
                        error_cache[j].append(m)
            n += 1
        
        return weights
    def linear_sweep(self , model , gt , backtest):
        ## Perform Linear Scale Sweep
        performance_slide = []
        rates = [0.00005 , 0.0005 , 0.005 , 0.05]
        for i in range(len(rates)):
            weight_vector = model.fit(alpha = rates[i])
            cf = self.predict(backtest , weights = weight_vector , targets= gt)
            performance_slide.append(self.MCC(cf[0][0],cf[0][1],cf[1][0],cf[1][1]))
            print(performance_slide)
        print(' Final Verdict ' + str(sum(performance_slide) / len(performance_slide)))
        return performance_slide
    def linear_batch_sweep(self, model , gt , backtest):
        n = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]
        perfomance_slide = []
        for i in range(len(n)):
            weight_vector = model.fit(batch_size = n[i])
            cf = self.predict(backtest , weights = weight_vector , targets= gt)
            perfomance_slide.append(self.MCC(cf[0][0],cf[0][1],cf[1][0],cf[1][1]))
            print(perfomance_slide)
        print(' Final Verdict ' + str(sum(perfomance_slide) / len(perfomance_slide)))
        return perfomance_slide

    def linear_batch_alpha_sweep(self,model,gt,backtest):
        rates = [0.00005 , 0.0005 , 0.005]
        n = [4000,4500,5000,5500]
        perfomance_slide = []
        for i in range(len(n)):
            for j in range(len(rates)):
                weight_vector = model.fit(batch_size = n[i] , alpha = rates[j])
                cf = self.predict(backtest , weights = weight_vector , targets= gt)
                perfomance_slide.append(self.MCC(cf[0][0],cf[0][1],cf[1][0],cf[1][1]))
                print(perfomance_slide)
        print(' Final Verdict ' + str(sum(perfomance_slide) / len(perfomance_slide)))
        return perfomance_slide



    
# create models
race_log = LogisticRegressor(path = r"C:\Users\agboo\logistic_Regressor\bcbs_risk.csv" , fit_race = True)

bmi_log = LogisticRegressor(path = r"C:\Users\agboo\logistic_Regressor\bcbs_risk.csv" , fit_race = False)

# load backtest

class btest:
    def __init__(self,path):
        self.path = path

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
        return [X,Y]


start_test = btest(r"C:\Users\agboo\logistic_Regressor\Logr_backtest.csv")

bmi = start_test.btest_bmi()

race = start_test.btest_race()


r_results = race_log.linear_sweep(race_log,race[1],race[0])

b_results = bmi_log.linear_sweep(bmi_log,bmi[1],bmi[0])


print(" Linear Sweep - Race Total  " + str(sum(r_results)/len(r_results)))
print(" Linear Sweep - BMI Total  " + str(sum(b_results)/len(b_results)))
print("                                                             ")



rr_results = race_log.linear_batch_sweep(race_log,race[1],race[0])

bb_results = bmi_log.linear_batch_sweep(bmi_log,bmi[1],bmi[0])

print(" Linear Batch Sweep - Race Total  " + str(sum(rr_results)/len(rr_results)))
print(" Linear Batch Sweep - BMI Total  " + str(sum(bb_results)/len(bb_results)))


rrr_results = race_log.linear_batch_alpha_sweep(race_log,race[1],race[0])

bbb_results = bmi_log.linear_batch_alpha_sweep(bmi_log,bmi[1],bmi[0])


print(" Linear Batch Alpha Sweep - Race Total  " + str(sum(rrr_results)/len(rrr_results)))
print(" Linear Batch Alpha Sweep - BMI Total  " + str(sum(bbb_results)/len(bbb_results)))










