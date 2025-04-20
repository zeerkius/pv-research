import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score, confusion_matrix, recall_score, precision_score , matthews_corrcoef

class knn:
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
            Y.append(df.iloc[index][6])
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
            Y.append(df.iloc[index][6])
            index += 1
        X = np.array(X)
        Y = np.array(Y)
        return [X,Y]

    def model_creation(self):
        if self.fit_race == True:
            train = self.preprocessrace()[0]
            targets = self.preprocessrace()[1]
        else:
             train = self.preprocessbmi()[0]
             targets = self.preprocessbmi()[1]


        X_train, X_test, y_train, y_test = train_test_split(train, targets, test_size=0.2, random_state=42)

        # Train the model
        knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust k
        knn.fit(X_train, y_train)
        

        # Predict and evaluate
        y_pred = knn.predict(X_test)

        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Recall: "  , recall_score(y_test, y_pred))
        print("Precision: " , precision_score(y_test, y_pred))
        print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
        print("F1-Score: " , f1_score(y_test, y_pred))
        print("Matthews Correlation Coeffcient: " , matthews_corrcoef(y_test, y_pred))

        return knn


class Test:
    def __init__(self ,path , model , fit_race = True):
        self.path = path
        self.model = model
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
            Y.append(df.iloc[index][6])
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
            Y.append(df.iloc[index][6])
            index += 1
        X = np.array(X)
        Y = np.array(Y)
        return [X,Y]

    def test(self , model):
        if self.fit_race == True:
            train = self.preprocessrace()[0]
            targets = self.preprocessrace()[1]
        else:
             train = self.preprocessbmi()[0]
             targets = self.preprocessbmi()[1]

       
        y_pred = model.predict(train)
        print("Accuracy: ", accuracy_score(targets, y_pred))
        print("Recall: "  , recall_score(targets, y_pred))
        print("Precision: " , precision_score(targets, y_pred))
        print("Confusion Matrix: ", confusion_matrix(targets, y_pred))
        print("F1-Score: " , f1_score(targets, y_pred))
        print("Matthews Correlation Coeffcient: " , matthews_corrcoef(targets, y_pred))

        return model





race_model = knn(r"C:\Users\agboo\k-nn\bcbs_risk.csv")

r = race_model.model_creation()


bmi_model = knn(r"C:\Users\agboo\k-nn\bcbs_risk.csv" , fit_race = False)

b = bmi_model.model_creation()

# testing

final1 = Test(path = r"C:\Users\agboo\k-nn\Logr_backtest.csv" , model = r)

final1.test(model = r)

final2 = Test(path = r"C:\Users\agboo\k-nn\Logr_backtest.csv" , model = b , fit_race = False)

final2.test(model = b)



