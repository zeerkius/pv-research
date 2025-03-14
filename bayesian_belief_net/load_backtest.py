# this will be the set of data that we will check its around 3000 patients

class BT():
    def __init__(self,path = str()):
        self.path = path
    def loadtxt(self):
        if ".txt" not in self.path:
            raise ValueError("Invalid File type , allowed file types - .txt ")
        import numpy as np
        import os
        og_path = r"C:\Users\agboo\Downloads\BCBS_Data\Testing Data"
        og_path = os.path.join(og_path,self.path)
        menopause , agerp, density , race , hispanic , bmi , agefirst , nrelbc , brstproc , lastmamm , surgmeno , hrt , invasive , diagnosis = np.loadtxt(og_path, unpack = True , delimiter = ",")
        backtest_dataframe = [menopause , agerp, density , race , hispanic , bmi , agefirst , nrelbc , brstproc , lastmamm , surgmeno , hrt , invasive , diagnosis]
        backtest_dataframe = [list(col) for col in zip(*backtest_dataframe)]
        for val in backtest_dataframe:
            val[len(val)-1] = "?"
            
        return [backtest_dataframe , diagnosis]



def parallel_exce









    





            
        
        

    