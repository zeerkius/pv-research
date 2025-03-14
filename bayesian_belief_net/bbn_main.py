
from bbn_structure import Bayesian_structure
from bbn_structure import fit
from load_backtest import BT
class DataVisualizer:
    # we first define two list that will be used to display the data for the graph
    # they will be the data points after instantiation of the class
    def __init__(self,feature,predict):
        self.feature = feature
        self.predict = predict
    # this function will be raised if someone decides to use [DATE] as a input for data
    # we will use backwards compatible UTF-8 Encoding 


    def coversion(self,fmt,encoding = "utf8"):
        import matplotlib
        def converter(byte):
            res = byte.decode(encoding)
            return matplotlib.mdates.datestr2num(res)
        return converter
        # if creating instance with numpy use converters = {0:coversion(%Y-%m-%d)}
        # when putting data in the graph we would have to do this so we could plot appropiatley
        # plt.gca().set_yaxis().get_major_formatter().set_useOffset(True)
        # XXXXX is the subplot name
        # XXXXX.xaxis.set_major_formatter(mdates.DateFormatter(%Y-%m-%d))
        # XXXXX.plot(Date_Data,Data_Data)
        # plt.grid = True
        # plt.subplots_adjust(0.1)
        # plt.show()      
    # this function creates a graph after getting the said feature and predictive feature


    def create_graph(self,string = None):
        import os
        import matplotlib.pyplot as plt
        import matplotlib.figure
        import numpy as np
        if "png" not in string:
            raise ValueError("Valid file types are only .png")
        else:           
            path = r"C:\Users\agboo\Downloads\BCBS_Data\Data_Visuals"
            res_path = os.path.join(path,string)
            plt.switch_backend("Agg")
            #fig1 = plt.figure(figsize = (10,7) , facecolor = "grey")
            if len(self.predict) != len(self.feature):
                raise ValueError("Iterables are not the same length") # error message for later use
            else:
                # first we do computation                
                feature_set = sorted(set(self.feature)) # we set this as list so we can do the operations , however its a set of all features , sorted so we can keep track
                k = len(feature_set)
                computation_arr = np.zeros(k) # this will be for the positive class
                comp_arr = np.zeros(k) # this will be for the negative class
                positive = []
                negative = []
                for i in range(len(self.feature)):
                    if self.predict[i] == 1:# filters for positive class
                        computation_arr[feature_set.index(self.feature[i])] += 1  # gives us the probability for the feature being positive
                        positive.append(self.feature[i])
                        
                    else:
                        comp_arr[feature_set.index(self.feature[i])] += 1  # gives us the probability for the feature to be negative
                        negative.append(self.feature[i])
                        
                # no we will create our histograms for each class

                fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
                axs[0].hist(np.array(positive) , bins=len(computation_arr)) # positive hist , vector is a numpy array
                axs[1].hist(np.array(negative), bins=len(comp_arr)) # negative hist vector is a numpy array
                plt.title("HRT & Diagnosis")
                plt.xlabel('Feature Type')
                plt.ylabel("Count")
                           
                fig.savefig(res_path)
                
                lis = [computation_arr,comp_arr]
                
                for m in range(len(computation_arr)):
                    computation_arr[m] = computation_arr[m]/(computation_arr[m] + comp_arr[m]) # gives probability of that feature positive
                for n in range(len(comp_arr)):
                    comp_arr[n] = comp_arr[n] / (computation_arr[n] + comp_arr[n]) # probability of thet feature being negative
                # we will return both arrays 
                # now we will create the histgram
                return [[computation_arr,comp_arr],[lis]]    # return in order of [[positive[n],positive[n+1]....],[neagtive[n],negative[n+1].........]]
class Get_Patient_Features:
    def __init__(self,string,binary_file_name = ""): # string is the name of the TXT file , # b_f_n is the name of the file the user wants to create
        self.binary_file_name = binary_file_name
        self.string = string
    def get_features(self):
        import numpy as np
        import os
        import pickle
        res = r"C:\Users\agboo\Downloads\BCBS_Data\SerializedData"
        if ".txt" not in self.string:
            raise ValueError("Method only takes .txt files")
        elif "." in self.binary_file_name:
            raise ValueError("Invalid Name - Remove file extension")
        else:          
            res_res = os.path.join(res,self.string)
            # we will load from a text file
            menopause , agerp , density , race , Hispanic , bmi , agefirst , nrelbc , brstproc , lastmamm , surgmeno , hrt , invasive , diagnosis = np.loadtxt(res_res,
            unpack = True , delimiter = ",")
            # create array in the same order as unpacking M X 1
            dataframe = np.array([menopause , agerp , density , race , Hispanic , bmi , agefirst , nrelbc , brstproc , lastmamm , surgmeno , hrt , invasive , diagnosis])
            # we will now create the pickle file
            if not self.binary_file_name:            
                res_binary = self.binary_file_name + ".pickle"
                serial = open(res_binary,"wb")
                pickle.dump(dataframe,serial)
                serial.close()
            else:
                pass
        return dataframe
    
BBNBMI = Bayesian_structure(13,"diagnosis")

bmi_top = BBNBMI


## Create Three Tree for BMI model


BBNBMI.left = Bayesian_structure(7,"nrelbc")
BBNBMI.mid = Bayesian_structure(0,"menopause")
BBNBMI.right = Bayesian_structure(5,"BMI")
BBNBMI.left.left = Bayesian_structure(9,"lastmamm")
BBNBMI.left.left.left = Bayesian_structure(2,"density")
BBNBMI.mid.left = Bayesian_structure(1,"agerp")
BBNBMI.mid.mid = Bayesian_structure(10,"surgmeno")
BBNBMI.mid.mid.mid = Bayesian_structure(11,"HRT")
BBNBMI.mid.left.left = Bayesian_structure(6,"agefirst")
BBNBMI.mid.left.mid = Bayesian_structure(8,"brst_proc")


# Create Three Tree For Race Model
## only structure diffrence in BMI Node - > Race Node

BBNRACE = Bayesian_structure(13,"diagnosis")

race_top = BBNRACE

BBNRACE.left = Bayesian_structure(7,"nrelbc")
BBNRACE.mid = Bayesian_structure(0,"menopause")
BBNRACE.right = Bayesian_structure(5,"BMI")
BBNRACE.left.left = Bayesian_structure(9,"lastmamm")
BBNRACE.left.left.left = Bayesian_structure(2,"density")
BBNRACE.mid.left = Bayesian_structure(1,"agerp")
BBNRACE.mid.mid = Bayesian_structure(10,"surgmeno")
BBNRACE.mid.mid.mid = Bayesian_structure(11,"HRT")
BBNRACE.mid.left.left = Bayesian_structure(6,"agefirst")
BBNRACE.mid.left.mid = Bayesian_structure(8,"brst_proc")


# now will serialize both models so we can use them in our main code and test

#bmi_top.serialize_model(bmi_top,"bmitree")
#race_top.serialize_model(race_top,"racetree")


# now we will globally define the dataframe from main
global nframe

nframe = Get_Patient_Features("TXT_TOTXT.txt")
nframe = nframe.get_features()
c_dict = bmi_top.get_dict_list(bmi_top)
d_dict = race_top.get_dict_list(race_top)

# since this a CPU bound task , and the time complexity of our fit function is really big we will utilize multiprocessing
# as multithreading is unwaise with a CPU bound task due to python's GIL
# we have a intel core i7 6th gen , AMD allows for hyper-threading meaning we have really 8 logical processors , so there is
# 2 virtual translations for each physical hardware core meaning we can do up to 8 processes at a time


import multiprocessing as mp

# Running on Core i7 6th Gen , 4 Physical Cores 8 Virtual Logical Processors
# we will use this for our batch fit
#logical_processors  = mp.cpu_count()

logical_processors = mp.cpu_count()


backtest1 = BT("Backtets_csv.txt")

instance_pool = backtest1.loadtxt()[0]
ground_truth = backtest1.loadtxt()[1]

# make list of tuple inputs

BMI_ACCURACY = 0
RACE_ACCURACY = 0
Sensitivity1 = 0
Sensitivity2 = 0

inputs1 = []
inputs2 = []

for val in instance_pool:
    inputs1.append((nframe,val,c_dict))
    
for val in instance_pool:
    inputs2.append((nframe,val,d_dict))


if __name__ == "__main__":
    with mp.Pool(processes=logical_processors) as pool:
        results1 = pool.starmap(fit,inputs1)
        print(results1)
    


      
        


            
 
    

