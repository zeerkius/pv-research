import pandas as pd
import numpy as np
import numba
import matplotlib.pyplot as plt


# pre_process data

data_frame = pd.read_csv(r"C:\Users\agboo\Downloads\BCBS_Data\Training Data\bcbs_risk.csv")


# instead of blindly just adding relu to the neural network we will make observations of the data
# we will do this to influence the activation functions we will use as well as find trends in the data

# we will use the conventional X and Y

X = []
Y = []
bmiX = []
raceX = []
nrelbc_arr = []

def pre_process(X,Y,W,Z,ZZ):
    for val in data_frame.diagnosis:   
        X.append(np.array(data_frame.iloc[val][0:3].tolist()))
        Y.append(np.array(data_frame.iloc[val][6:7].tolist()))
        W.append(np.array(data_frame.iloc[val][3:4].tolist()))
        Z.append(np.array(data_frame.iloc[val][2:3].tolist()))
        ZZ.append(np.array(data_frame.iloc[val][5:6].tolist()))
    return [np.array(X),np.array(Y),np.array(W),np.array(Z),np.array(ZZ)]
    
compressed_vectors = pre_process(X,Y,bmiX,raceX,nrelbc_arr) # both training data and targets






# race historgram

tuples_race_diag = list(zip(data_frame.race.tolist(),data_frame.diagnosis.tolist())) #

def create_graph(arr):
    import numpy as np
    import matplotlib.pyplot as plt
    
    arr = sorted(arr)
       
    x = [col[0] for col in arr if col[1] == 1] # getting vector for x axis
    y = [col[0] for col in arr if col[1] != 1]
    
    fig , axs = plt.subplots(1 , 2 , sharey = True , tight_layout = True)
    axs[0].hist(np.array(x), bins = len(set(x)))
    axs[1].hist(np.array(y), bins = len(set(y)))
    plt.title("Race & Diagnosis BFL")
    plt.xlabel("Race Category")   # white  1 , asian_pacific american 2 , black 3 , native american 4 , other/mixed 5
    plt.ylabel("Count of values")
    plt.savefig("Race_&_Diagnosis_BFL.png")
    

create_graph(tuples_race_diag)


# bmi histogram

tuples_bmi_diag = list(zip(data_frame.bmi.tolist(),data_frame.diagnosis.tolist())) #

def create_graph0(arr):
    import matplotlib.pyplot as plt
    import numpy as np
    
    arr = sorted(arr)
       
    x = [col[0] for col in arr if col[1] == 1] # getting vector for x axis
    y = [col[0] for col in arr if col[1] != 1]
    
    fig , axs = plt.subplots(1 , 2 , sharey = True , tight_layout = True)
    axs[0].hist(np.array(x), bins = len(set(x)))
    axs[1].hist(np.array(y), bins = len(set(y)))
    plt.title("BMI & Diagnosis BFL")
    plt.xlabel("BMI Category")
    plt.ylabel("Count of values")
    plt.savefig("BMI_&_Diagnosis_BFL.png")
    
     
create_graph0(tuples_bmi_diag)

# menopause diagram

tuples_menoapuase_diag = list(zip(data_frame.menoapause.tolist(),data_frame.diagnosis.tolist()))


def create_graph1(arr):
    import matplotlib.pyplot as plt
    import numpy as np
    
    arr = sorted(arr)
       
    x = [col[0] for col in arr if col[1] == 1] # getting vector for x axis
    y = [col[0] for col in arr if col[1] != 1]
    
    fig , axs = plt.subplots(1 , 2 , sharey = True , tight_layout = True)
    axs[0].hist(np.array(x), bins = len(set(x)))
    axs[1].hist(np.array(y), bins = len(set(y)))
    plt.title("Menopause & Diagnosis BFL")
    plt.xlabel("Y/N Menopause")
    plt.ylabel("Count of values")
    plt.savefig("Menopause & Diagnosis BFL.png")
    
     
create_graph1(tuples_menoapuase_diag)

# agefirst child diagram 

tuples_agefirst_diag = list(zip(data_frame.agefirst.tolist(),data_frame.diagnosis.tolist()))

def create_graph2(arr):
    import matplotlib.pyplot as plt
    import numpy as np
    
    arr = sorted(arr)
       
    x = [col[0] for col in arr if col[1] == 1] # getting vector for x axis
    y = [col[0] for col in arr if col[1] != 1]
    
    fig , axs = plt.subplots(1 , 2 , sharey = True , tight_layout = True)
    axs[0].hist(np.array(x), bins = len(set(x)))
    axs[1].hist(np.array(y), bins = len(set(y)))
    plt.title("Agefirst & Diagnosis BFL")
    plt.xlabel("Age During First Preg")
    plt.ylabel("Count of values")
    plt.savefig("Agefirst & Diagnosis BFL.png")
    
     
create_graph2(tuples_agefirst_diag)


# agerp diagram

tuples_agerp_diag = list(zip(data_frame.agerp.tolist(),data_frame.diagnosis.tolist()))

def create_graph3(arr):
    import matplotlib.pyplot as plt
    import numpy as np
    
    arr = sorted(arr)
       
    x = [col[0] for col in arr if col[1] == 1] # getting vector for x axis
    y = [col[0] for col in arr if col[1] != 1]
    
    fig , axs = plt.subplots(1 , 2 , sharey = True , tight_layout = True)
    axs[0].hist(np.array(x), bins = len(set(x)))
    axs[1].hist(np.array(y), bins = len(set(y)))
    plt.title("Agerp & Diagnosis BFL")
    plt.xlabel("Age Category")
    plt.ylabel("Count of values") 
    plt.savefig("Agerp & Diagnosis BFL.png")
    
     
create_graph3(tuples_agerp_diag)

# nrelbc diagram

tuples_nrelbc_diagram = list(zip(data_frame.nrelbc.tolist(),data_frame.diagnosis.tolist()))

def create_graph4(arr):
    import matplotlib.pyplot as plt
    import numpy as np
    
    arr = sorted(arr)
       
    x = [col[0] for col in arr if col[1] == 1] # getting vector for x axis
    y = [col[0] for col in arr if col[1] != 1]
    
    fig , axs = plt.subplots(1 , 2 , sharey = True , tight_layout = True)
    axs[0].hist(np.array(x), bins = len(set(x)))
    axs[1].hist(np.array(y), bins = len(set(y)))
    plt.title("nrelbc & Diagnosis BFL")
    plt.xlabel("Near Relative With BC")
    plt.ylabel("Count of values")
    plt.savefig("nrelbc & Diagnosis BFL.png")
    
create_graph4(tuples_nrelbc_diagram)


# serializing data for both backtest and model training


def serialize_training(X,Y,W,Z,ZZ):
    import pickle
    p = open("Training_NN.pickle","wb")
    pickle.dump(X,p)
    p.close()
    
    k = open("Targets_NN.pickle","wb")
    pickle.dump(Y,k)
    k.close()
        
    m = open("race_train_NN.pickle","wb")
    pickle.dump(W,m)
    m.close()
    
    n = open("bmi_train_NN.pickle","wb")
    pickle.dump(Z,n)
    n.close()
    
    nn = open("nrelbc_train_NN.pikcle","wb")
    pickle.dump(ZZ,nn)
    nn.close()
    
    # manually creating files
    
def serialize_backtest():
    import numpy as np
    import pickle
    menopause , agerp , race , bmi , age_first , nrelbc , diagnosis = np.loadtxt(r"C:\Users\agboo\Binary_Focal_Cross_Entropy_NN\BackGround_Research\BC_NN_BFL_backtest.txt" ,
    unpack = True , delimiter = ",")
    
    k_columns = [menopause , agerp , age_first]
    
    k_rows = [list(row) for row in zip(*k_columns)]
    
    k_rows = np.array(k_rows)

    BTarget = np.array(diagnosis)
    

    m = open("TrainingB.pickle","wb")
    pickle.dump(k_rows,m)
    m.close()
    
    n = open("TargetsB.pickle","wb")
    pickle.dump(BTarget,n)
    n.close()
    
    k = open("raceBFL.pickle","wb")
    pickle.dump(race,k)
    k.close()
    
    f = open("bmiBFL.pickle","wb")
    pickle.dump(bmi,f)
    f.close()
    
    ff = open("nrelbcBFL.pickle","wb")
    pickle.dump(nrelbc,ff)
    ff.close()
    

serialize_training(compressed_vectors[0],compressed_vectors[1],compressed_vectors[2],compressed_vectors[3],compressed_vectors[4])
serialize_backtest()





    

    
    
    
    
    













