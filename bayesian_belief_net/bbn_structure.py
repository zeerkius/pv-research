

bayesian_dict = dict()
bayesian_dict = {"menopause":0 , "agerp":1, "density2":2 , "race":3 , "Hispanic":4 , "bmi":5 , "agefirst":6 , "nrelbc":7 , "brstproc":8 ,
                 "lastmamm":9 , "surgmeno":10 , "hrt":11 , "invasive":12 , "diagnosis":13}

### this is mainly for refrence to the makeshift numpy dataframe we made from the user specified text file

class Bayesian_structure():
    import numba
    import numpy
    def __init__(self,index = 0 , name = str(),left = None , right = None , mid = None):
        self.index = index
        self.name = name
        self.left = left
        self.right = right
        self.mid = mid
        # a three node tree
    def get_children(self,root):
        condition = root.index
        res = []
        if root.left:
            res.append(root.left.index)
        if root.right:
            res.append(root.right.index)
        if root.mid:
            res.append(root.mid.index)
        return {condition:res}
        # in the form P(A|B = T and C = F and D = F) or dict(A|B = T and C = F and D = F)
        # our markov condition is a nodes parents , where we have three parents in this scenario
        # we get they key of the dict and run a search on thre of the iterables as items "B" , "C" , and "D"
        # we update when we find A where B = T and C = F and D = F then return the total probability to output
    def bfs_representation(self,root):
        if not root:
            return "Empty"
        else:
            output = list()
            import collections
            que = collections.deque()
            que.append(root)
            while que:
                length = len(que)
                level = []
                for i in range(length):
                    node = que.popleft()
                    if node:
                        level.append(node.index)
                        que.append(node.left)
                        que.append(node.mid)
                        que.append(node.right)
                if level:
                    output.append(level)
            return output
        ## this will return a bfs on our three node tree with a string representation from the make shift numpy
        #representation from the make shift numpy dataframe
    def bfs_string_representation(self,root):
        if not root:
            return "Empty"
        else:           
            output = list()
            import collections
            que = collections.deque()
            que.append(root)
            while que:
                length = len(que)
                level = []
                for i in range(length):
                    node = que.popleft()
                    if node:
                        level.append(node.name)
                        que.append(node.left)
                        que.append(node.mid)
                        que.append(node.right)
                if level:
                    output.append(level)
            return output  
        ## this will return a bfs on our three node tree with a string representation from the make shift numpy
        #representation from the make shift numpy dataframe
    def serialize_model(self,root,filename): 
        # this will serialize the three tree
        filename += ".pickle"
        import pickle
        BBN_FILE = open(filename,"wb")
        pickle.dump(root,BBN_FILE)
        BBN_FILE.close()
        return BBN_FILE # return closed serialized file
    
    # we can load the data frame
    # we can also load the model post creation

    # we well be using method get_children in bayesian calculation
    # I was thinking of using the create_graph method however we dont need that , It creates a graph every time and is very clunky with construction
    # we already will return the index of the data we need to search with the  get_children method
    # we will allow it return the probability as a feature


    # our markov condition is a nodes parents , where we have three parents in this scenario
    # we get they key of the dict and run a search on thre of the iterables as items "B" , "C" , and "D"
    # we update when we find A where B = T and C = F and D = F then return the total probability to output
    # we then calculate both the case where A = Y and A = N
    # in our case this means A = "cancer diagnosis" or 1 , B = "no cancer diagnosis" or 0
    def prod(self,arr = None):
        if not arr:
            raise("Empty Vector")
        else:
            tot = 1
            for val in arr:
                tot *= val
            return tot    
    def activation(self,x = float(),y = float()):
        # here x will be the positive case of cancer and y being the benign case of no cancer
        if x > 0.00:
            return 1
        elif x > y:
            return 1
        else:
            return 0
    def get_instance(self,arr , index_instances):
        res = []
        # this gives us the value of the instnaces we are concerned with ie the P(A| -> B, C, D ... <-)
        for val in index_instances:
            res.append(float(arr[val]))
        return res
    def get_dict_list(self,root):
        import collections
        import itertools
        res = []
        que = collections.deque()
        que.append(root)
        while que:
            length = len(que)
            level = []           
            for i in range(length):
                node = que.popleft()
                if node:
                    level.append(self.get_children(node))
                    que.append(node.left)
                    que.append(node.mid)
                    que.append(node.right)
            if level:
                res.append(level)
        res = list(itertools.chain(*res))
        return res
    def same(self,x,y):
        return x == y
    




global toolkit
toolkit = Bayesian_structure()




def fit(X,Y,dict_cache):          
    # X is the data we will work with
    # Y is the new instance to guess
    positive = []
    negative = []
    for dic in range(len(dict_cache)):
        for top_key in dict_cache[dic].keys():
            condition = dict_cache[dic][top_key]  # list of indexes in dataframe of each condition list
            if not condition:
                idle_result = 0
                for k in range(len(X[top_key])):
                    idle = 0
                    length = len(X[top_key])
                    if k == Y[top_key]:
                        idle += 1
                        idle_result = idle/length
                        if idle_result <= 0.001:
                            pass
                        else:
                            positive.append(idle_result)                       
                        negative.append(1 - idle_result) 
            elif dic == 0:   
                check = toolkit.get_instance(arr = Y,index_instances = condition) # this gets the values of the condition based on the same index in the new instance
                # we will compare this to every value to get the value of each moment
                import collections
                cache = []
                for val in condition:
                    cache.append(list(X[val]))
                cache.append(list(X[top_key])) # this appends the all the data from said column , as well as the value we are checking for top_key
                # we then transpose this list so we can get the data in the form similar to the text file and we can compare it to the new instance
                cache = [list(col) for col in zip(*cache)] 
                true = 0
                false = 0
                total = 0
                for val in range(len(cache)):
                    compare = cache[val][:-1:]
                    end = cache[val][-1::][0]
                    if toolkit.same(compare,check) == True:
                        if end == 1:
                            true += 1
                        if end == 0:
                            false += 1
                        total += 1
                true = true / total
                false = false / total
                positive.append(true)
                negative.append(false)
            else:
                check = toolkit.get_instance(arr = Y,index_instances = condition) # this gets the values of the condition based on the same index in the new instance
                # we will compare this to every value to get the value of each moment
                cache = []
                for val in condition:
                    cache.append(list(X[val]))
                cache.append(list(X[top_key])) # this appends the all the data from said column , as well as the value we are checking for top_key
                # we then transpose this list so we can get the data in the form similar to the text file and we can compare it to the new instance
                cache = [list(col) for col in zip(*cache)] 
                true = 0
                false = 0
                total = 0
                for val in range(len(cache)):
                    compare = cache[val][:-1:]
                    end = cache[val][-1::][0]
                    if toolkit.same(compare,check) == True:   
                        # we are not checking 1 and 0 as what is needed is the condition not wether its True or False
                        if end == Y[top_key]:
                            true += 1
                        total += 1
                true = true / total
                if true <= 0.001:
                    pass
                else:
                    positive.append(true)                       
                negative.append(1 - true) 
    positive.append(0.505) # prob of being a woman
    positive.append(0.9999999) # probability of getting a somatic mutation
    positive.append(0.9999999)  # probability of getting a somatic mutation in the breast
    # all these independent events multiplied equal 0.00428
    prob_diag = toolkit.prod(negative)
    prob_benign = toolkit.prod(positive)
    return toolkit.activation(x = toolkit.prod(positive), y = prob_diag)


        

       
        
            
                
        
        

                    










          

                

        




        
       
