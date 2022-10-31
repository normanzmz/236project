from sklearn.metrics import mean_absolute_error
import utils as ut
import numpy as np
import cvxpy as cp


ut.prepare_data_gaussian()["trainX"]

class MyRegressor:
    def __init__(self, alpha):
        self.weight = None
        self.bias = None
        self.training_cost = 0   # N * N
        self.alpha = alpha
        self.iternum = 0
        self.learningrate = lr



        
    def select_features(self):
        ''' Task 1-3
            Todo: '''
        
        return selected_feat # The index List of selected features
        
        
    def select_sample(self, trainX, trainY):
        ''' Task 1-4
            Todo: '''
        
        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    def select_data(self, trainX, trainY):
        ''' Task 1-5
            Todo: '''
        
        return selected_trainX, selected_trainY
    
    
    def train(self, trainX, trainY):
        ''' 
        Task 1-2
        input:
        trainX
        trainY
        self.alpha

        output:
        self.weight 
        self.bias 
        '''

        y = np.zeros(len(trainY))
        
        objective = cp.Minimize(1/len(trainY)+self.alpha*cp.norm1(self.weight))

        constraints = []
        prob = cp.Problem(objective, constraints)

        return train_error
    
    
    def train_online(self, trainX, trainY):
        ''' Task 2 '''

        # we simulate the online setting by handling training data samples one by one
        for index, x in enumerate(trainX):
            y = trainY[index]

            ### Todo:
            
        return self.training_cost, train_error

    
    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        error = mean_absolute_error(Y, predY)
        
        return predY, error
    
    
    def get_params(self):
        return self.weight, self.bias