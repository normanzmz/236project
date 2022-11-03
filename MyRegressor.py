from os import stat
from socket import TIPC_CRITICAL_IMPORTANCE
from sklearn.metrics import mean_absolute_error
import utils as ut
import numpy as np
import cvxpy as cp
from scipy import stats


class MyRegressor:
    def __init__(self, alpha,N):
        self.n = N # num of samples 
        # self.m = M # dimensions
        self.weight = np.zeros((self.n, 1))
        self.bias = 0
        self.training_cost = 0   # N * N
        self.alpha = alpha
        # self.iternum = 0
        # self.learningrate = lr

    def select_features(self,trainX,trainY,level):
        # Task 1-3
        feat_num = np.shape(trainX)[1]
        w = np.zeros((feat_num,2))
        # print(feat_num)

        for i in range(0,feat_num):
            w[i,1] = abs(stats.pearsonr(trainY,trainX[:,i])[0])
            w[i,0] = i
        ws = w[w[:,1].argsort()[::-1]]
        # print(ws)

        sel_i = round(level * feat_num)
        # print(level)
        # print(sel_i)
        select_feat_index = []

        for j in range(0,sel_i):
            select_feat_index.append(int(ws[j,0]))
            # select_feat_temp.append(trainX[:,int(ws[j,0])])
            # select_feat = np.transpose(select_feat_temp)
        return select_feat_index # The index List of selected features
        
        
    def select_sample(self, trainX, trainY,percent):
        # Task 1-4
        zscore_list = np.zeros((len(trainY),2))
        zscore_list_temp = np.abs(stats.zscore(trainY))

        for i in range(0,len(trainY)):
            zscore_list[i,1] = zscore_list_temp[i]
            zscore_list[i,0] = i
        z_sort = zscore_list[zscore_list[:,1].argsort()]
        # print(z_sort)

        sample_num = round(percent*len(trainY))
        # print(sample_num)

        selected_trainX = []
        selected_trainY = []

        for j in range(0,sample_num):
            selected_trainX.append(trainX[int(z_sort[j,0]),:])
            selected_trainY.append(trainY[int(z_sort[j,0])])

        print(np.shape(selected_trainX))
        print(np.shape(selected_trainY))
        
        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    # def select_data(self, trainX, trainY):
    #     ''' Task 1-5
    #         Todo: '''
        
    #     return selected_trainX, selected_trainY
    
    
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

        N = 600 used for training 
        M = 500 used for training 
        '''

        # ti = np.zeros((self.n,1))
        # mi = np.zeros((self.n,1))
        # biasi = np.ones((1,1)) * self.bias
        # x_hat = np.concatenate((self.weight, biasi, ti, mi), axis=0)
        # print("the value of x_hat is: \n", x_hat, "\n")
        # print("the shape of x_hat is: \n", np.shape(x_hat), "\n")
    
        # nones = np.ones((self.n, 1))
        # nzeros = np.zeros((self.n, 1))
        # nnzeros = np.zeros((self.n, self.n))
        # I = np.eye(2)
        # C_hat = np.concatenate((nzeros, np.zeros((1,1)), nones *(1./self.n), nones * self.alpha))
        # print("the value of C is: \n", C_hat, "\n")
        # print("the shape of C_hat is: \n", np.shape(C_hat), "\n")
        
        # A1_hat = np.concatenate((-trainX, -nones,-I,nnzeros),axis=1)
        # A2_hat = np.concatenate((trainX, nones,-I,nnzeros),axis=1)
        # A3_hat = np.concatenate((I,nzeros,nnzeros,-I),axis=1)
        # A4_hat = np.concatenate((-I,nzeros,nnzeros,-I),axis=1)
        # A_hat = np.concatenate((A1_hat,A2_hat,A3_hat,A4_hat))
        # B_hat = np.concatenate((-trainY,trainY,nones,nones))

        # print("the value of A_hat is: \n", A_hat, "\n")
        # print("the value of B_hat is: \n", B_hat, "\n")
        # print("the shape of A_hat * x_hat is: \n", np.shape(A_hat @ x_hat), "\n")
        # print("the shape of B_hat is: \n", np.shape(B_hat), "\n")
        # print(A_hat @ x_hat - B_hat)
        # return

        nones = np.ones((np.shape(trainX)[0]))
        mones = np.ones((np.shape(trainX)[1]))

        t = cp.Variable((np.shape(trainX)[0]))
        m = cp.Variable((np.shape(trainX)[1]))
        self.weight = cp.Variable(np.shape(trainX)[1])
        self.bias = cp.Variable(1)

        objective = cp.Minimize((1./np.shape(trainX)[0]) * nones * t + self.alpha * mones * m)

        constraints = [trainY-(trainX @ self.weight + self.bias) <= t,
                       -trainY+(trainX @ self.weight + self.bias) <= t,
                       self.weight <= m,
                       -self.weight <= m]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=True)
        self.weight = self.weight.value
        self.bias = self.bias.value

        print("the result of this LP is: \n", result, "\n")
        # print(self.weight)

        PredY = trainX @ self.weight + self.bias
        train_error = mean_absolute_error(trainY,PredY)
        # print("the train error is: \n", train_error, "\n")
        return train_error
    
    # def train_online(self, trainX, trainY):
    #     ''' Task 2 '''

    #     # we simulate the online setting by handling training data samples one by one
    #     for index, x in enumerate(trainX):
    #         y = trainY[index]

    #         ### Todo:
            
    #     return self.training_cost, train_error

    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        test_error = mean_absolute_error(Y, predY)
        # print("the test error is: \n", test_error, "\n")
        
        return test_error
    
    
    def get_params(self):
        return self.weight, self.bias
    