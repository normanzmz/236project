from os import stat
from socket import TIPC_CRITICAL_IMPORTANCE
from sklearn.metrics import mean_absolute_error
import utils as ut
import numpy as np
import cvxpy as cp
import math as ma
from scipy import stats


class MyRegressor:
    def __init__(self, alpha):
        self.weight = None
        self.bias = None
        self.training_cost = 0 # N * M
        self.alpha = alpha

    def select_features(self,trainX,trainY,level):
        # Task 1-3
        feat_num = np.shape(trainX)[1]
        w = np.zeros((feat_num,2))

        for i in range(0,feat_num):
            w[i,1] = abs(stats.pearsonr(trainY,trainX[:,i])[0])
            w[i,0] = i

        ws = w[w[:,1].argsort()[::-1]]
        sel_i = round(level * feat_num)
        select_feat_index = []

        for j in range(0,sel_i):
            select_feat_index.append(int(ws[j,0]))

        return select_feat_index # The index List of selected features
         
    def select_sample(self, trainX, trainY,percent):
        # Task 1-4
        # find the difference between the predict y and y 
        sample_num = round(percent*len(trainY))
        PredY = trainX @ self.weight + self.bias
        Y_error_list = np.zeros((len(trainY),2))
        Y_error_list_temp = np.abs(trainY-PredY)

        for i in range(0,len(trainY)):
            Y_error_list[i,1] = Y_error_list_temp[i]
            Y_error_list[i,0] = i

        y_sort_list = Y_error_list[Y_error_list[:,1].argsort()]
        selected_trainX = []
        selected_trainY = []

        for j in range(0,sample_num):
            selected_trainX.append(trainX[int(y_sort_list[j,0]),:])
            selected_trainY.append(trainY[int(y_sort_list[j,0])])

        return selected_trainX, selected_trainY    # A insubset of trainX and trainY


    def select_data(self, trainX, trainY, cost):
        # Task 1-5
        cost_init = len(trainY) * len(trainX[0])
        print(len(trainX[0]))
        if len(trainX[0]) >= 300:
            threshold = 0.8

            if cost >= threshold:
                new_cost = round(ma.sqrt(cost),2)
                sel_x_temp, selected_trainY = self.select_sample(trainX,trainY,new_cost)
                sel_feat_ind = self.select_features(np.array(sel_x_temp),np.array(selected_trainY),new_cost)
                sel_feat_num = new_cost*np.shape(trainX)[1]
                select_trainX_temp = []

                for i in range(0,round(sel_feat_num)):
                    select_trainX_temp.append(np.array(sel_x_temp)[:,sel_feat_ind[i]])
                    selected_trainX = np.transpose(select_trainX_temp)

            elif cost < threshold:
                sel_feat_ind2 = self.select_features(np.array(trainX),np.array(trainY),cost)
                sel_feat_num2 = cost*np.shape(trainX)[1]
                select_trainX_temp2 = []

                for i in range(0,round(sel_feat_num2)):
                    select_trainX_temp2.append(np.array(trainX)[:,sel_feat_ind2[i]])
                    selected_trainX = np.transpose(select_trainX_temp2)
                    selected_trainY = trainY

        elif len(trainX[0]) < 300:
                sel_feat_ind3 = self.select_features(np.array(trainX),np.array(trainY),cost)
                sel_feat_num3 = cost*np.shape(trainX)[1]
                select_trainX_temp3 = []

                for i in range(0,round(sel_feat_num3)):
                    select_trainX_temp3.append(np.array(trainX)[:,sel_feat_ind3[i]])
                    selected_trainX = np.transpose(select_trainX_temp3)               
                    selected_trainY = trainY
        print(np.shape(selected_trainX))
        print(np.shape(selected_trainY))       

        return selected_trainX, selected_trainY

    def select_testX(self, trainX, trainY, testX, cost):
            # Task 1-5
            if len(trainX[0]) >= 300:
                threshold = 0.8

                if cost >= threshold:
                    new_cost = round(ma.sqrt(cost),2)
                    sel_x_temp, selected_trainY = self.select_sample(trainX,trainY,new_cost)
                    sel_feat_ind = self.select_features(np.array(sel_x_temp),np.array(selected_trainY),new_cost)
                    sel_feat_num = new_cost*np.shape(trainX)[1]
                    select_testX_temp = []

                    for i in range(0,round(sel_feat_num)):
                        select_testX_temp.append(testX[:,sel_feat_ind[i]])
                        selected_testX = np.transpose(select_testX_temp)

                elif cost < threshold:
                    sel_feat_ind2 = self.select_features(np.array(trainX),np.array(trainY),cost)
                    sel_feat_num2 = cost*np.shape(trainX)[1]
                    select_testX_temp2 = []

                    for i in range(0,round(sel_feat_num2)):
                        select_testX_temp2.append(testX[:,sel_feat_ind2[i]])
                        selected_testX = np.transpose(select_testX_temp2)
                        selected_trainY = trainY

            elif len(trainX[0]) < 300:
                sel_feat_ind3 = self.select_features(np.array(trainX),np.array(trainY),cost)
                sel_feat_num3 = cost*np.shape(trainX)[1]
                select_testX_temp3 = []

                for i in range(0,round(sel_feat_num3)):
                    select_testX_temp3.append(testX[:,sel_feat_ind3[i]])
                    selected_testX = np.transpose(select_testX_temp3)

            return selected_testX 
    
    def train(self, trainX, trainY):
        nones = np.ones((np.shape(trainX)[0])) 
        mones = np.ones((np.shape(trainX)[1])) 

        t = cp.Variable((np.shape(trainX)[0])) 
        m = cp.Variable((np.shape(trainX)[1]))         
        theta = cp.Variable(np.shape(trainX)[1]) 
        b = cp.Variable(1) 

        objective = cp.Minimize((1/np.shape(trainX)[0]) * nones @ t + self.alpha * mones @ m)
        constraints = [trainY-(trainX @ theta + b) - t<= 0,
                       -trainY+(trainX @ theta + b) - t<= 0,
                       theta <= m,
                       -theta <= m]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=False)
        self.weight = theta.value
        self.bias = b.value

        print("the result of this LP is: \n", result, "\n")
        # print(self.weight)

        PredY = trainX @ self.weight + self.bias
        # print(np.shape(PredY))
        train_error = mean_absolute_error(trainY,PredY)
        return train_error
    
    def train_online(self, trainX, trainY, cost):
        
        x_set = []
        y_set = []
        x_tem =[]
        y_tem = []
        init_value_num = 10
        weight_ini = None
        bias_ini = None
        train_cost = 0 
        total_cost = 0
        sample_num = 0
        fea_level = np.linspace(0.01,1,10)
        threshold = 0.2

        # we simulate the online setting by handling training data samples one by one
        for index, x in enumerate(trainX):
            y = trainY[index]
            x_set.append(x)
            y_set.append(y)
            total_cost = total_cost + len(x)

            if index < init_value_num:
                x_tem.append(x)
                y_tem.append(y)
                train_cost = train_cost + len(x)

            elif index == init_value_num:
                self.train(np.array(x_tem),np.array(y_tem))
                train_error = []
                weight_ini = self.weight
                bias_ini = self.bias

                for f in fea_level:
                    self.weight = weight_ini
                    self.bias = bias_ini
                    feat_ind = self.select_features(np.array(x_tem),np.array(y_tem),f)
                    ind_num = f * np.shape(x_tem)[1]
                    select_x_tem = []

                    for i in range(0, round(ind_num)):
                        select_x_tem.append(np.array(x_tem)[:,feat_ind[i]])
                        select_x = np.transpose(select_x_tem)

                    err = self.train(np.array(select_x),np.array(y_tem))
                    train_error.append(err)
                    _,error = self.evaluate(np.array(select_x),np.array(y_tem))

                err_list = []
                err_list[:] = [x for x in train_error if x <= threshold * max(train_error)]
                max_err = max(err_list)
                best_pt = fea_level[(len(fea_level) - len(err_list) -1)]
                self.weight = weight_ini
                self.bias = bias_ini

                if best_pt <= cost:
                    feat_ind = self.select_features(np.array(x_tem),np.array(y_tem),cost)
                else:
                    feat_ind = self.select_features(np.array(x_tem),np.array(y_tem),best_pt)

                select_x = np.array(x_tem)[:,feat_ind]
                self.weight = weight_ini[feat_ind]
                self.bias = bias_ini
                self.feat_ind = feat_ind
                x_tem = np.array(select_x).tolist()

            else:
                add_set = x[self.feat_ind]
                predY = add_set @ self.weight + self.bias
                error = abs(predY - y)

                if (train_cost + len(add_set)) / total_cost <= cost and error >= 0.1 * max_err:
                    x_tem.append(add_set)
                    y_tem.append(y)
                    train_cost = train_cost + len(add_set)
                    sample_num = sample_num + 1

                if sample_num / (len(y_tem) - sample_num) >= 0.1:
                    train_error = self.train(np.array(x_tem),np.array(y_tem))
                    sample_num = 0
        
        self.training_cost = train_cost / total_cost
        _, train_error = self.evaluate(x_tem, y_tem)

        return self.training_cost, train_error

    def online_train_testX(self,testX,testY):
        testX_temp = testX[:,self.feat_ind]
        _,test_error = self.evaluate(testX_temp,testY)

        return test_error

    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        error = mean_absolute_error(Y, predY)
        # print("the test error is: \n", test_error, "\n")
        
        return predY, error
    
    
    def get_params(self):
        return self.weight, self.bias
    