from asyncio import selector_events
import MyRegressor 
import numpy as np
import utils as ut

# test & train data 
# data = ut.prepare_data_gaussian()
data = ut.prepare_data_news()

trainX = data["trainX"]
trainY = data["trainY"]
testX = data["testX"]
testY = data["testY"]

print(len(trainX[1]))
# define alpha value 
alpha = [0,0.0001,0.001,0.01,0.05,0.1,0.2,1,10,100]

# 1-2 linear regressor 
def task1_2(alpha, trainX, trainY, testX, testY):
    train_error = []
    test_error = []

    for i in range(0,len(alpha)):
        regressor_init = MyRegressor.MyRegressor(alpha[i])
        train_error.append(regressor_init.train(trainX, trainY))
        _,test_error_temp = regressor_init.evaluate(testX, testY)
        test_error.append(test_error_temp)
        result_1_2 = {'taskID':'1-2', 'alpha':alpha, 'train_err':train_error, 'test_err':test_error}

    error_diff = np.abs(np.array(train_error)-np.array(test_error))
    best_alpha = alpha[np.argmin(error_diff)]
    ut.plot_result(result_1_2)
    return best_alpha

# 1-3 feature Selection 
def task1_3(alpha, trainX, trainY, testX, testY):

    regressor_init = MyRegressor.MyRegressor(alpha[0])
    level = np.array([0.01,0.1,0.2,0.3,0.5,0.8,1])
    result_1_3 = {'taskID':'1-3', 'feat_num':[], 'train_err':[], 'test_err':[]}

    for j in range(0,len(level)):
        result_1_3['feat_num'].append(level[j])
        select_feat_ind = regressor_init.select_features(trainX,trainY,level[j])
        sel_feat_num = level[j]*np.shape(trainX)[1]
        select_trainX_temp = []
        select_testX_temp = []

        for i in range(0,round(sel_feat_num)):
            select_trainX_temp.append(trainX[:,select_feat_ind[i]])
            select_trainX = np.transpose(select_trainX_temp)
            select_testX_temp.append(testX[:,select_feat_ind[i]])
            select_testX = np.transpose(select_testX_temp)    
        
        train_error2 = regressor_init.train(select_trainX, trainY)
        _,test_error2 = regressor_init.evaluate(select_testX, testY)
        result_1_3['train_err'].append(train_error2)
        result_1_3['test_err'].append(test_error2)
    
    ut.plot_result(result_1_3)

# 1-4 Sample Selection 
def task1_4(alpha, trainX, trainY, testX, testY):
    regressor_init = MyRegressor.MyRegressor(alpha[0])
    regressor_init.train(trainX, trainY)
    percent = [0.01,0.1,0.3,0.5,0.8,0.9,1]
    train_error3 = []
    test_error3 = []

    for i in range(0,len(percent)):
        trainset = regressor_init.select_sample(trainX,trainY,percent[i])
        sel_trainX = np.array(trainset[0])
        sel_trainY = np.array(trainset[1])
        train_error3.append(regressor_init.train(sel_trainX, sel_trainY))
        _,test_error3_temp = regressor_init.evaluate(testX, testY)
        test_error3.append(test_error3_temp)    
        result_1_4 = {'taskID':'1-4', 'sample_num':percent, 'train_err':train_error3, 'test_err':test_error3}

    ut.plot_result(result_1_4)

# 1-5 Sample and Feature Selection
def task1_5(alpha, trainX, trainY, testX, testY):
    result_1_5 = {'taskID':'1-5', 'cost':[], 'train_err':[], 'test_err':[]}
    regressor_init = MyRegressor.MyRegressor(alpha[0])
    cost = [0.01,0.1,0.3,0.5,0.8,0.9,1]
    train_error4 = []
    test_error4 = []

    for i in range(0,len(cost)):
        regressor_init.train(trainX, trainY)
        trainset2 = regressor_init.select_data(trainX,trainY,cost[i])
        sel_testX = regressor_init.select_testX(trainX,trainY,testX,cost[i])
        sel_data_trainX = np.array(trainset2[0])
        sel_data_trainY = np.array(trainset2[1])
        train_error4.append(regressor_init.train(sel_data_trainX, sel_data_trainY))
        _,test_error4_temp = regressor_init.evaluate(sel_testX, testY)
        test_error4.append(test_error4_temp)

        result_1_5 = {'taskID':'1-5', 'cost':cost, 'train_err':train_error4, 'test_err':test_error4}

    ut.plot_result(result_1_5)

# Task 2 
def task2(alpha, trainX, trainY, testX, testY):
    regressor_init = MyRegressor.MyRegressor(alpha[0])
    percentage = [0.01,0.1,0.3,0.5,0.8,0.9,1]
    train_error5 = []
    test_error5 = []
    training_cost = []

    for i in range(0,len(percentage)):
        trainset3 = regressor_init.train_online(trainX, trainY, percentage[i])
        test_error = regressor_init.online_train_testX(testX, testY)
        training_cost.append(trainset3[0])
        train_error5.append(trainset3[1])
        test_error5.append(test_error)
        
    result2 = {'taskID':'2', 'cost':training_cost, 'train_err':train_error5, 'test_err':test_error5}
    ut.plot_result(result2)

# Main Functions
# Uncomment each command to run the file 

# 1-2
# sel_alpha = task1_2(alpha, trainX, trainY, testX, testY)

# optimized a = 0.05 for gasussian set
# optimized a = 0 for news set

# 1-3
# task1_3(alpha, trainX, trainY, testX, testY)

# 1-4
# task1_4(alpha, trainX, trainY, testX, testY)

# 1-5
# task1_5(alpha, trainX, trainY, testX, testY)

# 2-1
task2(alpha, trainX, trainY, testX, testY)






