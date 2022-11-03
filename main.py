from asyncio import selector_events
import MyRegressor 
import numpy as np
import utils as ut

# test & train data 
trainX = ut.prepare_data_gaussian()["trainX"]
trainY = ut.prepare_data_gaussian()["trainY"]
testX = ut.prepare_data_gaussian()["testX"]
testY = ut.prepare_data_gaussian()["testY"]

# test
# print("train data X is \n", trainX, "\n")
# print("train data Y is \n", trainY, "\n")
# print("test data X is \n", testX, "\n")
# print("test data Y is \n", testY, "\n")

trainN = np.shape(trainX)[0]
# trainM = np.shape(trainX)[1]
print(np.shape(trainX))

# define alpha value 
alpha = np.array([0,0.01,0.05,0.1,0.5,1])

# 1-2 linear regressor 
regressor_init = MyRegressor.MyRegressor(alpha[1],trainN)

# train_error1 = regressor_init.train(trainX, trainY)
# test_error1 = regressor_init.evaluate(testX, testY)
# print("the train error1 is: \n", train_error1, "\n")
# print("the test error1 is: \n", test_error1, "\n")
# # plot1 = ut.plot_result({'taskID':'1-2', 'alpha':[alpha[0]], 'train_err':[train_error1], 'test_err':[test_error1]})


# # 1-3 feeture Selection 
# level = np.array([0.01,0.1,0.3,0.5,1])
# select_feat_ind = regressor_init.select_features(trainX,trainY,level[2])
# sel_feat_num = level[2]*np.shape(trainX)[1]

# select_trainX_temp = []
# select_testX_temp = []
# for i in range(0,int(sel_feat_num)):
#     select_trainX_temp.append(trainX[:,select_feat_ind[i]])
#     select_trainX = np.transpose(select_trainX_temp)
#     select_testX_temp.append(testX[:,select_feat_ind[i]])
#     select_testX = np.transpose(select_testX_temp)    
# # print(np.shape(select_trainX))
# # print(np.shape(select_testX))

# train_error2 = regressor_init.train(select_trainX, trainY)
# test_error2 = regressor_init.evaluate(select_testX, testY)
# # print("the train error2 is: \n", train_error2, "\n")
# # print("the test error2 is: \n", test_error2, "\n")
# # plot2 = ut.plot_result({'taskID':'1-3', 'feat_num':[0.3], 'train_err':[train_error2], 'test_err':[test_error2]})


# 1-4 Sample Selection
percent = np.array([0.01,0.1,0.3,0.5,1])
trainset = regressor_init.select_sample(trainX,trainY,percent[1])
sel_trainX = np.array(trainset[0])
sel_trainY = np.array(trainset[1])


redN = round(percent[1]*len(trainY))
print(redN)
regressor_redN = MyRegressor.MyRegressor(alpha[4],redN)
print(np.shape(sel_trainX))
print(np.shape(sel_trainY))
train_error3 = regressor_redN.train(sel_trainX, sel_trainY)
# test_error3 = regressor_redN.evaluate(select_testX, testY)






