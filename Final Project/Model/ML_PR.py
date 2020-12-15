# -*- coding: utf-8 -*-
"""
@uthor: Sadegh Sadeghi Tabas
Department of Civil Engineering
Clemson University
Email: sadeghs@clemson.edu

CPSC8420- Advanced Machine Learning
Final Project: Assessment of Several Regression Based Machine Learning 
Approaches in Runoff Prediction

"""
# In[-]: import libraries
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from numpy import savetxt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# In[-]: Importing the dataset
datasets = pd.read_csv('dataset.csv')
x_train = datasets.iloc[0:8971, 2:8].values.astype(float)
y_train = datasets.iloc[0:8971, 11:12].values.astype(float)
x_test= datasets.iloc[8971:9471, 2:8].values.astype(float)
y_test= datasets.iloc[8971:9471, 11:12].values.astype(float)


# In[-]: Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x_train = StandardScaler()
sc_y_train = StandardScaler()
x_train = sc_x_train.fit_transform(x_train)
y_train = sc_y_train.fit_transform(y_train).ravel()
x_test=sc_x_train.transform(x_test)
observation = y_test

# In[-]: Parameter Tuning

# hiddensize = [int(x) for x in np.linspace(10, 300, num = 30)]
# from sklearn.model_selection import GridSearchCV
# # param_grid = {'hidden_layer_sizes':hiddensize, activation: 'relu,' [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
# param_grid={
# 'learning_rate': ["constant", "invscaling", "adaptive"],
# 'hidden_layer_sizes':hiddensize ,
# 'activation': ["logistic", "relu", "Tanh"],
# 'solver': ["sgd", "lbfgs", "adam"]
# }
# MLP=MLPRegressor(max_iter=10000)
# grid = GridSearchCV(MLP,param_grid,scoring = 'r2',n_jobs=2,refit=True,cv=3,verbose=2,return_train_score=True)
# grid.fit(X,y)
# print(grid.best_estimator_)
# print(grid.best_score_)

# In[1]: MLP

MLP = MLPRegressor(activation='relu', hidden_layer_sizes=(20), learning_rate='adaptive',
              max_iter=10000, random_state=42, solver='sgd')
MLP.fit(x_train, y_train)
#print(f"MLP loss: { MLP.loss_:.2f}")
MLP_Pred = MLP.predict(x_test)
MLP_Sim = sc_y_train.inverse_transform(MLP_Pred)


# In[2]: Linear Regression

LR = linear_model.LinearRegression()
LR.fit(x_train, y_train)
LR_Pred = LR.predict(x_test)
LR_Sim = sc_y_train.inverse_transform(LR_Pred)


# In[3]: Ridge Regression

RR = linear_model.Ridge(alpha=.5)
RR.fit(x_train, y_train)
RR_Pred = RR.predict(x_test)
RR_Sim = sc_y_train.inverse_transform(RR_Pred)

# In[4]: Lasso Regression

Lasso = linear_model.Lasso(alpha=0.1)
Lasso.fit(x_train, y_train)
Lasso_Pred = Lasso.predict(x_test)
Lasso_Sim = sc_y_train.inverse_transform(Lasso_Pred)

# In[5]: Support Vector Machine

SVM = svm.SVR()
SVM.fit(x_train, y_train)
SVM_Pred = SVM.predict(x_test)
SVM_Sim = sc_y_train.inverse_transform(SVM_Pred)

# In[]: Calculation of Different Performance Criteria

def kge(obs, sim):
    obs=array(obs).reshape(500,)
    obs_filtered = np.asarray(obs)
    sim_filtered = np.asarray(sim)
    sim_std = np.std(sim_filtered, ddof=1)
    obs_std = np.std(obs_filtered, ddof=1)
    sim_mu = np.mean(sim_filtered)
    obs_mu = np.mean(obs_filtered)
    r = np.corrcoef(sim_filtered, obs_filtered)[0, 1]
    var = sim_std / obs_std
    bias = sim_mu / obs_mu
    kge = 1 - np.sqrt((bias-1)**2 + (var-1)**2 + (r-1)**2)
    return kge


print('Nash–Sutcliffe model efficiency coefficient:')
print(f"MLP: {r2_score(observation, MLP_Sim):.2f}" )
print(f"Linear Reg: {r2_score(observation, LR_Sim):.2f}" )
print(f"Ridge Reg: {r2_score(observation, RR_Sim):.2f}" )
print(f"Lasso Reg: {r2_score(observation, Lasso_Sim):.2f}" )
print(f"SVM: {r2_score(observation, SVM_Sim):.2f}" )
print('======================================================\n')

print('Root Mean Squared Error model efficiency coefficient:')
print(f"MLP: {math.sqrt(mean_squared_error(observation, MLP_Sim)):.2f}" )
print(f"Linear Reg: {math.sqrt(mean_squared_error(observation, LR_Sim)):.2f}" )
print(f"Ridge Reg: {math.sqrt(mean_squared_error(observation, RR_Sim)):.2f}" )
print(f"Lasso Reg: {math.sqrt(mean_squared_error(observation, Lasso_Sim)):.2f}" )
print(f"SVM: {math.sqrt(mean_squared_error(observation, SVM_Sim)):.2f}" )
print('======================================================\n')

print('Kling-Gupta model efficiency coefficient:')
print(f"MLP: {kge(observation, MLP_Sim):.2f}" )
print(f"Linear Reg: {kge(observation, LR_Sim):.2f}" )
print(f"Ridge Reg: {kge(observation, RR_Sim):.2f}" )
print(f"Lasso Reg: {kge(observation, Lasso_Sim):.2f}" )
print(f"SVM: {kge(observation, SVM_Sim):.2f}" )

# In[]: Plot and Save Outputs
# plt.plot(simulation, y_test, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

# np.savetxt('observetrain.csv',observation , delimiter=',', fmt='%f')
# np.savetxt('modeltrain.csv', simulation, delimiter=',', fmt='%f')

# Plot results


fig, ax = plt.subplots(figsize=(12, 4))
ax.plot( observation, label="Observation")
ax.plot( MLP_Sim, label="MLP")
ax.plot( LR_Sim, label="Linear Reg")
ax.plot( RR_Sim, label="Ridge Reg")
ax.plot( Lasso_Sim, label="Lasso Reg")
ax.plot( SVM_Sim, label="SVM")

ax.legend()
plt.xlabel('time (day)')
plt.ylabel('Runoff (ft3/s)')

fig.savefig('Simulation.eps', format='eps')

# output=np.concatenate([MLP_Sim,LR_Sim,RR_Sim,Lasso_Sim,RF_Sim,Bayes_Sim,SVM_Sim,neigh_Sim,GBR_Sim,clf_Sim,VR_Sim, ELNet_Sim, BL_Sim])
# #output=np.concatenate((observation,output))
# savetxt('RegressionModelsOutputs.csv', output, delimiter=',')