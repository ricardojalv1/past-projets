from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error as MSE, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 
import sklearn
#import matplotlib as mpl

df = pd.read_csv(r'C:\Users\Ricardo\Documents\Circular AgTech\realistic_test_data.csv')
X = df.iloc[:, 1:]
y = df['yield']
X_nitrate = df['NH3']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=3)
X_train1, X_test1, y_train1, y_test1= train_test_split(X_nitrate, y, test_size=0.3, random_state=3)
X_train1 = (np.array(X_train1.values.tolist())).reshape(-1,1)
X_test1 = (np.array(X_test1.values.tolist())).reshape(-1,1)

dt = DecisionTreeRegressor(min_samples_leaf=0.2, random_state=3)
dt1 = DecisionTreeRegressor(min_samples_leaf=0.2, random_state=3)
dt = dt.fit(X_train, y_train)
dt_baseline = dt1.fit(X_train1, y_train1)

y_pred_test = dt.predict(X_test)
y_pred_train = dt.predict(X_train)
y_pred_test_baseline = dt_baseline.predict(X_test1)
#y_pred_train_baseline = dt_baseline.predict(X_train1)


mse_dt = MSE(y_test, y_pred_test)
rmse_dt = mse_dt ** (0.5)
mse_cv = - cross_val_score(dt, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error', n_jobs = -1)
rmse_baseline = (MSE(y_test1, y_pred_test_baseline))**(0.5)

print('RMSE test: ' + str(rmse_dt))
print('CV RMSE: {:.2f}'.format((mse_cv.mean())**(0.5)))
print('Train RMSE: {:.2f}'.format((MSE(y_train, y_pred_train))**(0.5)))
print('Test RMSE: {:.2f}'.format((MSE(y_test, y_pred_test))**(0.5)))
print('Baseline RMSE: {:.2f}'.format(rmse_baseline))



dot_data = sklearn.tree.export_graphviz(dt, out_file=None, filled=True, rounded=True, special_characters=True, feature_names = ['EC', 'pH', 'NH3','NO3', 'K'], proportion=True)
graph = graphviz.Source(dot_data)
graph.view()  


#fig, ax = plt.subplots(figsize=(10, 10))  
#plot_tree(dt, ax=ax, )
#plt.show()




