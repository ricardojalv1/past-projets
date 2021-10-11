

###@author: Ricardo Alvarez

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
tf.random.set_seed(7)

# Lowest RMSE: 1.1880 (4 x 5 nodes)
# Lowest RMSE without signs of overfitting: 

df = pd.read_csv(r'C:\Users\Ricardo\Documents\Circular AgTech\realistic_test_data.csv')
df2 = pd.read_csv(r'C:\Users\Ricardo\Documents\Circular AgTech\realistic_test_data2.csv')
test_df = pd.read_csv(r'C:\Users\Ricardo\Documents\Circular AgTech\val_data.csv')


inputs_list = df['yield']
yield1 = np.array(inputs_list, np.float32) 

features_list = df.iloc[:, 1:]
features1 = np.array(features_list, np.float32)




X = df.drop(columns=['yield'])
Y = df[['yield']]
X2 = df2.drop(columns=['yield'])
Y2 = df2[['yield']]

test_features = test_df.drop(columns=['yield'])
test_yield = test_df['yield']



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(5, activation='relu', input_shape=(5,)))
#model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
#model.add(tf.keras.layers.Dense(5, activation='relu'))

model.add(tf.keras.layers.Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
#model.summary()
model_hist = model.fit(X2,Y2, validation_split = 0.13, epochs= 10, callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)], verbose=False)



"""
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(5, activation='relu', input_shape=(5,)))
model2.add(tf.keras.layers.Dense(20, activation='relu'))
model2.add(tf.keras.layers.Dense(20, activation='relu'))
model2.add(tf.keras.layers.Dense(1))
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model2_hist = model.fit(X2,Y2, validation_split = 0.3, epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)], verbose=False)


predictions2 = model2.predict(test_features)

print("Predictions 2: ")
print(predictions2)
print(np.sqrt(mse(test_df['yield'], predictions2)))


model2_train = model2.evaluate(X2, Y2)

# Evaluate the large model using the test data
model2_test = model2.evaluate(test_features, test_yield)

print('model2 - Train: {}, Test: {}'.format(model2_train, model2_test))

"""

#One value prediction

columns= ["EC", "pH", "NH3", "N03", "K"]
single_record = [1.89, 7.2, 0.26, 77, 159.2]

single_df = pd.DataFrame(columns = columns)
single_df.loc[0] = single_record

single_prediction = model.predict(single_df)
print("Single prediction :")
print(single_prediction)



predictions = model.predict(test_features)
print("Predictions 1 :")
print(predictions)
print("Test data RMSE: ", np.sqrt(mse(test_df['yield'], predictions)))



plt.plot(model_hist.history['val_loss'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# Evaluate the small model using the train data
model1_train = model.evaluate(X, Y)

# Evaluate the small model using the test data
model1_test = model.evaluate(test_features, test_yield)


# Print losses
print('\n Model1 - Train: {}, Test (Val_loss): {}'.format(round(model1_train[0],4), round(model1_test[0], 4)))







