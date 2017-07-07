import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
from keras.utils import plot_model

# Hyperparameters
input_nodes = 7

df = pd.read_csv('lebron_stats.csv')

data = df.loc[:,['PTS','TRB','AST','STL','BLK','TOV', 'PF']]
labels = df.loc[:,['Outcome']]

data = data.as_matrix()
labels = labels.as_matrix()

def one_hot_vector_converter(inputMat, labels):
	# Change output column to a one-hot vector
	one_hot_Mat = np.zeros((len(inputMat), labels))
	for i in range(len(inputMat)):
		one_hot_Mat[i,inputMat[i]] = 1
	return one_hot_Mat

labels = one_hot_vector_converter(labels, 2)

model = Sequential()

model.add(Dense(14, activation = 'sigmoid', input_shape = (input_nodes,)))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(data, labels, epochs = 100, batch_size = 2)

# Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

# Plot the simplified keras graph of nn
plot_model(model, to_file='model.png')
