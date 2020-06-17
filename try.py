import numpy as np
import pandas as pd

import csv
X=[]
csv_dict=[]

file = open(r'C:\\Users\ADISHRI\\Desktop\\ecoli_data.csv')
data = csv.reader(file)

for column in data:
    csv_dict.append(column)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#for i in csv_dict:
 #   X = np.array(i)
  #  training_inputs = np.array[[X[0], X[1], X[2]]]  # how to take these inputs individually?
    # print(X[0],X[1],X[2])
    # print(X[3])
   # training_outputs = np.array(X[4])

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights :")
print(synaptic_weights)

for iteration in range(2000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = outputs - training_outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training :")
print(synaptic_weights)
print("Outputs after training :")
print(outputs)
