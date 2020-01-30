#assignment_1

import numpy
import csv
import random
import pickle

def norm(f):
    return f/255

def activation(x):
    return 1 if x >= 0 else 0

v_activation = numpy.vectorize(activation)

data = []
test = []

#Hyperparameters. Change these to complete assignment. 
epochs = 50
#l_r = 0.001

try:
    data = pickle.load(open("train.pickle", "rb"))
except(OSError, IOError) as _:
    with open("mnist_train.csv", 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            l = int(row.pop(0))
            row = [float(x)/255 for x in row]
            row.insert(0, l)
            data.append(row) 
        pickle.dump(data, open("train.pickle", "wb"))
        print("no train.pickle")
print("data read")

#Same as the try/except block above.
try: 
    test = pickle.load(open("test.pickle", "rb"))
except(OSError, IOError) as _:
    with open("mnist_test.csv", 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            l = int(row.pop(0))
            row = [float(x)/255 for x in row]
            row.insert(0, l)
            test.append(row)
        pickle.dump(test, open("test.pickle", "wb"))
        print("no test.pickle")
print("test read")
        
data_labels = []
test_labels = []

random.shuffle(data)
print("data shuffled")

for row in data:
    data_labels.append(row[0])
    row[0] = 1
print("data labels fetched")
    
for row in test:
    test_labels.append(row[0])
    row[0] = 1
print("test labels fetched")

data = numpy.array(data)
print("data converted")
data_labels = numpy.array(data_labels)
print("data labels converted")
test = numpy.array(test)
print("test converted")
test_labels = numpy.array(test_labels)
print("test labels converted")

output_file = open("output.txt", "a")
for l_r in [0.001, 0.01, 0.1]:
    #now weights is a matrix with 10 cols and 785 rows.
    weights = (numpy.random.rand(10,785) - 0.5) * 0.05
    print("weights initialized")
    total_error_train = []
    total_error_test = []
    for _ in range(epochs):
        #there are three things to do:
        
        #zeroth, calculate error of entirely untrained perceptrons:
        correct_prediction_count = 0
        for i in range(len(data)):
            predictions = weights.dot(data[i])
            p = numpy.where(predictions == numpy.amax(predictions))
            #print("p = " + str(p) + "\nl = " + str(data_labels[i]))
            if p[0] == data_labels[i]:
                correct_prediction_count += 1
        total_error_train.append(correct_prediction_count / len(data))
        print(correct_prediction_count / len(data))
        
        correct_prediction_count = 0
        for i in range(len(test)):
            predictions = weights.dot(test[i])
            p = numpy.where(predictions == numpy.amax(predictions))
            if p[0] == test_labels[i]:
                correct_prediction_count += 1
        total_error_test.append(correct_prediction_count / len(test))
        print(correct_prediction_count / len(test))
        print()
            
        #first, run training (updating weights)    
        for i in range(len(data)):
            output = v_activation(weights.dot(data[i])).reshape(10,1)
            #output = numpy.array([activation(x) for x in output])
    
            t = numpy.zeros((10,1))
            t[data_labels[i]] = 1
            error = t - output
            
            weights = weights + l_r * error * numpy.tile(data[i], (10,1))
    
    correct_prediction_count = 0
    for i in range(len(data)):
        predictions = weights.dot(data[i])
        p = numpy.where(predictions == numpy.amax(predictions))
        #print("p = " + str(p) + "\nl = " + str(data_labels[i]))
        if p[0] == data_labels[i]:
            correct_prediction_count += 1
    total_error_train.append(correct_prediction_count / len(data))
    print(total_error_train)
    output_file.write("Learning rate: " + str(l_r) + "\nErrors on training data: ")
    output_file.write(numpy.array2string(numpy.array(total_error_train), threshold=60) + "\n")
    
    correct_prediction_count = 0
    for i in range(len(test)):
        predictions = weights.dot(test[i])
        p = numpy.where(predictions == numpy.amax(predictions))
        if p[0] == test_labels[i]:
            correct_prediction_count += 1
    total_error_test.append(correct_prediction_count / len(test))
    print(total_error_test)
    output_file.write("Errors on test data: " + numpy.array2string(numpy.array(total_error_test), threshold=60))

output_file.close()    