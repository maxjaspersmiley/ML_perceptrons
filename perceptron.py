import numpy
import csv
import random

'''
x is our input vector (training data?)
W is the weight vector (sequence of integers)
z stands for the pre-activation (dot product of W and the input vector x)
a is boolean (0 or 1), deciding whether or not perceptron was activated
e represents the error in our prediction
'''

'''
I absolutely do not claim to be anything approaching a decent python programmer.
A lot of this (especially for loops) is written like c++ and fails to make use
of python's powerful Iterator support. I'll refactor once I get it working correctly.
'''

#test code from pythonmachinelearning.pro
#DO NOT USE in assignment submission.
class Perceptron(object):
    def __init__(self, input_size, learn_rate = 1, epochs = 50):
        self.W = numpy.zeros(input_size+1) #using input_size+1 
                            # to include bias in weight vector
        self.epochs = epochs
        self.learn_rate = learn_rate
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    #runs input through a perceptron and returns an output
    def predict(self, x):
        # insert a 1 at the head of sequence x, to 
        # correspond to the first element of W (bias)
        x = numpy.insert(x, 0, 1)
        z = self.W.T.dot(x) #WTF is T? Pattern matching?
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for _ in range (self.epochs):
            for i in range (d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y  #compute error
                self.W = self.W + self.learn_rate * e * numpy.insert(X[i], 0, 1)

if __name__ == '__main__':
    data = []
    numlines = 0

    #read the csv file. Each element of data is a line from the csv.
    with open("mnist_train.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
        numlines = csvreader.line_num
        print("Number of rows: %d"%numlines)

    #convert from list(list(string)) to list(list(int))
    for i in range(len(data)):
        data[i] = [int(x) for x in data[i]]

    #randomize the list
    random.shuffle(data)

    #pop first element of each list into a separate labels list.
    labels = []
    for i in range(len(data)):
        labels.append(data[i].pop(0))

    #This is wrong. Need to make 10 perceptrons.
    #something like 
    #   perceptron = [Perceptron(input_size = 784) for x in range(10)]
    #Working out how to feed training data into the correct perceptron at the moment. 
    perceptron = Perceptron(input_size=784)
    #train
    perceptron.fit(numpy.array(data), numpy.array(labels))
    
    print(perceptron.W)

    #duplicate of code above, but this time grabbing 
    #test data rather than training data.
    testdata = []
    with open("mnist_test.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            testdata.append(row)
        numlines = csvreader.line_num
        print("Number of rows: %d"%numlines)

    for i in range(len(testdata)):
        testdata[i] = [int(x) for x in testdata[i]]

    testlabels = []
    for i in range(len(testdata)):
        testlabels.append(testdata[i].pop(0))
    
    #this is wrong - just returning the activation function. (always 1)
    #need to have multiple perceptrons (10, one for each digit)
    #not sure how to code this...
    """
    for i in range(len(testdata)):
        z = perceptron.predict(testdata[i])
        print("Prediction: " + str(z) + "Actual: " + str(testlabels[i]))
    """