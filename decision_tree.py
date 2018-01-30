import scipy.io
import numpy as np

mat = scipy.io.loadmat('Data/cleandata_students.mat')

def loadExamples(input_array, emotion_number):
    """reads array of emotion values 1-6 and returns a binary array. Values in the returned array are 1 when the corresponding value in the input array match the value of emotion_number"""
    output_array = np.zeros(input_array.shape, dtype=np.int16)
    i = 0
    while i < input_array.size:
        if input_array[i] == emotion_number:
            output_array[i] = 1
        i += 1
    return output_array


def majorityValue():
    return 1

def bestAttribute():
    return 1

def decisionTree():
    return 1

#class tree:

examples = mat['y']
examples1 = loadExamples(examples, 1) # +ve and -ve examples for emotion 1
examples2 = loadExamples(examples, 2) # +ve and -ve examples for emotion 2
examples3 = loadExamples(examples, 3) # +ve and -ve examples for emotion 3
examples4 = loadExamples(examples, 4) # +ve and -ve examples for emotion 4
examples5 = loadExamples(examples, 5) # +ve and -ve examples for emotion 5
examples6 = loadExamples(examples, 6) # +ve and -ve examples for emotion 6
