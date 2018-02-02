import scipy.io
import numpy as np

from scipy import stats

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


def majorityValue(binary_targets):
    p = count_values(binary_targets, 1)
    n = count_values(binary_targets, 0)

    if p > n:
        return 1
    else:
        return 0

def bestAttribute(examples,attribute_matrix, binary_targets):
    max_info_gain = 0
    max_attr = 0
    for index_attr in range(attribute_matrix.shape[1]):
        info_gain = gain(examples,attribute_matrix,index_attr, binary_targets)

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_attr = index_attr
    return max_attr

def gain(examples, attribute_matrix, attribute_no, binary_targets):
    """Calculates the gain for attribute_no"""
    #p = count_values(examples, 1)
    #n = count_values(examples, 0)
    p = count_values(binary_targets, 1)
    n = count_values(binary_targets, 0)
    p0 = count_attribute_values(examples, 1, attribute_matrix, attribute_no, 0)
    n0 = count_attribute_values(examples, 0, attribute_matrix, attribute_no, 0)
    p1 = count_attribute_values(examples, 1, attribute_matrix, attribute_no, 1)
    n1 = count_attribute_values(examples, 0, attribute_matrix, attribute_no, 1)
    return funcI(p, n) - remainder(p, n, p0, n0, p1, n1)


def count_values(attribute, value):
    """returns the number of elements equal to value in array 'attribute'"""
    count = 0
    for i in attribute:
        if i == value:
            count += 1
    return count


def count_attribute_values(examples, e_value, attribute, col, a_value):
    """returns the number of elements in array 'example' with value 'e_value' where the corresponding value in column 'col' of array 'attribute' is equal to 'a_value'"""
    count = 0
    i = 0
    while i < examples.size:
        if examples[i] == e_value and np.hsplit(attribute, 45)[col][i] == a_value:
            count += 1
        i += 1
    return count


def funcI(p, n):
    total = p + n
    a = float(p) / total
    b = float(n) / total
    loga = np.log2(a)
    logb = np.log2(b)
    return (-1 * a * loga) - (b * logb)


def remainder(p, n, p0, n0, p1, n1):
    total = p + n
    a = float(p0 + n0) / total
    b = float(p1 + n1) / total
    i0 = funcI(p0, n0)
    i1 = funcI(p1, n1)
    return (a * i0) + (b * i1)

# def split(examples, best_attribute, value):
#     count = 0
#     a = []
#     for i in range(len(examples)):
#         if examples[best_attribute][i] == value:
#             a[count] = examples



#not done, need to split examples by values
#changed bestAttribute, gain, tree class, and decisionTree
def decisionTree(examples, attributes, binary_targets):
    x = tree()

    if len(set(binary_targets)) == 1:
        x.addLeaf(binary_targets[0])
        return x
    else if len(attributes) == 0:
        x.addLeaf(majorityValue(binary_targets))
        return x
    else:
        best_attribute = bestAttribute(examples, attributes, binary targets)
        x.addRoot(best_attribute)
        for i in range(2):
            y = tree()
            x.addKids(y)
            subset_examples = split(examples, best_attribute, i)
            if len(subset_examples) == 0:
            y.addLeaf(majorityValue(subset_binary))

            else y.addKids(decisionTree(subset_examples, attribute without best, subset_binary))
    return x

class tree:
        # self.op = attribute #attribute number for root
        # kids = [] #subtreees
        # leaf = None #value is 1 or 0 if leaf node, otherwise None
        #if put here, these are defined as class attributes and not as instance attributes
    def __init__(self):
        self.op = None #attribute number for root
        self.kids = [] #subtreees
        self.leaf = None #value is 1 or 0 if leaf node, otherwise None
    def addRoot(self, attribute):
        #add root node to tree, empty for leaf node
        self.op = attribute
    def addKids(self, kid):
        """add subtrees to the tree"""
        self.kids.append(kid)
    def addLeaf(value):
        self.leaf = value


attributes = mat['x']

examples = mat['y']
examples1 = loadExamples(examples, 1) # +ve and -ve examples for emotion 1
examples2 = loadExamples(examples, 2) # +ve and -ve examples for emotion 2
examples3 = loadExamples(examples, 3) # +ve and -ve examples for emotion 3
examples4 = loadExamples(examples, 4) # +ve and -ve examples for emotion 4
examples5 = loadExamples(examples, 5) # +ve and -ve examples for emotion 5
examples6 = loadExamples(examples, 6) # +ve and -ve examples for emotion 6

p = count_values(examples1, 1)
n = count_values(examples1, 0)
print(funcI(p, n))

print(count_values(examples1, 1))
print(count_values(examples1, 0))

print(gain(examples1, attributes, 1))

print(majorityValue(examples))
