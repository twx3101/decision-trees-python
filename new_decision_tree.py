import scipy.io
import numpy as np
#np.set_printoptions(threshold=np.inf)
import math

INDEX_LAST_ELEMENT = -1


def chooseEmotion(example, emotion):
    result_array =  []
    for element in example:
        if element == emotion:
            result_array.append([1])
        else:
            result_array.append([0])
    return np.asarray(result_array)

def count(column, target):
    count = 0
    if column.size == 0:
        return 0
    for element in range(column.size):
        if column[element] == target:
            count += 1
    return count

# combine 2 arrays together by attaching input array with emotion example
def merge(all_data, example):
    i = 0
    complete_array = []
    for row in all_data:
        row = np.append(row,example[i])
        i += 1
        complete_array.append(row)
    return np.asarray(complete_array)

def entropy(binary_targets):
    num_one = count(binary_targets,1)
    num_zero = count(binary_targets,0)
    summation = num_one + num_zero
    left_side = num_one/summation
    right_side = num_zero/summation

    if left_side == 0:
        return -((right_side)*math.log(right_side,2))
    elif right_side ==0 :
        return (-1*(left_side)*math.log(left_side,2))
    else:
        return (-1*(left_side)*math.log(left_side,2))-((right_side)*math.log(right_side,2))

def gain(data_merge,col_attribute,binary_targets):
    return entropy(binary_targets)-remainder(data_merge,col_attribute,binary_targets)

def remainder(data_merge,col_attribute,binary_targets):
    left_array = []
    right_array = []
    binary_left = []
    binary_right = []

    for index, row in enumerate(data_merge):
        if row[col_attribute] == 1:
            left_array.append(row)
            binary_left.append(binary_targets[index])
        else:
            right_array.append(row)
            binary_right.append(binary_targets[index])

    left_array = np.asarray(left_array)
    right_array = np.asarray(right_array)
    binary_left = np.asarray(binary_left)
    binary_right = np.asarray(binary_right)
    if left_array.size == 0:
        left_entropy = 0
        p0_left = 0
        n0_left = 0
    else:
        left_entropy = entropy(binary_left)

    p0_n0 = left_array.size

    if right_array.size == 0:
        right_entropy = 0
        p1_right = 0
        n1_right = 0
    else:
        right_entropy = entropy(binary_right)

    p1_n1 = right_array.size

    number_parent = p0_n0 + p1_n1

    return  (p0_n0/number_parent)*left_entropy + (p1_n1/number_parent)*right_entropy

def chooseBestAttr(data_merge,attr_header,binary_targets):
    col_best_attr = 0
    max_gain = 0

    if data_merge.size == 0:
        return

    for col_attr in range(data_merge.shape[1]):
        if col_attr not in attr_header:
            continue
        attr_gain = gain(data_merge,col_attr,binary_targets)
        print('col attr1: ',col_attr, " : values1 = ",attr_gain)

        if attr_gain > max_gain:
            max_gain = attr_gain
            col_best_attr = col_attr

    return col_best_attr


def majorityValue(binary_targets):
    p = count(binary_targets, 1)
    n = count(binary_targets, 0)

    if p > n:
        return 1
    else:
        return 0

# check whether emotion value in emotion example has ALL same value or not
def hasSameValueEmotion(data_merge):
    first_item = data_merge[0]
    for row in data_merge:
        if row != first_item:
            return False
    return True

# get element in attr column
def getType(data_merge,col_best_attr):
    array_element = []
    for row in data_merge:
        if row[col_best_attr] not in array_element:
            array_element.append(row[col_best_attr])
    return array_element


def getDataSample(data_merge, col_best_attr, binary_targets, val):
    array_row = []
    binary_row = []
    for index, row in enumerate(data_merge):
        if row[col_best_attr] == val:
            array_row.append(row)
            binary_row.append(binary_targets[index])
    return np.asarray(array_row), np.asarray(binary_row)

#check whether sample has same value or not
def isSameSample(data_merge):
    temp_data_merge = np.copy(data_merge[0])
    i = 0
    for row in data_merge:
        if i == 0:
            i += 1
            continue
        if not np.array_equal(temp_data_merge,row):
            return False
    return True

def decisionTree(examples, attributes, binary_targets):
    x = tree()
    if isSameSample(examples):
        x.addLeaf(majorityValue(binary_targets))
        return x
    if hasSameValueEmotion(binary_targets):
        x.addLeaf(binary_targets[0])
        return x
    elif len(attributes) == 0:
        value = (majorityValue(binary_targets))
        x.addLeaf(value)
        return x
    else:
        best_attribute = chooseBestAttr(examples, attributes, binary_targets)
        x.addRoot(best_attribute)
        for i in range(2):

            subset_examples, subset_binary = getDataSample(examples, best_attribute, binary_targets, i)
            print(subset_examples)

            if len(subset_examples) == 0:
                y = tree()
                y.addLeaf(majorityValue(subset_binary))
                x.addKids(y)
            else:
                subset_attribute = attributes[:]
                subset_attribute.remove(best_attribute)
                x.addKids(decisionTree(subset_examples, subset_attribute, subset_binary))
    return x

class tree:
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

    def addLeaf(self, value):
        self.leaf = value

    def printtree(self):
        if(self.op is not None):
            print("Root", self.op)
        if(len(self.kids)) != 0:
            for i in range(len(self.kids)):
                self.kids[i].printtree()
        if(self.leaf is not None):
            print("leaf" , self.leaf)

data = scipy.io.loadmat("Data/cleandata_students.mat")

array_data = np.array(data)
data_x = np.array(data['x'])
data_y = np.array(data['y'])

attribute_tracker = []
for row in range(data_x.shape[1]):
    attribute_tracker.append(1)

example_1 = chooseEmotion(data_y,1)
data_merge1 = merge(data_x, example_1)
example_2 = chooseEmotion(data_y,2)
data_merge2 = merge(data_x, example_2)
example_3 = chooseEmotion(data_y,3)
data_merge3 = merge(data_x, example_3)

attr_header = []
for i in range(len(data_merge1[0])):
    attr_header.append(i)
x = decisionTree(data_x, attr_header, example_1)
x.printtree()
