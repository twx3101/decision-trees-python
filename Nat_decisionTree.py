import scipy.io
import numpy as np
#np.set_printoptions(threshold=np.inf)
import math

INDEX_LAST_ELEMENT = -1

data = scipy.io.loadmat("cleandata_students.mat")

array_data = np.array(data)
data_x = np.array(data['x'])
data_y = np.array(data['y'])

attribute_tracker = []
for row in range(data_x.shape[1]):
    attribute_tracker.append(1)

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
    for element in column:
        if element == target:
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

def entropy(data_merge):
    num_one = count(data_merge[:,INDEX_LAST_ELEMENT],1)
    num_zero = count(data_merge[:,INDEX_LAST_ELEMENT],0)
    summation = num_one + num_zero
    left_side = num_one/summation
    right_side = num_zero/summation

    if left_side == 0:
        return -((right_side)*math.log(right_side,2))
    elif right_side ==0 :
        return (-1*(left_side)*math.log(left_side,2))
    else:
        return (-1*(left_side)*math.log(left_side,2))-((right_side)*math.log(right_side,2))

def gain(data_merge,col_attribute):
    return entropy(data_merge)-remainder(data_merge,col_attribute)

def remainder(data_merge,col_attribute):
    left_array = []
    right_array = []

    for row in data_merge:
        if row[col_attribute] == 1:
            left_array.append(row)
        else:
            right_array.append(row)

    left_array = np.asarray(left_array)
    right_array = np.asarray(right_array)

    if left_array.size == 0:
        left_entropy = 0
        p0_left = 0
        n0_left = 0
    else:
        left_entropy = entropy(left_array)
        p0_left = count(left_array[:,INDEX_LAST_ELEMENT],1)
        n0_left = count(left_array[:,INDEX_LAST_ELEMENT],0)
    p0_n0 = p0_left + n0_left

    if right_array.size == 0:
        right_entropy = 0
        p1_right = 0
        n1_right = 0
    else:
        right_entropy = entropy(right_array)
        p1_right = count(right_array[:,INDEX_LAST_ELEMENT],1)
        n1_right = count(right_array[:,INDEX_LAST_ELEMENT],0)
    p1_n1 = p1_right + n1_right

    number_parent = count(data_merge[:,INDEX_LAST_ELEMENT],1) + count(data_merge[:,INDEX_LAST_ELEMENT],0)

    return  (p0_n0/number_parent)*left_entropy + (p1_n1/number_parent)*right_entropy

def chooseBestAttr(data_merge,attr_header):
    col_best_attr = 0
    max_gain = 0

    if data_merge.size == 0:
        return

    for col_attr in range(data_merge.shape[1]-1):
        if col_attr not in attr_header:
            continue
        attr_gain = gain(data_merge,col_attr)
        print('col attr1: ',col_attr, " : values1 = ",attr_gain)

        if attr_gain > max_gain:
            max_gain = attr_gain
            col_best_attr = col_attr

    return col_best_attr

#if there is no attr, return majority of value in emotion example
def getMajorityValue(data_merge):
    num_zero = 0
    num_one = 0
    for row in data_merge:
        if row[INDEX_LAST_ELEMENT] == 1:
            num_one += 1
        else:
            num_zero += 1
    if num_one > num_zero:
        return 1
    else:
        return 0

# check whether emotion value in emotion example has ALL same value or not
def hasSameValueEmotion(data_merge):
    first_item = data_merge[0][INDEX_LAST_ELEMENT]
    for row in data_merge:
        if row[INDEX_LAST_ELEMENT] != first_item:
            return False
    return True

# get element in attr column
def getType(data_merge,col_best_attr):
    array_element = []
    for row in data_merge:
        if row[col_best_attr] not in array_element:
            array_element.append(row[col_best_attr])
    return array_element

# get combination array according to different value in attr col
# split tree  
def getDataSample(data_merge, col_best_attr, val):
    array_row = []
    for row in data_merge:
        if row[col_best_attr] == val:
            array_row.append(row)
    return np.asarray(array_row)

#check whether sample has same value or not
def isSameSample(data_merge):
    temp_data_merge = np.copy(data_merge[0][:INDEX_LAST_ELEMENT])
    i = 0
    for row in data_merge:
        if i == 0:
            i += 1
            continue
        if not np.array_equal(temp_data_merge,row[:INDEX_LAST_ELEMENT]):
            return False
    return True

def initiateDecisionTree(data_merge,attr_header):

    data_merge = data_merge[:]
    majority = getMajorityValue(data_merge)

    if isSameSample(data_merge):
        return data_merge[0][INDEX_LAST_ELEMENT]
    elif data_merge.size == 0:
        return majority
    elif hasSameValueEmotion(data_merge):
        return data_merge[0][INDEX_LAST_ELEMENT]
    else:
        col_best_attr = chooseBestAttr(data_merge,attr_header)
        print("COLUMN BEST ATTR : ", col_best_attr , " ~~~~~~~~~~~~~~~~~~~~~~")
        tree = {col_best_attr : {}}
        array_attr_element = getType(data_merge,col_best_attr)

        for element in array_attr_element:
            new_data_merge = getDataSample(data_merge, col_best_attr, element)
            new_attr = attr_header[:]
            print("new_attr: ", new_attr )
            print("Length new_attr : ",len(new_attr))
            print("Data_merge : ",data_merge)
            new_attr.remove(col_best_attr)
            print("After new_attr: ", new_attr)
            print('\n')
            subtree = initiateDecisionTree(new_data_merge,new_attr)
            tree[col_best_attr][element] = subtree
    return tree

example_1 = chooseEmotion(data_y,1)
data_merge1 = merge(data_x, example_1)
example_2 = chooseEmotion(data_y,2)
data_merge2 = merge(data_x, example_2)
example_3 = chooseEmotion(data_y,3)
data_merge3 = merge(data_x, example_3)

# attr_header created for tracking attr that is already used
attr_header = []
for i in range(len(data_merge1[0])):
    attr_header.append(i)

print(attr_header)
print(initiateDecisionTree(data_merge1,attr_header))


# {23: {0: {6: {0: {3: {1: {43: {0: {24: {1: 0, 0: {44: {0: {32: {0: {38: {0: {22: {0: 0, 1: {4: {0: 0, 1: 1}}}}, 1: {18: {0: 1, 1: 0}}}}, 1: {7: {0: 0, 1: 1}}}}, 1: 0}}}}, 1: {16: {1: {35: {0: {4: {0: 0, 1: {13: {0: 1, 1: 0}}}}, 1: 1}}, 0: 0}}}}, 0: {30: {0: 0, 1: {43: {0: 0, 1: {29: {1: 1, 0: 0}}}}}}}}, 1: {3: {1: {16: {0: {15: {0: {29: {0: 0, 1: {5: {0: 0, 1: 1}}}}, 1: {0: {1: {1: {0: {4: {0: {14: {0: 1, 1: 0}}, 1: 0}}, 1: 0}}, 0: 1}}}}, 1: {12: {0: {7: {0: {0: {0: 1, 1: {19: {1: 0, 0: 1}}}}, 1: 0}}, 1: 0}}}}, 0: {15: {0: {11: {1: 0, 0: {1: {0: {19: {0: {12: {0: {16: {1: 0, 0: {37: {0: {39: {0: 0, 1: 1}}, 1: 1}}}}, 1: {33: {0: 1, 1: 0}}}}, 1: 0}}, 1: 0}}}}, 1: {0: {0: 1, 1: 0}}}}}}}}, 1: {3: {0: {16: {0: 0, 1: {22: {1: {27: {0: {17: {0: 0, 1: 1}}, 1: 1}}, 0: {8: {0: {11: {1: 0, 0: {24: {0: {32: {0: {38: {0: {12: {0: 1, 1: {0: {0: {6: {0: 0, 1: 1}}, 1: 1}}}}, 1: 0}}, 1: 0}}, 1: 0}}}}, 1: 0}}}}}}, 1: {14: {1: {8: {1: 0, 0: {11: {0: {25: {0: {28: {0: 1, 1: 0}}, 1: 0}}, 1: 0}}}}, 0: {8:
#{0: {39: {0: {35: {0: 1, 1: {38: {0: 1, 1: 0}}}}, 1: {24: {1: {21: {0: 0, 1: 1}}, 0: 1}}}}, 1: {43: {1: 0, 0: {13: {0: 1, 1: {26: {0: 0, 1: 1}}}}}}}}}}}}}}
