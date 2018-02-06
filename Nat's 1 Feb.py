def bestAttribute(examples,attribute_matrix):
    max_info_gain = 0
    max_attr = 0
    for index_attr in range(attribute_matrix.shape[1]):
        info_gain = gain(examples,attribute_matrix,index_attr)

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_attr = index_attr
    return max_attr