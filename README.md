Intructions:

Library needed to be imported
1. scipy.io : for reading .mat file
2. numpy : for numpy array, functions, etc.
3. math : for math operation (log,...)
4. random : for random function  
5. Digraph (graphviz package): for visualizing trees
6. pickle : for turning object in python to pickle file

- To load the different .mat dataset files, change argument in loadmat
ex.
      data = scipy.io.loadmat("USER_DATASET")

<!-- - To use example data and binary target data, place data into different variable
  (indicated by Python dictionary key since our data becomes Numpy array)
ex.
      example_data = np.array(data[EXAMPLE_KEY])
      binary_target_data = np.array(data[BINARY_KEY])

- To train data, use trainTrees function
   Function: trainTrees( NUMBER_OF_EMOTION, EXAMPLE_DATA, BINARY_TARGET_DATA, LOCATION_SPLIT_VALUE)
   Return= an array of 6 trained tree objects
            [ Tree_emotion_1, Tree_emotion_2,....,Tree_emotion_6]
ex.  trainTrees(6, EXAMPLE_DATA, BINARY_TARGET_DATA, 1) -->

- To use trainTrees function, input example data and binary-target data
  It will return an array of 6 trained tree objects.

- To check decision tree from pickle format, open decision tree .pkl file
    and test it together with example data in testTrees function
  and Output will be predicted result (1 to 6 emotions) for each example data. 
