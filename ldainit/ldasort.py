import numpy as np

import sklearn
import sklearn.discriminant_analysis

import tensorflow as tf
from tensorflow import keras
# from  keras.utils import to_categorical
# from  keras.models import Sequential
# from  keras.layers import Dense
# from  keras.layers import Flatten

from  tensorflow.keras.utils import to_categorical
from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Dense
from  tensorflow.keras.layers import Flatten

def find_bias(projection,classes):
    """ Given the projection of input data into one dimension, 
    and binary 0-1 class data, computes and returns the bias b 
    maximizing the number of points of opposing classes on the
    opposite sides of b.     
    """
    
    # discretize the range of the projection
    xrange = np.linspace(min(projection),max(projection),50)

    count_zeros = []
    count_ones = []
    
    for x in xrange: 
        # count how many elements are 0 and 1 below each discretized 
        # bias value, creating a cumulative frequency distribution for each
        
        count_zeros.append(sum(np.logical_and(projection<x, classes==0)))
        count_ones.append(sum(np.logical_and(projection<x, classes==1)))

    # turns into cumulative distribution
    count_zeros = count_zeros/max(count_zeros)
    count_ones = count_ones/max(count_ones)

    # This computation determines the quantity (zeros to the left of x) + (ones to the right of x)
    sort_quantity = count_zeros + 1 - count_ones  
   
    # find the xcut threshold maximizing (zeros to the left of x) + (ones to the right of x) OR the opposite. 
    # also records which direction 
    if np.abs(np.max(sort_quantity))>np.abs(np.min(sort_quantity)):
        xcut = xrange[np.argmax(sort_quantity)]
        direction = 1
    else:
        xcut = xrange[np.argmin(sort_quantity)]
        direction = -1
        
    #the bias should correspond to subtracting this value; return the negative.
    bias = -xcut
    
    return bias, direction


def sort_LDA(points, labels, scale=1.0):
    """ Given point cloud data in format (n_points, n_features), and binarized (0,1) labels,
    this function performs one single "sorting step": Computes the linear discriminant, 
    finds the bias which maximizes sorting, and discards the points which are "sorted." 
    Returns the remaining unsorted points and labels, together with the weight and bias found. 
    """
    
    # Compute top linear discriminant.
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1,solver='lsqr')
    lda.fit(points,labels)
    
    # Use the top linear discriminant, (normalized), as the weight.
    component = lda.coef_[0]
    weight = component / np.linalg.norm(component)
    
    # Project the data on to the weights
    projection = np.matmul(weight, points.T)
        
    # Find bias maximizing sorting
    bias, direction = find_bias(projection, labels.T)
    
    #Find standard deviation of data
    stdev = np.std(projection)*scale
    
    # Find which points are not sorted. These are the points 
    # which are sufficiently far on the "wrong side" of the bias. 
    unsorted_indices = np.logical_or(np.logical_and(projection < -bias+stdev, labels == int(direction == -1)), 
                                     np.logical_and(projection > -bias-stdev, labels == int(direction == 1))
                                    )
    
    # Return the unsorted points and labels
    unsorted_points = points[unsorted_indices]
    unsorted_labels = labels[unsorted_indices]
    
    return unsorted_points, unsorted_labels, weight, bias

def ldasort(points, 
                 labels,
                 num_labels = 10, 
                 max_weights = 10, 
                 stop_at = None,
                 verbose = False, 
                scale=1.0
                       ): 
    ''' Given point cloud data in (n_points, n_features) format, labels size (n_points,), 
    a stopping point for a maximum number of weights to compute, and, if provided, a number
    of remaining points to indicate stopping (default: dimension of the dataset).
    Returns weights and biases for the first layer of a neural network. 
    '''
    
    if stop_at is None: 
        stop_at=len(points[0])
    
    sortweights=[]
    sortbiases=[]
    
    
    for i in range(num_labels):
        #binarize each label and perform the sorting game.
        
        templabels=labels==i
        temppoints=points
        
        for j in range(max_weights):
            #iteratively compute at most max_weights weights. 
            
            #perform a single sorting step, saving the weights and biases
            temppoints, templabels, tempweight, tempbias= sort_LDA(temppoints,templabels, scale=scale)
            sortweights.append(tempweight)
            sortbiases.append(tempbias)

            #stop if too few points are left to continue sorting.
            if sum(templabels)<stop_at or sum(1-templabels)<stop_at:
                if verbose:
                    print("Label {} sorted".format(i))
                break
                
           
        print("Label {} experienced maximum iteration".format(i))
                        
    return np.array(sortweights), np.array(sortbiases)

#quick setup for running experiments
def setup(intermediate_neurons=64, input_shape=64, output_shape=10):
    model = keras.Sequential()


    model.add(keras.layers.Dense(
        units=intermediate_neurons,
        input_shape=(input_shape,),
        activation='tanh',
        kernel_initializer="Orthogonal"
        ))
    

    model.add(keras.layers.Dense(
        units=output_shape, 
        activation=tf.nn.softmax
        ))
    
    return model