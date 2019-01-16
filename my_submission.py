'''

2017 IFN680 Assignment Two

Scaffholding code to get you started for the 2nd assignment.


'''

import random
import numpy as np

#import matplotlib.pyplot as plt

from tensorflow.contrib import keras

from tensorflow.contrib.keras import backend as K


import assign2_utils




def euclidean_distance(vects):
    '''
    Auxiliary function to compute the Euclidian distance between two vectors
    in a Keras layer.
    '''
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#------------------------------------------------------------------------------

def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param
      y_true : true label 1 for positive pair, 0 for negative pair
      y_pred : distance output of the Siamese network    
    '''
    margin = 1
    # if positive pair, y_true is 1, penalize for large distance returned by Siamese network
    # if negative pair, y_true is 0, penalize for distance smaller than the margin
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
#------------------------------------------------------------------------------

def compute_accuracy(predictions, labels):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    @param 
      predictions : values computed by the Siamese network
      labels : 1 for positive pair, 0 otherwise
    '''
    # the formula below, compute only the true positive rate]
    #    return labels[predictions.ravel() < 0.5].mean()
    n = labels.shape[0]
    acc =  (labels[predictions.ravel() < 0.5].sum() +  # count True Positive
               (1-labels[predictions.ravel() >= 0.5]).sum() ) / n  # True Negative
    return acc

#------------------------------------------------------------------------------

def create_pairs(x, digit_indices):
    '''
       Positive and negative pair creation.
       Alternates between positive and negative pairs.
       @param
         digit_indices : list of lists
            digit_indices[k] is the list of indices of occurences digit k in 
            the dataset
       @return
         P, L 
         where P is an array of pairs and L an array of labels
         L[i] ==1 if P[i] is a positive pair
         L[i] ==0 if P[i] is a negative pair
         
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            # z1 and z2 form a positive pair
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # z1 and z2 form a negative pair
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#------------------------------------------------------------------------------

def create_warped_dataset(input_array,input_lables, rotation, warp_strength, output_size = 100000, return_y = True):
    '''
    Create an array of warped images of size output_size
    
    @param
        input_array : Nnumpy array of image inputs
        input_labels : Numpy array of lables where input_labels[i] is
            The label for input_array[i]
        rotation : Maximum rotation value
        warp_strength : Maximum intensity of the warp
        output_size : Size of the output dataset (100K images by default)
        return_y : determines if the function should return labels with the dataset
    
    @return 
        if return_y is true
            warped_array warped version of input_array
        else
            warped_array
    '''
    # create an output array with the same shape as input_array but of size output_size
    warped_array = np.zeros((output_size, len(input_array[1]), len(input_array[2])))
    output_labels = np.zeros((output_size,),dtype=input_lables.dtype)
    print('Warping images...')
    
    # randomly warp images from input_array by rotation and warp_strenth and insert them into warped_array
    for i in range(output_size):
        warped_array[i] = assign2_utils.random_deform(input_array[i % len(input_array)], rotation, warp_strength)
        output_labels[i] = int(input_lables[i% len(input_lables)])
        
#    # Return Warped Dataset
#    if len(input_array) != output_size:
#        return warped_array, output_labels #, warped_array[split_range:],output_labels[split_range:]
#    else:
    if not return_y:
        return warped_array
    else:
        return warped_array, output_labels
    
#------------------------------------------------------------------------------
def simplistic_solution(x_train, y_train,input_dim,epochs = 1,batch_size = 128):
    '''
    
    Train a Siamese network to predict whether two input images correspond to the 
    same digit.
    
    WARNING: 
        in your submission, you should use auxiliary functions to create the 
        Siamese network, to train it, and to compute its performance.
    
    
    '''
    def create_simplistic_base_network(input_dim):
        '''
        Base network to be shared (eq. to feature extraction).
        '''
        seq = keras.models.Sequential()
        seq.add(keras.layers.Conv2D(32,kernel_size=(3,3),
                activation = 'relu',
                input_shape=input_dim))
        seq.add(keras.layers.Conv2D(64, (3,3),activation = 'relu'))
        seq.add(keras.layers.Conv2D(128,(3,3),activation = 'relu'))
        seq.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        seq.add(keras.layers.Dropout(0.25))
        seq.add(keras.layers.Flatten())
        seq.add(keras.layers.Dense(128,activation='relu'))
        seq.add(keras.layers.Dropout(0.5))
        seq.add(keras.layers.Dense(128, activation='relu'))
        return seq
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
      
    
    # create training+test positive and negative pairs
    print("Training and Testing Pairs...")
    print("Creating training pairs..")
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    print("Done..")    
    
    input_shape = (input_dim[0],input_dim[1],1)  
    # network definition
    base_network = create_simplistic_base_network(input_shape)
    
    input_a = keras.layers.Input(shape=(input_shape))
    input_b = keras.layers.Input(shape=(input_shape))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # node to compute the distance between the two vectors
    # processed_a and processed_a
    distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])
    
    # Our model take as input a pair of images input_a and input_b
    # and output the Euclidian distance of the mapped inputs
    model = keras.models.Model([input_a, input_b], distance)

    # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    #INSERT IN PREDICTION SECTION (CAN OVERWRITE)
    print("computing finall accuracy on training and test sets...")
    print("predicting accuracy with training pairs..")
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    print("predicting accuracy with testing pairs...")
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)
    print("predicting accuracy with small warped set...")
    pred = model.predict([te_S_pairs[:, 0], te_S_pairs[:, 1]])
    te_S_acc = compute_accuracy(pred, te_S_y)
    print("predicting accuracy with large warped set...")
    pred = model.predict([te_L_pairs[:, 0], te_L_pairs[:, 1]])
    te_L_acc = compute_accuracy(pred, te_L_y)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Accuracy on test set (small warp): %0.2f%%' % (100 * te_S_acc))
    print('* Accuracy on test set (large warp): %0.2f%%' % (100 * te_L_acc))


#------------------------------------------------------------------------------        

#----------------------------- SETTINGS ---------------------------------------
COLOR_INTENSITY = 255 # Scalar used to normalize values in the MNIST Dataset
ROTATION = 45 # Maximum Rotational Scalar for Warped Dataset
STRENGTH = 0.3 # Maximum Strength Value for Warped Dataset

#------------------------ Load MNIST Dataset ----------------------------------
x_train, y_train, x_test, y_test  = assign2_utils.load_dataset()
input_dim = x_train.shape[1:3]# 28x28

#---=------------------------- Mixed Dataset ----------------------------------
x_a, y_a = create_warped_dataset(x_train,y_train,0,0,output_size=50000) # Unwarped Data
x_b, y_b = create_warped_dataset(x_train,y_train,(ROTATION/2),(STRENGTH/2),output_size=25000) # Light Warped Data
x_c, y_c = create_warped_dataset(x_train,y_train,ROTATION,STRENGTH,output_size=25000) # Heavy Warped Data

# Concatinate Mixed Set together
x_train_mix = np.concatenate((x_a,x_b,x_c))
y_train_mix = np.concatenate((y_a,y_b,y_c))

x_train_mix = x_train_mix.reshape(x_train_mix.shape[0],input_dim[0],input_dim[1],1)
x_train_mix = x_train_mix.astype('float32')

#--------------------------- Small Warp Dataset -------------------------------
#Create small warp training set at half of ROTATION and STRENGTH
x_train_S_warps, y_train_S_warps = create_warped_dataset(x_train,y_train,(ROTATION/2),(STRENGTH/2)) # Light Warped Data
x_train_S_warps = x_train_S_warps.reshape(x_train_S_warps.shape[0],input_dim[0],input_dim[1],1)
x_train_S_warps = x_train_S_warps.astype('float32')
#Create small warp testing set at half of ROTATION and STRENGTH
x_test_S_warps = create_warped_dataset(x_test, y_test,(ROTATION/2),(STRENGTH/2),output_size=len(x_test), return_y = False)
x_test_S_warps = x_test_S_warps.reshape(x_test_S_warps.shape[0],input_dim[0],input_dim[1],1)
x_test_S_warps = x_test_S_warps.astype('float32')

#--------------------------- Large Warp Dataset -------------------------------
#Create large warp training set at full ROTATION and STRENGTH
x_train_L_warps, y_train_L_warps  = create_warped_dataset(x_train,y_train,ROTATION,STRENGTH) # Light Warped Data
x_train_L_warps = x_train_L_warps.reshape(x_train_L_warps.shape[0],input_dim[0],input_dim[1],1)
x_train_L_warps = x_train_L_warps.astype('float32')
#Create large warp testing set at full ROTATION and STRENGTH
x_test_L_warps = create_warped_dataset(x_test, y_test,ROTATION,STRENGTH,output_size=len(x_test), return_y = False)
x_test_L_warps = x_test_L_warps.reshape(x_test_L_warps.shape[0],input_dim[0],input_dim[1],1)
x_test_L_warps = x_test_L_warps.astype('float32')

#--------------------------- Unwarped Dataset ---------------------------------
# Reshape and convert original dataset for use in the CNN
x_train = x_train.reshape(x_train.shape[0],input_dim[0],input_dim[1],1)
x_test = x_test.reshape(x_test.shape[0],input_dim[0],input_dim[1],1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test /= COLOR_INTENSITY
x_train /= COLOR_INTENSITY

#------------ ----------------- Test Pairs ------------------------------------
print("Test Pairs..")
digit_indices = [np.where(y_test == i)[0] for i in range(10)]
print("Unwarped Test Pairs..")
te_pairs, te_y = create_pairs(x_test, digit_indices)
print("Done..\nSmall Warp Test Pairs..")
te_S_pairs, te_S_y = create_pairs(x_test_S_warps, digit_indices)
print("Done..\nLarge Warp Test Pairs..")
te_L_pairs, te_L_y = create_pairs(x_test_L_warps, digit_indices)
print("Done")
print("="*25)
#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------        

if __name__=='__main__':
    
    # Unwarped Dataset
    simplistic_solution(x_train, y_train, input_dim, epochs=8)

    # Small Warp Dataset
    simplistic_solution(x_train_S_warps, y_train_S_warps, input_dim, epochs = 8)
    
    # Large Warp Dataset
    simplistic_solution(x_train_L_warps,y_train_L_warps, input_dim, epochs = 8)

    # Mixed Dataset
    simplistic_solution(x_train_mix,y_train_mix, input_dim, epochs = 8)
#    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
