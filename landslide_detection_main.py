import tensorflow as tf
import numpy as np
from PIL import Image

import dataset

#creates a dataset from images in "earth_engine_good" directory using the first 4 landslides 
#as evaluation set, don't have to run create_dataset every time
eval_sets = [i for i in range(0,4)]
dataset.create_dataset("earth_engine_good", eval_sets)
train_xs, train_ys, eval_xs, eval_ys = dataset.load()

def print_statistics(curr_l, curr_preds, curr_y):
    '''Prints accuracy on each class as well as overall accuracy and balanced accuracy'''
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(curr_preds)):
        if curr_preds[i] == 1 and curr_y[i] == 1:
            tp += 1
        elif curr_preds[i] == 1 and curr_y[i] == 0:
            fp += 1
        elif curr_preds[i] == 0 and curr_y[i] == 1:
            fn += 1
        elif curr_preds[i] == 0 and curr_y[i] == 0:
            tn += 1
    try:   
        prec = tp/(tp+fp)
    except(ZeroDivisionError):
        prec = 0
        
    try:
        recall = tp/(tp+fn)
    except(ZeroDivisionError):
        recall = 0
        
    try:
        f1 = 2*prec*recall/(prec+recall)
    except(ZeroDivisionError):
        f1 = 0
    print("Eval: Loss:{:.3f}, landslide:{}/{}, no landslide:{}/{}, accur:{:.3f}, Mean accuracy:{:.3f}".format(curr_l,
                                                                                        tp, tp+fn, tn, fp+tn,
                                                                                        (tp+tn)/(tp+tn+fp+fn),
                                                                                        0.5*(tp/(tp+fn)+tn/(fp+tn))))

def rotate_flip_batch(x, noise_factor=0):
    """
    Randomly rotates and flips examples in given batch
    X is 5D array where the x and y image axes are axes 2 and 3,
    noise_factor is a multiplier of how much random noise we want to add to the image,
    using nonzero values of noise_factor significantly reduces performance

    Return augmented 5D array
    """
    #high = np.amax(x)
    #print("High: ", high)
    #print(x[0,0,:,:,0])
    #print(x[0,0,:,:,0]*float(256/high))
    #im = Image.fromarray(x[0,0,:,:,0]*256/float(high))
    #im.show()
    batch = x.shape[0]
    rotate_degree = np.random.choice([0,1,2,3])
    flip_axis = np.random.choice([0,2,3])
    to_select = np.random.randint(batch, size=batch//2)
    x[to_select] = np.rot90(x[to_select],axes=(2,3),k=rotate_degree)
    if noise_factor != 0:
        x= np.array(np.array(x,dtype=np.float16)+ noise_factor*np.random.random(x.shape),dtype=np.float16)
    if flip_axis != 0:
        #im = Image.fromarray(np.flip(x,axis=flip_axis)[0,0,:,:,0]/float(high))
        #im.show()
        return np.flip(x,axis=flip_axis)
    return x


print(np.shape(train_xs))
print(np.shape(train_ys))

#might want to change this if memery is an issue
batch_size = len(eval_xs)

x = tf.placeholder(tf.float32, [batch_size, 2, 750, 750, 5])
cropped = tf.random_crop(x, size = [batch_size, 2, 512, 512, 5])
cropped = tf.image.random_brightness(cropped, max_delta=0.3)
#cropped = tf.layers.batch_normalization(cropped, training = True)

conv1 = tf.layers.conv3d(cropped, filters=4, kernel_size=[3,7,7],
                         padding='same', activation=tf.nn.relu)
max_pool1 = tf.layers.max_pooling3d(conv1, pool_size=[1,2,2], strides=[1,2,2], padding='same')
#max_pool1 = tf.layers.batch_normalization(max_pool1, training = True)

conv2 = tf.layers.conv3d(max_pool1, filters=16, kernel_size=[3,5,5],
                         padding='same', activation=tf.nn.relu)
#conv2 = tf.layers.batch_normalization(conv2, training = True)

conv22 = tf.layers.conv3d(conv2, filters=16, kernel_size=[3,5,5],
                         padding='same', activation=tf.nn.relu)
max_pool2 = tf.layers.max_pooling3d(conv22, pool_size=[1,2,2], strides=[1,2,2], padding='same')
#max_pool2 = tf.layers.batch_normalization(max_pool2, training = True)

conv3 = tf.layers.conv3d(max_pool2, filters=16, kernel_size=[3,5,5],
                         padding='same', activation=tf.nn.relu)
#conv3= tf.layers.batch_normalization(conv3, training = True)
conv33 = tf.layers.conv3d(conv3, filters=16, kernel_size=[3,5,5],
                         padding='same', activation=tf.nn.relu)
max_pool3 = tf.layers.max_pooling3d(conv33, pool_size=[1,2,2], strides=[1,2,2], padding='same')
#max_pool3 = tf.layers.batch_normalization(max_pool3, training = True)

conv4 = tf.layers.conv3d(max_pool3, filters=32, kernel_size=[3,5,5],
                         padding='same', activation=tf.nn.relu)
#conv4 = tf.layers.batch_normalization(conv4, training = True)
conv44 = tf.layers.conv3d(conv4, filters=32, kernel_size=[3,5,5],
                         padding='same', activation=tf.nn.relu)
max_pool4 = tf.layers.max_pooling3d(conv44, pool_size=[1,2,2], strides=[1,2,2], padding='same')
#max_pool4 = tf.layers.batch_normalization(max_pool4, training = True)

conv5 = tf.layers.conv3d(max_pool4, filters=64, kernel_size=[3,5,5],
                         padding='same', activation=tf.nn.relu)
max_pool5 = tf.layers.max_pooling3d(conv5, pool_size=[2,2,2], strides=[2,2,2],padding='same')
#max_pool5 = tf.layers.batch_normalization(max_pool5, training = True)


keep_prob = tf.placeholder(tf.float32, None)

flattened = tf.layers.flatten(max_pool5)
fully_connected = tf.layers.dense(flattened, units=256, activation=tf.nn.relu)
dropout = tf.nn.dropout(fully_connected, keep_prob=keep_prob)
#dropout = tf.layers.batch_normalization(dropout, training = True)

output = tf.layers.dense(dropout, units=1, activation=None)

y = tf.placeholder(tf.float32, shape=[batch_size,1])
loss = tf.losses.sigmoid_cross_entropy(y, output)

preds = tf.round(tf.nn.sigmoid(output))
accuracy = tf.equal(y, preds)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
step = optimizer.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



#trains for 121 epochs
total = 0
correct = 0
for i in range(121*(len(train_xs)//batch_size)):
    s = [range(len(train_xs))]
    j = i%(len(train_xs)//batch_size)
    if j==0:
        np.random.shuffle(s)
        train_xs = train_xs[s]
        train_ys = train_ys[s]
    _, curr_l, accur = sess.run([step, loss, accuracy], feed_dict={x:rotate_flip_batch(train_xs[j*batch_size:(j+1)*batch_size]),
                                                                   y:train_ys[j*batch_size:(j+1)*batch_size],keep_prob:0.5})
    for value in accur:
        total += 1
        if value == True:
            correct +=1
            
    if i%(4*(len(train_xs)//batch_size))==0:
        print(i/(len(train_xs)//batch_size))
        print("Train: Loss:{:.3f}, Accur:{:.3f}".format(curr_l, correct/total))
        total = 0
        correct = 0
        curr_l, curr_preds, curr_y = sess.run([loss, preds, y], feed_dict={x:eval_xs, y:eval_ys,
                                                                           keep_prob:1})
        print_statistics(curr_l, curr_preds, curr_y)






