

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


n_classes = 10
batch_size = 128
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

```

    Extracting /tmp/data/train-images-idx3-ubyte.gz
    Extracting /tmp/data/train-labels-idx1-ubyte.gz
    Extracting /tmp/data/t10k-images-idx3-ubyte.gz
    Extracting /tmp/data/t10k-labels-idx1-ubyte.gz



```python
def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

```


```python
def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024,n_classes]))}
                      
    biases =  {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
               
    x = tf.reshape(x, shape=[-1,28,28,1])
    
    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1'])+biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2'])+biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    
    fc = tf.reshape(conv2, [-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])
    
    output = tf.matmul(fc, weights['out'])+biases['out']
    
    return output
```


```python
def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
        
    hm_epochs = 20
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost], feed_dict = {x:epoch_x,y:epoch_y})
                epoch_loss+=c
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss',epoch_loss)
                
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y :mnist.test.labels}))
        
```


```python
train_neural_network(x)
```

    WARNING:tensorflow:From <ipython-input-4-0a92b668c81a>:9: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.
    ('Epoch', 0, 'completed out of', 20, 'loss', 1592872.3987426758)
    ('Epoch', 1, 'completed out of', 20, 'loss', 272817.15617370605)
    ('Epoch', 2, 'completed out of', 20, 'loss', 175692.16151428223)
    ('Epoch', 3, 'completed out of', 20, 'loss', 117125.1330909729)
    ('Epoch', 4, 'completed out of', 20, 'loss', 85499.470458984375)
    ('Epoch', 5, 'completed out of', 20, 'loss', 64793.656631469727)
    ('Epoch', 6, 'completed out of', 20, 'loss', 48994.600750606507)
    ('Epoch', 7, 'completed out of', 20, 'loss', 38713.226399276406)
    ('Epoch', 8, 'completed out of', 20, 'loss', 30105.757152333856)
    ('Epoch', 9, 'completed out of', 20, 'loss', 25907.183464050293)
    ('Epoch', 10, 'completed out of', 20, 'loss', 19474.237625433598)
    ('Epoch', 11, 'completed out of', 20, 'loss', 16737.644833387341)
    ('Epoch', 12, 'completed out of', 20, 'loss', 16893.694105414208)
    ('Epoch', 13, 'completed out of', 20, 'loss', 14556.08163844794)
    ('Epoch', 14, 'completed out of', 20, 'loss', 9153.939933784306)
    ('Epoch', 15, 'completed out of', 20, 'loss', 12755.895959923044)
    ('Epoch', 16, 'completed out of', 20, 'loss', 11585.15943813324)
    ('Epoch', 17, 'completed out of', 20, 'loss', 8887.6408252716064)
    ('Epoch', 18, 'completed out of', 20, 'loss', 8871.3920216076076)
    ('Epoch', 19, 'completed out of', 20, 'loss', 5862.6824158816562)
    ('Accuracy:', 0.98009998)



```python

```
