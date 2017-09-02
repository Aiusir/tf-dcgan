import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
from dcgan import DCGAN

batch_size = 128

def mnistresize(images,size):
    '''
    doc: resize the mnist data
    input: images :2d [batchsize,28*28]
           size : 2d [height,weight]
    output: resize_images 4d [batchsize,height,weight,1]
    '''
    batch_size = images.shape[0]
    imgs = (np.resize(images,[batch_size,28,28]))*255
    for i in range(batch_size):
        resize_list = []
        img = Image.fromarray(imgs[i].astype(np.uint8),'L')
        img = img.resize((size[0],size[1]))
        resize_list.append(np.resize((np.array(img).astype(np.float32)/255)*2-1,[size[0],size[1],1]))
    
    resize_images = np.resize(resize_list,[batch_size,size[0],size[1],1])

    return resize_images


def train():
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    dcgan = DCGAN(channel = 1)
    train_img = tf.placeholder(tf.float32,[batch_size,64,64,1])
    losses = dcgan.loss(train_img)
    train_op = dcgan.train(losses)
    imgs = dcgan.sample_images()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for step in range(10000):
            train_images , _ = mnist.train.next_batch(batch_size)
            train_images = mnistresize(train_images,size=(64,64))

            _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.g], losses[dcgan.d]],feed_dict = {train_img:train_images})
            if step%500 ==0:
                imgs = sess.run(imgs)
                with open('train%d.jpg'%step,'wb')as f:
                    f.write(imgs)

        saver.save(sess,os.path.join('./model','model.ckpt'))
    
def test():
    dcgan = DCGAN(channel = 1)
    images = dcgan.sample_images()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,os.path.join('./model','model.ckpt'))

        generated = sess.run(images)
        with open('generated.jpg','wb') as f:
            f.write(generated)


if __name__ == '__main__':
    train()
        