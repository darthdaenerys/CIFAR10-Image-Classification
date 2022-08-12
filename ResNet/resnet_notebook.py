import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
x_train=(x_train)/255.0
x_test=(x_test)/255.0
x_train.shape,y_train.shape,x_test.shape,y_test.shape

class_labels={
    0:'Airplane',
    1:'Automobile',
    2:'Bird',
    3:'Cat',
    4:'Deer',
    5:'Dog',
    6:'Frog',
    7:'Horse',
    8:'Ship',
    9:'Truck'
}

def show_images(images,labels,preds=False):
    fig=plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    plt.style.use('ggplot')
    for idx in range(images.shape[0]):
        plt.subplot(4,8,idx+1)
        img=images[idx]
        plt.imshow(img)
        if preds:
            plt.title(class_labels[np.argmax(labels[idx])])
        else:
            plt.title(class_labels[labels[idx].item()])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

from tensorflow.keras.models import Sequential

augment_data=Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomContrast(.4),
    tf.keras.layers.experimental.preprocessing.RandomRotation(.1),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(.1,.1),
])

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.cache()
train_data=train_data.shuffle(50000)
train_data=train_data.batch(128)
train_data=train_data.map(lambda x,y:(augment_data(x),y))
train_data=train_data.prefetch(64)

test_data=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data=test_data.batch(32)
test_data=test_data.prefetch(16)

train_iterator=train_data.as_numpy_iterator()

images,labels=train_iterator.next()
show_images(images[:32],labels[:32])

from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,MaxPool2D,Activation,Add
from tensorflow.keras import Input,Model,initializers

def identity_block(inputs,filters):
    [filter1,filter2,filter3]=filters
    x_skip=inputs

    # layer 1
    x=Conv2D(filter1,(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.GlorotUniform())(inputs)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # layer 2
    x=Conv2D(filter2,(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.GlorotUniform())(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # layer 3
    x=Conv2D(filter3,(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.GlorotUniform())(x)
    x=BatchNormalization()(x)

    x=Add()([x,x_skip])
    x=Activation('relu')(x)

    return x

noise=np.random.normal(size=(1,32,32,3))
identity_block(noise,[32,32,3]).shape

def convolution_block(inputs,filters):
    [filter1,filter2,filter3]=filters

    x_skip=inputs

    # layer 1
    x=Conv2D(filter1,(3,3),strides=(2,2),padding='same',kernel_initializer=initializers.GlorotUniform())(inputs)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # layer 2
    x=Conv2D(filter3,(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.GlorotUniform())(x)
    x=BatchNormalization()(x)
    
    x_skip=Conv2D(filter3,(1,1),strides=(2,2),padding='same',kernel_initializer=initializers.GlorotUniform())(x_skip)
    x_skip=BatchNormalization()(x_skip)
    
    x=Add()([x,x_skip])
    x=Activation('relu')(x)

    return x

noise=np.random.normal(size=(1,32,32,3))
convolution_block(noise,[32,32,3]).shape

def ResNet(input_shape=(32,32,3),num_classes=10):
    inputs=Input(shape=input_shape,name='input_layer')

    x=Conv2D(128,(3,3),1,padding='same')(inputs)
    
    x=convolution_block(x,[64,64,64])
    x=identity_block(x,[64,64,64])
    x=identity_block(x,[64,64,64])
    x=identity_block(x,[64,64,64])

    x=convolution_block(x,[64,64,128])
    x=identity_block(x,[64,64,128])
    x=identity_block(x,[64,64,128])
    x=identity_block(x,[64,64,128])

    x=convolution_block(x,[64,64,128])
    x=identity_block(x,[64,64,128])
    x=identity_block(x,[64,64,128])
    x=identity_block(x,[128,128,128])
    x=identity_block(x,[128,128,128])

    x=Conv2D(64,(3,3),padding='same',activation='relu')(x)

    x=Flatten()(x)
    outputs=Dense(num_classes,activation='softmax')(x)

    model=Model(inputs=inputs,outputs=outputs,name='ResNet')
    return model

model=ResNet()
model.summary()

# from tensorflow.keras.utils import plot_model

# plot_model(model,to_file='model_plot.png')

model=tf.keras.models.load_model(os.path.join('models','cifar10_resnet.h5'))

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=.002),
    metrics=['accuracy']
)

# history=model.fit(train_data,epochs=10,validation_data=test_data,callbacks=[
#     tf.keras.callbacks.TensorBoard(log_dir='logs'),
#     tf.keras.callbacks.ModelCheckpoint('temp/checkpoint',save_weights_only=True,monitor='val_accuracy',save_best_only=True)
# ])

df=pd.DataFrame(history.history)
df.to_csv('model_loss.csv',index=False)

plt.style.use('fivethirtyeight')
df=pd.read_csv('model_loss.csv')
plt.figure(figsize=(20,5))
fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(20,7))
ax1.plot(df['loss'],label='Loss')
ax1.plot(df['val_loss'],label='Val_loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Value')
ax1.set_title('Loss Trend')
ax1.legend()

ax2.plot(df['accuracy'],label='Accuracy')
ax2.plot(df['val_accuracy'],label='Val_accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Value')
ax2.set_title('Accuracy Trend')
ax2.legend()

plt.show()

# tf.keras.models.save_model(model,os.path.join('models','cifar10_resnet.h5'))
# model.save('metadata')

model.load_weights('temp/checkpoint')

model.evaluate(test_data)