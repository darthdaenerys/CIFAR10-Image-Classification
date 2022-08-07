import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
x_train=(x_train)/255.0
x_test=(x_test)/255.0
x_train.shape,y_train.shape,x_test.shape,y_test.shape

x_train[0].min(),x_train[0].max()

y_train[0]

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

data_augmentation=Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomTranslation(width_factor=.3,height_factor=.3),
    tf.keras.layers.RandomContrast(.15),
    tf.keras.layers.RandomRotation(.04),
    tf.keras.layers.RandomZoom(.2),
])

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.cache()
train_data=train_data.shuffle(50000)
train_data=train_data.batch(128)
train_data=train_data.map(lambda x,y:(data_augmentation(x),y))
train_data=train_data.prefetch(64)

test_data=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data=test_data.batch(32)
test_data=test_data.prefetch(16)

train_iterator=train_data.as_numpy_iterator()

images,labels=train_iterator.next()
show_images(images[:32],labels[:32])

from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,MaxPool2D
from tensorflow.keras import Input,Model

def build_model():
    inputs=Input(shape=(32,32,3),name='input_layer')
    x=Conv2D(64,(3,3),1,padding='same',activation='relu')(inputs)
    x=BatchNormalization()(x)
    x=Conv2D(64,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=Conv2D(64,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPool2D()(x)

    x=Conv2D(128,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=Conv2D(128,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPool2D()(x)

    x=Flatten()(x)
    x=Dropout(.2)(x)
    x=Dense(256,activation='relu')(x)
    x=BatchNormalization()(x)
    x=Dropout(.2)(x)
    outputs=Dense(10,activation='softmax')(x)

    model=Model(inputs=inputs,outputs=outputs,name='cifar10_model')
    return model

model=build_model()
model.summary()

model=tf.keras.models.load_model(os.path.join('models','cifar10_model.h5'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# history=model.fit(train_data,epochs=35,validation_data=test_data,callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs')])

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

# tf.keras.models.save_model(model,os.path.join('models','cifar10_model.h5'))
# model.save('./metadata/')

"""## Predictions on training dataset"""

images,labels=train_iterator.next()
pred=model.predict(images)
show_images(images[:32],pred[:32],True)

"""## Predictions on testing dataset"""
loss,acc=model.evaluate(x_test,y_test,verbose=0)
print(f'Model Loss: {loss:.4f}')
print(f'Model Accuracy: {acc*100:.3f} %')

preds=model.predict(x_test,verbose=0)

indexes=np.random.randint(0,9999,size=(32,))
images=np.expand_dims(x_test[indexes[0]],axis=0)
pred=np.expand_dims(preds[indexes[0]],axis=0)
for i in range(1,len(indexes)):
    images=np.concatenate([images,np.expand_dims(x_test[indexes[i]],axis=0)],axis=0)
    pred=np.concatenate([pred,np.expand_dims(preds[indexes[i]],axis=0)],axis=0)
show_images(images,pred,True)