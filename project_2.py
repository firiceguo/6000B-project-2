
# coding: utf-8

# In[1]:


from PIL import Image
import keras
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D



# In[2]:


def get_data(mod):
    imdir = '/Users/qiufeng/Desktop/data/'
    if mod == 'train':
        file_dir = imdir + 'train.txt'
    elif mod == 'val':
        file_dir = imdir + 'val.txt'
    else:
        print('mod = train or mod = test')
        exit()
    ims = []
    labels = []
    with open(file_dir, 'r') as f:
        line = f.readline()
        while line:
            im, label = line.split(' ')
            image = Image.open(imdir + im).resize((64, 64))
            im = np.array(image, dtype=np.float) / 255
            ims.append(im)
            labels.append(int(label))
            line = f.readline()
    labels = to_categorical(np.array(labels))
    return np.array(ims), labels


if __name__ == '__main__':
    x, y = get_data('train')
    print x.shape
    print y.shape
    print x.dtype, y.dtype


# In[3]:


x_train, y_train = get_data('train')
x_val, y_val = get_data('val')


# In[4]:


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=70,
          validation_data=(x_val, y_val),
          shuffle=True)


# In[5]:


def get_data(mod):
    imdir = '/Users/qiufeng/Desktop/data/'
    mod == 'test'
    file_dir = imdir + 'test.txt'
    ims = []
    with open(file_dir, 'r') as f:
        line = f.readline()
        while line:
            im=line.strip('\n')
            image = Image.open(imdir + im).resize((64, 64))
            im = np.array(image, dtype=np.float) / 255
            ims.append(im)
            #labels.append(int(label))
            line = f.readline()
    #labels = to_categorical(np.array(labels))
    return np.array(ims)

if __name__ == '__main__':
    x_test= get_data('test')
    #print(x[0])
    print x_test.shape


# In[6]:


prediction=model.predict_classes(x_test)


# In[7]:


df=pd.DataFrame(prediction)
df.index += 1 
df.to_csv('project2_20398324.txt',header=False,index=False)

