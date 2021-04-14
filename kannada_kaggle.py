import numpy as np
import pandas as pd
from keras import utils
from keras.utils import np_utils
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split

train1_path = "data//train.csv"
test_path = "data//test.csv"
val_path = "data//Dig-MNIST.csv"

df_train1 = pd.read_csv(train1_path)
df_val = pd.read_csv(val_path)
X_test = pd.read_csv(test_path)

df_train = df_train1.append(df_val, sort=False)

print (X_test.head())
print (df_train.head())
# print (df_val.head())

X_train = df_train.drop(['label'], axis=1)
y_train = df_train['label']
#X_val = df_val.drop(['label'], axis=1)
#y_val = df_val['label']
X_test = X_test.drop(['id'], axis=1)

print('Dimension of training images:', np.shape(X_train))
print('Dimension of training labels:', np.shape(y_train))
# print('Dimension of validation images:', np.shape(X_val))
# print('Dimension of validation labels:', np.shape(y_val))
print('Dimension of testing images:', np.shape(X_test))

X_train = X_train.astype('float64')
#X_val = X_val.astype('float64')
X_test = X_test.astype('float64')
X_train /= 255
# X_val /= 255
X_test /= 255

X_train = np.asarray(X_train)
#X_val = np.asarray(X_val)
X_test = np.asarray(X_test)

X_train = X_train.reshape((-1, 28, 28, 1))
#X_val = X_val.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

print('Dimension of training images:', np.shape(X_train))
#print('Dimension of validation images:', np.shape(X_val))
print('Dimension of testing images:', np.shape(X_test))


y_train = np_utils.to_categorical(y_train)
#y_val = np_utils.to_categorical(y_val)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=2)

print('Build model...')
batch_size = 150
epochs = 20
input = Input(shape=(28, 28, 1), dtype='float64')

x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=output, name="kannada_model")

model.summary()

#utils.plot_model(model, "kannada_shape_info.png", show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_val, y_val))


test_pred = pd.DataFrame(model.predict(X_test))
test_pred = pd.DataFrame(test_pred.idxmax(axis=1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns={0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

print(test_pred.head())

test_pred.to_csv('mnist_submission.csv', index=False)