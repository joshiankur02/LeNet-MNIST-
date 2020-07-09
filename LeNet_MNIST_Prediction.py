import tensorflow as tf
import matplotlib.pyplot as plt



mnist=tf.keras.datasets.mnist

(x_train,y_train ), (x_test, y_test)=mnist.load_data()

rows,cols = 28,28
x_train = x_train.reshape(x_train.shape[0],rows,cols,1)
x_test = x_test.reshape(x_test.shape[0],rows,cols,1)
input_shape = (rows,cols,1)

#Normalize
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters = 6,kernel_size=(5,5),strides=(1,1),activation=tf.nn.relu,input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(tf.keras.layers.Conv2D(filters =16,kernel_size=(5,5),strides=(1,1),activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(120,activation=tf.nn.relu))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(84,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
            loss = "sparse_categorical_crossentropy",
             metrics = ["accuracy"])

model.fit(x_train,y_train,epochs=3,batch_size=120)

#validation
val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)

model.save('lenet_mnist_tensorflow.model')
new_model = tf.keras.models.load_model('lenet_mnist_tensorflow.model')

pred = new_model.predict(x_test)

x_train = x_train.reshape(x_train.shape[0],28,28)
x_test = x_test.reshape(x_test.shape[0],28,28)
plt.imshow(x_test[9999],cmap=plt.cm.binary)
plt.show()

print(y_test[9999])