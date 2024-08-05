
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
import keras
import tensorflow as tf

import numpy as np
from keras import layers

IMAGE_SIZE = (224, 224, 3)
batch_size = 32

training_dataset, test_dataset = tf.keras.utils.image_dataset_from_directory(
    "chest_Xray",
    subset="both",
    validation_split=0.2,
    seed=1337,
    image_size=(224, 224),
    batch_size=batch_size,
)

training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

training_dataset = training_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y))


test_dataset = test_dataset.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

#train_label = np.asarray(train_label).astype('float32').reshape((-1,1))
#test_label = np.asarray(test_label).astype('float32').reshape((-1,1))


def create_model():
    vgg = VGG16(input_shape= IMAGE_SIZE, weights='imagenet', include_top=False)
    
    for layer in vgg.layers:
        layer.trainable = False
    
    output = vgg.output

    x = Flatten()(output)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs = vgg.input, outputs = x)
    model.compile(loss= "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model

model = create_model()

epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("checkpoint_{epoch}.keras"),
]

model.fit(
    training_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=test_dataset,
)

keras.utils.plot_model(model, show_shapes=True)

score = model.evaluate(
    test_dataset,
    batch_size=batch_size

)

print("Test Loss: ", score[0])
print("Test accuracy: ", score[1])

model.save("number_gpu")