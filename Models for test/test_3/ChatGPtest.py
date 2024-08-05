from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import image_dataset_from_directory
import tensorflow as tf
import keras

batch_size = 32

training_dataset, test_dataset = image_dataset_from_directory(
    "chest_Xray",
    subset="both",
    validation_split=0.2,
    seed=1337,
    image_size=(224, 224),
    batch_size=batch_size,
)

training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(224,224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

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