import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Printing the shapes
print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape:", test_images.shape)
print("test_labels shape: ", test_labels.shape)

fig = plt.figure(figsize=(10,10))

nrows=3
ncols=3
for i in range(9) :
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(train_images[i]) 
    plt.title("Digit: {}".format(train_labels[i]))
    plt.axis(False)
plt.show()


# converting image pixel values to 0 - 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print("First Label before conversion:")
print(train_labels[0])

# Converting labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels  = tf.keras.utils.to_categorical(test_labels)

print("First Label after conversion:")
print(train_labels[0])


UNITS_TO_TEST = [16,8,4,2,1]
LOG_FILE= 'experiment_results.txt'
with open(LOG_FILE, 'w') as f: 
    f.write("Units\tTest Loss\tTest Accuracy\n")
    f.write("---------------------------------\n")

print(f"Starting experiments with units: {UNITS_TO_TEST}")

for units in UNITS_TO_TEST:
    print(f"\n--- Running experiment for units: {units} ---")

    # === Part 3 (ปรับปรุง) === 
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), 
        
        # Hidden Layer: ใช้ค่า 'units' จากการวนลูป
        tf.keras.layers.Dense(units=units, activation='relu'),
        
        # Output Layer: เหมือนเดิม (10 classes, softmax)
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    # === Part 4 == #
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )



    # == Part 5 == #
    history = model.fit(
        x = train_images,
        y = train_labels,
        epochs = 10,
        verbose = 0
    )

    # Showing Plot for loss
    plt.plot(history.history['loss'])
    plt.xlabel('epochs')
    plt.legend(['loss'])
    plt.show()

    # Showing plot for accuracy
    plt.plot(history.history['accuracy'], color='orange')
    plt.xlabel('epochs')
    plt.legend(['accuracy'])
    plt.show()

    # == Part 6 == #
    test_loss, test_accuracy = model.evaluate(
        x = test_images,
        y = test_labels,
        verbose = 0
    )



    print("Test Loss: %.4f"%test_loss)
    print("Test Accuracy: %.4f"%test_accuracy)

    print("Units: %d | Test Loss: %.4f | Test Accuracy: %.4f" % (units, test_loss, test_accuracy))
    
    with open(LOG_FILE, 'a') as f:
        f.write(f"{units}\t{test_loss:.4f}\t{test_accuracy:.4f}\n")
        
    print(f"\n✅ All experiments complete. Results saved to {LOG_FILE}")


# == Part 7 == #
predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

index=11

# Showing image
plt.imshow(test_images[index])

# Printing Probabilites 
print("Probabilites predicted for image at index",index)
print(predicted_probabilities[index])

print()

# Printing Predicted Class
print("Probabilities class for image at index",index)
print(predicted_classes[index])
