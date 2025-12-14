import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# =========================
# 1) Generate double spiral
# =========================
def generate_double_spiral(n_samples=1000, n_turns=2, noise=0.1, clockwise=True):
    samples_per_class = n_samples // 2
    theta = np.linspace(0, n_turns*2*np.pi, samples_per_class)
    if not clockwise:
        theta = -theta
    r = theta

    x0 = r * np.cos(theta) + np.random.normal(0, noise, samples_per_class)
    y0 = r * np.sin(theta) + np.random.normal(0, noise, samples_per_class)

    x1 = r * np.cos(theta + np.pi) + np.random.normal(0, noise, samples_per_class)
    y1 = r * np.sin(theta + np.pi) + np.random.normal(0, noise, samples_per_class)

    X = np.vstack([np.c_[x0, y0], np.c_[x1, y1]])
    y = np.array([0]*samples_per_class + [1]*samples_per_class)
    return X, y

# =========================
# 2) Training Data (2 turns)
# =========================
X_train, y_train = generate_double_spiral(n_samples=1000, n_turns=2, noise=0.1)
X_test, y_test   = generate_double_spiral(n_samples=1000, n_turns=4, noise=0.1)

# =========================
# 3) Feature Engineering: x, y, x^2, y^2
# =========================
X_train_scaled = X_train / X_train.max()
X_test_scaled  = X_test / X_test.max()

X_train_features = np.hstack([X_train_scaled, X_train_scaled**2])
X_test_features  = np.hstack([X_test_scaled, X_test_scaled**2])

# =========================
# 4) Build Neural Network 8,2,1
# =========================
model = models.Sequential([
    layers.Dense(8, activation='tanh', input_shape=(4,),
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(2, activation='tanh', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.001))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# =========================
# 5) Train
# =========================
history = model.fit(X_train_features, y_train,
                    epochs=500,
                    batch_size=10,
                    verbose=0)

# Plot Loss & Accuracy
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], color='orange')
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# =========================
# 6) Evaluate
# =========================
test_loss, test_acc = model.evaluate(X_test_features, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# =========================
# 7) Decision Boundary
# =========================
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = grid / X.max()
    grid_features = np.hstack([grid_scaled, grid_scaled**2])
    preds = np.argmax(model.predict(grid_features, verbose=0), axis=1)
    Z = preds.reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(title)
    plt.show()

plot_decision_boundary(model, X_train, y_train, "Decision Boundary - Training Data")
plot_decision_boundary(model, X_test, y_test, "Decision Boundary - Testing Data")
