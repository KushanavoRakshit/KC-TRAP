import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/UNSW_NB15_training-set.csv")

df = df.drop(columns=["id", "attack_cat"], errors="ignore")
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

BATCH_SIZE = 1
X_secret = tf.constant(X[:BATCH_SIZE], dtype=tf.float32)
y_secret = tf.constant(y[:BATCH_SIZE].reshape(-1, 1), dtype=tf.float32)

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

model = build_model(X.shape[1])
loss_fn = tf.keras.losses.BinaryCrossentropy()

with tf.GradientTape() as tape:
    pred = model(X_secret, training=True)
    loss = loss_fn(y_secret, pred)
real_grads = tape.gradient(loss, model.trainable_variables)

print("Real data (first sample):", X_secret.numpy()[0][:6], "...")

dummy_X = tf.Variable(
    tf.random.normal(X_secret.shape), trainable=True, dtype=tf.float32
)
dummy_y = tf.Variable(
    tf.random.uniform(y_secret.shape), trainable=True, dtype=tf.float32
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

print("\nStarting gradient inversion attack...")
for step in range(500):
    # compute gradients of the dummy batch w.r.t. model weights
    # tf.gradients can't be used when eager execution is on, so nest tapes
    with tf.GradientTape() as attack_tape:
        with tf.GradientTape() as dummy_tape:
            dummy_pred = model(dummy_X, training=True)
            dummy_loss = loss_fn(dummy_y, dummy_pred)
        dummy_grads = dummy_tape.gradient(dummy_loss, model.trainable_variables)

        grad_diff = sum(
            tf.reduce_sum(tf.square(dg - rg))
            for dg, rg in zip(dummy_grads, real_grads)
            if dg is not None
        )

    attack_grads = attack_tape.gradient(grad_diff, [dummy_X, dummy_y])
    optimizer.apply_gradients(zip(attack_grads, [dummy_X, dummy_y]))

    if step % 100 == 0:
        print(f"Step {step:4d} | Gradient diff loss: {grad_diff.numpy():.6f}")

print("\n--- Results ---")
print("Real data      :", X_secret.numpy()[0][:6])
print("Reconstructed  :", dummy_X.numpy()[0][:6])

mse = np.mean((X_secret.numpy() - dummy_X.numpy()) ** 2)
print(f"\nReconstruction MSE: {mse:.6f}")
print("(Lower MSE = better reconstruction = greater privacy risk)")