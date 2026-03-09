import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── 1. Load and preprocess data ────────────────────────────────────────────
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

# ── 2. Build model ─────────────────────────────────────────────────────────
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

loss_fn = tf.keras.losses.BinaryCrossentropy()

# ── 3. Function to run attack with a given noise level ─────────────────────
def run_attack(noise_multiplier):
    model = build_model(X.shape[1])

    # Get real gradients
    with tf.GradientTape() as tape:
        pred = model(X_secret, training=True)
        loss = loss_fn(y_secret, pred)
    real_grads = tape.gradient(loss, model.trainable_variables)

    # Add Gaussian noise (Differential Privacy defence)
    if noise_multiplier > 0:
        noisy_grads = []
        for g in real_grads:
            noise = tf.random.normal(g.shape, stddev=noise_multiplier)
            noisy_grads.append(g + noise)
        real_grads = noisy_grads

    # Initialize dummy data
    dummy_X = tf.Variable(tf.random.normal(X_secret.shape), trainable=True, dtype=tf.float32)
    dummy_y = tf.Variable(tf.random.uniform(y_secret.shape), trainable=True, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Attack loop - using nested GradientTape (fixes tf.gradients eager mode error)
    for step in range(500):
        with tf.GradientTape() as attack_tape:
            attack_tape.watch([dummy_X, dummy_y])
            with tf.GradientTape() as inner_tape:
                dummy_pred = model(dummy_X, training=True)
                dummy_loss = loss_fn(dummy_y, dummy_pred)
            dummy_grads = inner_tape.gradient(dummy_loss, model.trainable_variables)

            grad_diff = sum(
                tf.reduce_sum(tf.square(dg - rg))
                for dg, rg in zip(dummy_grads, real_grads)
                if dg is not None
            )

        attack_grads = attack_tape.gradient(grad_diff, [dummy_X, dummy_y])
        optimizer.apply_gradients(zip(attack_grads, [dummy_X, dummy_y]))

    mse = np.mean((X_secret.numpy() - dummy_X.numpy()) ** 2)
    return mse

# ── 4. Run for all noise levels and print comparison table ─────────────────
print("Real data (first 6 features):", X_secret.numpy()[0][:6])
print("\n" + "="*55)
print(f"{'Noise Level':<20} {'Reconstruction MSE':<20} {'Privacy'}")
print("="*55)

noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]

for noise in noise_levels:
    mse = run_attack(noise)
    if noise == 0.0:
        label = "No defence"
        privacy = "X High Risk"
    elif mse < 0.001:
        label = f"DP Noise = {noise}"
        privacy = "! Weak"
    elif mse < 0.1:
        label = f"DP Noise = {noise}"
        privacy = "* Moderate"
    else:
        label = f"DP Noise = {noise}"
        privacy = "** Strong"
    print(f"{label:<20} {mse:<20.6f} {privacy}")

print("="*55)
print("\nHigher MSE = harder to reconstruct = better privacy protection")