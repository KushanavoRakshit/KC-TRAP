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

# ── 3. Sparse gradient function — keep only top k% of gradients ───────────
def sparsify_gradients(grads, keep_ratio):
    """Zero out all but the top keep_ratio fraction of gradient values."""
    sparse_grads = []
    for g in grads:
        flat = tf.reshape(g, [-1])
        k = max(1, int(len(flat) * keep_ratio))
        # Find the threshold value
        top_k_vals = tf.math.top_k(tf.abs(flat), k=k).values
        threshold = top_k_vals[-1]
        # Zero out values below threshold
        mask = tf.cast(tf.abs(g) >= threshold, dtype=tf.float32)
        sparse_grads.append(g * mask)
    return sparse_grads

# ── 4. Function to run attack with a given sparsity level ─────────────────
def run_attack(keep_ratio):
    model = build_model(X.shape[1])

    # Get real gradients
    with tf.GradientTape() as tape:
        pred = model(X_secret, training=True)
        loss = loss_fn(y_secret, pred)
    real_grads = tape.gradient(loss, model.trainable_variables)

    # Apply sparse gradient defence
    if keep_ratio < 1.0:
        real_grads = sparsify_gradients(real_grads, keep_ratio)

    # Initialize dummy data
    dummy_X = tf.Variable(tf.random.normal(X_secret.shape), trainable=True, dtype=tf.float32)
    dummy_y = tf.Variable(tf.random.uniform(y_secret.shape), trainable=True, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Attack loop
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

# ── 5. Run for all sparsity levels and print comparison table ─────────────
print("Real data (first 6 features):", X_secret.numpy()[0][:6])
print("\n" + "="*60)
print(f"{'Sparsity Level':<25} {'Reconstruction MSE':<20} {'Privacy'}")
print("="*60)

# keep_ratio: 1.0 = send all gradients, 0.1 = send only top 10%
keep_ratios = [1.0, 0.5, 0.25, 0.1, 0.01]

for ratio in keep_ratios:
    mse = run_attack(ratio)
    if ratio == 1.0:
        label = "No defence (100%)"
        privacy = "X High Risk"
    elif mse < 0.01:
        label = f"Top {int(ratio*100)}% gradients"
        privacy = "! Weak"
    elif mse < 0.1:
        label = f"Top {int(ratio*100)}% gradients"
        privacy = "* Moderate"
    else:
        label = f"Top {int(ratio*100)}% gradients"
        privacy = "** Strong"
    print(f"{label:<25} {mse:<20.6f} {privacy}")

print("="*60)
print("\nLower keep ratio = fewer gradients shared = stronger privacy")
print("Higher MSE = harder to reconstruct = better privacy protection")