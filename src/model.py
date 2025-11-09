import tensorflow as tf

def build_model(num_features=2, output_dim=1):
    """
    Simple feed-forward model for posture risk estimation based on angles.
    Each input sample is a single frame (no sequence).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='sigmoid')  # 0–1 risk probability
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
