#!/usr/bin/env python3
import tensorflow as tf

"""training file for the transformer network"""
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Create and train a transformer model for machine translation.

    Parameters
    ----------
    N : int
        The number of blocks in the encoder and decoder.
    dm : int
        The dimensionality of the model.
    h : int
        The number of heads.
    hidden : int
        The number of hidden units in the fully connected layers.
    max_len : int
        The maximum number of tokens per sequence.
    batch_size : int
        The batch size for training.
    epochs : int
        The number of epochs to train for.

    Returns
    -------
    model : Transformer
        The trained transformer model.
    """
    """Load dataset"""
    dataset = Dataset(batch_size=batch_size, max_len=max_len)
    data_train = dataset.data_train
    data_valid = dataset.data_valid

    """Create transformer model"""
    model = Transformer(N, dm, h, hidden, dataset.tokenizer_pt.vocab_size,
                        dataset.tokenizer_en.vocab_size, max_len)

    """Set up optimizer with learning rate scheduling"""
    warmup_steps = 4000
    learning_rate = tf.keras.optimizers.schedules.LearningRateSchedule(
        lambda step: dm**-0.5 * tf.minimum(step**-0.5, step * warmup_steps**-1.5)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    """Compile model with sparse categorical crossentropy loss"""
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    """Training loop"""
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        steps = 0

        for (pt, en) in data_train:
            steps += 1
            with tf.GradientTape() as tape:
                """Create masks"""
                encoder_mask, combined_mask, decoder_mask = create_masks(
                    pt, en)

                """Forward pass"""
                predictions = model(pt, en, True, encoder_mask, combined_mask)
                loss = model.compiled_loss(en, predictions)

            """Compute gradients"""
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            """Update loss and accuracy"""
            batch_loss, batch_accuracy = model.evaluate(pt, en, verbose=0)
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy

            if steps % 50 == 0:
                print(f"Epoch {epoch + 1}, batch {steps}: "
                      f"loss {batch_loss:.4f} accuracy {batch_accuracy:.4f}")

        """Print epoch loss and accuracy"""
        epoch_loss /= steps
        epoch_accuracy /= steps
        print(
            f"Epoch {epoch + 1}: loss {epoch_loss:.4f} accuracy {epoch_accuracy:.4f}")

    return model
