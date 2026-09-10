import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
def causal_mask(seq):
    """
    Creates a causal attention mask that prevents tokens from attending to future positions
    and also masks padding tokens.

    Args:
        seq: Tensor of shape (batch_size, seq_len).

    Returns:
        Tensor of shape (batch_size, 1, seq_len, seq_len).
    """
    seq_len = tf.shape(seq)[1]

    look_ahead = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    look_ahead = look_ahead[None, None, :, :]

    padding = tf.cast(tf.equal(seq, 0), tf.float32)
    padding = padding[:, None, None, :]

    return tf.maximum(look_ahead, padding)


@keras.saving.register_keras_serializable()
class CausalMaskLayer(keras.layers.Layer):
    """A Keras layer that generates causal mask for transformer model."""

    def call(self, inputs):
        """
        Generates the necessary mask for the transformer model based on the input sequences.

        Args:
            inputs: Decoder input tensors.

        Returns:
            Decoder causal mask.
        """
        mask = causal_mask(inputs)

        return mask
