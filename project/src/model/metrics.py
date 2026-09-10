import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
def masked_loss(y_true, y_pred):
    """
    Compute masked sparse categorical crossentropy loss.

    Args:
        y_true: True labels.
        y_pred: Predicted logits.

    Returns:
        Masked loss value.
    """
    scc_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, True)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(scc_loss, mask)

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


@keras.saving.register_keras_serializable()
def masked_accuracy(y_true, y_pred):
    """
    Compute masked sparse categorical accuracy.

    Args:
        y_true: True labels.
        y_pred: Predicted logits.

    Returns:
        Masked accuracy value.
    """
    scc_accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    accuracy = tf.multiply(scc_accuracy, mask)

    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
