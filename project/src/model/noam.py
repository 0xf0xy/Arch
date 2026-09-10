import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class Noam(keras.optimizers.schedules.LearningRateSchedule):
    """Noam learning rate scheduler."""

    def __init__(self, d_model: int, warmup_steps: int, **kwargs):
        """
        Initialize the Noam scheduler.

        Args:
            d_model (int): Model dimensionality.
            warmup_steps (int): Number of warmup steps.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Compute the learning rate at the given step.

        Args:
            step: Current step.

        Returns:
            Learning rate.
        """
        step = tf.cast(step, tf.float32)
        d_model = tf.cast(self.d_model, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        """
        Return config for serialization.

        Returns:
            Config dictionary.
        """
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }
