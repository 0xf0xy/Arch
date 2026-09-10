from itertools import islice
import tensorflow as tf
import json


class DatasetLoader:
    """Loads and preprocesses the dataset from a JSONL file."""

    def __init__(self, path: str, limit: int):
        """
        Initializes the DatasetLoader instance.

        Args:
            path (str): The file path to the JSONL dataset.
            limit (int): The maximum number of samples to load. If None, loads the entire dataset.
        """
        self.path = path
        self.limit = limit

    def load(self) -> list:
        """
        Loads the dataset from the specified JSONL file and preprocesses it.

        Returns:
            A list of inputs.
        """
        inputs = []

        with open(self.path, "r", encoding="utf-8") as f:
            for sample in islice(f, self.limit):
                data = json.loads(sample)

                inputs.append(
                    f"<bos> {data['prompter']} <mos> {data['assistant']} <eos>"
                )

        return inputs


class DataPipeline:
    """Builds a TensorFlow data pipeline for training the model."""

    def __init__(self, tokenizer, batch_size: int, buffer_size: int):
        """
        Initializes the DataPipeline instance.

        Args:
            tokenizer: An instance of the Tokenizer for encoding text.
            batch_size (int): The number of samples per batch.
            buffer_size (int): The size of the buffer for shuffling the dataset.
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def build(self, inputs: list, training: bool):
        """
        Builds a TensorFlow data pipeline from the given inputs and targets.

        Args:
            inputs (list): A list of input strings.
            training (bool): Whether to shuffle the dataset for training.

        Returns:
            A TensorFlow dataset ready for training or evaluation.
        """
        tokens = self.tokenizer.encode(inputs, training=True)

        decoder_inputs = tokens[:, :-1]
        decoder_targets = tokens[:, 1:]

        dataset = tf.data.Dataset.from_tensor_slices((decoder_inputs, decoder_targets))

        if training:
            dataset = dataset.shuffle(self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
