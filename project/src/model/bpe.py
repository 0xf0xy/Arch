from collections import Counter, defaultdict
import numpy as np
import time
import json
import sys
import re


class BPETokenizer:
    """Byte Pair Encoding tokenizer."""

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<mos>": 3,
        "<eos>": 4,
    }

    PUNCT_PATTERN = re.compile(
        r"[^\w\s<>]",
        re.UNICODE,
    )

    TOKEN_PATTERN = re.compile(
        r"""
        <[^>\s]+>
        | \d+
        | [^\W\d_]+
        | [^\w\s]
        """,
        re.VERBOSE | re.UNICODE,
    )

    def __init__(
        self,
        max_tokens=0,
        output_sequence_length=0,
        min_frequency=0,
        standardize="lower",
        pattern_tokenize=False,
    ):
        """
        Initializes the BPETokenizer instance.

        Args:
            max_tokens (int): The maximum number of tokens in the vocabulary (including special tokens).
            output_sequence_length (int): The fixed length for output sequences.
            min_frequency (int): The minimum frequency for a token pair to be merged during training.
            standardize (str): Text standardization method: "lower", "strip_punctuation" or "lower_and_strip_punctuation".
            pattern_tokenize (bool): Whether to use regex pattern for tokenization instead of simple whitespace splitting.
        """
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.min_frequency = min_frequency
        self.standardize = standardize
        self.pattern_tokenize = pattern_tokenize

        self.token_to_id = dict(self.SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.merges = []
        self.merge_map = {}

        self.cache = {}

    def preprocess(self, text: str):
        """
        Preprocess the input text based on the standardization settings.

        Args:
            text (str): The input text to preprocess.

        Returns:
            The preprocessed text.
        """
        if not self.standardize:
            return " ".join(text.split())

        elif self.standardize == "lower":
            text = text.lower()

        elif self.standardize == "strip_punctuation":
            text = self.PUNCT_PATTERN.sub("", text)

        elif self.standardize == "lower_and_strip_punctuation":
            text = self.PUNCT_PATTERN.sub("", text.lower())

        return " ".join(text.split())

    def tokenize(self, text: str):
        """
        Tokenize the input text into words based on the tokenization settings.

        Args:
            text (str): The input text to tokenize.

        Returns:
            A list of tokens extracted from the input text.
        """
        text = self.preprocess(text)

        if self.pattern_tokenize:
            return self.TOKEN_PATTERN.findall(text)

        return text.split()

    def train(self, texts: list, verbose: bool = True):
        """
        Train the BPE tokenizer on the provided texts.

        Args:
            texts (list): The input texts to train the tokenizer on.
            verbose (bool): Whether to display training progress information.
        """
        word_freqs = Counter()

        for text in texts:
            for word in self.tokenize(text):
                if word in self.SPECIAL_TOKENS:
                    word_freqs[(word,)] += 1

                else:
                    word_freqs[tuple(word)] += 1

        charset = set()

        for word in word_freqs:
            charset.update(word)

        charset -= set(self.SPECIAL_TOKENS)

        current_id = len(self.token_to_id)

        for char in sorted(charset):
            self.token_to_id[char] = current_id
            self.id_to_token[current_id] = char

            current_id += 1

        words = {word: [list(word), freq] for word, freq in word_freqs.items()}

        pair_freqs = Counter()
        pair_words = defaultdict(set)

        for word_tuple, (tokens, freq) in words.items():
            for i in range(len(tokens) - 1):
                pair = (
                    tokens[i],
                    tokens[i + 1],
                )
                pair_freqs[pair] += freq
                pair_words[pair].add(word_tuple)

        initial_vocab_size = len(self.token_to_id)

        if verbose:
            print("Training BPE Tokenizer")
            update_every = max(1, self.max_tokens // 75)

        start_time = time.time()

        while len(self.token_to_id) < self.max_tokens:
            if not pair_freqs:
                break

            best_pair, best_freq = pair_freqs.most_common(1)[0]

            if best_freq < self.min_frequency:
                break

            merged_token = "".join(best_pair)

            self.merges.append(best_pair)
            self.merge_map[best_pair] = merged_token
            self.token_to_id[merged_token] = current_id
            self.id_to_token[current_id] = merged_token

            current_id += 1

            affected_words = list(pair_words[best_pair])

            for word_key in affected_words:
                tokens, freq = words[word_key]

                for i in range(len(tokens) - 1):
                    old_pair = (tokens[i], tokens[i + 1])

                    pair_freqs[old_pair] -= freq

                    if pair_freqs[old_pair] <= 0:
                        del pair_freqs[old_pair]

                    pair_words[old_pair].discard(word_key)

                new_tokens = []

                i = 0

                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]
                    ):
                        new_tokens.append(merged_token)

                        i += 2

                    else:
                        new_tokens.append(tokens[i])

                        i += 1

                words[word_key][0] = new_tokens

                for i in range(len(new_tokens) - 1):
                    new_pair = (new_tokens[i], new_tokens[i + 1])

                    pair_freqs[new_pair] += freq
                    pair_words[new_pair].add(word_key)

            merge_step = len(self.token_to_id) - initial_vocab_size

            if verbose and (
                merge_step % update_every == 0
                or len(self.token_to_id) == self.max_tokens
            ):
                elapsed = time.time() - start_time

                ms_per_merge = (elapsed / merge_step) * 1000

                bar_length = 20

                filled = merge_step % (bar_length + 1)

                bar = f"\033[32m{'━' * filled}\033[0m" + "━" * (bar_length - filled)

                sys.stdout.write(
                    (
                        f"\r"
                        f"\033[1m{merge_step} merges\033[0m "
                        f"{bar} "
                        f"\033[1m{elapsed:.0f}s\033[0m "
                        f"{ms_per_merge:.0f}ms/merge "
                        f"- vocab_size: "
                        f"{len(self.token_to_id)} "
                        f"- pair_freq: "
                        f"{best_freq}"
                    )
                )

                sys.stdout.flush()

        print()

    def encode_word(self, word: str):
        """
        Encode a single word into BPE tokens.

        Args:
            word (str): The input word to encode.
        """
        cached = self.cache.get(word)

        if cached is not None:
            return cached

        if word in self.SPECIAL_TOKENS:
            return [word]

        tokens = list(word)

        for pair in self.merges:
            merged = self.merge_map[pair]

            new_tokens = []

            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(merged)

                    i += 2

                else:
                    new_tokens.append(tokens[i])

                    i += 1

            tokens = new_tokens

        self.cache[word] = tokens

        return tokens

    def encode(self, texts: list, training=False):
        """
        Encode a list of texts into sequences of token IDs.

        Args:
            texts (list): The input texts to encode.
            training (bool): Whether to prepare the output for training or return variable-length sequences.

        Returns:
            A numpy array of shape (num_texts, output_sequence_length) containing token IDs if training is True,
            otherwise a list of lists of token IDs.
        """
        unk_id = self.token_to_id["<unk>"]

        encoded_texts = []

        for text in texts:
            ids = []

            for word in self.tokenize(text):
                for token in self.encode_word(word):
                    ids.append(self.token_to_id.get(token, unk_id))

                    if training and len(ids) >= self.output_sequence_length:
                        break

                if training and len(ids) >= self.output_sequence_length:
                    break

            encoded_texts.append(ids)

        if not training:
            return np.array(encoded_texts, dtype=np.int64)

        outputs = np.zeros(
            (len(encoded_texts), self.output_sequence_length),
            dtype=np.int64,
        )

        for row, ids in enumerate(encoded_texts):
            outputs[row, : len(ids)] = ids

        return outputs

    def decode(self, sequences):
        """
        Decode sequences of token IDs back into text.

        Args:
            sequences: A list of lists or a numpy array containing token IDs to decode.

        Returns:
            A list of decoded strings corresponding to the input token ID sequences.
        """
        outputs = []

        for seq in sequences:
            tokens = []

            for idx in seq:
                if idx == 0:
                    continue

                tokens.append(self.id_to_token.get(int(idx), "<unk>"))

            outputs.append(" ".join(tokens))

        return outputs

    def save(self, path):
        """
        Save the tokenizer configuration and vocabulary to a JSON file.

        Args:
            path: The file path to save the tokenizer data.
        """
        data = {
            "output_sequence_length": self.output_sequence_length,
            "standardize": self.standardize,
            "pattern_tokenize": self.pattern_tokenize,
            "token_to_id": self.token_to_id,
            "merges": [list(pair) for pair in self.merges],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self, path):
        """
        Load the tokenizer configuration and vocabulary from a JSON file.

        Args:
            path: The file path to load the tokenizer data.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.output_sequence_length = data["output_sequence_length"]
        self.standardize = data["standardize"]
        self.pattern_tokenize = data["pattern_tokenize"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.merge_map = {pair: "".join(pair) for pair in self.merges}
