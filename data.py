# the data is stored in the data/ directory as many 30MB text files
# each file contains 20_000 games
# each game is a sequence of moves

# a sample file looks like this:
# $ head -n 10 data/moves_42.txt
# 24 23 32 25 14 13 12 15 52 02 03 11 16 26 01 54 21 53 63 17 55 42 04 31 36 46 22 20 30 45 07 05 35 06 37 73 62 64 51 71 10 60 40 61 41 56 75 65 76 50 47 00 66 67 27 57 70 72 77 PASSS 74
# 42 32 23 52 31 22 13 14 51 30 20 54 55 41 62 21 35 72 11 24 05 60 63 56 53 01 15 64 73 26 40 61 45 46 36 12 03 10 71 25 67 66 47 50 57 04 06 37 74 02 00 16 27 17 77 75 07 70 65 76
# 42 54 25 22 65 53 45 51 11 35 55 23 62 00 32 66 14 71 21 31 77 76 64 15 16 46 30 63 60 05 61 20 56 03 12 75 40 73 74 13 10 01 36 67 02 07 72 26 57 41 17 50 37 27 24 70 06 52 04 47
# there are many such lines in the file
# xy are coordinates on the board, the board is 8x8 and initally 33, 34, 43,and 44 are already occupied so they never appear in the data.
# there is also a special token "PASS" that indicates a pass move
# we want to convert this to a sequence of tokens

import torch
from torch.utils.data import DataLoader
from itertools import product
import os
from datasets import Dataset, load_dataset  # Added import

class Tokenizer:
    def __init__(self, max_length):
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_size = 0
        self.pad_token_id = None  # Will be set in vocab_map
        self.max_length = max_length
        self.vocab_map()

    def vocab_map(self):
        for x, y in product(range(8), range(8)):
            if (x, y) not in [(3, 3), (3, 4), (4, 3), (4, 4)]:
                self.token_to_index[f"{x}{y}"] = len(self.token_to_index)
                self.index_to_token[len(self.index_to_token)] = f"{x}{y}"
        
        # Add special tokens
        special_tokens = ["PASSS", "PAD"]
        for token in special_tokens:
            self.token_to_index[token] = len(self.token_to_index)
            self.index_to_token[len(self.index_to_token)] = token
        
        self.vocab_size = len(self.token_to_index)
        self.pad_token_id = self.token_to_index["PAD"]

    def encode(self, text):
        # Convert text to token indices
        tokens = text.split()
        encoded = [self.token_to_index.get(token, self.pad_token_id) for token in tokens]
        
        # Truncate or pad to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded += [self.pad_token_id] * (self.max_length - len(encoded))
        return encoded
    
    def decode(self, tokens):
        return [self.index_to_token[token] for token in tokens]
    
def get_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(num_proc, max_sequence_length):
    data_dir = "train"
    dataset = load_dataset('text', data_dir=data_dir)['train']
    tokenizer = Tokenizer(max_length=max_sequence_length)  # Initialize with max length
    
    def tokenize_function(examples):
        # First encode all texts to get proper truncated/padded sequences
        encoded_texts = [tokenizer.encode(text) for text in examples["text"]]
        
        # Create attention masks based on actual encoded sequences
        attention_masks = [
            [1 if token != tokenizer.pad_token_id else 0 for token in seq]
            for seq in encoded_texts
        ]
        
        return {
            "input_ids": encoded_texts,
            "attention_mask": attention_masks
        }
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc
    )
    
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    data_loader = get_data_loader(tokenized_dataset, batch_size=32)

if __name__ == "__main__":
    num_proc = 32
    max_seq_length = 70
    main(num_proc, max_seq_length)