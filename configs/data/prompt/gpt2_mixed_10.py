import os


base_path = "/export/share/Data/pplm-gpt2"
max_decoding_length = 10

source_vocab_file = os.path.join(base_path, "vocab.source")
target_vocab_file = os.path.join(base_path, "vocab.target")

train = {
    "batch_size": 12,
    "allow_smaller_final_batch": False,
    "source_dataset": {
        "files": os.path.join(base_path, "train.sources.20210511-10"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "train.targets.20210511-10"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
val = {
    "batch_size": 7,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "validation.sources"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "validation.targets"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
test = {
    "batch_size": 7,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "validation.sources"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "validation.targets"),
        "vocab_file": target_vocab_file,
    }
}
