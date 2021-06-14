import os


base_path = "/export/share/Data/multinli/contradiction"
max_source_length = 512
max_decoding_length = 75

source_vocab_file = os.path.join(base_path, "vocab.30000")
target_vocab_file = os.path.join(base_path, "vocab.30000")

train = {
    "batch_size": 12,
    "allow_smaller_final_batch": False,
    "source_dataset": {
        "files": os.path.join(base_path, "train.sources"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "train.targets"),
        "vocab_file": target_vocab_file,
    }
}

val = {
    "batch_size": 12,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "valid.sources"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "valid.targets"),
        "vocab_file": target_vocab_file,
    }
}

test = {
    "batch_size": 12,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "valid.sources"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "valid.targets"),
        "vocab_file": target_vocab_file,
    }
}
