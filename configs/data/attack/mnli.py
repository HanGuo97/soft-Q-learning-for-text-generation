import os


base_path = "/export/share/Data/multinli/all"
max_source_length = 512
max_decoding_length = 10

source_vocab_file = os.path.join(base_path, "vocab.30000")
target_vocab_file = os.path.join(base_path, "vocab.30000")

train = {
    "batch_size": 16,
    "allow_smaller_final_batch": False,
    "source_dataset": {
        "files": os.path.join(base_path, "train.sources.blank"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "train.targets"),
        "vocab_file": target_vocab_file,
    }
}

val = {
    "batch_size": 16,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "valid.sources.blank"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "valid.targets"),
        "vocab_file": target_vocab_file,
    }
}

test = {
    "batch_size": 16,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "test.sources.blank"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "test.targets"),
        "vocab_file": target_vocab_file,
    }
}
