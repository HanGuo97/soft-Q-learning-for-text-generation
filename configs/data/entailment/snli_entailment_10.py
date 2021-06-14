import os


base_path = "/export/share/Data/snli/all"
max_source_length = 100
max_decoding_length = 65

source_vocab_file = os.path.join(base_path, "vocab.None")
target_vocab_file = os.path.join(base_path, "vocab.None")
val_unique_pairs_file = os.path.join(base_path, "validation-unique.pairs.unique.pth")
test_unique_pairs_file = os.path.join(base_path, "test-unique.pairs.unique.pth")

train = {
    "batch_size": 32,
    "allow_smaller_final_batch": False,
    "source_dataset": {
        "files": os.path.join(base_path, "../threshold/train.sources.0.1"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "../threshold/train.targets.0.1"),
        "vocab_file": target_vocab_file,
    }
}

val = {
    "batch_size": 32,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "validation-unique.sources"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "validation-unique.targets"),
        "vocab_file": target_vocab_file,
    }
}

test = {
    "batch_size": 32,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "test-unique.sources"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "test-unique.targets"),
        "vocab_file": target_vocab_file,
    }
}
