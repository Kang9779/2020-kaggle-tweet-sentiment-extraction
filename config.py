import tokenizers

class Config:
    SEED = 25
    N_FOLDS = 5
    MAX_LEN = 86
    EPOCHS = 3
    HIDDEN_SIZE = 768
    LEARNING_RATE = 0.2 * 3e-5
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    TRAINING_FILE = "./data/train_folds.csv"
    ROBERTA_PATH = "./torch-roberta"
    MODEL_SAVE_PATH = './roberta-model'
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab=f"{ROBERTA_PATH}/vocab.json", 
        merges=f"{ROBERTA_PATH}/merges.txt", 
        lowercase=True,
        add_prefix_space=True
    )
    USE_SWA = False
    WEIGHT_DECAY = 0.001
    N_LAST_HIDDEN = 12
    HIGH_DROPOUT = 0.5
    SOFT_ALPHA = 0.6
    WARMUP_RATIO = 0.25
    WEIGHT_DECAY = 0.001
    USE_SWA = False
    SWA_RATIO = 0.9
    SWA_FREQ = 30