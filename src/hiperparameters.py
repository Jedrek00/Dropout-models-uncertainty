# random seeds to train with
RANDOM_SEEDS = [100]

# datasets to train with
DATASETS = ['fashion_mnist', 'cifar10']

# model_architectures to train
MODEL_ARCHITECTURES = ['densenet', 'convnet']

# dropout types to train with
DROPOUT_TYPES = ['standard', 'drop_connect', 'spatial']

# dropout rates to train with
DROPOUTS_RATES = [.1, .25, .5]


# Hiperparameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
NUM_OF_CLASSES = 10
