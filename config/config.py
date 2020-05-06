import os
from dotenv import load_dotenv
from pathlib import Path


class Config:
    def __init__(self):
        env_path = Path('.') / '.env'
        load_dotenv(dotenv_path=env_path)

        self.penalty_equal_1 = os.getenv('PENALTY_EQUAL_1')
        self.penalty_bigger_than_1 = os.getenv('PENALTY_BIGGER_THEN_1')

        self.kernel_size_census = os.getenv('KERNEL_SIZE_CENSUS')
        self.blur_size = os.getenv('BLUR_SIZE')

        self.patch_height = os.getenv('PATCH_HEIGHT')
        self.patch_width = os.getenv('PATCH_WIDTH')
        self.height_stride = os.getenv('HEIGHT_STRIDE')
        self.width_stride = os.getenv('WIDTH_STRIDE')
        self.channel_number = os.getenv('CHANNEL_NUMBER')

        self.use_cuda = os.getenv('USE_CUDA')

        self.train_correct = os.getenv('TRAIN_CORRECT')
        self.train_incorrect = os.getenv('TRAIN_INCORRECT')

        self.seed = os.getenv('SEED')

        self.dataset_train = os.getenv('DATASET_TRAIN')
        self.dataset_test = os.getenv('DATASET_TEST')

        self.epochs_number = os.getenv('EPOCHS_NUMBER')
        self.batch_size = os.getenv('BATCH_SIZE')
