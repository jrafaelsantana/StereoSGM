import os
from dotenv import load_dotenv
from pathlib import Path


class Config:
    def __init__(self):
        env_path = Path('.') / '.env'
        load_dotenv(dotenv_path=env_path)

        self.penalty_equal_1 = os.getenv('PENALTY_EQUAL_1')
        self.penalty_bigger_than_1 = os.getenv('PENALTY_BIGGER_THEN_1')

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

        #Augumentation
        self.augumentation_hshift = os.getenv('AUGUMENTATION_HSHIFT')
        self.augumentation_vshift = os.getenv('AUGUMENTATION_VSHIFT')
        self.augumentation_bright_low = os.getenv('AUGUMENTATION_BRIGHT_LOW')
        self.augumentation_bright_high = os.getenv('AUGUMENTATION_BRIGHT_HIGH')
        self.augumentation_zoom = os.getenv('AUGUMENTATION_ZOOM')
        self.augumentation_chnshift = os.getenv('AUGUMENTATION_CHNSHIFT')
        self.augumentation_hflip = os.getenv('AUGUMENTATION_HFLIP')
        self.augumentation_vflip = os.getenv('AUGUMENTATION_VFLIP')
        self.augumentation_rotangle = os.getenv('AUGUMENTATION_ROTANGLE')