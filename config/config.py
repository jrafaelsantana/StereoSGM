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
        self.HSCALE = os.getenv('HSCALE')
        self.SCALE = os.getenv('SCALE')
        self.HFLIP = os.getenv('HFLIP')
        self.VFLIP = os.getenv('VFLIP')
        self.HSHEAR = os.getenv('HSHEAR')
        self.TRANS = os.getenv('TRANS')
        self.ROTATE = os.getenv('ROTATE')
        self.BRIGHTNESS = os.getenv('BRIGHTNESS')
        self.CONTRAST = os.getenv('CONTRAST')
        self.D_CONTRAST = os.getenv('D_CONTRAST')
        self.D_HSCALE = os.getenv('D_HSCALE')
        self.D_HSHEAR = os.getenv('D_HSHEAR')
        self.D_VTRANS = os.getenv('D_VTRANS')
        self.D_ROTATE = os.getenv('D_ROTATE')
        self.D_BRIGHTNESS = os.getenv('D_BRIGHTNESS')
        self.D_EXP = os.getenv('D_EXP')
        self.D_LIGHT = os.getenv('D_LIGHT')
        