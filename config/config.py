import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        self.max_disparity = os.getenv('MAX_DISPARITY')

        self.penalty_equal_1 = os.getenv('PENALTY_EQUAL_1')
        self.penalty_bigger_than_1 = os.getenv('PENALTY_BIGGER_THEN_1')

        self.kernel_size_census = os.getenv('KERNEL_SIZE_CENSUS')
        self.blur_size = os.getenv('BLUR_SIZE')

        self.patch_height = os.getenv('PATCH_HEIGHT')
        self.patch_width = os.getenv('PATCH_WIDTH')
        self.heght_stride = os.getenv('HEIGHT_STRIDE')
        self.width_stride = os.getenv('WIDTH_STRIDE')
        self.channel_number = os.getenv('CHANNEL_NUMBER')

        self.use_cuda = os.getenv('USE_CUDA')

        self.train_correct = os.getenv('TRAIN_CORRECT')
        self.train_incorrect = os.getenv('TRAIN_INCORRECT')
