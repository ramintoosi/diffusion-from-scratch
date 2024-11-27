import torchvision
from torch.utils.data.dataset import Dataset
from PIL import Image
import os


class AnimeDataset(Dataset):
    """
    Reads the MNIST data from csv file given file path.
    """

    def __init__(self, root='./data/anime', num_datapoints=None):
        """
        :param root: a folder containing images
        :param num_datapoints: number of datapoints to use
        """
        super().__init__()

        self.image_paths = [os.path.join(root, x) for x in os.listdir(root)]

        if num_datapoints is not None:
            self.image_paths = self.image_paths[0:num_datapoints]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])

        img = torchvision.transforms.Resize(48)(img)
        img_tensor = torchvision.transforms.ToTensor()(img)
        img_tensor = 2 * img_tensor - 1

        return img_tensor