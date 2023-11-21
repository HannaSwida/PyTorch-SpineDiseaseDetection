import pandas as pd
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np

class SpineDicomDataset(Dataset):
    def __init__(self, csv_file, parent_path=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            parent_path (string): Path to the directory containing DICOM files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dicom_images = pd.read_csv(csv_file)
        self.transform = transform
        self.parent_path = parent_path

    def __len__(self):
        return len(self.dicom_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dicom_path = self.dicom_images.iloc[idx, 0]  # accessing data in a DataFrame 
        label = self.dicom_images.iloc[idx, 2]

        # Load DICOM image
        dicom_image = pydicom.dcmread(f"{self.parent_path}/{dicom_path}").pixel_array
        dicom_image = dicom_image.astype(float)

        dicom_image -= dicom_image.min()
        dicom_image /= dicom_image.max()
        dicom_image *= 255.0
        dicom_image = dicom_image.astype(np.uint8)

        # Convert the NumPy array to a PIL Image
        dicom_image = Image.fromarray(dicom_image)

        if self.transform:
            dicom_image = self.transform(dicom_image)

        return dicom_image, label
