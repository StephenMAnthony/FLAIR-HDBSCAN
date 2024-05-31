import numpy as np
import pandas as pd
import rasterio
import datetime
import torch

from torch.utils.data import Dataset
from skimage import img_as_float

from utils_dataset import filter_dates


class FusedDataset(Dataset):

    def __init__(self, dict_files, config):

        # Aerial images
        self.list_aerial = np.array(dict_files["PATH_IMG"])
        # Sentinel Images
        self.list_satellite = np.array(dict_files["PATH_SP_DATA"])
        # Sentinel Cloud Masks
        self.list_clouds = np.array(dict_files["PATH_SP_MASKS"])
        # Sentinel Dates
        self.list_dates = np.array(dict_files["PATH_SP_DATES"])
        # Coordinates of the aerial image in the sentinel super area
        self.list_coords = np.array(dict_files["SP_COORDS"])

        # Labels
        self.list_labels = np.array(dict_files["PATH_LABELS"])

        self.use_metadata = config['aerial_metadata']
        if self.use_metadata:
            self.list_metadata = np.array(dict_files["MTD_AERIAL"])

        self.ref_year = config["ref_year"]
        self.ref_date = config["ref_date"]
        self.sat_patch_size = config["sat_patch_size"]
        self.num_classes = config["num_classes"]
        self.filter_mask = config["filter_clouds"]
        self.average_month = config["average_month"]

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_labels(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_label:
            labels = src_label.read()[0]
            labels[labels > self.num_classes] = self.num_classes
            labels = labels-1
            return labels

    def read_superarea_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode='r')
        half_patch = int(self.sat_patch_size/2)
        subset_sp = data[:, :, idx_centroid[0]-half_patch:idx_centroid[0]+half_patch,
                         idx_centroid[1]-half_patch:idx_centroid[1]+half_patch]
        return subset_sp

    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, 'r') as f:
            products = f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]),
                                                 int(self.ref_date.split('-')[1]))
                              - datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days)
            dates_arr.append(datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])))
        return np.array(diff_dates), np.array(dates_arr)

    def monthly_image(self, sp_patch, sp_raw_dates):
        average_patch, average_dates = [], []
        month_range = pd.period_range(start=sp_raw_dates[0].strftime('%Y-%m-%d'),
                                      end=sp_raw_dates[-1].strftime('%Y-%m-%d'), freq='M')
        for m in month_range:
            month_dates = list(filter(lambda i: (sp_raw_dates[i].month == m.month) and
                                                (sp_raw_dates[i].year == m.year), range(len(sp_raw_dates))))
            if len(month_dates) != 0:
                average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                average_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]),
                                                        int(self.ref_date.split('-')[1]))
                                     - datetime.datetime(int(self.ref_year), int(m.month), 15)).days)
        return np.array(average_patch), np.array(average_dates)

    def __len__(self):
        return len(self.list_aerial)

    def __getitem__(self, index):

        # aerial image
        image_file = self.list_aerial[index]
        img = self.read_img(raster_file=image_file)
        img = img_as_float(img)

        # metadata aerial images
        if self.use_metadata:
            mtd = self.list_metadata[index]
        else:
            mtd = []

        # labels
        labels_file = self.list_labels[index]
        labels = self.read_labels(raster_file=labels_file)

        sp_file = self.list_satellite[index]
        sp_file_coords = self.list_coords[index]
        sp_file_dates = self.list_dates[index]
        sp_file_clouds = self.list_clouds[index]

        sp_patch = self.read_superarea_and_crop(sp_file, sp_file_coords)
        sp_dates, sp_raw_dates = self.read_dates(sp_file_dates)
        sp_mask = self.read_superarea_and_crop(sp_file_clouds, sp_file_coords)
        sp_mask = sp_mask.astype(int)

        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask)
            sp_patch = sp_patch[dates_to_keep]
            sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]

        if self.average_month:
            sp_patch, sp_dates = self.monthly_image(sp_patch, sp_raw_dates)
        
        sp_patch = img_as_float(sp_patch)

        return {"aerial": torch.as_tensor(img, dtype=torch.float),
                "satellite": torch.as_tensor(sp_patch, dtype=torch.float),
                "dates": torch.as_tensor(sp_dates, dtype=torch.float),
                "labels": torch.as_tensor(labels, dtype=torch.float),
                "mtd": torch.as_tensor(mtd, dtype=torch.float),
                }
