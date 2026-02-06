import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
# import random
# from matplotlib import pyplot as plt


class RotorBladeDatasetClean(Dataset):
    """ This class creates a dataset instance from a csv-file or DataFrame with only complete triples of
        image, label and metadata.
        Therefore, each timestamp from the DataFrame is sufficient to provide the corresponding absolute paths of
        the image, label and metadata. Thus, number of images, labels and metadata is matching and the order of
        samples, too.
    """
    def __init__(self, paths, meta_params=None, start_date=None, end_date=None, data_aug=None, transform=None,
                 radar_format=None):
        if isinstance(paths, str):
            if paths.endswith('.csv'):
                self.data_paths = pd.read_csv(paths)
        elif isinstance(paths, pd.DataFrame):
            self.data_paths = paths
        else:
            print("Wrong dataset format!")
        if start_date is not None:
            self.data_paths = self.filter_by_date(start_date, end_date)
            self.data_paths = self.data_paths.reset_index(drop=True)  # DataLoader starts enumerating from 0
        self.timestamps_list = self.data_paths["timestamp"]
        self.img_path_list = self.data_paths["image_path"]
        self.label_path_list = self.data_paths["label_path"]
        self.meta_param_list = meta_params or []
        self.meta_path_list = [self.data_paths[meta_param] for meta_param in self.meta_param_list]
        self.data_aug = data_aug
        self.transform = transform
        self.radar_format = radar_format

    def filter_by_date(self, start_date, end_date):
        """ Transform input strings for start and end date into datetime objects.
            Apply lambda function to transform timestamp columns into datetime object too.
            Make a boolean mask for date column w.r.t. start and end date. Perform bitwise and operation
             to have a boolean mask that covers both conditions.
            Use this mask as input for .loc to have the dataframe only with columns from the given interval.
        """
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        mask = (self.data_paths.iloc[:, 0].apply(lambda x: datetime.strptime(x.split('_')[0],
                                                                             "%Y-%m-%d")) >= start_datetime) & \
               (self.data_paths.iloc[:, 0].apply(lambda x: datetime.strptime(x.split('_')[0],
                                                                             "%Y-%m-%d")) <= end_datetime)

        filtered_data = self.data_paths.loc[mask]

        return filtered_data

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_file = self.img_path_list[idx]
        label_file = self.label_path_list[idx]

        img_data = np.load(img_file).astype("float32")  # float32 for albumentations
        # random.seed(a=42)  # lets you control the randomness in albumentations
        if self.radar_format == "log":
            radargram = 20 * np.log10(img_data + 0.001)
            img_data = radargram

        elif self.radar_format == "log_norm":
            max_radargram = max([max(i) for i in img_data])
            max_max_radargram = 20 * np.log10(img_data / max_radargram + 0.001)
            img_data = max_max_radargram

        elif self.radar_format == "log_norm_thresh":
            max_radargram = max([max(i) for i in img_data])
            max_max_radargram = 20 * np.log10(img_data / max_radargram + 0.001)
            # set values to log(eps) if under db threshold
            max_max_radargram[max_max_radargram < -30] = 20 * np.log10(0.001)
            img_data = max_max_radargram

        if self.data_aug is not None:
            if self.transform is not None:
                img_data = self.transform(image=img_data)["image"]
            else:
                print("Data augmentation demanded, but transform is not defined! Trying to continue without transform.")

        # # # # Intermediate check of radar data being plottable
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # c = ax1.pcolormesh(img_data)
        # ax1.set_xlabel("Distance [m]")
        # ax1.set_ylabel("Time [s]")
        # # Add a colorbar based on the pcolormesh plot
        # plt.colorbar(c, ax=ax1, label='Intensity [dB]')
        # plt.savefig("/user_path/projects/TRM-of-Wind-Turbine-Blades-with-CNNs/test_plots/plot_log.png")
        # # # # End check

        img_data = torch.from_numpy(img_data)
        label_data = torch.from_numpy(np.load(label_file))
        sample_filename = Path(img_file).stem

        param_data = []
        for meta_files in self.meta_path_list:
            meta_file = meta_files[idx]
            meta = torch.from_numpy(np.load(meta_file))
            param_data.append(torch.flatten(meta))
        param_data = [elem for sublist in param_data for elem in sublist]  # remove sublists, i.e. 'flatten' the list
        meta_data = torch.stack(param_data)  # does not work, if you do not flatten angular params (x, y)

        sample = {"image": img_data, "label": label_data, "metadata": meta_data, "filename": sample_filename}
        return sample


if __name__ == '__main__':
    """ Example script to use defined class """
    from utils.env_config import *
    from torch.utils.data import DataLoader

    parameter_dir_names = \
        ["outside_temperature", "nacelle_orientation", "pitch_angle", "rotation_speed", "wind_speed", "wind_direction"]

    main_path_filenames = os.environ.get("COMPLETE_TRIPLE_PATH")
    csv_path = os.path.join(main_path_filenames, "complete_triples_test.csv")
    RawDatasetClean = RotorBladeDatasetClean(csv_path, parameter_dir_names)
    dataset_loader = DataLoader(RawDatasetClean, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    for i, batch in enumerate(dataset_loader, 0):
        images = batch["image"]
        labels = batch["label"]
        metadata = batch["metadata"]
        print(images)
        print(labels)
        print("METADATA:", metadata)
