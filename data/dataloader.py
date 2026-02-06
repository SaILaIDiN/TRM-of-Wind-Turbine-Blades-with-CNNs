from utils.env_config import *  # Includes os
import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from data.dataset import RotorBladeDatasetClean
from data.preanalysis.create_overview_data import filter_csv_by_intervals, plot_histogram_v1
import pandas as pd
import numpy as np
import mlflow


def prepare_dataloaders(dir_yaml, experiment_id=None, experiment_name=None, run_id=None, run_name=None, transform=None):
    """ This function combines the creation of dataset instances as partitions and DataLoader instances of these,
        for cases where the DataLoaders have to be created only once.
        For situations where the dataset is created only once while multiple variations of dataloaders are to be tested,
        the procedure is divided into two sub functions.
    """
    train, val, test = prepare_dataset_partitions(dir_yaml, experiment_id, experiment_name, run_id, run_name, transform)

    with open(dir_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    torch_seed = config["torch_seed"]

    train_loader, val_loader, test_loader = create_dataloaders(train, val, test, batch_size, num_workers, torch_seed)
    return train_loader, val_loader, test_loader


def prepare_dataset_partitions(dir_yaml, experiment_id=None, experiment_name=None, run_id=None, run_name=None,
                               transform=None):
    """ Loads the parameters relevant for dataset creation from yaml-configuration and creates dataset partitions.
        Performs filtering of dataset by various conditions of meta parameters, if required.
        NOTE: The number of entries found for csv_overview_data is usually higher than csv_path_data because
        csv_overview_data only checked correctness of meta parameters and labels to the same timestamp but
        csv_path_data additionally checked the correct format of the radargram for each timestamp.
        And if this radargram is of wrong shape, it is no longer considered and listed.
        Therefore, always provide a start and end date via interval_conditions in the yaml file even if other
        conditions are not defined.
        Otherwise, the statistics of meta parameters from df_value_partitions will not represent the
        distribution of df_path_partitions (Since alignment happens in extract_correct_samples_by_meta_parameters!)
    """
    with open(dir_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Define dataset origin
    csv_path_data = config["csv_path_data"]
    csv_path_data = os.path.join(os.environ.get("COMPLETE_TRIPLE_PATH"), csv_path_data)
    csv_overview_data = config["csv_overview_data"]
    csv_overview_data = os.path.join(os.environ.get("DATA_OVERVIEW_PATH"), csv_overview_data)

    # Check for meta parameter filter conditions, apply them and update path
    value_conditions = config["value_conditions"]
    interval_conditions = config["interval_conditions"]
    diff_conditions = config["diff_conditions"]

    if any(condition is not None for condition in [value_conditions, interval_conditions, diff_conditions]):
        csv_path_data, csv_overview_data = extract_correct_samples_by_meta_parameters(dir_yaml)

    # Load and split DataFrame
    df_paths = pd.read_csv(csv_path_data)
    df_values = pd.read_csv(csv_overview_data)
    dataset_tag = config["dataset_tag"]
    p_ratios = config["partition_ratios"]
    numpy_random_seed = config["numpy_random_seed"]
    partition_sizes = [int(len(df_paths) * p_ratios[0]), int(len(df_paths) * p_ratios[1]),
                       int(len(df_paths) - int(len(df_paths) * p_ratios[0]) - int(len(df_paths) * p_ratios[1]))]
    df_path_partitions, df_value_partitions = split_dataframe(df_paths, df_values, partition_sizes=partition_sizes,
                                                              numpy_seed=numpy_random_seed, check_split="True",
                                                              dataset_tag=dataset_tag)

    # Compute statistics of meta parameters if defined
    if config["metadata_statistics"] == "True":
        parameter_csv_keys = config["parameter_csv_keys"]
        hist_bins = config["hist_bins"]
        results_path = os.environ.get("RESULTS_PATH")
        if value_conditions and interval_conditions and diff_conditions is None:
            print("No metadata conditions provided. Statistics computed for complete dataset partitions. "
                  "Check start_date and end_date in yaml configuration file!")
        for df_sub, mode in zip(df_value_partitions, ["train", "val", "test"]):
            histogram_path = path_check_and_join(results_path, [experiment_name, run_name, "histograms", mode])
            for column, bins in zip(parameter_csv_keys[:8], hist_bins):
                histogram_file_path = plot_histogram_v1(df_sub, column, bins=int(bins), output_path=histogram_path,
                                                        mode=mode)
                if config["hist_metadata_artifacts"] == "True":
                    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):
                        artifact_path = path_dissect_and_join(histogram_file_path, n_folders=3)
                        mlflow.log_artifact(histogram_file_path, artifact_path)

    # Define dataset instances
    start_date = config["start_date"]
    end_date = config["end_date"]

    parameter_dir_names = config["parameter_dir_names"]
    data_aug = config.get("data_augmentation", None)
    radar_format = config.get("radar_format", None)

    train = RotorBladeDatasetClean(df_path_partitions[0], parameter_dir_names, start_date, end_date, data_aug, transform, radar_format)
    val = RotorBladeDatasetClean(df_path_partitions[1], parameter_dir_names, start_date, end_date, data_aug, transform, radar_format)
    test = RotorBladeDatasetClean(df_path_partitions[2], parameter_dir_names, start_date, end_date, data_aug, transform, radar_format)
    print(len(train), len(val), len(test))

    return train, val, test


def create_dataloaders(train, val, test, batch_size, num_workers, torch_seed=None):
    g_cpu = torch.Generator()
    g_cpu.manual_seed(torch_seed)
    train_sampler = RandomSampler(train, generator=g_cpu)
    val_sampler = RandomSampler(val, generator=g_cpu)
    test_sampler = RandomSampler(test, generator=g_cpu)

    train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=val_sampler)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=test_sampler)
    return train_loader, val_loader, test_loader


def concat_dataset_partitions(partition_set_one, partition_set_two):
    """
    Concatenate partitions of two sets of datasets.
    Args:
        partition_set_one: List of datasets representing partitions.
        partition_set_two: List of datasets representing partitions.
    Returns:
        Concatenated datasets for each partition.
    """
    assert len(partition_set_one) == len(partition_set_two), "Both sets of partitions must have the same length."

    num_partitions = len(partition_set_one)
    concatenated_datasets = []

    for partition_idx in range(num_partitions):
        concatenated_partition = ConcatDataset([partition_set_one[partition_idx], partition_set_two[partition_idx]])
        concatenated_datasets.append(concatenated_partition)

    return concatenated_datasets


def extract_correct_samples_by_meta_parameters(dir_yaml):
    """ Finds all samples of given meta parameter conditions from overview_data_*.csv and
        uses their timestamps to parse for the file paths of these samples within complete_triples_*.csv.
        Returns a path of the extracted complete triples.
        Note: The number of entries found for csv_overview_data is usually higher than csv_path_data because
        csv_overview_data only checked correctness of meta parameters and labels to the same timestamp but
        csv_path_data additionally checked the correct format of the radargram for each timestamp.
        And if this radargram is of wrong shape, it is no longer considered and listed.
    """
    with open(dir_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)

    value_conditions = config["value_conditions"]
    interval_conditions = config["interval_conditions"]
    diff_conditions = config["diff_conditions"]

    # Initialize the DataFrame with overview of correct samples and their values
    csv_overview_data = config["csv_overview_data"]
    csv_overview_data = os.path.join(os.environ.get("DATA_OVERVIEW_PATH"), csv_overview_data)
    overview_data_df = pd.read_csv(csv_overview_data)

    # Apply filter conditions on overview_data and extract timestamps
    overview_data_df = filter_csv_by_intervals(overview_data_df, value_conditions, interval_conditions, diff_conditions)

    # Initialize the DataFrame with complete triples and all absolute paths
    csv_path_data = config["csv_path_data"]
    csv_path_data = os.path.join(os.environ.get("COMPLETE_TRIPLE_PATH"), csv_path_data)
    complete_triples_df = pd.read_csv(csv_path_data)

    # Find subset of DataFrame of complete triples by masking with timestamps from DataFrame of overview data
    complete_triples_filtered_df = \
        complete_triples_df[complete_triples_df['timestamp'].isin(overview_data_df['timestamp'])]

    # Use these timestamps again to create a filtered version of the overview_data_df for meta parameter statistics
    overview_data_filtered_df = \
        overview_data_df[overview_data_df['timestamp'].isin(complete_triples_filtered_df['timestamp'])]

    assert len(complete_triples_filtered_df) == len(overview_data_filtered_df)

    # Store filtered DataFrames to csv files
    csv_path_data_filtered = os.path.join(os.environ.get("COMPLETE_TRIPLE_PATH"),
                                          config["csv_path_data_filtered"])
    complete_triples_filtered_df.to_csv(csv_path_data_filtered, index=False)
    csv_overview_data_filtered = os.path.join(os.environ.get("DATA_OVERVIEW_PATH"),
                                              config["csv_overview_data_filtered"])
    overview_data_filtered_df.to_csv(csv_overview_data_filtered, index=False)

    return csv_path_data_filtered, csv_overview_data_filtered


def split_dataframe(df_paths, df_values, partition_sizes=None, ratios=None, numpy_seed=None, check_split=None,
                    dataset_tag=None):
    """
    Split pandas DataFrames each into multiple DataFrames based on the given ratios/partition sizes.
    Specified to randomly split both DataFrames of paths and values in the same way to keep them in sync.
    When the main DataFrame is randomly divided into three DataFrames, the initial indices are also distributed across
    the three DataFrames. The DataLoader is randomly shuffling the indices based on length of the DataFrame.
    Therefore, the DataFrames indices must be reset to count from 0 to len(df), without interruptions.
    NOTE: Only works when both DataFrames are equally sorted w.r.t. timestamps column.

    Parameters:
    - df: pandas DataFrame
    - ratios: list of floats representing the ratios for each partition.
              The sum of ratios should be 1.
    - partition_sizes: list of ints representing absolute sizes for each partition

    Returns:
    - list of pandas DataFrames representing the split datasets containing the paths.
    """
    # Shuffle the indices of the DataFrame
    try:
        numpy_seed = int(numpy_seed)
        np.random.seed(numpy_seed)
    except ValueError:
        print(f"numpy_seed must be an integer! Argument is of type {type(numpy_seed)}.")
    shuffled_indices = np.random.permutation(df_paths.index.to_list())

    # Calculate the number of rows for each partition based on the ratios
    if ratios is not None:
        total_rows = len(df_paths)
        partition_sizes = [int(r * total_rows) for r in ratios]
    elif partition_sizes is None:
        print("Missing split parameters.")
        return None

    # Split the shuffled indices into partitions
    partitions = np.split(shuffled_indices, np.cumsum(partition_sizes)[:-1])

    # Create DataFrames for each partition
    split_df_paths = [df_paths.iloc[partition] for partition in partitions]
    split_df_values = [df_values.iloc[partition] for partition in partitions]

    if check_split is not None:
        for i, df in enumerate(split_df_paths):
            df.reset_index(drop=True, inplace=True)  # Prevents mismatch of DataFrame indices and DataLoader indices
            first_entry = df["timestamp"].iloc[0]
            print("First entry of split dataframe: ", first_entry)
            df.to_csv(os.path.join(os.environ.get("COMPLETE_TRIPLE_PATH"),
                                   f"complete_triples_{dataset_tag}_filtered_partition_{i}.csv"), index=False)
        for i, df in enumerate(split_df_values):
            df.reset_index(drop=True, inplace=True)  # Prevents mismatch of DataFrame indices and DataLoader indices
            df.to_csv(os.path.join(os.environ.get("DATA_OVERVIEW_PATH"),
                                   f"overview_data_{dataset_tag}_filtered_partition_{i}.csv"), index=False)

    return split_df_paths, split_df_values


def shuffle_dataframe_in_columns(dir_df, columns_to_shuffle, shuffle_in_sync=None, dir_output=None):
    """ Takes a dataframe and shuffles named columns. Either all in sync or each one independently. """
    # Load dataframe
    df = pd.read_csv(dir_df)

    # Extract metadata columns
    metadata_cols = df.columns[df.columns.isin(columns_to_shuffle)]

    # Create a copy of metadata columns
    shuffled_metadata = df[metadata_cols].copy()

    # Shuffle rows of metadata columns
    for col in shuffled_metadata.columns:
        if shuffle_in_sync is not None:
            np.random.seed(42)  # Seed must be set every time before a numpy permutation is performed!
        shuffled_metadata[col] = np.random.permutation(shuffled_metadata[col])

    # Combine shuffled metadata with non-metadata columns
    shuffled_df = pd.concat([df.drop(columns=metadata_cols), shuffled_metadata], axis=1)

    if dir_output is not None:
        shuffled_df.to_csv(dir_output, index=False)
    else:
        print("No shuffled DataFrame stored.")

    return shuffled_df
