import os


def setup_environment(system_mode="local-H-Drive"):
    if system_mode == "local-H-Drive":  # Example structure for local path management regarding this project
        os.environ['DATA_PATH'] = r"H:\Users\Username\Data"
        os.environ['COMPLETE_TRIPLE_PATH'] = r"H:\Users\Username\Data\filenames_csv"
        os.environ['DATA_OVERVIEW_PATH'] = r"H:\Users\Username\Data\overview_csv"
        os.environ['WRONG_FILES'] = r"H:\Users\Username\Data\wrong_files"

        os.environ['RAW_DATA_PATH_vb06'] = r"H:\Users\Username\Data\vb06"
        os.environ['LABELS_DATA_PATH_vb06'] = r"H:\Users\Username\Data\vb06_labels"
        os.environ['RADAR_DATA_PATH_vb06'] = r"H:\Users\Username\Data\vb06_radargram"
        os.environ['METADATA_PATH_vb06'] = r"H:\Users\Username\Data\vb06_metadata"
        os.environ['RAW_DATA_PATH_vb07'] = r"H:\Users\Username\Data\vb07"
        os.environ['LABELS_DATA_PATH_vb07'] = r"H:\Users\Username\Data\vb07_labels"
        os.environ['RADAR_DATA_PATH_vb07'] = r"H:\Users\Username\Data\vb07_radargram"
        os.environ['METADATA_PATH_vb07'] = r"H:\Users\Username\Data\vb07_metadata"

        os.environ['MLRUNS_LOCAL'] = \
            r"C:\github\TRM-of-Wind-Turbine-Blades-with-CNNs\mlruns"
        os.environ['CODE_PATH_OPTIMISE'] = \
            r"C:\github\TRM-of-Wind-Turbine-Blades-with-CNNs\model_optim\experiments_script"
        os.environ['CODE_PATH_ANALYSIS'] = \
            r"C:\github\TRM-of-Wind-Turbine-Blades-with-CNNs\model_analysis\experiments_script"
        os.environ['YAML_PATH_OPTIMISE'] = \
            r"C:\github\TRM-of-Wind-Turbine-Blades-with-CNNs\model_optim\experiments_yaml"
        os.environ['YAML_PATH_ANALYSIS'] = \
            r"C:\github\TRM-of-Wind-Turbine-Blades-with-CNNs\model_analysis\experiments_yaml"

        os.environ['RESULTS_PATH'] = r"H:\Users\Username\Data\results"
        os.environ['MODELS_PATH'] = r"H:\Users\Username\Data\results\models"

    elif system_mode == "remote-Server":  # Example structure for remote path management regarding this project
        os.environ['DATA_PATH'] = "/srv/data/user/measurements/"
        os.environ['COMPLETE_TRIPLE_PATH'] = "/srv/data/user/measurements/filenames_csv"
        os.environ['DATA_OVERVIEW_PATH'] = "/srv/data/user/measurements/overview_csv"
        os.environ['WRONG_FILES'] = "/srv/data/user/measurements/wrong_files"

        os.environ['RAW_DATA_PATH_vb06'] = "/srv/data/user/measurements/vb06"
        os.environ['LABELS_DATA_PATH_vb06'] = "/srv/data/user/measurements/vb06_labels"
        os.environ['RADAR_DATA_PATH_vb06'] = "/srv/data/user/measurements/vb06_radargram"
        os.environ['METADATA_PATH_vb06'] = "/srv/data/user/measurements/vb06_metadata"
        os.environ['RAW_DATA_PATH_vb07'] = "/srv/data/user/measurements/vb07"
        os.environ['LABELS_DATA_PATH_vb07'] = "/srv/data/user/measurements/vb07_labels"
        os.environ['RADAR_DATA_PATH_vb07'] = "/srv/data/user/measurements/vb07_radargram"
        os.environ['METADATA_PATH_vb07'] = "/srv/data/user/measurements/vb07_metadata"

        os.environ['MLRUNS_LOCAL'] = \
            "/home/user/projects/TRM-of-Wind-Turbine-Blades-with-CNNs/mlruns"
        os.environ['CODE_PATH_OPTIMISE'] = \
            "/home/user/projects/TRM-of-Wind-Turbine-Blades-with-CNNs/model_optim/experiments_script"
        os.environ['CODE_PATH_ANALYSIS'] = \
            "/home/user/projects/TRM-of-Wind-Turbine-Blades-with-CNNs/model_analysis/experiments_script"
        os.environ['YAML_PATH_OPTIMISE'] = \
            "/home/user/projects/TRM-of-Wind-Turbine-Blades-with-CNNs/model_optim/experiments_yaml"
        os.environ['YAML_PATH_ANALYSIS'] = \
            "/home/user/projects/TRM-of-Wind-Turbine-Blades-with-CNNs/model_analysis/experiments_yaml"

        os.environ['RESULTS_PATH'] = "/srv/data/user/measurements/results"
        os.environ['MODELS_PATH'] = "/srv/data/user/measurements/results/models"
    else:
        raise ValueError("Invalid system_mode!")


def path_check_and_join(path_main, list_folders, end_is_file=False):
    """ Check path existence and expand directories with extensions. """
    path_full = os.path.join(path_main, *list_folders)
    if not os.path.exists(path_full):
        if end_is_file:  # prevents making a file into a folder at the end of the path
            head, tail = os.path.split(path_full)
            os.makedirs(head, exist_ok=True)
            path_full = os.path.join(head, tail)
        else:
            os.makedirs(path_full, exist_ok=True)

    return path_full


def path_dissect_and_join(path_main, n_folders):
    """ Dissect and expand path names with extensions without existence check. """
    path_main_folder_list = path_main.split(sep=os.sep)
    path_full = os.path.join("", *path_main_folder_list[-(n_folders+1):-1])

    return path_full


def path_join(path_main, list_folders):
    """ Expand path names with extensions without existence check. """
    path_full = os.path.join(path_main, *list_folders)

    return path_full