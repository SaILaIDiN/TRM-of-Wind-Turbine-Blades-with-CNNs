import mlflow
import os
import math

def get_mlflow_experiment(experiment_name):
    """ Check if experiment with name experiment_name exist. Otherwise, create an experiment with that name. Get ID. """
    try:
        EXPERIMENT = mlflow.get_experiment_by_name(experiment_name)
        EXPERIMENT_ID = EXPERIMENT.experiment_id
    except AttributeError:
        EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
    print("Experiment_ID", EXPERIMENT_ID)
    return EXPERIMENT_ID


def get_mlflow_run(experiment_id, run_name, config_path_save=None):
    """ For a given run_name create a run and extract its unique run_id to access it multiple times afterward. """
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        RUN_ID = run.info.run_id
        if isinstance(config_path_save, str):
            mlflow.log_artifact(config_path_save, artifact_path=run_name)
        elif isinstance(config_path_save, list):
            [mlflow.log_artifact(config, artifact_path=run_name) for config in config_path_save]
    print("Run_ID", RUN_ID)
    return RUN_ID


def average_metric_of_all_runs(experiment_name, metric_names, average_type="average"):
    """ Searches for the specified metrics in each run, loads them, averages them and then logs them as another metric
        in the same run. Used if an averaging process is not performed in the initial run of an experiment.
        metric_name in metric_names is the full term of a metric with all subdirectories, also shown in the UI.
        Instead of 'f1_score' it would be 'performance_metrics/Test/f1_score/no_anomaly'.
    """
    try:
        EXPERIMENT = mlflow.get_experiment_by_name(experiment_name)
        EXPERIMENT_ID = EXPERIMENT.experiment_id
    except AttributeError:
        print("Experiment with name ", experiment_name, "does not exist!")
        return
    print("Experiment_ID", EXPERIMENT_ID)

    runs_df = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID])  # Returns a DataFrame with all run_ids and properties
    run_id_list = runs_df["run_id"].tolist()  # Format column to list
    for run_id in run_id_list:
        print("Run ID", run_id)

        # Retrieve history of metric, not only one value
        client = mlflow.tracking.MlflowClient()  # Interface to the Tracking Server
        metric_histories = []  # collects multiple metrics over same run
        for metric_name in metric_names:
            metric_history_tmp = client.get_metric_history(run_id, metric_name)  # Returns list of Metric instances
            metric_histories.append(metric_history_tmp)

        # Log average of chosen metrics with respect to epochs to the existing run with run_id
        mlflow.set_experiment(experiment_name)  # Set this experiment as active
        with mlflow.start_run(run_id=run_id):  # (Re-)Starts a run with run_id in the set experiment
            for m_by_step in zip(*metric_histories):
                m_avg = sum(m.value for m in m_by_step) / len(m_by_step)
                epoch = m_by_step[0].step  # Assuming all metrics have the same step (epoch)
                print("Average: ", m_avg, "at epoch: ", epoch)
                adjusted_path = os.path.join(*metric_names[0].split(sep=os.sep)[:-1])
                adjusted_path = os.path.join(adjusted_path, average_type)
                mlflow.log_metrics({adjusted_path: m_avg}, step=epoch)
                print("Logged average metric!")


def average_metric_of_selected_runs(experiment_name, run_id_list, metric_name, new_run_name, average_type="average"):
    """ For an existing experiment create the average of a given metric for all given run_id and store this in a new run.
        metric_name is the full term of a metric with all subdirectories, also shown in the UI.
        Instead of 'f1_score' it would be 'performance_metrics/Test/f1_score/no_anomaly'.
    """
    metric_histories = []  # collects same metric over multiple runs
    for run_id in run_id_list:
        print("Run ID", run_id)

        # Retrieve history of metric, not only one value
        client = mlflow.tracking.MlflowClient()  # Interface to the Tracking Server
        metric_history_tmp = client.get_metric_history(run_id, metric_name)  # Returns list of Metric instances
        metric_histories.append(metric_history_tmp)

    mlflow.set_experiment(experiment_name)  # Set this experiment as active
    experiment_id = get_mlflow_experiment(experiment_name)
    run_id_new = get_mlflow_run(experiment_id=experiment_id, run_name=new_run_name)
    with mlflow.start_run(run_id=run_id_new):  # Starts a run with run_id in the set experiment
        for m_by_step in zip(*metric_histories):
            m_avg = sum(m.value for m in m_by_step) / len(m_by_step)
            variance = sum((m.value - m_avg) ** 2 for m in m_by_step) / len(m_by_step)  # population std
            std = math.sqrt(variance)
            epoch = m_by_step[0].step  # Assuming all metrics have the same step (epoch)
            print("Average: ", m_avg, "at epoch: ", epoch)
            adjusted_path_core = os.path.join(*metric_name.split(sep=os.sep)[:-1])
            adjusted_path = os.path.join(adjusted_path_core, average_type)
            mlflow.log_metrics({adjusted_path: m_avg}, step=epoch)
            adjusted_path = os.path.join(adjusted_path_core, "std")
            mlflow.log_metrics({adjusted_path: std}, step=epoch)
            print("Logged average metric!")


def check_mlflow_run_metrics(run_id, metric_names):
    """ This snippet shows a way to access metrics logged in a run.
        Note: This way only gets the last logged value of a metric. (last step)
    """
    try:
        run = mlflow.get_run(run_id)
        keys = run.data.metrics.keys()
        print("Keys: ", keys)
        for metric_name in metric_names:
            metric_value = run.data.metrics[metric_name]
            print(f"Metric {metric_name}: ", metric_value)
        # # Example for what happens in the loop
        # metric_1_value = run.data.metrics["performance_metrics/Test/f1_score/anomaly"]
        # metric_2_value = run.data.metrics["performance_metrics/Test/f1_score/no_anomaly"]
        # print("Metric 1: ", metric_1_value)
        # print("Metric 2: ", metric_2_value)
    except KeyError:
        print("Key does not exist!")


def delete_wrong_log_in_run(log_file_path):
    """ Find and delete log file of specific object in mlflow run.
        Note: Mlflow does not encourage changing log files of a completed run, because the idea is to log a trial
        and learn from mistakes in optimization and dataset choices and run new trials afterwards.
        But this function is to delete redundant data or wrongly computed, or logged objects/data/metrics/parameters.
        The general approach would be to delete a wrong experiment, correct the logging and repeat the experiment.
        This can become very costly if the training process is elaborate.
        It is also bad practice when the learning process is even unaffected by the incorrect logging.
        Example use cases are:
            - Falsely computed metric.
            - Wrong logging path for metric or parameter.
            - Same metric was added and logged multiple times. (Stacking)
              Each logging process would be stored in the same log file without replacement.
              This becomes visible when the csv file of that metric history is downloaded from the UI.
    """
    try:
        os.remove(log_file_path)
        print("File deleted successfully:", log_file_path)
    except OSError as e:
        print(f"Error deleting file: {log_file_path} - {e}")


def delete_wrong_logs_in_experiment(experiment_name, mlruns_path, object_name, metric_name):
    """ Deletes a specific objects log file in each run of the given experiment.
        The absolute path of the specific log file is created for each run_id.
        mlruns_path is the directory where mlruns are stored locally by mlflow.
        object_name is either 'metrics', 'artifacts', 'params' or 'tags'.
        metric_name is the full term of a metric with all subdirectories, also shown in the UI.
        Instead of 'f1_score' it would be 'performance_metrics/Test/f1_score/no_anomaly'.
    """
    mlflow.set_experiment(experiment_name)
    try:
        EXPERIMENT = mlflow.get_experiment_by_name(experiment_name)
        EXPERIMENT_ID = EXPERIMENT.experiment_id
    except AttributeError:
        print("Experiment with name ", experiment_name, "does not exist!")
        return
    print("Experiment_ID", EXPERIMENT_ID)
    runs_df = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID])  # Returns a DataFrame with all run_ids and properties
    run_id_list = runs_df["run_id"].tolist()  # Format column to list

    for run_id in run_id_list:
        log_file_path = os.path.join(*[mlruns_path, EXPERIMENT_ID, run_id, object_name, metric_name])
        delete_wrong_log_in_run(log_file_path)
