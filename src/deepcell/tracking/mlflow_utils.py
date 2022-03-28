from typing import Optional

import mlflow
from mlflow.entities import ViewType

from deepcell.metrics import Metrics


class MLFlowTrackableMixin:
    """Handles MLFlow tracking"""
    def __init__(self, server_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        """
        @param server_uri: Optional MLFlow server URI. If not provided,
            sets _is_mlflow_tracking_enabled to False
        @param experiment_name: experiment name to use for tracking. If not
            provided, uses default experiment
        """
        self._mlflow_server_uri = server_uri
        self._mlflow_experiment_name = experiment_name
        self._is_mlflow_tracking_enabled = server_uri is not None
        if self._is_mlflow_tracking_enabled:
            mlflow.set_tracking_uri(server_uri)
            self._experiment_id = mlflow.get_experiment_by_name(
                name=experiment_name).experiment_id
            self._resume_active_parent_mlflow_run()

    def _create_parent_mlflow_run(self, run_name: str,
                                  sagemaker_job_name: Optional[str] = None):
        """
        Creates a parent MLFlow run under self._experiment_id
        @param run_name: Run name
        @param sagemaker_job_name:  An optional sagemaker job name to tag, in
            order to link to the sagemaker job if running on sagemaker
        @return:
        """
        tags = {
            'is_parent': 'true'
        }
        if sagemaker_job_name is not None:
            tags['sagemaker_job_name'] = sagemaker_job_name
        mlflow.start_run(experiment_id=self._experiment_id,
                         run_name=run_name,
                         tags=tags)

    def _resume_active_parent_mlflow_run(self) -> None:
        """
        If there is an active parent run, resume it. Needed if
        nested run is started on a
        different node (distributed training)
        @return: None
        """
        active_runs = \
            mlflow.list_run_infos(experiment_id=self._experiment_id,
                                  run_view_type=ViewType.ACTIVE_ONLY)
        active_runs = [x.run_id for x in active_runs]
        active_runs = [mlflow.get_run(run_id) for run_id in active_runs]
        parent_runs = [x for x in active_runs if
                       x.data.tags.get('is_parent', False) == 'true']
        if len(parent_runs) > 0:
            if len(parent_runs) > 1:
                raise RuntimeError('There are multiple active parent runs')
            parent_run = parent_runs[0]
            mlflow.start_run(run_id=parent_run.info.run_id)

    def _create_nested_mlflow_run(self, run_name: str,
                                  sagemaker_job_name: Optional[str] = None):
        """
        Creates an MLFlow run nested under the current parent
        @param run_name: Run name
        @param sagemaker_job_name:  An optional sagemaker job name to tag, in
            order to link to the sagemaker job if running on sagemaker
        @return:
        """
        tags = {}
        if sagemaker_job_name is not None:
            tags['sagemaker_job'] = sagemaker_job_name
        mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            nested=True,
            tags=tags
        )

    @staticmethod
    def _end_mlflow_run():
        mlflow.end_run()

    @staticmethod
    def _log_metrics_to_mlflow(train_metrics: Metrics,
                               val_metrics: Metrics,
                               epoch: int):
        """
        Logs metrics
        """
        mlflow.log_metric(key='train_f1', value=train_metrics.F1,
                          step=epoch)
        mlflow.log_metric(key='val_f1', value=val_metrics.F1,
                          step=epoch)
        mlflow.log_metric(key='train_loss', value=train_metrics.loss,
                          step=epoch)
        mlflow.log_metric(key='val_loss', value=val_metrics.loss,
                          step=epoch)
