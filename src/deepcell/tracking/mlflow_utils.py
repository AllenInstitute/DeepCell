import json
from collections import defaultdict
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import requests
from mlflow.entities import ViewType, RunStatus

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
        self._parent_run: Optional[mlflow.entities.Run] = None
        if self._is_mlflow_tracking_enabled:
            mlflow.set_tracking_uri(server_uri)
            self._experiment_id = mlflow.get_experiment_by_name(
                name=experiment_name).experiment_id

    def _create_parent_mlflow_run(
            self, run_name: str,
            hyperparameters: dict,
            hyperparameters_exclude_keys: Optional[list] = None,
            sagemaker_job_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Creates a parent MLFlow run under self._experiment_id
        @param run_name: Run name
        @param hyperparameters: Hyperparameters to log
        @param hyperparameters_exclude_keys: Exclude these keys from logging
        @param sagemaker_job_name:  An optional sagemaker job name to tag, in
            order to link to the sagemaker job if running on sagemaker
        @return: mlflow.ActiveRun
        """

        def construct_hyperparams_for_logging(
                hyperparams: dict,
                hyperparams_exclude_keys: Optional[list] = None) -> dict:
            """
            Flattens nested dict and removes keys which should not be logged
            @param hyperparams:
            @param hyperparams_exclude_keys:
            @return:
            """
            if hyperparams_exclude_keys is None:
                hyperparams_exclude_keys = []
            hyperparams = {k: v for k, v in hyperparams.items()
                           if k not in hyperparams_exclude_keys}
            # flatten
            hyperparams = pd.json_normalize(hyperparams, sep='_')
            hyperparams = hyperparams.to_dict(orient='records')[0]

            hyperparams = {k: v for k, v in hyperparams.items()
                           if 'log_level' not in k}
            return hyperparams

        tags = {
            'is_parent': 'true'
        }
        if sagemaker_job_name is not None:
            tags['sagemaker_job_name'] = sagemaker_job_name
        run = mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            tags=tags)
        hyperparameters = construct_hyperparams_for_logging(
            hyperparams=hyperparameters,
            hyperparams_exclude_keys=hyperparameters_exclude_keys)
        mlflow.log_params(params=hyperparameters)
        self._parent_run = run
        return run

    @staticmethod
    def _resume_parent_mlflow_run(run_id: str) -> mlflow.ActiveRun:
        """
        If there is an active parent run, resume it. Needed if
        nested run is started on a
        different node (distributed training)
        @return: ActiveRun
        """
        parent_run = mlflow.start_run(run_id=run_id)
        return parent_run

    def _create_nested_mlflow_run(
            self, run_name: str,
            sagemaker_job_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Creates an MLFlow run nested under the current parent
        @param run_name: Run name
        @param sagemaker_job_name:  An optional sagemaker job name to tag, in
            order to link to the sagemaker job if running on sagemaker
        @return: ActiveRun
        """
        tags = {}
        if sagemaker_job_name is not None:
            tags['sagemaker_job'] = sagemaker_job_name
        return mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            nested=True,
            tags=tags
        )

    @staticmethod
    def _log_epoch_end_metrics_to_mlflow(train_metrics: Metrics,
                                         val_metrics: Metrics,
                                         epoch: int) -> None:
        """
        Logs metrics during training at the end of an epoch
        """
        mlflow.log_metric(key='train_f1', value=train_metrics.F1,
                          step=epoch)
        mlflow.log_metric(key='val_f1', value=val_metrics.F1,
                          step=epoch)
        mlflow.log_metric(key='train_loss', value=train_metrics.loss,
                          step=epoch)
        mlflow.log_metric(key='val_loss', value=val_metrics.loss,
                          step=epoch)

    @staticmethod
    def _log_early_stopping_metrics(best_epoch: int) -> None:
        """
        Logs metrics when early stopping is triggered
        @param best_epoch: The best epoch, as identified by
            `deepcell.callbacks.EarlyStopping`
        @return: None
        """
        mlflow.set_tag(key='best_epoch', value=best_epoch)

    def _log_cross_validation_end_metrics_to_mlflow(self) -> None:
        """
        Logs average cross validation metrics at the end of training
        @return: None
        """
        if not self._is_mlflow_tracking_enabled:
            return
        query = f"tags.mlflow.parentRunId = '{self._parent_run.info.run_id}'"
        results = mlflow.search_runs(filter_string=query,
                                     experiment_ids=[self._experiment_id])
        child_run_ids = results['run_id']
        cv_metrics = defaultdict(list)
        for run_id in child_run_ids:
            run = mlflow.get_run(run_id=run_id)
            best_epoch = int(run.data.tags['best_epoch'])
            for metric in run.data.metrics:
                res = requests.get(url=f'{self._mlflow_server_uri}/'
                                       f'api/2.0/mlflow/metrics/get-history',
                                   params={
                                       'run_id': run_id,
                                       'metric_key': metric})
                res = json.loads(res.text)

                cv_metrics[metric].append(res['metrics'][best_epoch]['value'])
        avg_metrics = {k: np.array(v).mean() for k, v in cv_metrics.items()}
        mlflow.log_metrics(avg_metrics)
