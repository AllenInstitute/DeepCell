import argschema


class TrackingSchema(argschema.ArgSchema):
    """Training/model tracking"""
    mlflow_server_uri = argschema.fields.String(
        default=None,
        allow_none=True,
        description='MLFlow server URI. If provided, will log to MLFLow '
                    'during training'
    )
    mlflow_experiment_name = argschema.fields.String(
        default='deepcell-train',
        description='If mlflow_server_uri provided, which experiment to use '
                    'for tracking'
    )
    mlflow_run_name = argschema.fields.String(
        default=None,
        allow_none=True,
        description='If mlflow_server_uri provided, run name to use. If None,'
                    'a default value is used'
    )
    sagemaker_job_name = argschema.fields.String(
        default=None,
        allow_none=True,
        description='If running in sagemaker, pass the job name to log in '
                    'MLFlow'
    )
    parent_run_id = argschema.fields.String(
        default=None,
        allow_none=True,
        description='Provide an MLFlow run id to resume it'
    )
