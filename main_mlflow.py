import os
import sys

sys.path.append(os.path.dirname(__file__))
from Parameters import UserInfo
from otherFuncs.smallFuncs import terminalEntries
from full_multi_planar_framework import simulate
import mlflow


mlflow.create_experiment(name='/experiment_testing_samples')  # , artifact_location='dbfs:/artifacts_stuff'
mlflow.set_experiment(experiment_name='/experiment_testing_samples', )

mlflow.keras.autolog()

if __name__ == '__main__':

    # mlflow.set_tracking_uri("databricks")

    # mlflow.keras.log_model(model, "model")
    UserEntry = terminalEntries(UserInfo.__dict__)
    UserEntry['simulation'] = UserEntry['simulation']()
    simulate(UserEntry)
