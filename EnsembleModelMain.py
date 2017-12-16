#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2017 University of Westminster. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" An example for running the ensemble model
"""

import os
import subprocess
import sys
import numpy as np
from EnsembleModel.EnsembleModel import EnsembleModel


def main():
  # ##########################################################################
  # Read and set the sub-models
  # ##########################################################################
  # Read the response variable (y)
  y = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

  # Read the predicted probabilities (y_hat) for each sub-model
  y_hat_submodel_main = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  y_hat_submodel_1 = [0.2, 0.7, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                      1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  y_hat_submodel_2 = [0.3, 0.8, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0,
                      0.9, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.3, 0.2, 0.1]
  y_hat_submodel_3 = [0.4, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 1.0,
                      0.8, 0.7, 0.8, 0.5, 0.6, 0.3, 0.4, 0.3, 0.2, 0.1]
  y_hat_submodel_4 = [0.5, 1.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 1.0,
                      0.7, 0.6, 0.8, 0.4, 0.6, 0.2, 0.4, 0.3, 0.2, 0.1]

  # Set the sub-model names
  submodels_names = ["submodel_main", "submodel_1", "submodel_2", "submodel_3", "submodel_4"]

  # Set the  the response variable
  y = np.array(y)

  # Set predicted probabilities for sub-models
  y_hat_submodels = np.array([y_hat_submodel_main, y_hat_submodel_1, y_hat_submodel_2,
                              y_hat_submodel_3, y_hat_submodel_4])


  # ##########################################################################
  # Initialise the ensemble model
  # ##########################################################################
  ensemble_model = EnsembleModel(weight_sum_min=2,
                                 weight_sum_max=60,
                                 weight_max=59,
                                 trials_max=15,
                                 iteration_max=100,
                                 alpha_ensemble_min=0.005,
                                 alpha_model_min=0.5,
                                 ensemble_func_acc_weight=0.25,
                                 ensemble_func_auc_weight=0.25,
                                 ensemble_func_rmse_weight=0.25,
                                 ensemble_func_sar_weight=0.25,
                                 ensemble_type="mean",
                                 output_path=os.path.abspath("EnsembleModel/Outputs"),
                                 output_name="ensemble_model")


  # ##########################################################################
  # Generate ensemble models
  # ##########################################################################
  # Run the ensemble model optimiser
  ensemble_model.run(y=y,
                     y_hat_submodels=y_hat_submodels,
                     submodels_names=submodels_names,
                     submodel_main="submodel_main")


  # ##########################################################################
  # Examine the output of the ensemble modeller
  # ##########################################################################
  # Print the path of the output file
  print("Output file path: ", ensemble_model.log_output_path_full)

  # Open the output file
  if sys.platform.startswith('darwin'):
      subprocess.call(('open', ensemble_model.log_output_path_full))
  elif os.name == 'nt':
      os.startfile(ensemble_model.log_output_path_full)
  elif os.name == 'posix':
      subprocess.call(('xdg-open', ensemble_model.log_output_path_full))


  # Generate a specific ensemble model
  # Examine an ensemble model
  submodels_names_func = {"submodel_main": 12, "submodel_1": 15, "submodel_2": 6,
                          "submodel_3": 11, "submodel_4": 19}

  # Generate the predicted probabilities for an ensemble model
  y_hat_ensemble_1 = ensemble_model.ensemble_y_hat(y_hat_submodels=y_hat_submodels,
                                                   submodels_names=submodels_names,
                                                   submodels_names_func=submodels_names_func,
                                                   ensemble_type="mean")

  # Generate statistics for a model
  output_stats = ensemble_model.stat_report_full(y=y,
                                                 y_hat=y_hat_ensemble_1,
                                                 cut_off=0.5)
  print("Examine an ensemble model: \n" + str(submodels_names_func))
  print(output_stats)


if __name__ == '__main__':
  main()