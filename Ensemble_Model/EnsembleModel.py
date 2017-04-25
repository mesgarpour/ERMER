#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2017 The Project Authors. All Rights Reserved.
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
""" 
The Ensemble method is designed to increase overall precision & improve risk predictions of set of sub-models.
"""

import os
import sys
import logging
from Logger import Logger
import numpy as np
from scipy import stats
import copy
from typing import Dict, List, Tuple
from EnsembleStats import EnsembleStats


class EnsembleModel:
    """ 
    This Ensemble method was first designed to create an ensemble generative risk model of hospital emergency
    readmission sub-models. The developed Ensemble Risk Model of Emergency Admissions (ERMER) is presented in the 
    following paper: 
      
      http://dx.doi.org/10.1016/j.ijmedinf.2017.04.010
    
    The applied Ensemble algorithm (Algorithm 1) uses a bidirectional hill-climbing algorithm with a greedy initial
    solution set to generate an optimised Ensemble model from the sub-models. Firstly, it generates an initial solution
    based on the main model, & one other sub-model with the highest Area Under the Curve (AUC) of the Receiver 
    Operating Characteristic (ROC). Then, a bidirectional hill-climbing heuristic was applied to optimise the average 
    of the four performance metrics, through iterations, trials(trials) & across samples (samples). 
    
    The hill-climbing method is a greedy sequential search with forward & backward passes, where the learning rate for 
    each performance metric can be tuned manually prior to the execution. The learning rate in the algorithm defined 
    using "alpha_ensemble_min" for the performance indicators.  
    
    The sub-models in the Ensemble heuristic are selected using a Bagging Ensemble (selection with replacement). Then, 
    the sub-models are combined using a mean combiner, which is the approximate posterior probability based on the 
    weighted average of the  risk scores, without any additional training. When the first run of the algorithm with 
    less sensitive limits & thresholds, is executed using the best solutions of the first round.
    
    Finally, the best performing Ensemble model, with the minimum number of unique sub-models is selected.
    
    The cost function for the optimisation was defined as a normalised combination of four performance metrics:
      * ACC (Accuracy): The fraction of the sum of True Positives (TP) & True Negatives (TN) from the total
      * AUC of ROC: The Area Under the Curve of ROC curve, which y-axis is the True Positive Rate (TPR) & x-axis is 
        the False Positive Rate (FPR).
      * RMSE (Root Mean Square Error): Root of averaged of squared difference of the predicted outcomes from the actual 
        outcomes.
      * SAR (Squared Error, Accuracy, & ROC Area): a robust metric for performance: (ACC + ROC + (1 − RMS)) / 3
      
    """

    def __init__(self,
                 weight_sum_min: int=2,
                 weight_sum_max: int=20,
                 weight_max: int=15,
                 trials_max: int=150,
                 iteration_max: int=40,
                 alpha_ensemble_min: float=0.0005,
                 alpha_model_min: float=0.5,
                 ensemble_func_acc_weight=0.25,
                 ensemble_func_auc_weight=0.25,
                 ensemble_func_rmse_weight=0.25,
                 ensemble_func_sar_weight=0.25,
                 ensemble_type: str="mean",
                 output_path: str="",
                 output_name: str="ensemble_model"):
        """ Initialise the Ensembel model settings
        :param weight_sum_min: The minimum number of unique sub-models in the ensemble model
        :param weight_sum_max: The maximum number of sub-models (i.e. sum of weights) in the ensemble model 
        :param weight_max: The maximum weight a sub-model can take
        :param trials_max: The maximum number of optimisation trials (i.e. number of ensemble model generated)
        :param iteration_max: The maximum number of optimisation iteration 
        :param alpha_ensemble_min: The minimum threshold of the cost function improvement for performance indicator 
        (ACC, AUC, RMSE or SAR), a.k.a learning-rate.
        :param alpha_model_min: The threshold values on the prediction values, a.k.a cut-off point.
        :param ensemble_func_acc_weight: The wight of the ACC in the cost function
        :param ensemble_func_auc_weight: The wight of the AUC in the cost function
        :param ensemble_func_rmse_weight: The wight of the RMSE in the cost function
        :param ensemble_func_sar_weight: The wight of the SAR in the cost function
        :param ensemble_type: The ensemble combination type: 'mean', 'median', or 'max'
        :param output_path: The full-path to the output location (default is the local folder)
        :param output_name: The name of the log output and the generated ensemble models
        """
        self.__weight_sum_min = int(weight_sum_min)
        self.__weight_sum_max = int(weight_sum_max)
        self.__weight_max = int(weight_max)
        self.__trials_max = int(trials_max)
        self.__iteration_max = int(iteration_max)
        self.__alpha_ensemble_min = float(alpha_ensemble_min)
        self.__alpha_model_min = float(alpha_model_min)
        self.__ensemble_func_acc_weight = ensemble_func_acc_weight
        self.__ensemble_func_auc_weight = ensemble_func_auc_weight
        self.__ensemble_func_rmse_weight = ensemble_func_rmse_weight
        self.__ensemble_func_sar_weight = ensemble_func_sar_weight
        self.__ensemble_type = str(ensemble_type)
        self.__output_name = str(output_name)
        self.__output_path = str(output_path)

        self.__logger_main = Logger(app_name=output_name, path=output_path, extension="log")
        self.__logger = None
        self.__stats = EnsembleStats()
        self.log_output_path_full = os.path.join(output_path, output_name + ".log")

    def run(self,
            y: np.ndarray,
            y_hat_submodels: np.ndarray,
            submodels_names: List[str],
            submodel_main: str,
            submodels_init_solution: Dict[str, int]=None):
        """ Run the Ensemble model
        :param y: The response variable
        :param y_hat_submodels: The predicted probabilities for each sub-models
        :param submodels_names: The sub-model names
        :param submodel_main: The main sub-model (no condition on sub-population)
        :param submodels_init_solution: The initial solution
        """
        submodels_selected_trials = dict()
        y_hat_submodels_selected_trials = dict()

        # Init logger
        self.__logger = logging.getLogger(self.__output_name)

        # Produce summary of the settings
        self.__logger.info("Start")
        self.__logger.debug(
            "Settings:" + " weight_sum_min:" + str(self.__weight_sum_min) +
            "; weight_sum_max:" + str(self.__weight_sum_max) +
            "; weight_max:" + str(self.__weight_max) +
            "; trials_max:" + str(self.__trials_max) +
            "; iteration_max:" + str(self.__iteration_max) +
            "; ensemble_type:" + str(self.__ensemble_type) +
            "; alpha_ensemble_min:" + str(self.__alpha_ensemble_min) +
            "; alpha_model_min:" + str(self.__alpha_model_min))

        # validate
        self.__input_validations(y, y_hat_submodels, submodels_names, submodel_main, submodels_init_solution)

        # conversions
        y = np.array(y)
        y_hat_submodels = np.array(y_hat_submodels)

        # Generate a greedy solution if NO initial solution is provided
        if submodels_init_solution is None:
            submodels_init_solution = self.__ensemble_init(y, y_hat_submodels, submodels_names, submodel_main)
        else:
            submodels_init_solution = [k for k, v in submodels_init_solution.items() for _ in range(v)]

        # Optimise the ensemble
        # Run trials of the optimiser
        for trial in range(self.__trials_max):
            submodels_selected_trials[trial], y_hat_submodels_selected_trials[trial] = \
                self.__ensemble_optimise(y, y_hat_submodels, submodels_names, submodels_init_solution, trial)

        # Print the summary statistics
        self.__logger.info("Initial Solution: \n" + str(stats.itemfreq(submodels_init_solution)))
        self.__logger.info("Performance of the inputted sub-models:")
        for i in range(len(submodels_names)):
            stat = self.stat_report_full(y, y_hat_submodels[i], self.__alpha_model_min)
            self.__logger.info(str(i + 1) + "," + submodels_names[i] + "," + stat)

        self.__logger.info("Performance of the generated ensemble models:")
        for trial in submodels_selected_trials.keys():
            item_freq = stats.itemfreq(submodels_selected_trials[trial])
            ensemble_model = ' + '.join([str(r[1]) + " * " + r[0] for r in item_freq])
            stat = self.stat_report_full(y, y_hat_submodels_selected_trials[trial], self.__alpha_model_min)
            self.__logger.info(str(trial + 1) + "," + ensemble_model + "," + stat)

        # Terminate
        self.__logger_main.terminate()
        self.__logger.info("End")

    def ensemble_y_hat(self,
                       y_hat_submodels: np.ndarray,
                       submodels_names: List[str],
                       submodels_names_func: Dict[str, int],
                       ensemble_type: str = "mean"
                       ) -> np.ndarray:
        """ Generate the probabilities for the inputted ensemble model 
        :param y_hat_submodels: The predicted probabilities for each sub-models
        :param submodels_names: The sub-model names
        :param submodels_names_func: a dictionary of submodels and weights, which represent an ensemble model
        :param ensemble_type: The ensemble combination type: 'mean', 'median', or 'max'
        :return: The predicted probabilities using the ensemble model
        """
        submodels_selected = [k for k, v in submodels_names_func.items() for _ in range(v)]
        y_hat_submodels_key = dict(zip(submodels_names, range(len(submodels_names))))
        y_hat_selected = self.__ensemble_select_submodel_y_hat(y_hat_submodels, y_hat_submodels_key, submodels_selected)
        return self.__ensemble_combiner(y_hat_selected, ensemble_type)

    def stat_report_full(self,
                         y: np.ndarray,
                         y_hat: np.ndarray,
                         cut_off: float
                         ) -> str:
        """ Produce a performance statistic report
        :param y: The response variable
        :param y_hat: The predicted probabilities 
        :param cut_off: The threshold values on the prediction values, a.k.a cut-off point.
        :return: A summary of performance statistics
        """
        performance = self.__stat_performance_full(y, y_hat, cut_off)
        return "acc:" + str(performance["acc"]) + "; auc:" + str(performance["auc"]) + \
               "; rmse:" + str(performance["rmse"]) + "; sar:" + str(performance["sar"]) + \
               "; sensitivity:" + str(performance["sensitivity"]) + "; precision:" + str(performance["precision"]) + \
               "; specificity:" + str(performance["specificity"])

    def __ensemble_optimise(self,
                            y: np.ndarray,
                            y_hat_submodels: np.ndarray,
                            submodels_names: List[str],
                            submodels_init_solution: List[str],
                            trial: int
                            ) -> Tuple[List[str], np.ndarray]:
        """Apply the search algorithm of the ensemble optimiser"""
        y_hat_submodels_key = dict(zip(submodels_names,range(len(submodels_names))))
        submodels_selected = submodels_init_solution
        y_hat_submodels_selected = []
        flag_backward_complete = True
        flag_backward_failed = False
        submodel_backward_index = self.__weight_sum_min
        submodel_backward_removed = ""
        submodel_forward = ""
        iteration = 0
        y_hat_1 = np.logical_not(y).astype(int)
        y_hat_2 = copy.deepcopy(y_hat_1)

        # Run search iteration of the optimiser
        while len(submodels_selected) < self.__weight_sum_max \
                and iteration < self.__iteration_max:
            self.__logger.info("Trial:" + str(trial) + "; Iteration:" + str(iteration))
            iteration += 1

            # Performance of the current ensemble of sub-models
            y_hat_2 = self.__ensemble_select_submodel_y_hat(y_hat_submodels, y_hat_submodels_key, submodels_selected)
            y_hat_2 = self.__ensemble_combiner(y_hat_2, self.__ensemble_type)
            avg_improvement, avg_degradation, performance_2, performance_1 = \
                self.__ensemble_compare(y, y_hat_1, y_hat_2)
            self.__logger.debug("Avg. improvement:" + str(avg_improvement) +
                                "; Avg. degradation:" + str(avg_degradation) +
                                "; " + self.__ensemble_report(performance_2))

            # Configure step-wise algorithm
            y_hat_1, submodels_selected, flag_backward_failed, flag_backward_complete = \
                self.__ensemble_stepwise_move(submodels_selected, y_hat_1, y_hat_2, avg_improvement, avg_degradation,
                                              flag_backward_failed, flag_backward_complete)

            # Apply the backward-elimination
            if flag_backward_complete is False:
                flag_backward_complete, flag_backward_failed, submodel_backward_index, submodel_backward_removed = \
                    self.__ensemble_backward_elimination(
                        submodels_selected, flag_backward_complete, flag_backward_failed,
                        submodel_backward_index, submodel_backward_removed)

            # Apply the forward-selection
            if flag_backward_complete is True:
                submodels_selected, submodel_forward = self.__ensemble_forward_selection(
                    submodels_selected, submodels_names)
                if submodel_forward == "":
                    return submodels_selected, y_hat_2
        return submodels_selected, y_hat_2

    def __ensemble_stepwise_move(self,
                                 submodels_selected: List[str],
                                 y_hat_1: np.ndarray, y_hat_2: np.ndarray,
                                 avg_improvement: float,
                                 avg_degradation: float,
                                 flag_backward_failed: bool,
                                 flag_backward_complete: bool
                                 ) -> Tuple[np.ndarray, List[str], bool, bool]:
        """Assess the previous step-wise move and update variables"""
        improvement_cutoff = 0.5  # the cut-off depends on the defined cost function

        # NO improvement in forward-selection
        if flag_backward_complete is True \
                and avg_improvement < improvement_cutoff:
            self.__logger.debug("NO improvement in forward-selection")
            submodels_selected = submodels_selected[:-1]
        # NO improvement in backward-elimination
        elif flag_backward_complete is False \
                and (avg_improvement < improvement_cutoff
                     or avg_degradation >= improvement_cutoff):
            self.__logger.debug("NO improvement in backward-elimination")
            flag_backward_failed = True
        # Some improvement in a step-wise move
        elif avg_improvement >= 0.5 \
                or (flag_backward_complete is False
                    and avg_degradation < improvement_cutoff):
            self.__logger.debug("Some improvement in a step-wise move")
            flag_backward_complete = False
            # update the prediction for ensemble models
            y_hat_1 = copy.deepcopy(y_hat_2)
        return y_hat_1, submodels_selected, flag_backward_failed, flag_backward_complete

    def __ensemble_backward_elimination(self,
                                        submodels_selected: List[str],
                                        flag_backward_complete: bool,
                                        flag_backward_failed: bool,
                                        submodel_backward_index: int,
                                        submodel_backward_removed: str
                                        ) -> Tuple[bool, bool, int, str]:
        """Calculate the backward elimination move"""
        # If previous backward-elimination failed, then retry
        if flag_backward_failed is True:
            # Snap-back the removed model
            if submodel_backward_index == len(submodels_selected) + 1:
                submodels_selected = submodels_selected + [submodel_backward_removed]
            else:
                submodels_selected = submodels_selected[0:submodel_backward_index - 1] + \
                                     [submodel_backward_removed, submodels_selected[submodel_backward_index:]]

            # Select next model
            submodel_backward_index = submodel_backward_index + 1
            flag_backward_failed = False

        # Check the backward-elimination validity, then update variables
        if len(submodels_selected) >= self.__weight_sum_min \
                and submodel_backward_index < len(submodels_selected):
            submodel_backward_removed = submodels_selected[submodel_backward_index]
            del submodels_selected[submodel_backward_index]
            self.__logger.debug("Backward Elimination: " + submodel_backward_removed)
        else:
            flag_backward_complete = True
            flag_backward_failed = False
            submodel_backward_index = self.__weight_sum_min
            submodel_backward_removed = ""
        return flag_backward_complete, flag_backward_failed, submodel_backward_index, submodel_backward_removed

    def __ensemble_forward_selection(self,
                                     submodels_selected: List[str],
                                     submodels_names: List[str]):
        """Calculate the forward selection move"""
        # Select sub-models randomly
        submodel_forward = ""
        submodels_forward = np.random.choice(submodels_names, len(submodels_names))

        # Get frequency of sub-models weights
        item_freq = stats.itemfreq(submodels_selected)
        item_freq = dict(zip([item_freq[i][0] for i in range(item_freq.shape[0])],
                             [int(item_freq[i][1]) for i in range(item_freq.shape[0])]))

        # Select a sub-model, which its weight does not exceed the limit
        for submodel in submodels_forward:
            if submodel in item_freq.keys() \
                    and item_freq[submodel] > self.__weight_max:
                continue
            else:
                submodel_forward = submodel
                submodels_selected = submodels_selected + [submodel_forward]
                self.__logger.debug("Forward Selection: " + submodel_forward)
        return submodels_selected, submodel_forward

    def __ensemble_select_submodel_y_hat(self,
                                         y_hat_submodels: np.ndarray,
                                         y_hat_submodels_key: Dict[str, int],
                                         submodels_selected: List[str]
                                         ) -> np.ndarray:
        """Select a sub-model y_hat, by its name"""
        return np.array([y_hat_submodels[i] for i in
                         [y_hat_submodels_key[k] for k in submodels_selected]])

    def __ensemble_init(self,
                        y: np.ndarray,
                        y_hat_submodels: np.ndarray,
                        submodels_names: List[str],
                        submodel_main: str):
        """Generate an initial solution for the ensemble optimiser using a predefined greedy approach"""
        # Calculate sub-models' performances
        performances = []
        for i in range(len(submodels_names)):
            performance = self.__stat_performance(y, y_hat_submodels[i], self.__alpha_model_min)
            performance.update({"submodel_name": submodels_names[i]})
            performances.append(performance)

        # Sort sub-models based on the AUC of ROC (descending)
        performances = sorted(performances, key=lambda x: x['auc'], reverse=True)

        # select models based on a predefined greedy
        submodels_selected = [submodel_main, performances[0]["submodel_name"]]
        return submodels_selected

    def __ensemble_combiner(self,
                            y_hat_submodels_selected: np.ndarray,
                            ensemble_type: str):
        """Use an ensemble 'combiner' function, in order to calculate the overall predictions of sub-models"""
        if ensemble_type == "mean":
            return np.mean(y_hat_submodels_selected, axis=0)
        elif ensemble_type == "max":
            return np.max(y_hat_submodels_selected, axis=0)
        elif ensemble_type == "median":
            return np.median(y_hat_submodels_selected, axis=0)
        else:
            self.__logger.error("Invalid ensemble combiner: " + ensemble_type)
            sys.exit()

    def __ensemble_compare(self,
                           y: np.ndarray,
                           y_hat_1: np.ndarray,
                           y_hat_2: np.ndarray
                           ) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
        """Compare two ensemble models, based on a predefined function"""
        # Calculate performance statistics
        performance_1 = self.__stat_performance(y, y_hat_1, self.__alpha_model_min)
        performance_2 = self.__stat_performance(y, y_hat_2, self.__alpha_model_min)

        # Calculate the differences in performances
        delta_acc = performance_2["acc"] - performance_1["acc"]
        delta_auc = performance_2["auc"] - performance_1["auc"]
        delta_rmse = performance_2["rmse"] - performance_1["rmse"]
        delta_sar = performance_2["sar"] - performance_1["sar"]

        # Calculate average improvement and degradation in performance
        avg_improvement = ((int(delta_acc >= self.__alpha_ensemble_min) * self.__ensemble_func_acc_weight) +
                           (int(delta_auc >= self.__alpha_ensemble_min) * self.__ensemble_func_auc_weight) +
                           (int(delta_rmse >= self.__alpha_ensemble_min) * self.__ensemble_func_rmse_weight) +
                           (int(delta_sar >= self.__alpha_ensemble_min) * self.__ensemble_func_sar_weight))
        avg_degradation = ((int(delta_acc < -1 * self.__alpha_ensemble_min) * self.__ensemble_func_acc_weight) +
                           (int(delta_auc < -1 * self.__alpha_ensemble_min) * self.__ensemble_func_auc_weight) +
                           (int(delta_rmse < -1 * self.__alpha_ensemble_min) * self.__ensemble_func_rmse_weight) +
                           (int(delta_sar < -1 * self.__alpha_ensemble_min) * self.__ensemble_func_sar_weight))
        return avg_improvement, avg_degradation, performance_2, performance_1

    def __ensemble_report(self,
                          performance: Dict[str, float]
                          ) -> str:
        """Produce a basic report about the current iteration of the ensemble optimiser"""
        return "acc:" + str(performance["acc"]) + "; auc:" + str(performance["auc"]) + \
               "; rmse:" + str(performance["rmse"]) + "; sar:" + str(performance["sar"])

    def __stat_performance(self,
                           y: np.ndarray,
                           y_hat: np.ndarray,
                           cut_off: float
                           ) -> Dict[str, float]:
        """Generate the basic performance metrics, which is used by the ensemble optimiser's compare function"""
        performance = self.__stats.confusion_matrix(y, y_hat, cut_off)
        performance["acc"] = self.__stats.acc(
            performance["tp"], performance["tn"], performance["fp"], performance["fn"])
        performance["auc"] = self.__stats.auc(y, y_hat)
        performance["rmse"] = self.__stats.rmse(y, y_hat)
        performance["sar"] = self.__stats.sar(performance["acc"], performance["auc"], performance["rmse"])
        return performance

    def __stat_performance_full(self,
                                y: np.ndarray,
                                y_hat: np.ndarray,
                                cut_off: float
                                ) -> Dict[str, float]:
        """Generate a full performance statistics for reporting purpose"""
        performance = self.__stat_performance(y, y_hat, cut_off)
        performance["sensitivity"] = self.__stats.sensitivity(performance["tp"], performance["fn"])
        performance["precision"] = self.__stats.precision(performance["tp"], performance["fp"])
        performance["specificity"] = self.__stats.specificity(performance["tn"], performance["fp"])
        return performance

    def __input_validations(self,
                            y: np.ndarray,
                            y_hat_submodels: np.ndarray,
                            submodels_names: List[str],
                            submodel_main: str,
                            submodels_init_solution: Dict[str, int]):
        """Validate the inputs"""
        # Number of observations and probabilities must be consistent
        if y is None or y_hat_submodels is None \
                or len(y) != len(y_hat_submodels[0]) \
                or len(y) == 0:
            self.__logger.error("Invalid number of observation values or probabilities!")
            sys.exit()

        # Number & names of sub-models must mbe consistent
        if submodels_names is None \
                or len(y_hat_submodels) != len(submodels_names) \
                or len(submodels_names) <= 1 \
                or submodel_main not in set(submodels_names):
            self.__logger.error("Invalid sub-models!")
            sys.exit()

        # Observations values must be valid (must include zeros & ones)
        y_uniques = set(np.unique(y))
        if 1 not in y_uniques \
                or 0 not in y_uniques \
                or len(y_uniques) != 2:
            self.__logger.error("Invalid observation values (must include zeros & ones)!")
            sys.exit()

        # Probabilities values must be valid (must be between zero & one)
        for submodel in range(len(y_hat_submodels)):
            if np.any(y_hat_submodels[submodel] < 0) \
                    or np.any(y_hat_submodels[submodel] > 1):
                self.__logger.error("Invalid observation values (must be between zero & one)!")
                sys.exit()

        # ensemble combiner type must be either: max, mean or median
        if self.__ensemble_type not in {"mean", "median", "max"}:
            self.__logger.error("Invalid ensemble combiner function!")
            sys.exit()

        # The target parameter's values must be zero and one: y_hat in {0, 1}
        if self.__weight_sum_min <= 0 \
                or self.__weight_sum_max <= 0 \
                or self.__weight_sum_max < self.__weight_sum_min \
                or self.__weight_max <= 0 \
                or self.__trials_max <= 0 \
                or self.__iteration_max <= 0 \
                or self.__alpha_ensemble_min <= 0 \
                or self.__alpha_ensemble_min >= 1 \
                or self.__alpha_model_min <= 0 \
                or self.__alpha_model_min > 1:
            self.__logger.error("Invalid initialisation setting(s)!")
            sys.exit()

        # The defined initial solution must be valid
        if submodels_init_solution is not None \
                and (sum(submodels_init_solution.values()) < self.__weight_sum_min
                     or sum(submodels_init_solution.values()) > self.__weight_sum_max
                     or any([False for submodel in submodels_init_solution.keys()
                             if submodel not in set(submodels_names)])):
            self.__logger.error("Invalid initial solution!")
            sys.exit()
