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
The statistical measures that are required by the EnsembleModel.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict


class EnsembleStats:
    """ The summary statistical measures:
      * ACC (Accuracy)
      * AUC of ROC
      * Precision
      * RMSE (Root Mean Square Error)
      * SAR (Squared Error, Accuracy, & ROC Area)
      * Sensitivity (Recall)
      * Specificity
    """

    @staticmethod
    def confusion_matrix(y: np.ndarray,
                         y_hat: np.ndarray,
                         cut_off: float
                         ) -> Dict[str, float]:
        """Calculate the confusion-matrix"""
        # confusion matrix
        tp = sum(np.logical_and(y, (y_hat >= cut_off)))
        fp = sum(np.logical_and(np.logical_not(y), (y_hat >= cut_off)))
        tn = sum(np.logical_and(np.logical_not(y), (y_hat < cut_off)))
        fn = sum(np.logical_and(y, (y_hat < cut_off)))
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    @staticmethod
    def acc(tp: float,
            tn: float,
            fp: float,
            fn: float
            ) -> float:
        """ACC (Accuracy): The fraction of the sum of True Positives (TP) & True Negatives (TN) from the total"""
        return (tp + tn) / (tp + fp + fn + tn)

    @staticmethod
    def auc(y: np.ndarray,
            y_hat: np.ndarray
            ) -> float:
        """AUC of ROC: The Area Under the Curve of ROC curve, which y-axis is the True Positive Rate (TPR) & x-axis 
        is the False Positive Rate (FPR)."""
        return roc_auc_score(y, y_hat)

    @staticmethod
    def precision(tp: float,
                  fp: float,
                  ) -> float:
        """Precision or positive predictive value (PPV)"""
        return tp / (tp + fp)

    @staticmethod
    def rmse(y: np.ndarray,
             y_hat: np.ndarray
             ) -> float:
        """RMSE (Root Mean Square Error): Root of averaged of squared difference of the predicted outcomes from the 
        actual outcomes."""
        return np.sqrt(np.sum(np.power(np.subtract(y_hat, y), 2)) / len(y_hat))

    @staticmethod
    def sar(acc: float,
            auc: float,
            rmse: float
            ) -> float:
        """SAR (Squared Error, Accuracy, & ROC Area): a robust metric for performance: (ACC + ROC + (1 − RMS)) / 3"""
        return (acc + auc + (1 - rmse)) / 3

    @staticmethod
    def sensitivity(tp: float,
                    fn: float
                    ) -> float:
        """Sensitivity or true positive rate (TPR)"""
        return tp / (tp + fn)

    @staticmethod
    def specificity(tn: float,
                    fp: float,
                    ) -> float:
        """Specificity or true negative rate"""
        return tn / (tn + fp)
