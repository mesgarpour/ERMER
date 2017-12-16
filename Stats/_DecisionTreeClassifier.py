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
""" It is an interface for the 'DecisionTreeClassifier' training model (Decision Tree Classifier).
"""

from typing import Dict, List, Any, TypeVar
from Stats.Stats import Stats
from sklearn import tree
import pydotplus

PandasDataFrame = TypeVar('DataFrame')
SklearnDecisionTreeClassifier = TypeVar('DecisionTreeClassifier')
GraphvizDot = TypeVar('pydotplus.graphviz.Dot')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class _DecisionTreeClassifier(Stats):
    def __init__(self):
        """Initialise the objects and constants.
        """
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.debug("Run Decision Tree Classifier.")

    def train(self,
              features_indep_df: PandasDataFrame,
              feature_target: List,
              model_labals: List=[0, 1],
              **kwargs: Any) -> SklearnDecisionTreeClassifier:
        """Perform the training, using the Decision Tree Classifier.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param model_labals: the target labels (default [0, 1]).
        :param kwargs: criterion='gini', splitter='best', max_depth=None, min_samples_split=30,
        min_samples_leaf=30, min_weight_fraction_leaf=0.0, max_features=None,
        random_state=None, max_leaf_nodes=None, class_weight=None, presort=False
        :return: the trained model.
        """
        self._logger.debug("Train " + __name__)
        model_train = tree.DecisionTreeClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self,
                        model_train: SklearnDecisionTreeClassifier) -> Dict:
        """Produce the training summary.
        :param model_train: the instance of the trained model.
        :return: the training summary.
        """
        self._logger.debug("Summarise " + __name__)
        summaries = dict()
        summaries['classes_'] = model_train.classes_
        summaries['feature_importances_'] = model_train.feature_importances_
        summaries['max_features_'] = model_train.max_features_
        summaries['n_classes_'] = model_train.n_classes_
        summaries['n_features_'] = model_train.n_features_
        summaries['n_outputs_'] = model_train.n_outputs_
        summaries['tree_summaries'] = model_train.tree_
        return summaries

    def plot(self,
             model_train: SklearnDecisionTreeClassifier,
             feature_names: List,
             class_names: List=["True", "False"]) -> GraphvizDot:
        """Plot the tree diagram.
        :param model_train: the instance of the trained model.
        :param feature_names: the names of input features.
        :param class_names: the predicted class labels.
        :return: the model graph.
        """
        self._logger.debug("Plot " + __name__)
        dot_data = tree.export_graphviz(model_train, out_file=None, feature_names=feature_names,
                                        class_names=class_names, filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        return graph
