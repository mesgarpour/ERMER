
# coding: utf-8

# # Ensemble Risk Model of Emergency Admissions (ERMER) 

# [1. Initialise](#1.-Initialise)
# <br\>
# [2. Read Data](#2.-Read-Data)
# <br\>
# [3. Prepare Features](#3.-Prepare-Features)
# <br\>
# [4. Build Sub-Models](#4.-Build-Sub-Models)
# <br\>
# [5. Build Ensemble Model](#5.-Build-Ensemble-Model)
# <br\>

# This Jupyter IPython Notebook applies the Ensemble Risk Model of Emergency Admissions (ERMER).
# 
# The ERMER algorithm was developed as part of a PhD research at the <a href="http://www.healthcareanalytics.co.uk/">Health &amp; Social Care Modelling Group (HSCMG)</a> at the <a href="http://www.westminster.ac.uk">University of Westminster</a> (<a href="http://www.westminster.ac.uk">Predictive Risk Modelling of Hospital Emergency Readmission,
# and Temporal Comorbidity Index Modelling Using Machine
# Learning Methods</a>).
# 
# The published journal paper can be accessed via the following link: http://www.sciencedirect.com/science/article/pii/S1386505617300886
# 
# Note: The script blocks in the Notebook must be carefully reviewed and configured before execution on a new problem.

# <hr\>
# <font size="1" color="gray">Copyright 2017, Mohsen Mesgarpour. All Rights Reserved.
# 
# It is licensed under the Apache License, Version 2.0. you may not use this file except in compliance with the License. You may obtain a copy of the License at
# 
#   <a href="http://www.apache.org/licenses/LICENSE-2.0">http://www.apache.org/licenses/LICENSE-2.0</a>
# 
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.</font>
# <hr\>
# 
# ## 1. Initialise

# Reload modules

# In[ ]:


# Reload modules 
# It is an optional step. It can be useful to run when external Python modules are being modified
# It is reloading all modules (except those excluded by %aimport) every time before executing the Python code typed.
# Note: It may conflict with some python functions (like serialisation).

# %load_ext autoreload 
# %autoreload 2


# Import libraries

# In[ ]:


# Import Python libraries
import logging
import os
import sys
import gc
import pandas as pd
import numpy as np
import statistics
import pprint
from IPython.display import display, HTML
from collections import OrderedDict
from scipy.stats import stats


# In[ ]:


# Import local Python modules
from Stats.PreProcess import PreProcess
from Stats.FeatureSelection import FeatureSelection
from Stats.TrainingMethod import TrainingMethod
from Stats.Plots import Plots
from Configs.CONSTANTS import CONSTANTS
from Configs.Logger import Logger
from ReadersWriters.ReadersWriters import ReadersWriters
from EnsembleModel.EnsembleModel import EnsembleModel


# In[ ]:


# Check the interpreter
print("\nMake sure the correct Python interpreter is used!")
print(sys.version)
print("\nMake sure sys.path of the Python interpreter is correct (the ERMER project main folder))!")
print(os.getcwd())


# <br/><br/>

# ### 1.1.  Initialise General Settings

# <font style="font-weight:bold;color:red">Main configuration Settings: </font>
# - Specify the full path of the configuration file 
# <br/>&#9; &#8594; config_path
# - Specify the full path of the output folder 
# <br/>&#9; &#8594; output_path
# - Specify the application name (the suffix of the outputs file name) 
# <br/>&#9; &#8594; app_name
# <br/>
# <br/>
# 
# <font style="font-weight:bold;color:red">External Configration Files: </font>
# - The full path of the feature configuration file:
# <br/>&#9; &#8594; <i>ConfigInputs/input_features_configs</i>
# - TThe full path of the configuration file:
# <br/>&#9; &#8594; <i>ConfigInputs/CONFIGURATIONS</i>
# - The input features' confugration file (Note: only the CSV export of the XLSX will be used by this Notebook):
# <br/>&#9; &#8594; <i>ConfigInputs/input_features_configs.xlsx</i>
# <br/>&#9; &#8594; <i>ConfigInputs/input_features_configs.csv</i>

# In[ ]:


config_features_path = os.path.abspath("ConfigInputs/input_features_configs")
config_path = os.path.abspath("ConfigInputs/CONFIGURATIONS")
io_path = os.path.abspath("Outputs")
app_name = "ERMER"

print("\n The full path of the feature configuration file: \n\t", config_features_path,
      "\n The full path of the configuration file: \n\t", config_path,
      "\n The full path of the input and output folder: \n\t", io_path,
      "\n The application name (the suffix of the outputs file name): \n\t", app_name)


# <br/><br/>

# Initialise logs

# In[ ]:


if not os.path.exists(io_path):
    os.makedirs(io_path, exist_ok=True)

logger = Logger(path=io_path, app_name=app_name, ext="log")
logger = logging.getLogger(app_name)


# Initialise constants and some of classes

# In[ ]:


# Initialise constants        
CONSTANTS.set(io_path, app_name)


# In[ ]:


# Initialise other classes
readers_writers = ReadersWriters()
preprocess = PreProcess(io_path)


# In[ ]:


# Set print settings
pd.set_option('display.width', 1600, 'display.max_colwidth', 800)
pp = pprint.PrettyPrinter(indent=4)


# ### 1.2.  Initialise Features Metadata

# Read the input features' confugration file &amp; store the features metadata

# In[ ]:


# variables settings
features_metadata = dict()

features_metadata_all = readers_writers.load_csv(path="", title=CONSTANTS.config_features_path, dataframing=True)
features_metadata = features_metadata_all.loc[(features_metadata_all["Selected"] == 1)]
features_metadata.reset_index()
    
# print
display(features_metadata)


# Set input features' metadata dictionaries

# In[ ]:


# Features names, and Dictionary of features types and dtypes
features_names = []
features_types = dict()
features_dtypes = dict()

for _, row in features_metadata.iterrows():
    if row["Selected"] == 1:
        features_names.append(row["Variable_Name"])
        features_types[row["Variable_Name"]] = row["Variable_Type"]
        features_dtypes[row["Variable_Name"]] = row["Variable_dType"]

features_dtypes = pd.DataFrame(features_dtypes, index=[0]).dtypes


# In[ ]:


# Dictionary of features groups
features_types_group = OrderedDict()

f_types = set([f_type for f_type in features_types.values()])
features_types_group = OrderedDict(zip(list(f_types), [set() for _ in range(len(f_types))]))
for f_name, f_type in features_types.items():
    features_types_group[f_type].add(f_name)
    
print("Available features types: " + ','.join(f_types))


# <br/><br/>

# ## 2. Read Data

# ### 2.1. Option I: Read from CSV file

# Read the input features from the CSV input file
# - Specify the input folder path
# <br/>&#9; &#8594; input_path
# - Specify the input files names
# <br/>&#9; &#8594; input_files

# In[ ]:


input_path = os.path.abspath("Samples")
input_files = dict()
input_files["train"] = dict({"Age-65p_01": "sample__Cond_Age-65p_0__train",
                             "Age-65p_1": "sample__Cond_Age-65p_1__train",
                             "Main": "sample__Cond_Main__train",
                             "Prior-Acute-12-month_0": "sample__Cond_Prior-Acute-12-month_0__train",
                             "Prior-Acute-12-month_1": "sample__Cond_Prior-Acute-12-month_1__train",
                             "Cond_Prior-Oper-12-month_0": "sample__Cond_Prior-Oper-12-month_0__train",
                             "Cond_Prior-Oper-12-month_1": "sample__Cond_Prior-Oper-12-month_1__train"})
input_files["test"] = dict({"Age-65p_01": "sample__Cond_Age-65p_0__test",
                            "Age-65p_1": "sample__Cond_Age-65p_1__test",
                            "Main": "sample__Cond_Main__test",
                            "Prior-Acute-12-month_0": "sample__Cond_Prior-Acute-12-month_0__test",
                            "Prior-Acute-12-month_1": "sample__Cond_Prior-Acute-12-month_1__test",
                            "Cond_Prior-Oper-12-month_0": "sample__Cond_Prior-Oper-12-month_0__test",
                            "Cond_Prior-Oper-12-month_1": "sample__Cond_Prior-Oper-12-month_1__test"})


# In[ ]:


features_input = dict({"train": {}, "test": {}})
for sample in input_files.keys():
    for submodel in input_files[sample].keys():
        features_input[sample][submodel] = readers_writers.load_csv(path=input_path, title=input_files[sample][submodel], 
                                                                    dataframing=True)
        print("Sample Name: ", sample, 
              "; Submodel Name: ", submodel, 
              "; Number of columns: ", len(features_input[sample][submodel].columns), 
              "; Total records: ", len(features_input[sample][submodel].index))


# ### 2.2. Option II: Read from MySQL database

# Read the input features from the MySQL database
# - Specify the database name
# <br/>&#9; &#8594; db_schema
# - Specify the input table name
# <br/>&#9; &#8594; db_tables

# In[ ]:


# db_schema = ""
# db_tables = dict()
# db_tables["train"] = dict({"Age-65p_01": "sample__Cond_Age-65p_0__train",
#                            "Age-65p_1": "sample__Cond_Age-65p_1__train",
#                            "Main": "sample__Cond_Main__train",
#                            "Prior-Acute-12-month_0": "sample__Cond_Prior-Acute-12-month_0__train",
#                            "Prior-Acute-12-month_1": "sample__Cond_Prior-Acute-12-month_1__train",
#                            "Cond_Prior-Oper-12-month_0": "sample__Cond_Prior-Oper-12-month_0__train",
#                            "Cond_Prior-Oper-12-month_1": "sample__Cond_Prior-Oper-12-month_1__train"})
# db_tables["test"] = dict({"Age-65p_01": "sample__Cond_Age-65p_0__test",
#                           "Age-65p_1": "sample__Cond_Age-65p_1__test",
#                           "Main": "sample__Cond_Main__test.",
#                           "Prior-Acute-12-month_0": "sample__Cond_Prior-Acute-12-month_0__test",
#                           "Prior-Acute-12-month_1": "sample__Cond_Prior-Acute-12-month_1__test",
#                           "Cond_Prior-Oper-12-month_0": "sample__Cond_Prior-Oper-12-month_0__test",
#                           "Cond_Prior-Oper-12-month_1": "sample__Cond_Prior-Oper-12-month_1__test"})


# In[ ]:


# features_input = dict({"train": {}, "test": {}})
# for sample in input_files.keys():
#     for submodel in input_files[sample].keys():
#         features_input[sample][submodel] = readers_writers.load_mysql_table(db_schema, db_tables[sample][submodel], 
#                                                                             dataframing=True)
#         print("Sample Name: ", sample, "; Submodel Name: ", submodel, 
#               "; Number of columns: ", len(features_input[sample][submodel].columns), 
#               "; Total records: ", len(features_input[sample][submodel].index))


# <br/><br/>

# # 3. Prepare Features

# ## 3.1. Set Features Types

# In[ ]:


for sample in input_files.keys():
    for submodel in input_files[sample].keys():
        for col_name in features_input[sample][submodel].columns:
            if col_name not in features_dtypes.keys():
                logger.error("Invalid column in the input file: " + str(col_name))
        
        col_names = set(features_input[sample][submodel].columns)
        for col_name in features_dtypes.keys():
            if col_name not in col_names:
                logger.error("Missing column in the input file: " + str(col_name))


# In[ ]:


for sample in input_files.keys():
    for submodel in input_files[sample].keys():
        features_input[sample][submodel].astype(dtype=features_dtypes, inplace=True)


# ## 3.2. Summary Statistics

# Produce a descriptive stat report of 'Categorical', 'Continuous', & 'TARGET' features

# In[ ]:


for sample in input_files.keys():
    for submodel in input_files[sample].keys():
        print("Sample Name: ", sample, "; Submodel Name: ", submodel)
        o_stats = preprocess.stats_discrete_df(df=features_input[sample][submodel], 
                                               includes=features_types_group["CATEGORICAL"],
                                               file_name="Stats_Categorical_" + sample + "_" + submodel)
        o_stats = preprocess.stats_continuous_df(df=features_input[sample][submodel], 
                                                 includes=features_types_group["CONTINUOUS"], 
                                                 file_name="Stats_Continuous_" + sample + "_" + submodel)
        o_stats = preprocess.stats_discrete_df(df=features_input[sample][submodel], 
                                               includes=features_types_group["TARGET"], 
                                               file_name="Stats_Target_" + sample + "_" + submodel)


# <br/><br/>

# # 4. Build Sub-Models

# Set the path of the Infer.Net libarary. It will be used to set local path variables when calling Infer.Net functions.
# 
# More: <a href="http://infernet.azurewebsites.net/docs/Infer.NET%20Learners%20-%20Matchbox%20recommender%20-%20Command-line%20runners.aspx">the Infer.NET command-line runners</a>

# In[ ]:


cmdLearner = "set \"PATH=%PATH%;" + os.path.abspath("Libraries/Infer.NET_2.6/Bin/") + "\" && Learner"


# ## 4.1. Prepare Inputs

# Write the selected features for submodels into CSV files, to be used for training and testing using Infer.Net

# In[ ]:


for sample in input_files.keys():
    for submodel in input_files[sample].keys():
        readers_writers.save_csv(data=features_input[sample][submodel].drop(['hesid'], axis=1), 
                                 path=io_path, title=input_files[sample][submodel] + "_InferDotNet", append=False, ext="csv", 
                                 header=features_input[sample][submodel].drop(['hesid'], axis=1).columns)


# Convert the CSV to correct format, using the Infer.Net <a href="http://infernet.azurewebsites.net/docs/Infer.NET%20Learners%20-%20Matchbox%20recommender%20-%20Command-line%20runners.aspx">guideline</a>.

# In[ ]:


for sample in input_files.keys():
    for submodel in input_files[sample].keys():
        # execute shell commands
        cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
        cmd2 = os.path.abspath("ReadersWriters\script_vim.txt")
        get_ipython().system('vim $cmd1 -S $cmd2')
        
        cmd1 = os.path.join(io_path, "." + input_files[sample][submodel] + "_InferDotNet.csv.un~")
        get_ipython().system('DEL $cmd1')
        cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv~")
        get_ipython().system('DEL $cmd1')


# ## 4.2. Train

# ### 4.2.1. Learn

# Train submodels (using train sub-sample)

# In[ ]:


sample = "train"
for submodel in input_files[sample].keys():
    print("Sample Name: ", sample, "; Submodel Name: ", submodel)
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
    cmd2 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_model.mdl")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier BinaryBayesPointMachine Train --iterations 30 --batches 1 --training-set $cmd1 --model $cmd2  --compute-evidence')


# ### 4.2.2. Predict the Train Sample

# Test submodels using train sub-sample

# In[ ]:


sample = "train"
for submodel in input_files[sample].keys():
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
    cmd2 = os.path.join(io_path, input_files["train"][submodel] + "_evaluate_model.mdl")
    cmd3 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_predictions.predictions")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier BinaryBayesPointMachine Predict --test-set $cmd1 --model $cmd2 --predictions $cmd3')


# ### 4.2.3. Evaluate

# Evalute the perfromance of the predictions on train sub-sample

# In[ ]:


sample = "train"
for submodel in input_files[sample].keys():
    print("Sample Name: ", sample, "; Submodel Name: ", submodel)
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
    cmd2 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_predictions.predictions")
    cmd3 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_Report.txt")
    cmd4 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_CalibrationCurve.csv")
    cmd5 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_ROC.csv")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier Evaluate --ground-truth $cmd1 --predictions $cmd2 --report $cmd3 --calibration-curve $cmd4 --roc-curve $cmd5  --positive-class 1')


# ## 4.3. Test

# ### 4.3.1. Predict the Test Sample

# Test submodels using test sub-sample

# In[ ]:


sample = "test"
for submodel in input_files[sample].keys():
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
    cmd2 = os.path.join(io_path, input_files["train"][submodel] + "_evaluate_model.mdl")
    cmd3 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_predictions.predictions")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier BinaryBayesPointMachine Predict --test-set $cmd1 --model $cmd2 --predictions $cmd3')


# ### 4.3.2. Evaluate

# Evalute the perfromance of the predictions on test sub-sample

# In[ ]:


sample = "test"
for submodel in input_files[sample].keys():
    print("Sample Name: ", sample, "; Submodel Name: ", submodel)
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
    cmd2 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_predictions.predictions")
    cmd3 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_Report.txt")
    cmd4 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_CalibrationCurve.csv")
    cmd5 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_ROC.csv")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier Evaluate --ground-truth $cmd1 --predictions $cmd2 --report $cmd3 --calibration-curve $cmd4 --roc-curve $cmd5  --positive-class 1')


# ## 4.4. Weight Matrix

# Get the posterior weight distribution of features for the trained model

# In[ ]:


sample = "train"
for submodel in input_files[sample].keys():
    print("Sample Name: ", sample, "; Submodel Name: ", submodel)
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_evaluate_model.mdl")
    cmd2 = os.path.join(io_path, input_files[sample][submodel] + "_report_weights.txt")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier BinaryBayesPointMachine SampleWeights  --model $cmd1 --samples $cmd2')


# ## 4.5. Diagnose Train

# Assess the convergence of the message-passing algorithms used to train the Bayes Point Machine classifiers.

# In[ ]:


sample = "train"
for submodel in input_files[sample].keys():
    print("Sample Name: ", sample, "; Submodel Name: ", submodel)
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
    cmd2 = os.path.join(io_path, input_files[sample][submodel] + "_report_diagnoseTrain.csv")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier BinaryBayesPointMachine DiagnoseTrain --iterations 500 --batches 1 --training-set $cmd1  --results $cmd2')


# ## 4.6. Cross-Validate

# Assess the generalization performance of the Bayes Point Machine

# In[ ]:


sample = "train"
for submodel in input_files[sample].keys():
    print("Sample Name: ", sample, "; Submodel Name: ", submodel)
    cmd1 = os.path.join(io_path, input_files[sample][submodel] + "_InferDotNet.csv")
    cmd2 = os.path.join(io_path, input_files[sample][submodel] + "_report_crossValidation.csv")
    
    # execute shell command
    get_ipython().system('$cmdLearner Classifier BinaryBayesPointMachine CrossValidate --folds 5 --iterations 30 --batches 1 --data-set $cmd1 --results $cmd2 --compute-evidence')


# <br/><br/>

# # 5. Build Ensemble Model

# ## 5.1. Initialise Labels & Predictions

# Get the true labels

# In[ ]:


labels = dict()
ids = dict()
ids_all = set()

for sample in input_files.keys():
    ids[sample] = dict()
    for submodel in input_files[sample].keys():
        print("Sample Name: ", sample, "; Submodel Name: ", submodel)
        ids[sample][submodel] = dict(zip(features_input[sample][submodel]['hesid'], 
                                         range(1, len(features_input[sample][submodel]['hesid']))))
        ids_all = ids_all.union(set(features_input[sample][submodel]['hesid']))
        for i in range(len(features_input[sample][submodel]['hesid'])):
            labels[features_input[sample][submodel]['hesid'][i]] = features_input[sample][submodel]['label'][i]

ids_all = list(ids_all)


# Get the predicted probailisites of label-1 (to be readmitted)

# In[ ]:


predictions = dict()

for sample in input_files.keys():
    predictions[sample] = dict()
    for submodel in input_files[sample].keys():
        print("Sample Name: ", sample, "; Submodel Name: ", submodel)
        report = readers_writers.load_csv(path=os.path.join(io_path), 
                                          title=input_files[sample][submodel] + "_evaluate_predictions",
                                          ext="predictions",
                                          dataframing=False)
        predictions[sample][submodel] = [float((row[0].split(' ')[0]).split('=')[1]) for row in report]


# Combine labels and predictions for all the sample instances, including the instances that are not in submodels

# In[ ]:


predictions_all = dict()
labels_all = dict()

for sample in input_files.keys():
    predictions_all[sample] = dict()
    labels_all[sample] = dict()
    for submodel in input_files[sample].keys():
        predictions_all[sample][submodel] = []
        labels_all[sample][submodel] = []
        for i in ids_all:
            labels_all[sample][submodel].append(labels[i])
            if i in ids[sample][submodel].keys():
                predictions_all[sample][submodel].append(predictions[sample][submodel][ids[sample][submodel][i]])
            else:
                predictions_all[sample][submodel].append(0)


# ## 5.2. Train

# ### 5.2.1. Optimise

# Initialise the ensemble model

# In[ ]:


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
                               output_path=os.path.join(io_path),
                               output_name="ensemble_model_outputs")


# Prepare the ensemble model inputs

# In[ ]:


submodels_names = list(input_files[sample].keys())
submodels_names.sort()
sample = "train"

y = labels_all[sample][submodels_names[0]]
y_hat_submodels = np.array([predictions_all[sample][submodel] for submodel in submodels_names])


# Run the ensemble model optimiser

# In[ ]:


ensemble_model.run(y=y,
                   y_hat_submodels=y_hat_submodels,
                   submodels_names=submodels_names,
                   submodel_main="Main")


# ### 5.2.2. View Generated Models

# Open the summary output file that was generated by the ensemble model algorithm

# In[ ]:


output_path_full = os.path.join(io_path) + "/ensemble_model_outputs.log"
print("Output file path: ", output_path_full)

if sys.platform.startswith('darwin'):
    subprocess.call(('open', output_path_full))
elif os.name == 'nt':
    os.startfile(output_path_full)
elif os.name == 'posix':
    subprocess.call(('xdg-open', output_path_full))


# ## 5.3. Test

# ### 5.3.1. Configure an Ensemble Model

# Configure the weights for an ensemble model based on one of the generted ensemble models in the previous step

# In[ ]:


submodels_names_func = {"Age-65p_01": 17, "Age-65p_1": 6, "Cond_Prior-Oper-12-month_0": 8,
                        "Cond_Prior-Oper-12-month_1": 7, "Main": 6, "Prior-Acute-12-month_0": 12, 
                        "Prior-Acute-12-month_1": 7}


# ### 5.3.2. Assess Preformance for Train Sample

# Generate the predicted probabilities for the ensemble model

# In[ ]:


y_hat_ensemble_1 = ensemble_model.ensemble_y_hat(y_hat_submodels=y_hat_submodels,
                                                 submodels_names=submodels_names,
                                                 submodels_names_func=submodels_names_func,
                                                 ensemble_type="mean")


# Generate statistics for the ensemble model

# In[ ]:


output_stats = ensemble_model.stat_report_full(y=y,
                                               y_hat=y_hat_ensemble_1,
                                               cut_off=0.5)
print("Examine an ensemble model: \n" + str(submodels_names_func))
pp.pprint(output_stats)


# ### 5.3.3. Assess Preformance for Test Sample

# Generate the predicted probabilities for an ensemble model

# In[ ]:


y_hat_ensemble_1 = ensemble_model.ensemble_y_hat(y_hat_submodels=y_hat_submodels,
                                                 submodels_names=submodels_names,
                                                 submodels_names_func=submodels_names_func,
                                                 ensemble_type="mean")


# Generate statistics for a model

# In[ ]:


output_stats = ensemble_model.stat_report_full(y=y,
                                               y_hat=y_hat_ensemble_1,
                                               cut_off=0.5)
print("Examine an ensemble model: \n" + str(submodels_names_func))
pp.pprint(output_stats)


# End!
