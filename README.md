<div align="left" style="width:60%; height:60%;">
  <img src="https://github.com/mesgarpour/ERMER/blob/master/Documents/Logo/logo_ermer.png">
</div>
<br><br>



-----------------
## The Ensemble model
| **`Linux Debian`** | **`Linux Fedora`** | **`Mac OS`** | **`Windows`** |
|-----------------|---------------------|------------------|-------------------|
| [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) | [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) | [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) | [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) |

-----------------
## The ERMER Notebook:
| **`Windows`** |
|-----------------|
| [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) |



------
# [ERMER](https://github.com/mesgarpour/ERMER)
[Ensemble Risk Model of Emergency Admissions (ERMER)](https://github.com/mesgarpour/ERMER) is an optimised ensemble of sub-models that are trained using [Bayes Point Machine (BPM)](http://www.jmlr.org/papers/v1/herbrich01a.html). The features of the model are generated using the [Healthcare Pre-Processing Framework](https://github.com/mesgarpour/Healthcare_PreProcessing_Framework), but it is not integrated into the [ERMER](https://github.com/mesgarpour/ERMER) development toolkit, in order to preserve the tool's generic structure. The [ERMER](https://github.com/mesgarpour/ERMER) development toolkit is a generic, user-friendly and open-source software package that can be used for development of temporal comorbidity index independent of source of healthcare data.

The development toolkit consists of two parts:
+  [The Ensemble model](https://github.com/mesgarpour/ERMER/tree/master/Ensemble_Model): The generic implementation of the ensemble optimisation algorithm
+  [The ERMER Notebook](https://github.com/mesgarpour/ERMER/tree/master/ERMER_Notebook): The Ensemble model & the [BPM](http://www.jmlr.org/papers/v1/herbrich01a.html) algorithm (to be added!)

In addition the framework might be used to pre-prrocess and generate features:
+ The [Healthcare Pre-Processing Framework](https://github.com/mesgarpour/Healthcare_PreProcessing_Framework)


# Introduction
About half of hospital readmissions can be avoided with preventive interventions. Developing decision support tools for identification of patients' emergency readmission risk is an important area of research. Because, it remains unclear how to design features and develop predictive models that can adjust continuously to a fast-changing healthcare system and population characteristics. The objective of this study was to develop a generic [Ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) of [Bayesian](https://en.wikipedia.org/wiki/Bayesian_inference) emergency readmission risk models.

Most existing decision support tools, that are based on hospital administrative data, use [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) or [Coxian Phase-type Distribution models](https://en.wikipedia.org/wiki/Phase-type_distribution), and have very limited capability. This phase of research develops an [Ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) generative risk model of emergency readmission within a year to the [England's hospitals](https://www.england.nhs.uk/). The [Machine Learning Ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) method is a powerful technique, which uses a finite set of weaker models and an algorithm to combine and optimise the performance of the [Ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) model. 

An [Ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) of generated [BPM](http://www.jmlr.org/papers/v1/herbrich01a.html) models of emergency readmission has been developed, which is based on a collection of sub-models that are conditioned on different population characteristics. The proposed model, [Ensemble Risk Model of Emergency Admissions (ERMER)](https://github.com/mesgarpour/ERMER), utilises a weighted average ranking method to optimise the weights of sub-classifier using a [bidirectional hill-climbing heuristic](https://en.wikipedia.org/wiki/Hill_climbing). The novelty lies in the intuitive adaptation of an [Ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) modelling with a generative approach for prediction of patients' risks. Moreover, the [Ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) of specialised sub-models for prediction of patients risks has not been addressed with existing studies.

In this research, [Microsoft's Infer.Net](http://infernet.azurewebsites.net/) library was used to construct the [BPM](http://www.jmlr.org/papers/v1/herbrich01a.html) model. The applied algorithm uses the original version of the [BPM](http://www.jmlr.org/papers/v1/herbrich01a.html), with two main modifications. Firstly, it uses a mixture of [Gamma-Gamma](https://en.wikipedia.org/wiki/Gamma_distribution), a heavy-tailed prior probability distribution for the precision of weights and features. Secondly, it applies [Expectation Propagation](https://en.wikipedia.org/wiki/Expectation_propagation) message passing to infer posterior probabilities, which has been demonstrated in Gaussian Mixture problems to be better than approximation techniques. 

Therefore, the applied [BPM](http://www.jmlr.org/papers/v1/herbrich01a.html) is invariant to parameter rescaling or shifting, unlike[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) or [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) methods. Moreover, active [Bayesian](https://en.wikipedia.org/wiki/Bayesian_inference) training can allow continuous updates of the model and account for changes in the prior probabilities. Furthermore, the [BPM](http://www.jmlr.org/papers/v1/herbrich01a.html) can efficiently handle a relatively larger number of features. 



# Performance
Based on the defined [Healthcare Pre-Processing Framework](https://github.com/mesgarpour/Healthcare_PreProcessing_Framework) introduced in the Phase-I of our research, features were generated, filtered, and ranked. Thereafter, a number of sub-models based on population characteristics were trained using a [BPM](http://www.jmlr.org/papers/v1/herbrich01a.html) approach. Afterwards, an optimised Ensemble model of these sub-models was generated. The developed [ERMER](https://github.com/mesgarpour/ERMER) was trained and tested using three time-frames: 1999-2004, 2000-05, and 2004-09, each of which includes 20% of patients admitted within the trigger-year. In addition, a development toolkit is supplemented to ease the validation and adaptation of the [ERMER](https://github.com/mesgarpour/ERMER).

Comparisons are made for different time-frames, sub-populations, risk cut-offs, risk bands, and top risk segments. The [precision](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) was 71.6% to 73.9%, the [specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) was 88.3% to 91.7%, and the [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) was 42.1% to 49.2% across different time-frames. Moreover, the [area under the curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) was 75.9% to 77.1%. 

The proposed decision support tool performed considerably better than the previous modelling approaches, and it was robust and stable with high precision. Moreover, the [Healthcare Pre-Processing Framework](https://github.com/mesgarpour/Healthcare_PreProcessing_Framework) and the Ensemble [Bayesian](https://en.wikipedia.org/wiki/Bayesian_inference) approach allow the [ERMER](https://github.com/mesgarpour/ERMER) to continuously be adjusted to new significant features, different population characteristics and changes in the system. 



# Related Publications
+  Mesgarpour, M., Chaussalet, T. & Chahed, S. (2017) [Ensemble Risk Model of Emergency Readmission (ERMER)](http://dx.doi.org/10.1016/j.ijmedinf.2017.04.010). International Journal of Medical Informatics, 2017 Elsevier.
+  Mesgarpour, M. (2017) Using Machine Learning Techniques in Predictive Risk Modelling in Healthcare. (to be added!)



# License
[Apache License, Version 2.0.](https://www.apache.org/licenses/LICENSE-2.0.html)
Enjoy!



# Creadits
Original Author: [Mohsen Mesgarpour](https://uk.linkedin.com/in/mesgarpour), [Health and Social Care Modelling Group (HSCMG)](http://www.healthcareanalytics.co.uk/), [University of Westminster](https://www.westminster.ac.uk/).

Most Recent Author: [Mohsen Mesgarpour](https://uk.linkedin.com/in/mesgarpour), [Health and Social Care Modelling Group (HSCMG)](http://www.healthcareanalytics.co.uk/), [University of Westminster](https://www.westminster.ac.uk/).
