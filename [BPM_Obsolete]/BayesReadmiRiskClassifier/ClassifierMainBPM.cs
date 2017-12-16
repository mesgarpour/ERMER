/// <copyright file="ClassifierMain.cs" company="">
/// Copyright (c) 2014, 2016 All Right Reserved, https://github.com/mesgarpour/BayesReadmiRiskClassifier
///
/// This source is subject to the The Apache License, Version 2.0.
/// Please see the License.txt file for more information.
/// All other rights reserved.
///
/// THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY 
/// KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
/// PARTICULAR PURPOSE.
///
/// </copyright>
/// <author>Mohsen Mesgarpour</author>
/// <email>mohsen.meagrpour@email.com</email>
/// <date>2015-12-01</date>
/// <summary></summary>
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using MicrosoftResearch.Infer;

class ClassifierMainBPM
{
    private Models _model;
    private Features _features;
    private string _modellingProfile = null;
    private bool _verbose = false;

    /// <summary>
    /// The Main function for execution
    /// </summary>
    /// <param name="args">The command line arguments</param>
    static void Main(string[] args)
    {
        // Check input arguments
        if (args.Length != 1 & args.Length != 2)
        {
            TraceListeners.Log(TraceEventType.Information, 0,
                "Please enter the input arguments!"
                + "\n Compulsory Input Arguments:\n "
                + "[1] Model Name \n"
                + "\n Optional Input Arguments:\n "
                + "[2] Verbose Log (\"verbose\") \n", false, true);
            Console.ReadKey();
            return;
        }

        // Main
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain...", false, true);
        TraceListeners.Log(TraceEventType.Information, 0,
            "...Arguments: " + string.Join(", ", args), false, true);

        ClassifierMainBPM classifierMainBPM = new ClassifierMainBPM();
        classifierMainBPM.InitializeApp(args);
        classifierMainBPM.AnalysisProcessManagerGroup(
            classifierMainBPM._modellingProfile);

        TraceListeners.Log(TraceEventType.Information, 0, "Fin!", false, true);
        Console.ReadKey();
    }

    /// <summary>
    /// Initialise the logger and constants
    /// </summary>
    /// <param name="args">The command line arguments</param>
    public void InitializeApp(string[] args)
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::InitializeApp...", false, true);

        _modellingProfile = args[0];

        // Loggers
        if (args.Length == 2
            && args[1].ToLower() == "verbose")
        {
            _verbose = true;
            TraceListeners.Initialise(
                _verbose,
                Constants.GENERAL_LOG_PATH,
                Constants.DIAGNOSTIC_LOG_PATH);
        }
        else
        {
            TraceListeners.Initialise(_verbose,
                Constants.GENERAL_LOG_PATH,
                Constants.DIAGNOSTIC_LOG_PATH);
        }

        // Constants
        Constants.Initialise();
        ConstantsProfiles.Initialise();
        ConstantsModellings.Initialise();
    }

    /// <summary>
    /// Managing a group of <see cref="AnalysisProcessManager"/>
    /// </summary>
    /// <param name="modelSettingProfile"></param>
    public void AnalysisProcessManagerGroup(
        string modellingProfile)
    {
        TraceListeners.Log(TraceEventType.Information, 0, "ClassifierMain::AnalysisProcessManagerGroup...", false, true);

        //Get the profile
        string modellingSetting = ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.ModellingSetting][0];
        string modelName = ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.ModelName][0];
        string modelProcedure = ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.ModelProcedure][0];
        string[] modellingsGroup = ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.ModelingsGroup];
        string[] submodels = ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.Submodels];
        string[] submodelsCondValue = ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.SubmodelsNumCond];
        string[] samplingsCombName = ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.SamplingsCombName];
        Dictionary<DatasetName, string> samplingsCombDatabase = 
            new Dictionary<DatasetName, string>();
        Dictionary<DatasetName, string> samplingsCombSample = 
            new Dictionary<DatasetName, string>();
        //ConstantsModellings.Keys modellingSetting = (ConstantsModellings.Keys)Enum.Parse(typeof(ConstantsModellings.Keys),
        //    ConstantsProfiles.DIC[modellingProfile][ConstantsProfiles.Keys.ModellingSetting][0]);

        // Initialise the model
        _model = new Models();
        _model.Create(
            modellingSetting);

        // Iterate through the profile settings
        // samplingCombName
        foreach (string samplingCombName in samplingsCombName)
        {
            samplingsCombDatabase[DatasetName.Train] =
                Constants.SAMPLING_COMB_TRAIN_DB[samplingCombName];
            samplingsCombDatabase[DatasetName.Test] =
                Constants.SAMPLING_COMB_TEST_DB[samplingCombName];

            samplingsCombSample[DatasetName.Train] =
                Constants.SAMPLING_COMB_TRAIN_SAMPLE[samplingCombName];
            samplingsCombSample[DatasetName.Test] =
                Constants.SAMPLING_COMB_TEST_SAMPLE[samplingCombName];

            // modellingGroup 
            foreach (string modellingGroup in modellingsGroup)
            {
                // submodel
                foreach (string submodel in submodels)
                {
                    // SubmodelCondValue
                    foreach (string SubmodelCondValue in submodelsCondValue)
                    {

                        TraceListeners.Log(TraceEventType.Information, 0,
                            "...Model -> modelName:" + modelName
                            + "; modelProcedure:" + modelProcedure
                            + "; samplingsCombDatabase:" + string.Join(",", samplingsCombDatabase)
                            + "; samplingsCombSample:" + string.Join(",", samplingsCombSample)
                            + "; samplingCombName:" + samplingCombName
                            + "; modellingGroup:" + modellingGroup
                            + "; submodel:" + submodel
                            + "; SubmodelCondValue:" + SubmodelCondValue, false, true);

                        // Initialise the features
                        _features = new Features(
                            Constants.VARS_METADATA_PATH,
                            Constants.VARS_METADATA_COL_KEY,
                            Constants.VARS_METADATA_COL_TYPE,
                            Constants.VARS_METADATA_LABEL_NAME);

                        // An iteration of AnalysisProcessManager
                        AnalysisProcessManager(modelName, modelProcedure,
                            modellingGroup, submodel, SubmodelCondValue,
                            samplingsCombDatabase, samplingsCombSample);
                    } // SubmodelCondValue
                } // submodel
            } // modellingGroup 
        } // samplingCombName
    }

    /// <summary>
    /// Manage the getting data, model training, model testing, model validation and reporting.
    /// </summary>
    public void AnalysisProcessManager(
        string modelName,
        string modelProcedure,
        string modellingGroup,
        string submodel,
        string submodelCondValue,
        Dictionary<DatasetName, string> samplingsCombDatabase,
        Dictionary<DatasetName, string> samplingsCombSample)
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::AnalysisProcessManager...", false, true);

        // Get data, and train
        ReadInputData(modelName, modelProcedure, modellingGroup, submodel,
            submodelCondValue, DatasetName.Train, samplingsCombDatabase,
            samplingsCombSample);
        Train(true);

        // Clear features from the model instance (saveing memory)
        RemoveInputData();

        // Get data, and test
        ReadInputData(modelName, modelProcedure, modellingGroup, submodel, 
            submodelCondValue, DatasetName.Test, samplingsCombDatabase, 
            samplingsCombSample);
        Predict(true);

        // Evaluations and validations
        Evaluate();
        DiagnoseTRain();
        CrossValidate();

        // Clear features from the model instance (saveing memory)
        RemoveInputData();
    }


    /// <summary>
    /// Get train and test datsets.
    /// </summary>
    public void ReadInputData(
        string modelName,
        string modelProcedure,
        string modellingGroup,
        string submodel,
        string submodelCondValue,
        DatasetName datasetName,
        Dictionary<DatasetName, string> samplingsCombDatabase,
        Dictionary<DatasetName, string> samplingsCombSample)
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::ReadInputData...", false, true);

        string query;
        string[] featureNames;

        // Read
        query = "CALL " + Constants.MYSQL_DATABASES_MYSQL[samplingsCombDatabase[datasetName]]
            + "." + modelName + "(\"" + modelProcedure + "\", \"" + modellingGroup + "\", \""
            + submodel + "\"," + submodelCondValue + ",\"" + samplingsCombSample[datasetName] + "\");";
        ReadMySQL reader = new ReadMySQL(
            samplingsCombDatabase[datasetName],
            Constants.MYSQL_SERVER,
            Constants.MYSQL_USERNAME,
            Constants.MYSQL_PASSWORD,
            Constants.MYSQL_DATABASES_MYSQL[samplingsCombDatabase[datasetName]],
            Constants.MYSQL_CMD_TIMEOUT,
            Constants.MYSQL_CONN_TIMEOUT,
            Constants.MYSQL_NULL_DEFAULT.ToString());
        reader.Read(query);

        // Set
        featureNames = reader.GetColumnsNames();
        _features.Set(
            dataset: reader.GetColumns(featureNames),
            datasetName: datasetName,
            featuresByOrder: featureNames);

        // Convert & Transform
        _features.ConvertFeatures(
            datasetName: datasetName,
            standardConversion: true);

        // Close
        reader.CloseConnection();
    }

    /// <summary>
    /// Remove input data
    /// </summary>
    /// <param name="datasetName"></param>
    public void RemoveInputData()
    {
        _model.RemoveFeature();
    }

    /// <summary>
    /// Train
    /// </summary>
    public void Train(bool removeInputData)
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::Train...", false, true);

        // Set noise
        double noise = System.Convert.ToDouble(
            ConstantsModellings.DIC[ConstantsModellings.Keys.BPM][ConstantsModellings.KeysModels.NoiseTrain][0]);

        // Clear the features from the Feature instance (saveing memory)
        if (removeInputData)
            _features.RemoveFeature(DatasetName.Train);

        // Initialise features
        _model.InitialiseFeatures(
            DatasetName.Train,
            _features.GetFeaturesConverted(DatasetName.Train),
            _features.GetTargetConverted(DatasetName.Train));

        // Clear the features converted from the Feature instance (saveing memory)
        if(removeInputData)
            _features.RemoveFeatureConverted(DatasetName.Train);

        // Train
        _model.Train(
            outputModelFileName: Constants.OUTPUT_PATH + "model_bpm.mdl",
            iterationCount: 30,
            computeModelEvidence: true,
            batchCount: 1,
            distributionName: DistributionName.GaussianDefault,
            inferenceEngineAlgorithm: InferenceAlgorithm.EP,
            noise: noise);
    }

    /// <summary>
    /// Predict
    /// </summary>
    public void Predict(bool removeInputData)
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::Predict...", false, true);

        // Set noise
        double noise = System.Convert.ToDouble(
            ConstantsModellings.DIC[ConstantsModellings.Keys.BPM][ConstantsModellings.KeysModels.NoiseTest][0]);

        // Clear the features from the Feature instance (saveing memory)
        if (removeInputData)
            _features.RemoveFeature(DatasetName.Test);

        // Initialise features
        _model.InitialiseFeatures(
            DatasetName.Test,
            _features.GetFeaturesConverted(DatasetName.Test),
            _features.GetTargetConverted(DatasetName.Test));

        // Clear the features converted from the Feature instance (saveing memory)
        _features.RemoveFeatureConverted(DatasetName.Test);

        // Predict
        _model.Predict(
            inputModelFileName: Constants.OUTPUT_PATH + "model_bpm.mdl",
            distributionName: DistributionName.GaussianDefault,
            inferenceEngineAlgorithm: InferenceAlgorithm.EP,
            noise: noise);
    }

    /// <summary>
    /// Evalaute
    /// </summary>
    public void Evaluate()
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::Evaluate...", false, true);

        _model.Evaluate(
            inputModelFileName: Constants.OUTPUT_PATH + "model_bpm.mdl",
            reportFileName: Constants.OUTPUT_PATH + "report_evaluation.txt",
            positiveClassLabel: "1",
            groundTruthFileName: Constants.OUTPUT_PATH + "report_GroundTruth.txt",
            predictionsFileName: Constants.OUTPUT_PATH + "report_Predictions.txt",
            weightsFileName: Constants.OUTPUT_PATH + "report_Weights.txt",
            calibrationCurveFileName: Constants.OUTPUT_PATH + "report_Clibration_Curve.txt",
            precisionRecallCurveFileName: Constants.OUTPUT_PATH + "report_PrecisionRecall_Curve.txt",
            rocCurveFileName: Constants.OUTPUT_PATH + "report_ROC_Curve.txt");
    }

    /// <summary>
    /// Diagnose train
    /// </summary>
    public void DiagnoseTRain()
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::DiagnoseTRain...", false, true);

        _model.DiagnoseTrain(
            outputModelFileName: null,
            reportFileName: Constants.OUTPUT_PATH + "report_diagnoseTrain.txt",
            iterationCount: 30,
            computeModelEvidence: true,
            batchCount: 1);
    }

    /// <summary>
    /// Cross validate
    /// </summary>
    public void CrossValidate()
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "ClassifierMain::CrossValidate...", false, true);

        _model.CrossValidate(
            outputModelFileName: Constants.OUTPUT_PATH + "report_crossValidate.txt",
            crossValidationFoldCount: 3,
            iterationCount: 30,
            computeModelEvidence: true,
            batchCount: 1);
    }
}
