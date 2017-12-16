using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer;

interface IModels
{
    void InitialiseFeatures(
        DatasetName datasetName,
        object[,] features,
        object[] target);

    void Train(
        string outputModelFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        double noise);

    void TrainIncremental(
        string inputModelFileName,
        string outputModelFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        double noise);

    void Predict(
        string inputModelFileName,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        double noise);

    IDictionary<string, double> PredictInstance(
        string inputModelFileName,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        int instance,
        double noise);

    void Evaluate(
        string inputModelFileName,
        string reportFileName,
        string positiveClassLabel,
        string groundTruthFileName,
        string predictionsFileName,
        string weightsFileName,
        string calibrationCurveFileName,
        string precisionRecallCurveFileName,
        string rocCurveFileName);

    void DiagnoseTrain(
        string outputModelFileName,
        string reportFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount);

    void CrossValidate(
        string outputModelFileName,
        int crossValidationFoldCount,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount);

    void RemoveFeature();
}
