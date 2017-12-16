using System;
using System.Collections.Generic;
using System.Diagnostics;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Learners;
using MicrosoftResearch.Infer.Learners.Mappings;
using MicrosoftResearch.Infer;
using System.Linq;

class BPMMapped : Models, IModels
{
    /// <summary>Model's feature matrix</summary>
    private Vector[] _x = null;

    /// <summary>Model's target vector</summary>
    private IList<string> _y = null;

    /// <summary>Predicted distributions</summary>  
    IEnumerable<IDictionary<string, double>> _yPredicDistrib = null;

    /// <summary>Predicted labels</summary>  
    IEnumerable<string> _yPredicLabel = null;

    /// <summary>The classifier instance</summary>
    private IBayesPointMachineClassifier<
        IList<Vector>,       // the type of the instance source (it may also be defined as SparseVector or DenseVector), 
        int,                 // the type of an instance
        IList<string>,       // the type of the label source
        string,              // the type of a label.
        IDictionary<string, double>,
        BayesPointMachineClassifierTrainingSettings,
        BinaryBayesPointMachineClassifierPredictionSettings<string>> _classifier = null;

    /// <summary>The classifier evaluatior instance</summary>
    private ClassifierEvaluator<
        IList<Vector>,
        int,
        IList<string>,
        string> _evaluator = null;

    /// <summary>The classifier mapping instance</summary>
    private GenericClassifierMapping _mapping = null;

    /// <summary>Number of observations</summary>
    private int _numObservations;

    /// <summary>Number of features</summary>
    private int _numFeatures;

    /// <summary>The name of the current saved dataset</summary>
    private DatasetName _availableDatasetName;

    /// <summary>The validation instance</summary>
    private Validate _validate = null;



    public BPMMapped(
        string[] labels)
    {
        Debug.Assert(labels != null, "The labels must not be null.");
        Debug.Assert(labels.Length == 2, "The labels must have two possible values.");

        // Initialise the validations
        _validate = new Validate();

        // Create a BPM from the mapping
        _mapping = new GenericClassifierMapping(labels);
        _classifier = BayesPointMachineClassifier.CreateBinaryClassifier(_mapping);

        // Evaluator mapping
        var evaluatorMapping = _mapping.ForEvaluation();
        _evaluator = new ClassifierEvaluator
            <IList<Vector>, int, IList<string>, string>(evaluatorMapping);

        // Other initialisations
        _availableDatasetName = new DatasetName();
        _numObservations = 0;
        _numFeatures = 0;
    }

    public override void InitialiseFeatures(
        DatasetName datasetName,
        object[,] x,
        object[] y)
    {
        Debug.Assert(x != null, "The feature vector must not be null.");
        Debug.Assert(y != null, "The targe variable must not be null.");

        // Validate
        _validate.Dataset(
            labels: _mapping.GetClassLabels().ToArray(), 
            datasetName: datasetName, 
            x: x, 
            y: y);

        // Set meta data
        _numFeatures = x.GetLength(0);
        _numObservations = x.GetLength(1);

        // Transpose
        double[][] xTrans = new double[_numObservations][];

        for (int j = 0; j < _numObservations; j++)
        {
            xTrans[j] = new double[_numFeatures];
            for (int i = 0; i < _numFeatures; i++)
            {
                xTrans[j][i] = (double)x[i, j];
            }
        }

        // Set target
        _y = new List<string>(
        Array.ConvertAll(y, v => v.ToString()));

        // Set features
        _x = new Vector[_numObservations];
        for (int i = 0; i < _numObservations; i++)
        {
            _x[i] = Vector.FromArray(xTrans[i]);
        }
        
        _availableDatasetName = datasetName;
    }

    public override void Train(
        string outputModelFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        double noise)
    {
        TraceListeners.Log(TraceEventType.Warning, 0,
            "Advanced setting will not be used: " +
            "distributionName, inferenceEngineAlgorithm & noise.", false, true);

        // Validate
        _validate.Train(
            outputModelFileName: outputModelFileName,
            iterationCount: iterationCount,
            batchCount: batchCount);

        Train(
            outputModelFileName,
            iterationCount,
            computeModelEvidence,
            batchCount);
    }

    private void Train(
        string outputModelFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount)
    {
        // Set settings
        _classifier.Settings.Training.ComputeModelEvidence = computeModelEvidence;
        _classifier.Settings.Training.IterationCount = iterationCount;
        _classifier.Settings.Training.BatchCount = batchCount;

        // train
        _classifier.Train(_x, _y);
        _classifier.Save(outputModelFileName);
    }

    public override void TrainIncremental(
        string inputModelFileName,
        string outputModelFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        double noise)
    {
        TraceListeners.Log(TraceEventType.Warning, 0,
            "Advanced setting will not be used: " +
            "distributionName, inferenceEngineAlgorithm & noise.", false, true);

        // Validate
        _validate.TrainIncremental(
            inputModelFileName: inputModelFileName,
            outputModelFileName: outputModelFileName,
            iterationCount: iterationCount,
            batchCount: batchCount);

        // Load model
        IBayesPointMachineClassifier<
            IList<Vector>, int, IList<string>, string, IDictionary<string, double>,
            BayesPointMachineClassifierTrainingSettings,
            BinaryBayesPointMachineClassifierPredictionSettings<string>> classifier =
            BayesPointMachineClassifier.LoadBinaryClassifier<
                IList<Vector>, int, IList<string>, string, IDictionary<string, double>>
                (inputModelFileName);

        // Set settings
        classifier.Settings.Training.ComputeModelEvidence = computeModelEvidence;
        classifier.Settings.Training.IterationCount = iterationCount;
        classifier.Settings.Training.BatchCount = batchCount;

        // train
        classifier.TrainIncremental(_x, _y);
        classifier.Save(outputModelFileName);
    }

    public override void Predict(
        string inputModelFileName,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        double noise)
    {
        TraceListeners.Log(TraceEventType.Warning, 0,
            "Advanced setting will not be used: " +
            "distributionName, inferenceEngineAlgorithm & noise.", false, true);

        // Validate
        _validate.Predict(inputModelFileName);

        // Define the classifier
        IBayesPointMachineClassifier<
                IList<Vector>, int, IList<string>, string, IDictionary<string, double>,
                BayesPointMachineClassifierTrainingSettings,
                BinaryBayesPointMachineClassifierPredictionSettings<string>> classifier = null;

        // Load model
        if (string.IsNullOrEmpty(inputModelFileName))
        {
            classifier =
                BayesPointMachineClassifier.LoadBinaryClassifier<
                    IList<Vector>, int, IList<string>, string, IDictionary<string, double>>
                    (inputModelFileName);
        }
        else
        {
            classifier = _classifier;
        }

        _validate.ValidatePredict(_x, _x);
        _yPredicDistrib = classifier.PredictDistribution(_x);
        _yPredicLabel = classifier.Predict(_x);
    }

    public override IDictionary<string, double> PredictInstance(
        string inputModelFileName,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        int instance,
        double noise)
    {
        TraceListeners.Log(TraceEventType.Warning, 0,
            "Advanced setting will not be used: " +
            "distributionName, inferenceEngineAlgorithm  & noise.", false, true);

        // Validate
        _validate.PredictInstance(
            inputModelFileName: inputModelFileName,
            instance: instance,
            numObservations: _numObservations);

        IBayesPointMachineClassifier<
                IList<Vector>, int, IList<string>, string, IDictionary<string, double>,
                BayesPointMachineClassifierTrainingSettings,
                BinaryBayesPointMachineClassifierPredictionSettings<string>> classifier = null;

        // Load model
        if (string.IsNullOrEmpty(inputModelFileName))
        {
            classifier =
                BayesPointMachineClassifier.LoadBinaryClassifier<
                    IList<Vector>, int, IList<string>, string, IDictionary<string, double>>
                    (inputModelFileName);
        }
        else
        {
            classifier = _classifier;
        }

        IDictionary<string, double> yPredicted =
            classifier.PredictDistribution(instance, _x);
        // string yPredicLabel = classifier.Predict(instance, _x);
        return yPredicted;
    }

    public override void Evaluate(
        string inputModelFileName,
        string reportFileName,
        string positiveClassLabel,
        string groundTruthFileName,
        string predictionsFileName,
        string weightsFileName,
        string calibrationCurveFileName,
        string precisionRecallCurveFileName,
        string rocCurveFileName)
    {
        IBayesPointMachineClassifier<
                IList<Vector>, int, IList<string>, string, IDictionary<string, double>,
                BayesPointMachineClassifierTrainingSettings,
                BinaryBayesPointMachineClassifierPredictionSettings<string>> classifier = null;

        // Validate
        _validate.Evaluate(
            inputModelFileName: inputModelFileName,
            reportFileName: reportFileName,
            groundTruthFileName: groundTruthFileName,
            predictionsFileName: predictionsFileName,
            weightsFileName: weightsFileName,
            calibrationCurveFileName: calibrationCurveFileName,
            precisionRecallCurveFileName: precisionRecallCurveFileName,
            rocCurveFileName: rocCurveFileName);

        // Load model
        if (string.IsNullOrEmpty(inputModelFileName))
        {
            classifier =
                BayesPointMachineClassifier.LoadBinaryClassifier<
                    IList<Vector>, int, IList<string>, string, IDictionary<string, double>>
                    (inputModelFileName);
        }else
        {
            classifier = _classifier;
        }

        EvaluationReportsMapped evaluationReports = new EvaluationReportsMapped(
            classifier,
            _evaluator,
            _x,
            _y,
            _yPredicDistrib,
            _yPredicLabel,
            reportFileName,
            positiveClassLabel,
            groundTruthFileName,
            predictionsFileName,
            weightsFileName,
            calibrationCurveFileName,
            precisionRecallCurveFileName,
            rocCurveFileName);
    }

    public override void DiagnoseTrain(
        string outputModelFileName,
        string reportFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount)
    {
        // Validate
        _validate.DiagnoseTrain(outputModelFileName);

        // Diagnose Train
        DiagnoseTrainMapped diagnoseTrain = new DiagnoseTrainMapped();
        diagnoseTrain.DiagnoseClassifier(
            _x,
            _y,
            _mapping,
            reportFileName,
            outputModelFileName,
            iterationCount,
            computeModelEvidence,
            batchCount);
    }

    public override void CrossValidate(
        string outputModelFileName,
        int crossValidationFoldCount,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount)
    {
        // Validate
        _validate.CrossValidate(outputModelFileName);

        // Cross Validate
        CrossValidateMapped crossValidate = new CrossValidateMapped(
            _x,
            _y,
            _mapping,
            outputModelFileName,
            crossValidationFoldCount,
            iterationCount,
            computeModelEvidence,
            batchCount);
    }

    public override void RemoveFeature()
    {
        _x = null;
        _y = null;
    }
}

