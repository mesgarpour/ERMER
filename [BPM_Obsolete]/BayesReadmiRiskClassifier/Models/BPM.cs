using System;
using System.Collections.Generic;
using System.Diagnostics;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Learners;
using MicrosoftResearch.Infer.Learners.Mappings;
using MicrosoftResearch.Infer.Distributions;
using System.Linq;

class BPM : Models, IModels
{
    /// <summary>Model's feature matrix</summary>
    private VariableArray<Vector> _x = null;

    /// <summary>Model's target vector</summary>
    private VariableArray<bool> _y = null;

    /// <summary>Model's inference engines</summary>
    private Dictionary<DistributionType, InferenceEngine> _engine = null;

    /// <summary>Model's weight vectors</summary>
    private Dictionary<DistributionType, Variable<Vector>> _w = null;

    /// <summary>Distribution types of the features for the prior and posterior</summary>
    private Dictionary<DistributionType, DistributionName> _d = null;

    /// <summary>Predicted distributions</summary>  
    IEnumerable<IDictionary<string, double>> _yPredicDistrib = null;

    /// <summary>Predicted labels</summary>  
    IEnumerable<string> _yPredicLabel = null;

    /// <summary>The classifier evaluatior instance</summary>
    private ClassifierEvaluator<
        IList<Vector>,       // the type of the instance source,
        int,                 // the type of an instance
        IList<string>,       // the type of the label source
        string              // the type of a label.
        > _evaluator;

    /// <summary>The classifier mapping instance</summary>
    private GenericClassifierMapping _mapping = null;

    /// <summary>Number of observations</summary>
    private int _numObservations;

    /// <summary>Number of features</summary>
    private int _numFeatures;

    /// <summary>The name of the current saved dataset</summary>
    private DatasetName _availableDatasetName;

    /// <summary>The validation instance</summary>
    private Validate _validate;

    private static readonly double _cutoffPoint = 0.5;


    public BPM(
        string[] labels,
        double sparsityApproxThresh)
    {
        Debug.Assert(labels != null, "The labels must not be null.");
        Debug.Assert(labels.Length == 2, "The labels must have two possible values.");
        Debug.Assert(sparsityApproxThresh >= 0, "The sparsityApproxThresh must be greater than or equal to zero.");

        // Initialise the validations
        _validate = new Validate();

        // Initialise the BPM
        _engine = new Dictionary<DistributionType, InferenceEngine>();
        _w = new Dictionary<DistributionType, Variable<Vector>>();
        _w[DistributionType.Prior] = null;
        _w[DistributionType.Posterior] = null;
        _d = new Dictionary<DistributionType, DistributionName>();
        _yPredicDistrib = Enumerable.Empty<IDictionary<string, double>>();
        _yPredicLabel = new string[] { };

        _mapping = new GenericClassifierMapping(labels);
        // TO DO

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
        Vector[] xV = null;
        Range r = null;

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
        _y = Variable.Observed(
            Array.ConvertAll(y, v => (
            System.Convert.ToInt64(v) > 0) ? true : false)).Named(
            "y." + datasetName.ToString());

        // Set features
        xV = new Vector[_numObservations];
        r = _y.Range.Named("person");
        for (int i = 0; i < _numObservations; i++)
        {
            xV[i] = Vector.FromArray(xTrans[i]);
        }
        _x = Variable.Observed(xV, r).Named("x." + datasetName.ToString());

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
            "batchCount.", false, true);

        // Validate
        _validate.Train(
            outputModelFileName: outputModelFileName,
            iterationCount: iterationCount,
            batchCount: batchCount);

        // The inference engine
        _engine[DistributionType.Prior] = SetInferenceEngine(
            inferenceEngineAlgorithm, iterationCount);

        // Initialise prior weights
        _w[DistributionType.Prior] = InitialiseWeights(
            distributionType: DistributionType.Prior,
            distributionName: distributionName,
            dimension: _numFeatures,
            hyperParameters: null);

        // BPM
        BayesPointMachine(_x, _y, _w[DistributionType.Prior], noise);
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
            "batchCount.", false, true);

        // Validate
        _validate.TrainIncremental(
            inputModelFileName: inputModelFileName,
            outputModelFileName: outputModelFileName,
            iterationCount: iterationCount,
            batchCount: batchCount);

        // The inference engine
        _engine[DistributionType.Prior] = SetInferenceEngine(
            inferenceEngineAlgorithm, iterationCount);

        // weights
        if (!_w[DistributionType.Prior].IsDefined)
        {// Initialise prior weights
            _w[DistributionType.Prior] = InitialiseWeights(
                distributionType: DistributionType.Prior,
                distributionName: distributionName,
                dimension: _numFeatures,
                hyperParameters: null);
        }
        else
        {// Infer prior weights
            _w[DistributionType.Prior] = InferWeights(
                distributionType: DistributionType.Prior,
                distributionName: distributionName,
                hyperParameters: null);
        }

        // BPM
        BayesPointMachine(_x, _y, _w[DistributionType.Prior], noise);
    }

    public override void Predict(
        string inputModelFileName,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        double noise)
    {
        // Validate
        // _validate.Predict(inputModelFileName);

        // Initialise
        List<IDictionary<string, double>> yPredicDistrib = new List<IDictionary<string, double>>();
        List<string> yPredicLabel = new List<string>();
        VariableArray<bool> yTest = Variable.Array<bool>(new Range(_numObservations)).
            Named("y." + _availableDatasetName.ToString()); // to do: correct the size
        Bernoulli[] yPredic;

        // The inference engine
        _engine[DistributionType.Posterior] = SetInferenceEngine(
            inferenceEngineAlgorithm, 1);

        // Infer postrior weights from training
        _w[DistributionType.Posterior] = InferWeights(
            distributionType: DistributionType.Posterior,
            distributionName: distributionName,
            hyperParameters: null);

        // BPM
        BayesPointMachine(_x, yTest, _w[DistributionType.Posterior], noise);

        // predict
        // _yPredicLabel = _engine[DistributionType.Posterior].Infer(_y); 
        yPredic = _engine[DistributionType.Posterior].Infer<Bernoulli[]>(yTest);
        for (int i = 0; i < yPredic.Length; i++)
        {
            yPredicDistrib.Add(new Dictionary<string, double>(){
                {Convert.ToInt32(yPredic[i].GetProbFalse() > _cutoffPoint).ToString(), yPredic[i].GetProbFalse()},
                {Convert.ToInt32(yPredic[i].GetProbTrue() > _cutoffPoint).ToString(), yPredic[i].GetProbTrue()}});
            yPredicLabel.Add(Convert.ToInt32(yPredic[i].GetProbTrue() > _cutoffPoint).ToString());
        }
        _yPredicDistrib = yPredicDistrib;
        _yPredicLabel = yPredicLabel;

    }

    public override IDictionary<string, double> PredictInstance(
        string inputModelFileName,
        DistributionName distributionName,
        InferenceAlgorithm inferenceEngineAlgorithm,
        int instance,
        double noise)
    {
        // Validate
        // _validate.Predict(inputModelFileName);

        // Initialise
        Dictionary<string, double> yPredicDistrib = new Dictionary<string, double>();
        VariableArray<bool> yTest = Variable.Array<bool>(new Range(1)).
            Named("y." + _availableDatasetName.ToString()); // to do: correct the size
        Bernoulli yPredic;

        // The inference engine
        _engine[DistributionType.Posterior] = SetInferenceEngine(
            inferenceEngineAlgorithm, 1);

        // Infer postrior weights from training
        _w[DistributionType.Posterior] = InferWeights(
            distributionType: DistributionType.Posterior,
            distributionName: distributionName,
            hyperParameters: null);

        // BPM
        BayesPointMachine(_x[instance], yTest[0], _w[DistributionType.Posterior], noise);

        // predict
        // _yPredicLabel = _engine[DistributionType.Posterior].Infer(_y); 
        yPredic = _engine[DistributionType.Posterior].Infer<Bernoulli>(yTest[0]);
        yPredicDistrib = new Dictionary<string, double>(){
                {Convert.ToInt32(yPredic.GetProbFalse() > _cutoffPoint).ToString(), yPredic.GetProbFalse()},
                {Convert.ToInt32(yPredic.GetProbTrue() > _cutoffPoint).ToString(), yPredic.GetProbTrue()}};

        return yPredicDistrib;
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
        // Validate
        /* _validate.Evaluate(
            inputModelFileName: inputModelFileName,
            reportFileName: reportFileName,
            groundTruthFileName: groundTruthFileName,
            predictionsFileName: predictionsFileName,
            weightsFileName: weightsFileName,
            calibrationCurveFileName: calibrationCurveFileName,
            precisionRecallCurveFileName: precisionRecallCurveFileName,
            rocCurveFileName: rocCurveFileName);*/


        EvaluationReports evaluationReports = new EvaluationReports(
            this,
            _evaluator,
            _x.ObservedValue,
            Array.ConvertAll(_y.ObservedValue, v => Convert.ToInt32(v).ToString()),
            _yPredicDistrib,
            _yPredicLabel,
            reportFileName,
            positiveClassLabel,
            groundTruthFileName,
            predictionsFileName,
            weightsFileName,
            calibrationCurveFileName,
            precisionRecallCurveFileName,
            rocCurveFileName); // To do: continue from here
    }

    public override void DiagnoseTrain(
        string outputModelFileName,
        string reportFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount)
    {

    }

    public override void CrossValidate(
        string outputModelFileName,
        int crossValidationFoldCount,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount)
    {

    }

    public override void RemoveFeature()
    {
        _x = null;
        _y = null;
    }



    private Variable<Vector> InitialiseWeights(
        DistributionType distributionType,
        DistributionName distributionName,
        int dimension,
        string[] hyperParameters)
    {
        switch (distributionName)
        {
            case DistributionName.GaussianDefault:
                return Variable.Random(new VectorGaussian(
                    Vector.Zero(dimension),
                    PositiveDefiniteMatrix.Identity(dimension))).Named(
                    "w." + distributionType.ToString());
            case DistributionName.GaussianInit:
                return Variable<Vector>.Random(
                    Variable.New<VectorGaussian>().Named(
                    "w." + distributionType.ToString()));
            default:
                TraceListeners.Log(TraceEventType.Error, 0,
                    "Invalid distribution name: " + distributionName.ToString(), true, true);
                return null;
        }
    }

    private Variable<Vector> InferWeights(
        DistributionType distributionType,
        DistributionName distributionName,
       string[] hyperParameters)
    {
        switch (distributionName)
        {
            case DistributionName.GaussianDefault:
                VectorGaussian wObserved = _engine[distributionType].Infer<VectorGaussian>(_w[DistributionType.Prior]);
                return Variable.Random(wObserved).Named("w." + distributionType.ToString());
            default:
                TraceListeners.Log(TraceEventType.Error, 0,
                    "Invalid distribution name: " + distributionName.ToString(), true, true);
                return null;
        }
    }

    private InferenceEngine SetInferenceEngine(
        InferenceAlgorithm inferenceEngineAlgorithm,
        int iterationCount,
        bool ShowProgress = true,
        bool ShowTimings = true)
    {
        InferenceEngine engine = null;

        switch (inferenceEngineAlgorithm)
        {
            case InferenceAlgorithm.EP:
                engine = new InferenceEngine(new ExpectationPropagation());
                engine.ShowProgress = true;
                engine.ShowTimings = true;
                engine.NumberOfIterations = iterationCount;
                break;
            default:
                TraceListeners.Log(TraceEventType.Error, 0,
                    "Invalid inference algorithm: " + inferenceEngineAlgorithm, true, true);
                break;
        }
        return engine;
    }

    // Derive new y
    private void BayesPointMachine(
        VariableArray<Vector> x,
        VariableArray<bool> y,
        Variable<Vector> w,
        double noise)
    {
        Range r = y.Range.Named("person");
        y[r] = Variable.GaussianFromMeanAndVariance(
            Variable.InnerProduct(w, x[r]).Named("innerproduct"),
            noise) > 0;
    }

    private void BayesPointMachine(
        Variable<Vector> x,
        Variable<bool> y,
        Variable<Vector> w,
        double noise)
    {
        y = Variable.GaussianFromMeanAndVariance(
            Variable.InnerProduct(w, x).Named("innerproduct"),
            noise) > 0;
    }
}

