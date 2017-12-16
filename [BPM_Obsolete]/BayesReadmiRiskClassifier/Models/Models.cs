/// <copyright file="Models.cs" company="">
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
using MicrosoftResearch.Infer.Maths;

class Models : IModels
{
	private Models _model = null;
	/// <summary>Sparsity specification for discrete distributions</summary>
	protected Sparsity DenseSparsity;
	/// <summary>Sparsity specification for continuous distributions</summary>
	protected Sparsity ApproxSparsity;

	public Models()
	{
	}

	public void Create(string modellingSetting)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"Model::Create " + modellingSetting + "...", false, true);
		string[] labels = ConstantsModellings.DIC[ConstantsModellings.Keys.BPM][ConstantsModellings.KeysModels.labels];
		double sparsityApproxThresh = System.Convert.ToDouble(ConstantsModellings.DIC[ConstantsModellings.Keys.BPM][ConstantsModellings.KeysModels.SparsityApproxThresh][0]);


		switch (modellingSetting)
		{
			case "BPM":
				_model = new BPM(
					labels: labels,
					sparsityApproxThresh: sparsityApproxThresh);
				break;
			case "BPMMapped":
				_model = new BPMMapped(
					labels: labels);
				break;
			default:
				TraceListeners.Log(TraceEventType.Error, 0,
					"Invalid Model Selected: " + modellingSetting, true, true);
				_model = null;
				break;
		}
	}

	public virtual void InitialiseFeatures(
		DatasetName datasetName,
		object[,] features,
		object[] target)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"InitialiseVariables...", false, true);
		_model.InitialiseFeatures(datasetName, features, target);
	}

	public virtual void Train(
		string outputModelFileName,
		int iterationCount,
		bool computeModelEvidence,
		int batchCount,
		DistributionName distributionName = DistributionName.Null,
		InferenceAlgorithm inferenceEngineAlgorithm = InferenceAlgorithm.Null,
		double noise = 0)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"Train...", false, true);
		_model.Train(
			outputModelFileName,
			iterationCount,
			computeModelEvidence,
			batchCount,
			distributionName,
			inferenceEngineAlgorithm,
			noise);
	}

	public virtual void TrainIncremental(
		string inputModelFileName,
		string outputModelFileName,
		int iterationCount,
		bool computeModelEvidence,
		int batchCount,
		DistributionName distributionName = DistributionName.Null,
		InferenceAlgorithm inferenceEngineAlgorithm = InferenceAlgorithm.Null,
		double noise = 0)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"TrainIncremental...", false, true);
		_model.TrainIncremental(
			inputModelFileName,
			outputModelFileName,
			iterationCount,
			computeModelEvidence,
			batchCount,
			distributionName,
			inferenceEngineAlgorithm,
			noise);
	}

	public virtual void Predict(
		string inputModelFileName,
		DistributionName distributionName = DistributionName.Null,
		InferenceAlgorithm inferenceEngineAlgorithm = InferenceAlgorithm.Null,
		double noise = 0)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"Predict...", false, true);
		_model.Predict(
			inputModelFileName,
			distributionName,
			inferenceEngineAlgorithm, 
			noise);
	}

	public virtual IDictionary<string, double> PredictInstance(
		string inputModelFileName,
		DistributionName distributionName,
		InferenceAlgorithm inferenceEngineAlgorithm,
		int instance,
		double noise = 0)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"PredictInstance...", false, true);
		return _model.PredictInstance(
			inputModelFileName,
			distributionName,
			inferenceEngineAlgorithm,
			instance, 
			noise);
	}

	public virtual void Evaluate(
		string inputModelFileName,
		string reportFileName,
		string positiveClassLabel,
		string groundTruthFileName = "",
		string predictionsFileName = "",
		string weightsFileName = "",
		string calibrationCurveFileName = "",
		string precisionRecallCurveFileName = "",
		string rocCurveFileName = "")
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"Evaluate...", false, true);
		_model.Evaluate(
			inputModelFileName,
			reportFileName,
			positiveClassLabel,
			groundTruthFileName,
			predictionsFileName,
			weightsFileName,
			calibrationCurveFileName,
			precisionRecallCurveFileName,
			rocCurveFileName);
	}

	public virtual void DiagnoseTrain(
		string outputModelFileName,
		string reportFileName,
		int iterationCount,
		bool computeModelEvidence = false,
		int batchCount = 1)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"DiagnoseTrain...", false, true);
		_model.DiagnoseTrain(
			outputModelFileName,
			reportFileName,
			iterationCount,
			computeModelEvidence,
			batchCount);
	}

	public virtual void CrossValidate(
		string outputModelFileName,
		int crossValidationFoldCount,
		int iterationCount,
		bool computeModelEvidence = false,
		int batchCount = 1)
	{
		TraceListeners.Log(TraceEventType.Information, 0,
			"CrossValidate...", false, true);
		_model.CrossValidate(
			outputModelFileName,
			crossValidationFoldCount,
			iterationCount,
			computeModelEvidence,
			batchCount);
	}

	public virtual void RemoveFeature()
	{
		_model.RemoveFeature();
	}
}

