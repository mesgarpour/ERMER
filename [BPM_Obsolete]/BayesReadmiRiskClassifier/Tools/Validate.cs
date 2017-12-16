using System;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Maths;
using System.Diagnostics;
using System.IO;
using MicrosoftResearch.Infer.Models;

class Validate
{
    public void ValidatePredict(
        Vector[] xTrain,
        Vector[] xTest)
    {
        if (xTrain[0].Count() != xTest[0].Count())
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Number of features in train and test do not match!", true, true);
        }
    }

    public void Dataset(
        string[] labels,
        DatasetName datasetName,
        object[,] x,
        object[] y)
    {
        if (x == null ||
            y == null ||
            x.Rank != 2 ||
            x.GetLength(0) == 0 ||
            y.Length == 0)
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Invalid number of records in the " + datasetName.ToString() + " dataset!", true, true);
        }

        if (x.GetLength(1) != y.Count())
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Number of observations in features and target do not match!", true, true);
        }

        if (y.All(v => (string)v != labels[0]) ||
            y.All(v => (string)v != labels[1]))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "All the labels must be present in the " + datasetName.ToString() + " dataset!", true, true);
        }

        if (x.GetLength(1) < x.GetLength(0))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Number of observations is less than number of features!", true, true);
        }
    }


    public void Train(
        string outputModelFileName,
        int iterationCount,
        int batchCount)
    {
        Debug.Assert(outputModelFileName != null, "The model file name must not be null.");
        Debug.Assert(iterationCount > 0, "The iteration count must be greater than zero.");
        Debug.Assert(batchCount > 0, "The batch count must be greater than zero.");

        FilePathExist(outputModelFileName);
    }

    public void TrainIncremental(
        string inputModelFileName,
        string outputModelFileName,
        int iterationCount,
        int batchCount)
    {
        Debug.Assert(inputModelFileName != null, "The model file name must not be null.");
        Debug.Assert(outputModelFileName != null, "The model file name must not be null.");
        Debug.Assert(iterationCount > 0, "The iteration count must be greater than zero.");
        Debug.Assert(batchCount > 0, "The batch count must be greater than zero.");

        FileExist(inputModelFileName);
        FilePathExist(outputModelFileName);
    }

    public void Predict(
        string inputModelFileName)
    {

        FileExist(inputModelFileName);
    }

    public void PredictInstance(
        string inputModelFileName,
        int instance,
        int numObservations)
    {
        Debug.Assert(instance >= 0 && instance < numObservations,
            "The instance number must not be in range.");

        FileExist(inputModelFileName);
    }

    public void Evaluate(
        string inputModelFileName,
        string reportFileName,
        string groundTruthFileName,
        string predictionsFileName,
        string weightsFileName,
        string calibrationCurveFileName,
        string precisionRecallCurveFileName,
        string rocCurveFileName)
    {
        FileExist(inputModelFileName);
        FilePathExist(reportFileName);
        FilePathExist(groundTruthFileName);
        FilePathExist(predictionsFileName);
        FilePathExist(weightsFileName);
        FilePathExist(calibrationCurveFileName);
        FilePathExist(precisionRecallCurveFileName);
        FilePathExist(rocCurveFileName);
    }

    public void DiagnoseTrain(
        string outputModelFileName)
    {
        FilePathExist(outputModelFileName);
    }

    public void CrossValidate(
        string outputModelFileName)
    {
        FilePathExist(outputModelFileName);
    }


    private bool AvailableDataset(
        DatasetName availableDatasetName,
        DatasetName datasetName)
    {
        if (availableDatasetName != datasetName)
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "The requested database is not valid or is not loaed into the model!",
                true, true);
            return false;
        }
        return true;
    }

    private bool FileExist(string fileName)
    {
        if (!string.IsNullOrEmpty(fileName) && !File.Exists(@fileName))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "The following file can not be reached:\n" + fileName,
                true, true);
            return false;
        }
        return true;
    }

    private bool FilePathExist(string fileName)
    {
        string pathName = Path.GetDirectoryName(fileName);

        if (!string.IsNullOrEmpty(pathName) && !Directory.Exists(@pathName))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "The following directory path can not be reached:\n" + pathName,
                true, true);
            return false;
        }
        return true;
    }
}

