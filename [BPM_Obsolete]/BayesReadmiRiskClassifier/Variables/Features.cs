/// <copyright file="Features.cs" company="">
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


class Features
{
    private Dictionary<DatasetName, string[,]> _datasets = null;
    private Dictionary<DatasetName, object[,]> _featuresConverted = null;
    private Dictionary<DatasetName, object[]> _targetConverted = null;
    private Dictionary<DatasetName, Dictionary<string, int>> _featuresKey = null;
    private Dictionary<string, string> _featuresTypes = null;
    private ReadCsvMetdata _varsMetadata;
    private string _varsMetadataColType = null;
    private string _varsMetadataLabelName = null;

    public Features(
        string varsMetadataPath,
        string varsMetadataColKey,
        string varsMetadataColType,
        string varsMetadataLabelName)
    {
        _varsMetadata = new ReadCsvMetdata(varsMetadataPath, varsMetadataColKey);
        _varsMetadataColType = varsMetadataColType;
        _varsMetadataLabelName = varsMetadataLabelName;

        _datasets = new Dictionary<DatasetName, string[,]>();
        _featuresConverted = new Dictionary<DatasetName, object[,]>();
        _targetConverted = new Dictionary<DatasetName, object[]>();
        _featuresTypes = new Dictionary<string, string>();
        _featuresKey = new Dictionary<DatasetName, Dictionary<string, int>>();
    }

    public void RemoveFeatureConverted(DatasetName datasetName)
    {
        if (_featuresConverted.ContainsKey(datasetName) &
            _targetConverted.ContainsKey(datasetName))
        {
            _featuresConverted.Remove(datasetName);
            _targetConverted.Remove(datasetName);

            // Reclaim the memory reserved
            if (_featuresConverted.Count == 0)
            {
                _featuresConverted.Clear();
                _featuresConverted = null;
                _featuresConverted = new Dictionary<DatasetName, object[,]>();
                _targetConverted.Clear();
                _targetConverted = null;
                _targetConverted = new Dictionary<DatasetName, object[]>();
            }
        }
    }

    public void RemoveFeature(DatasetName datasetName)
    {
        if (_datasets.ContainsKey(datasetName))
        {
            _datasets.Remove(datasetName);

            // Reclaim the memory reserved
            if (_datasets.Count == 0)
            {
                _datasets.Clear();
                _datasets = null;
                _datasets = new Dictionary<DatasetName, string[,]>();
            }
        }
    }

    public void Set(
        Dictionary<string, List<string>> dataset,
        DatasetName datasetName,
        string[] featuresByOrder)
    {
        TraceListeners.Log(TraceEventType.Information, 0, "Features::Set...", false, true);
        int numFeatures = dataset.Keys.Count;
        int numObservations = dataset[dataset.Keys.First()].Count;

        // Evaluate
        if (_datasets.ContainsKey(datasetName))
        {
            TraceListeners.Log(TraceEventType.Warning, 0,
                "Overwriting: " + datasetName.ToString(), false, true);
        }

        // Set
        try
        {
            _featuresKey[datasetName] = new Dictionary<string, int>();
            _datasets[datasetName] = new string[numFeatures, numObservations];
            int i = 0;
            foreach (string k in featuresByOrder)
            {
                _featuresKey[datasetName][k] = i;
                for (int j = 0; j < numObservations; j++)
                {
                    _datasets[datasetName][i, j] = dataset[k][j];
                }
                i++;
            }
        }
        catch (Exception e)
        {
            TraceListeners.Log(TraceEventType.Error, 0, e.ToString(), true, true);
        }
    }

    public string[,] GetDataset(DatasetName datasetName)
    {
        // Evaluate
        if (!_datasets.ContainsKey(datasetName))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Dataset does not exist: " + datasetName.ToString(), true, true);
        }

        // Get
        return _datasets[datasetName];
    }

    public object[,] GetFeaturesConverted(DatasetName datasetName)
    {
        // Evaluate
        if (!_featuresConverted.ContainsKey(datasetName))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Dataset does not exist: " + datasetName.ToString(), true, true);
        }

        // Get
        return _featuresConverted[datasetName];
    }

    public object[] GetTargetConverted(DatasetName datasetName)
    {
        // Evaluate
        if (!_targetConverted.ContainsKey(datasetName))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Dataset does not exist: " + datasetName.ToString(), true, true);
        }

        // Get
        return _targetConverted[datasetName];
    }

    public string GetVariableType(string varName)
    {
        // Evaluate
        if (!_featuresTypes.ContainsKey(varName))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Feature does not exist: " + varName, true, true);
        }

        // Get
        return _featuresTypes[varName];
    }

    private void SetFeaturesMetdata()
    {
        string typeString;
        foreach (string featureName in _featuresKey[_featuresKey.Keys.First()].Keys)
        {
            typeString = _varsMetadata.Get(featureName, _varsMetadataColType);

            // Evaluate
            if (typeString == null)
            {
                TraceListeners.Log(TraceEventType.Error, 0,
                    "No Valid metadata for variable: " + featureName, true, true);
            }
            // Set
            else
            {
                _featuresTypes[featureName] = typeString.Split('(')[0];
            }
        }
    }

    public void ConvertFeatures(
        DatasetName datasetName,
        bool standardConversion = false)
    {
        TraceListeners.Log(TraceEventType.Information, 0,
            "Features::ConvertFeatures...", false, true);
        // Evaluate
        if (!_datasets.ContainsKey(datasetName))
        {
            TraceListeners.Log(TraceEventType.Error, 0,
                "Dataset does not exist: " + datasetName.ToString(), true, true);
        }
        if (_featuresConverted.ContainsKey(datasetName))
        {
            TraceListeners.Log(TraceEventType.Warning, 0,
                "Overwriting: " + datasetName.ToString(), false, true);
        }

        // Initialise
        string featureType = null;
        int numFeatures = _datasets[datasetName].GetLength(0) - 1;
        int numObservations = _datasets[datasetName].GetLength(1);
        SetFeaturesMetdata();

        // Convert features
        featureType = (standardConversion) ? "double" : null;
        _featuresConverted[datasetName] = new object[numFeatures, numObservations];
        string featureName;
        for (int i = 0, iOrig = 0; iOrig < numFeatures + 1; i++, iOrig++)
        {
            featureName = _featuresKey[datasetName].FirstOrDefault(v => v.Value == iOrig).Key;

            // Skip the label
            if (featureName == _varsMetadataLabelName)
            {
                i--;
                continue;
            }

            // Convert observations
            TraceListeners.Log(TraceEventType.Information, 0,
                " ...Convert: " + featureName + " to " +
                featureType + " ...", false, true);
            for (int j = 0; j < numObservations; j++)
            {
                _featuresConverted[datasetName][i, j] =
                    Convert(datasetName, featureName, iOrig, j, featureType);
            }
        }

        // Convert the label variable to labelType
        featureType = (standardConversion) ? "string" : null;
        _targetConverted[datasetName] = new object[numObservations];
        {
            int i = _featuresKey[datasetName][_varsMetadataLabelName];
            for (int j = 0; j < numObservations; j++)
            {
                _targetConverted[datasetName][j] =
                Convert(datasetName, _varsMetadataLabelName, i, j, featureType);
            }
        }
    }

    private object Convert(
        DatasetName datasetName,
        string featureName,
        int featureIndex,
        int observationIndex,
        string featureType = null)
    {
        object output = null;
        if (featureType == null)
        {
            featureType = _featuresTypes[featureName];
        }

        switch (featureType.ToLower())
        {
            case "bool":
                output = (System.Convert.ToInt64(
                    _datasets[datasetName][featureIndex, observationIndex]) > 0)
                    ? true : false;
                break;
            case "double":
                output = System.Convert.ToDouble(
                    _datasets[datasetName][featureIndex, observationIndex]);
                break;
            case "int":
                output = System.Convert.ToInt32(
                    _datasets[datasetName][featureIndex, observationIndex]);
                break;
            case "date":
            // break;
            case "string":
                output = _datasets[datasetName][featureIndex, observationIndex];
                break;
            default:
                TraceListeners.Log(TraceEventType.Error, 0,
                    "Invalid variable type for " + featureName +
                    " in the metadata specification", true, true);
                break;
        }
        return output;
    }
}

