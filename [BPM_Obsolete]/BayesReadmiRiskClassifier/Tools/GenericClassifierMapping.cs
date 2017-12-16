using System;
using System.Collections.Generic;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Learners.Mappings;

/// <summary>
/// A mapping for the Bayes Point Machine classifier tutorial.
/// </summary>
[Serializable]
public class GenericClassifierMapping
    : IClassifierMapping<IList<Vector>, int, IList<string>, string, Vector>
{
    private string[] _labels = null;

    public GenericClassifierMapping(string[] labels = null)
    {
        if(labels == null)
        {
            labels = new string[2] { "True", "False" };
        }

        SetClassLabels(labels);
    }

    public void SetClassLabels(string[] labels)
    {
        _labels = labels;
    }

    public IEnumerable<int> GetInstances(IList<Vector> featureVectors)
    {
        for (int instance = 0; instance < featureVectors.Count; instance++)
        {
            yield return instance;
        }
    }

    public Vector GetFeatures(int instance, IList<Vector> featureVectors)
    {
        return featureVectors[instance];
    }

    public string GetLabel(
        int instance, IList<Vector> featureVectors, IList<string> labels)
    {
        return labels[instance];
    }

    public IEnumerable<string> GetClassLabels(
        IList<Vector> featureVectors = null, IList<string> labels = null)
    {
        return _labels;
    }
}
