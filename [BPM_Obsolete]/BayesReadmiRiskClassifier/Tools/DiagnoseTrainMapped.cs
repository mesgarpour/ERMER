using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Learners;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions;

class DiagnoseTrainMapped
{
    /// <summary>
    /// Diagnoses the Bayes point machine classifier on the specified data set.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="mapping"></param>
    /// <param name="reportFileName">The name of the file to store the maximum parameter differences.</param>
    /// <param name="outputModelFileName">The name of the file to store the trained Bayes point machine model.</param>
    /// <param name="iterationCount"></param>
    /// <param name="computeModelEvidence"></param>
    /// <param name="batchCount"></param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    public void DiagnoseClassifier(
        Vector[] x,
        IList<string> y,
        GenericClassifierMapping mapping,
        string outputModelFileName,
        string reportFileName,
        int iterationCount,
        bool computeModelEvidence,
        int batchCount)
    {
        Debug.Assert(x != null, "The feature vector must not be null.");
        Debug.Assert(y != null, "The targe variable must not be null.");
        Debug.Assert(mapping != null, "The classifier map must not be null.");
        Debug.Assert(!string.IsNullOrEmpty(reportFileName), "The report file name must not be null/empty.");
        Debug.Assert(iterationCount > 0, "The iteration count must be greater than zero.");
        Debug.Assert(batchCount > 0, "The batch count must be greater than zero.");

        // create a BPM from the mapping
        var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(mapping);
        classifier.Settings.Training.ComputeModelEvidence = computeModelEvidence;
        classifier.Settings.Training.IterationCount = iterationCount;
        classifier.Settings.Training.BatchCount = batchCount;

        // Create prior distributions over weights
        Dictionary<int, double[]> maxMean;
        Dictionary<int, double[]> maxVar;
        int classCount = 2;
        int featureCount = x.Length;
        var priorWeightDistributions = Util.ArrayInit(classCount, c => Util.ArrayInit(featureCount, f => new Gaussian(0.0, 1.0)));

        // Create IterationChanged handler
        var watch = new Stopwatch();
        classifier.IterationChanged += (sender, eventArgs) =>
        {
            watch.Stop();
            double maxParameterChange = MaxDiff(eventArgs.WeightPosteriorDistributions, priorWeightDistributions, out maxMean, out maxVar);

            if (!string.IsNullOrEmpty(reportFileName))
            {
                SaveMaximumParameterDifference(
                    reportFileName,
                    eventArgs.CompletedIterationCount,
                    maxParameterChange,
                    watch.ElapsedMilliseconds,
                    maxMean,
                    maxVar);
            }

            Console.WriteLine(
                "[{0}] Iteration {1,-4}   dp = {2,-20}   dt = {3,5}ms",
                DateTime.Now.ToLongTimeString(),
                eventArgs.CompletedIterationCount,
                maxParameterChange,
                watch.ElapsedMilliseconds);

            // Copy weight marginals
            for (int c = 0; c < eventArgs.WeightPosteriorDistributions.Count; c++)
            {
                for (int f = 0; f < eventArgs.WeightPosteriorDistributions[c].Count; f++)
                {
                    priorWeightDistributions[c][f] = eventArgs.WeightPosteriorDistributions[c][f];
                }
            }

            watch.Restart();
        };

        // Write file header
        if (!string.IsNullOrEmpty(reportFileName))
        {
            using (var writer = new StreamWriter(reportFileName))
            {
                writer.WriteLine("# time, # iteration, "+
                    "# maximum absolute parameter difference, "+
                    "# iteration time in milliseconds, "+
                    "# Max Mean, # Max Var.");
            }
        }

        // Train the Bayes point machine classifier
        Console.WriteLine("[{0}] Starting training...", DateTime.Now.ToLongTimeString());
        watch.Start();

        classifier.Train(x, y);

        // Compute evidence
        if (classifier.Settings.Training.ComputeModelEvidence)
        {
            Console.WriteLine("Log evidence = {0,10:0.0000}", classifier.LogModelEvidence);
        }

        // Save trained model
        if (!string.IsNullOrEmpty(outputModelFileName))
        {
            classifier.Save(outputModelFileName);
        }
    }


    /// <summary>
    /// Saves the maximum absolute difference between two given weight distributions to a file with the specified name.
    /// </summary>
    /// <param name="fileName">The name of the file to save the maximum absolute difference between weight distributions to.</param>
    /// <param name="iteration">The inference algorithm iteration.</param>
    /// <param name="maxParameterChange">The maximum absolute difference in any parameter of two weight distributions.</param>
    /// <param name="elapsedMilliseconds">The elapsed milliseconds.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private static void SaveMaximumParameterDifference(
        string fileName,
        int iteration,
        double maxParameterChange,
        long elapsedMilliseconds,
        Dictionary<int, double[]> maxMean,
        Dictionary<int, double[]> maxVar)
    {
        using (var writer = new StreamWriter(fileName, true))
        {
            writer.WriteLine("{0}, {1}, {2}, {3}, {4}, {5}",
                DateTime.Now.ToLongTimeString(),
                iteration,
                maxParameterChange,
                elapsedMilliseconds,
                string.Join(";", maxMean[0]),
                string.Join(";", maxVar[0]));
        }
    }


    /// <summary>
    /// Computes the maximum difference in any parameter of two Gaussian distributions.
    /// </summary>
    /// <param name="first">The first Gaussian.</param>
    /// <param name="second">The second Gaussian.</param>
    /// <returns>The maximum absolute difference in any parameter.</returns>
    /// <remarks>This difference computation is based on mean and variance instead of mean*precision and precision.</remarks>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private static double MaxDiff(
        IReadOnlyList<IReadOnlyList<Gaussian>> first,
        Gaussian[][] second,
        out Dictionary<int, double[]> maxMean,
        out Dictionary<int, double[]> maxVar)
    {
        int classCount = first.Count;
        int featureCount = first[0].Count;
        double maxDiff = double.NegativeInfinity;
        maxMean = new Dictionary<int, double[]>();
        maxVar = new Dictionary<int, double[]>();

        for (int c = 0; c < classCount; c++)
        {
            maxMean[c] = new double[featureCount];
            maxVar[c] = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                double firstMean, firstVariance, secondMean, secondVariance;
                first[c][f].GetMeanAndVariance(out firstMean, out firstVariance);
                second[c][f].GetMeanAndVariance(out secondMean, out secondVariance);
                double meanDifference = Math.Abs(firstMean - secondMean);
                double varianceDifference = Math.Abs(firstVariance - secondVariance);
                maxMean[c][f] = Math.Abs(meanDifference);
                maxVar[c][f] = Math.Abs(varianceDifference);

                if (meanDifference > maxDiff)
                {
                    maxDiff = Math.Abs(meanDifference);
                }

                if (Math.Abs(varianceDifference) > maxDiff)
                {
                    maxDiff = Math.Abs(varianceDifference);
                }
            }
        }

        return maxDiff;
    }
}

