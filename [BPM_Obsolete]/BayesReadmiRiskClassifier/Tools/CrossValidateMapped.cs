using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Diagnostics;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Learners;
using MicrosoftResearch.Infer.Learners.Mappings;

class CrossValidateMapped
{
    /// <summary>
    /// CrossValidate diagnosis
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="mapping"></param>
    /// <param name="reportFileName"></param>
    /// <param name="crossValidationFoldCount"></param>
    /// <param name="iterationCount"></param>
    /// <param name="computeModelEvidence"></param>
    /// <param name="batchCount"></param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    public CrossValidateMapped(
        Vector[] x,
        IList<string> y,
        GenericClassifierMapping mapping,
        string reportFileName,
        int crossValidationFoldCount, //folds
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

        // Shuffle dataset
        shuffleVector(x);

        // Create evaluator 
        var evaluatorMapping = mapping.ForEvaluation();
        var evaluator = new ClassifierEvaluator<
               IList<Vector>,       // the type of the instance source,
               int,                 // the type of an instance
               IList<string>,       // the type of the label source
               string>(             // the type of a label.
               evaluatorMapping);


        // Create performance metrics
        var accuracy = new List<double>();
        var negativeLogProbability = new List<double>();
        var auc = new List<double>();
        var evidence = new List<double>();
        var iterationCounts = new List<double>();
        var trainingTime = new List<double>();

        // Run cross-validation
        int validationSetSize = x.Length / crossValidationFoldCount;
        int trainingSetSize = x.Length - validationSetSize;
        int validationFoldSetSize = 0;
        int trainingFoldSetSize = 0;
        Console.WriteLine(
            "Running {0}-fold cross-validation", crossValidationFoldCount);

        if (validationSetSize == 0 || trainingSetSize == 0)
        {
            Console.WriteLine("Invalid number of folds");
            Console.ReadKey();
            System.Environment.Exit(1);

        }

        for (int fold = 0; fold < crossValidationFoldCount; fold++)
        {
            // Construct training and validation sets for fold
            int validationSetStart = fold * validationSetSize;
            int validationSetEnd = (fold + 1 == crossValidationFoldCount)
                                       ? x.Length
                                       : (fold + 1) * validationSetSize;


            validationFoldSetSize = validationSetEnd - validationSetStart;
            trainingFoldSetSize = x.Length - validationFoldSetSize;

            Vector[] trainingSet = new Vector[trainingFoldSetSize];
            Vector[] validationSet = new Vector[validationFoldSetSize];
            IList<string> trainingSetLabels = new List<string>();
            IList<string> validationSetLabels = new List<string>();

            for (int instance = 0, iv = 0, it = 0; instance < x.Length; instance++)
            {
                if (validationSetStart <= instance && instance < validationSetEnd)
                {
                    validationSet[iv++] = x[instance];
                    validationSetLabels.Add(y[instance]);
                }
                else
                {
                    trainingSet[it++] = x[instance];
                    trainingSetLabels.Add(y[instance]);
                }
            }

            // Print info
            Console.WriteLine("   Fold {0} [validation set instances {1} - {2}]", fold + 1, validationSetStart, validationSetEnd - 1);

            // Create classifier
            var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(mapping);
            classifier.Settings.Training.IterationCount = iterationCount;
            classifier.Settings.Training.BatchCount = batchCount;
            classifier.Settings.Training.ComputeModelEvidence = computeModelEvidence;

            int currentIterationCount = 0;
            classifier.IterationChanged += (sender, eventArgs) => { currentIterationCount = eventArgs.CompletedIterationCount; };

            // Train classifier
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            classifier.Train(trainingSet, trainingSetLabels);
            stopWatch.Stop();

            // Produce predictions
            IEnumerable<IDictionary<string, double>> predictions =
                classifier.PredictDistribution(validationSet);
            var predictedLabels = classifier.Predict(validationSet);

            // Iteration count
            iterationCounts.Add(currentIterationCount);

            // Training time
            trainingTime.Add(stopWatch.ElapsedMilliseconds);

            // Compute accuracy
            accuracy.Add(1 - (evaluator.Evaluate(validationSet, validationSetLabels, predictedLabels, Metrics.ZeroOneError) / predictions.Count()));

            // Compute mean negative log probability
            negativeLogProbability.Add(evaluator.Evaluate(validationSet, validationSetLabels, predictions, Metrics.NegativeLogProbability) / predictions.Count());

            // Compute M-measure (averaged pairwise AUC)
            auc.Add(evaluator.AreaUnderRocCurve(validationSet, validationSetLabels, predictions));

            // Compute log evidence if desired
            evidence.Add(computeModelEvidence ? classifier.LogModelEvidence : double.NaN);

            // Persist performance metrics
            Console.WriteLine(
                "      Accuracy = {0,5:0.0000}   NegLogProb = {1,5:0.0000}   AUC = {2,5:0.0000}{3}   Iterations = {4}   Training time = {5}",
                accuracy[fold],
                negativeLogProbability[fold],
                auc[fold],
                computeModelEvidence ? string.Format("   Log evidence = {0,5:0.0000}", evidence[fold]) : string.Empty,
                iterationCounts[fold],
                FormatElapsedTime(trainingTime[fold]));

            SavePerformanceMetrics(
                reportFileName, accuracy, negativeLogProbability, auc, evidence, iterationCounts, trainingTime);
        }

    }


    /// <summary>
    /// Shuffle input Vector 
    /// </summary>
    /// <param name="input">A vector</param>
    /// <returns>A vector</returns>
    private Vector[] shuffleVector(Vector[] input)
    {
        Random rnd = new Random();
        Vector[] output = new Vector[input.Length];
        int[] indices = Enumerable.Range(0, input.Length).ToArray();
        
        // shuffle
        indices = indices.OrderBy(v => rnd.Next()).ToArray();
        for (int i = 0; i < input.Length; i++)
            output[i] = input[indices[i]];

        return output;
    }


    /// <summary>
    /// Converts elapsed time in milliseconds into a human readable format.
    /// </summary>
    /// <param name="elapsedMilliseconds">The elapsed time in milliseconds.</param>
    /// <returns>A human readable string of specified time.</returns>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private static string FormatElapsedTime(double elapsedMilliseconds)
    {
        TimeSpan time = TimeSpan.FromMilliseconds(elapsedMilliseconds);

        string formattedTime = time.Hours > 0 ? string.Format("{0}:", time.Hours) : string.Empty;
        formattedTime += time.Hours > 0 ? string.Format("{0:D2}:", time.Minutes) : time.Minutes > 0 ? string.Format("{0}:", time.Minutes) : string.Empty;
        formattedTime += time.Hours > 0 || time.Minutes > 0 ? string.Format("{0:D2}.{1:D3}", time.Seconds, time.Milliseconds) : string.Format("{0}.{1:D3} seconds", time.Seconds, time.Milliseconds);

        return formattedTime;
    }


    /// <summary>
    /// Saves the performance metrics to a file with the specified name.
    /// </summary>
    /// <param name="fileName">The name of the file to save the metrics to.</param>
    /// <param name="accuracy">The accuracy.</param>
    /// <param name="negativeLogProbability">The mean negative log probability.</param>
    /// <param name="auc">The AUC.</param>
    /// <param name="evidence">The model's log evidence.</param>
    /// <param name="iterationCount">The number of training iterations.</param>
    /// <param name="trainingTime">The training time in milliseconds.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private static void SavePerformanceMetrics(
        string fileName,
        ICollection<double> accuracy,
        IEnumerable<double> negativeLogProbability,
        IEnumerable<double> auc,
        IEnumerable<double> evidence,
        IEnumerable<double> iterationCount,
        IEnumerable<double> trainingTime)
    {
        using (var writer = new StreamWriter(fileName))
        {
            // Write header
            for (int fold = 0; fold < accuracy.Count; fold++)
            {
                if (fold == 0)
                {
                    writer.Write("# ");
                }

                writer.Write("Fold {0}, ", fold + 1);
            }

            writer.WriteLine("Mean, Standard deviation");
            writer.WriteLine();

            // Write metrics
            SaveSinglePerformanceMetric(writer, "Accuracy", accuracy);
            SaveSinglePerformanceMetric(writer, "Mean negative log probability", negativeLogProbability);
            SaveSinglePerformanceMetric(writer, "AUC", auc);
            SaveSinglePerformanceMetric(writer, "Log evidence", evidence);
            SaveSinglePerformanceMetric(writer, "Training time", trainingTime);
            SaveSinglePerformanceMetric(writer, "Iteration count", iterationCount);
        }

    }


    /// <summary>
    /// Writes a single performance metric to the specified writer.
    /// </summary>
    /// <param name="writer">The writer to write the metrics to.</param>
    /// <param name="description">The metric description.</param>
    /// <param name="metric">The metric.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private static void SaveSinglePerformanceMetric(
        TextWriter writer, 
        string description, 
        IEnumerable<double> metric)
    {
        // Write description
        writer.WriteLine("# " + description);

        // Write metric
        var mva = new MeanVarianceAccumulator();
        foreach (double value in metric)
        {
            writer.Write("{0}, ", value);
            mva.Add(value);
        }

        writer.WriteLine("{0}, {1}", mva.Mean, Math.Sqrt(mva.Variance));
        writer.WriteLine();
    }
}


