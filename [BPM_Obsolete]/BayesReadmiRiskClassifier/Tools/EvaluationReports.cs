using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Learners;
using MicrosoftResearch.Infer.Learners.Runners;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


class EvaluationReports
{
    public EvaluationReports(
        BPM classifier,
        ClassifierEvaluator<IList<Vector>, int, IList<string>, string> evaluator,
        Vector[] x,
        IList<string> y,
        IEnumerable<IDictionary<string, double>> yPredicDistrib,
        IEnumerable<string> yPredicLabel,
        string reportFileName,
        string positiveClassLabel,
        string groundTruthFileName = "",
        string predictionsFileName = "",
        string weightsFileName = "",
        string calibrationCurveFileName = "",
        string precisionRecallCurveFileName = "",
        string rocCurveFileName = "")
    {
        Debug.Assert(classifier != null, "The classifier must not be null.");
        Debug.Assert(evaluator != null, "The evaluator must not be null.");
        Debug.Assert(x != null, "The feature vector must not be null.");
        Debug.Assert(y != null, "The targe variable must not be null.");
        Debug.Assert(yPredicDistrib != null, "The predictive distribution must not be null.");
        Debug.Assert(yPredicLabel != null, "The predicted labels must not be null.");
        Debug.Assert(!string.IsNullOrEmpty(reportFileName), "The report file name must not be null/empty.");
        Debug.Assert(!string.IsNullOrEmpty(positiveClassLabel), "The positive class label must not be null/empty.");

        // Write evaluation report header information
        if (!string.IsNullOrEmpty(reportFileName))
        {
            using (var writer = new StreamWriter(reportFileName))
            {
                this.WriteReportHeader(writer, groundTruthFileName, predictionsFileName);
                this.WriteReport(writer, evaluator, x, y, yPredicDistrib, yPredicLabel);
            }
        }

        // Write the prediction distribution for all labels
        if (!string.IsNullOrEmpty(predictionsFileName))
        {
            SaveLabelDistributions(predictionsFileName, yPredicDistrib);
        }

        // Compute and write the empirical probability calibration curve
        if (!string.IsNullOrEmpty(calibrationCurveFileName))
        {
            this.WriteCalibrationCurve(calibrationCurveFileName, evaluator, x, y, yPredicDistrib, positiveClassLabel);
        }

        // Compute and write the precision-recall curve
        if (!string.IsNullOrEmpty(precisionRecallCurveFileName))
        {
            this.WritePrecisionRecallCurve(precisionRecallCurveFileName, evaluator, x, y, yPredicDistrib, positiveClassLabel);
        }

        // Compute and write the receiver operating characteristic curve
        if (!string.IsNullOrEmpty(rocCurveFileName))
        {
            this.WriteRocCurve(rocCurveFileName, evaluator, x, y, yPredicDistrib, positiveClassLabel);
        }
        // Compute and write the weights
        if (!string.IsNullOrEmpty(weightsFileName))
        {
            // this.SampleWeights(weightsFileName, classifier);
        }
    }

    /// <summary>
    /// Writes the evaluation results to a file with the specified name.
    /// </summary>
    /// <param name="writer">The name of the file to write the report to.</param>
    /// <param name="evaluator">The classifier evaluator.</param>
    /// <param name="x">The x vector of the ground truth.</param>
    /// <param name="y">The y of the ground truth.</param>
    /// <param name="yPredicDistrib">The predictive distributions.</param>
    /// <param name="yPredicLabel">The predicted labels.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteReport(
        StreamWriter writer,
        ClassifierEvaluator<IList<Vector>, int, IList<string>, string> evaluator,
        Vector[] x,
        IList<string> y,
        IEnumerable<IDictionary<string, double>> yPredicDistrib,
        IEnumerable<string> yPredicLabel)
    {
        // Compute confusion matrix
        var confusionMatrix = evaluator.ConfusionMatrix(x, y, yPredicLabel);

        // Compute mean negative log probability
        double meanNegativeLogProbability =
            evaluator.Evaluate(x, y, yPredicDistrib, Metrics.NegativeLogProbability) / yPredicDistrib.Count();

        // Compute M-measure (averaged pairwise AUC)
        IDictionary<string, IDictionary<string, double>> aucMatrix;
        double auc = evaluator.AreaUnderRocCurve(x, y, yPredicDistrib, out aucMatrix);

        // Compute per-label AUC as well as micro- and macro-averaged AUC
        double microAuc;
        double macroAuc;
        int macroAucClassLabelCount;
        var labelAuc = this.ComputeLabelAuc(
            confusionMatrix,
            evaluator,
            x,
            y,
            yPredicDistrib,
            out microAuc,
            out macroAuc,
            out macroAucClassLabelCount);

        // Instance-averaged performance
        this.WriteInstanceAveragedPerformance(writer, confusionMatrix, meanNegativeLogProbability, microAuc);

        // Class-averaged performance
        this.WriteClassAveragedPerformance(writer, confusionMatrix, auc, macroAuc, macroAucClassLabelCount);

        // Performance on individual classes
        this.WriteIndividualClassPerformance(writer, confusionMatrix, labelAuc);

        // Confusion matrix
        this.WriteConfusionMatrix(writer, confusionMatrix);

        // Pairwise AUC
        this.WriteAucMatrix(writer, aucMatrix);
    }

    /// <summary>
    /// Computes all per-label AUCs as well as the micro- and macro-averaged AUCs.
    /// </summary>
    /// <param name="confusionMatrix">The confusion matrix.</param>
    /// <param name="evaluator">The classifier evaluator.</param>
    /// <param name="x">The x vector of the ground truth.</param>
    /// <param name="y">The y of the ground truth.</param>
    /// <param name="yPredicDistrib">The predictive distributions.</param>
    /// <param name="microAuc">The micro-averaged area under the receiver operating characteristic curve.</param>
    /// <param name="macroAuc">The macro-averaged area under the receiver operating characteristic curve.</param>
    /// <param name="macroAucClassLabelCount">The number of class labels for which the AUC if defined.</param>
    /// <returns>The area under the receiver operating characteristic curve for each class label.</returns>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private IDictionary<string, double> ComputeLabelAuc(
        ConfusionMatrix<string> confusionMatrix,
        ClassifierEvaluator<IList<Vector>, int, IList<string>, string> evaluator,
        Vector[] x,
        IList<string> y,
        IEnumerable<IDictionary<string, double>> yPredicDistrib,
        out double microAuc,
        out double macroAuc,
        out int macroAucClassLabelCount)
    {
        int instanceCount = yPredicDistrib.Count();
        var classLabels = confusionMatrix.ClassLabelSet.Elements.ToArray();
        int classLabelCount = classLabels.Length;
        var labelAuc = new Dictionary<string, double>();

        // Compute per-label AUC
        macroAucClassLabelCount = classLabelCount;
        foreach (var classLabel in classLabels)
        {
            // One versus rest
            double auc;
            try
            {
                auc = evaluator.AreaUnderRocCurve(classLabel, x, y, yPredicDistrib);
            }
            catch (ArgumentException)
            {
                auc = double.NaN;
                macroAucClassLabelCount--;
            }

            labelAuc.Add(classLabel, auc);
        }

        // Compute micro- and macro-averaged AUC
        microAuc = 0;
        macroAuc = 0;
        foreach (var label in classLabels)
        {
            if (double.IsNaN(labelAuc[label]))
            {
                continue;
            }

            microAuc += confusionMatrix.TrueLabelCount(label) * labelAuc[label] / instanceCount;
            macroAuc += labelAuc[label] / macroAucClassLabelCount;
        }

        return labelAuc;
    }

    /// <summary>
    /// Writes the header of the evaluation report to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="yGroundTruthFileName">The name of the file containing the ground truth.</param>
    /// <param name="predictionsFileName">The name of the file containing the predictions.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteReportHeader(
        StreamWriter writer, string yGroundTruthFileName, string predictionsFileName)
    {
        writer.WriteLine();
        writer.WriteLine(" Classifier evaluation report ");
        writer.WriteLine("******************************");
        writer.WriteLine();
        writer.WriteLine("           Date:      {0}", DateTime.Now);
        writer.WriteLine("   Ground truth:      {0}", yGroundTruthFileName);
        writer.WriteLine("    Predictions:      {0}", predictionsFileName);
    }

    /// <summary>
    /// Writes instance-averaged performance results to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="confusionMatrix">The confusion matrix.</param>
    /// <param name="negativeLogProbability">The negative log-probability.</param>
    /// <param name="microAuc">The micro-averaged AUC.</param>
    private void WriteInstanceAveragedPerformance(
        StreamWriter writer,
        ConfusionMatrix<string> confusionMatrix,
        double negativeLogProbability,
        double microAuc)
    {
        long instanceCount = 0;
        long correctInstanceCount = 0;
        foreach (var classLabelIndex in confusionMatrix.ClassLabelSet.Indexes)
        {
            string classLabel = confusionMatrix.ClassLabelSet.GetElementByIndex(classLabelIndex);
            instanceCount += confusionMatrix.TrueLabelCount(classLabel);
            correctInstanceCount += confusionMatrix[classLabel, classLabel];
        }

        writer.WriteLine();
        writer.WriteLine(" Instance-averaged performance (micro-averages)");
        writer.WriteLine("================================================");
        writer.WriteLine();
        writer.WriteLine("                Precision = {0,10:0.0000}", confusionMatrix.MicroPrecision);
        writer.WriteLine("                   Recall = {0,10:0.0000}", confusionMatrix.MicroRecall);
        writer.WriteLine("                       F1 = {0,10:0.0000}", confusionMatrix.MicroF1);
        writer.WriteLine();
        writer.WriteLine("                 #Correct = {0,10}", correctInstanceCount);
        writer.WriteLine("                   #Total = {0,10}", instanceCount);
        writer.WriteLine("                 Accuracy = {0,10:0.0000}", confusionMatrix.MicroAccuracy);
        writer.WriteLine("                    Error = {0,10:0.0000}", 1 - confusionMatrix.MicroAccuracy);

        writer.WriteLine();
        writer.WriteLine("                      AUC = {0,10:0.0000}", microAuc);

        writer.WriteLine();
        writer.WriteLine("                 Log-loss = {0,10:0.0000}", negativeLogProbability);
    }

    /// <summary>
    /// Writes class-averaged performance results to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="confusionMatrix">The confusion matrix.</param>
    /// <param name="auc">The AUC.</param>
    /// <param name="macroAuc">The macro-averaged AUC.</param>
    /// <param name="macroAucClassLabelCount">The number of distinct class labels used to compute macro-averaged AUC.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteClassAveragedPerformance(
        StreamWriter writer,
        ConfusionMatrix<string> confusionMatrix,
        double auc,
        double macroAuc,
        int macroAucClassLabelCount)
    {
        int classLabelCount = confusionMatrix.ClassLabelSet.Count;

        writer.WriteLine();
        writer.WriteLine(" Class-averaged performance (macro-averages)");
        writer.WriteLine("=============================================");
        writer.WriteLine();
        if (confusionMatrix.MacroPrecisionClassLabelCount < classLabelCount)
        {
            writer.WriteLine(
                "                Precision = {0,10:0.0000}     {1,10}",
                confusionMatrix.MacroPrecision,
                "[only " + confusionMatrix.MacroPrecisionClassLabelCount + "/" + classLabelCount + " classes defined]");
        }
        else
        {
            writer.WriteLine("                Precision = {0,10:0.0000}", confusionMatrix.MacroPrecision);
        }

        if (confusionMatrix.MacroRecallClassLabelCount < classLabelCount)
        {
            writer.WriteLine(
                "                   Recall = {0,10:0.0000}     {1,10}",
                confusionMatrix.MacroRecall,
                "[only " + confusionMatrix.MacroRecallClassLabelCount + "/" + classLabelCount + " classes defined]");
        }
        else
        {
            writer.WriteLine("                   Recall = {0,10:0.0000}", confusionMatrix.MacroRecall);
        }

        if (confusionMatrix.MacroF1ClassLabelCount < classLabelCount)
        {
            writer.WriteLine(
                "                       F1 = {0,10:0.0000}     {1,10}",
                confusionMatrix.MacroF1,
                "[only " + confusionMatrix.MacroF1ClassLabelCount + "/" + classLabelCount + " classes defined]");
        }
        else
        {
            writer.WriteLine("                       F1 = {0,10:0.0000}", confusionMatrix.MacroF1);
        }

        writer.WriteLine();
        if (confusionMatrix.MacroF1ClassLabelCount < classLabelCount)
        {
            writer.WriteLine(
                "                 Accuracy = {0,10:0.0000}     {1,10}",
                confusionMatrix.MacroAccuracy,
                "[only " + confusionMatrix.MacroAccuracyClassLabelCount + "/" + classLabelCount + " classes defined]");
            writer.WriteLine(
                "                    Error = {0,10:0.0000}     {1,10}",
                1 - confusionMatrix.MacroAccuracy,
                "[only " + confusionMatrix.MacroAccuracyClassLabelCount + "/" + classLabelCount + " classes defined]");
        }
        else
        {
            writer.WriteLine("                 Accuracy = {0,10:0.0000}", confusionMatrix.MacroAccuracy);
            writer.WriteLine("                    Error = {0,10:0.0000}", 1 - confusionMatrix.MacroAccuracy);
        }

        writer.WriteLine();
        if (macroAucClassLabelCount < classLabelCount)
        {
            writer.WriteLine(
                "                      AUC = {0,10:0.0000}     {1,10}",
                macroAuc,
                "[only " + macroAucClassLabelCount + "/" + classLabelCount + " classes defined]");
        }
        else
        {
            writer.WriteLine("                      AUC = {0,10:0.0000}", macroAuc);
        }

        writer.WriteLine();
        writer.WriteLine("         M (pairwise AUC) = {0,10:0.0000}", auc);
    }

    /// <summary>
    /// Writes performance results for individual classes to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="confusionMatrix">The confusion matrix.</param>
    /// <param name="auc">The per-class AUC.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteIndividualClassPerformance(
        StreamWriter writer,
        ConfusionMatrix<string> confusionMatrix,
        IDictionary<string, double> auc)
    {
        writer.WriteLine();
        writer.WriteLine(" Performance on individual classes");
        writer.WriteLine("===================================");
        writer.WriteLine();
        writer.WriteLine(
            " {0,5} {1,15} {2,10} {3,11} {4,9} {5,10} {6,10} {7,10} {8,10}",
            "Index",
            "Label",
            "#Truth",
            "#Predicted",
            "#Correct",
            "Precision",
            "Recall",
            "F1",
            "AUC");

        writer.WriteLine("----------------------------------------------------------------------------------------------------");

        foreach (var classLabelIndex in confusionMatrix.ClassLabelSet.Indexes)
        {
            string classLabel = confusionMatrix.ClassLabelSet.GetElementByIndex(classLabelIndex);

            writer.WriteLine(
                " {0,5} {1,15} {2,10} {3,11} {4,9} {5,10:0.0000} {6,10:0.0000} {7,10:0.0000} {8,10:0.0000}",
                classLabelIndex + 1,
                classLabel,
                confusionMatrix.TrueLabelCount(classLabel),
                confusionMatrix.PredictedLabelCount(classLabel),
                confusionMatrix[classLabel, classLabel],
                confusionMatrix.Precision(classLabel),
                confusionMatrix.Recall(classLabel),
                confusionMatrix.F1(classLabel),
                auc[classLabel]);
        }
    }

    /// <summary>
    /// Writes the confusion matrix to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="confusionMatrix">The confusion matrix.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteConfusionMatrix(
        StreamWriter writer, ConfusionMatrix<string> confusionMatrix)
    {
        writer.WriteLine();
        writer.WriteLine(" Confusion matrix");
        writer.WriteLine("==================");
        writer.WriteLine();
        writer.WriteLine(confusionMatrix);
    }

    /// <summary>
    /// Writes the matrix of pairwise AUC metrics to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="aucMatrix">The matrix containing the pairwise AUC metrics.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteAucMatrix(
        StreamWriter writer, IDictionary<string, IDictionary<string, double>> aucMatrix)
    {
        writer.WriteLine();
        writer.WriteLine(" Pairwise AUC matrix");
        writer.WriteLine("=====================");
        writer.WriteLine();

        const int MaxLabelWidth = 20;
        const int MaxValueWidth = 6;

        // Widths of the columns
        string[] labels = aucMatrix.Keys.ToArray();
        int classLabelCount = aucMatrix.Count;
        var columnWidths = new int[classLabelCount + 1];

        // For each column of the confusion matrix...
        for (int c = 0; c < classLabelCount; c++)
        {
            // ...find the longest string among counts and label
            int labelWidth = labels[c].Length;

            columnWidths[c + 1] = labelWidth > MaxLabelWidth ? MaxLabelWidth : labelWidth;
            for (int r = 0; r < classLabelCount; r++)
            {
                int countWidth = MaxValueWidth;
                if (countWidth > columnWidths[c + 1])
                {
                    columnWidths[c + 1] = countWidth;
                }
            }

            if (labelWidth > columnWidths[0])
            {
                columnWidths[0] = labelWidth > MaxLabelWidth ? MaxLabelWidth : labelWidth;
            }
        }

        // Print title row
        string format = string.Format("{{0,{0}}} \\ Prediction ->", columnWidths[0]);
        writer.WriteLine(format, "Truth");

        // Print column labels
        this.WriteLabel(writer, string.Empty, columnWidths[0]);
        for (int c = 0; c < classLabelCount; c++)
        {
            this.WriteLabel(writer, labels[c], columnWidths[c + 1]);
        }

        writer.WriteLine();

        // For each row (true labels) in confusion matrix...
        for (int r = 0; r < classLabelCount; r++)
        {
            // Print row label
            this.WriteLabel(writer, labels[r], columnWidths[0]);

            // For each column (predicted labels) in the confusion matrix...
            for (int c = 0; c < classLabelCount; c++)
            {
                // Print count
                this.WriteAucValue(writer, labels[r].Equals(labels[c]) ? -1 : aucMatrix[labels[r]][labels[c]], columnWidths[c + 1]);
            }

            writer.WriteLine();
        }
    }

    /// <summary>
    /// Writes a label to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="label">The label.</param>
    /// <param name="width">The width in characters used to print the label.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteLabel(StreamWriter writer, string label, int width)
    {
        string paddedLabel = label.Length > width ? label.Substring(0, width) : label;
        paddedLabel = paddedLabel.PadLeft(width + 2);
        writer.Write(paddedLabel);
    }


    /// <summary>
    /// Writes a count to a specified stream writer.
    /// </summary>
    /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
    /// <param name="auc">The count.</param>
    /// <param name="width">The width in characters used to print the count.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteAucValue(StreamWriter writer, double auc, int width)
    {
        string paddedCount;
        if (auc > 0)
        {
            paddedCount = auc.ToString(CultureInfo.InvariantCulture);
        }
        else
        {
            if (auc < 0)
            {
                paddedCount = ".";
            }
            else
            {
                paddedCount = double.IsNaN(auc) ? "NaN" : "0";
            }
        }

        paddedCount = paddedCount.Length > width ? paddedCount.Substring(0, width) : paddedCount;
        paddedCount = paddedCount.PadLeft(width + 2);
        writer.Write(paddedCount);
    }



    /// <summary>
    /// Writes the probability calibration plot to the file with the specified name.
    /// </summary>
    /// <param name="fileName">The name of the file to write the calibration plot to.</param>
    /// <param name="evaluator">The classifier evaluator.</param>
    /// <param name="x">The x vector of the ground truth.</param>
    /// <param name="y">The y of the ground truth.</param>
    /// <param name="yPredicDistrib">The predictive distributions.</param>
    /// <param name="positiveClassLabel">The label of the positive class.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteCalibrationCurve(
        string fileName,
         ClassifierEvaluator<IList<Vector>, int, IList<string>, string> evaluator,
        Vector[] x,
        IList<string> y,
        IEnumerable<IDictionary<string, double>> yPredicDistrib,
        string positiveClassLabel)
    {
        Debug.Assert(yPredicDistrib != null, "The predictive distributions must not be null.");
        Debug.Assert(yPredicDistrib.Count() > 0, "The predictive distributions must not be empty.");
        Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");

        var calibrationCurve = evaluator.CalibrationCurve(positiveClassLabel, x, y, yPredicDistrib);
        double calibrationError = calibrationCurve.Select(v => Metrics.AbsoluteError(v.First, v.Second)).Average();

        using (var writer = new StreamWriter(fileName))
        {
            writer.WriteLine("# Empirical probability calibration plot");
            writer.WriteLine("#");
            writer.WriteLine("# Class '" + positiveClassLabel + "'     (versus the rest)");
            writer.WriteLine("# Calibration error = {0}     (mean absolute error)", calibrationError);
            writer.WriteLine("#");
            writer.WriteLine("# Predicted probability, empirical probability");
            foreach (var point in calibrationCurve)
            {
                writer.WriteLine("{0}, {1}", point.First, point.Second);
            }
        }
    }

    /// <summary>
    /// Writes the precision-recall curve to the file with the specified name.
    /// </summary>
    /// <param name="fileName">The name of the file to write the precision-recall curve to.</param>
    /// <param name="evaluator">The classifier evaluator.</param>
    /// <param name="x">The x vector of the ground truth.</param>
    /// <param name="y">The y of the ground truth.</param>
    /// <param name="yPredicDistrib">The predictive distributions.</param>
    /// <param name="positiveClassLabel">The label of the positive class.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WritePrecisionRecallCurve(
        string fileName,
         ClassifierEvaluator<IList<Vector>, int, IList<string>, string> evaluator,
        Vector[] x,
        IList<string> y,
        IEnumerable<IDictionary<string, double>> yPredicDistrib,
        string positiveClassLabel)
    {
        Debug.Assert(yPredicDistrib != null, "The predictive distributions must not be null.");
        Debug.Assert(yPredicDistrib.Count() > 0, "The predictive distributions must not be empty.");
        Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");

        var precisionRecallCurve = evaluator.PrecisionRecallCurve(positiveClassLabel, x, y, yPredicDistrib);
        using (var writer = new StreamWriter(fileName))
        {
            writer.WriteLine("# Precision-recall curve");
            writer.WriteLine("#");
            writer.WriteLine("# Class '" + positiveClassLabel + "'     (versus the rest)");
            writer.WriteLine("#");
            writer.WriteLine("# Recall (R), precision (P)");
            foreach (var point in precisionRecallCurve)
            {
                writer.WriteLine("{0}, {1}", point.First, point.Second);
            }
        }
    }

    /// <summary>
    /// Writes the receiver operating characteristic curve to the file with the specified name.
    /// </summary>
    /// <param name="fileName">The name of the file to write the receiver operating characteristic curve to.</param>
    /// <param name="evaluator">The classifier evaluator.</param>
    /// <param name="x">The x vector of the ground truth.</param>
    /// <param name="y">The y of the ground truth.</param>
    /// <param name="yPredicDistrib">The predictive distributions.</param>
    /// <param name="positiveClassLabel">The label of the positive class.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    private void WriteRocCurve(
        string fileName,
         ClassifierEvaluator<IList<Vector>, int, IList<string>, string> evaluator,
        Vector[] x,
        IList<string> y,
        IEnumerable<IDictionary<string, double>> yPredicDistrib,
        string positiveClassLabel)
    {
        Debug.Assert(yPredicDistrib != null, "The predictive distributions must not be null.");
        Debug.Assert(yPredicDistrib.Count() > 0, "The predictive distributions must not be empty.");
        Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");

        var rocCurve = evaluator.ReceiverOperatingCharacteristicCurve(positiveClassLabel, x, y, yPredicDistrib);

        using (var writer = new StreamWriter(fileName))
        {
            writer.WriteLine("# Receiver operating characteristic (ROC) curve");
            writer.WriteLine("#");
            writer.WriteLine("# Class '" + positiveClassLabel + "'     (versus the rest)");
            writer.WriteLine("#");
            writer.WriteLine("# False positive rate (FPR), true positive rate (TPR)");
            foreach (var point in rocCurve)
            {
                writer.WriteLine("{0}, {1}", point.First, point.Second);
            }
        }
    }

    /*
        /// <summary>
        /// Samples weights from the learned weight distribution of the Bayes point machine classifier.
        /// </summary>
        /// <param name="fileName">The name of the file to write the weights to.</param>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine used to sample weights from.</param>
        /// <param name="samplesFile">The name of the file to which the weights will be written.</param>
        /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
        public void SampleWeights(
            string fileName,
            BPM classifier)
        {
            // Sample weights
            IEnumerable<Vector> samples = SampleWeights(classifier);

            // Write samples to file
            using (var writer = new StreamWriter(fileName))
            {
                writer.WriteLine(string.Join(",", samples));
            }
        }


        /// <summary>
        /// Samples weights from the learned weight distribution of the Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine used to sample weights from.</param>
        /// <returns>The samples from the weight distribution of the Bayes point machine classifier.</returns>
        /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
        private IEnumerable<Vector> SampleWeights(BPM classifier)
        {
            Debug.Assert(classifier != null, "The classifier must not be null.");

            IReadOnlyList<IReadOnlyList<Gaussian>> weightPosteriorDistributions = classifier.WeightPosteriorDistributions;
            int classCount = weightPosteriorDistributions.Count < 2 ? 2 : weightPosteriorDistributions.Count;
            int featureCount = weightPosteriorDistributions[0].Count;

            var samples = new Vector[classCount - 1];
            for (int c = 0; c < classCount - 1; c++)
            {
                var sample = Vector.Zero(featureCount);
                for (int f = 0; f < featureCount; f++)
                {
                    sample[f] = weightPosteriorDistributions[c][f].Sample();
                }

                samples[c] = sample;
            }

            return samples;
        }*/



    /// <summary>
    /// Writes a collection of label distributions to the file with the specified name.
    /// </summary>
    /// <param name="fileName">The file name.</param>
    /// <param name="labelDistributions">A collection of label distributions.</param>
    /// <remarks>Adapted from MicrosoftResearch.Infer.Learners</remarks>
    public static void SaveLabelDistributions(
        string fileName, IEnumerable<IDictionary<string, double>> labelDistributions)
    {
        if (fileName == null)
        {
            throw new ArgumentNullException("fileName");
        }

        if (labelDistributions == null)
        {
            throw new ArgumentNullException("labelDistributions");
        }

        using (var writer = new StreamWriter(fileName))
        {
            foreach (var labelDistribution in labelDistributions)
            {
                foreach (var uncertainLabel in labelDistribution)
                {
                    writer.Write(
                        "{0}{1}{2}{3}",
                        uncertainLabel.Equals(labelDistribution.First()) ? string.Empty : string.Empty + LabelDistribution.PointSeparator,
                        uncertainLabel.Key,
                        LabelDistribution.LabelProbabilitySeparator,
                        uncertainLabel.Value);
                }

                writer.WriteLine();
            }
        }
    }
}

