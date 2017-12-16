﻿/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Runtime.Serialization;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// An abstract Bayes point machine classifier which wraps generated inference algorithms for training and prediction.
    /// </summary>
    /// <typeparam name="TWeightDistributions">The type of the distributions over weights.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TLabelDistribution">The type of a distribution over a label.</typeparam>
    [Serializable]
    internal abstract class InferenceAlgorithms<TWeightDistributions, TLabel, TLabelDistribution>
        : IInferenceAlgorithms<TLabel, TLabelDistribution> 
        where TWeightDistributions : 
            CanGetLogAverageOf<TWeightDistributions>,
            SettableTo<TWeightDistributions>, 
            SettableToProduct<TWeightDistributions>,
            SettableToRatio<TWeightDistributions>,
            SettableToUniform,
            ICloneable
    {
        #region Fields and constructor

        /// <summary>
        /// The number of iterations to run the training algorithm on an empty batch of data.
        /// </summary>
        private const int EmptyBatchIterationCount = 20;

        /// <summary>
        /// If true, the training algorithm computes evidence.
        /// </summary>
        private readonly bool computeModelEvidence;

        /// <summary>
        /// The inference algorithm generated by the Infer.NET compiler for training.
        /// </summary>
        [NonSerialized]
        private IGeneratedAlgorithm trainingAlgorithm;

        /// <summary>
        /// The inference algorithm generated by the Infer.NET compiler for prediction.
        /// </summary>
        [NonSerialized]
        private IGeneratedAlgorithm predictionAlgorithm;

        /// <summary>
        /// Initializes a new instance of the <see cref="InferenceAlgorithms{TWeightDistributions,TLabel,TLabelDistribution}"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        protected InferenceAlgorithms(bool computeModelEvidence, bool useSparseFeatures, int featureCount)
        {
            InferenceAlgorithmUtilities.CheckFeatureCount(featureCount);
            
            this.computeModelEvidence = computeModelEvidence;
            this.UseSparseFeatures = useSparseFeatures;
            this.FeatureCount = featureCount;
            this.BatchCount = 1;

            // Create the inference algorithms for training and prediction
            this.CreateInferenceAlgorithms();

            // Initialize the inference algorithms
            this.InitializeInferenceAlgorithms();
        }

        #endregion

        #region Events

        /// <summary>
        /// The event that is fired at the end of each iteration of the training algorithm.
        /// </summary>
        /// <remarks>
        /// Subscribing a handler to this event may have negative effects on the performance 
        /// of the training algorithm in terms of both memory consumption and execution speed.
        /// </remarks>
        public event EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> IterationChanged;

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets a value indicating whether the progress of the training algorithm is printed to the console.
        /// </summary>
        public bool ShowProgress { get; set; }

        /// <summary>
        /// Gets a value indicating whether the inference algorithms expect features in a sparse representation.
        /// </summary>
        public bool UseSparseFeatures { get; private set; }

        /// <summary>
        /// Gets the number of features that the inference algorithms use.
        /// </summary>
        public int FeatureCount { get; private set; }

        /// <summary>
        /// Gets the distributions over weights as factorized <see cref="Gaussian"/> distributions.
        /// </summary>
        public abstract IReadOnlyList<IReadOnlyList<Gaussian>> WeightDistributions { get; }

        /// <summary>
        /// Gets or sets the marginal distributions over weights.
        /// </summary>
        protected TWeightDistributions WeightMarginals { get; set; }

        /// <summary>
        /// Gets or sets the read-only marginal distributions over weights.
        /// </summary>
        protected IReadOnlyList<IReadOnlyList<Gaussian>> ReadOnlyWeightMarginals { get; set; }

        /// <summary>
        /// Gets or sets the marginal distributions over weights divided by their prior distributions.
        /// </summary>
        protected TWeightDistributions WeightMarginalsDividedByPriors { get; set; }

        /// <summary>
        /// Gets or sets the constraint distributions over weights.
        /// </summary>
        protected TWeightDistributions WeightConstraints { get; set; }

        /// <summary>
        /// Gets or sets the output messages for weights per training data batch.
        /// </summary>
        protected TWeightDistributions[] BatchWeightOutputMessages { get; set; }

        /// <summary>
        /// Gets the inference algorithm generated by the Infer.NET compiler for training.
        /// </summary>
        protected IGeneratedAlgorithm TrainingAlgorithm 
        {
            get
            {
                return this.trainingAlgorithm;
            }

            private set
            {
                Debug.Assert(value != null, "The generated training algorithm must not be null.");
                this.trainingAlgorithm = value;
            }
        }

        /// <summary>
        /// Gets the inference algorithm generated by the Infer.NET compiler for prediction.
        /// </summary>
        protected IGeneratedAlgorithm PredictionAlgorithm
        {
            get
            {
                return this.predictionAlgorithm;
            }

            private set
            {
                Debug.Assert(value != null, "The generated prediction algorithm must not be null.");
                this.predictionAlgorithm = value;
            }
        }

        /// <summary>
        /// Sets the labels of the training algorithm to the specified labels.
        /// </summary>
        protected abstract TLabel[] Labels { set; }

        /// <summary>
        /// Gets the number of batches the training data is split into.
        /// </summary>
        /// <returns>The number of batches used.</returns>
        protected int BatchCount { get; private set; }

        #endregion

        #region IInferenceAlgorithms methods

        /// <summary>
        /// Sets the number of batches the training data is split into and resets the per-batch output messages.
        /// </summary>
        /// <param name="value">The number of batches to use.</param>
        public void SetBatchCount(int value)
        {
            InferenceAlgorithmUtilities.CheckBatchCount(value);
            this.BatchCount = value;
            this.BatchWeightOutputMessages = this.CreateUniformBatchOutputMessages(this.BatchCount);
        }

        /// <summary>
        /// Runs the generated training algorithm for the specified features and labels.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <param name="labels">The labels.</param>
        /// <param name="iterationCount">The number of iterations to run the training algorithm for.</param>
        /// <param name="batchNumber">
        /// An optional batch number. Defaults to 0 and is used only if the training data is divided into batches.
        /// </param>
        /// <returns>
        /// The natural logarithm of the evidence for the Bayes point machine classifier model 
        /// if the training algorithm computes evidence and 0 otherwise.
        /// </returns>
        public virtual double Train(double[][] featureValues, int[][] featureIndexes, TLabel[] labels, int iterationCount, int batchNumber = 0)
        {
            // Run the training algorithm
            this.TrainInternal(featureValues, featureIndexes, labels, iterationCount, batchNumber);

            // Return model evidence, if supported
            return this.computeModelEvidence ? this.ComputeLogModelEvidence(batchNumber) : 0.0;
        }

        /// <summary>
        /// Runs the generated prediction algorithm for the specified features.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <param name="iterationCount">The number of iterations to run the prediction algorithm for.</param>
        /// <returns>The predictive distributions over labels.</returns>
        public IEnumerable<TLabelDistribution> PredictDistribution(double[][] featureValues, int[][] featureIndexes, int iterationCount)
        {
            InferenceAlgorithmUtilities.CheckIterationCount(iterationCount);
            InferenceAlgorithmUtilities.CheckFeatures(this.UseSparseFeatures, this.FeatureCount, featureValues, featureIndexes);

            // Update prior weight distributions of the prediction algorithm to the posterior weight distributions of the training algorithm
            this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightPriors, this.WeightMarginals);

            // Infer posterior distribution over labels, one instance after the other
            for (int i = 0; i < featureValues.Length; i++)
            {
                // Observe a single feature vector
                this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.InstanceCount, 1);

                if (this.UseSparseFeatures)
                {
                    this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.InstanceFeatureCounts, new[] { featureValues[i].Length });
                    this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureIndexes, new[] { featureIndexes[i] });
                }

                this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureValues, new[] { featureValues[i] });

                // Infer the posterior distribution over the label
                yield return this.CopyLabelDistribution(this.RunPredictionAlgorithm(iterationCount));
            }
        }

        #endregion

        #region Template methods

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains the Bayes point machine classifier.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the generated training algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        protected abstract IGeneratedAlgorithm CreateTrainingAlgorithm(bool computeModelEvidence, bool useSparseFeatures);

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from the Bayes point machine classifier.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the generated prediction algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        protected abstract IGeneratedAlgorithm CreatePredictionAlgorithm(bool useSparseFeatures);

        /// <summary>
        /// Creates uniform output messages for all training data batches.
        /// </summary>
        /// <param name="batchCount">The number of batches.</param>
        /// <returns>
        /// An array of uniform output messages, one per training data batch, and null if there is only one single batch.
        /// </returns>
        protected abstract TWeightDistributions[] CreateUniformBatchOutputMessages(int batchCount);

        /// <summary>
        /// Copies the specified distribution over labels.
        /// </summary>
        /// <param name="labelDistribution">The distribution over labels to be copied.</param>
        /// <returns>The copy of the specified label distribution.</returns>
        protected abstract TLabelDistribution CopyLabelDistribution(TLabelDistribution labelDistribution);

        #endregion

        #region Helper methods

        /// <summary>
        /// Runs the generated training algorithm for the specified features and labels.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <param name="labels">The labels.</param>
        /// <param name="iterationCount">The number of iterations to run the training algorithm for.</param>
        /// <param name="batchNumber">
        /// An optional batch number. Defaults to 0 and is used only if the training data is divided into batches.
        /// </param>
        protected virtual void TrainInternal(double[][] featureValues, int[][] featureIndexes, TLabel[] labels, int iterationCount, int batchNumber = 0)
        {
            InferenceAlgorithmUtilities.CheckIterationCount(iterationCount);
            InferenceAlgorithmUtilities.CheckBatchNumber(batchNumber, this.BatchCount);
            InferenceAlgorithmUtilities.CheckFeatures(this.UseSparseFeatures, this.FeatureCount, featureValues, featureIndexes);
            Debug.Assert(featureValues.Length == labels.Length, "There must be the same number of feature values and labels.");

            // Observe features and labels
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.InstanceCount, featureValues.Length);

            if (this.UseSparseFeatures)
            {
                this.TrainingAlgorithm.SetObservedValue(
                    InferenceQueryVariableNames.InstanceFeatureCounts,
                    Util.ArrayInit(featureValues.Length, instance => featureValues[instance].Length));
                this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureIndexes, featureIndexes);
            }

            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureValues, featureValues);
            this.Labels = labels;

            if (this.BatchCount == 1)
            {
                // Required for incremental training
                this.WeightConstraints = this.WeightMarginalsDividedByPriors;

                // Run the training algorithm
                this.RunTrainingAlgorithm(iterationCount);
            }
            else
            {
                // Print information about current batch to console
                if (this.ShowProgress)
                {
                    Console.WriteLine("Batch {0} [{1} instance{2}]", batchNumber + 1, featureValues.Length, featureValues.Length == 1 ? string.Empty : "s");
                }

                // Compute the constraint distributions for the weights for the given batch
                this.WeightConstraints.SetToRatio(this.WeightMarginalsDividedByPriors, this.BatchWeightOutputMessages[batchNumber]);

                // Run the training algorithm
                this.RunTrainingAlgorithm(iterationCount);

                // Update the output messages for the weights for the given batch
                this.BatchWeightOutputMessages[batchNumber].SetToRatio(this.WeightMarginalsDividedByPriors, this.WeightConstraints);
            }
        }

        /// <summary>
        /// Computes the logarithm of the model evidence corrections required in batched training.
        /// </summary>
        /// <returns>The logarithm of the model evidence corrections.</returns>
        /// <remarks>
        /// When the training data is split into several batches, it is necessary to eliminate evidence contributions
        /// which would otherwise be double counted. In essence, evidence corrections remove the superfluous contributions 
        /// of factors which are shared across data batches, such as priors and constraints. To compute the evidence 
        /// contributions of the factors shared across batches, one can compute the evidence on an empty batch.
        /// </remarks>
        protected virtual double ComputeLogEvidenceCorrection()
        {
            // Correct the below empty batch correction and add the missing evidence contribution 
            // for the Replicate operator on the weights for all batches
            double logModelEvidenceCorrection = InferenceAlgorithmUtilities.ComputeLogEvidenceCorrectionReplicateAllBatches(this.BatchWeightOutputMessages);

            // Compute the evidence contribution of all factors shared across batches and remove it for all but one batch.
            logModelEvidenceCorrection -= (this.BatchCount - 1) * this.ComputeLogEvidenceContributionEmptyBatch();

            return logModelEvidenceCorrection;
        }

        /// <summary>
        /// Computes the logarithm of the model evidence contribution of an empty batch.
        /// </summary>
        /// <returns>The logarithm of the model computed evidence contribution of an empty batch.</returns>
        protected virtual double ComputeLogEvidenceContributionEmptyBatch()
        {
            // Update the constraints on the distributions over weights
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightConstraints, this.WeightMarginalsDividedByPriors);

            // Compute the log evidence contribution of an empty batch
            this.ObserveEmptyTrainingData();

            this.TrainingAlgorithm.Execute(EmptyBatchIterationCount);

            return this.TrainingAlgorithm.Marginal<Bernoulli>(InferenceQueryVariableNames.ModelSelector).LogOdds;
        }

        /// <summary>
        /// Sets the training algorithm to use empty training data.
        /// </summary>
        protected virtual void ObserveEmptyTrainingData()
        {
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.InstanceCount, 0);
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureValues, new double[][] { });

            if (this.UseSparseFeatures)
            {
                this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureIndexes, new int[][] { });
            }

            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.Labels, new TLabel[] { });
        }

        /// <summary>
        /// Runs the training algorithm for the specified number of iterations.
        /// </summary>
        /// <param name="iterationCount">The number of iterations to run the training algorithm for.</param>
        private void RunTrainingAlgorithm(int iterationCount)
        {
            Debug.Assert(iterationCount >= 0, "The number of training algorithm iterations must not be negative.");

            // Update the constraints on the distributions over weights
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightConstraints, this.WeightConstraints);

            // Register ProgressChanged handler for duration of training
            this.TrainingAlgorithm.ProgressChanged += this.OnProgressChanged;

            // Reset the generated training algorithm
            this.TrainingAlgorithm.Reset();

            // Run generated training algorithm
            for (int iteration = 1; iteration <= iterationCount; iteration++)
            {
                this.TrainingAlgorithm.Update(1);
                this.OnIterationChanged(iteration);
            }

            // Unregister ProgressChanged handler and print termination status
            this.TrainingAlgorithm.ProgressChanged -= this.OnProgressChanged;
            
            this.WriteTrainingAlgorithmTerminationStatus(iterationCount);

            // Get the marginal posterior weight distributions
            this.WeightMarginals = this.TrainingAlgorithm.Marginal<TWeightDistributions>(InferenceQueryVariableNames.Weights);

            // Get the marginal posterior weight distributions divided by their prior distributions
            this.WeightMarginalsDividedByPriors = this.TrainingAlgorithm.Marginal<TWeightDistributions>(
                InferenceQueryVariableNames.Weights, QueryTypes.MarginalDividedByPrior.Name);
        }

        /// <summary>
        /// Runs the prediction algorithm and infers the posterior label distribution.
        /// </summary>
        /// <param name="iterationCount">The number of iterations to run the prediction algorithm for.</param>
        /// <returns>The posterior label distribution.</returns>
        private TLabelDistribution RunPredictionAlgorithm(int iterationCount)
        {
            Debug.Assert(iterationCount >= 0, "The number of prediction algorithm iterations must not be negative.");

            this.PredictionAlgorithm.Execute(iterationCount);
            return this.PredictionAlgorithm.Marginal<IList<TLabelDistribution>>(InferenceQueryVariableNames.Labels)[0];
        }

        /// <summary>
        /// Computes the logarithm of the model evidence for the batch with the specified number.
        /// </summary>
        /// <param name="batchNumber">The number of the batch for which model evidence is computed.</param>
        /// <returns>The logarithm of the model evidence for the specified batch.</returns>
        private double ComputeLogModelEvidence(int batchNumber)
        {
            // Compute the logarithm of the evidence of the model on the current batch
            double logModelEvidence = this.TrainingAlgorithm.Marginal<Bernoulli>(InferenceQueryVariableNames.ModelSelector).LogOdds;

            // Compute the logarithm of the evidence corrections once we have seen all batches
            if (this.BatchCount > 1 && batchNumber == this.BatchCount - 1)
            {
                logModelEvidence += this.ComputeLogEvidenceCorrection();
            }

            return logModelEvidence;
        }

        /// <summary>
        /// Creates the inference algorithms for training and prediction.
        /// </summary>
        private void CreateInferenceAlgorithms()
        {
            this.TrainingAlgorithm = this.CreateTrainingAlgorithm(this.computeModelEvidence, this.UseSparseFeatures);
            this.PredictionAlgorithm = this.CreatePredictionAlgorithm(this.UseSparseFeatures);
        }

        /// <summary>
        /// Initializes the inference algorithms for training and prediction.
        /// </summary>
        private void InitializeInferenceAlgorithms()
        {
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureCount, this.FeatureCount);
            this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.FeatureCount, this.FeatureCount);
        }

        /// <summary>
        /// Recreates the generated inference algorithms after deserialization.
        /// </summary>
        /// <param name="context">The streaming context.</param>
        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            this.CreateInferenceAlgorithms();
            this.InitializeInferenceAlgorithms();
        }

        /// <summary>
        /// Delivers the current marginal distributions over weights.
        /// </summary>
        /// <param name="completedIteration">The completed training algorithm iteration.</param>
        private void OnIterationChanged(int completedIteration)
        {
            // Raise IterationChanged event
            EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> handler = this.IterationChanged;
            if (handler != null)
            {
                this.WeightMarginals = this.TrainingAlgorithm.Marginal<TWeightDistributions>(InferenceQueryVariableNames.Weights);
                handler(this, new BayesPointMachineClassifierIterationChangedEventArgs(completedIteration, this.WeightDistributions)); // Read-only marginals
            }
        }

        /// <summary>
        /// Writes the progress of the training algorithm to console.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="progressChangedEventArgs">The information describing the training algorithm progress.</param>
        private void OnProgressChanged(object sender, ProgressChangedEventArgs progressChangedEventArgs)
        {
            if (!this.ShowProgress)
            {
                return;
            }

            int currentIteration = progressChangedEventArgs.Iteration + 1;
            if (currentIteration == 1)
            {
                Console.WriteLine("Iterating: ");
            }

            Console.Write(currentIteration % ProgressFormattingOptions.EmphasizedIterationCount == 0 ? "|" : ".");

            if (currentIteration % ProgressFormattingOptions.LineBreakIterationCount == 0)
            {
                Console.WriteLine(" " + currentIteration);
            }
        }

        /// <summary>
        /// Writes the final number of training algorithm iterations to the console.
        /// </summary>
        /// <param name="iterationCount">The number of training algorithm iterations.</param>
        private void WriteTrainingAlgorithmTerminationStatus(int iterationCount)
        {
            if (!this.ShowProgress)
            {
                return;
            }

            if (iterationCount % ProgressFormattingOptions.LineBreakIterationCount != 0)
            {
                Console.WriteLine(" " + iterationCount);
            }
        }

        /// <summary>
        /// The formatting options applied when printing the progress of an inference algorithm to console.
        /// </summary>
        private static class ProgressFormattingOptions
        {
            /// <summary>
            /// The number of iterations after which progress will be emphasized.
            /// </summary>
            public const int EmphasizedIterationCount = 10;

            /// <summary>
            /// The number of iterations after which a line break will occur.
            /// </summary>
            public const int LineBreakIterationCount = 50;
        }

        #endregion
    }
}
