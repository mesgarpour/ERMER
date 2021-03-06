﻿/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners.BayesPointMachineClassifierInternal
{
    using System;

    using MicrosoftResearch.Infer.Learners.Mappings;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// A multi-class Bayes point machine classifier with compound prior distributions over weights
    /// which operates on data in the standard format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    [SerializationVersion(5)]
    internal class CompoundMulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel> :
        MulticlassStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, BayesPointMachineClassifierTrainingSettings>
    {
        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="CompoundMulticlassStandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TLabel}"/> class.
        /// </summary>
        /// <param name="standardMapping">The mapping used for accessing data in the standard format.</param>
        public CompoundMulticlassStandardDataFormatBayesPointMachineClassifier(IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> standardMapping)
            : base(standardMapping)
        {
            this.Classifier = new CompoundMulticlassNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(this.NativeMapping);
            this.Settings = new MulticlassBayesPointMachineClassifierSettings<TLabel>(() => this.Classifier.IsTrained);
        }
    }
}