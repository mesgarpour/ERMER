﻿/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    /// <summary>
    /// Interface to a predictor which can be trained incrementally.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    public interface IPredictorIncrementalTraining<in TInstanceSource, in TLabelSource> 
    {
        /// <summary>
        /// Incrementally trains the predictor on the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="labelSource">An optional source of labels.</param>
        void TrainIncremental(TInstanceSource instanceSource, TLabelSource labelSource = default(TLabelSource));
    }
}