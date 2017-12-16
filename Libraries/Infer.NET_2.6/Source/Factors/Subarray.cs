// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.Subarray{T}(IList{T}, IList{int})" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of an array item.</typeparam>
    [FactorMethod(typeof(Factor), "Subarray<>")]
    [Quality(QualityBand.Mature)]
    public static class SubarrayOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(IList<T> items, IList<T> array, IList<int> indices)
        {
            return GetItemsOp<T>.LogAverageFactor(items, array, indices);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(IList<T> items, IList<T> array, IList<int> indices)
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<T> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                z += array[indices[i]].GetLogProb(items[i]);
            }
            return z;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<DistributionType> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : CanGetLogAverageOf<DistributionType>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                z += array[indices[i]].GetLogAverageOf(items[i]);
            }
            return z;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices))</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogAverageFactor<DistributionType>(IList<DistributionType> items, IList<T> array, IList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            double z = 0.0;
            for (int i = 0; i < indices.Count; i++)
            {
                z += items[i].GetLogProb(array[indices[i]]);
            }
            return z;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items) p(items) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<DistributionType>(IList<DistributionType> items) where DistributionType : CanGetLogAverageOf<DistributionType>
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(items,array) p(items,array) factor(items,array,indices) / sum_items p(items) messageTo(items))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        public static double LogEvidenceRatio<DistributionType>(IList<T> items, IList<DistributionType> array, IList<int> indices)
            where DistributionType : CanGetLogProb<T>
        {
            return LogAverageFactor(items, array, indices);
        }

        /// <summary>EP message to <c>items</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>items</c> as the random arguments are varied. The formula is <c>proj[p(items) sum_(array) p(array) factor(items,array,indices)]/p(items)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ResultType">The type of the outgoing message.</typeparam>
        public static ResultType ItemsAverageConditional<DistributionType, ResultType>([SkipIfAllUniform] IList<DistributionType> array, IList<int> indices, ResultType result)
            where ResultType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            Assert.IsTrue(result.Count == indices.Count, "result.Count != indices.Count");
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[i];
                value.SetTo(array[indices[i]]);
                result[i] = value;
            }
            return result;
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageConditionalInit<TDist>(
            [IgnoreDependency] DistributionStructArray<TDist, T> array, IList<int> indices)
            where TDist : struct,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageConditionalInit<TDist>([IgnoreDependency] DistributionRefArray<TDist, T> array, IList<int> indices)
            where TDist : class,
                SettableTo<TDist>,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(items) p(items) factor(items,array,indices)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="items" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>([SkipIfAllUniform] IList<DistributionType> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            Assert.IsTrue(items.Count == indices.Count, "items.Count != indices.Count");
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.SetTo(items[i]);
                result[indices[i]] = value;
            }
            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(items) p(items) factor(items,array,indices)]/p(array)</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageConditional<DistributionType, ArrayType>(IList<T> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            if (items.Count != indices.Count)
                throw new ArgumentException(indices.Count + " indices were given to Subarray but the output array has length " + items.Count);
            result.SetToUniform();
            for (int i = 0; i < indices.Count; i++)
            {
                DistributionType value = result[indices[i]];
                value.Point = items[i];
                result[indices[i]] = value;
            }
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(items,array,indices))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionStructArray<TDist, T> ItemsAverageLogarithmInit<TDist>([IgnoreDependency] DistributionStructArray<TDist, T> array, IList<int> indices)
            where TDist : struct,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionStructArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <summary />
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of a distribution over an array item.</typeparam>
        [Skip]
        public static DistributionRefArray<TDist, T> ItemsAverageLogarithmInit<TDist>([IgnoreDependency] DistributionRefArray<TDist, T> array, IList<int> indices)
            where TDist : class,
                SettableTo<TDist>,
                SettableToProduct<TDist>,
                SettableToRatio<TDist>,
                SettableToPower<TDist>,
                SettableToWeightedSum<TDist>,
                CanGetLogAverageOf<TDist>,
                CanGetLogAverageOfPower<TDist>,
                CanGetAverageLog<TDist>,
                IDistribution<T>,
                Sampleable<T>
        {
            return new DistributionRefArray<TDist, T>(indices.Count, i => (TDist)array[indices[i]].Clone());
        }

        /// <summary>VMP message to <c>items</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>items</c> as the random arguments are varied. The formula is <c>proj[sum_(array) p(array) factor(items,array,indices)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ResultType">The type of the outgoing message.</typeparam>
        public static ResultType ItemsAverageLogarithm<DistributionType, ResultType>([SkipIfAllUniform] IList<DistributionType> array, IList<int> indices, ResultType result)
            where ResultType : IList<DistributionType>
            where DistributionType : SettableTo<DistributionType>
        {
            return ItemsAverageConditional<DistributionType, ResultType>(array, indices, result);
        }

        [Skip]
        public static ResultType ItemsDeriv<ResultType>(ResultType result)
            where ResultType : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>items</c> integrated out. The formula is <c>sum_items p(items) factor(items,array,indices)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="items" /> is not a proper distribution.</exception>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>([SkipIfAllUniform] IList<DistributionType> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : SettableTo<DistributionType>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="items">Incoming message from <c>items</c>.</param>
        /// <param name="indices">Constant value for <c>indices</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>items</c> integrated out. The formula is <c>sum_items p(items) factor(items,array,indices)</c>.</para>
        /// </remarks>
        /// <typeparam name="DistributionType">The type of a distribution over an array item.</typeparam>
        /// <typeparam name="ArrayType">The type of the outgoing message.</typeparam>
        public static ArrayType ArrayAverageLogarithm<DistributionType, ArrayType>(IList<T> items, IList<int> indices, ArrayType result)
            where ArrayType : IList<DistributionType>, SettableToUniform
            where DistributionType : HasPoint<T>
        {
            return ArrayAverageConditional<DistributionType, ArrayType>(items, indices, result);
        }
    }
}
