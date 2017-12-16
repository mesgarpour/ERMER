// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Diagnostics;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="EnumSupport.DiscreteEnum{TEnum}(Vector)" /></description></item><item><description><see cref="Discrete.Sample(Vector)" /></description></item><item><description><see cref="Factor.Discrete(Vector)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Discrete), "Sample", typeof(Vector))]
    [FactorMethod(new String[] { "sample", "probs" }, typeof(Factor), "Discrete", typeof(Vector))]
    [FactorMethod(typeof(EnumSupport), "DiscreteEnum<>")]
    [Quality(QualityBand.Mature)]
    public static class DiscreteFromDirichletOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample) p(Sample) factor(Sample,Probs))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Discrete sample, [Fresh] Discrete to_sample)
        {
            return sample.GetLogAverageOf(to_sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample,Probs) p(Sample,Probs) factor(Sample,Probs))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(int sample, Dirichlet probs)
        {
            Discrete to_sample = SampleAverageConditional(probs, Discrete.Uniform(probs.Dimension, probs.Sparsity));
            return to_sample.GetLogProb(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample) p(Sample) factor(Sample,Probs))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(int sample, Vector probs)
        {
            return Math.Log(probs[sample]);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample,Probs) p(Sample,Probs) factor(Sample,Probs) / sum_Sample p(Sample) messageTo(Sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int sample, Dirichlet probs)
        {
            return LogAverageFactor(sample, probs);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample) p(Sample) factor(Sample,Probs) / sum_Sample p(Sample) messageTo(Sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int sample, Vector probs)
        {
            return LogAverageFactor(sample, probs);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample,Probs) p(Sample,Probs) factor(Sample,Probs) / sum_Sample p(Sample) messageTo(Sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Discrete sample, Dirichlet probs)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sample) p(Sample) factor(Sample,Probs) / sum_Sample p(Sample) messageTo(Sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Discrete sample, Vector probs)
        {
            return 0.0;
        }

        /// <summary>Gibbs message to <c>Sample</c>.</summary>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Discrete SampleConditional(Vector probs, Discrete result)
        {
            result.SetProbs(probs);
            return result;
        }

        /// <summary>EP message to <c>Sample</c>.</summary>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Discrete SampleAverageConditional(Vector probs, Discrete result)
        {
            return SampleConditional(probs, result);
        }

        /// <summary>VMP message to <c>Sample</c>.</summary>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Discrete SampleAverageLogarithm(Vector probs, Discrete result)
        {
            return SampleConditional(probs, result);
        }

        /// <summary />
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Discrete SampleAverageConditionalInit([IgnoreDependency] Dirichlet probs)
        {
            return Discrete.Uniform(probs.Dimension);
        }

        /// <summary />
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Discrete SampleAverageLogarithmInit([IgnoreDependency] Dirichlet probs)
        {
            return Discrete.Uniform(probs.Dimension);
        }

        /// <summary />
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Discrete SampleAverageConditionalInit([IgnoreDependency] Vector probs)
        {
            return Discrete.Uniform(probs.Count);
        }

        /// <summary />
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Discrete SampleAverageLogarithmInit([IgnoreDependency] Vector probs)
        {
            return Discrete.Uniform(probs.Count);
        }

        /// <summary>Gibbs message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Probs</c> conditioned on the given values.</para>
        /// </remarks>
        public static Dirichlet ProbsConditional(int sample, Dirichlet result)
        {
            result.TotalCount = result.Dimension + 1;
            result.PseudoCount.SetAllElementsTo(1);
            result.PseudoCount[sample] = 2;
            return result;
        }

        /// <summary>EP message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Probs</c> as the random arguments are varied. The formula is <c>proj[p(Probs) sum_(Sample) p(Sample) factor(Sample,Probs)]/p(Probs)</c>.</para>
        /// </remarks>
        public static Dirichlet ProbsAverageConditional(int sample, Dirichlet result)
        {
            return ProbsConditional(sample, result);
        }

        /// <summary>VMP message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Probs</c>. The formula is <c>exp(sum_(Sample) p(Sample) log(factor(Sample,Probs)))</c>.</para>
        /// </remarks>
        public static Dirichlet ProbsAverageLogarithm(int sample, Dirichlet result)
        {
            return ProbsConditional(sample, result);
        }

        /// <summary>EP message to <c>Sample</c>.</summary>
        /// <param name="probs">Incoming message from <c>Probs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Sample</c> as the random arguments are varied. The formula is <c>proj[p(Sample) sum_(Probs) p(Probs) factor(Sample,Probs)]/p(Sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probs" /> is not a proper distribution.</exception>
        public static Discrete SampleAverageConditional([SkipIfUniform] Dirichlet probs, Discrete result)
        {
            result.SetProbs(probs.GetMean(result.GetWorkspace()));
            return result;
        }

        /// <summary>VMP message to <c>Sample</c>.</summary>
        /// <param name="probs">Incoming message from <c>Probs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Sample</c>. The formula is <c>exp(sum_(Probs) p(Probs) log(factor(Sample,Probs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probs" /> is not a proper distribution.</exception>
        public static Discrete SampleAverageLogarithm([SkipIfUniform] Dirichlet probs, Discrete result)
        {
            // E[sum_k I(X=k) log(P[k])] = sum_k I(X=k) E[log(P[k])]
            Vector p = probs.GetMeanLog(result.GetWorkspace());
            double max = p.Max();
            p.SetToFunction(p, x => Math.Exp(x - max));
            result.SetProbs(p);
            return result;
        }

        /// <summary>VMP message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>Probs</c>. The formula is <c>exp(sum_(Sample) p(Sample) log(factor(Sample,Probs)))</c>.</para>
        /// </remarks>
        public static Dirichlet ProbsAverageLogarithm(Discrete sample, Dirichlet result)
        {
            // E[sum_k I(X=k) log(P[k])] = sum_k p(X=k) log(P[k])
            result.TotalCount = result.Dimension + 1;
            result.PseudoCount.SetAllElementsTo(1);
            result.PseudoCount.SetToSum(result.PseudoCount, sample.GetProbs());
            return result;
        }

        /// <summary>EP message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Probs</c> as the random arguments are varied. The formula is <c>proj[p(Probs) sum_(Sample) p(Sample) factor(Sample,Probs)]/p(Probs)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static Dirichlet ProbsAverageConditional([SkipIfUniform] Discrete sample, Vector probs, Dirichlet result)
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>EP message to <c>Probs</c>.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Probs</c> as the random arguments are varied. The formula is <c>proj[p(Probs) sum_(Sample) p(Sample) factor(Sample,Probs)]/p(Probs)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static Dirichlet ProbsAverageConditional([SkipIfUniform] Discrete sample, Dirichlet probs, Dirichlet result)
        {
            if (probs.IsPointMass)
                return ProbsAverageConditional(sample, probs.Point, result);
            if (sample.IsPointMass)
                return ProbsConditional(sample.Point, result);
            // Z = sum_x q(x) int_p f(x,p)*q(p) = sum_x q(x) E[p[x]]
            Vector sampleProbs = sample.GetProbs();
            double Z = sampleProbs.Inner(probs.PseudoCount);
            double invZ = 1.0 / Z;
            // the posterior is a mixture of Dirichlets having the following form:
            // sum_x q(x) (alpha(x)/sum_i alpha(i)) Dirichlet(p; alpha(x)+1, alpha(not x)+0)
            // where the mixture weights are w(x) =propto q(x) alpha(x)
            //                               w[i] = sample[i]*probs.PseudoCount[i]/Z
            // The posterior mean of probs(x) = (w(x) + alpha(x))/(1 + sum_x alpha(x))
            double invTotalCountPlus1 = 1.0 / (probs.TotalCount + 1);
            Vector m = Vector.Zero(sample.Dimension, sample.Sparsity);
            m.SetToFunction(sampleProbs, probs.PseudoCount, (x, y) => (x * invZ + 1.0) * y * invTotalCountPlus1);
            if (!Dirichlet.AllowImproperSum)
            {
                // To get the correct mean, we need (probs.PseudoCount[i] + delta[i]) to be proportional to m[i].
                // If we set delta[argmin] = 0, then we just solve the equation 
                //   (probs.PseudoCount[i] + delta[i])/probs.PseudoCount[argmin] = m[i]/m[argmin]
                // for delta[i].
                int argmin = sampleProbs.IndexOfMinimum();
                Debug.Assert(argmin != -1);
                double newTotalCount = probs.PseudoCount[argmin] / m[argmin];
                double argMinValue = sampleProbs[argmin];
                result.PseudoCount.SetToFunction(m, probs.PseudoCount, (x, y) => 1.0 + (x * newTotalCount) - y);
                result.PseudoCount.SetToFunction(result.PseudoCount, sampleProbs, (x, y) => (y == argMinValue) ? 1.0 : x);
                result.TotalCount = result.PseudoCount.Sum(); // result.Dimension + newTotalCount - probs.TotalCount;
                return result;
            }
            else
            {
                // The posterior meanSquare of probs(x) = (2 w(x) + alpha(x))/(2 + sum_x alpha(x)) * (1 + alpha(x))/(1 + sum_x alpha(x))
                double invTotalCountPlus2 = 1.0 / (2 + probs.TotalCount);
                Vector m2 = Vector.Zero(sample.Dimension, sample.Sparsity);
                m2.SetToFunction(sampleProbs, probs.PseudoCount, (x, y) => (2.0 * x * invZ + 1.0) * y * invTotalCountPlus2 * (1.0 + y) * invTotalCountPlus1);
                result.SetMeanAndMeanSquare(m, m2);
                result.SetToRatio(result, probs);
                return result;
            }
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(Sample,Probs) p(Sample,Probs) log(factor(Sample,Probs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probs" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(Discrete sample, [Proper] Dirichlet probs)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, probs);

            if (sample.Dimension != probs.Dimension)
                throw new ArgumentException("sample.Dimension (" + sample.Dimension + ") != probs.Dimension (" + probs.Dimension + ")");
            Vector sampleProbs = sample.GetProbs();
            Vector pSuffStats = probs.GetMeanLog();
            // avoid multiplication of 0*log(0)
            foreach (int i in sampleProbs.IndexOfAll(v => v == 0.0))
                pSuffStats[i] = 0.0;
            double total = Vector.InnerProduct(sampleProbs, pSuffStats);
            return total;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(Sample) p(Sample) log(factor(Sample,Probs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Discrete sample, Vector probs)
        {
            return AverageLogFactor(sample, Dirichlet.PointMass(probs));
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Incoming message from <c>Probs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(Sample,Probs) p(Sample,Probs) log(factor(Sample,Probs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probs" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(int sample, [Proper] Dirichlet probs)
        {
            return probs.GetMeanLogAt(sample);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>Sample</c>.</param>
        /// <param name="probs">Constant value for <c>Probs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(Sample) p(Sample) log(factor(Sample,Probs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(int sample, Vector probs)
        {
            return Math.Log(probs[sample]);
        }
    }
}
