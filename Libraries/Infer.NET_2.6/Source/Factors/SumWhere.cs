// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Linq;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.SumWhere(bool[], Vector)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "SumWhere")]
    [Quality(QualityBand.Preview)]
    [Buffers("CovarianceOfB", "MeanOfB")]
    public static class SumWhereOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Sum,A,B))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double sum, bool[] A, Vector B)
        {
            return (sum == Factor.SumWhere(A, B)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Sum,A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double sum, bool[] A, Vector B)
        {
            return LogAverageFactor(sum, A, B);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Sum,A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(double sum, bool[] A, Vector B)
        {
            return LogAverageFactor(sum, A, B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sum) p(Sum) factor(Sum,A,B))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian sum, bool[] A, Vector B)
        {
            return sum.GetLogProb(Factor.SumWhere(A, B));
        }

        /// <summary>EP message to <c>B</c>.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>B</c> as the random arguments are varied. The formula is <c>proj[p(B) sum_(Sum) p(Sum) factor(Sum,A,B)]/p(B)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        public static VectorGaussian BAverageConditional([SkipIfUniform] Gaussian sum, bool[] A, VectorGaussian result)
        {
            if (result == default(VectorGaussian))
                result = new VectorGaussian(A.Length);
            // (m - a'b)^2/v = (a'bb'a - 2a'bm + m^2)/v
            var ma = Vector.FromArray(A.Select(x => x ? 1.0 : 0.0).ToArray());
            result.Precision.SetToOuter(ma, ma);
            result.Precision.Scale(sum.Precision);
            result.MeanTimesPrecision.SetToProduct(ma, sum.MeanTimesPrecision);
            return result;
        }

        /// <summary>EP message to <c>Sum</c>.</summary>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanOfB">Buffer <c>MeanOfB</c>.</param>
        /// <param name="CovarianceOfB">Buffer <c>CovarianceOfB</c>.</param>
        /// <returns>The outgoing EP message to the <c>Sum</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Sum</c> as the random arguments are varied. The formula is <c>proj[p(Sum) sum_(B) p(B) factor(Sum,A,B)]/p(Sum)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian SumAverageConditional(bool[] A, [SkipIfUniform] VectorGaussian B, Vector MeanOfB, PositiveDefiniteMatrix CovarianceOfB)
        {
            return SumAverageLogarithm(A, B, MeanOfB, CovarianceOfB);
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gaussian SumAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>.</param>
        /// <param name="to_sum">Outgoing message to <c>sum</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Sum) p(Sum) factor(Sum,A,B))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian sum, [Fresh] Gaussian to_sum)
        {
            return to_sum.GetLogAverageOf(sum);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanOfB">Buffer <c>MeanOfB</c>.</param>
        /// <param name="CovarianceOfB">Buffer <c>CovarianceOfB</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(B) p(B) factor(Sum,A,B))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(
            double sum, bool[] A, [SkipIfUniform] VectorGaussian B, Vector MeanOfB, PositiveDefiniteMatrix CovarianceOfB)
        {
            Gaussian to_sum = SumAverageConditional(A, B, MeanOfB, CovarianceOfB);
            return to_sum.GetLogProb(sum);
        }

        //-- VMP ---------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Sum,A,B))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>Update the buffer <c>CovarianceOfB</c>.</summary>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>New value of buffer <c>CovarianceOfB</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static PositiveDefiniteMatrix CovarianceOfB([Proper] VectorGaussian B)
        {
            return B.GetVariance();
        }

        /// <summary>Update the buffer <c>MeanOfB</c>.</summary>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="CovarianceOfB">Buffer <c>CovarianceOfB</c>.</param>
        /// <returns>New value of buffer <c>MeanOfB</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Vector MeanOfB([Proper] VectorGaussian B, PositiveDefiniteMatrix CovarianceOfB)
        {
            return CovarianceOfB * B.MeanTimesPrecision;
        }

        /// <summary>VMP message to <c>Sum</c>.</summary>
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanOfB">Buffer <c>MeanOfB</c>.</param>
        /// <param name="CovarianceOfB">Buffer <c>CovarianceOfB</c>.</param>
        /// <returns>The outgoing VMP message to the <c>Sum</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Sum</c> as the random arguments are varied. The formula is <c>proj[sum_(A,B) p(A,B) factor(Sum,A,B)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian SumAverageLogarithm(
            DistributionStructArray<Bernoulli, bool> A, [SkipIfUniform] VectorGaussian B, Vector MeanOfB, PositiveDefiniteMatrix CovarianceOfB)
        {
            Gaussian result = new Gaussian();
            // p(x|a,b) = N(E[a]'*E[b], E[b]'*var(a)*E[b] + E[a]'*var(b)*E[a] + trace(var(a)*var(b)))
            Vector ma = Vector.Zero(A.Count);
            Vector va = Vector.Zero(A.Count);
            for (int i = 0; i < A.Count; i++)
            {
                ma[i] = A[i].GetMean();
                va[i] = A[i].GetVariance();
            }
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            var MeanOfBSquared = Vector.Zero(MeanOfB.Count);
            MeanOfBSquared.SetToFunction(MeanOfB, x => x * x);
            result.SetMeanAndVariance(ma.Inner(MeanOfB), va.Inner(MeanOfBSquared) + CovarianceOfB.QuadraticForm(ma) + va.Inner(CovarianceOfB.Diagonal()));
            return result;
        }

        /// <summary>VMP message to <c>Sum</c>.</summary>
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <returns>The outgoing VMP message to the <c>Sum</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Sum</c> as the random arguments are varied. The formula is <c>proj[sum_(A) p(A) factor(Sum,A,B)]</c>.</para>
        /// </remarks>
        public static Gaussian SumAverageLogarithm(DistributionStructArray<Bernoulli, bool> A, [SkipIfUniform] Vector B)
        {
            Gaussian result = new Gaussian();
            // p(x|a,b) = N(E[a]'*E[b], E[b]'*var(a)*E[b] + E[a]'*var(b)*E[a] + trace(var(a)*var(b)))
            Vector ma = Vector.Zero(A.Count);
            Vector va = Vector.Zero(A.Count);
            for (int i = 0; i < A.Count; i++)
            {
                ma[i] = A[i].GetMean();
                va[i] = A[i].GetVariance();
            }
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            var MeanOfBSquared = Vector.Zero(B.Count);
            MeanOfBSquared.SetToFunction(B, x => x * x);
            result.SetMeanAndVariance(ma.Inner(B), va.Inner(MeanOfBSquared));
            return result;
        }

        /// <summary>VMP message to <c>Sum</c>.</summary>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanOfB">Buffer <c>MeanOfB</c>.</param>
        /// <param name="CovarianceOfB">Buffer <c>CovarianceOfB</c>.</param>
        /// <returns>The outgoing VMP message to the <c>Sum</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Sum</c> as the random arguments are varied. The formula is <c>proj[sum_(B) p(B) factor(Sum,A,B)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian SumAverageLogarithm(bool[] A, [SkipIfUniform] VectorGaussian B, Vector MeanOfB, PositiveDefiniteMatrix CovarianceOfB)
        {
            Gaussian result = new Gaussian();
            // p(x|a,b) = N(E[a]'*E[b], E[b]'*var(a)*E[b] + E[a]'*var(b)*E[a] + trace(var(a)*var(b)))
            Vector ma = Vector.FromArray(A.Select(x => x ? 1.0 : 0.0).ToArray());
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            result.SetMeanAndVariance(ma.Inner(MeanOfB), CovarianceOfB.QuadraticForm(ma));
            return result;
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gaussian SumAverageLogarithmInit()
        {
            return Gaussian.Uniform();
        }

        /// <summary>VMP message to <c>Sum</c>.</summary>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <returns>The outgoing VMP message to the <c>Sum</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Sum</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian SumAverageLogarithm(bool[] A, Vector B)
        {
            Vector ma = Vector.FromArray(A.Select(x => x ? 1.0 : 0.0).ToArray());
            return Gaussian.PointMass(ma.Inner(B));
        }

        /// <summary>VMP message to <c>B</c>.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>B</c> with <c>Sum</c> integrated out. The formula is <c>sum_Sum p(Sum) factor(Sum,A,B)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        public static VectorGaussian BAverageLogarithm([SkipIfUniform] Gaussian sum, bool[] A, VectorGaussian result)
        {
            if (result == default(VectorGaussian))
                result = new VectorGaussian(A.Length);
            // E[log N(x; ab, 0)] = -0.5 E[(x-ab)^2]/0 = -0.5 (E[x^2] - 2 E[x] a' E[b] + trace(aa' E[bb']))/0
            // message to a = N(a; E[x]*inv(var(b)+E[b]E[b]')*E[b], var(x)*inv(var(b)+E[b]E[b]'))
            // result.Precision = (var(b)+E[b]*E[b]')/var(x)
            // result.MeanTimesPrecision = E[x]/var(x)*E[b] = E[b]*X.MeanTimesPrecision
            Vector ma = Vector.FromArray(A.Select(x => x ? 1.0 : 0.0).ToArray());
            // note this is exact if B is a point mass (vb=0).
            result.Precision.SetToOuter(ma, ma);
            result.Precision.Scale(sum.Precision);
            result.MeanTimesPrecision.SetToProduct(ma, sum.MeanTimesPrecision);
            return result;
        }

        /// <summary>VMP message to <c>B</c>.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>A</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>B</c>. Because the factor is deterministic, <c>Sum</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(A) p(A) log(sum_Sum p(Sum) factor(Sum,A,B)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        public static VectorGaussian BAverageLogarithm([SkipIfUniform] Gaussian sum, DistributionStructArray<Bernoulli, bool> A, VectorGaussian result)
        {
            if (result == default(VectorGaussian))
                result = new VectorGaussian(A.Count);
            // E[log N(x; ab, 0)] = -0.5 E[(x-ab)^2]/0 = -0.5 (E[x^2] - 2 E[x] a' E[b] + trace(aa' E[bb']))/0
            // message to a = N(a; E[x]*inv(var(b)+E[b]E[b]')*E[b], var(x)*inv(var(b)+E[b]E[b]'))
            // result.Precision = (var(b)+E[b]*E[b]')/var(x)
            // result.MeanTimesPrecision = E[x]/var(x)*E[b] = E[b]*X.MeanTimesPrecision
            Vector ma = Vector.FromArray(A.Select(x => x.GetMean()).ToArray());
            Vector va = Vector.FromArray(A.Select(x => x.GetVariance()).ToArray());
            result.Precision.SetToDiagonal(va);
            result.Precision.SetToSumWithOuter(result.Precision, 1, ma, ma);
            result.Precision.SetToProduct(result.Precision, sum.Precision);
            result.MeanTimesPrecision.SetToProduct(ma, sum.MeanTimesPrecision);
            return result;
        }

        /// <summary>VMP message to <c>A</c>.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>A</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanOfB">Buffer <c>MeanOfB</c>.</param>
        /// <param name="CovarianceOfB">Buffer <c>CovarianceOfB</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>A</c>. Because the factor is deterministic, <c>Sum</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(B) p(B) log(sum_Sum p(Sum) factor(Sum,A,B)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static DistributionStructArray<Bernoulli, bool> AAverageLogarithm(
            [SkipIfUniform] Gaussian sum,
            [Proper] DistributionStructArray<Bernoulli, bool> A,
            [Proper] VectorGaussian B,
            Vector MeanOfB,
            PositiveDefiniteMatrix CovarianceOfB,
            DistributionStructArray<Bernoulli, bool> result)
        {
            double mu = sum.GetMean();
            var Ebbt = new PositiveDefiniteMatrix(A.Count, A.Count);
            Ebbt.SetToSumWithOuter(CovarianceOfB, 1, MeanOfB, MeanOfB);
            var ma = A.Select(x => x.GetMean()).ToArray();
            for (int i = 0; i < A.Count; i++)
            {
                double term1 = 0.0;
                for (int j = 0; j < A.Count; j++)
                    if (j != i)
                        term1 += ma[i] * Ebbt[i, j];
                var ri = result[i];
                ri.LogOdds = -.5 * sum.Precision * (2.0 * term1 + Ebbt[i, i] - 2.0 * mu * MeanOfB[i]);
                result[i] = ri;
            }
            return result;
        }

        /// <summary>VMP message to <c>A</c>.</summary>
        /// <param name="sum">Incoming message from <c>Sum</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>A</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>A</c> with <c>Sum</c> integrated out. The formula is <c>sum_Sum p(Sum) factor(Sum,A,B)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static DistributionStructArray<Bernoulli, bool> AAverageLogarithm(
            [SkipIfUniform] Gaussian sum, [Proper] DistributionStructArray<Bernoulli, bool> A, Vector B, DistributionStructArray<Bernoulli, bool> result)
        {
            double mu = sum.GetMean();
            var ma = A.Select(x => x.GetMean()).ToArray();
            for (int i = 0; i < A.Count; i++)
            {
                double term1 = 0.0;
                for (int j = 0; j < A.Count; j++)
                    if (j != i)
                        term1 += ma[i] * B[i] * B[j];
                var ri = result[i];
                ri.LogOdds = -.5 * sum.Precision * (2.0 * term1 + B[i] * B[i] - 2.0 * mu * B[i]);
                result[i] = ri;
            }
            return result;
        }

        private const string NotSupportedMessage = "A SumWhere factor with fixed output is not supported";

        /// <summary>VMP message to <c>A</c>.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Incoming message from <c>A</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>A</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>A</c>. The formula is <c>exp(sum_(B) p(B) log(factor(Sum,A,B)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static DistributionStructArray<Bernoulli, bool> AAverageLogarithm(double sum, [Proper] DistributionStructArray<Bernoulli, bool> A, [Proper] VectorGaussian B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>A</c>.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Incoming message from <c>A</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>B</c>.</param>
        /// <returns>The outgoing VMP message to the <c>A</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>A</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static DistributionStructArray<Bernoulli, bool> AAverageLogarithm(double sum, [Proper] DistributionStructArray<Bernoulli, bool> A, Vector B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>B</c>.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>B</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>B</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static VectorGaussian BAverageLogarithm([SkipIfUniform] double sum, bool[] A, [SkipIfUniform] VectorGaussian B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
#pragma warning disable 1591
        /// <summary>EP message to <c>B</c>.</summary>
        /// <param name="sum">Constant value for <c>Sum</c>.</param>
        /// <param name="A">Constant value for <c>A</c>.</param>
        /// <param name="B">Incoming message from <c>B</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>B</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>B</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static VectorGaussian BAverageConditional([SkipIfUniform] double sum, bool[] A, [SkipIfUniform] VectorGaussian B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }
}
