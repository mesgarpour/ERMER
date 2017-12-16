// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.IsBetween(double, double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "IsBetween", typeof(double), typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("logZ")]
    public static class DoubleIsBetweenOp
    {
        /// <summary>
        /// Static flag to force a proper distribution
        /// </summary>
        public static bool ForceProper = true;

        //-- TruncatedGaussian ------------------------------------------------------------------------------

        /// <summary>EP message to <c>x</c>.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>The outgoing EP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>x</c> conditioned on the given values.</para>
        /// </remarks>
        public static TruncatedGaussian XAverageConditional(bool isBetween, double lowerBound, double upperBound)
        {
            if (!isBetween)
                throw new ArgumentException("TruncatedGaussian requires isBetween=true", "isBetween");
            return new TruncatedGaussian(0, Double.PositiveInfinity, lowerBound, upperBound);
        }

        /// <summary>EP message to <c>lowerBound</c>.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="x">Constant value for <c>x</c>.</param>
        /// <returns>The outgoing EP message to the <c>lowerBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>lowerBound</c> conditioned on the given values.</para>
        /// </remarks>
        public static TruncatedGaussian LowerBoundAverageConditional(bool isBetween, double x)
        {
            if (!isBetween)
                throw new ArgumentException("TruncatedGaussian requires isBetween=true", "isBetween");
            return new TruncatedGaussian(0, Double.PositiveInfinity, Double.NegativeInfinity, x);
        }

        /// <summary>EP message to <c>upperBound</c>.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="x">Constant value for <c>x</c>.</param>
        /// <returns>The outgoing EP message to the <c>upperBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>upperBound</c> conditioned on the given values.</para>
        /// </remarks>
        public static TruncatedGaussian UpperBoundAverageConditional(bool isBetween, double x)
        {
            if (!isBetween)
                throw new ArgumentException("TruncatedGaussian requires isBetween=true", "isBetween");
            return new TruncatedGaussian(0, Double.PositiveInfinity, x, Double.PositiveInfinity);
        }

        //-- Constant bounds --------------------------------------------------------------------------------

        /// <summary>
        /// The logarithm of the probability that L &lt;= X &lt; U.
        /// </summary>
        /// <param name="X"></param>
        /// <param name="L">Can be negative infinity.</param>
        /// <param name="U">Can be positive infinity.</param>
        /// <returns></returns>
        public static double LogProbBetween(Gaussian X, double L, double U)
        {
            if (L > U)
                throw new AllZeroException("low > high (" + L + " > " + U + ")");
            if (X.IsPointMass)
            {
                return Factor.IsBetween(X.Point, L, U) ? 0.0 : Double.NegativeInfinity;
            }
            else if (X.IsUniform())
            {
                if (Double.IsNegativeInfinity(L))
                {
                    if (Double.IsPositiveInfinity(U))
                        return 0.0; // always between
                    else
                        return -MMath.Ln2; // between half the time
                }
                else if (Double.IsPositiveInfinity(U))
                    return -MMath.Ln2; // between half the time
                else
                    return Double.NegativeInfinity; // never between two finite numbers
            }
            else
            {
                double sqrtPrec = Math.Sqrt(X.Precision);
                double mx = X.GetMean();
                double Ldiff = L - mx;
                double Udiff = U - mx;
                // we can either compute as:  p(x <= U) - p(x <= L)
                // or:  p(x > L) - p(x > U)
                // depending on whether the bigger absolute value is negative or positive
                bool LdiffIsBigger = Math.Abs(Ldiff) > Math.Abs(Udiff);
                bool biggerIsNegative;
                if (LdiffIsBigger)
                    biggerIsNegative = (Ldiff < 0);
                else
                    biggerIsNegative = (Udiff < 0);
                if (biggerIsNegative)
                {
                    // compute p(x <= U) - p(x <= L)
                    double pl = MMath.NormalCdfLn(sqrtPrec * (L - mx)); // log(p(x <= L))
                    double pu = MMath.NormalCdfLn(sqrtPrec * (U - mx)); // log(p(x <= U))
                    return MMath.LogDifferenceOfExp(pu, pl);
                }
                else
                {
                    // compute p(x > L) - p(x > U)
                    double pl = MMath.NormalCdfLn(sqrtPrec * (mx - L)); // log(p(x > L))
                    double pu = MMath.NormalCdfLn(sqrtPrec * (mx - U)); // log(p(x > U))
                    return MMath.LogDifferenceOfExp(pl, pu);
                }
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="x">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(isBetween,x) p(isBetween,x) factor(isBetween,x,lowerBound,upperBound))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli isBetween, [RequiredArgument] Gaussian x, double lowerBound, double upperBound)
        {
#if true
            Bernoulli to_isBetween = IsBetweenAverageConditional(x, lowerBound, upperBound);
            return to_isBetween.GetLogAverageOf(isBetween);
#else
			if (isBetween.LogOdds == 0.0) return -MMath.Ln2;
			else {
				double logitProbBetween = MMath.LogitFromLog(LogProbBetween(x, lowerBound, upperBound));
				return Bernoulli.LogProbEqual(isBetween.LogOdds, logitProbBetween);
			}
#endif
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="x">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(x) p(x) factor(isBetween,x,lowerBound,upperBound))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool isBetween, [RequiredArgument] Gaussian x, double lowerBound, double upperBound)
        {
            return LogAverageFactor(isBetween, x, lowerBound, upperBound);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="x">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(x) p(x) factor(isBetween,x,lowerBound,upperBound))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool isBetween, [RequiredArgument] Gaussian x, double lowerBound, double upperBound)
        {
            return LogAverageFactor(Bernoulli.PointMass(isBetween), x, lowerBound, upperBound);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="x">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(isBetween,x) p(isBetween,x) factor(isBetween,x,lowerBound,upperBound) / sum_isBetween p(isBetween) messageTo(isBetween))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli isBetween, Gaussian x, double lowerBound, double upperBound)
        {
            return 0.0;
        }

        /// <summary>EP message to <c>isBetween</c>.</summary>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>The outgoing EP message to the <c>isBetween</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>isBetween</c> as the random arguments are varied. The formula is <c>proj[p(isBetween) sum_(x) p(x) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.</para>
        /// </remarks>
        public static Bernoulli IsBetweenAverageConditional([RequiredArgument] Gaussian X, double lowerBound, double upperBound)
        {
            Bernoulli result = new Bernoulli();
            result.SetLogProbTrue(LogProbBetween(X, lowerBound, upperBound));
            return result;
        }

        /// <summary>EP message to <c>x</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>The outgoing EP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>x</c> as the random arguments are varied. The formula is <c>proj[p(x) sum_(isBetween) p(isBetween) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="isBetween" /> is not a proper distribution.</exception>
        public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, double lowerBound, double upperBound)
        {
            // TM: RequiredArgument is included to get good schedules
            // Note that SkipIfUniform would not be correct in this case
            Gaussian result = new Gaussian();
            if (X.IsPointMass)
            {
                result.SetToUniform();
            }
            else if (X.IsUniform())
            {
                if (Double.IsInfinity(lowerBound) || Double.IsInfinity(upperBound) ||
                    !Double.IsPositiveInfinity(isBetween.LogOdds))
                {
                    result.SetToUniform();
                }
                else
                {
                    double diff = upperBound - lowerBound;
                    result.SetMeanAndVariance((lowerBound + upperBound) / 2, diff * diff / 12);
                }
            }
            else
            {
                // X is not a point mass or uniform
                double mx, vx;
                X.GetMeanAndVariance(out mx, out vx);
                double logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                double logPhiL = Gaussian.GetLogProb(lowerBound, mx, vx);
                double alphaL = d_p * Math.Exp(logPhiL - logZ);
                double logPhiU = Gaussian.GetLogProb(upperBound, mx, vx);
                double alphaU = d_p * Math.Exp(logPhiU - logZ);
                if (logPhiL - logPhiU > 50 && d_p == 1.0)
                {
                    // factor reduces to (x > lowerBound)
                    double prec = X.Precision;
                    double sqrtPrec = Math.Sqrt(X.Precision);
                    //double z = X.MeanTimesPrecision - lowerBound * prec;
                    double z = (mx - lowerBound) * sqrtPrec;
                    // logZ = NormalCdfLn(z)
                    //Console.WriteLine("{0} {1}", logZ, MMath.NormalCdfLn(z));
                    // alphaL = sqrtPrec/NormalCdfRatio(z)
                    //Console.WriteLine("{0} {1}", alphaL, Math.Sqrt(X.Precision) / MMath.NormalCdfRatio(z));
                    if (z < -1e10)
                    {
                        double mp = -1 / (z * sqrtPrec);
                        double vp = 1 / (z * z * prec);
                        return (new Gaussian(mp + lowerBound, vp)) / X;
                    }
                    else if (z < -100)
                    {
                        // this code is from IsPositiveOp
                        double Y = MMath.NormalCdfRatio(z);
                        // double dY = MMath.NormalCdfMomentRatio(1, z);
                        // dY = z*Y + 1
                        // d2Y = z*dY + Y
                        // posterior mean = m + sqrt(v)/Y = sqrt(v)*(z + 1/Y) + L = sqrt(v)*dY/Y + L =approx sqrt(v)*(-1/z) + L
                        // posterior variance = v - v*dY/Y^2 =approx v/z^2
                        // posterior E[(x-L)^2] = v - v*dY/Y^2 + v*dY^2/Y^2 = v - v*dY/Y^2*(1 - dY) = v + v*z*dY/Y = v*d2Y/Y
                        double d2Y = 2 * MMath.NormalCdfMomentRatio(2, z);
                        // m2TimesPrec = z*dY/Y + 1
                        double m2TimesPrec = d2Y / Y;
                        double mp = (m2TimesPrec - 1) / (z * sqrtPrec);
                        double vp = m2TimesPrec / prec - mp * mp;
                        return (new Gaussian(mp + lowerBound, vp)) / X;
                    }
                    alphaL = sqrtPrec / MMath.NormalCdfRatio(z);
                }
                else if (logPhiU - logPhiL > 50 && d_p == 1.0)
                {
                    // factor reduces to (x < upperBound)
                    double prec = X.Precision;
                    double sqrtPrec = Math.Sqrt(X.Precision);
                    double z = (upperBound - mx) * sqrtPrec;
                    // logZ = NormalCdfLn(z)
                    //Console.WriteLine("{0} {1}", logZ, MMath.NormalCdfLn(z));
                    // alphaU = sqrtPrec/NormalCdfRatio(z)
                    //Console.WriteLine("{0} {1}", alphaU, Math.Sqrt(X.Precision) / MMath.NormalCdfRatio(z));
                    if (z < -1e10)
                    {
                        double mp = -1 / (z * sqrtPrec);
                        double vp = 1 / (z * z * prec);
                        return (new Gaussian(upperBound - mp, vp)) / X;
                    }
                    else if (z < -100)
                    {
                        // this code is from IsPositiveOp
                        double Y = MMath.NormalCdfRatio(z);
                        // double dY = MMath.NormalCdfMomentRatio(1, z);
                        // dY = z*Y + 1
                        // d2Y = z*dY + Y
                        // posterior mean = m - sqrt(v)/Y = U - sqrt(v)*(z + 1/Y) = U - sqrt(v)*dY/Y =approx U - sqrt(v)*(-1/z)
                        // posterior variance = v - v*dY/Y^2 =approx v/z^2
                        // posterior E[(x-U)^2] = v - v*dY/Y^2 + v*dY^2/Y^2 = v - v*dY/Y^2*(1 - dY) = v + v*z*dY/Y = v*d2Y/Y
                        double d2Y = 2 * MMath.NormalCdfMomentRatio(2, z);
                        // m2TimesPrec = z*dY/Y + 1
                        double m2TimesPrec = d2Y / Y;
                        double mp = (m2TimesPrec - 1) / (z * sqrtPrec);
                        double vp = m2TimesPrec / prec - mp * mp;
                        return (new Gaussian(mp + upperBound, vp)) / X;
                    }
                    alphaU = sqrtPrec / MMath.NormalCdfRatio(z);
                }
                double alphaX = alphaL - alphaU;
#if false
    // minka: testing numerical accuracy
				double diff = upperBound - lowerBound;
				double center = (lowerBound+upperBound)/2;
				double logNdiff = diff*(-X.MeanTimesPrecision + center*X.Precision);
				//logNdiff = logPhiL - logPhiU;
				if(logNdiff >= 0) {
					alphaX = d_p*Math.Exp(MMath.LogExpMinus1(logNdiff) + logPhiU-logZ);
				} else {
					alphaX = -d_p*Math.Exp(MMath.LogExpMinus1(-logNdiff) + logPhiL-logZ);
				}
				double m = mx + vx*alphaX;
#endif
                double betaX = alphaX * alphaX;
                if (alphaU != 0.0)
                    betaX += (upperBound - mx) / vx * alphaU;
                if (alphaL != 0.0)
                    betaX -= (lowerBound - mx) / vx * alphaL;
                return GaussianProductOp_Laplace.GaussianFromAlphaBeta(X, alphaX, betaX, ForceProper);
            }
            return result;
        }

        /// <summary>EP message to <c>x</c>.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>The outgoing EP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>x</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, double lowerBound, double upperBound)
        {
            return XAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
        }

        //-- Random bounds --------------------------------------------------------------------------------

        // Compute the mean and variance of (X-L) and (U-X)
        internal static void GetDiffMeanAndVariance(Gaussian X, Gaussian L, Gaussian U, out double yl, out double yu, out double r, out double invSqrtVxl,
                                                    out double invSqrtVxu)
        {
            double mx, vx, ml, vl, mu, vu;
            X.GetMeanAndVariance(out mx, out vx);
            L.GetMeanAndVariance(out ml, out vl);
            U.GetMeanAndVariance(out mu, out vu);
            if (X.IsPointMass && L.IsPointMass)
            {
                invSqrtVxl = Double.PositiveInfinity;
                yl = (X.Point >= L.Point) ? Double.PositiveInfinity : Double.NegativeInfinity;
            }
            else if (L.IsUniform())
            {
                invSqrtVxl = 0.0;
                yl = 0;
            }
            else
            {
                invSqrtVxl = 1.0 / Math.Sqrt(vx + vl);
                yl = (mx - ml) * invSqrtVxl;
            }
            if (X.IsPointMass && U.IsPointMass)
            {
                invSqrtVxu = Double.PositiveInfinity;
                yu = (X.Point < U.Point) ? Double.PositiveInfinity : Double.NegativeInfinity;
            }
            else if (U.IsUniform())
            {
                invSqrtVxu = 0.0;
                yu = 0;
            }
            else
            {
                invSqrtVxu = 1.0 / Math.Sqrt(vx + vu);
                yu = (mu - mx) * invSqrtVxu;
            }
            if (X.IsPointMass)
            {
                r = 0.0;
            }
            else
            {
                //r = -vx * invSqrtVxl * invSqrtVxu;
                // This is a more accurate way to compute the above.
                r = -1 / Math.Sqrt(1 + vl / vx) / Math.Sqrt(1 + vu / vx);
                if (r < -1 || r > 1)
                    throw new Exception("Internal: r is outside [-1,1]");
            }
        }

        /// <summary>
        /// The logarithm of the probability that L &lt;= X &lt; U.
        /// </summary>
        /// <param name="X"></param>
        /// <param name="L">Can be uniform.  Can be negative infinity.</param>
        /// <param name="U">Can be uniform.  Can be positive infinity.</param>
        /// <returns></returns>
        public static double LogProbBetween(Gaussian X, Gaussian L, Gaussian U)
        {
            if (L.IsPointMass && U.IsPointMass)
                return LogProbBetween(X, L.Point, U.Point);
            if (X.IsUniform())
            {
                if (L.IsPointMass && Double.IsNegativeInfinity(L.Point))
                {
                    if (U.IsPointMass && Double.IsPositiveInfinity(U.Point))
                        return 0.0; // always between
                    else
                        return -MMath.Ln2; // between half the time
                }
                else if (U.IsPointMass && Double.IsPositiveInfinity(U.Point))
                    return -MMath.Ln2; // between half the time
                else if (L.IsUniform() || U.IsUniform())
                {
                    return -2 * MMath.Ln2; // log(0.25)
                }
                else
                {
                    return Double.NegativeInfinity;
                }
            }
            else
            {
                // at this point, X is not uniform
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, L, U, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                return MMath.NormalCdfLn(yl, yu, r);
            }
        }

        /// <summary>Initialize the buffer <c>logZ</c>.</summary>
        /// <returns>Initial value of buffer <c>logZ</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static double LogZInit()
        {
            return 0.0;
        }

        /// <summary>Update the buffer <c>logZ</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>New value of buffer <c>logZ</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static double LogZ(Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return LogAverageFactor(isBetween, X, lowerBound, upperBound);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(isBetween,x,lowerBound,upperBound) p(isBetween,x,lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(
            Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            if (isBetween.LogOdds == 0.0)
                return -MMath.Ln2;
            else
            {
                double logProb = LogProbBetween(X, lowerBound, upperBound);
                double logitProbBetween = MMath.LogitFromLog(logProb);
                return Bernoulli.LogProbEqual(isBetween.LogOdds, logitProbBetween);
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <param name="logZ">Buffer <c>logZ</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(x,lowerBound,upperBound) p(x,lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound, double logZ)
        {
            //return LogAverageFactor(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
            return logZ;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(isBetween,x,lowerBound,upperBound) p(isBetween,x,lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound) / sum_isBetween p(isBetween) messageTo(isBetween))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            return 0.0;
        }

        /// <summary>EP message to <c>isBetween</c>.</summary>
        /// <param name="X">Incoming message from <c>x</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>isBetween</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>isBetween</c> as the random arguments are varied. The formula is <c>proj[p(isBetween) sum_(x,lowerBound,upperBound) p(x,lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="X" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="lowerBound" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="upperBound" /> is not a proper distribution.</exception>
        public static Bernoulli IsBetweenAverageConditional([SkipIfUniform] Gaussian X, [SkipIfUniform] Gaussian lowerBound, [SkipIfUniform] Gaussian upperBound)
        {
            Bernoulli result = new Bernoulli();
            result.SetLogProbTrue(LogProbBetween(X, lowerBound, upperBound));
            return result;
        }

        public static Gaussian LowerBoundAverageConditional_Slow(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return LowerBoundAverageConditional(isBetween, X, lowerBound, upperBound, LogZ(isBetween, X, lowerBound, upperBound));
        }

        /// <summary>EP message to <c>lowerBound</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <param name="logZ">Buffer <c>logZ</c>.</param>
        /// <returns>The outgoing EP message to the <c>lowerBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>lowerBound</c> as the random arguments are varied. The formula is <c>proj[p(lowerBound) sum_(isBetween,x,upperBound) p(isBetween,x,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="isBetween" /> is not a proper distribution.</exception>
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian LowerBoundAverageConditional(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound,
            [Fresh] double logZ)
        {
            Gaussian result = new Gaussian();
            if (isBetween.IsUniform())
                return result;
            if (X.IsUniform())
            {
                if (upperBound.IsUniform() || lowerBound.IsUniform())
                {
                    result.SetToUniform();
                }
                else if (isBetween.IsPointMass && isBetween.Point)
                {
                    double ml, vl, mu, vu;
                    lowerBound.GetMeanAndVariance(out ml, out vl);
                    upperBound.GetMeanAndVariance(out mu, out vu);
                    double vlu = vl + vu;
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn((mu - ml) / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (mu - ml + vlu * alpha);
                    double betaU = alphaU * (alphaU - alpha);
                    result.SetMeanAndVariance(ml - vl * alphaU, vl - vl * vl * betaU);
                    result.SetToRatio(result, lowerBound);
                }
                else
                    throw new NotImplementedException();
            }
            else if (lowerBound.IsUniform())
            {
                if (isBetween.IsPointMass && !isBetween.Point)
                {
                    // X < lowerBound < upperBound
                    // lowerBound is not a point mass so lowerBound==X is impossible
                    return XAverageConditional(Bernoulli.PointMass(true), lowerBound, X, upperBound, logZ);
                }
                else
                {
                    result.SetToUniform();
                }
            }
            else
            {
                //double logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                if (Double.IsNegativeInfinity(logZ))
                    throw new AllZeroException();
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                if (lowerBound.IsPointMass)
                {
                  if (double.IsInfinity(lowerBound.Point) || X.IsPointMass)
                    return result;
                  if (upperBound.IsPointMass)
                  {
                    // r = -1 case
                    // X is not uniform or point mass
                    // f(L) = d_p (NormalCdf((U-mx)*sqrtPrec) - NormalCdf((L-mx)*sqrtPrec)) + const.
                    // dlogf = -d_p N(L;mx,vx)/f
                    // ddlogf = -dlogf^2 + dlogf*(mx-L)/vx
                    double L = lowerBound.Point;
                    double dlogf = -d_p * Math.Exp(X.GetLogProb(L) - logZ);
                    double ddlogf = dlogf * (-dlogf + (X.MeanTimesPrecision - L * X.Precision));
                    return Gaussian.FromDerivatives(L, dlogf, ddlogf, ForceProper);
                  }
                }
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                // if we get here, we know that -1 < r <= 0 and invSqrtVxl is finite
                // since lowerBound is not uniform and X is not uniform, invSqrtVxl > 0
                // yl is always finite.  yu may be +/-infinity, in which case r = 0.
                double logPhiL = Math.Log(invSqrtVxl) + Gaussian.GetLogProb(yl, 0, 1) + MMath.NormalCdfLn((yu - r * yl) / Math.Sqrt(1 - r * r));
                double alphaL = -d_p * Math.Exp(logPhiL - logZ);
                // (mx - ml) / (vl + vx) = yl*invSqrtVxl
                double betaL = alphaL * (alphaL - yl * invSqrtVxl);
                if (r > -1 && r != 0)
                {
                    double logPhiR = -2 * MMath.LnSqrt2PI - 0.5 * Math.Log(1 - r * r) - 0.5 * (yl * yl + yu * (yu - 2 * r * yl)) / (1 - r * r);
                    double c = d_p * r * Math.Exp(logPhiR - logZ);
                    betaL += c * invSqrtVxl * invSqrtVxl;
                }
                return GaussianProductOp_Laplace.GaussianFromAlphaBeta(lowerBound, alphaL, betaL, ForceProper);
            }
            if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                throw new ApplicationException("result is NaN");
            return result;
        }

        public static Gaussian UpperBoundAverageConditional_Slow(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return UpperBoundAverageConditional(isBetween, X, lowerBound, upperBound, LogZ(isBetween, X, lowerBound, upperBound));
        }

        /// <summary>EP message to <c>upperBound</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <param name="logZ">Buffer <c>logZ</c>.</param>
        /// <returns>The outgoing EP message to the <c>upperBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>upperBound</c> as the random arguments are varied. The formula is <c>proj[p(upperBound) sum_(isBetween,x,lowerBound) p(isBetween,x,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="isBetween" /> is not a proper distribution.</exception>
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian UpperBoundAverageConditional(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound,
            [Fresh] double logZ)
        {
            Gaussian result = new Gaussian();
            if (isBetween.IsUniform())
                return result;
            if (X.IsUniform())
            {
                if (lowerBound.IsUniform() || upperBound.IsUniform())
                {
                    result.SetToUniform();
                }
                else if (isBetween.IsPointMass && isBetween.Point)
                {
                    double ml, vl, mu, vu;
                    lowerBound.GetMeanAndVariance(out ml, out vl);
                    upperBound.GetMeanAndVariance(out mu, out vu);
                    double vlu = vl + vu;
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn((mu - ml) / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (mu - ml + vlu * alpha);
                    double betaU = alphaU * (alphaU - alpha);
                    result.SetMeanAndVariance(mu + vu * alphaU, vu - vu * vu * betaU);
                    result.SetToRatio(result, upperBound);
                }
                else
                    throw new NotImplementedException();
            }
            else if (upperBound.IsUniform())
            {
                if (isBetween.IsPointMass && !isBetween.Point)
                {
                    // lowerBound <= upperBound <= X
                    // upperBound is not a point mass so upperBound==X is impossible
                    return XAverageConditional(Bernoulli.PointMass(true), upperBound, lowerBound, X, logZ);
                }
                else
                {
                    result.SetToUniform();
                }
            }
            else
            {
                //double logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                if (Double.IsNegativeInfinity(logZ))
                    throw new AllZeroException();
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                if (upperBound.IsPointMass)
                {
                  if (double.IsPositiveInfinity(upperBound.Point) || X.IsPointMass)
                    return result;
                  if (lowerBound.IsPointMass)
                  {
                    // r = -1 case
                    // X is not uniform or point mass
                    // f(U) = d_p (NormalCdf((U-mx)*sqrtPrec) - NormalCdf((L-mx)*sqrtPrec)) + const.
                    // dlogf/dU = d_p N(U;mx,vx)/f
                    // ddlogf = -dlogf^2 + dlogf*(mx-U)/vx
                    double U = upperBound.Point;
                    double dlogf = d_p * Math.Exp(X.GetLogProb(U) - logZ);
                    double ddlogf = dlogf * (-dlogf + (X.MeanTimesPrecision - U * X.Precision));
                    return Gaussian.FromDerivatives(U, dlogf, ddlogf, ForceProper);
                  }
                }
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                // if we get here, -1 < r <= 0 and invSqrtVxu is finite
                // since upperBound is not uniform and X is not uniform, invSqrtVxu > 0
                // yu is always finite.  yl may be infinity, in which case r = 0.
                double logPhiU = Math.Log(invSqrtVxu) + Gaussian.GetLogProb(yu, 0, 1) + MMath.NormalCdfLn((yl - r * yu) / Math.Sqrt(1 - r * r));
                double alphaU = d_p * Math.Exp(logPhiU - logZ);
                // (mu - mx) / (vx + vu) = yu*invSqrtVxu
                double betaU = alphaU * (alphaU + yu * invSqrtVxu);
                if (r > -1 && r != 0)
                {
                    double logPhiR = -2 * MMath.LnSqrt2PI - 0.5 * Math.Log(1 - r * r) - 0.5 * (yu * yu + yl * (yl - 2 * r * yu)) / (1 - r * r);
                    double c = d_p * r * Math.Exp(logPhiR - logZ);
                    betaU += c * invSqrtVxu * invSqrtVxu;
                }
                return GaussianProductOp_Laplace.GaussianFromAlphaBeta(upperBound, alphaU, betaU, ForceProper);
            }
            if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                throw new ApplicationException("result is NaN");
            return result;
        }

        public static Gaussian XAverageConditional_Slow(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return XAverageConditional(isBetween, X, lowerBound, upperBound, LogZ(isBetween, X, lowerBound, upperBound));
        }

        /// <summary>EP message to <c>x</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <param name="logZ">Buffer <c>logZ</c>.</param>
        /// <returns>The outgoing EP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>x</c> as the random arguments are varied. The formula is <c>proj[p(x) sum_(isBetween,lowerBound,upperBound) p(isBetween,lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="isBetween" /> is not a proper distribution.</exception>
        [SkipIfAllUniform("lowerBound", "upperBound")]
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian XAverageConditional(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound,
            [Fresh] double logZ)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
                return XAverageConditional(isBetween, X, lowerBound.Point, upperBound.Point);
            Gaussian result = new Gaussian();
            if (isBetween.IsUniform())
                return result;
            if (X.IsPointMass)
            {
                throw new NotImplementedException();
            }
            else if (X.IsUniform())
            {
                if (lowerBound.IsUniform() || upperBound.IsUniform() ||
                    (lowerBound.IsPointMass && Double.IsInfinity(lowerBound.Point)) ||
                    (upperBound.IsPointMass && Double.IsInfinity(upperBound.Point)) ||
                    !Double.IsPositiveInfinity(isBetween.LogOdds))
                {
                    result.SetToUniform();
                }
                else if (isBetween.IsPointMass && isBetween.Point)
                {
                    double ml, vl, mu, vu;
                    lowerBound.GetMeanAndVariance(out ml, out vl);
                    upperBound.GetMeanAndVariance(out mu, out vu);
                    double vlu = vl + vu; // vlu > 0
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn((mu - ml) / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (mu - ml + vlu * alpha);
                    double betaU = alphaU * (alphaU - alpha);
                    double munew = mu + vu * alphaU;
                    double mlnew = ml - vl * alphaU;
                    double vunew = vu - vu * vu * betaU;
                    double vlnew = vl - vl * vl * betaU;
                    double diff = munew - mlnew;
                    result.SetMeanAndVariance((munew + mlnew) / 2, diff * diff / 12 + (vunew + vlnew + vu * vl * betaU) / 3);
                }
                else
                    throw new NotImplementedException();
            }
            else
            {
                // X is not a point mass or uniform
                //double logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                if (Double.IsNegativeInfinity(logZ))
                    throw new AllZeroException();
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                // r == -1 iff lowerBound and upperBound are point masses
                // since X is not a point mass, invSqrtVxl is finite, invSqrtVxu is finite
                double alphaL = 0.0;
                if (!lowerBound.IsUniform() && !Double.IsInfinity(yl))
                {
                    // since X and lowerBound are not both uniform, invSqrtVxl > 0
                    double logPhiL = Math.Log(invSqrtVxl) + Gaussian.GetLogProb(yl, 0, 1);
                    if (r > -1)
                        logPhiL += MMath.NormalCdfLn((yu - r * yl) / Math.Sqrt(1 - r * r));
                    alphaL = -d_p * Math.Exp(logPhiL - logZ);
                }
                double alphaU = 0.0;
                if (!upperBound.IsUniform() && !Double.IsInfinity(yu))
                {
                    // since X and upperBound are not both uniform, invSqrtVxu > 0
                    double logPhiU = Math.Log(invSqrtVxu) + Gaussian.GetLogProb(yu, 0, 1);
                    if (r > -1)
                        logPhiU += MMath.NormalCdfLn((yl - r * yu) / Math.Sqrt(1 - r * r));
                    alphaU = d_p * Math.Exp(logPhiU - logZ);
                }
                double alphaX = -alphaL - alphaU;
                // (mx - ml) / (vl + vx) = yl*invSqrtVxl
                double betaX = alphaX * alphaX;
                if (!Double.IsInfinity(yl))
                {
                    betaX -= alphaL * (yl * invSqrtVxl);
                }
                if (!Double.IsInfinity(yu))
                {
                    betaX += alphaU * (yu * invSqrtVxu);
                }
                if (r > -1 && r != 0 && !Double.IsInfinity(yl) && !Double.IsInfinity(yu))
                {
                    double logPhiR = -2 * MMath.LnSqrt2PI - 0.5 * Math.Log(1 - r * r) - 0.5 * (yl * yl + yu * yu - 2 * r * yl * yu) / (1 - r * r);
                    double c = d_p * r * Math.Exp(logPhiR - logZ);
                    betaX += c * (-2 * X.Precision + invSqrtVxl * invSqrtVxl + invSqrtVxu * invSqrtVxu);
                }
                return GaussianProductOp_Laplace.GaussianFromAlphaBeta(X, alphaX, betaX, ForceProper);
            }
            return result;
        }

#if false
    /// <summary>
    /// EP message to 'isBetween'
    /// </summary>
    /// <param name="X">Incoming message from 'x'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
    /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
    /// <param name="upperBound">Incoming message from 'upperBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
    /// <returns>The outgoing EP message to the 'isBetween' argument</returns>
    /// <remarks><para>
    /// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
    /// The formula is <c>proj[p(isBetween) sum_(x,upperBound) p(x,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
    /// </para></remarks>
    /// <exception cref="ImproperMessageException"><paramref name="X"/> is not a proper distribution</exception>
    /// <exception cref="ImproperMessageException"><paramref name="upperBound"/> is not a proper distribution</exception>
		public static Bernoulli IsBetweenAverageConditional([SkipIfUniform] Gaussian X, double lowerBound, [SkipIfUniform] Gaussian upperBound)
		{
			return IsBetweenAverageConditional(X, Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'isBetween'
		/// </summary>
		/// <param name="X">Incoming message from 'x'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'isBetween' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
		/// The formula is <c>proj[p(isBetween) sum_(x,lowerBound) p(x,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="X"/> is not a proper distribution</exception>
		/// <exception cref="ImproperMessageException"><paramref name="lowerBound"/> is not a proper distribution</exception>
		public static Bernoulli IsBetweenAverageConditional([SkipIfUniform] Gaussian X, [SkipIfUniform] Gaussian lowerBound, double upperBound)
		{
			return IsBetweenAverageConditional(X, lowerBound, Gaussian.PointMass(upperBound));
		}

		/// <summary>
		/// EP message to 'isBetween'
		/// </summary>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <returns>The outgoing EP message to the 'isBetween' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
		/// The formula is <c>proj[p(isBetween) sum_(lowerBound,upperBound) p(lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="lowerBound"/> is not a proper distribution</exception>
		/// <exception cref="ImproperMessageException"><paramref name="upperBound"/> is not a proper distribution</exception>
		public static Bernoulli IsBetweenAverageConditional(double X, [SkipIfUniform] Gaussian lowerBound, [SkipIfUniform] Gaussian upperBound)
		{
			return IsBetweenAverageConditional(Gaussian.PointMass(X), lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'isBetween'
		/// </summary>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Constant value for 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <returns>The outgoing EP message to the 'isBetween' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
		/// The formula is <c>proj[p(isBetween) sum_(upperBound) p(upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="upperBound"/> is not a proper distribution</exception>
		public static Bernoulli IsBetweenAverageConditional(double X, double lowerBound, [SkipIfUniform] Gaussian upperBound)
		{
			return IsBetweenAverageConditional(Gaussian.PointMass(X), Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'isBetween'
		/// </summary>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'isBetween' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
		/// The formula is <c>proj[p(isBetween) sum_(lowerBound) p(lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="lowerBound"/> is not a proper distribution</exception>
		public static Bernoulli IsBetweenAverageConditional(double X, [SkipIfUniform] Gaussian lowerBound, double upperBound)
		{
			return IsBetweenAverageConditional(Gaussian.PointMass(X), lowerBound, Gaussian.PointMass(upperBound));
		}

		/// <summary>
		/// EP message to 'x'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Constant value for 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'x' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
		/// The formula is <c>proj[p(x) sum_(isBetween,upperBound) p(isBetween,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
		[SkipIfAllUniform("X", "upperBound")]
    public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, double lowerBound, [RequiredArgument] Gaussian upperBound)
		{
			return XAverageConditional(isBetween, X, Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'x'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'x' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
		/// The formula is <c>proj[p(x) sum_(isBetween,lowerBound) p(isBetween,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
		[SkipIfAllUniform("X", "lowerBound")]
    public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, double upperBound)
		{
			return XAverageConditional(isBetween, X, lowerBound, Gaussian.PointMass(upperBound));
		}

		/// <summary>
		/// EP message to 'x'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'x' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
		/// The formula is <c>proj[p(x) sum_(lowerBound,upperBound) p(lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
		/// </para></remarks>
		[SkipIfAllUniform("lowerBound", "upperBound")]
		[SkipIfAllUniform("X", "lowerBound")]
		[SkipIfAllUniform("X", "upperBound")]
    public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
		{
			return XAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'x'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Constant value for 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'x' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
		/// The formula is <c>proj[p(x) sum_(upperBound) p(upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
		/// </para></remarks>
		[SkipIfAllUniform("X", "upperBound")]
    public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, double lowerBound, [RequiredArgument] Gaussian upperBound)
		{
			return XAverageConditional(Bernoulli.PointMass(isBetween), X, Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'x'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'x' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
		/// The formula is <c>proj[p(x) sum_(lowerBound) p(lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
		/// </para></remarks>
		[SkipIfAllUniform("X", "lowerBound")]
    public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, double upperBound)
		{
			return XAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, Gaussian.PointMass(upperBound));
		}

		/// <summary>
		/// EP message to 'upperBound'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'upperBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
		/// The formula is <c>proj[p(upperBound) sum_(isBetween,lowerBound) p(isBetween,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
    public static Gaussian UpperBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
		{
			return UpperBoundAverageConditional(isBetween, Gaussian.PointMass(X), lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'upperBound'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Constant value for 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'upperBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
		/// The formula is <c>proj[p(upperBound) sum_(isBetween,x) p(isBetween,x) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
		[SkipIfAllUniform("X", "upperBound")]
    public static Gaussian UpperBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, double lowerBound, [RequiredArgument] Gaussian upperBound)
		{
			return UpperBoundAverageConditional(isBetween, X, Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'upperBound'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Constant value for 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'upperBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
		/// The formula is <c>proj[p(upperBound) sum_(isBetween) p(isBetween) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
    public static Gaussian UpperBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, double lowerBound, [RequiredArgument] Gaussian upperBound)
		{
			return UpperBoundAverageConditional(isBetween, Gaussian.PointMass(X), Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'upperBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'upperBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
		/// The formula is <c>proj[p(upperBound) sum_(x,lowerBound) p(x,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
		/// </para></remarks>
		[SkipIfAllUniform("X", "lowerBound")]
		[SkipIfAllUniform("X", "upperBound")]
		public static Gaussian UpperBoundAverageConditional(bool isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
		{
			return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'upperBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'upperBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
		/// The formula is <c>proj[p(upperBound) sum_(lowerBound) p(lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
		/// </para></remarks>
		public static Gaussian UpperBoundAverageConditional(bool isBetween, double X, Gaussian lowerBound, Gaussian upperBound)
		{
			return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'upperBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Constant value for 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'upperBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
		/// The formula is <c>proj[p(upperBound) sum_(x) p(x) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
		/// </para></remarks>
		[SkipIfAllUniform("X", "upperBound")]
		public static Gaussian UpperBoundAverageConditional(bool isBetween, Gaussian X, double lowerBound, Gaussian upperBound)
		{
			return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), X, Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'upperBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Constant value for 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'upperBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is the factor viewed as a function of 'upperBound' conditioned on the given values.
		/// </para></remarks>
		public static Gaussian UpperBoundAverageConditional(bool isBetween, double X, double lowerBound, Gaussian upperBound)
		{
			return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), Gaussian.PointMass(lowerBound), upperBound);
		}

		/// <summary>
		/// EP message to 'lowerBound'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
		/// The formula is <c>proj[p(lowerBound) sum_(isBetween,upperBound) p(isBetween,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
		public static Gaussian LowerBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, Gaussian lowerBound, Gaussian upperBound)
		{
			return LowerBoundAverageConditional(isBetween, Gaussian.PointMass(X), lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'lowerBound'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
		/// The formula is <c>proj[p(lowerBound) sum_(isBetween,x) p(isBetween,x) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
		[SkipIfAllUniform("X", "lowerBound")]
		public static Gaussian LowerBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, Gaussian X, Gaussian lowerBound, double upperBound)
		{
			return LowerBoundAverageConditional(isBetween, X, lowerBound, Gaussian.PointMass(upperBound));
		}

		/// <summary>
		/// EP message to 'lowerBound'
		/// </summary>
		/// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
		/// The formula is <c>proj[p(lowerBound) sum_(isBetween) p(isBetween) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
		/// </para></remarks>
		/// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
		public static Gaussian LowerBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, Gaussian lowerBound, double upperBound)
		{
			return LowerBoundAverageConditional(isBetween, Gaussian.PointMass(X), lowerBound, Gaussian.PointMass(upperBound));
		}

		/// <summary>
		/// EP message to 'lowerBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
		/// The formula is <c>proj[p(lowerBound) sum_(x,upperBound) p(x,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
		/// </para></remarks>
		[SkipIfAllUniform("X", "lowerBound")]
		[SkipIfAllUniform("X", "upperBound")]
		public static Gaussian LowerBoundAverageConditional(bool isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
		{
			return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'lowerBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Incoming message from 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
		/// The formula is <c>proj[p(lowerBound) sum_(upperBound) p(upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
		/// </para></remarks>
		public static Gaussian LowerBoundAverageConditional(bool isBetween, double X, Gaussian lowerBound, Gaussian upperBound)
		{
			return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), lowerBound, upperBound);
		}

		/// <summary>
		/// EP message to 'lowerBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Incoming message from 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
		/// The formula is <c>proj[p(lowerBound) sum_(x) p(x) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
		/// </para></remarks>
		[SkipIfAllUniform("X", "lowerBound")]
		public static Gaussian LowerBoundAverageConditional(bool isBetween, Gaussian X, Gaussian lowerBound, double upperBound)
		{
			return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, Gaussian.PointMass(upperBound));
		}

		/// <summary>
		/// EP message to 'lowerBound'
		/// </summary>
		/// <param name="isBetween">Constant value for 'isBetween'.</param>
		/// <param name="X">Constant value for 'x'.</param>
		/// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
		/// <param name="upperBound">Constant value for 'upperBound'.</param>
		/// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
		/// <remarks><para>
		/// The outgoing message is the factor viewed as a function of 'lowerBound' conditioned on the given values.
		/// </para></remarks>
		public static Gaussian LowerBoundAverageConditional(bool isBetween, double X, Gaussian lowerBound, double upperBound)
		{
			return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), lowerBound, Gaussian.PointMass(upperBound));
		}
#endif

        // ------------------AverageLogarithm -------------------------------------
        private const string NotSupportedMessage =
            "Variational Message Passing does not support the IsBetween factor with Gaussian distributions, since the factor is not conjugate to the Gaussian.";


        /// <summary>VMP message to <c>isBetween</c>.</summary>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>The outgoing VMP message to the <c>isBetween</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>isBetween</c> as the random arguments are varied. The formula is <c>proj[sum_(x,lowerBound,upperBound) p(x,lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound)]</c>.</para>
        /// </remarks>
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Bernoulli IsBetweenAverageLogarithm(Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>x</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>The outgoing VMP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>x</c>. Because the factor is deterministic, <c>isBetween</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(lowerBound,upperBound) p(lowerBound,upperBound) log(sum_isBetween p(isBetween) factor(isBetween,x,lowerBound,upperBound)))</c>.</para>
        /// </remarks>
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Gaussian XAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>lowerBound</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>The outgoing VMP message to the <c>lowerBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>lowerBound</c>. Because the factor is deterministic, <c>isBetween</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(x,upperBound) p(x,upperBound) log(sum_isBetween p(isBetween) factor(isBetween,x,lowerBound,upperBound)))</c>.</para>
        /// </remarks>
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Gaussian LowerBoundAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>upperBound</c>.</summary>
        /// <param name="isBetween">Incoming message from <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Incoming message from <c>lowerBound</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>The outgoing VMP message to the <c>upperBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>upperBound</c>. Because the factor is deterministic, <c>isBetween</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(x,lowerBound) p(x,lowerBound) log(sum_isBetween p(isBetween) factor(isBetween,x,lowerBound,upperBound)))</c>.</para>
        /// </remarks>
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Gaussian UpperBoundAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
        }

        private const string RandomBoundsNotSupportedMessage = "VMP does not support truncation with stochastic bounds.";

        /// <summary>VMP message to <c>x</c>.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <param name="to_X">Previous outgoing message to <c>X</c>.</param>
        /// <returns>The outgoing VMP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>x</c> conditioned on the given values.</para>
        /// </remarks>
        [Quality(QualityBand.Preview)]
        public static Gaussian XAverageLogarithm(bool isBetween, [Stochastic] Gaussian X, double lowerBound, double upperBound, Gaussian to_X)
        {
            if (!isBetween)
                throw new ArgumentException("TruncatedGaussian requires isBetween=true", "isBetween");
            var prior = X / to_X;
            var tg = new TruncatedGaussian(prior);
            tg.LowerBound = lowerBound;
            tg.UpperBound = upperBound;
            return tg.ToGaussian() / prior;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <param name="to_X">Previous outgoing message to <c>X</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Quality(QualityBand.Preview)]
        public static double AverageLogFactor(bool isBetween, [Stochastic] Gaussian X, double lowerBound, double upperBound, Gaussian to_X)
        {
            if (!isBetween)
                throw new ArgumentException("TruncatedGaussian requires isBetween=true", "isBetween");
            var prior = X / to_X;
            var tg = new TruncatedGaussian(prior);
            tg.LowerBound = lowerBound;
            tg.UpperBound = upperBound;
            return X.GetAverageLog(X) - tg.GetAverageLog(tg);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor(TruncatedGaussian X)
        {
            return 0;
        }

        /// <summary>VMP message to <c>x</c>.</summary>
        /// <param name="isBetween">Constant value for <c>isBetween</c>.</param>
        /// <param name="X">Incoming message from <c>x</c>.</param>
        /// <param name="lowerBound">Constant value for <c>lowerBound</c>.</param>
        /// <param name="upperBound">Constant value for <c>upperBound</c>.</param>
        /// <returns>The outgoing VMP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>x</c> conditioned on the given values.</para>
        /// </remarks>
        [Quality(QualityBand.Preview)]
        public static TruncatedGaussian XAverageLogarithm(bool isBetween, [IgnoreDependency] TruncatedGaussian X, double lowerBound, double upperBound)
        {
            if (!isBetween)
                throw new ArgumentException("TruncatedGaussian requires isBetween=true", "isBetween");
            var tg = TruncatedGaussian.Uniform();
            tg.LowerBound = lowerBound;
            tg.UpperBound = upperBound;
            return tg;
        }

        /// <summary>VMP message to <c>lowerBound</c>.</summary>
        /// <returns>The outgoing VMP message to the <c>lowerBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>lowerBound</c> conditioned on the given values.</para>
        /// </remarks>
        /// <remarks><para>
        /// Variational Message Passing does not support ConstrainBetween with Gaussian distributions, since the factor is not conjugate to the Gaussian.
        /// This method will throw an exception.
        /// </para></remarks>
        [NotSupported(RandomBoundsNotSupportedMessage)]
        public static Gaussian LowerBoundAverageLogarithm()
        {
            throw new NotSupportedException(RandomBoundsNotSupportedMessage);
        }

        /// <summary>VMP message to <c>upperBound</c>.</summary>
        /// <returns>The outgoing VMP message to the <c>upperBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>upperBound</c> conditioned on the given values.</para>
        /// </remarks>
        /// <remarks><para>
        /// Variational Message Passing does not support ConstrainBetween with Gaussian distributions, since the factor is not conjugate to the Gaussian.
        /// This method will throw an exception.
        /// </para></remarks>
        [NotSupported(RandomBoundsNotSupportedMessage)]
        public static Gaussian UpperBoundAverageLogarithm()
        {
            throw new NotSupportedException(RandomBoundsNotSupportedMessage);
        }
    }
}
