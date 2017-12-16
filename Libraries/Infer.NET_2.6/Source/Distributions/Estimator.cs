// (C) Copyright 2008 Microsoft Research Cambridge

using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Utils;

namespace MicrosoftResearch.Infer.Distributions
{
    /// <summary>
    /// Indicates support for adding an item to a distribution estimator
    /// </summary>
    /// <typeparam name="T">Type of item to add - could be a distribution type or a sample type</typeparam>
    public interface Accumulator<T>
    {
        /// <summary>
        /// Adds an item to the estimator
        /// </summary>
        /// <param name="item">The item</param>
        void Add(T item);

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        void Clear();
    }

    /// <summary>
    /// Indicates support for retrieving an estimated distribution
    /// </summary>
    /// <typeparam name="T">Distribution type</typeparam>
    public interface Estimator<T>
    {
        /// <summary>
        /// Retrieve the estimated distribution
        /// </summary>
        /// <param name="result">Where to put the result - ignored if the distribution type is a value type</param>
        /// <returns>The resulting estimated distribution</returns>
        T GetDistribution(T result);
    }

    /// <exclude/>
    public delegate object CreateEstimatorMethod(object dist);

    /// <summary>
    /// Estimator factor. Given a distribution instance, create a compatible estimator instance
    /// </summary>
    public sealed class EstimatorFactory
    {
        private static readonly EstimatorFactory instance = new EstimatorFactory();
        private Dictionary<Type, CreateEstimatorMethod> creators;
        private Dictionary<Type, Type> estimatorTypes;

        private EstimatorFactory()
        {
            creators = new Dictionary<Type, CreateEstimatorMethod>();
            estimatorTypes = new Dictionary<Type, Type>();

            // Add in the known creators
            RegisterEstimator(typeof (Bernoulli), typeof (BernoulliEstimator), delegate(object dst) { return new BernoulliEstimator(); });
            RegisterEstimator(typeof (Beta), typeof (BetaEstimator), delegate(object dst) { return new BetaEstimator(); });
            RegisterEstimator(typeof (Dirichlet), typeof (DirichletEstimator), delegate(object dst) { return new DirichletEstimator(((Dirichlet) dst).Dimension); });
            RegisterEstimator(typeof (Discrete), typeof (DiscreteEstimator), delegate(object dst) { return new DiscreteEstimator(((Discrete) dst).Dimension); });
            RegisterEstimator(typeof (Gamma), typeof (GammaEstimator), delegate(object dst) { return new GammaEstimator(); });
            RegisterEstimator(typeof (Gaussian), typeof (GaussianEstimator), delegate(object dst) { return new GaussianEstimator(); });
            RegisterEstimator(typeof (Poisson), typeof (PoissonEstimator), delegate(object dst) { return new PoissonEstimator(); });
            RegisterEstimator(typeof (VectorGaussian), typeof (VectorGaussianEstimator),
                              delegate(object dst) { return new VectorGaussianEstimator(((VectorGaussian) dst).Dimension); });
            RegisterEstimator(typeof (Wishart), typeof (WishartEstimator), delegate(object dst) { return new WishartEstimator(((Wishart) dst).Dimension); });
            RegisterEstimator(typeof (TruncatedGaussian), typeof (TruncatedGaussianEstimator), delegate(object dst) { return new TruncatedGaussianEstimator(); });
        }

        /// <summary>
        /// Registers an estimator. The factory is primed with stock
        /// stock estimators. This function allows clients to add in custom
        /// estimators
        /// </summary>
        /// <param name="distType">Distribution type</param>
        /// <param name="estType">Estimator type</param>
        /// <param name="c">Method for creating the estimator</param>
        public void RegisterEstimator(Type distType, Type estType, CreateEstimatorMethod c)
        {
            if (creators.ContainsKey(distType))
            {
                creators[distType] = c;
                estimatorTypes[distType] = estType;
            }
            else
            {
                creators.Add(distType, c);
                estimatorTypes.Add(distType, estType);
            }
        }

        /// <summary>
        /// Creates an estimator instance from a distribution prototype
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="TDomain"></typeparam>
        /// <param name="distProto">Distribution prototype</param>
        /// <returns>Estimator instance</returns>
        public Estimator<T> CreateEstimator<T, TDomain>(T distProto)
            where T : IDistribution<TDomain>
        {
            if (creators.ContainsKey(typeof (T)))
                return (Estimator<T>) creators[typeof (T)].Invoke(distProto);
            else
                throw new ArgumentException(StringUtil.TypeToString(typeof (T)) + " has no registered estimators");
        }

        /// <summary>
        /// Creates an estimator instance from a distribution prototype
        /// </summary>
        /// <param name="distType">Distribution type</param>
        /// <returns>Estimator instance</returns>
        public Type EstimatorType(Type distType)
        {
            if (estimatorTypes.ContainsKey(distType))
                return estimatorTypes[distType];
            else
                return null;
        }

        /// <summary>
        /// Estimator factory singleton instance
        /// </summary>
        public static EstimatorFactory Instance
        {
            get { return instance; }
        }
    }
}
