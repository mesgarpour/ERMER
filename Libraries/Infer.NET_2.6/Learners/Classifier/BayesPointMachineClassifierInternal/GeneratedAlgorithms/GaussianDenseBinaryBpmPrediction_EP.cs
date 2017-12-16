// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Factors;

namespace MicrosoftResearch.Infer.Models.User
{
	/// <summary>
	/// Generated algorithm for performing inference.
	/// </summary>
	/// <remarks>
	/// If you wish to use this class directly, you must perform the following steps:
	/// 1) Create an instance of the class.
	/// 2) Set the value of any externally-set fields e.g. data, priors.
	/// 3) Call the Execute(numberOfIterations) method.
	/// 4) Use the XXXMarginal() methods to retrieve posterior marginals for different variables.
	/// 
	/// Generated by Infer.NET 2.6.41114.1 at 11:57 PM on Friday, November 14, 2014.
	/// </remarks>
	public partial class GaussianDenseBinaryBpmPrediction_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		/// <summary>Field backing the InstanceCount property</summary>
		private int instanceCount;
		/// <summary>Field backing the FeatureCount property</summary>
		private int featureCount;
		/// <summary>Field backing the FeatureValues property</summary>
		private double[][] featureValues;
		/// <summary>Field backing the WeightPriors property</summary>
		private DistributionStructArray<Gaussian,double> weightPriors;
		/// <summary>Field backing the WeightConstraints property</summary>
		private DistributionStructArray<Gaussian,double> weightConstraints;
		/// <summary>The number of iterations last computed by Constant. Set this to zero to force re-execution of Constant</summary>
		public int Constant_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_WeightPriors. Set this to zero to force re-execution of Changed_WeightPriors</summary>
		public int Changed_WeightPriors_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_WeightConstraints_WeightPriors. Set this to zero to force re-execution of Changed_WeightConstraints_WeightPriors</summary>
		public int Changed_WeightConstraints_WeightPriors_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_FeatureCount. Set this to zero to force re-execution of Changed_FeatureCount</summary>
		public int Changed_FeatureCount_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_FeatureCount_WeightPriors. Set this to zero to force re-execution of Changed_FeatureCount_WeightPriors</summary>
		public int Changed_FeatureCount_WeightPriors_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints. Set this to zero to force re-execution of Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints</summary>
		public int Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_iterationsDone;
		/// <summary>True if Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints has performed initialisation. Set this to false to force re-execution of Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints</summary>
		public bool Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_isInitialised;
		/// <summary>The number of iterations last computed by Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount. Set this to zero to force re-execution of Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount</summary>
		public int Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone;
		/// <summary>True if Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount has performed initialisation. Set this to false to force re-execution of Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount</summary>
		public bool Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_isInitialised;
		/// <summary>The number of iterations last computed by Changed_FeatureCount_InstanceCount. Set this to zero to force re-execution of Changed_FeatureCount_InstanceCount</summary>
		public int Changed_FeatureCount_InstanceCount_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints. Set this to zero to force re-execution of Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints</summary>
		public int Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_InstanceCount. Set this to zero to force re-execution of Changed_InstanceCount</summary>
		public int Changed_InstanceCount_iterationsDone;
		/// <summary>The number of iterations last computed by Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10. Set this to zero to force re-execution of Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10</summary>
		public int Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone;
		/// <summary>Buffer for ReplicateOp_Divide.Marginal<Gaussian></summary>
		public DistributionStructArray<Gaussian,double> Weights_depth1_rep_B_toDef;
		/// <summary>Buffer for ReplicateOp_Divide.UsesAverageConditional<Gaussian></summary>
		public DistributionStructArray<Gaussian,double> Weights_depth1_rep_F_marginal;
		public DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]> Weights_depth1_rep_F;
		public DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]> Weights_depth1_rep_B;
		/// <summary>Messages to use of 'Weights'</summary>
		public DistributionStructArray<Gaussian,double>[] Weights_uses_F;
		/// <summary>Messages from use of 'Weights'</summary>
		public DistributionStructArray<Gaussian,double>[] Weights_uses_B;
		/// <summary>Buffer for ReplicateOp_Divide.UsesAverageConditional<DistributionStructArray<Gaussian, double>></summary>
		public DistributionStructArray<Gaussian,double> Weights_uses_F_marginal;
		/// <summary>Buffer for ReplicateOp_Divide.Marginal<DistributionStructArray<Gaussian, double>></summary>
		public DistributionStructArray<Gaussian,double> Weights_uses_B_toDef;
		public DistributionStructArray<Bernoulli,bool> Labels_F;
		/// <summary>Message to marginal of 'Labels'</summary>
		public DistributionStructArray<Bernoulli,bool> Labels_marginal_F;
		/// <summary>Message from use of 'Labels'</summary>
		public DistributionStructArray<Bernoulli,bool> Labels_use_B;
		public DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]> FeatureScores_F;
		public DistributionStructArray<Gaussian,double> Score_F;
		public DistributionStructArray<Gaussian,double> NoisyScore_F;
		#endregion

		#region Properties
		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		/// <summary>The externally-specified value of 'InstanceCount'</summary>
		public int InstanceCount
		{
			get {
				return this.instanceCount;
			}
			set {
				if (this.instanceCount!=value) {
					this.instanceCount = value;
					this.numberOfIterationsDone = 0;
					this.Changed_InstanceCount_iterationsDone = 0;
					this.Changed_FeatureCount_InstanceCount_iterationsDone = 0;
					this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_isInitialised = false;
					this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_isInitialised = false;
					this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
					this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
				}
			}
		}

		/// <summary>The externally-specified value of 'FeatureCount'</summary>
		public int FeatureCount
		{
			get {
				return this.featureCount;
			}
			set {
				if (this.featureCount!=value) {
					this.featureCount = value;
					this.numberOfIterationsDone = 0;
					this.Changed_FeatureCount_iterationsDone = 0;
					this.Changed_FeatureCount_InstanceCount_iterationsDone = 0;
					this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_iterationsDone = 0;
					this.Changed_FeatureCount_WeightPriors_iterationsDone = 0;
					this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 0;
					this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
					this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
				}
			}
		}

		/// <summary>The externally-specified value of 'FeatureValues'</summary>
		public double[][] FeatureValues
		{
			get {
				return this.featureValues;
			}
			set {
				if ((value!=null)&&(value.Length!=this.instanceCount)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+this.instanceCount)+" was expected for variable \'FeatureValues\'");
				}
				this.featureValues = value;
				this.numberOfIterationsDone = 0;
				this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
			}
		}

		/// <summary>The externally-specified value of 'WeightPriors'</summary>
		public DistributionStructArray<Gaussian,double> WeightPriors
		{
			get {
				return this.weightPriors;
			}
			set {
				this.weightPriors = value;
				this.numberOfIterationsDone = 0;
				this.Changed_WeightPriors_iterationsDone = 0;
				this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_iterationsDone = 0;
				this.Changed_FeatureCount_WeightPriors_iterationsDone = 0;
				this.Changed_WeightConstraints_WeightPriors_iterationsDone = 0;
				this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 0;
				this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
				this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
			}
		}

		/// <summary>The externally-specified value of 'WeightConstraints'</summary>
		public DistributionStructArray<Gaussian,double> WeightConstraints
		{
			get {
				return this.weightConstraints;
			}
			set {
				this.weightConstraints = value;
				this.numberOfIterationsDone = 0;
				this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_isInitialised = false;
				this.Changed_WeightConstraints_WeightPriors_iterationsDone = 0;
				this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 0;
				this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
				this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
			}
		}

		#endregion

		#region Methods
		/// <summary>Get the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		public object GetObservedValue(string variableName)
		{
			if (variableName=="InstanceCount") {
				return this.InstanceCount;
			}
			if (variableName=="FeatureCount") {
				return this.FeatureCount;
			}
			if (variableName=="FeatureValues") {
				return this.FeatureValues;
			}
			if (variableName=="WeightPriors") {
				return this.WeightPriors;
			}
			if (variableName=="WeightConstraints") {
				return this.WeightConstraints;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Set the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		/// <param name="value">Observed value</param>
		public void SetObservedValue(string variableName, object value)
		{
			if (variableName=="InstanceCount") {
				this.InstanceCount = (int)value;
				return ;
			}
			if (variableName=="FeatureCount") {
				this.FeatureCount = (int)value;
				return ;
			}
			if (variableName=="FeatureValues") {
				this.FeatureValues = (double[][])value;
				return ;
			}
			if (variableName=="WeightPriors") {
				this.WeightPriors = (DistributionStructArray<Gaussian,double>)value;
				return ;
			}
			if (variableName=="WeightConstraints") {
				this.WeightConstraints = (DistributionStructArray<Gaussian,double>)value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="Labels") {
				return this.LabelsMarginal();
			}
			throw new ArgumentException("This class was not built to infer "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName));
		}

		/// <summary>Get the query-specific marginal distribution of a variable.</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName, string query)
		{
			if (query=="Marginal") {
				return this.Marginal(variableName);
			}
			throw new ArgumentException(((("This class was not built to infer \'"+variableName)+"\' with query \'")+query)+"\'");
		}

		/// <summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName, string query)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName, query));
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			if (numberOfIterations<this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone) {
				this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_isInitialised = false;
				this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_isInitialised = false;
				this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
				this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
			}
			this.Changed_FeatureCount();
			this.Constant();
			this.Changed_InstanceCount();
			this.Changed_FeatureCount_InstanceCount();
			this.Changed_WeightPriors();
			this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints(initialise);
			this.Changed_FeatureCount_WeightPriors();
			this.Changed_WeightConstraints_WeightPriors();
			this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount(initialise);
			this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints(numberOfIterations);
			this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10();
			this.numberOfIterationsDone = numberOfIterations;
		}

		/// <summary>Update all marginals, by iterating message-passing the given number of times</summary>
		/// <param name="numberOfIterations">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>
		public void Execute(int numberOfIterations)
		{
			this.Execute(numberOfIterations, true);
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(this.numberOfIterationsDone+additionalIterations, false);
		}

		private void OnProgressChanged(ProgressChangedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<ProgressChangedEventArgs> handler = this.ProgressChanged;
			if (handler!=null) {
				handler(this, e);
			}
		}

		/// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
		public void Reset()
		{
			this.Execute(0);
		}

		/// <summary>Computations that depend on the observed value of FeatureCount</summary>
		private void Changed_FeatureCount()
		{
			if (this.Changed_FeatureCount_iterationsDone==1) {
				return ;
			}
			// Create array for replicates of 'Weights_depth1_rep_B_toDef'
			this.Weights_depth1_rep_B_toDef = new DistributionStructArray<Gaussian,double>(this.featureCount);
			// Create array for replicates of 'Weights_depth1_rep_F_marginal'
			this.Weights_depth1_rep_F_marginal = new DistributionStructArray<Gaussian,double>(this.featureCount);
			// Create array for replicates of 'Weights_depth1_rep_F'
			this.Weights_depth1_rep_F = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(this.featureCount);
			// Create array for replicates of 'Weights_depth1_rep_B'
			this.Weights_depth1_rep_B = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(this.featureCount);
			this.Changed_FeatureCount_iterationsDone = 1;
			this.Changed_FeatureCount_WeightPriors_iterationsDone = 0;
			this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_iterationsDone = 0;
			this.Changed_FeatureCount_InstanceCount_iterationsDone = 0;
		}

		/// <summary>Computations that do not depend on observed values</summary>
		private void Constant()
		{
			if (this.Constant_iterationsDone==1) {
				return ;
			}
			// Create array for 'Weights_uses' Forwards messages.
			this.Weights_uses_F = new DistributionStructArray<Gaussian,double>[2];
			// Create array for 'Weights_uses' Backwards messages.
			this.Weights_uses_B = new DistributionStructArray<Gaussian,double>[2];
			this.Constant_iterationsDone = 1;
			this.Changed_WeightPriors_iterationsDone = 0;
			this.Changed_InstanceCount_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of InstanceCount</summary>
		private void Changed_InstanceCount()
		{
			if (this.Changed_InstanceCount_iterationsDone==1) {
				return ;
			}
			// Create array for 'Labels' Forwards messages.
			this.Labels_F = new DistributionStructArray<Bernoulli,bool>(this.instanceCount);
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				this.Labels_F[InstanceRange] = Bernoulli.Uniform();
			}
			// Create array for replicates of 'FeatureScores_F'
			this.FeatureScores_F = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(this.instanceCount);
			// Create array for replicates of 'Score_F'
			this.Score_F = new DistributionStructArray<Gaussian,double>(this.instanceCount);
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				this.Score_F[InstanceRange] = Gaussian.Uniform();
			}
			// Create array for replicates of 'NoisyScore_F'
			this.NoisyScore_F = new DistributionStructArray<Gaussian,double>(this.instanceCount);
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				this.NoisyScore_F[InstanceRange] = Gaussian.Uniform();
			}
			// Create array for 'Labels_marginal' Forwards messages.
			this.Labels_marginal_F = new DistributionStructArray<Bernoulli,bool>(this.instanceCount);
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				this.Labels_marginal_F[InstanceRange] = Bernoulli.Uniform();
			}
			// Create array for 'Labels_use' Backwards messages.
			this.Labels_use_B = new DistributionStructArray<Bernoulli,bool>(this.instanceCount);
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				this.Labels_use_B[InstanceRange] = Bernoulli.Uniform();
			}
			this.Changed_InstanceCount_iterationsDone = 1;
			this.Changed_FeatureCount_InstanceCount_iterationsDone = 0;
			this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and InstanceCount</summary>
		private void Changed_FeatureCount_InstanceCount()
		{
			if (this.Changed_FeatureCount_InstanceCount_iterationsDone==1) {
				return ;
			}
			for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
				// Create array for 'Weights_depth1_rep' Forwards messages.
				this.Weights_depth1_rep_F[FeatureRange] = new DistributionStructArray<Gaussian,double>(this.instanceCount);
				// Create array for 'Weights_depth1_rep' Backwards messages.
				this.Weights_depth1_rep_B[FeatureRange] = new DistributionStructArray<Gaussian,double>(this.instanceCount);
			}
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
					this.Weights_depth1_rep_B[FeatureRange][InstanceRange] = Gaussian.Uniform();
					this.Weights_depth1_rep_F[FeatureRange][InstanceRange] = Gaussian.Uniform();
				}
				// Create array for 'FeatureScores' Forwards messages.
				this.FeatureScores_F[InstanceRange] = new DistributionStructArray<Gaussian,double>(this.featureCount);
				for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
					this.FeatureScores_F[InstanceRange][FeatureRange] = Gaussian.Uniform();
				}
			}
			this.Changed_FeatureCount_InstanceCount_iterationsDone = 1;
			this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
			this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of WeightPriors</summary>
		private void Changed_WeightPriors()
		{
			if (this.Changed_WeightPriors_iterationsDone==1) {
				return ;
			}
			for(int _ind = 0; _ind<2; _ind++) {
				this.Weights_uses_B[_ind] = ArrayHelper.MakeUniform<DistributionStructArray<Gaussian,double>>(this.weightPriors);
				this.Weights_uses_F[_ind] = ArrayHelper.MakeUniform<DistributionStructArray<Gaussian,double>>(this.weightPriors);
			}
			// Message to 'Weights_uses' from Replicate factor
			this.Weights_uses_F_marginal = ReplicateOp_Divide.MarginalInit<DistributionStructArray<Gaussian,double>>(this.weightPriors);
			// Message to 'Weights_uses' from Replicate factor
			this.Weights_uses_B_toDef = ReplicateOp_Divide.ToDefInit<DistributionStructArray<Gaussian,double>>(this.weightPriors);
			this.Changed_WeightPriors_iterationsDone = 1;
			this.Changed_WeightConstraints_WeightPriors_iterationsDone = 0;
			this.Changed_FeatureCount_WeightPriors_iterationsDone = 0;
			this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_iterationsDone = 0;
			this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and WeightPriors and must reset on changes to numberOfIterationsDecreased and InstanceCount and WeightConstraints</summary>
		/// <param name="initialise">If true, reset messages that initialise loops</param>
		private void Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints(bool initialise)
		{
			if ((this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_iterationsDone==1)&&((!initialise)||this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_isInitialised)) {
				return ;
			}
			for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
				// Message to 'Weights_depth1_rep' from Replicate factor
				this.Weights_depth1_rep_F_marginal[FeatureRange] = ReplicateOp_Divide.MarginalInit<Gaussian>(this.Weights_uses_F[1][FeatureRange]);
			}
			this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_iterationsDone = 1;
			this.Changed_FeatureCount_WeightPriors_Init_numberOfIterationsDecreased_InstanceCount_WeightConstraints_isInitialised = true;
			this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 0;
			this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and WeightPriors</summary>
		private void Changed_FeatureCount_WeightPriors()
		{
			if (this.Changed_FeatureCount_WeightPriors_iterationsDone==1) {
				return ;
			}
			for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
				// Message to 'Weights_depth1_rep' from Replicate factor
				this.Weights_depth1_rep_B_toDef[FeatureRange] = ReplicateOp_Divide.ToDefInit<Gaussian>(this.Weights_uses_F[1][FeatureRange]);
			}
			this.Changed_FeatureCount_WeightPriors_iterationsDone = 1;
			this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of WeightConstraints and WeightPriors</summary>
		private void Changed_WeightConstraints_WeightPriors()
		{
			if (this.Changed_WeightConstraints_WeightPriors_iterationsDone==1) {
				return ;
			}
			// Message to 'Weights_uses' from EqualRandom factor
			this.Weights_uses_B[0] = ArrayHelper.SetTo<DistributionStructArray<Gaussian,double>>(this.Weights_uses_B[0], this.weightConstraints);
			// Message to 'Weights_uses' from Replicate factor
			this.Weights_uses_B_toDef = ReplicateOp_Divide.ToDef<DistributionStructArray<Gaussian,double>>(this.Weights_uses_B, this.Weights_uses_B_toDef);
			// Message to 'Weights_uses' from Replicate factor
			this.Weights_uses_F_marginal = ReplicateOp_Divide.Marginal<DistributionStructArray<Gaussian,double>>(this.Weights_uses_B_toDef, this.weightPriors, this.Weights_uses_F_marginal);
			this.Changed_WeightConstraints_WeightPriors_iterationsDone = 1;
			this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and WeightPriors and WeightConstraints and must reset on changes to numberOfIterationsDecreased and InstanceCount</summary>
		/// <param name="initialise">If true, reset messages that initialise loops</param>
		private void Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount(bool initialise)
		{
			if ((this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone==1)&&((!initialise)||this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_isInitialised)) {
				return ;
			}
			// Message to 'Weights_uses' from Replicate factor
			this.Weights_uses_F[1] = ReplicateOp_Divide.UsesAverageConditional<DistributionStructArray<Gaussian,double>>(this.Weights_depth1_rep_B_toDef, this.Weights_uses_F_marginal, 1, this.Weights_uses_F[1]);
			for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
				// Message to 'Weights_depth1_rep' from Replicate factor
				this.Weights_depth1_rep_F_marginal[FeatureRange] = ReplicateOp_Divide.Marginal<Gaussian>(this.Weights_depth1_rep_B_toDef[FeatureRange], this.Weights_uses_F[1][FeatureRange], this.Weights_depth1_rep_F_marginal[FeatureRange]);
			}
			this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_iterationsDone = 1;
			this.Changed_FeatureCount_WeightPriors_WeightConstraints_Init_numberOfIterationsDecreased_InstanceCount_isInitialised = true;
			this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of numberOfIterationsDecreased and FeatureCount and InstanceCount and WeightPriors and WeightConstraints</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		private void Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints(int numberOfIterations)
		{
			if (this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone==numberOfIterations) {
				return ;
			}
			for(int iteration = this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone; iteration<numberOfIterations; iteration++) {
				for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
					for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
						// Message to 'Weights_depth1_rep' from Replicate factor
						this.Weights_depth1_rep_F[FeatureRange][InstanceRange] = ReplicateOp_Divide.UsesAverageConditional<Gaussian>(this.Weights_depth1_rep_B[FeatureRange][InstanceRange], this.Weights_depth1_rep_F_marginal[FeatureRange], InstanceRange, this.Weights_depth1_rep_F[FeatureRange][InstanceRange]);
						this.Weights_depth1_rep_F_marginal[FeatureRange] = ReplicateOp_Divide.MarginalIncrement<Gaussian>(this.Weights_depth1_rep_F_marginal[FeatureRange], this.Weights_depth1_rep_F[FeatureRange][InstanceRange], this.Weights_depth1_rep_B[FeatureRange][InstanceRange]);
					}
				}
				this.OnProgressChanged(new ProgressChangedEventArgs(iteration));
			}
			this.Changed_numberOfIterationsDecreased_FeatureCount_InstanceCount_WeightPriors_WeightConstraints_iterationsDone = numberOfIterations;
			this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 0;
		}

		/// <summary>Computations that depend on the observed value of InstanceCount and FeatureCount and FeatureValues and numberOfIterationsDecreased and WeightPriors and WeightConstraints</summary>
		private void Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10()
		{
			if (this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone==1) {
				return ;
			}
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
					// Message to 'FeatureScores' from Product factor
					this.FeatureScores_F[InstanceRange][FeatureRange] = GaussianProductOp.ProductAverageConditional(this.featureValues[InstanceRange][FeatureRange], this.Weights_depth1_rep_F[FeatureRange][InstanceRange]);
				}
				// Message to 'Score' from Sum factor
				this.Score_F[InstanceRange] = FastSumOp.SumAverageConditional(this.FeatureScores_F[InstanceRange]);
				// Message to 'NoisyScore' from GaussianFromMeanAndVariance factor
				this.NoisyScore_F[InstanceRange] = GaussianFromMeanAndVarianceOp.SampleAverageConditional(this.Score_F[InstanceRange], 1.0);
				// Message to 'Labels' from IsPositive factor
				this.Labels_F[InstanceRange] = IsPositiveOp.IsPositiveAverageConditional(this.NoisyScore_F[InstanceRange]);
				// Message to 'Labels_marginal' from DerivedVariable factor
				this.Labels_marginal_F[InstanceRange] = DerivedVariableOp.MarginalAverageConditional<Bernoulli>(this.Labels_use_B[InstanceRange], this.Labels_F[InstanceRange], this.Labels_marginal_F[InstanceRange]);
			}
			this.Changed_InstanceCount_FeatureCount_FeatureValues_numberOfIterationsDecreased_WeightPriors_WeightCons10_iterationsDone = 1;
		}

		/// <summary>
		/// Returns the marginal distribution for 'Labels' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Bernoulli,bool> LabelsMarginal()
		{
			return this.Labels_marginal_F;
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		#endregion

	}

}
