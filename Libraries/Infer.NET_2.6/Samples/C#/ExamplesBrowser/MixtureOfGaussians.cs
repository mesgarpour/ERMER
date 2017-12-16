using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MicrosoftResearch.Infer.Tutorials
{
	[Example("Tutorials", "Learning the means, precisions, and mixture probabilities of a mixture of two Gaussians", Prefix = "6.")]
	public class MixtureOfGaussians
	{
		public void Run()
		{
			// Define a range for the number of mixture components
			Range k = new Range(2).Named("k");

			// Mixture component means
			VariableArray<Vector> means = Variable.Array<Vector>(k).Named("means");			
			means[k] = Variable.VectorGaussianFromMeanAndPrecision(
				Vector.FromArray(0.0,0.0),
				PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)).ForEach(k);
	
			// Mixture component precisions
			VariableArray<PositiveDefiniteMatrix> precs = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs");
			precs[k] = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)).ForEach(k);
			
			// Mixture weights 
			Variable<Vector> weights = Variable.Dirichlet(k, new double[] { 1, 1 }).Named("weights");	

			// Create a variable array which will hold the data
			Range n = new Range(300).Named("n");
			VariableArray<Vector> data = Variable.Array<Vector>(n).Named("x");
			// Create latent indicator variable for each data point
			VariableArray<int> z = Variable.Array<int>(n).Named("z");

			// The mixture of Gaussians model
			using (Variable.ForEach(n)) {
				z[n] = Variable.Discrete(weights);
				using (Variable.Switch(z[n])) {
					data[n] = Variable.VectorGaussianFromMeanAndPrecision(means[z[n]], precs[z[n]]);
				}
			}

			// Attach some generated data
			data.ObservedValue = GenerateData(n.SizeAsInt);

			// Initialise messages randomly so as to break symmetry
			Discrete[] zinit = new Discrete[n.SizeAsInt];		
			for (int i = 0; i < zinit.Length; i++) 
			  zinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
			z.InitialiseTo(Distribution<int>.Array(zinit)); 

			// The inference
			InferenceEngine ie = new InferenceEngine();
			Console.WriteLine("Dist over pi=" + ie.Infer(weights));
			Console.WriteLine("Dist over means=\n" + ie.Infer(means));
			Console.WriteLine("Dist over precs=\n" + ie.Infer(precs));
		}


		/// <summary>
		/// Generates a data set from a particular true model.
		/// </summary>
		public Vector[] GenerateData(int nData)
		{
			Vector trueM1 = Vector.FromArray(2.0, 3.0);
			Vector trueM2 = Vector.FromArray(7.0, 5.0);
			PositiveDefiniteMatrix trueP1 = new PositiveDefiniteMatrix(
				new double[,] { { 3.0, 0.2 }, { 0.2, 2.0 } });
			PositiveDefiniteMatrix trueP2 = new PositiveDefiniteMatrix(
				new double[,] { { 2.0, 0.4 }, { 0.4, 4.0 } });
			VectorGaussian trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1);
			VectorGaussian trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2);
			double truePi = 0.6;
			Bernoulli trueB = new Bernoulli(truePi);
			// Restart the infer.NET random number generator
			Rand.Restart(12347);
			Vector[] data = new Vector[nData];
			for (int j = 0; j < nData; j++) {
				bool bSamp = trueB.Sample();
				data[j] = bSamp ? trueVG1.Sample() : trueVG2.Sample();
			}
			return data;
		}
	}
}
