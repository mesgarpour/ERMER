using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace engine
{
    internal class Infer<T> : VectorGaussian
    {
        private Variable<Vector> variable;

        public Infer(Variable<Vector> variable)
        {
            this.variable = variable;
        }
    }
}