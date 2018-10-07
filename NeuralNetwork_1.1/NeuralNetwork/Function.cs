using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    interface IFunction
    {
        double Solve();
        double SolveDerivative();
    }

    delegate double ActivationFunctionDel(double d);
    delegate double ActivationFunctionDerivativeDel(double d);

    delegate double ErrorFunctionDel(double[] outVector, double[] targetVector);
    delegate double ErrorFunctionDerivativeDel(double d);


    class ErrorFunction
    {
        ErrorFunctionDel errorFunction;
        ErrorFunctionDerivativeDel errorFunctionDerivative;

        public ErrorFunction(ErrorFunctionDel f, ErrorFunctionDerivativeDel d)
        {
            errorFunction = f;
            errorFunctionDerivative = d;
        }

        public double Solve(double[] outVector, double[] targetVector)
        {
            return errorFunction(outVector, targetVector);
        }

        public double SolveDerivative(double d)
        {
            return errorFunctionDerivative(d);
        }
    }

    class ActivationFunction
    {
        ActivationFunctionDel activationFunction;
        ActivationFunctionDerivativeDel activationFunctionDerivative;

        public ActivationFunction(ActivationFunctionDel f, ActivationFunctionDerivativeDel d)
        {
            activationFunction = f;
            activationFunctionDerivative = d;
        }
        public double Solve(double d)
        {
            return activationFunction(d);
        }

        public double SolveDerivative(double d)
        {
            return activationFunctionDerivative(d);
        }

    }
}
