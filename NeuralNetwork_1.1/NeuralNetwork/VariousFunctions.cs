using System;
using System.Windows;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class VariousFunctions
    {
        static double a =0.5;                   // параметр наклона сигмоиды

        public static double Sigmoid(double d)
        {
            return 1 / (1 + Math.Exp(-a * d));
           // return 2 / (1 + Math.Exp(-a * d)) - 1;
        }
        public static double DerivativeSigmoid(double d)
        {
            double sigm = Sigmoid(d);
            return a * sigm * (1 - sigm);
        }

        public static double Error(double[] outVector, double[] expectedVector)
        {
            if (outVector.Length != expectedVector.Length)
                MessageBox.Show("Размеры обучающего и выходного векторов не совпадают", "Функция ошибки");
            double error = 0;
            for (int i = 0; i < outVector.Length; i++)
            {
                error += Math.Pow((expectedVector[i] - outVector[i]), 2);
            }
            return Math.Sqrt(error);
        }

        public static double Equals(double d)
        {
            return d;
        }
    }
}
