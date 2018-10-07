using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace NeuralNetwork
{
    interface IFunc
    {
        
    }

    //delegate double ActivationFunction(double d);

    class Net
    {
        //public double[] inputs;                   // вектор входных сигналов
        public Layer[] layers;                      // массив слоёв нейронов
        double[] outputs;                           // вектор выходных сигналов
        ActivationFunction activationFunc;          // делегат на функцию активации - одна для всех нейронов во всех слоях
        ErrorFunction errorFunc;                    // экземпляр класса функции ошибки
        double learnSpeed;                           // скорость обучения 0<learnSpeed<1
        double alfa;                                 // инерционный момент

        private Net() { }
        /// <summary>
        /// Создаёт нейронную сеть
        /// </summary>
        /// <param name="neuronsCount">массив целых значений с количеством нейронов в каждом слое по-порядку, псевдослой входящих сигналов в этот массив не входит</param>
        /// <param name="f">делегат на функцию активации</param>
        /// <param name="e">делегат на функцию ошибки</param>
        /// <param name="learnSpeed">скорость обучения</param>
        /// <param name="alfa">инерционный момент</param>

        public Net(int[] neuronsCount, ActivationFunction f, ErrorFunction errorFunc, double learnSpeed, double alfa)
        {
            this.learnSpeed = learnSpeed;
            this.alfa = alfa;
            layers = new Layer[neuronsCount.Length + 1];                        // добавляем 1 входной псевдослой
            outputs = new double[neuronsCount[neuronsCount.Length - 1]];        // количество выходных сигналов = количеству нейронов в последнем слое
            activationFunc = f;
            this.errorFunc = errorFunc;

            layers[0] = new Layer(neuronsCount[0]);                                  // нулевой слой - входные сигналы, их кол-во совпадает с кол-вом нейронов первого слоя нейронов
            for (int k = 1; k < neuronsCount.Length + 1; k++)
            {
                layers[k] = new Layer(neuronsCount[k-1], layers[k-1].OUT.Length, f);      // нулевой слой уже учли. начинаем с первого
            }
        }

        /// <summary>
        /// Функция прямого прохода по сети со входным вектором inputs
        /// </summary>
        /// <param name="inputs">вектор входных сигналов</param>
        public void DirectSolve(double[] inputs)
        {
            layers[0].OUT = inputs;                         // первый псевдослой - это вход сети
            for (int k = 1; k < layers.Length; k++)         // считаем все слои
            {
                layers[k].SolveLayerDirect(layers[k - 1].OUT);
            }
        }

        /// <summary>
        /// Функция обучения. На вход подаётся эпоха обучения.
        /// </summary>
        /// <param name="learningEpoch">словарь: ожидаемый вектор на выходе сети - обучающее множество</param>
        public void LearnEpoch(Dictionary< double[], double[]> learningEpoch)
        {
            if (learningEpoch.Count != layers.Last().neuronsCount)
            {
                MessageBox.Show("Количество обучающих выборок в эпохе не совпадает с количеством нейронов последнего слоя", "Функция обучения");
                return;
            }
            
            double[][] targetVectors= learningEpoch.Keys.ToArray();
            for (int i = 0; i < learningEpoch.Count; i++)
            {
                if (layers[0].OUT.Length != learningEpoch[targetVectors[i]].Length)
                {
                    MessageBox.Show("Размеры обучающего множества и входного векторов не совпадают", "Функция обучения");
                    return;
                }
                DirectSolve(learningEpoch[targetVectors[i]]);
                SolveError(targetVectors[i]);
                SolveBackPropagation(targetVectors[i], i == learningEpoch.Count - 1);
            }
        }

        private void SolveBackPropagation(double[] targetVector, bool isNewEpoch)
        {
            layers.Last().SolveBackPropagationLastLayer(learnSpeed, alfa, targetVector, layers[layers.Length-2].OUT, isNewEpoch);
            for (int k = layers.Length-2; k > 1 ; k--)      // первый псевдослой - не слой, последний слой - не скрытый
            {
                double[] depositOfNxtLayerNrnsLclGrad = Multiply(layers[k + 1].localGrad, layers[k + 1].weights);       // массив сумм взвешенных вкладов локальной ошибки каждого нейрона следующего слоя, длина массива = кол-ву входов следующего k+1-го слоя = кол-ву нейронов текущего k-го слоя
                layers[k].SolveBackPropagationHiddenLayer(learnSpeed, alfa, layers[k - 1].OUT, depositOfNxtLayerNrnsLclGrad, isNewEpoch);
            }
        }

        /*
        /// <summary>
        /// Функция обучения
        /// </summary>
        /// <param name="learningSet">обучающее множество</param>
        /// <param name="targetVector">ожидаемый вектор на выходе сети</param>
        private void Learn(double[] learningSet, double[] targetVector, bool isNewEpoch)
        {
            if (layers[0].OUT.Length != learningSet.Length)
            {
                MessageBox.Show("Размеры обучающего множества и входного векторов не совпадают", "Функция обучения");
                return;
            }
            //layers[0].OUT = learningSet;
            for (int epochCounter = 0; epochCounter < learningSet.Length / layers.Last().OUT.Length; epochCounter++)
            {
                DirectSolve(learningSet);
                SolveError(targetVector);
                SolveBackPropagation(targetVector, isNewEpoch);
            }
        }
        */

        /// <summary>
        /// Вычисление ошибки при обучении
        /// </summary>
        /// <param name="targetVector">ожидаемый вектор на выходе сети</param>
        /// <returns></returns>
        public double? SolveError(double[] targetVector)
        {
            if (layers.Last().OUT.Length != targetVector.Length)
            {
                MessageBox.Show("Размеры обучающего и выходного векторов не совпадают", "Функция ошибки");
                return null;
            }
            return (errorFunc.Solve(layers.Last().OUT, targetVector));
        }



        /// <summary>
        /// Умножение матрицы на вектор
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static double[] Multiply(double[,] matrix, double[] vector)
        {
            double[] result = new double[matrix.GetLength(0)];
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j< matrix.GetLength(1); j++)
                {
                    result[i] += matrix[i, j] * vector[j];
                }
            }
            return result;
        }

        /// <summary>
        /// Умножение векора на матрицу
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static double[] Multiply(double[] vector, double[,] matrix)
        {
            double[] result = new double[matrix.GetLength(1)];
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    result[j] +=vector[i] * matrix[i, j];
                }
            }
            return result;
        }
    }
}
