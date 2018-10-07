using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Layer
    {
        public int neuronsCount;                    // количество нейронов в слое
        public int inputCount;                      // количество входных сигналов - количество нейронов предыдущего слоя (сеть полносвязная)      
        public double[,] weights;                   // матрица синаптических весов W[i,j] -  i-й нейрон, j-й вход
        public double[] OUT;                        // аксоны, вектор выходных сигналов, длина = кол-ву нейронов
        public double[] localField;                 // массив локальных индуцированных полей каждого нейрона
        public double[,] deltaW;                    // изменение весов
        public double[] localGrad;                  // вектор локальных градиентов каждого нейрона
        public double bias = 1;                     // порог
        //double[] input1;//


        public ActivationFunction activationFunc;          // функция активации

        /// <summary>
        /// Конструктор входного слоя
        /// </summary>
        /// <param name="inputs">вектор входных сигналов</param>
        public Layer(int inputsCount)
        {
            weights = null;
            OUT = new double[inputsCount];           // слой ничего не делает
        }

        /// <summary>
        /// Конструктор слоя
        /// </summary>
        /// <param name="neuronsCount">количество нейронов в слое</param>
        /// <param name="inputCount">количество входных сигналов - количество нейронов предыдущего слоя (сеть полносвязная)</param>
        /// <param name="f">делегат на функцию активации</param>
        public Layer(int neuronsCount, int inputCount, ActivationFunction f)
        {
            this.neuronsCount = neuronsCount;
            this.inputCount = inputCount + 1;                     //    + смещение bias
            weights = new double[neuronsCount, inputCount + 1];   //
            deltaW = new double[neuronsCount, inputCount + 1];    //
            localGrad = new double[neuronsCount];
            OUT = new double[neuronsCount];
            SetActivationFunction(f);
            Random rnd = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < neuronsCount; i++)
            {
                for (int j = 0; j < inputCount+1; j++)
                {
                    weights[i, j] = ((double)rnd.Next(0, 100)-50) / 1000;       // назначаем весам малые случайные значения
                }
            }
        }

        /// <summary>
        /// Рассчёт выходных сигналов каждого нейрона слоя при прямом прохождении сигнала
        /// </summary>
        /// <param name="input">входные сигналы из выхода предыдущего слоя</param>
        /// <returns></returns>
        public void SolveLayerDirect(double[] inputWithoutBias)
        {
            double[] input = InputWithBias(inputWithoutBias);
            localField = Net.Multiply(weights, input);     // рассчёт вектора локального индуцированного поля - вектора потенциалов активации

            for (int i = 0; i < neuronsCount; i++)
            {
                OUT[i] = activationFunc.Solve(localField[i]);
            }
        }


        /// <summary>
        /// Рассчёт обратного распространения ошибки в случае, если слой - последний
        /// </summary>
        /// <param name="learnSpeed">коэффициент скорости обучения</param>
        /// <param name="alfa">инерционный момент</param> 
        /// <param name="targetVector">ожидаемый векор выходных значений</param>
        /// <param name="input">вектор входных сигналов из предыдущего слоя</param>
        /// <param name="isNewEpoch">новая эпоха обучения?</param>
        public void SolveBackPropagationLastLayer(double learnSpeed, double alfa, double[] targetVector, double[] inputWithoutBias, bool isNewEpoch)
        {
            double[] input = InputWithBias(inputWithoutBias);
            double[,] deltaWThisEra = new double[neuronsCount, inputCount];       // дельты новой эпохи
            
            for (int i = 0; i < neuronsCount; i++)          // нейрон
            {
                localGrad[i] = (targetVector[i] - OUT[i]) * activationFunc.SolveDerivative(localField[i]);
                for (int j = 0; j < inputCount; j++)      // дендриты i-го нейрона
                {
                    deltaWThisEra[i, j] = alfa * deltaW[i,j] + learnSpeed * localGrad[i] * input[j];            //
                    weights[i, j] += deltaWThisEra[i, j];
                }
            }
            if(isNewEpoch)
                deltaW = deltaWThisEra; 
        }



        /// <summary>
        /// Рассчёт обратного распространения ошибки в случае, если слой - скрытый
        /// </summary>
        /// <param name="learnSpeed">коэффициент скорости обучения</param>
        /// <param name="alfa">инерционный момент<</param>
        /// <param name="input">вектор входных сигналов из предыдущего слоя</param>
        /// <param name="depositOfNxtLayerNrnsLclGrad">массив сумм взвешенных вкладов локальной ошибки каждого нейрона следующего слоя, длина массива = кол-ву входов следующего k+1-го слоя = кол-ву нейронов текущего k-го слоя</param>
        /// <param name="isNewEpoch">новая эпоха обучения?</param>
        public void SolveBackPropagationHiddenLayer(double learnSpeed, double alfa, double[] inputWithoutBias, double[] depositOfNxtLayerNrnsLclGrad, bool isNewEpoch)
        {
            double[] input = InputWithBias(inputWithoutBias);
            double[,] deltaWThisEra = new double[neuronsCount, inputCount];                     // дельты новой эпохи

            for (int i = 0; i < neuronsCount; i++)                                              // i-й нейрон
            {
                localGrad[i] = activationFunc.SolveDerivative(localField[i]) * depositOfNxtLayerNrnsLclGrad[i];
                for (int j = 0; j < inputCount; j++)      // дендриты i-го нейрона
                {
                    deltaWThisEra[i, j] = alfa * deltaW[i, j] + learnSpeed * localGrad[i] * input[j];               //
                    weights[i, j] += deltaWThisEra[i, j];
                }
            }
            if (isNewEpoch)
                deltaW = deltaWThisEra;
        }

        private double[] InputWithBias(double[] inputWithoutBias)
        {
            double[] input = new double[inputWithoutBias.Length + 1];
            input[0] = bias;
            for (int i = 0; i < inputWithoutBias.Length; i++)
            {
                input[i + 1] = inputWithoutBias[i];
            }
            return input;
        }

        public void SetActivationFunction(ActivationFunction f)
        {
            activationFunc = f;
        }
    }
}
