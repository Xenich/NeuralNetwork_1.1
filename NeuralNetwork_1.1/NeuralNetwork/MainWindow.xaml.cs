#define D1

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.IO;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading;

namespace NeuralNetwork
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {


        public MainWindow()
        {
            InitializeComponent();
        }

        private void button_Click(object sender, RoutedEventArgs e)
        {
            txtBox.Clear();
            double[] learningSet = new double[] { 1, 2, 3, 4,5,6,7,8 };
            double[] targetVector = new double[] { 1, 2, 3, 3, 0 };

            ActivationFunction f = new ActivationFunction(VariousFunctions.Equals, VariousFunctions.DerivativeSigmoid);
            ErrorFunction err = new ErrorFunction(VariousFunctions.Error, null);
            Net net = new Net(new int[] { learningSet.Length,6,5 , targetVector.Length }, f, err, 1, 1);
            net.DirectSolve(learningSet);

            InputsPrint(net);
            WeightsPrint(net);
            OutputsPrint(net);
            txtBox.AppendText("Ошибка сети:" + net.SolveError(targetVector).ToString() + Environment.NewLine);
            //net.Learn(learningSet, targetVector);
            WeightsPrint(net);
            net.DirectSolve(learningSet);
            txtBox.AppendText("Ошибка сети:" + net.SolveError(targetVector).ToString() + Environment.NewLine);

        }

        private void OutputsPrint(Net net)                        // вывод выходных сигналов каждого слоя после подсчёта сети
        {
            txtBox.AppendText("Вывод выходных сигналов каждого слоя после подсчёта сети" + Environment.NewLine);
            for (int k = 1; k < net.layers.Length; k++) 
            {
                txtBox.AppendText("Layer: " + k.ToString() + Environment.NewLine);
                for (int i = 0; i < net.layers[k].OUT.Length; i++)
                {
                    txtBox.AppendText("out[" + i + "]=" + net.layers[k].OUT[i].ToString() + ";   ");
                }
                txtBox.AppendText(Environment.NewLine + Environment.NewLine);
            }
        }

        private void InputsPrint(Net net)                  // вывод входных сигналов
        {
            txtBox.AppendText("Вывод входных сигналов" + Environment.NewLine);
            for (int j = 0; j < net.layers[0].OUT.Length; j++) 
            {
                txtBox.AppendText("in[" + j.ToString() + "]=" + net.layers[0].OUT[j].ToString() + ";  ");
            }
            txtBox.AppendText(Environment.NewLine + Environment.NewLine);
        }

        private void WeightsPrint(Net net)        // вывод весов каждого слоя
        {
            txtBox.AppendText("Вывод весов каждого слоя" + Environment.NewLine);
            for (int k = 1; k < net.layers.Length; k++)
            {
                txtBox.AppendText("Layer: " + k.ToString() + Environment.NewLine);
                for (int i = 0; i < net.layers[k].weights.GetLength(0); i++)
                {
                    for (int j = 0; j < net.layers[k].weights.GetLength(1); j++)
                    {
                        txtBox.AppendText("w[" + i.ToString() + "," + j.ToString() + "]= " + net.layers[k].weights[i, j].ToString() + ";  ");
                    }
                    txtBox.AppendText(Environment.NewLine);
                }
            }
            txtBox.AppendText(Environment.NewLine + Environment.NewLine);
        }

        private void buttonOpen_Click(object sender, RoutedEventArgs e)
        {
            ActivationFunction f = new ActivationFunction(VariousFunctions.Sigmoid, VariousFunctions.DerivativeSigmoid);
            ErrorFunction err = new ErrorFunction(VariousFunctions.Error, null);
#if D1
            string[] strings = File.ReadAllLines("mnist_train_4500.csv");                // "mnist_train_300.csv"
#endif
         
            List< Dictionary<double[], double[]> > learningEpochas = new List<Dictionary<double[], double[]>>();
            
            int counter = 0;
            Dictionary<double[], double[]> dict = new Dictionary<double[], double[]>();
            foreach (string str in strings)
            {
                string[] st = str.Split(',');
                double symbol = Convert.ToDouble(st[0]);            // 
                double[] doubles = new double[st.Length - 1];

                for (int i = 0; i < doubles.Length; i++)
                {
                    doubles[i] = Convert.ToDouble(st[i + 1]);          // /256 - для нормализации
                }
                double[] target = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                target[Convert.ToInt32(symbol)] = 1;
                dict.Add(target, doubles);
                counter++;
                if (counter == 10)
                {
                    counter = 0;
                    learningEpochas.Add(dict);
                    dict = new Dictionary<double[], double[]>();
                }
            }
#if D1
            Net net = new Net(new int[] { 784, 10 }, f, err, 0.01, 0.01);
#endif

            Thread thrd = new Thread(new ParameterizedThreadStart(Solve));
            thrd.Start(new parametrs(learningEpochas, net));
        }

        private void Solve(object parameters)
        {
            parametrs param = (parametrs)parameters;
            int k = 0;
            foreach (Dictionary<double[], double[]> epoch in param.epochas)
            {
                k++;
                param.net.LearnEpoch(epoch);
                foreach (double[] target in epoch.Keys)
                    Dispatcher.Invoke(new Action( ()=> { txtBox.AppendText(k.ToString()+" - Ошибка сети:   " + param.net.SolveError(target).ToString() + Environment.NewLine); }));
            }
        }

        private double Normalize(double d )
        {
            return (d/256);
        }

        class parametrs
        {
            public List<Dictionary<double[], double[]>> epochas;
            public Net net;
            public parametrs(List<Dictionary<double[], double[]>> epochas, Net net)
            {
                this.epochas = epochas;
                this.net = net;
            }
        }

        private void testBtn_Click(object sender, RoutedEventArgs e)
        {
            string[] strings = File.ReadAllLines("mnist_train.csv");                // "mnist_train_300.csv"
            List<string>[] list = new List<string>[10];
            for (int i = 0; i < 10; i++)
            {
                list[i] = new List<string>();
            }
                for (int i = 0; i < 6000; i++)
            {
                char ch = (strings[i])[0];
                double index = Convert.ToDouble(ch.ToString());
                int ind = Convert.ToInt32(index);
                list[ind].Add(strings[i]);
            }
            string s="";
            for (int i = 0; i < 450; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    s += list[j][i] + Environment.NewLine;
                }
            }
        }
    }
}
