using System;
using System.IO;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Neuron_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataset = new List<Tuple<double, double[]>>
            {
                new Tuple<double, double[]> ( 0, new double[] { 0, 0} ),
                new Tuple<double, double[]> ( 1, new double[] { 0, 1} ),
                new Tuple<double, double[]> ( 1, new double[] { 1, 0} ),
                new Tuple<double, double[]> ( 1, new double[] { 1, 1} ),
            };

            Topology topology = new Topology(2, 1, 0.3, new int[] { 8 });

            //var neuralNetwork = JsonConvert.DeserializeObject<NeuralNetwork>(File.ReadAllText("neuralnetwork.json"));
            //if (!loadlearning)
            //{
                var neuralNetwork = new NeuralNetwork(topology);
            //}

            var difference = neuralNetwork.Learn(dataset, 1000);

            var results = new List<double>();

            foreach (var data in dataset)
            {
                var res = neuralNetwork.FeedForward(data.Item2).Output;
                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 4);
                var actual = Math.Round(results[i], 4);
                Console.WriteLine("Ответ " + i + ": Ожидалось: " + expected + ", Сейчас: " + actual);
            }

            Console.WriteLine("\nТочность: " + (100 - Math.Round(difference * 100, 2)) + "%");

            String n = JsonConvert.SerializeObject(neuralNetwork);

            //File.WriteAllText("neuralnetwork.json", n);

            Console.ReadKey();
        }
    }
}