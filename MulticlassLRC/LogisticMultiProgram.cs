using System;
using System.Collections.Generic;
using System.Linq;

namespace LogisticMultiClassGradient
{
	public class MultiClassLogisticRegressionClassification
	{
		public int NumberOfFeatures { get; set; }
		public int NumberOfClassifications { get; set; }

		private double[][] mergedTrainingData; // [features + classifications]
		private double[][] weights; // [feature][classification]
		private double[] biases; // [classifications]

		public MultiClassLogisticRegressionClassification(int numberOfFeatures, int numberOfClassifications)
		{
			this.NumberOfFeatures = numberOfFeatures;
			this.NumberOfClassifications = numberOfClassifications;

			this.weights = MakeMatrix(numberOfFeatures, numberOfClassifications);
			this.biases = new double[numberOfClassifications];
		}

		/// <summary>
		/// Loads known input and output data for use in training the model.
		/// </summary>
		/// <param name="trainingData">The training data used to train the model</param>
		public void LoadTrainingData(IEnumerable<TrainingSet> trainingData)
		{
			if (trainingData == null) throw new ArgumentNullException("trainingData");

			Console.WriteLine("\nTraining data: \n");

			mergedTrainingData = trainingData.Select(s => s.FeatureValues.Concat(s.OutputValues).ToArray()).ToArray();

			ShowData(mergedTrainingData, 3, 2, true);
		}

		public void Train(int maxEpochs, double learnRate, double decay)
		{
			// 'batch' approach (aggregate gradients using all data)
			double[] targets = new double[this.NumberOfClassifications];
			int msgInterval = maxEpochs / 10;
			int epoch = 0;
			while (epoch < maxEpochs)
			{
				++epoch;

				if (epoch % msgInterval == 0 && epoch != maxEpochs)
				{
					double mse = Error(this.mergedTrainingData);
					Console.Write("epoch = " + epoch);
					Console.Write("  error = " + mse.ToString("F4"));
					double acc = Accuracy(this.mergedTrainingData);
					Console.WriteLine("  accuracy = " + acc.ToString("F4"));
				}

				double[][] weightGrads = MakeMatrix(this.NumberOfFeatures, this.NumberOfClassifications);
				double[] biasGrads = new double[this.NumberOfClassifications];

				// compute all weight gradients, (all classes, all inputs)
				for (int j = 0; j < this.NumberOfClassifications; ++j)
				{
					for (int i = 0; i < this.NumberOfFeatures; ++i)
					{
						for (int r = 0; r < this.mergedTrainingData.Length; ++r)
						{
							double[] outputs = ComputeOutputs(this.mergedTrainingData[r]);
							
							for (int k = 0; k < this.NumberOfClassifications; ++k)
							{
								targets[k] = this.mergedTrainingData[r][this.NumberOfFeatures + k];
							}

							double input = this.mergedTrainingData[r][i];
							weightGrads[i][j] += (targets[j] - outputs[j]) * input;
						}
					}
				}

				// compute all bias gradients (all classes, all inputs)
				for (int j = 0; j < this.NumberOfClassifications; ++j)
				{
					for (int i = 0; i < this.NumberOfFeatures; ++i)
					{
						for (int r = 0; r < this.mergedTrainingData.Length; ++r)
						{
							double[] outputs = ComputeOutputs(this.mergedTrainingData[r]);
							
							for (int k = 0; k < this.NumberOfClassifications; ++k)
							{
								targets[k] = this.mergedTrainingData[r][this.NumberOfFeatures + k];
							}

							double input = 1; // 1 is a dummy input
							biasGrads[j] += (targets[j] - outputs[j]) * input;
						}
					}
				}

				// update all weights
				for (int i = 0; i < this.NumberOfFeatures; ++i)
				{
					for (int j = 0; j < this.NumberOfClassifications; ++j)
					{
						weights[i][j] += learnRate * weightGrads[i][j];
						weights[i][j] *= (1 - decay);  // wt decay
					}
				}

				// update all biases
				for (int j = 0; j < this.NumberOfClassifications; ++j)
				{
					biases[j] += learnRate * biasGrads[j];
					biases[j] *= (1 - decay);
				}

			} // while
		}

		public void SetWeights(double[][] wts, double[] b)
		{
			// set weights[][] and biases[]
			for (int i = 0; i < this.NumberOfFeatures; ++i)
				for (int j = 0; j < this.NumberOfClassifications; ++j)
					this.weights[i][j] = wts[i][j];
			for (int j = 0; j < this.NumberOfClassifications; ++j)
				this.biases[j] = b[j];
		}

		public double[][] GetWeights()
		{
			double[][] result = new double[this.NumberOfFeatures][];
			for (int i = 0; i < this.NumberOfFeatures; ++i)
				result[i] = new double[this.NumberOfClassifications];
			for (int i = 0; i < this.NumberOfFeatures; ++i)
				for (int j = 0; j < this.NumberOfClassifications; ++j)
					result[i][j] = this.weights[i][j];
			return result;
		}

		public double[] GetBiases()
		{
			double[] result = new double[this.NumberOfClassifications];
			for (int j = 0; j < this.NumberOfClassifications; ++j)
				result[j] = this.biases[j];
			return result;
		}

		public double Error(double[][] mergedTrainingData)
		{
			// mean squared error
			double sumSquaredError = 0.0;
			for (int i = 0; i < mergedTrainingData.Length; ++i) // each training item
			{
				double[] outputs = this.ComputeOutputs(mergedTrainingData[i]);
				for (int j = 0; j < outputs.Length; ++j)
				{
					int jj = this.NumberOfFeatures + j; // column in this.mergedTrainingData to use
					sumSquaredError += ((outputs[j] - mergedTrainingData[i][jj]) *
					  (outputs[j] - mergedTrainingData[i][jj]));
				}
			}
			return sumSquaredError / mergedTrainingData.Length;
		}

		public double Accuracy(double[][] trainData)
		{
			// using curr wts and biases
			int numCorrect = 0;
			int numWrong = 0;

			for (int i = 0; i < trainData.Length; ++i)
			{
				int[] deps = ComputeDependents(trainData[i]); // ex: [0  1  0]
				double[] targets = new double[this.NumberOfClassifications]; // ex: [0.0  1.0  0.0]
				for (int j = 0; j < this.NumberOfClassifications; ++j)
					targets[j] = trainData[i][this.NumberOfFeatures + j];

				int di = MaxIndex(deps);
				int ti = MaxIndex(targets);
				if (di == ti)
					++numCorrect;
				else
					++numWrong;
			}

			return (numCorrect * 1.0) / (numWrong + numCorrect);
		}



		private int[] ComputeDependents(double[] dataItem)
		{
			double[] outputs = ComputeOutputs(dataItem); // 0.0 to 1.0
			int maxIndex = MaxIndex(outputs);
			int[] result = new int[this.NumberOfClassifications]; // [0 0 .. 0]
			result[maxIndex] = 1;
			return result;
		}

		private double[] ComputeOutputs(double[] dataItem)
		{
			// using curr weights[][] and biases[]
			// dataItem can be just x or x+y
			double[] result = new double[this.NumberOfClassifications];
			for (int j = 0; j < this.NumberOfClassifications; ++j) // compute z
			{
				for (int i = 0; i < this.NumberOfFeatures; ++i)
					result[j] += dataItem[i] * weights[i][j];
				result[j] += biases[j];
			}

			for (int j = 0; j < this.NumberOfClassifications; ++j) // 1 / 1 + e^-z
				result[j] = 1.0 / (1.0 + Math.Exp(-result[j]));

			double sum = 0.0; // softmax scaling
			for (int j = 0; j < this.NumberOfClassifications; ++j)
				sum += result[j];

			for (int j = 0; j < this.NumberOfClassifications; ++j)
				result[j] = result[j] / sum;

			return result;
		}



		public static void ShowData(double[][] data, int numRows, int decimals, bool indices)
		{
			int len = data.Length.ToString().Length;
			for (int i = 0; i < numRows; ++i)
			{
				if (indices == true) Console.Write("[" + i.ToString().PadLeft(len) + "]  ");

				for (int j = 0; j < data[i].Length; ++j)
				{
					double v = data[i][j];
					if (v >= 0.0) Console.Write(" "); // '+'
					Console.Write(v.ToString("F" + decimals) + "  ");
				}

				Console.WriteLine("");
			}

			if (numRows < data.Length)
			{
				Console.WriteLine(". . .");
				int lastRow = data.Length - 1;
				if (indices == true) Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");

				for (int j = 0; j < data[lastRow].Length; ++j)
				{
					double v = data[lastRow][j];
					if (v >= 0.0) Console.Write(" "); // '+'
					Console.Write(v.ToString("F" + decimals) + "  ");
				}
			}

			Console.WriteLine("\n");
		}

		public static void ShowVector(double[] vector, int decimals, bool newLine)
		{
			for (int i = 0; i < vector.Length; ++i)
				Console.Write(vector[i].ToString("F" + decimals) + " ");

			Console.WriteLine("");
			if (newLine == true) Console.WriteLine("");
		}

		private static double[][] MakeMatrix(int rows, int cols)
		{
			double[][] result = new double[rows][];
			for (int i = 0; i < rows; ++i)
				result[i] = new double[cols];
			return result;
		}

		private static int MaxIndex(double[] vector)
		{
			int maxIndex = 0;
			double maxVal = vector[0];
			for (int i = 0; i < vector.Length; ++i)
			{
				if (vector[i] > maxVal)
				{
					maxVal = vector[i];
					maxIndex = i;
				}
			}
			return maxIndex;
		}

		private static int MaxIndex(int[] vector)
		{
			int maxIndex = 0;
			int maxVal = vector[0];
			for (int i = 0; i < vector.Length; ++i)
			{
				if (vector[i] > maxVal)
				{
					maxVal = vector[i];
					maxIndex = i;
				}
			}
			return maxIndex;
		}
	}

	public struct TrainingSet
	{
		public double[] FeatureValues, OutputValues;

		public TrainingSet(double[] featureValues, double[] outputValues)
		{
			this.FeatureValues = featureValues;
			this.OutputValues = outputValues;
		}
	}

	class LogisticMultiProgram
	{
		static void Main(string[] args)
		{
			Console.WriteLine("\nBegin multi-class logistic regression classification demo");

			int numFeatures = 4;
			int numClasses = 3;
			int numRows = 1000;
			Console.WriteLine("\nGenerating " + numRows +
			  " artificial data items with " + numFeatures + " features");
			double[][] data = MakeDummyData(numFeatures, numClasses, numRows, 0);

			MultiClassLogisticRegressionClassification classifier = new MultiClassLogisticRegressionClassification(numFeatures, numClasses);
			classifier.LoadTrainingData(data.Take((int)(numRows*0.8)).Select(s => new TrainingSet(s.Take(numFeatures).ToArray(), s.Skip(numFeatures).ToArray())));

			var testData = data.Skip((int)(numRows * 0.8)).ToArray();
			Console.WriteLine("\nTest data: \n");
			MultiClassLogisticRegressionClassification.ShowData(testData, 3, 2, true);

			int maxEpochs = 100;
			Console.WriteLine("Setting training maxEpochs = " + maxEpochs);

			double learnRate = 0.01;
			Console.WriteLine("Setting learning rate      = " + learnRate.ToString("F2"));

			double decay = 0.10;
			Console.WriteLine("Setting weight decay       = " + decay.ToString("F2"));

			classifier.Train(maxEpochs, learnRate, decay);

			//Console.WriteLine("Splitting data to train (80%) and test matrices");
			//double[][] trainData;
			//double[][] testData;
			//SplitTrainTest(data, 0.80, 7, out trainData, out testData);
			//Console.WriteLine("Done");

			//Console.WriteLine("\nTraining data: \n");
			//ShowData(trainData, 3, 2, true);

			//Console.WriteLine("Creating multi-class LR classifier");
			//LogisticMulti lc = new LogisticMulti(numFeatures, numClasses);



			//Console.WriteLine("\nStarting training using (batch) gradient descent\n"); 
			//lc.Train(trainData, maxEpochs, learnRate, decay);
			//Console.WriteLine("\nDone\n");

			double[][] bestWts = classifier.GetWeights();
			double[] bestBiases = classifier.GetBiases();

			//Console.WriteLine("Best weights found:");
			MultiClassLogisticRegressionClassification.ShowData(bestWts, bestWts.Length, 3, true);

			//Console.WriteLine("Best biases found:");
			MultiClassLogisticRegressionClassification.ShowVector(bestBiases, 3, true);

			double trainAcc = classifier.Accuracy(testData);
			Console.WriteLine("Prediction accuracy on training data = " + trainAcc.ToString("F4"));

			Console.WriteLine("\nEnd demo\n");
			Console.ReadLine();
		} // Main

		public static void ShowData(double[][] data, int numRows,
		  int decimals, bool indices)
		{
			int len = data.Length.ToString().Length;
			for (int i = 0; i < numRows; ++i)
			{
				if (indices == true)
					Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
				for (int j = 0; j < data[i].Length; ++j)
				{
					double v = data[i][j];
					if (v >= 0.0)
						Console.Write(" "); // '+'
					Console.Write(v.ToString("F" + decimals) + "  ");
				}
				Console.WriteLine("");
			}

			if (numRows < data.Length)
			{
				Console.WriteLine(". . .");
				int lastRow = data.Length - 1;
				if (indices == true)
					Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
				for (int j = 0; j < data[lastRow].Length; ++j)
				{
					double v = data[lastRow][j];
					if (v >= 0.0)
						Console.Write(" "); // '+'
					Console.Write(v.ToString("F" + decimals) + "  ");
				}
			}
			Console.WriteLine("\n");
		}

		public static void ShowVector(double[] vector, int decimals,
		  bool newLine)
		{
			for (int i = 0; i < vector.Length; ++i)
				Console.Write(vector[i].ToString("F" + decimals) + " ");
			Console.WriteLine("");
			if (newLine == true)
				Console.WriteLine("");
		}

		static double[][] MakeDummyData(int numFeatures,
		  int numClasses, int numRows, int seed)
		{
			Random rnd = new Random(seed); // make random wts and biases
			double[][] wts = new double[numFeatures][];
			for (int i = 0; i < numFeatures; ++i)
				wts[i] = new double[numClasses];
			double hi = 10.0;
			double lo = -10.0;
			for (int i = 0; i < numFeatures; ++i)
				for (int j = 0; j < numClasses; ++j)
					wts[i][j] = (hi - lo) * rnd.NextDouble() + lo;
			double[] biases = new double[numClasses];
			for (int i = 0; i < numClasses; ++i)
				biases[i] = (hi - lo) * rnd.NextDouble() + lo;

			Console.WriteLine("Generating weights are: ");
			ShowData(wts, wts.Length, 2, true);
			Console.WriteLine("Generating biases are: ");
			ShowVector(biases, 2, true);

			double[][] result = new double[numRows][]; // allocate result
			for (int i = 0; i < numRows; ++i)
				result[i] = new double[numFeatures + numClasses];

			for (int i = 0; i < numRows; ++i) // create one row at a time
			{
				double[] x = new double[numFeatures]; // generate random x-values
				for (int j = 0; j < numFeatures; ++j)
					x[j] = (hi - lo) * rnd.NextDouble() + lo;

				double[] y = new double[numClasses]; // computed outputs storage
				for (int j = 0; j < numClasses; ++j) // compute z-values
				{
					for (int f = 0; f < numFeatures; ++f)
						y[j] += x[f] * wts[f][j];
					y[j] += biases[j];
				}

				// determine loc. of max (no need for 1 / 1 + e^-z)
				int maxIndex = 0;
				double maxVal = y[0];
				for (int c = 0; c < numClasses; ++c)
				{
					if (y[c] > maxVal)
					{
						maxVal = y[c];
						maxIndex = c;
					}
				}

				for (int c = 0; c < numClasses; ++c) // convert y to 0s or 1s
					if (c == maxIndex)
						y[c] = 1.0;
					else
						y[c] = 0.0;

				int col = 0; // copy x and y into result
				for (int f = 0; f < numFeatures; ++f)
					result[i][col++] = x[f];
				for (int c = 0; c < numClasses; ++c)
					result[i][col++] = y[c];
			}
			return result;
		}

		static void SplitTrainTest(double[][] allData, double trainPct,
		  int seed, out double[][] trainData, out double[][] testData)
		{
			Random rnd = new Random(seed);
			int totRows = allData.Length;
			int numTrainRows = (int)(totRows * trainPct); // typically 80% 
			int numTestRows = totRows - numTrainRows;
			trainData = new double[numTrainRows][];
			testData = new double[numTestRows][];

			double[][] copy = new double[allData.Length][]; // ref copy of all data
			for (int i = 0; i < copy.Length; ++i)
				copy[i] = allData[i];

			for (int i = 0; i < copy.Length; ++i) // scramble order
			{
				int r = rnd.Next(i, copy.Length); // use Fisher-Yates
				double[] tmp = copy[r];
				copy[r] = copy[i];
				copy[i] = tmp;
			}
			for (int i = 0; i < numTrainRows; ++i)
				trainData[i] = copy[i];

			for (int i = 0; i < numTestRows; ++i)
				testData[i] = copy[i + numTrainRows];
		} // MakeTrainTest

	} 

	public class LogisticMulti
	{
		private int numFeatures;
		private int numClasses;
		private double[][] weights; // [feature][class]
		private double[] biases;    // [class]

		public LogisticMulti(int numFeatures, int numClasses)
		{
			this.numFeatures = numFeatures;
			this.numClasses = numClasses;
			this.weights = MakeMatrix(numFeatures, numClasses);
			this.biases = new double[numClasses];
		}

		private double[][] MakeMatrix(int rows, int cols)
		{
			double[][] result = new double[rows][];
			for (int i = 0; i < rows; ++i)
				result[i] = new double[cols];
			return result;
		}

		public void SetWeights(double[][] wts, double[] b)
		{
			// set weights[][] and biases[]
			for (int i = 0; i < numFeatures; ++i)
				for (int j = 0; j < numClasses; ++j)
					this.weights[i][j] = wts[i][j];
			for (int j = 0; j < numClasses; ++j)
				this.biases[j] = b[j];
		}

		public double[][] GetWeights()
		{
			double[][] result = new double[numFeatures][];
			for (int i = 0; i < numFeatures; ++i)
				result[i] = new double[numClasses];
			for (int i = 0; i < numFeatures; ++i)
				for (int j = 0; j < numClasses; ++j)
					result[i][j] = this.weights[i][j];
			return result;
		}

		public double[] GetBiases()
		{
			double[] result = new double[numClasses];
			for (int j = 0; j < numClasses; ++j)
				result[j] = this.biases[j];
			return result;
		}

		private double[] ComputeOutputs(double[] dataItem)
		{
			// using curr weights[][] and biases[]
			// dataItem can be just x or x+y
			double[] result = new double[numClasses];
			for (int j = 0; j < numClasses; ++j) // compute z
			{
				for (int i = 0; i < numFeatures; ++i)
					result[j] += dataItem[i] * weights[i][j];
				result[j] += biases[j];
			}

			for (int j = 0; j < numClasses; ++j) // 1 / 1 + e^-z
				result[j] = 1.0 / (1.0 + Math.Exp(-result[j]));

			double sum = 0.0; // softmax scaling
			for (int j = 0; j < numClasses; ++j)
				sum += result[j];

			for (int j = 0; j < numClasses; ++j)
				result[j] = result[j] / sum;

			return result;
		} // ComputeOutputs

		public void Train(double[][] trainData, int maxEpochs,
		  double learnRate, double decay)
		{
			// 'batch' approach (aggregate gradients using all data)
			double[] targets = new double[numClasses];
			int msgInterval = maxEpochs / 10;
			int epoch = 0;
			while (epoch < maxEpochs)
			{
				++epoch;

				if (epoch % msgInterval == 0 && epoch != maxEpochs)
				{
					double mse = Error(trainData);
					Console.Write("epoch = " + epoch);
					Console.Write("  error = " + mse.ToString("F4"));
					double acc = Accuracy(trainData);
					Console.WriteLine("  accuracy = " + acc.ToString("F4"));
				}

				double[][] weightGrads = MakeMatrix(numFeatures, numClasses);
				double[] biasGrads = new double[numClasses];

				// compute all weight gradients, (all classes, all inputs)
				for (int j = 0; j < numClasses; ++j)
				{
					for (int i = 0; i < numFeatures; ++i)
					{
						for (int r = 0; r < trainData.Length; ++r)
						{
							double[] outputs = ComputeOutputs(trainData[r]);
							for (int k = 0; k < numClasses; ++k)
								targets[k] = trainData[r][numFeatures + k];
							double input = trainData[r][i];
							weightGrads[i][j] += (targets[j] - outputs[j]) * input;
						}
					}
				}

				// compute all bias gradients (all classes, all inputs)
				for (int j = 0; j < numClasses; ++j)
				{
					for (int i = 0; i < numFeatures; ++i)
					{
						for (int r = 0; r < trainData.Length; ++r)
						{
							double[] outputs = ComputeOutputs(trainData[r]);
							for (int k = 0; k < numClasses; ++k)
								targets[k] = trainData[r][numFeatures + k];
							double input = 1; // 1 is a dummy input
							biasGrads[j] += (targets[j] - outputs[j]) * input;
						}
					}
				}

				// update all weights
				for (int i = 0; i < numFeatures; ++i)
				{
					for (int j = 0; j < numClasses; ++j)
					{
						weights[i][j] += learnRate * weightGrads[i][j];
						weights[i][j] *= (1 - decay);  // wt decay
					}
				}

				// update all biases
				for (int j = 0; j < numClasses; ++j)
				{
					biases[j] += learnRate * biasGrads[j];
					biases[j] *= (1 - decay);
				}

			} // while
		} // Train



		//public void TrainStochastic(double[][] trainData, int maxEpochs, double learnRate)
		//{
		//  int epoch = 0;
		//  int[] sequence = new int[trainData.Length]; // random order
		//  for (int i = 0; i < sequence.Length; ++i)
		//    sequence[i] = i;

		//  while (epoch < maxEpochs)
		//  {
		//    ++epoch;

		//    if (epoch % 10 == 0 && epoch != maxEpochs)
		//    {
		//      double mse = Error(trainData);
		//      Console.Write("epoch = " + epoch);
		//      Console.WriteLine("  error = " + mse.ToString("F4"));
		//    }

		//    Shuffle(sequence); // process data in random order

		//    // "stochastic"/"online"/"incremental" approach
		//    // picture essential to understand
		//    for (int ti = 0; ti < trainData.Length; ++ti)
		//    {
		//      //Console.WriteLine("===================================");
		//      int r = sequence[ti]; // row index
		//      double[] computeds = ComputeOutputs(trainData[r]); // computed outputs

		//      //Console.WriteLine("Data:");
		//      //LogisticMultiProgram.ShowVector(trainData[r], 2, true);

		//      for (int j = 0; j < numClasses; ++j) // update weights for each class
		//      {
		//        double target = trainData[r][numFeatures + j]; // target 0.0 or 1.0 for curr class
		//        double computed = computeds[j];                // computed [0.0 to 1.0] for curr class

		//        for (int i = 0; i < numFeatures; ++i) // each class-specific weight
		//        {
		//          double input = trainData[r][i]; // associated curr input value
		//          //Console.WriteLine("target = " + target.ToString("F2"));
		//          //Console.WriteLine("computed = " + computed.ToString("F2"));
		//          //Console.WriteLine("input = " + input.ToString("F2"));
		//          //Console.WriteLine("old wt = " + weights[i][j].ToString("F2"));
		//          weights[i][j] += learnRate * (target - computed) * input;
		//          //Console.WriteLine("new wt = " + weights[i][j].ToString("F2"));
		//          //Console.WriteLine("");
		//        } // i = each feature
		//      } // j = each class

		//      // update biases for each class
		//      for (int j = 0; j < numClasses; ++j)
		//      {
		//        int jj = numFeatures + j;
		//        double target = trainData[r][jj];
		//        double computed = computeds[j];
		//        biases[j] += learnRate * (target - computed); // input is a dummy 1.0
		//      }

		//      //Console.WriteLine("===================================");
		//      //Console.ReadLine();
		//    } // ti
		//  } // while

		//  //return this.weights;
		//} // Train

		//private void Shuffle(int[] sequence)
		//{
		//  for (int i = 0; i < sequence.Length; ++i)
		//  {
		//    int r = rnd.Next(i, sequence.Length);
		//    int tmp = sequence[r];
		//    sequence[r] = sequence[i];
		//    sequence[i] = tmp;
		//  }
		//}

		public double Error(double[][] trainData)
		{
			// mean squared error
			double sumSquaredError = 0.0;
			for (int i = 0; i < trainData.Length; ++i) // each training item
			{
				double[] outputs = this.ComputeOutputs(trainData[i]);
				for (int j = 0; j < outputs.Length; ++j)
				{
					int jj = numFeatures + j; // column in trainData to use
					sumSquaredError += ((outputs[j] - trainData[i][jj]) *
					  (outputs[j] - trainData[i][jj]));
				}
			}
			return sumSquaredError / trainData.Length;
		}

		public double Accuracy(double[][] trainData)
		{
			// using curr wts and biases
			int numCorrect = 0;
			int numWrong = 0;

			for (int i = 0; i < trainData.Length; ++i)
			{
				int[] deps = ComputeDependents(trainData[i]); // ex: [0  1  0]
				double[] targets = new double[numClasses]; // ex: [0.0  1.0  0.0]
				for (int j = 0; j < numClasses; ++j)
					targets[j] = trainData[i][numFeatures + j];

				int di = MaxIndex(deps);
				int ti = MaxIndex(targets);
				if (di == ti)
					++numCorrect;
				else
					++numWrong;
			}

			return (numCorrect * 1.0) / (numWrong + numCorrect);
		}

		private static int MaxIndex(double[] vector)
		{
			int maxIndex = 0;
			double maxVal = vector[0];
			for (int i = 0; i < vector.Length; ++i)
			{
				if (vector[i] > maxVal)
				{
					maxVal = vector[i];
					maxIndex = i;
				}
			}
			return maxIndex;
		}

		private static int MaxIndex(int[] vector)
		{
			int maxIndex = 0;
			int maxVal = vector[0];
			for (int i = 0; i < vector.Length; ++i)
			{
				if (vector[i] > maxVal)
				{
					maxVal = vector[i];
					maxIndex = i;
				}
			}
			return maxIndex;
		}

		private int[] ComputeDependents(double[] dataItem)
		{
			double[] outputs = ComputeOutputs(dataItem); // 0.0 to 1.0
			int maxIndex = MaxIndex(outputs);
			int[] result = new int[numClasses]; // [0 0 .. 0]
			result[maxIndex] = 1;
			return result;
		}

	}

}
