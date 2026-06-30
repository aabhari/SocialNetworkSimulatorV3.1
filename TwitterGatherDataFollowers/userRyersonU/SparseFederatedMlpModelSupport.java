package TwitterGatherDataFollowers.userRyersonU;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Sparse, serializable MLP used only by the opt-in federated MLP engine.
 *
 * The legacy Neuroph MLP remains the default path. This class exists so large
 * TF-IDF corpora can be trained without materializing dense input vectors.
 */
final class SparseFederatedMlpModelSupport
{
	private SparseFederatedMlpModelSupport()
	{
	}

	static final class TrainingExample implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private final String user;
		private final Map<Integer,Double> features;
		private final int labelIndex;

		TrainingExample(String user, Map<Integer,Double> features, int labelIndex)
		{
			this.user = user;
			this.features = new LinkedHashMap<Integer,Double>(features);
			this.labelIndex = labelIndex;
		}

		String getUser()
		{
			return user;
		}

		Map<Integer,Double> getFeatures()
		{
			return features;
		}

		int getLabelIndex()
		{
			return labelIndex;
		}
	}

	static final class SparseMlpModel implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private final int inputSize;
		private final int hiddenLayerCount;
		private final int hiddenSize;
		private final int outputSize;
		private final double[][][] weights;
		private final double[][] biases;
		private int sampleCount;

		private SparseMlpModel(
				int inputSize,
				int hiddenLayerCount,
				int hiddenSize,
				int outputSize,
				double[][][] weights,
				double[][] biases)
		{
			this.inputSize = inputSize;
			this.hiddenLayerCount = hiddenLayerCount;
			this.hiddenSize = hiddenSize;
			this.outputSize = outputSize;
			this.weights = weights;
			this.biases = biases;
		}

		static SparseMlpModel create(
				int inputSize,
				int hiddenLayerCount,
				int hiddenSize,
				int outputSize,
				long seed)
		{
			int safeInputSize = Math.max(1, inputSize);
			int safeHiddenLayerCount = Math.max(1, hiddenLayerCount);
			int safeHiddenSize = Math.max(1, hiddenSize);
			int safeOutputSize = Math.max(1, outputSize);
			int layerCount = safeHiddenLayerCount + 1;
			double[][][] weights = new double[layerCount][][];
			double[][] biases = new double[layerCount][];
			Random random = new Random(seed);
			int previousSize = safeInputSize;
			for (int layer = 0; layer < layerCount; layer++)
			{
				int currentSize = layer == layerCount - 1
						? safeOutputSize : safeHiddenSize;
				weights[layer] = new double[previousSize][currentSize];
				biases[layer] = new double[currentSize];
				double limit = Math.sqrt(6.0 / (previousSize + currentSize));
				for (int source = 0; source < previousSize; source++)
				{
					for (int target = 0; target < currentSize; target++)
					{
						weights[layer][source][target] =
								(random.nextDouble() * 2.0 - 1.0) * limit;
					}
				}
				previousSize = currentSize;
			}
			return new SparseMlpModel(
					safeInputSize,
					safeHiddenLayerCount,
					safeHiddenSize,
					safeOutputSize,
					weights,
					biases);
		}

		void train(
				List<TrainingExample> examples,
				int epochs,
				double learningRate,
				double l2,
				double fedProxMu)
		{
			if (examples == null || examples.isEmpty())
			{
				throw new IllegalArgumentException(
						"Sparse MLP training requires at least one example.");
			}
			int safeEpochs = Math.max(1, epochs);
			double safeLearningRate = Math.max(0.0000001, learningRate);
			double safeL2 = Math.max(0.0, l2);
			double safeFedProxMu = Math.max(0.0, fedProxMu);
			double[][][] referenceWeights = deepCopy(weights);
			double[][] referenceBiases = copyBiases(biases);
			for (int epoch = 0; epoch < safeEpochs; epoch++)
			{
				for (TrainingExample example : examples)
				{
					trainOne(example, safeLearningRate, safeL2,
							safeFedProxMu, referenceWeights, referenceBiases);
				}
			}
			sampleCount = examples.size();
		}

		double[] predict(Map<Integer,Double> sparseFeatures)
		{
			double[][] activations = forward(sparseFeatures);
			return activations[activations.length - 1];
		}

		int getSampleCount()
		{
			return sampleCount;
		}

		void save(File file) throws IOException
		{
			File parent = file.getParentFile();
			if (parent != null && !parent.exists())
			{
				parent.mkdirs();
			}
			ObjectOutputStream output = new ObjectOutputStream(
					new BufferedOutputStream(new FileOutputStream(file)));
			try
			{
				output.writeObject(this);
			}
			finally
			{
				output.close();
			}
		}

		static SparseMlpModel load(File file)
				throws IOException, ClassNotFoundException
		{
			ObjectInputStream input = new ObjectInputStream(
					new BufferedInputStream(new FileInputStream(file)));
			try
			{
				Object object = input.readObject();
				if (!(object instanceof SparseMlpModel))
				{
					throw new IOException("Not a sparse MLP model: " + file);
				}
				return (SparseMlpModel)object;
			}
			finally
			{
				input.close();
			}
		}

		static SparseMlpModel average(List<SparseMlpModel> models)
		{
			if (models == null || models.isEmpty())
			{
				throw new IllegalArgumentException("No sparse MLP models to average.");
			}
			SparseMlpModel first = models.get(0);
			double[][][] averagedWeights = zerosLike(first.weights);
			double[][] averagedBiases = zerosLike(first.biases);
			int totalWeight = 0;
			for (SparseMlpModel model : models)
			{
				first.requireCompatible(model);
				int modelWeight = Math.max(1, model.sampleCount);
				totalWeight += modelWeight;
				addWeighted(averagedWeights, model.weights, modelWeight);
				addWeighted(averagedBiases, model.biases, modelWeight);
			}
			scale(averagedWeights, 1.0 / Math.max(1, totalWeight));
			scale(averagedBiases, 1.0 / Math.max(1, totalWeight));
			SparseMlpModel averaged = new SparseMlpModel(
					first.inputSize,
					first.hiddenLayerCount,
					first.hiddenSize,
					first.outputSize,
					averagedWeights,
					averagedBiases);
			averaged.sampleCount = totalWeight;
			return averaged;
		}

		private void trainOne(
				TrainingExample example,
				double learningRate,
				double l2,
				double fedProxMu,
				double[][][] referenceWeights,
				double[][] referenceBiases)
		{
			if (example.getLabelIndex() < 0
					|| example.getLabelIndex() >= outputSize)
			{
				return;
			}
			double[][] activations = forward(example.getFeatures());
			double[][] deltas = computeDeltas(activations, example.getLabelIndex());
			updateFirstLayer(
					example.getFeatures(),
					deltas[0],
					learningRate,
					l2,
					fedProxMu,
					referenceWeights[0]);
			for (int layer = 1; layer < weights.length; layer++)
			{
				updateDenseLayer(
						layer,
						activations[layer - 1],
						deltas[layer],
						learningRate,
						l2,
						fedProxMu,
						referenceWeights[layer]);
			}
			for (int layer = 0; layer < biases.length; layer++)
			{
				for (int index = 0; index < biases[layer].length; index++)
				{
					double proximal =
							fedProxMu * (biases[layer][index]
									- referenceBiases[layer][index]);
					biases[layer][index] -= learningRate
							* (deltas[layer][index] + proximal);
				}
			}
		}

		private double[][] forward(Map<Integer,Double> sparseFeatures)
		{
			double[][] activations = new double[weights.length][];
			activations[0] = new double[hiddenSize];
			for (int target = 0; target < hiddenSize; target++)
			{
				double sum = biases[0][target];
				for (Map.Entry<Integer,Double> entry : sparseFeatures.entrySet())
				{
					int source = entry.getKey();
					if (source >= 0 && source < inputSize)
					{
						sum += entry.getValue() * weights[0][source][target];
					}
				}
				activations[0][target] = Math.tanh(sum);
			}
			for (int layer = 1; layer < weights.length - 1; layer++)
			{
				activations[layer] = denseTanh(
						activations[layer - 1],
						weights[layer],
						biases[layer]);
			}
			int outputLayer = weights.length - 1;
			activations[outputLayer] = softmax(
					denseLinear(
							activations[outputLayer - 1],
							weights[outputLayer],
							biases[outputLayer]));
			return activations;
		}

		private double[][] computeDeltas(double[][] activations, int labelIndex)
		{
			double[][] deltas = new double[weights.length][];
			int outputLayer = weights.length - 1;
			deltas[outputLayer] = Arrays.copyOf(
					activations[outputLayer],
					activations[outputLayer].length);
			deltas[outputLayer][labelIndex] -= 1.0;
			for (int layer = outputLayer - 1; layer >= 0; layer--)
			{
				deltas[layer] = new double[activations[layer].length];
				for (int source = 0; source < activations[layer].length; source++)
				{
					double downstream = 0.0;
					for (int target = 0; target < deltas[layer + 1].length; target++)
					{
						downstream += weights[layer + 1][source][target]
								* deltas[layer + 1][target];
					}
					double activation = activations[layer][source];
					deltas[layer][source] = downstream
							* (1.0 - activation * activation);
				}
			}
			return deltas;
		}

		private void updateFirstLayer(
				Map<Integer,Double> sparseFeatures,
				double[] delta,
				double learningRate,
				double l2,
				double fedProxMu,
				double[][] referenceWeights)
		{
			for (Map.Entry<Integer,Double> entry : sparseFeatures.entrySet())
			{
				int source = entry.getKey();
				if (source < 0 || source >= inputSize)
				{
					continue;
				}
				double value = entry.getValue();
				for (int target = 0; target < delta.length; target++)
				{
					double current = weights[0][source][target];
					double gradient = value * delta[target]
							+ l2 * current
							+ fedProxMu * (current - referenceWeights[source][target]);
					weights[0][source][target] -= learningRate * gradient;
				}
			}
		}

		private void updateDenseLayer(
				int layer,
				double[] previousActivation,
				double[] delta,
				double learningRate,
				double l2,
				double fedProxMu,
				double[][] referenceWeights)
		{
			for (int source = 0; source < previousActivation.length; source++)
			{
				double activation = previousActivation[source];
				for (int target = 0; target < delta.length; target++)
				{
					double current = weights[layer][source][target];
					double gradient = activation * delta[target]
							+ l2 * current
							+ fedProxMu * (current - referenceWeights[source][target]);
					weights[layer][source][target] -= learningRate * gradient;
				}
			}
		}

		private void requireCompatible(SparseMlpModel other)
		{
			if (other.inputSize != inputSize
					|| other.hiddenLayerCount != hiddenLayerCount
					|| other.hiddenSize != hiddenSize
					|| other.outputSize != outputSize)
			{
				throw new IllegalArgumentException(
						"Sparse MLP model shapes are not compatible.");
			}
		}
	}

	static void saveAveragedModel(List<String> modelPaths, String outputPath)
			throws IOException, ClassNotFoundException
	{
		java.util.ArrayList<SparseMlpModel> models =
				new java.util.ArrayList<SparseMlpModel>();
		for (String modelPath : modelPaths)
		{
			models.add(SparseMlpModel.load(new File(modelPath)));
		}
		SparseMlpModel.average(models).save(new File(outputPath));
	}

	private static double[] denseLinear(
			double[] input,
			double[][] layerWeights,
			double[] layerBiases)
	{
		double[] output = Arrays.copyOf(layerBiases, layerBiases.length);
		for (int source = 0; source < input.length; source++)
		{
			double value = input[source];
			for (int target = 0; target < output.length; target++)
			{
				output[target] += value * layerWeights[source][target];
			}
		}
		return output;
	}

	private static double[] denseTanh(
			double[] input,
			double[][] layerWeights,
			double[] layerBiases)
	{
		double[] linear = denseLinear(input, layerWeights, layerBiases);
		for (int index = 0; index < linear.length; index++)
		{
			linear[index] = Math.tanh(linear[index]);
		}
		return linear;
	}

	private static double[] softmax(double[] logits)
	{
		double max = logits[0];
		for (int index = 1; index < logits.length; index++)
		{
			max = Math.max(max, logits[index]);
		}
		double sum = 0.0;
		double[] probabilities = new double[logits.length];
		for (int index = 0; index < logits.length; index++)
		{
			probabilities[index] = Math.exp(logits[index] - max);
			sum += probabilities[index];
		}
		if (sum <= 0.0 || !Double.isFinite(sum))
		{
			Arrays.fill(probabilities, 1.0 / probabilities.length);
			return probabilities;
		}
		for (int index = 0; index < probabilities.length; index++)
		{
			probabilities[index] /= sum;
		}
		return probabilities;
	}

	private static double[][][] deepCopy(double[][][] source)
	{
		double[][][] copy = new double[source.length][][];
		for (int layer = 0; layer < source.length; layer++)
		{
			copy[layer] = new double[source[layer].length][];
			for (int row = 0; row < source[layer].length; row++)
			{
				copy[layer][row] = Arrays.copyOf(
						source[layer][row], source[layer][row].length);
			}
		}
		return copy;
	}

	private static double[][] copyBiases(double[][] source)
	{
		double[][] copy = new double[source.length][];
		for (int layer = 0; layer < source.length; layer++)
		{
			copy[layer] = Arrays.copyOf(source[layer], source[layer].length);
		}
		return copy;
	}

	private static double[][][] zerosLike(double[][][] source)
	{
		double[][][] zero = new double[source.length][][];
		for (int layer = 0; layer < source.length; layer++)
		{
			zero[layer] = new double[source[layer].length][];
			for (int row = 0; row < source[layer].length; row++)
			{
				zero[layer][row] = new double[source[layer][row].length];
			}
		}
		return zero;
	}

	private static double[][] zerosLike(double[][] source)
	{
		double[][] zero = new double[source.length][];
		for (int row = 0; row < source.length; row++)
		{
			zero[row] = new double[source[row].length];
		}
		return zero;
	}

	private static void addWeighted(
			double[][][] destination,
			double[][][] source,
			double weight)
	{
		for (int layer = 0; layer < destination.length; layer++)
		{
			for (int row = 0; row < destination[layer].length; row++)
			{
				for (int column = 0; column < destination[layer][row].length; column++)
				{
					destination[layer][row][column] +=
							source[layer][row][column] * weight;
				}
			}
		}
	}

	private static void addWeighted(
			double[][] destination,
			double[][] source,
			double weight)
	{
		for (int row = 0; row < destination.length; row++)
		{
			for (int column = 0; column < destination[row].length; column++)
			{
				destination[row][column] += source[row][column] * weight;
			}
		}
	}

	private static void scale(double[][][] values, double multiplier)
	{
		for (int layer = 0; layer < values.length; layer++)
		{
			for (int row = 0; row < values[layer].length; row++)
			{
				for (int column = 0; column < values[layer][row].length; column++)
				{
					values[layer][row][column] *= multiplier;
				}
			}
		}
	}

	private static void scale(double[][] values, double multiplier)
	{
		for (int row = 0; row < values.length; row++)
		{
			for (int column = 0; column < values[row].length; column++)
			{
				values[row][column] *= multiplier;
			}
		}
	}
}
