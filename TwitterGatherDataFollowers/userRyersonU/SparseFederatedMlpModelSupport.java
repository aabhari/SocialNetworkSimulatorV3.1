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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
	static final long GLOBAL_INITIALIZATION_SEED = 20260708L;
	static final String FEDAVG_UPDATE_ONTOLOGY = "Sparse MLP FedAvg Update";
	static final String FEDAVG_ROUND_COMPLETE_ONTOLOGY = "Sparse MLP FedAvg Round Complete";
	static final String FEDAVG_COMPLETE_ONTOLOGY = "Averaged Sparse MLP Complete";
	static final String FEDAVG_FAILED_ONTOLOGY = "Averaged Sparse MLP Failed";
	static final String EVALUATION_UPDATE_ONTOLOGY = "Sparse MLP Evaluation Update";
	static final String EVALUATION_COMPLETE_ONTOLOGY = "Sparse MLP Evaluation Complete";
	static final String EVALUATION_FAILED_ONTOLOGY = "Sparse MLP Evaluation Failed";

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

	static final class FedAvgModelUpdate implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private final String modelPath;
		private final int round;
		private final int totalRounds;

		FedAvgModelUpdate(String modelPath, int round, int totalRounds)
		{
			this.modelPath = modelPath;
			this.round = round;
			this.totalRounds = totalRounds;
		}

		String getModelPath()
		{
			return modelPath;
		}

		int getRound()
		{
			return round;
		}

		int getTotalRounds()
		{
			return totalRounds;
		}
	}

	static final class FedAvgRoundModel implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private final String modelPath;
		private final int round;
		private final int totalRounds;

		FedAvgRoundModel(String modelPath, int round, int totalRounds)
		{
			this.modelPath = modelPath;
			this.round = round;
			this.totalRounds = totalRounds;
		}

		String getModelPath()
		{
			return modelPath;
		}

		int getRound()
		{
			return round;
		}

		int getTotalRounds()
		{
			return totalRounds;
		}
	}

	static final class Prediction implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private final String user;
		private final int actualIndex;
		private final int predictedIndex;

		Prediction(String user, int actualIndex, int predictedIndex)
		{
			this.user = user;
			this.actualIndex = actualIndex;
			this.predictedIndex = predictedIndex;
		}

		String getUser()
		{
			return user;
		}

		int getActualIndex()
		{
			return actualIndex;
		}

		int getPredictedIndex()
		{
			return predictedIndex;
		}

		boolean isCorrect()
		{
			return actualIndex >= 0 && actualIndex == predictedIndex;
		}
	}

	static final class EvaluationResult implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private final String sourceAgent;
		private final List<Prediction> predictions;

		EvaluationResult(String sourceAgent, List<Prediction> predictions)
		{
			this.sourceAgent = sourceAgent;
			this.predictions = new ArrayList<Prediction>(
					predictions == null ? Collections.<Prediction>emptyList()
							: predictions);
		}

		String getSourceAgent()
		{
			return sourceAgent;
		}

		List<Prediction> getPredictions()
		{
			return predictions;
		}
	}

	static final class AggregatedEvaluationResult implements Serializable
	{
		private static final long serialVersionUID = 1L;

		private final int correctCount;
		private final int totalInstances;
		private final int duplicateInstances;
		private final int[] predictedCounts;
		private final int[] actualCounts;

		AggregatedEvaluationResult(
				int correctCount,
				int totalInstances,
				int duplicateInstances,
				int[] predictedCounts,
				int[] actualCounts)
		{
			this.correctCount = correctCount;
			this.totalInstances = totalInstances;
			this.duplicateInstances = duplicateInstances;
			this.predictedCounts = predictedCounts == null
					? new int[0]
					: Arrays.copyOf(predictedCounts, predictedCounts.length);
			this.actualCounts = actualCounts == null
					? new int[0]
					: Arrays.copyOf(actualCounts, actualCounts.length);
		}

		int getCorrectCount()
		{
			return correctCount;
		}

		int getTotalInstances()
		{
			return totalInstances;
		}

		int getDuplicateInstances()
		{
			return duplicateInstances;
		}

		int[] getPredictedCounts()
		{
			return Arrays.copyOf(predictedCounts, predictedCounts.length);
		}

		int[] getActualCounts()
		{
			return Arrays.copyOf(actualCounts, actualCounts.length);
		}

		double getAccuracyPercent()
		{
			return totalInstances <= 0
					? 0.0 : (100.0 * correctCount) / totalInstances;
		}
	}

	static final class SparseMlpModel implements Serializable
	{
		private static final long serialVersionUID = 1L;
		private static final int DEEP_STABLE_LAYER_THRESHOLD = 4;
		private static final double DEEP_STABLE_LEAKY_RELU_SLOPE = 0.01;
		private static final double DEEP_STABLE_MAX_LEARNING_RATE = 0.03;

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
					double limit = initializationLimit(
							previousSize,
							currentSize,
							layer < layerCount - 1,
							safeHiddenLayerCount);
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
			double safeLearningRate = effectiveLearningRate(learningRate);
			double safeL2 = Math.max(0.0, l2);
			double safeFedProxMu = Math.max(0.0, fedProxMu);
			double[][][] referenceWeights = deepCopy(weights);
			double[][] referenceBiases = copyBiases(biases);
			List<TrainingExample> shuffledExamples =
					new ArrayList<TrainingExample>(examples);
			Random shuffleRandom = new Random(89173L
					+ (long)hiddenLayerCount * 1009L
					+ (long)hiddenSize * 9176L
					+ (long)outputSize * 65537L);
			for (int epoch = 0; epoch < safeEpochs; epoch++)
			{
				if (shuffledExamples.size() > 1)
				{
					Collections.shuffle(shuffledExamples, shuffleRandom);
				}
				for (TrainingExample example : shuffledExamples)
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

		int getHiddenLayerCount()
		{
			return hiddenLayerCount;
		}

		double effectiveLearningRate(double requestedLearningRate)
		{
			return effectiveLearningRate(hiddenLayerCount, requestedLearningRate);
		}

		boolean usesDeepStableMode()
		{
			return usesDeepStableMode(hiddenLayerCount);
		}

		String trainingModeLabel()
		{
			return trainingModeLabel(hiddenLayerCount);
		}

		String hiddenActivationLabel()
		{
			return hiddenActivationLabel(hiddenLayerCount);
		}

		String initializationLabel()
		{
			return initializationLabel(hiddenLayerCount);
		}

		static boolean usesDeepStableMode(int hiddenLayerCount)
		{
			return hiddenLayerCount >= DEEP_STABLE_LAYER_THRESHOLD;
		}

		static double effectiveLearningRate(
				int hiddenLayerCount,
				double requestedLearningRate)
		{
			double safeLearningRate = Math.max(0.0000001, requestedLearningRate);
			if (usesDeepStableMode(hiddenLayerCount))
			{
				return Math.min(safeLearningRate, DEEP_STABLE_MAX_LEARNING_RATE);
			}
			return safeLearningRate;
		}

		static String trainingModeLabel(int hiddenLayerCount)
		{
			return usesDeepStableMode(hiddenLayerCount)
					? "deep-stable" : "standard-tanh";
		}

		static String hiddenActivationLabel(int hiddenLayerCount)
		{
			return usesDeepStableMode(hiddenLayerCount)
					? "leakyReLU(" + DEEP_STABLE_LEAKY_RELU_SLOPE + ")"
					: "tanh";
		}

		static String initializationLabel(int hiddenLayerCount)
		{
			return usesDeepStableMode(hiddenLayerCount)
					? "heUniformFanIn"
					: "xavierUniform";
		}

			int getWeightLayerCount()
			{
				return weights.length;
			}

			SparseMlpModel copy()
			{
				SparseMlpModel copy = new SparseMlpModel(
						inputSize,
						hiddenLayerCount,
						hiddenSize,
						outputSize,
						deepCopy(weights),
						copyBiases(biases));
				copy.sampleCount = sampleCount;
				return copy;
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
				activations[0][target] = hiddenActivation(sum);
			}
			for (int layer = 1; layer < weights.length - 1; layer++)
			{
				activations[layer] = denseHidden(
						activations[layer - 1],
						weights[layer],
						biases[layer],
						usesDeepStableMode());
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
					deltas[layer][source] = downstream
							* hiddenDerivativeFromActivation(activations[layer][source]);
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

		private double hiddenActivation(double value)
		{
			if (usesDeepStableMode())
			{
				return value >= 0.0 ? value : DEEP_STABLE_LEAKY_RELU_SLOPE * value;
			}
			return Math.tanh(value);
		}

		private double hiddenDerivativeFromActivation(double activation)
		{
			if (usesDeepStableMode())
			{
				return activation >= 0.0 ? 1.0 : DEEP_STABLE_LEAKY_RELU_SLOPE;
			}
			return 1.0 - activation * activation;
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

	static AggregatedEvaluationResult aggregateEvaluationResults(
			List<EvaluationResult> results,
			int outputCount)
	{
		Map<String,Prediction> uniquePredictions =
				new LinkedHashMap<String,Prediction>();
		int duplicateInstances = 0;
		if (results != null)
		{
			for (EvaluationResult result : results)
			{
				if (result == null || result.getPredictions() == null)
				{
					continue;
				}
				for (Prediction prediction : result.getPredictions())
				{
					if (prediction == null || prediction.getUser() == null)
					{
						continue;
					}
					if (uniquePredictions.containsKey(prediction.getUser()))
					{
						duplicateInstances++;
						continue;
					}
					uniquePredictions.put(prediction.getUser(), prediction);
				}
			}
		}

		int safeOutputCount = Math.max(1, outputCount);
		int correctCount = 0;
		int[] predictedCounts = new int[safeOutputCount];
		int[] actualCounts = new int[safeOutputCount];
		for (Prediction prediction : uniquePredictions.values())
		{
			if (prediction.isCorrect())
			{
				correctCount++;
			}
			int predicted = prediction.getPredictedIndex();
			if (predicted >= 0 && predicted < predictedCounts.length)
			{
				predictedCounts[predicted]++;
			}
			int actual = prediction.getActualIndex();
			if (actual >= 0 && actual < actualCounts.length)
			{
				actualCounts[actual]++;
			}
		}
		return new AggregatedEvaluationResult(
				correctCount,
				uniquePredictions.size(),
				duplicateInstances,
				predictedCounts,
				actualCounts);
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

	private static double initializationLimit(
			int fanIn,
			int fanOut,
			boolean hiddenLayer,
			int hiddenLayerCount)
	{
		if (hiddenLayer
				&& hiddenLayerCount >= SparseMlpModel.DEEP_STABLE_LAYER_THRESHOLD)
		{
			double slope = SparseMlpModel.DEEP_STABLE_LEAKY_RELU_SLOPE;
			return Math.sqrt(6.0 / ((1.0 + slope * slope) * Math.max(1, fanIn)));
		}
		return Math.sqrt(6.0 / (Math.max(1, fanIn) + Math.max(1, fanOut)));
	}

	private static double[] denseHidden(
			double[] input,
			double[][] layerWeights,
			double[] layerBiases,
			boolean deepStableActivation)
	{
		double[] linear = denseLinear(input, layerWeights, layerBiases);
		for (int index = 0; index < linear.length; index++)
		{
			if (deepStableActivation)
			{
				linear[index] = linear[index] >= 0.0
						? linear[index]
						: SparseMlpModel.DEEP_STABLE_LEAKY_RELU_SLOPE
								* linear[index];
			}
			else
			{
				linear[index] = Math.tanh(linear[index]);
			}
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
