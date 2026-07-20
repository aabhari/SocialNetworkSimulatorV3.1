package TwitterGatherDataFollowers.userRyersonU;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.neuroph.contrib.learning.SoftMax;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

/**
 * MLP-only hyperparameter search for DSMP six-column text files.
 *
 * The search is intentionally independent from JADE so it can evaluate many MLP
 * configurations quickly, then return one bounded setting object to the normal
 * simulator workflow.
 */
final class MlpHyperparameterSearch
{
	private static final Pattern PUNCTUATION = Pattern.compile("[\\p{P}]");
	private static final Pattern URL_LIKE =
			Pattern.compile("http[a-zA-Z0-9]*|bitly[a-zA-Z0-9]*|www[a-zA-Z0-9]*");
	private static final Pattern NON_LETTER_SPACE =
			Pattern.compile("[^a-zA-Z\\p{Z}]");

	private MlpHyperparameterSearch()
	{
	}

	interface ProgressListener
	{
		void onMessage(String message);
		void onDatasetLoaded(DataProfile profile);
		void onCandidateStarted(Candidate candidate, int completed, int total);
		void onCandidateFinished(SearchResult result, int completed, int total);
		boolean isCancelled();
	}

	static SearchReport run(SearchConfig config, ProgressListener listener)
			throws Exception
	{
		SearchConfig safeConfig = config.copy();
		safeConfig.validate();
		long started = System.nanoTime();
		notifyMessage(listener, "Loading and vectorizing DSMP dataset...");
		DataSplit split = loadDsmp(safeConfig.datasetFile,
				safeConfig.validationPercent,
				safeConfig.testPercent,
				safeConfig.splitSeed);
		if (listener != null)
		{
			listener.onDatasetLoaded(split.profile);
		}
		List<Candidate> candidates = safeConfig.candidates();
		List<SearchResult> results = new ArrayList<SearchResult>();
		for (int i = 0; i < candidates.size(); i++)
		{
			if (listener != null && listener.isCancelled())
			{
				notifyMessage(listener, "Search cancelled after "
						+ results.size() + " completed run(s).");
				break;
			}
			Candidate candidate = candidates.get(i);
			if (listener != null)
			{
				listener.onCandidateStarted(candidate, i, candidates.size());
			}
			SearchResult result;
			try
			{
				result = runCandidate(split, candidate);
			}
			catch (RuntimeException ex)
			{
				result = SearchResult.failed(candidate, split.profile,
						ex.getClass().getSimpleName() + ": " + ex.getMessage());
			}
			results.add(result);
			if (listener != null)
			{
				listener.onCandidateFinished(result, i + 1, candidates.size());
			}
		}
		Collections.sort(results, SearchResult.BEST_FIRST);
		return new SearchReport(split.profile, results,
				(System.nanoTime() - started) / 1000000L);
	}

	static void exportCsv(File file, SearchReport report) throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(file, false));
		try
		{
			writer.write(SearchResult.csvHeader());
			writer.newLine();
			for (SearchResult result : report.results)
			{
				writer.write(result.toCsvRow());
				writer.newLine();
			}
		}
		finally
		{
			writer.close();
		}
	}

	private static SearchResult runCandidate(DataSplit split, Candidate candidate)
	{
		long trainStart = System.nanoTime();
		Metrics trainMetrics;
		Metrics validationMetrics;
		Metrics testMetrics;
		if (AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED.equals(
				candidate.engine))
		{
			SparseFederatedMlpModelSupport.SparseMlpModel model =
					SparseFederatedMlpModelSupport.SparseMlpModel.create(
							split.inputSize,
							candidate.hiddenLayers,
							candidate.hiddenNeurons,
							split.outputSize,
							candidate.seed);
			model.train(split.train, candidate.sparseEpochs,
					candidate.learningRate, candidate.sparseL2,
					candidate.fedProxMu);
			long trainEnd = System.nanoTime();
			trainMetrics = evaluateSparse(model, split.train, split.classNames);
			validationMetrics = evaluateSparse(model, split.validation,
					split.classNames);
			testMetrics = evaluateSparse(model, split.test, split.classNames);
			long evalEnd = System.nanoTime();
			return SearchResult.completed(candidate, split.profile,
					trainMetrics, validationMetrics, testMetrics,
					(trainEnd - trainStart) / 1000000L,
					(evalEnd - trainEnd) / 1000000L,
					estimatedModelBytes(split.inputSize, candidate.hiddenLayers,
							candidate.hiddenNeurons, split.outputSize),
					model.trainingModeLabel());
		}

		MultiLayerPerceptron model = createLegacyNeurophMlp(
				split.inputSize,
				candidate.hiddenNeurons,
				candidate.hiddenLayers,
				split.outputSize,
				candidate.learningRate,
				candidate.maxError,
				candidate.maxIterations);
		model.learn(toNeurophDataSet(split.train, split.inputSize,
				split.outputSize));
		long trainEnd = System.nanoTime();
		trainMetrics = evaluateLegacy(model, split.train, split.inputSize,
				split.classNames);
		validationMetrics = evaluateLegacy(model, split.validation,
				split.inputSize, split.classNames);
		testMetrics = evaluateLegacy(model, split.test, split.inputSize,
				split.classNames);
		long evalEnd = System.nanoTime();
		return SearchResult.completed(candidate, split.profile, trainMetrics,
				validationMetrics, testMetrics,
				(trainEnd - trainStart) / 1000000L,
				(evalEnd - trainEnd) / 1000000L,
				estimatedModelBytes(split.inputSize, candidate.hiddenLayers,
						candidate.hiddenNeurons, split.outputSize),
				"bounded-Neuroph");
	}

	private static MultiLayerPerceptron createLegacyNeurophMlp(
			int inputCount,
			int hiddenNeurons,
			int hiddenLayers,
			int outputCount,
			double learningRate,
			double maxError,
			int maxIterations)
	{
		int safeHiddenLayers = Math.max(1, hiddenLayers);
		int safeHiddenNeurons = Math.max(1, hiddenNeurons);
		MultiLayerPerceptron mlp;
		if (safeHiddenLayers == 1)
		{
			mlp = new MultiLayerPerceptron(TransferFunctionType.TANH,
					inputCount, safeHiddenNeurons, outputCount);
		}
		else
		{
			int[] layers = new int[safeHiddenLayers + 2];
			layers[0] = inputCount;
			for (int i = 1; i <= safeHiddenLayers; i++)
			{
				layers[i] = safeHiddenNeurons;
			}
			layers[layers.length - 1] = outputCount;
			mlp = new MultiLayerPerceptron(TransferFunctionType.TANH, layers);
		}
		Layer outputLayer = mlp.getLayers().get(mlp.getLayers().size() - 1);
		SoftMax softMax = new SoftMax(outputLayer);
		for (Neuron neuron : outputLayer.getNeurons())
		{
			neuron.setTransferFunction(softMax);
		}
		BackPropagation rule = (BackPropagation)mlp.getLearningRule();
		rule.setLearningRate(learningRate);
		rule.setMaxError(maxError);
		rule.setMaxIterations(Math.max(1, maxIterations));
		return mlp;
	}

	private static Metrics evaluateSparse(
			SparseFederatedMlpModelSupport.SparseMlpModel model,
			List<SparseFederatedMlpModelSupport.TrainingExample> examples,
			String[] classNames)
	{
		int[] predicted = new int[classNames.length];
		int[] actual = new int[classNames.length];
		int correct = 0;
		for (SparseFederatedMlpModelSupport.TrainingExample example : examples)
		{
			int prediction = indexOfMax(model.predict(example.getFeatures()));
			if (prediction == example.getLabelIndex())
			{
				correct++;
			}
			increment(predicted, prediction);
			increment(actual, example.getLabelIndex());
		}
		return new Metrics(correct, examples.size(), predicted, actual,
				classNames);
	}

	private static Metrics evaluateLegacy(
			MultiLayerPerceptron model,
			List<SparseFederatedMlpModelSupport.TrainingExample> examples,
			int inputSize,
			String[] classNames)
	{
		int[] predicted = new int[classNames.length];
		int[] actual = new int[classNames.length];
		int correct = 0;
		for (SparseFederatedMlpModelSupport.TrainingExample example : examples)
		{
			model.setInput(denseInput(example, inputSize));
			model.calculate();
			int prediction = indexOfMax(model.getOutput());
			if (prediction == example.getLabelIndex())
			{
				correct++;
			}
			increment(predicted, prediction);
			increment(actual, example.getLabelIndex());
		}
		return new Metrics(correct, examples.size(), predicted, actual,
				classNames);
	}

	private static DataSet toNeurophDataSet(
			List<SparseFederatedMlpModelSupport.TrainingExample> examples,
			int inputSize,
			int outputSize)
	{
		DataSet dataSet = new DataSet(inputSize, outputSize);
		for (SparseFederatedMlpModelSupport.TrainingExample example : examples)
		{
			dataSet.addRow(new DataSetRow(
					denseInput(example, inputSize),
					oneHot(example.getLabelIndex(), outputSize)));
		}
		return dataSet;
	}

	private static double[] denseInput(
			SparseFederatedMlpModelSupport.TrainingExample example,
			int inputSize)
	{
		double[] input = new double[inputSize];
		for (Map.Entry<Integer,Double> entry : example.getFeatures().entrySet())
		{
			int index = entry.getKey().intValue();
			if (index >= 0 && index < input.length)
			{
				input[index] = entry.getValue().doubleValue();
			}
		}
		return input;
	}

	private static double[] oneHot(int labelIndex, int outputSize)
	{
		double[] output = new double[outputSize];
		if (labelIndex >= 0 && labelIndex < output.length)
		{
			output[labelIndex] = 1.0;
		}
		return output;
	}

	private static DataSplit loadDsmp(
			File file,
			double validationPercent,
			double testPercent,
			long splitSeed)
			throws IOException
	{
		LinkedHashMap<String,LinkedHashMap<String,Double>> docs =
				new LinkedHashMap<String,LinkedHashMap<String,Double>>();
		LinkedHashMap<String,String> userFollowee =
				new LinkedHashMap<String,String>();
		TreeSet<String> terms = new TreeSet<String>();
		int acceptedRows = 0;
		int malformedRows = 0;
		int labelConflicts = 0;
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(file), StandardCharsets.ISO_8859_1),
				128 * 1024);
		try
		{
			String line;
			while ((line = reader.readLine()) != null)
			{
				String[] fields = line.split("\t", 6);
				if (fields.length < 6)
				{
					malformedRows++;
					continue;
				}
				String followee = fields[0].trim();
				String user = fields[4].trim();
				if (followee.length() == 0 || user.length() == 0)
				{
					malformedRows++;
					continue;
				}
				LinkedHashMap<String,Double> row = tokenize(fields[5]);
				if (row.size() < 3)
				{
					continue;
				}
				String oldFollowee = userFollowee.get(user);
				if (oldFollowee == null)
				{
					userFollowee.put(user, followee);
				}
				else if (!oldFollowee.equals(followee))
				{
					labelConflicts++;
					followee = oldFollowee;
				}
				LinkedHashMap<String,Double> doc = docs.get(user);
				if (doc == null)
				{
					doc = new LinkedHashMap<String,Double>();
					docs.put(user, doc);
				}
				for (Map.Entry<String,Double> entry : row.entrySet())
				{
					Double old = doc.get(entry.getKey());
					doc.put(entry.getKey(), Double.valueOf(
							(old == null ? 0.0 : old.doubleValue())
									+ entry.getValue().doubleValue()));
					terms.add(entry.getKey());
				}
				acceptedRows++;
			}
		}
		finally
		{
			reader.close();
		}
		if (docs.isEmpty())
		{
			throw new IllegalArgumentException(
					"No usable DSMP rows were found in " + file.getAbsolutePath());
		}
		normalizeTfidf(docs, terms);
		LinkedHashMap<String,Integer> termIndex =
				new LinkedHashMap<String,Integer>();
		for (String term : terms)
		{
			termIndex.put(term, Integer.valueOf(termIndex.size()));
		}
		LinkedHashMap<String,Integer> classIndex =
				buildClassIndex(userFollowee.values());
		String[] classNames = new String[classIndex.size()];
		for (Map.Entry<String,Integer> entry : classIndex.entrySet())
		{
			classNames[entry.getValue().intValue()] = entry.getKey();
		}
		Split split = splitUsersByClass(docs.keySet(), userFollowee,
				validationPercent, testPercent, splitSeed);
		List<SparseFederatedMlpModelSupport.TrainingExample> train =
				examplesFor(split.trainUsers, docs, userFollowee, classIndex,
						termIndex);
		List<SparseFederatedMlpModelSupport.TrainingExample> validation =
				examplesFor(split.validationUsers, docs, userFollowee, classIndex,
						termIndex);
		List<SparseFederatedMlpModelSupport.TrainingExample> test =
				examplesFor(split.testUsers, docs, userFollowee, classIndex,
						termIndex);
		if (train.isEmpty() || validation.isEmpty())
		{
			throw new IllegalArgumentException(
					"HPO requires non-empty train and validation splits.");
		}
		DataProfile profile = new DataProfile(
				file,
				acceptedRows,
				malformedRows,
				labelConflicts,
				docs.size(),
				termIndex.size(),
				classNames,
				counts(train, classNames.length),
				counts(validation, classNames.length),
				counts(test, classNames.length));
		return new DataSplit(termIndex.size(), classNames.length, classNames,
				train, validation, test, profile);
	}

	private static LinkedHashMap<String,Double> tokenize(String text)
	{
		String current = " " + text + " ";
		if (current.contains("Photo:"))
		{
			current = current.substring(0, current.indexOf("Photo:"));
		}
		if (current.contains("Photoset:"))
		{
			current = current.substring(0, current.indexOf("Photoset:"));
		}
		Matcher matcher = PUNCTUATION.matcher(current);
		current = matcher.replaceAll("");
		matcher = URL_LIKE.matcher(current);
		current = matcher.replaceAll(" ");
		matcher = NON_LETTER_SPACE.matcher(current);
		current = matcher.replaceAll(" ");
		current = current.toLowerCase().replaceAll("[^a-zA-Z ]", "")
				.replaceAll(" +", " ").trim();
		LinkedHashMap<String,Double> row =
				new LinkedHashMap<String,Double>();
		if (current.length() == 0)
		{
			return row;
		}
		String[] words = current.split("\\s+");
		if (words.length < 3)
		{
			return row;
		}
		for (String word : words)
		{
			Double old = row.get(word);
			row.put(word, Double.valueOf(
					(old == null ? 0.0 : old.doubleValue()) + 1.0));
		}
		return row;
	}

	private static void normalizeTfidf(
			LinkedHashMap<String,LinkedHashMap<String,Double>> docs,
			TreeSet<String> terms)
	{
		LinkedHashMap<String,Integer> df = new LinkedHashMap<String,Integer>();
		for (String term : terms)
		{
			df.put(term, Integer.valueOf(0));
		}
		for (Map<String,Double> doc : docs.values())
		{
			for (String term : doc.keySet())
			{
				df.put(term, Integer.valueOf(df.get(term).intValue() + 1));
			}
		}
		int totalDocs = docs.size();
		for (LinkedHashMap<String,Double> doc : docs.values())
		{
			double magnitude = 0.0;
			for (Map.Entry<String,Double> entry : doc.entrySet())
			{
				double value = entry.getValue().doubleValue()
						* log2Idf(totalDocs, df.get(entry.getKey()).intValue());
				entry.setValue(Double.valueOf(value));
				magnitude += value * value;
			}
			magnitude = Math.sqrt(magnitude);
			if (magnitude <= 0.0)
			{
				continue;
			}
			for (Map.Entry<String,Double> entry : doc.entrySet())
			{
				entry.setValue(Double.valueOf(
						entry.getValue().doubleValue() / magnitude));
			}
		}
	}

	private static double log2Idf(double docs, double df)
	{
		if (docs <= 0.0 || df <= 0.0)
		{
			return 0.0;
		}
		return Math.log(docs / df) / Math.log(2.0);
	}

	private static LinkedHashMap<String,Integer> buildClassIndex(
			Collection<String> labels)
	{
		LinkedHashMap<String,Integer> index =
				new LinkedHashMap<String,Integer>();
		for (String label : labels)
		{
			if (!index.containsKey(label))
			{
				index.put(label, Integer.valueOf(index.size()));
			}
		}
		return index;
	}

	private static Split splitUsersByClass(
			Collection<String> users,
			Map<String,String> userFollowee,
			double validationPercent,
			double testPercent,
			long splitSeed)
	{
		LinkedHashMap<String,List<String>> byClass =
				new LinkedHashMap<String,List<String>>();
		for (String user : users)
		{
			String label = userFollowee.get(user);
			List<String> rows = byClass.get(label);
			if (rows == null)
			{
				rows = new ArrayList<String>();
				byClass.put(label, rows);
			}
			rows.add(user);
		}
		List<String> train = new ArrayList<String>();
		List<String> validation = new ArrayList<String>();
		List<String> test = new ArrayList<String>();
		int classOrdinal = 0;
		for (List<String> rows : byClass.values())
		{
			Collections.shuffle(rows, new Random(splitSeed
					+ (long)classOrdinal * 1009L + rows.size()));
			int size = rows.size();
			int validationCount = size >= 2
					? Math.max(1, (int)Math.floor(size * validationPercent))
					: 0;
			int testCount = size >= 3
					? Math.max(1, (int)Math.floor(size * testPercent))
					: 0;
			while (validationCount + testCount > Math.max(0, size - 1))
			{
				if (validationCount >= testCount && validationCount > 0)
				{
					validationCount--;
				}
				else if (testCount > 0)
				{
					testCount--;
				}
				else
				{
					break;
				}
			}
			for (int i = 0; i < rows.size(); i++)
			{
				if (i < validationCount)
				{
					validation.add(rows.get(i));
				}
				else if (i < validationCount + testCount)
				{
					test.add(rows.get(i));
				}
				else
				{
					train.add(rows.get(i));
				}
			}
			classOrdinal++;
		}
		return new Split(train, validation, test);
	}

	private static List<SparseFederatedMlpModelSupport.TrainingExample>
			examplesFor(
					List<String> users,
					LinkedHashMap<String,LinkedHashMap<String,Double>> docs,
					LinkedHashMap<String,String> userFollowee,
					LinkedHashMap<String,Integer> classIndex,
					LinkedHashMap<String,Integer> termIndex)
	{
		List<SparseFederatedMlpModelSupport.TrainingExample> examples =
				new ArrayList<SparseFederatedMlpModelSupport.TrainingExample>();
		for (String user : users)
		{
			Map<Integer,Double> sparse = new LinkedHashMap<Integer,Double>();
			LinkedHashMap<String,Double> doc = docs.get(user);
			if (doc == null)
			{
				continue;
			}
			for (Map.Entry<String,Double> entry : doc.entrySet())
			{
				Integer index = termIndex.get(entry.getKey());
				if (index != null && entry.getValue().doubleValue() != 0.0)
				{
					sparse.put(index, entry.getValue());
				}
			}
			Integer label = classIndex.get(userFollowee.get(user));
			if (label != null)
			{
				examples.add(new SparseFederatedMlpModelSupport.TrainingExample(
						user, sparse, label.intValue()));
			}
		}
		return examples;
	}

	private static long estimatedModelBytes(
			int inputSize,
			int hiddenLayers,
			int hiddenNeurons,
			int outputSize)
	{
		long weights = (long)Math.max(1, inputSize) * Math.max(1, hiddenNeurons)
				+ (long)Math.max(0, hiddenLayers - 1)
						* Math.max(1, hiddenNeurons)
						* Math.max(1, hiddenNeurons)
				+ (long)Math.max(1, hiddenNeurons) * Math.max(1, outputSize);
		long biases = (long)Math.max(1, hiddenLayers) * Math.max(1, hiddenNeurons)
				+ Math.max(1, outputSize);
		return Math.max(1L, (weights + biases) * 8L);
	}

	private static int[] counts(
			List<SparseFederatedMlpModelSupport.TrainingExample> examples,
			int size)
	{
		int[] counts = new int[size];
		for (SparseFederatedMlpModelSupport.TrainingExample example : examples)
		{
			increment(counts, example.getLabelIndex());
		}
		return counts;
	}

	private static void increment(int[] counts, int index)
	{
		if (index >= 0 && index < counts.length)
		{
			counts[index]++;
		}
	}

	private static int indexOfMax(double[] values)
	{
		int maxIndex = 0;
		for (int i = 1; i < values.length; i++)
		{
			if (values[i] > values[maxIndex])
			{
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	private static int indexOfMax(int[] values)
	{
		int maxIndex = 0;
		for (int i = 1; i < values.length; i++)
		{
			if (values[i] > values[maxIndex])
			{
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	private static int countNonZero(int[] values)
	{
		int count = 0;
		for (int value : values)
		{
			if (value > 0)
			{
				count++;
			}
		}
		return count;
	}

	private static int sum(int[] values)
	{
		int total = 0;
		for (int value : values)
		{
			total += value;
		}
		return total;
	}

	private static void notifyMessage(ProgressListener listener, String message)
	{
		if (listener != null)
		{
			listener.onMessage(message);
		}
	}

	static final class SearchConfig
	{
		File datasetFile;
		boolean includeLegacy = true;
		boolean includeSparse = true;
		int[] hiddenLayers = new int[] {1, 3, 5};
		int[] hiddenNeurons = new int[] {10, 25};
		double[] learningRates = new double[] {0.1, 0.03, 0.01};
		double[] maxErrors = new double[] {0.01};
		int[] legacyMaxIterations = new int[] {50};
		int[] sparseEpochs = new int[] {80, 160};
		int[] fedAvgRounds = new int[] {16};
		double[] sparseL2Values = new double[] {0.0001};
		double[] fedProxMuValues = new double[] {0.0};
		long[] seeds = new long[] {
				SparseFederatedMlpModelSupport.GLOBAL_INITIALIZATION_SEED};
		double validationPercent = 0.20;
		double testPercent = 0.20;
		long splitSeed = 20260719L;

		SearchConfig copy()
		{
			SearchConfig copy = new SearchConfig();
			copy.datasetFile = datasetFile;
			copy.includeLegacy = includeLegacy;
			copy.includeSparse = includeSparse;
			copy.hiddenLayers = Arrays.copyOf(hiddenLayers, hiddenLayers.length);
			copy.hiddenNeurons = Arrays.copyOf(hiddenNeurons, hiddenNeurons.length);
			copy.learningRates = Arrays.copyOf(learningRates, learningRates.length);
			copy.maxErrors = Arrays.copyOf(maxErrors, maxErrors.length);
			copy.legacyMaxIterations = Arrays.copyOf(legacyMaxIterations,
					legacyMaxIterations.length);
			copy.sparseEpochs = Arrays.copyOf(sparseEpochs, sparseEpochs.length);
			copy.fedAvgRounds = Arrays.copyOf(fedAvgRounds, fedAvgRounds.length);
			copy.sparseL2Values = Arrays.copyOf(sparseL2Values,
					sparseL2Values.length);
			copy.fedProxMuValues = Arrays.copyOf(fedProxMuValues,
					fedProxMuValues.length);
			copy.seeds = Arrays.copyOf(seeds, seeds.length);
			copy.validationPercent = validationPercent;
			copy.testPercent = testPercent;
			copy.splitSeed = splitSeed;
			return copy;
		}

		void validate()
		{
			if (datasetFile == null || !datasetFile.isFile())
			{
				throw new IllegalArgumentException(
						"Choose a readable DSMP six-column dataset file.");
			}
			if (!includeLegacy && !includeSparse)
			{
				throw new IllegalArgumentException(
						"Select at least one MLP engine.");
			}
			requirePositive("hidden layers", hiddenLayers);
			requirePositive("hidden neurons", hiddenNeurons);
			requirePositive("learning rates", learningRates);
			requirePositive("seeds", seeds);
			if (includeLegacy)
			{
				requirePositive("max errors", maxErrors);
				requirePositive("legacy max iterations", legacyMaxIterations);
			}
			if (includeSparse)
			{
				requirePositive("sparse epochs", sparseEpochs);
				requirePositive("FedAvg rounds", fedAvgRounds);
				requireNonNegative("sparse L2", sparseL2Values);
				requireNonNegative("FedProx mu", fedProxMuValues);
			}
			if (validationPercent <= 0.0 || testPercent < 0.0
					|| validationPercent + testPercent >= 0.85)
			{
				throw new IllegalArgumentException(
						"Validation/test split leaves too little training data.");
			}
		}

		List<Candidate> candidates()
		{
			List<Candidate> candidates = new ArrayList<Candidate>();
			if (includeLegacy)
			{
				for (int layers : unique(hiddenLayers))
				{
					for (int hidden : unique(hiddenNeurons))
					{
						for (double rate : unique(learningRates))
						{
							for (double maxError : unique(maxErrors))
							{
								for (int maxIterations : unique(legacyMaxIterations))
								{
									for (long seed : unique(seeds))
									{
										candidates.add(Candidate.legacy(layers,
												hidden, rate, maxError,
												maxIterations, seed));
									}
								}
							}
						}
					}
				}
			}
			if (includeSparse)
			{
				for (int layers : unique(hiddenLayers))
				{
					for (int hidden : unique(hiddenNeurons))
					{
						for (double rate : unique(learningRates))
						{
							for (int epochs : unique(sparseEpochs))
							{
								for (int rounds : unique(fedAvgRounds))
								{
									for (double l2 : unique(sparseL2Values))
									{
										for (double fedProx : unique(fedProxMuValues))
										{
											for (long seed : unique(seeds))
											{
												candidates.add(Candidate.sparse(
														layers, hidden, rate,
														epochs, rounds, l2,
														fedProx, seed));
											}
										}
									}
								}
							}
						}
					}
				}
			}
			return candidates;
		}

		private static void requirePositive(String label, int[] values)
		{
			if (values == null || values.length == 0)
			{
				throw new IllegalArgumentException("Missing " + label + ".");
			}
			for (int value : values)
			{
				if (value <= 0)
				{
					throw new IllegalArgumentException(
							label + " must contain positive values.");
				}
			}
		}

		private static void requirePositive(String label, long[] values)
		{
			if (values == null || values.length == 0)
			{
				throw new IllegalArgumentException("Missing " + label + ".");
			}
			for (long value : values)
			{
				if (value < 0)
				{
					throw new IllegalArgumentException(
							label + " must contain non-negative values.");
				}
			}
		}

		private static void requirePositive(String label, double[] values)
		{
			if (values == null || values.length == 0)
			{
				throw new IllegalArgumentException("Missing " + label + ".");
			}
			for (double value : values)
			{
				if (!(value > 0.0))
				{
					throw new IllegalArgumentException(
							label + " must contain positive values.");
				}
			}
		}

		private static void requireNonNegative(String label, double[] values)
		{
			if (values == null || values.length == 0)
			{
				throw new IllegalArgumentException("Missing " + label + ".");
			}
			for (double value : values)
			{
				if (value < 0.0)
				{
					throw new IllegalArgumentException(
							label + " must contain non-negative values.");
				}
			}
		}
	}

	static final class Candidate
	{
		final String engine;
		final int hiddenLayers;
		final int hiddenNeurons;
		final double learningRate;
		final double maxError;
		final int maxIterations;
		final int sparseEpochs;
		final int fedAvgRounds;
		final double sparseL2;
		final double fedProxMu;
		final long seed;

		private Candidate(
				String engine,
				int hiddenLayers,
				int hiddenNeurons,
				double learningRate,
				double maxError,
				int maxIterations,
				int sparseEpochs,
				int fedAvgRounds,
				double sparseL2,
				double fedProxMu,
				long seed)
		{
			this.engine = engine;
			this.hiddenLayers = hiddenLayers;
			this.hiddenNeurons = hiddenNeurons;
			this.learningRate = learningRate;
			this.maxError = maxError;
			this.maxIterations = maxIterations;
			this.sparseEpochs = sparseEpochs;
			this.fedAvgRounds = fedAvgRounds;
			this.sparseL2 = sparseL2;
			this.fedProxMu = fedProxMu;
			this.seed = seed;
		}

		static Candidate legacy(
				int hiddenLayers,
				int hiddenNeurons,
				double learningRate,
				double maxError,
				int maxIterations,
				long seed)
		{
			return new Candidate(
					AlgorithmParameterSettings.MLP_ENGINE_LEGACY_NEUROPH,
					hiddenLayers,
					hiddenNeurons,
					learningRate,
					maxError,
					maxIterations,
					0,
					0,
					0.0,
					0.0,
					seed);
		}

		static Candidate sparse(
				int hiddenLayers,
				int hiddenNeurons,
				double learningRate,
				int sparseEpochs,
				int fedAvgRounds,
				double sparseL2,
				double fedProxMu,
				long seed)
		{
			return new Candidate(
					AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED,
					hiddenLayers,
					hiddenNeurons,
					learningRate,
					0.01,
					0,
					sparseEpochs,
					fedAvgRounds,
					sparseL2,
					fedProxMu,
					seed);
		}

		String engineLabel()
		{
			return AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED.equals(engine)
					? "Sparse Federated MLP" : "Legacy Neuroph MLP";
		}

		String compactLabel()
		{
			if (AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED.equals(engine))
			{
				return engineLabel() + " " + hiddenLayers + "L x "
						+ hiddenNeurons + ", lr=" + format(learningRate)
						+ ", epochs=" + sparseEpochs + ", seed=" + seed;
			}
			return engineLabel() + " " + hiddenLayers + "L x "
					+ hiddenNeurons + ", lr=" + format(learningRate)
					+ ", maxIter=" + maxIterations + ", seed=" + seed;
		}
	}

	static final class SearchResult
	{
		static final Comparator<SearchResult> BEST_FIRST =
				new Comparator<SearchResult>()
				{
					public int compare(SearchResult left, SearchResult right)
					{
						if (left.success != right.success)
						{
							return left.success ? -1 : 1;
						}
						if (left.collapsed != right.collapsed)
						{
							return left.collapsed ? 1 : -1;
						}
						int cmp = Double.compare(right.validationAccuracy,
								left.validationAccuracy);
						if (cmp != 0)
						{
							return cmp;
						}
						cmp = Double.compare(right.testAccuracy,
								left.testAccuracy);
						if (cmp != 0)
						{
							return cmp;
						}
						cmp = Long.compare(left.totalMillis, right.totalMillis);
						if (cmp != 0)
						{
							return cmp;
						}
						return Long.compare(left.modelPayloadBytes,
								right.modelPayloadBytes);
					}
				};

		final Candidate candidate;
		final boolean success;
		final String status;
		final String note;
		final double trainAccuracy;
		final double validationAccuracy;
		final double testAccuracy;
		final int validationCorrect;
		final int validationTotal;
		final int testCorrect;
		final int testTotal;
		final int predictedClasses;
		final String topPrediction;
		final String topActual;
		final boolean collapsed;
		final long trainMillis;
		final long evaluationMillis;
		final long totalMillis;
		final long modelPayloadBytes;
		final long denseDataBytes;
		final String trainingMode;

		private SearchResult(
				Candidate candidate,
				boolean success,
				String status,
				String note,
				double trainAccuracy,
				double validationAccuracy,
				double testAccuracy,
				int validationCorrect,
				int validationTotal,
				int testCorrect,
				int testTotal,
				int predictedClasses,
				String topPrediction,
				String topActual,
				boolean collapsed,
				long trainMillis,
				long evaluationMillis,
				long totalMillis,
				long modelPayloadBytes,
				long denseDataBytes,
				String trainingMode)
		{
			this.candidate = candidate;
			this.success = success;
			this.status = status;
			this.note = note;
			this.trainAccuracy = trainAccuracy;
			this.validationAccuracy = validationAccuracy;
			this.testAccuracy = testAccuracy;
			this.validationCorrect = validationCorrect;
			this.validationTotal = validationTotal;
			this.testCorrect = testCorrect;
			this.testTotal = testTotal;
			this.predictedClasses = predictedClasses;
			this.topPrediction = topPrediction;
			this.topActual = topActual;
			this.collapsed = collapsed;
			this.trainMillis = trainMillis;
			this.evaluationMillis = evaluationMillis;
			this.totalMillis = totalMillis;
			this.modelPayloadBytes = modelPayloadBytes;
			this.denseDataBytes = denseDataBytes;
			this.trainingMode = trainingMode;
		}

		static SearchResult completed(
				Candidate candidate,
				DataProfile profile,
				Metrics train,
				Metrics validation,
				Metrics test,
				long trainMillis,
				long evaluationMillis,
				long modelPayloadBytes,
				String trainingMode)
		{
			boolean collapsed = validation.collapsed(profile.classNames.length);
			String note = collapsed
					? "Validation predictions collapsed to one class."
					: "Ranked by validation accuracy.";
			return new SearchResult(candidate, true, "OK", note,
					train.accuracy,
					validation.accuracy,
					test.accuracy,
					validation.correct,
					validation.total,
					test.correct,
					test.total,
					validation.predictedClasses,
					validation.topPrediction,
					validation.topActual,
					collapsed,
					trainMillis,
					evaluationMillis,
					trainMillis + evaluationMillis,
					modelPayloadBytes,
					(long)(profile.trainRows + profile.validationRows
							+ profile.testRows)
							* (long)(profile.featureCount + profile.classNames.length)
							* 8L,
					trainingMode);
		}

		static SearchResult failed(
				Candidate candidate,
				DataProfile profile,
				String error)
		{
			return new SearchResult(candidate, false, "FAILED", error,
					0.0, 0.0, 0.0,
					0, profile.validationRows,
					0, profile.testRows,
					0, "", "", false,
					0L, 0L, 0L, 0L, 0L, "");
		}

		AlgorithmParameterSettings toAlgorithmParameterSettings()
		{
			AlgorithmParameterSettings settings =
					AlgorithmParameterSettings.defaults();
			settings.setCustomMlpEnabled(true);
			settings.setMlpEngine(candidate.engine);
			settings.setMlpHiddenLayers(candidate.hiddenLayers);
			settings.setMlpHiddenNeurons(candidate.hiddenNeurons);
			settings.setMlpLearningRate(candidate.learningRate);
			settings.setMlpMaxError(candidate.maxError);
			settings.setMlpMaxIterations(candidate.maxIterations);
			if (AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED.equals(
					candidate.engine))
			{
				settings.setMlpSparseEpochs(candidate.sparseEpochs);
				settings.setMlpFedAvgRounds(candidate.fedAvgRounds);
				settings.setMlpSparseL2(candidate.sparseL2);
				settings.setMlpFedProxMu(candidate.fedProxMu);
			}
			settings.validateForMlp();
			return settings;
		}

		String summary()
		{
			return candidate.engineLabel()
					+ " | layers=" + candidate.hiddenLayers
					+ ", neurons=" + candidate.hiddenNeurons
					+ ", lr=" + format(candidate.learningRate)
					+ ", val=" + format(validationAccuracy) + "%"
					+ ", test=" + format(testAccuracy) + "%"
					+ ", time=" + totalMillis + " ms";
		}

		static String csvHeader()
		{
			return "engine,layers,hiddenNeurons,learningRate,maxError,"
					+ "maxIterations,sparseEpochs,fedAvgRounds,sparseL2,"
					+ "fedProxMu,seed,trainAccuracy,validationAccuracy,"
					+ "testAccuracy,validationCorrect,validationTotal,"
					+ "testCorrect,testTotal,predictedClasses,topPrediction,"
					+ "topActual,collapsed,trainMillis,evaluationMillis,"
					+ "totalMillis,modelPayloadBytes,denseDataBytes,"
					+ "trainingMode,status,note";
		}

		String toCsvRow()
		{
			return csv(candidate.engineLabel()) + ","
					+ candidate.hiddenLayers + ","
					+ candidate.hiddenNeurons + ","
					+ format(candidate.learningRate) + ","
					+ format(candidate.maxError) + ","
					+ candidate.maxIterations + ","
					+ candidate.sparseEpochs + ","
					+ candidate.fedAvgRounds + ","
					+ format(candidate.sparseL2) + ","
					+ format(candidate.fedProxMu) + ","
					+ candidate.seed + ","
					+ format(trainAccuracy) + ","
					+ format(validationAccuracy) + ","
					+ format(testAccuracy) + ","
					+ validationCorrect + ","
					+ validationTotal + ","
					+ testCorrect + ","
					+ testTotal + ","
					+ predictedClasses + ","
					+ csv(topPrediction) + ","
					+ csv(topActual) + ","
					+ collapsed + ","
					+ trainMillis + ","
					+ evaluationMillis + ","
					+ totalMillis + ","
					+ modelPayloadBytes + ","
					+ denseDataBytes + ","
					+ csv(trainingMode) + ","
					+ csv(status) + ","
					+ csv(note);
		}
	}

	static final class SearchReport
	{
		final DataProfile profile;
		final List<SearchResult> results;
		final SearchResult best;
		final long elapsedMillis;

		SearchReport(
				DataProfile profile,
				List<SearchResult> results,
				long elapsedMillis)
		{
			this.profile = profile;
			this.results = new ArrayList<SearchResult>(results);
			this.best = firstSuccessful(this.results);
			this.elapsedMillis = elapsedMillis;
		}

		private static SearchResult firstSuccessful(List<SearchResult> results)
		{
			for (SearchResult result : results)
			{
				if (result.success)
				{
					return result;
				}
			}
			return null;
		}
	}

	static final class DataProfile
	{
		final File file;
		final int acceptedRows;
		final int malformedRows;
		final int labelConflicts;
		final int userCount;
		final int featureCount;
		final String[] classNames;
		final int[] trainCounts;
		final int[] validationCounts;
		final int[] testCounts;
		final int trainRows;
		final int validationRows;
		final int testRows;

		DataProfile(
				File file,
				int acceptedRows,
				int malformedRows,
				int labelConflicts,
				int userCount,
				int featureCount,
				String[] classNames,
				int[] trainCounts,
				int[] validationCounts,
				int[] testCounts)
		{
			this.file = file;
			this.acceptedRows = acceptedRows;
			this.malformedRows = malformedRows;
			this.labelConflicts = labelConflicts;
			this.userCount = userCount;
			this.featureCount = featureCount;
			this.classNames = Arrays.copyOf(classNames, classNames.length);
			this.trainCounts = Arrays.copyOf(trainCounts, trainCounts.length);
			this.validationCounts = Arrays.copyOf(validationCounts,
					validationCounts.length);
			this.testCounts = Arrays.copyOf(testCounts, testCounts.length);
			this.trainRows = sum(trainCounts);
			this.validationRows = sum(validationCounts);
			this.testRows = sum(testCounts);
		}

		double validationMajorityAccuracy()
		{
			return majorityAccuracy(validationCounts);
		}

		double testMajorityAccuracy()
		{
			return majorityAccuracy(testCounts);
		}

		String summary()
		{
			return "File: " + file.getAbsolutePath()
					+ "\nAccepted rows: " + acceptedRows
					+ "\nUsers/documents: " + userCount
					+ "\nFeatures: " + featureCount
					+ "\nClasses: " + classNames.length
					+ "\nTrain/validation/test users: " + trainRows + "/"
					+ validationRows + "/" + testRows
					+ "\nValidation majority baseline: "
					+ format(validationMajorityAccuracy()) + "%"
					+ "\nTest majority baseline: "
					+ format(testMajorityAccuracy()) + "%"
					+ "\nMalformed rows: " + malformedRows
					+ "\nUser label conflicts kept as first label: "
					+ labelConflicts;
		}

		String validationDistribution()
		{
			return distribution(validationCounts, classNames);
		}
	}

	static final class Metrics
	{
		final int correct;
		final int total;
		final double accuracy;
		final int predictedClasses;
		final String topPrediction;
		final String topActual;

		Metrics(
				int correct,
				int total,
				int[] predictedCounts,
				int[] actualCounts,
				String[] classNames)
		{
			this.correct = correct;
			this.total = total;
			this.accuracy = total <= 0 ? 0.0 : (100.0 * correct) / total;
			this.predictedClasses = countNonZero(predictedCounts);
			this.topPrediction = labelCount(predictedCounts, classNames);
			this.topActual = labelCount(actualCounts, classNames);
		}

		boolean collapsed(int classCount)
		{
			return classCount > 1 && total > 0 && predictedClasses <= 1;
		}
	}

	private static final class DataSplit
	{
		final int inputSize;
		final int outputSize;
		final String[] classNames;
		final List<SparseFederatedMlpModelSupport.TrainingExample> train;
		final List<SparseFederatedMlpModelSupport.TrainingExample> validation;
		final List<SparseFederatedMlpModelSupport.TrainingExample> test;
		final DataProfile profile;

		DataSplit(
				int inputSize,
				int outputSize,
				String[] classNames,
				List<SparseFederatedMlpModelSupport.TrainingExample> train,
				List<SparseFederatedMlpModelSupport.TrainingExample> validation,
				List<SparseFederatedMlpModelSupport.TrainingExample> test,
				DataProfile profile)
		{
			this.inputSize = inputSize;
			this.outputSize = outputSize;
			this.classNames = classNames;
			this.train = train;
			this.validation = validation;
			this.test = test;
			this.profile = profile;
		}
	}

	private static final class Split
	{
		final List<String> trainUsers;
		final List<String> validationUsers;
		final List<String> testUsers;

		Split(
				List<String> trainUsers,
				List<String> validationUsers,
				List<String> testUsers)
		{
			this.trainUsers = trainUsers;
			this.validationUsers = validationUsers;
			this.testUsers = testUsers;
		}
	}

	private static int[] unique(int[] values)
	{
		LinkedHashSet<Integer> unique = new LinkedHashSet<Integer>();
		for (int value : values)
		{
			unique.add(Integer.valueOf(value));
		}
		int[] result = new int[unique.size()];
		int i = 0;
		for (Integer value : unique)
		{
			result[i++] = value.intValue();
		}
		return result;
	}

	private static long[] unique(long[] values)
	{
		LinkedHashSet<Long> unique = new LinkedHashSet<Long>();
		for (long value : values)
		{
			unique.add(Long.valueOf(value));
		}
		long[] result = new long[unique.size()];
		int i = 0;
		for (Long value : unique)
		{
			result[i++] = value.longValue();
		}
		return result;
	}

	private static double[] unique(double[] values)
	{
		LinkedHashSet<Double> unique = new LinkedHashSet<Double>();
		for (double value : values)
		{
			unique.add(Double.valueOf(value));
		}
		double[] result = new double[unique.size()];
		int i = 0;
		for (Double value : unique)
		{
			result[i++] = value.doubleValue();
		}
		return result;
	}

	private static double majorityAccuracy(int[] counts)
	{
		int total = sum(counts);
		if (total <= 0)
		{
			return 0.0;
		}
		return 100.0 * counts[indexOfMax(counts)] / total;
	}

	private static String labelCount(int[] counts, String[] classNames)
	{
		if (counts.length == 0 || classNames.length == 0)
		{
			return "";
		}
		int index = indexOfMax(counts);
		if (index < 0 || index >= classNames.length)
		{
			return "";
		}
		return classNames[index] + ":" + counts[index];
	}

	private static String distribution(int[] counts, String[] classNames)
	{
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < counts.length && i < classNames.length; i++)
		{
			if (i > 0)
			{
				sb.append(" | ");
			}
			sb.append(classNames[i]).append(":").append(counts[i]);
		}
		return sb.toString();
	}

	private static String format(double value)
	{
		return String.format(java.util.Locale.US, "%.4f",
				Double.valueOf(value));
	}

	private static String csv(String value)
	{
		if (value == null)
		{
			value = "";
		}
		return "\"" + value.replace("\"", "\"\"") + "\"";
	}
}
