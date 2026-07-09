package TwitterGatherDataFollowers.userRyersonU;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Standalone Sparse MLP experiment harness for the Retail DSMP six-column file.
 *
 * It reproduces the two professor-report matrices without starting the Swing/JADE
 * application: n=1 with hidden layers 1/3/8, then hiddenLayers=3 with nodes
 * 1/2/4/8.
 */
public final class SparseMlpRetailExperiment
{
	private static final int HIDDEN_NEURONS = Integer.getInteger("dsmp.mlp.hiddenNeurons", 10).intValue();
	private static final int TOTAL_EPOCHS = Integer.getInteger("dsmp.mlp.sparseEpochs", 80).intValue();
	private static final int FEDAVG_ROUNDS = Integer.getInteger("dsmp.mlp.fedAvgRounds", 16).intValue();
	private static final double LEARNING_RATE = Double.parseDouble(System.getProperty("dsmp.mlp.learningRate", "0.1"));
	private static final double L2 = Double.parseDouble(System.getProperty("dsmp.mlp.sparseL2", "0.0001"));
	private static final double FEDPROX_MU = Double.parseDouble(System.getProperty("dsmp.mlp.fedProxMu", "0.0"));
	private static final double TEST_SET_PERCENT = 0.30;
	private static final double MAX_ALLOWED_DISTRIBUTED_DROP = 1.0;

	private SparseMlpRetailExperiment()
	{
	}

	public static void main(String[] args) throws Exception
	{
		File dataset = new File(args.length > 0 ? args[0] : "Dataset/Retail.txt");
		Corpus corpus = Corpus.load(dataset);
		System.out.println("Dataset: " + dataset.getPath());
		System.out.println("Rows accepted: " + corpus.acceptedRows);
		System.out.println("Documents/users: " + corpus.userDocuments.size());
		System.out.println("Features: " + corpus.termIndex.size());
		System.out.println("Classes: " + corpus.followeeIndex.size());
		System.out.println("Train users: " + corpus.trainUsers.size());
		System.out.println("Test users: " + corpus.testUsers.size());
		System.out.println("Hidden neurons: " + HIDDEN_NEURONS);
		System.out.println("Learning rate: " + LEARNING_RATE);
		System.out.println("Sparse epochs: " + TOTAL_EPOCHS);
		System.out.println("FedAvg rounds: " + FEDAVG_ROUNDS);
		System.out.println("Sparse L2: " + L2);
		System.out.println("FedProx mu: " + FEDPROX_MU);
		SparseFederatedMlpModelSupport.SparseMlpModel deepProbe =
				SparseFederatedMlpModelSupport.SparseMlpModel.create(
						corpus.termIndex.size(),
						8,
						HIDDEN_NEURONS,
						corpus.followeeIndex.size(),
						SparseFederatedMlpModelSupport.GLOBAL_INITIALIZATION_SEED);
		System.out.println("8-layer training mode: " + deepProbe.trainingModeLabel());
		System.out.println("8-layer hidden activation: " + deepProbe.hiddenActivationLabel());
		System.out.println("8-layer initialization: " + deepProbe.initializationLabel());
		System.out.println("8-layer effective learning rate: "
				+ deepProbe.effectiveLearningRate(LEARNING_RATE));
		System.out.println();

		System.out.println("Retail dataset accuracy n=1 fixed varying number of layers");
		System.out.println("# Layers\tTime (ms)\tAccuracy (%)\tCorrect\tTotal");
		for (int hiddenLayers : new int[] {1, 3, 8})
		{
			RunResult result = runExperiment(corpus, hiddenLayers, 1);
			printLayerResult(hiddenLayers, result);
		}

		System.out.println();
		System.out.println("Retail dataset timing (Sparse MLP with 3 hidden layers)");
		System.out.println("# Nodes\tTime (ms)\tAccuracy (%)\tCorrect\tTotal\tSpeedup\tPE (%)\tDrop vs n=1");
		RunResult baseline = null;
		boolean distributedDropDetected = false;
		for (int nodes : new int[] {1, 2, 4, 8})
		{
			RunResult result = runExperiment(corpus, 3, nodes);
			if (baseline == null)
			{
				baseline = result;
			}
			double speedup = baseline.elapsedMs <= 0.0 ? 0.0 : baseline.elapsedMs / result.elapsedMs;
			double pe = nodes <= 1 ? 0.0 : (speedup / nodes) * 100.0;
			double drop = baseline.accuracy - result.accuracy;
			if (nodes > 1 && drop > MAX_ALLOWED_DISTRIBUTED_DROP)
			{
				distributedDropDetected = true;
			}
			printNodeResult(nodes, result, speedup, pe, drop);
		}

		if (distributedDropDetected)
		{
			throw new AssertionError("Distributed Sparse MLP accuracy dropped by more than "
					+MAX_ALLOWED_DISTRIBUTED_DROP+" percentage point(s) on this dataset.");
		}
	}

	private static RunResult runExperiment(Corpus corpus, int hiddenLayers, int nodes)
	{
		List<SparseFederatedMlpModelSupport.TrainingExample> train =
				corpus.examples(corpus.trainUsers);
		List<SparseFederatedMlpModelSupport.TrainingExample> test =
				corpus.examples(corpus.testUsers);
		long start = System.nanoTime();
		SparseFederatedMlpModelSupport.SparseMlpModel model;
		if (nodes <= 1)
		{
			model = SparseFederatedMlpModelSupport.SparseMlpModel.create(
					corpus.termIndex.size(),
					hiddenLayers,
					HIDDEN_NEURONS,
					corpus.followeeIndex.size(),
					SparseFederatedMlpModelSupport.GLOBAL_INITIALIZATION_SEED);
			model.train(train, TOTAL_EPOCHS, LEARNING_RATE, L2, FEDPROX_MU);
		}
		else
		{
			model = trainFederated(corpus, hiddenLayers, nodes);
		}
		long end = System.nanoTime();
		Evaluation evaluation = evaluate(model, test);
		return new RunResult((end - start) / 1000000.0, evaluation.correct,
				evaluation.total);
	}

	private static SparseFederatedMlpModelSupport.SparseMlpModel trainFederated(
			Corpus corpus,
			int hiddenLayers,
			int nodes)
	{
		List<List<String>> nodeTrainUsers = corpus.partitionTrainUsers(nodes);
		SparseFederatedMlpModelSupport.SparseMlpModel global =
				SparseFederatedMlpModelSupport.SparseMlpModel.create(
						corpus.termIndex.size(),
						hiddenLayers,
						HIDDEN_NEURONS,
						corpus.followeeIndex.size(),
						SparseFederatedMlpModelSupport.GLOBAL_INITIALIZATION_SEED);
		int rounds = Math.max(1, Math.min(FEDAVG_ROUNDS, TOTAL_EPOCHS));
		for (int round = 1; round <= rounds; round++)
		{
			int localEpochs = epochsForRound(round, rounds, TOTAL_EPOCHS);
			List<SparseFederatedMlpModelSupport.SparseMlpModel> localModels =
					new ArrayList<SparseFederatedMlpModelSupport.SparseMlpModel>();
			for (int node = 0; node < nodes; node++)
			{
				List<SparseFederatedMlpModelSupport.TrainingExample> localExamples =
						corpus.examples(nodeTrainUsers.get(node));
				if (localExamples.isEmpty())
				{
					continue;
				}
				SparseFederatedMlpModelSupport.SparseMlpModel local = global.copy();
				local.train(localExamples, localEpochs, LEARNING_RATE, L2, FEDPROX_MU);
				localModels.add(local);
			}
			if (!localModels.isEmpty())
			{
				global = SparseFederatedMlpModelSupport.SparseMlpModel.average(localModels);
			}
		}
		return global;
	}

	private static int epochsForRound(int round, int rounds, int totalEpochs)
	{
		int base = totalEpochs / rounds;
		int remainder = totalEpochs % rounds;
		return Math.max(1, base + (round <= remainder ? 1 : 0));
	}

	private static Evaluation evaluate(
			SparseFederatedMlpModelSupport.SparseMlpModel model,
			List<SparseFederatedMlpModelSupport.TrainingExample> examples)
	{
		int correct = 0;
		for (SparseFederatedMlpModelSupport.TrainingExample example : examples)
		{
			int predicted = indexOfMax(model.predict(example.getFeatures()));
			if (predicted == example.getLabelIndex())
			{
				correct++;
			}
		}
		return new Evaluation(correct, examples.size());
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

	private static void printLayerResult(int hiddenLayers, RunResult result)
	{
		System.out.println(hiddenLayers+"\t"+Math.round(result.elapsedMs)+"\t"
				+format(result.accuracy)+"\t"+result.correct+"\t"+result.total);
	}

	private static void printNodeResult(
			int nodes,
			RunResult result,
			double speedup,
			double pe,
			double drop)
	{
		System.out.println(nodes+"\t"+Math.round(result.elapsedMs)+"\t"
				+format(result.accuracy)+"\t"+result.correct+"\t"+result.total
				+"\t"+(nodes == 1 ? "N/A" : format(speedup))
				+"\t"+(nodes == 1 ? "N/A" : format(pe))
				+"\t"+(nodes == 1 ? "N/A" : format(drop)));
	}

	private static String format(double value)
	{
		return String.format(java.util.Locale.US, "%.2f", value);
	}

	private static final class Corpus
	{
		private final LinkedHashMap<String,LinkedHashMap<String,Double>> userDocuments;
		private final LinkedHashMap<String,String> userFollowee;
		private final LinkedHashMap<String,Integer> followeeIndex;
		private final LinkedHashMap<String,Integer> termIndex;
		private final List<String> trainUsers;
		private final List<String> testUsers;
		private final int acceptedRows;

		private Corpus(
				LinkedHashMap<String,LinkedHashMap<String,Double>> userDocuments,
				LinkedHashMap<String,String> userFollowee,
				LinkedHashMap<String,Integer> followeeIndex,
				LinkedHashMap<String,Integer> termIndex,
				List<String> trainUsers,
				List<String> testUsers,
				int acceptedRows)
		{
			this.userDocuments = userDocuments;
			this.userFollowee = userFollowee;
			this.followeeIndex = followeeIndex;
			this.termIndex = termIndex;
			this.trainUsers = trainUsers;
			this.testUsers = testUsers;
			this.acceptedRows = acceptedRows;
		}

		static Corpus load(File file) throws Exception
		{
			LinkedHashMap<String,LinkedHashMap<String,Double>> docs =
					new LinkedHashMap<String,LinkedHashMap<String,Double>>();
			LinkedHashMap<String,String> userFollowee =
					new LinkedHashMap<String,String>();
			TreeSet<String> terms = new TreeSet<String>();
			int acceptedRows = 0;
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
						continue;
					}
					String followee = fields[0];
					String user = fields[4];
					LinkedHashMap<String,Double> row = tokenize(fields[5]);
					if (row.size() < 3)
					{
						continue;
					}
					if (!userFollowee.containsKey(user))
					{
						userFollowee.put(user, followee);
					}
					LinkedHashMap<String,Double> userDoc = docs.get(user);
					if (userDoc == null)
					{
						userDoc = new LinkedHashMap<String,Double>();
						docs.put(user, userDoc);
					}
					for (Map.Entry<String,Double> entry : row.entrySet())
					{
						Double old = userDoc.get(entry.getKey());
						userDoc.put(entry.getKey(), Double.valueOf(
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

			LinkedHashMap<String,Integer> termIndex =
					new LinkedHashMap<String,Integer>();
			int termCounter = 0;
			for (String term : terms)
			{
				termIndex.put(term, Integer.valueOf(termCounter++));
			}
			LinkedHashMap<String,Integer> followeeIndex =
					buildFolloweeIndex(userFollowee.values());
			normalizeTfidf(docs, terms);
			Split split = splitUsers(docs, userFollowee);
			return new Corpus(docs, userFollowee, followeeIndex, termIndex,
					split.trainUsers, split.testUsers, acceptedRows);
		}

		List<SparseFederatedMlpModelSupport.TrainingExample> examples(
				Collection<String> users)
		{
			List<SparseFederatedMlpModelSupport.TrainingExample> examples =
					new ArrayList<SparseFederatedMlpModelSupport.TrainingExample>();
			for (String user : users)
			{
				Map<String,Double> tfidf = userDocuments.get(user);
				String followee = userFollowee.get(user);
				Integer label = followee == null ? null : followeeIndex.get(followee);
				if (tfidf == null || label == null)
				{
					continue;
				}
				Map<Integer,Double> sparse = new LinkedHashMap<Integer,Double>();
				for (Map.Entry<String,Double> entry : tfidf.entrySet())
				{
					Integer index = termIndex.get(entry.getKey());
					if (index != null && entry.getValue().doubleValue() != 0.0)
					{
						sparse.put(index, entry.getValue());
					}
				}
				examples.add(new SparseFederatedMlpModelSupport.TrainingExample(
						user, sparse, label.intValue()));
			}
			return examples;
		}

		List<List<String>> partitionTrainUsers(int nodes)
		{
			List<List<String>> partitions = new ArrayList<List<String>>();
			int[] bins = new int[nodes];
			for (int i = 0; i < nodes; i++)
			{
				partitions.add(new ArrayList<String>());
			}
			Map<String,List<String>> byFollowee =
					new LinkedHashMap<String,List<String>>();
			for (String user : trainUsers)
			{
				String followee = userFollowee.get(user);
				List<String> users = byFollowee.get(followee);
				if (users == null)
				{
					users = new ArrayList<String>();
					byFollowee.put(followee, users);
				}
				users.add(user);
			}
			for (List<String> users : byFollowee.values())
			{
				for (String user : users)
				{
					int node = indexOfSmallest(bins);
					partitions.get(node).add(user);
					bins[node] += userDocuments.get(user).size();
				}
			}
			return partitions;
		}

		private static int indexOfSmallest(int[] values)
		{
			int index = 0;
			for (int i = 1; i < values.length; i++)
			{
				if (values[i] < values[index])
				{
					index = i;
				}
			}
			return index;
		}
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
		Matcher matcher = Pattern.compile("[\\p{P}]").matcher(current);
		current = matcher.replaceAll("");
		matcher = Pattern.compile("http[a-zA-Z0-9]*|bitly[a-zA-Z0-9]*|www[a-zA-Z0-9]*").matcher(current);
		current = matcher.replaceAll(" ");
		matcher = Pattern.compile("[^a-zA-Z\\p{Z}]").matcher(current);
		current = matcher.replaceAll(" ");
		current = current.toLowerCase().replaceAll("[^a-zA-Z ]", "")
				.replaceAll(" +", " ").trim();
		LinkedHashMap<String,Double> row = new LinkedHashMap<String,Double>();
		if (current.length() == 0 || current.split("\\s+").length < 3)
		{
			return row;
		}
		Scanner scanner = new Scanner(current);
		try
		{
			while (scanner.hasNext())
			{
				String term = scanner.next();
				Double old = row.get(term);
				row.put(term, Double.valueOf((old == null ? 0.0 : old.doubleValue()) + 1.0));
			}
		}
		finally
		{
			scanner.close();
		}
		return row;
	}

	private static LinkedHashMap<String,Integer> buildFolloweeIndex(
			Collection<String> followees)
	{
		LinkedHashMap<String,Integer> index = new LinkedHashMap<String,Integer>();
		for (String followee : followees)
		{
			if (!index.containsKey(followee))
			{
				index.put(followee, Integer.valueOf(index.size()));
			}
		}
		return index;
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
				double idf = log2Idf(totalDocs, df.get(entry.getKey()).intValue());
				double value = entry.getValue().doubleValue() * idf;
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
				entry.setValue(Double.valueOf(entry.getValue().doubleValue() / magnitude));
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

	private static Split splitUsers(
			LinkedHashMap<String,LinkedHashMap<String,Double>> docs,
			LinkedHashMap<String,String> userFollowee)
	{
		LinkedHashMap<String,List<String>> followeeFollowers =
				new LinkedHashMap<String,List<String>>();
		for (String user : docs.keySet())
		{
			String followee = userFollowee.get(user);
			List<String> users = followeeFollowers.get(followee);
			if (users == null)
			{
				users = new ArrayList<String>();
				followeeFollowers.put(followee, users);
			}
			users.add(user);
		}
		List<String> testUsers = new ArrayList<String>();
		List<String> trainUsers = new ArrayList<String>();
		for (List<String> users : followeeFollowers.values())
		{
			int testCount = (int)Math.floor(users.size() * TEST_SET_PERCENT);
			if (users.size() > 1)
			{
				testCount = Math.max(1, Math.min(testCount, users.size() - 1));
			}
			else
			{
				testCount = 0;
			}
			for (int i = 0; i < users.size(); i++)
			{
				if (i < testCount)
				{
					testUsers.add(users.get(i));
				}
				else
				{
					trainUsers.add(users.get(i));
				}
			}
		}
		return new Split(trainUsers, testUsers);
	}

	private static final class Split
	{
		private final List<String> trainUsers;
		private final List<String> testUsers;

		private Split(List<String> trainUsers, List<String> testUsers)
		{
			this.trainUsers = trainUsers;
			this.testUsers = testUsers;
		}
	}

	private static final class Evaluation
	{
		private final int correct;
		private final int total;

		private Evaluation(int correct, int total)
		{
			this.correct = correct;
			this.total = total;
		}
	}

	private static final class RunResult
	{
		private final double elapsedMs;
		private final int correct;
		private final int total;
		private final double accuracy;

		private RunResult(double elapsedMs, int correct, int total)
		{
			this.elapsedMs = elapsedMs;
			this.correct = correct;
			this.total = total;
			this.accuracy = total <= 0 ? 0.0 : (100.0 * correct) / total;
		}
	}
}
