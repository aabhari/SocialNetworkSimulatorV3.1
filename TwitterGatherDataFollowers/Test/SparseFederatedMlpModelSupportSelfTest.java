package TwitterGatherDataFollowers.userRyersonU;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SparseFederatedMlpModelSupportSelfTest
{
	private SparseFederatedMlpModelSupportSelfTest()
	{
	}

	public static void main(String[] args)
	{
		usesRequestedHiddenLayerDepth();
		deepStableModeStartsAtFourHiddenLayers();
		differentDepthsProduceDifferentPredictions();
		separableTenClassDataDoesNotCollapse();
		deDuplicatedEvaluationAggregationCountsEachUserOnce();
		System.out.println("SparseFederatedMlpModelSupportSelfTest passed");
	}

	private static void usesRequestedHiddenLayerDepth()
	{
		assertShape(1, 2);
		assertShape(4, 5);
		assertShape(8, 9);
	}

	private static void assertShape(int hiddenLayers, int expectedWeightMatrices)
	{
		SparseFederatedMlpModelSupport.SparseMlpModel model =
				SparseFederatedMlpModelSupport.SparseMlpModel.create(
						6, hiddenLayers, 5, 3, 123L);
		if (model.getHiddenLayerCount() != hiddenLayers)
		{
			throw new AssertionError("expected hidden layer count "
					+ hiddenLayers + " but got " + model.getHiddenLayerCount());
		}
		if (model.getWeightLayerCount() != expectedWeightMatrices)
		{
			throw new AssertionError("expected " + expectedWeightMatrices
					+ " weight matrices but got " + model.getWeightLayerCount());
		}
	}

	private static void deepStableModeStartsAtFourHiddenLayers()
	{
		SparseFederatedMlpModelSupport.SparseMlpModel shallow =
				SparseFederatedMlpModelSupport.SparseMlpModel.create(
						6, 3, 5, 3, 123L);
		SparseFederatedMlpModelSupport.SparseMlpModel deep =
				SparseFederatedMlpModelSupport.SparseMlpModel.create(
						6, 4, 5, 3, 123L);
		if (shallow.usesDeepStableMode())
		{
			throw new AssertionError("3 hidden layers should keep standard tanh mode");
		}
		if (!"standard-tanh".equals(shallow.trainingModeLabel())
				|| !"tanh".equals(shallow.hiddenActivationLabel())
				|| !"xavierUniform".equals(shallow.initializationLabel()))
		{
			throw new AssertionError("unexpected shallow sparse MLP mode metadata");
		}
		assertClose(0.1, shallow.effectiveLearningRate(0.1),
				"shallow learning rate");
		if (!deep.usesDeepStableMode())
		{
			throw new AssertionError("4 hidden layers should use deep-stable mode");
		}
		if (!"deep-stable".equals(deep.trainingModeLabel())
				|| !deep.hiddenActivationLabel().startsWith("leakyReLU")
				|| !"heUniformFanIn".equals(deep.initializationLabel()))
		{
			throw new AssertionError("unexpected deep sparse MLP mode metadata");
		}
		assertClose(0.03, deep.effectiveLearningRate(0.1),
				"deep capped learning rate");
		assertClose(0.02, deep.effectiveLearningRate(0.02),
				"deep already-safe learning rate");
	}

	private static void differentDepthsProduceDifferentPredictions()
	{
		double[] oneLayer = trainedPrediction(1);
		double[] fourLayers = trainedPrediction(4);
		double[] eightLayers = trainedPrediction(8);
		assertDifferent(oneLayer, fourLayers, "1 hidden layer", "4 hidden layers");
		assertDifferent(fourLayers, eightLayers, "4 hidden layers", "8 hidden layers");
	}

	private static double[] trainedPrediction(int hiddenLayers)
	{
		SparseFederatedMlpModelSupport.SparseMlpModel model =
				SparseFederatedMlpModelSupport.SparseMlpModel.create(
						6, hiddenLayers, 5, 3, 123L);
		List<SparseFederatedMlpModelSupport.TrainingExample> examples = examples();
		model.train(examples, 20, 0.1, 0.0001, 0.0);
		return model.predict(examples.get(0).getFeatures());
	}

	private static void assertDifferent(
			double[] left,
			double[] right,
			String leftName,
			String rightName)
	{
		double sum = 0.0;
		for (int i = 0; i < left.length && i < right.length; i++)
		{
			sum += Math.abs(left[i] - right[i]);
		}
		if (sum <= 0.000001)
		{
			throw new AssertionError(leftName + " and " + rightName
					+ " produced identical sparse MLP predictions");
		}
	}

	private static void separableTenClassDataDoesNotCollapse()
	{
		SparseFederatedMlpModelSupport.SparseMlpModel model =
				SparseFederatedMlpModelSupport.SparseMlpModel.create(
						10, 1, 10, 10, 321L);
		List<SparseFederatedMlpModelSupport.TrainingExample> examples =
				tenClassExamples();
		model.train(examples, 80, 0.1, 0.0001, 0.0);
		boolean[] predicted = new boolean[10];
		for (SparseFederatedMlpModelSupport.TrainingExample example : examples)
		{
			predicted[indexOfMax(model.predict(example.getFeatures()))] = true;
		}
		int predictedClasses = 0;
		for (boolean value : predicted)
		{
			if (value)
			{
				predictedClasses++;
			}
		}
		if (predictedClasses < 8)
		{
			throw new AssertionError("expected sparse MLP to preserve class spread, got "
					+ predictedClasses + " predicted classes");
		}
	}

	private static List<SparseFederatedMlpModelSupport.TrainingExample> examples()
	{
		List<SparseFederatedMlpModelSupport.TrainingExample> rows =
				new ArrayList<SparseFederatedMlpModelSupport.TrainingExample>();
		rows.add(new SparseFederatedMlpModelSupport.TrainingExample(
				"u1", row(0, 1.0, 3, 0.5), 0));
		rows.add(new SparseFederatedMlpModelSupport.TrainingExample(
				"u2", row(1, 0.8, 4, 0.7), 1));
		rows.add(new SparseFederatedMlpModelSupport.TrainingExample(
				"u3", row(2, 0.9, 5, 0.4), 2));
		return rows;
	}

	private static void deDuplicatedEvaluationAggregationCountsEachUserOnce()
	{
		List<SparseFederatedMlpModelSupport.EvaluationResult> results =
				new ArrayList<SparseFederatedMlpModelSupport.EvaluationResult>();
		List<SparseFederatedMlpModelSupport.Prediction> nodeOne =
				new ArrayList<SparseFederatedMlpModelSupport.Prediction>();
		nodeOne.add(new SparseFederatedMlpModelSupport.Prediction("u1", 0, 0));
		nodeOne.add(new SparseFederatedMlpModelSupport.Prediction("u2", 1, 0));
		List<SparseFederatedMlpModelSupport.Prediction> nodeTwo =
				new ArrayList<SparseFederatedMlpModelSupport.Prediction>();
		nodeTwo.add(new SparseFederatedMlpModelSupport.Prediction("u1", 0, 0));
		nodeTwo.add(new SparseFederatedMlpModelSupport.Prediction("u3", 1, 1));
		results.add(new SparseFederatedMlpModelSupport.EvaluationResult("n1", nodeOne));
		results.add(new SparseFederatedMlpModelSupport.EvaluationResult("n2", nodeTwo));

		SparseFederatedMlpModelSupport.AggregatedEvaluationResult aggregated =
				SparseFederatedMlpModelSupport.aggregateEvaluationResults(results, 2);
		if (aggregated.getTotalInstances() != 3)
		{
			throw new AssertionError("expected 3 de-duplicated users but got "
					+ aggregated.getTotalInstances());
		}
		if (aggregated.getCorrectCount() != 2)
		{
			throw new AssertionError("expected 2 correct predictions but got "
					+ aggregated.getCorrectCount());
		}
		if (aggregated.getDuplicateInstances() != 1)
		{
			throw new AssertionError("expected 1 duplicate prediction but got "
					+ aggregated.getDuplicateInstances());
		}
	}

	private static List<SparseFederatedMlpModelSupport.TrainingExample> tenClassExamples()
	{
		List<SparseFederatedMlpModelSupport.TrainingExample> rows =
				new ArrayList<SparseFederatedMlpModelSupport.TrainingExample>();
		for (int label = 0; label < 10; label++)
		{
			for (int sample = 0; sample < 4; sample++)
			{
				Map<Integer,Double> features = new LinkedHashMap<Integer,Double>();
				features.put(Integer.valueOf(label), Double.valueOf(1.0));
				features.put(Integer.valueOf((label + sample + 1) % 10),
						Double.valueOf(0.05 * (sample + 1)));
				rows.add(new SparseFederatedMlpModelSupport.TrainingExample(
						"c"+label+"_"+sample, features, label));
			}
		}
		return rows;
	}

	private static Map<Integer,Double> row(int first, double firstValue, int second, double secondValue)
	{
		Map<Integer,Double> map = new LinkedHashMap<Integer,Double>();
		map.put(Integer.valueOf(first), Double.valueOf(firstValue));
		map.put(Integer.valueOf(second), Double.valueOf(secondValue));
		return map;
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

	private static void assertClose(double expected, double actual, String label)
	{
		if (Math.abs(expected - actual) > 0.0000001)
		{
			throw new AssertionError(label + " expected " + expected
					+ " but got " + actual);
		}
	}
}
