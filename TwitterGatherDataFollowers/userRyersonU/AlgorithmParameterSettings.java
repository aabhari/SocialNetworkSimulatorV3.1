package TwitterGatherDataFollowers.userRyersonU;

import java.io.Serializable;

/**
 * MLP-only advanced settings for the v2.5fix UI.
 *
 * The default object intentionally preserves the legacy Neuroph MLP behavior:
 * one hidden layer, ten hidden neurons, learning rate 0.1, max error 0.01.
 */
final class AlgorithmParameterSettings implements Serializable
{
    private static final long serialVersionUID = 1L;

    static final String MLP_ENGINE_LEGACY_NEUROPH = "legacy-neuroph";
    static final String MLP_ENGINE_SPARSE_FEDERATED = "sparse-federated";

    private boolean customMlpEnabled;
    private String mlpEngine;
    private int mlpHiddenLayers;
    private int mlpHiddenNeurons;
    private double mlpLearningRate;
    private double mlpMaxError;
    private int mlpSparseEpochs;
    private double mlpSparseL2;
    private double mlpFedProxMu;

    private AlgorithmParameterSettings()
    {
        resetToDefaults();
    }

    static AlgorithmParameterSettings defaults()
    {
        return new AlgorithmParameterSettings();
    }

    static AlgorithmParameterSettings effective(AlgorithmParameterSettings settings)
    {
        if (settings == null || !settings.customMlpEnabled)
        {
            return defaults();
        }
        return settings.copy();
    }

    static AlgorithmParameterSettings fromSystemProperties()
    {
        AlgorithmParameterSettings settings = defaults();
        settings.customMlpEnabled = Boolean.parseBoolean(
                System.getProperty("dsmp.mlp.custom", "false"));
        settings.mlpEngine = System.getProperty(
                "dsmp.mlp.engine", MLP_ENGINE_LEGACY_NEUROPH);
        settings.mlpHiddenLayers = intProperty(
                "dsmp.mlp.hiddenLayers", settings.mlpHiddenLayers);
        settings.mlpHiddenNeurons = intProperty(
                "dsmp.mlp.hiddenNeurons", settings.mlpHiddenNeurons);
        settings.mlpLearningRate = doubleProperty(
                "dsmp.mlp.learningRate", settings.mlpLearningRate);
        settings.mlpMaxError = doubleProperty(
                "dsmp.mlp.maxError", settings.mlpMaxError);
        settings.mlpSparseEpochs = intProperty(
                "dsmp.mlp.sparseEpochs", settings.mlpSparseEpochs);
        settings.mlpSparseL2 = doubleProperty(
                "dsmp.mlp.sparseL2", settings.mlpSparseL2);
        settings.mlpFedProxMu = doubleProperty(
                "dsmp.mlp.fedProxMu", settings.mlpFedProxMu);
        settings.validateForMlp();
        return settings;
    }

    AlgorithmParameterSettings copy()
    {
        AlgorithmParameterSettings copy = new AlgorithmParameterSettings();
        copy.customMlpEnabled = customMlpEnabled;
        copy.mlpEngine = mlpEngine;
        copy.mlpHiddenLayers = mlpHiddenLayers;
        copy.mlpHiddenNeurons = mlpHiddenNeurons;
        copy.mlpLearningRate = mlpLearningRate;
        copy.mlpMaxError = mlpMaxError;
        copy.mlpSparseEpochs = mlpSparseEpochs;
        copy.mlpSparseL2 = mlpSparseL2;
        copy.mlpFedProxMu = mlpFedProxMu;
        return copy;
    }

    void resetToDefaults()
    {
        customMlpEnabled = false;
        mlpEngine = MLP_ENGINE_LEGACY_NEUROPH;
        mlpHiddenLayers = 1;
        mlpHiddenNeurons = 10;
        mlpLearningRate = 0.1;
        mlpMaxError = 0.01;
        mlpSparseEpochs = 8;
        mlpSparseL2 = 0.0001;
        mlpFedProxMu = 0.0;
    }

    void validateForMlp()
    {
        if (!MLP_ENGINE_LEGACY_NEUROPH.equals(mlpEngine)
                && !MLP_ENGINE_SPARSE_FEDERATED.equals(mlpEngine))
        {
            throw new IllegalArgumentException("Unknown MLP engine: " + mlpEngine);
        }
        if (mlpHiddenLayers < 1 || mlpHiddenLayers > 8)
        {
            throw new IllegalArgumentException("Hidden layers must be between 1 and 8.");
        }
        if (mlpHiddenNeurons < 1 || mlpHiddenNeurons > 512)
        {
            throw new IllegalArgumentException("Hidden neurons must be between 1 and 512.");
        }
        if (!(mlpLearningRate > 0.0 && mlpLearningRate <= 1.0))
        {
            throw new IllegalArgumentException("Learning rate must be greater than 0 and at most 1.");
        }
        if (!(mlpMaxError > 0.0 && mlpMaxError <= 1.0))
        {
            throw new IllegalArgumentException("Max error must be greater than 0 and at most 1.");
        }
        if (mlpSparseEpochs < 1 || mlpSparseEpochs > 500)
        {
            throw new IllegalArgumentException("Sparse epochs must be between 1 and 500.");
        }
        if (mlpSparseL2 < 0.0 || mlpSparseL2 > 1.0)
        {
            throw new IllegalArgumentException("Sparse L2 must be between 0 and 1.");
        }
        if (mlpFedProxMu < 0.0 || mlpFedProxMu > 10.0)
        {
            throw new IllegalArgumentException("FedProx mu must be between 0 and 10.");
        }
    }

    boolean isCustomMlpEnabled() { return customMlpEnabled; }
    void setCustomMlpEnabled(boolean value) { customMlpEnabled = value; }
    String getMlpEngine() { return mlpEngine; }
    void setMlpEngine(String value) { mlpEngine = value; }
    int getMlpHiddenLayers() { return mlpHiddenLayers; }
    void setMlpHiddenLayers(int value) { mlpHiddenLayers = value; }
    int getMlpHiddenNeurons() { return mlpHiddenNeurons; }
    void setMlpHiddenNeurons(int value) { mlpHiddenNeurons = value; }
    double getMlpLearningRate() { return mlpLearningRate; }
    void setMlpLearningRate(double value) { mlpLearningRate = value; }
    double getMlpMaxError() { return mlpMaxError; }
    void setMlpMaxError(double value) { mlpMaxError = value; }
    int getMlpSparseEpochs() { return mlpSparseEpochs; }
    void setMlpSparseEpochs(int value) { mlpSparseEpochs = value; }
    double getMlpSparseL2() { return mlpSparseL2; }
    void setMlpSparseL2(double value) { mlpSparseL2 = value; }
    double getMlpFedProxMu() { return mlpFedProxMu; }
    void setMlpFedProxMu(double value) { mlpFedProxMu = value; }

    String getMlpEngineLabel()
    {
        return MLP_ENGINE_SPARSE_FEDERATED.equals(mlpEngine)
                ? "Sparse Federated MLP" : "Legacy Neuroph MLP";
    }

    String profileLabel()
    {
        if (!customMlpEnabled)
        {
            return "Default";
        }
        if (MLP_ENGINE_SPARSE_FEDERATED.equals(mlpEngine))
        {
            return "Sparse";
        }
        return mlpHiddenLayers + " layer" + (mlpHiddenLayers == 1 ? "" : "s");
    }

    String summaryForMlp()
    {
        AlgorithmParameterSettings effective = effective(this);
        return effective.getMlpEngineLabel()
                + ", " + effective.mlpHiddenLayers + " hidden layer(s)"
                + ", " + effective.mlpHiddenNeurons + " neuron(s)/layer"
                + ", rate " + effective.mlpLearningRate
                + ", max error " + effective.mlpMaxError;
    }

    private static int intProperty(String name, int defaultValue)
    {
        try { return Integer.parseInt(System.getProperty(name, String.valueOf(defaultValue))); }
        catch (RuntimeException ignored) { return defaultValue; }
    }

    private static double doubleProperty(String name, double defaultValue)
    {
        try { return Double.parseDouble(System.getProperty(name, String.valueOf(defaultValue))); }
        catch (RuntimeException ignored) { return defaultValue; }
    }
}
