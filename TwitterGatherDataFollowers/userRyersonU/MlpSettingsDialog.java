package TwitterGatherDataFollowers.userRyersonU;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.SpinnerNumberModel;

/** MLP-only settings dialog for the old v2.5 UI. */
final class MlpSettingsDialog extends JDialog
{
    private static final long serialVersionUID = 1L;

    private final JCheckBox customEnabled;
    private final JComboBox<String> engineBox;
    private final JSpinner hiddenLayers;
    private final JSpinner hiddenNeurons;
    private final JSpinner learningRate;
    private final JSpinner maxError;
    private final JSpinner sparseEpochs;
    private final JSpinner fedAvgRounds;
    private final JSpinner sparseL2;
    private final JSpinner fedProxMu;
    private AlgorithmParameterSettings result;

    private MlpSettingsDialog(Frame owner, AlgorithmParameterSettings current)
    {
        super(owner, "MLP Advanced Settings", true);
        AlgorithmParameterSettings settings = current == null
                ? AlgorithmParameterSettings.defaults() : current.copy();

        customEnabled = new JCheckBox("Use custom MLP settings", settings.isCustomMlpEnabled());
        engineBox = new JComboBox<String>(new String[] {
                "Legacy Neuroph MLP",
                "Sparse Federated MLP (experimental)"
        });
        engineBox.setSelectedIndex(
                AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED.equals(settings.getMlpEngine()) ? 1 : 0);
        hiddenLayers = new JSpinner(new SpinnerNumberModel(settings.getMlpHiddenLayers(), 1, 8, 1));
        hiddenNeurons = new JSpinner(new SpinnerNumberModel(settings.getMlpHiddenNeurons(), 1, 512, 1));
        learningRate = new JSpinner(new SpinnerNumberModel(settings.getMlpLearningRate(), 0.0001, 1.0, 0.01));
        maxError = new JSpinner(new SpinnerNumberModel(settings.getMlpMaxError(), 0.000001, 1.0, 0.001));
        sparseEpochs = new JSpinner(new SpinnerNumberModel(settings.getMlpSparseEpochs(), 1, 500, 1));
        fedAvgRounds = new JSpinner(new SpinnerNumberModel(settings.getMlpFedAvgRounds(), 1, 100, 1));
        sparseL2 = new JSpinner(new SpinnerNumberModel(settings.getMlpSparseL2(), 0.0, 1.0, 0.0001));
        fedProxMu = new JSpinner(new SpinnerNumberModel(settings.getMlpFedProxMu(), 0.0, 10.0, 0.01));
        engineBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent event) { refreshEngineFields(); }
        });
        refreshEngineFields();

        JPanel root = new JPanel(new BorderLayout(12, 12));
        root.setBorder(BorderFactory.createEmptyBorder(14, 16, 14, 16));
        root.add(buildHeader(), BorderLayout.NORTH);
        root.add(buildForm(), BorderLayout.CENTER);
        root.add(buildButtons(), BorderLayout.SOUTH);
        setContentPane(root);
        pack();
        setMinimumSize(new Dimension(560, 500));
        setLocationRelativeTo(owner);
    }

    static AlgorithmParameterSettings showDialog(Frame owner, AlgorithmParameterSettings current)
    {
        MlpSettingsDialog dialog = new MlpSettingsDialog(owner, current);
        dialog.setVisible(true);
        return dialog.result;
    }

    private Component buildHeader()
    {
        JTextArea text = new JTextArea(
                "Default keeps the original v2.5 Neuroph behavior: one hidden layer, "
                + "10 hidden neurons, learning rate 0.1, max error 0.01.\n"
                + "Sparse Federated MLP is opt-in for large sparse TF-IDF data. "
                + "Sparse uses hidden layers, hidden neurons, learning rate, epochs, FedAvg rounds, L2, and FedProx; "
                + "its default is 80 shuffled epochs over 16 FedAvg rounds, and max error is Legacy-only. "
                + "Sparse configurations with 4 or more hidden layers use depth-stable leaky-ReLU training "
                + "and cap the effective learning rate at 0.03.");
        text.setEditable(false);
        text.setOpaque(true);
        text.setBackground(new Color(245, 248, 252));
        text.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(new Color(210, 220, 235)),
                BorderFactory.createEmptyBorder(10, 10, 10, 10)));
        text.setLineWrap(true);
        text.setWrapStyleWord(true);
        return text;
    }

    private Component buildForm()
    {
        JPanel form = new JPanel(new GridBagLayout());
        form.setBorder(BorderFactory.createTitledBorder("MLP Parameters"));
        GridBagConstraints c = new GridBagConstraints();
        c.insets = new Insets(6, 8, 6, 8);
        c.fill = GridBagConstraints.HORIZONTAL;
        c.weightx = 1.0;
        addRow(form, c, 0, "Enable", customEnabled);
        addRow(form, c, 1, "Engine", engineBox);
        addRow(form, c, 2, "Hidden layers", hiddenLayers);
        addRow(form, c, 3, "Hidden neurons per layer", hiddenNeurons);
        addRow(form, c, 4, "Learning rate", learningRate);
        addRow(form, c, 5, "Max error", maxError);
        addRow(form, c, 6, "Sparse epochs", sparseEpochs);
        addRow(form, c, 7, "Sparse FedAvg rounds", fedAvgRounds);
        addRow(form, c, 8, "Sparse L2", sparseL2);
        addRow(form, c, 9, "Sparse FedProx mu", fedProxMu);
        return form;
    }

    private void addRow(JPanel panel, GridBagConstraints c, int row, String label, Component field)
    {
        c.gridx = 0;
        c.gridy = row;
        c.weightx = 0.0;
        panel.add(new JLabel(label), c);
        c.gridx = 1;
        c.weightx = 1.0;
        panel.add(field, c);
    }

    private Component buildButtons()
    {
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton reset = new JButton("Reset Defaults");
        JButton cancel = new JButton("Cancel");
        JButton apply = new JButton("Apply");
        reset.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent event) { load(AlgorithmParameterSettings.defaults()); }
        });
        cancel.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent event) { result = null; dispose(); }
        });
        apply.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent event) { applyAndClose(); }
        });
        buttons.add(reset);
        buttons.add(cancel);
        buttons.add(apply);
        return buttons;
    }

    private void load(AlgorithmParameterSettings settings)
    {
        customEnabled.setSelected(settings.isCustomMlpEnabled());
        engineBox.setSelectedIndex(
                AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED.equals(settings.getMlpEngine()) ? 1 : 0);
        hiddenLayers.setValue(Integer.valueOf(settings.getMlpHiddenLayers()));
        hiddenNeurons.setValue(Integer.valueOf(settings.getMlpHiddenNeurons()));
        learningRate.setValue(Double.valueOf(settings.getMlpLearningRate()));
        maxError.setValue(Double.valueOf(settings.getMlpMaxError()));
        sparseEpochs.setValue(Integer.valueOf(settings.getMlpSparseEpochs()));
        fedAvgRounds.setValue(Integer.valueOf(settings.getMlpFedAvgRounds()));
        sparseL2.setValue(Double.valueOf(settings.getMlpSparseL2()));
        fedProxMu.setValue(Double.valueOf(settings.getMlpFedProxMu()));
        refreshEngineFields();
    }

    private void refreshEngineFields()
    {
        boolean sparse = engineBox.getSelectedIndex() == 1;
        maxError.setEnabled(!sparse);
        sparseEpochs.setEnabled(sparse);
        fedAvgRounds.setEnabled(sparse);
        sparseL2.setEnabled(sparse);
        fedProxMu.setEnabled(sparse);
        maxError.setToolTipText(sparse
                ? "Max error is used by Legacy Neuroph MLP. Sparse Federated MLP uses Sparse epochs instead."
                : "Legacy Neuroph MLP stopping error.");
        sparseEpochs.setToolTipText(sparse
                ? "Total number of local Sparse Federated MLP training epochs distributed across FedAvg rounds."
                : "Only used by Sparse Federated MLP.");
        fedAvgRounds.setToolTipText(sparse
                ? "Number of iterative FedAvg synchronization rounds. Total local epochs are split across these rounds."
                : "Only used by Sparse Federated MLP.");
        sparseL2.setToolTipText(sparse
                ? "L2 regularization used by Sparse Federated MLP."
                : "Only used by Sparse Federated MLP.");
        fedProxMu.setToolTipText(sparse
                ? "FedProx proximal regularization used by Sparse Federated MLP."
                : "Only used by Sparse Federated MLP.");
    }

    private void applyAndClose()
    {
        AlgorithmParameterSettings settings = AlgorithmParameterSettings.defaults();
        settings.setCustomMlpEnabled(customEnabled.isSelected());
        settings.setMlpEngine(engineBox.getSelectedIndex() == 1
                ? AlgorithmParameterSettings.MLP_ENGINE_SPARSE_FEDERATED
                : AlgorithmParameterSettings.MLP_ENGINE_LEGACY_NEUROPH);
        settings.setMlpHiddenLayers(((Number)hiddenLayers.getValue()).intValue());
        settings.setMlpHiddenNeurons(((Number)hiddenNeurons.getValue()).intValue());
        settings.setMlpLearningRate(((Number)learningRate.getValue()).doubleValue());
        settings.setMlpMaxError(((Number)maxError.getValue()).doubleValue());
        settings.setMlpSparseEpochs(((Number)sparseEpochs.getValue()).intValue());
        settings.setMlpFedAvgRounds(((Number)fedAvgRounds.getValue()).intValue());
        settings.setMlpSparseL2(((Number)sparseL2.getValue()).doubleValue());
        settings.setMlpFedProxMu(((Number)fedProxMu.getValue()).doubleValue());
        settings.setCustomMlpEnabled(customEnabled.isSelected() || differsFromDefaults(settings));
        try
        {
            settings.validateForMlp();
            result = settings;
            dispose();
        }
        catch (IllegalArgumentException ex)
        {
            JOptionPane.showMessageDialog(this, ex.getMessage(),
                    "Invalid MLP Settings", JOptionPane.ERROR_MESSAGE);
        }
    }

    private boolean differsFromDefaults(AlgorithmParameterSettings settings)
    {
        AlgorithmParameterSettings defaults = AlgorithmParameterSettings.defaults();
        return !defaults.getMlpEngine().equals(settings.getMlpEngine())
                || defaults.getMlpHiddenLayers() != settings.getMlpHiddenLayers()
                || defaults.getMlpHiddenNeurons() != settings.getMlpHiddenNeurons()
                || Double.compare(defaults.getMlpLearningRate(), settings.getMlpLearningRate()) != 0
                || Double.compare(defaults.getMlpMaxError(), settings.getMlpMaxError()) != 0
                || defaults.getMlpSparseEpochs() != settings.getMlpSparseEpochs()
                || defaults.getMlpFedAvgRounds() != settings.getMlpFedAvgRounds()
                || Double.compare(defaults.getMlpSparseL2(), settings.getMlpSparseL2()) != 0
                || Double.compare(defaults.getMlpFedProxMu(), settings.getMlpFedProxMu()) != 0;
    }
}
