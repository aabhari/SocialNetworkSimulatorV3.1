package TwitterGatherDataFollowers.userRyersonU;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Frame;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutionException;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.SwingConstants;
import javax.swing.SwingWorker;
import javax.swing.border.EmptyBorder;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.JTableHeader;

/**
 * Swing control surface for MLP-only automated hyperparameter search.
 */
final class MlpHyperparameterSearchDialog extends JDialog
{
	private static final long serialVersionUID = 1L;
	private static final Color INK = new Color(27, 38, 54);
	private static final Color MUTED = new Color(92, 105, 122);
	private static final Color PANEL = new Color(248, 250, 253);
	private static final Color LINE = new Color(218, 226, 236);
	private static final Color ACCENT = new Color(20, 135, 204);
	private static final Color ACCENT_DARK = new Color(13, 83, 128);
	private static final Color SUCCESS = new Color(23, 135, 84);
	private static final Color WARNING_BG = new Color(255, 244, 232);
	private static final Color BEST_BG = new Color(232, 248, 240);
	private static final int PROFILE_FAST = 0;
	private static final int PROFILE_BALANCED = 1;
	private static final int PROFILE_DEEP_RESCUE = 2;
	private static final int PROFILE_LEGACY_BOUNDED = 3;
	private static final int PROFILE_FULL_RESEARCH = 4;
	private static final int PROFILE_CUSTOM = 5;
	private static final int FULL_RESEARCH_PLANNED_RUNS = 28800;

	private final ControllerAgentGui owner;
	private final JTextField datasetField;
	private final JComboBox<String> profileBox;
	private final JCheckBox legacyBox;
	private final JCheckBox sparseBox;
	private final JTextField layersField;
	private final JTextField neuronsField;
	private final JTextField ratesField;
	private final JTextField maxErrorsField;
	private final JTextField maxIterationsField;
	private final JTextField sparseEpochsField;
	private final JTextField fedAvgRoundsField;
	private final JTextField l2Field;
	private final JTextField fedProxField;
	private final JTextField seedsField;
	private final JTextField validationField;
	private final JTextField testField;
	private final JLabel statusCard;
	private final JLabel bestCard;
	private final JLabel runsCard;
	private final JProgressBar progressBar;
	private final DefaultTableModel resultsModel;
	private final JTable resultsTable;
	private final JTextArea reportArea;
	private final JButton runButton;
	private final JButton cancelButton;
	private final JButton applyButton;
	private final JButton exportButton;

	private SearchWorker worker;
	private MlpHyperparameterSearch.SearchReport report;

	private MlpHyperparameterSearchDialog(
			ControllerAgentGui owner,
			File selectedDataset,
			AlgorithmParameterSettings currentSettings)
	{
		super((Frame)owner, "MLP Auto Hyperparameter Search", false);
		this.owner = owner;
		File defaultDataset = defaultDataset(selectedDataset);
		datasetField = new JTextField(defaultDataset == null
				? "" : defaultDataset.getAbsolutePath());
		profileBox = new JComboBox<String>(new String[] {
				"Fast bounded search",
				"Balanced staged search",
				"Deep rescue search",
				"Legacy Neuroph bounded",
				"Full research search (very long)",
				"Custom"
		});
		legacyBox = new JCheckBox("Legacy Neuroph", true);
		sparseBox = new JCheckBox("Sparse Federated", true);
		layersField = new JTextField();
		neuronsField = new JTextField();
		ratesField = new JTextField();
		maxErrorsField = new JTextField();
		maxIterationsField = new JTextField();
		sparseEpochsField = new JTextField();
		fedAvgRoundsField = new JTextField();
		l2Field = new JTextField();
		fedProxField = new JTextField();
		seedsField = new JTextField();
		validationField = new JTextField();
		testField = new JTextField();
		statusCard = metricLabel("Idle");
		bestCard = metricLabel("No result yet");
		runsCard = metricLabel("0 runs");
		progressBar = new JProgressBar(0, 100);
		progressBar.setStringPainted(true);
		progressBar.setString("Ready");
		resultsModel = new ResultsTableModel();
		resultsTable = new JTable(resultsModel);
		reportArea = new JTextArea(9, 40);
		runButton = primaryButton("Run Auto Search");
		cancelButton = secondaryButton("Cancel");
		applyButton = successButton("Apply Best Settings");
		exportButton = secondaryButton("Export CSV");

		buildDialog();
		applyProfile(0);
		loadCurrentHints(currentSettings);
		refreshButtons(false);
		setMinimumSize(new Dimension(1120, 720));
		setPreferredSize(new Dimension(1280, 820));
		pack();
		setLocationRelativeTo(owner);
	}

	static void showDialog(
			ControllerAgentGui owner,
			File selectedDataset,
			AlgorithmParameterSettings currentSettings)
	{
		MlpHyperparameterSearchDialog dialog =
				new MlpHyperparameterSearchDialog(owner, selectedDataset,
						currentSettings);
		dialog.setVisible(true);
	}

	private void buildDialog()
	{
		JPanel root = new JPanel(new BorderLayout(12, 12));
		root.setBorder(new EmptyBorder(14, 14, 14, 14));
		root.setBackground(new Color(238, 243, 249));
		root.add(buildHeader(), BorderLayout.NORTH);
		root.add(buildBody(), BorderLayout.CENTER);
		root.add(buildFooter(), BorderLayout.SOUTH);
		setContentPane(root);
	}

	private Component buildHeader()
	{
		JPanel header = new GradientPanel();
		header.setLayout(new BorderLayout(10, 6));
		header.setBorder(new EmptyBorder(18, 20, 18, 20));
		JLabel title = new JLabel("MLP Auto Hyperparameter Search");
		title.setForeground(Color.WHITE);
		title.setFont(new Font("Arial", Font.BOLD, 25));
		JLabel subtitle = new JLabel(
				"Automated, bounded, MLP-only search with validation ranking, collapse detection, timing, and exportable results.");
		subtitle.setForeground(new Color(226, 240, 255));
		subtitle.setFont(new Font("Arial", Font.PLAIN, 13));
		header.add(title, BorderLayout.NORTH);
		header.add(subtitle, BorderLayout.SOUTH);
		return header;
	}

	private Component buildBody()
	{
		JSplitPane split = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
		split.setResizeWeight(0.47);
		split.setBorder(BorderFactory.createEmptyBorder());
		JPanel top = new JPanel(new BorderLayout(12, 12));
		top.setOpaque(false);
		top.add(buildConfigurationPanel(), BorderLayout.CENTER);
		top.add(buildDashboard(), BorderLayout.EAST);
		split.setTopComponent(top);
		split.setBottomComponent(buildResultsPanel());
		return split;
	}

	private Component buildConfigurationPanel()
	{
		JPanel panel = cardPanel("Search Setup");
		panel.setLayout(new BorderLayout(10, 10));
		JPanel form = new JPanel(new GridBagLayout());
		form.setOpaque(false);
		GridBagConstraints c = new GridBagConstraints();
		c.insets = new Insets(5, 6, 5, 6);
		c.fill = GridBagConstraints.HORIZONTAL;
		c.weightx = 1.0;

		JButton browse = secondaryButton("Browse");
		browse.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) { chooseDataset(); }
		});
		JPanel datasetRow = new JPanel(new BorderLayout(8, 0));
		datasetRow.setOpaque(false);
		datasetRow.add(datasetField, BorderLayout.CENTER);
		datasetRow.add(browse, BorderLayout.EAST);
		addRow(form, c, 0, "Dataset", datasetRow);

		profileBox.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				int selectedProfile = profileBox.getSelectedIndex();
				applyProfile(selectedProfile);
				if (selectedProfile == PROFILE_FULL_RESEARCH)
				{
					showFullResearchWarning();
				}
			}
		});
		addRow(form, c, 1, "Profile", profileBox);

		JPanel enginePanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
		enginePanel.setOpaque(false);
		enginePanel.add(legacyBox);
		enginePanel.add(sparseBox);
		addRow(form, c, 2, "Engines", enginePanel);

		addRow(form, c, 3, "Hidden layers", layersField);
		addRow(form, c, 4, "Hidden neurons/layer", neuronsField);
		addRow(form, c, 5, "Learning rates", ratesField);
		addRow(form, c, 6, "Legacy max error", maxErrorsField);
		addRow(form, c, 7, "Legacy max iterations", maxIterationsField);
		addRow(form, c, 8, "Sparse epochs", sparseEpochsField);
		addRow(form, c, 9, "Sparse FedAvg rounds", fedAvgRoundsField);
		addRow(form, c, 10, "Sparse L2", l2Field);
		addRow(form, c, 11, "Sparse FedProx mu", fedProxField);
		addRow(form, c, 12, "Seeds", seedsField);

		JPanel splitPanel = new JPanel(new GridBagLayout());
		splitPanel.setOpaque(false);
		GridBagConstraints sc = new GridBagConstraints();
		sc.insets = new Insets(0, 0, 0, 6);
		sc.fill = GridBagConstraints.HORIZONTAL;
		sc.weightx = 1.0;
		sc.gridx = 0;
		splitPanel.add(new JLabel("Validation"), sc);
		sc.gridx = 1;
		splitPanel.add(validationField, sc);
		sc.gridx = 2;
		splitPanel.add(new JLabel("Test"), sc);
		sc.gridx = 3;
		splitPanel.add(testField, sc);
		addRow(form, c, 13, "Split fractions", splitPanel);

		JTextArea note = new JTextArea(
				"Ranking uses validation accuracy first. Test accuracy is shown for audit, not for selecting the best setting. Legacy Neuroph runs are forced to use a max-iteration cap during HPO.");
		note.setEditable(false);
		note.setLineWrap(true);
		note.setWrapStyleWord(true);
		note.setForeground(MUTED);
		note.setBackground(new Color(248, 250, 253));
		note.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
		panel.add(form, BorderLayout.CENTER);
		panel.add(note, BorderLayout.SOUTH);
		return panel;
	}

	private Component buildDashboard()
	{
		JPanel panel = new JPanel(new GridBagLayout());
		panel.setOpaque(false);
		panel.setPreferredSize(new Dimension(360, 260));
		GridBagConstraints c = new GridBagConstraints();
		c.gridx = 0;
		c.fill = GridBagConstraints.BOTH;
		c.weightx = 1.0;
		c.insets = new Insets(0, 0, 10, 0);
		c.gridy = 0;
		panel.add(metricCard("Status", statusCard), c);
		c.gridy = 1;
		panel.add(metricCard("Best Configuration", bestCard), c);
		c.gridy = 2;
		panel.add(metricCard("Progress", runsCard), c);
		c.gridy = 3;
		c.weighty = 1.0;
		JPanel progressPanel = cardPanel("Live Progress");
		progressPanel.setLayout(new BorderLayout(6, 6));
		progressPanel.add(progressBar, BorderLayout.NORTH);
		reportArea.setEditable(false);
		reportArea.setLineWrap(true);
		reportArea.setWrapStyleWord(true);
		reportArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
		reportArea.setBorder(new EmptyBorder(8, 8, 8, 8));
		progressPanel.add(new JScrollPane(reportArea), BorderLayout.CENTER);
		panel.add(progressPanel, c);
		return panel;
	}

	private Component buildResultsPanel()
	{
		JPanel panel = cardPanel("Ranked Results");
		panel.setLayout(new BorderLayout(8, 8));
		resultsTable.setAutoCreateRowSorter(true);
		resultsTable.setFillsViewportHeight(true);
		resultsTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		resultsTable.setRowHeight(26);
		resultsTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		resultsTable.setDefaultRenderer(Object.class, new ResultRenderer());
		resultsTable.setDefaultRenderer(Number.class, new ResultRenderer());
		resultsTable.setDefaultRenderer(Boolean.class, new ResultRenderer());
		JTableHeader header = resultsTable.getTableHeader();
		header.setFont(new Font("Arial", Font.BOLD, 12));
		header.setBackground(new Color(231, 238, 247));
		header.setForeground(INK);
		int[] widths = new int[] {
				46, 150, 58, 70, 72, 78, 82, 70, 70, 72,
				78, 90, 72, 90, 72, 82, 82, 96, 78, 84,
				130, 82, 280
		};
		for (int i = 0; i < widths.length; i++)
		{
			resultsTable.getColumnModel().getColumn(i).setPreferredWidth(widths[i]);
		}
		panel.add(new JScrollPane(resultsTable), BorderLayout.CENTER);
		return panel;
	}

	private Component buildFooter()
	{
		JPanel footer = new JPanel(new BorderLayout(10, 0));
		footer.setOpaque(false);
		JPanel left = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
		left.setOpaque(false);
		cancelButton.setEnabled(false);
		applyButton.setEnabled(false);
		exportButton.setEnabled(false);
		runButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) { runSearch(); }
		});
		cancelButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) { cancelSearch(); }
		});
		applyButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) { applyBest(); }
		});
		exportButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) { exportCsv(); }
		});
		left.add(runButton);
		left.add(cancelButton);
		left.add(applyButton);
		left.add(exportButton);
		JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
		right.setOpaque(false);
		JButton close = secondaryButton("Close");
		close.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) { dispose(); }
		});
		right.add(close);
		footer.add(left, BorderLayout.WEST);
		footer.add(right, BorderLayout.EAST);
		return footer;
	}

	private void chooseDataset()
	{
		JFileChooser chooser = new JFileChooser();
		File current = textFile(datasetField.getText());
		if (current != null)
		{
			chooser.setSelectedFile(current);
			if (current.getParentFile() != null)
			{
				chooser.setCurrentDirectory(current.getParentFile());
			}
		}
		int result = chooser.showOpenDialog(this);
		if (result == JFileChooser.APPROVE_OPTION)
		{
			datasetField.setText(chooser.getSelectedFile().getAbsolutePath());
		}
	}

	private void runSearch()
	{
		try
		{
			MlpHyperparameterSearch.SearchConfig config = buildConfigFromUi();
			int total = config.candidates().size();
			if (total <= 0)
			{
				throw new IllegalArgumentException("The selected grid is empty.");
			}
			if (total > 400)
			{
				int answer = JOptionPane.showConfirmDialog(this,
						"This search will run " + total
						+ " MLP configurations.\n\n"
						+ "Large research searches can take hours or days. "
						+ "Legacy Neuroph is bounded by the max-iteration "
						+ "values shown, but a long candidate may need to "
						+ "finish before Cancel takes effect.\n\nContinue?",
						"Large HPO Search",
						JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				if (answer != JOptionPane.OK_OPTION)
				{
					return;
				}
			}
			resultsModel.setRowCount(0);
			report = null;
			progressBar.setValue(0);
			progressBar.setIndeterminate(true);
			progressBar.setString("Loading dataset");
			reportArea.setText("");
			statusCard.setText("Preparing search");
			bestCard.setText("No result yet");
			runsCard.setText(total + " planned run(s)");
			refreshButtons(true);
			worker = new SearchWorker(config);
			worker.execute();
		}
		catch (RuntimeException ex)
		{
			JOptionPane.showMessageDialog(this, ex.getMessage(),
					"Invalid MLP Search Setup", JOptionPane.ERROR_MESSAGE);
		}
	}

	private void cancelSearch()
	{
		if (worker != null)
		{
			worker.cancel(true);
			statusCard.setText("Cancelling...");
			progressBar.setString("Cancelling");
		}
	}

	private void applyBest()
	{
		if (report == null || report.best == null)
		{
			JOptionPane.showMessageDialog(this,
					"No successful HPO result is available yet.",
					"No Best Configuration",
					JOptionPane.WARNING_MESSAGE);
			return;
		}
		AlgorithmParameterSettings settings =
				report.best.toAlgorithmParameterSettings();
		owner.applyMlpAutoSearchSettings(settings, report.best.summary());
		JOptionPane.showMessageDialog(this,
				"Applied best MLP settings:\n" + settings.summaryForMlp(),
				"MLP Settings Applied",
				JOptionPane.INFORMATION_MESSAGE);
	}

	private void exportCsv()
	{
		if (report == null)
		{
			return;
		}
		JFileChooser chooser = new JFileChooser();
		chooser.setSelectedFile(new File("mlp-hpo-results.csv"));
		int result = chooser.showSaveDialog(this);
		if (result != JFileChooser.APPROVE_OPTION)
		{
			return;
		}
		try
		{
			MlpHyperparameterSearch.exportCsv(chooser.getSelectedFile(), report);
			JOptionPane.showMessageDialog(this,
					"Saved MLP HPO results to:\n"
					+ chooser.getSelectedFile().getAbsolutePath(),
					"CSV Exported",
					JOptionPane.INFORMATION_MESSAGE);
		}
		catch (Exception ex)
		{
			JOptionPane.showMessageDialog(this,
					"Could not export CSV:\n" + ex.getMessage(),
					"Export Failed",
					JOptionPane.ERROR_MESSAGE);
		}
	}

	private MlpHyperparameterSearch.SearchConfig buildConfigFromUi()
	{
		MlpHyperparameterSearch.SearchConfig config =
				new MlpHyperparameterSearch.SearchConfig();
		config.datasetFile = new File(datasetField.getText().trim());
		config.includeLegacy = legacyBox.isSelected();
		config.includeSparse = sparseBox.isSelected();
		config.hiddenLayers = parseIntList(layersField.getText(),
				"Hidden layers");
		config.hiddenNeurons = parseIntList(neuronsField.getText(),
				"Hidden neurons");
		config.learningRates = parseDoubleList(ratesField.getText(),
				"Learning rates");
		config.maxErrors = parseDoubleList(maxErrorsField.getText(),
				"Legacy max errors");
		config.legacyMaxIterations = parseIntList(maxIterationsField.getText(),
				"Legacy max iterations");
		config.sparseEpochs = parseIntList(sparseEpochsField.getText(),
				"Sparse epochs");
		config.fedAvgRounds = parseIntList(fedAvgRoundsField.getText(),
				"Sparse FedAvg rounds");
		config.sparseL2Values = parseDoubleList(l2Field.getText(),
				"Sparse L2 values");
		config.fedProxMuValues = parseDoubleList(fedProxField.getText(),
				"Sparse FedProx mu values");
		config.seeds = parseLongList(seedsField.getText(), "Seeds");
		config.validationPercent = parseFraction(validationField.getText(),
				"Validation fraction");
		config.testPercent = parseFraction(testField.getText(),
				"Test fraction");
		config.validate();
		return config;
	}

	private void applyProfile(int index)
	{
		if (index == PROFILE_CUSTOM)
		{
			return;
		}
		if (index == PROFILE_FAST)
		{
			legacyBox.setSelected(true);
			sparseBox.setSelected(true);
			layersField.setText("1,3,5");
			neuronsField.setText("10,25");
			ratesField.setText("0.1,0.03,0.01");
			maxErrorsField.setText("0.01");
			maxIterationsField.setText("50");
			sparseEpochsField.setText("80,160");
			fedAvgRoundsField.setText("16");
			l2Field.setText("0.0001");
			fedProxField.setText("0.0");
			seedsField.setText(String.valueOf(
					SparseFederatedMlpModelSupport.GLOBAL_INITIALIZATION_SEED));
		}
		else if (index == PROFILE_BALANCED)
		{
			legacyBox.setSelected(true);
			sparseBox.setSelected(true);
			layersField.setText("1,3,5");
			neuronsField.setText("10,25,50");
			ratesField.setText("0.1,0.03,0.01");
			maxErrorsField.setText("0.01");
			maxIterationsField.setText("50,100");
			sparseEpochsField.setText("80,160");
			fedAvgRoundsField.setText("16");
			l2Field.setText("0.0001");
			fedProxField.setText("0.0");
			seedsField.setText("20260708,42");
		}
		else if (index == PROFILE_DEEP_RESCUE)
		{
			legacyBox.setSelected(false);
			sparseBox.setSelected(true);
			layersField.setText("3,5,8");
			neuronsField.setText("25,50,100");
			ratesField.setText("0.03,0.01,0.005");
			maxErrorsField.setText("0.01");
			maxIterationsField.setText("100");
			sparseEpochsField.setText("160,300");
			fedAvgRoundsField.setText("16");
			l2Field.setText("0.0001,0.001");
			fedProxField.setText("0.0");
			seedsField.setText("20260708,42");
		}
		else if (index == PROFILE_LEGACY_BOUNDED)
		{
			legacyBox.setSelected(true);
			sparseBox.setSelected(false);
			layersField.setText("1,3,5");
			neuronsField.setText("10,25");
			ratesField.setText("0.1,0.03,0.01");
			maxErrorsField.setText("0.01");
			maxIterationsField.setText("25,50,100");
			sparseEpochsField.setText("80");
			fedAvgRoundsField.setText("16");
			l2Field.setText("0.0001");
			fedProxField.setText("0.0");
			seedsField.setText("20260708");
		}
		else if (index == PROFILE_FULL_RESEARCH)
		{
			legacyBox.setSelected(true);
			sparseBox.setSelected(true);
			layersField.setText("1,2,3,4,5,6,7,8");
			neuronsField.setText("10,25,50,75,100");
			ratesField.setText("0.1,0.05,0.03,0.01,0.005,0.001");
			maxErrorsField.setText("0.01,0.005");
			maxIterationsField.setText("25,50,100,250,500,1000,2000,4000");
			sparseEpochsField.setText("80,160,300,500");
			fedAvgRoundsField.setText("16");
			l2Field.setText("0.0,0.0001,0.001");
			fedProxField.setText("0.0,0.001");
			seedsField.setText("20260708,42,20260719");
		}
		validationField.setText("0.20");
		testField.setText("0.20");
	}

	private void showFullResearchWarning()
	{
		JOptionPane.showMessageDialog(this,
				"Full research search is intentionally exhaustive and may "
				+ "take a very long time.\n\n"
				+ "This preset plans about " + FULL_RESEARCH_PLANNED_RUNS
				+ " bounded MLP configurations across Legacy Neuroph and "
				+ "Sparse Federated MLP.\n\n"
				+ "Legacy Neuroph is still protected by hard max-iteration "
				+ "caps up to 4000 iterations, but some candidates may still "
				+ "take a long time. Use this for overnight/deep research "
				+ "runs, not quick checks.",
				"Full Research HPO Warning",
				JOptionPane.WARNING_MESSAGE);
	}

	private void loadCurrentHints(AlgorithmParameterSettings current)
	{
		if (current == null)
		{
			return;
		}
		if (current.isCustomMlpEnabled())
		{
			statusCard.setText("Current: " + current.profileLabel());
		}
	}

	private void refreshButtons(boolean running)
	{
		runButton.setEnabled(!running);
		cancelButton.setEnabled(running);
		applyButton.setEnabled(!running && report != null && report.best != null);
		exportButton.setEnabled(!running && report != null);
		setCursor(Cursor.getPredefinedCursor(
				running ? Cursor.WAIT_CURSOR : Cursor.DEFAULT_CURSOR));
	}

	private void addDatasetProfile(MlpHyperparameterSearch.DataProfile profile)
	{
		reportArea.append(profile.summary());
		reportArea.append("\nValidation distribution: ");
		reportArea.append(profile.validationDistribution());
		reportArea.append("\n\n");
		statusCard.setText("Dataset loaded: " + profile.userCount
				+ " users, " + profile.featureCount + " features");
	}

	private void addResultRow(MlpHyperparameterSearch.SearchResult result,
			int rank)
	{
		resultsModel.addRow(rowFor(result, rank));
	}

	private Object[] rowFor(MlpHyperparameterSearch.SearchResult result,
			int rank)
	{
		MlpHyperparameterSearch.Candidate candidate = result.candidate;
		return new Object[] {
				Integer.valueOf(rank),
				candidate.engineLabel(),
				Integer.valueOf(candidate.hiddenLayers),
				Integer.valueOf(candidate.hiddenNeurons),
				Double.valueOf(candidate.learningRate),
				Double.valueOf(candidate.maxError),
				Integer.valueOf(candidate.maxIterations),
				Integer.valueOf(candidate.sparseEpochs),
				Integer.valueOf(candidate.fedAvgRounds),
				Double.valueOf(candidate.sparseL2),
				Double.valueOf(candidate.fedProxMu),
				Long.valueOf(candidate.seed),
				Double.valueOf(result.trainAccuracy),
				Double.valueOf(result.validationAccuracy),
				Double.valueOf(result.testAccuracy),
				Long.valueOf(result.trainMillis),
				Long.valueOf(result.totalMillis),
				Integer.valueOf(result.predictedClasses),
				result.collapsed ? "Yes" : "No",
				Long.valueOf(Math.round(result.modelPayloadBytes / 1024.0)),
				result.trainingMode,
				result.status,
				result.note
		};
	}

	private void replaceWithSortedResults()
	{
		resultsModel.setRowCount(0);
		if (report == null)
		{
			return;
		}
		int rank = 1;
		for (MlpHyperparameterSearch.SearchResult result : report.results)
		{
			addResultRow(result, rank++);
		}
	}

	private void completeReport()
	{
		if (report == null)
		{
			return;
		}
		reportArea.append("Search completed in ");
		reportArea.append(String.valueOf(report.elapsedMillis));
		reportArea.append(" ms.\n");
		reportArea.append("Completed runs: ");
		reportArea.append(String.valueOf(report.results.size()));
		reportArea.append("\n");
		if (report.best != null)
		{
			reportArea.append("Best setting: ");
			reportArea.append(report.best.summary());
			reportArea.append("\n");
			bestCard.setText("<html>" + html(report.best.candidate.engineLabel())
					+ "<br>" + report.best.candidate.hiddenLayers + " layer(s), "
					+ report.best.candidate.hiddenNeurons + " neuron(s)"
					+ "<br>Val " + fmt(report.best.validationAccuracy)
					+ "% | Test " + fmt(report.best.testAccuracy) + "%</html>");
		}
		else
		{
			bestCard.setText("No successful run");
		}
		statusCard.setText("Search complete");
		runsCard.setText(report.results.size() + " completed run(s)");
	}

	private void addRow(
			JPanel panel,
			GridBagConstraints c,
			int row,
			String label,
			Component field)
	{
		c.gridy = row;
		c.gridx = 0;
		c.weightx = 0.0;
		JLabel jLabel = new JLabel(label);
		jLabel.setFont(new Font("Arial", Font.BOLD, 12));
		jLabel.setForeground(INK);
		panel.add(jLabel, c);
		c.gridx = 1;
		c.weightx = 1.0;
		panel.add(field, c);
	}

	private JPanel cardPanel(String title)
	{
		JPanel panel = new JPanel();
		panel.setBackground(PANEL);
		panel.setBorder(BorderFactory.createCompoundBorder(
				BorderFactory.createLineBorder(LINE),
				new EmptyBorder(12, 12, 12, 12)));
		if (title != null)
		{
			panel.setBorder(BorderFactory.createTitledBorder(
					panel.getBorder(), title));
		}
		return panel;
	}

	private Component metricCard(String title, JLabel value)
	{
		JPanel panel = cardPanel(null);
		panel.setLayout(new BorderLayout(4, 4));
		JLabel titleLabel = new JLabel(title.toUpperCase());
		titleLabel.setForeground(MUTED);
		titleLabel.setFont(new Font("Arial", Font.BOLD, 11));
		panel.add(titleLabel, BorderLayout.NORTH);
		panel.add(value, BorderLayout.CENTER);
		return panel;
	}

	private JLabel metricLabel(String text)
	{
		JLabel label = new JLabel(text);
		label.setForeground(INK);
		label.setFont(new Font("Arial", Font.BOLD, 14));
		label.setVerticalAlignment(SwingConstants.TOP);
		return label;
	}

	private JButton primaryButton(String label)
	{
		JButton button = baseButton(label);
		button.setBackground(ACCENT);
		button.setForeground(Color.WHITE);
		return button;
	}

	private JButton successButton(String label)
	{
		JButton button = baseButton(label);
		button.setBackground(SUCCESS);
		button.setForeground(Color.WHITE);
		return button;
	}

	private JButton secondaryButton(String label)
	{
		JButton button = baseButton(label);
		button.setBackground(new Color(230, 236, 244));
		button.setForeground(INK);
		return button;
	}

	private JButton baseButton(String label)
	{
		JButton button = new JButton(label);
		button.setFont(new Font("Arial", Font.BOLD, 12));
		button.setFocusPainted(false);
		button.setBorder(BorderFactory.createCompoundBorder(
				BorderFactory.createLineBorder(new Color(185, 198, 214)),
				new EmptyBorder(6, 12, 6, 12)));
		return button;
	}

	private static File defaultDataset(File selected)
	{
		if (selected != null && selected.isFile())
		{
			return selected;
		}
		File retail = new File("Dataset/Retail.txt");
		if (retail.isFile())
		{
			return retail.getAbsoluteFile();
		}
		return null;
	}

	private static File textFile(String text)
	{
		if (text == null || text.trim().length() == 0)
		{
			return null;
		}
		return new File(text.trim());
	}

	private static int[] parseIntList(String text, String label)
	{
		String[] parts = parts(text, label);
		int[] values = new int[parts.length];
		for (int i = 0; i < parts.length; i++)
		{
			values[i] = Integer.parseInt(parts[i]);
		}
		return values;
	}

	private static long[] parseLongList(String text, String label)
	{
		String[] parts = parts(text, label);
		long[] values = new long[parts.length];
		for (int i = 0; i < parts.length; i++)
		{
			values[i] = Long.parseLong(parts[i]);
		}
		return values;
	}

	private static double[] parseDoubleList(String text, String label)
	{
		String[] parts = parts(text, label);
		double[] values = new double[parts.length];
		for (int i = 0; i < parts.length; i++)
		{
			values[i] = Double.parseDouble(parts[i]);
		}
		return values;
	}

	private static String[] parts(String text, String label)
	{
		if (text == null || text.trim().length() == 0)
		{
			throw new IllegalArgumentException(label + " cannot be empty.");
		}
		return text.trim().split("[,;\\s]+");
	}

	private static double parseFraction(String text, String label)
	{
		try
		{
			double value = Double.parseDouble(text.trim());
			if (!(value >= 0.0 && value < 1.0))
			{
				throw new NumberFormatException();
			}
			return value;
		}
		catch (RuntimeException ex)
		{
			throw new IllegalArgumentException(
					label + " must be a decimal between 0 and 1.");
		}
	}

	private static String fmt(double value)
	{
		return String.format(java.util.Locale.US, "%.2f",
				Double.valueOf(value));
	}

	private static String html(String value)
	{
		if (value == null)
		{
			return "";
		}
		return value.replace("&", "&amp;").replace("<", "&lt;")
				.replace(">", "&gt;");
	}

	private final class SearchWorker extends SwingWorker
			<MlpHyperparameterSearch.SearchReport, UiEvent>
			implements MlpHyperparameterSearch.ProgressListener
	{
		private final MlpHyperparameterSearch.SearchConfig config;

		SearchWorker(MlpHyperparameterSearch.SearchConfig config)
		{
			this.config = config;
		}

		protected MlpHyperparameterSearch.SearchReport doInBackground()
				throws Exception
		{
			return MlpHyperparameterSearch.run(config, this);
		}

		public void onMessage(String message)
		{
			publish(UiEvent.message(message));
		}

		public void onDatasetLoaded(MlpHyperparameterSearch.DataProfile profile)
		{
			publish(UiEvent.dataset(profile));
		}

		public void onCandidateStarted(
				MlpHyperparameterSearch.Candidate candidate,
				int completed,
				int total)
		{
			publish(UiEvent.started(candidate, completed, total));
		}

		public void onCandidateFinished(
				MlpHyperparameterSearch.SearchResult result,
				int completed,
				int total)
		{
			publish(UiEvent.finished(result, completed, total));
		}

		protected void process(List<UiEvent> events)
		{
			for (UiEvent event : events)
			{
				if (event.message != null)
				{
					reportArea.append(event.message);
					reportArea.append("\n");
				}
				if (event.profile != null)
				{
					progressBar.setIndeterminate(false);
					progressBar.setValue(0);
					addDatasetProfile(event.profile);
				}
				if (event.startedCandidate != null)
				{
					statusCard.setText(event.startedCandidate.compactLabel());
					progressBar.setIndeterminate(false);
					progressBar.setMaximum(Math.max(1, event.total));
					progressBar.setValue(event.completed);
					progressBar.setString(event.completed + " / "
							+ event.total);
				}
				if (event.result != null)
				{
					addResultRow(event.result, resultsModel.getRowCount() + 1);
					progressBar.setMaximum(Math.max(1, event.total));
					progressBar.setValue(event.completed);
					progressBar.setString(event.completed + " / "
							+ event.total);
					runsCard.setText(event.completed + " / "
							+ event.total + " completed");
				}
			}
		}

		protected void done()
		{
			try
			{
				report = get();
				replaceWithSortedResults();
				completeReport();
			}
			catch (InterruptedException ex)
			{
				Thread.currentThread().interrupt();
				statusCard.setText("Interrupted");
			}
			catch (ExecutionException ex)
			{
				Throwable cause = ex.getCause() == null ? ex : ex.getCause();
				statusCard.setText("Search failed");
				JOptionPane.showMessageDialog(MlpHyperparameterSearchDialog.this,
						cause.getMessage(),
						"MLP HPO Failed",
						JOptionPane.ERROR_MESSAGE);
			}
			catch (RuntimeException ex)
			{
				statusCard.setText("Cancelled");
			}
			finally
			{
				progressBar.setIndeterminate(false);
				if (report == null)
				{
					progressBar.setString("Stopped");
				}
				else
				{
					progressBar.setValue(progressBar.getMaximum());
					progressBar.setString("Complete");
				}
				refreshButtons(false);
				worker = null;
			}
		}
	}

	private static final class UiEvent
	{
		final String message;
		final MlpHyperparameterSearch.DataProfile profile;
		final MlpHyperparameterSearch.Candidate startedCandidate;
		final MlpHyperparameterSearch.SearchResult result;
		final int completed;
		final int total;

		private UiEvent(
				String message,
				MlpHyperparameterSearch.DataProfile profile,
				MlpHyperparameterSearch.Candidate startedCandidate,
				MlpHyperparameterSearch.SearchResult result,
				int completed,
				int total)
		{
			this.message = message;
			this.profile = profile;
			this.startedCandidate = startedCandidate;
			this.result = result;
			this.completed = completed;
			this.total = total;
		}

		static UiEvent message(String message)
		{
			return new UiEvent(message, null, null, null, 0, 0);
		}

		static UiEvent dataset(MlpHyperparameterSearch.DataProfile profile)
		{
			return new UiEvent(null, profile, null, null, 0, 0);
		}

		static UiEvent started(
				MlpHyperparameterSearch.Candidate candidate,
				int completed,
				int total)
		{
			return new UiEvent(null, null, candidate, null, completed, total);
		}

		static UiEvent finished(
				MlpHyperparameterSearch.SearchResult result,
				int completed,
				int total)
		{
			return new UiEvent(null, null, null, result, completed, total);
		}
	}

	private static final class ResultsTableModel extends DefaultTableModel
	{
		private static final long serialVersionUID = 1L;
		private final String[] columns = new String[] {
				"Rank", "Engine", "Layers", "Neurons", "LR", "Max Err",
				"Max Iter", "Epochs", "Rounds", "L2", "FedProx",
				"Seed", "Train %", "Validation %", "Test %", "Train ms",
				"Total ms", "Pred Classes", "Collapsed", "Payload KB",
				"Mode", "Status", "Note"
		};
		private final Class<?>[] classes = new Class<?>[] {
				Integer.class, String.class, Integer.class, Integer.class,
				Double.class, Double.class, Integer.class, Integer.class,
				Integer.class, Double.class, Double.class, Long.class,
				Double.class, Double.class, Double.class, Long.class,
				Long.class, Integer.class, String.class, Long.class,
				String.class, String.class, String.class
		};

		ResultsTableModel()
		{
			super();
			setColumnIdentifiers(columns);
		}

		public boolean isCellEditable(int row, int column)
		{
			return false;
		}

		public Class<?> getColumnClass(int columnIndex)
		{
			return classes[columnIndex];
		}
	}

	private final class ResultRenderer extends DefaultTableCellRenderer
	{
		private static final long serialVersionUID = 1L;

		public Component getTableCellRendererComponent(
				JTable table,
				Object value,
				boolean isSelected,
				boolean hasFocus,
				int row,
				int column)
		{
			Component component = super.getTableCellRendererComponent(table,
					value, isSelected, hasFocus, row, column);
			int modelRow = table.convertRowIndexToModel(row);
			Object rank = table.getModel().getValueAt(modelRow, 0);
			Object collapsed = table.getModel().getValueAt(modelRow, 18);
			if (!isSelected)
			{
				if (rank instanceof Integer
						&& ((Integer)rank).intValue() == 1)
				{
					component.setBackground(BEST_BG);
				}
				else if ("Yes".equals(String.valueOf(collapsed)))
				{
					component.setBackground(WARNING_BG);
				}
				else
				{
					component.setBackground(row % 2 == 0
							? Color.WHITE : new Color(246, 249, 253));
				}
				component.setForeground(INK);
			}
			if (value instanceof Number)
			{
				setHorizontalAlignment(SwingConstants.RIGHT);
				if (value instanceof Double)
				{
					setText(fmt(((Double)value).doubleValue()));
				}
			}
			else
			{
				setHorizontalAlignment(SwingConstants.LEFT);
			}
			setBorder(BorderFactory.createEmptyBorder(0, 6, 0, 6));
			return component;
		}
	}

	private static final class GradientPanel extends JPanel
	{
		private static final long serialVersionUID = 1L;

		protected void paintComponent(Graphics g)
		{
			Graphics2D g2 = (Graphics2D)g.create();
			try
			{
				g2.setPaint(new GradientPaint(0, 0, ACCENT_DARK,
						getWidth(), getHeight(), ACCENT));
				g2.fillRect(0, 0, getWidth(), getHeight());
			}
			finally
			{
				g2.dispose();
			}
			super.paintComponent(g);
		}

		public boolean isOpaque()
		{
			return false;
		}
	}
}
