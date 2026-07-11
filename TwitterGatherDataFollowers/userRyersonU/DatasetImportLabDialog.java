package TwitterGatherDataFollowers.userRyersonU;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.ScrollPaneConstants;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import javax.swing.border.Border;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.JTableHeader;

final class DatasetImportLabDialog extends JDialog {
	private static final long serialVersionUID = 1L;

	private static final Color DSMP_BLUE = new Color(0, 168, 239);
	private static final Color WHITE = Color.WHITE;
	private static final Color INK = Color.BLACK;
	private static final Color LINE = Color.BLACK;
	private static final Font LABEL_FONT = new Font("Arial", Font.BOLD, 12);
	private static final Font TITLE_FONT = new Font("Arial", Font.BOLD, 20);
	private static final Font BUTTON_FONT = new Font("Arial", Font.BOLD, 12);
	private static final int WORKFLOW_MIN_WIDTH = 340;
	private static final int WORKFLOW_PREF_WIDTH = 380;
	private static final int INSPECTION_MIN_WIDTH = 360;
	private static final int INSPECTION_PREF_WIDTH = 560;
	private static final int REPORT_MIN_WIDTH = 360;
	private static final int REPORT_PREF_WIDTH = 420;
	private static final int PANEL_PREF_HEIGHT = 680;
	private static final int SPLIT_DIVIDER_SIZE = 8;

	private final ControllerAgentGui owner;

	private JComboBox<UciRetailDatasetImporter.DatasetKind> datasetKindCombo;
	private UciRetailDatasetImporter.DatasetKind selectedDatasetKind =
			UciRetailDatasetImporter.DatasetKind.RETAIL;

	private JTextField fileField;
	private JTextField outputDirField;
	private JTextField outputNameField;
	private JComboBox<UciRetailDatasetImporter.MappingProfile> profileCombo;
	private JComboBox<String> invoiceCombo;
	private JComboBox<String> stockCombo;
	private JComboBox<String> descriptionCombo;
	private JComboBox<String> quantityCombo;
	private JComboBox<String> dateCombo;
	private JComboBox<String> priceCombo;
	private JComboBox<String> customerCombo;
	private JComboBox<String> countryCombo;
	private JComboBox<String> customReferenceCombo;
	private JComboBox<String> customPostCombo;
	private JComboBox<String> customDateCombo;
	private JComboBox<String> customUserIdCombo;
	private JComboBox<String> customUserNameCombo;
	private JComboBox<String> customTextCombo;
	private JCheckBox skipCancelledCheck;
	private JCheckBox skipMissingCustomerCheck;
	private JCheckBox skipQuantityCheck;
	private JCheckBox skipPriceCheck;
	private JCheckBox stockPrefixCheck;
	private JCheckBox stableFolloweeCheck;
	private JTextField minTokensField;
	private JTextField fallbackDateField;
	private JComboBox<UciRetailDatasetImporter.DatasetBalanceMode> balanceModeCombo;
	private JComboBox<UciRetailDatasetImporter.DatasetBalanceTargetPolicy> balanceTargetPolicyCombo;
	private JTextField balanceTargetField;
	private JTextField balanceSeedField;
	private JTextArea balanceGuide;
	private boolean balancePolicyUserSelected;
	private boolean suppressBalancePolicyEvents;

	private JButton inspectButton;
	private JButton buildButton;
	private JButton useOutputButton;
	private JButton resetButton;

	private JLabel rowsValue;
	private JLabel columnsValue;
	private JLabel formatValue;
	private JLabel usersValue;
	private JLabel outputValue;
	private JLabel statusLabel;
	private JLabel phaseValue;
	private JLabel rowsConsoleValue;
	private JLabel outputConsoleValue;
	private JLabel eventCountValue;
	private JProgressBar progressBar;
	private JTextArea profileGuide;
	private JTextArea logArea;
	private JPanel customMappingPanel;
	private JLabel invoiceColumnLabel;
	private JLabel stockColumnLabel;
	private JLabel descriptionColumnLabel;
	private JLabel quantityColumnLabel;
	private JLabel dateColumnLabel;
	private JLabel priceColumnLabel;
	private JLabel customerColumnLabel;
	private JLabel countryColumnLabel;

	private DefaultTableModel previewModel;
	private DefaultTableModel columnTableModel;
	private DefaultTableModel outputPreviewModel;
	private DefaultTableModel livePreviewModel;
	private DefaultTableModel imbalanceModel;
	private JTextArea imbalanceSummary;
	private JLabel imbalanceStatusBadge;
	private JButton refreshImbalanceButton;
	private JTextArea livePreviewNote;
	private final JLabel[] liveMappingLabels = new JLabel[6];
	private boolean suppressLivePreview;
	private int consoleEventCount;
	private int inspectProgressBucket = -1;
	private int buildProgressBucket = -1;

	private File selectedFile;
	private UciRetailDatasetImporter.SourceProfile sourceProfile;
	private UciRetailDatasetImporter.ImportReport lastReport;
	private UciRetailDatasetImporter.ImportReport lastBalanceDiagnosis;
	private SwingWorker<UciRetailDatasetImporter.ImportReport, Void> imbalanceWorker;
	private int imbalanceRequestId;

	DatasetImportLabDialog(ControllerAgentGui owner) {
		super(owner, "Dataset Import Lab", false);
		this.owner = owner;
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setContentPane(buildContent());
		setMinimumSize(new Dimension(1180, 680));
		setSize(new Dimension(1360, 780));
		setLocationRelativeTo(owner);
		setIdleState();
	}

	private JPanel buildContent() {
		JPanel root = new JPanel(new BorderLayout(8, 8));
		root.setBackground(DSMP_BLUE);
		root.setBorder(new EmptyBorder(8, 8, 8, 8));
		root.add(createHeader(), BorderLayout.NORTH);

		JPanel workflowPanel = createWorkflowPanel();
		JPanel inspectionPanel = createInspectionPanel();
		JPanel reportPanel = createReportPanel();

		JSplitPane middle = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, inspectionPanel, reportPanel);
		middle.setContinuousLayout(true);
		middle.setResizeWeight(1.0);
		middle.setDividerSize(SPLIT_DIVIDER_SIZE);
		middle.setBorder(null);
		sizePanel(middle,
				INSPECTION_MIN_WIDTH + REPORT_MIN_WIDTH + SPLIT_DIVIDER_SIZE,
				INSPECTION_PREF_WIDTH + REPORT_PREF_WIDTH + SPLIT_DIVIDER_SIZE);
		installRightPinnedDivider(middle, REPORT_MIN_WIDTH, REPORT_PREF_WIDTH, 0.36d);

		JSplitPane main = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, workflowPanel, middle);
		main.setContinuousLayout(true);
		main.setResizeWeight(0.0);
		main.setDividerSize(SPLIT_DIVIDER_SIZE);
		main.setBorder(null);
		installLeftPinnedDivider(main, WORKFLOW_MIN_WIDTH, WORKFLOW_PREF_WIDTH, 0.25d);
		root.add(main, BorderLayout.CENTER);
		root.add(createFooter(), BorderLayout.SOUTH);
		return root;
	}

	private JPanel createHeader() {
		JPanel panel = new JPanel(new BorderLayout());
		panel.setOpaque(false);
		panel.setBorder(oldTitle("DSMP Dataset Import Lab"));

		JLabel title = new JLabel("Convert Dataset to DSMP Text Format", SwingConstants.CENTER);
		title.setForeground(WHITE);
		title.setFont(new Font("Arial", Font.BOLD, 22));
		panel.add(title, BorderLayout.NORTH);

		JLabel subtitle = new JLabel(
				"Retail, journal, tweet, and custom files use the v4 inspection, mapping, validation, build, metadata, and audit flow.",
				SwingConstants.CENTER);
		subtitle.setForeground(WHITE);
		subtitle.setFont(new Font("Arial", Font.PLAIN, 13));
		panel.add(subtitle, BorderLayout.SOUTH);
		return panel;
	}

	private JPanel createWorkflowPanel() {
		JPanel workflow = oldPanel();
		workflow.setLayout(new BoxLayout(workflow, BoxLayout.Y_AXIS));

		workflow.add(createSourcePanel());
		workflow.add(Box.createVerticalStrut(8));
		workflow.add(createProfilePanel());
		workflow.add(Box.createVerticalStrut(8));
		workflow.add(createDetectedColumnsPanel());
		workflow.add(Box.createVerticalStrut(8));
		customMappingPanel = createCustomMappingPanel();
		workflow.add(customMappingPanel);
		workflow.add(Box.createVerticalStrut(8));
		workflow.add(createBalancePanel());
		workflow.add(Box.createVerticalStrut(8));
		workflow.add(createValidationPanel());
		workflow.add(Box.createVerticalStrut(8));
		workflow.add(createActionPanel());

		JScrollPane scroll = new JScrollPane(workflow);
		sizePanel(scroll, WORKFLOW_MIN_WIDTH, WORKFLOW_PREF_WIDTH);
		scroll.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
		scroll.getVerticalScrollBar().setUnitIncrement(18);
		scroll.setBorder(oldTitle("Import Workflow"));
		scroll.getViewport().setBackground(DSMP_BLUE);

		JPanel shell = oldPanel(new BorderLayout());
		shell.add(scroll, BorderLayout.CENTER);
		sizePanel(shell, WORKFLOW_MIN_WIDTH, WORKFLOW_PREF_WIDTH);
		return shell;
	}

	private JPanel createSourcePanel() {
		JPanel panel = oldPanel(new GridBagLayout());
		panel.setBorder(oldTitle("1. Dataset Family and File"));
		GridBagConstraints gbc = formConstraints();

		datasetKindCombo = new JComboBox<UciRetailDatasetImporter.DatasetKind>(
				UciRetailDatasetImporter.DatasetKind.values());
		datasetKindCombo.setFont(LABEL_FONT);
		datasetKindCombo.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				selectDatasetKind((UciRetailDatasetImporter.DatasetKind) datasetKindCombo.getSelectedItem());
			}
		});
		addFormRow(panel, gbc, 0, "Dataset family:", datasetKindCombo);

		fileField = textField("");
		fileField.setEditable(false);
		addFormRow(panel, gbc, 1, "Selected file:", fileField);

		JButton chooseButton = oldButton("Choose Dataset");
		chooseButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				chooseFile();
			}
		});
		inspectButton = oldButton("Inspect Dataset");
		inspectButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				inspectSelectedFile();
			}
		});
		addButtonRow(panel, gbc, 2, chooseButton, inspectButton);
		return panel;
	}

	private JPanel createProfilePanel() {
		JPanel panel = oldPanel(new BorderLayout(6, 6));
		panel.setBorder(oldTitle("2. Mapping Profile"));

		JPanel top = oldPanel(new GridBagLayout());
		GridBagConstraints gbc = formConstraints();
		profileCombo = new JComboBox<UciRetailDatasetImporter.MappingProfile>();
		profileCombo.setFont(LABEL_FONT);
		profileCombo.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				updateProfileGuide();
			}
		});
		addFormRow(top, gbc, 0, "Followee meaning:", profileCombo);
		panel.add(top, BorderLayout.NORTH);

		profileGuide = textArea(7, 24);
		profileGuide.setBackground(WHITE);
		profileGuide.setForeground(INK);
		panel.add(new JScrollPane(profileGuide), BorderLayout.CENTER);

		stableFolloweeCheck = oldCheck("Keep one stable followee label per user", true);
		panel.add(stableFolloweeCheck, BorderLayout.SOUTH);
		return panel;
	}

	private JPanel createDetectedColumnsPanel() {
		JPanel panel = oldPanel(new GridBagLayout());
		panel.setBorder(oldTitle("3. Detected Source Columns"));

		invoiceCombo = combo();
		stockCombo = combo();
		descriptionCombo = combo();
		quantityCombo = combo();
		dateCombo = combo();
		priceCombo = combo();
		customerCombo = combo();
		countryCombo = combo();
		invoiceColumnLabel = oldLabel("Record / post id:");
		stockColumnLabel = oldLabel("Optional category:");
		descriptionColumnLabel = oldLabel("Text / document body:");
		quantityColumnLabel = oldLabel("Optional numeric filter:");
		dateColumnLabel = oldLabel("Date / timestamp:");
		priceColumnLabel = oldLabel("Optional value filter:");
		customerColumnLabel = oldLabel("User / author id:");
		countryColumnLabel = oldLabel("Followee / reference:");

		GridBagConstraints gbc = formConstraints();
		addFormRow(panel, gbc, 0, invoiceColumnLabel, invoiceCombo);
		addFormRow(panel, gbc, 1, stockColumnLabel, stockCombo);
		addFormRow(panel, gbc, 2, descriptionColumnLabel, descriptionCombo);
		addFormRow(panel, gbc, 3, quantityColumnLabel, quantityCombo);
		addFormRow(panel, gbc, 4, dateColumnLabel, dateCombo);
		addFormRow(panel, gbc, 5, priceColumnLabel, priceCombo);
		addFormRow(panel, gbc, 6, customerColumnLabel, customerCombo);
		addFormRow(panel, gbc, 7, countryColumnLabel, countryCombo);
		return panel;
	}

	private JPanel createCustomMappingPanel() {
		JPanel panel = oldPanel(new GridBagLayout());
		panel.setBorder(oldTitle("4. Custom DSMP Column Mapping"));
		customReferenceCombo = combo();
		customPostCombo = combo();
		customDateCombo = combo();
		customUserIdCombo = combo();
		customUserNameCombo = combo();
		customTextCombo = combo();

		GridBagConstraints gbc = formConstraints();
		addFormRow(panel, gbc, 0, "Followee / reference:", customReferenceCombo);
		addFormRow(panel, gbc, 1, "Numeric post id:", customPostCombo);
		addFormRow(panel, gbc, 2, "Date:", customDateCombo);
		addFormRow(panel, gbc, 3, "Numeric user id:", customUserIdCombo);
		addFormRow(panel, gbc, 4, "User name:", customUserNameCombo);
		addFormRow(panel, gbc, 5, "Text / document body:", customTextCombo);
		return panel;
	}


	private JPanel createBalancePanel() {
		JPanel panel = oldPanel(new BorderLayout(6, 6));
		panel.setBorder(oldTitle("5. Class Balance"));

		JPanel controls = oldPanel(new GridBagLayout());
		GridBagConstraints gbc = formConstraints();
		balanceModeCombo = new JComboBox<UciRetailDatasetImporter.DatasetBalanceMode>(
				UciRetailDatasetImporter.DatasetBalanceMode.values());
		balanceModeCombo.setFont(LABEL_FONT);
		balanceModeCombo.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				applyRecommendedBalanceTargetPolicyIfUnpinned();
				updateBalanceGuide();
				markImbalanceStale();
			}
		});
		addFormRow(controls, gbc, 0, "Balance mode:", balanceModeCombo);

		balanceTargetPolicyCombo = new JComboBox<UciRetailDatasetImporter.DatasetBalanceTargetPolicy>(
				new UciRetailDatasetImporter.DatasetBalanceTargetPolicy[] {
						UciRetailDatasetImporter.DatasetBalanceTargetPolicy.PRESERVE_TOTAL_SIZE,
						UciRetailDatasetImporter.DatasetBalanceTargetPolicy.CONSERVATIVE_CAP,
						UciRetailDatasetImporter.DatasetBalanceTargetPolicy.STRICT_MINORITY,
						UciRetailDatasetImporter.DatasetBalanceTargetPolicy.BOUNDED_AUGMENTATION,
						UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MANUAL
				});
		balanceTargetPolicyCombo.setFont(LABEL_FONT);
		balanceTargetPolicyCombo.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				if (!suppressBalancePolicyEvents) {
					balancePolicyUserSelected = true;
				}
				updateBalanceGuide();
				markImbalanceStale();
			}
		});
		addFormRow(controls, gbc, 1, "Target policy:", balanceTargetPolicyCombo);

		balanceTargetField = textField("0");
		balanceSeedField = textField("1");
		addFormRow(controls, gbc, 2, "Manual target users/class:", balanceTargetField);
		addFormRow(controls, gbc, 3, "Balance seed:", balanceSeedField);
		panel.add(controls, BorderLayout.NORTH);

		balanceGuide = textArea(7, 24);
		balanceGuide.setBackground(WHITE);
		balanceGuide.setForeground(INK);
		panel.add(new JScrollPane(balanceGuide), BorderLayout.CENTER);
		updateBalanceGuide();
		return panel;
	}

	private JPanel createValidationPanel() {
		JPanel panel = oldPanel(new GridBagLayout());
		panel.setBorder(oldTitle("6. Validation and Output"));
		skipCancelledCheck = oldCheck("Skip cancelled invoices", true);
		skipMissingCustomerCheck = oldCheck("Skip missing customer ids", true);
		skipQuantityCheck = oldCheck("Skip non-positive quantities", true);
		skipPriceCheck = oldCheck("Skip non-positive prices", false);
		stockPrefixCheck = oldCheck("Group stock codes by prefix", true);
		minTokensField = textField("1");
		fallbackDateField = textField("2011-01-01");
		outputDirField = textField("");
		outputNameField = textField("ImportedDataset_DSMP");

		GridBagConstraints gbc = formConstraints();
		addWide(panel, gbc, 0, skipCancelledCheck);
		addWide(panel, gbc, 1, skipMissingCustomerCheck);
		addWide(panel, gbc, 2, skipQuantityCheck);
		addWide(panel, gbc, 3, skipPriceCheck);
		addWide(panel, gbc, 4, stockPrefixCheck);
		addFormRow(panel, gbc, 5, "Minimum text tokens:", minTokensField);
		addFormRow(panel, gbc, 6, "Fallback date:", fallbackDateField);
		addFormRow(panel, gbc, 7, "Output folder:", outputDirField);
		addFormRow(panel, gbc, 8, "Output dataset name:", outputNameField);
		installLivePreviewListeners();
		return panel;
	}

	private JPanel createActionPanel() {
		JPanel panel = oldPanel(new GridLayout(3, 1, 0, 6));
		panel.setBorder(oldTitle("7. Build and Use"));
		buildButton = oldButton("Build DSMP Dataset");
		buildButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				buildDataset();
			}
		});
		useOutputButton = oldButton("Use Output in Simulator");
		useOutputButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				useOutputInSimulator();
			}
		});
		resetButton = oldButton("Reset Defaults");
		resetButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				applyDefaultsFromSource();
			}
		});
		panel.add(buildButton);
		panel.add(useOutputButton);
		panel.add(resetButton);
		return panel;
	}

	private JPanel createInspectionPanel() {
		JPanel panel = oldPanel(new BorderLayout(8, 8));
		sizePanel(panel, INSPECTION_MIN_WIDTH, INSPECTION_PREF_WIDTH);
		panel.setBorder(oldTitle("Inspection and Output Preview"));
		panel.add(createMetricsPanel(), BorderLayout.NORTH);

		JTabbedPane tabs = new JTabbedPane();
		tabs.setFont(LABEL_FONT);
		tabs.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
		previewModel = new DefaultTableModel();
		tabs.addTab("Source Snapshot", tableScroll(styledTable(previewModel)));

		columnTableModel = new DefaultTableModel(
				new Object[] { "Column", "Non-empty", "Numeric", "Date-like", "Distinct", "Samples" }, 0);
		tabs.addTab("Column Intelligence", tableScroll(styledTable(columnTableModel)));

		outputPreviewModel = new DefaultTableModel(new Object[] {
				"Followee", "Post id", "Date", "User id", "User name", "Text" }, 0);
		tabs.addTab("DSMP Output Preview", tableScroll(styledTable(outputPreviewModel)));
		tabs.addTab("Imbalance", createImbalancePanel());
		panel.add(tabs, BorderLayout.CENTER);
		return panel;
	}


	private JPanel createImbalancePanel() {
		JPanel panel = oldPanel(new BorderLayout(6, 6));
		JPanel top = oldPanel(new BorderLayout(6, 6));
		JPanel titleBlock = oldPanel(new GridLayout(2, 1));
		JLabel title = oldLabel("Class Imbalance Diagnosis");
		title.setFont(new Font("Arial", Font.BOLD, 15));
		titleBlock.add(title);
		JLabel subtitle = oldLabel("Full-dataset followee distribution before build, then exact post-build balance reporting.");
		subtitle.setFont(new Font("Arial", Font.PLAIN, 11));
		titleBlock.add(subtitle);
		top.add(titleBlock, BorderLayout.CENTER);

		JPanel actions = oldPanel(new FlowLayout(FlowLayout.RIGHT, 6, 0));
		imbalanceStatusBadge = oldLabel("WAITING");
		imbalanceStatusBadge.setHorizontalAlignment(SwingConstants.CENTER);
		imbalanceStatusBadge.setBorder(BorderFactory.createLineBorder(LINE));
		imbalanceStatusBadge.setPreferredSize(new Dimension(90, 25));
		actions.add(imbalanceStatusBadge);
		refreshImbalanceButton = oldButton("Refresh Diagnosis");
		refreshImbalanceButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				refreshImbalanceDiagnosis(true);
			}
		});
		actions.add(refreshImbalanceButton);
		top.add(actions, BorderLayout.EAST);
		panel.add(top, BorderLayout.NORTH);

		imbalanceModel = new DefaultTableModel(new Object[] {
				"Followee", "Original users", "Original rows", "Final users", "User share", "Ratio", "Balance action" }, 0);
		panel.add(tableScroll(styledTable(imbalanceModel)), BorderLayout.CENTER);

		imbalanceSummary = textArea(6, 42);
		imbalanceSummary.setBackground(WHITE);
		imbalanceSummary.setForeground(INK);
		imbalanceSummary.setText("Inspect a dataset to see class imbalance, balance impact, and warnings here.");
		panel.add(new JScrollPane(imbalanceSummary), BorderLayout.SOUTH);
		resetImbalancePanel("Inspect a dataset to see class imbalance, balance impact, and warnings here.");
		return panel;
	}

	private JPanel createMetricsPanel() {
		JPanel metrics = oldPanel(new GridLayout(1, 5, 6, 0));
		rowsValue = metric("Rows", metrics);
		columnsValue = metric("Columns", metrics);
		formatValue = metric("Format", metrics);
		usersValue = metric("User signal", metrics);
		outputValue = metric("Output", metrics);
		return metrics;
	}

	private JLabel metric(String title, JPanel parent) {
		JPanel panel = oldPanel(new BorderLayout());
		panel.setBorder(oldTitle(title));
		JLabel value = new JLabel("-", SwingConstants.CENTER);
		value.setForeground(WHITE);
		value.setFont(new Font("Arial", Font.BOLD, 16));
		panel.add(value, BorderLayout.CENTER);
		parent.add(panel);
		return value;
	}

	private JPanel createReportPanel() {
		JPanel panel = oldPanel(new BorderLayout(8, 8));
		sizePanel(panel, REPORT_MIN_WIDTH, REPORT_PREF_WIDTH);
		panel.setBorder(oldTitle("Live Snapshot and Console"));
		panel.add(createLivePreviewPanel(), BorderLayout.CENTER);
		panel.add(createConsolePanel(), BorderLayout.SOUTH);
		return panel;
	}

	private JPanel createLivePreviewPanel() {
		JPanel panel = oldPanel(new BorderLayout(6, 6));
		panel.setBorder(oldTitle("Live DSMP Snapshot"));
		panel.add(createLiveMappingSummary(), BorderLayout.NORTH);

		livePreviewModel = new DefaultTableModel(new Object[] {
				"Followee", "Post id", "Date", "User id", "User name", "Text" }, 0);
		JTable table = styledTable(livePreviewModel);
		table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		panel.add(tableScroll(table), BorderLayout.CENTER);

		livePreviewNote = textArea(3, 24);
		livePreviewNote.setBackground(WHITE);
		livePreviewNote.setForeground(INK);
		livePreviewNote.setText("Inspect a source file to see the live DSMP shape.");
		panel.add(new JScrollPane(livePreviewNote), BorderLayout.SOUTH);
		return panel;
	}

	private JPanel createLiveMappingSummary() {
		JPanel wrapper = oldPanel(new GridLayout(6, 2, 5, 3));
		String[] fields = new String[] {
				"Followee:", "Post id:", "Date:", "User id:", "User name:", "Text:" };
		for (int i = 0; i < fields.length; i++) {
			wrapper.add(oldLabel(fields[i]));
			JLabel value = oldLabel("Inspect first");
			value.setFont(new Font("Arial", Font.PLAIN, 11));
			liveMappingLabels[i] = value;
			wrapper.add(value);
		}
		return wrapper;
	}

	private JPanel createConsolePanel() {
		JPanel panel = oldPanel(new BorderLayout(5, 5));
		panel.setPreferredSize(new Dimension(REPORT_PREF_WIDTH, 230));
		panel.setMinimumSize(new Dimension(REPORT_MIN_WIDTH, 180));
		panel.setBorder(oldTitle("Import Console"));

		JPanel stats = oldPanel(new GridLayout(2, 4, 5, 2));
		stats.add(oldLabel("Events:"));
		eventCountValue = oldLabel("0");
		stats.add(eventCountValue);
		stats.add(oldLabel("Phase:"));
		phaseValue = oldLabel("READY");
		stats.add(phaseValue);
		stats.add(oldLabel("Rows:"));
		rowsConsoleValue = oldLabel("-");
		stats.add(rowsConsoleValue);
		stats.add(oldLabel("Output:"));
		outputConsoleValue = oldLabel("Waiting");
		stats.add(outputConsoleValue);
		panel.add(stats, BorderLayout.NORTH);

		logArea = textArea(7, 28);
		logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
		logArea.setBackground(WHITE);
		logArea.setForeground(INK);
		panel.add(new JScrollPane(logArea), BorderLayout.CENTER);
		return panel;
	}

	private JPanel createFooter() {
		JPanel panel = oldPanel(new BorderLayout(8, 0));
		statusLabel = oldLabel("READY");
		statusLabel.setHorizontalAlignment(SwingConstants.CENTER);
		statusLabel.setBorder(BorderFactory.createLineBorder(LINE));
		statusLabel.setPreferredSize(new Dimension(140, 26));
		panel.add(statusLabel, BorderLayout.WEST);

		progressBar = new JProgressBar(0, 100);
		progressBar.setStringPainted(true);
		progressBar.setString("Choose a source file to inspect.");
		progressBar.setFont(LABEL_FONT);
		panel.add(progressBar, BorderLayout.CENTER);

		JButton closeButton = oldButton("Close");
		closeButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				dispose();
			}
		});
		panel.add(closeButton, BorderLayout.EAST);
		return panel;
	}

	private void selectDatasetKind(UciRetailDatasetImporter.DatasetKind kind) {
		if (kind == null) {
			kind = UciRetailDatasetImporter.DatasetKind.RETAIL;
		}
		if (selectedDatasetKind == kind) {
			updateDatasetKindUi();
			return;
		}
		selectedDatasetKind = kind;
		sourceProfile = null;
		lastReport = null;
		lastBalanceDiagnosis = null;
		clearTables();
		if (buildButton != null) {
			buildButton.setEnabled(false);
		}
		if (useOutputButton != null) {
			useOutputButton.setEnabled(false);
		}
		if (selectedFile != null && outputNameField != null) {
			outputNameField.setText(safeBase(selectedFile.getName()) + "_DSMP");
			setStatus("READY", kind + " mode selected. Inspect the file again so mappings are refreshed.", 0);
			log("Dataset family changed to " + kind + ". Re-inspect the selected file before building.");
		}
		updateDatasetKindUi();
	}

	private void updateDatasetKindUi() {
		if (datasetKindCombo != null && datasetKindCombo.getSelectedItem() != selectedDatasetKind) {
			datasetKindCombo.setSelectedItem(selectedDatasetKind);
		}
		updateProfileChoicesForKind(selectedDatasetKind, null);
		updateColumnLabelsForKind(selectedDatasetKind);
		updateValidationControlsForKind(selectedDatasetKind);
		updateProfileGuide();
	}

	private void updateProfileChoicesForKind(UciRetailDatasetImporter.DatasetKind kind,
			UciRetailDatasetImporter.MappingProfile preferred) {
		if (profileCombo == null) {
			return;
		}
		UciRetailDatasetImporter.MappingProfile current =
				preferred == null
						? (UciRetailDatasetImporter.MappingProfile) profileCombo.getSelectedItem()
						: preferred;
		UciRetailDatasetImporter.MappingProfile[] profiles = profilesForKind(kind);
		DefaultComboBoxModel<UciRetailDatasetImporter.MappingProfile> model =
				new DefaultComboBoxModel<UciRetailDatasetImporter.MappingProfile>();
		boolean containsCurrent = false;
		for (int i = 0; i < profiles.length; i++) {
			model.addElement(profiles[i]);
			if (profiles[i] == current) {
				containsCurrent = true;
			}
		}
		profileCombo.setModel(model);
		if (containsCurrent) {
			profileCombo.setSelectedItem(current);
		} else if (profiles.length > 0) {
			profileCombo.setSelectedItem(profiles[0]);
		}
		profileCombo.setEnabled(kind != UciRetailDatasetImporter.DatasetKind.OTHER);
	}

	private static UciRetailDatasetImporter.MappingProfile[] profilesForKind(
			UciRetailDatasetImporter.DatasetKind kind) {
		if (kind == UciRetailDatasetImporter.DatasetKind.JOURNAL) {
			return new UciRetailDatasetImporter.MappingProfile[] {
					UciRetailDatasetImporter.MappingProfile.JOURNAL_AUTHORS,
					UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS,
					UciRetailDatasetImporter.MappingProfile.CUSTOM
			};
		}
		if (kind == UciRetailDatasetImporter.DatasetKind.TWEETS) {
			return new UciRetailDatasetImporter.MappingProfile[] {
					UciRetailDatasetImporter.MappingProfile.TWEET_USERS,
					UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS,
					UciRetailDatasetImporter.MappingProfile.CUSTOM
			};
		}
		if (kind == UciRetailDatasetImporter.DatasetKind.OTHER) {
			return new UciRetailDatasetImporter.MappingProfile[] {
					UciRetailDatasetImporter.MappingProfile.CUSTOM
			};
		}
		return new UciRetailDatasetImporter.MappingProfile[] {
				UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS,
				UciRetailDatasetImporter.MappingProfile.SMART_PRODUCT_CATEGORY,
				UciRetailDatasetImporter.MappingProfile.PRODUCT_FAMILY,
				UciRetailDatasetImporter.MappingProfile.COUNTRY_BASELINE,
				UciRetailDatasetImporter.MappingProfile.STOCK_CODE_EXPERIMENTAL,
				UciRetailDatasetImporter.MappingProfile.CUSTOM
		};
	}

	private void updateColumnLabelsForKind(UciRetailDatasetImporter.DatasetKind kind) {
		if (invoiceColumnLabel == null) {
			return;
		}
		if (kind == UciRetailDatasetImporter.DatasetKind.RETAIL) {
			invoiceColumnLabel.setText("Invoice / order id:");
			stockColumnLabel.setText("Stock / product code:");
			descriptionColumnLabel.setText("Description / title text:");
			quantityColumnLabel.setText("Quantity:");
			dateColumnLabel.setText("Invoice date:");
			priceColumnLabel.setText("Unit price:");
			customerColumnLabel.setText("Customer id:");
			countryColumnLabel.setText("Country / region:");
		} else if (kind == UciRetailDatasetImporter.DatasetKind.JOURNAL) {
			invoiceColumnLabel.setText("Paper / article id:");
			stockColumnLabel.setText("Subject / category:");
			descriptionColumnLabel.setText("Title / abstract text:");
			quantityColumnLabel.setText("Optional numeric filter:");
			dateColumnLabel.setText("Publication date / year:");
			priceColumnLabel.setText("Optional value filter:");
			customerColumnLabel.setText("Author id / sequence id:");
			countryColumnLabel.setText("Journal / venue followee:");
		} else if (kind == UciRetailDatasetImporter.DatasetKind.TWEETS) {
			invoiceColumnLabel.setText("Tweet / status id:");
			stockColumnLabel.setText("Topic / hashtag:");
			descriptionColumnLabel.setText("Tweet text:");
			quantityColumnLabel.setText("Optional numeric filter:");
			dateColumnLabel.setText("Created date:");
			priceColumnLabel.setText("Optional value filter:");
			customerColumnLabel.setText("User id / screen name:");
			countryColumnLabel.setText("Reference / source account:");
		} else {
			invoiceColumnLabel.setText("Record / post id:");
			stockColumnLabel.setText("Optional category:");
			descriptionColumnLabel.setText("Text / document body:");
			quantityColumnLabel.setText("Optional numeric filter:");
			dateColumnLabel.setText("Date / timestamp:");
			priceColumnLabel.setText("Optional value filter:");
			customerColumnLabel.setText("User / entity id:");
			countryColumnLabel.setText("Followee / reference label:");
		}
	}

	private void updateValidationControlsForKind(UciRetailDatasetImporter.DatasetKind kind) {
		if (skipCancelledCheck == null) {
			return;
		}
		boolean retail = kind == UciRetailDatasetImporter.DatasetKind.RETAIL;
		skipCancelledCheck.setText(retail ? "Skip cancelled invoices" : "Skip cancelled rows if a cancel flag exists");
		skipQuantityCheck.setText(retail ? "Skip non-positive quantities" : "Skip non-positive numeric filter");
		skipPriceCheck.setText(retail ? "Skip non-positive prices" : "Skip non-positive value filter");
		stockPrefixCheck.setText(retail ? "Group stock codes by prefix" : "Group optional category values by prefix");
		skipCancelledCheck.setEnabled(retail);
		skipQuantityCheck.setEnabled(retail);
		skipPriceCheck.setEnabled(retail);
		stockPrefixCheck.setEnabled(retail);
	}

	private void chooseFile() {
		JFileChooser chooser = new JFileChooser();
		File start = selectedFile != null ? selectedFile.getParentFile() : null;
		if (start == null && owner.fileChooser != null) {
			start = owner.fileChooser.getCurrentDirectory();
		}
		if (start != null) {
			chooser.setCurrentDirectory(start);
		}
		chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
		chooser.setFileFilter(new FileNameExtensionFilter(
				"Dataset files (csv, tsv, txt, xlsx)", "csv", "tsv", "txt", "xlsx"));
		int result = chooser.showOpenDialog(this);
		if (result != JFileChooser.APPROVE_OPTION) {
			return;
		}
		selectedFile = chooser.getSelectedFile();
		fileField.setText(selectedFile.getAbsolutePath());
		outputDirField.setText(new File(selectedFile.getParentFile(), "DSMP_Imported").getAbsolutePath());
		outputNameField.setText(safeBase(selectedFile.getName()) + "_DSMP");
		sourceProfile = null;
		lastReport = null;
		lastBalanceDiagnosis = null;
		clearTables();
		setStatus("READY", "Dataset selected. Inspect it before building.", 0);
		updateConsoleStats("READY", "-", "Selected");
		inspectButton.setEnabled(true);
		buildButton.setEnabled(false);
		useOutputButton.setEnabled(false);
		log("SOURCE SELECTED - " + selectedDatasetKind
				+ "\nFile: " + selectedFile.getAbsolutePath()
				+ "\nSize: " + humanBytes(selectedFile.length())
				+ "\nNext: Inspect Dataset");
	}

	private void inspectSelectedFile() {
		if (selectedFile == null) {
			log("Choose a dataset file first.");
			return;
		}
		inspectProgressBucket = -1;
		setBusy(true);
		setStatus("INSPECTING", "Reading headers, delimiter, rows, and column statistics.", 8);
		updateConsoleStats("INSPECTING", "-", "Scanning");
		log("INSPECTION START"
				+ "\nFamily: " + selectedDatasetKind
				+ "\nFile: " + selectedFile.getName());
		SwingWorker<UciRetailDatasetImporter.SourceProfile, String> worker =
				new SwingWorker<UciRetailDatasetImporter.SourceProfile, String>() {
			protected UciRetailDatasetImporter.SourceProfile doInBackground() throws Exception {
				return UciRetailDatasetImporter.inspect(
						selectedFile,
						selectedDatasetKind,
						new UciRetailDatasetImporter.ProgressListener() {
					public void onProgress(final String message, final int percent) {
						SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								setStatus("INSPECTING", message, percent);
								logProgress("INSPECT", message, percent);
							}
						});
					}
				});
			}

			protected void done() {
				try {
					sourceProfile = get();
					selectedDatasetKind = sourceProfile.selectedKind;
					applyDefaultsFromSource();
					renderSourceProfile();
					setStatus("READY", "Dataset inspection complete. Review mapping, then build DSMP output.", 100);
					updateConsoleStats("INSPECTED",
							new DecimalFormat("#,##0").format(sourceProfile.dataRows),
							sourceProfile.sourceFormat);
					log("INSPECTION COMPLETE"
							+ "\n" + UciRetailDatasetImporter.describeSource(sourceProfile)
							+ "\nDetected columns: " + sourceProfile.columnNames.size()
							+ "\nMalformed rows: " + new DecimalFormat("#,##0").format(sourceProfile.malformedRows)
							+ "\nSamples loaded: " + sourceProfile.sampleRows.size());
					buildButton.setEnabled(true);
				} catch (Exception ex) {
					setStatus("FAILED", "Inspection failed: " + rootMessage(ex), 0);
					updateConsoleStats("FAILED", "-", "Inspect failed");
					log("INSPECTION FAILED\n" + rootMessage(ex));
				} finally {
					setBusy(false);
				}
			}
		};
		worker.execute();
	}

	private void buildDataset() {
		if (sourceProfile == null) {
			log("Inspect a dataset before building.");
			return;
		}
		final UciRetailDatasetImporter.Options options;
		try {
			options = collectOptions();
		} catch (Exception ex) {
			setStatus("FAILED", rootMessage(ex), 0);
			log("OPTIONS ERROR\n" + rootMessage(ex));
			return;
		}
		setBusy(true);
		useOutputButton.setEnabled(false);
		outputPreviewModel.setRowCount(0);
		setStatus("BUILDING", "Validating and writing DSMP dataset.", 4);
		buildProgressBucket = -1;
		updateConsoleStats("BUILDING", "-", "Writing");
		log("BUILD START"
				+ "\nProfile: " + options.profile
				+ "\nBalance mode: " + options.balanceMode
				+ "\nTarget policy: " + options.balanceTargetPolicy
				+ "\nManual target users/class: "
				+ (options.balanceTargetPolicy
						== UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MANUAL
								? Integer.toString(options.balanceTargetUsersPerClass)
								: "not used")
				+ "\nBalance seed: " + options.balanceSeed
				+ "\nOutput folder: " + options.outputDirectory.getAbsolutePath()
				+ "\nDataset name: " + options.outputBaseName
				+ "\nMinimum text tokens: " + options.minimumTextTokens);

		SwingWorker<UciRetailDatasetImporter.ImportReport, String> worker =
				new SwingWorker<UciRetailDatasetImporter.ImportReport, String>() {
			protected UciRetailDatasetImporter.ImportReport doInBackground() throws Exception {
				return UciRetailDatasetImporter.importDataset(
						sourceProfile,
						options,
						new UciRetailDatasetImporter.ProgressListener() {
					public void onProgress(final String message, final int percent) {
						SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								setStatus("BUILDING", message, percent);
								logProgress("BUILD", message, percent);
							}
						});
					}
				});
			}

			protected void done() {
				try {
					lastReport = get();
					renderReport(lastReport);
					setStatus("READY", lastReport.summaryLine(), 100);
					updateConsoleStats("COMPLETE",
							new DecimalFormat("#,##0").format(lastReport.writtenRows),
							lastReport.outputFile.getName());
					log(formatBuildReport(lastReport));
					useOutputButton.setEnabled(true);
				} catch (Exception ex) {
					setStatus("FAILED", "Build failed: " + rootMessage(ex), 0);
					updateConsoleStats("FAILED", "-", "Build failed");
					log("BUILD FAILED\n" + rootMessage(ex));
				} finally {
					setBusy(false);
				}
			}
		};
		worker.execute();
	}

	private void useOutputInSimulator() {
		if (lastReport == null || lastReport.outputFile == null || !lastReport.outputFile.isFile()) {
			log("Build a DSMP dataset first.");
			return;
		}
		String label = safeBase(lastReport.outputFile.getName());
		owner.loadImportedDataset(lastReport.outputFile, label);
		setStatus("READY", "Imported dataset sent to the simulator. Press Get Users there to inspect targets.", 100);
		updateConsoleStats("READY",
				new DecimalFormat("#,##0").format(lastReport.writtenRows),
				"Loaded");
		log("SIMULATOR UPDATED"
				+ "\nDataset label: " + label
				+ "\nSource file: " + lastReport.outputFile.getAbsolutePath());
	}

	private void applyDefaultsFromSource() {
		if (sourceProfile == null) {
			return;
		}
		UciRetailDatasetImporter.Options defaults = UciRetailDatasetImporter.Options.defaultsFor(sourceProfile);
		selectedDatasetKind = defaults.datasetKind;
		updateProfileChoicesForKind(selectedDatasetKind, defaults.profile);
		outputDirField.setText(defaults.outputDirectory.getAbsolutePath());
		outputNameField.setText(defaults.outputBaseName);
		skipCancelledCheck.setSelected(defaults.skipCancelledInvoices);
		skipMissingCustomerCheck.setSelected(defaults.skipMissingCustomer);
		skipQuantityCheck.setSelected(defaults.skipNonPositiveQuantity);
		skipPriceCheck.setSelected(defaults.skipNonPositiveUnitPrice);
		stockPrefixCheck.setSelected(defaults.groupStockCodesByPrefix);
		stableFolloweeCheck.setSelected(defaults.forceSingleFolloweePerUser);
		minTokensField.setText(Integer.toString(defaults.minimumTextTokens));
		fallbackDateField.setText(defaults.defaultDate);
		balancePolicyUserSelected = false;
		UciRetailDatasetImporter.DatasetBalanceMode initialBalanceMode =
				defaults.balanceMode == UciRetailDatasetImporter.DatasetBalanceMode.OFF
						? UciRetailDatasetImporter.DatasetBalanceMode.DIAGNOSE_ONLY
						: defaults.balanceMode;
		balanceModeCombo.setSelectedItem(initialBalanceMode);
		setBalanceTargetPolicy(recommendedPolicyForMode(initialBalanceMode));
		balanceTargetField.setText(Integer.toString(defaults.balanceTargetUsersPerClass));
		balanceSeedField.setText(Long.toString(defaults.balanceSeed));
		populateColumnCombos(sourceProfile, defaults);
		updateDatasetKindUi();
		updateProfileChoicesForKind(selectedDatasetKind, defaults.profile);
		updateProfileGuide();
	}

	private void populateColumnCombos(UciRetailDatasetImporter.SourceProfile source,
			UciRetailDatasetImporter.Options defaults) {
		suppressLivePreview = true;
		JComboBox<?>[] combos = new JComboBox<?>[] {
				invoiceCombo, stockCombo, descriptionCombo, quantityCombo, dateCombo, priceCombo,
				customerCombo, countryCombo, customReferenceCombo, customPostCombo, customDateCombo,
				customUserIdCombo, customUserNameCombo, customTextCombo
		};
		try {
			DefaultComboBoxModel<String> model = columnModel(source);
			for (int i = 0; i < combos.length; i++) {
				@SuppressWarnings("unchecked")
				JComboBox<String> typed = (JComboBox<String>) combos[i];
				typed.setModel(copyModel(model));
			}
			setCombo(invoiceCombo, defaults.invoiceColumn);
			setCombo(stockCombo, defaults.stockCodeColumn);
			setCombo(descriptionCombo, defaults.descriptionColumn);
			setCombo(quantityCombo, defaults.quantityColumn);
			setCombo(dateCombo, defaults.invoiceDateColumn);
			setCombo(priceCombo, defaults.unitPriceColumn);
			setCombo(customerCombo, defaults.customerIdColumn);
			setCombo(countryCombo, defaults.countryColumn);
			setCombo(customReferenceCombo, defaults.customReferenceColumn);
			setCombo(customPostCombo, defaults.customPostIdColumn);
			setCombo(customDateCombo, defaults.customDateColumn);
			setCombo(customUserIdCombo, defaults.customUserIdColumn);
			setCombo(customUserNameCombo, defaults.customUserNameColumn);
			setCombo(customTextCombo, defaults.customTextColumn);
		} finally {
			suppressLivePreview = false;
		}
		updateLivePreview();
	}

	private UciRetailDatasetImporter.Options collectOptions() {
		UciRetailDatasetImporter.Options options = UciRetailDatasetImporter.Options.defaultsFor(sourceProfile);
		options.datasetKind = selectedDatasetKind;
		options.profile = (UciRetailDatasetImporter.MappingProfile) profileCombo.getSelectedItem();
		options.forceSingleFolloweePerUser = stableFolloweeCheck.isSelected();
		options.skipCancelledInvoices = skipCancelledCheck.isSelected();
		options.skipMissingCustomer = skipMissingCustomerCheck.isSelected();
		options.skipNonPositiveQuantity = skipQuantityCheck.isSelected();
		options.skipNonPositiveUnitPrice = skipPriceCheck.isSelected();
		options.groupStockCodesByPrefix = stockPrefixCheck.isSelected();
		options.minimumTextTokens = parsePositiveInt(minTokensField.getText(), 1);
		options.defaultDate = fallbackDateField.getText().trim().length() == 0
				? "2011-01-01" : fallbackDateField.getText().trim();
		Object balanceMode = balanceModeCombo.getSelectedItem();
		options.balanceMode = balanceMode instanceof UciRetailDatasetImporter.DatasetBalanceMode
				? (UciRetailDatasetImporter.DatasetBalanceMode) balanceMode
				: UciRetailDatasetImporter.DatasetBalanceMode.OFF;
		Object targetPolicy = balanceTargetPolicyCombo.getSelectedItem();
		options.balanceTargetPolicy =
				targetPolicy instanceof UciRetailDatasetImporter.DatasetBalanceTargetPolicy
						? (UciRetailDatasetImporter.DatasetBalanceTargetPolicy) targetPolicy
						: UciRetailDatasetImporter.recommendedTargetPolicyForMode(options.balanceMode);
		options.balanceTargetUsersPerClass =
				options.balanceTargetPolicy
						== UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MANUAL
								? parseNonNegativeInt(balanceTargetField.getText(), 0)
								: 0;
		options.balanceSeed = parseLong(balanceSeedField.getText(), 1L);
		options.outputDirectory = new File(outputDirField.getText().trim());
		options.outputBaseName = outputNameField.getText().trim();

		options.invoiceColumn = comboIndex(invoiceCombo);
		options.stockCodeColumn = comboIndex(stockCombo);
		options.descriptionColumn = comboIndex(descriptionCombo);
		options.quantityColumn = comboIndex(quantityCombo);
		options.invoiceDateColumn = comboIndex(dateCombo);
		options.unitPriceColumn = comboIndex(priceCombo);
		options.customerIdColumn = comboIndex(customerCombo);
		options.countryColumn = comboIndex(countryCombo);
		options.customReferenceColumn = comboIndex(customReferenceCombo);
		options.customPostIdColumn = comboIndex(customPostCombo);
		options.customDateColumn = comboIndex(customDateCombo);
		options.customUserIdColumn = comboIndex(customUserIdCombo);
		options.customUserNameColumn = comboIndex(customUserNameCombo);
		options.customTextColumn = comboIndex(customTextCombo);
		return options;
	}

	private void renderSourceProfile() {
		DecimalFormat fmt = new DecimalFormat("#,##0");
		rowsValue.setText(fmt.format(sourceProfile.dataRows));
		columnsValue.setText(Integer.toString(sourceProfile.columnNames.size()));
		formatValue.setText(sourceProfile.detectedKind.toString());
		int customerIdx = sourceProfile.columnIndex(
				"CustomerID", "Customer ID", "UserID", "User ID", "AuthorID",
				"Author ID", "SequenceID", "ScreenName", "UserName");
		if (!sourceProfile.hasHeader && sourceProfile.columnNames.size() >= 6) {
			customerIdx = 3;
		}
		long userEstimate = customerIdx >= 0 ? sourceProfile.columns.get(customerIdx).distinctCount() : -1L;
		usersValue.setText(userEstimate >= 0L ? fmt.format(userEstimate) : "Mapped");
		outputValue.setText("-");

		previewModel.setColumnCount(0);
		previewModel.setRowCount(0);
		for (int i = 0; i < sourceProfile.columnNames.size(); i++) {
			previewModel.addColumn(sourceProfile.columnNames.get(i));
		}
		for (List<String> row : sourceProfile.previewRows) {
			previewModel.addRow(row.toArray(new Object[row.size()]));
		}

		columnTableModel.setRowCount(0);
		for (UciRetailDatasetImporter.ColumnProfile col : sourceProfile.columns) {
			String distinct = col.distinctCount() < 0 ? "10,000+" : fmt.format(col.distinctCount());
			columnTableModel.addRow(new Object[] {
					col.name,
					fmt.format(col.nonEmptyCount),
					fmt.format(col.numericCount),
					fmt.format(col.dateLikeCount),
					distinct,
					col.sampleText()
			});
		}
		updateLivePreview();
		refreshImbalanceDiagnosis(false);
	}

	private void renderReport(UciRetailDatasetImporter.ImportReport report) {
		DecimalFormat fmt = new DecimalFormat("#,##0");
		outputValue.setText(fmt.format(report.writtenRows));
		outputPreviewModel.setRowCount(0);
		for (String[] row : report.outputPreview) {
			outputPreviewModel.addRow(row);
		}
		renderImbalanceReport(report, true);
	}


	private void refreshImbalanceDiagnosis(final boolean userRequested) {
		if (imbalanceModel == null) {
			return;
		}
		if (sourceProfile == null) {
			resetImbalancePanel("Inspect a dataset to see class imbalance, balance impact, and warnings here.");
			return;
		}
		final UciRetailDatasetImporter.Options options;
		try {
			options = collectOptions();
		} catch (Exception ex) {
			setImbalanceStatus("WAITING", Color.ORANGE.darker());
			imbalanceSummary.setText("Balance diagnosis is waiting for a valid mapping: " + rootMessage(ex));
			return;
		}
		final int requestId = ++imbalanceRequestId;
		if (imbalanceWorker != null && !imbalanceWorker.isDone()) {
			imbalanceWorker.cancel(true);
		}
		setImbalanceStatus("SCANNING", WHITE);
		refreshImbalanceButton.setEnabled(false);
		if (userRequested || lastBalanceDiagnosis == null) {
			imbalanceSummary.setText("Scanning the full accepted dataset with the current mapping and balance mode...");
		}
		imbalanceWorker = new SwingWorker<UciRetailDatasetImporter.ImportReport, Void>() {
			protected UciRetailDatasetImporter.ImportReport doInBackground() throws Exception {
				return UciRetailDatasetImporter.diagnoseBalance(sourceProfile, options, null);
			}

			protected void done() {
				if (requestId != imbalanceRequestId) {
					return;
				}
				refreshImbalanceButton.setEnabled(true);
				try {
					lastBalanceDiagnosis = get();
					renderImbalanceReport(lastBalanceDiagnosis, false);
				} catch (Exception ex) {
					setImbalanceStatus("FAILED", Color.RED.darker());
					imbalanceSummary.setText("Balance diagnosis failed: " + rootMessage(ex));
				}
			}
		};
		imbalanceWorker.execute();
	}

	private void renderImbalanceReport(UciRetailDatasetImporter.ImportReport report, boolean postBuild) {
		if (imbalanceModel == null || report == null) {
			return;
		}
		DecimalFormat fmt = new DecimalFormat("#,##0");
		DecimalFormat ratioFmt = new DecimalFormat("#,##0.0");
		imbalanceModel.setRowCount(0);
		Set<String> labels = new LinkedHashSet<String>();
		labels.addAll(report.originalFolloweeUserCounts.keySet());
		labels.addAll(report.originalFolloweeRowCounts.keySet());
		labels.addAll(report.finalFolloweeUserCounts.keySet());
		labels.addAll(report.finalFolloweeCounts.keySet());
		List<String> sortedLabels = new ArrayList<String>(labels);
		Collections.sort(sortedLabels, new Comparator<String>() {
			public int compare(String left, String right) {
				long leftCount = mapLong(report.originalFolloweeUserCounts, left);
				long rightCount = mapLong(report.originalFolloweeUserCounts, right);
				int countCompare = Long.compare(rightCount, leftCount);
				return countCompare != 0 ? countCompare : left.compareTo(right);
			}
		});

		long originalTotal = sumLongMap(report.originalFolloweeUserCounts);
		long finalTotal = sumLongMap(report.finalFolloweeUserCounts);
		long minUsers = Long.MAX_VALUE;
		long maxUsers = 0L;
		for (int i = 0; i < sortedLabels.size(); i++) {
			String label = sortedLabels.get(i);
			long users = mapLong(report.originalFolloweeUserCounts, label);
			if (users > 0L) {
				minUsers = Math.min(minUsers, users);
				maxUsers = Math.max(maxUsers, users);
			}
		}
		if (minUsers == Long.MAX_VALUE) {
			minUsers = 0L;
		}

		for (int i = 0; i < sortedLabels.size(); i++) {
			String label = sortedLabels.get(i);
			long originalUsers = mapLong(report.originalFolloweeUserCounts, label);
			long originalRows = mapLong(report.originalFolloweeRowCounts, label);
			long finalUsers = mapLong(report.finalFolloweeUserCounts, label);
			String ratio = minUsers <= 0L || originalUsers <= 0L
					? "-" : ratioFmt.format(originalUsers / (double) minUsers) + "x";
			imbalanceModel.addRow(new Object[] {
					label,
					fmt.format(originalUsers),
					fmt.format(originalRows),
					fmt.format(finalUsers),
					formatPercent(originalUsers, originalTotal),
					ratio,
					balanceAction(report, originalUsers, finalUsers)
			});
		}

		if (sortedLabels.isEmpty()) {
			imbalanceModel.addRow(new Object[] {
					"No accepted followee labels", "-", "-", "-", "-", "-", "Check mapping and validation filters"
			});
		}

		setImbalanceStatus(postBuild ? "BUILT" : "READY", WHITE);
		StringBuilder sb = new StringBuilder();
		sb.append(postBuild ? "POST-BUILD BALANCE REPORT" : "PRE-BUILD FULL-DATASET DIAGNOSIS");
		sb.append("\nSelected mode: ").append(report.balanceMode);
		if (report.effectiveBalanceMode != null && report.balanceMode != report.effectiveBalanceMode) {
			sb.append(" | Effective mode: ").append(report.effectiveBalanceMode);
		}
		sb.append(" | Target policy: ").append(report.balanceTargetPolicy);
		sb.append(" | Engine: ").append(report.balanceEngine);
		sb.append(" | Seed: ").append(report.balanceSeed);
		if (report.balanceTargetUsersPerClass > 0) {
			sb.append(" | Target/class: ").append(fmt.format(report.balanceTargetUsersPerClass));
		} else {
			sb.append(" | Target/class: automatic or unchanged");
		}
		if (report.balanceTargetDescription != null && report.balanceTargetDescription.length() > 0) {
			sb.append("\nTarget source: ").append(report.balanceTargetDescription);
		}
		sb.append("\nAccepted rows: ").append(fmt.format(report.acceptedRows));
		sb.append(" | Original users: ").append(fmt.format(report.balanceOriginalUsers));
		sb.append(" | Followee labels: ").append(fmt.format(report.originalFolloweeUserCounts.size()));
		if (maxUsers > 0L && minUsers > 0L) {
			sb.append("\nOriginal imbalance: largest class ").append(fmt.format(maxUsers));
			sb.append(" users, smallest class ").append(fmt.format(minUsers));
			sb.append(" users, ratio ").append(ratioFmt.format(maxUsers / (double) minUsers)).append("x.");
		} else {
			sb.append("\nOriginal imbalance: not enough accepted classes to compute a ratio.");
		}
		sb.append("\nSelected-mode result: ").append(fmt.format(finalTotal));
		sb.append(" final user instances, ").append(fmt.format(report.balanceDroppedUsers));
		sb.append(" dropped users, ").append(fmt.format(report.balanceSyntheticUsers));
		sb.append(" synthetic users, ").append(fmt.format(report.balanceDroppedRows)).append(" dropped rows.");
		sb.append("\nRecommendation: ").append(balanceRecommendation(report, maxUsers, minUsers));
		for (int i = 0; i < report.warnings.size(); i++) {
			sb.append("\nWarning: ").append(report.warnings.get(i));
		}
		imbalanceSummary.setText(sb.toString());
	}

	private void markImbalanceStale() {
		if (sourceProfile == null || imbalanceSummary == null) {
			return;
		}
		setImbalanceStatus("REFRESH", Color.ORANGE.darker());
		if (lastBalanceDiagnosis == null && lastReport == null) {
			imbalanceSummary.setText("Click Refresh Diagnosis to scan the full accepted dataset with the current mapping and balance mode.");
		} else {
			imbalanceSummary.setText("Mapping or balance settings changed. Click Refresh Diagnosis for updated pre-build imbalance numbers.");
		}
	}

	private void resetImbalancePanel(String message) {
		if (imbalanceModel != null) {
			imbalanceModel.setRowCount(0);
		}
		if (imbalanceSummary != null) {
			imbalanceSummary.setText(message);
		}
		setImbalanceStatus("WAITING", Color.ORANGE.darker());
		lastBalanceDiagnosis = null;
	}

	private void setImbalanceStatus(String text, Color foreground) {
		if (imbalanceStatusBadge == null) {
			return;
		}
		imbalanceStatusBadge.setText(text);
		imbalanceStatusBadge.setForeground(foreground);
	}

	private static long mapLong(Map<String, Long> values, String key) {
		Long value = values.get(key);
		return value == null ? 0L : value.longValue();
	}

	private static long sumLongMap(Map<String, Long> values) {
		long total = 0L;
		for (Long value : values.values()) {
			if (value != null) {
				total += value.longValue();
			}
		}
		return total;
	}

	private static String formatPercent(long value, long total) {
		if (total <= 0L) {
			return "-";
		}
		return new DecimalFormat("0.0").format(value * 100.0d / total) + "%";
	}

	private static String balanceAction(UciRetailDatasetImporter.ImportReport report,
			long originalUsers, long finalUsers) {
		long change = finalUsers - originalUsers;
		if (change > 0L) {
			return "Augmented +" + change + " synthetic user instance" + (change == 1L ? "" : "s");
		}
		if (change < 0L) {
			return "Downsampled " + (-change) + " original user" + (change == -1L ? "" : "s");
		}
		UciRetailDatasetImporter.DatasetBalanceMode effectiveMode =
				report.effectiveBalanceMode == null ? report.balanceMode : report.effectiveBalanceMode;
		if (report.balanceMode != effectiveMode
				&& effectiveMode == UciRetailDatasetImporter.DatasetBalanceMode.DIAGNOSE_ONLY) {
			return "Not applied - only one followee class";
		}
		if (effectiveMode == UciRetailDatasetImporter.DatasetBalanceMode.DIAGNOSE_ONLY) {
			return "Diagnosed only - unchanged";
		}
		if (effectiveMode == UciRetailDatasetImporter.DatasetBalanceMode.OFF) {
			return "Off - unchanged";
		}
		return "Kept unchanged by selected mode";
	}

	private static String balanceRecommendation(UciRetailDatasetImporter.ImportReport report,
			long maxUsers, long minUsers) {
		if (report.originalFolloweeUserCounts.size() <= 1) {
			return "Only one followee class is present, so classifier balancing is not meaningful for this mapping.";
		}
		double ratio = minUsers <= 0L ? Double.POSITIVE_INFINITY : maxUsers / (double) minUsers;
		if (ratio <= 1.5d) {
			return "The class distribution is already fairly close; Off or Diagnose only is reasonable.";
		}
		UciRetailDatasetImporter.DatasetBalanceMode effectiveMode =
				report.effectiveBalanceMode == null ? report.balanceMode : report.effectiveBalanceMode;
		if (effectiveMode == UciRetailDatasetImporter.DatasetBalanceMode.OFF
				|| effectiveMode == UciRetailDatasetImporter.DatasetBalanceMode.DIAGNOSE_ONLY) {
			return "Use Weka supervised Resample with Preserve total size for a balanced exported artifact, or Conservative cap for a no-synthetic run.";
		}
		if (report.balanceSyntheticUsers > 0) {
			return "Hybrid/Weka balancing is compensating for minority classes; report the synthetic-user count in the audit when presenting results.";
		}
		if (report.balanceDroppedUsers > 0) {
			return "Majority classes are being capped; this is conservative and avoids synthetic data.";
		}
		return "The selected mode is valid; review the final-user column for any remaining large class gaps.";
	}

	private static String formatBuildReport(UciRetailDatasetImporter.ImportReport report) {
		DecimalFormat fmt = new DecimalFormat("#,##0");
		long skipped = report.skippedCancelledInvoices
				+ report.skippedMissingCustomer
				+ report.skippedMissingText
				+ report.skippedNonPositiveQuantity
				+ report.skippedNonPositiveUnitPrice
				+ report.malformedRows;
		StringBuilder sb = new StringBuilder();
		sb.append("BUILD COMPLETE");
		sb.append("\n").append(report.summaryLine());
		sb.append("\nRaw rows: ").append(fmt.format(report.rawRows));
		sb.append(" | Accepted rows: ").append(fmt.format(report.acceptedRows));
		sb.append(" | Written rows: ").append(fmt.format(report.writtenRows));
		sb.append("\nUnique users: ").append(fmt.format(report.uniqueUsers));
		sb.append(" | Followee labels: ").append(fmt.format(report.uniqueFollowees));
		sb.append("\nBalance mode: ").append(report.balanceMode);
		if (report.effectiveBalanceMode != null && report.balanceMode != report.effectiveBalanceMode) {
			sb.append(" | Effective mode: ").append(report.effectiveBalanceMode);
		}
		sb.append(" | Target policy: ").append(report.balanceTargetPolicy);
		sb.append(" | Engine: ").append(report.balanceEngine);
		sb.append("\nBalance users: original ").append(fmt.format(report.balanceOriginalUsers));
		sb.append(", selected ").append(fmt.format(report.balanceSelectedOriginalUsers));
		sb.append(", dropped ").append(fmt.format(report.balanceDroppedUsers));
		sb.append(", synthetic ").append(fmt.format(report.balanceSyntheticUsers));
		if (report.balanceTargetUsersPerClass > 0) {
			sb.append(" | Target/class: ").append(fmt.format(report.balanceTargetUsersPerClass));
		}
		if (report.balanceTargetDescription != null && report.balanceTargetDescription.length() > 0) {
			sb.append("\nTarget source: ").append(report.balanceTargetDescription);
		}
		sb.append("\nSkipped total: ").append(fmt.format(skipped));
		sb.append(" (cancelled ").append(fmt.format(report.skippedCancelledInvoices));
		sb.append(", missing user ").append(fmt.format(report.skippedMissingCustomer));
		sb.append(", missing text ").append(fmt.format(report.skippedMissingText));
		sb.append(", quantity ").append(fmt.format(report.skippedNonPositiveQuantity));
		sb.append(", price ").append(fmt.format(report.skippedNonPositiveUnitPrice));
		sb.append(", malformed ").append(fmt.format(report.malformedRows)).append(")");
		sb.append("\nBad/fallback dates: ").append(fmt.format(report.badDateRows));
		sb.append("\nOutput: ").append(report.outputFile.getAbsolutePath());
		sb.append(" (").append(humanBytes(report.outputFile.length())).append(")");
		sb.append("\nMetadata: ").append(report.metadataFile.getAbsolutePath());
		sb.append("\nAudit: ").append(report.auditFile.getAbsolutePath());
		sb.append("\nSHA-256: ").append(report.datasetSha256);
		return sb.toString();
	}

	private void installLivePreviewListeners() {
		final ActionListener listener = new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				updateLivePreview();
				markImbalanceStale();
			}
		};
		JComboBox<?>[] combos = new JComboBox<?>[] {
				invoiceCombo, stockCombo, descriptionCombo, quantityCombo, dateCombo, priceCombo,
				customerCombo, countryCombo, customReferenceCombo, customPostCombo, customDateCombo,
				customUserIdCombo, customUserNameCombo, customTextCombo
		};
		for (int i = 0; i < combos.length; i++) {
			combos[i].addActionListener(listener);
		}
		JCheckBox[] boxes = new JCheckBox[] {
				skipCancelledCheck, skipMissingCustomerCheck, skipQuantityCheck, skipPriceCheck,
				stockPrefixCheck, stableFolloweeCheck
		};
		for (int i = 0; i < boxes.length; i++) {
			boxes[i].addActionListener(listener);
		}
		DocumentListener documentListener = new DocumentListener() {
			public void insertUpdate(DocumentEvent event) {
				updateLivePreview();
				markImbalanceStale();
			}

			public void removeUpdate(DocumentEvent event) {
				updateLivePreview();
				markImbalanceStale();
			}

			public void changedUpdate(DocumentEvent event) {
				updateLivePreview();
				markImbalanceStale();
			}
		};
		minTokensField.getDocument().addDocumentListener(documentListener);
		fallbackDateField.getDocument().addDocumentListener(documentListener);
		DocumentListener balanceDocumentListener = new DocumentListener() {
			public void insertUpdate(DocumentEvent event) {
				updateBalanceGuide();
				markImbalanceStale();
			}

			public void removeUpdate(DocumentEvent event) {
				updateBalanceGuide();
				markImbalanceStale();
			}

			public void changedUpdate(DocumentEvent event) {
				updateBalanceGuide();
				markImbalanceStale();
			}
		};
		balanceTargetField.getDocument().addDocumentListener(balanceDocumentListener);
		balanceSeedField.getDocument().addDocumentListener(balanceDocumentListener);
	}

	private void updateLivePreview() {
		if (suppressLivePreview || livePreviewModel == null) {
			return;
		}
		livePreviewModel.setRowCount(0);
		if (outputPreviewModel != null && lastReport == null) {
			outputPreviewModel.setRowCount(0);
		}
		if (sourceProfile == null) {
			updateLiveMappingSummary(null);
			if (livePreviewNote != null) {
				livePreviewNote.setText("Inspect a source file to see the live DSMP shape.");
			}
			return;
		}
		try {
			UciRetailDatasetImporter.Options options = collectOptions();
			updateLiveMappingSummary(options);
			UciRetailDatasetImporter.LivePreviewReport preview =
					UciRetailDatasetImporter.previewOutputRows(sourceProfile, options, 5);
			if (outputPreviewModel != null) {
				outputPreviewModel.setRowCount(0);
			}
			for (String[] row : preview.rows) {
				livePreviewModel.addRow(row);
				if (outputPreviewModel != null) {
					outputPreviewModel.addRow(row);
				}
			}
			String note = preview.summaryLine();
			if (stableFolloweeCheck.isSelected()
					&& options.profile != UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS) {
				note += " Final build may use the full-dataset dominant followee per user.";
			}
			livePreviewNote.setText(note);
		} catch (Exception ex) {
			updateLiveMappingSummary(null);
			if (outputPreviewModel != null && lastReport == null) {
				outputPreviewModel.setRowCount(0);
			}
			livePreviewNote.setText("Preview waiting: " + rootMessage(ex));
		}
	}

	private void updateLiveMappingSummary(UciRetailDatasetImporter.Options options) {
		if (liveMappingLabels[0] == null) {
			return;
		}
		if (sourceProfile == null || options == null) {
			for (int i = 0; i < liveMappingLabels.length; i++) {
				setLiveMapping(i, "Inspect first");
			}
			return;
		}

		UciRetailDatasetImporter.MappingProfile profile = options.profile;
		if (profile == UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS) {
			setLiveMapping(0, "Constant collection label");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.SMART_PRODUCT_CATEGORY) {
			setLiveMapping(0, "Smart retail category");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.PRODUCT_FAMILY) {
			setLiveMapping(0, "Legacy text + product");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.COUNTRY_BASELINE) {
			setLiveMapping(0, columnLabel(options.countryColumn));
		} else if (profile == UciRetailDatasetImporter.MappingProfile.STOCK_CODE_EXPERIMENTAL) {
			setLiveMapping(0, columnLabel(options.stockCodeColumn));
		} else if (profile == UciRetailDatasetImporter.MappingProfile.JOURNAL_AUTHORS
				|| profile == UciRetailDatasetImporter.MappingProfile.TWEET_USERS) {
			setLiveMapping(0, columnLabel(options.countryColumn));
		} else {
			setLiveMapping(0, columnLabel(options.customReferenceColumn));
		}

		if (profile == UciRetailDatasetImporter.MappingProfile.CUSTOM) {
			setLiveMapping(1, columnLabel(options.customPostIdColumn));
			setLiveMapping(2, columnLabel(options.customDateColumn));
			setLiveMapping(3, columnLabel(options.customUserIdColumn));
			setLiveMapping(4, columnLabel(options.customUserNameColumn));
			setLiveMapping(5, columnLabel(options.customTextColumn));
		} else {
			setLiveMapping(1, options.invoiceColumn >= 0 ? columnLabel(options.invoiceColumn) : "Generated row id");
			setLiveMapping(2, options.invoiceDateColumn >= 0 ? columnLabel(options.invoiceDateColumn) : "Fallback date");
			setLiveMapping(3, columnLabel(options.customerIdColumn));
			if ((profile == UciRetailDatasetImporter.MappingProfile.JOURNAL_AUTHORS
					|| profile == UciRetailDatasetImporter.MappingProfile.TWEET_USERS)
					&& options.customUserNameColumn >= 0) {
				setLiveMapping(4, columnLabel(options.customUserNameColumn));
			} else {
				setLiveMapping(4, "Customer_<id>");
			}
			setLiveMapping(5, columnLabel(options.descriptionColumn));
		}
	}

	private void setLiveMapping(int index, String text) {
		if (index < 0 || index >= liveMappingLabels.length || liveMappingLabels[index] == null) {
			return;
		}
		String value = text == null || text.trim().length() == 0 ? "Not selected" : text;
		String display = value.length() > 30 ? value.substring(0, 27) + "..." : value;
		liveMappingLabels[index].setText(display);
		liveMappingLabels[index].setToolTipText(value);
	}

	private String columnLabel(int column) {
		return sourceProfile == null ? "Not selected" : sourceProfile.displayColumn(column);
	}

	private void updateProfileGuide() {
		if (profileGuide == null || profileCombo == null) {
			return;
		}
		UciRetailDatasetImporter.MappingProfile profile =
				(UciRetailDatasetImporter.MappingProfile) profileCombo.getSelectedItem();
		if (profile == null) {
			profile = selectedDatasetKind == UciRetailDatasetImporter.DatasetKind.OTHER
					? UciRetailDatasetImporter.MappingProfile.CUSTOM
					: UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS;
		}
		StringBuilder sb = new StringBuilder();
		sb.append(selectedDatasetKind).append(" source\n");
		sb.append(selectedDatasetKind.description());
		sb.append("\n\nProfile\n");
		sb.append(profile.description());
		sb.append("\n\nOutput rule: DSMP stays exactly six columns; simulator algorithms and scoring logic are unchanged.");
		if (profile == UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS) {
			sb.append("\nClassifier note: all users share one followee label, so SVM/MLP are not meaningful here. Use it for user-similarity algorithms.");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.SMART_PRODUCT_CATEGORY) {
			sb.append("\nClassifier note: this scans each retail description with a broad category taxonomy, then keeps each customer's dominant category as a stable SVM/MLP target.");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.PRODUCT_FAMILY) {
			sb.append("\nCompatibility note: this is the older compact product-family mapper. Prefer smart product category for new UCI Online Retail work.");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.STOCK_CODE_EXPERIMENTAL) {
			sb.append("\nRisk note: high-cardinality product labels can make SVM/MLP heavy. Prefix grouping is strongly recommended.");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.JOURNAL_AUTHORS) {
			sb.append("\nJournal note: the venue/journal becomes the followee label, authors become users, and titles/abstracts become text.");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.TWEET_USERS) {
			sb.append("\nTweet note: the reference/source account becomes the followee label, Twitter users become users, and tweet text becomes the document body.");
		} else if (profile == UciRetailDatasetImporter.MappingProfile.CUSTOM) {
			sb.append("\nCustom mode: choose every DSMP field yourself. The importer still validates six-column compatibility before writing.");
		}
		profileGuide.setText(sb.toString());
		if (customMappingPanel != null) {
			customMappingPanel.setVisible(profile == UciRetailDatasetImporter.MappingProfile.CUSTOM);
		}
		updateLivePreview();
		markImbalanceStale();
		revalidate();
		repaint();
	}

	private void updateBalanceGuide() {
		if (balanceGuide == null || balanceModeCombo == null || balanceTargetPolicyCombo == null) {
			return;
		}
		UciRetailDatasetImporter.DatasetBalanceMode mode =
				(UciRetailDatasetImporter.DatasetBalanceMode) balanceModeCombo.getSelectedItem();
		if (mode == null) {
			mode = UciRetailDatasetImporter.DatasetBalanceMode.OFF;
		}
		UciRetailDatasetImporter.DatasetBalanceTargetPolicy policy =
				(UciRetailDatasetImporter.DatasetBalanceTargetPolicy) balanceTargetPolicyCombo.getSelectedItem();
		if (policy == null) {
			policy = recommendedPolicyForMode(mode);
		}
		boolean manual = policy == UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MANUAL;
		balanceTargetField.setEnabled(manual);
		int target = parseNonNegativeInt(balanceTargetField.getText(), 0);
		long seed = parseLong(balanceSeedField.getText(), 1L);
		String targetText = manual
				? (target <= 0 ? "waiting for a positive manual target" : target + " users per followee class")
				: policy.description();
		StringBuilder sb = new StringBuilder();
		sb.append(mode.description());
		sb.append("\n\nScope: after validation and stable followee selection, before writing the DSMP file.");
		sb.append("\nTarget policy: ").append(policy).append(".");
		sb.append("\nTarget meaning: ").append(targetText).append(".");
		sb.append("\nSeed: ").append(seed).append(" for repeatable sampling.");
		if (mode == UciRetailDatasetImporter.DatasetBalanceMode.DIAGNOSE_ONLY
				|| mode == UciRetailDatasetImporter.DatasetBalanceMode.OFF) {
			sb.append("\nRecommended workflow: diagnose first; use Weka supervised Resample with Preserve total size when you need a balanced exported artifact.");
		} else if (mode == UciRetailDatasetImporter.DatasetBalanceMode.STRICT_USER_BALANCE
				|| mode == UciRetailDatasetImporter.DatasetBalanceMode.WEKA_SPREAD_SUBSAMPLE) {
			sb.append("\nStrict downsampling modes cap any target above the smallest class because they do not invent users.");
		}
		if (mode == UciRetailDatasetImporter.DatasetBalanceMode.HYBRID_CAP_AND_AUGMENT) {
			sb.append("\nSynthetic users get deterministic DSMP user/post ids while keeping the original text.");
		} else if (mode == UciRetailDatasetImporter.DatasetBalanceMode.WEKA_SPREAD_SUBSAMPLE
				|| mode == UciRetailDatasetImporter.DatasetBalanceMode.WEKA_SUPERVISED_RESAMPLE) {
			sb.append("\nWeka runs over one instance per DSMP user/class, then selected users are written as DSMP rows.");
		}
		balanceGuide.setText(sb.toString());
	}

	private void applyRecommendedBalanceTargetPolicyIfUnpinned() {
		if (!balancePolicyUserSelected && balanceModeCombo != null && balanceTargetPolicyCombo != null) {
			UciRetailDatasetImporter.DatasetBalanceMode mode =
					(UciRetailDatasetImporter.DatasetBalanceMode) balanceModeCombo.getSelectedItem();
			setBalanceTargetPolicy(recommendedPolicyForMode(mode));
		}
	}

	private void setBalanceTargetPolicy(UciRetailDatasetImporter.DatasetBalanceTargetPolicy policy) {
		if (balanceTargetPolicyCombo == null || policy == null) {
			return;
		}
		suppressBalancePolicyEvents = true;
		try {
			balanceTargetPolicyCombo.setSelectedItem(policy);
		} finally {
			suppressBalancePolicyEvents = false;
		}
	}

	private static UciRetailDatasetImporter.DatasetBalanceTargetPolicy recommendedPolicyForMode(
			UciRetailDatasetImporter.DatasetBalanceMode mode) {
		return UciRetailDatasetImporter.recommendedTargetPolicyForMode(mode);
	}

	private void setIdleState() {
		inspectButton.setEnabled(false);
		buildButton.setEnabled(false);
		useOutputButton.setEnabled(false);
		resetButton.setEnabled(true);
		updateDatasetKindUi();
		updateBalanceGuide();
		resetImbalancePanel("Inspect a dataset to see class imbalance, balance impact, and warnings here.");
		log("Ready.\nChoose Retail, Journal, Tweets, or Other, then inspect a source file before building DSMP output.");
	}

	private void setBusy(boolean busy) {
		inspectButton.setEnabled(!busy && selectedFile != null);
		buildButton.setEnabled(!busy && sourceProfile != null);
		resetButton.setEnabled(!busy);
		useOutputButton.setEnabled(!busy && lastReport != null
				&& lastReport.outputFile != null && lastReport.outputFile.isFile());
		if (refreshImbalanceButton != null) {
			refreshImbalanceButton.setEnabled(!busy && sourceProfile != null);
		}
		setCursor(Cursor.getPredefinedCursor(busy ? Cursor.WAIT_CURSOR : Cursor.DEFAULT_CURSOR));
	}

	private void setStatus(String state, String message, int progress) {
		statusLabel.setText(state);
		if ("FAILED".equals(state)) {
			statusLabel.setForeground(Color.RED.darker());
		} else {
			statusLabel.setForeground(WHITE);
		}
		progressBar.setValue(Math.max(0, Math.min(100, progress)));
		progressBar.setString(message);
		updateConsoleStats(state, null, null);
	}

	private void log(String message) {
		if (logArea == null) {
			return;
		}
		consoleEventCount++;
		if (eventCountValue != null) {
			eventCountValue.setText(Integer.toString(consoleEventCount));
		}
		String stamp = new SimpleDateFormat("HH:mm:ss").format(new Date());
		String normalized = message == null ? "" : message.trim();
		normalized = normalized.replace("\r\n", "\n").replace('\r', '\n');
		normalized = normalized.replace("\n", "\n           ");
		if (logArea.getText().length() > 0) {
			logArea.append("\n");
		}
		logArea.append("[" + stamp + "] " + normalized);
		logArea.setCaretPosition(logArea.getDocument().getLength());
	}

	private void logProgress(String phase, String message, int percent) {
		int bucket = Math.max(0, Math.min(5, percent / 20));
		if ("INSPECT".equals(phase)) {
			if (bucket == inspectProgressBucket && percent < 100) {
				return;
			}
			inspectProgressBucket = bucket;
		} else {
			if (bucket == buildProgressBucket && percent < 100) {
				return;
			}
			buildProgressBucket = bucket;
		}
		log(phase + " " + Math.max(0, Math.min(100, percent)) + "% - " + message);
	}

	private void updateConsoleStats(String phase, String rows, String output) {
		if (phaseValue != null && phase != null) {
			phaseValue.setText(phase);
		}
		if (rowsConsoleValue != null && rows != null) {
			rowsConsoleValue.setText(rows);
		}
		if (outputConsoleValue != null && output != null) {
			String value = output.trim().length() == 0 ? "-" : output;
			outputConsoleValue.setText(value.length() > 24 ? value.substring(0, 21) + "..." : value);
			outputConsoleValue.setToolTipText(value);
		}
	}

	private void clearTables() {
		if (previewModel != null) {
			previewModel.setRowCount(0);
			previewModel.setColumnCount(0);
		}
		if (columnTableModel != null) {
			columnTableModel.setRowCount(0);
		}
		if (outputPreviewModel != null) {
			outputPreviewModel.setRowCount(0);
		}
		if (livePreviewModel != null) {
			livePreviewModel.setRowCount(0);
		}
		if (livePreviewNote != null) {
			livePreviewNote.setText("Inspect a source file to see the live DSMP shape.");
		}
		resetImbalancePanel("Inspect a dataset to see class imbalance, balance impact, and warnings here.");
		updateLiveMappingSummary(null);
		if (rowsValue != null) {
			rowsValue.setText("-");
			columnsValue.setText("-");
			formatValue.setText("-");
			usersValue.setText("-");
			outputValue.setText("-");
		}
	}

	private static DefaultComboBoxModel<String> columnModel(UciRetailDatasetImporter.SourceProfile source) {
		DefaultComboBoxModel<String> model = new DefaultComboBoxModel<String>();
		model.addElement("Not selected");
		for (int i = 0; i < source.columnNames.size(); i++) {
			model.addElement((i + 1) + " - " + source.columnNames.get(i));
		}
		return model;
	}

	private static DefaultComboBoxModel<String> copyModel(DefaultComboBoxModel<String> source) {
		DefaultComboBoxModel<String> copy = new DefaultComboBoxModel<String>();
		for (int i = 0; i < source.getSize(); i++) {
			copy.addElement(source.getElementAt(i));
		}
		return copy;
	}

	private static void setCombo(JComboBox<String> combo, int column) {
		combo.setSelectedIndex(column < 0 ? 0 : column + 1);
	}

	private static int comboIndex(JComboBox<String> combo) {
		return combo.getSelectedIndex() <= 0 ? -1 : combo.getSelectedIndex() - 1;
	}

	private static int parsePositiveInt(String value, int fallback) {
		try {
			int parsed = Integer.parseInt(value.trim());
			return parsed > 0 ? parsed : fallback;
		} catch (Exception ex) {
			return fallback;
		}
	}

	private static int parseNonNegativeInt(String value, int fallback) {
		try {
			int parsed = Integer.parseInt(value.trim());
			return parsed >= 0 ? parsed : fallback;
		} catch (Exception ex) {
			return fallback;
		}
	}

	private static long parseLong(String value, long fallback) {
		try {
			return Long.parseLong(value.trim());
		} catch (Exception ex) {
			return fallback;
		}
	}

	private static String safeBase(String filename) {
		String raw = filename == null ? "ImportedDataset" : filename;
		int dot = raw.lastIndexOf('.');
		if (dot > 0) {
			raw = raw.substring(0, dot);
		}
		return raw.replaceAll("[^A-Za-z0-9_-]+", "_");
	}

	private static String rootMessage(Throwable ex) {
		Throwable t = ex;
		while (t.getCause() != null) {
			t = t.getCause();
		}
		return t.getMessage() == null ? t.toString() : t.getMessage();
	}

	private static String humanBytes(long bytes) {
		if (bytes < 1024L) {
			return bytes + " B";
		}
		double value = bytes;
		String[] units = new String[] { "KB", "MB", "GB", "TB" };
		int unit = -1;
		do {
			value /= 1024.0;
			unit++;
		} while (value >= 1024.0 && unit < units.length - 1);
		return new DecimalFormat("#,##0.0").format(value) + " " + units[unit];
	}

	private static JPanel oldPanel() {
		return oldPanel(new FlowLayout());
	}

	private static JPanel oldPanel(java.awt.LayoutManager layout) {
		JPanel panel = new JPanel(layout);
		panel.setOpaque(true);
		panel.setBackground(DSMP_BLUE);
		return panel;
	}

	private static void sizePanel(Component component, int minimumWidth, int preferredWidth) {
		component.setMinimumSize(new Dimension(minimumWidth, 0));
		component.setPreferredSize(new Dimension(preferredWidth, PANEL_PREF_HEIGHT));
	}

	private static void installLeftPinnedDivider(final JSplitPane split,
			final int minimumLeftWidth, final int preferredLeftWidth, final double maximumShare) {
		ComponentAdapter listener = new ComponentAdapter() {
			public void componentResized(ComponentEvent event) {
				positionLeftPinnedDivider(split, minimumLeftWidth, preferredLeftWidth, maximumShare);
			}

			public void componentShown(ComponentEvent event) {
				positionLeftPinnedDivider(split, minimumLeftWidth, preferredLeftWidth, maximumShare);
			}
		};
		split.addComponentListener(listener);
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				positionLeftPinnedDivider(split, minimumLeftWidth, preferredLeftWidth, maximumShare);
			}
		});
	}

	private static void installRightPinnedDivider(final JSplitPane split,
			final int minimumRightWidth, final int preferredRightWidth, final double maximumShare) {
		ComponentAdapter listener = new ComponentAdapter() {
			public void componentResized(ComponentEvent event) {
				positionRightPinnedDivider(split, minimumRightWidth, preferredRightWidth, maximumShare);
			}

			public void componentShown(ComponentEvent event) {
				positionRightPinnedDivider(split, minimumRightWidth, preferredRightWidth, maximumShare);
			}
		};
		split.addComponentListener(listener);
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				positionRightPinnedDivider(split, minimumRightWidth, preferredRightWidth, maximumShare);
			}
		});
	}

	private static void positionLeftPinnedDivider(JSplitPane split,
			int minimumLeftWidth, int preferredLeftWidth, double maximumShare) {
		int available = split.getWidth() - split.getDividerSize();
		if (available <= 0) {
			return;
		}
		int rightMinimum = split.getRightComponent().getMinimumSize().width;
		int responsiveWidth = (int) Math.round(available * maximumShare);
		int leftWidth = clamp(responsiveWidth, minimumLeftWidth, preferredLeftWidth);
		if (available - leftWidth < rightMinimum) {
			leftWidth = Math.max(0, available - rightMinimum);
		}
		split.setDividerLocation(clamp(leftWidth, 0, available));
	}

	private static void positionRightPinnedDivider(JSplitPane split,
			int minimumRightWidth, int preferredRightWidth, double maximumShare) {
		int available = split.getWidth() - split.getDividerSize();
		if (available <= 0) {
			return;
		}
		int leftMinimum = split.getLeftComponent().getMinimumSize().width;
		int responsiveWidth = (int) Math.round(available * maximumShare);
		int rightWidth = clamp(responsiveWidth, minimumRightWidth, preferredRightWidth);
		if (available - rightWidth < leftMinimum) {
			rightWidth = Math.max(0, available - leftMinimum);
		}
		split.setDividerLocation(clamp(available - rightWidth, 0, available));
	}

	private static int clamp(int value, int minimum, int maximum) {
		return Math.max(minimum, Math.min(maximum, value));
	}

	private static JScrollPane tableScroll(JTable table) {
		JScrollPane scroll = new JScrollPane(table);
		scroll.setMinimumSize(new Dimension(160, 120));
		scroll.setPreferredSize(new Dimension(360, 260));
		return scroll;
	}

	private static Border oldTitle(String title) {
		TitledBorder border = BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), title);
		border.setTitleJustification(TitledBorder.CENTER);
		border.setTitleFont(TITLE_FONT);
		border.setTitleColor(WHITE);
		return border;
	}

	private static JLabel oldLabel(String text) {
		JLabel label = new JLabel(text);
		label.setForeground(WHITE);
		label.setFont(LABEL_FONT);
		return label;
	}

	private static JTextField textField(String value) {
		JTextField field = new JTextField(value);
		field.setFont(LABEL_FONT);
		field.setHorizontalAlignment(JTextField.CENTER);
		field.setForeground(INK);
		return field;
	}

	private static JTextArea textArea(int rows, int cols) {
		JTextArea area = new JTextArea(rows, cols);
		area.setEditable(false);
		area.setLineWrap(true);
		area.setWrapStyleWord(true);
		area.setFont(new Font("Arial", Font.PLAIN, 12));
		area.setBorder(BorderFactory.createLineBorder(LINE));
		return area;
	}

	private static JComboBox<String> combo() {
		JComboBox<String> combo = new JComboBox<String>();
		combo.setFont(LABEL_FONT);
		return combo;
	}

	private static JCheckBox oldCheck(String label, boolean selected) {
		JCheckBox box = new JCheckBox(label, selected);
		box.setOpaque(false);
		box.setForeground(WHITE);
		box.setFont(LABEL_FONT);
		return box;
	}

	private static JButton oldButton(String label) {
		JButton button = new JButton(label);
		button.setFont(BUTTON_FONT);
		button.setFocusPainted(false);
		button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
		return button;
	}

	private static JTable styledTable(DefaultTableModel model) {
		JTable table = new JTable(model);
		table.setRowHeight(24);
		table.setShowGrid(true);
		table.setGridColor(Color.LIGHT_GRAY);
		table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		JTableHeader header = table.getTableHeader();
		header.setFont(LABEL_FONT);
		header.setBackground(WHITE);
		header.setForeground(INK);
		return table;
	}

	private static GridBagConstraints formConstraints() {
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.insets = new Insets(3, 4, 3, 4);
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.anchor = GridBagConstraints.WEST;
		return gbc;
	}

	private static void addFormRow(JPanel panel, GridBagConstraints gbc, int row, String label, Component component) {
		addFormRow(panel, gbc, row, oldLabel(label), component);
	}

	private static void addFormRow(JPanel panel, GridBagConstraints gbc, int row, JLabel label, Component component) {
		gbc.gridy = row;
		gbc.gridx = 0;
		gbc.weightx = 0.0;
		gbc.gridwidth = 1;
		panel.add(label, gbc);
		gbc.gridx = 1;
		gbc.weightx = 1.0;
		panel.add(component, gbc);
	}

	private static void addWide(JPanel panel, GridBagConstraints gbc, int row, Component component) {
		gbc.gridy = row;
		gbc.gridx = 0;
		gbc.gridwidth = 2;
		gbc.weightx = 1.0;
		panel.add(component, gbc);
		gbc.gridwidth = 1;
	}

	private static void addButtonRow(JPanel panel, GridBagConstraints gbc, int row, JButton left, JButton right) {
		JPanel buttons = oldPanel(new GridLayout(1, 2, 6, 0));
		buttons.add(left);
		buttons.add(right);
		addWide(panel, gbc, row, buttons);
	}
}
