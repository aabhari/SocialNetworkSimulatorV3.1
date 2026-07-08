package TwitterGatherDataFollowers.userRyersonU;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Converts raw source files into the six-column text format consumed by the
 * existing DSMP simulator. This class intentionally stays outside all
 * recommendation algorithms so importing a dataset cannot alter scoring.
 */
final class UciRetailDatasetImporter {
    private static final int PREVIEW_LIMIT = 60;
    private static final int DISTINCT_LIMIT = 10000;
    private static final String DEFAULT_COLLECTION_FOLLOWEE = "OnlineRetail";

    private UciRetailDatasetImporter() {
    }

    enum DatasetKind {
        RETAIL(
                "Retail",
                "Transactions, products, customers, dates, prices, and descriptions."),
        JOURNAL(
                "Journal",
                "Authors, papers or titles, venues or journals, and optional publication dates."),
        TWEETS(
                "Tweets",
                "Twitter-style rows with reference accounts, tweet ids, users, dates, and text."),
        OTHER(
                "Other",
                "Any delimited or workbook data source. Requires explicit custom DSMP mapping.");

        private final String label;
        private final String description;

        DatasetKind(String label, String description) {
            this.label = label;
            this.description = description;
        }

        String description() {
            return description;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    enum MappingProfile {
        SIMILAR_CUSTOMERS(
                "Similar users / entities",
                "One shared collection label. Best for Doc2Vec, Similarity, and K-Means when the goal is to recommend users/entities by document text alone."),
        SMART_PRODUCT_CATEGORY(
                "Retail: smart product category",
                "Scores every product description against a broad UCI-retail taxonomy, then assigns each customer their dominant category. Recommended retail classifier target for SVM and MLP."),
        PRODUCT_FAMILY(
                "Retail: legacy product family",
                "Uses the older compact keyword fallback. Kept for compatibility with previous imports; smart product category is usually a better retail classifier target."),
        COUNTRY_BASELINE(
                "Retail: country baseline",
                "Uses each customer's dominant country as the followee label. Useful as a sanity check, but less meaningful as a retail recommendation target."),
        STOCK_CODE_EXPERIMENTAL(
                "Retail: stock code experimental",
                "Uses dominant stock-code group as the followee label. Closest to customer-product edges, but can create many classes and stress classifiers."),
        JOURNAL_AUTHORS(
                "Journal: authors by venue",
                "Uses the journal, venue, or publication column as the DSMP followee and title text as the document body. This matches the author-paper-journal interpretation."),
        TWEET_USERS(
                "Tweets: users by reference account",
                "Uses the referenced account or source account as the DSMP followee and tweet text as the document body."),
        CUSTOM(
                "Other / custom DSMP mapping",
                "Use selected columns from any delimited file and decide exactly which column becomes the DSMP followee, user, date, and text.");

        private final String label;
        private final String description;

        MappingProfile(String label, String description) {
            this.label = label;
            this.description = description;
        }

        String description() {
            return description;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    enum Delimiter {
        COMMA(',', "Comma"),
        TAB('\t', "Tab"),
        SEMICOLON(';', "Semicolon"),
        PIPE('|', "Pipe");

        final char value;
        final String label;

        Delimiter(char value, String label) {
            this.value = value;
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    interface ProgressListener {
        void onProgress(String message, int percent);
    }

    static final class ColumnProfile {
        final int index;
        final String name;
        long nonEmptyCount;
        long numericCount;
        long dateLikeCount;
        boolean distinctCapped;
        final Set<String> distinctValues = new LinkedHashSet<String>();
        final List<String> samples = new ArrayList<String>();

        ColumnProfile(int index, String name) {
            this.index = index;
            this.name = name;
        }

        void observe(String value) {
            String trimmed = value == null ? "" : value.trim();
            if (trimmed.length() == 0) {
                return;
            }
            nonEmptyCount++;
            if (parseDouble(trimmed) != null) {
                numericCount++;
            }
            if (parseFlexibleDate(trimmed) != null) {
                dateLikeCount++;
            }
            if (!distinctCapped) {
                if (distinctValues.size() < DISTINCT_LIMIT) {
                    distinctValues.add(trimmed);
                } else {
                    distinctCapped = true;
                    distinctValues.clear();
                }
            }
            if (samples.size() < 5 && !samples.contains(trimmed)) {
                samples.add(trimmed);
            }
        }

        long distinctCount() {
            return distinctCapped ? -1 : distinctValues.size();
        }

        String sampleText() {
            if (samples.isEmpty()) {
                return "";
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < samples.size(); i++) {
                if (i > 0) {
                    sb.append(" | ");
                }
                sb.append(samples.get(i));
            }
            return sb.toString();
        }
    }

    static final class SourceProfile {
        final File inputFile;
        final File dataFile;
        final DatasetKind selectedKind;
        final DatasetKind detectedKind;
        final String sourceFormat;
        final Delimiter delimiter;
        final boolean hasHeader;
        final List<String> columnNames;
        final List<ColumnProfile> columns;
        final List<List<String>> previewRows;
        final List<SampleRow> sampleRows;
        final long dataRows;
        final long malformedRows;
        final boolean recognizedUciRetail;
        final boolean recognizedJournal;
        final boolean recognizedTweets;

        SourceProfile(File inputFile, File dataFile, DatasetKind selectedKind, DatasetKind detectedKind,
                String sourceFormat, Delimiter delimiter,
                boolean hasHeader, List<String> columnNames,
                List<ColumnProfile> columns, List<List<String>> previewRows, List<SampleRow> sampleRows, long dataRows,
                long malformedRows, boolean recognizedUciRetail, boolean recognizedJournal,
                boolean recognizedTweets) {
            this.inputFile = inputFile;
            this.dataFile = dataFile;
            this.selectedKind = selectedKind == null ? DatasetKind.RETAIL : selectedKind;
            this.detectedKind = detectedKind == null ? this.selectedKind : detectedKind;
            this.sourceFormat = sourceFormat;
            this.delimiter = delimiter;
            this.hasHeader = hasHeader;
            this.columnNames = Collections.unmodifiableList(columnNames);
            this.columns = Collections.unmodifiableList(columns);
            this.previewRows = Collections.unmodifiableList(previewRows);
            this.sampleRows = Collections.unmodifiableList(sampleRows);
            this.dataRows = dataRows;
            this.malformedRows = malformedRows;
            this.recognizedUciRetail = recognizedUciRetail;
            this.recognizedJournal = recognizedJournal;
            this.recognizedTweets = recognizedTweets;
        }

        int columnIndex(String... names) {
            for (String wanted : names) {
                for (int i = 0; i < columnNames.size(); i++) {
                    String actual = normalizeHeader(columnNames.get(i));
                    if (actual.equals(normalizeHeader(wanted))) {
                        return i;
                    }
                }
            }
            return -1;
        }

        String displayColumn(int index) {
            if (index < 0 || index >= columnNames.size()) {
                return "Not selected";
            }
            return columnNames.get(index);
        }
    }

    static final class Options {
        File inputFile;
        File outputDirectory;
        String outputBaseName;
        DatasetKind datasetKind = DatasetKind.RETAIL;
        MappingProfile profile = MappingProfile.SIMILAR_CUSTOMERS;
        boolean forceSingleFolloweePerUser = true;
        boolean skipCancelledInvoices = true;
        boolean skipMissingCustomer = true;
        boolean skipNonPositiveQuantity = true;
        boolean skipNonPositiveUnitPrice = false;
        boolean groupStockCodesByPrefix = true;
        int minimumTextTokens = 1;
        String defaultDate = "2011-01-01";

        int invoiceColumn = -1;
        int stockCodeColumn = -1;
        int descriptionColumn = -1;
        int quantityColumn = -1;
        int invoiceDateColumn = -1;
        int unitPriceColumn = -1;
        int customerIdColumn = -1;
        int countryColumn = -1;

        int customReferenceColumn = -1;
        int customPostIdColumn = -1;
        int customDateColumn = -1;
        int customUserIdColumn = -1;
        int customUserNameColumn = -1;
        int customTextColumn = -1;

        static Options defaultsFor(SourceProfile source) {
            Options options = new Options();
            DatasetKind kind = source == null || source.selectedKind == null
                    ? DatasetKind.RETAIL : source.selectedKind;
            options.datasetKind = kind;
            options.inputFile = source.inputFile;
            options.outputDirectory = defaultOutputDirectory(source.inputFile);
            options.outputBaseName = safeFileBase(source.inputFile.getName()) + "_DSMP";

            if (kind == DatasetKind.JOURNAL) {
                options.invoiceColumn = inferColumn(source, 1,
                        "ArticleID", "Article ID", "PaperID", "Paper ID", "PublicationID",
                        "RecordID", "PostID", "TweetID", "ID");
                options.stockCodeColumn = source.columnIndex(
                        "Subject", "Category", "Discipline", "Keyword", "Keywords");
                options.descriptionColumn = inferColumn(source, 5,
                        "Title", "PaperTitle", "Paper Title", "ArticleTitle", "Article Title",
                        "DocumentTitle", "Document Title", "Text", "Abstract");
                options.quantityColumn = -1;
                options.invoiceDateColumn = inferColumn(source, 2,
                        "Date", "PublicationDate", "Publication Date", "Published",
                        "Year", "Timestamp");
                options.unitPriceColumn = -1;
                options.customerIdColumn = inferColumn(source, 3,
                        "AuthorID", "Author ID", "UserID", "User ID", "SequenceID",
                        "Sequence ID", "FollowerID", "Follower ID", "Author",
                        "AuthorName", "Author Name");
                options.countryColumn = inferColumn(source, 0,
                        "Journal", "JournalName", "Journal Name", "Venue", "Publication",
                        "Conference", "Followee", "Reference", "ReferenceUser");
                options.customUserNameColumn = inferColumn(source, 4,
                        "Author", "AuthorName", "Author Name", "UserName", "User Name", "Name");
                options.customReferenceColumn = firstPresent(options.countryColumn, 0);
                options.customPostIdColumn = firstPresent(options.invoiceColumn, 1);
                options.customDateColumn = firstPresent(options.invoiceDateColumn, 2);
                options.customUserIdColumn = firstPresent(options.customerIdColumn, 3);
                options.customTextColumn = firstPresent(options.descriptionColumn, source.columnNames.size() > 0 ? source.columnNames.size() - 1 : -1);
                options.profile = MappingProfile.JOURNAL_AUTHORS;
                options.skipCancelledInvoices = false;
                options.skipNonPositiveQuantity = false;
                options.skipNonPositiveUnitPrice = false;
                options.groupStockCodesByPrefix = false;
                options.defaultDate = "2000-01-01";
            } else if (kind == DatasetKind.TWEETS) {
                options.invoiceColumn = inferColumn(source, 1,
                        "TweetID", "Tweet ID", "StatusID", "Status ID", "ID", "Id",
                        "id_str", "PostID", "Post ID");
                options.stockCodeColumn = source.columnIndex(
                        "Hashtag", "Hashtags", "Topic", "Category");
                options.descriptionColumn = inferColumn(source, 5,
                        "Text", "Tweet", "TweetText", "Tweet Text", "FullText", "Full Text",
                        "full_text", "Body");
                options.quantityColumn = -1;
                options.invoiceDateColumn = inferColumn(source, 2,
                        "Date", "CreatedAt", "Created At", "created_at", "Timestamp",
                        "TweetDate", "Tweet Date");
                options.unitPriceColumn = -1;
                options.customerIdColumn = inferColumn(source, 3,
                        "UserID", "User ID", "AuthorID", "Author ID", "TwitterID",
                        "Twitter ID", "UserName", "User Name", "ScreenName",
                        "Screen Name", "screen_name");
                options.countryColumn = inferColumn(source, 0,
                        "Followee", "Reference", "ReferenceUser", "Reference User", "Target",
                        "Account", "SourceAccount", "Source Account", "ScreenName");
                options.customUserNameColumn = inferColumn(source, 4,
                        "UserName", "User Name", "ScreenName", "Screen Name", "screen_name",
                        "Author", "Name");
                options.customReferenceColumn = firstPresent(options.countryColumn, 0);
                options.customPostIdColumn = firstPresent(options.invoiceColumn, 1);
                options.customDateColumn = firstPresent(options.invoiceDateColumn, 2);
                options.customUserIdColumn = firstPresent(options.customerIdColumn, 3);
                options.customTextColumn = firstPresent(options.descriptionColumn, source.columnNames.size() > 0 ? source.columnNames.size() - 1 : -1);
                options.profile = MappingProfile.TWEET_USERS;
                options.skipCancelledInvoices = false;
                options.skipNonPositiveQuantity = false;
                options.skipNonPositiveUnitPrice = false;
                options.groupStockCodesByPrefix = false;
                options.defaultDate = "2011-01-01";
            } else if (kind == DatasetKind.OTHER) {
                options.invoiceColumn = inferColumn(source, 1,
                        "ID", "RecordID", "Record ID", "PostID", "Post ID");
                options.stockCodeColumn = -1;
                options.descriptionColumn = inferColumn(source, 5,
                        "Text", "Title", "Description", "Body", "Document", "Content");
                options.quantityColumn = -1;
                options.invoiceDateColumn = inferColumn(source, 2,
                        "Date", "Timestamp", "CreatedAt", "Created At");
                options.unitPriceColumn = -1;
                options.customerIdColumn = inferColumn(source, 3,
                        "UserID", "User ID", "AuthorID", "Author ID", "CustomerID",
                        "Customer ID");
                options.countryColumn = inferColumn(source, 0,
                        "Followee", "Reference", "ReferenceUser", "Reference User",
                        "Target", "Class", "Label");
                options.customReferenceColumn = firstPresent(options.countryColumn, 0);
                options.customPostIdColumn = firstPresent(options.invoiceColumn, 1);
                options.customDateColumn = firstPresent(options.invoiceDateColumn, 2);
                options.customUserIdColumn = firstPresent(options.customerIdColumn, 3);
                options.customUserNameColumn = inferColumn(source, 4,
                        "UserName", "User Name", "Author", "AuthorName", "EntityName",
                        "Entity Name", "Name");
                options.customTextColumn = firstPresent(options.descriptionColumn, source.columnNames.size() > 0 ? source.columnNames.size() - 1 : -1);
                options.profile = MappingProfile.CUSTOM;
                options.skipCancelledInvoices = false;
                options.skipNonPositiveQuantity = false;
                options.skipNonPositiveUnitPrice = false;
                options.groupStockCodesByPrefix = false;
            } else {
                options.invoiceColumn = source.columnIndex("InvoiceNo", "Invoice", "OrderId", "Order ID");
                options.stockCodeColumn = source.columnIndex("StockCode", "Stock Code", "ProductID", "Product ID", "SKU");
                options.descriptionColumn = source.columnIndex("Description", "ProductDescription", "ItemDescription", "Title", "Text");
                options.quantityColumn = source.columnIndex("Quantity", "Qty");
                options.invoiceDateColumn = source.columnIndex("InvoiceDate", "Invoice Date", "Date", "CreatedDate", "Timestamp");
                options.unitPriceColumn = source.columnIndex("UnitPrice", "Unit Price", "Price");
                options.customerIdColumn = source.columnIndex("CustomerID", "Customer ID", "UserID", "User ID", "AuthorID");
                options.countryColumn = source.columnIndex("Country", "Nation", "Region");

                options.customReferenceColumn = firstPresent(options.countryColumn, options.stockCodeColumn, 0);
                options.customPostIdColumn = firstPresent(options.invoiceColumn, -1);
                options.customDateColumn = firstPresent(options.invoiceDateColumn, -1);
                options.customUserIdColumn = firstPresent(options.customerIdColumn, -1);
                options.customUserNameColumn = firstPresent(options.customerIdColumn, -1);
                options.customTextColumn = firstPresent(options.descriptionColumn, source.columnNames.size() > 0 ? source.columnNames.size() - 1 : -1);
                options.profile = MappingProfile.SIMILAR_CUSTOMERS;
            }
            return options;
        }

        private static int inferColumn(SourceProfile source, int dsmpNoHeaderIndex, String... names) {
            int named = source.columnIndex(names);
            if (named >= 0) {
                return named;
            }
            if (!source.hasHeader && source.columnNames.size() >= 6
                    && dsmpNoHeaderIndex >= 0
                    && dsmpNoHeaderIndex < source.columnNames.size()) {
                return dsmpNoHeaderIndex;
            }
            return -1;
        }

        private static int firstPresent(int... values) {
            for (int value : values) {
                if (value >= 0) {
                    return value;
                }
            }
            return -1;
        }
    }

    static final class ImportReport {
        File outputFile;
        File metadataFile;
        File auditFile;
        long rawRows;
        long acceptedRows;
        long writtenRows;
        long malformedRows;
        long skippedCancelledInvoices;
        long skippedMissingCustomer;
        long skippedMissingText;
        long skippedNonPositiveQuantity;
        long skippedNonPositiveUnitPrice;
        long badDateRows;
        int uniqueUsers;
        int uniqueFollowees;
        String datasetSha256;
        final Map<String, Long> finalFolloweeCounts = new LinkedHashMap<String, Long>();
        final List<String> warnings = new ArrayList<String>();
        final List<String[]> outputPreview = new ArrayList<String[]>();

        String summaryLine() {
            return "Wrote " + writtenRows + " DSMP rows for " + uniqueUsers + " users and "
                    + uniqueFollowees + " followee labels.";
        }
    }

    static final class LivePreviewReport {
        long sampledRows;
        long acceptedRows;
        long rejectedRows;
        long malformedRows;
        final List<String[]> rows = new ArrayList<String[]>();

        String summaryLine() {
            if (sampledRows == 0L) {
                return "No sample rows are available yet.";
            }
            if (rows.isEmpty()) {
                return "No preview rows passed the current mapping and filters.";
            }
            return "Showing " + rows.size() + " mapped sample rows from " + sampledRows
                    + " inspected samples.";
        }
    }

    private static final class SampleRow {
        final long rowNumber;
        final List<String> cells;

        SampleRow(long rowNumber, List<String> cells) {
            this.rowNumber = rowNumber;
            this.cells = cells;
        }
    }

    private static final class PassState {
        final Map<String, Map<String, Long>> labelsByUser = new LinkedHashMap<String, Map<String, Long>>();
        final Set<String> users = new LinkedHashSet<String>();
        final Set<String> followees = new LinkedHashSet<String>();
    }

    private static final class MappedRow {
        final String followee;
        final long postId;
        final String date;
        final long userId;
        final String userName;
        final String text;
        final boolean accepted;

        MappedRow(String followee, long postId, String date, long userId, String userName, String text, boolean accepted) {
            this.followee = followee;
            this.postId = postId;
            this.date = date;
            this.userId = userId;
            this.userName = userName;
            this.text = text;
            this.accepted = accepted;
        }

        static MappedRow rejected() {
            return new MappedRow("", 0L, "", 0L, "", "", false);
        }
    }

    static SourceProfile inspect(File inputFile, ProgressListener listener) throws IOException {
        return inspect(inputFile, DatasetKind.RETAIL, listener);
    }

    static SourceProfile inspect(File inputFile, DatasetKind selectedKind, ProgressListener listener) throws IOException {
        if (inputFile == null || !inputFile.isFile()) {
            throw new IOException("Choose an existing CSV, TSV, or delimited text file.");
        }
        DatasetKind kind = selectedKind == null ? DatasetKind.RETAIL : selectedKind;
        if (listener != null) {
            listener.onProgress("Reading the source header and detecting delimiter.", 4);
        }

        File scanFile = inputFile;
        String sourceFormat = "Delimited text";
        Delimiter forcedDelimiter = null;
        if (isSpreadsheet(inputFile)) {
            if (listener != null) {
                listener.onProgress("Reading the first worksheet from the Excel file.", 3);
            }
            scanFile = materializeWorkbookAsTsv(inputFile);
            sourceFormat = "Excel workbook";
            forcedDelimiter = Delimiter.TAB;
        }

        List<String> firstLines = readFirstNonEmptyLines(scanFile, 12);
        if (firstLines.isEmpty()) {
            throw new IOException("The selected file is empty.");
        }
        Delimiter delimiter = forcedDelimiter == null ? detectDelimiter(firstLines) : forcedDelimiter;
        List<String> firstRow = parseDelimitedLine(firstLines.get(0), delimiter.value);
        List<String> secondRow = firstLines.size() > 1 ? parseDelimitedLine(firstLines.get(1), delimiter.value) : Collections.<String>emptyList();
        boolean hasHeader = looksLikeHeader(firstRow, secondRow);
        if (kind != DatasetKind.RETAIL && looksLikeDsmpDataRow(firstRow)) {
            hasHeader = false;
        }
        List<String> columnNames = hasHeader ? normalizeColumnNames(firstRow) : generatedColumnNames(firstRow.size());
        List<ColumnProfile> columnProfiles = new ArrayList<ColumnProfile>();
        for (int i = 0; i < columnNames.size(); i++) {
            columnProfiles.add(new ColumnProfile(i, columnNames.get(i)));
        }

        long dataRows = 0L;
        long malformedRows = 0L;
        List<List<String>> previewRows = new ArrayList<List<String>>();
        List<SampleRow> sampleRows = new ArrayList<SampleRow>();
        Random sampleRandom = new Random(391721L);
        long sampleSeen = 0L;

        BufferedReader reader = newReader(scanFile);
        try {
            String line;
            boolean first = true;
            long processed = 0L;
            while ((line = reader.readLine()) != null) {
                processed++;
                if (processed % 50000L == 0L && listener != null) {
                    listener.onProgress("Inspecting rows: " + processed, 8 + (int) Math.min(60L, processed / 8000L));
                }
                if (first) {
                    first = false;
                    line = stripBom(line);
                    if (hasHeader) {
                        continue;
                    }
                }
                if (line.trim().length() == 0) {
                    continue;
                }
                List<String> cells = parseDelimitedLine(line, delimiter.value);
                if (cells.size() != columnNames.size()) {
                    malformedRows++;
                    continue;
                }
                dataRows++;
                for (int i = 0; i < cells.size(); i++) {
                    columnProfiles.get(i).observe(cells.get(i));
                }
                if (previewRows.size() < PREVIEW_LIMIT) {
                    previewRows.add(cells);
                }
                sampleSeen++;
                if (sampleRows.size() < PREVIEW_LIMIT) {
                    sampleRows.add(new SampleRow(processed, cells));
                } else {
                    long slot = Math.floorMod(sampleRandom.nextLong(), sampleSeen);
                    if (slot < PREVIEW_LIMIT) {
                        sampleRows.set((int) slot, new SampleRow(processed, cells));
                    }
                }
            }
        } finally {
            reader.close();
        }

        boolean recognizedRetail = recognizesUciRetail(columnNames);
        boolean recognizedJournal = recognizesJournal(columnNames);
        boolean recognizedTweets = recognizesTweets(columnNames);
        DatasetKind detected = detectedKind(kind, recognizedRetail, recognizedJournal, recognizedTweets);
        if (listener != null) {
            listener.onProgress("Inspection complete.", 100);
        }
        return new SourceProfile(inputFile, scanFile, kind, detected, sourceFormat, delimiter, hasHeader, columnNames, columnProfiles,
                previewRows, sampleRows, dataRows, malformedRows, recognizedRetail, recognizedJournal, recognizedTweets);
    }

    static LivePreviewReport previewOutputRows(SourceProfile source, Options options, int limit) throws IOException {
        validatePreviewOptions(source, options);
        LivePreviewReport preview = new LivePreviewReport();
        ImportReport counter = new ImportReport();
        int rowLimit = Math.max(1, limit);
        for (SampleRow sample : source.sampleRows) {
            preview.sampledRows++;
            if (sample.cells.size() != source.columnNames.size()) {
                preview.malformedRows++;
                continue;
            }
            MappedRow mapped = mapRow(sample.cells, sample.rowNumber, options, counter, true);
            if (!mapped.accepted) {
                preview.rejectedRows++;
                continue;
            }
            preview.acceptedRows++;
            if (preview.rows.size() < rowLimit) {
                preview.rows.add(outputFields(mapped, mapped.followee));
            }
            if (preview.rows.size() >= rowLimit) {
                break;
            }
        }
        return preview;
    }

    static ImportReport importDataset(SourceProfile source, Options options, ProgressListener listener) throws IOException {
        validateOptions(source, options);
        if (listener != null) {
            listener.onProgress("Pass 1 of 2: validating rows and choosing stable labels.", 5);
        }

        ImportReport report = new ImportReport();
        report.outputFile = uniqueOutputFile(options.outputDirectory, options.outputBaseName, ".txt");
        String base = safeFileBase(report.outputFile.getName());
        report.metadataFile = new File(report.outputFile.getParentFile(), base + ".metadata.json");
        report.auditFile = new File(report.outputFile.getParentFile(), base + ".audit.txt");

        PassState state = new PassState();
        readRowsForImport(source, options, report, state, null, true, listener);

        Map<String, String> dominantLabelByUser = dominantLabels(state.labelsByUser);
        report.uniqueUsers = state.users.size();
        report.uniqueFollowees = options.forceSingleFolloweePerUser
                ? new LinkedHashSet<String>(dominantLabelByUser.values()).size()
                : state.followees.size();

        if (report.acceptedRows == 0L) {
            throw new IOException("No usable rows remained after validation. Check the column mapping and filters.");
        }

        if (listener != null) {
            listener.onProgress("Pass 2 of 2: writing DSMP six-column dataset.", 58);
        }

        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(report.outputFile), StandardCharsets.UTF_8));
        try {
            readRowsForImport(source, options, report, state, new RowWriter(
                            writer,
                            dominantLabelByUser,
                            options.forceSingleFolloweePerUser,
                            report),
                    false, listener);
        } finally {
            writer.close();
        }

        report.datasetSha256 = sha256(report.outputFile);
        writeMetadata(source, options, report);
        writeAudit(source, options, report);

        if (listener != null) {
            listener.onProgress("Import complete: " + report.summaryLine(), 100);
        }
        return report;
    }

    private static void readRowsForImport(SourceProfile source, Options options, ImportReport report, PassState state,
            RowWriter rowWriter, boolean countStats, ProgressListener listener) throws IOException {
        BufferedReader reader = newReader(source.dataFile);
        try {
            String line;
            boolean first = true;
            long rowNumber = 0L;
            while ((line = reader.readLine()) != null) {
                rowNumber++;
                if (rowNumber % 50000L == 0L && listener != null) {
                    int offset = countStats ? 5 : 58;
                    int span = countStats ? 48 : 36;
                    listener.onProgress((countStats ? "Validating" : "Writing") + " rows: " + rowNumber,
                            offset + (int) Math.min(span, rowNumber / 15000L));
                }
                if (first) {
                    first = false;
                    line = stripBom(line);
                    if (source.hasHeader) {
                        continue;
                    }
                }
                if (line.trim().length() == 0) {
                    continue;
                }
                List<String> cells = parseDelimitedLine(line, source.delimiter.value);
                if (cells.size() != source.columnNames.size()) {
                    if (countStats) {
                        report.malformedRows++;
                    }
                    continue;
                }
                if (countStats) {
                    report.rawRows++;
                }
                MappedRow mapped = mapRow(cells, rowNumber, options, report, countStats);
                if (!mapped.accepted) {
                    continue;
                }
                if (countStats) {
                    report.acceptedRows++;
                    state.users.add(mapped.userName);
                    String rowLabel = mapped.followee;
                    state.followees.add(rowLabel);
                    Map<String, Long> counts = state.labelsByUser.get(mapped.userName);
                    if (counts == null) {
                        counts = new LinkedHashMap<String, Long>();
                        state.labelsByUser.put(mapped.userName, counts);
                    }
                    Long old = counts.get(rowLabel);
                    counts.put(rowLabel, old == null ? 1L : old.longValue() + 1L);
                } else if (rowWriter != null) {
                    rowWriter.write(mapped);
                }
            }
        } finally {
            reader.close();
        }
    }

    private static MappedRow mapRow(List<String> cells, long rowNumber, Options options, ImportReport report, boolean countStats) {
        String invoice = get(cells, options.invoiceColumn);
        if (options.skipCancelledInvoices && invoice.trim().toUpperCase(Locale.ENGLISH).startsWith("C")) {
            if (countStats) {
                report.skippedCancelledInvoices++;
            }
            return MappedRow.rejected();
        }

        String customer = get(cells, options.customerIdColumn);
        if (options.profile == MappingProfile.CUSTOM && options.customUserIdColumn >= 0) {
            customer = get(cells, options.customUserIdColumn);
        }
        if ((options.profile == MappingProfile.JOURNAL_AUTHORS
                || options.profile == MappingProfile.TWEET_USERS)
                && customer.trim().length() == 0
                && options.customUserNameColumn >= 0) {
            customer = get(cells, options.customUserNameColumn);
        }
        if (options.skipMissingCustomer && customer.trim().length() == 0) {
            if (countStats) {
                report.skippedMissingCustomer++;
            }
            return MappedRow.rejected();
        }

        if (options.quantityColumn >= 0 && options.skipNonPositiveQuantity) {
            Double quantity = parseDouble(get(cells, options.quantityColumn));
            if (quantity == null || quantity.doubleValue() <= 0.0d) {
                if (countStats) {
                    report.skippedNonPositiveQuantity++;
                }
                return MappedRow.rejected();
            }
        }

        if (options.unitPriceColumn >= 0 && options.skipNonPositiveUnitPrice) {
            Double price = parseDouble(get(cells, options.unitPriceColumn));
            if (price == null || price.doubleValue() <= 0.0d) {
                if (countStats) {
                    report.skippedNonPositiveUnitPrice++;
                }
                return MappedRow.rejected();
            }
        }

        String text = options.profile == MappingProfile.CUSTOM
                ? get(cells, options.customTextColumn)
                : get(cells, options.descriptionColumn);
        text = cleanText(text);
        if (tokenCount(text) < Math.max(1, options.minimumTextTokens)) {
            if (countStats) {
                report.skippedMissingText++;
            }
            return MappedRow.rejected();
        }

        String rawDate = options.profile == MappingProfile.CUSTOM
                ? get(cells, options.customDateColumn)
                : get(cells, options.invoiceDateColumn);
        String date = formatDate(rawDate, options.defaultDate);
        if (date.equals(options.defaultDate) && rawDate.trim().length() > 0 && parseFlexibleDate(rawDate) == null && countStats) {
            report.badDateRows++;
        }

        String rawUserId = customer;
        long userId = numericOrHash(rawUserId.length() == 0 ? "row-" + rowNumber : rawUserId);
        String userNameSource;
        if ((options.profile == MappingProfile.CUSTOM
                || options.profile == MappingProfile.JOURNAL_AUTHORS
                || options.profile == MappingProfile.TWEET_USERS)
                && options.customUserNameColumn >= 0) {
            userNameSource = get(cells, options.customUserNameColumn);
        } else {
            userNameSource = "Customer_" + (customer.trim().length() == 0 ? Long.toString(userId) : customer.trim());
        }
        String userName = safeEntityName(userNameSource, "User_" + userId);

        long postId;
        if (options.profile == MappingProfile.CUSTOM && options.customPostIdColumn >= 0) {
            postId = numericOrHash(get(cells, options.customPostIdColumn) + "-" + rowNumber);
        } else if (options.invoiceColumn >= 0) {
            postId = numericOrHash(invoice + "-" + get(cells, options.stockCodeColumn) + "-" + rowNumber);
        } else {
            postId = rowNumber;
        }

        String followee = deriveFollowee(cells, options);
        followee = safeEntityName(followee, defaultCollectionLabel(options.datasetKind));
        return new MappedRow(followee, postId, date, userId, userName, text, true);
    }

    private static String deriveFollowee(List<String> cells, Options options) {
        if (options.profile == MappingProfile.SIMILAR_CUSTOMERS) {
            return defaultCollectionLabel(options.datasetKind);
        }
        if (options.profile == MappingProfile.PRODUCT_FAMILY) {
            return inferProductFamily(get(cells, options.descriptionColumn), get(cells, options.stockCodeColumn));
        }
        if (options.profile == MappingProfile.SMART_PRODUCT_CATEGORY) {
            return inferSmartProductCategory(get(cells, options.descriptionColumn), get(cells, options.stockCodeColumn));
        }
        if (options.profile == MappingProfile.COUNTRY_BASELINE) {
            String country = get(cells, options.countryColumn);
            return country.trim().length() == 0 ? "UnknownCountry" : country;
        }
        if (options.profile == MappingProfile.STOCK_CODE_EXPERIMENTAL) {
            String stock = get(cells, options.stockCodeColumn);
            return options.groupStockCodesByPrefix ? stockCodeGroup(stock) : stock;
        }
        if (options.profile == MappingProfile.JOURNAL_AUTHORS
                || options.profile == MappingProfile.TWEET_USERS) {
            String reference = get(cells, options.countryColumn);
            return reference.trim().length() == 0 ? options.profile.toString() : reference;
        }
        return get(cells, options.customReferenceColumn);
    }

    private static String defaultCollectionLabel(DatasetKind kind) {
        if (kind == DatasetKind.JOURNAL) {
            return "JournalCollection";
        }
        if (kind == DatasetKind.TWEETS) {
            return "TweetCollection";
        }
        if (kind == DatasetKind.OTHER) {
            return "DocumentCollection";
        }
        return DEFAULT_COLLECTION_FOLLOWEE;
    }

    private static Map<String, String> dominantLabels(Map<String, Map<String, Long>> labelsByUser) {
        Map<String, String> result = new LinkedHashMap<String, String>();
        for (Map.Entry<String, Map<String, Long>> userEntry : labelsByUser.entrySet()) {
            String winner = DEFAULT_COLLECTION_FOLLOWEE;
            long winnerCount = Long.MIN_VALUE;
            for (Map.Entry<String, Long> labelEntry : userEntry.getValue().entrySet()) {
                long count = labelEntry.getValue().longValue();
                if (count > winnerCount || (count == winnerCount && labelEntry.getKey().compareTo(winner) < 0)) {
                    winner = labelEntry.getKey();
                    winnerCount = count;
                }
            }
            result.put(userEntry.getKey(), winner);
        }
        return result;
    }

    private static final class RowWriter {
        private final BufferedWriter writer;
        private final Map<String, String> dominantLabelByUser;
        private final boolean forceSingleFolloweePerUser;
        private final ImportReport report;

        RowWriter(BufferedWriter writer, Map<String, String> dominantLabelByUser,
                boolean forceSingleFolloweePerUser, ImportReport report) {
            this.writer = writer;
            this.dominantLabelByUser = dominantLabelByUser;
            this.forceSingleFolloweePerUser = forceSingleFolloweePerUser;
            this.report = report;
        }

        void write(MappedRow row) throws IOException {
            String followee = forceSingleFolloweePerUser && dominantLabelByUser.containsKey(row.userName)
                    ? dominantLabelByUser.get(row.userName)
                    : row.followee;
            Long oldCount = report.finalFolloweeCounts.get(followee);
            report.finalFolloweeCounts.put(followee, oldCount == null ? 1L : oldCount.longValue() + 1L);
            String[] fields = outputFields(row, followee);
            for (int i = 0; i < fields.length; i++) {
                if (i > 0) {
                    writer.write('\t');
                }
                writer.write(sanitizeTsv(fields[i]));
            }
            writer.newLine();
            report.writtenRows++;
            if (report.outputPreview.size() < 20) {
                report.outputPreview.add(fields);
            }
        }
    }

    private static String[] outputFields(MappedRow row, String followee) {
        return new String[] {
                followee,
                Long.toString(row.postId),
                row.date,
                Long.toString(row.userId),
                row.userName,
                row.text
        };
    }

    private static void validateOptions(SourceProfile source, Options options) throws IOException {
        if (source == null) {
            throw new IOException("Inspect the source dataset first.");
        }
        if (options == null) {
            throw new IOException("Importer options are missing.");
        }
        if (options.outputDirectory == null) {
            throw new IOException("Choose an output directory.");
        }
        if (!options.outputDirectory.exists() && !options.outputDirectory.mkdirs()) {
            throw new IOException("Could not create output directory: " + options.outputDirectory.getAbsolutePath());
        }
        if (!options.outputDirectory.isDirectory()) {
            throw new IOException("Output location is not a directory: " + options.outputDirectory.getAbsolutePath());
        }
        if (options.outputBaseName == null || options.outputBaseName.trim().length() == 0) {
            throw new IOException("Provide an output dataset name.");
        }
        validatePreviewOptions(source, options);
    }

    private static void validatePreviewOptions(SourceProfile source, Options options) throws IOException {
        if (source == null) {
            throw new IOException("Inspect the source dataset first.");
        }
        if (options == null) {
            throw new IOException("Importer options are missing.");
        }
        requireColumn(source, options.profile == MappingProfile.CUSTOM ? options.customTextColumn : options.descriptionColumn,
                "text/description");
        if (options.profile == MappingProfile.PRODUCT_FAMILY || options.profile == MappingProfile.STOCK_CODE_EXPERIMENTAL) {
            requireColumn(source, options.stockCodeColumn, "stock/product code");
        }
        if (options.profile == MappingProfile.COUNTRY_BASELINE) {
            requireColumn(source, options.countryColumn, "country");
        }
        if (options.profile == MappingProfile.JOURNAL_AUTHORS) {
            requireColumn(source, options.countryColumn, "journal/venue followee");
            requireColumn(source, options.customerIdColumn, "author/user id");
        }
        if (options.profile == MappingProfile.TWEET_USERS) {
            requireColumn(source, options.countryColumn, "tweet reference/source account");
            requireColumn(source, options.customerIdColumn, "tweet user id");
        }
        if (options.profile == MappingProfile.CUSTOM) {
            requireColumn(source, options.customReferenceColumn, "custom followee/reference");
            requireColumn(source, options.customUserIdColumn, "custom numeric user or customer id");
            requireColumn(source, options.customTextColumn, "custom text");
        } else {
            requireColumn(source, options.customerIdColumn, "customer id");
        }
    }

    private static void requireColumn(SourceProfile source, int column, String purpose) throws IOException {
        if (column < 0 || column >= source.columnNames.size()) {
            throw new IOException("Select a valid column for " + purpose + ".");
        }
    }

    private static List<String> readFirstNonEmptyLines(File file, int limit) throws IOException {
        List<String> lines = new ArrayList<String>();
        BufferedReader reader = newReader(file);
        try {
            String line;
            while ((line = reader.readLine()) != null && lines.size() < limit) {
                line = stripBom(line);
                if (line.trim().length() > 0) {
                    lines.add(line);
                }
            }
        } finally {
            reader.close();
        }
        return lines;
    }

    private static BufferedReader newReader(File file) throws IOException {
        return new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8), 65536);
    }

    private static boolean isSpreadsheet(File file) {
        String name = file == null ? "" : file.getName().toLowerCase(Locale.ENGLISH);
        return name.endsWith(".xlsx") || name.endsWith(".xls");
    }

    private static File materializeWorkbookAsTsv(File workbookFile) throws IOException {
        String lowerName = workbookFile.getName().toLowerCase(Locale.ENGLISH);
        if (lowerName.endsWith(".xls") && !lowerName.endsWith(".xlsx")) {
            throw new IOException("Legacy .xls files are not supported by the safe importer. Save the workbook as .xlsx or CSV.");
        }
        File temp = File.createTempFile("dsmp-uci-retail-workbook-", ".tsv");
        temp.deleteOnExit();
        ZipFile zip = new ZipFile(workbookFile);
        try {
            List<String> sharedStrings = readSharedStrings(zip);
            ZipEntry sheetEntry = firstWorksheet(zip);
            if (sheetEntry == null) {
                throw new IOException("The workbook does not contain a worksheet.");
            }
            BufferedWriter writer = new BufferedWriter(
                    new OutputStreamWriter(new FileOutputStream(temp), StandardCharsets.UTF_8));
            try {
                streamWorksheetToTsv(zip, sheetEntry, sharedStrings, writer);
            } finally {
                writer.close();
            }
            return temp;
        } catch (IOException ex) {
            throw ex;
        } catch (Exception ex) {
            throw new IOException("Could not read Excel workbook. Export it to CSV if the workbook is protected or malformed.", ex);
        } finally {
            zip.close();
        }
    }

    private static ZipEntry firstWorksheet(ZipFile zip) {
        ZipEntry preferred = zip.getEntry("xl/worksheets/sheet1.xml");
        if (preferred != null) {
            return preferred;
        }
        TreeSet<String> sheetNames = new TreeSet<String>();
        java.util.Enumeration<? extends ZipEntry> entries = zip.entries();
        while (entries.hasMoreElements()) {
            ZipEntry entry = entries.nextElement();
            String name = entry.getName();
            if (name.startsWith("xl/worksheets/sheet") && name.endsWith(".xml")) {
                sheetNames.add(name);
            }
        }
        return sheetNames.isEmpty() ? null : zip.getEntry(sheetNames.first());
    }

    private static List<String> readSharedStrings(ZipFile zip) throws Exception {
        ZipEntry entry = zip.getEntry("xl/sharedStrings.xml");
        if (entry == null) {
            return Collections.emptyList();
        }
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(false);
        disableXmlExternalEntities(factory);
        DocumentBuilder builder = factory.newDocumentBuilder();
        InputStream input = zip.getInputStream(entry);
        Document document;
        try {
            document = builder.parse(input);
        } finally {
            input.close();
        }
        NodeList items = document.getElementsByTagName("si");
        List<String> strings = new ArrayList<String>(items.getLength());
        for (int i = 0; i < items.getLength(); i++) {
            Node item = items.item(i);
            NodeList textNodes = ((org.w3c.dom.Element) item).getElementsByTagName("t");
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < textNodes.getLength(); j++) {
                sb.append(textNodes.item(j).getTextContent());
            }
            strings.add(sb.toString());
        }
        return strings;
    }

    private static void disableXmlExternalEntities(DocumentBuilderFactory factory) {
        trySetFeature(factory, "http://apache.org/xml/features/disallow-doctype-decl", true);
        trySetFeature(factory, "http://xml.org/sax/features/external-general-entities", false);
        trySetFeature(factory, "http://xml.org/sax/features/external-parameter-entities", false);
        factory.setExpandEntityReferences(false);
    }

    private static void trySetFeature(DocumentBuilderFactory factory, String feature, boolean value) {
        try {
            factory.setFeature(feature, value);
        } catch (Exception ignored) {
            // Some JDK XML implementations do not expose every hardening flag.
        }
    }

    private static void streamWorksheetToTsv(ZipFile zip, ZipEntry sheetEntry, List<String> sharedStrings,
            BufferedWriter writer) throws Exception {
        XMLInputFactory factory = XMLInputFactory.newInstance();
        try {
            factory.setProperty(XMLInputFactory.SUPPORT_DTD, Boolean.FALSE);
            factory.setProperty("javax.xml.stream.isSupportingExternalEntities", Boolean.FALSE);
        } catch (IllegalArgumentException ignored) {
            // Continue with the platform defaults when a property is unavailable.
        }
        InputStream input = zip.getInputStream(sheetEntry);
        XMLStreamReader xml = factory.createXMLStreamReader(input);
        Map<Integer, String> rowValues = new LinkedHashMap<Integer, String>();
        int outputColumns = -1;
        int currentColumn = -1;
        String currentType = "";
        StringBuilder currentValue = new StringBuilder();
        boolean inValue = false;
        boolean inInlineText = false;
        try {
            while (xml.hasNext()) {
                int event = xml.next();
                if (event == XMLStreamConstants.START_ELEMENT) {
                    String local = xml.getLocalName();
                    if ("row".equals(local)) {
                        rowValues.clear();
                    } else if ("c".equals(local)) {
                        currentColumn = columnIndexFromCellRef(xml.getAttributeValue(null, "r"));
                        currentType = xml.getAttributeValue(null, "t");
                        currentValue.setLength(0);
                    } else if ("v".equals(local)) {
                        inValue = true;
                        currentValue.setLength(0);
                    } else if ("t".equals(local) && "inlineStr".equals(currentType)) {
                        inInlineText = true;
                        currentValue.setLength(0);
                    }
                } else if (event == XMLStreamConstants.CHARACTERS || event == XMLStreamConstants.CDATA) {
                    if (inValue || inInlineText) {
                        currentValue.append(xml.getText());
                    }
                } else if (event == XMLStreamConstants.END_ELEMENT) {
                    String local = xml.getLocalName();
                    if ("v".equals(local)) {
                        inValue = false;
                    } else if ("t".equals(local) && inInlineText) {
                        inInlineText = false;
                    } else if ("c".equals(local)) {
                        rowValues.put(Integer.valueOf(currentColumn),
                                resolveCellValue(currentType, currentValue.toString(), sharedStrings));
                        currentColumn = -1;
                        currentType = "";
                        currentValue.setLength(0);
                    } else if ("row".equals(local)) {
                        int rowMax = -1;
                        for (Integer col : rowValues.keySet()) {
                            if (col.intValue() > rowMax) {
                                rowMax = col.intValue();
                            }
                        }
                        if (outputColumns < 0) {
                            outputColumns = rowMax + 1;
                        }
                        int columnsToWrite = Math.max(outputColumns, rowMax + 1);
                        for (int col = 0; col < columnsToWrite; col++) {
                            if (col > 0) {
                                writer.write('\t');
                            }
                            String value = rowValues.get(Integer.valueOf(col));
                            writer.write(sanitizeTsv(value == null ? "" : value));
                        }
                        writer.newLine();
                    }
                }
            }
        } finally {
            xml.close();
            input.close();
        }
    }

    private static String resolveCellValue(String type, String raw, List<String> sharedStrings) {
        if ("s".equals(type)) {
            try {
                int index = Integer.parseInt(raw.trim());
                if (index >= 0 && index < sharedStrings.size()) {
                    return sharedStrings.get(index);
                }
            } catch (NumberFormatException ignored) {
                return raw;
            }
        }
        return raw == null ? "" : raw;
    }

    private static int columnIndexFromCellRef(String reference) {
        if (reference == null || reference.length() == 0) {
            return 0;
        }
        int col = 0;
        for (int i = 0; i < reference.length(); i++) {
            char c = reference.charAt(i);
            if (c >= 'A' && c <= 'Z') {
                col = col * 26 + (c - 'A' + 1);
            } else if (c >= 'a' && c <= 'z') {
                col = col * 26 + (c - 'a' + 1);
            } else {
                break;
            }
        }
        return Math.max(0, col - 1);
    }

    private static Delimiter detectDelimiter(List<String> lines) {
        Delimiter best = Delimiter.COMMA;
        int bestScore = -1;
        for (Delimiter delimiter : Delimiter.values()) {
            int score = 0;
            int expected = -1;
            for (String line : lines) {
                List<String> cells = parseDelimitedLine(line, delimiter.value);
                if (cells.size() > 1) {
                    score += cells.size();
                    if (expected < 0) {
                        expected = cells.size();
                    } else if (expected == cells.size()) {
                        score += 4;
                    }
                }
            }
            if (score > bestScore) {
                best = delimiter;
                bestScore = score;
            }
        }
        return best;
    }

    static List<String> parseDelimitedLine(String line, char delimiter) {
        List<String> cells = new ArrayList<String>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == delimiter && !inQuotes) {
                cells.add(current.toString());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        cells.add(current.toString());
        return cells;
    }

    private static boolean looksLikeHeader(List<String> firstRow, List<String> secondRow) {
        if (recognizesUciRetail(firstRow) || recognizesJournal(firstRow) || recognizesTweets(firstRow)) {
            return true;
        }
        int headerish = 0;
        for (String cell : firstRow) {
            String trimmed = cell.trim();
            if (trimmed.length() > 0 && parseDouble(trimmed) == null && parseFlexibleDate(trimmed) == null) {
                headerish++;
            }
        }
        int dataish = 0;
        for (String cell : secondRow) {
            String trimmed = cell.trim();
            if (trimmed.length() == 0 || parseDouble(trimmed) != null || parseFlexibleDate(trimmed) != null) {
                dataish++;
            }
        }
        return firstRow.size() > 1 && headerish >= Math.max(2, firstRow.size() / 2) && dataish >= Math.max(1, secondRow.size() / 3);
    }

    private static boolean looksLikeDsmpDataRow(List<String> row) {
        if (row == null || row.size() < 6) {
            return false;
        }
        String reference = get(row, 0).trim();
        String date = get(row, 2).trim();
        String userIdentity = get(row, 3).trim();
        String userName = get(row, 4).trim();
        String text = get(row, 5).trim();
        if (reference.length() == 0 || userIdentity.length() == 0
                || userName.length() == 0 || tokenCount(text) == 0) {
            return false;
        }
        return parseFlexibleDate(date) != null || date.matches("\\d{4}");
    }

    private static boolean recognizesUciRetail(List<String> names) {
        List<String> normalized = new ArrayList<String>();
        for (String name : names) {
            normalized.add(normalizeHeader(name));
        }
        return normalized.contains("invoiceno")
                && normalized.contains("stockcode")
                && normalized.contains("description")
                && normalized.contains("quantity")
                && normalized.contains("invoicedate")
                && normalized.contains("unitprice")
                && normalized.contains("customerid")
                && normalized.contains("country");
    }

    private static boolean recognizesJournal(List<String> names) {
        Set<String> normalized = normalizedHeaderSet(names);
        return containsAnyHeader(normalized, "journal", "journalname", "venue", "publication", "conference")
                && containsAnyHeader(normalized, "authorid", "author", "authorname", "sequenceid", "userid")
                && containsAnyHeader(normalized, "title", "papertitle", "articletitle", "documenttitle", "abstract", "text");
    }

    private static boolean recognizesTweets(List<String> names) {
        Set<String> normalized = normalizedHeaderSet(names);
        return containsAnyHeader(normalized, "tweetid", "statusid", "id", "idstr", "postid")
                && containsAnyHeader(normalized, "userid", "username", "screenname", "authorid", "author")
                && containsAnyHeader(normalized, "text", "tweet", "tweettext", "fulltext", "body");
    }

    private static Set<String> normalizedHeaderSet(List<String> names) {
        Set<String> normalized = new LinkedHashSet<String>();
        for (String name : names) {
            normalized.add(normalizeHeader(name));
        }
        return normalized;
    }

    private static boolean containsAnyHeader(Set<String> headers, String... candidates) {
        for (String candidate : candidates) {
            if (headers.contains(normalizeHeader(candidate))) {
                return true;
            }
        }
        return false;
    }

    private static DatasetKind detectedKind(DatasetKind selected, boolean retail, boolean journal, boolean tweets) {
        if (retail) {
            return DatasetKind.RETAIL;
        }
        if (journal) {
            return DatasetKind.JOURNAL;
        }
        if (tweets) {
            return DatasetKind.TWEETS;
        }
        return selected == null ? DatasetKind.OTHER : selected;
    }

    private static List<String> normalizeColumnNames(List<String> names) {
        List<String> result = new ArrayList<String>();
        Set<String> seen = new LinkedHashSet<String>();
        for (int i = 0; i < names.size(); i++) {
            String name = names.get(i) == null ? "" : names.get(i).trim();
            if (name.length() == 0) {
                name = "Column " + (i + 1);
            }
            String unique = name;
            int suffix = 2;
            while (seen.contains(unique)) {
                unique = name + " " + suffix++;
            }
            seen.add(unique);
            result.add(unique);
        }
        return result;
    }

    private static List<String> generatedColumnNames(int count) {
        List<String> names = new ArrayList<String>();
        for (int i = 0; i < count; i++) {
            names.add("Column " + (i + 1));
        }
        return names;
    }

    private static String normalizeHeader(String value) {
        String raw = value == null ? "" : value.toLowerCase(Locale.ENGLISH);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static String get(List<String> cells, int index) {
        if (index < 0 || index >= cells.size()) {
            return "";
        }
        return cells.get(index) == null ? "" : cells.get(index);
    }

    private static String cleanText(String text) {
        String raw = text == null ? "" : text;
        raw = raw.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ');
        raw = raw.replaceAll("\\s+", " ").trim();
        return raw;
    }

    private static int tokenCount(String text) {
        if (text == null || text.trim().length() == 0) {
            return 0;
        }
        return text.trim().split("\\s+").length;
    }

    private static String formatDate(String value, String fallback) {
        Date parsed = parseFlexibleDate(value);
        if (parsed == null) {
            return fallback == null || fallback.trim().length() == 0 ? "2011-01-01" : fallback.trim();
        }
        return new SimpleDateFormat("yyyy-MM-dd", Locale.ENGLISH).format(parsed);
    }

    private static Date parseFlexibleDate(String value) {
        String raw = value == null ? "" : value.trim();
        if (raw.length() == 0) {
            return null;
        }
        if (raw.matches("\\d{4}")) {
            try {
                SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd", Locale.ENGLISH);
                sdf.setLenient(false);
                return sdf.parse(raw + "-01-01");
            } catch (ParseException ignored) {
                return null;
            }
        }
        Double numeric = parseDouble(raw);
        if (numeric != null && numeric.doubleValue() > 20000.0d && numeric.doubleValue() < 80000.0d) {
            long millis = Math.round((numeric.doubleValue() - 25569.0d) * 86400000.0d);
            return new Date(millis);
        }
        List<String> patterns = Arrays.asList(
                "M/d/yyyy H:mm",
                "M/d/yyyy HH:mm",
                "MM/dd/yyyy HH:mm",
                "MM/dd/yyyy H:mm",
                "M/d/yy H:mm",
                "yyyy-MM-dd",
                "yyyy-MM-dd HH:mm:ss",
                "yyyy/MM/dd",
                "dd/MM/yyyy HH:mm",
                "dd/MM/yyyy",
                "MMM d yyyy",
                "MMMM d yyyy");
        for (String pattern : patterns) {
            try {
                SimpleDateFormat sdf = new SimpleDateFormat(pattern, Locale.ENGLISH);
                sdf.setLenient(false);
                return sdf.parse(raw);
            } catch (ParseException ignored) {
                // Try the next common export format.
            }
        }
        return null;
    }

    private static Double parseDouble(String value) {
        String raw = value == null ? "" : value.trim();
        if (raw.length() == 0) {
            return null;
        }
        raw = raw.replace(",", "");
        try {
            return Double.valueOf(raw);
        } catch (NumberFormatException ex) {
            return null;
        }
    }

    private static long numericOrHash(String value) {
        String raw = value == null ? "" : value.trim();
        if (raw.endsWith(".0")) {
            raw = raw.substring(0, raw.length() - 2);
        }
        try {
            if (raw.length() > 0) {
                long parsed = Long.parseLong(raw);
                if (parsed > 0L) {
                    return parsed;
                }
            }
        } catch (NumberFormatException ignored) {
            // Alphanumeric ids become stable positive numeric ids below.
        }
        return stablePositiveHash(raw.length() == 0 ? "blank" : raw);
    }

    private static long stablePositiveHash(String value) {
        long hash = 1125899906842597L;
        String raw = value == null ? "" : value;
        for (int i = 0; i < raw.length(); i++) {
            hash = 31L * hash + raw.charAt(i);
        }
        hash = hash & Long.MAX_VALUE;
        return hash == 0L ? 1L : hash;
    }

    private static String safeEntityName(String value, String fallback) {
        String raw = value == null ? "" : value.trim();
        StringBuilder sb = new StringBuilder();
        boolean capitalizeNext = false;
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                if (sb.length() == 0) {
                    sb.append(Character.isLetter(c) ? Character.toUpperCase(c) : c);
                } else if (capitalizeNext && Character.isLetter(c)) {
                    sb.append(Character.toUpperCase(c));
                } else {
                    sb.append(c);
                }
                capitalizeNext = false;
            } else {
                capitalizeNext = true;
            }
            if (sb.length() >= 80) {
                break;
            }
        }
        if (sb.length() == 0) {
            return fallback;
        }
        return sb.toString();
    }

    private static String sanitizeTsv(String value) {
        return cleanText(value).replace('\t', ' ');
    }

    private static final ProductCategoryRule[] SMART_PRODUCT_CATEGORY_RULES = new ProductCategoryRule[] {
            new ProductCategoryRule("ServiceAdjustment", 120,
                    "postage", "manual", "bank charges", "amazon fee", "dotcom postage",
                    "discount", "samples", "adjustment", "commission", "next day carriage"),
            new ProductCategoryRule("SeasonalHoliday", 100,
                    "christmas", "xmas", "advent", "easter", "valentine", "halloween",
                    "santa", "reindeer", "bauble", "stocking", "snowflake", "snowmen",
                    "snowy", "gingerbread", "nativity", "jingle bell", "noel", "festive",
                    "christmas tree", "tree decoration", "tree top", "yuletide", "crackers",
                    "sleigh"),
            new ProductCategoryRule("BagsAccessories", 88,
                    "lunch bag", "jumbo bag", "charlotte bag", "shopping bag", "cotton tote",
                    "tote bag", "shopper", "purse", "wallet", "handbag", "backpack",
                    "rucksack", "vanity bag", "make up bag", "wash bag", "peg bag",
                    "passport cover", "umbrella", "bag"),
            new ProductCategoryRule("KidsToysGames", 84,
                    "toy", "doll", "dolly girl", "spaceboy", "game", "games", "jigsaw",
                    "children", "childrens", "baby", "teddy", "robot", "dinosaur", "skull",
                    "feltcraft", "plasters", "dominoes", "skittles", "puppet", "harmonica",
                    "magic tree", "paint set", "playtime", "circus parade", "woodland animals",
                    "alphabet blocks", "spinning top", "spinning tops", "catch cup",
                    "pick sticks", "balloon art", "race car", "boys", "girls", "glider",
                    "gliders", "skipping rope", "colouring set", "coloring set",
                    "school colouring", "school coloring", "rounders", "water pistol",
                    "modelling clay", "playhouse", "crayons", "chalk sticks", "bingo",
                    "snakes ladders", "naughts crosses", "piggy bank"),
            new ProductCategoryRule("CandlesLighting", 82,
                    "t-light", "tealight", "tea light", "candle", "candles", "candleholder",
                    "lantern", "night light", "fairy lights", "lamp", "lampshade",
                    "chandelier", "light holder", "candle holder", "lights"),
            new ProductCategoryRule("KitchenDining", 78,
                    "cake stand", "cakestand", "cake", "baking", "cupcake", "recipe",
                    "kitchen", "pantry", "jam", "spice", "mould", "moulds", "mold",
                    "molds", "scales", "oven", "apron", "tea towel", "tea towels", "cookie",
                    "bread bin", "chopping board", "frying pan", "pan", "skillet", "cutters",
                    "egg fry", "jelly mould", "jelly moulds", "cake tin", "cake tins", "bake",
                    "biscuit tin", "food container", "food cover", "lolly makers",
                    "measuring jug", "milk jug", "sugar dispenser", "bottle opener",
                    "washing up gloves"),
            new ProductCategoryRule("TablewareDrinkware", 76,
                    "teacup", "saucer", "mug", "cup", "beaker", "glass", "bowl", "plate",
                    "jug", "cutlery", "spoon", "fork", "teapot", "coaster", "tray",
                    "egg cup", "dinner", "tea set", "placemat", "placemats", "napkin",
                    "napkins", "snack box", "snack boxes", "lunch box", "dish", "serving",
                    "popcorn holder"),
            new ProductCategoryRule("PaperStationery", 72,
                    "card", "cards", "paper", "wrap", "tag", "tags", "notebook",
                    "notebooks", "pencil", "pencils", "pen", "pens", "sticker", "stickers",
                    "diary", "calendar", "book", "bookmark", "ruler", "tape", "paperweight",
                    "magnet", "magnets", "shopping list", "memo", "notepad", "chalkboard",
                    "notecard", "envelope", "exercise book", "stationery", "stamp set",
                    "memoboard", "memo board"),
            new ProductCategoryRule("CraftSewing", 68,
                    "sewing", "craft", "feltcraft", "paint", "ribbon", "ribbons", "lace",
                    "twine", "knitting", "crochet", "button", "needle", "string", "bead",
                    "doily", "doilies", "kit", "card kit", "wool", "thread", "cotton reel",
                    "colouring", "coloring"),
            new ProductCategoryRule("HomeDecor", 64,
                    "heart", "decoration", "ornament", "sign", "doormat", "frame", "mirror",
                    "clock", "wall", "wicker", "home", "word", "block", "slate", "drawer",
                    "knob", "hook", "cabinet", "shelf", "plaque", "garland", "mobile",
                    "bunting", "doorstop", "photo frame", "picture frame", "black board",
                    "noticeboard", "key cabinet", "letter rack", "candlestick", "metal pears",
                    "mini chest", "pear", "pears", "apple decoration", "decorative",
                    "hanging owls", "coat hanger", "fan", "mat"),
            new ProductCategoryRule("StorageOrganization", 58,
                    "storage", "box", "boxes", "basket", "trunk", "cabinet", "drawer",
                    "rack", "organizer", "organiser", "holder", "tin", "tins", "case",
                    "cases", "jar", "jars", "container", "canister", "bin", "bucket",
                    "caddy", "pegs", "clothes pegs", "coat rack", "crate"),
            new ProductCategoryRule("GardenOutdoor", 54,
                    "garden", "gardeners", "flower", "plant", "watering", "birdhouse",
                    "bird bath", "kneeling pad", "herb marker", "greenhouse", "outdoor",
                    "parasol", "picnic", "bird feeder", "watering can"),
            new ProductCategoryRule("JewelryFashion", 50,
                    "jewellery", "jewelry", "necklace", "bracelet", "earring", "earrings",
                    "brooch", "bangle", "hairclip", "hair clip", "tiara", "scarf",
                    "pendant", "charm", "bead", "ring"),
            new ProductCategoryRule("BathPersonalCare", 46,
                    "bath", "soap", "lip gloss", "cosmetic", "perfume", "toothbrush",
                    "washbag", "tissues", "first aid", "hand warmer", "hot water bottle",
                    "flannel", "compact mirror", "mirror compact"),
            new ProductCategoryRule("TextilesComfort", 42,
                    "cushion", "pillow", "blanket", "throw", "quilt", "cover", "cotton",
                    "linen", "tea towel", "towel", "cloth", "napkin", "apron",
                    "hot water bottle", "hand warmer"),
            new ProductCategoryRule("ToolsHardware", 38,
                    "tool", "tools", "hammer", "screwdriver", "repair kit", "bicycle",
                    "magazine rack", "key fob", "key ring", "hook", "hooks", "door knob",
                    "drawer knob", "scissors", "tape measure", "measuring tape")
    };

    private static final class ProductCategoryRule {
        final String category;
        final int baseScore;
        final String[] terms;

        ProductCategoryRule(String category, int baseScore, String... terms) {
            this.category = category;
            this.baseScore = baseScore;
            this.terms = new String[terms.length];
            for (int i = 0; i < terms.length; i++) {
                this.terms[i] = normalizeCategoryCore(terms[i]);
            }
        }
    }

    private static String inferSmartProductCategory(String description, String stockCode) {
        String normalized = normalizeCategoryText(description);
        String bare = normalized.trim();
        if (bare.length() == 0) {
            return "GeneralRetail";
        }
        if ("carriage".equals(bare)) {
            return "ServiceAdjustment";
        }

        String bestCategory = "GeneralRetail";
        int bestScore = 0;
        for (ProductCategoryRule rule : SMART_PRODUCT_CATEGORY_RULES) {
            int score = 0;
            for (String term : rule.terms) {
                if (term.length() > 0 && normalized.indexOf(" " + term + " ") >= 0) {
                    score += rule.baseScore + (termWordCount(term) * 4);
                }
            }
            if (score > bestScore) {
                bestScore = score;
                bestCategory = rule.category;
            }
        }
        return bestCategory;
    }

    private static String normalizeCategoryText(String value) {
        String core = normalizeCategoryCore(value);
        return core.length() == 0 ? "" : " " + core + " ";
    }

    private static String normalizeCategoryCore(String value) {
        String raw = value == null ? "" : value.toLowerCase(Locale.ENGLISH);
        StringBuilder sb = new StringBuilder();
        boolean previousSpace = true;
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                sb.append(c);
                previousSpace = false;
            } else if (!previousSpace) {
                sb.append(' ');
                previousSpace = true;
            }
        }
        int length = sb.length();
        if (length > 0 && sb.charAt(length - 1) == ' ') {
            sb.setLength(length - 1);
        }
        return sb.toString();
    }

    private static int termWordCount(String term) {
        if (term == null || term.length() == 0) {
            return 0;
        }
        int count = 1;
        for (int i = 0; i < term.length(); i++) {
            if (term.charAt(i) == ' ') {
                count++;
            }
        }
        return count;
    }

    private static String inferProductFamily(String description, String stockCode) {
        String d = description == null ? "" : description.toLowerCase(Locale.ENGLISH);
        if (containsAny(d, "postage", "manual", "amazon fee", "bank charges")) {
            return "ServiceAdjustment";
        }
        if (containsAny(d, "christmas", "xmas", "easter", "valentine", "halloween")) {
            return "Seasonal";
        }
        if (containsAny(d, "bag", "lunch bag", "jumbo bag", "purse", "wallet")) {
            return "BagsAccessories";
        }
        if (containsAny(d, "card", "paper", "wrap", "tag", "notebook", "pencil", "pen ", "sticker")) {
            return "PaperStationery";
        }
        if (containsAny(d, "candle", "light", "lantern", "t-light", "holder", "chandelier")) {
            return "CandlesLighting";
        }
        if (containsAny(d, "mug", "cup", "bowl", "plate", "glass", "cutlery", "spoon", "fork", "teapot")) {
            return "Tableware";
        }
        if (containsAny(d, "cake", "baking", "cupcake", "tea time", "teatime", "recipe", "kitchen")) {
            return "KitchenDining";
        }
        if (containsAny(d, "toy", "doll", "game", "jigsaw", "children", "baby", "skull")) {
            return "ChildrenToys";
        }
        if (containsAny(d, "jewellery", "jewelry", "necklace", "bracelet", "ring", "earring")) {
            return "JewelryAccessories";
        }
        if (containsAny(d, "garden", "flower", "plant", "watering", "bird")) {
            return "GardenOutdoor";
        }
        if (containsAny(d, "heart", "decoration", "ornament", "sign", "doormat", "frame", "mirror", "clock", "vintage", "retrospot")) {
            return "HomeDecor";
        }
        String group = stockCodeGroup(stockCode);
        return group.length() == 0 ? "GeneralRetail" : "ProductFamily" + group;
    }

    private static boolean containsAny(String value, String... needles) {
        for (String needle : needles) {
            if (value.contains(needle)) {
                return true;
            }
        }
        return false;
    }

    private static String stockCodeGroup(String stockCode) {
        String raw = stockCode == null ? "" : stockCode.trim().toUpperCase(Locale.ENGLISH);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                sb.append(c);
            }
            if (sb.length() >= 4) {
                break;
            }
        }
        return sb.length() == 0 ? "UnknownStock" : "Stock" + sb.toString();
    }

    private static File defaultOutputDirectory(File inputFile) {
        File parent = inputFile == null ? null : inputFile.getParentFile();
        if (parent == null) {
            return new File("Dataset");
        }
        return new File(parent, "DSMP_Imported");
    }

    private static String safeFileBase(String name) {
        String raw = name == null ? "ImportedDataset" : name;
        int dot = raw.lastIndexOf('.');
        if (dot > 0) {
            raw = raw.substring(0, dot);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '-' || c == '_') {
                sb.append(c);
            } else if (sb.length() == 0 || sb.charAt(sb.length() - 1) != '_') {
                sb.append('_');
            }
        }
        return sb.length() == 0 ? "ImportedDataset" : sb.toString();
    }

    private static File uniqueOutputFile(File dir, String baseName, String extension) {
        String cleanBase = safeFileBase(baseName);
        File file = new File(dir, cleanBase + extension);
        int suffix = 2;
        while (file.exists()) {
            file = new File(dir, cleanBase + "_" + suffix + extension);
            suffix++;
        }
        return file;
    }

    private static String sha256(File file) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            FileInputStream in = new FileInputStream(file);
            try {
                byte[] buffer = new byte[65536];
                int read;
                while ((read = in.read(buffer)) >= 0) {
                    digest.update(buffer, 0, read);
                }
            } finally {
                in.close();
            }
            byte[] hash = digest.digest();
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b & 0xff));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException ex) {
            throw new IOException("SHA-256 digest is not available.", ex);
        }
    }

    private static void writeMetadata(SourceProfile source, Options options, ImportReport report) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(report.metadataFile), StandardCharsets.UTF_8));
        try {
            writer.write("{\n");
            jsonField(writer, "source_file", source.inputFile.getAbsolutePath(), true);
            jsonField(writer, "selected_dataset_type", source.selectedKind.toString(), true);
            jsonField(writer, "detected_dataset_type", source.detectedKind.toString(), true);
            jsonField(writer, "source_format", source.sourceFormat, true);
            jsonField(writer, "source_rows", Long.toString(source.dataRows), true, true);
            jsonField(writer, "recognized_uci_online_retail", Boolean.toString(source.recognizedUciRetail), true, true);
            jsonField(writer, "recognized_journal_columns", Boolean.toString(source.recognizedJournal), true, true);
            jsonField(writer, "recognized_tweet_columns", Boolean.toString(source.recognizedTweets), true, true);
            jsonField(writer, "delimiter", source.delimiter.label, true);
            jsonField(writer, "mapping_profile", options.profile.toString(), true);
            jsonField(writer, "stable_followee_per_user", Boolean.toString(options.forceSingleFolloweePerUser), true, true);
            jsonField(writer, "output_file", report.outputFile.getAbsolutePath(), true);
            jsonField(writer, "sha256", report.datasetSha256, true);
            jsonField(writer, "written_rows", Long.toString(report.writtenRows), true, true);
            jsonField(writer, "unique_users", Integer.toString(report.uniqueUsers), true, true);
            jsonField(writer, "unique_followees", Integer.toString(report.uniqueFollowees), true, true);
            jsonLongMap(writer, "followee_label_counts", report.finalFolloweeCounts, true);
            writer.write("  \"columns\": {\n");
            jsonField(writer, "invoice", source.displayColumn(options.invoiceColumn), true);
            jsonField(writer, "stock_code", source.displayColumn(options.stockCodeColumn), true);
            jsonField(writer, "description", source.displayColumn(options.descriptionColumn), true);
            jsonField(writer, "quantity", source.displayColumn(options.quantityColumn), true);
            jsonField(writer, "invoice_date", source.displayColumn(options.invoiceDateColumn), true);
            jsonField(writer, "unit_price", source.displayColumn(options.unitPriceColumn), true);
            jsonField(writer, "customer_id", source.displayColumn(options.customerIdColumn), true);
            jsonField(writer, "country", source.displayColumn(options.countryColumn), false);
            writer.write("  }\n");
            writer.write("}\n");
        } finally {
            writer.close();
        }
    }

    private static void jsonField(BufferedWriter writer, String key, String value, boolean comma) throws IOException {
        jsonField(writer, key, value, comma, false);
    }

    private static void jsonField(BufferedWriter writer, String key, String value, boolean comma, boolean rawValue) throws IOException {
        writer.write("  \"" + escapeJson(key) + "\": ");
        if (rawValue) {
            writer.write(value);
        } else {
            writer.write("\"" + escapeJson(value) + "\"");
        }
        writer.write(comma ? ",\n" : "\n");
    }

    private static void jsonLongMap(BufferedWriter writer, String key, Map<String, Long> values, boolean comma) throws IOException {
        writer.write("  \"" + escapeJson(key) + "\": {\n");
        List<Map.Entry<String, Long>> entries = new ArrayList<Map.Entry<String, Long>>(values.entrySet());
        Collections.sort(entries, new Comparator<Map.Entry<String, Long>>() {
            public int compare(Map.Entry<String, Long> left, Map.Entry<String, Long> right) {
                int countCompare = Long.compare(right.getValue().longValue(), left.getValue().longValue());
                if (countCompare != 0) {
                    return countCompare;
                }
                return left.getKey().compareTo(right.getKey());
            }
        });
        for (int i = 0; i < entries.size(); i++) {
            Map.Entry<String, Long> entry = entries.get(i);
            writer.write("    \"" + escapeJson(entry.getKey()) + "\": " + entry.getValue());
            writer.write(i + 1 < entries.size() ? ",\n" : "\n");
        }
        writer.write("  }");
        writer.write(comma ? ",\n" : "\n");
    }

    private static String escapeJson(String value) {
        String raw = value == null ? "" : value;
        return raw.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static void writeAudit(SourceProfile source, Options options, ImportReport report) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(report.auditFile), StandardCharsets.UTF_8));
        try {
            writer.write("DSMP Dataset Import Audit\n");
            writer.write("=========================\n\n");
            writer.write("Source: " + source.inputFile.getAbsolutePath() + "\n");
            writer.write("Selected dataset type: " + source.selectedKind + "\n");
            writer.write("Detected dataset type: " + source.detectedKind + "\n");
            writer.write("Source format: " + source.sourceFormat + "\n");
            writer.write("Profile: " + options.profile + "\n");
            writer.write("Stable followee per user: " + options.forceSingleFolloweePerUser + "\n");
            writer.write("Output: " + report.outputFile.getAbsolutePath() + "\n");
            writer.write("SHA-256: " + report.datasetSha256 + "\n\n");
            writer.write("Rows read: " + report.rawRows + "\n");
            writer.write("Rows accepted: " + report.acceptedRows + "\n");
            writer.write("Rows written: " + report.writtenRows + "\n");
            writer.write("Malformed rows: " + report.malformedRows + "\n");
            writer.write("Cancelled invoices skipped: " + report.skippedCancelledInvoices + "\n");
            writer.write("Missing customers skipped: " + report.skippedMissingCustomer + "\n");
            writer.write("Missing text skipped: " + report.skippedMissingText + "\n");
            writer.write("Non-positive quantity skipped: " + report.skippedNonPositiveQuantity + "\n");
            writer.write("Non-positive unit price skipped: " + report.skippedNonPositiveUnitPrice + "\n");
            writer.write("Fallback dates used: " + report.badDateRows + "\n");
            writer.write("Unique users: " + report.uniqueUsers + "\n");
            writer.write("Unique followees: " + report.uniqueFollowees + "\n\n");
            writer.write("Followee label counts:\n");
            List<Map.Entry<String, Long>> entries =
                    new ArrayList<Map.Entry<String, Long>>(report.finalFolloweeCounts.entrySet());
            Collections.sort(entries, new Comparator<Map.Entry<String, Long>>() {
                public int compare(Map.Entry<String, Long> left, Map.Entry<String, Long> right) {
                    int countCompare = Long.compare(right.getValue().longValue(), left.getValue().longValue());
                    if (countCompare != 0) {
                        return countCompare;
                    }
                    return left.getKey().compareTo(right.getKey());
                }
            });
            for (Map.Entry<String, Long> entry : entries) {
                writer.write("- " + entry.getKey() + ": " + entry.getValue() + "\n");
            }
            writer.write("\n");
            writer.write("DSMP output format:\n");
            writer.write("followee/referenceUser<TAB>numericPostId<TAB>yyyy-MM-dd<TAB>numericUserId<TAB>userName<TAB>text\n");
        } finally {
            writer.close();
        }
    }

    private static String stripBom(String line) {
        if (line != null && line.length() > 0 && line.charAt(0) == '\ufeff') {
            return line.substring(1);
        }
        return line;
    }

    static String describeSource(SourceProfile source) {
        if (source == null) {
            return "No dataset inspected yet.";
        }
        StringBuilder sb = new StringBuilder();
        sb.append(source.detectedKind).append(" source inspected");
        if (source.detectedKind != source.selectedKind) {
            sb.append(" (selected ").append(source.selectedKind).append(")");
        }
        sb.append(". ");
        if (source.recognizedUciRetail) {
            sb.append("Official UCI Online Retail columns recognized. ");
        } else if (source.recognizedJournal) {
            sb.append("Journal-style author/title/venue columns recognized. ");
        } else if (source.recognizedTweets) {
            sb.append("Tweet-style user/text/id columns recognized. ");
        } else {
            sb.append("Generic column layout inspected. ");
        }
        sb.append(source.dataRows).append(" data rows, ");
        sb.append(source.columnNames.size()).append(" columns, ");
        sb.append(source.sourceFormat).append(", ");
        sb.append(source.delimiter.label).append(" delimiter");
        if (source.malformedRows > 0) {
            sb.append(", ").append(source.malformedRows).append(" malformed rows observed");
        }
        sb.append('.');
        return sb.toString();
    }

    static List<ColumnProfile> sortedColumnsByDistinct(SourceProfile source) {
        List<ColumnProfile> copy = new ArrayList<ColumnProfile>(source.columns);
        Collections.sort(copy, new Comparator<ColumnProfile>() {
            public int compare(ColumnProfile a, ColumnProfile b) {
                long da = a.distinctCount();
                long db = b.distinctCount();
                return Long.compare(db, da);
            }
        });
        return copy;
    }
}
