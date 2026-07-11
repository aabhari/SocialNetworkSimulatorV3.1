package TwitterGatherDataFollowers.userRyersonU;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class UciRetailDatasetImporterSelfTest {
	private UciRetailDatasetImporterSelfTest() {
	}

	public static void main(String[] args) throws Exception {
		Path work = Files.createTempDirectory("uci-retail-import");
		File source = work.resolve("online-retail-sample.csv").toFile();
		String csv =
				"InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
				+ "536365,85123A,\"WHITE HANGING HEART, T-LIGHT HOLDER\",6,12/1/2010 8:26,2.55,17850,United Kingdom\n"
				+ "536365,71053,WHITE METAL LANTERN,6,12/1/2010 8:26,3.39,17850,United Kingdom\n"
				+ "C536379,D,Discount,-1,12/1/2010 9:41,27.50,14527,United Kingdom\n"
				+ "536370,22728,ALARM CLOCK BAKELIKE PINK,24,12/1/2010 8:45,3.75,12583,France\n"
				+ "536371,220,,1,12/1/2010 8:45,1.00,12583,France\n"
				+ "536372,220,RED BAG,0,12/1/2010 8:45,1.00,12583,France\n";
		Files.write(source.toPath(), csv.getBytes(StandardCharsets.UTF_8));

		UciRetailDatasetImporter.SourceProfile profile =
				UciRetailDatasetImporter.inspect(source, null);
		require(profile.recognizedUciRetail, "UCI Online Retail columns were not recognized");
		require(profile.dataRows == 6L, "Unexpected data-row count: " + profile.dataRows);
		require(profile.columnNames.size() == 8, "Unexpected column count");
		require(profile.previewRows.size() == 6, "Preview did not include all sample rows");
		require(profile.previewRows.get(0).get(2).contains(","), "Quoted comma was not preserved in description");

		UciRetailDatasetImporter.Options options =
				UciRetailDatasetImporter.Options.defaultsFor(profile);
		options.outputDirectory = work.resolve("out").toFile();
		options.outputBaseName = "uci_retail_dsmp_test";
		options.profile = UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS;
		UciRetailDatasetImporter.LivePreviewReport livePreview =
				UciRetailDatasetImporter.previewOutputRows(profile, options, 5);
		require(livePreview.rows.size() == 3, "Live preview should show three accepted sample rows");
		require("OnlineRetail".equals(livePreview.rows.get(0)[0]), "Live preview followee mismatch");
		require(livePreview.rows.get(0).length == 6, "Live preview must use six DSMP columns");
		UciRetailDatasetImporter.ImportReport report =
				UciRetailDatasetImporter.importDataset(profile, options, null);

		require(report.outputFile.isFile(), "DSMP output file was not created");
		require(report.metadataFile.isFile(), "Metadata JSON was not created");
		require(report.auditFile.isFile(), "Audit report was not created");
		require(report.writtenRows == 3L, "Expected three accepted rows, saw " + report.writtenRows);
		require(report.skippedCancelledInvoices == 1L, "Cancelled invoice was not skipped");
		require(report.skippedMissingText == 1L, "Missing description row was not skipped");
		require(report.skippedNonPositiveQuantity == 1L, "Zero-quantity row was not skipped");
		require(report.uniqueUsers == 2, "Unexpected unique user count: " + report.uniqueUsers);
		require(report.uniqueFollowees == 1, "Similar-customers profile should have one followee label");

		UciRetailDatasetImporter.Options oneClassBalanceOptions =
				UciRetailDatasetImporter.Options.defaultsFor(profile);
		oneClassBalanceOptions.profile =
				UciRetailDatasetImporter.MappingProfile.SIMILAR_CUSTOMERS;
		oneClassBalanceOptions.balanceMode =
				UciRetailDatasetImporter.DatasetBalanceMode.WEKA_SPREAD_SUBSAMPLE;
		oneClassBalanceOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.BOUNDED_AUGMENTATION;
		UciRetailDatasetImporter.ImportReport oneClassDiagnosis =
				UciRetailDatasetImporter.diagnoseBalance(
						profile,
						oneClassBalanceOptions,
						null);
		require(oneClassDiagnosis.balanceMode
				== UciRetailDatasetImporter.DatasetBalanceMode.WEKA_SPREAD_SUBSAMPLE,
				"One-class diagnosis should preserve the requested balance mode");
		require(oneClassDiagnosis.effectiveBalanceMode
				== UciRetailDatasetImporter.DatasetBalanceMode.DIAGNOSE_ONLY,
				"One-class diagnosis should mark balancing as effectively diagnostic-only");
		require(oneClassDiagnosis.balanceEngine.contains("only one followee class"),
				"One-class diagnosis should explain why the requested mode was not applied");

		List<String> lines = Files.readAllLines(report.outputFile.toPath(), StandardCharsets.UTF_8);
		require(lines.size() == 3, "Output file line count mismatch");
		for (int i = 0; i < lines.size(); i++) {
			DatasetRow row = DatasetRow.parse(lines.get(i), i + 1);
			require("OnlineRetail".equals(row.referenceUser), "Unexpected followee label: " + row.referenceUser);
			require(row.tweetDate.matches("\\d{4}-\\d{2}-\\d{2}"), "Date was not normalized");
			require(row.text.trim().length() > 0, "Empty text reached DSMP output");
			}

			String metadata = new String(Files.readAllBytes(report.metadataFile.toPath()), StandardCharsets.UTF_8);
			require(metadata.contains("\"recognized_uci_online_retail\": true"), "Metadata did not record UCI recognition");
			require(metadata.contains("\"mapping_profile\": \"Similar users / entities\""), "Metadata did not record profile");
			require(metadata.contains("\"followee_label_counts\""), "Metadata did not record followee label counts");
			require(metadata.contains("\"OnlineRetail\": 3"), "Metadata did not record OnlineRetail count");
			require(metadata.contains("\"columns\""), "Metadata JSON columns object was not written");

			UciRetailDatasetImporter.Options smartOptions =
					UciRetailDatasetImporter.Options.defaultsFor(profile);
			smartOptions.outputDirectory = work.resolve("smart-out").toFile();
			smartOptions.outputBaseName = "uci_retail_smart_category_test";
			smartOptions.profile = UciRetailDatasetImporter.MappingProfile.SMART_PRODUCT_CATEGORY;
			UciRetailDatasetImporter.ImportReport smartReport =
					UciRetailDatasetImporter.importDataset(profile, smartOptions, null);
			require(smartReport.writtenRows == 3L, "Smart category profile changed accepted row count");
			require(smartReport.uniqueFollowees == 2,
					"Smart category profile should create two dominant customer categories");
			List<String> smartLines = Files.readAllLines(smartReport.outputFile.toPath(), StandardCharsets.UTF_8);
			require("CandlesLighting".equals(DatasetRow.parse(smartLines.get(0), 1).referenceUser),
					"Lighting row was not categorized as CandlesLighting");
			require("HomeDecor".equals(DatasetRow.parse(smartLines.get(2), 3).referenceUser),
					"Clock row was not categorized as HomeDecor");
			String smartMetadata = new String(Files.readAllBytes(smartReport.metadataFile.toPath()), StandardCharsets.UTF_8);
			require(smartMetadata.contains("\"mapping_profile\": \"Retail: smart product category\""),
					"Smart category metadata did not record profile");
			require(smartMetadata.contains("\"CandlesLighting\": 2"),
					"Smart category metadata did not record CandlesLighting count");
			require(smartMetadata.contains("\"HomeDecor\": 1"),
					"Smart category metadata did not record HomeDecor count");

		File workbookFile = work.resolve("online-retail-sample.xlsx").toFile();
		createMinimalXlsx(workbookFile);
		UciRetailDatasetImporter.SourceProfile workbookProfile =
				UciRetailDatasetImporter.inspect(workbookFile, null);
		require(workbookProfile.recognizedUciRetail, "Workbook input did not recognize UCI columns");
		require("Excel workbook".equals(workbookProfile.sourceFormat), "Workbook source format was not recorded");
		require(workbookProfile.dataRows == 2L, "Workbook data-row count mismatch");
		UciRetailDatasetImporter.Options workbookOptions =
				UciRetailDatasetImporter.Options.defaultsFor(workbookProfile);
		workbookOptions.outputDirectory = work.resolve("xlsx-out").toFile();
		workbookOptions.outputBaseName = "uci_retail_xlsx_dsmp_test";
		UciRetailDatasetImporter.ImportReport workbookReport =
				UciRetailDatasetImporter.importDataset(workbookProfile, workbookOptions, null);
		List<String> workbookLines = Files.readAllLines(workbookReport.outputFile.toPath(), StandardCharsets.UTF_8);
		require(workbookLines.size() == 2, "Workbook import line count mismatch");
		require(DatasetRow.parse(workbookLines.get(0), 1).tweetDate.equals("2010-12-02"),
				"Excel serial date was not normalized");

		File journalFile = work.resolve("journal-sample.tsv").toFile();
		String journal =
				"Journal\tPaperID\tYear\tAuthorID\tAuthorName\tTitle\n"
				+ "DataScienceJournal\tP-1\t2020\tA-7\tAda Lovelace\tGraph models for distributed recommendation\n"
				+ "DataScienceJournal\tP-2\t2021\tA-8\tGrace Hopper\tEfficient parallel text mining\n";
		Files.write(journalFile.toPath(), journal.getBytes(StandardCharsets.UTF_8));
		UciRetailDatasetImporter.SourceProfile journalProfile =
				UciRetailDatasetImporter.inspect(
						journalFile,
						UciRetailDatasetImporter.DatasetKind.JOURNAL,
						null);
		require(journalProfile.selectedKind == UciRetailDatasetImporter.DatasetKind.JOURNAL,
				"Journal selected kind was not preserved");
		require(journalProfile.recognizedJournal, "Journal columns were not recognized");
		UciRetailDatasetImporter.Options journalOptions =
				UciRetailDatasetImporter.Options.defaultsFor(journalProfile);
		require(journalOptions.profile == UciRetailDatasetImporter.MappingProfile.JOURNAL_AUTHORS,
				"Journal default profile was not selected");
		journalOptions.outputDirectory = work.resolve("journal-out").toFile();
		journalOptions.outputBaseName = "journal_dsmp_test";
		UciRetailDatasetImporter.ImportReport journalReport =
				UciRetailDatasetImporter.importDataset(journalProfile, journalOptions, null);
		List<String> journalLines = Files.readAllLines(journalReport.outputFile.toPath(), StandardCharsets.UTF_8);
		require(journalLines.size() == 2, "Journal import line count mismatch");
		DatasetRow journalRow = DatasetRow.parse(journalLines.get(0), 1);
		require("DataScienceJournal".equals(journalRow.referenceUser), "Journal followee was not mapped");
		require("AdaLovelace".equals(journalRow.userName), "Journal author name was not mapped");
		require("2020-01-01".equals(journalRow.tweetDate), "Journal year was not normalized to a date");

		File tweetsFile = work.resolve("tweet-dsmp-shaped.tsv").toFile();
		String tweets =
				"TorontoStar\t745618222267924480\t2016-06-22 10:03:31\t18776647\tRyersonU\tNo packed lunch no problem\n"
				+ "weathernetwork\t715201978985529344\t2016-03-30 11:40:14\t385008941\tWeatherFan\tStorm warning issued downtown\n";
		Files.write(tweetsFile.toPath(), tweets.getBytes(StandardCharsets.UTF_8));
		UciRetailDatasetImporter.SourceProfile tweetsProfile =
				UciRetailDatasetImporter.inspect(
						tweetsFile,
						UciRetailDatasetImporter.DatasetKind.TWEETS,
						null);
		UciRetailDatasetImporter.Options tweetsOptions =
				UciRetailDatasetImporter.Options.defaultsFor(tweetsProfile);
		require(tweetsOptions.profile == UciRetailDatasetImporter.MappingProfile.TWEET_USERS,
				"Tweet default profile was not selected");
		tweetsOptions.outputDirectory = work.resolve("tweets-out").toFile();
		tweetsOptions.outputBaseName = "tweets_dsmp_test";
		UciRetailDatasetImporter.ImportReport tweetsReport =
				UciRetailDatasetImporter.importDataset(tweetsProfile, tweetsOptions, null);
		List<String> tweetLines = Files.readAllLines(tweetsReport.outputFile.toPath(), StandardCharsets.UTF_8);
		require(tweetLines.size() == 2, "Tweet import line count mismatch");
		DatasetRow tweetRow = DatasetRow.parse(tweetLines.get(0), 1);
		require("TorontoStar".equals(tweetRow.referenceUser), "Tweet reference account was not mapped");
		require("RyersonU".equals(tweetRow.userName), "Tweet username was not mapped");

		File otherFile = work.resolve("custom-other.csv").toFile();
		String other =
				"Class,RecordId,Date,EntityId,EntityName,Body\n"
				+ "ClassA,1,2024-01-05,U1,Entity One,Custom document body\n";
		Files.write(otherFile.toPath(), other.getBytes(StandardCharsets.UTF_8));
		UciRetailDatasetImporter.SourceProfile otherProfile =
				UciRetailDatasetImporter.inspect(
						otherFile,
						UciRetailDatasetImporter.DatasetKind.OTHER,
						null);
		UciRetailDatasetImporter.Options otherOptions =
				UciRetailDatasetImporter.Options.defaultsFor(otherProfile);
		require(otherOptions.profile == UciRetailDatasetImporter.MappingProfile.CUSTOM,
				"Other dataset should default to custom mapping");
		otherOptions.outputDirectory = work.resolve("other-out").toFile();
		otherOptions.outputBaseName = "other_dsmp_test";
		UciRetailDatasetImporter.ImportReport otherReport =
				UciRetailDatasetImporter.importDataset(otherProfile, otherOptions, null);
		DatasetRow otherRow = DatasetRow.parse(
				Files.readAllLines(otherReport.outputFile.toPath(), StandardCharsets.UTF_8).get(0),
				1);
		require("ClassA".equals(otherRow.referenceUser), "Other custom reference was not mapped");
		require("EntityOne".equals(otherRow.userName), "Other custom entity name was not mapped");

		File imbalancedFile = work.resolve("custom-imbalanced.csv").toFile();
		String imbalanced =
				"Class,RecordId,Date,EntityId,EntityName,Body\n"
				+ "ClassA,1,2024-01-01,101,Ada,Majority class document one\n"
				+ "ClassA,2,2024-01-02,102,Alan,Majority class document two\n"
				+ "ClassA,3,2024-01-03,103,Grace,Majority class document three\n"
				+ "ClassA,4,2024-01-04,104,Katherine,Majority class document four\n"
				+ "ClassB,5,2024-01-05,201,Barbara,Minority class document one\n";
		Files.write(imbalancedFile.toPath(), imbalanced.getBytes(StandardCharsets.UTF_8));
		UciRetailDatasetImporter.SourceProfile imbalancedProfile =
				UciRetailDatasetImporter.inspect(
						imbalancedFile,
						UciRetailDatasetImporter.DatasetKind.OTHER,
						null);
		UciRetailDatasetImporter.Options balanceOptions =
				UciRetailDatasetImporter.Options.defaultsFor(imbalancedProfile);
		balanceOptions.outputDirectory = work.resolve("balance-out").toFile();
		balanceOptions.outputBaseName = "hybrid_balance_dsmp_test";
		balanceOptions.balanceMode =
				UciRetailDatasetImporter.DatasetBalanceMode.HYBRID_CAP_AND_AUGMENT;
		balanceOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MANUAL;
		balanceOptions.balanceTargetUsersPerClass = 3;
		balanceOptions.balanceSeed = 17L;
		UciRetailDatasetImporter.ImportReport balanceDiagnosis =
				UciRetailDatasetImporter.diagnoseBalance(
						imbalancedProfile,
						balanceOptions,
						null);
		require(balanceDiagnosis.balanceOriginalUsers == 5,
				"Pre-build balance diagnosis should see five original users");
		require(balanceDiagnosis.balanceSyntheticUsers == 2,
				"Pre-build balance diagnosis should predict two synthetic users");
		require(balanceDiagnosis.finalFolloweeUserCounts.size() == 2,
				"Pre-build balance diagnosis should report two final followee classes");
		UciRetailDatasetImporter.ImportReport balanceReport =
				UciRetailDatasetImporter.importDataset(
						imbalancedProfile,
						balanceOptions,
						null);
		require(balanceReport.balanceOriginalUsers == 5,
				"Hybrid balance should see five original users, saw "
						+ balanceReport.balanceOriginalUsers);
		require(balanceReport.balanceSelectedOriginalUsers == 4,
				"Hybrid balance should keep four original users, saw "
						+ balanceReport.balanceSelectedOriginalUsers);
		require(balanceReport.balanceDroppedUsers == 1,
				"Hybrid balance should drop one majority user");
		require(balanceReport.balanceSyntheticUsers == 2,
				"Hybrid balance should add two synthetic minority users");
		require(balanceReport.balanceDroppedRows == 1L,
				"Hybrid balance should drop one majority row");
		require(balanceReport.uniqueUsers == 6,
				"Hybrid balance should finish with six user instances");
		require(balanceReport.uniqueFollowees == 2,
				"Hybrid balance should preserve two followee classes");
		List<String> balanceLines = Files.readAllLines(
				balanceReport.outputFile.toPath(),
				StandardCharsets.UTF_8);
		require(balanceLines.size() == 6,
				"Hybrid balance output should contain six DSMP rows");
		int classARows = 0;
		int classBRows = 0;
		for (int i = 0; i < balanceLines.size(); i++) {
			DatasetRow row = DatasetRow.parse(balanceLines.get(i), i + 1);
			if ("ClassA".equals(row.referenceUser)) {
				classARows++;
			} else if ("ClassB".equals(row.referenceUser)) {
				classBRows++;
			}
		}
		require(classARows == 3, "Hybrid balance should write three ClassA rows");
		require(classBRows == 3, "Hybrid balance should write three ClassB rows");
		String balanceMetadata = new String(
				Files.readAllBytes(balanceReport.metadataFile.toPath()),
				StandardCharsets.UTF_8);
		require(balanceMetadata.contains("\"balance_mode\": \"Hybrid cap + augmentation\""),
				"Hybrid balance metadata did not record the balance mode");
		require(balanceMetadata.contains("\"balance_synthetic_users\": 2"),
				"Hybrid balance metadata did not record synthetic users");
		require(balanceMetadata.contains("\"balance_dropped_users\": 1"),
				"Hybrid balance metadata did not record dropped users");
		require(balanceMetadata.contains("\"final_followee_user_counts\""),
				"Hybrid balance metadata did not record final user counts");
		require(balanceMetadata.contains("\"ClassA\": 3"),
				"Hybrid balance metadata did not record ClassA count");
		require(balanceMetadata.contains("\"ClassB\": 3"),
				"Hybrid balance metadata did not record ClassB count");

		UciRetailDatasetImporter.Options wekaBalanceOptions =
				UciRetailDatasetImporter.Options.defaultsFor(imbalancedProfile);
		wekaBalanceOptions.outputDirectory = work.resolve("weka-balance-out").toFile();
		wekaBalanceOptions.outputBaseName = "weka_spread_balance_dsmp_test";
		wekaBalanceOptions.balanceMode =
				UciRetailDatasetImporter.DatasetBalanceMode.WEKA_SPREAD_SUBSAMPLE;
		wekaBalanceOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MANUAL;
		wekaBalanceOptions.balanceTargetUsersPerClass = 1;
		wekaBalanceOptions.balanceSeed = 17L;
		UciRetailDatasetImporter.ImportReport wekaBalanceReport =
				UciRetailDatasetImporter.importDataset(
						imbalancedProfile,
						wekaBalanceOptions,
						null);
		require(wekaBalanceReport.balanceEngine.contains("SpreadSubsample"),
				"Weka balance did not use SpreadSubsample");
		require(wekaBalanceReport.uniqueUsers == 2,
				"Weka SpreadSubsample should finish with two users");
		require(wekaBalanceReport.writtenRows == 2L,
				"Weka SpreadSubsample should write two DSMP rows");
		List<String> wekaBalanceLines = Files.readAllLines(
				wekaBalanceReport.outputFile.toPath(),
				StandardCharsets.UTF_8);
		require(wekaBalanceLines.size() == 2,
				"Weka balance output file should contain two rows");
		for (int i = 0; i < wekaBalanceLines.size(); i++) {
			DatasetRow.parse(wekaBalanceLines.get(i), i + 1);
		}

		File autoTargetFile = work.resolve("auto-targets.csv").toFile();
		StringBuilder autoTargets = new StringBuilder(
				"Class,RecordId,Date,EntityId,EntityName,Body\n");
		appendAutoTargetRows(autoTargets, "ClassA", 20, 1000);
		appendAutoTargetRows(autoTargets, "ClassB", 10, 2000);
		appendAutoTargetRows(autoTargets, "ClassC", 6, 3000);
		appendAutoTargetRows(autoTargets, "ClassD", 3, 4000);
		appendAutoTargetRows(autoTargets, "ClassE", 1, 5000);
		Files.write(autoTargetFile.toPath(),
				autoTargets.toString().getBytes(StandardCharsets.UTF_8));
		UciRetailDatasetImporter.SourceProfile autoTargetProfile =
				UciRetailDatasetImporter.inspect(
						autoTargetFile,
						UciRetailDatasetImporter.DatasetKind.OTHER,
						null);
		UciRetailDatasetImporter.Options autoTargetOptions =
				UciRetailDatasetImporter.Options.defaultsFor(autoTargetProfile);
		autoTargetOptions.balanceMode =
				UciRetailDatasetImporter.DatasetBalanceMode.STRICT_USER_BALANCE;
		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MODE_RECOMMENDED;
		UciRetailDatasetImporter.ImportReport strictAuto =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(strictAuto.balanceTargetUsersPerClass == 1,
				"Strict mode-recommended target should use the smallest class");
		require(strictAuto.balanceTargetPolicy
				== UciRetailDatasetImporter.DatasetBalanceTargetPolicy.STRICT_MINORITY,
				"Strict mode-recommended policy should resolve to strict minority");
		autoTargetOptions.balanceMode =
				UciRetailDatasetImporter.DatasetBalanceMode.CONSERVATIVE_MAJORITY_CAP;
		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MODE_RECOMMENDED;
		UciRetailDatasetImporter.ImportReport conservativeAuto =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(conservativeAuto.balanceTargetUsersPerClass == 10,
				"Conservative mode-recommended target should use the 75th percentile");
		require(conservativeAuto.balanceTargetPolicy
				== UciRetailDatasetImporter.DatasetBalanceTargetPolicy.CONSERVATIVE_CAP,
				"Conservative mode-recommended policy should resolve to conservative cap");
		autoTargetOptions.balanceMode =
				UciRetailDatasetImporter.DatasetBalanceMode.HYBRID_CAP_AND_AUGMENT;
		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MODE_RECOMMENDED;
		UciRetailDatasetImporter.ImportReport hybridAuto =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(hybridAuto.balanceTargetUsersPerClass == 6,
				"Hybrid mode-recommended target should use the minority-safe bounded target");
		require(hybridAuto.balanceTargetPolicy
				== UciRetailDatasetImporter.DatasetBalanceTargetPolicy.BOUNDED_AUGMENTATION,
				"Hybrid mode-recommended policy should resolve to bounded augmentation");
		autoTargetOptions.balanceMode =
				UciRetailDatasetImporter.DatasetBalanceMode.WEKA_SUPERVISED_RESAMPLE;
		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.MODE_RECOMMENDED;
		UciRetailDatasetImporter.ImportReport resampleAuto =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(resampleAuto.balanceTargetUsersPerClass == 8,
				"Weka resample mode-recommended target should use average class size");
		require(resampleAuto.balanceTargetPolicy
				== UciRetailDatasetImporter.DatasetBalanceTargetPolicy.PRESERVE_TOTAL_SIZE,
				"Weka resample mode-recommended policy should resolve to preserve total size");

		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.STRICT_MINORITY;
		UciRetailDatasetImporter.ImportReport explicitStrict =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(explicitStrict.balanceTargetUsersPerClass == 1,
				"Explicit strict-minority policy should use the smallest class");
		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.CONSERVATIVE_CAP;
		UciRetailDatasetImporter.ImportReport explicitCap =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(explicitCap.balanceTargetUsersPerClass == 10,
				"Explicit conservative-cap policy should use the 75th percentile");
		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.BOUNDED_AUGMENTATION;
		UciRetailDatasetImporter.ImportReport explicitBounded =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(explicitBounded.balanceTargetUsersPerClass == 6,
				"Explicit bounded-augmentation policy should use the bounded target");
		autoTargetOptions.balanceTargetPolicy =
				UciRetailDatasetImporter.DatasetBalanceTargetPolicy.PRESERVE_TOTAL_SIZE;
		UciRetailDatasetImporter.ImportReport explicitPreserve =
				UciRetailDatasetImporter.diagnoseBalance(autoTargetProfile,
						autoTargetOptions, null);
		require(explicitPreserve.balanceTargetUsersPerClass == 8,
				"Explicit preserve-total-size policy should use average class size");

		System.out.println("UciRetailDatasetImporterSelfTest passed");
	}

	private static void appendAutoTargetRows(StringBuilder sb,
			String className,
			int users,
			int offset) {
		for (int i = 0; i < users; i++) {
			int id = offset + i;
			sb.append(className).append(',')
					.append(id).append(',')
					.append("2024-02-01").append(',')
					.append(id).append(',')
					.append(className).append("User").append(i).append(',')
					.append("Auto target test document ").append(id)
					.append('\n');
		}
	}

	private static void createMinimalXlsx(File target) throws Exception {
		ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(target.toPath()));
		try {
			zip.putNextEntry(new ZipEntry("xl/worksheets/sheet1.xml"));
			String xml =
					"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
					+ "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>"
					+ row(1, new String[] { "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country" }, true)
					+ row(2, new String[] { "536500", "85000", "BLUE CERAMIC BOWL", "3", "40514.4243", "4.25", "17777", "United Kingdom" }, false)
					+ row(3, new String[] { "536501", "22900", "GARDEN FLOWER SIGN", "2", "40514.4271", "1.95", "18888", "France" }, false)
					+ "</sheetData></worksheet>";
			zip.write(xml.getBytes(StandardCharsets.UTF_8));
			zip.closeEntry();
		} finally {
			zip.close();
		}
	}

	private static String row(int rowNumber, String[] values, boolean inlineAll) {
		StringBuilder sb = new StringBuilder();
		sb.append("<row r=\"").append(rowNumber).append("\">");
		for (int i = 0; i < values.length; i++) {
			String ref = columnName(i) + rowNumber;
			boolean inline = inlineAll || i == 0 || i == 1 || i == 2 || i == 7;
			if (inline) {
				sb.append("<c r=\"").append(ref).append("\" t=\"inlineStr\"><is><t>")
						.append(escapeXml(values[i]))
						.append("</t></is></c>");
			} else {
				sb.append("<c r=\"").append(ref).append("\"><v>")
						.append(values[i])
						.append("</v></c>");
			}
		}
		sb.append("</row>");
		return sb.toString();
	}

	private static String columnName(int index) {
		StringBuilder sb = new StringBuilder();
		int value = index + 1;
		while (value > 0) {
			int rem = (value - 1) % 26;
			sb.insert(0, (char) ('A' + rem));
			value = (value - 1) / 26;
		}
		return sb.toString();
	}

	private static String escapeXml(String value) {
		return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
	}

	private static void require(boolean condition, String message) {
		if (!condition) {
			throw new AssertionError(message);
		}
	}
}
