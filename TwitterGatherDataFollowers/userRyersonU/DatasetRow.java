package TwitterGatherDataFollowers.userRyersonU;

final class DatasetRow
{
	final String referenceUser;
	final long tweetId;
	final String tweetDate;
	final long userId;
	final String userName;
	final String text;

	private DatasetRow(
			String referenceUser,
			long tweetId,
			String tweetDate,
			long userId,
			String userName,
			String text)
	{
		this.referenceUser = referenceUser;
		this.tweetId = tweetId;
		this.tweetDate = tweetDate;
		this.userId = userId;
		this.userName = userName;
		this.text = text;
	}

	static DatasetRow parse(String line, long lineNumber)
	{
		if (lineNumber == 1 && line.startsWith("\uFEFF"))
		{
			line = line.substring(1);
		}

		String[] fields = line.split("\t", 6);
		if (fields.length != 6)
		{
			throw new IllegalArgumentException(
					"Line " + lineNumber + " must contain exactly six tab-separated fields.");
		}
		if (fields[0].trim().isEmpty())
		{
			throw new IllegalArgumentException(
					"Line " + lineNumber + " has an empty reference user.");
		}
		if (fields[2].trim().isEmpty())
		{
			throw new IllegalArgumentException(
					"Line " + lineNumber + " has an empty date.");
		}
		if (fields[4].trim().isEmpty())
		{
			throw new IllegalArgumentException(
					"Line " + lineNumber + " has an empty user name.");
		}

		try
		{
			return new DatasetRow(
					fields[0],
					Long.parseLong(fields[1]),
					fields[2],
					Long.parseLong(fields[3]),
					fields[4],
					fields[5]);
		}
		catch (NumberFormatException e)
		{
			throw new IllegalArgumentException(
					"Line " + lineNumber + " has a non-numeric tweet ID or user ID.", e);
		}
	}
}
