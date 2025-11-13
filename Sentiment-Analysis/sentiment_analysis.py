import pandas as pd
from textblob import TextBlob

# Load tweets data
data = pd.read_csv("tweets.csv")

# Analyze sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

data["Sentiment"] = data["Tweet"].apply(get_sentiment)

print(data[["Tweet", "Sentiment"]])
data.to_csv("sentiment_results.csv", index=False)
print("Sentiment analysis completed and saved to sentiment_results.csv")
