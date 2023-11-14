from helpers import clean_text, analyze_sentiment_distilbert
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

input_file = "data/shoes_data.csv"
output_file = "out/shoes_data_cleaned.csv"
output_sematic_file = "out/amazon/shoes_data_semantic"
plot_output_file_path = "out/amazon/"


file = pd.read_csv(input_file)
df = file[["reviews", "reviews_rating"]]

product_id = []
reviews = []
rates = []

for j in df.index:
    lst = [i for i in df.iloc[j].reviews.split("||")]
    lst2 = [i for i in df.iloc[j].reviews_rating.split("||")]
    for k in lst:
        product_id.append(j + 1)
        reviews.append(k)
    for l in lst2:
        rates.append(l)

df = pd.DataFrame(
    list(zip(product_id, reviews, rates)),
    columns=["Product_id", "Review", "Review_rating"],
)

df["clean_review"] = df["Review"].apply(clean_text)

df.drop("Review", axis=1, inplace=True)

df.to_csv(output_file, index=False)

if os.path.isfile(output_sematic_file):
    df = pd.read_csv(output_sematic_file)
else:
    df[["Emotion", "Sentiment_Score"]] = (
        df["clean_review"].apply(analyze_sentiment_distilbert).apply(pd.Series)
    )
    df.to_csv(
        output_sematic_file,
        index=False,
    )


# Group the DataFrame by 'Product_id'
grouped_df = df.groupby("Product_id").agg(
    {"Emotion": ["count", lambda x: (x == "POSITIVE").mean()]}
)
grouped_df.columns = ["Review_count", "Average_sentiment"]
grouped_df.reset_index(inplace=True)

# Sort the DataFrame based on the 'Review_count' and 'Average_sentiment'
sorted_df = grouped_df.sort_values(
    ["Review_count", "Average_sentiment"], ascending=[False, False]
)

# Save the recommendations results to a CSV file
sorted_df.to_csv(plot_output_file_path + "clean_data_recommendations.csv", index=False)

# Recommendation based on Positive Sentiment
positive_recommendations = df.groupby("Product_id").filter(
    lambda x: (x["Emotion"] == "POSITIVE").mean() > 0.8
)
positive_recommendations = positive_recommendations.drop_duplicates(subset="Product_id")

# Recommendation based on Negative Sentiment
negative_recommendations = df.groupby("Product_id").filter(
    lambda x: (x["Emotion"] == "NEGATIVE").mean() > 0.8
)
negative_recommendations = negative_recommendations.drop_duplicates(subset="Product_id")

# Recommendation based on Sentiment Score
average_sentiment = df.groupby("Product_id")["Sentiment_Score"].mean()
sentiment_recommendations = average_sentiment.nlargest(3).index.tolist()

# Save recommendations to a text file
recommendations_path = plot_output_file_path + "recommendations.txt"
with open(recommendations_path, "w") as file:
    file.write("Positive Sentiment Recommendations:\n")
    file.write(str(positive_recommendations) + "\n\n")
    file.write("Negative Sentiment Recommendations:\n")
    file.write(str(negative_recommendations) + "\n\n")

    file.write("\n\n")
    file.write("Sentiment Score Recommendations:\n")
    file.write(str(sentiment_recommendations) + "\n")
