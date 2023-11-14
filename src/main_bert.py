from helpers import clean_text, analyze_sentiment
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

input_file = "data/shoes_data.csv"
output_file = "out/shoes_data_cleaned.csv"
output_sematic_file = "out/shoes_data_semantic"
plot_output_file_path = "out"


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
        df["clean_review"].apply(analyze_sentiment).apply(pd.Series)
    )
    df.to_csv(
        output_sematic_file,
        index=False,
    )

# Convert 'Review_rating' column to numeric
df["Review_rating"] = df["Review_rating"].str.extract("(\d+)").astype(float)

# Group the DataFrame by 'Product_id'
grouped_df = df.groupby("Product_id").agg(
    {"Emotion": "count", "Sentiment_Score": "mean", "Review_rating": "mean"}
)
grouped_df.columns = [
    "Review_count",
    "Average_sentiment_score",
    "Average_review_rating",
]
grouped_df.reset_index(inplace=True)

# Sort the DataFrame based on the 'Review_count', 'Average_sentiment_score', and 'Average_review_rating'
sorted_df = grouped_df.sort_values(
    ["Review_count", "Average_sentiment_score", "Average_review_rating"],
    ascending=[False, False, False],
)

# Apply the recommendations
top_products = sorted_df.head(
    10
)  # Select top 10 products with the highest review count

# Recommendation 1: Analyze average sentiment scores
recommendation_1 = top_products[["Product_id", "Average_sentiment_score"]]

# Recommendation 2: Consider average review ratings
recommendation_2 = top_products[["Product_id", "Average_review_rating"]]

# Recommendation 3: Identify areas for improvement
low_sentiment_products = sorted_df.tail(
    10
)  # Select bottom 10 products with the lowest sentiment scores
recommendation_3 = low_sentiment_products[["Product_id", "Average_sentiment_score"]]

# Visualize Recommendation 1: Average sentiment scores
plt.figure(figsize=(10, 6))
plt.barh(
    np.arange(len(recommendation_1)),
    recommendation_1["Average_sentiment_score"],
    color="#ff7f0e",
)
plt.yticks(np.arange(len(recommendation_1)), recommendation_1["Product_id"])
plt.xlabel("Average Sentiment Score")
plt.title("Products with High Average Sentiment Scores")
plt.grid(axis="x")
plt.tight_layout()
plt.savefig(plot_output_file_path + "/recommendation_1.png")

# Visualize Recommendation 2: Average review ratings
plt.figure(figsize=(10, 6))
plt.barh(
    np.arange(len(recommendation_2)),
    recommendation_2["Average_review_rating"],
    color="#1f77b4",
)
plt.yticks(np.arange(len(recommendation_2)), recommendation_2["Product_id"])
plt.xlabel("Average Review Rating")
plt.title("Products with High Average Review Ratings")
plt.grid(axis="x")
plt.tight_layout()
plt.savefig(plot_output_file_path + "/recommendation_2.png")

# Visualize Recommendation 3: Areas for improvement
plt.figure(figsize=(10, 6))
plt.barh(
    np.arange(len(recommendation_3)),
    recommendation_3["Average_sentiment_score"],
    color="#d62728",
)
plt.yticks(np.arange(len(recommendation_3)), recommendation_3["Product_id"])
plt.xlabel("Average Sentiment Score")
plt.title("Products with Low Sentiment Scores (Areas for Improvement)")
plt.grid(axis="x")
plt.tight_layout()
plt.savefig(plot_output_file_path + "/recommendation_3.png")

import matplotlib.pyplot as plt
import numpy as np

# Create a single figure and axis
fig, ax = plt.subplots(figsize=(18, 10))

# Calculate the width of each sub-plot
width = 0.25

# Visualize Recommendation 1: Average sentiment scores
ax.bar(
    np.arange(len(recommendation_1)),
    recommendation_1["Average_sentiment_score"],
    width,
    color="#ff7f0e",
    label="Sentiment Scores",
)
ax.set_xticks(np.arange(len(recommendation_1)))
ax.set_xticklabels(recommendation_1["Product_id"])
ax.set_ylabel("Average Sentiment Score")
ax.set_title("Products with High Average Sentiment Scores")

# Visualize Recommendation 2: Average review ratings
ax2 = ax.twinx()
ax2.bar(
    np.arange(len(recommendation_2)) + width,
    recommendation_2["Average_review_rating"],
    width,
    color="#1f77b4",
    label="Review Ratings",
)
ax2.set_xticks(np.arange(len(recommendation_2)) + width)
ax2.set_xticklabels(recommendation_2["Product_id"])
ax2.set_ylabel("Average Review Rating")

# Visualize Recommendation 3: Areas for improvement
ax3 = ax.twinx()
ax3.bar(
    np.arange(len(recommendation_3)) + 2 * width,
    recommendation_3["Average_sentiment_score"],
    width,
    color="#d62728",
    label="Sentiment Scores",
)
ax3.set_xticks(np.arange(len(recommendation_3)) + 2 * width)
ax3.set_xticklabels(recommendation_3["Product_id"])
ax3.set_ylabel("Average Sentiment Score")

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Combine the legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="lower right")

# Save the combined plot
plt.savefig(plot_output_file_path + "/recommendations_combined.png")

# Show the combined plot
plt.show()
