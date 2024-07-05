import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plot(df_path, output_path):
    sns.set_theme(style="whitegrid", context="talk")

    # Load the data from the CSV file
    df = pd.read_csv(df_path)

    # Split and clean the data
    performance_df = df.applymap(lambda x: float(x.split(' ± ')[0]) if ' ± ' in x else (float(x) if x != '-' else None))

    # Normalize each row by its maximum value
    normalized_df = performance_df.div(performance_df.max(axis=1), axis=0)

    # Calculate the average for each column
    average_performance = normalized_df.mean()

    # Sort the averages
    sorted_performance = average_performance.sort_values(ascending=False)

    # Create the bar plot
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=sorted_performance.values, y=sorted_performance.index, palette="viridis")

    # Add value labels to each bar
    for index, value in enumerate(sorted_performance):
        plt.text(0.05, index, f"{value:.2f}", color='white', ha="left", va="center", fontsize=20, weight='bold')

    # Plot customization
    plt.xlabel("Average Normalized Performance")
    plt.ylabel("Library")
    plt.title("Average Normalized Performance of Libraries")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Create a plot from benchmark results DataFrame")
    parser.add_argument(
        "-f", "--file_path", required=True, help="Path to the CSV file containing the benchmark results"
    )
    parser.add_argument("-o", "--output_path", required=True, help="Path where the plot image will be saved")
    return parser.parse_args()


def main():
    args = parse_args()
    create_plot(args.file_path, args.output_path)


if __name__ == "__main__":
    main()
