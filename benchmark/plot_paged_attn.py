import pandas as pd
import matplotlib.pyplot as plt


def plot_profiling_results(csv_filename: str, output_filename: str):
    """
    Reads profiling data from a CSV file and generates a line chart.

    Args:
        csv_filename (str): The name of the input CSV file.
        output_filename (str): The name of the output image file.
    """
    try:
        # Read the data from the specified CSV file
        df = pd.read_csv(csv_filename)
    except FileNotFoundError as e:
        print(f"File not found. Error: {e}")
        return

    # Create subplots for each batch_size
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 16), sharex=True)
    fig.suptitle("Performance Comparison by Batch Size", fontsize=20, y=0.98)

    batch_sizes = sorted(df["batch_size"].unique())
    line_styles = ["-", "--", "-."]
    markers = ["o", "s", "^"]

    # Use a colormap for visual distinction between devices
    colors = plt.cm.viridis

    for i, batch_size in enumerate(batch_sizes):
        ax = axes[i]
        subset = df[df["batch_size"] == batch_size]

        devices = sorted(subset["device"].unique())
        block_sizes = sorted(subset["block_size"].unique())

        # Create a color map for the devices
        color_map = {
            device: colors(i / len(devices))
            for i, device in enumerate(devices)
        }

        for device in devices:
            for k, block_size in enumerate(block_sizes):
                final_subset = subset[
                    (subset["device"] == device)
                    & (subset["block_size"] == block_size)
                ]
                if not final_subset.empty:
                    label = f"{device} (Block Size: {block_size})"
                    ax.plot(
                        final_subset["seq_len"],
                        final_subset["time_per_execution"],
                        label=label,
                        marker=markers[k % len(markers)],
                        linestyle=line_styles[k % len(line_styles)],
                        color=color_map[device],
                    )

        ax.set_title(f"Batch Size: {batch_size}", fontsize=16)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylabel("Time per Execution (seconds)", fontsize=12)
        ax.grid(True, which="both", ls="--")
        ax.legend(title="Device and Block Size")

    plt.xlabel("Sequence Length", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot to a file
    plt.savefig(output_filename)
    print(f"Chart saved as '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    plot_profiling_results(
        csv_filename="paged_attn_benchmark_results.csv",
        output_filename="paged_attn_benchmark_results.png",
    )
