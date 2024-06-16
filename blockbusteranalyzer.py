#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-15 21:16:00 UTC
# @Version - 1.0.10
# @Description - This script analyzes block sizes in a blockchain and generates various visualizations.

import os
import sys
import json
import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import networkx as nx
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tsa.seasonal import seasonal_decompose
from tabulate import tabulate

# Define colors for console output
bash_color_reset = "\033[0m"
bash_color_red = "\033[91m"
bash_color_green = "\033[92m"
bash_color_yellow = "\033[93m"
bash_color_blue = "\033[94m"
bash_color_orange = "\033[38;5;214m"
bash_color_magenta = "\033[95m"
bash_color_cyan = "\033[96m"
bash_color_light_blue = "\033[94m"
bash_color_teal = "\033[36m"
bash_color_light_green = "\033[92m"

# Define colors for plots
py_color_green = "green"
py_color_yellow = "yellow"
py_color_orange = "orange"
py_color_red = "red"
py_color_magenta = "magenta"
py_color_blue = "blue"
py_color_teal = "teal"
py_color_dark_grey = "darkgrey"

# Initialize the shutdown event
shutdown_event = threading.Event()
executor = None

def signal_handler(sig, frame):
    print(f"{bash_color_red}\nProcess interrupted. Exiting gracefully...{bash_color_reset}")
    shutdown_event.set()
    if executor:
        executor.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def calculate_avg(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

def check_endpoint(endpoint_type, endpoint_url):
    # Placeholder function to simulate endpoint checking
    return True

def find_lowest_height(endpoint_type, endpoint_url):
    # Placeholder function to simulate finding the lowest height from an endpoint
    return 1

def parse_timestamp(timestamp):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    raise ValueError(f"time data '{timestamp}' does not match any known format")

def process_block(height, endpoint_type, endpoint_url):
    if shutdown_event.is_set():
        return None
    try:
        # Simulate fetching block data
        block_data = {
            "height": height,
            "size": np.random.uniform(0.01, 6.0),
            "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        return (block_data["height"], block_data["size"], block_data["time"])
    except Exception as e:
        print(f"Error fetching data for block {height}: {e}")
        return None

def categorize_block(block, categories):
    size = block["size"]
    if size < 1:
        categories["less_than_1MB"].append(block)
    elif 1 <= size < 2:
        categories["1MB_to_2MB"].append(block)
    elif 2 <= size < 3:
        categories["2MB_to_3MB"].append(block)
    elif 3 <= size < 5:
        categories["3MB_to_5MB"].append(block)
    else:
        categories["greater_than_5MB"].append(block)

# Chart generation functions
def generate_scatter_plot(times, sizes, colors, output_image_file_base, lower_height, upper_height):
    print(f"{bash_color_light_blue}Generating scatter chart...{bash_color_reset}")
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.scatter(times, sizes, c=colors, s=10)
    ax.set_title(f'Block Size Over Time (Scatter Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Block Size (MB)', fontsize=24)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelrotation=45, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_green, markersize=10, label='< 1MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_yellow, markersize=10, label='1MB to 2MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_orange, markersize=10, label='2MB to 3MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_red, markersize=10, label='3MB to 5MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_magenta, markersize=10, label='> 5MB')
    ]
    ax.legend(handles=legend_patches, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_scatter_plot.png")
    print(f"{bash_color_light_green}Scatter chart generated successfully.{bash_color_reset}")

def generate_enhanced_scatter_plot(times, sizes, colors, output_image_file_base, lower_height, upper_height):
    print(f"{bash_color_light_blue}Generating enhanced scatter chart...{bash_color_reset}")
    fig, ax = plt.subplots(figsize=(38, 20))
    scatter = ax.scatter(times, sizes, c=colors, s=10, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.set_title(f'Enhanced Block Size Over Time (Scatter Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Block Size (MB)', fontsize=24)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelrotation=45, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_green, markersize=10, label='< 1MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_yellow, markersize=10, label='1MB to 2MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_orange, markersize=10, label='2MB to 3MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_red, markersize=10, label='3MB to 5MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_magenta, markersize=10, label='> 5MB')
    ]
    ax.legend(handles=legend_patches, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_enhanced_scatter_plot.png")
    print(f"{bash_color_light_green}Enhanced scatter chart generated successfully.{bash_color_reset}")

def generate_cumulative_sum_plot(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating cumulative sum chart...{bash_color_reset}")
    cumulative_sum = np.cumsum(sizes)
    plt.figure(figsize=(38, 20))
    plt.plot(times, cumulative_sum, color=py_color_blue)
    plt.title('Cumulative Sum of Block Sizes Over Time', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Cumulative Sum (MB)', fontsize=24)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_cumulative_sum_plot.png")
    print(f"{bash_color_light_green}Cumulative sum chart generated successfully.{bash_color_reset}")

def generate_rolling_average_plot(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating rolling average chart...{bash_color_reset}")
    rolling_avg = pd.Series(sizes).rolling(window=100).mean()
    plt.figure(figsize=(38, 20))
    plt.plot(times, rolling_avg, color=py_color_green)
    plt.title('Rolling Average of Block Sizes Over Time', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Rolling Average Size (MB)', fontsize=24)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_rolling_average_plot.png")
    print(f"{bash_color_light_green}Rolling average chart generated successfully.{bash_color_reset}")

def generate_violin_plot(sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating violin chart...{bash_color_reset}")
    plt.figure(figsize=(38, 20))
    sns.violinplot(data=sizes)
    plt.title('Violin Plot of Block Sizes', fontsize=28)
    plt.xlabel('Block Sizes', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_violin_plot.png")
    print(f"{bash_color_light_green}Violin chart generated successfully.{bash_color_reset}")

def generate_autocorrelation_plot(sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating autocorrelation chart...{bash_color_reset}")
    pd.plotting.autocorrelation_plot(pd.Series(sizes))
    plt.title('Autocorrelation of Block Sizes', fontsize=28)
    plt.xlabel('Lag', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_autocorrelation_plot.png")
    print(f"{bash_color_light_green}Autocorrelation chart generated successfully.{bash_color_reset}")

def generate_seasonal_decomposition_plot(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating seasonal decomposition chart...{bash_color_reset}")
    result = seasonal_decompose(pd.Series(sizes, index=times), model='additive', period=365)
    fig = result.plot()
    fig.set_size_inches(38, 20)
    plt.title('Seasonal Decomposition of Block Sizes', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_seasonal_decomposition_plot.png")
    print(f"{bash_color_light_green}Seasonal decomposition chart generated successfully.{bash_color_reset}")

def generate_lag_plot(sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating lag chart...{bash_color_reset}")
    pd.plotting.lag_plot(pd.Series(sizes))
    plt.title('Lag Plot of Block Sizes', fontsize=28)
    plt.xlabel('Previous Size', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_lag_plot.png")
    print(f"{bash_color_light_green}Lag chart generated successfully.{bash_color_reset}")

def generate_heatmap_with_additional_dimensions(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating heatmap with additional dimensions...{bash_color_reset}")
    data = pd.DataFrame({'Time': pd.to_datetime(times), 'Size': sizes})
    data['Hour'] = data['Time'].dt.hour
    data['DayOfWeek'] = data['Time'].dt.dayofweek
    heatmap_data = data.pivot_table(index='Hour', columns='DayOfWeek', values='Size', aggfunc='mean')
    plt.figure(figsize=(38, 20))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title('Heatmap of Block Sizes by Hour and Day of Week', fontsize=28)
    plt.xlabel('Day of Week', fontsize=24)
    plt.ylabel('Hour of Day', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_heatmap.png")
    print(f"{bash_color_light_green}Heatmap with additional dimensions generated successfully.{bash_color_reset}")

def generate_network_graph(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating network graph...{bash_color_reset}")
    G = nx.Graph()
    for i in range(len(times)):
        G.add_node(i, time=times[i], size=sizes[i])
        if i > 0:
            G.add_edge(i, i-1)
    pos = {i: (mdates.date2num(times[i]), sizes[i]) for i in range(len(times))}
    plt.figure(figsize=(38, 20))
    nx.draw(G, pos, with_labels=False, node_size=50, node_color=py_color_teal, edge_color=py_color_dark_grey)
    plt.title('Network Graph of Block Sizes Over Time', fontsize=28)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_network_graph.png")
    print(f"{bash_color_light_green}Network graph generated successfully.{bash_color_reset}")

def generate_outlier_detection_plot(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating outlier detection chart...{bash_color_reset}")
    data = pd.Series(sizes)
    mean = data.mean()
    std_dev = data.std()
    outliers = data[(data - mean).abs() > 2 * std_dev]
    plt.figure(figsize=(38, 20))
    plt.plot(times, sizes, 'b-', label='Block Size')
    plt.scatter(outliers.index, outliers, color=py_color_red, label='Outliers')
    plt.title('Outlier Detection in Block Sizes', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_outlier_detection_plot.png")
    print(f"{bash_color_light_green}Outlier detection chart generated successfully.{bash_color_reset}")

def generate_segmented_bar_chart(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating segmented bar chart...{bash_color_reset}")
    categories = pd.cut(sizes, bins=[0, 1, 2, 3, 5, np.inf], right=False, labels=['<1MB', '1-2MB', '2-3MB', '3-5MB', '>5MB'])
    category_counts = categories.value_counts().sort_index()
    plt.figure(figsize=(38, 20))
    category_counts.plot(kind='bar', color=[py_color_green, py_color_yellow, py_color_orange, py_color_red, py_color_magenta])
    plt.title('Segmented Bar Chart of Block Sizes', fontsize=28)
    plt.xlabel('Block Size Range', fontsize=24)
    plt.ylabel('Count', fontsize=24)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_segmented_bar_chart.png")
    print(f"{bash_color_light_green}Segmented bar chart generated successfully.{bash_color_reset}")

def generate_graphs_and_table(block_data, output_image_file_base, lower_height, upper_height):
    categories = {
        "less_than_1MB": [block for block in block_data if block["size"] < 1],
        "1MB_to_2MB": [block for block in block_data if 1 <= block["size"] < 2],
        "2MB_to_3MB": [block for block in block_data if 2 <= block["size"] < 3],
        "3MB_to_5MB": [block for block in block_data if 3 <= block["size"] < 5],
        "greater_than_5MB": [block for block in block_data if block["size"] >= 5]
    }

    total_blocks = len(block_data)
    table = [
        [f"{bash_color_green}Less than 1MB{bash_color_reset}", f"{bash_color_green}{len(categories['less_than_1MB']):,}{bash_color_reset}", f"{bash_color_green}{len(categories['less_than_1MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_green}{calculate_avg([b['size'] for b in categories['less_than_1MB']]):.2f}{bash_color_reset}", f"{bash_color_green}{min([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_green}{max([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_yellow}1MB to 2MB{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB']):,}{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_yellow}{calculate_avg([b['size'] for b in categories['1MB_to_2MB']]):.2f}{bash_color_reset}", f"{bash_color_yellow}{min([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_yellow}{max([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_orange}2MB to 3MB{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB']):,}{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_orange}{calculate_avg([b['size'] for b in categories['2MB_to_3MB']]):.2f}{bash_color_reset}", f"{bash_color_orange}{min([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_orange}{max([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_red}3MB to 5MB{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']):,}{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_red}{calculate_avg([b['size'] for b in categories['3MB_to_5MB']]):.2f}{bash_color_reset}", f"{bash_color_red}{min([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_red}{max([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_magenta}Greater than 5MB{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']):,}{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_magenta}{calculate_avg([b['size'] for b in categories['greater_than_5MB']]):.2f}{bash_color_reset}", f"{bash_color_magenta}{min([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_magenta}{max([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}"]
    ]

    headers = [f"{bash_color_teal}Block Size Range{bash_color_reset}", f"{bash_color_teal}Count{bash_color_reset}", f"{bash_color_teal}Percentage{bash_color_reset}", f"{bash_color_teal}Average Size (MB){bash_color_reset}", f"{bash_color_teal}Min Size (MB){bash_color_reset}", f"{bash_color_teal}Max Size (MB){bash_color_reset}"]
    table_str = tabulate(table, headers, tablefmt="grid")
    print(f"\n{bash_color_light_blue}Number of blocks in each group for block heights {lower_height} to {upper_height}:{bash_color_reset}\n{table_str}")

    times = [parse_timestamp(block["time"]) for block in block_data]
    sizes = [block["size"] for block in block_data]
    colors = [
        py_color_green if block["size"] < 1 else
        py_color_yellow if 1 <= block["size"] < 2 else
        py_color_orange if 2 <= block["size"] < 3 else
        py_color_red if 3 <= block["size"] < 5 else
        py_color_magenta
        for block in block_data
    ]

    # Generate scatter and enhanced scatter charts first
    generate_scatter_plot(times, sizes, colors, output_image_file_base, lower_height, upper_height)
    generate_enhanced_scatter_plot(times, sizes, colors, output_image_file_base, lower_height, upper_height)

    # Generate other charts
    generate_cumulative_sum_plot(times, sizes, output_image_file_base)
    generate_rolling_average_plot(times, sizes, output_image_file_base)
    generate_violin_plot(sizes, output_image_file_base)
    generate_autocorrelation_plot(sizes, output_image_file_base)
    generate_seasonal_decomposition_plot(times, sizes, output_image_file_base)
    generate_lag_plot(sizes, output_image_file_base)
    generate_heatmap_with_additional_dimensions(times, sizes, output_image_file_base)
    generate_network_graph(times, sizes, output_image_file_base)
    generate_outlier_detection_plot(times, sizes, output_image_file_base)
    generate_segmented_bar_chart(times, sizes, output_image_file_base)

def main():
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} <block_interval> <num_threads> <lower_height> <upper_height> <connection_type> <endpoint_url> <json_file_path>")
        sys.exit(1)

    block_interval = int(sys.argv[0])
    num_threads = int(sys.argv[1])
    lower_height = int(sys.argv[2])
    upper_height = int(sys.argv[3])
    connection_type = sys.argv[4]
    endpoint_url = sys.argv[5]
    json_file_path = sys.argv[6]
    output_image_file_base = os.path.splitext(json_file_path)[0]

    # If a JSON file is specified, skip fetching and directly process the JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        generate_graphs_and_table(data["block_data"], output_image_file_base, lower_height, upper_height)
        return

    # Check endpoint availability
    if not check_endpoint(connection_type, endpoint_url):
        print(f"{bash_color_red}Error: Unable to reach the endpoint.{bash_color_reset}")
        sys.exit(1)

    # Find the lowest available height if necessary
    lowest_height = find_lowest_height(connection_type, endpoint_url)
    if lower_height < lowest_height:
        print(f"{bash_color_red}Lower height {lower_height} is less than the lowest available height {lowest_height}. Adjusting to {lowest_height}.{bash_color_reset}")
        lower_height = lowest_height

    global executor
    executor = ThreadPoolExecutor(max_workers=num_threads)

    # Fetch block data
    block_data = []
    with tqdm(total=(upper_height - lower_height + 1)) as pbar:
        futures = {
            executor.submit(process_block, height, connection_type, endpoint_url): height
            for height in range(lower_height, upper_height + 1, block_interval)
        }
        for future in as_completed(futures):
            if shutdown_event.is_set():
                print(f"{bash_color_red}Shutdown event detected. Exiting...{bash_color_reset}")
                sys.exit(0)
            result = future.result()
            if result:
                block_data.append({"height": result[0], "size": result[1], "time": result[2]})
            pbar.update(1)

    executor.shutdown(wait=True)

    if shutdown_event.is_set():
        print(f"{bash_color_red}Shutdown event detected. Exiting...{bash_color_reset}")
        sys.exit(0)

    # Categorize blocks and prepare data for JSON output
    categories = {
        "less_than_1MB": [],
        "1MB_to_2MB": [],
        "2MB_to_3MB": [],
        "3MB_to_5MB": [],
        "greater_than_5MB": []
    }

    for height, size, time in block_data:
        block = {"height": height, "size": size, "time": time}
        categorize_block(block, categories)

    data = {
        "block_data": block_data,
        "categories": {
            "less_than_1MB": categories["less_than_1MB"],
            "1MB_to_2MB": categories["1MB_to_2MB"],
            "2MB_to_3MB": categories["2MB_to_3MB"],
            "3MB_to_5MB": categories["3MB_to_5MB"],
            "greater_than_5MB": categories["greater_than_5MB"]
        }
    }

    # Save data to JSON file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, default=str)

    # Generate graphs and table
    generate_graphs_and_table(block_data, output_image_file_base, lower_height, upper_height)

if __name__ == "__main__":
    main()
