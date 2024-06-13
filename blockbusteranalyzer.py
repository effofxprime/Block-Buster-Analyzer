#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-17 17:00:00 UTC
# @Version - 1.0.8
# @Description - A tool to analyze block sizes in a blockchain.

import requests
import requests_unixsocket
import json
import time
import sys
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import signal
import threading
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import networkx as nx

# ANSI escape sequences for 256 colors (Bash colors)
bash_color_green = "\033[38;5;10m"  # Green
bash_color_yellow = "\033[38;5;11m"  # Yellow
bash_color_orange = "\033[38;5;214m"  # Orange
bash_color_red = "\033[38;5;9m"  # Red
bash_color_magenta = "\033[38;5;13m"  # Magenta
bash_color_light_blue = "\033[38;5;123m"  # Light Blue
bash_color_dark_grey = "\033[38;5;245m"  # Dark Grey
bash_color_light_green = "\033[38;5;121m"  # Light Green
bash_color_teal = "\033[38;5;74m"  # Teal
bash_color_reset = "\033[0m"  # Reset

# Python color names for Matplotlib
py_color_green = "green"
py_color_yellow = "yellow"
py_color_orange = "orange"
py_color_red = "red"
py_color_magenta = "magenta"
py_color_light_blue = "lightblue"
py_color_dark_grey = "darkgrey"
py_color_light_green = "lightgreen"
py_color_teal = "teal"
py_color_blue = "blue"

# Global variable to manage executor shutdown
executor = None
shutdown_event = threading.Event()

def check_endpoint(endpoint_type, endpoint_url):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            encoded_url = f"http+unix://{quote_plus(endpoint_url)}/health"
            response = session.get(encoded_url, timeout=5)
        else:
            response = requests.get(f"{endpoint_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def fetch_block_info(endpoint_type, endpoint_url, height):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            encoded_url = f"http+unix://{quote_plus(endpoint_url)}/block?height={height}"
            response = session.get(encoded_url, timeout=10)
        else:
            response = requests.get(f"{endpoint_url}/block?height={height}", timeout=10)
            response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data for block {height}: {e}")
        return None

def find_lowest_height(endpoint_type, endpoint_url):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            encoded_url = f"http+unix://{quote_plus(endpoint_url)}/block?height=1"
            response = session.get(encoded_url, timeout=10)
        else:
            response = requests.get(f"{endpoint_url}/block?height=1", timeout=10)
            response.raise_for_status()
        block_info = response.json()
        if 'error' in block_info and 'data' in block_info['error']:
            data_message = block_info['error']['data']
            print(f"Data message: {data_message}")  # Essential message
            if "lowest height is" in data_message:
                return int(data_message.split("lowest height is")[1].strip())
    except requests.HTTPError as e:
        if e.response.status_code == 500:
            error_response = e.response.json()
            if 'error' in error_response and 'data' in error_response['error']:
                data_message = error_response['error']['data']
                print(f"Data message: {data_message}")  # Essential message
                if "lowest height is" in data_message:
                    return int(data_message.split("lowest height is")[1].strip())
        else:
            print(f"HTTPError: {e}")  # Debugging output
    except requests.RequestException as e:
        print(f"RequestException: {e}")  # Debugging output
        return None

    return 1  # Return 1 if height 1 is available or no error is found

def calculate_avg(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

def parse_timestamp(timestamp):
    try:
        if '.' in timestamp:
            timestamp = timestamp.split('.')[0] + 'Z'
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError(f"time data '{timestamp}' does not match any known format")

def process_block(height, endpoint_type, endpoint_url):
    if shutdown_event.is_set():
        return None

    block_info = fetch_block_info(endpoint_type, endpoint_url, height)
    if block_info is None:
        return None

    block_size = len(json.dumps(block_info))
    block_size_mb = block_size / 1048576  # Base 2: 1MB = 1,048,576 bytes

    block_time = parse_timestamp(block_info['result']['block']['header']['time'])
    return (height, block_size_mb, block_time)

def signal_handler(sig, frame):
    print(f"{bash_color_red}\nProcess interrupted. Exiting gracefully...{bash_color_reset}")
    shutdown_event.set()
    if executor:
        executor.shutdown(wait=False, cancel_futures=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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

# New chart generation functions
def generate_scatter_plot(times, sizes, colors, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.scatter(times, sizes, color=colors)
    ax.set_title(f'Block Size Over Time (Scatter Plot)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Block Size (MB)', fontsize=24)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', labelrotation=45, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', label='Less than 1MB', markerfacecolor=py_color_green, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='1MB to 2MB', markerfacecolor=py_color_yellow, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='2MB to 3MB', markerfacecolor=py_color_orange, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='3MB to 5MB', markerfacecolor=py_color_red, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Greater than 5MB', markerfacecolor=py_color_magenta, markersize=10)
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_scatter_plot.png")
    print(f"{bash_color_light_green}Scatter plot generated successfully.{bash_color_reset}")

def generate_enhanced_scatter_plot(times, sizes, colors, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.scatter(times, sizes, color=colors, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.set_title(f'Enhanced Block Size Over Time (Scatter Plot)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Block Size (MB)', fontsize=24)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', labelrotation=45, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', label='Less than 1MB', markerfacecolor=py_color_green, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='1MB to 2MB', markerfacecolor=py_color_yellow, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='2MB to 3MB', markerfacecolor=py_color_orange, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='3MB to 5MB', markerfacecolor=py_color_red, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Greater than 5MB', markerfacecolor=py_color_magenta, markersize=10)
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_enhanced_scatter_plot.png")
    print(f"{bash_color_light_green}Enhanced scatter plot generated successfully.{bash_color_reset}")

def generate_cumulative_sum_plot(times, sizes, output_image_file_base):
    cumulative_sum = np.cumsum(sizes)
    plt.figure(figsize=(38, 20))
    plt.plot(times, cumulative_sum, color=py_color_blue)
    plt.title('Cumulative Sum of Block Sizes Over Time', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Cumulative Size (MB)', fontsize=24)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_cumulative_sum_plot.png")
    print(f"{bash_color_light_green}Cumulative sum plot generated successfully.{bash_color_reset}")

def generate_rolling_average_plot(times, sizes, output_image_file_base):
    rolling_avg = pd.Series(sizes).rolling(window=100).mean()
    plt.figure(figsize=(38, 20))
    plt.plot(times, rolling_avg, color=py_color_green)
    plt.title('Rolling Average of Block Sizes', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Rolling Average Size (MB)', fontsize=24)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_rolling_average_plot.png")
    print(f"{bash_color_light_green}Rolling average plot generated successfully.{bash_color_reset}")

def generate_violin_plot(sizes, output_image_file_base):
    plt.figure(figsize=(38, 20))
    sns.violinplot(data=sizes)
    plt.title('Violin Plot of Block Sizes', fontsize=28)
    plt.xlabel('Block Sizes', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_violin_plot.png")
    print(f"{bash_color_light_green}Violin plot generated successfully.{bash_color_reset}")

def generate_autocorrelation_plot(sizes, output_image_file_base):
    pd.plotting.autocorrelation_plot(pd.Series(sizes))
    plt.title('Autocorrelation of Block Sizes', fontsize=28)
    plt.xlabel('Lag', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_autocorrelation_plot.png")
    print(f"{bash_color_light_green}Autocorrelation plot generated successfully.{bash_color_reset}")

def generate_seasonal_decomposition_plot(times, sizes, output_image_file_base):
    result = seasonal_decompose(pd.Series(sizes, index=times), model='additive', period=365)
    fig = result.plot()
    fig.set_size_inches(38, 20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_seasonal_decomposition_plot.png")
    print(f"{bash_color_light_green}Seasonal decomposition plot generated successfully.{bash_color_reset}")

def generate_lag_plot(sizes, output_image_file_base):
    pd.plotting.lag_plot(pd.Series(sizes))
    plt.title('Lag Plot of Block Sizes', fontsize=28)
    plt.xlabel('Previous Size', fontsize=24)
    plt.ylabel('Current Size', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_lag_plot.png")
    print(f"{bash_color_light_green}Lag plot generated successfully.{bash_color_reset}")

def generate_heatmap_with_additional_dimensions(times, sizes, output_image_file_base):
    data = pd.DataFrame({'Time': times, 'Size': sizes})
    data['Hour'] = data['Time'].dt.hour
    data['DayOfWeek'] = data['Time'].dt.dayofweek
    heatmap_data = data.pivot_table(index='Hour', columns='DayOfWeek', values='Size', aggfunc='mean')
    plt.figure(figsize=(20, 20))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
    plt.title('Heatmap of Block Sizes by Hour and Day of Week', fontsize=28)
    plt.xlabel('Day of Week', fontsize=24)
    plt.ylabel('Hour of Day', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_heatmap.png")
    print(f"{bash_color_light_green}Heatmap generated successfully.{bash_color_reset}")

def generate_network_graph(times, sizes, output_image_file_base):
    G = nx.Graph()
    for i in range(len(times)):
        G.add_node(i, time=times[i], size=sizes[i])
        if i > 0:
            G.add_edge(i, i-1)
    pos = {i: (times[i], sizes[i]) for i in range(len(times))}
    plt.figure(figsize=(38, 20))
    nx.draw(G, pos, with_labels=False, node_size=50, node_color=py_color_teal, edge_color=py_color_dark_grey)
    plt.title('Network Graph of Block Sizes Over Time', fontsize=28)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_network_graph.png")
    print(f"{bash_color_light_green}Network graph generated successfully.{bash_color_reset}")

def generate_outlier_detection_plot(times, sizes, output_image_file_base):
    data = pd.Series(sizes)
    mean = data.mean()
    std_dev = data.std()
    outliers = data[(data - mean).abs() > 2 * std_dev]
    plt.figure(figsize=(38, 20))
    plt.plot(times, sizes, label='Block Size', color=py_color_blue)
    plt.scatter(outliers.index, outliers, color=py_color_red, label='Outliers')
    plt.title('Outlier Detection in Block Sizes', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_outlier_detection_plot.png")
    print(f"{bash_color_light_green}Outlier detection plot generated successfully.{bash_color_reset}")

def generate_segmented_bar_chart(times, sizes, output_image_file_base):
    categories = pd.cut(sizes, bins=[0, 1, 2, 3, 5, np.inf], right=False, labels=['<1MB', '1-2MB', '2-3MB', '3-5MB', '>5MB'])
    category_counts = categories.value_counts().sort_index()
    plt.figure(figsize=(38, 20))
    category_counts.plot(kind='bar', color=[py_color_green, py_color_yellow, py_color_orange, py_color_red, py_color_magenta])
    plt.title('Segmented Bar Chart of Block Sizes', fontsize=28)
    plt.xlabel('Block Size (MB)', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_segmented_bar_chart.png")
    print(f"{bash_color_light_green}Segmented bar chart generated successfully.{bash_color_reset}")

def generate_graphs_and_table(data, output_image_file_base, lower_height, upper_height):
    block_data = data["block_data"]
    times = [parse_timestamp(block["time"]) for block in block_data]
    sizes = [block["size"] for block in block_data]

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
        [f"{bash_color_yellow}1MB to 2MB{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB']):,}{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_yellow}{calculate_avg([b['size'] for b in categories['1MB_to_2MB']])::.2f}{bash_color_reset}", f"{bash_color_yellow}{min([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_yellow}{max([b['size'] for b in categories['1MB_to_2MB']], default=0)::.2f}{bash_color_reset}"],
        [f"{bash_color_orange}2MB to 3MB{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB']):,}{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_orange}{calculate_avg([b['size'] for b in categories['2MB_to_3MB']]):.2f}{bash_color_reset}", f"{bash_color_orange}{min([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_orange}{max([b['size'] for b in categories['2MB_to_3MB']], default=0)::.2f}{bash_color_reset}"],
        [f"{bash_color_red}3MB to 5MB{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']):,}{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_red}{calculate_avg([b['size'] for b in categories['3MB_to_5MB']]):.2f}{bash_color_reset}", f"{bash_color_red}{min([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_red}{max([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_magenta}Greater than 5MB{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']):,}{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_magenta}{calculate_avg([b['size'] for b in categories['greater_than_5MB']]):.2f}{bash_color_reset}", f"{bash_color_magenta}{min([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_magenta}{max([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}"]
    ]

    headers = ["Block Size Range", "Count", "Percentage", "Average Size (MB)", "Min Size (MB)", "Max Size (MB)"]
    table_str = tabulate(table, headers, tablefmt="grid")
    print(f"\nNumber of blocks in each group for block heights {lower_height} to {upper_height}:\n{table_str}")

    print(f"{bash_color_light_blue}Generating scatter plot...{bash_color_reset}")
    generate_scatter_plot(times, sizes, [py_color_green if size < 1 else py_color_yellow if size < 2 else py_color_orange if size < 3 else py_color_red if size < 5 else py_color_magenta for size in sizes], output_image_file_base, lower_height, upper_height)
    print(f"{bash_color_light_blue}Generating enhanced scatter plot...{bash_color_reset}")
    generate_enhanced_scatter_plot(times, sizes, [py_color_green if size < 1 else py_color_yellow if size < 2 else py_color_orange if size < 3 else py_color_red if size < 5 else py_color_magenta for size in sizes], output_image_file_base, lower_height, upper_height)
    print(f"{bash_color_light_blue}Generating cumulative sum plot...{bash_color_reset}")
    generate_cumulative_sum_plot(times, sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating rolling average plot...{bash_color_reset}")
    generate_rolling_average_plot(times, sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating violin plot...{bash_color_reset}")
    generate_violin_plot(sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating autocorrelation plot...{bash_color_reset}")
    generate_autocorrelation_plot(sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating seasonal decomposition plot...{bash_color_reset}")
    generate_seasonal_decomposition_plot(times, sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating lag plot...{bash_color_reset}")
    generate_lag_plot(sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating heatmap with additional dimensions...{bash_color_reset}")
    generate_heatmap_with_additional_dimensions(times, sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating network graph...{bash_color_reset}")
    generate_network_graph(times, sizes, output_image_file_base)
    print(f"{bash_color_light_blue}Generating outlier detection plot...{bash_color_reset}")
    generate_outlier_detection_plot(times, sizes, output_image_file_base)

def main():
    if len(sys.argv) < 8:
        print("Usage: python3 blockbusteranalyzer.py <num_workers> <batch_size> <lower_height> <upper_height> <endpoint_type> <endpoint_url> <output_json_file>")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    lower_height = int(sys.argv[3])
    upper_height = int(sys.argv[4])
    endpoint_type = sys.argv[5]
    endpoint_url = sys.argv[6]
    output_json_file = sys.argv[7]
    output_image_file_base = output_json_file.replace('.json', '')

    # If a JSON file is specified, skip fetching and directly process the JSON file
    if os.path.exists(output_json_file):
        with open(output_json_file, 'r') as f:
            data = json.load(f)
        generate_graphs_and_table(data, output_image_file_base, lower_height, upper_height)
        return

    global executor
    executor = ThreadPoolExecutor(max_workers=num_workers)

    if not check_endpoint(endpoint_type, endpoint_url):
        print(f"Endpoint {endpoint_url} is not reachable.")
        sys.exit(1)

    lowest_height = find_lowest_height(endpoint_type, endpoint_url)
    if lower_height < lowest_height:
        print(f"Lower height {lower_height} is less than the lowest available height {lowest_height}. Adjusting to {lowest_height}.")
        lower_height = lowest_height

    tasks = []
    for start in range(lower_height, upper_height, batch_size):
        end = min(start + batch_size, upper_height + 1)
        for height in range(start, end):
            tasks.append(executor.submit(process_block, height, endpoint_type, endpoint_url))

    results = []
    for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing blocks", unit="block"):
        result = future.result()
        if result is not None:
            results.append(result)

    if shutdown_event.is_set():
        print(f"{bash_color_red}Shutdown event detected. Exiting...{bash_color_reset}")
        sys.exit(0)

    # Categorize blocks and prepare data for JSON output
    block_data = []
    categories = {
        "less_than_1MB": [],
        "1MB_to_2MB": [],
        "2MB_to_3MB": [],
        "3MB_to_5MB": [],
        "greater_than_5MB": []
    }

    for height, size, time in results:
        block = {"height": height, "size": size, "time": time}
        block_data.append(block)
        categorize_block(block, categories)

    data = {
        "block_data": block_data,
        "less_than_1MB": categories["less_than_1MB"],
        "1MB_to_2MB": categories["1MB_to_2MB"],
        "2MB_to_3MB": categories["2MB_to_3MB"],
        "3MB_to_5MB": categories["3MB_to_5MB"],
        "greater_than_5MB": categories["greater_than_5MB"]
    }

    # Save data to JSON file
    with open(output_json_file, 'w') as f:
        json.dump(data, f, default=str)

    # Generate graphs and table
    generate_graphs_and_table(data, output_image_file_base, lower_height, upper_height)

if __name__ == "__main__":
    main()
