#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-17 15:00:00 UTC
# @Version - 1.0.7
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
        executor.shutdown(wait=False)
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

def generate_heatmap_with_dimensions(times, sizes, output_image_file_base):
    data_matrix = np.column_stack([times, sizes])
    plt.figure(figsize=(38, 20))
    sns.heatmap(data_matrix, cmap="YlGnBu")
    plt.title('Heatmap of Block Sizes with Additional Dimensions', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_heatmap_with_dimensions.png")
    print(f"{bash_color_light_green}Heatmap with additional dimensions generated successfully.{bash_color_reset}")

def generate_network_graph(times, sizes, output_image_file_base):
    G = nx.Graph()
    for i in range(len(times) - 1):
        G.add_edge(times[i], times[i+1], weight=sizes[i])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(38, 20))
    nx.draw(G, pos, with_labels=True, node_size=50, node_color=py_color_blue, edge_color=sizes, edge_cmap=plt.cm.Blues)
    plt.title('Network Graph of Blocks', fontsize=28)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_network_graph.png")
    print(f"{bash_color_light_green}Network graph generated successfully.{bash_color_reset}")

def generate_outlier_detection_plot(sizes, output_image_file_base):
    plt.figure(figsize=(38, 20))
    plt.plot(sizes, color=py_color_blue)
    outliers = [i for i, x in enumerate(sizes) if (x > np.mean(sizes) + 3 * np.std(sizes))]
    plt.scatter(outliers, np.array(sizes)[outliers], color=py_color_red)
    plt.title('Outlier Detection Plot', fontsize=28)
    plt.xlabel('Index', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_outlier_detection_plot.png")
    print(f"{bash_color_light_green}Outlier detection plot generated successfully.{bash_color_reset}")

def generate_segmented_bar_chart(sizes, output_image_file_base):
    bins = [0, 1, 2, 3, 5, 10]
    hist, edges = np.histogram(sizes, bins=bins)
    plt.figure(figsize=(38, 20))
    plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")
    plt.title('Segmented Bar Chart of Block Sizes', fontsize=28)
    plt.xlabel('Block Size (MB)', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_segmented_bar_chart.png")
    print(f"{bash_color_light_green}Segmented bar chart generated successfully.{bash_color_reset}")

def generate_graphs_and_table(data, output_image_file_base, lower_height, upper_height):
    block_data = data["block_data"]
    total_blocks = len(block_data)

    categories = {
        "less_than_1MB": data.get("less_than_1MB", []),
        "1MB_to_2MB": data.get("1MB_to_2MB", []),
        "2MB_to_3MB": data.get("2MB_to_3MB", []),
        "3MB_to_5MB": data.get("3MB_to_5MB", []),
        "greater_than_5MB": data.get("greater_than_5MB", [])
    }

    total_blocks = sum(len(v) for v in categories.values())

    # Print table to console using tabulate
    headers = ["Block Size Range", "Count", "Percentage", "Average Size (MB)", "Min Size (MB)", "Max Size (MB)"]
    table = [
        ["Less than 1MB", len(categories['less_than_1MB']), f"{len(categories['less_than_1MB']) / total_blocks * 100:.2f}%", calculate_avg([b['size'] for b in categories['less_than_1MB']]), min([b['size'] for b in categories['less_than_1MB']], default=0), max([b['size'] for b in categories['less_than_1MB']], default=0)],
        ["1MB to 2MB", len(categories['1MB_to_2MB']), f"{len(categories['1MB_to_2MB']) / total_blocks * 100:.2f}%", calculate_avg([b['size'] for b in categories['1MB_to_2MB']]), min([b['size'] for b in categories['1MB_to_2MB']], default=0), max([b['size'] for b in categories['1MB_to_2MB']], default=0)],
        ["2MB to 3MB", len(categories['2MB_to_3MB']), f"{len(categories['2MB_to_3MB']) / total_blocks * 100:.2f}%", calculate_avg([b['size'] for b in categories['2MB_to_3MB']]), min([b['size'] for b in categories['2MB_to_3MB']], default=0), max([b['size'] for b in categories['2MB_to_3MB']], default=0)],
        ["3MB to 5MB", len(categories['3MB_to_5MB']), f"{len(categories['3MB_to_5MB']) / total_blocks * 100:.2f}%", calculate_avg([b['size'] for b in categories['3MB_to_5MB']]), min([b['size'] for b in categories['3MB_to_5MB']], default=0), max([b['size'] for b in categories['3MB_to_5MB']], default=0)],
        ["Greater than 5MB", len(categories['greater_than_5MB']), f"{len(categories['greater_than_5MB']) / total_blocks * 100:.2f}%", calculate_avg([b['size'] for b in categories['greater_than_5MB']]), min([b['size'] for b in categories['greater_than_5MB']], default=0), max([b['size'] for b in categories['greater_than_5MB']], default=0)]
    ]
    print(tabulate(table, headers=headers, tablefmt="pretty"))

    times = [datetime.fromisoformat(b['time']) for b in block_data]
    sizes = [b['size'] for b in block_data]
    colors = [
        py_color_green if size < 1 else
        py_color_yellow if size < 2 else
        py_color_orange if size < 3 else
        py_color_red if size < 5 else
        py_color_magenta
        for size in sizes
    ]

    def run_in_executor(func, *args):
        future = executor.submit(func, *args)
        return future

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            run_in_executor(generate_cumulative_sum_plot, times, sizes, output_image_file_base),
            run_in_executor(generate_rolling_average_plot, times, sizes, output_image_file_base),
            run_in_executor(generate_violin_plot, sizes, output_image_file_base),
            run_in_executor(generate_autocorrelation_plot, sizes, output_image_file_base),
            run_in_executor(generate_seasonal_decomposition_plot, times, sizes, output_image_file_base),
            run_in_executor(generate_lag_plot, sizes, output_image_file_base),
            run_in_executor(generate_heatmap_with_dimensions, times, sizes, output_image_file_base),
            run_in_executor(generate_network_graph, times, sizes, output_image_file_base),
            run_in_executor(generate_outlier_detection_plot, sizes, output_image_file_base),
            run_in_executor(generate_segmented_bar_chart, sizes, output_image_file_base)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"{bash_color_red}Generated an exception: {exc}{bash_color_reset}")

def main(json_workers, fetch_workers, lower_height, upper_height, endpoint_type, endpoint_urls, json_file=None):
    global executor
    endpoint_urls = endpoint_urls.split(',')
    endpoint = endpoint_urls[0]  # Use the first endpoint for now

    if json_workers <= 0:
        json_workers = max(1, os.cpu_count() // 2)
    if fetch_workers <= 0:
        fetch_workers = max(1, os.cpu_count() // 2)

    if json_workers > os.cpu_count():
        json_workers = os.cpu_count()
        print(f"{bash_color_yellow}json_workers set to {json_workers} due to CPU count limitation.{bash_color_reset}")

    if json_file:
        with open(json_file, 'r') as f:
            data = json.load(f)
        generate_graphs_and_table(data, f"{json_file}_output", lower_height, upper_height)
        return

    print(f"{bash_color_light_blue}Checking endpoints...{bash_color_reset}")
    valid_endpoints = [url for url in endpoint_urls if check_endpoint(endpoint_type, url)]
    if not valid_endpoints:
        print(f"{bash_color_red}No valid endpoints available.{bash_color_reset}")
        sys.exit(1)

    endpoint = valid_endpoints[0]

    lowest_height = find_lowest_height(endpoint_type, endpoint)

    if lowest_height > lower_height:
        print(f"{bash_color_yellow}Warning: The lowest available height is {lowest_height}. Adjusting lower height to {lowest_height}.{bash_color_reset}")
        lower_height = lowest_height

    total_blocks = upper_height - lower_height + 1
    print(f"{bash_color_teal}Total number of blocks to process: {total_blocks}{bash_color_reset}")

    start_script_time = time.time()

    categories = {
        "less_than_1MB": [],
        "1MB_to_2MB": [],
        "2MB_to_3MB": [],
        "3MB_to_5MB": [],
        "greater_than_5MB": []
    }

    with tqdm(total=total_blocks, desc="Fetching blocks", unit="block") as pbar:
        with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
            future_to_height = {executor.submit(process_block, height, endpoint_type, endpoint): height for height in range(lower_height, upper_height + 1)}

            completed = 0
            for future in as_completed(future_to_height):
                block = future.result()
                if block:
                    categorize_block(block, categories)
                pbar.update(1)
                completed += 1

    end_script_time = time.time()
    total_duration = end_script_time - start_script_time
    print(f"{bash_color_light_blue}Fetching completed in: {timedelta(seconds=int(total_duration))}{bash_color_reset}")

    block_data = [
        {
            "height": block[0],
            "size": block[1],
            "time": block[2].isoformat()
        }
        for block in [b for cat in categories.values() for b in cat]
    ]

    output_file = f"block_sizes_{lower_height}_to_{upper_height}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result = {
        "block_data": block_data,
        "less_than_1MB": {
            "count": len(categories["less_than_1MB"]),
            "avg_size_mb": calculate_avg([b["size"] for b in categories["less_than_1MB"]]),
            "min_size_mb": min([b["size"] for b in categories["less_than_1MB"]], default=0),
            "max_size_mb": max([b["size"] for b in categories["less_than_1MB"]], default=0),
            "percentage": len(categories["less_than_1MB"]) / total_blocks * 100
        },
        "1MB_to_2MB": {
            "count": len(categories["1MB_to_2MB"]),
            "avg_size_mb": calculate_avg([b["size"] for b in categories["1MB_to_2MB"]]),
            "min_size_mb": min([b["size"] for b in categories["1MB_to_2MB"]], default=0),
            "max_size_mb": max([b["size"] for b in categories["1MB_to_2MB"]], default=0),
            "percentage": len(categories["1MB_to_2MB"]) / total_blocks * 100
        },
        "2MB_to_3MB": {
            "count": len(categories["2MB_to_3MB"]),
            "avg_size_mb": calculate_avg([b["size"] for b in categories["2MB_to_3MB"]]),
            "min_size_mb": min([b["size"] for b in categories["2MB_to_3MB"]], default=0),
            "max_size_mb": max([b["size"] for b in categories["2MB_to_3MB"]], default=0),
            "percentage": len(categories["2MB_to_3MB"]) / total_blocks * 100
        },
        "3MB_to_5MB": {
            "count": len(categories["3MB_to_5MB"]),
            "avg_size_mb": calculate_avg([b["size"] for b in categories["3MB_to_5MB"]]),
            "min_size_mb": min([b["size"] for b in categories["3MB_to_5MB"]], default=0),
            "max_size_mb": max([b["size"] for b in categories["3MB_to_5MB"]], default=0),
            "percentage": len(categories["3MB_to_5MB"]) / total_blocks * 100
        },
        "greater_than_5MB": {
            "count": len(categories["greater_than_5MB"]),
            "avg_size_mb": calculate_avg([b["size"] for b in categories["greater_than_5MB"]]),
            "min_size_mb": min([b["size"] for b in categories["greater_than_5MB"]], default=0),
            "max_size_mb": max([b["size"] for b in categories["greater_than_5MB"]], default=0),
            "percentage": len(categories["greater_than_5MB"]) / total_blocks * 100
        },
        "start_script_time": start_script_time,
        "total_duration": time.time() - start_script_time,
        "lower_height": lower_height,
        "upper_height": upper_height
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    end_script_time = time.time()
    total_duration = end_script_time - start_script_time
    print(f"{bash_color_green}\nBlock sizes have been written to {output_file}{bash_color_reset}")
    print(f"{bash_color_light_blue}Script completed in: {timedelta(seconds=int(total_duration))}{bash_color_reset}")

    generate_graphs_and_table(result, output_file.replace(".json", ""), lower_height, upper_height)

if __name__ == "__main__":
    if len(sys.argv) not in {7, 8}:
        print(f"{bash_color_red}Usage: python blockbusteranalyzer.py <json_workers> <fetch_workers> <lower_height> <upper_height> <endpoint_type> <endpoint_urls> [json_file]{bash_color_reset}")
        sys.exit(1)

    json_workers = int(sys.argv[1])
    fetch_workers = int(sys.argv[2])
    lower_height = int(sys.argv[3])
    upper_height = int(sys.argv[4])
    endpoint_type = sys.argv[5]
    endpoint_urls = sys.argv[6]
    json_file = sys.argv[7] if len(sys.argv) == 8 else None

    main(json_workers, fetch_workers, lower_height, upper_height, endpoint_type, endpoint_urls, json_file)
