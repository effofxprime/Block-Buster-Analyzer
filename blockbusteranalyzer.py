#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-15 17:00:00 UTC
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

def generate_histogram_plot(sizes, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.hist(sizes, bins=50, color=py_color_blue, edgecolor='black')
    ax.set_title(f'Block Size Distribution (Histogram)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Block Size (MB)', fontsize=24)
    ax.set_ylabel('Frequency', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_histogram.png")
    print(f"{bash_color_light_green}Histogram plot generated successfully.{bash_color_reset}")

def generate_box_plot(times, sizes, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    sns.boxplot(x=times, y=sizes, ax=ax)
    ax.set_title(f'Block Size Distribution Over Time (Box Plot)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Block Size (MB)', fontsize=24)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', labelrotation=45, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_box_plot.png")
    print(f"{bash_color_light_green}Box plot generated successfully.{bash_color_reset}")

def generate_heatmap(sizes, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    data_matrix = np.array([sizes]).T
    sns.heatmap(data_matrix, ax=ax, cmap="YlGnBu")
    ax.set_title(f'Block Size Heatmap\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Block Number', fontsize=24)
    ax.set_ylabel('Block Size (MB)', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_heatmap.png")
    print(f"{bash_color_light_green}Heatmap generated successfully.{bash_color_reset}")

# New chart types

def generate_cumulative_sum_plot(sizes, output_image_file_base, lower_height, upper_height):
    cumulative_sizes = np.cumsum(sizes)
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.plot(cumulative_sizes, color=py_color_blue)
    ax.set_title(f'Cumulative Sum of Block Sizes\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Block Number', fontsize=24)
    ax.set_ylabel('Cumulative Block Size (MB)', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_cumulative_sum_plot.png")
    print(f"{bash_color_light_green}Cumulative sum plot generated successfully.{bash_color_reset}")

def generate_rolling_average_plot(sizes, output_image_file_base, lower_height, upper_height, window=100):
    rolling_avg = pd.Series(sizes).rolling(window=window).mean()
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.plot(rolling_avg, color=py_color_blue)
    ax.set_title(f'Rolling Average of Block Sizes (Window={window})\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Block Number', fontsize=24)
    ax.set_ylabel('Rolling Average Block Size (MB)', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_rolling_average_plot.png")
    print(f"{bash_color_light_green}Rolling average plot generated successfully.{bash_color_reset}")

def generate_violin_plot(times, sizes, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    sns.violinplot(x=times, y=sizes, ax=ax)
    ax.set_title(f'Block Size Distribution Over Time (Violin Plot)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Block Size (MB)', fontsize=24)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', labelrotation=45, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_violin_plot.png")
    print(f"{bash_color_light_green}Violin plot generated successfully.{bash_color_reset}")

def generate_autocorrelation_plot(sizes, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    pd.plotting.autocorrelation_plot(pd.Series(sizes), ax=ax)
    ax.set_title(f'Autocorrelation of Block Sizes\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Lag', fontsize=24)
    ax.set_ylabel('Autocorrelation', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_autocorrelation_plot.png")
    print(f"{bash_color_light_green}Autocorrelation plot generated successfully.{bash_color_reset}")

def generate_seasonal_decomposition_plot(times, sizes, output_image_file_base, lower_height, upper_height):
    result = seasonal_decompose(pd.Series(sizes), model='additive', period=30)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(38, 40))
    result.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_seasonal_decomposition_plot.png")
    print(f"{bash_color_light_green}Seasonal decomposition plot generated successfully.{bash_color_reset}")

def generate_lag_plot(sizes, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    pd.plotting.lag_plot(pd.Series(sizes), ax=ax)
    ax.set_title(f'Lag Plot of Block Sizes\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Previous Block Size (MB)', fontsize=24)
    ax.set_ylabel('Current Block Size (MB)', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_lag_plot.png")
    print(f"{bash_color_light_green}Lag plot generated successfully.{bash_color_reset}")

def generate_outlier_detection_plot(sizes, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.boxplot(sizes, vert=False)
    ax.set_title(f'Outlier Detection in Block Sizes\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Block Size (MB)', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_outlier_detection_plot.png")
    print(f"{bash_color_light_green}Outlier detection plot generated successfully.{bash_color_reset}")

def generate_segmented_bar_chart(sizes, output_image_file_base, lower_height, upper_height):
    bins = [0, 1, 2, 3, 5, 15]  # Adjust bins as necessary
    labels = ['<1MB', '1MB-2MB', '2MB-3MB', '3MB-5MB', '>5MB']
    binned_sizes = pd.cut(sizes, bins=bins, labels=labels, right=False)
    size_counts = binned_sizes.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(38, 20))
    size_counts.plot(kind='bar', ax=ax, color=py_color_blue, edgecolor='black')
    ax.set_title(f'Block Size Distribution (Segmented Bar Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Block Size Range', fontsize=24)
    ax.set_ylabel('Frequency', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_segmented_bar_chart.png")
    print(f"{bash_color_light_green}Segmented bar chart generated successfully.{bash_color_reset}")

def generate_heatmap_with_additional_dimensions(data, output_image_file_base, lower_height, upper_height):
    fig, ax = plt.subplots(figsize=(38, 20))
    heatmap_data = pd.DataFrame(data)
    sns.heatmap(heatmap_data.corr(), ax=ax, annot=True, cmap="YlGnBu")
    ax.set_title(f'Heatmap of Block Sizes with Additional Dimensions\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_heatmap_with_additional_dimensions.png")
    print(f"{bash_color_light_green}Heatmap with additional dimensions generated successfully.{bash_color_reset}")

def generate_network_graph(data, output_image_file_base, lower_height, upper_height):
    import networkx as nx
    G = nx.Graph()
    for block in data:
        G.add_node(block["height"], size=block["size"])
    pos = nx.spring_layout(G)
    sizes = [G.nodes[node]['size']*100 for node in G.nodes]
    fig, ax = plt.subplots(figsize=(38, 20))
    nx.draw(G, pos, node_size=sizes, with_labels=True, ax=ax)
    ax.set_title(f'Network Graph of Block Relationships\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_network_graph.png")
    print(f"{bash_color_light_green}Network graph generated successfully.{bash_color_reset}")

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

    # Print table to console
    print(f"{bash_color_teal}\nNumber of blocks in each group for block heights {lower_height} to {upper_height}:{bash_color_reset}")
    headers = [f"{bash_color_teal}Block Size Range{bash_color_reset}", f"{bash_color_teal}Count{bash_color_reset}", f"{bash_color_teal}Percentage{bash_color_reset}", f"{bash_color_teal}Average Size (MB){bash_color_reset}", f"{bash_color_teal}Min Size (MB){bash_color_reset}", f"{bash_color_teal}Max Size (MB){bash_color_reset}"]
    table = [
        [f"{bash_color_green}Less than 1MB{bash_color_reset}", f"{bash_color_green}{len(categories['less_than_1MB'])}{bash_color_reset}", f"{bash_color_green}{len(categories['less_than_1MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_green}{calculate_avg([b['size'] for b in categories['less_than_1MB']]):.2f}{bash_color_reset}", f"{bash_color_green}{min([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_green}{max([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_yellow}1MB to 2MB{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB'])}{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_yellow}{calculate_avg([b['size'] for b in categories['1MB_to_2MB']]):.2f}{bash_color_reset}", f"{bash_color_yellow}{min([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_yellow}{max([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_orange}2MB to 3MB{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB'])}{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_orange}{calculate_avg([b['size'] for b in categories['2MB_to_3MB']]):.2f}{bash_color_reset}", f"{bash_color_orange}{min([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_orange}{max([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_red}3MB to 5MB{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB'])}{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_red}{calculate_avg([b['size'] for b in categories['3MB_to_5MB']]):.2f}{bash_color_reset}", f"{bash_color_red}{min([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_red}{max([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_magenta}Greater than 5MB{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB'])}{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_magenta}{calculate_avg([b['size'] for b in categories['greater_than_5MB']]): .2f}{bash_color_reset}", f"{bash_color_magenta}{min([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_magenta}{max([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}"]
    ]
    print_table(headers, table)

    table_str = tabulate(table, headers=headers, tablefmt="pretty")
    table_str = table_str.replace("+", f"{bash_color_dark_grey}+{bash_color_reset}").replace("-", f"{bash_color_dark_grey}-{bash_color_reset}").replace("|", f"{bash_color_dark_grey}|{bash_color_reset}")
    print(table_str)

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

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            run_in_executor(generate_scatter_plot, times, sizes, colors, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_enhanced_scatter_plot, times, sizes, colors, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_histogram_plot, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_box_plot, times, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_heatmap, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_cumulative_sum_plot, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_rolling_average_plot, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_violin_plot, times, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_autocorrelation_plot, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_lag_plot, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_outlier_detection_plot, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_segmented_bar_chart, sizes, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_heatmap_with_additional_dimensions, block_data, output_image_file_base, lower_height, upper_height),
            run_in_executor(generate_network_graph, block_data, output_image_file_base, lower_height, upper_height)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"{bash_color_red}Generated an exception: {exc}{bash_color_reset}")

def print_table(headers, rows):
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
    separator = f"{bash_color_dark_grey}|{bash_color_reset}".join(['-' * (width + 2) for width in col_widths])
    print(f"{'|'.join(f' {header.ljust(width)} ' for header, width in zip(headers, col_widths))}")
    print(separator)
    for row in rows:
        print(f"{'|'.join(f' {cell.ljust(width)} ' for cell, width in zip(row, col_widths))}")

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
            "percentage": len(categories["3MB_to_5MB"]) / total blocks * 100
        },
        "greater_than 5MB": {
            "count": len(categories["greater_than_5MB"]),
            "avg_size_mb": calculate_avg([b["size"] for b in categories["greater_than_5MB"]]),
            "min_size_mb": min([b["size"] for b in categories["greater_than_5MB"]], default=0),
            "max_size_mb": max([b["size"] for b in categories["greater_than_5MB"]], default=0),
            "percentage": len(categories["greater_than_5MB"]) / total blocks * 100
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

