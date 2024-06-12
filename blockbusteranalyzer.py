#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-12 03:00:00 UTC
# @Version - 1.0.4
# @Description - A tool to analyze block sizes in a blockchain.

import requests
import requests_unixsocket
import json
import time
import sys
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import signal
import threading
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px

# ANSI escape sequences for 256 colors (bash)
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
        block_info = response.json()
        block_size = len(json.dumps(block_info))
        block_time = block_info['result']['block']['header']['time']
        return {
            "height": height,
            "size": block_size / 1048576,  # Base 2: 1MB = 1,048,576 bytes
            "time": block_time
        }
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

    block_size_mb = block_info["size"]
    block_time = parse_timestamp(block_info['time'])
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
        [f"{bash_color_red}3MB to 5MB{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB'])}{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_red}{calculate_avg([b['size'] for b in categories['3MB_to_5MB']]):.2f}{bash_color_reset}", f"{bash_color_red}{min([b['size'] for b in categories['3MB_to_5MB']], default=0): .2f}{bash_color_reset}", f"{bash_color_red}{max([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_magenta}Greater than 5MB{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB'])}{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_magenta}{calculate_avg([b['size'] for b in categories['greater_than_5MB']]):.2f}{bash_color_reset}", f"{bash_color_magenta}{min([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_magenta}{max([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}"]
    ]
    
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table, headers)]
    table_str = "\n".join(
        " | ".join(f"{cell:>{width}}" for cell, width in zip(row, col_widths))
        for row in [headers] + table
    )
    table_str = table_str.replace("+", f"{bash_color_dark_grey}+{bash_color_reset}").replace("-", f"{bash_color_dark_grey}-{bash_color_reset}").replace("|", f"{bash_color_dark_grey}|{bash_color_reset}")
    print(table_str)

    times = [datetime.fromisoformat(b['time']) for b in block_data]
    sizes = [b['size'] for b in block_data]
    colors = [
        bash_color_green if size < 1 else
        bash_color_yellow if size < 2 else
        bash_color_orange if size < 3 else
        bash_color_red if size < 5 else
        bash_color_magenta
        for size in sizes
    ]

    legend_patches = [
        mpatches.Patch(color='green', label='< 1MB'),
        mpatches.Patch(color='yellow', label='1MB to 2MB'),
        mpatches.Patch(color='orange', label='2MB to 3MB'),
        mpatches.Patch(color='red', label='3MB to 5MB'),
        mpatches.Patch(color='magenta', label='> 5MB')
    ]

    # Scatter plot
    print(f"{bash_color_teal}Generating the scatter plot...{bash_color_reset}")
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
    ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_scatter_plot.png")
    print(f"{bash_color_light_green}Scatter plot generated successfully.{bash_color_reset}")

    # Histogram plot
    print(f"{bash_color_teal}Generating the histogram plot...{bash_color_reset}")
    fig, ax = plt.subplots(figsize=(38, 20))
    ax.hist(sizes, bins=50, color='b', edgecolor='black')
    ax.set_title(f'Block Size Distribution (Histogram)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    ax.set_xlabel('Block Size (MB)', fontsize=24)
    ax.set_ylabel('Frequency', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_histogram.png")
    print(f"{bash_color_light_green}Histogram plot generated successfully.{bash_color_reset}")

    # Interactive scatter plot
    print(f"{bash_color_teal}Generating the interactive scatter plot...{bash_color_reset}")
    df = pd.DataFrame({
        "Time": times,
        "Block Size (MB)": sizes,
        "Block Category": [
            '< 1MB' if size < 1 else
            '1MB to 2MB' if size < 2 else
            '2MB to 3MB' if size < 3 else
            '3MB to 5MB' if size < 5 else
            '> 5MB'
            for size in sizes
        ]
    })
    fig = px.scatter(df, x="Time", y="Block Size (MB)", color="Block Category", title="Block Size Over Time (Interactive Scatter Plot)")
    fig.write_html(f"{output_image_file_base}_interactive_scatter_plot.html")
    print(f"{bash_color_light_green}Interactive scatter plot generated successfully.{bash_color_reset}")

def print_table(headers, rows):
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
    print(f"{'|'.join(f' {header.ljust(width)} ' for header, width in zip(headers, col_widths))}")
    print(f"{'|'.join('-' * (width + 2) for width in col_widths)}")
    for row in rows:
        print(f"{'|'.join(f' {cell.ljust(width)} ' for cell, width in zip(row, col_widths))}")

def main(num_workers, lower_height, upper_height, endpoint_type, endpoint_urls, json_file=None):
    global executor
    endpoint_urls = endpoint_urls.split(',')
    endpoint = endpoint_urls[0]  # Use the first endpoint for now

    if json_file:
        with open(json_file, 'r') as f:
            data = json.load(f)
        if "lower_height" not in data or "upper_height" not in data:
            match = re.search(r'block_sizes_(\d+)_to_(\d+)_\d{8}_\d{6}\.json', json_file)
            if match:
                lower_height = int(match.group(1))
                upper_height = int(match.group(2))
            else:
                print(f"{bash_color_red}Error: The provided JSON file does not contain 'lower_height' or 'upper_height' keys.{bash_color_reset}")
                return
        else:
            lower_height = data["lower_height"]
            upper_height = data["upper_height"]
        generate_graphs_and_table(data, json_file.split('.json')[0], lower_height, upper_height)
        return

    print(f"{bash_color_light_blue}\nChecking the specified starting block height...{bash_color_reset}")

    retries = 3
    for attempt in range(retries):
        if check_endpoint(endpoint_type, endpoint_urls):
            break
        else:
            print(f"{bash_color_yellow}RPC endpoint unreachable. Retrying {attempt + 1}/{retries}...{bash_color_reset}")
            time.sleep(5)
    else:
        print(f"{bash_color_red}RPC endpoint unreachable after multiple attempts. Exiting.{bash_color_reset}")
        sys.exit(1)

    block_info = fetch_block_info(endpoint_type, endpoint_urls, lower_height)
    if block_info is None:
        print(f"{bash_color_yellow}Block height {lower_height} does not exist. Finding the earliest available block height...{bash_color_reset}")
        lower_height = find_lowest_height(endpoint_type, endpoint_urls)
        if lower_height is None:
            print(f"{bash_color_red}Failed to determine the earliest block height. Exiting.{bash_color_reset}")
            sys.exit(1)
        print(f"{bash_color_light_blue}Using earliest available block height: {lower_height}{bash_color_reset}")

    if lower_height > upper_height:
        print(f"{bash_color_red}The specified lower height {lower_height} is greater than the specified upper height {upper_height}. Exiting.{bash_color_reset}")
        sys.exit(1)

    print(f"{bash_color_light_blue}\nFetching block information. This may take a while for large ranges. Please wait...{bash_color_reset}")

    start_time = datetime.utcnow()
    current_date = start_time.strftime("%B %A %d, %Y %H:%M:%S UTC")
    output_file = f"block_sizes_{lower_height}_to_{upper_height}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    output_image_file_base = f"block_sizes_{lower_height}_to_{upper_height}_{start_time.strftime('%Y%m%d_%H%M%S')}"

    block_data = []

    total_blocks = upper_height - lower_height + 1
    start_script_time = time.time()

    print(f"{bash_color_dark_grey}\n{'='*40}\n{bash_color_reset}")

    executor = ThreadPoolExecutor(max_workers=num_workers)
    future_to_height = {executor.submit(process_block, height, endpoint_type, endpoint_urls): height for height in range(lower_height, upper_height + 1)}

    completed = 0
    try:
        for future in as_completed(future_to_height):
            if shutdown_event.is_set():
                break

            try:
                result = future.result()
                if result is None:
                    continue

                height, block_size_mb, block_time = result
                block_data.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})

            except Exception as e:
                print(f"{bash_color_red}Error processing block {future_to_height[future]}: {e}{bash_color_reset}")

            completed += 1
            progress = (completed / total_blocks) * 100
            elapsed_time = time.time() - start_script_time
            estimated_total_time = elapsed_time / completed * total_blocks
            time_left = estimated_total_time - elapsed_time
            print(f"{bash_color_light_blue}Progress: {progress:.2f}% ({completed}/{total_blocks}) - Estimated time left: {timedelta(seconds=int(time_left))}{bash_color_reset}", end='\r')
    except KeyboardInterrupt:
        shutdown_event.set()
        if executor:
            executor.shutdown(wait=False)
        print(f"{bash_color_red}\nProcess interrupted. Exiting gracefully...{bash_color_reset}")
        sys.exit(0)

    executor.shutdown(wait=True)

    categories = {
        "less_than_1MB": [],
        "1MB_to_2MB": [],
        "2MB_to_3MB": [],
        "3MB_to_5MB": [],
        "greater_than_5MB": []
    }
    for block in block_data:
        categorize_block(block, categories)

    result = {
        "connection_type": endpoint_type,
        "endpoint": endpoint_urls,
        "run_time": current_date,
        "less_than_1MB": categories["less_than_1MB"],
        "1MB_to_2MB": categories["1MB_to_2MB"],
        "2MB_to_3MB": categories["2MB_to_3MB"],
        "3MB_to_5MB": categories["3MB_to_5MB"],
        "greater_than_5MB": categories["greater_than_5MB"],
        "block_data": block_data,
        "stats": {
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
            }
        },
        "start_script_time": start_script_time,
        "total_duration": time.time() - start_script_time,
        "lower_height": lower_height,
        "upper_height": upper_height
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    generate_graphs_and_table(result, output_image_file_base, lower_height, upper_height)

if __name__ == "__main__":
    if len(sys.argv) not in {6, 7}:
        print(f"{bash_color_red}Usage: python blockbusteranalyzer.py <num_workers> <lower_height> <upper_height> <endpoint_type> <endpoint_urls> [json_file]{bash_color_reset}")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    lower_height = int(sys.argv[2])
    upper_height = int(sys.argv[3])
    endpoint_type = sys.argv[4]
    endpoint_urls = sys.argv[5]

    json_file = sys.argv[6] if len(sys.argv) == 7 else None

    main(num_workers, lower_height, upper_height, endpoint_type, endpoint_urls, json_file)
