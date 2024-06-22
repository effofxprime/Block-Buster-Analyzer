#!/usr/bin/env python3
# LOCKED - Except @Last_Modified_Time which needs the date and time to be updated each edit based on UTC-0 current time and date.
# This is Metadata containing information about the script and author.
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-22 15:19:00 UTC
# @Version - 1.0.26
# @Description - This script analyzes block sizes in a blockchain and generates various visualizations.

# LOCKED - Only edit when we need to add or remove imports
import os
import sys
import json
import signal
import matplotlib.pyplot as plt
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, date  # Added date import
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
import re
import time as tm
import requests
import requests_unixsocket
from urllib.parse import quote_plus
from tqdm import tqdm, trange
from http.client import IncompleteRead
import logging
import psutil  # For system load monitoring
import aiohttp
import asyncio

# LOCKED
# Define colors for console output
bash_color_reset = "\033[0m"
bash_color_red = "\033[91m"
bash_color_green = "\033[92m"
bash_color_yellow = "\033[93m"
bash_color_orange = "\033[33m"
bash_color_magenta = "\033[38;5;13m"
bash_color_blue = "\033[34m"
bash_color_light_blue = "\033[38;5;123m"
bash_color_teal = "\033[36m"
bash_color_light_green = "\033[92m"
bash_color_dark_grey = "\033[38;5;245m"

# LOCKED
# Define colors for charts
py_color_green = "green"
py_color_yellow = "yellow"
py_color_orange = "orange"
py_color_red = "red"
py_color_magenta = "magenta"
py_color_blue = "blue"
py_color_light_blue = "lightblue"
py_color_teal = "teal"
py_color_light_green = "lightgreen"
py_color_dark_grey = "darkgrey"

# LOCKED
def calculate_avg(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

# LOCKED
def check_endpoint(endpoint_type, endpoint_url):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            encoded_url = f"http+unix://{quote_plus(endpoint_url)}/health"
            response = session.get(encoded_url, timeout=3)
        else:
            response = requests.get(f"{endpoint_url}/health", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False

# LOCKED
async def fetch_block_info_aiohttp(session, endpoint_url, height):
    backoff_factor = 1.5
    attempt = 0
    while True:
        try:
            async with session.get(f"{endpoint_url}/block?height={height}") as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError) as e:
            attempt += 1
            error_message = f"Error fetching block {height} from {endpoint_url}: {e}. Attempt {attempt}. Retrying in {backoff_factor ** attempt} seconds."
            logging.error(error_message)
            await asyncio.sleep(backoff_factor ** attempt)

def fetch_block_info_socket(endpoint_url, height):
    backoff_factor = 1.5
    attempt = 0
    while True:
        try:
            encoded_url = f"http+unix://{quote_plus(endpoint_url)}/block?height={height}"
            response = requests_unixsocket.Session().get(encoded_url, timeout=3)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException) as e:
            attempt += 1
            error_message = f"Error fetching block {height} from {endpoint_url}: {e}. Attempt {attempt}. Retrying in {backoff_factor ** attempt} seconds."
            logging.error(error_message)
            with open(log_file, 'a') as log:
                log.write(f"{datetime.now(timezone.utc)} - ERROR - {error_message}\n")
            tm.sleep(backoff_factor ** attempt)

async def fetch_all_blocks(endpoint_type, endpoint_url, heights):
    if endpoint_type == "tcp":
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_block_info_aiohttp(session, endpoint_url, height) for height in heights]
            return await asyncio.gather(*tasks)
    else:
        with requests_unixsocket.Session() as session:
            results = [fetch_block_info_socket(endpoint_url, height) for height in heights]
            return results

# LOCKED
def find_lowest_height(endpoint_type, endpoint_url):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            encoded_url = f"http+unix://{quote_plus(endpoint_url)}/block?height=1"
            response = session.get(encoded_url, timeout=3)
        else:
            response = requests.get(f"{endpoint_url}/block?height=1", timeout=3)
            response.raise_for_status()
        block_info = response.json()
        if 'error' in block_info and 'data' in block_info['error']:
            data_message = block_info['error']['data']
            print(f"Data message: {data_message}")
            if "lowest height is" in data_message:
                return int(data_message.split("lowest height is")[1].strip())
    except requests.HTTPError as e:
        if e.response.status_code == 500:
            error_response = e.response.json()
            if 'error' in error_response and 'data' in error_response['error']:
                data_message = error_response['error']['data']
                print(f"Data message: {data_message}")
                if "lowest height is" in data_message:
                    return int(data_message.split("lowest height is")[1].strip())
        else:
            logging.error(f"HTTPError while finding the lowest height from {endpoint_url} using {endpoint_type}: {e}")
    except requests.RequestException as e:
        logging.error(f"RequestException while finding the lowest height from {endpoint_url} using {endpoint_type}: {e}")
        return None

    return 1

# LOCKED
def parse_timestamp(timestamp):
    try:
        if isinstance(timestamp, datetime):
            return timestamp
        if '.' in timestamp:
            timestamp = timestamp.split('.')[0] + 'Z'
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError(f"time data '{timestamp}' does not match any known format")

# LOCKED
def process_block(height, endpoint_type, endpoint_url):
    if shutdown_event.is_set():
        return None
    try:
        if endpoint_type == "tcp":
            async def fetch():
                async with aiohttp.ClientSession() as session:
                    return await fetch_block_info_aiohttp(session, endpoint_url, height)
            block_info = asyncio.run(fetch())
        else:
            block_info = fetch_block_info_socket(endpoint_url, height)
        if block_info is None:
            return None

        block_size = len(json.dumps(block_info))
        block_size_mb = block_size / 1048576
        block_time = parse_timestamp(block_info['result']['block']['header']['time'])
        return (height, block_size_mb, block_time)
    except Exception as e:
        error_message = f"Error processing block {height} from {endpoint_url} using {endpoint_type}: {e}"
        logging.error(error_message)
        with open(log_file, 'a') as log:
            log.write(f"{datetime.now(timezone.utc)} - ERROR - {error_message}\n")
        return None

# LOCKED
def signal_handler(sig, frame):
    logging.info(f"Signal {sig} received. Shutting down.")
    print(f"{bash_color_red}\nProcess interrupted. Exiting gracefully...{bash_color_reset}")
    shutdown_event.set()
    tasks = asyncio.all_tasks()
    for task in tasks:
        task.cancel()
    sys.exit(0)

# LOCKED
def categorize_block(block, categories):
    try:
        size = float(block["size"])
    except ValueError:
        logging.error(f"Error converting block size to float: {block['size']} for block {block['height']}")
        return
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

# LOCKED - All chart functions are locked
# Chart generation functions
def generate_scatter_chart(times, sizes, colors, output_image_file_base, lower_height, upper_height):
    print(f"{bash_color_light_blue}Generating scatter chart...{bash_color_reset}")
    fig, ax = plt.subplots(figsize=(38, 20))
    scatter = ax.scatter(times, sizes, c=colors, s=30)  # Increased dot size
    ax.set_title(f'Block Size Over Time (Scatter Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=32)
    ax.set_xlabel('Time', fontsize=32)
    ax.set_ylabel('Block Size (MB)', fontsize=32)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelrotation=45, labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_green, markersize=10, label='< 1MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_yellow, markersize=10, label='1MB to 2MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_orange, markersize=10, label='2MB to 3MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_red, markersize=10, label='3MB to 5MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_magenta, markersize=10, label='> 5MB')
    ]
    ax.legend(handles=legend_patches, fontsize=32)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_scatter_chart.png")
    print(f"{bash_color_light_green}Scatter chart generated successfully.{bash_color_reset}")

def generate_enhanced_scatter_chart(times, sizes, colors, output_image_file_base, lower_height, upper_height):
    print(f"{bash_color_light_blue}Generating enhanced scatter chart...{bash_color_reset}")
    fig, ax = plt.subplots(figsize=(38, 20))
    scatter = ax.scatter(times, sizes, c=colors, s=30, alpha=0.6, edgecolors='w', linewidth=0.5)  # Increased dot size
    ax.set_title(f'Block Size Over Time (Enhanced Scatter Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=32)
    ax.set_xlabel('Time', fontsize=32)
    ax.set_ylabel('Block Size (MB)', fontsize=32)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelrotation=45, labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_green, markersize=10, label='< 1MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_yellow, markersize=10, label='1MB to 2MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_orange, markersize=10, label='2MB to 3MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_red, markersize=10, label='3MB to 5MB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=py_color_magenta, markersize=10, label='> 5MB')
    ]
    ax.legend(handles=legend_patches, fontsize=32)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_enhanced_scatter_chart.png")
    print(f"{bash_color_light_green}Enhanced scatter chart generated successfully.{bash_color_reset}")

def generate_heatmap_with_additional_dimensions(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating heatmap with additional dimensions...{bash_color_reset}")
    data = pd.DataFrame({'times': pd.to_datetime(times), 'sizes': sizes})
    data["hour"] = data["times"].dt.hour
    data["day_of_week"] = data["times"].dt.dayofweek
    data["day_of_week_name"] = data["times"].dt.day_name()
    heatmap_data = pd.pivot_table(data, values="sizes", index="hour", columns="day_of_week_name", aggfunc=np.mean)
    plt.figure(figsize=(38, 20))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", annot_kws={"size": 32})  # Increased font size for annotations
    plt.title('Heatmap of Block Sizes by Hour and Day of Week', fontsize=32)
    plt.xlabel('Day of Week', fontsize=32)
    plt.ylabel('Hour of Day', fontsize=32)
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_heatmap_with_dimensions.png")
    print(f"{bash_color_light_green}Heatmap with additional dimensions generated successfully.{bash_color_reset}")

def generate_segmented_bar_chart(times, sizes, output_image_file_base):
    print(f"{bash_color_light_blue}Generating segmented bar chart...{bash_color_reset}")
    data = pd.DataFrame({"times": times, "sizes": sizes})
    data["size_range"] = pd.cut(data["sizes"], bins=[0, 1, 2, 3, 5, np.inf], labels=["<1MB", "1MB-2MB", "2MB-3MB", "3MB-5MB", ">5MB"])
    size_ranges = data["size_range"].value_counts().sort_index()
    plt.figure(figsize=(38, 20))
    bars = size_ranges.plot(kind="bar", color=[py_color_green, py_color_yellow, py_color_orange, py_color_red, py_color_magenta], log=True)
    plt.title('Segmented Bar Chart of Block Sizes', fontsize=32)
    plt.xlabel('Block Size Range', fontsize=32)
    plt.ylabel('Count', fontsize=32)
    plt.xticks(rotation=0, fontsize=32)
    plt.yticks(fontsize=32)
    for bar in bars.patches:
        bars.annotate(f'{int(bar.get_height())}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', fontsize=32)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_segmented_bar_chart.png")
    print(f"{bash_color_light_green}Segmented bar chart generated successfully.{bash_color_reset}")

# LOCKED
def generate_graphs_and_table(block_data, output_image_file_base, lower_height, upper_height):
    categories = {
        "less_than_1MB": [],
        "1MB_to_2MB": [],
        "2MB_to_3MB": [],
        "3MB_to_5MB": [],
        "greater_than_5MB": []
    }

    for block in block_data:
        categorize_block(block, categories)

    total_blocks = len(block_data)
    table = [
        [f"{bash_color_light_blue}Category{bash_color_reset}", f"{bash_color_light_blue}Count{bash_color_reset}", f"{bash_color_light_blue}Percentage{bash_color_reset}", f"{bash_color_light_blue}Average Size (MB){bash_color_reset}", f"{bash_color_light_blue}Min Size (MB){bash_color_reset}", f"{bash_color_light_blue}Max Size (MB){bash_color_reset}"],
        [f"{bash_color_green}Less than 1MB{bash_color_reset}", f"{bash_color_green}{len(categories['less_than_1MB']):,}{bash_color_reset}", f"{bash_color_green}{len(categories['less_than_1MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_green}{calculate_avg([b['size'] for b in categories['less_than_1MB']]):.2f}{bash_color_reset}", f"{bash_color_green}{min([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_green}{max([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_yellow}1MB to 2MB{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB']):,}{bash_color_reset}", f"{bash_color_yellow}{len(categories['1MB_to_2MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_yellow}{calculate_avg([b['size'] for b in categories['1MB_to_2MB']]):.2f}{bash_color_reset}", f"{bash_color_yellow}{min([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_yellow}{max([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_orange}2MB to 3MB{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB']):,}{bash_color_reset}", f"{bash_color_orange}{len(categories['2MB_to_3MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_orange}{calculate_avg([b['size'] for b in categories['2MB_to_3MB']]):.2f}{bash_color_reset}", f"{bash_color_orange}{min([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_orange}{max([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_red}3MB to 5MB{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']):,}{bash_color_reset}", f"{bash_color_red}{len(categories['3MB_to_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_red}{calculate_avg([b['size'] for b in categories['3MB_to_5MB']]):.2f}{bash_color_reset}", f"{bash_color_red}{min([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_red}{max([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{bash_color_reset}"],
        [f"{bash_color_magenta}Greater than 5MB{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']):,}{bash_color_reset}", f"{bash_color_magenta}{len(categories['greater_than_5MB']) / total_blocks * 100:.2f}%{bash_color_reset}", f"{bash_color_magenta}{calculate_avg([b['size'] for b in categories['greater_than_5MB']]):.2f}{bash_color_reset}", f"{bash_color_magenta}{min([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}", f"{bash_color_magenta}{max([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{bash_color_reset}"]
    ]

    print(tabulate(table, headers="firstrow", tablefmt="grid"))

    times = [block["time"] for block in block_data]
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
    generate_scatter_chart(times, sizes, colors, output_image_file_base, lower_height, upper_height)
    generate_enhanced_scatter_chart(times, sizes, colors, output_image_file_base, lower_height, upper_height)

    # Generate remaining charts
    generate_heatmap_with_additional_dimensions(times, sizes, output_image_file_base)
    generate_segmented_bar_chart(times, sizes, output_image_file_base)

# LOCKED
def determine_optimal_workers():
    cpu_count = os.cpu_count()
    system_load = psutil.getloadavg()[0]  # 1 minute system load average
    optimal_fetch_workers = max(1, min(cpu_count * 10, int(cpu_count / (system_load + 0.5))))
    optimal_json_workers = cpu_count  # Assuming reading JSON is less intensive
    return optimal_fetch_workers, optimal_json_workers

# LOCKED
def save_data_incrementally(block_data, json_file_path):
    with open(json_file_path, 'w') as f:
        for block in block_data:
            json.dump(block, f, default=default)
            f.write('\n')

# LOCKED
def default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type not serializable")

def main():
    global shutdown_event, executor, log_file, json_file_path  # Add log_file and json_file_path as global variables
    shutdown_event = threading.Event()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logging.info("Signal handlers configured.")

    if len(sys.argv) < 5 or len(sys.argv) > 6:
        logging.error("Incorrect number of arguments.")
        print(f"Usage: {sys.argv[0]} <lower_height> <upper_height> <connection_type> <endpoint_url> [<json_file_path>]")
        sys.exit(1)

    lower_height = int(sys.argv[1])
    upper_height = int(sys.argv[2])
    connection_type = sys.argv[3]
    endpoint_url = sys.argv[4]

    start_time = datetime.now(timezone.utc)
    file_timestamp = start_time.strftime('%Y%m%d_%H%M%S')

    # Ensure the json_file_path and log_file follow the correct naming convention
    json_file_path = sys.argv[5] if len(sys.argv) == 6 else f"block_sizes_{lower_height}_to_{upper_height}_{file_timestamp}.json"
    output_image_file_base = os.path.splitext(json_file_path)[0]
    log_file = f"error_log_{lower_height}_to_{upper_height}_{file_timestamp}.log"

    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]

    # Calculate optimal workers
    fetch_workers, json_workers = determine_optimal_workers()
    print(f"{bash_color_light_blue}Optimal fetch workers: {fetch_workers}, Optimal JSON workers: {json_workers}{bash_color_reset}")

    # If a JSON file is specified, skip fetching and directly process the JSON file
    if json_file_path and os.path.exists(json_file_path):
        try:
            with open(json_file_path) as f:
                data = json.load(f)
        
            # Convert sizes to float and times to datetime
            for block in data["block_data"]:
                block["size"] = float(block["size"])
                block["time"] = parse_timestamp(block["time"])

            # Infer lower and upper height from JSON file name
            match = re.search(r"(\d+)_to_(\d+)", json_file_path)
            if match:
                lower_height = int(match.group(1))
                upper_height = int(match.group(2))

            generate_graphs_and_table(data["block_data"], output_image_file_base, lower_height, upper_height)
        except Exception as e:
            logging.error(f"Error processing JSON file: {e}")
        return

    # Check endpoint availability
    retries = 3
    for attempt in range(retries):
        if check_endpoint(connection_type, endpoint_url):
            break
        else:
            logging.warning(f"RPC endpoint unreachable. Retrying {attempt + 1}/{retries}...")
            tm.sleep(5)
    else:
        logging.error("RPC endpoint unreachable after multiple attempts. Exiting.")
        print(f"{bash_color_red}RPC endpoint unreachable after multiple attempts. Exiting.{bash_color_reset}")
        sys.exit(1)

    # Find the lowest available height if necessary
    if lower_height == 0:
        lowest_height = find_lowest_height(connection_type, endpoint_url)
        if lowest_height is None:
            logging.error("Failed to determine the earliest block height. Exiting.")
            print(f"{bash_color_red}Failed to determine the earliest block height. Exiting.{bash_color_reset}")
            sys.exit(1)
        logging.info(f"Using earliest available block height: {lowest_height}")
        print(f"{bash_color_light_blue}Using earliest available block height: {lowest_height}{bash_color_reset}")
        lower_height = lowest_height

    if lower_height > upper_height:
        logging.error(f"The specified lower height {lower_height} is greater than the specified upper height {upper_height}. Exiting.")
        print(f"{bash_color_red}The specified lower height {lower_height} is greater than the specified upper height {upper_height}. Exiting.{bash_color_reset}")
        sys.exit(1)

    json_file_path = f"{output_image_file_base}.json"
    with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
        block_data = []
        logging.info("Fetching block information. This may take a while for large ranges. Please wait.")
        print(f"{bash_color_light_blue}\nFetching block information. This may take a while for large ranges. Please wait...{bash_color_reset}")

        start_script_time = tm.time()
        total_blocks = upper_height - lower_height + 1

        print(f"{bash_color_dark_grey}\n{'='*40}\n{bash_color_reset}")

        heights = range(lower_height, upper_height + 1)
        futures = [executor.submit(process_block, height, connection_type, endpoint_url) for height in heights]

        tqdm_progress = tqdm(
            total=len(futures), 
            desc="Fetching Blocks", 
            unit="block", 
            bar_format=(
                f"{bash_color_light_blue}{{l_bar}}{{bar}} [Blocks: {{n}}/{{total}}, "
                f"Elapsed: {{elapsed}}, Remaining: {{remaining}}, Speed: {{rate:.2f}} blocks/s]{bash_color_reset}"
            )
        )
        with open(json_file_path, 'w') as f:
            for future in as_completed(futures):
                if shutdown_event.is_set():
                    logging.info("Shutdown event detected. Exiting.")
                    print(f"{bash_color_red}Shutdown event detected. Exiting...{bash_color_reset}")
                    break
                try:
                    result = future.result()
                    if result:
                        block = {"height": result[0], "size": result[1], "time": result[2]}
                        block_data.append(block)
                        f.write(json.dumps(block, default=default) + '\n')
                except Exception as e:
                    error_message = f"Error processing future result: {e}"
                    logging.error(error_message)
                    with open(log_file, 'a') as log:
                        log.write(f"{datetime.now(timezone.utc)} - ERROR - {error_message}\n")

                tqdm_progress.update(1)

    tqdm_progress.close()  # Ensure the progress bar is closed properly
    print("\n")

    if shutdown_event.is_set():
        logging.info("Shutdown event detected. Exiting.")
        print(f"{bash_color_red}Shutdown event detected. Exiting...{bash_color_reset}")
        sys.exit(0)

    end_time = datetime.now(timezone.utc)
    actual_time = end_time - start_time
    logging.info(f"Fetching completed in {actual_time}. Saving data.")
    print(f"{bash_color_light_green}\nFetching completed in {actual_time}. Saving data...{bash_color_reset}")

    # Save data incrementally
    save_data_incrementally(block_data, json_file_path)

    # Generate graphs and table
    generate_graphs_and_table(block_data, output_image_file_base, lower_height, upper_height)

if __name__ == "__main__":
    main()
