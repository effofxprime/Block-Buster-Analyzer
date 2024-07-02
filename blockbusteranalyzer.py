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
# @Last_Modified_Time - 2024-06-24 17:30:00 UTC
# @Version - 1.1.0
# @Description - This script analyzes block sizes in a blockchain and generates various visualizations.

# LOCKED - Only edit when we need to add or remove imports
import os
import sys
import json
import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timezone, date
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tabulate import tabulate
import re
import requests
import requests_unixsocket
from urllib.parse import quote_plus
from tqdm.asyncio import tqdm as tqdm_async
import logging
import aiohttp
from aiohttp.connector import UnixConnector
import aiofiles
import asyncio
import time
import warnings  # Added for capturing warnings

# LOCKED
# Set up logging configuration globally
log_file = None
start_time = datetime.now(timezone.utc)
file_timestamp = start_time.strftime('%Y%m%d_%H%M%S')

# LOCKED
class AsyncFileHandler(logging.Handler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        logging.Handler.__init__(self)
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.loop = asyncio.get_event_loop()

    async def aio_write(self, message):
        async with aiofiles.open(self.filename, mode=self.mode, encoding=self.encoding) as log_file:
            await log_file.write(message + '\n')

    def emit(self, record):
        try:
            msg = self.format(record)
            # Check if the event loop is closed before creating the task
            if not self.loop.is_closed():
                self.loop.create_task(self.aio_write(msg))
        except Exception:
            self.handleError(record)

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
# Add this function to set up logging and capture warnings
async def configure_logging():
    global log_file
    if log_file:
        async with aiofiles.open(log_file, 'a') as f:
            await f.write("Logging configured globally.\n")
    else:
        raise RuntimeError("log_file is not initialized.")
    
    async_handler = AsyncFileHandler(log_file)
    async_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    async_handler.setFormatter(formatter)
    logging.getLogger().handlers = [async_handler]
    warnings.simplefilter('always')  # Ensure all warnings are caught
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.DEBUG)


# LOCKED
async def log_handler(level, message):
    global log_file
        
    if log_file is None:
        raise RuntimeError("log_file is not initialized.")
    
    log_message = f"{datetime.now(timezone.utc)} - {level.upper()} - {message}"
    if level.lower() == 'error':
        logging.error(log_message)
    elif level.lower() == 'info':
        logging.info(log_message)
    elif level.lower() == 'warning':
        logging.warning(log_message)
    elif level.lower() == 'debug':
        logging.debug(log_message)

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
    except requests.RequestException as e:
        asyncio.run(log_handler('error', f"Error checking endpoint {endpoint_type} at {endpoint_url}: {e}"))
        return False

# LOCKED
async def fetch_block_info_aiohttp(session, endpoint_url, height):
    try:
        async with session.get(f"{endpoint_url}/block?height={height}") as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        await log_handler('error', f"Error fetching block {height} from {endpoint_url}: {e}")
        return None

# LOCKED
async def fetch_block_info_socket(session, endpoint_url, height):
    try:
        async with session.get(f"http://localhost/block?height={height}") as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        await log_handler('error', f"Error fetching block {height} from {endpoint_url}: {e}")
        return None

# LOCKED
async def get_progress_indicator(total, description):
    await log_handler('info', f"Creating progress indicator for {description} with total: {total}")
    return tqdm_async(total=total, desc=description, unit="block",
                      bar_format=f"{bash_color_light_blue}{{l_bar}}{{bar}} [Blocks: {{n}}/{{total}}, Elapsed: {{elapsed}}, Remaining: {{remaining}}, Speed: {{rate_fmt}}]{bash_color_reset}",
                      position=0, leave=False)

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
            asyncio.run(log_handler('error', f"HTTPError while finding the lowest height from {endpoint_url} using {endpoint_type}: {e}"))
    except requests.RequestException as e:
        asyncio.run(log_handler('error', f"RequestException while finding the lowest height from {endpoint_url} using {endpoint_type}: {e}"))
        return None
    except Exception as e:
        asyncio.run(log_handler('error', f"Catch all unknown error while finding the lowest height from {endpoint_url} using {endpoint_type}: {e}"))
        return None

    return 1

# LOCKED
def parse_timestamp(timestamp):
    try:
        if isinstance(timestamp, datetime):
            return timestamp
        if '.' in timestamp and timestamp.endswith('Z'):
            timestamp = timestamp.split('.')[0] + 'Z'
        formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        raise ValueError(f"time data '{timestamp}' does not match any known format")
    except ValueError as e:
        asyncio.run(log_handler('error', f"ValueError parsing timestamp: {e}"))
        return None
    except Exception as e:
        asyncio.run(log_handler('error', f"Unknown error parsing timestamp: {e}"))
        return None

# LOCKED
async def process_block(height, endpoint_type, endpoint_url, semaphore):
    backoff_factor = 1.5
    max_retries = 5
    attempt = 0
    while attempt < max_retries:
        async with semaphore:
            try:
                if endpoint_type == "tcp":
                    async with aiohttp.ClientSession() as session:
                        block_info = await fetch_block_info_aiohttp(session, endpoint_url, height)
                else:
                    connector = UnixConnector(path=endpoint_url)
                    async with aiohttp.ClientSession(connector=connector) as session:
                        block_info = await fetch_block_info_socket(session, endpoint_url, height)

                if block_info is None:
                    attempt += 1
                    await asyncio.sleep(backoff_factor ** attempt)
                    continue

                block_size = len(json.dumps(block_info))
                block_size_mb = block_size / 1048576
                block_time = block_info['result']['block']['header']['time']
                parsed_time = parse_timestamp(block_time)
                if parsed_time is None:
                    raise ValueError(f"Invalid time format for block {height}")
                return {
                    "height": height,
                    "size": block_size_mb,
                    "time": parsed_time.isoformat()
                }
            except Exception as e:
                attempt += 1
                error_message = f"Error processing block {height} from {endpoint_url} using {endpoint_type}: {e}. Attempt {attempt}"
                await log_handler('error', error_message)
                await asyncio.sleep(backoff_factor ** attempt)
    await log_handler('error', f"Max retries reached for block {height}. Skipping.")
    return None

# LOCKED
async def retry_failed_blocks(failed_heights, connection_type, endpoint_url, semaphore, tqdm_progress, lower_height, upper_height):
    retries = 0
    max_retries = 5
    backoff_factor = 1.5

    # Check JSON file for missing heights before retrying
    await log_handler('info', "Checking JSON file for missing heights before retrying.")
    existing_heights = set()
    async with semaphore, aiofiles.open(json_file_path, 'r') as f:
        raw_data = await f.read()
        data = json.loads(raw_data)
        existing_heights.update(block["height"] for block in data)
    
    missing_heights = [height for height in range(lower_height, upper_height + 1) if height not in existing_heights]

    if missing_heights:
        await log_handler('warning', f"Missing blocks detected: {missing_heights}. Adding to failed heights for retry.")
        failed_heights.extend(height for height in missing_heights if height not in failed_heights)
    else:
        await log_handler('info', "No missing blocks detected in the JSON file.")

    if not failed_heights:
        await log_handler('info', "No failed heights to retry. Exiting retry process.")
        return

    while failed_heights and retries < max_retries:
        retries += 1
        await log_handler('info', f"Retrying {len(failed_heights)} failed blocks. Attempt {retries}")
        retry_tasks = [process_block(height, connection_type, endpoint_url, semaphore) for height in failed_heights]
        new_failed_heights = []
        for future in asyncio.as_completed(retry_tasks):
            try:
                result = await future
                if result:
                    block = json_structure({"height": result["height"], "size": result["size"], "time": result["time"]})
                    block_data.append(block)
                    async with aiofiles.open(json_file_path, 'a') as f:
                        await f.write(', ' + json.dumps(block, default=default))
                    tqdm_progress.update(1)
                else:
                    new_failed_heights.append(result["height"])
            except Exception as e:
                new_failed_heights.append(result["height"])
                error_message = f"Error processing retry result: {e}"
                await log_handler('error', error_message)
        failed_heights = new_failed_heights
        if failed_heights:
            await asyncio.sleep(backoff_factor ** retries)
    if failed_heights:
        await log_handler('error', f"Max retries reached for some blocks. Heights: {failed_heights}")
    else:
        await log_handler('info', "All blocks fetched successfully.")

# LOCKED
# Signal handling improvements for graceful shutdown with async operations
async def shutdown():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    print(f"{bash_color_red}\nProcess interrupted. Exiting gracefully...{bash_color_reset}")
    total_tasks = len(tasks)
    print(f"{total_tasks} async operations left to shutdown", end='\r')
    for task in asyncio.as_completed(tasks):
        try:
            await asyncio.wait_for(task, timeout=1)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        total_tasks -= 1
        print(f"{total_tasks} async operations left to shutdown", end='\r')
    print(f"{bash_color_green}\nAll async operations have been shut down.{bash_color_reset}")
    sys.exit(0)

# LOCKED
def signal_handler(sig, frame):
    asyncio.create_task(shutdown())
    shutdown_event.set()

# LOCKED
def categorize_block(block, categories):
    try:
        size = float(block["size"])
    except ValueError:
        asyncio.run(log_handler('error', f"Error converting block size to float: {block['size']} for block {block['height']}"))
        return
    except Exception as e:
        asyncio.run(log_handler('error', f"Catch all unknown error converting block size to float: {e} for block {block['height']} with size {block['size']}"))
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
    scatter = ax.scatter(times, sizes, c=colors, s=30)
    ax.set_title(f'Block Size Over Time (Scatter Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=32)
    ax.set_xlabel('Time', fontsize=32)
    ax.set_ylabel('Block Size (MB)', fontsize=32)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelrotation=45, labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=10))  # Add this line to limit the number of ticks
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
    plt.close(fig)
    print(f"{bash_color_light_green}Scatter chart generated successfully.{bash_color_reset}")

def generate_enhanced_scatter_chart(times, sizes, colors, output_image_file_base, lower_height, upper_height):
    print(f"{bash_color_light_blue}Generating enhanced scatter chart...{bash_color_reset}")
    fig, ax = plt.subplots(figsize=(38, 20))
    scatter = ax.scatter(times, sizes, c=colors, s=30, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.set_title(f'Block Size Over Time (Enhanced Scatter Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=32)
    ax.set_xlabel('Time', fontsize=32)
    ax.set_ylabel('Block Size (MB)', fontsize=32)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelrotation=45, labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=10))  # Add this line to limit the number of ticks
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
    plt.close(fig)
    print(f"{bash_color_light_green}Enhanced scatter chart generated successfully.{bash_color_reset}")

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
    plt.close()
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

    times = []
    sizes = []
    colors = []

    for block in block_data:
        try:
            time = parse_timestamp(block["time"])
            if time is None:
                raise ValueError(f"Invalid time format for block {block['height']}")
            times.append(time)
            sizes.append(block["size"])
            colors.append(
                py_color_green if block["size"] < 1 else
                py_color_yellow if 1 <= block["size"] < 2 else
                py_color_orange if 2 <= block["size"] < 3 else
                py_color_red if 3 <= block["size"] < 5 else
                py_color_magenta
            )
        except ValueError as e:
            await log_handler('error', f"ValueError for block {block['height']}: {e}")

    generate_scatter_chart(times, sizes, colors, output_image_file_base, lower_height, upper_height)
    generate_enhanced_scatter_chart(times, sizes, colors, output_image_file_base, lower_height, upper_height)
    generate_segmented_bar_chart(times, sizes, output_image_file_base)

# LOCKED
def default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type not serializable")

# LOCKED
def json_structure(block_info):
    return {
        "height": block_info["height"],
        "size": float(block_info["size"]),
        "time": block_info["time"] if isinstance(block_info["time"], str) else block_info["time"].isoformat()
    }

# LOCKED
async def read_json_file(json_file_path, semaphore):
    await log_handler('info', "Attempting to open JSON file...")
    try:
        async with semaphore, aiofiles.open(json_file_path, 'r') as f:
            raw_data = await f.read()
            await log_handler('info', "Read JSON data from file")
    except FileNotFoundError as e:
        await log_handler('error', f"FileNotFoundError: {e}")
        return None
    except Exception as e:
        await log_handler('error', f"Error opening JSON file: {e}")
        return None

    await log_handler('info', "Attempting to parse JSON data...")
    try:
        data = json.loads(raw_data)
        await log_handler('info', f"JSON data parsed successfully, total records: {len(data)}")
        return data
    except json.JSONDecodeError as e:
        await log_handler('error', f"JSONDecodeError: {e}")
    except Exception as e:
        await log_handler('error', f"Unknown error parsing JSON file: {e}")
    return None

# LOCKED
async def main():
    global log_file, json_file_path, block_data
    global shutdown_event
    shutdown_event = asyncio.Event()

    # Validate the number of arguments
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        await log_handler('error', "Incorrect number of arguments.")
        print(f"Usage: {sys.argv[0]} <lower_height> <upper_height> <connection_type> <endpoint_url> [<json_file_path>]")
        sys.exit(1)

    lower_height = int(sys.argv[1])
    upper_height = int(sys.argv[2])
    connection_type = sys.argv[3]
    endpoint_url = sys.argv[4]

    start_time = datetime.now(timezone.utc)
    file_timestamp = start_time.strftime('%Y%m%d_%H%M%S')

    # Configure logging
    log_file = f"error_log_{lower_height}_to_{upper_height}_{file_timestamp}.log"
    await configure_logging()
    await log_handler('info', "Logging module has been configured.")

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGTERM, None)
    loop.add_signal_handler(signal.SIGTERM, signal_handler, signal.SIGTERM, None)
    await log_handler('info', "Signal handlers configured.")
    
    # Ensure the json_file_path follows the correct naming convention
    json_file_path = sys.argv[5] if len(sys.argv) == 6 else f"block_sizes_{lower_height}_to_{upper_height}_{file_timestamp}.json"

    # Resolve full path for json_file_path
    if not os.path.isabs(json_file_path):
        json_file_path = os.path.join(os.getcwd(), json_file_path)

    # Handle path for output image file base
    output_image_file_base = os.path.splitext(os.path.basename(json_file_path))[0]

    # If a JSON file is specified and exists, skip fetching and directly process the JSON file
    if len(sys.argv) == 6:
        await log_handler('info', f"JSON file specified: {json_file_path}")
        if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
            await log_handler('info', "Confirmed JSON file exists and is not empty.")
            semaphore = asyncio.Semaphore(50)
            try:
                await log_handler('info', "About to read JSON file...")
                data = await read_json_file(json_file_path, semaphore)
                await log_handler('info', f"Read JSON file successfully. Sample data: {data[:2]}")
            except Exception as e:
                await log_handler('error', f"Error reading JSON file: {e}")
                return
            if data is None:
                await log_handler('error', "No data found in JSON file. Exiting.")
                return

            await log_handler('info', "Structuring block data...")
            try:
                block_data = [
                    json_structure({
                        "height": block["height"],
                        "size": float(block["size"]),
                        "time": parse_timestamp(block["time"])
                    })
                    for block in data
                ]
                await log_handler('info', f"Block data structured successfully. Sample data: {block_data[:2]}")
            except KeyError as e:
                await log_handler('error', f"KeyError in block data structure: {e}")
                return
            except Exception as e:
                await log_handler('error', f"Unknown error structuring block data: {e}")
                return

            # Infer lower and upper height from JSON file name
            match = re.search(r"(\d+)_to_(\d+)", json_file_path)
            if match:
                lower_height = int(match.group(1))
                upper_height = int(match.group(2))
            await log_handler('info', f"Inferred heights: lower_height={lower_height}, upper_height={upper_height}")

            # Ensure block_data is not empty before generating graphs and table
            if block_data:
                await log_handler('info', "Block data is not empty. Generating graphs and table.")
                generate_graphs_and_table(block_data, output_image_file_base, lower_height, upper_height)
            else:
                await log_handler('error', "No block data found in the supplied JSON file.")
            return
        elif not os.path.exists(json_file_path):
            await log_handler('error', f"JSON file {json_file_path} does not exist. Exiting.")
            return

    # The rest of the code will only execute if the JSON file does not exist
    # Check endpoint availability
    retries = 3
    for attempt in range(retries):
        if check_endpoint(connection_type, endpoint_url):
            break
        else:
            await log_handler('warning', f"RPC endpoint unreachable. Retrying {attempt + 1}/{retries}...")
            await asyncio.sleep(5)
    else:
        await log_handler('error', "RPC endpoint unreachable after multiple attempts. Exiting.")
        print(f"{bash_color_red}RPC endpoint unreachable after multiple attempts. Exiting.{bash_color_reset}")
        sys.exit(1)

    if lower_height == 0:
        lowest_height = find_lowest_height(connection_type, endpoint_url)
        if lowest_height is None:
            await log_handler('error', "Failed to determine the earliest block height. Exiting.")
            print(f"{bash_color_red}Failed to determine the earliest block height. Exiting.{bash_color_reset}")
            sys.exit(1)
        await log_handler('info', f"Using earliest available block height: {lowest_height}")
        print(f"{bash_color_light_blue}Using earliest available block height: {lowest_height}{bash_color_reset}")
        lower_height = lowest_height

    if lower_height > upper_height:
        await log_handler('error', f"The specified lower height {lower_height} is greater than the specified upper height {upper_height}. Exiting.")
        print(f"{bash_color_red}The specified lower height {lower_height} is greater than the specified upper height {upper_height}. Exiting.{bash_color_reset}")
        sys.exit(1)

    json_file_path = f"{output_image_file_base}.json"
    block_data = []
    await log_handler('info', "Fetching block information. This may take a while for large ranges. Please wait.")
    print(f"{bash_color_light_blue}\nFetching block information. This may take a while for large ranges. Please wait...{bash_color_reset}")

    print(f"{bash_color_dark_grey}\n{'='*40}\n{bash_color_reset}")

    semaphore = asyncio.Semaphore(50)
    failed_heights = []

    heights = range(lower_height, upper_height + 1)
    tqdm_progress = await get_progress_indicator(len(heights), "Fetching Blocks")
    async with aiofiles.open(json_file_path, 'w') as f:
        await f.write('[')  # Start JSON array
        first_block = True
        tasks = [process_block(height, connection_type, endpoint_url, semaphore) for height in heights]
        for future in asyncio.as_completed(tasks):
            if shutdown_event.is_set():
                await log_handler('info', "Shutdown event detected. Exiting.")
                print(f"{bash_color_red}Shutdown event detected. Exiting...{bash_color_reset}")
                break
            try:
                result = await future
                if result:
                    block = json_structure({"height": result["height"], "size": result["size"], "time": result["time"]})
                    block_data.append(block)
                    if not first_block:
                        await f.write(', ')
                    await f.write(json.dumps(block, default=default))
                    first_block = False
                else:
                    failed_heights.append(result["height"])
            except Exception as e:
                failed_heights.append(result["height"])
                error_message = f"Error processing future result: {e}"
                await log_handler('error', error_message)

            tqdm_progress.update(1)
        await f.write(']')  # End JSON array

    # Retry fetching failed blocks
    # Check if there are failed heights before retrying
    if failed_heights:
        await retry_failed_blocks(failed_heights, connection_type, endpoint_url, semaphore, tqdm_progress, lower_height, upper_height)

    tqdm_progress.close()
    print("\n")

    if shutdown_event.is_set():
        await log_handler('info', "Shutdown event detected. Exiting.")
        print(f"{bash_color_red}Shutdown event detected. Exiting...{bash_color_reset}")
        sys.exit(0)

    end_time = datetime.now(timezone.utc)
    actual_time = end_time - start_time
    await log_handler('info', f"Fetching completed in {actual_time}. Saving data.")
    print(f"{bash_color_light_green}\nFetching completed in {actual_time}. Saving data...{bash_color_reset}")

    # Generate graphs and table
    generate_graphs_and_table(block_data, output_image_file_base, lower_height, upper_height)

if __name__ == "__main__":
    asyncio.run(main())
