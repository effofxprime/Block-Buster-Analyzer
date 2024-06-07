#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-09 17:24:00 UTC
# @Description - A tool to analyze block sizes in a blockchain.

import requests
import requests_unixsocket
import json
import time
import sys
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, Style, init
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import signal
import threading

# Initialize colorama
init(autoreset=True)

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
    except requests.RequestException as e:
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
    block_size_mb = block_size / 1048576

    block_time = parse_timestamp(block_info['result']['block']['header']['time'])
    return (height, block_size_mb, block_time)

def signal_handler(sig, frame):
    print("\nProcess interrupted. Exiting gracefully...")
    shutdown_event.set()
    if executor:
        executor.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main(lower_height, upper_height, endpoint_type, endpoint_url):
    global executor
    print("\nChecking the specified starting block height...")

    # Health check
    retries = 3
    for attempt in range(retries):
        if check_endpoint(endpoint_type, endpoint_url):
            break
        else:
            print(f"RPC endpoint unreachable. Retrying {attempt + 1}/{retries}...")
            time.sleep(5)
    else:
        print("RPC endpoint unreachable after multiple attempts. Exiting.")
        sys.exit(1)

    block_info = fetch_block_info(endpoint_type, endpoint_url, lower_height)
    if block_info is None:
        print(f"Block height {lower_height} does not exist. Finding the earliest available block height...")
        lower_height = find_lowest_height(endpoint_type, endpoint_url)
        if lower_height is None:
            print("Failed to determine the earliest block height. Exiting.")
            sys.exit(1)
        print(f"Using earliest available block height: {lower_height}")

    if lower_height > upper_height:
        print(f"The specified lower height {lower_height} is greater than the specified upper height {upper_height}. Exiting.")
        sys.exit(1)

    print("\nFetching block information. This may take a while for large ranges. Please wait...")

    start_time = datetime.utcnow()
    current_date = start_time.strftime("%B %A %d, %Y %H:%M:%S UTC")
    output_file = f"block_sizes_{lower_height}_to_{upper_height}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    output_image_file_base = f"block_sizes_{lower_height}_to_{upper_height}_{start_time.strftime('%Y%m%d_%H%M%S')}"

    yellow_blocks = []
    red_blocks = []
    magenta_blocks = []
    block_data = []

    total_blocks = upper_height - lower_height + 1
    start_script_time = time.time()

    print("\n" + "="*40 + "\n")

    executor = ThreadPoolExecutor(max_workers=10)
    future_to_height = {executor.submit(process_block, height, endpoint_type, endpoint_url): height for height in range(lower_height, upper_height + 1)}

    try:
        for future in as_completed(future_to_height):
            if shutdown_event.is_set():
                break

            height = future_to_height[future]
            try:
                result = future.result()
                if result is None:
                    continue

                height, block_size_mb, block_time = result
                block_data.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})

                if block_size_mb > 5:
                    magenta_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})
                elif block_size_mb > 3:
                    red_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})
                elif block_size_mb > 1:
                    yellow_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})

            except Exception as e:
                print(f"Error processing block {height}: {e}")

            completed = height - lower_height + 1
            progress = (completed / total_blocks) * 100
            elapsed_time = time.time() - start_script_time
            estimated_total_time = elapsed_time / completed * total_blocks
            time_left = estimated_total_time - elapsed_time
            print(f"Progress: {progress:.2f}% ({completed}/{total_blocks}) - Estimated time left: {timedelta(seconds=int(time_left))}", end='\r')
    except KeyboardInterrupt:
        shutdown_event.set()
        if executor:
            executor.shutdown(wait=False)
        print("\nProcess interrupted. Exiting gracefully...")
        sys.exit(0)

    executor.shutdown(wait=True)

    result = {
        "connection_type": endpoint_type,
        "endpoint": endpoint_url,
        "run_time": current_date,
        "1MB_to_3MB": yellow_blocks,
        "3MB_to_5MB": red_blocks,
        "greater_than_5MB": magenta_blocks,
        "block_data": block_data,
        "stats": {
            "1MB_to_3MB": {
                "count": len(yellow_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in yellow_blocks])
            },
            "3MB_to_5MB": {
                "count": len(red_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in red_blocks])
            },
            "greater_than_5MB": {
                "count": len(magenta_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in magenta_blocks])
            }
        }
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    end_script_time = time.time()
    total_duration = end_script_time - start_script_time
    print(f"\nBlock sizes have been written to {output_file}")
    print(f"Script completed in: {timedelta(seconds=int(total_duration))}")

    print("\nNumber of blocks in each group:")

    table = [
        ["1MB to 3MB", len(yellow_blocks), f"{calculate_avg([b['size'] for b in yellow_blocks]):.2f}"],
        ["3MB to 5MB", len(red_blocks), f"{calculate_avg([b['size'] for b in red_blocks]):.2f}"],
        ["Greater than 5MB", len(magenta_blocks), f"{calculate_avg([b['size'] for b in magenta_blocks]):.2f}"]
    ]

    print(tabulate(table, headers=["Block Size Range", "Count", "Average Size (MB)"], tablefmt="pretty"))

    # Plotting the graphs
    if block_data:
        times = [datetime.fromisoformat(b['time']) for b in block_data]
        sizes = [b['size'] for b in block_data]
        colors = ['green' if size < 1 else 'yellow' if size < 3 else 'red' if size < 5 else 'magenta' for size in sizes]

        legend_patches = [
            mpatches.Patch(color='green', label='< 1MB'),
            mpatches.Patch(color='yellow', label='1MB to 3MB'),
            mpatches.Patch(color='red', label='3MB to 5MB'),
            mpatches.Patch(color='magenta', label='> 5MB')
        ]

        # Grouped bar chart
        plt.figure(figsize=(38, 20))  # Increase the figure size
        plt.bar(times, sizes, color=colors)
        plt.title('Block Size Over Time (Grouped Bar Chart)')
        plt.xlabel('Time')
        plt.ylabel('Block Size (MB)')
        plt.xticks(rotation=45)
        plt.legend(handles=legend_patches, loc='upper right')  # Move the legend to the top right
        plt.tight_layout()
        plt.savefig(f"{output_image_file_base}_bar_chart.png")

        # Scatter plot
        plt.figure(figsize=(38, 20))  # Increase the figure size
        plt.scatter(times, sizes, color=colors)
        plt.title('Block Size Over Time (Scatter Plot)')
        plt.xlabel('Time')
        plt.ylabel('Block Size (MB)')
        plt.xticks(rotation=45)
        plt.legend(handles=legend_patches, loc='upper right')  # Move the legend to the top right
        plt.tight_layout()
        plt.savefig(f"{output_image_file_base}_scatter_plot.png")

        # Histogram plot
        plt.figure(figsize=(38, 20))  # Increase the figure size
        plt.hist(sizes, bins=50, color='b', edgecolor='black')
        plt.title('Block Size Distribution (Histogram)')
        plt.xlabel('Block Size (MB)')
        plt.ylabel('Frequency')
        plt.legend(handles=legend_patches, loc='upper right')  # Move the legend to the top right
        plt.tight_layout()
        plt.savefig(f"{output_image_file_base}_histogram.png")
    else:
        print("No data to plot.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python blockbusteranalyzer.py <lower_height> <upper_height> <endpoint_type> <endpoint_url>")
        sys.exit(1)

    lower_height = int(sys.argv[1])
    upper_height = int(sys.argv[2])
    endpoint_type = sys.argv[3]
    endpoint_url = sys.argv[4]

    main(lower_height, upper_height, endpoint_type, endpoint_url)
