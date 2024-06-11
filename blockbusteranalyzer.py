#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-11 20:23:00 UTC
# @Version - 1.0.3
# @Description - A tool to analyze block sizes in a blockchain.

import requests
import requests_unixsocket
import json
import time
import sys
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import signal
import threading

# ANSI escape sequences for 256 colors
color_green = "\033[38;5;10m"  # Green
color_yellow = "\033[38;5;11m"  # Yellow
color_orange = "\033[38;5;214m"  # Orange
color_red = "\033[38;5;9m"  # Red
color_magenta = "\033[38;5;13m"  # Magenta
color_light_blue = "\033[38;5;123m"  # Light Blue
color_dark_grey = "\033[38;5;245m"  # Dark Grey
color_light_green = "\033[38;5;121m"  # Light Green
color_teal = "\033[38;5;74m"  # Teal
color_reset = "\033[0m"  # Reset

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
    except requests.RequestException:
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
    print(f"{color_red}\nProcess interrupted. Exiting gracefully...{color_reset}")
    shutdown_event.set()
    if executor:
        executor.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def generate_graphs_and_table(data, output_image_file_base, lower_height, upper_height):
    block_data = data["block_data"]
    total_blocks = len(block_data)

    green_blocks = data["less_than_1MB"]
    yellow_blocks = data["1MB_to_2MB"]
    orange_blocks = data["2MB_to_3MB"]
    red_blocks = data["3MB_to_5MB"]
    magenta_blocks = data["greater_than_5MB"]

    print(f"{color_light_green}\nNumber of blocks in each group for block heights {lower_height} to {upper_height}:{color_reset}")
    headers = [f"{color_light_blue}Block Size Range{color_reset}", f"{color_light_blue}Count{color_reset}", f"{color_light_blue}Percentage{color_reset}", f"{color_light_blue}Average Size (MB){color_reset}", f"{color_light_blue}Min Size (MB){color_reset}", f"{color_light_blue}Max Size (MB){color_reset}"]
    table = [
        [f"{color_green}Less than 1MB{color_reset}", f"{color_green}{len(green_blocks)}{color_reset}", f"{color_green}{len(green_blocks) / total_blocks * 100:.2f}%{color_reset}", f"{color_green}{calculate_avg([b['size'] for b in green_blocks]):.2f}{color_reset}", f"{color_green}{min([b['size'] for b in green_blocks], default=0):.2f}{color_reset}", f"{color_green}{max([b['size'] for b in green_blocks], default=0):.2f}{color_reset}"],
        [f"{color_yellow}1MB to 2MB{color_reset}", f"{color_yellow}{len(yellow_blocks)}{color_reset}", f"{color_yellow}{len(yellow_blocks) / total_blocks * 100:.2f}%{color_reset}", f"{color_yellow}{calculate_avg([b['size'] for b in yellow_blocks]):.2f}{color_reset}", f"{color_yellow}{min([b['size'] for b in yellow_blocks], default=0):.2f}{color_reset}", f"{color_yellow}{max([b['size'] for b in yellow_blocks], default=0):.2f}{color_reset}"],
        [f"{color_orange}2MB to 3MB{color_reset}", f"{color_orange}{len(orange_blocks)}{color_reset}", f"{color_orange}{len(orange_blocks) / total_blocks * 100:.2f}%{color_reset}", f"{color_orange}{calculate_avg([b['size'] for b in orange_blocks]):.2f}{color_reset}", f"{color_orange}{min([b['size'] for b in orange_blocks], default=0):.2f}{color_reset}", f"{color_orange}{max([b['size'] for b in orange_blocks], default=0):.2f}{color_reset}"],
        [f"{color_red}3MB to 5MB{color_reset}", f"{color_red}{len(red_blocks)}{color_reset}", f"{color_red}{len(red_blocks) / total_blocks * 100:.2f}%{color_reset}", f"{color_red}{calculate_avg([b['size'] for b in red_blocks]):.2f}{color_reset}", f"{color_red}{min([b['size'] for b in red_blocks], default=0):.2f}{color_reset}", f"{color_red}{max([b['size'] for b in red_blocks], default=0):.2f}{color_reset}"],
        [f"{color_magenta}Greater than 5MB{color_reset}", f"{color_magenta}{len(magenta_blocks)}{color_reset}", f"{color_magenta}{len(magenta_blocks) / total_blocks * 100:.2f}%{color_reset}", f"{color_magenta}{calculate_avg([b['size'] for b in magenta_blocks]):.2f}{color_reset}", f"{color_magenta}{min([b['size'] for b in magenta_blocks], default=0):.2f}{color_reset}", f"{color_magenta}{max([b['size'] for b in magenta_blocks], default=0):.2f}{color_reset}"]
    ]

    table_str = tabulate(table, headers=headers, tablefmt="pretty")
    table_str = table_str.replace("+", f"{color_dark_grey}+{color_reset}").replace("-", f"{color_dark_grey}-{color_reset}").replace("|", f"{color_dark_grey}|{color_reset}")
    print(table_str)

    if block_data:
        times = [datetime.fromisoformat(b['time']) for b in block_data]
        sizes = [b['size'] for b in block_data]
        colors = ['green' if size < 1 else 'yellow' if size < 2 else 'orange' if size < 3 else 'red' if size < 5 else 'magenta' for size in sizes]

        legend_patches = [
            mpatches.Patch(color='green', label='< 1MB'),
            mpatches.Patch(color='yellow', label='1MB to 2MB'),
            mpatches.Patch(color='orange', label='2MB to 3MB'),
            mpatches.Patch(color='red', label='3MB to 5MB'),
            mpatches.Patch(color='magenta', label='> 5MB')
        ]

        # Grouped bar chart
        print(f"{color_teal}Generating the bar chart...{color_reset}")
        fig, ax = plt.subplots(figsize=(38, 20))

        unique_days = list(sorted(set([dt.date() for dt in times])))
        bar_width = 0.15
        bar_positions = np.arange(len(unique_days))

        green_sizes = [sum(sizes[i] for i in range(len(sizes)) if times[i].date() == day and colors[i] == 'green') for day in unique_days]
        yellow_sizes = [sum(sizes[i] for i in range(len(sizes)) if times[i].date() == day and colors[i] == 'yellow') for day in unique_days]
        orange_sizes = [sum(sizes[i] for i in range(len(sizes)) if times[i].date() == day and colors[i] == 'orange') for day in unique_days]
        red_sizes = [sum(sizes[i] for i in range(len(sizes)) if times[i].date() == day and colors[i] == 'red') for day in unique_days]
        magenta_sizes = [sum(sizes[i] for i in range(len(sizes)) if times[i].date() == day and colors[i] == 'magenta') for day in unique_days]

        ax.bar(bar_positions - bar_width * 2, green_sizes, bar_width, label='< 1MB', color='green')
        ax.bar(bar_positions - bar_width, yellow_sizes, bar_width, label='1MB to 2MB', color='yellow')
        ax.bar(bar_positions, orange_sizes, bar_width, label='2MB to 3MB', color='orange')
        ax.bar(bar_positions + bar_width, red_sizes, bar_width, label='3MB to 5MB', color='red')
        ax.bar(bar_positions + bar_width * 2, magenta_sizes, bar_width, label='> 5MB', color='magenta')

        ax.set_title(f'Block Size Over Time (Grouped Bar Chart)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel('Block Size (MB)', fontsize=24)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([str(day) for day in unique_days], rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper right', fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{output_image_file_base}_bar_chart.png")
        print(f"{color_teal}Bar chart generated successfully.{color_reset}")

        # Scatter plot
        print(f"{color_teal}Generating the scatter plot...{color_reset}")
        fig, ax = plt.subplots(figsize=(38, 20))
        ax.scatter(times, sizes, color=colors)
        ax.set_title(f'Block Size Over Time (Scatter Plot)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel('Block Size (MB)', fontsize=24)
        ax.tick_params(axis='x', rotation=45, labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{output_image_file_base}_scatter_plot.png")
        print(f"{color_teal}Scatter plot generated successfully.{color_reset}")

        # Histogram plot
        print(f"{color_teal}Generating the histogram plot...{color_reset}")
        fig, ax = plt.subplots(figsize=(38, 20))
        ax.hist(sizes, bins=50, color='b', edgecolor='black')
        ax.set_title(f'Block Size Distribution (Histogram)\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
        ax.set_xlabel('Block Size (MB)', fontsize=24)
        ax.set_ylabel('Frequency', fontsize=24)
        ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{output_image_file_base}_histogram.png")
        print(f"{color_teal}Histogram plot generated successfully.{color_reset}")
    else:
        print("No data to plot.")

def main(num_workers, lower_height, upper_height, endpoint_type, endpoint_urls, json_file=None):
    global executor
    if json_file:
        with open(json_file, 'r') as f:
            data = json.load(f)
        if "lower_height" not in data or "upper_height" not in data:
            match = re.search(r'block_sizes_(\d+)_to_(\d+)_\d{8}_\d{6}\.json', json_file)
            if match:
                lower_height = int(match.group(1))
                upper_height = int(match.group(2))
            else:
                print("Error: The provided JSON file does not contain 'lower_height' or 'upper_height' keys.")
                return
        else:
            lower_height = data["lower_height"]
            upper_height = data["upper_height"]
        generate_graphs_and_table(data, json_file.split('.json')[0], lower_height, upper_height)
        return

    print(f"{color_light_blue}\nChecking the specified starting block height...{color_reset}")

    retries = 3
    for attempt in range(retries):
        if check_endpoint(endpoint_type, endpoint_urls):
            break
        else:
            print(f"{color_yellow}RPC endpoint unreachable. Retrying {attempt + 1}/{retries}...{color_reset}")
            time.sleep(5)
    else:
        print(f"{color_red}RPC endpoint unreachable after multiple attempts. Exiting.{color_reset}")
        sys.exit(1)

    block_info = fetch_block_info(endpoint_type, endpoint_urls, lower_height)
    if block_info is None:
        print(f"{color_yellow}Block height {lower_height} does not exist. Finding the earliest available block height...{color_reset}")
        lower_height = find_lowest_height(endpoint_type, endpoint_urls)
        if lower_height is None:
            print(f"{color_red}Failed to determine the earliest block height. Exiting.{color_reset}")
            sys.exit(1)
        print(f"{color_light_blue}Using earliest available block height: {lower_height}{color_reset}")

    if lower_height > upper_height:
        print(f"{color_red}The specified lower height {lower_height} is greater than the specified upper height {upper_height}. Exiting.{color_reset}")
        sys.exit(1)

    print(f"{color_light_blue}\nFetching block information. This may take a while for large ranges. Please wait...{color_reset}")

    start_time = datetime.utcnow()
    current_date = start_time.strftime("%B %A %d, %Y %H:%M:%S UTC")
    output_file = f"block_sizes_{lower_height}_to_{upper_height}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    output_image_file_base = f"block_sizes_{lower_height}_to_{upper_height}_{start_time.strftime('%Y%m%d_%H%M%S')}"

    green_blocks = []
    yellow_blocks = []
    orange_blocks = []
    red_blocks = []
    magenta_blocks = []
    block_data = []

    total_blocks = upper_height - lower_height + 1
    start_script_time = time.time()

    print(f"{color_dark_grey}\n{'='*40}\n{color_reset}")

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

                if block_size_mb > 5:
                    magenta_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})
                elif block_size_mb > 3:
                    red_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})
                elif block_size_mb > 2:
                    orange_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})
                elif block_size_mb > 1:
                    yellow_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})
                else:
                    green_blocks.append({"height": height, "size": block_size_mb, "time": block_time.isoformat()})

            except Exception as e:
                print(f"Error processing block {future_to_height[future]}: {e}")

            completed += 1
            progress = (completed / total_blocks) * 100
            elapsed_time = time.time() - start_script_time
            estimated_total_time = elapsed_time / completed * total_blocks
            time_left = estimated_total_time - elapsed_time
            print(f"{color_light_blue}Progress: {progress:.2f}% ({completed}/{total_blocks}) - Estimated time left: {timedelta(seconds=int(time_left))}", end='\r')
    except KeyboardInterrupt:
        shutdown_event.set()
        if executor:
            executor.shutdown(wait=False)
        print(f"{color_red}\nProcess interrupted. Exiting gracefully...{color_reset}")
        sys.exit(0)

    executor.shutdown(wait=True)

    result = {
        "connection_type": endpoint_type,
        "endpoint": endpoint_urls,
        "run_time": current_date,
        "less_than_1MB": green_blocks,
        "1MB_to_2MB": yellow_blocks,
        "2MB_to_3MB": orange_blocks,
        "3MB_to_5MB": red_blocks,
        "greater_than_5MB": magenta_blocks,
        "block_data": block_data,
        "stats": {
            "less_than_1MB": {
                "count": len(green_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in green_blocks]),
                "min_size_mb": min([b["size"] for b in green_blocks], default=0),
                "max_size_mb": max([b["size"] for b in green_blocks], default=0),
                "percentage": len(green_blocks) / total_blocks * 100
            },
            "1MB_to_2MB": {
                "count": len(yellow_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in yellow_blocks]),
                "min_size_mb": min([b["size"] for b in yellow_blocks], default=0),
                "max_size_mb": max([b["size"] for b in yellow_blocks], default=0),
                "percentage": len(yellow_blocks) / total_blocks * 100
            },
            "2MB_to_3MB": {
                "count": len(orange_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in orange_blocks]),
                "min_size_mb": min([b["size"] for b in orange_blocks], default=0),
                "max_size_mb": max([b["size"] for b in orange_blocks], default=0),
                "percentage": len(orange_blocks) / total_blocks * 100
            },
            "3MB_to_5MB": {
                "count": len(red_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in red_blocks]),
                "min_size_mb": min([b["size"] for b in red_blocks], default=0),
                "max_size_mb": max([b["size"] for b in red_blocks], default=0),
                "percentage": len(red_blocks) / total_blocks * 100
            },
            "greater_than_5MB": {
                "count": len(magenta_blocks),
                "avg_size_mb": calculate_avg([b["size"] for b in magenta_blocks]),
                "min_size_mb": min([b["size"] for b in magenta_blocks], default=0),
                "max_size_mb": max([b["size"] for b in magenta_blocks], default=0),
                "percentage": len(magenta_blocks) / total_blocks * 100
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
        print(f"{color_red}Usage: python blockbusteranalyzer.py <num_workers> <lower_height> <upper_height> <endpoint_type> <endpoint_urls> [json_file]{color_reset}")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    lower_height = int(sys.argv[2])
    upper_height = int(sys.argv[3])
    endpoint_type = sys.argv[4]
    endpoint_urls = sys.argv[5]

    json_file = sys.argv[6] if len(sys.argv) == 7 else None

    main(num_workers, lower_height, upper_height, endpoint_type, endpoint_urls, json_file)
