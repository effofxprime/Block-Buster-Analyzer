import requests
import requests_unixsocket
import json
import time
import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
from colorama import Fore, Style

# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-07 15:19:53 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-09 23:24:00 UTC
# @Description - script to analyze block sizes

# Color constants
color_reset = '\033[0m'
color_green = '\033[38;5;10m'
color_yellow = '\033[38;5;11m'
color_orange = '\033[38;5;214m'
color_red = '\033[38;5;9m'
color_magenta = '\033[38;5;13m'
color_lightblue = '\033[38;5;123m'
color_darkgrey = '\033[38;5;245m'

def signal_handler(sig, frame):
    print("\nProcess interrupted. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def check_endpoint(endpoint_type, endpoint_url):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            response = session.get(f"http+unix://{endpoint_url}/health")
        else:
            response = requests.get(f"{endpoint_url}/health")

        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error checking endpoint: {e}")
        return False

def fetch_block_info(height, endpoint_type, endpoint_url):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            response = session.get(f"http+unix://{endpoint_url}/block?height={height}")
        else:
            response = requests.get(f"{endpoint_url}/block?height={height}")

        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching block info for height {height}: {e}")
        return None

def process_block(height, endpoint_type, endpoint_url):
    block_info = fetch_block_info(height, endpoint_type, endpoint_url)
    if block_info:
        size = block_info['result']['block']['data']['txs']
        block_size = sum(len(tx) for tx in size) / (1024 * 1024) if size else 0

        block_time_str = block_info['result']['block']['header']['time']
        block_time = datetime.fromisoformat(block_time_str.replace("Z", "+00:00"))

        return {"height": height, "size": block_size, "time": block_time.isoformat()}
    return None

def calculate_avg(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

def main(lower_height, upper_height, endpoint_type, endpoint_url):
    print(f"\n{color_lightblue}Fetching block information. This may take a while for large ranges. Please wait...{color_reset}\n")
    print(f"{color_darkgrey}{'='*40}{color_reset}\n")

    start_script_time = time.time()
    block_data = []
    progress = 0
    total_blocks = upper_height - lower_height + 1

    green_blocks = []
    yellow_blocks = []
    orange_blocks = []
    red_blocks = []
    magenta_blocks = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_height = {
            executor.submit(process_block, height, endpoint_type, endpoint_url): height
            for height in range(lower_height, upper_height + 1)
        }

        try:
            for future in as_completed(future_to_height):
                height = future_to_height[future]
                try:
                    block_info = future.result()
                    if block_info:
                        block_data.append(block_info)
                        block_size = block_info["size"]

                        if block_size < 1:
                            green_blocks.append(block_info)
                        elif block_size < 2:
                            yellow_blocks.append(block_info)
                        elif block_size < 3:
                            orange_blocks.append(block_info)
                        elif block_size < 5:
                            red_blocks.append(block_info)
                        else:
                            magenta_blocks.append(block_info)

                except Exception as e:
                    print(f"Error processing block {height}: {e}")

                progress += 1
                elapsed_time = time.time() - start_script_time
                estimated_time_left = (elapsed_time / progress) * (total_blocks - progress)
                print(f"\rProgress: {progress / total_blocks:.2%} ({progress}/{total_blocks}) - Estimated time left: {timedelta(seconds=int(estimated_time_left))}", end='')

        except KeyboardInterrupt:
            print("\nProcess interrupted. Exiting gracefully...")
            sys.exit(0)

    output_file = f"block_sizes_{lower_height}_to_{upper_height}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    output_image_file_base = f"block_sizes_{lower_height}_to_{upper_height}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    result = {
        "metadata": {
            "connection_type": endpoint_type,
            "endpoint_url": endpoint_url,
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            "block_range": {
                "start": lower_height,
                "end": upper_height
            },
            "group_counts": {
                "less_than_1MB": {
                    "count": len(green_blocks),
                    "avg_size_mb": calculate_avg([b["size"] for b in green_blocks]),
                    "min_size_mb": min([b["size"] for b in green_blocks], default=0),
                    "max_size_mb": max([b["size"] for b in green_blocks], default=0)
                },
                "1MB_to_2MB": {
                    "count": len(yellow_blocks),
                    "avg_size_mb": calculate_avg([b["size"] for b in yellow_blocks]),
                    "min_size_mb": min([b["size"] for b in yellow_blocks], default=0),
                    "max_size_mb": max([b["size"] for b in yellow_blocks], default=0)
                },
                "2MB_to_3MB": {
                    "count": len(orange_blocks),
                    "avg_size_mb": calculate_avg([b["size"] for b in orange_blocks]),
                    "min_size_mb": min([b["size"] for b in orange_blocks], default=0),
                    "max_size_mb": max([b["size"] for b in orange_blocks], default=0)
                },
                "3MB_to_5MB": {
                    "count": len(red_blocks),
                    "avg_size_mb": calculate_avg([b["size"] for b in red_blocks]),
                    "min_size_mb": min([b["size"] for b in red_blocks], default=0),
                    "max_size_mb": max([b["size"] for b in red_blocks], default=0)
                },
                "greater_than_5MB": {
                    "count": len(magenta_blocks),
                    "avg_size_mb": calculate_avg([b["size"] for b in magenta_blocks]),
                    "min_size_mb": min([b["size"] for b in magenta_blocks], default=0),
                    "max_size_mb": max([b["size"] for b in magenta_blocks], default=0)
                }
            }
        },
        "blocks": block_data
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"\n\n{color_green}Block sizes have been written to {output_file}{color_reset}")
    elapsed_time = time.time() - start_script_time
    print(f"{color_green}Script completed in: {timedelta(seconds=int(elapsed_time))}{color_reset}\n")

    # Print table
    headers = [f"{color_lightblue}Block Size Range{color_reset}", f"{color_lightblue}Count{color_reset}", f"{color_lightblue}Average Size (MB){color_reset}", f"{color_lightblue}Min Size (MB){color_reset}", f"{color_lightblue}Max Size (MB){color_reset}"]
    table_data = [
        [f"{color_green}< 1MB{color_reset}", f"{color_green}{len(green_blocks)}{color_reset}", f"{color_green}{calculate_avg([b['size'] for b in green_blocks]):.2f}{color_reset}", f"{color_green}{min([b['size'] for b in green_blocks], default=0):.2f}{color_reset}", f"{color_green}{max([b['size'] for b in green_blocks], default=0):.2f}{color_reset}"],
        [f"{color_yellow}1MB to 2MB{color_reset}", f"{color_yellow}{len(yellow_blocks)}{color_reset}", f"{color_yellow}{calculate_avg([b['size'] for b in yellow_blocks])::.2f}{color_reset}", f"{color_yellow}{min([b['size'] for b in yellow_blocks], default=0)::2f}{color_reset}", f"{color_yellow}{max([b['size'] for b in yellow_blocks], default=0):.2f}{color_reset}"],
        [f"{color_orange}2MB to 3MB{color_reset}", f"{color_orange}{len(orange_blocks)}{color_reset}", f"{color_orange}{calculate_avg([b['size'] for b in orange_blocks])::.2f}{color_reset}", f"{color_orange}{min([b['size'] for b in orange_blocks], default=0):.2f}{color_reset}", f"{color_orange}{max([b['size'] for b in orange_blocks], default=0):.2f}{color_reset}"],
        [f"{color_red}3MB to 5MB{color_reset}", f"{color_red}{len(red_blocks)}{color_reset}", f"{color_red}{calculate_avg([b['size'] for b in red_blocks])::.2f}{color_reset}", f"{color_red}{min([b['size'] for b in red_blocks], default=0):.2f}{color_reset}", f"{color_red}{max([b['size'] for b in red_blocks], default=0):.2f}{color_reset}"],
        [f"{color_magenta}Greater than 5MB{color_reset}", f"{color_magenta}{len(magenta_blocks)}{color_reset}", f"{color_magenta}{calculate_avg([b['size'] for b in magenta_blocks])::.2f}{color_reset}", f"{color_magenta}{min([b['size'] for b in magenta_blocks], default=0):.2f}{color_reset}", f"{color_magenta}{max([b['size'] for b in magenta_blocks], default=0):.2f}{color_reset}"]
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

    # Plotting graphs
    block_times = [datetime.fromisoformat(b['time']) for b in block_data]
    block_sizes = [b['size'] for b in block_data]

    if block_data:
        fig, ax = plt.subplots(figsize=(16, 9))
        scatter = ax.scatter(block_times, block_sizes, c=block_sizes, cmap='coolwarm', alpha=0.6, edgecolors='w', linewidth=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Block Size (MB)")
        ax.set_title(f"Block Sizes from {lower_height} to {upper_height}")
        legend_labels = ["< 1MB", "1MB to 2MB", "2MB to 3MB", "3MB to 5MB", "> 5MB"]
        legend_colors = [color_green, color_yellow, color_orange, color_red, color_magenta]
        legend_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(legend_colors, legend_labels)]
        ax.legend(handles=legend_handles, loc="upper right")
        plt.colorbar(scatter, ax=ax, label="Block Size (MB)")
        plt.savefig(f"{output_image_file_base}_scatter.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 9))
        heights = [b['height'] for b in block_data]
        sizes = [b['size'] for b in block_data]
        ax.hist(sizes, bins=30, color=color_lightblue, edgecolor=color_darkgrey, alpha=0.7)
        ax.set_xlabel("Block Size (MB)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of Block Sizes from {lower_height} to {upper_height}")
        plt.savefig(f"{output_image_file_base}_histogram.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 9))
        unique_days = sorted(set([bt.date() for bt in block_times]))
        day_indices = {day: idx for idx, day in enumerate(unique_days)}
        colors = [color_green if s < 1 else color_yellow if s < 2 else color_orange if s < 3 else color_red if s < 5 else color_magenta for s in block_sizes]
        grouped_blocks = {day: [] for day in unique_days}
        for bt, bs in zip(block_times, block_sizes):
            grouped_blocks[bt.date()].append(bs)
        bar_width = 0.35
        for day, sizes in grouped_blocks.items():
            bar_positions = np.arange(len(sizes))
            ax.bar(bar_positions, sizes, color=colors[:len(sizes)], edgecolor=color_darkgrey)
        ax.set_xlabel("Block")
        ax.set_ylabel("Block Size (MB)")
        ax.set_title(f"Grouped Bar Chart of Block Sizes from {lower_height} to {upper_height}")
        ax.legend(handles=legend_handles, loc="upper right")
        plt.savefig(f"{output_image_file_base}_grouped_bar.png", dpi=300)
        plt.close()
    else:
        print(f"{color_red}No block data available to plot.{color_reset}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(f"Usage: python3 blockbusteranalyzer.py <lower_height> <upper_height> <endpoint_type> <endpoint_url>")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    lower_height = int(sys.argv[2])
    upper_height = int(sys.argv[3])
    endpoint_type = sys.argv[4]
    endpoint_urls = sys.argv[5].split(',')

    main(lower_height, upper_height, endpoint_type, endpoint_urls)
