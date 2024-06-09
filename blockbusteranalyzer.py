import requests
import requests_unixsocket
import json
import sys
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, Style
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2023-06-07 15:24:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-08 23:10:00 UTC
# @Description - BlockBusterAnalyzer - A tool to monitor and analyze the sizes of blocks in a blockchain

color_reset = Style.RESET_ALL
color_green = Fore.GREEN
color_yellow = Fore.YELLOW
color_orange = Fore.LIGHTYELLOW_EX
color_red = Fore.RED
color_magenta = Fore.MAGENTA
color_light_blue = Fore.CYAN
color_dark_grey = Fore.LIGHTBLACK_EX

def calculate_avg(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

def fetch_block_info(endpoint_type, endpoint_url, height):
    if endpoint_type == "socket":
        session = requests_unixsocket.Session()
        url = f"http+unix://{endpoint_url}/block?height={height}"
    else:
        session = requests.Session()
        url = f"{endpoint_url}/block?height={height}"

    response = session.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def check_endpoint(endpoint_type, endpoint_url):
    try:
        if endpoint_type == "socket":
            session = requests_unixsocket.Session()
            url = f"http+unix://{endpoint_url}/health"
        else:
            session = requests.Session()
            url = f"{endpoint_url}/health"

        response = session.get(url)
        if response.status_code == 200:
            return True
    except Exception as e:
        print(f"Error checking endpoint: {e}")
    return False

def main(num_workers, lower_height, upper_height, endpoint_type, endpoint_urls):
    start_script_time = time.time()
    current_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

    # Check endpoint health
    for attempt in range(3):
        if check_endpoint(endpoint_type, endpoint_urls[0]):
            break
        else:
            print(f"{color_red}RPC endpoint unreachable. Retrying {attempt + 1}/3...{color_reset}")
            time.sleep(5)
    else:
        print(f"{color_red}Failed to reach the RPC endpoint after 3 attempts. Exiting.{color_reset}")
        sys.exit(1)

    # Check if the starting block height exists
    try:
        block_info = fetch_block_info(endpoint_type, endpoint_urls[0], lower_height)
        if 'result' not in block_info or 'block' not in block_info['result']:
            raise Exception(f"Block height {lower_height} does not exist.")
    except Exception as e:
        print(f"Block height {lower_height} does not exist. Finding the earliest available block height...")
        earliest_height_found = False
        for url in endpoint_urls:
            try:
                error_response = fetch_block_info(endpoint_type, url, 1)
                if 'error' in error_response and 'data' in error_response['error']:
                    data_message = error_response['error']['data']
                    print(f"Data message: {data_message}")
                    if "lowest height is" in data_message:
                        lower_height = int(data_message.split("lowest height is")[1].strip())
                        earliest_height_found = True
                        break
            except Exception as e:
                print(f"Error querying earliest block height: {e}")

        if not earliest_height_found:
            print(f"{color_red}Failed to determine the earliest block height. Exiting.{color_reset}")
            sys.exit(1)

        print(f"{color_green}Using earliest available block height: {lower_height}{color_reset}")

    total_blocks = upper_height - lower_height + 1
    completed = 0
    block_data = []
    green_blocks, yellow_blocks, orange_blocks, red_blocks, magenta_blocks = [], [], [], [], []
    output_file = f"block_sizes_{lower_height}_to_{upper_height}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    output_image_file_base = f"block_sizes_{lower_height}_to_{upper_height}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    shutdown_event = False

    def fetch_blocks(start_height, end_height, endpoint_url):
        blocks = []
        for height in range(start_height, end_height + 1):
            if shutdown_event:
                break
            try:
                block_info = fetch_block_info(endpoint_type, endpoint_url, height)
                if 'result' in block_info and 'block' in block_info['result']:
                    blocks.append(block_info['result']['block'])
            except Exception as e:
                print(f"Error fetching block {height}: {e}")
        return blocks

    print(f"\nFetching block information. This may take a while for large ranges. Please wait...\n")
    print(f"{'='*40}\n")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_height = {
                executor.submit(fetch_blocks, height, min(height + 49, upper_height), endpoint_urls[i % len(endpoint_urls)]): height
                for i, height in enumerate(range(lower_height, upper_height + 1, 50))
            }

            for future in as_completed(future_to_height):
                if shutdown_event:
                    break
                try:
                    blocks = future.result()
                    for block in blocks:
                        height = int(block['header']['height'])
                        block_size = sum(len(tx) for tx in block['data']['txs']) if block['data']['txs'] else 0
                        block_size_mb = block_size / (1024 * 1024)
                        block_time = datetime.strptime(block['header']['time'], "%Y-%m-%dT%H:%M:%S.%fZ")

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
                    print(f"Error processing blocks {future_to_height[future]}: {e}")

                completed += len(future_to_height[future])
                progress = (completed / total_blocks) * 100
                elapsed_time = time.time() - start_script_time
                estimated_total_time = elapsed_time / completed * total_blocks
                time_left = estimated_total_time - elapsed_time
                print(f"{color_light_blue}Progress: {progress:.2f}% ({completed}/{total_blocks}) - Estimated time left: {timedelta(seconds=int(time_left))}", end='\r')
    except KeyboardInterrupt:
        shutdown_event = True
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
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"{color_green}\nBlock sizes have been written to {output_file}{color_reset}")
    print(f"Script completed in: {timedelta(seconds=int(time.time() - start_script_time))}\n")

    headers = ["Block Size Range", "Count", "Average Size (MB)", "Min Size (MB)", "Max Size (MB)"]
    table_data = [
        [f"{color_green}< 1MB{color_reset}", f"{color_green}{len(green_blocks)}{color_reset}", f"{color_green}{calculate_avg([b['size'] for b in green_blocks]):.2f}{color_reset}", f"{color_green}{min([b['size'] for b in green_blocks], default=0):.2f}{color_reset}", f"{color_green}{max([b['size'] for b in green_blocks], default=0):.2f}{color_reset}"],
        [f"{color_yellow}1MB to 2MB{color_reset}", f"{color_yellow}{len(yellow_blocks)}{color_reset}", f"{color_yellow}{calculate_avg([b['size'] for b in yellow_blocks]):.2f}{color_reset}", f"{color_yellow}{min([b['size'] for b in yellow_blocks], default=0):.2f}{color_reset}", f"{color_yellow}{max([b['size'] for b in yellow_blocks], default=0):.2f}{color_reset}"],
        [f"{color_orange}2MB to 3MB{color_reset}", f"{color_orange}{len(orange_blocks)}{color_reset}", f"{color_orange}{calculate_avg([b['size'] for b in orange_blocks]):.2f}{color_reset}", f"{color_orange}{min([b['size'] for b in orange_blocks], default=0):.2f}{color_reset}", f"{color_orange}{max([b['size'] for b in orange_blocks], default=0):.2f}{color_reset}"],
        [f"{color_red}3MB to 5MB{color_reset}", f"{color_red}{len(red_blocks)}{color_reset}", f"{color_red}{calculate_avg([b['size'] for b in red_blocks])::.2f}{color_reset}", f"{color_red}{min([b['size'] for b in red_blocks], default=0):.2f}{color_reset}", f"{color_red}{max([b['size'] for b in red_blocks], default=0):.2f}{color_reset}"],
        [f"{color_magenta}> 5MB{color_reset}", f"{color_magenta}{len(magenta_blocks)}{color_reset}", f"{color_magenta}{calculate_avg([b['size'] for b in magenta_blocks]):.2f}{color_reset}", f"{color_magenta}{min([b['size'] for b in magenta_blocks], default=0):.2f}{color_reset}", f"{color_magenta}{max([b['size'] for b in magenta_blocks], default=0):.2f}{color_reset}"]
    ]
    
    print(f"\nNumber of blocks in each group for block heights {lower_height} to {upper_height}:\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center", numalign="center"))

    if block_data:
        times = [datetime.fromisoformat(b["time"]) for b in block_data]
        sizes = [b["size"] for b in block_data]
        colors = ['green' if s < 1 else 'yellow' if s < 2 else 'orange' if s < 3 else 'red' if s < 5 else 'magenta' for s in sizes]

        legend_patches = [
            mpatches.Patch(color='green', label='< 1MB'),
            mpatches.Patch(color='yellow', label='1MB to 2MB'),
            mpatches.Patch(color='orange', label='2MB to 3MB'),
            mpatches.Patch(color='red', label='3MB to 5MB'),
            mpatches.Patch(color='magenta', label='> 5MB')
        ]

        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(38, 20))
        unique_days = list(sorted(set(time.date() for time in times)))
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

        # Scatter plot
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

        # Histogram plot
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
    else:
        print(f"{color_red}No block data available to plot.{color_reset}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(f"Usage: python3 blockbusteranalyzer.py <num_workers> <lower_height> <upper_height> <endpoint_type> <endpoint_urls_comma_separated>")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    lower_height = int(sys.argv[2])
    upper_height = int(sys.argv[3])
    endpoint_type = sys.argv[4]
    endpoint_urls = sys.argv[5].split(',')

    main(num_workers, lower_height, upper_height, endpoint_type, endpoint_urls)
