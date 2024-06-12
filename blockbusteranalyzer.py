#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-12 03:00:00 UTC
# @Version - 1.0.5
# @Description - A tool to analyze block sizes in a blockchain.

import requests
import requests_unixsocket
import json
import time
import sys
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import signal
import threading
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px

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
    return {"height": height, "size": block_size_mb, "time": block_time}

def signal_handler(sig, frame):
    print(f"{color_red}\nProcess interrupted. Exiting gracefully...{color_reset}")
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
    categories = {
        "less_than_1MB": data.get("less_than_1MB", []),
        "1MB_to_2MB": data.get("1MB_to_2MB", []),
        "2MB_to_3MB": data.get("2MB_to_3MB", []),
        "3MB_to_5MB": data.get("3MB_to_5MB", []),
        "greater_than_5MB": data.get("greater_than_5MB", [])
    }
    
    total_blocks = sum(len(v) for v in categories.values())

    print(f"{color_teal}\nNumber of blocks in each group for block heights {lower_height} to {upper_height}:{color_reset}")
    headers = [f"{color_teal}Block Size Range{color_reset}", f"{color_teal}Count{color_reset}", f"{color_teal}Percentage{color_reset}", f"{color_teal}Average Size (MB){color_reset}", f"{color_teal}Min Size (MB){color_reset}", f"{color_teal}Max Size (MB){color_reset}"]
    table = [
        [f"{color_green}Less than 1MB{color_reset}", f"{color_green}{len(categories['less_than_1MB'])}{color_reset}", f"{color_green}{len(categories['less_than_1MB']) / total_blocks * 100:.2f}%{color_reset}", f"{color_green}{calculate_avg([b['size'] for b in categories['less_than_1MB']]):.2f}{color_reset}", f"{color_green}{min([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{color_reset}", f"{color_green}{max([b['size'] for b in categories['less_than_1MB']], default=0):.2f}{color_reset}"],
        [f"{color_yellow}1MB to 2MB{color_reset}", f"{color_yellow}{len(categories['1MB_to_2MB'])}{color_reset}", f"{color_yellow}{len(categories['1MB_to_2MB']) / total_blocks * 100:.2f}%{color_reset}", f"{color_yellow}{calculate_avg([b['size'] for b in categories['1MB_to_2MB']]):.2f}{color_reset}", f"{color_yellow}{min([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{color_reset}", f"{color_yellow}{max([b['size'] for b in categories['1MB_to_2MB']], default=0):.2f}{color_reset}"],
        [f"{color_orange}2MB to 3MB{color_reset}", f"{color_orange}{len(categories['2MB_to_3MB'])}{color_reset}", f"{color_orange}{len(categories['2MB_to_3MB']) / total_blocks * 100:.2f}%{color_reset}", f"{color_orange}{calculate_avg([b['size'] for b in categories['2MB_to_3MB']]):.2f}{color_reset}", f"{color_orange}{min([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{color_reset}", f"{color_orange}{max([b['size'] for b in categories['2MB_to_3MB']], default=0):.2f}{color_reset}"],
        [f"{color_red}3MB to 5MB{color_reset}", f"{color_red}{len(categories['3MB_to_5MB'])}{color_reset}", f"{color_red}{len(categories['3MB_to_5MB']) / total_blocks * 100:.2f}%{color_reset}", f"{color_red}{calculate_avg([b['size'] for b in categories['3MB_to_5MB']]):.2f}{color_reset}", f"{color_red}{min([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{color_reset}", f"{color_red}{max([b['size'] for b in categories['3MB_to_5MB']], default=0):.2f}{color_reset}"],
        [f"{color_magenta}Greater than 5MB{color_reset}", f"{color_magenta}{len(categories['greater_than_5MB'])}{color_reset}", f"{color_magenta}{len(categories['greater_than_5MB']) / total_blocks * 100:.2f}%{color_reset}", f"{color_magenta}{calculate_avg([b['size'] for b in categories['greater_than_5MB']]):.2f}{color_reset}", f"{color_magenta}{min([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{color_reset}", f"{color_magenta}{max([b['size'] for b in categories['greater_than_5MB']], default=0):.2f}{color_reset}"]
    ]
    print_table(headers, table)

    # Scatter plot
    print(f"{color_light_blue}Generating the scatter plot...{color_reset}")
    all_blocks = [block for blocks in categories.values() for block in blocks]
    times = [block['time'] for block in all_blocks]
    sizes = [block['size'] for block in all_blocks]
    colors = [color_green if block['size'] < 1 else color_yellow if block['size'] < 2 else color_orange if block['size'] < 3 else color_red if block['size'] < 5 else color_magenta for block in all_blocks]

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
        plt.Line2D([0], [0], marker='o', color='w', label='Less than 1MB', markerfacecolor=color_green, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='1MB to 2MB', markerfacecolor=color_yellow, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='2MB to 3MB', markerfacecolor=color_orange, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='3MB to 5MB', markerfacecolor=color_red, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Greater than 5MB', markerfacecolor=color_magenta, markersize=10)
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_scatter_plot.png")
    print(f"{color_light_green}Scatter plot generated successfully.{color_reset}")

    # Density Scatter Plot
    print(f"{color_light_blue}Generating the density scatter plot...{color_reset}")
    plt.figure(figsize=(38, 20))
    sns.kdeplot(x=times, y=sizes, fill=True, cmap="viridis", levels=100, thresh=0)
    plt.title(f'Block Size Density Over Time\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(label='Density')
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_density_scatter_plot.png")
    print(f"{color_light_green}Density scatter plot generated successfully.{color_reset}")

    # Interactive Scatter Plot
    print(f"{color_light_blue}Generating the interactive scatter plot...{color_reset}")
    fig = px.scatter(x=times, y=sizes, color=sizes, labels={'x': 'Time', 'y': 'Block Size (MB)'}, title=f'Block Size Over Time (Interactive Scatter Plot)\nBlock Heights {lower_height} to {upper_height}')
    fig.update_layout(title_font_size=28, xaxis_title_font_size=24, yaxis_title_font_size=24)
    fig.write_html(f"{output_image_file_base}_interactive_scatter_plot.html")
    print(f"{color_light_green}Interactive scatter plot generated successfully.{color_reset}")

    # Time Series Line Chart
    print(f"{color_light_blue}Generating the time series line chart...{color_reset}")
    df = pd.DataFrame({'time': times, 'size': sizes})
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df_resampled = df.resample('D').mean()
    plt.figure(figsize=(38, 20))
    plt.plot(df_resampled.index, df_resampled['size'], marker='o')
    plt.title(f'Average Block Size Over Time\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Average Block Size (MB)', fontsize=24)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_time_series_line_chart.png")
    print(f"{color_light_green}Time series line chart generated successfully.{color_reset}")

    # Box Plot
    print(f"{color_light_blue}Generating the box plot...{color_reset}")
    plt.figure(figsize=(38, 20))
    sns.boxplot(x=df.index.month, y='size', data=df)
    plt.title(f'Block Size Distribution Per Month\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    plt.xlabel('Month', fontsize=24)
    plt.ylabel('Block Size (MB)', fontsize=24)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_box_plot.png")
    print(f"{color_light_green}Box plot generated successfully.{color_reset}")

    # Heatmap
    print(f"{color_light_blue}Generating the heatmap...{color_reset}")
    heatmap_data = df.pivot_table(values='size', index=df.index.day, columns=df.index.month, aggfunc='mean')
    plt.figure(figsize=(38, 20))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Block Size (MB)'})
    plt.title(f'Block Size Heatmap\nBlock Heights {lower_height} to {upper_height}', fontsize=28)
    plt.xlabel('Month', fontsize=24)
    plt.ylabel('Day of Month', fontsize=24)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_image_file_base}_heatmap.png")
    print(f"{color_light_green}Heatmap generated successfully.{color_reset}")

def calculate_avg(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

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
                print(f"{color_red}Error: The provided JSON file does not contain 'lower_height' or 'upper_height' keys.{color_reset}")
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

    categories = {
        "less_than_1MB": [],
        "1MB_to_2MB": [],
        "2MB_to_3MB": [],
        "3MB_to_5MB": [],
        "greater_than_5MB": [],
    }

    total_blocks = upper_height - lower_height + 1
    start_script_time = time.time()

    print(f"{color_dark_grey}\n{'='*40}\n{color_reset}")

    executor = ThreadPoolExecutor(max_workers=num_workers)
    future_to_height = {executor.submit(process_block, height, endpoint_type, endpoint_urls): height for height in range(lower_height, upper_height + 1)}

    completed = 0
    block_data = []
    try:
        for future in as_completed(future_to_height):
            if shutdown_event.is_set():
                break

            try:
                result = future.result()
                if result is None:
                    continue

                block_data.append(result)
                categorize_block(result, categories)

            except Exception as e:
                print(f"{color_red}Error processing block {future_to_height[future]}: {e}{color_reset}")

            completed += 1
            progress = (completed / total_blocks) * 100
            elapsed_time = time.time() - start_script_time
            estimated_total_time = elapsed_time / completed * total_blocks
            time_left = estimated_total_time - elapsed_time
            print(f"{color_light_blue}Progress: {progress:.2f}% ({completed}/{total_blocks}) - Estimated time left: {timedelta(seconds=int(time_left))}{color_reset}", end='\r')
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
        print(f"{color_red}Usage: python blockbusteranalyzer.py <num_workers> <lower_height> <upper_height> <endpoint_type> <endpoint_urls> [json_file]{color_reset}")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    lower_height = int(sys.argv[2])
    upper_height = int(sys.argv[3])
    endpoint_type = sys.argv[4]
    endpoint_urls = sys.argv[5]

    json_file = sys.argv[6] if len(sys.argv) == 7 else None

    main(num_workers, lower_height, upper_height, endpoint_type, endpoint_urls, json_file)
