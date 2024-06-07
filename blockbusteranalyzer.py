#!/usr/bin/env python3
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 15:19:00 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-07 15:24:00 UTC
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

# Initialize colorama
init(autoreset=True)

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

def calculate_avg(sizes):
    return sum(sizes) / len(sizes) if sizes else 0

def process_block(height, endpoint_type, endpoint_url):
    block_info = fetch_block_info(endpoint_type, endpoint_url, height)
    if block_info is None:
        return None

    block_size = len(json.dumps(block_info))
    block_size_mb = block_size / 1048576
    return (height, block_size_mb)

def main(lower_height, upper_height, endpoint_type, endpoint_url):
    print("\nFetching block information. This may take a while for large ranges. Please wait...")

    start_time = datetime.utcnow()
    current_date = start_time.strftime("%B %A %d, %Y %H:%M:%S UTC")
    output_file = f"block_sizes_{lower_height}_to_{upper_height}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"

    yellow_blocks = []
    red_blocks = []
    magenta_blocks = []

    total_blocks = upper_height - lower_height + 1
    start_script_time = time.time()

    if not check_endpoint(endpoint_type, endpoint_url):
        print("RPC endpoint unreachable. Exiting.")
        sys.exit(1)

    print("\n" + "="*40 + "\n")

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_height = {executor.submit(process_block, height, endpoint_type, endpoint_url): height for height in range(lower_height, upper_height + 1)}

        for future in as_completed(future_to_height):
            height = future_to_height[future]
            try:
                result = future.result()
                if result is None:
                    continue

                height, block_size_mb = result

                if block_size_mb > 5:
                    magenta_blocks.append({"height": height, "size": block_size_mb})
                elif block_size_mb > 3:
                    red_blocks.append({"height": height, "size": block_size_mb})
                elif block_size_mb > 1:
                    yellow_blocks.append({"height": height, "size": block_size_mb})

            except Exception as e:
                print(f"Error processing block {height}: {e}")

            completed = height - lower_height + 1
            progress = (completed / total_blocks) * 100
            elapsed_time = time.time() - start_script_time
            estimated_total_time = elapsed_time / completed * total_blocks
            time_left = estimated_total_time - elapsed_time
            print(f"Progress: {progress:.2f}% ({completed}/{total_blocks}) - Estimated time left: {timedelta(seconds=int(time_left))}", end='\r')

    result = {
        "connection_type": endpoint_type,
        "endpoint": endpoint_url,
        "run_time": current_date,
        "1MB_to_3MB": yellow_blocks,
        "3MB_to_5MB": red_blocks,
        "greater_than_5MB": magenta_blocks,
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

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python blockbusteranalyzer.py <lower_height> <upper_height> <endpoint_type> <endpoint_url>")
        sys.exit(1)

    lower_height = int(sys.argv[1])
    upper_height = int(sys.argv[2])
    endpoint_type = sys.argv[3]
    endpoint_url = sys.argv[4]

    main(lower_height, upper_height, endpoint_type, endpoint_url)
