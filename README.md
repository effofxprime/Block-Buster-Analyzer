# BlockBusterAnalyzer

BlockBusterAnalyzer is a fun and catchy tool to monitor and analyze the sizes of blocks in a blockchain. It supports both Python and Bash scripts to query block sizes over a specified range and categorize them into different size groups.

## Features

- Query block sizes within a specified range
- Categorize blocks into size groups (1MB to 3MB, 3MB to 5MB, greater than 5MB)
- Supports both Unix sockets and TCP endpoints
- Generates a JSON report with the results
- Provides progress updates during execution, including estimated time left and total duration
- Uses multithreading (Python) and parallel processing (Bash) to improve performance for large block ranges
- Pretty console output with colors and table formatting
- Plots a graph of block size over time

## Dependencies

### Python

- `requests`
- `requests-unixsocket`
- `colorama`
- `tabulate`
- `matplotlib`
- `concurrent.futures` (standard library, no extra installation required)

### Bash

- `curl`
- `jq`
- `bc`

## Installation

### Python

#### Using `pip`

1. **Install Python dependencies**

    ```sh
    pip install requests requests-unixsocket colorama tabulate matplotlib
    ```

2. **Create a virtual environment (optional but recommended)**

    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Linux and macOS
    ```

#### Using `apt-get` (Ubuntu/Debian)

1. **Install Python dependencies**

    ```sh
    sudo apt-get update
    sudo apt-get install python3-requests python3-requests-unixsocket python3-colorama python3-tabulate python3-matplotlib
    ```

### Bash

1. **Install `curl`, `jq`, and `bc`**

    On Ubuntu/Debian:

    ```sh
    sudo apt-get update
    sudo apt-get install curl jq bc
    ```

    On CentOS/RHEL:

    ```sh
    sudo yum install curl jq bc
    ```

## Usage

### Python Script

1. **Run the script**

    ```sh
    python3 blockbusteranalyzer.py <lower_height> <upper_height> <endpoint_type> <endpoint_url>
    ```

    Example:

    ```sh
    python3 blockbusteranalyzer.py 7874000 7875000 "socket" "/dev/shm/jackal/trpc.socket"
    python3 blockbusteranalyzer.py 7874000 7875000 "tcp" "https://rpc.jackalprotocol.com:443"
    ```

### Bash Script

1. **Make the script executable**

    ```sh
    chmod +x blockbusteranalyzer.sh
    ```

2. **Run the script**

    ```sh
    ./blockbusteranalyzer.sh <lower_height> <upper_height> <endpoint_type> <endpoint_url>
    ```

    Example:

    ```sh
    ./blockbusteranalyzer.sh 7874000 7875000 "socket" "/dev/shm/jackal/trpc.socket"
    ./blockbusteranalyzer.sh 7874000 7875000 "tcp" "https://rpc.jackalprotocol.com:443"
    ```

## Output

The scripts will generate a JSON file with a name in the format `block_sizes_<lower_height>_to_<upper_height>_<current_date>.json` containing the block sizes categorized into the specified groups. The file will be saved in the current directory.

Additionally, a graph of block size over time will be saved as a PNG file.

## Contributing

Feel free to fork the project, submit pull requests, and file issues. We welcome contributions and improvements to BlockBusterAnalyzer!

## License

This project is licensed under the MIT License.
