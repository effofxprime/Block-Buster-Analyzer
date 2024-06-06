#!/bin/bash
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2024-06-06 09:19:53
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-06 09:19:53
# @Description - A tool to analyze block sizes in a blockchain.

# Define the lower and upper block height
LOWER_HEIGHT=$1
UPPER_HEIGHT=$2
ENDPOINT_TYPE=$3 # "socket" or "tcp"
ENDPOINT_URL=$4  # For example: "/dev/shm/jackal/trpc.socket" or "http://localhost:26657"

# Get the current date and time to create a unique output file name
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="block_sizes_${LOWER_HEIGHT}_to_${UPPER_HEIGHT}_${CURRENT_DATE}.json"

# Calculate the total number of blocks
TOTAL_BLOCKS=$((UPPER_HEIGHT - LOWER_HEIGHT + 1))

# Initialize JSON structure
echo '{"1MB_to_3MB":[],"3MB_to_5MB":[],"greater_than_5MB":[]}' >$OUTPUT_FILE

# Function to check if the endpoint is reachable
check_endpoint() {
    if [ "$ENDPOINT_TYPE" == "socket" ]; then
        curl --unix-socket "$ENDPOINT_URL" -m 5 -s -f "http://localhost/health" >/dev/null
    else
        curl -m 5 -s -f "$ENDPOINT_URL/health" >/dev/null
    fi
}

# Main loop to process blocks
for ((height = $LOWER_HEIGHT; height <= $UPPER_HEIGHT; height++)); do
    # Wait up to 60 seconds if the endpoint is unreachable
    SECONDS_WAITED=0
    while ! check_endpoint; do
        if [ $SECONDS_WAITED -ge 60 ]; then
            echo "RPC endpoint unreachable for 60 seconds. Exiting."
            exit 1
        fi
        echo "RPC endpoint unreachable. Waiting..."
        sleep 1
        SECONDS_WAITED=$((SECONDS_WAITED + 1))
    done

    # Query the block info at the specified height
    if [ "$ENDPOINT_TYPE" == "socket" ]; then
        block_info=$(curl -s --unix-socket "$ENDPOINT_URL" "http://localhost/block?height=$height")
    else
        block_info=$(curl -s "$ENDPOINT_URL/block?height=$height")
    fi

    # Calculate the size in bytes
    block_size=$(echo "$block_info" | jq -c . | wc -c)

    # Convert to MB
    block_size_mb=$(echo "scale=2; $block_size / 1048576" | bc)

    # Determine the output based on block size
    if (($(echo "$block_size_mb > 5" | bc -l))); then
        # Greater than 5MB group
        jq --argjson height "$height" --arg size "$block_size_mb" '.["greater_than_5MB"] += [{"height": $height, "size": $size}]' $OUTPUT_FILE >tmp.json && mv tmp.json $OUTPUT_FILE
    elif (($(echo "$block_size_mb > 3" | bc -l))); then
        # 3MB to 5MB group
        jq --argjson height "$height" --arg size "$block_size_mb" '.["3MB_to_5MB"] += [{"height": $height, "size": $size}]' $OUTPUT_FILE >tmp.json && mv tmp.json $OUTPUT_FILE
    elif (($(echo "$block_size_mb > 1" | bc -l))); then
        # 1MB to 3MB group
        jq --argjson height "$height" --arg size "$block_size_mb" '.["1MB_to_3MB"] += [{"height": $height, "size": $size}]' $OUTPUT_FILE >tmp.json && mv tmp.json $OUTPUT_FILE
    fi

    # Calculate and display progress
    COMPLETED=$((height - LOWER_HEIGHT + 1))
    PROGRESS=$(echo "scale=2; ($COMPLETED / $TOTAL_BLOCKS) * 100" | bc)
    echo -ne "Progress: $PROGRESS% ($COMPLETED/$TOTAL_BLOCKS)\r"
done

echo -e "\nWriting results to $OUTPUT_FILE"

# Display the counts of each group
count_1MB_to_3MB=$(jq '.["1MB_to_3MB"] | length' $OUTPUT_FILE)
count_3MB_to_5MB=$(jq '.["3MB_to_5MB"] | length' $OUTPUT_FILE)
count_greater_than_5MB=$(jq '.["greater_than_5MB"] | length' $OUTPUT_FILE)

echo "Number of blocks in each group:"
echo "1MB to 3MB: $count_1MB_to_3MB"
echo "3MB to 5MB: $count_3MB_to_5MB"
echo "Greater than 5MB: $count_greater_than_5MB"
