#!/bin/bash
# @Author - Jonathan - Erialos
# @Email - erialos@thesilverfox.pro
# @Website - https://thesilverfox.pro
# @GitHub - https://github.com/effofxprime
# @Twitter - https://twitter.com/ErialosOfAstora
# @Date - 2023-08-04 15:19:53 UTC
# @Last_Modified_By - Jonathan - Erialos
# @Last_Modified_Time - 2024-06-07 15:24:00 UTC
# @Description - A tool to analyze block sizes in a blockchain.

# Define the lower and upper block height
LOWER_HEIGHT=$1
UPPER_HEIGHT=$2
ENDPOINT_TYPE=$3  # "socket" or "tcp"
ENDPOINT_URL=$4   # For example: "/dev/shm/jackal/trpc.socket" or "http://localhost:26657"

# Get the current date and time to create a unique output file name
START_TIME=$(date -u +"%s")
CURRENT_DATE=$(date -u +"%B %A %d, %Y %H:%M:%S UTC")
OUTPUT_FILE="block_sizes_${LOWER_HEIGHT}_to_${UPPER_HEIGHT}_$(date +"%Y%m%d_%H%M%S").json"

# Calculate the total number of blocks
TOTAL_BLOCKS=$((UPPER_HEIGHT - LOWER_HEIGHT + 1))

# Initialize JSON structure
echo "{\"connection_type\": \"$ENDPOINT_TYPE\", \"endpoint\": \"$ENDPOINT_URL\", \"run_time\": \"$CURRENT_DATE\", \"1MB_to_3MB\":[], \"3MB_to_5MB\":[], \"greater_than_5MB\":[], \"stats\": {\"1MB_to_3MB\": {\"count\": 0, \"avg_size_mb\": 0}, \"3MB_to_5MB\": {\"count\": 0, \"avg_size_mb\": 0}, \"greater_than_5MB\": {\"count\": 0, \"avg_size_mb\": 0}}}" > $OUTPUT_FILE

# Function to check if the endpoint is reachable
check_endpoint() {
  if [ "$ENDPOINT_TYPE" == "socket" ]; then
    curl --unix-socket "$ENDPOINT_URL" -m 5 -s -f "http://localhost/health" >/dev/null
  else
    curl -m 5 -s -f "$ENDPOINT_URL/health" >/dev/null
  fi
}

# Function to calculate average size
calculate_avg() {
  local total=0
  local count=0
  for size in "$@"; do
    total=$(echo "$total + $size" | bc)
    ((count++))
  done
  echo "scale=2; $total / $count" | bc
}

# Main loop to process blocks
START_SCRIPT_TIME=$(date +%s)
for (( height=$LOWER_HEIGHT; height<=$UPPER_HEIGHT; height++ ))
do
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
  if (( $(echo "$block_size_mb > 5" | bc -l) )); then
    # Greater than 5MB group
    jq --argjson height "$height" --arg size "$block_size_mb" '.["greater_than_5MB"] += [{"height": $height, "size": $size}]' $OUTPUT_FILE > tmp.json && mv tmp.json $OUTPUT_FILE
  elif (( $(echo "$block_size_mb > 3" | bc -l) )); then
    # 3MB to 5MB group
    jq --argjson height "$height" --arg size "$block_size_mb" '.["3MB_to_5MB"] += [{"height": $height, "size": $size}]' $OUTPUT_FILE > tmp.json && mv tmp.json $OUTPUT_FILE
  elif (( $(echo "$block_size_mb > 1" | bc -l) )); then
    # 1MB to 3MB group
    jq --argjson height "$height" --arg size "$block_size_mb" '.["1MB_to_3MB"] += [{"height": $height, "size": $size}]' $OUTPUT_FILE > tmp.json && mv tmp.json $OUTPUT_FILE
  fi

  # Calculate and display progress
  COMPLETED=$((height - LOWER_HEIGHT + 1))
  PROGRESS=$(echo "scale=2; ($COMPLETED / $TOTAL_BLOCKS) * 100" | bc)
  ELAPSED_TIME=$(($(date +%s) - START_SCRIPT_TIME))
  ESTIMATED_TOTAL_TIME=$(echo "scale=2; $ELAPSED_TIME / $COMPLETED * $TOTAL_BLOCKS" | bc)
  TIME_LEFT=$(echo "$ESTIMATED_TOTAL_TIME - $ELAPSED_TIME" | bc)
  echo -ne "Progress: $PROGRESS% ($COMPLETED/$TOTAL_BLOCKS) - Estimated time left: $(date -u -d @${TIME_LEFT} +%H:%M:%S)\r"
done

echo -e "\nWriting results to $OUTPUT_FILE"

# Calculate averages and counts
count_1MB_to_3MB=$(jq '.["1MB_to_3MB"] | length' $OUTPUT_FILE)
avg_1MB_to_3MB=$(jq '[.["1MB_to_3MB"][] | .size] | add / length' $OUTPUT_FILE)
count_3MB_to_5MB=$(jq '.["3MB_to_5MB"] | length' $OUTPUT_FILE)
avg_3MB_to_5MB=$(jq '[.["3MB_to_5MB"][] | .size] | add / length' $OUTPUT_FILE)
count_greater_than_5MB=$(jq '.["greater_than_5MB"] | length' $OUTPUT_FILE)
avg_greater_than_5MB=$(jq '[.["greater_than_5MB"][] | .size] | add / length' $OUTPUT_FILE)

jq --arg count1 "$count_1MB_to_3MB" --arg avg1 "$avg_1MB_to_3MB" --arg count2 "$count_3MB_to_5MB" --arg avg2 "$avg_3MB_to_5MB" --arg count3 "$count_greater_than_5MB" --arg avg3 "$avg_greater_than_5MB" \
   '.stats["1MB_to_3MB"].count = ($count1 | tonumber) | .stats["1MB_to_3MB"].avg_size_mb = ($avg1 | tonumber) | .stats["3MB_to_5MB"].count = ($count2 | tonumber) | .stats["3MB_to_5MB"].avg_size_mb = ($avg2 | tonumber) | .stats["greater_than_5MB"].count = ($count3 | tonumber) | .stats["greater_than_5MB"].avg_size_mb = ($avg3 | tonumber)' \
   $OUTPUT_FILE > tmp.json && mv tmp.json $OUTPUT_FILE

END_SCRIPT_TIME=$(date +%s)
TOTAL_DURATION=$(($END_SCRIPT_TIME - $START_SCRIPT_TIME))
echo "Script completed in: $(date -u -d @${TOTAL_DURATION} +%H:%M:%S)"

# Display the counts of each group
echo "Number of blocks in each group:"
echo "1MB to 3MB: $count_1MB_to_3MB"
echo "3MB to 5MB: $count_3MB_to_5MB"
echo "Greater than 5MB: $count_greater_than_5MB"
