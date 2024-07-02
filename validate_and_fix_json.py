#!/usr/bin/env python3
import json
import sys
import os

def log_message(level, message):
    print(f"{level.upper()}: {message}")

def validate_and_fix_json(file_path):
    """
    Validate and fix JSON format issues. This function will read the JSON file, remove any incomplete
    entries, and ensure the JSON format is correct.
    """
    try:
        # Read the JSON file
        with open(file_path, 'r') as file:
            data = file.read()

        # Ensure the JSON starts with '[' and ends with ']', fixing if necessary
        if not data.startswith('['):
            data = '[' + data
        if not data.endswith(']'):
            data = data + ']'

        # Attempt to load the JSON data
        try:
            blocks = json.loads(data)
        except json.JSONDecodeError as e:
            log_message('error', f"JSONDecodeError: {e}")
            return False

        # Check and remove incomplete entries
        valid_blocks = []
        for block in blocks:
            if isinstance(block, dict) and all(key in block for key in ["height", "size", "time"]):
                valid_blocks.append(block)
            else:
                log_message('warning', f"Incomplete block entry found and removed: {block}")

        # Save the corrected JSON data
        with open(file_path, 'w') as file:
            json.dump(valid_blocks, file, default=str)

        log_message('info', "JSON file has been validated and fixed.")
        return True

    except FileNotFoundError:
        log_message('error', f"FileNotFoundError: {file_path} not found.")
        return False
    except Exception as e:
        log_message('error', f"An unknown error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    if not os.path.exists(json_file_path):
        log_message('error', f"JSON file {json_file_path} does not exist.")
        sys.exit(1)

    if validate_and_fix_json(json_file_path):
        log_message('info', f"JSON file {json_file_path} has been successfully validated and fixed.")
    else:
        log_message('error', f"Failed to validate and fix JSON file {json_file_path}.")
