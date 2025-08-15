
### process_filesinfolder
### _20250805091013

import os  
import sys
import json
import logging
import time


def ensure_folder_exists(folder_path):
    """Ensure that the specified folder exists, creating it if necessary.
    Args:
        folder_path (str): The path to the folder to check or create.
    Returns:
        str: The path to the folder.
    """

    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            logging.info(f"Created folder: {folder_path}")
        except Exception as e:
            logging.error(f"Error creating folder {folder_path}: {e}")
            return None
    else:
        logging.info(f"Folder already exists: {folder_path}")

    if not os.path.isdir(folder_path):
        logging.error(f"Path is not a directory: {folder_path}")
        return None
    
    if not os.access(folder_path, os.R_OK):
        logging.error(f"Folder is not readable: {folder_path}")
        return None
    
    if not os.access(folder_path, os.W_OK):
        logging.error(f"Folder is not writable: {folder_path}")
        return None
    
    logging.info(f"Folder is ready for use: {folder_path}")

    return folder_path


def move_file(file_path, new_file_path):
    """Rename a file by appending a timestamp to its name.
    Args:
        file_path (str): The path to the file to be renamed.
        new_filename (str, optional): The new name for the file. If None, the file will be renamed with a timestamp.
    Returns:
        str: The new file path after renaming.
    """
    if not os.path.exists(file_path):
        logging.error(f"File does not exist: {file_path}")
        return file_path
    
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_{int(os.path.getmtime(file_path))}{ext}"
    try:
        os.rename(file_path, new_file_path)
        logging.info(f"Renamed file {file_path} to {new_file_path}")
        return new_file_path
    except Exception as e:
        logging.error(f"Error renaming file {file_path}: {e}")
        return file_path


def process_files_in_folder(folder_path):
    """
    Process all files in the specified folder.
    Args:
        folder_path (str): The path to the folder containing files to process.
    Returns:
        list: A list of processed file names.
    """
    currentdate = time.strftime("%Y%m%d%H%M%S")
    print(f"Current date: {currentdate}")

    # timestamp = int(os.path.getmtime(folder_path))
    processed_files = []
    try:
        if not os.path.exists(folder_path):
            logging.error(f"Folder does not exist: {folder_path}")
            return processed_files
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as file:
                        data = file.read()
                        
                        # Process the data (this is a placeholder for actual processing logic)
                        processed_data = data.upper()
                        processed_files.append({
                            'filename': filename,
                            'processed_data': processed_data
                        }) 
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")
        return processed_files
    except Exception as e:
        logging.error(f"An error occurred while processing files in folder {folder_path}: {e}")
        return processed_files
    
if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python process_filesinfolder.py <folder_path>")
    #     sys.exit(1)

    # folder_path = sys.argv[1]
    # processed_files = process_files_in_folder(folder_path)

# "C:\tmp\dmp\tmp\py_test\process_filesinfolder\zz done"
# "C:\tmp\dmp\tmp\py_test\process_filesinfolder\-- inc"

    folder_path = r"C:\tmp\dmp\tmp\py_test\process_filesinfolder\-- inc"
    folder_path_done = r"C:\tmp\dmp\tmp\py_test\process_filesinfolder\zz done"
    processed_files = process_files_in_folder(folder_path)

    print(json.dumps(processed_files, indent=4))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Processed {len(processed_files)} files in folder: {folder_path}")

# Example usage:
# python process_filesinfolder.py /path/to/folder
# This will process all files in the specified folder and print the processed data.
# Ensure the logging is set up to capture errors and info messages
# in the console or a log file.
# Note: The actual processing logic can be modified as per requirements.
# Ensure to handle exceptions and log errors appropriately.
# This script processes files in a specified folder, reading their contents,
# converting them to uppercase, and returning the processed data.
# It also logs errors if any issues occur during file processing.

# The script can be run from the command line with the folder path as an argument.
# It will output the processed file names and their contents in JSON format.
# Make sure to have the necessary permissions to read files in the specified folder.        
# The script is designed to be run in a Python environment with access to the specified folder.
# Ensure to have the necessary permissions to read files in the specified folder.
# The script is designed to be run in a Python environment with access to the specified folder.
# Ensure to have the necessary permissions to read files in the specified folder.           

