"""
Overview: Script to download the dataset from Zenodo.

This script does the following: - Downloads the files from Zenodo at ./zenodo -
Decompresses the tar.gz files containing the images for `raw`, `cropped` and
  `backgrounds` into their expected locations relative to the path of this
  repository.
- Moves the files from the decompressed annotations.tar.gz to their expected
  locations relative to the path of this repository.

Expect some longer processing time, mostly depending on your internet speed
since the download is around 23 Gb. It will decompress to around 29 Gb.

Usage: 
1. Activate the corresponding environment:
   $ source ./envs/general/bin/activate

2. Run the script from the root folder of the project:
   $ python3 ./data/code/download_zenodo_dataset.py
"""


import os
import sys
import re
import json
import subprocess
import time
from tqdm import tqdm
import git
import tarfile
import shutil


def get_user_input():
    info_url = """
    If the dataset is restricted and you have an access URL provided to you, 
    such as 
    https://zenodo.org/records/15096610?token=<SOME-LONG-TOKEN>, 
    please type "yes" when prompted below.

    If you do not have an access URL, an attempt will be made to download the dataset. 
    If the dataset is still restricted, you will be informed that you do not have 
    access at the moment. Once the dataset becomes fully public, downloading will 
    be possible without an access URL too.
    """
    print(info_url)

    has_url = input("Do you have an access URL? (yes/no): ").strip().lower()
    if has_url == 'yes':
        url = input("Enter the URL: ")
        return url
    elif has_url == 'no':
        return None
    else:
        raise ValueError("Invalid input. Please enter 'yes' or 'no'.")


def extract_record_id_and_token(url):
    """
    Uses regular expression to extract the Zenodo record ID and token from the
    provided access URL.
    
    :param url: Zenodo URL that grants access to the dataset if restricted.
                Looks like this: 
                https://zenodo.org/records/15096610?token=<SOME-LONG-TOKEN>
    """
    # Regular expression patters. Note the "records" in the access URL vs
    # "record" in the curl call.
    record_id_match = re.search(r'records/(\d+)', url)
    token_match = re.search(r'token=([^&]+)', url)
    
    if record_id_match and token_match:
        record_id = record_id_match.group(1)
        # print(f"Extracted record id from URL: {record_id}") # for debugging purposes
        token = token_match.group(1)
        # print(f"Extracted token from URL: {token}") # for debugging purposes
        return record_id, token
    else:
        raise ValueError("Invalid URL format. Please check the URL and try again.")


def run_curl_command(command):
    """
    Execute system call. It will be needed for `curl` calls.
    
    :param command: a list with the text parts that form a system call and it
                    will be evaluated.
    """
    try:
        # print(f"Executing: {command}") # for debugging purposes
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command failed with error: {e.stderr}")
    

def generate_cookies(record_id, token):
    """
    Generate the zenodo-cookies.txt needed when accessing the restricted dataset
    with "secret" URL. See https://github.com/zenodo/zenodo/issues/2003
    
    :param record_id: Zenodo record ID extracted from "secret" URL
    :param token: Zenodo token extracted from "secret" URL
    """
    cookies_file = 'zenodo-cookies.txt'
    # Note the "record" in the curl call vs "records" in the access URL.
    cookies_url = f"https://zenodo.org/record/{record_id}?token={token}"
    command = ['curl', '--cookie-jar', cookies_file, cookies_url]
    run_curl_command(command)


def fetch_record_metadata(record_id, token):
    """
    Fetch the Zenodo JSON metadata with file names and URLs to each file in the
    Zenodo repository.
    
    :param record_id: Zenodo record ID extracted from "secret" URL
    :param token: Zenodo token extracted from "secret" URL. Can also be empty.
    """
    metadata_file = 'zenodo-metadata.json'
    if token:
        metadata_url = f"https://zenodo.org/api/records/{record_id}"
        command = ['curl', '--cookie', 'zenodo-cookies.txt', metadata_url, '-o', metadata_file]
        run_curl_command(command)
    else:
        # Use the export URL for metadata when no token is provided
        metadata_url = f"https://zenodo.org/records/{record_id}/export/json"
        command = ['curl', metadata_url, '-o', metadata_file]
        run_curl_command(command)
    
    return metadata_file


def decompress_tar_gz(file_path, destination_dir):
    """
    Decompress *.tar.gz file
    
    :param file_path: Path to *.tar.gz file
    :param destination_dir: Path to the destination directory where the content will
                            be decompressed.
    """
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=destination_dir)
    print(f"Decompressed {file_path} to {destination_dir}")


def download_files(metadata_file, prj_path, token):
    """
    Download and decompress the archived files from Zenodo.
    
    :param metadata_file: JSON file with server response from Zenodo;
                          contains file names and URLs.
    :param prj_path: Absolute path to this git repository on local computer.
    :param token: Zenodo token if provided.
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Directory path where to download
    download_dir = os.path.join(prj_path, 'zenodo')
    os.makedirs(download_dir, exist_ok=True)
    
    # Directory path where to decompress
    dest_dir = os.path.join(prj_path, 'data', 'images')
    os.makedirs(dest_dir, exist_ok=True)
    
    if token:
        access_status = "with token"
        # Download files if a token is present. The JSON metadata file has a
        # certain structure in this case and might differ otherwise.
        files = metadata.get('files', [])
        
        for file_info in files:
            file_name = file_info['key']
            file_url = file_info['links']['self']
            download_path = os.path.join(download_dir, file_name)
            
            print(f"Downloading {file_name} to {download_dir}")
            command = ['curl', '--cookie', 'zenodo-cookies.txt', file_url, '--output', download_path]
            # print(f"Executing: {command}")  # for debugging purposes
            # Execute command without the use of run_curl_command() to get the
            # raw progress info of curl
            subprocess.run(command, check=True)
            
            # Decompress *.tar.gz file:
            if download_path.endswith('.tar.gz'):
                print(f"Decompressing {file_name}. This might take some moments ...")
                with tarfile.open(download_path, "r:gz") as tar:
                    tar.extractall(path=dest_dir)
            else:
                print("*.tar.gz file expected for decompression and got this instead: {file_name}")
    else:
        # Handle the case for no token required. This was not fully tested since
        # I built this function when data was restricted on Zenodo. I only
        # tested on another Zenodo repository, but one that was publicly
        # available. It is then when I assumed that the JSON metadata file might
        # differ for a public repository.
        print("Attempting without Zenodo token...")
        # Check if the access status is "restricted"
        access_status = metadata.get('access', {}).get('status', 'public')
        if access_status == "restricted":
            print("The dataset is restricted at the moment.\nDo you have a 'secret' URL?")
            return access_status # Exit the function here if dataset restricted
        
        # If not restricted, proceed with downloading files
        entries = metadata.get('files', {}).get('entries', {})
        
        for file_info in entries.values():
            # Extract file name from URL provided in 'self' field because the
            # 'key' field could end up containing something like <xyz/file_name>
            file_name = os.path.basename(file_info['links']['self'])
            # Use the 'content' link to get actual file
            file_url = file_info['links']['content']
            download_path = os.path.join(download_dir, file_name)
            
            print(f"Downloading {file_name} to {download_dir}")
            command = ['curl', file_url, '--output', download_path]
            subprocess.run(command, check=True)
            
            # Decompress *.tar.gz file:
            if download_path.endswith('.tar.gz'):
                print(f"Decompressing {file_name}. This might take some moments ...")
                with tarfile.open(download_path, "r:gz") as tar:
                    tar.extractall(path=dest_dir)
            else:
                print("*.tar.gz file expected for decompression and got this instead: {file_name}")
    return access_status


def move_contents(source_dir, destination_dir):
    """
    Move all contents from the source directory to the destination directory.
    
    :param source_dir: Path to the source directory whose contents are to be moved.
    :param destination_dir: Path to the destination directory where contents are to be placed.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Move each item from the source directory to the destination directory
    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)
        destination_item_path = os.path.join(destination_dir, item)
        shutil.move(source_item_path, destination_item_path)
    
    print(f"Contents of \n{source_dir} \nhave been moved to \n{destination_dir}\n")


def move_files(prj_path):
    """
    Files from the decompressed annotations.tar.gz need to be moved  to their
    expected locations in this repository.
    Also, move Zenodo JSON metadata file to default download dir (./zenodo).
    
    :param prj_path: Absolute path to this git repository on local computer.
    """
    # Directory where annotations.tar.gz was decompressed.
    source_dir = os.path.join(prj_path, 'data', 'images', 'annotations')
    
    # Move files from ./data/images/annotations/raw to ./data/annotations
    move_contents(os.path.join(source_dir, 'raw'),
                  os.path.join(prj_path, 'data', 'annotations'))
    
    # Move files from ./data/images/annotations/cropped/processed to ./data/processed
    move_contents(os.path.join(source_dir, 'cropped', 'processed'),
                  os.path.join(prj_path, 'data', 'processed'))
    
    # Move files from ./data/images/annotations/cropped/processed to ./data/processed
    move_contents(os.path.join(source_dir, 'cropped', 'predictions', 'yolov5s'),
                  os.path.join(prj_path, 'detectors', 'predictions', 'yolov5s'))
    
    # Forcefully remove the source directory. It should be empty after the
    # operations above.
    try:
        shutil.rmtree(source_dir)
        print(f'{source_dir} has been completely removed.')
    except OSError as e:
        print(f'Error: {source_dir} cannot be deleted. \n{e}')   
    
    # Finally, move Zenodo JSON metadata file to default download dir
    download_dir = os.path.join(prj_path, 'zenodo')
    destination_file = shutil.move('zenodo-metadata.json', download_dir)
    print(f"Moved 'zenodo-metadata.json' to {destination_file}")
    

if __name__ == "__main__":
    try:
        # Start the timer
        start_time = time.time()
        
        # Ask for access URL if user has it. This is important only during
        # Zenodo restriction. If not, attempt without it (but it will only work
        # after restriction is lifted). See also
        # https://github.com/zenodo/zenodo/issues/2003
        url = get_user_input()
        if url:
            record_id, token = extract_record_id_and_token(url)
        else:
            record_id = 15096610
            print(f"Using the default record ID {record_id} with no access token.")
            token = ""
        
        # If the token is empty, do not generate access cookies. The
        # zenodo-cookies.txt is only needed for the access URL during the data
        # restriction period. 
        if token:
            generate_cookies(record_id, token)
        
        # Fetch JSON metadata file from Zenodo
        metadata_file = fetch_record_metadata(record_id, token)
        
        # Downloading. download_files() will return an access_status as well
        prj_path = git.Repo('.', search_parent_directories=True).working_tree_dir
        access_status = download_files(metadata_file, prj_path, token)
        print(f"Access status: {access_status}")
        if access_status == "restricted":
            # Exit the script with a non-zero status if dataset is restricted
            # and no 'secret' URL was provided. 
            sys.exit(1)
        else:
            print("Download completed.")
        
        # Cleanup
        move_files(prj_path)
        
        # Remove temporary access cookies file
        if os.path.exists('zenodo-cookies.txt'):
            os.remove('zenodo-cookies.txt')
            print("Removed temporary file 'zenodo-cookies.txt'")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
