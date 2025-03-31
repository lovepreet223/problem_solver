from dotenv import load_dotenv
from typing import Union
from starlette.datastructures import UploadFile as StarletteUploadFile
import hashlib
import colorsys
import tempfile
import zipfile
from typing import Tuple, Any
from io import StringIO
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from typing import Dict
from datetime import timedelta
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from typing import Optional
import urllib.parse
import os
import base64
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import TimeoutError
import tabula
from datetime import datetime
import gzip
from collections import defaultdict
import pandas as pd
from fuzzywuzzy import process
import json
from youtube_transcript_api import YouTubeTranscriptApi
import re
import requests
from PIL import Image
from io import BytesIO
import numpy as np

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GOOGLE_API_TOKEN = os.getenv("GOOGLE_API_TOKEN")
VERCEL_TOKEN = os.environ["VERCEL_TOKEN"]
DOCKER_TOKEN=os.environ["DOCKER_TOKEN"]
DOCKER_USERNAME=os.environ["DOCKER_USERNAME"]
DOCKER_REPO_NAME=os.environ["DOCKER_REPO_NAME"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
BBC_TOKEN=os.environ["BBC_TOKEN"]


def question57(question: str):
    # Extract image URL
    url_match = re.search(r'(https?://\S+)', question)
    if not url_match:
        raise ValueError("Image URL not found in the question.")
    image_url = url_match.group(0)

    # Extract mapping data
    mapping_pattern = re.compile(r'(\d)\s+(\d)\s+(\d)\s+(\d)')
    mapping = []

    for match in mapping_pattern.finditer(question):
        orig_row, orig_col, scram_row, scram_col = map(int, match.groups())
        mapping.append(((scram_row, scram_col), (orig_row, orig_col)))

    # Download image
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("Failed to download image.")

    img = Image.open(BytesIO(response.content))
    piece_size = img.width // 5  # Assuming a 5x5 grid

    # Create an empty image
    reconstructed = Image.new('RGB', (img.width, img.height))

    # Rearrange pieces
    for (scram_row, scram_col), (orig_row, orig_col) in mapping:
        box = (scram_col * piece_size, scram_row * piece_size,
               (scram_col + 1) * piece_size, (scram_row + 1) * piece_size)
        piece = img.crop(box)

        new_pos = (orig_col * piece_size, orig_row * piece_size)
        reconstructed.paste(piece, new_pos)

    # Save the reconstructed image
    #reconstructed.save("reconstructed.png")
    buffered = BytesIO()
    reconstructed.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return img_base64


def question56(question):
    # Extract YouTube URL using regex
    url_pattern = r'https://youtu\.be/[a-zA-Z0-9_-]+'
    url_match = re.search(url_pattern, question)

    if not url_match:
        raise ValueError("No YouTube URL found in the question")

    video_url = url_match.group(0)
    video_id = video_url.split('/')[-1]

    # Extract time window using regex
    time_pattern = r'(\d+\.?\d*)\s*and\s*(\d+\.?\d*)\s*seconds'
    time_match = re.search(time_pattern, question)

    if not time_match:
        raise ValueError("No time window found in the question")

    start_time = float(time_match.group(1))
    end_time = float(time_match.group(2))

    try:
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Filter transcript entries within the time window
        relevant_segments = [
            entry['text'] for entry in transcript
            if start_time <= entry['start'] <= end_time
        ]

        # Join the segments with spaces
        result = ' '.join(relevant_segments)
        return result

    except Exception as e:
        raise Exception(f"Error transcribing video: {str(e)}")


def question55(question):
    # Pattern to match the SQL query description format
    pattern = r"posts IDs after ([\d\-T:\.]+Z) with at least (\d+) comment with (\d+) useful stars, sorted\. The result should be a table with a single column called (\w+), and the relevant"

    match = re.search(pattern, question)
    if match:
        date = match.group(1)
        comments_count = int(match.group(2))
        stars_count = int(match.group(3))
        column_name = match.group(4)

        return f'''SELECT DISTINCT {column_name}
FROM social_media, 
LATERAL UNNEST(json_extract(comments, '$')::JSON[]) AS comment
WHERE timestamp >= '{date}'
AND (json_extract(comment, '$.stars.useful')::INT) > {stars_count}
ORDER BY post_id;'''

    raise ValueError(f"Could not extract parameters from query description: {question}")


def count_key_occurrences(data, target_key):
    """
    Recursively count occurrences of target_key in nested JSON.
    """
    count = 0

    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                count += 1
            count += count_key_occurrences(value, target_key)

    elif isinstance(data, list):
        for item in data:
            count += count_key_occurrences(item, target_key)

    return count

# Load JSON file
def question54(question, file):
    pattern = r"many times does (\w+) appear as"
    match = re.search(pattern, question)
    target_key = match.group(1)

    temp_path=None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    with open(file_path, "r", encoding="utf-8") as log_data:
        log_data = json.load(log_data)

    occurrences = count_key_occurrences(log_data, target_key)

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
    return str(occurrences)


def question52(question, file):
    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")


    # Pattern to match the query format
    pattern = r"many units of (\w+) were sold in (\w+) on transactions with at least (\d+) units\?"

    # Search for matches in the query
    match = re.search(pattern, question)
    product = match.group(1)
    country = match.group(2)
    units = int(match.group(3))

    # Load dataset
    df = pd.read_json(file_path)

    # List of known correct city names
    known_cities = ["Dhaka", "Tokyo", "Karachi", "Istanbul", "Chongqing", "Kinshasa", "Tianjin"]

    # Function to correct city names using fuzzy matching
    def correct_city_name(city):
        match, score = process.extractOne(city, known_cities)
        return match if score > 80 else city  # Use threshold of 80 for accuracy

    # Apply city name correction
    df["corrected_city"] = df["city"].apply(correct_city_name)

    # Filter sales entries for Bacon with at least 191 units
    bacon_sales = df[(df["product"] == product) & (df["sales"] >= units)]

    # Aggregate sales by corrected city names
    aggregated_sales = bacon_sales.groupby("corrected_city")["sales"].sum().reset_index()

    # Get sales for Dhaka
    dhaka_sales = aggregated_sales[aggregated_sales["corrected_city"] == country]

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
    ll = dhaka_sales["sales"].values[0] if not dhaka_sales.empty else 0
    return str(ll)


def question51(question, file):

    pattern = r'all\s+requests\s+under\s+([^/]+)/\s+on\s+(\d{4}-\d{2}-\d{2})'

    match = re.search(pattern, question, re.IGNORECASE)

    path = "/" + match.group(1).strip() + "/"
    date = match.group(2)
    date = datetime.strptime(date, "%Y-%m-%d")
    date = date.strftime("%d/%b/%Y")

    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    # Regular expression to parse log entries
    log_pattern = re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<datetime>\d{2}/May/2024:\d{2}:\d{2}:\d{2}) [+-]\d{4}\] '
        r'"(?P<method>\S+) (?P<url>\S+) \S+" (?P<status>\d{3}) (?P<size>\S+)'
    )

    bandwidth_usage = defaultdict(int)  # Dictionary to store bytes per IP

    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                ip = match.group("ip")
                timestamp = match.group("datetime")
                method = match.group("method")
                url = match.group("url")
                status = match.group("status")
                size = match.group("size")

                # Check if the request is on 23-May-2024 and URL starts with /hindimp3/
                if date in timestamp and url.startswith(path):
                    # Ensure it's a valid size and successful request
                    if size.isdigit() and 200 <= int(status) < 300:
                        bandwidth_usage[ip] += int(size)

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
    # Find the IP with the highest bandwidth usage
    if bandwidth_usage:
        top_ip = max(bandwidth_usage, key=bandwidth_usage.get)
        return str(bandwidth_usage[top_ip])
    else:
        return None, 0  # No valid data found


def question50(question, file):
    # Extract path between / and /
    path_pattern = r'pages under /([^/]+)/'
    path_match = re.search(path_pattern, question)
    target_path = path_match.group(1) if path_match else None

    # Extract time range (HH:MM format) and get only the hours
    time_pattern = r'from (\d{2}):\d{2} until before (\d{2}):\d{2} on'
    time_match = re.search(time_pattern, question)
    start_hour = int(time_match.group(1)) if time_match else None
    end_hour = int(time_match.group(2)) if time_match else None

    # Extract day
    day_pattern = r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
    day_match = re.search(day_pattern, question)
    target_day = day_match.group(1) if day_match else None

    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    # Define the regex pattern to parse the Apache log format
    log_pattern = re.compile(
        r'(?P<ip>\S+) (?P<remote_logname>\S+) (?P<remote_user>\S+) \[(?P<time>[^\]]+)\] '
        r'"(?P<method>\S+) (?P<url>\S+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+) '
        r'"(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)" (?P<vhost>\S+) (?P<server>\S+)'
    )

    # Extract time window
    target_path = f"/{target_path}/"
    success_range = range(200, 300)
    count = 0

    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as log_file:
        for line in log_file:
            match = log_pattern.match(line)
            if match:
                log_data = match.groupdict()

                # Extract request details
                method = log_data['method']
                url = log_data['url']
                status = int(log_data['status'])

                # Check for GET request and URL match
                if method == "GET" and target_path in url and status in success_range:

                    # Parse time and convert to datetime
                    log_time = datetime.strptime(log_data['time'].split()[0], "%d/%b/%Y:%H:%M:%S")
                    log_day = log_time.strftime("%A")
                    log_hour = log_time.hour

                    # Check for Friday and the time range
                    if log_day == target_day and start_hour <= log_hour < end_hour:
                        count += 1

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return str(count)


def question49(question, file):
    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    student_id_pattern = re.compile(r'-\s*([A-Z0-9]{3,})\b')  # Match IDs with at least 3 characters
    unique_student_ids = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = student_id_pattern.search(line)
            if match:
                student_id = match.group(1).strip()
                unique_student_ids.add(student_id)
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
    return str(len(unique_student_ids))


def question46(question, file):
    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    # Extract subject names using regex
    subject_pattern = r'Calculate the total (\w+) marks of students who scored (\d+) or more marks in (\w+)'
    subject_match = re.search(subject_pattern, question)

    if not subject_match:
        return "Could not extract subject information from the question"

    target_subject = subject_match.group(1)  # Biology
    score_threshold = int(subject_match.group(2))  # 27
    filter_subject = subject_match.group(3)  # Physics

    # Extract group range
    group_pattern = r'groups (\d+)-(\d+)'
    group_match = re.search(group_pattern, question)

    if not group_match:
        return "Could not extract group range from the question"

    group_start = int(group_match.group(1))  # 62
    group_end = int(group_match.group(2))  # 96

    try:
        # Read the PDF file
        dfs = tabula.read_pdf(file_path, pages='all')

        # Combine all dataframes
        all_data = []
        for i, df in enumerate(dfs):
            # Skip the header row that contains "Student marks - Group X"
            if 'Maths' in df.columns:
                # Add group number based on the page number
                group_num = i + 1  # Groups start from 1
                df['Group'] = group_num
                all_data.append(df)

        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert marks to numeric, handling any non-numeric values
        combined_df[filter_subject] = pd.to_numeric(combined_df[filter_subject], errors='coerce')
        combined_df[target_subject] = pd.to_numeric(combined_df[target_subject], errors='coerce')

        # Filter based on conditions
        filtered_df = combined_df[
            (combined_df[filter_subject] >= score_threshold) &
            (combined_df['Group'].between(group_start, group_end))
            ]

        # Calculate total marks
        total_marks = filtered_df[target_subject].sum()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return str(total_marks)

    except Exception as e:
        return f"Error processing the data: {str(e)}"


def create_session_with_retries():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def question45(question):
    """
    Sets up a GitHub workflow for daily updates, triggers it, and verifies completion.
    Returns the repository URL only when everything is successful.
    """
    # Extract email using regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, question)
    if not email_match:
        return "Error: No email found in the question"

    email = email_match.group(0)

    # GitHub API headers
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Create session with retry logic
    session = create_session_with_retries()

    try:
        # Get authenticated user's username
        user_response = session.get('https://api.github.com/user', headers=headers, timeout=10)
        user_response.raise_for_status()
        owner = user_response.json()['login']

        # Repository details
        repo_name = "temp5"

        # First check if repository exists
        check_repo_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        response = session.get(check_repo_url, headers=headers, timeout=10)

        if response.status_code == 404:
            # Repository doesn't exist, create it
            create_repo_url = "https://api.github.com/user/repos"
            repo_data = {
                "name": repo_name,
                "private": False,
                "auto_init": True
            }

            response = session.post(create_repo_url, headers=headers, json=repo_data, timeout=10)
            response.raise_for_status()

        repo_url = response.json()['html_url']

        # Create workflow file
        workflow_content = f'''name: Daily Repository Update

on:
  schedule:
    - cron: '0 0 * * *'  # Runs at 00:00 UTC every day
  workflow_dispatch:  # Allow manual triggering

permissions:
  contents: write
  pull-requests: write

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
        with:
          token: ${{{{ secrets.GITHUB_TOKEN }}}}

      - name: Configure Git for {email}
        run: |
          git config --global user.name 'GitHub Actions'
          git config --global user.email '{email}'

      - name: Create daily update commit for {email}
        run: |
          echo "Daily update: $(date)" > daily_update.txt
          git add daily_update.txt
          git commit -m "Daily automated update"
          git push
'''

        # First, check if the workflow file exists
        workflows_dir = ".github/workflows"
        workflow_path = f"{workflows_dir}/daily_update.yml"
        check_file_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{workflow_path}"

        try:
            # Try to get the existing file
            file_response = session.get(check_file_url, headers=headers, timeout=10)
            if file_response.status_code == 200:
                # File exists, get its SHA
                sha = file_response.json()['sha']
            else:
                sha = None
        except requests.exceptions.RequestException:
            sha = None

        # Create or update the workflow file
        create_file_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{workflow_path}"

        # Encode content in base64
        content_bytes = workflow_content.encode('utf-8')
        content_base64 = base64.b64encode(content_bytes).decode('utf-8')

        file_data = {
            "message": "Update daily update workflow",
            "content": content_base64,
            "branch": "main"
        }

        # If file exists, include its SHA
        if sha:
            file_data["sha"] = sha

        file_response = session.put(create_file_url, headers=headers, json=file_data, timeout=10)
        file_response.raise_for_status()

        # Get workflow ID
        workflows_url = f"https://api.github.com/repos/{owner}/{repo_name}/actions/workflows"
        workflows_response = session.get(workflows_url, headers=headers, timeout=10)
        workflows_response.raise_for_status()
        workflows = workflows_response.json()['workflows']

        if not workflows:
            return "Error: No workflows found in repository"

        workflow_id = workflows[0]['id']

        # Trigger workflow manually
        trigger_url = f"https://api.github.com/repos/{owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"
        trigger_data = {"ref": "main"}
        trigger_response = session.post(trigger_url, headers=headers, json=trigger_data, timeout=10)
        trigger_response.raise_for_status()

        # Wait for workflow completion with better timeout handling
        start_time = time.time()
        max_wait_time = 300  # 5 minutes
        check_interval = 5  # seconds

        while time.time() - start_time < max_wait_time:
            try:
                # Check workflow status with timeout
                runs_url = f"https://api.github.com/repos/{owner}/{repo_name}/actions/workflows/{workflow_id}/runs"
                runs_response = session.get(runs_url, headers=headers, timeout=10)
                runs_response.raise_for_status()
                runs = runs_response.json()['workflow_runs']

                if runs:
                    latest_run = runs[0]
                    status = latest_run['conclusion']

                    if status == 'success':
                        # Verify commit was created with timeout
                        commits_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits"
                        commits_response = session.get(commits_url, headers=headers, timeout=10)
                        commits_response.raise_for_status()
                        commits = commits_response.json()

                        if commits:
                            latest_commit = commits[0]
                            commit_message = latest_commit['commit']['message']

                            if "Daily automated update" in commit_message:
                                return repo_url

                    elif status in ['failure', 'cancelled']:
                        # Get workflow run details with timeout
                        run_details_url = f"https://api.github.com/repos/{owner}/{repo_name}/actions/runs/{latest_run['id']}"
                        run_details_response = session.get(run_details_url, headers=headers, timeout=10)
                        run_details_response.raise_for_status()
                        run_details = run_details_response.json()

                        error_msg = f"Error: Workflow {status}\n"
                        if 'jobs' in run_details:
                            for job in run_details['jobs']:
                                if job['conclusion'] == 'failure':
                                    error_msg += f"Job '{job['name']}' failed\n"
                                    if 'steps' in job:
                                        for step in job['steps']:
                                            if step['conclusion'] == 'failure':
                                                error_msg += f"  Step '{step['name']}' failed\n"

                        return error_msg

                time.sleep(check_interval)

            except TimeoutError:
                # If we get a timeout, continue waiting but log it
                print(f"Timeout checking workflow status, continuing to wait...")
                time.sleep(check_interval)
                continue
            except requests.exceptions.RequestException as e:
                # For other request errors, try to continue but log the error
                print(f"Error checking workflow status: {str(e)}")
                time.sleep(check_interval)
                continue

        return "Error: Workflow timed out"

    except requests.exceptions.RequestException as e:
        error_msg = f"Error: {str(e)}"
        if hasattr(e.response, 'text'):
            error_msg += f"\nResponse: {e.response.text}"
        return error_msg
    finally:
        session.close()


def question44(question):
    # Extract location and follower count using regex
    location_pattern = r'in the city\s*(\w+)'
    followers_pattern = r'with over\s*(\d+)\s*followers'

    location = re.search(location_pattern, question)
    followers = re.search(followers_pattern, question)

    if not location or not followers:
        raise ValueError("Could not extract location or follower count from question")

    location = location.group(1)
    min_followers = int(followers.group(1))

    # GitHub API endpoint for user search
    base_url = "https://api.github.com/search/users"

    # Get GitHub token from environment variable
    github_token = GITHUB_TOKEN
    if not github_token:
        return "GitHub token not found. Please set GITHUB_TOKEN environment variable."

    # Headers for authentication
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Construct query
    query = f"location:{location} followers:>{min_followers}"

    # Parameters for the API request
    params = {
        "q": query,
        "sort": "joined",
        "order": "desc",
        "per_page": 1
    }

    try:
        # Make API request for search
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()

        if not data.get("items"):
            return "No users found matching the criteria"

        # Get the newest user's login
        newest_user_login = data["items"][0]["login"]

        # Fetch complete user details
        user_url = f"https://api.github.com/users/{newest_user_login}"
        user_response = requests.get(user_url, headers=headers)
        user_response.raise_for_status()

        user_data = user_response.json()

        # Check if created_at exists in the response
        if "created_at" not in user_data:
            return "User data does not contain creation date"

        created_at = user_data["created_at"]

        # Convert to datetime to check if it's too new
        created_datetime = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        cutoff_date = datetime.strptime("2025-03-30T20:56:50Z", "%Y-%m-%dT%H:%M:%SZ")

        if created_datetime > cutoff_date:
            return "User is too new (joined after 2025-03-30)"

        return str(created_at)

    except requests.exceptions.RequestException as e:
        return f"Error accessing GitHub API: {str(e)}"
    except ValueError as e:
        return f"Error parsing date: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def question43(question: str) -> Optional[str]:
    topic_match = re.search(r'mentioning\s+([A-Za-z][A-Za-z\s]*?)(?=\s+and\s+having)', question)
    points_match = re.search(r'(\d+)\s+points', question)

    if not topic_match or not points_match:
        return None

    topic = topic_match.group(1).strip()
    min_points = int(points_match.group(1))


    # Try multiple endpoints in order of relevance
    endpoints = [
        "https://hnrss.org/newest",
        "https://hnrss.org/frontpage",
        "https://hnrss.org/best"
    ]

    for base_url in endpoints:

        # URL encode the topic for safety
        encoded_topic = urllib.parse.quote(topic)

        params = {
            "q": encoded_topic,
            "search_attrs": "default",  # Search in all attributes
            "count": 50,  # Get maximum allowed posts
            "description": 1  # Include description to get more context
        }

        try:
            # Make request to HNRSS API
            response = requests.get(base_url, params=params)
            response.raise_for_status()


            # Parse XML response
            root = ET.fromstring(response.content)

            # Find all items and get the most recent one with sufficient points
            items = root.findall(".//item")
            if items:
                for item in items:
                    link = item.find("link")
                    title = item.find("title")
                    description = item.find("description")
                    if link is not None and description is not None:
                        # Extract points from description
                        points_text = description.text
                        points_match = re.search(r'Points: (\d+)', points_text)
                        if points_match:
                            points = int(points_match.group(1))
                            if points >= min_points:
                                return str(link.text)

        except Exception as e:
            continue

    return None


def question42(question: str) -> Optional[float]:
    country_pattern = r"country\s+(\w+)"
    country_match = re.search(country_pattern, question, re.IGNORECASE)

    if not country_match:
        return None

    country = country_match.group(1)

    # Extract city name (assuming it's mentioned before "in the country")
    city_pattern = r"city\s+(\w+)\s+in the country"
    city_match = re.search(city_pattern, question, re.IGNORECASE)

    if not city_match:
        return None

    city = city_match.group(1)

    # Construct Nominatim API URL
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{city}, {country}",
        "format": "json",
        "limit": 1
    }

    # Add User-Agent header as required by Nominatim's terms of service
    headers = {
        "User-Agent": "UrbanRideGeospatial/1.0"
    }

    try:
        # Make request to Nominatim API
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()

        # Parse response
        data = response.json()

        if not data:
            return None

        # Get boundingbox from first result
        boundingbox = data[0].get("boundingbox")
        if not boundingbox:
            return None

        # Convert to float and return minimum latitude
        # boundingbox format is [min_lat, max_lat, min_lon, max_lon]
        return str(float(boundingbox[0]))

    except (requests.RequestException, ValueError, IndexError, KeyError) as e:
        print(f"Error: {str(e)}")
        return None

    finally:
        # Respect Nominatim's rate limiting policy
        time.sleep(1)


def question41(question: str) -> Dict[str, str]:
    city_match = re.search(r'What is the JSON weather forecast description for (\w+)\?', question)
    if not city_match:
        raise ValueError("Could not extract city name from the question")

    city = city_match.group(1)

    # Get location ID for the city
    location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
        'api_key': BBC_TOKEN,
        's': city,
        'stack': 'aws',
        'locale': 'en',
        'filter': 'international',
        'place-types': 'settlement,airport,district',
        'order': 'importance',
        'a': 'true',
        'format': 'json'
    })


    try:
        # Get location data
        location_response = requests.get(location_url)
        location_response.raise_for_status()
        location_data = location_response.json()

        if not location_data.get('response', {}).get('results', {}).get('results'):
            raise ValueError(f"Could not find location data for {city}")

        location_id = location_data['response']['results']['results'][0]['id']

        # Get weather data from BBC Weather website
        weather_url = f'https://www.bbc.com/weather/{location_id}'

        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(weather_response.content, 'html.parser')

        # Find weather summary
        daily_summary = soup.find('div', attrs={'class': 'wr-day-summary'})
        if not daily_summary:
            raise ValueError("Could not find weather summary data")

        # Extract weather descriptions
        daily_summary_list = re.findall('[a-zA-Z][^A-Z]*', daily_summary.text)

        # Generate dates
        current_date = datetime.now()
        datelist = [current_date + timedelta(days=i) for i in range(14)]
        datelist = [date.strftime('%Y-%m-%d') for date in datelist]

        # Create forecast dictionary
        forecast_dict = {}
        for date, description in zip(datelist, daily_summary_list):
            forecast_dict[date] = description

        return json.dumps(forecast_dict, indent=2)

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching weather data: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")


def clean_title(title: str) -> str:
    """Remove numbering and clean up the title."""
    # Remove numbering at the start (e.g., "1. ", "2. ")
    title = re.sub(r'^\d+\.\s*', '', title)
    return title.strip()


def question39(question: str) -> List[Dict[str, str]]:

    # Extract rating range using regex
    rating_pattern = r"rating between (\d+) and (\d+)"
    rating_match = re.search(rating_pattern, question)

    if not rating_match:
        raise ValueError("Rating range not found in the question")

    min_rating = float(rating_match.group(1))
    max_rating = float(rating_match.group(2))

    # IMDb search URL with rating filter
    base_url = "https://www.imdb.com/search/title/"
    params = {
        "title_type": "feature",
        "user_rating": f"{min_rating},{max_rating}",
        "count": 100,  # Increased to get more movies
        "sort": "user_rating,desc",
        "view": "simple"  # Use simple view for better scraping
    }

    # Construct full URL
    url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument(
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Set page load timeout
        driver.set_page_load_timeout(30)

        # Remove navigator.webdriver flag
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        driver.get(url)

        # Wait for the content to load
        wait = WebDriverWait(driver, 20)

        # Wait for any of the possible movie item selectors
        selectors = [
            "div.lister-item",
            "div.ipc-metadata-list-summary-item",
            "div[data-testid='title-list-item']",
            "div[class*='lister-item']",
            "div[class*='ipc-metadata']"
        ]

        for selector in selectors:
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                break
            except TimeoutException:
                continue

        # Add a longer delay to ensure all content is loaded
        time.sleep(8)

        # Scroll the page to load more content
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Get the page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Try multiple ways to find movie items
        movie_items = []
        for selector in selectors:
            items = soup.select(selector)
            if items:
                movie_items = items
                break

        if not movie_items:

            # Try to find any links that might be movie titles
            movie_links = soup.find_all('a', href=re.compile(r'tt\d+'))
            if movie_links:
                movie_items = [link.parent for link in movie_links]

        # Extract movie information
        movies = []
        seen_ids = set()  # Track unique movie IDs

        for index, item in enumerate(movie_items, 1):  # Removed limit to get more movies
            try:

                # Try different ways to find the movie ID
                id_element = (
                        item.find('a', href=re.compile(r'tt\d+')) or
                        item.find('a', {'data-testid': re.compile(r'title-.*')})
                )
                movie_id = id_element['href'].split('/')[2] if id_element and 'href' in id_element.attrs else None

                # Skip if we've already seen this movie ID
                if movie_id in seen_ids:
                    continue

                seen_ids.add(movie_id)

                # Try different ways to find the title
                title_element = (
                        item.find('h3', class_='lister-item-header') or
                        item.find('a', {'data-testid': re.compile(r'title-.*')}) or
                        item.find('h3') or
                        item.find('a', href=re.compile(r'tt\d+'))
                )
                title = title_element.text if title_element else None
                title = clean_title(title) if title else None

                # Try different ways to find the year
                year_element = (
                        item.find('span', class_='lister-item-year') or
                        item.find('span', class_='sc-43986a27-8') or
                        item.find('span', string=re.compile(r'\d{4}'))  # Updated to use string parameter
                )
                year = re.search(r'\d{4}', year_element.text).group() if year_element else None

                # Try different ways to find the rating
                rating_element = (
                        item.find('div', class_='ratings-imdb-rating') or
                        item.find('span', class_='rating-rating') or
                        item.find('span', {'data-testid': 'rating-score'}) or
                        item.find('span', string=re.compile(r'\d+\.\d+'))  # Updated to use string parameter
                )
                rating = rating_element.text if rating_element else None
                if rating:
                    rating = re.search(r'\d+\.\d+', rating).group()

                if all([movie_id, title, year, rating]):
                    movies.append({
                        "id": movie_id,
                        "title": title,
                        "year": year,
                        "rating": rating
                    })

                    # Break if we have 25 unique movies
                    if len(movies) >= 25:
                        break
                else:
                    pass
            except Exception as e:
                continue

        return json.dumps(movies, indent=2)

    except Exception as e:
        return []

    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                pass


def question38(question):
    # Extract page number and URL using regex
    page_number = int(re.search(r'page number (\d+)', question).group(1))
    base_url = re.search(r'https://[^\s<>"]+', question).group(0)
    url = f"{base_url}&page={page_number}"

    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

    try:
        # Add a small delay to avoid rate limiting
        time.sleep(1)

        # Fetch and parse the webpage with headers
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all tables and print their classes for debugging
        all_tables = soup.find_all('table')

        # Try to find the correct table that contains the statistics
        table = None
        for t in all_tables:
            if t.get('class') and 'engineTable' in t.get('class'):
                # Look for the table that contains the actual statistics
                # It should have a header row with column names like 'Player', 'Span', etc.
                header_row = t.find('tr', {'class': 'headlinks'})
                if header_row:
                    header_text = header_row.get_text().lower()
                    if any(keyword in header_text for keyword in ['player', 'span', 'mat', 'inns', 'runs']):
                        table = t
                        break

        if not table:
            raise ValueError("Could not find the statistics table on the page")

        # Clean the table HTML by removing unwanted elements
        for unwanted in table.find_all(['div', 'span']):
            unwanted.decompose()

        # Convert table to DataFrame using StringIO to avoid deprecation warning
        table_html = str(table)
        df = pd.read_html(StringIO(table_html))[0]

        # Clean column names and data
        df.columns = [str(col).strip() for col in df.columns]

        # Try different possible column names for ducks
        possible_duck_columns = ['0', '0s', 'Ducks', 'Duck', 'Zeroes']
        ducks_column = None

        for col_name in possible_duck_columns:
            if col_name in df.columns:
                ducks_column = col_name
                break

        if not ducks_column:
            # If no exact match, try to find a column containing '0' or 'duck'
            for col in df.columns:
                if '0' in col or 'duck' in col.lower():
                    ducks_column = col
                    break

        if not ducks_column:
            raise ValueError("Could not find the ducks column in the table")

        # Convert the ducks column to numeric, replacing any non-numeric values with 0
        df[ducks_column] = pd.to_numeric(df[ducks_column], errors='coerce').fillna(0)

        total_ducks = int(df[ducks_column].sum())

        return str(total_ducks)

    except requests.exceptions.RequestException as e:
        if response.status_code == 403:
            raise Exception(
                "Access denied. The website might be blocking automated requests. Please try using a different method or check if the website is accessible.")
        raise Exception(f"Error fetching data: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")


def question34(question):
    code = """
    import numpy as np
    from typing import List, Tuple, Union, Dict

    def most_similar(embeddings: Dict[str, List[float]]) -> Tuple[str, str]:
        if not embeddings:
            raise ValueError("No embeddings provided")

        if len(embeddings) < 2:
            raise ValueError("At least two embeddings are required to calculate similarity")

        # Convert all embedding vectors to numpy arrays once
        embedding_vectors = {phrase: np.array(vector) for phrase, vector in embeddings.items()}

        max_similarity = -1
        most_similar_pair = None

        # Calculate cosine similarity between each pair of embeddings
        phrases = list(embedding_vectors.keys())
        for i in range(len(phrases)):
            for j in range(i + 1, len(phrases)):
                vec1 = embedding_vectors[phrases[i]]
                vec2 = embedding_vectors[phrases[j]]

                # Calculate cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                # Update if this is the highest similarity so far
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (phrases[i], phrases[j])

        return most_similar_pair
        """
    return code


def question33(question: str) -> Dict[str, Any]:
    # Extract verification messages using regex
    message_pattern = r'Dear user, please verify your transaction code \d+ sent to [^\n]+'
    messages = re.findall(message_pattern, question)

    if len(messages) != 2:
        raise ValueError("Expected exactly 2 verification messages in the question")

    json_data = {
        "model": "text-embedding-3-small",
        "input": [messages[0],messages[1]]
    }

    return json.dumps(json_data, indent=2)


def question32(question: str) -> str:
    # Extract URL using regex
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    urls = re.findall(url_pattern, question)

    if not urls:
        raise ValueError("No URL found in the question")

    image_url = urls[0]
    print(image_url)

    # Download the image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
    except requests.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

    # Convert image to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')

    # Construct the OpenAI API request
    request_json = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    return json.dumps(request_json, indent=2)


def question31(question: str) -> List[Tuple[str, str]]:
    # Pattern to match: field_name (type)
    pattern = r'(\w+)\s*\((\w+)\)'
    # Find all matches in the string
    matches = re.findall(pattern, question)

    json_data = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "Respond in JSON"
    },
    {
      "role": "user",
      "content": "Generate 10 random addresses in the US"
    }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "addresses_response",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "addresses": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "apartment": {
                  "type": "string"
                },
                "longitude": {
                  "type": "number"
                },
                "county": {
                  "type": "string"
                }
              },
              "required": ["apartment", "longitude", "county"],
              "additionalProperties": False
            }
          }
        },
        "required": ["addresses"],
        "additionalProperties": False
      }
    }
  }
}

    fields = {
                matches[0][0]: {
                  "type": matches[0][1]
                },
                matches[1][0]: {
                  "type": matches[1][1]
                },
                matches[2][0]: {
                  "type": matches[2][1]
                }
              }
    required = [matches[0][0],matches[1][0],matches[2][0]]
    json_data["response_format"]["json_schema"]["schema"]["properties"]["addresses"]["items"]["properties"] = fields
    json_data["response_format"]["json_schema"]["schema"]["properties"]["addresses"]["items"]["required"] = required

    return json.dumps(json_data, indent=2)


def question30(question):
    # Step 1: Extract the user message using regex
    match = re.search(r'this user message:(.*?)\.\.\. how many', question, re.DOTALL)
    if match:
        user_message = match.group(1).strip()
    else:
        return "User message extraction failed!"

    # Step 2: Prepare the prompt for OpenAI's API
    prompt = user_message

    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = OPENAI_API_KEY

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(api_url, headers=headers, json=payload)

    return str(response.json()['usage']['prompt_tokens'])


def question29(question):
    # Define the regex pattern to extract the meaningless text between the specified phrases
    pattern = r"meaningless text:\s*(.*?)\s*(?=write a python|$)"

    # Use regex to search for the meaningless text in the question
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        meaningless_text = match.group(1).strip()  # Extract the text and remove extra spaces
        meaningless_text = re.sub(r'[^\w\s]*$', '', meaningless_text)

        req = '''import httpx
import json

# Set up the dummy API key
api_key = "your-dummy-api-key-here"

# Set up the headers for the request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# Set up the data payload for the POST request
data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Analyze the sentiment of the following text. Classify the sentiment as GOOD, BAD, or NEUTRAL."},
        {"role": "user", "content": {meaningless_text}}
    ]
}

# Send the POST request
with httpx.Client() as client:
    response = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

# Print the response
if response.status_code == 200:
    result = response.json()
    sentiment = result['choices'][0]['message']['content']
    print(f"Sentiment: {sentiment}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
'''
        req = req.replace("{meaningless_text}", meaningless_text)
        return req


def question25(question):
    # Extract email using regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, question)

    if not email_match:
        return "No email found in the question"

    email = email_match.group(0)

    # GitHub API configuration
    github_token = GITHUB_TOKEN
    if not github_token:
        return "GITHUB_TOKEN environment variable not found"

    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        # Get authenticated user
        user_response = requests.get('https://api.github.com/user', headers=headers)
        user_response.raise_for_status()
        user_data = user_response.json()
        owner = user_data['login']

        # Check for existing repositories with temp-repo prefix
        repos_url = 'https://api.github.com/user/repos'
        repos_response = requests.get(repos_url, headers=headers)
        repos_response.raise_for_status()
        existing_repos = repos_response.json()

        # Find existing temp repository
        temp_repo = None
        for repo in existing_repos:
            if repo['name'].startswith('temp3'):
                temp_repo = repo
                break

        if temp_repo:
            # Use existing repository
            repo_name = temp_repo['name']
            print(f"Using existing repository: {repo_name}")
        else:
            # Create a new repository
            repo_name = "temp3"
            print(f"Creating new repository: {repo_name}")

            create_repo_data = {
                "name": repo_name,
                "private": False,
                "auto_init": True,
                "description": "Temporary repository for GitHub Action test",
                "gitignore_template": "Python",
                "license_template": "mit"
            }

            try:
                response = requests.post(repos_url, headers=headers, json=create_repo_data)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 422:
                    # Try without auto_init if it fails
                    create_repo_data.pop("auto_init", None)
                    response = requests.post(repos_url, headers=headers, json=create_repo_data)
                    response.raise_for_status()
                else:
                    raise

        # Create workflow file content
        workflow_content = {
            "name": "Custom Workflow",
            "on": {
                "workflow_dispatch": {}
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": email,
                            "run": "echo 'Hello, world!'"
                        }
                    ]
                }
            }
        }

        # Create or update workflow file
        workflow_path = f'.github/workflows/custom_workflow.yml'
        create_file_url = f'https://api.github.com/repos/{owner}/{repo_name}/contents/{workflow_path}'

        # Check if workflow file exists
        try:
            existing_file = requests.get(create_file_url, headers=headers)
            if existing_file.status_code == 200:
                # Update existing file
                file_data = existing_file.json()
                create_file_data = {
                    "message": "Update custom workflow",
                    "content": base64.b64encode(json.dumps(workflow_content, indent=2).encode('utf-8')).decode('utf-8'),
                    "sha": file_data['sha']
                }
            else:
                # Create new file
                create_file_data = {
                    "message": "Create custom workflow",
                    "content": base64.b64encode(json.dumps(workflow_content, indent=2).encode('utf-8')).decode('utf-8')
                }
        except requests.exceptions.RequestException:
            # Create new file if check fails
            create_file_data = {
                "message": "Create custom workflow",
                "content": base64.b64encode(json.dumps(workflow_content, indent=2).encode('utf-8')).decode('utf-8')
            }

        response = requests.put(create_file_url, headers=headers, json=create_file_data)
        response.raise_for_status()

        # Wait for the workflow file to be available
        time.sleep(5)  # Give GitHub some time to process the file

        # Verify workflow file exists before triggering
        workflow_check_url = f'https://api.github.com/repos/{owner}/{repo_name}/contents/.github/workflows/custom_workflow.yml'
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts:
            try:
                workflow_response = requests.get(workflow_check_url, headers=headers)
                if workflow_response.status_code == 200:
                    # Now trigger the workflow
                    dispatch_url = f'https://api.github.com/repos/{owner}/{repo_name}/actions/workflows/custom_workflow.yml/dispatches'
                    dispatch_data = {"ref": "main"}

                    response = requests.post(dispatch_url, headers=headers, json=dispatch_data)
                    response.raise_for_status()
                    break
                time.sleep(2)
                attempt += 1
            except requests.exceptions.RequestException:
                time.sleep(2)
                attempt += 1

        # Verify repository is accessible
        repo_url = f"https://github.com/{owner}/{repo_name}"
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts:
            try:
                response = requests.get(repo_url, headers=headers)
                if response.status_code == 200:
                    # Additional verification: check if workflow file exists
                    workflow_check_url = f'https://api.github.com/repos/{owner}/{repo_name}/contents/.github/workflows/custom_workflow.yml'
                    workflow_response = requests.get(workflow_check_url, headers=headers)
                    if workflow_response.status_code == 200:
                        return repo_url
                time.sleep(2)  # Wait before next attempt
                attempt += 1
            except requests.exceptions.RequestException:
                time.sleep(2)
                attempt += 1

        return f"Repository created but verification failed: {repo_url}"

    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_message = f"Error details: {error_details}"
            except:
                pass
        return f"Error occurred: {error_message}"


def question23(question):
    # Extract URL using regex - updated pattern to exclude closing parenthesis
    url_pattern = r'https?://[^\s<>")]+|www\.[^\s<>")]+'
    urls = re.findall(url_pattern, question)

    if not urls:
        raise ValueError("No URL found in the question")

    image_url = urls[0]
    print(image_url)

    # Download the image
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

    # Convert image data to PIL Image
    image = Image.open(BytesIO(response.content))

    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array and normalize
    rgb = np.array(image) / 255.0

    # Calculate lightness using colorsys
    lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)

    # Count pixels with lightness > 0.109
    light_pixels = np.sum(lightness > 0.109)

    return str(light_pixels)


def question21(question):
    """
    Process the GitHub Pages question using GitHub REST API directly.
    Creates/updates repository and waits for the site to be published.
    Returns the GitHub Pages URL once the site is accessible.
    """
    # Extract email from the question using regex
    email_pattern = r'<!--email_off-->(.*?)<!--/email_off-->'
    email_match = re.search(email_pattern, question)
    if not email_match:
        return "Error: Email not found in the question"

    email = email_match.group(1)

    # Get GitHub token from environment variable
    github_token = GITHUB_TOKEN
    if not github_token:
        return "Error: GITHUB_TOKEN environment variable not set"

    # GitHub API headers
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        username = GITHUB_USERNAME
        repo_name = "temp1"

        # Check if repository exists
        repo_url = f"https://api.github.com/repos/{username}/{repo_name}"
        response = requests.get(repo_url, headers=headers)

        if response.status_code == 404:
            # Create new repository
            create_repo_data = {
                "name": repo_name,
                "auto_init": True,
                "private": False
            }
            response = requests.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json=create_repo_data
            )
            if response.status_code != 201:
                return f"Error creating repository: {response.text}"

        # Create index.html content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>My GitHub Pages</title>
        </head>
        <body>
            <p><!--email_off-->{email}<!--/email_off--></p>
        </body>
        </html>
        """

        # Check if index.html exists
        try:
            response = requests.get(f"{repo_url}/contents/index.html", headers=headers)
            if response.status_code == 200:
                # Update existing file
                file_data = response.json()
                update_data = {
                    "message": "Update index.html",
                    "content": base64.b64encode(html_content.encode()).decode(),
                    "sha": file_data["sha"]
                }
                response = requests.put(
                    f"{repo_url}/contents/index.html",
                    headers=headers,
                    json=update_data
                )
            else:
                # Create new file
                create_data = {
                    "message": "Create index.html",
                    "content": base64.b64encode(html_content.encode()).decode()
                }
                response = requests.put(
                    f"{repo_url}/contents/index.html",
                    headers=headers,
                    json=create_data
                )
        except Exception as e:
            return f"Error updating index.html: {str(e)}"

        # Create GitHub Pages workflow with GitHub's official deploy-pages action
        workflow_content = """
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Checkout
        uses: actions/checkout@v2
      - name: Setup Pages
        uses: actions/configure-pages@v2
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v1
        with:
          path: '.'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v1
        """

        # Create workflow file
        workflow_data = {
            "message": "Create GitHub Pages workflow",
            "content": base64.b64encode(workflow_content.encode()).decode()
        }

        try:
            response = requests.put(
                f"{repo_url}/contents/.github/workflows/pages.yml",
                headers=headers,
                json=workflow_data
            )
        except Exception as e:
            return f"Error creating workflow: {str(e)}"

        # Wait for GitHub Pages to be published
        print("Waiting for GitHub Pages to be published...")
        url = f"https://{username}.github.io/{repo_name}/"

        # First, check if the workflow is running
        print("Checking GitHub Actions status...")
        for attempt in range(30):  # Check workflow status for 30 seconds
            try:
                response = requests.get(f"{repo_url}/actions/workflows", headers=headers)
                if response.status_code == 200:
                    workflows = response.json()
                    if workflows.get('workflows'):
                        print("GitHub Actions workflow is running...")
                        break
            except:
                pass
            time.sleep(1)

        # Now wait for the site to be published
        print("Waiting for site to be accessible...")
        for attempt in range(120):  # 120 attempts, 5 seconds each = 10 minutes total
            try:
                # Check GitHub Pages status through API
                response = requests.get(f"{repo_url}/pages", headers=headers)
                if response.status_code == 200:
                    pages_status = response.json()
                    if pages_status.get('status') == 'built':
                        # Check if the site is actually accessible
                        site_response = requests.get(url, timeout=10)
                        if site_response.status_code == 200:
                            print("Site is published and accessible!")
                            return url
                        else:
                            print(f"Site built but not accessible yet (Status: {site_response.status_code})")
                    else:
                        print(f"Pages status: {pages_status.get('status')}")
            except Exception as e:
                print(f"Error checking status: {str(e)}")
            time.sleep(5)

        return "Error: GitHub Pages failed to publish within the expected time (10 minutes)"

    except Exception as e:
        return f"Error: {str(e)}"


def question18(question: str) -> str:
    # Extract table name
    table_match = re.search(r'\b(\w+)\s+table\b', question, re.IGNORECASE)
    table_name = table_match.group(1) if table_match else "tickets"

    # Extract column names
    columns_match = re.search(r'columns\s+([\w,\s]+)', question, re.IGNORECASE)
    columns = columns_match.group(1).split(',') if columns_match else ["type", "units", "price"]

    # Extract item name (ticket type in this case)
    item_match = re.search(r'\"(.*?)\"', question)
    item_name = item_match.group(1) if item_match else "Gold"

    # Construct the SQL query
    sql_query = f'''
    SELECT SUM(units * price) AS total_sales
    FROM {table_name}
    WHERE LOWER(TRIM({columns[0]})) = LOWER(TRIM("{item_name}"));
    '''

    return sql_query.strip()


def question17(question, file: Union[str, StarletteUploadFile]) -> int:
    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    # Extract ZIP file
    extract_path = "extracted_files"
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Read a.txt and b.txt
    a_path = os.path.join(extract_path, "a.txt")
    b_path = os.path.join(extract_path, "b.txt")

    with open(a_path, 'r', encoding='utf-8') as file_a, open(b_path, 'r', encoding='utf-8') as file_b:
        a_lines = file_a.readlines()
        b_lines = file_b.readlines()

    # Compare lines
    differing_lines = sum(1 for a, b in zip(a_lines, b_lines) if a != b)

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return str(differing_lines)


def question15(question: str, file: Union[StarletteUploadFile, str]):
    # Extract file size and date-time from question using regex
    size_match = re.search(r"(\d+) bytes", question)
    date_match = re.search(r"(\w{3}, \d{2} \w{3}, \d{4}, \d{2}:\d{2} (?:am|pm) IST)", question)

    if not size_match or not date_match:
        raise ValueError("Could not extract file size or date-time from the question")

    min_size = int(size_match.group(1))
    min_date = datetime.strptime(date_match.group(1), "%a, %d %b, %Y, %I:%M %p IST")
    print(min_size)
    print(min_date)

    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    extracted_folder = "extracted_files"
    os.makedirs(extracted_folder, exist_ok=True)

    total_size = 0

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

        for file_info in zip_ref.infolist():
            extracted_file_path = os.path.join(extracted_folder, file_info.filename)
            if os.path.isfile(extracted_file_path):
                file_size = os.path.getsize(extracted_file_path)
                mod_time = datetime(*file_info.date_time)

                if file_size >= min_size and mod_time >= min_date:
                    total_size += file_size

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return str(total_size)


def question14(question, file):
    # Determine file path
    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    # Create a new folder for extraction
    extract_dir = file_path.rstrip(".zip") + "_unzipped"
    os.makedirs(extract_dir, exist_ok=True)

    # Extract ZIP file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Regular expression pattern to match "IITM" in all case permutations
    pattern = re.compile(r"(?i)\bIITM\b")  # `(?i)` makes it case-insensitive

    # Replace in all files
    for root, _, files in os.walk(extract_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Read file content while preserving encoding and line endings
            with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                content = f.read()

            # Replace all case permutations of "IITM" with "IIT Madras"
            new_content = pattern.sub("IIT Madras", content)

            # Write back to the file
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(new_content)

    # Compute SHA-256 hash equivalent to `cat * | sha256sum`
    hash_sha256 = hashlib.sha256()
    for root, _, files in os.walk(extract_dir):
        for filename in sorted(files):  # Sorting ensures consistent hash
            with open(os.path.join(root, filename), "rb") as f:
                while chunk := f.read(8192):
                    hash_sha256.update(chunk)
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return hash_sha256.hexdigest()


def question13(question: str):
    # Extract filename and JSON content from the question
    match = re.search(r'called\s+([a-zA-Z0-9_.]+)\s+with the value\s+(\{.*?\})', question)
    if not match:
        raise ValueError(f"Could not extract required information from the question: {question}")

    filename = match.group(1)  # Extract filename
    json_data = match.group(2)  # Extract JSON string
    try:
        json_content = json.loads(json_data)  # Convert string to dictionary
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format extracted: {json_data}")

    print(f"Extracted filename: {filename}")  # Debugging
    print(f"Extracted JSON content: {json_content}")  # Debugging

    # GitHub details
    github_token = GITHUB_TOKEN
    github_id = GITHUB_USERNAME
    repo_name = "temp0"

    if not github_token:
        raise ValueError("GITHUB_TOKEN is not set in environment variables.")

    # GitHub API headers
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Step 1: Create a new public repository
    create_repo_url = "https://api.github.com/user/repos"
    repo_data = {
        "name": repo_name,
        "private": False,
        "auto_init": True  # Initialize with a README
    }

    response = requests.post(create_repo_url, headers=headers, json=repo_data)
    if response.status_code not in [201, 422]:  # 422 if repo already exists
        raise Exception(f"Failed to create repository: {response.text}")

    # Step 2: Create a new file in the repository
    repo_path = f"{github_id}/{repo_name}"
    file_url = f"https://api.github.com/repos/{repo_path}/contents/{filename}"
    file_content = json.dumps(json_content, indent=4)

    file_data = {
        "message": f"Add {filename}",
        "content": base64.b64encode(file_content.encode("utf-8")).decode("utf-8"),  # ✅ FIXED: Base64 Encoding
        "branch": "main"
    }

    response = requests.put(file_url, headers=headers, json=file_data)
    if response.status_code not in [201, 200]:
        raise Exception(f"Failed to commit file: {response.text}")

    # Step 3: Ensure the file is publicly accessible
    raw_url = f"https://raw.githubusercontent.com/{repo_path}/main/{filename}"
    max_retries = 10
    for _ in range(max_retries):
        time.sleep(5)  # Wait for GitHub to process the file
        check_response = requests.get(raw_url)
        if check_response.status_code == 200:
            return raw_url

    raise Exception("File did not become publicly accessible within the expected time frame.")


def question12(question, file):
    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    total_sum = 0
    extracted_files = []
    temp_dir = "temp_extracted_files"

    # If file is a ZIP, extract its contents
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)  # Extract to temp directory
            extracted_files = [os.path.join(temp_dir, f) for f in zip_ref.namelist()]
    else:
        extracted_files.append(file_path)

    # Encoding and delimiter settings based on file name
    encoding_map = {
        "data1.csv": "cp1252",
        "data2.csv": "utf-8",
        "data3.txt": "utf-16"
    }
    delimiter_map = {
        "data1.csv": ",",
        "data2.csv": ",",
        "data3.txt": "\t"
    }

    # Process each extracted file
    for file_path in extracted_files:
        file_name = os.path.basename(file_path)
        encoding = encoding_map.get(file_name, "utf-8")  # Default to UTF-8
        delimiter = delimiter_map.get(file_name, ",")  # Default to comma

        # Load the file into a DataFrame
        try:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")

        # Define the regex pattern to match the required symbols
        pattern = r"[„•…]"

        # Filter rows where the symbol matches the regex
        df_filtered = df[df.iloc[:, 0].astype(str).str.contains(pattern, regex=True, na=False)]

        # Sum up the values
        total_sum += df_filtered.iloc[:, 1].sum()
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return str(total_sum)


def question11(html: str) -> int:
    # Extract the HTML body dynamically using regex to find the full HTML content
    match = re.search(r'<div.*</div>', html, re.DOTALL)
    if not match:
        return 0  # Return 0 if no valid HTML body is found

    html_body = match.group(0)

    # Extract the class name using regex
    class_match = re.search(r'Find all <div>s having a ([a-zA-Z0-9_-]+) class', html)
    if not class_match:
        return 0  # Return 0 if no class name is found

    class_name = class_match.group(1)

    # Parse the HTML
    soup = BeautifulSoup(html_body, 'html.parser')

    # Find all divs with the specified class
    divs = soup.find_all('div', class_=lambda c: c and class_name in c.split())

    # Extract and sum the data-value attributes
    total = sum(int(div['data-value']) for div in divs if div.has_attr('data-value'))

    return str(total)


def question10(question: str, file: Union[StarletteUploadFile, str]) -> str:
    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
            temp_file.write(file.file.read().decode("utf-8"))
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Convert key=value pairs to dictionary
    data = {}
    for line in content.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()

    # Convert to JSON string
    json_data = json.dumps(data, separators=(',', ':'))

    # Compute hash using SHA256 (to simulate JSON Hash function behavior)
    hash_value = hashlib.sha256(json_data.encode("utf-8")).hexdigest()

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return hash_value


def question9(question: str):
    # Extract sorting keys using improved regex
    match = re.findall(r'by the value of the (\w+)|sort by the (\w+)', question)
    keys = [key for pair in match for key in pair if key]
    print(keys)

    # Ensure only valid keys (i.e., ones appearing in JSON) are used
    json_match = re.search(r'(\[.*\])', question, re.DOTALL)
    if not json_match:
        return "No JSON found in the input."

    json_data = json.loads(json_match.group(1))

    # Filter keys to remove any accidental words (like 'the')
    valid_keys = [k for k in keys if all(k in obj for obj in json_data)]

    if not valid_keys:
        return "No valid sorting keys found in the question."

    # Sort JSON based on extracted and validated keys
    sorted_data = sorted(json_data, key=lambda x: tuple(x[k] for k in valid_keys))

    return json.dumps(sorted_data, separators=(',', ':'))


def question8(question: str, file: Union[StarletteUploadFile, str]) -> str:
    # Extract the column name using regex
    match = re.search(r'value in the\s+(.*?)\s+column of the', question, re.IGNORECASE)
    if not match:
        raise ValueError("Could not determine column name from the question")
    column_name = match.group(1).strip().strip('"').strip("'")

    temp_path = None
    if isinstance(file, StarletteUploadFile):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        file.file.seek(0)  # Reset pointer
        file_path = temp_zip_path  # Use the temp file for processing
        temp_path = file_path
    elif isinstance(file, str):
        file_path = file
    else:
        raise ValueError("Unsupported file type")

    # Unzip and extract CSV
    with zipfile.ZipFile(file_path, 'r') as z:
        csv_file = [f for f in z.namelist() if f.endswith('.csv')][0]  # Assuming a single CSV
        with z.open(csv_file) as f:
            df = pd.read_csv(f)

    # Return the value in the specified column
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file")

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return df[column_name].iloc[0]  # Assuming we need the first row value


def question7(question):
    # Regex pattern to extract the day of the week and date range
    pattern = re.search(r"how many (\w+?)s? .*?(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", question, re.IGNORECASE)

    if not pattern:
        return "Could not extract valid parameters from the question."

    day = pattern.group(1).capitalize()  # Capitalize for consistency
    start_date = pattern.group(2)
    end_date = pattern.group(3)

    # Convert day string to corresponding weekday index (0 = Monday, 6 = Sunday)
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if day not in weekdays:
        return "Invalid day"

    day_index = weekdays.index(day)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    count = sum(1 for d in range((end - start).days + 1) if (start + timedelta(days=d)).weekday() == day_index)

    return str(count)


def question6(question_html: str) -> str:
    """
    Extracts the value of the hidden input field from the given HTML string.

    :param question_html: A string containing HTML.
    :return: The value of the hidden input field if found, otherwise an empty string.
    """
    # Extract only the HTML part if the input contains extra text
    html_start = question_html.find("<div")
    if html_start != -1:
        question_html = question_html[html_start:]

    soup = BeautifulSoup(question_html, 'html.parser')
    hidden_input = soup.find('input', {'type': 'hidden'})

    if hidden_input and 'value' in hidden_input.attrs:
        return hidden_input['value']
    return ""


def question5(question: str):
    # Extract only numbers inside curly braces using regex
    matches = re.findall(r'\{([^}]*)\}', question)

    if len(matches) < 2:
        raise ValueError("Could not find two numeric lists inside curly braces.")

    # Convert extracted string lists to integer lists
    array_values = list(map(int, matches[0].split(',')))
    sort_order = list(map(int, matches[1].split(',')))

    if len(array_values) != 16 or len(sort_order) != 16:
        raise ValueError("Extracted lists do not contain 16 elements each.")

    # Sort array based on the given order
    sorted_pairs = sorted(zip(sort_order, array_values))
    sorted_array = [x[1] for x in sorted_pairs]  # Extract sorted values

    # Take first row (all 16 elements) and sum them
    result = sum(sorted_array)

    return str(result)


def question4(question: str):
    # Extract numbers using regex
    numbers = list(map(int, re.findall(r'\d+', question)))

    if len(numbers) < 4:
        raise ValueError("Not enough numerical values found in the question.")

    rows, cols, start, step = numbers[:4]  # Extract first 4 values

    # Generate the sequence matrix
    sequence_matrix = np.array([[(start + j * step) for j in range(cols)] for i in range(rows)])

    # Apply ARRAY_CONSTRAIN logic (take first row and first 10 columns)
    constrained_array = sequence_matrix[0, :10]  # First row, first 10 elements

    # Compute sum
    result = np.sum(constrained_array)

    return str(result)


def question2(question):
    # Extract URL using regex
    url_pattern = r'https://[^\s]+'
    url_match = re.search(url_pattern, question)
    if not url_match:
        raise ValueError("No URL found in the question")
    url = url_match.group(0)

    # Extract parameter name and value
    param_pattern = r'parameter\s+(\w+)\s+set\s+to\s+([^\s]+)'
    param_match = re.search(param_pattern, question)
    if not param_match:
        raise ValueError("No parameter found in the question")
    param_name = param_match.group(1)
    param_value = param_match.group(2)

    try:
        response = requests.get(url, params={param_name: param_value})
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()  # Return parsed JSON response

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

