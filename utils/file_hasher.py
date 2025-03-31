import hashlib
import zipfile
import json
import os
import tempfile
import gzip
from starlette.datastructures import UploadFile as StarletteUploadFile

async def compute_hash(file):
    """Computes the hash of a file (UploadFile or file path)."""

    if isinstance(file, StarletteUploadFile):
        if not file.filename:
            return None
        if file.filename.lower().endswith('.zip'):
            return compute_hash_zip(file)
        if file.filename.lower().endswith('.gz'):
            return compute_hash_gz(file)
        return compute_hash_files(file)

    elif isinstance(file, str):
        if not os.path.exists(file):
            return None
        if file.lower().endswith('.zip'):
            return compute_hash_zip(file)
        if file.lower().endswith('.gz'):
            return compute_hash_gz(file)
        return compute_hash_files(file)

    return None



def compute_hash_files(file, algorithm='sha256'):
    """Computes the hash of a regular file, supporting both UploadFile and file paths."""
    hash_func = hashlib.new(algorithm)

    if isinstance(file, StarletteUploadFile):  # Ensure correct check
        file.file.seek(0)  # Reset file pointer
        content = file.file.read()
        #print(f"[DEBUG] Read {len(content)} bytes from file.")  # Debugging
        if not content:
            #print("Error: Empty file content!")
            return None

        hash_func.update(content)
        file.file.seek(0)  # Reset again for further processing if needed

    else:
        with open(file, 'rb') as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)

    return hash_func.hexdigest()


def compute_hash_gz(gz_file):
    """Computes a consistent hash for a GZ file (ignoring metadata)."""
    hash_func = hashlib.sha256()

    # Handle FastAPI UploadFile
    if isinstance(gz_file, StarletteUploadFile):
        gz_file.file.seek(0)
        with gzip.GzipFile(fileobj=gz_file.file, mode='rb') as gz:
            while chunk := gz.read(8192):
                hash_func.update(chunk)
        gz_file.file.seek(0)  # Reset pointer for further use if needed

    # Handle file path
    elif isinstance(gz_file, str):
        with gzip.open(gz_file, 'rb') as gz:
            while chunk := gz.read(8192):
                hash_func.update(chunk)

    return hash_func.hexdigest()


def compute_hash_zip(zip_file):
    """Computes a consistent hash for a ZIP file."""
    file_hashes = []
    temp_zip_path = None  # Track temp file path

    # If the input is an UploadFile, save it to a temporary file
    if isinstance(zip_file, StarletteUploadFile):
        zip_file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(zip_file.file.read())
            temp_file.flush()
            temp_zip_path = temp_file.name
        zip_file.file.seek(0)  # Reset pointer
        zip_file = temp_zip_path  # Use the temp file for processing

    # Ensure the file is a valid ZIP before proceeding
    if not zipfile.is_zipfile(zip_file):
        if temp_zip_path and os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)  # Delete only temp file
        return None

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_name in sorted(zip_ref.namelist()):  # Sort to maintain consistency
                with zip_ref.open(file_name) as file:
                    hash_func = hashlib.sha256()
                    while chunk := file.read(8192):
                        hash_func.update(chunk)
                    file_hashes.append(hash_func.hexdigest())

        # Compute final hash from sorted hashes
        final_hash_func = hashlib.sha256()
        for h in sorted(file_hashes):
            final_hash_func.update(h.encode())

        final_hash = final_hash_func.hexdigest()

    except Exception as e:
        final_hash = None

    # Cleanup only the temporary file, NOT the original file
    if temp_zip_path and os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)

    return final_hash




def compute_hashes_all_files(json_file="./files/files.json", output_file="./files/hashes.json"):
    """Reads JSON, computes file hashes, and saves updated JSON to a new file."""
    if not os.path.exists(json_file):
        #print("Error: JSON file not found!")
        return
    #print("Opening JSON file...")
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            #print("Error: Invalid JSON format!")
            return
    print("Started computing hashes...")
    result = {}
    for key, value in data.items():
        if value:
            file_path = f"./files/{value}"
            #print(f"Computing hash for: {file_path}")
            file_hash = compute_hash(file_path)
            if not file_hash:
                pass
                #print(f"Error: Failed to compute hash for {file_path}")
            result[key] = file_hash
        else:
            result[key] = None
    print("Writing hashes to JSON file...")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    #print("Hash computation complete!")
