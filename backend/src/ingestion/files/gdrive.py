"""
Author: Pranav Iyer
Date: 2024-10-02
Description: This script demonstrates how to ingest GDrive files using the Google Drive API.
"""
import sys
import os
import io
import re
import googleapiclient
import googleapiclient.discovery
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry import FunctionRegistry
from src.utils.datetime_utils import get_current_utc_datetime, parse_datetime
from src.utils.ingestion_utils import update_ingestion_with_metadata
from src.schemas.schemas import (
    ChunkingMethod,
    ContentType,
    Document,
    Entry,
    FileType,
    Ingestion,
    IngestionMethod,
    ParsedFeatureType,
    ParsingMethod,
    Scope,
    mime_type_to_file_type
)

def sanitize_filename(filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    # Remove invalid characters and replace some with underscores
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove any non-ASCII characters
    name = re.sub(r'[^\x00-\x7F]+', '_', name)
    # Truncate name if it's too long (adjust max_length as needed)
    max_length = 255 - len(ext)  # Reserve space for the extension
    if len(name) > max_length:
        name = name[:max_length]
    # Combine the sanitized name with the original extension
    sanitized_filename = name.strip() + ext
    return sanitized_filename

def get_service(api: str, version: str) -> googleapiclient.discovery.Resource:
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive', 'https://mail.google.com/']
    creds = None
    if os.path.exists(os.environ.get("GMAIL_TOKEN_PATH")):  # The file token.json stores the user's access and refresh tokens.
        creds = Credentials.from_authorized_user_file(os.environ.get("GMAIL_TOKEN_PATH"), SCOPES)
    if not creds or not creds.valid:  # If there are no (valid) credentials available, let the user log in.
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(os.environ.get("GMAIL_CREDENTIAL_PATH"), SCOPES)
            creds = flow.run_local_server(port=0)
        assert (creds) is not None, "Credentials are None"
        with open("token.json", "w") as token:  # Save the credentials for the next run
            token.write(creds.to_json())
    assert (creds) is not None, "Credentials are None"
    service = build(api, version, credentials=creds)
    assert (type(service)) == googleapiclient.discovery.Resource, "Service is not a googleapiclient.discovery.Resource"
    return service

def get_folder_id(service: googleapiclient.discovery.Resource, folder_name: str, parent_id: str = 'root') -> Optional[str]:
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id != 'root':
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, spaces='drive', fields="files(id, name)").execute()
    items = results.get('files', [])
    
    if not items:
        return None
    return items[0]['id']

def get_shared_drive_id(service: googleapiclient.discovery.Resource, drive_name: str) -> Optional[str]:
    results = service.drives().list(fields="drives(id, name)").execute()
    for drive in results.get('drives', []):
        if drive['name'] == drive_name:
            return drive['id']
    return None

def get_folder_contents(service: googleapiclient.discovery.Resource, folder_id: str, is_shared_drive: bool = False) -> list[dict]:
    results = []
    page_token = None
    while True:
        try:
            params = {
                'q': f"'{folder_id}' in parents",
                'spaces': 'drive',
                'fields': 'nextPageToken, files(id, name, mimeType, createdTime, modifiedTime)',
                'pageToken': page_token
            }
            if is_shared_drive:
                params.update({
                    'includeItemsFromAllDrives': True,
                    'supportsAllDrives': True,
                    'corpora': 'drive',
                    'driveId': folder_id
                })
            
            response = service.files().list(**params).execute()
            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken')
            if not page_token:
                break
        except googleapiclient.errors.HttpError as error:
            if error.resp.status == 404 and is_shared_drive:
                print(f"Shared Drive not found: {folder_id}")
                return []
            else:
                print(f"An error occurred: {error}")
                raise
    return results

def download_file(service: googleapiclient.discovery.Resource, file_id: str, file_name: str, output_dir: str):
    try:
        request = service.files().get_media(fileId=file_id)
        sanitized_file_name = sanitize_filename(file_name)
        file_path = os.path.join(output_dir, sanitized_file_name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        print(f"Downloaded: {file_path}")
    except googleapiclient.errors.HttpError as error:
        if error.resp.status == 403 and "Use Export with Docs Editors files" in str(error):
            try:
                # This is a Google Workspace file, so we need to export it
                file_metadata = service.files().get(fileId=file_id, fields='mimeType').execute()
                mime_type = file_metadata['mimeType']
                
                if 'document' in mime_type:
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    file_extension = '.docx'
                elif 'spreadsheet' in mime_type:
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    file_extension = '.xlsx'
                elif 'presentation' in mime_type:
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    file_extension = '.pptx'
                else:
                    print(f"Unsupported Google Workspace file type: {mime_type}")
                    return

                request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
                file_path = os.path.join(output_dir, f"{os.path.splitext(sanitize_filename(file_name))[0]}{file_extension}")
                with io.FileIO(file_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
            except Exception as e:
                print(f"An error occurred while exporting the file: {e}")
        else:
            print(f"An error occurred while downloading the file: {error}")

def format_ingestion(file_info: dict, file_path: str, added_metadata: dict) -> Ingestion:
    timestamp = get_current_utc_datetime()
    
    # Check if the file exists and get its size
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
    file_type = mime_type_to_file_type(file_info['mimeType'])
    ingestion = Ingestion(
        document_title=file_info['name'],
        scope=Scope.INTERNAL,
        content_type=ContentType.FOUNDATIONAL_DOCUMENTS,  # This might need to be adjusted based on file type
        file_type=file_type,
        file_path=file_path,
        total_length=file_size,  # Use None if file doesn't exist
        public_url=None,
        creator_name="GoogleDrive",
        creation_date=parse_datetime(file_info['createdTime']),
        metadata={
            "modifiedTime": file_info['modifiedTime'],
            "mimeType": file_info['mimeType'],
            "driveFileId": file_info['id'],
        },
        ingestion_date=timestamp,
        ingestion_method=IngestionMethod.GDRIVE_API,
        parsing_method=ParsingMethod.NONE,
        parsing_date=timestamp,
        parsed_feature_type=ParsedFeatureType.TEXT,  # This might need to be adjusted based on file type
        bounding_box=None,
        parsed_file_path=file_path,
        chunking_method=ChunkingMethod.NONE,
        chunking_metadata=None,
        unprocessed_citations=None,
        embedded_feature_type=None,
    )
    ingestion = update_ingestion_with_metadata(ingestion, added_metadata)
    return ingestion

def process_folder(service: googleapiclient.discovery.Resource, folder_id: str, folder_path: str, ingestions: list, added_metadata: dict, is_shared_drive: bool = False):
    contents = get_folder_contents(service, folder_id, is_shared_drive)
    for item in contents:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            subfolder_name = sanitize_filename(item['name'])
            subfolder_path = os.path.join(folder_path, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            process_folder(service, item['id'], subfolder_path, ingestions, added_metadata, is_shared_drive)
        else:
            file_name = sanitize_filename(item['name'])
            file_path = os.path.join(folder_path, file_name)
            download_file(service, item['id'], item['name'], folder_path)
            ingestion = format_ingestion(item, file_path, added_metadata)
            ingestions.append(ingestion)

def process_shared_drive(service: googleapiclient.discovery.Resource, drive_id: str, drive_path: str, ingestions: list, added_metadata: dict, include_root_files: bool):
    try:
        os.makedirs(drive_path, exist_ok=True)
        root_files = get_folder_contents(service, drive_id, is_shared_drive=True)
        for item in root_files:
            if item['mimeType'] != 'application/vnd.google-apps.folder':
                file_name = item['name']
                file_path = os.path.join(drive_path, sanitize_filename(file_name))
                download_file(service, item['id'], file_name, drive_path)
                ingestion = format_ingestion(item, file_path, added_metadata)
                ingestions.append(ingestion)
        
        # Process folders in Shared Drive
        folders = [item for item in root_files if item['mimeType'] == 'application/vnd.google-apps.folder']
        for folder in folders:
            folder_path = os.path.join(drive_path, sanitize_filename(folder['name']))
            os.makedirs(folder_path, exist_ok=True)
            process_folder(service, folder['id'], folder_path, ingestions, added_metadata, is_shared_drive=True)
    except Exception as e:
        print(f"An error occurred while processing the Shared Drive: {e}")

def get_root_files(service: googleapiclient.discovery.Resource, drive_id: Optional[str] = None) -> list[dict]:
    if drive_id and drive_id != 'root':
        query = "'{}' in parents and mimeType != 'application/vnd.google-apps.folder'".format(drive_id)
        params = {
            'q': query,
            'spaces': 'drive',
            'fields': 'nextPageToken, files(id, name, mimeType, createdTime, modifiedTime)',
            'driveId': drive_id,
            'includeItemsFromAllDrives': True,
            'supportsAllDrives': True,
            'corpora': 'drive'
        }
    else:
        query = "mimeType != 'application/vnd.google-apps.folder' and 'root' in parents"
        params = {
            'q': query,
            'spaces': 'drive',
            'fields': 'nextPageToken, files(id, name, mimeType, createdTime, modifiedTime)'
        }

    results = []
    page_token = None
    while True:
        if page_token:
            params['pageToken'] = page_token
        try:
            response = service.files().list(**params).execute()
            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken')
            if not page_token:
                break
        except googleapiclient.errors.HttpError as error:
            print(f"An error occurred: {error}")
            break
    return results

def process_shared_item(service, item, shared_path, shared_with_me_files, added_metadata, ingestions):
    file_name = item['name']
    if not shared_with_me_files or file_name in shared_with_me_files:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Process shared folder
            folder_path = os.path.join(shared_path, sanitize_filename(file_name))
            os.makedirs(folder_path, exist_ok=True)
            process_folder(service, item['id'], folder_path, ingestions, added_metadata)
        else:
            # Process shared file
            file_path = os.path.join(shared_path, sanitize_filename(file_name))
            download_file(service, item['id'], file_name, shared_path)
            ingestion = format_ingestion(item, file_path, added_metadata)
            ingestions.append(ingestion)

@FunctionRegistry.register("ingest", "gdrive")
def ingest(folders: list[str], shared_drives: list[str] = [], include_shared_with_me: bool = False, shared_with_me_files: list[str] = [], include_root_files: bool = True, added_metadata: dict = {}) -> list[Entry]:
    service = get_service("drive", "v3")
    ingestions = []
    output_dir = "gdrive_downloads"
    os.makedirs(output_dir, exist_ok=True)

    # Process root files in My Drive
    if include_root_files:
        root_files = get_root_files(service)
        root_path = os.path.join(output_dir, "My Drive Root")
        os.makedirs(root_path, exist_ok=True)
        for item in root_files:
            file_name = item['name']
            file_path = os.path.join(root_path, file_name)
            download_file(service, item['id'], file_name, root_path)
            ingestion = format_ingestion(item, file_path, added_metadata)
            ingestions.append(ingestion)

    # Process regular folders
    for folder_name in folders:
        if folder_name.lower() == "my drive":
            continue  # Skip "My Drive" as it's not a real folder
        folder_id = get_folder_id(service, folder_name)
        if folder_id:
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            process_folder(service, folder_id, folder_path, ingestions, added_metadata)
        else:
            print(f"Folder '{folder_name}' not found.")

    # Process Shared Drives
    for drive_name in shared_drives:
        drive_id = get_shared_drive_id(service, drive_name)
        if drive_id:
            drive_path = os.path.join(output_dir, f"Shared Drive - {drive_name}")
            process_shared_drive(service, drive_id, drive_path, ingestions, added_metadata, include_root_files)
        else:
            print(f"Shared Drive '{drive_name}' not found.")

    # Process "Shared with me" items
    if include_shared_with_me:
        shared_path = os.path.join(output_dir, "Shared with me")
        os.makedirs(shared_path, exist_ok=True)
        query = "sharedWithMe=true"
        page_token = None
        while True:
            results = service.files().list(q=query, 
                                             spaces='drive', 
                                             fields="nextPageToken, files(id, name, mimeType, createdTime, modifiedTime)",
                                             pageToken=page_token).execute()
            for item in results.get('files', []):
                process_shared_item(service, item, shared_path, shared_with_me_files, added_metadata, ingestions)
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break

    return ingestions


if __name__ == '__main__':
    folder_names = []
    include_root_files = False

    shared_with_me_files = []
    include_shared_with_me = False

    shared_drive_names = ["2.DataCollection-Recommendations"]
    results = ingest(folder_names, shared_drives=shared_drive_names, shared_with_me_files=shared_with_me_files, include_shared_with_me=include_shared_with_me, include_root_files=include_root_files)
