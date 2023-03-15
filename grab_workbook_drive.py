from google.oauth2 import service_account
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from googleapiclient.discovery import build
import openpyxl
import io
import os
import json
from grab_all_workouts import grab_all_workouts
from config import key



def grab_workbook_from_drive (name):
    json_data = json.loads(key)

    creds = service_account.Credentials.from_service_account_file(json_data)
    

    # Authenticate with your credentials
    gauth = GoogleAuth()
    #gauth.credentials = creds
    drive = GoogleDrive(gauth)
    folder_id = '1QkTChu814mOKeZP7UKcMtUiOOfZ0RmQr'

    # Authenticate and build the Drive API client
    service = build('drive', 'v3', credentials=creds)

    # Get a list of files in the folder
    results = service.files().list(q=f"'{folder_id}' in parents and trashed=false", fields="nextPageToken, files(id, name)").execute()
    files = results.get('files', [])

    # Find the file with the desired name
    file_name = f"{name}"
    file_id = None
    for file in files:
        if file['name'] == file_name:
            file_id = file['id']
            break

    # Load the workbook from the file
    if file_id:
        file_content = service.files().export(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet').execute()
        workbook = openpyxl.load_workbook(filename=io.BytesIO(file_content))
    else:
        print(f'File "{file_name}" not found in folder "{folder_id}"')

    os.remove('temp.json')
        
    return workbook





    
    




