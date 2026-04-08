from colab import auth
auth.authenticate_user()
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
def upload_to_drive(file_path, folder_id=None):
    service = build('drive', 'v3')
    file_metadata = {'name': file_path.split('/')[-1]}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File ID: {file.get("id")}')
            