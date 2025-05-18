import os
import json
from pathlib import Path
from nylas import Client


class NylasService:
    def __init__(self):
        self.client = Client(
            api_key=os.environ.get("NYLAS_API_KEY"),
            api_uri=os.environ.get("NYLAS_API_URI"),
        )

    def load_webhook_data(self, uuid):
        """Load webhook data from a JSON file in the events directory."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        events_dir = base_dir / "requests" / "events"
        file_path = events_dir / f"{uuid}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"No webhook data found for UUID: {uuid}")

        with open(file_path, "r") as f:
            return json.load(f)

    def download_attachment(self, attachment_id, grant_id, message_id):
        """Download an attachment from Nylas."""
        # Get the attachment using the grant_id and message_id
        attachment = self.client.attachments.find(
            identifier=grant_id,
            attachment_id=attachment_id,
            query_params={"message_id": message_id},
        )

        # Download the file content
        file_content = self.client.attachments.download_bytes(
            identifier=grant_id,
            attachment_id=attachment_id,
            query_params={"message_id": message_id},
        )

        return {
            "content": file_content,
            "content_type": attachment.data.content_type,
            "filename": attachment.data.filename,
        }
