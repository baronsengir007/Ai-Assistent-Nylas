import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "app"))

from services.nylas_service import NylasService

nylas_service = NylasService()

webhook_data = nylas_service.load_webhook_data("fdb7c494-2b2e-11f0-86ef-21276343affe")

# Get attachment info from the webhook data
attachment = webhook_data["data"]["object"]["attachments"][0]
message = webhook_data["data"]["object"]

result = nylas_service.download_attachment(
    attachment_id=attachment["id"],
    grant_id=attachment["grant_id"],
    message_id=message["id"],
)
