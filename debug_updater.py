import sys
import logging
from pyupdater.client import Client

# Configure logging to verify what's happening
logging.basicConfig(level=logging.DEBUG)

class AppConfig(object):
    APP_NAME = "BG Remover"
    COMPANY_NAME = "Meeran Rashith"
    HTTP_TIMEOUT = 30
    MAX_DOWNLOAD_RETRIES = 3
    UPDATE_URLS = ["https://raw.githubusercontent.com/meeranrashith166-lang/BG-Remover/main/updates/"]
    PUBLIC_KEY = 'djeV18vHUKwPhFXxHL8BX+Q6SsqsQXe8PoEDuker95A'

print("Attempting to initialize Client...")
try:
    client = Client(AppConfig(), refresh=True)
    print("Client initialized successfully.")
    print(f"Update Check Result: {client.update_check(AppConfig.APP_NAME, '1.0.0')}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
