import sys
import os
import warnings

# --- DLL LOAD FIX ---
# Necessary for Python 3.8+ on Windows with PyInstaller
if sys.platform == "win32" and getattr(sys, 'frozen', False):
    try:
        # Get the real base path
        # Get the base path
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(sys.executable))
        
        # Check for _internal folder (PyInstaller 6+)
        internal_path = os.path.join(base_path, '_internal')
        search_base = internal_path if os.path.isdir(internal_path) else base_path
        
        # Explicit DLL directories to add to search path
        dll_dirs = [
            search_base,
            os.path.join(search_base, 'onnxruntime', 'capi'),
            os.path.join(search_base, 'torch', 'lib'),
            os.path.join(search_base, 'cv2'),
        ]
        
        # Log paths for debugging
        print(f"DLL Fix: search_base={search_base}")
        
        for d in dll_dirs:
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                    print(f"DLL Fix: Added {d}")
                except Exception as e:
                    print(f"DLL Fix: Failed to add {d}: {e}")
                
                # Update PATH for legacy lookups and subprocesses
                os.environ['PATH'] = d + os.pathsep + os.environ['PATH']
        
        # Set environment variable to help ONNX Runtime find its providers
        ort_capi_path = os.path.join(search_base, 'onnxruntime', 'capi')
        if os.path.isdir(ort_capi_path):
            os.environ['ORT_DYNLIB_PATH'] = ort_capi_path
            print(f"DLL Fix: Set ORT_DYNLIB_PATH to {ort_capi_path}")


    except Exception as e:
        print(f"DLL Fix Error: {e}")
# --------------------

# --- WARNING SUPPRESSION ---
# Must be done before importing libraries that might emit warnings
# Suppress specific Qt warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;qt.text.font.warning=false;qt.qpa.fonts*=false"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rembg.*")
warnings.filterwarnings("ignore", message=".*'mode' parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid: in an upcoming release.*")

# ---------------------------

import webbrowser
import struct
import numpy as np
from numpy import packbits
import cv2
import logging
import mimetypes
import hashlib
import json
from PIL import Image
import onnxruntime as ort
from rembg import remove as rembg_remove, new_session

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QSlider, QComboBox, QProgressBar, QScrollArea, QMessageBox, QSpinBox,
    QDialog, QFrame
)
import gc
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont, QKeySequence, QShortcut
from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QEvent, QTimer, QSettings
# moved up
import math
import threading
import urllib.request
try:
    import torch
    has_torch = True
except ImportError:
    torch = None
    has_torch = False
from pytoshop.user import nested_layers
from pytoshop import enums
import re





# --- IMPROVED LOGGING CONFIGURATION ---
def setup_logging():
    log_dir = os.path.join(os.getenv('LOCALAPPDATA'), 'BG Remover')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    return log_file

log_file_path = setup_logging()
# --- END IMPROVED LOGGING ---

try:
    from pyupdater.client import Client
    has_updater = True
except ImportError:
    has_updater = False

try:
    import ben2
    has_ben2 = True
except ImportError:
    has_ben2 = False



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    # 1. Try temp folder (bundled resources)
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
        possible_path = os.path.join(base_path, relative_path)
        if os.path.exists(possible_path):
            return possible_path

    # 2. Try relative to executable (external resources for onefile/onedir)
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
        # Modern PyInstaller places dependencies in _internal for ONEDIR usually, 
        # but pure external assets are usually next to the EXE.
        possible_path = os.path.join(base_path, relative_path)
        if os.path.exists(possible_path):
             return possible_path
    
    # 3. Development mode
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)
APP_VERSION = "3.0.1" # --- MODEL PATH CONFIGURATION ---
# We use LOCALAPPDATA for writable model cache to avoid Administrator requirement
USER_DATA_DIR = os.path.join(os.getenv('LOCALAPPDATA'), 'BG Remover')
os.makedirs(USER_DATA_DIR, exist_ok=True)

# The directory where rembg and other libraries will look for/save models
MODEL_CACHE_DIR = os.path.join(USER_DATA_DIR, 'models')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Force rembg to use the writable directory
os.environ["U2NET_HOME"] = MODEL_CACHE_DIR

def migrate_models():
    """Copies bundled models to the writable USER_DATA_DIR if they aren't there."""
    bundled_models_src = resource_path("models")
    if os.path.exists(bundled_models_src):
        import shutil
        for item in os.listdir(bundled_models_src):
            s = os.path.join(bundled_models_src, item)
            d = os.path.join(MODEL_CACHE_DIR, item)
            if os.path.isfile(s) and not os.path.exists(d):
                try:
                    shutil.copy2(s, d)
                    logging.info(f"Migrated model: {item}")
                except Exception as e:
                    logging.error(f"Failed to migrate model {item}: {e}")

# Run migration to ensure models are available in the writable path
migrate_models()
# --------------------------------

class AppConfig(object):
    APP_NAME = "BGRemover"
    COMPANY_NAME = "Meeran Rashith"
    UPDATE_URLS = ["https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/"]
    DATA_DIR = os.path.join(os.getenv('LOCALAPPDATA'), 'BG Remover', 'updates')
    # --- SECURITY ENHANCEMENT ---
    # Public Key for verifying updates.
    PUBLIC_KEY = '/9uGLH0xRS+mh+F7cX15ueVeP/29LpLM5ZA4LV5PynY'

try:
    import win32com.client
    has_pywin32 = True
except ImportError:
    has_pywin32 = False




ASSETS_DIR = resource_path("assets")

def create_arrow_assets():
    try:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        # Down arrows
        down_dark = os.path.join(ASSETS_DIR, "down_arrow_dark.png")
        if not os.path.exists(down_dark):
            img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.polygon([(4, 6), (12, 6), (8, 11)], fill=(136, 136, 160, 255))
            img.save(down_dark)

        down_light = os.path.join(ASSETS_DIR, "down_arrow_light.png")
        if not os.path.exists(down_light):
            img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.polygon([(4, 6), (12, 6), (8, 11)], fill=(102, 102, 136, 255))
            img.save(down_light)

        # Up arrows
        up_dark = os.path.join(ASSETS_DIR, "up_arrow_dark.png")
        if not os.path.exists(up_dark):
            img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.polygon([(4, 11), (12, 11), (8, 6)], fill=(136, 136, 160, 255))
            img.save(up_dark)

        up_light = os.path.join(ASSETS_DIR, "up_arrow_light.png")
        if not os.path.exists(up_light):
            img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.polygon([(4, 11), (12, 11), (8, 6)], fill=(102, 102, 136, 255))
            img.save(up_light)
    except Exception as e:
        print(f"Error creating arrow assets: {e}")

create_arrow_assets()

down_arrow_dark_path = os.path.join(ASSETS_DIR, "down_arrow_dark.png").replace("\\", "/")
down_arrow_light_path = os.path.join(ASSETS_DIR, "down_arrow_light.png").replace("\\", "/")
up_arrow_dark_path = os.path.join(ASSETS_DIR, "up_arrow_dark.png").replace("\\", "/")
up_arrow_light_path = os.path.join(ASSETS_DIR, "up_arrow_light.png").replace("\\", "/")

def get_model_path(relative_path):
    """Always returns the writable cache path for a model file.
    Models are downloaded on-demand if not present."""
    filename = relative_path.replace("models/", "")
    return os.path.join(MODEL_CACHE_DIR, filename)

# --- MODEL DOWNLOAD URLS (Hugging Face) ---
# Models are downloaded on first use and cached permanently in LOCALAPPDATA
MODEL_DOWNLOAD_URLS = {
    "u2net.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/u2net.onnx",
    "u2netp.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/u2netp.onnx",
    "u2net_human_seg.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/u2net_human_seg.onnx",
    "modnet.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/modnet.onnx",
    "isnet-general-use.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/isnet-general-use.onnx",
    "isnet-anime.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/isnet-anime.onnx",
    "bria_rmbg_1.4.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/bria_rmbg_1.4.onnx",
    "basnet.onnx": "https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/models/basnet.onnx",
}

def download_model_on_demand(model_name, model_path, progress_callback=None):
    """Downloads the selected model from Hugging Face if not present locally.
    Downloads once and caches permanently in LOCALAPPDATA."""
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    filename = os.path.basename(model_path)
    url = MODEL_DOWNLOAD_URLS.get(filename)
    
    if not url:
        raise FileNotFoundError(
            f"No download URL configured for model '{filename}'.\n"
            f"Please check if the model file exists at: {model_path}"
        )
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    temp_path = model_path + ".tmp"
    
    logging.info(f"Downloading model '{model_name}' from {url}")
    logging.info(f"Destination: {model_path}")
    
    def download_report(block_num, block_size, total_size):
        if progress_callback and total_size > 0:
            percent = int(block_num * block_size * 100 / total_size)
            progress_callback(min(25, int(percent * 0.25)))  # Map to 0-25% range

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=download_report)
        # Rename temp file to final path (atomic on same filesystem)
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rename(temp_path, model_path)
        logging.info(f"Successfully downloaded and cached '{filename}' ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        logging.error(f"Failed to download model '{filename}': {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(
            f"Failed to download AI model '{model_name}'.\n\n"
            f"Please check your internet connection and try again.\n"
            f"Error: {e}"
        )

MODELS = {
    "Auto (Rembg)": None,
    "U\u00b2Net": get_model_path("models/u2net.onnx"),
    "U\u00b2NetP": get_model_path("models/u2netp.onnx"),
    "U\u00b2Net Human Seg": get_model_path("models/u2net_human_seg.onnx"),
    "MODNet": get_model_path("models/modnet.onnx"),
    "IS-Net": get_model_path("models/isnet-general-use.onnx"),
    "IS-Net Anime": get_model_path("models/isnet-anime.onnx"),
    "Bria RMBG v1.4": get_model_path("models/bria_rmbg_1.4.onnx"),
    "BASNet": get_model_path("models/basnet.onnx"),

}
MODEL_HASHES = {
    get_model_path("models/u2net.onnx"): "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    get_model_path("models/u2netp.onnx"): "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
    get_model_path("models/u2net_human_seg.onnx"): "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
    get_model_path("models/modnet.onnx"): "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
    get_model_path("models/isnet-general-use.onnx"): "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    get_model_path("models/isnet-anime.onnx"): "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
    get_model_path("models/bria_rmbg_1.4.onnx"): "8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f",
    get_model_path("models/basnet.onnx"): "2766aaedd02b2e301ba3efe908f7b10455077b842c328a0fb900c5c0d8080b8a",
    get_model_path("models/birefnet.onnx"): "6470117bac6f8d82a3f62921056f52d0f5c4d36d1d832096331d5ea38a03acb5",
}
# ---------------------------------------

def verify_file_hash(file_path, expected_hash):
    """Calculates the SHA-256 of a file and compares it to an expected value."""
    if not os.path.exists(file_path):
        return False
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest() == expected_hash

# Custom exception for security-related issues
class SecurityException(Exception):
    pass
# --- END NEWLY ADDED SECTION ---

LANGUAGES = {
    "English": {"model": "Model","feather": "Feather","language": "Language","browse": "Browse","remove_bg": "Remove Background",
                "undo": "Undo","redo": "Redo","save": "Save","zoom_in": "Zoom In","zoom_out": "Zoom Out","switch_theme": "Switch Theme",
                "next": "Next", "back": "Back", "skip": "Skip", "finish": "Finish",
                "tour_welcome_title": "Welcome to BG Remover!",
                "tour_welcome_desc": "Let's take a quick tour to see how to use the software.",
                "tour_models_title": "AI Models & Feathering",
                "tour_models_desc": "Choose from various AI models for different types of images. Use 'Feather' to soften the edges of the removed background.",
                "tour_process_title": "Load & Process",
                "tour_process_desc": "Click 'Browse' to load an image, then 'Remove Background' to let the AI work its magic.",
                "tour_utils_title": "Utilities & Export",
                "tour_utils_desc": "Use Zoom, Undo/Redo, and Themes to perfect your work. Save your result in various formats including PNG and PSD."},
    "தமிழ்": {"model": "மாதிரி","feather": "மெல்லியப்பு","language": "மொழி","browse": "கோப்பு திறக்க","remove_bg": "பின்புலம் அகற்று",
              "undo": "திருத்தத்தை முன்னோக்கி","redo": "திருத்தத்தை பின்னோக்கி","save": "சேமிக்கவும்",
              "zoom_in": "பெரிதாக்கு","zoom_out": "சிறியதாக்கு","switch_theme": "தீமை மாற்றுக",
              "next": "அடுத்து", "back": "பின்பு", "skip": "தவிர்", "finish": "முடிக்க",
              "tour_welcome_title": "BG Remover-க்கு வரவேற்கிறோம்!",
              "tour_welcome_desc": "மென்பொருளை எவ்வாறு பயன்படுத்துவது என்பதைப் பற்றிய சிறிய பயணத்தை மேற்கொள்வோம்.",
              "tour_models_title": "AI மாதிரிகள் & மென்மையாக்குதல்",
              "tour_models_desc": "வெவ்வேறு வகையான படங்களுக்கு பல்வேறு AI மாதிரிகளைத் தேர்வு செய்யவும். அகற்றப்பட்ட பின்புலத்தின் விளிம்புகளை மென்மையாக்க 'Feather' பயன்படுத்தவும்.",
              "tour_process_title": "ஏற்று & செயலாக்கு",
              "tour_process_desc": "'Browse' கிளிக் செய்து ஒரு படத்தை ஏற்றவும், பின்னர் 'Remove Background' கிளிக் செய்து AI அதன் வேலையைச் செய்ய அனுமதிக்கவும்.",
              "tour_utils_title": "பயன்பாடுகள் & ஏற்றுமதி",
              "tour_utils_desc": "உங்கள் வேலையைச் செம்மைப்படுத்த ஜூம், செயல்தவிர்த்தல்/மீண்டும் செய்தல் மற்றும் தீம்களைப் பயன்படுத்தவும். PNG மற்றும் PSD உள்ளிட்ட பல்வேறு வடிவங்களில் உங்கள் முடிவைச் சேமிக்கவும்."},
    "हिन्दी": {"model": "मॉडल","feather": "फेदर","language": "भाषा","browse": "ब्राउज़ करें","remove_bg": "पृष्ठभूमि हटाएं",
              "undo": "पूर्ववत करें","redo": "पुनः करें","save": "सहेजें","zoom_in": "ज़ूम इन","zoom_out": "ज़ूम आउट","switch_theme": "थीμ बदलें",
              "next": "अगला", "back": "पीछे", "skip": "छोड़ें", "finish": "समाप्त",
              "tour_welcome_title": "BG Remover में आपका स्वागत है!",
              "tour_welcome_desc": "आइए सॉफ़्टवेयर का उपयोग करने के तरीके को जानने के लिए एक त्वरित दौरा करें।",
              "tour_models_title": "AI मॉडल और फेदरिंग",
              "tour_models_desc": "विभिन्न प्रकार की छवियों के लिए विभिन्न AI मॉडल चुनें। हटाए गए बैकग्राउंड के किनारों को नरम करने के लिए 'फेदर' का उपयोग करें।",
              "tour_process_title": "लोड और प्रोसेस",
              "tour_process_desc": "छवि लोड करने के लिए 'ब्राउज़ करें' पर क्लिक करें, फिर AI को अपना जादू दिखाने के लिए 'पृष्ठभूमि हटाएँ' पर क्लिक करें।",
              "tour_utils_title": "उपयोगिताएँ और निर्यात",
              "tour_utils_desc": "अपने काम को बेहतरीन बनाने के लिए ज़ूम, अनडू/रीडू और थीम का उपयोग करें। अपने परिणाम को PNG और PSD सहित विभिन्न स्वरूपों में सहेजें."},
    "Français": {"model": "Modèle","feather": "Doux","language": "Langue","browse": "Parcourir","remove_bg": "Supprimer fond",
                 "undo": "Annuler","redo": "Rétablir","save": "Enregistrer","zoom_in": "Zoom avant","zoom_out": "Zoom arrière","switch_theme": "Changer thème",
                 "next": "Suivant", "back": "Précédent", "skip": "Passer", "finish": "Terminer",
                 "tour_welcome_title": "Bienvenue sur BG Remover !",
                 "tour_welcome_desc": "Faisons un tour rapide pour voir comment utiliser le logiciel.",
                 "tour_models_title": "Modèles IA & Adoucissement",
                 "tour_models_desc": "Choisissez parmi différents modèles d'IA pour différents types d'images. Utilisez 'Doux' pour adoucir les bords de l'image.",
                 "tour_process_title": "Charger & Traiter",
                 "tour_process_desc": "Cliquez sur 'Parcourir' pour charger une image, puis sur 'Supprimer fond' pour laisser l'IA agir.",
                 "tour_utils_title": "Utilitaires & Export",
                 "tour_utils_desc": "Utilisez le Zoom, Annuler/Rétablir et les Thèmes. Enregistrez votre résultat dans divers formats, y compris PNG et PSD."},
    "Deutsch": {"model": "Modell","feather": "Weichheit","language": "Sprache","browse": "Durchsuchen","remove_bg": "Hintergrund entfernen",
                "undo": "Rückgängig","redo": "Wiederherstellen","save": "Speichern","zoom_in": "Vergrößern","zoom_out": "Verkleinerung","switch_theme": "Thema wechseln",
                "next": "Weiter", "back": "Zurück", "skip": "Überspringen", "finish": "Fertigstellen",
                "tour_welcome_title": "Willkommen bei BG Remover!",
                "tour_welcome_desc": "Lassen Sie uns einen kurzen Rundgang machen, um zu sehen, wie die Software funktioniert.",
                "tour_models_title": "KI-Modelle & Weichheit",
                "tour_models_desc": "Wählen Sie aus verschiedenen KI-Modellen für unterschiedliche Bildtypen. Verwenden Sie 'Weichheit', um die Kanten zu glätten.",
                "tour_process_title": "Laden & Verarbeiten",
                "tour_process_desc": "Klicken Sie auf 'Durchsuchen', um ein Bild zu laden, und dann auf 'Hintergrund entfernen', um die KI arbeiten zu lassen.",
                "tour_utils_title": "Werkzeuge & Export",
                "tour_utils_desc": "Nutzen Sie Zoom, Rückgängig/Wiederherstellen und Themen. Speichern Sie Ihr Ergebnis in verschiedenen Formaten wie PNG und PSD."}
}



def preprocess_for_model(img, model_name, session):
    img_copy = img.copy()
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) >= 4:
        _, c, h, w = input_shape[:4]
    else:
        c, h, w = 3, 320, 320

    if not isinstance(h, int) or not isinstance(w, int) or h == 0 or w == 0:
        h, w = 320, 320

    resized = cv2.resize(img_copy, (w, h), interpolation=cv2.INTER_LANCZOS4)
    arr = resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return arr


def refine_mask(mask_raw, original_img, original_size):
    """Refine a raw model mask to produce accurate, smooth edges.
    Uses sigmoid normalization, high-res upscaling, morphological cleanup,
    and a guided filter for precise edge alignment."""
    h, w = original_size  # (height, width)

    # 1. Sigmoid normalization – normalize raw logits to [0, 1] smoothly
    mask_f = mask_raw.astype(np.float64)
    m_min, m_max = mask_f.min(), mask_f.max()
    if m_max - m_min > 1e-6:
        mask_f = (mask_f - m_min) / (m_max - m_min)
    else:
        mask_f = np.zeros_like(mask_f)
    # Apply sigmoid to sharpen the transition while preserving gradients
    mask_f = 1.0 / (1.0 + np.exp(-12.0 * (mask_f - 0.5)))

    # 2. High-quality upscale to original resolution
    mask_hr = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LANCZOS4)
    mask_hr = np.clip(mask_hr, 0, 1)

    # 3. Guided filter – use the original image as guide to align mask edges
    #    with actual object boundaries (hair, fur, fine details)
    try:
        guide = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mask_guided = mask_hr.astype(np.float32)
        # Parameters: radius=8, eps=1e-4 for fine edge preservation
        radius = max(4, min(16, int(min(h, w) * 0.008)))
        eps = 1e-4
        # Use OpenCV's guided filter (ximgproc) if available, else manual box filter
        try:
            mask_guided = cv2.ximgproc.guidedFilter(guide, mask_guided, radius, eps)
        except AttributeError:
            # Fallback: bilateral filter for edge-aware smoothing
            mask_u8 = (mask_guided * 255).astype(np.uint8)
            mask_u8 = cv2.bilateralFilter(mask_u8, d=9, sigmaColor=75, sigmaSpace=75)
            mask_guided = mask_u8.astype(np.float32) / 255.0
        mask_hr = np.clip(mask_guided, 0, 1).astype(np.float64)
    except Exception:
        pass  # If guided filter fails, continue with unguided mask

    # 4. Morphological cleanup – remove small noise and fill small holes
    mask_u8 = (mask_hr * 255).astype(np.uint8)
    kernel_size = max(3, int(min(h, w) * 0.005) | 1)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Final edge smoothing – light Gaussian to remove staircase artifacts
    mask_u8 = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=0.8, sigmaY=0.8)

    return mask_u8

class TourDialog(QDialog):
    """A multi-step dialog to introduce the software features to new users.
    Shows mini-replicas of the actual app interface for each tour step."""
    def __init__(self, parent=None, language_func=None):
        super().__init__(parent)
        self.get_translation = language_func
        self.current_step = 0
        self.steps = [
            {"title": "tour_welcome_title", "desc": "tour_welcome_desc"},
            {"title": "tour_models_title", "desc": "tour_models_desc"},
            {"title": "tour_process_title", "desc": "tour_process_desc"},
            {"title": "tour_utils_title", "desc": "tour_utils_desc"},
        ]
        
        self.init_ui()
        self.update_step()

    def _build_ui_preview(self, step_index):
        """Build a mini replica of the actual app interface for each tour step."""
        preview = QFrame()
        preview.setObjectName("previewContainer")
        preview.setFixedSize(600, 320)
        preview.setStyleSheet("""
            QFrame#previewContainer { background-color: #f4f5fa; border: 1px solid rgba(99,102,241,0.15); border-radius: 12px; }
            QLabel { color: #333355; font-family: 'Inter', 'Segoe UI', sans-serif; border: none; background: transparent; }
            QPushButton { font-family: 'Inter', 'Segoe UI', sans-serif; border-radius: 6px; padding: 5px 12px; font-size: 10pt; }
        """)
        p_layout = QVBoxLayout(preview)
        p_layout.setContentsMargins(14, 12, 14, 12)
        p_layout.setSpacing(8)

        if step_index == 0:  # Welcome – show the full app skeleton
            # Mini toolbar
            tb = QFrame()
            tb.setObjectName("miniToolbar")
            tb.setStyleSheet("QFrame#miniToolbar { background-color: #ffffff; border: 1px solid #e0e1ec; border-radius: 6px; }")
            tb_lay = QHBoxLayout(tb)
            tb_lay.setContentsMargins(8, 4, 8, 4)
            tb_lay.setSpacing(6)
            for icon_text in ["🔍+", "🔍−", "🎨"]:
                b = QPushButton(icon_text)
                b.setFixedSize(36, 28)
                b.setStyleSheet("background: #f0f0f8; border: 1px solid #d0d1e0; font-size: 10pt; color: #333355;")
                tb_lay.addWidget(b)
            tb_lay.addStretch()
            model_lbl = QLabel("Model: Bria RMBG v1.4")
            model_lbl.setStyleSheet("font-size: 10pt; color: #6366f1; font-weight: bold;")
            tb_lay.addWidget(model_lbl)
            tb_lay.addStretch()
            feather_lbl = QLabel("Feather: 2")
            feather_lbl.setStyleSheet("font-size: 10pt; color: #8888a0;")
            tb_lay.addWidget(feather_lbl)
            p_layout.addWidget(tb)

            # Image area – two panels
            img_row = QHBoxLayout()
            for label_text in ["📂 Original Image", "✨ Processed Image"]:
                panel = QLabel(label_text)
                panel.setAlignment(Qt.AlignCenter)
                panel.setStyleSheet("background: #e8e9f4; border: 2px dashed #d0d1e0; border-radius: 8px; color: #555570; font-size: 11pt; font-weight: bold; min-height: 160px;")
                img_row.addWidget(panel)
            p_layout.addLayout(img_row)

            # Bottom bar
            bb = QFrame()
            bb.setObjectName("miniBottomBar")
            bb.setStyleSheet("QFrame#miniBottomBar { background-color: #ffffff; border: 1px solid #e0e1ec; border-radius: 6px; }")
            bb_lay = QHBoxLayout(bb)
            bb_lay.setContentsMargins(8, 4, 8, 4)
            bb_lay.addStretch()
            for btn_text, is_primary in [("📂 Browse", True), ("✂ Remove BG", True), ("↩ Undo", False), ("↪ Redo", False), ("💾 Save", True)]:
                b = QPushButton(btn_text)
                if is_primary:
                    b.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6366f1,stop:1 #8b5cf6); color: white; font-weight: bold; font-size: 10pt; padding: 5px 10px; border-radius: 6px;")
                else:
                    b.setStyleSheet("background: #f0f0f8; color: #333355; border: 1px solid #d0d1e0; font-size: 10pt; padding: 5px 10px; border-radius: 6px;")
                bb_lay.addWidget(b)
            bb_lay.addStretch()
            p_layout.addWidget(bb)

        elif step_index == 1:  # AI Models & Feathering
            title = QLabel("⚙️  AI Model Selection")
            title.setStyleSheet("font-size: 13pt; font-weight: bold; color: #4f46e5;")
            title.setAlignment(Qt.AlignCenter)
            p_layout.addWidget(title)
            p_layout.addSpacing(6)

            # Model selector replica
            model_frame = QFrame()
            model_frame.setObjectName("modelFrame")
            model_frame.setStyleSheet("QFrame#modelFrame { background: #ffffff; border: 1px solid #d0d1e0; border-radius: 8px; }")
            mf_lay = QVBoxLayout(model_frame)
            mf_lay.setContentsMargins(12, 10, 12, 10)
            mf_lay.setSpacing(6)
            lbl = QLabel("MODEL")
            lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #8888a0; letter-spacing: 1px;")
            mf_lay.addWidget(lbl)
            for model_name in ["✦ Auto (Rembg)", "✦ U²Net", "✦ Bria RMBG v1.4 ◀ selected", "✦ IS-Net", "✦ MODNet"]:
                m = QLabel(model_name)
                if "selected" in model_name:
                    m.setStyleSheet("font-size: 10.5pt; color: #6366f1; font-weight: bold; padding: 4px 8px; background: rgba(99,102,241,0.08); border-radius: 4px;")
                else:
                    m.setStyleSheet("font-size: 10.5pt; color: #555570; padding: 4px 8px;")
                mf_lay.addWidget(m)
            p_layout.addWidget(model_frame)

            # Feather control replica
            feather_frame = QFrame()
            feather_frame.setObjectName("featherFrame")
            feather_frame.setStyleSheet("QFrame#featherFrame { background: #ffffff; border: 1px solid #d0d1e0; border-radius: 8px; }")
            ff_lay = QHBoxLayout(feather_frame)
            ff_lay.setContentsMargins(12, 10, 12, 10)
            ff_lbl = QLabel("FEATHER")
            ff_lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #8888a0;")
            ff_lay.addWidget(ff_lbl)
            ff_val = QLabel("▸ 0  1  [2]  3  4  5 ◂")
            ff_val.setStyleSheet("font-size: 11pt; color: #6366f1; font-weight: bold;")
            ff_val.setAlignment(Qt.AlignCenter)
            ff_lay.addWidget(ff_val)
            p_layout.addWidget(feather_frame)
            p_layout.addStretch()

        elif step_index == 2:  # Load & Process
            title = QLabel("🖼️  Load & Process")
            title.setStyleSheet("font-size: 13pt; font-weight: bold; color: #4f46e5;")
            title.setAlignment(Qt.AlignCenter)
            p_layout.addWidget(title)
            p_layout.addSpacing(6)

            # Step-by-step flow
            flow_frame = QFrame()
            flow_frame.setObjectName("flowFrame")
            flow_frame.setStyleSheet("QFrame#flowFrame { background: #ffffff; border: 1px solid #d0d1e0; border-radius: 8px; }")
            fl = QVBoxLayout(flow_frame)
            fl.setContentsMargins(14, 12, 14, 12)
            fl.setSpacing(10)
            steps_data = [
                ("① Click  📂 Browse  or drag & drop an image", "#333355"),
                ("② Select an AI model from the toolbar", "#333355"),
                ("③ Click  ✂ Remove Background", "#333355"),
                ("④ Watch the progress bar as AI processes", "#333355"),
                ("⑤ View your result in the left panel!", "#6366f1"),
            ]
            for text, color in steps_data:
                s = QLabel(text)
                s.setStyleSheet(f"font-size: 11pt; color: {color}; padding: 4px 0px;")
                fl.addWidget(s)
            p_layout.addWidget(flow_frame)

            # Progress bar replica
            prog = QProgressBar()
            prog.setValue(75)
            prog.setFixedHeight(22)
            prog.setStyleSheet("""
                QProgressBar { background: #e8e9f4; border: 1px solid #d0d1e0; border-radius: 6px; text-align: center; font-size: 10pt; color: #333355; }
                QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6366f1,stop:0.5 #8b5cf6,stop:1 #a78bfa); border-radius: 5px; }
            """)
            p_layout.addWidget(prog)
            p_layout.addStretch()

        elif step_index == 3:  # Utilities & Export
            title = QLabel("🛠️  Utilities & Export")
            title.setStyleSheet("font-size: 13pt; font-weight: bold; color: #4f46e5;")
            title.setAlignment(Qt.AlignCenter)
            p_layout.addWidget(title)
            p_layout.addSpacing(6)

            # Utilities grid
            utils_frame = QFrame()
            utils_frame.setObjectName("utilsFrame")
            utils_frame.setStyleSheet("QFrame#utilsFrame { background: #ffffff; border: 1px solid #d0d1e0; border-radius: 8px; }")
            uf_lay = QVBoxLayout(utils_frame)
            uf_lay.setContentsMargins(14, 12, 14, 12)
            uf_lay.setSpacing(8)
            utils_data = [
                ("🔍  Zoom In / Out", "Ctrl++ / Ctrl+−"),
                ("↩  Undo / Redo", "Ctrl+Z / Ctrl+Y"),
                ("🎨  Switch Theme", "Ctrl+T  (Light ↔ Dark)"),
                ("💾  Save As", "PNG, JPEG, PSD, WEBP"),
                ("🌐  Languages", "English, தமிழ், हिन्दी, Français, Deutsch"),
            ]
            for icon_text, shortcut in utils_data:
                row = QHBoxLayout()
                lbl = QLabel(icon_text)
                lbl.setStyleSheet("font-size: 11pt; font-weight: bold; color: #333355;")
                row.addWidget(lbl)
                row.addStretch()
                sc = QLabel(shortcut)
                sc.setStyleSheet("font-size: 10pt; color: #6366f1; background: rgba(99,102,241,0.06); padding: 3px 8px; border-radius: 4px;")
                row.addWidget(sc)
                uf_lay.addLayout(row)
            p_layout.addWidget(utils_frame)
            p_layout.addStretch()

        return preview

    def init_ui(self):
        self.setWindowTitle("BG Remover Tour")
        self.setFixedSize(650, 600)
        self.setModal(True)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(14)
        
        # Preview area – placeholder that gets replaced each step
        self.preview_container = QVBoxLayout()
        self.preview_container.setAlignment(Qt.AlignCenter)
        self.current_preview = None
        layout.addLayout(self.preview_container)
        
        # Text area
        text_container = QFrame()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(10, 0, 10, 0)
        
        self.title_label = QLabel()
        self.title_label.setFont(QFont("Inter", 16, QFont.Bold))
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignCenter)
        text_layout.addWidget(self.title_label)
        
        self.desc_label = QLabel()
        self.desc_label.setFont(QFont("Inter", 11))
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setStyleSheet("color: #8888a0; margin-top: 6px; line-height: 1.4;")
        text_layout.addWidget(self.desc_label)
        
        layout.addWidget(text_container)
        layout.addStretch()

        # Step indicator dots
        self.dots_layout = QHBoxLayout()
        self.dots_layout.setAlignment(Qt.AlignCenter)
        self.dots_layout.setSpacing(8)
        self.dot_labels = []
        for i in range(len(self.steps)):
            dot = QLabel("●" if i == 0 else "○")
            dot.setFont(QFont("Inter", 10))
            dot.setAlignment(Qt.AlignCenter)
            dot.setFixedWidth(18)
            self.dots_layout.addWidget(dot)
            self.dot_labels.append(dot)
        layout.addLayout(self.dots_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)
        
        self.skip_btn = QPushButton(self.get_translation("skip"))
        self.skip_btn.setObjectName("secondaryBtn")
        self.skip_btn.clicked.connect(self.reject)
        
        self.back_btn = QPushButton(self.get_translation("back"))
        self.back_btn.setObjectName("secondaryBtn")
        self.back_btn.clicked.connect(self.prev_step)
        
        self.next_btn = QPushButton(self.get_translation("next"))
        self.next_btn.setObjectName("primaryBtn")
        self.next_btn.clicked.connect(self.next_step)
        
        nav_layout.addWidget(self.skip_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.back_btn)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        self.setLayout(layout)

    def update_step(self):
        step_data = self.steps[self.current_step]
        self.title_label.setText(self.get_translation(step_data["title"]))
        self.desc_label.setText(self.get_translation(step_data["desc"]))
        
        # Replace the preview widget
        if self.current_preview is not None:
            self.preview_container.removeWidget(self.current_preview)
            self.current_preview.deleteLater()
        self.current_preview = self._build_ui_preview(self.current_step)
        self.preview_container.addWidget(self.current_preview)

        # Update buttons
        self.back_btn.setEnabled(self.current_step > 0)
        if self.current_step == len(self.steps) - 1:
            self.next_btn.setText(self.get_translation("finish"))
        else:
            self.next_btn.setText(self.get_translation("next"))
        
        # Update step indicator dots
        for i, dot in enumerate(self.dot_labels):
            if i == self.current_step:
                dot.setText("●")
                dot.setStyleSheet("color: #6366f1; font-size: 12pt;")
            else:
                dot.setText("○")
                dot.setStyleSheet("color: #555570; font-size: 10pt;")

    def next_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_step()
        else:
            self.accept()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step()

def safe_to_uint8(img):
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class RemoveBackgroundThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, image_np, model_name, feather):
        super().__init__()
        self.image_np = image_np
        self.model_name = model_name
        self.feather = feather

    def run(self):
        try:
            if self.image_np is None:
                self.error.emit("No image loaded or image failed to load.")
                self.finished.emit(None)
                return

            self.progress.emit(5)
            result = None

            if self.model_name == "Auto (Rembg)":
                self.progress.emit(10)
                pil_out = rembg_remove(Image.fromarray(self.image_np), session=new_session("u2net", providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
                if pil_out is None:
                    raise RuntimeError("rembg failed to process image.")
                pil_out = pil_out.convert("RGBA")
                result = np.array(pil_out)



            else:
                model_path = MODELS.get(self.model_name)
                if not model_path:
                    raise FileNotFoundError(f"Model path was not configured for: {self.model_name}")

                # --- ON-DEMAND DOWNLOAD: Download model if not cached locally ---
                if not os.path.exists(model_path):
                    self.progress.emit(5)
                    download_model_on_demand(
                        self.model_name, model_path,
                        progress_callback=lambda p: self.progress.emit(p)
                    )
                
                # --- NEWLY ADDED: Verify model before loading ---
                expected_hash = MODEL_HASHES.get(model_path)
                if expected_hash:
                    if not verify_file_hash(model_path, expected_hash):
                        raise SecurityException(
                            f"Model file '{os.path.basename(model_path)}' has been modified "
                            "or is not the official version. Aborting."
                        )
                else:
                    logging.warning(f"[SECURITY WARNING] No hash found for model '{model_path}'. Proceeding without verification.")
                # --- END NEWLY ADDED SECTION ---

                available_providers = ort.get_available_providers()
                if has_torch and torch.cuda.is_available():
                    logging.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    logging.warning("CUDA not available (or torch missing), falling back to CPU")
                    providers = ["CPUExecutionProvider"]

                # --- MEMORY OPTIMIZATION: Reduce memory usage ---
                gc.collect()
                sess_options = ort.SessionOptions()
                sess_options.enable_cpu_mem_arena = False
                session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
                # -----------------------------------------------

                logging.info(f"[INFO] Using providers: {providers} for model {self.model_name}")

                img_input = preprocess_for_model(self.image_np, self.model_name, session)
                if img_input is None:
                    self.error.emit("Image preprocessing failed.")
                    self.finished.emit(None)
                    return

                self.progress.emit(30)
                preds = session.run(None, {session.get_inputs()[0].name: img_input})
                
                # ... (rest of the model processing logic is unchanged)
                mask_single = None
                if isinstance(preds, (list, tuple)):
                    candidates = []
                    for out in preds:
                        if isinstance(out, np.ndarray):
                            arr = out
                            if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] in (1,3):
                                arr = arr[0]
                            if arr.ndim == 3 and arr.shape[0] in (1,3):
                                cand = arr[0]
                            elif arr.ndim == 3 and arr.shape[2] in (1,3) and arr.shape[0] != 1:
                                cand = arr[:, :, 0]
                            elif arr.ndim == 2:
                                cand = arr
                            elif arr.ndim == 3 and arr.shape[0] == 1:
                                cand = arr[0]
                            else:
                                cand = np.squeeze(arr)
                            if isinstance(cand, np.ndarray) and cand.ndim == 2:
                                candidates.append(cand)
                    if candidates:
                        best = max(candidates, key=lambda x: float(np.nanstd(x)))
                        mask_single = best
                else:
                    pred = preds
                    if isinstance(pred, np.ndarray):
                        arr = pred
                        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] in (1,3):
                            arr = arr[0]
                        if arr.ndim == 3 and arr.shape[0] in (1,3):
                            mask_single = arr[0]
                        elif arr.ndim == 2:
                            mask_single = arr
                        else:
                            mask_single = np.squeeze(arr)
                if mask_single is None:
                    try:
                        if isinstance(preds, (list, tuple)) and len(preds) > 0:
                            mask_single = np.squeeze(preds[0])
                        elif isinstance(preds, np.ndarray):
                            mask_single = np.squeeze(preds)
                    except Exception:
                        mask_single = None
                if mask_single is None:
                    raise RuntimeError("Could not interpret model outputs into a mask.")

                # --- IMPROVED EDGE DETECTION & REFINEMENT ---
                mask = refine_mask(
                    mask_single,
                    self.image_np,
                    (self.image_np.shape[0], self.image_np.shape[1])
                )
                
                result = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2RGBA)
                result[:, :, 3] = mask
                self.progress.emit(85)

            if result is None:
                self.error.emit("Processing failed. Result is None.")
                self.finished.emit(None)
                return

            # --- EDGE SMOOTHENING FOR ALL MODELS ---
            # Always apply at least a minimal edge smooth (sigma=1.0)
            # to eliminate jagged staircase artifacts on edges
            if result.shape[2] == 4:
                alpha = result[:, :, 3]
                smooth_sigma = max(1.0, float(self.feather))
                blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
                result[:, :, 3] = blurred

            self.progress.emit(100)
            self.finished.emit(result)
        
        # --- NEWLY MODIFIED: Added catch for SecurityException ---
        except SecurityException as se:
            logging.error(f"Security exception during background removal: {se}", exc_info=True)
            self.error.emit(str(se)) # Pass the specific security message to the user
        except Exception as e:
            # --- NEW SECURITY ENHANCEMENT: Secure Error Handling ---
            logging.error("An unexpected error occurred during background removal.", exc_info=True)
            error_msg = str(e)
            if "bad allocation" in error_msg or "memory" in error_msg.lower():
                self.error.emit(
                    f"Processing failed due to insufficient memory.\n\n"
                    f"The selected model ({self.model_name}) is too large for your system.\n"
                    f"Please try selecting a smaller model like 'U2Net' or 'IS-Net'.\n\n"
                    f"Technical Error: {error_msg}"
                )
            else:
                self.error.emit(f"Processing failed: {error_msg}\n\nSee log at:\n{log_file_path}")
            # --- END NEW SECURITY ENHANCEMENT ---

            # --- END NEW SECURITY ENHANCEMENT ---

# --- UPDATE COMPONENTS ---

class UpdateCheckerThread(QThread):
    """Checks for updates in the background."""
    update_found = pyqtSignal(object)  # Emits app_update object
    update_not_found = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        if not has_updater:
            return
        
        try:
            # Ensure data dir exists
            if not os.path.exists(AppConfig.DATA_DIR):
                os.makedirs(AppConfig.DATA_DIR, exist_ok=True)
                
            client = Client(AppConfig(), refresh=True)
            app_update = client.update_check(AppConfig.APP_NAME, APP_VERSION)
            
            if app_update:
                self.update_found.emit(app_update)
            else:
                self.update_not_found.emit()
        except Exception as e:
            logging.error("Update check failed", exc_info=True)
            self.error.emit(str(e))

class UpdateDownloaderThread(QThread):
    """Downloads the update with progress tracking."""
    progress = pyqtSignal(int, str)  # percent, status message
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, app_update):
        super().__init__()
        self.app_update = app_update

    def progress_hook(self, info):
        """Callback for PyUpdater progress."""
        # info dict keys: 'total', 'downloaded', 'percent_complete', 'status'
        try:
            # percent_complete can be a string like "22.6"
            percent_str = info.get('percent_complete', '0')
            percent = int(float(percent_str))
        except (ValueError, TypeError):
            percent = 0
        status = info.get('status', 'Downloading...')
        
        # Calculate downloaded MB
        downloaded = info.get('downloaded', 0)
        total = info.get('total', 0)
        if total > 0:
            msg = f"{status}: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB"
        else:
            msg = status
            
        self.progress.emit(percent, msg)

    def run(self):
        try:
            # Hack: Append our hook to the existing hooks
            if not hasattr(self.app_update, 'progress_hooks'):
                self.app_update.progress_hooks = []
            self.app_update.progress_hooks.append(self.progress_hook)
            
            self.progress.emit(0, "Starting download...")
            logging.info("UpdateDownloaderThread: Calling app_update.download")
            
            # Use background=False because we are already in a thread (UpdateDownloaderThread)
            success = self.app_update.download(background=False)
            logging.info(f"UpdateDownloaderThread: download returned {success}")
            
            if success:
                if self.app_update.is_downloaded():
                    self.progress.emit(100, "Download complete. Extracting...")
                    self.finished.emit()
                else:
                    self.error.emit("Download verification failed.")
            else:
                self.error.emit("Download failed.")
        except Exception as e:
            logging.error("Update download failed", exc_info=True)
            self.error.emit(str(e))

class DownloadProgressDialog(QDialog):
    """Dialog to show update download progress."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading Update")
        self.setFixedSize(450, 180)
        self.setModal(True)
        
        # Disable close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(14)
        
        title_label = QLabel("⬇️  Downloading Update...")
        title_label.setFont(QFont("Inter", 13, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Inter", 9))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #8888a0;")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)

    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

# -------------------------

# =============================================
# MODERN UI STYLESHEET SYSTEM
# =============================================

DARK_STYLESHEET = """
/* ── Global ── */
QMainWindow, QWidget {
    background-color: #13141f;
    color: #e0e0ee;
    font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
    font-size: 9.5pt;
}

/* ── Menu Bar ── */
QMenuBar {
    background-color: #1a1b2e;
    color: #c0c0d0;
    border-bottom: 1px solid #2a2b3d;
    padding: 2px 0px;
    font-size: 9.5pt;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 6px;
    margin: 2px 1px;
}
QMenuBar::item:selected {
    background-color: rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
}
QMenu {
    background-color: #1e1f33;
    color: #c0c0d0;
    border: 1px solid #2a2b3d;
    border-radius: 8px;
    padding: 6px;
}
QMenu::item {
    padding: 8px 28px 8px 16px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: rgba(99, 102, 241, 0.3);
    color: #c7d2fe;
}
QMenu::separator {
    height: 1px;
    background-color: #2a2b3d;
    margin: 4px 10px;
}

/* ── Toolbar Panel ── */
QFrame#toolbarFrame {
    background-color: #1a1b2e;
    border: 1px solid #2a2b3d;
    border-radius: 8px;
    padding: 4px 8px;
    margin: 2px;
}
QFrame#bottomBarFrame {
    background-color: #1a1b2e;
    border: 1px solid #2a2b3d;
    border-radius: 8px;
    padding: 4px 8px;
    margin: 2px;
}

/* ── Primary Action Buttons ── */
QPushButton#primaryBtn {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:1 #8b5cf6);
    color: #ffffff;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 9.5pt;
    min-height: 18px;
}
QPushButton#primaryBtn:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7c7ff7, stop:1 #a78bfa);
    border: 1px solid rgba(165, 180, 252, 0.4);
}
QPushButton#primaryBtn:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4f46e5, stop:1 #7c3aed);
    border: 2px solid #a5b4fc;
    padding: 5px 15px;
}
QPushButton#primaryBtn:disabled {
    background-color: #2a2b3d;
    color: #555;
}

/* ── Secondary Action Buttons ── */
QPushButton#secondaryBtn {
    background-color: rgba(42, 43, 61, 0.8);
    color: #c0c0d0;
    border: 1px solid #3a3b5d;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 9.5pt;
    min-height: 18px;
}
QPushButton#secondaryBtn:hover {
    background-color: rgba(99, 102, 241, 0.18);
    border-color: #6366f1;
    color: #a5b4fc;
}
QPushButton#secondaryBtn:pressed {
    background-color: rgba(99, 102, 241, 0.30);
    border: 2px solid #818cf8;
    padding: 5px 13px;
}

/* ── Utility Buttons (Zoom, Theme) ── */
QPushButton#utilBtn {
    background-color: transparent;
    color: #8888a0;
    border: 1px solid #2a2b3d;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 9pt;
}
QPushButton#utilBtn:hover {
    background-color: rgba(99, 102, 241, 0.12);
    border-color: #4a4b6d;
    color: #a5b4fc;
}
QPushButton#utilBtn:pressed {
    background-color: rgba(99, 102, 241, 0.22);
    border: 2px solid #818cf8;
    padding: 4px 9px;
}

/* ── Control Container ── */
QFrame#controlContainer {
    background-color: #1e1f33;
    border: 1px solid #3a3b5d;
    border-radius: 8px;
    padding: 2px 4px;
}
QFrame#controlContainer:hover {
    border-color: #6366f1;
}
QFrame#controlContainer QComboBox, QFrame#controlContainer QSpinBox {
    background-color: transparent;
    border: none;
    padding: 2px 4px;
}
QFrame#controlContainer QComboBox:hover, QFrame#controlContainer QSpinBox:hover {
    border: none;
    background-color: transparent;
}
QFrame#controlContainer QComboBox:focus, QFrame#controlContainer QSpinBox:focus {
    border: none;
    background-color: transparent;
}
QFrame#controlContainer QLabel {
    color: #8888a0;
    font-size: 8.5pt;
    font-weight: bold;
    text-transform: uppercase;
    padding-left: 4px;
}

/* ── ComboBox ── */
QComboBox {
    background-color: #1e1f33;
    color: #c0c0d0;
    border: 1px solid #3a3b5d;
    border-radius: 6px;
    padding: 4px 8px;
    min-width: 100px;
    font-size: 9.5pt;
}
QComboBox:hover {
    border-color: #6366f1;
}
QComboBox:focus {
    border-color: #8b5cf6;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: url(DOWN_ARROW_DARK_PLACEHOLDER);
    width: 12px;
    height: 12px;
    margin-right: 4px;
}
QComboBox QAbstractItemView {
    background-color: #1e1f33;
    color: #c0c0d0;
    border: 1px solid #3a3b5d;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: rgba(99, 102, 241, 0.3);
    selection-color: #c7d2fe;
    outline: none;
}

/* ── SpinBox ── */
QSpinBox {
    background-color: #1e1f33;
    color: #c0c0d0;
    border: 1px solid #3a3b5d;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 9.5pt;
}
QSpinBox:hover {
    border-color: #6366f1;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #2a2b3d;
    border: none;
    width: 20px;
    border-radius: 4px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: rgba(99, 102, 241, 0.3);
}
QSpinBox::up-arrow {
    image: url(UP_ARROW_DARK_PLACEHOLDER);
    width: 8px;
    height: 8px;
}
QSpinBox::down-arrow {
    image: url(DOWN_ARROW_DARK_PLACEHOLDER);
    width: 8px;
    height: 8px;
}

/* ── Progress Bar ── */
QProgressBar {
    background-color: #1e1f33;
    border: 1px solid #2a2b3d;
    border-radius: 6px;
    text-align: center;
    color: #c0c0d0;
    font-weight: bold;
    min-height: 16px;
    margin: 4px 6px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:0.5 #8b5cf6, stop:1 #a78bfa);
    border-radius: 5px;
}

/* ── Scroll Area ── */
QScrollArea {
    background-color: #0f1019;
    border: 1px solid #2a2b3d;
    border-radius: 8px;
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background-color: transparent;
    width: 10px;
    margin: 4px 2px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #3a3b5d;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #6366f1;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background-color: transparent;
    height: 10px;
    margin: 2px 4px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #3a3b5d;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #6366f1;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Labels ── */
QLabel {
    color: #c0c0d0;
    font-size: 9.5pt;
}
QLabel#sectionLabel {
    color: #8888a0;
    font-size: 8.5pt;
    font-weight: bold;
    text-transform: uppercase;
}
QLabel#dropZoneLabel {
    color: #555570;
    font-size: 11pt;
    font-weight: bold;
    border: 2px dashed #2a2b3d;
    border-radius: 8px;
    background-color: #0f1019;
}

/* ── Separator Line ── */
QFrame#separator {
    background-color: #2a2b3d;
    max-width: 1px;
    margin: 4px 8px;
}

/* ── Dialogs ── */
QDialog {
    background-color: #1a1b2e;
    color: #e0e0ee;
    border-radius: 8px;
}
QDialog QLabel {
    color: #c0c0d0;
}

/* ── Message Box ── */
QMessageBox {
    background-color: #1a1b2e;
    color: #e0e0ee;
}
QMessageBox QPushButton {
    background-color: #2a2b3d;
    color: #c0c0d0;
    border: 1px solid #3a3b5d;
    border-radius: 6px;
    padding: 6px 16px;
    min-width: 80px;
}
QMessageBox QPushButton:hover {
    background-color: rgba(99, 102, 241, 0.2);
    border-color: #6366f1;
}

/* ── Tooltip ── */
QToolTip {
    background-color: #1e1f33;
    color: #c0c0d0;
    border: 1px solid #3a3b5d;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 9pt;
}
"""

LIGHT_STYLESHEET = """
/* ── Global ── */
QMainWindow, QWidget {
    background-color: #f4f5fa;
    color: #1e1e2e;
    font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
    font-size: 9.5pt;
}

/* ── Menu Bar ── */
QMenuBar {
    background-color: #ffffff;
    color: #333355;
    border-bottom: 1px solid #e0e1ec;
    padding: 2px 0px;
    font-size: 9.5pt;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 6px;
    margin: 2px 1px;
}
QMenuBar::item:selected {
    background-color: rgba(99, 102, 241, 0.12);
    color: #4f46e5;
}
QMenu {
    background-color: #ffffff;
    color: #333355;
    border: 1px solid #e0e1ec;
    border-radius: 8px;
    padding: 6px;
}
QMenu::item {
    padding: 8px 28px 8px 16px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: rgba(99, 102, 241, 0.12);
    color: #4f46e5;
}
QMenu::separator {
    height: 1px;
    background-color: #e0e1ec;
    margin: 4px 10px;
}

/* ── Toolbar Panel ── */
QFrame#toolbarFrame {
    background-color: #ffffff;
    border: 1px solid #e0e1ec;
    border-radius: 8px;
    padding: 4px 8px;
    margin: 2px;
}
QFrame#bottomBarFrame {
    background-color: #ffffff;
    border: 1px solid #e0e1ec;
    border-radius: 8px;
    padding: 4px 8px;
    margin: 2px;
}

/* ── Primary Action Buttons ── */
QPushButton#primaryBtn {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:1 #8b5cf6);
    color: #ffffff;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 9.5pt;
    min-height: 18px;
}
QPushButton#primaryBtn:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7c7ff7, stop:1 #a78bfa);
    border: 1px solid rgba(99, 102, 241, 0.4);
}
QPushButton#primaryBtn:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4f46e5, stop:1 #7c3aed);
    border: 2px solid #4f46e5;
    padding: 5px 15px;
}
QPushButton#primaryBtn:disabled {
    background-color: #d0d0dd;
    color: #999;
}

/* ── Secondary Action Buttons ── */
QPushButton#secondaryBtn {
    background-color: #ffffff;
    color: #333355;
    border: 1px solid #d0d1e0;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 9.5pt;
    min-height: 18px;
}
QPushButton#secondaryBtn:hover {
    background-color: rgba(99, 102, 241, 0.08);
    border-color: #6366f1;
    color: #4f46e5;
}
QPushButton#secondaryBtn:pressed {
    background-color: rgba(99, 102, 241, 0.16);
    border: 2px solid #6366f1;
    padding: 5px 13px;
}

/* ── Utility Buttons (Zoom, Theme) ── */
QPushButton#utilBtn {
    background-color: transparent;
    color: #666688;
    border: 1px solid #d0d1e0;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 9pt;
}
QPushButton#utilBtn:hover {
    background-color: rgba(99, 102, 241, 0.08);
    border-color: #6366f1;
    color: #4f46e5;
}
QPushButton#utilBtn:pressed {
    background-color: rgba(99, 102, 241, 0.16);
    border: 2px solid #6366f1;
    padding: 4px 9px;
}

/* ── Control Container ── */
QFrame#controlContainer {
    background-color: #ffffff;
    border: 1px solid #d0d1e0;
    border-radius: 8px;
    padding: 2px 4px;
}
QFrame#controlContainer:hover {
    border-color: #6366f1;
}
QFrame#controlContainer QComboBox, QFrame#controlContainer QSpinBox {
    background-color: transparent;
    border: none;
    padding: 2px 4px;
}
QFrame#controlContainer QComboBox:hover, QFrame#controlContainer QSpinBox:hover {
    border: none;
    background-color: transparent;
}
QFrame#controlContainer QComboBox:focus, QFrame#controlContainer QSpinBox:focus {
    border: none;
    background-color: transparent;
}
QFrame#controlContainer QLabel {
    color: #666688;
    font-size: 8.5pt;
    font-weight: bold;
    text-transform: uppercase;
    padding-left: 4px;
}

/* ── ComboBox ── */
QComboBox {
    background-color: #ffffff;
    color: #333355;
    border: 1px solid #d0d1e0;
    border-radius: 6px;
    padding: 4px 8px;
    min-width: 100px;
    font-size: 9.5pt;
}
QComboBox:hover {
    border-color: #6366f1;
}
QComboBox:focus {
    border-color: #8b5cf6;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: url(DOWN_ARROW_LIGHT_PLACEHOLDER);
    width: 12px;
    height: 12px;
    margin-right: 4px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #333355;
    border: 1px solid #d0d1e0;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: rgba(99, 102, 241, 0.15);
    selection-color: #4f46e5;
    outline: none;
}

/* ── SpinBox ── */
QSpinBox {
    background-color: #ffffff;
    color: #333355;
    border: 1px solid #d0d1e0;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 9.5pt;
}
QSpinBox:hover {
    border-color: #6366f1;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #f0f0f8;
    border: none;
    width: 20px;
    border-radius: 4px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: rgba(99, 102, 241, 0.15);
}
QSpinBox::up-arrow {
    image: url(UP_ARROW_LIGHT_PLACEHOLDER);
    width: 8px;
    height: 8px;
}
QSpinBox::down-arrow {
    image: url(DOWN_ARROW_LIGHT_PLACEHOLDER);
    width: 8px;
    height: 8px;
}

/* ── Progress Bar ── */
QProgressBar {
    background-color: #e8e9f4;
    border: 1px solid #d0d1e0;
    border-radius: 6px;
    text-align: center;
    color: #333355;
    font-weight: bold;
    min-height: 16px;
    margin: 4px 6px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:0.5 #8b5cf6, stop:1 #a78bfa);
    border-radius: 5px;
}

/* ── Scroll Area ── */
QScrollArea {
    background-color: #ecedf6;
    border: 1px solid #d0d1e0;
    border-radius: 8px;
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background-color: transparent;
    width: 10px;
    margin: 4px 2px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #c0c1d0;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #6366f1;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background-color: transparent;
    height: 10px;
    margin: 2px 4px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #c0c1d0;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #6366f1;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Labels ── */
QLabel {
    color: #333355;
    font-size: 9.5pt;
}
QLabel#sectionLabel {
    color: #666688;
    font-size: 8.5pt;
    font-weight: bold;
    text-transform: uppercase;
}
QLabel#dropZoneLabel {
    color: #9999bb;
    font-size: 11pt;
    font-weight: bold;
    border: 2px dashed #c0c1d0;
    border-radius: 8px;
    background-color: #ecedf6;
}

/* ── Separator Line ── */
QFrame#separator {
    background-color: #d0d1e0;
    max-width: 1px;
    margin: 4px 8px;
}

/* ── Dialogs ── */
QDialog {
    background-color: #ffffff;
    color: #1e1e2e;
    border-radius: 8px;
}
QDialog QLabel {
    color: #333355;
}

/* ── Message Box ── */
QMessageBox {
    background-color: #ffffff;
    color: #1e1e2e;
}
QMessageBox QPushButton {
    background-color: #f0f0f8;
    color: #333355;
    border: 1px solid #d0d1e0;
    border-radius: 6px;
    padding: 6px 16px;
    min-width: 80px;
}
QMessageBox QPushButton:hover {
    background-color: rgba(99, 102, 241, 0.12);
    border-color: #6366f1;
}

/* ── Tooltip ── */
QToolTip {
    background-color: #ffffff;
    color: #333355;
    border: 1px solid #d0d1e0;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 9pt;
}
"""

# Replace placeholders with runtime absolute paths
DARK_STYLESHEET = DARK_STYLESHEET.replace("DOWN_ARROW_DARK_PLACEHOLDER", down_arrow_dark_path)
LIGHT_STYLESHEET = LIGHT_STYLESHEET.replace("DOWN_ARROW_LIGHT_PLACEHOLDER", down_arrow_light_path)
DARK_STYLESHEET = DARK_STYLESHEET.replace("UP_ARROW_DARK_PLACEHOLDER", up_arrow_dark_path)
LIGHT_STYLESHEET = LIGHT_STYLESHEET.replace("UP_ARROW_LIGHT_PLACEHOLDER", up_arrow_light_path)

# =============================================
# END MODERN UI STYLESHEET SYSTEM
# =============================================


class BackgroundRemoverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.processed_image = None
        self.image_path = None
        self.scale_factor = 1.0
        self.setWindowTitle("AI Background Remover")
        app_icon_path = os.path.join(ASSETS_DIR, "Firefly_GeminiFlash_replace the centre lady image into the next reference image with fully fitteed inside 468388 copy.ico")
        if os.path.exists(app_icon_path):
            self.setWindowIcon(QIcon(app_icon_path))
        self.setGeometry(100, 100, 1200, 800)
        self.zoom_factor = 1.0
        self.image_np = None
        self.result_np = None
        self.original_image_np = None
        self.is_dark = False
        self.undo_stack = []
        self.redo_stack = []
        self.drag_pos = None
        self._is_syncing = False

        # --- SECURITY ENHANCEMENT: Decompression Bomb Protection ---
        # Set a reasonable limit for the number of pixels in an image (e.g., 100 Megapixels)
        Image.MAX_IMAGE_PIXELS = 100_000_000

        # ... (rest of __init__ is unchanged)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open", self.load_image, "Ctrl+O")
        file_menu.addAction("Save", self.save_image, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo", self.undo, "Ctrl+Z")
        edit_menu.addAction("Redo", self.redo, "Ctrl+Y")
        edit_menu.addSeparator()
        edit_menu.addAction("Clear", self.clear_image, "Del")
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Zoom In", self.zoom_in, "Ctrl++")
        view_menu.addAction("Zoom Out", self.zoom_out, "Ctrl+-")
        view_menu.addAction("Switch Theme", self.toggle_theme, "Ctrl+T")
        settings_menu = menubar.addMenu("Settings")
        settings_menu.addAction("Language", self.update_language)
        settings_menu.addAction("Model", lambda: self.model_select.showPopup() if hasattr(self, "model_select") else None)
        settings_menu.addAction("Feather Strength", lambda: self.feather_spinbox.setFocus() if hasattr(self, "feather_spinbox") else None)
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Check for Updates", self.manual_update)
        help_menu.addAction("Software Tour", self.show_tour)
        help_menu.addAction("Feedback / Report Bug", self.show_feedback_dialog)
        help_menu.addAction("About", self.show_about)
        self.left_image_label = QLabel("✨  Processed Image")
        self.left_image_label.setObjectName("dropZoneLabel")
        self.left_image_label.setAlignment(Qt.AlignCenter)
        self.left_image_label.setAcceptDrops(True)
        
        self.right_image_label = QLabel("📂  Drop or Browse an Image")
        self.right_image_label.setObjectName("dropZoneLabel")
        self.right_image_label.setAlignment(Qt.AlignCenter)
        
        self.left_scroll_area = QScrollArea()
        self.left_scroll_area.setWidgetResizable(True)
        self.left_scroll_area.setWidget(self.left_image_label)
        
        self.right_scroll_area = QScrollArea()
        self.right_scroll_area.setWidgetResizable(True)
        self.right_scroll_area.setWidget(self.right_image_label)
        
        # Synchronize scrollbars with flag to prevent recursive updates
        def sync_h_scroll_left_to_right(val):
            if not self._is_syncing:
                self._is_syncing = True
                self.right_scroll_area.horizontalScrollBar().setValue(val)
                self._is_syncing = False
                
        def sync_h_scroll_right_to_left(val):
            if not self._is_syncing:
                self._is_syncing = True
                self.left_scroll_area.horizontalScrollBar().setValue(val)
                self._is_syncing = False
                
        def sync_v_scroll_left_to_right(val):
            if not self._is_syncing:
                self._is_syncing = True
                self.right_scroll_area.verticalScrollBar().setValue(val)
                self._is_syncing = False
                
        def sync_v_scroll_right_to_left(val):
            if not self._is_syncing:
                self._is_syncing = True
                self.left_scroll_area.verticalScrollBar().setValue(val)
                self._is_syncing = False

        self.left_scroll_area.horizontalScrollBar().valueChanged.connect(sync_h_scroll_left_to_right)
        self.right_scroll_area.horizontalScrollBar().valueChanged.connect(sync_h_scroll_right_to_left)
        self.left_scroll_area.verticalScrollBar().valueChanged.connect(sync_v_scroll_left_to_right)
        self.right_scroll_area.verticalScrollBar().valueChanged.connect(sync_v_scroll_right_to_left)
        
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.right_scroll_area)
        images_layout.addWidget(self.left_scroll_area)
        
        font = QFont("Inter", 10)
        font_bold = QFont("Inter", 10, QFont.Bold)

        self.language_select = QComboBox()
        self.language_select.addItems(LANGUAGES.keys())
        self.language_select.setCurrentText("English")
        self.language_select.currentTextChanged.connect(self.update_language)

        self.model_label = QLabel()
        self.model_label.setObjectName("sectionLabel")
        self.model_select = QComboBox()
        self.model_select.addItems(MODELS.keys())
        self.model_select.setCurrentText("Bria RMBG v1.4")

        self.feather_label = QLabel()
        self.feather_label.setObjectName("sectionLabel")
        self.feather_spinbox = QSpinBox()
        self.feather_spinbox.setMinimum(0)
        self.feather_spinbox.setMaximum(10)
        self.feather_spinbox.setValue(2)
        self.feather_spinbox.setFixedWidth(70)

        self.language_label = QLabel()
        self.language_label.setObjectName("sectionLabel")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # --- Create Buttons with CSS Object Names ---
        self.browse_btn = self.create_icon_button("", "browse.png", self.load_image, font_bold, "primaryBtn")
        self.remove_btn = self.create_icon_button("", "remove.png", self.remove_background, font_bold, "primaryBtn")
        self.undo_btn = self.create_icon_button("", "undo.png", self.undo, font, "secondaryBtn")
        self.redo_btn = self.create_icon_button("", "redo.png", self.redo, font, "secondaryBtn")
        self.save_btn = self.create_icon_button("", "save.png", self.save_image, font_bold, "primaryBtn")
        self.zoom_in_btn = self.create_icon_button("", "zoom_in.png", self.zoom_in, font, "utilBtn")
        self.zoom_out_btn = self.create_icon_button("", "zoom_out.png", self.zoom_out, font, "utilBtn")
        self.theme_btn = self.create_icon_button("", "theme.png", self.toggle_theme, font, "utilBtn")

        self.update_ui_texts()

        # --- Create Control Containers ---
        self.model_container = QFrame()
        self.model_container.setObjectName("controlContainer")
        model_cont_layout = QHBoxLayout(self.model_container)
        model_cont_layout.setContentsMargins(6, 2, 6, 2)
        model_cont_layout.setSpacing(4)
        model_cont_layout.addWidget(self.model_label)
        model_cont_layout.addWidget(self.model_select)

        self.feather_container = QFrame()
        self.feather_container.setObjectName("controlContainer")
        feather_cont_layout = QHBoxLayout(self.feather_container)
        feather_cont_layout.setContentsMargins(6, 2, 6, 2)
        feather_cont_layout.setSpacing(4)
        feather_cont_layout.addWidget(self.feather_label)
        feather_cont_layout.addWidget(self.feather_spinbox)

        self.language_container = QFrame()
        self.language_container.setObjectName("controlContainer")
        language_cont_layout = QHBoxLayout(self.language_container)
        language_cont_layout.setContentsMargins(6, 2, 6, 2)
        language_cont_layout.setSpacing(4)
        language_cont_layout.addWidget(self.language_label)
        language_cont_layout.addWidget(self.language_select)

        # --- Top Toolbar in a styled QFrame ---
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("toolbarFrame")
        top_layout = QHBoxLayout(toolbar_frame)
        top_layout.setContentsMargins(8, 4, 8, 4)
        top_layout.setSpacing(8)

        # Left: Zoom and Theme
        top_layout.addWidget(self.zoom_in_btn)
        top_layout.addWidget(self.zoom_out_btn)
        top_layout.addWidget(self.theme_btn)

        # Center: Model selection
        top_layout.addStretch()
        top_layout.addWidget(self.model_container)
        top_layout.addStretch()

        # Right: Feather and Language
        top_layout.addWidget(self.feather_container)
        top_layout.addWidget(self.language_container)

        # --- Bottom Action Bar in a styled QFrame ---
        bottom_frame = QFrame()
        bottom_frame.setObjectName("bottomBarFrame")
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(8, 4, 8, 4)
        bottom_layout.setSpacing(10)

        bottom_layout.addStretch()

        # Center: Browse & Process
        bottom_layout.addWidget(self.browse_btn)
        bottom_layout.addWidget(self.remove_btn)

        sep_mid1 = QFrame()
        sep_mid1.setObjectName("separator")
        sep_mid1.setFrameShape(QFrame.VLine)
        sep_mid1.setFixedWidth(2)
        sep_mid1.setFixedHeight(24)
        bottom_layout.addWidget(sep_mid1)

        # Center: Undo & Redo
        bottom_layout.addWidget(self.undo_btn)
        bottom_layout.addWidget(self.redo_btn)

        sep_mid2 = QFrame()
        sep_mid2.setObjectName("separator")
        sep_mid2.setFrameShape(QFrame.VLine)
        sep_mid2.setFixedWidth(2)
        sep_mid2.setFixedHeight(24)
        bottom_layout.addWidget(sep_mid2)

        # Center: Save
        bottom_layout.addWidget(self.save_btn)
        bottom_layout.addStretch()

        # --- Main Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)
        main_layout.addWidget(toolbar_frame)
        main_layout.addLayout(images_layout, 10)
        main_layout.addWidget(bottom_frame)
        main_layout.addWidget(self.progress)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- Apply default theme (Light) ---
        self.setStyleSheet(LIGHT_STYLESHEET)
        self.left_image_label.installEventFilter(self)
        self.left_image_label.mousePressEvent = self.start_drag
        self.left_image_label.mouseMoveEvent = self.drag_image
        self.left_image_label.mouseReleaseEvent = self.end_drag
        
        self.right_image_label.installEventFilter(self)
        self.right_image_label.mousePressEvent = self.start_drag
        self.right_image_label.mouseMoveEvent = self.drag_image
        self.right_image_label.mouseReleaseEvent = self.end_drag

        # --- AUTO-UPDATE CHECK ---
        if has_updater:
            # Run silent check after UI init
            from PySide6.QtCore import QTimer
            QTimer.singleShot(2000, self.check_for_updates_silently)

        # --- FIRST-TIME USER TOUR ---
        QTimer.singleShot(500, self.check_first_run)

    def check_first_run(self):
        settings = QSettings(AppConfig.COMPANY_NAME, AppConfig.APP_NAME)
        if not settings.value("tour_finished", False, type=bool):
            self.show_tour()

    def show_tour(self):
        tour = TourDialog(self, self.get_translation)
        if tour.exec_() == QDialog.Accepted:
            settings = QSettings(AppConfig.COMPANY_NAME, AppConfig.APP_NAME)
            settings.setValue("tour_finished", True)
            logging.info("User finished the tour.")
        else:
            # If they skipped, we still mark it as finished so it doesn't bother them again
            # unless we want it to show until they actually "Finish" it.
            # User request said "skip option bixes", so skipping should probably count as "I've seen it".
            settings = QSettings(AppConfig.COMPANY_NAME, AppConfig.APP_NAME)
            settings.setValue("tour_finished", True)
            logging.info("User skipped the tour.")

    def check_for_updates_silently(self):
        logging.info("Checking for updates silently...")
        self.update_checker = UpdateCheckerThread()
        self.update_checker.update_found.connect(self.on_update_found)
        # We don't connect update_not_found or error for silent check
        self.update_checker.start()

    def on_update_found(self, app_update):
        # Store for use
        self.pending_update = app_update
        
        reply = QMessageBox.question(
            self, "Update Available",
            f"A new version of {AppConfig.APP_NAME} is available.\n\n"
            f"Current Version: {APP_VERSION}\n"
            f"New Version: {app_update.version}\n\n"
            "Do you want to download and install it now?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.start_update_download(app_update)

    def start_update_download(self, app_update):
        self.progress_dialog = DownloadProgressDialog(self)
        self.progress_dialog.show()
        
        self.downloader_thread = UpdateDownloaderThread(app_update)
        self.downloader_thread.progress.connect(self.progress_dialog.update_progress)
        self.downloader_thread.finished.connect(self.on_download_finished)
        self.downloader_thread.error.connect(self.on_download_error)
        self.downloader_thread.start()

    def on_download_finished(self):
        self.progress_dialog.close()
        QMessageBox.information(self, "Update Ready", "Update downloaded successfully. The application will now restart.")
        # Perform restart
        if self.pending_update:
            try:
                self.pending_update.extract_restart()
            except Exception as e:
                QMessageBox.critical(self, "Restart Error", f"Failed to restart: {e}")

    def on_download_error(self, error_msg):
        self.progress_dialog.close()
        QMessageBox.warning(self, "Update Failed", f"Failed to download update:\n{error_msg}")

    def show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About BG Remover")
        dialog.setFixedSize(440, 360)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(12)
        
        title = QLabel("🎨 AI Background Remover")
        title.setFont(QFont("Inter", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        is_dark = getattr(self, "is_dark", False)
        if is_dark:
            title.setStyleSheet("color: #a5b4fc;")
            box_bg = "#25263b"
            box_border = "1px solid #3f3f5f"
            title_color = "#a5b4fc"
            sub_text_color = "#8888a0"
        else:
            title.setStyleSheet("color: #4f46e5;")
            box_bg = "#f4f5fa"
            box_border = "1px solid #d0d1e0"
            title_color = "#4f46e5"
            sub_text_color = "#666688"
            
        layout.addWidget(title)
        
        version = QLabel(f"Version {APP_VERSION}")
        version.setFont(QFont("Inter", 10))
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: #8888a0;")
        layout.addWidget(version)
        
        layout.addSpacing(4)
        
        box_style = f"""
            QFrame {{
                background-color: {box_bg};
                border: {box_border};
                border-radius: 10px;
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
        """
        
        # Box 1: Developer info
        dev_frame = QFrame()
        dev_frame.setStyleSheet(box_style)
        dev_layout = QVBoxLayout(dev_frame)
        dev_layout.setContentsMargins(12, 10, 12, 10)
        dev_layout.setSpacing(4)
        
        dev_label = QLabel(f"Developed by <b>{AppConfig.COMPANY_NAME}</b>")
        dev_label.setFont(QFont("Inter", 10))
        dev_label.setTextFormat(Qt.RichText)
        dev_label.setAlignment(Qt.AlignCenter)
        if is_dark:
            dev_label.setStyleSheet("color: #e0e0ee;")
        else:
            dev_label.setStyleSheet("color: #333355;")
            
        copyright_label = QLabel("© 2025 All Rights Reserved")
        copyright_label.setFont(QFont("Inter", 9))
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet(f"color: {sub_text_color};")
        
        dev_layout.addWidget(dev_label)
        dev_layout.addWidget(copyright_label)
        layout.addWidget(dev_frame)
        
        # Box 2: AI models info
        ai_frame = QFrame()
        ai_frame.setStyleSheet(box_style)
        ai_layout = QVBoxLayout(ai_frame)
        ai_layout.setContentsMargins(12, 10, 12, 10)
        ai_layout.setSpacing(4)
        
        ai_title = QLabel("Powered by advanced AI models")
        ai_title.setFont(QFont("Inter", 9, QFont.Bold))
        ai_title.setAlignment(Qt.AlignCenter)
        ai_title.setStyleSheet(f"color: {title_color};")
        
        ai_desc = QLabel("U²Net · MODNet · IS-Net · BASNet · BiRefNet")
        ai_desc.setFont(QFont("Inter", 9))
        ai_desc.setAlignment(Qt.AlignCenter)
        if is_dark:
            ai_desc.setStyleSheet("color: #e0e0ee;")
        else:
            ai_desc.setStyleSheet("color: #333355;")
            
        ai_layout.addWidget(ai_title)
        ai_layout.addWidget(ai_desc)
        layout.addWidget(ai_frame)
        
        layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondaryBtn")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def manual_update(self):
        if not has_updater:
            QMessageBox.warning(self, "Update", "Updater not available. Please install pyupdater.")
            return

        # Show feedback that we are checking
        self.update_msg = QMessageBox(self)
        self.update_msg.setWindowTitle(self.get_translation("update_check_title") if hasattr(self, "get_translation") else "Checking for Updates")
        self.update_msg.setText(self.get_translation("connecting_server") if hasattr(self, "get_translation") else "Connecting to server...")
        self.update_msg.setStandardButtons(QMessageBox.Cancel)
        
        self.update_checker = UpdateCheckerThread()
        self.update_checker.update_found.connect(lambda u: self._manual_update_found(u))
        self.update_checker.update_not_found.connect(self._manual_update_not_found)
        self.update_checker.error.connect(self._manual_update_error)
        
        # Add a timeout timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._manual_update_timeout)
        
        # Connect cancel button
        self.update_msg.buttonClicked.connect(self._manual_update_cancelled)
        
        self.update_checker.start()
        self.update_timer.start(15000) # 15 second timeout
        self.update_msg.exec_()

    def _manual_update_timeout(self):
        if self.update_checker.isRunning():
            self.update_checker.terminate()
            self.update_msg.close()
            QMessageBox.warning(self, "Update Timeout", "The update server is taking too long to respond. Please try again later or check your internet connection.")

    def _manual_update_cancelled(self, button):
        if self.update_msg.standardButton(button) == QMessageBox.Cancel:
            if self.update_checker.isRunning():
                self.update_checker.terminate()
            self.update_timer.stop()

    def _manual_update_found(self, app_update):
        self.update_msg.close()
        self.on_update_found(app_update)

    def _manual_update_not_found(self):
        self.update_msg.close()
        QMessageBox.information(self, "Update", "You are already running the latest version.")

    def _manual_update_error(self, error_msg):
        self.update_msg.close()
        QMessageBox.warning(self, "Update Error", f"Could not check for updates:\n{error_msg}")

    def show_feedback_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Feedback / Report Bug")
        dialog.setFixedWidth(440)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(16)
        
        info_label = QLabel(
            "<h3 style='color: #a5b4fc;'>💬 Found a bug or have feedback?</h3>"
            "<p>We appreciate your feedback! If you are experiencing issues or have suggestions, please email us directly.</p>"
            "<p><b>Email:</b> <a href='mailto:meeranrashith166@gmail.com' style='color: #8b5cf6;'>meeranrashith166@gmail.com</a></p>"
            "<br>"
            "<i>Your feedback helps us make BG Remover better!</i>"
        )
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        info_label.setOpenExternalLinks(True)
        info_label.setFont(QFont("Inter", 10))
        layout.addWidget(info_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        email_btn = QPushButton("✉  Send Feedback via Email")
        email_btn.setObjectName("primaryBtn")
        email_btn.setFont(QFont("Inter", 10, QFont.Bold))
        email_btn.clicked.connect(lambda: webbrowser.open("mailto:meeranrashith166@gmail.com?subject=BG Remover Feedback"))
        
        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondaryBtn")
        close_btn.setFont(QFont("Inter", 10))
        close_btn.clicked.connect(dialog.accept)
        
        btn_layout.addWidget(email_btn)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def create_icon_button(self, text, icon_file, callback, font, object_name=None):
        btn = QPushButton(text)
        icon_path = os.path.join(ASSETS_DIR, icon_file)
        if os.path.exists(icon_path):
            btn.setIcon(QIcon(icon_path))
            from PySide6.QtCore import QSize
            btn.setIconSize(QSize(24, 24))
        btn.setFont(font)
        btn.clicked.connect(callback)
        if object_name:
            btn.setObjectName(object_name)
        return btn

    def get_translation(self, key):
        # ... (function is unchanged)
        lang = self.language_select.currentText()
        return LANGUAGES.get(lang, LANGUAGES["English"]).get(key, key)
    
    def update_language(self): self.update_ui_texts()
    def update_ui_texts(self):
        # ... (function is unchanged)
        self.model_label.setText(self.get_translation("model") + ":")
        self.feather_label.setText(self.get_translation("feather") + ":")
        self.language_label.setText(self.get_translation("language") + ":")
        self.browse_btn.setText(self.get_translation("browse"))
        self.remove_btn.setText(self.get_translation("remove_bg"))
        self.undo_btn.setText(self.get_translation("undo"))
        self.redo_btn.setText(self.get_translation("redo"))
        self.save_btn.setText(self.get_translation("save"))
        self.zoom_in_btn.setText(self.get_translation("zoom_in"))
        self.zoom_out_btn.setText(self.get_translation("zoom_out"))
        self.theme_btn.setText(self.get_translation("switch_theme"))
        
    def load_image(self):
        # ... (function is unchanged)
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        if file_path:
            self.load_image_from_path(file_path)

    def clear_image(self):
        # ... (function is unchanged)
        self.image_np = None
        self.result_np = None
        self.original_image_np = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.left_image_label.clear()
        self.right_image_label.clear()
        self.left_image_label.setText("✨  Processed Image")
        self.right_image_label.setText("📂  Drop or Browse an Image")
        self.left_image_label.resize(self.left_scroll_area.size())
        self.right_image_label.resize(self.right_scroll_area.size())
        
        # Reset cursors
        self.left_image_label.setCursor(Qt.ArrowCursor)
        self.right_image_label.setCursor(Qt.ArrowCursor)

    def update_image_display(self):
        if self.image_np is None:
            self.clear_image()
            return
            
        # Left Image (Processed)
        left_img = self.result_np if self.result_np is not None else self.image_np
        left_img = safe_to_uint8(left_img)
        if left_img.ndim == 3:
            if left_img.shape[2] == 4: fmt = QImage.Format_RGBA8888
            else: fmt = QImage.Format_RGB888
            qimg_left = QImage(left_img.data, left_img.shape[1], left_img.shape[0], left_img.shape[1] * left_img.shape[2], fmt)
        else:
            qimg_left = QImage(left_img.data, left_img.shape[1], left_img.shape[0], left_img.shape[1], QImage.Format_Grayscale8)
        
        pixmap_left = QPixmap.fromImage(qimg_left)
        scaled_pixmap_left = pixmap_left.scaled(
            int(pixmap_left.width() * self.zoom_factor), int(pixmap_left.height() * self.zoom_factor),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.left_image_label.setPixmap(scaled_pixmap_left)
        self.left_image_label.resize(scaled_pixmap_left.size())

        # Right Image (Original)
        right_img = self.original_image_np
        right_img = safe_to_uint8(right_img)
        if right_img.ndim == 3:
            if right_img.shape[2] == 4: fmt = QImage.Format_RGBA8888
            else: fmt = QImage.Format_RGB888
            qimg_right = QImage(right_img.data, right_img.shape[1], right_img.shape[0], right_img.shape[1] * right_img.shape[2], fmt)
        else:
            qimg_right = QImage(right_img.data, right_img.shape[1], right_img.shape[0], right_img.shape[1], QImage.Format_Grayscale8)
            
        pixmap_right = QPixmap.fromImage(qimg_right)
        scaled_pixmap_right = pixmap_right.scaled(
            int(pixmap_right.width() * self.zoom_factor), int(pixmap_right.height() * self.zoom_factor),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.right_image_label.setPixmap(scaled_pixmap_right)
        self.right_image_label.resize(scaled_pixmap_right.size())

    def remove_background(self):
        # ... (function is unchanged)
        if self.image_np is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        selected_model_text = self.model_select.currentText()
        if selected_model_text.startswith("BEN2") and not has_ben2:
            QMessageBox.warning(self, "Model not installed", "BEN2 is not installed on this system. See the app instructions for installation.")
            return
        model_name = selected_model_text.replace(" (not installed)", "").replace(" (missing model)", "")
        self.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.thread = RemoveBackgroundThread(self.image_np, model_name, self.feather_spinbox.value())
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.remove_bg_finished)
        self.thread.error.connect(self.log_error)
        self.thread.finished.connect(lambda: self.setEnabled(True))
        self.thread.finished.connect(lambda: self.progress.setVisible(False))
        self.thread.start()

    def remove_bg_finished(self, result):
        # ... (function is unchanged)
        if result is None: return
        self.result_np = result
        if self.result_np.ndim == 3 and self.result_np.shape[2] == 3:
            self.result_np = np.dstack([self.result_np, np.full(self.result_np.shape[:2], 255, np.uint8)])
        self.undo_stack.append(self.result_np.copy())
        self.redo_stack.clear()
        self.zoom_factor = 1.0
        self.update_image_display()
    
    # --- NEW SECURITY ENHANCEMENT: Filename Sanitization ---
    def sanitize_filename(self, filename):
        """Removes illegal characters from a filename."""
        # Remove directory traversal attempts
        safe_base = os.path.basename(filename)
        # Remove characters that are invalid in Windows/Linux/macOS filenames
        return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', safe_base)
    # --- END NEW SECURITY ENHANCEMENT ---

    def save_image(self):
        if self.result_np is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return

        # --- NEW SECURITY ENHANCEMENT: Suggest a sanitized filename and original directory ---
        original_basename = ""
        default_dir = ""
        if self.image_path:
            name, _ = os.path.splitext(os.path.basename(self.image_path))
            original_basename = self.sanitize_filename(name + "_removed_bg")
            default_dir = os.path.dirname(self.image_path)
        # --- END NEW SECURITY ENHANCEMENT ---
        
        default_save_path = os.path.join(default_dir, original_basename) if default_dir else original_basename

        filename, ext = QFileDialog.getSaveFileName(
            self, "Save Image", default_save_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF Flat (*.tiff *.tif);;TIFF Layered (*.tiff *.tif);;PSD (*.psd)"
        )

        if filename:
            # --- SECURITY ENHANCEMENT: Validate save path ---
            if not self.is_safe_save_path(filename):
                QMessageBox.critical(self, "Save Error", "Cannot save to a protected system directory.")
                return
            # --- END SECURITY ENHANCEMENT ---
            try:
                if "TIFF Flat" in ext: self.save_tiff_flat(filename)
                elif "TIFF Layered" in ext or ".psd" in ext.lower(): self.save_layered(filename)
                else: self.save_flat_image(filename)
                QMessageBox.information(self, "Saved", f"Image saved successfully:\n{filename}")
            except Exception as e:
                # --- NEW SECURITY ENHANCEMENT: Secure Error Handling ---
                logging.error(f"Failed to save image to {filename}", exc_info=True)
                QMessageBox.critical(self, "Save Error", f"Failed to save image: {str(e)}\n\nSee log at:\n{log_file_path}")
                # --- END NEW SECURITY ENHANCEMENT ---
    
    # --- SECURITY ENHANCEMENT: Safe Save Path Check ---
    def is_safe_save_path(self, filepath):
        """Checks if the proposed save path is in a restricted directory."""
        # Get the absolute, normalized path to avoid traversal attacks (e.g., "C:/Users/../Windows")
        abs_path = os.path.abspath(os.path.normpath(filepath))
        
        # Define a list of restricted system directories
        restricted_paths = []
        if sys.platform == "win32":
            # e.g., C:\Windows, C:\Program Files
            restricted_paths.extend([os.environ.get("SystemRoot", "C:\\Windows"), os.environ.get("ProgramFiles", "C:\\Program Files")])
        elif sys.platform == "darwin": # macOS
             restricted_paths.extend(["/System", "/Library", "/bin", "/sbin"])
        else: # Linux
             restricted_paths.extend(["/etc", "/boot", "/bin", "/sbin", "/usr/bin"])
        
        # Normalize restricted paths as well
        restricted_paths = [os.path.abspath(os.path.normpath(p)) for p in restricted_paths if p]

        for restricted in restricted_paths:
            if abs_path.lower().startswith(restricted.lower()):
                return False
        return True
    # --- END SECURITY ENHANCEMENT ---

    def save_flat_image(self, filename):
        # ... (function is unchanged)
        ext = os.path.splitext(filename)[1].lower()
        img_array = self.result_np.copy()
        if img_array.shape[2] == 4:
            if ext in [".jpg", ".jpeg", ".bmp"]:
                alpha = img_array[:, :, 3] / 255.0
                img_rgb = img_array[:, :, :3].astype(np.float32)
                background = np.ones_like(img_rgb, dtype=np.float32) * 255
                img_rgb = (img_rgb * alpha[..., None] + background * (1 - alpha[..., None])).astype(np.uint8)
                img_to_save = Image.fromarray(img_rgb)
            else:
                img_to_save = Image.fromarray(img_array)
        else:
            img_to_save = Image.fromarray(img_array)
        save_kwargs = {}
        if ext in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = 100
            save_kwargs['subsampling'] = 0
        img_to_save.save(filename, **save_kwargs)
        
    def save_tiff_flat(self, filename):
        # ... (function is unchanged)
        img_orig = Image.fromarray(self.original_image_np.astype(np.uint8)).convert("RGBA")
        img_result = Image.fromarray(self.result_np.astype(np.uint8)).convert("RGBA")
        img_orig.save(filename, save_all=True, append_images=[img_result], compression='tiff_deflate')

    def save_layered(self, filename):
        # ... (function is unchanged)
        orig = self.original_image_np.astype(np.uint8)
        proc = self.result_np.astype(np.uint8)
        if 'pytoshop.compression' in sys.modules:
            sys.modules['pytoshop.compression'].packbits = packbits
        if orig.shape[2] == 3: orig = np.dstack([orig, np.full(orig.shape[:2], 255, np.uint8)])
        if proc.shape[2] == 3: proc = np.dstack([proc, np.full(proc.shape[:2], 255, np.uint8)])
        alpha = proc[:, :, 3]
        proc[alpha == 0, 0:3] = 0
        h, w = orig.shape[:2]
        def to_channels_rgba(arr): return { 0: arr[:, :, 0].copy(), 1: arr[:, :, 1].copy(), 2: arr[:, :, 2].copy(), -1: arr[:, :, 3].copy() }
        orig_channels = to_channels_rgba(orig)
        proc_channels = to_channels_rgba(proc)
        layer_orig = nested_layers.Image(name="Original", channels=orig_channels, top=0, left=0, visible=False, color_mode=enums.ColorMode.rgb)
        layer_removed = nested_layers.Image(name="Background Removed", channels=proc_channels, top=0, left=0, visible=True, color_mode=enums.ColorMode.rgb)
        psd = nested_layers.nested_layers_to_psd([layer_orig, layer_removed], color_mode=enums.ColorMode.rgb, depth=enums.ColorDepth.depth8, size=(w, h), compression=enums.Compression.rle)
        with open(filename, "wb") as fd: psd.write(fd)

    def undo(self):
        # ... (function is unchanged)
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            last_state = self.undo_stack[-1]
            self.result_np = last_state.copy() if last_state.shape[0] > 0 else None
            self.update_image_display()
    
    def redo(self):
        # ... (function is unchanged)
        if self.redo_stack:
            img = self.redo_stack.pop()
            self.undo_stack.append(img)
            self.result_np = img.copy()
            self.update_image_display()

    def zoom_in(self): self.zoom_factor *= 1.25; self.update_image_display()
    def zoom_out(self): self.zoom_factor /= 1.25; self.update_image_display()
    
    def toggle_theme(self):
        if self.is_dark:
            self.setStyleSheet(LIGHT_STYLESHEET)
            self.is_dark = False
        else:
            self.setStyleSheet(DARK_STYLESHEET)
            self.is_dark = True
            
    def start_drag(self, event):
        # Allow dragging if right labeled image event triggers
        if event.button() == Qt.LeftButton:
            source = self.sender()
            # If sender fails to resolve to right component, check based on which has focus or determine statically.
            # Using widget under mouse
            widget = QApplication.widgetAt(event.globalPos())
            if isinstance(widget, QLabel) and widget.pixmap() and not widget.pixmap().isNull():
                self.drag_pos = event.pos()
                widget.setCursor(Qt.ClosedHandCursor)
    
    def drag_image(self, event):
        if self.drag_pos:
            delta = event.pos() - self.drag_pos
            if not self._is_syncing:
                self._is_syncing = True
                
                # Update left scroll area
                self.left_scroll_area.horizontalScrollBar().setValue(self.left_scroll_area.horizontalScrollBar().value() - delta.x())
                self.left_scroll_area.verticalScrollBar().setValue(self.left_scroll_area.verticalScrollBar().value() - delta.y())
                
                # Update right scroll area
                self.right_scroll_area.horizontalScrollBar().setValue(self.right_scroll_area.horizontalScrollBar().value() - delta.x())
                self.right_scroll_area.verticalScrollBar().setValue(self.right_scroll_area.verticalScrollBar().value() - delta.y())
                
                self._is_syncing = False
            self.drag_pos = event.pos()
            
    def end_drag(self, event): 
        self.drag_pos = None
        widget = QApplication.widgetAt(event.globalPos())
        if isinstance(widget, QLabel) and widget.pixmap() and not widget.pixmap().isNull():
            widget.setCursor(Qt.OpenHandCursor)
    
    def log_error(self, msg):
        # --- NEW SECURITY ENHANCEMENT: Secure Error Handling ---
        # Log the detailed message and show a generic one in the UI.
        logging.error(f"Error displayed to user: {msg}")
        QMessageBox.critical(self, "Error", msg)
        # --- END NEW SECURITY ENHANCEMENT ---
        self.setEnabled(True)
        self.progress.setVisible(False)

    def eventFilter(self, source, event):
        # Check for drag events for left label
        if event.type() == QEvent.DragEnter and source in (self.left_image_label, self.right_image_label):
            if event.mimeData().hasUrls(): event.acceptProposedAction()
            return True
        if event.type() == QEvent.Drop and source in (self.left_image_label, self.right_image_label):
            urls = event.mimeData().urls()
            if urls: self.load_image_from_path(urls[0].toLocalFile())
            return True
        if source in (self.left_image_label, self.right_image_label) and event.type() == QEvent.Wheel:
            if event.angleDelta().y() > 0: self.zoom_in()
            else: self.zoom_out()
            return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        # ... (function is unchanged)
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_O: self.load_image()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S: self.save_image()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_R: self.remove_background()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z: self.undo()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Y: self.redo()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Plus: self.zoom_in()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Minus: self.zoom_out()
        elif event.key() == Qt.Key_Delete: self.clear_image()
        else: super().keyPressEvent(event)
            
    def load_image_from_path(self, path):
        # --- NEW SECURITY ENHANCEMENT: Robust Input Validation ---
        # 1. Check file existence.
        if not os.path.exists(path):
            QMessageBox.critical(self, "Error Loading Image", f"File does not exist:\n{path}")
            return

        # 2. Check MIME type to ensure it's a recognized image format before processing.
        # This is more reliable than just checking the file extension.
        allowed_mime_types = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type not in allowed_mime_types:
            QMessageBox.critical(self, "Error Loading Image", f"Unsupported file type: {mime_type}.\nPlease select a valid image (PNG, JPG, BMP, TIFF).")
            return
        # --- END NEW SECURITY ENHANCEMENT ---

        try:
            # Using Pillow to open first can catch more format errors and apply MAX_IMAGE_PIXELS
            with Image.open(path) as pil_img:
                pil_img.load() # Read data to trigger potential errors
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if img is None:
                raise ValueError("File could not be read. It might be corrupted or in an unsupported format.")
            
            self.image_path = path # Store the path for later use (e.g., saving)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image_np = img
            self.original_image_np = img.copy()
            self.result_np = None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.undo_stack.append(self.original_image_np.copy())
            self.zoom_factor = 1.0
            
            # Set cursor to OpenHand when image is loaded
            self.left_image_label.setCursor(Qt.OpenHandCursor)
            self.right_image_label.setCursor(Qt.OpenHandCursor)
            
            self.update_image_display()
        except Image.DecompressionBombError:
            QMessageBox.critical(self, "Error Loading Image", f"Image is too large to process safely.\n{path}")
        except Exception as e:
            # --- NEW SECURITY ENHANCEMENT: Secure Error Handling ---
            logging.error(f"Failed to load image from path: {path}", exc_info=True)
            QMessageBox.critical(self, "Error Loading Image", f"Failed to load image: {str(e)}\n\nSee log at:\n{log_file_path}")
            # --- END NEW SECURITY ENHANCEMENT ---


def refresh_desktop_shortcut():
    """Forces Windows to refresh the desktop shortcut icon cache for BG Remover."""
    if not has_pywin32:
        return
        
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut_name = "BG Remover.lnk"
        
        # Check both User Desktop and Public Desktop
        desktop_paths = [
            shell.SpecialFolders("Desktop"),
            shell.SpecialFolders("AllUsersDesktop")
        ]
        
        shortcut_refreshed = False
        for desktop_path in desktop_paths:
            if not desktop_path:
                continue
            shortcut_path = os.path.join(desktop_path, shortcut_name)
            if os.path.exists(shortcut_path):
                # Load shortcut and save it to update its "modified" timestamp
                try:
                    shortcut = shell.CreateShortCut(shortcut_path)
                    shortcut.Save() # Touching the file
                    shortcut_refreshed = True
                    logging.info(f"Refreshed shortcut timestamp at: {shortcut_path}")
                except Exception as e:
                    logging.warning(f"Could not save shortcut at {shortcut_path}: {e}")
        
        if shortcut_refreshed:
            # Notify Windows Explorer to rebuild icon cache
            import ctypes
            # SHCNE_ASSOCCHANGED = 0x08000000, SHCNF_IDLIST = 0x0000
            SHCNE_ASSOCCHANGED = 0x08000000
            SHCNF_IDLIST = 0x0000
            ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
            logging.info("Sent SHChangeNotify to refresh icon cache.")
            
    except Exception as e:
        logging.warning(f"Error refreshing desktop shortcut: {e}")


if __name__ == "__main__":
    refresh_desktop_shortcut()
    app = QApplication(sys.argv)
    window = BackgroundRemoverApp()
    window.show()
    # Start the update check in a separate thread to not block the UI
    # update_thread = threading.Thread(target=check_for_updates_async, daemon=True)
    # update_thread.start()
    sys.exit(app.exec_())
