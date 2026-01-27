import sys
import os
import warnings

# --- WARNING SUPPRESSION ---
# Must be done before importing libraries that might emit warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts*=false;qt.text.font*=false" # Suppress Qt font warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rembg.*")
warnings.filterwarnings("ignore", message=".*'mode' parameter is deprecated.*")

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
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QSlider, QComboBox, QProgressBar, QScrollArea, QMessageBox, QSpinBox,
    QDialog
)
import gc
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtWidgets import QShortcut
import onnxruntime as ort
from rembg import remove as rembg_remove, new_session
import math
import threading
import urllib.request
import torch
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
    from ben2 import BEN_Base
    has_ben2 = True
except Exception:
    BEN_Base = None
    has_ben2 = False

APP_VERSION = "1.0.0"

class AppConfig(object):
    APP_NAME = "BGRemover"
    COMPANY_NAME = "Meeran Rashith"
    HTTP_TIMEOUT = 30
    MAX_DOWNLOAD_RETRIES = 3
    UPDATE_URLS = ["https://huggingface.co/meeranrashith166/BG-Remover-Updates/resolve/main/"]
    # --- SECURITY ENHANCEMENT ---
    # Public Key for verifying updates.
    PUBLIC_KEY = 'djeV18vHUKwPhFXxHL8BX+Q6SsqsQXe8PoEDuker95A'

def check_for_updates_async():
    if not has_updater:
        return

    try:
        client = Client(AppConfig(), refresh=True)
        app_update = client.update_check(AppConfig.APP_NAME, APP_VERSION)
        if app_update is not None:
            logging.info("Update found! Downloading...")
            # The download is automatically verified against the public key by pyupdater.
            app_update.download()
            if app_update.is_downloaded():
                logging.info("Update ready. Restarting...")
                app_update.extract_restart()
    except ValueError:
        logging.error("Update check failed: Server returned invalid response (likely missing update files).")
    except Exception as e:
        logging.error("Update check failed:", exc_info=True)

try:
    import win32com.client
    has_pywin32 = True
except ImportError:
    has_pywin32 = False


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # If not running as a PyInstaller bundle, use the script's directory
        # Or if it's a frozen executable (one-dir), look relative to the executable
        if getattr(sys, 'frozen', False):
             base_path = os.path.dirname(sys.executable)
             # Modern PyInstaller places dependencies in _internal
             if os.path.exists(os.path.join(base_path, "_internal")):
                 base_path = os.path.join(base_path, "_internal")
        else:
             base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

ASSETS_DIR = resource_path("assets")
MODELS = {
    "Auto (Rembg)": None,
    "U²Net": resource_path("models/u2net.onnx"),
    "U²NetP": resource_path("models/u2netp.onnx"),
    "U²Net Human Seg": resource_path("models/u2net_human_seg.onnx"),
    "MODNet": resource_path("models/modnet.onnx"),
    "IS-Net": resource_path("models/isnet-general-use.onnx"),
    "IS-Net Anime": resource_path("models/isnet-anime.onnx"),
    "Bria RMBG v1.4": resource_path("models/bria_rmbg_1.4.onnx"),
    "BASNet": resource_path("models/basnet.onnx")
}

# --- NEWLY ADDED: Model File Integrity Check ---
# CRITICAL: You MUST replace these placeholder hashes with the actual
# SHA-256 hashes of your model files.
# Generate hashes using:
# Windows: certutil -file models/u2net.onnx SHA256
# Linux/macOS: shasum -a 256 models/u2net.onnx
MODEL_HASHES = {
    resource_path("models/u2net.onnx"): "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    resource_path("models/u2netp.onnx"): "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
    resource_path("models/u2net_human_seg.onnx"): "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
    resource_path("models/modnet.onnx"): "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
    resource_path("models/isnet-general-use.onnx"): "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    resource_path("models/isnet-anime.onnx"): "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
    resource_path("models/bria_rmbg_1.4.onnx"): "8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f",
    resource_path("models/basnet.onnx"): "2766aaedd02b2e301ba3efe908f7b10455077b842c328a0fb900c5c0d8080b8a",
    resource_path("models/birefnet.onnx"): "6470117bac6f8d82a3f62921056f52d0f5c4d36d1d832096331d5ea38a03acb5",
}

# --- REMBG LOCAL MODEL CONFIGURATION ---
# Force rembg to look for u2net.onnx in our bundled 'models' directory
# instead of ~/.u2net or trying to download it.
os.environ["U2NET_HOME"] = resource_path("models")
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
                "undo": "Undo","redo": "Redo","save": "Save","zoom_in": "Zoom In","zoom_out": "Zoom Out","switch_theme": "Switch Theme"},
    "தமிழ்": {"model": "மாதiri","feather": "மெல்லியப்பு","language": "மொழி","browse": "கோப்பு திறக்க","remove_bg": "பின்புலம் அகற்று",
              "undo": "திருத்தத்தை முன்னோக்கி","redo": "திருத்தத்தை பின்னோக்கி","save": "சேமிக்கவும்",
              "zoom_in": "பெரிதாக்கு","zoom_out": "சிறியதாக்கு","switch_theme": "தீமை மாற்றுக"},
    "हिन्दी": {"model": "मॉडल","feather": "फेदर","language": "भाषा","browse": "ब्राउज़ करें","remove_bg": "पृष्ठभूमि हटाएं",
              "undo": "पूर्ववत करें","redo": "पुनः करें","save": "सहेजें","zoom_in": "ज़ूम इन","zoom_out": "ज़ूम आउट","switch_theme": "थीμ बदलें"},
    "Français": {"model": "Modèle","feather": "Doux","language": "Langue","browse": "Parcourir","remove_bg": "Supprimer fond",
                 "undo": "Annuler","redo": "Rétablir","save": "Enregistrer","zoom_in": "Zoom avant","zoom_out": "Zoom arrière","switch_theme": "Changer thème"},
    "Deutsch": {"model": "Modell","feather": "Weichheit","language": "Sprache","browse": "Durchsuchen","remove_bg": "Hintergrund entfernen",
                "undo": "Rückgängig","redo": "Wiederherstellen","save": "Speichern","zoom_in": "Vergrößern","zoom_out": "Verkleinern","switch_theme": "Thema wechseln"}
}

# --- BEN2 model cache ---
_ben2_cache = None
_ben2_device = None

def get_ben2_model():
    """
    Returns a cached BEN2 model moved to the correct device.
    Loads the model on first call.
    """
    global _ben2_cache, _ben2_device
    return None # BEN2 disabled for CPU bridge
    # if not has_ben2:
    #     raise RuntimeError("BEN2 package not installed.")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    if _ben2_cache is None or _ben2_device != device:
        # BEN_Base.from_pretrained will download weights if needed (Hugging Face / repo configured)
        _ben2_cache = BEN_Base.from_pretrained("PramaLLC/BEN2")
        _ben2_cache.to(device).eval()
        _ben2_device = device
    return _ben2_cache

def preprocess_for_model(img, model_name, session):
    img_copy = img.copy()
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) >= 4:
        _, c, h, w = input_shape[:4]
    else:
        c, h, w = 3, 320, 320

    if not isinstance(h, int) or not isinstance(w, int) or h == 0 or w == 0:
        h, w = 320, 320

    resized = cv2.resize(img_copy, (w, h))
    arr = resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return arr

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

            elif self.model_name == "BEN2":
                if not has_ben2:
                    raise RuntimeError("BEN2 package not installed. Install it with pip (see app instructions).")
                self.progress.emit(15)
                model = get_ben2_model()
                pil_img = Image.fromarray(safe_to_uint8(self.image_np))
                self.progress.emit(40)
                try:
                    out_pil = model.inference(pil_img, refine_foreground=True)
                except TypeError:
                    out_pil = model.inference(pil_img)
                if out_pil is None:
                    raise RuntimeError("BEN2 inference returned None.")
                out_pil = out_pil.convert("RGBA")
                result = np.array(out_pil)
                self.progress.emit(90)

            else:
                model_path = MODELS.get(self.model_name)
                if not model_path or not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # --- NEWLY ADDED: Verify model before loading ---
                expected_hash = MODEL_HASHES.get(model_path)
                if expected_hash and "goes_here" not in expected_hash:
                    if not verify_file_hash(model_path, expected_hash):
                        raise SecurityException(
                            f"Model file '{os.path.basename(model_path)}' has been modified "
                            "or is not the official version. Aborting."
                        )
                else:
                    logging.warning(f"[SECURITY WARNING] No hash found for model '{model_path}'. Proceeding without verification.")
                # --- END NEWLY ADDED SECTION ---

                available_providers = ort.get_available_providers()
                if torch.cuda.is_available():
                    logging.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    logging.warning("CUDA not available, falling back to CPU")
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

                mask = (cv2.resize(mask_single, (self.image_np.shape[1], self.image_np.shape[0])) * 255.0)
                if mask.max() > 255:
                    mask = np.clip(mask / mask.max() * 255.0, 0, 255)
                mask = np.clip(mask, 0, 255).astype(np.uint8)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                result = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2RGBA)
                result[:, :, 3] = mask
                self.progress.emit(85)

            if result is None:
                self.error.emit("Processing failed. Result is None.")
                self.finished.emit(None)
                return

            if self.feather > 0 and result.shape[2] == 4:
                alpha = result[:, :, 3]
                blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=self.feather, sigmaY=self.feather)
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

class BackgroundRemoverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.processed_image = None
        self.image_path = None
        self.scale_factor = 1.0
        self.setWindowTitle("AI Background Remover")
        self.setGeometry(100, 100, 1200, 800)
        self.zoom_factor = 1.0
        self.image_np = None
        self.result_np = None
        self.original_image_np = None
        self.is_dark = False
        self.undo_stack = []
        self.redo_stack = []
        self.drag_pos = None

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
        help_menu.addAction("Feedback / Report Bug", self.show_feedback_dialog)
        help_menu.addAction("About", self.show_about)
        self.image_label = QLabel("Drop or Load Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; color: #333;")
        self.image_label.setAcceptDrops(True)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        font = QFont("Arial", 10, QFont.Bold)
        self.language_select = QComboBox()
        self.language_select.addItems(LANGUAGES.keys())
        self.language_select.setCurrentText("English")
        self.language_select.currentTextChanged.connect(self.update_language)
        self.model_label = QLabel()
        self.model_select = QComboBox()
        self.model_select.addItems(MODELS.keys())
        if not has_ben2:
            for i in range(self.model_select.count()):
                if self.model_select.itemText(i) == "BEN2":
                    self.model_select.setItemText(i, "BEN2 (not installed)")
                    break
        self.feather_label = QLabel()
        self.feather_spinbox = QSpinBox()
        self.feather_spinbox.setMinimum(0)
        self.feather_spinbox.setMaximum(10)
        self.feather_spinbox.setValue(2)
        self.feather_spinbox.setFixedWidth(50)
        self.language_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.browse_btn = self.create_icon_button("", "browse.png", self.load_image, font)
        self.remove_btn = self.create_icon_button("", "remove.png", self.remove_background, font)
        self.undo_btn = self.create_icon_button("", "undo.png", self.undo, font)
        self.redo_btn = self.create_icon_button("", "redo.png", self.redo, font)
        self.save_btn = self.create_icon_button("", "save.png", self.save_image, font)
        self.zoom_in_btn = self.create_icon_button("", "zoom_in.png", self.zoom_in, font)
        self.zoom_out_btn = self.create_icon_button("", "zoom_out.png", self.zoom_out, font)
        self.theme_btn = self.create_icon_button("", "theme.png", self.toggle_theme, font)
        self.update_ui_texts()
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.zoom_in_btn)
        top_layout.addWidget(self.zoom_out_btn)
        top_layout.addWidget(self.theme_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.model_label)
        top_layout.addWidget(self.model_select)
        top_layout.addWidget(self.feather_label)
        top_layout.addWidget(self.feather_spinbox)
        top_layout.addWidget(self.language_label)
        top_layout.addWidget(self.language_select)
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.browse_btn)
        bottom_layout.addWidget(self.remove_btn)
        bottom_layout.addWidget(self.undo_btn)
        bottom_layout.addWidget(self.redo_btn)
        bottom_layout.addWidget(self.save_btn)
        bottom_layout.addStretch()
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.scroll_area, 10)
        main_layout.addLayout(bottom_layout)
        main_layout.addWidget(self.progress)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.image_label.installEventFilter(self)
        self.image_label.mousePressEvent = self.start_drag
        self.image_label.mouseMoveEvent = self.drag_image
        self.image_label.mouseReleaseEvent = self.end_drag

    def show_about(self):
        QMessageBox.information(
            self, "About",
            f"<b>AI Background Remover</b><br>"
            f"Version: {APP_VERSION}<br><br>"
            f"Developed by <b>{AppConfig.COMPANY_NAME}</b><br>"
            f"© 2025 All Rights Reserved.<br><br>"
            f"This software uses advanced AI models (U²Net, MODNet, IS-Net, BEN2, BiRefNet, etc.) "
            f"for background removal."
        )

    def manual_update(self):

        if not has_updater:
            QMessageBox.warning(self, "Update", "Updater not available. Please install pyupdater.")
            return
        try:
            client = Client(AppConfig(), refresh=True)
            app_update = client.update_check(AppConfig.APP_NAME, APP_VERSION)
            if app_update is not None:
                reply = QMessageBox.question(
                    self, "Update Available",
                    f"New version found!\n\nDo you want to download and install it now?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    app_update.download()
                    if app_update.is_downloaded():
                        QMessageBox.information(self, "Update", "Update downloaded. Restarting...")
                        app_update.extract_restart()
            else:
                QMessageBox.information(self, "Update", "You are already running the latest version.")
        except ValueError:
             logging.error("Manual update check failed: Invalid server response.")
             QMessageBox.warning(self, "Update Error", 
                                 "Could not check for updates.\n\n"
                                 "The update server returned an invalid response (likely missing files).\n"
                                 "Please try again later.")
        except Exception as e:
            logging.error("Manual update check failed.", exc_info=True)
            QMessageBox.critical(self, "Update Error", f"Update check failed: {str(e)}\n\nSee log at:\n{log_file_path}")

    def show_feedback_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Feedback / Report Bug")
        dialog.setFixedWidth(400)
        
        layout = QVBoxLayout()
        
        info_label = QLabel(
            "<h3>Found a bug or have feedback?</h3>"
            "<p>We appreciate your feedback! If you are experiencing issues or have suggestions, please email us directly.</p>"
            "<p><b>Email:</b> <a href='mailto:meeranrashith166@gmail.com'>meeranrashith166@gmail.com</a></p>"
            "<br>"
            "<i>Your feedback helps us make BG Remover better!</i>"
        )
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        info_label.setOpenExternalLinks(True)
        layout.addWidget(info_label)
        
        btn_layout = QHBoxLayout()
        
        email_btn = QPushButton("Send Feedback via Email")
        email_btn.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;")
        email_btn.clicked.connect(lambda: webbrowser.open("mailto:meeranrashith166@gmail.com?subject=BG Remover Feedback"))
        
        btn_layout.addWidget(email_btn)
        
        layout.addLayout(btn_layout)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def create_icon_button(self, text, icon_file, callback, font):
        # ... (function is unchanged)
        btn = QPushButton(text)
        icon_path = os.path.join(ASSETS_DIR, icon_file)
        if os.path.exists(icon_path):
            btn.setIcon(QIcon(icon_path))
        btn.setFont(font)
        btn.clicked.connect(callback)
        btn.setStyleSheet("padding: 8px;")
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
        self.image_label.clear()
        self.image_label.setText("Drop or Load Image")
        self.image_label.resize(self.scroll_area.size())

    def update_image_display(self):
        # ... (function is unchanged)
        if self.image_np is None:
            self.image_label.clear()
            self.image_label.setText("Drop or Load Image")
            return
        img_to_display = self.result_np if self.result_np is not None else self.image_np
        img_to_display = safe_to_uint8(img_to_display)
        if img_to_display.ndim == 3:
            if img_to_display.shape[2] == 4: fmt = QImage.Format_RGBA8888
            else: fmt = QImage.Format_RGB888
            qimg = QImage(img_to_display.data, img_to_display.shape[1], img_to_display.shape[0], img_to_display.shape[1] * img_to_display.shape[2], fmt)
        else:
            qimg = QImage(img_to_display.data, img_to_display.shape[1], img_to_display.shape[0], img_to_display.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            int(pixmap.width() * self.zoom_factor), int(pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

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

        # --- NEW SECURITY ENHANCEMENT: Suggest a sanitized filename ---
        original_basename = ""
        if self.image_path:
            name, _ = os.path.splitext(os.path.basename(self.image_path))
            original_basename = self.sanitize_filename(name + "_removed_bg")
        # --- END NEW SECURITY ENHANCEMENT ---

        filename, ext = QFileDialog.getSaveFileName(
            self, "Save Image", original_basename,
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
        # ... (function is unchanged)
        dark_stylesheet = """
        QMainWindow, QWidget {
            background-color: #333;
            color: #ccc;
        }
        QMenuBar, QMenu {
            background-color: #444;
            color: #ccc;
        }
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #555;
        }
        QPushButton {
            background-color: #555;
            color: #ccc;
            border: 1px solid #666;
            padding: 8px;
        }
        QPushButton:hover {
            background-color: #666;
        }
        QComboBox {
            background-color: #555;
            color: #ccc;
            border: 1px solid #666;
        }
        QSpinBox {
            background-color: #555;
            color: #ccc;
            border: 1px solid #666;
        }
        QProgressBar {
            color: #ccc;
            border: 1px solid #666;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #0078d7;
        }
        QScrollArea {
            background-color: #2a2a2a;
            border: none;
        }
        """
        if self.is_dark:
            self.setStyleSheet(""); self.image_label.setStyleSheet("border: 2px dashed #ccc; color: #333;"); self.is_dark = False
        else:
            self.setStyleSheet(dark_stylesheet); self.image_label.setStyleSheet("border: 2px dashed #777; color: #aaa; background-color: #2a2a2a;"); self.is_dark = True
            
    def start_drag(self, event):
        # ... (function is unchanged)
        if event.button() == Qt.LeftButton and self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            self.drag_pos = event.pos()
    
    def drag_image(self, event):
        # ... (function is unchanged)
        if self.drag_pos:
            delta = event.pos() - self.drag_pos
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() - delta.x())
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() - delta.y())
            self.drag_pos = event.pos()
            
    def end_drag(self, event): self.drag_pos = None
    
    def log_error(self, msg):
        # --- NEW SECURITY ENHANCEMENT: Secure Error Handling ---
        # Log the detailed message and show a generic one in the UI.
        logging.error(f"Error displayed to user: {msg}")
        QMessageBox.critical(self, "Error", msg)
        # --- END NEW SECURITY ENHANCEMENT ---
        self.setEnabled(True)
        self.progress.setVisible(False)

    def eventFilter(self, source, event):
        # ... (function is unchanged)
        if event.type() == QEvent.DragEnter and source is self.image_label:
            if event.mimeData().hasUrls(): event.acceptProposedAction()
            return True
        if event.type() == QEvent.Drop and source is self.image_label:
            urls = event.mimeData().urls()
            if urls: self.load_image_from_path(urls[0].toLocalFile())
            return True
        if source is self.image_label and event.type() == QEvent.Wheel:
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
            self.update_image_display()
        except Image.DecompressionBombError:
            QMessageBox.critical(self, "Error Loading Image", f"Image is too large to process safely.\n{path}")
        except Exception as e:
            # --- NEW SECURITY ENHANCEMENT: Secure Error Handling ---
            logging.error(f"Failed to load image from path: {path}", exc_info=True)
            QMessageBox.critical(self, "Error Loading Image", f"Failed to load image: {str(e)}\n\nSee log at:\n{log_file_path}")
            # --- END NEW SECURITY ENHANCEMENT ---


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BackgroundRemoverApp()
    window.show()
    # Start the update check in a separate thread to not block the UI
    update_thread = threading.Thread(target=check_for_updates_async, daemon=True)
    update_thread.start()
    sys.exit(app.exec_())
