# 🎨 BG Remover (v3.0.1)

[![GitHub stars](https://img.shields.io/github/stars/meeranrashith166-lang/BG-Remover.svg?style=flat-square&logo=github)](https://github.com/meeranrashith166-lang/BG-Remover/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/meeranrashith166-lang/BG-Remover.svg?style=flat-square&logo=github)](https://github.com/meeranrashith166-lang/BG-Remover/network/members)
[![GitHub license](https://img.shields.io/github/license/meeranrashith166-lang/BG-Remover.svg?style=flat-square)](LICENSE.txt)
[![WinGet Version](https://img.shields.io/winget/v/MeeranRashith.BGRemover?style=flat-square)](https://winstall.app/apps/MeeranRashith.BGRemover)

BG Remover is a lightweight, high-performance desktop application for Windows that utilizes state-of-the-art local AI models to automatically remove backgrounds from images offline. 

Running 100% locally on your machine, it ensures complete data privacy since no photos are uploaded to any external servers.

> [!TIP]
> **Support the Project!** ⭐ Star this repository and 🍴 fork it. Reaching **100 stars** and **50 forks** enables the app to get approved and integrated directly into official package manager indices like the main Scoop Extras bucket!

---

## 🚀 Key Features

- **8 Selectable AI Models**: Select from specialized models (U²Net, BASNet, MODNet, Bria, etc.) optimized for portraits, products, animals, and anime art.
- **Advanced Edge Smoothing**: Built-in edge feathering, threshold controls, and guided filters to cleanly handle fine details like hair or fur.
- **Multiple Export Formats**: Export your results directly to transparent PNG, layered Photoshop PSD (with vector clipping paths), JPEG (with custom color background fills), and WebP.
- **Data Privacy & Security**: Works completely offline. No internet connection or account registration required after initial model setup.
- **No Limits**: Free, open-source, ad-free, and leaves no watermarks or output resolution restrictions.

---

## 📥 Installation

### 💻 1. Direct Installer (Recommended)
You can download the safe, virus-scanned standalone setup installer from our official hosting and distribution partners:
* **Featured On: [MajorGeeks](https://www.majorgeeks.com/files/details/bg_remover.html)**
* **Direct Hosting: [SourceForge](https://sourceforge.net/projects/bg-remover/)**
* **Releases: [GitHub Releases](https://github.com/meeranrashith166-lang/BG-Remover/releases)**

### 📦 2. Package Managers
Choose your preferred Windows package manager to install BG Remover:

#### **Scoop**
To install via our custom Scoop bucket, run:
```powershell
scoop install https://raw.githubusercontent.com/meeranrashith166-lang/scoop-bucket/main/bucket/bg-remover.json
```

#### **WinGet (Official Registry — LIVE ✅)**
```powershell
winget install MeeranRashith.BGRemover
```

#### **Chocolatey (Under Moderation ⏳)**
```powershell
choco install bg-remover --version=3.0.1 --pre
```

---

## 🛠️ Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Memory**: 4GB RAM minimum (8GB recommended for larger AI models)
- **Graphics**: Runs on CPU by default. Automatically accelerates with NVIDIA CUDA-capable GPUs if PyTorch is installed on the host system.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.
