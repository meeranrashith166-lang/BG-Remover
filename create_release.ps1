# create_release.ps1

Write-Host "--- 1. Initializing PyUpdater ---" -ForegroundColor Cyan
# Initialize configuration if missing
if (-not (Test-Path "pyu-data")) {
    pyupdater init --app-name "BG Remover" --repo-name "BG-Remover" --urls "https://raw.githubusercontent.com/meeranrashith166-lang/BG-Remover/main/updates/" --company "Meeran Rashith"
}

# Import keys if not already imported (check for client_config.py)
if (-not (Test-Path "client_config.py")) {
    Write-Host "Importing Keys..."
    pyupdater keys -i keypack.pyu
}

Write-Host "--- 2. Building Application (Version 1.0.0) ---" -ForegroundColor Cyan
# Build the application spec
pyupdater build --app-version=1.0.0 app.py

Write-Host "--- 3. Packaging and Signing ---" -ForegroundColor Cyan
# Create the secure package and keys.gz
pyupdater pkg --process --sign

Write-Host "--- 4. Preparing Update Folder ---" -ForegroundColor Cyan
# Prepare the updates directory to be pushed to GitHub
if (-not (Test-Path "updates")) {
    New-Item -ItemType Directory -Force -Path "updates"
}

# Copy deployable files to updates folder
Copy-Item "pyu-data\deploy\*" -Destination "updates" -Recurse -Force

Write-Host "--- DONE! ---" -ForegroundColor Green
Write-Host "Now run the following commands to push these files to GitHub:"
Write-Host "git add updates/"
Write-Host "git commit -m 'Add release 1.0.0'"
Write-Host "git push"
