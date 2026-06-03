$ErrorActionPreference = 'Stop'

$packageArgs = @{
  packageName   = 'bg-remover'
  fileType      = 'exe'
  url           = 'https://github.com/meeranrashith166-lang/BG-Remover/releases/download/v3.0.1/BG_Remover_Setup.exe'
  silentArgs    = '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-'
  validExitCodes= @(0)
  checksum      = '3082a39b571d97c8405d21fc0d3146a97fbfa9a5e57259c1ae0e5302633791dc'
  checksumType  = 'sha256'
}

Install-ChocolateyPackage @packageArgs
