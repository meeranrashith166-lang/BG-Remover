$ErrorActionPreference = 'Stop'

$packageArgs = @{
  packageName   = 'bg-remover'
  fileType      = 'exe'
  url           = 'https://github.com/meeranrashith166-lang/BG-Remover/releases/download/v3.0.1/BG_Remover_Setup.exe'
  silentArgs    = '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-'
  validExitCodes= @(0)
  checksum      = 'dca0ca396bff38acff06f5d5d4c9329fef4c690711a67ca165c2e714df6c589a'
  checksumType  = 'sha256'
}

Install-ChocolateyPackage @packageArgs
