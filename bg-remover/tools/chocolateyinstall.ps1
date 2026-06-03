$ErrorActionPreference = 'Stop'

$packageArgs = @{
  packageName   = 'bg-remover'
  fileType      = 'exe'
  url           = 'https://github.com/meeranrashith166-lang/BG-Remover/releases/download/v3.0.1/BG_Remover_Setup.exe'
  silentArgs    = '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-'
  validExitCodes= @(0)
  checksum      = '55e676732699a0c7cf4630991c04fef4bb0348d1daf34c3abc661f47a2f11045'
  checksumType  = 'sha256'
}

Install-ChocolateyPackage @packageArgs
