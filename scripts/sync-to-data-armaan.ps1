# PowerShell cannot run .bash files. This wrapper calls WSL or Git Bash.
#
#   cd C:\Users\Armaan\Desktop\PURS
#   .\scripts\sync-to-data-armaan.ps1
#   .\scripts\sync-to-data-armaan.ps1 armaan@10.244.120.178 /data/armaan/purs
#
# For DRY_RUN / SYNC_SLIM, use Git Bash or WSL directly, e.g.:
#   wsl bash -lc 'cd /mnt/c/Users/Armaan/Desktop/PURS && DRY_RUN=1 ./scripts/sync-to-data-armaan.bash'

param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"
$BashScript = Join-Path $PSScriptRoot "sync-to-data-armaan.bash"
if (-not (Test-Path -LiteralPath $BashScript)) {
  throw "Missing: $BashScript"
}

function Convert-ToWslPath {
  param([string]$WindowsPath)
  $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
  if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Cannot convert to WSL path: $resolved"
  }
  $drive = $Matches[1].ToLower()
  $rest = $Matches[2] -replace '\\', '/'
  return "/mnt/$drive/$rest"
}

function Convert-ToGitBashPath {
  param([string]$WindowsPath)
  $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
  if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Cannot convert to Git Bash path: $resolved"
  }
  $drive = $Matches[1].ToLower()
  $rest = $Matches[2] -replace '\\', '/'
  return "/$drive/$rest"
}

if ($null -ne (Get-Command wsl -ErrorAction SilentlyContinue)) {
  $unixScript = Convert-ToWslPath $BashScript
  Write-Host "Using WSL: bash $unixScript $($ForwardArgs -join ' ')"
  $wslArgs = @('bash', $unixScript) + $ForwardArgs
  wsl @wslArgs
  exit $LASTEXITCODE
}

$gitBashCandidates = @(
  (Join-Path ${env:ProgramFiles} "Git\bin\bash.exe"),
  (Join-Path ${env:ProgramFiles(x86)} "Git\bin\bash.exe"),
  (Join-Path $env:LocalAppData "Programs\Git\bin\bash.exe")
)
$bashExe = $gitBashCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq $bashExe) {
  throw "WSL not found and Git Bash not found. Install WSL or Git for Windows, then retry."
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$unixRepo = Convert-ToGitBashPath $repoRoot
$unixScript = Convert-ToGitBashPath $BashScript
Write-Host "Using Git Bash: $bashExe"
$argline = ($ForwardArgs | ForEach-Object { "'" + ($_.ToString() -replace "'", "'\''") + "'" }) -join ' '
& $bashExe -lc "cd `"$unixRepo`" && bash `"$unixScript`" $argline"
