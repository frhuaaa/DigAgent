param(
    [string]$ServerAddress = "192.168.90.201",
    [string]$ServerUser = "huafengrui",
    [int]$ServerPort = 22,
    [string]$RemoteProjectRoot = "/finance_ML/huafengrui/DigAgent"
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Program,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

foreach ($value in @($ServerAddress, $ServerUser, $RemoteProjectRoot)) {
    if ($value -notmatch '^[A-Za-z0-9_./-]+$') {
        throw "Unsafe SSH destination value: $value"
    }
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LocalEnv = Join-Path $ProjectRoot ".env"
$LocalDataRoot = Join-Path $ProjectRoot "data"
$RequiredLocalPaths = @(
    $LocalEnv,
    (Join-Path $LocalDataRoot "cn_data"),
    (Join-Path $LocalDataRoot "portfolio\c_2_c_1D.csv"),
    (Join-Path $LocalDataRoot "portfolio\mask_limit_up_1D.csv"),
    (Join-Path $LocalDataRoot "portfolio\mask_limit_down_1D.csv")
)

foreach ($requiredPath in $RequiredLocalPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required local input is missing: $requiredPath"
    }
}

foreach ($program in @("tar.exe", "scp.exe", "ssh.exe")) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
        throw "Required program is not available: $program"
    }
}

$Remote = "${ServerUser}@${ServerAddress}"
$TransferId = [Guid]::NewGuid().ToString("N")
$ArchiveName = "diagagent-inputs-${TransferId}.tar.gz"
$LocalArchive = Join-Path ([IO.Path]::GetTempPath()) $ArchiveName
$RemoteArchive = "/finance_ML/huafengrui/${ArchiveName}"
$RemoteStage = "/finance_ML/huafengrui/.diagagent-inputs-${TransferId}"
$RemoteEnvTemp = "${RemoteProjectRoot}/.env.incoming-${TransferId}"

$SshBase = @(
    "-o", "BatchMode=yes",
    "-p", $ServerPort.ToString(),
    $Remote
)
$ScpBase = @(
    "-o", "BatchMode=yes",
    "-P", $ServerPort.ToString()
)

$Preflight = @"
set -eu
test -d '$RemoteProjectRoot'
test ! -e '$RemoteProjectRoot/.env'
test ! -e '$RemoteProjectRoot/data/cn_data'
test ! -e '$RemoteProjectRoot/data/portfolio'
mkdir -p '$RemoteStage'
"@

Write-Host "Checking the remote destination (existing inputs will not be overwritten)..."
Invoke-NativeChecked ssh.exe @SshBase $Preflight

try {
    Write-Host "Creating a compressed archive for cn_data and portfolio..."
    Invoke-NativeChecked tar.exe "-czf" $LocalArchive "-C" $LocalDataRoot "cn_data" "portfolio"

    $LocalHash = (Get-FileHash -LiteralPath $LocalArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    $ArchiveSizeGB = [Math]::Round((Get-Item -LiteralPath $LocalArchive).Length / 1GB, 3)
    Write-Host "Archive size: ${ArchiveSizeGB} GB"
    Write-Host "Uploading data archive with scp..."
    Invoke-NativeChecked scp.exe @ScpBase $LocalArchive "${Remote}:${RemoteArchive}"

    $RemoteHashOutput = & ssh.exe @SshBase "sha256sum '$RemoteArchive'"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote SHA256 calculation failed with exit code $LASTEXITCODE"
    }
    $RemoteHash = (($RemoteHashOutput | Out-String).Trim() -split '\s+')[0].ToLowerInvariant()
    if ($RemoteHash -ne $LocalHash) {
        throw "Archive SHA256 mismatch: local=$LocalHash remote=$RemoteHash"
    }
    Write-Host "Archive SHA256 verified."

    $InstallData = @"
set -eu
tar -xzf '$RemoteArchive' -C '$RemoteStage'
test -d '$RemoteStage/cn_data'
test -f '$RemoteStage/portfolio/c_2_c_1D.csv'
test -f '$RemoteStage/portfolio/mask_limit_up_1D.csv'
test -f '$RemoteStage/portfolio/mask_limit_down_1D.csv'
mkdir -p '$RemoteProjectRoot/data'
mv '$RemoteStage/cn_data' '$RemoteProjectRoot/data/cn_data'
mv '$RemoteStage/portfolio' '$RemoteProjectRoot/data/portfolio'
rmdir '$RemoteStage'
rm -f '$RemoteArchive'
"@

    Write-Host "Extracting and installing verified data on the server..."
    Invoke-NativeChecked ssh.exe @SshBase $InstallData

    Write-Host "Uploading .env separately with restrictive permissions..."
    Invoke-NativeChecked scp.exe @ScpBase $LocalEnv "${Remote}:${RemoteEnvTemp}"
    Invoke-NativeChecked ssh.exe @SshBase "chmod 600 '$RemoteEnvTemp' && mv '$RemoteEnvTemp' '$RemoteProjectRoot/.env'"

    $Verify = @"
set -eu
test -f '$RemoteProjectRoot/.env'
test -d '$RemoteProjectRoot/data/cn_data'
test -f '$RemoteProjectRoot/data/portfolio/c_2_c_1D.csv'
test -f '$RemoteProjectRoot/data/portfolio/mask_limit_up_1D.csv'
test -f '$RemoteProjectRoot/data/portfolio/mask_limit_down_1D.csv'
stat -c '.env mode=%a owner=%U' '$RemoteProjectRoot/.env'
du -sh '$RemoteProjectRoot/data/cn_data' '$RemoteProjectRoot/data/portfolio'
"@

    Write-Host "Verifying installed inputs..."
    Invoke-NativeChecked ssh.exe @SshBase $Verify
    Write-Host "Server inputs uploaded successfully." -ForegroundColor Green
}
finally {
    if (Test-Path -LiteralPath $LocalArchive) {
        Remove-Item -LiteralPath $LocalArchive -Force
    }
}
