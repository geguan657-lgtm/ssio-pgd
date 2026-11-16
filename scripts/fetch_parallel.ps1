# Parallel range downloader for large files behind a slow single connection.
#
#   pwsh/powershell -File scripts/fetch_parallel.ps1 -Url <url> -Out <path> [-Connections 16]
#
# Why: HuggingFace-style CDNs often throttle one connection to ~1 MB/s while a
# ranged request elsewhere on the same file runs at 20+ MB/s. Splitting the file
# into N ranges and fetching them concurrently turns a 3-hour download into
# minutes. Falls back to a single connection when the server ignores ranges.

param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Connections = 16
)

$ErrorActionPreference = 'Stop'
$curl = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source
if (-not $curl) { throw 'curl.exe not found' }

function Get-RemoteSize([string]$u) {
    $h = & $curl -sSIL --max-time 60 $u
    $len = ($h | Select-String -Pattern '^content-length:\s*(\d+)' | Select-Object -Last 1)
    if (-not $len) { return -1 }
    return [int64]$len.Matches[0].Groups[1].Value
}

function Wait-ForSize([string]$path) {
    if (-not (Test-Path $path)) { return 0 }
    return (Get-Item $path).Length
}

$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

$total = Get-RemoteSize $Url
if ($total -le 0) {
    Write-Host 'server did not report a size; falling back to a single connection'
    & $curl -L --fail --retry 5 -C - -o $Out $Url
    return
}
Write-Host ("remote size: {0:N0} bytes ({1:N2} GB), {2} connections" -f $total, ($total / 1GB), $Connections)

$chunk = [int64][math]::Ceiling($total / $Connections)
$parts = @()
$procs = @()
for ($i = 0; $i -lt $Connections; $i++) {
    $start = [int64]($i * $chunk)
    $end = [int64][math]::Min($start + $chunk - 1, $total - 1)
    if ($start -gt $end) { continue }
    $part = "$Out.part$i"
    $parts += $part
    if ((Test-Path $part) -and (Get-Item $part).Length -eq ($end - $start + 1)) {
        continue  # already complete
    }
    $procs += Start-Process -FilePath $curl -PassThru -NoNewWindow -ArgumentList @(
        '-sS', '-L', '--fail', '--retry', '10', '--retry-delay', '3',
        '-r', "$start-$end", '-o', $part, $Url
    )
}

$lastReport = Get-Date
while ($procs | Where-Object { -not $_.HasExited }) {
    Start-Sleep -Seconds 5
    $now = Get-Date
    if (($now - $lastReport).TotalSeconds -ge 15) {
        $done = 0; foreach ($p in $parts) { $done += (Wait-ForSize $p) }
        Write-Host ("  {0:N2} / {1:N2} GB  ({2:P1})" -f ($done / 1GB), ($total / 1GB), ($done / $total))
        $lastReport = $now
    }
}

$done = 0; foreach ($p in $parts) { $done += (Wait-ForSize $p) }
if ($done -ne $total) {
    Write-Host ("size mismatch: got {0:N0}, expected {1:N0}; rerun to resume" -f $done, $total) -ForegroundColor Red
    exit 2
}

Write-Host 'concatenating...'
if (Test-Path $Out) { Remove-Item $Out -Force }
$fs = [System.IO.File]::Create($Out)
try {
    foreach ($p in $parts) {
        $in = [System.IO.File]::OpenRead($p)
        try { $in.CopyTo($fs) } finally { $in.Dispose() }
    }
} finally { $fs.Dispose() }
foreach ($p in $parts) { Remove-Item $p -Force -ErrorAction SilentlyContinue }

$final = (Get-Item $Out).Length
Write-Host ("done: {0} ({1:N2} GB)" -f $Out, ($final / 1GB)) -ForegroundColor Green
if ($final -ne $total) { exit 2 }