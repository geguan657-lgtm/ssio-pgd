# One-command health check for the SSIO reproduction scaffold.
#
#   pwsh -File scripts/run_baseline.ps1
#
# It answers one question: "is my evaluation pipeline trustworthy today?"
# Anything that fails here would silently corrupt every number you report later.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$env:PYTHONPATH = Join-Path $root 'src'
$env:PYTHONUTF8 = '1'

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    OK   $msg" -ForegroundColor Green }
function Bad($msg) { Write-Host "    FAIL $msg" -ForegroundColor Red; $script:failed++ }
$script:failed = 0

Step 1 'Python + dependencies'
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { Bad 'python not found on PATH'; Pop-Location; exit 1 }
$ver = python -c "import sys;print(sys.version.split()[0])"
Ok "python $ver at $py"
python -c "import numpy, yaml" 2>$null
if ($LASTEXITCODE -ne 0) { Bad 'numpy/pyyaml missing: python -m pip install -r requirements.txt' } else { Ok 'numpy + pyyaml importable' }

Step 2 'Unit tests'
python tests\run_tests.py | Out-Host
if ($LASTEXITCODE -ne 0) { Bad 'unit tests failed' } else { Ok '19 numeric checks passed' }

Step 3 'Known-answer self test'
python -m ssio_eval.cli selftest | Out-Host
if ($LASTEXITCODE -ne 0) { Bad 'selftest failed' } else { Ok 'metrics reproduce the injected error' }

Step 4 'Synthetic round trip (CSV write -> read -> evaluate)'
python -m ssio_eval.cli synth --out runs\synth | Out-Null
python -m ssio_eval.cli eval --est runs\synth\est.csv --gt runs\synth\gt.csv `
    --align se3 --ave-frame body --out runs\synth\report.json | Out-Null
if ($LASTEXITCODE -ne 0) { Bad 'synthetic evaluation failed' }
else {
    $res = (Get-Content runs\synth\report.json -Raw | ConvertFrom-Json).results
    $ave = $res.ave.mean
    $ate = $res.ate.rmse
    Write-Host ("    ATE {0:N3} m (expected ~1.4, linear drift over 90 s)" -f $ate)
    if ([math]::Abs($ave - 0.0539) -lt 0.001) { Ok ("AVE {0:N4} m/s matches the injected bias 0.0539" -f $ave) }
    else { Bad ("AVE {0:N4} does not match 0.0539 -- evaluator bug" -f $ave) }
}

Step 5 'Dataset layout (optional)'
if (Test-Path 'data\euroc') {
    python scripts\verify_data.py --root data\euroc --format euroc | Out-Host
    if ($LASTEXITCODE -ne 0) { Bad 'some sequences are not usable yet' } else { Ok 'all EuRoC sequences look sane' }
} else {
    Write-Host '    SKIP no data/euroc yet (see docs/DATA.md)' -ForegroundColor Yellow
}

Pop-Location
if ($script:failed -gt 0) {
    Write-Host "`n$($script:failed) check(s) failed -- do not trust any metric until these pass." -ForegroundColor Red
    exit 1
}
Write-Host "`nAll checks passed. The evaluator is trustworthy; the model is another matter." -ForegroundColor Green
Write-Host 'Next: docs/DATA.md -> download EuRoC -> python -m ssio_eval.cli eval --help'
exit 0