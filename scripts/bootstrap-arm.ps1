[CmdletBinding()]
param(
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. "$PSScriptRoot\arm-common.ps1"

$python = Resolve-ArmPythonInvoker
if (-not $python) {
    $report = Get-ArmPythonDiscoveryReport
    if ($report -and $report.Count -gt 0) {
        $reportText = ($report | ForEach-Object {
            if ($_.Error) {
                " - $($_.Path): ERROR $($_.Error)"
            } elseif ($_.Supported) {
                " - $($_.Path): $($_.Machine) Python $($_.Version) (supported)"
            } else {
                " - $($_.Path): $($_.Machine) Python $($_.Version) (not supported)"
            }
        }) -join "`n"
    } else {
        $reportText = " - No Python executables were discovered."
    }

    throw @"
No supported Python interpreter was found.
This ARM fork needs x64 Python 3.11 or 3.12 for DirectML.
What I discovered:
$reportText

If you already have x64 Python 3.11 or 3.12 installed, point COMFYUI_ARM_PYTHON at the exact python.exe path and try again.
"@
}

$pythonMachine = Get-ArmPythonMachine -Invoker $python
if ($pythonMachine -notin @("AMD64", "X64", "X86_64")) {
    throw @"
Found Python '$($python.Display)' but it reports '$pythonMachine'.
This ARM build expects x64 Python because torch-directml currently ships as an x64-friendly setup.
Install x64 Python 3.11 or 3.12 and try again.
"@
}

$pythonVersion = Get-ArmPythonVersion -Invoker $python
if ($pythonVersion -notmatch '^3\.(11|12)\.') {
    throw @"
Found Python '$($python.Display)' with version '$pythonVersion'.
This ARM build is currently validated on Python 3.11 and 3.12 only because torch-directml wheels are published for those versions.
Install x64 Python 3.11 or 3.12 and try again.
"@
}

$requirementsArm = Join-Path $root "requirements-windows-arm.txt"

if (-not (Test-Path $requirementsArm)) {
    throw "Missing requirements-windows-arm.txt."
}

$requirementsCore = Join-Path $root "requirements.txt"
$stampPath = Join-Path $root ".arm-bootstrap.stamp"
$currentStamp = Get-ArmBootstrapStamp -Invoker $python -RequirementPaths @($requirementsCore, $requirementsArm)
$existingStamp = if (Test-Path $stampPath) { (Get-Content $stampPath -Raw).Trim() } else { "" }

if ($existingStamp -eq $currentStamp) {
    if (-not $Quiet) {
        Write-Host "ARM dependencies are already ready."
    }
    return
}

if (-not $Quiet) {
    Write-Host "Installing ARM dependencies with $($python.Display)..."
}

Invoke-ArmPython -Invoker $python -Arguments @("-m", "pip", "install", "-r", $requirementsArm)
Set-Content -Path $stampPath -Value $currentStamp -Encoding ASCII -NoNewline

if (-not $Quiet) {
    Write-Host "ARM dependency bootstrap complete."
}
