[CmdletBinding()]
param(
    [switch]$Quiet,
    [ValidateSet("DirectML", "QNN")]
    [string]$Runtime = "DirectML"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. "$PSScriptRoot\arm-common.ps1"

$runtimeName = Get-ArmRuntimeName -Runtime $Runtime
$python = Resolve-ArmPythonInvoker -Runtime $runtimeName
if (-not $python) {
    $report = Get-ArmPythonDiscoveryReport -Runtime $runtimeName
    if ($report -and $report.Count -gt 0) {
        $reportText = ($report | ForEach-Object {
            if ($_.Error) {
                " - $($_.Path): ERROR $($_.Error)"
            } elseif ($_.Supported) {
                " - $($_.Path): $($_.Machine) Python $($_.Version) (supported)"
            } elseif ($_.Reason) {
                " - $($_.Path): $($_.Machine) Python $($_.Version) (not supported: $($_.Reason))"
            } else {
                " - $($_.Path): $($_.Machine) Python $($_.Version) (not supported)"
            }
        }) -join "`n"
    } else {
        $reportText = " - No Python executables were discovered."
    }

    throw @"
No supported Python interpreter was found for the $runtimeName runtime.
This runtime needs $(Get-ArmRuntimeRequirementsText -Runtime $runtimeName).
What I discovered:
$reportText

If you already have a matching Python install, point COMFYUI_ARM_PYTHON at the exact python.exe path and try again.
"@
}

$pythonMachine = Get-ArmPythonMachine -Invoker $python
$expectedMachine = if ($runtimeName -eq "QNN") { @("ARM64") } else { @("AMD64", "X64", "X86_64") }
if ($pythonMachine -notin $expectedMachine) {
    throw @"
Found Python '$($python.Display)' but it reports '$pythonMachine'.
This runtime expects $(Get-ArmRuntimeRequirementsText -Runtime $runtimeName).
Install a matching Python interpreter and try again.
"@
}

$pythonVersion = Get-ArmPythonVersion -Invoker $python
$expectedVersionRegex = if ($runtimeName -eq "QNN") { '^3\.11\.' } else { '^3\.(11|12)\.' }
if ($pythonVersion -notmatch $expectedVersionRegex) {
    throw @"
Found Python '$($python.Display)' with version '$pythonVersion'.
This runtime is currently validated on $(Get-ArmRuntimeRequirementsText -Runtime $runtimeName) only.
Install a matching Python interpreter and try again.
"@
}

$requirementsRuntime = if ($runtimeName -eq "QNN") {
    Join-Path $root "requirements-windows-arm-qnn.txt"
} else {
    Join-Path $root "requirements-windows-arm.txt"
}

if (-not (Test-Path $requirementsRuntime)) {
    throw "Missing $(Split-Path -Leaf $requirementsRuntime)."
}

$requirementsCore = Join-Path $root "requirements.txt"
$stampPath = Join-Path $root ".arm-bootstrap-$($runtimeName.ToLowerInvariant()).stamp"
$bootstrapRequirementPaths = @($requirementsCore, $requirementsRuntime)
$packageNames = if ($runtimeName -eq "QNN") { @("onnxruntime-qnn", "onnx") } else { @("torch-directml") }
$currentStamp = Get-ArmBootstrapStamp -Invoker $python -Runtime $runtimeName -PackageNames $packageNames -RequirementPaths $bootstrapRequirementPaths
$existingStamp = if (Test-Path $stampPath) { (Get-Content $stampPath -Raw).Trim() } else { "" }

if ($existingStamp -eq $currentStamp) {
    if (-not $Quiet) {
        Write-Host "ARM dependencies are already ready."
    }
    return
}

if (-not $Quiet) {
    Write-Host "Installing $runtimeName dependencies with $($python.Display)..."
}

if ($runtimeName -eq "QNN") {
    Invoke-ArmPython -Invoker $python -Arguments @("-m", "pip", "install", "-r", $requirementsRuntime)
} else {
    Invoke-ArmPython -Invoker $python -Arguments @("-m", "pip", "install", "-r", $requirementsCore, "-r", $requirementsRuntime)
}

if ($LASTEXITCODE -ne 0) {
    throw "pip install failed with exit code $LASTEXITCODE."
}

Set-Content -Path $stampPath -Value $currentStamp -Encoding ASCII -NoNewline

if (-not $Quiet) {
    Write-Host "$runtimeName dependency bootstrap complete."
}
