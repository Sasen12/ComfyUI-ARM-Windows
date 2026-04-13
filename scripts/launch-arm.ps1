[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$SafeMode,
    [string[]]$WhitelistCustomNodes = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

& "$PSScriptRoot\bootstrap-arm.ps1" -Quiet
. "$PSScriptRoot\arm-common.ps1"

$python = Resolve-ArmPythonInvoker
if (-not $python) {
    throw "No Python interpreter was found after bootstrap."
}

$mainArgs = @("main.py", "--auto-launch")
if ($CpuOnly) {
    $mainArgs += "--cpu"
}

if ($SafeMode -or $WhitelistCustomNodes.Count -gt 0) {
    if ($WhitelistCustomNodes.Count -gt 0) {
        Write-Host "Launching ARM safe mode with whitelisted custom nodes: $($WhitelistCustomNodes -join ', ')"
    } else {
        Write-Host "Launching ARM safe mode (custom nodes disabled)."
    }
    $mainArgs += "--disable-all-custom-nodes"
    $mainArgs += "--disable-api-nodes"
    if ($WhitelistCustomNodes.Count -gt 0) {
        $mainArgs += "--whitelist-custom-nodes"
        $mainArgs += $WhitelistCustomNodes
    }
}

Invoke-ArmPython -Invoker $python -Arguments $mainArgs
