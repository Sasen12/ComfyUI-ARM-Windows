function Get-ArmPythonCandidatePaths {
    $candidate_paths = @()

    if ($env:COMFYUI_ARM_PYTHON) {
        $candidate_paths += $env:COMFYUI_ARM_PYTHON
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            foreach ($line in (& $pyLauncher.Source -0p 2>$null)) {
                if ($line -match '^\s*-V:[^\s]+\s+\*?\s*(?<path>.+?python(?:w)?\.exe)\s*$') {
                    $candidate_paths += $Matches.path
                }
            }
        } catch {
        }
    }

    foreach ($command_name in @("python3.12", "python3.11", "python3", "python")) {
        $command = Get-Command $command_name -ErrorAction SilentlyContinue
        if ($command -and $command.Source) {
            $candidate_paths += $command.Source
        }
    }

    $common_paths = @()
    $local_app_data = $env:LOCALAPPDATA
    if ($local_app_data) {
        $common_paths += @(
            (Join-Path $local_app_data "Programs\Python\Python311\python.exe"),
            (Join-Path $local_app_data "Programs\Python\Python312\python.exe"),
            (Join-Path $local_app_data "Programs\Python\Python311-64\python.exe"),
            (Join-Path $local_app_data "Programs\Python\Python312-64\python.exe"),
            (Join-Path $local_app_data "Python\pythoncore-3.11-64\python.exe"),
            (Join-Path $local_app_data "Python\pythoncore-3.12-64\python.exe")
        )
    }

    foreach ($program_files in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($program_files) {
            $common_paths += @(
                (Join-Path $program_files "Python311\python.exe"),
                (Join-Path $program_files "Python312\python.exe")
            )
        }
    }

    $candidate_paths += $common_paths

    foreach ($root in @(
        "HKCU:\Software\Python\PythonCore",
        "HKLM:\Software\Python\PythonCore",
        "HKLM:\Software\WOW6432Node\Python\PythonCore"
    )) {
        if (-not (Test-Path $root)) {
            continue
        }

        foreach ($version_key in Get-ChildItem $root -ErrorAction SilentlyContinue) {
            $install_path_key = Join-Path $version_key.PSPath "InstallPath"
            if (-not (Test-Path $install_path_key)) {
                continue
            }

            try {
                $install_props = Get-ItemProperty -Path $install_path_key -ErrorAction Stop
                foreach ($install_candidate in @(
                    $install_props.ExecutablePath,
                    $install_props.'(default)'
                )) {
                    if ($install_candidate) {
                        $candidate_paths += $install_candidate
                    }
                }
            } catch {
                continue
            }
        }
    }

    $candidate_paths | Where-Object { $_ } | Select-Object -Unique
}

if (-not $script:ArmPythonDiscoveryReports) {
    $script:ArmPythonDiscoveryReports = @{}
}

function Get-ArmRuntimeName {
    param(
        [string]$Runtime = "DirectML"
    )

    if ([string]::IsNullOrWhiteSpace($Runtime)) {
        $Runtime = "DirectML"
    }

    $normalized = $Runtime.Trim().ToLowerInvariant()
    if ($normalized -eq "qnn") {
        return "QNN"
    }
    return "DirectML"
}

function Get-ArmRuntimeRequirementsText {
    param(
        [string]$Runtime = "DirectML"
    )

    switch (Get-ArmRuntimeName -Runtime $Runtime) {
        "QNN" { return "x64 Python 3.11 or 3.12" }
        default { return "x64 Python 3.11 or 3.12" }
    }
}

function Test-ArmPythonRuntimeSupport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Machine,

        [Parameter(Mandatory = $true)]
        [string]$Version,

        [string]$Runtime = "DirectML"
    )

    $runtimeName = Get-ArmRuntimeName -Runtime $Runtime
    if ($runtimeName -eq "QNN") {
        if ($Machine -notin @("AMD64", "X64", "X86_64")) {
            return [pscustomobject]@{
                Supported = $false
                Reason    = "This runtime needs x64 Python."
            }
        }

        if ($Version -notmatch '^3\.(11|12)\.') {
            return [pscustomobject]@{
                Supported = $false
                Reason    = "This runtime is currently validated on x64 Python 3.11 and 3.12 only."
            }
        }

        return [pscustomobject]@{
            Supported = $true
            Reason    = $null
        }
    }

    if ($Machine -notin @("AMD64", "X64", "X86_64")) {
        return [pscustomobject]@{
            Supported = $false
            Reason    = "This runtime needs x64 Python."
        }
    }

    if ($Version -notmatch '^3\.(11|12)\.') {
        return [pscustomobject]@{
            Supported = $false
            Reason    = "This runtime is currently validated on Python 3.11 and 3.12 only."
        }
    }

    return [pscustomobject]@{
        Supported = $true
        Reason    = $null
    }
}

function Resolve-ArmPythonPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $expanded = [Environment]::ExpandEnvironmentVariables($Path).Trim()
    if (-not $expanded) {
        return $null
    }

    if (Test-Path $expanded) {
        $item = Get-Item $expanded
        if ($item.PSIsContainer) {
            foreach ($exe_name in @("python.exe", "python3.exe")) {
                $exe_path = Join-Path $item.FullName $exe_name
                if (Test-Path $exe_path) {
                    return (Get-Item $exe_path).FullName
                }
            }
            return $null
        }

        return $item.FullName
    }

    return $null
}

function Get-ArmExecutableMachine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    $stream = $null
    $reader = $null

    try {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        $reader = New-Object System.IO.BinaryReader($stream)

        $stream.Position = 0x3C
        $peOffset = $reader.ReadInt32()
        if ($peOffset -lt 0) {
            return $null
        }

        $stream.Position = $peOffset + 4
        $machine = $reader.ReadUInt16()
        switch ($machine) {
            0x8664 { return "AMD64" }
            0xAA64 { return "ARM64" }
            0x014C { return "X86" }
            0x01C0 { return "ARM" }
            default { return ("0x{0:X4}" -f $machine) }
        }
    } catch {
        return $null
    } finally {
        if ($reader) {
            $reader.Dispose()
        }
        if ($stream) {
            $stream.Dispose()
        }
    }
}

function Get-ArmPythonDiscoveryReport {
    param(
        [string]$Runtime = "DirectML"
    )

    $runtimeName = Get-ArmRuntimeName -Runtime $Runtime
    if ($script:ArmPythonDiscoveryReports.ContainsKey($runtimeName)) {
        return $script:ArmPythonDiscoveryReports[$runtimeName]
    }

    return @()
}

function Resolve-ArmPythonInvoker {
    param(
        [string]$Runtime = "DirectML"
    )

    $results = @()
    $runtimeName = Get-ArmRuntimeName -Runtime $Runtime

    foreach ($candidate_path in Get-ArmPythonCandidatePaths) {
        $resolved_path = Resolve-ArmPythonPath -Path $candidate_path
        if (-not $resolved_path) {
            continue
        }

        $invoker = [pscustomobject]@{
            Command = $resolved_path
            PrefixArgs = @()
            Display = $resolved_path
        }

        try {
            $machine = Get-ArmPythonMachine -Invoker $invoker
            $version = Get-ArmPythonVersion -Invoker $invoker
            $support = Test-ArmPythonRuntimeSupport -Machine $machine -Version $version -Runtime $runtimeName
            $supported = $support.Supported

            $results += [pscustomobject]@{
                Path = $resolved_path
                Machine = $machine
                Version = $version
                Supported = $supported
                Reason = $support.Reason
            }

            if ($supported) {
                $script:ArmPythonDiscoveryReports[$runtimeName] = $results
                return $invoker
            }
        } catch {
            $results += [pscustomobject]@{
                Path = $resolved_path
                Error = $_.Exception.Message
                Supported = $false
                Reason = $null
            }
        }
    }

    $script:ArmPythonDiscoveryReports[$runtimeName] = $results
    return $null
}

function Invoke-ArmPython {
    param(
        [Parameter(Mandatory = $true)]
        $Invoker,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $allArguments = @($Invoker.PrefixArgs) + $Arguments
    & $Invoker.Command @allArguments
}

function Get-ArmPythonMachine {
    param(
        [Parameter(Mandatory = $true)]
        $Invoker
    )

    $machine = Get-ArmExecutableMachine -Path $Invoker.Command
    if ($machine) {
        return $machine
    }

    # Fall back to the runtime report if the executable header could not be read.
    (Invoke-ArmPython -Invoker $Invoker -Arguments @("-c", "import platform; print(platform.machine().upper())")).Trim()
}

function Get-ArmPythonVersion {
    param(
        [Parameter(Mandatory = $true)]
        $Invoker
    )

    (Invoke-ArmPython -Invoker $Invoker -Arguments @("-c", "import platform; print(platform.python_version())")).Trim()
}

function Test-ArmPythonModuleInstalled {
    param(
        [Parameter(Mandatory = $true)]
        $Invoker,

        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    $code = "import importlib.util; print('1' if importlib.util.find_spec('$ModuleName') else '0')"
    (Invoke-ArmPython -Invoker $Invoker -Arguments @("-c", $code)).Trim() -eq "1"
}

function Get-ArmPythonModuleVersion {
    param(
        [Parameter(Mandatory = $true)]
        $Invoker,

        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    $code = @"
import importlib.metadata as metadata
try:
    print(metadata.version('$ModuleName'))
except Exception:
    print('')
"@
    (Invoke-ArmPython -Invoker $Invoker -Arguments @("-c", $code)).Trim()
}

function Get-ArmBootstrapStamp {
    param(
        [Parameter(Mandatory = $true)]
        $Invoker,

        [Parameter(Mandatory = $true)]
        [string[]]$RequirementPaths,

        [string]$Runtime = "DirectML",

        [string[]]$PackageNames = @()
    )

    $parts = @(
        (Get-ArmRuntimeName -Runtime $Runtime),
        (Get-ArmPythonMachine -Invoker $Invoker),
        (Get-ArmPythonVersion -Invoker $Invoker)
    )

    foreach ($package_name in $PackageNames) {
        $parts += (Get-ArmPythonModuleVersion -Invoker $Invoker -ModuleName $package_name)
    }

    foreach ($path in $RequirementPaths) {
        if (Test-Path $path) {
            $parts += (Get-Item $path).LastWriteTimeUtc.ToFileTimeUtc().ToString()
        } else {
            $parts += "missing"
        }
    }

    $parts -join "|"
}
