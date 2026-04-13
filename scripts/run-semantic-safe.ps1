param(
    [Parameter(Mandatory = $true)]
    [string]$Query,
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [int]$TopK = 5,
    [int]$TimeoutSeconds = 30,
    [switch]$Bench,
    [int]$MaxDistance,
    [ValidateSet("release", "debug")]
    [string]$Profile = "release"
)

$ErrorActionPreference = "Stop"

Get-Process vrep-core -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$cargoArgs = @("run")
if ($Profile -eq "release") {
    $cargoArgs += "--release"
}
$cargoArgs += "--"
$cargoArgs += ('"{0}"' -f $Query)
$cargoArgs += ('"{0}"' -f $InputPath)
$cargoArgs += "--top-k"
$cargoArgs += "$TopK"
if ($Bench) {
    $cargoArgs += "--bench"
}
if ($PSBoundParameters.ContainsKey("MaxDistance")) {
    $cargoArgs += "--max-distance"
    $cargoArgs += "$MaxDistance"
}

$proc = Start-Process -FilePath "cargo" -ArgumentList $cargoArgs -PassThru -NoNewWindow

if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
    Get-Process vrep-core -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    [Console]::Error.WriteLine("semantic query timed out after $TimeoutSeconds seconds and was terminated.")
    exit 124
}

exit $proc.ExitCode
