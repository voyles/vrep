param(
    [int]$TimeoutSeconds = 15,
    [ValidateSet("release", "debug")]
    [string]$Profile = "release"
)

$ErrorActionPreference = "Stop"

# Clear stale process instances before starting.
Get-Process vrep-core -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$runArgs = @("run")
if ($Profile -eq "release") {
    $runArgs += "--release"
}
$runArgs += "--"
$runArgs += "--check-model"

$proc = Start-Process -FilePath "cargo" -ArgumentList $runArgs -PassThru -NoNewWindow

if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
    Get-Process vrep-core -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    Write-Error "check-model timed out after $TimeoutSeconds seconds and was terminated."
    exit 124
}

exit $proc.ExitCode
