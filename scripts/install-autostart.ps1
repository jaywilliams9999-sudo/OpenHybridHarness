# Start OpenHybridHarness (and Ollama) at logon.
#   scripts\install-autostart.ps1          register the two logon tasks
#   scripts\install-autostart.ps1 -Remove  stop and unregister them
param([switch]$Remove)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ollamaTask = "OpenHybridHarness Ollama"
$appTask = "OpenHybridHarness"

if ($Remove) {
    foreach ($name in @($appTask, $ollamaTask)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Removed task '$name'."
        } else {
            Write-Host "No task '$name'."
        }
    }
    exit 0
}

$user = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

function New-LogonTrigger([int]$delaySeconds) {
    $t = New-ScheduledTaskTrigger -AtLogOn -User $user
    if ($delaySeconds -gt 0) { $t.Delay = "PT${delaySeconds}S" }
    return $t
}

# Ollama: KV cache quantisation and flash attention are set in the task's own environment.
$ollamaAction = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c set OLLAMA_KV_CACHE_TYPE=q4_0 && set OLLAMA_FLASH_ATTENTION=1 && ollama serve"
Register-ScheduledTask -TaskName $ollamaTask -Action $ollamaAction -Trigger (New-LogonTrigger 0) `
    -Principal $principal -Settings $settings -Force | Out-Null

# The harness: 30 s later, so Ollama is up first.
$appAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c npm start" -WorkingDirectory $root
Register-ScheduledTask -TaskName $appTask -Action $appAction -Trigger (New-LogonTrigger 30) `
    -Principal $principal -Settings $settings -Force | Out-Null

# Record the choice in .env (once).
$envFile = Join-Path $root ".env"
$has = (Test-Path $envFile) -and (Select-String -Path $envFile -Pattern "^\s*OHH_KV_CACHE_TYPE=" -Quiet)
if (-not $has) {
    $prefix = ""
    if ((Test-Path $envFile) -and ((Get-Content $envFile -Raw) -match "[^\n]\z")) { $prefix = "`n" }
    [System.IO.File]::AppendAllText($envFile, "${prefix}OHH_KV_CACHE_TYPE=q4_0`n", (New-Object System.Text.UTF8Encoding($false)))
}

Write-Host "Registered '$ollamaTask' and '$appTask' to run at logon."
Write-Host ""
Write-Host "To stop them now:      Stop-ScheduledTask -TaskName '$appTask'; Stop-ScheduledTask -TaskName '$ollamaTask'"
Write-Host "To disable autostart:  Disable-ScheduledTask -TaskName '$appTask'; Disable-ScheduledTask -TaskName '$ollamaTask'"
Write-Host "To remove them:        scripts\install-autostart.ps1 -Remove"
