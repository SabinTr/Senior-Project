# Run this on the PC (PowerShell), NOT on the robot:
#   .\push_to_robot.ps1                      -> copies roam_and_orbit.py
#   .\push_to_robot.ps1 "other file.py"      -> copies the files you name
# Copies to ~/sabin on the robot (uses the "robot" entry in ~/.ssh/config).
# Careful with "Telegram voice robot.py": the robot's copy has your real token,
# copying the PC version over it puts the placeholders back.

param([string[]]$Files = @("roam_and_orbit.py"))

if ($env:OS -notlike "*Windows*") {
    Write-Host "This script runs on your PC, not on the robot." -ForegroundColor Red
    exit 1
}

foreach ($f in $Files) {
    $path = Join-Path $PSScriptRoot $f
    Write-Host "Copying $f ..."
    scp $path "robot:~/sabin/"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED - $f was not copied." -ForegroundColor Red
        exit 1
    }
}
Write-Host "Done." -ForegroundColor Green
