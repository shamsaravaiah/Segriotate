# Creates Segriotate.lnk in the project folder and on the Desktop (mango icon).
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$App,
    [string]$Icon = ""
)

$ErrorActionPreference = "Stop"

function Ensure-Shortcut([string]$Path) {
    if (Test-Path -LiteralPath $Path) { return }
    $ws = New-Object -ComObject WScript.Shell
    $shortcut = $ws.CreateShortcut($Path)
    $shortcut.TargetPath = $Python
    $shortcut.Arguments = '"' + $App + '"'
    $shortcut.WorkingDirectory = $Root
    $shortcut.WindowStyle = 1
    $shortcut.Description = "Segriotate"
    if ($Icon -and (Test-Path -LiteralPath $Icon)) {
        $shortcut.IconLocation = $Icon + ",0"
    }
    $shortcut.Save()
}

Ensure-Shortcut (Join-Path $Root "Segriotate.lnk")
$desktop = [Environment]::GetFolderPath("Desktop")
if ($desktop) {
    Ensure-Shortcut (Join-Path $desktop "Segriotate.lnk")
}
