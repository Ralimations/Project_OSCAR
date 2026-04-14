param(
    [ValidateSet("gui", "harness", "harness-live", "main", "main-live", "main-debug", "download-models", "tests")]
    [string]$Mode = "gui"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot "tools\python313\python.exe"

if (-not (Test-Path $python)) {
    throw "Embedded Python runtime not found at $python"
}

Set-Location $repoRoot

function Invoke-OscarMode {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedMode
    )

    switch ($SelectedMode) {
        "harness" {
            $env:OSCAR_SANDBOX_ACTIONS_ONLY = "1"
            & $python "scripts\transcript_harness.py"
        }
        "harness-live" {
            $env:OSCAR_SANDBOX_ACTIONS_ONLY = "0"
            & $python "scripts\transcript_harness.py"
        }
        "main" {
            $env:OSCAR_SANDBOX_ACTIONS_ONLY = "1"
            & $python "main.py"
        }
        "main-live" {
            $env:OSCAR_SANDBOX_ACTIONS_ONLY = "0"
            & $python "main.py"
        }
        "main-debug" {
            $env:OSCAR_DEVELOPMENT_MODE = "1"
            $env:OSCAR_SHOW_TRACKING_WINDOW = "1"
            $env:OSCAR_SHOW_SANDBOX_WINDOW = "1"
            $env:OSCAR_SANDBOX_ACTIONS_ONLY = "1"
            & $python "main.py"
        }
        "download-models" {
            & $python "scripts\download_models.py"
        }
        "tests" {
            & $python -m unittest discover -s tests -p "test_*.py"
        }
        default {
            throw "Unsupported mode: $SelectedMode"
        }
    }
}

function Start-OscarProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedMode
    )

    $command = "& '$PSCommandPath' -Mode '$SelectedMode'"
    $isRuntimeMode = $SelectedMode -in @("main", "main-live", "main-debug")

    $readyFile = $null
    if ($isRuntimeMode) {
        $readyFile = Join-Path $env:TEMP ("oscar_ready_" + [guid]::NewGuid().ToString("N") + ".tmp")
        $command = "`$env:OSCAR_READY_FILE='$readyFile'; $command"
    }

    $arguments = @(
        "-ExecutionPolicy", "Bypass",
        "-Command", $command
    )

    if (-not $isRuntimeMode) {
        $arguments = @(
            "-ExecutionPolicy", "Bypass",
            "-NoExit",
            "-Command", $command
        )
    }

    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -PassThru

    if ($isRuntimeMode) {
        Show-OscarLoadingScreen -Process $process -ReadyFile $readyFile -ModeLabel $SelectedMode
    }
}

function Show-OscarLoadingScreen {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$ReadyFile,
        [Parameter(Mandatory = $true)]
        [string]$ModeLabel
    )

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Starting OSCAR"
    $form.Size = New-Object System.Drawing.Size(360, 190)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.ControlBox = $false
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false
    $form.TopMost = $true
    $form.BackColor = [System.Drawing.Color]::FromArgb(24, 28, 36)
    $form.ForeColor = [System.Drawing.Color]::White

    $title = New-Object System.Windows.Forms.Label
    $title.Text = "Project O.S.C.A.R."
    $title.Font = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(20, 18)
    $title.Size = New-Object System.Drawing.Size(250, 34)
    $form.Controls.Add($title)

    $status = New-Object System.Windows.Forms.Label
    $status.Text = "Preparing $ModeLabel..."
    $status.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $status.Location = New-Object System.Drawing.Point(22, 62)
    $status.Size = New-Object System.Drawing.Size(300, 24)
    $form.Controls.Add($status)

    $detail = New-Object System.Windows.Forms.Label
    $detail.Text = "Loading camera, audio, and routing layers."
    $detail.Font = New-Object System.Drawing.Font("Segoe UI", 8)
    $detail.Location = New-Object System.Drawing.Point(22, 90)
    $detail.Size = New-Object System.Drawing.Size(310, 20)
    $detail.ForeColor = [System.Drawing.Color]::FromArgb(180, 190, 210)
    $form.Controls.Add($detail)

    $bar = New-Object System.Windows.Forms.ProgressBar
    $bar.Location = New-Object System.Drawing.Point(22, 120)
    $bar.Size = New-Object System.Drawing.Size(300, 18)
    $bar.Style = [System.Windows.Forms.ProgressBarStyle]::Marquee
    $bar.MarqueeAnimationSpeed = 30
    $form.Controls.Add($bar)

    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 150
    $timer.Add_Tick({
        if ($Process.HasExited) {
            $status.Text = "OSCAR exited during startup."
            $timer.Stop()
            if (Test-Path $ReadyFile) {
                Remove-Item -LiteralPath $ReadyFile -Force -ErrorAction SilentlyContinue
            }
            $form.Close()
            return
        }
        if (Test-Path $ReadyFile) {
            Remove-Item -LiteralPath $ReadyFile -Force -ErrorAction SilentlyContinue
            $timer.Stop()
            $form.Close()
        }
    })

    $form.Add_Shown({
        $timer.Start()
    })

    [void]$form.ShowDialog()
}

function Show-OscarLauncher {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "OSCAR Launcher"
    $form.Size = New-Object System.Drawing.Size(430, 420)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false
    $form.BackColor = [System.Drawing.Color]::FromArgb(24, 28, 36)
    $form.ForeColor = [System.Drawing.Color]::White

    $title = New-Object System.Windows.Forms.Label
    $title.Text = "Project O.S.C.A.R."
    $title.Font = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
    $title.Location = New-Object System.Drawing.Point(20, 18)
    $title.Size = New-Object System.Drawing.Size(280, 34)
    $form.Controls.Add($title)

    $subtitle = New-Object System.Windows.Forms.Label
    $subtitle.Text = "Choose a task for smoke testing or runtime work."
    $subtitle.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $subtitle.Location = New-Object System.Drawing.Point(22, 55)
    $subtitle.Size = New-Object System.Drawing.Size(360, 20)
    $form.Controls.Add($subtitle)

    $status = New-Object System.Windows.Forms.Label
    $status.Text = "Ready."
    $status.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $status.Location = New-Object System.Drawing.Point(22, 335)
    $status.Size = New-Object System.Drawing.Size(370, 20)
    $status.ForeColor = [System.Drawing.Color]::FromArgb(150, 210, 255)
    $form.Controls.Add($status)

    $buttonStyle = @{
        Size = New-Object System.Drawing.Size(170, 44)
        BackColor = [System.Drawing.Color]::FromArgb(38, 46, 60)
        ForeColor = [System.Drawing.Color]::White
        FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    }

    function New-LauncherButton {
        param(
            [string]$Text,
            [int]$X,
            [int]$Y,
            [string]$LaunchMode,
            [string]$Description
        )

        $button = New-Object System.Windows.Forms.Button
        $button.Text = $Text
        $button.Location = New-Object System.Drawing.Point($X, $Y)
        $button.Size = $buttonStyle.Size
        $button.BackColor = $buttonStyle.BackColor
        $button.ForeColor = $buttonStyle.ForeColor
        $button.FlatStyle = $buttonStyle.FlatStyle
        $button.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(90, 110, 140)
        $button.Tag = [pscustomobject]@{
            Mode = $LaunchMode
            Description = $Description
        }
        $button.Add_Click({
            param($sender, $eventArgs)
            $payload = $sender.Tag
            $status.Text = "Launching $($payload.Description)..."
            Start-OscarProcess -SelectedMode $payload.Mode
        })
        $form.Controls.Add($button)
    }

    New-LauncherButton -Text "Harness Safe" -X 22 -Y 95 -LaunchMode "harness" -Description "transcript harness in safe mode"
    New-LauncherButton -Text "Harness Live" -X 212 -Y 95 -LaunchMode "harness-live" -Description "transcript harness with live actions"
    New-LauncherButton -Text "Main Safe" -X 22 -Y 150 -LaunchMode "main" -Description "main runtime in safe mode"
    New-LauncherButton -Text "Main Live" -X 212 -Y 150 -LaunchMode "main-live" -Description "main runtime with live actions"
    New-LauncherButton -Text "Main Debug View" -X 22 -Y 205 -LaunchMode "main-debug" -Description "main debug view"
    New-LauncherButton -Text "Run Tests" -X 212 -Y 205 -LaunchMode "tests" -Description "test suite"
    New-LauncherButton -Text "Download Models" -X 22 -Y 260 -LaunchMode "download-models" -Description "model download"

    $hint = New-Object System.Windows.Forms.Label
    $hint.Text = "Safe modes simulate actions. Live modes control the real desktop."
    $hint.Font = New-Object System.Drawing.Font("Segoe UI", 8)
    $hint.Location = New-Object System.Drawing.Point(22, 358)
    $hint.Size = New-Object System.Drawing.Size(360, 20)
    $hint.ForeColor = [System.Drawing.Color]::FromArgb(180, 190, 210)
    $form.Controls.Add($hint)

    [void]$form.ShowDialog()
}

switch ($Mode) {
    "gui" {
        Show-OscarLauncher
    }
    default {
        Invoke-OscarMode -SelectedMode $Mode
    }
}
