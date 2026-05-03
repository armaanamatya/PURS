Add-Type -AssemblyName System.Drawing

function New-Color {
    param(
        [int]$A,
        [int]$R,
        [int]$G,
        [int]$B
    )

    return [System.Drawing.Color]::FromArgb($A, $R, $G, $B)
}

function Draw-Paragraph {
    param(
        [System.Drawing.Graphics]$Graphics,
        [string]$Text,
        [float]$X,
        [float]$Y,
        [float]$Width,
        [float]$Height,
        [System.Drawing.Font]$Font,
        [System.Drawing.Brush]$Brush,
        [System.Drawing.StringAlignment]$Alignment = [System.Drawing.StringAlignment]::Near
    )

    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = $Alignment
    $format.LineAlignment = [System.Drawing.StringAlignment]::Near
    $Graphics.DrawString($Text, $Font, $Brush, [System.Drawing.RectangleF]::new($X, $Y, $Width, $Height), $format)
    $format.Dispose()
}

$root = Split-Path -Parent $PSScriptRoot
$outPath = Join-Path $root "figures\\omnizip_omnisift_failure_venn.png"

$bitmap = New-Object System.Drawing.Bitmap 1500, 980
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.Clear([System.Drawing.Color]::FromArgb(250, 248, 242))

$zipFillBrush = New-Object System.Drawing.SolidBrush (New-Color 92 110 168 217)
$zipBorderPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(63, 121, 168)), 5
$siftFillBrush = New-Object System.Drawing.SolidBrush (New-Color 92 239 177 103)
$siftBorderPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(207, 133, 50)), 5
$titleBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(18, 18, 18))
$bodyBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(38, 38, 38))
$zipLabelBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(47, 98, 140))
$siftLabelBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(168, 100, 24))
$sharedBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(60, 60, 60))

$titleFont = New-Object System.Drawing.Font("Segoe UI Semibold", 34)
$subtitleFont = New-Object System.Drawing.Font("Segoe UI", 18, [System.Drawing.FontStyle]::Italic)
$sharedFont = New-Object System.Drawing.Font("Segoe UI Semibold", 21)
$bodyFont = New-Object System.Drawing.Font("Segoe UI Semibold", 15)

$graphics.FillEllipse($zipFillBrush, 125, 205, 640, 640)
$graphics.DrawEllipse($zipBorderPen, 125, 205, 640, 640)
$graphics.FillEllipse($siftFillBrush, 535, 205, 640, 640)
$graphics.DrawEllipse($siftBorderPen, 535, 205, 640, 640)

$graphics.DrawString("OmniZip", $titleFont, $zipLabelBrush, 250, 92)
$graphics.DrawString("audio-guided", $subtitleFont, $zipLabelBrush, 260, 138)
$graphics.DrawString("OmniSIFT", $titleFont, $siftLabelBrush, 945, 92)
$graphics.DrawString("vision-guided", $subtitleFont, $siftLabelBrush, 976, 138)
$graphics.DrawString("Shared Limits", $sharedFont, $sharedBrush, 590, 285)

Draw-Paragraph `
    -Graphics $graphics `
    -Text "Silent visual cues:`nText, scoreboards,`nvisual gags.`n`nGeneric audio:`nB-roll, diagrams,`nmusic beds.`n`nNoisy / misaligned audio:`nCrowd noise,`nlagging narration." `
    -X 165 `
    -Y 355 `
    -Width 230 `
    -Height 320 `
    -Font $bodyFont `
    -Brush $bodyBrush

Draw-Paragraph `
    -Graphics $graphics `
    -Text "Dense clips:`nAlmost everything matters.`n`nMicro-events:`nTiny decisive moments.`n`nA/V mismatch:`nLip-sync, dubbing,`ndeepfakes." `
    -X 520 `
    -Y 355 `
    -Width 270 `
    -Height 300 `
    -Font $bodyFont `
    -Brush $bodyBrush `
    -Alignment ([System.Drawing.StringAlignment]::Center)

Draw-Paragraph `
    -Graphics $graphics `
    -Text "Audio-dominant clips:`nPodcasts,`nstatic heads.`n`nOff-screen sounds:`nHeard, not seen.`n`nVisual distraction:`nBusy motion masks`nspeech." `
    -X 960 `
    -Y 355 `
    -Width 220 `
    -Height 320 `
    -Font $bodyFont `
    -Brush $bodyBrush

[System.IO.Directory]::CreateDirectory((Split-Path -Parent $outPath)) | Out-Null
$bitmap.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)

$graphics.Dispose()
$bitmap.Dispose()
$zipFillBrush.Dispose()
$zipBorderPen.Dispose()
$siftFillBrush.Dispose()
$siftBorderPen.Dispose()
$titleBrush.Dispose()
$bodyBrush.Dispose()
$zipLabelBrush.Dispose()
$siftLabelBrush.Dispose()
$sharedBrush.Dispose()
$titleFont.Dispose()
$subtitleFont.Dispose()
$sharedFont.Dispose()
$bodyFont.Dispose()

Write-Output "Saved $outPath"
