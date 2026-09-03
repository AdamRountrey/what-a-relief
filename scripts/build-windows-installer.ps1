param(
    [string]$Version = "0.2.1",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$appName = "What A Relief"
$exeName = "what-a-relief.exe"
$publisher = "What A Relief Contributors"
$buildOut = Join-Path $repo "build-vcpkg-direct"
$share = Join-Path $repo "build\ninja-vcpkg\vcpkg_installed\x64-windows\share"
$dist = Join-Path $repo "dist"
$work = Join-Path $dist "installer-work"
$payload = Join-Path $work "payload"
$cabPath = Join-Path $work "what-a-relief-payload.cab"
$ddfPath = Join-Path $work "payload.ddf"
$makecabLog = Join-Path $work "makecab.log"
$stubSource = Join-Path $work "InstallerStub.cs"
$stubExe = Join-Path $work "InstallerStub.exe"
$installerOut = Join-Path $dist "What-A-Relief-$Version-Setup.exe"
$marker = [Text.Encoding]::ASCII.GetBytes("WHAT_A_RELIEF_PAYLOAD_V1")

function Add-LicenseSection {
    param(
        [string]$OutputPath,
        [string]$ShareDir,
        [string]$PackageName,
        [string]$DisplayName
    )

    $copyright = Join-Path $ShareDir "$PackageName\copyright"
    if (-not (Test-Path $copyright)) {
        Write-Warning "License notice not found for $DisplayName at $copyright"
        return
    }

    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value ""
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "================================================================================"
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value $DisplayName
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "vcpkg notice file: share/$PackageName/copyright"
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "================================================================================"
    Get-Content -LiteralPath $copyright | Add-Content -LiteralPath $OutputPath -Encoding UTF8
}

function Write-ThirdPartyLicenseBundle {
    param(
        [string]$OutputPath,
        [string]$ShareDir
    )

    if (-not (Test-Path $ShareDir)) {
        throw "vcpkg notice directory not found: $ShareDir"
    }

    $packages = @(
        @{ Package = "opencv4"; Display = "OpenCV 4" },
        @{ Package = "abseil"; Display = "Abseil" },
        @{ Package = "protobuf"; Display = "Protocol Buffers" },
        @{ Package = "libjpeg-turbo"; Display = "libjpeg-turbo" },
        @{ Package = "libpng"; Display = "libpng" },
        @{ Package = "libwebp"; Display = "libwebp" },
        @{ Package = "liblzma"; Display = "XZ Utils / liblzma" },
        @{ Package = "tiff"; Display = "LibTIFF" },
        @{ Package = "zlib"; Display = "zlib" }
    )

    Set-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "Third-party license notices for the What A Relief Windows runtime package."
    Add-Content -LiteralPath $OutputPath -Encoding UTF8 -Value "Generated from vcpkg package copyright files."
    foreach ($package in $packages) {
        Add-LicenseSection -OutputPath $OutputPath -ShareDir $ShareDir -PackageName $package.Package -DisplayName $package.Display
    }
}

if (-not $SkipBuild) {
    & powershell.exe -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build-vcpkg-direct-msvc.ps1") -Version $Version
    & powershell.exe -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "package-vcpkg-runtime.ps1")
}

$exe = Join-Path $buildOut $exeName
if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe. Build first or run without -SkipBuild."
}

if (Test-Path $work) {
    Remove-Item -LiteralPath $work -Recurse -Force
}
New-Item -ItemType Directory -Force $payload | Out-Null

Copy-Item -LiteralPath $exe -Destination $payload -Force
Copy-Item -Path (Join-Path $buildOut "*.dll") -Destination $payload -Force
Copy-Item -LiteralPath (Join-Path $repo "README.md") -Destination $payload -Force
Copy-Item -LiteralPath (Join-Path $repo "LICENSE") -Destination $payload -Force
Copy-Item -LiteralPath (Join-Path $repo "SECURITY.md") -Destination $payload -Force
Copy-Item -LiteralPath (Join-Path $repo "AI_ATTRIBUTION.md") -Destination $payload -Force
Copy-Item -LiteralPath (Join-Path $repo "THIRD_PARTY_NOTICES.md") -Destination $payload -Force
Copy-Item -LiteralPath (Join-Path $repo "docs\algorithm.md") -Destination (Join-Path $payload "ALGORITHM.md") -Force
if (Test-Path (Join-Path $buildOut "models")) {
    Copy-Item -LiteralPath (Join-Path $buildOut "models") -Destination (Join-Path $payload "models") -Recurse -Force
}
Write-ThirdPartyLicenseBundle -OutputPath (Join-Path $payload "THIRD_PARTY_LICENSES.txt") -ShareDir $share

$payloadFiles = @(Get-ChildItem -LiteralPath $payload -File | Sort-Object Name)
$ddfFileLines = $payloadFiles | ForEach-Object { '"' + $_.FullName + '"' }
$ddf = @"
.Set CabinetNameTemplate=$(Split-Path -Leaf $cabPath)
.Set DiskDirectoryTemplate=$(Split-Path -Parent $cabPath)
.Set CompressionType=LZX
.Set CompressionLevel=7
.Set InfFileName=$(Join-Path $work "payload.inf")
.Set RptFileName=$(Join-Path $work "payload.rpt")
.Set Cabinet=ON
.Set Compress=ON
.Set MaxDiskSize=CDROM
.Set MaxCabinetSize=999999999
$($ddfFileLines -join "`r`n")
"@
Set-Content -LiteralPath $ddfPath -Value $ddf -Encoding ASCII

& makecab.exe /F $ddfPath | Set-Content -LiteralPath $makecabLog -Encoding ASCII
if ($LASTEXITCODE -ne 0) {
    throw "makecab failed with exit code $LASTEXITCODE. See $makecabLog."
}
if (-not (Test-Path $cabPath)) {
    throw "Payload cabinet was not created: $cabPath"
}

$stub = @"
using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

internal static class InstallerStub {
    private const string AppName = "$appName";
    private const string ExeName = "$exeName";
    private const string Version = "$Version";
    private const string Publisher = "$publisher";
    private static readonly byte[] Marker = Encoding.ASCII.GetBytes("WHAT_A_RELIEF_PAYLOAD_V1");

    [STAThread]
    private static int Main(string[] args) {
        string tempDir = null;
        bool extractOnly = args.Length == 2 && string.Equals(args[0], "--extract", StringComparison.OrdinalIgnoreCase);
        try {
            tempDir = extractOnly ? Path.GetFullPath(args[1]) : Path.Combine(Path.GetTempPath(), "WhatAReliefInstall-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);
            string cabPath = Path.Combine(tempDir, "payload.cab");
            ExtractEmbeddedPayload(cabPath);
            ExpandCabinet(cabPath, tempDir);
            try { File.Delete(cabPath); } catch { }
            if (extractOnly) {
                return 0;
            }
            InstallPayload(tempDir);
            MessageBox.Show(AppName + " " + Version + " was installed for this user.", AppName + " Installer", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 0;
        } catch (Exception ex) {
            MessageBox.Show(ex.Message, AppName + " Installer Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        } finally {
            if (tempDir != null && !extractOnly) {
                try { Directory.Delete(tempDir, true); } catch { }
            }
        }
    }

    private static void ExtractEmbeddedPayload(string cabPath) {
        string self = Assembly.GetExecutingAssembly().Location;
        using (FileStream input = File.OpenRead(self)) {
            if (input.Length < Marker.Length + 8) {
                throw new InvalidOperationException("Installer payload is missing.");
            }

            input.Seek(-8, SeekOrigin.End);
            byte[] lengthBytes = ReadExact(input, 8);
            long payloadLength = BitConverter.ToInt64(lengthBytes, 0);
            long markerOffset = input.Length - 8 - Marker.Length;
            long payloadOffset = markerOffset - payloadLength;
            if (payloadLength <= 0 || payloadOffset < 0) {
                throw new InvalidOperationException("Installer payload is invalid.");
            }

            input.Seek(markerOffset, SeekOrigin.Begin);
            byte[] actualMarker = ReadExact(input, Marker.Length);
            for (int i = 0; i < Marker.Length; ++i) {
                if (actualMarker[i] != Marker[i]) {
                    throw new InvalidOperationException("Installer payload marker is invalid.");
                }
            }

            input.Seek(payloadOffset, SeekOrigin.Begin);
            using (FileStream output = File.Create(cabPath)) {
                CopyBytes(input, output, payloadLength);
            }
        }
    }

    private static void ExpandCabinet(string cabPath, string tempDir) {
        string expand = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "expand.exe");
        ProcessStartInfo psi = new ProcessStartInfo(expand, "-F:* \"" + cabPath + "\" \"" + tempDir + "\"");
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        using (Process process = Process.Start(psi)) {
            process.WaitForExit();
            if (process.ExitCode != 0) {
                throw new InvalidOperationException("Could not extract installer payload. expand.exe exit code: " + process.ExitCode);
            }
        }
    }

    private static void InstallPayload(string tempDir) {
        string installDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", AppName);
        string startMenuDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Microsoft", "Windows", "Start Menu", "Programs", AppName);
        Directory.CreateDirectory(installDir);
        Directory.CreateDirectory(startMenuDir);

        foreach (string sourcePath in Directory.GetFileSystemEntries(tempDir)) {
            string name = Path.GetFileName(sourcePath);
            if (string.Equals(name, "payload.cab", StringComparison.OrdinalIgnoreCase)) {
                continue;
            }

            string destinationPath = Path.Combine(installDir, name);
            if (Directory.Exists(sourcePath)) {
                CopyDirectory(sourcePath, destinationPath);
            } else {
                File.Copy(sourcePath, destinationPath, true);
            }
        }

        WriteUninstaller(installDir, startMenuDir);
        CreateShortcut(Path.Combine(startMenuDir, AppName + ".lnk"), Path.Combine(installDir, ExeName), installDir);
        RegisterUninstall(installDir);
    }

    private static void WriteUninstaller(string installDir, string startMenuDir) {
        string uninstallScript = @"`$ErrorActionPreference = ""Stop""
`$installDir = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$expectedInstallDir = Join-Path `$env:LOCALAPPDATA ""Programs\What A Relief""
`$actual = [IO.Path]::GetFullPath(`$installDir).TrimEnd('\')
`$expected = [IO.Path]::GetFullPath(`$expectedInstallDir).TrimEnd('\')
if (`$actual -ne `$expected) {
    throw ""Refusing to uninstall from unexpected directory: `$installDir""
}
`$startMenuDir = """ + startMenuDir.Replace("\"", "\\") + @"""
`$uninstallKey = ""HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\What A Relief""
Remove-Item -LiteralPath `$startMenuDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath `$uninstallKey -Recurse -Force -ErrorAction SilentlyContinue
`$cmd = ""/c timeout /t 1 /nobreak > nul & rmdir /s /q """""" + `$installDir + """"""""
Start-Process -FilePath `$env:ComSpec -ArgumentList `$cmd -WindowStyle Hidden
";
        File.WriteAllText(Path.Combine(installDir, "uninstall.ps1"), uninstallScript, new UTF8Encoding(false));
    }

    private static void CreateShortcut(string shortcutPath, string targetPath, string workingDirectory) {
        Type shellType = Type.GetTypeFromProgID("WScript.Shell");
        object shell = Activator.CreateInstance(shellType);
        object shortcut = shellType.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { shortcutPath });
        Type shortcutType = shortcut.GetType();
        shortcutType.InvokeMember("TargetPath", BindingFlags.SetProperty, null, shortcut, new object[] { targetPath });
        shortcutType.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, shortcut, new object[] { workingDirectory });
        shortcutType.InvokeMember("IconLocation", BindingFlags.SetProperty, null, shortcut, new object[] { targetPath + ",0" });
        shortcutType.InvokeMember("Save", BindingFlags.InvokeMethod, null, shortcut, null);
        Marshal.FinalReleaseComObject(shortcut);
        Marshal.FinalReleaseComObject(shell);
    }

    private static void CopyDirectory(string sourceDir, string destinationDir) {
        Directory.CreateDirectory(destinationDir);
        foreach (string directory in Directory.GetDirectories(sourceDir, "*", SearchOption.AllDirectories)) {
            string relative = directory.Substring(sourceDir.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            Directory.CreateDirectory(Path.Combine(destinationDir, relative));
        }
        foreach (string file in Directory.GetFiles(sourceDir, "*", SearchOption.AllDirectories)) {
            string relative = file.Substring(sourceDir.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string outPath = Path.Combine(destinationDir, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(outPath));
            File.Copy(file, outPath, true);
        }
    }

    private static void RegisterUninstall(string installDir) {
        string uninstallScript = Path.Combine(installDir, "uninstall.ps1");
        using (RegistryKey key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\What A Relief")) {
            key.SetValue("DisplayName", AppName);
            key.SetValue("DisplayVersion", Version);
            key.SetValue("Publisher", Publisher);
            key.SetValue("InstallLocation", installDir);
            key.SetValue("DisplayIcon", Path.Combine(installDir, ExeName));
            key.SetValue("UninstallString", "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"" + uninstallScript + "\"");
            key.SetValue("NoModify", 1, RegistryValueKind.DWord);
            key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
        }
    }

    private static byte[] ReadExact(Stream stream, int count) {
        byte[] buffer = new byte[count];
        int offset = 0;
        while (offset < count) {
            int read = stream.Read(buffer, offset, count - offset);
            if (read == 0) {
                throw new EndOfStreamException();
            }
            offset += read;
        }
        return buffer;
    }

    private static void CopyBytes(Stream input, Stream output, long count) {
        byte[] buffer = new byte[1024 * 1024];
        long remaining = count;
        while (remaining > 0) {
            int toRead = remaining > buffer.Length ? buffer.Length : (int)remaining;
            int read = input.Read(buffer, 0, toRead);
            if (read == 0) {
                throw new EndOfStreamException();
            }
            output.Write(buffer, 0, read);
            remaining -= read;
        }
    }
}
"@

Set-Content -LiteralPath $stubSource -Value $stub -Encoding UTF8

$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) {
    $csc = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe"
}
if (-not (Test-Path $csc)) {
    throw "Could not find csc.exe to build the installer stub."
}

& $csc /nologo /target:winexe /reference:System.Windows.Forms.dll /out:$stubExe $stubSource
if ($LASTEXITCODE -ne 0) {
    throw "csc failed with exit code $LASTEXITCODE"
}

if (Test-Path $installerOut) {
    Remove-Item -LiteralPath $installerOut -Force
}
Copy-Item -LiteralPath $stubExe -Destination $installerOut -Force

$payloadBytes = [IO.File]::ReadAllBytes($cabPath)
$lengthBytes = [BitConverter]::GetBytes([Int64]$payloadBytes.Length)
$stream = [IO.File]::Open($installerOut, [IO.FileMode]::Append, [IO.FileAccess]::Write)
try {
    $stream.Write($payloadBytes, 0, $payloadBytes.Length)
    $stream.Write($marker, 0, $marker.Length)
    $stream.Write($lengthBytes, 0, $lengthBytes.Length)
} finally {
    $stream.Dispose()
}

if (Test-Path $work) {
    Remove-Item -LiteralPath $work -Recurse -Force
}

Write-Host "Created installer: $installerOut"
