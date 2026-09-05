param(
    [string]$Version = "0.2.1",
    [string]$CacheDirectory = ""
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $repo "dist"
if ([string]::IsNullOrWhiteSpace($CacheDirectory)) {
    $CacheDirectory = Join-Path $dist "mitsuba-backend-cache"
}
$cache = [IO.Path]::GetFullPath($CacheDirectory)
$work = Join-Path $dist "mitsuba-backend-installer-work"
$runtime = Join-Path $work "runtime"
$payloadZip = Join-Path $work "mitsuba-runtime.zip"
$stubSource = Join-Path $work "MitsubaInstallerStub.cs"
$stubExe = Join-Path $work "MitsubaInstallerStub.exe"
$installerOut = Join-Path $dist "what-a-relief-$Version-mitsuba-backend-setup.exe"
$marker = [Text.Encoding]::ASCII.GetBytes("WHAT_A_RELIEF_MITSUBA_PAYLOAD_V1")

$pythonVersion = "3.13.13"
$pythonArchiveName = "python-$pythonVersion-embed-amd64.zip"
$pythonArchiveSha256 = "8766a8775746235e23cf5aee5027ab1060bb981d93110577adcf3508aa0cbd55"
$pythonArchiveUrl = "https://www.python.org/ftp/python/$pythonVersion/$pythonArchiveName"
$llvmVersion = "18.1.6"
$llvmInstallerName = "LLVM-$llvmVersion-win64.exe"
$llvmInstallerSha256 = "e4cf89db2f4ce3aa8f661891faa59f4961b1e12df0217c9a88d20de9ca2fe25e"
$llvmInstallerUrl = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$llvmVersion/$llvmInstallerName"
$llvmDllSha256 = "76208d0506c4cde1178b4af7eec75dc907448f97a69c5a1d6259a29f629a1a8b"
$llvmLicenseSha256 = "8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee"
$llvmLicenseUrl = "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-$llvmVersion/LICENSE.TXT"

$wheels = @(
    @{
        Name = "drjit-1.3.1-cp313-cp313-win_amd64.whl"
        Sha256 = "a26e395e9f8d0e084768eeb7183b047ae425ef30a5eecf262baf60d4114e2a31"
        Url = "https://files.pythonhosted.org/packages/20/6f/1d79672b1b5eb2f8eb220fe054b0cb3dde4c10c973a26cc9e2b973cb1146/drjit-1.3.1-cp313-cp313-win_amd64.whl"
    },
    @{
        Name = "mitsuba-3.8.0-cp313-cp313-win_amd64.whl"
        Sha256 = "d94a0883906a0d1493b2b6c37e7f7a9b61676e30e65752d747984de6567c34fe"
        Url = "https://files.pythonhosted.org/packages/8a/b4/c712a4081a3892f5ca27f8584738b8ced40170d6621de20e6a5350a1e756/mitsuba-3.8.0-cp313-cp313-win_amd64.whl"
    },
    @{
        Name = "numpy-2.3.3-cp313-cp313-win_amd64.whl"
        Sha256 = "f0dadeb302887f07431910f67a14d57209ed91130be0adea2f9793f1a4f817cf"
        Url = "https://files.pythonhosted.org/packages/1b/b5/263ebbbbcede85028f30047eab3d58028d7ebe389d6493fc95ae66c636ab/numpy-2.3.3-cp313-cp313-win_amd64.whl"
    }
)

function Assert-SafeChildPath {
    param([string]$Parent, [string]$Child)
    $resolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $resolvedChild = [IO.Path]::GetFullPath($Child)
    if (-not $resolvedChild.StartsWith($resolvedParent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside $resolvedParent`: $resolvedChild"
    }
}

function Get-VerifiedArtifact {
    param(
        [string]$Path,
        [string]$Url,
        [string]$Sha256
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "Downloading $(Split-Path -Leaf $Path)..."
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Path
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Sha256 but found $actual."
    }
}

function Assert-FileHash {
    param([string]$Path, [string]$Sha256)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required file not found: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Sha256 but found $actual."
    }
}

function Expand-ZipMerged {
    param([string]$ArchivePath, [string]$Destination)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $destinationRoot = [IO.Path]::GetFullPath($Destination).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $zip = [IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($ArchivePath))
    try {
        foreach ($entry in $zip.Entries) {
            $relative = $entry.FullName.Replace('/', [IO.Path]::DirectorySeparatorChar)
            $target = [IO.Path]::GetFullPath((Join-Path $destinationRoot $relative))
            if (-not $target.StartsWith($destinationRoot, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Unsafe path in archive $ArchivePath`: $($entry.FullName)"
            }
            if ([string]::IsNullOrEmpty($entry.Name)) {
                New-Item -ItemType Directory -Path $target -Force | Out-Null
                continue
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            $input = $entry.Open()
            $output = [IO.File]::Open($target, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $input.CopyTo($output)
            } finally {
                $output.Dispose()
                $input.Dispose()
            }
        }
    } finally {
        $zip.Dispose()
    }
}

function Test-BackendProbe {
    param([string]$RuntimeDirectory, [string]$Backend)
    $python = Join-Path $RuntimeDirectory "python.exe"
    $worker = Join-Path $RuntimeDirectory "worker.py"
    $result = Join-Path $RuntimeDirectory "probe-$Backend.json"
    Remove-Item -LiteralPath $result -Force -ErrorAction SilentlyContinue
    & $python $worker --probe --backend $Backend --result $result
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $result)) {
        return $false
    }
    $probe = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
    return $probe.status -eq "available"
}

New-Item -ItemType Directory -Path $dist -Force | Out-Null
New-Item -ItemType Directory -Path $cache -Force | Out-Null
Assert-SafeChildPath -Parent $dist -Child $work
if (Test-Path -LiteralPath $work) {
    Remove-Item -LiteralPath $work -Recurse -Force
}
New-Item -ItemType Directory -Path $runtime -Force | Out-Null

$pythonArchive = Join-Path $cache $pythonArchiveName
Get-VerifiedArtifact -Path $pythonArchive -Url $pythonArchiveUrl -Sha256 $pythonArchiveSha256
Expand-ZipMerged -ArchivePath $pythonArchive -Destination $runtime

$sitePackages = Join-Path $runtime "Lib\site-packages"
New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
foreach ($wheel in $wheels) {
    $wheelPath = Join-Path $cache $wheel.Name
    Get-VerifiedArtifact -Path $wheelPath -Url $wheel.Url -Sha256 $wheel.Sha256
    Expand-ZipMerged -ArchivePath $wheelPath -Destination $sitePackages
}

Set-Content -LiteralPath (Join-Path $runtime "python313._pth") -Encoding ASCII -Value @(
    "python313.zip",
    ".",
    "Lib\site-packages",
    "import site"
)
Move-Item -LiteralPath (Join-Path $runtime "LICENSE.txt") -Destination (Join-Path $runtime "PYTHON_LICENSE.txt") -Force

$llvmDllCache = Join-Path $cache "LLVM-C-$llvmVersion.dll"
if (-not (Test-Path -LiteralPath $llvmDllCache)) {
    $llvmInstaller = Join-Path $cache $llvmInstallerName
    Get-VerifiedArtifact -Path $llvmInstaller -Url $llvmInstallerUrl -Sha256 $llvmInstallerSha256
    $llvmStage = Join-Path $work "llvm-stage"
    Assert-SafeChildPath -Parent $work -Child $llvmStage
    try {
        $process = Start-Process -FilePath $llvmInstaller -ArgumentList @("/S", "/D=$llvmStage") -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "LLVM staging installer exited with code $($process.ExitCode)."
        }
        $stagedDll = Join-Path $llvmStage "bin\LLVM-C.dll"
        if (-not (Test-Path -LiteralPath $stagedDll)) {
            throw "LLVM staging did not produce bin\LLVM-C.dll."
        }
        Copy-Item -LiteralPath $stagedDll -Destination $llvmDllCache -Force
    } finally {
        $llvmUninstaller = Join-Path $llvmStage "Uninstall.exe"
        if (Test-Path -LiteralPath $llvmUninstaller) {
            $cleanup = Start-Process -FilePath $llvmUninstaller -ArgumentList "/S" -WindowStyle Hidden -Wait -PassThru
            if ($cleanup.ExitCode -ne 0) {
                Write-Warning "LLVM staging cleanup exited with code $($cleanup.ExitCode)."
            }
        } elseif (Test-Path -LiteralPath $llvmStage) {
            Remove-Item -LiteralPath $llvmStage -Recurse -Force
        }
    }
}
Assert-FileHash -Path $llvmDllCache -Sha256 $llvmDllSha256
$llvmBin = Join-Path $runtime "llvm\bin"
New-Item -ItemType Directory -Path $llvmBin -Force | Out-Null
Copy-Item -LiteralPath $llvmDllCache -Destination (Join-Path $llvmBin "LLVM-C.dll") -Force

$llvmLicense = Join-Path $cache "LLVM-$llvmVersion-LICENSE.TXT"
Get-VerifiedArtifact -Path $llvmLicense -Url $llvmLicenseUrl -Sha256 $llvmLicenseSha256
Copy-Item -LiteralPath $llvmLicense -Destination (Join-Path $runtime "LLVM_LICENSE.TXT") -Force
Copy-Item -LiteralPath (Join-Path $repo "tools\mitsuba_backend\worker.py") -Destination (Join-Path $runtime "worker.py") -Force
Copy-Item -LiteralPath (Join-Path $repo "tools\mitsuba_backend\requirements.lock.txt") -Destination (Join-Path $runtime "requirements.lock.txt") -Force
Copy-Item -LiteralPath (Join-Path $repo "tools\mitsuba_backend\README.md") -Destination (Join-Path $runtime "MITSUBA_BACKEND.md") -Force
Copy-Item -LiteralPath (Join-Path $repo "LICENSE") -Destination (Join-Path $runtime "WHAT_A_RELIEF_LICENSE.txt") -Force
Copy-Item -LiteralPath (Join-Path $repo "THIRD_PARTY_NOTICES.md") -Destination (Join-Path $runtime "THIRD_PARTY_NOTICES.md") -Force

$manifest = [ordered]@{
    product = "what-a-relief-mitsuba"
    release_version = $Version
    python_version = $pythonVersion
    mitsuba_version = "3.8.0"
    drjit_version = "1.3.1"
    numpy_version = "2.3.3"
    llvm_version = $llvmVersion
    architecture = "win_amd64"
    build_inputs = [ordered]@{
        python_embed_sha256 = $pythonArchiveSha256
        llvm_c_sha256 = $llvmDllSha256
        wheel_sha256 = [ordered]@{
            drjit = $wheels[0].Sha256
            mitsuba = $wheels[1].Sha256
            numpy = $wheels[2].Sha256
        }
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runtime "backend-manifest.json") -Encoding UTF8

$runtimePython = Join-Path $runtime "python.exe"
& $runtimePython `
    (Join-Path $repo "tests\test_mitsuba_worker_progress.py") `
    (Join-Path $runtime "worker.py")
if ($LASTEXITCODE -ne 0) {
    throw "The Mitsuba network progress-writer regression test failed."
}
& $runtimePython `
    (Join-Path $repo "tests\test_mitsuba_worker_png.py") `
    (Join-Path $runtime "worker.py")
if ($LASTEXITCODE -ne 0) {
    throw "The Mitsuba preview PNG-writer regression test failed."
}

if (-not (Test-BackendProbe -RuntimeDirectory $runtime -Backend "llvm")) {
    throw "The self-contained LLVM CPU backend failed its live render probe."
}
$cudaAvailable = Test-BackendProbe -RuntimeDirectory $runtime -Backend "cuda"
Remove-Item -LiteralPath (Join-Path $runtime "probe-llvm.json") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $runtime "probe-cuda.json") -Force -ErrorAction SilentlyContinue

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path -LiteralPath $payloadZip) {
    Remove-Item -LiteralPath $payloadZip -Force
}
[IO.Compression.ZipFile]::CreateFromDirectory(
    $runtime,
    $payloadZip,
    [IO.Compression.CompressionLevel]::Optimal,
    $false)

$stub = @"
using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

internal static class MitsubaInstallerStub {
    private const string ProductName = "what-a-relief Mitsuba backend";
    private const string InstallFolderName = "what-a-relief-mitsuba";
    private const string Version = "$Version";
    private static readonly byte[] Marker = Encoding.ASCII.GetBytes("WHAT_A_RELIEF_MITSUBA_PAYLOAD_V1");

    [STAThread]
    private static int Main(string[] args) {
        string payloadPath = null;
        string tempRoot = null;
        bool extractOnly = args.Length == 2 && string.Equals(args[0], "--extract", StringComparison.OrdinalIgnoreCase);
        bool quietInstall = args.Length == 1 && string.Equals(args[0], "--quiet", StringComparison.OrdinalIgnoreCase);
        try {
            tempRoot = Path.Combine(Path.GetTempPath(), "WhatAReliefMitsubaInstall-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempRoot);
            payloadPath = Path.Combine(tempRoot, "runtime.zip");
            ExtractEmbeddedPayload(payloadPath);
            if (extractOnly) {
                string extractionTarget = Path.GetFullPath(args[1]);
                Directory.CreateDirectory(extractionTarget);
                ExtractZipSafely(payloadPath, extractionTarget);
                VerifyRuntime(extractionTarget);
                return 0;
            }

            string availability = InstallPayload(payloadPath);
            if (!quietInstall) {
                MessageBox.Show(
                    ProductName + " " + Version + " is ready.\n\n" + availability +
                    "\n\nRestart what-a-relief if it is currently open.",
                    ProductName + " Installer",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
            }
            return 0;
        } catch (Exception ex) {
            if (!quietInstall) {
                MessageBox.Show(ex.Message, ProductName + " Installer Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            return 1;
        } finally {
            if (tempRoot != null) {
                try { Directory.Delete(tempRoot, true); } catch { }
            }
        }
    }

    private static string InstallPayload(string payloadPath) {
        string programs = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs");
        string installDir = Path.Combine(programs, InstallFolderName);
        string stagingDir = installDir + ".installing-" + Guid.NewGuid().ToString("N");
        string backupDir = installDir + ".previous-" + Guid.NewGuid().ToString("N");
        Directory.CreateDirectory(stagingDir);
        try {
            ExtractZipSafely(payloadPath, stagingDir);
            VerifyRuntime(stagingDir);
            string availability = ProbeRuntime(stagingDir);

            bool hadPrevious = Directory.Exists(installDir);
            if (hadPrevious) {
                if (!IsRecognizedInstall(installDir)) {
                    throw new InvalidOperationException("Refusing to replace an unrecognized directory: " + installDir);
                }
                if (IsFileLocked(Path.Combine(installDir, "python.exe"))) {
                    throw new InvalidOperationException(
                        "The Mitsuba backend is still running. Close what-a-relief, then run this installer again.");
                }
                Directory.Move(installDir, backupDir);
            }

            try {
                Directory.Move(stagingDir, installDir);
                WriteUninstaller(installDir);
                RegisterUninstall(installDir);
            } catch {
                try { if (Directory.Exists(installDir)) Directory.Delete(installDir, true); } catch { }
                if (hadPrevious && Directory.Exists(backupDir)) {
                    Directory.Move(backupDir, installDir);
                }
                throw;
            }

            if (Directory.Exists(backupDir)) {
                try { Directory.Delete(backupDir, true); } catch { }
            }
            return availability;
        } finally {
            if (Directory.Exists(stagingDir)) {
                try { Directory.Delete(stagingDir, true); } catch { }
            }
        }
    }

    private static bool IsRecognizedInstall(string path) {
        return File.Exists(Path.Combine(path, "backend-manifest.json")) &&
            File.Exists(Path.Combine(path, "python.exe")) &&
            File.Exists(Path.Combine(path, "worker.py"));
    }

    private static bool IsFileLocked(string path) {
        if (!File.Exists(path)) return false;
        try {
            using (FileStream stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.None)) { }
            return false;
        } catch (IOException) {
            return true;
        } catch (UnauthorizedAccessException) {
            return true;
        }
    }

    private static string ProbeRuntime(string runtimeDir) {
        bool cpu = RunProbe(runtimeDir, "llvm");
        bool cuda = RunProbe(runtimeDir, "cuda");
        if (!cpu) {
            throw new InvalidOperationException(
                "The bundled CPU backend failed its live render probe. No changes were installed.");
        }
        return cuda
            ? "CPU and NVIDIA CUDA processing passed live render probes."
            : "CPU processing passed its live render probe. NVIDIA CUDA was not available, so Auto mode will use the CPU.";
    }

    private static bool RunProbe(string runtimeDir, string backend) {
        string result = Path.Combine(runtimeDir, "probe-" + backend + ".json");
        try { File.Delete(result); } catch { }
        ProcessStartInfo psi = new ProcessStartInfo(
            Path.Combine(runtimeDir, "python.exe"),
            "\"" + Path.Combine(runtimeDir, "worker.py") + "\" --probe --backend " + backend + " --result \"" + result + "\"");
        psi.WorkingDirectory = runtimeDir;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        using (Process process = Process.Start(psi)) {
            process.WaitForExit();
            bool available = process.ExitCode == 0 && File.Exists(result) &&
                File.ReadAllText(result).IndexOf("\"status\": \"available\"", StringComparison.Ordinal) >= 0;
            try { File.Delete(result); } catch { }
            return available;
        }
    }

    private static void VerifyRuntime(string runtimeDir) {
        string[] required = {
            "python.exe",
            "python313.dll",
            "python313.zip",
            "worker.py",
            "backend-manifest.json",
            Path.Combine("llvm", "bin", "LLVM-C.dll"),
            Path.Combine("Lib", "site-packages", "mitsuba", "__init__.py"),
            Path.Combine("Lib", "site-packages", "drjit", "__init__.py"),
            Path.Combine("Lib", "site-packages", "numpy", "__init__.py")
        };
        foreach (string relative in required) {
            string path = Path.Combine(runtimeDir, relative);
            if (!File.Exists(path) || new FileInfo(path).Length == 0) {
                throw new InvalidOperationException("Backend payload is incomplete: " + relative);
            }
        }
    }

    private static void ExtractZipSafely(string zipPath, string destination) {
        string root = Path.GetFullPath(destination).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        using (ZipArchive archive = ZipFile.OpenRead(zipPath)) {
            foreach (ZipArchiveEntry entry in archive.Entries) {
                string relative = entry.FullName.Replace('/', Path.DirectorySeparatorChar);
                string target = Path.GetFullPath(Path.Combine(root, relative));
                if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase)) {
                    throw new InvalidOperationException("Unsafe path in backend payload: " + entry.FullName);
                }
                if (string.IsNullOrEmpty(entry.Name)) {
                    Directory.CreateDirectory(target);
                    continue;
                }
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                using (Stream input = entry.Open())
                using (FileStream output = new FileStream(target, FileMode.Create, FileAccess.Write, FileShare.None)) {
                    input.CopyTo(output);
                }
            }
        }
    }

    private static void ExtractEmbeddedPayload(string payloadPath) {
        string self = Assembly.GetExecutingAssembly().Location;
        using (FileStream input = File.OpenRead(self)) {
            if (input.Length < Marker.Length + 8) {
                throw new InvalidOperationException("Installer payload is missing.");
            }
            input.Seek(-8, SeekOrigin.End);
            long payloadLength = BitConverter.ToInt64(ReadExact(input, 8), 0);
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
            using (FileStream output = File.Create(payloadPath)) {
                CopyBytes(input, output, payloadLength);
            }
        }
    }

    private static void WriteUninstaller(string installDir) {
        string uninstallScript = @"`$ErrorActionPreference = ""Stop""
`$installDir = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$expectedInstallDir = Join-Path `$env:LOCALAPPDATA ""Programs\what-a-relief-mitsuba""
`$actual = [IO.Path]::GetFullPath(`$installDir).TrimEnd('\')
`$expected = [IO.Path]::GetFullPath(`$expectedInstallDir).TrimEnd('\')
if (`$actual -ne `$expected) {
    throw ""Refusing to uninstall from unexpected directory: `$installDir""
}
`$uninstallKey = ""HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\what-a-relief-mitsuba""
Remove-Item -LiteralPath `$uninstallKey -Recurse -Force -ErrorAction SilentlyContinue
`$cmd = ""/c timeout /t 1 /nobreak > nul & rmdir /s /q """""" + `$installDir + """"""""
Start-Process -FilePath `$env:ComSpec -ArgumentList `$cmd -WindowStyle Hidden
";
        File.WriteAllText(Path.Combine(installDir, "uninstall.ps1"), uninstallScript, new UTF8Encoding(false));
    }

    private static void RegisterUninstall(string installDir) {
        string uninstallScript = Path.Combine(installDir, "uninstall.ps1");
        using (RegistryKey key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\what-a-relief-mitsuba")) {
            key.SetValue("DisplayName", ProductName);
            key.SetValue("DisplayVersion", Version);
            key.SetValue("Publisher", "what-a-relief Contributors");
            key.SetValue("InstallLocation", installDir);
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
            if (read == 0) throw new EndOfStreamException();
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
            if (read == 0) throw new EndOfStreamException();
            output.Write(buffer, 0, read);
            remaining -= read;
        }
    }
}
"@

Set-Content -LiteralPath $stubSource -Value $stub -Encoding UTF8
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $csc)) {
    $csc = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe"
}
if (-not (Test-Path -LiteralPath $csc)) {
    throw "Could not find csc.exe to build the backend installer."
}

& $csc /nologo /target:winexe /reference:System.Windows.Forms.dll /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll /out:$stubExe $stubSource
if ($LASTEXITCODE -ne 0) {
    throw "csc failed with exit code $LASTEXITCODE"
}

if (Test-Path -LiteralPath $installerOut) {
    Remove-Item -LiteralPath $installerOut -Force
}
Copy-Item -LiteralPath $stubExe -Destination $installerOut -Force
$payloadBytes = [IO.File]::ReadAllBytes($payloadZip)
$lengthBytes = [BitConverter]::GetBytes([Int64]$payloadBytes.Length)
$stream = [IO.File]::Open($installerOut, [IO.FileMode]::Append, [IO.FileAccess]::Write)
try {
    $stream.Write($payloadBytes, 0, $payloadBytes.Length)
    $stream.Write($marker, 0, $marker.Length)
    $stream.Write($lengthBytes, 0, $lengthBytes.Length)
} finally {
    $stream.Dispose()
}

$verify = Join-Path $work "verify"
New-Item -ItemType Directory -Path $verify -Force | Out-Null
$verifyProcess = Start-Process -FilePath $installerOut -ArgumentList @("--extract", $verify) -WindowStyle Hidden -Wait -PassThru
if ($verifyProcess.ExitCode -ne 0) {
    throw "Backend installer self-extraction verification failed with exit code $($verifyProcess.ExitCode)."
}
if (-not (Test-BackendProbe -RuntimeDirectory $verify -Backend "llvm")) {
    throw "The extracted backend installer payload failed its LLVM CPU probe."
}
Remove-Item -LiteralPath $work -Recurse -Force

$artifact = Get-Item -LiteralPath $installerOut
$artifactHash = (Get-FileHash -LiteralPath $installerOut -Algorithm SHA256).Hash
Write-Host "Created self-contained backend installer: $installerOut"
Write-Host "Size: $($artifact.Length) bytes"
Write-Host "SHA-256: $artifactHash"
Write-Host "Build-machine CUDA probe: $cudaAvailable"
