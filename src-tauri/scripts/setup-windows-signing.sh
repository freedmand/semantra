#!/usr/bin/env bash
# Prepare Azure Trusted Signing on a Windows GitHub runner (Git Bash): find
# signtool, fetch Microsoft's Trusted Signing dlib, write its metadata, export
# SIGNTOOL / SIGNING_DLIB / SIGNING_METADATA to later steps, and write
# tauri.windows-sign.conf.json pointing Tauri's signCommand at
# sign-windows.sh. Used by release.yml and sign-test.yml.
#
# Needs AZURE_SIGNING_ENDPOINT, AZURE_SIGNING_ACCOUNT, AZURE_SIGNING_PROFILE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tools="$RUNNER_TEMP/signing"
mkdir -p "$tools"
signtool="$(find "/c/Program Files (x86)/Windows Kits/10/bin" -name signtool.exe -path '*x64*' | sort -V | tail -1)"
[[ -n "$signtool" ]] || { echo "signtool.exe not found" >&2; exit 1; }
curl -fsSL --retry 3 -o "$tools/client.nupkg" https://www.nuget.org/api/v2/package/Microsoft.Trusted.Signing.Client
unzip -q -o "$tools/client.nupkg" -d "$tools/client"
dlib="$(find "$tools/client" -iname Azure.CodeSigning.Dlib.dll -path '*x64*' | head -1)"
[[ -n "$dlib" ]] || { echo "Azure.CodeSigning.Dlib.dll not found in the NuGet package" >&2; exit 1; }
# Authenticate only through the Azure CLI (logged in per file by the script).
python - "$tools/metadata.json" <<'PY'
import json, os, sys
json.dump({
    "Endpoint": os.environ["AZURE_SIGNING_ENDPOINT"],
    "CodeSigningAccountName": os.environ["AZURE_SIGNING_ACCOUNT"],
    "CertificateProfileName": os.environ["AZURE_SIGNING_PROFILE"],
    "ExcludeCredentials": ["ManagedIdentityCredential", "WorkloadIdentityCredential", "EnvironmentCredential",
                           "SharedTokenCacheCredential", "VisualStudioCredential", "VisualStudioCodeCredential",
                           "AzurePowerShellCredential", "AzureDeveloperCliCredential", "InteractiveBrowserCredential"],
}, open(sys.argv[1], "w"))
PY
{
  echo "SIGNTOOL=$signtool"
  echo "SIGNING_DLIB=$(cygpath -w "$dlib")"
  echo "SIGNING_METADATA=$(cygpath -w "$tools/metadata.json")"
} >> "$GITHUB_ENV"
# Absolute Git Bash: Tauri spawns signCommand as a plain Windows process, and
# a bare `bash` there resolves to the WSL launcher (C:\Windows\System32),
# which fails on runners without a distro. The cmd/args form keeps the
# spaces in "Program Files" intact.
python - "$(cygpath -w "$(command -v bash)")" "$(cygpath -m "$ROOT")/scripts/sign-windows.sh" \
  "$ROOT/tauri.windows-sign.conf.json" <<'PY'
import json, sys
bash, script, out = sys.argv[1:]
json.dump({"bundle": {"windows": {"signCommand": {"cmd": bash, "args": [script, "%1"]}}}}, open(out, "w"))
PY
cat "$ROOT/tauri.windows-sign.conf.json"
"$signtool" /? 2>&1 | head -2 || true
