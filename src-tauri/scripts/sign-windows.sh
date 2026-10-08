#!/usr/bin/env bash
# Sign one Windows file with Azure Trusted Signing — Tauri's
# bundle.windows.signCommand during release builds (see release.yml, which
# generates the config pointing here). Tauri calls it for the app executable,
# the ffmpeg sidecar and the NSIS installer/uninstaller.
#
# Auth is GitHub OIDC -> Entra ID federated credential (no client secret to
# expire): each call fetches a fresh GitHub token and logs the Azure CLI in
# with it, so a long build can't outlive the login. The Trusted Signing dlib
# then authenticates through the Azure CLI.
#
# Needs (set by release.yml): AZURE_CLIENT_ID, AZURE_TENANT_ID, SIGNTOOL,
# SIGNING_DLIB, SIGNING_METADATA, and the job's id-token permission
# (ACTIONS_ID_TOKEN_REQUEST_URL/_TOKEN).
set -euo pipefail
file="$1"
for v in AZURE_CLIENT_ID AZURE_TENANT_ID SIGNTOOL SIGNING_DLIB SIGNING_METADATA ACTIONS_ID_TOKEN_REQUEST_URL ACTIONS_ID_TOKEN_REQUEST_TOKEN; do
  if [[ -z "${!v:-}" ]]; then
    echo "sign-windows: ${v} is not set (signing only works in the release workflow)" >&2
    exit 1
  fi
done

token="$(curl -fsS -H "Authorization: bearer ${ACTIONS_ID_TOKEN_REQUEST_TOKEN}" \
  "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=api://AzureADTokenExchange" \
  | python -c "import json, sys; print(json.load(sys.stdin)['value'])")"
az login --service-principal --username "${AZURE_CLIENT_ID}" --tenant "${AZURE_TENANT_ID}" \
  --federated-token "${token}" --allow-no-subscriptions --output none
echo "sign-windows: signing ${file} as $(az account show --query user.name -o tsv 2>/dev/null)" >&2

# MSYS_NO_PATHCONV: Git Bash would otherwise rewrite signtool's /flags into
# paths. The dlib/metadata/file paths are already native Windows paths.
MSYS_NO_PATHCONV=1 "${SIGNTOOL}" sign /v /fd SHA256 /tr "http://timestamp.acs.microsoft.com" /td SHA256 \
  /dlib "${SIGNING_DLIB}" /dmdf "${SIGNING_METADATA}" "${file}" 1>&2
# Tauri shows only stderr from signCommand, so keep signtool's output there.
MSYS_NO_PATHCONV=1 "${SIGNTOOL}" verify /pa /v "${file}" 1>&2
echo "signed: ${file}"
