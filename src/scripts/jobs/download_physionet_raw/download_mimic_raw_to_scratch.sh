#!/bin/bash
set -euo pipefail

# Download raw PhysioNet MIMIC resources to scratch and expose stable symlinks
# inside the repo. These datasets are credentialed; configure ~/.netrc first:
#   machine physionet.org login YOUR_USERNAME password YOUR_PASSWORD
#   chmod 600 ~/.netrc
#
# Versions match the preprocessing notebooks in this repo:
#   MIMIC-IV       2.2
#   MIMIC-IV-Note 2.2
#   MIMIC-CXR     2.1.0
#   MIMIC-IV-ECG  1.0

ROOT="${ROOT:-/scratch/gilbreth/pham156/FuseMoE_data/raw/physionet}"
REPO_RAW_LINK_ROOT="${REPO_RAW_LINK_ROOT:-/home/pham156/MoE/FuseMoE_poly/data/raw}"

# Full MIMIC-CXR image download is very large. Metadata is enough for matching
# subject/study/dicom ids; set DOWNLOAD_CXR_IMAGES=1 when you truly want images.
DOWNLOAD_CXR_IMAGES="${DOWNLOAD_CXR_IMAGES:-0}"

mkdir -p "$ROOT" "$REPO_RAW_LINK_ROOT"

if [[ ! -f "$HOME/.netrc" ]]; then
  echo "Missing $HOME/.netrc. PhysioNet credentialed downloads require it." >&2
  echo "Create it with:" >&2
  echo "  machine physionet.org login YOUR_USERNAME password YOUR_PASSWORD" >&2
  echo "  chmod 600 $HOME/.netrc" >&2
  exit 2
fi

chmod 600 "$HOME/.netrc"

wget_common=(
  --continue
  --timestamping
  --recursive
  --no-parent
  --no-host-directories
  --cut-dirs=2
  --directory-prefix "$ROOT"
  --auth-no-challenge
  --load-cookies /dev/null
)

download_project() {
  local name="$1"
  local url="$2"
  shift 2
  echo "===== downloading $name ====="
  echo "url: $url"
  wget "${wget_common[@]}" "$@" "$url"
}

# Structured EHR: TS/labs/vitals and ICU stay tables.
download_project \
  "MIMIC-IV v2.2" \
  "https://physionet.org/files/mimiciv/2.2/"

# Clinical notes.
download_project \
  "MIMIC-IV-Note v2.2" \
  "https://physionet.org/files/mimic-iv-note/2.2/"

# CXR metadata is small enough and useful for matching. Images are optional.
if [[ "$DOWNLOAD_CXR_IMAGES" == "1" ]]; then
  download_project \
    "MIMIC-CXR v2.1.0 full" \
    "https://physionet.org/files/mimic-cxr/2.1.0/"
else
  download_project \
    "MIMIC-CXR v2.1.0 metadata only" \
    "https://physionet.org/files/mimic-cxr/2.1.0/" \
    --reject-regex '/files/'
fi

# ECG matched subset. This includes waveform records and record_list.csv.
download_project \
  "MIMIC-IV-ECG v1.0" \
  "https://physionet.org/files/mimic-iv-ecg/1.0/"

ln -sfn "$ROOT/mimiciv/2.2" "$REPO_RAW_LINK_ROOT/mimiciv-2.2"
ln -sfn "$ROOT/mimic-iv-note/2.2" "$REPO_RAW_LINK_ROOT/mimic-iv-note-2.2"
ln -sfn "$ROOT/mimic-cxr/2.1.0" "$REPO_RAW_LINK_ROOT/mimic-cxr-2.1.0"
ln -sfn "$ROOT/mimic-iv-ecg/1.0" "$REPO_RAW_LINK_ROOT/mimic-iv-ecg-1.0"

echo "===== done ====="
echo "scratch root: $ROOT"
echo "repo symlinks:"
ls -lh "$REPO_RAW_LINK_ROOT"
