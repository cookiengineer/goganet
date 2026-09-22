#!/usr/bin/env bash
#
# download-datasets.sh - download and extract the network-traffic datasets used
# by GoGANet.
#
#   CTU-13  -> datasets/CTU-13-Dataset.tar.bz2         -> datasets/ctu-13/
#   IoT-23  -> datasets/iot_23_datasets_full.tar.gz    -> datasets/iot-23/
#
# The script is re-runnable: a download is skipped when the target file already
# exists and is non-empty, and extraction is skipped when the target directory
# is already populated. Use --force to re-download, and --verify if a download
# may be incomplete (it runs gzip -t / bzip2 -t before extracting).
#
# Usage:
#   ./download-datasets.sh [--force] [--verify] [--only ctu|iot]

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
datasets="$root/datasets"
mkdir -p "$datasets"

CTU_URL="https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2"
CTU_TAR="$datasets/CTU-13-Dataset.tar.bz2"
CTU_DIR="$datasets/ctu-13"

IOT_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_full.tar.gz"
IOT_TAR="$datasets/iot_23_datasets_full.tar.gz"
IOT_DIR="$datasets/iot-23"

force=0
verify=0
only=""

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --force) force=1 ;;
        --verify) verify=1 ;;
        --only) only="${2:-}"; shift ;;
        --only=*) only="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

if [ -n "$only" ] && [ "$only" != "ctu" ] && [ "$only" != "iot" ]; then
    echo "error: --only must be 'ctu' or 'iot'" >&2
    exit 2
fi

have_cmd() { command -v "$1" >/dev/null 2>&1; }

download() {
    local url="$1" dest="$2"
    if [ -s "$dest" ] && [ "$force" -eq 0 ]; then
        echo "already downloaded: $dest ($(du -h "$dest" 2>/dev/null | cut -f1))"
        return 0
    fi
    if [ "$force" -eq 1 ] && [ -e "$dest" ]; then
        echo "removing existing $dest (--force)"
        rm -f "$dest"
    fi
    echo "downloading $url"
    if have_cmd curl; then
        curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
    elif have_cmd wget; then
        wget -c -O "$dest" "$url"
    else
        echo "error: curl or wget is required" >&2
        exit 1
    fi
}

verify_archive() {
    local tar="$1"
    echo "verifying $tar (this can take a while)"
    case "$tar" in
        *.tar.gz|*.tgz) gzip -t "$tar" ;;
        *.tar.bz2) bzip2 -t "$tar" ;;
    esac
}

extract() {
    local tar="$1" dest="$2"
    shift 2
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null || true)" ]; then
        echo "already extracted: $dest"
        return 0
    fi
    if [ ! -s "$tar" ]; then
        echo "error: $tar is missing or empty" >&2
        exit 1
    fi
    if [ "$verify" -eq 1 ]; then
        verify_archive "$tar"
    fi
    mkdir -p "$dest"
    echo "extracting $tar -> $dest"
    tar -xf "$tar" -C "$dest" "$@"
}

if [ -z "$only" ] || [ "$only" = "ctu" ]; then
    download "$CTU_URL" "$CTU_TAR"
    # Single top-level CTU-13-Dataset/ directory: strip it.
    extract "$CTU_TAR" "$CTU_DIR" --strip-components=1
fi

if [ -z "$only" ] || [ "$only" = "iot" ]; then
    download "$IOT_URL" "$IOT_TAR"
    # Do not strip components: IoT-23 layouts differ between releases and the
    # trainer discovers captures recursively.
    extract "$IOT_TAR" "$IOT_DIR"
fi

echo "done."
