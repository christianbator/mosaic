#!/usr/bin/env bash

#
# download_data.sh
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

set -euo pipefail

#
# Colors
#
cyan="\033[36m"
green="\033[32m"
reset="\033[0m"

#
# Locations
#
data_dir=benchmark/data

mkdir -p $data_dir

#
# Download
#
download() {
    local filename=$1
    local url=$2

    if [ -f "$data_dir/$filename" ]; then
        echo -e "> ${cyan}$filename${reset} already downloaded, skipping ..."
    else
        echo -e "> Downloading ${cyan}$filename${reset} ..."
        curl --fail --location --silent --show-error --output "$data_dir/$filename" "$url"
    fi
}

#
# Images
#

# Scarlet macaw (7792 x 5195)
#   Photographer: David Clode
#   Source: https://unsplash.com/photos/Gooxe2__kds
#   License: Unsplash License (https://unsplash.com/license)
download macaw.jpeg "https://unsplash.com/photos/Gooxe2__kds/download?force=true"

#
# Notify of success
#
echo -e "> Benchmark data ready ${green}✔${reset}"
