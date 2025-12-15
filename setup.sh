#!/usr/bin/env bash

# 1. Install Python deps
pip install fair-esm

# 2. Download Nsight Systems .deb
NSYS_DEB=nsight-systems-2025.6.1_2025.6.1.190-1_amd64.deb
if [ ! -f "$NSYS_DEB" ]; then
  wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/${NSYS_DEB}
fi

# 3. Install the .deb (may leave package unconfigured due to missing libs)
sudo dpkg -i "$NSYS_DEB" || true

# 4. Fix dependencies non-interactively
sudo apt-get update -y
sudo apt-get -f install -y