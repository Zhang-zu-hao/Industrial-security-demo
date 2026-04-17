#!/bin/bash
# Docker Installation Script for NVIDIA Jetson (JetPack 6.x)
# This script installs Docker Engine with NVIDIA Container Runtime

set -euo pipefail

echo "=== Docker Installation for Jetson ==="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   echo "Please run: sudo bash $0"
   exit 1
fi

# Remove old versions
echo "Removing old Docker versions..."
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Install prerequisites
echo "Installing prerequisites..."
apt-get update
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
echo "Adding Docker GPG key..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo "Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
echo "Installing Docker Engine..."
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
echo "Adding user to docker group..."
usermod -aG docker ${SUDO_USER:-$USER}

# Configure NVIDIA Container Runtime
echo "Configuring NVIDIA Container Runtime..."
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

# Restart Docker
echo "Restarting Docker service..."
systemctl restart docker

# Verify installation
echo ""
echo "=== Installation Complete ==="
echo ""
docker --version
docker compose version
echo ""
echo "Docker service status:"
systemctl status docker --no-pager | head -n 5
echo ""
echo "✅ Docker installed successfully!"
echo ""
echo "⚠️  IMPORTANT: Please log out and log back in, or run:"
echo "   newgrp docker"
echo ""
echo "Then you can use Docker commands without sudo."
