#!/bin/bash
# vrep installer for Linux
set -e

echo "🔨 Building vrep-core in release mode..."
cargo build --release

echo "🚀 Installing to /usr/local/bin/vrep (requires sudo)..."
sudo cp target/release/vrep-core /usr/local/bin/vrep

echo "✅ Installation complete. Run 'vrep --help' to get started."
