#!/usr/bin/env bash
# run_demo.sh — Execute the RiTa DPDP Compliance Agent demo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check for .env
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "⚠️  No .env file found."
    echo "   Copy .env.example → .env and set your GOOGLE_API_KEY."
    echo ""
    read -rp "Enter your GOOGLE_API_KEY (or press Enter to abort): " api_key
    if [ -z "$api_key" ]; then
        echo "Aborted."
        exit 1
    fi
    echo "GOOGLE_API_KEY=$api_key" > "$SCRIPT_DIR/.env"
    echo "✅ .env file created."
fi

echo ""
echo "🚀 Running RiTa Agent..."
echo ""

python "$SCRIPT_DIR/langgraph_agent.py"
