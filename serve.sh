#!/usr/bin/env bash
# Run this from inside the folder that contains index.html, data.json, and Wifi7Tests.csv
# It will start a local web server, then you open http://localhost:8000 in your browser.
# Press Ctrl+C to stop.

cd "$(dirname "$0")"
echo "Serving site at http://localhost:8000"
echo "Press Ctrl+C to stop."
python3 -m http.server 8000
