#!/bin/bash
# Publish the Linux + NVIDIA round of this machine; the macOS results online are left untouched.
exec "$(dirname "$0")/tools/publish.sh" linux "$@"
