#!/bin/bash
# Publish the round of a Mac; the Linux results online are left untouched.
exec "$(dirname "$0")/tools/publish.sh" macos "$@"
