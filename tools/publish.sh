#!/bin/bash
# Publish one platform's round to the website, leaving the other platforms online untouched.
#
#     tools/publish.sh <platform id> [round directory, default: result/full or results/full]
#
# Only what visitors see is uploaded: pages, charts, gallery images and videos. The platform's
# own directory is mirrored (stale files deleted); platforms.json is merged with the online copy
# so every other platform stays listed. Use publish-linux.sh or publish-macos.sh.
set -euo pipefail

REMOTE=pyvrciz@ssh.cluster028.hosting.ovh.net
REMOTE_DIR=/home/pyvrciz/mlh3wnip3f

id=${1:?usage: tools/publish.sh <platform id> [round directory]}
round=""
if [ -n "${2:-}" ]; then round=$(cd "$2" && pwd); fi
cd "$(dirname "$0")/.."
if [ -z "$round" ]; then
    for candidate in result/full results/full; do
        if [ -f "$candidate/round-plan.json" ]; then round=$candidate; break; fi
    done
    : "${round:?no round in result/full or results/full; pass its directory}"
fi

for report in "$round"/*/index.html; do
    if ! grep -q 'id="pm-platform"' "$report"; then
        echo "$report predates the platform selector; regenerate it first:" >&2
        echo "  cargo run --release -- report $(dirname "$report")" >&2
        exit 1
    fi
done

# One SSH connection for every transfer, so the password is asked once.
socket_dir=$(mktemp -d /tmp/physmark-ssh.XXXXXX)
socket="$socket_dir/control"
cleanup() {
    ssh -S "$socket" -O exit "$REMOTE" 2>/dev/null || true
    rm -rf "$socket_dir"
}
trap cleanup EXIT
ssh -f -N -M -S "$socket" "$REMOTE"
ssh_command="ssh -S $socket"

site=result/publish
rm -rf "$site"
mkdir -p "$site"
# Start from the online manifest; it is missing only before the first publication.
rsync -a -e "$ssh_command" "$REMOTE:$REMOTE_DIR/platforms.json" "$site/" 2>/dev/null || echo "No platforms.json online yet; starting a new one."
python3 tools/site.py "$site" "$round" --id "$id" --report-only

rsync -av -e "$ssh_command" "$site/index.html" "$site/platforms.json" "$REMOTE:$REMOTE_DIR/"
rsync -av --delete-after -e "$ssh_command" "$site/$id/" "$REMOTE:$REMOTE_DIR/$id/"
