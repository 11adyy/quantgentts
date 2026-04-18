#!/bin/bash
# Regenerate + reload the 5 launchd plists for quantgents.
#
# Why this script exists: on macOS Sequoia, editing `run_if_et_window.sh`
# or the plist files themselves re-applies the `com.apple.provenance`
# extended attribute, after which launchd refuses to exec the file with
# exit code 126 ("Operation not permitted") — silently. The fix is to
# have launchd exec /bin/bash (a system binary, always exec-able) and
# pass the wrapper script as an argument. bash READS the wrapper as
# text, so provenance doesn't apply.
#
# Re-run this script any time you edit the wrapper or plist content.
# Needs login-session auth; do NOT run from ssh.

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/Users/noah/Documents/Claude-workspace/quantgents}"
SCRIPT="${PROJECT_ROOT}/scripts/run_if_et_window.sh"
AGENTS="${HOME}/Library/LaunchAgents"

mkdir -p "$AGENTS"

write_plist() {
    local mode="$1"     # e.g. morning
    local suffix="$2"   # filename suffix: morning / midday / evening / intra / earnings
    local log="$3"      # log filename

    cat > "${AGENTS}/com.quantgents.${suffix}.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quantgents.${suffix}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${SCRIPT}</string>
        <string>${mode}</string>
    </array>
    <key>StartInterval</key>
    <integer>1800</integer>
    <key>StandardOutPath</key>
    <string>${PROJECT_ROOT}/logs/${log}</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_ROOT}/logs/${log}</string>
</dict>
</plist>
EOF
    echo "wrote ${AGENTS}/com.quantgents.${suffix}.plist"
}

write_plist "morning"             "morning"  "launchd_morning.log"
write_plist "midday"              "midday"   "launchd_midday.log"
write_plist "evening"             "evening"  "launchd_evening.log"
write_plist "intra_check"         "intra"    "launchd_intra.log"
write_plist "earnings_preprocess" "earnings" "launchd_earnings.log"

# Drop provenance xattr so even direct-exec retries work. Harmless if absent.
for p in \
    "${SCRIPT}" \
    "${AGENTS}/com.quantgents.morning.plist" \
    "${AGENTS}/com.quantgents.midday.plist" \
    "${AGENTS}/com.quantgents.evening.plist" \
    "${AGENTS}/com.quantgents.intra.plist" \
    "${AGENTS}/com.quantgents.earnings.plist"; do
    xattr -d com.apple.provenance "$p" 2>/dev/null || true
done
echo "xattr com.apple.provenance cleared"

# Bootout + bootstrap so launchd picks up the new plists. `|| true` on
# bootout so a first-time install (nothing to unload) doesn't abort.
for suffix in morning midday evening intra earnings; do
    launchctl bootout "gui/${UID}/com.quantgents.${suffix}" 2>/dev/null || true
    launchctl bootstrap "gui/${UID}" "${AGENTS}/com.quantgents.${suffix}.plist"
    echo "reloaded com.quantgents.${suffix}"
done

echo ""
echo "Done. Verify with:  launchctl list | grep quantgents"
echo "A healthy row shows a PID (when running) or '-' with last-exit 0."
