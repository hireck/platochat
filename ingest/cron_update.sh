#!/bin/sh
# Nightly corpus update for PLATO Chat: `make update`, fit for cron.
#
#   crontab -e
#   30 3 * * *  /path/to/Platopub/ingest/cron_update.sh
#
# cron starts jobs with next to nothing on the PATH and no login environment,
# so this sets up what `make update` needs -- LaTeXML, pandoc and Ghostscript
# for LaTeX, marker for PDFs -- and appends everything it prints to a monthly
# log in the data folder. Weaviate has to be running for the index step.
#
# update_corpus.py holds a lock, so a run that starts while the previous one is
# still going exits at once. The exit status is non-zero when the list could
# not be refreshed or the index not updated -- that, or the log, is what to
# watch. A paper whose download failed does not count as a failure: it is
# reported in the log and in `make status`, and tried again on later runs.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PLATO_DATA_DIR:-$HOME/data/plato_data}"
LOG_DIR="$DATA_DIR/logs"
LOG="$LOG_DIR/update-$(date +%Y-%m).log"
mkdir -p "$LOG_DIR"

# Homebrew (LaTeXML, Ghostscript) and /usr/local/bin (pandoc).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
# marker is installed in a Python of its own on the development Mac; elsewhere
# set PLATO_MARKER_PYTHON in the crontab line, or put marker_single on the PATH.
MAC_MARKER=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
if [ -z "${PLATO_MARKER_PYTHON:-}" ] && [ -x "$MAC_MARKER" ]; then
    export PLATO_MARKER_PYTHON="$MAC_MARKER"
fi

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') update starting"
    make -C "$REPO" update
    status=$?
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') exit $status"
} >>"$LOG" 2>&1
exit "$status"
