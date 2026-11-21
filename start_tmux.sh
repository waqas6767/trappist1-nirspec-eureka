#!/bin/bash
# Start (or resume) run_reduction.py in a tmux window, so the reduction keeps
# running after you log out of the server.
#
# Usage:  bash start_tmux.sh [options for run_reduction.py]
#   e.g.  bash start_tmux.sh --visits jw01331001001 --s1-ncpu 4 --s3-ncpu 8   # one test visit
#         bash start_tmux.sh --jobs 6 --s1-ncpu 4 --s3-ncpu 6 --offline       # all visits
#
# Settings (environment variables, all optional):
#   WORKDIR   folder that contains jwst_data/ and receives reduction/
#             (default: the folder one level above this repository)
#   PY        Python interpreter with Eureka! installed
#             (default: $WORKDIR/envs/eureka/bin/python if it exists, else python)
#   SESSION   tmux session name (default: eureka)
# If $WORKDIR/env.sh exists it is sourced first (CRDS settings, proxy, ...).
# Finished visits are skipped, so running this script again resumes the reduction.

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=${WORKDIR:-$(dirname "$REPO")}
SESSION=${SESSION:-eureka}
if [ -z "$PY" ]; then
    if [ -x "$WORKDIR/envs/eureka/bin/python" ]; then PY=$WORKDIR/envs/eureka/bin/python; else PY=python; fi
fi
ENVSRC=""
[ -f "$WORKDIR/env.sh" ] && ENVSRC="source $WORKDIR/env.sh;"

mkdir -p "$WORKDIR/reduction/logs"
LOG=$WORKDIR/reduction/logs/main_$(date +%Y%m%d_%H%M).log
CMD="$ENVSRC cd $REPO; $PY run_reduction.py --drive $WORKDIR $* 2>&1 | tee $LOG; echo; echo 'Reduction finished - press Enter to close'; read"

if [ -n "$TMUX" ]; then
    tmux new-window -n reduce "$CMD"
else
    tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION"
    tmux new-window -t "$SESSION" -n reduce "$CMD"
fi
echo "Started in tmux window 'reduce'. Main log: $LOG"
echo "Progress:  $PY $REPO/run_reduction.py --drive $WORKDIR --status"
