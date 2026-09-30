#!/bin/zsh
# Started by macOS (LaunchAgent com.options.databento) at login and every 30 minutes.
# For each ticker group in turn: continue its Databento download; once it's complete,
# repair flagged days once. When every group is done, it does nothing.
cd /Users/parisaetemadi/Desktop/Projects/options || exit 1
exec >> logs/databento.log 2>&1
D=data/databento
if pgrep -f "python -u download_databento.py" > /dev/null; then exit 0; fi   # already running
for G in core extra more; do
  [ -f $D/COMPLETE_$G ] && [ -f $D/REPAIRED_$G ] && continue
  echo "=== $G: started by macOS $(date)"
  /usr/bin/caffeinate -i .venv/bin/python -u download_databento.py --group $G
  if [ -f $D/COMPLETE_$G ] && [ ! -f $D/REPAIRED_$G ]; then
    echo "=== $G: repairing flagged days $(date)"
    /usr/bin/caffeinate -i .venv/bin/python -u repair_databento.py --group $G && date > $D/REPAIRED_$G
  fi
  [ -f $D/REPAIRED_$G ] || exit 0          # finish this group before starting the next
done
