#!/bin/zsh
# Started by macOS (LaunchAgent com.options.databento) at login and every 30 minutes.
# Continues the Databento download; once it's complete, repairs flagged days once, then does nothing.
cd /Users/parisaetemadi/Desktop/Projects/options || exit 1
exec >> logs/databento.log 2>&1
D=data/databento
[ -f $D/COMPLETE ] && [ -f $D/REPAIRED ] && exit 0                     # all done
if pgrep -f "python -u download_databento.py" > /dev/null; then exit 0; fi   # already running
echo "=== started by macOS $(date)"
/usr/bin/caffeinate -i .venv/bin/python -u download_databento.py
if [ -f $D/COMPLETE ] && [ ! -f $D/REPAIRED ]; then
  echo "=== repairing flagged days $(date)"
  /usr/bin/caffeinate -i .venv/bin/python -u repair_databento.py && date > $D/REPAIRED
fi
