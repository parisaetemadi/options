#!/bin/zsh
# Rebuild the dataset (after a data fix), redo the year-by-year predictions, rerun the strategy search.
cd /Users/parisaetemadi/Desktop/Projects/options
.venv/bin/python -u -c "import train; train.load_dataset(refresh=True)" > logs/rebuild.log 2>&1 && \
.venv/bin/python -u walkforward.py > logs/walkforward.log 2>&1 && \
.venv/bin/python -u strategy_search.py > logs/strategy_search.log 2>&1
echo "finished $(date)" >> logs/rebuild.log
