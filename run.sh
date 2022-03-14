#! /bin/sh

python setup.py install
python examples/analytic.py oracle --max_episodes=2000 --seed=1336 --tier=3
