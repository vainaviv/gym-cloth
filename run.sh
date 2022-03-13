#! /bin/sh

python setup.py install
python examples/analytic.py behavior_cloning --max_episodes=10 --seed=1336 --tier=3
