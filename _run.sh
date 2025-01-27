#!/bin/bash
virtualenv -p python3 venv
source venv/bin/activate

#requirements
python3 -m pip install -r requirements.txt
python3 -m pip install tensorflow==2.11.0

#set pythonpath
export PYTHONPATH=$PYTHONPATH:/home/lucyh/Documents/ACL/open_spiel_acl
export PYTHONPATH=$PYTHONPATH:/home/lucyh/Documents/ACL/open_spiel_acl/build/python

#run code
python3 open_spiel/python/examples/psro_v2_example.py --game_name="kuhn_poker" --n_players=2 --meta_strategy_method="alpharank" --gpsro_iterations=10

