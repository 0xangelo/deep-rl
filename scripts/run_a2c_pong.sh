#!/bin/sh
python -m proj.run a2c --log_dir data/ --exp_name a2c_pong --env PongNoFrameskip-v4 --policy:class CNNWeightSharingAC --log_interval 10 --latest_only False --save_interval 100 --format_strs stdout,csv --datestamp --seed 234
