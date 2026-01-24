#! /bin/bash

###############################################################################
# 1.add Additional startup programs
# start_rosmaster_app
# bash /home/pi/Rosmaster/rosmaster/start_app.sh
# start app program
# 
# 2.add Additional startup programs
# Resolution
# xrandr --fb 1024x600
# set Resolution when no screen
###############################################################################


sleep 5

sudo python3 /home/pi/Rosmaster/rosmaster/rosmaster_main.py

wait
exit 0
