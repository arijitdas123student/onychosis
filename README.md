# nAIl
Onychosis (Nail Disease) classifier using Raspberry Pi 4 and Edge Impulse.

# Instructions to Run the ML Model

Open **Terminal** and type the following commands: 

```$ cp /etc/xdg/lxsession/LXDE-pi/autostart ~/.config/lxsession/LXDE-pi/```

```$ nano ~/.config/lxsession/LXDE-pi/autostart```

Then edit the file by adding the following lines: 

``@lxpanel --profile LXDE-pi
@pcmanfm --desktop --profile LXDE-pi
#@xscreensaver -no-splash
point-rpi
chromium --start https://smartphone.edgeimpulse.com/classifier.html?apiKey=ei_171ba15b5f3a4ea267c48625cad0b11dc093f8dfb1fad7c8a5a0d2470d336ae8``

Then exit, save and do a ```sudo reboot now```.
