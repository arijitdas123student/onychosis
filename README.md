# nAIl
Onychosis (Nail Disease) classifier using Raspberry Pi 4 and Edge Impulse.

# Instructions to Run the ML Model

Copy **API Key** from Edge Impulse Studio **Dashboard** page and save it somewhere you can access in the future.

Open **Terminal** and type the following commands: 

```$ cp /etc/xdg/lxsession/LXDE-pi/autostart ~/.config/lxsession/LXDE-pi/```

```$ nano ~/.config/lxsession/LXDE-pi/autostart```

Then edit the file by adding the following lines: 

``@lxpanel --profile LXDE-pi
@pcmanfm --desktop --profile LXDE-pi
#@xscreensaver -no-splash
point-rpi
chromium --start https://smartphone.edgeimpulse.com/classifier.html?apiKey=[PUT YOUR API KEY HERE]``

Then exit, save and do a ```sudo reboot now```.
