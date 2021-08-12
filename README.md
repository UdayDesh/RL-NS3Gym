# RL-NS3Gym
-Reinforced Learning for NS3 in Cognitive Radio spectrum selection

-Spectrum Hole sensing/prediction is paramount importance to Secondary Users in a CR environment

-Problem of radio channel selection in a wireless multi-channel environment, e.g. 802.11 networks with external interference. 

-RL optimisation algorithms can help predict the occurences of Spectrum White Space so that Channel can be effectively utilized

-The objective of the agent is to select for the next time slot a channel free of interference considering a basic example where the external interference follows a periodic pattern, i.e. time-slice sweeping over all available channels

## Basic Setup
WSL Ubuntu 18.04

NS3 3.17

Python3.6

GCC

TensorFlow 2.0

## Python Libraries

Keras

openpyxl


![image](https://user-images.githubusercontent.com/87072503/124752361-62e75f80-df45-11eb-8bb1-91c54bd3fa0d.png)

Copy folder under NS3-Gym / Scratch folder for execution

### Window 1:
 $ cd $NS3-GYM
 
 $ ./waf --run "interference-pattern"
 
### Window 2:
 $ cd $NS3-GYM
 
 $ cd scratch/interference-pattern
 
 $ ./cognitive-agent-v3.py
 
