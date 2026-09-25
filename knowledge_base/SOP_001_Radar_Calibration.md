# SOP 001: Radar System Calibration
## Overview
This document outlines the monthly calibration of the Mark-IV Navigation Radar.
## Procedure
1. Power down the primary transmitter array.
2. Connect the diagnostic terminal via the RS-232 port.
3. Run the command `RDR_CAL_INIT`. The baseline voltage must read between 4.5V and 4.8V. 
If voltage exceeds 5.0V, initiate emergency shutdown and replace the capacitator board.
4. Reboot the system and verify target acquisition at 10 nautical miles.