Code was developed in the following order. Since format and functionality is constantly changing, it might be good to keep track of this. Some major changes are included.

1. Constant-GM_Current_Reference (Legacy folder)
1. OmarTesting
1. NPN_LVR
1. PMOS_LDO
    - Reorganized. Headers for navigation
    - Actual run vs dry run in same function, determined by argument
1. PMOS_LDO_V2
    - Plotting now done from saved csv files. Can now replot experimental data afterwards
    - Procedure split into individual tests, which can easily be selected or unselected
    - Functions that don't need to be visible are moved into a background file named ICASL.py
1. PMOS_LDO_V3 (untested)
    - Experimental test branch
    - Features not incorporated into future code
        - Used ICASL.py functions that dynamically read hierarchically saved data into a tree of dictionaries
        - Plotting code could be applied to either cadence sims or experiments
        - Some function arguments were made into global constants
    - Missing functionality from future code
1. Constant-Gm_Current_Reference (untested)
1. PTAT_Current_Reference
    - Sweep and trigger settings organized into functions by "leader" and "follower"
    - "Continue Experiment" functionality added
1. Constant-IC_Current_Reference
1. TI-RLDO (just sims)
    - Plotted sims with ICASL.py



