# TreeLoss repo

Project structure:

1. `requirements.txt` a list of all python libraries needed to run this code; they can be installed with the command 
   ```
   $ pip3 install -r requirements.txt
   ```
   and that is the only command that needs to be run before all the code in this repo will work
1. `/TreeLoss` contains all of the generic library files that can be used in any project
1. `/paper` contains the latex source for the paper
1. `/experiments` contains the code for reproducing the experiments from the paper
    1. each experiment will have several files, which should be located in its own subfolder
