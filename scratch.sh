#!/bin/bash

#SBATCH -c 4
#SBATCH --mem 120G
#SBATCH --time 24:00:00
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --qos scavenger

apptainer pull ruler.sif docker://cphsieh/ruler:0.1.0
