#!/bin/bash # COMMENT:The interpreter used to execute the script

#SBATCH --job-name=temptraq_INA_classify
#SBATCH --mail-user=lseungwo@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --cpus-per-task=50
#SBATCH --ntasks-per-node=1

#SBATCH --account=mtewari0
#SBATCH --partition=standard
#SBATCH --output=/nfs/turbo/umms-mtewari-sen/TempTraq_ClassifyINA/log/%u/%j.log
# COMMENT:The application(s) to execute along with its input arguments and options: