#!/bin/bash
#PBS -N BERT-FINE-TUNE
#PBS -q gpu
#PBS -l select=1:ncpus=32:ngpus=1:mem=64gb:cpu_flag=avx512dq:scratch_ssd=50gb:gpu_cap=cuda75:cl_adan=True
#PBS -l walltime=10:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris/ZPJa
SCRATCHDIR="$SCRATCHDIR/ZPJa"
mkdir $SCRATCHDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add conda-modules-py37
conda env remove -n zpja
conda env create -f environment.yml
conda activate zpja

cp -r "$DATADIR/src" "$SCRATCHDIR/" || { echo >&2 "Couldnt copy srcdir to scratchdir."; exit 2; }

export PYTHONPATH="${PYTHONPATH}:$SCRATCHDIR"
python3 $SCRATCHDIR/src/training/trainer.py --verbose 0 --batch-size 32

cp -r "$SCRATCHDIR/models" "$DATADIR/" || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch