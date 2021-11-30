#!/bin/bash
#PBS -N BERT-FINE-TUNE
#PBS -q gpu
#PBS -l select=1:ncpus=32:ngpus=1:mem=64gb:cpu_flag=avx512dq:scratch_ssd=20gb:gpu_cap=cuda75:cl_adan=True
#PBS -l walltime=3:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris

cd $SCRATCHDIR
git clone https://github.com/LadaOndris/text_classification
PROJDIR="$SCRATCHDIR/text_classification"
cd $PROJDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $PROJDIR" >> $DATADIR/jobs_info.txt

module add conda-modules-py37
conda env remove -n zpja
conda env create -f $PROJDIR/env.yml
conda activate zpja

export PYTHONPATH="${PYTHONPATH}:$PROJDIR"
python3 $PROJDIR/src/training/trainer.py --verbose 0 --batch-size 32

mkdir $DATADIR/zpja/
cp -r "$PROJDIR/models" "$DATADIR/zpja/" || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch