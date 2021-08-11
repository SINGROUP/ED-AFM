#!/bin/bash

python sims.py
python ptcda.py
python bcb.py
python bcb_surface_sim.py
pdflatex model_schem.tex
python stats.py
python esmap_sample.py
pdflatex esmap_schem.tex
python afm_stacks.py
python sims_hartree.py
python ptcda_surface_sim.py
python single_tip.py
python sims_Cl.py
python height_dependence.py
python extra_electron.py
python background_gradient.py