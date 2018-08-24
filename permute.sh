#!/bin/bash -l

# Pretrain the parameter with the self-permutation
python src/main.py --task self_model  --src data/en.conllu --model $(pwd)/self_model.pkl

# Using the previous pretrained model to train & permute the source language towards the target language 
python src/main.py --src data/en.conllu --tgt data/fr.txt --output en~fr.conllu --pretrain $(pwd)/self_model.pkl
