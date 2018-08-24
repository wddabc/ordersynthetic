# Synthetic Data Made to Order: The Case of Parsing 

Implementation of the paper "Synthetic Data Made to Order: The Case of Parsing" by Dingquan Wang and Jason Eisner. EMNLP 2018


### Requirements 

* Python3 
* PyTorch 0.3


### Run 

* To permute a source treebank `src.conllu` in UD format towards a target language with only POS-tags `tgt.txt` in POS-spaced format (see `data/fr.txt`) and output to `src~tgt.conllu`:

         python src/main.py --src src.conllu --tgt tgt.txt --output src~tgt.conllu

  For convenience, `--tgt` could also take UD format input (endswith `.conllu`) which will ignore everything but the POS-tags.

* To pre-train a self-permutation model and use it as initialization: 

         python src/main.py --task self_model  --src src.conllu --model $(pwd)/pre_model.pkl
         python src/main.py --src src.conllu --tgt tgt.txt --output src~tgt.conllu --pretrain $(pwd)/pre_model.pkl

* For more options, please use:

        python src/main.py --help

    the default hyperparameters are the ones used in the original paper
    
* We also release a sample data in `data`  and a script `permute.sh`

### Reference 
```latex
@inproceedings{wang-eisner-2018-emnlp,
  author =      {Dingquan Wang and Jason Eisner},
  title =       {Synthetic Data Made to Order: The Case of Parsing},
  booktitle =   {Proceedings of the Conference on Empirical Methods in
                 Natural Language Processing (EMNLP)},
  year =        {2018},
  month =       nov,
  address =     {Brussels},
  url =         {http://cs.jhu.edu/~jason/papers/#wang-eisner-2018-emnlp}
}
