# XLing-Eval
Code and resources for inducing and evaluating cross-lingual embedding spaces

This repository accompanies the following ACL 2019 publication: 

Goran Glavaš, Robert Litschko, Sebastian Ruder and Ivan Vulić. How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), pages 710-721, Florence, 2019. 

If you are using the BLI datasets and/or the code in your work, please cite the above paper. Here's a Bibtex entry: 
```
@inproceedings{glavas-etal-2019-properly,
    title = "How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions",
    author = "Glava{\v{s}}, Goran  and
      Litschko, Robert  and
      Ruder, Sebastian  and
      Vuli{\'c}, Ivan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/P19-1070",
    pages = "710--721"
}
```

## Datasets

Directory "bli_datasets" contains bilingual dictionaries for 28 language pairs. For each of the language pairs, there are 5 dictionary files: 4 training dictionaries of varying sizes (500, 1K, 3K, and 5K translation pairs) and one testing dictionary containing 2K test word pairs. All results reported in the above paper have been obtained on test dictionaries of respective language pairs.

Corresponding monolingual FastText embeddings (cut to first 200K vocabulary entries) for 8 languages involved in our experiments are available for download: 
https://tinyurl.com/y5shy5gt

## Code

We offer code that induces CLWEs with three different methods (included in the comparative evaluation from the paper): 

(1) PROC (by solving the Procrustes problem),
(2) CCA (Cannonical Correlation Analysis), and
(3) PROC-B (our bootstrapping extension of PROC)

In order to induce the mapping, that is, the cross-lingual (bilingual) word embedding space, one must first transform the word embeddings commonly stored in textual format. Serialization of the embeddings is done with the script *code/emb_serializer.py*: it takes in the location of text-formatted embeddings file and produces two files -- pickled vocabulary dictionary and a serialized Numpy array containing all the vectors: 

*emb_serializer.py [-h] [-n TOPN] [-d DIM] <text_embs_path> <vocab_path> <vectors_path>*

Once you've serialized both your source and target monolingual embeddings, you can run the *code/map.py* to induce the bilingual space with one of the three methods (PROC, PROC-B, or CCA): 

*map.py [-h] [-m MODEL] [-d TRANS_DICT] [--lang_src LANG_SRC] [--lang_trg LANG_TRG] <embs_src> <vocab_src> <embs_trg> <vocab_trg> <output>*

The location where the shared space will be stored is to be specified with the argument <output>. The mapping method is specified with the option -m ("p" for PROC, "b" for PROC-B and "c" for CCA; default is "p"). The training dictionary is given with the option -d. The mapping will create four files: 

- "lang_src-lang_trg.lang_src.vectors": contains the vectors of source language words (after their mapping to the shared space)
- "lang_src-lang_trg.lang_src.vocab": contains the vocabulary of the source language space (should always be the same as the input file <vocab_src>)
- "lang_src-lang_trg.lang_trg.vectors": contains the vectors of target language words (for PROC and PROC-B methods, these will be the same as in the input file <embs_trg>, for CCA they will be different)
- "lang_src-lang_trg.lang_trg.vocab": contains the vocabulary of the target language space (should always be the same as the input file <vocab_trg>)

Once the shared embedding space has been induced, you can evaluate its BLI performance using the script *code/eval.py*

*eval.py [-h] <test_set_path> <yacle_embs_src> <yacle_embs_trg> <vocab_src> <vocab_trg>*

The first argument to the eval script is the path to the test set dictionary and the remaining four files are embeddings (presumably already mapped into the same space with *map.py*) and vocabulary files of the two languages.

Finally, if you'd like to convert back the embeddings from the serialized format and store them in the textual file, use the script *code/emb_deserialize.py*:

*emb_deserializer.py [-h] <vocab_path> <vectors_path> <text_embs_path>*
