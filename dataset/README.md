# How to generate the datasets

**CIFAR10/CIFAR100**. These datasets can be downloaded with the help of the built-in 
functions in Pytorch. In the same way, you can easily add MNIST, EMNIST, and SVHN datasets.

**Shakespeare**. The codes to generate Shakespeare dataset refers to 
[FedEM](https://github.com/omarfoq/FedEM). You can run the following codes to generate it.
```
# For Linux 
cd ./shakespeare
bash get_data.sh
python generate_data.py
```

**tinyShakespeare**. This dataset is the same as the dataset used in [char-rnn](https://github.com/karpathy/char-rnn).
We have downloaded it in the repository, and you just need to run the code `generate_data.py` to 
generate the post-processed data for the natural language task.

**Wikitext2**. The original address (https://s3.amazonaws.com/research.metamind.io/wikitext/) to get the WikiText2 
dataset is out of work, and you can download it from this [link](https://github.com/Snail1502/dataset_d2l). 
Once you have download this dataset successfully, put the ".zip" file to the `raw` folder, then run the code
`download_wikitext2.py`.

**StackOverflow**: you can run the following codes to generate StackOverflow dataset. However, the repository 
does not support model training on this dataset currently.
```
# For Linux 
cd ./stackoverflow
bash downlaod_stackoverflow.sh
python processed.py # convert the *.h5 files to *.pt files
```