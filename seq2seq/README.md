# Sequence To Sequence for Topic Recommendation

This project is forked from <https://github.com/lc222/seq2seq_chatbot_new>

## Install

1. `pip install -r requirements.txt`
2. `apt install -y opencc wget`

## Preprocessing

Decide using which dataset to train seq2seq.
* dcard: `python3 data_helpers.py -d dcard -f <filepath of word2vec model> -rest <true/false>`
* sogou: `python3 data_helpers.py -d sogou -f <filepath of word2vec model> -rest <true/false>`

## Run

`nohup python3 train.py --batch_size 1  --data_path data/gensim2file.pkl --rnn_size 50 --embedding_size 200 --numEpochs 3000 &`

## Demo

## Evaluation