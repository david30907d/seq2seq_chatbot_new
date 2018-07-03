# seq2seq_chatbot_new
基于seq2seq模型的简单对话系统的tf实现，具有embedding、attention、beam_search等功能，数据集是Cornell Movie Dialogs

`nohup python3 train.py --batch_size 1  --data_path data/gensim2file.pkl --rnn_size 50 --embedding_size 200 --numEpochs 3000 &`