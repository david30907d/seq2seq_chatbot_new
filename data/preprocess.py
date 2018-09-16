from sklearn.model_selection import train_test_split
from udicOpenData.stopwords import rmsw
from collections import Counter
from itertools import chain
import json, pickle, tqdm

RESERVED_WORDS = 20000
SOGOU = json.load(open('sogou.json', 'r'))
DOCS = [list(rmsw(doc['content'])) for doc in tqdm.tqdm(SOGOU)]
TITLES = [list(rmsw(doc['contenttitle'])) for doc in tqdm.tqdm(SOGOU)]
X_train, X_test, y_train, y_test = train_test_split(DOCS, TITLES, test_size=0.33, random_state=42)
TABLE = {}

def extract_character_vocab(X_train, y_train):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
    set_words = [word for word, count in sorted(Counter(list(chain(*X_train)) + list(chain(*y_train))).items(), key=lambda x: -x[1])[:RESERVED_WORDS]]
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

TABLE['id2word'], TABLE['word2id'] = extract_character_vocab(X_train, y_train)

def dump_dataset(name='training'):
    dataset = {}
    dataset['id2word'] = TABLE['id2word']
    dataset['word2id'] = TABLE['word2id']
    if name == 'training':
        dataset['trainingSamples'] = [([TABLE['word2id'].get(c, 1) for c in context], [TABLE['word2id'].get(t, 1) for t in title]) for context, title in zip(X_train, y_train)]
    else:
        TABLE['trainingSamples'] = [([TABLE['word2id'].get(c, 1) for c in context], [TABLE['word2id'].get(t, 1) for t in title]) for context, title in zip(X_test, y_test)]       
    pickle.dump(dataset, open('{}.pkl'.format(name), 'wb'))

dump_dataset('training')
dump_dataset('testing')