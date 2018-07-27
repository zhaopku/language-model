import os
from collections import defaultdict, Counter
import numpy as np
import random
from tqdm import tqdm

class Sample:
    def __init__(self, data, steps, flag_word):
        self.input_ = data[0:steps]
        self.target = data[1:steps+1]
        self.length = 0

        for word in self.input_:
            if word == flag_word:
                break
            self.length += 1

class Batch:
    def __init__(self, samples):
        self.samples = samples
        self.batch_size = len(samples)

class TextData:
    def __init__(self, args):
        self.args = args

        #note: use 20k most frequent words
        self.UNK_WORD = '<unk>'
        self.EOS_WORD = '<eos>'
        self.BOS_WORD = '<bos>'
        self.PAD_WORD = '<pad>'

        # list of batches
        self.train_batches = []
        self.val_batches = []
        self.test_batches = []

        self.word2id = {}
        self.id2word = {}

        self.train_samples = None
        self.valid_samples = None
        self.test_samples = None


        self.train_samples, self.valid_samples, self.test_samples, \
        self.vocab_size, self.word2id, self.id2word = self._create_data()


        self.preTrainedEmbedding = self._build_embeddings()
        # [num_batch, batch_size, maxStep]
        self.train_batches = self._create_batch(self.train_samples)
        self.val_batches = self._create_batch(self.valid_samples)

        # note: test_batches is none here
        self.test_batches = self._create_batch(self.test_samples)


    def getVocabularySize(self):
        assert len(self.word2id) == len(self.id2word)
        return len(self.word2id)


    def _read_embed(self):
        with open(self.args.embedFile, 'r') as file:
            pretrainedEmbeds = dict()
            lines = file.readlines()
            for line in lines[1:]:
                splits = line.split()
                word = splits[0]
                embeddings = [float(s) for s in splits[1:]]
                assert len(embeddings) == self.args.embeddingSize
                pretrainedEmbeds[word] = embeddings
        return pretrainedEmbeds

    def _build_embeddings(self):
        embeddings = []
        pretrainedEmbeds = self._read_embed()
        word2emb = dict()
        print('Building pretrained embeddings')
        for word in tqdm(self.word2id.keys()):
            if word in pretrainedEmbeds.keys():
                word2emb[word] = np.asarray(pretrainedEmbeds[word])
            else:
                word2emb[word] = np.random.uniform(low=-0.25, high=0.25, size=self.args.embeddingSize)

        for id in range(len(self.id2word)):
            word = self.id2word[id]
            emb = word2emb[word]
            embeddings.append(emb)

        embeddings = np.asarray(embeddings)

        return embeddings


    def _create_batch(self, all_samples):
        all_batches = []

        if all_samples is None:
            return all_batches

        num_batch = len(all_samples)//self.args.batchSize + 1
        for i in range(num_batch):
            samples = all_samples[i*self.args.batchSize:(i+1)*self.args.batchSize]
            feed_samples = []
            for sample in samples:
                feed_sample = Sample(data=sample, steps=self.args.maxSteps, flag_word=self.word2id[self.PAD_WORD])
                feed_samples.append(feed_sample)

            if len(feed_samples) == 0:
                continue
            batch = Batch(feed_samples)
            all_batches.append(batch)

        return all_batches


    def _create_data(self):

        train_path = os.path.join(self.args.dataDir, self.args.trainFile)
        valid_path = os.path.join(self.args.dataDir, self.args.valFile)
        test_path = os.path.join(self.args.dataDir, self.args.testFile)

        word_to_id, id_to_word = self._build_vocab(train_path)


        print('mapping training data to ids')
        train_samples = self._file_to_word_ids(train_path, word_to_id)
        print('mapping val data to ids')
        valid_samples = self._file_to_word_ids(valid_path, word_to_id)
        print('mapping test data to ids')
        test_samples = self._file_to_word_ids(test_path, word_to_id)

        vocab_size = len(word_to_id)

        assert len(word_to_id) == len(id_to_word)

        return train_samples, valid_samples, test_samples, vocab_size, word_to_id, id_to_word

    def _read_sents(self, filename):
        with open(filename, 'r') as file:
            all_sents = []
            all_words = []
            lines = file.readlines()
            for idx, line in enumerate(lines):
                words = line.split()
                all_words.extend(words)
                # add BOS and EOS words
                words = [self.BOS_WORD] + words + [self.EOS_WORD]
                if len(words) > self.args.maxSteps:
                    all_sents.append(words[:self.args.maxSteps+1])
                    continue
                while len(words) < self.args.maxSteps+1:
                    words.append(self.PAD_WORD)
                all_sents.append(words)

        return all_sents, all_words

    def _build_vocab(self, filename):
        all_sents, all_words = self._read_sents(filename)

        counter = Counter(all_words)

        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        # keep the most frequent vocabSize words, including the special tokens
        count_pairs = count_pairs[0:self.args.vocabSize-4]
        count_pairs.append((self.UNK_WORD, 100000))
        count_pairs.append((self.PAD_WORD, 100000))
        count_pairs.append((self.BOS_WORD, 100000))
        count_pairs.append((self.EOS_WORD, 100000))

        assert len(count_pairs) == self.args.vocabSize

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        id_to_word = {v: k for k, v in word_to_id.items()}

        return word_to_id, id_to_word

    def _file_to_word_ids(self, filename, word_to_id):
        all_samples = []
        if not os.path.exists(filename):
            return all_samples

        all_sents, all_words = self._read_sents(filename)

        oov_cnt = 0
        word_cnt = 0
        for sent in tqdm(all_sents):
            sample = []
            for word in sent:
                word_cnt += 1
                if word in word_to_id.keys():
                    sample.append(word_to_id[word])
                else:
                    sample.append(word_to_id[self.UNK_WORD])
                    oov_cnt += 1
            all_samples.append(sample)

        print('OOV rate = {} for {}'.format(oov_cnt*1.0/word_cnt, filename))
        return all_samples

    def get_batches(self, tag='train'):
        if tag == 'train':
            random.shuffle(self.train_batches)
            return self.train_batches
        elif tag == 'val':
            return self.val_batches
        else:
            return self.test_batches
