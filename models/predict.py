import tensorflow as tf
import argparse
from models import utils
from models.textData import TextData
from models.model import Model
from models.model_generate import ModelG
import os
import pickle as p
from tqdm import tqdm
import numpy as np

class Predict:

    def __init__(self):
        self.args = None


        self.textData = None
        self.model = None
        self.outFile = None
        self.sess = None
        self.saver = None
        self.model_name = None
        self.model_path = None
        self.globalStep = 0
        self.summaryDir = None

        self.summaryWriter = None
        self.mergedSummary = None

    @staticmethod
    def parse_args(args):

        parser = argparse.ArgumentParser()

        parser.add_argument('--resultDir', type=str, default='result', help='result directory')
        # data location
        dataArgs = parser.add_argument_group('Dataset options')

        dataArgs.add_argument('--summaryDir', type=str, default='./summaries')
        dataArgs.add_argument('--dataDir', type=str, default='data', help='dataset directory, save pkl here')
        dataArgs.add_argument('--datasetName', type=str, default='dataset', help='a TextData object')
        dataArgs.add_argument('--trainFile', type=str, default='sentences.train')

        # use val file for generation task
        dataArgs.add_argument('--valFile', type=str, default='sentences.continuation')
        dataArgs.add_argument('--testFile', type=str, default='sentences.eval')
        dataArgs.add_argument('--embedFile', type=str, default='wordembeddings-dim100.word2vec')
        dataArgs.add_argument('--doTest', action='store_true')
        dataArgs.add_argument('--vocabSize', type=int, default=20000, help='vocab size, use the most frequent words')
        # neural network options
        nnArgs = parser.add_argument_group('Network options')
        nnArgs.add_argument('--embeddingSize', type=int, default=100)
        nnArgs.add_argument('--hiddenSize', type=int, default=512, help='hiddenSize for RNN sentence encoder')
        nnArgs.add_argument('--oriSize', type=int, default=512)
        nnArgs.add_argument('--rnnLayers', type=int, default=1)
        nnArgs.add_argument('--maxSteps', type=int, default=30)
        nnArgs.add_argument('--project', action='store_true')
        # training options
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--modelPath', type=str, default='saved')
        trainingArgs.add_argument('--preEmbedding', action='store_true')
        trainingArgs.add_argument('--dropOut', type=float, default=0.8, help='dropout rate for RNN (keep prob)')
        trainingArgs.add_argument('--learningRate', type=float, default=0.001, help='learning rate')
        trainingArgs.add_argument('--batchSize', type=int, default=64, help='batch size')
        # max_grad_norm
        trainingArgs.add_argument('--maxGradNorm', type=int, default=5)
        ## do not add dropOut in the test mode!
        trainingArgs.add_argument('--test', action='store_true', help='if in test mode')
        trainingArgs.add_argument('--epochs', type=int, default=100, help='most training epochs')
        trainingArgs.add_argument('--device', type=str, default='/gpu:0', help='use the first GPU as default')
        trainingArgs.add_argument('--loadModel', action='store_true', help='whether or not to use old models')
        trainingArgs.add_argument('--testModel', action='store_true', help='do not train, only test')
        trainingArgs.add_argument('--generate', action='store_true', help='for task 2, generate sentences greedily')
        # note: we can set this number larger than 20, then we truncated it when handing in
        trainingArgs.add_argument('--maxGenerateLength', type=int, default=25, help='maximum length when generating sentences')
        trainingArgs.add_argument('--writePerplexity', action='store_true')
        return parser.parse_args(args)

    def main(self, args=None):
        print('Tensorflow version {}'.format(tf.VERSION))

        # initialize args
        self.args = self.parse_args(args)


        self.outFile = utils.constructFileName(self.args, prefix=self.args.resultDir)
        self.args.datasetName = utils.constructFileName(self.args, prefix=self.args.dataDir)
        datasetFileName = os.path.join(self.args.dataDir, self.args.datasetName)


        if not os.path.exists(datasetFileName):
            self.textData = TextData(self.args)
            with open(datasetFileName, 'wb') as datasetFile:
                p.dump(self.textData, datasetFile)
            print('dataset created and saved to {}'.format(datasetFileName))
        else:
            with open(datasetFileName, 'rb') as datasetFile:
                self.textData = p.load(datasetFile)
            print('dataset loaded from {}'.format(datasetFileName))

        sessConfig = tf.ConfigProto(allow_soft_placement=True)
        sessConfig.gpu_options.allow_growth = True

        self.model_path = utils.constructFileName(self.args, prefix=self.args.modelPath, tag='model')
        self.model_name = os.path.join(self.model_path, 'model')

        self.sess = tf.Session(config=sessConfig)
        # summary writer
        self.summaryDir = utils.constructFileName(self.args, prefix=self.args.summaryDir)


        with tf.device(self.args.device):
            if not self.args.generate:
                self.model = Model(self.args, self.textData)
            else:
                print('Creating model for generation')
                self.model = ModelG(self.args, self.textData)
            params = tf.trainable_variables()
            print('Model created')

            # saver can only be created after we have the model
            self.saver = tf.train.Saver()

            self.summaryWriter = tf.summary.FileWriter(self.summaryDir, self.sess.graph)
            self.mergedSummary = tf.summary.merge_all()

            if self.args.loadModel:
                # load model from disk
                if not os.path.exists(self.model_path):
                    print('model does not exist on disk!')
                    print(self.model_path)
                    exit(-1)

                self.saver.restore(sess=self.sess, save_path=self.model_name)
                print('Variables loaded from disk {}'.format(self.model_name))
            else:
                init = tf.global_variables_initializer()
                # initialize all global variables
                self.sess.run(init)
                print('All variables initialized')

            if not self.args.testModel and not self.args.generate and not self.args.writePerplexity:
                self.train(self.sess)
            elif self.args.writePerplexity:
                self.test(self.sess, tag='test')
            elif self.args.generate:
                self.generate(self.sess)
            else:
                self.test_model()



    def generate(self, sess):
        print('generating!')
        batches = self.textData.get_batches(tag='val')
        all_sents = []
        for nextBatch in tqdm(batches):
            # set dummy initial values for predictions
            predictions = np.zeros(shape=(nextBatch.batch_size), dtype=np.int32)
            sents = []
            for i in range(nextBatch.batch_size):
                sents.append([self.textData.BOS_WORD])

            for time_step in range(self.args.maxGenerateLength):
                ops, feed_dict, sents = self.model.step(nextBatch, predictions=predictions, time_step=time_step, sents=sents)
                predictions = sess.run(ops, feed_dict)

            all_sents.extend(sents)

        with open('generated.txt', 'w') as file:
            for sent in all_sents:
                sentence = ' '.join(sent)
                file.write(sentence+'\n')


    def train(self, sess):
        print('Start training')

        out = open(self.outFile, 'w', 1)
        out.write(self.outFile + '\n')
        utils.writeInfo(out, self.args)

        current_val_loss = np.inf

        for e in range(self.args.epochs):
            # training
            trainBatches = self.textData.get_batches(tag='train')
            totalTrainLoss = 0.0

            # cnt of batches
            cnt = 0

            total_steps = 0
            for nextBatch in tqdm(trainBatches):
                cnt += 1
                self.globalStep += 1

                for sample in nextBatch.samples:
                    total_steps += sample.length

                ops, feed_dict = self.model.step(nextBatch, test=False)

                _, loss, trainPerplexity = sess.run(ops, feed_dict)

                totalTrainLoss += loss

                # average across samples in this step
                trainPerplexity = np.mean(trainPerplexity)
                self.summaryWriter.add_summary(utils.makeSummary({"trainLoss": loss}), self.globalStep)
                self.summaryWriter.add_summary(utils.makeSummary({"trainPerplexity": trainPerplexity}), self.globalStep)

            # compute perplexity over all samples in an epoch
            trainPerplexity = np.exp(totalTrainLoss/total_steps)

            print('\nepoch = {}, Train, loss = {}, perplexity = {}'.
                  format(e, totalTrainLoss, trainPerplexity))
            out.write('\nepoch = {}, loss = {}, perplexity = {}\n'.
                  format(e, totalTrainLoss, trainPerplexity))
            out.flush()

            valLoss, val_num = self.test(sess, tag='val')

            testLoss, test_num = self.test(sess, tag='test')

            valPerplexity = np.exp(valLoss/val_num)
            testPerplexity = np.exp(testLoss/test_num)

            print('Val, loss = {}, perplexity = {}'.
                  format(valLoss, valPerplexity))
            out.write('Val, loss = {}, perplexity = {}\n'.
                  format(valLoss, valPerplexity))

            print('Test, loss = {}, perplexity = {}'.
                  format(testLoss, testPerplexity))
            out.write('Test, loss = {}, perplexity = {}\n'.
                  format(testLoss, testPerplexity))

            # we do not use cross val currently, just train, then evaluate
            #if True:
            if valLoss < current_val_loss:
                current_val_loss = valLoss
                print('New val loss {} at epoch {}'.format(valLoss, e))
                out.write('New val loss {} at epoch {}\n'.format(valLoss, e))
                save_path = self.saver.save(sess, save_path=self.model_name)
                print('model saved at {}'.format(save_path))
                out.write('model saved at {}\n'.format(save_path))

            out.flush()
        out.close()


    def write_perplexity(self, batch, perplexity):
        assert len(batch.samples) == len(perplexity)
        input_ = []
        for sample in batch.samples:
            sent = []
            for word_id in sample.input_:
                word = self.textData.id2word[word_id]
                sent.append(word)
                if word == self.textData.EOS_WORD:
                    break
            sent = ' '.join(sent).strip()
            input_.append(sent)
        with open('perplexity.txt', 'a') as file:
            for idx, sent in enumerate(input_):
                file.write(sent+'\t'+str(perplexity[idx])+'\n')

    def test(self, sess, tag = 'val'):
        if tag == 'val':
            print('Validating\n')
            batches = self.textData.val_batches
        else:
            print('Testing\n')
            batches = self.textData.test_batches

        cnt = 0

        total_loss = 0.0
        total_steps = 0
        for idx, nextBatch in enumerate(tqdm(batches)):
            cnt += 1
            ops, feed_dict = self.model.step(nextBatch, test=True)
            loss, perplexity = sess.run(ops, feed_dict)

            total_loss += loss
            for sample in nextBatch.samples:
                total_steps += sample.length

            if self.args.writePerplexity:
                self.write_perplexity(nextBatch, perplexity)
        return total_loss, total_steps


    def test_model(self):
        # TODO: placeholder, this function is useless in this implementation, ignore
        pass