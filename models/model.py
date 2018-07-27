import tensorflow as tf

class Model:
    def __init__(self, args, textData):
        self.args = args
        self.textData = textData

        self.dropOutRate = None
        self.initial_state = None
        self.learning_rate = None
        self.new_learning_rate = None
        self.lr_update = None
        self.loss = None
        self.optOp = None

        self.input = None
        self.target = None
        self.length = None
        self.embedded = None
        self.predictions = None
        self.batch_size = None
        self.output_probs = None
        self.perplexities = None
        self.buildNetwork()

    def buildNetwork(self):
        with tf.name_scope('rnn'):
            # [batchSize, maxSteps, hiddenSize]
            outputs = self.buildRNN()

        if self.args.project:
            print('Using projection over LSTM outputs!')
            with tf.name_scope('project'):
                weights_project = tf.get_variable(shape=[self.args.hiddenSize, self.args.oriSize],
                                        initializer=tf.contrib.layers.xavier_initializer(), name='weights_project')

                outputs = tf.matmul(tf.reshape(outputs, [-1, self.args.hiddenSize]), weights_project)

                outputs = tf.reshape(outputs, [-1, self.args.maxSteps, self.args.hiddenSize])
            with tf.name_scope('output'):

                # [hiddenSize, vocab_size]
                weights = tf.get_variable(shape=[self.args.oriSize, self.textData.getVocabularySize()],
                                        initializer=tf.contrib.layers.xavier_initializer(), name='weights')
                # [vocab_size]
                biases = tf.get_variable(shape=[self.textData.getVocabularySize()],
                                         initializer=tf.contrib.layers.xavier_initializer(), name='biases')

                outputs_reshape = tf.reshape(outputs, [-1, self.args.oriSize], name='outputs_reshape')

                # [batchSize*maxSteps, vocab_size]
                logits = tf.nn.xw_plus_b(outputs_reshape, weights=weights, biases=biases)

                logits = tf.reshape(logits, [-1, self.args.maxSteps, self.textData.getVocabularySize()],
                                    name='logits')
        else:
            with tf.name_scope('output'):

                # [hiddenSize, vocab_size]
                weights = tf.get_variable(shape=[self.args.hiddenSize, self.textData.getVocabularySize()],
                                        initializer=tf.contrib.layers.xavier_initializer(), name='weights')
                # [vocab_size]
                biases = tf.get_variable(shape=[self.textData.getVocabularySize()],
                                         initializer=tf.contrib.layers.xavier_initializer(), name='biases')

                outputs_reshape = tf.reshape(outputs, [-1, self.args.hiddenSize], name='outputs_reshape')

                # [batchSize*maxSteps, vocab_size]
                logits = tf.nn.xw_plus_b(outputs_reshape, weights=weights, biases=biases)
                # [batchSize*maxSteps, vocab_size]
                logits = tf.reshape(logits, [-1, self.args.maxSteps, self.textData.getVocabularySize()],
                                    name='logits')


        with tf.name_scope('loss'):
            # [batchSize, maxSteps]
            mask = tf.sequence_mask(lengths=self.length, maxlen=self.args.maxSteps, name='mask', dtype=tf.float32)
            # [batchSize, maxSteps]
            unmasked_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target, name='unmasked_loss')
            # [batchSize, maxSteps]
            # note: here we have log_e probs, not log_2
            loss_ = tf.multiply(unmasked_loss, mask, name='masked_loss')

            # [batchSize], average across valid time steps
            self.perplexities = tf.exp(tf.divide(tf.reduce_sum(loss_, axis=1), tf.cast(self.length, tf.float32)), name='perplexities')
            # note: perplexity should be computed in the train() function
            self.loss = tf.reduce_sum(loss_, name='loss')


        with tf.name_scope('backpropagation'):
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            gradients, _ = tf.clip_by_global_norm(gradients, self.args.maxGradNorm)

            opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
                                               epsilon=1e-08)
            self.optOp = opt.apply_gradients(zip(gradients, trainable_params))

    def buildRNN(self):
        with tf.name_scope('placeholders'):
            input_shape = [None, self.args.maxSteps]
            self.input = tf.placeholder(tf.int32, shape=input_shape, name='input')
            self.target = tf.placeholder(tf.int32, shape=input_shape, name='target')
            self.length = tf.placeholder(tf.int32, shape=[None,], name='length')
            self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

            self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')

        with tf.name_scope('embedding_layer'):
            if not self.args.preEmbedding:
                embeddings = tf.get_variable(
                    shape=[self.textData.getVocabularySize(), self.args.embeddingSize],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    name='embeddings')
            else:
                print('Using pretrained word embeddings!')
                embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding', dtype=tf.float32)

            # [batchSize, maxSteps, embeddingSize]
            self.embedded = tf.nn.embedding_lookup(embeddings, self.input)
            self.embedded = tf.nn.dropout(self.embedded, self.dropOutRate, name='embedding_dropout')
        with tf.name_scope('lstm'):
            with tf.variable_scope('cell', reuse=False):

                def get_cell(hiddenSize, dropOutRate):
                    cell = tf.contrib.rnn.LSTMCell(num_units=hiddenSize, state_is_tuple=True,
                                                   initializer=tf.contrib.layers.xavier_initializer())
                    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
                                                             output_keep_prob=dropOutRate)
                    return cell

                # https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn
                multiCell = []
                for i in range(self.args.rnnLayers):
                    multiCell.append(get_cell(self.args.hiddenSize, self.dropOutRate))
                multiCell = tf.contrib.rnn.MultiRNNCell(multiCell, state_is_tuple=True)

            self.initial_state = multiCell.zero_state(self.batch_size, dtype=tf.float32)
            # [batchSize, maxSteps, hiddenSize]
            state = self.initial_state
            outputs = []
            #TODO: remember the length of each sentence
            with tf.variable_scope("loop", reuse=tf.AUTO_REUSE):
                for time_step in range(self.args.maxSteps):
                    # [batch_size, hidden_size]
                    (cell_output, state) = multiCell(self.embedded[:, time_step, :], state)
                    outputs.append(cell_output)

            # [maxSteps, batchSize, hiddenSize]
            outputs = tf.stack(outputs)
            # [batchSize, maxSteps, hiddenSize]
            outputs = tf.transpose(outputs, [1, 0, 2], name='outputs')

        return outputs

    def step(self, batch, test=False):
        feed_dict = {}

        # [batchSize, maxSteps]
        input_ = []
        target = []
        length = []

        for sample in batch.samples:
            input_.append(sample.input_)
            target.append(sample.target)
            length.append(sample.length)

        feed_dict[self.input] = input_
        feed_dict[self.target] = target
        feed_dict[self.length] = length
        feed_dict[self.batch_size] = len(length)

        if not test:
            feed_dict[self.dropOutRate] = self.args.dropOut
            ops = (self.optOp, self.loss, self.perplexities)
        else:
            # during test, do not use drop out!!!!
            feed_dict[self.dropOutRate] = 1.0
            ops = (self.loss, self.perplexities)

        return ops, feed_dict
