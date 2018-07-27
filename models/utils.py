import os
import tensorflow as tf

def makeSummary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])


def constructFileName(args, prefix=None, tag=None):

    if prefix == args.dataDir:
        file_name = ''
        file_name += args.dataDir + '-'
        file_name += str(args.vocabSize) + '-'
        file_name += str(args.batchSize) + '-'
        file_name += str(args.maxSteps) + '.pkl'
        return file_name

    file_name = ''
    file_name += 'embeddingSize_' + str(args.embeddingSize)
    file_name += '_hiddenSize_' + str(args.hiddenSize)
    file_name += '_rnnLayers_' + str(args.rnnLayers)
    file_name += '_maxSteps_' + str(args.maxSteps)
    file_name += '_dropOut_' + str(args.dropOut)

    file_name += '_learningRate_' + str(args.learningRate)
    file_name += '_batchSize_' + str(args.batchSize)
    file_name += '_vocabSize_' + str(args.vocabSize)
    file_name += '_project_' + str(args.project)
    file_name += '_preEmbedding_' + str(args.preEmbedding)
    #file_name += '_epochs_' + str(args.epochs)
    if tag != 'model':
        file_name += '_loadModel_' + str(args.loadModel)
        #file_name += '_test_' + str(args.test)
        file_name += '_testModel_' + str(args.testModel)

    file_name = os.path.join(prefix, file_name)

    return file_name

def writeInfo(out, args):
    out.write('embeddingSize {}\n'.format(args.embeddingSize))
    out.write('hiddenSize {}\n'.format(args.hiddenSize))
    out.write('rnnLayers {}\n'.format(args.rnnLayers))
    out.write('maxSteps {}\n'.format(args.maxSteps))
    out.write('dropOut {}\n'.format(args.dropOut))

    out.write('learningRate {}\n'.format(args.learningRate))
    out.write('batchSize {}\n'.format(args.batchSize))
    out.write('epochs {}\n'.format(args.epochs))

    out.write('loadModel {}\n'.format(args.loadModel))
    out.write('test {}\n'.format(args.test))
    out.write('testModel {}\n'.format(args.testModel))

    out.write('vocabSize {}\n'.format(args.vocabSize))
    out.write('preEmbeddings {}\n'.format(args.preEmbedding))

    out.write('project {}\n'.format(args.project))