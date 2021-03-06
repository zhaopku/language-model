# Language Model. ETH NLU project, 2018

## Before training

    1. Please move the training data to ./data, and the embedding file to the root directory of the project.

## Notice

    1. The model is trained for 12 hours and then we save the parameters for computing perplexities on the test set (sentences_test).
    2. We used a dropout rate of 0.2 to normalize the model.


## Requirements

    1. Python 3.6.4
    2. TensorFlow 1.7
    3. tqdm

## Usage

    Please see predict.py for details of the command line options.

    Specifically:

        For task 1, training:

            A: python3 main.py --hiddenSize 512 --embeddingSize 100 --rnnLayers 1 --testFile sentences_test
            B: python3 main.py --hiddenSize 512 --embeddingSize 100 --rnnLayers 1 --preEmbedding --testFile sentences_test
            C: python3 main.py --hiddenSize 1024 --embeddingSize 100 --rnnLayers 1 --project --testFile sentences_test

        For task 1, computing perplexities for sentences_test

            A: python3 main.py --hiddenSize 512 --embeddingSize 100 --rnnLayers 1 --testFile sentences_test --writePerplexity --testModel --loadModel
            B: python3 main.py --hiddenSize 512 --embeddingSize 100 --rnnLayers 1 --preEmbedding --testFile sentences_test --writePerplexity --testModel --loadModel
            C: python3 main.py --hiddenSize 1024 --embeddingSize 100 --rnnLayers 1 --project --testFile sentences_test --writePerplexity --testModel --loadModel

        For task 2:

            python3 main.py --generate --loadModel --hiddenSize 1024 --project

## Contact
	Zhao Meng, zhmeng@student.ethz.ch

