# USEsum
USEsum is an extractive and abstractive summarisation system that can generate short summaries from reviews corpora. This code supports the paper [Sentence Embeddings as an intermediate target in end-to-end summarisation] (https://github.com/USE-sum/usesum/blob/master/Sentence_Embeddings_as_an_intermediate_target_in_end_to_end_summarisation_MZ_SM.pdf) .


The uniquness of this approach is that it can summarize hundreds of sentences, while most systems summarize articles containing up several dozens of sentences. 


This an adaptation of [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) , most of the code in this repository is an old version of OpenNMT. The main contribution is [in this script](https://github.com/USE-sum/usesum/blob/master/onmt/decoders/vecdiff.py) .

# Installation
Please run pip install -r requirements.txt

To perform end to end predictions and preprocess datasets one needs Universal Sentence Encoder running as a sever, using port 8501. If you use the pretrained model, pleasue use version 2 of the Universal Sentence Encoder. If you train from scratch, most recent version is recommended.

# Training custom models

To train custom models please follow the below procedure:

## Preprocessing

Please preproces your data using preprocessing/convert_corpora_to_np.py . This script requires tensorflow and tensorflow hub. It requires text corpora, like https://github.com/useg-data/useg-data/tree/master/data and results in npy files which contain Universal Sentence Embeddings of the sentences from the corpora. 
The npy files are needed for extractive model training, text files are needed for end to end inference.

Next please execute preprocess_vectors.py script. For example:

python preprocess_vectors.py -train_src data/sample/src-train.npy -train_tgt data/sample/tgt-train.npy -valid_src data/sample/src-valid.npy -valid_tgt data/sample/tgt-valid.npy -save_data data/sample/ -data_type vector

## Training

To execute trainig on previously preprocessed data, please run:

python train.py -data data/sample/   -save_model models/sample/  -learning_rate 0.0002  -word_vec_size 512 -layers 1 -dec_rnn_size 512 -enc_rnn_size 1024  -max_grad_norm 0   -optim adam  -encoder_type brnn -decoder_type vecdif  -dropout 0.0   -param_init 0   -label_smoothing 0.1    -normalization sents   -max_generator_batches 2   -accum_count 1   -param_init_glorot  -batch_size 1 -generator_function none -model_type vector  -log_file train_log.txt -start_decay_steps 12320 -decay_steps 12320 -valid_steps 12320 -valid_batch_size 1 -save_checkpoint_steps 12320 -report_every 50 -train_steps 300000 -world_size 1 -gpu_ranks 0

Batch size of 1 is required for proper trainig.

## Evaluation

To evaluate angle distances between predicted and target vectors please run:

python translate.py -model models/extractive_USEsum.pt -src data/sample/src-test.npy -tgt data/sample/tgt-test.npy -output predictions.txt -verbose -data_type vector -decoder_type vecdif -batch_size 1

# Pretrained models

The abstractive model is available (here)[https://s3-eu-west-1.amazonaws.com/use-sum-abstractive-model/_step_140250.pt] 

The extractive model is available (here)[https://s3-eu-west-1.amazonaws.com/use-sum-abstractive-model/extractive_USEsum.pt] 


# Run extractive summarization
Please run 

* python server.py
* Run Universal Sentence Encoder with Tensorflow serving ([quick gude](https://www.tensorflow.org/tfx/serving/docker))
* python use_sum.py  -t extractive

# Run end-to-end summarization
Please run :

* python server.py
* Run Universal Sentence Encoder with Tensorflow serving ([quick gude](https://www.tensorflow.org/tfx/serving/docker))
* python use_sum.py

## For both end-to-end and extractive summarization:
The default parameters of the use_sum script match default OpenNMT server settings and attached dockerized USE server. 
Input file should contain document to summarize. Each document should be represented by one line containig all its sentences. 
For [example](https://github.com/useg-data/useg-data/blob/master/data/src-test-txt.txt) 

