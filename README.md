# EfficientPunct

This repository holds the code for our paper "Efficient Ensemble Architecture for Multimodal Embeddings in Punctuation Restoration using Time-Delay Neural Networks", submitted to ASRU 2023.

Some familiarity with Kaldi is highly recommended for usage of the EfficientPunct framework. You can find documentation of Kaldi at [https://kaldi-asr.org/](https://kaldi-asr.org/).

## Installation

1. Install [Kaldi](https://kaldi-asr.org/) by following instructions [here](https://github.com/kaldi-asr/kaldi). Let the root Kaldi directory be referred to in the following documentation as `kaldi/`.
2. Run the following commands:
```bash
cd kaldi/egs/tedlium
git clone https://github.com/GitHubAccountAnonymous/EfficientPunct
mv EfficientPunct/* s5_r3/
rm -rf EfficientPunct
# The framework of EfficientPunct is now located in kaldi/egs/tedlium/s5_r3.
cd s5_r3
```
3. Download an additional zip file from [this Google Drive link](https://drive.google.com/file/d/1CrBgWyZEI9xDG3OBsFUtL_jAKhxv1_MS/view?usp=share_link) and place it inside `kaldi/egs/tedlium/s5_r3/`.
4. Run the following commands:
```bash
unzip additional.zip
rm additional.zip cmd.sh steps utils
rm -r conf
mv additional/* ./
rm -r additional
```

From now on, we will refer to the `kaldi/egs/tedlium/s5_r3` directory as simply `s5_r3/`.

## Data Preparation

Depending on whether you're using data for training or inference, you should use either the `custom_train` and `custom_train_text` or `custom_predict` and `custom_predict_text` subdirectories, respectively. For example, `s5_r3/data` and `s5_r3/db` contain these subdirectories to separately hold each data split. In the following documentation, let `[split]` be either `train` or `predict`, depending on your situation.

- Place each utterance's audio (`.wav` files) in `s5_r3/db/custom_[split]`. Each filename should be of the format `[utterance-id].wav`. 
- Place each utterance's text (`.txt` files) in `s5_r3/db/custom_[split]_text`. Each filename should be of the format `[utterance-id].txt`, and each file should simply contain a single line with the utterance's transcription.
- Create `s5_r3/data/custom_[split]/utt2spk`, a text file with one line for each utterance, and each line should be of the format `[utterance-id] [spk-id]`.
- Create `s5_r3/data/custom_[split]/wav.scp`, a text file with one line for each utterance, and each line should be of the format `[utterance-id] [fully-qualified-path-to-utterance's-wav-file]`.

Here,
- `[utterance-id]` is a unique identifier for the utterance.
- `[spk-id]` is the speaker ID. This should be unique for each speaker.

This data preparation process is exactly the same as according to Kaldi's guidance. Thus, if you run into issues, you may consult [https://kaldi-asr.org/doc/data_prep.html](https://kaldi-asr.org/doc/data_prep.html).

## Scripts

`train.sh` and `predict.sh` are full scripts to train and predict using our framework, respectively, from start to finish. They are both structured in stages, removing the need to run already completed stages if a later stage fails. This is structured in the same fashion as Kaldi scripts. Crucially, the final stage in each of `train.sh` and `predict.sh` calls `tdnn_train.py` and `tdnn_predict.py`, respectively.

To run either script, you will need to prepare the corresponding data `[split]` according to the Data Preparation section. These scripts include both the embedding extraction and TDNN forward passing stages. Trained TDNNs and their evaluation results are saved in `s5_r3/tdnn/`.

Run the scripts using Bash:
```bash
./train.sh
./predict.sh
```

To evaluate the ensemble, please use `ensemble_predict.py`.

## Models

We provide pretrained models `bert/bert.pt` and `tdnn/tdnn.pt` so that you do not have to re-train everything from scratch. By default, `tdnn_predict.py` and `ensemble_predict.py` both use these models, but you may alter the code to use your custom model.
