# Punctuation Restoration

## Data

This section is about the `data/` directory formatting and its contents.

Datasets should be placed inside `data/` as subdirectories. Each dataset should contain its splits, such as `train/`, `dev/`, and `test/`. Please make sure that you do not have a split named `feat/`, as this is a special keyword used in the framework to store embeddings. Each split should then contain an `audio/` folder, a `text/` folder, and an `utt2spk` file. The `audio/` and `text/` folders should have `.wav` speech audio and `.txt` transcript files inside, respectively, and these files' names should be utterance IDs `[utt-id]`. Utterance IDs should have speaker ID `[spk-id]` as prefixes. `utt2spk` should be a text file in which each line is of the format:
```
[spk-id] [utt-id]
```
`data`'s directory structure should look like this:
```
data/
|---dataset1/
|   |---train/
|   |   |---audio/
|   |   |   |---[utt-id1].wav
|   |   |   |---[utt-id2].wav
|   |   |   |---...
|   |   |---text/
|   |   |   |---[utt-id1].txt
|   |   |   |---[utt-id2].txt
|   |   |   |---...
|   |   |---utt2spk
|   |---...
|---...
```
In the code's comments, the term *standard data format* is used to refer to this directory structure.

## Installing Kaldi

1. In `pr/`, run
```bash
git clone https://github.com/kaldi-asr/kaldi
```
and proceed with installing [Kaldi](https://github.com/kaldi-asr/kaldi).
2. Download an additional zip file from [this Google Drive link](https://drive.google.com/file/d/1yfxuqtXrFMi1GhDl9dDxhHbVQE6-tXlf/view?usp=sharing) and place it inside `extras/`. Then, run:
```bash
bash extras/kaldi_setup.sh
```
3. Depending on what models you need, pretrained ones are available for download [here](https://drive.google.com/drive/folders/1YospBmQgXOWE3C5PexAm_3UeJnU1HMXD?usp=sharing). Please place them in the same directory under `models/` as found in the download folder.

## Running the Main Program

To run the main program, please execute `.sh` files in the `scripts/` directory. For example, to use `scripts/run.sh`, run:
```bash
bash scripts/run.sh &
```
By default, messages and outputs will be saved to `run.log`.

The main program's behavior can be customized by modifying arguments in the `.sh` files, as well as the configuration file specified by the `config_path` argument. Arguments and configuration parameters will be checked by the main program for validity. However, even passing all validity checks does not guarantee that the program will run successfully. On the other hand, failing any validity check guarantees that the program will fail.

Certain arguments (e.g. `optimizer` and `batch_size`) are applicable only when `mode='train'`.