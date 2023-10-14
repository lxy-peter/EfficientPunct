#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh


set -e -o pipefail -u

nj=1
decode_nj=1   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=1

. utils/parse_options.sh # accept options

if [ $stage -le 1 ]; then

  echo "===================="
  echo "Stage 1 starting"
  python3 create_text.py other custom_predict

  utils/fix_data_dir.sh data/custom_predict
  utils/utt2spk_to_spk2utt.pl data/custom_predict/utt2spk > data/custom_predict/spk2utt
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > data/custom_predict/glm
  utils/fix_data_dir.sh data/custom_predict
  utils/validate_data_dir.sh --no-feats --no-text data/custom_predict
  echo "Stage 1 ending"
  echo ""
fi

# Feature extraction
if [ $stage -le 2 ]; then
  echo "===================="
  echo "Stage 2 starting"
  steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" data/custom_predict
  steps/compute_cmvn_stats.sh data/custom_predict
  utils/fix_data_dir.sh data/custom_predict

  echo "Stage 2 ending"
  echo ""
fi

if [ $stage -le 3 ]; then
  echo "===================="
  echo "Stage 3 starting"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/custom_predict exp/nnet3_cleaned_1d/extractor \
    exp/nnet3_cleaned_1d/ivectors_custom_predict
  echo "Stage 3 ending"
  echo ""
fi

if [ $stage -le 4 ]; then
  echo "===================="
  echo "Stage 4 starting"
  # Obtaining final hidden layer's output
  ./compute_output_custom.sh --nj $nj --iter upto12th \
    --online-ivector-dir exp/nnet3_cleaned_1d/ivectors_custom_predict \
    data/custom_predict exp/chain_cleaned_1d/tdnn1d_sp exp/chain_cleaned_1d/tdnn1d_sp/output12_custom_predict
  echo "Stage 4 ending"
  echo ""
fi

if [ $stage -le 5 ]; then
  echo "===================="
  echo "Stage 5 starting"

  chain_dir=exp/chain_cleaned_1d
  steps/nnet3/align_custom.sh --nj $nj \
    data/custom_predict \
    data/lang \
    $chain_dir/tdnn1d_sp \
    $chain_dir/tdnn1d_sp_custom_predict_ali
  gunzip $chain_dir/tdnn1d_sp_custom_predict_ali/ali.*.gz

  for JOB in $(seq 1 $nj)
  do
    ali-to-phones --ctm-output \
      $chain_dir/tdnn1d_sp/final.mdl \
      ark:$chain_dir/tdnn1d_sp_custom_predict_ali/ali.$JOB \
      $chain_dir/tdnn1d_sp_custom_predict_ali/ali.$JOB.ctm
  done

  echo "Stage 5 ending"
  echo ""
fi

if [ $stage -le 6 ]; then
  echo "===================="
  echo "Stage 6 starting"
  # Generating concatenated BERT+Kaldi embeddings
  if [ -d "embed_custom_predict" ] 
  then
    rm -r embed_custom_predict
  fi
  mkdir embed_custom_predict
  mkdir embed_custom_predict/1792

  if [ -d "exp/chain_cleaned_1d/tdnn1d_sp/output12_custom_predict/split_scp" ] 
  then
    rm -r exp/chain_cleaned_1d/tdnn1d_sp/output12_custom_predict/split_scp
  fi
  mkdir exp/chain_cleaned_1d/tdnn1d_sp/output12_custom_predict/split_scp

  echo "Splitting layer 12 .scp file"
  python3 split_scp.py custom_predict

  echo "Parsing alignments"
  python3 parse_ali.py other custom_predict
  echo "Stage 6 ending"
  echo ""
fi

if [ $stage -le 7 ]; then
  echo "===================="
  echo "Stage 7 starting"
  if [ -d "embed_custom_predict/512/egs" ] 
  then
    rm -r embed_custom_predict/1792/egs
  fi
  mkdir embed_custom_predict/1792/egs

  if [ -d "embed_custom_predict/1792/egs_txt" ] 
  then
    rm -r embed_custom_predict/1792/egs_txt
  fi
  mkdir embed_custom_predict/1792/egs_txt

  python3 prepare_egs.py custom_predict 1792
  echo "Stage 7 ending"
  echo ""
fi

if [ $stage -le 8 ]; then
  echo "===================="
  echo "Stage 8 starting"
  model=model
  nohup python3 -u tdnn_predict.py tdnn/$model.pt > tdnn/$model.txt &
  echo "Stage 8 ending"
  echo ""
fi

echo "predict.sh ending"
exit 0
