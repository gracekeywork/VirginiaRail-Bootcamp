# Virginia Rail Beg Call Classifier — Notebook Descriptions

This repository contains a pipeline for training a bioacoustic classifier to detect Virginia Rail (`virail`) beg calls using [OpenSoundscape](https://opensoundscape.org/). The notebooks should be run roughly in the order listed below.

---

## 1. `build_positives.ipynb`

Loads hand-annotated Raven Pro selection tables for Virginia Rail recordings and converts them into a labeled DataFrame of 1-second positive clips.

**Reads from:**
- Raven `.txt` annotation files: `/home/brg226/projects/vira_beg/training_data/annotated_positive_audio/audio_viratrain/`
- Corresponding `.wav` audio files in the same directory

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/positives/positive_vira_1s_gk.pkl`

Also includes exploratory visualizations of training audio spectrograms and confusion species (Blue Jay, Red-winged Blackbird, Black-capped Chickadee).

---

## 2. `Extract_targetneg_XC_snapshot_gk.ipynb`

Extracts negative and (optionally) target-positive clips from the [BirdSet XCL Xeno-Canto snapshot](https://huggingface.co/datasets/DBD-research-group/BirdSet) based on eBird species codes. Negatives are drawn from a user-specified list of confusion/co-occurring species.

**Reads from:**
- BirdSet XCL dataset cache: `/media/kiwi/datasets/annotated/xeno_canto_snapshot/data_cache`
- eBird taxonomy codes: `/media/auk/projects/srg/Kitzes_projects/ECOO53_CBAN_classifier/ebird_codes.txt`

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/negative_multihot_labels_gk.pkl`

---

## 3. `grab_field_negatives_gk.ipynb`

Builds a negative dataset from real field recordings. Randomly selects one `.WAV` file per recorder subdirectory from the `rloc2025a` dataset and splits each file into 1-second clips.

**Reads from:**
- Field recordings: `/media/kiwi/datasets/unfinalized/rloc2025a/` (one `.WAV` randomly selected per subdirectory)

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/field_negatives/selected_field_negatives_gk.csv`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/field_negatives/field_negatives_gk.pkl`

---

## 4. `create_fulltrain.ipynb`

Merges the positive clips, Xeno-Canto negatives, and field negatives into a single unified dataset in OpenSoundscape's `(file, start_time, end_time)` MultiIndex format. Handles column alignment and fills missing label columns with zeros.

**Reads from:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/positives/positive_vira_1s_gk.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/negative_multihot_labels_gk.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/field_negatives/field_negatives_gk.pkl`

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/fulltrain/full_dataset_1s_gk.pkl`

---

## 5. `build_split.ipynb`

Performs a leakage-safe train/test split on the full dataset. Positive test files are manually specified so that no audio from the same recording appears in both splits. Negative files are split 80/20 at the file level.

**Reads from:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/fulltrain/full_dataset_1s_gk.pkl`

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/train_1s_gk.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/test_1s_gk.pkl`

---

## 6. `upsample.ipynb`

Balances the training and test sets using OpenSoundscape's `resample` utility. Creates a `SpectrogramPreprocessor` with field audio overlay augmentation and a bandpass filter (2000–7000 Hz), then trains a ResNet18 CNN for 20 epochs with evaluation metrics (mAP, AUROC, precision-recall curve, confusion matrix) and false positive analysis.

**Reads from:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/train_1s_gk.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/test_1s_gk.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/field_negatives/field_negatives_gk.pkl`

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/train_df_gk_40926.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/test_df_gk_40926.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/shallow_train_df_gk_40926.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/shallow_test_df_gk_40926.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_runs/upsampled_data/balanced_train_1s_gk.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_runs/upsampled_data/balanced_train_1s_gk.csv`
- CNN checkpoints: `model_training_checkpoints/` (local directory)

---

## 7. `gen_embedding_gk.ipynb`

Generates Perch2 embeddings for all train and test clips using the `bioacoustics_model_zoo` library. Verifies that all audio files are readable before embedding. Configures TensorFlow to use a specific GPU with a 5 GB memory cap.

**Reads from:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/shallow_train_df_gk_40926.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/shallow_test_df_gk_40926.pkl`

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/embeddings/shallow_train_embeddings_041326.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/embeddings/shallow_test_embeddings_041326.pkl`

---

## 8. `train_shallow.ipynb`

Trains a shallow MLP classifier head on top of Perch2 embeddings using OpenSoundscape's `MLPClassifier`. Evaluates predictions on the test set and visualizes score distributions broken down by TP/TN/FP/FN, with audio playback of top false positives.

**Reads from:**
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/shallow_train_df_gk_40926.pkl`
- `/media/auk/projects/gak76/vira_beg_outputs/training_data/splits/shallow_test_df_gk_40926.pkl`

**Saves to:**
- `/media/auk/projects/gak76/vira_beg_outputs/models/virail_perch_041326_try3.pth`

---

## Output Directory Summary

```
/media/auk/projects/gak76/vira_beg_outputs/
├── training_data/
│   ├── positives/
│   │   └── positive_vira_1s_gk.pkl
│   ├── negative_multihot_labels_gk.pkl
│   ├── field_negatives/
│   │   ├── selected_field_negatives_gk.csv
│   │   └── field_negatives_gk.pkl
│   ├── fulltrain/
│   │   └── full_dataset_1s_gk.pkl
│   └── splits/
│       ├── train_1s_gk.pkl
│       ├── test_1s_gk.pkl
│       ├── train_df_gk_40926.pkl
│       ├── test_df_gk_40926.pkl
│       ├── shallow_train_df_gk_40926.pkl
│       └── shallow_test_df_gk_40926.pkl
├── embeddings/
│   ├── shallow_train_embeddings_041326.pkl
│   └── shallow_test_embeddings_041326.pkl
├── models/
│   └── virail_perch_041326_try3.pth
└── training_runs/
    └── upsampled_data/
        ├── balanced_train_1s_gk.pkl
        └── balanced_train_1s_gk.csv
```
