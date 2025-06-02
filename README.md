# LyriCover

LyriCover is a part of a [CoverDetectionHub](https://github.com/cncPomper/CoverDetectionHub) project for the Music Information Retrieval (WIMU) course realized at Warsaw University of Technology, winter semester 2024. It is inspired by [THE WORDS REMAIN THE SAME: COVER DETECTION WITH LYRICS TRANSCRIPTION](https://archives.ismir.net/ismir2021/paper/000089.pdf) paper published by Deezer researchers.
It is a fusion model that combines two methods of comparing two samples: text feature extraction with further comparison and analysis of HPCP features. They are joined by a classifier in order to predict whether the pieces are covers or not.


## Text extraction

This part is performed by [OpenAI Whisper](https://github.com/openai/whisper) model. It comes in several sizes:

| Model Size | Parameters | Performance Characteristics                                                                                   |
|------------|------------|----------------------------------------------------------------------------------------------------------------|
| Tiny       | 39 M       | Fastest inference speed; suitable for simple tasks; lower accuracy, especially with complex audio inputs.       |
| Base       | 74 M       | Slightly improved accuracy over Tiny; maintains high speed; still limited in handling complex audio.            |
| Small      | 244 M      | Balanced performance; better accuracy; suitable for general-purpose use cases.                                  |
| Medium     | 769 M      | High accuracy; capable of handling more complex audio; increased computational requirements.                     |
| Large      | 1550 M     | Highest accuracy; best for complex audio and noisy environments; requires significant computational resources.   |

After initial tests, the small model seems to be the best tradeoff between efficiency and achieved results. The SHS100k dataset contains pieces in different languages, so a multilingual model is needed. That comes as a benefit and makes the model more universal for future research. 


## Audio features extraction

The audio features extraction is performed by [librosa](https://github.com/librosa/librosa) library. Currently, it draws out the MFCC features with mean values for the extraction of audio as a vector.

## Fusion 

As described in the paper, the lyrics part detects the instrumental samples by counting unique words. If the model detects no words or less than the defined threshold, the sample is considered instrumental, and the prediction is rescaled based mostly on audio features.

Contrary to the approach presented in paper, the model uses a simple, 2-layer neural network for joining the information and returning the prediction. It was implemented in PyTorch.


## Training

The model has been trained on a subset of the SHS100k set (1000 cover pairs and 1000 non-cover pairs). It is organized into 9998 cliques of covers. Each one contains several samples with different performances of the same piece. The process of data preparation is as follows: first, there are generated pairs that are marked as covers. There are randomly picked songs from the cliques, and along with each one, there is taken one more sample from the same clique. Accordingly, there are picked random pieces for non-cover pairs, and their pairs are samples from other cliques.
The process runs till a number of pairs reaches the given limit. The default assumption is that class balance is equal - the same number of cover and non-cover pair is generated.
The details are described in [main repo](https://github.com/cncPomper/CoverDetectionHub).

## Performance metrics

The model was trained on 2 datasets, Covers80, a well-known set for cover detection, and its variation "Injected Abracadabra", a synthetic dataset where a portion of “Abracadabra” by Steve Miller Band is injected into other audio samples, as described in [Batlle-Roca et al.](https://arxiv.org/pdf/2407.14364).

Results:

| Dataset       | mAP     | mP@10   | mMR1    |
|---------------|---------|---------|---------|
| Injected Abracadabra | 0.82029 | 0.90000 | 1.00000 |
| Covers80     | 0.83425 | 0.09939 | 7.41463 |


# WIMU2025L

As for 2025L WIMU semester course augmentations experimentations were performed. The Lyricover Model was enhanced with custom DataLoader, that allows for extracting features each epoch, robusting augmentation pipeline, assuring that each epoch training set differs a little bit.

Furthermore W&B framework was implemented into the code, allowing for more robust experimentation tracking, saving model configuration, logging events and weights of the model after each run. 

We conducted several experiments, testing the dataset with different number of augmentations, measuring their effect on the model performence. Our initial strategyu was to perform fine tuning with augmentations on new training data and compare the effects of new learning. However due to taking too much epochs and too large of a learning rate we overfitted the model in initial experimentation phase. After lowering number of epochs and limiting learining rate we conducted further experimentations on larger pool of augmentations. 


## Augmentations pipeline used in training

Library we decided to use for augmentations was audiomentations (docs. https://iver56.github.io/audiomentations/). Our decision was based on the recomendations given by Valerio Velardo course (https://www.youtube.com/watch?v=HH_h52I_Qeg&list=PL-wATfeyAMNoR4aqS-Fv0GRmS6bx5RtTW) as well on our own experimentations (torchaudio, our own librosa based implementations, audiomentations), where audiomentations proved to be easiest to use and implement into workflow.

Values of augmentations were selected based on percepting augmented audio files with selected values and anotating if given value makes outcome seem reasonable to human ear.

More interesting augmentation type was selected in later stage of experimentations, when Impulse response was added to pipeline. We selected https://www.echothief.com/ database which is a set of hundreds locations and rooms collected in USA. From this we filtered that slowing down the processing of audio (with sample rate of 32000 Hz). From this filtered data we constructed our own dataset to apply impulse response augmentation.

In the end we concluded that augmentation pipeline should follow the most natural way data could be augmented in real situation (eg. recording song by phone on concert), so changing pitch and time stretching are applied as first, as they can annotate different style or singer, then Impulse response to annotate place where recording could take place, noise is added as it could be applied by any recording device and in the end augmentations that are more connected to file corruption/modification, like MP3 codec or clipping distortions due to data corruption. 

Effects of augmentations can be downloaded and listened to from following link: https://drive.google.com/file/d/1oGQVgj9jmcRsp8YEOKKQJdaqh2gLg4mP/view?usp=drive_link.


## Augmentations effects



Our final effect is the model that beats it's predecessor in terms of precsision, by the price of becoming more conservative and lowering Recall.

For example on test dataset after training our model received following score:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8650  |
| Precision  | 1.0000  |
| Recall     | 0.7286  |
| F1 Score   | 0.8430  |

While original model received:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8725  |
| Precision  | 0.8524  |
| Recall     | 0.8995  |
| F1 Score   | 0.8753  |

Another evaluation was conducted on dataset which was augmented with minor augmentations (pitch shift, time stretch and clipping_distortion). Our augmented model metrics are presented below:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.7788  |
| Precision  | 1.0000  |
| Recall     | 0.5589  |
| F1 Score   | 0.7171  |

and original model for comparisson:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.7838  |
| Precision  | 0.9075  |
| Recall     | 0.6331  |
| F1 Score   | 0.7458  |


Judging by the F1 score, the overall performance of the model was lowered.
On datasets available for evaluation on CoverDetectionHub as well as our new Test called DistractedDataset (where clean audio is sompared with it's cover, where additional audio plays constantly in the background of cover song, momentarly increasing it's volume to match the volume of first song) our model performed slightly better on Inejcted Abracadabra set. It was also better in general in terms of Mean Avarage Precision and Mean rank of first relevant item.

Lyricover trained with augmentations
| Dataset               | mAP     | mP@10   | mMR1   |
|-----------------------|---------|---------|--------|
| Injected Abracadabra  | 0.88648 | 0.90000 | 1.00000 |
| Covers80              | 0.80529 | 0.09939 | 4.98780 |
| Distracted Dataset    | 0.26877 | 0.05300 | 41.69000 |
| Distracted Reference  | 0.28922 | 0.07000 | 34.90000 |

Original Lyricover
| Dataset               | mAP     | mP@10   | mMR1    |
|-----------------------|---------|---------|---------|
| Injected Abracadabra  | 0.82029 | 0.90000 | 1.00000 |
| Covers80              | 0.83425 | 0.09939 | 7.41463 |
| Distracted Dataset    | 0.25801 | 0.05750 | 44.14000 |
| Distracted Reference  | 0.28662 | 0.06950 | 43.84500 |

## Usage and requirements
For reproductibility we attach datasets metadata, json files containing generated pairs from metadata and utility files we used for downloading datasets. We recomend testing download with initial configuration listed in youtube download script in utility_scripts directory. Then replacing data to match shs100k_unique.json file, as this file was the one we used for pair generations. It was also filtered for possibly lacking youtube files. In order to prevent any non exsisting tracks from clustering the training progress, we recomend running filter_data.py file. 

Recomended Python Version: 3.10

Download impulse responses zip file from [(https://drive.google.com/file/d/134V7bc82_P-wG4jMNEoE08HbsImQQtVD/view?usp=drive_link)], unzip and paste as directory to this folder

Set venv envrionment (example: python3 -m venv venv) then run it (source venv/bin/activate)
run pip install -r requirments.txt
Initializing WandB environment and account is described on WandB docs - https://docs.wandb.ai/quickstart/#sign-up-and-create-an-api-key.

For running sweep experiment with singular augmentations selected run command: wandb sweep sweep.yaml

For training with more selected augmentations modify augmentation.yaml and run: python main.py

For evaluation run: python evaluation.py

For the first run we recomend running downloadDataset.py from utils directory and then main.py to test model and data retrival. Then modify given files (eg. download proper dataset like shs100k_unique and conduct learning on larger dataset, change split size in main from 0.5 to more reasonable, etc.)

## Changes to model
 - added augmentations
-  added possibility to extract features on the fly not just before loop
-  improved arguments passed for trainig, including number of epochs, selection of initial model, number of pairs and many more
-  added WandB environment for easier tracking of experiments and saving data configuration and trained models


## Possible further works

Due to time constrains we had to optimize time required for generating data. As generating lyrics was especially time consuming, we opted for generating most of the lyrics once, then loading it each training. This has obvious effect of not taking into consideration augmentation effects on lyrics generation performance. As such we recomend conducting further experimenations on sets of augmentations, but with lyrics generated each time. This task is expected to take long time even on relatively strong machines (we trained it on machine with RTX4060 and i5-14400). 

From our work with the model we recomend starting with low number of epoch, wheter for fine tuning (1-3, sample_rate around 0.0001) or training (3-5,, sample_rate around 0.001) when dealing with 2000 and more pairs. As after each training model is logged to it's own WandB directory, there is no problem with resuming training afterwards. 

Another possible addition would be exploring torchaudio library and whisper model parraller options in order to make feature extraction more robust. 

Additionaly further works on W&B configuration are welcomed.
