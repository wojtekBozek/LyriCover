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

The model was trained on 2 datasets, Covers80, a well-known set for cover detection, and its variation "Injected Abracadabra" " a synthetic dataset where a portion of “Abracadabra” by Steve Miller Band is injected into other audio samples, as described in [Batlle-Roca et al.](https://arxiv.org/pdf/2407.14364).

Results:

| Dataset       | mAP     | mP@10   | mMR1    |
|---------------|---------|---------|---------|
| Injected Abracadabra | 0.82029 | 0.90000 | 1.00000 |
| Covers80     | 0.83425 | 0.09939 | 7.41463 |


