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



## Training

The model has been trained on a subset of the SHS100k set. It is organized into 9900 cliques of covers. Each one contains several samples with different performances of the same piece. The process of data preparation is as follows: first there are generated pairs that are marked as covers. There are randomly picked songs from the cliques, and along with each one, there is taken one more sample from the same clique. Accordingly, for non-cover pairs, there are picked random pieces and their pairs are samples from other cliques.
The process runs till number of pairs reaches given limit. The default assumption is that class balance is equal - the same number of cover and non-cover pair is generated.


## Performance metrics

