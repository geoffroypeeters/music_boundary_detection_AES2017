# music_boundary_detection_AES2017

This is a pytorch implementation of the Music Boundary Detection system proposed in [Cohen-Hadria and Peeters, AES, 2017] [Link](https://hal.archives-ouvertes.fr/hal-01534850/document).

It is based on training a two-branches convolutional network on temporal patches of audio features to output a boundary probability.
The left branch processes Log-Mel-Spectrogram (a). The right branch processes an MFCC-SSM stacked (b) in depth with a Chroma-SSM (c).

## Code

Given a pyjama dataset definition.
- First compute the audio features (a), (b) and (c) for all the files using ``extract_audiofeatures.py''.
- Second, train the model 'CohenConvNet' using ''train_model.py''

## TODO
- save the weights of the model
