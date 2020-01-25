# Vocalist Gender Identification

Identification of the Gender of a Vocalist with Mel-spectrogram and 1D CNN.  Sample vocal WAV files from DSD100 and ccmixter.org were split into 20s segments.  In some cases these files were isolated, in others they were source files.  800 Samples were then fed into a 1D CNN with a train, validation, and test set.

## Dependencies

1. Librosa
2. Numpy
3. Pandas
4. Keras
5. Scikit-learn
6. MongoDB
7. Python 3.6+
8. Scipy

## Preprocessing

1. Chunk into 20s
2. Librosa Mel-Spectrogram
3. Librosa power_to_db
4. Cepstral Mean Normalization
5. Flatten

## Results

Precision: 81.8%, Recall: 97.6%, Test Accuracy 89.7%

For more information on results, see this post https://mblance.com/blog/identification-of-a-vocalist-s-gender-with-cnn/
