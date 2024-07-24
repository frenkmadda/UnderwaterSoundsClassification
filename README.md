# UnderwaterSoundsClassification 
 A Project made for a course in Biometrics and Artficial Vision of the Data Science and Machine Learning Master Degree of the University of Salerno.
 The project consists in studying and analyzing underwater audio data using spectograms.
The first goal is to obtain a Binary classificator to distinguish Animal sounds from antropologic ones.
The second goal is to obtain a multiclass classificator to distinguish the exact source of the sound.
 
The project has been divided in 4 main steps:
- Data Analysis
- Pre-processing
- Data Augmentation
- Training, Validation and Testing

The dataset is divided in two main classes: Target (Antropologic Sounds) and Non-Target (Animal sources).
Those classes are divided in subclasses that identify the source.

After the first step the audios have been resampled at 192Khz through Nyquist Shannon Theorem, trimmed at 3 seconds, converted to mono and to 16bit of bit-depth.

The Binary classificator obtained has excellent results with validation and testing metrics in the range of the 98%.

The multiclass classificator need to be improved through a better process of Augmentation, by collecting more data, or by changing the trimming valor.
 
