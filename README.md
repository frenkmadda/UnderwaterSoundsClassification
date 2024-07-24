# UnderwaterSoundsClassification 
 A Project made for a course in Biometrics and Artificial Vision of the Data Science and Machine Learning Master Degree of the University of Salerno.
 The project consists in studying and analyzing underwater audio data using spectograms.
The first goal is to obtain a Binary classificator to distinguish Animal sounds from antropologic ones.
The second goal is to obtain a multiclass classificator to distinguish the exact source of the sound.
 
The project has been divided in 4 main steps:
- Data Analysis (EDA)
- Pre-processing
- Data Augmentation
- Training, Validation and Testing

The dataset is divided in two main classes: 
- Target (Antropologic Sounds)
- Non-Target (Animal sources).
Those classes are divided in subclasses that identify the source.

After the first step the audios have been resampled at 192Khz through Nyquist Shannon Theorem, trimmed at 3 seconds, converted to mono and to 16bit of bit-depth.

Training has been done using pre-trained CNN on image datas like GoogleNet, AlexNet and ResNet50.
Various experiments were conducted by changing the Average paramenter in both cases.
The Binary classificator obtained has excellent results with validation and testing metrics in the range of the 98%.
| Modello   | Average  | Precision | Recall   | F1       |
|-----------|----------|-----------|----------|----------|
| GoogLeNet | Binary   | **0.9973**| **0.9974**| **0.9974**|
|           | Weighted | 0.9946    | 0.9945   | 0.9945   |
|           | Macro    | 0.9899    | 0.9836   | 0.9867   |
|           | Micro    | 0.9949    | 0.9949   | 0.9949   |
| AlexNet   | Weighted | 0.9755    | 0.9625   | 0.9666   |

Multiclass training has been done using one-hot-encoding to ease class rappresentation.
The multiclass classificator need to be improved through a better process of Augmentation, by collecting more data, or by changing the trimming valor.

| Dataset      | Modello   | Encoding         | Average  | Precision     | Recall        | F1            |
|--------------|-----------|------------------|----------|---------------|---------------|---------------|
| Sbilanciato  | GoogLeNet | One Hot Encoding | Weighted | 0.2326        | 0.2246        | 0.2262        |
|              |           |                  | Micro    | 0.1966        | 0.1966        | 0.1966        |
|              | AlexNet   |                  | Weighted | 0.2617        | 0.2367        | 0.2419        |
|              |           |                  | Micro    | 0.2424        | **0.2424**    | **0.2424**    |
|              | ResNet50  |                  | Weighted | **0.2651**    | 0.1630        | 0.1883        |
|              |           |                  | Micro    | 0.1857        | 0.1857        | 0.1857        |
| Bilanciato   | AlexNet   | One Hot Encoding | Weighted | 0.1376        | 0.1275        | 0.1297        |

 
