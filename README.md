# Cycle Consistent VAEs

This repository contains a PyTorch implementation of the ECCV 2018 Paper, [Disentangling Factors of Variation with Cycle-Consistent Variational Auto-Encoders](https://arxiv.org/abs/1804.10469). For the original implementation of the paper, please refer to the author's implementation [here](https://github.com/ananyahjha93/cycle-consistent-vae).

For further details on the results and analysis, please refer to ```Problem 2``` in the report - ```report.pdf``` 


## Installing Dependencies

Make sure you have a Python3+ version. Run the following command - 

```
pip install -r requirements.txt
```

## Training the Model

### Dataset

Download the __2D Sprites__ dataset from [here](http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-data.tar.gz) and run the following steps - 
- Create a folder named ```./splits/``` in the root folder. 
- Run the code present in ```create_sprites.ipynb``` to read the dataset, and process the images. 
- The code will create train and test splits for the dataset in the ```./splits/```  folder

### Model

The model definition is present in ```cycle_consistent_vae_in.py``` and the training script is present in ```train_512_64_in.py```. 

## Results

### Loss Plots

The loss plots are generated while the training is going on, with the help of the ```livelossplot``` library. 

![Loss Plot](./figures/cycle_vae_512_64_in.png)

### Generated Images

A set of images are randomly sampled from the latent space and plotted as follows - 

![Generated Images](./figures/trained_random_generated.png)

### Generation Quality

With the increase in epochs, it is observed that the generation quality of the VAE improves.

![Generation Quality](./figures/generation_quality.png)

### Evaluating the Latent representation

An SVM is trained on the Latent representations of the samples, achieving a test __micro-average F1 score of 0.81__. For further details on the results, please refer to the report - ```report.pdf``` .

#### t-SNE visualisation of the Latent space

![t-SNE visualisation of the Latent space](./figures/latent_tsne.png)

#### Classifier Performance

![Classifier Performance](./figures/latent_confusion_matrix.png)


## License 

Copyright (c) 2020 Aditya Chetan and Brihi Joshi

For license information, see [LICENSE](LICENSE) or http://mit-license.org


- - -

Done by [Aditya Chetan](https://github.com/justachetan) and Brihi Joshi


This code was written as a part of a course group assignment in **Deep Learning** with [Dr. Saket Anand](https://www.iiitd.ac.in/anands) at IIIT Delhi during Winter 2020 Semester.

For bugs in the code, please write to: brihi16142 [at] iiitd [dot] ac [dot] in or aditya16217 [at] iiitd [dot] ac [dot] in
