# UI2Code implementation

A machine learning (ML) model that translates a screenshot of a UI of a website to its corresponding code representation. Inspired by the original [pix2code](https://github.com/tonybeltramelli/pix2code) problem and dataset.
This PyTorch codebase has used Timo Angerer and Marvin Knoll (@[marvinknoll](https://github.com/marvinknoll))'s university project (https://github.com/timoangerer/pix2code-pytorch) as a reference.

## Setup and run the model
Follow the follwing steps to train and evaluate the model on Google Cloud Platform. 
**Note:** GCP already has Pytorch and necessary cuda dependencies hence those are not mentioned in the `requirements.txt` file.

1. **Clone the repository**

2. **Install the dependencies**

    Install the dependencies given in requirements.txt:

        pip install -r requirements.txt

3. **Dataset**

    Unzip `data.zip` to find 3 datasets, D1, D2 and D3, in the folder named `data`

    The folder structure of the `data` looks like:

        data
        ├── ...
        └── D3
            └── input
                ├── AF4840B2-2B9F-4ED0-A58D-E260B14858E1.png
                └── AF4840B2-2B9F-4ED0-A58D-E260B14858E1.gui
                └── ...

4. **Split the dataset**

    The `train.py` and `evaluate.py` scripts assume the existence of 2 data split files `train_dataset.txt` and `test_dataset.txt`, each containing the IDs of the data examples for the respective data split. The data split files have to be at the same folder level as the folder containing the data examples.
    85--15 rule is applied for splitting the dataset into train-test splits.
    **Note** The current implementation doesn't use the validation set to monitor the training process or for storing the best weights.

    Run `split_data.py` to generate the data split files for each dataset, for e.g.:

        python split_data.py --data_path=./data/D1/input

5. **Vocabulary file**

    To generate a `vocab.txt` file that contains all the tokens the model should be able to predict, separated by whitespaces.

    Run `build_vocab.py` to generate a vocabulary file named `vocab.txt` at the same folder level as the folder containing the data examples, based on the tokens that appear in the specified dataset.

        python build_vocab.py --data_path=./data/D1/input

6. **Model Architecture**

    An Encoder-Decoder architecture is used for modelling this problem.
    
    **Encoder:** Pretrained ResNet-152 without the last layer is used as a feature extractor backbone. The last classification layer is replaced with FC layer based on the embedding size.
    Embedding Size used : 256

    **Decoder:**

7. **Train the model**

    Run the following command to train the model:

        python train.py --data_path=./data/D1/input --epochs=50 --save_after_epochs=10 --batch_size=4 --split=train --models_dir=./models/D1/

8. **Evaluate the model**

    Run the following command to evaluate the model:

        python evaluate.py --data_path=./data/D1/input --model_file_path=<path_to_model_file> --split=validation

    To visualize the results of the model evaluation, run `visualize_inference.ipynb`.

# References & credits

- Tony Beltramelli for the original pix2code [paper](https://arxiv.org/pdf/1705.07962.pdf) and the [dataset](https://github.com/tonybeltramelli/pix2code).
- Imagine captioning tutorials: [Basic idea of image captioning](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/), [Image captioning PyTorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning), [image captioning TensorFlow](https://blog.insightdatascience.com/automated-front-end-development-using-deep-learning-3169dd086e82)
- [Show, attend and tell](https://arxiv.org/pdf/1502.03044.pdf) paper for image captioning
- (https://github.com/timoangerer/pix2code-pytorch) reference codebase
