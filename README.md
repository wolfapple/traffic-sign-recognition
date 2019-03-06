# Traffic Sign Recognition

## Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

## Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

### 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

### 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

1. **Clone** the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
  
  ```sh
  git clone https://github.com/wolfapple/traffic-sign-recognition.git
  cd traffic-sign-recognition
  ```

2. **Create** (and activate) a new environment, named `traffic-sign-recognition`. Running this command will create a new `conda` environment that is provisioned with all libraries you need to be successful in this program.
  
  - __Linux__ or __Mac__: 
  ```sh
  conda env create -f environment.yml
  source activate traffic-sign-recognition
  ```
  - __Windows__: 
  ```sh
  conda env create -f environment.yml
  activate traffic-sign-recognition
  ```
  
  At this point your command line should look something like: `(traffic-sign-recognition) <User>:traffic-sign-recognition <user>$`. The `(traffic-sign-recognition)` indicates that your environment has been activated, and you can proceed with further package installations.

3. **Verify** that the environment was created in your environments:  
  
  ```sh
  conda info --envs
  ```

4. **Cleanup** downloaded libraries (remove tarballs, zip files, etc):
  
  ```sh
  conda clean -tp
  ```

5. **That's it!**
  
  Now most of the libraries are available to you. Very occasionally, you will see a repository with an addition requirements file, which exists should you want to use TensorFlow and Keras, for example. In this case, you're encouraged to install another library to your existing environment, or create a new environment for a specific project.
  
  To exit the environment when you have completed your work session, simply close the terminal window.
  
### 3. Uninstalling

To uninstall the environment:

  ```sh
  conda env remove -n traffic-sign-recognition
  ```

## Dataset

The GTSRB dataset (German Traffic Sign Recognition Benchmark) is provided by the Institut für Neuroinformatik group [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). It was published for a competition held in 2011 ([results](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results)). Images are spread across 43 different types of traffic signs and contain a total of 39,209 train examples and 12,630 test ones.

<p align="center"><img src="./images/traffic-signs.png" /></p>

1. [Download the dataset](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which resized the images to 32x32.

2. Unzip the dataset into `./data` directory.

## Training and validating model

Run the script train.py to train the model.

  ```sh
  python train.py
  ```

## Evaluating the model on the test set

As the model trains, model checkpoints are saved to files such as model_x.pth to the current working directory. You can take one of the checkpoints and run:

  ```sh
  python evaluate.py --data [data_dir] --model [model_file]
  ```