# FETCH
## NOTE THIS IS A WORK IN PROGRESS

**6/7/2025: 
Currently cleaning up code, adding some additional features and such in the devel branch.  I hope to release a stable version of this code within the next week**

This is a `PyTorch` version of FETCH based on the original version found [here](https://github.com/devanshkv/fetch).
This version is provided a

There are several key differences between this version and the original version of FETCH

1. The code only works for pre-trained models available in Pytorch Vision
    - Currenlty only pre-trained DenseNet and VGG models were used for transfer training
    - TorchVision has some but not all of the other models used (e.g. XCeption)
    - See the section below for guidelines on how to extend to use other models
2. Predictions can be done based on just frequency data, just DM data, or both
3. Models must be downloaded for local use and are available through Globus [here]()
    - If you do not currenlty have a Globus login, you can get one for free
    - All models here were trained and tested using the original FETCH data 
      available at [astro.phys.wvu.edu/fetch](http://astro.phys.wvu.edu/fetch/).
4. Training and predicting is done on .h*5 files.
    - Those files can contain single or multiple observations
    - The file must contain at least data labeled as data_freq_time and data_dm_time
    - It may also contain data_labels for training data to indicate an observation
      is or is not a pulsar
    - For more details on the structure of the data, see the code in pulsar_data.py
5. There are some training and model differences (minor as far as I can tell) due simply
   to differences between Keras/Tensorflow and Pytorch

Installation
---
Code:
    git clone 
    cd fetch
    python -m pip install .

Models: 
    Must be downloaded for local use.
    Models are accessible through Globus [here]()

Training
---
    
Predicting
---

Extending Pytorch FECTH
---

Citing this work
---

If you use this code I would ask you cite both of the following which includes
the original FETCH:

    @article{Agarwal2020,
      doi = {10.1093/mnras/staa1856},
      url = {https://doi.org/10.1093/mnras/staa1856},
      year = {2020},
      month = jun,
      publisher = {Oxford University Press ({OUP})},
      author = {Devansh Agarwal and Kshitij Aggarwal and Sarah Burke-Spolaor and Duncan R Lorimer and Nathaniel Garver-Daniels},
      title = {{FETCH}: A deep-learning based classifier for fast transient classification},
      journal = {Monthly Notices of the Royal Astronomical Society}
    }
    @software{
        author      = {Weaver, Tony},
        title       = {Pytorch FETCH [source code]},
        year        = 2025,
        url         = {}
    }
