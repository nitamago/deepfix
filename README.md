# License

DeepFix is available under the Apache License, version 2.0. Please see the LICENSE file for details.

# Reference

Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade. "DeepFix: Fixing common C language errors by deep learning", AAAI 2017.

# Dependencies

The essential dependencies include `numpy`, `subprocess32`, `regex`, `tensorflow`:

    $ pip install --user numpy subprocess32 regex tensorflow

In order to run the server, you must install `bottle` and `paste` as well as `clang-format`:

    $ pip install --user bottle paste
    $ sudo apt-get install clang-format

# Training 

To train, organize the input into bins for training and validation, with each bin containing a
set of source files that compile. Then run the script `data_processing/data_generator.py` to generate
training data. The data required for training the network is stored in the `network_inputs` directory.

To train the network, use the script `neural_net/train.py`. This saves the model periodically in the
directory `checkpoints`.

# Testing/Serving

Serve one of the trained models by using the script `server/serve.py`. The best checkpoint achieved during
training is automatically used.
