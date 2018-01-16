# License

DeepFix is available under the Apache License, version 2.0. Please see the LICENSE file for details.

DeepFixは、Apache Licenseバージョン2.0で利用できます。 詳細については、LICENSEファイルを参照してください。

# Reference

Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade. "DeepFix: Fixing common C language errors by deep learning", AAAI 2017.

# Dependencies

The essential dependencies include `numpy`, `subprocess32`, `regex`, `tensorflow`:

    $ pip install --user numpy subprocess32 regex tensorflow

In order to run the server, you must install `bottle` and `paste` as well as `clang-format`:

    $ pip install --user bottle paste
    $ sudo apt-get install clang-format
    
本質的な依存関係には、 `numpy`、` subprocess32`、 `regex`、` tensorflow`があります：

    $ pip install --user numpy subprocess32 regex tensorflow

サーバを走らせるには、 `clang-format`だけでなく` bottle`と `paste`もインストールする必要があります：

    $ pip install --user bottle paste
    $ sudo apt-get install clang-format


# Training 

To train, organize the input into bins for training and validation, with each bin containing a
set of source files that compile. Then run the script `data_processing/data_generator.py` to generate
training data. The data required for training the network is stored in the `network_inputs` directory.

訓練するには、訓練と検証のために入力をビンに整理します。各ビンには、コンパイルされた一連のソースファイルが含まれています。 次に、data_processing / data_generator.pyスクリプトを実行してトレーニングデータを生成します。 ネットワークのトレーニングに必要なデータは、 `network_inputs`ディレクトリに保存されています。

To train the network, use the script `neural_net/train.py`. This saves the model periodically in the
directory `checkpoints`.

ネットワークを訓練するには、 `neural_net / train.py`スクリプトを使用します。 これは、モデルを `checkpoints`ディレクトリに定期的に保存します。


# Testing/Serving

Serve one of the trained models by using the script `server/serve.py`. The best checkpoint achieved during
training is automatically used.


`server / serve.py`スクリプトを使用して、訓練されたモデルの1つを提供します。 トレーニング中に達成された最高のチェックポイントが自動的に使用されます。
