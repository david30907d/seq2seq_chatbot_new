# Dcard Topic Recommender

This project contains several models to predict topics for an article.

## Install

* (Recommended): Use [docker](https://hub.docker.com/r/tensorflow/tensorflow/tags/)
	* `nvidia-docker run -itd --name <name> -p <port>:8888 tensorflow/tensorflow:nightly-gpu-py3 /run_jupyter.sh --allow-root --ip=0.0.0.0`

## Manually Install

`pip3 install tf-nightly-gpu=1.9.0.dev20180603`

## Data

1. Dcard hash tag
2. [搜狗資料集](http://www.sogou.com/labs/resource/cs.php)

## Model

* [DNN](dnn)
* [NER](ner)
* [Seq2Seq](seq2seq/README.md)

## Built With

python3.5

## Contributors

* __王浩丞__ []()
* __Ｍartina__ []()
* __張泰瑋__ [david](https://github.com/david30907d)

## License

not yet