# Combined Group and Exclusive Sparsity for Deep Networks(CGES)
+ Jaehong Yoon(UNIST), and Sung Ju Hwang(UNIST)

This project hosts the code for our **ICML 2017** paper.

We propose a sparsity regularization method that exploits both positive and negative correlations among the features to enforce the network to be sparse, and at the same time remove any redundancies among the features to fully utilize the capacity of the network. Specifically, we propose to use an exclusive sparsity regularization based on (1; 2)-norm, which promotes competition for features between different weights, thus enforcing them to fit to disjoint sets of features. We further combine the exclusive sparsity with the group sparsity based on (2; 1)-norm, to promote both sharing and competition for features in training of a deep neural network. We validate our method on multiple public datasets, and the results show that our method can obtain more compact and efficient networks while also improving the performance over the base networks with full weights, as opposed to existing sparsity regularizations that often obtain efficiency at the expense of prediction accuracy.

## Reference

If you use this code as part of any published research, please refer the following paper.

```
Reference will be updated soon.
```

## Running Code

We implemented a combined regularizer as described in the paper based on Tensorflow library[Tensorflow](https://www.tensorflow.org/).

### Get our code
```
git clone --recursive https://github.com/jaehong-yoon93/CGES.git CGES
```

### Run examples

In the code, you can run our model on MNIST dataset. Then, you don't need to download dataset on your own, just you can get the daataset when you run our code.
If you want to apply the model to your own data, you need to edit code a little bit. 

For convinence, we added that code prints out sparsity of each layer, training & test accuracy, and several parameter information.
If you execute run.sh script, you can reproduce our model. And when you want to compare with baseline, turn off the **cges** option to False. 
L2 baseline test accuracy might be 98.8x%.

```
./run.sh
```

## Acknowledgement

This work was supported by Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science, ICT & Future Planning (NRF-2016M3C4A7952634).

## Authors

[Jaehong Yoon](http://vision.snu.ac.kr/jaehong-yoon93/)<sup>1</sup>, and [Sung Ju Hwang](http://www.sungjuhwang.com/)<sup>1</sup>

<sup>1</sup>[MLVR Lab](http://ml.unist.ac.kr/) @ School of Electrical and Computer Engineering, UNIST, Ulsan, South Korea
