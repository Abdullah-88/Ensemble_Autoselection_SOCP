# Autoselection of the Ensemble of Convolutional Neural Networks with Second-Order Cone Programming

Buse Çisil Güldoğuş, 
Abdullah Nazhat Abdullah, 
Muhammad Ammar Ali, 
Süreyya Özöğür-Akyüz

## Abstract
Ensemble techniques are frequently encountered in machine learning and engineering problems since the method combines different models and produces
an optimal predictive solution. The ensemble concept can be adapted to deep
learning models to provide robustness and reliability. Due to the growth of the
models in deep learning, using ensemble pruning is highly important to deal with
computational complexity. Hence, this study proposes a mathematical model
which prunes the ensemble of Convolutional Neural Networks (CNN) consisting
of different depths and layers that maximizes accuracy and diversity simultaneously with a sparse second order conic optimization model. The proposed
model is tested on CIFAR-10, CIFAR-100 and MNIST data sets which gives
promising results while reducing the complexity of models, significantly.

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/Abdullah-88/Ensemble_Autoselection_SOCP.git

or [download a zip archive](https://github.com/Abdullah-88/Ensemble_Autoselection_SOCP/archive/master.zip).

## Dependencies

python                    3.9.5

tensorflow-gpu            2.4.1
cudatoolkit               10.1.243           
cudnn                     7.6.5
keras-tuner               1.0.3 
Cvxpy                     1.6.3

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication.
