#Some experimental results

##Baselines

Results are on the validation subset, ± indicates standard deviation when running with ten different random seeds. This
wasn't done in all cases due to time constraints (results without ± are from a single run).

The algorithms that were run are random forest (RF) and a linear SVM. In the former case, no preprocessing was done, and
in the latter each feature was normalised to [0, 1] range. See `experiment.run_random_forest_baseline` and
`experiment.run_linear_baseline` for implementation details.

Algorithm       | MNIST digits | Local (grayscale) | Full (grayscale) | Full (rgb 50x50) | Full (rgb 100x100) 
----------------|--------------|-------------------|------------------|------------------|-------------------
Linear SVM      | 92.09%       | 17%               | Memory issues    | 11.10%           | 11.50%
RF (100 trees)  | 97.19±0.08%  | 11.70±2.10%       | 13.95±1.09%      | 13.90%           | 14.70%
RF (1000 trees) | 97.42±0.02%  | 13.50±1.57%       | 14.87±0.84%      | 14.90%           | 15%

##Caffe-based models

All of the following results are on the full RGB dataset. The full-scale images (350x350) were used, with Caffe being
resposible for model-specific downscaling, cropping and reflection.

**Feature extraction**: extracted features from the highest dense layer that's not specific to ImageNet -- 4,096
activations in [CaffeNet](http://caffe.berkeleyvision.org/model_zoo.html) and
[VGGNet-19](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) (layer name is fc7 in both cases).
See `02-caffe-games.ipynb` for details.  

Algorithm       | CaffeNet     | VGGNet-19  
----------------|--------------|-------------------
Linear SVM      | 14.30%       | 15.3%               
RF (100 trees)  | 14.52±0.43%  | 15.10±0.54%         
RF (1000 trees) | 16.72±0.50%  | 16.40±0.64%        

**TODO:** these relatively-poor results may be due to fc7 being too ImageNet-specific. It's worth experimenting with
features extracted from fc6 instead.

**Fine tuning**: fine-tuned CaffeNet, run as in the
[Flickr style demo](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) yields **22.6%** on
the validation set after 10,000 iterations. The only change to the demo was adding example shuffling to reduce
overfitting to a single class (simply added `shuffle: true` to the training data layer's `image_data_param` in
`train_val.prototxt`). 