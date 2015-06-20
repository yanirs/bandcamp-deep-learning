#Some experimental results

##Random forest baseline: MNIST digits

97-97.5% on the validation set with 100-1000 trees

##Random forest baseline: Bandcamp dataset

Results are on the validation subset, ± indicates standard deviation when running with ten different random seeds. This
wasn't done in all cases due to time constraints (results without ± are from a single random forest run).

Number of trees | Local (grayscale) | Full (grayscale) | Full (rgb 50x50) | Full (rgb 100x100) 
----------------|-------------------|------------------|------------------|-------------------
100             | 11.7±2.1%         | 13.95±1.09%      | 13.9%            | 14.7%
1000            | 13.5±1.57%        | 14.87±0.84%      | 14.9%            | 15%

