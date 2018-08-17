# Deep Convolutional GAN in Keras
## Prerequisites
#### The following package(s) are required:
`tensorflow`, `keras`, `pillow`, `numpy`

#### The following package(s) are recommended:
`tensorflow-gpu` (requires a GPU)

## Structures
#### Discriminator
1. 5x5 filter, 32 channels, same padding
2. Average pooling
3. Dropout layer
4. 5x5 filter, 64 channels, same padding
5. Average pooling
6. Dropout layer
7. 1024-unit fully connected layer
8. Single output layer (sigmoid activation for cleaner results)

#### Generator
1. 9800-unit fully connected layer, reshaped to 14x14 convolutional input with 50 channels
2. 5x5 filter, 100 channel convolutional layer, same padding with upsampling
3. 5x5 filter, 150 channel convolutional layer, same padding
4. 5x5 filter, 1 channel convolutional output layer, same padding (sigmoid activation for cleaner results)

## Results
The following pictures are results after training for 25,000 iterations

![alt text](readme-images/0.png)
![alt text](readme-images/1.png)
![alt text](readme-images/2.png)
![alt text](readme-images/3.png)
![alt text](readme-images/4.png)
![alt text](readme-images/5.png)
![alt text](readme-images/6.png)
![alt text](readme-images/7.png)
![alt text](readme-images/8.png)
![alt text](readme-images/9.png)

## References
Please see the following links for more information about DCGANs and solving this specific problem:

https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

https://github.com/llSourcell/Generative_Adversarial_networks_LIVE/blob/master/EZGAN.ipynb
