# QuantumTestTask
A test task with trained UNet model for image semantic segmentation

# Techniques 
### Model
A standart U-Net model (https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png) was created.
The model has 19 convolutional layers, 4 MaxPooling layers and 4 Deconvolutional layers.

### Learning
For learning binary-crossentropy was used as a loss function.
As metrics was 2 functions: Mean Intersection over Unit function (built in Keras) whith 2 classes segmentation and dice score.
For each training mask in csv file a dice score is represented.

### Image Preprocessing
Each image coverts to 128x128 fromat, which speeds up learning process. Predicted models's mask is a binary image in 128x128 format.

# Files
1. **training_dice_scores.csv** contains all dice scores, calculated on training masks
2. **out/** contains all images and their predicted masks
3. **model.py** contains the defenition of the developed U-Net model as well as metric functions and load/save functions
4. **unet_model.h5** tained model
