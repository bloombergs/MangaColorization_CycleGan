<p align="center">
  <img src="![resultlandscape](https://github.com/user-attachments/assets/dafee68c-266d-410f-abdd-1456a2627b81)
" alt="Colorized Manga Example" />
</p>
# MangaColorization_CycleGan
I perform Manga(Japanese Comics) Colorization using CycleGan(Cycle-Consistent Generative Adversarial Networks),the model is trained using a bw(Black and White) dan Color(Colored) Manga dataset,CycleGan design to convert the bw images to color images and vice versa,while preserving the underlying structure and content of the images.

Dataset = https://huggingface.co/datasets/MichaelP84/manga-colorization-dataset/viewer

# Preprocessing
using datafetch.py load the dataset,were only taking the images(bw,color) loaded into different folder with 500 images each
the images are resized to 256x256 pixels,random horizontal flips are also applied to improve generalization,the images then normalized to a [-1,1] range

# Model
Generator,The generator network follows a ResNet-based architecture, consisting of downsampling layers (for feature extraction) and upsampling layers (for image reconstruction).
Residual blocks help in preserving important details in the generated images.
The generator converts BW manga pages to colorized manga and vice versa, ensuring that the content structure of the image is retained during the translation.

Discriminator,The discriminator network is designed to distinguish between real and fake images for both domains (BW and Colored).
It uses several convolutional layers to classify whether the generated images are realistic or not, helping improve the quality of the generated images.

# Training
Loss Function,Adversarial Loss: Measures how well the generator can create realistic images that fool the discriminator.
Cycle Consistency Loss: Ensures that the generated image, when converted back to the original domain, retains the same content.
Identity Loss: Ensures that images that do not need to be transformed (e.g., already colorized BW images) retain their identity.

Optimizer,the model uses the Adam optimizer with different learning rates for the generators and discriminators.

Learning Rate Scheduler,scheduler is used to decay the learning rate after a set number of epochs to stabilize training.

# Inference and Result
After training, the model can be used to generate colorized manga pages from BW manga. Given an input BW manga image, the generator creates a corresponding colorized version.

Result : 

![mergedresult](https://github.com/user-attachments/assets/0cc13c9e-5f04-4f9c-81cc-3925932bfef1)


