**CATR**: Image Captioning with Transformers
========
PyTorch training code and pretrained models for **CATR** (**CA**ption **TR**ansformer).

The models are also available via torch hub,
to load model with pretrained weights simply do:
```python
model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)  # you can choose between v1, v2 and v3
```
### Samples:

All these images has been annotated by CATR.

Test with your own bunch of images:
````bash
$ python predict.py --path /path/to/image --v v2  // You can choose between v1, v2, v3 [default is v3]
````
Or Try it out in colab [notebook](catr_demo.ipynb)

# Usage 
There are no extra compiled components in CATR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies.
First, clone the repository locally:
```
$ git clone https://github.com/talha-khalid-qureshi/Image-Captioning.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+ along with remaining dependencies:
```
$ pip install -r requirements.txt
```
That's it, should be good to train and test caption models.

We train CATR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales an crops are used for augmentation.
Images are rescaled to have max size 299.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

## Testing
To test CATR with your own images.
```
$ python predict.py --path /path/to/image --v v2  // You can choose between v1, v2, v3 [default is v3]
```
