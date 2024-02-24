Example code for training model:

```
python Dropout-models-uncertainty/src/track_model_training.py --param_file params/params.json
```

Example command for ploting uncertainty

```
python Dropout-models-uncertainty/src/plot_uncertainty.py --model_path models/cifar10/convnet-spatial-0.1/random-seed-50.pt --dataset cifar10 --image_a airplane-0029.png --image_b airplane-0030.png --morph_steps 10 --repeat_count 100
