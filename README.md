# Tensorflow Sinusodial Representation Networks (SIREN)
Tensorflow 2.0 implementation of Sinusodial Representation networks (SIREN) from the paper [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

# Installation

 - Pip install

```bash
$ pip install --upgrade tf_siren
```

 - Pip install (test support)

```bash
$ pip install --upgrade tf_siren[tests]
```

# Usage
Copy the `tf_siren` folder to your local directory and import either `SinusodialRepresentationDense` or `SIRENModel`.

```python
from tf_siren import SinusodialRepresentationDense
from tf_siren import SIRENModel

# You can use SinusodialRepresentationDense exactly like you ordinarily use Dense layers.
ip = tf.keras.layers.Input(shape=[2])
x = SinusodialRepresentationDense(32,
                                  activation='sine', # default activation function
                                  w0=1.0)(ip)        # w0 represents sin(w0 * x) where x is the input.
                                  
model = tf.keras.Model(inputs=ip, outputs=x)

# Or directly use the model class to build a multi layer SIREN
model = SIRENModel(units=256, final_units=3, final_activation='sigmoid',
                   num_layers=5, w0=1.0, w0_initial=30.0)
```

# Results on Image Inpainting task
A partial implementation of the image inpainting task is available as the `train_inpainting_siren.py` and `eval_inpainting_siren.py` scripts inside the `scripts` directory.

Weight files are made available in the repository under the `Release` tab of the project. Extract the weights and place the `checkpoints` folder at the scripts directory

These weights generates the following output after 5000 epochs of training with batch size 8192 while using only 10% of the available pixels in the image during training phase.

<img src="https://github.com/titu1994/tf_SIREN/blob/master/images/celtic_knot.png?raw=true" height=100% width=100%>

-----

If we train for using only 20% of the available pixels in the image during training phase -

<img src="https://github.com/titu1994/tf_SIREN/blob/master/images/celtic_knot_20pct.png?raw=true" height=100% width=100%>

-----

If we train for using only 30% of the available pixels in the image during training phase -

<img src="https://github.com/titu1994/tf_SIREN/blob/master/images/celtic_knot_30pct.png?raw=true" height=100% width=100%>

# Citation

```
@misc{sitzmann2020implicit,
    title={Implicit Neural Representations with Periodic Activation Functions},
    author={Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
    year={2020},
    eprint={2006.09661},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# Requirements
 - Tensorflow 2.0+
 - Matplotlib to visualize eval result
