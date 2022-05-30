---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Training a Binary Classifier"
subtitle: ""
summary: "Training a Binary Classifier"
authors: ["Scott Miner"]
tags: ""
categories: ""
date: 2022-05-30T03:23:49.005554
lastmod: 2022-05-30T03:23:49.005554
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

```python
import os
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# Where to save the figures
PROJECT_ROOT_DIR = "."
NOTEBOOK_NAME = "binary-classifier"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", NOTEBOOK_NAME)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

```


```python
# fetch the MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])



Let's look at the data shape.




```python
X, y = mnist["data"], mnist["target"]
X.shape
```




    (70000, 784)




```python
y.shape
```




    (70000,)




```python
28 * 28
```




    784



The above shows that there are 70,000 images and that each picture has 784 features. The reason for this is that each image is 28x28 pixels. Each feature represents one pixel's intensity from 0 (white) to 255 (black).


```python
X.dtype
```




    dtype('float64')




```python
dt = y.dtype
dt
```




    dtype('O')




```python
dt.itemsize
```




    8




```python
dt.name
```




    'object'




```python
print(*[[feature for feature in mnist.feature_names][:10],[feature for feature in mnist.feature_names][-10:]])
```

    ['pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9', 'pixel10'] ['pixel775', 'pixel776', 'pixel777', 'pixel778', 'pixel779', 'pixel780', 'pixel781', 'pixel782', 'pixel783', 'pixel784']
    


```python
import pandas as pd
pd.DataFrame(mnist.data[0:1], columns=mnist.feature_names)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>pixel10</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 784 columns</p>
</div>




```python
import numpy as np
unique, counts = np.unique(y, return_counts=True)
print(np.c_[unique, counts])
plt.bar(unique, counts)
plt.title('MNIST Dataset')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.show()
```

    [['0' 6903]
     ['1' 7877]
     ['2' 6990]
     ['3' 7141]
     ['4' 6824]
     ['5' 6313]
     ['6' 6876]
     ['7' 7293]
     ['8' 6825]
     ['9' 6958]]
    


    
![png](output_14_1.png)
    


Let's print one of the digits.


```python
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()
```

    Saving figure some_digit_plot
    


    
![png](output_16_1.png)
    



```python
y[0]
```




    '5'




```python
y = y.astype(np.uint8)
```

Let's look at some more images from the dataset.


```python
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
```


```python
plt.figure(figsize=(18,18))
example_images = X[0:200]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()
```

    Saving figure more_digits_plot
    


    
![png](output_21_1.png)
    


The MNIST dataset is split into a training (the first 60,000 images) and a test set (the last 10,000 images). 


```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

Next, we shuffle the training set, guaranteeing that all cross-validation folds are similar and not missing any digits. Also, some algorithms are sensitive to the order of training instances. However, shuffling is bad when working on time-series data, such as stock market prices and weather conditions.


```python
np.random.seed(42)
shuffle_index = np.random.permutation(60000)
shuffle_index
```




    array([12628, 37730, 39991, ...,   860, 15795, 56422])




```python
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
```

Let's simplify the problem and only try to identify the number five, an example of a *binary classifier*. Binary classifiers distinguish between just two classes. In this instance, five and not five. We need to create target vectors for the classification task:


```python
y_train_5 = (y_train == 5)
np.nonzero(y_train_5)

```




    (array([    8,    11,    25, ..., 59928, 59942, 59965], dtype=int64),)




```python
y_test_5 = (y_test == 5)
np.nonzero(y_test_5)

```




    (array([   8,   15,   23,   45,   52,   53,   59,  102,  120,  127,  129,
             132,  152,  153,  155,  162,  165,  167,  182,  187,  207,  211,
             218,  219,  240,  253,  261,  283,  289,  317,  319,  333,  340,
             347,  351,  352,  356,  364,  367,  375,  395,  397,  406,  412,
             433,  460,  469,  478,  483,  491,  502,  509,  518,  540,  570,
             588,  604,  618,  638,  645,  654,  674,  692,  694,  710,  711,
             720,  739,  751,  766,  778,  779,  785,  791,  797,  812,  856,
             857,  866,  869,  897,  934,  935,  951,  955,  970,  978, 1003,
            1022, 1032, 1041, 1046, 1070, 1073, 1082, 1087, 1089, 1102, 1113,
            1115, 1131, 1135, 1144, 1146, 1168, 1169, 1190, 1221, 1233, 1235,
            1243, 1252, 1258, 1272, 1281, 1285, 1289, 1299, 1331, 1334, 1339,
            1340, 1370, 1376, 1378, 1393, 1405, 1406, 1421, 1447, 1460, 1466,
            1467, 1471, 1473, 1476, 1493, 1510, 1521, 1525, 1550, 1598, 1618,
            1629, 1635, 1637, 1639, 1641, 1653, 1670, 1672, 1677, 1684, 1693,
            1737, 1747, 1752, 1755, 1761, 1810, 1833, 1846, 1847, 1860, 1866,
            1874, 1879, 1896, 1902, 1910, 1911, 1917, 1931, 1940, 1948, 1954,
            1967, 1970, 1999, 2001, 2003, 2021, 2029, 2030, 2035, 2037, 2040,
            2064, 2073, 2077, 2078, 2100, 2103, 2113, 2114, 2125, 2134, 2159,
            2162, 2180, 2192, 2207, 2214, 2224, 2237, 2241, 2247, 2279, 2282,
            2291, 2322, 2339, 2346, 2369, 2400, 2413, 2445, 2452, 2460, 2476,
            2487, 2515, 2518, 2525, 2526, 2540, 2545, 2546, 2554, 2556, 2558,
            2559, 2569, 2573, 2574, 2581, 2586, 2597, 2604, 2606, 2611, 2616,
            2644, 2653, 2668, 2670, 2682, 2686, 2689, 2697, 2698, 2727, 2743,
            2768, 2772, 2773, 2775, 2790, 2797, 2798, 2805, 2810, 2814, 2829,
            2832, 2839, 2850, 2855, 2903, 2909, 2913, 2919, 2922, 2925, 2930,
            2948, 2951, 2956, 2957, 2969, 2970, 2986, 2987, 3007, 3022, 3028,
            3053, 3093, 3095, 3100, 3102, 3113, 3115, 3117, 3119, 3127, 3145,
            3157, 3171, 3183, 3199, 3220, 3275, 3295, 3311, 3312, 3321, 3334,
            3335, 3336, 3345, 3372, 3393, 3408, 3414, 3416, 3462, 3468, 3470,
            3506, 3537, 3552, 3556, 3558, 3565, 3569, 3570, 3590, 3591, 3595,
            3619, 3623, 3631, 3636, 3645, 3654, 3663, 3678, 3691, 3702, 3750,
            3754, 3756, 3763, 3776, 3778, 3788, 3797, 3806, 3810, 3814, 3826,
            3837, 3855, 3860, 3863, 3877, 3890, 3893, 3898, 3902, 3907, 3917,
            3918, 3928, 3929, 3952, 3955, 3957, 3960, 3968, 3994, 4031, 4052,
            4054, 4056, 4059, 4067, 4072, 4076, 4094, 4108, 4118, 4131, 4152,
            4177, 4196, 4202, 4219, 4226, 4233, 4236, 4254, 4255, 4261, 4263,
            4271, 4300, 4302, 4307, 4310, 4312, 4315, 4323, 4330, 4338, 4340,
            4355, 4356, 4359, 4360, 4364, 4368, 4374, 4378, 4381, 4420, 4422,
            4440, 4461, 4463, 4472, 4520, 4529, 4548, 4569, 4577, 4583, 4596,
            4637, 4645, 4689, 4696, 4711, 4712, 4722, 4728, 4749, 4762, 4763,
            4766, 4771, 4809, 4810, 4828, 4830, 4844, 4867, 4888, 4892, 4902,
            4915, 4933, 4942, 4971, 4979, 5020, 5021, 5056, 5083, 5098, 5102,
            5111, 5134, 5152, 5160, 5170, 5174, 5187, 5194, 5196, 5197, 5206,
            5207, 5222, 5223, 5229, 5275, 5285, 5295, 5302, 5325, 5339, 5347,
            5351, 5364, 5374, 5389, 5397, 5400, 5410, 5420, 5432, 5445, 5451,
            5473, 5480, 5488, 5510, 5518, 5528, 5570, 5571, 5572, 5574, 5579,
            5598, 5608, 5618, 5624, 5632, 5633, 5658, 5662, 5668, 5682, 5697,
            5706, 5711, 5726, 5735, 5742, 5752, 5769, 5779, 5802, 5807, 5821,
            5833, 5843, 5852, 5862, 5867, 5874, 5885, 5891, 5910, 5913, 5922,
            5937, 5947, 5957, 5964, 5972, 5981, 5982, 5985, 5997, 6028, 6042,
            6043, 6053, 6067, 6077, 6087, 6095, 6120, 6136, 6142, 6146, 6148,
            6155, 6165, 6186, 6196, 6206, 6215, 6216, 6227, 6236, 6244, 6257,
            6270, 6277, 6282, 6291, 6314, 6324, 6333, 6341, 6368, 6385, 6386,
            6390, 6392, 6405, 6414, 6415, 6476, 6483, 6486, 6491, 6500, 6518,
            6522, 6525, 6530, 6537, 6544, 6548, 6573, 6598, 6600, 6611, 6620,
            6638, 6706, 6716, 6728, 6746, 6775, 6788, 6803, 6813, 6823, 6832,
            6860, 6866, 6879, 6880, 6884, 6886, 6899, 6908, 6909, 6932, 6942,
            6952, 6964, 6965, 6977, 6981, 6991, 7003, 7018, 7029, 7036, 7057,
            7067, 7077, 7090, 7108, 7134, 7142, 7155, 7160, 7178, 7187, 7195,
            7240, 7241, 7264, 7274, 7284, 7294, 7304, 7306, 7315, 7324, 7351,
            7352, 7372, 7388, 7393, 7397, 7403, 7414, 7430, 7437, 7448, 7451,
            7454, 7474, 7475, 7478, 7498, 7511, 7521, 7531, 7541, 7542, 7559,
            7577, 7578, 7583, 7602, 7612, 7622, 7630, 7643, 7649, 7659, 7672,
            7673, 7676, 7679, 7698, 7715, 7732, 7742, 7752, 7777, 7779, 7793,
            7797, 7808, 7809, 7819, 7826, 7842, 7850, 7859, 7870, 7888, 7918,
            7938, 7948, 7958, 7965, 7974, 7988, 7996, 7997, 8034, 8035, 8038,
            8049, 8062, 8072, 8082, 8089, 8122, 8132, 8142, 8149, 8158, 8160,
            8170, 8180, 8185, 8192, 8214, 8224, 8232, 8270, 8275, 8299, 8327,
            8331, 8348, 8366, 8386, 8412, 8415, 8444, 8445, 8447, 8453, 8463,
            8473, 8487, 8502, 8507, 8531, 8539, 8553, 8563, 8571, 8578, 8601,
            8630, 8632, 8643, 8645, 8652, 8653, 8656, 8665, 8676, 8686, 8696,
            8702, 8710, 8711, 8737, 8741, 8747, 8761, 8774, 8783, 8788, 8803,
            8813, 8823, 8834, 8835, 8847, 8853, 8855, 8863, 8878, 8909, 8940,
            8948, 8964, 8982, 8987, 9013, 9035, 9065, 9075, 9085, 9109, 9114,
            9117, 9119, 9132, 9133, 9159, 9160, 9176, 9184, 9194, 9228, 9234,
            9260, 9268, 9277, 9289, 9290, 9298, 9315, 9329, 9331, 9337, 9338,
            9349, 9360, 9372, 9382, 9391, 9398, 9400, 9422, 9427, 9428, 9465,
            9478, 9481, 9482, 9493, 9503, 9513, 9523, 9533, 9545, 9583, 9584,
            9588, 9590, 9600, 9606, 9616, 9626, 9651, 9671, 9675, 9685, 9702,
            9709, 9719, 9729, 9747, 9749, 9754, 9770, 9777, 9786, 9814, 9830,
            9831, 9841, 9853, 9870, 9877, 9883, 9907, 9941, 9970, 9982, 9988,
            9998], dtype=int64),)



Now we can pick a classifier and train it. *Stochastic Gradient Descent* (SGD) is a good place to start. The classifier can handle very large datasets efficiently since it deals with training instances independently, one at a time, making SGD well suited for _online learning_. Let's create an `SGDClassifier` and train it on the whole training set.


```python
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```




    SGDClassifier(random_state=42)



Now you can use the classifier to detect images of the number 5:


```python
sgd_clf.predict([some_digit])
```




    array([ True])



In this case, the classifier guesses correct. The image represents a 5. Next, we want to evaluate the model's performance.
