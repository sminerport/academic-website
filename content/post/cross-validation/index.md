---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Evaluating a Binary Classifier"
subtitle: "Uses Cross-Validation to Evaluate a Binary Classifier"
summary: "Uses Cross-Validation to Evaluate a Binary Classifier"
authors: ["Scott Miner"]
tags: ["Cross-Validation", "Machine Learning", "Artificial Intelligence", "Sklearn", "Scikit-Learn", "Python"]
categories: ["Cross-Validation", "Machine Learning", "Artificial Intelligence", "Sklearn", "Scikit-Learn", "Python"]
date: 2022-06-12T02:14:35.066313
lastmod: 2022-06-12T02:14:35.066313
featured: false
draft: false

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

The following discusses using cross-validation to evaluate the classifier we built in the [previous post](https://scottminer.netlify.app/post/binary-classifier/), which classifies images from the MNIST dataset as either five or not five.

## Measuring Accuracy Using Cross-Validation

---

Let's take a brief look at the problem that cross-validation solves.

When building a model, we risk **overfitting** the model on the test set when evaluating different hyperparameters. This is because we can tweak the hyperparameters until the model performs optimally. In overfitting, knowledge about the test set "leaks" into the model, and evaluation metrics no longer report on generalization.

One solution is to divide the data into another hold-out set, called the "validation set," which provides evaluation after the model completes training. We can then perform a final model assessment using our test set once the model appears successful.

A disadvantage of this approach is that it reduces the number of samples available to train the model since it partitions the data into three sets: training, validation, and test sets. Also, our results may depend on a particular random pairing of the training and validation sets.

The solution is to use cross-validation. When performing cross-validation, the validation set is no longer needed. We still need the test set for final evaluation. The most basic version of this approach is known as *k*-fold CV, which splits the training data into _k_ smaller groups. The approach trains and evaluates the model *k* times, picking a different fold for evaluation every time and training on the remaining folds. The following describes this process:

* Train a model using $k - 1$ of the folds as training data
* Use the remaining part of the data to validate the model
* Compute a performance measure such as accuracy

The performance measure reported by *k*-fold CV is the average of the values computed in the loop. *k*-fold CV is computationally expensive but offers a significant advantage in problems like inverse inference when sample sizes are small. Also, the process does not waste as much data as fixing an arbitrary validation set. Last but not least, *k*-fold CV allows us to get not only an estimate of the model's performance but also a measure of how precise that estimate is (i.e., standard deviation). We would not have this information with only one validation set. Figure 1 depicts *k*-fold CV.

![cross-val](./images-md/grid_search_cross_validation.png)<center>
<div class="caption">

Figure 1. [Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
</center></div>

Let's look at a simple example of *k*-fold CV using `cross_val_score()` from `sklearn`.


<div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">cross_val_score</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">datasets</span><span class="p">,</span> <span class="n">linear_model</span>
<span class="n">diabetes</span> <span class="o">=</span> <span class="n">datasets</span><span class="o">.</span><span class="n">load_diabetes</span><span class="p">()</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">diabetes</span><span class="o">.</span><span class="n">data</span><span class="p">[:</span><span class="mi">150</span><span class="p">]</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">diabetes</span><span class="o">.</span><span class="n">target</span><span class="p">[:</span><span class="mi">150</span><span class="p">]</span>
<span class="n">lasso</span> <span class="o">=</span> <span class="n">linear_model</span><span class="o">.</span><span class="n">Lasso</span><span class="p">()</span>
<span class="n">scores</span> <span class="o">=</span> <span class="n">cross_val_score</span><span class="p">(</span><span class="n">lasso</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
</pre></div>


Below are all the functions available on the `scores` object, including the `mean` and `std`.


<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="o">*</span><span class="p">[</span><span class="n">x</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">dir</span><span class="p">(</span><span class="n">scores</span><span class="p">)</span> <span class="k">if</span> <span class="ow">not</span> <span class="n">x</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s1">&#39;_&#39;</span><span class="p">)])</span>
</pre></div>


    T all any argmax argmin argpartition argsort astype base byteswap choose clip compress conj conjugate copy ctypes cumprod cumsum data diagonal dot dtype dump dumps fill flags flat flatten getfield imag item itemset itemsize max mean min nbytes ndim newbyteorder nonzero partition prod ptp put ravel real repeat reshape resize round searchsorted setfield setflags shape size sort squeeze std strides sum swapaxes take tobytes tofile tolist tostring trace transpose var view



<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Scores: </span><span class="si">{</span><span class="n">scores</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Mean: </span><span class="si">{</span><span class="n">scores</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Standard Deviation: </span><span class="si">{</span><span class="n">scores</span><span class="o">.</span><span class="n">std</span><span class="p">()</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    Scores: [0.3315057  0.08022103 0.03531816]
    Mean: 0.14901496176589216
    Standard Deviation: 0.13033602441858105


(The model does not perform that well.)

Sometimes we may need more control over the CV process than a function like `cross_val_score()` provides. In these cases, we can implement a version of the cross-validation process from scratch. To do so, we use `StratifiedKFold`. Let's take a closer look at it.

### Stratified k-fold

---

`StratifiedKFold` is a variation of *k*-fold CV that returns *stratified* folds. In stratified folds, each fold contains approximately the same percentage of samples for each class. Let's take a look at a simple example. `StratifiedKFold` provides train/test indices to split data into train/test sets.


<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">StratifiedKFold</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span> <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span> <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">]])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">])</span>
<span class="n">original_data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">c_</span><span class="p">[</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">],</span> <span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Feature1&#39;</span><span class="p">,</span> <span class="s1">&#39;Feature2&#39;</span><span class="p">,</span> <span class="s1">&#39;Target&#39;</span><span class="p">],</span> <span class="n">index</span><span class="o">=</span><span class="p">[</span>
                             <span class="s1">&#39;index_0&#39;</span><span class="p">,</span> <span class="s1">&#39;index_1&#39;</span><span class="p">,</span> <span class="s1">&#39;index_2&#39;</span><span class="p">,</span> <span class="s1">&#39;index_3&#39;</span><span class="p">])</span><br><br><span class="n">display</span><span class="p">(</span><span class="n">original_data</span><span class="p">)</span>
<span class="n">skf</span> <span class="o">=</span> <span class="n">StratifiedKFold</span><span class="p">(</span><span class="n">n_splits</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Number of splits: </span><span class="si">{</span><span class="n">skf</span><span class="o">.</span><span class="n">get_n_splits</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">skf</span><span class="p">)</span>
</pre></div>



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
      <th>Feature1</th>
      <th>Feature2</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>index_0</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>index_1</th>
      <td>3</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>index_2</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>index_3</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    Number of splits: 2
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)


The `split()` method of a cross-validation object yields two $n$-dimensional arrays containing the training and testing set indices for each partition.



<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Original Dataset&#39;</span><span class="p">)</span>
<span class="n">display</span><span class="p">(</span><span class="n">original_data</span><span class="p">)</span>
<span class="k">for</span> <span class="n">fold</span><span class="p">,</span> <span class="p">(</span><span class="n">train_index</span><span class="p">,</span> <span class="n">test_index</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">skf</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">),</span> <span class="mi">1</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;---&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Fold #</span><span class="si">{</span><span class="n">fold</span><span class="si">}</span><span class="s1">:&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;TRAIN_INDEX: </span><span class="si">{</span><span class="n">train_index</span><span class="si">}</span><span class="s1"> TEST_INDEX: </span><span class="si">{</span><span class="n">test_index</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="n">X_train_ex</span><span class="p">,</span> <span class="n">X_test_ex</span> <span class="o">=</span> <span class="n">X</span><span class="p">[</span><span class="n">train_index</span><span class="p">],</span> <span class="n">X</span><span class="p">[</span><span class="n">test_index</span><span class="p">]</span>
    <span class="n">y_train_ex</span><span class="p">,</span> <span class="n">y_test_ex</span> <span class="o">=</span> <span class="n">y</span><span class="p">[</span><span class="n">train_index</span><span class="p">],</span> <span class="n">y</span><span class="p">[</span><span class="n">test_index</span><span class="p">]</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;---&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Training Set&#39;</span><span class="p">)</span>
    <span class="n">display</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">c_</span><span class="p">[</span><span class="n">X_train_ex</span><span class="p">,</span> <span class="n">y_train_ex</span><span class="o">.</span><span class="n">T</span><span class="p">],</span>
                         <span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Feature1&#39;</span><span class="p">,</span> <span class="s1">&#39;Feature2&#39;</span><span class="p">,</span> <span class="s1">&#39;Target&#39;</span><span class="p">],</span>
                         <span class="n">index</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Sample1&#39;</span><span class="p">,</span> <span class="s1">&#39;Sample2&#39;</span><span class="p">]))</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Testing Set&#39;</span><span class="p">)</span>
    <span class="n">display</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">c_</span><span class="p">[</span><span class="n">X_test_ex</span><span class="p">,</span> <span class="n">y_test_ex</span><span class="o">.</span><span class="n">T</span><span class="p">],</span>
                         <span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Feature1&#39;</span><span class="p">,</span> <span class="s1">&#39;Feature2&#39;</span><span class="p">,</span> <span class="s1">&#39;Target&#39;</span><span class="p">],</span>
                         <span class="n">index</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Sample1&#39;</span><span class="p">,</span> <span class="s1">&#39;Sample2&#39;</span><span class="p">]))</span>
</pre></div>


    Original Dataset



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
      <th>Feature1</th>
      <th>Feature2</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>index_0</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>index_1</th>
      <td>3</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>index_2</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>index_3</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    ---

    Fold #1:
    TRAIN_INDEX: [1 3] TEST_INDEX: [0 2]

    ---

    Training Set



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
      <th>Feature1</th>
      <th>Feature2</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sample1</th>
      <td>3</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sample2</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    Testing Set



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
      <th>Feature1</th>
      <th>Feature2</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sample1</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sample2</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    ---

    Fold #2:
    TRAIN_INDEX: [0 2] TEST_INDEX: [1 3]

    ---

    Training Set



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
      <th>Feature1</th>
      <th>Feature2</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sample1</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sample2</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    Testing Set



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
      <th>Feature1</th>
      <th>Feature2</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sample1</th>
      <td>3</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sample2</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


In the above example, indices `[1, 3]` serve as the premise for the `training set` in the first fold, and indices `[0, 2]` serve as the basis for the validation set. The second fold reverses these indices. In other words, each fold serves as the validation set exactly once.

#### np.bincount()

---

We can use `np.bincount()` to compare stratified and non-stratified *k*-fold cross-validation.

First, let's take a closer look at `np.bincount()`. The function `np.bincount()` counts the number of occurrences of each value in an array of non-negative integers.


<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">5</span><span class="p">)))</span>
</pre></div>


    [0 1 2 3 4]
    [1 1 1 1 1]


Each value in the initial array occurs once, which the output from `np.bincount()` demonstrates. The numbers in the array correspond to the counts of the original value. The indices represent the values of the original array. Let's look at another example.


<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">7</span><span class="p">]))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">7</span><span class="p">])))</span>
</pre></div>


    [0 1 1 3 2 1 7]
    [1 3 1 1 0 0 0 1]



<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">7</span><span class="p">]))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">7</span><span class="p">])))</span>
</pre></div>


    [0 1 1 3 2 1 7]
    [1 3 1 1 0 0 0 1]


The number of bins (of size 1) is one larger than the largest value in the array.



<div class="highlight"><pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">7</span><span class="p">,</span> <span class="mi">23</span><span class="p">])</span>
<span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">.</span><span class="n">size</span> <span class="o">==</span> <span class="n">np</span><span class="o">.</span><span class="n">amax</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="o">+</span> <span class="mi">1</span>
</pre></div>





    True



#### Random 50 samples

---

Let's use `np.bincount()` to compare stratified and non-stratified *k*-fold CV using 50 samples from two unbalanced classes.



<div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">StratifiedKFold</span><span class="p">,</span> <span class="n">KFold</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span><br><br><span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">((</span><span class="mi">50</span><span class="p">,</span> <span class="mi">1</span><span class="p">)),</span> <span class="n">np</span><span class="o">.</span><span class="n">hstack</span><span class="p">(([</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">45</span><span class="p">,</span> <span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">*</span> <span class="mi">5</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;X.shape: </span><span class="si">{</span><span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;First five rows of Features: </span><span class="se">\n</span><span class="si">{</span><span class="n">X</span><span class="p">[:</span><span class="mi">5</span><span class="p">]</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;y.shape: </span><span class="si">{</span><span class="n">y</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;First and last five indices of target variable&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">((</span><span class="n">y</span><span class="p">[:</span><span class="mi">5</span><span class="p">],</span> <span class="n">y</span><span class="p">[</span><span class="o">-</span><span class="mi">5</span><span class="p">:]),</span> <span class="n">axis</span><span class="o">=</span><span class="kc">None</span><span class="p">))</span>
</pre></div>


    X.shape: (50, 1)
    First five rows of Features:
    [[1.]
     [1.]
     [1.]
     [1.]
     [1.]]
    y.shape: (50,)
    First and last five indices of target variable
    [0 0 0 0 0 1 1 1 1 1]


Notice the difference in the shapes of the `X` and `y` $n$-dimensional arrays. The shape of `X` is `(50, 1)`, which correlates to 50 rows and 1 column. The shape of `y`, on the other hand, is `(50,)`. In the latter case, the array is 1-dimensional and is otherwise known as a vector. In the former case, the array is a 2-dimensional array.


<div class="highlight"><pre><span></span><span class="n">skf</span> <span class="o">=</span> <span class="n">StratifiedKFold</span><span class="p">(</span><span class="n">n_splits</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="k">for</span> <span class="n">train</span><span class="p">,</span> <span class="n">test</span> <span class="ow">in</span> <span class="n">skf</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;train - </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">y</span><span class="p">[</span><span class="n">train</span><span class="p">])</span><span class="si">}</span><span class="s1"> | test - </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">y</span><span class="p">[</span><span class="n">test</span><span class="p">])</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    train - [30  3] | test - [15  2]
    train - [30  3] | test - [15  2]
    train - [30  4] | test - [15  1]



<div class="highlight"><pre><span></span><span class="n">kf</span> <span class="o">=</span> <span class="n">KFold</span><span class="p">(</span><span class="n">n_splits</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="k">for</span> <span class="n">train</span><span class="p">,</span> <span class="n">test</span> <span class="ow">in</span> <span class="n">kf</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;train - </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">y</span><span class="p">[</span><span class="n">train</span><span class="p">])</span><span class="si">}</span><span class="s1"> | test - </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">bincount</span><span class="p">(</span><span class="n">y</span><span class="p">[</span><span class="n">test</span><span class="p">])</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    train - [28  5] | test - [17]
    train - [28  5] | test - [17]
    train - [34] | test - [11  5]


The first method, `StratifiedKFold`, preserves the class ratios (approximately $\frac{1}{10}$) in both the training and testing sets compared to the basic implementation, which does not.

#### Visualizing Cross-Validation Behaviour

---

We can visualize the behavior of *k*-fold, stratified *k*-fold, and group *k*-fold CV. The code below creates a dataset with 100 randomly generated input data points, three classes split unevenly across data points, and 10 "groups" split evenly across data points. Some CV approaches do specific things with labeled data, others behave differently with grouped data, and others do not use this information. Let's take a look at the structure of the data.




<div class="highlight"><pre><span></span><span class="sd">&quot;&quot;&quot;</span>
<span class="sd">Visualizing cross-validation behavior in scikit-learn</span>
<span class="sd">=====================================================</span><br><br><span class="sd">Choosing the right cross-validation object is a crucial part of fitting a</span>
<span class="sd">model properly. There are many ways to split data into training and test</span>
<span class="sd">sets in order to avoid model overfitting, to standardize the number of</span>
<span class="sd">groups in test sets, etc.</span><br><br><span class="sd">This example visualizes the behavior of several common scikit-learn objects</span>
<span class="sd">for comparison.</span><br><br><span class="sd">&quot;&quot;&quot;</span><br><br><span class="kn">import</span> <span class="nn">numpy.random</span> <span class="k">as</span> <span class="nn">rng</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="p">(</span>
    <span class="n">KFold</span><span class="p">,</span>
    <span class="n">StratifiedKFold</span><span class="p">,</span>
    <span class="n">GroupKFold</span>
<span class="p">)</span><br><br><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">from</span> <span class="nn">matplotlib.patches</span> <span class="kn">import</span> <span class="n">Patch</span><br><br><span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">RandomState</span><span class="p">(</span><span class="mi">1338</span><span class="p">)</span>
<span class="n">cmap_data</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">Paired</span>
<span class="n">cmap_cv</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">coolwarm</span>
<span class="n">n_splits</span> <span class="o">=</span> <span class="mi">4</span>
<span class="c1"># Generate the class/group data</span>
<span class="n">n_points</span> <span class="o">=</span> <span class="mi">100</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">rng</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">100</span><span class="p">,</span> <span class="mi">10</span><span class="p">)</span><br><br><span class="n">percentiles_classes</span> <span class="o">=</span> <span class="p">[</span><span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.6</span><span class="p">]</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">hstack</span><span class="p">([[</span><span class="n">ii</span><span class="p">]</span> <span class="o">*</span> <span class="nb">int</span><span class="p">(</span><span class="mi">100</span> <span class="o">*</span> <span class="n">perc</span><span class="p">)</span>
               <span class="k">for</span> <span class="n">ii</span><span class="p">,</span> <span class="n">perc</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">percentiles_classes</span><span class="p">)])</span><br><br><span class="c1"># Generate uneven groups</span>
<span class="n">group_prior</span> <span class="o">=</span> <span class="n">rng</span><span class="o">.</span><span class="n">dirichlet</span><span class="p">([</span><span class="mi">2</span><span class="p">]</span> <span class="o">*</span> <span class="mi">10</span><span class="p">)</span>
<span class="n">groups</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">repeat</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">10</span><span class="p">),</span> <span class="n">rng</span><span class="o">.</span><span class="n">multinomial</span><span class="p">(</span><span class="mi">100</span><span class="p">,</span> <span class="n">group_prior</span><span class="p">))</span><br><br>
<span class="k">def</span> <span class="nf">visualize_groups</span><span class="p">(</span><span class="n">classes</span><span class="p">,</span> <span class="n">groups</span><span class="p">,</span> <span class="n">name</span><span class="p">):</span>
    <span class="c1"># Visualize dataset groups</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">()</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
        <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">groups</span><span class="p">)),</span>
        <span class="p">[</span><span class="mf">0.5</span><span class="p">]</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">groups</span><span class="p">),</span>
        <span class="n">c</span><span class="o">=</span><span class="n">groups</span><span class="p">,</span>
        <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;_&quot;</span><span class="p">,</span>
        <span class="n">lw</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
        <span class="n">cmap</span><span class="o">=</span><span class="n">cmap_data</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
        <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">groups</span><span class="p">)),</span>
        <span class="p">[</span><span class="mf">3.5</span><span class="p">]</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">groups</span><span class="p">),</span>
        <span class="n">c</span><span class="o">=</span><span class="n">classes</span><span class="p">,</span>
        <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;_&quot;</span><span class="p">,</span>
        <span class="n">lw</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
        <span class="n">cmap</span><span class="o">=</span><span class="n">cmap_data</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set</span><span class="p">(</span>
        <span class="n">ylim</span><span class="o">=</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">5</span><span class="p">],</span>
        <span class="n">yticks</span><span class="o">=</span><span class="p">[</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">3.5</span><span class="p">],</span>
        <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;Data</span><span class="se">\n</span><span class="s2">group&quot;</span><span class="p">,</span> <span class="s2">&quot;Data</span><span class="se">\n</span><span class="s2">class&quot;</span><span class="p">],</span>
        <span class="n">xlabel</span><span class="o">=</span><span class="s2">&quot;Sample index&quot;</span><span class="p">,</span>
    <span class="p">)</span><br><br>
<span class="n">visualize_groups</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="n">groups</span><span class="p">,</span> <span class="s2">&quot;no groups&quot;</span><span class="p">)</span>
</pre></div>




![png](output_24_0.png)



##### Define a function to visualize cross-validation behavior

---

Let's define a function that visualizes the behavior of each cross-validation object. We perform four splits of the data. We'll visualize the indices chosen for the training set (in blue) and the test set (in red) on each fold.


<div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">plot_cv_indices</span><span class="p">(</span><span class="n">cv</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">group</span><span class="p">,</span> <span class="n">ax</span><span class="p">,</span> <span class="n">n_splits</span><span class="p">,</span> <span class="n">lw</span><span class="o">=</span><span class="mi">10</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;Create a sample plot for indices of a cross-validation object.&quot;&quot;&quot;</span><br><br>    <span class="c1"># Generate the training/testing visualizations for each CV split</span>
    <span class="k">for</span> <span class="n">ii</span><span class="p">,</span> <span class="p">(</span><span class="n">tr</span><span class="p">,</span> <span class="n">tt</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">cv</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="n">X</span><span class="o">=</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="n">y</span><span class="p">,</span> <span class="n">groups</span><span class="o">=</span><span class="n">group</span><span class="p">)):</span>
        <span class="c1"># Fill in indices with the training/test groups</span>
        <span class="n">indices</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">nan</span><span class="p">]</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">))</span>
        <span class="n">indices</span><span class="p">[</span><span class="n">tt</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
        <span class="n">indices</span><span class="p">[</span><span class="n">tr</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span><br><br>        <span class="c1"># Visualize the results</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
            <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">indices</span><span class="p">)),</span>
            <span class="p">[</span><span class="n">ii</span> <span class="o">+</span> <span class="mf">0.5</span><span class="p">]</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">indices</span><span class="p">),</span>
            <span class="n">c</span><span class="o">=</span><span class="n">indices</span><span class="p">,</span>
            <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;_&quot;</span><span class="p">,</span>
            <span class="n">lw</span><span class="o">=</span><span class="n">lw</span><span class="p">,</span>
            <span class="n">cmap</span><span class="o">=</span><span class="n">cmap_cv</span><span class="p">,</span>
            <span class="n">vmin</span><span class="o">=-</span><span class="mf">0.2</span><span class="p">,</span>
            <span class="n">vmax</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span>
        <span class="p">)</span><br><br>    <span class="c1"># Plot the data classes and groups at the end</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
        <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">)),</span> <span class="p">[</span><span class="n">ii</span> <span class="o">+</span> <span class="mf">1.5</span><span class="p">]</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">),</span> <span class="n">c</span><span class="o">=</span><span class="n">y</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;_&quot;</span><span class="p">,</span> <span class="n">lw</span><span class="o">=</span><span class="n">lw</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">cmap_data</span>
    <span class="p">)</span><br><br>    <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
        <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">)),</span> <span class="p">[</span><span class="n">ii</span> <span class="o">+</span> <span class="mf">2.5</span><span class="p">]</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">),</span> <span class="n">c</span><span class="o">=</span><span class="n">group</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;_&quot;</span><span class="p">,</span> <span class="n">lw</span><span class="o">=</span><span class="n">lw</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">cmap_data</span>
    <span class="p">)</span><br><br>    <span class="c1"># Formatting</span>
    <span class="n">yticklabels</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">n_splits</span><span class="p">))</span> <span class="o">+</span> <span class="p">[</span><span class="s2">&quot;class&quot;</span><span class="p">,</span> <span class="s2">&quot;group&quot;</span><span class="p">]</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set</span><span class="p">(</span>
        <span class="n">yticks</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="n">n_splits</span> <span class="o">+</span> <span class="mi">2</span><span class="p">)</span> <span class="o">+</span> <span class="mf">0.5</span><span class="p">,</span>
        <span class="n">yticklabels</span><span class="o">=</span><span class="n">yticklabels</span><span class="p">,</span>
        <span class="n">xlabel</span><span class="o">=</span><span class="s2">&quot;Sample index&quot;</span><span class="p">,</span>
        <span class="n">ylabel</span><span class="o">=</span><span class="s2">&quot;CV iteration&quot;</span><span class="p">,</span>
        <span class="n">ylim</span><span class="o">=</span><span class="p">[</span><span class="n">n_splits</span> <span class="o">+</span> <span class="mf">2.2</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.2</span><span class="p">],</span>
        <span class="n">xlim</span><span class="o">=</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">100</span><span class="p">],</span>
    <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">type</span><span class="p">(</span><span class="n">cv</span><span class="p">)</span><span class="o">.</span><span class="vm">__name__</span><span class="p">),</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">15</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">ax</span>
</pre></div>


Let's take a look at the `KFold` cross-validation object:


<div class="highlight"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">()</span>
<span class="n">cv</span> <span class="o">=</span> <span class="n">KFold</span><span class="p">(</span><span class="n">n_splits</span><span class="p">)</span>
<span class="n">plot_cv_indices</span><span class="p">(</span><span class="n">cv</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">groups</span><span class="p">,</span> <span class="n">ax</span><span class="p">,</span> <span class="n">n_splits</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>




![png](output_29_0.png)



As you can see, the _k_-fold CV iterator does not consider the datapoint class or group by default. We can change this by using `StratifiedKFold` or `GroupKFold` (or potentially other options).

* `StratifiedKFold` preserves the percentage of samples for each category.
* `GroupKFold` ensures that the same group will not appear in two different folds.

Let's plot all three together.


<div class="highlight"><pre><span></span><span class="n">cvs</span> <span class="o">=</span> <span class="p">[</span><span class="n">KFold</span><span class="p">,</span> <span class="n">StratifiedKFold</span><span class="p">,</span> <span class="n">GroupKFold</span><span class="p">]</span><br><br><span class="k">for</span> <span class="n">cv</span> <span class="ow">in</span> <span class="n">cvs</span><span class="p">:</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span> <span class="mi">3</span><span class="p">))</span>
    <span class="n">plot_cv_indices</span><span class="p">(</span><span class="n">cv</span><span class="p">(</span><span class="n">n_splits</span><span class="p">),</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">groups</span><span class="p">,</span> <span class="n">ax</span><span class="p">,</span> <span class="n">n_splits</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="p">[</span><span class="n">Patch</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="n">cmap_cv</span><span class="p">(</span><span class="mf">0.8</span><span class="p">)),</span> <span class="n">Patch</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="n">cmap_cv</span><span class="p">(</span><span class="mf">0.02</span><span class="p">))],</span>
        <span class="p">[</span><span class="s2">&quot;Testing set&quot;</span><span class="p">,</span> <span class="s2">&quot;Training set&quot;</span><span class="p">],</span>
        <span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mf">1.02</span><span class="p">,</span> <span class="mf">0.8</span><span class="p">),</span>
    <span class="p">)</span>
<span class="c1"># Make the legend fit</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">fig</span><span class="o">.</span><span class="n">subplots_adjust</span><span class="p">(</span><span class="n">right</span><span class="o">=</span><span class="mf">0.7</span><span class="p">)</span>
</pre></div>




![png](output_31_0.png)





![png](output_31_1.png)





![png](output_31_2.png)



# SGDClassifier

---

The following uses stratified 3-fold cross-validation to evaluate the `SGDClassifer` we trained on the MNIST dataset. The code then counts the number of correct predictions and outputs the ratio of accurate predictions.


<div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">StratifiedKFold</span>
<span class="kn">from</span> <span class="nn">sklearn.base</span> <span class="kn">import</span> <span class="n">clone</span><br><br><span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">X_train</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">X_test</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">y_train</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">y_test</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">y_train_5</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">y_test_5</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">sgd_clf</span><br><br><span class="n">skfolds</span> <span class="o">=</span> <span class="n">StratifiedKFold</span><span class="p">(</span><span class="n">n_splits</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span><br><br><span class="k">for</span> <span class="n">train_index</span><span class="p">,</span> <span class="n">test_index</span> <span class="ow">in</span> <span class="n">skfolds</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train_5</span><span class="p">):</span>
    <span class="n">clone_clf</span> <span class="o">=</span> <span class="n">clone</span><span class="p">(</span><span class="n">sgd_clf</span><span class="p">)</span>
    <span class="n">X_train_folds</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="n">train_index</span><span class="p">]</span>
    <span class="n">y_train_folds</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_train_5</span><span class="p">[</span><span class="n">train_index</span><span class="p">])</span>
    <span class="n">X_test_fold</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="n">test_index</span><span class="p">]</span>
    <span class="n">y_test_fold</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_train_5</span><span class="p">[</span><span class="n">test_index</span><span class="p">])</span><br><br>    <span class="n">clone_clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_folds</span><span class="p">,</span> <span class="n">y_train_folds</span><span class="p">)</span>
    <span class="n">y_pred</span> <span class="o">=</span> <span class="n">clone_clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test_fold</span><span class="p">)</span>
    <span class="n">n_correct</span> <span class="o">=</span> <span class="nb">sum</span><span class="p">(</span><span class="n">y_pred</span> <span class="o">==</span> <span class="n">y_test_fold</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">n_correct</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>


    0.9681
    0.95655
    0.95515


We can also use the `cross_val_score()` function to evaluate the `SGDClassifier` model using *k*-fold CV, and we get the same results.


<div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">cross_val_score</span>
<span class="n">cross_val_score</span><span class="p">(</span><span class="n">sgd_clf</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train_5</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">&#39;accuracy&#39;</span><span class="p">)</span>
</pre></div>





    array([0.9681 , 0.95655, 0.95515])



The accuracy is above 95% on all cross-validation folds. However, let's look at a dumb classifier that classifies every image as belonging to the "not-5" class. The following classifier, termed `Never5Classifier`, outputs a false prediction for each record passed to it.


<div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.base</span> <span class="kn">import</span> <span class="n">BaseEstimator</span><br><br><span class="k">class</span> <span class="nc">Never5Classifier</span><span class="p">(</span><span class="n">BaseEstimator</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">fit</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
        <span class="k">pass</span><br><br>    <span class="k">def</span> <span class="nf">predict</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">X</span><span class="p">):</span>
        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">),</span> <span class="mi">1</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="nb">bool</span><span class="p">)</span>
</pre></div>


Let's check this model's accuracy.


<div class="highlight"><pre><span></span><span class="n">never_5_clf</span> <span class="o">=</span> <span class="n">Never5Classifier</span><span class="p">()</span>
<span class="n">scores</span> <span class="o">=</span> <span class="n">cross_val_score</span><span class="p">(</span><span class="n">never_5_clf</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train_5</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s2">&quot;accuracy&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Scores: </span><span class="si">{</span><span class="n">scores</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Mean: </span><span class="si">{</span><span class="n">scores</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span><span class="si">:</span><span class="s1">.4f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Standard Deviation: </span><span class="si">{</span><span class="n">scores</span><span class="o">.</span><span class="n">std</span><span class="p">()</span><span class="si">:</span><span class="s1">.4f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    Scores: [0.909   0.90745 0.9125 ]
    Mean: 0.9097
    Standard Deviation: 0.0021


The model that never predicts a 5 has over 90% accuracy since only about 10% of the images are 5. Suppose the classifier always guesses _not_ a 5. In that case, it will be right about 90% of the time, which is why accuracy is not generally the preferred performance measure for classifiers, especially when the dataset is unbalanced. On the other hand, confusion matrices are much better suited to evaluate classifiers, which we will look at in the next post.
