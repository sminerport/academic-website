---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Confusion Matrices"
subtitle: "Uses Confusion Matrices to Evaluate Classifiers"
summary: "Uses Confusion Matrices to Evaluate Classifiers"
authors: ["Scott Miner"]
tags: ["Confusion Matrices", "Sensitivity", "True Positive Rate", "Recall", "Precision", "Confusion Matrix", "Machine Learning", "Artificial Intelligence"]
categories: ["Confusion Matrices", "Sensitivity", "True Positive Rate", "Recall", "Precision", "Confusion Matrix", "Machine Learning", "Artificial Intelligence"]
date: 2022-07-03T23:33:03.494921
lastmod: 2022-07-03T23:33:03.494921
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

---


# Overview

---

In the [last post](https://scottminer.netlify.app/post/cross-validation/), we discussed using stratified *k*-fold cross-validation to evaluate the accuracy of an `SGDClassifier` that classifies images from the MNIST dataset as either fives or non-fives.

Confusion matrices are often better at evaluating classifiers than accuracy metrics are, especially when dealing with unbalanced datasets. Consider building a classifier that aims to predict the prevalence of a rare disease. If only 5% of the people have the disease, a classifier that always predicts healthy individuals will achieve an accuracy score of 95%. However, the classifier is relatively useless since it never predicts any instances of the disease.

 Confusion matrices provide additional metrics to investigate a classifier's performance. Before diving deeper into confusion matrices, let's look at baseline classifiers.

# Baseline Classifiers

---

In `sklearn` , baseline methods are known as dummy methods and represent the most basic estimators. Baseline methods make predictions based on simple statistics or random guesses.

`sklearn` provides four baseline classification methods. Two use _random_ strategies to make predictions, and two use _constant_ techniques.

## Random Strategies

---

* The `uniform` strategy chooses evenly amongst target classes based on the *number* of categories.
* The `stratified` technique chooses evenly amongst target classes based on the *frequency* of the groups.

The two random methods behave differently on unbalanced datasets. The `uniform` strategy picks evenly amongst classes despite the distribution of classes, whereas the `stratified` technique considers this distribution.

## Constant Strategies

---

* The `constant` strategy returns a user-defined pre-determined target class.
* The `most_frequent` technique returns the single most likely category (also available under the name `prior`).

Let's import the needed classes for this tutorial so we can take a look at a simple example of a Dummy Classifier.

# Imports

---



<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="k">as</span> <span class="nn">mpl</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">metrics</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">datasets</span>
<span class="kn">from</span> <span class="nn">sklearn.dummy</span> <span class="kn">import</span> <span class="n">DummyClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.base</span> <span class="kn">import</span> <span class="n">BaseEstimator</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">accuracy_score</span><span class="p">,</span> <span class="n">precision_score</span><span class="p">,</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">recall_score</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span><span class="p">,</span> <span class="n">cross_val_predict</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">textwrap</span>
<span class="kn">import</span> <span class="nn">warnings</span>
<span class="n">warnings</span><span class="o">.</span><span class="n">simplefilter</span><span class="p">(</span><span class="n">action</span><span class="o">=</span><span class="s1">&#39;ignore&#39;</span><span class="p">,</span> <span class="n">category</span><span class="o">=</span><span class="p">[</span><span class="ne">FutureWarning</span><span class="p">])</span>
</pre></div>


## Matplotlib & Seaborn Settings

---



<div class="highlight"><pre><span></span><span class="n">mpl</span><span class="o">.</span><span class="n">rcParams</span><span class="o">.</span><span class="n">update</span><span class="p">({</span><span class="s1">&#39;font.size&#39;</span><span class="p">:</span> <span class="mi">26</span><span class="p">})</span>
<span class="n">mpl</span><span class="o">.</span><span class="n">rc</span><span class="p">(</span><span class="s1">&#39;figure&#39;</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">set</span><span class="p">(</span><span class="n">font_scale</span><span class="o">=</span><span class="mf">2.0</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">color_palette</span><span class="p">(</span><span class="s2">&quot;deep&quot;</span><span class="p">)</span>
<span class="o">%</span><span class="n">matplotlib</span> <span class="n">inline</span>
</pre></div>


Now, let's build a simple dummy classifier.

# Simple DummyClassifier

---


<div class="highlight"><pre><span></span><span class="c1"># Create the features</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">])</span><br><br><span class="c1"># Create the target class</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">])</span><br><br><span class="c1"># Create a dummy classifier using the most frequest strategy</span>
<span class="n">dummy_clf</span> <span class="o">=</span> <span class="n">DummyClassifier</span><span class="p">(</span><span class="n">strategy</span><span class="o">=</span><span class="s2">&quot;most_frequent&quot;</span><span class="p">)</span><br><br><span class="c1"># Fit the dummy classifier</span>
<span class="n">dummy_clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span><br><br><span class="c1"># Print the predictions and the score</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Predictions: </span><span class="si">{</span><span class="n">dummy_clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Score: </span><span class="si">{</span><span class="n">dummy_clf</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    Predictions: [1 1 1 1]
    Score: 0.75


The dummy classifier always predicts `1` 's because that is the most frequently occurring class in `y` (the target class). The dummy classifier achieved a 75% accuracy since there was a single `0` in the initial dataset.

Let's expand this example to the Iris dataset. First, we need to load the data.


# Load Iris Dataset

---



<div class="highlight"><pre><span></span><span class="c1"># Load the iris dataset</span>
<span class="n">iris</span> <span class="o">=</span> <span class="n">datasets</span><span class="o">.</span><span class="n">load_iris</span><span class="p">()</span><br><br><span class="c1"># How many elements are in the data set?</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Size of full dataset: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span><br><br><span class="c1"># Split the data into training and testing sets</span>
<span class="n">tts_iris</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">,</span>
                            <span class="n">test_size</span><span class="o">=</span><span class="mf">.33</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">21</span><span class="p">)</span><br><br><span class="c1"># Output the result of the train_test_split to a tuple</span>
<span class="p">(</span><span class="n">iris_train_ftrs</span><span class="p">,</span> <span class="n">iris_test_ftrs</span><span class="p">,</span>
 <span class="n">iris_train_tgt</span><span class="p">,</span> <span class="n">iris_test_tgt</span><span class="p">)</span> <span class="o">=</span> <span class="n">tts_iris</span><br><br><span class="c1"># The test set should contain 50 elements, since we set the size to a third</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Size of test dataset: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">iris_test_ftrs</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    Size of full dataset: 150
    Size of test dataset: 50


As you can see, `train_test_split` splits arrays into random training and test sets. The `test_size` parameter corresponds to the proportion of the dataset to include in the test split. The `random_state` parameter allows for reproducible output across multiple function calls. Let's look at a few simple examples.


## Splitting data into training and test sets

---


<div class="highlight"><pre><span></span><span class="c1"># Create 5 rows and 2 columns of features ranging from 0 - 9</span>
<span class="c1"># and a target variable ranging from 0 - 5</span>
<span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span><span class="o">.</span><span class="n">reshape</span><span class="p">((</span><span class="mi">5</span><span class="p">,</span> <span class="mi">2</span><span class="p">)),</span> <span class="nb">range</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span><br><br><span class="c1"># Divide the data into training and testing sets</span>
<span class="n">tts</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.33</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span><br><br><span class="c1"># Get the training and testing features and targets</span>
<span class="p">(</span><span class="n">train_features</span><span class="p">,</span> <span class="n">test_features</span><span class="p">,</span>
 <span class="n">train_target</span><span class="p">,</span>   <span class="n">test_target</span><span class="p">)</span> <span class="o">=</span> <span class="n">tts</span><br><br><span class="c1"># Print them out to verify the results</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Training Features:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">train_features</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Training Target:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">train_target</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Evaluation Features:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">test_features</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Evaluation Target:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">test_target</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    Training Features:
    [[4 5]
     [0 1]
     [6 7]]
    Training Target:
    [2, 0, 3]
    Evaluation Features:
    [[2 3]
     [8 9]]
    Evaluation Target:
    [1, 4]


Setting `shuffle` to `False` specifies whether or not to shuffle the data before splitting. If `shuffle=False` , then stratify must be `None` (the default).



<div class="highlight"><pre><span></span><span class="n">train_test_split</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>





    [[0, 1, 2], [3, 4]]



Let's get back to the Iris dataset.



<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Iris Training Features&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;----------------------&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">iris_train_ftrs</span><span class="p">[:</span><span class="mi">10</span><span class="p">])</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Iris Training Target&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;--------------------&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">iris_train_tgt</span><span class="p">[:</span><span class="mi">10</span><span class="p">])</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Iris Testing Features&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;---------------------&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">iris_test_ftrs</span><span class="p">[:</span><span class="mi">10</span><span class="p">])</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Iris Testing Target&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;---------------------&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">[:</span><span class="mi">10</span><span class="p">])</span>
</pre></div>


    Iris Training Features
    ----------------------
    [[6.9 3.1 4.9 1.5]
     [5.  3.3 1.4 0.2]
     [6.7 3.1 4.4 1.4]
     [7.7 2.6 6.9 2.3]
     [5.7 2.8 4.5 1.3]
     [5.8 2.7 4.1 1. ]
     [4.6 3.1 1.5 0.2]
     [5.1 3.5 1.4 0.3]
     [7.7 3.  6.1 2.3]
     [4.7 3.2 1.6 0.2]]

    Iris Training Target
    --------------------
    [1 0 1 2 1 1 0 0 2 0]

    Iris Testing Features
    ---------------------
    [[5.8 2.6 4.  1.2]
     [5.1 3.8 1.9 0.4]
     [5.  3.4 1.5 0.2]
     [5.1 3.7 1.5 0.4]
     [5.7 3.  4.2 1.2]
     [6.6 3.  4.4 1.4]
     [5.4 3.4 1.7 0.2]
     [5.6 2.8 4.9 2. ]
     [5.  3.4 1.6 0.4]
     [5.1 3.8 1.5 0.3]]

    Iris Testing Target
    ---------------------
    [1 0 0 0 1 1 0 2 0 0]


# Iris Dataset DummyClassifier

---

What's the most frequently occurring class in the Iris dataset?

Let's look at a bar chart of the target variable. As a reminder, the target variable in the Iris dataset is the type of Iris plant (e.g., Setosa, Versicolour, or Virginica).



<div class="highlight"><pre><span></span><span class="c1"># Get the unique values and counts of the target variable</span>
<span class="n">values</span><span class="p">,</span> <span class="n">counts</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">unique</span><span class="p">(</span><span class="n">iris_train_tgt</span><span class="p">,</span> <span class="n">return_counts</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><br><br><span class="c1"># Print the output</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Class counts&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;------------&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">c_</span><span class="p">[</span><span class="n">values</span><span class="p">,</span> <span class="n">counts</span><span class="p">])</span><br><br><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span><br><br><span class="c1"># Create the bar chart</span>
<span class="n">bars</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">values</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="n">counts</span><span class="p">)</span><br><br><span class="c1"># Create the labels above the bars</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar_label</span><span class="p">(</span><span class="n">bars</span><span class="o">.</span><span class="n">containers</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">label_type</span><span class="o">=</span><span class="s2">&quot;edge&quot;</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="mi">26</span><span class="p">)</span><br><br><span class="c1"># Set the titles, axes, and layout</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Frequency of Target Variable Classes in the Iris Dataset&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;Class&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Counts&quot;</span><span class="p">)</span>
<span class="n">labels</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Iris Setosa (&quot;0&quot;)&#39;</span><span class="p">,</span> <span class="s1">&#39;Iris Versicolour (&quot;1&quot;)&#39;</span><span class="p">,</span>
          <span class="s1">&#39;Iris Virginica (&quot;2&quot;)&#39;</span><span class="p">]</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">values</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">rotation</span><span class="o">=</span><span class="mi">9</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">40</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>


    Class counts
    ------------
    [[ 0 32]
     [ 1 33]
     [ 2 35]]




![png](output_20_1.png)



As you can see, `2` or `Iris Virginica` is the most frequently occurring class in the Iris dataset. Supplying the `most_frequent` argument to the `strategy` parameter of the `DummyClassifier` class should return a classifier that predicts all `2` 's.


## Iris Dataset - `most_frequent` DummyClassifier

---

Let's create a baseline classifier for the Iris dataset using the `most_frequent` strategy.



<div class="highlight"><pre><span></span><span class="c1"># Create the classifier</span>
<span class="n">baseline</span> <span class="o">=</span> <span class="n">DummyClassifier</span><span class="p">(</span><span class="n">strategy</span><span class="o">=</span><span class="s2">&quot;most_frequent&quot;</span><span class="p">)</span><br><br><span class="c1"># Fit the classifier</span>
<span class="n">baseline</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">iris_train_ftrs</span><span class="p">,</span> <span class="n">iris_train_tgt</span><span class="p">)</span><br><br><span class="c1"># Make predictions based on the classifier</span>
<span class="n">base_preds</span> <span class="o">=</span> <span class="n">baseline</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">iris_test_ftrs</span><span class="p">)</span><br><br><span class="c1"># Print to verify results</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Length of baseline predictions: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">base_preds</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Ten baseline predictions (they are all 2</span><span class="se">\&#39;</span><span class="s1">s): </span><span class="si">{</span><span class="n">base_preds</span><span class="p">[:</span><span class="mi">10</span><span class="p">]</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    Length of baseline predictions: 50
    Ten baseline predictions (they are all 2's): [2 2 2 2 2 2 2 2 2 2]


Let's check the accuracy of these predictions on the test dataset.



<div class="highlight"><pre><span></span><span class="c1"># Check the accuracy of the baseline classifier</span>
<span class="n">base_acc</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">base_preds</span><span class="p">,</span> <span class="n">iris_test_tgt</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Accuracy of the baseline classifier: </span><span class="si">{</span><span class="n">base_acc</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>


    Accuracy of the baseline classifier: 0.3


As expected, the baseline classifier does not perform that well. Now, let's check the accuracies of all the baseline classifier strategies.

## Comparing the accuracies of all baseline strategies

---



<div class="highlight"><pre><span></span><span class="c1"># Create a list of strategies</span>
<span class="n">strategies</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;constant&#39;</span><span class="p">,</span> <span class="s1">&#39;uniform&#39;</span><span class="p">,</span> <span class="s1">&#39;stratified&#39;</span><span class="p">,</span> <span class="s1">&#39;prior&#39;</span><span class="p">,</span> <span class="s1">&#39;most_frequent&#39;</span><span class="p">]</span><br><br><span class="c1"># Set up args to create different DummyClassifier strategies</span>
<span class="n">baseline_args</span> <span class="o">=</span> <span class="p">[{</span><span class="s1">&#39;strategy&#39;</span><span class="p">:</span> <span class="n">s</span><span class="p">}</span> <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="n">strategies</span><span class="p">]</span><br><br><span class="c1"># Class 0 is setosa</span>
<span class="n">baseline_args</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s1">&#39;constant&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">accuracies</span> <span class="o">=</span> <span class="p">[]</span><br><br><span class="c1"># Loop through the classifiers and display the results in a DF</span>
<span class="k">for</span> <span class="n">bla</span> <span class="ow">in</span> <span class="n">baseline_args</span><span class="p">:</span>
    <span class="n">baseline</span> <span class="o">=</span> <span class="n">DummyClassifier</span><span class="p">(</span><span class="o">**</span><span class="n">bla</span><span class="p">)</span>
    <span class="n">baseline</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">iris_train_ftrs</span><span class="p">,</span> <span class="n">iris_train_tgt</span><span class="p">)</span>
    <span class="n">base_preds</span> <span class="o">=</span> <span class="n">baseline</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">iris_test_ftrs</span><span class="p">)</span>
    <span class="n">accuracies</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">base_preds</span><span class="p">,</span> <span class="n">iris_test_tgt</span><span class="p">))</span><br><br><span class="n">display</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;accuracy&#39;</span><span class="p">:</span> <span class="n">accuracies</span><span class="p">},</span> <span class="n">index</span><span class="o">=</span><span class="n">strategies</span><span class="p">))</span>
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
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>constant</th>
      <td>0.36</td>
    </tr>
    <tr>
      <th>uniform</th>
      <td>0.28</td>
    </tr>
    <tr>
      <th>stratified</th>
      <td>0.34</td>
    </tr>
    <tr>
      <th>prior</th>
      <td>0.30</td>
    </tr>
    <tr>
      <th>most_frequent</th>
      <td>0.30</td>
    </tr>
  </tbody>
</table>
</div>


The `uniform` and `stratified` strategies will return different results when re-run multiple times on a fixed train-test split because they are _randomized_ methods. The other techniques always return the same values for a fixed train-test split.

Let's do the same thing using the MNIST dataset.

# MNIST Dataset - DummyClassifier

---



<div class="highlight"><pre><span></span><span class="c1"># Read in vars from previous notebook</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span><br><br><span class="c1"># Create a list of strategies</span>
<span class="n">strategies</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;constant&#39;</span><span class="p">,</span> <span class="s1">&#39;uniform&#39;</span><span class="p">,</span> <span class="s1">&#39;stratified&#39;</span><span class="p">,</span> <span class="s1">&#39;prior&#39;</span><span class="p">,</span> <span class="s1">&#39;most_frequent&#39;</span><span class="p">]</span><br><br><span class="c1"># Set up args to create different DummyClassifier strategies</span>
<span class="n">baseline_args</span> <span class="o">=</span> <span class="p">[{</span><span class="s1">&#39;strategy&#39;</span><span class="p">:</span> <span class="n">s</span><span class="p">}</span> <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="n">strategies</span><span class="p">]</span><br><br><span class="c1"># False is the constant class</span>
<span class="n">baseline_args</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s1">&#39;constant&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="kc">False</span>
<span class="n">accuracies</span> <span class="o">=</span> <span class="p">[]</span><br><br><span class="c1"># Loop through the classifiers and display the results in a DF</span>
<span class="k">for</span> <span class="n">bla</span> <span class="ow">in</span> <span class="n">baseline_args</span><span class="p">:</span>
    <span class="n">baseline</span> <span class="o">=</span> <span class="n">DummyClassifier</span><span class="p">(</span><span class="o">**</span><span class="n">bla</span><span class="p">)</span>
    <span class="n">baseline</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train_5</span><span class="p">)</span>
    <span class="n">base_preds</span> <span class="o">=</span> <span class="n">baseline</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
    <span class="n">accuracies</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">base_preds</span><span class="p">,</span> <span class="n">y_test_5</span><span class="p">))</span><br><br><span class="n">display</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;accuracy&#39;</span><span class="p">:</span> <span class="n">accuracies</span><span class="p">},</span> <span class="n">index</span><span class="o">=</span><span class="n">strategies</span><span class="p">))</span>
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
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>constant</th>
      <td>0.9108</td>
    </tr>
    <tr>
      <th>uniform</th>
      <td>0.4944</td>
    </tr>
    <tr>
      <th>stratified</th>
      <td>0.8396</td>
    </tr>
    <tr>
      <th>prior</th>
      <td>0.9108</td>
    </tr>
    <tr>
      <th>most_frequent</th>
      <td>0.9108</td>
    </tr>
  </tbody>
</table>
</div>


The `constant` , `prior` , and `most_frequent` strategies all perform the same. The accuracy is so high because only about 10% of the images in the dataset are 5s and this classifier predicts whether or not an image is a 5.

What are some of the other metrics we can use to evaluate a classifier? Let's take a look.


# Metrics Keys

---



<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">textwrap</span><span class="o">.</span><span class="n">fill</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="nb">sorted</span><span class="p">(</span><span class="n">metrics</span><span class="o">.</span><span class="n">SCORERS</span><span class="o">.</span><span class="n">keys</span><span class="p">())),</span> <span class="n">width</span><span class="o">=</span><span class="mi">70</span><span class="p">))</span>
</pre></div>


    ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score',
    'average_precision', 'balanced_accuracy', 'completeness_score',
    'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
    'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score',
    'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples',
    'jaccard_weighted', 'matthews_corrcoef', 'max_error',
    'mutual_info_score', 'neg_brier_score', 'neg_log_loss',
    'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error',
    'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance',
    'neg_mean_squared_error', 'neg_mean_squared_log_error',
    'neg_median_absolute_error', 'neg_root_mean_squared_error',
    'normalized_mutual_info_score', 'precision', 'precision_macro',
    'precision_micro', 'precision_samples', 'precision_weighted', 'r2',
    'rand_score', 'recall', 'recall_macro', 'recall_micro',
    'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo',
    'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted',
    'top_k_accuracy', 'v_measure_score']


There certainly are a lot. How can we figure out the default scorer for a particular classifier? Let's take a look at a *k*-nearest neighbor classifier, for instance.


## Default Scorer for Classifier

---



<div class="highlight"><pre><span></span><span class="n">knn</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">()</span><br><br><span class="n">Returns_index</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">knn</span><span class="o">.</span><span class="n">score</span><span class="o">.</span><span class="vm">__doc__</span><span class="o">.</span><span class="n">splitlines</span><span class="p">())</span><span class="o">.</span><span class="n">index</span><span class="p">(</span><span class="s1">&#39;Returns&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">knn</span><span class="o">.</span><span class="n">score</span><span class="o">.</span><span class="vm">__doc__</span><span class="o">.</span><span class="n">splitlines</span><span class="p">())[</span><span class="n">Returns_index</span><span class="p">:])</span>
</pre></div>


    Returns
            -------
            score : float
                Mean accuracy of ``self.predict(X)`` wrt. `y`.



The above shows that the default evaluation metric for _k_-NN is mean accuracy. Let's look into confusion matrices and some other metrics, including the `precision`, `recall`, and `specificity`.

First, we'll look at a generic confusion matrix.

# Generic Confusion Matrix

---


The below provides a sample of what hypothetical predictions returned by a binary classifier might look like, along with corresponding result labels.



<div class="highlight"><pre><span></span><span class="n">d</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;Reality&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">],</span>
    <span class="s1">&#39;Prediction&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">],</span>
    <span class="s1">&#39;Result&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;True Negative&#39;</span><span class="p">,</span> <span class="s1">&#39;False Negative&#39;</span><span class="p">,</span> <span class="s1">&#39;False Positive&#39;</span><span class="p">,</span> <span class="s1">&#39;True Positive&#39;</span><span class="p">],</span>
    <span class="s1">&#39;Abbreviation&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;TN&#39;</span><span class="p">,</span> <span class="s1">&#39;FN&#39;</span><span class="p">,</span> <span class="s1">&#39;FP&#39;</span><span class="p">,</span> <span class="s1">&#39;TP&#39;</span><span class="p">]}</span><br><br><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">d</span><span class="p">)</span>
<span class="n">df</span>
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
      <th>Reality</th>
      <th>Prediction</th>
      <th>Result</th>
      <th>Abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>True Negative</td>
      <td>TN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>False Negative</td>
      <td>FN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>False Positive</td>
      <td>FP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>True Positive</td>
      <td>TP</td>
    </tr>
  </tbody>
</table>
</div>



In a generic confusion matrix, the rows represent reality, and the columns represent predictions. However, depending on who draws the confusion matrix, the rows and columns might be flip-flopped. Let's take a look at a generic confusion matrix.


<div class="highlight"><pre><span></span><span class="n">data</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;Predicted Negative (PredN)&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;True Negative (TN)&#39;</span><span class="p">,</span> <span class="s1">&#39;False Negative (FN)&#39;</span><span class="p">],</span>
        <span class="s1">&#39;Predicted Positive (PredP)&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;False Positive (FP)&#39;</span><span class="p">,</span> <span class="s1">&#39;True Positive (TP)&#39;</span><span class="p">]}</span><br><br><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span>
    <span class="n">data</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Real Negative (RealN)&#39;</span><span class="p">,</span> <span class="s1">&#39;Real Positive (RealP)&#39;</span><span class="p">])</span><br><br><span class="n">df</span>
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
      <th>Predicted Negative (PredN)</th>
      <th>Predicted Positive (PredP)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Real Negative (RealN)</th>
      <td>True Negative (TN)</td>
      <td>False Positive (FP)</td>
    </tr>
    <tr>
      <th>Real Positive (RealP)</th>
      <td>False Negative (FN)</td>
      <td>True Positive (TP)</td>
    </tr>
  </tbody>
</table>
</div>



We can use the following equations to represent reality and our predictions:

$$\text{Real Negatives} = TN + FP$$
$$\text{Real Positives} = FN + TP$$
$$\text{Predicted Negatives} = TN + FN$$
$$\text{Predicted Positives} = FP + TP$$

Let's look at some additional metrics the confusion matrix provides.

# Metrics from the Confusion Matrix

---

The following provides an overview of some metrics we can calculate from a confusion matrix.

![confusion-matrix](./images-md/confusion-matrix-snippet.PNG)

## Specificity, False Positives, False Alarms, Type I Errors, and Overestimation

---

Let's start with the specificity, which we can calculate using the top row of the above confusion matrix. The specificity is also known as the _true negative rate_ (TNR). The specificity evaluates how many cases we correctly identified as false out of all the real negative cases. The best value is 1 and the worst value is 0. $$\text{specificity} = \frac{TN}{FP + TN} = \frac{TN}{RealN}$$

Another way to think about the specificity is whether or not the classifier raises a flag in the _specific_ cases we want it to. The specificity is intuitively the ability of the classifier to find all the negative samples. The specificity aims to minimize false positives. False negatives do not affect its outcome.

Two ways come to mind to calculate it in `sklearn` on a binary classifier. The first is by calculating the true negatives, etc., from the confusion matrix and performing the calculation by hand. Second, we can set the `pos_label` argument of the `recall_score` function to `0`, producing the same result. Let's look at how different predictions output differing specificities on a balanced dataset.



<div class="highlight"><pre><span></span><span class="c1"># specificity</span>
<span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;All False Predictions&#39;</span><span class="p">,</span> <span class="s1">&#39;Some False Positives&#39;</span><span class="p">,</span>
          <span class="s1">&#39;Some False Negatives&#39;</span><span class="p">,</span> <span class="s1">&#39;Perfect Classifier&#39;</span><span class="p">]</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">i</span> <span class="o">+</span> <span class="mi">1</span><span class="si">}</span><span class="s1">. </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;-----------------------------&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;y_true: </span><span class="si">{</span><span class="n">y_t</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;y_pred: </span><span class="si">{</span><span class="n">y_p</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="p">(</span><span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span><span class="p">)</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Negatives: </span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Positives: </span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Positives: </span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negatives: </span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Negative Rate 1 (Specificity): </span><span class="si">{</span><span class="n">tn</span> <span class="o">/</span> <span class="p">(</span><span class="n">fp</span> <span class="o">+</span> <span class="n">tn</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;True Negative Rate 2 (Specificity): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">pos_label</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
</pre></div>


    1. All False Predictions
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    True Negatives: 5
    False Positives: 0
    True Positives: 0
    False Negatives: 5

    True Negative Rate 1 (Specificity): 1.000
    True Negative Rate 2 (Specificity): 1.000

    2. Some False Positives
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    True Negatives: 3
    False Positives: 2
    True Positives: 5
    False Negatives: 0

    True Negative Rate 1 (Specificity): 0.600
    True Negative Rate 2 (Specificity): 0.600

    3. Some False Negatives
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    True Negatives: 5
    False Positives: 0
    True Positives: 3
    False Negatives: 2

    True Negative Rate 1 (Specificity): 1.000
    True Negative Rate 2 (Specificity): 1.000

    4. Perfect Classifier
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    True Negatives: 5
    False Positives: 0
    True Positives: 5
    False Negatives: 0

    True Negative Rate 1 (Specificity): 1.000
    True Negative Rate 2 (Specificity): 1.000



The specificity only decreases when we ramp up the number of false positives. As long as the classifier identifies all negative cases, the specificity will be 1.0, even if false negatives exist.

## Precision, Positive Predictive Value (PPV)

---

Traveling clockwise around the confusion matrix, let's look at another evaluation metric, this time the _precision_. The precision answers the question, "What is the value of a hit?" and is also known as the _positive predictive value_ (PPV). The formula for the precision is the following:

$$\text{precision} = \frac{TP}{PredP} = \frac{TP}{TP + FP}$$

The precision is appropriate when we want to minimize false positives and is a metric that quantifies the number of correct positive predictions made. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1, and the worst value is 0. We can use the `precision_score` metric of `sklearn` to calculate a classifier's precision.


<div class="highlight"><pre><span></span><span class="c1"># precision</span>
<span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;All False Predictions&#39;</span><span class="p">,</span> <span class="s1">&#39;Some False Positives&#39;</span><span class="p">,</span>
          <span class="s1">&#39;Some False Negatives&#39;</span><span class="p">,</span> <span class="s1">&#39;Perfect Classifier&#39;</span><span class="p">]</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">i</span> <span class="o">+</span> <span class="mi">1</span><span class="si">}</span><span class="s1">. </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;-----------------------------&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;y_true: </span><span class="si">{</span><span class="n">y_t</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;y_pred: </span><span class="si">{</span><span class="n">y_p</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="p">(</span><span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span><span class="p">)</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Negatives: </span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Positives: </span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Positives: </span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negatives: </span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;Positive Predictive Value (Precision): </span><span class="si">{</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span><span class="n">zero_division</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
</pre></div>


    1. All False Predictions
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    True Negatives: 5
    False Positives: 0
    True Positives: 0
    False Negatives: 5

    Positive Predictive Value (Precision): 0.000

    2. Some False Positives
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    True Negatives: 3
    False Positives: 2
    True Positives: 5
    False Negatives: 0

    Positive Predictive Value (Precision): 0.714

    3. Some False Negatives
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    True Negatives: 5
    False Positives: 0
    True Positives: 3
    False Negatives: 2

    Positive Predictive Value (Precision): 1.000

    4. Perfect Classifier
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    True Negatives: 5
    False Positives: 0
    True Positives: 5
    False Negatives: 0

    Positive Predictive Value (Precision): 1.000



Whereas the specificity for a classifier that predicts all negative cases was 1.0, the precision is 0. The precision increases as the true positives increase and the false positives decrease. False negatives do not affect the calculation.

## Recall, Sensitivity, and True Positive Rate (TPR)

---

Now to the bottom row of the above confusion matrix to calculate the recall. The recall is also known as the _sensitivity_ or _true positive rate_ (TPR) and is appropriate when focusing on minimizing false negatives. Imagine a classifier that predicts whether a web page results from a search engine request. The recall answers the question, "Of the valuable web page results, how many did the classifier identify or _recall_ correctly?" Another way to phrase this is, "Within real-world cases of the target class, how many did the classifier identify?" Intuitively, the recall is the classifier's ability to find all the positive samples. At last, the recall is not concerned with false positives and _minimizes_ false negatives.

$$ \text{recall} = \frac{TP}{FN + TP} = \frac{TP}{RealP}$$

## The Complement of the Recall, The False Negative Rate, Type II error, Miss, Underestimation

---

The complement to caring about the number of hits we got right is the number of real hits we got wrong, which would produce a Type II error known as a _false negative_.

$$ \text{false negative rate} = \frac{FN}{TP + FN} = \frac{FN}{RealP} $$

The false negative rate is equivalent to the following:

$$ \text{false negative rate} = 1 - \text{true positive rate} $$

In other words, we can break the target class down into two groups:

* Real positives the classifier identified correctly
* Real positives the classifier identified incorrectly

Using the recall and its complement, we can add up the hits we got right with the hits we got wrong, giving us all the real positive cases.

$$ \frac{TP}{TP+FN} + \frac{FN}{TP+FN} = \frac{TP+FN}{TP+FN} = 1 $$

Let's develop some intution regarding the recall and its complement.


<div class="highlight"><pre><span></span><span class="c1"># Recall and False Negative Rate</span>
<span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;All False Predictions&#39;</span><span class="p">,</span> <span class="s1">&#39;Some False Positives&#39;</span><span class="p">,</span>
          <span class="s1">&#39;Some False Negatives&#39;</span><span class="p">,</span> <span class="s1">&#39;Perfect Classifier&#39;</span><span class="p">]</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">i</span> <span class="o">+</span> <span class="mi">1</span><span class="si">}</span><span class="s1">. </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;-----------------------------&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;y_true: </span><span class="si">{</span><span class="n">y_t</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;y_pred: </span><span class="si">{</span><span class="n">y_p</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="p">(</span><span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span><span class="p">)</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Negatives: </span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Positives: </span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Positives: </span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negatives: </span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Positive Rate (Recall): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;False Negative Rate (Miss Rate): </span><span class="si">{</span><span class="n">fn</span> <span class="o">/</span> <span class="p">(</span><span class="n">tp</span> <span class="o">+</span> <span class="n">fn</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
</pre></div>


    1. All False Predictions
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    True Negatives: 5
    False Positives: 0
    True Positives: 0
    False Negatives: 5

    True Positive Rate (Recall): 0.000
    False Negative Rate (Miss Rate): 1.000

    2. Some False Positives
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    True Negatives: 3
    False Positives: 2
    True Positives: 5
    False Negatives: 0

    True Positive Rate (Recall): 1.000
    False Negative Rate (Miss Rate): 0.000

    3. Some False Negatives
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    True Negatives: 5
    False Positives: 0
    True Positives: 3
    False Negatives: 2

    True Positive Rate (Recall): 0.600
    False Negative Rate (Miss Rate): 0.400

    4. Perfect Classifier
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    True Negatives: 5
    False Positives: 0
    True Positives: 5
    False Negatives: 0

    True Positive Rate (Recall): 1.000
    False Negative Rate (Miss Rate): 0.000



# Coding the Confusion Matrix

---

Let's code several confusion matrics for fictitious classifiers and calculate the metrics defined above for each. We use a `seaborn` `heatmap` to visualize the confusion matrices.



<div class="highlight"><pre><span></span><span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;All False Predictions&#39;</span><span class="p">,</span> <span class="s1">&#39;Some False Positives&#39;</span><span class="p">,</span>
          <span class="s1">&#39;Some False Negatives&#39;</span><span class="p">,</span> <span class="s1">&#39;Perfect Classifier&#39;</span><span class="p">]</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">12</span><span class="p">),</span> <span class="n">sharex</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span>
    <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span><br><br><span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="o">.</span><span class="n">flat</span><span class="p">,</span> <span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span>
    <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">,</span> <span class="n">cbar</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">24</span><span class="p">})</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">i</span> <span class="o">+</span> <span class="mi">1</span><span class="si">}</span><span class="s1">. </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span><br><br><span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">i</span> <span class="o">+</span> <span class="mi">1</span><span class="si">}</span><span class="s1">. </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;-----------------------------&#39;</span><span class="p">)</span>
    <span class="p">(</span><span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span><span class="p">)</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Negatives: </span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Positives: </span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Positives: </span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negatives: </span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;True Negative Rate (Specificity): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">pos_label</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;Positive Predictive Value (Precision): </span><span class="si">{</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;True Positive Rate (Recall): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="n">fnr</span> <span class="o">=</span> <span class="n">fn</span><span class="o">/</span><span class="p">(</span><span class="n">tp</span> <span class="o">+</span> <span class="n">fn</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negative Rate (Recall Complement): </span><span class="si">{</span><span class="n">fnr</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="c1"># Overall accuracy</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Accuracy: </span><span class="si">{</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
</pre></div>




![png](output_49_0.png)



    1. All False Predictions
    -----------------------------
    True Negatives: 5
    False Positives: 0
    True Positives: 0
    False Negatives: 5

    True Negative Rate (Specificity): 1.000
    Positive Predictive Value (Precision): 0.000

    True Positive Rate (Recall): 0.000
    False Negative Rate (Recall Complement): 1.000

    Accuracy: 0.500

    2. Some False Positives
    -----------------------------
    True Negatives: 3
    False Positives: 2
    True Positives: 5
    False Negatives: 0

    True Negative Rate (Specificity): 0.600
    Positive Predictive Value (Precision): 0.714

    True Positive Rate (Recall): 1.000
    False Negative Rate (Recall Complement): 0.000

    Accuracy: 0.800

    3. Some False Negatives
    -----------------------------
    True Negatives: 5
    False Positives: 0
    True Positives: 3
    False Negatives: 2

    True Negative Rate (Specificity): 1.000
    Positive Predictive Value (Precision): 1.000

    True Positive Rate (Recall): 0.600
    False Negative Rate (Recall Complement): 0.400

    Accuracy: 0.800

    4. Perfect Classifier
    -----------------------------
    True Negatives: 5
    False Positives: 0
    True Positives: 5
    False Negatives: 0

    True Negative Rate (Specificity): 1.000
    Positive Predictive Value (Precision): 1.000

    True Positive Rate (Recall): 1.000
    False Negative Rate (Recall Complement): 0.000

    Accuracy: 1.000



False positives affect the TNR (specificity) and the Positive Predictive Value (precision). False negatives, on the other hand, affect the true positive and false negative rates, otherwise known as the recall and its complement.

Let's move on to evaluating the `SGDClassifier` we built for the MNIST dataset that classifies images as either fives or not fives.

# MNIST Dataset

---

To compute a confusion matrix, we need to obtain a set of predictions on the training data. We use the training data to obtain predictions since we should only use the evaluation split for final evaluations. We can use the `cross_val_predict` method from `sklearn` to get the predictions for the `sgd_clf` we created.



<div class="highlight"><pre><span></span><span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">sgd_clf</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">X_train</span>
<span class="o">%</span><span class="n">store</span> <span class="o">-</span><span class="n">r</span> <span class="n">y_train_5</span><br><br><span class="n">y_train_pred</span> <span class="o">=</span> <span class="n">cross_val_predict</span><span class="p">(</span><span class="n">sgd_clf</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train_5</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
</pre></div>


`y_train_pred` contains an array of true and false values, indicating whether or not the classifier predicts a given sample to be a five.


<div class="highlight"><pre><span></span><span class="n">y_train_pred</span><span class="p">[:</span><span class="mi">10</span><span class="p">]</span>
</pre></div>





    array([False, False, False, False, False, False, False, False,  True,
           False])



Let's build the confusion matrix.


<div class="highlight"><pre><span></span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_pred</span><span class="p">)</span>
</pre></div>





    array([[53124,  1455],
           [  949,  4472]], dtype=int64)



We can use the `seaborn` package to make the confusion matrix easier to read.


<div class="highlight"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_pred</span><span class="p">)</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span>
                 <span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span> <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                 <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;flare&quot;</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">24</span><span class="p">})</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Confusion Matrix for SGDClassifier Built on MNIST Dataset&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>




![png](output_57_0.png)



The classifier correctly identified 53,124 images as non-fives (_true negatives_). The remaining 1,455 were wrongly classified as fives (_false positives_). The second row reveals that 949 real fives were incorrectly classified as non-fives (_false negatives_), while the remaining 4,472 were correctly classified as fives (_true positives_).

Let's calculate the remaining metrics.


<div class="highlight"><pre><span></span><span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span>
    <span class="n">y_true</span><span class="o">=</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_pred</span><span class="o">=</span><span class="n">y_train_pred</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span><br><br><span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Negatives: </span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Positives: </span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Positives: </span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negatives: </span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span>
    <span class="sa">f</span><span class="s1">&#39;True Negative Rate (Specificity): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_pred</span><span class="p">,</span> <span class="n">pos_label</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span>
    <span class="sa">f</span><span class="s1">&#39;Positive Predictive Value (Precision): </span><span class="si">{</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_pred</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span>
    <span class="sa">f</span><span class="s1">&#39;True Positive Rate (Recall): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_pred</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="n">fnr</span> <span class="o">=</span> <span class="n">fn</span><span class="o">/</span><span class="p">(</span><span class="n">tp</span> <span class="o">+</span> <span class="n">fn</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negative Rate (Recall Complement): </span><span class="si">{</span><span class="n">fnr</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="c1"># Overall accuracy</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Accuracy: </span><span class="si">{</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_pred</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
</pre></div>


    True Negatives: 53124
    False Positives: 1455
    True Positives: 4472
    False Negatives: 949

    True Negative Rate (Specificity): 0.973
    Positive Predictive Value (Precision): 0.755

    True Positive Rate (Recall): 0.825
    False Negative Rate (Recall Complement): 0.175

    Accuracy: 0.960



The precision indicates that when the classifier picks a five, it is right about 75.5% of the time. The recall demonstrates that the classifier detects 82.5% of the 5s. Let's compare this with a baseline classifier, using a different technique to create the classifier.

# Baseline Classifier on MNIST Dataset

---


<div class="highlight"><pre><span></span><span class="c1"># Create the class</span>
<span class="k">class</span> <span class="nc">Never5Classifier</span><span class="p">(</span><span class="n">BaseEstimator</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">fit</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
        <span class="k">pass</span>
    <span class="k">def</span> <span class="nf">predict</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">X</span><span class="p">):</span>
        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">),</span> <span class="mi">1</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="nb">bool</span><span class="p">)</span><br><br><span class="c1"># Instantiate and create predictions</span>
<span class="n">never_5_clf</span> <span class="o">=</span> <span class="n">Never5Classifier</span><span class="p">()</span>
<span class="n">y_train_never_5_pred</span> <span class="o">=</span> <span class="n">cross_val_predict</span><span class="p">(</span><span class="n">never_5_clf</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train_5</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span><br><br><span class="c1"># Draw the confusion matrix </span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_never_5_pred</span><span class="p">)</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span>
                 <span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span> <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                 <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;flare&quot;</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">24</span><span class="p">})</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Confusion Matrix for Baseline Estimator Built on MNIST Dataset&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>




![png](output_61_0.png)



We never predict any fives. Let's check the metrics.


<div class="highlight"><pre><span></span><span class="c1"># Calculate statistics</span>
<span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span>
    <span class="n">y_true</span><span class="o">=</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_pred</span><span class="o">=</span><span class="n">y_train_never_5_pred</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span><br><br><span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Negatives: </span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Positives: </span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;True Positives: </span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negatives: </span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span>
    <span class="sa">f</span><span class="s1">&#39;True Negative Rate (Specificity): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_never_5_pred</span><span class="p">,</span> <span class="n">pos_label</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span>
    <span class="sa">f</span><span class="s1">&#39;Positive Predictive Value (Precision): </span><span class="si">{</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_never_5_pred</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span>
    <span class="sa">f</span><span class="s1">&#39;True Positive Rate (Recall): </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_never_5_pred</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="n">fnr</span> <span class="o">=</span> <span class="n">fn</span><span class="o">/</span><span class="p">(</span><span class="n">tp</span> <span class="o">+</span> <span class="n">fn</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;False Negative Rate (Recall Complement): </span><span class="si">{</span><span class="n">fnr</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="c1"># Overall accuracy</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Accuracy: </span><span class="si">{</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">,</span> <span class="n">y_train_never_5_pred</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
</pre></div>


    True Negatives: 54579
    False Positives: 0
    True Positives: 0
    False Negatives: 5421

    True Negative Rate (Specificity): 1.000
    Positive Predictive Value (Precision): 0.000

    True Positive Rate (Recall): 0.000
    False Negative Rate (Recall Complement): 1.000

    Accuracy: 0.910



These metrics show why the accuracy can be misleading when evaluating classifiers. We'll draw confusion matrices for multi-class classification problems in the next post.
