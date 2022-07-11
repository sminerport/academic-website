---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Confusion Matrices - Part 2"
subtitle: "Multiclass Classification on the Iris Dataset"
summary: "Multiclass Classification on the Iris Dataset"
authors: ["Scott Miner"]
tags: ["Confusion Matrices", "Sensitivity", "True Positive Rate", "Recall", "Precision", "Confusion Matrix", "Machine Learning", "Artificial Intelligence"]
categories: ["Confusion Matrices", "Sensitivity", "True Positive Rate", "Recall", "Precision", "Confusion Matrix", "Machine Learning", "Artificial Intelligence"]
date: 2022-07-10T21:21:38.786821
lastmod: 2022-07-10T21:21:38.786821
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

This post takes off where the [last one](https://scottminer.netlify.app/post/confusion-matrix/) left off and talks about building confusion matrices for multi-class classification problems. We load the Iris dataset, split it into training and test sets, build a _K_-Nearest Neighbors (_k_-NN) classifier that attempts to predict the class of Iris plant (setosa, versicolor, or virginica), and craft a confusion matrix using these predictions. We then describe some additional metrics, including the `macro` and `micro` precision, and discuss `sklearn`'s `classification_report`, discussing the $F_1$ metric and delving slightly deeper into the $F_{0.5}$ and $F_2$ metrics. In the end, we discuss the `classification_report` for the confusion matrix we built on the Iris dataset.

Let's import the needed libraries and set the `matplotlib` and `seaborn` settings.

# Imports

---



<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="k">as</span> <span class="nn">mpl</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">datasets</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">accuracy_score</span><span class="p">,</span> <span class="n">precision_score</span><span class="p">,</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">recall_score</span><span class="p">,</span> <span class="n">classification_report</span><span class="p">,</span> <span class="n">f1_score</span><span class="p">,</span> <span class="n">fbeta_score</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">warnings</span>
<span class="kn">import</span> <span class="nn">re</span>
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


# Iris Dataset

---

Let's fit a *k*-NN classifier on the Iris training data and generate predictions on the test features to build a confusion matrix.


<div class="highlight"><pre><span></span><span class="c1"># Load the iris dataset</span>
<span class="n">iris</span> <span class="o">=</span> <span class="n">datasets</span><span class="o">.</span><span class="n">load_iris</span><span class="p">()</span><br><br><span class="c1"># How many elements are in the data set?</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Size of full dataset: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span><br><br><span class="c1"># Split the data into training and testing sets</span>
<span class="n">tts_iris</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">,</span>
                            <span class="n">test_size</span><span class="o">=</span><span class="mf">.33</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">21</span><span class="p">)</span><br><br><span class="c1"># Output the result of the train_test_split to a tuple</span>
<span class="p">(</span><span class="n">iris_train_ftrs</span><span class="p">,</span> <span class="n">iris_test_ftrs</span><span class="p">,</span>
 <span class="n">iris_train_tgt</span><span class="p">,</span> <span class="n">iris_test_tgt</span><span class="p">)</span> <span class="o">=</span> <span class="n">tts_iris</span><br><br><span class="c1"># The test set should contain 50 elements, since we set the size to a third</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Size of test dataset: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">iris_test_ftrs</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="n">tgt_preds</span> <span class="o">=</span> <span class="p">(</span><span class="n">KNeighborsClassifier</span><span class="p">()</span>
             <span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">iris_train_ftrs</span><span class="p">,</span> <span class="n">iris_train_tgt</span><span class="p">)</span>
             <span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">iris_test_ftrs</span><span class="p">))</span><br><br><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;accuracy:&quot;</span><span class="p">,</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span> <span class="n">tgt_preds</span><span class="p">))</span>
<span class="nb">print</span><span class="p">()</span>
<span class="c1"># Print classifier</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span> <span class="n">tgt_preds</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;confusion matrix:&quot;</span><span class="p">,</span> <span class="n">cm</span><span class="p">,</span> <span class="n">sep</span><span class="o">=</span><span class="s2">&quot;</span><span class="se">\n</span><span class="s2">&quot;</span><span class="p">)</span><br><br><span class="c1"># Draw confusion matrix using seaborn</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                 <span class="n">xticklabels</span><span class="o">=</span><span class="n">iris</span><span class="o">.</span><span class="n">target_names</span><span class="p">,</span>
                 <span class="n">yticklabels</span><span class="o">=</span><span class="n">iris</span><span class="o">.</span><span class="n">target_names</span><span class="p">,</span>
                 <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">24</span><span class="p">})</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Confusion Matrix for the Iris Dataset&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>


    Size of full dataset: 150
    Size of test dataset: 50

    accuracy: 0.94

    confusion matrix:
    [[18  0  0]
     [ 0 16  1]
     [ 0  2 13]]




![png](output_7_1.png)



The _k_-NN classifier identified all setosa examples correctly. The classifier misclassified one versicolor as a setosa. Also, the classifier misclassified two virginicas as versicolors. The remaining 16 versicolors and 13 virginicas were classified correctly.

# Averaging Multiple Classes

---

When dealing with the Iris dataset, we are no longer dealing with two classes like we were with the MNIST dataset, causing our dichotomous formulas for `precision` and `recall` to break down. However, we can calculate something similar to `precision`, taking a one-versus-rest approach and comparing each class to all others. For the setosa class, we predict $\frac{18}{18} = 1$, versicolor, $\frac{16}{18}$, and virginica, $\frac{13}{14}$. Let's take the mean.


<div class="highlight"><pre><span></span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">16</span><span class="o">/</span><span class="mi">18</span><span class="p">,</span> <span class="mi">13</span><span class="o">/</span><span class="mi">14</span><span class="p">])</span>
</pre></div>





    0.9391534391534391



We calculate this same mean in `sklearn` by setting the `average` parameter of the `precision_score` function to `macro`.

## Macro Precision

---

To calculate the macro precision, we take the diagonal entry for each column in the confusion matrix and divide it by the total predictions per column. We then sum these values and divide the result by the total number of columns.


<div class="highlight"><pre><span></span><span class="n">macro_prec</span> <span class="o">=</span> <span class="n">precision_score</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span>
                             <span class="n">tgt_preds</span><span class="p">,</span>
                             <span class="n">average</span><span class="o">=</span><span class="s1">&#39;macro&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Macro Precision: </span><span class="si">{</span><span class="n">macro_prec</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span><br><br><span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span> <span class="n">tgt_preds</span><span class="p">)</span>
<span class="n">n_labels</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">target_names</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span>                                 <span class="c1"># correct     # column calc         # total cols</span>
    <span class="sa">f</span><span class="s2">&quot;Should Equal &#39;Macro Precision&#39;: </span><span class="si">{</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">cm</span><span class="p">)</span> <span class="o">/</span> <span class="n">cm</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">/</span> <span class="n">n_labels</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
</pre></div>


    Macro Precision: 0.9391534391534391
    Should Equal 'Macro Precision': 0.9391534391534391


## Micro Precision

---

The micro precision is named somewhat counterintuitively, providing a broader look at the results than the macro average. To calculate the micro precision, we divide all the correct predictions by the total number of predictions. We can perform this calculation manually by summing the values on the diagonal of the confusion matrix and dividing by the sum of all values in the confusion matrix.


<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Micro Precision:&quot;</span><span class="p">,</span> <span class="n">precision_score</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span> <span class="n">tgt_preds</span><span class="p">,</span> <span class="n">average</span><span class="o">=</span><span class="s1">&#39;micro&#39;</span><span class="p">))</span><br><br><span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span> <span class="n">tgt_preds</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Should Equal &#39;Micro Precision&#39;:&quot;</span><span class="p">,</span>
      <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">cm</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">/</span> <span class="n">cm</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
</pre></div>


    Micro Precision: 0.94
    Should Equal 'Micro Precision': 0.94


## Classification Report

---

The `classification_report` builds a text report showing some of these metrics. Let's take a look at what it returns.


<div class="highlight"><pre><span></span><span class="c1"># initial search param</span>
<span class="n">init_index</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">item</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span>
    <span class="n">classification_report</span><span class="o">.</span><span class="vm">__doc__</span><span class="o">.</span><span class="n">splitlines</span><span class="p">())</span> <span class="k">if</span> <span class="n">re</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="sa">r</span><span class="s1">&#39;Returns&#39;</span><span class="p">,</span> <span class="n">item</span><span class="p">)]</span><br><br><span class="c1"># find dash index from init_index</span>
<span class="n">dash_index</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">item</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">classification_report</span><span class="o">.</span><span class="vm">__doc__</span><span class="o">.</span><span class="n">splitlines</span><span class="p">()[</span>
                                         <span class="n">init_index</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">+</span> <span class="mi">2</span><span class="p">:])</span> <span class="k">if</span> <span class="n">re</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="sa">r</span><span class="s1">&#39;^\s+----+&#39;</span><span class="p">,</span> <span class="n">item</span><span class="p">)]</span><br><br><span class="c1"># add to the dash index to make it correct</span>
<span class="n">dash_index</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">+=</span> <span class="n">init_index</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><br><br><span class="c1"># print final __doc__ string</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">classification_report</span><span class="o">.</span><span class="vm">__doc__</span><span class="o">.</span><span class="n">splitlines</span><span class="p">()</span>
      <span class="p">[</span><span class="n">init_index</span><span class="p">[</span><span class="mi">0</span><span class="p">]:</span><span class="n">dash_index</span><span class="p">[</span><span class="mi">0</span><span class="p">]]))</span>
</pre></div>


        Returns
        -------
        report : str or dict
            Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the
            following structure::

                {'label 1': {'precision':0.5,
                             'recall':1.0,
                             'f1-score':0.67,
                             'support':1},
                 'label 2': { ... },
                  ...
                }

            The reported averages include macro average (averaging the unweighted
            mean per label), weighted average (averaging the support-weighted mean
            per label), and sample average (only for multilabel classification).
            Micro average (averaging the total true positives, false negatives and
            false positives) is only shown for multi-label or multi-class
            with a subset of classes, because it corresponds to accuracy
            otherwise and would be the same for all metrics.
            See also :func:`precision_recall_fscore_support` for more details
            on averages.

            Note that in binary classification, recall of the positive class
            is also known as "sensitivity"; recall of the negative class is
            "specificity".


According to the docs, the `classification_report` returns each class's `precision`, `recall`, and `f1-score`. We have yet to talk about the $F_1$ score, but we will shortly. The `classification_report` also returns the `macro` average, which is the unweighted mean per label--as described above in the section on `macro` precision--and the weighted average, which averages the support-weighted mean per category.

### Support

---

The support of a classification rule, such as "if x is a big house in a good location, then its value is high," is the count of the examples where the rule applies. In other words, if 100 of 1000 houses are big and in a good location, then the support is 10%. The `classification_report` is concerned with the "support in reality," equivalent to the sum of each row in the confusion matrix.

Before looking at some simple `classification_report` examples, let's look at the $F_1$ score.

# $F_1$ Score

---



The $F_1$ score is a standard measure to rate a classifier's success. It is also known as the balanced F-score or F-measure. The $F_1$ score is the harmonic mean of the precision and recall, reaching its best value at 1 and worst score at 0. We compute a harmonic mean by finding the arithmetic mean of the reciprocals of the data and then taking the reciprocal of that. The formula is below:

$$ H = \frac{n}{\frac{1}{x_1} + \frac{1}{x_2} + \frac{1}{x_3} + \ldots + \frac{1}{x_n}} $$

$H$ is the harmonic mean, $n$ is the number of data points, and $x_n$ is the nth value in the dataset.

Applying this formula to the precision and recall, we get the following:

$$ F_1 = \frac{2}{\frac{1}{\text{precision}} + \frac{1}{\text{recall}}} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{\text{tp}}{\text{tp} + \frac{1}{2}(\text{fp} + \text{fn})} $$

The precision and recall measure two types of errors we can make regarding the positive class. Maximizing the precision minimizes false positives, whereas maximizing the recall minimizes false negatives. The $F_1$ score represents an equal tradeoff between precision and recall, meaning we want to be accurate both in the value of our predictions and concerning reality.

Let's take a look at a few worst-case scenarios.

## Worst-Case Scenarios

---

If our classifier perfectly mispredicts all instances, we have zero `precision` and zero `recall`, resulting in a zero $F_1$ score. All negative predictions result in the same values for all three metrics.


<div class="highlight"><pre><span></span><span class="c1"># Worst Case</span>
<span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Perfect Mispredictions&#39;</span><span class="p">,</span> <span class="s1">&#39;All Negative Predicitons&#39;</span><span class="p">]</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]]</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span><br><br><span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="o">.</span><span class="n">flat</span><span class="p">,</span> <span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                     <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                     <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                     <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">,</span> <span class="n">cbar</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">24</span><span class="p">})</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">title</span><span class="p">)</span><br><br><span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">subplots_adjust</span><span class="p">(</span><span class="n">wspace</span><span class="o">=</span><span class="mf">0.3</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
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
        <span class="sa">f</span><span class="s1">&#39;Recall: </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Precision: </span><span class="si">{</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;F1-Score: </span><span class="si">{</span><span class="n">f1_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
</pre></div>




![png](output_21_0.png)



    1. Perfect Mispredictions
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    True Negatives: 0
    False Positives: 5
    True Positives: 0
    False Negatives: 5

    Recall: 0.000
    Precision: 0.000
    F1-Score: 0.000

    2. All Negative Predicitons
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    True Negatives: 5
    False Positives: 0
    True Positives: 0
    False Negatives: 5

    Recall: 0.000
    Precision: 0.000
    F1-Score: 0.000



Given that neither classifier output any positive predictions, the `recall`, `precision`, and `f1-score` are all 0.

## Best Case

---

Conversely, a perfect classifier outputs perfect `recall`, `precision`, and `f1-scores`.


<div class="highlight"><pre><span></span><span class="c1"># Best Case</span>
<span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Perfect Classifier&#39;</span><span class="p">]</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">),</span> <span class="n">squeeze</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><br><br><span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="o">.</span><span class="n">flat</span><span class="p">,</span> <span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                     <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                     <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                     <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">,</span> <span class="n">cbar</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">24</span><span class="p">})</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">title</span><span class="p">)</span><br><br><span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">subplots_adjust</span><span class="p">(</span><span class="n">wspace</span><span class="o">=</span><span class="mf">0.3</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
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
        <span class="sa">f</span><span class="s1">&#39;Recall: </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Precision: </span><span class="si">{</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;F1-Score: </span><span class="si">{</span><span class="n">f1_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
</pre></div>




![png](output_24_0.png)



    1. Perfect Classifier
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    True Negatives: 5
    False Positives: 0
    True Positives: 5
    False Negatives: 0

    Recall: 1.000
    Precision: 1.000
    F1-Score: 1.000



Before discussing a classifier with perfect recall and 50% precision, let's discuss the $F_{\beta}$ measure.

# $F_{\beta}$ Score

---


The $F_1$ score equally balances precision and recall. However, in some problems, we may be more interested in minimizing false negatives, such as when attempting to identify a rare disease. In other circumstances, we may be more interested in reducing false positives, such as when trying to classify web pages according to search engine requests.

To solve problems of this nature, we can use $F_{\beta}$, an abstraction of $F_1$ that allows us to control the balance of precision and recall using a coefficient, $\beta$.

$$ F_\beta = \frac{((1 + \beta^2) \times \text{Precision} \times \text{Recall})}{\beta^2 \times \text{Precision} + \text{Recall}} $$

Three common values for $\beta$ are as follows:

* $F_{0.5}$: places more weight on precision.
* $F_1$: places equal weight on precision and recall.
* $F_2$: places more weight on recall.

Let's take a closer look at each.

## $F_{0.5}$ Measure

---

$F_{0.5}$ raises the importance of precision and lowers the significance of recall, focusing more on minimizing false positives.

$$ F_{0.5} = \frac{((1 + 0.5^2) \times \text{Precision} \times \text{Recall})}{(0.5^2 \times \text{Precision} + \text{Recall})} = \frac{(1.25 \times \text{Precision} + \text{Recall})}{(0.25 \times \text{Precision} + \text{Recall})}$$

## $F_1$ Measure

---

The $F_1$ measure discussed above is an example of $F_{\beta}$ with a $\beta$ value of 1.

$$ F_{1} = \frac{((1 + 1^2) \times \text{Precision} \times \text{Recall})}{(1 ^ 2 \times \text{Precision} + \text{Recall})} = \frac{(2 \times \text{Precision} + \text{Recall})}{(\text{Precision} + \text{Recall})}$$

## $F_2$ Measure

---

The $F_2$ measure increases the significance of recall and lowers the importance of precision. The $F_{2}$ measure focuses more on minimizing false negatives.

$$ F_2 = \frac{((1 + 2^2) \times \text{Precision} \times \text{Recall})}{(2 ^ 2 \times \text{Precision} + \text{Recall})} = \frac{(5 \times \text{Precision} + \text{Recall})}{(4 \times \text{Precision} + \text{Recall})}$$


Because precision and recall require true positives, having one without the other is impossible.

Let's consider a case where a classifier makes predictions resulting in perfect recall and 50% precision.

We'll create a classifier that predicts all positives.

# Perfect Recall, 50% Precision

---



<div class="highlight"><pre><span></span><span class="c1">#50% Precision, Perfect Recall</span>
<span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;All Positives Predictor&#39;</span><span class="p">]</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span><br><br><span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">),</span> <span class="n">squeeze</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><br><br><span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="o">.</span><span class="n">flat</span><span class="p">,</span> <span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
    <span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                     <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                     <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                     <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">,</span> <span class="n">cbar</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">24</span><span class="p">})</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">title</span><span class="p">)</span><br><br><span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span><br><br><span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
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
        <span class="sa">f</span><span class="s1">&#39;Recall: </span><span class="si">{</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Precision: </span><span class="si">{</span><span class="n">precision_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;F0.5-Score: </span><span class="si">{</span><span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">beta</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;F1-Score: </span><span class="si">{</span><span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">beta</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span>
        <span class="sa">f</span><span class="s1">&#39;F2-Score: </span><span class="si">{</span><span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_t</span><span class="p">,</span> <span class="n">y_p</span><span class="p">,</span> <span class="n">beta</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
</pre></div>




![png](output_33_0.png)



    1. All Positives Predictor
    -----------------------------
    y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    True Negatives: 0
    False Positives: 5
    True Positives: 5
    False Negatives: 0

    Recall: 1.000
    Precision: 0.500

    F0.5-Score: 0.556
    F1-Score: 0.667
    F2-Score: 0.833



There are no false negatives (incorrect misses), so the `recall` is 1.0.
Because there are false positives, the `precision` (0.50) takes a hit, and ultimately the $F_1$ score (0.667).
Because the $F_{0.5}$ score (0.556) places more emphasis on `precision`, its value is lower than that of the $F_1$ score (0.667).
The $F_{2}$ score is the highest of all three `f-scores` (0.833) since it emphasizes `recall` over `precision`.

Let's look at some simple examples of the `classification_report` function.

# Classification Report - Simple Example #1

---



<div class="highlight"><pre><span></span><span class="c1"># Classification Report Simple Example 1</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>
<span class="n">target_names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;class 0&#39;</span><span class="p">,</span> <span class="s1">&#39;class 1&#39;</span><span class="p">,</span> <span class="s1">&#39;class 2&#39;</span><span class="p">]</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                 <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Class 0&#39;</span><span class="p">,</span> <span class="s1">&#39;Class 1&#39;</span><span class="p">,</span> <span class="s1">&#39;Class 2&#39;</span><span class="p">],</span>
                 <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;Class 0&#39;</span><span class="p">,</span> <span class="s1">&#39;Class 1&#39;</span><span class="p">,</span> <span class="s1">&#39;Class 2&#39;</span><span class="p">],</span>
                 <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">,</span> <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="mi">25</span><span class="p">})</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Confusion Matrix Simple Example #1&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Y True: </span><span class="si">{</span><span class="n">y_true</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Y Pred: </span><span class="si">{</span><span class="n">y_pred</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="n">target_names</span><span class="p">))</span>
</pre></div>




![png](output_37_0.png)



    Y True: [0, 1, 2, 2, 2]
    Y Pred: [0, 0, 2, 2, 1]

                  precision    recall  f1-score   support

         class 0       0.50      1.00      0.67         1
         class 1       0.00      0.00      0.00         1
         class 2       1.00      0.67      0.80         3

        accuracy                           0.60         5
       macro avg       0.50      0.56      0.49         5
    weighted avg       0.70      0.60      0.61         5



Regarding `Class 0`, the classifier predicted one instance correctly and incorrectly identified a real `1` as a `0`, resulting in a false positive. Consequently, the `precision` takes a hit (0.5), though the `recall` remains high at 1.0. The $F_1$ score is 0.67.

The classifier did not predict any instances of `Class 1` correctly, resulting in all metrics having values of 0.0.

The classifier did the best on `Class 2`, predicting two instances correctly and one as `Class 1`. As a result, there are no false positives, so the `precision` remains high at 1.0. There is one false negative, so the `recall` is reduced to 0.67, and the $F_1$ measure is 0.80.

# Classification Report - Simple Example #2

---


<div class="highlight"><pre><span></span><span class="c1"># Classification Report Simple Example #2</span>
<span class="n">y_true</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">set</span><span class="p">(</span><span class="n">font_scale</span><span class="o">=</span><span class="mf">2.0</span><span class="p">)</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                 <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                 <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;False&#39;</span><span class="p">,</span> <span class="s1">&#39;True&#39;</span><span class="p">],</span>
                 <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Confusion Matrix Simple Examples #2&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Y True: </span><span class="si">{</span><span class="n">y_true</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Y Pred: </span><span class="si">{</span><span class="n">y_pred</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">zero_division</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
</pre></div>




![png](output_40_0.png)



    Y True: [1, 1, 1]
    Y Pred: [0, 1, 1]

                  precision    recall  f1-score   support

               0       0.00      0.00      0.00         0
               1       1.00      0.67      0.80         3

        accuracy                           0.67         3
       macro avg       0.50      0.33      0.40         3
    weighted avg       1.00      0.67      0.80         3



There are no instances of `Class 0`, resulting in `precision`, `recall`, and `f1-scores` of 0.

The classifier achieves perfect `precision` (1.0) on the positive class, never predicting any false positives. The `recall` (0.67) takes a hit because there is one false negative.

The `macro avg` considers both classes in its `precision`, `recall`, and `f1-score` calculations.

In contrast, the `weighted avg` only considers the positive class since the support weights the calculation according to reality.

This is why class `1` and the `weighted avg` rows contain equal values for `precision` (1.00), `recall` (0.67), and the `f1-score` (0.80).

Let's apply the `classification_report` function to the _k_-NN classifier we built on the Iris dataset.

# Classification Report - Iris Dataset

---


<div class="highlight"><pre><span></span><span class="c1"># Classification report for Iris dataset</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span> <span class="n">tgt_preds</span><span class="p">)</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">square</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                 <span class="n">xticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;setosa&#39;</span><span class="p">,</span> <span class="s1">&#39;versicolor&#39;</span><span class="p">,</span> <span class="s1">&#39;virginica&#39;</span><span class="p">],</span>
                 <span class="n">yticklabels</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;setosa&#39;</span><span class="p">,</span> <span class="s1">&#39;versicolor&#39;</span><span class="p">,</span> <span class="s1">&#39;virginica&#39;</span><span class="p">],</span>
                 <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Confusion Matrix for the Iris Dataset&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">iris_test_tgt</span><span class="p">,</span> <span class="n">tgt_preds</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;setosa&#39;</span><span class="p">,</span> <span class="s1">&#39;versicolor&#39;</span><span class="p">,</span> <span class="s1">&#39;virginica&#39;</span><span class="p">]))</span>
</pre></div>




![png](output_44_0.png)



                  precision    recall  f1-score   support

          setosa       1.00      1.00      1.00        18
      versicolor       0.89      0.94      0.91        17
       virginica       0.93      0.87      0.90        15

        accuracy                           0.94        50
       macro avg       0.94      0.94      0.94        50
    weighted avg       0.94      0.94      0.94        50



We see the setosa class is the easiest to predict, and the _k_-NN classifier predicted all examples in the test dataset successfully. The `precision`, `recall`, and `f1-score` for this row are 1.00.

The classifier is slightly less precise on the versicolor class (0.89) since it misclassifies two virginicas as versicolors, as compared to the virginica class (0.93), where it only misclassifies one versicolor as a virginica.

The `recall` for the versicolor class (0.94) is higher than that for the virginica class (0.87) because the classifier only misclassified one versicolor, whereas it misclassified two virginicas.

The resulting `f1-scores` for versicolor and virginica are 0.91 and 0.90, respectively. Versicolor is slightly above virginica, indicating that the classifier performed slightly better on that class.

The `macro avg` and `weighted avg` for the `precision`, `recall`, and `f1-score` are equal (0.94), indicating the _k_-NN classifier performs relatively well, and the classes are distributed relatively equally amongst the target variable.

In the next post, we will discuss ROC curves, otherwise known as _receiver operating characteristic_ curves.
