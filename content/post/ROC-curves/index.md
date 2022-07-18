---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "ROC Curves"
subtitle: "Plots and Discusses ROC curves for Binary and Multiclass Classification Problems Using the Iris Dataset"
summary: "Plots and Discusses ROC curves for Binary and Multiclass Classification Problems Using the Iris Dataset"
authors: ["Scott Miner"]
tags: ["ROC Curves", "Receiver Operating Characteristic", "Multiclass Classification", "Confusion Matrix", "Confusion Matrices", "Thresholds", "Sklearn", "Scikit-Learn", "Python", "Iris Dataset", "One-versus-Rest", "Classifiers"]
categories: ["ROC Curves", "Receiver Operating Characteristic", "Multiclass Classification", "Thresholds", "Confusion Matrix", "Confusion Matricies", "Sklearn", "Scikit-Learn", "Python", "Iris Dataset", "One-versus-Rest", "Classifiers"]
date: 2022-07-17T22:26:35.413282
lastmod: 2022-07-17T22:26:35.413282
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

# Overview

---

This post takes off where the [last one](https://scottminer.netlify.app/post/multi-class-confusion-matrix/) left off and discusses ROC curves, otherwise known as Receiver Operating Characteristic curves. ROC curves have a long history in classification. ROC curves were first used in World War II to quantify the radar tracking of bombers headed toward England.

ROC curves typically feature the true positive rate (TPR) on the Y axis and the false positive rate (FPR) on the X axis. The TPR is also known as the sensitivity. The FPR can be calculated by subtracting the specificity from 1: $\text{FPR} = 1 - \text{specificity}$. Both the TPR and FPR measure the classifier's performance with respect to the real world. TPR and FPR values of 1.0 and 0.0, respectively, indicate a perfect classifier.

In `sklearn`, the `roc_curve` function from the `metrics` module computes ROC curves via the `y_true` and `y_score` parameters. The `y_true` parameter accepts true binary labels as `ndarrays` of shape `(n_samples,)`. If the labels are not either `{-1, 1}` or `{0, 1}` then `pos_label` (positive label) should be set explicitly. Meanwhile, the `y_score` parameter accepts target scores as `ndarrays` of shape `(n_samples,)`. The target scores can be probability estimates of the positive class, confidence values, or non-thresholded measures of decisions (as returned by the `decision_function` of some classifiers).

The implementation of the `roc_curve` function in `sklearn` is restricted to binary classification. To demonstrate it, let's convert the Iris dataset to a binary classification problem. Instead of asking, "Is the Iris class that of setosa, versicolor, or virginica?" we ask whether the sample is versicolor. We use the `predict_proba` method rather than the `predict` method to return the false and true probabilities in two columns. For the ROC curve, we are interested in the true column, which is the second column.

Let's import the needed libraries and look at an example.

# Imports

---



<div class="highlight"><pre><span></span><span class="c1"># imports</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">textwrap</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="k">as</span> <span class="nn">mpl</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">warnings</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">datasets</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">metrics</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">roc_curve</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span><span class="p">,</span> <span class="n">cross_val_score</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">GaussianNB</span>
<span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="kn">import</span> <span class="n">label_binarize</span><span class="p">,</span> <span class="n">LabelEncoder</span>
<span class="kn">from</span> <span class="nn">sklearn.multiclass</span> <span class="kn">import</span> <span class="n">OneVsRestClassifier</span>
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


# Iris Dataset - Binary Classification Problem

---

After converting the problem to a binary classification problem indicating whether the target variable is that of the Iris veriscolor class, we split the data into training and test sets. We then use the `GaussianNB` method of the `naive_bayes` module to fit our classifier on the Iris dataset's training features and target variable. We get the likelihood of each prediction being true by applying the `predict_proba` method to the dataset's test features and storing the second column of the result in the `prob_true` variable.


<div class="highlight"><pre><span></span><span class="c1"># Load the iris dataset</span>
<span class="n">iris</span> <span class="o">=</span> <span class="n">datasets</span><span class="o">.</span><span class="n">load_iris</span><span class="p">()</span><br><br><span class="c1"># Convert to one class</span>
<span class="n">is_versicolor</span> <span class="o">=</span> <span class="n">iris</span><span class="o">.</span><span class="n">target</span> <span class="o">==</span> <span class="mi">1</span><br><br><span class="c1"># Train test split</span>
<span class="n">tts_1c</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="n">is_versicolor</span><span class="p">,</span>
                          <span class="n">test_size</span><span class="o">=</span><span class="mf">.33</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">21</span><span class="p">)</span><br><br><span class="p">(</span><span class="n">iris_1c_train_ftrs</span><span class="p">,</span> <span class="n">iris_1c_test_ftrs</span><span class="p">,</span>
 <span class="n">iris_1c_train_tgt</span><span class="p">,</span> <span class="n">iris_1c_test_tgt</span><span class="p">)</span> <span class="o">=</span> <span class="n">tts_1c</span><br><br><span class="c1"># build, fit, predict (probability scores) for NB model</span>
<span class="n">gnb</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">prob_true</span> <span class="o">=</span> <span class="p">(</span><span class="n">gnb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">iris_1c_train_ftrs</span><span class="p">,</span> <span class="n">iris_1c_train_tgt</span><span class="p">)</span>
                <span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">iris_1c_test_ftrs</span><span class="p">)[:,</span> <span class="mi">1</span><span class="p">])</span>  <span class="c1"># [:, 1]==&quot;true&quot;</span>
</pre></div>


We then pipe the `prob_true` variable into the `roc_curve` function to draw the ROC curve.


<div class="highlight"><pre><span></span><span class="c1"># Draw the ROC curve</span>
<span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">thresh</span> <span class="o">=</span> <span class="n">roc_curve</span><span class="p">(</span><span class="n">iris_1c_test_tgt</span><span class="p">,</span> <span class="n">prob_true</span><span class="p">)</span><br><br><span class="n">auc_score</span> <span class="o">=</span> <span class="n">metrics</span><span class="o">.</span><span class="n">auc</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">)</span><br><br><span class="nb">print</span><span class="p">(</span><span class="s1">&#39;False Positive Rate:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">textwrap</span><span class="o">.</span><span class="n">fill</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="p">[</span><span class="s1">&#39;</span><span class="si">%.2f</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="n">elem</span> <span class="k">for</span> <span class="n">elem</span> <span class="ow">in</span> <span class="n">fpr</span><span class="p">]</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">width</span><span class="o">=</span><span class="mi">120</span><span class="p">))</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;True Positive Rate:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">textwrap</span><span class="o">.</span><span class="n">fill</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;</span><span class="si">{</span><span class="n">tpr</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">,</span> <span class="n">width</span><span class="o">=</span><span class="mi">120</span><span class="p">))</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Threshold:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">textwrap</span><span class="o">.</span><span class="n">fill</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">thresh</span><span class="p">),</span> <span class="n">width</span><span class="o">=</span><span class="mi">180</span><span class="p">))</span><br><br><span class="c1"># Create the ROC plot</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="s1">&#39;o--&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;1-Class Iris ROC Curve</span><span class="se">\n</span><span class="s1">AUC: </span><span class="si">{</span><span class="n">auc_score</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;FPR&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;TPR&quot;</span><span class="p">)</span><br><br><span class="n">investigate</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="k">for</span> <span class="n">idx</span> <span class="ow">in</span> <span class="n">investigate</span><span class="p">:</span>
    <span class="n">th</span><span class="p">,</span> <span class="n">f</span><span class="p">,</span> <span class="n">t</span> <span class="o">=</span> <span class="n">thresh</span><span class="p">[</span><span class="n">idx</span><span class="p">],</span> <span class="n">fpr</span><span class="p">[</span><span class="n">idx</span><span class="p">],</span> <span class="n">tpr</span><span class="p">[</span><span class="n">idx</span><span class="p">]</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">annotate</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;thresh = </span><span class="si">{</span><span class="n">th</span><span class="si">:</span><span class="s1">.3f</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">,</span> <span class="n">xy</span><span class="o">=</span><span class="p">(</span><span class="n">f</span><span class="o">+</span><span class="mf">.01</span><span class="p">,</span> <span class="n">t</span><span class="o">-</span><span class="mf">.01</span><span class="p">),</span> <span class="n">xytext</span><span class="o">=</span><span class="p">(</span><span class="n">f</span><span class="o">+</span><span class="mf">.1</span><span class="p">,</span> <span class="n">t</span><span class="p">),</span>
                <span class="n">arrowprops</span><span class="o">=</span><span class="p">{</span><span class="s1">&#39;arrowstyle&#39;</span><span class="p">:</span> <span class="s1">&#39;-&gt;&#39;</span><span class="p">,</span> <span class="s1">&#39;color&#39;</span><span class="p">:</span> <span class="s1">&#39;black&#39;</span><span class="p">},</span> <span class="n">size</span><span class="o">=</span><span class="mi">17</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>


    False Positive Rate:
    ['0.00', '0.00', '0.00', '0.06', '0.06', '0.12', '0.12', '0.18', '1.00']
    
    True Positive Rate:
    [0.         0.05882353 0.88235294 0.88235294 0.94117647 0.94117647  1.         1.         1.        ]
    
    Threshold:
    [1.97986528e+00 9.79865278e-01 3.93653599e-01 3.19020482e-01  2.63886933e-01 1.15989808e-01 1.06081289e-01 4.99637540e-02  4.93114034e-20]
    


    
![png](output_10_1.png)
    


Most of the FPR values are between 0.0 and 0.2, while the TPR values quickly jump into the range of 0.9 to 1.0. Each point represents a different confusion matrix based on its unique threshold. The following shows the confusion matrices for all the thresholds returned by the `roc_curve` function.


<div class="highlight"><pre><span></span><span class="c1"># Threshold confusion matrices</span><br><br><span class="c1"># How many thresholds were there, anyway?</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s1">&#39;Total thresholds: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">thresh</span><span class="p">)</span><span class="si">}</span><span class="s1">&#39;</span><span class="p">)</span><br><br><span class="n">title_fmt</span> <span class="o">=</span> <span class="s2">&quot;Threshold </span><span class="si">{}</span><span class="se">\n</span><span class="s2">~</span><span class="si">{:5.3f}</span><span class="se">\n</span><span class="s2">TPR : </span><span class="si">{:.3f}</span><span class="se">\n</span><span class="s2">FPR : </span><span class="si">{:.3f}</span><span class="s2">&quot;</span><br><br><span class="n">np</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Negative&#39;</span><span class="p">,</span> <span class="s1">&#39;Positive&#39;</span><span class="p">]</span>
<span class="n">add_args</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;xticklabels&#39;</span><span class="p">:</span> <span class="n">np</span><span class="p">,</span>
            <span class="s1">&#39;yticklabels&#39;</span><span class="p">:</span> <span class="n">np</span><span class="p">,</span>
            <span class="s1">&#39;square&#39;</span><span class="p">:</span> <span class="kc">True</span><span class="p">}</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">20</span><span class="p">,</span> <span class="mi">20</span><span class="p">))</span>
<span class="k">for</span> <span class="n">ax</span><span class="p">,</span> <span class="n">thresh_idx</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="o">.</span><span class="n">flat</span><span class="p">,</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">thresh</span><span class="p">))):</span>
    <span class="n">preds_at_th</span> <span class="o">=</span> <span class="n">prob_true</span> <span class="o">&gt;=</span> <span class="n">thresh</span><span class="p">[</span><span class="n">thresh_idx</span><span class="p">]</span>
    <span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">iris_1c_test_tgt</span><span class="p">,</span> <span class="n">preds_at_th</span><span class="p">)</span>
    <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">cbar</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;flare&#39;</span><span class="p">,</span>
                <span class="n">annot_kws</span><span class="o">=</span><span class="p">{</span><span class="s1">&#39;size&#39;</span><span class="p">:</span> <span class="mi">24</span><span class="p">},</span> <span class="o">**</span><span class="n">add_args</span><span class="p">)</span><br><br>    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Actual&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">title_fmt</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">thresh_idx</span><span class="p">,</span>
                                  <span class="n">thresh</span><span class="p">[</span><span class="n">thresh_idx</span><span class="p">],</span>
                                  <span class="n">tpr</span><span class="p">[</span><span class="n">thresh_idx</span><span class="p">],</span>
                                  <span class="n">fpr</span><span class="p">[</span><span class="n">thresh_idx</span><span class="p">]))</span><br><br><span class="c1">#axes[0].set_ylabel(&#39;Actual&#39;)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">(</span><span class="n">pad</span><span class="o">=</span><span class="mi">4</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>


    Total thresholds: 9
    


    
![png](output_12_1.png)
    


As we lower the threshold, the classifier predicts more positive examples, and the TPR increases. To choose a threshold, we can identify an acceptable TPR and then choose the threshold that gets the best FPR. Alternatively, we can perform the opposite action if we care more about the FPR than the TPR, as may be the case with identifying web pages from search engine requests, for example, where we do not want the classifier to return any incorrect web pages.

# AUC: Area-Under-the-(ROC)-Curve

---

The area under the curve (AUC) metric condenses the ROC curve into a single value. A diagonal line on a ROC curve generates an AUC value of 0.5, representing a classifier that makes predictions based on random coin flips. In contrast, a line that traces the perimeter of the graph generates an AUC value of 1.0, representing a perfect classifier.

We should approach the AUC metric cautiously because it is an overall measure of a classifier's performance at a series of thresholds, summarizing a lot of information into the subtlety of a single number. On the one hand, the number neglects that a classifier's behavior and rank order may change at any threshold. On the other hand, a single-valued AUC offers the benefit of efficiently computing other statistics from it and summarizing them graphically. For instance, let's look at several cross-validated AUCs displayed simultaneously on a strip plot.


<div class="highlight"><pre><span></span><span class="c1"># Cross-validated AUCs on strip plot</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
<span class="n">cv_auc</span> <span class="o">=</span> <span class="n">cross_val_score</span><span class="p">(</span>
    <span class="n">model</span><span class="p">,</span> <span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span> <span class="n">iris</span><span class="o">.</span><span class="n">target</span> <span class="o">==</span> <span class="mi">1</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">&#39;roc_auc&#39;</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span><br><br><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span> <span class="mi">6</span><span class="p">))</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">swarmplot</span><span class="p">(</span><span class="n">y</span><span class="o">=</span><span class="n">cv_auc</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="mf">10.0</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;10-Fold AUCs&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>



    
![png](output_15_0.png)
    


From this summarization, we can see that many folds return perfect results.

# Iris Dataset - Multiclass Classification

---

The `roc_curve` function from the `metrics` module is designed for use on binary classification problems. Concerning multiclass classification problems, one approach is to re-code the dataset into a series of one-versus-rest (OvR) alternatives. In such scenarios, the classifier considers each target class compared to all the others.
* 0 versus [1, 2]
* 1 versus [0, 2]
* 2 versus [0, 1]

The OvR multiclass strategy, also known as the one-vs-all strategy, offers computational efficiency compared to other techniques since only `n_classes` classifiers are needed. Additionally, the OvR strategy offers readily interpretable results, meaning we can gain knowledge about each class by inspecting its corresponding classifier since a single classifier represents a single class. Finally, the OvR strategy is the most commonly used method for multiclass classification problems and is what the `sklearn` documentation describes as a suitable default starting choice.

Before using our dataset with the `OneVsRestClassifier` function from the `multiclass` module, we must preprocess it using the `label_binarize` function from the `preprocessing` module. Let's look at samples 0, 50, and 100 from the original multiclass Iris dataset. We look at the samples' original encoding and view how the `label_binarize` function re-encodes them.


<div class="highlight"><pre><span></span><span class="c1"># label_binarize on Iris dataset samples</span>
<span class="n">checkout</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">50</span><span class="p">,</span> <span class="mi">100</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Original Encoding&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">[</span><span class="n">checkout</span><span class="p">])</span>
<span class="nb">print</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;&#39;Multi-label&#39; Encoding&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">label_binarize</span><span class="p">(</span>
    <span class="n">y</span><span class="o">=</span><span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">,</span> <span class="n">classes</span><span class="o">=</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">])[</span><span class="n">checkout</span><span class="p">])</span>
</pre></div>


    Original Encoding
    [0 1 2]
    
    'Multi-label' Encoding
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    

It may be a little easier to interpret these results when viewed as a DataFrame since the DataFrame provides column headers.


<div class="highlight"><pre><span></span><span class="c1"># print label_binarize dataframe</span>
<span class="n">d</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;0&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;1&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">],</span>
     <span class="s1">&#39;1&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">],</span>
     <span class="s1">&#39;2&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">]}</span><br><br><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">d</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">,</span> <span class="s1">&#39;2&#39;</span><span class="p">])</span>
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's look at a few more examples from the `sklearn` documentation. 


<div class="highlight"><pre><span></span><span class="c1"># label_binarize simple example #1</span>
<span class="nb">print</span><span class="p">(</span><span class="n">label_binarize</span><span class="p">(</span><span class="n">y</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">6</span><span class="p">],</span> <span class="n">classes</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">6</span><span class="p">]))</span><br><br><span class="n">d</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;1&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;1&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">],</span>
     <span class="s1">&#39;2&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">],</span>
     <span class="s1">&#39;4&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;0&#39;</span><span class="p">],</span>
     <span class="s1">&#39;6&#39;</span><span class="p">:</span> <span class="p">[</span><span class="s1">&#39;0&#39;</span><span class="p">,</span> <span class="s1">&#39;1&#39;</span><span class="p">]}</span><br><br><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">d</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;1&#39;</span><span class="p">,</span> <span class="s1">&#39;6&#39;</span><span class="p">])</span>
<span class="n">df</span>
</pre></div>


    [[1 0 0 0]
     [0 0 0 1]]
    




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
      <th>1</th>
      <th>2</th>
      <th>4</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The binary indicator (`1`) appears wherever the row labels form a cross-section with the column header. The `label_binarize` function also preserves the ordering of the classes, as in the following example:


<div class="highlight"><pre><span></span><span class="c1"># label_binarize preserve class ordering</span>
<span class="n">label_binarize</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">6</span><span class="p">],</span> <span class="n">classes</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>
</pre></div>





    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])



At last, binary targets transform to column vectors:


<div class="highlight"><pre><span></span><span class="c1"># label_binarize w/ binary targets</span>
<span class="n">label_binarize</span><span class="p">([</span><span class="s1">&#39;yes&#39;</span><span class="p">,</span> <span class="s1">&#39;no&#39;</span><span class="p">,</span> <span class="s1">&#39;no&#39;</span><span class="p">,</span> <span class="s1">&#39;yes&#39;</span><span class="p">],</span> <span class="n">classes</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;no&#39;</span><span class="p">,</span> <span class="s1">&#39;yes&#39;</span><span class="p">])</span>
</pre></div>





    array([[1],
           [0],
           [0],
           [1]])



Concerning the Iris dataset, after re-encoding the target class using the `label_binarize` function, we can look at the individual performance of the three classifiers, comparing their ROC curves. Let's create a _k_-NN classifier setting the `n_neighbors` to 5. We use the `predict_proba` function to calculate the probabilities of each class occurring after wrapping the estimator in the `OneVsRestClassifer` method from the `multiclass` module. Then we plot the ROC curves for each OvR classifier, allowing us to compare them all side-by-side.



<div class="highlight"><pre><span></span><span class="c1"># Iris dataset multiclass classification</span>
<span class="n">iris_multi_tgt</span> <span class="o">=</span> <span class="n">label_binarize</span><span class="p">(</span>
    <span class="n">y</span><span class="o">=</span><span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">,</span> <span class="n">classes</span><span class="o">=</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span><br><br><span class="n">le</span> <span class="o">=</span> <span class="n">LabelEncoder</span><span class="p">()</span>
<span class="n">le</span><span class="o">.</span><span class="n">fit</span><span class="p">([</span><span class="s1">&#39;Setosa&#39;</span><span class="p">,</span> <span class="s1">&#39;Versicolor&#39;</span><span class="p">,</span> <span class="s1">&#39;Virginica&#39;</span><span class="p">])</span><br><br><span class="p">(</span><span class="n">iris_multi_train_ftrs</span><span class="p">,</span> <span class="n">iris_multi_test_ftrs</span><span class="p">,</span>
 <span class="n">iris_multi_train_tgt</span><span class="p">,</span> <span class="n">iris_multi_test_tgt</span><span class="p">)</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span>
                                                               <span class="n">iris_multi_tgt</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">.33</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">21</span><span class="p">)</span><br><br><span class="c1"># knn wrapped up in one-versus-rest (3 classifiers)</span>
<span class="n">knn</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
<span class="n">ovr_knn</span> <span class="o">=</span> <span class="n">OneVsRestClassifier</span><span class="p">(</span><span class="n">knn</span><span class="p">)</span>
<span class="n">pred_probs</span> <span class="o">=</span> <span class="p">(</span><span class="n">ovr_knn</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">iris_multi_train_ftrs</span><span class="p">,</span> <span class="n">iris_multi_train_tgt</span><span class="p">)</span>
              <span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">iris_multi_test_ftrs</span><span class="p">))</span><br><br><span class="c1"># make ROC plots</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
<span class="k">for</span> <span class="bp">cls</span> <span class="ow">in</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">]:</span>
    <span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">roc_curve</span><span class="p">(</span><span class="n">iris_multi_test_tgt</span><span class="p">[:,</span> <span class="bp">cls</span><span class="p">],</span>
                            <span class="n">pred_probs</span><span class="p">[:,</span> <span class="bp">cls</span><span class="p">])</span><br><br>    <span class="n">label</span> <span class="o">=</span> <span class="sa">f</span><span class="s1">&#39;Class </span><span class="si">{</span><span class="n">le</span><span class="o">.</span><span class="n">inverse_transform</span><span class="p">([</span><span class="bp">cls</span><span class="p">])[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s1"> vs Rest (AUC = </span><span class="si">{</span><span class="n">metrics</span><span class="o">.</span><span class="n">auc</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span><span class="n">tpr</span><span class="p">)</span><span class="si">:</span><span class="s1">.2f</span><span class="si">}</span><span class="s1">)&#39;</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="s1">&#39;o--&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">label</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span><br><br><span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;FPR&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;TPR&quot;</span><span class="p">)</span><br><br><span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>



    
![png](output_28_0.png)
    


We see that all three classifiers perform reasonably well. As we saw previously, the setosa class is the easiest to predict. The remaining classifiers achieve excellent TPR rates (>= .75) at minimal FPR rates (<= .18). We will discuss another approach to multiclass classification in the next post, known as the One-versus-One approach.
