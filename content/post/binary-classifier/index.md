---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Training a Binary Classifier"
subtitle: "Trains an SGD Classifier on the MNIST Dataset"
summary: "Trains an SGD Classifier on the MNIST Dataset"
authors: ["Scott Miner"]
tags: ["Machine Learning", "Artificial Intelligence", "Sklearn", "Scikit-Learn", "Python", "MNIST"]
categories: ["Machine Learning", "Artificial Intelligence", "Sklearn", "Scikit-Learn", "Python", "MNIST"]
date: 2022-05-30T01:24:19.362213
lastmod: 2022-05-30T01:24:19.362213
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

This post discusses training a binary classifier on the MNIST dataset.


<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">os</span>
<span class="o">%</span><span class="n">matplotlib</span> <span class="n">inline</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="k">as</span> <span class="nn">mpl</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="n">mpl</span><span class="o">.</span><span class="n">rc</span><span class="p">(</span><span class="s1">&#39;axes&#39;</span><span class="p">,</span> <span class="n">labelsize</span><span class="o">=</span><span class="mi">14</span><span class="p">)</span>
<span class="n">mpl</span><span class="o">.</span><span class="n">rc</span><span class="p">(</span><span class="s1">&#39;xtick&#39;</span><span class="p">,</span> <span class="n">labelsize</span><span class="o">=</span><span class="mi">12</span><span class="p">)</span>
<span class="n">mpl</span><span class="o">.</span><span class="n">rc</span><span class="p">(</span><span class="s1">&#39;ytick&#39;</span><span class="p">,</span> <span class="n">labelsize</span><span class="o">=</span><span class="mi">12</span><span class="p">)</span>
<span class="c1"># Where to save the figures</span>
<span class="n">PROJECT_ROOT_DIR</span> <span class="o">=</span> <span class="s2">&quot;.&quot;</span>
<span class="n">NOTEBOOK_NAME</span> <span class="o">=</span> <span class="s2">&quot;binary-classifier&quot;</span>
<span class="n">IMAGES_PATH</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">PROJECT_ROOT_DIR</span><span class="p">,</span> <span class="s2">&quot;images&quot;</span><span class="p">,</span> <span class="n">NOTEBOOK_NAME</span><span class="p">)</span>
<span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">IMAGES_PATH</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><br><br><span class="k">def</span> <span class="nf">save_fig</span><span class="p">(</span><span class="n">fig_id</span><span class="p">,</span> <span class="n">tight_layout</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">fig_extension</span><span class="o">=</span><span class="s2">&quot;png&quot;</span><span class="p">,</span> <span class="n">resolution</span><span class="o">=</span><span class="mi">300</span><span class="p">):</span>
    <span class="n">path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">IMAGES_PATH</span><span class="p">,</span> <span class="n">fig_id</span> <span class="o">+</span> <span class="s2">&quot;.&quot;</span> <span class="o">+</span> <span class="n">fig_extension</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Saving figure&quot;</span><span class="p">,</span> <span class="n">fig_id</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">tight_layout</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">savefig</span><span class="p">(</span><span class="n">path</span><span class="p">,</span> <span class="nb">format</span><span class="o">=</span><span class="n">fig_extension</span><span class="p">,</span> <span class="n">dpi</span><span class="o">=</span><span class="n">resolution</span><span class="p">)</span>
</pre></div>


The following takes a little over 1 minute to complete.


<div class="highlight"><pre><span></span><span class="c1"># fetch the MNIST dataset</span>
<span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="kn">import</span> <span class="n">fetch_openml</span>
<span class="n">mnist</span> <span class="o">=</span> <span class="n">fetch_openml</span><span class="p">(</span><span class="s1">&#39;mnist_784&#39;</span><span class="p">,</span> <span class="n">version</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">as_frame</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="n">mnist</span><span class="o">.</span><span class="n">keys</span><span class="p">()</span>
</pre></div>





    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])



Let's look at the data shape.




<div class="highlight"><pre><span></span><span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">mnist</span><span class="p">[</span><span class="s2">&quot;data&quot;</span><span class="p">],</span> <span class="n">mnist</span><span class="p">[</span><span class="s2">&quot;target&quot;</span><span class="p">]</span>
<span class="n">X</span><span class="o">.</span><span class="n">shape</span>
</pre></div>





    (70000, 784)




<div class="highlight"><pre><span></span><span class="n">y</span><span class="o">.</span><span class="n">shape</span>
</pre></div>





    (70000,)




<div class="highlight"><pre><span></span><span class="mi">28</span> <span class="o">*</span> <span class="mi">28</span>
</pre></div>





    784



The above shows that there are 70,000 images and that each picture has 784 features. The reason for this is that each image is 28x28 pixels. Each column represents one pixel's intensity from 0 (white) to 255 (black).


<div class="highlight"><pre><span></span><span class="n">X</span><span class="o">.</span><span class="n">dtype</span>
</pre></div>





    dtype('float64')




<div class="highlight"><pre><span></span><span class="n">dt</span> <span class="o">=</span> <span class="n">y</span><span class="o">.</span><span class="n">dtype</span>
<span class="n">dt</span>
</pre></div>





    dtype('O')




<div class="highlight"><pre><span></span><span class="n">dt</span><span class="o">.</span><span class="n">itemsize</span>
</pre></div>





    8




<div class="highlight"><pre><span></span><span class="n">dt</span><span class="o">.</span><span class="n">name</span>
</pre></div>





    'object'




<div class="highlight"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="o">*</span><span class="p">[[</span><span class="n">feature</span> <span class="k">for</span> <span class="n">feature</span> <span class="ow">in</span> <span class="n">mnist</span><span class="o">.</span><span class="n">feature_names</span><span class="p">][:</span><span class="mi">5</span><span class="p">],[</span><span class="n">feature</span> <span class="k">for</span> <span class="n">feature</span> <span class="ow">in</span> <span class="n">mnist</span><span class="o">.</span><span class="n">feature_names</span><span class="p">][</span><span class="o">-</span><span class="mi">5</span><span class="p">:]])</span>
</pre></div>


    ['pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5'] ['pixel780', 'pixel781', 'pixel782', 'pixel783', 'pixel784']



<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">mnist</span><span class="o">.</span><span class="n">data</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">],</span> <span class="n">columns</span><span class="o">=</span><span class="n">mnist</span><span class="o">.</span><span class="n">feature_names</span><span class="p">)</span>
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




<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="n">unique</span><span class="p">,</span> <span class="n">counts</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">unique</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="n">return_counts</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">c_</span><span class="p">[</span><span class="n">unique</span><span class="p">,</span> <span class="n">counts</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">unique</span><span class="p">,</span> <span class="n">counts</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;MNIST Dataset&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Target Class&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>


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




![png](output_15_1.png)



Let's print one of the digits.


<div class="highlight"><pre><span></span><span class="o">%</span><span class="n">matplotlib</span> <span class="n">inline</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="k">as</span> <span class="nn">mpl</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span><br><br><span class="n">some_digit</span> <span class="o">=</span> <span class="n">X</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">some_digit_image</span> <span class="o">=</span> <span class="n">some_digit</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span> <span class="mi">28</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">some_digit_image</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">mpl</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">binary</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s2">&quot;off&quot;</span><span class="p">)</span><br><br><span class="n">save_fig</span><span class="p">(</span><span class="s2">&quot;some_digit_plot&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>


    Saving figure some_digit_plot




![png](output_17_1.png)




<div class="highlight"><pre><span></span><span class="n">y</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
</pre></div>





    '5'




<div class="highlight"><pre><span></span><span class="n">y</span> <span class="o">=</span> <span class="n">y</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">uint8</span><span class="p">)</span>
</pre></div>


Let's look at some more images from the dataset.


<div class="highlight"><pre><span></span><span class="k">def</span> <span class="nf">plot_digits</span><span class="p">(</span><span class="n">instances</span><span class="p">,</span> <span class="n">images_per_row</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span> <span class="o">**</span><span class="n">options</span><span class="p">):</span>
    <span class="n">size</span> <span class="o">=</span> <span class="mi">28</span>
    <span class="n">images_per_row</span> <span class="o">=</span> <span class="nb">min</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">instances</span><span class="p">),</span> <span class="n">images_per_row</span><span class="p">)</span>
    <span class="c1"># This is equivalent to n_rows = ceil(len(instances) / images_per_row):</span>
    <span class="n">n_rows</span> <span class="o">=</span> <span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">instances</span><span class="p">)</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span> <span class="o">//</span> <span class="n">images_per_row</span> <span class="o">+</span> <span class="mi">1</span><br><br>    <span class="c1"># Append empty images to fill the end of the grid, if needed:</span>
    <span class="n">n_empty</span> <span class="o">=</span> <span class="n">n_rows</span> <span class="o">*</span> <span class="n">images_per_row</span> <span class="o">-</span> <span class="nb">len</span><span class="p">(</span><span class="n">instances</span><span class="p">)</span>
    <span class="n">padded_instances</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">([</span><span class="n">instances</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="n">n_empty</span><span class="p">,</span> <span class="n">size</span> <span class="o">*</span> <span class="n">size</span><span class="p">))],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><br><br>    <span class="c1"># Reshape the array so it&#39;s organized as a grid containing 28×28 images:</span>
    <span class="n">image_grid</span> <span class="o">=</span> <span class="n">padded_instances</span><span class="o">.</span><span class="n">reshape</span><span class="p">((</span><span class="n">n_rows</span><span class="p">,</span> <span class="n">images_per_row</span><span class="p">,</span> <span class="n">size</span><span class="p">,</span> <span class="n">size</span><span class="p">))</span><br><br>    <span class="c1"># Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),</span>
    <span class="c1"># and axes 1 and 3 (horizontal axes). We first need to move the axes that we</span>
    <span class="c1"># want to combine next to each other, using transpose(), and only then we</span>
    <span class="c1"># can reshape:</span>
    <span class="n">big_image</span> <span class="o">=</span> <span class="n">image_grid</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="n">n_rows</span> <span class="o">*</span> <span class="n">size</span><span class="p">,</span>
                                                         <span class="n">images_per_row</span> <span class="o">*</span> <span class="n">size</span><span class="p">)</span>
    <span class="c1"># Now that we have a big image, we just need to show it:</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">big_image</span><span class="p">,</span> <span class="n">cmap</span> <span class="o">=</span> <span class="n">mpl</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">binary</span><span class="p">,</span> <span class="o">**</span><span class="n">options</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s2">&quot;off&quot;</span><span class="p">)</span>
</pre></div>



<div class="highlight"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">18</span><span class="p">,</span><span class="mi">18</span><span class="p">))</span>
<span class="n">example_images</span> <span class="o">=</span> <span class="n">X</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">200</span><span class="p">]</span>
<span class="n">plot_digits</span><span class="p">(</span><span class="n">example_images</span><span class="p">,</span> <span class="n">images_per_row</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span>
<span class="n">save_fig</span><span class="p">(</span><span class="s2">&quot;more_digits_plot&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>


    Saving figure more_digits_plot




![png](output_22_1.png)



The MNIST dataset is split into a training (the first 60,000 images) and a test set (the last 10,000 images).


<div class="highlight"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">X</span><span class="p">[:</span><span class="mi">60000</span><span class="p">],</span> <span class="n">X</span><span class="p">[</span><span class="mi">60000</span><span class="p">:],</span> <span class="n">y</span><span class="p">[:</span><span class="mi">60000</span><span class="p">],</span> <span class="n">y</span><span class="p">[</span><span class="mi">60000</span><span class="p">:]</span>
</pre></div>


Next, we shuffle the training set, guaranteeing that all cross-validation folds are similar and not missing any digits. Also, some algorithms are sensitive to the order of training instances. However, shuffling is bad when working on time-series data, such as stock market prices and weather conditions.


<div class="highlight"><pre><span></span><span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">seed</span><span class="p">(</span><span class="mi">42</span><span class="p">)</span>
<span class="n">shuffle_index</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">permutation</span><span class="p">(</span><span class="mi">60000</span><span class="p">)</span>
<span class="n">shuffle_index</span>
</pre></div>





    array([12628, 37730, 39991, ...,   860, 15795, 56422])




<div class="highlight"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="n">shuffle_index</span><span class="p">],</span> <span class="n">y_train</span><span class="p">[</span><span class="n">shuffle_index</span><span class="p">]</span>
<span class="o">%</span><span class="n">store</span> <span class="n">X_train</span> <span class="n">X_test</span> <span class="n">y_train</span> <span class="n">y_test</span>
</pre></div>


    Stored 'X_train' (ndarray)
    Stored 'X_test' (ndarray)
    Stored 'y_train' (ndarray)
    Stored 'y_test' (ndarray)


Let's simplify the problem and only try to identify the number five, an example of a *binary classifier*. Binary classifiers distinguish between just two classes. In this instance, five and not five. We need to create target vectors for the classification task.

The below code creates a boolean NumPy array for both the training and test sets.



<div class="highlight"><pre><span></span><span class="n">y_train_5</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_train</span> <span class="o">==</span> <span class="mi">5</span><span class="p">)</span>
<span class="n">y_test_5</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_test</span> <span class="o">==</span> <span class="mi">5</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="nb">type</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="nb">type</span><span class="p">(</span><span class="n">y_test_5</span><span class="p">))</span><br><br><span class="o">%</span><span class="n">store</span> <span class="n">y_train_5</span> <span class="n">y_test_5</span>
</pre></div>


    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>
    Stored 'y_train_5' (ndarray)
    Stored 'y_test_5' (ndarray)


Let's use `np.flatnonzero()` to check that we correctly set up the boolean NumPy arrays for the target variable in both training and test sets. `np.flatnonzero()` returns indices of the non-zero elements of an input array and is equivalent to `np.nonzero(np.ravel(a))[0]`. `np.ravel()` returns a 1-D array containing the input array elements.


<div class="highlight"><pre><span></span><span class="c1"># the indices of the true values in the y_train_5 NumPy array</span>
<span class="n">np</span><span class="o">.</span><span class="n">flatnonzero</span><span class="p">(</span><span class="n">y_train_5</span><span class="p">)</span>
</pre></div>





    array([    8,    11,    25, ..., 59928, 59942, 59965], dtype=int64)



Let's confirm the values that we expect to be true are true and vice versa.


<div class="highlight"><pre><span></span><span class="n">y_train_5</span><span class="p">[[</span><span class="mi">7</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">11</span><span class="p">,</span> <span class="mi">25</span><span class="p">,</span> <span class="mi">26</span><span class="p">]]</span>
</pre></div>





    array([False,  True,  True,  True, False])



We can use `np.concatenate()` to look at the test dataset's first five and last five indices.


<div class="highlight"><pre><span></span><span class="n">y_test_5</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_test</span> <span class="o">==</span> <span class="mi">5</span><span class="p">)</span>
<span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">((</span><span class="n">np</span><span class="o">.</span><span class="n">flatnonzero</span><span class="p">(</span><span class="n">y_test_5</span><span class="p">)[:</span><span class="mi">5</span><span class="p">],</span> <span class="n">np</span><span class="o">.</span><span class="n">flatnonzero</span><span class="p">(</span><span class="n">y_test_5</span><span class="p">)[</span><span class="o">-</span><span class="mi">5</span><span class="p">:]))</span>
</pre></div>





    array([   8,   15,   23,   45,   52, 9941, 9970, 9982, 9988, 9998],
          dtype=int64)



Again, let's confirm these indices give us the values we expect.


<div class="highlight"><pre><span></span><span class="n">y_test_5</span><span class="p">[[</span><span class="mi">8</span><span class="p">,</span> <span class="mi">9</span><span class="p">,</span> <span class="mi">23</span><span class="p">,</span> <span class="mi">26</span><span class="p">,</span> <span class="mi">9941</span><span class="p">,</span> <span class="mi">9970</span><span class="p">]]</span>
</pre></div>





    array([ True, False,  True, False,  True,  True])



Now we can pick a classifier and train it. *Stochastic Gradient Descent* (SGD) is a good place to start. The classifier can handle very large datasets efficiently since it deals with training instances independently, one at a time, making SGD well suited for _online learning_. Let's create an `SGDClassifier` and train it on the whole training set.


<div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="kn">import</span> <span class="n">SGDClassifier</span>
<span class="n">sgd_clf</span> <span class="o">=</span> <span class="n">SGDClassifier</span><span class="p">(</span><span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>
<span class="n">sgd_clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train_5</span><span class="p">)</span>
<span class="o">%</span><span class="n">store</span> <span class="n">sgd_clf</span>
</pre></div>


    Stored 'sgd_clf' (SGDClassifier)


Now you can use the classifier to detect images of the number 5:


<div class="highlight"><pre><span></span><span class="n">sgd_clf</span><span class="o">.</span><span class="n">predict</span><span class="p">([</span><span class="n">some_digit</span><span class="p">])</span>
</pre></div>





    array([ True])



In this case, the classifier guesses correct. The image represents a 5. Let's print the digit once more so that we remember.


<div class="highlight"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">some_digit_image</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">mpl</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">binary</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s2">&quot;off&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>




![png](output_43_0.png)



 We will evaluate the model's performance more closely in the next post.
