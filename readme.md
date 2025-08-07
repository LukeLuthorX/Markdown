---


---

<p>dataset download - <a href="https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/DIGQGQ&amp;version=1.0">dataset</a></p>
<h1 id="unsegarmanet">Unsegarmanet</h1>
<p>link - <a href="https://github.com/ksgr5566/UnSeGArmaNet/">github</a></p>
<pre class=" language-bash"><code class="prism  language-bash"><span class="token function">git</span> clone https://github.com/ksgr5566/unsegarmanet.git
<span class="token function">cd</span> unsegarmanet

python3 -m venv venv 
<span class="token keyword">.</span> venv/bin/activate
</code></pre>
<pre class=" language-bash"><code class="prism  language-bash">pip <span class="token function">install</span> torch<span class="token operator">==</span>2.6.0 torchvision<span class="token operator">==</span>0.21.0 torchaudio<span class="token operator">==</span>2.6.0 --index-url https://download.pytorch.org/whl/cu118
</code></pre>
<pre class=" language-bash"><code class="prism  language-bash">pip <span class="token function">install</span> torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
pip <span class="token function">install</span> torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
</code></pre>
<p>python --version</p>
<blockquote>
<p>Python 3.13.5  // this is causing the issue, 3.11 is needed</p>
</blockquote>
<p>get Python 3.11.9, delete the other version</p>
<p>torch-scatter and all is available in torch 2.6 ish version, so need to install lower version</p>
<pre class=" language-bash"><code class="prism  language-bash">pip <span class="token function">install</span> torch_geometric<span class="token operator">==</span>2.0.4
pip <span class="token function">install</span> -r requirements.txt
gdown <span class="token string">"1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_&amp;confirm=t"</span>

pip <span class="token function">install</span> segment-anything
</code></pre>
<pre class=" language-bash"><code class="prism  language-bash">python main.py --dataset <span class="token string">"WSI"</span> --dataset_path <span class="token string">"./datasets/NTNU_dataset"</span> --process <span class="token string">"DINO"</span> --conv_type <span class="token string">"ARMA"</span>
</code></pre>
<p>and changes in <a href="http://dataset.py">dataset.py</a></p>
<pre class=" language-py"><code class="prism  language-py"><span class="token keyword">import</span> os

<span class="token keyword">import</span> deeplake

<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np

<span class="token keyword">from</span> PIL <span class="token keyword">import</span> Image

<span class="token keyword">from</span> tqdm <span class="token keyword">import</span> tqdm

  

<span class="token comment"># Modified code</span>

<span class="token keyword">class</span> <span class="token class-name">Dataset</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> dataset<span class="token punctuation">,</span> dataset_path<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token keyword">if</span> dataset <span class="token operator">not</span> <span class="token keyword">in</span> <span class="token punctuation">[</span><span class="token string">"CUB"</span><span class="token punctuation">,</span> <span class="token string">"ECSSD"</span><span class="token punctuation">,</span> <span class="token string">"DUTS"</span><span class="token punctuation">,</span> <span class="token string">"WSI"</span><span class="token punctuation">]</span><span class="token punctuation">:</span>

            <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span>f<span class="token string">'Dataset: {dataset} is not supported'</span><span class="token punctuation">)</span>

        self<span class="token punctuation">.</span>dataset <span class="token operator">=</span> dataset

        <span class="token keyword">if</span> dataset <span class="token operator">==</span> <span class="token string">"CUB"</span><span class="token punctuation">:</span>

            self<span class="token punctuation">.</span>images<span class="token punctuation">,</span> self<span class="token punctuation">.</span>masks <span class="token operator">=</span> load_cub<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>os<span class="token punctuation">.</span>getcwd<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token string">'datasets'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

        <span class="token keyword">elif</span> dataset <span class="token operator">==</span> <span class="token string">"ECSSD"</span><span class="token punctuation">:</span>

            ds <span class="token operator">=</span> deeplake<span class="token punctuation">.</span>load<span class="token punctuation">(</span><span class="token string">"hub://activeloop/ecssd"</span><span class="token punctuation">)</span>

            self<span class="token punctuation">.</span>images <span class="token operator">=</span> ds<span class="token punctuation">[</span><span class="token string">"images"</span><span class="token punctuation">]</span>

            self<span class="token punctuation">.</span>masks <span class="token operator">=</span> ds<span class="token punctuation">[</span><span class="token string">"masks"</span><span class="token punctuation">]</span>

        <span class="token keyword">elif</span> dataset <span class="token operator">==</span> <span class="token string">"DUTS"</span><span class="token punctuation">:</span>

  

            self<span class="token punctuation">.</span>images<span class="token punctuation">,</span> self<span class="token punctuation">.</span>masks <span class="token operator">=</span> load_duts<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>os<span class="token punctuation">.</span>getcwd<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token string">'datasets'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

  

        <span class="token keyword">elif</span> dataset <span class="token operator">==</span> <span class="token string">"WSI"</span><span class="token punctuation">:</span>

            <span class="token keyword">if</span> <span class="token operator">not</span> dataset_path<span class="token punctuation">:</span>

                <span class="token keyword">raise</span> ValueError<span class="token punctuation">(</span><span class="token string">"For the WSI dataset, you must provide the path using --dataset_path."</span><span class="token punctuation">)</span>

            self<span class="token punctuation">.</span>images<span class="token punctuation">,</span> self<span class="token punctuation">.</span>masks <span class="token operator">=</span> load_colorectal_wsi<span class="token punctuation">(</span>dataset_path<span class="token punctuation">)</span>

        self<span class="token punctuation">.</span>loader <span class="token operator">=</span> <span class="token builtin">len</span><span class="token punctuation">(</span>self<span class="token punctuation">.</span>images<span class="token punctuation">)</span>

  

    <span class="token keyword">def</span> <span class="token function">load_samples</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token keyword">for</span> imagep<span class="token punctuation">,</span> true_maskp <span class="token keyword">in</span> <span class="token builtin">zip</span><span class="token punctuation">(</span>self<span class="token punctuation">.</span>images<span class="token punctuation">,</span> self<span class="token punctuation">.</span>masks<span class="token punctuation">)</span><span class="token punctuation">:</span>

            <span class="token keyword">try</span><span class="token punctuation">:</span>

                <span class="token keyword">if</span> self<span class="token punctuation">.</span>dataset <span class="token operator">==</span> <span class="token string">"CUB"</span><span class="token punctuation">:</span>

                    img <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>imagep<span class="token punctuation">)</span><span class="token punctuation">)</span>

                    seg <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>true_maskp<span class="token punctuation">)</span><span class="token punctuation">.</span>convert<span class="token punctuation">(</span><span class="token string">'L'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

                    true_mask <span class="token operator">=</span> np<span class="token punctuation">.</span>where<span class="token punctuation">(</span>seg <span class="token operator">&gt;=</span> <span class="token number">200</span><span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">0</span><span class="token punctuation">)</span>

                <span class="token keyword">elif</span> self<span class="token punctuation">.</span>dataset <span class="token operator">==</span> <span class="token string">"ECSSD"</span><span class="token punctuation">:</span>

                    img <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>imagep<span class="token punctuation">)</span>

                    seg <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>true_maskp<span class="token punctuation">)</span>

                    true_mask <span class="token operator">=</span> np<span class="token punctuation">.</span>where<span class="token punctuation">(</span>seg <span class="token operator">==</span> <span class="token boolean">True</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span>

                <span class="token keyword">if</span> self<span class="token punctuation">.</span>dataset <span class="token operator">==</span> <span class="token string">"DUTS"</span><span class="token punctuation">:</span>

                    img <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>imagep<span class="token punctuation">)</span><span class="token punctuation">)</span>

                    seg <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>true_maskp<span class="token punctuation">)</span><span class="token punctuation">.</span>convert<span class="token punctuation">(</span><span class="token string">'L'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

                    true_mask <span class="token operator">=</span> np<span class="token punctuation">.</span>where<span class="token punctuation">(</span>seg <span class="token operator">==</span> <span class="token number">255</span><span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span>np<span class="token punctuation">.</span>uint8<span class="token punctuation">)</span>

                <span class="token keyword">elif</span> self<span class="token punctuation">.</span>dataset <span class="token operator">==</span> <span class="token string">"WSI"</span><span class="token punctuation">:</span>

                    img <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>imagep<span class="token punctuation">)</span><span class="token punctuation">.</span>convert<span class="token punctuation">(</span><span class="token string">'RGB'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

                    seg <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>true_maskp<span class="token punctuation">)</span><span class="token punctuation">.</span>convert<span class="token punctuation">(</span><span class="token string">'L'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

                    true_mask <span class="token operator">=</span> np<span class="token punctuation">.</span>where<span class="token punctuation">(</span>seg <span class="token operator">&gt;</span> <span class="token number">128</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span>np<span class="token punctuation">.</span>uint8<span class="token punctuation">)</span>

                <span class="token keyword">yield</span> img<span class="token punctuation">,</span> true_mask

            <span class="token keyword">except</span> Exception <span class="token keyword">as</span> e<span class="token punctuation">:</span>

                <span class="token keyword">print</span><span class="token punctuation">(</span>e<span class="token punctuation">)</span>

            <span class="token keyword">finally</span><span class="token punctuation">:</span>

                self<span class="token punctuation">.</span>loader <span class="token operator">-=</span> <span class="token number">1</span>

  
  

<span class="token keyword">def</span> <span class="token function">load_colorectal_wsi</span><span class="token punctuation">(</span>dataset_path<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token triple-quoted-string string">"""

    Loads the Colorectal WSI dataset from the specified path.

    Assumes the dataset is organized with 'images' and 'masks' subdirectories.

    """</span>

    images_path <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>dataset_path<span class="token punctuation">,</span> <span class="token string">'images'</span><span class="token punctuation">)</span>

    masks_path <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>dataset_path<span class="token punctuation">,</span> <span class="token string">'masks'</span><span class="token punctuation">)</span>

  

    image_files <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span><span class="token punctuation">[</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>images_path<span class="token punctuation">,</span> f<span class="token punctuation">)</span> <span class="token keyword">for</span> f <span class="token keyword">in</span> os<span class="token punctuation">.</span>listdir<span class="token punctuation">(</span>images_path<span class="token punctuation">)</span> <span class="token keyword">if</span> f<span class="token punctuation">.</span>endswith<span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token string">'.jpg'</span><span class="token punctuation">,</span> <span class="token string">'.jpeg'</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

    mask_files <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span><span class="token punctuation">[</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>masks_path<span class="token punctuation">,</span> f<span class="token punctuation">)</span> <span class="token keyword">for</span> f <span class="token keyword">in</span> os<span class="token punctuation">.</span>listdir<span class="token punctuation">(</span>masks_path<span class="token punctuation">)</span> <span class="token keyword">if</span> f<span class="token punctuation">.</span>endswith<span class="token punctuation">(</span><span class="token string">'.png'</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

  

    <span class="token comment"># Basic check to ensure you have corresponding images and masks</span>

    <span class="token keyword">if</span> <span class="token builtin">len</span><span class="token punctuation">(</span>image_files<span class="token punctuation">)</span> <span class="token operator">!=</span> <span class="token builtin">len</span><span class="token punctuation">(</span>mask_files<span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"Warning: Found {len(image_files)} images and {len(mask_files)} masks. They should match."</span><span class="token punctuation">)</span>

    <span class="token comment"># More detailed check for filename correspondence</span>

    img_basenames <span class="token operator">=</span> <span class="token punctuation">{</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>splitext<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>basename<span class="token punctuation">(</span>f<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span> <span class="token keyword">for</span> f <span class="token keyword">in</span> image_files<span class="token punctuation">}</span>

    mask_basenames <span class="token operator">=</span> <span class="token punctuation">{</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>splitext<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>basename<span class="token punctuation">(</span>f<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span> <span class="token keyword">for</span> f <span class="token keyword">in</span> mask_files<span class="token punctuation">}</span>

    <span class="token keyword">if</span> img_basenames <span class="token operator">!=</span> mask_basenames<span class="token punctuation">:</span>

        <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Warning: Image and mask filenames do not perfectly match."</span><span class="token punctuation">)</span>

        <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"Images without masks: {img_basenames - mask_basenames}"</span><span class="token punctuation">)</span>

        <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"Masks without images: {mask_basenames - img_basenames}"</span><span class="token punctuation">)</span>

  
  

    <span class="token keyword">return</span> image_files<span class="token punctuation">,</span> mask_files

  

<span class="token keyword">def</span> <span class="token function">load_cub</span><span class="token punctuation">(</span>cp<span class="token punctuation">)</span><span class="token punctuation">:</span>

    cp <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>os<span class="token punctuation">.</span>getcwd<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token string">'datasets'</span><span class="token punctuation">)</span>

    fold <span class="token operator">=</span> f<span class="token string">'{cp}/segmentations'</span>

    file_paths <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

    <span class="token keyword">for</span> root<span class="token punctuation">,</span> _<span class="token punctuation">,</span> files <span class="token keyword">in</span> os<span class="token punctuation">.</span>walk<span class="token punctuation">(</span>fold<span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token keyword">for</span> <span class="token builtin">file</span> <span class="token keyword">in</span> files<span class="token punctuation">:</span>

            file_paths<span class="token punctuation">.</span>append<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>root<span class="token punctuation">,</span><span class="token builtin">file</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

  

    fold2 <span class="token operator">=</span> f<span class="token string">'{cp}/CUB_200_2011/images'</span>

    fp2 <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

    <span class="token keyword">for</span> root<span class="token punctuation">,</span> _<span class="token punctuation">,</span> files <span class="token keyword">in</span> os<span class="token punctuation">.</span>walk<span class="token punctuation">(</span>fold2<span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token keyword">for</span> <span class="token builtin">file</span> <span class="token keyword">in</span> files<span class="token punctuation">:</span>

            fp2<span class="token punctuation">.</span>append<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>root<span class="token punctuation">,</span><span class="token builtin">file</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

  

    fp2  <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span>fp2<span class="token punctuation">)</span>

    file_paths <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span>file_paths<span class="token punctuation">)</span>

  

    <span class="token keyword">with</span> <span class="token builtin">open</span><span class="token punctuation">(</span>f<span class="token string">'{cp}/CUB_200_2011/train_test_split.txt'</span><span class="token punctuation">)</span> <span class="token keyword">as</span> f<span class="token punctuation">:</span>

        count <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span>

        pretest <span class="token operator">=</span> <span class="token builtin">set</span><span class="token punctuation">(</span><span class="token punctuation">)</span>

        <span class="token keyword">for</span> line <span class="token keyword">in</span> f<span class="token punctuation">:</span>

            x <span class="token operator">=</span> line<span class="token punctuation">.</span>split<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span>

            <span class="token keyword">if</span> x <span class="token keyword">in</span> count<span class="token punctuation">:</span>

                count<span class="token punctuation">[</span>x<span class="token punctuation">]</span><span class="token operator">+=</span><span class="token number">1</span>

            <span class="token keyword">else</span><span class="token punctuation">:</span>

                count<span class="token punctuation">[</span>x<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token number">1</span>

            <span class="token keyword">if</span> x <span class="token operator">==</span> <span class="token string">"0"</span><span class="token punctuation">:</span>

                pretest<span class="token punctuation">.</span>add<span class="token punctuation">(</span>line<span class="token punctuation">.</span>split<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

  

    <span class="token keyword">with</span> <span class="token builtin">open</span><span class="token punctuation">(</span>f<span class="token string">'{cp}/CUB_200_2011/images.txt'</span><span class="token punctuation">)</span> <span class="token keyword">as</span> u<span class="token punctuation">:</span>

        test <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

        <span class="token keyword">for</span> line <span class="token keyword">in</span> u<span class="token punctuation">:</span>

            x<span class="token punctuation">,</span>y  <span class="token operator">=</span> line<span class="token punctuation">.</span>split<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span>line<span class="token punctuation">.</span>split<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span>

            <span class="token keyword">if</span> x <span class="token keyword">in</span> pretest<span class="token punctuation">:</span>

                test<span class="token punctuation">.</span>append<span class="token punctuation">(</span>y<span class="token punctuation">)</span>

  

    masks <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span><span class="token punctuation">[</span>f<span class="token string">'{cp}/segmentations/'</span> <span class="token operator">+</span> x<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token builtin">len</span><span class="token punctuation">(</span>x<span class="token punctuation">)</span><span class="token operator">-</span><span class="token number">3</span><span class="token punctuation">]</span> <span class="token operator">+</span> <span class="token string">'png'</span> <span class="token keyword">for</span> x <span class="token keyword">in</span> test<span class="token punctuation">]</span><span class="token punctuation">)</span>

    test <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span><span class="token punctuation">[</span>f<span class="token string">'{cp}/CUB_200_2011/images/'</span> <span class="token operator">+</span> x <span class="token keyword">for</span> x <span class="token keyword">in</span> test<span class="token punctuation">]</span><span class="token punctuation">)</span>

  

    <span class="token keyword">return</span> test<span class="token punctuation">,</span> masks

  
  

<span class="token keyword">def</span> <span class="token function">load_duts</span><span class="token punctuation">(</span>cp<span class="token punctuation">)</span><span class="token punctuation">:</span>

    cp <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>os<span class="token punctuation">.</span>getcwd<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token string">'datasets'</span><span class="token punctuation">)</span>

  

    fold <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>cp<span class="token punctuation">,</span> <span class="token string">'DUTS-TE/DUTS-TE-Image'</span><span class="token punctuation">)</span>

    file_paths <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

    <span class="token keyword">for</span> root<span class="token punctuation">,</span> _<span class="token punctuation">,</span> files <span class="token keyword">in</span> os<span class="token punctuation">.</span>walk<span class="token punctuation">(</span>fold<span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token keyword">for</span> <span class="token builtin">file</span> <span class="token keyword">in</span> files<span class="token punctuation">:</span>

            file_paths<span class="token punctuation">.</span>append<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>root<span class="token punctuation">,</span><span class="token builtin">file</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

  

    fold2 <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>cp<span class="token punctuation">,</span> <span class="token string">'DUTS-TE/DUTS-TE-Mask'</span><span class="token punctuation">)</span>

    fp2 <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

    <span class="token keyword">for</span> root<span class="token punctuation">,</span> _<span class="token punctuation">,</span> files <span class="token keyword">in</span> os<span class="token punctuation">.</span>walk<span class="token punctuation">(</span>fold2<span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token keyword">for</span> <span class="token builtin">file</span> <span class="token keyword">in</span> files<span class="token punctuation">:</span>

            fp2<span class="token punctuation">.</span>append<span class="token punctuation">(</span>os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>root<span class="token punctuation">,</span><span class="token builtin">file</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

  

    masks <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span>fp2<span class="token punctuation">)</span>

    test <span class="token operator">=</span> <span class="token builtin">sorted</span><span class="token punctuation">(</span>file_paths<span class="token punctuation">)</span>

  

    <span class="token keyword">return</span> test<span class="token punctuation">,</span> masks
</code></pre>
<p><a href="http://main.py">main.py</a></p>
<pre class=" language-py"><code class="prism  language-py"><span class="token keyword">from</span> segment <span class="token keyword">import</span> Segmentation

<span class="token keyword">from</span> datasets<span class="token punctuation">.</span>dataset <span class="token keyword">import</span> Dataset

<span class="token keyword">from</span> argparse <span class="token keyword">import</span> ArgumentParser

  

parser <span class="token operator">=</span> ArgumentParser<span class="token punctuation">(</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--bs"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">bool</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--epochs"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">int</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token number">20</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--resolution"</span><span class="token punctuation">,</span> nargs<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">int</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token punctuation">[</span><span class="token number">224</span><span class="token punctuation">,</span> <span class="token number">224</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--activation"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token string">'selu'</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--loss_type"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token string">'DMON'</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--process"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token string">'DINO'</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--dataset"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token string">'ECSSD'</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--dataset_path"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">,</span> <span class="token builtin">help</span><span class="token operator">=</span><span class="token string">"Path to your custom dataset folder (e.g., ./datasets/NTNU_dataset)"</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--threshold"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">float</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>

parser<span class="token punctuation">.</span>add_argument<span class="token punctuation">(</span><span class="token string">"--conv_type"</span><span class="token punctuation">,</span> <span class="token builtin">type</span><span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">,</span> default<span class="token operator">=</span><span class="token string">'ARMA'</span><span class="token punctuation">)</span>

  

args <span class="token operator">=</span> parser<span class="token punctuation">.</span>parse_args<span class="token punctuation">(</span><span class="token punctuation">)</span>

  

<span class="token keyword">if</span> __name__ <span class="token operator">==</span> <span class="token string">'__main__'</span><span class="token punctuation">:</span>

    seg <span class="token operator">=</span> Segmentation<span class="token punctuation">(</span>args<span class="token punctuation">.</span>process<span class="token punctuation">,</span> args<span class="token punctuation">.</span>bs<span class="token punctuation">,</span> args<span class="token punctuation">.</span>epochs<span class="token punctuation">,</span> <span class="token builtin">tuple</span><span class="token punctuation">(</span>args<span class="token punctuation">.</span>resolution<span class="token punctuation">)</span><span class="token punctuation">,</span> args<span class="token punctuation">.</span>activation<span class="token punctuation">,</span> args<span class="token punctuation">.</span>loss_type<span class="token punctuation">,</span> args<span class="token punctuation">.</span>threshold<span class="token punctuation">,</span> args<span class="token punctuation">.</span>conv_type<span class="token punctuation">)</span>

    ds <span class="token operator">=</span> Dataset<span class="token punctuation">(</span>args<span class="token punctuation">.</span>dataset<span class="token punctuation">,</span> args<span class="token punctuation">.</span>dataset_path<span class="token punctuation">)</span>

  

    total_iou <span class="token operator">=</span> <span class="token number">0</span>

    total_samples <span class="token operator">=</span> <span class="token number">0</span>

    <span class="token keyword">while</span> ds<span class="token punctuation">.</span>loader <span class="token operator">&gt;</span> <span class="token number">0</span><span class="token punctuation">:</span>

        <span class="token keyword">for</span> img<span class="token punctuation">,</span> mask <span class="token keyword">in</span> ds<span class="token punctuation">.</span>load_samples<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>

            <span class="token keyword">try</span><span class="token punctuation">:</span>

                iou<span class="token punctuation">,</span> _<span class="token punctuation">,</span> _ <span class="token operator">=</span> seg<span class="token punctuation">.</span>segment<span class="token punctuation">(</span>img<span class="token punctuation">,</span> mask<span class="token punctuation">)</span>

                total_iou <span class="token operator">+=</span> iou

                total_samples <span class="token operator">+=</span> <span class="token number">1</span>

                <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">"IoU for Current Image: {iou:.2f}  mIoU so far: {(total_iou/total_samples):.2f}"</span><span class="token punctuation">)</span>

  

            <span class="token keyword">except</span> Exception <span class="token keyword">as</span> e<span class="token punctuation">:</span>

                <span class="token keyword">print</span><span class="token punctuation">(</span>e<span class="token punctuation">)</span>

                <span class="token keyword">continue</span>

    <span class="token keyword">print</span><span class="token punctuation">(</span>f<span class="token string">'Final mIoU: {(total_iou / total_samples):.4f}'</span><span class="token punctuation">)</span>
</code></pre>
<h1 id="medsam">MedSam</h1>
<p>link - <a href="https://github.com/bowang-lab/MedSAM">github</a></p>
<h6 id="directly-running-it-from-unsegarmanet-repo">Directly Running it from Unsegarmanet Repo</h6>
<p><strong>MedSAM (<code>--process "MEDSAM_INFERENCE"</code>)</strong> is a <strong>prompt-based</strong> model. It’s incredibly powerful, but it’s not fully automatic. You have to give it a “prompt” to tell it <em>what</em> you want it to segment. This prompt can be a point, a scribble, or, most commonly, a bounding box.</p>
<p>This is the clever part of the repository’s setup. To benchmark MedSAM fairly in an automatic pipeline, the code in <code>segment.py</code> essentially “cheats” by using the ground truth to create the prompt.</p>
<p>Here’s the workflow when you run MedSAM:</p>
<ol>
<li>
<p>For each image, it first loads your <strong>ground truth mask</strong>.</p>
</li>
<li>
<p>It then runs the <code>get_bounding_box</code> function on that mask to draw the tightest possible box around the cancerous region.</p>
</li>
<li>
<p>This perfect, machine-generated bounding box is then fed to MedSAM as the prompt.</p>
</li>
<li>
<p>MedSAM performs the segmentation based on that prompt, and the result is compared to the ground truth mask to get the IoU score.</p>
</li>
</ol>
<pre class=" language-bash"><code class="prism  language-bash"> python main.py --dataset <span class="token string">"WSI"</span> --dataset_path <span class="token string">"./datasets/NTNU_dataset"</span> --process <span class="token string">"MEDSAM_INFERENCE"</span>
</code></pre>
<p><strong>fix of medsam</strong></p>
<pre class=" language-bash"><code class="prism  language-bash">pip uninstall transformers
pip <span class="token function">install</span> transformers<span class="token operator">==</span>4.33.2
</code></pre>
<p><a href="http://segment.py">segment.py</a> changes</p>
<pre class=" language-py"><code class="prism  language-py">    <span class="token keyword">def</span> <span class="token function">medsam_inference</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> image<span class="token punctuation">,</span> mask<span class="token punctuation">)</span><span class="token punctuation">:</span>

        <span class="token triple-quoted-string string">"""

        @param image: Image to segment (numpy array)

        @param mask: Ground truth mask (binary numpy array)

        """</span>

        image <span class="token operator">=</span> cv2<span class="token punctuation">.</span>resize<span class="token punctuation">(</span>image<span class="token punctuation">.</span>astype<span class="token punctuation">(</span><span class="token string">'float'</span><span class="token punctuation">)</span><span class="token punctuation">,</span> self<span class="token punctuation">.</span>resolution<span class="token punctuation">,</span> interpolation<span class="token operator">=</span>cv2<span class="token punctuation">.</span>INTER_NEAREST<span class="token punctuation">)</span>

  

        mask <span class="token operator">=</span> <span class="token punctuation">(</span>cv2<span class="token punctuation">.</span>resize<span class="token punctuation">(</span>mask<span class="token punctuation">,</span> self<span class="token punctuation">.</span>resolution<span class="token punctuation">,</span> interpolation<span class="token operator">=</span>cv2<span class="token punctuation">.</span>INTER_NEAREST<span class="token punctuation">)</span> <span class="token operator">&gt;</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span>np<span class="token punctuation">.</span>uint8<span class="token punctuation">)</span>

  
  

        input_boxes <span class="token operator">=</span> Segmentation<span class="token punctuation">.</span>get_bounding_box<span class="token punctuation">(</span>mask<span class="token punctuation">)</span>

        inputs <span class="token operator">=</span> self<span class="token punctuation">.</span>processor<span class="token punctuation">(</span>image<span class="token punctuation">,</span> input_boxes<span class="token operator">=</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token punctuation">[</span>input_boxes<span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">,</span> return_tensors<span class="token operator">=</span><span class="token string">"pt"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>to<span class="token punctuation">(</span>self<span class="token punctuation">.</span>device<span class="token punctuation">)</span>

        <span class="token keyword">with</span> torch<span class="token punctuation">.</span>no_grad<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>

            outputs <span class="token operator">=</span> self<span class="token punctuation">.</span>model<span class="token punctuation">(</span><span class="token operator">**</span>inputs<span class="token punctuation">,</span> multimask_output<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span>

  

        medsam_seg_prob <span class="token operator">=</span> torch<span class="token punctuation">.</span>sigmoid<span class="token punctuation">(</span>outputs<span class="token punctuation">.</span>pred_masks<span class="token punctuation">.</span>squeeze<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

  

        medsam_seg_prob <span class="token operator">=</span> medsam_seg_prob<span class="token punctuation">.</span>cpu<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>numpy<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>squeeze<span class="token punctuation">(</span><span class="token punctuation">)</span>

        medsam_seg <span class="token operator">=</span> <span class="token punctuation">(</span>medsam_seg_prob <span class="token operator">&gt;</span> <span class="token number">0.5</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span>np<span class="token punctuation">.</span>uint8<span class="token punctuation">)</span>

  

        <span class="token keyword">return</span> medsam_seg
</code></pre>

