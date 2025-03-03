# 介绍
一个快速理解旋转位置编码在自然语言，多模态场景应用的库  
以下先简单介绍ROPE论文原理，后拓展到多模态场景的2D、3D的旋转位置编码

# ROPE详解
目前流行的llm通常在attention和embedding层使用这种相对位置编码，使当前位置的token更加关注临近位置token，在长时序捕捉能力更强。[论文](https://arxiv.org/pdf/2104.09864) [论文解析](https://zhuanlan.zhihu.com/p/647109286)
<div align="center">
  <img src="doc/Rope.png" alt="rope" width="700" height="400">
  <p style="font-size: 10px; color: gray;">rope原理</p>
</div>

#### *原理介绍*  
* 相对位置编码通过在注意力计算`q[m]*k[n]^T`引入`q[m]`相对于`k[n]`的位置信息(m-n)，标记位置信息，使得注意力计算不仅与`q[m]`和`k[n]`本身相关也与他们的处在序列的位置相关，*这里q,k都是行向量*。     。
* 先从二维向量的考量，对于`q[m]和k[n]`，假如存在这样的一个二维旋转矩阵函数`R(posion)`，若R满足`q[m]*R(m) * (k[n] * R(n))^T) = q[m]*R(m-n)*k[n]^T`，此时对于注意力分数的计算结果受到相对位置(m-n)的影响，成功引入了相对位置信息   
* 推广到多维向量，只需要对q，k的向量元素按照二维两两配对，R的控制参数在位置参数基础上引入维度参数，由于注意力点乘后具有加性，刚好也能够性质`q[m]*R(m) * (k[n] * R(n))^T) = q[m]*R(m-n)*k[n]^T`，作者设计这种R函数机制，如上图rope原理
<div align="center">
  <img src="doc/rope_ddim.jpg" alt="rope" width="600" height="200">
  <p style="font-size: 10px; color: gray;">旋转矩阵拓展</p>
</div>


#### *远程衰减性的证明：随着相对距离变大，qk^T的权重普遍应该是下降*  
原始论文通不等式转换，在进行旋转位置编码后计算q*k^T的注意力分数是一个与相对距离有关的分数，其最大值存在远程衰减性。这一性质使得注意力计算中更加关注邻近token
<div align="center">
  <img src="doc/rope_decay.png" alt="rope" width="550" height="300">
  <p style="font-size: 10px; color: gray;">注意力上限衰减曲线</p>
</div>

#### *外推延展性的证明：当序列长度超出预训练设定，仍能捕捉不超过预训练相对长度的位置关系*   
* 论文中没有提到这种外推的说明，说一下自己的直观理解。远程衰减性可以看出，ROPE使得token更加关注近邻位置，关注度与q,k的相对距离有关，而与绝对的位置无关。   
* 我理解注意力计算的q*k^T只要相对关系不超出预训练的最大相对长度，都是能够在训练效果内的，那么对于每个token，应该有一个前后有效的最大窗口。而对相对距离超出的窗口的，这些q,k注意依赖本身也很弱，因此影响不会太大，加上前后信息的传递性，这种影响会被进一步削弱。   
* 因此，当新增序列，仍然能够被有效处理。

#### *代码解析*   
* 矩阵点乘实现：论文中为了避免稀疏矩阵做内积的低效性，提出了点乘实现寻转位置编码，可以参照```/Rope/rope_chatGLM.py```   
* 虚数内积实现：另一种优化方式是，使用虚数矩阵做内积实现，可以参照```/Rope/rope_llama.py```

# ROPE拓展
相对于文本，图片和视频的位置信息更加复杂。图片的长宽天然表示两个不同维度，处于不同区域的物体相对于其他物体的位置关系是不同的。视频由多帧连续图片构成，在图片基础上增加了一个时间维度。

对于图片和视频的ROPE设置尚未形成一种共识，不像文本，在一维的序列上，计算一个token与其他token的相对位置是唯一的，天然在注意力计算中具有明确的位置标识。而在二维或者三维的空间内，会存在重复干扰，目前还没看到能有这种标准的标记方法产生。e.g，图片的二维空间，计算第一个patch和右边第一个patch的相对位置，与计算第一个patch和下边第一个patch的相对位置，沿用rope的性质描述容易得到相同的位置关系。

尽管如此，才视觉多模态中，将1D的ROPE，拓展为2D甚至3D是有利于提高模型效果的，参考[VisionLLaMA](https://papers.cool/arxiv/2403.00522)
- 图片2D拓展方式: 对patch token四个维度一组注入位置编码(w, h)参数
- 文本3D拓展方式: 对patch token四个维度一组注入位置编码(t, w, h)参数

<div align="center">
  <img src="doc/1d-2d-Rope.jpg" alt="rope" width="800" height="400">
  <p style="font-size: 10px; color: gray;">1D拓展2D</p>
</div>

预设token维度为4，分析2D ROPE的旋转矩阵可以表述为如下图。根据矩阵乘法的加性原理，注意力计算满足`q[wm,hm]*R(wm,hm) * (k[wn,hn] * R(wn,hn))^T) = q[wm,hm]*R(wm-wn,hm-hn)*k[wn,hm]^T`，此时相对位置的描述为`(wm-wn,hm-hn)`。同理拓展到高纬与3D的情况是类似的。
<div align="center">
  <img src="doc/4dim-2d-rope.png" alt="rope" width="320" height="150">
  <p style="font-size: 10px; color: gray;">2d 4dim</p>
</div>

需要注意的是，位置和维度控制参数设置规则，不同模态融合的规则，需要考虑更多的因素，如可扩展性，对称性，公平性等等，可以参照这一篇[博文](https://kexue.fm/archives/10352/comment-page-1)

值得一提的是，在多模态的应用场景下，一个简单的位置区分和模态区分的方式是，沿用1D ROPE，将patch token展平和text token拼接，只是在图片token的起始位置加入特殊的标记touken `<image/>...</image>`区分模态差异，如LLaVA使用了这种处理方式

## 应用
QwenVL2以后使用了这种3D多模态位置编码，特别地，可以兼容图片2D和文本1D的位置编码，论文中将这种编码机制定义为M-RoPE
<div align="center">
  <img src="doc/mrope.png" alt="mrope" width="925" height="289">
  <p style="font-size: 10px; color: gray;">mrope示意</p>
</div>

这种位置编码将ROPE解构成三个维度，时间，空间height，空间width。位置标记序号表示为(t, h, w)，t用于定位视频帧号，(h, w)用于定位帧的patch位置。特别地，文本的三个坐标相同，取上一个模态的max(t, h, w)

3D Rope函数控制变量包括3个位置参数t,h,w以及对应的维度参数，同样作用于注意力计算能够满足性质`q[tm,hm,wm]*R(tm,hm,wm) * (k[tn,hn,wn] * R(tn,hn,wn))^T) = q[tm,hm,wm]*R(tm-tn,hm-hn,wm-wn)*k[tn,hn,wn]^T`

不同的编码方式对于ROPE函数的设置有差异，但整体上需要遵循该性质进行设计