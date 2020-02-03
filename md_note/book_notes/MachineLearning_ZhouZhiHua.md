# 周志华《机器学习》笔记

## 1 绪论

略  

---

## 2 模型评估与选择

### 2.1 经验误差与过拟合

若可彻底避免过拟合，则通过经验误差（training error）最小化就能获得最优解，这就意味着我们构造性的证明了$P = NP$，因此，只要相信$P\not ={NP}$，过拟合就不可避免  

### 2.2 评估方法

选取尽可能与训练样本不同的测试集  

#### 2.2.1 留出法

通常留出1/5~1/3  

#### 2.2.2 交叉验证

将样本划分为K个互斥子集

#### 2.2.3 自助法

给定包含m个样本的数据集D，我们对他采样得到数据集D'：每次从D中有放回的抽样，重复m次，用D'作训练集，D - D'作测试集  

- 在数据集小、难划分训练/测试集时很有用  
- 但是改变了数据的分布，会引入估计偏差，数据够多一般不用此方法  

#### 2.2.4 调参

略  

### 2.3 性能度量

分类问题中常用的性能度量  

#### 2.3.1 错误率与精度

二分类中常用，也可适用于多分类  
用$f$模型来衡量样例集$D$的错误率：  
$$
E(f;D)=\frac{1}{m}\displaystyle\sum_{i=1}^{m}\mathbb{I}(f(x_i)\not ={y_i})
$$
精确度：
$$
acc(f;D)=1-E(f;D)
$$
更一般的，对数据分布${D}$和概率密度函数$p(\cdot)$，错误率：  
$$
E(f;D)=\int_{x\sim D}\mathbb{I}(f(x)\not =y)\cdot p(x)\space dx
$$

#### 2.3.2 查准率、查全率与F1

> precision/racall

**混淆矩阵：**

| 实际\预测      | 预测为真 | 预测为假 |
| -------------- | :------: | :------: |
| 为真(positive) |    TP    |    FN    |
| 为假(nagetive) |    FP    |    TN    |

**查准率P**(挑出的瓜有多少是好的）和**查全率R**（所有好瓜有多少被挑出来了）分别定义为：  
$$
P=\frac{TP}{TP+FP}
$$
$$
R=\frac{TP}{TP+FN}
$$

度量模型好坏：F1度量（更一般地，$F_\beta$度量）：
$$
\frac{1}{F_\beta}=\frac{1}{(1+\beta^2)}\times(\frac{1}{P}+\frac{\beta^2}{R})
$$

- $\beta$反映了对P，R重视程度
- F1是$\beta=1$的情况
- 和算术平均/几何平均相比，调和平均更重视较小值

对于有多个混淆矩阵的情况（如多次测试/多个数据集上测试/多分类）有两种做法：

1. 各矩阵上分别计算P，R，取平均，计算F1（macro）
2. 平均各个混淆矩阵，再计算P，R，F1（micro）

#### 2.3.3 ROC与AUC

> Receiver Operating Characteristic / Area Under Curve

![Receiver Operating Character](images/ROC.jpg)

反映的是『对结果预测置信度**排序能力**』的好坏：先把所有点设为F（阈值设为最大，从(0,0)开始），从高到低，依次把点设为正例（逐步降低阈值），如果是真正例就向上走一步（让AUC变大更快），否则向右走一步  
loss = 1 - AUC  

#### 2.3.4 代价敏感错误率与代价曲线\？?不太懂

对于不同类型错误造成的代价不一样的情况（如误判健康人为不健康/误判不健康为健康），可为错误赋予非均等代价  

| 实际\预测 |    第0类    |    第1类    |
| --------- | :---------: | :---------: |
| 第0类     |      0      | $cost_{01}$ |
| 第1类     | $cost_{10}$ |      0      |

先前的性能度量大多隐式的假设均等代价，计算的是错误**次数**而非**总体代价**  

![cost_curve](images/cost_curve.jpg)

### 2.4 比较检验

存在的问题：

1. 训练和测试的结果有差异
2. 结果与测试集的属性有很大联系
3. 随机性

解决问题的重要方法：**假设检验**  
通过统计来判断一个命题的真伪，需要量化一个评判标准$\alpha$  
可能的情况:

1. 标准太严，拒绝了本来为真的假设，记为$\alpha$，称之为显著水平(离谱水平)
2. 标准太松，接受了本来为假的假设。在一定的$\alpha$下我们希望受伪的概率越小越好。记受伪概率为$\pi=1-P(接受H_0|H_0为假)$，称之为检验的势。即在不冤枉好人的前提下最能发现坏人

![hypothesis_test](images/hypogthesis_test.png)

> 记training error为$\hat{e}$，实际（泛化）错误率为$e$

#### 2.4.1 假设检验

在有m个样本的测试集上，泛化错误率为$e$的模型被测得错误率（training error）为$\hat{e}$的概率：  
$$
P(\hat{e};e)=\left(^m _{\hat{e}\times m}\right)\cdot e^{\hat{e}\times m}\cdot (1-e)^{m-\hat{e}\times m}
$$

已知training error的情况下，解$\partial P(\hat{e};e)/\partial e=0$有，$P(\hat{e};e)$在$e=\hat{e}$时最大  
  
假设$e\leqslant e_0$，则计算$\alpha$显著度下的临界$\bar{e}$，若测试错误率$\hat{e}$小于临界值$\bar{e}$，则可以得出结论：在$\alpha$显著度下不能拒绝$e\leqslant e_0$的假设，即能以$1-\alpha$的置信度认为泛化错误率不大于$e_0$  

对于多次测试的情况，可以使用"t检验"  
假定我们得到了k个错误率$\hat{e}_1,\hat{e}_2,...,\hat{e}_k$，则平均错误率$\mu$和方差$\sigma^2$为：  
$$
\mu = \frac{1}{k}\displaystyle\sum_{i=1}^{k}\hat{e}_i
$$
$$
\sigma^2 = \frac{1}{k-1}\displaystyle\sum_{i=1}^k(\hat{e}_i-\mu)^2
$$

考虑这k个测试错误率可以看作泛化错误率$e_0$的独立采样，则变量  
$$
\tau_t = \frac{\sqrt{k}(\mu-e_0)}{\sigma}
$$
服从自由的为k-1的t分布（详情见下文）。对假设"$\mu=e_0$"和显著度$\alpha$，我们可以计算在均值为$e_0$时，在$1-\alpha$概率内应该观察到的最大错误率（临界值），这里要考虑双边假设，两边“出界范围”都是$\alpha/2$，若平均错误率$|\mu-e_0| \in [t_{-\alpha/2},t_{\alpha/2}]$（在认为合理的面积内），则不能拒绝假设"$\mu=e_0$"，即有$1-\alpha$的置信度认为泛化错误率为$e_0$。

##### t检验云云

？?不知道下面这段对不对。。。  

- 假设：A，B是同一个分布（$A～N(\mu_0,?)$，B是已知分布）
- 检验：取样出手头有的样本的概率大不大，根据直觉，这个概率应该和以下因素有关：
  - 样本平均值$\bar{x}$
  - 样本方差$s$
  - 采样数$n$

在用样本估计整体时，研究样本的**平均数的分布**，这个分布（记为分布S---我随便取得名字）是一个[ 方差是总体分布方差的$1/\sqrt{n},n$为样本量，期望与总体期望相同 ]的正态分布（*p.s. 抽样次数足够多才有这个结论，但如果原分布是正态分布就没有任何限制*）

综上，得出以下统计量：  
$$
\tau = \frac{\sqrt{n}\cdot|\bar{x}-\mu_0|}{s}
$$
这个量服从自由度为n-1的t分布（n趋于无穷时为正态分布），代表的是[ 假设母体A服从B分布，对它抽样应该得到的平均数分布 ]，再看手头的数据，他从这个分布里sample出来的概率有多大呢（用$\tau$反映差距），再把这个结果和置信区间比较来决定是否接受它。  

具体参见：
[t检验(知乎,啤酒)](https://www.zhihu.com/answer/296723303)|
[t检验(知乎,情书)](https://www.zhihu.com/answer/589141978)

#### 2.4.2 交叉验证t检验

**目标**：判断两个模型性能差异大不大，如果大就取用平均误差小的那个  
**思路**：先对两个模型喂入一样的数据（用k折的方法），对他们表现的差异进行t检验  
**问题**：训练集有一定的重叠，使得测试错误率并不独立（每个测试集都被使用了k-1次），会高估假设成立的概率，我们采用"5\*2交叉检验"  
**做法**：做5次2折交叉检验，仅计算第一次二折上的平均值，计算每次二折的方差  

#### 2.4.3 McNemar检验

|  A\B  |   正确   |   错误   |
| :---: | :------: | :------: |
| 正确  | $e_{00}$ | $e_{01}$ |
| 错误  | $e_{10}$ | $e_{11}$ |

假设两个模型性能相同，则应有$e_{01}=e_{10}$，变量$|e_{01}-e_{10}|$应该服从正态分布，均值为1，方差为$e_{01}+e_{10}$，因此变量
$$
\tau_{\chi^2}=\frac{(|e_{01}-e_{10}|-1)^2}{e_{01}+e_{10}}
$$

服从自由度为1的卡方分布?？。给定显著度，当上述变量小于临界值时。不能拒绝假设，否则认为两个模型有显著差异，选取平均值较小的那个  

#### 2.4.4 Friedman检验和Nemenyi后续检验

当一个数据集上有多个模型要检验的时候，固然可以用前述方法逐对检验，但是更快的方法是**基于算法排序的Friedman检验**  

考虑我们在$N$个数据集上比较$k$个模型。首先对他们各自进行性能评估（用前述方法），然后在每一个数据集上对性能排序，对所有模型按表现进行排序，并根据他们的排序结果赋予每个模型一个*序值*（第一名得1分，第3名3分，并列就平分序值之和）。  
  
先看看这些模型能不能看成表现相同  
令$r_i$表示第$i$个算法的**平均序值**，则$r_i$服从正态分布，其均值和方差分别为$(k+1)/2,\space (k^2-1)/12$，变量  
$$
\tau_{\chi^2}=\frac{k-1}{k}\cdot\frac{12N}{k^2-1}\cdot\displaystyle\sum_{i=1}^{k}(r_i-\frac{k+1}{2})^2
$$
在$k，N$都较大的时候服从自由度为$k-1$的$\chi^2$分布  
然而上述模型过于保守，现在通常用变量  
$$
\tau_F=\frac{(N-1)\cdot \tau_{\chi^2}}{N\cdot(k-1)-\tau_{\chi^2}}
$$
其中$\tau_{\chi^2}$由上上式得，$\tau_F$服从自由度为$k-1$和$(k-1)\cdot(N-1)$的F分布  
  
若“所有模型性能相同”这个假设被拒绝，说明模型性能显著不同，此时进行后续检验，常用的有Nemenyi后续检验  
Nemenyi检验计算出平均序值差别的临界值域：  
$$
CD=q_\alpha\sqrt{\frac{k(k+1)}{6N}}
$$
若两个模型平均序值之差超出临界值，则以相应的置信度拒绝“两个模型性能相同“这个假设  
  
画出Friefman检验图：横轴为平均序值。纵轴是不同算法，并在水平方向上以每一个均值为中心，向左右CD范围内画线段，线段在竖直方向上有重叠的模型认为是相近的模型，其他按均值排序。  

### 2.5 偏差与方差

- **偏差**(bias)：度量了模型的预测和真实结果间的差别，刻画模型本身的拟合能力
- **方差**(var)：度量了同样大小的训练集的变动导致的性能变化，刻画了数据扰动造成的影响
- **噪声**：表达了在当前任务上任何模型的误差下界，刻画了问题本身的难度

bias v.s. variance ：要在过拟合，欠拟合之间找到一个平衡  

---

## 3 线性模型

### 3.1 基本形式

给定由$d$个属性描述的示例$x = (x_1;x_2;...;x_d)$，其中$x_i$是$x$在第$i$个属性上的取值，线性模型试图学得[ 通过属性的线性组合来进行预测 ]的函数，用向量的方式就写成  
$$
f(x)=w^Tx+b
$$

### 3.2 线性回归

我们试图学得$f(x_i)=w^Tx_i+b,\space s.t.\space f(x_i)\simeq y_i$  
可利用最小二乘法对w, b进行估计。记$\hat{w}=(w;b)$，相应的，把数据集表示为一个$m\times (d+1)$的矩阵X：
$$
X=
\left(
\begin{matrix}
x_1^T & 1\\
x_2^t & 1\\
\vdots & \vdots\\
x_m^T & 1
\end{matrix}
\right) \tag{1}
$$
再把标记也写成向量形式$y=(y_1;y_2;...;y_m)$，则有下式：  
$$
\hat{w}^*=\argmin_{\hat{w}}(y-X\hat{w})^T(y-X\hat{w})
$$
其中$X\hat{w}$即预测的y值，上式即使偏差的平方和  
  
令$E_{\hat{w}}=(y-X\hat{w})^T(y-X\hat{w})$，E对$\hat{w}$求偏导：
$$
\frac{\partial E_{\hat{w}}}{\partial \hat{w}}=2X^T(X\hat{w}-y)
$$
解上式等于0即可得到w  
?开始懵逼？  
当$X^TX$为满秩矩阵时或正定矩阵时
$$
\hat{w}^*=(X^TX)^{-1}X^Ty
$$
令$\hat{x}_i=(x_i,1)$,最终的模型为  
$$
f(\hat{x}_i)=\hat{x}_i^T(X^TX)^{-1}X^Ty
$$
然而实际任务中$X^TX$往往不是满秩矩阵，此时可以解出多个$\hat{w}$，常见的做法是引入正则化  
?结束懵逼？  
更一般的，考虑单调可微函数$g(\cdot)$，令  
$$
y = g^{-1}(w^Ts+b)
$$
这样得到的模型称为“广义线性模型”，其中$g(\cdot)$称为“联系函数”  

### 3.3 对数几率回归（logistic regression）

对于分类问题，产生的结果$y\in \{0,1\}$，而线性回归产生的是实值，我们要将实值转换成0/1值，最理想的是分段函数，但是他并不连续，不可微，于是我们希望找到一定程度上近似的替代函数，通常用对数几率函数（logistic function）代替：  
$$
y = \frac{1}{1+e^{-z}}\tag{*}
$$
变换后得：
$$
ln\frac{y}{1-y}=w^Tx+b\tag{\#}
$$
若将y视为产生正例的可能性，则1-y是产生反例的可能性，两者的比例称为“几率”（odds），反映了x作为正例的相对可能性，对几率取对数得到对数几率（log odds，亦称logit）  
  
由此可看出（*）式实际上是在用线性模型的预测结果去逼近真实标记的对数几率，因此称这个模型为“对数几率回归”，他的优点有：  

- 无需事先假设数据的分布
- 不仅能预测类别，还能得到近似的概率
- 对数几率函数任意阶可导

求解w，b：将（#）式重写为：  
$$
ln\frac{p(y=1|x)}{p(y=0|x)}=w^Tx+b
$$
显然有
$$
p(y=1|x)=\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}}\tag{1}
$$
$$
p(y=0|x)=\frac{1}{1+e^{w^Tx+b}}\tag{2}
$$
于是我们可以用“极大似然法”估计w，b。对给定数据集，需要最大化如下函数（对数似然函数）：  
$$
l(w,b)=\displaystyle\sum_{i=1}^mln\space p(y_i|x_i;w,b)\tag{3}
$$
为了便于讨论，令$\beta=(w;b),\hat{x}=(x;1)$，则$w^Tx+b$可简写为$\beta^T\hat{x}$，再令$p_1(\hat{x};\beta)=p(y=1|\hat{x};\beta)$, $p_0(\hat{x};\beta)=p(y=0|\hat{x};\beta)$，则似然函数可以重写为：  
$$
p(y_i|x_i;w,b)=y_i\cdot p_1(\hat{x}_i;\beta)+(1-y_i)\cdot p_0(\hat{x}_i;\beta)
$$
结合上式和123式，最大化3式相当于最小化下式：  
$$
l(\beta)=\displaystyle\sum_{i=1}^m(-y_i\beta^T\hat{x}_i+ln(1+e^{\beta^T\hat{x}}))
$$
证明如下：  
![maximum likelihood](images/maximum_likelihood.jpg)  
$l(\beta)$是一个关于$\beta$的高阶可导连续凸函数，根据凸优化理论，可用牛顿法或者梯度下降迭代求解  

### 3.4 线性判别分析

> LDA : Linear Disciminant Analysis

给定训练样例集，设法将样例投影到一条直线上，使得同类的投影点尽可能接近，异类投影点尽可能远离。在对新样本进行分析时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别  
  
给定数据集$D=\{(x_i,y_i)\}_{i=1}^m,y_i\in \{0,1\}$，令$X_i, \mu_i,\Sigma_i$分别为第$i$类示例的集合、均值向量、协方差矩阵，若将数据投影到直线$w$上，则两类样本的中心在直线上的投影分别是$w^T\mu_0,w^T\mu_1$；若将所有样本点投影到直线上，两类样本的协方差分别为$w^T\Sigma_0w,w^T\Sigma_1w$？?  
  
欲使同类样本尽可能近，可以让同类样本的协方差尽可能小，即使$w^T\Sigma_0w+w^T\Sigma_1w$尽可能小；欲使异类样本尽可能远离，可以让异类样本中心距离尽可能大，即使$||w^T\mu_0-w^T\mu_1||_2^2$尽可能大。同时考虑两者，可得到最大化目标：  
$$
\begin{aligned}
J& = \frac{||w^T\mu_0-w^T\mu_1||_2^2}{w^T\Sigma_0w+w^T\Sigma_1w}
\\
&=\frac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{w^T(\Sigma_0+\Sigma_1)w}
\end{aligned}
$$
定义“类内散度矩阵”：
$$
\begin{aligned}
S_w&=\Sigma_0+\Sigma_1\\
& = \displaystyle\sum_{x\in X_0}(x-\mu_0)(x-\mu_0)^T+\displaystyle\sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^T
\end{aligned}
$$
以及“类间散度矩阵”：  
$$
S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T
$$
则J重写为：  
$$
J=\frac{w^TS_bw}{w^TS_ww}
$$
这就是LDA欲最大化的目标，即$S_b,S_w$的“广义瑞利商”  
  
如何确定$w$呢，注意到$J$的分子分母都是关于$w$的二次项，所以解和$w$的长度无关，仅与方向有关，不失一般性，令$w^TS_ww=1$，则$J$等价于：  
$$
\min_w\quad -w^TS_bw\\
s.t.\quad w^TS_ww=1
$$
由拉格朗日乘子法，上式等价于  
$$
S_bw=\lambda S_ww\tag{*}
$$
其中$\lambda$是拉格朗日乘子？?  
注意到$S_bw$的方向恒为$\mu_0-\mu_1$，不妨令  
$$
S_bw=\lambda(\mu_0-\mu_1)
$$
代入(*)式得  
$$
w=S^{-1}_w(\mu_0-\mu_1)
$$
实践中通常对$S_w$进行奇异值分解  
  
LDA可以从贝叶斯决策理论的角度阐释，并可证明，当两类数据同先验、满足高斯分布且协方差相同时，LDA可达最优分类：  
![LDA的贝叶斯证明](images/Bayes.jpg)  
可以推广到多分类问题，具体见西瓜书3.4末尾？?  

### 3.5 多分类问题

基本思路：拆解法，将多分类问题拆成若干二分类问题。拆分策略如下：  

- OvO（one vs one）
- OvR（one vs rest）
- MvM（many vs many）

OvO, OvR：不赘述，如图  
![OvO,OvR](images/OvO_OvR.jpg)  

MvM：每次将若干类作为正类，若干其他类作为反类，正反类要有特殊的构造，不能随意选取，一种最常用的技术叫“纠错输出码技术”（Error Correction Output Codes，ECOC）  
  
ECOC是将编码思想引入类别拆分，并尽可能在解码过程中具有容错性，主要工作分为两步：  
  
- 编码：对N个类别做M”次划分，每次划分都将一部分类别划为正类，一部分划为反类，从而形成一个二分类训练集；一共产生M个训练集，训练出M个分类器
- M个分类器分别对测试样本进行预测，这些预测标记组成一个编码。将这些编码和各个类别的编码进行比较，返回其中距离最小的类别作为预测结果

类别划分通过“编码矩阵”制定，编码矩阵有多种形式，常见的主要有二元码（指定为正反两类）和三元码（额外制定一个停用类）  
  
为什么称为“纠错输出码”呢，因为在测试阶段ECOC对分类器的错误有一定容忍和修正能力？?  

### 3.6 类别不平衡问题

#### 再放缩

执行对数概率预测时令  
$$
\frac{y'}{1-y'}=\frac{y}{1-y}\times \frac{m^-}{m^+}\tag{*}
$$
其中$m^+,m^-$代表正例、反例数目  
  
实际操作时的问题：未必能有效的基于训练数据观测几率来推断真实几率  

#### 欠采样

直接去除较多数据中的部分（代表算法EasyEnsenble，利用集成学习机制，将反例划分为若干集合供不同模型使用）

#### 过采样

增加一些少的例子（不是简单的重复采样，而是对正例进行插值来产生正例，代表算法有SMOTE）

#### 阀值移动

训练方法不变，预测时调整阀值为(*)式  

---

## 4 决策树