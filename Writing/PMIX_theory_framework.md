# PMIX 理论框架：形式化定义、IGM 保证与架构认证

> 工作标题：**Beyond Hypernetworks: Partial Monotonic Mixing for Cooperative Multi-Agent Reinforcement Learning**
>
> 本文档是论文理论章节底稿。目标是将 PMIX 定义为统一的学习范式，而不是把三种 mixer 简单并列。

## 0. 推荐结构

论文理论部分建议按以下顺序组织：

1. **Value Factorisation and IGM**：给出 CTDE、个体效用和 IGM 的问题定义。
2. **Partial Monotonic Mixing**：形式化定义可容许上下文、条件部分单调函数和 Direct-Context PMIX。
3. **PMIX Guarantees IGM**：给出统一主定理和完整证明。
4. **Certified PMIX Instantiations**：分别给出 PMIX-MLP、PMIX-Lattice、PMIX-KAN 的构造和认证命题。
5. **Approximation Capacity and Scope**：统一定义目标函数空间并讨论逼近能力，再分别说明三种实例能够安全主张的理论结论。

逼近能力最好统一放在三个架构之后。单调认证回答“任意参数下是否保持 IGM”，逼近能力回答“容量增加时能否覆盖目标函数类”，两者不是同一个问题。每个架构小节末尾只保留简短的 **Expressivity note**，统一指向第 5 节。

---

## 1. Value Factorisation and IGM

考虑包含 $n$ 个智能体的合作式 Dec-POMDP。时刻 $t$，智能体 $i$ 根据局部动作观测历史 $\tau_i$ 选择动作 $u_i\in\mathcal U_i$。联合动作为

$$
\mathbf u=(u_1,\ldots,u_n)\in\mathcal U
=\mathcal U_1\times\cdots\times\mathcal U_n.
$$

个体效用网络输出

$$
q_i(u_i)=Q_i(\tau_i,u_i;\theta_Q),
$$

并记

$$
\mathbf q(\mathbf u)
=\big(q_1(u_1),\ldots,q_n(u_n)\big)^\top.
$$

集中训练阶段可使用全局状态 $s$，或更一般的集中式历史 $\tau^c$；执行阶段只使用各智能体的局部效用网络。

### 定义 1（IGM）

若集中式联合价值与个体效用满足

$$
\mathop{\mathrm{arg\,max}}_{\mathbf u\in\mathcal U}
Q_{\mathrm{tot}}(\boldsymbol\tau,\mathbf u,s)
=
\prod_{i=1}^{n}
\mathop{\mathrm{arg\,max}}_{u_i\in\mathcal U_i}
Q_i(\tau_i,u_i),
$$

则称该价值分解满足 Individual-Global-Max（IGM）原则。

如果 mixer 仅为非递减函数，平坦区间可能使集中式价值出现额外最优动作。足以支持分散贪心执行的严谨表述是

$$
\prod_{i=1}^{n}\mathop{\mathrm{arg\,max}}_{u_i}Q_i(\tau_i,u_i)
\subseteq
\mathop{\mathrm{arg\,max}}_{\mathbf u}
Q_{\mathrm{tot}}(\boldsymbol\tau,\mathbf u,s).
\qquad\text{(1)}
$$

当 mixer 对每个 $q_i$ 严格递增且个体最优动作唯一时，式 (1) 可加强为 argmax 等式。

---

## 2. Partial Monotonic Mixing 的形式化定义

### 2.1 可容许的集中式上下文

令

$$
\mathbf z=E_\psi(c),
$$

其中 $c=s$ 或 $c=\tau^c$。

### 定义 2（Action-invariant context）

若对于同一决策时刻的任意两个候选联合动作 $\mathbf u,\mathbf u'$，都有

$$
c(\boldsymbol\tau,s,\mathbf u)
=c(\boldsymbol\tau,s,\mathbf u'),
\qquad\text{(2)}
$$

则称 $c$ 是当前决策的动作不变上下文。

全局状态、当前动作选择前的联合历史以及由它们得到的编码满足式 (2)。包含当前待优化联合动作的信息通常不满足该条件。该条件是后续 IGM 证明中“比较不同动作时 $\mathbf z$ 保持不变”的必要前提。

### 2.2 条件部分单调函数

在 $\mathbb R^n$ 上定义逐坐标偏序

$$
\mathbf q\preceq\mathbf q'
\quad\Longleftrightarrow\quad
q_i\le q_i',\quad i=1,\ldots,n.
\qquad\text{(3)}
$$

### 定义 3（Conditionally partially monotone function）

设

$$
M_\phi:\mathcal Q\times\mathcal Z\rightarrow\mathbb R.
$$

如果

$$
\forall\mathbf z\in\mathcal Z,\qquad
\mathbf q\preceq\mathbf q'
\Longrightarrow
M_\phi(\mathbf q,\mathbf z)
\le M_\phi(\mathbf q',\mathbf z),
\qquad\text{(4)}
$$

则称 $M_\phi$ 相对于 $\mathbf q$ 条件部分单调，而相对于 $\mathbf z$ 自由。

当 $M_\phi$ 关于 $\mathbf q$ 连续可微，且 $\mathcal Q$ 是矩形域（更一般地，对逐坐标线段封闭）时，式 (4) 等价于

$$
\frac{\partial M_\phi(\mathbf q,\mathbf z)}{\partial q_i}
\ge 0,\qquad i=1,\ldots,n,
\qquad\text{(5)}
$$

而不对

$$
\frac{\partial M_\phi(\mathbf q,\mathbf z)}{\partial z_j}
$$

施加符号约束。

这里的“条件”表示：每个固定上下文 $\mathbf z$ 索引一个关于 $\mathbf q$ 的单调函数

$$
M_{\phi,\mathbf z}(\mathbf q):=M_\phi(\mathbf q,\mathbf z),
$$

而 $\mathbf z\mapsto M_{\phi,\mathbf z}$ 可以自由变化。

### 定义 4（PMIX）

PMIX 将联合价值表示为

$$
Q_{\mathrm{tot}}(\boldsymbol\tau,\mathbf u,c)
=V_\omega(\mathbf z)
+M_\phi\big(\mathbf q(\mathbf u),\mathbf z\big),
\qquad \mathbf z=E_\psi(c),
\qquad\text{(6)}
$$

其中：

- $E_\psi$ 是无约束上下文编码器；
- $V_\omega$ 是仅依赖上下文的自由基线；
- $M_\phi$ 满足式 (4)；
- $c$ 满足定义 2。

$V_\omega(\mathbf z)$ 不依赖当前动作，因此可以拟合状态价值平移，但不会改变联合动作排序。

### 定义 5（Direct-Context PMIX）

若 $M_\phi(\mathbf q,\mathbf z)$ 使用一组与输入无关的固定可学习参数 $\phi$，并将 $\mathbf z$ 作为普通激活输入参与计算，则称其为 Direct-Context PMIX。

QMIX 的典型参数化是

$$
Q_{\mathrm{tot}}^{\mathrm{QMIX}}
=m\big(\mathbf q;H_\eta(s)\big),
\qquad\text{(7)}
$$

其中 hypernetwork $H_\eta$ 根据状态生成另一个网络的权重或偏置。PMIX 直接学习

$$
M_\phi\big(\mathbf q,E_\psi(c)\big).
\qquad\text{(8)}
$$

这个区别是参数化方式和归纳偏置上的区别。不能声称式 (7) 无法整体写成 $(\mathbf q,s)$ 的函数，也不应声称 QMIX 完全不属于状态条件单调函数。

### 引理 1（条件部分单调函数的闭包性质）

设 $f_r(\mathbf q,\mathbf z)$ 均对 $\mathbf q$ 非递减，则：

1. 对任意 $a_r\ge0$，$\sum_r a_rf_r$ 对 $\mathbf q$ 非递减；
2. 对任意仅依赖 $\mathbf z$ 的 $a(\mathbf z)\ge0$ 和 $b(\mathbf z)$，
   $a(\mathbf z)f_r+b(\mathbf z)$ 对 $\mathbf q$ 非递减；
3. 若 $g$ 是非递减标量函数，则 $g\circ f_r$ 对 $\mathbf q$ 非递减；
4. 若每个 $C_i:\mathbb R\to\mathbb R$ 非递减，则

$$
f_r(C_1(q_1),\ldots,C_n(q_n),\mathbf z)
$$

仍对原始 $q_i$ 非递减。

**证明。** 固定 $\mathbf z$，对任意 $\mathbf q\preceq\mathbf q'$，有

$$
f_r(\mathbf q,\mathbf z)\le f_r(\mathbf q',\mathbf z).
$$

非负加权求和保持不等号方向；两侧乘以相同的 $a(\mathbf z)\ge0$ 并加入相同的
$b(\mathbf z)$ 不改变次序；非递减函数作用于不等式两侧保持次序；逐坐标非递减校准满足

$$
\mathbf C(\mathbf q)\preceq\mathbf C(\mathbf q').
$$

将其代入 $f_r$ 即得第 4 条。证毕。

该引理统一了三篇来源论文反复使用的非负求和、单调复合、状态项固定和单调校准等证明工具。

---

## 3. PMIX 保持 IGM

### 定理 1（Conditional partial monotonicity implies IGM）

考虑式 (6) 定义的 PMIX。若上下文 $c$ 动作不变，且 $M_\phi$ 满足式 (4)，则 PMIX 满足式 (1) 的 IGM 包含关系。

**证明。** 对每个智能体任取局部贪心动作

$$
u_i^*\in
\mathop{\mathrm{arg\,max}}_{u_i\in\mathcal U_i}
Q_i(\tau_i,u_i),
$$

并记 $\mathbf u^*=(u_1^*,\ldots,u_n^*)$。对任意联合动作 $\mathbf u=(u_1,\ldots,u_n)$，根据局部最大值定义，

$$
Q_i(\tau_i,u_i)
\le Q_i(\tau_i,u_i^*),
\qquad i=1,\ldots,n.
\qquad\text{(9)}
$$

由偏序定义，式 (9) 等价于

$$
\mathbf q(\mathbf u)\preceq\mathbf q(\mathbf u^*).
\qquad\text{(10)}
$$

因为 $c$ 动作不变，所以比较 $\mathbf u$ 与 $\mathbf u^*$ 时

$$
\mathbf z=E_\psi(c)
$$

完全相同。将式 (10) 代入条件部分单调性式 (4)，得到

$$
M_\phi(\mathbf q(\mathbf u),\mathbf z)
\le
M_\phi(\mathbf q(\mathbf u^*),\mathbf z).
\qquad\text{(11)}
$$

在式 (11) 两侧加入相同的上下文基线，

$$
\begin{aligned}
Q_{\mathrm{tot}}(\boldsymbol\tau,\mathbf u,c)
&=V_\omega(\mathbf z)
 +M_\phi(\mathbf q(\mathbf u),\mathbf z)\\
&\le V_\omega(\mathbf z)
 +M_\phi(\mathbf q(\mathbf u^*),\mathbf z)\\
&=Q_{\mathrm{tot}}(\boldsymbol\tau,\mathbf u^*,c).
\end{aligned}
\qquad\text{(12)}
$$

式 (12) 对任意 $\mathbf u\in\mathcal U$ 成立，因此任意由个体贪心动作组成的 $\mathbf u^*$ 都是 $Q_{\mathrm{tot}}$ 的全局最大化动作。故式 (1) 成立。证毕。

### 推论 1（严格 IGM）

若每个局部最优动作唯一，且固定 $\mathbf z$ 后 $M_\phi$ 对每个 $q_i$ 严格递增，则任何非个体贪心联合动作至少在一个坐标上严格劣于 $\mathbf u^*$，式 (12) 成为严格不等式。因此

$$
\mathop{\mathrm{arg\,max}}_{\mathbf u}Q_{\mathrm{tot}}
=
\prod_i\mathop{\mathrm{arg\,max}}_{u_i}Q_i.
$$

---

## 4. 三种可认证的 PMIX 实例

三个实例共享式 (6) 的外层结构以及相同的 $E_\psi$ 和 $V_\omega$。
差异只位于 $M_\phi$ 如何表示条件部分单调函数。

### 4.1 PMIX-MLP

#### 4.1.1 Activation-switch 单调层

根据 Sartor et al. 的 post-activation switch，令

$$
W^+=\max(W,0),\qquad W^-=\min(W,0),
$$

并定义

$$
\mathcal A_W(x)
=W^+\sigma(x)+W^-\sigma(-x)+b,
\qquad\text{(13)}
$$

其中 $\sigma$ 非递减。

### 引理 2（Activation-switch 层单调性）

$\mathcal A_W$ 对每个输入坐标非递减。

**序关系证明。** 若 $x\preceq x'$，则

$$
\sigma(x)\preceq\sigma(x'),
\qquad
\sigma(-x)\succeq\sigma(-x').
$$

因为 $W^+\ge0$，

$$
W^+\sigma(x)\preceq W^+\sigma(x').
$$

因为 $W^-\le0$，第二个不等式乘以 $W^-$ 后方向反转：

$$
W^-\sigma(-x)\preceq W^-\sigma(-x').
$$

两式相加并加入相同偏置，得到

$$
\mathcal A_W(x)\preceq\mathcal A_W(x').
$$

证毕。

可微情况下也可以直接计算

$$
\frac{\partial\mathcal A_W}{\partial x}
=W^+\operatorname{diag}(\sigma'(x))
-W^-\operatorname{diag}(\sigma'(-x))
\succeq0.
\qquad\text{(14)}
$$

#### 4.1.2 部分单调输入层

第一层写成

$$
h^{(1)}
=A_q(\mathbf q)+B_z(\mathbf z),
\qquad\text{(15)}
$$

其中 $A_q$ 是对 $\mathbf q$ 非递减的 activation-switch 映射，$B_z$ 是无约束状态映射。后续层为

$$
h^{(\ell+1)}=\mathcal A_{W_\ell}(h^{(\ell)}),
\qquad
M_{\mathrm{MLP}}=\mathcal A_{W_L}(h^{(L)}).
\qquad\text{(16)}
$$

### 命题 1（PMIX-MLP 认证）

式 (15)-(16) 定义的 $M_{\mathrm{MLP}}$ 对 $\mathbf q$ 条件部分单调。

**证明。** 固定任意 $\mathbf z$，取 $\mathbf q\preceq\mathbf q'$。由 $A_q$ 单调，

$$
A_q(\mathbf q)\preceq A_q(\mathbf q').
$$

两侧加入相同的自由状态项 $B_z(\mathbf z)$，得到

$$
h^{(1)}(\mathbf q,\mathbf z)
\preceq
h^{(1)}(\mathbf q',\mathbf z).
$$

由引理 2，每个后续层都是坐标非递减映射。逐层归纳可得

$$
M_{\mathrm{MLP}}(\mathbf q,\mathbf z)
\le
M_{\mathrm{MLP}}(\mathbf q',\mathbf z).
$$

故 PMIX-MLP 满足定义 3，并由定理 1 保持 IGM。证毕。

**Expressivity note.** AMCO 来源论文的 Theorem 3.5 与 Proposition 3.9 证明至少四层、交替饱和方向的受约束 MLP 可以逼近全单调函数；Proposition 4.1 说明 activation-switch 包含相应构造。但从全单调 UAT 到式 (15) 的任意条件部分单调函数 UAT 仍需要额外论证，不能直接把原定理改名引用。

### 4.2 PMIX-Lattice

#### 4.2.1 单调坐标与条件顶点值

对每个个体效用使用非递减校准器

$$
\bar q_i=C_i(q_i)\in[0,1].
\qquad\text{(17)}
$$

在 $n$ 维单位超立方体上构建 lattice，顶点集合记为 $\mathcal V$。对每个顶点 $v\in\mathcal V$，固定参数网络计算

$$
g_\theta(\mathbf z,v)\in[0,1].
\qquad\text{(18)}
$$

这里 $(\mathbf z,v)$ 是 $g_\theta$ 的普通输入，$\theta$ 不由状态生成。式 (18) 是 direct-context function evaluation，而不是生成另一网络权重的 hypernetwork。实现中一次输出全部顶点值只是对 $g_\theta(\mathbf z,v)$ 的向量化。

令 $L_{v,\pi}$ 和 $U_{v,\pi}$ 分别为 HLL 总顺序 $\pi$ 下的最小被支配集合和最小支配集合。定义

$$
\ell(\mathbf z,v)=
\begin{cases}
\max_{u\in L_{v,\pi}}F(\mathbf z,u),&L_{v,\pi}\neq\varnothing,\\
0,&L_{v,\pi}=\varnothing,
\end{cases}
\qquad\text{(19)}
$$

$$
r(\mathbf z,v)=
\begin{cases}
\min_{u\in U_{v,\pi}}F(\mathbf z,u),&U_{v,\pi}\neq\varnothing,\\
1,&U_{v,\pi}=\varnothing.
\end{cases}
\qquad\text{(20)}
$$

顶点值递归定义为

$$
F(\mathbf z,v)
=
\big(1-g_\theta(\mathbf z,v)\big)\ell(\mathbf z,v)
+g_\theta(\mathbf z,v)r(\mathbf z,v).
\qquad\text{(21)}
$$

由于 $g_\theta\in[0,1]$，

$$
\ell(\mathbf z,v)
\le F(\mathbf z,v)
\le r(\mathbf z,v).
\qquad\text{(22)}
$$

令 $\widetilde M_{\mathrm{Lat}}(\mathbf q,\mathbf z)$ 表示使用这些有序顶点值对
$\mathbf C(\mathbf q)$ 进行插值得到的归一化 lattice 输出。由于
$\widetilde M_{\mathrm{Lat}}\in[0,1]$，PMIX-Lattice 使用架构内部的反归一化

$$
M_{\mathrm{Lat}}(\mathbf q,\mathbf z)
=A_\xi(\mathbf z)
\left(\widetilde M_{\mathrm{Lat}}(\mathbf q,\mathbf z)-\frac12\right),
\qquad
A_\xi(\mathbf z)
=\varepsilon+\operatorname{softplus}(a_\xi(\mathbf z))>0.
\qquad\text{(22a)}
$$

$A_\xi$ 是 PMIX-Lattice 核内部的 range calibration，而不是 PMIX 外层定义或其他
变式必须共享的模块。

### 命题 2（PMIX-Lattice 认证）

若使用保持顶点单调序的 multilinear 或 simplex interpolation，则 PMIX-Lattice 对所有原始 $q_i$ 条件部分单调。

**证明。** 固定 $\mathbf z$。设顶点 $v$ 支配顶点 $u$。若 $u$ 在总顺序中先于 $v$，根据 HLL 最小被支配集合的定义，存在链

$$
u=v_0\prec v_1\prec\cdots\prec v_J=v,
\qquad
v_{j-1}\in L_{v_j,\pi}.
$$

由式 (19)、(21)-(22)，

$$
F(\mathbf z,v_{j-1})
\le \ell(\mathbf z,v_j)
\le F(\mathbf z,v_j).
$$

沿链传递得到

$$
F(\mathbf z,u)\le F(\mathbf z,v).
\qquad\text{(23)}
$$

若 $v$ 在总顺序中先于 $u$，则使用 $U_{v,\pi}$ 和式 (20) 构造反向链，同样得到式 (23)。因此所有 lattice 顶点满足逐 Q 轴偏序。该步骤对应 HLL Proposition 3.4。

在任一 cell 内，multilinear interpolation 关于第 $i$ 个坐标的偏导是该轴两端顶点差值的非负凸组合除以正的网格宽度，所以

$$
\frac{\partial \widetilde M_{\mathrm{Lat}}}{\partial\bar q_i}\ge0.
$$

simplex interpolation 同样保持相邻顶点偏序。最后由式 (17) 和链式法则，

$$
\frac{\partial M_{\mathrm{Lat}}}{\partial q_i}
=
A_\xi(\mathbf z)
\frac{\partial \widetilde M_{\mathrm{Lat}}}{\partial\bar q_i}
\frac{\partial C_i}{\partial q_i}
\ge0.
$$

故 PMIX-Lattice 满足定义 3，并由定理 1 保持 IGM。证毕。

#### 4.2.2 大规模智能体组合

完整 lattice 顶点数为 $\prod_i k_i$，对大 $n$ 呈指数增长。可以使用多个低维 HLL：

$$
M_{\mathrm{Lat}}(\mathbf q,\mathbf z)
=
\sum_{r=1}^{R}a_rL_r(\mathbf q_{S_r},\mathbf z),
\qquad a_r\ge0,
\qquad\text{(24)}
$$

并要求每个 $q_i$ 至少出现在一个子集 $S_r$ 中。由引理 1，式 (24) 仍满足部分单调性。若每组至多包含 $m\ll n$ 个 Q 坐标，复杂度由 $O(k^n)$ 降为约 $O(Rk^m)$。

**Expressivity note.** HLL 来源论文严格证明了顶点偏序、插值单调性和复杂度，但没有给出完整 UAT。其网格逼近能力应作为本文的新推论在第 5 节单独表述。

### 4.3 PMIX-KAN

#### 4.3.1 KAN 边函数与 Hermite 条件

KAN 第 $l$ 层从节点 $i$ 到节点 $j$ 的边函数写成

$$
\Phi_{l,j,i}(x)
=
\omega^\varphi_{l,j,i}\varphi_{l,j,i}(x)
+\omega^b_{l,j,i}b(x),
\qquad\text{(25)}
$$

节点聚合为

$$
x_{l+1,j}
=
\sum_{i=1}^{n_l}\Phi_{l,j,i}(x_{l,i})+\theta_{l,j}.
\qquad\text{(26)}
$$

$\varphi$ 是 cubic Hermite spline，$b$ 是非递减基础激活。

在区间 $[x_k,x_{k+1}]$ 上，令

$$
t=\frac{x-x_k}{x_{k+1}-x_k},
\qquad \Delta x_k=x_{k+1}-x_k.
$$

Hermite 段为

$$
\begin{aligned}
p_k(x)=&\ h_{00}(t)y_k
+h_{10}(t)\Delta x_km_k\\
&+h_{01}(t)y_{k+1}
+h_{11}(t)\Delta x_km_{k+1},
\end{aligned}
\qquad\text{(27)}
$$

其中

$$
h_{00}=2t^3-3t^2+1,\quad
h_{10}=t^3-2t^2+t,
$$

$$
h_{01}=-2t^3+3t^2,\quad
h_{11}=t^3-t^2.
$$

定义

$$
d_k=\frac{y_{k+1}-y_k}{x_{k+1}-x_k},
\qquad
\alpha_k=\frac{m_k}{d_k},
\qquad
\beta_k=\frac{m_{k+1}}{d_k}.
\qquad\text{(28)}
$$

根据 Fritsch-Carlson 条件，一个充分的递增约束是

$$
y_{k+1}\ge y_k,\qquad m_k,m_{k+1}\ge0.
\qquad\text{(29)}
$$

当 $d_k>0$ 时还要求

$$
\alpha_k^2+\beta_k^2\le9,
\qquad\text{(30)}
$$

当 $d_k=0$ 时要求

$$
m_k=m_{k+1}=0.
\qquad\text{(31)}
$$

再结合

$$
\omega^\varphi_{l,j,i}\ge0,
\qquad
\omega^b_{l,j,i}\ge0,
\qquad\text{(32)}
$$

即可使式 (25) 非递减。

第一层中，从每个 $q_i$ 出发的边满足式 (29)-(32)，从状态坐标 $z_j$ 出发的边完全自由。所有承载 Q 影响的后续边均满足式 (29)-(32)。

### 命题 3（PMIX-KAN 认证）

在上述约束下，PMIX-KAN 对每个 $q_i$ 条件部分单调。

**证明。** 固定任意 $\mathbf z$ 和任一 Q 输入 $q_r$。

**基础层。** 由式 (29)-(31)，从 $q_r$ 出发的 Hermite spline 在每个网格区间非递减。使用端点斜率线性外推后，该性质扩展到整个实数域。由式 (32) 和基础激活 $b$ 的非递减性，每条从 $q_r$ 出发的 $\Phi_{0,j,r}$ 非递减。式 (26) 是这些边输出与固定状态边输出的求和，因此第一层相关节点对 $q_r$ 非递减。

**归纳假设。** 假设第 $l$ 层所有受 $q_r$ 影响的节点 $x_{l,i}$ 均对 $q_r$ 非递减。

**归纳步骤。** 后续边 $\Phi_{l,j,i}$ 根据式 (29)-(32) 对输入 $x_{l,i}$ 非递减。因此复合映射

$$
q_r\longmapsto x_{l,i}(q_r)
\longmapsto\Phi_{l,j,i}(x_{l,i}(q_r))
$$

非递减。式 (26) 对这些复合项求和，由引理 1 仍然非递减。因此第 $l+1$ 层性质成立。

由层数归纳，最终输出满足

$$
\frac{\partial M_{\mathrm{KAN}}}{\partial q_r}\ge0.
$$

该结论对所有 $r=1,\ldots,n$ 成立，故 PMIX-KAN 满足定义 3，并由定理 1 保持 IGM。该证明对应 MonoKAN Theorem 3 和 Appendix B 的逐层归纳结构。证毕。

**Expressivity note.** MonoKAN 论文证明 Hermite spline 对足够光滑的一维函数具有一致逼近能力，并证明网络的全域部分单调认证；它没有直接证明受上述路径约束的多变量 MonoKAN 对所有连续部分单调函数稠密。

---

## 5. Approximation Capacity and Scope

### 5.1 为什么统一放在三个实例之后

前三个命题回答：

> 给定任意参数，网络是否必然满足 IGM 所需的部分单调约束？

逼近能力回答：

> 当容量增加时，模型是否能逼近任意目标条件部分单调函数？

前者是认证，后者是函数类稠密性。把二者分别处理，可以避免将“能表达非凸单调函数”“一维 spline 一致收敛”误写成“完整多变量架构具有 UAT”。

### 5.2 PMIX 的目标函数空间

令 $\mathcal Q\subset\mathbb R^n$ 和 $\mathcal Z\subset\mathbb R^d$ 为紧集，定义

$$
\mathcal C_{\uparrow}(\mathcal Q\times\mathcal Z)
=
\left\{
f\in C(\mathcal Q\times\mathcal Z):
\mathbf q\preceq\mathbf q'
\Rightarrow
f(\mathbf q,\mathbf z)
\le f(\mathbf q',\mathbf z)
\right\}.
\qquad\text{(33)}
$$

理想的条件部分单调逼近器族 $\mathcal M$ 应满足：对任意 $f\in\mathcal C_{\uparrow}$ 和 $\varepsilon>0$，存在 $M\in\mathcal M$，使

$$
\sup_{(\mathbf q,\mathbf z)\in\mathcal Q\times\mathcal Z}
|f(\mathbf q,\mathbf z)-M(\mathbf q,\mathbf z)|
<\varepsilon.
\qquad\text{(34)}
$$

### 定理 2（条件单调函数的有限混合逼近）

对任意 $f\in\mathcal C_{\uparrow}(\mathcal Q\times\mathcal Z)$ 和 $\varepsilon>0$，存在有限个状态中心 $\mathbf z_1,\ldots,\mathbf z_R$ 和连续非负函数

$$
\rho_r:\mathcal Z\to[0,1],
\qquad
\sum_{r=1}^{R}\rho_r(\mathbf z)=1,
\qquad\text{(35)}
$$

使得

$$
\widehat f(\mathbf q,\mathbf z)
=
\sum_{r=1}^{R}\rho_r(\mathbf z)
f(\mathbf q,\mathbf z_r)
\qquad\text{(36)}
$$

仍属于 $\mathcal C_{\uparrow}$，且

$$
\|f-\widehat f\|_\infty<\varepsilon.
\qquad\text{(37)}
$$

**证明。** 因为 $f$ 在紧集 $\mathcal Q\times\mathcal Z$ 上连续，所以一致连续。存在 $\delta>0$，使得当

$$
\|\mathbf z-\mathbf z'\|<\delta
$$

时，对所有 $\mathbf q\in\mathcal Q$ 均有

$$
|f(\mathbf q,\mathbf z)-f(\mathbf q,\mathbf z')|
<\varepsilon.
\qquad\text{(38)}
$$

由 $\mathcal Z$ 紧，可取有限个半径小于 $\delta$ 的开集覆盖，并取从属于该覆盖的连续 partition of unity $\{\rho_r\}_{r=1}^{R}$。每个 $\rho_r$ 非负、满足式 (35)，且其支撑集中的状态与对应中心 $\mathbf z_r$ 距离小于 $\delta$。因此

$$
\begin{aligned}
|f(\mathbf q,\mathbf z)-\widehat f(\mathbf q,\mathbf z)|
&=
\left|
\sum_r\rho_r(\mathbf z)
[f(\mathbf q,\mathbf z)-f(\mathbf q,\mathbf z_r)]
\right|\\
&\le
\sum_r\rho_r(\mathbf z)
|f(\mathbf q,\mathbf z)-f(\mathbf q,\mathbf z_r)|\\
&<
\varepsilon\sum_r\rho_r(\mathbf z)
=\varepsilon.
\end{aligned}
$$

另一方面，对任意 $\mathbf q\preceq\mathbf q'$，每个固定状态切片 $f(\cdot,\mathbf z_r)$ 都单调，所以

$$
f(\mathbf q,\mathbf z_r)
\le f(\mathbf q',\mathbf z_r).
$$

乘以非负的 $\rho_r(\mathbf z)$ 并求和，得到

$$
\widehat f(\mathbf q,\mathbf z)
\le\widehat f(\mathbf q',\mathbf z).
$$

因此 $\widehat f\in\mathcal C_{\uparrow}$，式 (37) 成立。证毕。

### 5.3 定理 2 的正确定位

定理 2 给出 PMIX 范式级的表达能力解释：连续条件部分单调函数可以由有限个“关于 Q 单调的专家”的状态依赖非负凸组合一致逼近。状态权重 $\rho_r(\mathbf z)$ 可以任意非单调，而非负组合不会破坏 Q 单调性。

如果一种实现能够：

1. 逼近固定 $\mathbf z_r$ 下的单调 Q 切片；
2. 逼近连续非负状态权重；
3. 实现或一致逼近非负乘积与求和；

它就能继承式 (34) 的条件部分单调逼近能力。

但定理 2 不能自动证明当前三个具体计算图都完整实现了式 (36)。若论文要声称 PMIX-MLP、PMIX-Lattice、PMIX-KAN 三者都拥有 UAT，还需要逐一证明它们包含或能一致逼近该 gating-product 构造。

### 5.4 三个实例分别可以怎样介绍

#### PMIX-MLP

AMCO 的 activation-switch 网络包含对全单调 Q 切片具有通用逼近能力的构造。若进一步证明自由状态通路能够实现定理 2 的 partition-of-unity gating，才可以严格声称 PMIX-MLP 对式 (33) 稠密。

主文当前可以写：

> PMIX-MLP inherits the universal approximation capability for monotone Q-slices from activation-switch monotonic networks, while its unconstrained context pathway allows the represented monotone surface to vary with the centralized context.

不要写成“AMCO Theorem 3.5 已经直接证明 PMIX-MLP 对所有条件部分单调函数 universal”。

#### PMIX-Lattice

PMIX-Lattice 最容易补充独立网格逼近证明。对 $f\in\mathcal C_{\uparrow}$，在 Q 网格顶点取条件值 $f(\mathbf z,v)$。因为目标对 Q 单调，这些顶点值天然满足 HLL 偏序。随着最大网格直径 $h\to0$，连续函数的 multilinear interpolation 满足

$$
\|f-I_hf\|_\infty\to0.
\qquad\text{(39)}
$$

若辅助网络 $g_\theta(\mathbf z,v)$ 能一致逼近式 (21) 所需的条件系数，HLL 参数化即可逼近状态相关顶点值。正式证明需单独处理 $r(\mathbf z,v)=\ell(\mathbf z,v)$ 时系数不唯一的问题：此时可约定 $g=0$ 或任意固定值，因为式 (21) 与 $g$ 无关。

因此，可以在“紧定义域、目标连续、网格加密、辅助网络容量充分”的条件下为 PMIX-Lattice 给出较完整的稠密性命题。该命题是基于 HLL 构造的新推论，不应标成 HLL 原文定理。

#### PMIX-KAN

MonoKAN 提供两项可直接引用的理论：

1. cubic Hermite spline 对足够光滑的一维函数可一致逼近；
2. 满足式 (29)-(32) 的 KAN 在整个输入空间上可认证地部分单调。

Kolmogorov-Arnold 表示定理、单条 spline 的一致逼近能力，以及“受单调路径约束的多层 KAN 对式 (33) 稠密”不是同一个结论。除非补充新的 constrained-KAN UAT，否则建议将 PMIX-KAN 表述为具有灵活局部非线性和全域单调认证的 expressive approximator，而不是已被证明 universal 的 approximator。

### 5.5 推荐的理论声明层级

- **Theorem 1：** 所有满足 PMIX 定义的 mixer 都保持 IGM。
- **Propositions 1-3：** 三种架构都是 PMIX 的可认证实例。
- **Theorem 2：** PMIX 目标函数类存在状态依赖单调专家有限混合的一致逼近构造。
- **Architecture-specific discussion：** 说明三种实现与定理 2 充分构造之间的关系，不做超出来源论文的统一 UAT 声明。

如果后续希望主张“三种 PMIX 变式全部具有通用逼近能力”，需要新增三个独立的架构稠密性证明。当前更稳妥的贡献是：

> 统一 IGM 保证 + 三种不同归纳偏置的可认证实例 + 范式级逼近构造。

---

## 6. 公平实现对应的统一数学形式

三个主模型统一写成

$$
Q_{\mathrm{tot}}^{(a)}
=V_\omega(E_\psi(c))
+M_{\phi_a}^{(a)}(\mathbf q,E_\psi(c)),
\qquad
a\in\{\mathrm{MLP},\mathrm{Lat},\mathrm{KAN}\}.
\qquad\text{(40)}
$$

应统一：

- 状态编码器深度、宽度和输出维度；
- $V_\omega$ 的结构与是否启用；
- agent network、learner、优化器和训练预算；
- 参数量预算和调参预算；
- seeds、评估频率和模型选择规则。

不建议只给 PMIX-MLP 增加 Q residual annealing。若研究该机制，应定义范式级扩展

$$
Q_{\mathrm{tot}}^{(a,+R)}
=Q_{\mathrm{tot}}^{(a)}
+\lambda(t)\frac{1}{n}\sum_{i=1}^{n}q_i,
\qquad \lambda(t)\ge0,
\qquad\text{(41)}
$$

并对三个 PMIX 实例使用同一调度。由引理 1，式 (41) 不破坏 IGM，但它改变优化路径和归纳偏置，因此应作为统一消融，而不是某个实例的默认优势。

Lattice 的 $[0,1]$ 校准和 KAN 的 spline-domain 校准属于各自部分单调逼近器内部的必要参数化差异。只需保证校准器非递减并报告其形式，不必强行让 MLP 使用相同饱和变换。

当前 HLL 实现包含 state-dependent positive output scale。它用于把归一化、中心化的
lattice 输出恢复到联合价值所需的动态范围，因此定义在 PMIX-Lattice 核内部：

$$
Q_{\mathrm{tot}}^{(a)}
=V_\omega(\mathbf z)+M_{\phi_a}^{(a)}(\mathbf q,\mathbf z),
\qquad
M_{\mathrm{Lat}}
=A_\xi(\mathbf z)\left(\widetilde M_{\mathrm{Lat}}-\frac12\right),
\qquad A_\xi(\mathbf z)>0,
\qquad\text{(42)}
$$

这里的 $A_\xi$ 与式 (22a) 中的定义一致。它不是 MLP 和 KAN 必须共享的
公共模块。允许该差异的理由是：它解除 HLL 原生 $[0,1]$ 输出域的数值限制，而 MLP 和 KAN
可在各自核内部直接学习所需输出范围。因为 scale 为正，PMIX-Lattice 的单调认证和定理 1
均保持不变。

这与只给某个变式增加 VDN/Q residual 不同。正尺度是 HLL 原生表示的 range calibration；
residual 则绕过部分单调核，额外引入 additive credit shortcut，因而只能作为范式级统一消融。

---

## 7. 与三篇来源论文的对应关系

| PMIX 内容 | 主要来源 | 可直接继承 | 需要本文新证明 |
|---|---|---|---|
| PMIX 定义与 IGM | QMIX/IGM 背景 + 本文 | 单调性是 IGM 的充分条件 | Direct-Context PMIX 形式化与定理 1 |
| PMIX-MLP | Sartor et al., 2025 | Theorem 3.5、Propositions 3.9 和 4.1；activation-switch 与全单调切片表达力 | 自由状态输入下的命题 1；条件部分单调 UAT（若需要） |
| PMIX-Lattice | Yanagisawa et al., HLL | Definitions 3.1-3.3、Proposition 3.4 和复杂度分析 | 与 PMIX 的组合；网格加密下的条件逼近命题 |
| PMIX-KAN | Polo-Molina et al., MonoKAN | Hermite 条件、Theorem 3、Appendix B 的逐层认证 | 与 PMIX/IGM 的组合；受约束多变量 KAN UAT（若要声明） |
| 范式级逼近 | 本文 | 非负和与单调复合的标准闭包性质 | 定理 2 的条件单调专家混合构造 |

## 8. 当前建议

论文现阶段最合适的理论主线是：

1. 将 PMIX 定义为条件部分单调价值分解，而不是某个具体网络。
2. 用定理 1 证明整个范式保持 IGM。
3. 用三个命题证明 MLP、Lattice、KAN 都是 PMIX 的可认证实例。
4. 用定理 2 说明目标函数类存在统一的 direct-context 逼近构造。
5. 对三个实例分别准确标注已有逼近结论，不急于声称三者均具有完整 UAT。

这一结构能够支撑 “Beyond Hypernetworks” 的标题，同时不会把贡献弱化为简单的 mixer 替换实验。

## 9. 来源论文

1. Yanagisawa et al. **Hierarchical Lattice Layer for Partially Monotone Neural Networks**. 重点对应 Definitions 3.1-3.3、Proposition 3.4 和 Section 3.6。
2. Sartor, Sinigaglia, and Susto. **Advancing Constrained Monotonic Neural Networks: Achieving Universal Approximation Beyond Bounded Activations**. 重点对应 Theorem 3.5、Propositions 3.9、3.10、4.1 和 Equations (12)-(13)。
3. Polo-Molina, Alfaya, and Portela. **MonoKAN: Certified Monotonic Kolmogorov-Arnold Network**. 重点对应 Lemmas 1-2、Theorem 3 和 Appendix B。
