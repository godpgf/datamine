#Reinforcement Learning

##一、提出问题

###1.介绍

增强学习（reinforcementlearning, RL）又叫做强化学习，是近年来机器学习和智能控制领域的主要方法之一。  
> Reinforcement learning is learning what to do ----how to map situations to actions ---- so as to maximize a numerical reward signal.

也就是说增强学习关注的是智能体(Agent)如何在环境中采取一系列行为，从而获得最大的累积回报。通过增强学习，一个Agent应该知道在什么状态下应该采取什么行为。RL是从环境状态到动作的映射的学习，我们把这个映射称为策略。  
可以看到，增强学习和监督学习的区别主要有以下两点：  

1.  增强学习是试错学习(Trail-and-error)，由于没有直接的指导信息，智能体要以不断与环境进行交互，通过试错的方式来获得最佳策略。  
2.  延迟回报，增强学习的指导信息很少，而且往往是在事后（最后一个状态）才给出的，这就导致了一个问题，就是获得正回报或者负回报以后，如何将回报分配给前面的状态。  

增强学习的其中一个挑战就是在直觉给出的行为（exploitation）和探索新行为（exploration）之间进行试错（trial-and-error）。就像中午选择餐厅吃饭，可以凭借直觉选择已知的好吃的餐厅，也可以探索一个新餐厅（也行更加好吃）。

> Exploitation is the right thing to do to maximize the expected reward on the one play, but exploration may produce the greater total reward in the long run. 

####主要概念:

策略(policy)：state到action的映射

> Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states.

奖励函数（reward function）：某个action得到的直接奖励信号，比如吃个雪糕感觉高兴。所谓的Reward就是Agent执行了动作与环境进行交互后，环境会发生变化，变化的好与坏就用reward来表示。

> Roughly speaking, it maps each perceived state (or state-action pair) of the environment to a single number, a reward, indicating the intrinsic desirability of that state.

值函数（value function）：评价某个state长远来说的好坏，表示一个状态未来的潜在价值期望。

> A value function specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state.

###2.评价性反馈（Evaluative Feedback）

评价性反馈（Evaluative Feedback）反映了一个action直接（短期）的好坏，指导性反馈（instructive feedback）反映了一个action的准确性。督导学习就是指导性反馈，这很好区分：评价性反馈依赖已经做过的行为，指导性反馈不依赖。

> Purely evaluative feedback indicates how good the action taken is, but not whether it is the best or the worst action possible. Evaluative feedback is the basis of methods for function optimization, including evolutionary methods. Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken. 

动作值函数（Action-Value Methods）：  
前面我们引出了估值函数，考虑到每个状态之后都有多种动作可以选择，每个动作之下的状态又多不一样，我们更关心在某个状态下的不同动作的估值。显然，如果知道了每个动作的估值，那么就可以选择估值最好的一个动作去执行了。 

> 动作值函数类似于人类在某个情况下采取某种行为的直觉，它不是通过严谨的推理或想象，而仅仅凭借经验来决策的。在试错寻找最大回报策略的过程中，我们不能完全依靠直觉，但可以用直觉来剪枝。 
 
向前t步的估值表示为![qt](file:./RL/inimgtmp86.png)，在经过action a后接下来所有可选择的action的反馈奖励是![rt_all](file:./RL/inimgtmp91.png)，一种最简单的很自然的方法可以用均值评估a，写成：

![qt_formul](file:./RL/numeqtmp0.png)

> where ![rt_all](file:./RL/inimgtmp91.png) are all the rewards received following all selections of action a prior to play 

当然，这只是评估行为的一种方法，也不是最好的。不过目前可以用它来提出行为选择的问题。最简单的行为选择规则就是用贪心法选择![a_select](file:./RL/inimgtmp99.png),即每次选择评估最大值最大的行为![a_select_formul](file:./RL/inimgtmp100.png)。

每次试错时（trial-and-error）选择评估最好的action可以大大降低收索空间，但这种贪心算法效果不一定很好。提出一种![e2](file:./RL/img2.png)-greedy算法，以小概率![e2](file:./RL/img2.png)去探索新的行为（exploration）。

进一步改进，可以让每个action的选中概率正比于评估，即Softmax Action Selection。

不同的策略对应不同的动作值函数，不是每个都需要记录，只需要记录最好的那个就行。
>  It is easy to devise incremental update formulas for computing averages with small, constant computation required to process each new reward.

增量实现（Incremental Implementation）

动作值函数可以增量实现，设它的前k个奖励是![qk](file:./RL/inimgtmp151.png)(注意不要和![qt](file:./RL/inimgtmp86.png)搞混，t是第t步，k使用action a接下来的前k个反馈奖励)，可以得到：

![qk_formal](file:./RL/qk_formul.png)

形式如下：

![qk_format](file:./RL/numeqtmp3.png)  
其中![qk_format2](file:./RL/inimgtmp164.png)可以看成是误差，![qk_step](file:./RL/inimgtmp166.png)可以看成学习率。

跟踪一个不确定的环境模型（Tracking a Nonstationary Problem）：

确定的环境模型是指环境的状态空间很小（比如井字棋就那么多种排列情况）。而我们生活中多数问题的状态空间很大，甚至环境还会改变，是不确定的环境模型（比如围棋，所有排列情况太多，不是计算机能够处理的）  
上面提到的平均法适合确定的环境模型，但在不确定的环境模型中就不准确了，因为越往后的动作估值越不准确。可以用如下公式替代平均法：  

![qk_format2](file:./RL/qk_formul2.png) 

> The averaging methods discussed so far are appropriate in a stationary environment, but not if the bandit is changing over time. As noted earlier, we often encounter reinforcement learning problems that are effectively nonstationary. In such cases it makes sense to weight recent rewards more heavily than long-past ones. One of the most popular ways of doing this is to use a constant step-size parameter.

初始化动作值函数：

为了鼓励exploration，可以把某个state下直接奖励反馈大的行为对应的动作值函数初始化一个小的数值，给直接反馈小的行为对应的动作值函数初始化一个大的数值，这样在试错的时候一开始就会做出更多的尝试。

强化比较（Reinforcement Comparison）：

奖励期望越大的行为试错时能得到更多的复现机会，但我们要怎么去对比奖励期望？
>A central intuition underlying reinforcement learning is that actions followed by large rewards should be made more likely to recur, whereas actions followed by small rewards should be made less likely to recur. But how is the learner to know what constitutes a large or a small reward? 

如果像之前那样用动作值函数来对比不太好，因为它是根据策略来定义的，一开始尝试的策略比较少，所以动作值函数得到的值不是很准确。  
先定义第t次试错选择行为a的概率如下：  
![percent_a](file:./RL/numeqtmp6.png)  
其中![pt](file:./RL/inimgtmp212.png)表示对行为a的偏好(preference)。“强化比较”在这里就可以用来更新这个偏好的取值。假设某个试错策略下的到的奖励数是5，那么我们应该增加![pt](file:./RL/inimgtmp212.png)还是减小？这就需要一个参考的奖励值（reference reward）作为对比，比参考值大的就增加![pt](file:./RL/inimgtmp212.png)，小的就减小。用符号![reference_reward](file:./RL/inimgtmp218.png)表示参考的奖励值，![real_reward](file:./RL/inimgtmp217.png)表示当前奖励。得到偏好的更新公式：  
![pt_formul](file:./RL/numeqtmp7.png)  
以及参考奖励的更新公式：  
![reference_formul](file:./RL/numeqtmp8.png)  
其中![alpha_range](file:./RL/inimgtmp221.png)。

追赶法（Pursuit Methods）：

和强化比较一样，追赶法是另外一种学习的方法。不同的是，在学习（更新偏好）的过程中，它不止考虑了当前的偏好，还考虑了当前的动作值函数。试想一下，如果能在学习初期各个action被选中的概率比较平均，越到后面越按照动作值函数的指导来设置概率，概率值有一定的延迟但不断追赶着动作值函数，这就是追赶法。  
定义当前动作值函数最大的事件是：![argmax](file:./RL/inimgtmp237.png)
试错时某个事件被选中概率如下：  
![argmax_1](file:./RL/numeqtmp9.png)  
![argmax_2](file:./RL/numeqtmp10.png)  
这样就可以做到越往后，动作值函数平均越大的事件被选中的概率越大，但一开始概率每个事件被选中的概率差不多。

Evaluation VS Instruction：

如果某个state下有100个可选择的action，经过督导学习训练后得到最大奖励反馈的action是67（Instruction feedback），但是用动作值函数得到的最大奖励反馈的action是32（Evaluative Feedback），是选择这两个行为中的哪一个？

其实可以试错时随机选择，然后试错结束得到反馈后就知道选67好还是32好，给好的那个加分。最后统计谁的得分高就更加趋向于选谁。

###3.强化学习问题（The Reinforcement Learning Problem）

目标和奖励（Goals and Rewards）：

人类在做某件事的时候是有目标的，但是强化学习并没有引入目标这个概念，这是因为目标和奖励信号可以挂钩，只要给达成目标的上一个state到目标state的行为一个非常大的奖励信号，agent就会有目标（得到最大奖励就是它的目标）

回报期望（Returns）：

上文已经提到，确定的环境模型下回报期望是： 
![return1](file:./RL/numeqtmp11.png)  
其中，T是最终的步骤。agent在和环境交互的过程中，（因为步骤有限）很自然的会被分割成很多子序列（从某个state开始，然后结束，这就是一个子序列）。把每个子序列叫做一个episodic，把这种形式的交互叫做episodic tasks。

不确定的环境模型下回报期望是：  
![return2](file:./RL/numeqnarraytmp2-0-0.png)  
其中，![r_Range](file:./RL/inimgtmp296.png)。可见![lamda](file:./RL/inimgtmp295.png)越大，agent看得越远（但远的不一定准确），当等于1时退化成第一个公式。  
可见，这样的交互下T趋向于无穷大，把这种交互称为continuing tasks。

统一“episodic tasks”和“continuing tasks”：

为了方便计算，需要把episodic tasks和continuing tasks这两种交互方式统一成一种。可以这样，让episodic tasks的每个最终状态都加上一个自己到自己的行为连线，奖励是0，这样有限的episodic tasks就可以变成无限的continuing tasks。

![episodic_demo](file:./RL/imgtmp1.png)

马尔科夫性（The Markov Property）：

马尔科夫性说的是某个状态转到新状态的概率与之前所有经历过的状态没有关系，仅于当前状态有关。即： 
![mp1](file:./RL/numeqtmp13.png)  
等于：  
![mp2](file:./RL/numeqtmp14.png)

马尔科夫决策过程（Markov Decision Processes）：

由MDP的假设，如果这个世界就是MDP的，如果再加上一个假设，每个动作都是由完全的环境（比如人的每个细胞导致的精神状态，意识）决定，那么有一个初始状态，后继状态就是全部确定的。当然，现实情况环境一般不完全可观察，然后有一些随机性stochastic是人类无法确定的。绝大多数的增强学习都可以模型化为MDP的问题。  

如果行为和状态空间是有限的，就叫做有限马尔科夫过程（finite MDP）。

> If the state and action spaces are finite, then it is called a finite Markov decision process (finite MDP). 

设状态![s](file:./RL/inimgtmp357.png)经过行为![a](file:./RL/inimgtmp361.png)转换到![s,](file:./RL/inimgtmp362.png)的概率：  

![pssa](file:./RL/numeqtmp15.png)

得到的奖励期望是：

![rssa](file:./RL/numeqtmp16.png)

值函数（Value Functions）：

值函数用来评估一个状态的好坏。设一个策略是![pi](file:./RL/inimgtmp411.png)，它是状态![state](file:./RL/inimgtmp412.png)到行为![action](file:./RL/inimgtmp413.png)的映射。某状态下采取某行为的概率是![psa](file:./RL/inimgtmp414.png)，当前策略下的值函数![vpi](file:./RL/inimgtmp419.png)可以表示为：  
![vpi_formul](file:./RL/numeqtmp17.png)

类似的，动作值函数可以表示为：  
![qsa_formul](file:./RL/numeqtmp18.png)  
表示是当前策略下（某个状态的）某个行为到评估值的映射。

可以把这个方程写成“动态规划方程”的形式（Bellman equation）：

![v_bellman](file:./RL/v_bellman.png)

值函数优化：

强化学习就是找到一个策略让（长远来说）得到的奖励最大化。前文已经说了某个策略下的值函数怎么求，那么对于所有策略![pi_all](file:./RL/inimgtmp512.png)呢？

对于估值函数：  
![v_all_formul](file:./RL/numeqtmp19.png)

对于动作值函数：  
![q_all_formul](file:./RL/numeqtmp20.png)

结合起来：  
![q_all_formul2](file:./RL/numeqtmp21.png)  
其中![s_](file:./RL/inimgtmp362.png)表示s经过a有可能转移到的所有新状态（当然一般可能只有一个，所以不要被累加符号吓到）。

因为对所有策略的估值函数等于产生奖励最大的那个行为的动作值函数，所以可以写成：  
![v_formul2](file:./RL/v_formul2.png)

写成贝尔曼方程(动态规划方程)：  
![q_bellman](file:./RL/numeqtmp21.png)

##二、基本的实现算法

###1.动态规划法(dynamic programming methods)

####策略评估（Policy Evaluation）：
首先，先算出在某个策略![pi](file:./RL/inimgtmp411.png)下的所有值函数。按照之前给出的值函数bellman方程：  
![vbellman](file:./RL/vbellman.png)  
有如下算法：  
![pseudotmp0](file:./RL/pseudotmp0.png)

####策略改进（Policy Improvement）：  
计算值函数的一个原因是为了找到更好的策略。在状态s下，是否存在一个更好的行为![bettle_a](file:./RL/inimgtmp667.png)？要想判断行为的好坏，我们就需要计算动作值函数。  
![qpi](file:./RL/qpi.png)
已知估值函数V是评估某个状态的回报期望，动作值函数是评估某个状态下使用行为a的回报期望。所以，如果在新的策略下![new_pi](file:./RL/inimgtmp689.png)有![Q>V](file:./RL/inimgtmp691.png)就说明新的策略比当前的好。这里的不等式等价于:  
![v>v](file:./RL/numeqtmp23.png)  
有了策略改进定理，我们可以遍历所有状态和所有可能的动作，并采用贪心策略来获得新策略：  
![best_pi](file:./RL/bestpi.png)  
我们不止要更新策略，还需要更新估值函数：  
![v_new](file:./RL/v_new.png)

####策略迭代（Policy Iteration）：  
当有了一个策略![pi](file:./RL/inimgtmp411.png)，通过策略评估得到![v_pi](file:./RL/inimgtmp729.png)，再通过策略改进得到新的策略![pi_2](file:./RL/inimgtmp730.png)并可以计算出新的估值函数![v_2](file:./RL/inimgtmp731.png)，再次通过策略改进得到![pi_3](file:./RL/inimgtmp732.png)...

![pi_2_v](file:./RL/imgtmp35.png)

这样通过不断的迭代，就会收敛到一个很好的策略。

![pseudotmp1](file:./RL/pseudotmp1.png)

####值迭代（Value Iteration）：

策略迭代需要遍历所有的状态若干次，其中巨大的计算量直接影响了策略迭代算法的效率。我们必须要获得精确的![v_pi](file:./RL/inimgtmp729.png)值吗？事实上不必，有几种方法可以在保证算法收敛的情况下，缩短策略估计的过程。  
值函数的bellman方程如下：  
![v_k1](file:./RL/v_k1.png)  
值迭代算法直接用下一步![s_](file:./RL/inimgtmp362.png)的估值函数来更新当前s的估值函数，不断迭代，最后通过估值函数来得到最优策略。算法如下：  
![pseudotmp2](file:./RL/pseudotmp2.png)

####异步动态规划（Asynchronous Dynamic Programming）：  
对于值迭代，在s很多的情况下，我们可以多开一些线程，同时并行的更新多个s以提高效率；对于策略迭代，我们也没必要等待一个策略评估完成再开始下一个策略评估，可以多线程的更新。

####总结：
可以看到，动态规划去求解强化学习问题需要遍历所有状态，对于不确定的环境模型（比如围棋的排列组合方式太多，不可能遍历每个状态）并不适用。

###2.蒙特卡罗方法(Monte Carlo methods)

####基本思想：
蒙特卡洛的思想很简单，就是反复测试求平均。

一个简单的例子可以解释蒙特卡罗方法，假设我们需要计算一个不规则图形的面积，那么图形的不规则程度和分析性计算（比如积分）的复杂程度是成正比的。而采用蒙特卡罗方法是怎么计算的呢？首先你把图形放到一个已知面积的方框内，然后假想你有一些豆子，把豆子均匀地朝这个方框内撒，散好后数这个图形之中有多少颗豆子，再根据图形内外豆子的比例来计算面积。当你的豆子越小，撒的越多的时候，结果就越精确。

需要注意的是，我们仅仅将蒙特卡洛方法定义在episodic tasks上（就是指不管采取哪种策略都会在有限时间内到达终止状态并获得回报的任务）。

####蒙特卡洛策略评估（Monte Carlo Policy Evaluation）：

首先考虑用蒙特卡洛方法学习估值函数，方法很简单，就是先初始化一个策略，在这个策略下随机产生很多行为序列，针对每个行为序列计算出对应估值函数的取值，这样某个状态下估值函数针对不同的行为序列就得到不同的估值，只要再一平均就可以算出最后的估值。  
![pseudotmp3](file:./RL/pseudotmp3.png)

####蒙特卡洛动作函数评估（Monte Carlo Estimation of Action Values）：

为了像动态规划那样做策略改进，必须得到动作函数。这和前文类似，某个状态下动作是a，然后再随机尝试很多行为序列，得到反馈奖励后计算平均值（不是最大值，可能是考虑在不确定的环境模型下，使用平均最优更加靠谱）就是当前策略的动作函数。

####蒙特卡洛控制（Monte Carlo Control）

即蒙特卡洛版本的策略迭代：生成动作函数，改进策略，循环前两步。  
![imgtmp6](file:./RL/imgtmp6.png)  
过程如下：  
![imgtmp36](file:./RL/imgtmp36.png)  
具体到MC control，就是在每个episode后都重新估计下动作值函数（尽管不是真实值），然后根据近似的动作值函数，进行策略更新。这是一个episode by episode的过程。  
![pseudotmp4](file:./RL/pseudotmp4.png)

On-Policy Monte Carlo Control：


为了更好的探测状态空间避免陷入局部最优解，加入![e2](file:./RL/img2.png)-greedy算法：  
![20160512105522475](file:./RL/20160512105522475.png)  
训练越长时间![e2](file:./RL/img2.png)越小，可以提高收敛速度。

Off-Policy Monte Carlo Control：

为了让更加接近结果的步骤得到更多的调整机会，而不是一个episode中每一步都有一样多的调整机会，于是提出了Off-Policy方法。这就类似人类下棋，一开始变数太大没有必要花太多精力去学习（调整动作值函数），但越接近结果时越需要更准确的行为估值（需要花费更多的运算去调整动作值函数以便更加准确）。  
在off-policy中，使用两个策略。一个策略用来生成行为，叫做行为策略（behavior policy）；另外一个策略用来（从终点向起点）评估行为序列准确性的，叫做评估策略（estimation policy）。当第一次发现这两个策略指示的行为出现不同时，开始更新动作值函数，可见越靠近终点的得到调整的概率越大。算法如下：  
![pseudotmp6](file:./RL/pseudotmp6.png)  
其中，行为策略![inimgtmp908](file:./RL/inimgtmp908.png)是一种soft policy，评估策略![inimgtmp909](file:./RL/inimgtmp909.png)是一种greedy policy。

增量实现（Incremental Implementation）：

在第一章第二节已经提出过动作值函数的增量实现，这里就可以用上了。以值函数为例的增量实现如下：  
![numeqtmp27](file:./RL/numeqtmp27.png)  
其中：  
![imgtmp39](file:./RL/imgtmp39.png)，![inimgtmp925](file:./RL/inimgtmp925.png)


###3.时间差分法(temporal difference)

####基本思想：

动态规划优点：  
动态规划使用估值函数，已知动态规划在值迭代的时候可以直接用下一个状态的值更新当前状态的值；蒙特卡洛方法使用动作值函数，它只有在等待随机生成的episode所有行为执行完后，才能从后向前更新动作值函数。

蒙特卡洛优点：  
动态规划在值迭代的时候需要遍历所有state，这就需要一个确定的环境模型（state不能太多），但现实往往是state的数量特别多；蒙特卡洛方法就没有这个限制。

时序差分法可以看成是动态规划和蒙特卡洛方法的结合。和蒙特卡洛方法一样，它不需要环境模型V，能够直接从与环境的交互中学习，因此它也只能用动作值函数Q来指导行为；而它又与动态规划相似，可以基于对其他状态的估计来更新对当前状态估值函数的估计，不用等待最后的结果。  
>If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-difference (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap). The relationship between TD, DP, and Monte Carlo methods is a recurring theme in the theory of reinforcement learning. 



##自己的想法
有很多强化学习和督导学习可以结合的地方，比如如何提取有用的输入信号（注意力放在哪），如何将第一次接触的信号不训练直接分类后放到合适的模型中去训练（分区保存数据，比如第一次看到两个轮子的东西知道它是车），如何提取数据间的共性并仅记忆共性（特征提取）