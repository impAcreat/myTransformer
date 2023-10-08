  
##  Learning Chemical Rules of Retrosynthesis with Pre-training
  
*** 
##  Abstract:
  
* 本文主要关注无模板（template-free）方法的逆合成
  * 优势：较少受模板泛化问题和原子映射问题的影响
  * 缺点：生成的反应可能不符合化学规则
</br>
  
* 本文提出的**解决方式**:
  **预训练解决: 通过编码化学规则来增强预训练模型的能力**
  * 通过分子重构预训练任务来执行原子守恒规则
    (通过针对性的预训练任务，使得模型学习到化学规则)
  
</br>
  
* 效果：预训练解决方案大大提高了三个下游数据集的单步反合成精度
  
***
  
##  方法介绍
  
###  分析各种方法：
  
* Template-based & Semi-template-based:
  这2种方法严重依赖于template或者原子间映射，而这是化学中未解决的挑战
  
</br>
  
* Template-free:
  在之前的模型中，由于缺乏化学信息，无模板方法效果不佳，主要原因：
  1. 生成的分子无效
  2. 生成的反应物违反原子守恒定律
  3. 生成的反应物不会发生反应或无法产生目标产物
  
    即：由于无模板自由度太高，容易出现生成反应无效的情况
  
###  提出单步逆合成的预训练模型（PMSR）
  
####  重点技术：
  
* 自回归技术：
  >auto-aggression
  
* 对分子进行区域覆盖后的恢复任务：
  >a molecule recovery task with regional masks
  >1. 被掩盖的元素可通过分析周围的其他可见原子而恢复：有助于模型生成有效的分子
  >2. 被掩盖的元素也可通过给定的产物来预测，鼓励模型遵循原子守恒
  
* 监督对比任务
  >supervised contrastive task
  >将反应类型作为先验知识能大大提高模型性能，故在PMSR中提出了一个监督对比任务，以迫使模型更多地关注反应中心
  
####  从PMSR模型的角度看single-step retrosynthesis
  
* 用SMILES将生成物和反应物表示为字符串，进行预测以及训练
* 由于生成物和反应物共享词表，且原子守恒，故从该模型的角度来说：逆合成预测是条件生成（conditional generation）而非翻译
</br>
  
####  问题 & 解决：
  
  ![Alt text](pictures/image.png )   
 </br> 
  **因为化学规则是隐含在反应中的，故生成的反应物可能不符合规则**
  * 问题1：生成错误的分子部分
  >例如, 分子的基本化学规律所有的分子都遵循基本的价键理论，氟原子不应该与其他三个原子连接，如图1(a)中所示的无效分子
  
  解决方式：通过masked element recovery task训练模型，让模型恢复原子与键，使模型学会分子构成的基本规则
  
  * 问题2：破坏原子守恒定律（文中提到：对守恒定律的破坏一般出现在产物生成反应物前体的过程中，官能团的错位）
  
  >例如：反应物中除反应中心以外的部分在反应中发生了变化。例如，在图1(b)中，甲基错误地附着在羧基的邻位上，这与产物完全不同
  
  解决方式：通过masked fragment recovery task训练模型，该方法屏蔽前体（precursor）的连续元素，而模型对该屏蔽部分的原子进行正确的排列（文中提到：fragment recovery与element不同，fragment任务鼓励前体与产物保持一致）
  
* 试剂的生成
  
####  总结 PMSR
  
* 提出对单步逆合成预测面临问题的解决方法：
  >* masked element recovery：被掩盖单元的恢复
  >* masker fragment recovery：被掩盖片段的恢复
  >* reaction classification：反应分类
  （反应分类有助于模型找到反应中心，从而提高预测的准确率）
  
* 设计3个预训练任务
  >* 自回归
  >* 分子恢复
  >* 对比分类
  
* 模型特点：
  >* transformer-based
  >* 专注于 化学反应级 任务（reaction level task）
  >* 用化学靶向预训练任务来进行训练
  >* sequence2sequence 模型
  
  
  