### Cner_v2

```xml
     bert_ner
      |——bert_ner # 模型与数据文件
	  |   |——bert_base  # bert的训练代码
	  |   |——train  # 主要训练代码
	  |   |——bert_lstm_ner.py  # 训练的主函数代码
	  |   |——train_helper.py  # 参数预设代码
	  |   |——chinese_L-12_H-768_A-12 # bert-base的中文预训练模型
	  |   |——ltp_data_v3.4.0 # pyltp的词性分析、句法分析的模型
	  |   |——NERdata  # 切分后的数据集
	  |   |    |——dev.txt  # 验证集
	  |   |    |——test.txt  # 测试集
	  |   |    |——train.txt  # 训练集
	  |   |——output  # 模型输出
	  |——data_process # 处理训练数据格式
	  |   |——data_IB  # 原始数据集
	  |   |——ner_data  # 处理后的ner格式数据集，在“。"、"？"、"！"后面加入空行
	  |   |——split_data  # 将处理后的数据按7:1:2切分成训练集、验证集、测试集，放入NERdata目录中
	  |   IB_tag.py  # 第一步
	  |   ner_data.py  # 第二步
	  |   split_data.py  # 第三步
	  |——src # 运行调用的脚本
	  |   |——bert_ner_predict.py  # bert-ner预测脚本
	  bert_ner_api.py  # 控制bert-ner训练和测试的主函数脚本
	  README.md # 说明文件
	```
2020-07-02 First commit

**项目简介**：启动对项目Cner_v1的升级，具体如下：

- 预处理部分的升级
  - 文本切分处理这块，主要是对文本的切分位置和长短句做了处理。
- 数据输入的增强
  - 加入了一些额外的特征来对数据进行增强
    - 分词特征（上一个版本已经加入了）
    - 词边界和词性特征
    - 拼音和偏旁部首特征
- 模型的改进
  - 采用两层的BiLstm进行建模

2020-07-03  commit prepare_data.py

完善改文件，主要是为了对数据进行添加一些特征的操作，然后保存到dict.pkl文件去

2020-07-04 fix一些bug，遗留一些问题，就是[[4],[1],[0]]这样一个的这种情况存在。

2020-07-07 去除掉了以上的包含最后的换行符的情况，现在开始着手进行模型的构建，所以添加model.py文件

2020-07-11 加入了train.py文件用于训练模型，然后完善了model的构建以及加入了一些方法

2020-07-15 加入了test.py文件用于测试和评估模型，加入conlleval.py采用conll评估方式去评估，
后续准备集合train和test文件加入到main.py中。

2020-07-17 更新汇总：修改了一些bug以及添加了加载预训练模型的功能。
第一阶段的评估结果：F1 67左右效果不太理想。

分析，数据集的质量不是太好，有很多长尾分布，最多提升到74左右（参考测评任务第一名团队）。
