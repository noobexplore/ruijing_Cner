### Cner_v2

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

2020-07-04  fix一些bug，遗留一些问题，就是[[4],[1],[0]]这样一个的这种情况存在。

2020-07-07 去除掉了以上的包含最后的换行符的情况，现在开始着手进行模型的构建，所以添加model.py文件

2020-07-11 加入了train.py文件用于训练模型，然后完善了model的构建以及加入了一些方法