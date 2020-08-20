Dataset类用于处理表格类数据集。
参数解释：
df:读取的数据集
feature_cols:用于训练的特征
target_cols:需要预测的目标特征
test_size:训练集和测试集的比例
stratified_features:在训练集和测试集中必须分布均匀的特征(即针对某特征进行分层抽样)
scaler:可选参数为 None,"minmax","standard"，用于对连续型特征进行缩放
imputer:仅处理连续型特征。可选参数为 None,"knn","drop","mean","most_frequent","median","constant"
当选择了"constant"时，必须填写fill_value；
当选择了"knn"时，必须填写n_neighbors；
cat_features为离散性特征；否则,离散性特征也会被视为连续型特征进行缩放等操作;
cat_imputer,cat_fill_value,cat_n_neighbors参考imputer,不过仅处理离散性特征。
