%% 1. 加载数据集路径
pathname = 'C:\Users\an\Desktop\train1\';   %数据集路径
%% 2. 读取原始输入特征和目标输出
[X, NRCS_1,NRCS1_mean,NRCS1_mean_matix, filelist] = Pre_input(pathname);

x = X';
t = NRCS_1';
NRCS1_mean_matix = NRCS1_mean_matix';

%% 3. 设置神经网络结构（隐藏层 [30, 30, 30, 30]，训练函数 trainlm）
trainFcn = 'trainlm';
hiddenLayerSize = [30,30,30,30];
net = fitnet(hiddenLayerSize,trainFcn);

%% 4. 将数据集按 block 方式切分，随机选择训练/验证/测试数据
% 假设 Q 是 131*43 的倍数
Q = size(x, 2);
% 每个 131*43 的子块的大小
block_size = 131 * 43;
% 计算每个子块中需要抽取的点数
train_size = round(block_size * 0.15); % 20%
val_size = round(block_size * 0.1); % 15%
test_size = round(block_size * 0.1); % 15%
% 初始化 x_train, x_val, x_test
xTrain = [];tTrain = [];
xVal = [];tVal = [];
xTest = [];tTest = [];
nrcs_mean_test = [];
% 初始化索引集合
trainIndices = [];
valIndices = [];
testIndices = [];

% 遍历每个 131*43 的子块
for i = 1:block_size:Q
    % 当前子块的索引范围
    block_indices = i:i+block_size-1;
    
    % 随机打乱当前子块的索引
    shuffled_indices = block_indices(randperm(block_size));
    
    % 分配到训练集、验证集和测试集
    train_indices = shuffled_indices(1:train_size);
    val_indices = shuffled_indices(train_size+1:train_size+val_size);
    test_indices = shuffled_indices(train_size+val_size+1:train_size+val_size+test_size);
    
    % 提取对应的点
    xTrain = [xTrain, x(:, train_indices)];
    tTrain = [tTrain, t(:, train_indices)];
    xVal = [xVal, x(:, val_indices)];
    tVal = [tVal, t(:, val_indices)];
    xTest = [xTest, x(:, test_indices)];
    tTest = [tTest, t(:, test_indices)];
    nrcs_mean_test =[ nrcs_mean_test,NRCS1_mean_matix(:, test_indices)];

     % 保存全局索引
    trainIndices = [trainIndices, train_indices];
    valIndices = [valIndices, val_indices];
    testIndices = [testIndices, test_indices];
end

%% 5. 手动设置训练索引到网络对象
% Manually set the training, validation, and test sets
net.divideFcn = 'divideind'; %手动设置
net.divideParam.trainInd = trainIndices;
net.divideParam.valInd = valIndices;
net.divideParam.testInd = testIndices;

%% 6. 设置最大训练轮次
net.trainParam.epochs = 800;  % 设置训练的最大轮次为 1000

%% 7. 开始训练
[net,tr] = train(net,xTrain,tTrain);
tr.trainInd = trainIndices;
tr.valInd = valIndices;
tr.testInd = testIndices;

%% 8. 使用测试集评估网络表现，计算误差指标
yTest = net(xTest);
e = gsubtract(tTest,yTest);%计算误差
performance = perform(net,tTest,yTest)%返回的默认为MSE

% View the Network
view(net);

%% 9. 保存模型和预测结果

