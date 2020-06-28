function classifier = findnum(train,target,e,labelnum)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%
signal=1;%标记算法是否终止，当signal=1执行，signal=0终止 
num=min([size(train,2),size(train,1)]);%获取神经元数目的上限
v=randi([1,5],1,1);%随机选取激活函数
if v==1 %选取sigmoid函数
    type='sig';
elseif v==2 %选取sin函数
    type='sin';
elseif v==3 %选取硬限制函数
    type='hardlim';
elseif v==4 %选取三角偏倚函数
    type='tribas';
elseif v==5 %选取径向基函数
    type='radbas';
elseif v==6 %选取多项式函数
    type='multi';
end
left=1;%初始化神经元数目取值区间的起始值
right=num;%初始化神经元数目取值区间的起始值
while signal==1
    middle=floor((left+right)/2);%选取区间初始中间点
    temp1=ELMtrain(train,target,middle,type,labelnum);%先训练一个中间位置的极限学习机分类器
    [accuracy1,waste]=classification(temp1,train,target);%计算中间位置(数目)的极限学习分类器的准确率
    temp2=ELMtrain(train,target,floor((middle-1+left)/2),type,labelnum);%再训练一个左侧位置的极限学习机分类器
    [accuracy2,waste]=classification(temp2,train,target);%计算左侧位置(数目)的极限学习分类器的准确率
    if accuracy2-accuracy1>e %说明最优的分类器在左边区域
        right=middle-1;%更新神经元数目取值的上限
    else
        temp3=ELMtrain(train,target,floor((middle+1+right)/2),type,labelnum);%再训练一个右侧位置的极限学习机分类器
        [accuracy3,waste]=classification(temp3,train,target);%计算右侧位置(数目)的极限学习分类器的准确率
        if accuracy3-accuracy2>e %说明最优的分类器在右边区域
            left=middle+1;%更新神经元数目的下限
        else
            signal=0;%中间位置的分类器即为最优的分类器，终止算法
        end
    end    
end
classifier=temp1;%获得最终的分类器
end

