function [accuracy,label] = classification(classifier,data,target)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%该函数的功能主要是利用已经训练好的极限学习机来对新数据进行分类
H=zeros(size(data,1),size(classifier.hiddenoutput,2));%初始化隐含层输出矩阵
for i=1:size(data,1)%计算隐含层输出矩阵
    for j=1:size(classifier.hiddenoutput,2)
        if strcmp(classifier.type,'sig')==1%激活函数为sigmoid函数
            ds=1/(1+exp(-(classifier.b(j)+classifier.inputweight(j,:)*data(i,:)')));
            H(i,j)=ds;
        elseif strcmp(classifier.type,'sin')==1%激活函数为正弦函数
            ds=sin(classifier.b(j)+classifier.inputweight(j,:)*data(i,:)');
            H(i,j)=ds;
        elseif strcmp(classifier.type,'hardlim')==1%激活函数为硬限制函数
            if classifier.b(j)+classifier.inputweight(j,:)*data(i,:)'>=0 %a*x+b>=0
               H(i,j)=1;
            else
               H(i,j)=0; 
            end
        elseif strcmp(classifier.type,'tribas')==1%激活函数为三角基函数
            if -1<=classifier.b(j)+classifier.inputweight(j,:)*data(i,:)'<=1 %-1=<a*x+1<=1
                ds=1-abs(classifier.b(j)+classifier.inputweight(j,:)*data(i,:)');
                H(i,j)=ds;
            else
                H(i,j)=0;
            end
        elseif strcmp(classifier.type,'radbas')==1%激活函数为高斯径向基函数
            ds=exp(-(sum((data(i,:)-classifier.inputweight(j,:)).^2))/classifier.b(j));
            H(i,j)=ds;
        elseif strcmp(classifier.type,'multi')==1%激活函数为多项式函数
            ds=sqrt(classifier.b(j)^2+sum((data(i,:)-classifier.inputweight(j,:)).^2));
            H(i,j)=ds;
        end
    end
end
T=H*classifier.outputweight;%计算输出的0-1矩阵
[c,thistarget]=max(T,[],2);%最大投票原则，thistarget记录分类结果
count=0;%记录正确分类的事例数目
for s=1:size(thistarget,1)%统计分类结果
    if thistarget(s,1)==target(s,1)
        count=count+1;
    end
end
accuracy=count/(size(thistarget,1));%计算准确率
label=thistarget;
end

