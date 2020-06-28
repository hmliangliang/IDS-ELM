function classifier = ELMtrain( data,target,num,type,labelnum)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%该函数的主要功能是训练一个极限学习机分类器
maxnum=labelnum;%获取标号中的最大值
target1=zeros(size(data,1),maxnum);%临时保存target转换矩阵
C=3;
for s=1:size(target,1)%将类标号转换成0-1矩阵
    target1(s,target(s,1))=1;
end
target=target1;
myclassifier=struct('inputweight',[],'hiddenoutput',[],'b',0,'outputweight',zeros(size(data,1),num),'type',[]);%b为隐含层节点的偏倚值
myclassifier.inputweight=normrnd(0,1,num,size(data,2));%随机生成输入层权值服从标准正态分布
myclassifier.b=normrnd(0,1,1,num);%随机生成隐含层偏倚
myclassifier.hiddenoutput=zeros(size(data,1),num);
for i=1:size(data,1)%计算隐含层输出矩阵
    for j=1:num
        if strcmp(type,'sig')==1%激活函数为sigmoid函数
          ds=1/(1+exp(myclassifier.b(1,j)-myclassifier.inputweight(j,:)*data(i,:)'));
           myclassifier.hiddenoutput(i,j)=ds;
        elseif strcmp(type,'sin')==1%激活函数为正弦函数
           ds=sin(myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(j));
           myclassifier.hiddenoutput(i,j)=ds;
        elseif strcmp(type,'hardlim')==1%激活函数为硬限制函数
            if (myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(j))>=0 %a*x+b>=0
               myclassifier.hiddenoutput(i,j)=1;
            else %a*x+b<0
               myclassifier.hiddenoutput(i,j)=0;
            end
        elseif strcmp(type,'tribas')==1%激活函数为三角基函数
            if 1<=(myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(1,j))<=1 %-1=<a*x+1<=1
               ds=1-abs(myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(1,j));
               myclassifier.hiddenoutput(i,j)=ds;
            else
               myclassifier.hiddenoutput(i,j)=0;
            end
        elseif strcmp(type,'radbas')==1%激活函数为高斯径向基函数
           ds=exp((-1)*(sum((data(i,:)-myclassifier.inputweight(j,:)).^2)/(myclassifier.b(1,j))));
           myclassifier.hiddenoutput(i,j)=ds;
        elseif strcmp(type,'multi')==1%激活函数为多项式函数
           ds=sqrt(myclassifier.b(1,j)^2+sum((data(i,:)-myclassifier.inputweight(j,:)).^2));
           myclassifier.hiddenoutput(i,j)=ds;
        end
    end
end
de=inv((1/C)*eye(num)+myclassifier.hiddenoutput'*myclassifier.hiddenoutput);%求(I/C+H'H)
ds=de*myclassifier.hiddenoutput'*target;%计算矩阵的伪逆，求出输出输出权值矩阵的Moore-Propose逆
myclassifier.outputweight=ds;
myclassifier.type=type;
classifier=myclassifier;
end

