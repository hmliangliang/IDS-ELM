data=student;%进入系统中的数据流
%需要在ELMtrain函数中指定C值
C=3;
col=size(data,2);%数据的维数
train=data(:,1:(col-1));%获取数据
train=zscore(train);%标准化
target=data(:,col);%获取数据流的类标签
labelnum=max(target);%类标签的最大值
winsize=250;%代表数据块的大小
k=5;%k代表分类器数目的最大值
alpha=0.05;%显著性水平
ssignal=1;%用来标记第一次分类器数目到达k后前一次分类准确率的初始化
n=10*winsize;%n为Hoeffding界中的参数取值
u=0.0001;%非常小的常数，防止除数为0
e=sqrt((log(1/alpha))/(2*n));%Hoeffding界的边界值
ensemble=struct('inputweight',[],'hiddenoutput',[],'b',0,'outputweight',[],'type',[]);%b为隐含层节点的偏倚值
weight=zeros(1,k);%分类器的权值矩阵
w=1;%记录当前权值的标号
K=struct('inverse',[]);%保存H'*H的逆
i=2*winsize;%i标记扫描数据序号
accuracy1=0;%之前上一个数据块的准确率
accuracy2=0;%当前数据块分类的准确率
cc=0;%记录测试次数
accuracy=[];%保存每一次测试结果
tic;
while i<=size(train,1)
    if mod(i,2*winsize)==0
        %获取训练集和测试集
        traindata=train((i-2*winsize+1):(i-winsize),:);
        traintarget=target((i-2*winsize+1):(i-winsize),:);
        testdata=train((i-winsize+1):i,:);
        testtarget=target((i-winsize+1):i,:);
        if size(ensemble,2)<k %分类器数目为1或者为0
            if (size(ensemble,2)==1)&&(isempty(ensemble(1).inputweight)==1)%分类器系统为空
               ds=findnum(traindata,traintarget,e,labelnum);
               ensemble(1)=ds;
               weight(1)=1;%权值初始化为1
               w=1;%标记位置向前移动
               ds=inv((1/C)*eye(size(ensemble(1).hiddenoutput,2))+ensemble(1).hiddenoutput'*ensemble(1).hiddenoutput);%计算H'H的逆
               K(1).inverse=ds;
            else%分类器系统不为空,加入新分类器
               ds=findnum(traindata,traintarget,e,labelnum);%先训练一个分类器,临时保存
               temp=ds;
               ensemble=[ensemble,temp];
               w=w+1;
               weight(w)=1;%权值初始化为1
               ds=inv((1/C)*eye(size(ensemble(w).hiddenoutput,2))+ensemble(w).hiddenoutput'*ensemble(w).hiddenoutput);%计算H'H的逆
               K(w).inverse=ds;
            end
        else  %分类器的数目到达k
           tresult=[];%保存所有分类器的分类结果
           for ti=1:size(ensemble,2)%分别对数据进行分类
               [acc,thistarget]=classification(ensemble(ti),traindata,traintarget);%acc为分类结果的准确率,thistarget为分类结果的矩阵
               weight(ti)=1/(1-acc+u);%根据分类结果更新权值
               tresult=[tresult,thistarget];%保存结果
           end
           tweight=zeros(size(traindata,1),labelnum);%保存集成式分类期投票的权值之和
           for si=1:size(tresult,1)%加权投票
               for sj=1:size(tresult,2)
                   tweight(si,tresult(si,sj))=tweight(si,tresult(si,sj))+weight(sj);
               end
           end
           [waste,tlast]=max(tweight,[],2);%根据权值的累积和进行决策
           tcount=0;%统计正确分类的事例个数
           for pi=1:size(tlast,1)
               if tlast(pi,1)==traintarget(pi,1)%分类正确
                   tcount=tcount+1;
               end
           end
           accuracy2=tcount/(size(traindata,1));
           if ssignal==1%说明是第一次分类器的数目达到k
              accuracy1=accuracy2;
              ssignal=0;%分类器数目不再是第一次到达k
           end
           
           if accuracy1-accuracy2>e%发生概念漂移，先删除一半数目的分类器
               disp('发生概念漂移!');
               midvalue=median(sort(weight));%寻找权值的中位数
               di=1;
               while di<=size(weight,2)%扫描分类器系统
                   if weight(di)<=midvalue%权值小于中位数
                      %删除分类器和相应的权值和分类器
                      weight(di)=[];%删除权值
                      ensemble(di)=[];%删除分类器
                      K(di)=[];%删除分类器保存的以上一个逆
                      w=w-1;
                   else
                       di=di+1;
                   end
               end
           end
           
           accuracy1=accuracy2;%把此时的分类准确率作为上一次准确率
           
           Tk=zeros(size(traindata,1),labelnum);%定义输出结果矩阵
           %把Tk转换成0-1矩阵
           for tk=1:size(traindata,1)
               Tk(tk,traintarget(tk,1))=1;
           end
           for xi=1:size(ensemble,2)%对于每一个分类器而言
               Hk=zeros(size(traindata,1),size(ensemble(xi).hiddenoutput,2));%初始化新数据的矩阵
               for ai=1:size(traindata,1)%先计算增量式隐含层输出矩阵
                  for aj=1:size(ensemble(xi).hiddenoutput,2)
                      if strcmp(ensemble(xi).type,'sig')==1%激活函数为sigmoid函数
                         ds=1/(1+exp(-(ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)'))); 
                         Hk(ai,aj)=ds;
                      elseif strcmp(ensemble(xi).type,'sin')==1%激活函数为sin函数
                          ds=sin(ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)');
                          Hk(ai,aj)=ds;
                      elseif strcmp(ensemble(xi).type,'hardlim')==1%激活函数为硬限制函数
                          if ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)'>=0 %a*x+b>=0
                             Hk(ai,aj)=1;
                          else
                             Hk(ai,aj)=0;
                          end
                      elseif strcmp(ensemble(xi).type,'tribas')==1%激活函数为三角基函数
                          if -1<=ensemble(xi).b+ensemble(xi).inputweight(aj,:)*traindata(ai,:)'<=1%-1<=a*x+b<=1
                             ds=1-abs(ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)');
                             Hk(ai,aj)=ds;
                          else
                             Hk(ai,aj)=0;
                          end
                      elseif strcmp(ensemble(xi).type,'radbas')==1%激活函数为高斯径向基函数
                           ds=exp((-norm(traindata(ai,:)-ensemble(xi).inputweight(aj,:))^2)/ensemble(xi).b(aj));
                           Hk(ai,aj)=ds;
                      elseif strcmp(ensemble(xi).type,'multi')==1%激活函数为多项式函数 
                           ds=sqrt(ensemble(xi).b(aj)^2+norm(traindata(ai,:)-ensemble(xi).inputweight(aj,:))^2);
                           Hk(ai,aj)=ds;
                      end
                 end%for aj
               end%for ai
               %增量式更新分类器的权值和隐含层矩阵
               ds=K(xi).inverse-K(xi).inverse*Hk'*inv(eye(size(traindata,1))+Hk*K(xi).inverse*Hk')*Hk*K(xi).inverse;%计算每个分类器的逆
               K(xi).inverse=ds;
               ds=ensemble(xi).outputweight+K(xi).inverse*Hk'*(Tk-Hk*ensemble(xi).outputweight);%更新分类器的权值
               ensemble(xi).outputweight=ds;
          end%for xi
        end %<k
        %对新数据进行测试
        cc=cc+1;%更新统计值
        tres=[];
        tresult=[];
        numcount=0;%记录所有极限学习机的隐含层神经元数目平均值
        for yi=1:size(ensemble,2)
            [waste,tres]=classification(ensemble(yi),testdata,testtarget);%进行分类获得结果，tres保存当前分类器的真实分类结果
            tresult=[tresult,tres];%tresult保存所有分类器对测试数据的分类结果
            numcount=numcount+size(ensemble(yi).hiddenoutput,2);
        end
        numcount=numcount/(size(ensemble,2));
        ttweight=zeros(size(testdata,1),labelnum);%保存投票的累积权值
        for vi=1:size(tresult,1)%统计投票情况
            for vj=1:size(tresult,2)
                ttweight(vi,tresult(vi,vj))=ttweight(vi,tresult(vi,vj))+weight(vj);%累计权值
            end
        end
        [waste,lastlabel]=max(ttweight,[],2);%最大权值投票原则
        ttcount=0;%统计测试数据中被正确分类的数据数目
        for ef=1:size(lastlabel,1)
            if lastlabel(ef,1)==testtarget(ef,1)%分类正确
                ttcount=ttcount+1;
            end
        end
        currentright=ttcount/(size(lastlabel,1));%当前数据测试的准确率
        accuracy=[accuracy,currentright];
        disp(['第',num2str(cc),'次测试结果的准确率为:',num2str(currentright),'   隐含层神经元的数目平均值为：',num2str(numcount)]);
    end%mod
    
    i=i+2*winsize;%处理下一个数据
end
disp(['此算法在当前数据集上测试结果的平均准确率为:',num2str(mean(accuracy(1,6:size(accuracy,2))))]);
toc;
