function classifier = ELMtrain( data,target,num,type,labelnum)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%�ú�������Ҫ������ѵ��һ������ѧϰ��������
maxnum=labelnum;%��ȡ����е����ֵ
target1=zeros(size(data,1),maxnum);%��ʱ����targetת������
C=3;
for s=1:size(target,1)%������ת����0-1����
    target1(s,target(s,1))=1;
end
target=target1;
myclassifier=struct('inputweight',[],'hiddenoutput',[],'b',0,'outputweight',zeros(size(data,1),num),'type',[]);%bΪ������ڵ��ƫ��ֵ
myclassifier.inputweight=normrnd(0,1,num,size(data,2));%������������Ȩֵ���ӱ�׼��̬�ֲ�
myclassifier.b=normrnd(0,1,1,num);%�������������ƫ��
myclassifier.hiddenoutput=zeros(size(data,1),num);
for i=1:size(data,1)%�����������������
    for j=1:num
        if strcmp(type,'sig')==1%�����Ϊsigmoid����
          ds=1/(1+exp(myclassifier.b(1,j)-myclassifier.inputweight(j,:)*data(i,:)'));
           myclassifier.hiddenoutput(i,j)=ds;
        elseif strcmp(type,'sin')==1%�����Ϊ���Һ���
           ds=sin(myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(j));
           myclassifier.hiddenoutput(i,j)=ds;
        elseif strcmp(type,'hardlim')==1%�����ΪӲ���ƺ���
            if (myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(j))>=0 %a*x+b>=0
               myclassifier.hiddenoutput(i,j)=1;
            else %a*x+b<0
               myclassifier.hiddenoutput(i,j)=0;
            end
        elseif strcmp(type,'tribas')==1%�����Ϊ���ǻ�����
            if 1<=(myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(1,j))<=1 %-1=<a*x+1<=1
               ds=1-abs(myclassifier.inputweight(j,:)*data(i,:)'+myclassifier.b(1,j));
               myclassifier.hiddenoutput(i,j)=ds;
            else
               myclassifier.hiddenoutput(i,j)=0;
            end
        elseif strcmp(type,'radbas')==1%�����Ϊ��˹���������
           ds=exp((-1)*(sum((data(i,:)-myclassifier.inputweight(j,:)).^2)/(myclassifier.b(1,j))));
           myclassifier.hiddenoutput(i,j)=ds;
        elseif strcmp(type,'multi')==1%�����Ϊ����ʽ����
           ds=sqrt(myclassifier.b(1,j)^2+sum((data(i,:)-myclassifier.inputweight(j,:)).^2));
           myclassifier.hiddenoutput(i,j)=ds;
        end
    end
end
de=inv((1/C)*eye(num)+myclassifier.hiddenoutput'*myclassifier.hiddenoutput);%��(I/C+H'H)
ds=de*myclassifier.hiddenoutput'*target;%��������α�棬���������Ȩֵ�����Moore-Propose��
myclassifier.outputweight=ds;
myclassifier.type=type;
classifier=myclassifier;
end

