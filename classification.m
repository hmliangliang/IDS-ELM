function [accuracy,label] = classification(classifier,data,target)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%�ú����Ĺ�����Ҫ�������Ѿ�ѵ���õļ���ѧϰ�����������ݽ��з���
H=zeros(size(data,1),size(classifier.hiddenoutput,2));%��ʼ���������������
for i=1:size(data,1)%�����������������
    for j=1:size(classifier.hiddenoutput,2)
        if strcmp(classifier.type,'sig')==1%�����Ϊsigmoid����
            ds=1/(1+exp(-(classifier.b(j)+classifier.inputweight(j,:)*data(i,:)')));
            H(i,j)=ds;
        elseif strcmp(classifier.type,'sin')==1%�����Ϊ���Һ���
            ds=sin(classifier.b(j)+classifier.inputweight(j,:)*data(i,:)');
            H(i,j)=ds;
        elseif strcmp(classifier.type,'hardlim')==1%�����ΪӲ���ƺ���
            if classifier.b(j)+classifier.inputweight(j,:)*data(i,:)'>=0 %a*x+b>=0
               H(i,j)=1;
            else
               H(i,j)=0; 
            end
        elseif strcmp(classifier.type,'tribas')==1%�����Ϊ���ǻ�����
            if -1<=classifier.b(j)+classifier.inputweight(j,:)*data(i,:)'<=1 %-1=<a*x+1<=1
                ds=1-abs(classifier.b(j)+classifier.inputweight(j,:)*data(i,:)');
                H(i,j)=ds;
            else
                H(i,j)=0;
            end
        elseif strcmp(classifier.type,'radbas')==1%�����Ϊ��˹���������
            ds=exp(-(sum((data(i,:)-classifier.inputweight(j,:)).^2))/classifier.b(j));
            H(i,j)=ds;
        elseif strcmp(classifier.type,'multi')==1%�����Ϊ����ʽ����
            ds=sqrt(classifier.b(j)^2+sum((data(i,:)-classifier.inputweight(j,:)).^2));
            H(i,j)=ds;
        end
    end
end
T=H*classifier.outputweight;%���������0-1����
[c,thistarget]=max(T,[],2);%���ͶƱԭ��thistarget��¼������
count=0;%��¼��ȷ�����������Ŀ
for s=1:size(thistarget,1)%ͳ�Ʒ�����
    if thistarget(s,1)==target(s,1)
        count=count+1;
    end
end
accuracy=count/(size(thistarget,1));%����׼ȷ��
label=thistarget;
end

