function classifier = findnum(train,target,e,labelnum)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%
signal=1;%����㷨�Ƿ���ֹ����signal=1ִ�У�signal=0��ֹ 
num=min([size(train,2),size(train,1)]);%��ȡ��Ԫ��Ŀ������
v=randi([1,5],1,1);%���ѡȡ�����
if v==1 %ѡȡsigmoid����
    type='sig';
elseif v==2 %ѡȡsin����
    type='sin';
elseif v==3 %ѡȡӲ���ƺ���
    type='hardlim';
elseif v==4 %ѡȡ����ƫ�к���
    type='tribas';
elseif v==5 %ѡȡ���������
    type='radbas';
elseif v==6 %ѡȡ����ʽ����
    type='multi';
end
left=1;%��ʼ����Ԫ��Ŀȡֵ�������ʼֵ
right=num;%��ʼ����Ԫ��Ŀȡֵ�������ʼֵ
while signal==1
    middle=floor((left+right)/2);%ѡȡ�����ʼ�м��
    temp1=ELMtrain(train,target,middle,type,labelnum);%��ѵ��һ���м�λ�õļ���ѧϰ��������
    [accuracy1,waste]=classification(temp1,train,target);%�����м�λ��(��Ŀ)�ļ���ѧϰ��������׼ȷ��
    temp2=ELMtrain(train,target,floor((middle-1+left)/2),type,labelnum);%��ѵ��һ�����λ�õļ���ѧϰ��������
    [accuracy2,waste]=classification(temp2,train,target);%�������λ��(��Ŀ)�ļ���ѧϰ��������׼ȷ��
    if accuracy2-accuracy1>e %˵�����ŵķ��������������
        right=middle-1;%������Ԫ��Ŀȡֵ������
    else
        temp3=ELMtrain(train,target,floor((middle+1+right)/2),type,labelnum);%��ѵ��һ���Ҳ�λ�õļ���ѧϰ��������
        [accuracy3,waste]=classification(temp3,train,target);%�����Ҳ�λ��(��Ŀ)�ļ���ѧϰ��������׼ȷ��
        if accuracy3-accuracy2>e %˵�����ŵķ��������ұ�����
            left=middle+1;%������Ԫ��Ŀ������
        else
            signal=0;%�м�λ�õķ�������Ϊ���ŵķ���������ֹ�㷨
        end
    end    
end
classifier=temp1;%������յķ�����
end

