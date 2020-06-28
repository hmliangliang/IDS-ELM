data=student;%����ϵͳ�е�������
%��Ҫ��ELMtrain������ָ��Cֵ
C=3;
col=size(data,2);%���ݵ�ά��
train=data(:,1:(col-1));%��ȡ����
train=zscore(train);%��׼��
target=data(:,col);%��ȡ�����������ǩ
labelnum=max(target);%���ǩ�����ֵ
winsize=250;%�������ݿ�Ĵ�С
k=5;%k�����������Ŀ�����ֵ
alpha=0.05;%������ˮƽ
ssignal=1;%������ǵ�һ�η�������Ŀ����k��ǰһ�η���׼ȷ�ʵĳ�ʼ��
n=10*winsize;%nΪHoeffding���еĲ���ȡֵ
u=0.0001;%�ǳ�С�ĳ�������ֹ����Ϊ0
e=sqrt((log(1/alpha))/(2*n));%Hoeffding��ı߽�ֵ
ensemble=struct('inputweight',[],'hiddenoutput',[],'b',0,'outputweight',[],'type',[]);%bΪ������ڵ��ƫ��ֵ
weight=zeros(1,k);%��������Ȩֵ����
w=1;%��¼��ǰȨֵ�ı��
K=struct('inverse',[]);%����H'*H����
i=2*winsize;%i���ɨ���������
accuracy1=0;%֮ǰ��һ�����ݿ��׼ȷ��
accuracy2=0;%��ǰ���ݿ�����׼ȷ��
cc=0;%��¼���Դ���
accuracy=[];%����ÿһ�β��Խ��
tic;
while i<=size(train,1)
    if mod(i,2*winsize)==0
        %��ȡѵ�����Ͳ��Լ�
        traindata=train((i-2*winsize+1):(i-winsize),:);
        traintarget=target((i-2*winsize+1):(i-winsize),:);
        testdata=train((i-winsize+1):i,:);
        testtarget=target((i-winsize+1):i,:);
        if size(ensemble,2)<k %��������ĿΪ1����Ϊ0
            if (size(ensemble,2)==1)&&(isempty(ensemble(1).inputweight)==1)%������ϵͳΪ��
               ds=findnum(traindata,traintarget,e,labelnum);
               ensemble(1)=ds;
               weight(1)=1;%Ȩֵ��ʼ��Ϊ1
               w=1;%���λ����ǰ�ƶ�
               ds=inv((1/C)*eye(size(ensemble(1).hiddenoutput,2))+ensemble(1).hiddenoutput'*ensemble(1).hiddenoutput);%����H'H����
               K(1).inverse=ds;
            else%������ϵͳ��Ϊ��,�����·�����
               ds=findnum(traindata,traintarget,e,labelnum);%��ѵ��һ��������,��ʱ����
               temp=ds;
               ensemble=[ensemble,temp];
               w=w+1;
               weight(w)=1;%Ȩֵ��ʼ��Ϊ1
               ds=inv((1/C)*eye(size(ensemble(w).hiddenoutput,2))+ensemble(w).hiddenoutput'*ensemble(w).hiddenoutput);%����H'H����
               K(w).inverse=ds;
            end
        else  %����������Ŀ����k
           tresult=[];%�������з������ķ�����
           for ti=1:size(ensemble,2)%�ֱ�����ݽ��з���
               [acc,thistarget]=classification(ensemble(ti),traindata,traintarget);%accΪ��������׼ȷ��,thistargetΪ�������ľ���
               weight(ti)=1/(1-acc+u);%���ݷ���������Ȩֵ
               tresult=[tresult,thistarget];%������
           end
           tweight=zeros(size(traindata,1),labelnum);%���漯��ʽ������ͶƱ��Ȩֵ֮��
           for si=1:size(tresult,1)%��ȨͶƱ
               for sj=1:size(tresult,2)
                   tweight(si,tresult(si,sj))=tweight(si,tresult(si,sj))+weight(sj);
               end
           end
           [waste,tlast]=max(tweight,[],2);%����Ȩֵ���ۻ��ͽ��о���
           tcount=0;%ͳ����ȷ�������������
           for pi=1:size(tlast,1)
               if tlast(pi,1)==traintarget(pi,1)%������ȷ
                   tcount=tcount+1;
               end
           end
           accuracy2=tcount/(size(traindata,1));
           if ssignal==1%˵���ǵ�һ�η���������Ŀ�ﵽk
              accuracy1=accuracy2;
              ssignal=0;%��������Ŀ�����ǵ�һ�ε���k
           end
           
           if accuracy1-accuracy2>e%��������Ư�ƣ���ɾ��һ����Ŀ�ķ�����
               disp('��������Ư��!');
               midvalue=median(sort(weight));%Ѱ��Ȩֵ����λ��
               di=1;
               while di<=size(weight,2)%ɨ�������ϵͳ
                   if weight(di)<=midvalue%ȨֵС����λ��
                      %ɾ������������Ӧ��Ȩֵ�ͷ�����
                      weight(di)=[];%ɾ��Ȩֵ
                      ensemble(di)=[];%ɾ��������
                      K(di)=[];%ɾ�����������������һ����
                      w=w-1;
                   else
                       di=di+1;
                   end
               end
           end
           
           accuracy1=accuracy2;%�Ѵ�ʱ�ķ���׼ȷ����Ϊ��һ��׼ȷ��
           
           Tk=zeros(size(traindata,1),labelnum);%��������������
           %��Tkת����0-1����
           for tk=1:size(traindata,1)
               Tk(tk,traintarget(tk,1))=1;
           end
           for xi=1:size(ensemble,2)%����ÿһ������������
               Hk=zeros(size(traindata,1),size(ensemble(xi).hiddenoutput,2));%��ʼ�������ݵľ���
               for ai=1:size(traindata,1)%�ȼ�������ʽ�������������
                  for aj=1:size(ensemble(xi).hiddenoutput,2)
                      if strcmp(ensemble(xi).type,'sig')==1%�����Ϊsigmoid����
                         ds=1/(1+exp(-(ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)'))); 
                         Hk(ai,aj)=ds;
                      elseif strcmp(ensemble(xi).type,'sin')==1%�����Ϊsin����
                          ds=sin(ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)');
                          Hk(ai,aj)=ds;
                      elseif strcmp(ensemble(xi).type,'hardlim')==1%�����ΪӲ���ƺ���
                          if ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)'>=0 %a*x+b>=0
                             Hk(ai,aj)=1;
                          else
                             Hk(ai,aj)=0;
                          end
                      elseif strcmp(ensemble(xi).type,'tribas')==1%�����Ϊ���ǻ�����
                          if -1<=ensemble(xi).b+ensemble(xi).inputweight(aj,:)*traindata(ai,:)'<=1%-1<=a*x+b<=1
                             ds=1-abs(ensemble(xi).b(aj)+ensemble(xi).inputweight(aj,:)*traindata(ai,:)');
                             Hk(ai,aj)=ds;
                          else
                             Hk(ai,aj)=0;
                          end
                      elseif strcmp(ensemble(xi).type,'radbas')==1%�����Ϊ��˹���������
                           ds=exp((-norm(traindata(ai,:)-ensemble(xi).inputweight(aj,:))^2)/ensemble(xi).b(aj));
                           Hk(ai,aj)=ds;
                      elseif strcmp(ensemble(xi).type,'multi')==1%�����Ϊ����ʽ���� 
                           ds=sqrt(ensemble(xi).b(aj)^2+norm(traindata(ai,:)-ensemble(xi).inputweight(aj,:))^2);
                           Hk(ai,aj)=ds;
                      end
                 end%for aj
               end%for ai
               %����ʽ���·�������Ȩֵ�����������
               ds=K(xi).inverse-K(xi).inverse*Hk'*inv(eye(size(traindata,1))+Hk*K(xi).inverse*Hk')*Hk*K(xi).inverse;%����ÿ������������
               K(xi).inverse=ds;
               ds=ensemble(xi).outputweight+K(xi).inverse*Hk'*(Tk-Hk*ensemble(xi).outputweight);%���·�������Ȩֵ
               ensemble(xi).outputweight=ds;
          end%for xi
        end %<k
        %�������ݽ��в���
        cc=cc+1;%����ͳ��ֵ
        tres=[];
        tresult=[];
        numcount=0;%��¼���м���ѧϰ������������Ԫ��Ŀƽ��ֵ
        for yi=1:size(ensemble,2)
            [waste,tres]=classification(ensemble(yi),testdata,testtarget);%���з����ý����tres���浱ǰ����������ʵ������
            tresult=[tresult,tres];%tresult�������з������Բ������ݵķ�����
            numcount=numcount+size(ensemble(yi).hiddenoutput,2);
        end
        numcount=numcount/(size(ensemble,2));
        ttweight=zeros(size(testdata,1),labelnum);%����ͶƱ���ۻ�Ȩֵ
        for vi=1:size(tresult,1)%ͳ��ͶƱ���
            for vj=1:size(tresult,2)
                ttweight(vi,tresult(vi,vj))=ttweight(vi,tresult(vi,vj))+weight(vj);%�ۼ�Ȩֵ
            end
        end
        [waste,lastlabel]=max(ttweight,[],2);%���ȨֵͶƱԭ��
        ttcount=0;%ͳ�Ʋ��������б���ȷ�����������Ŀ
        for ef=1:size(lastlabel,1)
            if lastlabel(ef,1)==testtarget(ef,1)%������ȷ
                ttcount=ttcount+1;
            end
        end
        currentright=ttcount/(size(lastlabel,1));%��ǰ���ݲ��Ե�׼ȷ��
        accuracy=[accuracy,currentright];
        disp(['��',num2str(cc),'�β��Խ����׼ȷ��Ϊ:',num2str(currentright),'   ��������Ԫ����Ŀƽ��ֵΪ��',num2str(numcount)]);
    end%mod
    
    i=i+2*winsize;%������һ������
end
disp(['���㷨�ڵ�ǰ���ݼ��ϲ��Խ����ƽ��׼ȷ��Ϊ:',num2str(mean(accuracy(1,6:size(accuracy,2))))]);
toc;
