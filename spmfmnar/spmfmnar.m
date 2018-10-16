function [finalresult]=spmfmnar(chdata,chtest,chtrust)
fid=fopen('a2.txt','wt');
CoreNum=12;



k=10; %number of latent dimensions of u,v
nep=15; %number of global iteration
net=10; %number of local iteration
data=load(chdata); %load training data
test=load(chtest); %load test data
trust=load(chtrust); %load trust data
nzhong=max([data;test]);
tzhong=max(trust);
n=max(max(nzhong(1),tzhong(1)),tzhong(2));
m=nzhong(2);
HT=sparse(trust(:,1),trust(:,2),ones(length(trust),1),n,n);
HT=HT|HT';
[trsuti,trustj]=find(HT);
trust=[trsuti,trustj];

tr=data;
tt=test;

nn=size(tr,1);
nt=size(tt,1);

PR=sparse(test(:,1),test(:,2),test(:,3),n,m);
pid=cell(n,1);
pr=cell(n,1);
YR1=sparse(data(:,1),data(:,2),data(:,3),n,m);
YR2=sparse(data(:,1),data(:,2),ones(nn,1),n,m);
YR3=sparse([data(:,1);test(:,1)],[data(:,2);test(:,2)],ones(nn+nt,1),n,m);

YR=(YR1>0);
YT=sparse(trust(:,1),trust(:,2),ones(length(trust(:,1)),1),n,n);
TR=cell(n,1);
TR1=YT;
gbei=cell(n,1);
hbei=cell(n,1);
mbei=zeros(n,1);
pbei=cell(n,1);
gg=cell(n,1);
ge=cell(n,1);
gf=cell(n,1);
gx=cell(n,1);
gh=cell(n,1);
aucn=zeros(n,1);

he=cell(m,1);
hf=cell(m,1);
hx=cell(m,1);
for i=1:m
      he{i}=find(YR(:,i)>0);
    hf{i}=YR1(he{i},i);
end
sig=1;

me=mean(tr(:,3));
sigu=sig;
sigv=sig;
siga=sig;
sigb=sig;
sigr=10;
sigz=1;
sigar=sig;
sigmar=sig;
sigbei=sig;
sigmbei=sig;
rel=0.5;
sigrel=sig;
sigd=5;
ybe=ones(2,2);
gbe=ones(2,2);
gae=2*ones(2,1);
hu=zeros(k,k,n);
fhu=zeros(k,k,n);
for i=1:n
    hu(:,:,i)=diag(sigu*ones(k,1));
    fhu(:,:,i)=inv(hu(:,:,i));
end
pu=zeros(n,k);
gu=(rand(n,k))*0.1;
hv=zeros(k,k,m);
fhv=zeros(k,k,m);
for i=1:m
    hv(:,:,i)=diag(sigv*ones(k,1));
    fhv(:,:,i)=inv(hv(:,:,i));
end
gv=(rand(m,k))*0.1;
pv=zeros(m,k);
ha=siga*ones(n,1);
ga=(rand(n,1))*0.1;
pa=zeros(n,1);
hb=sigb*ones(m,1);
gb=(rand(m,1))*0.1;
pb=zeros(m,1);
har=sigar*ones(n,1);
gar=ones(n,1)*0.1;
par=zeros(n,1);
mar=0.1;
gc=rand(n,1)*0.1;
hc=sigrel*ones(n,1);
pc=zeros(n,1);
gd=ones(n,1)*me;
hd=sigd*ones(n,1);
pd=zeros(n,1);
md=me;
ybeio=ones(n,1);
beio=ones(n,1);
[qq,qw,qe]=find(TR1);
ybei1=sparse(qq,qw,ones(length(qq),1),n,n);
bei1=ybei1;
al=psi(beio+sum(bei1,2));




for ep=1:nep
    %% local step
    % sample
    fprintf('epoch %d/%d\n',ep,nep);
    nsam=2*nn;
    rasam=0.5;
    jiang=1;
    ran=randperm(nn,floor(rasam*nsam));
    sam1=[tr(ran,:),ones(floor(rasam*nsam),1)];
   x=repmat(sam1(:,1),jiang,1);
    y=randi(m,length(x),1);
    
    sam2=[x,y,YR1(sub2ind(size(YR1),x,y)),YR2(sub2ind(size(YR2),x,y))];
    id=find(sam2(:,4)==0);
    sam2=sam2(id,:);
    sam=[sam1;sam2];
    nsam=length(sam);
    %update z,xi
    %init
    x=sam(:,1);
    y=sam(:,2);
    gz=sum(gu(x,:).*gv(y,:),2)+me+ga(x,:)+gb(y,:);
    hz=sigz*ones(nsam,1);

     zhongt=TR1(x,:);
    [qq,qw,qe]=find(zhongt);
    zhongbei=bei1(x,:);
    [~,~,qf]=find(zhongbei);


    for et=1:net
    %gao xi
  
    xc=rel*sqrt(1./hz+gz.^2+1./hd(x)+gd(x).^2-2*gd(x).*gz);
    xl=gaola(xc);
    %gao fa
     fao=exp(psi(beio(x))-al(x)+log(1./(1+exp(-xc)))+((2*sam(:,4)-1)*rel.*(gz-gd(x))-xc)/2);

 
    zhongqg=YR2(sub2ind(size(YR2),qw,y(qq)));
    qg=exp(psi(full(qf))-al(x(qq))+psi(gbe(4-zhongqg*2-sam(qq,4)))-psi(gae(2-zhongqg)));

    fa1=sparse(qq,qw,qg,nsam,n);

    fsum1=sum(fa1,2)+fao;
 
    fao=fao./fsum1;

    fa1=bsxfun(@rdivide,fa1,fsum1);
 
    hz=sigz+sigr.*sam(:,4)+2*xl.*rel^2.*fao;
    pre=double(me+ga(x)+gb(y)+(sum(gu(x,:).*gv(y,:),2)));
    gz=((2*sam(:,4)-1)*rel.*fao/2+sigr*sam(:,4).*sam(:,3)+sigz*(pre)+2*rel^2*xl.*fao.*gd(x))./hz;
    end
    
  %% global step   
  ro=ep^(-0.8);
    % update a,b,u,v  
        rat=1;
      [gid,ge]=myv2c(full(sam(:,1)),full(sam(:,2)),n);
      [~,gf]=myv2c(full(sam(:,1)),full(gz),n);
      [~,gl]=myv2c(full(sam(:,1)),full(xl),n);
      [~,gh]=myv2c(full(sam(:,1)),full(hz),n);
      [~,gx]=myv2c(full(sam(:,1)),full(sam(:,4)),n);
      [~,gfa]=myv2c(full(sam(:,1)),full(fao),n);
      [~,he]=myv2c(full(sam(:,2)),full(sam(:,1)),m);
      [~,hf]=myv2c(full(sam(:,2)),full(gz),m);



      


    %up u,a
      op=[1:n];
        parfor i=1:n
          
            it=ge{i};
            numit=length(it)+1e-12;
            nha=length(it)*sigz*rat+siga;
            pre=gu(i,:)*gv(it,:)'+me+gb(it)'-gf{i};
            npa=-sum(pre)*sigz*rat;
            ha(i)=ha(i)*(1-ro)+ro*nha;
            pa(i)=pa(i)*(1-ro)+ro*npa;
            ga(i)=pa(i)/ha(i);

              nhu=((sum(fhv(:,:,it),3)+gv(it,:)'*gv(it,:))*rat*sigz+sigu*eye(k));
            npu=((gf{i}-me-ga(i)-gb(it)')*gv(it,:))*rat*sigz;

            pu(i,:)=pu(i,:)*(1-ro)+npu*ro;
            hu(:,:,i)=hu(:,:,i)*(1-ro)+nhu*ro;
            fhu(:,:,i)=inv(hu(:,:,i));
            gu(i,:)=pu(i,:)*fhu(:,:,i)'; 
        end


        %up v,b
        op=[1:m];
        parfor i=1:m
      
            it=he{i}';
            numit=length(it)+1e-12;
            nhb=length(it)*sigz*rat+sigb;
            pre=gu(it,:)*gv(i,:)'+me+ga(it)-hf{i}';
            npb=-sum(pre)*sigz*rat;
            hb(i)=hb(i)*(1-ro)+ro*nhb;
            pb(i)=pb(i)*(1-ro)+ro*npb;
            gb(i)=pb(i)/hb(i);

             nhv=((sum(fhu(:,:,it),3)+gu(it,:)'*gu(it,:))*rat*sigz+sigv*eye(k));
            npv=(hf{i}-me-ga(it)'-gb(i))*gu(it,:)*rat*sigz;

            pv(i,:)=pv(i,:)*(1-ro)+ro*npv;
            hv(:,:,i)=hv(:,:,i)*(1-ro)+ro*nhv;
            fhv(:,:,i)=inv(hv(:,:,i));
            gv(i,:)=pv(i,:)*fhv(:,:,i)'; 
        end
        
        %up bei,d
       
         nhdzhong=fao.*xl;
         nhdzhong=myv2s(sam(:,1),full(nhdzhong),n);
         nhd=sigd+2*rel.^2*rat*nhdzhong;
         hd=hd*(1-ro)+ro*nhd;
     
         npdzhong=fao.*(2*gz.*xl*(rel^2)-rel/2*(2*sam(:,4)-1));
         npdzhong=myv2s(sam(:,1),full(npdzhong),n);
         npd=sigd*md+rat*npdzhong;
         pd=pd*(1-ro)+ro*npd;
         gd=pd./hd;
        md=mean(gd'*hd+me*10)./(10+sum(hd));


         zhongbei=sparse(sam(:,1),[1:nsam]',ones(nsam,1),n,nsam);
         qbei1=ybei1+zhongbei*fa1;

          bei1=bei1*(1-ro)+ro*qbei1;
         nbeio=myv2s(sam(:,1),full(fao),n);
         beio=beio*(1-ro)+ro*(ybeio+nbeio*rat);
         al=psi(beio+sum(bei1,2));
       
          % up be 
          [qq,qw,qe]=find(fa1);
          zhongyr=YR2(sub2ind(size(YR2),qw,y(qq)));
          zhongnbei=(zhongyr==1&sam(qq,4)==1).*qe;
          nbe111=sum(zhongnbei);
          zhongnbei=(zhongyr==1&sam(qq,4)==0).*qe;
          nbe110=sum(zhongnbei);
          zhongnbei=(zhongyr==0&sam(qq,4)==1).*qe;
          nbe101=sum(zhongnbei);
           zhongnbei=(zhongyr==0&sam(qq,4)==0).*qe;
          nbe100=sum(zhongnbei);
          nbe=ybe+[nbe111,nbe110;nbe101,nbe100];
         gbe=nbe*(1-ro)+ro*nbe;
         gae=sum(gbe,2);
         
end



 
%% test
%testprediction
a1=tt(:,1);
b1=tt(:,2);
c1=tt(:,3);
nt=length(tt);
  % local step
 
sam=[tt,ones(nt,1)];
nsam=length(sam);
%update z,xi
%init
x=sam(:,1);
y=sam(:,2);
gz=sum(gu(x,:).*gv(y,:),2)+me+ga(x,:)+gb(y,:);
hz=sigz*ones(nt,1);
zhongt=TR1(x,:);
[qq,qw,qe]=find(zhongt);
 zhongbei=bei1(x,:);
 [~,~,qf]=find(zhongbei);

for et=1:net

  xc=rel*sqrt(1./hz+gz.^2+1./hd(x)+gd(x).^2-2*gd(x).*gz);
  xl=gaola(xc);

  fao=exp(psi(beio(x))-al(x)+log(1./(1+exp(-xc)))+((2*sam(:,4)-1)*rel.*(gz-gd(x))-xc)/2-xl.*(rel^2*(gz.^2+1./hz+gd(x).^2+1./hd(x)-2*gz.*gd(x))-xc.^2));
 
  zhongqg=YR3(sub2ind(size(YR3),qw,y(qq)));
  qg=exp(psi(full(qf))-al(x(qq))+psi(gbe(4-zhongqg*2-sam(qq,4)))-psi(gae(2-zhongqg)));
  fa1=sparse(qq,qw,qg,nsam,n);

  fsum1=sum(fa1,2)+fao;
  fao=fao./fsum1;
  fa1=bsxfun(@rdivide,fa1,fsum1);
  hz=sigz+2*xl.*rel^2.*fao;
  pre=double(me+ga(x)+gb(y)+(sum(gu(x,:).*gv(y,:),2)));
  gz=((2*sam(:,4)-1)*rel.*fao/2+sigz*(pre)+2*rel^2*xl.*fao.*gd(x))./hz;

end

pre=gz;
id=find(pre>5);
pre(id)=5;
id=find(pre<1);
pre(id)=1;
mae=sum(abs(pre-c1))/nt;
rmse=sqrt(sum((pre-c1).^2)/nt);
fprintf('mae=%f rmse=%f\n',mae,rmse);

% testrank-ndcg: make recommenation   
refhv=reshape(fhv,k*k,m);
refhu=reshape(fhu,k*k,n);
ciu=zeros(k,k,n);
for i=1:n
    ciu(:,:,i)=gu(i,:)'*gu(i,:);
end
reciu=reshape(ciu,k*k,n);

civ=zeros(k,k,m);
for i=1:m
    civ(:,:,i)=gv(i,:)'*gv(i,:);
end
reciv=reshape(civ,k*k,m);

FC=(refhu')*(refhv+reciv)+reciu'*refhv+repmat(1./(ha+hd),1,m)+repmat(1./hb',n,1)+1/sigz;
fc=(1+pi*FC/8).^(-0.5);
gc=me+gu*gv'+repmat(ga,1,m)+repmat(gb',n,1)-gd;
D1=1./(1+exp(-fc.*gc));
clear FC;
clear fc;
clear gc;
nbq=nbe./repmat(sum(nbe,2),1,2);
bgs=beio+sum(bei1,2);
bq=beio./bgs;
bw=bsxfun(@rdivide,bei1,bgs);

D2=zeros(n,m);
for i=1:n
    TR{i}=find(TR1(i,:));
    [~,~,bwa]=find(bw(i,:));
    D2(i,:)=bwa*nbq(2-YR3(TR{i},:));
end
D=D1.*bq+D2;


r1=data(:,1);
r2=data(:,2);
D(sub2ind(size(D),r1,r2))=-inf;   


% testndcg: rank based on D.
H=(PR>=3);
num=sum(H,2);
idcg=zeros(n,1);
   for i=1:n 
        for j=1:num(i)
            idcg(i)=idcg(i)+1/log2(j+1);
        end
   end
nt=length(test(:,1));
 num=sum(H,2);
 ndcg=zeros(n,1);
ca=0;  
pq=zeros(1,m);
for i=1:n
    id1=find(H(i,:)~=0);
    [~,an]=sort(D(i,:),2,'descend');
    if(num(i)~=0)
    pq(an(1,1:m))=1:m;
    pg=pq(id1);
    ndcg(i)=(sum(H(i,id1)./(log2(pg+1))))/idcg(i);   
   
    end  
end
ca=full(sum(num~=0));


nd=sum(ndcg)/ca;

fprintf('ndcg=%f\n',nd);
end
          
        
   
            
            
            
        
        