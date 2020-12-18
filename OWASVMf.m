function [Ytest,solvector,bias,nSV] = OWASVMf(Y,X,Xt,C,sigma,OWAtype,alphaOWA,quantif,IOWAslack,OWAWAbeta,alphaOWAWA,quantifOWAWA)
%OWAtype=1 => OWA	OWAtype=2 => IOWA,	OWAtype=3 => OWAWA,   OWAtype=4 => IOWAWA
%quantif=1 => Basic RIM quantifier, quantif=2 => Quadratic quantifier, quantif=3 => Exponential quantifier,
%quantif=4 => Trigonometric quantifier, quantif=5 => O'Hagans method.
%IOWAslack=1 => Suma de desvest IOWAslack=2 => fuzzyDBSCAN IOWAslack=3 =>
%kNN + sumdist IOWAslack=4=> LOF IOWAslack=5 => Parzen window
if OWAWAbeta==0
    IOWAslack=6;
end
slack=zeros(size(Y,1),1);
weight = OWAFunction(size(slack,1),alphaOWA,quantif);
switch OWAtype
    case {1,3}
        if sigma==0
            strlibsvm=strcat({'-c '}, {num2str(C)}, {' -t '}, {num2str(0)});
            model = svmtrain(Y,X,strlibsvm{1});
            w = model.SVs' * model.sv_coef;
            slack=1-Y.*(X*w-model.rho); %slack negativos pueden ser 0 o no
        else
            gamma=1/(2*sigma*sigma);
            strlibsvm=strcat({' -c '}, {num2str(C)}, {' -g '}, {num2str(gamma)});
            model = svmtrain(Y,X,strlibsvm{1});
            Xsv=full(model.SVs);
            kernelMatrix = Xsv*Xsv';
            kerMaDegN = sum(Xsv.^2,2);
            kernelMatrix = ones(size(Xsv,1),1)*kerMaDegN' + kerMaDegN*ones(1,size(Xsv,1)) - 2*kernelMatrix;
            kernelMatrix = exp(-kernelMatrix/(2*sigma^2));
            f=model.sv_coef'*kernelMatrix-model.rho;
            slack(model.sv_indices)=1-Y(model.sv_indices).*f'; %slack negativos pueden ser 0 o no
        end
    case {2,4}
        switch IOWAslack
            case 1 
                slack=IOWAdesvest(X);
            case 2
                slack=IOWAparzenwindow(X,X,10); %k
            case 3
                slack=IOWAlof(X,X,10);%k
            case 4
                slack=IOWAfuzzyDBSCAN(X,0.5,5,20);%(epsilon,MinPts,MaxPts)
            case 5                
                slack=IOWAkNN(X,10);%k
            case 6
                slack=ones(size(X,1),1); %if OWAWAbeta=0
        end
end
[B,I]=sort(slack);
if OWAtype<=2
    B=weight(I)';
else
    weight2 = OWAFunction(size(slack,1),alphaOWAWA,quantifOWAWA);
    weight2=sort(weight2); %lower weight for older period
    B=OWAWAbeta*weight(I)'+(1-OWAWAbeta)*weight2'; 
end    
K = X*X';
opts= optimset('display','off','MaxIter',10000,'LargeScale','off');
if sigma==0
    K=K.*(Y*Y');
    [alpha,fval,exitflag,output,lambda]= quadprog(K,-ones(size(K,1),1),[],[],Y',0,zeros(size(K,1),1),C*B/mean(B),[],opts);
    bias=lambda.eqlin;
    clear K
    alpha=alpha.*Y;
    solvector=X'*alpha;
    Ytest=(Xt*solvector+bias);
else
    kerMaDegN = sum(X.^2,2);
    kernelMatrix = ones(size(X,1),1)*kerMaDegN' + kerMaDegN*ones(1,size(X,1)) - 2*K;
    K = exp(-kernelMatrix/(2*sigma^2));
    K=K.*(Y*Y');
    [alpha,fval,exitflag,output,lambda]= quadprog(K,-ones(size(K,1),1),[],[],Y',0,zeros(size(K,1),1),C*B/mean(B),[],opts);
    bias=lambda.eqlin;
    clear K
    solvector=alpha.*Y;
    kernelMatrix = X*Xt';
    kerMaDegN = sum(X.^2,2);
    kerMaDegNT = sum(Xt.^2,2);
    kernelMatrix = ones(size(X,1),1)*kerMaDegNT' + kerMaDegN*ones(1,size(Xt,1)) - 2*kernelMatrix;
    kernelMatrix = exp(-kernelMatrix/(2*sigma^2));
    Ytest=(solvector'*kernelMatrix+bias)';  
end
calcsv=(C*B/mean(B))-alpha;
ind=calcsv<0.01;%?
nSV=sum(ind);
Ytest=sign(Ytest);
end

