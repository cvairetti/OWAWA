function SVMModelF = OWASVMf_fitclinear(Y,X,SVMModel,Regu,OWAtype,alphaOWA,quantif,IOWAslack,OWAWAbeta,alphaOWAWA,quantifOWAWA,finalflag)
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
        slack=1-Y.*(X*SVMModel.Beta+SVMModel.Bias);
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
Weig=B/mean(B);
if finalflag==1
    SVMModelF = fitclinear(X,Y,'Weights',Weig,'OptimizeHyperparameters','auto');
else
    SVMModelF = fitclinear(X,Y,'Weights',Weig,'lambda',SVMModel.Lambda,'Regularization',Regu,'Learner',SVMModel.Learner);
end
end
