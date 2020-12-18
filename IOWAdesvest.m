function slack = IOWAdesvest(X)
xbar=mean(X);
for i=1:size(X,1)
    desv(i,:)=abs(X(i,:)-xbar);
end
slack=sum(desv,2);
end



