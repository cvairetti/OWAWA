function weights = IOWAkNN(points,k)
points = zscore(points);
nPoints = size(points, 1);
neighbours = getKNN(points, k);
%check for regularisation
if(k>size(points, 2) )
    fprintf(1,'   [note: K>D; regularization will be used]\n');
    tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
    tol=0;
end
%convert to format used by my_lle
W = zeros(k,nPoints);
X = points';
neighborhood = neighbours';
%get weights
for i=1:nPoints
    z = X(:,neighborhood(:,i))-repmat(X(:,i),1,k); % shift ith pt to origin
    C = z'*z;                                        % local covariance
    C = C + eye(k,k)*tol*trace(C);                   % regularlization (K>D)
    W(:,i) = C\ones(k,1);                           % solve Cw=1
    W(:,i) = W(:,i)/sum(W(:,i));                  % enforce sum(w)=1
end;
%revert to normal notation
W = W';
%find points with large weights
weights = sum(abs(W),2);
%[sorted idxsToSort] = sort(summed, 'descend');
end

