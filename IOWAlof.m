function lofValue = IOWAlof(x,testx,k)
%params.minptslb: Lower bound of minpts, determines how many
%       neighbors are uses for calculation of LOF
%       params.minptsub: Upper bound for minpts
N = size(x,1);
tN = size(testx,1);
minptsub = k;
minptslb = k;
kVals = minptslb:minptsub;
kCount = size(kVals,2);
lofValue = zeros(tN,1);
resultslof = zeros(tN, kCount);


% calculate kdistances and kneighbors for samples in training data
kdistances = zeros(N, kCount);
kneighbors = cell(N, kCount);
for i = 1:N
    dist = sum((x - repmat(x(i,:), N, 1)).^2, 2);
    dist(i) = max(dist);
    sdist = sort(dist);
    kdistances(i,:) = sdist(kVals);
    for k = 1:kCount
        kneighbors{i, k} = find(dist <= kdistances(i,k));
    end
end

% for every sample in test data
for i = 1:tN
    % find k-distance and neighbors of test sample
    dist = sum((x - repmat(testx(i,:), N, 1)).^2, 2);
    sdist = sort(dist);
    for k = 1:kCount
        kdist = sdist(kVals(k));
        neighbors = find(dist <= kdist);
        neighborCount = size(neighbors,1);
        % calculate reachability distance of test sample to each sample
        % in neighborhood
        rd = max( kdistances(neighbors, k), dist(neighbors));
        lrd = neighborCount ./ sum(rd);
        % calculate local reachability distance of all neighbors
        nlrd = zeros(1, neighborCount);
        for n = 1:neighborCount
            nkneighbors = kneighbors{neighbors(n), k};
            nkncount = size(nkneighbors,1);
            ndist = sum( (x(nkneighbors,:) - repmat(x(neighbors(n),:), nkncount, 1)).^2, 2);
            nrd = max( kdistances(nkneighbors, k), ndist);
            nlrd(1, n) = nkncount ./ sum(nrd);
        end
        lof = sum(nlrd) ./ (neighborCount .* lrd);
        if isnan(lof) || isinf(lof)
            lof = 0;
        end
        resultslof(i,k)  = lof;
    end
    lofValue(i) = max(resultslof(i,:));
end
end

