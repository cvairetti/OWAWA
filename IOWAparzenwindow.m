function yprob = IOWAparzenwindow(x,testx,k)
N = size(x,1);
tN = size(testx,1);
yprob = zeros(tN,1);
for i = 1:tN
    s = getSigma(x, testx(i,:), k);
    yprob(i) = sum(exp(-sum((x - repmat(testx(i,:), N, 1)).^2 ./ repmat(s, N, 1), 2))) ./ N;
end
end

function s = getSigma(x, p, k)
dist = (x - repmat(p, size(x,1), 1)).^2;
totdist = sum(dist,2);
for i=1:k
    totdist(totdist==min(totdist)) = max(totdist);
end
s = repmat(min(totdist) ./ 4, 1, size(x,2));
%     mini = find(totdist==min(totdist),1);
%     s = dist(mini,:) ./ 4; % 2 * std = dist
end



