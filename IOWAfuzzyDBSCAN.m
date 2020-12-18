function [IDX,FuzzyIDX]=IOWAfuzzyDBSCAN(X,epsilon,MinPtsMin,MinPtsMax)
    C=0;
    n=size(X,1);
    IDX=zeros(n,1);
    FuzzyIDX=zeros(n,1);
    D=pdist2(X,X);   
    visited=false(n,1);
    isnoise=false(n,1);
    for i=1:n
        if ~visited(i)
            visited(i)=true;           
            Neighbors=RegionQuery(i);
            if numel(Neighbors)<MinPtsMin
                isnoise(i)=true;
            else
                C=C+1;
                ExpandCluster(i,Neighbors,C,MinPtsMin,MinPtsMax);
            end    
        end    
    end
    
    function ExpandCluster(i,Neighbors,C,MinPtsMin,MinPtsMax)
        IDX(i)=C;
        CardNeighbors=size(Neighbors,2);
        if CardNeighbors >= MinPtsMax 
            FuzzyIDX(i)=1;
        elseif CardNeighbors <= MinPtsMin
            FuzzyIDX(i)=0;
        else
            FuzzyIDX(i)=(CardNeighbors-MinPtsMin)/(MinPtsMax-MinPtsMin);
        end
        k = 1;
        while true
            j = Neighbors(k);            
            if ~visited(j)
                visited(j)=true;
                Neighbors2=RegionQuery(j);
                if numel(Neighbors2)>=MinPtsMin
                    Neighbors=[Neighbors Neighbors2];  
                end
            end
            if IDX(j)==0
                IDX(j)=C;
                if ~visited(j)
                    CardNeighbors=size(Neighbors2,2);
                    if CardNeighbors >= MinPtsMax
                        FuzzyIDX(j)=1;
                    elseif CardNeighbors <= MinPtsMin
                        FuzzyIDX(j)=0;
                    else
                        FuzzyIDX(j)=(CardNeighbors-MinPtsMin)/(MinPtsMax-MinPtsMin);
                    end
                end
            end           
            k = k + 1;
            if k > numel(Neighbors)
                break;
            end
        end
    end
    
    function Neighbors=RegionQuery(i)
        Neighbors=find(D(i,:)<=epsilon);
    end

end



