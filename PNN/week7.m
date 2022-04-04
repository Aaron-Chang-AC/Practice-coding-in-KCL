function [eigenvector, eigenvalue] = KLT(X)
    M = mean(X);
    diff = X - M;
    numX = length(X);
    for i = 1:numX
        c_matrix = diff(:)*diff(:).'; % outerproduct
    end
    c_matrix = (1/numX)*(c_matrix); 


end


X = [1 2 1; 2 3 1; 3 5 1; 2 2 1];
