Z=[1 2 3;2 0 0;0 6 0];
r=1;
index=[2 2;2 3;3 1;3 3];
for K=1:1000
    [U,S,V]=svd(Z);
    M=S(1,1)*U(:,1)*V(:,1)';
    Z=getMask(Z,M,index);
end
M
function [a]=getMask(a,b,index)
    for i=1:size(index,1)
       a(index(i,1),index(i,2))=b(index(i,1),index(i,2));
    end
end