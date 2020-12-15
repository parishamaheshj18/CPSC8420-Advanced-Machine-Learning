Z=[1 2 3;2 0 0;0 6 0];
Z_old=zeros(size(Z));
r=1;
index_recover=[2 2;2 3;3 1;3 3];
index_filled=[1 1;1 2;1 3;2 1;3 2];
for K=1:1000
    Z_comb=combine(Z,Z_old,index_filled,index_recover);
    Z_old=hardImpute(Z_comb,r);
end
Z_old

function [Z_comb]=combine(a,b,index_filled,index_recover)
    Z_comb=zeros(size(a));
    for i=1:size(index_recover,1)
       Z_comb(index_recover(i,1),index_recover(i,2))=b(index_recover(i,1),index_recover(i,2));
    end
    for i=1:size(index_filled,1)
       Z_comb(index_filled(i,1),index_filled(i,2))=a(index_filled(i,1),index_filled(i,2));
    end
end
function [M]=hardImpute(Z_comb,r)
    M=zeros(size(Z_comb));
    [U,S,V]=svd(Z_comb);
    for k=1:r
        M=S(k,k)*U(:,k)*V(:,k)';
    end
end