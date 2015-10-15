function [T,Z1,Z2] = LDA(Input,Input_test,Target,r,beta)

% Input:
%    Input:      n x d matrix of original samples
%   
%    Input_test:  n x d matrix of original samples
%    Target: Labels for Input
%    r: reduced dimension
%    beta: regularization term

% Output:
%    T: d x r transformation matrix (Z=T'*X)
%    Z1: n x r matrix of dimensionality reduced traing samples 
%    Z2: n1 x r matrix of dimensionality reduced testing samples 

% Determine size of input data
Input_fina = Input;
Input_test_fina = Input_test;
Target_fina = Target;
idx=bsxfun(@(Input,y)all(Input==y),Input',0);
Input(idx,:)=[];
Target(idx,:)=[];
idx=bsxfun(@(Input_test,y)all(Input_test==y),Input_test',0);
Input_test(idx,:)=[];


[n d] = size(Input);
% Discover and count unique class labels
ClassLabel = unique(Target);
k = length(ClassLabel);

% Initialize
nGroup     = NaN(k,1);     % Group counts
GroupMean  = NaN(k,d);     % Group sample means
PooledCov  = zeros(d,d);   % Pooled covariance
betweenCla = zeros(d,d);
regCov     = eye(d);
W          = NaN(1,d);   % model coefficients
Wholemean  = mean(Input);
% Loop over classes to perform intermediate calculations
for i = 1:k,
    % Establish location and size of each class
    Group      = (Target == ClassLabel(i));
  
    nGroup(i)  = sum(double(Group));
    
    % Calculate group mean vectors
   
    GroupMean(i,:) = mean(Input(Group,:));
    
    % Accumulate pooled covariance information
    PooledCov = PooledCov + (nGroup(i) - 1) .* cov(Input(Group,:));
    betweenCla = betweenCla + nGroup(i).*((GroupMean(i,:)-Wholemean)'*(GroupMean(i,:)-Wholemean));
end

Sw = PooledCov+beta*regCov;
Sb = betweenCla;


Srlb=(Sb+Sb')/2;
Srlw=(Sw+Sw')/2;



if r==d
  [eigvec,eigval_matrix]=eig(Srlb,Srlw);
else
  opts.disp = 0; 
  [eigvec,eigval_matrix]=eigs(Srlb,Srlw,r,'la',opts);
end   
eigval=diag(eigval_matrix);

[sort_eigval,sort_eigval_index]=sort(eigval);

T=eigvec(:,sort_eigval_index(end:-1:1));
T = sqrt(sort_eigval(end)).*T;
Z1 = Input_fina*T;
Z2 = Input_test_fina*T;
end

