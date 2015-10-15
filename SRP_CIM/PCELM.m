
function [TrainingTime,TestingTime,TrainingAccuracy, TestingAccuracy] = PCELM(matrix_train, label_train, matrix_test, label_test,Num_hidden,C)
% Usage: pcelm(TrainingData, TrainingLabel, TestingData, TestingLabel, NumberofHiddenNeurons,Regularization Term)
%
% Input:
% TrainingData     - n1 * d (n1 and d are sample size and dimensionality)
% TrainingData     - n2 * d (n2 and d are sample size and dimensionality)
% label_train and label_test  -  label: 1 and -1
% NumberofHiddenNeurons - Number of hidden neurons 
%  C     - The regularization term for Least-square layer
% num_subfeature    - The 
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%


%%%%%%%%%%%%%%%%%%%%%%%
%%%%size about training data and testing data acquired
%%%%%%%%%%%%%%%%%%%%%%%
size_matrix = size(matrix_train);
NumberofTrainingData = size_matrix(1);
len_feature = size_matrix(2);
size_matrix = size(matrix_test);
NumberofTestingData = size_matrix(1);
num_subfeature = floor(sqrt(len_feature));     

start_time_train=cputime;

%%%%%%%%%%% Calculate weights
Output_hidden_train = zeros(NumberofTrainingData,Num_hidden);
Output_hidden_test = zeros(NumberofTestingData,Num_hidden);
Feature_Sele = randi(len_feature,Num_hidden,num_subfeature);
Input_Weight = zeros(num_subfeature,Num_hidden); % Input_weight (n m) where n is number of sub-features, m is the iteration times.
BiasofHiddenNeurons = rand(1,Num_hidden);
beta1 = 0;
beta = 2;

for i=1:Num_hidden  
    train_data_100 = matrix_train(:,Feature_Sele(i,:));
    test_data_100 = matrix_test(:,Feature_Sele(i,:));    
    [W,Output_hidden_train(:,i),Output_hidden_test(:,i)] = semiLDA( train_data_100, test_data_100,label_train,1,beta,beta1);% weights are learned through SRP
    Input_Weight(:,i) = W;
end

%%%%%%%%%%% Calculate bias
BiasMatrix_train = repmat(BiasofHiddenNeurons,NumberofTrainingData,1);
BiasMatrix_test = repmat(BiasofHiddenNeurons,NumberofTestingData,1);

%%%%%%%%%%% Calculate hidden neuron output with sigmoid function
train_data =   Output_hidden_train + BiasMatrix_train;
test_data = Output_hidden_test + BiasMatrix_test;
H_train = 1 ./ (1 + exp(-train_data));  
H_test = 1 ./ (1 + exp(-test_data));

%%%%%%%%%%%%%%% Calculate output weights OutputWeight
OutputWeight=inv(eye(size(H_train',1))/C+H_train' * H_train) * H_train' * label_train;
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train      %   Calculate CPU time (seconds) spent for training ELM
Y=(H_train * OutputWeight); 
Y(Y>0)=1;
Y(Y==0)=1;
Y(Y<0)=-1;
label_trainp = Y;
start_time_test=cputime;
TY=(H_test * OutputWeight);
end_time_test=cputime;
TestingTime = end_time_test-start_time_test
TY(TY>0)=1;
TY(TY<0)=-1;
label_testp=TY;
TrainingAccuracy = 1-length(find(label_train-label_trainp ))/length(label_train)
TestingAccuracy = 1-length(find(label_test-label_testp ))/length(label_test)
end

