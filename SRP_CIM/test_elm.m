clear
matrix_train = load('D:\data\20news_SRP\matrix_train1');
matrix_test = load('D:\data\20news_SRP\matrix_test1');
label_train = load('D:\data\20news_SRP\train_label1');
label_test = load('D:\data\20news_SRP\test_label1');
matrix_train = matrix_train.A;
matrix_test = matrix_test.B;
label_train = label_train.C;
label_train(label_train==0)=-1;
label_test = label_test.D;
label_test(label_test==0)=-1;
test_acc = zeros(10,1);
run_time = zeros(10,1);



for i=1:10
[a,b,c,d]=PCELM(matrix_train, label_train, matrix_test, label_test, 500, 80);
test_acc(i) = d;
run_time(i) = a +b;
end
mean(test_acc)
std(test_acc)
mean(run_time)