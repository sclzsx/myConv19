clear all;close all;clc;

%%SEIR模型
A = [0.5 0.1 0.05 0.02];
[t,h] = ode45(@(t,x)SEIR(t,x,A),[0 300],[0.01 0.98 0.01 0]);  %[初始感染人口占比 初始健康人口占比 初始潜伏人口占比 初始治愈人口占比]
plot(t,h(:,1),'r');
hold on;
plot(t,h(:,2),'b');
plot(t,h(:,3),'m');
plot(t,h(:,4),'g');
legend('感染人口占比I','健康人口占比S','潜伏人口占比E','治愈人口占比R');
title('SEIR模型')

data=[t h];
data = data(1:3:80,:);      %间隔取一部分数据用来拟合
figure;
plot(data(:,1),data(:,2),'ro');
hold on;
plot(data(:,1),data(:,3),'bo');
plot(data(:,1),data(:,4),'mo');
plot(data(:,1),data(:,5),'go');

T=min(data(:,1)):0.1:max(data(:,1));        %插值处理，如果数据多，也可以不插值
I=spline(data(:,1),data(:,2),T)';
S=spline(data(:,1),data(:,3),T)';
E=spline(data(:,1),data(:,4),T)';
R=spline(data(:,1),data(:,5),T)';

plot(T,I,'r.');plot(T,S,'b.');
plot(T,E,'m.');plot(T,R,'g.');

%求微分，如果数据帧间导数变化太大，可以先平均或者拟合估计一个导数
%因为前面T是以0.1为步长，这里乘以10
dI = diff(I)*10; dI=[dI;dI(end)];       
dS = diff(S)*10; dS=[dS;dS(end)];
dE = diff(E)*10; dE=[dE;dE(end)];
dR = diff(R)*10; dR=[dR;dR(end)];

X = [zeros(length(I),1) -I.*S zeros(length(I),2);   %构造线性最小二乘方程组形式
     -E I.*S -E zeros(length(I),1);
     E zeros(length(I),2) -I;
     zeros(length(I),2) E I];
Y = [dS;dE;dI;dR];

A = inv(X'*X)*X'*Y

%用估计参数代入模型
[t,h] = ode45(@(t,x)SEIR(t,x,A),[0 300],[I(1) S(1) E(1) R(1)]);  %[初始感染人口占比 初始健康人口占比 初始潜伏人口占比 初始治愈人口占比]
plot(t,h(:,1),'r');
hold on;
plot(t,h(:,2),'b');
plot(t,h(:,3),'m');
plot(t,h(:,4),'g');

function dy=SEIR(t,x,A)
%x(3):E, x(1):I, x(2):S
alpha = A(1);  
beta = A(2);
gamma1 = A(3);
gamma2 = A(4);
dy=[alpha*x(3) - gamma2*x(1);%dI
    -beta*x(1)*x(2);%dS
    beta*x(1)*x(2) - (alpha+gamma1)*x(3);%dE
    gamma1*x(3)+gamma2*x(1)];%dR
end
