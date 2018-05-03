% Model-Free Q-Learning Designs for Linear Discrete-Time Zero-Sum Games
% with Application to H-infinity Control: Example
close all;
clc;

%% F-16 discrete-time short-period pitch dynamics (ZOH)
% x = [alpha q delta_e]
% alpha = angle of attack
% q = pitch rate
% delta_e = elevator deflection angle
x0 = [10 5 -2].'; 
A = [0.906488 0.0816012 -0.0005;
     0.0741349 0.90121 -0.000708383;
     0 0 0.132655];
 
B = [-0.00150808;
     -0.0096;
     0.867345];
 
E = [0.00951892;
     0.00038373;
     0];
 
dt = 0.1; % seconds

gamma = 1; % disturbance attenuation

% The known solution to the DARE:
[Ptrue, ~, G] = dare(A, [B E], eye(3), [eye(1) 0; 0 -gamma^2*eye(1)]);
Ltrue = -G(1,:);
Ktrue = -G(2,:);

Htrue = [A.'*Ptrue*A + eye(3) A.'*Ptrue*B A.'*Ptrue*E;
         B.'*Ptrue*A B.'*Ptrue*B + eye(1) B.'*Ptrue*E;
         E.'*Ptrue*A E.'*Ptrue*B E.'*Ptrue*E - gamma^2*eye(1)];

% Model-Free Online Tuning
[~, n] = size(A); % state dimension
[~, m1] = size(B); % control input dimension
[~, m2] = size(E); % disturbance dimension
T = 80000; % number of timesteps in simulation
T_learn = 30000; % number of timesteps to learn
gamma_learn = 1; % learning discount factor

w = randn(1, floor(dt*T)); % disturbance input
% w = sin(1:floor(dt*T));
w = ones(1, floor(dt*T));
dynamics = @(xin, uin, win) A*xin + B*uin + E*win; % dynamics

[t, x, L, K, H, P] = q_learning_h_infinity(dynamics, n, m1, m2, x0, w, dt, T, gamma, T_learn, gamma_learn);

%% Results
figure()
subplot(3,1,1)
plot(t, x(1,:))
grid on
ylabel('x_1')
subplot(3,1,2)
plot(t, x(2,:))
grid on
ylabel('x_2')
subplot(3,1,3)
plot(t, x(3,:))
ylabel('x_3')
xlabel('Time (s)')
grid on

iter = length(K);
Kvec = reshape([K{:}],[n,length(K)]);
figure()
subplot(1, 2, 1)
hold on
stairs(1:iter, Kvec(1,:))
stairs(1:iter, Kvec(2,:))
stairs(1:iter, Kvec(3,:))
title('K')
xlabel('Q-Learning Iteration')
grid on

% figure()
subplot(1, 2, 2)
semilogy(1:iter, vecnorm(Kvec-repmat(Ktrue', 1, length(Kvec))));
title('||K_i-K_{true}||')
xlabel('Q-Learning Iteration')
grid on

figure()
subplot(1, 2, 1)
Lvec = reshape([L{:}],[n,length(L)]);
hold on
stairs(1:iter, Lvec(1,:))
stairs(1:iter, Lvec(2,:))
stairs(1:iter, Lvec(3,:))
title('L')
xlabel('Q-Learning Iteration')
grid on

% figure()
subplot(1, 2, 2)
semilogy(1:iter, vecnorm(Lvec-repmat(Ltrue', 1, length(Lvec))));
title('||L_i-L_{true}||')
xlabel('Q-Learning Iteration')
grid on