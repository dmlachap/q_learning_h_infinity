function [t, x, L, K, H, P] = q_learning_h_infinity(dynamics, n, m1, m2, x0, w, dt, T, gamma, T_learn, gamma_learn)
% dynamics: Dynamics function handle, of the form @(xin, uin, win) A*xin + B*uin + E*win
% n: state dimension
% m1: control input dimension
% m2: disturbance input dimension
% x0: initial state
% w: disturbance input
% dt: timestep (s)
% T: end time (s)
% gamma: disturbance attenuation (usually 1)
% T_learn: learning end time (s)
% gamma_learn: learning discount factor (best close to 1)

%% Model-free online tuning
Kend = floor(dt*T);
K_learn = floor(dt*T_learn);
t = dt*(0:Kend);

q = n + m1 + m2;
Nmin = q*(q+1)/2; % minimum number of data points needed

N = Nmin*2 + 1; % number of data points used in computation

sigma_1 = 3;
P1 = sigma_1*eye(m1);
sigma_2 = 3;
P2 = sigma_2*eye(m2);

R = eye(n);

x = zeros(n, Kend); x(:, 1) = x0;
u = zeros(m1, Kend);
% w = zeros(m2, Kend);
z = zeros(q, Kend);

H0 = eye(q);
h0 = v(H0);
L0 = zeros(m1, n);
K0 = zeros(m2, n);

P0 = [eye(n) L0.' K0.']*H0*[eye(n) L0.' K0.'].';

h = {}; h{1} = h0;
H = {}; H{1} = H0;
L = {}; L{1} = L0;
K = {}; K{1} = K0;
P = {}; P{1} = P0;

zbar = zeros(q*(q+1)/2, N);
d_target = zeros(N, 1);

i = 1; % Q-learning iteration
for k = 1:Kend

    if k <= T_learn
        n1k = mvnrnd(zeros(m1,1), P1).';
        n2k = mvnrnd(zeros(m2,1), P2).';
    else
        n1k = zeros(m1, 1);
        n2k = zeros(m2, 1);
    end

%     w(:, k) = 0;
    u(:, k) = L{i}*x(:, k) + n1k;
%     w(:, k) = K{i}*x(:, k) + n2k;
    
    z(:, k) = [x(:, k); u(:, k); w(:, k)];

%     x(:, k+1) = A*x(:, k) + B*u(:, k) + E*w(:, k); 
    x(:, k+1) = dynamics(x(:,k), u(:,k), w(:,k));
    
    z(:, k+1) = [x(:, k+1); L{i}*x(:, k+1); K{i}*x(:, k+1)];

    for n_data = 1:N-1
        d_target(n_data, 1) = d_target(n_data + 1, 1);
        zbar(:, n_data) = zbar(:, n_data + 1);
    end
    d_target(N, 1) = x(:, k).'*R*x(:, k) + (u(:, k).'*u(:, k)) - gamma^2*(w(:, k).'*w(:, k)) + gamma_learn*z(:, k+1).'*H{i}*z(:, k+1);
    zbar(:, N) = v(mat(kron(z(:, k), z(:, k))));

    if k<=K_learn && mod(k, N)==0
        i=i+1;
%         h{i} = (zbar*zbar.')\zbar*d_target;
        h{i} = lsqminnorm(zbar*zbar.', zbar*d_target);
        H{i}= f(h{i});     

        Hxx = H{i}(1:n, 1:n);
        Hxu = H{i}(1:n, n+1:n+m1);
        Hxw = H{i}(1:n, n+m1+1:n+m1+m2);
        Hux = H{i}(n+1:n+m1,1:n);
        Huu = H{i}(n+1:n+m1,n+1:n+m1);
        Huw = H{i}(n+1:n+m1,n+m1+1:n+m1+m2);
        Hwx = H{i}(n+m1+1:n+m1+m2,1:n);
        Hwu = H{i}(n+m1+1:n+m1+m2,n+1:n+m1);
        Hww = H{i}(n+m1+1:n+m1+m2,n+m1+1:n+m1+m2);
        Hwwinv = inv(Hww);
        Huuinv = inv(Huu);
        L{i} = (Huu - Huw*Hwwinv*Hwu)\(Huw*Hwwinv*Hwx - Hux);
        K{i} = (Hww - Hwu*Huuinv*Huw)\(Hwu*Huuinv*Hux - Hwx);
        
        P{i} = [eye(n) L{i}.' K{i}.']*H{i}*[eye(n) L{i}.' K{i}.'].';

        dL(i) = norm(L{i} - L{i-1});
        dK(i) = norm(K{i} - K{i-1});
        dH(i) = norm(H{i} - H{i-1});
    end

end

end

function h = v(H)
[q, ~] = size(H);

h = [];
for i = 1:q
    for j = 1:q
        if i == j
            h = [h; H(i, j)];
        elseif i<j
            h = [h; H(i,j) + H(j,i)];
        end
    end
end

end

function H = f(h)
len = length(h);
r = roots([1 1 -2*len]);
q = r(r>0);

idxs_square = mat(1:q*q).';

idxs_triu = zeros(q);
vstart = 1;
vend = q;
for i = 1:q
    vec_row = vstart:vend;
    
    idxs_triu(i, q-length(vec_row)+1:q) = vstart:vend;
    
    vstart = vend+1;
    vend = vend + q - i;
end
idxs_triu = idxs_triu + triu(idxs_triu, 1).';

T = zeros(q*q, q*(q+1)/2);
for i = 1:q
    for j = 1:q
        if i==j
            T(idxs_square(i,j), idxs_triu(i,j)) = 1;
        else
%             T(idxs_square(i,j), idxs_triu(i,j)) = 0.5;
            T(idxs_square(i,j), idxs_triu(i,j)) = 1;
        end
    end 
end

tallh = T*h;
H = mat(tallh);

end
