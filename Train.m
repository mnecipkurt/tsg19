% Online Cyber-Attack Detection using Model-free RL for POMDPs
% Learning Phase (Algorithm 1)

% System (Smart Grid) Parameters
K = 23; N = 13;
sigvsq = 1e-4; sigwsq = 2e-4;
load H;
load A;
load new_angles;

I=4;  % number of quantization intervals
betas = [0 0.95e-2 1.05e-2 1.15e-2];  % quantization thresholds
M = 4;  % size of the finite history (memory) window

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 0.2; % relative cost, tradeoff parameter for ADD vs PFA (or FAP)

Q = zeros(I^M,2);   % Q-table 

no_episodes = 4e5;   % number of training episodes (need to be chosen higher for better learning)

max_time = 200;     % max. duration of an episode (if "stop" is chosen, then episode is terminated)
settle_time = 20;
attack_time = settle_time;  % choose the attack launch time
% attack_time = max_time - 100;  % choose the attack launch time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = 0.1;
eps = 0.1;
level_eps = norminv(1-eps);

load ind_mat_v2;

tic;
for i = 1:no_episodes
    i
    t = 1;
    a = 2; % initial action (stop: a=1, continue: a=2)
                                                                          
    P = eye(N);
    G = zeros(N,K);
    x = new_angles; % initial state parameters obtained in Matpower
    hx0_m = new_angles; % initialization of the state estimates
    
    % For Kalman filter convergence (during normal operating conditions)
    while (t <= settle_time)       
        % State update at time t
        v = mvnrnd(zeros(N,1),sigvsq*eye(N))';
        x = A*x + v;
        % Collect the measurements at time t
        w = mvnrnd(zeros(K,1),sigwsq*eye(K))';
        y = H*x + w;
        % Prediction Step of Kalman Filter
        hx0_p = A*hx0_m;
        P = A*P*A' + sigvsq*eye(N);
        % Measurement Update Step of Kalman Filter
        G = P*H'/(H*P*H'+(sigwsq)*eye(K));
        hx0_m = hx0_p + G*(y-H*hx0_p);
        P = P - G*H*P;  
        % Compute quantized measurements
        eta_t = (y - H*hx0_m)'*(y - H*hx0_m); % negative log-scaled likelihood
        if (eta_t >= betas(4))
            recent = 4;
        elseif ((eta_t >= betas(3)) && (eta_t < betas(4)))
            recent = 3;
        elseif ((eta_t >= betas(2)) && (eta_t < betas(3)))
            recent = 2;
        elseif ((eta_t >= betas(1)) && (eta_t < betas(2)))
            recent = 1;
        end
        if (t == settle_time)
            quant_now = recent;     % (initial) most recent sliding window of size 4
        elseif (t == settle_time-1)
            quant_1 = recent;
        elseif (t == settle_time-2)
            quant_2 = recent;
        elseif (t == settle_time-3)
            quant_3 = recent;   
        end
        % Time update
        t = t+1;
    end
    
    tmp3 = quant_3; tmp2 = quant_2; tmp1 = quant_1; tmp_now = quant_now;
    rslt = strcat(num2str(tmp3),num2str(tmp2),num2str(tmp1),num2str(tmp_now));
    index = str2double(rslt);    
    o = ind_mat(index); % initial o

    terminal = 0;
    
    while (settle_time < t && t < max_time && terminal == 0)
        
        % State update at time t
        v = mvnrnd(zeros(N,1),sigvsq*eye(N))';
        x = A*x + v;
        % Collect the measurements at time t
        w = mvnrnd(zeros(K,1),sigwsq*eye(K))';
        % Attack parameters: train the agent so that it can detect low-magnitude hybrid attacks (choose desired minimum levels for detection purposes)
        ddd = randn(1); sgn = 1*(ddd>0) - 1*(ddd<=0);
        fdata = sgn*(0.02 + 0.04*rand(K,1)); % false data
        ss = (randn(1) > 0)*2e-4; % with prob. 0.5, jamming occurs
        sigma = (ss+ss*rand(K,1));    % jamming noise variance
%         tmp_mat = randn(K,K);      
%         rand_mat = abs(tmp_mat*tmp_mat');   % correlation terms 
%         rand_mat = diag(sigma) + rand_mat - diag(diag(rand_mat));  % variance terms
        noise = mvnrnd(zeros(K,1),diag(sigma))'; % correlated (over space) jamming noise
        y = H*x + w + (fdata+noise)*(t >= attack_time);
        % Prediction Step of Kalman Filter
        hx0_p = A*hx0_m;
        P = A*P*A' + sigvsq*eye(N);
        % Measurement Update Step of Kalman Filter
        G = P*H'/(H*P*H'+(sigwsq)*eye(K));
        hx0_m = hx0_p + G*(y-H*hx0_p);
        P = P - G*H*P;    
        
        % Compute quantized measurements
        eta_t = (y - H*hx0_m)'*(y - H*hx0_m);
        if (eta_t >= betas(4))
            recent = 4;
        elseif ((eta_t >= betas(3)) && (eta_t < betas(4)))
            recent = 3;
        elseif ((eta_t >= betas(2)) && (eta_t < betas(3)))
            recent = 2;
        elseif ((eta_t >= betas(1)) && (eta_t < betas(2)))
            recent = 1;
        end        
        % Update on the sliding window
        quant_3 = quant_2;
        quant_2 = quant_1; 
        quant_1 = quant_now; 
        quant_now = recent;  
        % Determine the new observation
        tmp3 = quant_3; tmp2 = quant_2; tmp1 = quant_1; tmp_now = quant_now;
        rslt = strcat(num2str(tmp3),num2str(tmp2),num2str(tmp1),num2str(tmp_now));
        index = str2double(rslt);        
        o_new = ind_mat(index);

        if ((t < attack_time) && (a == 1))
            r = 1;
        elseif ((t >= attack_time) && (a == 1))
            r = 0;
        elseif ((t >= attack_time) && (a == 2))
            r = c;
        elseif ((t < attack_time) && (a == 2))
            r = 0;
        end
        
        % Determine the next action based on the new observation and the epsilon-greedy strategy
        if (randn(1) < level_eps) 
            [tmp,a_new] = min(Q(o_new,:));
        else
            a_new = randi(2);
        end
        
        if (a == 1)
            terminal = 1;
            delta = r - Q(o,a);
        elseif (a == 2)           
            delta = r + Q(o_new,a_new) - Q(o,a);
        end
        
        Q(o,a) = Q(o,a) + alpha*delta;
        
        % Updates for the next time
        o = o_new;
        a = a_new;
        t = t + 1;            
    end
        
end
toc;
elapsed_training_time = toc;

save('Q_c0p2_new_sarsa','Q');




