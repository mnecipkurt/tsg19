% Online Cyber-Attack Detection using Model-free RL for POMDPs
% Testing Phase: Algorithm 2 and The Benchmark Tests
% Performance Metrics: Average Detection Delay (ADD), Probability of False
% Alarm (PFA), and Miss Detection Ratio (MDR) for attacks happening at 
% tau = geornd(rho), where rho is uniform random variable: U[0.0001,0.001]
 
clear variables;
clc;

K = 23; N = 13;
sigvsq = 1e-4; sigwsq = 2e-4;
load H;
load A;
load new_angles;

betas = [0 0.95e-2 1.05e-2 1.15e-2];  % quantization thresholds

load Q_c0p2_new_sarsa;
Q02 = Q;
load Q_c0p02_new_sarsa;
Q002 = Q;

load ind_mat_v2;

ntrials = 10000;   % number of trials

settle_time = 20;   % wait some time for Kalman filter to converge
 
cnt_falarm02 = 0;     % number of false alarm events over all trials
cnt_detected02 = 0;   % number of detection events (given a maximum detection delay constraint)
sum_delay02 = 0;      % sum of all delays for all delay events

cnt_falarm002 = 0;     
cnt_detected002 = 0;
sum_delay002 = 0;   

euc_thresh_all = [0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32];
euc_sum_delay = zeros(1,length(euc_thresh_all));
euc_cnt_add = zeros(1,length(euc_thresh_all));
euc_cnt_falarm = zeros(1,length(euc_thresh_all));
euc_detected = zeros(1,length(euc_thresh_all));

cos_thresh_all = [5.4e-5 5.8e-5 6.2e-5 6.6e-5 7e-5 7.4e-5 7.8e-5 8.2e-5 8.6e-5 8.8e-5];
cos_sum_delay = zeros(1,length(cos_thresh_all));
cos_cnt_add = zeros(1,length(cos_thresh_all));
cos_cnt_falarm = zeros(1,length(cos_thresh_all));
cos_detected = zeros(1,length(cos_thresh_all));


for i = 1:ntrials    
    i
    t = 1;
    a02 = 2; % initial action
    a002 = 2; % initial action
    
    P = eye(N);
    G = zeros(N,K);
    x = new_angles; % initial state parameters obtained in Matpower
    hx0_m = new_angles; % initialization of the state estimates
    
    rho = 1e-4 + 9e-4*rand(1);  % rho is uniform r.v. U[0.0001,0.001]
    attack_time = settle_time+1 + geornd(rho);  % attack launch time is geometric r.v. with parameter rho
    delay_constraint = 10;  % maximum tolerable detection delay (if not detected, then missed)
    max_time = attack_time + delay_constraint;   % max. duration of an episode (if "stop" is chosen, then episode is terminated)
    
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
    flag_falarm02 = 0; flag_delay02 = 0; 
    flag_falarm002 = 0; flag_delay002 = 0; 
    euc_tmpadd = zeros(1,length(euc_thresh_all));
    euc_flag_det_delay = zeros(1,length(euc_thresh_all));
    euc_flag_falarm = zeros(1,length(euc_thresh_all));    
    cos_tmpadd = zeros(1,length(cos_thresh_all));
    cos_flag_det_delay = zeros(1,length(cos_thresh_all));
    cos_flag_falarm = zeros(1,length(cos_thresh_all));
    
    while (settle_time < t && t <= max_time && terminal == 0)
        
        % State update at time t
        v = mvnrnd(zeros(N,1),sigvsq*eye(N))';
        x = A*x + v;
        % Collect the measurements at time t
        w = mvnrnd(zeros(K,1),sigwsq*eye(K))';
        %ddd = randn(1); sgn = 1*(ddd>0) - 1*(ddd<=0);
        %fdata = sgn*(0.02 + 0.04*rand(K,1)); % false data
        %fdata = zeros(K,1);
        %fdata = -0.07 + 0.14*rand(K,1);
        %fdata = -0.05 + 0.1*rand(K,1);
        fdata = zeros(K,1);
        %fdata = zeros(K,1);
        %ss = (randn(1) > 0)*2e-4; % with prob. 0.5, jamming occurs
        %ss = 1e-3;
        %ss = 0;
        %ss = 5e-4;
        ss = sqrt(8e-5);
        %ss = 0;
        %sigma = (ss+ss*rand(K,1));    % jamming noise variance
        tmp_mtrx = ss*randn(K);
        tmp_cov = tmp_mtrx*tmp_mtrx';
        jamm_noise = mvnrnd(zeros(K,1),tmp_cov)';
        %noise = mvnrnd(zeros(K,1),diag(sigma))'; % correlated (over space) jamming noise
        y = H*x + w + (fdata+jamm_noise)*(t >= attack_time);
%         if (t >= attack_time)
%             y = (randn(K,1) > norminv(0.2)).*y; % random DoS attack where each meter meas. becomes unavailable with prob. 0.1
%         end
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
        o = ind_mat(index);
        
        [tmp,a02] = min(Q02(o,:));      % greedy action based on the (learned) Q-table
        [tmpp,a002] = min(Q002(o,:));   % greedy action based on the (learned) Q-table

        if ((t < attack_time) && (a02 == 1))
            cnt_falarm02 = cnt_falarm02 + (flag_falarm02 == 0);
            flag_falarm02 = flag_falarm02 + (flag_falarm02 == 0);
        elseif ((t >= attack_time) && (a02 == 1))
            cnt_detected02 = cnt_detected02 + (flag_falarm02 == 0)*(flag_delay02 == 0); 
            sum_delay02 = sum_delay02 + (t - attack_time)*(flag_falarm02 == 0)*(flag_delay02 == 0); 
            flag_delay02 = flag_delay02 + (flag_falarm02 == 0)*(flag_delay02 == 0);
        end
        
        if ((t < attack_time) && (a002 == 1))
            cnt_falarm002 = cnt_falarm002 + (flag_falarm002 == 0);
            flag_falarm002 = flag_falarm002 + (flag_falarm002 == 0);
        elseif ((t >= attack_time) && (a002 == 1))
            cnt_detected002 = cnt_detected002 + (flag_falarm002 == 0)*(flag_delay002 == 0);
            sum_delay002 = sum_delay002 + (t - attack_time)*(flag_falarm002 == 0)*(flag_delay002 == 0);  
            flag_delay002 = flag_delay002 + (flag_falarm002 == 0)*(flag_delay002 == 0);
        end
        
        euc_stat = norm((y-H*hx0_p),2);                
        euc_flag_falarm = euc_flag_falarm + (euc_flag_falarm == 0).*(euc_stat >= euc_thresh_all)*(t < attack_time);
        euc_tmpadd = euc_tmpadd + (t-attack_time)*(euc_flag_falarm == 0).*(euc_flag_det_delay == 0).*(euc_stat >= euc_thresh_all)*(t >= attack_time);
        euc_flag_det_delay = euc_flag_det_delay + (euc_flag_falarm == 0).*(euc_flag_det_delay == 0).*(euc_stat >= euc_thresh_all)*(t >= attack_time); 
        
        cos_stat = 1 - (y'*(H*hx0_p)) / ( norm(y,2)*norm((H*hx0_p),2) ); 
        cos_flag_falarm = cos_flag_falarm + (cos_flag_falarm == 0).*(cos_stat >= cos_thresh_all)*(t < attack_time);
        cos_tmpadd = cos_tmpadd + (t-attack_time)*(cos_flag_falarm == 0).*(cos_flag_det_delay == 0).*(cos_stat >= cos_thresh_all)*(t >= attack_time);
        cos_flag_det_delay = cos_flag_det_delay + (cos_flag_falarm == 0).*(cos_flag_det_delay == 0).*(cos_stat >= cos_thresh_all)*(t >= attack_time); 
        
        cond02 = flag_falarm02 || flag_delay02;
        cond002 = flag_falarm002 || flag_delay002;        
        euc_cond = euc_flag_falarm(length(euc_flag_falarm)) || euc_flag_det_delay(length(euc_flag_det_delay));
        cos_cond = cos_flag_falarm(length(cos_flag_falarm)) || cos_flag_det_delay(length(cos_flag_det_delay));
        
        if (cond02 && cond002 && euc_cond && cos_cond)
            terminal = 1;            
        end
        
        % Time update
        t = t + 1;             
    end
    
    euc_cnt_falarm = euc_cnt_falarm + euc_flag_falarm;  
    euc_detected = euc_detected + euc_flag_det_delay;
    if ( euc_flag_det_delay( length(euc_thresh_all) ) == 1 )
        euc_sum_delay = euc_sum_delay + euc_tmpadd;
        euc_cnt_add = euc_cnt_add + euc_flag_det_delay;
    end
    
    cos_cnt_falarm = cos_cnt_falarm + cos_flag_falarm;
    cos_detected = cos_detected + cos_flag_det_delay;
    if ( cos_flag_det_delay( length(cos_thresh_all) ) == 1 )
        cos_sum_delay = cos_sum_delay + cos_tmpadd;
        cos_cnt_add = cos_cnt_add + cos_flag_det_delay;
    end
         
end

add02 = sum_delay02/cnt_detected02; % average detection delay
pfa02 = cnt_falarm02/ntrials; % probability of false alarm
mdr02 = (ntrials - cnt_detected02 - cnt_falarm02)/ntrials; % miss detection ratio
precision02 = cnt_detected02/(cnt_detected02 + cnt_falarm02);
recall02 = cnt_detected02/(ntrials - cnt_falarm02);
F02 = (2*precision02*recall02)/(precision02 + recall02);

add002 = sum_delay002/cnt_detected002;
pfa002 = cnt_falarm002/ntrials;
mdr002 = (ntrials - cnt_detected002 - cnt_falarm002)/ntrials;
precision002 = cnt_detected002/(cnt_detected002 + cnt_falarm002);
recall002 = cnt_detected002/(ntrials - cnt_falarm002);
F002 = (2*precision002*recall002)/(precision002 + recall002);

euc_pfa = euc_cnt_falarm/ntrials;   % probability of false alarm for all thresholds
euc_cnt_add(euc_cnt_add==0) = 1;    
euc_add = euc_sum_delay./euc_cnt_add;   % average detection delay
euc_mdr = (ntrials - euc_cnt_falarm - euc_detected)/ntrials; % miss detection ratio
euc_precision = euc_detected./(euc_detected + euc_cnt_falarm);
euc_recall = euc_detected./(ntrials - euc_cnt_falarm);
euc_F = (2*euc_precision.*euc_recall)./(euc_precision + euc_recall);

cos_pfa = cos_cnt_falarm/ntrials;
cos_cnt_add(cos_cnt_add==0) = 1;
cos_add = cos_sum_delay./cos_cnt_add;
cos_mdr = (ntrials - cos_cnt_falarm - cos_detected)/ntrials;
cos_precision = cos_detected./(cos_detected + cos_cnt_falarm);
cos_recall = cos_detected./(ntrials - cos_cnt_falarm);
cos_F = (2*cos_precision.*cos_recall)./(cos_precision + cos_recall);

save('perf_corr_jamm_v4_all_maxdelay10');



