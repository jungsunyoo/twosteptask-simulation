    dtQ(1) = Q2(s) - Qmf(current_state_index,a);    % backup with actual choice (i.e., sarsa)                    
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lr1*dtQ(1); % update TD value function
    
    % updating for counterfactual Qmf
    Qmf(current_state_index, a_ct) = Qmf(current_state_index,a_ct) + lr2*(Q2(s_cf)-Qmf(current_state_index,a_ct));
                         
    r = rew(t,s);                                         % sample reward
    r_cf = rew(t, s_cf); % counterfactual reward
    
    dtQ(2) = r - Q2(s);                                   % prediction error (2nd choice)

    Q2(s) = Q2(s) + lr1*dtQ(2);                          % update TD value function
    
    % Update counterfacual TD value
    Q2(s_cf) = Q2(s_cf) + lr2 * (r_cf - Q2(s_cf));
    
    Qmf(current_state_index,a) = Qmf(current_state_index,a) + lambda*lr1*dtQ(2);                     % eligibility trace
     % Jungsun added for counterfactual eligiility trace
    Qmf(current_state_index,a_cf) = Qmf(current_state_index,a_cf) + lambda * lr2 * (r_cf - Q2(s_cf));
   
    % Decaying unchosen states and/or action pairs for this trial
    % don't decay counterfacually observed 
    for s_ = 1:nstates
        if s_ ~= s && s_ ~= s_cf % this part changed 
            Q2(s_) = Q2(s_) * (1-gamma);
        end
    end
    
    for s_ = 1:size(states_total,1)
        if s_~=current_state_index % commented this out to enable counterfactual eligibility trace || a_~=a
            for a_ = 1:2
           
                Qmf(s_,a_) = Qmf(s_,a_) * (1-gamma);
            end
        end
    end