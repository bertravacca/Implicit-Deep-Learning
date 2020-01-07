classdef ImplicitDeepLearning
    properties
        activation_type = 'leakyReLU'
        wp_type = 'infty'                % well_posedness specification
        loss_type = 'mse'
        method = 'linear'
        precision = 10^-6;             % precision used in the solver
        tolerance = 10^-4;             % tolerance for fenchel
        lambda = 0                       % dual variable for fenchel
        L2reg = 10^-4;                  % L2 regularization for the parameters
        L1reg = 0;                         % L1 regularization for the parameters
        fval
        score
        utils
        %  The following matrices and vectors correspond to the implicit
        %   prediction rule:
        %                 y = Ax+Bu+c; x = max(0,Dx+Eu+f)
        U                 % input matrix (training)
        Y                 % output matrix (training)
        X                  % hidden features matrix (training)
        A
        B
        c
        D
        E
        f
        h                   % # of hidden variables
        n                   % # of features for the input
        p                   % # of outputs
        additional
        harpagon = 1  % keep extra information
        verbose = 1    % training display parameter 
        max_iter = 100;
        dual_step = 1;
        seed 
        ratio = 0.7
    end
    
    methods
        function s = ImplicitDeepLearning(U, Y, h, max_iter, dual_step)
            s.utils = UtilitiesIDL;
            s.dual_step = dual_step;
            s.h = h;
            s.max_iter = max_iter;
            s.U = struct('trn',[], 'val', []);
            s.Y = struct('trn',[], 'val', []);
            s.X = struct('trn',[], 'val', []);
            empty = NaN*ones(s.max_iter+1,1);
            s.fval = struct('X_trn', empty, 'X_val', empty, 'param_out_trn', empty, 'param_out_val', empty, 'param_hidden_trn', empty, 'param_hidden_val', empty);
            s.score = struct('trn', empty, 'val', empty);
            
            [s.U.trn, s.Y.trn, s.U.val, s.Y.val, s.seed] = s.utils.divide_data(U, Y, s.ratio);
            s.n = size(U, 1);
            s.p = size(Y, 1);
            
            s.additional = struct();
        end
        
        %% Implicit training
        function s = train(s)
            trial = 2;
            if trial == 1
                    m_trn = size(s.U.trn, 2);
                    m_val = size(s.U.val, 2);
                    s = s.parameter_initialization;
                    s.X.trn = s.utils.picard_iterations(s.U.trn, s.D, s.E, s.f, s.activation_type);
                    s.X.val = s.utils.picard_iterations(s.U.val, s.D, s.E, s.f, s.activation_type);
                    [s.A, s.B, s.c] = s.block_update_parameters_mse_loss;
                    disp('RMSE after A,B, c full update')
                    disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                    K = 10 * s.h;
                    D_tilde = rand(K-s.h, s.h) - 0.5;
                    E_tilde = rand(K - s.h, s.n);
                    f_tilde = rand(K - s.h, 1);
                    Z_trn = [s.X.trn; s.utils.activation( D_tilde * s.X.trn + E_tilde * s.U.trn + f_tilde * ones(1, m_trn), s.activation_type)];
                    H_0 = zeros(s.h, K);
                    H_0(:, 1:s.h)=eye(s.h);
                    Y_tmp = (s.Y.trn - s.B * s.U.trn -s.c * ones(1, m_trn));
                    %cvx_begin quiet
                    %variable H(s.h, K)
                    %minimize(norm(Y_tmp-s.A*H*Z_trn, 'fro'))
                    %cvx_end
                    H = ( s.A' * s.A +s.L2reg * eye(s.h) ) \ (s.A' * Y_tmp * Z_trn') / (Z_trn * Z_trn' + s.L2reg * eye(K));
                    X_new = H * Z_trn;
                    disp('New RMSE after Z trick')
                    disp(s.utils.rmse(s.Y.trn, s.A * X_new+ s.B * s.U.trn + s.c * ones(1, m_trn)))
                    figure()
                    plot(s.U.trn, s.Y.trn, 'b.')
                    hold on
                    plot(s.U.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn), 'r.');
                    plot(s.U.trn, s.A * X_new + s.B * s.U.trn + s.c * ones(1, m_trn), 'g.');
                    disp('L2 actual error of the implicit')
                    disp((1/sqrt(m_trn))*norm(s.X.trn - X_new, 'fro'))
                    s.X.trn = X_new;
                    disp('max of X')
                    disp(max(max(X_new)))
                    [s.D, s.E, s.f, fvals] = block_update_HiddenParameters(s,1000);
                    s.X.trn = s.utils.picard_iterations(s.U.trn, s.D, s.E, s.f, s.activation_type);
                    plot(s.U.trn, s.A *s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn), 'm.');
                    
                    disp('New RMSE')
                    disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                    figure()
                    semilogx(fvals)
                    disp('L2 implicit error')
                    disp(fvals(100))
                    disp('L2 actual error')
                    disp((1/sqrt(m_trn))*norm(s.X.trn - X_new, 'fro'))
                    disp('Max difference')
                    disp(max(max(s.X.trn - X_new)))
                    
                    [s.A, s.B, s.c] = s.block_update_parameters_mse_loss;
                    disp('RMSE after A,B, c full update')
                    disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                    
            elseif trial == 2
                m_trn = size(s.U.trn, 2);
                s = s.parameter_initialization;
                s.X.trn = s.utils.picard_iterations(s.U.trn, s.D, s.E, s.f, s.activation_type);
                s.X.val = s.utils.picard_iterations(s.U.val, s.D, s.E, s.f, s.activation_type);
                [s.A, s.B, s.c] = s.block_update_parameters_mse_loss;
                
                disp('RMSE after A,B, c full update')
                disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                figure()
                [U_sort, I] = sort(s.U.trn);
                plot(U_sort, s.Y.trn(I), 'b-')
                hold on
                Z = s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn);
                Z = Z(I);
                plot(U_sort, Z, 'r-');
                
                rho = 10^3;
                s.X.trn = (s.A'*s.A + rho*eye(s.h)) \ (s.A' * (s.Y.trn - s.B*s.U.trn - s.c*ones(1,m_trn)) + rho * s.X.trn);
                
                
                disp('RMSE after X brute force update')
                disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                
                % Z = s.utils.inverse_activation(s.X.trn, s.activation_type);
                Z = s.utils.inverse_activation(s.X.trn, s.activation_type)';
                % V = [s.X.trn; s.U.trn; ones(1, m_trn)];
                V = [s.X.trn', s.U.trn', ones(m_trn, 1)];
                % Theta = Z * V' / (V * V' + 10^-9 * eye(s.n+s.h+1));
                Theta = V' * inv(V * V'+ 10^-12 *eye(m_trn)) * Z;
                Theta = Theta';
                s.D = Theta(:,1:s.h);
                s.E = Theta(:, s.h + 1:s.h+s.n);
                s.f = Theta(:, s.h + s.n+1);
                disp('Infty Norm of D')
                disp(norm(s.D, 'Inf'))
                
                disp('Linear error')
                disp(norm(s.D *  s.X.trn + s.E * s.U.trn + s.f * ones(1, m_trn) -  s.utils.inverse_activation(s.X.trn, s.activation_type), 'fro'))
                
                disp('Operator Norm of D')
                disp(norm(s.D))
                
                X_new =  s.utils.picard_iterations(s.U.trn, s.D, s.E, s.f, s.activation_type);
                disp('The error on implicit is')
                disp(norm(X_new - s.X.trn, 'fro'))
                
                %s.X.trn = X_new;
                disp('RMSE')
                disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                
            elseif trial == 3
                m_trn = size(s.U.trn, 2);
                s = s.parameter_initialization;
                s.X.trn = s.utils.picard_iterations(s.U.trn, s.D, s.E, s.f, s.activation_type);
                s.X.val = s.utils.picard_iterations(s.U.val, s.D, s.E, s.f, s.activation_type);
                [s.A, s.B, s.c] = s.block_update_parameters_mse_loss;
                
                disp('RMSE after A,B, c full update')
                disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                figure()
                [U_sort, I] = sort(s.U.trn);
                plot(U_sort, s.Y.trn(I), 'b-')
                hold on
                Z = s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn);
                Z = Z(I);
                plot(U_sort, Z, 'r-');
                rho = 10^4;
                s.X.trn = (s.A'*s.A + rho*eye(s.h)) \ (s.A' * (s.Y.trn - s.B*s.U.trn - s.c*ones(1,m_trn)) + rho * s.X.trn);
                
                disp('RMSE after X brute force update')
                disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
                
                disp('Infty Norm of D')
                disp(norm(s.D, 'Inf'))
                
                 [s.D, s.E, s.f] = s.block_update_HiddenParameters(1000);
                
                
                disp('Infty Norm of D')
                disp(norm(s.D, 'inf'))
                
                disp('Linear error')
                disp(norm(s.D * s.X.trn + s.E * s.U.trn + s.f * ones(1, m_trn) - s.utils.inverse_activation(s.X.trn, s.activation_type), 'fro'))
                
                disp('Operator Norm of D')
                disp(norm(s.D))
                
                X_new = s.utils.picard_iterations(s.U.trn, s.D, s.E, s.f, s.activation_type);
                disp('The error on implicit is')
                disp(norm(X_new - s.X.trn, 'fro'))
                
                s.X.trn = X_new;
                disp('RMSE')
                disp(s.utils.rmse(s.Y.trn, s.A * s.X.trn + s.B * s.U.trn + s.c * ones(1, m_trn)))
            end
        end

 
        %% Initial training
        function s = initial_train(s)
            s = s.parameter_initialization;
            num_max_iter_bcd=5;
            s.fval_reg = NaN*zeros(100,1);
            s.rmse = NaN*zeros(100,1);
            
            % initial implicit problem (lambda=0) start with (A,B,c,X)...
            num_iter_X = 500;
            num_iter_hidden_param=10^4;
            s.X = s.utils.picard_iterations(s.U_train, s.D, s.E, s.f, s.activation);
            
            % initial rmse
            if s.verbose == 1
                val = s.utils.RMSE_actual_implicit(s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.activation);
                disp(['Initialization started, the initial RMSE is: ', num2str(val)])
            end
            if s.harpagon ==1
                s.additional_info.fval_X = NaN*zeros(num_iter_X+1, num_max_iter_bcd);
                s.additional_info.diff_X = NaN*zeros(num_max_iter_bcd,1);
                iter_bcd = 1;
                s.additional_info.diff_X(1)=1;
                while iter_bcd<num_max_iter_bcd && s.additional_info.diff_X(iter_bcd)>s.lower_precision
                    % updates
                    s = s.block_update_regParameters;
                    [s, s.additional_info.fval_X(:,iter_bcd), s.additional_info.diff_X(iter_bcd)] = s.block_update_X(num_iter_X);
                    [s, s.additional_info.fval_hidden_param(:, iter_bcd), s.additional_info.diff_hidden_param(iter_bcd)] = s.block_update_HiddenParameters( num_iter_hidden_param, 'armijo_gradient');
                    s.fval_reg(iter_bcd) = s.utils.MSE_implicit_objective(s.X, s.A, s.B, s.c, s.U_train, s.Y_train);
                    s.rmse(iter_bcd) = s.utils.RMSE_actual_implicit(s.A,s.B,s.c,s.D,s.E,s.f,s.U_train,s.Y_train, s.activation);
                    iter_bcd= iter_bcd + 1;
                end
            end
            
            if s.verbose == 1
                disp(['The number of bcd iterations used for intitialization was: ', num2str(iter_bcd-1)])
                val = s.utils.RMSE_actual_implicit(s.A,s.B,s.c,s.D,s.E,s.f,s.U_train,s.Y_train, s.activation);
                disp(['Initialization finished, the RMSE is: ', num2str(val)])
            end
        end
        
        %% Algorithms
        
        % Full block update for X
        function  [s, fvals, diff] = block_update_X(s,num_iter)
            fvals = NaN*zeros(num_iter+1,1);
            fvals(1) = s.utils.implicit_objective(s.X, s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.lambda); fval_prev=fvals(1)+1;
            X_prev = s.X;
            iter=1;
            while iter <= num_iter  && fvals(iter) > s.precision && abs(fvals(iter) - fval_prev) > s.precision
                fval_prev = fvals(iter);
                grad_X = s.gradient_hidden_var;
                step_X = s.step_size_X;
                s.X = max(0, s.X - step_X*grad_X);
                iter = iter+1;
                fvals(iter) = s.utils.implicit_objective(s.X, s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.lambda);
            end
            diff = (1/s.m)*norm(s.X - X_prev, 'fro');
        end
        
        % Full block update for X considering loss only (i.e. not considering implicit constraint)
        function X = block_update_X_regParameters(s,num_iter)
            if strcmp(s.activation, 'ReLU')
                for k = 1:num_iter
                    grad_X = s.gradient_hidden_var;
                    step_X = s.step_size_X;
                    X = max(0, s.X - step_X*grad_X);
                end
            elseif strcmp(s.activation, 'leakyReLU')
                X = (s.A'*s.A + s.L2reg*eye(s.h)) \ s.A' * (s.Y_train - s.B*s.U_train - s.c*ones(1,s.m));
            end
        end
        
        % Full block update for the regression parameters
        function [A, B, c] = block_update_parameters_mse_loss(s)
                m = size(s.U.trn, 2);
                Z = [ s.X.trn; s.U.trn; ones(1,m) ];
                Theta = s.Y.trn * Z' / ( Z*Z' + s.L2reg * eye(s.h+s.n+1) );
                A = Theta(:,1:s.h);
                B = Theta(:, s.h + 1:s.h+s.n);
                c = Theta(:, s.h + s.n+1);
        end
        
        % Full block update for hidden parameters
        function [D, E, f, fvals, diff] = block_update_HiddenParameters(s,num_iter)
            m = size(s.U.trn, 2);
            if nargin == 2
                fvals = NaN*zeros(num_iter+1,1);
            end
            D_prev = s.D; E_prev = s.E; f_prev = s.f;
            iter=1;
            
            if strcmp(s.activation_type, 'ReLU')
                if norm(s.lambda) == 0
                    lam = ones(s.h, 1);
                else
                    lam = s.lambda;
                end
                fvals(1) = s.utils.scalar_fenchel_divergence(s.X,  s.D* s.X + s.E*s.U_train + s.f*ones(1, s.m) , lam); 
                fval_prev=fvals(1)+1;
                if strcmp(s.method, 'classic_gradient')
                    while iter <= num_iter && fvals(iter) > s.precision && abs(fvals(iter) - fval_prev) > s.precision
                        fval_prev = fvals(iter);
                        [grad_D, grad_E, grad_f] =  s.gradient_parameters_hid;
                        step_theta_hid = s.step_size_parameters_hid;
                        s.D = s.well_posedness_projection(s.D - step_theta_hid*grad_D, s.radius);
                        s.E = s.E - step_theta_hid*grad_E;
                        s.f = s.f - step_theta_hid*grad_f;
                        iter = iter+1;
                        fvals(iter) = s.utils.scalar_fenchel_divergence(s.X, s.D* s.X + s.E*s.U_train + s.f*ones(1, s.m), lam);
                    end
                    
                elseif strcmp(s.method, 'armijo_gradient')
                    % initialize the step size
                    step = 100*s.step_size_parameters_hid;
                    c_armijo = 10^-4;
                    step_divide = 2;
                    while iter <= num_iter && fvals(iter) > s.precision && abs(fvals(iter) - fval_prev) > s.precision
                        fval_prev = fvals(iter);
                        armijo_condition = false;
                        [grad_D, grad_E, grad_f] =  s.gradient_parameters_hid;
                        norm_gradient_square = norm(grad_D, 'fro')^2 + norm(grad_E, 'fro')^2 + norm(grad_f)^2;
                        step_new = step;
                        while armijo_condition == false
                            D_new = s.well_posedness_projection(s.D - step_new*grad_D, s.radius);
                            E_new = s.E - step_new*grad_E;
                            f_new = s.f - step_new*grad_f;
                            fval_new = s.utils.scalar_fenchel_divergence(s.X, D_new* s.X + E_new*s.U_train + f_new*ones(1, s.m), lam);
                            armijo_condition = (fval_new < fval_prev - c_armijo * step_new * norm_gradient_square) ;
                            step_new = step_new/step_divide;
                        end
                        step = step_new * step_divide^2;
                        s.D = D_new;
                        s.E = E_new;
                        s.f = f_new;
                        iter = iter+1;
                        fvals(iter) = fval_new;
                    end
                end
                

            elseif strcmp(s.activation_type, 'leakyReLU')
                tryout = 3;
                if tryout == 1
                    D = s.D; E= s.E; f = s.f;
                    fvals(1) = s.utils.L2_implicit_constraint_error(s.X.trn, D * s.X.trn + E * s.U.trn + f *ones(1, m), s.activation_type);
                    fval_prev=fvals(1)+1;
                    while iter <= num_iter  
                        [D, E, f] = s.utils.gradient_update_hidden_parameters_linear(s.U.trn, s.X.trn, D, E, f, s.activation_type, s.wp_type, s.L2reg, s.L1reg);
                        iter = iter +1;
                        fvals(iter) = s.utils.L2_implicit_constraint_error(s.X.trn, D * s.X.trn + E * s.U.trn + s.f *ones(1, m), s.activation_type);
                    end
                    
                elseif tryout == 2
                    D = NaN * zeros(s.h, s.h);
                    E = NaN * zeros(s.h, s.n);
                    f = NaN * zeros(s.h, 1);
                    Z =  s.utils.inverse_activation(s.X.trn, s.activation_type);
                    disp('Linear error before ')
                    disp(norm(s.D * s.X.trn  + s.E* s.U.trn + s.f *ones(1, m) - Z,'fro'))
                    for i =1 : s.h
                        [w, v, history] = s.utils.extended_basis_pursuit(s.X.trn', [s.U.trn', ones(m,1)], Z(i,:)' , 1, 1);
                        D(i, :) = w';
                        E(i, :) = v(1:s.n)';
                        f(i) = v(s.n + 1);
                        if i ==1
                            figure()
                        end
                        plot(history.linear_error, 'b')
                        hold on
                    end
                    disp('Linear error after')
                    disp(norm(D * s.X.trn  + E* s.U.trn + f *ones(1, m) - Z,'fro'))
                    
                elseif tryout == 3
                    disp('hello')
                    D = NaN * zeros(s.h, s.h);
                    E = NaN * zeros(s.h, s.n);
                    f = NaN * zeros(s.h, 1);
                    Z =  s.utils.inverse_activation(s.X.trn, s.activation_type);
                    disp('Linear error before ')
                    disp(norm(s.D * s.X.trn  + s.E* s.U.trn + s.f *ones(1, m) - Z,'fro'))
                    cvx_begin 
                    variable D(s.h, s.h)
                    variable E(s.h, s.n)
                    variable f(s.h,1)
                    minimize(norm(D, 'fro'))
                    D * s.X.trn + E * s.U.trn + f *ones(1, m) == Z
                    cvx_end
                    
                    disp('Linear error after')
                    disp(norm(D * s.X.trn  + E* s.U.trn + f *ones(1, m) - Z,'fro'))

                end
                
            end
            
            if nargout == 5
                diff = sqrt(norm(s.D - D_prev, 'fro')^2 + norm(s.E - E_prev, 'fro')^2 + norm(s.f - f_prev)^2);
            end
        end
      
  
        %% parameter initialization
        function s = parameter_initialization(s)
            s.A = rand(s.p, s.h) - 0.5;
            s.B  =rand(s.p, s.n) - 0.5;
            s.c = rand(s.p, 1) - 0.5;
            if strcmp(s.wp_type, 'infty')
                s.D = s.utils.infty_norm_projection(rand(s.h,s.h) - 0.5, 0.5);
            end
            s.E = rand(s.h, s.n) - 0.5;
            s.f = rand(s.h, 1) - 0.5;
        end
        
        %% Implicit prediction rule
        function Y_prediction = implicit_prediction(s, U)
            m_pred = size(U,2);
            X_pred = s.utils.picard_iterations(U, s.D, s.E, s.f, s.activation);
            Y_prediction = s.A * X_pred +s.B * U + s.c * ones(1,m_pred);
        end
        
        %% Visualization
        function visualize_algo_init(s)
            s.utils.visualize_algo_init(s.additional_info.fval_X, s.additional_info.fval_hidden_param)
        end
        
        function visualize_algo(s)
            s.utils.visualize_algo(s.fval, s.fval_reg, s.rmse);
        end
    end
end 

