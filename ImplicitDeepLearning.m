classdef ImplicitDeepLearning
    properties
        activation = 'leakyReLU'
        precision = 10^-6;             % precision used in the solver
        lower_precision = 10^-5;   % lower precision parameter used
        lambda = 0                       % dual variable for fenchel
        well_posedness = 'infty'    % well_posedness specification
        L2reg = 10^-3;                  % L2 regularization for the parameters
        fval
        fval_reg
        fval_fenchel_divergence
        rmse
        utils
        %  The following matrices and vectors correspond to the implicit
        %   prediction rule:
        %                 y = Ax+Bu+c; x = max(0,Dx+Eu+f)
        U_train         % input matrix (training)
        Y_train         % output matrix
        X                  % hidden features matrix
        A
        B
        c
        D
        E
        f
        h                   % # of hidden variables
        m                  % # of datapoints
        n                   % # of features for the input
        p                   % # of outputs
        additional_info
        harpagon = 1
        verbose = 1
        radius = 0.5
        initial_learning = 0;
        max_iter = 100;
        dual_step = 1;
    end
    
    methods
        function s = ImplicitDeepLearning(U, Y, h, max_iter, dual_step, radius)
            s.U_train = U;
            s.Y_train = Y;
            s.h = h;
            [s.n,s.m] = size(U);
            [s.p,~] = size(Y);
            s.utils = UtilitiesIDL;
            s.radius = radius;
            s.max_iter = max_iter;
            s.dual_step = dual_step;
            %TODO: include checks of inputs
            if s.harpagon == 1
                s.additional_info = struc('fval_X',[],'diff_X', [], 'fval_hidden_param', [], 'diff_hidden_param', []);
            end
        end
        
        %% Implicit training
        function s = train(s)
            if strcmp(s.activation, 'ReLU')
                if s.initial_learning == 1
                    s = s.initial_train;
                else
                    s=s.parameter_initialization;
                    s.X = s.utils.picard_iterations(s.U_train, s.D, s.E, s.f, s.activation);
                    s.lambda = 10^-3;
                end
                s.lambda = s.dual_variable_update(s.lambda, s.dual_step);
                
                
                dual_update_period = 10;
                s.fval = NaN*ones(s.max_iter, 1);
                s.rmse = NaN*ones(s.max_iter,1);
                for iter =1:s.max_iter
                    % block X
                    grad_X = s.gradient_hidden_var;
                    step_X = s.step_size_X;
                    s.X = max(0, s.X-step_X * grad_X);
                    % block reg
                    [grad_A, grad_B, grad_c] = s.gradient_parameters_reg;
                    step_reg = s.step_size_parameters_reg;
                    s.A = s.A - step_reg * grad_A;
                    s.B = s.B - step_reg * grad_B;
                    s.c = s.c - step_reg * grad_c;
                    % block hidden
                    [grad_D, grad_E, grad_f] =  s.gradient_parameters_hid;
                    step_hid = s.step_size_parameters_hid;
                    s.D = s.well_posedness_projection(s.D - step_hid*grad_D, s.radius);
                    s.E = s.E - step_hid*grad_E;
                    s.f = s.f - step_hid*grad_f;
                    % compute fvals and rmse
                    s.fval(iter) = s.utils.implicit_objective(s.X, s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.lambda);
                    s.fval_reg(iter) = s.utils.RMSE(s.Y_train, s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m));
                    s.rmse(iter) = s.utils.RMSE_actual_implicit(s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.activation);
                    % dual variable update
                    if mod(iter,dual_update_period) == 0
                        s.lambda = s.dual_variable_update(s.lambda, s.dual_step);
                    end
                    if s.verbose == 1 && mod(iter, ceil( s.max_iter / 100 )) == 0
                        disp(['The RMSE at iteration ', num2str(iter), ' is: ',num2str(s.rmse(iter))])
                    end
                end
            elseif strcmp(s.activation, 'leakyReLU')
                s=s.parameter_initialization;
                s.X = s.utils.picard_iterations(s.U_train, s.D, s.E, s.f, s.activation);
                s.rmse = NaN*ones(s.max_iter+1,1);
                s.fval = NaN*ones(s.max_iter, 1);
                num_iter_hidden = 10^2;
                s.additional_info.fval_hidden_param = NaN* zeros(num_iter_hidden+1, s.max_iter);
                s.rmse(1) = s.utils.RMSE_actual_implicit(s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.activation);
                method = 'blo ck';
                for iter = 1:s.max_iter
                    if strcmp(method, 'block')
                        figure(iter)
                        plot(s.U_train, s.Y_train, 'g.')
                        hold on
                        plot(s.U_train,s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m), 'b.')
                        disp('RMSE before reg update')
                        disp(s.utils.RMSE(s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m), s.Y_train))
                        
                        s = s.block_update_regParameters;
                        
                        plot(s.U_train,s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m), 'r.')
                        disp('RMSE after reg update')
                        disp(s.utils.RMSE(s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m), s.Y_train))
                        
                        s.X = s.block_update_X_regParameters;
                        
                        plot(s.U_train,s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m), 'c.')
                        disp('RMSE after X reg update')
                        disp(s.utils.RMSE(s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m), s.Y_train))
                        disp('RMSE of the implicit error before hidden updates')
                        disp(s.utils.L2_implicit_constraint(s.X,s.D*s.X+s.E*s.U_train+s.f*ones(1,s.m),'leakyReLU'))
                        
                        [s, s.additional_info.fval_hidden_param(:,iter)] = s.block_update_HiddenParameters(num_iter_hidden, 'leakyReLU');
                        
                        disp('RMSE of the implicit error after hidden updates')
                        disp(s.utils.L2_implicit_constraint(s.X,s.D*s.X+s.E*s.U_train+s.f*ones(1,s.m),'leakyReLU'))
                        
                        X_prev= s.X;
                        s.X = s.utils.picard_iterations(s.U_train, s.D, s.E, s.f, s.activation);
                        
                        plot(s.U_train,s.A*s.X+s.B*s.U_train+s.c*ones(1,s.m), 'k.')
                        disp('Difference with the real implicit solution')
                        disp(1/sqrt(s.m)*norm(X_prev-s.X,'fro'))
                        s.rmse(iter+1)  = s.utils.RMSE_actual_implicit(s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.activation);
                        hold off
                    else
                        s = s.block_update_regParameters;
                        grad_X = s.gradient_hidden_var;
                        step = s.step_size_X;
                        s.X = s.X - step * grad_X;
                        [grad_D, grad_E, grad_f] = s.gradient_parameters_hid;
                        step_theta_hid = s.step_size_parameters_hid;
                        s.D = s.well_posedness_projection(s.D - step_theta_hid*grad_D, s.radius);
                        s.E = s.E - step_theta_hid*grad_E;
                        s.f = s.f - step_theta_hid*grad_f;
                        s.X = s.utils.picard_iterations(s.U_train, s.D, s.E, s.f, s.activation);
                        s.rmse(iter+1)  = s.utils.RMSE_actual_implicit(s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.activation);
                    end
                end
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
        function [s, info] = block_update_regParameters(s)
            Z=[s.X;s.U_train;ones(1,s.m)];
            Theta=s.Y_train*Z'/(Z*Z'+s.L2reg*eye(s.h+s.n+1));
            s.A=Theta(:,1:s.h); s.B=Theta(:,s.h+1:s.h+s.n); s.c=Theta(:,s.h+s.n+1);
            if nargout == 2
                info = s.utils.MSE_implicit_objective(s.X,s.A,s.B,s.c,s.U_train,s.Y_train);
            end
        end
        
        % Full block update for hidden parameters
        function [s, fvals, diff] = block_update_HiddenParameters(s,num_iter, method)
            if norm(s.lambda) == 0
                lam = ones(s.h, 1);
            else
                lam = s.lambda;
            end
            fvals = NaN*zeros(num_iter+1,1);
            D_prev = s.D; E_prev = s.E; f_prev = s.f;
            iter=1;
            if strcmp(s.activation, 'ReLU')
                fvals(1) = s.utils.scalar_fenchel_divergence(s.X,  s.D* s.X + s.E*s.U_train + s.f*ones(1, s.m) , lam); 
                fval_prev=fvals(1)+1;
                if strcmp(method, 'classic_gradient')
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
                    
                elseif strcmp(method, 'armijo_gradient')
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
                

            elseif strcmp(s.activation, 'leakyReLU')
                fvals(1) = s.utils.L2_implicit_constraint(s.X, s.D * s.X + s.E * s.U_train + s.f *ones(1, s.m), s.activation);
                fval_prev=fvals(1)+1;
                while iter <= num_iter  && fvals(iter) > s.precision && abs(fvals(iter) - fval_prev) > s.precision
                    [grad_D, grad_E, grad_f] = s.gradient_parameters_hid;
                    step_theta_hid = s.step_size_parameters_hid;
                    s.D = s.well_posedness_projection(s.D - step_theta_hid*grad_D, s.radius);
                    s.E = s.E - step_theta_hid*grad_E;
                    s.f = s.f - step_theta_hid*grad_f;
                    iter = iter +1;
                    fvals(iter) = s.utils.L2_implicit_constraint(s.X, s.D * s.X + s.E * s.U_train + s.f *ones(1, s.m), s.activation);
                end
                
            end
            
            if nargout == 3
                diff = sqrt(norm(s.D - D_prev, 'fro')^2 + norm(s.E - E_prev, 'fro')^2 + norm(s.f - f_prev)^2);
            end
        end
        
        %% Gradient steps
        
        % dual update
        function lambda = dual_variable_update(s, lambda, dual_step)
            F_star = s.utils.fenchel_divergence(s.X, s.D*s.X + s.E*s.U_train + s.f*ones(1, s.m));
            v = 1*(F_star>s.lower_precision);
            lambda = lambda + dual_step * v;
        end
        
        %% Gradient computation
        function grad_X = gradient_hidden_var(s)
            if norm(s.lambda)>0
                grad_X = (1/s.m)*( s.A'*(s.A*s.X + s.B*s.U_train + s.c*ones(1 ,s.m)-s.Y_train) + ...
                    (diag(s.lambda) - diag(s.lambda)*s.D - s.D'*diag(s.lambda))*s.X + ...
                    s.D'*diag(s.lambda)*max(0, s.D*s.X + s.E*s.U_train + s.f*ones(1, s.m)) - ...
                    diag(s.lambda)*(s.E*s.U_train + s.f*ones(1,s.m)));
            else
                 grad_X = (1/s.m)*(s.A'*(s.A*s.X + s.B*s.U_train + s.c*ones(1, s.m)-s.Y_train));
            end
        end
        
        function [grad_A, grad_B, grad_c] = gradient_parameters_reg(s)
            Cst = (1/s.m)*(s.A*s.X + s.B*s.U_train + s.c*ones(1,s.m) - s.Y_train);
            grad_A = Cst*s.X';
            grad_B = Cst*s.U_train';
            grad_c = Cst*ones(s.m,1);
        end
        
        function [grad_D, grad_E, grad_f] = gradient_parameters_hid(s)
            if strcmp(s.activation, 'ReLU')
                if norm(s.lambda)>0
                    Cst = diag(s.lambda) * (1/s.m) * (max(0,s.D * s.X + s.E * s.U_train + s.f * ones(1,s.m)) - s.X);
                    grad_D = Cst * s.X';
                    grad_E = Cst * s.U_train';
                    grad_f = Cst * ones(s.m,1);
                else
                    Cst = (1/s.m) * (max(0,s.D * s.X + s.E * s.U_train + s.f * ones(1,s.m)) - s.X);
                    grad_D = Cst * s.X';
                    grad_E = Cst * s.U_train';
                    grad_f = Cst * ones(s.m, 1);
                end
            elseif strcmp(s.activation, 'leakyReLU')
                Cst = (1/s.m) * (s.D * s.X + s.E * s.U_train + s.f * ones(1, s.m)  - s.utils.inverse_leakyReLU(s.X)) ;
                grad_D = Cst * s.X' + s.L2reg * s.D;
                grad_E = Cst * s.U_train' + s.L2reg * s.E; 
                grad_f = Cst * ones(s.m, 1) + s.L2reg * s.f;
            end
        end
        
        %%  Step size computation
        function  out = step_size_parameters_reg(s)
            out = s.m/max([s.m,norm(s.X)^2, norm(s.U_train)^2, norm(s.X*s.U_train')]);
        end
        
        function  out = step_size_parameters_hid(s)
            if strcmp(s.activation, 'ReLU')
                if norm(s.lambda)>0
                    out = s.m/(max(s.lambda)*max([s.m,norm(s.X)^2,norm(s.U_train)^2, norm(s.U_train)*norm(s.X)]));
                else
                    out = s.m/(max([s.m,norm(s.X)^2, norm(s.U_train)^2,norm(s.U_train)*norm(s.X)]));
                end
            elseif strcmp(s.activation, 'leakyReLU')
                L = (1/s.m) * max([s.m,norm(s.X)^2, norm(s.U_train)^2,norm(s.U_train)*norm(s.X)]) + s.L2reg;
                out = 1/L;
            end
        end
        
        function  out = step_size_X(s)
            if norm(s.lambda)>0
                out = s.m/(norm(s.A'*s.A + diag(s.lambda) - diag(s.lambda)*s.D + s.D'*diag(s.lambda)) + max(s.lambda)*norm(s.D)^2);
            else
                out = s.m/(norm(s.A'*s.A));
            end
        end
   
        %% well posed projection
        function D = well_posedness_projection(s,D, radius)
            if strcmp(s.well_posedness, 'infty')
                D = s.utils.infty_norm_projection(D, radius);
            elseif strcmp(s.well_posedness,'LMI')
                D = s.utils.lmi_projection(s.A, D, s.lambda);
            end
        end
        
        %% parameter initialization
        function s = parameter_initialization(s)
            s.A = rand(s.p, s.h) - 0.5;
            s.B  =rand(s.p, s.n) - 0.5;
            s.c = rand(s.p, 1) - 0.5;
            s.D = s.well_posedness_projection(rand(s.h,s.h) - 0.5, s.radius);
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

