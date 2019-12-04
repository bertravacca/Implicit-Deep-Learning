classdef ImplicitDeepLearning
    properties
        precision = 10^-6;             % precision used in the solver
        lower_precision = 10^-4;   % lower precision parameter used
        lambda = 0                       % dual variable for fenchel  
        well_posedness = 'infty'    % well_posedness specification
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
    end
    
    methods
        function s = ImplicitDeepLearning(U, Y, h)
            s.U_train = U;
            s.Y_train = Y;
            s.h = h;
            [s.n,s.m] = size(U);
            [s.p,~] = size(Y);
            s.utils = UtilitiesIDL;
            %TODO: include checks of inputs
            if s.harpagon == 1
                s.additional_info = struc('fval_X',[],'diff_X', [], 'fval_hidden_param', [], 'diff_hidden_param', []);
            end
        end
        
        %% Implicit training
        function s=train(s)

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
            s.X = s.utils.picard_iterations(s.U_train, s.D, s.E, s.f);
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
                    s.rmse(iter_bcd) = s.utils.RMSE_actual_implicit(s.A,s.B,s.c,s.D,s.E,s.f,s.U_train,s.Y_train);
                    iter_bcd= iter_bcd + 1;
                end
            end
            
            if s.verbose == 1
                disp(['The number of bcd iterations used for intitialization was: ', num2str(iter_bcd-1)])
            end
        end
        
        %% Algorithms
        
        % Block update for X
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
        
        % Block update for X considering loss only (i.e. not considering implicit constraint)
        function s = block_update_X_regParameters(s,num_iter)
            for k = 1:num_iter
                [grad_A, grad_B, grad_c] = s.gradient_parameters_reg;
                grad_X = s.gradient_hidden_var;
                step_theta_reg = s.step_size_parameters_reg;
                step_X = s.step_size_X;
                s.A = s.A-step_theta_reg*grad_A;
                s.B = s.B - step_theta_reg*grad_B;
                s.c = s.c - step_theta_reg*grad_c;
                s.X = max(0, s.X - step_X*grad_X);
            end
        end
        
        % Block update for the regression parameters
        function [s, info] = block_update_regParameters(s)
            Z=[s.X;s.U_train;ones(1,s.m)];
            Theta=s.Y_train*Z'/(Z*Z'+s.precision*eye(s.h+s.n+1));
            s.A=Theta(:,1:s.h); s.B=Theta(:,s.h+1:s.h+s.n); s.c=Theta(:,s.h+s.n+1);
            info = s.utils.MSE_implicit_objective(s.X,s.A,s.B,s.c,s.U_train,s.Y_train);
        end
        
        % Block update for hidden parameters
        function [s, fvals, diff] = block_update_HiddenParameters(s,num_iter, method)
            if norm(s.lambda) == 0
                lam = ones(s.h, 1);
            else
                lam = s.lambda;
            end

            fvals = NaN*zeros(num_iter+1,1);
            fvals(1) = s.utils.scalar_fenchel_divergence(s.X,  s.D* s.X + s.E*s.U_train + s.f*ones(1, s.m) , lam); fval_prev=fvals(1)+1;
            D_prev = s.D; E_prev = s.E; f_prev = s.f;
            iter=1;
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
            
                diff = sqrt(norm(s.D - D_prev, 'fro')^2 + norm(s.E - E_prev, 'fro')^2 + norm(s.f - f_prev)^2);
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
            if norm(s.lambda)>0
                Cst = diag(s.lambda)*(1/s.m)*(max(0,s.D*s.X + s.E*s.U_train + s.f*ones(1,s.m)) - s.X);
                grad_D = Cst*s.X';
                grad_E = Cst*s.U_train';
                grad_f = Cst*ones(s.m,1);
            else
                Cst = (1/s.m)*(max(0,s.D*s.X + s.E*s.U_train + s.f*ones(1,s.m)) - s.X);
                grad_D = Cst*s.X';
                grad_E = Cst*s.U_train';
                grad_f = Cst*ones(s.m,1);
            end
        end
        
        %%  Step size computation
        function  out = step_size_parameters_reg(s)
            out = s.m/max([s.m,norm(s.X)^2, norm(s.U_train)^2, norm(s.X*s.U_train')]);
        end
        
        function  out = step_size_parameters_hid(s)
            if norm(s.lambda)>0
                out = s.m/(max(s.lambda)*max([s.m,norm(s.X)^2,norm(s.U_train)^2, norm(s.U_train)*norm(s.X)]));
            else
                out = s.m/(max([s.m,norm(s.X)^2, norm(s.U_train)^2,norm(s.U_train)*norm(s.X)]));
            end
        end
        
        function  out = step_size_X(s)
            if norm(s.lambda)>0
                out = s.m/(norm(s.A'*s.A + diag(s.lambda) - diag(s.lambda)*s.D + s.D'*diag(s.lambda)) + max(s.lambda)*norm(s.D)^2);
            else
                out = s.m/(norm(s.A'*s.A));
            end
        end
   
        function D = well_posedness_projection(s,D, radius)
            if strcmp(s.well_posedness, 'infty')
                D = s.utils.infty_norm_projection(D, radius);
            elseif strcmp(s.well_posedness,'LMI')
                D = s.utils.lmi_projection(s.A, D, s.lambda);
            end
        end
        
        function s = parameter_initialization(s)
            s.A = rand(s.p, s.h) - 0.5;
            s.B  =rand(s.p, s.n) - 0.5;
            s.c = rand(s.p, 1) - 0.5;
            s.D = s.well_posedness_projection(rand(s.h,s.h) - 0.5, s.radius);
            s.E = rand(s.h, s.n) - 0.5;
            s.f = rand(s.h, 1) - 0.5;
        end
        
        %% Visualization
        function visualize_algo_init(s)
            s.utils.visualize_algo_init(s.additional_info.fval_X, s.additional_info.fval_hidden_param)
        end
    end
end