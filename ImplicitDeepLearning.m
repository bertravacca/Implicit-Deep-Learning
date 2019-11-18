classdef ImplicitDeepLearning
    properties
        precision = 10^-5;             % precision used in the solver
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
                s.additional_info = struc('fval_X',[],'diff_X');
            end
        end
         
        function s=train(s)

        end
        
        function s = initial_train(s)
            s = s.parameter_initialization;
            num_iter_bcd=2;
            s.fval_reg = NaN*zeros(100,1);
            s.rmse = NaN*zeros(100,1);
            %% Initial training
            % initial implicit problem (lambda=0) start with (A,B,c,X)...
            num_iter_X = 500;
            if s.harpagon ==1
                s.additional_info.fval_X = NaN*zeros(num_iter_X+1, num_iter_bcd);
                s.additional_info.diff_X = NaN*zeros(num_iter_bcd,1);
                for iter_bcd=1:num_iter_bcd
                    % updates
                    s.X = s.utils.picard_iterations(s.U_train, s.D, s.E, s.f);
                    s = s.block_update_regParameters;
                    [s, s.additional_info.fval_X(:,iter_bcd), s.additional_info.diff_X(iter_bcd)] = s.block_update_X(num_iter_X);
                    s = s.block_update_HiddenParameters(500);
                    s.fval_reg(iter_bcd) = s.utils.MSE_implicit_objective(s.X, s.A, s.B, s.c, s.U_train, s.Y_train);
                    s.rmse(iter_bcd) = s.utils.RMSE_actual_implicit(s.A,s.B,s.c,s.D,s.E,s.f,s.U_train,s.Y_train);
                    s.fval_fenchel_divergence(iter_bcd) = s.utils.scalar_fenchel_divergence(s.X, s.D*s.X + s.E*s.U_train + s.f*ones(1,s.m));
                end
            end
        end
        
        % algorithms
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
        
        function [s, info] = block_update_regParameters(s)
            Z=[s.X;s.U_train;ones(1,s.m)];
            Theta=s.Y_train*Z'/(Z*Z'+s.precision*eye(s.h+s.n+1));
            s.A=Theta(:,1:s.h); s.B=Theta(:,s.h+1:s.h+s.n); s.c=Theta(:,s.h+s.n+1);
            info = s.utils.MSE_implicit_objective(s.X,s.A,s.B,s.c,s.U_train,s.Y_train);
        end
        
        function  [s, fvals, diff] = block_update_X(s,num_iter)
            fvals = NaN*zeros(num_iter+1,1);
            fvals(1) = s.utils.implicit_objective(s.X, s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.lambda);
            X_prev = s.X;
            for inner_iter = 1:num_iter
                grad_X = s.gradient_hidden_var;
                step_X = s.step_size_X;
                s.X = max(0, s.X - step_X*grad_X);
                fvals(inner_iter+1) = s.utils.implicit_objective(s.X, s.A, s.B, s.c, s.D, s.E, s.f, s.U_train, s.Y_train, s.lambda);
            end
            diff = norm(s.X - X_prev, 'fro');
        end
        
        function s=block_update_HiddenParameters(s,num_iter)
            for k = 1:num_iter
                [grad_D, grad_E, grad_f] = s.gradient_parameters_hid;
                step_theta_hid = 2*s.step_size_parameters_hid;
                s.D = s.well_posedness_projection(s.D-step_theta_hid*grad_D);
                s.E = s.E - step_theta_hid*grad_E;
                s.f = s.f - step_theta_hid*grad_f;
            end
        end
        
        % gradients
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
        
        % step sizes
     
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
   
        function D = well_posedness_projection(s,D)
            if strcmp(s.well_posedness, 'infty')
                D = s.utils.infty_norm_projection(D);
            elseif strcmp(s.well_posedness,'LMI')
                D = s.utils.lmi_projection(s.A, D, s.lambda);
            end
        end
        
        function s = parameter_initialization(s)
            s.A = rand(s.p, s.h) - 0.5;
            s.B  =rand(s.p, s.n) - 0.5;
            s.c = rand(s.p, 1) - 0.5;
            s.D = s.well_posedness_projection(rand(s.h,s.h) - 0.5);
            s.E = rand(s.h, s.n) - 0.5;
            s.f = rand(s.h, 1) - 0.5;
        end
        
    end
end