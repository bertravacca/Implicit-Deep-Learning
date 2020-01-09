classdef UtilitiesIDL
    properties
        wp_precision=10^-3;
        precision=10^-6;
    end
    methods
        % constructor
        function s=UtilitiesIDL
        end
        
        %% Projections 
        function D=infty_norm_projection(s,D,radius)
            [h,~]=size(D);
            for j=1:h
                D(j,:)=(1-radius)*s.proj_b1(1/(1-radius)*D(j,:));
            end
        end
        
        % (!!! not sur this works !!!)
        function [A,D]=lmi_projection(s,A,D,lambda)
            if norm(lambda)>0
                A_init=A;
                D_init=D;
                Lam=diag(lambda);
                [h,~]=size(D);
                [p,~]=size(A);
                prec=s.wp_precision;
                cvx_begin sdp quiet
                variables A(p,h)
                variables D(h,h)
                minimize(norm([A;D]-[A_init;D_init]))
                Lam*D+D'*Lam<=Lam+A'*A
                norm(D)<=1-prec
                cvx_end
            end
        end
        
        % vector projection on the L1 ball
        function [sol,info] = proj_b1(s, x, ~, param)
            %PROJ_B1 Projection onto a L1-ball
            %   Usage:  sol=proj_b1(x, ~, param)
            %           [sol,infos]=proj_b1(x, ~, param)
            %
            %   Input parameters:
            %         x     : Input signal.
            %         param : Structure of parameters.
            %   Output parameters:
            %         sol   : Solution.
            %         info  : Structure summarizing informations at convergence
            %
            %   PROJ_B1(x,~,param) solves:
            %
            %      sol = argmin_{z} ||x - z||_2^2   s.t.  ||w.*z||_1 < epsilon
            %
            %   Remark: the projection is the proximal operator of the indicative function of
            %   w.*z||_1 < epsilon. So it can be written:
            %
            %      prox_{f, gamma }(x)      where       f= i_c(||w.*z||_1 < epsilon)
            %
            %   param is a Matlab structure containing the following fields:
            %
            %    param.epsilon : Radius of the L1 ball (default = 1).
            %
            %    param.weight : contain the weights (default ones).
            %
            %    param.verbose : 0 no log, 1 a summary at convergence, 2 print main
            %     steps (default: 1)
            %
            %
            %   info is a Matlab structure containing the following fields:
            %
            %    info.algo : Algorithm used
            %
            %    info.iter : Number of iteration
            %
            %    info.time : Time of exectution of the function in sec.
            %
            %    info.final_eval : Final evaluation of the function
            %
            %    info.crit : Stopping critterion used
            %
            %
            %   Rem: The input "~" is useless but needed for compatibility issue.
            %
            %   This code is partly borrowed from the SPGL toolbox!
            %
            %   See also:  proj_b2 prox_l1
            %
            %   Url: https://epfl-lts2.github.io/unlocbox-html/doc/prox/proj_b1.html
            
            % Copyright (C) 2012-2016 Nathanael Perraudin.
            % This file is part of UNLOCBOX version 1.7.4
            %
            % This program is free software: you can redistribute it and/or modify
            % it under the terms of the GNU General Public License as published by
            % the Free Software Foundation, either version 3 of the License, or
            % (at your option) any later version.
            %
            % This program is distributed in the hope that it will be useful,
            % but WITHOUT ANY WARRANTY; without even the implied warranty of
            % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
            % GNU General Public License for more details.
            %
            % You should have received a copy of the GNU General Public License
            % along with this program.  If not, see <http://www.gnu.org/licenses/>.
            
            %
            % Author: Nathanael Perraudin
            % Date: February 2015
            % Testing: test_proj_b1
            
            % Start the time counter
            t1 = tic;
            
            % Optional input arguments
            if nargin<3, param=struct; end
            if ~isfield(param, 'epsilon'), param.epsilon = 1; end
            if ~isfield(param, 'verbose'), param.verbose = 0; end
            if ~isfield(param, 'weight'), param.weight = ones(size(x)); end
            
            if isfield(param,'w')
                error('Change in the UNLocBoX! Use weight instead of w!');
            end
            
            if isscalar(param.weight), param.weight = ones(size(x))* param.weight; end
            param.weight = abs(param.weight);
            
            % Quick return for the easy cases.
            if sum(param.weight) == 0
                sol   = x;
                iter = 0;
                
                crit='--';
                info.algo=mfilename;
                info.iter=iter;
                info.final_eval=0;
                info.crit=crit;
                info.time=toc(t1);
                return
            end
            
            % Get sign of b and set to absolute values
            signx = sign(x);
            x = abs(x);
            
            idx = find(x > eps); % Get index of all non-zero entries of d
            sol   = x;             % Ensure x_i = b_i for all i not in index set idx
            [sol(idx),iter] = s.one_projector(sol(idx),param.weight(idx),param.epsilon);
            
            
            % Restore signs in x
            sol = sol.*signx;
            
            
            
            % Log after the projection onto the L2-ball
            if param.verbose >= 1
                fprintf('  Proj. B1: epsilon = %e, ||x||_2 = %e,\n', param.epsilon, norm(sol,1));
            end
            
            crit='--';
            info.algo=mfilename;
            info.iter=iter;
            info.final_eval=norm(param.weight.*sol,1);
            info.crit=crit;
            info.time=toc(t1);
            
        end
        
        %% Picard Iteration
        function [X, num_iter]=picard_iterations(s, U, D, E, f, activation_type)
            [h,~]=size(D);
            [~,m]=size(U);
            X=rand(h,m)-0.5;
            k=1;
            while k<10^3 && norm( X - s.activation( D * X + E * U + f * ones(1,m), activation_type),'fro')>s.precision
                X = s.activation( D * X + E * U + f * ones(1,m) , activation_type);
                k=k+1;
            end    
            if k>=10^3
                disp('picard iterations were not able to find a solution X to the implicit model within 1,000 iterations')
            end
            if nargout == 2
                num_iter = k-1;
            end
        end
        
        %% Gradients and step
        function [gradX, step] = gradient_X_mse_loss_fenchel(s, U, Y, X, A, B, c, D, E, f, lambda, activation_type, L2reg, L1reg)
            m = size(U,2);
            if norm(lambda)>0
                Lam = diag( lambda );
                c_1 =  A' * ( A * X + B * U + c * ones(1 ,m) - Y );
                c_2 = ( Lam - Lam * D - D' * Lam) * X; 
                c_3 = D' * Lam * s.activation( D * X +  E * U + f * ones(1, m), activation_type);
                c_4 =  -Lam * (E * U + f * ones(1,m));
                gradX = (1/m)*( c_1 + c_2 + c_3 + c_4)' + L2reg * X + L1reg * sign(X);
                if nargout == 2
                    step = m/( norm( A' * A + Lam - Lam * D + D' * Lam ) + max( lambda ) * norm( D )^2 +L2reg);
                end
            else
                 gradX = (1/m)*(A' * ( A * X + B * U + c * ones(1, m) - Y));
                 if nargout == 2
                     step = m/( norm( A' * A ) + L2reg);
                 end
            end
        end
            
        function [grad_A, grad_B, grad_c, step] = gradient_parameters_mse_loss(s, U, Y, X, A, B, c, L2reg, L1reg)
            m = size(U,2);
            Cst = (1/m) * (A * X + B* U + c * ones(1, m) - Y);
            grad_A = Cst * X' + L2reg * A + L1reg * sign(A);
            grad_B = Cst * U' + L2reg * B + L1reg * sign(B);
            grad_c = Cst * ones(m,1) + L2reg * c + L1reg * sign(c);
            if nargout == 4
                step = m/max([m, norm(X)^2, norm(U)^2, norm(X*U')] + L2reg);
            end
        end
        
        function [grad_D, grad_E, grad_f, step] = gradient_hidden_parameters_fenchel(s, U, X, D, E, f, lambda, activation_type, L2reg, L1reg)
            m = size(U,2);
            if norm(lambda)>0
                Cst = diag(lambda) * (1/m) * ( activation( D * X + E * U + f * ones(1, m), activation_type) - X ) ;
                if nargout == 4
                    step = m / ( max( lambda ) * max( [ m, norm( X )^2, norm( U )^2 , norm( U )*norm(X) ] ) +L2reg);
                end
            else
                Cst = (1/m) * (max(0,s.D * s.X + s.E * s.U_train + s.f * ones(1,s.m)) - s.X);
                if nargout == 4
                    step = m / (  max( [ m, norm( X )^2, norm( U )^2 , norm( U )*norm(X) ] ) +L2reg);
                end
            end
            grad_D = Cst * X' + L2reg * D + L1reg * sign(D);
            grad_E = Cst * U' + L2reg * E + L1reg * sign(E);
            grad_f = Cst * ones(m,1) + L2reg * f + L1reg * sign(f);
        end
            
        function [grad_D, grad_E, grad_f, step] = gradient_hidden_parameters_linear(s, U, X, D, E, f, activation_type, L2reg, L1reg)
            m = size(U,2);
            Cst = (1/m) * (D * X + E * U + f * ones(1, m)  - s.inverse_activation(X, activation_type)) ;
            grad_D = Cst * X' + L2reg * D + L1reg * sign(D);
            grad_E = Cst * U'+ L2reg * E + L1reg *sign(E);
            grad_f = Cst * ones(m, 1) + L2reg * f + L1reg * sign(f);
            step = m / ( max( [ m, norm(X)^2, norm(U)^2, norm(U)*norm(X)] ) + L2reg );
        end
        
        function [grad_D, grad_E, grad_f] = gradient_hidden_parameters_implicit_chain_rule(s, data_index,  U, Y,  X, A, B, c, D, E, f, activation_type)
            m = size(U, 2);
            if isempty(data_index)
                data_index = 1:1:m ;
            end
            m_sub = length(data_index); 
            h = size(X, 1);
            n = size(U, 1);
            grad_D = zeros(h, h);
            grad_E = zeros(h, n);
            grad_f = zeros(h, 1);
            for j = 1 : m_sub
                i = data_index(j);
                z = D * X(:, i) + E * U(:, i) + f;
                V = diag( s. elementwise_derivative_activation(z, activation_type));
                y_pred = A * X(:, i) + B * U(:, i) + c;
                K =  V * (eye(h) - D' * V) \ (A' * ( y_pred - Y(:, i)));
                grad_f = grad_f + K;
                for k = 1 : h
                    delta = s.kron_delta(k, h);
                    grad_D(k, :) = grad_D( k, :) + (X(:, i) * delta' * K)';
                    grad_E(k, :) = grad_E( k, :) + ( U(:, i) * delta' * K)';
                end
            end
            grad_D = (1/m_sub) * grad_D;
            grad_E = (1/m_sub) * grad_E;
            grad_f = (1/m_sub) * grad_f;
        end
       
        %% Gradient updates
        function lambda = dual_update(s, U, X, D, E, f, lambda, dual_step)
            F_star = s.fenchel_divergence( X, D * X + E * U + f * ones(1, m) );
            v = 1*( F_star> tolerance );
            lambda = lambda + dual_step * v;
        end
        
        function X = gradient_update_X_mse_loss_fenchel (s, U, Y, X, A, B, c, D, E, f, lambda, activation_type, L2reg, L1reg)
            [gradX, step] = s.gradient_X_mse_loss_fenchel(U, Y, X, A, B, c, D, E, f, activation_type, lambda, L2reg, L1reg);
            X = X - step * gradX;
            if strcmp(activation_type, 'ReLU')
                X = max(0, X);
            end
        end
            
        function [A, B, c] = gradient_update_parameters_mse_loss(s, U, Y, X, A, B, c, L2reg, L1reg, step)
            if nargin == 9
                [grad_A, grad_B, grad_c, step] = s.gradient_parameters_mse_loss(U, Y, X, A, B, c, L2reg, L1reg);
            else
                [grad_A, grad_B, grad_c, ~] = s.gradient_parameters_mse_loss(U, Y, X, A, B, c, L2reg, L1reg);
            end
            A = A - step * grad_A;
            B = B - step * grad_B;
            c = c - step * grad_c;
        end
        
        function [D, E, f] = gradient_update_hidden_parameters_fenchel(s, U, X, D, E, f, lambda, activation_type, wp_type, L2reg, L1reg)
            [grad_D, grad_E, grad_f, step] = gradient_hidden_parameters_fenchel(s, U, X, D, E, f, activation_type, lambda, L2reg, L1reg);
            D = D - step * grad_D;
            E = E - step * grad_E;
            f = f - step * grad_f;
            if strcmp(wp_type, 'infty')
                D = s.infty_norm_projection(D, 0.5);
            end
        end
        
        function [D, E, f] = gradient_update_hidden_parameters_linear(s, U, X, D, E, f, activation_type, wp_type, L2reg, L1reg)
            [grad_D, grad_E, grad_f, step] = gradient_hidden_parameters_linear(s, U, X, D, E, f, activation_type, L2reg, L1reg);
            step = step;
            D = D - step * grad_D;
            E = E - step * grad_E;
            f = f - step * grad_f;
            if strcmp(wp_type, 'infty')
                D = s.infty_norm_projection(D, 0.5);
            end
        end
        
        function [D, E, f] = gradient_update_hidden_parameters_implicit_chain_rule(s, data_index,  U, Y,  X, A, B, c, D, E, f, activation_type, wp_type,  step)
            [grad_D, grad_E, grad_f] = s.gradient_hidden_parameters_implicit_chain_rule(data_index,  U, Y,  X, A, B, c, D, E, f, activation_type);
            D = D - step * grad_D;
            E = E - step * grad_E;
            f = f - step * grad_f;

            if strcmp(wp_type, 'infty')
                D = s.infty_norm_projection(D, 0.5);
            end
        end
        
        %% Activation
        function out = activation(s, z, activation_type)
            if strcmp(activation_type, 'ReLU')
                out = s.ReLU(z);
            elseif strcmp(activation_type, 'leakyReLU')
                out = s.leakyReLU(z);
            end
        end
        
        function out = inverse_activation(s, z, activation_type)
            if strcmp(activation_type, 'ReLU')
                error('ReLU is not bijective, cannot use the linear method for learning the hidden parameters')
            elseif strcmp(activation_type, 'leakyReLU')
                out = s.inverse_leakyReLU(z);
            end
        end
      
        
        %% Scores and errors
        function val = L2_implicit_constraint_error(s, U,V, activation_type)
            m = size(U,2);
            val = (1/sqrt(m))*norm(U-s.activation(V, activation_type),'fro');
        end
        
        function val = score(s, U, Y, A, B, c, D, E, f, activation_type, loss_type)
            m = size(U, 2);
            X = s.picard_iterations(U, D, E, f, activation_type);
            if strcmp(loss_type, 'mse')
                val = rmse(Y, A * X + B* U + c * ones(1, m));
            end
        end
        
        function val = fval_X(s, U, Y, X, A, B, c, D, E, f, lambda, method)
            m = size(U, 2);
            if strcmp(method, 'linear')
                val = s.rmse(Y, A * X + B* U + c * ones(1,m) );
            elseif strcmp(method, 'fenchel')
                val =  s.rmse(Y, A * X + B* U + c * ones(1,m) )^2 +  scalar_fenchel_divergence( X, D * X + E* U + f *ones(1, m), lambda) ;
            end
        end
        
        %% Miscellanous
        % generate random orthogonal matrix
        function M=RandOrthMat(s,n)
            % (c) Ofek Shilon , 2006.
            tol=s.precision;
            if nargin==1
                tol=1e-6;
            end
            M = zeros(n); 
            % gram-schmidt on random column vectors
            vi = randn(n,1);
            % the n-dimensional normal distribution has spherical symmetry, which implies
            % that after normalization the drawn vectors would be uniformly distributed on the
            % n-dimensional unit sphere.
            M(:,1) = vi ./ norm(vi);
            for i=2:n
                nrm = 0;
                while nrm<tol
                    vi = randn(n,1);
                    vi = vi -  M(:,1:i-1)  * ( M(:,1:i-1).' * vi )  ;
                    nrm = norm(vi);
                end
                M(:,i) = vi ./ nrm;
            end
        end
        

    end
    
    methods(Static)
        %% Visualize
        function [] = visualize_algo_init(fval_X, fval_hidden_param)
            figure()
            subplot(2,1,1)
            semilogx(fval_X, 'b', 'LineWidth',0.5)
            hold on 
            semilogx(nanmean(fval_X,2), 'r', 'LineWidth',2)
            title('Convergence of the X-hidden var for each BCD update')
            xlabel('Inner BCD iterations')
            ylabel('Implicit Fenchel Objective')
            
            subplot(2,1,2)
            semilogx(fval_hidden_param, 'b', 'LineWidth',0.5 )
            hold on
            semilogx(nanmean(fval_hidden_param,2), 'r', 'LineWidth',2)
            title('Convergence of the hidden parameters for each BCD update')
            xlabel('Inner BCD iterations')
            ylabel('Fenchel divergence')
        end
        
        function [] = visualize_algo(fval, fval_reg, rmse)
            figure()
            subplot(2,1,1)
            semilogx(fval, 'b', 'LineWidth',0.5)
            hold on 
             semilogx(fval_reg, 'r', 'LineWidth',0.5)
            xlabel('iterations')
            title('Evolution of implicit objective across iterations')
            
            subplot(2,1,2)
            semilogx(rmse, 'r', 'LineWidth',0.5)
            title('Evolution of rmse across iterations')
            xlabel('iterations')

        end
                
        function [] = live_visualize_algo(fval_point, rmse_point, iter)
            if iter == 1
                figure()
            end
            subplot(2,1,1)
            plot([fval_point, iter], 'b', 'LineWidth',0.5)
            if iter == 1
                hold on
            end
            xlabel('iterations')
            title('Evolution of implicit objective across iterations')
            
            subplot(2,1,2)
            plot([rmse_point, iter], 'r', 'LineWidth',0.5)
            if iter == 1
                hold on
            end
            title('Evolution of rmse across iterations')
            xlabel('iterations')
            legend('objective implicit', 'rmse')
        end
        
        %% Distance 
        % rmse distance
        function out=rmse(Y_1, Y_2)
            [~,m]=size(Y_1);
            out=(1/sqrt(m))*norm(Y_1-Y_2,'fro');
        end
        
        % fenchel divergence (vector)
        function fval = fenchel_divergence(U, V)
            m = size(U, 2);
            fval = (1/m)*(0.5*sum(U.^2,2)+0.5*sum(max(0,V).^2,2)-sum(U.*V,2));
        end
        
        % scalar fenchel divergence (vector)
        function fval = scalar_fenchel_divergence(U, V, lambda)
            m = size(U, 2);
            fval = lambda' * (1/m)*(0.5*sum(U.^2,2)+0.5*sum(max(0,V).^2,2)-sum(U.*V,2));
        end
        
        %% Activations 
        function out = ReLU(z)
            out = max(0, z);
        end
        
        function out = leakyReLU(z)
            out = max(0,z) + 0.5 * min(0,z);
        end
        
        function out = inverse_leakyReLU(z)
            out = max(0, z) + 2* min(0, z);
        end
        
        function out = elementwise_derivative_activation(z, activation_type)
            if strcmp(activation_type, 'ReLu')
                out = 1 * ( z > 0 );
            elseif strcmp(activation_type, 'leakyReLU')
                out = 1* ( z > 0 ) + 0.5 * ( z < 0 );
            end
        end
        
        %% Miscellanous
        function [sol,iter] = one_projector(x,weight,tau)
            % This code is partly borrowed from the SPGL toolbox
            % Initialization
            N = length(x);
            sol = zeros(N,1);
            
            % Check for quick exit.
            if (tau >= norm(weight.*x,1)), sol = x; iter = 0; return; end
            if (tau <  eps         ),        iter = 0; return; end
            
            % Preprocessing (b is assumed to be >= 0)
            [sw,idx] = sort(x ./ weight,'descend'); % Descending.
            x  = x(idx);
            weight  = weight(idx);
            
            % Optimize
            csdb = 0; csd2 = 0;
            soft = 0; ii = 1;
            while (ii <= N)
                csdb = csdb + weight(ii).*x(ii);
                csd2 = csd2 + weight(ii).*weight(ii);
                
                alpha1 = (csdb - tau) / csd2;
                alpha2 = sw(ii);
                
                if alpha1 >= alpha2
                    break;
                end
                
                soft = alpha1;  ii = ii + 1;
            end
            sol(idx(1:ii-1)) = x(1:ii-1) - weight(1:ii-1) * max(0,soft);
            
            % Set number of iterations
            iter = ii;
        end
        
        % divive data in training and validation
        function [U_trn, Y_trn, U_val, Y_val, seed] = divide_data(U, Y, ratio)
            m = size(U, 2);
            seed = randperm(m);
            m_trn = ceil(ratio * m);
            U_trn = U(:, seed(1:m_trn));
            Y_trn = Y(:, seed(1:m_trn));
            U_val = U(:, seed(m_trn+1:m));
            Y_val = Y(:, seed(m_trn+1:m));
        end
        
        % one hot encoding for categorical datasets
        function T=createOneHotEncoding(T,tableVariable)
            %
            % Code written by Christopher L. Stokely, January 30, 2019
            % Written in MATLAB R2018B.
            %
            % Command:
            % outputTable = createOneHotEncoding(T,tableVariable)
            %
            % Input variable T needs to be a table and the tableVariable should be
            % a variable in that table.  tableVariable should be a variable that is
            % categorical but it does not have to be.  The code below converts the
            % variable to categorical if it is not already so.  A table will be
            % returned that is the original input table without tableVariable, but
            % with new variables representing the one-hot encoded tableVariable.
            %
            % By one hot encoding, predictor importances can become very useful
            % when employing machine learning - from a model interpretability stand
            % -point. Being able to assign an importance to an individual category
            % can be useful and important in some cases.
            %
            % For educational purposes, try looking into these Machine Learning
            % toolbox commands after building a model:
            % 1) oobPermutedPredictorImportance
            % 2) predictorImportance  (Be careful - this one is known to mislead)
            % 3) FeatureSelectionNCARRegression
            % 4) fsrnca or fscnca
            % 5) sequentialfs
            % 6) plotPartialDependence
            % 7) Individual Conditional Expectation (ICE) plots
            %
            % Note a MATLAB bug or oversight from MathWorks regarding having an
            % underscore in the variable names that are in the table...
            % Note that the output table has new variables with labels that have an
            % underscore.  Removing these variables with "removevars" requires the
            % user to specify the column to be removed with the column number, not
            % the variable name.  Otherwise unintended columns will be deleted.
            
            
            %%
            % determine if it is a table and throw an error if not
            if ~istable(T)
                error('Input table variable is not a table!')
            end
            
            %%
            % determine if the table variable to be encoded actually exists - throw an error otherwise
            if ~iscolumn(T.(genvarname(tableVariable)))
                error('Column variable does not exist in table!')
            end
            
            %%
            % do we want to make this a categorical variable?  how many instances are
            % there for this variable?
            numCategories=numel(unique(T.(genvarname(tableVariable))));
            
            %fprintf('There are %i unique values in the tableVariable. \n', numCategories);
            
            % start creating one hot encoding variables if the number of categories after
            % variable expansion is reasonable
            
            if numCategories<100 % maximum of 99 categories for the variable splitting to be created
                
                tempCatVariable=T.(genvarname(tableVariable));
                
                % convert it to categorical in case it is not already
                tempCatVariable=categorical(tempCatVariable);
                
                %now get the unique categories that will be one hot encoded
                uniqueCategories=unique(tempCatVariable);
                
                %remove variable from table that is being one-hot encoded
                T=removevars(T,tableVariable);
                
                % BEWARE - dynamic variables created using the EVAL command
                % Other useful commands to try out are genvarname,
                % matlab.lang.makeUniqueStrings, and matlab.lang.makeValidName
                for indexCategories=1:numel(uniqueCategories)
                    oneHot=double((tempCatVariable==uniqueCategories(indexCategories))); %want a numeric value - not a Boolean value
                    eval((strcat(char(tableVariable),'_',char(uniqueCategories(indexCategories)),'=oneHot;')));
                    eval(strcat('T=addvars(T,',strcat(char(tableVariable),'_',char(uniqueCategories(indexCategories)),');')));
                end
                % NOTE: the user cannot remove variables created that have an
                % underscore in their variable name
                
            else
                error('There are over 99 categories for this table variable.  Unable to proceed unless user increases conditional on line 42 to a more appropriate level.');
            end
            
        end
        
        % solving basis pursuit with ADMM
        function [z, history] = basis_pursuit(A, b, rho, alpha)
            % basis_pursuit  Solve basis pursuit via ADMM
            %
            % [x, history] = basis_pursuit(A, b, rho, alpha)
            %
            % Solves the following problem via ADMM:
            %
            %   minimize     ||x||_1
            %   subject to   Ax = b
            %
            % The solution is returned in the vector x.
            %
            % history is a structure that contains the objective value, the primal and
            % dual residual norms, and the tolerances for the primal and dual residual
            % norms at each iteration.
            %
            % rho is the augmented Lagrangian parameter.
            %
            % alpha is the over-relaxation parameter (typical values for alpha are
            % between 1.0 and 1.8).
            %
            % More information can be found in the paper linked at:
            % http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
            %
            
            t_start = tic;
            QUIET    = 0;
            MAX_ITER = 1000;
            ABSTOL   = 1e-6;
            RELTOL   = 1e-6;

            [m n] = size(A);

            x = zeros(n,1);
            z = zeros(n,1);
            u = zeros(n,1);
            
            if ~QUIET
                fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
                    'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
            end
            
            % precompute static variables for x-update (projection on to Ax=b)
            AAt = A*A';
            P = eye(n) - A' * (AAt \ A);
            q = A' * (AAt \ b);
            
            for k = 1:MAX_ITER
                % x-update
                x = P*(z - u) + q;
                
                % z-update with relaxation
                zold = z;
                x_hat = alpha*x + (1 - alpha)*zold;
                z = max(0, x_hat + u - 1/rho) - max(0, - x_hat -u -1/rho);
                u = u + (x_hat - z);
                
                % diagnostics, reporting, termination checks
                history.objval(k)  = norm(x, 1);
                
                history.r_norm(k)  = norm(x - z);
                history.s_norm(k)  = norm(-rho*(z - zold));
                
                history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
                history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
                
                if ~QUIET
                    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
                        history.r_norm(k), history.eps_pri(k), ...
                        history.s_norm(k), history.eps_dual(k), history.objval(k));
                end
                
                if (history.r_norm(k) < history.eps_pri(k) && ...
                        history.s_norm(k) < history.eps_dual(k))
                    break;
                end
            end
            
            if ~QUIET
                toc(t_start);
            end
        end
        
        % solving extended basis pursuit with ADMM
        function [x, y, history] = extended_basis_pursuit(A, B, c,  rho, alpha)
            % basis_pursuit  Solve basis pursuit via ADMM
            %
            % [x, history] = extended_basis_pursuit(A, b, rho, alpha)
            %
            % Solves the following problem via ADMM:
            %
            %   minimize_{x,y}     ||x||_1
            %   subject to   Ax + By = c
            %
            % The solution is returned in the vector pair (x, y).
            %
            % history is a structure that contains the objective value, the primal and
            % dual residual norms, and the tolerances for the primal and dual residual
            % norms at each iteration.
            %
            % rho is the augmented Lagrangian parameter.
            %
            % alpha is the over-relaxation parameter (typical values for alpha are
            % between 1.0 and 1.8).
            %
            % More information can be found in the paper linked at:
            % http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
            %
            
            t_start = tic;
            QUIET    = 1;
            MAX_ITER = 10;
            ABSTOL   = 1e-6;
            RELTOL   = 1e-6;

            [~, n] = size(A);
            [~, p] = size(B);
            
            x = zeros(n,1);
            y = zeros(p,1);
            z = zeros(n,1);
            u = zeros(n,1);
            
            if ~QUIET
                fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
                    'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
            end
          
            
            for k = 1:MAX_ITER
                % x, y-update
                H = [eye(n), zeros(n, p); zeros(p, n+p)];
                f = [u - z; zeros(p,1)];
                Aeq = [A, B];
                beq = c;
                options =  optimset('Display','off');
                w = quadprog(H, f, [], [], Aeq, beq, [], [], [], options);
                x = w(1:n);
                y = w(n+1:n+p);
                
                % z-update with relaxation
                zold = z;
                x_hat = alpha*x + (1 - alpha)*zold;
                z = max(0, x_hat + u - 1/rho) - max(0, - x_hat -u -1/rho);
                u = u + (x_hat - z);
                
                % diagnostics, reporting, termination checks
                history.objval(k)  = norm(x, 1);
                
                history.r_norm(k)  = norm(x - z);
                history.s_norm(k)  = norm(-rho*(z - zold));
                
                history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
                history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
                history.linear_error(k) = norm(A*x + B*y -c, 2);
                
                if ~QUIET
                    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
                        history.r_norm(k), history.eps_pri(k), ...
                        history.s_norm(k), history.eps_dual(k), history.objval(k));
                end
                
                if (history.r_norm(k) < history.eps_pri(k) && ...
                        history.s_norm(k) < history.eps_dual(k))
                    break;
                end
            end
            
            if ~QUIET
                toc(t_start);
            end
        end
        
        % Kronecker delta vector
        function out = kron_delta(k, m)
            out = zeros(m,1);
            out(k) = 1;
        end

    end
end