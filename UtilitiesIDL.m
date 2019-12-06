classdef UtilitiesIDL
    properties
        wp_precision=10^-3;
        precision=10^-6;
    end
    methods
        % constructor
        function s=UtilitiesIDL
        end
        
        % projections
        function D=infty_norm_projection(s,D,radius)
            [h,~]=size(D);
            for j=1:h
                D(j,:)=(1-radius)*s.proj_b1(1/(1-radius)*D(j,:));
            end
        end
        
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
        
        % compute implicit X
        function X=picard_iterations(s,U,D,E,f, activation)
            [h,~]=size(D);
            [~,m]=size(U);
            X=rand(h,m)-0.5;
            k=1;
            if strcmp(activation, 'ReLU')
                while k<10^3 && norm(X-max(0,D*X+E*U+f*ones(1,m)),'fro')>s.precision
                    X=max(0,D*X+E*U+f*ones(1,m));
                    k=k+1;
                end
            elseif strcmp(activation, 'leakyReLU')
                while k<10^3 && norm(X-s.leakyReLU(D*X+E*U+f*ones(1,m)),'fro')>s.precision
                    X=s.leakyReLU(D*X+E*U+f*ones(1,m));
                    k=k+1;
                end
            end
            if k>=10^3
                disp('picard iterations were not able to find a solution X to the implicit model within 1,000 iterations')
            end

        end
        
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
        
        % compute the RMSE given model parameters
        function out=RMSE_actual_implicit(s,A,B,c,D,E,f,U,Y, activation)
            X=s.picard_iterations(U,D,E,f, activation);
            out=sqrt(s.MSE_implicit_objective(X,A,B,c,U,Y));
        end
        
        % full objective for implicit deep learning
        function fval = implicit_objective(s,X, A, B, c, D, E, f, U, Y, lambda)
             [h,m]=size(X);
            fval = s.MSE_implicit_objective(X, A, B, c, U, Y)+(lambda.*ones(h,1))'* s.fenchel_divergence(X, D*X+E*U+f*ones(1,m));
        end
         
        function fval = scalar_fenchel_divergence(s,U, V, lambda)
            fval=s.fenchel_divergence(U, V)'*lambda;
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
        
    end
    
    methods(Static)
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
            title('Evolution of RMSE across iterations')
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
            title('Evolution of RMSE across iterations')
            xlabel('iterations')
            legend('objective implicit', 'RMSE')
        end

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
        
        % just compute RMSE
        function out=RMSE(Y_1, Y_2)
            [~,m]=size(Y_1);
            out=(1/sqrt(m))*norm(Y_1-Y_2,'fro');
        end
        
        % compute the MSE, given X (not considering satisfaction of the implicit constraint)
        function fval=MSE_implicit_objective(X, A, B, c, U, Y)
            [~,m]=size(U);
            fval=(1/m)*norm(A*X+B*U+c*ones(1,m)-Y,'fro')^2;
        end
        
        function fval = fenchel_divergence(U, V)
            m = size(U, 2);
            fval = (1/m)*(0.5*sum(U.^2,2)+0.5*sum(max(0,V).^2,2)-sum(U.*V,2));
        end
        
        function val = L2_implicit_constraint(U,V)
            m = size(U,2);
            val = (1/sqrt(m))*norm(U-max(0,V),'fro');
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
        
        function out = leakyReLU(z)
            out = max(0,z) + 0.5 * min(0,z);
        end
        
        function out = inverse_leakyReLU(z)
            out = max(0, z) + 2* min(0, z);
        end

    end
end