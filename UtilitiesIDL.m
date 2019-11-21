classdef UtilitiesIDL
    properties
        wp_precision=10^-3;
        precision=10^-6;
    end
    methods
        % constructor
        function s=UtilitiesIDL
        end
        
        function D=infty_norm_projection(s,D)
            [h,~]=size(D);
            for j=1:h
                D(j,:)=(1-s.wp_precision)*proj_b1(1/(1-s.wp_precision)*D(j,:));
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
        
        function X=picard_iterations(s,U,D,E,f)
            [h,~]=size(D);
            [~,m]=size(U);
            X=rand(h,m)-0.5;
            k=1;
            while k<10^3 && norm(X-max(0,D*X+E*U+f*ones(1,m)),'fro')>s.precision
                X=max(0,D*X+E*U+f*ones(1,m));
                k=k+1;
            end
            if k>=10^3
                disp('picard iterations were not able to find a solution X to the implicit model within 1,000 iterations')
            end
        end
        
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
        
        % compute the MSE given model parameters
        function out=RMSE_actual_implicit(s,A,B,c,D,E,f,U,Y)
            X=s.picard_iterations(U,D,E,f);
            out=sqrt(s.MSE_implicit_objective(X,A,B,c,U,Y));
        end
            
        function fval = implicit_objective(s,X, A, B, c, D, E, f, U, Y, lambda)
             [h,m]=size(X);
            fval = s.MSE_implicit_objective(X, A, B, c, U, Y)+(lambda.*ones(h,1))'* s.fenchel_divergence(X, D*X+E*U+f*ones(1,m));
        end
        
                
        function fval=scalar_fenchel_divergence(s,U, V, lambda)
            fval=s.fenchel_divergence(U, V)'*lambda;
        end
        
    end
    
    methods(Static)
        function []=visualize_algo(fval_X, diff_X)
            close all;
            subplot(2,1,1)
            semilogx(fval_X, 'b', 'LineWidth',0.5)
            hold on 
            semilogx(nanmean(fval_X,2), 'r', 'LineWidth',2)
            title('Convergence of the X-hidden var for each BCD update')
            xlabel('Inner BCD iterations')
            ylabel('Implicit Fenchel Objective')
            subplot(2,1,2)
            plot(diff_X)
            title('Norm of update difference for X-hidden var across BCD updates')
            xticks(1:1:length(diff_X))
            xlabel('BCD iterations')
            ylabel('1/m ||X^{k+1}-X^{k}||_F')
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
            [~,m]=size(U);
            fval=(1/m)*(0.5*sum(U,2).^2+0.5*sum(max(0,V),2).^2-sum(U.*V,2));
        end
        
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

    end
end