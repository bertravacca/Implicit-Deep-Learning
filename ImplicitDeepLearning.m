classdef ImplicitDeepLearning
    properties
        input % input
        output % output
        X %     hidden features
        h %     # of hidden variables
        m%     # of datapoints
        n %     # of features for the input
        p %     # of outputs
           %     the following matrices and vectors correspond to the implicit
           %     prediction rule
           %                            y=Ax+Bu+c; x=max(0,Dx+Eu+f)
        A 
        B
        c
        D
        E
        f
        % precision used in the solver
        precision=10^-5;
        % dual variable for fenchel
        lambda=0
        % well_posedness specification
        well_posedness='infty'
        fval_reg
        fval_fenchel_divergence
    end
    
    methods
        function self=ImplicitDeepLearning(U,Y,h)
            self.input=U;
            self.output=Y;
            self.h=h;
            [self.n,self.m]=size(U);
            [self.p,~]=size(Y);
            %TODO: include checks of inputs   
        end
        
        function self=train(self)
            self=self.initialization;
            self.X=self.picard_iterations;
            self.fval_reg=NaN*zeros(100,1);
            % initial implicit problem (lambda=0) start with (A,B,c,X)...
            for k=1:100
                [grad_A,grad_B,grad_c]=self.gradient_parameters_reg;
                grad_X=self.gradient_hidden_var;
                step_theta_reg=self.step_size_parameters_reg;
                step_X=self.step_size_X;
                self.A=self.A-step_theta_reg*grad_A;
                self.B=self.B-step_theta_reg*grad_B;
                self.c=self.c-step_theta_reg*grad_c;
                self.X=max(0,self.X-step_X*grad_X);
                self.fval_reg(k)=self.objective_reg;
            end
            
            % ... then continue with (D,E,f)
            self.fval_fenchel_divergence=NaN*zeros(200,1);
            for k=1:200
                [grad_D,grad_E,grad_f]=self.gradient_parameters_hid;
                step_theta_hid=self.step_size_parameters_hid;
                self.D=self.well_posedness_projection(self.D-step_theta_hid*grad_D);
                self.E=self.E-step_theta_hid*grad_E;
                self.f=self.f-step_theta_hid*grad_f;   
                self.fval_fenchel_divergence(k)=self.objective_scalar_fenchel_divergence;
            end
        end
        
        function X=picard_iterations(self,X)
            if nargin==1
                X=rand(self.h,self.m);
            end
            k=1;
            while k<10^3 && norm(X-max(0,self.D*X+self.E*self.input+self.f*ones(1,self.m)),'fro')>self.precision
                X=max(0,self.D*X+self.E*self.input+self.f*ones(1,self.m));
                k=k+1;
            end
            if k>=10^3
                disp('picard iterations were not able to find a solution X to the implicit model')
            end
        end
        
        % gradients
        function grad_X=gradient_hidden_var(self)
            if norm(self.lambda)>0
                grad_X=(1/self.m)*( self.A'*(self.A*self.X+self.B*self.input+self.c*ones(1,self.m))+ ...
                    (diag(self.lambda)-diag(self.lambda)*self.D-self.D'*diag(self.lambda))*self.X+ ...
                    self.D'*diag(self.lambda)*max(0,self.D*self.X+self.E*self.input+self.f*ones(1,self.m))- ...
                    diag(self.lambda)*(self.E*self.input+self.f*ones(1,self.m)) );
            else
                 grad_X=(1/self.m)*( self.A'*(self.A*self.X+self.B*self.input+self.c*ones(1,self.m)));
            end
        end
        
        function [grad_A,grad_B,grad_c]=gradient_parameters_reg(self)
            Cst=(1/self.m)*(self.A*self.X+self.B*self.input+self.c*ones(1,self.m)-self.output);
            grad_A=Cst*self.X';
            grad_B=Cst*self.input';
            grad_c=Cst*ones(self.m,1);
        end
        
        function [grad_D,grad_E,grad_f]=gradient_parameters_hid(self)
            if norm(self.lambda)>0
                Cst=diag(self.lambda)*(1/self.m)*(max(0,self.D*self.X+self.E*self.input+self.f*ones(1,self.m))-self.X);
                grad_D=Cst*self.X';
                grad_E=Cst*self.input';
                grad_f=Cst*ones(self.m,1);
            else
                Cst=(1/self.m)*(max(0,self.D*self.X+self.E*self.input+self.f*ones(1,self.m))-self.X);
                grad_D=Cst*self.X';
                grad_E=Cst*self.input';
                grad_f=Cst*ones(self.m,1);
            end
        end
        
        % step sizes
     
        function  out=step_size_parameters_reg(self)
            out=self.m/max([self.m,norm(self.X)^2,norm(self.input)^2,norm(self.X*self.input')]);
        end
        
        function  out=step_size_parameters_hid(self)
            if norm(self.lambda)>0
                out=self.m/(max(self.lambda)*max([self.m,norm(self.X)^2,norm(self.input)^2,norm(self.input)*norm(self.X)]));
            else
                out=self.m/(max([self.m,norm(self.X)^2,norm(self.input)^2,norm(self.input)*norm(self.X)]));
            end
        end
        
        function  out=step_size_X(self)
            if norm(self.lambda)>0
                out=self.m/(norm(self.A'*self.A+diag(self.lambda)-diag(self.lambda)*self.D+self.D'*diag(self.lambda))+max(self.lambda)*norm(self.D)^2);
            else
                out=self.m/(norm(self.A'*self.A));
            end
        end
        
        function fval=objective_reg(self)
            fval=(1/sqrt(self.m))*norm(self.A*self.X+self.B*self.input+self.c*ones(1,self.m)-self.output,'fro');
        end
        
        function fval=objective_scalar_fenchel_divergence(self)
            fval=(1/sqrt(self.m))*sqrt(0.5*norm(self.X,'fro')^2+0.5*norm(max(0,self.D*self.X+self.E*self.input+self.f*ones(1,self.m)),'fro')-trace(self.X'*(self.D*self.X+self.E*self.input+self.f*ones(1,self.m))));
        end
        
        function [A,D]=lmi_projection(self,A,D,lambda)
            if norm(lambda)>0
                A_init=A;
                D_init=D;
                Lam=diag(lambda);
                h=self.h;
                p=self.p;
                prec=self.precision;
                cvx_begin sdp quiet
                variables A(p,h)
                variables D(h,h)
                minimize(norm([A;D]-[A_init;D_init]))
                Lam*D+D'*Lam<=Lam+A'*A
                norm(D)<=1-prec
                cvx_end
            end
        end
        
        function D=infty_norm_projection(self,D)
            for j=1:self.h
                D(j,:)=(1-self.precision)*proj_b1(1/(1-self.precision)*D(j,:));
            end
        end
        
        function D=well_posedness_projection(self,D)
            if strcmp(self.well_posedness,'infty')
                D=self.infty_norm_projection(D);
            elseif strcmp(self.well_posedness,'LMI')
                D=self.lmi_projection(self.A,D,self.lambda);
            end
        end
        
        function self=initialization(self)
            self.A=rand(self.p,self.h)-0.5;
            self.B=rand(self.p,self.n)-0.5;
            self.c=rand(self.p,1)-0.5;
            self.D=self.well_posedness_projection(rand(self.h,self.h)-0.5);
            self.E=rand(self.h,self.n)-0.5;
            self.f=rand(self.h,1)-0.5;
        end
  
    end
end