classdef ImplicitDeepLearning
    properties
        U % input
        Y % output
        X % hidden features
        h % # of hidden variables
        m% # of datapoints
        n % # of features
        p % # of outputs
        % model parameters y=Ax+Bu+f; x=max(0,Dx+Eu+f)
        A 
        B
        c
        D
        E
        f
        precision=10^-6;
        lambda
    end
    
    methods
        function self=ImplicitDeepLearning(U,Y,h)
            self.U=U;
            self.Y=Y;
            self.h=h;
            [self.n,self.m]=size(U);
            [self.p,~]=size(Y);
            %TODO: include checks of inputs   
        end
        
        function X=picard_iterations(self,X)
            X_prev=X+1;
            while norm(X-X_prev)>self.precision
                X=max(0,self.D*X+self.E*self.U+self.f*ones(1,self.m));
            end
        end
        
        function [grad_A,grad_B,grad_c,grad_D,grad_E,grad_f]=gradient_parameters(self)
            Cst=(1/self.m)*(self.A*self.X+self.B*self.U+self.c*ones(1,self.m)-self.Y);
            grad_A=Cst*self.X';
            grad_B=Cst*self.U';
            grad_c=Cst*ones(self.m,1);
            
            Cst=diag(self.lambda)*(1/self.m)*(max(0,self.D*self.X+self.E*self.U+self.f*ones(1,self.m))-self.X);
            grad_D=Cst*self.X';
            grad_E=Cst*self.U';
            grad_f=Cst*ones(self.m,1);
        end
        
        function [alpha_theta_1,alpha_theta_2,alpha_X]=step_size(self)
            alpha_theta_1=self.m/max(self.m,norm(self.X)^2,norm(self.U)^2,norm(self.X*self.U'));
            alpha_theta_2=self.m/(max(self.lambda)*max(self.m,norm(self.X)^2,norm(self.U)^2,norm(self.U)*norm(self.X)));
            alpha_X=self.m/(norm(self.A'*self.A+diag(self.lambda)-diag(self.lambda)*self.D+self.D'*diag(self.lambda))+max(self.lambda)*norm(self.D)^2);
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
   
    end
end