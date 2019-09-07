%% Simulator - MLP with backpropagation learning on XOR problem
%% Modified by Andrew Rice and Nayan Thapa
%% Version 10: 2019 - Adapted from the literature
%% Main Sources:
%% [1] Trappenberg 
%% [2] Dayan and Abbot
%% [3] Churchland and Sejnowski
%% [4] Rolls and Deco
%% [5] Izhikevich
%% [6] Wilson
%% [7] Gerstner

clear all;
close all;
hold off; 
clc;

 
 N_i=12*13; N_h=2; N_o=26;
 w_h=rand(N_h,N_i)-0.5; w_o=rand(N_o,N_h)-0.5;

 % training vectors (XOR)
 
  nIn=12*13; nOut=26;
 wOut=rand(nOut,nIn)-0.5; 
 
% training vectors 
 load pattern1;
 
 % Adding the noisy noise 
%  xmin=0.01;
% xmax=1;
% n=1;
t = 0.01;
 pattern1 = imnoise(pattern1,'salt & pepper',t);
 r_i=reshape(pattern1', N_i, N_o); 
 r_d=diag(ones(1,N_o));

 % Updating and training network with sigmoid activation function
 for sweep=1:80;
     
%     pattern1 = imnoise(pattern1,'salt & pepper',t);
%     imshow(pattern1);
    
    %r_i=reshape(pattern1', N_i, N_o); 
%     t  = t+0.01
%     t=xmin+rand(1,n)*(xmax-xmin);
%     noise(sweep)=t; 
     
   % training randomly on one pattern
     i=ceil(4*rand);
     r_h=1./(1+exp(-w_h*r_i(:,i)));
     r_o=1./(1+exp(-w_o*r_h));
     d_o=(r_o.*(1-r_o)).*(r_d(:,i)-r_o);
     d_h=(r_h.*(1-r_h)).*(w_o'*d_o);
     w_o=w_o+5*(r_h*d_o')';
     w_h=w_h+5*(r_i(:,i)*d_h')';
   % test all pattern
     r_o_test=1./(1+exp(-w_o*(1./(1+exp(-w_h*r_i)))));
     distH=sum(sum((r_o_test-r_d)))/nOut;
     d(sweep)=distH;
     
     % Average recognition rate
      test = sum(sum(r_d))-sum(sum(w_o));
      ttt(sweep) = (test/sum(sum(r_d)));

  end
%  subplot(2,1,1);
%  plot(d);
%  xlabel('Training Steps'); ylabel('Training Error');
%  subplot(2,1,2);
%  plot(noise,d);
 hold on;
 xlabel('Noise'); ylabel('Average Recognition Rate');
 

 nIn=12*13; nOut=26;
 wOut=rand(nOut,nIn)-0.5; 
 
% training vectors 
 load pattern1;
 rIn=reshape(pattern1', nIn, nOut); 
 rDes=diag(ones(1,nOut));
 
% Updating and training network 
 for training_step=1:80;
     % test all pattern
      rOut=(wOut*rIn)>0.5;
      distH=sum(sum((rDes-rOut).^2))/nOut;
      error1(training_step)=distH;
     % training with delta rule
      wOut=wOut+0.1*(rDes-rOut)*rIn';  
      
 end
 
%  figure;
%  plot(0:79,error)
%  xlabel('Training step')
%  ylabel('Average Hamming distance')
 
 
 %% 2
 nIn=12*13; nOut=26;
% training vectors 
 load pattern1;
 %pattern1 = rand(312,13);
%  imshow(pattern1);
 t = 0.01;
 pattern1 = imnoise(pattern1,'salt & pepper',t);
%  imshow(pattern1);
 
 rIn=reshape(pattern1', nIn, nOut); 
 rDes1=diag(ones(1,nOut));
 
% Updating and training network 
 for training_step=1:80
     
     pattern1 = imnoise(pattern1,'salt & pepper',t);
%     imshow(pattern1);
 
    rIn=reshape(pattern1', nIn, nOut); 
    
    t = t + 0.01;
    noise(training_step)=t; 
     % test all pattern
      rOut=(wOut*rIn)>0.5;
      distH=sum(sum((rDes1-rOut).^2))/nOut;
      error(training_step)=distH;

      % Average recognition rate
%       imshow(pattern1);
      test = sum(sum(rDes1))-sum(sum(rOut));
      ttt(training_step) = (test/sum(sum(rDes1)));
 end

 figure;
 plot(d);
 hold on;
 xlabel('Training step'); ylabel('Training Error');
 
%   plot(error);
  plot(0:79,error1)
   xlabel('Training step'); ylabel('Average Recognition Rate');
   legend('MLP','Single Layer');
%  xlabel('Noise')
%  ylabel('Average recognition rate')
%  plot(0:19,error)
%  xlabel('Training step')
%  ylabel('Average Hammering distance'
 
 
