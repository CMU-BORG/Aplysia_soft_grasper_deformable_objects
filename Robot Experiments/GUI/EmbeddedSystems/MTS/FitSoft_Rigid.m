clc
clear all 
close all

%%
T_soft = readtable('Soft_Rigid_ForMATLAB_Raw.xlsx','Sheet','SoftGrasper_Raw');
T_rigid = readtable('Soft_Rigid_ForMATLAB_Raw.xlsx','Sheet','RigidGrasper_Raw');



softNames = {"F1_N","F2_N","F3_N"};
rigidNames ={"AxialF_1","AxialF_2","AxialF_3"};

for k = 1:length(softNames)

    %Convert all forces to positive
    T_soft{:,softNames{k}} = abs(T_soft{:,softNames{k}});
    T_rigid{:,rigidNames{k}} = abs(T_rigid{:,rigidNames{k}});

    %Adjust for weight set on top of sensor
    T_soft{:,softNames{k}+"_jaw2"} = T_soft{:,softNames{k}}+(0.088*9.81); %adjusted for additional weight for third trial
    T_rigid{:,rigidNames{k}+"_adjusted"} = T_rigid{:,rigidNames{k}}+(0.088*9.81); %adjusted for additional weight for third trial

end

%% Plot and fit for Soft Grasper Pressure vs. Force
PressureNames1 = {"P1_1_psi","P1_2_psi","P1_3_psi"};
ForceNames1 = {"F1_N","F2_N","F3_N"};
PressureNames2 = {"P2_1_psi","P2_2_psi","P2_3_psi"};
ForceNames2 = {"F1_N_jaw2","F2_N_jaw2","F3_N_jaw2"};

figure()
xlabel("MTS Force (N)")
ylabel("Pressure ([psi])")
hold on
legend()

for k = 1:length(softNames)

    xd = T_soft{:,ForceNames1{k}};
    yd = T_soft{:,PressureNames1{k}};
    plot(xd,yd,'r-.');
    p = polyfit(xd,yd,4);
    yfit = polyval(p,xd);
    plot(xd,yfit,'b-','LineWidth',2,'DisplayName',sprintf('Jaw 1 fit exp %i',k));
    disp(sprintf("Fit for Soft experiment %i jaw 1",k));
    disp(vpa(p,6))
    
    xd = T_soft{:,ForceNames2{k}};
    yd = T_soft{:,PressureNames2{k}};
    plot(xd,yd,'r--');
    p = polyfit(xd,yd,4);
    yfit = polyval(p,xd);
    plot(xd,yfit,'b-','LineWidth',2,'DisplayName',sprintf('Jaw 2 fit exp %i',k));
    disp(sprintf("Fit for Soft experiment %i jaw 2",k));
    disp(vpa(p,6))

end

%% Plot and fit for rigid grasper vs force 
SensorNames1 = {"SensorF_1","SensorF_2","SensorF_3"};
ForceNames1 = {"AxialF_1","AxialF_2","AxialF_3"};

figure()
xlabel("Sensor Reading")
ylabel("MTS Force (N)")
hold on

for k = 1:length(SensorNames1)

    xd = T_rigid{:,SensorNames1{k}};
    yd = T_rigid{:,ForceNames1{k}};
    plot(xd,yd,'r-.');
    p = polyfit(xd,yd,1);
    yfit = polyval(p,xd);
    plot(xd,yfit,'b-','LineWidth',2,'DisplayName','Rigid fit');
    disp(sprintf("Fit for rigid experiment %i",k));
    disp(vpa(p,6))

end

%% Find the equivalent gains
syms F_feedback K_rigid K_soft F F_th P P_th Pval p1 p2 p3 p4 p5 f1 f2 f3 f4 f5

Feedback_R = K_rigid*(F - F_th); %Feedback for Rigid
Feedback_S =  K_soft*(P - P_th); %Feedback for Soft
P_Fit = p1*F^4 + p2*F^3 + p3*F^2 +p4*F +p5; %polynomial fit
F_fit = f1*Pval^4+ f2*Pval^3 + f3*Pval^2 + f4*Pval +f5; %polynomial fit

F_fit_threshold = F_th-subs(F_fit,Pval, P_th); %the equivalent force threshold for a given pressure threshold

eqs = [Feedback_R - subs(Feedback_S,P,P_Fit);
       Feedback_R-F_feedback ];

soln = solve(eqs(1),[K_rigid]);

soln2 = solve(eqs,[K_rigid,K_soft]);

mag_K_rigid_fn = matlabFunction(soln,'Vars',[p1 p2 p3 p4 p5 K_soft F F_th P_th] );
mag_K_rigid = mag_K_rigid_fn(-0.0037, 0.0392, -0.1140, 0.2033, 0.0422, 100, 0.5, 0, 0);

mag_K_rigid_fn2 = matlabFunction([soln2.K_rigid,soln2.K_soft],'Vars',[p1 p2 p3 p4 p5 F_feedback F F_th P_th] );
mag_K_rigid2 = mag_K_rigid_fn2(-0.0037, 0.0392, -0.1140, 0.2033, 0.0422, 2, 0.5, 0, 0);












