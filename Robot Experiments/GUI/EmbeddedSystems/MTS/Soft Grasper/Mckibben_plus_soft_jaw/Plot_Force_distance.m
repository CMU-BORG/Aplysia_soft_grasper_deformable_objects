close all

figure('Position',[100,100,390,390*2])
ax1= subplot(2,1,1)
xlabel("Displacement (mm)")
ylabel("Force (N)")
legend();
grid on
hold on

%figure('Position',[100,100,390,390])
ax2= subplot(2,1,2)
xlabel("Displacement (mm)")
ylabel("Young's Modulus (Pa)")
legend('Location','BestOutside');
yscale log
grid on
hold on

%% Define moduli functions
youngs_modulus = @(p_tangent,delta,mean_force)polyval(p_tangent,delta).*(1-(0.5.^2))./(0.0125); %E = P*(1-vu^2)/(2*a*delta)
youngs_modulus_diff = @(p_tangent,delta,mean_force) (diff(mean_force)./diff(delta.')*(1-(0.5.^2))./(0.0125)); %E = 4*P*36/(pi*delta*d^2)

youngs_modulus_playdoh = @(p_tangent,delta,mean_force) (polyval(p_tangent,delta)*(4*36/1000)./(pi*(50/1000)^2)); %E = 4*P*36/(pi*delta*d^2)
youngs_modulus_playdoh_diff = @(p_tangent,delta,mean_force) (diff(mean_force)./diff(delta.')*(4*36/1000)./(pi*(50/1000)^2)); %E = 4*P*36/(pi*delta*d^2)

%% Rigid
% distance = [0       0.098       0.192       0.259       0.299        0.35       0.373       0.391       0.428       0.458       0.515       0.575       0.609       0.636       0.663       0.687        0.71       0.731       0.744       0.761       0.777       0.791       0.841       0.889       0.966       1.033       1.101       1.155       1.212        1.28] ;
% 
% force = [0         0.05        0.094        0.153        0.206        0.308        0.366        0.403          0.5        0.509        0.755        1.022        1.253        1.493        1.745        2.028        2.551        3.042        3.462        4.028        4.579        5.008        7.441        10.04        15.01        20.02         25.4         30.1         34.3         40.5];
% 
% plot(ax1,distance,force,"p",'LineStyle','-', 'MarkerSize',5, 'LineWidth',2,'Color','#009E73','DisplayName','Rigid')


% p = polyfit(distance/1000,force,2);
% py = polyval(p,distance/1000);
% plot(ax1,distance,py,'k-')
% 
% p_tangent = p(1:2).*(2:-1:1);  %for a quadratic f(x), the slope of the tangent is df(x)/dx
% 
% %semilogy(ax2,distance,youngs_modulus(p_tangent,distance./1000, force),markers,'LineStyle','-','MarkerSize',5, 'LineWidth',2,'Color','#009E73','DisplayName','Rigid');
% 
% semilogy(ax2,distance(1:end-1),youngs_modulus_diff(p_tangent,distance.'./1000, force),'p','LineStyle','-','MarkerSize',5, 'LineWidth',2,'Color','#009E73','DisplayName','Rigid');


%% All others


fnames = {"Feb23_6p943psi_pivot.csv",
          "Feb23_7p969psi_pivot.csv",
          "Feb23_9p493psi_pivot.csv",
          "Feb23_noMcKibbens_pivot.csv",
          "Feb23_rigidplate_on_McKibbens_6p943psi_pivot.csv",
          "Feb23_rigidplate_on_McKibbens_7p969psi_pivot.csv",
          "Feb23_rigidplate_on_McKibbens_9p434psi_pivot.csv",
          "PlayDoh_Force_compression_Stiff3.xlsx",
          "Rigid_Oct26th_2024.xlsx"};

use_diff = {false,false,false,true,false,false,false,false,false};
pd = {2,2,2,2,2,2,2,6,9};
fcns = {youngs_modulus,youngs_modulus,youngs_modulus,youngs_modulus_diff,youngs_modulus,youngs_modulus,youngs_modulus,youngs_modulus_playdoh,youngs_modulus};

displaynames = {"5 mm", "10 mm", "15 mm", "No McKibbens","5 mm rigid", "10 mm rigid", "15 mm rigid", "Play Doh",'Rigid'}

colors ={"#000000","#000000","#000000" "#E69F00","#56B4E9","#56B4E9","#56B4E9","#D55E00", "#009E73" };
markers = {"^","^","^","s","o","o","o","diamond","p"};
linestyle = {"--","-.",":","-","--","-.",":","-","-"};

fits = [];
youngs_modulus_0mm = [];
youngs_modulus_5mm = [];

for fn = 1:length(fnames)

    DF = readtable(fnames{fn});
    p = interpolate_plot(DF,colors{fn},markers{fn},displaynames{fn},linestyle{fn},ax1,ax2, fcns{fn}, use_diff{fn}, pd{fn});

    % p_tangent = p(1:2);
    % p_tangent(1) = p_tangent(1)*2; %for a quadratic f(x), the slope of the tangent is df(x)/dx
    % youngs_modulus = @(delta)polyval(p_tangent,delta).*(1-(0.5.^2))./(0.0125); %E = P*(1-vu^2)/(2*a*delta)
    % 
    % 
    % youngs_modulus_0mm(fn) = youngs_modulus(0);
    % youngs_modulus_5mm(fn) = youngs_modulus(5/1000);
    % fits{fn} = p;
    


end



xlim(ax1,[0,7.5])
ylim(ax1,[0,inf]);
xlim(ax2,[0,7.5])
ylim(ax2,[0,inf]);


%%estimate deformation of object
delta_o = @(Ei,Eo,zo1,zg1,delta_Ri) (delta_Ri*(Ei*zo1)/(Eo*zg1))/(1+((Ei*zo1)/(Eo*zg1)));

function [p] = interpolate_plot(DF,colors, markers,dfname,lstyle,ax1,ax2,youngs_modulus,use_diff, pd)

    DF = table2array(DF);
    force_idx = [2,5,8];
    displacement_idx = [3,6,9];
    n = length(force_idx);

    max_dist = min(max(DF(:,displacement_idx)),[],2);
    ix = [0:0.5:max_dist]; %interpolated distance
    i_force = zeros(length(ix),3); %interpolated force
   

    for jj = 1:n
        i_force(:,jj) = interpn(DF(:,displacement_idx(jj)),DF(:,force_idx(jj)),ix,'spline');
    end

    mean_force = mean(i_force,2);
    std_force = std(i_force,[],2);

    min_force =mean_force-std_force;
    max_force = mean_force + std_force;
    %errorbar(ix,-mean_force,std_force,'DisplayName',dfname);
    fill(ax1,[ix';flip(ix)'],[-min_force;-flip(max_force)],'w','FaceColor',colors,'FaceAlpha',0.15,'EdgeColor','None', 'DisplayName','');
    plot(ax1,ix,-mean_force,markers,'LineStyle',lstyle,'MarkerSize',5, 'LineWidth',2,'Color',colors,'DisplayName',dfname);
    



    %% For the Effective Young's Modulus
    p = polyfit(ix/1000,-mean_force,pd);
    py = polyval(p,ix/1000);
    plot(ax1,ix,py,'k-')

    p_tangent = p(1:pd).*(pd:-1:1);  %for a quadratic f(x), the slope of the tangent is df(x)/dx
    

    
    %semilogy(ax2,ix(1:end-1),estimated_modulus,markers,'LineStyle',lstyle,'MarkerSize',5, 'LineWidth',2,'Color',colors,'DisplayName',dfname);
    if use_diff == false
        semilogy(ax2,ix,youngs_modulus(p_tangent,ix./1000,-mean_force),markers,'LineStyle',lstyle,'MarkerSize',5, 'LineWidth',2,'Color',colors,'DisplayName',dfname);
    else
        semilogy(ax2,ix(1:end-1),youngs_modulus(p_tangent,ix./1000,-mean_force),markers,'LineStyle',lstyle,'MarkerSize',5, 'LineWidth',2,'Color',colors,'DisplayName',dfname);
    end
   



end