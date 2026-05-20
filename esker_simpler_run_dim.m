% CLEAR WORKSPACE
set(0,'DefaultAxesFontsize',12,'DefaultTextFontsize',12,'DefaultTextInterpreter','tex');
clear

% OPTIONS
oo.nondiminput = 0;     % use dimenionless inputs/outputs

% PARAMETERS
pd.rho_i = 916;         % ice density, kg/m^3
pd.rho_w = 1000;        % water density, kg/m^3
pd.g = 9.8;            % gravity, m/s^2
pd.f = 0.2;             % Darch-Weisbach friction coefficient (NB this is Darcy-Weisbach friction coefficient; previous versions have had a differnt f which is one eigth of this one
pd.L = 3.3e5;           % ice latent heat, J/kg
pd.ty = 365*24*60*60;   % one year, s
pd.l_eq = 100;          % equilibrium length for sediment transport, m
pd.rho_s = 2600;        % sediment density, kg/m^3
pd.D_50 = 1e-3;         % sediment median grain size, m
pd.n_s = 0.4;           % sediment deposit porosity, dimensionless
pd.tau_s = 0.047;       % critical Shields stress, dimensionless
pd.n = 3;               % ice flow-law exponent
%pd.A_i = 2.4e-24;      % ice flow-law coefficient, Pa^(-3) s^(-1) [Cuffey & Paterson]
pd.A_i = 6.8e-24*(pd.n^pd.n/2); % to give \tilde{A} = 6.8e-24
pd.n_sed = 1;           % sediment flow-law epxonent
pd.A_s = 0;             % sediment flow-law coefficient, Pa^(-1) s^(-1)
pd.Z_0 = 0;             % sea level, m
pd.h_r = 0.1;           % bed roughness height, m
pd.l_r = 1;             % bed roughness length, m
pd.u_b = 30/pd.ty;      % ice sliding speed, m/s
pd.l_c = 10e3;          % catchment width, m
pd.sigma = 0;           % void fraction, dimensionless
pd.alpha = 5/4;         % channel flow parametersation exponent
pd.k_c = 2*2^(1/4)*pi^(1/4)/(pi+2)^(1/2)/pd.rho_w^(1/2)/pd.f^(1/2); % channel flow parameterisation coefficient, m^(3/2)/kg^(1/2)
pd.gamma = 1/2;
pd.Atilde_i = 2*pd.A_i/pd.n^pd.n; % modified ice flow-law coefficient
pd.Atilde_s = 2*pd.A_s/pd.n^pd.n; % modified sediment flow-law coefficient
pd.k_s = 8*((pd.rho_s-pd.rho_w)/pd.rho_w)^(1/2)*pd.g^(1/2)*pd.D_50^(3/2)*(8/pi)^(1/2); % equilibrium sediment flux pre-factor
pd.A_min = 0;           % minimum deposit area for erosion, m^2

% DIMENSIONAL GEOMETRY AND INPUTS
ad.x = linspace(-50e3,0,400)';                  % grid pts, m
ad.Z_b = 0 -400/(pd.rho_w/pd.rho_i)+ 0*(ad.x);  % bed elevation, m
ad.Z_s = ad.Z_b + ( 400 + 0.01*(-ad.x) );       % surface elevation, m
ad.M_in = @(t) (10)/(ad.x(end)-ad.x(1))*ad.x.^0*t.^0;    % channel source, m^2/s
ad.Ms_in = @(t) 0*ad.M_in(t);                   % channel sediment source, m^2/s
ad.xi_m = 1;                                    % index of moulin
ad.S_m = 0;                                     % moulin cross-sectional area
ad.Q_m = @(t) 0*10*t.^0;                          % moulin input, m^3/s
ad.Qs_m = @(t) 0*t.^0;                          % moulin sediment input, m^3/s

% RUN TOWARDS STEADY STATE WITH NO SEDIMENT TRANSPORT
rem = pd.tau_s; pd.tau_s = inf;
td = [0:.1:20]*(pd.ty/365); 
[td,ad,ud] = esker_simpler(td,ad,pd,[],oo);

% figure(2); clf;
%     plot(ad.x,[ud.S]);
%     xlabel('x'); ylabel('S');
%     drawnow; shg;

% calculate expected equilibirum sediment transport
pd.tau_s = rem;
Q_app = ud(end).Q;
S_app = (Q_app./pd.k_c./max(abs(ad.psi_s),eps).^((1/2-1))./ad.psi_s).^(1/pd.alpha);
Qs_app = pd.k_s*S_app.^(pd.gamma).*max(1/8*pd.f*pd.rho_w/((pd.rho_s-pd.rho_w)*pd.g*pd.D_50)*Q_app.^2./S_app.^2 - pd.tau_s,0).^(3/2);
Qs0 = Qs_app(2);  % moulin input consistent with equilibrium transport
Qs0 = 0;

% TURN ON SEDIMENT TRANSPORT AND RE-RUN
ad.Qs_m = @(t) Qs0*t.^0;  
ad.Ms_in = @(t) 1e-3*ad.M_in(t);   
td = [0:.1:50]*(pd.ty/365);
[td,ad,ud] = esker_simpler(td,ad,pd,ud(end),oo);

% figure(2); clf;
%     plot(ad.x,[ud.S]);
%     xlabel('x'); ylabel('S');
%     drawnow; shg;
% 
% figure(3); clf;
%     tmp = imagesc(1/1e3*ad.x,1/pd.ty*365*td,[ud.S]'); set(gca,'YDir','normal');
%     colorbar;
%     xlabel('x'); ylabel('t');
%     drawnow; shg;

% PLOT SOLUTION
ti = length(ud);

figure(1); clf;
% set(gcf,'Paperpositionmode','auto','units','centimeters','position',[2 4 20 30]);
xlimits = [min(ad.x) max(ad.x)]*1/1e3;
ax0 = axes('position',[0.15 0.95 0.75 0.04]);
ax1 = axes('position',[0.15 0.78 0.75 0.14]);
ax2 = axes('position',[0.15 0.59 0.75 0.14]);
ax3 = axes('position',[0.15 0.42 0.75 0.14]);
ax4 = axes('position',[0.15 0.25 0.75 0.14]);
ax5 = axes('position',[0.15 0.08 0.75 0.14]);

% discharge
axes(ax0);
    plot(1/1e3*ad.x,[ud(ti).Q],'b','linewidth',2); hold on; 
    xlim(xlimits); 
    % ylim([0 25]);
    ax0.XTickLabel = {}; 
    ylabel('$Q$ [m${}^3$/s]','Interpreter','latex','fontsize',12);

% hydraulic potential
axes(ax1);
    plot(1/1e3*ad.x,ad.Z_b,'k',1/1e3*ad.x,ad.Z_s,'k','linewidth',2); hold on; % bed and surface elevation
    plot(1/1e3*ad.x,ad.phi_s/pd.rho_w/pd.g,'k--','linewidth',2);              % overburdent potential as head
    plot(1/1e3*ad.x,ud(ti).phi/pd.rho_w/pd.g,'b','linewidth',2);              % hydraulic potential as head
    xlim(xlimits); 
    % ylim([-500 1200]);
%     ax1.XTickLabel = {}; 
    ylabel('Elevation, $z$ [m]','Interpreter','latex','fontsize',12);

% effective pressure
axes(ax2);
    plot(1/1e3*ad.x,(ad.phi_s-ud(ti).phi)/(pd.rho_w*pd.g),'b','linewidth',2);
    hold on;
    xlim(xlimits); % ylim([0 10]);
    ax2.XTickLabel = {}; 
    ylabel('$N$ [m w.e.]','Interpreter','latex','fontsize',12);
    
% channel area
axes(ax3);
    plot(1/1e3*ad.x,[ud(ti).S],'b','linewidth',2); hold on;
    xlim(xlimits); 
    % ylim([0 20]);
    ax3.XTickLabel = {}; 
    ylabel('$S$ [m${}^2$]','Interpreter','latex','fontsize',12);

% sediment flux
axes(ax4);
    plot(1/1e3*ad.x,ud(ti).Qeq,'b--','linewidth',2); hold on;
    plot(1/1e3*ad.x,ud(ti).Qs,'b-','linewidth',2); hold on; 
    xlim(xlimits); 
    % ylim([0 0.15]);
    ax4.XTickLabel = {}; 
    ylabel('$Q_{s}$ [m${}^3$/s]','Interpreter','latex','fontsize',12);

% deposit area
axes(ax5);
    plot(1/1e3*ad.x,[ud(ti).A],'b','linewidth',2); hold on;
    xlim(xlimits); 
    % ylim([0 20]);
    ylabel('$A$ [m${}^2$]','Interpreter','latex','fontsize',12);
    xlabel('Distance, $x-x_m$ [km]','Interpreter','latex','fontsize',12); 
    
    linkaxes([ax5 ax4 ax3 ax2 ax1 ax0],'x');

exportgraphics(gcf,[mfilename,'_fig1.pdf'],'ContentType','vector');


% SPACE TIME PLOT
figure(2); clf; 
% set(gcf,'Paperpositionmode','auto','units','centimeters','position',[2 4 16 10]);
ax1 = axes('position',[0.1 0.6 0.35 0.35]);
ax2 = axes('position',[0.55 0.6 0.35 0.35]);
ax3 = axes('position',[0.1 0.1 0.35 0.35]);
ax4 = axes('position',[0.55 0.1 0.35 0.35]);
tax = 1/pd.ty*365*[min(td) max(td)];
xax = [min(1/1e3*ad.x) max(1/1e3*ad.x)];

axes(ax1);
    tmp = imagesc(1/1e3*ad.x,1/pd.ty*365*td,1/1e6*[ud.N]'); set(gca,'YDir','normal'); 
    c = colorbar; title(c,'$N$ [MPa]','interpreter','latex'); 
    box on;
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

axes(ax2);
    tmp = imagesc(1/1e3*ad.x,1/pd.ty*365*td,[ud.S]'); set(gca,'YDir','normal');
    c = colorbar; title(c,'$S$ [m${}^2$]','interpreter','latex');
    box on;
    % clim([0 20]);
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

axes(ax3);
    tmp = imagesc(1/1e3*ad.x,1/pd.ty*365*td,[ud.Qs]'); set(gca,'YDir','normal');
    c = colorbar; title(c,'$Q_s$ [m${}^3$/s]','interpreter','latex');
    box on;
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

axes(ax4);
    tmp = imagesc(1/1e3*ad.x,1/pd.ty*365*td,(pd.ty/365)*[ud.m_s]'/(1-pd.n_s)); set(gca,'YDir','normal');
    c = colorbar; title(c,'$E$ [m${}^2$/d]','interpreter','latex');
    box on;
    clim([-1 1]); load cmapbwr; colormap(gca,cmap);
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

    linkaxes([ax1 ax2 ax3 ax4]);
    xlim(xax); ylim(tax);

exportgraphics(gcf,[mfilename,'_fig2.pdf']);