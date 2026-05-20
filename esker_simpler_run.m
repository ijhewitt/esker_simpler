% 10 Apr 2026 to run canal_simpler

% CLEAR WORKSPACE
set(0,'DefaultAxesFontsize',12,'DefaultTextFontsize',12,'DefaultTextInterpreter','tex');
clear

% OPTIONS
oo.nondiminput = 1;     % use dimenionless inputs/outputs

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

% SCALING
pd.Q = 10;                      % water flux scale, m^3/s
pd.x = 5000;                    % length scale, m
pd.z = 500;                     % depth scale, m
pd.phi = pd.rho_i*pd.g*pd.z;    % hydropotential scale, Pa
% pd.x = ( pd.Q^(1-2*pd.gamma/pd.alpha)/pd.Atilde_i/pd.rho_i/pd.L*pd.phi^(1-pd.n+2*pd.gamma/2/pd.alpha)*pd.k_c^(2*pd.gamma/pd.alpha) ).^(1/(1+2*pd.gamma/2/pd.alpha)); % length scale, m (natural scale for boundary layer)
pd.psi = pd.phi/pd.x;           % hydropotential gradient scale, Pa/m
pd.S = (pd.Q/pd.k_c/pd.psi^(1/2))^(1/pd.alpha); % channel area scale, m^2
pd.m = pd.Q*pd.psi/pd.rho_w/pd.L; % channel melt rate scale, m^2/s 
pd.U = pd.Q/pd.S;               % water speed scale, m/s
pd.Qs = pd.k_s*pd.S^pd.gamma*( 1/8*pd.f*pd.rho_w*pd.U^2/((pd.rho_s-pd.rho_w)*pd.g*pd.D_50) ).^(3/2); % sediment flux scale, m^3/s
pd.m_s = pd.Qs/pd.x;             % erosion rate scale, m^2/s
pd.t = pd.ty/365;               % time scale, s
% pd.t = pd.S*pd.rho_i*pd.L/pd.Q/pd.psi; % time scale, s (natural choice)
pd.A = pd.S;                    % deposit area scale, m^2

% DIMENSIONLESS PARAMETERS
pp.alpha = pd.alpha;
pp.n = pd.n;
pp.n_sed = pd.n_sed;
pp.gamma = pd.gamma;
pp.r = pd.rho_w/pd.rho_i;
pp.epsilon = pd.S*pd.x/pd.Q/pd.t;
pp.nu = pd.l_eq/pd.x;
pp.mu_i = pd.Q*pd.psi/pd.rho_i/pd.L/pd.S*pd.t;
pp.mu_s = pd.m_s/(1-pd.n_s)/pd.S*pd.t;
pp.omega_i = pd.Atilde_i*pd.S^(2*pd.gamma)*pd.phi.^pd.n/pd.S*pd.t;
pp.omega_s = pd.Atilde_s*pd.S^(2*pd.gamma)*pd.phi.^pd.n_sed/pd.S*pd.t;
pp.upsilon = pd.h_r*pd.u_b/pd.S*pd.t;
pp.upsilon2 = pd.h_r*pd.l_r/pd.S;
pp.beta = pd.rho_i/pd.rho_w*pd.S*pd.z/pd.Q/pd.t;
pp.sigma = pd.rho_i/pd.rho_w*pd.sigma*pd.l_c*pd.z*pd.x/pd.Q/pd.t;
pp.tau_s = pd.tau_s*((pd.rho_s-pd.rho_w)*pd.g*pd.D_50)/(1/8*pd.f*pd.rho_w*pd.U^2);
pp.Z_0 = pd.Z_0/pd.z;
pp.A_min = pd.A_min/pd.A;

% REGULARISATION PARAMETERS
pp.eps38 = 1e-4; % dimensionless coefficient of compressibility S*dphi/dt in mass conservation equation
pp.eps53 = 1e-4; % dimensionless coefficient of dQs/dt in sediment conservation equation

% EDIT DIMENSIONLESS PARAMETERS
pp.epsilon = 0;             % remove contribution of melting to mass conservation

% DIMENSIONLESS GEOMETRY AND INPUTS
aa.x = linspace(-50e3,0,400)'/pd.x;                     % grid pts
aa.Z_b = 0/pd.z -400/pp.r/pd.z+ 0*(aa.x);               % bed elevation
aa.Z_s = aa.Z_b + ( 400 + 0.01*(-pd.x*aa.x) )/pd.z;     % surface elevation
aa.M_in = @(t) (10/pd.Q)/(aa.x(end)-aa.x(1))*aa.x.^0*t.^0;    % channel source
aa.Ms_in = @(t) 0*aa.M_in(t);                  % channel sediment source
aa.xi_m = 1;                                   % index of moulin
aa.S_m = 0;                                    % moulin cross-sectional area
aa.Q_m = @(t) 0*10/pd.Q*t.^0;                    % moulin input
aa.Qs_m = @(t) 0*t.^0;                         % moulin sediment input

% RUN TOWARDS STEADY STATE WITH NO SEDIMENT TRANSPORT
rem = pp.tau_s; pp.tau_s = inf;
td = [0:.1:10]*(pd.ty/365)/pd.t; 
[td,ad,ud] = esker_simpler(td,aa,pp,[],oo);

% figure(2); clf;
%     plot(ad.x,[ud.S]);
%     xlabel('x'); ylabel('S');
%     drawnow; shg;

% calculate expected equilibirum sediment transport
pp.tau_s = rem;
Q_app = ud(end).Q;
S_app = (Q_app./max(abs(ad.psi_s),eps).^((1/2-1))./ad.psi_s).^(1/pp.alpha);
Qs_app = S_app.^pp.gamma.*max(Q_app.^2./S_app.^2-pp.tau_s,0).^(3/2); 
Qs0 = Qs_app(2);  % moulin input consistent with equilibrium transport
Qs0 = 0;

% TURN ON SEDIMENT AND RE-RUN
aa.Qs_m = @(t) Qs0*t.^0;  
aa.Ms_in = @(t) 1e-1*aa.M_in(t);   
td = [0:.1:50]*(pd.ty/365)/pd.t; 
[td,ad,ud] = esker_simpler(td,aa,pp,ud(end),oo);

% figure(2); clf;
%     plot(ad.x,[ud.S]);
%     xlabel('x'); ylabel('S');
%     drawnow; shg;
% 
% figure(3); clf;
%     tmp = imagesc(pd.x/1e3*ad.x,pd.t/pd.ty*365*td,pd.S*[ud.S]'); set(gca,'YDir','normal');
%     colorbar;
%     xlabel('x'); ylabel('t');
%     drawnow; shg;

% PLOT SOLUTION
ti = length(ud);

figure(1); clf;
% set(gcf,'Paperpositionmode','auto','units','centimeters','position',[2 4 20 30]);
ax0 = axes('position',[0.15 0.95 0.75 0.04]);
ax1 = axes('position',[0.15 0.78 0.75 0.14]);
ax2 = axes('position',[0.15 0.59 0.75 0.14]);
ax3 = axes('position',[0.15 0.42 0.75 0.14]);
ax4 = axes('position',[0.15 0.25 0.75 0.14]);
ax5 = axes('position',[0.15 0.08 0.75 0.14]);
xax = [min(ad.x) max(ad.x)]*pd.x/1e3;

% discharge
axes(ax0);
    plot(pd.x/1e3*ad.x,pd.Q*[ud(ti).Q],'b','linewidth',2); hold on; 
    xlim(xax); 
    % ylim([0 25]);
    ax0.XTickLabel = {}; 
    ylabel('$Q$ [m${}^3$/s]','Interpreter','latex','fontsize',12);

% hydraulic potential
axes(ax1);
    plot(pd.x/1e3*ad.x,pd.z*ad.Z_b,'k',pd.x/1e3*ad.x,pd.z*ad.Z_s,'k','linewidth',2); hold on; % bed and surface elevation
    plot(pd.x/1e3*ad.x,pd.phi*ad.phi_s/pd.rho_w/pd.g,'k--','linewidth',2);              % overburdent potential as head
    plot(pd.x/1e3*ad.x,pd.phi*ud(ti).phi/pd.rho_w/pd.g,'b','linewidth',2);              % hydraulic potential as head
    xlim(xax); 
    % ylim([-500 1200]);
%     ax1.XTickLabel = {}; 
    ylabel('Elevation, $z$ [m]','Interpreter','latex','fontsize',12);

% effective pressure
axes(ax2);
    plot(pd.x/1e3*ad.x,pd.phi*(ad.phi_s-ud(ti).phi)/(pd.rho_w*pd.g),'b','linewidth',2);
    hold on;
    xlim(xax); % ylim([0 10]);
    ax2.XTickLabel = {}; 
    ylabel('$N$ [m w.e.]','Interpreter','latex','fontsize',12);
    
% channel area
axes(ax3);
    plot(pd.x/1e3*ad.x,pd.S*[ud(ti).S],'b','linewidth',2); hold on;
    xlim(xax); 
    % ylim([0 20]);
    ax3.XTickLabel = {}; 
    ylabel('$S$ [m${}^2$]','Interpreter','latex','fontsize',12);

% sediment flux
axes(ax4);
    plot(pd.x/1e3*ad.x,pd.Qs*ud(ti).Qeq,'b--','linewidth',2); hold on;
    plot(pd.x/1e3*ad.x,pd.Qs*ud(ti).Qs,'b-','linewidth',2); hold on; 
    xlim(xax); 
    % ylim([0 0.15]);
    ax4.XTickLabel = {}; 
    ylabel('$Q_{s}$ [m${}^3$/s]','Interpreter','latex','fontsize',12);
    
% deposit area
axes(ax5);
    plot(pd.x/1e3*ad.x,pd.A*[ud(ti).A],'b','linewidth',2); hold on;
    xlim(xax); 
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
tax = pd.t/pd.ty*365*[min(td) max(td)];
xax = [min(pd.x/1e3*ad.x) max(pd.x/1e3*ad.x)];

axes(ax1);
    tmp = imagesc(pd.x/1e3*ad.x,pd.t/pd.ty*365*td,pd.phi/1e6*[ud.N]'); set(gca,'YDir','normal'); 
    c = colorbar; title(c,'$N$ [MPa]','interpreter','latex'); 
    box on;
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

axes(ax2);
    tmp = imagesc(pd.x/1e3*ad.x,pd.t/pd.ty*365*td,pd.S*[ud.S]'); set(gca,'YDir','normal');
    c = colorbar; title(c,'$S$ [m${}^2$]','interpreter','latex');
    box on;
    % clim([0 20]);
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

axes(ax3);
    tmp = imagesc(pd.x/1e3*ad.x,pd.t/pd.ty*365*td,pd.Qs*[ud.Qs]'); set(gca,'YDir','normal');
    c = colorbar; title(c,'$Q_s$ [m${}^3$/s]','interpreter','latex');
    box on;
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

axes(ax4);
    tmp = imagesc(pd.x/1e3*ad.x,pd.t/pd.ty*365*td,(pd.ty/365)*(pd.S/pd.t)*pp.mu_s*[ud.m_s]'); set(gca,'YDir','normal');
    c = colorbar; title(c,'$E$ [m${}^2$/d]','interpreter','latex');
    box on;
    clim([-1 1]); load cmapbwr; colormap(gca,cmap);
    xlabel('$x$ [km]','Interpreter','latex','fontsize',12); ylabel('$t$ [d]','Interpreter','latex','fontsize',12);

    linkaxes([ax1 ax2 ax3 ax4]);
    xlim(xax); ylim(tax);

exportgraphics(gcf,[mfilename,'_fig2.pdf']);