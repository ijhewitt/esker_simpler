function [td,ad,ud] = esker_simpler(td,ad,pd,ud,oo)
% Solves time-dependent discretised channel equations for [S,phi,Qs,A] over timespan td
% Inputs
%   td  timespan to solve for, with solution stored at intermediate times
%   ad  struct containing topography and input fields
%   pd  struct containing parameters
%   ud  struct containing initial conditions ?
%   oo  struct containing options
% Outputs
%   td  times at which output provided
%   ad  struct containing topography and input fields (may be expanded from
%       what was provided as input)
%   ud  struct containing solution variables
% If oo.nondiminput = 1, all inputs are interpreted as non-dimensional and ouputs are non-dimensional
%
% IJH 10 May 2026 - edited from canal_simpler.m to reproduce esker model

    % OPTIONS
    if nargin<5, oo = struct; end
    if ~isfield(oo,'nondiminput') oo.nondiminput = 1; end % default to non-dimensional inputs

    % SCALE
    if oo.nondiminput
        tt = td;
        aa = ad;
        pp = pd;
    else
        [aa,pp,pd] = non_dimensionalise(ad,pd); 
        tt = td/pd.t; 
    end

    % DISCRETIZATION
    dd = discretize(aa.x);
    
    % FILL IN MISSING FIELDS AND PARAMETERS WITH DEFAULTS   
    if ~isfield(aa,'xi_m'), aa.xi_m = []; end                   % x indices of moulins
    if ~isfield(aa,'S_m'), aa.S_m = 0*aa.xi_m.^0; end            % moulin areas
    if ~isfield(aa,'Q_m'), aa.Q_m = @(t) 0*aa.xi_m.^0*t; end     % moulin inputs 
    if ~isfield(aa,'M_in'), aa.M_in = @(t) 0*aa.x.^0*t; end     % channel source
    if ~isfield(aa,'Qs_m'), aa.Qs_m = @(t) 0*aa.xi_m.^0*t; end     % moulin sediment inputs 
    if ~isfield(aa,'Ms_in'), aa.Ms_in = @(t) 0*aa.x.^0*t; end     % channel sediment source
    if ~isfield(pp,'phi_m'), pp.phi_m = max(pp.r*aa.Z_b(end),pp.Z_0); end % boundary potential 
    if ~isfield(pp,'Q_0'), pp.Q_0 = 0; end                         % boundary discharge
    if ~isfield(pp,'Qs_0'), pp.Qs_0 = 0; end                        % boundary sediment flux
    if ~isfield(pp,'eps38'), pp.eps38 = 1e-4; end % dimensionless coefficient of compressibility S*dphi/dt in mass conservation equation
    if ~isfield(pp,'eps53'), pp.eps53 = 1e-4; end % dimensionless coefficient of dQs/dt in sediment conservation equation
    if ~isfield(pp,'A_min'), pp.A_min = 0; end  % dimensionless minimum deposit area for erosion

    % DIMENSIONLESS DERIVED OR PRESCRIBED FIELDS
    aa.phi_s = aa.Z_s + (pp.r-1)*aa.Z_b;  % overburden potential
    aa.phi_b = pp.r*aa.Z_b;               % atmospheric potential
    aa.psi_s = -dd.ddxx*aa.phi_s;         % overburden potential gradient

    % INITIAL CONDITION
    if isempty(ud)
        [S,phi,Qs,A] = initial(td(1),aa,dd,pp);
    else
        if oo.nondiminput
            S = ud(end).S; phi = ud(end).phi; Qs = ud(end).Qs; A = ud(end).A;
        else
            S = ud(end).S/pd.S; phi = ud(end).phi/pd.phi; Qs = ud(end).Qs/pd.Qs; A = ud(end).A/pd.A;
        end
    end
    Y = pack(S,phi,Qs,A); 
            
    % SOLVE
    disp('Solving ...');
    mass = @(t,Y) fun_mass(Y,aa,dd,pp);     
    opts = odeset('Mass',mass,'MvPattern',fun_mvpattern(dd),'JPattern',fun_jpattern(dd),'Vectorized','on','reltol',1e-6,'abstol',1e-4); % ode15s options [ reltol 1e-4 abstol 1e-4 ]
    [tt,YY] = ode15s(@fun_F,tt,Y,opts);
    tt = tt'; YY = YY';

    % EXTRACT VARIABLES
    for ti = 1:length(tt)
        uui = struct;
        [uui.S,uui.phi,uui.Q,uui.A,uui.N,uui.Qeq,uui.Qs,uui.m,uui.m_s] = unpack_full(YY(:,ti),aa,dd,pp);
        if ti==1 
            uu = uui; 
        else 
            uu(ti) = uui; 
        end
    end

    % UNSCALE
    if oo.nondiminput
        td = tt;
        ad = aa;
        ud = uu;
    else
        td = pd.t*tt;
        ad = unscale_aa(aa,pd);
        ud = unscale_uu(uu,pd);
    end
    
    disp('Done');
    
    
%% SUBFUNCTIONS
function F = fun_F(t,Y)
% right hand side of equations 
    [S,phi,Q,A,N,Qeq,Qs,m,m_s] = unpack_full(Y,aa,dd,pp);

    % source terms
    ext = ones(1,size(Y,2));
    Q_in = 0*dd.x; Q_in(aa.xi_m) = aa.Q_m(t);   % assign moulins to Q_in vector
    M_in = aa.M_in(t);
    Qs_in = 0*dd.x; Qs_in(aa.xi_m) = aa.Qs_m(t);  % assign sediment flux at moulins
    Ms_in = aa.Ms_in(t);
    
    % channel area evolution
    F1 = pp.upsilon*max(0,1-S/pp.upsilon2) + pp.mu_i*m + pp.mu_s*m_s - pp.omega_i*S.^(2*pp.gamma).*abs(N).^(pp.n-1).*N - pp.omega_s*S.^(2*pp.gamma).*abs(N).^(pp.n_sed-1).*N; 
    % water mass conservation
    F2 = -dd.ddx*Q + pp.epsilon/pp.r*pp.mu_i*m + (Q_in.*dd.dx.^(-1))*ext + M_in*ext;
    % sediment conservation
    F3 = -dd.ddx*Qs + m_s + (Qs_in.*dd.dx.^(-1))*ext + Ms_in*ext; 
     % eroded/deposited area
    F4 = -pp.mu_s*m_s; 
    
    F = [F1(dd.xin,:); F2(dd.xin,:); F3(dd.xin,:); F4(dd.xin,:)];
end

function mass = fun_mass(Y,aa,dd,pp)
% mass matrix for equations
    xin = dd.xin; Iin = length(xin);
    S_in = 0*dd.x; S_in(aa.xi_m) = aa.S_m;  % assign moulins to S_in vector
    S = NaN(dd.I,1); S(xin) = Y(1:Iin); % unpack S from solution vector
    
    mass = sparse([1:Iin                Iin+(1:Iin)          Iin+(1:Iin)       2*Iin+(1:Iin)        3*Iin+(1:Iin) ], ...
                   [1:Iin                1:Iin                Iin+(1:Iin)          2*Iin+(1:Iin)        3*Iin+(1:Iin) ], ...
                    [ones(1,Iin)    pp.epsilon*ones(1,Iin)   pp.sigma*ones(1,Iin)+pp.eps38*S(xin)'+pp.beta*S_in(xin)'.*dd.dx(xin)'.^(-1)    pp.eps53*ones(1,Iin)  ones(1,Iin)], ...
                     4*Iin , 4*Iin );
end

function [aa,pp,pd] = non_dimensionalise(ad,pd)
% non-dimensionalise inputs using scales in pd or as defined below

% FILL IN MISSING SCALES
if ~isfield(pd,'Q'), pd.Q = 10; end                   % water flux scale, m^3/s
if ~isfield(pd,'x'), pd.x = 5000; end                   % length scale, m
if ~isfield(pd,'z'), pd.z = 500; end                    % depth scale, m
if ~isfield(pd,'phi'), pd.phi = pd.rho_i*pd.g*pd.z; end    % hydropotential scale, Pa
if ~isfield(pd,'psi'), pd.psi = pd.phi/pd.x; end          % hydropotential gradient scale, Pa/m
if ~isfield(pd,'S'), pd.S = (pd.Q/pd.k_c/pd.psi^(1/2))^(1/pd.alpha); end % channel area scale, m^2
if ~isfield(pd,'m'), pd.m = pd.Q*pd.psi/pd.rho_w/pd.L; end % channel melt rate scale, m^2/s 
if ~isfield(pd,'U'), pd.U = pd.Q/pd.S; end              % water speed scale, m/s
if ~isfield(pd,'Qs'), pd.Qs = pd.k_s*pd.S^pd.gamma*( 1/8*pd.f*pd.rho_w*pd.U^2/((pd.rho_s-pd.rho_w)*pd.g*pd.D_50) ).^(3/2); end % sediment flux scale, m^3/s
if ~isfield(pd,'m_s'), pd.m_s = pd.Qs/pd.x; end            % erosion rate scale, m^2/s
if ~isfield(pd,'t'), pd.t = 24*60*60; end             % timescale, s
if ~isfield(pd,'A'), pd.A = pd.S; end                  % deposit area scale, m^2

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
pp.beta = pd.rho_i/pd.rho_w*pd.S*pd.z/pd.Q/pd.t; % earlier typo had pd.z/pd.x;
pp.sigma = pd.rho_i/pd.rho_w*pd.sigma*pd.l_c*pd.z*pd.x/pd.Q/pd.t; % earlier typo had pd.sigma*pd.l_c*pd.z/pd.S;
pp.tau_s = pd.tau_s*((pd.rho_s-pd.rho_w)*pd.g*pd.D_50)/(1/8*pd.f*pd.rho_w*pd.U^2);
pp.Z_0 = pd.Z_0/pd.z;
if isfield(pd,'A_min'), pp.A_min = pd.A_min/pd.A; end

% DIMENSIONLESS GEOMETRY AND INPUTS
aa.x = ad.x/pd.x;
aa.Z_b = ad.Z_b/pd.z;
aa.Z_s = ad.Z_s/pd.z;
if isfield(ad,'xi_m'), aa.xi_m = ad.xi_m; end
if isfield(ad,'S_m'), aa.S_m = ad.S_m/pd.S; end
if isfield(ad,'Q_m'), aa.Q_m = @(t) ad.Q_m(pd.t*t)/pd.Q; end
if isfield(ad,'M_in'), aa.M_in = @(t) ad.M_in(pd.t*t)/(pd.Q/pd.x); end
if isfield(ad,'Qs_m'), aa.Qs_m = @(t) ad.Qs_m(pd.t*t)/pd.Qs; end
if isfield(ad,'Ms_in'), aa.Ms_in = @(t) ad.Ms_in(pd.t*t)/(pd.Qs/pd.x); end

end

function ad = unscale_aa(aa,ps)
% unscale prescribed field struct aa using scales in ps
    ad.x = ps.x*aa.x;
    ad.Z_b = ps.z*aa.Z_b;
    ad.Z_s = ps.z*aa.Z_s;
    ad.xi_m = aa.xi_m;
    ad.Q_m = @(t) ps.Q*aa.Q_m(t/ps.t);
    ad.M_in = @(t) (ps.Q/ps.x)*aa.M_in(t/ps.t);
    ad.Qs_m = @(t) ps.Qs*aa.Qs_m(t/ps.t);
    ad.Ms_in = @(t) (ps.Qs/ps.x)*aa.Ms_in(t/ps.t);
    
    ad.phi_s = ps.phi*aa.phi_s;
    ad.phi_b = ps.phi*aa.phi_b;
    ad.psi_s = ps.phi/ps.x*aa.psi_s;
end

function ud = unscale_uu(uu,ps)
% unscale solution struct uu using scales in ps
    ud = uu;
    for i = 1:length(uu)
        ud(i).S = ps.S*uu(i).S;
        ud(i).phi = ps.phi*uu(i).phi;
        ud(i).Q = ps.Q*uu(i).Q;
        ud(i).A = ps.A*uu(i).A;
        ud(i).N = ps.phi*uu(i).N;
        ud(i).Qeq = ps.Qs*uu(i).Qeq;
        ud(i).Qs = ps.Qs*uu(i).Qs;
        ud(i).m = ps.m*uu(i).m;
        ud(i).m_s = ps.m_s*uu(i).m_s;
    end
end

function dd = discretize(x)
% produce struct of discretization details

    % grid point positions and spacing
    I = length(x);
    dd.I = I;
    dd.x = x; % positions of grid points (nodes)
    dd.xx = [dd.x(1); (dd.x(1:I-1)+dd.x(2:I))/2]; % positions of staggered grid points (edges), first edge the same as first node
    dd.dx = [dd.xx(2:I); dd.x(end)]-dd.xx; % spacing between edges
    dd.dxx = dd.x(1:I)-[dd.x(1); dd.x(1:I-1)]; % spacing between nodes, first entry zero
    % x(1)   |    x(2)   |    x(3)    ...      |    x(I)
    % xx(1) xx(2)  .    xx(3)   .     ...     xx(I)   .
    % -dx(1)- ---dx(2)--- ---dx(3)--- ...       -dx(I)-
    %  ---dxx(2)--- ---dxx(3)--- ---  ...     dxx(I)---
    
    % matrix derivative and averaging operators
    dd.ddx = sparse([1:I 1:I-1],[1:I 1+(1:I-1)],[-dd.dx.^(-1); dd.dx(1:I-1).^(-1)],I,I); % divergence operator, gives derivative on nodes of quantity defined on edges
    dd.ddxx = sparse([2:I 1:I],[(2:I)-1 (1:I)],[-dd.dxx(2:I).^(-1); dd.dxx.^(-1)],I,I); dd.ddxx(isinf(dd.ddxx)) = 0; % gradient operator, gives derivative on edges of quantity defined on nodes
    dd.ddxx = sparse([1:I 1:I],[1 (2:I)-1 2 (2:I)],[-dd.dxx([2 2:I]).^(-1); dd.dxx([2 2:I]).^(-1)],I,I); % gradient operator, gives derivative on edges of quantity defined on nodes [ with first entry duplicated from second entry ]
    dd.avx = sparse([1:I 1:I-1],[1:I 1+(1:I-1)],[0.5*ones(I-1,1); 1; 0.5*ones(I-1,1)],I,I); % averaging operator on nodes
    dd.avxx = sparse([2:I 1:I],[(2:I)-1 (1:I)],[0.5*ones(I-1,1); 1; 0.5*ones(I-1,1)],I,I); % averaging operator on edges

    % indices
    dd.xext = I;                        % Dirichlet nodes (prescribed phi)
    dd.xxext = 1;                       % Neumann edges (prescribed flux)
    dd.xin = setdiff(1:I,dd.xext);      % intetior nodes
    dd.xxin = setdiff(1:I,dd.xxext);    % interior edges
    
    % averaging operators restricted to interior nodes and edges
    tmp = dd.avx(:,dd.xxin)*ones(length(dd.xxin),1); 
    dd.avxin = spdiags(tmp.^(-1),0,I,I)*dd.avx; dd.avxin(:,dd.xxext) = 0; % averaging operator on nodes using interior edges
    tmp = dd.avxx(:,dd.xin)*ones(length(dd.xin),1); 
    dd.avxxin = spdiags(tmp.^(-1),0,I,I)*dd.avxx; dd.avxxin(:,dd.xext) = 0; % averaging operator on edges using interior nodes
end

function mvpattern = fun_mvpattern(dd)
% sparsity pattern for mass matrix
    Iin = length(dd.xin);
    tmp = speye(Iin);
    mvpattern = [tmp 0*tmp 0*tmp 0*tmp; tmp tmp 0*tmp 0*tmp; 0*tmp tmp tmp 0*tmp; 0*tmp 0*tmp 0*tmp tmp];
end

function jpattern = fun_jpattern(dd)
% sparsity pattern for jacobian of equations (conservative estimate - it is
% really more sparse than this)
    Iin = length(dd.xin);
    tmp = sparse([1:Iin 1:Iin-1 2:Iin],[1:Iin 2:Iin 1:Iin-1],ones(1,3*Iin-2),Iin,Iin);
    jpattern = [tmp tmp 0*tmp 0*tmp; tmp tmp 0*tmp 0*tmp; tmp tmp tmp tmp; tmp tmp tmp tmp]; 
end

function [S,phi,Qs,A] = initial(t,aa,dd,pp)
% estimate for initial condition using potential phi_s and prescribed inputs
    Smin = 1e-3;
    Q_in = 0*dd.x; Q_in(aa.xi_m) = aa.Q_m(t);   % assign moulins to Q_in vector
    M_in = aa.M_in(t);
    Qs_in = 0*dd.x; Qs_in(aa.xi_m) = aa.Qs_m(t);   % assign moulins to Q_in vector
    Ms_in = aa.Ms_in(t);
    Q = cumsum(Q_in+(M_in.*dd.dx)); % approximate discharge by integrating source terms 
    phi = aa.phi_s;
    dphi = dd.ddxx*phi;
    S = abs(-Q./(max(abs(dphi),eps).^((1/2-1)).*dphi)).^(1/pp.alpha);
    S = max(S,Smin);
    Qs = cumsum(Qs_in+(Ms_in.*dd.dx)); % approximate sediment flux by integrating source terms 
    A = 0*dd.x;
end

function Y = pack(S,phi,Qs,A)
% pack together variables into solution vector Y
    Y = [S(dd.xin); phi(dd.xin); Qs(dd.xxin); A(dd.xin)];
end

function [S,phi,Qs,A] = unpack(Y,aa,dd)
% unpack variables from solution vector Y
    xin = dd.xin;
    xxin = dd.xxin;
    Iin = length(xin);
    ext = ones(1,size(Y,2));
    
    % inialize primary variables, including boundary values
    S = aa.S_ext*ext;
    phi = aa.phi_ext*ext;
    Qs = aa.Qs_ext*ext;
    A = aa.A_ext*ext;
    
    % assign interior values from solution vector
    tmp1 = 1; tmp2 = tmp1 + length(xin)-1;
    S(xin,:) = Y(tmp1:tmp2,:); tmp1 = tmp2+1; tmp2 = tmp1+Iin-1;
    phi(xin,:) = Y(tmp1:tmp2,:); tmp1 = tmp2+1; tmp2 = tmp1+Iin-1;
    Qs(xxin,:) = Y(tmp1:tmp2,:); tmp1 = tmp2+1; tmp2 = tmp1+Iin-1;
    A(xin,:) = Y(tmp1:tmp2,:);
end

function [S,phi,Q,A,N,Qeq,Qs,m,m_s] = unpack_full(Y,aa,dd,pp)
% unpack variables and derived functions from solution vector Y

    % indices of interior nodes and edges
    xin = dd.xin;
    xxin = dd.xxin;
    Iin = length(xin);
    ext = ones(1,size(Y,2));
    
    % inialize primary variables, including boundary conditions
    % S = aa.S_ext*ext;
    % phi = aa.phi_ext*ext;
    % Qs = aa.Qs_ext*ext;
    % A = AA_ext*ext;
    S = NaN(dd.I,1)*ext;
    phi = NaN(dd.I,1)*ext;
    Qs = NaN(dd.I,1)*ext;
    A = NaN(dd.I,1)*ext;
    phi(dd.xext,:) = pp.phi_m*ext;
    Qs(dd.xxext,:) = pp.Qs_0*ext; 
    
    % assign interior values from solution vector
    tmp1 = 1; tmp2 = tmp1 + length(xin)-1;
    S(xin,:) = Y(tmp1:tmp2,:); tmp1 = tmp2+1; tmp2 = tmp1+Iin-1;
    phi(xin,:) = Y(tmp1:tmp2,:); tmp1 = tmp2+1; tmp2 = tmp1+Iin-1;
    Qs(xxin,:) = Y(tmp1:tmp2,:); tmp1 = tmp2+1; tmp2 = tmp1+Iin-1;
    A(xin,:) = Y(tmp1:tmp2,:);

    % discharge on edges
    Q = NaN*ext; 
    Q(dd.xxext,:) = pp.Q_0*ext;
    Sav = dd.avxxin*S;      % average S on edges
    dphi = dd.ddxx*phi;     % potential gradient
    tmp = -Sav.^pp.alpha.*max(abs(dphi),eps).^((1/2-1)).*dphi;
    Q(xxin,:) = tmp(xxin,:);
    
    % melt rate on nodes
    m = dd.avxin*(-Q.*dphi);
    
    % effective pressure on nodes
    N = aa.phi_s*ext - phi;
    
    % equilibrium sediment transport on edges
    U = Q./Sav; 
    Qeq = Sav.^(pp.gamma).*max(U.^2-pp.tau_s,0).^(3/2);  
    
    % erosion rate on nodes
    m_s = 0*dd.x.^0*ext;
    tmp = -dd.avxin*(Qs-Qeq)/pp.nu; % averaging over neighouring edges
    % tmp = -(Qs-Qeq)/pp.nu; % value from upstream edge
    % tmp = -(Qs-Qeq)/pp.nu; tmp = tmp([2:end end],:); % value from upstream edge (seems to work better in some cases) 
    ind = (tmp<0 | A>pp.A_min); m_s(ind) = tmp(ind); 
end

end