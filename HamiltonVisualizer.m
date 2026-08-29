function HamiltonVisualizer
%% HamiltonVisualizer — NP/SC 哈密顿量矩阵可视化
% 横排布局: 上NP参数+SC参数, 下NP矩阵+SC矩阵
% 格点≤8显示文字, >8纯热力图
% 上限20格点, 超过回退
clear; close all;

%% ===== 全局状态 =====
state = struct();
state.NX_np=2; state.NY_np=2; state.NX_sc=5; state.NY_sc=2;
state.t=1; state.phi=pi/4; state.omg=1; state.tc=1;
state.symbolic=true; state.maxSites=36;
state.half_np=false; state.half_sc=true;  % NP默认无, SC默认有半元胞
state.sc_order='cell';  % SC排序: 'cell'=元胞列优先, 'row'=行优先
state.np_order='cell';  % NP排序: 'cell'=元胞列优先, 'row'=行优先
state.boundary='semi';  % 'open'=双开边界, 'semi'=半无限(x-Bloch) 默认半无限
state.kx=0;  % 半无限模式的x-Bloch波矢
state.lastNX_np=2; state.lastNY_np=2; state.lastNX_sc=5; state.lastNY_sc=2;

%% ===== GUI =====
fig = figure('Name','Hamiltonian Visualizer — NP vs SC',...
    'NumberTitle','off','Units','normalized',...
    'Position',[0.02 0.05 0.95 0.88],'Color',[0.94 0.94 0.94],...
    'Resize','on');

% 矩阵显示轴 (左) + 晶格图 (右)
ax_np = axes('Parent',fig,'Units','normalized',...
    'Position',[0.03 0.02 0.28 0.55],'Color',[1 1 1],...
    'XTick',[],'YTick',[],'Box','on');
ax_np_lat = axes('Parent',fig,'Units','normalized',...
    'Position',[0.32 0.02 0.17 0.55],'Color',[1 1 1],...
    'XTick',[],'YTick',[],'Box','on');
ax_sc = axes('Parent',fig,'Units','normalized',...
    'Position',[0.52 0.02 0.28 0.55],'Color',[1 1 1],...
    'XTick',[],'YTick',[],'Box','on');
ax_sc_lat = axes('Parent',fig,'Units','normalized',...
    'Position',[0.81 0.02 0.17 0.55],'Color',[1 1 1],...
    'XTick',[],'YTick',[],'Box','on');

% 切换按钮
uicontrol(fig,'Style','togglebutton','String',{'数值与符号切换'},...
    'Units','normalized','Position',[0.39 0.955 0.10 0.03],...
    'Value',state.symbolic,'BackgroundColor',[0.85 0.85 1.0],...
    'FontSize',11,'FontWeight','bold',...
    'Callback',@(src,~) toggle_mode(src));
uicontrol(fig,'Style','togglebutton','String',{'边界: 双开 ↔ 半无限'},...
    'Units','normalized','Position',[0.50 0.955 0.12 0.03],...
    'Value',true,'BackgroundColor',[0.85 1.0 0.85],...
    'FontSize',10,'FontWeight','bold',...
    'Callback',@(src,~) toggle_boundary(src));

% 创建NP参数 (左上)
p_np = create_param_panel(fig,'NP (4-site cell, full fill)',...
    0.03,0.63,0.46,0.30,'np');
p_sc = create_param_panel(fig,'SC (checkerboard, 2-site cell)',...
    0.52,0.63,0.46,0.30,'sc');

handles = struct('fig',fig,'ax_np',ax_np,'ax_np_lat',ax_np_lat,...
    'ax_sc',ax_sc,'ax_sc_lat',ax_sc_lat,'p_np',p_np,'p_sc',p_sc);
guidata(fig,handles);
setappdata(fig,'state',state);

% 初始绘制
redraw(fig);

%% ===== 创建参数面板 (嵌套) =====
function p = create_param_panel(fig,ttl,x0,y0,w0,h0,tagSfx)
    fs = 10; fs_big = 12;
    p = struct();
    p.pnl = uipanel(fig,'Title',ttl,'Units','normalized',...
        'Position',[x0 y0 w0 h0],'BackgroundColor',[0.92 0.92 0.92],...
        'FontSize',fs_big,'FontWeight','bold');

    % NX行
    y = 0.72;
    make_row('NX',1,8,y0,y,1,tagSfx);
    % NY行
    y = y - 0.12;
    make_row('NY',1,8,y0,y,1,tagSfx);
    % t行
    y = y - 0.12;
    make_row('t',0.1,3,y0,y,0.1,tagSfx);
    % phi行
    y = y - 0.12;
    make_row('φ/π',0.05,1,y0,y,0.01,tagSfx);
    % kx行 (半无限x-Bloch波矢)
    y = y - 0.12;
    make_row('kx/π',-1,1,y0,y,0.01,tagSfx);

    % 半元胞复选框
    y = y - 0.12;
    halfDefault = state.(['half_' tagSfx]);
    p.chk_half = uicontrol(p.pnl,'Style','checkbox',...
        'String','半元胞 (右+上)',...
        'Units','normalized','Position',[0.08 y+0.02 0.42 0.10],...
        'Value',halfDefault,'BackgroundColor',[0.92 0.92 0.92],...
        'FontSize',fs,'Callback',@(src,~) half_changed(src,tagSfx));

    % 元胞优先复选框 (NP + SC共用)
    p.chk_order = uicontrol(p.pnl,'Style','checkbox',...
        'String','元胞优先',...
        'Units','normalized','Position',[0.52 y+0.02 0.22 0.10],...
        'Value',true,'BackgroundColor',[0.92 0.92 0.92],...
        'FontSize',fs,'Callback',@(src,~) order_changed(src,tagSfx));

    % 信息栏
    y = y - 0.12;
    p.txt_info = uicontrol(p.pnl,'Style','text',...
        'String','Cells: --  Sites: --',...
        'Units','normalized','Position',[0.08 y+0.01 0.84 0.10],...
        'BackgroundColor',[1 1 0.85],'FontSize',fs_big,'FontWeight','bold');

    % 嵌套: 创建标签+编辑+滑块行
    function make_row(label,mn,mx,y0_pnl,yp,step,tagSfx)
        if contains(label,'kx')
            tag = ['kx' tagSfx];   % kx控件tag: kxn/kxsc
        else
            tag = [regexprep(label,'[\\/]','') tagSfx];
        end
        % 标签 (左)
        uicontrol(p.pnl,'Style','text','String',label,...
            'Units','normalized','Position',[0.04 yp+0.02 0.16 0.09],...
            'BackgroundColor',[0.92 0.92 0.92],...
            'FontSize',fs+2,'FontWeight','bold');
        % 编辑框 (中)
        if contains(label,'phi')
            str0 = sprintf('%.3f',state.(['phi'])/pi);
        elseif contains(label,'kx')
            str0 = sprintf('%.3f',state.(['kx'])/pi);
        else
            val0 = state.(['t']);
            str0 = sprintf('%.2f',val0);
            if contains(label,'NX')||contains(label,'NY')
                str0 = sprintf('%d',round(mn));
            end
        end
        % Override for NX/NY
        if contains(label,'NX')
            str0 = sprintf('%d',state.(['NX_' tagSfx]));
        elseif contains(label,'NY')
            str0 = sprintf('%d',state.(['NY_' tagSfx]));
        end
        edt = uicontrol(p.pnl,'Style','edit','String',str0,...
            'Units','normalized','Position',[0.22 yp 0.18 0.12],...
            'Tag',['edt_' tag],'FontSize',fs+1,...
            'Callback',@(src,~) edit_changed(src,tag));
        % 滑块 (右)
        sl_step = [step/(mx-mn), step*5/(mx-mn)];
        initVal = str2double(str0);
        uicontrol(p.pnl,'Style','slider','Min',mn,'Max',mx,'Value',initVal,...
            'Units','normalized','Position',[0.42 yp+0.01 0.54 0.10],...
            'Tag',['sld_' tag],'SliderStep',sl_step,...
            'Callback',@(src,~) slider_changed(src,tag));
    end
end

%% ===== 回调函数 =====
function slider_changed(src,tag)
    fig = ancestor(src,'figure');
    val = get(src,'Value');
    isInt = contains(tag,'NX')||contains(tag,'NY');
    if isInt, val = round(val); end
    edt = findobj(fig,'Tag',['edt_' tag]);
    if ~isempty(edt)
        if isInt, set(edt,'String',sprintf('%d',val));
        elseif contains(tag,'phi')||contains(tag,'kx'), set(edt,'String',sprintf('%.3f',val));
        else, set(edt,'String',sprintf('%.2f',val)); end
    end
    param_changed(fig,tag,val);
end

function edit_changed(src,tag)
    fig = ancestor(src,'figure');
    val = str2double(get(src,'String'));
    if isnan(val), return; end
    isInt = contains(tag,'NX')||contains(tag,'NY');
    sld = findobj(fig,'Tag',['sld_' tag]);
    if ~isempty(sld)
        mn=get(sld,'Min'); mx=get(sld,'Max');
        val = max(mn,min(mx,val));
        if isInt, val=round(val); end
        set(sld,'Value',val);
    end
    param_changed(fig,tag,val);
end

function param_changed(fig,tag,val)
    st = getappdata(fig,'state');
    handles = guidata(fig);

    % 更新状态
    if contains(tag,'kx')
        st.kx = val*pi;   % kx为公共参数, 两面板共用
    elseif endsWith(tag,'np')
        if contains(tag,'NX'), st.NX_np=round(val);
        elseif contains(tag,'NY'), st.NY_np=round(val);
        elseif contains(tag,'t'), st.t=val;
        elseif contains(tag,'phi'), st.phi=val*pi; end
    elseif endsWith(tag,'sc')
        if contains(tag,'NX'), st.NX_sc=round(val);
        elseif contains(tag,'NY'), st.NY_sc=round(val);
        elseif contains(tag,'t'), st.t=val;
        elseif contains(tag,'phi'), st.phi=val*pi; end
    end

    % NP格点上限检查
    ns_np = count_sites(st,'np');
    if ns_np > st.maxSites
        st.NX_np=st.lastNX_np; st.NY_np=st.lastNY_np;
        restore_ctrl(fig,'NXnp',st.NX_np); restore_ctrl(fig,'NYnp',st.NY_np);
    else, st.lastNX_np=st.NX_np; st.lastNY_np=st.NY_np; end

    % SC格点上限检查
    ns_sc = count_sites(st,'sc');
    if ns_sc > st.maxSites
        st.NX_sc=st.lastNX_sc; st.NY_sc=st.lastNY_sc;
        restore_ctrl(fig,'NXsc',st.NX_sc); restore_ctrl(fig,'NYsc',st.NY_sc);
    else, st.lastNX_sc=st.NX_sc; st.lastNY_sc=st.NY_sc; end

    setappdata(fig,'state',st);
    redraw(fig);
end

function restore_ctrl(fig,tag,val)
    edt=findobj(fig,'Tag',['edt_' tag]); sld=findobj(fig,'Tag',['sld_' tag]);
    if ~isempty(edt)
        if contains(tag,'NX')||contains(tag,'NY'), set(edt,'String',sprintf('%d',val));
        else, set(edt,'String',sprintf('%.2f',val)); end
    end
    if ~isempty(sld), set(sld,'Value',val); end
end

function restore_all_ctrls(fig,st)
    restore_ctrl(fig,'NXnp',st.NX_np); restore_ctrl(fig,'NYnp',st.NY_np);
    restore_ctrl(fig,'NXsc',st.NX_sc); restore_ctrl(fig,'NYsc',st.NY_sc);
end

function half_changed(src,tagSfx)
    fig = ancestor(src,'figure');
    st = getappdata(fig,'state');
    newHalf = get(src,'Value');
    st.(['half_' tagSfx]) = newHalf;
    % 检查格点上限, 超限则缩减NX/NY
    ns = count_sites(st,tagSfx);
    while ns > st.maxSites
        if strcmp(tagSfx,'np')
            if st.NX_np>1, st.NX_np=st.NX_np-1; elseif st.NY_np>1, st.NY_np=st.NY_np-1; else break; end
        else
            if st.NX_sc>1, st.NX_sc=st.NX_sc-1; elseif st.NY_sc>1, st.NY_sc=st.NY_sc-1; else break; end
        end
        ns = count_sites(st,tagSfx);
    end
    if ns > st.maxSites
        % 缩减无效, 回退
        st.(['half_' tagSfx]) = ~newHalf;
        set(src,'Value',st.(['half_' tagSfx]));
    end
    restore_all_ctrls(fig,st);
    setappdata(fig,'state',st);
    redraw(fig);
end

function order_changed(src,tagSfx)
    fig = ancestor(src,'figure');
    st = getappdata(fig,'state');
    orderVal = iif(get(src,'Value'),'cell','row');
    if strcmp(tagSfx,'sc')
        st.sc_order = orderVal;
    else
        st.np_order = orderVal;
    end
    setappdata(fig,'state',st);
    redraw(fig);
end

function toggle_mode(src)
    fig = ancestor(src,'figure');
    st = getappdata(fig,'state');
    st.symbolic = get(src,'Value');
    setappdata(fig,'state',st);
    redraw(fig);
end

function toggle_boundary(src)
    fig = ancestor(src,'figure');
    st = getappdata(fig,'state');
    if get(src,'Value')
        st.boundary = 'semi';
        set(src,'String','边界: 半无限 ↔ 双开');
    else
        st.boundary = 'open';
        set(src,'String','边界: 双开 ↔ 半无限');
    end
    setappdata(fig,'state',st);
    try
        redraw(fig);
    catch e
        fprintf(2,'ERROR in redraw: %s (line %d)\n',e.message,e.stack(1).line);
        set(src,'Value',~get(src,'Value'));  % 回退
    end
end

function ns = count_sites(st,tagSfx)
    if strcmp(st.boundary,'semi')
        % 半无限: 仅NY决定格点数, NX不参与
        if strcmp(tagSfx,'np')
            ns = 4*st.NY_np + 2*st.half_np;  % 4格点/元胞 + 半元胞2格点
        else
            ns = 2*st.NY_sc + 1*st.half_sc;  % checkerboard: ~2格点/2行
        end
        return;
    end
    if strcmp(tagSfx,'np')
        if st.half_np
            Lx=2*st.NX_np+1; Ly=2*st.NY_np+1;
        else
            Lx=2*st.NX_np; Ly=2*st.NY_np;
        end
        ns = Lx*Ly;
    else
        if st.half_sc
            Lx=2*st.NX_sc+1; Ly=2*st.NY_sc+1;
        else
            Lx=2*st.NX_sc; Ly=2*st.NY_sc;
        end
        ns=0;
        for y=0:Ly-1, for x=0:Lx-1
            if (mod(x,2)==1&&mod(y,2)==0)||(mod(x,2)==0&&mod(y,2)==1), ns=ns+1; end
        end, end
    end
end

%% ===== 主重绘 =====
function redraw(fig)
    handles = guidata(fig); st = getappdata(fig,'state');
    isSemi = strcmp(st.boundary,'semi');

    % 同步公共参数 (t, phi)
    sync_ctrl(fig,'tnp',st.t,false);
    sync_ctrl(fig,'phipinp',st.phi/pi,false);
    sync_ctrl(fig,'tsc',st.t,false);
    sync_ctrl(fig,'phipisc',st.phi/pi,false);

    % 半无限: 灰掉NX(双开启用), 灰掉kx(半无限启用)
    set_nx_enable(fig,'NXnp',~isSemi,'—');
    set_nx_enable(fig,'NXsc',~isSemi,'—');
    set_nx_enable(fig,'kxn',isSemi,'—');
    set_nx_enable(fig,'kxsc',isSemi,'—');
    % kx同步(公共参数)
    sync_ctrl(fig,'kxn',st.kx/pi,false);
    sync_ctrl(fig,'kxsc',st.kx/pi,false);

    % 更新复选框状态
    set(handles.p_np.chk_order,'Value',strcmp(st.np_order,'cell'));
    set(handles.p_sc.chk_order,'Value',strcmp(st.sc_order,'cell'));

    % 计算哈密顿量
    if isSemi
        % 半无限: x-Bloch(kx), y有限 (仅NY控制)
        kx = st.kx;
        if st.symbolic
            % 符号模式: 用符号参数t,φ,ω,tc,kx重建, 纯符号无数值
            try
                st_sym = st;
                st_sym.t = sym('t'); st_sym.phi = sym('phi');
                st_sym.omg = sym('omg'); st_sym.tc = sym('tc');
                [Hrib_np,info_np] = build_H_np_ribbon(st_sym,st.half_np,st.np_order);
                [Hrib_sc,info_sc] = build_H_sc_ribbon(st_sym,st.half_sc,st.sc_order);
                kx_s = sym('kx');
                H_np = Hrib_np.H0 + Hrib_np.H1*exp(1i*kx_s) + Hrib_np.H1'*exp(-1i*kx_s);
                H_sc = Hrib_sc.H0 + Hrib_sc.H1*exp(1i*kx_s) + Hrib_sc.H1'*exp(-1i*kx_s);
            catch
                [Hrib_np,info_np] = build_H_np_ribbon(st,st.half_np,st.np_order);
                [Hrib_sc,info_sc] = build_H_sc_ribbon(st,st.half_sc,st.sc_order);
                H_np = Hrib_np.H0 + Hrib_np.H1*exp(1i*kx) + Hrib_np.H1'*exp(-1i*kx);
                H_sc = Hrib_sc.H0 + Hrib_sc.H1*exp(1i*kx) + Hrib_sc.H1'*exp(-1i*kx);
            end
        else
            [Hrib_np,info_np] = build_H_np_ribbon(st,st.half_np,st.np_order);
            [Hrib_sc,info_sc] = build_H_sc_ribbon(st,st.half_sc,st.sc_order);
            H_np = Hrib_np.H0 + Hrib_np.H1*exp(1i*kx) + Hrib_np.H1'*exp(-1i*kx);
            H_sc = Hrib_sc.H0 + Hrib_sc.H1*exp(1i*kx) + Hrib_sc.H1'*exp(-1i*kx);
        end
        info_np.Ncells = st.NY_np;
        info_sc.Ncells = st.NY_sc;
    else
        [H_np,info_np] = build_H_np(st,st.half_np,st.np_order);
        [H_sc,info_sc] = build_H_sc(st,st.half_sc,st.sc_order);
    end

    % 更新信息栏
    boundStr = iif(isSemi,'[半无限]','[双开]');
    set(handles.p_np.txt_info,'String',...
        sprintf('%s NY=%d | Cells=%d | Sites=%d',boundStr,...
        st.NY_np,info_np.Ncells,info_np.Nsites));
    set(handles.p_sc.txt_info,'String',...
        sprintf('%s NY=%d | Cells=%d | Sites=%d',boundStr,...
        st.NY_sc,info_sc.Ncells,info_sc.Nsites));

    useSym = st.symbolic;
    symTag = iif(useSym,'[Symbol]','[Numeric]');

    % 绘制NP矩阵 + 晶格
    draw_matrix(handles.ax_np,H_np,info_np,st,...
        sprintf('NP: %d\\times%d  NY=%d  Cells=%d  half=%d  %s',...
        info_np.Nsites,info_np.Nsites,st.NY_np,...
        info_np.Ncells,st.half_np,symTag),useSym);
    draw_lattice(handles.ax_np_lat,info_np,'np',isSemi,st);

    % 绘制SC矩阵 + 晶格
    draw_matrix(handles.ax_sc,H_sc,info_sc,st,...
        sprintf('SC: %d\\times%d  NY=%d  Cells=%d  half=%d  %s',...
        info_sc.Nsites,info_sc.Nsites,st.NY_sc,...
        info_sc.Ncells,st.half_sc,symTag),useSym);
    draw_lattice(handles.ax_sc_lat,info_sc,'sc',isSemi,st);

    drawnow;  % 强制渲染完成

    % 最终一致性检查: 如果渲染期间状态变了, 再来一次
    stNow = getappdata(fig,'state');
    if ~strcmp(stNow.boundary, st.boundary)
        busy = false;
        redraw(fig);
        return;
    end
    busy = false;
end

function set_nx_enable(fig,tag,en,altStr)
    edt=findobj(fig,'Tag',['edt_' tag]); sld=findobj(fig,'Tag',['sld_' tag]);
    if en
        % 解禁并恢复具体数值
        st = getappdata(fig,'state');
        if ~isempty(edt)
            set(edt,'Enable','on');
            if contains(tag,'NXnp'), set(edt,'String',sprintf('%d',st.NX_np));
            elseif contains(tag,'NXsc'), set(edt,'String',sprintf('%d',st.NX_sc)); end
        end
        if ~isempty(sld)
            set(sld,'Enable','on');
            if contains(tag,'NXnp'), set(sld,'Value',st.NX_np);
            elseif contains(tag,'NXsc'), set(sld,'Value',st.NX_sc); end
        end
    else
        if ~isempty(edt)
            set(edt,'Enable','off','String',altStr);
        end
        if ~isempty(sld), set(sld,'Enable','off','Value',1); end
    end
end

function sync_ctrl(fig,tag,val,isInt)
    edt=findobj(fig,'Tag',['edt_' tag]); sld=findobj(fig,'Tag',['sld_' tag]);
    if ~isempty(edt)
        if isInt, set(edt,'String',sprintf('%d',val));
        elseif contains(tag,'phi')||contains(tag,'kx'), set(edt,'String',sprintf('%.3f',val));
        else, set(edt,'String',sprintf('%.2f',val)); end
    end
    if ~isempty(sld), set(sld,'Value',val); end
end

%% ===== 矩阵显示 =====
function draw_matrix(ax,H,info,st,ttl,useSym)
    if nargin<6, useSym=st.symbolic; end
    cla(ax); hold(ax,'on');
    N = size(H,1);
    if N==0, title(ax,'Empty'); return; end

    % 颜色: 按物理类型区分
    cZero=[0.94 0.94 0.94];    % 零: 灰
    cDiag=[0.95 0.95 0.80];    % 对角/onsite: 黄
    cNN=[0.70 0.85 1.0];       % NN复跃迁: 浅蓝(原色)
    cNNsum=[1.0 0.82 0.60];    % NN x-Bloch和(如-2t cosφ): 橙
    cNNN=[0.90 0.80 0.75];     % NNN实跃迁(-t): 暖灰褐

    t=st.t; phi=st.phi; omg=st.omg; tol=1e-8;

    % 符号矩阵: 代入固定数值判断颜色, 显示含kx符号表达式
    if isa(H,'sym')
        sv = {sym('t'),sym('phi'),sym('omg'),sym('tc'),sym('kx')};
        rv = {st.t, st.phi, st.omg, st.tc, st.kx};
        % 第一遍: 画所有方块
        for i=1:N
            for j=1:N
                v=H(i,j); x=j; y=N-i+1;
                if isequal(v,sym(0))
                    cFace=cZero; cEdge=[0.85 0.85 0.85];
                else
                    vn = double(subs(v, sv, rv));  % 代入固定数值判断类型
                    if i==j
                        cFace=cDiag; cEdge=[0.75 0.75 0.55];
                    elseif abs(vn)<1e-10
                        cFace=cZero; cEdge=[0.85 0.85 0.85];
                    elseif abs(imag(vn))<1e-10
                        % 实数: 区分NNN(-t) vs NN-sum(-2t cosφ)
                        isNNN=false; isNNsum=false;
                        for n=1:4
                            if abs(abs(vn)-n*t)<tol, isNNN=true; break; end
                        end
                        if abs(abs(vn)-2*t*cos(phi))<tol, isNNsum=true; end
                        if isNNN && ~isNNsum
                            cFace=cNNN; cEdge=[0.6 0.5 0.4];
                        elseif isNNsum
                            cFace=cNNsum; cEdge=[0.8 0.5 0.2];
                        else
                            cFace=cNN; cEdge=[0.5 0.6 0.8];
                        end
                    else
                        cFace=cNN; cEdge=[0.5 0.6 0.8];
                    end
                end
                rectangle(ax,'Position',[x-0.48,y-0.48,0.96,0.96],...
                    'FaceColor',cFace,'EdgeColor',cEdge,'LineWidth',0.4);
            end
        end
        % 第二遍: 画所有文字 (方块之上)
        for i=1:N
            for j=1:N
                v=H(i,j); x=j; y=N-i+1;
                if ~isequal(v,sym(0))
                    str = format_elem(v,true,st,i==j);
                    % 长表达式在第二个e指数前换行 (两e指数之间), 保留减号
                    if length(str) > 12
                        eidx = strfind(str, 'e^{');
                        if length(eidx) >= 2
                            brk = eidx(2);
                            pm = strfind(str(1:brk), ' - ');
                            if ~isempty(pm)
                                brk = pm(end);
                                str = [str(1:brk-1) newline str(brk:end)];
                            end
                        end
                    end
                    fs = max(8, min(20, 85/N));
                    text(ax,x,y,str,'FontSize',fs,'HorizontalAlignment','center',...
                        'Interpreter','tex','VerticalAlignment','middle');
                end
            end
        end
        xlim(ax,[0.5 N+0.5]); ylim(ax,[0.5 N+0.5]);
        axis(ax,'equal'); set(ax,'YDir','normal','XTick',1:N,'YTick',1:N,'YTickLabel',N:-1:1,'XAxisLocation','top');
        title(ax,ttl,'FontSize',10,'FontWeight','bold','Interpreter','tex');
        drawnow;
        return;
    end

    % 第一遍: 画所有方块
    for i=1:N
        for j=1:N
            v=H(i,j); x=j; y=N-i+1;
            if abs(v)<1e-12
                cFace=cZero; cEdge=[0.85 0.85 0.85];
            elseif i==j
                cFace=cDiag; cEdge=[0.75 0.75 0.55];
            elseif abs(imag(v))<1e-12
                % 实数: 区分NNN(-t) vs NN-sum(-2t cosφ)
                isNNN=false; isNNsum=false;
                for n=1:4
                    if abs(abs(v)-n*t)<tol, isNNN=true; break; end
                end
                if abs(abs(v)-2*t*cos(phi))<tol, isNNsum=true; end
                if isNNN && ~isNNsum
                    cFace=cNNN; cEdge=[0.6 0.5 0.4];
                elseif isNNsum
                    cFace=cNNsum; cEdge=[0.8 0.5 0.2];
                else
                    cFace=cNN; cEdge=[0.5 0.6 0.8];
                end
            else
                cFace=cNN; cEdge=[0.5 0.6 0.8];
            end
            rectangle(ax,'Position',[x-0.48,y-0.48,0.96,0.96],...
                'FaceColor',cFace,'EdgeColor',cEdge,'LineWidth',0.4);
        end
    end
    % 第二遍: 画所有文字 (方块之上)
    for i=1:N
        for j=1:N
            v=H(i,j); x=j; y=N-i+1;
            if abs(v)>1e-12
                str = format_elem(v,useSym,st,i==j);
                fs = max(8, min(20, 85/N));
                text(ax,x,y,str,'FontSize',fs,'HorizontalAlignment','center',...
                    'Interpreter','tex','VerticalAlignment','middle');
            end
        end
    end
    xlim(ax,[0.5 N+0.5]); ylim(ax,[0.5 N+0.5]);
    axis(ax,'equal'); set(ax,'YDir','normal','XTick',1:N,'YTick',1:N,'YTickLabel',N:-1:1,'XAxisLocation','top');
    title(ax,ttl,'FontSize',10,'FontWeight','bold','Interpreter','tex');
    drawnow;
end

function str = format_elem(v,symbolic,st,isDiag)
    if nargin<4, isDiag=false; end
    % 符号表达式 (含kx): 显示符号而非代入数值
    if isa(v,'sym')
        if isequal(v,sym(0)), str='0'; return; end
        str = sym_pretty(v);
        return;
    end
    if ~symbolic
        if abs(imag(v))<1e-12, str=sprintf('%.2f',real(v));
        else, str=sprintf('%.2f%+.2fi',real(v),imag(v)); end
        return;
    end
    t=st.t; phi=st.phi; omg=st.omg; tol=1e-8;
    if abs(v)<tol, str='0'; return; end
    % omg: 仅对角元 (避免omg=t=1时误标非对角元)
    if isDiag
        for n=4:-1:1
            if abs(v - n*omg) < tol
                if n==1, str=sprintf('\\omega'); else, str=sprintf('%d\\omega',n); end
                return;
            end
        end
        % omg ± n*t (SC B-site NNN x-Bloch自跳)
        for n=4:-1:1
            if abs(v - (omg + n*t)) < tol
                str=sprintf('\\omega+%dt',n); return;
            elseif abs(v - (omg - n*t)) < tol
                if n==1, str=sprintf('\\omega-t');
                else, str=sprintf('\\omega-%dt',n); end
                return;
            end
        end
    end
    % 检查 ±n*t (实跃迁)
    for n=4:-1:1
        if abs(v - n*t) < tol
            if n==1, str='t'; else, str=sprintf('%dt',n); end; return;
        elseif abs(v + n*t) < tol
            if n==1, str='-t'; else, str=sprintf('-%dt',n); end; return;
        end
    end
    % 检查 ±n*t*exp(±iφ) (复跃迁, 4种符号组合)
    bases={t*exp(1i*phi), t*exp(-1i*phi)};
    names={sprintf('t e^{i\\phi}'),sprintf('t e^{-i\\phi}')};
    for n=4:-1:1
        for k=1:2
            b=bases{k}; nm=names{k};
            if abs(v - n*b) < tol
                if n==1, str=nm; else, str=sprintf('%d%s',n,nm); end; return;
            elseif abs(v + n*b) < tol
                if n==1, str=['-' nm]; else, str=sprintf('-%d%s',n,nm); end; return;
            end
        end
    end
    % 检查 2t·cosφ 和 2it·sinφ (SC模型常见: e^{iφ}+e^{-iφ}=2cosφ)
    if abs(imag(v))<1e-12 && abs(v - 2*t*cos(phi))<tol
        str=sprintf('2t cos\\phi'); return;
    elseif abs(real(v))<1e-12 && abs(imag(v) - 2*t*sin(phi))<tol
        str=sprintf('2it sin\\phi'); return;
    elseif abs(imag(v))<1e-12 && abs(v + 2*t*cos(phi))<tol
        str=sprintf('-2t cos\\phi'); return;
    elseif abs(real(v))<1e-12 && abs(imag(v) + 2*t*sin(phi))<tol
        str=sprintf('-2it sin\\phi'); return;
    end
    % 兜底: 数值
    if abs(imag(v))<1e-12, str=sprintf('%.2f',real(v));
    else, str=sprintf('%.2f%+.2fi',real(v),imag(v)); end
end

function str = sym_pretty(v)
    % 符号表达式 → 纯符号TeX: t, tc, ω, e^{±ikx}, e^{±iφ}, 无数字无conj
    s = char(v);
    % 去掉conj (实参数: conj(t)=t 等)
    s = strrep(s,'conj(phi)','phi');
    s = strrep(s,'conj(kx)','kx');
    s = strrep(s,'conj(tc)','tc');
    s = strrep(s,'conj(t)','t');
    s = strrep(s,'conj(omg)','omg');
    % exp → e^{±...} (正负号在指数, kx→k_x下标)
    s = strrep(s,'exp(-phi*1i)','e^{-i\phi}');
    s = strrep(s,'exp(phi*1i)','e^{i\phi}');
    s = strrep(s,'exp(-kx*1i)','e^{-ik_x}');
    s = strrep(s,'exp(kx*1i)','e^{ik_x}');
    s = strrep(s,'exp(-1i*phi)','e^{-i\phi}');
    s = strrep(s,'exp(1i*phi)','e^{i\phi}');
    s = strrep(s,'exp(-1i*kx)','e^{-ik_x}');
    s = strrep(s,'exp(1i*kx)','e^{ik_x}');
    s = strrep(s,'exp(phi*(-1)*1i)','e^{-i\phi}');
    s = strrep(s,'exp(kx*(-1)*1i)','e^{-ik_x}');
    s = strrep(s,'omg','\omega');
    s = strrep(s,'1i','i');
    s = strrep(s,'*',' ');
    % t/tc/ω 永远在 e 指数左边(移到所有 e 序列最前): 'e^{...} e^{...} tc' → 'tc e^{...} e^{...}'
    s = regexprep(s, '((?:e\^\{[^}]*\}\s*)+)(tc|t|\\omega)', '$2 $1');
    str = s;
end

%% ===== NP哈密顿量 =====
function [H,info] = build_H_np(st,halfCell,order)
    NX=st.NX_np; NY=st.NY_np; t=st.t; phi=st.phi; omg=st.omg; tc=st.tc;
    if halfCell, Lx=2*NX+1; Ly=2*NY+1; else, Lx=2*NX; Ly=2*NY; end
    Nat=Lx*Ly;
    % 构建索引映射: grid(x,y) → site index
    smap=zeros(Ly,Lx);  % smap(y+1,x+1) = site index
    smap=zeros(Ly,Lx);
    if strcmp(order,'cell')
        % 元胞列优先: 列cx(左→右), 行cy(下→上), 胞内顺时针(左下1,左上2,右上3,右下4)
        Ncx=NX; if halfCell, Ncx=NX+1; end
        Ncy=NY; if halfCell, Ncy=NY+1; end
        ids=0;
        for cx=0:Ncx-1, for cy=0:Ncy-1
            x1=2*cx;   y1=2*cy;     % 左下(1)
            x2=2*cx;   y2=2*cy+1;   % 左上(2)
            x3=2*cx+1; y3=2*cy+1;   % 右上(3)
            x4=2*cx+1; y4=2*cy;     % 右下(4)
            for k=1:4
                if k==1, xx=x1; yy=y1; elseif k==2, xx=x2; yy=y2;
                elseif k==3, xx=x3; yy=y3; else, xx=x4; yy=y4; end
                if xx<Lx&&yy<Ly, ids=ids+1; smap(yy+1,xx+1)=ids; end
            end
        end, end
        idx=@(x,y) smap(y+1,x+1);
    else
        % 列优先 (原始): idx = x*Ly + y + 1
        idx=@(x,y) x*Ly + y + 1;
        for yy=0:Ly-1, for xx=0:Lx-1, smap(yy+1,xx+1)=idx(xx,yy); end, end
    end
    % 构建反向查找 (用于晶格绘制)
    rmap=zeros(Ly,Lx);
    for yy=0:Ly-1, for xx=0:Lx-1
        rmap(yy+1,xx+1)=idx(xx,yy);
    end, end
    H=zeros(Nat);
    for yi=0:Ly-1
        for xi=0:Lx-1
            i=idx(xi,yi); isA=(mod(xi+yi,2)==0);
            for d=1:4
                dirs=[1,0;-1,0;0,1;0,-1];
                nx=xi+dirs(d,1); ny=yi+dirs(d,2);
                if nx<0||nx>=Lx||ny<0||ny>=Ly, continue; end
                j=idx(nx,ny); if j<=i, continue; end
                inCell=(floor(xi/2)==floor(nx/2))&&(floor(yi/2)==floor(ny/2));
                t_amp=t*inCell+t*(~inCell);
                % A子格 右-φ左-φ上+φ下+φ; B子格 右+φ左+φ上-φ下-φ
                if isA
                    if d==1, bp=-phi; elseif d==2, bp=-phi;
                    elseif d==3, bp=phi; else, bp=phi; end
                else
                    if d==1, bp=phi; elseif d==2, bp=phi;
                    elseif d==3, bp=-phi; else, bp=-phi; end
                end
                H(i,j)=-t_amp*exp(1i*bp); H(j,i)=conj(H(i,j));
            end
            for ndx=[-1,1], for ndy=[-1,1]
                if ndx*ndy==0, continue; end
                nx=xi+ndx; ny=yi+ndy;
                if nx<0||nx>=Lx||ny<0||ny>=Ly, continue; end
                j=idx(nx,ny); if j<=i, continue; end
                sameCell=(floor(xi/2)==floor(nx/2))&&(floor(yi/2)==floor(ny/2));
                diagCell=(floor(xi/2)~=floor(nx/2))&&(floor(yi/2)~=floor(ny/2));
                if sameCell||diagCell
                    H(i,j)=-t; H(j,i)=-t;
                end
            end, end
        end
    end
    H(1:Nat+1:end)=omg;
    info=struct('Ncells',NX*NY,'Nsites',Nat,'Lx',Lx,'Ly',Ly,'smap',smap,'rmap',rmap,'order',order);
end

%% ===== SC哈密顿量 =====
function [H,info] = build_H_sc(st,halfCell,order)
    NX=st.NX_sc; NY=st.NY_sc; t=st.t; phi=st.phi;
    if halfCell, Lx=2*NX+1; Ly=2*NY+1; else, Lx=2*NX; Ly=2*NY; end
    site=[]; smap=zeros(Ly,Lx); ids=0;
    if strcmp(order,'cell')
        % 列优先按元胞: 每列cx, 从下往上cy, 每胞A(蓝)→B(橙)
        Ncx=NX; if halfCell, Ncx=NX+1; end
        Ncy=NY; if halfCell, Ncy=NY+1; end
        for cx=0:Ncx-1, for cy=0:Ncy-1
            xA=2*cx+1; yA=2*cy;
            if xA<Lx&&yA<Ly&&mod(xA,2)==1&&mod(yA,2)==0, ids=ids+1; smap(yA+1,xA+1)=ids; site(ids,:)=[xA,yA,true]; end
            xB=2*cx; yB=2*cy+1;
            if xB<Lx&&yB<Ly&&mod(xB,2)==0&&mod(yB,2)==1, ids=ids+1; smap(yB+1,xB+1)=ids; site(ids,:)=[xB,yB,false]; end
        end, end
    else
        % 行优先 (原始)
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x,2)==1&&mod(y,2)==0); isB=(mod(x,2)==0&&mod(y,2)==1);
            if isA||isB, ids=ids+1; smap(y+1,x+1)=ids; site(ids,:)=[x,y,isA]; end
        end, end
    end
    Nat=ids; H=zeros(Nat); gi=@(x,y)smap(y+1,x+1);
    for s=1:Nat
        x1=site(s,1);y1=site(s,2);isA1=site(s,3);
        for dx=[-1,1], for dy=[-1,1]
            x2=x1+dx;y2=y1+dy;
            if x2<0||x2>=Lx||y2<0||y2>=Ly, continue; end
            s2=gi(x2,y2); if s2==0||s2<=s, continue; end
            if isA1, isCW=(dx*dy==-1); else, isCW=(dx*dy==+1); end
            pv=iif(isCW,-phi,phi); H(s,s2)=-t*exp(-1i*pv); H(s2,s)=conj(H(s,s2));
        end, end
        if isA1, nd=[0,2;0,-2]; else, nd=[2,0;-2,0]; end
        for d=1:2
            dx=nd(d,1);dy=nd(d,2); x2=x1+dx;y2=y1+dy;
            if x2<0||x2>=Lx||y2<0||y2>=Ly, continue; end
            s2=gi(x2,y2); if s2==0||s2<=s, continue; end
            H(s,s2)=-t; H(s2,s)=-t;
        end
    end
    H(1:Nat+1:end)=st.omg;  % on-site
    info=struct('Ncells',NX*NY,'Nsites',Nat,'Lx',Lx,'Ly',Ly);
end

%% ===== NP半无限ribbon哈密顿量 =====
function [Hrib,info] = build_H_np_ribbon(st,halfCell,order)
    % x-Bloch, y有限: 返回 H0, H1 使得 H(k)=H0+H1*e^{ik}+H1'*e^{-ik}
    % order='cell': 胞优先序(左下1左上2右上3右下4); 其他: 按列排序
    if nargin<3, order='cell'; end
    N=st.NY_np; t=st.t; phi=st.phi; omg=st.omg; tc=st.tc;
    Lx=2;
    if halfCell, Ly=2*N+1; Nat=4*N+2; Ncy=N+1;
    else, Ly=2*N; Nat=4*N; Ncy=N; end

    % ——— 构建 basis: 坐标→序号 ———
    nb=0; basis=zeros(Nat,2);
    keys=containers.Map();
    if strcmp(order,'cell')
        % 胞优先序: 每胞左下(1)左上(2)右上(3)右下(4)
        for cy=0:N-1
            pts=[0,2*cy; 0,2*cy+1; 1,2*cy+1; 1,2*cy];
            for k=1:4
                nb=nb+1; basis(nb,:)=pts(k,:);
                keys(sprintf('%d_%d',pts(k,1),pts(k,2)))=nb;
            end
        end
        if halfCell
            nb=nb+1; basis(nb,:)=[0,2*N]; keys(sprintf('%d_%d',0,2*N))=nb;
            nb=nb+1; basis(nb,:)=[1,2*N]; keys(sprintf('%d_%d',1,2*N))=nb;
        end
    else
        % 列优先: 先x=0列(从下到上), 再x=1列
        for xx=0:1
            for yy=0:Ly-1
                nb=nb+1; basis(nb,:)=[xx,yy];
                keys(sprintf('%d_%d',xx,yy))=nb;
            end
        end
    end

    % ——— 构建 smap (用于晶格绘图) ———
    smap=zeros(Ly,Lx);
    if strcmp(order,'cell')
        for cy=0:N-1
            smap(2*cy+1,1)=cy*4+1; smap(2*cy+2,1)=cy*4+2;  % col0: BL(1), TL(2)
            smap(2*cy+2,2)=cy*4+3; smap(2*cy+1,2)=cy*4+4;  % col1: TR(3), BR(4)
        end
        if halfCell
            smap(Ly,1)=4*N+1; smap(Ly,2)=4*N+2;
        end
    else
        for yy=0:Ly-1, for xx=0:1
            smap(yy+1,xx+1)=xx*Ly+yy+1;
        end, end
    end

    % ——— 坐标驱动构建 H0, H1 ———
    if isa(st.t,'sym')
        H0=sym(zeros(Nat)); H1=sym(zeros(Nat));
    else
        H0=zeros(Nat); H1=zeros(Nat);
    end
    for i=1:Nat
        x0=basis(i,1); y0=basis(i,2);
        isA0=(mod(x0+y0,2)==0);

        % === NN: 4方向, 复跃迁 ===
        for d=1:4
            dirs=[1,0;-1,0;0,1;0,-1];
            dx=dirs(d,1); dy=dirs(d,2);
            xt=x0+dx; yt=y0+dy;
            if yt<0||yt>=Ly, continue; end
            % x-Bloch 折叠
            cs=0; xm=xt;
            while xm<0, xm=xm+Lx; cs=cs-1; end
            while xm>=Lx, xm=xm-Lx; cs=cs+1; end
            kk=sprintf('%d_%d',xm,yt);
            if ~keys.isKey(kk), continue; end
            j=keys(kk);
            if cs==0 && j<=i, continue; end  % H0去重; cs=1跨周期键正常进H1

            % 相位: A子格 右-φ左-φ上+φ下+φ; B子格 右+φ左+φ上-φ下-φ
            if isA0
                if d==1, bp=-phi; elseif d==2, bp=-phi;
                elseif d==3, bp=phi; else, bp=phi; end
            else
                if d==1, bp=phi; elseif d==2, bp=phi;
                elseif d==3, bp=-phi; else, bp=-phi; end
            end
            hop=-t*exp(1i*bp);

            if cs==0
                H0(i,j)=H0(i,j)+hop; H0(j,i)=H0(j,i)+conj(hop);
            elseif cs==1
                H1(i,j)=H1(i,j)+hop;
            end
            % cs==-1: 跳过, 避免k=0时与cs=0键叠加成-2t
        end

        % === NNN: 对角线 |dx|=|dy|=1, 实跃迁 -t ===
        % 仅胞内 + 对角元胞间(cx和cy都不同), 用未折叠xt判断
        cx0=floor(x0/2); cy0=floor(y0/2);
        for ndx=[-1,1], for ndy=[-1,1]
            if ndx*ndy==0, continue; end
            xt=x0+ndx; yt=y0+ndy;
            if yt<0||yt>=Ly, continue; end
            cxt=floor(xt/2); cyt=floor(yt/2);
            sameCell=(cx0==cxt)&&(cy0==cyt);
            diagCell=(cx0~=cxt)&&(cy0~=cyt);
            if ~(sameCell||diagCell), continue; end  % 竖直/水平相邻胞不算
            cs=0; xm=xt;
            while xm<0, xm=xm+Lx; cs=cs-1; end
            while xm>=Lx, xm=xm-Lx; cs=cs+1; end
            if cs==-1, continue; end  % 避免双计数
            kk=sprintf('%d_%d',xm,yt);
            if ~keys.isKey(kk), continue; end
            j=keys(kk);
            hop=-t;
            if cs==0
                if j<=i, continue; end  % H0去重
                H0(i,j)=H0(i,j)+hop; H0(j,i)=H0(j,i)+hop;
            else  % cs==1: x-Bloch NNN 进H1 (e^{ikx})
                H1(i,j)=H1(i,j)+hop;
            end
        end, end
    end
    % on-site
    H0(1:Nat+1:end)=H0(1:Nat+1:end)+omg;

    Hrib=struct('H0',H0,'H1',H1);
    info=struct('Ncells',N,'Nsites',Nat,'Lx',Lx,'Ly',Ly,'smap',smap,'order','cell');
end

%% ===== SC半无限ribbon哈密顿量 =====
function [Hrib,info] = build_H_sc_ribbon(st,halfCell,order)
    if nargin<3, order='cell'; end
    NY=st.NY_sc; t=st.t; phi=st.phi; Lx=2;
    if halfCell, Ly=2*NY+1; else, Ly=2*NY; end
    nb=0; basis=[]; keys=containers.Map();
    if strcmp(order,'cell')
        % 行优先(每行y, 列x) — 与元胞优先对应
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x,2)==1)&&(mod(y,2)==0); isB=(mod(x,2)==0)&&(mod(y,2)==1);
            if isA||isB, nb=nb+1; basis(nb,:)=[x,y,isA]; keys(sprintf('%d_%d',x,y))=nb; end
        end, end
    else
        % 列优先: 先x=0列(从下到上), 再x=1列
        for x=0:Lx-1, for y=0:Ly-1
            isA=(mod(x,2)==1)&&(mod(y,2)==0); isB=(mod(x,2)==0)&&(mod(y,2)==1);
            if isA||isB, nb=nb+1; basis(nb,:)=[x,y,isA]; keys(sprintf('%d_%d',x,y))=nb; end
        end, end
    end
    Nat=nb;
    if isa(st.t,'sym')
        H0=sym(zeros(Nat)); H1=sym(zeros(Nat));
    else
        H0=zeros(Nat); H1=zeros(Nat);
    end
    for i=1:Nat
        x0=basis(i,1); y0=basis(i,2); isA0=basis(i,3);
        for dx=[-1,1], for dy=[-1,1]
            xt=x0+dx; yt=y0+dy;
            if yt<0||yt>=Ly, continue; end
            cs=0; xm=xt;
            while xm<0, xm=xm+Lx; cs=cs-1; end
            while xm>=Lx, xm=xm-Lx; cs=cs+1; end
            if cs==-1, continue; end  % NN x-Bloch仅cs=1
            kk=sprintf('%d_%d',xm,yt);
            if ~keys.isKey(kk), continue; end
            j=keys(kk);
            if cs==0 && j<=i, continue; end  % H0每键去重; H1不跳(无重复)
            if isA0, isCW=(dx*dy==-1); else, isCW=(dx*dy==+1); end
            pv=iif(isCW,phi,-phi); hop=-t*exp(-1i*pv);
            if cs==0
                H0(i,j)=H0(i,j)+hop; H0(j,i)=H0(j,i)+conj(hop);
            else
                H1(i,j)=H1(i,j)+hop;
            end
        end, end
        if isA0, nd=[0,2;0,-2]; else, nd=[2,0;-2,0]; end
        for d=1:2
            dx=nd(d,1); dy=nd(d,2); xt=x0+dx; yt=y0+dy;
            if yt<0||yt>=Ly, continue; end
            cs=0; xm=xt;
            while xm<0, xm=xm+Lx; cs=cs-1; end
            while xm>=Lx, xm=xm-Lx; cs=cs+1; end
            if cs==-1, continue; end  % 避免x-Bloch双计数
            kk=sprintf('%d_%d',xm,yt);
            if ~keys.isKey(kk), continue; end
            j=keys(kk);
            hop=-t;
            if cs==0
                if j<=i, continue; end  % H0去重
                H0(i,j)=H0(i,j)+hop; H0(j,i)=H0(j,i)+hop;
            else  % cs==1: B-site x-Bloch自跳
                H1(i,j)=H1(i,j)+hop;
            end
        end
    end
    % 对角项
    H0(1:Nat+1:end)=H0(1:Nat+1:end)+st.omg;
    Hrib=struct('H0',H0,'H1',H1);
    % smap: 晶格坐标→序号
    smap=zeros(Ly,Lx);
    for s=1:Nat
        smap(basis(s,2)+1,basis(s,1)+1)=s;
    end
    info=struct('Ncells',NY,'Nsites',Nat,'Lx',Lx,'Ly',Ly,'smap',smap,'order','cell');
end

%% ===== 晶格绘制 =====
function draw_lattice(ax,info,model,isSemi,st)
    cla(ax); hold(ax,'on');
    Lx=info.Lx; Ly=info.Ly;
    cA=[0.20 0.55 0.80]; cB=[0.90 0.40 0.20]; cG=[0.85 0.85 0.85];
    cDim=[0.65 0.65 0.65];  % 半无限黯淡色
    cNN=[0.80 0.15 0.15]; cNNN=[0.15 0.60 0.20];
    ms = max(10,22-max(Lx,Ly)*1.0); lw=max(0.3,2-max(Lx,Ly)*0.15);

    if strcmp(model,'np')
        % 连线: NN (水平/竖直) — 红色, NNN (对角) — 绿色
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x+y,2)==0);
            if x+1<Lx, plot(ax,[x x+1],[y y],'-','Color',cNN,'LineWidth',lw); end
            if y+1<Ly, plot(ax,[x x],[y y+1],'-','Color',cNN,'LineWidth',lw); end
            if mod(x+y,2)==0 && x+1<Lx && y+1<Ly
                plot(ax,[x x+1],[y y+1],'--','Color',cNNN,'LineWidth',lw*0.7);
                plot(ax,[x+1 x],[y y+1],'--','Color',cNNN,'LineWidth',lw*0.7);
            end
        end, end
        % 元胞格线 + 元胞框 (线层之下,格点之上)
        for i=0:2:Lx, plot(ax,[i-0.5 i-0.5],[-0.5 Ly-0.5],':','Color',[0.6 0.6 0.6],'LineWidth',0.3); end
        for i=0:2:Ly, plot(ax,[-0.5 Lx-0.5],[i-0.5 i-0.5],':','Color',[0.6 0.6 0.6],'LineWidth',0.3); end
        for cx=0:2:Lx-1, for cy=0:2:Ly-1
            if cx+2<=Lx && cy+2<=Ly
                if cx==0 && cy==0
                    rectangle(ax,'Position',[cx-0.5,cy-0.5,2,2],'EdgeColor','k','LineWidth',2,'FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
                else
                    rectangle(ax,'Position',[cx-0.5,cy-0.5,2,2],'EdgeColor',[0.5 0.5 0.5],'LineWidth',0.8,'LineStyle','--','FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
                end
            end
        end, end
        % 格点 + 序号 (最上层)
        hasSmap = isfield(info,'smap') && strcmp(info.order,'cell');
        for y=0:Ly-1, for x=0:Lx-1
            if hasSmap, idx=info.smap(y+1,x+1);
            elseif isfield(info,'rmap'), idx=info.rmap(y+1,x+1);
            else, idx=x*Ly+y+1; end
            isA=(mod(x+y,2)==0);
            if isA, plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',cA,'MarkerEdgeColor','k','LineWidth',0.3);
            else, plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',cB,'MarkerEdgeColor','k','LineWidth',0.3); end
            text(ax,x,y,sprintf('%d',idx),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
        end, end
        % 如果有半元胞: 标记新增行列
        if Lx>2*floor(Lx/2)  % 奇数=有半元胞
            Lnp=2*floor(Lx/2);
            % 原NP区域
            rectangle(ax,'Position',[-0.5,-0.5,Lnp,Lnp],'EdgeColor','k','LineWidth',2,'LineStyle','-');
            % 右半列
            rectangle(ax,'Position',[Lnp-0.5,-0.5,1,Lnp],'EdgeColor',[1 0 0],'LineWidth',1.5,'LineStyle','--','FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
            % 上半行
            rectangle(ax,'Position',[-0.5,Lnp-0.5,Lnp,1],'EdgeColor',[0 0 1],'LineWidth',1.5,'LineStyle','--','FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
            % 角
            rectangle(ax,'Position',[Lnp-0.5,Lnp-0.5,1,1],'EdgeColor',[0.5 0 0.5],'LineWidth',1.5,'LineStyle','--','FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
        end
    else
        % SC: NN=对角(红), NNN=轴向步2(绿)
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x,2)==1&&mod(y,2)==0); isB=(mod(x,2)==0&&mod(y,2)==1);
            if ~(isA||isB), continue; end
            % NN: 对角
            for dx=[-1,1], for dy=[-1,1]
                nx=x+dx; ny=y+dy;
                if nx<0||nx>=Lx||ny<0||ny>=Ly, continue; end
                nA=(mod(nx,2)==1&&mod(ny,2)==0); nB=(mod(nx,2)==0&&mod(ny,2)==1);
                if nA||nB
                    plot(ax,[x nx],[y ny],'-','Color',cNN,'LineWidth',lw);
                end
            end, end
            % NNN: 轴向步2
            if isA
                if y+2<Ly, nA=(mod(x,2)==1&&mod(y+2,2)==0); nB=(mod(x,2)==0&&mod(y+2,2)==1);
                    if nA||nB, plot(ax,[x x],[y y+2],'--','Color',cNNN,'LineWidth',lw*0.7); end
                end
            else
                if x+2<Lx, nA=(mod(x+2,2)==1&&mod(y,2)==0); nB=(mod(x+2,2)==0&&mod(y,2)==1);
                    if nA||nB, plot(ax,[x x+2],[y y],'--','Color',cNNN,'LineWidth',lw*0.7); end
                end
            end
        end, end
        % SC晶格序号: 优先用info已有smap
        if isfield(info,'smap')
            smap_lat = info.smap;
        else
            smap_lat=zeros(Ly,Lx); idx_lat=0;
            st_lat=getappdata(ancestor(ax,'figure'),'state');
            if strcmp(st_lat.sc_order,'cell')
                NX_lat=floor(Lx/2); NY_lat=floor(Ly/2); hasHalf=(Lx>2*NX_lat);
                Ncx_lat=NX_lat; if hasHalf, Ncx_lat=NX_lat+1; end
                Ncy_lat=NY_lat; if hasHalf, Ncy_lat=NY_lat+1; end
                for cx=0:Ncx_lat-1, for cy=0:Ncy_lat-1
                    xA=2*cx+1; yA=2*cy; if xA<Lx&&yA<Ly&&mod(xA,2)==1&&mod(yA,2)==0, idx_lat=idx_lat+1; smap_lat(yA+1,xA+1)=idx_lat; end
                    xB=2*cx; yB=2*cy+1; if xB<Lx&&yB<Ly&&mod(xB,2)==0&&mod(yB,2)==1, idx_lat=idx_lat+1; smap_lat(yB+1,xB+1)=idx_lat; end
                end, end
            else
                for y=0:Ly-1, for x=0:Lx-1, isA=(mod(x,2)==1&&mod(y,2)==0); isB=(mod(x,2)==0&&mod(y,2)==1); if isA||isB, idx_lat=idx_lat+1; smap_lat(y+1,x+1)=idx_lat; end, end, end
            end
        end
    end
    % 元胞框和半元胞标记 (仅SC, 格点之前绘制)
    if strcmp(model,'sc')
        rectangle(ax,'Position',[-0.5,-0.5,2,2],'EdgeColor','k','LineWidth',2,'FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
        for mx=0:2:Lx-1, for my=0:2:Ly-1
            if mx+2<=Lx&&my+2<=Ly
                rectangle(ax,'Position',[mx-0.5,my-0.5,2,2],'EdgeColor',[0.4 0.4 0.4],'LineWidth',0.6,'LineStyle',':','FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
            end
        end, end
        % 半元胞标记
        if Lx>2*floor(Lx/2)  % 奇数=有半元胞
            % 右半列
            for my=0:2:Ly-1
                if my+2<Ly
                    rectangle(ax,'Position',[Lx-1-0.5,my-0.5,1,2],'EdgeColor',[1 0 0],'LineWidth',1.2,'LineStyle','--','FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
                end
            end
            % 上半行
            for mx=0:2:Lx-1
                if mx+2<Lx
                    rectangle(ax,'Position',[mx-0.5,Ly-1-0.5,2,1],'EdgeColor',[0 0 1],'LineWidth',1.2,'LineStyle','--','FaceColor',[0.9 0.9 1.0],'FaceAlpha',0.25);
                end
            end
        end
        % SC格点+空位 (最上层, 元胞框之后)
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x,2)==1&&mod(y,2)==0); isB=(mod(x,2)==0&&mod(y,2)==1);
            if ~(isA||isB), plot(ax,x,y,'o','MarkerSize',ms*0.5,'MarkerFaceColor',cG,'MarkerEdgeColor','none'); end
        end, end
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x,2)==1&&mod(y,2)==0); isB=(mod(x,2)==0&&mod(y,2)==1);
            if isA, plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',cA,'MarkerEdgeColor','k','LineWidth',0.3);
                text(ax,x,y,sprintf('%d',smap_lat(y+1,x+1)),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
            elseif isB, plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',cB,'MarkerEdgeColor','k','LineWidth',0.3);
                text(ax,x,y,sprintf('%d',smap_lat(y+1,x+1)),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle'); end
        end, end
    end
    % 半无限: 虚影格点+连线
    if isSemi
        text(ax,Lx/2-0.5,Ly+0.6,'x: Bloch (∞)','FontSize',9,'Color',[0.4 0.4 0.4],'FontWeight','bold','HorizontalAlignment','center');
        sm=info.smap; cAL=cA*0.35+0.65; cBL=cB*0.35+0.65;
        if strcmp(model,'np')
            extL=-1; extR=Lx;  % 左右各一列
            % 虚影连线 (仅涉及虚影格点, 不重画内部)
            for y=0:Ly-1, for x=extL:extR
                isGhost=(x<0||x>=Lx); srcX=mod(x+Lx*99,Lx); if sm(y+1,srcX+1)==0, continue; end
                % NN水平 (至少一端是虚影)
                if x+1<=extR, nsrcX=mod(x+1+Lx*99,Lx); nx=x+1; isNG=(nx<0||nx>=Lx);
                    if sm(y+1,nsrcX+1)>0 && (isGhost||isNG), plot(ax,[x nx],[y y],'-','Color',cNN*0.5+0.5,'LineWidth',lw*0.6); end
                end
                % NN竖直 (至少一端是虚影)
                if y+1<Ly, nsrcX=mod(x+Lx*99,Lx);
                    if sm(y+2,nsrcX+1)>0 && isGhost, plot(ax,[x x],[y y+1],'-','Color',cNN*0.5+0.5,'LineWidth',lw*0.6); end
                end
                % NNN对角 (至少一端是虚影)
                inCell=(mod(srcX,2)==0&&mod(y,2)==0); diagCell=(mod(srcX,2)==1&&mod(y,2)==1);
                if inCell||diagCell
                    if x+1<=extR&&y+1<Ly, nx=x+1; nsrcX=mod(nx+Lx*99,Lx); isNG=(nx<0||nx>=Lx);
                        if sm(y+2,nsrcX+1)>0 && (isGhost||isNG), plot(ax,[x nx],[y y+1],'--','Color',cNNN*0.4+0.6,'LineWidth',lw*0.5); end
                    end
                    if x+1<=extR&&y+1<Ly, nx2=x+1; nsrcX2=mod(x+Lx*99,Lx); isNG2=(nx2<0||nx2>=Lx);
                        if sm(y+2,nsrcX2+1)>0 && (isNG2||isGhost), plot(ax,[x+1 x],[y y+1],'--','Color',cNNN*0.4+0.6,'LineWidth',lw*0.5); end
                    end
                end
            end, end
            % 虚影格点
            for y=0:Ly-1, for x=extL:extR
                srcX=mod(x+Lx*99,Lx); if sm(y+1,srcX+1)==0, continue; end
                if x>=0&&x<Lx, continue; end  % 只画虚影
                isA=(mod(srcX+y,2)==0); fc=iif(isA,cAL,cBL); idx=sm(y+1,srcX+1);
                plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',fc,'MarkerEdgeColor','k','LineWidth',0.3);
                text(ax,x,y,sprintf('%d',idx),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center');
            end, end
        else % SC: 左右各放完整元胞(Lx列)
            extL=-Lx; extR=2*Lx-1;
            % 连线
            % 虚影连线 (仅涉及虚影格点)
            for y=0:Ly-1, for x=extL:extR
                isGhost=(x<0||x>=Lx); srcX=mod(x+Lx*99,Lx); isA=(mod(srcX,2)==1&&mod(y,2)==0); isB=(mod(srcX,2)==0&&mod(y,2)==1);
                if ~(isA||isB), continue; end
                for dx=[-1,1], for dy=[-1,1]
                    nx=x+dx; ny=y+dy; if nx<extL||nx>extR||ny<0||ny>=Ly, continue; end
                    isNG=(nx<0||nx>=Lx); if ~isGhost&&~isNG, continue; end  % 至少一端虚影
                    nsrcX=mod(nx+Lx*99,Lx); nA=(mod(nsrcX,2)==1&&mod(ny,2)==0); nB=(mod(nsrcX,2)==0&&mod(ny,2)==1);
                    if nA||nB, plot(ax,[x nx],[y ny],'-','Color',cNN*0.5+0.5,'LineWidth',lw*0.6); end
                end, end
                if isA&&y+2<Ly, nsrcX=mod(x+Lx*99,Lx); ny2=y+2; isNG=(y+2>=Ly);
                    if mod(nsrcX,2)==1&&mod(ny2,2)==0 && (isGhost||isNG), plot(ax,[x x],[y ny2],'--','Color',cNNN*0.4+0.6,'LineWidth',lw*0.5); end, end
                if isB, nx2=x+2; if nx2<=extR, nsrcX=mod(nx2+Lx*99,Lx); isNG=(nx2<0||nx2>=Lx);
                    if mod(nsrcX,2)==0&&mod(y,2)==1 && (isGhost||isNG), plot(ax,[x nx2],[y y],'--','Color',cNNN*0.4+0.6,'LineWidth',lw*0.5); end, end, end
            end, end
            % 虚影格点
            for y=0:Ly-1, for x=extL:extR
                srcX=mod(x+Lx*99,Lx); isA=(mod(srcX,2)==1&&mod(y,2)==0); isB=(mod(srcX,2)==0&&mod(y,2)==1);
                if ~(isA||isB), continue; end
                if x>=0&&x<Lx, continue; end  % 只画虚影
                fc=iif(isA,cAL,cBL); idx=sm(y+1,srcX+1);
                plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',fc,'MarkerEdgeColor','k','LineWidth',0.3);
                text(ax,x,y,sprintf('%d',idx),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center');
            end, end
        end
        xlim(ax,[extL-0.5 extR+0.5]); ylim(ax,[-0.5 Ly+0.7]);
    else
        xlim(ax,[-0.5 Lx-0.5]); ylim(ax,[-0.5 Ly-0.5]);
    end
    axis(ax,'equal'); set(ax,'YDir','normal','XTick',[],'YTick',[]);
    ttlStr = iif(isSemi, ...
        sprintf('x: Bloch (∞), y: %d rows (%d sites)',Ly,info.Nsites), ...
        sprintf('%d\\times%d (%d sites)',Lx,Ly,info.Nsites));
    title(ax,ttlStr,'FontSize',8);
    % 最后重绘所有格点和序号置顶
    redrawSitesOnTop(ax,info,model,isSemi,ms,cA,cB,st);
    drawnow;
end

function redrawSitesOnTop(ax,info,model,isSemi,ms,cA,cB,st)
    Lx=info.Lx; Ly=info.Ly;
    if isfield(info,'smap'), sm=info.smap;
    else, sm=get_sc_smap(Lx,Ly,st.sc_order); end
    cAL=cA*0.35+0.65; cBL=cB*0.35+0.65; cG=[0.85 0.85 0.85];
    if strcmp(model,'np')
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x+y,2)==0); idx=sm(y+1,x+1);
            fc=iif(isA,cA,cB);
            plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',fc,'MarkerEdgeColor','k','LineWidth',0.3);
            text(ax,x,y,sprintf('%d',idx),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
        end, end
        if isSemi
            extL=-1; extR=Lx;
            for y=0:Ly-1, for x=extL:extR
                if x>=0&&x<Lx, continue; end
                srcX=mod(x+Lx*99,Lx); if sm(y+1,srcX+1)==0, continue; end
                isA=(mod(srcX+y,2)==0); fc=iif(isA,cAL,cBL); idx=sm(y+1,srcX+1);
                plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',fc,'MarkerEdgeColor','k','LineWidth',0.3);
                text(ax,x,y,sprintf('%d',idx),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
            end, end
        end
    else % SC
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x,2)==1&&mod(y,2)==0); isB=(mod(x,2)==0&&mod(y,2)==1);
            if ~(isA||isB), plot(ax,x,y,'o','MarkerSize',ms*0.5,'MarkerFaceColor',cG,'MarkerEdgeColor','none'); continue; end
            fc=iif(isA,cA,cB); idx=sm(y+1,x+1);
            plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',fc,'MarkerEdgeColor','k','LineWidth',0.3);
            text(ax,x,y,sprintf('%d',idx),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
        end, end
        if isSemi
            extL=-Lx; extR=2*Lx-1;
            for y=0:Ly-1, for x=extL:extR
                if x>=0&&x<Lx, continue; end
                srcX=mod(x+Lx*99,Lx); isA=(mod(srcX,2)==1&&mod(y,2)==0); isB=(mod(srcX,2)==0&&mod(y,2)==1);
                if ~(isA||isB), continue; end
                fc=iif(isA,cAL,cBL); idx=sm(y+1,srcX+1);
                plot(ax,x,y,'o','MarkerSize',ms,'MarkerFaceColor',fc,'MarkerEdgeColor','k','LineWidth',0.3);
                text(ax,x,y,sprintf('%d',idx),'FontSize',max(7,ms*0.6),'Color','w','FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
            end, end
        end
    end
end

function smap = get_sc_smap(Lx,Ly,order)
    smap=zeros(Ly,Lx); ids=0;
    if strcmp(order,'cell')
        NX_lat=floor(Lx/2); NY_lat=floor(Ly/2);
        hasHalf=(Lx>2*NX_lat);
        Ncx=NX_lat; if hasHalf, Ncx=NX_lat+1; end
        Ncy=NY_lat; if hasHalf, Ncy=NY_lat+1; end
        for cx=0:Ncx-1, for cy=0:Ncy-1
            xA=2*cx+1; yA=2*cy;
            if xA<Lx&&yA<Ly&&mod(xA,2)==1&&mod(yA,2)==0, ids=ids+1; smap(yA+1,xA+1)=ids; end
            xB=2*cx; yB=2*cy+1;
            if xB<Lx&&yB<Ly&&mod(xB,2)==0&&mod(yB,2)==1, ids=ids+1; smap(yB+1,xB+1)=ids; end
        end, end
    else
        for y=0:Ly-1, for x=0:Lx-1
            isA=(mod(x,2)==1&&mod(y,2)==0); isB=(mod(x,2)==0&&mod(y,2)==1);
            if isA||isB, ids=ids+1; smap(y+1,x+1)=ids; end
        end, end
    end
end

function v=iif(cond,v1,v2)
    if cond, v=v1; else, v=v2; end
end
end
