function r_val = plot_panel_trxsh ...
     ( ...
       cfreq, ...
       dflopspercycle, ...
       nth, ...
       thr_str, ...
       dt_ch, ...
       stor_str, ...
       smalldims, ...
       dirpath, ...
       arch_str, ...
       vend_str, ...
       impl ...
     )

%cfreq = 1.8;
%dflopspercycle = 32;

% Create filename "templates" for the files that contain the performance
% results.
filetemp_blissup  = '%s/output_%s_%s_blissup.m';
filetemp_blislpab = '%s/output_%s_%s_blislpab.m';
filetemp_eigen    = '%s/output_%s_%s_eigen.m';
filetemp_open     = '%s/output_%s_%s_openblas.m';
filetemp_bfeo     = '%s/output_%s_%s_blasfeo.m';
filetemp_xsmm     = '%s/output_%s_%s_libxsmm.m';
filetemp_vend     = '%s/output_%s_%s_vendor.m';

% Create a variable name "template" for the variables contained in the
% files outlined above.
vartemp = 'data_%s_%s_%s( :, : )';

% Define the datatypes and operations we will be plotting.
oproot      = sprintf( '%cgemm', dt_ch );
ops( 1, : ) = sprintf( '%s_nn', oproot );
ops( 2, : ) = sprintf( '%s_nt', oproot );
ops( 3, : ) = sprintf( '%s_tn', oproot );
ops( 4, : ) = sprintf( '%s_tt', oproot );

% Generate datatype-specific operation names from the set of operations
% and datatypes.
[ opsupnames, opnames ] = gen_opsupnames( ops, stor_str, smalldims );
n_opsupnames = size( opsupnames, 1 );

%opsupnames
%opnames
%return

if 1 == 1
	%fig = figure('Position', [100, 100, 2400, 1500]);
	fig = figure('Position', [100, 100, 2400, 1200]);
	orient( fig, 'portrait' );
	set(gcf,'PaperUnits', 'inches');
	if impl == 'matlab'
		set(gcf,'PaperSize', [11.5 20.4]);
		set(gcf,'PaperPosition', [0 0 11.5 20.4]);
		set(gcf,'PaperPositionMode','manual');
	else % impl == 'octave' % octave 4.x
	   set(gcf,'PaperSize', [12 22.0]);
	   set(gcf,'PaperPositionMode','auto');
	end
	set(gcf,'PaperOrientation','landscape');
end


% Iterate over the list of datatype-specific operation names.
for opi = 1:n_opsupnames
%for opi = 1:1

	% Grab the current datatype combination.
	opsupname = opsupnames( opi, : );
	opname    = opnames( opi, : );

	str = sprintf( 'Plotting %2d: %s', opi, opsupname ); disp(str);

	% Construct filenames for the data files from templates.
	file_blissup  = sprintf( filetemp_blissup,  dirpath, thr_str, opsupname );
	file_blislpab = sprintf( filetemp_blislpab, dirpath, thr_str, opsupname );
	file_eigen    = sprintf( filetemp_eigen,    dirpath, thr_str, opsupname );
	file_open     = sprintf( filetemp_open,     dirpath, thr_str, opsupname );
	file_bfeo     = sprintf( filetemp_bfeo,     dirpath, thr_str, opsupname );
	file_vend     = sprintf( filetemp_vend,     dirpath, thr_str, opsupname );

	% Load the data files.
	%str = sprintf( '  Loading %s', file_blissup ); disp(str);
	run( file_blissup )
	run( file_blislpab )
	run( file_eigen )
	run( file_open )
	run( file_bfeo )
	run( file_vend )

	% Construct variable names for the variables in the data files.
	var_blissup  = sprintf( vartemp, thr_str, opname, 'blissup' );
	var_blislpab = sprintf( vartemp, thr_str, opname, 'blislpab' );
	var_eigen    = sprintf( vartemp, thr_str, opname, 'eigen' );
	var_open     = sprintf( vartemp, thr_str, opname, 'openblas' );
	var_bfeo     = sprintf( vartemp, thr_str, opname, 'blasfeo' );
	var_vend     = sprintf( vartemp, thr_str, opname, 'vendor' );

	% Use eval() to instantiate the variable names constructed above,
	% copying each to a simplified name.
	data_blissup  = eval( var_blissup );  % e.g. data_st_dgemm_blissup( :, : );
	data_blislpab = eval( var_blislpab ); % e.g. data_st_dgemm_blislpab( :, : );
	data_eigen = eval( var_eigen );       % e.g. data_st_dgemm_eigen( :, : );
	data_open = eval( var_open );         % e.g. data_st_dgemm_openblas( :, : );
	data_bfeo = eval( var_bfeo );         % e.g. data_st_dgemm_blasfeo( :, : );
	data_vend = eval( var_vend );         % e.g. data_st_dgemm_vendor( :, : );

	if stor_str == 'ccc'
		% Only read xsmm data for the column storage case, since that's the
		% only format that libxsmm supports.
		file_xsmm = sprintf( filetemp_xsmm,     dirpath, thr_str, opsupname );
		run( file_xsmm )
		var_xsmm  = sprintf( vartemp, thr_str, opname, 'libxsmm' );
		data_xsmm = eval( var_xsmm );     % e.g. data_st_dgemm_libxsmm( :, : );
	else
		% Set the data variable to zeros using the same dimensions as the other
		% variables.
		data_xsmm = zeros( size( data_blissup, 1 ), ...
		                   size( data_blissup, 2 ) );
	end
	%str = sprintf( '  Reading %s', var_blissup ); disp(str);
	%str = sprintf( '  Reading %s', var_blislpab ); disp(str);
	%str = sprintf( '  Reading %s', var_eigen ); disp(str);
	%str = sprintf( '  Reading %s', var_open ); disp(str);
	%str = sprintf( '  Reading %s', var_bfeo ); disp(str);
	%str = sprintf( '  Reading %s', var_xsmm ); disp(str);
	%str = sprintf( '  Reading %s', var_vend ); disp(str);

	% Plot one result in an m x n grid of plots, via the subplot()
	% function.
	if 1 == 1
	plot_l3sup_perf( opsupname, ...
	                 data_blissup, ...
	                 data_blislpab, ...
	                 data_eigen, ...
	                 data_open, ...
	                 data_bfeo, ...
	                 data_xsmm, ...
	                 data_vend, vend_str, ...
	                 nth, ...
	                 4, 7, ...
	                 cfreq, ...
	                 dflopspercycle, ...
	                 opi, impl );

	clear data_st_*gemm_*;
	clear data_blissup;
	clear data_blislpab;
	clear data_eigen;
	clear data_open;
	clear data_bfeo;
	clear data_xsmm;
	clear data_vend;

	end

end

% Construct the name of the file to which we will output the graph.
outfile = sprintf( 'l3sup_%s_%s_%s_nt%d.pdf', oproot, stor_str, arch_str, nth );

% Output the graph to pdf format.
%print(gcf, 'gemm_md','-fillpage','-dpdf');
%print(gcf, outfile,'-bestfit','-dpdf');
if impl == 'octave'
	print(gcf, outfile);
else % if impl == 'matlab'
	print(gcf, outfile,'-bestfit','-dpdf');
end

