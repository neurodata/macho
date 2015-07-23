function ilastik_runIlastik(ilastikProjectPath, stackPattern, outputPath)
% J. Matelsky - jordan.matelsky@jhu.edu

% ilastikProjectPath    The path to the Ilastik project that contains the
%                       classifiers you wish to use.
% stackPattern          The *-matched pattern of files which should be run
%                       through the classifier. e.g. stack*.png
% outputPath            The path to which to save the output (should LONI
%                       be in play here?)


% Usage Example:
%   ilastik_runIlastik('~/ilastik-Linux/', './tmp/results/{nickname}_results.tiff', "stack_name_base*.png")

system(['python ./ilastik/ilastikRun.py ' ilastikProjectPath ' ' outputPath ' ' stackPattern]);
% Nest in double-quotes to prevent shell auto-expansion

% TODO: Check for failure in Ilastik's exit-code here,
%       and notify the user accordingly.
%       We'll need to store this output to be sure that
%       we can access the files that have been created
%       successfully.
