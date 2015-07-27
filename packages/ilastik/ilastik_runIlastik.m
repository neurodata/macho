function ilastikReturn = ilastik_runIlastik(ilastikProjectPath, stackPattern, outputPath)
% J. Matelsky - jordan.matelsky@jhu.edu

% ilastikProjectPath    The path to the Ilastik project that contains the
%                       classifiers you wish to use.
% stackPattern          The *-matched pattern of files which should be run
%                       through the classifier. e.g. stack*.png
% outputPath            The path to which to save the output


% Usage Example:
%   ilastik_runIlastik('~/ilastik-Linux/', './tmp/results/{nickname}_results.tiff', "stack_name_base*.png")


MACHO_PATH = '~/Documents/ocp/macho';

ilastikReturn = system(['python ' MACHO_PATH '/code/packages/ilastik/ilastikRun.py ' ilastikProjectPath ' ' outputPath ' ' stackPattern]);
% Nest the above in double-quotes to prevent shell auto-expansion