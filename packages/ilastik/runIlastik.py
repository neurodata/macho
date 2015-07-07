#!/usr/bin/python

## 
# (c) [2014] The Johns Hopkins University / Applied Physics Laboratory All Rights Reserved.
# Contact the JHU/APL Office of Technology Transfer for any additional rights.
# www.jhuapl.edu/ott
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ILASTIK_PATH = "/usr/bin/ilastik"

import sys, os
# read in command line args
params = list(argv)[1:]

project        = params[0]
outfile_format = params[1]
stack_pattern  = params[2]

args = ['--headless', '--project', project,
        '--output_format', 'tiff',
        '--output_filename_format', outfile_format,
        '"', stack_pattern, '"']

cmd = ' '.join([ILASTIK_PATH] + args)

# Call Ilastik.
process = Popen(cmd, stdout=PIPE, stderr=PIPE)
output = process.communicate()
proc_error = output[1]
proc_output = output[0]
exit_code = process.wait()

# Write std out stream
print "#######################\n"
print "  Output From Ilastik  \n"
print "#######################\n\n\n"
print proc_output

# If exit code != 0 exit
if exit_code != 0:
    # it bombed.  Write out matlab errors and return error code
    sys.stderr.write("Error from Ilastik:\n\n")
    sys.stderr.write(proc_error)
    exit(exit_code)
