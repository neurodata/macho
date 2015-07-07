%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) [2014] The Johns Hopkins University / Applied Physics Laboratory All Rights Reserved. Contact the JHU/APL Office of Technology Transfer for any additional rights.  www.jhuapl.edu/ott
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
%    http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


classdef BatchMergeTool
    %BatchMergeTool Class for using Redis DB to track massive merges
    %   Uses redis DB 3
    % 
    
    %% Properties
    properties
        jedisCli = [];
        settings = [];
        disposed = false;
    end
    
    %% Public Methods
    methods ( Access = public )
        function this = BatchMergeTool(varargin)
            % Setup Java network interface class
            try
                % Load java               
                import redis.clients.*
            catch
                try                    
                    javaaddpath(fullfile(cajal3d.getRootDir,'api','matlab','ocp','cajal3d.jar'));
                    javaaddpath(fullfile(cajal3d.getRootDir,'api','matlab','ocp','jedis-2.1.0.jar'));
                    import redis.clients.*
                catch jErr
                    fprintf('%s\n',jErr.identifier);
                    ex = MException('BatchMergeTool:JavaImportError','Could not load redis java library.');
                    throw(ex);
                end
            end
            
            %  Set up redis client
            switch nargin
                case 0
                    % No settings provided. Load from file (assuming
                    % things have been configured)try
                    try
                        load(fullfile(fileparts(which('cajal3d')),'api','matlab','ocp','semaphore_settings.mat'));
                        this.settings = ocp_settings;
                    catch jErr
                        fprintf('%s\n',jErr.identifier);
                        ex = MException('BatchMergeTool:NotConfigured','There is no local $CAJAL3D_ROOT/api/matlab/ocp/semaphore_settings.mat file to load server info.');
                        throw(ex);
                    end
                    
                case 2
                    % Use provided settings
                    this.settings.server = varargin{1};
                    this.settings.port = varargin{2};
                    
                otherwise
                    ex = MException('BatchMergeTool:InvalidConstructor','Invalid params to the constructor.');
                    throw(ex);
            end
            
            % Create jedis-cli object
            this.jedisCli = jedis.Jedis(this.settings.server,this.settings.port);
            % Select database 3
            this.selectDatabaseIndex(3);
            this.disposed = false;
        end
        
        function this = delete(this)
            % destroy java object
            if this.disposed == false
                this.jedisCli.quit();
                this.jedisCli = 0;
                this.disposed = true;
            end
        end
        
        function reset(this)
            response = this.jedisCli.flushDB();
            if strcmp(char(response),'OK') == 0
                error('BatchMergeTool:ResetFail','Failed to reset DB. Response: %s',char(response))
            end
        end
        
        function selectDatabaseIndex(this,index)
            this.jedisCli.select(index);
        end
        
        
        function merge(this,parent,children)
            % Always add the parent (to cover case of new key) so the
            % parent is in the set and intersects will work properly.
            
            if isempty(children)
               warning('BatchMergeTool:NoChildren','No children to merge. Skipping.'); 
            end
            
            this.jedisCli.sadd(num2str(parent),num2str(parent));
            for ii = 1:length(children)
                this.jedisCli.sadd(num2str(parent),num2str(children(ii)));
            end
        end
        
        function consolidate(this)
            % Work through the table merging duplicated merges into 1 big
            % object
            
            % Get all keys
            key_list = this.jedisCli.keys('*');
            key = key_list.iterator();
            
            % Convert java object to matlab object
            keys = cell(key_list.size(),1);
            cnt = 1;
            while key.hasNext()
                keys{cnt} = key.next();
                cnt = cnt + 1;
            end
            
            % For each key work through the table checking for duplicates
            for current_ind = 1:size(keys,1) - 1
                % Check each entry "below" the current one, restarting
                % every time you suck in a set
                
                % prep java array for the sinter command
                jKeys = javaArray('java.lang.String',2);
                jKeys(1) = java.lang.String(keys{current_ind});
                
                target_ind = current_ind+1;
                
                while target_ind <= size(keys,1)
                    % compute intersect                    
                    jKeys(2) = java.lang.String(keys{target_ind});
                    inter = this.jedisCli.sinter(jKeys);
                    if inter.size() == 0
                        target_ind = target_ind + 1;
                    else
                        % There is a hit on the intersection. Absorb the
                        % target set and start check over.
                        
                        % union the sets and store in the current set
                        this.jedisCli.sunionstore(keys{current_ind},jKeys);
                        
                        % Delete target set and key
                        this.jedisCli.del(keys{target_ind});                        
                        
                        % Reset check for the current set
                        target_ind = current_ind+1;
                    end                                     
                end                
            end            
        end        
        
        function write_csv(this,filename)
            % Take each key and write out its members to a csv file.
            
            % Get all the keys.            
            key_list = this.jedisCli.keys('*');
            key = key_list.iterator();            
            
            
            try
                % Open the file
                fid = fopen(filename,'wt');
                
                % Loop through keys
                while key.hasNext()
                    curr_key = key.next();                    
                    members = this.jedisCli.smembers(curr_key);                    
                    member = members.iterator();      
                    
                    % Print parent
                    fprintf(fid,'%s',curr_key);
                    
                    % Print children
                    while member.hasNext()
                        val = member.next();
                        if strcmpi(val,curr_key)
                            continue;
                        end
                        fprintf(fid,',%s',val);
                    end                    
                    
                    fprintf(fid,'\n');                    
                end
                
            catch ex
                fclose(fid);
                throw(ex);
            end
            
            % Close the file.
            fclose(fid);
            
        end
        
        function merge_set = transitive_closure(this,id_cellarray)
            % This method runs transitive_closure on a cell array and
            % outputs a cell array.  The format is an Nx1 cell array where
            % each cell is an array of IDs that should be merged.
            idList2 = this.convertToList(id_cellarray);
            [A,LUT] = this.convertToMatrix(idList2);
            B2 = this.warshall1b(A);
            [merge_set,~] = this.convertFromMatrix(B2,LUT);
        end
        
        function num = get_num_parents(this)
            key_list = this.jedisCli.keys('*');
            num = key_list.size();
        end
        
        function num = get_num_children(this,key)
            num = double(this.jedisCli.scard(num2str(key)));
        end
        
    end
    
    %% Private Methods
    methods(Access = private)
        function A = warshall1b(this,A)
        % This method implements warshall transitive closure.  This is
        % useful for combining IDs into a single merge list

            n = size(A,1);

            for i = 1:n
                for j = 1:n
                    if A(i,j) == 1
                        for k = 1:n
                            if A(j,k) == 1
                                A(i,k) = 1;
                            end
                        end
                    end
                end
            end
        end
        
        function [A,cc] = convertToMatrix(this,idList)
            % This helper function converts Nx2 array to an adjacency matrix for
            % transitive closure
            n = size(idList,1);

            cc = unique(idList);
            cc(cc==0) = [];
            A = false(length(cc),length(cc)); % this is index - need a LUT later

            % Need to form index matrix

            for i = 1:n
                %for each row, nCk the elements
                val =  idList(i,:);
                val(val==0) = [];

                for j = 1:length(val)
                    idx(j) = find(val(j)==cc);
                end

                combo = combnk(idx,2);

                for j = 1:size(combo,1)
                    A(combo(j,1),combo(j,2)) = 1;
                    A(combo(j,2),combo(j,1)) = 1;
                end

            end
        end
        
        function idList2 = convertToList(this,idList)
            %This helper function takes a cell array and makes an Nx2
            %matrix

            n = length(idList);

            idList2 = [];
            count = 1;
            for i = 1:n

                val = idList{i};
                if ~isempty(val) %handle empty rows
                    val(val==0) = []; %check corner case

                    combo = combnk(val,2);

                    for j = 1:size(combo,1)
                        idList2(count,1) = combo(j,1);
                        idList2(count,2) = combo(j,2);
                        count = count + 1;
                    end
                end
            end
        end
        
        function [mergeList, globalMerge] = convertFromMatrix(this,A, LUT)
        % This helper function takes warshall output to a Nx2 cell array
        % where each cell contains a vector of IDs to merge
        
            n = size(A,1);

            %Symmeterize
            [r,c] = find(A>0);
            for i = 1:length(r)
                A(c(i),r(i)) = 1;
            end

            globalMerge = [];
            mergeList = [];
            count = 1;
            for i = 1:n
               idToMerge = find(A(i,:) > 0);
               idToMerge = idToMerge(idToMerge > i);
               if ~isempty(idToMerge)
                    mergeList{count} = [LUT(i), LUT(idToMerge)'];
                    globalMerge(count).mergeParent = i;
                    globalMerge(count).idToMerge= idToMerge;
                    globalMerge(count).globalIdx = count;
                    globalMerge(count).globalList = [i; idToMerge(:)];
                    A(:,i) = 0;
                    A(i,:) = 0;
                    A(idToMerge,:) = 0;
                    A(:,idToMerge) = 0;
                    count = count+1;
               end
            end
        end



    end
    
end
