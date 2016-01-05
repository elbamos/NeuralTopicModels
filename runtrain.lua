require 'dpnn'
require "cunn"
require "nngraph"
require 'hdf5'
require 'xlua'
require 'optim'
require 'itorch.Plot'
require 'gfx.js'
require 'display'
require 'itorch'

local pre_file = hdf5.open('../dataminingcapstone/ntm40.hdf5', 'r');
pre_w1 = pre_file:read('/w1'):all():transpose(1,2):type('torch.DoubleTensor');
examples = pre_file:read('/examples'):all():add(1):transpose(1,2):type('torch.DoubleTensor');
le_in = pre_file:read('/le'):all():transpose(1,2):type('torch.DoubleTensor');
local pre_file = hdf5.open('../dataminingcapstone/W1_pretrain_40.hdf5', 'r')
pre_w2 = pre_file:read('/layer_2/param_0'):all():type('torch.DoubleTensor')
-- the gram stack
gram_stack = nn.Sequential()
le_start = nn.Dictionary(le_in:size()[1], le_in:size()[2])
le_start.weight = le_in
-- mark this not-trainable
function le_start:updateParameters(learningRate)
end
gram_stack:add(le_start)
gram_stack:add(nn.Reshape(le_in:size()[2], true))

local lt1 = nn.Linear(pre_w2:size()[1], pre_w2:size()[2])
lt1.weight = pre_w2:transpose(1,2)
lt1.bias = torch.zeros(pre_w2:size()[2])
-- No bias in the lt module
function lt1:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   self.gradWeight = self.weight
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
end

function lt1:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end

gram_stack:add(lt1)
gram_stack:add(nn.Sigmoid())

-- the document stack
doc_stack = nn.Sequential()
local ld_1 = nn.Dictionary(pre_w1:size()[1], pre_w1:size()[2])
ld_1.weight = pre_w1
doc_stack:add(ld_1)
doc_stack:add(nn.SoftMax())
doc_stack:add(nn.Reshape(2,pre_w1:size()[2],true))
doc_stack:add(nn.SplitTable(2,3))
-- the scoring stack


din = nn.Identity()()
ld_pos, ld_neg = doc_stack(din):split(2)

ld_pos:annotate({name = 'ld_pos'})
ld_neg:annotate({name = 'ld_neg'})

g = nn.Identity()()
lt = gram_stack({g}) 

ls_pos = nn.DotProduct()({lt, ld_pos})
ls_pos:annotate({name = 'ls_pos'})
ls_neg = nn.DotProduct()({lt, ld_neg})
ls_neg:annotate({name = 'ls_neg'})

ntm = nn.gModule({g, din}, {ls_pos, ls_neg})

loss_out = nn.L1HingeEmbeddingCriterion(0.5):cuda()
ntm:cuda();
input_meta = {
    __index = function( self, k ) 
--        local i = torch.LongTensor{k}
        return {{self.g[k], self.d[k]}, -1} 
    end
}

input_data = {
    g = examples:index(2, torch.LongTensor{2}), 
    d = examples:index(2, torch.LongTensor{1,3}),
    size = function (self) 
        return self.g:size()[1] 
    end,
    shuffle = function (self, len)
        self.shuffledIndices = torch.randperm(self.g:size()[1], 'torch.LongTensor')
        self.batchIndex = 1
        self.g_batch = torch.DoubleTensor(len, self.g:size()[2])
        self.d_batch = torch.DoubleTensor(len, self.d:size()[2])
        
        self.t_batch = torch.DoubleTensor(len, 1):fill(-1)
        return {self.g_batch, self.d_batch}, self.t_batch
    end,
    next_batch = function (self, len) 
        local maxidx = math.min(self.batchIndex + len - 1, self:size())
        local batchindices = self.shuffledIndices[{{self.batchIndex, maxidx}}]
--        return {self.g:index(1, batchindices, 
--                self.d:index(1, batchindices))}
        self.g_batch:index(self.g, 1, batchindices)
        self.d_batch:index(self.d, 1, batchindices)
        self.batchIndex = self.batchIndex + len
        return {self.g_batch, self.d_batch}, self.t_batch
                --{self.g[{{self.batchIndex - len, self.batchIndex}}],
               -- self.d[{{self.batchIndex - len, self.batchIndex}}]}
    end
}
setmetatable(input_data, input_meta)
-- function for useful plotting interface
smart_trainer = function(model, criterion,
        input_data, n_classes,
        batch_size, cv_split, class_train, max_epochs, 
        optimizer, optimizer_params, 
        l1, l2, chart)
    local in_itorch = true
    if itorch._iopub == nil then
        in_itorch = false
    end
    local training_loss_history = {}
    local validation_loss_history = {}
    irregularloss = {}
    local function feval(x_in)
        local prediction = model:forward(inputs, targets)
        local losses = criterion:forward(prediction, targets)
        gradients:zero()
        local df = criterion:backward(prediction, targets)
        model:backward(inputs, df)
        -- regularize
        local norm,sign = torch.norm, torch.sign
        local loss = 0
        if type(losses) == 'number' then
            loss = losses
        else
            for i=1,#losses do loss = loss + losses[i] end
        end
	 irregularloss = loss
        if l1 ~= 0 then
            loss = loss + l1 * norm(x, 1)
            gradients:add(sign(x):mul(l1))
        end
        if l2 ~= 0 then
            loss = loss + l2 * norm(x,2)^2/2       
            gradients:add( x:clone():mul(l2))
        end
        -- accuracy matrix
        if class_train then confusion:batchAdd(prediction, targets) end
--        gradients:div(n_batches)    
        return loss, gradients
    end
    local epoch = 0
    local max_idx = input_data:size() * (1 - cv_split)
    local old_text = 'Progress:<br>'
    if in_itorch then
        local window = itorch.html(old_text)
    end
    local disp = require 'display'
    while true do
        model:training()
        if class_train then
            train_confusion = optim.ConfusionMatrix(n_classes)
            val_confusion = optim.ConfusionMatrix(n_classes)
        end
        local startTime = os.time()
        epoch = epoch + 1
        local currentError = 0
        if class_train then 
            train_confusion:zero()
            val_confusion:zero()
        end
        inputs,targets = input_data:shuffle(batch_size)
        x, gradients = model:getParameters()
        for batch_index=1, max_idx, batch_size do
            collectgarbage()
            inputs, targets = input_data:next_batch(math.min(batch_size, max_idx - batch_index - 1))
            _, f_table = optimizer(feval, x, optimizer_params)
            local thisLoss = 0
            if type(irregularloss) == 'number' then
		thisLoss = thisLoss + irregularloss
	     else
		for i=1,#irregularloss do  thisLoss = thisLoss + irregularloss[i] end
            end
            currentError = ((currentError * batch_index) + (thisLoss * batch_size)) / (batch_index + batch_size)
            if in_itorch then 
                local percCompl = math.floor(50 * batch_index / max_idx)
                local eta = ((os.time() - startTime) / (percCompl / 50)) - (os.time() - startTime)
                itorch.html(old_text .. 
                    string.format('[%d / %d] [' .. 
                        string.rep("=", percCompl - 1) .. '>' .. 
                        string.rep(".", 49 - percCompl ) .. '] ETA: %d seconds - Batch Loss: %.6f - Avg. Epoch Loss: %.6f<br>', 
                            batch_index, input_data:size(), 
                            eta, 
                            thisLoss * batch_size, 
                            currentError * batch_size
                    ),
                    window) 
            else
--                xlua:progress(batch_index, max_idx)
            end
        end
        table.insert(training_loss_history, currentError)
        -- validation
        model:evaluate()
        local validation_loss = 0
        for batch_index=max_idx,input_data:size(),batch_size do 
            collectgarbage()
            inputs, targets = input_data:next_batch(math.min(batch_size, 
                    input_data:size() - batch_index - 1))
            local prediction = model:forward(inputs, targets)
            local losses = criterion:forward(prediction, targets)
            if type(losses) == 'number' then
                validation_loss = validation_loss + losses
            else
                for i=1,#losses do validation_loss = validation_loss + losses[i] end
            end
        end
        table.insert(validation_loss_history, validation_loss)
        -- report update
        if in_itorch then
            old_text = old_text .. string.format('Epoch %d completed in %d seconds with training avg loss %.8f - ' ..
                                                    'Val loss %.8f.<br>', 
                        epoch, os.time() - startTime, currentError, validation_loss)
            itorch.html(old_text,
                    window)
            if chart then 
                plot = Plot()
                local x_vals = torch.linspace(1,epoch + 10)
                plot:line(x_vals, training_loss_history, 
                    'red', 'Training Loss')
                plot:line(x_vals, validation_loss_history, 
                    'blue', 'Validation Loss')
                plot:xaxis('Training vs Validation Loss')
                plot:legend(true)
                plot:gfx()
            end
        else
            print(string.format('Epoch %d completed in %d seconds with training avg loss %.8f - ' ..
                                                    'Val loss %.8f.', 
                        epoch, os.time() - startTime, currentError, validation_loss))
            if chart then
		 local chart_data = torch.DoubleTensor(epoch, 3)
		 for i=1,epoch do 
			chart_data[{i,1}] = i 
			chart_data[{i,2}] = training_loss_history[i]
			chart_data[{i,3}] = validation_loss_history[i]
		 end
                local chart_config = {
                    chart = 'line',
                    width = 600,
                    height = 400,
                    ylabel = 'loss',
		     xrangepad = 30,
		     digitsafterdecimal = 8,
		     labels = {'epoch', 'training', 'validation'},
                    useInteractiveGuideline = true,
		     logscale=true
                }
		 if win ~= nil then
			chart_config['win'] = win
		 end
                win = disp.plot(chart_data, chart_config)
            end
        end
        if max_epochs and epoch >= max_epochs then
            break
        end
    end
end
smart_trainer(ntm, loss_out,
        input_data, nil,
        10000, 0.2, false, 500, 
        optim.sgd, {learningRate = 0.01}, 
        0, 0.001, true)
