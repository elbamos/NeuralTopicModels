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

loss_out = nn.MarginRankingCriterion(0.5):cuda()
ntm:cuda();
input_meta = {
    __index = function( self, k ) 
--        local i = torch.LongTensor{k}
        return {{self.g[k], self.d[k]}, -1} 
    end
}

input_data = {
    g = examples:select(2,2), 
    d = torch.cat(
        examples[{{1, examples:size()[1]},  {1}}], 
	 examples[{{1, examples:size()[1]}, {3, examples:size()[2]}}]),
    colindex = 0,
    size = function (self) 
        return self.g:size()[1]
    end,
    shuffle = function (self, len)
        self.shuffledIndices = torch.randperm(self.g:size()[1], 'torch.LongTensor')
        self.batchIndex = 1
	 self.colindex = (self.colindex + 1) % (self.d:size()[2])
	 if self.colindex == 0 then
		self.colindex = 1
	 end

        self.g_batch = torch.DoubleTensor(len, 1)
        self.d_batch = torch.DoubleTensor(len, 2)

	 if colindex == 1 then 
		self.d_view = self.d[{{1, self.d:size()[1]},{1,2}}]
	 else
		self.d_view = torch.cat(  self.d[{{1, self.d:size()[1]},{1}}], 
					   self.d[{{1, self.d:size()[1]},{self.colindex + 1}}]  ) 
	 end
        
        self.t_batch = torch.CudaTensor(len, 1):fill(1)
        return {self.g_batch, self.d_batch}, 1
    end,
    next_batch = function (self, len) 
        local maxidx = math.min(self.batchIndex + len - 1, self.d:size()[1])
        local batchindices = self.shuffledIndices[{{self.batchIndex, maxidx}}]

        self.g_batch:index(self.g, 1, batchindices)
        self.d_batch:index(self.d_view, 1, batchindices)

        self.batchIndex = self.batchIndex + len
        return {self.g_batch, self.d_batch}, 1

    end
}
setmetatable(input_data, input_meta)
-- function for useful plotting interface
smart_trainer = function(model, criterion,
        input_data, n_classes,
        batch_size, cv_split, class_train, max_epochs, 
        optimizer, optimizer_params, 
        l1, l2, chart, save_prefix)
    local in_itorch = true
    if itorch._iopub == nil then
        in_itorch = false
    end
    local training_loss_history = {}
    local validation_loss_history = {}
    local function feval(x_in)
        local prediction = model:forward(inputs, targets)
        local loss = criterion:forward(prediction, targets)
        gradients:zero()
        local df = criterion:backward(prediction, targets)
        model:backward(inputs, df)
        -- regularize
        local norm,sign = torch.norm, torch.sign
	 irregularloss = torch.sum(loss)
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
    local best_val = 1e10
    local logger = optim.Logger(save_prefix .. '.log', true)
    local weightLogger = optim.Logger('weights.log', true)
    local allLogger = optim.Logger('all.log', false)
    logger:setNames({'training loss', 'validation loss'})
    weightLogger:setNames({'lt mean', 'lt var', 'ld mean', 'ld var'})
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
            local thisLoss = irregularloss
            currentError = ((currentError * batch_index) + (thisLoss)) / (batch_index + batch_size)
            if in_itorch then 
                local percCompl = math.floor(50 * batch_index / max_idx)
                local eta = ((os.time() - startTime) / (percCompl / 50)) - (os.time() - startTime)
                itorch.html(old_text .. 
                    string.format('[%d / %d] [' .. 
                        string.rep("=", percCompl - 1) .. '>' .. 
                        string.rep(".", 49 - percCompl ) .. '] ETA: %d seconds - Batch Loss: %.6f - Avg. Epoch Loss: %.6f<br>', 
                            batch_index, input_data:size(), 
                            eta, 
                            thisLoss, 
                            currentError
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
                validation_loss = validation_loss + (losses * math.min(batch_size, 
                    					input_data:size() - batch_index - 1))
            else
                validation_loss = validation_loss + torch.sum(losses) 
            end
        end
	 validation_loss = validation_loss / (input_data:size() - max_idx)
	 if validation_loss < best_val then
		local filename = string.format(save_prefix .. 'batch_%d_lr_%.4f_epoch_%d_val_%.5f_.t7', 
			batch_size, optimizer_params.learningRate, epoch, validation_loss)
		torch.save(filename, model, 'binary')
		print(string.format('Achieved best val loss %.6f on epoch %d, saved as ', validation_loss, epoch) ..
			filename)
		best_val = validation_loss
	 end
        table.insert(validation_loss_history, validation_loss)

	logger:add({currentError, validation_loss})
	if (epoch - 1) % 10 == 0 then
		local lt_weights =  model:get(2):get(3).weight
		local lt_mean = torch.mean(lt_weights)
		local lt_var = torch.std(lt_weights)
		local ld_weights = model:get(4):get(1).weight
		local ld_mean = torch.mean(ld_weights)
		local ld_var = torch.std(ld_weights)
		weightLogger:add({lt_mean, lt_var, ld_mean, ld_var})
		weightLogger:plot()
	end
        -- report update
        print(string.format('Epoch %d completed in %d seconds with training avg loss %.8f - ' ..
                                                    'Val loss %.8f.', 
                        epoch, os.time() - startTime, currentError, validation_loss))
	logger:plot()
        if max_epochs and epoch >= max_epochs then
            break
        end
    end
end
smart_trainer(ntm, loss_out,
        input_data, nil,
        20000, 0.2, false, 1000, 
        optim.sgd, {learningRate = 0.01}, 
        0, 0, true, 'fix1_batch20000_lr01_l2000Õ)
