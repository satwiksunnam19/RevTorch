import torch
import torch.nn as nn
import sys 
import random 

class ReversibleBlock(nn.Module):
    """
    This is an elementary block for building (partially) reversible architectures

    Args: 
    f_block(nn.Module): arbitaty subnetwork who's output size is equal to its input shape.
    g_block(nn.Module): arbitaty subnetwork who's output size is equal to its input shape.
    split_along_dim(integer): dimension along which the tensor need to split into the two parts required for the reversible block.
    fix_random_seed(bool): Use the random seed for the forward and backard pass if set to True.
    """

    def __init__(self,f_block,g_block,split_along_dim=1,fix_random_seed=False):
        super(ReversibleBlock,self).__init__()
        self.f_block=f_block
        self.g_block=g_block
        self.split_along_dim=split_along_dim
        self.fix_random_seed=fix_random_seed
        self.random_seeds={}

    def _init_seed(self,namespace):
        if self.fix_random_seed:
            self.random_seeds[namespace]=random.randint(0,sys.maxsize)
    
    def _set_seed(self,namesapce):
        if self.fix_random_seed:
            torch.manual_seed(self.random_seeds[namesapce])
    

    def forward(self,x):
        """
        perform forward pass of the reversible block. Does not record any gradients.
        Params: 
        x: input tensor, which should be splitted along the dim i.e 1
        returns : ouput an tensor, which is of same shape as input
        """
        x1,x2=torch.chunk(x,2,dim=self.split_along_dim)
        y1,y2=None,None
        with torch.no_grad:
            self._init_seed('f')
            y1=x1+self.f_block(x2)
            self._init_seed('g')
            y2=x2+self.g_block(y1)

        return torch.cat([y1,y2],dim=self.split_along_dim)
    
    def backward_pass(self,y,dy,retain_graph):
        """
        This performs the backward pass for the reversible block. 
        Calculates the derivatives of the block's parameters in f_block,g_block, inputs of the forward pass and it's gradients.
        params: 
        y: outputs of the reversible block.
        dy: derivatives of the outputs (y).
        retain_graph: wheather tio retain the graph at intercepted backwards.
        returns: A tuple of (block input, block input derivatives). 
        """
        # split the args in channel-wise
        y1,y2=torch.chunk(y,2,dim=self.split_along_dim)
        del y 

        assert (not y1.requires_grad), "y1 must already be detached"
        assert (not y2.requires_grad), "y2 must already be detached"

        dy1,dy2=torch.chunk(dy,2,dim=self.split_along_dim)
        del dy 
        assert (not dy1.requires_grad), "dy1 must not require grad"
        assert (not dy2.requires_grad), "dy2 must not require grad"

        # enable the autograd for y1,y2 this endures that pytorch keep track of the ops. That uses the y1,y2 as input in the DAG(Direct Acyclic Graph).

        with torch.enable_grad():
            self._set_seed('g')
            gy1= self.g_block(y1)
            gy1.backward(dy2,retain_graph=retain_graph)
        
        with torch.no_grad():
            x2=y2-gy1 # restoring the x values for the forward function.
            del y2,gy1

            # gradient of the x1 is stored in the (dy1,y1.grad)
            dx1=dy1+y1.grad
            del dy1
            y1.grad=None

        with torch.enable_grad():
            x2.requires_grad=True
            self._set_seed('f')
            fx2=self.f_block(x2)
            fx2.backward(dx1,retain_graph=retain_graph)

        with torch.no_grad():
            x1 = y1 - fx2 # Restore second input of forward()
            del y1, fx2

            # The gradient of x2 is the sum of the gradient of the output
            # y2 as well as the gradient that flows back through F
            # (The gradient that flows back through F is stored in x2.grad)
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            # Undo the split
            x = torch.cat([x1, x2.detach()], dim=self.split_along_dim)
            dx = torch.cat([dx1, dx2], dim=self.split_along_dim)
        return x,dx
    
class _ReversibleModuleFunction(torch.autograd.function.Function):
    """
    Integrates the reversible sequence into the autograd framework.
    """

    @staticmethod
    
    def forward(ctx,x,reversible_blocks,eagerly_discared_variables):
        '''
        Performs the forward pass of the reversible sequence within the autograd framework.
        : param : ctx : autograd context.
        : param : x :input tensor
        : param : eagerly_discarded_blocks: nn.Modulelist of reversible blocks
        : returns : output tensor
        '''
        assert (isinstance(reversible_blocks,nn.ModuleList))
        for block in reversible_blocks:
            assert(isinstance(block,ReversibleBlock))
            x=block(x)
        ctx.y=x.detach()
        ctx.reversible_blocks=reversible_blocks
        ctx.eagerly_discard_variables=eagerly_discared_variables

        return x 
    
    @staticmethod 

    def backward(ctx,dy):
        '''
        Performs the backward pass of a reversible sequence within the autograd framework.
        params: 
        ctx: autograd context.
        dy: derivatives of the inputs
        returns : derivatives of the inputs.
        '''
        y=ctx.y
        if ctx.eagerly_discarded_variables:
            del ctx.y 
        
        for i in range(len(ctx.reversible_blocks)-1,-1,-1):
            y,dy=ctx.reversible_blocks[i].backward_pass(y,dy,ctx.multiple_backwards)
        
        if ctx.eagerly_discarded_variables:
            del ctx.eagerly_discarded_variables
        
        return dy, None, None 
    

class ReversibleSequence(nn.Module):
    '''
    Basic Building block of (partially) reversible networks.

    A reversible sequence is a sequence of arbitary many reversible blocks. The entire sequence is reversible. 
    The activations are only saved at the end of sequence. BackProp leverages the reversible nature of the reversible sequnce to save memory.

    Args: 
    reversible_blocks(nn.Modulelist): A ModuleList that exclusively contains instances of reversible blocks.
    eagerly_discarded_variables(bool): Should the module eagerly discard the output and not retain the graph for further memory savings.
    '''
    def __init__(self,reversible_blocks,eagerly_discarded_variables):
        super(ReversibleSequence,self).__init__()
        assert (isinstance(reversible_blocks,nn.ModuleList))
        for block in reversible_blocks: 
            assert(isinstance(block,ReversibleBlock))

        self.reversible_blocks=reversible_blocks
        self.eagerly_discarded_variables=eagerly_discarded_variables

    def forward(self,x): 
        '''
        Forward pass of a reversible Sequence
        : param x: Input tensor.
        : return : Output tensor.
        '''
        x=_ReversibleModuleFunction.apply(x,self.reversible_blocks,self.eagerly_discarded_variables)
        return x
    


