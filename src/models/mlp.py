import torch
import torch.nn as nn
import torch.nn.functional as F


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)
        
        
class ResidualMLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
        ResidualMLP with skip connections to improve gradient flow and learning
        '''
        super(ResidualMLP, self).__init__()
        
        self.num_layers = num_layers
        
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model for single layer
            self.linear = nn.Linear(input_dim, output_dim)
            self.linear_or_not = True
        else:
            # Multi-layer with residual connections
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            
            # Input projection
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            
            # Hidden layers
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                
            # Output projection
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            # Batch norms (applied before activation)
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                
            # Skip connection from input to output (if dimensions differ)
            if input_dim != output_dim:
                self.skip = nn.Linear(input_dim, output_dim)
            else:
                self.skip = nn.Identity()
                
         # IMPORTANT: Call initialization method after all layers are defined
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights using He initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming/He initialization - designed for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP with residual connections
            identity = x  # Save input for final skip connection
            
            # First layer (no residual yet)
            h = F.relu(self.batch_norms[0](self.linears[0](x)))
            
            # Middle layers with residual connections
            for layer in range(1, self.num_layers - 1):
                # Save previous layer output
                residual = h
                # Apply linear, batch norm, and activation
                h = self.linears[layer](h)
                h = self.batch_norms[layer](h)
                h = F.relu(h)
                # Add residual connection
                h = h + residual
            
            # Final layer
            out = self.linears[self.num_layers - 1](h)
            
            # Add skip connection from input to output
            if self.num_layers > 2:
                out = out + self.skip(identity)
                
            return out
        
        
class ResidualMLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
        ResidualMLP for the actor with tanh activations
        '''
        super(ResidualMLPActor, self).__init__()
        
        self.num_layers = num_layers
        
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model for single layer
            self.linear = nn.Linear(input_dim, output_dim)
            self.linear_or_not = True
        else:
            # Multi-layer with residual connections
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            
            # Input projection
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            
            # Hidden layers
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                
            # Output projection
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            # Skip connection from input to output (if dimensions differ)
            if input_dim != output_dim:
                self.skip = nn.Linear(input_dim, output_dim)
            else:
                self.skip = nn.Identity()
                
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights using Xavier initialization for tanh activations"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization - designed for tanh/sigmoid activations
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP with residual connections
            identity = x  # Save input for final skip connection
            
            # First layer (no residual yet)
            h = torch.tanh(self.linears[0](x))
            
            # Middle layers with residual connections
            for layer in range(1, self.num_layers - 1):
                # Save previous layer output
                residual = h
                # Apply linear and activation
                h = torch.tanh(self.linears[layer](h))
                # Add residual connection
                h = h + residual
            
            # Final layer
            out = self.linears[self.num_layers - 1](h)
            
            # Add skip connection from input to output
            if self.num_layers > 2:
                out = out + self.skip(identity)
                
            return out
        
class ResidualMLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
        ResidualMLP for the critic with tanh activations
        '''
        super(ResidualMLPCritic, self).__init__()
        
        self.num_layers = num_layers
        
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model for single layer
            self.linear = nn.Linear(input_dim, output_dim)
            self.linear_or_not = True
        else:
            # Multi-layer with residual connections
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            
            # Input projection
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            
            # Hidden layers
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                
            # Output projection
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            # Skip connection from input to output (if dimensions differ)
            if input_dim != output_dim:
                self.skip = nn.Linear(input_dim, output_dim)
            else:
                self.skip = nn.Identity()
    
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights using Xavier initialization for tanh activations"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization - designed for tanh/sigmoid activations
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP with residual connections
            identity = x  # Save input for final skip connection
            
            # First layer (no residual yet)
            h = torch.tanh(self.linears[0](x))
            
            # Middle layers with residual connections
            for layer in range(1, self.num_layers - 1):
                # Save previous layer output
                residual = h
                # Apply linear and activation
                h = torch.tanh(self.linears[layer](h))
                # Add residual connection
                h = h + residual
            
            # Final layer
            out = self.linears[self.num_layers - 1](h)
            
            # Add skip connection from input to output
            if self.num_layers > 2:
                out = out + self.skip(identity)
                
            return out