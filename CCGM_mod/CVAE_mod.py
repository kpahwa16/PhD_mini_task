
import torch
import torch.nn as nn                        
import torch.nn.functional as F              

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#CVAE 
class CVAE(nn.Module):
    '''
    VAE with causal layer and twp heads and tails (label and image)
    '''
    def __init__(self, num_labels, mode, 
                      img_size = 50, 
                      num_channels = 1, 
                      label_width = 16, 
                      label_depth = 1,
                      mask_width = 16,
                      mask_depth = 1,
                      image_width = 512,
                      image_depth = 2,
                      DAG_bias = False,
                      name = 'no_name',
                      z_mask = True,
                      u_mask = True):
        super().__init__()
    
        # Class Variables
        self.num_labels = num_labels
        self.mode = mode
        self.img_size = img_size
        self.num_channels = num_channels
        self.label_width = label_width
        self.label_depth = label_depth
        self.image_width = image_width
        self.image_depth = image_depth
        self.mask_width = mask_width
        self.mask_depth = mask_depth
        self.DAG_bias = DAG_bias
        self.name = name
        self.losses = None
        self.z_mask = z_mask
        self.use_u_mask = u_mask
       

        ## Image Enc/Dec
        self.img_enc = image_Encoder(self.img_size, 
                                     self.num_channels, 
                                     self.num_labels,
                                     self.image_depth, 
                                     self.image_width)

       
        self.img_dec = image_Decoder(self.img_size,
                                            self.num_channels, 
                                            self.num_labels,
                                            self.image_depth, 
                                            self.image_width,
                                            self.img_size)
       
        self.img_decs = nn.ModuleList([image_Decoder(self.img_size,
                                                self.num_channels,
                                                1,
                                                self.image_depth, 
                                                self.image_width,
                                                self.img_size) for _ in range(num_labels)])

        self.label_encs = nn.ModuleList([label_Encoder(1,depth=label_depth,width=label_width) for _ in range(num_labels)])
        self.label_decs = nn.ModuleList([label_Decoder(1,depth=label_depth,width=label_width) for _ in range(num_labels)])

    
        if self.z_mask: 
            self.mask_nets = nn.ModuleList([label_Encoder(self.num_labels,mask_depth,mask_width) for _ in range(num_labels)])
        if self.use_u_mask:  self.u_mask = label_Encoder(self.num_labels, mask_depth, mask_width)
    
        self.causal = Dag_Layer(num_labels, DAG_bias)

    def forward(self, x_in,causal = True,mask=None,val = 0):

        if self.mode[0] == 'label':
            labels = x_in.view(-1, self.num_labels)
        elif self.mode[0] == 'image':
            image =  x_in.view(-1, self.num_channels * self.img_size * self.img_size)
            print("image.shape", image.shape)
            

        if self.mode[0] == 'label':
            means = torch.zeros(labels.size()[0], self.num_labels, device=device)
            logvars = torch.zeros(labels.size()[0], self.num_labels, device=device)
           
            for i in range(self.num_labels):
                means[:, i], logvars[:, i] = self.label_encs[i](labels[:, i])

        elif self.mode[0] == 'image':
            means, logvars = self.img_enc(image)

        ### Sampling: Sample z from latent space using mu and logvar
        if self.training:
            z = torch.randn_like(means, device=device).mul(torch.exp(0.5 * logvars)).add_(means)
        else:
            z = means

        ### Masking and Causal Layer
        if causal:
            if mask is not None: z[:,mask] = val  # mask in for exo variables
            
           
            if self.z_mask: 
                z_hadamard = self.causal(z,self.z_mask)
            else:
                z_recon = self.causal(z)

            if self.z_mask: 
                z_recon = torch.zeros(z_hadamard.shape[0],self.num_labels,device=device)
                z_logvars = torch.zeros(z_hadamard.shape[0],self.num_labels,device=device)
                for i in range(self.num_labels):
                    z_recon[:, i], z_logvars[:,i] = self.mask_nets[i](z_hadamard[:,:,i])

            if mask is not None: z_recon[:,mask] = val # mask out for endo variables
        else: 
            z_recon = z

     
            if self.mode[1] == 'label': labels = x_in.view(-1, self.num_labels)


      
        if self.mode[1] == 'label':
            reconstruction = self.decode_labels(z_recon)
        elif self.mode[1] == 'image':
            reconstruction = self.img_dec(z_recon)
        

        return reconstruction, means, logvars, z, z_recon, z_logvars

    def decode_labels(self, z):
        z = z.view(-1, self.num_labels)
        u_recon = torch.zeros(z.size()[0], self.num_labels)
        for i in range(self.num_labels):
            u_recon[:, i:i + 1] = self.label_decs[i](z[:, i:i + 1])
        return u_recon
    
    def decode_images(self, z):
        z = z.view(-1, self.num_labels)
        print("z.shape", z.shape)
        x_recon = torch.zeros(z.size()[0],  self.num_labels, self.num_channels * self.img_size * self.img_size)
        print("x_recon.shape", x_recon.shape)
        for i in range(self.num_labels):
            x_recon[:, i,:] = self.img_decs[i](z[:, i])
        return torch.sigmoid(x_recon.sum(axis=1))


class Dag_Layer(nn.Module):

    def __init__(self, n_features, bias):
        super(Dag_Layer, self).__init__()

        self.n_features = n_features

        self.A = nn.Parameter(torch.zeros(n_features, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))

        self.I = nn.Parameter(torch.eye(n_features), requires_grad=False)
        self.I.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_features))
        else:
            self.register_parameter('bias', None)

    @property
    def M(self):
        return self.A + torch.diag(self.b) #torch.diag(torch.sigmoid(self.b))


    def forward(self, x, z_mask=False):
        if z_mask:
            return torch.einsum('ln,bl->bln', self.M, x)
        else:
            return F.linear(x, self.M.t())



class image_Encoder(nn.Module):

    def __init__(self, img_size, channels, num_labels , depth, width):
        super(image_Encoder, self).__init__()
     

        self.img_size = img_size
        self.channels = channels
        self.depth = depth
        self.width = width
        self.num_labels = num_labels
        self.input_dim = self.channels * self.img_size * self.img_size
        self.output_dim = num_labels

        self.image2hidden = nn.Linear(in_features=self.input_dim, out_features=width)
        hidden_layers = []
        for i_hlayer in range(depth-1): 
            hidden_layers.append(nn.Linear(in_features=width, out_features=width))
            hidden_layers.append(nn.GELU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.hidden2mean = nn.Linear(in_features=width, out_features=num_labels)
        self.hidden2logvar = nn.Linear(in_features=width, out_features=num_labels)

    def forward(self, img):
        x = img.view(-1, 1)
        x = self.hidden_layers(F.gelu(self.image2hidden(x)))
        return self.hidden2mean(x), self.hidden2logvar(x)

class image_Decoder(nn.Module):

    def __init__(self, img_size, channels, num_labels, depth, width,height):
        super(image_Decoder, self).__init__()
   
        self.img_size = img_size
        self.channels = channels
        self.depth = depth
        self.width = width
        self.height = height
        self.num_labels = num_labels
        self.input_dim = self.num_labels
        self.output_dim = self.channels * self.img_size * self.height

        self.latent2hidden = nn.Linear(in_features=num_labels, out_features=width)
        hidden_layers = []
        for i_hlayer in range(depth - 1): 
            hidden_layers.append(nn.Linear(in_features=width, out_features=width))
            hidden_layers.append(nn.GELU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.hidden2image = nn.Linear(in_features=width, out_features=self.output_dim)

    def forward(self, z):
        z = z.view(-1, self.num_labels)
        z = self.hidden_layers(F.gelu(self.latent2hidden(z)))
        if self.num_labels > 1:
            return torch.sigmoid(self.hidden2image(z))
        else:
            return self.hidden2image(z)




class label_Encoder(nn.Module):

    def __init__(self, input_dim = 1, depth = 1, width=20):
        super(label_Encoder, self).__init__()

        # Class Variables
        self.width = width
        self.depth = depth
        self.input_dim = input_dim

        # Network
        self.label2hidden = nn.Linear(in_features=input_dim, out_features=width)
        hidden_layers = []
        for i_hlayer in range(depth - 1): 
            hidden_layers.append(nn.Linear(in_features=width, out_features=width))
            hidden_layers.append(nn.GELU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.hidden2mean = nn.Linear(in_features=width, out_features=1)
        self.hidden2logvar = nn.Linear(in_features=width, out_features=1)

    def forward(self, u):
        u = u.view(-1, self.input_dim)
        u = self.hidden_layers(F.gelu(self.label2hidden(u)))
        return torch.squeeze(self.hidden2mean(u)), torch.squeeze(self.hidden2logvar(u))


class label_Decoder(nn.Module):
  
    def __init__(self,  input_dim = 1, depth = 1, width=20):
        super(label_Decoder, self).__init__()

    

        self.width = width
        self.depth = depth
        self.input_dim = input_dim

        self.latent2hidden = nn.Linear(in_features=self.input_dim, out_features=width)
        hidden_layers = []
        for i_hlayer in range(depth): 
            hidden_layers.append(nn.Linear(in_features=width, out_features=width))
            hidden_layers.append(nn.GELU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.hidden2label = nn.Linear(in_features=width, out_features=1)

    def forward(self, z):
        z = z.view(-1, self.input_dim)
        z = self.hidden_layers(F.gelu(self.latent2hidden(z)))
        u_recon = self.hidden2label(z)
        return u_recon





