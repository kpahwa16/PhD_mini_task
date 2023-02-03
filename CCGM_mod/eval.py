import torch
import torchvision                            
import torchvision.transforms as transforms    
from torchvision.utils import make_grid        
import matplotlib.pyplot as plt                
import numpy as np
import math
import torch.nn.functional as F   
from skimage import data, color

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_latent_space(model,mode,device,train_dataset,concept1=0,concept2=1):
    if mode == 'label':
        for i,(u,l) in enumerate(train_dataset):
            label_recon, latent_mu, latent_logvar, latent_sample, latent_recon,_  = model(l.to(device))
            plt.scatter(latent_mu[:,concept1].cpu().detach().numpy(),latent_mu[:,concept2].cpu().detach().numpy())
            plt.xlabel('Concept {}'.format(concept1))
            plt.ylabel('Concept {}'.format(concept2))
            plt.title('Latent Space Distribution',fontsize=14,fontweight='bold')
            break
    elif mode == 'image':
        for i,(u,l) in enumerate(train_dataset):
            if model.z_mask == True:
                image_recon, latent_mu, latent_logvar, latent_sample, latent_recon,_  = model(l.to(device))
            else:
                image_recon, latent_mu, latent_logvar, latent_sample, latent_recon  = model(l.to(device))
            print(latent_recon)
            plt.scatter(latent_mu[:,concept1].cpu().detach().numpy(),latent_mu[:,concept2].cpu().detach().numpy())
            plt.xlabel('Concept {}'.format(concept1))
            plt.ylabel('Concept {}'.format(concept2))
            plt.title('Latent Space Distribution',fontsize=14,fontweight='bold')
            break


def plot_label_latent(model,mode,device,concepts=[0,1],sweep_ranges=[(-1.5, 1.5),(-1.5, 1.5)], default_vals = [0,0,0,0]):

    concept1, concept2 = concepts
    with torch.no_grad():
      
        # create a sample grid in 2d latent space
        linspace_1 = np.linspace(sweep_ranges[0][0],sweep_ranges[0][1],6)
        linspace_2 = np.linspace(sweep_ranges[1][0],sweep_ranges[1][1],6)
        latents = torch.tensor(default_vals).to(device)*torch.ones(len(linspace_2), len(linspace_1), 4, device=device)
        latents = model.causal(latents)
        for i, lx in enumerate(linspace_1):
            for j, ly in enumerate(linspace_2):
                latents[j, i, concept1] = lx
                latents[j, i, concept2] = ly
        
        latents = latents.view(-1, 4) 
  
    if mode == 'label':
        x_recon = torch.zeros((latents.shape[0],3, 96, 96), device=device)
        label_recon = model.decode_labels(latents)
        for i,iLabel in enumerate(label_recon):
            x_recon[i] = torch.from_numpy(label_to_Img(iLabel.cpu().detach().numpy(),96,gray=True))
            plt.close()
    elif mode == 'image':
        if model.split_Dec: x_recon = model.decode_images(latents).reshape(latents.shape[0],model.num_channels,model.img_size,model.img_size)
        else: x_recon = model.img_dec(latents).reshape(latents.shape[0],model.num_channels,model.img_size,model.img_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    show_image(torchvision.utils.make_grid(x_recon.data,6,6).cpu())
    fig.savefig('./test/' + model.name + '_' + mode + f'_latent_{concept1}_{concept2}.png')



def bce_loss(image, reconstruction,image_size,channel=3):
    # Binary Cross Entropy for batch
    BCE = F.binary_cross_entropy(input=reconstruction.view(-1, image_size*image_size), target=image.view(-1, image_size*image_size), reduction='sum')
    return BCE
    
def reconstruction(model,test_dataset,mode,device,num_test=5):
    
    test_images = np.random.choice(range(len(test_dataset)),num_test)
    
    label_images = []
    true_images = []
    MSEs = []
    BCEs = []
    if mode == 'label':
        for i,(u,l) in enumerate(test_dataset):
          if i in test_images:
            label_recon, latent_mu, latent_logvar, latent_sample, latent_recon,_  = model(l.to(device))
            label_images.append(torch.from_numpy(label_to_Img(label_recon.cpu().detach().numpy()[0],96,gray=True)))
            true_images.append(u[0])

    elif mode == 'image':
        for i,(u,l) in enumerate(test_dataset):
            if i in test_images:
                if model.z_mask == True: 
                    image_recon, latent_mu, latent_logvar, latent_sample, latent_recon,_  = model(l.to(device))
                else:
                    image_recon, latent_mu, latent_logvar, latent_sample, latent_recon  = model(l.to(device))
                label_images.append(image_recon.reshape(model.num_channels,model.img_size,model.img_size))
                true_images.append(u[0])
                BCEs.append(bce_loss(label_images[-1].float().to(device),u[0].to(device),model.img_size,model.num_channels))

    grid_img_labels = make_grid(label_images)
    grid_img_true = make_grid(true_images)
    
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.set_title('True Images',fontsize = 12,fontweight='bold')
  
    ax1.imshow(grid_img_true.permute(1, 2, 0).cpu());
    ax2.imshow(grid_img_labels.permute(1, 2, 0).cpu());

    fig.savefig('./test/' + model.name + '_' + mode + 'reconstruction.png')
    
def sun_interventions(model,test_dataset,mode,device,image_num = None,mask_val = [-1.5, -0.5, 0.5, 1.5]):
    
    if image_num == None:
        image_num = np.random.choice(range(len(test_dataset)),1)
    
    images = []
    for i,(u,l) in enumerate(test_dataset):   
        if i in image_num:
            u,l = u.to(device), l.to(device)
            print(l)
            if mode == 'label' and u.shape[1] > 1: 
              u = transforms.functional.rgb_to_grayscale(u[0,:3,:,:])
              images.append(u.to(device))
            else: 
              images.append(u[0].to(device))
            for mask in range(4): 
                if mode == 'label':
                    label_recon, latent_mu, latent_logvar, latent_sample, latent_recon = model(l.to(device), mask=1,val=mask_val[mask])
                    images.append(torch.from_numpy(label_to_Img(label_recon.cpu().detach().numpy()[0],96,gray=True)).to(device))
                elif mode == 'image':
                    image_recon, latent_mu, latent_logvar, latent_sample, latent_recon  = model(l.to(device), mask=1,val=mask_val[mask])
                    # print(latent_recon)
                    images.append(image_recon.reshape(model.num_channels,model.img_size,model.img_size).to(device))
            break

    grid_img_labels = make_grid(images)
    
    fig, ax1 = plt.subplots(1,1)
    ax1.set_title('Interventions Image {}'.format(image_num[0]),fontsize = 12,fontweight='bold')
    ax1.set_xlabel('Image Type')
    ax1.set_xticks(50 + 100 * np.arange(5))
    ax1.set_xticklabels(['True'] + ['{:d}'.format(x) for x in range(4)])
    ax1.imshow(grid_img_labels.permute(1, 2, 0).cpu())
    fig.savefig('./test/' + model.name + '_' + mode + '_intervention.png')
  
def interventions(model,test_dataset,mode,device,image_num = None,mask_val = [0,0,0,0]):
    
    if image_num == None:
        image_num = np.random.choice(range(len(test_dataset)),1)
    
    images = []
    for i,(u,l) in enumerate(test_dataset):   
        if i in image_num:
            u,l = u.to(device), l.to(device)
            print(l)
            if mode == 'label' and u.shape[1] > 1: 
              u = transforms.functional.rgb_to_grayscale(u[0,:3,:,:])
              images.append(u.to(device))
            else: 
              images.append(u[0].to(device))
            for mask in range(4): 
                if mode == 'label':
                    label_recon, latent_mu, latent_logvar, latent_sample, latent_recon, _ = model(l.to(device), mask=mask,val=mask_val[mask])
                    images.append(torch.from_numpy(label_to_Img(label_recon.cpu().detach().numpy()[0],model.img_size,gray=True)).to(device))
                elif mode == 'image':
                    if model.z_mask == True:
                        image_recon, latent_mu, latent_logvar, latent_sample, latent_recon,_  = model(l.to(device), mask=mask,val=mask_val[mask])
                    else:
                        image_recon, latent_mu, latent_logvar, latent_sample, latent_recon  = model(l.to(device), mask=mask,val=mask_val[mask])
                    # print(latent_recon)
                    images.append(image_recon.reshape(model.num_channels,model.img_size,model.img_size).to(device))
            break

    grid_img_labels = make_grid(images)
    
    fig, ax1 = plt.subplots(1,1)
    ax1.set_title('Interventions Image {}'.format(image_num[0]),fontsize = 12,fontweight='bold')
    ax1.set_xlabel('Image Type')
    ax1.set_xticks(50 + 100 * np.arange(5))
    ax1.set_xticklabels(['True'] + ['{:d}'.format(x) for x in range(4)])
    ax1.imshow(grid_img_labels.permute(1, 2, 0).cpu())
    fig.savefig('./test/' + model.name + '_' + mode + '_intervention.png')
        
def projection(theta, phi, x, y, base = -0.5):
    b = y-x*math.tan(phi)
    shade = (base - b)/math.tan(phi + 1e-10)
    return shade

def label_to_Img(label_array,img_size,gray=False):

    i = label_array[0]
    j = label_array[1]
    shade =  label_array[2]
    mid = label_array[3]
    
    fig = plt.figure(figsize=(1,1),dpi = img_size);
    theta = i*math.pi/200.0
    phi = j*math.pi/200.0
    x = 10 + 8*math.sin(theta)
    y = 10.5 - 8*math.cos(theta)
    
    ball = plt.Circle((x,y), 1.5, color = 'firebrick')
    gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)
    
    light = projection(theta, phi, 10, 10.5, 20.5)
    sun = plt.Circle((light,20.5), 3, color = 'orange')
    
    shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
    ax = plt.gca()
    ax.add_artist(gun)
    ax.add_artist(ball)
    ax.add_artist(sun)
    ax.add_artist(shadow)
    ax.set_xlim((0, 20))
    ax.set_ylim((-1, 21))
    plt.axis('off');
    ax.add_artist(shadow)
    
    # fig to matrix
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    mplimage = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    if not gray:
        return mplimage.transpose(2,0,1)
    else:
        return color.rgb2gray(mplimage).reshape(1,img_size,img_size)