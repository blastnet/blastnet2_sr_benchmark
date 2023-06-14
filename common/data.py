#written by W.T. Chung
#functions for loading and augmenting data
import numpy as np
import torch
from .functions import divergence
import csv
from scipy.ndimage import uniform_filter


def my_box_filter(phi,fw):
    filtered = uniform_filter(phi,size=fw)
    return filtered[fw//2::fw,fw//2::fw,fw//2::fw]

# for favre filtering to create features
def my_favre_filter(phi,rho,fw):
    return my_box_filter(rho*phi,fw)/my_box_filter(rho,fw)


def my_read_csv(path):
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(row[col])
    return data

def get_mean_std_test():
    #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
    my_mean = torch.tensor([0.24,29.0, 29.0, 29.0])
    my_std = torch.tensor([0.068,48.0, 48.0, 48.0])

    return my_mean, my_std

def get_mean_std_extrapRe():
    #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
    my_mean = torch.tensor([0.23,34.0, 34.0, 34.0])
    my_std = torch.tensor([0.059,55.0, 55.0, 55.0])

    return my_mean, my_std

def get_mean_std_extrapffcm():
    #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
    my_mean = torch.tensor([11,-0.051, -0.051, -0.051])
    my_std = torch.tensor([4.6,1.4, 1.4, 1.4])

    return my_mean, my_std

def get_mean_std():
    #evaluated thorugh mean of train data in rho and (u,v,w lumped together)
    my_mean = torch.tensor([0.24,28.0, 28.0, 28.0])
    my_std = torch.tensor([0.068,48.0, 48.0, 48.0])

    return my_mean, my_std


def get_file(idx,train_dict,data_path,mode,upscale):
    hash_id = train_dict['hash'][idx]
    scalars = ['RHO_kgm-3_id','UX_ms-1_id','UY_ms-1_id','UZ_ms-1_id']
    #return a 4channel numpy array of the 4 scalars
    X = []
    for scalar in scalars:
        xpath = data_path+'LR_'+str(upscale)+'x/'+mode+'/'+scalar+hash_id+'.dat'
        X.append(np.memmap(xpath,dtype=np.float32).reshape(128//upscale,128//upscale,128//upscale))
    X = np.stack(X,axis=0)
    Y = []
    for scalar in scalars:
        ypath = data_path+'HR/'+mode+'/'+scalar+hash_id+'.dat'
        Y.append(np.memmap(ypath,dtype=np.float32).reshape(128,128,128))
    Y = np.stack(Y,axis=0)
    dx = torch.tensor(np.float32(train_dict['dx [m]'][idx]))
    if train_dict['dy [m]'][idx] != '':
        dy = torch.tensor(np.float32(train_dict['dy [m]'][idx]))
    else:
        dy = dx
    if train_dict['dz [m]'][idx] != '':
        dz = torch.tensor(np.float32(train_dict['dz [m]'][idx]))
    else:
        dz = dx

    return torch.from_numpy(X),torch.from_numpy(Y),dx,dy,dz

def rot90_3D(input,k,dims):
    #input is C,H,W,D
    #C is rho,u,v,w
    #dims is 0,1,2 for x,y,z
    #preserves the divergence
    assert len(dims) == 2
    for i in range(k):
        #plus 1 because of channel
        swp = (dims[0]+1,dims[1]+1)
        input = torch.rot90(input, k=1, dims=swp)
        
        #plus 1 because of rho
        input[[swp[0],swp[1]]] = input[[swp[1],swp[0]]]
        
        input[swp[0]] = -input[swp[0]]
    return input

def rot_dx(dx_list,k,dims):
    assert len(dims) == 2
    for i in range(k):
        dx_list[[dims[0],dims[1]]] = dx_list[[dims[1],dims[0]]]
    return dx_list[0],dx_list[1],dx_list[2]

def flip_3D(input,dims):
    #input is C,H,W,D
    #dims is 0,1,2 for x,y,z
    #plus 1 because of channel
    #preserves the divergence
    assert len(dims) == 1
    swp = [dims[0]+1]
    input = torch.flip(input, dims=swp)
    input[swp[0]] = -input[swp[0]]
    return input

#data augmentation
def random_rot90_3D(X,Y,dx,dy,dz):
    dx_list = torch.tensor([dx,dy,dz]).reshape(3,1)
    k = np.random.randint(0,4)
    axis = np.random.choice([0,1,2],2,replace=False)
    X=rot90_3D(X,k,axis)
    Y=rot90_3D(Y,k,axis)
    dx,dy,dz = rot_dx(dx_list,k,axis)
    return X,Y,dx,dy,dz

#data augmentation
def random_flip_3D(X,Y):
    for axis in range(3):
        p = np.random.rand()
        if p > 0.5:
            X=flip_3D(X,[axis])
            Y=flip_3D(Y,[axis])
    return X,Y


def test_rot90_3D():
    X = torch.rand(4,8,8,8,dtype=torch.float32)
    Y = torch.rand(4,32,32,32,dtype=torch.float32)
    dx_list = np.array([1,2,3]).reshape(3,1)
    ru = X[0]*X[1]
    rv = X[0]*X[2]
    rw = X[0]*X[3]
    ru = ru.unsqueeze(0).unsqueeze(0)
    rv = rv.unsqueeze(0).unsqueeze(0)
    rw = rw.unsqueeze(0).unsqueeze(0)
    print(len(ru.shape),rv.shape,rw.shape)
    div = divergence(ru,rv,rw,dx=dx_list[0],dy=dx_list[1],dz=dx_list[2])
    div.type(torch.float32)

    X,Y,dx,dy,dz = random_rot90_3D(X,Y,dx_list[0],dx_list[1],dx_list[2])
    ru = X[0]*X[1]
    rv = X[0]*X[2]
    rw = X[0]*X[3]
    ru = ru.unsqueeze(0).unsqueeze(0)
    rv = rv.unsqueeze(0).unsqueeze(0)
    rw = rw.unsqueeze(0).unsqueeze(0)
    
    div_rot = divergence(ru,rv,rw,dx=dx,dy=dy,dz=dz)
    div_rot.type(torch.float32)
    print(div.sum(),div_rot.sum())
    torch.allclose(div.sum(),div_rot.sum(),atol=1e-4)

def test_flip_3D():
    X = torch.rand(4,8,8,8,dtype=torch.float32)
    Y = torch.rand(4,32,32,32,dtype=torch.float32)
    dx_list = np.array([1,2,3]).reshape(3,1)
    ru = X[0]*X[1]
    rv = X[0]*X[2]
    rw = X[0]*X[3]
    
    div = divergence(ru[None],rv[None],rw[None],dx=dx_list[0],dy=dx_list[1],dz=dx_list[2])
    div.type(torch.float32)
    X = random_flip_3D(X)
    ru = X[0]*X[1]
    rv = X[0]*X[2]
    rw = X[0]*X[3]
    div_flip = divergence(ru[None],rv[None],rw[None],dx=dx_list[0],dy=dx_list[1],dz=dx_list[2])
    print(div.sum(),div_flip.sum())
    torch.allclose(div.sum(),div_flip.sum(),atol=1e-4)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, my_dict,path, mode,upscale, transform,target_transform,dx_min):
        self.train_dict = my_dict
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.path = path
        self.upscale = upscale
        self.dx_min = dx_min
    def __len__(self):
        return len(self.train_dict['hash'])

    def __getitem__(self, idx):
        X,Y,dx,dy,dz = get_file(idx,self.train_dict,self.path,self.mode,self.upscale)
        dx = dx/self.dx_min
        dy = dy/self.dx_min
        dz = dz/self.dx_min

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            Y = self.target_transform(Y)
        
        #manually transfrom with rotations and flips
        if self.mode == 'train':
            X,Y,dx,dy,dz = random_rot90_3D(X,Y,dx,dy,dz)
            X,Y = random_flip_3D(X,Y)
        return X, Y,dx,dy,dz

class ScaleTransform(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,mean,std):
        self.mean = mean[:,None,None,None]
        self.std = std[:,None,None,None]
    def __call__(self, sample):
        return (sample- self.mean)/self.std
