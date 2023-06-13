import numpy as np # linear algebra
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import uniform_filter
import numpy as np

def torch_dx(phi,h):
    assert len(phi.shape) == 5
    batch_size = phi.shape[0]
    h = h.reshape(-1)
    my_list = []
    for i in range(batch_size):
        my_list.append(torch.gradient(phi[i:i+1], spacing=h[i],dim=2,edge_order=1)[0])
        assert len(my_list[-1].shape) == 5
    t = torch.cat(my_list,0)

    return t

def torch_dy(phi,h):
    assert len(phi.shape) == 5
    batch_size = phi.shape[0]
    h = h.reshape(-1)
    my_list = []
    for i in range(batch_size):
        my_list.append(torch.gradient(phi[i:i+1], spacing=h[i],dim=3,edge_order=1)[0])
        assert len(my_list[-1].shape) == 5
    t = torch.cat(my_list,0)

    return t

def torch_dz(phi,h):
    assert len(phi.shape) == 5
    batch_size = phi.shape[0]
    h = h.reshape(-1)
    my_list = []
    for i in range(batch_size):
        my_list.append(torch.gradient(phi[i:i+1], spacing=h[i],dim=4,edge_order=1)[0])
        assert len(my_list[-1].shape) == 5
    t = torch.cat(my_list,0)

    return t

    
def torch_diff(phi, dx,dy,dz): # x is a 3D tensor (batch, channel, x, y, z)
    diff_x = torch_dx(phi, dx)
    diff_y = torch_dy(phi, dy)
    diff_z = torch_dz(phi, dz)
    return diff_x, diff_y, diff_z

def remove_edges(arr):
    return arr[:,:,1:-1,1:-1,1:-1]

def torch_filter(phi,fw):
    #Kernel
    kernel = torch.ones((1,fw,fw,fw),device=phi.device,dtype=torch.float32)/torch.tensor(fw**3,dtype=torch.float32,device=phi.device)
    filtered = F.conv3d(phi, kernel.unsqueeze(0), stride=fw)
    return filtered
def favre_filter(rho,phi, fw):
    return torch_filter(rho*phi,fw)/torch_filter(rho,fw)

def sgs(ruvw,fw): #nbatch,ruvw,nx,ny,nz
    assert(ruvw.shape[1]==4)
    rho,u,v,w = ruvw[:,0:1],ruvw[:,1:2],ruvw[:,2:3],ruvw[:,3:4]
    # print(rho.shape,u.shape,v.shape,w.shape)
    rhof = torch_filter(rho,fw)
    uf = favre_filter(rho,u,fw)
    vf = favre_filter(rho,v,fw)
    wf = favre_filter(rho,w,fw)

    uuf = favre_filter(rho,u*u,fw)
    uvf = favre_filter(rho,u*v,fw)
    uwf = favre_filter(rho,u*w,fw)
    vvf = favre_filter(rho,v*v,fw)
    vwf = favre_filter(rho,v*w,fw)
    wwf = favre_filter(rho,w*w,fw)
    sgs = torch.cat([uuf-uf*uf,vvf-vf*vf,wwf-wf*wf,uvf-uf*vf,uwf-uf*wf,vwf-vf*wf],axis=1)
    return sgs*rhof #nbatch,6,nx,ny,nz

def divergence(ru,rv,rw, dx=1,dy=1,dz=1):  # inputs are 3D tensor (batch, channel, x, y, z)
    div = torch_dx(ru,dx) + torch_dy(rv,dy) + torch_dz(rw,dz)
    return div

def divergence_sgs_separate(stress,dx,dy,dz): #nbatch,ruvw,nx,ny,nz
    tau11 = stress[:,0:1]
    tau22 = stress[:,1:2]
    tau33 = stress[:,2:3]
    tau12 = stress[:,3:4]
    tau13 = stress[:,4:5]
    tau23 = stress[:,5:6]
    divtau1 = torch_dx(tau11,dx) + torch_dy(tau12,dy) + torch_dz(tau13,dz)
    divtau2 = torch_dx(tau12,dx) + torch_dy(tau22,dy) + torch_dz(tau23,dz)
    divtau3 = torch_dx(tau13,dx) + torch_dy(tau23,dy) + torch_dz(tau33,dz)
    return divtau1,divtau2,divtau3 #nx,ny,nz



def test_filter():
    # Define the input tensor
    phi = torch.randn(2,1,128,128,128)

    # Define the step size
    for fw in [2,4,8,16,32,64]:

        # Calculate the central difference using PyTorch
        filtered = torch_filter(phi,fw)
        print(filtered.shape)

        # Convert the input tensor to a NumPy array
        phi_np = phi.numpy()

        # Calculate the central difference using NumPy's diff function
        filtered_np = uniform_filter(phi_np, size=[1,1,fw,fw,fw],origin=0)
        filtered_np = filtered_np[:,:,fw//2::fw,fw//2::fw,fw//2::fw]
        print(filtered_np.shape)
        # Assert that the results are equal
        np.testing.assert_allclose(filtered.numpy(), filtered_np,atol=1e-5)


def test_diff():
    # Define the input tensor
    phi = torch.randn(2,1,128, 128,128)

    # Define the step size
    h = torch.tensor([0.5,0.4])

    # Calculate the central difference using PyTorch
    diff_x, diff_y, diff_z = torch_diff(phi, h,h,h)
    print(diff_x[0].sum())
    # Convert the input tensor to a NumPy array
    phi_np = phi.numpy()

    # Calculate the central difference using NumPy's diff function
    batch_size = phi_np.shape[0]

    for i in range(batch_size):
        print(i)
        diff_x_np = np.gradient(phi_np[i:i+1],h[i], axis=2)
        diff_y_np = np.gradient(phi_np[i:i+1],h[i], axis=3)
        diff_z_np = np.gradient(phi_np[i:i+1],h[i], axis=4)

        print(diff_x_np[0].sum())
        np.testing.assert_allclose(diff_x.numpy()[i:i+1], diff_x_np, rtol=1e-5)
        np.testing.assert_allclose(diff_y.numpy()[i:i+1], diff_y_np, rtol=1e-5)
        np.testing.assert_allclose(diff_z.numpy()[i:i+1], diff_z_np, rtol=1e-5)

    print(diff_z_np.shape)
    # Assert that the results are equal


if __name__ == '__main__':
    test_diff()
    test_filter()

