import torch

def create_rot_mat(alpha):
    rot_mat = torch.tensor([[torch.cos(alpha), -torch.sin(alpha)], [torch.sin(alpha), torch.cos(alpha)]])
    return rot_mat

def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot, center=None):
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(torch.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2
    gauss = flux * torch.exp(
        -((x_0 - x) ** 2 / (2 * (x_fwhm) ** 2) + (y_0 - y) ** 2 / (2 * (y_fwhm) ** 2))
    )
    return gauss

def create_grid(pixel):
    x = torch.linspace(0, pixel - 1, steps=pixel)
    y = torch.linspace(0, pixel - 1, steps=pixel)
    X, Y = torch.meshgrid(x, y)
    X.unsqueeze_(0)
    Y.unsqueeze_(0)
    mesh = torch.cat((X,Y))
    grid = torch.tensor((torch.zeros(X.shape) + 1e-10))
    grid = torch.cat((grid, mesh))
    return grid

def gauss_valid(params): # setzt aus den einzelen parametern (54) ein bild zusammen
    gauss_param = torch.split(params, 9)
    grid = create_grid(63)
    source = torch.tensor((grid[0]))
    for i in range(len(gauss_param)):
        cent = torch.tensor([len(grid[0]) // 2 + gauss_param[1][i], len(grid[0]) // 2 + gauss_param[2][i]])
        s = gaussian_component(grid[1], grid[2], gauss_param[0][i], gauss_param[3][i], gauss_param[4][i], 
                   rot=gauss_param[5][i], center=cent)
        source = torch.add(source, s) 
    return source 

def vaild_gauss_bs(in_put):
    for i in range(in_put.shape[0]):
        if i == 0:
            source = gauss_valid(in_put[i]) # gauss parameter des ersten gausses
            source.unsqueeze_(0)
        else:
            h = gauss_valid(in_put[i])
            h.unsqueeze_(0)
            source = torch.cat((source, h))
    return source