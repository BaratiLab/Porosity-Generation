import torch
import torch.nn as nn

def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

#Define the generator
class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu, size = 64):
        
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if size == 64:
       # 64 x 64
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm3d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32

                nn.ConvTranspose3d(ngf, nc, 4, 2, 1, bias=False),
                #nn.Tanh()
                nn.Softmax(dim=1)
                # state size. (nc) x 64 x 64
            )
        elif size == 128:
# 128 x 128

            self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d(ngf, ngf, 4, 2, 1, bias=False),
            #nn.Tanh()
            #nn.Softmax(dim=1)
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),


            nn.ConvTranspose3d(ngf, nc, 4, 2, 1, bias=False),
            #nn.Tanh()
            nn.Softmax(dim=1)
            # state size. (nc) x 64 x 64
        )



        # 32 x 32
        elif size == 32:
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm3d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose3d(ngf * 2, nc, 4, 2, 1, bias=False),
                #nn.BatchNorm3d(ngf),
                #nn.ReLU(True),
                # state size. (ngf) x 32 x 32            
            # nn.ConvTranspose3d(ngf, nc, 4, 2, 1, bias=False),
                #nn.Tanh()
                nn.Softmax(dim=1)
                #nn.Sigmoid()
                # state size. (nc) x 64 x 64
            )

    def forward(self, input):
        # breakpoint()
        return self.main(input)
    
#Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, nz, nc, ndf, ngpu, size = 64):
        #breakpoint()
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        if size == 64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),


                # state size. (ndf*2) x 16 x 16
                nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),



                # state size. (ndf*4) x 8 x 8
                nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv3d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif size == 128:
        # 128 x 128

            self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv3d(ndf, ndf , 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True),


            # state size. (ndf*2) x 16 x 16
            # nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm3d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),


            
            # state size. (ndf*4) x 8 x 8
            nn.Conv3d(ndf , ndf, 8, 4, 2, bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv3d(ndf, 1, 4, 1, 0, bias=False),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv3d(1, 1, 5, 1, 0, bias=False),
            nn.Sigmoid()
            )


        # 32 x 32
        elif size == 32:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                # nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
                # nn.BatchNorm3d(ndf * 2),
                # nn.LeakyReLU(0.2, inplace=True),


                # state size. (ndf*2) x 16 x 16
                nn.Conv3d(ndf, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),



                # state size. (ndf*4) x 8 x 8
                nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv3d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
      #  breakpoint()
        return self.main(input)
