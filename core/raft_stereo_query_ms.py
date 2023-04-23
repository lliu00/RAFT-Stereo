import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8
from core.transformer import MultiScaleMaskedTransformerDecoder



try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        # self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            # self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
            self.fnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn='instance', downsample=args.n_downsample)

        self.transformerDecoder = MultiScaleMaskedTransformerDecoder(args = self.args,in_channels = 128, hidden_dim = 128, 
                                                                     num_queries = 64, nheads = 8,
                                                                     dim_feedforward = 2048, dec_layers = 9,
                                                                     pre_norm = False, mask_dim = 128,
                                                                     enforce_input_project = False)
        
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [x[0] for x in cnet_list]
            # inp_list = [torch.relu(x[1]) for x in cnet_list]
            # net_list[0].shape = [4, 128, 80, 180]    net_list[1].shape = [4, 128, 40, 90]
            
            
            

        if self.args.corr_implementation == "reg": # Default
            # corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        # elif self.args.corr_implementation == "alt": # More memory efficient than reg
        #     corr_block = PytorchAlternateCorrBlock1D
        #     fmap1, fmap2 = fmap1.float(), fmap2.float()
        # elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
        #     corr_block = CorrBlockFast1D
        # elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
        #     corr_block = AlternateCorrBlock
        # corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        # coords0, coords1 = self.initialize_flow(net_list[0])

        # if flow_init is not None:
        #     coords1 = coords1 + flow_init


        # coords1 = coords1.detach()
        # corr = corr_fn(coords1)  #�����Ӳ���Ҷ�Ӧ�ĵ�?ȡcorr
        # coor_list = list(corr)
        # flow = coords1 - coords0
        with autocast(enabled=self.args.mixed_precision):
            flow_predictions = self.transformerDecoder(fmap1, fmap2, net_list, test_mode)


        return flow_predictions
