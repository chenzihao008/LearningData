import torch
import time 
import cpp_interpolation
import cpp_cuda_interpolation

class trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx,feats,points):
        feat_interp = cpp_cuda_interpolation.trilinearinterpolation(feats,points)
        ctx.save_for_backward(feats,points)
        return feat_interp
    @staticmethod
    def backward(ctx,dl_dfeat_interp):
        feats,points = ctx.saved_tensors
        dl_dfeats = cpp_cuda_interpolation.trilinearinterpolation_bw(dl_dfeat_interp.contiguous()
                                                                     ,feats
                                                                     ,points)
        #跟forward输入要对齐，输入多少个就返回多少个，无梯度的补NONE
        return dl_dfeats, None



def trilinear_interpolation_py(feats,points):
    u = ((points[:,0:1])+1)/2
    v = ((points[:,1:2])+1)/2
    w = ((points[:,2:3])+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c
    feat_interp= (1-u)*( a*feats[:,0]+
                                b*feats[:,1]+
                                c*feats[:,2]+
                                d*feats[:,3])+\
                            u*( a*feats[:,4]+
                                b*feats[:,5]+
                                c*feats[:,6]+
                                d*feats[:,7])

    return feat_interp

if  __name__ == "__main__":
    # 测试文件
    # cpp测试
    # feats = torch.ones(2)
    # point = torch.zeros(2)
    # out = cpp_interpolation.trilinearinterpolation(feats,point)
    # print(out)

    N = 5
    F = 4

    # cuda测试
    tem = torch.rand(N,8,F,device='cuda:0')
    feats =  tem.clone().requires_grad_()
    feats2 = tem.clone().requires_grad_()
    point = torch.rand(N,3,device='cuda:0')*2-1
    cu_time_list = []
    py_time_list = []
    nums = 1
    for i in range (nums):
        t = time.time()
        # 注意这里需要使用apply
        out = trilinear_interpolation_cuda.apply(feats,point)
        torch.cuda.synchronize()
        e = time.time()
        # print('cu_time',e-t)
        cu_time_list.append(e-t)

        t = time.time()
        out_py = trilinear_interpolation_py(feats2,point)
        torch.cuda.synchronize()
        e = time.time()
        py_time_list.append(e-t)
        # print('py_time',e-t)
        # print(feats)
        # print(point)
        # print(out)
        # print(out_py)
        

    print('cu_fw_time',sum(cu_time_list)/nums)
    print('py_fw_time',sum(py_time_list)/nums)
    print('fw是否相同：',torch.allclose(out,out_py))




    s = time.time()
    loss2 = out.sum()
    loss2.backward()
    torch.cuda.synchronize()
    e = time.time()
    print('cu_bw_time',e-s)

    s = time.time()
    loss = out_py.sum()
    loss.backward()
    torch.cuda.synchronize()
    e = time.time()
    print('py_bw_time',e-s)

    print('bw是否相同：',torch.allclose(feats.grad,feats2.grad))