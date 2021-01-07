from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from data import CreateTrgDataSSLLoader
from model import CreateSSLModel
import os
from options.test_options import TestOptions
import scipy.io as sio
import gzip     # to compresss numpy files so that less cache space is required


                    #####This is the optimized pseudo label generation file which takes at max 12 gb CPU RAM and 3 GB GPU ram.#####

def main():
    opt = TestOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    args.restore_from = args.restore_opt1
    model1 = CreateSSLModel(args)
    model1.eval()
    model1.cuda()

    args.restore_from = args.restore_opt2
    model2 = CreateSSLModel(args)
    model2.eval()
    model2.cuda()

    args.restore_from = args.restore_opt3
    model3 = CreateSSLModel(args)
    model3.eval()
    model3.cuda()

    targetloader = CreateTrgDataSSLLoader(args)

    # change the mean for different dataset
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
    mean_img = torch.zeros(1, 1)

   
    image_name = []

    x = [None]*19     # x values for all 19 classes    
  

    cachepath = "../cache"    # Directory to save the numpy files as cache.

    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 1 == 0:
                print( '%d processd' % index )
            image, _, name = batch
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
            image = image.clone() - mean_img
            image = Variable(image).cuda()

            # forward
            output1 = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2 = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)

            output3 = model3(image)
            output3 = nn.functional.softmax(output3, dim=1)

            a, b = 0.3333, 0.3333
            output = a*output1 + b*output2 + (1.0-a-b)*output3

            output = nn.functional.interpolate(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1,2,0)
       
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            
           
            # Saving the prob and label files for each index seperately so that while loading the whole array need not be loaded in turn saving a lot of CPU ram.

            f1 = gzip.GzipFile(os.path.join(cachepath, "label_" + str(index))+'.npy.gz', "w")
            f2 = gzip.GzipFile(os.path.join(cachepath, "prob_" + str(index))+'.npy.gz', "w")
            np.save(f1,label)
            np.save(f2, prob)
            f1.close()
            f2.close()


            for i in range(19):
                d = prob[label==i]
                if(len(d)==0):
                  continue
                 
                if x[i] is None:
                    x[i]=d
                else:   
                    x[i]= np.concatenate((x[i],d))
                  

            image_name.append(name[0])
          

    thres = []

    

    thres = []
    for i in range(19):
        if x[i] is None:
            thres.append(0)
            continue
        temp=x[i]
        temp=np.sort(temp)
        print("temp[np.int(np.round(len(temp*0.66))]", temp[np.int(np.round(len(temp)*0.66))])
        thres.append(temp[np.int(np.round(len(temp)*0.66))].item())
       

    # print(thres)
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    print("Cuda", thres)

    
    for index in range(len(targetloader)):
        name = image_name[index]

        #Loading the prob and label files for each index.
        f3 = gzip.GzipFile(os.path.join(cachepath, "label_" + str(index))+'.npy.gz', "r")
        f4 = gzip.GzipFile(os.path.join(cachepath, "prob_" + str(index))+'.npy.gz', "r")
        label = np.load(f3)
        prob = np.load(f4)
        
        for i in range(19):
            label[   (prob<thres[i]) * (label==i)   ] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]

        #Deleting the prob and label files to clear the cache space.
        os.remove(os.path.join(cachepath,"label_"+str(index)+".npy.gz"))
        os.remove(os.path.join(cachepath,"prob_"+str(index)+".npy.gz"))
        output.save('%s/%s' % (args.save, name)) 
    
if __name__ == '__main__':
    main()
    
