import re
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DatasetWithHeatmap:
    def __init__(self, gaze_heatmap=None):
        # train_imgs, train_lbl, train_fid, train_size, train_weight = None, None, None, None, None
        self.frameid2pos = None
        self.GHmap = gaze_heatmap # GHmap means gaze heap map
        self.NUM_ACTION = 18
        self.xSCALE, self.ySCALE = 8, 4 # was 6,3
        self.SCR_W, self.SCR_H = 160*self.xSCALE, 210*self.ySCALE
        self.train_size = 10
        self.HEATMAP_SHAPE = 14
        
    def createGazeHeatmap(self, gaze_coords, heatmap_shape):
        # converting per-frame gaze positions to heat map...
        self.frameid2pos = self.get_gaze_data(gaze_coords)
        self.train_size = len(self.frameid2pos.keys())
        self.HEATMAP_SHAPE = heatmap_shape
    
        
        self.GHmap = np.zeros([self.train_size, self.HEATMAP_SHAPE, self.HEATMAP_SHAPE, 1], dtype=np.float32)
        
        # print("Running BIU.convert_gaze_pos_to_heap_map() and convolution...")
        t1 = time.time()
        
        bad_count, tot_count = 0, 0
        for (i,fid) in enumerate(self.frameid2pos.keys()):
            tot_count += len(self.frameid2pos[fid])
            bad_count += self.convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.GHmap[i])
            
        # print("Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count))    
        # print("'Bad' means the gaze position is outside the 160*210 screen")
        
        sigmaH = 28.50 * self.HEATMAP_SHAPE / self.SCR_H
        sigmaW = 44.58 * self.HEATMAP_SHAPE / self.SCR_W
        self.GHmap = self.preprocess_gaze_heatmap(sigmaH, sigmaW, 0).astype(np.float32)

        for i in range(len(self.GHmap)):
            max_val = self.GHmap[i].max()
            min_val = self.GHmap[i].min()
            if max_val!=min_val:
                self.GHmap[i] = (self.GHmap[i] - min_val)/(max_val - min_val)

        if not np.count_nonzero(self.GHmap):
            print('The gaze map is all zeros')
            

        return self.GHmap
    
    def get_gaze_data(self, gaze_coords):
        frameid2pos = {}
        frame_id = 0
        for gaze_list in gaze_coords:
            frameid2pos[frame_id] = gaze_list
            frame_id += 1    

        # if len(frameid2pos) < 1000: # simple sanity check
        #     print ("Warning: did you provide the correct gaze data? Because the data for only %d frames is detected" % (len(frameid2pos)))

        few_cnt = 0
        for v in frameid2pos.values():
            if len(v) < 10: few_cnt += 1
        # print ("Warning:  %d frames have less than 10 gaze samples. (%.1f%%, total frame: %d)" % \
            # (few_cnt, 100.0*few_cnt/len(frameid2pos), len(frameid2pos)))     

        return frameid2pos

    
    # bg_prob_density seems to hurt accuracy. Better set it to 0
    def preprocess_gaze_heatmap(self, sigmaH, sigmaW, bg_prob_density, debug_plot_result=False):
        from scipy.stats import multivariate_normal
        import tensorflow as tf, keras as K # don't move this to the top, as people who import this file might not have keras or tf
#         
        model = K.models.Sequential()
        model.add(K.layers.Lambda(lambda x: x+bg_prob_density, input_shape=(self.GHmap.shape[1],self.GHmap.shape[2],1)))

        if sigmaH > 1 and sigmaW > 1: # was 0,0; don't blur if size too small
            lh, lw = int(4*sigmaH), int(4*sigmaW)
            x, y = np.mgrid[-lh:lh+1:1, -lw:lw+1:1] # so the kernel size is [lh*2+1,lw*2+1]
            pos = np.dstack((x, y))
            gkernel=multivariate_normal.pdf(pos,mean=[0,0],cov=[[sigmaH*sigmaH,0],[0,sigmaW*sigmaW]])
            assert gkernel.sum() > 0.95, "Simple sanity check: prob density should add up to nearly 1.0"

            model.add(K.layers.Lambda(lambda x: tf.pad(x,[(0,0),(lh,lh),(lw,lw),(0,0)],'REFLECT')))
            model.add(K.layers.Conv2D(1, kernel_size=gkernel.shape, strides=1, padding="valid", use_bias=False,
                activation="linear", kernel_initializer=K.initializers.Constant(gkernel)))
        else:
            print ("WARNING: Gaussian filter's sigma is 0, i.e. no blur.")
        # The following normalization hurts accuracy. I don't know why. But intuitively it should increase accuracy
        model.compile(optimizer='rmsprop', # not used
            loss='categorical_crossentropy', # not used
            metrics=None)
        
        output=model.predict(self.GHmap, batch_size=500)
        # print(np.count_nonzero(output))

        if debug_plot_result:
            print (r"""debug_plot_result is True. Entering IPython console. You can run:
                    %matplotlib
                    import matplotlib.pyplot as plt
                    f, axarr = plt.subplots(1,2)
                    axarr[0].imshow(gkernel)
                    rnd=np.random.randint(output.shape[0]); print "rand idx:", rnd
                    axarr[1].imshow(output[rnd,...,0])""")
            embed()
        
        shape_before, shape_after = self.GHmap.shape, output.shape
        assert shape_before == shape_after, """
        Simple sanity check: shape changed after preprocessing. 
        Your preprocessing code might be wrong. Check the shape of output tensor of your tensorflow code above"""
        return output
    
    def make_unique_frame_id(self, UTID, frameid):
        return (hash(UTID), int(frameid))
    
    def convert_gaze_pos_to_heap_map(self, gaze_pos_list, out):
        h,w = out.shape[0], out.shape[1]
        bad_count = 0
        if not np.isnan(gaze_pos_list).all():
            for j in range(0,len(gaze_pos_list),2):
                x = gaze_pos_list[j]
                y = gaze_pos_list[j+1]
                try:
                    out[int(y/self.SCR_H*h), int(x/self.SCR_W*w)] += 1
                except IndexError: # the computed X,Y position is not in the gaze heat map
                    bad_count += 1
        return bad_count
    
