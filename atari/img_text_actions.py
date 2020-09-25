import cv2
import argparse
import os

def confound(data_dir, dest_dir):
    # loop through all trials
    trials = [t for t in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,t))]

    for t in trials:

        if not os.path.exists(os.path.join(dest_dir,t)):
            os.mkdir(os.path.join(dest_dir,t))

        # read txt file for action labels
        # action - index 5 in every row
        txt_file = os.path.join(data_dir,t+".txt")
        f = open(txt_file,"r")
        rows = f.readlines()
        

        # imgs in trial
        img_paths = [p for p in os.listdir(os.path.join(data_dir,t)) if p.endswith('.png')]
        img_prefix = '_'.join(img_paths[0].split('_')[:-1])

        action = '0'

        for i in range(len(img_paths)):
            img_path = img_prefix+'_'+str(i+1)+'.png'
            img = cv2.imread(os.path.join(data_dir,t,img_path))
            height, width = img.shape[0], img.shape[1]

            row = f.read()

            if i%16==0 and i>16:
                
                row = rows[i+1].split(',')
                curr_action = row[5]
                action = curr_action
                

            # overlay action text in white
            cv2.putText(img, action, (int(1/5*width),int(9/10*height)), cv2.FONT_HERSHEY_SIMPLEX, \
                fontScale=1, color=(255, 255, 255), thickness=3)

            # save modified data
            dest_img_name = os.path.join(dest_dir,t,img_path)
            cv2.imwrite(dest_img_name, img)
        


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data_dir', default='./test_data/asterix/', help="test images")
    parser.add_argument('--dest_dir', default='./test_data/asterix_confounded/', help="test images")

    args = parser.parse_args()

    if not os.path.exists(args.dest_dir):
        os.mkdir(args.dest_dir)

    # create data with overlaid actions
    confound(args.data_dir, args.dest_dir)