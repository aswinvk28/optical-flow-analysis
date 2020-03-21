import argparse
from complex_analysis import *
from copy import copy
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="", type=str)
    parser.add_argument("--simulate_data", default=False, type=bool)
    parser.add_argument("--simulate_config", default=False, type=bool)
    parser.add_argument("--camera_constant", default=False, type=float)
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    view = json.load(open("view.json", "r"))
    view_copy = copy(view)

    filename = args.image.replace("analysis/", "").replace(".jpg", "").replace(".jpeg", "")

    args.image = cv2.imread(args.image)

    camera_constant = args.camera_constant

    if args.simulate_config:
        heatmap = np.ones((240,320))
        for i in range(100):
            for s in range(2):
                if s == 0:
                    view['shutter_open']['camera']['x'] = view_copy['shutter_open']['camera']['x'] + \
                        camera_constant * i
                    view['shutter_open']['camera']['y'] = view_copy['shutter_open']['camera']['y'] - \
                        camera_constant * i
                    view['shutter_close']['camera']['x'] = view_copy['shutter_close']['camera']['x'] + \
                        camera_constant * i
                    view['shutter_close']['camera']['y'] = view_copy['shutter_close']['camera']['y'] - \
                        camera_constant * i
                    amplitude, real_part, phase, temporal, optical_flow_derivatives, rgb = \
                        compute_series_vectors(heatmap, args.image, view)

                    imageio.imwrite("images/config/plus/"+filename+"/amplitude-"+i.__str__()+".jpg", amplitude)
                    imageio.imwrite("images/config/plus/"+filename+"/phase-"+i.__str__()+".jpg", phase)
                    imageio.imwrite("images/config/plus/"+filename+"/real_part-"+i.__str__()+".jpg", real_part)
                    imageio.imwrite("images/config/plus/"+filename+"/rgb-"+i.__str__()+".jpg", rgb)
                    imageio.imwrite("images/config/plus/"+filename+"/temporal-"+i.__str__()+".jpg", temporal)
                elif s == 1:
                    view['shutter_open']['camera']['x'] = view_copy['shutter_open']['camera']['x'] - \
                        camera_constant * i
                    view['shutter_open']['camera']['y'] = view_copy['shutter_open']['camera']['y'] + \
                        camera_constant * i
                    view['shutter_close']['camera']['x'] = view_copy['shutter_close']['camera']['x'] - \
                        camera_constant * i
                    view['shutter_close']['camera']['y'] = view_copy['shutter_close']['camera']['y'] + \
                        camera_constant * i
                    amplitude, real_part, phase, temporal, optical_flow_derivatives, rgb = \
                        compute_series_vectors(heatmap, args.image, view)
                    imageio.imwrite("images/config/minus/"+filename+"/amplitude-"+i.__str__()+".jpg", amplitude)
                    imageio.imwrite("images/config/minus/"+filename+"/phase-"+i.__str__()+".jpg", phase)
                    imageio.imwrite("images/config/minus/"+filename+"/real_part-"+i.__str__()+".jpg", real_part)
                    imageio.imwrite("images/config/minus/"+filename+"/rgb-"+i.__str__()+".jpg", rgb)
                    imageio.imwrite("images/config/minus/"+filename+"/temporal-"+i.__str__()+".jpg", temporal)

    elif args.simulate_data:

        for i in range(10):
            if i < 1:
                heatmap = np.ones((240,320))
            else:
                x = np.random.beta(0.5, 0.5, 4096)
                y = np.random.beta(0.5, 0.5, 4096)

                # Create heatmap
                heatmap, xedges, yedges = np.histogram2d(x, y, bins=(240,320))

                # sigmoid
                heatmap = 1 / np.exp(heatmap)

            amplitude, real_part, phase, temporal, optical_flow_derivatives, rgb = \
                compute_series_vectors(heatmap, args.image, view)

            imageio.imwrite("images/data/"+filename+"/amplitude-"+i.__str__()+".jpg", amplitude)
            imageio.imwrite("images/data/"+filename+"/phase-"+i.__str__()+".jpg", phase)
            imageio.imwrite("images/data/"+filename+"/real_part-"+i.__str__()+".jpg", real_part)
            imageio.imwrite("images/data/"+filename+"/rgb-"+i.__str__()+".jpg", rgb)
            imageio.imwrite("images/data/"+filename+"/temporal-"+i.__str__()+".jpg", temporal)
