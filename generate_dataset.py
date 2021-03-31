import os
import sys

from src_util.image_regions import images_to_dataset

# from src_util.images_labelled import defaultdict

sys.path.extend(['/home/petrmiculek/Code/light_matches',
                 '/home/petrmiculek/Code/light_matches/src',
                 '/home/petrmiculek/Code/light_matches/src_util'])

dims = [32, 64, 128]
scales = [25, 50]
background_sampling = [(100, 100), (500, 200), (500, 500)]
print('dim, scale, close, far -> out_folder')
for dim in dims:
    for scale in scales:
        for close, far in background_sampling:
            for val in [False, True]:
                out_folder = f'/data/datasets/{dim}x_{scale:03d}s_{close + far}bg' + ('_val' * val)
                if os.path.isdir(out_folder):
                    print('Skipping existing:', out_folder)
                    continue
                print(f'{dim}, {scale}, {close}, {far} -> {out_folder}')
                kwargs = {
                    'do_background': True,
                    'val': val,
                    'region_side': dim,
                    'output_folder': out_folder,
                    'scale_percentage': scale
                }
                images_to_dataset(do_foreground=True,
                                  per_image_samples=close,
                                  reduced_sampling_area=True,
                                  **kwargs
                                  )

                images_to_dataset(do_foreground=False,
                                  per_image_samples=far,
                                  reduced_sampling_area=False,
                                  **kwargs
                                  )
