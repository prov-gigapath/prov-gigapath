# Preprocessing

Whole-slide images are giga-pixel images. To efficiently process these massive images, we use a tiling technique that splits them into smaller, manageable tile images. This repository includes sample code to help you perform this tiling process. Before running the code, ensure that all additional requirements are installed.

## Library Requirements

**OpenSlide library**
Download and install the OpenSlide library by following the instructions on the [OpenSlide Download Page](https://openslide.org/download/).

**Important:** Use the correct version of pixman. Pixman 0.38 has a known issue that can result in silently corrupted images. Verify your pixman version with:
```bash
ldd $(which ls) | grep pixman
```

References:
- https://github.com/openslide/openslide/issues/278
- https://github.com/openslide/openslide/issues/291
- https://histolab.readthedocs.io/en/latest/installation.html

You can try this snippet to install the pixman 0.40.0:

```bash
mkdir pixman
cd pixman
wget https://cairographics.org/releases/pixman-0.40.0.tar.gz
tar -xvf pixman-0.40.0.tar.gz
cd pixman-0.40.0
./configure
make
sudo make install
export LD_PRELOAD=/usr/local/lib/libpixman-1.so.0.40.0:$LD_PRELOAD
```

## Running the Demo

Once the requirements are installed, you can run the demo script to check if everything works:

`demo/2_tiling_demo.py`

Please check the produced tile images to ensure they are generated correctly.
