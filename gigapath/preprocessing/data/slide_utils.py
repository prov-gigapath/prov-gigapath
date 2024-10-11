import openslide

def find_level_for_target_mpp(slide_path, target_mpp):
    """
    Find the level in the slide that corresponds to the target MPP.

    Parameters:
    slide_path (str): Path to the slide file.
    target_mpp (float): Target microns per pixel (MPP).

    Returns:
    int: Level number that corresponds to the target MPP or None if not found.
    """
    slide = openslide.OpenSlide(slide_path)
    print(slide.properties)
    resolution_unit = slide.properties.get('tiff.ResolutionUnit')
    if(slide_path.endswith(".svs")):
        """tcga WSI's format is .svs and the level 0 mpp can be found in properties"""
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        print(f"LEVEL 0 MPP X: {mpp_x} microns per pixel")
        print(f"LEVEL 0 MPP Y: {mpp_y} microns per pixel")
        for downsample in slide.level_downsamples:
            level_mpp_x = float(mpp_x) * downsample
            level_mpp_y = float(mpp_y) * downsample
            if abs(level_mpp_x - target_mpp) < 0.1 and abs(level_mpp_y - target_mpp) < 0.1:
                print(f"Level {slide.level_downsamples.index(downsample)} corresponds to approximately {target_mpp} MPP.")
                print("xmpp:",level_mpp_x,"ympp:",level_mpp_y)
                return slide.level_downsamples.index(downsample)
    else:
        # Retrieve resolution information from properties
        x_resolution = float(slide.properties.get('tiff.XResolution'))
        y_resolution = float(slide.properties.get('tiff.YResolution'))
        # Convert resolution to microns per pixel (MPP)
        if resolution_unit == 'centimeter':
            mpp_x = 10000 / x_resolution
            mpp_y = 10000 / y_resolution
        else:
            print("Resolution unit is not in centimeters. Adjust the calculation accordingly.")
            return None

        # Check if MPP information is available
        if not mpp_x or not mpp_y:
            print("Could not calculate MPP due to missing or invalid resolution information.")
            return None

        # Iterate through each level and calculate MPP
        for level in range(slide.level_count):
            # Calculate MPP for the current level
            level_mpp_x = mpp_x * slide.level_downsamples[level]
            level_mpp_y = mpp_y * slide.level_downsamples[level]

            # Check if this level's MPP is close to the target MPP
            if abs(level_mpp_x - target_mpp) < 0.1 and abs(level_mpp_y - target_mpp) < 0.1:
                print(f"Level {level} corresponds to approximately {target_mpp} MPP.")
                print("xmpp:",level_mpp_x,"ympp:",level_mpp_y)
                return level

        print(f"No level corresponds to approximately {target_mpp} MPP.")
    return None
