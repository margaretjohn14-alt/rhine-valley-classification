import os
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import warnings
warnings.filterwarnings('ignore')

#configuration
TRAIN = False #set to True to train the model, False to visualize the results
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENE_ID = "LC08_L1TP_195026_20230601_20230607_02_T1"
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"] #Blue, Green, Red, NIR, SWIR1, SWIR2
#skipped B1 - Coastal Aerosol, B8 - panchromatic(different res), B9 - cirrus, B10, B11 - thermal
CORINE_PATH = "data/U2018_CLC2018_V2020_20u1.tif"

#read the metadata
#MTL.txt is a metadata with calibration coefficients. It comes with every LANDSAT scene. 
'''
Raw Landsat pixels is stored as digital numbers (DN). To convert it into actual reflectance values(0 to 1).
reflectance = (MULT X DN) + ADD 
MULT & ADD comes from the MTL.txt file.
'''
def read_md(scene_id, data_dir):
    md_path = os.path.join(data_dir, f"{scene_id}_MTL.txt")
    coeffs = {}
    sun_elevation = None

    with open(md_path) as f:
        for line in f:
            line = line.strip() #For each line, checks if it contains a coefficient needed
            for band_num in range(1, 10):
                if f"REFLECTANCE_MULT_BAND_{band_num}" in line:
                    coeffs[f"mult_{band_num}"] = float(line.split('=')[1].strip())
                if f"REFLECTANCE_ADD_BAND_{band_num}" in line:
                    coeffs[f"add_{band_num}"] = float(line.split('=')[1].strip())
            if "SUN_ELEVATION" in line and "=" in line:
                sun_elevation = float(line.split('=')[1].strip())

    print(f"Sun elevation angle: {sun_elevation:.2f} degrees")
    return coeffs, sun_elevation

#radiometric calibration
'''
converting digital numbers stored as raw landsat pixels to reflectances values(0 - 1). This is called radiometric calibration.
PART-1
Linear scaling
deg to rad
toa = (mult x dn) + add
using these coeff from the MLT.txt file, calculated the toa

PART-2
Solar Angle Correction
toa_corrected = toa/sin(sun_elevation)
'''
def dn_to_reflectance(dn, mult, add, sun_elevation_deg):
    sun_elevation_rad = np.deg2rad(sun_elevation_deg) #convert sun elevation from deg to rad
    toa = (mult * dn.astype(np.float32) + add) / np.sin(sun_elevation_rad)
    return toa

'''
There is a problem that needs to be solved. Its the Rayleigh scattering.
The water areas look hazier and more brighter(chance to get misclassified is higher) in the image so in order to get a more accurate reflectance values, 
we need to correct for this atmoshphere.
This is solved by Dark Object Subtraction where it subtracts the haze value from every pixel in the band.
This is the simplest atmospheric correction — more sophisticated methods
    (MODTRAN, 6S) require atmospheric profile data. DOS is standard for
    basic land cover analysis.
Selects only pixels with positive values, ignoring no-data zeros --> toa_band[toa_band > 0]
finds the 1st percentile value, value below which only 1% of pixels fall --> np.percentile
'''
def apply_dos_correction(toa_band):
    dark_object = np.percentile(toa_band[toa_band > 0], 1) #anything above 0 would be extra bright, which indicates atmospheric haze.
    corrected = toa_band - dark_object 
    corrected = np.clip(corrected, 0, 1) #only need reflectance values between 0 and 1.
    return corrected

'''
now I loop through each of the 6 bands(B2 - B7) applies the toa and dos correction to each of them and then stacks them all into a single 3D array.
'''
def load_and_preprocess_band(scene_id, data_dir, band_list, coeffs, sun_elevation):
    band_arrays = []
    profile = None

    #extracts the number from the band name string. 
    #"B2"[1:] gives "2", then int("2") gives 2. Needed to look up the correct coefficients in the dictionary — coeffs["mult_2"]

    for band_name in band_list:
        band_num = int(band_name[1:]) 
        band_path = os.path.join(data_dir, f"{scene_id}_{band_name}.TIF")

        print(f"Processing {band_name}...")

        '''
        open the GEOTIFF file at band_path. src is the file to read data from.
        src.read(1) --> reads band number 1 from the file. 
        So always say .read(1) to get the first (and only) band.

        '''
        with rasterio.open(band_path) as file:
            dn = file.read(1).astype(np.float32)
            dn[dn == 0] = np.nan
            if profile is None:
                profile = file.profile
        
        toa = dn_to_reflectance(dn, coeffs[f"mult_{band_num}"], coeffs[f"add_{band_num}"], sun_elevation)
        '''
        Imagine you're calculating the average height of people in a room, but some chairs are empty. 
        You can't average "empty" — so you temporarily mark empty chairs as 0, calculate, 
        then mark them as empty again so nobody thinks a 0cm person was in the room.
        '''
        toa_valid = np.where(np.isnan(toa), 0, toa)
        corrected = apply_dos_correction(toa_valid)
        corrected[np.isnan(toa)] = np.nan #np.percentile cannot handle NaN values — it would return NaN as the result, breaking the whole correction. So you temporarily swap NaN → 0 just for the DOS calculation.
        #put NaN back in those pixels — because they're still no-data pixels, you just needed to hide them temporarily during the DOS step.

        band_arrays.append(corrected)
        print(f"min = {np.nanmin(corrected):.4f}, max = {np.nanmax(corrected):.4f}")

    stacked = np.stack(band_arrays, axis=0) #stack of 6 arrays each with shape(H, W). so combined array of (6, H, W)

    print(f"\n Stacked array shape: {stacked.shape} ")
    return stacked, profile

'''
reads the QA_PIXEL.TIF File and then identifies cloud masks, cloud shadows and bad pixels and then sets them to Nan in all of the 6 bands.
each pixel in the image is an integer where the individual bits acts as flags, eg bit 3 is cloud flag, bit 4 is cloud shadow flag.            
'''
def apply_cloud_mask(stacked_bands, scene_id, data_dir):
    qa_path = os.path.join(data_dir, f"{scene_id}_QA_PIXEL.TIF")

    with rasterio.open(qa_path) as file:
        qa = file.read(1)
    
    cloud_mask = (qa & (1 << 3)) > 0 #bit 3 ....if its greater than 0 then cloud present.
    cloud_shadow_mask = (qa & (1 << 4)) > 0 #bit 4... if its greater than 0 then cloud shadow present
    bad_pixels = cloud_mask | cloud_shadow_mask #Combines both masks with OR — a pixel is bad if it's a cloud OR a cloud shadow.

    masked = stacked_bands.copy()#dont modify the original pixels, make a copy
    #now loop through the 6 bands and set each bad pixel as Nan
    for i in range(masked.shape[0]):
        masked[i][bad_pixels] = np.nan #select band i,  sets all cloud/shadow pixels in that band to NaN.
    
    #This tells you how much of the scene is obscured.
    cloud_pct = bad_pixels.sum() / bad_pixels.size * 100 #count of True values/total number of bad pixels 
    print(f"Cloud/Shadow pixels {cloud_pct:.1f}%")
    return masked

'''
Compute spectral indices. 
NDVI - normalized diff vegetation index 
NDWI - normalized diff water index
NDBI - normalized diff built-up index
NDBI = (SWIR1 - NIR)/ (SWIR1 + NIR)
Areas like concrete, asphalt and rooftops reflect SWIR very strongly and absorb NIR. whereas vegetation does the opposite.
NDBI is high for urban areas and neg for vegetation.
B2-Blue, B3-Green, B4-Blue, B5-NIR, B6-SWIR1, B7-SWIR2
'''
def compute_indices(stacked):
    blue = stacked[0]
    green = stacked[1]
    red = stacked[2]
    nir = stacked[3]
    swir1 = stacked[4]
    swir2 = stacked[5]

    ndvi = (nir - red) / (nir + red + 1e-10) #1e-10 instead of normalizedDifference()
    ndwi = (green - nir) / (green + nir + 1e-10)
    ndbi = (swir1 - nir)/(swir1 + nir + 1e-10)
    
    return ndvi, ndwi, ndbi

'''
CORINE uses numbers 1-44 to represent different land cover types across Europe. 
Each number corresponds to a specific land cover category defined in the CORINE nomenclature system.
There are 44 classes in total, ranging from artificial surfaces to agricultural areas, forests, wetlands, and water bodies.
Code 1 = Continuous urban fabric
Code 2 = Discontinuous urban fabric
Code 23 = Broad-leaved forest
Code 40 = Inland waters
Code 44 = Sea and ocean

I want only 5 classes for the classification: Water, Forest, Cropland, Urban, Bare soil. So I will map each of the 44 classes into these 5 classes.
Multiple corine can map to the same class as its more detailed than the simplified classes. For example, both 1 and 2 map to Urban.
'''
#selecting pixels where we're confident about the class based on their index values.
def define_training_samples(stacked, profile):
    '''
    Uses CORINE Land Cover 2018 as ground truth.
    
    CORINE is a validated European land cover dataset produced by the EU
    Copernicus Land Monitoring Service. It maps 44 land cover classes at
    100m resolution across Europe.
    
    Map CORINE's 44 classes into 5 simplified classes:
    0 = Water
    1 = Forest
    2 = Cropland
    3 = Urban
    4 = Bare soil
    
    Process:
    1. Reproject CORINE to match Landsat CRS and extent
    2. Resample from 100m to 30m to match Landsat pixel size
    3. Map CORINE class codes to our 5 simplified classes
    4. Use resulting map as training labels
    '''
    #CORINE class mapping
    #CORINE uses numeric codes 1-44 for different land cover types
    #Group them into 5 classes relevant for the classification
    _, H, W = stacked.shape
    corine_to_class = {
        # Water (class 0)
        40: 0, 41: 0, 42: 0, 43: 0, 44: 0,  # water bodies and wetlands
        35: 0, 36: 0, 37: 0, 38: 0, 39: 0,  # more water/wetland types

        # Forest (class 1)
        23: 1, 24: 1, 25: 1,  # broad-leaved, coniferous, mixed forest
        26: 1, 27: 1, 28: 1,  # scrub and transitional woodland

        # Cropland (class 2)
        12: 2, 13: 2, 14: 2,  # arable land, permanent crops
        15: 2, 16: 2, 17: 2, 18: 2,  # pastures, agricultural areas
        19: 2, 20: 2, 21: 2, 22: 2,

        # Urban (class 3)
        1: 3, 2: 3, 3: 3, 4: 3,   # urban fabric
        5: 3, 6: 3, 7: 3, 8: 3,   # industrial, transport
        9: 3, 10: 3, 11: 3,        # mines, construction, urban green

        # Bare soil (class 4)
        29: 4, 30: 4, 31: 4,  # beaches, bare rock, sparsely vegetated
        32: 4, 33: 4, 34: 4,  # burnt areas, glaciers
    }

    # ── LOAD AND REPROJECT CORINE ─────────────────────────────────────────────
    print("  Loading CORINE data...")
    with rasterio.open(CORINE_PATH) as corine_src:
        corine_crs       = corine_src.crs
        corine_transform = corine_src.transform
        corine_data      = corine_src.read(1)

    # Create empty array to hold reprojected CORINE data
    # matching exact dimensions and CRS of your Landsat scene
    corine_reprojected = np.zeros((H, W), dtype=np.uint8)

    # Reproject CORINE from its native CRS (EPSG:3035) to Landsat CRS (UTM)
    reproject(
        source=corine_data,
        destination=corine_reprojected,
        src_transform=corine_transform,
        src_crs=corine_crs,
        dst_transform=profile['transform'],
        dst_crs=profile['crs'],
        resampling=Resampling.nearest  # nearest neighbour — preserves class codes
    )

    print("  Reprojection complete")

    # ── MAP CORINE CODES TO 5 CLASSES ─────────────────────────────────────────
    labels = np.full((H, W), -1, dtype=np.int8)  # -1 = unlabeled

    for corine_code, our_class in corine_to_class.items():
        labels[corine_reprojected == corine_code] = our_class

    # ── PRINT STATISTICS ──────────────────────────────────────────────────────
    class_names = ["Water", "Forest", "Cropland", "Urban", "Bare soil"]
    total_labeled = (labels != -1).sum()
    print(f"  Total labeled pixels: {total_labeled:,}")

    for cls, name in enumerate(class_names):
        count = (labels == cls).sum()
        pct   = count / total_labeled * 100 if total_labeled > 0 else 0
        print(f"  Class {cls} ({name:10s}): {count:,} pixels ({pct:.1f}%)")

    unlabeled = (labels == -1).sum()
    print(f"  Unlabeled pixels: {unlabeled:,}")

    return labels

# train the classifier on a small subset of labeled pixels.
#train Random Forest
def train_classifier(stacked, ndvi, ndwi, ndbi, labels):
    
    #This forces the Random Forest to learn spectral patterns from raw reflectance values 
    # not the derived indices used to define labels.
    #stack all the features into one array
    features = np.stack([
        stacked[0], stacked[1], stacked[2],
        stacked[3], stacked[4], stacked[5],
    ], axis=-1)

    H, W, n_features = features.shape

    X = features.reshape(-1, n_features) #every row becomes one row with 9 features=columns
    y = labels.reshape(-1)#one label per pixel

    '''
    (y !=-1) - True for pixels that have a class label(not unlabeled).
    np.any(np.isnan(X), axis=1) - True for pixels where ANY of the 9 features is NaN(clouds, no-data)
    Both conditions must be true-pixels must be labeled and not have any NaN data.
    '''
    #Keep only labeled pixels
    valid_mask = (y !=-1) & ~np.any(np.isnan(X), axis=1) 
    X_labeled = X[valid_mask]
    y_labeled = y[valid_mask]

    print(f" Total Labeled Pixels: {len(y_labeled):,}")
    #add random subsampling 
    MAX_SAMPLES = 500000 #to avoid memory issues, limit the number of training samples to 500,000
    if len(y_labeled) > MAX_SAMPLES:
        idx = np.random.choice(len(y_labeled), MAX_SAMPLES, replace=False)
        X_labeled = X_labeled[idx]
        y_labeled = y_labeled[idx]
        print(f"  Randomly sampled {MAX_SAMPLES:,} pixels for training.")
    print(f"  Training samples: {len(y_labeled):,}")


    '''
    #To ensure each class is reped in both train and test sets. Without this, by chance there might get very few water pixels in the test set, making evaluation unreliable.
    for example, So if water is 3% of your labeled pixels, it will be exactly 3% in both train and test:
    Test set (30% of 1000 = 300 pixels):
    - 210 Forest    (70% of 300)
    - 60 Cropland   (20% of 300)
    - 15 Urban      (5% of 300)
    - 9 Water       (3% of 300)
    - 6 Bare soil   (2% of 300)
    '''
    #Training and Testing split - 70%-train, 30%-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled,
        test_size = 0.3,
        random_state = 42,
        stratify = y_labeled 
    ) 
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    #Train
    rf = RandomForestClassifier(
        n_estimators=100, # 100 decision trees 
        max_depth=15, #Atleast 15 questions from each tree
        min_samples_leaf=5, #each leaf should have atleast 5 pixels. Single pixels alone will give only noise from the decison trees.
        n_jobs=-1,#use all available CPU cores to train trees in parallel
        random_state=42
    )
    rf.fit(X_train, y_train) #to fit the classifier to x_train and y_train

    #Evaluate
    y_pred = rf.predict(X_test)
    oa     = accuracy_score(y_test, y_pred) * 100
    kappa  = cohen_kappa_score(y_test, y_pred)

    print(f"\n Overall Accuracy: {oa:.2f}%")
    print(f"Kappa Coefficient: {kappa:.4f}")

    #Define the classes
    class_names = ["Water", "Forest", "Cropland", "Urban Areas", "Bare Soil"]
    print(classification_report(y_test, y_pred, target_names=class_names))

    return rf, X, H, W, n_features, oa, kappa

# Predict full scene- Apply the above classifier to the every single pixel in the full scene(millions of pixels) - to get the final classification
def predict_scene(rf, X, H, W):
    classified = np.full(H * W, -1, dtype=np.int8)
    valid_mask = ~np.any(np.isnan(X), axis=1)

    chunk_size = 100000 #pixels per loop
    #indices of all True values in valid_mask — i.e. the positions of all valid pixels, [0]extracts the array from the tuple that np.where returns.
    valid_indices = np.where(valid_mask)[0] 

    for i in range(0, len(valid_indices), chunk_size): #loop with step size 100,000
        chunk_idx = valid_indices[i:i + chunk_size] #takes the next 100,000 pixel positions
        classified[chunk_idx] = rf.predict(X[chunk_idx])

    '''
    After the loop, classified is still a flat 1D array of H x W values. 
    Reshape it back to a 2D spatial grid so it can be saved as a GeoTIFF and visualized as a map
    '''
    classified = classified.reshape(H, W)
    return classified

#main pipeline
if TRAIN:
    print(f"Reading the metadata....")
    coeffs, sun_elevation = read_md(SCENE_ID, DATA_DIR) 

    print(f"\n Loading and preprocessing bands....")
    stacked, profile = load_and_preprocess_band(SCENE_ID, DATA_DIR, BANDS, coeffs, sun_elevation)

    print(f"\n Applying cloud masking...")
    stacked = apply_cloud_mask(stacked, SCENE_ID, DATA_DIR)

    print(f"\n Computing Spectral Indices...")
    ndvi, ndwi, ndbi = compute_indices(stacked)
    print(f" NDVI range: {np.nanmin(ndvi):.3f} to {np.nanmax(ndvi):.3f}")
    print(f" NDWI range: {np.nanmin(ndwi):.3f} to {np.nanmax(ndwi):.3f}")
    print(f" NDBI range: {np.nanmin(ndbi):.3f} to {np.nanmax(ndbi):.3f}")

    print(f"\n Defining the training samples...")
    labels = define_training_samples(stacked, profile)

    print(f"\n Training the Random Forest Classifier...")
    rf, X, H, W, n_features, oa, kappa = train_classifier(stacked, ndvi, ndwi, ndbi, labels)

    print(f"\n Classifying full scene...")
    classified = predict_scene(rf, X, H, W)

    # unpacks the dictionary as arguments to rasterio --> out_profile is a dict copied from the original band profile.
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.int8, count=1, compress='lzw')

    with rasterio.open(f"{OUTPUT_DIR}/classified.tif", 'w', **out_profile) as dst:
        dst.write(classified.astype(np.int8), 1) #converts the array to int8 to match what the profile specifies and then write to band 1 of the output file.

    with rasterio.open(f"{OUTPUT_DIR}/ndvi.tif", 'w', 
                       **{**profile, 'dtype': rasterio.float32, 'count': 1}) as dst:

        dst.write(ndvi.astype(np.float32), 1)

    np.save(f"{OUTPUT_DIR}/stacked.npy", stacked)
    np.save(f"{OUTPUT_DIR}/ndvi.npy", ndvi)
    np.save(f"{OUTPUT_DIR}/classified.npy", classified)
    np.save(f"{OUTPUT_DIR}/metrics.npy", np.array([oa, kappa]))

    print(f"Saved Outputs to {OUTPUT_DIR}/ ")
    print(f"Set TRAIN = False to visualize.")
else:
    print("Loading saved outputs...")
    classified = np.load(f"{OUTPUT_DIR}/classified.npy")
    ndvi       = np.load(f"{OUTPUT_DIR}/ndvi.npy")
    stacked    = np.load(f"{OUTPUT_DIR}/stacked.npy")
    metrics    = np.load(f"{OUTPUT_DIR}/metrics.npy")
    oa, kappa  = metrics[0], metrics[1]

    class_names  = ["Water", "Forest", "Cropland", "Urban", "Bare soil"]
    class_colors = ["#2166ac", "#2d6a4f", "#a8d08d", "#d9b365", "#c8a96e"]
    class_cmap   = mcolors.ListedColormap(class_colors)
    class_norm   = mcolors.BoundaryNorm(
        [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], class_cmap.N + 1
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        "Supervised Land Cover Classification — Rhine Valley, Germany\n"
        f"Landsat 8 | June 2023 | Random Forest | "
        f"OA: {oa:.1f}% | Kappa: {kappa:.3f}",
        fontsize=13, fontweight="bold"
    )

    # True colour composite
    rgb = np.stack([stacked[2], stacked[1], stacked[0]], axis=-1)
    rgb = np.clip(rgb * 3.5, 0, 1)
    rgb = np.where(np.isnan(rgb), 0, rgb)
    axes[0].imshow(rgb)
    axes[0].set_title("True Colour (B4-B3-B2)")
    axes[0].axis("off")

    # NDVI
    im1 = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[1].set_title("NDVI")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Classification
    im2 = axes[2].imshow(classified, cmap=class_cmap, norm=class_norm)
    axes[2].set_title("Land Cover Classification")
    axes[2].axis("off")
    cbar = plt.colorbar(im2, ax=axes[2], fraction=0.046, ticks=[0,1,2,3,4])
    cbar.set_ticklabels(class_names)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/classification_map.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/classification_map.png")

    print(f"\n Land Cover Statistics ")
    total = (classified >= 0).sum()
    for i, name in enumerate(class_names):
        count = (classified == i).sum()
        pct   = count / total * 100
        print(f"  {name:12s}: {pct:5.1f}%")



     


































    

