"""Preprocessing module for NIH Chest X-rays dataset.

This module provides functionality to download, load, clean, and preprocess
the NIH Chest X-rays dataset from Kaggle. It handles data cleaning (duplicate
removal, outlier detection), patient filtering, and image loading and resizing.
"""

import os
import pandas as pd
import kagglehub
from PIL import Image


class Preprocessing():
    """Handle preprocessing of NIH Chest X-rays dataset.
    
    This class manages the download, loading, and preprocessing of chest X-ray
    images and associated metadata. It filters patients by condition labels,
    removes outliers and duplicates, and loads and resizes images.
    """
    def __init__(self, label):
        """Initialize Preprocessing instance and download dataset.
        
        Args:
            label (str): Finding label to filter for disease cases (e.g., 'Pneumonia').
        """
        self.label = label
        self.pneumonia = None  # Will store filtered pneumonia patient dataframe
        self.healthy = None  # Will store filtered healthy patient dataframe
        self.download_path = kagglehub.dataset_download("nih-chest-xrays/data")  # Download dataset from Kaggle
        self.metadata = None  # Will store the CSV metadata

    def _load_dataframe(self):
        """Load metadata CSV file from downloaded dataset.
        
        Reads the Data_Entry_2017.csv file containing patient metadata including
        finding labels and demographics. Removes the unnamed column and stores
        the result in self.metadata.
        """
        file_path_in_dataset = "Data_Entry_2017.csv"
        full_csv_path = os.path.join(self.download_path, file_path_in_dataset)
        
        # Load CSV with pandas
        df = pd.read_csv(full_csv_path)
        df.drop('Unnamed: 11', axis=1, inplace=True)  # Remove unnecessary column
        self.metadata = df.copy()
    
    def _outlier_removal(self, df):
        """Remove age outliers using Interquartile Range (IQR) method.
        
        Removes records where Patient Age falls outside 1.5 * IQR bounds
        (values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR).
        
        Args:
            df (pd.DataFrame): Input dataframe to remove outliers from.
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        # Calculate quartiles and IQR for age
        q1 = df['Patient Age'].quantile(0.25)
        q3 = df['Patient Age'].quantile(0.75)
        iqr = q3 - q1
        
        # Determine outlier bounds using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter records within acceptable age range
        initial_rows = df.shape[0]
        df = df[(df['Patient Age'] >= lower_bound) & (df['Patient Age'] <= upper_bound)]
        
        # Report results
        removed_rows = initial_rows - df.shape[0]
        print(f"Removed {removed_rows} outliers based on Patient Age.")
        print(f"New shape of dataframe: {df.shape}")
        return df

    def _remove_duplicates(self, df):
        """Remove duplicate records from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe to remove duplicates from.
            
        Returns:
            pd.DataFrame: Dataframe with duplicates removed.
        """
        duplicates = df.duplicated().sum()
        print(f"Number of duplicated rows: {duplicates}")
        
        # Remove duplicates if any exist
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"Duplicated rows removed. New dataframe length: {df.shape[0]}")
        return df
    

    def _normalize_age(self, df, method='minmax'):
        """Normalize patient age values using specified method.
        
        Applies age normalization to create features suitable for machine learning
        models. Supports Min-Max scaling (0-1 range) and standardization (z-score).
        
        Args:
            df (pd.DataFrame): Input dataframe containing 'Patient Age' column.
            method (str): Normalization method. Options are:
                         'minmax': Min-Max scaling to [0, 1] range (default).
                         'standard': Standardization (z-score normalization).
                         
        Returns:
            pd.DataFrame: Dataframe with new 'Patient Age Normalized' column.
        """
        df_normalized = df.copy()
        
        if method == 'minmax':
            # Min-Max scaling: (x - min) / (max - min)
            min_age = df['Patient Age'].min()
            max_age = df['Patient Age'].max()
            df_normalized['Patient Age Normalized'] = (df['Patient Age'] - min_age) / (max_age - min_age)
            print(f"Age normalized using Min-Max scaling. Range: [{min_age}, {max_age}] -> [0, 1]")
            
        elif method == 'standard':
            # Standardization (z-score): (x - mean) / std
            mean_age = df['Patient Age'].mean()
            std_age = df['Patient Age'].std()
            df_normalized['Patient Age Normalized'] = (df['Patient Age'] - mean_age) / std_age
            print(f"Age normalized using standardization. Mean: {mean_age:.2f}, Std: {std_age:.2f}")
            
        else:
            raise ValueError(f"Unknown normalization method: {method}. Choose 'minmax' or 'standard'.")
        
        return df_normalized

    def _encode_gender(self, df, encoding_map=None):
        """Encode categorical gender values to numeric format.
        
        Converts gender values ('M', 'F') to numeric values suitable for
        machine learning models. Default: M=1, F=0.
        
        Args:
            df (pd.DataFrame): Input dataframe containing 'Patient Gender' column.
            encoding_map (dict, optional): Custom mapping for gender values.
                                          Format: {'M': 1, 'F': 0}.
                                          If None, uses default mapping.
                                          
        Returns:
            pd.DataFrame: Dataframe with new 'Patient Gender Encoded' column.
        """
        df_encoded = df.copy()
        
        # Use default mapping if not provided
        if encoding_map is None:
            encoding_map = {'M': 1, 'F': 0}
        
        # Apply encoding
        df_encoded['Patient Gender Encoded'] = df['Patient Gender'].map(encoding_map)
        
        return df_encoded
    
    def _get_pneumonia_patients(self):
        """Filter, clean, and store pneumonia patient records.
        
        Filters metadata for records matching the specified finding label,
        removes duplicates, and removes age outliers.
        """
        # Filter patients with the specified condition
        pneumonia_only_df = self.metadata[self.metadata['Finding Labels'] == self.label]
        
        # Apply data cleaning steps
        pneumonia_only_df = self._remove_duplicates(pneumonia_only_df)
        pneumonia_only_df = self._outlier_removal(pneumonia_only_df)

        #Normalize Age
        pneumonia_only_df = self._normalize_age(pneumonia_only_df, method='minmax')
        #Encode Gender
        pneumonia_only_df = self._encode_gender(pneumonia_only_df, encoding_map={'M': 1, 'F': 0})
        
        # Store cleaned dataframe
        self.pneumonia = pneumonia_only_df.copy()

    def _get_healthy_patients(self):
        """Filter, clean, and store healthy patient records.
        
        Filters metadata for records with 'No Finding' label (healthy controls),
        removes duplicates, and removes age outliers.
        """
        # Filter patients with no medical findings (healthy controls)
        healthy_df = self.metadata[self.metadata['Finding Labels'] == 'No Finding']
        
        # Apply data cleaning steps
        healthy_df = self._remove_duplicates(healthy_df)
        healthy_df = self._outlier_removal(healthy_df)
        
        # Normalize Age
        healthy_df = self._normalize_age(healthy_df, method='minmax')
        # Encode Gender
        healthy_df = self._encode_gender(healthy_df, encoding_map={'M': 1, 'F': 0})

        # Store cleaned dataframe
        self.healthy = healthy_df.copy()

    def _load_image(self, image_filename, base_path, size=None):
        """Load and optionally resize a single chest X-ray image.
        
        Images are organized in subdirectories (images_001 through images_012).
        This method searches through all subdirectories to locate the image.
        
        Args:
            image_filename (str): Name of the image file to load.
            base_path (str): Base path to the images directory.
            size (tuple, optional): Target size (width, height) for resizing.
                                   If None, image is not resized. Defaults to None.
                                   
        Returns:
            PIL.Image.Image: Loaded and optionally resized image, or None if not found.
        """
        # Search through image subdirectories (images_001 to images_012)
        for i in range(1, 13):
            folder_name = f'images_{i:03d}'
            image_path = os.path.join(base_path, folder_name, 'images', image_filename)
            
            # Load image if found
            if os.path.exists(image_path):
                img = Image.open(image_path)
                
                # Resize if size parameter provided
                if size:
                    img = img.resize(size)
                return img
        
        # Return None if image not found in any directory
        return None

    def _load_all_images(self, size, df):
        """Load all images for records in a given dataframe.
        
        Args:
            size (tuple): Target size (width, height) for resizing images.
            df (pd.DataFrame): Dataframe containing 'Image Index' column with image filenames.
            
        Returns:
            list: List of loaded PIL Image objects.
        """
        all_images = []
        
        # Load each image from the dataframe
        for img_idx in df['Image Index']:
            img = self._load_image(img_idx, self.download_path, size=size)
            if img:  # Only add successfully loaded images
                all_images.append(img)
        
        print(f"Loaded {len(all_images)} images.")
        return all_images
    
    
    def load_images(self, size):
        """Main pipeline to load and preprocess all images and metadata.
        
        Orchestrates the complete preprocessing workflow:
        1. Loads metadata CSV
        2. Filters and cleans pneumonia patient data
        3. Filters and cleans healthy patient data
        4. Loads all images for both groups
        
        Args:
            size (tuple): Target size (width, height) for resizing all images.
            
        Returns:
            tuple: Two lists (all_pneumonia_images, all_healthy_images) containing
                   PIL Image objects.
        """
        # Load and filter metadata
        self._load_dataframe()
        self._get_pneumonia_patients()
        self._get_healthy_patients()
        
        # Load images for both patient groups
        all_pneumonia_images = self._load_all_images(size, self.pneumonia)
        all_healthy_images = self._load_all_images(size, self.healthy)
        
        return all_pneumonia_images, all_healthy_images

    def get_pneumonia_dataframe(self):
        """Get a copy of the processed pneumonia patient dataframe.
        
        Returns:
            pd.DataFrame: Cleaned pneumonia patient records.
        """
        return self.pneumonia.copy()
    
    def get_healthy_dataframe(self):
        """Get a copy of the processed healthy patient dataframe.
        
        Returns:
            pd.DataFrame: Cleaned healthy patient records.
        """
        return self.healthy.copy()


if __name__ == '__main__':
    """Example usage of the Preprocessing class."""
    
    # Initialize preprocessor for pneumonia classification
    preprocessing = Preprocessing('Pneumonia')
    
    # Run full preprocessing pipeline
    all_pneumonia_images, all_healthy_images = preprocessing.load_images(size=(128, 128))
    
    # Retrieve processed dataframes
    pneumonia_df = preprocessing.get_pneumonia_dataframe()
    healthy_df = preprocessing.get_healthy_dataframe()
    
    # Print summary statistics
    print('\n--- Summary Statistics ---')
    print('Number of pneumonia images: ', len(all_pneumonia_images))
    print('Number of healthy images: ', len(all_healthy_images))