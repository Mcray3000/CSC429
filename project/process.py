import polars as pl
import os

def process_ecu_ioht(file_path: str) -> pl.DataFrame:
    """
    Cleans and processes the ECU-IoHT dataset
    """
    
    df = pl.read_excel(file_path)
    
    df = df.with_columns(
        # Standardize target label (0 = Normal, 1 = Attack)
        pl.when(pl.col("Type") == "Attack").then(1).otherwise(0).cast(pl.UInt8).alias("label"),
        
        pl.col("Source").cast(pl.Categorical),
        pl.col("Destination").cast(pl.Categorical),
        pl.col("Protocol").cast(pl.Categorical),
        
        pl.col("Length").cast(pl.Int32),
        pl.col("Time").cast(pl.Float32)
    )
    
    # Drop unused, leaky, or replaced columns
    df = df.drop(["No.", "Info", "Type of attack", "Type"])
    
    return df


def process_wustl_ehms(file_path: str) -> pl.DataFrame:
    """
    Cleans and processes the WUSTL-EHMS-2020 dataset
    """

    df = pl.read_csv(file_path, try_parse_dates=False, infer_schema_length=None)
    
    df = df.with_columns(
        pl.col("Dir").str.strip_chars().cast(pl.Categorical),
        pl.col("Flgs").str.strip_chars().cast(pl.Categorical),
        pl.col("SrcAddr").cast(pl.Categorical),
        pl.col("DstAddr").cast(pl.Categorical),
        pl.col("SrcMac").cast(pl.Categorical),
        pl.col("DstMac").cast(pl.Categorical),
        
        # Make label name constant
        pl.col("Label").cast(pl.UInt8).alias("label")
    )
    
    # 3. Drop unused, leaky, or replaced columns
    df = df.drop(["Packet_num", "Attack Category", "Label"])
    
    return df

if __name__ == "__main__":
    # Get the absolute path to the directory containing process.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct exact paths to the data using the real filenames
    ecu_path = os.path.join(script_dir, "data", "ECU_IoHT.xlsx")
    wustl_path = os.path.join(script_dir, "data", "wustl-ehms-2020_with_attacks_categories.csv")
    
    # Process the datasets
    ecu_df = process_ecu_ioht(ecu_path)
    wustl_df = process_wustl_ehms(wustl_path)
    
    # Save the datasets as parquet
    ecu_df.write_parquet(os.path.join(script_dir, "data", "cleaned_ecu.parquet"))
    wustl_df.write_parquet(os.path.join(script_dir, "data", "cleaned_wustl.parquet"))
    
    print("Non-lossy preprocessing complete. Data saved to Parquet.")