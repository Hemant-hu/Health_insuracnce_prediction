import pandas as pd
import os

# Check if submission.csv exists
submission_file = 'submission.csv'
if not os.path.exists(submission_file):
    print(f"❌ ERROR: {submission_file} not found!")
    raise FileNotFoundError(f"{submission_file} does not exist")

print(f"✅ File exists: {submission_file}")

# Get file size
file_size = os.path.getsize(submission_file)
print(f"📁 File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

# Load submission file
submission = pd.read_csv(submission_file)

print(f"\n📊 Submission DataFrame Info:")
print(f"   Shape: {submission.shape}")
print(f"   Columns: {list(submission.columns)}")
print(f"   Data types:\n{submission.dtypes}")

# Check required columns
required_cols = ['id', 'target']
missing_cols = [col for col in required_cols if col not in submission.columns]
if missing_cols:
    print(f"\n❌ ERROR: Missing required columns: {missing_cols}")
else:
    print(f"\n✅ Required columns present: {required_cols}")

# Check number of rows
expected_rows = 119043
actual_rows = len(submission)
print(f"\n📏 Row count:")
print(f"   Expected: {expected_rows:,}")
print(f"   Actual: {actual_rows:,}")
if actual_rows == expected_rows:
    print(f"   ✅ Row count matches!")
else:
    print(f"   ❌ Row count mismatch! Difference: {actual_rows - expected_rows:,}")

# Check missing values
missing_vals = submission.isnull().sum()
print(f"\n🔍 Missing values:")
print(missing_vals)
if missing_vals.sum() == 0:
    print("   ✅ No missing values!")
else:
    print(f"   ❌ Found {missing_vals.sum()} missing values")

# Check prediction range
if 'target' in submission.columns:
    target_min = submission['target'].min()
    target_max = submission['target'].max()
    target_mean = submission['target'].mean()
    target_std = submission['target'].std()
    
    print(f"\n📈 Prediction statistics:")
    print(f"   Min: {target_min:.6f}")
    print(f"   Max: {target_max:.6f}")
    print(f"   Mean: {target_mean:.6f}")
    print(f"   Std: {target_std:.6f}")
    
    # Check if predictions are in valid range [0, 1]
    if target_min >= 0 and target_max <= 1:
        print(f"   ✅ All predictions in valid range [0, 1]")
    else:
        print(f"   ❌ Predictions outside valid range [0, 1]!")
        out_of_range = ((submission['target'] < 0) | (submission['target'] > 1)).sum()
        print(f"   Number of out-of-range predictions: {out_of_range:,}")

# Display first rows
print(f"\n📄 First 10 rows:")
print(submission.head(10))

# Display last rows
print(f"\n📄 Last 10 rows:")
print(submission.tail(10))

# Summary verdict
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)
all_checks_passed = (
    os.path.exists(submission_file) and
    set(required_cols).issubset(submission.columns) and
    len(submission) == expected_rows and
    submission.isnull().sum().sum() == 0 and
    'target' in submission.columns and
    submission['target'].min() >= 0 and
    submission['target'].max() <= 1
)

if all_checks_passed:
    print("✅ ALL VALIDATION CHECKS PASSED!")
    print("   The submission file is ready for submission.")
else:
    print("❌ SOME VALIDATION CHECKS FAILED!")
    print("   Please review the issues above.")
print("="*60)
