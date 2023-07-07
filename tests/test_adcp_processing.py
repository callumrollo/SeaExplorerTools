import xarray as xr
import numpy as np
from pathlib import Path
import sys
module_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(module_dir))
from seaexplorertools import process_adcp


def test_processing():
    adcp_path = 'ADCP_refactoring_test_files/sea045_M44.ad2cp.00000*.nc'
    glider_pqt_path = 'ADCP_refactoring_test_files/Skag_test.pqt'
    options = {
        'debug': False,
        'correctADCPHeading': True,
        'ADCP_discardFirstBins': 0,
        'ADCP_correlationThreshold': 70,
        'ADCP_amplitudeThreshold': 75,
        'ADCP_velocityThreshold': 0.8,
        'correctXshear': False,
        'correctYshear': False,
        'correctZshear': False,
        'correctZZshear': False,
        'ADCP_regrid_correlation_threshold': 20,
    }
    out = process_adcp.process_mission(adcp_path, glider_pqt_path, options)

    profiles = np.arange(out["Pressure"].shape[1])
    depth_bins = np.arange(out["Pressure"].shape[0])

    ds_dict = {}
    for key, val in out.items():
        ds_dict[key] = (("depth_bin", "profile_num",), val)
    coords_dict = {"profile_num": (("profile_num"), profiles),
                   "depth_bin": (("depth_bin"), depth_bins)
                   }
    ds = xr.Dataset(data_vars=ds_dict, coords=coords_dict)
    ds_min = ds[['Sh_E', 'Sh_N', 'Sh_U', 'ADCP_E', 'ADCP_N']]
    ds_min_test = xr.open_dataset("tests/test_files/ds_out_min.nc")
    for var in list(ds_min):
        assert np.allclose(ds_min[var], ds_min_test[var], equal_nan=True, atol=1e-7, rtol=1e-3)

