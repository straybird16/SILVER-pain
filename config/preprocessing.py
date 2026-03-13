"""
Default and custom configurations of preprocessing raw data streams.
"""

from silver_pain.preprocessing import MergeConfig

default_merge_config_older_adults = MergeConfig(
    map_method="snap",  # "snap" or "interp"
    map_snap_kind = "one_to_one",   # "one_to_one" or "per_grid"
    map_interp_kind="linear",   # for EDA/TEMP if map_method="interp"
    # HR derivation / resampling
    #   Default to lowpass on HR after resampling to 64 Hz (Nyquist = 32 Hz)
    #       band pass filter frequency of inferred heart rate (HR)
    hr_bp_low_hz=0,
    hr_bp_high_hz=4,
    hr_bp_order = 2,
    # interpolation parameters of HR
    hr_target="1hz",
    hr_interp_kind="cubic", # "linear" / "quadratic" / "cubic"
    # robust global threshold and spike rejection on instantaneous HR
    hr_min_bpm = 30.0,
    hr_max_bpm = 220.0,
    hr_mad_z = 3.0, 

    # HR from HR.csv (Empatica E4)
    hr_map_method = "interp",          # "snap" or "interp"
    hr_map_interp_kind = "cubic",      # "linear" / "quadratic" / "cubic"
    
    # Pain/self-report time zone (pain_data.csv uses local ET)
    pain_tz = "America/New_York",
    pain_max_snap_s = 1,    # max snapping distance of self-report None means no max; otherwise ignore if farther than this

    # Section logic across segments
    gap_threshold_s = 3.0,     # default
    fill_short_gaps = True,
    
    # Which channels to interpolate across segment gaps (BVP usually should NOT)
    gap_fill_channels = ("eda", "temperature", "hr"),

    # Grid properties
    fs_bvp = 64.0,
    extend_grid_to_union = True,  # extend beyond BVP span to cover earliest/latest among channels in that segment
    
    # -------------- RR/IBI artifact correction pipeline --------------

    # Step A: robust local outlier marking on RR
    enable_rr_ratio_median = True,
    rr_ratio_win_beats = 11,
    rr_ratio_thr = 0.25,       # flag if |RR-med|/med > thr

    enable_rr_hampel = False,
    rr_hampel_win_beats = 11,
    rr_hampel_k = 3.0,          # Hampel/MAD z threshold

    # Step B: Kubios "Threshold" correction (optional)
    enable_kubios_threshold = False,
    kubios_threshold_level = "medium",      # very_low/low/medium/strong/very_strong
    kubios_threshold_sec_60bpm = None,
    kubios_threshold_med_win = 11,
    kubios_threshold_scale_by_mean_rr = True,
    kubios_threshold_interp_kind = "cubic", # used to replace flagged RR

    # Step C: Kubios "Automatic" (Lipponen & Tarvainen 2019)
    enable_kubios_auto = True,
    kubios_auto_edit_peaks = True,         # remove "extra" peaks, insert midpoint for "missed",
    kubios_auto_interp_kind = "cubic",      # replace ectopic/longshort-only RR

    # Windowing used by the paper/method:
    # - QD thresholds use 91 surrounding beats => halfwin=45
    # - medRR uses 11-beat median => win=11
    kubios_auto_qd_halfwin = 45,
    kubios_auto_medrr_win = 11,
    kubios_auto_alpha = 5.2,
    kubios_auto_c1 = 0.13,
    kubios_auto_c2 = 0.17,

    # Step D: HR resampling + optional Gaussian smoothing (default OFF)
    # If None, infer from hr_target: "1hz" -> 1.0, "64hz" -> fs_bvp
    hr_resample_hz = None,
    enable_hr_gaussian = False,
    hr_gaussian_sigma_s = 1.5,

    # Final cleanup action after all corrections (for remaining HR outliers)
    final_outlier_action = "drop",  # drop | interpolate | nan
)


default_merge_config_young_adults = MergeConfig(
    # Self-report parsing (local time) and snapping tolerance (seconds)
    
    map_method="snap",  # "snap" or "interp"
    map_snap_kind = "one_to_one",   # "one_to_one" or "per_grid"
    map_interp_kind="linear",   # for EDA/TEMP if map_method="interp"
    # HR derivation / resampling
    #   Default to lowpass on HR after resampling to 64 Hz (Nyquist = 32 Hz)
    #       band pass filter frequency of inferred heart rate (HR)
    hr_bp_low_hz=0,
    hr_bp_high_hz=4,
    hr_bp_order = 2,
    # interpolation parameters of HR
    hr_target="1hz",
    hr_interp_kind="cubic", # "linear" / "quadratic" / "cubic"
    # robust global threshold and spike rejection on instantaneous HR
    hr_min_bpm = 30.0,
    hr_max_bpm = 220.0,
    hr_mad_z = 3.0, 
    # HR from Empatica E4 HR.csv (already at 1 Hz)
    hr_map_method = "snap",          # "snap" or "interp"
    hr_map_interp_kind = "cubic",      # "linear" / "quadratic" / "cubic"
    
    # Pain/self-report time zone (pain_data.csv uses local ET)
    pain_tz = "America/New_York",
    pain_max_snap_s = 1,    # max snapping distance of self-report None means no max; otherwise ignore if farther than this

    # Section logic across segments
    gap_threshold_s = 3.0,     # default
    fill_short_gaps = True,
    
    # Which channels to interpolate across segment gaps (BVP usually should NOT)
    gap_fill_channels = ("eda", "temperature", "hr"),

    # Grid properties
    fs_bvp = 64.0,
    extend_grid_to_union = True,  # extend beyond BVP span to cover earliest/latest among channels in that segment
    
    # -------------- RR/IBI artifact correction pipeline --------------

    # Step A: robust local outlier marking on RR
    enable_rr_ratio_median = True,
    rr_ratio_win_beats = 11,
    rr_ratio_thr = 0.25,       # flag if |RR-med|/med > thr

    enable_rr_hampel = False,
    rr_hampel_win_beats = 11,
    rr_hampel_k = 3.0,          # Hampel/MAD z threshold

    # Step B: Kubios "Threshold" correction (optional)
    enable_kubios_threshold = False,
    kubios_threshold_level = "medium",      # very_low/low/medium/strong/very_strong
    kubios_threshold_sec_60bpm = None,
    kubios_threshold_med_win = 11,
    kubios_threshold_scale_by_mean_rr = True,
    kubios_threshold_interp_kind = "cubic", # used to replace flagged RR

    # Step C: Kubios "Automatic" (Lipponen & Tarvainen 2019)
    enable_kubios_auto = True,
    kubios_auto_edit_peaks = True,         # remove "extra" peaks, insert midpoint for "missed",
    kubios_auto_interp_kind = "cubic",      # replace ectopic/longshort-only RR

    # Windowing used by the paper/method:
    # - QD thresholds use 91 surrounding beats => halfwin=45
    # - medRR uses 11-beat median => win=11
    kubios_auto_qd_halfwin = 45,
    kubios_auto_medrr_win = 11,
    kubios_auto_alpha = 5.2,
    kubios_auto_c1 = 0.13,
    kubios_auto_c2 = 0.17,

    # Step D: HR resampling + optional Gaussian smoothing (default OFF)
    # If None, infer from hr_target: "1hz" -> 1.0, "64hz" -> fs_bvp
    hr_resample_hz = None,
    enable_hr_gaussian = False,
    hr_gaussian_sigma_s = 1.5,

    # Final cleanup action after all corrections (for remaining HR outliers)
    final_outlier_action = "drop",  # drop | interpolate | nan
)

#________________________________________#
"""
You may put your custom preprocessing config below
"""
custom_merge_config_older_adults = MergeConfig(

)
custom_merge_config_young_adults = MergeConfig(

)
