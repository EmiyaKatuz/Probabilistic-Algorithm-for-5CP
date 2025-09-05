thr=0.4
ECA
RF@0.5_TP,15
RF@0.5_FP,33
RF@0.5_TN,73
RF@0.5_FN,1

ET@dyn_TP,14
ET@dyn_FP,2
ET@dyn_TN,104
ET@dyn_FN,2

HYB_hourly_TP,16
HYB_hourly_FP,52
HYB_hourly_TN,54
HYB_hourly_FN,0

HYB_daily_TP,16
HYB_daily_FP,56
HYB_daily_TN,50
HYB_daily_FN,0

RTO
RF@0.5_TP,14
RF@0.5_FP,2
RF@0.5_TN,104
RF@0.5_FN,2

ET@dyn_TP,15
ET@dyn_FP,1
ET@dyn_TN,105
ET@dyn_FN,1

HYB_hourly_TP,16
HYB_hourly_FP,60
HYB_hourly_TN,46
HYB_hourly_FN,0

HYB_daily_TP,16
HYB_daily_FP,65
HYB_daily_TN,41
HYB_daily_FN,0

TESLA
RF@0.5_TP,14
RF@0.5_FP,2
RF@0.5_TN,104
RF@0.5_FN,2

ET@dyn_TP,15
ET@dyn_FP,1
ET@dyn_TN,105
ET@dyn_FN,1

HYB_hourly_TP,16
HYB_hourly_FP,24
HYB_hourly_TN,82
HYB_hourly_FN,0

HYB_daily_TP,16
HYB_daily_FP,31
HYB_daily_TN,75
HYB_daily_FN,0

+-----------------------------------------------------------------------------------+
|                               DATA & PREPROCESSING                                |
|  Six 1–6 day-ahead forecasts (ECA/RTO/TESLA) + actuals -> cleaned hourly series   |
+-----------------------------------------------------------------------------------+
               |                                                                
               |                                                                
               |                                                                
   +------------------------------+                       +-----------------------------------------------+
   | Day-level supervised model   |                       |             [paper algorithm]                  |
   | RF & ExtraTrees -> p_RF,p_ET |                       |  Hourly MC scenario engine (residual -> POT   |
   | Gate prior: p_day = max(...) |                       |  + probit -> GraphicalLasso -> simulate)      |
   +------------------------------+                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                       +-----------------------------------------------+
               |                                       | Running thresholds: monthly Top-4 (to date)   |
               |                                       | + Early-month fallback (same-month/all-hist) *|
               |                                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                       +-----------------------------------------------+
               |                                       | p_dayTop4  (daily exceed prob)                 |
               |                                       | p_hourly_MLH (MLH-weighted hourly exceed)  *  |
               |                                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                       +-----------------------------------------------+
               |                                       | Horizon fusion 1–6d via inverse Brier weights *|
               |                                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                       +-----------------------------------------------+
               |                                       | Cross-source fusion (ECA/RTO/TESLA)            |
               |                                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                       +-----------------------------------------------+
               |                                       | Soft-OR with day prior:                        |
               |                                       | p_final = 1 − (1−p_day)(1−p_mc)            *  |
               |                                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                       +-----------------------------------------------+
               |                                       | Decision threshold τ = 0.4 (tunable)        *  |
               |                                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                       +-----------------------------------------------+
               |                                       | Color grading (red/orange/yellow/green)     * |
               |                                       | + Monthly budget capping                    * |
               |                                       +-----------------------------------------------+
               |                                                      |
               |                                                      v
               |                                           +------------------+
               +------------------------------------------> |     result      |
                                                           +------------------+
