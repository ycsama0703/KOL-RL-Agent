# Case Study Contrast Sequence

- Source: `youtube`
- KOL: `The_Maverick_of_Wall_Street`
- Focus window: `2024-10-25 -> 2024-12-20`

## Focus Days

```text
 source                         kol  focus_day mapped_event_day      gap    d_gap  day_total_contribution window_start window_end
youtube The_Maverick_of_Wall_Street 2024-11-19       2024-11-11 0.122625 0.042437                0.010560   2024-10-25 2024-12-20
youtube The_Maverick_of_Wall_Street 2024-11-06       2024-11-04 0.051120 0.020112                0.015506   2024-10-25 2024-12-20
```

## Sequence (Discourse -> Sentiment -> Baseline -> Ours)

```text
 focus_day ticker  sentiment     baseline_behavior             ours_behavior                    relation  contribution                                                                                                                                                                                                                 text_preview
2024-11-19   SMCI       -0.9 Baseline long (0.092)  INCREASE -> long (0.118)      Overweight vs baseline      0.013065 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-11-19    AMD       -0.9 Baseline long (0.037) HOLD -> near flat (0.000) De-risk to flat vs baseline      0.002794 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-11-19   INTC       -0.9 Baseline long (0.081)  INCREASE -> long (0.047)     Underweight vs baseline      0.001943 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-11-19  GOOGL       -0.9 Baseline long (0.090)      OPEN -> long (0.068)     Underweight vs baseline      0.001338 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-11-19   AAPL       -0.5 Baseline long (0.074)  INCREASE -> long (0.093)              Track baseline      0.000895 ...you look at apple it's flirting with the 50 in blue it's above the 100 in red but it's not looking good right now all of these names went higher after the elections apple stored and the earnings were [&nbsp;__&nbsp;] 
2024-11-06   TSLA        0.2 Baseline long (0.033)  INCREASE -> long (0.081)      Overweight vs baseline      0.021219 the most popular names nvidia tesla but also for amazon after earnings and apple 2 and of course super micro uh because of the disastrous news that we got over the week. then we got another bullish one for you for tesla 
2024-11-06   ABBV        0.2 Baseline long (0.072)      OPEN -> long (0.021)     Underweight vs baseline      0.006535 the most popular names nvidia tesla but also for amazon after earnings and apple 2 and of course super micro uh because of the disastrous news that we got over the week. then we got another bullish one for you for tesla 
2024-11-06   SMCI        0.2 Baseline long (0.072)      OPEN -> long (0.024)     Underweight vs baseline      0.005156 the most popular names nvidia tesla but also for amazon after earnings and apple 2 and of course super micro uh because of the disastrous news that we got over the week. then we got another bullish one for you for tesla 
2024-11-06   WYNN        0.2 Baseline long (0.057) OPEN -> near flat (0.012) De-risk to flat vs baseline      0.004283 the most popular names nvidia tesla but also for amazon after earnings and apple 2 and of course super micro uh because of the disastrous news that we got over the week. then we got another bullish one for you for tesla 
2024-11-06   CTVA        0.2 Baseline long (0.046) OPEN -> near flat (0.017) De-risk to flat vs baseline      0.001260 the most popular names nvidia tesla but also for amazon after earnings and apple 2 and of course super micro uh because of the disastrous news that we got over the week. then we got another bullish one for you for tesla 
2024-12-02    GME       -0.9 Baseline long (0.040) OPEN -> near flat (0.010) De-risk to flat vs baseline      0.002888 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-12-02   TMUS       -0.9 Baseline long (0.032) HOLD -> near flat (0.000) De-risk to flat vs baseline      0.002275 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-12-02   INTC       -0.9 Baseline long (0.050)      HOLD -> long (0.033)              Track baseline      0.002008 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-12-02      F       -0.9 Baseline long (0.053) OPEN -> near flat (0.016) De-risk to flat vs baseline      0.001683 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
2024-12-02    AMD       -0.9 Baseline long (0.050)      OPEN -> long (0.027)     Underweight vs baseline      0.000924 ...then we talked about super micro the trashiest company of them all and basically here this is the chart that i gave you if it closes below 2482 you probably want to take a short on it as a day trade if it closes above
```