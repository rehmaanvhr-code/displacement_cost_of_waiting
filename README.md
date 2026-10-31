# Displacement and the Cost of Waiting

A scenario model of who pays when AI adoption is slow. It takes forty occupations with employment, median wages and an AI exposure score, runs a fast adoption scenario with reskilling and a slow adoption scenario where the shock arrives late and lands harder, and compares the ten year income paths. The difference is the cost of waiting. The repo breaks that cost down by wage quintile, by displacement burden and by Canadian province.

## Connection to the research

This is the code companion to "Who Bears the Cost of Waiting: The Uneven Price of Slow AI Adoption" (Rehman, 2026), which extends "The Governance Lag: Why Slow AI Adoption Is the Larger Risk" (Rehman, 2026). Both papers build on the comparison in "The Age of AI and the Industrial Revolution: A Comparative Study of Technological Transformation and Inequality" (Rehman, 2026). The Industrial Revolution paper's comparison table contrasts the two transformations on speed, labour displaced, ownership of capital and geography. This model is the AI side of that table made quantitative. In the slow scenario the people with the resources to adopt tools on their own (high wage, high exposure occupations) capture gains early regardless of policy, which is the modern version of owners of the scarce factor capturing the surplus, while exposed workers with modest wages get wage stagnation now and a sharper displacement shock later.

## Why it matters for AI governance in financial services

Workforce impact assessment is turning up in responsible AI frameworks and in regulator expectations, and financial services has a concentration of highly exposed occupations (tellers, claims adjusters, loan officers, accounting clerks, analysts). The model gives a governance team a way to argue the cost side of delay, not just the risk side, and to show which groups of staff or customers need reskilling investment first. The same structure applies to any inclusion analysis where the question is who gains and who loses from a deployment decision.

## How it works

Each occupation has an AI exposure score in the spirit of Eloundou, Manning, Mishkin and Rock (2023), employment and a median wage. Each year the model computes an adoption level (zero before the start year, then a ramp), a productivity gain on exposed tasks of which a scenario specific share reaches wages, a displacement rate that rises with adoption and fades over a recovery period, and income for employed and displaced workers. The slow scenario adds two things: high wage exposed occupations pick up tools before formal adoption and capture most of the gain, and exposed workers see a small yearly real wage loss while the economy waits.

The cost of waiting per occupation is the fast income path minus the slow income path summed over eleven years. The repo reports it in dollars, as a percent of baseline income, and as extra displacement person years, grouped by employment weighted wage quintile. Province level costs apply each province's occupation group mix to the group level cost per worker. An employment weighted Gini of occupational income is computed each year under both scenarios.

## Run

```
pip install -r requirements.txt
python model.py
pytest
```

Outputs in `outputs/`: `cost_by_quintile.png` shows income lost as a percent of baseline and the share of extra displacement by quintile. `gini_paths.png` shows the inequality path under each scenario. `region_map_bar.png` ranks provinces by cost per worker. `scenario_summary.csv` holds the headline numbers and `occupation_costs.csv` and `quintile_costs.csv` the detail.

With the default parameters the cost falls hardest on the middle. Quintiles three and four lose about four to five percent of ten year income, the top quintile loses about three percent because its early adopters capture gains anyway, and the bottom quintile loses the least because its occupations (cashiers, food service, personal support) have low exposure. The extra displacement is spread across quintiles two to five, with the bottom quintile almost untouched. Inequality peaks higher under slow adoption but ends slightly lower, because fast adoption also raises wages at the top. Ontario has the highest cost per worker because of its weight in business and administrative occupations.

## Data and sources

`data/occupations.csv` holds forty occupations with approximate Canadian employment counts and median wages in the spirit of Statistics Canada Labour Force Survey and Job Bank figures, and exposure scores in the style of the GPT exposure scores in Eloundou et al. (2023), "GPTs are GPTs", and the task based measures in Felten, Raj and Seamans (2021). `data/regions.csv` holds approximate employment shares by occupation group for each province. All values are approximate and assembled by hand so the repo runs without downloads. Replace them with published series before using the numbers.

## Limitations

Forty occupations is a thin slice of the labour market. The exposure scores are judgement calls. Scenario parameters (gain, worker share, displacement, recovery, stagnation) drive the results and are set by assumption, not estimated. There is no labour mobility between occupations, no general equilibrium wage response and no new occupations created. Province shares are coarse. The model shows a mechanism and a distribution, not a forecast.

## Author

Saif Rehman, MMAI, Smith School of Business, Queen's University.
