"""Shared execution of the three estimator stages."""
from time import perf_counter

from libs.shared.results import OptimizationResult


def run_estimator_stages(case, dispatch_estimator, physical_estimator,
                         economic_estimator, **physical_options):
    """Execute each stage once and measure its complete wall time."""
    started = perf_counter()
    stage = perf_counter()
    dispatch = dispatch_estimator.estimate(case)
    dispatch.solve_time = perf_counter() - stage
    stage = perf_counter()
    physical = physical_estimator.estimate(case, dispatch=dispatch, **physical_options)
    physical.solve_time = perf_counter() - stage
    stage = perf_counter()
    economic = economic_estimator.estimate(case, physical, dispatch=dispatch)
    economic.solve_time = perf_counter() - stage
    diagnostics = dict(physical.diagnostics)
    diagnostics.update(
        dispatch_method=dispatch.method_name, physical_method=physical.method_name,
        economic_method=economic.method_name, solver_stage="estimator_pipeline",
        economic_diagnostics=dict(economic.diagnostics),
    )
    return OptimizationResult(
        physical_state=physical, dispatch=dispatch, economic_state=economic,
        diagnostics=diagnostics, method_name="EstimatorPipeline",
        solve_time=perf_counter() - started,
    )
