"""
Visualization tools for PowerSOC results.
"""

try:
	from .network import (
		build_plot_inputs,
		verify_load_balance,
		create_combined_plot,
		create_deviation_plot,
		create_white_network_plot,
		create_merit_order_plot,
		export_network_plot,
		export_side_by_side_dashboard,
		get_visual_theme,
		get_line_color_mode,
		set_visual_theme,
		set_line_color_mode,
	)
except ModuleNotFoundError as exc:
	_missing_network_error = exc

	def _missing_network_dependency(*_args, **_kwargs):
		raise ModuleNotFoundError(
			"Network visualization helpers require pandapower and are unavailable in this environment."
		) from _missing_network_error

	build_plot_inputs = _missing_network_dependency
	verify_load_balance = _missing_network_dependency
	create_combined_plot = _missing_network_dependency
	create_deviation_plot = _missing_network_dependency
	create_white_network_plot = _missing_network_dependency
	create_merit_order_plot = _missing_network_dependency
	export_network_plot = _missing_network_dependency
	export_side_by_side_dashboard = _missing_network_dependency
	get_visual_theme = _missing_network_dependency
	get_line_color_mode = _missing_network_dependency
	set_visual_theme = _missing_network_dependency
	set_line_color_mode = _missing_network_dependency

from .database import (
	create_branch_timeseries_plot,
	create_branch_boxplot,
	create_bus_timeseries_plot,
	create_bus_boxplot,
	create_duration_curve_plot,
	create_price_duration_curve_plot,
	compute_duration_curve,
	create_database_dashboard,
	create_generator_timeseries_plot,
	create_solve_time_timeseries_plot,
)

__all__ = [
	"build_plot_inputs",
	"verify_load_balance",
	"create_combined_plot",
	"create_deviation_plot",
	"create_branch_timeseries_plot",
	"create_branch_boxplot",
	"create_bus_timeseries_plot",
	"create_bus_boxplot",
	"create_duration_curve_plot",
	"create_price_duration_curve_plot",
	"compute_duration_curve",
	"create_database_dashboard",
	"create_generator_timeseries_plot",
	"create_solve_time_timeseries_plot",
	"create_white_network_plot",
	"create_merit_order_plot",
	"export_network_plot",
	"export_side_by_side_dashboard",
	"get_visual_theme",
	"get_line_color_mode",
	"set_visual_theme",
	"set_line_color_mode",
]
