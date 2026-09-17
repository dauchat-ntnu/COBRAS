"""
Example: Analyze load profiles from MATPOWER-style CSV inputs.

Generates a box plot for active and reactive demand at each bus based on
p_load.csv and q_load.csv, and optionally a timeseries plot of all normalized
load profiles.

Run with --help for CLI arguments, or run directly without arguments for GUI.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.analysis import (
    analyze_load_profiles,
    plot_load_profile_timeseries,
    plot_load_profile_timeseries_interactive,
)


def build_gui():
    """Build and run an interactive GUI dashboard for load profile analysis."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter not available; falling back to CLI mode.")
        return None

    # Find available data folders
    data_folder = ROOT / "data"
    available_networks = []
    if data_folder.exists():
        available_networks = sorted([
            d.name for d in data_folder.iterdir() 
            if d.is_dir() and (d / "p_load.csv").exists()
        ])

    # Create window
    window = tk.Tk()
    window.title("Load Profile Analysis Dashboard")
    window.geometry("600x500")

    # Title
    title = ttk.Label(window, text="Load Profile Analysis", font=("Arial", 16, "bold"))
    title.grid(row=0, column=0, columnspan=2, pady=10)

    # Network selection
    ttk.Label(window, text="Network:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    network_var = tk.StringVar(value=available_networks[0] if available_networks else "")
    network_combo = ttk.Combobox(window, textvariable=network_var, values=available_networks, state="readonly", width=50)
    network_combo.grid(row=1, column=1, padx=10, pady=5)

    # Plot type selection
    ttk.Label(window, text="Plot Type:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    plot_type_var = tk.StringVar(value="both")
    plot_type_combo = ttk.Combobox(
        window,
        textvariable=plot_type_var,
        values=["boxplot", "timeseries", "both"],
        state="readonly",
        width=50
    )
    plot_type_combo.grid(row=2, column=1, padx=10, pady=5)

    # Output directory
    ttk.Label(window, text="Output Directory:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    output_var = tk.StringVar(value="temp-load-analysis")
    output_entry = ttk.Entry(window, textvariable=output_var, width=52)
    output_entry.grid(row=3, column=1, padx=10, pady=5)

    # Profile multiplier
    ttk.Label(window, text="Profile Multiplier:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
    multiplier_var = tk.StringVar(value="1.0")
    multiplier_entry = ttk.Entry(window, textvariable=multiplier_var, width=52)
    multiplier_entry.grid(row=4, column=1, padx=10, pady=5)

    # Start timestep
    ttk.Label(window, text="Start Timestep (0-based):").grid(row=5, column=0, sticky="w", padx=10, pady=5)
    start_var = tk.StringVar(value="")
    start_entry = ttk.Entry(window, textvariable=start_var, width=52)
    start_entry.grid(row=5, column=1, padx=10, pady=5)

    # End timestep
    ttk.Label(window, text="End Timestep (0-based):").grid(row=6, column=0, sticky="w", padx=10, pady=5)
    end_var = tk.StringVar(value="")
    end_entry = ttk.Entry(window, textvariable=end_var, width=52)
    end_entry.grid(row=6, column=1, padx=10, pady=5)

    # Show plot checkbox
    show_var = tk.BooleanVar(value=True)
    show_check = ttk.Checkbutton(window, text="Show plots interactively", variable=show_var)
    show_check.grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    # Status label
    status_label = ttk.Label(window, text="", foreground="blue")
    status_label.grid(row=8, column=0, columnspan=2, pady=10)

    def run_analysis():
        """Run the analysis with selected options."""
        network = network_var.get()
        plot_type = plot_type_var.get()
        output_dir = output_var.get()
        multiplier_text = multiplier_var.get().strip()
        start = start_var.get()
        end = end_var.get()

        if not network:
            status_label.config(text="Error: Please select a network.", foreground="red")
            return

        input_folder = data_folder / network
        if not input_folder.exists():
            status_label.config(text=f"Error: {input_folder} does not exist.", foreground="red")
            return

        try:
            scale_factor = float(multiplier_text) if multiplier_text else 1.0
        except ValueError:
            status_label.config(text="Error: Profile multiplier must be a number.", foreground="red")
            return

        if scale_factor <= 0:
            status_label.config(text="Error: Profile multiplier must be greater than zero.", foreground="red")
            return

        status_label.config(text="Running analysis...", foreground="blue")
        window.update()

        try:
            # Run main analysis
            result = analyze_load_profiles(
                input_folder,
                output_dir=Path(output_dir) if output_dir else None,
                show=show_var.get(),
                scale_factor=scale_factor,
            )

            print(f"\n✓ Loaded {len(result.p_load.index)} snapshots across {len(result.p_load.columns)} buses")
            print(result.summary.head().to_string(index=False))

            if result.summary_path is not None:
                print(f"✓ Summary exported to {result.summary_path}")
            if result.plot_path is not None:
                print(f"✓ Box plot exported to {result.plot_path}")

            # Plot timeseries if requested
            if plot_type in ["timeseries", "both"]:
                start_step = int(start) if start else None
                end_step = int(end) if end else None
                ts_plot_path = plot_load_profile_timeseries(
                    result.p_load,
                    output_dir=Path(output_dir) if output_dir else None,
                    show=show_var.get(),
                    start_step=start_step,
                    end_step=end_step,
                )
                if ts_plot_path is not None:
                    print(f"✓ Timeseries plot exported to {ts_plot_path}")

                interactive_plot_path = plot_load_profile_timeseries_interactive(
                    result.p_load,
                    output_dir=Path(output_dir) if output_dir else None,
                    show=show_var.get(),
                    start_step=start_step,
                    end_step=end_step,
                )
                if interactive_plot_path is not None:
                    print(f"✓ Interactive timeseries exported to {interactive_plot_path}")

            status_label.config(text="✓ Analysis complete!", foreground="green")
        except Exception as ex:
            status_label.config(text=f"Error: {str(ex)}", foreground="red")
            print(f"Error: {ex}", file=sys.stderr)

    # Run button
    run_button = ttk.Button(window, text="Run Analysis", command=run_analysis)
    run_button.grid(row=9, column=0, columnspan=2, pady=20)

    window.mainloop()
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze p_load.csv and q_load.csv by bus.")
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to folder containing p_load.csv and q_load.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the box plot PNG and summary CSV",
    )
    parser.add_argument(
        "--timeseries",
        action="store_true",
        help="Also plot normalized load profiles over time",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Starting timestep index (0-based, inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending timestep index (0-based, inclusive)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open plots interactively in addition to saving them",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = analyze_load_profiles(args.input_folder, output_dir=args.output_dir, show=args.show)

    print(f"✓ Loaded {len(result.p_load.index)} snapshots across {len(result.p_load.columns)} buses")
    print(result.summary.head().to_string(index=False))

    if result.summary_path is not None:
        print(f"✓ Summary exported to {result.summary_path}")
    if result.plot_path is not None:
        print(f"✓ Box plot exported to {result.plot_path}")

    # Plot timeseries if requested
    if args.timeseries:
        output_dir = args.output_dir if args.output_dir else None
        ts_plot_path = plot_load_profile_timeseries(
            result.p_load,
            output_dir=output_dir,
            show=args.show,
            start_step=args.start,
            end_step=args.end,
        )
        if ts_plot_path is not None:
            print(f"✓ Timeseries plot exported to {ts_plot_path}")

        interactive_plot_path = plot_load_profile_timeseries_interactive(
            result.p_load,
            output_dir=output_dir,
            show=args.show,
            start_step=args.start,
            end_step=args.end,
        )
        if interactive_plot_path is not None:
            print(f"✓ Interactive timeseries exported to {interactive_plot_path}")


if __name__ == "__main__":
    # If no arguments provided, show GUI
    if len(sys.argv) == 1:
        gui_result = build_gui()
        if gui_result is None:
            # GUI not available, show help
            sys.argv.append("--help")
            main()
    else:
        # Parse CLI arguments
        main()



"""
example usage:
python examples/analyze_load_profiles.py data/80_bus_solar --output-dir temp-load-analysis-range --timeseries --start 0 --end 400

"""
