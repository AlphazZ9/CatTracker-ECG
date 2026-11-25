"""
ecg.app
-------
The application layer the interface drives, with no user-interface
toolkit: the state of an open recording (state), the controllers that run
the analyses and write their results into it (analysis_controller,
detection_controller, signal_controller, session_controller,
event_locked_controller, navigation_controller, export_controller), the
figures (plot_controller, the *_figure / *_view modules, plots) and their
colours (theme), and what each result was computed from (freshness).

A controller works on an "app" object: the Qt interface's FigureHost
(ecg.qt.figure_host), or the stub app of the reference outputs
(tests/regression/golden.py). Message boxes and file choosers go through
ecg.app.dialogs, which the interface connects to its own.

These modules come from the CustomTkinter interface (ecg/ui, removed),
moved unchanged apart from their Tk widgets.
"""
