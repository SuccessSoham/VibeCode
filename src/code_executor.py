import io
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def execute_code(code: str, df: pd.DataFrame):
    """
    Executes the given Python code in a restricted environment.
    Captures stdout and any plots generated.
    """
    # Create a dictionary to serve as the execution environment
    # Only allow access to pandas, matplotlib.pyplot, seaborn, and the DataFrame 'df'
    local_vars = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'df': df,
        '__builtins__': {
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'dict': dict,
            'list': list,
            'tuple': tuple,
            'set': set,
            'range': range,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'type': type,
            'isinstance': isinstance,
            'getattr': getattr,
            'setattr': setattr,
            'hasattr': hasattr,
            'dir': dir,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'all': all,
            'any': any,
            'next': next,
            'iter': iter,
            'super': super,
            'Exception': Exception,
            'TypeError': TypeError,
            'ValueError': ValueError,
            'KeyError': KeyError,
            'AttributeError': AttributeError,
            'IndexError': IndexError,
            'NameError': NameError,
            'SyntaxError': SyntaxError,
            'ZeroDivisionError': ZeroDivisionError,
            'MemoryError': MemoryError,
            'SystemExit': SystemExit,
            'KeyboardInterrupt': KeyboardInterrupt,
            'ImportError': ImportError,
            'RuntimeError': RuntimeError,
            'NotImplementedError': NotImplementedError,
            'Warning': Warning,
            'UserWarning': UserWarning,
            'DeprecationWarning': DeprecationWarning,
            'PendingDeprecationWarning': PendingDeprecationWarning,
            'SyntaxWarning': SyntaxWarning,
            'RuntimeWarning': RuntimeWarning,
            'FutureWarning': FutureWarning,
            'ImportWarning': ImportWarning,
            'UnicodeWarning': UnicodeWarning,
            'BytesWarning': BytesWarning,
            'ResourceWarning': ResourceWarning,
        }
    }

    # Capture stdout
    stdout_capture = io.StringIO()
    # Store generated figures
    figures = []

    # Override plt.show to capture figures instead of displaying them
    original_show = plt.show
    def captured_show():
        fig = plt.gcf()
        figures.append(fig)
        plt.close(fig) # Close the figure to prevent it from being displayed by default
    plt.show = captured_show

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {"__builtins__": local_vars["__builtins__"]}, local_vars)
        output = stdout_capture.getvalue()
        return output, figures, None
    except Exception as e:
        return stdout_capture.getvalue(), figures, str(e)
    finally:
        plt.show = original_show # Restore original plt.show
        plt.close('all') # Close any remaining figures
