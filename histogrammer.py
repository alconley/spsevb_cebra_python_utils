import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.colors import LogNorm
from matplotlib.backend_bases import LocationEvent
from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib import colormaps
from math import floor
from colorama import Fore, Style
from matplotlib.path import Path as mplPath
from polars import Series
import json
from pathlib import Path
from lmfit.models import GaussianModel, LinearModel
from lmfit.model import save_modelresult, load_modelresult, save_model, load_model
from scipy.signal import find_peaks
from tabulate import tabulate

class CutHandler:
    '''
    Handler to recieve vertices from a matplotlib selector (i.e. PolygonSelector).
    Typically will be used interactively, most likely via cmd line interpreter. The onselect
    method should be passed to the selector object at construction. CutHandler can also be used in analysis
    applications to store cuts.

    Example:

    ```[python]
    from evbutils import CutHandler, Cut2D, write_cut_json
    from matplotlib.widgets import PolygonSelector
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)
    handler = CutHandler()
    selector = PolygonSelector(ax, handler.onselect)

    #plot some data here...

    plt.show()
    
    #wait for user to draw a cut and close the window

    mycut = handler.cuts['cut_0']
    mycut.name = 'mycut'
    write_cut_json(mycut, 'mycut.json')
    ```
    '''
    def __init__(self):
        self.cuts: dict[str, Cut2D] = {}

    def onselect(self, vertices: list[tuple[float, float]]):
        cut_default_name = f'cut_{len(self.cuts)}'
        self.cuts[cut_default_name] = Cut2D(cut_default_name, vertices)

class Cut2D:
    '''
    Implementation of 2D cuts as used in many types of graphical analyses with matplotlib
    Path objects. Takes in a name (to identify the cut) and a list of points. The Path
    takes the verticies, and can then be used to check if a point(s) is inside of the polygon using the 
    is_*_inside functions. Can be serialized to json format. Can also retreive Nx2 ndarray of vertices
    for plotting after the fact.
    '''
    def __init__(self, name: str, vertices: list[tuple[float, float]]):
        self.path: mplPath = mplPath(vertices, closed=False) # Has to be false, sometimes without this, the algorithm does some weird jumping between the first and last point
        self.name = name
        
    def is_point_inside(self, x: float, y: float) -> bool:
        return self.path.contains_point((x,  y))

    def is_arr_inside(self, points: list[tuple[float, float]]) -> list[bool]:
        return self.path.contains_points(points)

    def is_cols_inside(self, columns: Series) -> Series:
        data = np.transpose([columns.struct.field(name).to_list() for name in columns.struct.fields])
        return Series(values=self.path.contains_points(data))

    def get_vertices(self) -> np.ndarray:
        return self.path.vertices

    def to_json_str(self) -> str:
        return json.dumps(self, default=lambda obj: {'name': obj.name, 'vertices': obj.path.vertices.tolist()}, indent=4)

def write_cut_json(cut: Cut2D, filepath: Path) -> bool:
    json_str = cut.to_json_str()
    try:
        with open(filepath, 'a') as output:
            output.write(json_str)
            return True
    except OSError as error:
        print(f'An error occurred writing cut {cut.name} to file {filepath}: {error}')
        return False

def load_cut_json(filepath: Path) -> Cut2D | None:
    try:
        with open(filepath, 'r') as input:
            buffer = input.read()
            cut_dict = json.loads(buffer)
            if 'name' not in cut_dict or 'vertices' not in cut_dict:
                print(f'Data in file {filepath} is not the right format for Cut2D, could not load')
                return None
            return Cut2D(cut_dict['name'], cut_dict['vertices'])
    except OSError as error:
        print(f'An error occurred reading trying to read a cut from file {filepath}: {error}')
        return None


'''
Built on top/modified this script which was created by Gordan McCann
'''

CMAP = colormaps.get_cmap('viridis').with_extremes(under='white')

#Utility functions
def clamp_low(x: float, edge: float) -> float:
    return x if x > edge else edge

def clamp_hi(x: float, edge: float) -> float:
    return x if x < edge else edge

def clamp_range(xrange: tuple[float, float], min_max: tuple[float, float]):
    return (clamp_low(xrange[0], min_max[0]), clamp_hi(xrange[1], min_max[1]))

'''
Hist1D, Hist2D
Dataclasses storing histogram data (name, counts per bin, bin edges)
When going to plot use the following idioms:

Hist1D:
    matplotlib.pyplot.stairs(hist.counts, hist.bins)

Hist2D:
    matplotlib.pyplot.pcolormesh(hist.x_bins, hist.y_bins, hist.counts)
'''
@dataclass
class Hist1D:
    '''
    Container wrapping a numpy array used to store histogram data and retrieve histogram statistics
    '''
    name: str
    counts: NDArray[np.float64]
    bins: NDArray[np.float64]
    bin_width: float

    def get_bin(self, x: float) -> int | None:
        if x < self.bins.min() or x > self.bins.max():
            return None
        
        return int(floor((x - self.bins[0]) / self.bin_width))

    #returns (integral, mean, std_dev)
    def stats_for_range(self, xrange: tuple[float, float]) -> tuple[float, float, float] | None:
        clamped_range = clamp_range(xrange, (self.bins.min(), self.bins.max()))
        bin_min = self.get_bin(clamped_range[0])
        bin_max = self.get_bin(clamped_range[1])
        if bin_min is None or bin_max is None:
            return None
        integral = np.sum(self.counts[bin_min:bin_max])
        if integral == 0:
            mean = 0
            variance = 0
        else:
            mean = np.average(self.bins[bin_min:bin_max], weights=self.counts[bin_min:bin_max])
            variance = np.average((self.bins[bin_min:bin_max] - mean)**2.0, weights=self.counts[bin_min:bin_max])
        return (integral, mean, np.sqrt(variance))
    
    #returns subrange bin_edges, counts
    def get_subrange(self, xrange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        mask = np.logical_and(self.bins > xrange[0], self.bins < xrange[1])
        return (self.bins[mask], self.counts[mask[:-1]])
    
    #returns subrange bin_centers, counts
    def get_subrange_centers(self, xrange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        bin_centers = self.bins + self.bin_width/2
        mask = np.logical_and(bin_centers > xrange[0], bin_centers < xrange[1])
        return (bin_centers[mask], self.counts[mask[:-1]])
    
    def get_counts(self) -> np.ndarray:
        return self.counts
    
    def get_bin_centers(self) -> np.ndarray:
        return (self.bins[:-1] + self.bins[1:]) / 2

@dataclass
class Hist2D:
    
    '''
    Container wrapping a numpy array used to store two-dimensional histogram data and retrieve histogram statistics
    '''
    name: str
    counts: NDArray[np.float64]
    x_bins: NDArray[np.float64]
    y_bins: NDArray[np.float64]
    x_bin_width: float
    y_bin_width: float

    def get_bin(self, coords: tuple[float, float]) -> tuple[int, int] | None:
        if (coords[0] < self.x_bins.min() or coords[0] > self.x_bins.max()) or (coords[1] < self.y_bins.min() or coords[1] > self.y_bins.max()):
            return None

        y_bin = int(floor((coords[1] - self.y_bins[0]) / self.y_bin_width))
        x_bin = int(floor((coords[0] - self.x_bins[0]) / self.x_bin_width))
        return (x_bin, y_bin)

    #returns (integral, mean x, std_dev x, mean y, std_dev y)
    def stats_for_range(self, xrange: tuple[float, float], yrange: tuple[float, float]) -> tuple[float, float, float, float, float] | None:
        clamped_x_range = clamp_range(xrange, (self.x_bins.min(), self.x_bins.max()))
        clamped_y_range = clamp_range(yrange, (self.y_bins.min(), self.y_bins.max()))
        bin_min = self.get_bin((clamped_x_range[0], clamped_y_range[0]))
        bin_max = self.get_bin((clamped_x_range[1], clamped_y_range[1]))

        x_bin_range = np.arange(start=bin_min[0], stop=bin_max[0], step=1)
        y_bin_range = np.arange(start=bin_min[1], stop=bin_max[1], step=1)
        bin_mesh = np.ix_(y_bin_range, x_bin_range)

        integral = np.sum(self.counts[bin_mesh])
        if integral == 0:
            mean_x = 0
            mean_y = 0
            var_x = 0
            var_y = 0
        else:
            mean_x = np.average(self.x_bins[bin_min[0]:bin_max[0]], weights=np.sum(self.counts.T[bin_min[0]:bin_max[0]], 1))
            mean_y = np.average(self.y_bins[bin_min[1]:bin_max[1]], weights=np.sum(self.counts[bin_min[1]:bin_max[1]], 1))
            var_x = np.average((self.x_bins[bin_min[0]:bin_max[0]] - mean_x)**2.0, weights=np.sum(self.counts.T[bin_min[0]:bin_max[0]], 1))
            var_y = np.average((self.y_bins[bin_min[1]:bin_max[1]] - mean_y)**2.0, weights=np.sum(self.counts[bin_min[1]:bin_max[1]], 1))
        return (integral, mean_x, mean_y, np.sqrt(var_x), np.sqrt(var_y))

    def get_subrange(self, xrange: tuple[float, float], yrange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_mask = np.logical_and(self.x_bins > xrange[0], self.x_bins < xrange[1])
        y_mask = np.logical_and(self.y_bins > yrange[0], self.y_bins < yrange[1])
        bin_mesh = np.ix_(y_mask, x_mask)
        return (self.x_bins[x_mask], self.y_bins[y_mask], self.counts[bin_mesh])

class Histogrammer:
    '''
    Histogrammer is a wrapper around a dictionary of str->Hist1D|Hist2D
    A new histogram can be added to the dictionary using the add_hist1d/add_hist2d methods. The name passed to
    these methods is used as the key for the dictionary. To add data to the histograms use the fill_hist1d/fill_hist2d methods.
    The fill methods accept arrays of data, and this is by intent. It would not be efficient to fill the histograms point by point. Rather, prefer
    passing entire data sets (like dataframe columns). Finally, to retrieve a histogram (for plotting, etc), use the get_hist1d/get_hist2d methods.
    Prefer the getters over direct access to the underlying dictionary as the getters perfom some error checking.

    ## Note
    Should be pickle-able -> We can save histograms in a concise binary way
    '''
    
    tex_fonts = {
                    # Use LaTeX to write all text
                    # "text.usetex": True,
                    # "font.family": "serif",
                    # "font.serif": "Computer Modern Roman",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }

    plt.rcParams.update(tex_fonts)

    plt.rcParams['keymap.pan'].remove('p')
    plt.rcParams['keymap.home'].remove('r')
    plt.rcParams['keymap.fullscreen'].remove('f')
    plt.rcParams['keymap.grid'].remove('g')
    plt.rcParams['keymap.grid_minor'].remove('G')
    plt.rcParams['keymap.quit_all'].append('Q')
    plt.rcParams['keymap.xscale'].remove('L')
    plt.rcParams['keymap.xscale'].remove('k')
    plt.rcParams['keymap.yscale'].remove('l')
        
    def __init__(self):
        self.histograms: dict[str, Hist1D | Hist2D] = {}
        self.axes: dict[Axes, tuple[str, Text | None]] = {}
        self.figures: dict[str, bool] = {} #used to indicate if callbacks have been bound for that figure
        
        self.handler = CutHandler()
        self.selectors = {}
        
        self.interactive_fitters = {}
        
    def add_hist1d(self, name: str, bins: int, range: tuple[float, float]):
        if name in self.histograms:
            print(f'Overwriting histogram named {name} in Histogrammer.add_histogram!')

        hist = Hist1D(name, np.empty(0), np.empty(0), np.abs(range[0] - range[1])/float(bins))
        hist.counts, hist.bins = np.histogram(a=[], bins=bins, range=range)
        self.histograms[name] = hist

    def add_hist2d(self, name: str, bins: tuple[int, int], ranges: tuple[tuple[float, float], tuple[float, float]]):
        if name in self.histograms:
            print(f'Overwriting histogram named {name} in Histogrammer.add_histogram!')

        hist = Hist2D(name, np.empty(0), np.empty(0), np.empty(0), np.abs(ranges[0][0] - ranges[0][1])/float(bins[0]), np.abs(ranges[1][0] - ranges[1][1])/float(bins[1]))
        hist.counts, hist.x_bins, hist.y_bins = np.histogram2d(x=[], y=[], bins=bins, range=ranges)
        hist.counts = hist.counts.T
        self.histograms[name] = hist

    def fill_hist1d(self, name: str, data: np.ndarray) -> bool:
        if name not in self.histograms:
            return False

        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return False

        hist.counts = hist.counts + np.histogram(a=data, bins=hist.bins)[0]
        return True

    def fill_hist2d(self, name: str, x_data: np.ndarray, y_data: np.ndarray) -> bool:
        if name not in self.histograms:
            return False
        
        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return False
        counts, _, _ = np.histogram2d(x_data, y_data, bins=(hist.x_bins, hist.y_bins))
        hist.counts += counts.T
        return True

    def add_fill_hist1d(self, name, bins, range, data):
        self.add_hist1d(name=name, bins=bins, range=range)
        self.fill_hist1d(name=name, data=data)
        
    def add_fill_hist2d(self, name, bins, ranges, x_data, y_data):
        self.add_hist2d(name=name, bins=bins, ranges=ranges)
        self.fill_hist2d(name=name, x_data=x_data, y_data=y_data)

    def get_hist1d(self, name: str) -> Hist1D | None:
        if name not in self.histograms:
            return None
        
        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return None
        else:
            return hist

    def get_hist2d(self, name: str) -> Hist2D | None:
        if name not in self.histograms:
            return None
        
        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return None
        else:
            return hist

    def on_axes_enter_hist1d(self, event: LocationEvent):
        data = self.axes[event.inaxes]
        xrange = event.inaxes.get_xbound()
        yrange = event.inaxes.get_ybound()
        stats = self.histograms[data[0]].stats_for_range(xrange)
        if data[1] is not None:
            data[1].remove()

        text = f'Integral: {stats[0]}\nCentroid: {stats[1]:.3f}\nSigma: {stats[2]:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')  
        self.axes[event.inaxes] = (data[0], event.inaxes.text(0.95, 0.95, text, transform= event.inaxes.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props))
        
        plt.draw()

    def on_axes_enter_hist2d(self, event: LocationEvent):
        data = self.axes[event.inaxes]
        xrange = event.inaxes.get_xbound()
        yrange = event.inaxes.get_ybound()
        stats = self.histograms[data[0]].stats_for_range(xrange, yrange)
        if data[1] is not None:
            data[1].remove()
        
        text = f'Integral: {stats[0]}\nCentroid X: {stats[1]:.3f}\nCentroid Y: {stats[2]:.3f}\nSigma X: {stats[3]:.3f}\nSigma Y: {stats[4]:.3f}'

        props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')  
        self.axes[event.inaxes] = (data[0], event.inaxes.text(0.95, 0.95, text, transform=event.inaxes.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props))
        
        plt.draw()

    def on_axes_enter(self, event: LocationEvent):
        if event.inaxes not in self.axes:
            return
        
        if type(self.histograms[self.axes[event.inaxes][0]]) is Hist1D:
            self.on_axes_enter_hist1d(event)
            
        elif type(self.histograms[self.axes[event.inaxes][0]]) is Hist2D:
            self.on_axes_enter_hist2d(event)

    def on_axes_leave(self, event: LocationEvent):
        if event.inaxes not in self.axes:
            return
        data = self.axes[event.inaxes]
        if data[1] is None:
            return
        data[1].remove()
        self.axes[event.inaxes] = (data[0], None)
        plt.draw()
        
    def connect_mpl_callbacks(self, axis: Axes):
        if not hasattr(axis.figure, '_suptitle'):
            axis.figure.suptitle(f'Figure {len(self.figures)}')
        elif axis.figure._suptitle in self.figures:
            return

        self.figures[axis.figure._suptitle] = True
        axis.figure.canvas.mpl_connect('axes_enter_event', self.on_axes_enter)
        axis.figure.canvas.mpl_connect('axes_leave_event', self.on_axes_leave)
                            
    def draw_hist1d(
        self, 
        name: str, 
        axis: Axes,
        xlabel: str = None,
        ylabel: str = None,
        label: str = None,
        title: str = None,
        color: str = None,
        linestyle: str = None,
        linewidth: float = None,
        display_stats: bool = True):
        
        if name not in self.histograms:
            return

        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return

        if linewidth is None: linewidth = 0.5

        axis.stairs(hist.counts, hist.bins, linewidth=linewidth, color=color)
        
        axis.set_xlabel(xlabel if xlabel is not None else name)
        axis.set_ylabel(ylabel if ylabel is not None else "Counts")
        if title is not None: axis.set_title(title)
        axis.legend() if label is not None else None
    
        axis.minorticks_on()
        axis.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
        axis.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)
         
        if display_stats:
            self.axes[axis] = (name, None)
            self.connect_mpl_callbacks(axis)
                    
        interactive_1d_histogram = Interactive1DHistogram(axis, hist.counts, hist.bins)

        axis.figure.canvas.mpl_connect('key_press_event', interactive_1d_histogram.on_press)

        # Store the interactive histogram instance in self.axes
        self.interactive_fitters[axis] = [name, interactive_1d_histogram]
                                
    def draw_hist2d(
        self, 
        name: str, 
        axis: Axes,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        display_stats: bool = True,
        log_z: bool = False) -> QuadMesh | None:
        if name not in self.histograms:
            return None
        
        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return None
        mesh = None
        
        if log_z:
            mesh = axis.pcolormesh(hist.x_bins, hist.y_bins, hist.counts, cmap=CMAP, norm=LogNorm())
        else:
            mesh = axis.pcolormesh(hist.x_bins, hist.y_bins, hist.counts, cmap=CMAP, vmin=1.0e-6)
            
        axis.set_title(title if title is not None else name)        
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
                
        axis.minorticks_on()
        axis.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
        axis.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)

        self.axes[axis] = (name, None)

        if display_stats:
            self.connect_mpl_callbacks(axis)
            

        selector = PolygonSelector(ax=axis, onselect=self.handler.onselect, useblit=True)
        selector.set_active(False)
        
        self.selectors[axis] = [name, selector]
        
        interactive_2d_histogram = Interactive2DHistogram(axis, hist.counts, hist.x_bins, hist.y_bins, self.handler, selector)
        axis.figure.canvas.mpl_connect('key_press_event', interactive_2d_histogram.on_press)
        
        self.interactive_fitters[axis] = [name, interactive_2d_histogram]
        
        return mesh
    
class Interactive2DHistogram:
    
    def __init__(self, axis: Axes, counts, x_bins, y_bins, cut_handler, selector):
        
        self.selector = selector
        self.cut_handler = cut_handler
        
        self.axis = axis
        self.counts = counts
        self.x_bins = x_bins
        self.x_bin_width = x_bins[1]-x_bins[0]
        
        self.y_bins = y_bins
        self.y_bin_width = self.y_bins[1] - self.y_bins[0]
        
        self.y_markers = []
        self.x_markers = []
        
        self.interactive_fitters = {}
        
    def get_subrange(self, xrange: tuple[float, float], yrange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_mask = np.logical_and(self.x_bins > xrange[0], self.x_bins < xrange[1])
        y_mask = np.logical_and(self.y_bins > yrange[0], self.y_bins < yrange[1])
        bin_mesh = np.ix_(y_mask, x_mask)
        return (self.x_bins[x_mask], self.y_bins[y_mask], self.counts[bin_mesh])
        
    def handle_show_keybindings_help(self):
        keybindings = {
                
                'x': {
                    'description': "Add a vertical line to view the X-projection",
                    'note': "Must have two lines to view the X-projection",
                },
                'X': {
                    'description': "Opens a 1D histogram of the X-projection between the two X-projection lines",
                    'note': "",
                },
                'y': {
                    'description': "Add a vertical line to view the Y-projection",
                    'note': "Must have two lines to view the Y-projection",
                },
                'Y': {
                    'description': "Opens a 1D histogram of the Y-projection between the two Y-projection lines",
                    'note': "",
                },
                'c': {
                    'description': "Enables Matplotlib's polygon selector tool",
                    'note': "Left click to place vertices. Once the shape is completed, the vertices can be moved by dragging them.",
                },
                'C': {
                    'description': "Saves the cut",
                    'note': "User has to input the filename in the terminal (e.g. cut.json)",
                },

                'space-bar': {
                    'description': "Show keybindings help",
                    'note': "",
                },
            }
            
        print("\nKeybindings Help:")
        for key, info in keybindings.items():
            description = info['description']
            note = info['note']
            print(f"  {Fore.YELLOW}{key}{Style.RESET_ALL}: {description}")
            if note:
                print(f"      Note: {note}")
        
    def remove_lines(self, lines): # removes axvlines from an array
        for line in lines:
            line.remove()
        lines.clear()
        
    def handle_y_markers(self, event):
        if len(self.y_markers) >= 2: # If two lines are present, remove them from the plot and the list  
            self.remove_lines(self.y_markers)

        x_coord = event.xdata
        line = self.axis.axvline(x_coord, color='red', linewidth=0.5)
        self.y_markers.append(line)

        plt.draw()
        
    def handle_x_markers(self, event):
                    
        if len(self.x_markers) >= 2:
            self.remove_lines(self.x_markers)

        y_coord = event.ydata
        line = self.axis.axhline(y_coord, color='green', linewidth=0.5)
        self.x_markers.append(line)

        plt.draw()
        
    def handle_y_projections(self):
        
        if len(self.y_markers) < 2: 
            print(f"{Fore.RED}{Style.BRIGHT}Must have two lines!{Style.RESET_ALL}")

        x_coordinates = []
        for line in self.y_markers:
            x_coordinate = line.get_xdata()[0] 
            x_coordinates.append(x_coordinate)
        x_coordinates.sort() 

        y_lims = self.axis.get_ylim()

        projection_data = self.get_subrange(xrange=x_coordinates, yrange=y_lims)

        y_bins_projection = projection_data[1]
        y_bins_projection = np.append(y_bins_projection, self.y_bin_width) #plt.stairs need the length of edges to be 1 more than data

        y_data = projection_data[2].sum(axis=1) #sum over rows

        fig, ax = plt.subplots(1,1)
        ax.stairs(y_data, y_bins_projection, linewidth=0.5)
        
        interactive_1d_histogram = Interactive1DHistogram(ax, y_data, y_bins_projection)
        ax.figure.canvas.mpl_connect('key_press_event', interactive_1d_histogram.on_press)
        
        self.interactive_fitters[ax] = [interactive_1d_histogram]
        
        plt.show()
        
    def handle_x_projections(self):
        
        if len(self.x_markers) < 2: 
            print(f"{Fore.RED}{Style.BRIGHT}Must have two lines!{Style.RESET_ALL}")
        
        y_coordinates = []
        for line in self.x_markers:
            y_coordinate = line.get_ydata()[0] 
            y_coordinates.append(y_coordinate)
        y_coordinates.sort() 
        
        x_lims = self.axis.get_xlim()
        
        projection_data = self.get_subrange(xrange=x_lims, yrange=y_coordinates)
        
        x_bins_projection = projection_data[0]
        x_bins_projection = np.append(x_bins_projection, self.x_bin_width) #plt.stairs need the length of edges to be 1 more than data
        
        x_data = projection_data[2].sum(axis=0) #sum over columns 
        
        fig, ax = plt.subplots(1,1)
        ax.stairs(x_data, x_bins_projection, linewidth=0.5)
        
        interactive_1d_histogram = Interactive1DHistogram(ax, x_data, x_bins_projection)
        ax.figure.canvas.mpl_connect('key_press_event', interactive_1d_histogram.on_press)
        
        self.interactive_fitters[ax] = [interactive_1d_histogram]
        
        
        plt.show()

    def handle_cut_selector(self):         
        print(f"{Fore.YELLOW}Activating the polygon selector tool:\n\tPress 'C' to save the cut (must enter cut name e.g. cut.json){Style.RESET_ALL}")
        self.selector.set_active(True)
        plt.show()
        
    def handle_save_cut(self, event):
        self.selector.set_active(False)
        plt.show()
        
        # hist_name_cut = self.axes[event.inaxes][0]
        
        self.cut_handler.cuts["cut_0"].name = "cut"
        
        # Prompt the user for the output file name
        output_file = input(f"{Fore.YELLOW}Enter a name for the output file (e.g., cut.json): {Style.RESET_ALL}")

        # Write the cut to the specified output file
        try:
            write_cut_json(cut=self.cut_handler.cuts["cut_0"], filepath=output_file)
            print(f"{Fore.GREEN}Cut saved to '{output_file}' successfully.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error: {e}. Failed to save the cut to '{Style.RESET_ALL}'.")
            
    def on_press(self, event: LocationEvent):  # Function to handle mouse click events like x/y projections
        
        if event.inaxes == self.axis:
            
            if event.key == ' ': 
                self.handle_show_keybindings_help()
                
            elif event.key == 'y': # For drawing lines to do a y-projection
                self.handle_y_markers(event)
                
            if event.key == 'x': # For drawing lines to do a x-projection
                self.handle_x_markers(event)
    
            if event.key == 'Y': # For showing the y-projection
                self.handle_y_projections()
                                        
            if event.key == 'X': # For showing the y-projection
                self.handle_x_projections()
                    
            if event.key == 'c': # create a cut
                self.handle_cut_selector()

            if event.key == 'C': # save the cut to a file name that the user must enter
                self.handle_save_cut(event)

class Interactive1DHistogram:
    COLOR_REGION = 'blue'
    COLOR_BACKGROUND = 'green'
    COLOR_PEAK = 'purple'
    LINE_WIDTH = 0.5

    def __init__(self, axis: Axes, counts, bins):
        
        self.axis = axis
        self.counts = counts

        self.bins = bins # left edge
        self.bin_width = bins[1]-bins[0]
        
        self.bin_centers = self.bins + self.bin_width/2 
        self.bin_centers = self.bin_centers[:-1] # center of bins
        
        self.region_markers = []
        self.peak_markers = []
        self.background_markers = []
        self.background_lines = []
        self.fit_lines = []
        self.temp_fits = {}
        
    #returns subrange bin_centers, counts
    def get_subrange(self, xrange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        mask = np.logical_and(self.bin_centers > xrange[0], self.bin_centers < xrange[1])
        return (self.bin_centers[mask], self.counts[mask])
       
    def create_marker(self, x_coord: float, color: str):
        line = self.axis.axvline(x_coord, color=color, linewidth=self.LINE_WIDTH)
        plt.draw()
        return line
                
    def remove_lines(self, lines: list):
        for line in lines:
            line.remove()
        lines.clear()   
        
    def remove_marker(self, marker_array:list, index:int):
        if index < len(marker_array):
            marker_array[index].remove()
            marker_array.pop(index)
             
    def get_marker_positions(self, markers:list): 
        positions = [marker.get_xdata()[0] for marker in markers]
        positions.sort()
        return positions  
        
    def handle_show_keybindings_help(self): # Function to display the keybindings help
    
        keybindings = {
            'r': {
                'description': "Add region marker",
                'note': "Must have 2 region markers to preform a fit.",
            },
            'b': {
                'description': "Add background marker",
                'note': "must have at least two background markers to estimate the background",
            },
            'p': {
                'description': "Add peak marker",
                'note': "If no peak markers are supplied, the program will assume there is one peak at the maximum value",
            },
            'P': {
                'description': "Auto peak finder",
                'note': "Trys to find all the peaks between the region markers",
            },
            '-': {
                'description': "Remove the nearest marker to curser",
                'note': "",
            },
            '_': {
                'description': "Remove all markers and temp fits",
                'note': "",
            },
            'B': {
                'description': "Fit background",
                'note': "Fits the background markers using a linear line",
            },
            'f': {
                'description': "Fit Gaussians to region",
                'note': "Must have two region markers. If no background markers are supplied, the background will be esitamted at the region markers. Number of Gaussians fitted will depend on the number of peak markers inbetween the region markers",
            },
            'F': {
                'description': "Store the fits",
                'note': "",
            },
            'S': {
                'description': "Save fits to file",
                'note': "Saves the stored fits to a ASCII (user must input the file name)",
            },
            'L': {
                'description': "Load fits from file",
                'note': "",
            },
            'space-bar': {
                'description': "Show keybindings help",
                'note': "",
            },
       }

        print("\nKeybindings Help:")
        for key, info in keybindings.items():
            description = info['description']
            note = info['note']
            print(f"  {Fore.YELLOW}{key}{Style.RESET_ALL}: {description}")
            if note:
                print(f"      Note: {note}")
        
    def handle_region_marker(self, event: LocationEvent):
        if len(self.region_markers) >= 2:
            print(f"{Fore.BLUE}Removed region markers{Style.RESET_ALL}")
            self.remove_lines(self.region_markers)
        x_coord = event.xdata
        line = self.create_marker(x_coord, self.COLOR_REGION)
        self.region_markers.append(line)
       
    def handle_background_marker(self, event: LocationEvent):
        x_coord = event.xdata
        line = self.create_marker(x_coord, self.COLOR_BACKGROUND)
        self.background_markers.append(line)
        
    def handle_peak_marker(self, event: LocationEvent):
        x_coord = event.xdata
        line = self.create_marker(x_coord, self.COLOR_PEAK)
        self.peak_markers.append(line)
        
    def handle_remove_markers(self):
        self.remove_lines(self.region_markers)
        self.remove_lines(self.peak_markers)
        self.remove_lines(self.background_markers)
        self.remove_lines(self.background_lines)
        self.remove_lines(self.fit_lines)
        self.temp_fits.clear()
        plt.draw()
       
    def handle_remove_closest_marker(self, event: LocationEvent):
        self.remove_lines(self.background_lines)
        self.remove_lines(self.fit_lines)
        self.temp_fits.clear()

        x_cursor = event.xdata

        def line_distances(marker_array):
            if len(marker_array) > 0:
                distances = [x_cursor - line.get_xdata()[0] for line in marker_array]
                min_distance = np.min(np.abs(distances))
                min_distance_index = np.argmin(np.abs(distances))
                return [min_distance, min_distance_index]
            else:
                return None

        marker_distances = [line_distances(self.region_markers), line_distances(self.peak_markers), line_distances(self.background_markers)]

        # Filter out None values and find minimum distances
        valid_marker_distances = [min_distance for min_distance in marker_distances if min_distance is not None]

        # Check if there are valid distances
        if valid_marker_distances:
            # Find the minimum distance based on the first index
            min_distance = min(valid_marker_distances, key=lambda x: x[0])

            if min_distance == marker_distances[0]:
                self.remove_marker(self.region_markers, min_distance[1])

            elif min_distance == marker_distances[1]:
                self.remove_marker(self.peak_markers, min_distance[1])

            elif min_distance == marker_distances[2]:
                self.remove_marker(self.background_markers, min_distance[1])
        else:
            print(f"{Fore.RED}{Style.BRIGHT}No valid distances found.{Style.RESET_ALL}")

        plt.draw()
    
    def handle_fit_background(self, markers: list, hist_counts: np.ndarray, hist_bin_centers: np.ndarray):
            
        self.remove_lines(self.background_lines)
        
        if len(markers) < 2:
            print(f"{Fore.RED}{Style.BRIGHT}Must have two or more background markers!{Style.RESET_ALL}")
            return None, None, None
                    
        background_positions = self.get_marker_positions(markers)
        background_y_values = [hist_counts[np.argmin(np.abs(hist_bin_centers - pos))] for pos in background_positions]
        
        background_model = LinearModel()
        background_result = background_model.fit(background_y_values, x=background_positions)
        
        background_x_values = np.linspace(background_positions[0], background_positions[-1], 1000)
        background_values = background_result.eval(x=background_x_values)
        
        background_line = self.axis.plot(background_x_values, background_values, color=self.COLOR_BACKGROUND, linewidth=self.LINE_WIDTH)
        self.background_lines.append(background_line[0])
        
        plt.draw()
        
        return background_result, background_model, background_line
    
    def handle_auto_fit_peaks(self):
        if len(self.region_markers) != 2:
            print(f"{Fore.RED}{Style.BRIGHT}Must have two region markers!{Style.RESET_ALL}")
        else:
            region_markers_pos = self.get_marker_positions(self.region_markers)
            
            bins_fit_range, counts_fit_range = self.get_subrange(xrange=region_markers_pos)
            
            self.remove_lines(self.peak_markers)

            if not self.background_markers:
                # if no background markers, estimate background at region markers
                self.remove_lines(self.background_lines)
                background_result, background_model, background_line = self.handle_fit_background(self.region_markers, counts_fit_range, bins_fit_range)
            else:
                self.remove_lines(self.background_lines)
                background_result, background_model, background_line  = self.handle_fit_background(self.background_markers, counts_fit_range, bins_fit_range)

            hist_counts_subtracted = counts_fit_range - background_result.eval(x=bins_fit_range)
            
            peaks, _ = find_peaks(x=hist_counts_subtracted, height=np.max(hist_counts_subtracted)*0.05, threshold=0.05)
            
            for peak in bins_fit_range[peaks]:                            
                line = self.axis.axvline(peak, color=self.COLOR_PEAK, linewidth=self.LINE_WIDTH)                    
                self.peak_markers.append(line)
                
            plt.draw()
    
    def initial_gaussian_parameters(self, counts: np.ndarray, bins: np.ndarray, peak_positions: list, position_uncertainty: float): # estimates the initital fit parameters

        if len(peak_positions) == 0: # if there are no peak markers, guess the center is at the max value and append that value to the list
            peak_positions.append(bins[np.argmax(counts)])
            
        def hist_counts_subtracted_value(number): # get the value of the closest bin to the peak posititon
            index = np.argmin(np.abs(bins - number))
            value = counts[index]
            return value
        
        # for guessing the amplitude
        total_peak_height = 0
        for i, peak in enumerate(peak_positions):
            total_peak_height += hist_counts_subtracted_value(peak)
        
        initial_parameters = []
        
        for peak in peak_positions:
        
            center = dict(value=peak,
                        min=peak - position_uncertainty,
                        max=peak + position_uncertainty)
            
            sigma = dict(value=self.bin_width, min=self.bin_width, max=self.bin_width*4) # could use some work to better determine this
            
            height_guess = hist_counts_subtracted_value(peak)
            height = dict(value=height_guess,
                        min=height_guess*0.9,
                        max=height_guess*1.1)
            
            amp_scale = height_guess/total_peak_height
            amplitude = dict(value=self.bin_width * np.sum(counts)*amp_scale)
            
            initial_parameters.append([sigma, center, height, amplitude])
            
        return initial_parameters
    
    def fit_multiple_gaussians(self, counts: np.ndarray, bins: np.ndarray, peak_positions: list):
    
        # Initialize the list of Gaussian models and their parameters
        gaussian_models = []
        initial_parameters = self.initial_gaussian_parameters(counts=counts, bins=bins, 
                                                            peak_positions=peak_positions, position_uncertainty=3*self.bin_width)
        
        # Loop over the peak_positions and create Gaussian models and parameters
        for i, peak_position in enumerate(peak_positions):
            gauss = GaussianModel(prefix=f'g{i}_')
            
            init_para = initial_parameters[i] 
            if i == 0:
                params = gauss.make_params(sigma=init_para[0],
                                            center=init_para[1],
                                            height=init_para[2],
                                            amplitude=init_para[3])
                
            else:
                params.update(gauss.make_params(sigma=init_para[0],
                                            center=init_para[1],
                                            height=init_para[2],
                                            amplitude=init_para[3]))
                
            gaussian_models.append(gauss)
            
        # Create the composite model by adding all Gaussian models together
        composite_model = gaussian_models[0]
        for gauss in gaussian_models[1:]:
            composite_model += gauss

        # Fit the composite model to the data
        result = composite_model.fit(counts, params, x=bins)
        
        return result, composite_model
  
    def gaussian_result_formatted(self, result, prefix, name, print_table=False):
        
        # Get the center, amplitude, sigma, and FWHM for each Gaussian
        center_value = result.params[f'{prefix}center'].value
        amplitude_value = result.params[f'{prefix}amplitude'].value
        area_value = amplitude_value/self.bin_width
        fwhm_value = abs(result.params[f'{prefix}fwhm'].value)
        relative_width_value = abs(fwhm_value/center_value *100)
        
        center_uncertainty = result.params[f'{prefix}center'].stderr
        if center_uncertainty is not None:
            center_formatted = f"{center_value:.4f} ± {center_uncertainty:4f}"
        else:
            center_formatted = f"{center_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
            
        amplitude_uncertainty = result.params[f'{prefix}amplitude'].stderr
        if amplitude_uncertainty is not None:
            area_uncertainty = amplitude_uncertainty/self.bin_width
            area_formatted = f"{area_value:.4f} ± {area_uncertainty:4f}"
        else:
            area_formatted = f"{area_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
            
        fwhm_uncertainty = result.params[f'{prefix}fwhm'].stderr
        if fwhm_uncertainty is not None:
            fwhm_formatted = f"{fwhm_value:.4f} ± {fwhm_uncertainty:4f}"
        else:
            fwhm_formatted = f"{fwhm_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
            
        if fwhm_uncertainty and center_uncertainty is not None:
            relative_width_value_uncertainty = relative_width_value*np.sqrt( (fwhm_uncertainty/fwhm_value)**2 +  (center_uncertainty/abs(center_value))**2 )
            relative_width_formatted = f"{relative_width_value:.4f} ± {relative_width_value_uncertainty:.4f}"
        else:
            relative_width_formatted = f"{relative_width_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
            
        # Append the formatted results to the list
        fit_result = [f'{name}', center_formatted, area_formatted, fwhm_formatted, relative_width_formatted]
        
        if print_table:
            # Define column headers for the table
            headers = ["Gaussian", "Position", "Volume", "FWHM", "Relative Width [%]"]

            # Print the table
            table = tabulate(fit_result, headers, tablefmt="pretty")
            print(table)

        return fit_result
        
    def handle_fit(self):
        self.remove_lines(self.background_lines)
        self.remove_lines(self.fit_lines)
        self.temp_fits.clear()
                        
        if len(self.region_markers) != 2:
            print(f"{Fore.RED}{Style.BRIGHT}Must have two region markers!{Style.RESET_ALL}")
        else:
            region_markers_pos = self.get_marker_positions(self.region_markers)
            
            bins_fit_range, counts_fit_range = self.get_subrange(xrange=region_markers_pos)

            # removes peak markers that are not in between the region markers
            peak_positions = []
            for marker in self.peak_markers:
                if region_markers_pos[0] < marker.get_xdata()[0] < region_markers_pos[1]:
                    peak_positions.append(marker.get_xdata()[0])
            peak_positions.sort()

            self.remove_lines(self.peak_markers)
                                    
            if not self.background_markers: # if no background markers/fit estimate the background at the region markers
                markers = self.region_markers
            else:
                markers = self.background_markers
            self.remove_lines(self.background_lines)
            
            background_result, background_model, background_line = self.handle_fit_background(markers, counts_fit_range, bins_fit_range)
                                                                
            hist_counts_subtracted_raw = counts_fit_range - background_result.eval(x=bins_fit_range)
            hist_counts_subtracted = [max(0.0, count) for count in hist_counts_subtracted_raw]
            
            initial_parameters = self.initial_gaussian_parameters(hist_counts_subtracted, bins_fit_range, peak_positions, position_uncertainty=3*self.bin_width)
            
            result, composite_model = self.fit_multiple_gaussians(hist_counts_subtracted, bins_fit_range, peak_positions)
            
            # plot result on top of the background
            total_x = np.linspace(region_markers_pos[0], region_markers_pos[1],2000)
            fit_p_background_line = self.axis.plot(total_x, result.eval(x=total_x) + background_result.eval(x=total_x), color='blue', linewidth=self.LINE_WIDTH) 
            self.fit_lines.append(fit_p_background_line[0])
            
            print(f"{Fore.GREEN}{Style.BRIGHT}Fit Report{Style.RESET_ALL}")
            # print(result.fit_report())
            
            fit_results = []
            
            # Decomposition of gaussians
            for i, peak_position in enumerate(peak_positions):
                
                prefix = f'g{i}_'
                sigma_plot_width = 4
                x_comp = np.linspace(result.params[f'{prefix}center'].value - sigma_plot_width * result.params[f'{prefix}sigma'].value,
                    result.params[f'{prefix}center'].value + sigma_plot_width * result.params[f'{prefix}sigma'].value, 1000)

                components = result.eval_components(x=x_comp)

                fit_results.append(self.gaussian_result_formatted(result, prefix=prefix, name=i))

                fit_line_comp = self.axis.plot(x_comp, components[prefix], color='blue', linewidth=0.5)  # Gaussian without background
                self.fit_lines.append(fit_line_comp[0])
                
                fit_line_comp_p_background = self.axis.plot(x_comp, components[prefix] + background_result.eval(x=x_comp), color='blue', linewidth=self.LINE_WIDTH)  # Gaussian and background
                self.fit_lines.append(fit_line_comp_p_background[0])

                line = self.axis.axvline(result.params[f'{prefix}center'].value, color='purple', linewidth=self.LINE_WIDTH)                    
                self.peak_markers.append(line)
                
            # Define column headers for the table
            headers = ["Gaussian", "Position", "Volume", "FWHM", "Relative Width [%]"]

            # Print the table
            table = tabulate(fit_results, headers, tablefmt="pretty")
            print(table)
                    
            temp_fit_id = f"temp_fit_{len(self.temp_fits)}"
            self.temp_fits[temp_fit_id] = {
                "region_markers": region_markers_pos,
                "fit_model": composite_model,
                "fit_result": result,
                "fit_lines": self.fit_lines,
                "background_model": background_model,
                "background_result": background_result,
                "background_line": background_line, 
                "fit_p_background_line": fit_p_background_line[0]
            }
                
            plt.draw()
            
    def on_press(self, event: LocationEvent):  # Function to handle fitting gaussians

        if event.inaxes == self.axis:
                    
            if event.key == ' ':
                self.handle_show_keybindings_help()
        
            elif event.key == 'r':
                self.handle_region_marker(event)
                
            elif event.key == 'b':
                self.handle_background_marker(event)

            elif event.key == 'p':
                self.handle_peak_marker(event)
                    
            elif event.key == '_':
                self.handle_remove_markers()
            
            elif event.key == '-':
                self.handle_remove_closest_marker(event)
                
            elif event.key == 'B':
                self.handle_fit_background(self.background_markers, self.counts, self.bin_centers)

            elif event.key == 'P':
                self.handle_auto_fit_peaks()  
                    
            if event.key == 'f':  # Fit Gaussians to region
                self.handle_fit()

    # def call_back(self):        
    #     self.axis.figure.canvas.mpl_connect('key_press_event', self.on_press)
        
