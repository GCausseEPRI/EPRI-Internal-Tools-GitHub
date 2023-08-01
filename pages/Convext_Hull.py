import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely import MultiPoint, Polygon
import shapely

from bokeh.plotting import figure, Figure
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Range1d, LinearAxis, DateRangeSlider, BasicTicker, LogTicker, LabelSet, HoverTool, FactorRange, Span, LinearColorMapper, ColorBar, LogColorMapper
from bokeh.layouts import gridplot, layout, grid, row, column
from bokeh.transform import dodge
from bokeh.core.properties import value
from bokeh.palettes import YlGnBu, linear_palette, Inferno256, Magma256, Greys256, Plasma256, Viridis256, Cividis256, Turbo256
from bokeh.transform import linear_cmap, log_cmap
from bokeh.util.hex import hexbin, axial_to_cartesian
from bokeh.embed import file_html
from bokeh.resources import CDN

def read_file(file: str) -> pd.DataFrame:
    if '.xlsx' in file.name:
        df = pd.read_excel(file, sheet_name=0)
    elif '.csv' in file.name:
        df = pd.read_csv(file)
    else:
        df = None
    
    return df

def get_plot_columns(df: pd.DataFrame) -> (str, str):
    # st.write(df.dtypes)
    plot_columns = df.select_dtypes(['int', 'float']).columns
    x_column = st.selectbox(
        label='Select X Column for plot',
        options=plot_columns
    )
    y_column = st.selectbox(
        label='Select Y Column for plot',
        options=plot_columns
    )

    return x_column, y_column

def plot(df, x_column, y_column):
    points = df[[x_column, y_column]].to_numpy()
    # st.write(points)
    # hull = ConvexHull(points)
    coords = list(map(tuple, np.asarray(points)))
    poly = Polygon(coords)
    x, y = shapely.concave_hull(poly, ratio=0.000001, allow_holes=True).exterior.xy

    p = figure(
		# x_axis_label='Time from Beginning of Cold Start',
		# y_axis_label=plot_column,
		aspect_ratio = 17.5 / 8,
		sizing_mode='scale_width',
		output_backend='webgl'
	)
    source = ColumnDataSource(df)
    p.circle(
        source=source,
        x=x_column,
        y=y_column
    )
    # st.write(x)
    # st.write(y)
    p.line(
        x=x,
        y=y,
        color='red'
    )

    st.bokeh_chart(p)

def main():
    st.header('Convex Hull Graphing Tool')

    st.info('NOTE: This tool assume your data is clean.')
    
    file = st.file_uploader(
        label='Upload your data',
        type=['csv', 'xlsx']
    )

    if file is not None:
        df = read_file(file)
        st.write(df.head())
        # df = pd.read_csv(file)
        # st.write(df.head())
        x_column, y_column = get_plot_columns(df)
        # st.write(x_column, y_column)
        plot(df, x_column, y_column)
    else:
        st.info('Please upload some data to get started')

if __name__ == '__main__':
    main()