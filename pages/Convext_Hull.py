import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import numpy as np
from shapely import Polygon
import shapely

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FixedTicker, HoverTool, ColorBar
from bokeh.palettes import Inferno, Inferno256, Magma, Magma256, Greys, Greys256, Plasma, Plasma256, Viridis, Viridis256, Cividis, Cividis256, Turbo, Turbo256
from bokeh.transform import linear_cmap, log_cmap
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import CustomJSHover, HoverTool

import math

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

def get_additional_column(df: pd.DataFrame) -> (str, str):
    plot_columns = df.select_dtypes(['int', 'float']).columns

    z_column = st.selectbox(
        label='Select Z Column for plot',
        options=plot_columns
    )

    palette = st.selectbox(
        label='Select a color palette',
        options=[
            'Cividis',
            'Inferno',
            'Magma',
            'Plasma',
            'Viridis',
            'Turbo',
            'Grey'
        ]
    )

    return z_column, palette


def plot(df: pd.DataFrame, x_column: str, y_column: str, ratio: float, holes: bool) -> None:
    points = df[[x_column, y_column]].to_numpy()
    coords = list(map(tuple, np.asarray(points)))
    poly = Polygon(coords)
    x, y = shapely.concave_hull(poly, ratio=ratio, allow_holes=holes).exterior.xy

    p = figure(
		x_axis_label=x_column,
		y_axis_label=y_column,
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
    p.line(
        x=x,
        y=y,
        color='black'
    )
    p.toolbar.logo = None

    st.bokeh_chart(p)

    hull_df = pd.DataFrame({'Hull X Coordinates' : x, 'Hull Y Coordinates': y})
    st.download_button(
        label='Download Hull Coordinates',
        data=hull_df.to_csv(index=False).encode('utf-8'),
        file_name='Hull Coordinates.csv',
        mime='text/csv'
    )

def plot_contour(df: pd.DataFrame, x_column: str, y_column: str, z_column: str, ratio: float, holes: bool, palette: list, log_scale: bool, small_palette_num: int) -> None:
    points = df[[x_column, y_column]].to_numpy()
    coords = list(map(tuple, np.asarray(points)))
    poly = Polygon(coords)
    x, y = shapely.concave_hull(poly, ratio=ratio, allow_holes=holes).exterior.xy

    if small_palette_num:
        if palette == 'Cividis':
            palette = Cividis[small_palette_num]
        elif palette == 'Inferno':
            palette = Inferno[small_palette_num]
        elif palette == 'Magma':
            palette = Magma[small_palette_num]
        elif palette == 'Plasma':
            palette = Plasma[small_palette_num]
        elif palette == 'Viridis':
            palette = Viridis[small_palette_num]
        elif palette == 'Turbo':
            palette = Turbo[small_palette_num]
        elif palette == 'Grey':
            palette = Greys[small_palette_num]   
    else:
        if palette == 'Cividis':
            palette = Cividis256
        elif palette == 'Inferno':
            palette = Inferno256
        elif palette == 'Magma':
            palette = Magma256
        elif palette == 'Plasma':
            palette = Plasma256
        elif palette == 'Viridis':
            palette = Viridis256
        elif palette == 'Turbo':
            palette = Turbo256
        elif palette == 'Grey':
            palette = Greys256        
        
    # Define a color mapper based on the chosen palette and the range of z values
    if log_scale:
        mapper = log_cmap(field_name=z_column, palette=palette, low=min(df[z_column]), high=max(df[z_column]))
    else:
        mapper = linear_cmap(field_name=z_column, palette=palette, low=min(df[z_column]), high=max(df[z_column]))

    p = figure(
		x_axis_label=x_column,
		y_axis_label=y_column,
		aspect_ratio = 17.5 / 8,
		sizing_mode='scale_width',
		output_backend='webgl',
	)
    source = ColumnDataSource(df)
    p.circle(
        source=source,
        x=x_column,
        y=y_column,
        color=mapper
    )
    p.line(
        x=x,
        y=y,
        color='black'
    )

    if small_palette_num and not log_scale:
        n_ticks = np.linspace(df[z_column].min(), df[z_column].max(), small_palette_num + 1).round()
        color_bar = ColorBar(
            color_mapper=mapper['transform'],
            width=8,
            location=(0, 0),
            ticker=FixedTicker(ticks=n_ticks)
        )
    elif small_palette_num and log_scale:
        n_ticks = np.logspace(math.log(df[z_column].min()) / math.log(10), math.log(df[z_column].max()) / math.log(10), small_palette_num + 1).round()
        st.write(n_ticks)
        color_bar = ColorBar(
            color_mapper=mapper['transform'],
            width=8,
            location=(0, 0),
            ticker=FixedTicker(ticks=n_ticks)
        )
    else:
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))

    p.add_layout(color_bar, 'right')
    
    t = f'''
    <div @{x_column}\u007bcustom\u007d>
        <b>{x_column}: </b> @{x_column}<br>
        <b>{y_column}: </b> @{y_column}<br>
        <b>{z_column}: </b> @{z_column} 
    </div>
    '''
    num = 1
    f = CustomJSHover(
        code = f'''
        special_vars.indices = special_vars.indices.slice(0, {num})
        return special_vars.indices.includes(special_vars.index) ? " " : " hidden "
        ''')
    p.add_tools(HoverTool(tooltips = t, formatters = {f'@{x_column}': f}))

    p.toolbar.logo = None

    st.bokeh_chart(p)

    hull_df = pd.DataFrame({'Hull X Coordinates' : x, 'Hull Y Coordinates': y})
    st.download_button(
        label='Download Hull Coordinates',
        data=hull_df.to_csv(index=False).encode('utf-8'),
        file_name='Hull Coordinates.csv',
        mime='text/csv'
    )

def main() -> None:
    st.header('Convex Hull Graphing Tool')

    st.info('NOTE: This tool assumes your data is clean.')
    
    file = st.file_uploader(
        label='Upload your data',
        type=['csv', 'xlsx']
    )

    if file is None:
        st.info('Please upload some data to get started')
        return
    
    df = read_file(file)
    st.write(df.head())

    data_columns, additional_column, color_column = st.columns(3)
    with data_columns:
        x_column, y_column = get_plot_columns(df)
        sensitivity = st.number_input(
            label='Sensitivity level',
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            format='%.4f'
        )
        holes = st.checkbox(
            label='Allow holes?',
            value=False
        )
    with additional_column:
        z_column = None
        if st.checkbox('Add a third column for color?'):
            z_column, palette = get_additional_column(df)

    with color_column:
        if z_column is not None:
            small_palette_num = None
            log_scale = st.checkbox('Use Log Scale for Z Column?')
            continuous_palette = st.checkbox('Use a Continuous palette? (Smooth color transition)', value=True)
            if not continuous_palette:
                small_palette_num = st.number_input(
                    label='How many unique colors?',
                    min_value=3,
                    value=8,
                    max_value=11
                )

    try:
        if x_column == y_column:
            st.warning('Your X and Y columns must be different')
            return
        
        if z_column is None:
            plot(df, x_column, y_column, sensitivity, holes)
        else:
            plot_contour(df, x_column, y_column, z_column, sensitivity, holes, palette, log_scale, small_palette_num)
    except:
        st.warning("We're sorry, something went wrong. Please make sure your data is clean")
        

if __name__ == '__main__':
    main()