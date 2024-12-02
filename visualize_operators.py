import plotly.graph_objects as go
import operators

def visualize_join(v1, v2):
    u1, u2 = operators.join(v1, v2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]], 
                               mode='lines', line=dict(color='red', width=5), name='v1'))
    fig.add_trace(go.Scatter3d(x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]], 
                               mode='lines', line=dict(color='green', width=5), name='v2'))
    fig.add_trace(go.Scatter3d(x=[0, u1[0]], y=[0, u1[1]], z=[0, u1[2]], 
                               mode='lines', line=dict(color='blue', width=5), name='u1 (Join)'))
    fig.add_trace(go.Scatter3d(x=[0, u2[0]], y=[0, u2[1]], z=[0, u2[2]], 
                               mode='lines', line=dict(color='purple', width=5), name='u2 (Join)'))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title="Join operator"
    )

    fig.show()

def visualize_meet(v1, v2, v3, v4):
    intersection = operators.meet(v1, v2, v3, v4)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]], 
                               mode='lines', line=dict(color='red', width=5), name='v1 (Subspace 1)'))
    fig.add_trace(go.Scatter3d(x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]], 
                               mode='lines', line=dict(color='green', width=5), name='v2 (Subspace 1)'))
    fig.add_trace(go.Scatter3d(x=[0, v3[0]], y=[0, v3[1]], z=[0, v3[2]], 
                               mode='lines', line=dict(color='cyan', width=5), name='v3 (Subspace 2)'))
    fig.add_trace(go.Scatter3d(x=[0, v4[0]], y=[0, v4[1]], z=[0, v4[2]], 
                               mode='lines', line=dict(color='magenta', width=5), name='v4 (Subspace 2)'))
    fig.add_trace(go.Scatter3d(x=[0, intersection[0]], y=[0, intersection[1]], z=[0, intersection[2]], 
                               mode='lines', line=dict(color='yellow', width=5), name='Intersection (Meet)'))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title="Meet operator"
    )

    fig.show()

v1 = [0.8, 1, 0.2]
v2 = [0.5, 1, 1.3]
v3 = [0, 0.1, 0.9]
v4 = [1, 1, 0.4]

visualize_join(v1, v2)

visualize_meet(v1, v2, v3, v4)
