import vtk
import pandas as pd
import numpy as np

COLORS = ["red", "green", "blue"]


class SkeletonViewer(object):

    def __init__(self, sphere_radius: float = 0.01):
        """
        Constructor for Skeleton Viewer using VTK graphics framework
        @param sphere_radius: the radius of spheres
        """
        self.colors = vtk.vtkNamedColors()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.renderer.ResetCamera()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)
        self.render_window_interactor.Initialize()
        self.timer_id = self.render_window_interactor.CreateRepeatingTimer(33)
        self.render_window_interactor.AddObserver('KeyPressEvent', self.keypress_callback, 1.0)

        # Data objects
        self.__skeleton_objects = []
        self.__max_frames = float('inf')
        self.__cur_frame = 0
        self.__break = False
        self.__trans_vector = np.array([0, 0, 0])
        self.__scale_factor = 1.0
        self._sphere_radius = sphere_radius

    def add_skeleton(self, data: pd.DataFrame, connections=None):
        """
        Add a new skeleton to the renderer
        @param data: new skeleton data in a numpy array
        @param connections: the connections of skeletons
        @return: None
        """
        actors_markers = []  # each marker has an own actor
        actors_bones = []  # actors for each line segment between two markers
        lines = []
        _, cols = data.shape

        # Create all instances for all markers
        for marker in range(cols // 3):
            sphere = vtk.vtkSphereSource()
            sphere.SetPhiResolution(100)
            sphere.SetThetaResolution(100)
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(self._sphere_radius)
            mapper = vtk.vtkPolyDataMapper()
            mapper.AddInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self.__skeleton_objects)]))
            self.renderer.AddActor(actor)
            actors_markers.append(actor)

        if connections is not None:
            for _ in connections:
                line = vtk.vtkLineSource()
                line.SetPoint1(0, 0, 0)
                line.SetPoint2(0, 0, 0)
                lines.append(line)
                # Setup actor and mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.AddInputConnection(line.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self.__skeleton_objects)]))
                self.renderer.AddActor(actor)
                actors_bones.append(actor)

        # Invert y-coordinate axis for Azure Kinect data
        data[data.filter(like='(y)').columns] *= -1

        self.__skeleton_objects.append({
            'data': data.to_numpy(),
            'connections': connections,
            'lines': lines,
            'actors_markers': actors_markers,
        })

    def show_window(self, axis_scale: float = 0.3):
        """
        Start the rendering sequence
        @param axis_scale: scaling factor for coordinate system axes
        @return: None
        """
        # Initialize a timer for the animation
        self.render_window_interactor.AddObserver('TimerEvent', self.update)
        self.__max_frames = min(map(lambda x: len(x['data']), self.__skeleton_objects))

        # Calculate bounding box and scale to center
        data = np.concatenate(list(map(lambda x: x['data'].reshape(-1, 3), self.__skeleton_objects)))
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.__trans_vector = np.array([(min_vals[0] + max_vals[0]) / 2, min_vals[1], (min_vals[2] + max_vals[2]) / 2])
        self.__scale_factor = np.max(data - self.__trans_vector)

        # create coordinate actor
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(axis_scale, axis_scale, axis_scale)
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.renderer.AddActor(axes)

        # Begin Interaction
        self.render_window.SetSize(2560, 1520)
        self.render_window.Render()
        self.render_window_interactor.Start()

    def update(self, obj, event):
        if self.__cur_frame >= self.__max_frames:
            self.__cur_frame = 0

        # Draw individual skeleton
        for skeleton_data in self.__skeleton_objects:
            # Update marker position for current frame
            data = skeleton_data['data']
            actors_markers = skeleton_data['actors_markers']
            points = (data[self.__cur_frame].reshape(-1, 3) - self.__trans_vector) / self.__scale_factor

            for c_points, actor in enumerate(actors_markers):
                x, y, z = points[c_points]
                actor.SetPosition(x, y, z)

            # Update bone connections
            bones = skeleton_data['connections']
            lines = skeleton_data['lines']
            if bones is None:
                continue

            for line, (j1, j2) in zip(lines, bones):
                line.SetPoint1(points[j1])
                line.SetPoint2(points[j2])

        obj.GetRenderWindow().Render()
        if not self.__break:
            self.__cur_frame += 1

    def keypress_callback(self, obj, ev):
        key = obj.GetKeySym()
        if key == 'space':
            self.__break = not self.__break
        elif key == 'Left':
            new_frame = self.__cur_frame - 1
            self.__cur_frame = new_frame if new_frame > 0 else self.__cur_frame
            print(f"Current Frame: {self.__cur_frame}")
        elif key == 'Right':
            new_frame = self.__cur_frame + 1
            self.__cur_frame = new_frame if new_frame < self.__max_frames else self.__cur_frame
            print(f"Current Frame: {self.__cur_frame}")
