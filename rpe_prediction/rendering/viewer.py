import vtk

COLORS = ["red", "green", "blue"]


class SkeletonViewer(object):

    def __init__(self):
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
        self._skeleton_objects = []
        self._max_frames = 1e10
        self.__cur_frame = 0
        self.__break = False

    def add_skeleton(self, data, connections=None, radius: float = 0.02):
        """
        Add a new skeleton to the renderer
        @param data: new skeleton data in a numpy array
        @param connections: the connections of skeletons
        @param radius: radius size of markers
        @return: None
        """
        actors_markers = []  # each marker has an own actor
        actors_bones = []  # actors for each line segment between two markers
        lines = []
        rows, cols = data.shape
        if rows < self._max_frames:
            self._max_frames = rows  # set max size to smallest video length

        # Create all instances for all markers
        for marker in range(cols // 3):
            sphere = vtk.vtkSphereSource()
            sphere.SetPhiResolution(100)
            sphere.SetThetaResolution(100)
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(radius)
            mapper = vtk.vtkPolyDataMapper()
            mapper.AddInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self._skeleton_objects)]))
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
                actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self._skeleton_objects)]))
                self.renderer.AddActor(actor)
                actors_bones.append(actor)

        self._skeleton_objects.append({
            'data': data,
            'connections': connections,
            'lines': lines,
            'actors_markers': actors_markers,
        })

    def show_window(self, scale:float =0.5):
        # Initialize a timer for the animation
        self.render_window_interactor.AddObserver('TimerEvent', self.update)

        # create coordinate actor
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(scale, scale, scale)
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.renderer.AddActor(axes)

        # Begin Interaction
        self.render_window.SetSize(2560, 1520)
        self.render_window.Render()
        self.render_window_interactor.Start()

    def update(self, obj, event):
        if self.__cur_frame >= self._max_frames:
            self.__cur_frame = 0

        # Draw individual skeleton
        for skeleton_data in self._skeleton_objects:
            # Update marker position for current frame
            data = skeleton_data['data']
            actors_markers = skeleton_data['actors_markers']
            points = data[self.__cur_frame].reshape(-1, 3)

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
        print(key, 'was pressed')
        if key == 'space':
            self.__break = not self.__break
        elif key == 'Left':
            new_frame = self.__cur_frame - 1
            self.__cur_frame = new_frame if new_frame > 0 else self.__cur_frame
        elif key == 'Right':
            new_frame = self.__cur_frame + 1
            self.__cur_frame = new_frame if new_frame < self._max_frames else self.__cur_frame
