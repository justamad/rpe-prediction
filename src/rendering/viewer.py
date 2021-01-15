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

        # Data objects
        self.skeleton_objects = []
        self.max_iterations = 1e10
        self.iterations = 0

    def add_skeleton(self, data, connections=None, radius=0.02):
        actors_markers = []  # each marker has an own actor
        actors_bones = []  # actors for each line segment between two markers
        lines = []
        rows, cols = data.shape
        if rows < self.max_iterations:
            self.max_iterations = rows  # set max size to smallest video length

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
            actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self.skeleton_objects)]))
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
                actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self.skeleton_objects)]))
                self.renderer.AddActor(actor)
                actors_bones.append(actor)

        self.skeleton_objects.append({
            'data': data,
            'connections': connections,
            "lines": lines,
            'actors_markers': actors_markers,
        })

    def show_window(self, scale=0.5):
        # Initialize a timer for the animation
        self.render_window_interactor.AddObserver('TimerEvent', self.update)
        self.timer_id = self.render_window_interactor.CreateRepeatingTimer(10)

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
        if self.iterations >= self.max_iterations:
            # obj.DestroyTimer(self.timer_id)
            self.iterations = 0

        # Draw individual skeleton
        for skeleton_data in self.skeleton_objects:
            # Update marker position for current frame
            data = skeleton_data['data']
            actors_markers = skeleton_data['actors_markers']
            points = data[self.iterations].reshape(-1, 3)

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
        self.iterations += 1
