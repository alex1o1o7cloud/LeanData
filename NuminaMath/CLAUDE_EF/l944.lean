import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l944_94425

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2 * Real.log x + a * x^2 + b * x

-- Define the function g
noncomputable def g (b x : ℝ) : ℝ := Real.exp (x - 1) + (1/2) * x * f 0 b x

-- Theorem for part (1)
theorem part_one (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) 
  (h₃ : f 1 1 x₁ + f 1 1 x₂ = 4) : x₁ + x₂ ≥ 2 := by
  sorry

-- Theorem for part (2)
theorem part_two (b : ℝ) (h : ∀ x > 0, g b x ≥ 0) : b ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l944_94425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l944_94413

/-- The probability of selecting three plates of the same color -/
theorem same_color_probability (red blue green : ℕ) 
  (h_red : red = 7) 
  (h_blue : blue = 5) 
  (h_green : green = 3) : 
  (Nat.choose red 3 + Nat.choose blue 3 + Nat.choose green 3 : ℚ) / Nat.choose (red + blue + green) 3 = 46 / 455 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l944_94413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_theorem_l944_94483

-- Define the type for a pile of tokens
structure Pile where
  tokens : ℕ

-- Define the type for the game state
structure GameState where
  piles : List Pile

-- Define the allowed operations
inductive Operation
  | split : Pile → Pile → Pile → Operation
  | merge : Pile → Pile → Pile → Operation

-- Define a function to get the nth prime (simplified for this example)
def nthPrime (n : ℕ) : ℕ :=
  sorry -- We'll use sorry here as implementing nthPrime is complex

-- Define the initial state
def initialState : GameState :=
  { piles := List.map (λ i => Pile.mk (nthPrime (i + 1))) (List.range 2018) }

-- Define the target state
def targetState : GameState :=
  { piles := List.replicate 2018 (Pile.mk 2018) }

-- Define the transition function
def transition (state : GameState) (op : Operation) : GameState :=
  sorry

-- Define the theorem
theorem impossibility_theorem :
  ¬ ∃ (ops : List Operation), 
    (List.foldl transition initialState ops) = targetState := by
  sorry

#check impossibility_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_theorem_l944_94483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COD_area_l944_94468

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: The area of triangle COD is 5|p| -/
theorem triangle_COD_area (p : ℝ) :
  triangleArea 0 0 0 p 10 10 = 5 * abs p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_COD_area_l944_94468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_surface_area_l944_94446

/-- The surface area of a sphere with diameter 8.5 inches is 289π/4 square inches. -/
theorem bowling_ball_surface_area :
  let diameter : ℝ := 17/2  -- 8.5 inches expressed as a fraction
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * (radius ^ 2)
  surface_area = 289 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_surface_area_l944_94446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l944_94487

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

/-- The function g(x) defined in the problem -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (2^x + 1) * (1 - 2 / (2^x + 1)) + k

theorem problem_solution (a k : ℝ) :
  (a > 0 ∧ a ≠ 1 ∧ f a 0 = 0) →
  (a = 2 ∧ (∃ x : ℝ, g k x = 0) → k < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l944_94487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l944_94462

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := 1 - Real.sqrt x

-- State the theorem
theorem inverse_function_theorem :
  (∀ x ≤ 1, f x = (x - 1)^2) →
  (∀ x ≥ 0, f_inv x = 1 - Real.sqrt x) →
  (∀ x ≤ 1, f_inv (f x) = x) ∧
  (∀ x ≥ 0, f (f_inv x) = x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l944_94462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt34_l944_94474

/-- The speed of a particle with position (3t + 5, 5t - 9) at time t -/
noncomputable def particle_speed : ℝ := Real.sqrt 34

/-- The position of the particle at time t -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 5, 5 * t - 9)

theorem particle_speed_is_sqrt34 (t : ℝ) :
  let p₁ := particle_position t
  let p₂ := particle_position (t + 1)
  let dx := p₂.1 - p₁.1
  let dy := p₂.2 - p₁.2
  Real.sqrt (dx ^ 2 + dy ^ 2) = particle_speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt34_l944_94474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l944_94477

theorem distance_between_points : 
  let p1 : Fin 3 → ℝ := ![3, 2, -5]
  let p2 : Fin 3 → ℝ := ![7, 5, -1]
  Real.sqrt ((p2 0 - p1 0)^2 + (p2 1 - p1 1)^2 + (p2 2 - p1 2)^2) = Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l944_94477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_don_arrival_time_l944_94465

/-- The time it takes Don to reach the hospital -/
noncomputable def dons_time (ambulance_speed : ℝ) (don_speed : ℝ) (ambulance_time : ℝ) : ℝ :=
  (ambulance_speed * ambulance_time) / don_speed

/-- Theorem stating that Don takes 30 minutes to reach the hospital -/
theorem don_arrival_time :
  let ambulance_speed : ℝ := 60
  let don_speed : ℝ := 30
  let ambulance_time : ℝ := 15 / 60  -- 15 minutes converted to hours
  dons_time ambulance_speed don_speed ambulance_time = 0.5 := by
  sorry

/-- Evaluating Don's arrival time -/
def eval_dons_time : ℚ := 
  (60 : ℚ) * (15 / 60 : ℚ) / (30 : ℚ)

#eval eval_dons_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_don_arrival_time_l944_94465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l944_94489

noncomputable def y (x : ℝ) : ℝ := Real.arccos ((2 * x - 1) / Real.sqrt 3)

theorem y_derivative (x : ℝ) : 
  deriv y x = -2 / Real.sqrt (3 - (2 * x - 1)^2) :=
by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l944_94489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_hiding_number_l944_94431

def hides (a b : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    digits.all (· < 10) ∧ 
    (digits.foldl (fun acc d ↦ acc * 10 + d) 0 = a) ∧
    ∃ (subdigits : List ℕ), subdigits ⊆ digits ∧ 
      (subdigits.foldl (fun acc d ↦ acc * 10 + d) 0 = b)

theorem smallest_hiding_number : 
  (∀ n : ℕ, n < 1201201 → ¬(hides n 2021 ∧ hides n 2120 ∧ hides n 1220 ∧ hides n 1202)) ∧
  (hides 1201201 2021 ∧ hides 1201201 2120 ∧ hides 1201201 1220 ∧ hides 1201201 1202) := by
  sorry

#check smallest_hiding_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_hiding_number_l944_94431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l944_94471

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- Theorem: The area of triangle ABC is 1 -/
theorem triangle_ABC_area :
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (0, 1)
  let C : ℝ × ℝ := (2, 1)
  triangle_area A B C = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l944_94471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_is_five_eighteenths_l944_94459

/-- The sum of the infinite series Σ(1/(n(n+3))) for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, 1 / (n * (n + 3))

/-- Theorem stating that the sum of the infinite series is equal to 5/18 -/
theorem infinite_series_sum_is_five_eighteenths : infinite_series_sum = 5 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_is_five_eighteenths_l944_94459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_theorem_l944_94486

/-- Represents a tank with two pipes. -/
structure Tank where
  fill_time_A : ℚ  -- Time for pipe A to fill the tank
  empty_time_B : ℚ  -- Time for pipe B to empty the tank

/-- Calculates the time to fill the tank when both pipes work simultaneously. -/
def simultaneous_fill_time (t : Tank) : ℚ :=
  1 / (1 / t.fill_time_A - 1 / t.empty_time_B)

/-- Theorem: For a tank where pipe A fills in 9 minutes and pipe B empties in 18 minutes,
    the simultaneous fill time is 18 minutes. -/
theorem simultaneous_fill_theorem (t : Tank) 
    (h1 : t.fill_time_A = 9) 
    (h2 : t.empty_time_B = 18) : 
    simultaneous_fill_time t = 18 := by
  sorry

#check simultaneous_fill_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_theorem_l944_94486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_transformed_function_l944_94467

noncomputable def original_function (x : ℝ) : ℝ := (1/2) * x^2 + 3*x + 5/2

noncomputable def transformed_function (x : ℝ) : ℝ := original_function (x - 2) + 3

theorem vertex_of_transformed_function :
  let vertex := (λ (f : ℝ → ℝ) => 
    let a := (f 1 - 2*f 0 + f (-1)) / 2
    let b := (f 1 - f (-1)) / 2
    let c := f 0
    (-b / (2*a), c - b^2 / (4*a)))
  vertex transformed_function = (-1, 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_transformed_function_l944_94467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_500_l944_94484

/-- The speed of an empty plane in MPH -/
def emptyPlaneSpeed : ℚ := 600

/-- The speed reduction per passenger in MPH -/
def speedReductionPerPassenger : ℚ := 2

/-- The number of passengers on the first plane -/
def passengersPlane1 : ℕ := 50

/-- The number of passengers on the second plane -/
def passengersPlane2 : ℕ := 60

/-- The number of passengers on the third plane -/
def passengersPlane3 : ℕ := 40

/-- The speed of a plane given the number of passengers -/
def planeSpeed (passengers : ℕ) : ℚ :=
  emptyPlaneSpeed - (speedReductionPerPassenger * passengers)

/-- The average speed of the three planes -/
def averageSpeed : ℚ :=
  (planeSpeed passengersPlane1 + planeSpeed passengersPlane2 + planeSpeed passengersPlane3) / 3

theorem average_speed_is_500 : averageSpeed = 500 := by
  -- Unfold the definitions
  unfold averageSpeed planeSpeed emptyPlaneSpeed speedReductionPerPassenger
  unfold passengersPlane1 passengersPlane2 passengersPlane3
  -- Simplify the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_500_l944_94484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l944_94427

noncomputable def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 20 = 1

def are_foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 / 8 - b^2 / 10 = 1 ∧
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4 * (a^2 + b^2)

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem hyperbola_focal_distance 
  (P F₁ F₂ : ℝ × ℝ) 
  (h₁ : is_on_hyperbola P.1 P.2) 
  (h₂ : are_foci F₁ F₂) 
  (h₃ : distance P F₁ = 9) : 
  distance P F₂ = 17 := by
  sorry

#check hyperbola_focal_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l944_94427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_5R_squared_l944_94414

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  R : ℝ  -- Radius of the inscribed circle
  h : ℝ  -- Height of the trapezoid
  upper_base : ℝ  -- Length of the upper base
  lower_base : ℝ  -- Length of the lower base
  h_eq : h = 2 * R  -- Height is twice the radius
  upper_base_eq : upper_base = h / 2  -- Upper base is half the height

/-- The area of the trapezoid -/
noncomputable def trapezoid_area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  (t.upper_base + t.lower_base) * t.h / 2

/-- Theorem: The area of the trapezoid is 5R^2 -/
theorem trapezoid_area_is_5R_squared (t : IsoscelesTrapezoidWithInscribedCircle) :
  trapezoid_area t = 5 * t.R^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_5R_squared_l944_94414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_trig_functions_l944_94403

theorem symmetry_of_trig_functions (a : ℝ) :
  (∀ x, Real.sin (2*x - π/3) = Real.sin (2*(2*a - x) - π/3)) ∧
  (∀ x, Real.cos (2*x + 2*π/3) = Real.cos (2*x + 5*π/6 - 4*a)) →
  a = π/24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_trig_functions_l944_94403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l944_94404

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix intersection with x-axis
def directrix_intersection : ℝ × ℝ := (-1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem parabola_triangle_area (P : PointOnParabola) :
  distance (P.x, P.y) focus = 5 →
  triangle_area (P.x, P.y) focus directrix_intersection = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l944_94404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_theorem_l944_94466

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line
def line_equation (x y a : ℝ) : Prop := y = x + a

-- Define the chord length condition
def chord_length_is_two (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ,
  circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
  line_equation x₁ y₁ a ∧ line_equation x₂ y₂ a ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4

-- The theorem to prove
theorem chord_intercept_theorem (a : ℝ) :
  chord_length_is_two a → a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_theorem_l944_94466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_connecting_circle_centers_l944_94423

/-- The equation of the line connecting the centers of two circles -/
theorem line_connecting_circle_centers 
  (circle1 circle2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, circle1 x y ↔ x^2 + y^2 - 4*x + 6*y = 0)
  (h2 : ∀ x y, circle2 x y ↔ x^2 + y^2 - 6*x = 0) :
  ∃ A B C : ℝ, 
    (A ≠ 0 ∨ B ≠ 0) ∧ 
    (∀ x y, (circle1 x y ∨ circle2 x y) → A*x + B*y + C = 0) ∧
    A = 3 ∧ B = -1 ∧ C = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_connecting_circle_centers_l944_94423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_problem_l944_94488

theorem smallest_divisor_problem (n : ℕ) (h1 : n = 1011) 
  (h2 : ∀ d ∈ ({16, 18, 21, 28} : Finset ℕ), (n - 3) % d = 0) : 
  16 = (Finset.filter (λ d => (n - 3) % d = 0) {16, 18, 21, 28}).min := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_problem_l944_94488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l944_94434

/-- The circle x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The line x + √3y - 4 = 0 -/
def line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 4 = 0

/-- Distance from a point (x, y) to the line x + √3y - 4 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + Real.sqrt 3 * y - 4) / (Real.sqrt 4)

theorem min_distance_circle_to_line :
  ∃ d : ℝ, d = 1 ∧
    (∀ x y : ℝ, unit_circle x y → distance_to_line x y ≥ d) ∧
    (∃ x y : ℝ, unit_circle x y ∧ distance_to_line x y = d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l944_94434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_bound_and_maximum_l944_94458

/-- A circle centered at (0,1) intersecting the parabola y = x^2 at four points -/
structure IntersectionConfiguration where
  R : ℝ
  h : (Real.sqrt 3)/2 < R ∧ R < 1

/-- The area of the quadrilateral formed by the four intersection points -/
def quadrilateralArea (config : IntersectionConfiguration) : ℝ :=
  sorry

theorem quadrilateral_area_bound_and_maximum (config : IntersectionConfiguration) :
  quadrilateralArea config < Real.sqrt 2 ∧
  ∃ (maxConfig : IntersectionConfiguration),
    ∀ (c : IntersectionConfiguration), quadrilateralArea c ≤ quadrilateralArea maxConfig ∧
    quadrilateralArea maxConfig = (4/3) * Real.sqrt (2/3) := by
  sorry

#check quadrilateral_area_bound_and_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_bound_and_maximum_l944_94458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_radius_l944_94416

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Define membership for a point in a circle -/
def Point.mem (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

instance : Membership Point Circle where
  mem := Point.mem

theorem circle_intersection_radius (C1 C2 : Circle) (O X Y Z : Point) :
  C1.center = O →
  C2.center ≠ O →
  O ∈ C2 →
  Z ∈ C2 →
  Z ∉ C1 →
  X ∈ C1 →
  Y ∈ C1 →
  X ∈ C2 →
  Y ∈ C2 →
  distance X Z = 12 →
  distance O Z = 16 →
  distance Y Z = 9 →
  C1.radius = 4 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_radius_l944_94416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_when_s_is_6_sqrt_2_l944_94493

/-- A solid with an equilateral triangular base and parallel upper edge -/
structure TriangularSolid where
  s : ℝ  -- Side length of the base
  base_equilateral : True  -- Assumption that the base is equilateral
  upper_edge_parallel : True  -- Assumption that the upper edge is parallel to the base
  upper_edge_length : True  -- Assumption that the upper edge has length 2s
  side_edge_length : True  -- Assumption that the side edges have length s

/-- The volume of the triangular solid -/
noncomputable def volume (solid : TriangularSolid) : ℝ :=
  (solid.s^3 * Real.sqrt 3) / 12

/-- Theorem stating the volume of the solid when s = 6√2 -/
theorem volume_when_s_is_6_sqrt_2 (solid : TriangularSolid) (h : solid.s = 6 * Real.sqrt 2) :
  volume solid = 108 := by
  sorry

#check volume_when_s_is_6_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_when_s_is_6_sqrt_2_l944_94493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l944_94426

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

def sequenceValue (m : ℕ) : ℕ := Q + m

def is_in_sequence (n : ℕ) : Prop :=
  ∃ m, 3 ≤ m ∧ m ≤ 55 ∧ sequenceValue m = n

theorem no_primes_in_sequence :
  ∀ n, is_in_sequence n → ¬ Nat.Prime n :=
by
  intro n hn
  sorry

#check no_primes_in_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l944_94426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_properties_l944_94481

theorem negation_properties :
  (∀ x : ℝ, -(-x) = x) ∧ 
  (∀ x : ℝ, -abs x = -max x (-x)) ∧ 
  (∀ x : ℝ, -(-(x^2)) = x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_properties_l944_94481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_a_locus_b_l944_94485

noncomputable section

-- Define the equilateral triangle
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the distance squared function
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Theorem for part (a)
theorem locus_a (x y : ℝ) :
  let P := (x, y)
  dist_squared P A + dist_squared P B = dist_squared P C ↔
  x^2 + (y + Real.sqrt 3)^2 = 4 := by sorry

-- Theorem for part (b)
theorem locus_b (x y : ℝ) :
  let P := (x, y)
  dist_squared P A + dist_squared P B = 2 * dist_squared P C ↔
  y = Real.sqrt 3 / 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_a_locus_b_l944_94485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_2x_over_cos_plus_sin_l944_94473

open Real

theorem integral_cos_2x_over_cos_plus_sin :
  ∫ x in (0)..(π/4), (cos (2*x)) / (cos x + sin x) = sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_2x_over_cos_plus_sin_l944_94473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_transformation_l944_94450

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

theorem function_properties_and_transformation (a b : ℝ) (h_a : a ≠ 0) :
  (∀ x, f a b x ≤ 2) ∧ 
  (deriv (f a b) (π / 6) = 1) →
  a = Real.sqrt 3 ∧ b = 1 ∧
  ∃ α : ℝ, α ∈ Set.Ioo (π / 6) (π / 2) ∧
    2 * Real.sin (2 * (α + π / 4) - π / 6) = 10 / 13 ∧
    Real.cos (2 * α) = (5 * Real.sqrt 3 - 12) / 26 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_transformation_l944_94450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_midpoint_l944_94409

/-- The ellipse Γ -/
def Γ : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The point P -/
def P : ℝ × ℝ := (1, 1)

/-- A line passing through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- The theorem statement -/
theorem ellipse_chord_midpoint :
  ∀ A B : ℝ × ℝ,
  A ∈ Γ →
  B ∈ Γ →
  P ∈ Line A B →
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = P →
  ∀ x y : ℝ,
  (x, y) ∈ Line A B ↔ 4 * y + 3 * x - 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_midpoint_l944_94409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_long_segment_in_U_l944_94480

/-- The set of points between two parabolas y = x^2 and y = x^2 - 1, including points on the parabolas -/
def U : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ p.1^2 ∧ p.2 ≥ p.1^2 - 1}

/-- A line segment between two points -/
def LineSegment (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = (1 - t) • p + t • q}

/-- The length of a line segment -/
noncomputable def SegmentLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating that U contains a line segment longer than 10^6 -/
theorem exists_long_segment_in_U :
  ∃ p q : ℝ × ℝ, p ∈ U ∧ q ∈ U ∧ (LineSegment p q) ⊆ U ∧ SegmentLength p q > 1000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_long_segment_in_U_l944_94480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l944_94445

/-- The equation of the tangent line to a circle at point P(1,2) -/
theorem tangent_line_equation (P : ℝ × ℝ) (center : ℝ × ℝ) :
  P = (1, 2) →
  center = (0, 0) →
  ‖P - center‖ = ‖P‖ →
  ∃ (A B C : ℝ), A * P.1 + B * P.2 + C = 0 ∧
                 A = 1 ∧ B = 2 ∧ C = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l944_94445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l944_94463

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Theorem statement
theorem problem_statement :
  (∃ (S : Set ℝ), S = {x : ℝ | f x ≤ 1} ∧ S = Set.Ici (-1)) ∧
  (∃ (m : ℝ), m = 3 ∧ ∀ x, f x ≤ m) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 3 → 3/a + a/b ≥ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l944_94463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_62_acquaintances_not_necessarily_63_acquaintances_l944_94472

/-- Represents a group of people with acquaintance relationships -/
structure AcquaintanceGroup where
  people : Finset ℕ
  knows : ℕ → ℕ → Prop
  acquaintance_count : ℕ → ℕ

/-- Properties of the AcquaintanceGroup -/
axiom acquaintance_group_properties (G : AcquaintanceGroup) :
  (∀ x y, x ∈ G.people → y ∈ G.people → G.knows x y → G.acquaintance_count x = G.acquaintance_count y) ∧
  (∀ x y, x ∈ G.people → y ∈ G.people → ¬G.knows x y → G.acquaintance_count x ≠ G.acquaintance_count y)

/-- The main theorem to be proved -/
theorem exists_62_acquaintances (G : AcquaintanceGroup) 
  (h : G.people.card = 1995) :
  ∃ x, x ∈ G.people ∧ G.acquaintance_count x ≥ 62 := by
  sorry

/-- The statement is not necessarily true for 63 acquaintances -/
theorem not_necessarily_63_acquaintances : 
  ∃ G : AcquaintanceGroup, G.people.card = 1995 ∧ ∀ x ∈ G.people, G.acquaintance_count x ≤ 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_62_acquaintances_not_necessarily_63_acquaintances_l944_94472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_results_l944_94408

-- Define the bags and their contents
def bag_A : Nat × Nat := (9, 1)  -- (red balls, white balls)
def bag_B : Nat × Nat := (2, 8)

-- Define the probability of choosing each bag initially
def initial_prob : ℚ := 1/2

-- Define the probability of drawing a red ball from each bag
def prob_red_A : ℚ := bag_A.1 / (bag_A.1 + bag_A.2)
def prob_red_B : ℚ := bag_B.1 / (bag_B.1 + bag_B.2)

theorem experiment_results :
  -- 1. Probability of drawing a red ball on the first trial
  (initial_prob * prob_red_A + initial_prob * prob_red_B = 11/20) ∧
  
  -- 2a. Probability of having chosen bag A given a white ball was drawn
  (let prob_white := 1 - (initial_prob * prob_red_A + initial_prob * prob_red_B);
   let prob_white_A := 1 - prob_red_A;
   (prob_white_A * initial_prob) / prob_white = 1/9) ∧
  
  -- 2b. Probability of drawing red in second trial is higher when switching bags
  (let prob_A_given_white := (1 - prob_red_A) * initial_prob / (1 - (initial_prob * prob_red_A + initial_prob * prob_red_B));
   let prob_B_given_white := 1 - prob_A_given_white;
   let prob_red_stay := prob_A_given_white * prob_red_A + prob_B_given_white * prob_red_B;
   let prob_red_switch := prob_B_given_white * prob_red_A + prob_A_given_white * prob_red_B;
   prob_red_switch > prob_red_stay) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_experiment_results_l944_94408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_pi_over_nine_l944_94415

/-- The probability that a randomly selected point from a square with vertices at (±3, ±3) is within 2 units of the origin -/
noncomputable def probability_within_circle (Q : ℝ × ℝ) : ℝ := by
  sorry

/-- The square region from which Q is selected -/
def square_region : Set (ℝ × ℝ) :=
  {p | -3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3}

/-- The circle region within 2 units of the origin -/
def circle_region : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 ≤ 4}

/-- The area of the square region -/
def square_area : ℝ := 36

/-- The area of the circle region -/
noncomputable def circle_area : ℝ := 4 * Real.pi

/-- The probability is equal to the ratio of the circle area to the square area -/
theorem probability_equals_pi_over_nine (Q : ℝ × ℝ) :
  probability_within_circle Q = circle_area / square_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_pi_over_nine_l944_94415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seat_five_occupied_l944_94444

-- Define the type for seats
inductive Seat : Type where
  | one | two | three | four | five | six | seven | eight | nine

-- Define the type for kids
inductive Kid : Type where
  | asya | borya | vasilina | grisha

-- Define a function to represent the seating arrangement
def seating : Kid → Seat := sorry

-- Define the conditions
axiom total_seats : ∀ s : Seat, s = Seat.one ∨ s = Seat.two ∨ s = Seat.three ∨ s = Seat.four ∨ 
                    s = Seat.five ∨ s = Seat.six ∨ s = Seat.seven ∨ s = Seat.eight ∨ s = Seat.nine

axiom borya_not_four_or_six : seating Kid.borya ≠ Seat.four ∧ seating Kid.borya ≠ Seat.six

axiom asya_next_to_others : 
  (seating Kid.asya = Seat.one ∧ seating Kid.vasilina = Seat.two ∧ seating Kid.grisha = Seat.three) ∨
  (seating Kid.asya = Seat.two ∧ seating Kid.vasilina = Seat.three ∧ seating Kid.grisha = Seat.four) ∨
  (seating Kid.asya = Seat.three ∧ seating Kid.vasilina = Seat.four ∧ seating Kid.grisha = Seat.five) ∨
  (seating Kid.asya = Seat.four ∧ seating Kid.vasilina = Seat.five ∧ seating Kid.grisha = Seat.six) ∨
  (seating Kid.asya = Seat.five ∧ seating Kid.vasilina = Seat.six ∧ seating Kid.grisha = Seat.seven) ∨
  (seating Kid.asya = Seat.six ∧ seating Kid.vasilina = Seat.seven ∧ seating Kid.grisha = Seat.eight) ∨
  (seating Kid.asya = Seat.seven ∧ seating Kid.vasilina = Seat.eight ∧ seating Kid.grisha = Seat.nine)

axiom nobody_next_to_borya : 
  (seating Kid.borya = Seat.one → seating Kid.asya ≠ Seat.two ∧ seating Kid.vasilina ≠ Seat.two ∧ seating Kid.grisha ≠ Seat.two) ∧
  (seating Kid.borya = Seat.nine → seating Kid.asya ≠ Seat.eight ∧ seating Kid.vasilina ≠ Seat.eight ∧ seating Kid.grisha ≠ Seat.eight) ∧
  (seating Kid.borya ≠ Seat.one ∧ seating Kid.borya ≠ Seat.nine → 
    (∀ k : Kid, seating k ≠ Seat.one ∧ seating k ≠ Seat.two ∧ seating k ≠ Seat.three) ∨
    (∀ k : Kid, seating k ≠ Seat.seven ∧ seating k ≠ Seat.eight ∧ seating k ≠ Seat.nine))

axiom max_two_seats_between : 
  (seating Kid.asya = Seat.one ∨ seating Kid.asya = Seat.two ∨ seating Kid.asya = Seat.three) →
    (seating Kid.borya = Seat.one ∨ seating Kid.borya = Seat.two ∨ seating Kid.borya = Seat.three ∨
     seating Kid.borya = Seat.four ∨ seating Kid.borya = Seat.five)

-- The theorem to prove
theorem seat_five_occupied : 
  ∃ k : Kid, seating k = Seat.five := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seat_five_occupied_l944_94444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_velocity_l944_94497

/-- A particle moves according to s = t³. Its instantaneous velocity at t = 3s is 27. -/
theorem particle_velocity :
  let s : ℝ → ℝ := fun t => t^3
  let v := fun t => deriv s t
  v 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_velocity_l944_94497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_lower_bound_l944_94432

/-- The blackboard operation that replaces two numbers with their average divided by 2 -/
noncomputable def blackboard_operation (a b : ℝ) : ℝ := (a + b) / 4

/-- The process of applying the blackboard operation n-1 times -/
noncomputable def blackboard_process (n : ℕ) (numbers : List ℝ) : ℝ :=
  match n, numbers with
  | 0, _ | 1, _ => numbers.head!
  | n+2, a :: b :: rest =>
    let new_number := blackboard_operation a b
    blackboard_process (n+1) (new_number :: rest)
  | _, _ => 0 -- This case should never happen if the input is valid

/-- Theorem: The final number after the blackboard process is at least 1/n -/
theorem final_number_lower_bound (n : ℕ) (h : n ≥ 2) :
  ∀ (permutation : List ℝ),
    permutation.length = n →
    (∀ x ∈ permutation, x = 1) →
    blackboard_process n permutation ≥ 1 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_lower_bound_l944_94432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_recorder_price_proof_l944_94438

/-- Calculates the price an employee pays for a video recorder given the wholesale cost,
    store markup percentage, and employee discount percentage. -/
noncomputable def employee_price (wholesale_cost : ℝ) (store_markup_percent : ℝ) (employee_discount_percent : ℝ) : ℝ :=
  let retail_price := wholesale_cost * (1 + store_markup_percent / 100)
  retail_price * (1 - employee_discount_percent / 100)

/-- Proves that given a wholesale cost of $200, a store markup of 20%,
    and an employee discount of 20%, the final price an employee pays for a video recorder is $192. -/
theorem video_recorder_price_proof :
  employee_price 200 20 20 = 192 := by
  -- Unfold the definition of employee_price
  unfold employee_price
  -- Simplify the expression
  simp
  -- The proof is completed using numerical approximation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_recorder_price_proof_l944_94438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l944_94439

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 12)

noncomputable def f' (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 12)

noncomputable def y (x : ℝ) : ℝ := 2 * f x + f' x

theorem monotone_decreasing_interval :
  StrictAntiOn y (Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l944_94439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l944_94499

/-- Represents a point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- The main theorem stating the possible values of t -/
theorem triangle_area_theorem (t : ℝ) : 
  let D : Point := ⟨3, 15⟩
  let E : Point := ⟨15, 0⟩
  let F : Point := ⟨0, t⟩
  triangleArea D E F = 50 → t = 325/12 ∨ t = 125/12 := by
  sorry

#check triangle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l944_94499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_when_m_is_one_f_monotone_increasing_interval_l944_94479

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + (m^2 - 1) * x

-- Theorem for part (1)
theorem f_extrema_when_m_is_one :
  let m : ℝ := 1
  ∀ x ∈ Set.Icc (-3 : ℝ) 2, f m x ≤ 18 ∧ f m 0 = 0 ∧ ∃ y ∈ Set.Icc (-3 : ℝ) 2, f m y = 18 :=
by
  sorry

-- Theorem for part (2)
theorem f_monotone_increasing_interval (m : ℝ) (h : m > 0) :
  StrictMonoOn (f m) (Set.Ioo (1 - m) (m + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_when_m_is_one_f_monotone_increasing_interval_l944_94479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circleplus_calculation_l944_94492

-- Define the ⊕ operation
noncomputable def circleplus (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

-- Theorem statement
theorem circleplus_calculation :
  circleplus 7 4 = 11 / 29 ∧
  circleplus 2 (circleplus 7 4) = 23 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circleplus_calculation_l944_94492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_autograph_problem_l944_94460

theorem autograph_problem (n : ℕ) (A B C : Finset ℕ) : 
  n = 42 ∧ 
  A.card = 21 ∧ 
  B.card = 20 ∧ 
  C.card = 18 ∧ 
  (A ∩ B).card = 7 ∧ 
  (A ∩ C).card = 10 ∧ 
  (B ∩ C).card = 11 ∧ 
  (A ∩ B ∩ C).card = 6 → 
  n - (A ∪ B ∪ C).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_autograph_problem_l944_94460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_perpendicular_l944_94464

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the median function
noncomputable def median (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ :=
  ((v.1 + (t.B.1 + t.C.1) / 2) / 2, (v.2 + (t.B.2 + t.C.2) / 2) / 2)

-- Define the perpendicular function
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem triangle_median_perpendicular (t : Triangle) :
  perpendicular (median t t.A) (median t t.B) →
  distance t.B t.C = 7 →
  distance t.A t.C = 6 →
  distance t.A t.B = Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_perpendicular_l944_94464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_equation_l944_94449

/-- Homothety transformation in a vector space -/
def homothety {V : Type*} [AddCommGroup V] [Module ℝ V] (S : V) (k : ℝ) (A : V) : V :=
  k • A + (1 - k) • S

/-- Theorem: The homothety transformation satisfies the expected equation -/
theorem homothety_equation {V : Type*} [AddCommGroup V] [Module ℝ V] (S : V) (k : ℝ) (A : V) :
  homothety S k A = k • A + (1 - k) • S :=
by
  -- The proof is trivial as it's the definition of homothety
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_equation_l944_94449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_ω_property_l944_94430

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

def P (a b c : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + 12

def has_ω_property (a b c : ℝ) : Prop :=
  ∀ r : ℂ, P a b c r = 0 → P a b c (ω * r) = 0

theorem unique_polynomial_with_ω_property :
  ∃! (a b c : ℝ), has_ω_property a b c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_ω_property_l944_94430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_partitioned_rectangle_l944_94405

/-- The area of the shaded region in a specially partitioned rectangle --/
theorem shaded_area_in_partitioned_rectangle : 
  ∀ (width height : ℝ) (distance_from_edge : ℝ),
  width = 10 → height = 12 → distance_from_edge = 3 →
  2 * (1/2 * (width / 2) * height) = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_partitioned_rectangle_l944_94405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_extrema_l944_94402

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_distance_between_extrema (x₁ x₂ : ℝ) :
  (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  ∃ d : ℝ, d = |x₁ - x₂| ∧ d ≥ 2 ∧ ∀ y z : ℝ, (∀ x : ℝ, f y ≤ f x ∧ f x ≤ f z) → |y - z| ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_extrema_l944_94402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_12_l944_94401

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_12 (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7) + 3 * a 9 = 15 →
  sum_arithmetic a 12 = 30 := by
  sorry

#check arithmetic_sequence_sum_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_12_l944_94401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_pie_cost_l944_94478

-- Define the constants for ingredient costs and quantities
noncomputable def flour_cost : ℝ := 2
noncomputable def sugar_cost : ℝ := 1
noncomputable def eggs_butter_cost : ℝ := 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_container_oz : ℝ := 8
noncomputable def blueberry_pounds_needed : ℝ := 3
noncomputable def cherry_bag_cost : ℝ := 14
noncomputable def cherry_bag_pounds : ℝ := 4

-- Define the function to calculate the cost of the blueberry pie
noncomputable def blueberry_pie_cost : ℝ :=
  flour_cost + sugar_cost + eggs_butter_cost +
  (blueberry_pounds_needed * 16 / blueberry_container_oz * blueberry_container_cost)

-- Define the function to calculate the cost of the cherry pie
noncomputable def cherry_pie_cost : ℝ :=
  flour_cost + sugar_cost + eggs_butter_cost + cherry_bag_cost

-- Theorem statement
theorem cheapest_pie_cost :
  min blueberry_pie_cost cherry_pie_cost = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_pie_cost_l944_94478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_pass_rate_l944_94490

/-- The pass rate of a product going through two independent processing steps -/
theorem product_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = (1 - a) * (1 - b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_pass_rate_l944_94490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_heap_reorganization_l944_94456

/-- Represents a ball with its original and current heap sizes -/
structure Ball where
  originalHeapSize : ℕ+
  currentHeapSize : ℕ+

/-- The theorem statement -/
theorem ball_heap_reorganization 
  (n k : ℕ+) 
  (balls : Finset Ball) 
  (initial_heaps : Finset (Finset Ball))
  (final_heaps : Finset (Finset Ball)) :
  (initial_heaps.card = n) →
  (final_heaps.card = n + k) →
  (∀ h, h ∈ initial_heaps → h.Nonempty) →
  (∀ h, h ∈ final_heaps → h.Nonempty) →
  (∀ b, b ∈ balls → ∃! h, h ∈ initial_heaps ∧ b ∈ h) →
  (∀ b, b ∈ balls → ∃! h, h ∈ final_heaps ∧ b ∈ h) →
  ∃ reduced_balls : Finset Ball,
    reduced_balls.card = k + 1 ∧
    ∀ b, b ∈ reduced_balls → b.originalHeapSize > b.currentHeapSize :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_heap_reorganization_l944_94456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r₂_bound_l944_94461

/-- A function f(x) = x² - r₂x + r₃ --/
def f (r₂ r₃ : ℝ) : ℝ → ℝ := λ x ↦ x^2 - r₂*x + r₃

/-- The sequence g_n defined recursively --/
def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The theorem stating the properties of f and g, and the conclusion about r₂ --/
theorem r₂_bound (r₂ r₃ : ℝ) : 
  (∀ i ∈ Finset.range 2012, g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) →
  (∃ j : ℕ, ∀ i > j, g r₂ r₃ (i + 1) > g r₂ r₃ i) →
  (∀ M : ℝ, ∃ N : ℕ, g r₂ r₃ N > M) →
  |r₂| > 2 ∧ ∀ ε > 0, ∃ r₂' r₃' : ℝ, |r₂'| < 2 + ε ∧ 
    (∀ i ∈ Finset.range 2012, g r₂' r₃' (2*i) < g r₂' r₃' (2*i + 1) ∧ g r₂' r₃' (2*i + 1) > g r₂' r₃' (2*i + 2)) ∧
    (∃ j : ℕ, ∀ i > j, g r₂' r₃' (i + 1) > g r₂' r₃' i) ∧
    (∀ M : ℝ, ∃ N : ℕ, g r₂' r₃' N > M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r₂_bound_l944_94461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l944_94410

def b : ℕ → ℝ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | n + 1 => (64 * (b n)^3)^(1/3)

theorem b_50_value : b 50 = 2 * 4^49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l944_94410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_54_l944_94429

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a train is 54 km/hr -/
theorem train_speed_is_54 (length : ℝ) (time : ℝ) 
  (h1 : length = 135) 
  (h2 : time = 9) : 
  train_speed length time = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_54_l944_94429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_poly_satisfies_recurrence_v_poly_initial_condition_sum_of_coefficients_is_three_l944_94419

/-- Sequence v_n defined by the given recurrence relation -/
def v : ℕ → ℚ
  | 0 => 5  -- Adding the base case for 0
  | n + 1 => v n + n^2 + n - 1

/-- v_n as a polynomial function -/
def v_poly (n : ℕ) : ℚ := (1/3) * n^3 - (4/3) * n + (20/3)

/-- Theorem stating that v_poly satisfies the recurrence relation -/
theorem v_poly_satisfies_recurrence :
  ∀ n : ℕ, v_poly (n + 1) - v_poly n = n^2 + n - 1 := by
  sorry

/-- Theorem stating that v_poly matches the initial condition -/
theorem v_poly_initial_condition : v_poly 1 = 5 := by
  sorry

/-- Main theorem: The sum of coefficients of v_poly is 3 -/
theorem sum_of_coefficients_is_three :
  (1/3 : ℚ) + 0 + (-4/3) + (20/3) = 3 := by
  sorry

#eval v_poly 1  -- This will evaluate v_poly at n = 1 for verification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_poly_satisfies_recurrence_v_poly_initial_condition_sum_of_coefficients_is_three_l944_94419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_invertible_interval_for_g_l944_94442

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 8

theorem largest_invertible_interval_for_g :
  ∃ (a : ℝ), (∀ x y, x ∈ Set.Ici a → y ∈ Set.Ici a → g x = g y → x = y) ∧
             (∀ b, b < a → ¬(∀ x y, x ∈ Set.Ici b → y ∈ Set.Ici b → g x = g y → x = y)) ∧
             (2 ∈ Set.Ici a) ∧
             (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_invertible_interval_for_g_l944_94442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_correct_l944_94448

-- Define the equation
noncomputable def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 27) / x = 10

-- Define the smallest solution
noncomputable def smallest_solution : ℝ := (1 - Real.sqrt 649) / 12

-- Theorem statement
theorem smallest_solution_is_correct :
  equation smallest_solution ∧
  ∀ y, y < smallest_solution → ¬ equation y := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_correct_l944_94448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l944_94469

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem geometric_sequence_property
  (b : ℕ → ℝ)
  (h_geom : is_geometric_sequence b)
  (m n : ℕ)
  (h_mn : n - m ≥ 2)
  (c d : ℝ)
  (h_c : b m = c)
  (h_d : b n = d)
  (h_pos : ∀ k, b k > 0) :
  b (m + n) = (d^n / c^m)^(1 / (n - m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l944_94469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l944_94494

noncomputable def a (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.sin (ω * x))

noncomputable def b (ω : ℝ) (x : ℝ) : ℝ × ℝ := (2 * Real.cos (ω * x) + Real.sin (ω * x), Real.cos (ω * x))

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2

def is_periodic (g : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, g (x + T) = g x

def smallest_positive_period (g : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic g T ∧ T > 0 ∧ ∀ T' > 0, is_periodic g T' → T ≤ T'

theorem f_properties (ω : ℝ) (h1 : ω > 0) (h2 : smallest_positive_period (f ω) (Real.pi / 4)) :
  ω = 4 ∧
  (∀ x, f ω x ≤ 1 + Real.sqrt 2) ∧
  (∀ x, f ω x = 1 + Real.sqrt 2 ↔ ∃ k : ℤ, x = Real.pi / 32 + k * Real.pi / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l944_94494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_distance_l944_94424

noncomputable section

open Real

/-- Curve C₁ in parametric form -/
def curve_C₁ (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (-sqrt 2 / 2 + r * cos θ, -sqrt 2 / 2 + r * sin θ)

/-- Curve C₂ in polar form -/
def curve_C₂ (ρ θ : ℝ) : Prop :=
  ρ * cos (θ - π/4) = 1

/-- Distance between a point and a line -/
def distance_point_line (x y : ℝ) : ℝ :=
  abs (x + y - sqrt 2) / sqrt 2

/-- Maximum distance from C₁ to C₂ -/
def max_distance (r : ℝ) : ℝ :=
  distance_point_line (-sqrt 2 / 2) (-sqrt 2 / 2) + r

theorem curves_intersection_and_max_distance (r : ℝ) :
  (r > 0) →
  (r = sqrt 5 → ∃ θ, ∃ ρ, curve_C₂ ρ θ ∧ (curve_C₁ r θ).1 = ρ * cos θ ∧ (curve_C₁ r θ).2 = ρ * sin θ) ∧
  (max_distance r = 3 → r = 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_distance_l944_94424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l944_94482

noncomputable section

/-- Definition of the hyperbola C -/
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of eccentricity -/
def eccentricity (a c : ℝ) : ℝ := c / a

/-- Definition of the real axis length -/
def real_axis_length (a : ℝ) : ℝ := 2 * a

/-- Definition of the intersecting line -/
def line (x y m : ℝ) : Prop := y = x + m

/-- Definition of the chord length -/
noncomputable def chord_length (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem hyperbola_properties (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  (∃ c : ℝ, eccentricity a c = Real.sqrt 3 ∧ real_axis_length a = 2) →
  (∀ x y : ℝ, hyperbola a b x y ↔ hyperbola 1 (Real.sqrt 2) x y) ∧
  (∀ m : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ,
    hyperbola 1 (Real.sqrt 2) x₁ y₁ ∧
    hyperbola 1 (Real.sqrt 2) x₂ y₂ ∧
    line x₁ y₁ m ∧
    line x₂ y₂ m ∧
    chord_length x₁ y₁ x₂ y₂ = 4 * Real.sqrt 2) →
    m = 1 ∨ m = -1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l944_94482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l944_94498

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 - Real.rpow 3 x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Iio 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l944_94498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_plus_two_phi_l944_94407

theorem theta_plus_two_phi (θ φ : ℝ) (h1 : 0 < θ ∧ θ < π/2) (h2 : 0 < φ ∧ φ < π/2)
  (h3 : Real.tan θ = 2/9) (h4 : Real.sin φ = 3/5) :
  θ + 2*φ = Real.arctan (122/39) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_plus_two_phi_l944_94407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_under_central_symmetry_l944_94476

/-- Central symmetry transformation -/
def central_symmetry (O : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (2 * O.1 - P.1, 2 * O.2 - P.2)

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle -/
def on_circle (P : ℝ × ℝ) (C : Circle) : Prop :=
  distance P C.center = C.radius

theorem circle_under_central_symmetry 
  (S : Circle) (O : ℝ × ℝ) :
  ∃ (S' : Circle), 
    (S'.radius = S.radius) ∧ 
    (∀ (X : ℝ × ℝ), on_circle X S ↔ on_circle (central_symmetry O X) S') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_under_central_symmetry_l944_94476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l944_94440

/-- For two arithmetic sequences {a_n} and {b_n}, the sums of the first n terms are S_n and T_n respectively.
    If S_n / T_n = (7n + 3) / (n + 3), then a_8 / b_8 = 6. -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ) 
    (h : ∀ n, S n / T n = (7 * n + 3) / (n + 3)) : 
    a 8 / b 8 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l944_94440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_set_l944_94470

noncomputable def tan_period : ℝ := Real.pi

noncomputable def tan_domain (k : ℤ) : Set ℝ := Set.Ioo (-(Real.pi/2) + k * Real.pi) (Real.pi/2 + k * Real.pi)

noncomputable def solution_set (k : ℤ) : Set ℝ := Set.Ioo (-(Real.pi/6) + k * Real.pi) (Real.pi/2 + k * Real.pi)

theorem tan_inequality_solution_set :
  ∀ α : ℝ, (∃ k : ℤ, α ∈ tan_domain k) →
  (Real.tan α + Real.sqrt 3 / 3 > 0 ↔ ∃ k : ℤ, α ∈ solution_set k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_set_l944_94470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_dice_proof_l944_94453

/-- The probability of rolling three standard dice and obtaining numbers a, b, and c
    such that (a-1)(a-6)(b-1)(b-6)(c-1)(c-6) ≠ 0 -/
def prob_three_dice : ℚ := 8 / 27

/-- A standard die has 6 faces -/
def standard_die_faces : ℕ := 6

/-- The number of favorable outcomes per die (2, 3, 4, 5) -/
def favorable_outcomes : ℕ := 4

theorem prob_three_dice_proof :
  prob_three_dice = (favorable_outcomes : ℚ) / standard_die_faces ^ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_dice_proof_l944_94453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l944_94417

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 - 2*x + 1

-- Define the interval
def I : Set ℝ := Set.Icc (-3) 0

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 13/3 ∧ f b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l944_94417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l944_94428

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = t.a - (1/2) * t.c) : 
  t.B = π/3 ∧ 
  (t.b = 1 → ∀ (a c : Real), a = t.a → c = t.c → a + c ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l944_94428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_6_smallest_positive_period_monotonically_increasing_interval_l944_94411

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi - 2*x) + 2*Real.sqrt 3 * (Real.cos x)^2

-- Theorem for part I
theorem f_at_pi_over_6 : f (Real.pi/6) = 2 * Real.sqrt 3 := by sorry

-- Theorem for part II (smallest positive period)
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ 
  ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T' := by sorry

-- Theorem for part II (monotonically increasing interval)
theorem monotonically_increasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Icc (k*Real.pi - 5*Real.pi/12) (k*Real.pi + Real.pi/12)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_6_smallest_positive_period_monotonically_increasing_interval_l944_94411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_passes_through_point_l944_94441

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a ray of light -/
structure LightRay where
  origin : Point
  direction : Point

def reflect_point (p : Point) (axis : Line) : Point :=
  { x := p.x, y := -p.y }

noncomputable def reflection_line (incident : LightRay) (axis : Line) : Line :=
  sorry

-- Define a membership relation for Point and Line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

instance : Membership Point Line where
  mem := point_on_line

theorem reflection_passes_through_point :
  ∀ (incident : LightRay) (axis : Line),
    incident.origin = Point.mk (-3) 2 →
    axis = Line.mk 0 0 →
    ∃ (reflected : Line),
      reflected = reflection_line incident axis ∧
      ∃ (p : Point), p ∈ reflected ∧ p = Point.mk (-3) (-2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_passes_through_point_l944_94441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_property_l944_94418

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ := λ n => a * r^(n - 1)

noncomputable def sum_geometric (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sum_property (a : ℝ) (r : ℝ) (n : ℕ) :
  (sum_geometric a r n = 24) →
  (sum_geometric a r (3*n) = 42) →
  (sum_geometric a r (2*n) = 36) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_property_l944_94418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l944_94433

theorem abc_relationship : 
  let a : ℝ := (Real.log 2) / 2
  let b : ℝ := 3 / (2 * Real.sqrt (Real.exp 1))
  let c : ℝ := 4 / (3 * (Real.exp 1) ^ (1/3))
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l944_94433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l944_94406

theorem equation_solution :
  ∀ x : ℝ, (1 + 2 * Real.sqrt x - x^(1/3) - 2 * x^(1/6) = 0) ↔ (x = 1 ∨ x = 1/64) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l944_94406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BRS_l944_94436

-- Define the point B
def B : ℝ × ℝ := (4, 12)

-- Define the slopes of the two perpendicular lines
variable (m₁ m₂ : ℝ)

-- Define the y-intercepts of the two lines
variable (b₁ b₂ : ℝ)

-- Define the perpendicularity condition
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define the sum of y-intercepts condition
def sum_of_intercepts (b₁ b₂ : ℝ) : Prop := b₁ + b₂ = 4

-- Define the line equations
def line1_eq (m₁ b₁ : ℝ) (x y : ℝ) : Prop := y = m₁ * x + b₁
def line2_eq (m₂ b₂ : ℝ) (x y : ℝ) : Prop := y = m₂ * x + b₂

-- Define that B is on both lines
def B_on_lines (m₁ m₂ b₁ b₂ : ℝ) : Prop :=
  line1_eq m₁ b₁ B.1 B.2 ∧ line2_eq m₂ b₂ B.1 B.2

-- Define the area of triangle BRS
def area_BRS : ℝ := 8

-- Theorem statement
theorem area_of_triangle_BRS (m₁ m₂ b₁ b₂ : ℝ) :
  perpendicular m₁ m₂ → sum_of_intercepts b₁ b₂ → B_on_lines m₁ m₂ b₁ b₂ → area_BRS = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BRS_l944_94436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l944_94421

noncomputable def original_function (x : ℝ) : ℝ := 4 * Real.sin (x + Real.pi / 5)

noncomputable def target_function (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 5)

def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * x)

theorem graph_transformation (x : ℝ) :
  transform original_function x = target_function x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l944_94421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_swallow_problem_l944_94454

/-- Represents the weight of a sparrow in jin -/
def x : Real := Real.mk 0  -- Placeholder value

/-- Represents the weight of a swallow in jin -/
def y : Real := Real.mk 0  -- Placeholder value

/-- The total weight of five sparrows and six swallows is one jin -/
def total_weight : Prop := 5 * x + 6 * y = 1

/-- Sparrows are heavier than swallows -/
def sparrow_heavier : Prop := x > y

/-- Exchanging one sparrow with one swallow results in equal weight -/
def equal_after_exchange : Prop := 4 * x + 7 * y = 5 * x + 6 * y

/-- The system of equations correctly represents the problem -/
theorem sparrow_swallow_problem : 
  ∃ (x y : Real), (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) := by
  sorry

#check sparrow_swallow_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_swallow_problem_l944_94454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_age_of_siblings_l944_94495

-- Define the ages of the siblings
def susan_age : ℕ := 15
def bob_age : ℕ := 11

-- Define the age differences
def arthur_age_diff : ℕ := 2
def tom_age_diff : ℕ := 3

-- Calculate Arthur's and Tom's ages
def arthur_age : ℕ := susan_age + arthur_age_diff
def tom_age : ℕ := bob_age - tom_age_diff

-- Theorem: The sum of all siblings' ages is 51
theorem total_age_of_siblings : susan_age + arthur_age + bob_age + tom_age = 51 := by
  -- Unfold the definitions
  unfold susan_age arthur_age bob_age tom_age
  unfold arthur_age_diff tom_age_diff
  -- Simplify the arithmetic
  simp_arith
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_age_of_siblings_l944_94495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_spectral_density_formula_l944_94496

/-- A type representing a differentiable stationary random function -/
structure DiffStationaryRandomFunction where
  func : ℝ → ℂ
  is_differentiable : Differentiable ℝ func
  is_stationary : ∀ t₁ t₂ : ℝ, (func (t₁ + t₂) - func t₁) = (func t₂ - func 0)

/-- Spectral density of a differentiable stationary random function -/
noncomputable def spectral_density (X : DiffStationaryRandomFunction) : ℝ → ℂ := sorry

/-- Cross-spectral density of a function and its derivative -/
noncomputable def cross_spectral_density (X : DiffStationaryRandomFunction) : ℝ → ℂ := sorry

/-- The main theorem: cross-spectral density equals i * ω * spectral_density -/
theorem cross_spectral_density_formula (X : DiffStationaryRandomFunction) (ω : ℝ) :
  cross_spectral_density X ω = Complex.I * ω * spectral_density X ω := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_spectral_density_formula_l944_94496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l944_94475

-- Define what a quadratic equation is
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the four equations
noncomputable def eq_A (x : ℝ) : ℝ := x^3 + 3*x
noncomputable def eq_B (x : ℝ) : ℝ := (x-1)^2 - x^2
noncomputable def eq_C (x : ℝ) : ℝ := x + 4/x - 1
noncomputable def eq_D (x : ℝ) : ℝ := x^2 - 3*x - 2

-- Theorem stating that eq_D is quadratic while others are not
theorem quadratic_equation_identification :
  is_quadratic eq_D ∧ 
  ¬is_quadratic eq_A ∧ 
  ¬is_quadratic eq_B ∧ 
  ¬is_quadratic eq_C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l944_94475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tromino_to_square_l944_94422

/-- Represents a T-shaped tromino composed of three unit squares -/
structure TTromino :=
  (squares : Finset (ℤ × ℤ))
  (is_t_shape : squares = {(0, 0), (1, 0), (0, 1)})

/-- Represents a cut of the T-shaped tromino -/
structure Cut :=
  (part1 part2 : Set (ℚ × ℚ))
  (covers_tromino : part1 ∪ part2 = {(x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 1})
  (disjoint : part1 ∩ part2 = ∅)

/-- Represents a square with area 3 -/
def SquareArea3 : Set (ℚ × ℚ) :=
  {(x, y) | 0 ≤ x ∧ x ≤ Real.sqrt 3 ∧ 0 ≤ y ∧ y ≤ Real.sqrt 3}

/-- Main theorem: T-shaped tromino can be cut into two parts reassemblable into a square -/
theorem tromino_to_square (t : TTromino) :
  ∃ (c : Cut), ∃ (f1 f2 : Set (ℚ × ℚ) → Set (ℚ × ℚ)),
    f1 c.part1 ∪ f2 c.part2 = SquareArea3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tromino_to_square_l944_94422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l944_94451

theorem sine_graph_transformation (x : ℝ) :
  Real.sin (4 * x - π / 3) = Real.sin ((4 * (x - π / 8)) + π / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l944_94451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_contradiction_step_l944_94447

/-- A triangle is a geometric figure with three angles. -/
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : angles 0 + angles 1 + angles 2 = Real.pi

/-- An angle is obtuse if it is greater than π/2. -/
def is_obtuse (angle : ℝ) : Prop := angle > Real.pi/2

/-- The correct first step in a proof by contradiction for the proposition
    "In a triangle, there cannot be two obtuse angles" is to assume that
    in a triangle, there can be two obtuse angles. -/
theorem correct_contradiction_step (t : Triangle) : 
  (∃ (i j : Fin 3), i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) →
  (∀ (t : Triangle), ¬(∃ (i j : Fin 3), i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j))) →
  False := by
  sorry

#check correct_contradiction_step

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_contradiction_step_l944_94447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_coefficient_sum_l944_94457

/-- Angle bisector of a triangle -/
noncomputable def angle_bisector (P Q R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  { point | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
    let PQ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
    let PR := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
    point = (P.1 + t * (PQ * (R.1 - P.1) + PR * (Q.1 - P.1)) / (PQ + PR),
             P.2 + t * (PQ * (R.2 - P.2) + PR * (Q.2 - P.2)) / (PQ + PR)) }

/-- The sum of coefficients a and c in the angle bisector equation of a specific triangle -/
theorem angle_bisector_coefficient_sum : 
  let P : ℝ × ℝ := (-7, 6)
  let Q : ℝ × ℝ := (-12, -20)
  let R : ℝ × ℝ := (2, -8)
  ∃ (a c : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ angle_bisector P Q R ↔ a * x + 3 * y + c = 0) ∧
    a + c = 123 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_coefficient_sum_l944_94457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l944_94412

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (Real.cos (ω * x - Real.pi / 6))^2 - (Real.sin (ω * x))^2

theorem problem_solution (ω : ℝ) (x₀ : ℝ) (h_ω : ω > 0) 
  (h_zero : f ω x₀ = 0 ∧ f ω (x₀ + Real.pi / 2) = 0) :
  (f ω (Real.pi / 12) = Real.sqrt 3 / 2) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (-7 * Real.pi / 12) 0, |f ω x - m| ≤ 1) → 
    m ∈ Set.Icc (-1 / 4) (1 - Real.sqrt 3 / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l944_94412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_diamond_four_equals_six_l944_94437

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a - a / b

-- Theorem statement
theorem eight_diamond_four_equals_six :
  diamond 8 4 = 6 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_diamond_four_equals_six_l944_94437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_face_centers_form_icosahedron_l944_94420

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  -- Add necessary properties here

/-- A regular icosahedron -/
structure RegularIcosahedron where
  -- Add necessary properties here

/-- The operation of connecting the centers of faces of a polyhedron -/
def connectFaceCenters (d : RegularDodecahedron) : RegularIcosahedron :=
  sorry

/-- Theorem: Connecting the centers of the faces of a regular dodecahedron results in a regular icosahedron -/
theorem dodecahedron_face_centers_form_icosahedron (d : RegularDodecahedron) :
  ∃ (i : RegularIcosahedron), i = connectFaceCenters d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_face_centers_form_icosahedron_l944_94420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_from_inclination_angle_l944_94443

theorem line_slope_from_inclination_angle (θ : Real) (h : θ = π / 3) :
  Real.tan θ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_from_inclination_angle_l944_94443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l944_94435

noncomputable def sequence_a : ℕ → ℝ := sorry

noncomputable def S : ℕ → ℝ := sorry

noncomputable def b (n : ℕ) : ℝ := S n / (2 * ↑n + 1)

noncomputable def T : ℕ → ℝ := sorry

theorem sequence_properties :
  (∀ n ≥ 2, (S n) ^ 2 = sequence_a n * (S n - 1/2)) ∧
  sequence_a 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → S n = 1 / (2 * ↑n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → T n = ↑n / (2 * ↑n + 1)) :=
by sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l944_94435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_51_l944_94400

def numbers : List Nat := [39, 51, 77, 91, 121]

def largest_prime_factor (n : Nat) : Nat :=
  (Nat.factors n).maximum?.getD 1

theorem largest_prime_factor_is_51 :
  ∃ (n : Nat), n ∈ numbers ∧
  ∀ (m : Nat), m ∈ numbers → largest_prime_factor n ≥ largest_prime_factor m :=
by
  -- Use 51 as the witness
  use 51
  constructor
  · simp [numbers] -- Prove 51 ∈ numbers
  · intro m hm
    cases hm
    all_goals
      simp [largest_prime_factor]
      -- We would need to prove these inequalities, but for now we'll use sorry
      sorry

#eval numbers.map largest_prime_factor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_51_l944_94400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_x_minus_y_plus_three_eq_zero_l944_94452

/-- The inclination angle of a line with slope m is the angle θ such that tan(θ) = m -/
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

/-- The slope of a line given by ax + by + c = 0 where b ≠ 0 is -a/b -/
noncomputable def slope_from_general_form (a b c : ℝ) : ℝ := -a / b

theorem inclination_angle_of_x_minus_y_plus_three_eq_zero :
  let m := slope_from_general_form 1 (-1) 3
  inclination_angle m = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_x_minus_y_plus_three_eq_zero_l944_94452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l944_94455

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (a b : ℝ × ℝ) : Prop :=
  ∀ (k : ℝ), a ≠ k • b

/-- Three points are collinear if the vector from the first to the second is a scalar multiple of the vector from the first to the third -/
def ThreePointsCollinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), B - A = k • (C - A)

theorem collinear_vectors (a b : ℝ × ℝ) (m : ℝ) :
  NonCollinear a b →
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := m • a + 2 • b
  let C : ℝ × ℝ := B + (3 • a + m • b)
  ThreePointsCollinear A B C →
  m^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l944_94455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_prime_in_sequence_l944_94491

def a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * a n + 1

def is_smallest_non_prime (n : ℕ) : Prop :=
  ¬ Nat.Prime (a n) ∧
  ∀ k < n, Nat.Prime (a k)

theorem smallest_non_prime_in_sequence :
  ∃ n, is_smallest_non_prime n ∧ a n = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_prime_in_sequence_l944_94491
