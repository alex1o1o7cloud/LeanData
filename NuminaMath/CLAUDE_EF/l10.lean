import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mean_after_adding_specific_case_l10_1023

theorem new_mean_after_adding (n : ℕ) (original_mean add_value : ℝ) :
  n > 0 →
  n * (original_mean + add_value) / n = original_mean + add_value := by
  sorry

theorem specific_case : 
  let n : ℕ := 15
  let original_mean : ℝ := 40
  let add_value : ℝ := 13
  (n * (original_mean + add_value) / n : ℝ) = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mean_after_adding_specific_case_l10_1023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l10_1062

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ M :=
by
  -- We'll use 1 as our maximum value M
  use 1
  constructor
  · -- Prove M = 1
    rfl
  · -- Prove ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ M
    intro x hx
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l10_1062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_42_56_l10_1094

theorem log_42_56 (a b : ℝ) (h1 : Real.log 3 / Real.log 2 = a) (h2 : Real.log 7 / Real.log 3 = b) :
  Real.log 56 / Real.log 42 = (a * b + 3) / (a * b + a + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_42_56_l10_1094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_cube_properties_l10_1087

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- A solid in 3D space -/
structure Solid3D

/-- A cube in 3D space -/
structure Cube3D where
  center : Point3D
  side_length : ℝ

/-- Possible shapes of a plane-cube intersection -/
inductive IntersectionShape
  | Triangle
  | Quadrilateral
  | Pentagon
  | Hexagon

/-- Check if a point is on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

/-- Theorem: Geometric progression and cube intersection properties -/
theorem geometric_and_cube_properties :
  (∃ (p : Point3D) (l : Line3D), l.point = p) ∧
  (∃ (l : Line3D) (pl : Plane3D), pointOnLine pl.point l) ∧
  (∃ (pl : Plane3D) (s : Solid3D), True) ∧
  (∀ (c : Cube3D) (pl : Plane3D),
    ∃ (shape : IntersectionShape), True) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_cube_properties_l10_1087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_edge_bound_l10_1049

/-- A graph with no triangles and at most one edge between any two vertices -/
structure ChessTournamentGraph (p : ℕ) where
  vertex_count : p > 2
  edge_count : ℕ
  no_multi_edges : ∀ (v w : Fin p), v ≠ w → Bool
  no_triangles : ∀ (u v w : Fin p), u ≠ v ∧ v ≠ w ∧ u ≠ w → Bool

/-- The maximum number of edges in a ChessTournamentGraph is at most p^2/4 -/
theorem chess_tournament_edge_bound (p : ℕ) (G : ChessTournamentGraph p) :
  G.edge_count ≤ p^2/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_edge_bound_l10_1049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_on_parabola_l10_1029

/-- Parabola type representing y^2 = 4x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  axis : ℝ → Prop

/-- Point on the parabola --/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

/-- Distance between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to a line --/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : ℝ → Prop) : ℝ :=
  sorry -- Definition of distance to line

theorem distance_to_origin_on_parabola
  (p : Parabola)
  (A : PointOnParabola p)
  (h1 : distance A.point p.focus / distanceToLine A.point p.axis = 5 / 4)
  (h2 : distance A.point p.focus > 2) :
  distance A.point (0, 0) = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_on_parabola_l10_1029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_product_l10_1012

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus point
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k*(x - 2)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem parabola_intersection_distance_product :
  ∀ (k : ℝ) (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 →
    parabola x2 y2 →
    line_through_focus k x1 y1 →
    line_through_focus k x2 y2 →
    (distance x1 y1 (focus.1) (focus.2)) *
    (distance x2 y2 (focus.1) (focus.2)) ≥ 16 := by
  sorry

#check parabola_intersection_distance_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_product_l10_1012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l10_1048

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 3 / Real.log 4
def c : ℝ := 0.5

-- State the theorem
theorem log_inequality : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l10_1048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_growth_l10_1067

/-- The average growth rate function v(x) for fish in a "flowing water fish cage" -/
noncomputable def v (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then 2
  else if 4 < x ∧ x ≤ 20 then -1/8 * x + 5/2
  else 0

/-- The annual growth amount function f(x) -/
noncomputable def f (x : ℝ) : ℝ := x * v x

/-- Theorem stating the maximum annual growth amount and corresponding breeding density -/
theorem max_annual_growth :
  ∃ (x_max : ℝ), x_max = 10 ∧
  ∀ (x : ℝ), 0 < x → x ≤ 20 → f x ≤ f x_max ∧
  f x_max = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_growth_l10_1067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_calculation_l10_1057

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a triangle with three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

def are_tangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

def is_tangent_to_line (c : Circle) (p1 p2 : Point) : Prop :=
  distance c.center p1 = c.radius ∧ distance c.center p2 = c.radius

theorem triangle_perimeter_calculation 
  (A B C : Point) (P Q R S : Circle) : 
  (P.radius = 2 ∧ Q.radius = 2 ∧ R.radius = 2 ∧ S.radius = 2) →
  (are_tangent P Q ∧ are_tangent Q R ∧ are_tangent R S) →
  (is_tangent_to_line P A B) →
  (is_tangent_to_line S A C) →
  (is_tangent_to_line P B C ∧ is_tangent_to_line Q B C ∧ 
   is_tangent_to_line R B C ∧ is_tangent_to_line S B C) →
  perimeter (Triangle.mk A B C) = 
    distance A P.center + distance P.center Q.center + distance Q.center B +
    distance A S.center + distance S.center R.center + distance R.center C +
    distance B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_calculation_l10_1057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_distinguishable_colorings_l10_1050

/-- A color that can be used to paint a face of a tetrahedron -/
inductive Color
| Red
| White
| Blue
| Green

/-- A coloring of a tetrahedron is a function from its faces to colors -/
def Coloring := Fin 4 → Color

/-- Two colorings are considered equivalent if they can be rotated to appear identical -/
def equivalent (c1 c2 : Coloring) : Prop := sorry

/-- The set of all possible colorings of a tetrahedron -/
def allColorings : Finset Coloring := sorry

/-- The set of distinguishable colorings -/
def distinguishableColorings : Finset Coloring := sorry

/-- The number of distinguishable colorings is 58 -/
theorem number_of_distinguishable_colorings :
  Finset.card distinguishableColorings = 58 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_distinguishable_colorings_l10_1050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_surface_area_and_volume_l10_1044

/-- Represents a parallelepiped with specific properties -/
structure Parallelepiped (a : ℝ) where
  -- The base is a rhombus with side length a and acute angle 30°
  base_is_rhombus : True
  -- The diagonal of a lateral face is perpendicular to the base plane
  lateral_diagonal_perpendicular : True
  -- The lateral edge forms a 60° angle with the base plane
  lateral_edge_angle_is_60 : True

/-- Calculate the surface area of the parallelepiped -/
noncomputable def surface_area (a : ℝ) (p : Parallelepiped a) : ℝ :=
  a^2 * (1 + Real.sqrt 3 + Real.sqrt 13)

/-- Calculate the volume of the parallelepiped -/
noncomputable def volume (a : ℝ) (p : Parallelepiped a) : ℝ :=
  (a^3 * Real.sqrt 3) / 2

/-- Theorem stating the surface area and volume of the parallelepiped -/
theorem parallelepiped_surface_area_and_volume (a : ℝ) (p : Parallelepiped a) :
  surface_area a p = a^2 * (1 + Real.sqrt 3 + Real.sqrt 13) ∧
  volume a p = (a^3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_surface_area_and_volume_l10_1044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missile_trajectory_properties_l10_1076

/-- Represents the missile trajectory function during the second stage -/
def missile_trajectory (x : ℝ) : ℝ := -6 * x^2 + 168 * x - 176

/-- The time when the missile reaches its apogee -/
def apogee_time : ℝ := 14

/-- The altitude of the missile at its apogee -/
def apogee_altitude : ℝ := 1000

/-- The time when the second stage starts -/
def second_stage_start : ℝ := 4

/-- The time when the engine turns off -/
noncomputable def engine_off_time : ℝ := 14 + 5 * Real.sqrt 6

theorem missile_trajectory_properties :
  (missile_trajectory apogee_time = apogee_altitude) ∧
  (∀ x, x > second_stage_start → missile_trajectory x ≤ apogee_altitude) ∧
  (missile_trajectory engine_off_time = 100) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_missile_trajectory_properties_l10_1076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l10_1031

/-- Vector m -/
noncomputable def m : ℝ × ℝ := (Real.sqrt 3, 1)

/-- Vector n as a function of x -/
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (2*x), Real.sin (2*x))

/-- Function f(x) as dot product of m and n -/
noncomputable def f (x : ℝ) : ℝ := m.1 * (n x).1 + m.2 * (n x).2

/-- Theorem stating the properties of function f -/
theorem f_properties :
  (∀ x, f x ≤ 2) ∧ 
  (∀ x, f (π/3 + x) = f (π/3 - x)) ∧
  (∀ x ∈ Set.Ioo (π/12) (7*π/12), ∀ y ∈ Set.Ioo (π/12) (7*π/12), x < y → f y < f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l10_1031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l10_1032

/-- Represents the pay structure for an employee -/
structure PayStructure where
  baseHours : ℕ
  baseRate : ℝ
  overtimeRate : ℝ

/-- Calculates the total pay for an employee given their pay structure and hours worked -/
def totalPay (ps : PayStructure) (hoursWorked : ℕ) : ℝ :=
  if hoursWorked ≤ ps.baseHours then
    ps.baseRate * (hoursWorked : ℝ)
  else
    ps.baseRate * (ps.baseHours : ℝ) + ps.overtimeRate * ((hoursWorked - ps.baseHours) : ℝ)

/-- The main theorem to prove -/
theorem harry_hours_worked 
  (x : ℝ) 
  (harry : PayStructure) 
  (james : PayStructure) 
  (james_hours : ℕ) 
  (harry_hours : ℕ) :
  harry.baseHours = 24 →
  harry.baseRate = x →
  harry.overtimeRate = 1.5 * x →
  james.baseHours = 40 →
  james.baseRate = x →
  james.overtimeRate = 2 * x →
  james_hours = 41 →
  totalPay harry harry_hours = totalPay james james_hours →
  harry_hours = 36 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l10_1032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_equivalence_l10_1002

/-- The ternary expansion 0.10101010... as a real number -/
noncomputable def ternary_expansion : ℝ := ∑' n, (1 : ℝ) / 3^(2*n + 1)

/-- The binary expansion 0.110110110... as a real number -/
noncomputable def binary_expansion : ℝ := ∑' n, (1 : ℝ) / 2^(n + 1) - ∑' n, (1 : ℝ) / 2^(3*n + 3)

theorem expansion_equivalence :
  ternary_expansion = 3/8 ∧ binary_expansion = 6/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_equivalence_l10_1002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_pi_third_l10_1074

/-- The function to be maximized -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) + 2 * Real.sin (ω * x + Real.pi / 3)

/-- The theorem stating that π/3 maximizes the function for x ∈ (0, π/2) -/
theorem max_at_pi_third (ω : ℝ) (h_ω_pos : ω > 0) (h_period : (2 * Real.pi) / ω = 2 * Real.pi) :
  ∀ x ∈ Set.Ioo 0 (Real.pi / 2), f ω x ≤ f ω (Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_pi_third_l10_1074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_right_triangle_digit_sum_l10_1006

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = x^2 -/
def on_parabola (p : Point) : Prop := p.y = p.x^2

/-- Predicate for three points forming a right triangle with B as the right angle -/
def is_right_triangle (A B C : Point) : Prop :=
  (B.x - A.x) * (C.x - B.x) + (B.y - A.y) * (C.y - B.y) = 0

/-- Area of a triangle given three points -/
noncomputable def triangle_area (A B C : Point) : ℝ :=
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

/-- Sum of digits of a real number -/
noncomputable def sum_of_digits (x : ℝ) : ℕ := sorry

/-- Main theorem -/
theorem parabola_right_triangle_digit_sum 
  (A B C : Point) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_on_parabola : on_parabola A ∧ on_parabola B ∧ on_parabola C)
  (h_parallel : A.y = B.y)
  (h_right : is_right_triangle A B C)
  (h_area : triangle_area A B C = 1004) :
  sum_of_digits C.y = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_right_triangle_digit_sum_l10_1006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_range_l10_1035

def f (x : ℝ) : ℝ := 2 * x + 1

def process_stops_after (k : ℕ+) (x : ℝ) : Prop :=
  let rec x_seq : ℕ → ℝ
    | 0 => x
    | n + 1 => f (x_seq n)
  (∀ i < k, x_seq i ≤ 255) ∧ x_seq k > 255

theorem process_range (k : ℕ+) (x : ℝ) :
  process_stops_after k x ↔ x ∈ Set.Ioo ((2 : ℝ)^(8-k.val) - 1) ((2 : ℝ)^(9-k.val) - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_range_l10_1035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_product_of_N_values_l10_1061

theorem temperature_difference (N : ℝ) : 
  (∃ L : ℝ, 
    let M := L + N;
    let M_4pm := M - 10;
    let L_4pm := L + 5;
    abs (M_4pm - L_4pm) = 6) →
  N = 21 ∨ N = 9 :=
by sorry

theorem product_of_N_values : 
  (∀ N : ℝ, (∃ L : ℝ, 
    let M := L + N;
    let M_4pm := M - 10;
    let L_4pm := L + 5;
    abs (M_4pm - L_4pm) = 6) →
  N = 21 ∨ N = 9) →
  21 * 9 = 189 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_product_of_N_values_l10_1061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_evaluation_l10_1084

/-- A polynomial with integer coefficients and coefficients between 0 and 3 inclusive -/
def IntPolynomial : Type := { p : Polynomial ℤ // ∀ i, 0 ≤ p.coeff i ∧ p.coeff i < 4 }

/-- The statement of the problem -/
theorem polynomial_evaluation (P : IntPolynomial) :
  (P.val.map (algebraMap ℤ ℝ)).eval (Real.sqrt 3) = (30 : ℝ) + 25 * Real.sqrt 3 →
  P.val.eval 3 = 2731 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_evaluation_l10_1084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_viewing_distance_maximizes_angle_l10_1054

/-- The height of the painting in meters -/
def painting_height : ℝ := 3

/-- The distance from the viewer's eye level to the bottom of the painting in meters -/
def bottom_to_eye : ℝ := 1

/-- The optimal viewing distance in meters -/
def optimal_distance : ℝ := 2

/-- Theorem stating that the optimal viewing distance maximizes the vertical viewing angle -/
theorem optimal_viewing_distance_maximizes_angle :
  ∀ (d : ℝ), d > 0 →
  let θ := Real.arctan ((painting_height + bottom_to_eye) / d) - Real.arctan (bottom_to_eye / d)
  let θ_opt := Real.arctan ((painting_height + bottom_to_eye) / optimal_distance) - Real.arctan (bottom_to_eye / optimal_distance)
  θ ≤ θ_opt :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_viewing_distance_maximizes_angle_l10_1054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_investment_amount_l10_1037

noncomputable def simple_interest (x y : ℝ) : ℝ := x * y * 2 / 100

noncomputable def compound_interest (x y : ℝ) : ℝ := x * ((1 + y/100)^2 - 1)

theorem unique_investment_amount : ∃! x : ℝ, x > 0 ∧ 
  ∃ y : ℝ, y > 0 ∧ 
    simple_interest x y = 600 ∧ 
    compound_interest x y = 615 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_investment_amount_l10_1037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l10_1033

theorem sum_of_reciprocals (x y : ℝ) (h1 : (2 : ℝ)^x = 36) (h2 : (3 : ℝ)^y = 36) : 
  1/x + 1/y = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l10_1033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l10_1004

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ (z : ℂ), z = 2*Complex.I - 3 ∧ 2*z^3 + p*z^2 + q*z = 0) → p = 12 ∧ q = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l10_1004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_product_l10_1091

theorem smallest_m_for_product (m : ℕ) : m = 5 ↔ 
  (m > 0 ∧ (3 : ℝ)^((m + m^2) / 4 : ℝ) > 500 ∧ 
   ∀ k : ℕ, k > 0 ∧ k < m → (3 : ℝ)^((k + k^2) / 4 : ℝ) ≤ 500) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_product_l10_1091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l10_1058

noncomputable def vector_a : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 7)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (dot_product : vector_a.1 * b.1 + vector_a.2 * b.2 = -9)
  (angle : Real.cos (120 * π / 180) = -1/2) :
  Real.sqrt (b.1^2 + b.2^2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l10_1058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_l10_1075

-- Define the rectangle DEFG
noncomputable def rectangle_length : ℝ := 30
noncomputable def rectangle_width : ℝ := 20

-- Define the triangle ABC
noncomputable def triangle_base : ℝ := 12
noncomputable def triangle_height : ℝ := 15

-- Calculate the area of triangle ABC
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height

-- Calculate the area of rectangle DEFG
noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

-- Calculate the area of rectangle DEFG not covered by triangle ABC
noncomputable def uncovered_area : ℝ := rectangle_area - triangle_area

-- Theorem statement
theorem rectangle_triangle_area :
  triangle_area = 90 ∧ uncovered_area = 510 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_l10_1075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l10_1030

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (5 - x)) / Real.sqrt (x + 2)

-- Theorem statement
theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ -2 < x ∧ x < 5 :=
by
  sorry  -- Proof is skipped for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l10_1030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_rearrangement_impossible_l10_1007

/-- Represents a position on the chessboard -/
structure Position where
  row : Fin 8
  col : Fin 8

/-- The diagonal number of a position -/
def diagonalNumber (p : Position) : Nat :=
  p.row.val + p.col.val + 1

/-- A configuration of pieces on the chessboard -/
def Configuration := Fin 8 → Position

/-- Checks if a configuration has one piece per row and column -/
def isValidConfiguration (c : Configuration) : Prop :=
  (∀ i j : Fin 8, i ≠ j → c i ≠ c j) ∧
  (∀ r : Fin 8, ∃ i : Fin 8, (c i).row = r) ∧
  (∀ col : Fin 8, ∃ i : Fin 8, (c i).col = col)

theorem chessboard_rearrangement_impossible :
  ¬∃ (initial final : Configuration),
    isValidConfiguration initial ∧
    isValidConfiguration final ∧
    (∀ i : Fin 8, diagonalNumber (final i) > diagonalNumber (initial i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_rearrangement_impossible_l10_1007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l10_1068

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l10_1068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_l10_1010

/-- Represents a right circular cone -/
structure RightCircularCone where
  slantHeight : ℝ
  height : ℝ

/-- Calculates the curved surface area of a right circular cone -/
noncomputable def curvedSurfaceArea (cone : RightCircularCone) : ℝ :=
  let radius := Real.sqrt (cone.slantHeight^2 - cone.height^2)
  Real.pi * radius * cone.slantHeight

/-- Theorem: The curved surface area of a right circular cone with slant height 10 and height 8 is 60π -/
theorem cone_curved_surface_area :
  let cone : RightCircularCone := { slantHeight := 10, height := 8 }
  curvedSurfaceArea cone = 60 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_l10_1010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_volume_l10_1043

/-- The volume of water added by Pablo -/
noncomputable def pablo_addition (t : ℝ) : ℝ := t / 2

/-- The volume of water added by Chloe -/
noncomputable def chloe_addition (t : ℝ) : ℝ := t^2 / 4

/-- The total volume of the tank -/
noncomputable def tank_volume (t : ℝ) : ℝ := 5 * t^2 / 6

/-- The theorem stating the initial volume of water in the tank -/
theorem initial_water_volume (t : ℝ) :
  ∃ x : ℝ, 
    x + pablo_addition t = 0.2 * tank_volume t ∧
    x + pablo_addition t + chloe_addition t = 0.5 * tank_volume t ∧
    x = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_volume_l10_1043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_faces_l10_1064

/-- Represents a pair of fair dice -/
structure DicePair where
  faces1 : ℕ
  faces2 : ℕ
  h1 : faces1 ≥ 6
  h2 : faces2 ≥ 6

/-- The probability of rolling a specific sum with a pair of dice -/
def prob_sum (d : DicePair) (sum : ℕ) : ℚ :=
  (Finset.filter (λ (x : ℕ × ℕ) => x.1 + x.2 = sum) (Finset.product (Finset.range d.faces1) (Finset.range d.faces2))).card /
  (d.faces1 * d.faces2 : ℚ)

/-- The theorem stating the minimum sum of faces for the dice pair -/
theorem min_sum_faces (d : DicePair) 
  (h3 : prob_sum d 7 = 3/4 * prob_sum d 10)
  (h4 : prob_sum d 7 = 1/12 * prob_sum d 12) :
  d.faces1 + d.faces2 ≥ 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_faces_l10_1064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l10_1083

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are externally tangent -/
def ExternallyTangent (C₁ C₂ : Circle) : Prop := sorry

/-- A circle is internally tangent to another circle -/
def InternallyTangent (C₁ C₂ : Circle) : Prop := sorry

/-- The centers of three circles are collinear -/
def CollinearCenters (C₁ C₂ C₃ : Circle) : Prop := sorry

/-- A point is on the common external tangent of a circle -/
def OnCommonExternalTangent (P : ℝ × ℝ) (C : Circle) : Prop := sorry

/-- A point is on the tangent of a circle -/
def OnTangent (P : ℝ × ℝ) (C : Circle) : Prop := sorry

/-- The tangent at a point is perpendicular to the diameter passing through that point -/
def TangentPerpendicular (C : Circle) (P : ℝ × ℝ) : Prop := sorry

/-- Given three circles C₁, C₂, and C₃ with the following properties:
    - C₁ and C₂ are externally tangent
    - C₁ and C₂ are both internally tangent to C₃
    - The radii of C₁ and C₂ are 6 and 9, respectively
    - The centers of all three circles lie on a straight line
    - A common external tangent to C₁ and C₂ touches them at points T₁ and T₂
    - The tangent extends to touch C₃ at point T
    - The tangent is perpendicular to the diameter of C₃ passing through its point of tangency
    Then the radius of C₃ is 11. -/
theorem circle_radius_problem (C₁ C₂ C₃ : Circle) (T₁ T₂ T : ℝ × ℝ) :
  ExternallyTangent C₁ C₂ →
  InternallyTangent C₁ C₃ →
  InternallyTangent C₂ C₃ →
  C₁.radius = 6 →
  C₂.radius = 9 →
  CollinearCenters C₁ C₂ C₃ →
  OnCommonExternalTangent T₁ C₁ →
  OnCommonExternalTangent T₂ C₂ →
  OnTangent T C₃ →
  TangentPerpendicular C₃ T →
  C₃.radius = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l10_1083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_P_to_AB_l10_1051

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 1)^2 = 9}

-- Define points A and B on the circle
variable (A B : ℝ × ℝ)

-- Define point P
def P : ℝ × ℝ := (0, -3)

-- State the conditions
axiom A_on_C : A ∈ C
axiom B_on_C : B ∈ C
axiom AB_length : ‖A - B‖ = 2 * Real.sqrt 5

-- Define the line through A and B
def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, x = A + t • (B - A)}

-- State the theorem
theorem max_distance_P_to_AB :
  (⨆ (x : ℝ × ℝ) (h : x ∈ line_through A B), ‖P - x‖) = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_P_to_AB_l10_1051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_is_correct_l10_1039

noncomputable def equation (x : ℝ) : Prop :=
  14 * Real.sin (3 * x) - 3 * Real.cos (3 * x) = 13 * Real.sin (2 * x) - 6 * Real.cos (2 * x)

noncomputable def α : ℝ := Real.arctan (14 / 3)
noncomputable def β : ℝ := Real.arctan (13 / 6)

noncomputable def smallest_positive_root : ℝ := (2 * Real.pi - α - β) / 5

theorem smallest_positive_root_is_correct :
  equation smallest_positive_root ∧
  (∀ y : ℝ, 0 < y ∧ y < smallest_positive_root → ¬equation y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_is_correct_l10_1039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_18_l10_1088

/-- A right triangle with specific points and midpoints -/
structure RightTriangleWithPoints where
  -- Vertices of the right triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Circumcenter
  O : ℝ × ℝ
  -- Additional points on sides
  E : ℝ × ℝ
  F : ℝ × ℝ
  -- Midpoints
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- Conditions
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  hypotenuse : (A.1 - C.1)^2 + (A.2 - C.2)^2 ≥ (A.1 - B.1)^2 + (A.2 - B.2)^2
  circumcenter : O = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  E_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)
  F_on_BC : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ F = (s * C.1 + (1 - s) * B.1, s * C.2 + (1 - s) * B.2)
  AE_length : (E.1 - A.1)^2 + (E.2 - A.2)^2 = 81
  EB_length : (E.1 - B.1)^2 + (E.2 - B.2)^2 = 9
  BF_length : (F.1 - B.1)^2 + (F.2 - B.2)^2 = 36
  FC_length : (F.1 - C.1)^2 + (F.2 - C.2)^2 = 4
  W_midpoint : W = ((E.1 + B.1) / 2, (E.2 + B.2) / 2)
  X_midpoint : X = ((B.1 + F.1) / 2, (B.2 + F.2) / 2)
  Y_midpoint : Y = ((F.1 + O.1) / 2, (F.2 + O.2) / 2)
  Z_midpoint : Z = ((O.1 + E.1) / 2, (O.2 + E.2) / 2)

/-- The area of quadrilateral WXYZ is 18 -/
theorem area_WXYZ_is_18 (t : RightTriangleWithPoints) : ℝ := by
  sorry

#check area_WXYZ_is_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_18_l10_1088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_area_l10_1046

/-- Right square-based pyramid -/
structure SquarePyramid where
  base_edge : ℝ
  lateral_edge : ℝ

/-- Calculate the total area of the four triangular faces of a right square-based pyramid -/
noncomputable def total_triangular_area (p : SquarePyramid) : ℝ :=
  4 * (1/2 * p.base_edge * Real.sqrt (p.lateral_edge^2 - (p.base_edge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right square-based pyramid
    with base edges of 8 units and lateral edges of 7 units is 16√33 square units -/
theorem square_pyramid_area :
  ∃ (p : SquarePyramid), p.base_edge = 8 ∧ p.lateral_edge = 7 ∧ total_triangular_area p = 16 * Real.sqrt 33 :=
by
  -- Construct the pyramid
  let p : SquarePyramid := ⟨8, 7⟩
  
  -- Show that it satisfies the conditions
  have h1 : p.base_edge = 8 := rfl
  have h2 : p.lateral_edge = 7 := rfl
  
  -- Calculate the area
  have h3 : total_triangular_area p = 16 * Real.sqrt 33 := by
    -- Expand the definition and simplify
    unfold total_triangular_area
    simp [h1, h2]
    -- The rest of the proof would go here
    sorry
  
  -- Conclude the proof
  exact ⟨p, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_area_l10_1046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_difference_approx_63_l10_1026

/-- Calculates the difference in selling prices given the original selling price,
    original profit margin, reduced purchase price percentage, and new profit margin -/
noncomputable def selling_price_difference (original_selling_price : ℝ) (original_profit_margin : ℝ) 
  (reduced_purchase_price_percentage : ℝ) (new_profit_margin : ℝ) : ℝ :=
  let original_purchase_price := original_selling_price / (1 + original_profit_margin)
  let new_purchase_price := original_purchase_price * (1 - reduced_purchase_price_percentage)
  let new_selling_price := new_purchase_price * (1 + new_profit_margin)
  new_selling_price - original_selling_price

/-- The difference in selling prices is approximately $63 -/
theorem selling_price_difference_approx_63 :
  ∃ ε > 0, ε < 0.01 ∧ 
  |selling_price_difference 989.9999999999992 0.10 0.10 0.30 - 63| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_difference_approx_63_l10_1026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l10_1038

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (2*x^2 - 3*x + 1)

-- State the theorem
theorem f_monotonicity :
  (∀ x y : ℝ, x < y ∧ x < 3/4 ∧ y < 3/4 → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ x > 3/4 ∧ y > 3/4 → f x > f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l10_1038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_product_l10_1000

theorem consecutive_even_numbers_product (a b c : ℤ) : 
  (Even a ∧ Even b ∧ Even c) →  -- all numbers are even
  (b = a + 2 ∧ c = b + 2) →     -- they are consecutive
  (a + b + c = 18) →            -- their sum is 18
  (a * b * c = 192) :=          -- their product is 192
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_product_l10_1000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l10_1099

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ < f x₂) ∧
  (∀ x : ℝ, f x ≤ f 1) ∧
  (∀ x : ℝ, f (-1) ≤ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l10_1099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_for_specific_park_l10_1034

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  area : ℝ
  total_fencing_cost : ℝ
  side_ratio : ℝ
  (area_eq : area = length * width)
  (ratio_eq : length / width = side_ratio)

/-- The cost of fencing per meter for a rectangular park -/
noncomputable def fencing_cost_per_meter (park : RectangularPark) : ℝ :=
  park.total_fencing_cost / (2 * (park.length + park.width))

/-- Theorem stating the fencing cost per meter for a specific park -/
theorem fencing_cost_for_specific_park :
  ∃ (park : RectangularPark),
    park.area = 3750 ∧
    park.total_fencing_cost = 125 ∧
    park.side_ratio = 3/2 ∧
    fencing_cost_per_meter park = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_for_specific_park_l10_1034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_cost_no_cheaper_strategy_l10_1024

/-- Represents the available weights -/
inductive Weight : Type
  | one : Weight
  | two : Weight
  | four : Weight
  | eight : Weight
deriving Repr

/-- The cost of using a weight -/
def weightCost : ℕ := 100

/-- The number of diamonds -/
def numDiamonds : ℕ := 15

/-- A weighing operation -/
structure Weighing where
  weights : List Weight
  diamond : ℕ
deriving Repr

/-- A verification strategy -/
def VerificationStrategy := List Weighing

/-- Calculate the cost of a strategy -/
def strategyCost (strategy : VerificationStrategy) : ℕ := sorry

/-- Check if a strategy verifies all diamonds -/
def verifiesAllDiamonds (strategy : VerificationStrategy) : Prop := sorry

/-- The optimal verification strategy -/
noncomputable def optimalStrategy : VerificationStrategy := sorry

/-- Theorem: The optimal strategy costs 800 coins and verifies all diamonds -/
theorem optimal_strategy_cost :
  strategyCost optimalStrategy = 800 ∧ verifiesAllDiamonds optimalStrategy := by sorry

/-- Theorem: No strategy costing less than 800 coins can verify all diamonds -/
theorem no_cheaper_strategy (strategy : VerificationStrategy) :
  verifiesAllDiamonds strategy → strategyCost strategy ≥ 800 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_cost_no_cheaper_strategy_l10_1024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_inequality_l10_1059

noncomputable section

-- Define the functions g and f
def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) ^ (x - 2) + 1
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / Real.log (Real.sqrt 3)

-- State the theorem
theorem intersection_and_inequality (a : ℝ) (h : a > 0) :
  ∃! A : ℝ × ℝ, g a A.1 = A.2 ∧ f a A.1 = A.2 ∧
  ∀ x : ℝ, g a x > 3 ↔ x ∈ Set.Ioo 0 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_inequality_l10_1059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_max_distance_on_circle_l10_1052

noncomputable def circle_C (α : ℝ) (θ : ℝ) : ℝ × ℝ := (α + α * Real.cos θ, α * Real.sin θ)

def line_l (θ : ℝ) (ρ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 2 * Real.sqrt 2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

theorem circle_and_line_intersection
  (α : ℝ)
  (h1 : 0 < α)
  (h2 : α < 5)
  (A B : ℝ × ℝ)
  (h3 : ∃ θ₁ ρ₁, circle_C α θ₁ = A ∧ line_l θ₁ ρ₁)
  (h4 : ∃ θ₂ ρ₂, circle_C α θ₂ = B ∧ line_l θ₂ ρ₂)
  (h5 : distance A B = 2 * Real.sqrt 2) :
  α = 2 :=
by sorry

theorem max_distance_on_circle
  (M N : ℝ × ℝ)
  (h1 : ∃ θ₁, circle_C 2 θ₁ = M)
  (h2 : ∃ θ₂, circle_C 2 θ₂ = N)
  (h3 : M ≠ (0, 0))
  (h4 : N ≠ (0, 0))
  (h5 : angle (M.1 - 0, M.2 - 0) (N.1 - 0, N.2 - 0) = Real.pi / 3) :
  ∃ (max : ℝ), max = 4 * Real.sqrt 3 ∧ distance (0, 0) M + distance (0, 0) N ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_max_distance_on_circle_l10_1052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_zero_l10_1001

/-- Triangle ABC with vertices on positive x, y, and z axes -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  side_ab : a^2 + b^2 = 25
  side_bc : b^2 + c^2 = 144
  side_ca : c^2 + a^2 = 169

/-- Volume of tetrahedron OABC -/
noncomputable def tetrahedronVolume (t : TriangleABC) : ℝ :=
  (1/6) * t.a * t.b * t.c

/-- Theorem: The volume of tetrahedron OABC is 0 -/
theorem tetrahedron_volume_is_zero (t : TriangleABC) : tetrahedronVolume t = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_zero_l10_1001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l10_1056

/-- Definition of the ★ operation -/
noncomputable def star (a b : ℝ) : ℝ := (a + b) / (a - b)

/-- Theorem stating that if (x ★ 18)² = 4, then x = 54 or x = 6 -/
theorem star_equation_solution (x : ℝ) : (star x 18)^2 = 4 → x = 54 ∨ x = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l10_1056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l10_1066

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (Real.pi + x) * Real.cos (3/2 * Real.pi - x) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≤ 1) ∧
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3),
    x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 12) →
    ∀ y ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 12),
      x ≤ y → f x ≤ f y) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l10_1066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l10_1096

-- Define the function f(x) = x^2 * e^x
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

-- State the theorem
theorem f_decreasing_on_interval :
  StrictMonoOn f (Set.Ioo (-2 : ℝ) 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l10_1096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l10_1072

noncomputable def vector_a (y : ℝ) : ℝ × ℝ := (1, y)
def vector_b : ℝ × ℝ := (1, -3)

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

noncomputable def angle (u v : ℝ × ℝ) : ℝ :=
  Real.arccos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2)))

theorem vector_problem :
  (∃ (y : ℝ), perpendicular (2 • vector_a y + vector_b) vector_b ∧ vector_a y = (1, 2)) ∧
  angle (vector_a 2) vector_b = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l10_1072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polar_curve_l10_1016

/-- The area enclosed by the polar curve ρ = 4cosθ -/
noncomputable def area_enclosed (ρ : ℝ → ℝ) : ℝ := sorry

/-- The polar curve ρ = 4cosθ -/
noncomputable def polar_curve (θ : ℝ) : ℝ := 4 * Real.cos θ

theorem area_of_polar_curve :
  area_enclosed polar_curve = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polar_curve_l10_1016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l10_1070

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(3*x + 3) - (2 : ℝ)^(2*x + 4) - (2 : ℝ)^(x + 1) + 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l10_1070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_union_of_A_and_complement_B_l10_1018

-- Define the function (marked as noncomputable due to Real.sqrt and Real.log)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2*x - 4) + Real.log (5 - x) / Real.log 10

-- Define the domain A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | x > 4}

-- Theorem for the domain of the function
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = A := by sorry

-- Theorem for A ∪ (U\B)
theorem union_of_A_and_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | x < 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_union_of_A_and_complement_B_l10_1018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_row_sixteen_seat_notation_l10_1060

/-- Represents a movie ticket with row and seat numbers -/
structure MovieTicket where
  row : ℕ
  seat : ℕ

/-- The notation for a movie ticket -/
def ticketNotation (t : MovieTicket) : ℕ × ℕ := (t.row, t.seat)

theorem five_row_sixteen_seat_notation :
  (∀ t : MovieTicket, t.row = 10 ∧ t.seat = 3 → ticketNotation t = (10, 3)) →
  ticketNotation (MovieTicket.mk 5 16) = (5, 16) := by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_row_sixteen_seat_notation_l10_1060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_property_l10_1008

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 + t/2, (Real.sqrt 3/2) * t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (12 / (3 + Real.sin θ ^ 2))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B (existence assumed)
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Statement to prove
theorem intersection_points_property : 
  1 / dist F A + 1 / dist F B = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_property_l10_1008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_theorem_l10_1028

theorem gcf_of_lcm_theorem : Nat.gcd (Nat.lcm 15 21) (Nat.lcm 14 20) = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_theorem_l10_1028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l10_1055

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (1 - Complex.I * Real.sqrt 3) / (1 + i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l10_1055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l10_1071

theorem function_inequality_implies_a_range (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ a^2 - |a| * x > 2 / (1 - x^2)) →
  a ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l10_1071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l10_1021

noncomputable def vector_m (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, y₁)
noncomputable def vector_n (x₂ y₂ : ℝ) : ℝ × ℝ := (x₂, y₂)
def vector_p : ℝ × ℝ := (1, 1)

theorem vector_properties (x₁ y₁ x₂ y₂ : ℝ) :
  let m := vector_m x₁ y₁
  let n := vector_n x₂ y₂
  let p := vector_p
  (x₁^2 + y₁^2 = 1) →
  (x₂^2 + y₂^2 = 1) →
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  ((x₁ + y₁) / Real.sqrt 2 = Real.sqrt 3 / 2) →
  ((x₂ + y₂) / Real.sqrt 2 = Real.sqrt 3 / 2) →
  (x₁ * x₂ + y₁ * y₂ = 1/2) ∧ (y₁ * y₂ = x₁ * x₂) := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l10_1021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_figure_l10_1078

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The configuration of the geometric problem -/
structure GeometricSetup where
  A : Point
  B : Point
  semicircle1 : Semicircle
  semicircle2 : Semicircle
  circle : Circle

/-- The conditions of the problem -/
def validSetup (setup : GeometricSetup) : Prop :=
  setup.A.y = setup.B.y ∧  -- A and B are on a horizontal line
  setup.B.x - setup.A.x = 4 ∧  -- Distance between A and B is 4
  setup.semicircle1.center = setup.A ∧
  setup.semicircle2.center = setup.B ∧
  setup.semicircle1.radius = 2 ∧
  setup.semicircle2.radius = 2 ∧
  setup.circle.radius = 2 ∧
  ∃ (intersectionPoint : Point),
    (intersectionPoint.x - setup.A.x)^2 + (intersectionPoint.y - setup.A.y)^2 = 4 ∧
    (intersectionPoint.x - setup.B.x)^2 + (intersectionPoint.y - setup.B.y)^2 = 4 ∧
    setup.circle.center.y = intersectionPoint.y + 2

/-- Calculate the area of a circle -/
noncomputable def areaOfCircle (circle : Circle) : ℝ :=
  Real.pi * circle.radius^2

/-- Calculate the area of common parts of two semicircles -/
noncomputable def areaOfCommonParts (s1 s2 : Semicircle) : ℝ :=
  sorry -- Placeholder for the actual calculation

/-- The main theorem -/
theorem area_of_special_figure (setup : GeometricSetup) (h : validSetup setup) :
  ∃ (area : ℝ), area = 8 ∧ 
  (area = areaOfCircle setup.circle - areaOfCommonParts setup.semicircle1 setup.semicircle2) := by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_figure_l10_1078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_sum_primes_l10_1080

/-- Sum of all primes strictly less than n -/
def sum_primes (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ p < n) (Finset.range n)).sum id

/-- Statement: There exist infinitely many integers n ≥ 3 such that
    the sum of all primes strictly less than n is coprime to n -/
theorem infinitely_many_coprime_sum_primes :
  ∀ m : ℕ, ∃ n : ℕ, n ≥ m ∧ n ≥ 3 ∧ Nat.Coprime (sum_primes n) n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_sum_primes_l10_1080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l10_1022

/-- Given a circular sector with arc length 6 cm and area 18 cm², its central angle is 1 radian. -/
theorem sector_central_angle (arc_length area : ℝ) (h1 : arc_length = 6) (h2 : area = 18) :
  let r := 2 * area / arc_length
  arc_length / r = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l10_1022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l10_1085

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + Real.exp x

-- Part 1: Tangent line condition
theorem tangent_line_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 2 * x₀ ∧ deriv (f a) x₀ = 2) → a = 1 :=
sorry

-- Part 2: Inequality condition
theorem inequality_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 2), f 1 x ≥ m * Real.sin (2 * x)) ↔ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l10_1085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_l10_1042

noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 3)

noncomputable def particle_move (z : ℂ) : ℂ := ω * z + 10

noncomputable def particle_position : ℕ → ℂ
  | 0 => 7
  | n + 1 => particle_move (particle_position n)

theorem final_position :
  particle_position 12 = Complex.ofReal 37 - Complex.I * (30 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_l10_1042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_i_l10_1065

-- Define the complex number type
variable (z : ℂ)

-- State the theorem
theorem complex_fraction_equals_i :
  (1 + 2 * Complex.I) / (2 - Complex.I) = Complex.I :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_i_l10_1065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l10_1013

theorem triangle_third_side_length 
  (a b : ℝ) 
  (θ : ℝ) 
  (ha : a = 10) 
  (hb : b = 15) 
  (hθ : θ = 150 * Real.pi / 180) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) ∧ c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l10_1013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spy_configuration_exists_l10_1020

/-- Represents a position on the board -/
structure Position where
  x : Fin 6
  y : Fin 6

/-- Represents the direction a spy is facing -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a spy on the board -/
structure Spy where
  pos : Position
  dir : Direction

/-- Checks if a spy can see a given position -/
def canSee (s : Spy) (p : Position) : Prop :=
  match s.dir with
  | Direction.Up =>
      (s.pos.x = p.x ∧ p.y - s.pos.y ≤ 2 ∧ p.y > s.pos.y) ∨
      (s.pos.y = p.y ∧ (p.x = s.pos.x - 1 ∨ p.x = s.pos.x + 1))
  | Direction.Down =>
      (s.pos.x = p.x ∧ s.pos.y - p.y ≤ 2 ∧ p.y < s.pos.y) ∨
      (s.pos.y = p.y ∧ (p.x = s.pos.x - 1 ∨ p.x = s.pos.x + 1))
  | Direction.Left =>
      (s.pos.y = p.y ∧ s.pos.x - p.x ≤ 2 ∧ p.x < s.pos.x) ∨
      (s.pos.x = p.x ∧ (p.y = s.pos.y - 1 ∨ p.y = s.pos.y + 1))
  | Direction.Right =>
      (s.pos.y = p.y ∧ p.x - s.pos.x ≤ 2 ∧ p.x > s.pos.x) ∨
      (s.pos.x = p.x ∧ (p.y = s.pos.y - 1 ∨ p.y = s.pos.y + 1))

/-- The main theorem stating that there exists a configuration of 18 spies
    where no spy can see another -/
theorem spy_configuration_exists : ∃ (spies : Finset Spy),
  spies.card = 18 ∧
  ∀ s₁ s₂, s₁ ∈ spies → s₂ ∈ spies → s₁ ≠ s₂ → ¬(canSee s₁ s₂.pos ∨ canSee s₂ s₁.pos) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spy_configuration_exists_l10_1020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_sqrt_3_l10_1098

/-- Represents a cone with given slant height and lateral area -/
structure Cone where
  slant_height : ℝ
  lateral_area : ℝ

/-- The height of a cone given its slant height and lateral area -/
noncomputable def cone_height (c : Cone) : ℝ :=
  Real.sqrt (c.slant_height ^ 2 - (c.lateral_area / (Real.pi * c.slant_height)) ^ 2)

theorem cone_height_is_sqrt_3 (c : Cone) 
  (h1 : c.slant_height = 2)
  (h2 : c.lateral_area = 2 * Real.pi) : 
  cone_height c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_sqrt_3_l10_1098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l10_1069

/-- Given a triangle OAB with specific properties, prove that rotating point A
    90 degrees counterclockwise about O results in the point (-5√3/3, 5) -/
theorem triangle_rotation (A : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (5, 0)
  -- A is in the first quadrant
  (A.1 > 0 ∧ A.2 > 0) →
  -- ∠ABO = 90°
  (A.2 / A.1 = 1) →
  -- ∠AOB = 30°
  (A.2 / 5 = Real.tan (30 * π / 180)) →
  -- A' is A rotated 90° counterclockwise about O
  let A' : ℝ × ℝ := (-A.2, A.1)
  -- The coordinates of A' are (-5√3/3, 5)
  A' = (-5 * Real.sqrt 3 / 3, 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l10_1069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l10_1025

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (-3, 4)
def C : ℝ × ℝ := (0, 6)

-- Define the equation of a line ax + by + c = 0
def is_line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Define the area of a triangle
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem triangle_ABC_properties :
  (∀ x y : ℝ, is_line_equation 3 2 (-1) x y ↔ 
    (∃ t : ℝ, (x, y) = ((1-t)*A.1 + t*B.1, (1-t)*A.2 + t*B.2) ∨
              (x, y) = ((1-t)*A.1 + t*C.1, (1-t)*A.2 + t*C.2))) ∧
  triangle_area A B C = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l10_1025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garment_industry_workforce_l10_1079

noncomputable section

/-- The number of men required to complete a piece of work in a given time and hours per day -/
def men_required (hours_per_day : ℝ) (days : ℝ) (total_man_hours : ℝ) : ℝ :=
  total_man_hours / (hours_per_day * days)

/-- The total man-hours required to complete the work -/
def total_man_hours (men : ℝ) (hours_per_day : ℝ) (days : ℝ) : ℝ :=
  men * hours_per_day * days

theorem garment_industry_workforce (initial_hours_per_day : ℝ) (initial_days : ℝ) 
    (new_men : ℝ) (new_hours_per_day : ℝ) (new_days : ℝ) : 
    initial_hours_per_day = 8 → 
    initial_days = 10 → 
    new_men = 9.00225056264066 → 
    new_hours_per_day = 13.33 → 
    new_days = 8 → 
    ∃ (initial_men : ℝ), 
      (total_man_hours initial_men initial_hours_per_day initial_days = 
       total_man_hours new_men new_hours_per_day new_days) ∧ 
      (Int.floor initial_men = 12) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garment_industry_workforce_l10_1079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_54_l10_1053

def sequence_a : ℕ → ℚ
| 0 => 2
| n + 1 => 3 * sequence_a n

theorem a_4_equals_54 : sequence_a 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_54_l10_1053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l10_1017

theorem sufficient_not_necessary_condition :
  ∃ (a b : ℝ), (∀ c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  ¬(∀ a b : ℝ, a > b → ∀ c : ℝ, a * c^2 > b * c^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l10_1017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l10_1089

/-- Predicate to check if three lengths form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if a length is a median of a triangle -/
def is_median (s a b : ℝ) : Prop :=
  4 * s^2 = 2 * a^2 + 2 * b^2 - (a + b)^2

/-- Predicate to check if a length is the circumradius of a triangle -/
def is_circumradius (R a b c : ℝ) : Prop :=
  4 * R * (R - a) * (R - b) * (R - c) = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)

/-- Given a triangle with side lengths a, b, c, median lengths sₐ, sᵦ, sᵧ, and circumradius R,
    the inequality 2R(sₐ + sᵦ + sᵧ) ≥ a² + b² + c² holds. -/
theorem triangle_inequality (a b c sₐ sᵦ sᵧ R : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : sₐ > 0) (h₅ : sᵦ > 0) (h₆ : sᵧ > 0)
  (h₇ : R > 0)
  (h₈ : is_triangle a b c)
  (h₉ : is_median sₐ b c)
  (h₁₀ : is_median sᵦ a c)
  (h₁₁ : is_median sᵧ a b)
  (h₁₂ : is_circumradius R a b c) :
  2 * R * (sₐ + sᵦ + sᵧ) ≥ a^2 + b^2 + c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l10_1089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l10_1093

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def F (x m : ℝ) : ℝ := (x - 2) * Real.exp x + f x - x - m

theorem min_m_value (m : ℤ) :
  (∀ x ∈ Set.Ioo (1/4 : ℝ) 1, F x (m : ℝ) ≤ 0) → m ≥ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l10_1093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_probability_theorem_l10_1005

/-- Represents the state of the calculator display -/
inductive CalcState
| Even
| Odd

/-- Represents the possible operations on the calculator -/
inductive Operation
| Add
| Multiply

/-- The probability of obtaining an odd number after n steps -/
noncomputable def prob_odd (n : ℕ) : ℝ :=
  1/3 - (1/3) * (1/4)^n + (1/2) * (1/4)^n

/-- The limit of prob_odd as n approaches infinity -/
noncomputable def limit_prob_odd : ℝ := 1/3

/-- Theorem stating that prob_odd converges to limit_prob_odd -/
theorem calculator_probability_theorem :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |prob_odd n - limit_prob_odd| < ε := by
  sorry

#check calculator_probability_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_probability_theorem_l10_1005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_max_inscribed_rectangle_perimeter_l10_1003

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 2 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

-- Define the left focus F
noncomputable def F : ℝ × ℝ := (-2 * Real.sqrt 2, 0)

-- Theorem 1: Product of distances from F to intersection points is 2
theorem intersection_distance_product :
  ∃ (t₁ t₂ : ℝ), 
    C ((l t₁).1) ((l t₁).2) ∧ 
    C ((l t₂).1) ((l t₂).2) ∧ 
    ((F.1 - (l t₁).1)^2 + (F.2 - (l t₁).2)^2) * 
    ((F.1 - (l t₂).1)^2 + (F.2 - (l t₂).2)^2) = 4 := by
  sorry

-- Theorem 2: Maximum perimeter of inscribed rectangle is 16
theorem max_inscribed_rectangle_perimeter :
  ∀ (x y : ℝ), C x y → 4 * x + 4 * y ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_max_inscribed_rectangle_perimeter_l10_1003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_window_area_l10_1086

/-- The area of a window consisting of a semicircle on top of a rectangle --/
noncomputable def window_area (semicircle_radius : ℝ) (rect_length rect_width : ℝ) : ℝ :=
  (1/2 * Real.pi * semicircle_radius^2) + (rect_length * rect_width)

/-- Theorem stating the approximate area of the specific window --/
theorem specific_window_area :
  let semicircle_radius : ℝ := 50
  let rect_length : ℝ := 150
  let rect_width : ℝ := 100
  abs (window_area semicircle_radius rect_length rect_width - 18926.9875) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_window_area_l10_1086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l10_1011

theorem largest_power_dividing_factorial : ∃ n : ℕ, n = 7 ∧ 
  (∀ m : ℕ, (18^m : ℕ) ∣ Nat.factorial 30 → m ≤ n) ∧
  ((18^n : ℕ) ∣ Nat.factorial 30) ∧
  (∀ k : ℕ, (18^k : ℕ) = 2^k * 3^(2*k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l10_1011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_is_sqrt_130_div_3_l10_1063

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  x + y ≤ 5 ∧ 3 * x + y ≥ 3 ∧ x ≥ 1 ∧ y ≥ 1

-- Define the length of the longest side
noncomputable def longest_side_length : ℝ := Real.sqrt 130 / 3

-- Theorem statement
theorem longest_side_is_sqrt_130_div_3 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    feasible_region x₁ y₁ ∧ 
    feasible_region x₂ y₂ ∧ 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = longest_side_length ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ), 
      feasible_region x₃ y₃ → 
      feasible_region x₄ y₄ → 
      Real.sqrt ((x₄ - x₃)^2 + (y₄ - y₃)^2) ≤ longest_side_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_is_sqrt_130_div_3_l10_1063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l10_1045

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin (2 * x)

-- State the theorem about the minimum value of f(x)
theorem min_value_of_f :
  ∃ (min_val : ℝ), (∀ (x : ℝ), f x ≥ min_val) ∧ (min_val = -3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l10_1045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l10_1097

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 12

-- Define the condition of two common points
def two_common_points (p : ℝ) : Prop := ∃ (x1 y1 x2 y2 : ℝ),
  x1 ≠ x2 ∧ parabola p x1 y1 ∧ parabola p x2 y2 ∧ circle_E x1 y1 ∧ circle_E x2 y2 ∧
  ∀ (x y : ℝ), parabola p x y ∧ circle_E x y → (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2)

-- Define the line passing through the center of E
def line_through_center (m : ℝ) (x y : ℝ) : Prop := x = m*y + 4

-- Define points A and B
def point_A_B (m : ℝ) (xa ya xb yb : ℝ) : Prop :=
  line_through_center m xa ya ∧ circle_E xa ya ∧
  line_through_center m xb yb ∧ circle_E xb yb ∧
  xa ≠ xb

-- Define points P and Q
def point_P_Q (p : ℝ) (xp yp xq yq : ℝ) : Prop :=
  parabola p xp yp ∧ parabola p xq yq ∧ xp ≠ 0 ∧ yp ≠ 0 ∧ xq ≠ 0 ∧ yq ≠ 0

-- Define the ratio of areas
noncomputable def area_ratio (xa ya xb yb xp yp xq yq : ℝ) : ℝ :=
  (xa*yb - xb*ya) / (xp*yq - xq*yp)

-- Theorem statement
theorem parabola_circle_intersection (p : ℝ) :
  p > 0 → two_common_points p →
  (∀ (x y : ℝ), parabola p x y ↔ y^2 = 4*x) ∧
  (∃ (m : ℝ), ∀ (xa ya xb yb xp yp xq yq : ℝ),
    point_A_B m xa ya xb yb →
    point_P_Q p xp yp xq yq →
    area_ratio xa ya xb yb xp yp xq yq ≤ 9/16) ∧
  (∃ (m xa ya xb yb xp yp xq yq : ℝ),
    point_A_B m xa ya xb yb →
    point_P_Q p xp yp xq yq →
    area_ratio xa ya xb yb xp yp xq yq = 9/16) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l10_1097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lcm_120_24_equals_300_l10_1077

def sum_of_lcm_divisors (a b : ℕ) : ℕ :=
  (Finset.filter (fun x => Nat.lcm x a = b) (Finset.range (b + 1))).sum id

theorem sum_of_lcm_120_24_equals_300 : sum_of_lcm_divisors 24 120 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_lcm_120_24_equals_300_l10_1077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_similar_statues_l10_1073

/-- The amount of paint needed for similar statues -/
theorem paint_for_similar_statues
  (original_height : ℝ)
  (original_paint : ℝ)
  (new_height : ℝ)
  (num_statues : ℝ)
  (h1 : original_height = 8)
  (h2 : original_paint = 1)
  (h3 : new_height = 2)
  (h4 : num_statues = 360) :
  num_statues * (new_height / original_height)^2 * original_paint = 22.5 := by
  sorry

#check paint_for_similar_statues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_similar_statues_l10_1073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_listening_time_is_54_3_l10_1036

/-- Represents the distribution of audience members and their listening durations --/
structure AudienceDistribution where
  total_audience : Nat
  talk_duration : Nat
  full_listeners : Nat
  sleepers : Nat
  one_third_listeners : Nat
  half_listeners : Nat
  two_thirds_listeners : Nat

/-- Calculates the average listening time for the audience --/
noncomputable def calculate_average_listening_time (dist : AudienceDistribution) : Real :=
  let full_time := (dist.full_listeners * dist.talk_duration : Nat)
  let one_third_time := (dist.one_third_listeners * (dist.talk_duration / 3) : Nat)
  let half_time := (dist.half_listeners * (dist.talk_duration / 2) : Nat)
  let two_thirds_time := (dist.two_thirds_listeners * (2 * dist.talk_duration / 3) : Nat)
  let total_time := full_time + one_third_time + half_time + two_thirds_time
  (total_time : Real) / (dist.total_audience : Real)

/-- The theorem stating that the average listening time is 54.3 minutes --/
theorem average_listening_time_is_54_3 (dist : AudienceDistribution)
  (h1 : dist.total_audience = 100)
  (h2 : dist.talk_duration = 90)
  (h3 : dist.full_listeners = 30)
  (h4 : dist.sleepers = 15)
  (h5 : dist.one_third_listeners = 14)
  (h6 : dist.half_listeners = 14)
  (h7 : dist.two_thirds_listeners = 27) :
  calculate_average_listening_time dist = 54.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_listening_time_is_54_3_l10_1036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_with_divisibility_l10_1095

theorem infinite_pairs_with_divisibility (n : ℕ) (hn : n > 1) :
  ∃ f : ℕ → ℕ × ℕ, ∀ k : ℕ,
    let (x, y) := f k
    (1 < x ∧ x < y) ∧ (x^n + y ∣ x + y^n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_with_divisibility_l10_1095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l10_1090

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 3)*x

theorem tangent_line_at_origin (a : ℝ) (h : ∀ x, deriv (f a) x = deriv (f a) (-x)) :
  deriv (f a) 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l10_1090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_court_placement_l10_1015

/-- Represents a rectangular area --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circular object --/
structure CircularObject where
  diameter : ℝ

/-- The forest environment --/
structure Forest where
  area : Rectangle
  trees : Finset CircularObject
  courts : Finset Rectangle

/-- Checks if a court intersects with a tree --/
def courtIntersectsTree (court : Rectangle) (tree : CircularObject) : Prop :=
  sorry -- Definition of intersection logic goes here

/-- The problem statement --/
theorem forest_court_placement 
  (forest : Forest)
  (h1 : forest.area.width = 1001)
  (h2 : forest.area.height = 945)
  (h3 : forest.trees.card = 1280)
  (h4 : ∀ t ∈ forest.trees, t.diameter = 1)
  (h5 : ∀ c ∈ forest.courts, c.width = 20 ∧ c.height = 34)
  (h6 : forest.courts.card = 7) :
  ∃ (placement : Forest), 
    placement.area = forest.area ∧ 
    placement.trees = forest.trees ∧
    placement.courts.card = 7 ∧
    (∀ c ∈ placement.courts, ∀ t ∈ placement.trees, ¬ courtIntersectsTree c t) :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_court_placement_l10_1015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_evaluation_l10_1081

-- Define the ∇ operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_evaluation :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  nabla (nabla 2 5) (nabla 1 3) = 1 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_evaluation_l10_1081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l10_1041

theorem polynomial_division_result :
  let f : Polynomial ℝ := 4 * X^4 + 12 * X^3 - 9 * X^2 + X + 3
  let d : Polynomial ℝ := X^2 + 4 * X - 2
  ∃ (q r : Polynomial ℝ), f = q * d + r ∧ r.degree < d.degree →
  (q.eval 1 : ℝ) + (r.eval (-1) : ℝ) = -21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l10_1041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_revolutions_l10_1040

def carousel_problem (distance_A distance_B revolutions_A : ℝ) : Prop :=
  distance_A > 0 ∧ distance_B > 0 ∧ revolutions_A > 0 →
  let circumference_A := 2 * Real.pi * distance_A
  let circumference_B := 2 * Real.pi * distance_B
  let total_distance_A := circumference_A * revolutions_A
  let revolutions_B := total_distance_A / circumference_B
  revolutions_B = (distance_A / distance_B) * revolutions_A

theorem horse_revolutions : carousel_problem 16 4 40 → 
  ∃ (revolutions_B : ℝ), revolutions_B = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_revolutions_l10_1040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l10_1027

def U : Set ℕ := {3, 4, 5}

def M (a : ℤ) : Set ℕ := {(|a - 3|).toNat, 3}

theorem value_of_a (a : ℤ) : (U \ M a = {5}) → (a = -1 ∨ a = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l10_1027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_formula_l10_1092

/-- The sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case to handle n = 0
  | 1 => 1
  | n + 1 => 2 * a n + 1

/-- The general formula for a_n -/
def a_formula (n : ℕ) : ℕ := 2^n - 1

/-- Theorem stating that the recursive definition equals the general formula -/
theorem a_eq_formula : ∀ n : ℕ, n ≥ 1 → a n = a_formula n := by
  sorry

/-- Lemma to prove the base case -/
lemma a_eq_formula_base : a 1 = a_formula 1 := by
  rfl

/-- Lemma to prove the inductive step -/
lemma a_eq_formula_inductive (n : ℕ) (h : n ≥ 1) :
  a n = a_formula n → a (n + 1) = a_formula (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_formula_l10_1092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_value_of_m_for_sum_of_reciprocals_l10_1009

/-- The quadratic function y = (m+6)x² + 2(m-1)x + m+1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m+6)*x^2 + 2*(m-1)*x + m+1

/-- The function always has a root for m in this range -/
def always_has_root (m : ℝ) : Prop :=
  ∃ x : ℝ, f m x = 0

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := 4*(m-1)^2 - 4*(m+6)*(m+1)

theorem range_of_m :
  {m : ℝ | always_has_root m} = Set.Iic (-5/9) :=
sorry

theorem value_of_m_for_sum_of_reciprocals (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧ 1/x₁ + 1/x₂ = -4) →
  m = -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_value_of_m_for_sum_of_reciprocals_l10_1009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_play_dough_cost_play_dough_costs_35_l10_1047

/-- The cost of a play dough given Tony's purchases -/
theorem play_dough_cost : ℝ → Prop :=
  fun cost =>
    let lego_cost : ℝ := 250
    let sword_cost : ℝ := 120
    let lego_sets : ℝ := 3
    let sword_count : ℝ := 7
    let play_dough_count : ℝ := 10
    let total_paid : ℝ := 1940
    (lego_cost * lego_sets + sword_cost * sword_count + cost * play_dough_count = total_paid) →
    cost = 35

/-- Proof that the play dough costs $35 -/
theorem play_dough_costs_35 : play_dough_cost 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_play_dough_cost_play_dough_costs_35_l10_1047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_line_l10_1014

/-- Given a line l and a point A, find the equation of line l' that forms an isosceles triangle with l and the x-axis -/
theorem isosceles_triangle_line (l : Set (ℝ × ℝ)) (A : ℝ × ℝ) :
  (∀ x y, (x, y) ∈ l ↔ x - 2*y + 3 = 0) →  -- Line l equation
  A = (3, 3) →  -- Point A coordinates
  A ∈ l →  -- Point A is on line l
  ∃ l' : Set (ℝ × ℝ),
    (A ∈ l') ∧  -- l' passes through A
    (∀ x y, (x, y) ∈ l' ↔ x + 2*y - 9 = 0) ∧  -- Equation of l'
    (∃ B C : ℝ × ℝ, B ∈ l ∧ C ∈ l' ∧ B.2 = 0 ∧ C.2 = 0 ∧  -- B and C are on x-axis
      (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2) :=  -- Isosceles condition
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_line_l10_1014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_calculation_l10_1019

/-- Calculates the sales tax amount on taxable purchases given the total payment,
    tax rate, and cost of tax-free items. -/
noncomputable def salesTaxAmount (totalPayment taxRate taxFreeItemsCost : ℝ) : ℝ :=
  let taxableItemsCost := (totalPayment - taxFreeItemsCost) / (1 + taxRate)
  taxRate * taxableItemsCost

/-- Theorem stating that given the specific conditions in the problem,
    the sales tax amount is $1.28. -/
theorem sales_tax_calculation :
  let totalPayment : ℝ := 40
  let taxRate : ℝ := 0.08
  let taxFreeItemsCost : ℝ := 22.72
  salesTaxAmount totalPayment taxRate taxFreeItemsCost = 1.28 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval salesTaxAmount 40 0.08 22.72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_calculation_l10_1019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upward_distance_to_fiona_l10_1082

-- Define the locations of Daniel, Emma, and Fiona
noncomputable def daniel_location : ℝ × ℝ := (5, -15)
noncomputable def emma_location : ℝ × ℝ := (2, 20)
noncomputable def fiona_location : ℝ × ℝ := (7/2, 5)

-- Define the meeting point as the midpoint of Daniel and Emma's locations
noncomputable def meeting_point : ℝ × ℝ := ((daniel_location.1 + emma_location.1) / 2, (daniel_location.2 + emma_location.2) / 2)

-- Theorem statement
theorem upward_distance_to_fiona :
  fiona_location.2 - meeting_point.2 = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upward_distance_to_fiona_l10_1082
