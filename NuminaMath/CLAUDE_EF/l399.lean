import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_multiplication_l399_39936

theorem scaled_multiplication (a : ℝ) :
  29942.163 * a * 146.539 = 171 →
  299.42163 * a * 1.46539 = 1.71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_multiplication_l399_39936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l399_39992

theorem max_value_of_function : 
  ∃ M : ℝ, M = 1 ∧ ∀ x : ℝ, 1 - 8 * Real.cos x - 2 * (Real.sin x) ^ 2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l399_39992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linden_squares_l399_39945

theorem linden_squares (jesus_squares pedro_squares linden_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : pedro_squares = 200)
  (h3 : pedro_squares = jesus_squares + 65 + linden_squares) : 
  linden_squares = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linden_squares_l399_39945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l399_39919

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Calculates the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1/3) * p.baseEdge^2 * p.altitude

/-- Represents a frustum formed by removing a smaller pyramid from a larger one -/
structure Frustum where
  originalPyramid : SquarePyramid
  altitudeRatio : ℝ

/-- Calculates the volume of the frustum -/
noncomputable def frustumVolume (f : Frustum) : ℝ :=
  pyramidVolume f.originalPyramid - pyramidVolume {
    baseEdge := f.originalPyramid.baseEdge * f.altitudeRatio,
    altitude := f.originalPyramid.altitude * f.altitudeRatio
  }

theorem frustum_volume_ratio (f : Frustum) (h : f.altitudeRatio = 1/4) :
  frustumVolume f / pyramidVolume f.originalPyramid = 63/64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l399_39919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l399_39934

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 8*y + 20 = 0

/-- The line equation -/
def line_equation (y : ℝ) : Prop :=
  y = 2

/-- The area of the circle above the line -/
noncomputable def area_above_line : ℝ := 8 * Real.pi

theorem circle_area_above_line :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_y - 2 = radius / 2 ∧
    area_above_line = (π * radius^2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l399_39934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_interval_l399_39933

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 1 else -x^2 + 4*x - 3

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem f_monotone_interval :
  ∃ a : ℝ, a = 2 ∧ 
    (∀ b, 0 < b ∧ b ≤ a → MonotonicallyIncreasing f 0 b) ∧
    (∀ c, c > a → ¬MonotonicallyIncreasing f 0 c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_interval_l399_39933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gym_problem_l399_39979

/-- The number of people initially lifting weights in the gym -/
def initial_lifters : ℕ := sorry

/-- The number of people who entered the gym to run on the treadmill -/
def treadmill_runners : ℕ := 5

/-- The number of people who left the gym -/
def people_left : ℕ := 2

/-- The final number of people in the gym -/
def final_count : ℕ := 19

theorem gym_problem :
  initial_lifters = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gym_problem_l399_39979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_addition_l399_39932

/-- The direction vector of the line x = -2y = 2z -/
def b : Fin 3 → ℝ := ![1, -2, 2]

/-- The vector to be projected -/
def a : Fin 3 → ℝ := ![6, 3, -2]

/-- The vector to be added after projection -/
def c : Fin 3 → ℝ := ![1, -1, 2]

/-- Dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

/-- Vector scalar multiplication -/
def scalar_mult (s : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => s * (v i)

/-- Vector addition -/
def vector_add (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => (v i) + (w i)

/-- The main theorem -/
theorem projection_and_addition :
  let proj := scalar_mult (dot_product a b / dot_product b b) b
  vector_add proj c = ![5/9, -1/9, 10/9] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_addition_l399_39932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_factor_value_l399_39943

/-- Given a cubic polynomial with factors (x - 3) and (x + 1), prove |3t - 2q| = 99 -/
theorem cubic_factor_value (t q : ℝ) : 
  (∃ a r : ℝ, ∀ x : ℝ, 3 * x^3 - t * x + q = a * (x - 3) * (x + 1) * (x - r)) → 
  |3 * t - 2 * q| = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_factor_value_l399_39943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_female_employees_l399_39971

/-- Proves that the percentage of female employees is 60% given the conditions -/
theorem percentage_female_employees (total_employees : ℕ) 
  (computer_literate_percentage : ℚ) (male_computer_literate_percentage : ℚ)
  (female_computer_literate : ℕ) :
  total_employees = 1600 →
  computer_literate_percentage = 62 / 100 →
  male_computer_literate_percentage = 1 / 2 →
  female_computer_literate = 672 →
  (↑female_computer_literate / (computer_literate_percentage * ↑total_employees) : ℚ) = 60 / 100 := by
  intros h1 h2 h3 h4
  sorry

#check percentage_female_employees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_female_employees_l399_39971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l399_39964

noncomputable def f (x : ℝ) : ℝ := 3 * Real.tan (2 * x + Real.pi / 3)

theorem center_of_symmetry (k : ℤ) :
  ∃ (x : ℝ), x = k * Real.pi / 4 - Real.pi / 6 ∧
  ∀ (y : ℝ), f (x - y) = -f (x + y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l399_39964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l399_39966

/-- A right triangle with sides 3, 4, and 5 -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  ab_length : norm (B - A) = 3
  bc_length : norm (C - B) = 4
  ac_length : norm (C - A) = 5
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

/-- An inscribed square in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  w_on_ab : ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ W = (1 - k) • t.A + k • t.B
  x_on_ac : ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ X = (1 - k) • t.A + k • t.C
  y_on_ac : ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ Y = (1 - k) • t.A + k • t.C
  z_on_bc : ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ Z = (1 - k) • t.B + k • t.C
  is_square : norm (X - W) = norm (Y - X) ∧ norm (Y - X) = norm (Z - Y) ∧ norm (Z - Y) = norm (W - Z)

/-- The side length of the inscribed square is 60/37 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  norm (s.X - s.W) = 60 / 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l399_39966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_composition_equation_solutions_l399_39993

-- Define the function f(x) = sin(sin(sin(sin(sin(x)))))
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.sin (Real.sin (Real.sin (Real.sin x))))

-- Define the function g(x) = x/3
noncomputable def g (x : ℝ) : ℝ := x / 3

-- State the theorem
theorem sin_composition_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = g x := by
  sorry

#check sin_composition_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_composition_equation_solutions_l399_39993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equals_open_closed_interval_l399_39956

open Set
open Function
open Real

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | x > 0 ∧ f x ≤ log x}

theorem solution_set_equals_open_closed_interval
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (f 1 = 0) →
  (∀ x > 0, x * (deriv f x) > 1) →
  solution_set f = Ioc 0 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equals_open_closed_interval_l399_39956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l399_39937

/-- The length of a train given its speed and time to cross a point -/
noncomputable def train_length (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_km_hr * (1000 / 3600) * time_seconds

/-- Theorem stating the length of the train -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) 
  (h1 : speed_km_hr = 144)
  (h2 : time_seconds = 3.499720022398208) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_length speed_km_hr time_seconds - 139.99| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l399_39937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l399_39994

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∃ k : ℝ, k = -3 ∧ k = (deriv (f a)) 1) →
  (a = -3 ∧ b = -1) ∧
  (∀ A : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 4 → f a x ≤ A - 1994) ↔ A ≥ 2011) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l399_39994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_point_l399_39960

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point on an ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := Real.sqrt (1 - E.b^2 / E.a^2)

/-- Predicate to represent PF₁ ⟂ PF₂ -/
def PF₁_perpendicular_PF₂ (E : Ellipse) (P : PointOnEllipse E) : Prop := sorry

/-- Predicate to represent |PF₁| = 2|PF₂| -/
def PF₁_twice_PF₂ (E : Ellipse) (P : PointOnEllipse E) : Prop := sorry

/-- Theorem: If there exists a point P on the ellipse such that PF₁ ⟂ PF₂ and |PF₁| = 2|PF₂|,
    then the eccentricity of the ellipse is √5/3 -/
theorem ellipse_eccentricity_special_point (E : Ellipse) (P : PointOnEllipse E)
  (h_perpendicular : PF₁_perpendicular_PF₂ E P)
  (h_length_ratio : PF₁_twice_PF₂ E P) :
  eccentricity E = Real.sqrt 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_point_l399_39960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_division_l399_39950

noncomputable section

/-- Regular hexagon with side length 2 and center at origin -/
def RegularHexagon : Set ℂ :=
  {z : ℂ | ∃ k : ℕ, k < 6 ∧ z = 2 * Complex.exp (k * Real.pi * Complex.I / 3)}

/-- Point M divides diagonal AC in ratio r -/
def M (r : ℝ) : ℂ := 2 * ((1 - r) + r * Complex.exp (2 * Real.pi * Complex.I / 3))

/-- Point N divides diagonal CE in ratio r -/
def N (r : ℝ) : ℂ := 2 * Complex.exp (2 * Real.pi * Complex.I / 3) * ((1 - r) + r * Complex.exp (2 * Real.pi * Complex.I / 3))

/-- B is a vertex of the hexagon -/
def B : ℂ := 2 * Complex.exp (Real.pi * Complex.I / 3)

/-- Collinearity condition for three complex points -/
def AreCollinear (z₁ z₂ z₃ : ℂ) : Prop :=
  (z₂ - z₁).im * (z₃ - z₁).re = (z₃ - z₁).im * (z₂ - z₁).re

theorem hexagon_diagonal_division (r : ℝ) :
  AreCollinear B (M r) (N r) → r = Real.sqrt 3 / 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_division_l399_39950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l399_39968

noncomputable def a (n : ℕ+) : ℝ := 2 * (n : ℝ) + 1

noncomputable def S (n : ℕ+) : ℝ := ((n : ℝ) * (a n + a 1)) / 2

noncomputable def b (n : ℕ+) : ℝ := (((n : ℝ) + 1) ^ 2 + 1) / (S n)

noncomputable def T (n : ℕ+) : ℝ := (n : ℝ) + 3 / 2 - (2 * (n : ℝ) + 3) / (((n : ℝ) + 1) * ((n : ℝ) + 3))

theorem sequence_properties (n : ℕ+) :
  (a n > 0) ∧
  ((a n + 1) ^ 2 = 4 * (S n + 1)) ∧
  (b n * S n - 1 = ((n : ℝ) + 1) ^ 2) ∧
  (a n = 2 * (n : ℝ) + 1) ∧
  (T n = (n : ℝ) + 3 / 2 - (2 * (n : ℝ) + 3) / (((n : ℝ) + 1) * ((n : ℝ) + 3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l399_39968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l399_39975

/-- Given a rectangle ABCD with dimensions AB = 6√3 and BC = 13√3, and point P at the intersection
    of diagonals AC and BD, prove that the volume of the pyramid formed by creasing along CP and DP
    after removing triangle ABP is (1/3) * 39√3 * z, where z is the height of the pyramid. --/
theorem pyramid_volume (A B C D P : ℝ × ℝ × ℝ) (z : ℝ) : 
  let AB : ℝ := 6 * Real.sqrt 3
  let BC : ℝ := 13 * Real.sqrt 3
  let volume : ℝ := (1/3) * 39 * Real.sqrt 3 * z
  -- P is the midpoint of AC
  -- All faces of the pyramid are isosceles triangles
  -- z is the height of the pyramid from P to the base BCD
  volume = (1/3) * 39 * Real.sqrt 3 * z := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l399_39975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l399_39985

/-- Function representing the number of arrangements of n balls in m boxes -/
def number_of_arrangements (n m : ℕ) : ℕ :=
  sorry  -- Definition not provided, as it should come from the problem conditions

/-- The number of ways to distribute n identical balls into m distinguishable boxes,
    such that no box is empty and n ≥ m, is equal to (n-1) choose (m-1) -/
theorem balls_in_boxes (n m : ℕ) (h : n ≥ m) :
  (number_of_arrangements n m) = Nat.choose (n - 1) (m - 1) :=
by sorry

/-- n is the number of identical balls -/
def n : ℕ := sorry

/-- m is the number of distinguishable boxes -/
def m : ℕ := sorry

/-- Condition that n ≥ m -/
axiom n_geq_m : n ≥ m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_l399_39985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l399_39952

/-- Function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The maximum area of a triangle DEF with DE = 10 and DF:EF = 30:31 is 1250 -/
theorem triangle_max_area (DE EF DF : ℝ) (h1 : DE = 10) 
  (h2 : DF / EF = 30 / 31) : 
  (∀ A : ℝ, A = area_triangle DE EF DF → A ≤ 1250) ∧ 
  (∃ A : ℝ, A = area_triangle DE EF DF ∧ A = 1250) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l399_39952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normals_to_parabola_l399_39939

/-- The number of normals from a point to a parabola -/
noncomputable def num_normals (p a b : ℝ) : ℕ :=
  if 4 * (2 * p - a)^2 + 27 * p * b^2 > 0 then 1
  else if 4 * (2 * p - a)^2 + 27 * p * b^2 = 0 then 2
  else 3

/-- Theorem about the number of normals from a point to a parabola -/
theorem normals_to_parabola (p a b : ℝ) (hp : p > 0) :
  let S := {(x, y) : ℝ × ℝ | y^2 = 4 * p * x}
  let P := (a, b)
  num_normals p a b = 
    match (4 * (2 * p - a)^2 + 27 * p * b^2) with
    | v => if v > 0 then 1 else if v = 0 then 2 else 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normals_to_parabola_l399_39939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l399_39906

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  circle_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 :=
by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l399_39906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_calculation_l399_39903

/-- Calculates the total time for a car journey with two segments -/
noncomputable def total_time (b n : ℝ) : ℝ :=
  let first_segment_time := b / 50
  let second_segment_time := (2 * n + b) / 75
  first_segment_time + second_segment_time

/-- Theorem stating that the total time for the journey is (5b + 4n) / 150 hours -/
theorem journey_time_calculation (b n : ℝ) :
  total_time b n = (5 * b + 4 * n) / 150 := by
  unfold total_time
  -- Expand the definition and simplify
  simp [add_div, mul_div_assoc]
  -- Perform algebraic manipulations
  ring

#check journey_time_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_calculation_l399_39903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_perpendicularity_l399_39935

/-- Two circles intersect at two points -/
def CircleIntersection (O₁ O₂ A B : Point) : Prop :=
  sorry

/-- One circle is on the perimeter of another circle -/
def OnPerimeter (O₁ O₂ : Point) : Prop :=
  sorry

/-- A chord of one circle intersects another circle at a point -/
def ChordIntersection (O₁ O₂ A C D : Point) : Prop :=
  sorry

/-- Two line segments are perpendicular -/
def Perpendicular (P Q R S : Point) : Prop :=
  sorry

/-- Given two circles O₁ and O₂ that intersect at points A and B, 
    with O₁ on the perimeter of O₂, and a chord AC of O₁ intersecting O₂ at D,
    prove that O₁D is perpendicular to BC. -/
theorem circle_intersection_perpendicularity 
  (O₁ O₂ A B C D : Point) : 
  CircleIntersection O₁ O₂ A B → 
  OnPerimeter O₁ O₂ → 
  ChordIntersection O₁ O₂ A C D → 
  Perpendicular O₁ D B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_perpendicularity_l399_39935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_handshakes_proof_l399_39917

/-- The number of men in the group -/
def n : ℕ := 30

/-- The number of men involved in each handshake -/
def k : ℕ := 2

/-- Calculates the number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The maximum number of handshakes without cyclic handshakes -/
def maxHandshakes : ℕ := combinations n k

theorem max_handshakes_proof :
  maxHandshakes = 435 := by
  -- Unfold the definitions
  unfold maxHandshakes
  unfold combinations
  -- Simplify the expression
  simp [n, k]
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_handshakes_proof_l399_39917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_area_is_correct_l399_39922

/-- The area of a triangular region bounded by the coordinate axes and the line 3x + y = 9 -/
noncomputable def triangular_area : ℝ := 27 / 2

/-- The equation of the bounding line -/
def line_equation (x y : ℝ) : Prop := 3 * x + y = 9

theorem triangular_area_is_correct : 
  ∃ (x_intercept y_intercept : ℝ),
    line_equation x_intercept 0 ∧ 
    line_equation 0 y_intercept ∧
    triangular_area = (1/2) * x_intercept * y_intercept := by
  sorry

#check triangular_area_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_area_is_correct_l399_39922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l399_39963

theorem subset_count : ∀ (S : Finset ℕ),
  S = {1, 2, 3, 4, 5, 6, 7} →
  (Finset.filter (λ X : Finset ℕ => {1, 2, 3} ⊆ X ∧ X ⊆ S) (Finset.powerset S)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l399_39963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l399_39980

-- Define the circle C
def CircleC (D E : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + D*p.1 + E*p.2 + 3 = 0}

-- Define the line of symmetry
def LineOfSymmetry : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 1 = 0}

-- Define the second quadrant
def SecondQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

-- Define the center of a circle
noncomputable def Center (C : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the radius of a circle
noncomputable def Radius (C : Set (ℝ × ℝ)) : ℝ := sorry

-- Define symmetry with respect to a line
def SymmetricWRT (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : Prop := sorry

theorem circle_properties :
  ∀ D E : ℝ,
  SymmetricWRT (CircleC D E) LineOfSymmetry →
  Center (CircleC D E) ∈ SecondQuadrant →
  Radius (CircleC D E) = Real.sqrt 2 →
  (∃ a b : ℝ, CircleC D E = {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 2}) ∧
  (¬∃ k : ℝ, 
    let L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2*p.1 + k}
    ∃ A B : ℝ × ℝ, A ∈ CircleC D E ∩ L ∧ B ∈ CircleC D E ∩ L ∧ A ≠ B ∧
    (0, 0) ∈ {p : ℝ × ℝ | (p.1 - (A.1 + B.1)/2)^2 + (p.2 - (A.2 + B.2)/2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2)/4}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l399_39980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l399_39989

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.sin (1/3))
  (hb : b = (1/3) ^ (0.9 : ℝ))
  (hc : c = (1/2) * (Real.log 9 / Real.log 27)) :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l399_39989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l399_39946

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (5, 0)

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 3*x + 4*y = 0

-- Define the distance function between a point and a line
noncomputable def distance_point_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x, y) := p
  abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

theorem hyperbola_focus_asymptote_distance :
  distance_point_line right_focus 3 4 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l399_39946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_calculation_l399_39970

theorem deposit_calculation (remaining : ℝ) (deposit_percentage : ℝ) 
  (h1 : remaining = 990)
  (h2 : deposit_percentage = 0.1) : 
  (remaining / (1 - deposit_percentage)) * deposit_percentage = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_calculation_l399_39970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_original_price_l399_39967

/-- Given two successive discounts and a final price, calculate the original price --/
theorem calculate_original_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  let remaining_factor1 := 1 - discount1
  let remaining_factor2 := 1 - discount2
  let original_price := final_price / (remaining_factor1 * remaining_factor2)
  final_price = 59.22 ∧ discount1 = 0.1 ∧ discount2 = 0.06000000000000002 →
  ∃ ε > 0, |original_price - 70| < ε := by
sorry

#eval Float.toString (59.22 / ((1 - 0.1) * (1 - 0.06000000000000002)))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_original_price_l399_39967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_maximal_monotone_interval_l399_39955

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 - sqrt 3 * cos x * cos (x + π / 2)

/-- Theorem stating that f(x) is monotonically increasing on [0, π/3] -/
theorem f_monotone_increasing :
  MonotoneOn f (Set.Icc 0 (π / 3)) :=
by sorry

/-- Theorem stating that [0, π/3] is the maximal interval of monotonic increase for f(x) on [0, π/2] -/
theorem f_maximal_monotone_interval :
  ∀ a b, 0 ≤ a ∧ b ≤ π / 2 ∧ MonotoneOn f (Set.Icc a b) → Set.Icc a b ⊆ Set.Icc 0 (π / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_maximal_monotone_interval_l399_39955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_correct_l399_39902

/-- The x-coordinate of the point on the x-axis that is equidistant from A(-3, 0) and B(2, 5) -/
def equidistant_point : ℝ := 2

/-- Point A coordinates -/
def A : ℝ × ℝ := (-3, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ := (2, 5)

/-- Distance function between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem equidistant_point_correct :
  let P : ℝ × ℝ := (equidistant_point, 0)
  distance A P = distance B P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_correct_l399_39902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_theorem_l399_39969

/-- Represents a tripod with given initial leg length and height -/
structure Tripod :=
  (leg_length : ℝ)
  (initial_height : ℝ)

/-- Calculates the new height of a tripod after one leg is shortened -/
def new_height (t : Tripod) (shortened_length : ℝ) : ℝ := 
  sorry

/-- Expresses the new height as a fraction p/√q -/
def height_fraction (h : ℝ) : ℚ × ℕ := 
  sorry

theorem tripod_height_theorem (t : Tripod) : 
  t.leg_length = 6 → 
  t.initial_height = 5 → 
  let h := new_height t 1.5
  let (p, q) := height_fraction h
  ∃ (p' q' : ℕ), p = p' ∧ q = q' ∧ 
    Nat.Coprime q' (q' * q') ∧
    ⌊(p' : ℝ) + Real.sqrt q'⌋ = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_theorem_l399_39969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l399_39961

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 4)^2 = 8

-- Define the line that the center lies on
def center_line (x y : ℝ) : Prop := y = -4 * x

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (3, -2)

theorem circle_properties :
  -- The center of the circle lies on the line y = -4x
  ∃ (cx cy : ℝ), center_line cx cy ∧ circle_equation cx cy ∧
  -- The circle is tangent to the line x + y - 1 = 0
  (∃ (tx ty : ℝ), tangent_line tx ty ∧ circle_equation tx ty) ∧
  -- The point of tangency is (3, -2)
  circle_equation tangent_point.1 tangent_point.2 ∧ tangent_line tangent_point.1 tangent_point.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l399_39961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l399_39944

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (3, -1)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the perimeter of the triangle
noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C A

-- Theorem statement
theorem triangle_perimeter :
  perimeter = Real.sqrt 52 + Real.sqrt 65 + Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l399_39944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l399_39951

theorem expression_value : (2^2 * 2^3 + 2^1) / ((1:ℝ)/2 + (1:ℝ)/4 + (1:ℝ)/8) = 272 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l399_39951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l399_39995

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem smallest_positive_sum
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_pos : d > 0)
  (h_roots : a 2009 * a 2010 = -5 ∧ a 2009 + a 2010 = 3)
  : (∀ n < 4018, sum_of_arithmetic_sequence a n ≤ 0) ∧
    sum_of_arithmetic_sequence a 4018 > 0 := by
  sorry

#check smallest_positive_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l399_39995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_divisor_24_factorial_l399_39984

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem prob_odd_divisor_24_factorial :
  let n := factorial 24
  let total_divisors := Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))
  let odd_divisors := Finset.filter (λ x => x % 2 ≠ 0) total_divisors
  (Finset.card odd_divisors : ℚ) / (Finset.card total_divisors) = 1 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_divisor_24_factorial_l399_39984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_score_l399_39977

theorem class_average_score (total_students : ℕ) 
  (assigned_day_percentage : ℚ) (assigned_day_score : ℚ) 
  (makeup_score : ℚ) 
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 70 / 100)
  (h3 : assigned_day_score = 60 / 100)
  (h4 : makeup_score = 80 / 100) :
  (assigned_day_percentage * assigned_day_score * total_students + 
    (1 - assigned_day_percentage) * makeup_score * total_students) / total_students = 66 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_score_l399_39977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_to_income_ratio_example_l399_39928

/-- Represents the financial data of a business -/
structure BusinessFinance where
  total_profit : ℕ
  total_income : ℕ

/-- Calculates the ratio of spending to income for a business -/
def spending_to_income_ratio (b : BusinessFinance) : ℚ :=
  let total_spending := b.total_income - b.total_profit
  (total_spending : ℚ) / b.total_income

/-- Theorem stating that for the given business data, the spending to income ratio is 5/9 -/
theorem spending_to_income_ratio_example :
  let b := BusinessFinance.mk 48000 108000
  spending_to_income_ratio b = 5 / 9 := by
  sorry

#eval spending_to_income_ratio (BusinessFinance.mk 48000 108000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_to_income_ratio_example_l399_39928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_bob_journey_l399_39940

/-- Calculates the distance traveled north after riding west in a specific journey -/
noncomputable def distance_north_after_west (west : ℝ) (east : ℝ) (north_after : ℝ) (total_distance : ℝ) : ℝ :=
  let net_west := west - east
  let total_north := Real.sqrt (total_distance ^ 2 - net_west ^ 2) - north_after
  total_north

/-- Theorem stating the distance traveled north after riding west in Biker Bob's journey -/
theorem biker_bob_journey :
  let west := 30
  let east := 15
  let north_after := 18
  let total_distance := 28.30194339616981
  abs (distance_north_after_west west east north_after total_distance - 6.020274) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_bob_journey_l399_39940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_point_existence_l399_39982

-- Define the piecewise function
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else k*x + 2

-- Define what it means for a point to be "beautiful"
def is_beautiful_point (k : ℝ) (x : ℝ) : Prop :=
  f k x + f k (-x) = 0

-- State the theorem
theorem beautiful_point_existence (k : ℝ) :
  (∃ x, is_beautiful_point k x) ↔ k ≤ 2 - 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_point_existence_l399_39982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_count_proof_l399_39905

/-- The number of cookies per bag -/
def cookies_per_bag : ℝ := 9.0

/-- The number of chocolate chip cookies -/
def chocolate_chip_cookies : ℝ := 13.0

/-- The number of baggies of oatmeal cookies -/
def oatmeal_baggies : ℝ := 3.111111111

/-- The total number of cookies -/
def total_cookies : ℕ := 41

theorem cookie_count_proof :
  ⌊cookies_per_bag * oatmeal_baggies⌋ + ⌊chocolate_chip_cookies⌋ = total_cookies :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_count_proof_l399_39905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_length_l399_39930

-- Define the line l
noncomputable def line_l (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (-Real.sqrt 2 / 2 * t, a + Real.sqrt 2 / 2 * t)

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop :=
  y^2 = 4*x

-- Define the condition that line l passes through (0, 1)
def line_passes_through_point (a : ℝ) : Prop :=
  ∃ t, line_l a t = (0, 1)

-- Theorem statement
theorem intercept_length (a : ℝ) (h : line_passes_through_point a) :
  ∃ t₁ t₂, t₁ ≠ t₂ ∧
    curve_C (line_l a t₁).1 (line_l a t₁).2 ∧
    curve_C (line_l a t₂).1 (line_l a t₂).2 ∧
    Real.sqrt ((t₁ - t₂)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_length_l399_39930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l399_39954

theorem cube_root_inequality (a b : ℝ) (h : a > b) : Real.rpow a (1/3) > Real.rpow b (1/3) := by
  by_contra h'
  have h'' : Real.rpow a (1/3) ≤ Real.rpow b (1/3) := not_lt.mp h'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l399_39954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_segment_volume_ratio_l399_39921

/-- Represents a cone with height and base radius -/
structure Cone where
  height : ℝ
  radius : ℝ

/-- Represents a segment of a cone -/
structure ConeSegment where
  lower_radius : ℝ
  upper_radius : ℝ
  height : ℝ

/-- Calculates the volume of a cone segment -/
noncomputable def volume_cone_segment (segment : ConeSegment) : ℝ :=
  (1/3) * Real.pi * segment.height * (segment.lower_radius^2 + segment.upper_radius^2 + segment.lower_radius * segment.upper_radius)

/-- Theorem stating the volume ratio of cone segments when divided into three equal parts -/
theorem cone_segment_volume_ratio (cone : Cone) :
  let h := cone.height / 3
  let r := cone.radius
  let top_segment := ConeSegment.mk (r/3) 0 h
  let middle_segment := ConeSegment.mk (2*r/3) (r/3) h
  let bottom_segment := ConeSegment.mk r (2*r/3) h
  let v1 := volume_cone_segment top_segment
  let v2 := volume_cone_segment middle_segment
  let v3 := volume_cone_segment bottom_segment
  (v1 : ℝ) / (v1 + v2 + v3) = 1 / 27 ∧
  (v2 : ℝ) / (v1 + v2 + v3) = 7 / 27 ∧
  (v3 : ℝ) / (v1 + v2 + v3) = 19 / 27 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_segment_volume_ratio_l399_39921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_10_equals_50_l399_39957

def a (n : ℕ) : ℤ := 11 - 2 * n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ i => Int.natAbs (a (i + 1)))

theorem S_10_equals_50 : S 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_10_equals_50_l399_39957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l399_39997

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + b * x

noncomputable def f' (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

theorem tangent_line_and_extrema :
  ∃ (a b : ℝ),
    (f' a b 2 = -1) ∧
    (f a b 2 = 2/3) ∧
    (a = 2) ∧
    (b = 3) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b x ≤ 4/3) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b x = 4/3) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b x ≥ -50/3) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a b x = -50/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l399_39997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_issues_duration_l399_39998

/-- The number of days the store was short-staffed and self-checkout was broken -/
def days_with_issues (normal_complaints : ℕ) (short_staffed_increase : ℚ) 
  (broken_checkout_increase : ℚ) (total_complaints : ℕ) : ℚ :=
  let short_staffed_complaints := (normal_complaints : ℚ) * (1 + short_staffed_increase)
  let both_issues_complaints := short_staffed_complaints * (1 + broken_checkout_increase)
  (total_complaints : ℚ) / both_issues_complaints

theorem store_issues_duration :
  days_with_issues 120 (1/3) (1/5) 576 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_issues_duration_l399_39998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_l399_39987

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

/-- Calculates the new height of a cone after adding a sphere -/
noncomputable def newConeHeight (c : Cone) (s : Sphere) : ℝ :=
  c.height + sphereVolume s / ((1/3) * Real.pi * c.radius^2)

/-- Theorem: The ratio of liquid level rise in narrower cone to broader cone is 4:1 -/
theorem liquid_level_rise_ratio :
  let narrowCone : Cone := { radius := 4, height := h₁ }
  let broadCone : Cone := { radius := 8, height := h₂ }
  let marble : Sphere := { radius := 2 }
  ∀ h₁ h₂ : ℝ,
    coneVolume narrowCone = coneVolume broadCone →
    (newConeHeight narrowCone marble - narrowCone.height) /
    (newConeHeight broadCone marble - broadCone.height) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_l399_39987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_product_implies_same_product_l399_39909

open BigOperators

theorem same_product_implies_same_product
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_same_product : ∃ c : ℝ, ∀ i : Fin n, ∏ j, (a i + b j) = c) :
  ∃ c : ℝ, ∀ j : Fin n, ∏ i, (a i + b j) = c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_product_implies_same_product_l399_39909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l399_39999

/-- The maximum area of an equilateral triangle inscribed in a rectangle --/
theorem max_equilateral_triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a = 8 ∧ b = 15 →
  ∃ (s : ℝ),
  s ≤ min a b ∧
  (Real.sqrt 3 / 4) * s^2 = 64 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l399_39999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_occurs_at_four_l399_39920

theorem min_value_occurs_at_four (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 8) :
  (∀ x y : ℝ, x > 0 → y > 0 → x * y = 8 → (2 : ℝ)^a * (4 : ℝ)^b ≤ (2 : ℝ)^x * (4 : ℝ)^y) → a = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_occurs_at_four_l399_39920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_cutting_l399_39962

theorem rod_cutting (rod_length : ℝ) (piece_length : ℝ) : 
  rod_length = 42.5 → piece_length = 0.85 → 
  ⌊rod_length / piece_length⌋ = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_cutting_l399_39962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l399_39941

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x

-- State the theorem
theorem f_monotonic_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 2 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l399_39941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_three_l399_39916

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarFuelEfficiency where
  highway_miles_per_tank : ℚ
  city_miles_per_tank : ℚ
  city_miles_per_gallon : ℚ

/-- Calculates the difference in miles per gallon between highway and city driving -/
noncomputable def mpg_difference (car : CarFuelEfficiency) : ℚ :=
  (car.highway_miles_per_tank / (car.city_miles_per_tank / car.city_miles_per_gallon)) - car.city_miles_per_gallon

/-- Theorem: The difference in miles per gallon between highway and city driving is 3 -/
theorem mpg_difference_is_three (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tank = 462)
  (h2 : car.city_miles_per_tank = 336)
  (h3 : car.city_miles_per_gallon = 8) :
  mpg_difference car = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_three_l399_39916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_f_explicit_l399_39983

/-- The function f : ℝ → ℝ -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

/-- f is symmetric about x = 1 -/
axiom f_sym : ∀ x : ℝ, f (2 - x) = f x

/-- Definition of f on (0, 1] -/
axiom f_def : ∀ x : ℝ, 0 < x → x ≤ 1 → f x = Real.sqrt x

/-- Prove that f is periodic with period 4 -/
theorem f_periodic : ∀ x : ℝ, f (x + 4) = f x := by
  sorry

/-- Prove the explicit formula for f on [-5, -4] -/
theorem f_explicit : ∀ x : ℝ, -5 ≤ x → x ≤ -4 → f x = -Real.sqrt (-(x + 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_f_explicit_l399_39983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_henderson_goldfish_preference_l399_39915

/-- Represents a class in the survey -/
structure SurveyClass where
  name : String
  students : ℕ
  goldfish_preference : ℚ

/-- Represents the survey results -/
structure Survey where
  classes : List SurveyClass
  total_goldfish_preference : ℕ

theorem henderson_goldfish_preference (survey : Survey) :
  survey.classes.length = 3 ∧
  (∀ c ∈ survey.classes, c.students = 30) ∧
  (∃ c ∈ survey.classes, c.name = "Miss Johnson" ∧ c.goldfish_preference = 1/6) ∧
  (∃ c ∈ survey.classes, c.name = "Mr. Feldstein" ∧ c.goldfish_preference = 2/3) ∧
  (∃ c ∈ survey.classes, c.name = "Ms. Henderson") ∧
  survey.total_goldfish_preference = 31 →
  ∃ c ∈ survey.classes, c.name = "Ms. Henderson" ∧ c.goldfish_preference = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_henderson_goldfish_preference_l399_39915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_when_k_is_neg_three_l399_39901

/-- Parameterization of the first line -/
noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ := (2 + s, 1 - 2*k*s, 4 + k*s)

/-- Parameterization of the second line -/
noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ := (-t/3, 2 + t, 5 - 2*t)

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  v₁ * w₂ - v₂ * w₁ = 0 ∧ v₁ * w₃ - v₃ * w₁ = 0 ∧ v₂ * w₃ - v₃ * w₂ = 0

/-- Direction vector of the first line -/
noncomputable def dir1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -2*k, k)

/-- Direction vector of the second line -/
noncomputable def dir2 : ℝ × ℝ × ℝ := (-1/3, 1, -2)

/-- The lines are coplanar when k = -3 -/
theorem lines_coplanar_when_k_is_neg_three :
  ∃ (s t : ℝ), line1 s (-3) = line2 t ∨ are_parallel (dir1 (-3)) dir2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_when_k_is_neg_three_l399_39901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_radii_l399_39904

-- Define the truncated cone and inscribed sphere
structure TruncatedConeWithSphere where
  R : ℝ  -- Radius of the bottom base
  r : ℝ  -- Radius of the top base
  s : ℝ  -- Radius of the inscribed sphere
  h : ℝ  -- Height of the truncated cone

-- Define the properties of the truncated cone and sphere
def satisfiesConditions (cone : TruncatedConeWithSphere) : Prop :=
  cone.r = cone.R / 2 ∧  -- Top radius is half of bottom radius
  (1/3) * Real.pi * cone.h * (cone.R^2 + cone.R * cone.r + cone.r^2) = 4 * Real.pi * cone.s^3  -- Volume relation

-- Theorem to prove
theorem ratio_of_radii (cone : TruncatedConeWithSphere) 
  (h : satisfiesConditions cone) : cone.R / cone.r = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_radii_l399_39904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l399_39990

/-- The function g(x) defined as sin^4(x) + cos^4(x) + 2sin(x)cos(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 4 + 2 * Real.sin x * Real.cos x

/-- Theorem stating that the range of g(x) is [-1/2, 3/2] -/
theorem g_range :
  (∀ x : ℝ, -1/2 ≤ g x ∧ g x ≤ 3/2) ∧
  (∃ x : ℝ, g x = -1/2) ∧
  (∃ x : ℝ, g x = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l399_39990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l399_39991

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 4 + y^2 / m = 1 → 
    (∃ c : ℝ, c = Real.sqrt (4 - m) ∧ c / 2 = 1/2)) → 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l399_39991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_breaking_orders_l399_39986

theorem target_breaking_orders : 
  (let n : ℕ := 6
   let k : ℕ := 3
   let items_per_type : ℕ := 2
   Nat.factorial n / (Nat.factorial items_per_type ^ k) = 90) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_breaking_orders_l399_39986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_hugo_rolls_seven_given_wins_l399_39924

/-- Represents the outcome of a die roll. -/
def DieRoll := Fin 8

/-- Represents a player in the game. -/
structure Player where
  roll : DieRoll

/-- Represents the game with 5 players. -/
structure Game where
  players : Fin 5 → Player

/-- Hugo is one of the players in the game. -/
def hugo : Fin 5 := 0

/-- The event that Hugo wins the game. -/
def HugoWins (g : Game) : Prop := sorry

/-- The probability that Hugo wins the game. -/
noncomputable def probHugoWins : ℝ := 1 / 5

/-- The probability that Hugo's first roll is 7. -/
noncomputable def probHugoRollsSeven : ℝ := 1 / 8

/-- The probability that Hugo wins given his first roll was 7. -/
noncomputable def probHugoWinsGivenSeven : ℝ := 2118 / 4096

/-- The main theorem to prove. -/
theorem prob_hugo_rolls_seven_given_wins :
  (probHugoRollsSeven * probHugoWinsGivenSeven) / probHugoWins = 5295 / 20480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_hugo_rolls_seven_given_wins_l399_39924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l399_39907

theorem tan_one_condition (k : ℤ) (x : ℝ) :
  (x = 2 * k * Real.pi + Real.pi / 4 → Real.tan x = 1) ∧
  ¬(Real.tan x = 1 → x = 2 * k * Real.pi + Real.pi / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l399_39907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l399_39974

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The ellipse problem -/
theorem ellipse_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let f (x y : ℝ) := x^2 / a^2 + y^2 / b^2
  ∃ A : ℝ × ℝ, A = (-1, Real.sqrt 14 / 2) ∧ 
    f A.1 A.2 = 1 ∧
    (∃ B F : ℝ × ℝ, 
      B.1 = -A.1 ∧ B.2 = -A.2 ∧
      Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) + 
      Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) + 
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 7 * Real.sqrt 2) →
  a^2 = 8 ∧ b^2 = 4 ∧
  (∀ C D : ℝ × ℝ, 
    f C.1 C.2 = 1 → f D.1 D.2 = 1 →
    (D.2 - C.2) * (A.2 + 1) = (D.1 - C.1) * (A.1 + 1) →
    area_triangle A C D ≤ 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l399_39974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l399_39923

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + 2*t, 2*t)

-- Define the curve C in polar form
noncomputable def curve_C (θ : ℝ) : ℝ := 2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

-- Define the rectangular form of curve C
def curve_C_rect (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ curve_C_rect p.1 p.2}

-- Theorem statement
theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
  (A.1 + B.1) / 2 = 4/5 ∧ (A.2 + B.2) / 2 = -1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l399_39923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l399_39925

-- Define the hyperbola and its properties
def Hyperbola (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-Real.sqrt 10, 0) ∧ F₂ = (Real.sqrt 10, 0)

def PointOnHyperbola (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  let MF₁ := (M.1 - F₁.1, M.2 - F₁.2)
  let MF₂ := (M.1 - F₂.1, M.2 - F₂.2)
  MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0 ∧
  ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) * ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 4

-- State the theorem
theorem hyperbola_equation 
  (F₁ F₂ : ℝ × ℝ) 
  (h₁ : Hyperbola F₁ F₂) 
  (M : ℝ × ℝ) 
  (h₂ : PointOnHyperbola M F₁ F₂) :
  ∃ (x y : ℝ), x^2/9 - y^2 = 1 ∧ M = (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l399_39925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l399_39911

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2 * y = 2) :
  (2 : ℝ)^x + (4 : ℝ)^y ≥ 4 ∧ ((2 : ℝ)^x + (4 : ℝ)^y = 4 ↔ x = 1 ∧ y = 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l399_39911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_domain_sine_l399_39926

theorem max_domain_sine (a b : ℝ) : 
  (∀ x ∈ Set.Icc a b, -1 ≤ Real.sin x ∧ Real.sin x ≤ -1/2) →
  (∃ x ∈ Set.Icc a b, Real.sin x = -1) →
  (∃ x ∈ Set.Icc a b, Real.sin x = -1/2) →
  b - a ≤ 4 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_domain_sine_l399_39926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_properties_inequality_condition_l399_39900

noncomputable def f (m : ℕ) (x : ℝ) : ℝ := 4 * x / (2 * x^2 + m)

def tangent_equation (t : ℝ) (x y : ℝ) : Prop := 8 * x - 9 * y + t = 0

theorem function_and_tangent_properties (m : ℕ) (t : ℝ) :
  (∃ y, tangent_equation t (1/2) y ∧ y = f m (1/2)) →
  (m = 1 ∧ t = 8) := by
  sorry

theorem inequality_condition (m : ℕ) (a : ℝ) :
  (∀ x ≥ (1/2 : ℝ), f m x ≤ a * x + 8/9) →
  a ≥ 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_properties_inequality_condition_l399_39900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_approx_l399_39972

/-- The rate of cloth weaving in meters per second -/
noncomputable def weaving_rate (cloth_length : ℝ) (time : ℝ) : ℝ :=
  cloth_length / time

/-- Theorem: The loom weaves approximately 0.129 meters of cloth per second -/
theorem loom_weaving_rate_approx :
  let cloth_length : ℝ := 15
  let time : ℝ := 116.27906976744185
  abs (weaving_rate cloth_length time - 0.129) < 0.0001 := by
  -- Unfold the definitions
  unfold weaving_rate
  -- Simplify the expression
  simp
  -- Prove the approximation
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_approx_l399_39972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_restaurant_bill_l399_39978

/-- James' restaurant bill calculation -/
theorem james_restaurant_bill :
  ∀ (steak_egg_price chicken_fried_steak_price : ℚ),
    steak_egg_price = 16 →
    chicken_fried_steak_price = 14 →
    let total_bill := steak_egg_price + chicken_fried_steak_price
    let half_bill := total_bill / 2
    let tip_rate := (20 : ℚ) / 100
    let tip_amount := total_bill * tip_rate
    half_bill + tip_amount = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_restaurant_bill_l399_39978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l399_39949

theorem simplify_trig_expression (x : ℝ) 
  (h1 : 1 + Real.cos x ≠ 0) (h2 : Real.sin x ≠ 0) : 
  (Real.sin x) / (1 + Real.cos x) + (1 + Real.cos x) / (Real.sin x) = 2 / Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l399_39949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_difference_is_ten_l399_39965

/-- Represents the number of scoops of each ice cream flavor for a person's banana split. -/
structure BananaSplit where
  vanilla : ℕ
  chocolate : ℕ
  strawberry : ℕ

/-- Calculates the total difference in scoops between two banana splits. -/
def totalDifference (a b : BananaSplit) : ℕ :=
  (if a.vanilla ≥ b.vanilla then a.vanilla - b.vanilla else b.vanilla - a.vanilla) +
  (if a.chocolate ≥ b.chocolate then a.chocolate - b.chocolate else b.chocolate - a.chocolate) +
  (if a.strawberry ≥ b.strawberry then a.strawberry - b.strawberry else b.strawberry - a.strawberry)

/-- The banana splits of Oli, Victoria, and Brian. -/
def oli : BananaSplit := ⟨2, 1, 1⟩
def victoria : BananaSplit := ⟨4, 2, 2⟩
def brian : BananaSplit := ⟨3, 3, 1⟩

/-- The theorem stating that the total difference in scoops between all pairs of banana splits is 10. -/
theorem total_difference_is_ten :
  totalDifference oli victoria + totalDifference oli brian + totalDifference victoria brian = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_difference_is_ten_l399_39965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l399_39973

noncomputable section

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

/-- Definition of the foci of the ellipse -/
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c^2 = 12 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

/-- A point is on the ellipse -/
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2

/-- PF₁ is perpendicular to PF₂ -/
def perpendicular_lines (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

/-- The area of a triangle given its vertices -/
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem ellipse_triangle_area 
  (P F₁ F₂ : ℝ × ℝ) 
  (h_ellipse : point_on_ellipse P) 
  (h_foci : are_foci F₁ F₂) 
  (h_perp : perpendicular_lines P F₁ F₂) :
  triangle_area P F₁ F₂ = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l399_39973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l399_39953

-- Define the parametric equations of line l
noncomputable def line_l (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

-- Define the polar equation of circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the range of α
def α_range (α : ℝ) : Prop := Real.pi / 2 ≤ α ∧ α < Real.pi

-- Define point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the trajectory of point P
noncomputable def trajectory_P (θ : ℝ) : ℝ := Real.sin θ

-- State the theorem
theorem intersection_chord_length :
  ∃ α θ, α_range α ∧ 0 ≤ θ ∧ θ < Real.pi / 2 ∧
  circle_C θ = trajectory_P θ ∧
  circle_C θ = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l399_39953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_at_6_l399_39914

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | 1 => 0
  | n + 2 => a (n + 1) + 2 * (n + 2) - 1

noncomputable def b (n : ℕ) : ℝ := 
  Real.sqrt (a n + 1) * Real.sqrt (a (n + 1) + 1) * (8 / 11) ^ (n - 1)

theorem max_b_at_6 : ∀ n : ℕ, n ≥ 1 → b 6 ≥ b n := by
  sorry

#check max_b_at_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_at_6_l399_39914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_t_min_value_of_y_l399_39908

-- Define the inequality function
noncomputable def f (x t : ℝ) : ℝ := |2*x + 5| + |2*x - 1| - t

-- Define the maximum value of t
def s : ℝ := 6

-- Define the function y
noncomputable def y (a b : ℝ) : ℝ := 1 / (a + 2*b) + 4 / (3*a + 3*b)

-- Theorem 1: Maximum value of t
theorem max_value_of_t : ∀ x : ℝ, ∃ t : ℝ, f x t ≥ 0 ∧ t ≤ s := by
  sorry

-- Theorem 2: Minimum value of y
theorem min_value_of_y :
  ∀ a b : ℝ, a > 0 → b > 0 → 4*a + 5*b = s →
  y a b ≥ 3/2 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 4*a₀ + 5*b₀ = s ∧ y a₀ b₀ = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_t_min_value_of_y_l399_39908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_a_2019_l399_39912

/-- An arithmetic-geometric sequence -/
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem units_digit_a_2019 (a : ℕ → ℝ) :
  arithmetic_geometric_sequence a →
  (∀ n, a n > 0) →
  a 2^2 + a 4^2 = 900 - 2*a 1*a 5 →
  a 5 = 9*a 3 →
  Int.mod (Int.floor (a 2019)) 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_a_2019_l399_39912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_reciprocal_linear_l399_39948

/-- The domain of the function f(x) = 1/(x-5) is all real numbers except 5. -/
theorem domain_of_reciprocal_linear (f : ℝ → ℝ) :
  (∀ x, f x = 1 / (x - 5)) →
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 5} := by
  sorry

#check domain_of_reciprocal_linear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_reciprocal_linear_l399_39948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_chain_l399_39988

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define necessary functions
noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

noncomputable def inradius (t : Triangle) : ℝ := 
  let s := semiperimeter t
  (s - t.a) * (s - t.b) * (s - t.c) / s

noncomputable def altitude_a (t : Triangle) : ℝ := 2 * inradius t * semiperimeter t / t.a
noncomputable def altitude_b (t : Triangle) : ℝ := 2 * inradius t * semiperimeter t / t.b
noncomputable def altitude_c (t : Triangle) : ℝ := 2 * inradius t * semiperimeter t / t.c

noncomputable def tangent_a (t : Triangle) : ℝ := 2 * inradius t * semiperimeter t / (t.b + t.c - t.a)
noncomputable def tangent_b (t : Triangle) : ℝ := 2 * inradius t * semiperimeter t / (t.c + t.a - t.b)
noncomputable def tangent_c (t : Triangle) : ℝ := 2 * inradius t * semiperimeter t / (t.a + t.b - t.c)

noncomputable def exradius_a (t : Triangle) : ℝ := t.a / (semiperimeter t - t.a)
noncomputable def exradius_b (t : Triangle) : ℝ := t.b / (semiperimeter t - t.b)
noncomputable def exradius_c (t : Triangle) : ℝ := t.c / (semiperimeter t - t.c)

-- State the theorem
theorem triangle_inequality_chain (t : Triangle) :
  9 * inradius t ≤ 
  altitude_a t + altitude_b t + altitude_c t ∧
  altitude_a t + altitude_b t + altitude_c t ≤ 
  tangent_a t + tangent_b t + tangent_c t ∧
  tangent_a t + tangent_b t + tangent_c t ≤ 
  Real.sqrt 3 * semiperimeter t ∧
  Real.sqrt 3 * semiperimeter t ≤ 
  exradius_a t + exradius_b t + exradius_c t ∧
  exradius_a t + exradius_b t + exradius_c t ≤ 
  (t.a^2 + t.b^2 + t.c^2) / (4 * inradius t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_chain_l399_39988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l399_39942

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-10) 6

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := f (-3 * x + 1)

-- Theorem: The domain of h is [-5/3, 11/3]
theorem domain_of_h : 
  {x : ℝ | h x ∈ Set.range f} = Set.Icc (-5/3) (11/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l399_39942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l399_39931

-- Define the circle C
def circleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + 24 = 0

-- Define the points A and B
def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the perpendicularity condition
def perpendicular (P A B : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- State the theorem
theorem max_m_value :
  ∀ m : ℝ, (∃ P : ℝ × ℝ, circleC P.1 P.2 ∧ 
    perpendicular P (point_A m) (point_B m)) →
  m ≤ 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l399_39931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l399_39913

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * φ + 2 * Real.pi / 3)

theorem min_shift_for_odd_function :
  ∀ φ : ℝ, φ > 0 →
  (∀ x : ℝ, f x φ = -f (-x) φ) →
  φ ≥ Real.pi / 6 := by
  sorry

#check min_shift_for_odd_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l399_39913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abcd_value_l399_39938

/-- Represents the configuration of numbers in the diagram -/
structure Configuration :=
  (a b c d : Nat)
  (h1 : a ≤ 5 ∧ b ≤ 5 ∧ c ≤ 5 ∧ d ≤ 5)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

/-- Checks if the differences form the set {1, 2, 3, 4, 5} -/
def validDifferences (config : Configuration) : Prop :=
  let diffs := [
    (max config.a config.b) - (min config.a config.b),
    (max config.a config.c) - (min config.a config.c),
    (max config.a config.d) - (min config.a config.d),
    (max config.b config.c) - (min config.b config.c),
    (max config.d config.c) - (min config.d config.c)
  ]
  (∀ n, n ∈ [1, 2, 3, 4, 5] → n ∈ diffs) ∧
  (∀ d, d ∈ diffs → d ∈ [1, 2, 3, 4, 5])

/-- Calculates the four-digit number ABCD -/
def abcdValue (config : Configuration) : Nat :=
  1000 * config.a + 100 * config.b + 10 * config.c + config.d

/-- Main theorem: The maximum ABCD value is 5304 -/
theorem max_abcd_value :
  ∀ config : Configuration,
    validDifferences config →
    abcdValue config ≤ 5304 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abcd_value_l399_39938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l399_39958

noncomputable section

def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 3) - 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ k : ℤ, ∃ c : ℝ, c = k * Real.pi / 2 - Real.pi / 6 ∧ ∀ x : ℝ, f (c + x) = f (c - x)) ∧
  (∀ y : ℝ, y ∈ Set.Ioo (-1/2 : ℝ) 1 ↔ ∃ x : ℝ, x ∈ Set.Ioc (-Real.pi/4 : ℝ) (Real.pi/4) ∧ f x = y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l399_39958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_zeros_l399_39947

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) + 1

-- State the theorem
theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ π ∧ 0 ≤ x₂ ∧ x₂ ≤ π ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x, 0 ≤ x ∧ x ≤ π ∧ f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

-- Additional lemma to show that π/12 and 3π/4 are the zeros
theorem f_zeros :
  f (π/12) = 0 ∧ f (3*π/4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_f_zeros_l399_39947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolicArchHeightAtTen_l399_39996

/-- Represents a parabolic arch -/
structure ParabolicArch where
  a : ℝ  -- Curvature constant
  k : ℝ  -- Maximum height

/-- Height of the arch at a given horizontal distance from the center -/
def archHeight (arch : ParabolicArch) (x : ℝ) : ℝ :=
  arch.a * x^2 + arch.k

/-- Creates a parabolic arch given its maximum height and span -/
noncomputable def createArch (maxHeight : ℝ) (span : ℝ) : ParabolicArch :=
  let halfSpan := span / 2
  let a := -(maxHeight / (halfSpan^2))
  { a := a, k := maxHeight }

theorem parabolicArchHeightAtTen (maxHeight span : ℝ) 
  (hMaxHeight : maxHeight = 20)
  (hSpan : span = 50) :
  let arch := createArch maxHeight span
  archHeight arch 10 = 16.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolicArchHeightAtTen_l399_39996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_existence_l399_39976

noncomputable section

/-- Given three lines in a 2D plane --/
def l₁ (a : ℝ) (x y : ℝ) : Prop := 2 * x - y + a = 0
def l₂ (x y : ℝ) : Prop := -4 * x + 2 * y + 1 = 0
def l₃ (x y : ℝ) : Prop := x + y - 1 = 0

/-- Distance between two parallel lines --/
noncomputable def distance_between_lines (a : ℝ) : ℝ := (7 / 10) * Real.sqrt 5

/-- Distance from a point to a line --/
noncomputable def distance_to_line (x y : ℝ) (line : ℝ → ℝ → Prop) : ℝ := 
  sorry

/-- Theorem stating the existence of a point P satisfying all conditions --/
theorem point_existence (a : ℝ) : 
  (∃ x y : ℝ, 
    x > 0 ∧ y > 0 ∧  -- P is in the first quadrant
    distance_to_line x y (l₁ a) = (1/2) * distance_to_line x y l₂ ∧  -- Condition (ii)
    distance_to_line x y (l₁ a) / distance_to_line x y l₃ = Real.sqrt 2 / Real.sqrt 5) ∧  -- Condition (iii)
  a = 3 ∧  -- Value of a
  (∃ x y : ℝ, x = 1/9 ∧ y = 37/18 ∧  -- Coordinates of P
    x > 0 ∧ y > 0 ∧
    distance_to_line x y (l₁ a) = (1/2) * distance_to_line x y l₂ ∧
    distance_to_line x y (l₁ a) / distance_to_line x y l₃ = Real.sqrt 2 / Real.sqrt 5) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_existence_l399_39976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bounces_below_target_l399_39929

noncomputable def initial_height : ℝ := 15
noncomputable def bounce_ratio : ℝ := 2/3
noncomputable def target_height : ℝ := 2

noncomputable def height_after_bounces (k : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ k)

theorem smallest_bounces_below_target :
  ∀ k : ℕ, (∀ j : ℕ, j < k → height_after_bounces j ≥ target_height) ∧
           height_after_bounces k < target_height ↔ k = 3 :=
by sorry

#check smallest_bounces_below_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bounces_below_target_l399_39929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l399_39918

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x^2 - a*x + a
  else -((-x)^2 - a*(-x) + a)

-- State the theorem
theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is odd
  f a (-1) = -1 ∧
  (Set.range (f a) = Set.univ → a ∈ Set.Iic 0 ∪ Set.Ici 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l399_39918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_perpendicular_to_same_line_l399_39910

-- Define the necessary structures
structure Line : Type

structure Plane : Type

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (p1 p2 : Plane) : Prop := sorry

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) : 
  perpendicular m α → perpendicular m β → parallel α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_perpendicular_to_same_line_l399_39910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_properties_l399_39959

/-- Service life data points (in years) -/
def x : List ℝ := [2, 4, 6, 8, 10]

/-- Residual value data points (in ten thousand yuan) -/
def y : List ℝ := [17, 16, 14, 13, 11]

/-- The y-intercept of the regression line -/
def a : ℝ := 18.7

/-- The slope of the regression line -/
noncomputable def b : ℝ := (List.sum (List.zipWith (· * ·) x y) - (List.sum x * List.sum y) / x.length) /
                           (List.sum (List.map (λ xi => xi * xi) x) - (List.sum x)^2 / x.length)

/-- The regression line equation -/
noncomputable def regression_line (x : ℝ) : ℝ := b * x + a

theorem regression_line_properties :
  (regression_line 6 = 14.2) ∧ (b < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_properties_l399_39959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l399_39981

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- State the theorem
theorem f_max_value :
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ x : ℝ, f x = 2) := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l399_39981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_eighth_term_l399_39927

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- The common ratio
  h : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumGeometric (seq : GeometricSequence) (n : ℕ) : ℝ :=
  if seq.q = 1 then
    n * seq.a 1
  else
    seq.a 1 * (1 - seq.q^n) / (1 - seq.q)

theorem geometric_sequence_eighth_term
  (seq : GeometricSequence)
  (h1 : sumGeometric seq 3 + sumGeometric seq 9 = 2 * sumGeometric seq 6)
  (h2 : seq.a 2 + seq.a 5 = 4) :
  seq.a 8 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_eighth_term_l399_39927
