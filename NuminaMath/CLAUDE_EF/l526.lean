import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_from_sin_shift_l526_52654

theorem cos_2x_from_sin_shift (x : ℝ) : 
  Real.cos (2 * x) = Real.sin (2 * (x + π / 3) - π / 6) :=
by
  sorry

#check cos_2x_from_sin_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_from_sin_shift_l526_52654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_with_asymptote_l526_52662

/-- The denominator of our rational function -/
noncomputable def q (x : ℝ) : ℝ := 3*x^6 - x^5 + 2*x^3 - 3

/-- A general polynomial of degree n -/
noncomputable def polynomial (n : ℕ) (x : ℝ) : ℝ → ℝ := λ (a : ℝ) ↦ a*x^n

/-- The rational function with a polynomial numerator of degree n -/
noncomputable def f (n : ℕ) (x : ℝ) : ℝ → ℝ := λ (a : ℝ) ↦ (polynomial n x a) / (q x)

/-- A function has a horizontal asymptote if its limit as x approaches infinity exists -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ N, ∀ x > N, |f x - L| < ε

/-- The main theorem: the largest degree of p(x) that allows for a horizontal asymptote is 6 -/
theorem largest_degree_with_asymptote :
  (∀ n > 6, ¬∃ a : ℝ, has_horizontal_asymptote (f n · a)) ∧
  (∃ a : ℝ, has_horizontal_asymptote (f 6 · a)) := by
  sorry

#check largest_degree_with_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_with_asymptote_l526_52662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l526_52690

-- Define the geometric series sum function T
noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

-- Main theorem
theorem geometric_series_sum_property (b : ℝ) 
  (h1 : -1 < b) (h2 : b < 1) 
  (h3 : T b * T (-b) = 2430) : 
  T b + T (-b) = 324 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l526_52690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l526_52696

/-- Given a hyperbola C with equation x²/m - y² = 1 where m > 0,
    and its asymptote √3x + my = 0, the focal length of C is 4. -/
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) : 
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 / m - y^2 = 1}
  let asymptote : Set (ℝ × ℝ) := {(x, y) | Real.sqrt 3 * x + m * y = 0}
  4 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l526_52696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt3_over_2_l526_52694

theorem cos_squared_difference_equals_sqrt3_over_2 :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt3_over_2_l526_52694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_range_l526_52611

theorem cosine_equation_range (m : ℝ) : 
  (∃ x : ℝ, Real.cos x = 1 - m) ↔ 0 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_range_l526_52611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vector_sum_l526_52682

-- Define the plane and points
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (O A B C : V)

-- Define the collinearity condition
def collinear (A B C : V) : Prop :=
  ∃ (t : ℝ), C - A = t • (B - A)

-- State the theorem
theorem collinear_vector_sum 
  (h_collinear : collinear A B C)
  (h_vector_sum : C - O = x • (A - O) + y • (B - O)) :
  x + y = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vector_sum_l526_52682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_two_implies_special_trig_sum_l526_52671

theorem tan_sqrt_two_implies_special_trig_sum (α : ℝ) :
  Real.tan α = Real.sqrt 2 → (1/3) * (Real.sin α)^2 + (Real.cos α)^2 = 5/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_two_implies_special_trig_sum_l526_52671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_AB_l526_52644

/-- The distance from the origin to a line passing through two points -/
noncomputable def distanceFromOriginToLine (x1 y1 x2 y2 : ℝ) : ℝ :=
  let a := y2 - y1
  let b := x1 - x2
  let c := x2 * y1 - x1 * y2
  abs c / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the origin to the line passing through A(0, 6) and B(-8, 0) is 24/5 -/
theorem distance_origin_to_AB : distanceFromOriginToLine 0 6 (-8) 0 = 24 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_AB_l526_52644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_sin_squared_l526_52633

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2

-- State the theorem
theorem second_derivative_sin_squared (x : ℝ) :
  (deriv (deriv f)) x = 2 * Real.cos (2 * x) := by
  sorry

#check second_derivative_sin_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_sin_squared_l526_52633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_path_probability_l526_52658

/-- Represents a cube with diagonal stripes on each face -/
structure StripedCube where
  faces : Fin 6 → Bool

/-- A path from one vertex to the diagonally opposite vertex -/
inductive DiagonalPath
| path1
| path2

/-- Checks if a given path is continuous based on the cube's stripes -/
def isContinuousPath (cube : StripedCube) (path : DiagonalPath) : Bool := sorry

/-- Counts the number of cubes with at least one continuous diagonal path -/
def countContinuousPaths (cubes : List StripedCube) : Nat := sorry

/-- The total number of possible stripe combinations -/
def totalCombinations : Nat := 2^6

/-- Theorem: The probability of at least one continuous diagonal path is 3/64 -/
theorem continuous_path_probability :
  let allCubes := List.map StripedCube.mk (List.range (2^6))
  (countContinuousPaths allCubes : Rat) / totalCombinations = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_path_probability_l526_52658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_number_of_zeros_l526_52639

/-- The function f(x) = e^x - ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

/-- The derivative of f(x) --/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem tangent_line_through_origin (a : ℝ) :
  a = 1 → ∃ m : ℝ, f a m = 0 ∧ f_derivative a m * m = f a m := by
  sorry

theorem number_of_zeros (a : ℝ) :
  (0 ≤ a ∧ a < Real.exp 1 → ∀ x, f a x ≠ 0) ∧
  ((a < 0 ∨ a = Real.exp 1) → ∃! x, f a x = 0) ∧
  (a > Real.exp 1 → ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_number_of_zeros_l526_52639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_inequality_l526_52608

theorem sin_cos_sum_inequality (α β a b : ℝ) 
  (h1 : π/4 < α) (h2 : α < β) (h3 : β < π/2)
  (h4 : Real.sin α + Real.cos α = a) (h5 : Real.sin β + Real.cos β = b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_inequality_l526_52608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l526_52635

noncomputable def f (x : ℝ) := Real.sin (2 * x) - 4 * (Real.sin x)^3 * Real.cos x

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi / 2 := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l526_52635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_product_l526_52627

theorem parallel_vectors_product (x z : ℝ) : 
  let a : ℝ × ℝ × ℝ := (x, 4, 3)
  let b : ℝ × ℝ × ℝ := (3, 2, z)
  (∃ (l : ℝ), a = l • b) → x * z = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_product_l526_52627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_iff_t_in_open_1_closed_2_l526_52688

-- Define the piecewise function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4*x

-- State the theorem
theorem three_roots_iff_t_in_open_1_closed_2 (t : ℝ) :
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f t x = x - 6) ↔ t ∈ Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_iff_t_in_open_1_closed_2_l526_52688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l526_52619

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 ∧ y > 0 → f x - f (x + y) = f (x / y) * f (x + y)

theorem function_characterization (f : ℝ → ℝ) :
  functional_equation f →
  ((∀ x : ℝ, x > 0 → f x = 0) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l526_52619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_problem_l526_52600

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given conditions and definitions -/
def P : Point := ⟨2, 2⟩
def O : Point := ⟨0, 0⟩
def C : Circle := ⟨⟨0, 4⟩, 4⟩

/-- Trajectory of point M -/
def trajectory_M (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 2

/-- Line l when |OP| = |OM| -/
def line_l : Line := ⟨1, 3, -8⟩

/-- Area of triangle POM -/
noncomputable def area_POM : ℝ := 16/5

/-- Main theorem -/
theorem geometric_problem :
  (∀ x y : ℝ, trajectory_M x y ↔ 
    ∃ A B : Point, ∃ l : Line,
      (l.a * A.x + l.b * A.y + l.c = 0) ∧
      (l.a * B.x + l.b * B.y + l.c = 0) ∧
      (l.a * P.x + l.b * P.y + l.c = 0) ∧
      ((A.x - B.x)^2 + (A.y - B.y)^2 = 4 * ((x - A.x)^2 + (y - A.y)^2)) ∧
      (A.x^2 + A.y^2 - 8*A.y = 0) ∧
      (B.x^2 + B.y^2 - 8*B.y = 0)) ∧
  (∀ M : Point, trajectory_M M.x M.y →
    (M.x - O.x)^2 + (M.y - O.y)^2 = (P.x - O.x)^2 + (P.y - O.y)^2 →
    (line_l.a * M.x + line_l.b * M.y + line_l.c = 0) ∧
    (line_l.a * P.x + line_l.b * P.y + line_l.c = 0) ∧
    (area_POM = 1/2 * |line_l.a * P.x + line_l.b * P.y + line_l.c| / 
      Real.sqrt (line_l.a^2 + line_l.b^2) * 
      Real.sqrt ((P.x - O.x)^2 + (P.y - O.y)^2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_problem_l526_52600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rearrangement_l526_52676

theorem clock_rearrangement (p : Fin 12 → Fin 12) (h : Function.Bijective p) :
  ∃ i : Fin 12, (p i).val + (p (i + 1)).val + (p (i + 2)).val ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rearrangement_l526_52676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_average_l526_52625

def average (s : Finset ℚ) : ℚ :=
  s.sum id / s.card

theorem set_average (T : Finset ℚ) (m : ℕ) (b₁ bₘ : ℚ) : 
  T.Nonempty →
  (∀ x ∈ T, x > 0) →
  T.card = m →
  b₁ ∈ T →
  bₘ ∈ T →
  (∀ x ∈ T, b₁ ≤ x ∧ x ≤ bₘ) →
  average (T \ {bₘ}) = 45 →
  average ((T \ {b₁}) \ {bₘ}) = 50 →
  average (T \ {b₁}) = 55 →
  bₘ = b₁ + 80 →
  average T = 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_average_l526_52625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l526_52691

noncomputable def series (n : ℕ) : ℝ := (4 * n - 2) / (3 ^ n)

theorem series_sum : ∑' n, series n = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l526_52691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l526_52629

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point D on BC
def D (t : Triangle) (r : ℝ) : ℝ × ℝ :=
  (r * t.B.1 + (1 - r) * t.C.1, r * t.B.2 + (1 - r) * t.C.2)

-- Define the angle between two points and a vertex
noncomputable def angle (p q v : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_angle_ratio (t : Triangle) :
  angle t.B t.A t.C = 70 * π / 180 →
  angle t.C t.A t.B = 40 * π / 180 →
  let d := D t (1/5)
  Real.sin (angle t.B t.A d) / Real.sin (angle t.C t.A d) = 
    Real.sin (70 * π / 180) / (4 * Real.sin (40 * π / 180)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l526_52629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_l526_52631

/-- The line mx - y - 2m - 1 = 0 for some real m -/
def line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1 - p.2 - 2*m - 1 = 0}

/-- A circle with center (-2, 3) and radius r -/
def circleWithRadius (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 - 3)^2 = r^2}

/-- A circle is tangent to a line if there exists exactly one point in their intersection -/
def isTangent (c : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ c ∩ l

theorem largest_circle : 
  ∀ r : ℝ, (∃ m : ℝ, isTangent (circleWithRadius r) (line m)) → 
  r ≤ Real.sqrt 20 ∧ (r = Real.sqrt 20 → circleWithRadius r = circleWithRadius (Real.sqrt 20)) :=
by
  sorry

#check largest_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_l526_52631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_landing_probability_l526_52683

noncomputable section

/-- The side length of the octagonal dart board -/
def octagon_side : ℝ := 1

/-- The side length of the hexagon at the center of the board -/
noncomputable def hexagon_side : ℝ := octagon_side / Real.sqrt 2

/-- The area of the regular hexagon at the center of the board -/
noncomputable def hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * hexagon_side^2

/-- The area of the octagonal dart board -/
noncomputable def octagon_area : ℝ := 2 * (1 + Real.sqrt 2) * octagon_side^2

/-- The probability of a dart landing in the center hexagon -/
noncomputable def landing_probability : ℝ := hexagon_area / octagon_area

theorem dart_landing_probability :
  landing_probability = (3 * Real.sqrt 3) / (8 * (1 + Real.sqrt 2)) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_landing_probability_l526_52683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perpendicular_constant_chord_l526_52698

-- Define the curve
def curve (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2

-- Define the points A and B as the x-axis intersections
def points_AB (m : ℝ) : Set ℝ := {x | curve m x = 0}

-- Define point C
def point_C : ℝ × ℝ := (0, 1)

-- Theorem 1: AC is not perpendicular to BC for any m
theorem not_perpendicular (m : ℝ) : 
  ∀ x₁ x₂, x₁ ∈ points_AB m → x₂ ∈ points_AB m → x₁ ≠ x₂ → 
    (1 - 0) / (0 - x₁) * (1 - 0) / (0 - x₂) ≠ -1 :=
sorry

-- Define the circle passing through A, B, and C
def circle_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + m*x + y - 2 = 0

-- Theorem 2: The circle intersects the y-axis at (0, 1) and (0, -2)
theorem constant_chord (m : ℝ) :
  ∀ x₁ x₂, x₁ ∈ points_AB m → x₂ ∈ points_AB m → x₁ ≠ x₂ →
    (circle_equation m 0 1 ∧ circle_equation m 0 (-2)) ∧
    ∀ y, circle_equation m 0 y → y = 1 ∨ y = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perpendicular_constant_chord_l526_52698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_OC_AB_perpendicular_AC_BC_angle_OB_OC_l526_52646

-- Define the points
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 3)
noncomputable def C (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def O : ℝ × ℝ := (0, 0)

-- Define vectors
noncomputable def OC (α : ℝ) : ℝ × ℝ := C α
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
noncomputable def AC (α : ℝ) : ℝ × ℝ := ((C α).1 - A.1, (C α).2 - A.2)
noncomputable def BC (α : ℝ) : ℝ × ℝ := ((C α).1 - B.1, (C α).2 - B.2)
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B

-- Theorem 1
theorem parallel_OC_AB (α : ℝ) : 
  (∃ k : ℝ, OC α = k • AB) → Real.tan α = -1 := by sorry

-- Theorem 2
theorem perpendicular_AC_BC (α : ℝ) : 
  (AC α).1 * (BC α).1 + (AC α).2 * (BC α).2 = 0 → Real.sin (2 * α) = -8/9 := by sorry

-- Theorem 3
theorem angle_OB_OC (α : ℝ) : 
  α ∈ Set.Ioo 0 Real.pi → 
  (OA.1 + (OC α).1)^2 + (OA.2 + (OC α).2)^2 = 13 → 
  Real.arccos ((OB.1 * (OC α).1 + OB.2 * (OC α).2) / (Real.sqrt (OB.1^2 + OB.2^2) * Real.sqrt ((OC α).1^2 + (OC α).2^2))) = Real.pi/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_OC_AB_perpendicular_AC_BC_angle_OB_OC_l526_52646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_length_100_l526_52607

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Sequence of points A_n on the x-axis -/
noncomputable def A : ℕ → Point
  | 0 => ⟨0, 0⟩
  | n + 1 => ⟨(2 * (n + 1)) / 3, 0⟩

/-- Sequence of points B_n on y = √x -/
noncomputable def B : ℕ+ → Point
  | n => ⟨(2 * n) / 3, Real.sqrt ((2 * n) / 3)⟩

/-- Length between two points -/
noncomputable def length (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The main theorem -/
theorem least_n_for_length_100 :
  (∀ n : ℕ+, length (A (n-1)) (B n) = length (B n) (A n) ∧
             length (A (n-1)) (A n) = length (B n) (A n)) →
  (∀ n < 17, length (A 0) (A n) < 100) ∧
  length (A 0) (A 17) ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_length_100_l526_52607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_equals_fifty_l526_52640

/-- The perimeter of a rectangle with given height and width -/
noncomputable def rectangle_perimeter (height width : ℝ) : ℝ := 2 * (height + width)

/-- The side length of a square with a given perimeter -/
noncomputable def square_side_from_perimeter (perimeter : ℝ) : ℝ := perimeter / 4

theorem square_side_equals_fifty :
  let rect_height : ℝ := 36
  let rect_width : ℝ := 64
  let rect_perimeter := rectangle_perimeter rect_height rect_width
  let square_side := square_side_from_perimeter rect_perimeter
  square_side = 50 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_equals_fifty_l526_52640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_balls_in_original_position_l526_52693

/-- Represents a ball in the circle -/
structure Ball :=
  (position : Fin 6)

/-- Represents a swap of two adjacent balls -/
def Swap := Fin 6

/-- The game state -/
structure GameState :=
  (balls : Fin 6 → Ball)

/-- Perform a swap on the game state -/
def performSwap (state : GameState) (swap : Swap) : GameState :=
  sorry

/-- Check if a ball is in its original position -/
def isInOriginalPosition (state : GameState) (pos : Fin 6) : Bool :=
  sorry

/-- The probability of a specific swap being chosen -/
noncomputable def swapProbability : ℝ := 1 / 6

/-- The expected number of balls in their original positions after three swaps -/
noncomputable def expectedBallsInOriginalPosition : ℝ :=
  sorry

theorem expected_balls_in_original_position :
  expectedBallsInOriginalPosition = 2 :=
by sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_balls_in_original_position_l526_52693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_fourth_power_abs_l526_52615

theorem greatest_fourth_power_abs (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : z₁ = -1 - I)
  (h₂ : z₂ = -2 * I)
  (h₃ : z₃ = Real.sqrt 3 - I)
  (h₄ : z₄ = 1 - 2 * I)
  (h₅ : z₅ = 3) :
  Complex.abs (z₅^4) = max (Complex.abs (z₁^4)) (max (Complex.abs (z₂^4)) (max (Complex.abs (z₃^4)) (max (Complex.abs (z₄^4)) (Complex.abs (z₅^4))))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_fourth_power_abs_l526_52615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_quadrilateral_area_l526_52605

/-- Represents a square with side length 8 cm -/
structure Square where
  side : ℝ
  is_eight : side = 8

/-- Represents a point on the square -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the central quadrilateral formed by the partition -/
structure CentralQuadrilateral (sq : Square) where
  pointA : Point
  pointB : Point
  is_midpoint_A : pointA.x = sq.side / 2 ∧ pointA.y = sq.side
  is_midpoint_B : pointB.x = sq.side / 2 ∧ pointB.y = 0

/-- The theorem to be proved -/
theorem central_quadrilateral_area 
  (sq : Square) 
  (quad : CentralQuadrilateral sq) : 
  ∃ (area : ℝ), area = 16 ∧ 
  area = (sq.side^2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_quadrilateral_area_l526_52605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l526_52604

noncomputable def f (φ : Real) (x : Real) : Real := Real.sin (2 * x + φ)

noncomputable def translated_f (φ : Real) (x : Real) : Real := f φ (x + Real.pi/6)

theorem min_phi_value (φ : Real) :
  (∃ x₁ x₂ : Real, x₁ < 0 ∧ x₂ > 0 ∧
    translated_f φ x₁ = 0 ∧
    translated_f φ x₂ = 0 ∧
    x₁ = -x₂ ∧
    (∀ x : Real, translated_f φ x = 0 → |x| ≥ |x₁|)) →
  |φ| ≥ Real.pi/6 ∧ (∀ ψ : Real, |ψ| < Real.pi/6 →
    ¬(∃ x₁ x₂ : Real, x₁ < 0 ∧ x₂ > 0 ∧
      translated_f ψ x₁ = 0 ∧
      translated_f ψ x₂ = 0 ∧
      x₁ = -x₂ ∧
      (∀ x : Real, translated_f ψ x = 0 → |x| ≥ |x₁|))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l526_52604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_special_composites_l526_52647

theorem infinite_special_composites :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧
    (∀ l : ℕ, 
      let n := 7^(3^l) - 3^(3^l)
      n > 0 ∧
      ¬ Nat.Prime n ∧
      (7^(n - 1) - 3^(n - 1)) % n = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_special_composites_l526_52647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l526_52632

/-- Calculates the time (in seconds) for two trains to meet --/
noncomputable def time_to_meet (length1 length2 initial_distance speed1 speed2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := initial_distance + length1 + length2
  total_distance / relative_speed

/-- Theorem stating that the time for the trains to meet is approximately 20.67 seconds --/
theorem trains_meet_time :
  let length1 := (100 : ℝ)
  let length2 := (200 : ℝ)
  let initial_distance := (630 : ℝ)
  let speed1 := (90 : ℝ)
  let speed2 := (72 : ℝ)
  abs ((time_to_meet length1 length2 initial_distance speed1 speed2) - 20.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l526_52632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_range_l526_52687

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (3*a - 1)*x + 4*a else a/x

theorem decreasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Icc (1/6) (1/3) ∧ a ≠ 1/3 := by
  sorry

#check decreasing_function_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_range_l526_52687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l526_52613

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g
def g (f : ℝ → ℝ) (m : ℝ) (x : ℝ) : ℝ := f x - m * x

-- Define the function h
def h (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := (f x - x^2 + 2) * |x - a|

theorem quadratic_function_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_diff : ∀ x, f a b c (x + 1) - f a b c x = 2 * x + 2)
  (h_f0 : f a b c 0 = -2)
  (m : ℝ)
  (h_max : ∀ x ∈ Set.Icc 1 2, g (f a b c) m x ≤ 3)
  (h_exists : ∃ x ∈ Set.Icc 1 2, g (f a b c) m x = 3) :
  (f a b c = λ x => x^2 + x - 2) ∧
  (m = 1/2) ∧
  (a = 0 ∨ a ≥ 4 ∨ a ≤ -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l526_52613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_product_l526_52689

theorem smallest_sum_product (a : Fin 95 → Int) 
  (h1 : ∀ i, a i = 1 ∨ a i = -1) : 
  (Finset.sum (Finset.range 95) (λ i => 
    Finset.sum (Finset.range 95) (λ j => 
      if j > i then a i * a j else 0))) ≥ 13 ∧ 
  ∃ a₀ : Fin 95 → Int, (∀ i, a₀ i = 1 ∨ a₀ i = -1) ∧ 
    (Finset.sum (Finset.range 95) (λ i => 
      Finset.sum (Finset.range 95) (λ j => 
        if j > i then a₀ i * a₀ j else 0))) = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_product_l526_52689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_plus_alpha_eq_two_implies_fraction_eq_one_third_l526_52661

theorem tan_pi_plus_alpha_eq_two_implies_fraction_eq_one_third (α : ℝ) :
  Real.tan (π + α) = 2 →
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π + α)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_plus_alpha_eq_two_implies_fraction_eq_one_third_l526_52661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_price_increase_l526_52618

/-- Represents the percentage increase in price of kerosene oil -/
def price_increase_percent : ℝ → Prop := sorry

/-- Represents the percentage decrease in consumption of kerosene oil -/
def consumption_decrease_percent : ℝ → Prop := sorry

theorem kerosene_price_increase 
  (h1 : consumption_decrease_percent 20)
  (h2 : ∀ (old_price old_consumption new_price : ℝ),
    old_price > 0 → old_consumption > 0 → new_price > old_price →
    old_price * old_consumption = 
    new_price * (old_consumption * (1 - 20 / 100))) :
  price_increase_percent 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_price_increase_l526_52618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properly_colored_squares_even_l526_52679

/-- Represents the colors used in the grid -/
inductive Color
| Blue
| Green
| Red
deriving Repr, DecidableEq

/-- Represents a vertex in the grid -/
structure Vertex where
  x : Nat
  y : Nat
  color : Color
deriving Repr

/-- Represents the grid -/
structure Grid where
  m : Nat
  n : Nat
  vertices : List Vertex

/-- Checks if a vertex is on the boundary of the grid -/
def isBoundary (g : Grid) (v : Vertex) : Bool :=
  v.x = 0 || v.x = g.m - 1 || v.y = 0 || v.y = g.n - 1

/-- Checks if a square is properly colored -/
def isProperlyColored (g : Grid) (x y : Nat) : Bool :=
  let vertices := [
    g.vertices.find? (fun v => v.x = x && v.y = y),
    g.vertices.find? (fun v => v.x = x + 1 && v.y = y),
    g.vertices.find? (fun v => v.x = x && v.y = y + 1),
    g.vertices.find? (fun v => v.x = x + 1 && v.y = y + 1)
  ]
  match vertices with
  | [some v1, some v2, some v3, some v4] =>
    let colors := [v1.color, v2.color, v3.color, v4.color]
    colors.toFinset.card = 3 &&
    ((v1.color = v2.color) || (v2.color = v4.color) || (v4.color = v3.color) || (v3.color = v1.color))
  | _ => false

/-- Counts the number of properly colored squares in the grid -/
def countProperlyColoredSquares (g : Grid) : Nat :=
  (List.range g.m).foldl (fun acc x =>
    acc + (List.range g.n).foldl (fun inner_acc y =>
      inner_acc + if isProperlyColored g x y then 1 else 0
    ) 0
  ) 0

/-- The main theorem to be proved -/
theorem properly_colored_squares_even (g : Grid) 
  (h1 : ∀ v ∈ g.vertices, isBoundary g v → v.color = Color.Red) :
  Even (countProperlyColoredSquares g) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_properly_colored_squares_even_l526_52679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polio_cases_1990_l526_52634

/-- Calculates the number of polio cases in a given year, assuming a linear decrease from 1970 to 2000 -/
def polioCases (year : ℕ) : ℕ :=
  let initialYear : ℕ := 1970
  let finalYear : ℕ := 2000
  let initialCases : ℕ := 300000
  let finalCases : ℕ := 600
  let slope : ℚ := (finalCases - initialCases : ℚ) / (finalYear - initialYear)
  let intercept : ℚ := initialCases - slope * initialYear
  ((slope * year + intercept).floor : ℤ).toNat

theorem polio_cases_1990 : polioCases 1990 = 100400 := by
  sorry

#eval polioCases 1990

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polio_cases_1990_l526_52634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_hyperbola_from_ellipse_foci_l526_52621

noncomputable def hyperbola (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / a^2 - y^2 / b^2 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola a b 2 (Real.sqrt 2) ∧
  eccentricity a (Real.sqrt (a^2 + b^2)) = Real.sqrt 2 →
  hyperbola (Real.sqrt 2) (Real.sqrt 2) = hyperbola a b :=
by
  sorry

-- For the second part of the problem
noncomputable def ellipse (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / a^2 + y^2 / b^2 = 1

theorem hyperbola_from_ellipse_foci :
  ∃ a b : ℝ, ellipse (Real.sqrt 8) 2 = ellipse a b ∧
             hyperbola 1 (Real.sqrt 3) = λ x y => x^2 - y^2 / 3 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_hyperbola_from_ellipse_foci_l526_52621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52650

noncomputable def f (x : ℝ) := (x + 1) / Real.exp (2 * x)

theorem f_properties :
  (∃ (m : ℝ), ∀ (x : ℝ), x ≥ 0 → f x ≤ m / (x + 1)) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ≥ 0 → f x ≤ m / (x + 1)) → m ≥ 1) ∧
  (∀ (x a : ℝ), x ≥ 0 → a ≤ 2 → f x * Real.log (2 * x + a) < x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l526_52669

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 1)

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x > -1 ∧ f x = y :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l526_52669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l526_52659

/-- Represents the journey of a train with two parts -/
structure TrainJourney where
  x : ℝ  -- Distance of the first part in miles
  v1 : ℝ  -- Speed of the first part in mph
  v2 : ℝ  -- Speed of the second part in mph

/-- Calculates the total time of the journey in hours -/
noncomputable def totalTime (j : TrainJourney) : ℝ :=
  j.x / j.v1 + (3 * j.x) / j.v2

/-- Theorem stating that for the given journey, the total time is 23x/400 hours -/
theorem train_journey_time (j : TrainJourney) 
    (h1 : j.v1 = 50) 
    (h2 : j.v2 = 80) : 
  totalTime j = 23 * j.x / 400 := by
  sorry

#check train_journey_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l526_52659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_difference_l526_52685

noncomputable section

-- Define the curves in polar coordinates
def C₁ (θ : ℝ) : ℝ := 2 * Real.cos θ

def C₂ (θ : ℝ) : ℝ := 4 * Real.cos θ / (Real.sin θ)^2

-- Define curve C in parametric form
def C (t : ℝ) : ℝ × ℝ := (2 + t/2, Real.sqrt 3 * t/2)

-- Define the intersection points
variable (t₁ t₂ t₃ t₄ : ℝ)

-- Assume the points are distinct and in order
axiom distinct_points : t₁ < t₂ ∧ t₂ < t₃ ∧ t₃ < t₄

-- Define the distance between two points on curve C
def distance (t₁ t₂ : ℝ) : ℝ := |t₂ - t₁|

theorem intersection_difference : 
  |distance t₁ t₂ - distance t₃ t₄| = 11/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_difference_l526_52685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l526_52657

-- Define the vectors
noncomputable def a (α : ℝ) : ℝ × ℝ := (4 * Real.cos α, Real.sin α)
noncomputable def b (β : ℝ) : ℝ × ℝ := (Real.sin β, 4 * Real.cos β)
noncomputable def c (β : ℝ) : ℝ × ℝ := (Real.cos β, -Real.sin β)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Define parallelism
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = (k * w.1, k * w.2)

theorem problem_solution (α β : ℝ) :
  (perpendicular (a α) (b β - 2 • (c β)) → Real.tan (α + β) = 2) ∧
  (∃ (max_value : ℝ), max_value = 4 * Real.sqrt 2 ∧
    ∀ β, Real.sqrt ((Real.sin β + Real.cos β)^2 + (4 * Real.cos β - 4 * Real.sin β)^2) ≤ max_value) ∧
  (Real.tan α * Real.tan β = 16 → parallel (a α) (b β)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l526_52657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l526_52656

/-- If the point (2, 8) lies on the graph of a power function y = x^a, then a = 3 -/
theorem power_function_through_point (a : ℝ) : (2 : ℝ)^a = 8 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l526_52656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l526_52626

noncomputable def vector_a : Fin 3 → ℝ := ![4, 5, 1]
noncomputable def vector_b : Fin 3 → ℝ := ![2, 6, 3]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem cosine_of_angle_between_vectors :
  (dot_product vector_a vector_b) / (magnitude vector_a * magnitude vector_b) = 41 / (7 * Real.sqrt 42) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l526_52626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_thirty_l526_52680

theorem determinant_zero_implies_sum_thirty (a b : ℝ) (h1 : a ≠ b) 
  (h2 : Matrix.det !![2, 5, 10; 4, a, b; 4, b, a] = 0) : 
  a + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_thirty_l526_52680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_problem_l526_52630

/-- Point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculate the distance between two points given in polar coordinates -/
noncomputable def distancePolar (p1 p2 : PolarPoint) : ℝ :=
  Real.sqrt (p1.r^2 + p2.r^2 - 2 * p1.r * p2.r * Real.cos (p1.θ - p2.θ))

/-- Calculate the area of a triangle given two points in polar coordinates and the pole -/
noncomputable def triangleAreaPolar (p1 p2 : PolarPoint) : ℝ :=
  (1 / 2) * p1.r * p2.r * Real.sin (p2.θ - p1.θ)

theorem polar_problem (A B : PolarPoint) 
    (hA : A = ⟨2, π/3⟩) 
    (hB : B = ⟨3, 0⟩) : 
    distancePolar A B = Real.sqrt 7 ∧ 
    triangleAreaPolar A B = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_problem_l526_52630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l526_52622

/-- Function h as defined in the problem -/
noncomputable def h (x : ℝ) : ℝ := 4.125 - (x + 0.5)^2 / 2

/-- The theorem stating the sum of coordinates of the intersection point -/
theorem intersection_point_sum : 
  ∃ (a b : ℝ), h a = h (a + 2) ∧ h a = b ∧ a + b = 4.125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l526_52622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_property_l526_52643

theorem unique_set_property (n : ℕ) (S : Finset ℕ) (h1 : S.Nonempty) (h2 : S.card = n) :
  (∀ (x : ℕ), (S.prod id) ∣ (S.prod (λ a => x + a))) →
  S = Finset.range n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_property_l526_52643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_fixed_point_l526_52620

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point P on circle O
def point_P (x y : ℝ) : Prop := circle_O x y

-- Define the relationship between P, D, and Q
def point_Q (xp yp xq yq : ℝ) : Prop := xp = xq / 2 ∧ yp = yq

-- Define curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 1)

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x - 3 / 5

-- Define the orthogonality condition for AM and AN
def orthogonal_condition (xm ym xn yn : ℝ) : Prop :=
  (xm - 0) * (xn - 0) + (ym - 1) * (yn - 1) = 0

theorem curve_C_and_fixed_point :
  ∀ (xp yp xq yq k xm ym xn yn : ℝ),
    point_P xp yp →
    point_Q xp yp xq yq →
    curve_C xq yq →
    line_l k xm ym →
    line_l k xn yn →
    curve_C xm ym →
    curve_C xn yn →
    orthogonal_condition xm ym xn yn →
    (∀ x y, curve_C x y ↔ x^2 / 4 + y^2 = 1) ∧
    (∃ x y, x = 0 ∧ y = -3/5 ∧ line_l k x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_fixed_point_l526_52620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_assignment_count_l526_52699

/-- Represents the set of volunteers -/
inductive Volunteer : Type
| A : Volunteer
| B : Volunteer
| Other1 : Volunteer
| Other2 : Volunteer
| Other3 : Volunteer

/-- Represents the set of projects -/
inductive Project : Type
| A : Project
| B : Project
| C : Project

/-- A valid assignment of volunteers to projects -/
def ValidAssignment : Type := Project → Volunteer

/-- Predicate to check if an assignment is valid -/
def isValidAssignment (assignment : ValidAssignment) : Prop :=
  (assignment Project.A ≠ Volunteer.A) ∧
  (assignment Project.B ≠ Volunteer.A) ∧
  (assignment Project.B ≠ Volunteer.B) ∧
  (assignment Project.C ≠ Volunteer.B) ∧
  (∀ p1 p2 : Project, p1 ≠ p2 → assignment p1 ≠ assignment p2)

/-- The set of all valid assignments -/
def AllValidAssignments : Set ValidAssignment :=
  {assignment | isValidAssignment assignment}

/-- Instance to make AllValidAssignments a finite type -/
instance : Fintype AllValidAssignments :=
  sorry

theorem valid_assignment_count :
  Fintype.card AllValidAssignments = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_assignment_count_l526_52699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52624

/-- Definition of the function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if a = 0 then
    2 - (1/2) * Real.log (abs x)
  else if x > -a then
    x^2 + a - (1/2) * Real.log x
  else if 0 < x ∧ x < a then
    -x^2 - x - (1/2) * Real.log x
  else
    0  -- undefined for other cases

/-- Definition of an extreme point -/
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, y ≠ x → abs (y - x) < ε → f y ≤ f x ∨ f y ≥ f x

/-- Theorem stating the properties of f(x) for different ranges of a -/
theorem f_properties (a : ℝ) :
  (a = 0 → ∀ x > 0, f a x = 2 - (1/2) * Real.log (abs x)) ∧
  (a < -2 → ∃ x₁ x₂, x₁ = (-a - Real.sqrt (a^2 - 4)) / 4 ∧ 
                     x₂ = (-a + Real.sqrt (a^2 - 4)) / 4 ∧ 
                     is_extreme_point (f a) x₁ ∧ 
                     is_extreme_point (f a) x₂) ∧
  (-2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2 → ∀ x, ¬ is_extreme_point (f a) x) ∧
  (-Real.sqrt 2 / 2 < a ∧ a < 0 → 
    ∃ x, x = (-a + Real.sqrt (a^2 + 4)) / 4 ∧ is_extreme_point (f a) x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_lambda_equation_tangent_line_equation_fixed_tangent_circle_l526_52603

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the relation between points P and M
def point_relation (lambda : ℝ) (xP yP xM yM : ℝ) : Prop :=
  lambda > 1 ∧ xM = 2*lambda*xP ∧ yM = lambda*yP

-- Theorem 1: Equation of curve C_λ
theorem curve_C_lambda_equation (lambda : ℝ) (x y : ℝ) :
  (∃ xP yP : ℝ, ellipse_C xP yP ∧ point_relation lambda xP yP x y) →
  x^2/(16*lambda^2) + y^2/lambda^2 = 1 :=
sorry

-- Theorem 2: Equation of tangent line MA
theorem tangent_line_equation (x1 y1 x y : ℝ) :
  ellipse_C x1 y1 →
  (x1*x)/4 + y1*y = 1 :=
sorry

-- Theorem 3: Existence of fixed tangent circle
theorem fixed_tangent_circle (lambda : ℝ) :
  ∃ r : ℝ, ∀ x y : ℝ,
    (∃ xP yP : ℝ, ellipse_C xP yP ∧ point_relation lambda xP yP x y) →
    (∃ a b : ℝ, (a*x + b*y = 1) ∧ (a^2 + b^2 = lambda^2) ∧ (x^2 + y^2 = r^2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_lambda_equation_tangent_line_equation_fixed_tangent_circle_l526_52603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l526_52645

def mySequence (n : ℕ) : ℕ := 
  match n with
  | 0 => 2
  | n + 1 => mySequence n + 2^n

theorem mySequence_formula (n : ℕ) : mySequence n = 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l526_52645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l526_52636

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 8 = 0

-- Define the distance from a point to the axis of symmetry
def d_1 (x y : ℝ) : ℝ := |x|

-- Define the distance from a point to the line
noncomputable def d_2 (x y : ℝ) : ℝ := |4*x + 3*y + 8| / Real.sqrt (4^2 + 3^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (min_val : ℝ), min_val = 16/5 ∧
  ∀ (x y : ℝ), parabola x y → d_1 x y + d_2 x y ≥ min_val :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l526_52636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_less_than_half_x_l526_52653

theorem negation_of_exists_sin_less_than_half_x :
  (¬ ∃ x : ℝ, Real.sin x < (1/2) * x) ↔ (∀ x : ℝ, Real.sin x ≥ (1/2) * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_less_than_half_x_l526_52653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_focus_distance_l526_52665

noncomputable section

/-- The parabola is defined by x² = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point P with coordinates (a, a²/4) -/
def point_P (a : ℝ) : ℝ × ℝ := (a, a^2/4)

/-- The tangent line at point P(a, a²/4) -/
def tangent_line (a x y : ℝ) : Prop := y - a^2/4 = a/2 * (x - a)

/-- The area of the triangle formed by the tangent line and coordinate axes -/
noncomputable def triangle_area (a : ℝ) : ℝ := (1/2) * (a/2) * (a^2/4)

/-- The focus of the parabola x² = 4y -/
def focus : ℝ × ℝ := (0, 1)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_tangent_focus_distance (a : ℝ) :
  a > 0 →
  parabola a (a^2/4) →
  triangle_area a = 1/2 →
  distance (point_P a) focus = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_focus_distance_l526_52665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_four_months_l526_52612

/-- Calculates the height of a tree after a given number of months -/
noncomputable def tree_height (initial_height : ℝ) (growth_rate : ℝ) (growth_period : ℝ) (months : ℝ) : ℝ :=
  initial_height * 100 + (months * 4 / growth_period) * growth_rate

/-- Theorem stating the height of the tree after 4 months -/
theorem tree_height_after_four_months :
  tree_height 2 50 2 4 = 600 := by
  -- Unfold the definition of tree_height
  unfold tree_height
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_four_months_l526_52612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52675

noncomputable def f (x : ℝ) := 2 * Real.sin x ^ 2 - Real.sin (2 * x - 5 * Real.pi / 6)

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = 2) ∧
  (∀ (x : ℝ), f x = 2 ↔ ∃ (k : ℤ), x = k * Real.pi + Real.pi / 3) ∧
  (∀ (θ : ℝ), 0 < θ ∧ θ < Real.pi / 2 → Real.tan θ = 2 * Real.sqrt 2 →
    f θ = 25 / 18 + 2 * Real.sqrt 6 / 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l526_52660

/-- Given a triangle ABC with two sides of lengths 12 and 15, and angles satisfying
    cos(2A) + cos(2B) + cos(2C) = 1, the maximum length of the third side is √369. -/
theorem max_third_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 12 →
  b = 15 →
  Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C) = 1 →
  c ≤ Real.sqrt 369 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l526_52660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l526_52637

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- The foci of a hyperbola -/
noncomputable def Hyperbola.foci (h : Hyperbola a b) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (-c, 0, c, 0)

/-- The asymptote of a hyperbola -/
noncomputable def Hyperbola.asymptote (h : Hyperbola a b) (x : ℝ) : ℝ :=
  b / a * x

/-- The eccentricity of a hyperbola -/
noncomputable def Hyperbola.eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- The center of a hyperbola -/
def Hyperbola.center : ℝ × ℝ := (0, 0)

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The symmetric point about the asymptote -/
noncomputable def symmetricPoint (h : Hyperbola a b) : ℝ × ℝ := 
  let (_, _, x2, _) := h.foci
  let y := h.asymptote x2
  (x2, 2 * y)

theorem hyperbola_eccentricity_is_two (a b : ℝ) (h : Hyperbola a b) :
  let (x1, y1, x2, y2) := h.foci
  let (sx, sy) := symmetricPoint h
  distance sx sy x1 y1 = distance 0 0 x1 y1 →
  h.eccentricity = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l526_52637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_one_wins_l526_52609

/-- Represents a player in the game -/
inductive Player
| One
| Two

/-- Represents the game state -/
structure GameState where
  moves : List Real
  currentPlayer : Player

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Real) : Prop :=
  0 ≤ move ∧ move ≤ 10 ∧
  ∀ m ∈ state.moves, |move - m| ≥ 1.5

/-- Defines the game's rules and winning condition -/
def gameRules (initialState : GameState) (strategy : GameState → Real) : Prop :=
  ∀ state : GameState,
    (state = initialState ∨ state.moves ≠ []) →
    (∃ move, isValidMove state move) →
    isValidMove state (strategy state)

/-- Theorem stating that Player One has a winning strategy -/
theorem player_one_wins :
  ∃ (strategy : GameState → Real),
    gameRules { moves := [], currentPlayer := Player.One } strategy ∧
    ∀ (opponent_strategy : GameState → Real),
      ∃ (n : Nat),
        let finalState := (fun state =>
          { moves := (strategy state) :: state.moves,
            currentPlayer := match state.currentPlayer with
                             | Player.One => Player.Two
                             | Player.Two => Player.One })^[n]
          { moves := [], currentPlayer := Player.One }
        ¬∃ move, isValidMove finalState move ∧
        finalState.currentPlayer = Player.Two :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_one_wins_l526_52609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_toy_cost_l526_52681

noncomputable def toy_cost : ℚ := 12
def num_toys : ℕ := 4

noncomputable def calculate_pair_cost (regular_price : ℚ) : ℚ :=
  regular_price + (regular_price / 2)

noncomputable def calculate_total_cost (regular_price : ℚ) (num_pairs : ℕ) : ℚ :=
  (calculate_pair_cost regular_price) * num_pairs

theorem dog_toy_cost :
  calculate_total_cost toy_cost (num_toys / 2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_toy_cost_l526_52681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_8_meters_l526_52670

/-- The length of a boat given its breadth, sinking depth, and the mass of a man -/
noncomputable def boat_length (breadth : ℝ) (sinking_depth : ℝ) (man_mass : ℝ) : ℝ :=
  let water_density : ℝ := 1000
  let gravity : ℝ := 9.81
  let volume_displaced : ℝ := man_mass * gravity / (water_density * gravity)
  volume_displaced / (breadth * sinking_depth)

/-- Theorem stating that the length of the boat is 8 meters under given conditions -/
theorem boat_length_is_8_meters :
  boat_length 3 0.01 240 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_8_meters_l526_52670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52674

/-- The function f(x) defined as cos²x + sin x cos x - 1/2 --/
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x * Real.cos x - 1/2

/-- The maximum value of f(x) --/
noncomputable def f_max : ℝ := Real.sqrt 2 / 2

/-- The period of f(x) --/
noncomputable def f_period : ℝ := Real.pi

/-- Properties of the function f(x) --/
theorem f_properties :
  (∀ x, f x ≤ f_max) ∧
  (∀ x, f (x + f_period) = f x) ∧
  (f (-Real.pi/8) = 0) ∧
  (∀ x, f x = f_max * Real.sin (2*x + Real.pi/4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l526_52674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l526_52686

def productSequence : List ℕ := List.range 10 |> List.map (λ n => 10 * n + 3)

theorem product_remainder : 
  productSequence.prod % 5 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l526_52686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_percentage_l526_52692

/-- Represents the trader's buying and selling practices --/
structure TraderPractices where
  buy_excess : ℝ  -- Percentage excess taken when buying
  sell_deficit : ℝ  -- Percentage deficit given when selling
  claimed_weight : ℝ  -- Weight claimed to be sold

/-- Calculates the profit percentage for the trader --/
noncomputable def calculate_profit_percentage (practices : TraderPractices) : ℝ :=
  let actual_bought := practices.claimed_weight * (1 + practices.buy_excess)
  let actual_sold := practices.claimed_weight / (1 + practices.sell_deficit)
  let profit := practices.claimed_weight - actual_sold
  (profit / practices.claimed_weight) * 100

/-- Theorem stating the trader's profit percentage --/
theorem trader_profit_percentage :
  ∀ (practices : TraderPractices),
    practices.buy_excess = 0.1 ∧
    practices.sell_deficit = 0.3 ∧
    practices.claimed_weight > 0 →
    |calculate_profit_percentage practices - 23.08| < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_percentage_l526_52692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_five_equation_l526_52673

theorem power_of_five_equation (n : ℕ) : 112 * (5 : ℕ)^n = 70000 ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_five_equation_l526_52673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_equation_l526_52655

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 10 = 0

-- Define point M
def M : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem longest_chord_equation :
  ∃ (C : ℝ × ℝ),
    (∀ x y, circle_eq x y ↔ (x - C.1)^2 + (y - C.2)^2 = (C.1 - M.1)^2 + (C.2 - M.2)^2) ∧
    (∀ x y, line_equation x y ↔ (y - M.2) = (C.2 - M.2) / (C.1 - M.1) * (x - M.1)) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_equation_l526_52655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_two_formula_g_recursive_g_seven_three_l526_52684

/-- g(n, k) represents the number of ways to partition an n-element set into k subsets each containing at least 2 elements -/
def g : ℕ → ℕ → ℕ := sorry

/-- Formula for g(n, 2) when n ≥ 4 -/
theorem g_two_formula (n : ℕ) (h : n ≥ 4) : g n 2 = 2^(n-2) - n - 1 := by sorry

/-- Recursive relation for g(n+1, k) -/
theorem g_recursive (n k : ℕ) (h : n ≥ 2*k ∧ k ≥ 3) : 
  g (n+1) k = n * g (n-1) (k-1) + k * g n k := by sorry

/-- Specific value of g(7, 3) -/
theorem g_seven_three : g 7 3 = 105 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_two_formula_g_recursive_g_seven_three_l526_52684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_sum_difference_l526_52663

theorem odd_even_sum_difference : 
  let n : ℕ := 1012  -- number of odd terms
  let m : ℕ := 1010  -- number of even terms
  let sum_odd : ℕ := n^2  -- sum of odd numbers
  let sum_even : ℕ := m * (m + 1)  -- sum of even numbers
  (sum_odd : ℤ) - (sum_even^2 : ℤ) = -104271921956 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_sum_difference_l526_52663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_factors_144_mult_6_eq_7_l526_52601

/-- The number of positive factors of 144 that are also multiples of 6 -/
def num_factors_144_mult_6 : ℕ :=
  (Finset.filter (λ n ↦ 144 % n = 0 ∧ n % 6 = 0) (Finset.range 145)).card

/-- Theorem stating that the number of positive factors of 144 that are also multiples of 6 is 7 -/
theorem num_factors_144_mult_6_eq_7 : num_factors_144_mult_6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_factors_144_mult_6_eq_7_l526_52601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_volume_equality_l526_52697

/-- The volume of a cylinder given its lateral surface properties -/
theorem cylinder_volume (a α : ℝ) (h : 0 < a) (h_angle : 0 < α ∧ α < Real.pi / 2) :
  ∃ (V : ℝ), V = (a^3 * Real.cos α^2 * Real.sin α) / (4 * Real.pi) :=
sorry

/-- Helper function to represent the volume of a cylinder -/
noncomputable def volume_of_cylinder (a α : ℝ) : ℝ :=
  (a^3 * Real.cos α^2 * Real.sin α) / (4 * Real.pi)

/-- The calculated volume equals the volume of the cylinder -/
theorem volume_equality (a α : ℝ) (h : 0 < a) (h_angle : 0 < α ∧ α < Real.pi / 2) :
  (a^3 * Real.cos α^2 * Real.sin α) / (4 * Real.pi) = volume_of_cylinder a α :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_volume_equality_l526_52697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_law_tetrahedron_edge_center_relation_l526_52617

/-- Parallelogram Law -/
theorem parallelogram_law' (a b d₁ d₂ : ℝ) (h : a > 0 ∧ b > 0) :
  2 * a^2 + 2 * b^2 = d₁^2 + d₂^2 := by sorry

/-- Tetrahedron Edge-Center Relation -/
theorem tetrahedron_edge_center_relation (a b c d e f x y z : ℝ) :
  4 * (x^2 + y^2 + z^2) = a^2 + b^2 + c^2 + d^2 + e^2 + f^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_law_tetrahedron_edge_center_relation_l526_52617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_quantity_is_one_l526_52602

-- Define the prices and quantities
noncomputable def salad_price : ℚ := 3
noncomputable def beef_price : ℚ := 2 * salad_price
noncomputable def potato_price : ℚ := (1 / 3) * salad_price
noncomputable def juice_price : ℚ := 3/2

def salad_quantity : ℚ := 2
def beef_quantity : ℚ := 2
def juice_quantity : ℚ := 2

def total_cost : ℚ := 22

-- Define the theorem
theorem potato_quantity_is_one :
  ∃ (potato_quantity : ℚ),
    salad_quantity * salad_price +
    beef_quantity * beef_price +
    potato_quantity * potato_price +
    juice_quantity * juice_price = total_cost ∧
    potato_quantity = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_quantity_is_one_l526_52602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_finish_time_l526_52638

/-- The number of days A needs to finish the work alone -/
noncomputable def a_days : ℝ := 12

/-- The number of days B needs to finish the work alone -/
noncomputable def b_days : ℝ := 15

/-- The number of days C needs to finish the work alone -/
noncomputable def c_days : ℝ := 18

/-- The number of days B worked before leaving -/
noncomputable def b_worked_days : ℝ := 10

/-- The portion of work B completed before leaving -/
noncomputable def b_completed_work : ℝ := b_worked_days / b_days

/-- The remaining work after B left -/
noncomputable def remaining_work : ℝ := 1 - b_completed_work

/-- The combined work rate of A and C per day -/
noncomputable def ac_combined_rate : ℝ := 1 / a_days + 1 / c_days

/-- The number of days A and C need to finish the remaining work -/
noncomputable def ac_days : ℝ := remaining_work / ac_combined_rate

theorem ac_finish_time : ac_days = 2.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_finish_time_l526_52638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_power_function_l526_52668

noncomputable def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

noncomputable def f (x : ℝ) : ℝ := 1 / x^2
noncomputable def g (x : ℝ) : ℝ := -x^2
noncomputable def h (x : ℝ) : ℝ := x^2 + x

theorem one_power_function : 
  (is_power_function f ∧ ¬is_power_function g ∧ ¬is_power_function h) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_power_function_l526_52668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l526_52672

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.cos x

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2 * x - Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l526_52672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l526_52648

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) - 1

theorem smallest_positive_period 
  (ω : ℝ) 
  (x₁ x₂ x₃ : ℝ) 
  (h₁ : f ω x₁ = 0)
  (h₂ : f ω x₂ = 0)
  (h₃ : f ω x₃ = 0)
  (h₄ : 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 5 * Real.pi / (2 * ω))
  (h₅ : x₁ + 2 * x₂ + x₃ = 7 * Real.pi / 3) :
  ∃ (T : ℝ), T = Real.pi ∧ ∀ (x : ℝ), f ω (x + T) = f ω x ∧ 
  ∀ (T' : ℝ), 0 < T' ∧ T' < T → ∃ (x : ℝ), f ω (x + T') ≠ f ω x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l526_52648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l526_52616

/-- A cylinder with a square cross-section -/
structure SquareCylinder where
  side : ℝ
  height : ℝ

/-- A sphere -/
structure Sphere where
  radius : ℝ

/-- The lateral surface area of a cylinder -/
def lateral_surface_area (c : SquareCylinder) : ℝ :=
  4 * c.side * c.height

/-- The surface area of a sphere -/
noncomputable def surface_area (s : Sphere) : ℝ :=
  4 * Real.pi * s.radius^2

/-- The volume of a cylinder -/
def cylinder_volume (c : SquareCylinder) : ℝ :=
  c.side^2 * c.height

/-- The volume of a sphere -/
noncomputable def sphere_volume (s : Sphere) : ℝ :=
  (4 / 3) * Real.pi * s.radius^3

theorem cylinder_sphere_volume_ratio
  (c : SquareCylinder)
  (s : Sphere)
  (h1 : c.height = c.side)
  (h2 : lateral_surface_area c = surface_area s) :
  cylinder_volume c / sphere_volume s = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l526_52616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_interval_l526_52652

open Real

theorem equation_solution_interval (n : ℤ) :
  (∃ x : ℝ, x > n ∧ x < n + 1 ∧ Real.log x + 2 * x - 6 = 0) →
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_interval_l526_52652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_triangle_inequality_l526_52606

theorem acute_angles_triangle_inequality (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : 0 < β ∧ β < Real.pi / 2) 
  (h3 : α + β < Real.pi) : 
  Real.sin α > Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_triangle_inequality_l526_52606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l526_52667

theorem problem_solution (x : ℝ) : (16 : ℝ)^(x+2) = 112 + (16 : ℝ)^x → x = Real.log (112/255) / Real.log 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l526_52667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l526_52614

/-- Translate a point in ℝ² by a given vector -/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem midpoint_after_translation :
  let s₁_start : ℝ × ℝ := (4, 1)
  let s₁_end : ℝ × ℝ := (-8, 5)
  let translation : ℝ × ℝ := (-3, -2)
  let s₂_start := translate s₁_start translation
  let s₂_end := translate s₁_end translation
  (s₂_start.1 + s₂_end.1) / 2 = -5 ∧ (s₂_start.2 + s₂_end.2) / 2 = 1 := by
  sorry

#check midpoint_after_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l526_52614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_distance_l526_52678

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  S : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Check if three points form an isosceles right triangle -/
def isIsoscelesRightTriangle (A B C : Point3D) : Prop :=
  distance A B = distance A C ∧ distance A B = distance B C ∧
  (distance A B)^2 = (distance A C)^2 + (distance B C)^2

/-- Check if four points lie on a sphere -/
def lieOnSphere (O S A B C : Point3D) : Prop :=
  ∃ r : ℝ, distance O S = r ∧ distance O A = r ∧ distance O B = r ∧ distance O C = r

/-- Distance from a point to a plane defined by three points -/
noncomputable def distanceToPlane (O A B C : Point3D) : ℝ := 
  sorry

theorem tetrahedron_sphere_distance 
  (O : Point3D) (t : Tetrahedron) :
  isIsoscelesRightTriangle t.A t.B t.C →
  distance t.S t.A = 2 ∧ distance t.S t.B = 2 ∧ distance t.S t.C = 2 ∧ distance t.A t.B = 2 →
  lieOnSphere O t.S t.A t.B t.C →
  distanceToPlane O t.A t.B t.C = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_distance_l526_52678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_face_parallelepiped_implies_rhombus_l526_52610

/-- A parallelepiped with equal parallelogram faces -/
structure EqualFaceParallelepiped where
  /-- The lengths of the three edges meeting at a vertex -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- All faces are parallelograms -/
  faces_are_parallelograms : Prop
  /-- All faces are equal -/
  faces_are_equal : Prop

/-- Theorem: If all 6 faces of a parallelepiped are equal parallelograms, then they are rhombuses -/
theorem equal_face_parallelepiped_implies_rhombus (P : EqualFaceParallelepiped) :
  P.a = P.b ∧ P.b = P.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_face_parallelepiped_implies_rhombus_l526_52610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l526_52642

/-- Given vectors a and b in ℝ³, we define c₁ and c₂ and prove they are collinear. -/
theorem vectors_collinear (a b : Fin 3 → ℝ) 
  (ha : a = ![3, 7, 0]) 
  (hb : b = ![1, -3, 4]) : 
  ∃ (k : ℝ), (4 • a - 2 • b) = k • (b - 2 • a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l526_52642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_proof_l526_52628

-- Define the circles and their properties
def circle_A : ℝ × ℝ → Prop := λ p ↦ (p.1 + 2)^2 + p.2^2 = 4
def circle_B : ℝ × ℝ → Prop := λ p ↦ (p.1 - 2)^2 + p.2^2 = 4
def circle_C : ℝ × ℝ → Prop := λ p ↦ p.1^2 + (p.2 - 3)^2 = 1

-- Define the area function
noncomputable def area_inside_C_outside_AB : ℝ := Real.pi

-- Theorem statement
theorem area_proof :
  (∀ p, circle_A p → ¬circle_C p) ∧
  (∀ p, circle_B p → ¬circle_C p) →
  area_inside_C_outside_AB = Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_proof_l526_52628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_circles_l526_52651

/-- Three pairwise externally tangent circles with radii a, b, and c -/
structure TangentCircles (a b c : ℝ) : Prop where
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  externally_tangent : a + b < c ∧ b + c < a ∧ c + a < b

/-- The length of the chord intercepted by the third circle from the common internal tangent of the first two circles -/
noncomputable def chord_length (a b c : ℝ) : ℝ := 4 * c * Real.sqrt (a * b) / (a + b)

theorem chord_length_tangent_circles (a b c : ℝ) (h : TangentCircles a b c) :
  chord_length a b c = 4 * c * Real.sqrt (a * b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_circles_l526_52651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_sum_l526_52664

noncomputable def y (x : ℝ) : ℝ := (x^3 + 10*x^2 + 33*x + 36) / (x + 3)

def A : ℝ := 1
def B : ℝ := 7
def C : ℝ := 12
def D : ℝ := -3

theorem simplification_and_sum :
  (∀ x, x ≠ D → y x = A * x^2 + B * x + C) ∧
  A + B + C + D = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_sum_l526_52664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_point_l526_52666

-- Define the line equation
def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 4 = 0

-- Define the curve using parametric equations
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Theorem statement
theorem one_common_point :
  ∃! p : ℝ × ℝ, ∃ θ : ℝ, curve θ = p ∧ line p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_point_l526_52666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_min_side_a_achievable_l526_52695

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the minimum possible value of side a -/
theorem min_side_a (t : Triangle) 
  (h1 : f (t.A / 2) = 1 / 2) 
  (h2 : t.b + t.c = 2) : 
  t.a ≥ 1 := by sorry

/-- The theorem stating that the minimum value is achievable -/
theorem min_side_a_achievable : 
  ∃ t : Triangle, f (t.A / 2) = 1 / 2 ∧ t.b + t.c = 2 ∧ t.a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_min_side_a_achievable_l526_52695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_count_l526_52641

/-- A point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2

/-- Check if a triangle is right-angled -/
def isRightTriangle (a b c : Point) : Prop :=
  let ab := distance a b
  let bc := distance b c
  let ca := distance c a
  ab^2 + bc^2 = ca^2 ∨ bc^2 + ca^2 = ab^2 ∨ ca^2 + ab^2 = bc^2

theorem right_triangle_count :
  ∃ (a b : Point),
    distance a b = 10 ∧
    (∃ (s : Finset Point),
      s.card = 8 ∧
      (∀ c ∈ s, isRightTriangle a b c ∧ triangleArea a b c = 20) ∧
      (∀ c : Point, isRightTriangle a b c ∧ triangleArea a b c = 20 → c ∈ s)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_count_l526_52641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_modulus_on_circle_max_modulus_attained_l526_52677

theorem max_modulus_on_circle (z : ℂ) : 
  Complex.abs (z + 5 - 12 * Complex.I) = 3 → Complex.abs z ≤ 16 := by
  sorry

theorem max_modulus_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ z : ℂ, Complex.abs (z + 5 - 12 * Complex.I) = 3 ∧ Complex.abs z > 16 - ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_modulus_on_circle_max_modulus_attained_l526_52677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_m_zero_range_of_m_l526_52623

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := |x + 4/x - m| + m

-- Theorem for part I
theorem min_value_when_m_zero :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ x : ℝ, x ≠ 0 → |x + 4/x| ≥ min_val := by
  sorry

-- Theorem for part II
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 4 → f x m ≤ 5) → m ≤ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_m_zero_range_of_m_l526_52623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_identical_digits_time_l526_52649

/-- Represents a time on a 24-hour digital clock -/
structure DigitalTime where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Converts a DigitalTime to minutes since midnight -/
def timeToMinutes (t : DigitalTime) : Nat :=
  t.hours * 60 + t.minutes

/-- Checks if all digits in a time are identical -/
def allDigitsIdentical (t : DigitalTime) : Bool :=
  let h1 := t.hours / 10
  let h2 := t.hours % 10
  let m1 := t.minutes / 10
  let m2 := t.minutes % 10
  h1 = h2 ∧ h2 = m1 ∧ m1 = m2

/-- The theorem to be proved -/
theorem next_identical_digits_time (start : DigitalTime) 
  (h : start.hours = 5 ∧ start.minutes = 55) :
  ∃ (endTime : DigitalTime), 
    allDigitsIdentical endTime ∧ 
    timeToMinutes endTime - timeToMinutes start = 316 ∧
    (∀ (t : DigitalTime), 
      timeToMinutes start < timeToMinutes t ∧ 
      timeToMinutes t < timeToMinutes endTime → 
      ¬allDigitsIdentical t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_identical_digits_time_l526_52649
