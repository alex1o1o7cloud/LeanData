import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l1049_104916

-- Define the angle α
variable (α : ℝ)

-- Define the point on the terminal side of the angle
def point : ℝ × ℝ := (-4, 3)

-- Define the cosine of the angle
noncomputable def cosine_of_angle : ℝ := -4/5

-- Theorem statement
theorem cosine_of_angle_through_point :
  (point.1 = -4 ∧ point.2 = 3) → cosine_of_angle = Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l1049_104916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_is_75pi_l1049_104999

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 = 6*y - 16*x + 8

-- Define the area of the region
noncomputable def region_area : ℝ := 75 * Real.pi

-- Theorem statement
theorem region_area_is_75pi :
  ∃ (center_x center_y radius : ℝ),
    (∀ (x y : ℝ), region_equation x y → (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_is_75pi_l1049_104999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_DB_length_l1049_104919

-- Define the triangle ABC and point D
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

variable (t : Triangle)
variable (D : ℝ × ℝ)

-- Define the conditions
axiom right_angle_ABC : t.B.1 = t.A.1 ∧ t.B.2 = t.C.2
axiom right_angle_ADB : D.1 = t.A.1 ∧ D.2 = t.B.2
axiom AC_length : Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) = 20
axiom AD_length : Real.sqrt ((t.A.1 - D.1)^2 + (t.A.2 - D.2)^2) = 4

-- State the theorem
theorem DB_length : 
  Real.sqrt ((D.1 - t.B.1)^2 + (D.2 - t.B.2)^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_DB_length_l1049_104919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_expression_l1049_104974

open InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem max_value_of_vector_expression (a b c : E) 
  (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hc : ‖c‖ = 4) :
  (∀ x y z : E, ‖x‖ = 2 → ‖y‖ = 3 → ‖z‖ = 4 → 
    ‖x - 3 • y‖^2 + ‖y - 3 • z‖^2 + ‖z - 3 • x‖^2 ≤ ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2) ∧
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2 = 377 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_expression_l1049_104974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cos_tan_sum_l1049_104977

theorem cos_cos_tan_sum : Real.cos (25 * Real.pi / 6) + Real.cos (25 * Real.pi / 3) + Real.tan (-(25 * Real.pi / 4)) = Real.sqrt 3 / 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cos_tan_sum_l1049_104977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l1049_104938

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem ellipse_chord_theorem (e : Ellipse) (A B F : Point) :
  e.a = 5 ∧ e.b = 3 ∧
  isOnEllipse A e ∧ isOnEllipse B e ∧
  F.x = 4 ∧ F.y = 0 ∧
  distance A F = 3/2 →
  distance B F = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l1049_104938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_thirds_l1049_104996

/-- A square with sides divided into segments -/
structure SegmentedSquare where
  s : ℝ  -- side length of the square
  n : ℕ  -- number of segments on two opposite sides
  m : ℕ  -- number of segments on the other two opposite sides

/-- Triangle formed by joining a segment endpoint to the square's center -/
structure CenterTriangle (sq : SegmentedSquare) where
  is_n_side : Bool  -- true if the triangle is from an n-divided side, false if from an m-divided side

/-- The area of a CenterTriangle -/
noncomputable def triangle_area (sq : SegmentedSquare) (t : CenterTriangle sq) : ℝ :=
  if t.is_n_side then
    (sq.s^2) / (8 * sq.n)
  else
    (sq.s^2) / (8 * sq.m)

theorem area_ratio_is_four_thirds (sq : SegmentedSquare) 
    (h_n : sq.n = 6) (h_m : sq.m = 8) 
    (t_A : CenterTriangle sq) (h_A : t_A.is_n_side = true)
    (t_B : CenterTriangle sq) (h_B : t_B.is_n_side = false) :
  triangle_area sq t_A / triangle_area sq t_B = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_thirds_l1049_104996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_fixed_point_and_min_ratio_l1049_104939

/-- Parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point P not on x-axis -/
structure PointP where
  a : ℝ
  b : ℝ
  not_on_x_axis : b ≠ 0

/-- Line segment l_P -/
def line_lP (P : PointP) (x y : ℝ) : Prop :=
  P.b * y = 2 * (x + P.a)

/-- Point R where l_P intersects x-axis -/
def point_R : ℝ × ℝ := (2, 0)

/-- Ratio |PQ|/|QR| -/
noncomputable def ratio_PQ_QR (P : PointP) : ℝ :=
  (8 + P.b^2) / (2 * abs P.b)

theorem parabola_tangent_fixed_point_and_min_ratio
  (P : PointP)
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), parabola x1 y1 ∧ parabola x2 y2 ∧ line_lP P x1 y1 ∧ line_lP P x2 y2)
  (h2 : ∀ (x y : ℝ), line_lP P x y → (x - P.a) * P.b + (y - P.b) * 2 = 0) :
  (point_R = (2, 0)) ∧
  (∀ (P' : PointP), ratio_PQ_QR P' ≥ 2 * Real.sqrt 2) ∧
  (∃ (P' : PointP), ratio_PQ_QR P' = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_fixed_point_and_min_ratio_l1049_104939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1049_104989

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - Real.log (x + 1)
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x

-- State the theorem
theorem min_a_value (a : ℝ) :
  (∃ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2,
    (deriv f) x₁ ≥ g a x₂) →
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l1049_104989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_cube_sum_l1049_104937

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem age_cube_sum (d t h : ℕ) 
  (h1 : 4 * d + 2 * t = 3 * h)
  (h2 : 3 * h^2 = 2 * d^2 + 4 * t^2)
  (h3 : d > 0 ∧ t > 0 ∧ h > 0)
  (h4 : is_coprime d t ∧ is_coprime d h ∧ is_coprime t h) :
  d^3 + t^3 + h^3 = 349 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_cube_sum_l1049_104937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divides_n2n_plus_1_solutions_for_p_equals_3_l1049_104964

theorem odd_prime_divides_n2n_plus_1 (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ f : ℕ → ℕ, StrictMono f ∧
    ∀ k : ℕ, p ∣ (f k * 2^(f k) + 1) :=
sorry

theorem solutions_for_p_equals_3 :
  ∀ n : ℕ, (3 ∣ (n * 2^n + 1)) ↔ (∃ k : ℕ, n = 6*k + 1 ∨ n = 6*k + 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_divides_n2n_plus_1_solutions_for_p_equals_3_l1049_104964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_focus_l1049_104923

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ b ≤ a

/-- Represents a point on an ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The distance from the center to a focus of the ellipse -/
noncomputable def focal_distance (E : Ellipse) : ℝ :=
  Real.sqrt (E.a^2 - E.b^2)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  focal_distance E / E.a

/-- Theorem: For an ellipse with equation x^2/100 + y^2/36 = 1 and a point P on the ellipse
    at a distance of 10 from the left directrix, the distance from point P to the right focus
    of the ellipse is 12 units. -/
theorem distance_to_right_focus
  (E : Ellipse)
  (h_eq : E.a = 10 ∧ E.b = 6)
  (P : PointOnEllipse E)
  (h_dist : (eccentricity E) * 10 = 8) :
  2 * E.a - (eccentricity E) * 10 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_focus_l1049_104923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_l1049_104975

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the four functions
def func1 (f : ℝ → ℝ) (x : ℝ) : ℝ := -(abs (f x))
def func2 (f : ℝ → ℝ) (x : ℝ) : ℝ := abs x * f (x^2)
def func3 (f : ℝ → ℝ) (x : ℝ) : ℝ := -(f (-x))
def func4 (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + f (-x)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem stating which functions are even
theorem even_functions (f : ℝ → ℝ) : 
  ¬(is_even (func1 f)) ∧ 
  (is_even (func2 f)) ∧ 
  ¬(is_even (func3 f)) ∧ 
  (is_even (func4 f)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_l1049_104975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1049_104927

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ := (floor (2 * x) : ℝ) - 2 * x

theorem f_range : Set.range f = Set.Ioc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1049_104927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1049_104956

/-- The solution set of the inequality ax^2 + 2x - 1 > 0 -/
def solutionSet (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x > 1/2}
  else if a ≤ -1 then
    ∅
  else if a > 0 then
    {x | x > (-1 + Real.sqrt (1+a))/a ∨ x < (-1 - Real.sqrt (1+a))/a}
  else
    {x | (-1 + Real.sqrt (1+a))/a < x ∧ x < (-1 - Real.sqrt (1+a))/a}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | a * x^2 + 2 * x - 1 > 0} = solutionSet a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1049_104956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l1049_104988

noncomputable def z : ℂ := sorry

noncomputable def arg_z_plus_2 : ℝ := Real.pi / 3
noncomputable def arg_z_minus_2 : ℝ := 5 * Real.pi / 6

axiom h1 : Complex.arg (z + 2) = arg_z_plus_2
axiom h2 : Complex.arg (z - 2) = arg_z_minus_2

theorem z_value : z = -1 + Complex.I * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l1049_104988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_dimension_value_l1049_104958

/-- Represents a rectangle that can be cut into two congruent pentagons -/
structure CuttableRectangle where
  length : ℝ
  width : ℝ
  can_be_cut_into_pentagons : Prop

/-- Represents a square formed by repositioning two congruent pentagons -/
structure PentagonSquare where
  side : ℝ
  formed_from_pentagons : Prop

/-- The dimension y in the pentagon -/
noncomputable def pentagon_dimension (r : CuttableRectangle) : ℝ :=
  r.width / 3

theorem pentagon_dimension_value (r : CuttableRectangle) (s : PentagonSquare) :
  r.length = 20 ∧ r.width = 10 ∧ r.can_be_cut_into_pentagons ∧
  s.formed_from_pentagons ∧ s.side^2 = r.length * r.width →
  pentagon_dimension r = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_dimension_value_l1049_104958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_symmetric_sine_l1049_104942

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem min_value_of_shifted_symmetric_sine 
  (φ : ℝ) 
  (h1 : |φ| < π/2) 
  (h2 : ∀ x, f (x + π/6) φ = -f (-x - π/6) φ) -- symmetry about origin after shift
  : ∃ x ∈ Set.Icc 0 (π/2), ∀ y ∈ Set.Icc 0 (π/2), f x φ ≤ f y φ ∧ f x φ = -Real.sqrt 3 / 2 := by
  sorry

#check min_value_of_shifted_symmetric_sine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_symmetric_sine_l1049_104942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_one_zero_quadratic_function_at_most_two_zeros_exponential_function_no_zeros_logarithmic_function_one_zero_power_function_may_have_zero_l1049_104995

-- Define the types of functions
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b ∧ a ≠ 0

def QuadraticFunction (f : ℝ → ℝ) : Prop := ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c ∧ a ≠ 0

noncomputable def ExponentialFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * Real.exp (b * x) ∧ a ≠ 0 ∧ b ≠ 0

noncomputable def LogarithmicFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * Real.log x + b ∧ a ≠ 0

def PowerFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x^b ∧ a ≠ 0

-- Define what it means for a function to have a zero point
def HasZeroPoint (f : ℝ → ℝ) : Prop := ∃ x : ℝ, f x = 0

-- Theorem statements
theorem linear_function_one_zero (f : ℝ → ℝ) (hf : LinearFunction f) : 
  ∃! x : ℝ, f x = 0 :=
sorry

theorem quadratic_function_at_most_two_zeros (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∃ x y : ℝ, (∀ z : ℝ, f z = 0 → z = x ∨ z = y) :=
sorry

theorem exponential_function_no_zeros (f : ℝ → ℝ) (hf : ExponentialFunction f) :
  ¬ HasZeroPoint f :=
sorry

theorem logarithmic_function_one_zero (f : ℝ → ℝ) (hf : LogarithmicFunction f) :
  ∃! x : ℝ, f x = 0 :=
sorry

theorem power_function_may_have_zero (f : ℝ → ℝ) (hf : PowerFunction f) :
  (HasZeroPoint f ∨ ¬HasZeroPoint f) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_one_zero_quadratic_function_at_most_two_zeros_exponential_function_no_zeros_logarithmic_function_one_zero_power_function_may_have_zero_l1049_104995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1049_104901

theorem vector_equation_solution :
  ∃ (u v : ℝ), 
    (⟨3, 1⟩ : ℝ × ℝ) + u • (⟨8, -6⟩ : ℝ × ℝ) = (⟨2, -2⟩ : ℝ × ℝ) + v • (⟨-3, 4⟩ : ℝ × ℝ) ∧ 
    u = -13/14 ∧ 
    v = 15/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1049_104901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l1049_104970

/-- The line 3x + 4y + 5 = 0 -/
def line (x y : ℝ) : Prop := 3*x + 4*y + 5 = 0

/-- The circle (x-2)² + (y-2)² = 4 -/
def circle_eq (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 4

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_line_circle :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    line x₁ y₁ → circle_eq x₂ y₂ → 
    distance x₁ y₁ x₂ y₂ ≥ 9/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l1049_104970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_is_strong_chi_square_exceeds_critical_value_l1049_104926

-- Define the data types
structure SalesData :=
  (year : ℕ)
  (sales : ℝ)

structure ContingencyTable :=
  (male_traditional : ℕ)
  (male_new_energy : ℕ)
  (female_traditional : ℕ)
  (female_new_energy : ℕ)

-- Define the given data and conditions
def sales_data : List SalesData := [
  ⟨2016, 1.00⟩, ⟨2017, 1.40⟩, ⟨2018, 1.70⟩, ⟨2019, 1.90⟩, ⟨2020, 2.00⟩
]

def partial_contingency_data : ContingencyTable :=
  ⟨0, 12, 4, 0⟩

def total_male : ℕ := 48
def total_sample : ℕ := 60

-- Define the correlation coefficient function
noncomputable def correlation_coefficient (data : List SalesData) : ℝ := 
  sorry

-- Define the chi_square function
noncomputable def chi_square (table : ContingencyTable) : ℝ := 
  sorry

-- Define the critical value for 99% certainty
def chi_square_critical_value : ℝ := 6.635

-- Theorem statements
theorem correlation_is_strong :
  correlation_coefficient sales_data > 0.75 := by
  sorry

theorem chi_square_exceeds_critical_value :
  chi_square (ContingencyTable.mk 36 12 4 8) > chi_square_critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_is_strong_chi_square_exceeds_critical_value_l1049_104926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1049_104936

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log (sin x) * log (cos x)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x + π/4)

-- Theorem statement
theorem f_properties :
  (∀ k : ℤ, ∀ x : ℝ, f x ≠ 0 → x ∈ Set.Ioo (2 * ↑k * π) (2 * ↑k * π + π/2)) ∧
  (∀ x : ℝ, g (-x) = g x) ∧
  (∃! x : ℝ, x ∈ Set.Ioo 0 (π/2) ∧ ∀ y ∈ Set.Ioo 0 (π/2), f y ≤ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1049_104936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_D_l1049_104903

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.A < Real.pi / 2 ∧ t.B < Real.pi / 2 ∧ t.C < Real.pi / 2

-- Statement A
theorem statement_A (t : Triangle) :
  t.a / Real.cos t.A = t.b / Real.sin t.B → t.A = Real.pi / 4 := by
  sorry

-- Statement D
theorem statement_D (t : Triangle) :
  is_acute_triangle t → Real.sin t.A + Real.sin t.B > Real.cos t.A + Real.cos t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_D_l1049_104903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l1049_104907

/-- The time taken for a diver to reach a certain depth -/
noncomputable def time_to_reach (depth : ℝ) (descent_rate : ℝ) : ℝ :=
  depth / descent_rate

/-- Theorem: The time taken for a diver to reach a depth of 4000 feet,
    descending at a rate of 80 feet per minute, is 50 minutes -/
theorem diver_descent_time :
  time_to_reach 4000 80 = 50 := by
  -- Unfold the definition of time_to_reach
  unfold time_to_reach
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l1049_104907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_cover_probability_l1049_104952

/-- The side length of the outer square -/
noncomputable def outerSquareSide : ℝ := 8

/-- The leg length of each isosceles right triangle -/
noncomputable def triangleLeg : ℝ := 2

/-- The side length of the center black square -/
noncomputable def centerSquareSide : ℝ := 2 * Real.sqrt 2

/-- The diameter of the circular coin -/
noncomputable def coinDiameter : ℝ := 1

/-- The probability of the coin covering black regions -/
noncomputable def blackCoverProbability : ℝ := (44 + 24 * Real.sqrt 2 + Real.pi) / 196

theorem coin_cover_probability :
  let totalArea := outerSquareSide ^ 2
  let blackTrianglesArea := 4 * (triangleLeg ^ 2 / 2)
  let centerBlackSquareArea := centerSquareSide ^ 2
  let totalBlackArea := blackTrianglesArea + centerBlackSquareArea
  let coinArea := Real.pi * (coinDiameter / 2) ^ 2
  (totalBlackArea + coinArea) / totalArea = blackCoverProbability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_cover_probability_l1049_104952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_child_age_l1049_104963

/-- A set of six natural numbers representing the ages of children in a family. -/
def ChildrenAges : Set (Set ℕ) :=
  {ages | ∃ (x : ℕ), ages = {x, x + 2, x + 6, x + 8, x + 12, x + 14}}

/-- Predicate to check if a natural number is prime. -/
def IsPrime (n : ℕ) : Prop := Nat.Prime n

/-- The theorem stating that the youngest child's age is 5. -/
theorem youngest_child_age :
  ∃ (ages : Set ℕ), ages ∈ ChildrenAges ∧
    (∀ n ∈ ages, IsPrime n) ∧
    (∃ (youngest : ℕ), youngest ∈ ages ∧
      ∀ m ∈ ages, m ≥ youngest) ∧
    (5 ∈ ages) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_child_age_l1049_104963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_area_ratio_l1049_104934

/-- Represents a rectangle with two non-parallel cuts -/
structure CutRectangle where
  length : ℝ
  width : ℝ
  longCutRatio : ℝ
  shortCutRatio : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  long_cut_valid : longCutRatio = 1/3
  short_cut_valid : shortCutRatio = 1/5

/-- The area of the quadrilateral formed by the cuts at one corner -/
noncomputable def cornerArea (rect : CutRectangle) : ℝ :=
  (rect.length * rect.width * rect.longCutRatio / 2) +
  (rect.length * rect.width * rect.shortCutRatio / 2)

/-- The theorem stating that the corner area is 2/15 of the total area -/
theorem corner_area_ratio (rect : CutRectangle) :
  cornerArea rect = (2/15) * (rect.length * rect.width) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_area_ratio_l1049_104934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_pi_l1049_104910

noncomputable section

open Real

-- Define the function f(x) = sin x
noncomputable def f (x : ℝ) := sin x

-- Define the point of tangency
noncomputable def point : ℝ × ℝ := (π, 0)

-- State the theorem
theorem tangent_line_at_pi :
  ∃ (m : ℝ), HasDerivAt f m point.fst ∧
  ∀ (x y : ℝ), (x + y - π = 0) ↔ y - point.snd = m * (x - point.fst) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_pi_l1049_104910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_list_price_equal_commissions_l1049_104972

/-- The list price of the item -/
def x : ℝ := 30

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 10

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 20

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := 0.1 * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := 0.2 * bob_price x

/-- Theorem: The list price that results in equal commissions is $30 -/
theorem list_price_equal_commissions : 
  alice_commission x = bob_commission x → x = 30 :=
by
  intro h
  -- The proof goes here
  sorry

#eval x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_list_price_equal_commissions_l1049_104972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biggest_intersection_size_l1049_104911

/-- A subset of {1,2,...,1000} with 500 elements -/
def Subset500 := {A : Finset Nat // A ⊆ Finset.range 1000 ∧ A.card = 500}

/-- The property that among any 5 subsets, there exist 2 with intersection of size at least m -/
def IntersectionProperty (m : Nat) : Prop :=
  ∀ (A₁ A₂ A₃ A₄ A₅ : Subset500),
    ∃ (i j : Fin 5), i ≠ j ∧ ((A₁.val ∩ A₂.val).card ≥ m ∨
                              (A₁.val ∩ A₃.val).card ≥ m ∨
                              (A₁.val ∩ A₄.val).card ≥ m ∨
                              (A₁.val ∩ A₅.val).card ≥ m ∨
                              (A₂.val ∩ A₃.val).card ≥ m ∨
                              (A₂.val ∩ A₄.val).card ≥ m ∨
                              (A₂.val ∩ A₅.val).card ≥ m ∨
                              (A₃.val ∩ A₄.val).card ≥ m ∨
                              (A₃.val ∩ A₅.val).card ≥ m ∨
                              (A₄.val ∩ A₅.val).card ≥ m)

theorem biggest_intersection_size : 
  (IntersectionProperty 200 ∧ ∀ m > 200, ¬IntersectionProperty m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biggest_intersection_size_l1049_104911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_range_of_odd_function_l1049_104946

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem positive_range_of_odd_function 
  (f : ℝ → ℝ) 
  (f_odd : OddFunction f)
  (f_deriv : Differentiable ℝ f)
  (h_zero : f (-1) = 0)
  (h_pos : ∀ x > 0, x * (deriv f x) + f x > 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_range_of_odd_function_l1049_104946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_earthling_arrangements_l1049_104913

/-- The number of ways to arrange Martians and Earthlings in a line -/
def lineArrangements (m n : ℕ) : ℕ :=
  n.factorial * Nat.choose (n + 1) m * m.factorial

/-- The number of ways to arrange Martians and Earthlings around a circular table -/
def circularArrangements (m n : ℕ) : ℕ :=
  (n - 1).factorial * Nat.choose n m * m.factorial

/-- Theorem stating the correct number of arrangements for both linear and circular cases -/
theorem martian_earthling_arrangements (m n : ℕ) :
  (∀ arrangement : Fin (m + n) → Bool,
    (∀ i j : Fin (m + n), i < j → arrangement i = true → arrangement j = true → ∃ k, i < k ∧ k < j ∧ arrangement k = false) →
    (lineArrangements m n = Fintype.card (Fin (m + n) → Bool))) ∧
  (∀ arrangement : Fin (m + n) → Bool,
    (∀ i j : Fin (m + n), arrangement i = true → arrangement j = true → ∃ k, k ≠ i ∧ k ≠ j ∧ arrangement k = false) →
    (circularArrangements m n = Fintype.card (Fin (m + n) → Bool))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_earthling_arrangements_l1049_104913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1049_104985

noncomputable def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

noncomputable def V (n : ℕ) : ℝ := Real.sqrt ((fib n)^2 + (fib (n + 2))^2)

theorem triangle_area (n : ℕ) :
  ∃ (a b c : ℝ), a = V n ∧ b = V (n + 1) ∧ c = V (n + 2) ∧
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 1 / 2 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1049_104985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1049_104965

/-- The eccentricity of a hyperbola with equation x²/12 - y²/b² = 1 (b > 0) 
    and distance from foci to asymptotes equal to 2 is 2√3/3 -/
theorem hyperbola_eccentricity (b : ℝ) (h1 : b > 0) 
  (h2 : ∀ (x y : ℝ), x^2/12 - y^2/b^2 = 1 → 
    ∃ (f : ℝ × ℝ), (f.1 - x)^2 + (f.2 - y)^2 = 4) : 
  Real.sqrt ((12 + b^2) / 12) = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1049_104965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_travel_time_l1049_104935

/-- Represents the time taken for a squirrel to travel a given distance at a constant speed -/
noncomputable def travel_time (speed : ℝ) (distance : ℝ) : ℝ :=
  distance / speed

/-- Converts time in hours to minutes -/
noncomputable def hours_to_minutes (hours : ℝ) : ℝ :=
  hours * 60

theorem squirrel_travel_time :
  let speed : ℝ := 3  -- miles per hour
  let distance : ℝ := 2  -- miles
  hours_to_minutes (travel_time speed distance) = 40 := by
  -- Unfold the definitions
  unfold hours_to_minutes travel_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_travel_time_l1049_104935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cantaloupes_l1049_104904

/-- The total number of cantaloupes grown by Keith, Fred, and Jason is 65. -/
theorem total_cantaloupes : 29 + 16 + 20 = 65 := by
  -- Compute the sum
  calc
    29 + 16 + 20 = 45 + 20 := by rfl
    _ = 65 := by rfl

#check total_cantaloupes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cantaloupes_l1049_104904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_seven_highest_average_l1049_104976

def average_of_multiples (n : ℕ) (upper_bound : ℕ) : ℚ :=
  let first_term := n
  let last_term := (upper_bound / n) * n
  (first_term + last_term) / 2

theorem multiples_of_seven_highest_average :
  let upper_bound := 50
  ∀ m : ℕ, m ∈ ({3, 4, 5, 6} : Set ℕ) → 
    average_of_multiples 7 upper_bound > average_of_multiples m upper_bound :=
by
  intro upper_bound m h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_seven_highest_average_l1049_104976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_guilty_l1049_104902

/-- Represents a defendant -/
inductive Defendant
  | A
  | B
  | C

/-- Represents the guilt status of a defendant -/
def IsGuilty (d : Defendant) : Prop := sorry

/-- Represents the truth value of a defendant's statement -/
def ToldTruth (d : Defendant) : Prop := sorry

/-- Represents the accusation made by a defendant -/
def Accused (d1 d2 : Defendant) : Prop := sorry

/-- Only one defendant is guilty -/
axiom one_guilty : ∃! d : Defendant, IsGuilty d

/-- Each defendant accused one of the other two -/
axiom all_accused : ∀ d : Defendant, ∃ d' : Defendant, d ≠ d' ∧ Accused d d'

/-- Some defendants told the truth while others lied -/
axiom some_truth_some_lie : (∃ d : Defendant, ToldTruth d) ∧ (∃ d : Defendant, ¬ToldTruth d)

/-- Either there were two consecutive truths or two consecutive lies -/
axiom consecutive_statements : 
  (∃ d1 d2 : Defendant, d1 ≠ d2 ∧ ToldTruth d1 ∧ ToldTruth d2) ∨
  (∃ d1 d2 : Defendant, d1 ≠ d2 ∧ ¬ToldTruth d1 ∧ ¬ToldTruth d2)

/-- A defendant's statement is true if and only if the accused is guilty -/
axiom statement_truth : ∀ d1 d2 : Defendant, Accused d1 d2 → (ToldTruth d1 ↔ IsGuilty d2)

/-- Prove that Defendant C is guilty -/
theorem c_is_guilty : IsGuilty Defendant.C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_guilty_l1049_104902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_4_16_5_25_l1049_104914

theorem digits_of_4_16_5_25 : ∀ n : ℕ, n = 4^16 * 5^25 → (Nat.log 10 n + 1 : ℕ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_4_16_5_25_l1049_104914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1049_104993

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x : ℕ | 1 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1049_104993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l1049_104978

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem liquid_rise_ratio 
  (r₁ r₂ r_marble : ℝ) 
  (h₁ h₂ : ℝ) 
  (h₁_pos : 0 < h₁) 
  (h₂_pos : 0 < h₂) 
  (r₁_pos : 0 < r₁) 
  (r₂_pos : 0 < r₂) 
  (r_marble_pos : 0 < r_marble)
  (vol_equal : cone_volume r₁ h₁ = cone_volume r₂ h₂) 
  (r₁_val : r₁ = 5) 
  (r₂_val : r₂ = 10) 
  (r_marble_val : r_marble = 2) :
  (sphere_volume r_marble / cone_volume r₁ h₁) / (sphere_volume r_marble / cone_volume r₂ h₂) = 4 := by
  sorry

#eval 1  -- This line is added to ensure the file is parsed without errors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l1049_104978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1049_104983

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (3, 0)
def focus2 : ℝ × ℝ := (-3, 0)

-- Theorem statement
theorem ellipse_focus_distance (x y : ℝ) :
  is_on_ellipse x y →
  distance x y focus1.1 focus1.2 = 6 →
  distance x y focus2.1 focus2.2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1049_104983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_BDFE_l1049_104943

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (6, 0)

-- Define midpoints
def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
def F : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the area function for a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_of_quadrilateral_BDFE : 
  triangleArea B D F + triangleArea D F E = 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_BDFE_l1049_104943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_sum_factorial_squares_100_l1049_104928

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorial_squares (n : ℕ) : ℕ :=
  (List.range n).map (λ i => (factorial (i + 1))^2) |>.sum

theorem units_digit_sum_factorial_squares_100 :
  units_digit (sum_factorial_squares 100) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_sum_factorial_squares_100_l1049_104928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_intervals_g_min_value_g_max_value_l1049_104954

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := 1 + 2 * sqrt 3 * sin x * cos x - 2 * (sin x)^2

-- Define the function g
def g (x : ℝ) : ℝ := f (x - π/6)

-- Theorem for monotonic increase intervals of f
theorem f_monotonic_increase_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π - π/3) (k * π + π/6)) :=
sorry

-- Theorem for minimum value of g in [-π/2, 0]
theorem g_min_value :
  ∃ (x : ℝ), x ∈ Set.Icc (-π/2) 0 ∧ g x = -2 ∧ ∀ y ∈ Set.Icc (-π/2) 0, g y ≥ -2 :=
sorry

-- Theorem for maximum value of g in [-π/2, 0]
theorem g_max_value :
  ∃ (x : ℝ), x ∈ Set.Icc (-π/2) 0 ∧ g x = 1 ∧ ∀ y ∈ Set.Icc (-π/2) 0, g y ≤ 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_intervals_g_min_value_g_max_value_l1049_104954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1049_104982

/-- Represents the speed of a train in km/hr given its length in meters and time to cross a pole in seconds -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (18 / 5)

/-- Theorem stating that a train with length 125 meters crossing a pole in 9 seconds has a speed of 50 km/hr -/
theorem train_speed_calculation :
  train_speed 125 9 = 50 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1049_104982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l1049_104924

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_side_sum_range (t : Triangle) :
  t.b^2 + t.c^2 - t.a^2 = t.b * t.c →  -- Law of cosines condition
  0 < t.b * t.c * Real.cos t.A →  -- AB · BC > 0 condition
  t.a = Real.sqrt 3 / 2 →  -- Given side length
  3/2 < t.b + t.c ∧ t.b + t.c < 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l1049_104924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_multiple_has_infinite_zeros_or_nines_l1049_104940

-- Define a predicate for a real number having infinitely many 0s or 9s in its decimal expansion
noncomputable def has_infinite_zeros_or_nines (x : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ (Int.floor (x * 10^m) % 10 = 0 ∨ Int.floor (x * 10^m) % 10 = 9)

-- The main theorem
theorem irrational_multiple_has_infinite_zeros_or_nines (α : ℝ) (h : Irrational α) :
  ∃ k : ℤ, k ≠ 0 ∧ has_infinite_zeros_or_nines (k * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_multiple_has_infinite_zeros_or_nines_l1049_104940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boil_certain_l1049_104915

/-- Represents an event that can occur --/
inductive Event
| CoinToss : Event
| WaterBoil : Event
| MeetAcquaintance : Event
| SunRevolvesEarth : Event

/-- Defines what it means for an event to be certain --/
def is_certain (e : Event) : Prop :=
  match e with
  | Event.WaterBoil => True
  | _ => False

/-- States that water boils at 100°C under standard atmospheric pressure --/
axiom water_boils_100C : ∀ (temp : ℝ) (pressure : ℝ),
  pressure = 1 → temp = 100 → is_certain Event.WaterBoil

/-- Theorem stating that heating water to 100°C at standard atmospheric pressure is a certain event --/
theorem water_boil_certain :
  is_certain Event.WaterBoil := by
  exact True.intro


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boil_certain_l1049_104915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l1049_104921

open MeasureTheory

-- Define the curve function
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the area calculation
noncomputable def area : ℝ := ∫ x in Set.Icc 0 1, f x

-- Theorem statement
theorem area_under_curve : area = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l1049_104921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1049_104953

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem f_min_max :
  ∃ (min max : ℝ), min = 0 ∧ max = 9 ∧
  (∀ x ∈ domain, f x ≥ min) ∧
  (∃ x ∈ domain, f x = min) ∧
  (∀ x ∈ domain, f x ≤ max) ∧
  (∃ x ∈ domain, f x = max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1049_104953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1049_104971

theorem trig_problem (θ : Real) 
  (h1 : Real.sin (θ + Real.pi/4) = Real.sqrt 2/4) 
  (h2 : θ ∈ Set.Ioo (-Real.pi/2) 0) : 
  Real.sin θ * Real.cos θ = -3/8 ∧ Real.cos θ - Real.sin θ = Real.sqrt 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1049_104971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_wins_hv_game_alice_wins_hvd_game_l1049_104917

/-- Represents the possible moves in the game -/
inductive Move where
  | Horizontal
  | Vertical
  | Diagonal

/-- Represents a player in the game -/
inductive Player where
  | Alice
  | Bob

/-- Represents a position on the 8x8 chessboard -/
structure Position where
  row : Fin 8
  col : Fin 8

/-- Represents the game state -/
structure GameState where
  alicePos : Position
  bobPos : Position
  currentPlayer : Player
  round : Nat

/-- Defines a legal move in the game -/
def isLegalMove (start finish : Position) (moveType : List Move) : Prop :=
  (Move.Horizontal ∈ moveType ∧ (start.row = finish.row ∧ (start.col.val + 1 = finish.col.val ∨ start.col.val = finish.col.val + 1))) ∨
  (Move.Vertical ∈ moveType ∧ (start.col = finish.col ∧ (start.row.val + 1 = finish.row.val ∨ start.row.val = finish.row.val + 1))) ∨
  (Move.Diagonal ∈ moveType ∧ ((start.row.val + 1 = finish.row.val ∨ start.row.val = finish.row.val + 1) ∧ 
                                (start.col.val + 1 = finish.col.val ∨ start.col.val = finish.col.val + 1)))

/-- Defines a winning condition for Alice -/
def aliceWins (state : GameState) : Prop :=
  state.alicePos = state.bobPos

/-- Defines a winning condition for Bob -/
def bobWins (state : GameState) : Prop :=
  state.round > 2012 ∧ ¬aliceWins state

/-- Apply strategies for both players to reach the final game state -/
def applyStrategies (initialState : GameState) (aliceStrategy bobStrategy : GameState → Position) (allowedMoves : List Move) : GameState :=
  sorry

/-- Theorem: Bob has a winning strategy in the horizontal-vertical movement game -/
theorem bob_wins_hv_game :
  ∃ (initialState : GameState),
    ∀ (aliceStrategy : GameState → Position),
      ∃ (bobStrategy : GameState → Position),
        bobWins (applyStrategies initialState aliceStrategy bobStrategy [Move.Horizontal, Move.Vertical]) :=
  by sorry

/-- Theorem: Alice has a winning strategy in the horizontal-vertical-diagonal movement game -/
theorem alice_wins_hvd_game :
  ∀ (initialState : GameState),
    ∃ (aliceStrategy : GameState → Position),
      ∀ (bobStrategy : GameState → Position),
        aliceWins (applyStrategies initialState aliceStrategy bobStrategy [Move.Horizontal, Move.Vertical, Move.Diagonal]) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_wins_hv_game_alice_wins_hvd_game_l1049_104917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_example_l1049_104979

/-- The number of terms in an arithmetic sequence with given parameters -/
def arithmeticSequenceLength (start : ℕ) (step : ℕ) (last : ℕ) : ℕ :=
  (last - start) / step + 1

/-- Theorem: The arithmetic sequence starting at 6, incrementing by 4, and ending at 154 has 38 terms -/
theorem arithmetic_sequence_length_example : arithmeticSequenceLength 6 4 154 = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_example_l1049_104979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1049_104966

def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := Set.range (λ n : ℕ => (n : ℝ))

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1049_104966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l1049_104948

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ (a d : ℝ), 
    (Real.sqrt (49 + k) = a) ∧ 
    (Real.sqrt (325 + k) = a + d) ∧ 
    (Real.sqrt (784 + k) = a + 2*d)) → 
  k = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l1049_104948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_linear_maps_points_l1049_104998

/-- A fractional-linear function mapping complex points -/
noncomputable def fractional_linear (z : ℂ) : ℂ :=
  Complex.I * (Complex.I - z) / (Complex.I + z)

theorem fractional_linear_maps_points :
  let z₁ : ℂ := 1
  let z₂ : ℂ := Complex.I
  let z₃ : ℂ := -1
  let w₁ : ℂ := -1
  let w₂ : ℂ := 0
  let w₃ : ℂ := 1
  fractional_linear z₁ = w₁ ∧
  fractional_linear z₂ = w₂ ∧
  fractional_linear z₃ = w₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_linear_maps_points_l1049_104998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_determination_l1049_104967

/-- A trigonometric function with amplitude A, frequency ω, and phase φ. -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- The theorem stating the conditions and conclusion about the frequency ω. -/
theorem frequency_determination (A φ : ℝ) (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x y, x ∈ Set.Icc 0 (π/3) → y ∈ Set.Icc 0 (π/3) → x < y → 
    (f A ω φ x < f A ω φ y ∨ f A ω φ x > f A ω φ y)) →
  f A ω φ 0 = f A ω φ (5*π/6) →
  f A ω φ 0 = -f A ω φ (π/3) →
  ω = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_determination_l1049_104967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1049_104944

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  f (2 - 2*a) > f a ↔ a < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1049_104944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_y_value_l1049_104951

noncomputable section

open Real

variable (a b c x p q r y : ℝ)

theorem log_equality_implies_y_value
  (h1 : x ≠ 1)
  (h2 : ∃ (base : ℝ), base > 0 ∧ base ≠ 1 ∧
    log a / p = log b / q ∧
    log b / q = log c / r ∧
    log c / r = log x)
  (h3 : a^3 * b / c^2 = x^y) :
  y = 3*p + q - 2*r :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_y_value_l1049_104951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_U_A_is_closed_interval_l1049_104918

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x ≥ -3}

-- Define set A
def A : Set ℝ := {x : ℝ | x > 1}

-- Define the complement of A with respect to U
def complement_U_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_U_A_is_closed_interval :
  complement_U_A = Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_U_A_is_closed_interval_l1049_104918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_purchase_optimization_l1049_104994

/-- Represents the flour purchasing problem for a factory --/
structure FlourPurchase where
  daily_need : ℝ  -- Daily flour need in tons
  price_per_ton : ℝ  -- Price per ton of flour in yuan
  storage_cost : ℝ  -- Storage cost per ton per day in yuan
  shipping_fee : ℝ  -- Shipping fee per purchase in yuan
  discount_threshold : ℝ  -- Minimum purchase amount for discount in tons
  discount_rate : ℝ  -- Discount rate (e.g., 0.9 for 10% discount)

/-- Calculates the average daily total cost without discount --/
noncomputable def avg_daily_cost (fp : FlourPurchase) (interval : ℝ) : ℝ :=
  (fp.storage_cost * fp.daily_need * interval * (interval + 1) / 2 + fp.shipping_fee) / interval +
  fp.daily_need * fp.price_per_ton

/-- Calculates the average daily total cost with discount --/
noncomputable def avg_daily_cost_with_discount (fp : FlourPurchase) (interval : ℝ) : ℝ :=
  (fp.storage_cost * fp.daily_need * interval * (interval + 1) / 2 + fp.shipping_fee) / interval +
  fp.daily_need * fp.price_per_ton * fp.discount_rate

/-- The main theorem to prove --/
theorem flour_purchase_optimization (fp : FlourPurchase) 
  (h1 : fp.daily_need = 6)
  (h2 : fp.price_per_ton = 1800)
  (h3 : fp.storage_cost = 3)
  (h4 : fp.shipping_fee = 900)
  (h5 : fp.discount_threshold = 210)
  (h6 : fp.discount_rate = 0.9) :
  (∃ (optimal_interval : ℝ), 
    optimal_interval = 10 ∧ 
    ∀ (x : ℝ), x > 0 → avg_daily_cost fp optimal_interval ≤ avg_daily_cost fp x) ∧
  (avg_daily_cost_with_discount fp 35 < avg_daily_cost fp 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_purchase_optimization_l1049_104994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_not_necessarily_prism_l1049_104947

-- Define a polyhedron
structure Polyhedron where
  faces : Set Face
  is_closed : Bool
  is_bounded : Bool

-- Define a face
structure Face where
  vertices : Set Point
  is_planar : Bool

-- Define a parallelogram
def is_parallelogram (f : Face) : Prop := sorry

-- Define parallel faces
def are_parallel (f1 f2 : Face) : Prop := sorry

-- Define a prism
def is_prism (p : Polyhedron) : Prop := sorry

-- Theorem statement
theorem polyhedron_not_necessarily_prism 
  (p : Polyhedron) 
  (h1 : ∃ f1 f2 : Face, f1 ∈ p.faces ∧ f2 ∈ p.faces ∧ are_parallel f1 f2)
  (h2 : ∀ f : Face, f ∈ p.faces → (∀ f1 f2, f1 ∈ p.faces → f2 ∈ p.faces → are_parallel f1 f2 → f ≠ f1 ∧ f ≠ f2) → is_parallelogram f) :
  ¬(∀ q : Polyhedron, 
    (∃ f1 f2 : Face, f1 ∈ q.faces ∧ f2 ∈ q.faces ∧ are_parallel f1 f2) ∧ 
    (∀ f : Face, f ∈ q.faces → (∀ g1 g2, g1 ∈ q.faces → g2 ∈ q.faces → are_parallel g1 g2 → f ≠ g1 ∧ f ≠ g2) → is_parallelogram f) 
    → is_prism q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_not_necessarily_prism_l1049_104947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_number_combinations_l1049_104949

def valid_combination (a b c d : ℕ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧
  (Finset.filter (λ x => x < 9) {a + b, a + c, a + d, b + c, b + d, c + d}).card = 2 ∧
  (Finset.filter (λ x => x = 9) {a + b, a + c, a + d, b + c, b + d, c + d}).card = 2 ∧
  (Finset.filter (λ x => x > 9) {a + b, a + c, a + d, b + c, b + d, c + d}).card = 2

theorem card_number_combinations :
  {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | valid_combination a b c d} =
  {(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_number_combinations_l1049_104949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_monotonous_condition_max_value_l1049_104931

-- Define the function f(x) = x^2 + ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the interval [-5, 5]
def interval : Set ℝ := Set.Icc (-5) 5

-- Part I: f(x) is an even function iff a = 0
theorem even_function_condition (a : ℝ) :
  (∀ x ∈ interval, f a x = f a (-x)) ↔ a = 0 :=
sorry

-- Part II: f(x) is monotonous iff a ≥ 10 or a ≤ -10
theorem monotonous_condition (a : ℝ) :
  (StrictMono (f a) ∨ StrictAnti (f a)) ↔ (a ≥ 10 ∨ a ≤ -10) :=
sorry

-- Part III: Maximum value of f(x)
theorem max_value (a : ℝ) :
  (∃ x ∈ interval, ∀ y ∈ interval, f a x ≥ f a y) ∧
  (a ≥ 0 → ∃ x ∈ interval, f a x = 27 + 5*a) ∧
  (a < 0 → ∃ x ∈ interval, f a x = 27 - 5*a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_monotonous_condition_max_value_l1049_104931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1049_104933

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent line
def tangentLine : Set (ℝ × ℝ) := {p | p.1 + p.2 - 1 = 0}

-- Define the line on which the center lies
def centerLine : Set (ℝ × ℝ) := {p | p.2 = -4 * p.1}

-- Define the tangent point
def tangentPoint : ℝ × ℝ := (3, -2)

-- Define the property of being tangent
def isTangent (c : Circle) (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ l ∧ Real.sqrt ((c.center.1 - p.1)^2 + (c.center.2 - p.2)^2) = c.radius

-- Theorem statement
theorem circle_equation (c : Circle) :
  c.center ∈ centerLine ∧
  isTangent c tangentLine tangentPoint →
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔
    (x - 1)^2 + (y + 4)^2 = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1049_104933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_initial_temp_l1049_104986

/-- Represents the candy-making process with given heating and cooling rates,
    final temperatures, and total time. -/
structure CandyProcess where
  heating_rate : ℝ
  cooling_rate : ℝ
  heating_final_temp : ℝ
  cooling_final_temp : ℝ
  total_time : ℝ

/-- Calculates the initial temperature of the candy mixture given the candy-making process. -/
noncomputable def initial_temperature (p : CandyProcess) : ℝ :=
  p.heating_final_temp - p.heating_rate * (p.total_time - (p.heating_final_temp - p.cooling_final_temp) / p.cooling_rate)

/-- Theorem stating that for the given candy-making process, the initial temperature is 60 degrees. -/
theorem candy_initial_temp :
  let p : CandyProcess := {
    heating_rate := 5,
    cooling_rate := 7,
    heating_final_temp := 240,
    cooling_final_temp := 170,
    total_time := 46
  }
  initial_temperature p = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_initial_temp_l1049_104986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_together_days_l1049_104929

/-- Represents the work completion rates and total time --/
structure WorkData where
  a_rate : ℚ  -- A's work rate (fraction of work completed per day)
  b_rate : ℚ  -- B's work rate (fraction of work completed per day)
  total_time : ℚ  -- Total time to complete the work
  total_work : ℚ  -- Total amount of work (normalized to 1)

/-- Calculates the number of days A and B worked together --/
def days_worked_together (data : WorkData) : ℚ :=
  let combined_rate := data.a_rate + data.b_rate
  (data.total_work - data.a_rate * data.total_time) / (combined_rate - data.a_rate)

/-- Theorem stating that A and B worked together for 2 days --/
theorem work_together_days (data : WorkData) 
  (h1 : data.a_rate = 1 / 15)
  (h2 : data.b_rate = 1 / 10)
  (h3 : data.total_time = 12)
  (h4 : data.total_work = 1) :
  days_worked_together data = 2 := by
  sorry

#eval days_worked_together ⟨1/15, 1/10, 12, 1⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_together_days_l1049_104929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_closed_figure_l1049_104969

open Real MeasureTheory

/-- The area of the closed figure formed by f(x) and g(x) over one period -/
theorem area_closed_figure (a : ℝ) (h : a > 0) : 
  let f (x : ℝ) := a * sin (a * x) + cos (a * x)
  let g (x : ℝ) := 1
  let period := 2 * π / a
  ∫ x in (Set.Icc 0 period), (g x - f x) = π / a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_closed_figure_l1049_104969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l1049_104905

/-- The distance traveled by the center of a ball rolling along a track with two semicircular arcs and a straight segment -/
theorem ball_travel_distance (ball_diameter R₁ R₂ straight_length : ℝ) :
  ball_diameter = 6 →
  R₁ = 150 →
  R₂ = 90 →
  straight_length = 100 →
  let ball_radius := ball_diameter / 2
  let arc1_distance := π * (R₁ - ball_radius)
  let arc2_distance := π * (R₂ - ball_radius)
  let total_distance := arc1_distance + arc2_distance + straight_length
  total_distance = 234 * π + 100 := by
  intro h1 h2 h3 h4
  -- Proof steps would go here
  sorry

#check ball_travel_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l1049_104905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mothers_full_time_jobs_l1049_104909

/-- Represents the fraction of mothers who held full-time jobs -/
def M : ℚ := sorry

/-- Represents the fraction of fathers who held full-time jobs -/
def F : ℚ := 9/10

/-- Represents the fraction of parents surveyed who were women -/
def W : ℚ := 2/5

/-- Represents the fraction of parents who did not hold full-time jobs -/
def N : ℚ := 4/25

theorem mothers_full_time_jobs : M = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mothers_full_time_jobs_l1049_104909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_your_money_calculation_l1049_104925

/-- Represents an amount of money in yuan and jiao -/
structure Money where
  yuan : ℕ
  jiao : ℕ
  jiao_valid : jiao < 10

/-- Converts a decimal number to Money -/
def toMoney (amount : ℚ) : Money :=
  let total_jiao := (amount * 10).floor.toNat
  ⟨total_jiao / 10, total_jiao % 10, by sorry⟩

theorem your_money_calculation (my_money : ℚ) (difference : ℚ) 
    (h1 : my_money = 13.5)
    (h2 : difference = 0.5) : 
    toMoney (my_money - difference) = ⟨13, 0, by sorry⟩ := by
  sorry

#check your_money_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_your_money_calculation_l1049_104925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_difference_count_l1049_104906

theorem set_difference_count {U A B : Finset ℕ} : 
  (Finset.card U = 190) →
  (Finset.card A = 105) →
  (Finset.card B = 49) →
  (Finset.card (A ∩ B) = 23) →
  Finset.card (U \ (A ∪ B)) = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_difference_count_l1049_104906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_is_18_l1049_104960

/-- The cost of an album in dollars -/
def album_cost : ℚ := 20

/-- The percentage discount of the CD compared to the album -/
def cd_discount_percentage : ℚ := 30

/-- The additional cost of the book compared to the CD in dollars -/
def book_additional_cost : ℚ := 4

/-- The cost of the CD in dollars -/
noncomputable def cd_cost : ℚ := album_cost * (1 - cd_discount_percentage / 100)

/-- The cost of the book in dollars -/
noncomputable def book_cost : ℚ := cd_cost + book_additional_cost

theorem book_cost_is_18 : book_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_is_18_l1049_104960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l1049_104990

/-- Represents a vertically oriented parabola -/
structure VerticalParabola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ
  h : ∀ x, eq x = a * x^2 + b

/-- The directrix of a vertical parabola -/
noncomputable def directrix (p : VerticalParabola) : ℝ := -1 / (4 * p.a) + p.b

/-- The given parabola y = 4x^2 + 8 -/
def given_parabola : VerticalParabola where
  a := 4
  b := 8
  eq := λ x ↦ 4 * x^2 + 8
  h := by intro x; rfl

theorem directrix_of_given_parabola :
  directrix given_parabola = 127 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l1049_104990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_graph_l1049_104920

-- Define the basic types and constants
variable (X : ℝ × ℝ)  -- Coordinates of Island X
variable (r : ℝ)      -- Radius of the semicircle and distance from X to B, C, and D

-- Define the path of the ship
noncomputable def ship_path : ℝ → ℝ × ℝ := sorry

-- Define the distance function from a point to Island X
noncomputable def distance_to_X (p : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem ship_distance_graph :
  ∃ (t₁ t₂ t₃ : ℝ), 0 < t₁ ∧ t₁ < t₂ ∧ t₂ < t₃ ∧
  (∀ t, 0 ≤ t ∧ t ≤ t₁ → distance_to_X (ship_path t) = r) ∧
  (∃ t, t₁ < t ∧ t < t₂ ∧ distance_to_X (ship_path t) > r) ∧
  (∃ t, t₂ < t ∧ t < t₃ ∧ distance_to_X (ship_path t) < r) ∧
  (distance_to_X (ship_path t₃) = r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_graph_l1049_104920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OCD_l1049_104955

-- Define the shapes
structure Trapezoid (A B C D : Point) : Type := (dummy : Unit)
structure Parallelogram (C D E F : Point) : Type := (dummy : Unit)
structure Triangle (O C D : Point) : Type := (dummy : Unit)

-- Define the area function
noncomputable def area {α : Type} : α → ℝ := sorry

-- Define the given shapes
variable (A B C D E F O : Point)
variable (trapezoid : Trapezoid A B C D)
variable (parallelogram : Parallelogram C D E F)
variable (triangle : Triangle O C D)

-- State the theorem
theorem area_of_triangle_OCD :
  area trapezoid = 320 →
  area parallelogram = 240 →
  area triangle = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OCD_l1049_104955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_test_count_l1049_104957

theorem joes_test_count (n : ℕ) (initial_avg new_avg lowest_score : ℚ)
  (initial_avg_def : initial_avg = 45)
  (new_avg_def : new_avg = 50)
  (lowest_score_def : lowest_score = 30)
  : n = 4 := by
  -- All tests are equally weighted (implicit in the arithmetic mean calculation)
  
  -- The proof would go here, but we'll use sorry to skip it
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_test_count_l1049_104957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_harmonic_series_convergence_l1049_104922

/-- Geometric series -/
noncomputable def geometric_series (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

/-- Harmonic series -/
noncomputable def harmonic_series (n : ℕ) : ℝ := 1 - 1 / (n + 1 : ℝ)

theorem geometric_series_convergence (a q : ℝ) :
  (∀ n : ℕ, geometric_series a q n = a * (1 - q^n) / (1 - q)) ∧
  (abs q < 1 → ∃ S : ℝ, Filter.Tendsto (fun n => geometric_series a q n) Filter.atTop (nhds S) ∧ S = a / (1 - q)) ∧
  (abs q ≥ 1 → ¬ ∃ S : ℝ, Filter.Tendsto (fun n => geometric_series a q n) Filter.atTop (nhds S)) :=
by sorry

theorem harmonic_series_convergence :
  (∀ n : ℕ, harmonic_series n = 1 - 1 / (n + 1 : ℝ)) ∧
  ∃ S : ℝ, Filter.Tendsto harmonic_series Filter.atTop (nhds S) ∧ S = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_harmonic_series_convergence_l1049_104922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1049_104959

open Real Set

noncomputable section

def f (x : ℝ) : ℝ := 4 * cos x * sin (x + π/6) - 1

theorem f_properties :
  (∀ k : ℤ, StrictMonoOn f (Icc (-(π/3) + k*π) (π/6 + k*π))) ∧
  (∀ a : ℝ, (∀ x ∈ Ioo (-π/4) (π/4), sin x ^ 2 + a * f (x + π/6) + 1 > 6 * cos x ^ 4) → a > 5/2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1049_104959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_expansion_ratio_l1049_104968

-- Define the original radius and the increase
variable (r k : ℝ)

-- Define the condition that k is positive
variable (h : k > 0)

-- Define the ratio of new circumference to original diameter
noncomputable def ratio (r k : ℝ) : ℝ := (2 * Real.pi * (r + k)) / (2 * r)

-- State the theorem
theorem circle_expansion_ratio (r k : ℝ) (h : k > 0) :
  ratio r k = Real.pi * (1 + k / r) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_expansion_ratio_l1049_104968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1049_104941

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) :
  d < 0 →
  arithmetic_sequence a d →
  (3 * Real.sqrt 5)^2 = (-a 2) * (a 9) →
  sum_of_arithmetic_sequence a 10 = 20 →
  d = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l1049_104941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l1049_104980

theorem triangle_angle_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) ≤ 
  Real.sin (α + β) + Real.sin (β + γ) + Real.sin (γ + α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l1049_104980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_triangles_l1049_104950

/-- A triangle with integral side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of all triangles with integral side lengths and perimeter 9 -/
def triangles_with_perimeter_9 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 9}

/-- Two triangles are considered different if they have different sets of side lengths -/
def different_triangles (t1 t2 : IntTriangle) : Prop :=
  (t1.a ≠ t2.a ∨ t1.a ≠ t2.b ∨ t1.a ≠ t2.c) ∧
  (t1.b ≠ t2.a ∨ t1.b ≠ t2.b ∨ t1.b ≠ t2.c) ∧
  (t1.c ≠ t2.a ∨ t1.c ≠ t2.b ∨ t1.c ≠ t2.c)

/-- The main theorem stating that there are exactly 2 different triangles with integral side lengths and perimeter 9 -/
theorem two_different_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ triangles_with_perimeter_9 ∧
    t2 ∈ triangles_with_perimeter_9 ∧
    different_triangles t1 t2 ∧
    ∀ (t3 : IntTriangle),
      t3 ∈ triangles_with_perimeter_9 →
      (¬different_triangles t1 t3 ∨ ¬different_triangles t2 t3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_triangles_l1049_104950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_donation_correct_l1049_104961

/-- Proves that Margo's donation amount is $4300, given Julie's donation and half the difference --/
def margo_donation (julie_donation : ℕ) (half_difference : ℕ) : ℕ :=
  julie_donation - (2 * half_difference)

theorem margo_donation_correct :
  margo_donation 4700 200 = 4300 := by
  rfl

#eval margo_donation 4700 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_donation_correct_l1049_104961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_nonempty_l1049_104997

/-- A closed interval on the real line -/
structure ClosedInterval where
  left : ℝ
  right : ℝ
  is_valid : left ≤ right

/-- The property that any two intervals in a collection have a point in common -/
def PairwiseOverlap (S : Set ClosedInterval) : Prop :=
  ∀ I J, I ∈ S → J ∈ S → ∃ x : ℝ, I.left ≤ x ∧ x ≤ I.right ∧ J.left ≤ x ∧ x ≤ J.right

/-- The theorem stating that the intersection of all intervals is non-empty -/
theorem intersection_nonempty
  (S : Set ClosedInterval)
  (h_finite : Set.Finite S)
  (h_overlap : PairwiseOverlap S) :
  ∃ x : ℝ, ∀ I ∈ S, I.left ≤ x ∧ x ≤ I.right := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_nonempty_l1049_104997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_income_l1049_104945

/-- Calculates the annual income from a stock investment -/
theorem stock_investment_income 
  (investment : ℝ) 
  (stock_price : ℝ) 
  (dividend_rate : ℝ) 
  (h1 : investment = 6800) 
  (h2 : stock_price = 136) 
  (h3 : dividend_rate = 0.5) : 
  (investment / stock_price) * (stock_price * dividend_rate) = 3400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_income_l1049_104945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_is_18_cups_l1049_104912

/-- Represents the water flow rate in cups per 10 minutes -/
def flowRate (time : ℕ) : ℚ :=
  if time ≤ 6 then 2
  else if time ≤ 12 then 4
  else 0

/-- Calculates the total water collected over a given time period in 10-minute intervals -/
def totalWater (intervals : ℕ) : ℚ :=
  (List.sum (List.map flowRate (List.range intervals)))

/-- The amount of water left after Shawn dumps half -/
def waterLeft : ℚ := totalWater 12 / 2

theorem water_left_is_18_cups : waterLeft = 18 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_is_18_cups_l1049_104912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_construction_theorem_l1049_104962

-- Define a hexagon type
structure Hexagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define a function to check if two line segments are parallel
def are_parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := q1
  let (x4, y4) := q2
  (x2 - x1) * (y4 - y3) = (y2 - y1) * (x4 - x3)

-- Define a function to calculate the length of a line segment
noncomputable def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem hexagon_construction_theorem 
  (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  ∃ (h : Hexagon), 
    segment_length h.A h.B = a ∧
    segment_length h.B h.C = b ∧
    segment_length h.C h.D = c ∧
    segment_length h.D h.E = d ∧
    segment_length h.E h.F = e ∧
    segment_length h.F h.A = f ∧
    are_parallel h.A h.B h.D h.E ∧
    are_parallel h.B h.C h.E h.F ∧
    are_parallel h.C h.D h.F h.A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_construction_theorem_l1049_104962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tilling_time_is_220_minutes_l1049_104981

/-- Represents the tilling problem with given dimensions and tilling parameters -/
structure TillingProblem where
  plot_width : ℚ
  plot_length : ℚ
  tiller_width : ℚ
  tilling_speed : ℚ  -- feet per second

/-- Calculates the time required to till a plot with given parameters -/
def time_to_till (p : TillingProblem) : ℚ :=
  let area := p.plot_width * p.plot_length
  let num_swaths := p.plot_width / p.tiller_width
  let swath_time := p.plot_length / p.tilling_speed
  (num_swaths * swath_time) / 60  -- Convert to minutes

/-- The main theorem stating that the time to till the given plot is 220 minutes -/
theorem tilling_time_is_220_minutes :
  let p : TillingProblem := {
    plot_width := 110,
    plot_length := 120,
    tiller_width := 2,
    tilling_speed := 1/2
  }
  time_to_till p = 220 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tilling_time_is_220_minutes_l1049_104981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_sees_emerson_for_five_minutes_one_twelfth_hour_equals_five_minutes_l1049_104932

/-- Represents the scenario of Emily and Emerson's movement --/
structure ScenarioData where
  emily_speed : ℚ
  emerson_speed : ℚ
  initial_distance : ℚ
  billboard_distance : ℚ
  loss_of_sight_distance : ℚ

/-- Calculates the time Emily can see Emerson --/
noncomputable def timeEmilySeesEmerson (data : ScenarioData) : ℚ :=
  let relative_speed := data.emily_speed - data.emerson_speed
  let total_distance := data.initial_distance + data.billboard_distance
  let time_to_billboard := total_distance / relative_speed
  let time_to_lose_sight := data.loss_of_sight_distance / relative_speed
  min time_to_billboard time_to_lose_sight

/-- Theorem stating that Emily sees Emerson for 5 minutes --/
theorem emily_sees_emerson_for_five_minutes (data : ScenarioData) 
  (h1 : data.emily_speed = 15)
  (h2 : data.emerson_speed = 9)
  (h3 : data.initial_distance = 1/4)
  (h4 : data.billboard_distance = 3/4)
  (h5 : data.loss_of_sight_distance = 1/2) :
  timeEmilySeesEmerson data = 1/12 := by
  sorry

/-- Converts the time from hours to minutes --/
def hoursToMinutes (hours : ℚ) : ℚ :=
  hours * 60

/-- Theorem stating that 1/12 hour equals 5 minutes --/
theorem one_twelfth_hour_equals_five_minutes :
  hoursToMinutes (1/12) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_sees_emerson_for_five_minutes_one_twelfth_hour_equals_five_minutes_l1049_104932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_12_in_18_factorial_l1049_104991

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem highest_power_of_12_in_18_factorial : 
  ∃ k : ℕ, k = 8 ∧ 12^k ∣ factorial 18 ∧ ∀ m : ℕ, 12^m ∣ factorial 18 → m ≤ k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_12_in_18_factorial_l1049_104991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_sum_of_triangles_l1049_104992

/-- Represents a quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Calculates the area of a triangle using Heron's formula -/
noncomputable def heronArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of the given quadrilateral is the sum of two triangles -/
theorem quadrilateral_area_is_sum_of_triangles 
  (quad : Quadrilateral) 
  (h1 : quad.a = 25) 
  (h2 : quad.b = 30) 
  (h3 : quad.c = 25) 
  (h4 : quad.d = 18) 
  (h5 : quad.b > quad.a ∧ quad.b > quad.c ∧ quad.b > quad.d) : 
  ∃ (diag : ℝ), 
    let areaABC := heronArea quad.a quad.b diag
    let areaADC := heronArea quad.c quad.d diag
    ∃ (totalArea : ℝ), totalArea = areaABC + areaADC := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_sum_of_triangles_l1049_104992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_integral_l1049_104973

-- Define the binomial
noncomputable def binomial (x : ℝ) := (x^2 - 1/x)^5

-- Define the coefficient of x in the expansion
noncomputable def a : ℝ := -10

-- Theorem statement
theorem binomial_integral : ∫ (x : ℝ) in a..(-1), 2*x = -99 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_integral_l1049_104973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_fraction_l1049_104984

noncomputable section

-- Define the side length of the larger square
def large_square_side : ℝ := 6

-- Define the side length of the inner square
def inner_square_side : ℝ := 2

-- Define the side length of the rhombus (half the diagonal of a 1x1 square)
def rhombus_side : ℝ := Real.sqrt 2 / 2

-- Theorem statement
theorem rhombus_area_fraction :
  (1 / 2 * rhombus_side ^ 2) / (large_square_side ^ 2) = 1 / 144 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_fraction_l1049_104984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l1049_104987

def f (x : ℝ) : ℝ := x^2 + x^(2/3) - 4

theorem root_interval (a : ℤ) : 
  (∃ m : ℝ, m ∈ Set.Ioo (a : ℝ) ((a + 1) : ℝ) ∧ f m = 0) → a = 1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l1049_104987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1049_104900

theorem power_equation_solution (k : ℝ) : 
  (1/2 : ℝ)^22 * (1/81 : ℝ)^k = (1/18 : ℝ)^22 → k = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1049_104900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_480_degrees_l1049_104930

theorem tan_negative_480_degrees : Real.tan ((-480 : ℝ) * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_480_degrees_l1049_104930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_vertical_asymptotes_l1049_104908

/-- The function g(x) defined by (x^2 - 3x + k) / (x^2 - 2x - 8) -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + k) / (x^2 - 2*x - 8)

/-- Definition of having a vertical asymptote at a point -/
def HasVerticalAsymptoteAt (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ M > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x| > M

/-- Theorem stating that g(x) has exactly two vertical asymptotes iff k ≠ -4 and k ≠ -10 -/
theorem two_vertical_asymptotes (k : ℝ) : 
  (∃! (a b : ℝ), a ≠ b ∧ HasVerticalAsymptoteAt (g k) a ∧ HasVerticalAsymptoteAt (g k) b) ↔ 
  (k ≠ -4 ∧ k ≠ -10) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_vertical_asymptotes_l1049_104908
