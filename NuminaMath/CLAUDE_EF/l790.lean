import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extensions_from_one_and_two_five_operations_form_l790_79064

/-- Extension operation for two positive real numbers -/
noncomputable def extend (a b : ℝ) : ℝ := a * b + a + b

/-- Perform n operations starting with two numbers -/
noncomputable def performOperations (a b : ℝ) : ℕ → ℝ
  | 0 => max a b
  | n + 1 =>
    let c := extend a b
    performOperations (max a c) (max b c) n

theorem two_extensions_from_one_and_two :
  performOperations 1 2 2 = 17 := by sorry

theorem five_operations_form (p q : ℝ) (m n : ℕ) (h : p > q) (h' : q > 0) :
  ∃ (m n : ℕ), performOperations p q 5 = (q + 1)^m * (p + 1)^n - 1 ∧ m + n = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extensions_from_one_and_two_five_operations_form_l790_79064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faucet_drips_ten_times_per_minute_l790_79075

/-- Represents the properties of a dripping faucet -/
structure DrippingFaucet where
  water_wasted_per_hour : ℚ
  water_per_drop : ℚ

/-- Calculates the number of drops per minute for a given faucet -/
def drops_per_minute (faucet : DrippingFaucet) : ℚ :=
  (faucet.water_wasted_per_hour / faucet.water_per_drop) / 60

/-- Theorem: A faucet wasting 30 mL per hour with 0.05 mL per drop drips 10 times per minute -/
theorem faucet_drips_ten_times_per_minute :
  let faucet : DrippingFaucet := { water_wasted_per_hour := 30, water_per_drop := 1/20 }
  drops_per_minute faucet = 10 := by
  sorry

#eval drops_per_minute { water_wasted_per_hour := 30, water_per_drop := 1/20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faucet_drips_ten_times_per_minute_l790_79075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l790_79028

theorem cos_sin_equation (x : ℝ) : 
  Real.cos x - 5 * Real.sin x = 2 → Real.sin x + 5 * Real.cos x = -676 / 211 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l790_79028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_octal_l790_79076

def is_two_digit_octal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 7 ∧ n = 8 * a + b

def reverse_octal_digits (n : ℕ) : ℕ :=
  let a := n / 8
  let b := n % 8
  8 * b + a

theorem arithmetic_geometric_mean_octal (x y : ℕ) :
  x ≠ y →
  x > 0 →
  y > 0 →
  is_two_digit_octal ((x + y) / 2) →
  (Real.sqrt (x * y : ℝ) : ℝ) = reverse_octal_digits ((x + y) / 2) →
  |Int.ofNat x - Int.ofNat y| = 66 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_octal_l790_79076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l790_79025

-- Define the base-2 logarithm
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem order_of_numbers :
  let a : ℝ := (2 : ℝ) ^ (0.6 : ℝ)
  let b : ℝ := log2 0.6
  let c : ℝ := log2 0.4
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l790_79025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l790_79088

-- Define the circles
def circle_C : Real × Real × Real := (4, 4, 5)
def circle_D : Real × Real × Real := (10, 4, 3)

-- Define the function to calculate the area
noncomputable def area_between_circles_and_x_axis (C D : Real × Real × Real) : Real :=
  let (xC, yC, rC) := C
  let (xD, yD, rD) := D
  44.484 - 25 * Real.arccos (4/5) - 9 * Real.arccos (1/3)

-- State the theorem
theorem area_calculation :
  area_between_circles_and_x_axis circle_C circle_D =
  44.484 - 25 * Real.arccos (4/5) - 9 * Real.arccos (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l790_79088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l790_79027

theorem expression_evaluation (α : Real) : 
  let expr := (Real.sqrt (1 + Real.sin α) + Real.sqrt (1 - Real.sin α)) / 
               (Real.sqrt (1 + Real.sin α) - Real.sqrt (1 - Real.sin α))
  (0 < α ∧ α < π/2 → expr = 1 / Real.tan (α/2)) ∧
  (π/2 < α ∧ α < π → expr = Real.tan (α/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l790_79027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_draws_count_probability_sum_five_l790_79015

-- Define the set of ball labels
def BallLabels : Finset Nat := {1, 2, 3, 4}

-- Define the type for a draw (a pair of distinct integers)
def Draw := {pair : Nat × Nat // pair.1 ∈ BallLabels ∧ pair.2 ∈ BallLabels ∧ pair.1 < pair.2}

-- Define the set of all possible draws
def AllDraws : Finset Draw := sorry

-- Define the condition for a draw to sum to 5
def SumsToFive (d : Draw) : Prop := d.val.1 + d.val.2 = 5

-- Define the set of draws that sum to 5
def DrawsSummingToFive : Finset Draw := sorry

-- Theorem statements
theorem all_draws_count : Finset.card AllDraws = 6 := sorry

theorem probability_sum_five : 
  (Finset.card DrawsSummingToFive : ℚ) / (Finset.card AllDraws : ℚ) = 1/3 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_draws_count_probability_sum_five_l790_79015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solution_in_interval_l790_79002

-- Define the function f(x) = 2x^2 - 2^x
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.exp (Real.log 2 * x)

-- Theorem statement
theorem exists_solution_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solution_in_interval_l790_79002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_of_inclination_l790_79036

/-- 
The angle_of_inclination function calculates the angle of inclination
of a line passing through two points in degrees.
-/
noncomputable def angle_of_inclination (p1 p2 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  if x2 = x1 then 90 else Real.arctan ((y2 - y1) / (x2 - x1)) * (180 / Real.pi)

/-- 
Given two points C(m,n) and D(m,-n) where n ≠ 0, 
the angle of inclination of the line passing through these points is 90°.
-/
theorem line_angle_of_inclination (m n : ℝ) (hn : n ≠ 0) :
  let C : ℝ × ℝ := (m, n)
  let D : ℝ × ℝ := (m, -n)
  angle_of_inclination C D = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_of_inclination_l790_79036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_equivalence_min_distance_ellipse_to_line_l790_79005

-- Define the line l in polar coordinates
def line_l_polar (p θ : ℝ) : Prop := p * Real.sin (θ - Real.pi/4) = 2 * Real.sqrt 2

-- Define the line l in rectangular coordinates
def line_l_rect (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 / 9 = 1

-- Statement 1: Equivalence of polar and rectangular equations of line l
theorem polar_to_rect_equivalence :
  ∀ x y : ℝ, (∃ p θ : ℝ, x = p * Real.cos θ ∧ y = p * Real.sin θ ∧ line_l_polar p θ) ↔ line_l_rect x y :=
by sorry

-- Statement 2: Minimum distance from ellipse C to line l
theorem min_distance_ellipse_to_line :
  ∃ d : ℝ, d = 2 * Real.sqrt 2 - Real.sqrt 6 ∧
  (∀ x y : ℝ, ellipse_C x y → 
    ∀ d' : ℝ, (d' = |x - y + 4| / Real.sqrt 2 → d ≤ d')) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_equivalence_min_distance_ellipse_to_line_l790_79005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_intersection_l790_79039

/-- The relationship between k and c when y₂ passes through the vertex of y₁ -/
theorem vertex_intersection (k c : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, x^2 + 2*x + c = k*(-1) + 2) → c + k = 3 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_intersection_l790_79039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l790_79043

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 5 then x^2 - x + 12 else 2^x

-- State the theorem
theorem solution_exists (a : ℝ) : f (f a) = 16 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l790_79043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l790_79006

-- Define the focus point F
def F : ℝ × ℝ := (0, -4)

-- Define the directrix line
def directrix : ℝ → ℝ := λ _ => -4

-- Define the distance from a point to the focus
noncomputable def dist_to_focus (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - F.1)^2 + (p.2 - F.2)^2)

-- Define the distance from a point to the directrix
def dist_to_directrix (p : ℝ × ℝ) : ℝ := |p.2 - directrix p.1|

-- Define the condition for the moving point
def moving_point_condition (p : ℝ × ℝ) : Prop :=
  dist_to_focus p = dist_to_directrix p + 1

-- Theorem: The trajectory of the moving point is x^2 = -16y
theorem trajectory_equation :
  ∀ p : ℝ × ℝ, moving_point_condition p ↔ p.1^2 = -16 * p.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l790_79006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l790_79077

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the line y = 2
def on_line_y_2 (x y : ℝ) : Prop := y = 2

-- Define perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- The main theorem
theorem min_distance_AB :
  ∀ (xA yA xB yB : ℝ),
    on_line_y_2 xA yA →
    on_ellipse xB yB →
    perpendicular xA yA xB yB →
    ∀ (x y : ℝ),
      on_line_y_2 x y →
      on_ellipse x y →
      distance xA yA xB yB ≤ distance xA yA x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l790_79077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_mass_ratio_l790_79035

/-- Represents the ratio of masses in an equilateral triangle -/
noncomputable def mass_ratio (a x : ℝ) : ℝ := (a * (2 * a - 3 * x)) / ((a - x)^2) - 1

theorem equilateral_triangle_mass_ratio 
  (a : ℝ) 
  (h_a : a > 0) 
  (x : ℝ) 
  (h_x : 0 ≤ x ∧ x ≤ a/2) :
  -- 1. The ratio of masses
  mass_ratio a x = (a * (2 * a - 3 * x)) / ((a - x)^2) - 1 ∧
  -- 2. Equality of masses
  (mass_ratio a x = 1 ↔ (x = 0 ∨ x = a/2)) ∧
  -- 3. Maximum ratio occurs at x = a/3
  (∀ y, 0 ≤ y ∧ y ≤ a/2 → mass_ratio a (a/3) ≥ mass_ratio a y) ∧
  -- 4. Maximum ratio value
  mass_ratio a (a/3) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_mass_ratio_l790_79035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OH_squared_value_l790_79051

/-- Triangle ABC with circumcenter O and orthocenter H -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ

/-- The circumradius of the triangle -/
def circumradius : ℝ := 10

/-- The side lengths of the triangle -/
noncomputable def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The sum of squares of side lengths -/
noncomputable def sum_of_squares (t : Triangle) : ℝ :=
  let (a, b, c) := side_lengths t
  a^2 + b^2 + c^2

/-- The squared distance between circumcenter and orthocenter -/
noncomputable def OH_squared (t : Triangle) : ℝ := sorry

theorem OH_squared_value (t : Triangle) :
  sum_of_squares t = 50 →
  OH_squared t = 850 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_OH_squared_value_l790_79051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_3_l790_79096

-- Definition of supporting function
def is_supporting_function (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x ≥ g x

-- Proposition 2
theorem prop_2 : is_supporting_function (λ x ↦ x + Real.sin x) (λ x ↦ x - 1) := by
  sorry

-- Proposition 3
theorem prop_3 : ∀ a : ℝ, is_supporting_function Real.exp (λ x ↦ a * x) → 0 ≤ a ∧ a ≤ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_3_l790_79096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_condition_l790_79085

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The vector a parameterized by l -/
def a (l : ℝ) : ℝ × ℝ := (2, l)

/-- The vector b parameterized by l -/
def b (l : ℝ) : ℝ × ℝ := (l - 1, 1)

theorem parallel_vectors_condition (l : ℝ) :
  are_parallel (a l) (b l) ↔ l = -1 ∨ l = 2 := by
  sorry

#check parallel_vectors_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_condition_l790_79085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_color_count_l790_79008

theorem smallest_color_count :
  ∃ (n : ℕ),
    0 < n ∧
    (∃ (coloring : ℕ → Fin n),
      (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 100 → (x + y) % 4 = 0 → coloring x ≠ coloring y)) ∧
    (∀ m, 0 < m →
      (∃ f : ℕ → Fin m, ∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 100 → (x + y) % 4 = 0 → f x ≠ f y) →
      n ≤ m) ∧
    n = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_color_count_l790_79008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_radius_l790_79020

theorem larger_sphere_radius (r : ℝ) (V : ℝ → ℝ) (R : ℝ) : 
  r = 2 →
  (∀ x, V x = (4 / 3) * Real.pi * x^3) →
  V R = 9 * V r →
  R = 2 * (9 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_radius_l790_79020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_l790_79047

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the externally_tangent relation
def externally_tangent (c1 c2 : Circle) : Prop :=
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = c1.radius + c2.radius

-- Define the internally_tangent relation
def internally_tangent (c1 c2 : Circle) : Prop :=
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = |c1.radius - c2.radius|

-- Define the internally_tangent_at_center relation
def internally_tangent_at_center (c1 c2 : Circle) : Prop :=
  c1.center = c2.center ∧ c2.radius = 2 * c1.radius

-- Define the problem setup
def problem_setup (A B C D : Circle) : Prop :=
  -- Circles A, B, and C are externally tangent to each other
  externally_tangent A B ∧ externally_tangent B C ∧ externally_tangent A C ∧
  -- Circle A has radius 3
  A.radius = 3 ∧
  -- Circle A is internally tangent to circle D at the center of D
  internally_tangent_at_center A D ∧
  -- Circles B and C have different radii
  B.radius ≠ C.radius ∧
  -- rC = 2rB
  C.radius = 2 * B.radius ∧
  -- All circles are internally tangent to circle D
  internally_tangent B D ∧ internally_tangent C D

-- The theorem to be proved
theorem circle_radii (A B C D : Circle) :
  problem_setup A B C D → B.radius = 1.5 ∧ C.radius = 3 :=
by
  sorry

-- Example usage (optional)
#check circle_radii

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_l790_79047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l790_79017

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) else -3*x + 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (f a) = 2^(-(f a))) ↔ a ∈ Set.Ici (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l790_79017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_domain_range_pair_l790_79031

def f (x : ℝ) : ℝ := x^2 - 2 * abs x + 2

theorem unique_domain_range_pair :
  ∃! (a b : ℝ), a < b ∧
    (∀ x, f x ∈ Set.Icc (2 * a) (2 * b) ↔ x ∈ Set.Icc a b) ∧
    a = 1/2 ∧ b = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_domain_range_pair_l790_79031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l790_79095

open Real

/-- A, B, and C are interior angles of a triangle ABC -/
def is_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π

/-- Definition of the function f -/
noncomputable def f (B : ℝ) : ℝ :=
  4 * sin B * (cos (π/4 - B/2))^2 + cos (2*B)

/-- Theorem stating the range of m -/
theorem range_of_m (m : ℝ) :
  (∀ A B C : ℝ, is_triangle A B C → ∀ B, f B - m < 2) → m > 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l790_79095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l790_79022

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 - 2*x + 2)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ 
    ((1 - Real.sqrt 2) / 2 ≤ y ∧ y ≤ (1 + Real.sqrt 2) / 2) := by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l790_79022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l790_79042

-- Define the triangle DEF
def triangle_DEF (DE EF DF : ℝ) : Prop := DE = 26 ∧ EF = 26 ∧ DF = 40

-- Define the semi-perimeter
noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define Heron's formula for the area of a triangle
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem area_of_triangle_DEF :
  ∀ DE EF DF : ℝ, triangle_DEF DE EF DF → heron_area DE EF DF = 332 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l790_79042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l790_79055

noncomputable def f (x : ℝ) : ℝ := min (min (4 * x + 1) (x + 4)) (-x + 8)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 6 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l790_79055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_increasing_iff_m_lower_bound_l790_79037

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a * x - 6 * Real.log x
def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

-- Theorem 1
theorem f_monotone_increasing : 
  Monotone (f 1) := by sorry

-- Theorem 2
theorem g_increasing_iff (a : ℝ) : 
  Monotone (g a) ↔ a ≥ 5/2 := by sorry

-- Theorem 3
theorem m_lower_bound (m : ℝ) :
  (∃ x₁ ∈ Set.Icc 1 2, ∃ x₂, g 2 x₁ ≥ h m x₂) → m ≥ 8 - 5 * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_increasing_iff_m_lower_bound_l790_79037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l790_79050

/-- Given a sequence where the first three terms are x, 3x+3, and 6x+6 respectively,
    the fourth term of this sequence is -24, assuming it is a geometric sequence. -/
theorem geometric_sequence_fourth_term (x : ℝ) : 
  ∃ (r : ℝ), (3*x + 3 = x * r) ∧ (6*x + 6 = (3*x + 3) * r) → (6*x + 6) * r = -24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l790_79050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_free_subset_l790_79078

theorem arithmetic_mean_free_subset (k : ℕ) : ∃ (S : Finset ℕ), 
  (∀ x, x ∈ S → x < 3^k) ∧ 
  S.card = 2^k ∧
  (∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → x + y ≠ 2*z) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_free_subset_l790_79078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_16_exponent_l790_79057

theorem base_16_exponent (x : ℝ) : (16 : ℝ) ^ x = (4 : ℝ) ^ 14 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_16_exponent_l790_79057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_perpendicular_lines_result_l790_79058

-- Define the slopes of the two lines
noncomputable def slope1 (a : ℝ) : ℝ := -a / (1 + a)
noncomputable def slope2 (a : ℝ) : ℝ := -(a + 1) / (3 - 2*a)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := 
  a ≠ -1 ∧ a ≠ 3/2 ∧ slope1 a * slope2 a = -1

-- State the theorem
theorem perpendicular_lines (a : ℝ) : 
  perpendicular a ↔ a = 3 := by
  sorry

-- State the final result including the special case a = -1
theorem perpendicular_lines_result (a : ℝ) :
  (perpendicular a ∨ a = -1) ↔ (a = 3 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_perpendicular_lines_result_l790_79058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l790_79021

/-- Given a vector a = (1, 2, -2) and a unit vector b, 
    the maximum value of |a - 2b| is 5. -/
theorem max_vector_difference :
  let a : ℝ × ℝ × ℝ := (1, 2, -2)
  (⨆ (b : ℝ × ℝ × ℝ) (_ : ‖b‖ = 1), ‖a - 2 • b‖) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l790_79021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A₁_coordinates_l790_79080

/-- The coordinates of point A -/
def A : ℝ × ℝ := (-1, 2)

/-- The coordinates of point M as a function of t -/
def M (t : ℝ) : ℝ × ℝ := (t - 1, 2 * t + 2)

/-- The distance A moves to become A₁ -/
noncomputable def distance : ℝ := Real.sqrt 5

/-- The possible coordinates of A₁ -/
def A₁_possibilities : Set (ℝ × ℝ) := {(-2, 0), (0, 4)}

/-- Theorem stating that A₁ must be one of the two possible points -/
theorem A₁_coordinates :
  ∃ (A₁ : ℝ × ℝ), 
    (∃ (t : ℝ), A₁.1 = A.1 + t * (M t).1 - A.1 ∧
                A₁.2 = A.2 + t * (M t).2 - A.2) ∧
    (A₁.1 - A.1)^2 + (A₁.2 - A.2)^2 = distance^2 ∧
    A₁ ∈ A₁_possibilities :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A₁_coordinates_l790_79080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_approx_l790_79026

/-- The molar mass of barium (Ba) in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.33

/-- The molar mass of bromine (Br) in g/mol -/
noncomputable def molar_mass_Br : ℝ := 79.90

/-- The molar mass of barium bromide (BaBr2) in g/mol -/
noncomputable def molar_mass_BaBr2 : ℝ := molar_mass_Ba + 2 * molar_mass_Br

/-- The mass percentage of bromine (Br) in barium bromide (BaBr2) -/
noncomputable def mass_percentage_Br : ℝ := (2 * molar_mass_Br / molar_mass_BaBr2) * 100

/-- Theorem stating that the mass percentage of Br in BaBr2 is approximately 53.79% -/
theorem mass_percentage_Br_approx :
  |mass_percentage_Br - 53.79| < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_approx_l790_79026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l790_79059

theorem cos_alpha_plus_pi_third (α : ℝ) 
  (h1 : Real.sin α = (4 * Real.sqrt 3) / 7)
  (h2 : 0 < α ∧ α < π / 2) :
  Real.cos (α + π / 3) = -11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l790_79059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_integers_l790_79062

theorem product_of_integers (P Q R S : ℕ) : 
  P + Q + R + S = 48 →
  P + 3 = Q - 3 →
  P + 3 = R * 3 →
  P + 3 = S / 3 →
  P * Q * R * S = 5832 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_integers_l790_79062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_arrangement_theorem_l790_79033

def blue_plates : ℕ := 6
def red_plates : ℕ := 3
def green_plates : ℕ := 3
def orange_plates : ℕ := 2

def total_plates : ℕ := blue_plates + red_plates + green_plates + orange_plates

def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def arrangements_with_adjacent_greens : ℕ :=
  circular_arrangements (total_plates - green_plates + 1) * Nat.factorial green_plates

theorem plate_arrangement_theorem :
  circular_arrangements total_plates / (Nat.factorial blue_plates * Nat.factorial red_plates * Nat.factorial green_plates * Nat.factorial orange_plates) -
  arrangements_with_adjacent_greens / (Nat.factorial blue_plates * Nat.factorial red_plates * Nat.factorial orange_plates) = 79695 := by
  sorry

#eval circular_arrangements total_plates / (Nat.factorial blue_plates * Nat.factorial red_plates * Nat.factorial green_plates * Nat.factorial orange_plates) -
  arrangements_with_adjacent_greens / (Nat.factorial blue_plates * Nat.factorial red_plates * Nat.factorial orange_plates)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_arrangement_theorem_l790_79033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l790_79045

-- Define the differential equation
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv (deriv y))) x - 100 * (deriv y) x = 20 * Real.exp (10 * x) + 100 * Real.cos (10 * x)

-- Define the general solution
noncomputable def general_solution (C₁ C₂ C₃ : ℝ) (x : ℝ) : ℝ :=
  C₁ + C₂ * Real.exp (10 * x) + C₃ * Real.exp (-10 * x) + (x / 10) * Real.exp (10 * x) - (1 / 20) * Real.sin (10 * x)

-- Theorem statement
theorem general_solution_satisfies_equation :
  ∀ C₁ C₂ C₃, differential_equation (general_solution C₁ C₂ C₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l790_79045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_difference_l790_79023

theorem quadratic_equation_solution_difference : 
  let f : ℝ → ℝ := λ x => 2*x^2 - 10*x + 18 - (2*x + 42)
  let solutions := {x : ℝ | f x = 0}
  ∃ x₁ x₂ : ℝ, x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 21 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_difference_l790_79023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_corner_is_three_l790_79018

-- Define a 3x3 grid type
def Grid := Fin 3 → Fin 3 → Nat

-- Define a predicate to check if a number is valid (1, 2, or 3)
def isValidNumber (n : Nat) : Prop := n = 1 ∨ n = 2 ∨ n = 3

-- Define a predicate to check if a row or column contains unique valid numbers
def hasUniqueValidNumbers (v : Fin 3 → Nat) : Prop :=
  ∀ i j, i ≠ j → v i ≠ v j ∧ isValidNumber (v i) ∧ isValidNumber (v j)

-- Define the initial grid setup
def initialGrid : Grid :=
  fun i j => match i, j with
  | 0, 0 => 1
  | 0, 2 => 3
  | 1, 0 => 3
  | 1, 1 => 2
  | _, _ => 0

-- Define a predicate to check if a grid is valid
def isValidGrid (g : Grid) : Prop :=
  (∀ i, hasUniqueValidNumbers (fun j => g i j)) ∧
  (∀ j, hasUniqueValidNumbers (fun i => g i j))

-- Theorem statement
theorem lower_right_corner_is_three :
  ∀ (g : Grid), g = initialGrid → isValidGrid g → g 2 2 = 3 := by
  sorry

#check lower_right_corner_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_corner_is_three_l790_79018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letters_per_day_l790_79079

/-- The number of packages received per day -/
def packages_per_day : ℕ := 20

/-- The total pieces of mail handled in six months -/
def total_mail_six_months : ℕ := 14400

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of months -/
def months : ℕ := 6

/-- Theorem: The post office receives 60 letters per day -/
theorem letters_per_day :
  total_mail_six_months / (months * days_per_month) - packages_per_day = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letters_per_day_l790_79079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_shopping_cost_is_135_l790_79097

/-- Represents the cost of Bob's shopping trip --/
def bobsShoppingCost (
  hammerPrice : ℚ)
  (hammerCount : ℕ)
  (nailPrice : ℚ)
  (nailCount : ℕ)
  (sawPrice : ℚ)
  (sawCount : ℕ)
  (paintPrice : ℚ)
  (paintCount : ℕ)
  (sawDiscount : ℚ)
  (couponValue : ℚ)
  (couponThreshold : ℚ) : ℚ :=
  let hammerTotal := hammerPrice * hammerCount
  let nailTotal := nailPrice * nailCount
  let sawTotal := sawPrice * sawCount * (1 - sawDiscount)
  let paintTotal := paintPrice * paintCount
  let subtotal := hammerTotal + nailTotal + sawTotal + paintTotal
  if subtotal ≥ couponThreshold then subtotal - couponValue else subtotal

/-- Theorem stating that Bob's shopping cost is $135 --/
theorem bobs_shopping_cost_is_135 :
  bobsShoppingCost 15 4 3 6 12 3 20 2 (1/4) 10 50 = 135 := by
  -- Unfold the definition of bobsShoppingCost
  unfold bobsShoppingCost
  -- Simplify the arithmetic expressions
  simp [Rat.cast_mul, Rat.cast_add, Rat.cast_sub, Rat.cast_le]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_shopping_cost_is_135_l790_79097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l790_79013

/-- Given a line ℓ passing through point P (x₀, y₀) and perpendicular to the line Ax + By + C = 0,
    the equation of line ℓ is B(x - x₀) - A(y - y₀) = 0 -/
theorem perpendicular_line_equation (A B C x₀ y₀ : ℝ) (hB : B ≠ 0) :
  let original_line := {p : ℝ × ℝ | A * p.1 + B * p.2 + C = 0}
  let point_P := (x₀, y₀)
  let perpendicular_line := {p : ℝ × ℝ | B * (p.1 - x₀) - A * (p.2 - y₀) = 0}
  (point_P ∈ perpendicular_line) ∧
  (∀ p q : ℝ × ℝ, p ∈ original_line → q ∈ perpendicular_line →
    (q.2 - p.2) * (q.1 - p.1) = - (A / B) * (B / A)) := by
  intro original_line point_P perpendicular_line
  have h1 : point_P ∈ perpendicular_line := by
    simp [perpendicular_line, point_P]
  have h2 : ∀ p q : ℝ × ℝ, p ∈ original_line → q ∈ perpendicular_line →
    (q.2 - p.2) * (q.1 - p.1) = - (A / B) * (B / A) := by
    intro p q hp hq
    sorry -- The actual proof would go here
  exact ⟨h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l790_79013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l790_79082

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / a^2 - y^2 / b^2 = 1}

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_eccentricity (a b : ℝ) :
  a > 0 ∧ b > 0 ∧                          -- Given conditions on a and b
  (∀ x y, y = -2*x → (x, y) ∈ hyperbola a b) ∧  -- One asymptote is 2x + y = 0
  (Real.sqrt 5, 0) ∈ hyperbola a b →       -- One focus is at (√5, 0)
  eccentricity a b = Real.sqrt 5 :=        -- Conclusion: eccentricity is √5
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l790_79082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l790_79066

-- Define the plane and points
variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (A B C P : V)

-- Define the theorem
theorem vector_relation (h : (C - B) + 2 • (A - B) = 3 • (P - B)) :
  2 • (A - P) + (C - P) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l790_79066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_comparison_l790_79092

/-- Calculates the cost after discount for Mall A -/
noncomputable def cost_mall_a (amount : ℝ) : ℝ :=
  if amount ≤ 200 then amount
  else 200 + 0.85 * (amount - 200)

/-- Calculates the cost after discount for Mall B -/
noncomputable def cost_mall_b (amount : ℝ) : ℝ :=
  if amount ≤ 100 then amount
  else 100 + 0.9 * (amount - 100)

theorem mall_comparison : 
  cost_mall_b 300 < cost_mall_a 300 ∧ 
  ∃ (x : ℝ), x > 100 ∧ cost_mall_a x = cost_mall_b x ∧ x = 400 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_comparison_l790_79092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_sum_l790_79014

theorem triangle_angles_sum (A B C : ℝ) (p q r s : ℕ) : 
  A + B + C = π →
  π/2 < B →
  (Real.cos A)^2 + (Real.cos B)^2 + 2 * Real.sin A * Real.sin B * Real.cos C = 17/9 →
  (Real.cos B)^2 + (Real.cos C)^2 + 2 * Real.sin B * Real.sin C * Real.cos A = 15/8 →
  (Real.cos C)^2 + (Real.cos A)^2 + 2 * Real.sin C * Real.sin A * Real.cos B = (p - q * Real.sqrt (r : ℝ)) / s →
  Nat.Coprime (p + q) s →
  ∀ (prime : ℕ), Nat.Prime prime → ¬(prime^2 ∣ r) →
  p + q + r + s = 182 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_sum_l790_79014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_reflection_l790_79048

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (3, 11π/9, π/4),
    prove that the point (-x, y, z) has spherical coordinates (3, 2π/9, π/4). -/
theorem spherical_coordinates_reflection (x y z : ℝ) :
  (∃ ρ θ φ : ℝ, ρ = 3 ∧ θ = (11 * π) / 9 ∧ φ = π / 4 ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    z = ρ * Real.cos φ) →
  (∃ ρ' θ' φ' : ℝ, ρ' = 3 ∧ θ' = (2 * π) / 9 ∧ φ' = π / 4 ∧
    -x = ρ' * Real.sin φ' * Real.cos θ' ∧
    y = ρ' * Real.sin φ' * Real.sin θ' ∧
    z = ρ' * Real.cos φ') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_reflection_l790_79048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_condition_l790_79086

/-- The function f(x) = -1/2 * x^2 + x + 4 -/
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + x + 4

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := -x + 1

theorem function_increasing_condition (x : ℝ) :
  f' x > 0 ↔ x < 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_condition_l790_79086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximations_for_small_alpha_l790_79030

/-- For small values of α, the following approximations hold:
    1. 1/(1 + α) ≈ 1 - α
    2. 1/(1 - α) ≈ 1 + α
    3. N/(N + α) ≈ 1 - α/N
-/
theorem approximations_for_small_alpha (α : ℝ) (N : ℝ) (h : |α| < 1) :
  (∃ ε₁ ε₂ ε₃ : ℝ, 
    (|ε₁| < |α|) ∧ 
    (|ε₂| < |α|) ∧ 
    (|ε₃| < |α|) ∧ 
    ((1 : ℝ) / (1 + α) = 1 - α + ε₁) ∧
    ((1 : ℝ) / (1 - α) = 1 + α + ε₂) ∧
    (N / (N + α) = 1 - α / N + ε₃)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximations_for_small_alpha_l790_79030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_x_intercepts_l790_79081

noncomputable def x_intercepts (a b : Real) : Int :=
  ⌊(1000 / Real.pi)⌋ - ⌊(100 / Real.pi)⌋

theorem count_x_intercepts :
  x_intercepts 0.001 0.01 = 287 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_x_intercepts_l790_79081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l790_79067

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the expression S as a function of n
noncomputable def S (n : ℤ) : ℂ := i^(n+1) + i^(-n)

-- Theorem statement
theorem distinct_values_of_S :
  ∃ (s : Finset ℂ), (∀ n : ℤ, S n ∈ s) ∧ Finset.card s = 4 := by
  -- We'll use sorry to skip the proof for now
  sorry

#check distinct_values_of_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_S_l790_79067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_R_squared_better_fit_narrower_band_better_fit_regression_analysis_fitting_effect_l790_79019

/-- Represents the coefficient of determination in regression analysis -/
def R_squared : ℝ → ℝ := sorry

/-- Represents the width of the residual plot's band in regression analysis -/
def residual_band_width : ℝ → ℝ := sorry

/-- Represents the fitting effect of a regression model -/
def fitting_effect : ℝ → ℝ := sorry

/-- States that a larger R² indicates a better fitting effect -/
theorem larger_R_squared_better_fit :
  ∀ (R1 R2 : ℝ), R1 > R2 → fitting_effect (R_squared R1) > fitting_effect (R_squared R2) := by sorry

/-- States that a narrower residual band indicates a better fitting effect -/
theorem narrower_band_better_fit :
  ∀ (w1 w2 : ℝ), w1 < w2 → fitting_effect (1 / residual_band_width w1) > fitting_effect (1 / residual_band_width w2) := by sorry

/-- The main theorem combining both conditions for better fitting effect -/
theorem regression_analysis_fitting_effect :
  (∀ (R1 R2 : ℝ), R1 > R2 → fitting_effect (R_squared R1) > fitting_effect (R_squared R2)) ∧
  (∀ (w1 w2 : ℝ), w1 < w2 → fitting_effect (1 / residual_band_width w1) > fitting_effect (1 / residual_band_width w2)) := by
  constructor
  · exact larger_R_squared_better_fit
  · exact narrower_band_better_fit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_R_squared_better_fit_narrower_band_better_fit_regression_analysis_fitting_effect_l790_79019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_remainders_l790_79041

theorem max_distinct_remainders (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ i : ℕ, i ≥ 1 → a i ∣ a (i + 1)) →
  (∀ i : ℕ, b i = a i % 210) →
  (∃ n : ℕ, ∀ m : ℕ, m > n → b m ∈ Finset.image b (Finset.range n)) →
  Finset.card (Finset.image b (Finset.range (Nat.succ 10000))) ≤ 127 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_remainders_l790_79041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_R_l790_79087

/-- The area of rectangle R in a specific geometric configuration -/
theorem area_of_rectangle_R 
  (large_square_side : ℝ)
  (rect1_side1 : ℝ)
  (rect1_side2 : ℝ)
  (small_square_side : ℝ) :
  rect1_side1 = 2 →
  rect1_side2 = 2 * Real.sqrt 2 →
  small_square_side = 2 →
  large_square_side = 4 + 2 * Real.sqrt 2 →
  large_square_side ^ 2 - (rect1_side1 * rect1_side2 + small_square_side ^ 2) = 20 + 12 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_R_l790_79087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joans_travel_time_l790_79074

/-- Calculates the total travel time given distance, speed, and break time. -/
noncomputable def totalTravelTime (distance : ℝ) (speed : ℝ) (breakTime : ℝ) : ℝ :=
  distance / speed + breakTime

/-- Proves that Joan's travel time is 9 hours given the conditions of the problem. -/
theorem joans_travel_time :
  let distance : ℝ := 480
  let speed : ℝ := 60
  let breakTime : ℝ := 1
  totalTravelTime distance speed breakTime = 9 := by
  -- Unfold the definition of totalTravelTime
  unfold totalTravelTime
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  sorry

#check joans_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joans_travel_time_l790_79074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l790_79040

/-- The time (in hours) required to fill a rectangular tank -/
noncomputable def fill_time (length width depth rate : ℝ) : ℝ :=
  (length * width * depth) / rate

/-- Theorem: The time to fill a rectangular tank with given dimensions and fill rate is 60 hours -/
theorem tank_fill_time :
  let length : ℝ := 10
  let width : ℝ := 6
  let depth : ℝ := 5
  let rate : ℝ := 5
  fill_time length width depth rate = 60 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l790_79040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_is_six_l790_79071

/-- Represents a hyperbola with equation x²/a² - y² = 1 (a > 0) -/
structure Hyperbola where
  a : ℝ
  h_pos : a > 0

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ := sorry

/-- A point on the hyperbola -/
noncomputable def point_on_hyperbola : ℝ × ℝ := (1, 2 * Real.sqrt 2)

/-- Condition that the point lies on the hyperbola -/
def point_satisfies_equation (h : Hyperbola) : Prop :=
  (point_on_hyperbola.1)^2 / h.a^2 - (point_on_hyperbola.2)^2 = 1

/-- Condition that the foci and the point form a right triangle -/
def forms_right_triangle (h : Hyperbola) : Prop :=
  let c := focal_distance h / 2
  (1 - c) * (1 + c) + point_on_hyperbola.2^2 = 0

/-- Main theorem: If the conditions are satisfied, the focal distance is 6 -/
theorem focal_distance_is_six (h : Hyperbola) 
  (h_point : point_satisfies_equation h) 
  (h_triangle : forms_right_triangle h) : 
  focal_distance h = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_is_six_l790_79071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_floor_shaded_area_l790_79091

/-- The shaded area of a rectangular floor tiled with square tiles containing white quarter circles -/
noncomputable def shaded_area (floor_length floor_width tile_size circle_radius : ℝ) : ℝ :=
  let num_tiles := (floor_length / tile_size) * (floor_width / tile_size)
  let tile_area := tile_size^2
  let circle_area := Real.pi * circle_radius^2
  num_tiles * (tile_area - circle_area)

/-- Theorem stating the shaded area of a specific floor configuration -/
theorem specific_floor_shaded_area :
  shaded_area 12 15 2 1 = 180 - 45 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_floor_shaded_area_l790_79091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l790_79063

noncomputable def Hyperbola (f : ℝ) := {(x, y) : ℝ × ℝ | x^2 / f^2 - y^2 / (5 - f^2) = 1}

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

def OnHyperbola (P : ℝ × ℝ) (f : ℝ) : Prop :=
  P ∈ Hyperbola f

def Perpendicular (P : ℝ × ℝ) : Prop :=
  (P.1 + Real.sqrt 5) * (P.1 - Real.sqrt 5) + P.2^2 = 0

def ProductCondition (P : ℝ × ℝ) : Prop :=
  ((P.1 + Real.sqrt 5)^2 + P.2^2) * ((P.1 - Real.sqrt 5)^2 + P.2^2) = 4

theorem hyperbola_equation : 
  ∃ (P : ℝ × ℝ) (f : ℝ), OnHyperbola P f ∧ Perpendicular P ∧ ProductCondition P → f^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l790_79063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l790_79032

theorem inequality_problem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y > 1) :
  x^2 > y^2 ∧ (2:ℝ)^x > (2:ℝ)^y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l790_79032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l790_79084

/-- The function f(x) = 1 - 2sin²(2x) has a smallest positive period of π/2. -/
theorem smallest_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧
    (∀ x : ℝ, (1 - 2 * Real.sin (2 * x)^2) = (1 - 2 * Real.sin (2 * (x + T))^2)) ∧
    (∀ S : ℝ, S > 0 ∧
      (∀ x : ℝ, (1 - 2 * Real.sin (2 * x)^2) = (1 - 2 * Real.sin (2 * (x + S))^2)) → T ≤ S) ∧
    T = π / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l790_79084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ahmed_minimum_grade_to_win_l790_79065

/-- Represents a student's grades and calculates their total score -/
structure Student where
  name : String
  current_grade : ℕ
  final_grade : ℕ
  deriving Repr

/-- Calculates the total score for a student -/
def total_score (s : Student) (num_assignments : ℕ) : ℕ :=
  s.current_grade * num_assignments + s.final_grade

/-- The main theorem stating the minimum grade Ahmed needs to beat Emily -/
theorem ahmed_minimum_grade_to_win 
  (num_assignments : ℕ)
  (ahmed : Student)
  (emily : Student)
  (h_num_assignments : num_assignments = 9)
  (h_ahmed_current : ahmed.current_grade = 91)
  (h_emily_current : emily.current_grade = 92)
  (h_emily_final : emily.final_grade = 90)
  (h_ahmed_beats_emily : total_score ahmed (num_assignments + 1) > total_score emily (num_assignments + 1))
  : ahmed.final_grade = 100 := by
  sorry

/-- A helper function to evaluate the theorem -/
def evaluate_theorem : Bool :=
  let ahmed : Student := ⟨"Ahmed", 91, 100⟩
  let emily : Student := ⟨"Emily", 92, 90⟩
  total_score ahmed 10 > total_score emily 10

#eval evaluate_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ahmed_minimum_grade_to_win_l790_79065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_triangle_ratio_l790_79090

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * (Real.sin x)^2 + (Real.cos (Real.pi/4 - x))^2 - (1 + Real.sqrt 3)/2

theorem max_value_and_triangle_ratio :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = 1 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi/2) → f y ≤ f x) ∧
  (∀ (A B C : ℝ), 0 < A ∧ A < B ∧ B < Real.pi ∧ A + B + C = Real.pi ∧ f A = 1/2 ∧ f B = 1/2 →
    Real.sin A / Real.sin C = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_triangle_ratio_l790_79090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_chord_inclination_l790_79094

theorem parabola_focal_chord_inclination 
  (p : ℝ) (θ : ℝ) 
  (h1 : p > 0)
  (h2 : θ ∈ Set.Icc 0 π)
  (h3 : 8 * p = 2 * p / (Real.sin θ)^2) :
  θ = π / 6 ∨ θ = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_chord_inclination_l790_79094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_cats_correct_l790_79098

def bobs_cats (susan_initial : ℕ) (susan_gave : ℕ) (difference : ℕ) : ℕ :=
  let susan_final := susan_initial - susan_gave
  susan_final - difference

theorem bobs_cats_correct (susan_initial : ℕ) (susan_gave : ℕ) (difference : ℕ) :
  bobs_cats susan_initial susan_gave difference = susan_initial - susan_gave - difference := by
  simp [bobs_cats]

#eval bobs_cats 21 4 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_cats_correct_l790_79098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_speaking_probability_at_least_three_babies_speaking_l790_79068

theorem baby_speaking_probability (n : ℕ) (p : ℝ) (h1 : n = 6) (h2 : p = 1/3) :
  let q := 1 - p
  (Nat.choose n 0 : ℝ) * q^n +
  (Nat.choose n 1 : ℝ) * p * q^(n-1) +
  (Nat.choose n 2 : ℝ) * p^2 * q^(n-2) = 496/729 :=
by sorry

theorem at_least_three_babies_speaking (n : ℕ) (p : ℝ) (h1 : n = 6) (h2 : p = 1/3) :
  let q := 1 - p
  1 - ((Nat.choose n 0 : ℝ) * q^n +
       (Nat.choose n 1 : ℝ) * p * q^(n-1) +
       (Nat.choose n 2 : ℝ) * p^2 * q^(n-2)) = 233/729 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_speaking_probability_at_least_three_babies_speaking_l790_79068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2006_value_l790_79012

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 2 => (2005 * sequence_a (n + 1)) / (2003 * sequence_a (n + 1) + 2005)

theorem a_2006_value : sequence_a 2006 = 1 / 2004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2006_value_l790_79012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_rowed_against_stream_l790_79073

/-- Calculates the distance rowed against a stream given the conditions of the problem -/
noncomputable def distance_rowed (still_water_speed : ℝ) (time_against : ℝ) (time_with : ℝ) : ℝ :=
  let stream_speed := (still_water_speed * (time_against - time_with)) / (time_against + time_with)
  (still_water_speed - stream_speed) * time_against

/-- The theorem stating the distance rowed against the stream under given conditions -/
theorem distance_rowed_against_stream :
  distance_rowed 5 675 450 = 2700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_rowed_against_stream_l790_79073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_longest_side_is_half_l790_79024

/-- Represents a kite-shaped field -/
structure KiteField where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  angle_abc : ℝ
  angle_cda : ℝ

/-- The fraction of crop brought to the longest side of a kite-shaped field -/
noncomputable def fraction_to_longest_side (k : KiteField) : ℝ :=
  1 / 2

/-- Theorem stating that the fraction of crop brought to the longest side is 1/2 -/
theorem fraction_to_longest_side_is_half (k : KiteField) 
  (h1 : k.ab = 120)
  (h2 : k.bc = 80)
  (h3 : k.cd = 80)
  (h4 : k.da = 120)
  (h5 : k.angle_abc = 120)
  (h6 : k.angle_cda = 120) :
  fraction_to_longest_side k = 1 / 2 := by
  sorry

#check fraction_to_longest_side_is_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_longest_side_is_half_l790_79024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_third_l790_79054

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x / ((3x+1)(x-a)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x / ((3*x + 1) * (x - a))

/-- Theorem: If f is an odd function, then a = 1/3 -/
theorem odd_function_implies_a_eq_one_third (a : ℝ) :
  IsOdd (f a) → a = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_third_l790_79054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_graph_properties_l790_79060

structure ColoredGraph (n : ℕ) where
  (vertices : Fin n)
  (red_edge : Fin n → Fin n)
  (blue_edge : Fin n → Fin n)
  (green_edge : Fin n → Fin n)
  (connected : ∀ (u v : Fin n), ∃ (path : List (Fin n)), path.head? = some u ∧ path.getLast? = some v)
  (edge_count : ∀ (v : Fin n), ∃! (r b g : Fin n), 
    red_edge v = r ∧ blue_edge v = b ∧ green_edge v = g ∧ r ≠ v ∧ b ≠ v ∧ g ≠ v)

theorem colored_graph_properties {n : ℕ} (G : ColoredGraph n) :
  (∃ k : ℕ, n = 2 * k) ∧
  (∀ (X : Finset (Fin n)) (h₁ : 1 < X.card) (h₂ : X.card < n),
    ∃ (R B G : ℕ), (Even R ∧ Even B ∧ Even G) ∨ (Odd R ∧ Odd B ∧ Odd G)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_graph_properties_l790_79060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l790_79038

/-- Represents a polynomial with integer coefficients -/
def MyPolynomial (n : ℕ) := Fin n → ℤ

/-- Horner's method for evaluating a polynomial -/
def horner_eval (p : MyPolynomial 7) (x : ℝ) : ℝ :=
  (((((p 6 * x + p 5) * x + p 4) * x + p 3) * x + p 2) * x + p 1) * x + p 0

/-- The number of multiplication operations in Horner's method -/
def horner_mul_ops (n : ℕ) : ℕ := n

/-- The number of addition operations in Horner's method -/
def horner_add_ops (n : ℕ) : ℕ := n

/-- Our specific polynomial -/
def f : MyPolynomial 7 := λ i =>
  match i with
  | 0 => 1
  | 1 => 8
  | 2 => 7
  | 3 => 6
  | 4 => 5
  | 5 => 4
  | 6 => 3

theorem horner_method_operations :
  horner_mul_ops 6 = 6 ∧ horner_add_ops 6 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l790_79038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l790_79061

theorem problem_solution :
  ((3 * Real.sqrt 27 - 2 * Real.sqrt 12) / Real.sqrt 3 = 5) ∧
  (2 * Real.sqrt 8 + 4 * Real.sqrt (1/2) - 3 * Real.sqrt 32 = -4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l790_79061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integers_satisfy_condition_l790_79069

def sum_of_divisors (n : ℕ) : ℕ := (Nat.divisors n).sum id

theorem no_integers_satisfy_condition : 
  ¬ ∃ j : ℕ, 1 ≤ j ∧ j ≤ 5000 ∧ (sum_of_divisors j : ℝ) = 1 + j + 2 * Real.sqrt (j : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integers_satisfy_condition_l790_79069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_left_calculation_correct_l790_79072

-- Define the variables
def mike_apples : ℝ := 7.0
def nancy_eaten : ℝ := 3.0
def keith_apples : ℝ := 6.0
def keith_pears : ℝ := 4.0

-- Define the theorem
theorem apples_left : 
  mike_apples + keith_apples - nancy_eaten = 10.0 := by
  -- Proof steps would go here
  sorry

-- Define a function to calculate the number of apples left
def calculate_apples_left : ℝ :=
  mike_apples + keith_apples - nancy_eaten

-- Theorem to show that the calculation is correct
theorem calculation_correct : 
  calculate_apples_left = 10.0 := by
  -- Unfold the definition of calculate_apples_left
  unfold calculate_apples_left
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_left_calculation_correct_l790_79072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_speed_is_8_l790_79049

/-- Represents the chase scenario between a policeman and a thief -/
structure ChaseScenario where
  initial_distance : ℝ  -- in meters
  policeman_speed : ℝ   -- in km/hr
  thief_distance : ℝ    -- in meters

/-- Calculates the speed of the thief given a chase scenario -/
noncomputable def thief_speed (scenario : ChaseScenario) : ℝ :=
  (scenario.thief_distance / 1000) / 
  ((scenario.initial_distance + scenario.thief_distance) / 1000 / scenario.policeman_speed)

/-- Theorem stating that given the specific chase scenario, the thief's speed is 8 km/hr -/
theorem thief_speed_is_8 : 
  let scenario : ChaseScenario := {
    initial_distance := 150,
    policeman_speed := 10,
    thief_distance := 600
  }
  thief_speed scenario = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_speed_is_8_l790_79049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_irreducible_fractions_l790_79009

def product_2_to_10 : ℕ := 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10

def is_irreducible (n d : ℕ) : Prop := Nat.Coprime n d

def satisfies_condition (n d : ℕ) : Prop :=
  is_irreducible n d ∧ n * d = product_2_to_10

theorem count_irreducible_fractions :
  ∃ (fractions : Finset (ℕ × ℕ)), 
    (∀ (f : ℕ × ℕ), f ∈ fractions ↔ satisfies_condition f.1 f.2) ∧
    fractions.card = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_irreducible_fractions_l790_79009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_principal_correct_l790_79004

/-- The initial principal amount that grows to 4410 after 4 years with given interest rates -/
noncomputable def initial_principal : ℝ :=
  4410 / ((1 + 0.07 / 4)^(4 * 2) * (1 + 0.09 / 2)^(2 * 2))

/-- The final amount after 4 years given the initial principal and interest rates -/
noncomputable def final_amount (P : ℝ) : ℝ :=
  P * (1 + 0.07 / 4)^(4 * 2) * (1 + 0.09 / 2)^(2 * 2)

theorem initial_principal_correct :
  ‖final_amount initial_principal - 4410‖ < 0.01 ∧
  ‖initial_principal - 3238.78‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_principal_correct_l790_79004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l790_79044

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + Real.sin (ω * x) ^ 2 - 1 / 2

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_properties (ω : ℝ) (h : has_period (f · ω) π) :
  (∀ x, f x 1 = Real.sin (2 * x - π / 6)) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), ∀ y ∈ Set.Icc 0 (π / 2), f x 1 ≥ f y 1) ∧
  (f 0 1 = -1 / 2) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f x 1 = 1) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l790_79044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l790_79003

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  3 * Real.sqrt x + 2 * (y ^ (1/3)) + 1 / (x * y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l790_79003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_drop_distance_l790_79056

/-- The horizontal distance at which an airplane should drop a package to hit a boat --/
noncomputable def drop_distance (v V h g : ℝ) : ℝ × ℝ × ℝ :=
  let t := Real.sqrt (2 * h / g)
  ((V + v) * t, (V - v) * t, V * t)

/-- Theorem stating the correct drop distances for different scenarios --/
theorem correct_drop_distance (v V h g : ℝ) (h_pos : h > 0) (g_pos : g > 0) :
  drop_distance v V h g =
    ((V + v) * Real.sqrt (2 * h / g),
     (V - v) * Real.sqrt (2 * h / g),
     V * Real.sqrt (2 * h / g)) := by
  sorry

#check correct_drop_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_drop_distance_l790_79056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_cube_plus_seven_not_prime_l790_79011

theorem prime_cube_plus_seven_not_prime (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^3 + 5)) :
  ¬Nat.Prime (P^3 + 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_cube_plus_seven_not_prime_l790_79011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_l790_79016

theorem square_difference (x y : ℝ) : 
  x = 10^5 - 10^(-5 : ℝ) → y = 10^5 + 10^(-5 : ℝ) → x^2 - y^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_l790_79016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_primes_in_sequence_l790_79083

def sequence_next (pn pm : Nat) : Nat :=
  (pn + pm + 2018).factors.foldl max 1

def is_valid_sequence (s : ℕ → ℕ) : Prop :=
  Nat.Prime (s 0) ∧ Nat.Prime (s 1) ∧
  ∀ n, n ≥ 1 → s (n + 2) = sequence_next (s n) (s (n + 1))

theorem finite_primes_in_sequence :
  ∀ s : ℕ → ℕ, is_valid_sequence s →
  ∃ N : ℕ, ∀ n ≥ N, ¬Nat.Prime (s n) :=
sorry

#check finite_primes_in_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_primes_in_sequence_l790_79083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_scooter_gain_percent_l790_79070

/-- Calculates the gain percent on a scooter sale given the purchase details and selling price --/
noncomputable def scooter_gain_percent (purchase_price : ℝ) (repair_cost : ℝ) (tire_cost : ℝ) (paint_cost : ℝ) (depreciation_rate : ℝ) (years : ℕ) (selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost + tire_cost + paint_cost
  let gain := selling_price - total_cost
  (gain / total_cost) * 100

/-- The gain percent on Sandy's scooter sale is approximately 4.35% --/
theorem sandy_scooter_gain_percent :
  let purchase_price : ℝ := 800
  let repair_cost : ℝ := 200
  let tire_cost : ℝ := 50
  let paint_cost : ℝ := 100
  let depreciation_rate : ℝ := 0.1
  let years : ℕ := 2
  let selling_price : ℝ := 1200
  ∃ ε > 0, abs (scooter_gain_percent purchase_price repair_cost tire_cost paint_cost depreciation_rate years selling_price - 4.35) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_scooter_gain_percent_l790_79070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_concave_on_interval_f_not_concave_on_interval_l790_79034

open Real

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) := x * exp x

-- Define the first derivative of f
noncomputable def f' (x : ℝ) := exp x + x * exp x

-- Define the second derivative of f
noncomputable def f'' (x : ℝ) := (2 + x) * exp x

theorem not_concave_on_interval :
  ∀ x ∈ Set.Ioo (0 : ℝ) (π / 2), f'' x > 0 :=
by sorry

-- The main theorem stating that f is not concave on (0, π/2)
theorem f_not_concave_on_interval :
  ¬ (∀ x ∈ Set.Ioo (0 : ℝ) (π / 2), f'' x < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_concave_on_interval_f_not_concave_on_interval_l790_79034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_of_three_items_l790_79099

/-- Represents a luxury item with its listed price and discount rates -/
structure LuxuryItem where
  name : String
  listedPrice : ℕ
  firstPurchaseDiscount : ℚ
  secondPurchaseDiscount : ℚ
  thirdPurchaseDiscount : ℚ

/-- Calculates the discounted price of an item based on the purchase order -/
def discountedPrice (item : LuxuryItem) (purchaseOrder : ℕ) : ℚ :=
  match purchaseOrder with
  | 1 => item.listedPrice * (1 - item.firstPurchaseDiscount)
  | 2 => item.listedPrice * (1 - item.secondPurchaseDiscount)
  | 3 => item.listedPrice * (1 - item.thirdPurchaseDiscount)
  | _ => item.listedPrice

/-- Theorem stating that the total cost of purchasing the three items in order is 131,000 rs. -/
theorem total_cost_of_three_items (watch necklace handbag : LuxuryItem) :
  watch.name = "Watch" ∧
  watch.listedPrice = 50000 ∧
  watch.firstPurchaseDiscount = 12/100 ∧
  watch.secondPurchaseDiscount = 15/100 ∧
  watch.thirdPurchaseDiscount = 20/100 ∧
  necklace.name = "Necklace" ∧
  necklace.listedPrice = 75000 ∧
  necklace.firstPurchaseDiscount = 15/100 ∧
  necklace.secondPurchaseDiscount = 24/100 ∧
  necklace.thirdPurchaseDiscount = 28/100 ∧
  handbag.name = "Handbag" ∧
  handbag.listedPrice = 40000 ∧
  handbag.firstPurchaseDiscount = 10/100 ∧
  handbag.secondPurchaseDiscount = 18/100 ∧
  handbag.thirdPurchaseDiscount = 25/100 →
  (discountedPrice watch 1 + discountedPrice necklace 2 + discountedPrice handbag 3).ceil = 131000 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_of_three_items_l790_79099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l790_79046

-- Define the coefficients of the quadratic equation
noncomputable def a : ℝ := 5 + 3 * Real.sqrt 2
noncomputable def b : ℝ := 3 + Real.sqrt 2
def c : ℝ := -1

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the quadratic equation
noncomputable def root1 : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
noncomputable def root2 : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)

-- State the theorem
theorem root_difference : |root1 - root2| = 2 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l790_79046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_l790_79029

/-- The differential equation -/
def diff_eq (x y dx dy : ℝ) : Prop :=
  (3 * x + 2 * y + y^2) * dx + (x + 4 * x * y + 5 * y^2) * dy = 0

/-- The integrating factor -/
noncomputable def integrating_factor (x y : ℝ) : ℝ :=
  x + y^2

/-- The solution function -/
def solution (x y C : ℝ) : Prop :=
  (x + y^2)^3 = C

/-- Theorem stating that the solution satisfies the differential equation -/
theorem solution_satisfies_diff_eq (x y C : ℝ) :
  solution x y C → ∃ (dx dy : ℝ), diff_eq x y dx dy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_l790_79029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_f_less_than_two_l790_79089

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x|

-- Theorem for part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 2} = Set.Iic (-1/2) ∪ Set.Ici (3/2) := by sorry

-- Theorem for part II
theorem range_of_a_for_f_less_than_two :
  {a : ℝ | ∃ x, f a x < 2} = Set.Ioo (-2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_f_less_than_two_l790_79089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_dividing_pairs_l790_79010

def coprime_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 212 * coprime_sequence (n + 1) - coprime_sequence n

def is_solution (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ x ∣ (y^2 + 210) ∧ y ∣ (x^2 + 210)

theorem coprime_dividing_pairs (x y : ℕ) :
  is_solution x y →
  ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) ∨
   (∃ n : ℕ, (x = coprime_sequence n ∧ y = coprime_sequence (n + 1)) ∨
             (x = coprime_sequence (n + 1) ∧ y = coprime_sequence n))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_dividing_pairs_l790_79010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_negative_five_halves_l790_79001

/-- The projection matrix --/
def P : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4/29, -10/29; -10/29, 25/29]

/-- The vector onto which P projects --/
def v : Fin 2 → ℚ := sorry

/-- P projects onto v --/
axiom proj_property : P.mulVec v = v

/-- The ratio y/x equals -5/2 --/
theorem ratio_is_negative_five_halves :
  v 1 / v 0 = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_negative_five_halves_l790_79001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l790_79093

/-- The equation of a line passing through two points -/
def line_equation (x1 y1 x2 y2 : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

/-- The point (-2, 1) lies on the line -/
def point1_on_line (f : ℝ → ℝ → Prop) : Prop :=
  f (-2) 1

/-- The point (3, -3) lies on the line -/
def point2_on_line (f : ℝ → ℝ → Prop) : Prop :=
  f 3 (-3)

/-- The equation 4x + 5y + 3 = 0 represents a line -/
def target_equation : ℝ → ℝ → Prop :=
  fun x y ↦ 4 * x + 5 * y + 3 = 0

theorem line_through_points :
  ∀ f : ℝ → ℝ → Prop,
  (f = line_equation (-2) 1 3 (-3)) →
  point1_on_line f →
  point2_on_line f →
  f = target_equation := by
  sorry

#check line_through_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l790_79093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l790_79052

-- Define the constants
def cylinder_radius : ℝ := 15
def cylinder_height : ℝ := 30
def cone_radius : ℝ := 15
def cone_height : ℝ := 15
def sphere_radius : ℝ := 8

-- Define the volumes
noncomputable def cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * cylinder_height
noncomputable def cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height
noncomputable def sphere_volume : ℝ := (4/3) * Real.pi * sphere_radius^3

-- Theorem statement
theorem unoccupied_volume :
  cylinder_volume - 2 * cone_volume - sphere_volume = 3817.33 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l790_79052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l790_79007

/-- The minimum distance from the origin to a point on the line 3x - 4y - 10 = 0 is 2 -/
theorem min_distance_to_line : ∀ x₀ y₀ : ℝ, 
  (3 * x₀ - 4 * y₀ - 10 = 0) → 
  (∀ x y : ℝ, (3 * x - 4 * y - 10 = 0) → (x₀^2 + y₀^2 ≤ x^2 + y^2)) →
  Real.sqrt (x₀^2 + y₀^2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l790_79007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ADG_l790_79000

/-- A regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Triangle ADG in the regular octagon -/
structure TriangleADG (octagon : RegularOctagon) :=
  (A : ℝ × ℝ)
  (D : ℝ × ℝ)
  (G : ℝ × ℝ)

/-- Helper function to calculate the area of a triangle given its vertices -/
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- The area of triangle ADG in a regular octagon with side length 3 is 9√2 -/
theorem area_triangle_ADG (octagon : RegularOctagon) (triangle : TriangleADG octagon) :
  area_triangle triangle.A triangle.D triangle.G = 9 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ADG_l790_79000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_five_l790_79053

/-- Given two points A and B in polar coordinates, prove that the length of AB is 5 -/
theorem length_AB_is_five (A B : ℝ × ℝ) : 
  A = (4, 1) → B = (3, 1 + π/2) → 
  Real.sqrt ((A.1 * Real.cos A.2 - B.1 * Real.cos B.2)^2 + (A.1 * Real.sin A.2 - B.1 * Real.sin B.2)^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_five_l790_79053
