import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l162_16216

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := log (cos (2 * x + π / 4))

-- State the theorem
theorem f_monotone_decreasing : 
  ∀ a b : ℝ, -π/8 < a ∧ a < b ∧ b < π/8 → 
    ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x :=
by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l162_16216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_inequality_l162_16219

theorem min_inequality (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  min
    (min ((10*a^2 - 5*a + 1) / (b^2 - 5*b + 10))
         ((10*b^2 - 5*b + 1) / (c^2 - 5*c + 10)))
    ((10*c^2 - 5*c + 1) / (a^2 - 5*a + 10))
  ≤ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_inequality_l162_16219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_a_range_l162_16295

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - (9/2) * x^2 + 6*x - a

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 9*x + 6

-- Theorem for the maximum value of m
theorem max_m_value :
  (∃ m : ℝ, ∀ x : ℝ, f' x ≥ m) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f' x ≥ m) → m ≤ -3/4) := by
  sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∃! x : ℝ, f a x = 0) ↔ (a < 2 ∨ a > 5/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_a_range_l162_16295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l162_16258

theorem triangle_cosine_value (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = c * Real.sin A ∧ b = c * Real.sin B →
  S = (1/2) * a * b * Real.sin C →
  (a^2 + b^2) * Real.tan C = 8 * S →
  Real.sin A * Real.cos B = 2 * Real.cos A * Real.sin B →
  Real.cos A = Real.sqrt 30 / 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l162_16258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_properties_l162_16291

noncomputable def parabola_point (x y : ℝ) : ℝ × ℝ := (x, y)

def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def on_parabola (p : ℝ × ℝ) : Prop :=
  p.1^2 = 8 * p.2

theorem parabola_point_properties :
  let V : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (0, 2)
  let P : ℝ × ℝ := parabola_point (Real.sqrt 464) 58
  is_in_first_quadrant P ∧
  on_parabola P ∧
  distance P F = 60 := by
  sorry

#check parabola_point_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_properties_l162_16291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l162_16270

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x + 2*x - 1 else -(2^(-x) + 2*(-x) - 1)

-- State the theorem
theorem odd_function_value : 
  (∀ x, f (-x) = -(f x)) →  -- f is odd
  (∀ x ≥ 0, f x = 2^x + 2*x - 1) →  -- f(x) = 2^x + 2x - 1 for x ≥ 0
  f (-1) = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l162_16270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_exists_sequence_bound_value_l162_16208

noncomputable def a : ℕ → ℝ
| 0 => 1/5
| n + 1 => 3 * (a n)^2 + 1/2

noncomputable def prod : ℕ → ℝ
| 0 => 1
| n + 1 => prod n * a n

def bound (d : ℝ) : Prop :=
  ∀ n : ℕ, |prod n| ≤ d / 3^n

theorem sequence_bound_exists :
  ∃ d : ℝ, bound d ∧ ∀ d', bound d' → d ≤ d' :=
sorry

theorem sequence_bound_value (d : ℝ) 
  (h : bound d ∧ ∀ d', bound d' → d ≤ d') :
  107.5 < 100 * d ∧ 100 * d < 108.5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_exists_sequence_bound_value_l162_16208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_in_arithmetic_progression_l162_16204

theorem polynomial_roots_in_arithmetic_progression :
  let p (x : ℝ) := x^3 - 3/2 * x^2 - 1/4 * x + 3/8
  let roots : Set ℝ := {-1/2, 1/2, 3/2}
  (∀ r, r ∈ roots → p r = 0) ∧ 
  (∃ d : ℝ, ∀ x y z, x ∈ roots → y ∈ roots → z ∈ roots → x < y → y < z → y - x = z - y) →
  roots = {r : ℝ | p r = 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_in_arithmetic_progression_l162_16204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_average_speed_l162_16273

-- Define the segments of Jane's journey
noncomputable def segment1_distance : ℝ := 40
noncomputable def segment1_speed : ℝ := 8
noncomputable def segment2_distance : ℝ := 20
noncomputable def segment2_speed : ℝ := 40
noncomputable def segment3_distance : ℝ := 10
noncomputable def segment3_speed : ℝ := 20

-- Define the total distance
noncomputable def total_distance : ℝ := segment1_distance + segment2_distance + segment3_distance

-- Define the time for each segment
noncomputable def segment1_time : ℝ := segment1_distance / segment1_speed
noncomputable def segment2_time : ℝ := segment2_distance / segment2_speed
noncomputable def segment3_time : ℝ := segment3_distance / segment3_speed

-- Define the total time
noncomputable def total_time : ℝ := segment1_time + segment2_time + segment3_time

-- Define the average speed
noncomputable def average_speed : ℝ := total_distance / total_time

-- Theorem statement
theorem jane_average_speed : 
  abs (average_speed - 35/3) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_average_speed_l162_16273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_g_3_l162_16299

-- Define the functions u and g
noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - u x

-- State the theorem
theorem u_of_g_3 : u (g 3) = Real.sqrt (37 - 5 * Real.sqrt 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_g_3_l162_16299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_l162_16272

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  let rem := x % 5
  if rem < 3 then x - rem else x + (5 - rem)

def sum_rounded_to_five (n : ℕ) : ℕ :=
  (List.range n).map (λ x => round_to_nearest_five (x + 1)) |> List.sum

theorem difference_of_sums (n : ℕ) : n = 60 →
  (sum_to_n n : ℤ) - (sum_rounded_to_five n : ℤ) = 1560 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_l162_16272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l162_16288

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + cos (2 * x + π / 6)

theorem f_properties :
  -- 1. Maximum value is √2
  (∀ x, f x ≤ sqrt 2) ∧
  (∃ x, f x = sqrt 2) ∧
  -- 2. Smallest positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ p, 0 < p → p < π → ∃ x, f (x + p) ≠ f x) ∧
  -- 3. Decreasing in the interval (π/24, 13π/24)
  (∀ x y, π/24 < x → x < y → y < 13*π/24 → f y < f x) ∧
  -- 4. Coincides with √2*cos(2x) translated π/24 units right
  (∀ x, f x = sqrt 2 * cos (2 * (x - π/24))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l162_16288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_representation_twenty_cm_representation_l162_16214

noncomputable def map_scale (cm : ℝ) (km : ℝ) : ℝ := km / cm

theorem map_representation (cm1 cm2 km1 : ℝ) (h : cm1 > 0) :
  let scale := map_scale cm1 km1
  km1 * cm2 / cm1 = scale * cm2 := by
  sorry

theorem twenty_cm_representation :
  let cm1 : ℝ := 15
  let cm2 : ℝ := 20
  let km1 : ℝ := 90
  map_scale cm1 km1 * cm2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_representation_twenty_cm_representation_l162_16214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_polynomial_l162_16239

theorem root_of_polynomial : ∃ (p : Polynomial ℚ), 
  Polynomial.Monic p ∧ 
  Polynomial.degree p = 4 ∧ 
  (∀ x : ℝ, x = Real.sqrt 3 + Real.sqrt 5 → Polynomial.eval₂ (algebraMap ℚ ℝ) x p = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_polynomial_l162_16239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parameter_l162_16225

/-- The quadratic equation ax^2 + 2x + 1 = 0 -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + 2*x + 1 = 0

/-- The discriminant of the quadratic equation -/
noncomputable def discriminant (a : ℝ) : ℝ :=
  2^2 - 4*a*1

/-- The square of the difference of the roots -/
noncomputable def root_difference_squared (a : ℝ) : ℝ :=
  (2/a)^2 - 4*(1/a)

/-- The condition that the discriminant is 9 times the square of the difference of the roots -/
def discriminant_condition (a : ℝ) : Prop :=
  discriminant a = 9 * root_difference_squared a

theorem unique_parameter : 
  ∃! a : ℝ, discriminant_condition a ∧ a = -3 := by
  sorry

#check unique_parameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parameter_l162_16225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l162_16280

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define vectors m and n
noncomputable def m (a : ℝ) : ℝ × ℝ := (a, 1/2)
noncomputable def n (C b c : ℝ) : ℝ × ℝ := (Real.cos C, c - 2*b)

-- State the perpendicularity condition
axiom m_perp_n (a b c C : ℝ) : 
  (m a).1 * (n C b c).1 + (m a).2 * (n C b c).2 = 0

-- State the theorem
theorem triangle_properties :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ -- Angle constraints
  a > 0 ∧ b > 0 ∧ c > 0 ∧ -- Side length constraints
  A + B + C = π ∧ -- Angle sum in a triangle
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C → -- Sine law
  (A = π/3 ∧ 
   (a = 1 → ∀ l, l = a + b + c → 2 < l ∧ l ≤ 3)) := 
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l162_16280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_f_l162_16207

/-- Given the equation 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * f)) = 3200.0000000000005,
    prove that f is approximately equal to 1.25 -/
theorem value_of_f (f : ℝ) :
  4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * f)) = 3200.0000000000005 →
  ∃ ε > 0, |f - 1.25| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_f_l162_16207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonic_increase_interval_l162_16218

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 6))

theorem g_monotonic_increase_interval (k : ℤ) :
  StrictMonoOn g (Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonic_increase_interval_l162_16218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l162_16275

theorem membership_change (initial_members : ℝ) : 
  initial_members > 0 → 
  let fall_members := initial_members * 1.09
  let spring_members := fall_members * 0.81
  let percentage_change := (spring_members - initial_members) / initial_members * 100
  abs (percentage_change + 11.71) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l162_16275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l162_16236

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

-- Define the tangent line function
def tangent_line (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

-- State the theorem
theorem tangent_line_b_value :
  ∃ (x₀ : ℝ), (∀ x, f x ≤ tangent_line 1 x) ∧ (f x₀ = tangent_line 1 x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l162_16236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l162_16247

/-- Rotation about x-axis by 90 degrees --/
def rotateX90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.fst, -p.snd.snd, p.snd.fst)

/-- Reflection through xz-plane --/
def reflectXZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.fst, -p.snd.fst, p.snd.snd)

/-- Reflection through yz-plane --/
def reflectYZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.fst, p.snd.fst, p.snd.snd)

/-- The sequence of transformations --/
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflectYZ (rotateX90 (reflectXZ (rotateX90 p)))

theorem point_transformation :
  transform (1, 2, 3) = (-1, -2, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l162_16247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hua_luogeng_optimal_selection_uses_golden_ratio_l162_16253

/-- The optimal selection method as popularized by Hua Luogeng -/
structure OptimalSelectionMethod where
  /-- The mathematician who popularized the method -/
  popularizer : String
  /-- The mathematical concept used in the method -/
  concept : String

/-- Theorem stating that Hua Luogeng's optimal selection method uses the golden ratio -/
theorem hua_luogeng_optimal_selection_uses_golden_ratio :
  ∃ (method : OptimalSelectionMethod),
    method.popularizer = "Hua Luogeng" ∧
    method.concept = "golden ratio" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hua_luogeng_optimal_selection_uses_golden_ratio_l162_16253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l162_16240

/-- Represents the profit distribution in a partnership business --/
structure PartnershipProfit where
  investment_a : ℚ
  investment_b : ℚ
  period_a : ℚ
  period_b : ℚ
  profit_b : ℚ

/-- Calculates the total profit of the partnership --/
def total_profit (p : PartnershipProfit) : ℚ :=
  p.profit_b * (1 + (p.investment_a * p.period_a) / (p.investment_b * p.period_b))

/-- Theorem stating the total profit of the partnership --/
theorem partnership_profit (p : PartnershipProfit) 
  (h1 : p.investment_a = 3 * p.investment_b)
  (h2 : p.period_a = 2 * p.period_b)
  (h3 : p.profit_b = 5000) :
  total_profit p = 35000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l162_16240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_l162_16256

-- Define the point M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; ∀ (x' y' : ℝ), |x' - 4| = 2 * Real.sqrt ((x' - 1)^2 + y'^2) → x = x' ∧ y = y'}

-- Define the trajectory C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; (x^2 / 4) + (y^2 / 3) = 1}

-- Define point P
def P : ℝ × ℝ := (0, 3)

-- Define the line m
def m (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; y = k * x + 3}

-- State the theorem
theorem trajectory_and_slope :
  (M ⊆ C) ∧
  (∃ (k : ℝ), k = 3/2 ∨ k = -3/2) ∧
  (∀ (k : ℝ), k = 3/2 ∨ k = -3/2 →
    ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ∈ m k ∧ B ∈ m k ∧
    let (xa, ya) := A
    let (xb, yb) := B
    2 * xa = xb ∧ 2 * ya = 3 + yb) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_l162_16256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l162_16260

noncomputable def f (α : Real) : Real :=
  (Real.sin (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (3 * Real.pi / 2 - α)) /
  (Real.cos (Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = Real.cos α := by
  sorry

theorem f_specific_value (α : Real) 
  (h1 : Real.pi / 2 < α ∧ α < Real.pi)  -- α is in the second quadrant
  (h2 : Real.cos (Real.pi / 2 + α) = -1/3) :
  f α = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l162_16260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l162_16254

/-- Represents a rectangular yard with flower beds -/
structure YardWithFlowerBeds where
  length : ℝ
  height : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- Calculates the area of an isosceles right triangle -/
noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg^2

/-- Calculates the total area of two congruent isosceles right triangles -/
noncomputable def flower_beds_area (yard : YardWithFlowerBeds) : ℝ :=
  2 * isosceles_right_triangle_area ((yard.trapezoid_long_side - yard.trapezoid_short_side) / 2)

/-- Calculates the total area of the rectangular yard -/
noncomputable def total_yard_area (yard : YardWithFlowerBeds) : ℝ :=
  yard.length * yard.height

/-- Theorem: The fraction of the yard occupied by flower beds is 5/16 -/
theorem flower_beds_fraction (yard : YardWithFlowerBeds) 
  (h1 : yard.length = 30)
  (h2 : yard.height = 6)
  (h3 : yard.trapezoid_short_side = 20)
  (h4 : yard.trapezoid_long_side = 35) :
  flower_beds_area yard / total_yard_area yard = 5 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l162_16254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_completes_in_40_days_l162_16223

/-- The number of days it takes for B to complete the job alone, given that:
    1. A can complete the job in 30 days
    2. A and B working together for 10 days complete 0.5833333333333334 of the work -/
noncomputable def days_for_B : ℝ :=
  let a_rate : ℝ := 1 / 30
  let combined_work : ℝ := 0.5833333333333334
  let combined_days : ℝ := 10
  (combined_days * combined_work - combined_days * a_rate) / (combined_work - combined_days * a_rate)

theorem B_completes_in_40_days : days_for_B = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_completes_in_40_days_l162_16223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minus_alpha_value_l162_16209

noncomputable def f (ω : ℕ) (x : ℝ) : ℝ := 
  (1/2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x + Real.pi/2)

noncomputable def g (ω : ℕ) (x : ℝ) : ℝ := f ω (x + Real.pi/4)

theorem g_minus_alpha_value (ω : ℕ) (α : ℝ) :
  (ω > 0) →
  (∃ (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 3*Real.pi/2 ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ f ω x₃ = 0 ∧
    ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3*Real.pi/2 → (f ω x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  g ω (α/2) = 4/5 →
  g ω (-α) = -7/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minus_alpha_value_l162_16209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_argument_equals_17pi_over_36_l162_16266

noncomputable def complex_exp (x : ℝ) : ℂ := Complex.exp (Complex.I * x)

theorem sum_argument_equals_17pi_over_36 :
  let sum := complex_exp (5 * π / 36) + complex_exp (11 * π / 36) + 
             complex_exp (17 * π / 36) + complex_exp (23 * π / 36) + 
             complex_exp (29 * π / 36)
  Complex.arg sum = 17 * π / 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_argument_equals_17pi_over_36_l162_16266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario_2_is_hypergeometric_l162_16222

/-- A population with two types of items -/
structure Population :=
  (total : ℕ)
  (success : ℕ)
  (failure : ℕ)
  (h_total : total = success + failure)

/-- Parameters for a hypergeometric distribution -/
structure HypergeometricParams :=
  (pop : Population)
  (sample_size : ℕ)
  (h_sample : sample_size ≤ pop.total)

/-- Definition of a hypergeometric distribution -/
def is_hypergeometric (X : ℕ → ℝ) (params : HypergeometricParams) : Prop :=
  ∀ k : ℕ, 
    X k = (Nat.choose params.pop.success k * Nat.choose (params.pop.total - params.pop.success) (params.sample_size - k)) / 
           Nat.choose params.pop.total params.sample_size

/-- The scenario described in option ② -/
def scenario_2 : HypergeometricParams :=
  { pop := { total := 10, success := 3, failure := 7, h_total := by rfl },
    sample_size := 5,
    h_sample := by norm_num }

/-- Theorem stating that the random variable X in scenario ② follows a hypergeometric distribution -/
theorem scenario_2_is_hypergeometric :
  ∃ X : ℕ → ℝ, is_hypergeometric X scenario_2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario_2_is_hypergeometric_l162_16222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_rate_l162_16284

-- Define the total investment
noncomputable def total_investment : ℝ := 10000

-- Define the total interest for 1 year
noncomputable def total_interest : ℝ := 840

-- Define the amount invested at 8%
noncomputable def investment_at_8_percent : ℝ := 6000

-- Define the known interest rate (8%)
noncomputable def known_rate : ℝ := 0.08

-- Define the function to calculate the interest
noncomputable def calculate_interest (principal : ℝ) (rate : ℝ) : ℝ := principal * rate

-- Define the function to calculate the unknown rate
noncomputable def calculate_unknown_rate (total_investment : ℝ) (investment_at_known_rate : ℝ) 
  (total_interest : ℝ) (known_rate : ℝ) : ℝ :=
  let remaining_investment := total_investment - investment_at_known_rate
  let remaining_interest := total_interest - calculate_interest investment_at_known_rate known_rate
  remaining_interest / remaining_investment

-- Theorem statement
theorem second_investment_rate :
  calculate_unknown_rate total_investment investment_at_8_percent total_interest known_rate = 0.09 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_rate_l162_16284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l162_16203

/-- Given a sector with perimeter 30, prove that its area is maximized when
    the radius is 15/2 and the central angle is 2, with a maximum area of 225/4. -/
theorem sector_max_area (R α : ℝ) (S : ℝ → ℝ → ℝ) :
  (∀ r a, S r a ≤ S (15/2) 2) ∧
  S (15/2) 2 = 225/4 ∧
  2 * R + R * α = 30 →
  R = 15/2 ∧ α = 2 := by
  sorry

#check sector_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l162_16203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l162_16257

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 5 * Real.sin θ

-- Define the line l in parametric form
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 - Real.sqrt 2 / 2 * t, Real.sqrt 5 + Real.sqrt 2 / 2 * t)

-- Define point P
noncomputable def point_P : ℝ × ℝ := (3, Real.sqrt 5)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (M N : ℝ × ℝ),
    (∃ (θ₁ θ₂ : ℝ), (curve_C θ₁ * Real.cos θ₁, curve_C θ₁ * Real.sin θ₁) = M ∧
                     (curve_C θ₂ * Real.cos θ₂, curve_C θ₂ * Real.sin θ₂) = N) ∧
    (∃ (t₁ t₂ : ℝ), line_l t₁ = M ∧ line_l t₂ = N) ∧
    Real.sqrt ((point_P.1 - M.1)^2 + (point_P.2 - M.2)^2) +
    Real.sqrt ((point_P.1 - N.1)^2 + (point_P.2 - N.2)^2) = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l162_16257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_integers_l162_16243

def first_twelve_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

theorem median_of_first_twelve_integers :
  let sorted_list := first_twelve_integers
  let n := sorted_list.length
  let middle_index := n / 2
  (sorted_list[middle_index - 1]! + sorted_list[middle_index]!) / 2 = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_integers_l162_16243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_months_approximately_nine_l162_16271

/-- Represents the pasture rental scenario -/
structure PastureRental where
  totalCost : ℕ
  aHorses : ℕ
  aMonths : ℕ
  bHorses : ℕ
  bPayment : ℕ
  cHorses : ℕ
  cMonths : ℕ

/-- Calculates the number of months b put in the horses -/
def calculateBMonths (rental : PastureRental) : ℚ :=
  sorry

/-- Theorem stating that b put in the horses for approximately 9 months -/
theorem b_months_approximately_nine (rental : PastureRental) 
  (h1 : rental.totalCost = 841)
  (h2 : rental.aHorses = 12)
  (h3 : rental.aMonths = 8)
  (h4 : rental.bHorses = 16)
  (h5 : rental.bPayment = 348)
  (h6 : rental.cHorses = 18)
  (h7 : rental.cMonths = 6) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |calculateBMonths rental - 9| < ε :=
sorry

#check b_months_approximately_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_months_approximately_nine_l162_16271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l162_16210

noncomputable def f (x : ℝ) := 1 - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (f (π / 6) = 1 / 2) ∧
  (∀ x ∈ Set.Icc (-π / 4) (π / 6), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc (-π / 4) (π / 6), f x ≥ 0) ∧
  (∃ x ∈ Set.Icc (-π / 4) (π / 6), f x = 1) ∧
  (∃ x ∈ Set.Icc (-π / 4) (π / 6), f x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l162_16210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l162_16211

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Define the set of m values that satisfy the conditions
def m_range : Set ℝ := {m | p m ∧ ¬(q m)}

-- State the theorem
theorem m_range_theorem : m_range = Set.Iic (-2) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l162_16211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_for_achievable_goal_l162_16248

/-- Represents a gymnast's nationality -/
inductive Nationality
| Swiss
| Liechtensteiner

/-- Represents a position on the grid -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents the state of the grid -/
structure GridState where
  n : ℕ
  k : ℕ
  swiss_positions : List Position
  liechtensteiner_positions : List Position

/-- Represents a valid move for a gymnast -/
inductive ValidMove : Nationality → Position → Position → Prop
| swiss_right : ∀ x y, ValidMove Nationality.Swiss ⟨x, y⟩ ⟨x + 1, y⟩
| swiss_up : ∀ x y, ValidMove Nationality.Swiss ⟨x, y⟩ ⟨x, y + 1⟩
| liechtensteiner_left : ∀ x y, ValidMove Nationality.Liechtensteiner ⟨x, y⟩ ⟨x - 1, y⟩
| liechtensteiner_down : ∀ x y, ValidMove Nationality.Liechtensteiner ⟨x, y⟩ ⟨x, y - 1⟩

/-- Represents the goal state -/
def GoalState (gs : GridState) : Prop :=
  gs.swiss_positions.all (fun p => p.x = gs.n - 1 ∧ p.y = gs.n - 1) ∧
  gs.liechtensteiner_positions.all (fun p => p.x = 0 ∧ p.y = 0)

/-- Represents a valid sequence of moves -/
def ValidMoveSequence (gs : GridState) : List (Nationality × Position × Position) → Prop :=
  sorry

/-- Apply moves to a grid state -/
def applyMoves (gs : GridState) (moves : List (Nationality × Position × Position)) : GridState :=
  sorry

/-- The main theorem -/
theorem largest_k_for_achievable_goal (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, k = (n - 1)^2 ∧
    (∀ gs : GridState, gs.n = n → gs.k = k →
      ∃ moves, ValidMoveSequence gs moves ∧ GoalState (applyMoves gs moves)) ∧
    (∀ k' : ℕ, k' > k →
      ¬∃ gs : GridState, gs.n = n ∧ gs.k = k' ∧
        ∃ moves, ValidMoveSequence gs moves ∧ GoalState (applyMoves gs moves)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_for_achievable_goal_l162_16248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_l162_16278

def salary_distribution : List (ℕ × ℕ × ℕ) := [
  (1, 140000, 1),
  (7, 95000, 2),
  (8, 80000, 3),
  (4, 55000, 4),
  (43, 25000, 5)
]

def total_employees : ℕ := 63

theorem median_salary (
  h1 : (salary_distribution.map (λ x => x.1)).sum = total_employees
) : ∃ (median : ℕ), median = 25000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_l162_16278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_orthocenter_l162_16200

-- Define the angle measure
noncomputable def angle (A O B : ℝ × ℝ) : ℝ := sorry

-- Define the triangle OAB
structure Triangle (O A B : ℝ × ℝ) : Prop where
  angle_less_90 : angle A O B < 90

-- Define a point in the triangle
def PointInTriangle (O A B M : ℝ × ℝ) : Prop :=
  ∃ t u : ℝ, t ≥ 0 ∧ u ≥ 0 ∧ t + u ≤ 1 ∧
    M = (t * A.1 + u * B.1 + (1 - t - u) * O.1, t * A.2 + u * B.2 + (1 - t - u) * O.2)

-- Define a line
def Line (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ t : ℝ, P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)}

-- Define perpendicularity
def Perpendicular (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Define the orthocenter
def Orthocenter (H P Q R : ℝ × ℝ) : Prop :=
  Perpendicular (Line H P) (Line Q R) ∧
  Perpendicular (Line H Q) (Line P R) ∧
  Perpendicular (Line H R) (Line P Q)

-- Define the interior of a triangle
def Interior (T : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem locus_of_orthocenter 
  (O A B M P Q H C D : ℝ × ℝ) 
  (triangle : Triangle O A B) 
  (m_in_triangle : PointInTriangle O A B M)
  (perp_mp : Perpendicular (Line M P) (Line O A))
  (perp_mq : Perpendicular (Line M Q) (Line O B))
  (h_orthocenter : Orthocenter H O P Q)
  (c_on_ob : C ∈ Line O B)
  (d_on_oa : D ∈ Line O A)
  (perp_ac : Perpendicular (Line A C) (Line O B))
  (perp_bd : Perpendicular (Line B D) (Line O A)) :
  (M ∈ Line A B → H ∈ Line C D) ∧
  (M ∈ Interior {O, A, B} → H ∈ Interior {O, C, D}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_orthocenter_l162_16200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_1729728_l162_16230

theorem sum_of_distinct_prime_factors_1729728 : 
  (Finset.sum (Finset.filter (fun p => Nat.Prime p ∧ 1729728 % p = 0) (Finset.range 1729729)) id) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_1729728_l162_16230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equivalence_intersection_range_l162_16242

noncomputable section

-- Define the curve C
def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def l_polar (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Define the line l in Cartesian form
def l_cartesian (x y m : ℝ) : Prop := Real.sqrt 3 * x + y + 2 * m = 0

-- Theorem 1: Equivalence of polar and Cartesian forms of l
theorem polar_to_cartesian_equivalence (m : ℝ) :
  ∀ x y ρ θ, x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  (l_polar ρ θ m ↔ l_cartesian x y m) := by
  sorry

-- Theorem 2: Range of m for intersection of l and C
theorem intersection_range :
  ∀ m, (∃ t, l_cartesian (C t).1 (C t).2 m) ↔ -19/12 ≤ m ∧ m ≤ 5/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equivalence_intersection_range_l162_16242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l162_16237

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 500)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 45)
  : ∃ (speed : ℝ), |speed - 64| < 0.01 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l162_16237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_count_l162_16263

/-- The number of ways to assign four distinct students to three distinct classes -/
def assignment_count : ℕ := 36

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- Theorem stating that the number of ways to assign four distinct students to three distinct classes,
    where each class must have at least one student, is equal to 36 -/
theorem student_assignment_count :
  (∀ (assignment : Fin num_students → Fin num_classes),
    (∀ c : Fin num_classes, ∃ s : Fin num_students, assignment s = c) →
    assignment_count = Fintype.card {assignment : Fin num_students → Fin num_classes |
      ∀ c : Fin num_classes, ∃ s : Fin num_students, assignment s = c}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_count_l162_16263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log3_sufficient_not_necessary_for_exp2_l162_16279

-- Define the logarithm function
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_sufficient_not_necessary_for_exp2 :
  (∀ x y, x > 0 → y > 0 → (log3 x < log3 y ↔ x < y)) →  -- log3 is monotonically increasing
  (∀ a b, a > 0 → b > 0 → log3 a > log3 b → (2 : ℝ)^a > (2 : ℝ)^b) ∧  -- Sufficient condition
  ¬(∀ a b, (2 : ℝ)^a > (2 : ℝ)^b → log3 a > log3 b) :=  -- Not necessary condition
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log3_sufficient_not_necessary_for_exp2_l162_16279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l162_16289

/-- Calculates the compound interest amount -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t)

/-- Helper function to check if two real numbers are approximately equal -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  abs (x - y) < ε

notation:50 a " ≈ " b:50 => approx_equal a b 0.01

/-- The problem statement -/
theorem interest_problem (P : ℝ) : 
  (P > 0) →  -- Principal is positive
  (compound_interest P 0.04 2 12 - P = P - 340) →  -- Interest condition
  (P ≈ 852.24) :=  -- Approximation of the result
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l162_16289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_between_10_and_150_l162_16261

def is_odd (n : ℕ) : Bool := n % 2 = 1

def between (a b n : ℕ) : Bool := a < n ∧ n < b

def product_of_odds (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).filter (λ n => is_odd (n + a) && between a b (n + a))
    |>.map (λ n => n + a)
    |>.prod

theorem units_digit_of_product_between_10_and_150 :
  (product_of_odds 10 150) % 10 = 5 := by
  sorry

#eval (product_of_odds 10 150) % 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_between_10_and_150_l162_16261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_trigonometric_identity_l162_16282

theorem arithmetic_progression_trigonometric_identity 
  (α β γ : ℝ) 
  (h : β = (α + γ) / 2) : 
  (Real.sin α - Real.sin γ) / (Real.cos γ - Real.cos α) = Real.tan (π/2 - β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_trigonometric_identity_l162_16282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_theorem_l162_16281

-- Define the triangle
structure Triangle where
  side1 : ℝ → ℝ → Prop
  side2 : ℝ → ℝ → Prop
  side3 : ℝ → ℝ → Prop

-- Define the vertex B as the intersection of side1 and side2
def vertexB : ℝ × ℝ :=
  (4, 5)

-- Define the equations of median, angle bisector, and altitude
def medianEq (x y : ℝ) : Prop := 2 * x - 3 * y = -7

def angleBisectorEq (x y : ℝ) : Prop := x - 4 = (y - 5) / -15.39

def altitudeEq (x y : ℝ) : Prop := y = x + 1

-- Theorem statement
theorem triangle_lines_theorem (t : Triangle) 
  (h1 : t.side1 = fun x y => 3 * x - 4 * y + 8 = 0)
  (h2 : t.side2 = fun x y => 12 * x + 5 * y - 73 = 0)
  (h3 : t.side3 = fun x y => x + y + 12 = 0) :
  let (x, y) := vertexB
  medianEq x y ∧ angleBisectorEq x y ∧ altitudeEq x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_theorem_l162_16281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_function_below_line_l162_16221

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

-- Define the tangent line
def tangent_line (x y b : ℝ) : Prop := 2 * x + y + b = 0

-- Theorem for part 1
theorem tangent_line_values (a b : ℝ) :
  (∀ x, tangent_line x (f a x) b) → a = -1 ∧ b = 1/2 := by
  sorry

-- Theorem for part 2
theorem function_below_line (a : ℝ) :
  (∀ x, x > 1 → f a x < 2 * a * x) ↔ -1/2 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_function_below_line_l162_16221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l162_16290

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : Real.cos (2 * α) + Real.sin α * (2 * Real.sin α - 1) = 2/5)
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (α + π/4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l162_16290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_3_l162_16228

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 14 * x + 6) / (x - 3)

theorem limit_f_at_3 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ → |f x - 10| < ε ∧ δ = ε/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_3_l162_16228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_sum_l162_16206

def box1_marbles : ℕ := 18
def box2_marbles : ℕ := 12
def total_marbles : ℕ := 30
def red_prob : ℚ := 4/9

def red1 : ℕ := 12
def red2 : ℕ := 8
def blue1 : ℕ := 6
def blue2 : ℕ := 4

theorem marble_probability_sum (p q : ℕ) : 
  box1_marbles + box2_marbles = total_marbles →
  (red1 : ℚ)/box1_marbles * (red2 : ℚ)/box2_marbles = red_prob →
  (blue1 : ℚ)/box1_marbles * (blue2 : ℚ)/box2_marbles = p/q →
  red1 + blue1 = box1_marbles →
  red2 + blue2 = box2_marbles →
  Nat.Coprime p q →
  p + q = 10 := by
  sorry

#eval box1_marbles + box2_marbles
#eval (red1 : ℚ)/box1_marbles * (red2 : ℚ)/box2_marbles
#eval (blue1 : ℚ)/box1_marbles * (blue2 : ℚ)/box2_marbles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_sum_l162_16206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_diameter_endpoints_l162_16259

-- Define the points C and D
def C : ℝ × ℝ := (2, 3)
def D : ℝ × ℝ := (8, 9)

-- Define the diameter as the distance between C and D
noncomputable def diameter : ℝ := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

-- Define the radius as half the diameter
noncomputable def radius : ℝ := diameter / 2

-- State the theorem
theorem circle_area_with_diameter_endpoints (h : C = (2, 3) ∧ D = (8, 9)) : 
  π * radius^2 = 18 * π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_diameter_endpoints_l162_16259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_a_less_than_6_l162_16226

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - (a/2 + 3)*x^2 + 2*a*x + 3

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - (a + 6)*x + 2*a

theorem local_minimum_implies_a_less_than_6 (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2) →
  a < 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_a_less_than_6_l162_16226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiled_floor_area_l162_16213

-- Define the tile dimensions in centimeters
noncomputable def tile_width : ℚ := 30
noncomputable def tile_length : ℚ := 30

-- Define the number of rows and tiles per row
def num_rows : ℕ := 5
def tiles_per_row : ℕ := 8

-- Define the conversion factor from cm² to m²
noncomputable def cm2_to_m2 : ℚ := 1 / 10000

-- Theorem statement
theorem tiled_floor_area :
  let tile_area_cm2 := tile_width * tile_length
  let tile_area_m2 := tile_area_cm2 * cm2_to_m2
  let total_tiles := num_rows * tiles_per_row
  let total_area_m2 := tile_area_m2 * (total_tiles : ℚ)
  total_area_m2 = 3.6 := by
  sorry

#eval num_rows * tiles_per_row

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiled_floor_area_l162_16213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_exist_l162_16298

-- Define the function f(x) = 2^x * x^2 - 1
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) * x^2 - 1

-- State the theorem
theorem three_solutions_exist :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
    (f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
    (∀ x : ℝ, f x = 0 → (x = a ∨ x = b ∨ x = c)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_exist_l162_16298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_box_distribution_l162_16202

/-- Represents the denominations of English silver coins in pence -/
inductive Coin : Type
  | threepence : Coin
  | fourpence : Coin
  | sixpence : Coin
  | shilling : Coin

/-- The value of a coin in pence -/
def coin_value : Coin → Nat
  | Coin.threepence => 3
  | Coin.fourpence => 4
  | Coin.sixpence => 6
  | Coin.shilling => 12

/-- A distribution of coins to a person -/
structure Distribution where
  coins : List Coin
  sum_to_19 : List.sum (List.map coin_value coins) = 19

/-- The Christmas box distribution problem -/
theorem christmas_box_distribution :
  ∃ (distributions : List Distribution),
    distributions.length = 19 ∧
    (List.sum (distributions.map (λ d => d.coins.length)) = 100) ∧
    (∀ d ∈ distributions, d.coins.length = 5 ∨ d.coins.length = 6) ∧
    (List.length (List.filter (λ d => d.coins.length = 5) distributions) = 14) ∧
    (List.length (List.filter (λ d => d.coins.length = 6) distributions) = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_box_distribution_l162_16202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swapping_matrix_l162_16251

theorem row_swapping_matrix (a b c d : ℝ) : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  N * M = !![c, d; a, b] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swapping_matrix_l162_16251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_h_range_l162_16217

-- Define the set M
def M : Set ℝ := {x : ℝ | |2*x - 1| < 1}

-- Define the function h
noncomputable def h (a b : ℝ) : ℝ := max (2 / Real.sqrt a) (max ((a + b) / Real.sqrt (a * b)) (2 / Real.sqrt b))

-- Theorem statement
theorem inequality_and_h_range (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1 > a + b) ∧ (h a b > 2 ∧ ∀ (k : ℝ), ∃ (x y : ℝ), x ∈ M ∧ y ∈ M ∧ h x y > k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_h_range_l162_16217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_asymptotes_l162_16205

/-- Given a hyperbola and a circle that intersect to form a square, 
    prove that the asymptote equations of the hyperbola have a specific form. -/
theorem hyperbola_circle_intersection_asymptotes 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c = Real.sqrt (a^2 + b^2)) : 
  let hyperbola := fun (x y : ℝ) ↦ y^2 / a^2 - x^2 / b^2 = 1
  let circle := fun (x y : ℝ) ↦ x^2 + y^2 = c^2
  let intersection_is_square := ∃ (x y : ℝ), 
    hyperbola x y ∧ circle x y ∧ x^2 = y^2
  let asymptote := fun (x : ℝ) ↦ Real.sqrt (Real.sqrt 2 - 1) * x
  intersection_is_square → 
  (fun (x : ℝ) ↦ hyperbola x (asymptote x)) = 
  (fun (x : ℝ) ↦ hyperbola x (-(asymptote x))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_asymptotes_l162_16205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l162_16245

noncomputable section

/-- The area of a closed figure bounded by x = e, y = x, and y = 1/x -/
def boundedArea : ℝ := (Real.exp 2 - 3) / 2

/-- The upper bound of the integration interval -/
def upperBound : ℝ := Real.exp 1

/-- The lower bound of the integration interval -/
def lowerBound : ℝ := 1

/-- The function representing the difference between y = x and y = 1/x -/
def areaFunction (x : ℝ) : ℝ := x - 1/x

theorem area_calculation :
  boundedArea = ∫ x in lowerBound..upperBound, areaFunction x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l162_16245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l162_16212

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (5 * x - 2) / Real.sqrt (x^2 - 3 * x - 4)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y ∧ x ∈ domain_f} = domain_f :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l162_16212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l162_16235

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * x + Real.pi / 3) + Real.sqrt 3 * (Real.sin x) ^ 2 - Real.sqrt 3 * (Real.cos x) ^ 2 - 1 / 2

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_monotone_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ is_periodic f p ∧ ∀ q, q > 0 ∧ is_periodic f q → p ≤ q) ∧
  (∀ k : ℤ, is_monotone_on f (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) ∧
  (∀ x₀ : ℝ, x₀ ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3) → 
    f x₀ = Real.sqrt 3 / 3 - 1 / 2 → Real.cos (2 * x₀) = -(3 + Real.sqrt 6) / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l162_16235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l162_16220

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (abs x + x) / 2 + 1

-- State the theorem
theorem range_of_inequality :
  {x : ℝ | f (1 - x^2) > f (2*x)} = {x : ℝ | -1 < x ∧ x < Real.sqrt 2 - 1} := by
  sorry

-- You can add more lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l162_16220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l162_16294

theorem triangle_angle_measure (x : ℝ) (h1 : 115 + x = 180) 
  (h2 : 30 + 25 + x = 180) : x = 125 := by
  -- Proof goes here
  sorry

#check triangle_angle_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l162_16294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l162_16297

-- Define the moving point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the fixed point F₁
def F₁ : Point := { x := 1, y := 0 }

-- Define point A
def A : Point := { x := 2, y := 0 }

-- Define the distance condition
def distance_condition (P : Point) : Prop :=
  |P.x - 4| = 2 * Real.sqrt ((P.x - 1)^2 + P.y^2)

-- Define the trajectory equation
def trajectory_equation (P : Point) : Prop :=
  P.x^2 / 4 + P.y^2 / 3 = 1

-- Define the line passing through F₁ with slope 1
def line_equation (P : Point) : Prop :=
  P.y = P.x - 1

-- Define the area of triangle ACD
noncomputable def area_ACD (C D : Point) : ℝ :=
  (1/2) * |A.x - F₁.x| * |C.y - D.y|

-- State the theorem
theorem trajectory_and_area :
  ∀ P : Point, distance_condition P →
  ∃ C D : Point,
    trajectory_equation P ∧
    trajectory_equation C ∧
    trajectory_equation D ∧
    line_equation C ∧
    line_equation D ∧
    area_ACD C D = 6 * Real.sqrt 2 / 7 := by
  sorry

#check trajectory_and_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l162_16297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_range_l162_16250

-- Define the fixed points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x - 2)^2 + y^2 = 2 * ((x - 1)^2 + y^2)

-- Define the trajectory C
def C : Set (ℝ × ℝ) :=
  {P | satisfies_condition P}

-- Define the slope of a line through the origin and a point
noncomputable def slope_origin (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  y / x

-- Define the slope of a line through two points
noncomputable def slope_two_points (P Q : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  (y₂ - y₁) / (x₂ - x₁)

-- The main theorem
theorem trajectory_and_slope_range :
  (∀ P ∈ C, (P.1)^2 + (P.2)^2 = 2) ∧
  (∀ A B : ℝ × ℝ, A ∈ C → B ∈ C → A ≠ B →
    let k₁ := slope_origin A
    let k₂ := slope_origin B
    let k := slope_two_points A B
    k₁ * k₂ = 3 →
    (k > -Real.sqrt 3 ∧ k < -Real.sqrt 3 / 3) ∨
    (k > Real.sqrt 3 / 3 ∧ k < Real.sqrt 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_slope_range_l162_16250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_smallest_angle_l162_16246

theorem triangle_cosine_smallest_angle :
  ∀ (a b c : ℝ) (x y : ℝ),
    a = 4 ∧ b = 6 ∧ c = 8 →  -- Side lengths are 4, 6, and 8
    x > 0 ∧ y > 0 →  -- Angles are positive
    y = 3 * x →  -- Largest angle is three times the smallest
    c^2 = a^2 + b^2 - 2*a*b*Real.cos y →  -- Law of cosines for largest angle
    a^2 = b^2 + c^2 - 2*b*c*Real.cos x →  -- Law of cosines for smallest angle
    Real.cos x = 17/16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_smallest_angle_l162_16246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_percentage_l162_16269

/-- The percentage of employees in a department given its sector angle in a circle graph --/
noncomputable def department_percentage (sector_angle : ℚ) : ℚ :=
  (sector_angle / 360) * 100

/-- Theorem: The manufacturing department with a 144° sector represents 40% of employees --/
theorem manufacturing_percentage :
  department_percentage 144 = 40 := by
  -- Unfold the definition of department_percentage
  unfold department_percentage
  -- Simplify the rational number arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_percentage_l162_16269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l162_16233

noncomputable def f (x : Real) : Real :=
  abs (Matrix.det 
    ![![Real.cos (2 * x), -Real.sin x],
      ![Real.cos x, Real.sqrt 3 / 2]])

theorem min_value_f (x : Real) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  ∃ (y : Real), y ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f y = -Real.sqrt 3 / 2 ∧
    ∀ (z : Real), z ∈ Set.Icc 0 (Real.pi / 2) → f z ≥ f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l162_16233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_triangle_area_relation_l162_16293

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle constructed on a side of a triangle --/
structure CircleOnSide where
  center : Point
  radius : ℝ

/-- Represents a curvilinear triangle formed by the circle construction --/
structure CurvilinearTriangle where
  area : ℝ

/-- Represents an acute-angled triangle with circles constructed on its sides --/
structure TriangleWithCircles where
  A : Point
  B : Point
  C : Point
  sideAB : CircleOnSide
  sideBC : CircleOnSide
  sideCA : CircleOnSide
  externalTriangle1 : CurvilinearTriangle
  externalTriangle2 : CurvilinearTriangle
  externalTriangle3 : CurvilinearTriangle
  internalTriangle : CurvilinearTriangle

/-- Calculate the area of a triangle given its three vertices --/
def area_of_triangle (A B C : Point) : ℝ := sorry

/-- The theorem to be proved --/
theorem curvilinear_triangle_area_relation (t : TriangleWithCircles) :
  t.externalTriangle1.area + t.externalTriangle2.area + t.externalTriangle3.area - t.internalTriangle.area =
  2 * area_of_triangle t.A t.B t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_triangle_area_relation_l162_16293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_equation_system_l162_16238

/-- Represents the number of rooms in the shop -/
def x : ℕ := sorry

/-- Represents the total number of guests -/
def y : ℕ := sorry

/-- The system of equations correctly represents the given conditions -/
theorem shop_equation_system : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_equation_system_l162_16238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_value_when_condition_M_independent_of_x_l162_16277

variable (x y : ℝ)

def A (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y + 1
def M (x y : ℝ) : ℝ := 4 * A x - (3 * A x - 2 * B x y)

theorem M_value_when_condition (h : (x + 1)^2 + |y - 2| = 0) : M x y = -1 := by
  sorry

theorem M_independent_of_x (h : y = 1) : ∀ x₁ x₂, M x₁ y = M x₂ y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_value_when_condition_M_independent_of_x_l162_16277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_equal_parts_l162_16244

theorem complex_multiplication_equal_parts (a : ℝ) : 
  (((a : ℂ) + Complex.I) * (3 + 4 * Complex.I)).re = 
  (((a : ℂ) + Complex.I) * (3 + 4 * Complex.I)).im → 
  a = -7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_equal_parts_l162_16244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_ab_power_l162_16241

theorem max_sum_of_ab_power (a b : ℕ) (ha : a > 0) (hb : b > 1) (h : a^b < 399) : a + b ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_ab_power_l162_16241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l162_16292

noncomputable section

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

/-- The line equation -/
def line (x y : ℝ) : Prop := y = x - Real.sqrt 3

/-- The intersection points of the line and the ellipse -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ ellipse x y ∧ line x y}

/-- The length of the chord AB -/
def chord_length : ℝ := 8 / 5

theorem chord_length_is_correct :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
  A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l162_16292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_condition_l162_16262

theorem ln_inequality_condition (x : ℝ) : 
  (∀ x, Real.log (x + 1) < 0 → x < 0) ∧ 
  (∃ x, x < 0 ∧ Real.log (x + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_condition_l162_16262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l162_16267

/-- Definition of the ellipse Γ -/
noncomputable def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 2 = 1}

/-- Left endpoint of the ellipse -/
def C : ℝ × ℝ := (-2, 0)

/-- Right endpoint of the ellipse -/
def D : ℝ × ℝ := (2, 0)

/-- A point on the perpendicular line at D -/
def M (y₀ : ℝ) : ℝ × ℝ := (2, y₀)

/-- The intersection point of CM and the ellipse -/
noncomputable def P (y₀ : ℝ) : ℝ × ℝ :=
  ((-2 * y₀^2 + 16) / (y₀^2 + 8), (8 * y₀) / (y₀^2 + 8))

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem constant_dot_product (y₀ : ℝ) :
  dot_product (M y₀ - O) (P y₀ - O) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l162_16267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l162_16268

-- Define the initial speed and the speed reduction function
noncomputable def initial_speed : ℝ := 30

-- k is the constant of variation
noncomputable def speed_reduction (k : ℝ) (n : ℝ) : ℝ := k * (n ^ (1/2 : ℝ))

-- Define the theorem
theorem train_speed_proof (k : ℝ) (n : ℝ) : 
  -- Given conditions
  (initial_speed - speed_reduction k n = 18) →
  (initial_speed - speed_reduction k 16 = 14) →
  -- Conclusion
  (speed_reduction k 16 = 16) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l162_16268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_l162_16234

/-- Represents a traveler with a constant speed -/
structure Traveler where
  speed : ℚ
  deriving Repr

/-- Represents the road between two points -/
structure Road where
  length : ℚ
  deriving Repr

/-- Represents the meeting of two travelers -/
def meet (t1 t2 : Traveler) (r : Road) (time : ℚ) : Prop :=
  t1.speed * time + t2.speed * time = r.length

/-- The main theorem -/
theorem second_meeting_time 
  (alex bob : Traveler) 
  (road : Road) 
  (h1 : alex.speed = 3 * bob.speed) 
  (h2 : meet alex bob road (15 / 60)) : 
  meet alex bob road (30 / 60) := by
  sorry

#check second_meeting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_l162_16234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_g_l162_16227

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x / 2) - Real.cos (x / 2)

noncomputable def g (x : ℝ) : ℝ := f (x - 2 * Real.pi / 3)

theorem decreasing_interval_of_g :
  ∀ x₁ x₂, -Real.pi / 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < -Real.pi / 4 → g x₁ > g x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_g_l162_16227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_simplification_l162_16276

-- Define the expressions
def expr_A (x y : ℝ) := (-x - y) * (-x + y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (y + x) * (x - y)
def expr_D (x y : ℝ) := (y - x) * (x + y)

-- Define a predicate for expressions that can be simplified using difference of squares
def is_difference_of_squares (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ → ℝ), ∀ x y, f x y = (g x y) ^ 2 - (h x y) ^ 2

-- Theorem statement
theorem difference_of_squares_simplification :
  ¬(is_difference_of_squares expr_B) ∧
  (is_difference_of_squares expr_A) ∧
  (is_difference_of_squares expr_C) ∧
  (is_difference_of_squares expr_D) := by
  sorry

#check difference_of_squares_simplification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_simplification_l162_16276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_normal_equation_l162_16287

/-- The normal equation of a plane given a point and a normal vector -/
theorem plane_normal_equation (P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) :
  P = (1, 2, -1) →
  n = (-2, 3, 1) →
  ∀ (x y z : ℝ), (2*x - 3*y - z + 3 = 0) ↔ 
    (n.fst * (x - P.fst) + n.snd.fst * (y - P.snd.fst) + n.snd.snd * (z - P.snd.snd) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_normal_equation_l162_16287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sale_reduction_l162_16249

theorem special_sale_reduction (initial_price : ℝ) (initial_price_positive : initial_price > 0) : 
  let price_after_first_reduction := initial_price * (1 - 0.2)
  let price_after_special_sale := price_after_first_reduction * (1 - 0.25)
  ∃ ε > 0, |price_after_special_sale * (1 + 0.6667) - initial_price| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sale_reduction_l162_16249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_quadratic_solution_l162_16255

theorem sum_product_quadratic_solution (S P : ℝ) :
  ∃ x y : ℝ, x + y = S ∧ x * y = P →
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_quadratic_solution_l162_16255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_l162_16296

theorem walkway_time (length : ℝ) (time_with : ℝ) (time_against : ℝ) : 
  length = 200 → time_with = 60 → time_against = 360 →
  (let v_p := (length / time_with + length / time_against) / 2;
   length / v_p) = 7200 / 65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_l162_16296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_value_calculation_l162_16215

open Real

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (π + α) * Real.sin (2*π - α) * Real.cos (-π - α) * Real.cos (π/2 + α)) /
  (Real.sin (3*π + α) * Real.cos (π - α) * Real.cos (3*π/2 + α)) = Real.sin α := by sorry

theorem trigonometric_value_calculation :
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_value_calculation_l162_16215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_small_number_l162_16286

/-- Expresses 0.00000164 in scientific notation -/
theorem scientific_notation_of_small_number :
  (0.00000164 : ℝ) = 1.64 * (10 : ℝ)^(-6 : ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_small_number_l162_16286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l162_16264

/-- The distance between two stations A and B on a straight line --/
def distance_AB : ℝ := sorry

/-- The speed of the first train (in km/h) --/
def speed_train1 : ℝ := 20

/-- The speed of the second train (in km/h) --/
def speed_train2 : ℝ := 25

/-- The time difference between the starts of the two trains (in hours) --/
def time_difference : ℝ := 1

/-- The time it takes for the trains to meet (in hours) --/
def meeting_time : ℝ := 1

theorem distance_between_stations :
  distance_AB = speed_train1 * meeting_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l162_16264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_l162_16274

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.tan x + 1

-- State the theorem
theorem f_negative_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_l162_16274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_inventory_theorem_l162_16265

theorem bookstore_inventory_theorem 
  (total_inventory : ℕ) 
  (historical_fiction_ratio : ℚ)
  (historical_fiction_new_release_ratio : ℚ)
  (other_new_release_ratio : ℚ)
  (h1 : historical_fiction_ratio = 2/5)
  (h2 : historical_fiction_new_release_ratio = 2/5)
  (h3 : other_new_release_ratio = 2/5) :
  let historical_fiction := total_inventory * historical_fiction_ratio
  let historical_fiction_new_releases := historical_fiction * historical_fiction_new_release_ratio
  let other_books := total_inventory - historical_fiction
  let other_new_releases := other_books * other_new_release_ratio
  let total_new_releases := historical_fiction_new_releases + other_new_releases
  historical_fiction_new_releases / total_new_releases = 2/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_inventory_theorem_l162_16265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l162_16201

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the focus F and point A
def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (3, 0)

-- Define the condition OF = 2FA
def OF_eq_2FA : Prop := 
  let O : ℝ × ℝ := (0, 0)
  (F.1 - O.1)^2 + (F.2 - O.2)^2 = 4 * ((A.1 - F.1)^2 + (A.2 - F.2)^2)

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

-- Define the line PQ
def LineEq (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 3)

theorem ellipse_properties :
  -- Part 1: Prove the equation of the ellipse and its eccentricity
  (∀ x y : ℝ, Ellipse x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  (eccentricity = Real.sqrt 6 / 3) ∧
  -- Part 2: Prove the equation of line PQ when OP ⊥ OQ
  (∀ P Q : ℝ × ℝ, 
    Ellipse P.1 P.2 → Ellipse Q.1 Q.2 → 
    (∃ k : ℝ, LineEq k P.1 P.2 ∧ LineEq k Q.1 Q.2 ∧ LineEq k A.1 A.2) →
    P.1 * Q.1 + P.2 * Q.2 = 0 →
    (∃ k : ℝ, k = Real.sqrt 5 / 5 ∨ k = -Real.sqrt 5 / 5) ∧
    (∀ x y : ℝ, LineEq (Real.sqrt 5 / 5) x y ∨ LineEq (-Real.sqrt 5 / 5) x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l162_16201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_minimum_time_l162_16285

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Represents the ant's speed in different directions -/
structure AntSpeed where
  ascending : ℚ
  descending : ℚ
  horizontal : ℚ

/-- Calculates the time taken for a given distance and speed -/
def timeTaken (distance : ℚ) (speed : ℚ) : ℚ := distance / speed

/-- Theorem stating the minimum time for the ant to complete the path -/
theorem ant_path_minimum_time (block : BlockDimensions) (speed : AntSpeed) 
  (h1 : block.length = 12)
  (h2 : block.width = 4)
  (h3 : block.height = 2)
  (h4 : speed.ascending = 2)
  (h5 : speed.descending = 3)
  (h6 : speed.horizontal = 4) :
  let t1 := timeTaken block.height speed.ascending
  let t2 := timeTaken (5 * block.width) speed.horizontal
  let t3 := timeTaken block.height speed.descending
  let t4 := timeTaken (2 * block.length) speed.horizontal
  let t5 := timeTaken block.height speed.ascending
  let t6 := timeTaken (5 * block.width) speed.horizontal
  let t7 := timeTaken block.height speed.descending
  t1 + t2 + t3 + t4 + t5 + t6 + t7 = 58 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_minimum_time_l162_16285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_perpendicular_l162_16232

/-- Given two lines in the plane, prove that if the direction vector of one line
    is perpendicular to the direction vector of the other line, then a specific
    condition on the parameter 'a' holds. -/
theorem direction_vector_perpendicular (a : ℝ) : 
  (1 : ℝ) * (1 : ℝ) + (-(a + 3)) * (5 / (3 - a)) = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_perpendicular_l162_16232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_usage_theorem_l162_16252

/-- Amount of flour in the canister in cups -/
def canister_flour : ℚ := 5/2

/-- Amount of flour required for one recipe in cups -/
def recipe_flour : ℚ := 4/3

/-- Number of recipes to be made -/
def recipe_multiplier : ℚ := 3/2

/-- Calculates the percentage of flour used, rounded to the nearest whole percent -/
def flour_percentage_used (canister : ℚ) (recipe : ℚ) (multiplier : ℚ) : ℕ :=
  (recipe * multiplier / canister * 100 + 1/2).floor.toNat

theorem flour_usage_theorem :
  flour_percentage_used canister_flour recipe_flour recipe_multiplier = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_usage_theorem_l162_16252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_valid_function_l162_16231

-- Define the sets A and B
noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {y : ℝ | 1 ≤ y ∧ y ≤ 4}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Theorem stating that f is not a valid function from A to B
theorem f_not_valid_function : ¬(∀ x ∈ A, f x ∈ B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_valid_function_l162_16231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromes_l162_16229

/-- A function that checks if a number is a 5-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Bool :=
  n ≥ 10000 ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The set of all 5-digit palindromes -/
def five_digit_palindromes : Set ℕ :=
  {n : ℕ | is_five_digit_palindrome n = true}

/-- The main theorem stating that there are 900 5-digit palindromes -/
theorem count_five_digit_palindromes : 
  Finset.card (Finset.filter (fun n => is_five_digit_palindrome n) (Finset.range 100000)) = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromes_l162_16229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_properties_l162_16283

-- Define points A and B
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 0)

-- Define the length of AB
noncomputable def length_AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the line segment AB
def in_line_segment (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t • B.1 + (1 - t) • A.1, t • B.2 + (1 - t) • A.2)

-- Theorem statement
theorem AB_properties :
  (length_AB = 2 * Real.sqrt 5) ∧
  (∀ x y : ℝ, in_line_segment (x, y) ↔ line_AB x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_properties_l162_16283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l162_16224

theorem triangle_angles (A B C : ℝ) : 
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  Real.sin A + Real.cos A = Real.sqrt 2 →
  Real.sqrt 3 * Real.cos A = - Real.sqrt 2 * Real.cos (Real.pi - B) →
  A = Real.pi / 4 ∧ B = Real.pi / 6 ∧ C = 7 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l162_16224
