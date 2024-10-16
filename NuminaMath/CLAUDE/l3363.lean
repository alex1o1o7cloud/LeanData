import Mathlib

namespace NUMINAMATH_CALUDE_intersection_and_union_range_of_m_l3363_336366

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m}

-- Theorem for part (Ⅰ)
theorem intersection_and_union :
  (A ∩ B = {x | 2 ≤ x ∧ x < 5}) ∧
  ((Aᶜ ∪ B) = {x | -3 < x ∧ x < 5}) := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m (m : ℝ) :
  (B ∩ C m = C m) → (m < -1 ∨ (2 < m ∧ m < 5/2)) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_range_of_m_l3363_336366


namespace NUMINAMATH_CALUDE_triangle_strike_interval_l3363_336343

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem triangle_strike_interval (cymbal_interval triangle_interval coincidence_interval : ℕ) :
  cymbal_interval = 7 →
  is_factor triangle_interval coincidence_interval →
  is_factor cymbal_interval coincidence_interval →
  coincidence_interval = 14 →
  triangle_interval ≠ cymbal_interval →
  triangle_interval > 0 →
  triangle_interval = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_strike_interval_l3363_336343


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l3363_336334

/-- Given two lines l₁ and l₂ with equations 3x + my - 1 = 0 and (m+2)x - (m-2)y + 2 = 0 respectively,
    if l₁ is parallel to l₂, then m = -6 or m = 1. -/
theorem parallel_lines_m_values (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | 3 * x + m * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (m + 2) * x - (m - 2) * y + 2 = 0}
  (∀ (a b c d : ℝ), a * (m + 2) = 3 * c ∧ b * (m - 2) = -m * d → (a, b) = (c, d)) →
  m = -6 ∨ m = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_values_l3363_336334


namespace NUMINAMATH_CALUDE_real_part_of_z_l3363_336322

theorem real_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  Complex.re z = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3363_336322


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l3363_336335

def canoes_built (month : Nat) : Nat :=
  match month with
  | 0 => 4  -- February
  | n + 1 => 3 * canoes_built n

theorem total_canoes_by_april : 
  (canoes_built 0) + (canoes_built 1) + (canoes_built 2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l3363_336335


namespace NUMINAMATH_CALUDE_rotation_coordinates_l3363_336361

/-- 
Given a point (x, y) in a Cartesian coordinate plane and a rotation by angle α around the origin,
the coordinates of the rotated point are (x cos α - y sin α, x sin α + y cos α).
-/
theorem rotation_coordinates (x y α : ℝ) : 
  let original_point := (x, y)
  let rotated_point := (x * Real.cos α - y * Real.sin α, x * Real.sin α + y * Real.cos α)
  ∃ (R φ : ℝ), 
    x = R * Real.cos φ ∧ 
    y = R * Real.sin φ ∧ 
    rotated_point = (R * Real.cos (φ + α), R * Real.sin (φ + α)) :=
by sorry

end NUMINAMATH_CALUDE_rotation_coordinates_l3363_336361


namespace NUMINAMATH_CALUDE_velocity_at_4_seconds_l3363_336318

-- Define the motion equation
def motion_equation (t : ℝ) : ℝ := t^2 - t + 2

-- Define the instantaneous velocity function
def instantaneous_velocity (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem velocity_at_4_seconds :
  instantaneous_velocity 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_velocity_at_4_seconds_l3363_336318


namespace NUMINAMATH_CALUDE_flow_rate_is_twelve_l3363_336362

/-- Represents the flow rate problem described in the question -/
def flow_rate_problem (tub_capacity : ℕ) (leak_rate : ℕ) (fill_time : ℕ) : ℕ :=
  let cycles := fill_time / 2
  let net_fill_per_cycle := (tub_capacity / cycles) + (2 * leak_rate)
  net_fill_per_cycle

/-- Theorem stating that the flow rate is 12 liters per minute under the given conditions -/
theorem flow_rate_is_twelve :
  flow_rate_problem 120 1 24 = 12 := by
  sorry

end NUMINAMATH_CALUDE_flow_rate_is_twelve_l3363_336362


namespace NUMINAMATH_CALUDE_gianna_savings_l3363_336314

def total_savings : ℕ := 14235
def days_in_year : ℕ := 365
def daily_savings : ℚ := total_savings / days_in_year

theorem gianna_savings : daily_savings = 39 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_l3363_336314


namespace NUMINAMATH_CALUDE_equation_solution_range_l3363_336370

theorem equation_solution_range (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k / (2 * x - 4) - 1 = x / (x - 2)) → 
  (k > -4 ∧ k ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3363_336370


namespace NUMINAMATH_CALUDE_minimum_growth_rate_for_doubling_output_l3363_336339

theorem minimum_growth_rate_for_doubling_output :
  let r : ℝ := Real.sqrt 2 - 1
  ∀ x : ℝ, (1 + x)^2 ≥ 2 → x ≥ r :=
by sorry

end NUMINAMATH_CALUDE_minimum_growth_rate_for_doubling_output_l3363_336339


namespace NUMINAMATH_CALUDE_sequence_divisibility_l3363_336331

/-- A sequence of 2007 elements, each either 2 or 3 -/
def Sequence := Fin 2007 → Fin 2

/-- The property that all elements of a sequence are divisible by 5 -/
def AllDivisibleBy5 (x : Fin 2007 → ℤ) : Prop :=
  ∀ i, x i % 5 = 0

/-- The main theorem -/
theorem sequence_divisibility (a : Sequence) (x : Fin 2007 → ℤ)
    (h : ∀ i : Fin 2007, (a i.val + 2 : Fin 2) * x i + x ((i + 2) % 2007) ≡ 0 [ZMOD 5]) :
    AllDivisibleBy5 x := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l3363_336331


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3363_336344

theorem arithmetic_mean_problem : 
  let a := 9/16
  let b := 3/4
  let c := 5/8
  c = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3363_336344


namespace NUMINAMATH_CALUDE_lines_perpendicular_when_a_is_neg_six_l3363_336325

/-- Given two lines l₁ and l₂ defined by their equations, prove that they are perpendicular when a = -6 -/
theorem lines_perpendicular_when_a_is_neg_six (a : ℝ) :
  a = -6 →
  let l₁ := {(x, y) : ℝ × ℝ | a * x + (1 - a) * y - 3 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (a - 1) * x + 2 * (a + 3) * y - 2 = 0}
  let m₁ := a / (1 - a)
  let m₂ := (a - 1) / (2 * (a + 3))
  m₁ * m₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_when_a_is_neg_six_l3363_336325


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3363_336328

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 - b*x + c

-- Define the theorem
theorem inequality_solution_set 
  (b c : ℝ) 
  (hb : b > 0) 
  (hc : c > 0) 
  (x₁ x₂ : ℝ) 
  (h_zeros : f b c x₁ = 0 ∧ f b c x₂ = 0) 
  (h_progression : (∃ r : ℝ, x₁ = -1 * r ∧ x₂ = -1 / r) ∨ 
                   (∃ d : ℝ, x₁ = -1 - d ∧ x₂ = -1 + d)) :
  {x : ℝ | (x - b) / (x - c) ≤ 0} = Set.Ioo 1 (5/2) ∪ {5/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3363_336328


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3363_336389

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 512 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3363_336389


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3363_336378

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

-- Define the conditions
def triangle_conditions (t : IsoscelesTriangle) : Prop :=
  t.base = 4 ∧ t.leg^2 - 5*t.leg + 6 = 0

-- Define the perimeter
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2*t.leg

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, triangle_conditions t → perimeter t = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3363_336378


namespace NUMINAMATH_CALUDE_triangle_max_area_l3363_336306

/-- The maximum area of a triangle with one side of length 3 and the sum of the other two sides equal to 5 is 3. -/
theorem triangle_max_area :
  ∀ (b c : ℝ),
  b > 0 → c > 0 →
  b + c = 5 →
  let a := 3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area ≤ 3 ∧ ∃ (b₀ c₀ : ℝ), b₀ > 0 ∧ c₀ > 0 ∧ b₀ + c₀ = 5 ∧
    let s₀ := (a + b₀ + c₀) / 2
    Real.sqrt (s₀ * (s₀ - a) * (s₀ - b₀) * (s₀ - c₀)) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3363_336306


namespace NUMINAMATH_CALUDE_ap_sequence_a_equals_one_l3363_336345

/-- Given a sequence 1, 6 + 2a, 10 + 5a, ..., if it forms an arithmetic progression, then a = 1 -/
theorem ap_sequence_a_equals_one (a : ℝ) :
  (∀ n : ℕ, (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) n.succ - 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) n = 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) 1 - 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) 0) →
  a = 1 := by
sorry


end NUMINAMATH_CALUDE_ap_sequence_a_equals_one_l3363_336345


namespace NUMINAMATH_CALUDE_negation_equivalence_l3363_336390

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 4 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3363_336390


namespace NUMINAMATH_CALUDE_f_shift_three_l3363_336381

/-- Given a function f(x) = x(x-1)/2, prove that f(x+3) = f(x) + 3x + 3 for all real x. -/
theorem f_shift_three (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (x - 1) / 2
  f (x + 3) = f x + 3 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_f_shift_three_l3363_336381


namespace NUMINAMATH_CALUDE_function_range_l3363_336308

/-- Given a^2 - a < 2 and a is a positive integer, 
    the range of f(x) = x + 2a/x is (-∞, -2√2] ∪ [2√2, +∞) -/
theorem function_range (a : ℕ+) (h : a^2 - a < 2) :
  Set.range (fun x : ℝ => x + 2*a/x) = 
    Set.Iic (-2 * Real.sqrt 2) ∪ Set.Ici (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_function_range_l3363_336308


namespace NUMINAMATH_CALUDE_percentage_of_women_in_survey_l3363_336321

theorem percentage_of_women_in_survey (mothers_full_time : Real) 
  (fathers_full_time : Real) (total_not_full_time : Real) :
  mothers_full_time = 5/6 →
  fathers_full_time = 3/4 →
  total_not_full_time = 1/5 →
  ∃ (w : Real), w = 3/5 ∧ 
    w * (1 - mothers_full_time) + (1 - w) * (1 - fathers_full_time) = total_not_full_time :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_survey_l3363_336321


namespace NUMINAMATH_CALUDE_difference_of_squares_and_product_l3363_336324

theorem difference_of_squares_and_product (a b : ℝ) 
  (h1 : a^2 + b^2 = 150) 
  (h2 : a * b = 25) : 
  |a - b| = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_and_product_l3363_336324


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3363_336338

theorem simplify_and_evaluate : (Real.sqrt 2 + 1)^2 - 2*(Real.sqrt 2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3363_336338


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3363_336380

theorem triangle_side_lengths (n : ℕ) : 
  (n + 4 + n + 10 > 4*n + 7) ∧ 
  (n + 4 + 4*n + 7 > n + 10) ∧ 
  (n + 10 + 4*n + 7 > n + 4) ∧ 
  (4*n + 7 > n + 10) ∧ 
  (n + 10 > n + 4) →
  (∃ (count : ℕ), count = 2 ∧ 
    (∀ m : ℕ, (m + 4 + m + 10 > 4*m + 7) ∧ 
              (m + 4 + 4*m + 7 > m + 10) ∧ 
              (m + 10 + 4*m + 7 > m + 4) ∧ 
              (4*m + 7 > m + 10) ∧ 
              (m + 10 > m + 4) ↔ 
              (m = n ∨ m = n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3363_336380


namespace NUMINAMATH_CALUDE_probability_of_selecting_two_first_class_with_s_equal_four_l3363_336323

/-- A product with quality indicators -/
structure Product where
  x : ℕ
  y : ℕ
  z : ℕ

/-- The overall quality indicator -/
def S (p : Product) : ℕ := p.x + p.y + p.z

/-- A product is first-class if its overall indicator is ≤ 4 -/
def isFirstClass (p : Product) : Prop := S p ≤ 4

/-- The sample of 10 products -/
def sample : Finset Product := sorry

/-- The set of first-class products in the sample -/
def firstClassProducts : Finset Product := sorry

/-- The set of first-class products with S = 4 -/
def firstClassProductsWithSEqualFour : Finset Product := sorry

theorem probability_of_selecting_two_first_class_with_s_equal_four :
  (sample.card = 10) →
  (firstClassProducts.card = 6) →
  (firstClassProductsWithSEqualFour.card = 4) →
  (Nat.choose 6 2 : ℚ)⁻¹ * (Nat.choose 4 2 : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_two_first_class_with_s_equal_four_l3363_336323


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3363_336330

/-- The coordinates of a point (3, -2) with respect to the origin are (3, -2). -/
theorem point_coordinates_wrt_origin :
  let p : ℝ × ℝ := (3, -2)
  p = (3, -2) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3363_336330


namespace NUMINAMATH_CALUDE_rain_probability_l3363_336398

/-- Given probabilities of rain events in counties, prove the probability of rain on both days -/
theorem rain_probability (p_monday p_tuesday p_no_rain : ℝ) 
  (h1 : p_monday = 0.6)
  (h2 : p_tuesday = 0.55)
  (h3 : p_no_rain = 0.25) :
  p_monday + p_tuesday - (1 - p_no_rain) = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_rain_probability_l3363_336398


namespace NUMINAMATH_CALUDE_jakes_drink_volume_l3363_336346

/-- A drink recipe with parts of different ingredients -/
structure DrinkRecipe where
  coke_parts : ℕ
  sprite_parts : ℕ
  mountain_dew_parts : ℕ

/-- Calculate the total volume of a drink given its recipe and the volume of one ingredient -/
def total_volume (recipe : DrinkRecipe) (coke_volume : ℚ) : ℚ :=
  let total_parts := recipe.coke_parts + recipe.sprite_parts + recipe.mountain_dew_parts
  let volume_per_part := coke_volume / recipe.coke_parts
  total_parts * volume_per_part

/-- Theorem stating that for the given recipe and Coke volume, the total volume is 22 ounces -/
theorem jakes_drink_volume : 
  let recipe := DrinkRecipe.mk 4 2 5
  total_volume recipe 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_jakes_drink_volume_l3363_336346


namespace NUMINAMATH_CALUDE_exists_line_with_F_as_incenter_l3363_336320

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The right focus F -/
def F : ℝ × ℝ := (1, 0)

/-- The upper vertex M -/
def M : ℝ × ℝ := (0, 1)

/-- The line l -/
def line_l (x y : ℝ) : Prop := x + (2 - Real.sqrt 6) * y + 6 - 3 * Real.sqrt 6 = 0

/-- P and Q are points on both the ellipse and line l -/
def P_Q_on_ellipse_and_line (P Q : ℝ × ℝ) : Prop :=
  ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ line_l P.1 P.2 ∧ line_l Q.1 Q.2

/-- F is the incenter of triangle MPQ -/
def F_is_incenter (P Q : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
    (F.1 - P.1)^2 + (F.2 - P.2)^2 = r^2 ∧
    (F.1 - Q.1)^2 + (F.2 - Q.2)^2 = r^2 ∧
    (F.1 - M.1)^2 + (F.2 - M.2)^2 = r^2

/-- The main theorem -/
theorem exists_line_with_F_as_incenter :
  ∃ (P Q : ℝ × ℝ), P_Q_on_ellipse_and_line P Q ∧ F_is_incenter P Q := by sorry

end NUMINAMATH_CALUDE_exists_line_with_F_as_incenter_l3363_336320


namespace NUMINAMATH_CALUDE_joan_football_games_l3363_336319

/-- The number of football games Joan attended this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan attended last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games : total_games = 13 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l3363_336319


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3363_336365

theorem sin_2alpha_value (α : Real) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = -17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3363_336365


namespace NUMINAMATH_CALUDE_product_from_sum_and_difference_l3363_336337

theorem product_from_sum_and_difference (a b : ℝ) : 
  a + b = 60 ∧ a - b = 10 → a * b = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_from_sum_and_difference_l3363_336337


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l3363_336341

theorem unique_root_quadratic (c : ℝ) : 
  (∃ b : ℝ, b = c^2 + 1 ∧ 
   (∃! x : ℝ, x^2 + b*x + c = 0)) → 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l3363_336341


namespace NUMINAMATH_CALUDE_equation_solution_l3363_336348

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (4 * x^2 + 5 * x + 2) / (x + 2) = 4 * x - 2 ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3363_336348


namespace NUMINAMATH_CALUDE_officer_election_ways_l3363_336347

def club_size : ℕ := 12
def num_officers : ℕ := 5

theorem officer_election_ways :
  (club_size * (club_size - 1) * (club_size - 2) * (club_size - 3) * (club_size - 4) : ℕ) = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_election_ways_l3363_336347


namespace NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l3363_336388

/-- An infinite geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- A sequence is monotonically increasing -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The first three terms of a sequence are in ascending order -/
def FirstThreeAscending (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_monotonicity (a : ℕ → ℝ) 
  (h : GeometricSequence a) : 
  MonotonicallyIncreasing a ↔ FirstThreeAscending a := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l3363_336388


namespace NUMINAMATH_CALUDE_inequality_proof_l3363_336395

-- Define the functions f and g
def f (x : ℝ) : ℝ := |2*x + 1| - |2*x - 4|
def g (x : ℝ) : ℝ := 9 + 2*x - x^2

-- State the theorem
theorem inequality_proof (x : ℝ) : |8*x - 16| ≥ g x - 2 * f x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3363_336395


namespace NUMINAMATH_CALUDE_initial_red_marbles_l3363_336382

theorem initial_red_marbles (r g : ℕ) : 
  r * 3 = g * 5 → 
  (r - 20) * 5 = (g + 40) * 1 → 
  r = 317 := by
sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l3363_336382


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3363_336360

theorem arithmetic_mean_problem (x y : ℝ) : 
  ((x + y) + (y + 30) + (3 * x) + (y - 10) + (2 * x + y + 20)) / 5 = 50 → 
  x = 21 ∧ y = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3363_336360


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3363_336377

theorem fourth_root_equation_solutions : 
  {x : ℝ | x > 0 ∧ x^(1/4) = 15 / (8 - x^(1/4))} = {81, 625} := by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3363_336377


namespace NUMINAMATH_CALUDE_parabola_intersection_point_l3363_336367

theorem parabola_intersection_point (n m : ℕ) (x₀ y₀ : ℝ) 
  (hn : n ≥ 2) 
  (hm : m > 0) 
  (h1 : y₀^2 = n * x₀ - 1) 
  (h2 : y₀ = x₀) : 
  ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_point_l3363_336367


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_l3363_336393

theorem geometry_biology_overlap (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h_total : total = 232)
  (h_geometry : geometry = 144)
  (h_biology : biology = 119) :
  min geometry biology - (geometry + biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_l3363_336393


namespace NUMINAMATH_CALUDE_alicia_book_cost_l3363_336368

/-- The total cost of books given the number of each type and their individual costs -/
def total_cost (math_books art_books science_books : ℕ) (math_cost art_cost science_cost : ℕ) : ℕ :=
  math_books * math_cost + art_books * art_cost + science_books * science_cost

/-- Theorem stating that the total cost of Alicia's books is $30 -/
theorem alicia_book_cost : total_cost 2 3 6 3 2 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_alicia_book_cost_l3363_336368


namespace NUMINAMATH_CALUDE_rhombus_area_l3363_336304

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 21*d₁ + 30 = 0 → 
  d₂^2 - 21*d₂ + 30 = 0 → 
  d₁ ≠ d₂ →
  (1/2 : ℝ) * d₁ * d₂ = 15 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_l3363_336304


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l3363_336369

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  2 * x + 3 * y = 8 ∧ x = y - 1

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  2 * x - y = -1 ∧ x + 3 * y = 17

-- Theorem for the first system
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 1 ∧ y = 2 := by
  sorry

-- Theorem for the second system
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 2 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l3363_336369


namespace NUMINAMATH_CALUDE_no_perfect_square_solution_l3363_336333

theorem no_perfect_square_solution (n : ℕ) : ¬∃ (m : ℕ), n^5 - 5*n^3 + 4*n + 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_solution_l3363_336333


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3363_336394

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  prop1 : a 5 * a 7 = 2
  prop2 : a 2 + a 10 = 3

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 12 / seq.a 4 = 2 ∨ seq.a 12 / seq.a 4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3363_336394


namespace NUMINAMATH_CALUDE_lot_length_l3363_336302

/-- Given a rectangular lot with width 20 meters, height 2 meters, and volume 1600 cubic meters,
    prove that the length of the lot is 40 meters. -/
theorem lot_length (width : ℝ) (height : ℝ) (volume : ℝ) (length : ℝ) :
  width = 20 →
  height = 2 →
  volume = 1600 →
  volume = length * width * height →
  length = 40 := by
  sorry

end NUMINAMATH_CALUDE_lot_length_l3363_336302


namespace NUMINAMATH_CALUDE_system_equation_sum_l3363_336332

theorem system_equation_sum (a b c x y z : ℝ) 
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 19 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (ha : a ≠ 11)
  (hx : x ≠ 0) :
  a / (a - 11) + b / (b - 19) + c / (c - 37) = 1 := by
sorry

end NUMINAMATH_CALUDE_system_equation_sum_l3363_336332


namespace NUMINAMATH_CALUDE_years_ago_p_half_q_l3363_336359

/-- The number of years ago when p was half of q's age, given their current ages' ratio and sum. -/
theorem years_ago_p_half_q (p q : ℕ) (h1 : p * 4 = q * 3) (h2 : p + q = 28) : 
  ∃ y : ℕ, p - y = (q - y) / 2 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_years_ago_p_half_q_l3363_336359


namespace NUMINAMATH_CALUDE_smallest_four_digit_numbers_l3363_336383

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1

theorem smallest_four_digit_numbers :
  let valid_numbers := [1021, 1081, 1141, 1201]
  (∀ n ∈ valid_numbers, is_valid n) ∧
  (∀ m, is_valid m → m ≥ 1021) ∧
  (∀ n ∈ valid_numbers, ∀ m, is_valid m ∧ m < n → m ∈ valid_numbers) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_numbers_l3363_336383


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3363_336313

/-- Given a principal amount where the compound interest at 5% per annum for 2 years is $51.25,
    prove that the simple interest at the same rate and time is $50 -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3363_336313


namespace NUMINAMATH_CALUDE_door_ticket_cost_l3363_336350

/-- Proves the cost of tickets purchased at the door given ticket sales information -/
theorem door_ticket_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advance_ticket_cost : ℕ)
  (advance_tickets_sold : ℕ)
  (h1 : total_tickets = 140)
  (h2 : total_revenue = 1720)
  (h3 : advance_ticket_cost = 8)
  (h4 : advance_tickets_sold = 100) :
  (total_revenue - advance_ticket_cost * advance_tickets_sold) / (total_tickets - advance_tickets_sold) = 23 := by
  sorry


end NUMINAMATH_CALUDE_door_ticket_cost_l3363_336350


namespace NUMINAMATH_CALUDE_max_black_pens_l3363_336349

/-- The maximum number of pens in the basket -/
def max_pens : ℕ := 2500

/-- The probability of selecting two pens of the same color -/
def same_color_prob : ℚ := 1 / 3

/-- The function that calculates the probability of selecting two pens of the same color
    given the number of black pens and total pens -/
def calc_prob (black_pens total_pens : ℕ) : ℚ :=
  let red_pens := total_pens - black_pens
  (black_pens * (black_pens - 1) + red_pens * (red_pens - 1)) / (total_pens * (total_pens - 1))

theorem max_black_pens :
  ∃ (total_pens : ℕ) (black_pens : ℕ),
    total_pens ≤ max_pens ∧
    calc_prob black_pens total_pens = same_color_prob ∧
    black_pens = 1275 ∧
    ∀ (t : ℕ) (b : ℕ),
      t ≤ max_pens →
      calc_prob b t = same_color_prob →
      b ≤ 1275 :=
by sorry

end NUMINAMATH_CALUDE_max_black_pens_l3363_336349


namespace NUMINAMATH_CALUDE_scaled_right_triangle_area_l3363_336300

theorem scaled_right_triangle_area :
  ∀ (a b : ℝ) (scale : ℝ),
    a = 50 →
    b = 70 →
    scale = 2 →
    (1/2 : ℝ) * (a * scale) * (b * scale) = 7000 := by
  sorry

end NUMINAMATH_CALUDE_scaled_right_triangle_area_l3363_336300


namespace NUMINAMATH_CALUDE_total_cost_with_tax_l3363_336358

def nike_price : ℝ := 150
def boots_price : ℝ := 120
def tax_rate : ℝ := 0.1

theorem total_cost_with_tax :
  let pre_tax_total := nike_price + boots_price
  let tax_amount := pre_tax_total * tax_rate
  let total_with_tax := pre_tax_total + tax_amount
  total_with_tax = 297 := by sorry

end NUMINAMATH_CALUDE_total_cost_with_tax_l3363_336358


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3363_336336

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (A ∪ B a b = univ) → (A ∩ B a b = Ioc 3 4) → a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3363_336336


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3363_336309

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = 2}
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

-- State the theorem
theorem subset_implies_a_values (a : ℝ) :
  N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3363_336309


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l3363_336356

/-- Given a rectangular solid with adjacent face areas of 2, 3, and 6, 
    and all vertices lying on a sphere, the surface area of this sphere is 14π. -/
theorem sphere_surface_area_from_rectangular_solid (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 6 →
  b * c = 2 →
  a * c = 3 →
  (∃ (r : ℝ), r > 0 ∧ a^2 + b^2 + c^2 = (2*r)^2) →
  4 * π * ((a^2 + b^2 + c^2) / 4) = 14 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l3363_336356


namespace NUMINAMATH_CALUDE_point_on_parametric_curve_l3363_336371

theorem point_on_parametric_curve (θ : Real) (x y : Real) : 
  0 ≤ θ ∧ θ ≤ π →
  x = 3 * Real.cos θ →
  y = 4 * Real.sin θ →
  y / x = 1 →
  x = 12 / 5 ∧ y = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_point_on_parametric_curve_l3363_336371


namespace NUMINAMATH_CALUDE_marked_price_calculation_l3363_336357

/-- Given a pair of articles bought for $50 with a 30% discount, 
    prove that the marked price of each article is 50 / 1.4 -/
theorem marked_price_calculation (total_price : ℝ) (discount_percent : ℝ) 
    (h1 : total_price = 50)
    (h2 : discount_percent = 30) : 
  (total_price / (2 * (1 - discount_percent / 100))) = 50 / 1.4 := by
  sorry

#eval (50 : Float) / 1.4

end NUMINAMATH_CALUDE_marked_price_calculation_l3363_336357


namespace NUMINAMATH_CALUDE_matrix_determinant_from_eigenvectors_l3363_336340

/-- Given a 2x2 matrix A with specific eigenvectors and eigenvalues, prove that its determinant is -4 -/
theorem matrix_determinant_from_eigenvectors (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.mulVec ![1, -1] = (-1 : ℝ) • ![1, -1]) → 
  (A.mulVec ![3, 2] = 4 • ![3, 2]) → 
  a * d - b * c = -4 := by
  sorry


end NUMINAMATH_CALUDE_matrix_determinant_from_eigenvectors_l3363_336340


namespace NUMINAMATH_CALUDE_luke_finances_duration_l3363_336342

/-- Represents Luke's financial situation --/
structure LukeFinances where
  total_income : ℕ
  weekly_expenses : ℕ

/-- Calculates how many full weeks Luke's money will last --/
def weeks_money_lasts (finances : LukeFinances) : ℕ :=
  finances.total_income / finances.weekly_expenses

/-- Calculates the remaining money after the last full week --/
def remaining_money (finances : LukeFinances) : ℕ :=
  finances.total_income % finances.weekly_expenses

/-- Theorem stating how long Luke's money will last and how much will remain --/
theorem luke_finances_duration (finances : LukeFinances) 
  (h1 : finances.total_income = 34)
  (h2 : finances.weekly_expenses = 7) : 
  weeks_money_lasts finances = 4 ∧ remaining_money finances = 6 := by
  sorry

#eval weeks_money_lasts ⟨34, 7⟩
#eval remaining_money ⟨34, 7⟩

end NUMINAMATH_CALUDE_luke_finances_duration_l3363_336342


namespace NUMINAMATH_CALUDE_coupon_value_l3363_336305

/-- Calculates the value of a coupon for eyeglass frames -/
theorem coupon_value (frame_cost lens_cost insurance_percentage total_cost_after : ℚ) : 
  frame_cost = 200 →
  lens_cost = 500 →
  insurance_percentage = 80 / 100 →
  total_cost_after = 250 →
  (frame_cost + lens_cost * (1 - insurance_percentage)) - total_cost_after = 50 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l3363_336305


namespace NUMINAMATH_CALUDE_betty_herb_garden_l3363_336373

/-- The number of basil plants in Betty's herb garden -/
def basil : ℕ := 5

/-- The number of oregano plants in Betty's herb garden -/
def oregano : ℕ := 2 * basil + 2

/-- The total number of plants in Betty's herb garden -/
def total_plants : ℕ := basil + oregano

theorem betty_herb_garden :
  total_plants = 17 := by sorry

end NUMINAMATH_CALUDE_betty_herb_garden_l3363_336373


namespace NUMINAMATH_CALUDE_supremum_of_expression_l3363_336387

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -1 / (2 * a) - 2 / b ≤ -9/2 ∧ 
  ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 1 ∧ -1 / (2 * a') - 2 / b' = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_supremum_of_expression_l3363_336387


namespace NUMINAMATH_CALUDE_tangent_line_equation_f_positive_when_a_is_one_minimum_value_when_a_is_e_squared_l3363_336329

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 1 - (a * x^2) / Real.exp x

-- State the theorems to be proved
theorem tangent_line_equation (a : ℝ) :
  (∃ k, ∀ x, k * x + (f a 1 - k) = f a x + (deriv (f a)) 1 * (x - 1)) →
  ∃ k, k = 1 ∧ ∀ x, x + 1 = f a x + (deriv (f a)) 1 * (x - 1) :=
sorry

theorem f_positive_when_a_is_one :
  ∀ x > 0, f 1 x > 0 :=
sorry

theorem minimum_value_when_a_is_e_squared :
  (∃ x, f (Real.exp 2) x = -3) ∧ (∀ x, f (Real.exp 2) x ≥ -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_f_positive_when_a_is_one_minimum_value_when_a_is_e_squared_l3363_336329


namespace NUMINAMATH_CALUDE_johns_age_l3363_336315

theorem johns_age (j d m : ℕ) 
  (h1 : j = d - 20)
  (h2 : j = m - 15)
  (h3 : j + d = 80)
  (h4 : m = d + 5) :
  j = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l3363_336315


namespace NUMINAMATH_CALUDE_divisors_of_18m_squared_l3363_336386

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem divisors_of_18m_squared (m : ℕ) 
  (h1 : is_even m) 
  (h2 : num_divisors m = 9) : 
  num_divisors (18 * m^2) = 54 := by sorry

end NUMINAMATH_CALUDE_divisors_of_18m_squared_l3363_336386


namespace NUMINAMATH_CALUDE_num_divisors_360_eq_24_l3363_336327

/-- The number of positive divisors of 360 -/
def num_divisors_360 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 360 is 24 -/
theorem num_divisors_360_eq_24 : num_divisors_360 = 24 := by sorry

end NUMINAMATH_CALUDE_num_divisors_360_eq_24_l3363_336327


namespace NUMINAMATH_CALUDE_button_ratio_problem_l3363_336353

/-- Represents the button problem with Mark, Shane, and Sam -/
theorem button_ratio_problem (initial_buttons : ℕ) (shane_multiplier : ℕ) (final_buttons : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  let total_after_shane := initial_buttons + shane_multiplier * initial_buttons
  let sam_took := total_after_shane - final_buttons
  (sam_took : ℚ) / total_after_shane = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_button_ratio_problem_l3363_336353


namespace NUMINAMATH_CALUDE_complex_magnitude_l3363_336352

theorem complex_magnitude (z : ℂ) (h : z / (2 - Complex.I) = 2 * Complex.I) : 
  Complex.abs (z + 1) = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3363_336352


namespace NUMINAMATH_CALUDE_part_1_part_2_l3363_336392

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - 2*a + 1 < (1 - a) * x

-- Define the solution set for part (1)
def solution_set_1 (x : ℝ) : Prop := x < -4 ∨ x > 1

-- Define the condition for part (2)
def condition_2 (a : ℝ) : Prop := a > 0

-- Define the property of having exactly 7 prime elements in the solution set
def has_seven_primes (a : ℝ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 p6 p7 : ℕ),
    Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime p5 ∧ Prime p6 ∧ Prime p7 ∧
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧ p5 < p6 ∧ p6 < p7 ∧
    (∀ x : ℝ, inequality a x ↔ (x < p1 ∨ x > p7))

-- Theorem for part (1)
theorem part_1 : 
  (∀ x : ℝ, inequality a x ↔ solution_set_1 x) → a = -1/2 :=
sorry

-- Theorem for part (2)
theorem part_2 :
  condition_2 a → has_seven_primes a → 1/21 ≤ a ∧ a < 1/19 :=
sorry

end NUMINAMATH_CALUDE_part_1_part_2_l3363_336392


namespace NUMINAMATH_CALUDE_first_puncture_time_l3363_336310

/-- Given a tyre with two punctures, this theorem proves the time it takes
    for the first puncture alone to flatten the tyre. -/
theorem first_puncture_time
  (second_puncture_time : ℝ)
  (both_punctures_time : ℝ)
  (h1 : second_puncture_time = 6)
  (h2 : both_punctures_time = 336 / 60)
  (h3 : both_punctures_time > 0) :
  ∃ (first_puncture_time : ℝ),
    first_puncture_time > 0 ∧
    1 / first_puncture_time + 1 / second_puncture_time = 1 / both_punctures_time ∧
    first_puncture_time = 84 := by
  sorry

end NUMINAMATH_CALUDE_first_puncture_time_l3363_336310


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3363_336326

theorem inequality_solution_set (x : ℝ) :
  (2 * x) / (x - 2) ≤ 1 ↔ x ∈ Set.Icc (-2) 2 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3363_336326


namespace NUMINAMATH_CALUDE_first_digit_base_7_of_528_l3363_336303

/-- The first digit of the base 7 representation of a natural number -/
def first_digit_base_7 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 7).succ
    (n / 7^(k-1)) % 7

/-- Theorem: The first digit of the base 7 representation of 528 is 1 -/
theorem first_digit_base_7_of_528 :
  first_digit_base_7 528 = 1 := by sorry

end NUMINAMATH_CALUDE_first_digit_base_7_of_528_l3363_336303


namespace NUMINAMATH_CALUDE_winter_clothing_mittens_per_box_l3363_336316

theorem winter_clothing_mittens_per_box 
  (num_boxes : ℕ) 
  (scarves_per_box : ℕ) 
  (total_pieces : ℕ) 
  (h1 : num_boxes = 3)
  (h2 : scarves_per_box = 3)
  (h3 : total_pieces = 21) :
  (total_pieces - num_boxes * scarves_per_box) / num_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_winter_clothing_mittens_per_box_l3363_336316


namespace NUMINAMATH_CALUDE_average_bowling_score_l3363_336363

-- Define the players and their scores
def gretchen_score : ℕ := 120
def mitzi_score : ℕ := 113
def beth_score : ℕ := 85

-- Define the number of players
def num_players : ℕ := 3

-- Define the total score
def total_score : ℕ := gretchen_score + mitzi_score + beth_score

-- Theorem to prove
theorem average_bowling_score :
  (total_score : ℚ) / num_players = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_bowling_score_l3363_336363


namespace NUMINAMATH_CALUDE_linear_function_k_value_l3363_336354

/-- A linear function passing through a specific point -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

/-- The point through which the function passes -/
def point : ℝ × ℝ := (2, 5)

theorem linear_function_k_value :
  ∃ k : ℝ, linear_function k (point.1) = point.2 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l3363_336354


namespace NUMINAMATH_CALUDE_collapsible_iff_power_of_two_l3363_336301

/-- A token arrangement in the plane -/
structure TokenArrangement :=
  (n : ℕ+)  -- number of tokens
  (positions : Fin n → ℝ × ℝ)  -- positions of tokens in the plane

/-- Predicate for an arrangement being collapsible -/
def Collapsible (arrangement : TokenArrangement) : Prop :=
  ∃ (final_pos : ℝ × ℝ), ∀ i : Fin arrangement.n, 
    ∃ (moves : ℕ), arrangement.positions i = final_pos

/-- Predicate for a number being a power of 2 -/
def IsPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- The main theorem -/
theorem collapsible_iff_power_of_two :
  ∀ n : ℕ+, (∀ arrangement : TokenArrangement, arrangement.n = n → Collapsible arrangement) ↔ IsPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_collapsible_iff_power_of_two_l3363_336301


namespace NUMINAMATH_CALUDE_mixed_number_properties_l3363_336397

theorem mixed_number_properties :
  let x : ℚ := -1 - 2/7
  (1 / x = -7/9) ∧
  (-x = 1 + 2/7) ∧
  (|x| = 1 + 2/7) :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_properties_l3363_336397


namespace NUMINAMATH_CALUDE_zoo_animal_types_l3363_336399

/-- The time (in minutes) it takes to see each animal type -/
def time_per_animal : ℕ := 6

/-- The number of new species added -/
def new_species : ℕ := 4

/-- The total time (in minutes) to see all animal types after adding new species -/
def total_time : ℕ := 54

/-- The initial number of animal types at the zoo -/
def initial_types : ℕ := 5

theorem zoo_animal_types :
  initial_types * time_per_animal + new_species * time_per_animal = total_time :=
sorry

end NUMINAMATH_CALUDE_zoo_animal_types_l3363_336399


namespace NUMINAMATH_CALUDE_post_check_time_l3363_336372

/-- Proves that given a pay rate of 25 cents per post and an hourly rate of $90, 
    the time taken to check a single post is 10 seconds. -/
theorem post_check_time 
  (pay_per_post : ℚ) 
  (hourly_rate : ℚ) 
  (seconds_per_hour : ℕ) :
  pay_per_post = 25 / 100 →
  hourly_rate = 90 →
  seconds_per_hour = 3600 →
  (seconds_per_hour : ℚ) / (hourly_rate / pay_per_post) = 10 :=
by sorry

end NUMINAMATH_CALUDE_post_check_time_l3363_336372


namespace NUMINAMATH_CALUDE_volume_of_T_l3363_336379

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p
    |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the volume of T is 32/3 -/
theorem volume_of_T : volume T = 32/3 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l3363_336379


namespace NUMINAMATH_CALUDE_locker_count_proof_l3363_336391

/-- The cost of each digit in cents -/
def digit_cost : ℚ := 3

/-- The total cost of labeling all lockers in dollars -/
def total_cost : ℚ := 771.90

/-- The number of lockers -/
def num_lockers : ℕ := 6369

/-- The cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let one_digit := (min n 9 : ℚ) * digit_cost / 100
  let two_digit := (min n 99 - min n 9 : ℚ) * 2 * digit_cost / 100
  let three_digit := (min n 999 - min n 99 : ℚ) * 3 * digit_cost / 100
  let four_digit := (min n 9999 - min n 999 : ℚ) * 4 * digit_cost / 100
  let five_digit := (n - min n 9999 : ℚ) * 5 * digit_cost / 100
  one_digit + two_digit + three_digit + four_digit + five_digit

theorem locker_count_proof :
  labeling_cost num_lockers = total_cost := by
  sorry

end NUMINAMATH_CALUDE_locker_count_proof_l3363_336391


namespace NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l3363_336311

theorem max_value_of_a : ∀ a : ℝ, 
  (∀ x : ℝ, x < a → x^2 - x - 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≥ a) →
  a ≤ -2 :=
by sorry

theorem max_value_is_negative_two : 
  ∃ a : ℝ, a = -2 ∧
  (∀ x : ℝ, x < a → x^2 - x - 6 > 0) ∧
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≥ a) ∧
  (∀ b : ℝ, b > a →
    (∃ x : ℝ, x < b ∧ x^2 - x - 6 ≤ 0) ∨
    (∀ x : ℝ, x^2 - x - 6 > 0 → x < b)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l3363_336311


namespace NUMINAMATH_CALUDE_bird_population_theorem_l3363_336351

/-- Represents the bird population in the nature reserve -/
structure BirdPopulation where
  total : ℝ
  hawks : ℝ
  paddyfield_warblers : ℝ
  kingfishers : ℝ

/-- Conditions for the bird population -/
def ValidBirdPopulation (bp : BirdPopulation) : Prop :=
  bp.total > 0 ∧
  bp.hawks = 0.3 * bp.total ∧
  bp.paddyfield_warblers = 0.4 * (bp.total - bp.hawks) ∧
  bp.kingfishers = 0.25 * bp.paddyfield_warblers

/-- Theorem stating the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
theorem bird_population_theorem (bp : BirdPopulation) 
  (h : ValidBirdPopulation bp) : 
  (bp.total - (bp.hawks + bp.paddyfield_warblers + bp.kingfishers)) / bp.total = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_bird_population_theorem_l3363_336351


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l3363_336396

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |6 - (250 : ℝ)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l3363_336396


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3363_336355

theorem smallest_n_for_inequality : 
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 15 → k ≥ n) ∧ 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3363_336355


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3363_336312

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 → b = 23 → a > 0 → b > 0 → s > 0 → 
  a + b > s → a + s > b → b + s > a → 
  ∃ n : ℕ, n = 60 ∧ ∀ m : ℕ, (a + b + s < m ∧ m < n) → False :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3363_336312


namespace NUMINAMATH_CALUDE_circle_problem_l3363_336385

-- Define the equation of the general circle
def general_circle (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*k*x - (2*k+6)*y - 2*k - 31 = 0

-- Define the specific circle E
def circle_E (x y : ℝ) : Prop :=
  (x+2)^2 + (y-1)^2 = 32

-- Theorem statement
theorem circle_problem :
  (∀ k : ℝ, general_circle k (-6) 5 ∧ general_circle k 2 (-3)) ∧
  (circle_E (-6) 5 ∧ circle_E 2 (-3)) ∧
  (∀ P : ℝ × ℝ, ¬circle_E P.1 P.2 →
    ∃ A B : ℝ × ℝ,
      circle_E A.1 A.2 ∧
      circle_E B.1 B.2 ∧
      (∀ X : ℝ × ℝ, circle_E X.1 X.2 →
        (P.1 - A.1) * (X.1 - A.1) + (P.2 - A.2) * (X.2 - A.2) = 0 ∧
        (P.1 - B.1) * (X.1 - B.1) + (P.2 - B.2) * (X.2 - B.2) = 0) ∧
      ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) ≥ 64 * Real.sqrt 2 - 96)) :=
sorry

end NUMINAMATH_CALUDE_circle_problem_l3363_336385


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3363_336376

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3363_336376


namespace NUMINAMATH_CALUDE_find_d_l3363_336375

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 3

-- State the theorem
theorem find_d (c : ℝ) (d : ℝ) :
  (∀ x, f c (g c x) = 15 * x + d) → d = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l3363_336375


namespace NUMINAMATH_CALUDE_valid_intersection_numbers_l3363_336384

/-- A circle with arcs that intersect each other. -/
structure CircleWithArcs where
  num_arcs : ℕ
  intersections_per_arc : ℕ

/-- Predicate to check if a number is not a multiple of 8. -/
def not_multiple_of_eight (n : ℕ) : Prop :=
  n % 8 ≠ 0

/-- Theorem stating the conditions for valid intersection numbers in a circle with 100 arcs. -/
theorem valid_intersection_numbers (circle : CircleWithArcs) :
    circle.num_arcs = 100 →
    1 ≤ circle.intersections_per_arc ∧
    circle.intersections_per_arc ≤ 99 ∧
    not_multiple_of_eight (circle.intersections_per_arc + 1) :=
by sorry

end NUMINAMATH_CALUDE_valid_intersection_numbers_l3363_336384


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3363_336364

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 10) :
  let a := S * (1 - r)
  a = 40/3 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3363_336364


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3363_336307

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 41040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3363_336307


namespace NUMINAMATH_CALUDE_perfect_fit_R_squared_eq_one_l3363_336374

/-- A structure representing a set of observations in a linear regression model. -/
structure LinearRegressionData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  e : Fin n → ℝ

/-- The coefficient of determination (R-squared) for a linear regression model. -/
def R_squared (data : LinearRegressionData) : ℝ := sorry

/-- Theorem stating that if all error terms are zero, then R-squared equals 1. -/
theorem perfect_fit_R_squared_eq_one (data : LinearRegressionData) 
  (h1 : ∀ i, data.y i = data.b * data.x i + data.a + data.e i)
  (h2 : ∀ i, data.e i = 0) :
  R_squared data = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_fit_R_squared_eq_one_l3363_336374


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3363_336317

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1076 ∧ m = 23) :
  (5 : ℕ) = Nat.minFac ((n + 5) % m) :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3363_336317
