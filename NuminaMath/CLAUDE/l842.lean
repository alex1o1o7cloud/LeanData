import Mathlib

namespace NUMINAMATH_CALUDE_inequality_implies_range_l842_84287

/-- The inequality condition for all x > 1 -/
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 1, a * (x - 1) ≥ Real.log (x - 1)

/-- The range of a satisfying the inequality condition -/
def a_range (a : ℝ) : Prop :=
  a ≥ 1 / Real.exp 1

theorem inequality_implies_range :
  ∀ a : ℝ, inequality_condition a → a_range a :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l842_84287


namespace NUMINAMATH_CALUDE_principal_calculation_l842_84284

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest / (rate * time)

/-- Theorem: Given the specified conditions, the principal is 44625 -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 1 / 100  -- 1% converted to decimal
  let time : ℕ := 9
  calculate_principal simple_interest rate time = 44625 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l842_84284


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l842_84283

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve (m k a b c d : ℝ) :
  b = m * a^2 + k →
  d = m * c^2 + k →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |c - a| * Real.sqrt (1 + m^2 * (c + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l842_84283


namespace NUMINAMATH_CALUDE_problem_solution_l842_84285

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (a+2)*x + 4

-- Define the function g
def g (m x : ℝ) : ℝ := m*x + 5 - 2*m

theorem problem_solution :
  -- Part 1
  (∀ a : ℝ, 
    (a < 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | a ≤ x ∧ x ≤ 2}) ∧
    (a = 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | x = 2}) ∧
    (a > 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | 2 ≤ x ∧ x ≤ a})) ∧
  -- Part 2
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 4 → f a x + a + 1 ≥ 0) → a ≤ 4) ∧
  -- Part 3
  (∀ m : ℝ, 
    (∀ x₁ : ℝ, x₁ ∈ Set.Icc 1 4 → 
      ∃ x₂ : ℝ, x₂ ∈ Set.Icc 1 4 ∧ f 2 x₁ = g m x₂) →
    m ≤ -5/2 ∨ m ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l842_84285


namespace NUMINAMATH_CALUDE_age_ratio_l842_84253

def cody_age : ℕ := 14
def grandmother_age : ℕ := 84

theorem age_ratio : (grandmother_age : ℚ) / (cody_age : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l842_84253


namespace NUMINAMATH_CALUDE_floor_plus_double_eq_15_4_l842_84234

theorem floor_plus_double_eq_15_4 :
  ∃! r : ℝ, ⌊r⌋ + 2 * r = 15.4 ∧ r = 5.2 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_double_eq_15_4_l842_84234


namespace NUMINAMATH_CALUDE_divisibility_of_factorials_l842_84238

theorem divisibility_of_factorials (n : ℕ+) : 
  ∃ k : ℤ, 2 * (3 * n.val).factorial = k * n.val.factorial * (n.val + 1).factorial * (n.val + 2).factorial := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_factorials_l842_84238


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l842_84209

theorem max_sum_of_factors (p q : ℕ+) (h : p * q = 100) : 
  ∃ (a b : ℕ+), a * b = 100 ∧ a + b ≤ p + q ∧ a + b = 101 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l842_84209


namespace NUMINAMATH_CALUDE_max_gross_profit_l842_84220

/-- Represents the daily sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -x + 26

/-- Represents the gross profit as a function of selling price -/
def gross_profit (x : ℝ) : ℝ := x * (sales_volume x) - 4 * (sales_volume x)

/-- Theorem stating the maximum gross profit under given constraints -/
theorem max_gross_profit :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, 6 ≤ x ∧ x ≤ 12 ∧ sales_volume x ≤ 10 → gross_profit x ≤ max_profit) ∧
    (∃ x : ℝ, 6 ≤ x ∧ x ≤ 12 ∧ sales_volume x ≤ 10 ∧ gross_profit x = max_profit) ∧
    max_profit = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_gross_profit_l842_84220


namespace NUMINAMATH_CALUDE_mouse_breeding_problem_l842_84221

theorem mouse_breeding_problem (initial_mice : ℕ) (first_round_pups : ℕ) (eaten_pups : ℕ) (final_mice : ℕ) :
  initial_mice = 8 →
  first_round_pups = 6 →
  eaten_pups = 2 →
  final_mice = 280 →
  ∃ (second_round_pups : ℕ),
    final_mice = initial_mice + initial_mice * first_round_pups +
      (initial_mice + initial_mice * first_round_pups) * second_round_pups -
      (initial_mice + initial_mice * first_round_pups) * eaten_pups ∧
    second_round_pups = 6 :=
by sorry

end NUMINAMATH_CALUDE_mouse_breeding_problem_l842_84221


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l842_84295

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -6)
  parallel a b → x = -4 :=
by
  sorry

#check parallel_vectors_x_value

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l842_84295


namespace NUMINAMATH_CALUDE_tv_price_change_l842_84254

theorem tv_price_change (P : ℝ) (x : ℝ) : 
  (P - (x / 100) * P) * (1 + 30 / 100) = P * (1 + 4 / 100) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l842_84254


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l842_84244

/-- Represents a hyperbola with equation y²/a² - x²/4 = 1 -/
structure Hyperbola where
  a : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  (1 / h.a^2 - 4 / 4 = 1) →  -- The hyperbola passes through (2, -1)
  eccentricity h = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l842_84244


namespace NUMINAMATH_CALUDE_ellipse_arithmetic_sequence_eccentricity_l842_84233

/-- An ellipse with focal length, minor axis length, and major axis length in arithmetic sequence has eccentricity 3/5 -/
theorem ellipse_arithmetic_sequence_eccentricity :
  ∀ (a b c : ℝ),
    a > b ∧ b > 0 →  -- Conditions for a valid ellipse
    b = (a + c) / 2 →  -- Arithmetic sequence condition
    c^2 = a^2 - b^2 →  -- Relation between focal length and axes lengths
    c / a = 3 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_arithmetic_sequence_eccentricity_l842_84233


namespace NUMINAMATH_CALUDE_remainder_after_adding_4500_l842_84211

theorem remainder_after_adding_4500 (n : ℤ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_4500_l842_84211


namespace NUMINAMATH_CALUDE_wuzhen_conference_impact_l842_84230

/-- Represents the cultural impact of the World Internet Conference in Wuzhen -/
structure CulturalImpact where
  promote_chinese_culture : Bool
  innovate_world_culture : Bool
  enhance_chinese_influence : Bool

/-- The World Internet Conference venue -/
def Wuzhen : String := "Wuzhen, China"

/-- Characteristics of Wuzhen -/
structure WuzhenCharacteristics where
  tradition_modernity_blend : Bool
  chinese_foreign_embrace : Bool

/-- The cultural impact of the World Internet Conference -/
def conference_impact (venue : String) (characteristics : WuzhenCharacteristics) : CulturalImpact :=
  { promote_chinese_culture := true,
    innovate_world_culture := true,
    enhance_chinese_influence := true }

/-- Theorem stating the cultural impact of the World Internet Conference in Wuzhen -/
theorem wuzhen_conference_impact :
  let venue := Wuzhen
  let characteristics := { tradition_modernity_blend := true, chinese_foreign_embrace := true }
  let impact := conference_impact venue characteristics
  impact.promote_chinese_culture ∧ impact.innovate_world_culture ∧ impact.enhance_chinese_influence :=
by
  sorry

end NUMINAMATH_CALUDE_wuzhen_conference_impact_l842_84230


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l842_84263

theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1/2) * L * W) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l842_84263


namespace NUMINAMATH_CALUDE_area_above_x_axis_half_total_l842_84240

-- Define the parallelogram PQRS
def P : ℝ × ℝ := (4, 4)
def Q : ℝ × ℝ := (-2, -2)
def R : ℝ × ℝ := (-8, -2)
def S : ℝ × ℝ := (-2, 4)

-- Define a function to calculate the area of a parallelogram
def parallelogramArea (a b c d : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the area of the part of the parallelogram above the x-axis
def areaAboveXAxis (a b c d : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_above_x_axis_half_total : 
  areaAboveXAxis P Q R S = (1/2) * parallelogramArea P Q R S := by sorry

end NUMINAMATH_CALUDE_area_above_x_axis_half_total_l842_84240


namespace NUMINAMATH_CALUDE_angle_measure_proof_l842_84271

theorem angle_measure_proof :
  ∀ (A B : ℝ),
  A + B = 180 →
  A = 5 * B →
  A = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l842_84271


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_negative_l842_84260

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing_negative : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_negative_l842_84260


namespace NUMINAMATH_CALUDE_candies_problem_l842_84223

theorem candies_problem (n : ℕ) (a : ℕ) (h1 : n > 0) (h2 : a > 1) 
  (h3 : ∀ i : Fin n, a = n * a - a - 7) : n * a = 21 := by
  sorry

end NUMINAMATH_CALUDE_candies_problem_l842_84223


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l842_84227

theorem correct_average_after_error_correction (n : ℕ) (incorrect_sum correct_sum : ℝ) :
  n = 10 →
  incorrect_sum = 46 * n →
  correct_sum = incorrect_sum + 50 →
  correct_sum / n = 51 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l842_84227


namespace NUMINAMATH_CALUDE_sequence_inequality_l842_84294

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) ↔ k > -3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l842_84294


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_cube_condition_nonzero_sum_of_squares_iff_not_both_zero_l842_84213

-- Statement 1
theorem necessary_not_sufficient_cube_condition (x : ℝ) :
  (x^3 = -27 → x^2 = 9) ∧ ¬(x^2 = 9 → x^3 = -27) :=
sorry

-- Statement 2
theorem nonzero_sum_of_squares_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_cube_condition_nonzero_sum_of_squares_iff_not_both_zero_l842_84213


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l842_84298

/-- Given a tetrahedron with vertices A₁, A₂, A₃, A₄ in ℝ³ -/
def A₁ : ℝ × ℝ × ℝ := (-1, -5, 2)
def A₂ : ℝ × ℝ × ℝ := (-6, 0, -3)
def A₃ : ℝ × ℝ × ℝ := (3, 6, -3)
def A₄ : ℝ × ℝ × ℝ := (-10, 6, 7)

/-- Calculate the volume of the tetrahedron -/
def tetrahedron_volume (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the height from A₄ to face A₁A₂A₃ -/
def tetrahedron_height (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem tetrahedron_properties :
  tetrahedron_volume A₁ A₂ A₃ A₄ = 190 ∧
  tetrahedron_height A₁ A₂ A₃ A₄ = 2 * Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l842_84298


namespace NUMINAMATH_CALUDE_math_club_team_selection_l842_84212

def mathClubSize : ℕ := 15
def teamSize : ℕ := 5

theorem math_club_team_selection :
  Nat.choose mathClubSize teamSize = 3003 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l842_84212


namespace NUMINAMATH_CALUDE_min_max_m_l842_84281

theorem min_max_m (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (eq1 : 3 * a + 2 * b + c = 5) (eq2 : 2 * a + b - 3 * c = 1) :
  let m := 3 * a + b - 7 * c
  ∃ (m_min m_max : ℝ), (∀ m', m' = m → m' ≥ m_min) ∧
                       (∀ m', m' = m → m' ≤ m_max) ∧
                       m_min = -5/7 ∧ m_max = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_min_max_m_l842_84281


namespace NUMINAMATH_CALUDE_cube_root_abs_sqrt_equality_l842_84243

theorem cube_root_abs_sqrt_equality : 
  (64 : ℝ)^(1/3) - |Real.sqrt 3 - 3| + Real.sqrt 36 = 7 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_abs_sqrt_equality_l842_84243


namespace NUMINAMATH_CALUDE_chess_tournament_games_l842_84247

/-- The number of games in a chess tournament --/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem stating the number of games in the specific tournament --/
theorem chess_tournament_games :
  num_games 20 3 = 570 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l842_84247


namespace NUMINAMATH_CALUDE_vacation_miles_per_day_l842_84259

theorem vacation_miles_per_day 
  (vacation_days : ℝ) 
  (total_miles : ℝ) 
  (h1 : vacation_days = 5.0) 
  (h2 : total_miles = 1250) : 
  total_miles / vacation_days = 250 := by
sorry

end NUMINAMATH_CALUDE_vacation_miles_per_day_l842_84259


namespace NUMINAMATH_CALUDE_sum_of_decimals_l842_84290

theorem sum_of_decimals : 0.305 + 0.089 + 0.007 = 0.401 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l842_84290


namespace NUMINAMATH_CALUDE_boys_in_third_group_l842_84210

/-- Represents the work rate of a single person --/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers --/
structure WorkGroup where
  boys : ℕ
  girls : ℕ

/-- Calculates the total work done by a group in a given number of days --/
def totalWork (group : WorkGroup) (boyRate girlRate : WorkRate) (days : ℕ) : ℝ :=
  (group.boys : ℝ) * boyRate.rate * (days : ℝ) + (group.girls : ℝ) * girlRate.rate * (days : ℝ)

/-- The main theorem stating that the number of boys in the third group is 26 --/
theorem boys_in_third_group : 
  ∀ (x : ℕ) (boyRate girlRate : WorkRate),
  let group1 := WorkGroup.mk x 20
  let group2 := WorkGroup.mk 6 8
  let group3 := WorkGroup.mk 26 48
  totalWork group1 boyRate girlRate 4 = totalWork group2 boyRate girlRate 10 ∧
  totalWork group1 boyRate girlRate 4 = totalWork group3 boyRate girlRate 2 →
  group3.boys = 26 := by
sorry

end NUMINAMATH_CALUDE_boys_in_third_group_l842_84210


namespace NUMINAMATH_CALUDE_speed_conversion_l842_84261

theorem speed_conversion (speed_ms : ℝ) (speed_kmh : ℝ) : 
  speed_ms = 9/36 → speed_kmh = 0.9 → (1 : ℝ) * 3.6 = speed_kmh / speed_ms :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l842_84261


namespace NUMINAMATH_CALUDE_remaining_volume_of_cube_with_hole_l842_84280

/-- The remaining volume of a cube with a square hole cut through its center -/
theorem remaining_volume_of_cube_with_hole (cube_side : ℝ) (hole_side : ℝ) : 
  cube_side = 8 → hole_side = 4 → 
  cube_side ^ 3 - (hole_side ^ 2 * cube_side) = 384 := by
  sorry

end NUMINAMATH_CALUDE_remaining_volume_of_cube_with_hole_l842_84280


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l842_84231

theorem cubic_expression_evaluation :
  1001^3 - 1000 * 1001^2 - 1000^2 * 1001 + 1000^3 = 2001 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l842_84231


namespace NUMINAMATH_CALUDE_valid_numbers_l842_84217

def is_valid_number (n : ℕ) : Prop :=
  30 ∣ n ∧ (Finset.card (Nat.divisors n) = 30)

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {11250, 4050, 7500, 1620, 1200, 720} := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l842_84217


namespace NUMINAMATH_CALUDE_parabola_vertex_first_quadrant_l842_84262

/-- A parabola with equation y = (x-m)^2 + (m-1) has its vertex in the first quadrant if and only if m > 1 -/
theorem parabola_vertex_first_quadrant (m : ℝ) : 
  (∃ x y : ℝ, y = (x - m)^2 + (m - 1) ∧ x = m ∧ y = m - 1 ∧ x > 0 ∧ y > 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_first_quadrant_l842_84262


namespace NUMINAMATH_CALUDE_sum_floor_equals_217_l842_84268

theorem sum_floor_equals_217 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_squares : x^2 + y^2 = 4050 ∧ z^2 + w^2 = 4050)
  (products : x*z = 2040 ∧ y*w = 2040) : 
  ⌊x + y + z + w⌋ = 217 := by
sorry

end NUMINAMATH_CALUDE_sum_floor_equals_217_l842_84268


namespace NUMINAMATH_CALUDE_green_garden_yield_l842_84273

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * feet_per_step * (width_steps : ℝ) * feet_per_step * yield_per_sqft

/-- Proves that Mr. Green's garden yields the expected amount of potatoes -/
theorem green_garden_yield : 
  expected_potato_yield 18 25 3 (3/4) = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_green_garden_yield_l842_84273


namespace NUMINAMATH_CALUDE_product_as_sum_of_squares_l842_84245

theorem product_as_sum_of_squares : 
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^ 2 := by
  sorry

end NUMINAMATH_CALUDE_product_as_sum_of_squares_l842_84245


namespace NUMINAMATH_CALUDE_root_minus_one_implies_k_equals_minus_two_l842_84232

theorem root_minus_one_implies_k_equals_minus_two (k : ℝ) :
  ((-1 : ℝ)^2 - k*(-1) + 1 = 0) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_minus_one_implies_k_equals_minus_two_l842_84232


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l842_84278

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_point_one (x y : ℝ) :
  f 1 = -1 →
  f' 1 = 1 →
  (y - f 1 = f' 1 * (x - 1)) ↔ x - y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l842_84278


namespace NUMINAMATH_CALUDE_area_to_paint_l842_84218

-- Define the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_width : ℝ := 5
def door_height : ℝ := 2
def door_width : ℝ := 7

-- Define the theorem
theorem area_to_paint :
  wall_height * wall_length - (window_height * window_width + door_height * door_width) = 121 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_l842_84218


namespace NUMINAMATH_CALUDE_notebook_price_is_3_l842_84251

-- Define the prices as real numbers
variable (pencil_price notebook_price : ℝ)

-- Define the purchase equations
def xiaohong_purchase : Prop :=
  4 * pencil_price + 5 * notebook_price = 15.8

def xiaoliang_purchase : Prop :=
  4 * pencil_price + 7 * notebook_price = 21.8

-- Theorem statement
theorem notebook_price_is_3
  (h1 : xiaohong_purchase pencil_price notebook_price)
  (h2 : xiaoliang_purchase pencil_price notebook_price) :
  notebook_price = 3 := by sorry

end NUMINAMATH_CALUDE_notebook_price_is_3_l842_84251


namespace NUMINAMATH_CALUDE_special_sequence_growth_l842_84237

/-- A sequence of positive integers satisfying the given condition -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ i ≥ 1, Nat.gcd (a i) (a (i + 1)) > a (i - 1))

/-- The main theorem: for any special sequence, each term is at least 2^n -/
theorem special_sequence_growth (a : ℕ → ℕ) (h : SpecialSequence a) :
    ∀ n, a n ≥ 2^n := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_growth_l842_84237


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_inequality_solution_3_l842_84225

-- Define the functions for each inequality
def f₁ (x : ℝ) := (x - 2)^11 * (x + 1)^22 * (x + 3)^33
def f₂ (x : ℝ) := (4*x + 3)^5 * (3*x + 2)^3 * (2*x + 1)
def f₃ (x : ℝ) := (x + 3) * (x + 1)^2 * (x - 2)^3 * (x - 4)

-- Define the solution sets
def S₁ : Set ℝ := {x | x ∈ (Set.Ioo (-3) (-1)) ∪ (Set.Ioo (-1) 2)}
def S₂ : Set ℝ := {x | x ∈ (Set.Iic (-3/4)) ∪ (Set.Icc (-2/3) (-1/2))}
def S₃ : Set ℝ := {x | x ∈ (Set.Iic (-3)) ∪ {-1} ∪ (Set.Icc 2 4)}

-- State the theorems
theorem inequality_solution_1 : {x : ℝ | f₁ x < 0} = S₁ := by sorry

theorem inequality_solution_2 : {x : ℝ | f₂ x ≤ 0} = S₂ := by sorry

theorem inequality_solution_3 : {x : ℝ | f₃ x ≤ 0} = S₃ := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_inequality_solution_3_l842_84225


namespace NUMINAMATH_CALUDE_noah_in_middle_chair_l842_84201

/- Define the friends as an enumeration -/
inductive Friend
| Liam
| Noah
| Olivia
| Emma
| Sophia

/- Define the seating arrangement as a function from chair number to Friend -/
def Seating := Fin 5 → Friend

def is_valid_seating (s : Seating) : Prop :=
  /- Sophia sits in the first chair -/
  s 1 = Friend.Sophia ∧
  /- Emma sits directly in front of Liam -/
  (∃ i : Fin 4, s i = Friend.Emma ∧ s (i + 1) = Friend.Liam) ∧
  /- Noah sits somewhere in front of Emma -/
  (∃ i j : Fin 5, i < j ∧ s i = Friend.Noah ∧ s j = Friend.Emma) ∧
  /- At least one person sits between Noah and Olivia -/
  (∃ i j k : Fin 5, i < j ∧ j < k ∧ s i = Friend.Noah ∧ s k = Friend.Olivia) ∧
  /- All friends are seated -/
  (∃ i : Fin 5, s i = Friend.Liam) ∧
  (∃ i : Fin 5, s i = Friend.Noah) ∧
  (∃ i : Fin 5, s i = Friend.Olivia) ∧
  (∃ i : Fin 5, s i = Friend.Emma) ∧
  (∃ i : Fin 5, s i = Friend.Sophia)

theorem noah_in_middle_chair (s : Seating) (h : is_valid_seating s) :
  s 3 = Friend.Noah :=
by sorry

end NUMINAMATH_CALUDE_noah_in_middle_chair_l842_84201


namespace NUMINAMATH_CALUDE_abc_product_l842_84258

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30 * Real.rpow 4 (1/3))
  (hac : a * c = 40 * Real.rpow 4 (1/3))
  (hbc : b * c = 24 * Real.rpow 4 (1/3)) :
  a * b * c = 120 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l842_84258


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l842_84267

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = -7)
  (h_s3 : sum_arithmetic_sequence a 3 = -15) :
  (∀ n : ℕ, a n = 2 * n - 9) ∧
  (∀ n : ℕ, sum_arithmetic_sequence a n = (n - 4)^2 - 16) ∧
  (∀ n : ℕ, sum_arithmetic_sequence a n ≥ -16) ∧
  (sum_arithmetic_sequence a 4 = -16) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l842_84267


namespace NUMINAMATH_CALUDE_lost_card_number_l842_84207

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Finset.range (n + 1)) : 
  (n * (n + 1)) / 2 - 101 = 4 := by
sorry

end NUMINAMATH_CALUDE_lost_card_number_l842_84207


namespace NUMINAMATH_CALUDE_puzzle_solution_l842_84214

theorem puzzle_solution :
  ∀ (E H O Y A : ℕ),
    (10 ≤ E * 10 + H) ∧ (E * 10 + H < 100) ∧
    (10 ≤ O * 10 + Y) ∧ (O * 10 + Y < 100) ∧
    (10 ≤ A * 10 + Y) ∧ (A * 10 + Y < 100) ∧
    (10 ≤ O * 10 + H) ∧ (O * 10 + H < 100) ∧
    (E * 10 + H = 4 * (O * 10 + Y)) ∧
    (A * 10 + Y = 4 * (O * 10 + H)) →
    (E * 10 + H) + (O * 10 + Y) + (A * 10 + Y) + (O * 10 + H) = 150 :=
by sorry


end NUMINAMATH_CALUDE_puzzle_solution_l842_84214


namespace NUMINAMATH_CALUDE_cats_not_liking_catnip_or_tuna_l842_84219

/-- Given a pet shop with cats, prove the number of cats that don't like catnip or tuna -/
theorem cats_not_liking_catnip_or_tuna
  (total_cats : ℕ)
  (cats_like_catnip : ℕ)
  (cats_like_tuna : ℕ)
  (cats_like_both : ℕ)
  (h1 : total_cats = 80)
  (h2 : cats_like_catnip = 15)
  (h3 : cats_like_tuna = 60)
  (h4 : cats_like_both = 10) :
  total_cats - (cats_like_catnip + cats_like_tuna - cats_like_both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_cats_not_liking_catnip_or_tuna_l842_84219


namespace NUMINAMATH_CALUDE_complex_power_simplification_l842_84229

theorem complex_power_simplification :
  ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 2000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l842_84229


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l842_84248

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l842_84248


namespace NUMINAMATH_CALUDE_provisions_duration_l842_84282

theorem provisions_duration (initial_soldiers : ℕ) (initial_consumption : ℚ)
  (additional_soldiers : ℕ) (new_consumption : ℚ) (new_duration : ℕ) :
  initial_soldiers = 1200 →
  initial_consumption = 3 →
  additional_soldiers = 528 →
  new_consumption = 5/2 →
  new_duration = 25 →
  (↑initial_soldiers * initial_consumption * ↑new_duration =
   ↑(initial_soldiers + additional_soldiers) * new_consumption * ↑new_duration) →
  (↑initial_soldiers * initial_consumption * (1080000 / 3600 : ℚ) =
   ↑initial_soldiers * initial_consumption * 300) :=
by sorry

end NUMINAMATH_CALUDE_provisions_duration_l842_84282


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l842_84293

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (1, Real.sqrt 3, 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l842_84293


namespace NUMINAMATH_CALUDE_solution_characterization_l842_84226

def divides (x y : ℤ) : Prop := ∃ k : ℤ, y = k * x

def is_solution (a b : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ 
  divides (2 * a + 1) (3 * b - 1) ∧
  divides (2 * b + 1) (3 * a - 1)

theorem solution_characterization :
  ∀ a b : ℕ, is_solution a b ↔ ((a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l842_84226


namespace NUMINAMATH_CALUDE_median_angle_relation_l842_84208

/-- Represents a triangle with sides a, b, c, angle γ opposite to side c, and median sc to side c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  γ : ℝ
  sc : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_γ : 0 < γ
  pos_sc : 0 < sc
  tri_ineq : a + b > c ∧ b + c > a ∧ c + a > b

theorem median_angle_relation (t : Triangle) :
  (t.γ < 90 ↔ t.sc > t.c / 2) ∧
  (t.γ = 90 ↔ t.sc = t.c / 2) ∧
  (t.γ > 90 ↔ t.sc < t.c / 2) :=
sorry

end NUMINAMATH_CALUDE_median_angle_relation_l842_84208


namespace NUMINAMATH_CALUDE_min_vertices_for_quadrilateral_l842_84200

theorem min_vertices_for_quadrilateral (n : ℕ) (hn : n ≥ 10) :
  let k := ⌊(3 * n : ℝ) / 4⌋ + 1
  ∀ S : Finset (Fin n),
    S.card ≥ k →
    ∃ (a b c d : Fin n), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
      ((b - a) % n = 1 ∨ (b - a) % n = n - 1) ∧
      ((c - b) % n = 1 ∨ (c - b) % n = n - 1) ∧
      ((d - c) % n = 1 ∨ (d - c) % n = n - 1) :=
by sorry

#check min_vertices_for_quadrilateral

end NUMINAMATH_CALUDE_min_vertices_for_quadrilateral_l842_84200


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l842_84286

theorem rectangular_prism_diagonal (length width height : ℝ) :
  length = 24 ∧ width = 16 ∧ height = 12 →
  Real.sqrt (length^2 + width^2 + height^2) = 4 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l842_84286


namespace NUMINAMATH_CALUDE_cube_volume_problem_l842_84249

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a - 2) * a * (a + 2) = a^3 - 16 → 
  a^3 = 64 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l842_84249


namespace NUMINAMATH_CALUDE_angela_action_figures_l842_84236

theorem angela_action_figures (initial : ℕ) (sold_fraction : ℚ) (given_fraction : ℚ) : 
  initial = 24 → 
  sold_fraction = 1/4 → 
  given_fraction = 1/3 → 
  initial - (initial * sold_fraction).floor - ((initial - (initial * sold_fraction).floor) * given_fraction).floor = 12 := by
  sorry

end NUMINAMATH_CALUDE_angela_action_figures_l842_84236


namespace NUMINAMATH_CALUDE_min_value_expression_l842_84289

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l842_84289


namespace NUMINAMATH_CALUDE_largest_number_proof_l842_84216

def is_valid_expression (expr : ℕ → ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    (a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 8 ∧ e = 8 ∧ f = 8) ∧
    (∀ n, expr n = n ∨ expr n = a ∨ expr n = b ∨ expr n = c ∨ expr n = d ∨ expr n = e ∨ expr n = f ∨
      ∃ (x y : ℕ), (expr n = expr x + expr y ∨ expr n = expr x - expr y ∨ 
                    expr n = expr x * expr y ∨ expr n = expr x / expr y ∨ 
                    expr n = expr x ^ expr y))

def largest_expression : ℕ → ℕ :=
  fun n => 3^(3^(3^(8^(8^8))))

theorem largest_number_proof :
  (is_valid_expression largest_expression) ∧
  (∀ expr, is_valid_expression expr → ∀ n, expr n ≤ largest_expression n) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_proof_l842_84216


namespace NUMINAMATH_CALUDE_tank_full_time_l842_84265

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  fill_rate_a : ℕ
  fill_rate_b : ℕ
  drain_rate : ℕ

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  let net_fill_per_cycle := system.fill_rate_a + system.fill_rate_b - system.drain_rate
  let cycles := system.capacity / net_fill_per_cycle
  cycles * 3

/-- Theorem stating that the tank will be full after 54 minutes -/
theorem tank_full_time (system : TankSystem) 
  (h1 : system.capacity = 900)
  (h2 : system.fill_rate_a = 40)
  (h3 : system.fill_rate_b = 30)
  (h4 : system.drain_rate = 20) :
  time_to_fill system = 54 := by
  sorry

#eval time_to_fill { capacity := 900, fill_rate_a := 40, fill_rate_b := 30, drain_rate := 20 }

end NUMINAMATH_CALUDE_tank_full_time_l842_84265


namespace NUMINAMATH_CALUDE_car_distance_formula_l842_84264

/-- The distance traveled by a car after time t -/
def distance (t : ℝ) : ℝ :=
  10 + 60 * t

/-- The initial distance traveled by the car -/
def initial_distance : ℝ := 10

/-- The constant speed of the car after the initial distance -/
def speed : ℝ := 60

theorem car_distance_formula (t : ℝ) :
  distance t = initial_distance + speed * t :=
by sorry

end NUMINAMATH_CALUDE_car_distance_formula_l842_84264


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l842_84275

theorem min_value_sum_squares (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 3*y₁ + 2*y₂ + y₃ = 90) : 
  y₁^2 + 4*y₂^2 + 9*y₃^2 ≥ 4050/7 ∧ 
  ∃ y₁' y₂' y₃', y₁'^2 + 4*y₂'^2 + 9*y₃'^2 = 4050/7 ∧ 
                 y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 
                 3*y₁' + 2*y₂' + y₃' = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l842_84275


namespace NUMINAMATH_CALUDE_two_true_propositions_l842_84277

theorem two_true_propositions :
  let P : ℝ → Prop := λ a => a > -3
  let Q : ℝ → Prop := λ a => a > -6
  let original := ∀ a, P a → Q a
  let converse := ∀ a, Q a → P a
  let inverse := ∀ a, ¬(P a) → ¬(Q a)
  let contrapositive := ∀ a, ¬(Q a) → ¬(P a)
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l842_84277


namespace NUMINAMATH_CALUDE_x_in_open_interval_one_two_l842_84246

/-- A monotonically increasing function on (0,+∞) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f x < f y

theorem x_in_open_interval_one_two
  (f : ℝ → ℝ)
  (h_mono : MonoIncreasing f)
  (h_gt : ∀ x, 0 < x → f x > f (2 - x)) :
  ∃ x, 1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_x_in_open_interval_one_two_l842_84246


namespace NUMINAMATH_CALUDE_employee_payment_l842_84205

theorem employee_payment (total : ℝ) (x y : ℝ) (h1 : total = 528) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 240 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_l842_84205


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l842_84224

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 25 →           -- Hypotenuse is 25
  a ≤ b →            -- a is the shorter leg
  (a = 7 ∨ a = 20) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l842_84224


namespace NUMINAMATH_CALUDE_ellipse_condition_l842_84276

/-- An equation represents an ellipse if it's of the form (x^2)/a + (y^2)/b = 1,
    where a and b are positive real numbers and a ≠ b. -/
def IsEllipse (m : ℝ) : Prop :=
  m > 0 ∧ 2*m - 1 > 0 ∧ m ≠ 2*m - 1

/-- If the equation (x^2)/m + (y^2)/(2m-1) = 1 represents an ellipse,
    then m > 1/2 and m ≠ 1. -/
theorem ellipse_condition (m : ℝ) :
  IsEllipse m → m > 1/2 ∧ m ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_condition_l842_84276


namespace NUMINAMATH_CALUDE_train_length_proof_l842_84239

theorem train_length_proof (passing_time : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  passing_time = 8 →
  platform_length = 279 →
  crossing_time = 20 →
  ∃ (train_length : ℝ),
    train_length = passing_time * (train_length + platform_length) / crossing_time ∧
    train_length = 186 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l842_84239


namespace NUMINAMATH_CALUDE_fold_five_cut_once_l842_84291

/-- The number of segments created by folding a rope n times and then cutting it once -/
def rope_segments (n : ℕ) : ℕ :=
  2^n + 1

/-- Theorem: Folding a rope 5 times and cutting it once results in 33 segments -/
theorem fold_five_cut_once : rope_segments 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fold_five_cut_once_l842_84291


namespace NUMINAMATH_CALUDE_five_digit_twice_divisible_by_11_l842_84296

theorem five_digit_twice_divisible_by_11 (a : ℕ) (h : 10000 ≤ a ∧ a < 100000) :
  ∃ k : ℕ, 100001 * a = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_five_digit_twice_divisible_by_11_l842_84296


namespace NUMINAMATH_CALUDE_final_number_is_odd_l842_84235

/-- Represents the operation of replacing two numbers with their difference -/
def replace_with_difference (s : Finset ℕ) : Finset ℕ := sorry

/-- The initial set of numbers on the board -/
def initial_numbers : Finset ℕ := Finset.range 1967 \ {0}

/-- Applies the replace_with_difference operation n times -/
def apply_n_times (s : Finset ℕ) (n : ℕ) : Finset ℕ :=
  match n with
  | 0 => s
  | n + 1 => replace_with_difference (apply_n_times s n)

theorem final_number_is_odd :
  ∃ (x : ℕ), (apply_n_times initial_numbers 1965).card = 1 ∧ Odd x ∧ x ∈ apply_n_times initial_numbers 1965 := by
  sorry

end NUMINAMATH_CALUDE_final_number_is_odd_l842_84235


namespace NUMINAMATH_CALUDE_min_value_constraint_min_value_achieved_l842_84299

theorem min_value_constraint (x y : ℝ) (h : 2 * x + 8 * y = 3) :
  x^2 + 4 * y^2 - 2 * x ≥ -19/20 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 2 * x + 8 * y = 3 ∧ x^2 + 4 * y^2 - 2 * x < -19/20 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_min_value_achieved_l842_84299


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l842_84255

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 4 + a 7 + a 10 = 30 →
  a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l842_84255


namespace NUMINAMATH_CALUDE_acid_dilution_l842_84222

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 → 
  initial_concentration = 0.4 → 
  final_concentration = 0.25 → 
  water_added = 30 → 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration := by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l842_84222


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l842_84270

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 4 ∧
  f x = 16 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-4 : ℝ) 4 → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l842_84270


namespace NUMINAMATH_CALUDE_cube_triangles_area_sum_l842_84241

/-- Represents a 3D point in a cube --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space --/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2x2x2 cube --/
def cubeVertices : List Point3D := sorry

/-- Calculates the area of a triangle in 3D space --/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- Generates all possible triangles from the cube vertices --/
def allTriangles : List Triangle3D := sorry

/-- Expresses a real number in the form m + √n + √p --/
structure SqrtForm where
  m : ℤ
  n : ℤ
  p : ℤ

/-- Converts a real number to SqrtForm --/
def toSqrtForm (r : ℝ) : SqrtForm := sorry

/-- The main theorem --/
theorem cube_triangles_area_sum :
  let totalArea := (allTriangles.map triangleArea).sum
  let sqrtForm := toSqrtForm totalArea
  sqrtForm.m + sqrtForm.n + sqrtForm.p = 121 := by sorry

end NUMINAMATH_CALUDE_cube_triangles_area_sum_l842_84241


namespace NUMINAMATH_CALUDE_kozel_garden_problem_l842_84202

theorem kozel_garden_problem (x : ℕ) (y : ℕ) : 
  (y = 3 * x + 1) → 
  (y = 4 * (x - 1)) → 
  (x = 5 ∧ y = 16) :=
by sorry

end NUMINAMATH_CALUDE_kozel_garden_problem_l842_84202


namespace NUMINAMATH_CALUDE_complex_absolute_value_problem_l842_84279

theorem complex_absolute_value_problem : 
  let z₁ : ℂ := 3 - 5*I
  let z₂ : ℂ := 3 + 5*I
  Complex.abs z₁ * Complex.abs z₂ + 2 * Complex.abs z₁ = 34 + 2 * Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_complex_absolute_value_problem_l842_84279


namespace NUMINAMATH_CALUDE_at_op_sum_equals_six_l842_84256

-- Define the @ operation for positive integers
def at_op (a b : ℕ+) : ℚ := (a * b : ℚ) / (a + b : ℚ)

-- State the theorem
theorem at_op_sum_equals_six :
  at_op 7 14 + at_op 2 4 = 6 := by sorry

end NUMINAMATH_CALUDE_at_op_sum_equals_six_l842_84256


namespace NUMINAMATH_CALUDE_class_fraction_proof_l842_84292

theorem class_fraction_proof (G : ℚ) (B : ℚ) (T : ℚ) (F : ℚ) :
  B / G = 7 / 3 →
  T = B + G →
  (2 / 3) * G = F * T →
  F = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_class_fraction_proof_l842_84292


namespace NUMINAMATH_CALUDE_max_students_per_classroom_l842_84250

/-- Theorem: Maximum students per classroom with equal gender distribution -/
theorem max_students_per_classroom 
  (num_classrooms : ℕ) 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (h1 : num_classrooms = 7)
  (h2 : num_boys = 68)
  (h3 : num_girls = 53) :
  ∃ (students_per_classroom : ℕ),
    students_per_classroom = 14 ∧
    students_per_classroom ≤ min num_boys num_girls ∧
    students_per_classroom % 2 = 0 ∧
    (students_per_classroom / 2) * num_classrooms ≤ min num_boys num_girls :=
by
  sorry

#check max_students_per_classroom

end NUMINAMATH_CALUDE_max_students_per_classroom_l842_84250


namespace NUMINAMATH_CALUDE_cross_placements_count_l842_84228

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)

/-- Represents a rectangle --/
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a cross shape --/
structure Cross :=
  (size : ℕ)

/-- Function to calculate the number of ways to place a cross in a grid with a rectangle removed --/
def count_cross_placements (g : Grid) (r : Rectangle) (c : Cross) : ℕ :=
  sorry

/-- Theorem stating the number of ways to place a 5-cell cross in a 40x40 grid with a 36x37 rectangle removed --/
theorem cross_placements_count :
  let g := Grid.mk 40
  let r := Rectangle.mk 36 37
  let c := Cross.mk 5
  count_cross_placements g r c = 113 := by
  sorry

end NUMINAMATH_CALUDE_cross_placements_count_l842_84228


namespace NUMINAMATH_CALUDE_theater_revenue_l842_84204

/-- The total number of tickets sold -/
def total_tickets : ℕ := 800

/-- The price of an advanced ticket in cents -/
def advanced_price : ℕ := 1450

/-- The price of a door ticket in cents -/
def door_price : ℕ := 2200

/-- The number of tickets sold at the door -/
def door_tickets : ℕ := 672

/-- The total money taken in cents -/
def total_money : ℕ := 1664000

theorem theater_revenue :
  total_money = 
    (total_tickets - door_tickets) * advanced_price +
    door_tickets * door_price :=
by sorry

end NUMINAMATH_CALUDE_theater_revenue_l842_84204


namespace NUMINAMATH_CALUDE_roses_cut_proof_l842_84272

/-- Given a vase with an initial number of roses and a final number of roses,
    calculate the number of roses that were added. -/
def roses_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 2 initial roses and 23 final roses,
    the number of roses added is 21. -/
theorem roses_cut_proof :
  roses_added 2 23 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_proof_l842_84272


namespace NUMINAMATH_CALUDE_power_two_divides_odd_power_minus_one_l842_84269

theorem power_two_divides_odd_power_minus_one (k : ℕ) (h : Odd k) :
  ∀ n : ℕ, n ≥ 1 → (2^(n+2) : ℕ) ∣ k^(2^n) - 1 :=
by sorry

end NUMINAMATH_CALUDE_power_two_divides_odd_power_minus_one_l842_84269


namespace NUMINAMATH_CALUDE_ratio_difference_theorem_l842_84257

theorem ratio_difference_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / b = 2 / 3 → (a + 4) / (b + 4) = 5 / 7 → b - a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_theorem_l842_84257


namespace NUMINAMATH_CALUDE_binary_11101_equals_29_l842_84206

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11101_equals_29 :
  binary_to_decimal [true, false, true, true, true] = 29 := by
  sorry

end NUMINAMATH_CALUDE_binary_11101_equals_29_l842_84206


namespace NUMINAMATH_CALUDE_function_condition_implies_a_range_l842_84297

/-- Given a function f and a positive real number a, proves that if the given condition holds, then a ≥ 1 -/
theorem function_condition_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ (x : ℝ), x > 0 → ∃ (f : ℝ → ℝ), f x = a * Real.log x + (1/2) * x^2) →
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (a * Real.log x₁ + (1/2) * x₁^2 - (a * Real.log x₂ + (1/2) * x₂^2)) / (x₁ - x₂) ≥ 2) →
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_condition_implies_a_range_l842_84297


namespace NUMINAMATH_CALUDE_sqrt_280_between_16_and_17_l842_84203

theorem sqrt_280_between_16_and_17 : 16 < Real.sqrt 280 ∧ Real.sqrt 280 < 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_280_between_16_and_17_l842_84203


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l842_84215

theorem largest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l842_84215


namespace NUMINAMATH_CALUDE_base3_12012_equals_140_l842_84288

def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base3_12012_equals_140 :
  base3ToBase10 [2, 1, 0, 2, 1] = 140 := by
  sorry

end NUMINAMATH_CALUDE_base3_12012_equals_140_l842_84288


namespace NUMINAMATH_CALUDE_cube_root_simplification_l842_84274

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 : ℝ)^(1/3) = 10 * 99^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l842_84274


namespace NUMINAMATH_CALUDE_total_emails_received_l842_84266

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 5

theorem total_emails_received : morning_emails + afternoon_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_emails_received_l842_84266


namespace NUMINAMATH_CALUDE_train_meeting_theorem_l842_84242

/-- Represents the meeting point of three trains given their speeds and departure times. -/
structure TrainMeeting where
  speed_a : ℝ
  speed_b : ℝ
  speed_c : ℝ
  time_b_after_a : ℝ
  time_c_after_b : ℝ

/-- Calculates the meeting point of three trains. -/
def calculate_meeting_point (tm : TrainMeeting) : ℝ × ℝ := sorry

/-- Theorem stating the correct speed of Train C and the meeting distance. -/
theorem train_meeting_theorem (tm : TrainMeeting) 
  (h1 : tm.speed_a = 30)
  (h2 : tm.speed_b = 36)
  (h3 : tm.time_b_after_a = 2)
  (h4 : tm.time_c_after_b = 1) :
  let (speed_c, distance) := calculate_meeting_point tm
  speed_c = 45 ∧ distance = 180 := by sorry

end NUMINAMATH_CALUDE_train_meeting_theorem_l842_84242


namespace NUMINAMATH_CALUDE_group_a_trees_l842_84252

theorem group_a_trees (group_a_plots : ℕ) (group_b_plots : ℕ) : 
  (4 * group_a_plots = 5 * group_b_plots) →  -- Both groups planted the same total number of trees
  (group_b_plots = group_a_plots - 3) →      -- Group B worked on 3 fewer plots than Group A
  (4 * group_a_plots = 60) :=                -- Group A planted 60 trees in total
by
  sorry

#check group_a_trees

end NUMINAMATH_CALUDE_group_a_trees_l842_84252
