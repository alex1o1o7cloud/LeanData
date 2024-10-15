import Mathlib

namespace NUMINAMATH_CALUDE_donna_weekly_episodes_l22_2295

/-- The number of episodes Donna can watch on a weekday -/
def weekday_episodes : ℕ := 8

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The multiplier for weekend episodes compared to weekday episodes -/
def weekend_multiplier : ℕ := 3

/-- The total number of episodes Donna can watch in a week -/
def total_episodes_per_week : ℕ :=
  weekday_episodes * weekdays_per_week +
  (weekday_episodes * weekend_multiplier) * weekend_days_per_week

theorem donna_weekly_episodes :
  total_episodes_per_week = 88 := by
  sorry


end NUMINAMATH_CALUDE_donna_weekly_episodes_l22_2295


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l22_2208

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) 
  (h : Real.sin (α - π/6) = 1/3) : 
  Real.cos ((2*π)/3 - α) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l22_2208


namespace NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l22_2250

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l22_2250


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l22_2286

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (k - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l22_2286


namespace NUMINAMATH_CALUDE_binomial_9_8_l22_2217

theorem binomial_9_8 : Nat.choose 9 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_8_l22_2217


namespace NUMINAMATH_CALUDE_inequality_solution_l22_2261

theorem inequality_solution (x : ℕ) (h : x > 1) :
  (6 * (9 ^ (1 / x)) - 13 * (3 ^ (1 / x)) * (2 ^ (1 / x)) + 6 * (4 ^ (1 / x)) ≤ 0) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l22_2261


namespace NUMINAMATH_CALUDE_grocery_stock_problem_l22_2254

theorem grocery_stock_problem (asparagus_bundles : ℕ) (asparagus_price : ℚ)
  (grape_boxes : ℕ) (grape_price : ℚ)
  (apple_price : ℚ) (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  grape_price = (5/2) →
  apple_price = (1/2) →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + grape_boxes * grape_price)) / apple_price = 700 := by
  sorry

end NUMINAMATH_CALUDE_grocery_stock_problem_l22_2254


namespace NUMINAMATH_CALUDE_triangle_properties_l22_2277

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.b^2 + t.c^2 - t.a^2) / Real.cos t.A = 2)
  (h2 : (t.a * Real.cos t.B - t.b * Real.cos t.A) / (t.a * Real.cos t.B + t.b * Real.cos t.A) - t.b / t.c = 1) :
  t.b * t.c = 1 ∧ 
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l22_2277


namespace NUMINAMATH_CALUDE_equation_equivalent_to_two_lines_l22_2294

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 2*y)^2 = x^2 + y^2

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 0
def line2 (x y : ℝ) : Prop := y = (4/3) * x

-- Theorem statement
theorem equation_equivalent_to_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_two_lines_l22_2294


namespace NUMINAMATH_CALUDE_adams_earnings_l22_2288

/-- Adam's lawn mowing earnings problem -/
theorem adams_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) 
  (h1 : rate = 9)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 8) :
  (total_lawns - forgotten_lawns) * rate = 36 := by
  sorry

#check adams_earnings

end NUMINAMATH_CALUDE_adams_earnings_l22_2288


namespace NUMINAMATH_CALUDE_sum_of_common_elements_l22_2248

def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

def common_elements (m : ℕ) : ℕ := 10 * 4^m

theorem sum_of_common_elements : 
  (Finset.range 10).sum (λ i => common_elements i) = 3495250 := by sorry

end NUMINAMATH_CALUDE_sum_of_common_elements_l22_2248


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l22_2264

/-- Two lines are parallel if their slopes are equal and they are not the same line -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 * c2 ≠ a2 * c1

/-- The theorem stating that if two lines (3+m)x+4y=5-3m and 2x+(5+m)y=8 are parallel, then m = -7 -/
theorem parallel_lines_m_value (m : ℝ) :
  are_parallel (3 + m) 4 (3*m - 5) 2 (5 + m) (-8) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l22_2264


namespace NUMINAMATH_CALUDE_gcd_2197_2209_l22_2278

theorem gcd_2197_2209 : Nat.gcd 2197 2209 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2197_2209_l22_2278


namespace NUMINAMATH_CALUDE_ab_value_l22_2200

theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 + b^2 = 9) (h2 : a^4 + b^4 = 65) : a * b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l22_2200


namespace NUMINAMATH_CALUDE_opposite_points_theorem_l22_2265

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Given two points on a number line, proves that if they represent opposite numbers, 
    their distance is 8, and the first point is to the left of the second, 
    then they represent -4 and 4 respectively -/
theorem opposite_points_theorem (A B : Point) : 
  A.value + B.value = 0 →  -- A and B represent opposite numbers
  |A.value - B.value| = 8 →  -- Distance between A and B is 8
  A.value < B.value →  -- A is to the left of B
  A.value = -4 ∧ B.value = 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_points_theorem_l22_2265


namespace NUMINAMATH_CALUDE_sachins_age_l22_2228

theorem sachins_age (sachin rahul : ℕ) 
  (age_difference : rahul = sachin + 4)
  (age_ratio : sachin * 9 = rahul * 7) : 
  sachin = 14 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l22_2228


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l22_2291

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ = 2 → 
  x₂^2 + x₂ = 2 → 
  x₁ ≠ x₂ → 
  1/x₁ + 1/x₂ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l22_2291


namespace NUMINAMATH_CALUDE_expression_value_l22_2247

theorem expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l22_2247


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l22_2209

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > b,
    if the angle between its asymptotes is 45°, then a/b = √2 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ t : ℝ, ∃ x y : ℝ, y = (b / a) * x ∨ y = -(b / a) * x) →
  (Real.pi / 4 : ℝ) = Real.arctan ((b / a - (-b / a)) / (1 + (b / a) * (-b / a))) →
  a / b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l22_2209


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l22_2239

theorem circle_area_from_circumference (c : ℝ) (h : c = 24) :
  let r := c / (2 * Real.pi)
  (Real.pi * r ^ 2) = 144 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l22_2239


namespace NUMINAMATH_CALUDE_cone_volume_l22_2280

/-- The volume of a cone with given slant height and lateral surface angle -/
theorem cone_volume (s : ℝ) (θ : ℝ) (h : s = 6) (h' : θ = 2 * π / 3) :
  ∃ (v : ℝ), v = (16 * Real.sqrt 2 / 3) * π ∧ v = (1/3) * π * (s * θ / (2 * π))^2 * Real.sqrt (s^2 - (s * θ / (2 * π))^2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l22_2280


namespace NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l22_2256

def base_expansion (n : ℕ) (b : ℕ) : List ℕ :=
  sorry

def sum_digits (digits : List ℕ) : ℕ :=
  sorry

def is_largest_base (b : ℕ) : Prop :=
  (∀ k > b, sum_digits (base_expansion ((k + 2)^4) k) = 32) ∧
  sum_digits (base_expansion ((b + 2)^4) b) ≠ 32

theorem largest_base_for_12_4th_power : is_largest_base 7 :=
  sorry

end NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l22_2256


namespace NUMINAMATH_CALUDE_average_weight_calculation_l22_2289

/-- Given the average weights of pairs of individuals and the weight of one individual,
    calculate the average weight of all three individuals. -/
theorem average_weight_calculation
  (avg_ab avg_bc b_weight : ℝ)
  (h_avg_ab : (a + b_weight) / 2 = avg_ab)
  (h_avg_bc : (b_weight + c) / 2 = avg_bc)
  (h_b : b_weight = 37)
  (a c : ℝ) :
  (a + b_weight + c) / 3 = 45 :=
by sorry


end NUMINAMATH_CALUDE_average_weight_calculation_l22_2289


namespace NUMINAMATH_CALUDE_fifteen_shaded_cubes_l22_2249

/-- Represents a 3x3x3 cube constructed from smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  shaded_per_face : Nat

/-- Calculates the number of uniquely shaded cubes in the large cube -/
def count_shaded_cubes (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the number of uniquely shaded cubes is 15 -/
theorem fifteen_shaded_cubes (cube : LargeCube) 
  (h1 : cube.size = 3) 
  (h2 : cube.total_cubes = 27) 
  (h3 : cube.shaded_per_face = 3) : 
  count_shaded_cubes cube = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_shaded_cubes_l22_2249


namespace NUMINAMATH_CALUDE_sin_2010_degrees_l22_2292

theorem sin_2010_degrees : Real.sin (2010 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2010_degrees_l22_2292


namespace NUMINAMATH_CALUDE_lunch_cost_calculation_l22_2234

/-- Calculates the total cost of lunch for all students in an elementary school --/
theorem lunch_cost_calculation (third_grade_classes : ℕ) (third_grade_students_per_class : ℕ)
  (fourth_grade_classes : ℕ) (fourth_grade_students_per_class : ℕ)
  (fifth_grade_classes : ℕ) (fifth_grade_students_per_class : ℕ)
  (hamburger_cost : ℚ) (carrots_cost : ℚ) (cookie_cost : ℚ) :
  third_grade_classes = 5 →
  third_grade_students_per_class = 30 →
  fourth_grade_classes = 4 →
  fourth_grade_students_per_class = 28 →
  fifth_grade_classes = 4 →
  fifth_grade_students_per_class = 27 →
  hamburger_cost = 2.1 →
  carrots_cost = 0.5 →
  cookie_cost = 0.2 →
  (third_grade_classes * third_grade_students_per_class +
   fourth_grade_classes * fourth_grade_students_per_class +
   fifth_grade_classes * fifth_grade_students_per_class) *
  (hamburger_cost + carrots_cost + cookie_cost) = 1036 :=
by sorry

end NUMINAMATH_CALUDE_lunch_cost_calculation_l22_2234


namespace NUMINAMATH_CALUDE_min_phase_shift_l22_2275

/-- Given a sinusoidal function with a phase shift, prove that under certain symmetry conditions, 
    the smallest possible absolute value of the phase shift is π/4. -/
theorem min_phase_shift (φ : ℝ) : 
  (∀ x, 3 * Real.sin (3 * (x - π/4) + φ) = 3 * Real.sin (3 * (2*π/3 - x) + φ)) →
  (∃ k : ℤ, φ = k * π - π/4) →
  ∃ ψ : ℝ, abs ψ = π/4 ∧ (∀ θ : ℝ, (∃ k : ℤ, θ = k * π - π/4) → abs θ ≥ abs ψ) :=
by sorry

end NUMINAMATH_CALUDE_min_phase_shift_l22_2275


namespace NUMINAMATH_CALUDE_triangle_sine_ratio_l22_2293

/-- Given a triangle ABC where the ratio of sines of angles is 5:7:8, 
    prove the ratio of sides and the measure of angle B -/
theorem triangle_sine_ratio (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sine_ratio : ∃ k : ℝ, k > 0 ∧ Real.sin A = 5*k ∧ Real.sin B = 7*k ∧ Real.sin C = 8*k) :
  (∃ m : ℝ, m > 0 ∧ a = 5*m ∧ b = 7*m ∧ c = 8*m) ∧ B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_ratio_l22_2293


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l22_2229

def U : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_equals_set : A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l22_2229


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l22_2221

theorem original_price_after_discounts (price : ℝ) : 
  price * (1 - 0.2) * (1 - 0.1) * (1 - 0.05) = 6840 → price = 10000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discounts_l22_2221


namespace NUMINAMATH_CALUDE_function_equation_solution_l22_2266

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l22_2266


namespace NUMINAMATH_CALUDE_probability_two_rainy_days_l22_2213

/-- Represents the weather condition for a day -/
inductive Weather
| Rainy
| NotRainy

/-- Represents the weather for three consecutive days -/
def ThreeDayWeather := (Weather × Weather × Weather)

/-- Checks if a ThreeDayWeather has exactly two rainy days -/
def hasTwoRainyDays (w : ThreeDayWeather) : Bool :=
  match w with
  | (Weather.Rainy, Weather.Rainy, Weather.NotRainy) => true
  | (Weather.Rainy, Weather.NotRainy, Weather.Rainy) => true
  | (Weather.NotRainy, Weather.Rainy, Weather.Rainy) => true
  | _ => false

/-- The total number of weather groups in the sample -/
def totalGroups : Nat := 20

/-- The number of groups with exactly two rainy days -/
def groupsWithTwoRainyDays : Nat := 5

/-- Theorem: The probability of exactly two rainy days out of three is 0.25 -/
theorem probability_two_rainy_days :
  (groupsWithTwoRainyDays : ℚ) / totalGroups = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_probability_two_rainy_days_l22_2213


namespace NUMINAMATH_CALUDE_tanner_has_16_berries_l22_2260

/-- The number of berries each person has -/
structure Berries where
  skylar : ℕ
  steve : ℕ
  stacy : ℕ
  tanner : ℕ

/-- Calculate the number of berries each person has based on the given conditions -/
def calculate_berries : Berries :=
  let skylar := 20
  let steve := 4 * (skylar / 3)^2
  let stacy := 2 * steve + 50
  let tanner := (8 * stacy) / (skylar + steve)
  { skylar := skylar, steve := steve, stacy := stacy, tanner := tanner }

/-- Theorem stating that Tanner has 16 berries -/
theorem tanner_has_16_berries : (calculate_berries.tanner) = 16 := by
  sorry

#eval calculate_berries.tanner

end NUMINAMATH_CALUDE_tanner_has_16_berries_l22_2260


namespace NUMINAMATH_CALUDE_reptiles_in_swamps_l22_2273

theorem reptiles_in_swamps (num_swamps : ℕ) (reptiles_per_swamp : ℕ) :
  num_swamps = 4 →
  reptiles_per_swamp = 356 →
  num_swamps * reptiles_per_swamp = 1424 := by
  sorry

end NUMINAMATH_CALUDE_reptiles_in_swamps_l22_2273


namespace NUMINAMATH_CALUDE_green_balls_count_l22_2271

def bag_problem (blue_balls : ℕ) (prob_blue : ℚ) (red_balls : ℕ) (green_balls : ℕ) : Prop :=
  blue_balls = 10 ∧ 
  prob_blue = 2/7 ∧ 
  red_balls = 2 * blue_balls ∧
  prob_blue = blue_balls / (blue_balls + red_balls + green_balls)

theorem green_balls_count : 
  ∃ (green_balls : ℕ), bag_problem 10 (2/7) 20 green_balls → green_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l22_2271


namespace NUMINAMATH_CALUDE_greatest_possible_median_l22_2233

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  t = 20 →
  r ≤ 13 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 20) / 5 = 10 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 20 ∧
    r' = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l22_2233


namespace NUMINAMATH_CALUDE_total_toes_on_bus_l22_2285

/-- Represents a race of beings on Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toes_per_hand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of students of each race on the bus -/
def students (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes for a single being of a given race -/
def toes_per_being (r : Race) : ℕ :=
  hands r * toes_per_hand r

/-- Total number of toes for all students of a given race on the bus -/
def total_toes_per_race (r : Race) : ℕ :=
  students r * toes_per_being r

/-- Theorem: The total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus :
  total_toes_per_race Race.Hoopit + total_toes_per_race Race.Neglart = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_toes_on_bus_l22_2285


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l22_2244

/-- An arithmetic sequence with the given property has a common difference of 3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h : a 2015 = a 2013 + 6)  -- The given condition
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l22_2244


namespace NUMINAMATH_CALUDE_large_coin_equivalent_mass_l22_2276

theorem large_coin_equivalent_mass (large_coin_mass : ℝ) (pound_coin_mass : ℝ) :
  large_coin_mass = 100000 →
  pound_coin_mass = 10 →
  (large_coin_mass / pound_coin_mass : ℝ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_large_coin_equivalent_mass_l22_2276


namespace NUMINAMATH_CALUDE_range_of_k_for_decreasing_proportional_function_l22_2201

/-- A proportional function y = (k+4)x where y decreases as x increases -/
def decreasing_proportional_function (k : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k + 4) * x₁ > (k + 4) * x₂

/-- The range of k for a decreasing proportional function y = (k+4)x -/
theorem range_of_k_for_decreasing_proportional_function :
  ∀ k : ℝ, decreasing_proportional_function k → k < -4 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_decreasing_proportional_function_l22_2201


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l22_2216

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = -2 * Real.sqrt 2

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define a point on line l
def point_on_line_l (x y : ℝ) : Prop := line_l x

-- Define the perpendicular line l' through P
def line_l_prime (x y x_p y_p : ℝ) : Prop :=
  y - y_p = -(3 * y_p) / (2 * Real.sqrt 2) * (x - x_p)

theorem ellipse_intersection_theorem :
  ∀ (x_p y_p : ℝ),
    point_on_line_l x_p y_p →
    ∃ (x_m y_m x_n y_n : ℝ),
      point_on_ellipse x_m y_m ∧
      point_on_ellipse x_n y_n ∧
      (x_p - x_m)^2 + (y_p - y_m)^2 = (x_p - x_n)^2 + (y_p - y_n)^2 →
      line_l_prime (-4 * Real.sqrt 2 / 3) 0 x_p y_p :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l22_2216


namespace NUMINAMATH_CALUDE_fraction_irreducible_l22_2236

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l22_2236


namespace NUMINAMATH_CALUDE_robin_cupcakes_proof_l22_2287

/-- Represents the number of cupcakes Robin initially made -/
def initial_cupcakes : ℕ := 42

/-- Represents the number of cupcakes Robin sold -/
def sold_cupcakes : ℕ := 22

/-- Represents the number of cupcakes Robin made later -/
def new_cupcakes : ℕ := 39

/-- Represents the final number of cupcakes Robin had -/
def final_cupcakes : ℕ := 59

/-- Proves that the initial number of cupcakes is correct given the conditions -/
theorem robin_cupcakes_proof : 
  initial_cupcakes - sold_cupcakes + new_cupcakes = final_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcakes_proof_l22_2287


namespace NUMINAMATH_CALUDE_alani_earnings_l22_2296

/-- Calculates the earnings for baby-sitting given the base earnings, base hours, and actual hours worked. -/
def calculate_earnings (base_earnings : ℚ) (base_hours : ℚ) (actual_hours : ℚ) : ℚ :=
  (base_earnings / base_hours) * actual_hours

/-- Proves that Alani will earn $75 for 5 hours of baby-sitting given her rate of $45 for 3 hours. -/
theorem alani_earnings : calculate_earnings 45 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_alani_earnings_l22_2296


namespace NUMINAMATH_CALUDE_boxes_with_neither_l22_2298

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ)
  (h1 : total = 12)
  (h2 : markers = 8)
  (h3 : erasers = 5)
  (h4 : both = 4) :
  total - (markers + erasers - both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l22_2298


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_33_l22_2224

theorem smallest_four_digit_divisible_by_33 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 → n ≥ 1023 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_33_l22_2224


namespace NUMINAMATH_CALUDE_trig_identity_proof_l22_2215

theorem trig_identity_proof : 
  Real.cos (28 * π / 180) * Real.cos (17 * π / 180) - 
  Real.sin (28 * π / 180) * Real.cos (73 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l22_2215


namespace NUMINAMATH_CALUDE_lamp_lit_area_l22_2225

/-- Given a square plot with a lamp on one corner, if the light reaches 21 m
    and the lit area is 346.36 m², then the side length of the square plot is 21 m. -/
theorem lamp_lit_area (light_reach : ℝ) (lit_area : ℝ) (side_length : ℝ) : 
  light_reach = 21 →
  lit_area = 346.36 →
  lit_area = (1/4) * Real.pi * light_reach^2 →
  side_length = light_reach →
  side_length = 21 := by sorry

end NUMINAMATH_CALUDE_lamp_lit_area_l22_2225


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_28_l22_2219

theorem consecutive_integers_around_sqrt_28 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 28 ∧ Real.sqrt 28 < ↑b) → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_28_l22_2219


namespace NUMINAMATH_CALUDE_gcd_154_90_l22_2251

theorem gcd_154_90 : Nat.gcd 154 90 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_154_90_l22_2251


namespace NUMINAMATH_CALUDE_factor_x4_minus_64_l22_2284

theorem factor_x4_minus_64 (x : ℝ) : x^4 - 64 = (x^2 + 8) * (x^2 - 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_64_l22_2284


namespace NUMINAMATH_CALUDE_original_group_size_l22_2282

theorem original_group_size (original_days : ℕ) (absent_men : ℕ) (new_days : ℕ) :
  original_days = 6 →
  absent_men = 4 →
  new_days = 12 →
  ∃ (total_men : ℕ), 
    total_men > absent_men ∧
    (1 : ℚ) / (original_days * total_men) = (1 : ℚ) / (new_days * (total_men - absent_men)) ∧
    total_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l22_2282


namespace NUMINAMATH_CALUDE_function_range_l22_2227

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.exp x + x - a)

theorem function_range (a : ℝ) :
  (∃ y₀ : ℝ, y₀ ∈ Set.Icc (-1) 1 ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Icc 1 (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l22_2227


namespace NUMINAMATH_CALUDE_unique_quadratic_family_l22_2202

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The property that the product of roots equals the sum of coefficients -/
def roots_product_equals_coeff_sum (p : QuadraticPolynomial) : Prop :=
  ∃ r s : ℝ, r * s = p.a + p.b + p.c ∧ p.a * r^2 + p.b * r + p.c = 0 ∧ p.a * s^2 + p.b * s + p.c = 0

/-- The theorem stating that there's exactly one family of quadratic polynomials satisfying the condition -/
theorem unique_quadratic_family :
  ∃! f : ℝ → QuadraticPolynomial,
    (∀ c : ℝ, (f c).a = 1 ∧ (f c).b = -1 ∧ (f c).c = c) ∧
    (∀ p : QuadraticPolynomial, roots_product_equals_coeff_sum p ↔ ∃ c : ℝ, p = f c) :=
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_family_l22_2202


namespace NUMINAMATH_CALUDE_complexity_theorem_l22_2245

/-- Complexity of an integer is the number of prime factors in its prime decomposition -/
def complexity (n : ℕ) : ℕ := sorry

/-- n is a power of two -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem complexity_theorem (n : ℕ) (h : n > 1) :
  (∀ m : ℕ, n < m ∧ m ≤ 2*n → complexity m ≤ complexity n) ↔ is_power_of_two n ∧
  ¬∃ n : ℕ, ∀ m : ℕ, n < m ∧ m ≤ 2*n → complexity m < complexity n :=
sorry

end NUMINAMATH_CALUDE_complexity_theorem_l22_2245


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l22_2240

theorem product_of_two_numbers (x y : ℝ) 
  (sum_condition : x + y = 24) 
  (sum_squares_condition : x^2 + y^2 = 400) : 
  x * y = 88 := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l22_2240


namespace NUMINAMATH_CALUDE_fourth_root_of_256_l22_2299

theorem fourth_root_of_256 (m : ℝ) : (256 : ℝ) ^ (1/4) = 4^m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_256_l22_2299


namespace NUMINAMATH_CALUDE_student_event_arrangements_l22_2235

theorem student_event_arrangements (n m : ℕ) (h1 : n = 7) (h2 : m = 5) : 
  (n.choose m * m.factorial) - ((n - 1).choose m * m.factorial) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_student_event_arrangements_l22_2235


namespace NUMINAMATH_CALUDE_cube_side_length_ratio_l22_2222

/-- Given two cubes of the same material, if the weight of the second cube
    is 8 times the weight of the first cube, then the ratio of the side length
    of the second cube to the side length of the first cube is 2:1. -/
theorem cube_side_length_ratio (s1 s2 : ℝ) (w1 w2 : ℝ) : 
  s1 > 0 → s2 > 0 → w1 > 0 → w2 > 0 →
  (w2 = 8 * w1) →
  (w1 = s1^3) →
  (w2 = s2^3) →
  s2 / s1 = 2 := by sorry

end NUMINAMATH_CALUDE_cube_side_length_ratio_l22_2222


namespace NUMINAMATH_CALUDE_min_value_theorem_l22_2220

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 10*x + 100/x^2 ≥ 79 ∧ ∃ y > 0, y^2 + 10*y + 100/y^2 = 79 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l22_2220


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l22_2253

-- Define the sets N and M
def N (q : ℝ) : Set ℝ := {x | x^2 + 6*x - q = 0}
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 6 = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N q = {2} → p + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l22_2253


namespace NUMINAMATH_CALUDE_certain_event_good_product_l22_2241

theorem certain_event_good_product (total : Nat) (good : Nat) (defective : Nat) (draw : Nat) :
  total = good + defective →
  good = 10 →
  defective = 2 →
  draw = 3 →
  Fintype.card {s : Finset (Fin total) // s.card = draw ∧ (∃ i ∈ s, i.val < good)} / Fintype.card {s : Finset (Fin total) // s.card = draw} = 1 :=
sorry

end NUMINAMATH_CALUDE_certain_event_good_product_l22_2241


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l22_2206

/-- The area of the shaded region formed by two intersecting rectangles minus a circular cut-out -/
theorem shaded_area_calculation (rect1_width rect1_length rect2_width rect2_length : ℝ)
  (circle_radius : ℝ) (h1 : rect1_width = 3) (h2 : rect1_length = 12)
  (h3 : rect2_width = 4) (h4 : rect2_length = 7) (h5 : circle_radius = 1) :
  let rect1_area := rect1_width * rect1_length
  let rect2_area := rect2_width * rect2_length
  let overlap_area := min rect1_width rect2_width * min rect1_length rect2_length
  let circle_area := Real.pi * circle_radius^2
  rect1_area + rect2_area - overlap_area - circle_area = 64 - Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l22_2206


namespace NUMINAMATH_CALUDE_first_train_speed_calculation_l22_2243

/-- The speed of the first train in kmph -/
def first_train_speed : ℝ := 72

/-- The speed of the second train in kmph -/
def second_train_speed : ℝ := 36

/-- The length of the first train in meters -/
def first_train_length : ℝ := 200

/-- The length of the second train in meters -/
def second_train_length : ℝ := 300

/-- The time taken for the first train to cross the second train in seconds -/
def crossing_time : ℝ := 49.9960003199744

theorem first_train_speed_calculation :
  first_train_speed = 
    (first_train_length + second_train_length) / crossing_time * 3600 / 1000 + second_train_speed :=
by sorry

end NUMINAMATH_CALUDE_first_train_speed_calculation_l22_2243


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l22_2259

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2_eq_18 (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → f a b 2 = 18 := by
  sorry

#check extremum_implies_f_2_eq_18

end NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l22_2259


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l22_2210

theorem negative_fractions_comparison : (-1/2 : ℚ) < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l22_2210


namespace NUMINAMATH_CALUDE_corporation_total_employees_l22_2263

/-- The total number of employees in a corporation -/
def total_employees (part_time full_time contractors interns consultants : ℕ) : ℕ :=
  part_time + full_time + contractors + interns + consultants

/-- Theorem: The corporation employs 66907 workers in total -/
theorem corporation_total_employees :
  total_employees 2047 63109 1500 333 918 = 66907 := by
  sorry

end NUMINAMATH_CALUDE_corporation_total_employees_l22_2263


namespace NUMINAMATH_CALUDE_unique_complex_solution_l22_2204

theorem unique_complex_solution :
  ∃! (z : ℂ), Complex.abs z < 20 ∧ Complex.cos z = (z - 2) / (z + 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_complex_solution_l22_2204


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l22_2279

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l22_2279


namespace NUMINAMATH_CALUDE_college_student_count_l22_2283

theorem college_student_count (boys girls : ℕ) (h1 : boys = 2 * girls) (h2 : girls = 200) :
  boys + girls = 600 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l22_2283


namespace NUMINAMATH_CALUDE_min_composite_with_small_factors_l22_2246

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def has_prime_factorization (n : ℕ) (max_factor : ℕ) : Prop :=
  ∃ (factors : List ℕ), 
    factors.length ≥ 2 ∧
    (∀ p ∈ factors, is_prime p ∧ p ≤ max_factor) ∧
    factors.prod = n

theorem min_composite_with_small_factors :
  ∀ n : ℕ, 
    ¬ is_prime n →
    has_prime_factorization n 10 →
    n ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_composite_with_small_factors_l22_2246


namespace NUMINAMATH_CALUDE_divisibility_by_two_l22_2272

theorem divisibility_by_two (a b : ℕ) (h : 2 ∣ (a * b)) : ¬(¬(2 ∣ a) ∧ ¬(2 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_two_l22_2272


namespace NUMINAMATH_CALUDE_unique_integer_value_of_expression_l22_2205

theorem unique_integer_value_of_expression 
  (m n p : ℕ) 
  (hm : 2 ≤ m ∧ m ≤ 9) 
  (hn : 2 ≤ n ∧ n ≤ 9) 
  (hp : 2 ≤ p ∧ p ≤ 9) 
  (hdiff : m ≠ n ∧ m ≠ p ∧ n ≠ p) : 
  (∃ k : ℤ, (m + n + p : ℚ) / (m + n) = k) → (m + n + p : ℚ) / (m + n) = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_integer_value_of_expression_l22_2205


namespace NUMINAMATH_CALUDE_even_sine_function_phi_l22_2255

/-- Given a function f(x) = sin((x + φ) / 3) where φ ∈ [0, 2π],
    prove that if f is even, then φ = 3π/2 -/
theorem even_sine_function_phi (φ : ℝ) (h1 : φ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∀ x, Real.sin ((x + φ) / 3) = Real.sin ((-x + φ) / 3)) →
  φ = 3 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_even_sine_function_phi_l22_2255


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_two_l22_2226

def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem subset_implies_a_leq_two (a : ℝ) (h : A ⊆ B a) : a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_two_l22_2226


namespace NUMINAMATH_CALUDE_impossible_transformation_l22_2297

/-- Represents the operation of replacing two numbers with their updated values -/
def replace_numbers (numbers : List ℕ) (x y : ℕ) : List ℕ :=
  (x - 1) :: (y + 3) :: (numbers.filter (λ n => n ≠ x ∧ n ≠ y))

/-- Checks if a list of numbers is valid according to the problem rules -/
def is_valid_list (numbers : List ℕ) : Prop :=
  numbers.length = 10 ∧ numbers.sum % 2 = 1

/-- The initial list of numbers on the board -/
def initial_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- The target list of numbers we want to achieve -/
def target_numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 2012]

/-- Theorem stating that it's impossible to transform the initial numbers into the target numbers -/
theorem impossible_transformation :
  ¬ ∃ (n : ℕ) (operations : List (ℕ × ℕ)),
    operations.length = n ∧
    (operations.foldl (λ acc (x, y) => replace_numbers acc x y) initial_numbers) = target_numbers :=
sorry

end NUMINAMATH_CALUDE_impossible_transformation_l22_2297


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l22_2218

/-- The value of m for which the circle x^2 + y^2 = 4m is tangent to the line x + y = 2√m -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4*m ∧ x + y = 2*Real.sqrt m → 
    (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 4*m ∧ p.1 + p.2 = 2*Real.sqrt m)) → 
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l22_2218


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l22_2281

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l22_2281


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l22_2268

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 16) (h2 : x - y = 2) : x^2 - y^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l22_2268


namespace NUMINAMATH_CALUDE_chips_price_calculation_l22_2230

/-- Given a discount and a final price, calculate the original price --/
def original_price (discount : ℝ) (final_price : ℝ) : ℝ :=
  discount + final_price

theorem chips_price_calculation :
  let discount : ℝ := 17
  let final_price : ℝ := 18
  original_price discount final_price = 35 := by
sorry

end NUMINAMATH_CALUDE_chips_price_calculation_l22_2230


namespace NUMINAMATH_CALUDE_parallelogram_area_l22_2258

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 20 is equal to 100√3. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l22_2258


namespace NUMINAMATH_CALUDE_percentage_without_fulltime_jobs_is_19_l22_2267

/-- The percentage of parents who do not hold full-time jobs -/
def percentage_without_fulltime_jobs (total_parents : ℕ) 
  (mother_ratio : ℚ) (father_ratio : ℚ) (women_ratio : ℚ) : ℚ :=
  let mothers := (women_ratio * total_parents).floor
  let fathers := total_parents - mothers
  let mothers_with_jobs := (mother_ratio * mothers).floor
  let fathers_with_jobs := (father_ratio * fathers).floor
  let parents_without_jobs := total_parents - mothers_with_jobs - fathers_with_jobs
  (parents_without_jobs : ℚ) / total_parents * 100

/-- Theorem stating that given the conditions in the problem, 
    the percentage of parents without full-time jobs is 19% -/
theorem percentage_without_fulltime_jobs_is_19 :
  ∀ n : ℕ, n > 0 → 
  percentage_without_fulltime_jobs n (9/10) (3/4) (2/5) = 19 := by
  sorry

end NUMINAMATH_CALUDE_percentage_without_fulltime_jobs_is_19_l22_2267


namespace NUMINAMATH_CALUDE_browns_utility_bill_l22_2270

/-- The total amount of Mrs. Brown's utility bills -/
def utility_bill_total (fifty_count : ℕ) (ten_count : ℕ) : ℕ :=
  50 * fifty_count + 10 * ten_count

/-- Theorem stating that Mrs. Brown's utility bills total $170 -/
theorem browns_utility_bill : utility_bill_total 3 2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_browns_utility_bill_l22_2270


namespace NUMINAMATH_CALUDE_max_profit_at_grade_5_l22_2290

def profit (x : ℕ) : ℝ :=
  (4 * (x - 1) + 8) * (60 - 6 * (x - 1))

theorem max_profit_at_grade_5 :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → profit x ≤ profit 5 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_grade_5_l22_2290


namespace NUMINAMATH_CALUDE_xy_inequality_and_equality_l22_2274

theorem xy_inequality_and_equality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  ((x * y - 10)^2 ≥ 64) ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_and_equality_l22_2274


namespace NUMINAMATH_CALUDE_vector_problem_proof_l22_2223

def vector_problem (a b : ℝ × ℝ) : Prop :=
  a = (1, 1) ∧
  (b.1^2 + b.2^2) = 16 ∧
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2)) = -2 →
  ((3*a.1 - b.1)^2 + (3*a.2 - b.2)^2) = 10

theorem vector_problem_proof : ∀ a b : ℝ × ℝ, vector_problem a b :=
by
  sorry

end NUMINAMATH_CALUDE_vector_problem_proof_l22_2223


namespace NUMINAMATH_CALUDE_three_Z_five_equals_eight_l22_2232

-- Define the operation Z
def Z (a b : ℝ) : ℝ := b + 10 * a - 3 * a^2

-- Theorem to prove
theorem three_Z_five_equals_eight : Z 3 5 = 8 := by sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_eight_l22_2232


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l22_2237

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l22_2237


namespace NUMINAMATH_CALUDE_equation_solutions_count_l22_2203

theorem equation_solutions_count :
  let f : ℝ → ℝ := fun x => (x^2 - 7)^2 + 2*x^2 - 33
  ∃! (s : Finset ℝ), (∀ x ∈ s, f x = 0) ∧ Finset.card s = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l22_2203


namespace NUMINAMATH_CALUDE_replaced_man_age_l22_2207

theorem replaced_man_age (n : ℕ) (avg_increase : ℝ) (man1_age : ℕ) (women_avg_age : ℝ) :
  n = 10 ∧ 
  avg_increase = 6 ∧ 
  man1_age = 18 ∧ 
  women_avg_age = 50 → 
  ∃ (original_avg : ℝ) (man2_age : ℕ),
    n * (original_avg + avg_increase) = n * original_avg + 2 * women_avg_age - (man1_age + man2_age) ∧
    man2_age = 22 := by
  sorry

#check replaced_man_age

end NUMINAMATH_CALUDE_replaced_man_age_l22_2207


namespace NUMINAMATH_CALUDE_sum_calculation_l22_2252

def sequence_S : ℕ → ℕ
  | 0 => 0
  | (n + 1) => sequence_S n + (2 * n + 1)

def sequence_n : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_n n + 2

theorem sum_calculation :
  ∃ k : ℕ, sequence_n k > 50 ∧ sequence_n (k - 1) ≤ 50 ∧ sequence_S (k - 1) = 625 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l22_2252


namespace NUMINAMATH_CALUDE_theta_range_l22_2212

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2*k*Real.pi + Real.pi/12 < θ ∧ θ < 2*k*Real.pi + 5*Real.pi/12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l22_2212


namespace NUMINAMATH_CALUDE_solution_set_equality_l22_2242

theorem solution_set_equality (x : ℝ) : 
  (1 / ((x + 1) * (x - 1)) ≤ 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l22_2242


namespace NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l22_2211

/-- A regular (4k+2)-gon inscribed in a circle -/
structure RegularPolygon (k : ℕ) where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- The vertices of the polygon -/
  vertices : Fin (4*k+2) → ℝ × ℝ
  /-- Condition that the polygon is regular and inscribed -/
  regular_inscribed : ∀ i : Fin (4*k+2), dist O (vertices i) = R

/-- The sum of segments cut by a central angle on diagonals -/
def sum_of_segments (p : RegularPolygon k) : ℝ := sorry

/-- Theorem: The sum of segments equals the radius -/
theorem sum_of_segments_equals_radius (k : ℕ) (p : RegularPolygon k) :
  sum_of_segments p = p.R := by sorry

end NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l22_2211


namespace NUMINAMATH_CALUDE_rabbit_turning_point_theorem_l22_2269

/-- The point where the rabbit starts moving away from the fox -/
def rabbit_turning_point : ℝ × ℝ := (2.8, 5.6)

/-- The location of the fox -/
def fox_location : ℝ × ℝ := (10, 8)

/-- The slope of the rabbit's path -/
def rabbit_path_slope : ℝ := -3

/-- The y-intercept of the rabbit's path -/
def rabbit_path_intercept : ℝ := 14

/-- The equation of the rabbit's path: y = mx + b -/
def rabbit_path (x : ℝ) : ℝ := rabbit_path_slope * x + rabbit_path_intercept

theorem rabbit_turning_point_theorem :
  let (c, d) := rabbit_turning_point
  let (fx, fy) := fox_location
  -- The turning point lies on the rabbit's path
  d = rabbit_path c ∧
  -- The line from the fox to the turning point is perpendicular to the rabbit's path
  (d - fy) / (c - fx) = -1 / rabbit_path_slope := by
  sorry

end NUMINAMATH_CALUDE_rabbit_turning_point_theorem_l22_2269


namespace NUMINAMATH_CALUDE_odd_terms_sum_l22_2238

def sequence_sum (n : ℕ) : ℕ := n^2 + 2*n - 1

def arithmetic_sum (first last : ℕ) (step : ℕ) : ℕ :=
  ((last - first) / step + 1) * (first + last) / 2

theorem odd_terms_sum :
  (arithmetic_sum 1 25 2) = 350 :=
by sorry

end NUMINAMATH_CALUDE_odd_terms_sum_l22_2238


namespace NUMINAMATH_CALUDE_xiaojie_purchase_solution_l22_2257

/-- Represents the stationery purchase problem --/
structure StationeryPurchase where
  red_black_pen_price : ℕ
  black_refill_price : ℕ
  red_refill_price : ℕ
  black_discount : ℚ
  red_discount : ℚ
  red_black_pens_bought : ℕ
  total_refills_bought : ℕ
  total_spent : ℕ

/-- The specific purchase made by Xiaojie --/
def xiaojie_purchase : StationeryPurchase :=
  { red_black_pen_price := 10
  , black_refill_price := 6
  , red_refill_price := 8
  , black_discount := 1/2
  , red_discount := 3/4
  , red_black_pens_bought := 2
  , total_refills_bought := 10
  , total_spent := 74
  }

/-- Theorem stating the correct number of refills bought and amount saved --/
theorem xiaojie_purchase_solution (p : StationeryPurchase) (h : p = xiaojie_purchase) :
  ∃ (black_refills red_refills : ℕ) (savings : ℕ),
    black_refills + red_refills = p.total_refills_bought ∧
    black_refills = 2 ∧
    red_refills = 8 ∧
    savings = 22 ∧
    p.red_black_pen_price * p.red_black_pens_bought +
    (p.black_refill_price * black_refills + p.red_refill_price * red_refills) -
    p.total_spent = savings :=
  sorry

end NUMINAMATH_CALUDE_xiaojie_purchase_solution_l22_2257


namespace NUMINAMATH_CALUDE_sin_double_angle_on_unit_circle_l22_2231

/-- Given a point B on the unit circle with coordinates (-3/5, 4/5), 
    prove that sin(2α) = -24/25, where α is the angle formed by OA and OB, 
    and O is the origin and A is the point (1,0) on the unit circle. -/
theorem sin_double_angle_on_unit_circle 
  (B : ℝ × ℝ) 
  (h_B_on_circle : B.1^2 + B.2^2 = 1) 
  (h_B_coords : B = (-3/5, 4/5)) 
  (α : ℝ) 
  (h_α_def : α = Real.arccos B.1) : 
  Real.sin (2 * α) = -24/25 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_on_unit_circle_l22_2231


namespace NUMINAMATH_CALUDE_painting_gift_options_l22_2214

theorem painting_gift_options (n : ℕ) (h : n = 10) : 
  (Finset.powerset (Finset.range n)).card - 1 = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_painting_gift_options_l22_2214


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l22_2262

/-- The number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches -/
theorem ten_player_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l22_2262
