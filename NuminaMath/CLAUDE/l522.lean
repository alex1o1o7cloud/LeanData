import Mathlib

namespace NUMINAMATH_CALUDE_unique_p_for_three_positive_integer_roots_l522_52290

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

/-- Predicate to check if a number is a positive integer -/
def is_positive_integer (x : ℝ) : Prop :=
  x > 0 ∧ ∃ n : ℕ, x = n

/-- The main theorem -/
theorem unique_p_for_three_positive_integer_roots :
  ∃! p : ℝ, ∃ x y z : ℝ,
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    is_positive_integer x ∧ is_positive_integer y ∧ is_positive_integer z ∧
    cubic_equation p x = 0 ∧ cubic_equation p y = 0 ∧ cubic_equation p z = 0 ∧
    p = 76 :=
sorry

end NUMINAMATH_CALUDE_unique_p_for_three_positive_integer_roots_l522_52290


namespace NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_achievable_l522_52253

theorem min_sum_of_squares (a b : ℝ) (h : a * b = -6) : a^2 + b^2 ≥ 12 := by
  sorry

theorem min_sum_of_squares_achievable : ∃ (a b : ℝ), a * b = -6 ∧ a^2 + b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_achievable_l522_52253


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l522_52216

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem statement
theorem f_strictly_increasing :
  StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l522_52216


namespace NUMINAMATH_CALUDE_jack_second_half_time_is_six_l522_52254

/-- The time Jack took to run up the hill -/
def jack_total_time (jill_time first_half_time time_diff : ℕ) : ℕ :=
  jill_time - time_diff

/-- The time Jack took to run up the second half of the hill -/
def jack_second_half_time (total_time first_half_time : ℕ) : ℕ :=
  total_time - first_half_time

/-- Proof that Jack took 6 seconds to run up the second half of the hill -/
theorem jack_second_half_time_is_six :
  ∀ (jill_time first_half_time time_diff : ℕ),
    jill_time = 32 →
    first_half_time = 19 →
    time_diff = 7 →
    jack_second_half_time (jack_total_time jill_time first_half_time time_diff) first_half_time = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_second_half_time_is_six_l522_52254


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l522_52277

theorem shortest_altitude_of_triangle (a b c : ℝ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) :
  ∃ h : ℝ, h = 9.6 ∧ h ≤ min a b ∧ h ≤ (2 * (a * b) / c) := by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l522_52277


namespace NUMINAMATH_CALUDE_survey_result_l522_52261

/-- Calculates the percentage of the surveyed population that supports a new environmental policy. -/
def survey_support_percentage (men_support_rate : ℚ) (women_support_rate : ℚ) (men_count : ℕ) (women_count : ℕ) : ℚ :=
  let total_count := men_count + women_count
  let supporting_count := men_support_rate * men_count + women_support_rate * women_count
  supporting_count / total_count

/-- Theorem stating that given the survey conditions, 74% of the population supports the policy. -/
theorem survey_result :
  let men_support_rate : ℚ := 70 / 100
  let women_support_rate : ℚ := 75 / 100
  let men_count : ℕ := 200
  let women_count : ℕ := 800
  survey_support_percentage men_support_rate women_support_rate men_count women_count = 74 / 100 := by
  sorry

#eval survey_support_percentage (70 / 100) (75 / 100) 200 800

end NUMINAMATH_CALUDE_survey_result_l522_52261


namespace NUMINAMATH_CALUDE_count_palindrome_pairs_l522_52209

def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (n / 1000 = n % 10) ∧ 
  ((n / 100) % 10 = (n / 10) % 10)

def palindrome_pair (p1 p2 : ℕ) : Prop :=
  is_four_digit_palindrome p1 ∧ 
  is_four_digit_palindrome p2 ∧ 
  p1 - p2 = 3674

theorem count_palindrome_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S ↔ palindrome_pair p.1 p.2) ∧ 
    Finset.card S = 35 := by
  sorry

end NUMINAMATH_CALUDE_count_palindrome_pairs_l522_52209


namespace NUMINAMATH_CALUDE_school_gender_ratio_l522_52283

/-- The number of boys in the school -/
def num_boys : ℕ := 50

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys + 80

/-- The ratio of boys to girls as a pair of natural numbers -/
def boys_to_girls_ratio : ℕ × ℕ := (5, 13)

theorem school_gender_ratio :
  (num_boys, num_girls) = (boys_to_girls_ratio.1 * 10, boys_to_girls_ratio.2 * 10) := by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l522_52283


namespace NUMINAMATH_CALUDE_method_two_more_swims_at_300_method_one_cheaper_for_40_plus_swims_l522_52211

-- Define the cost functions for both methods
def cost_method_one (x : ℕ) : ℕ := 120 + 10 * x
def cost_method_two (x : ℕ) : ℕ := 15 * x

-- Theorem 1: For a total cost of 300 yuan, Method two allows more swims
theorem method_two_more_swims_at_300 :
  ∃ (x y : ℕ), cost_method_one x = 300 ∧ cost_method_two y = 300 ∧ y > x :=
sorry

-- Theorem 2: For 40 or more swims, Method one is less expensive
theorem method_one_cheaper_for_40_plus_swims :
  ∀ x : ℕ, x ≥ 40 → cost_method_one x < cost_method_two x :=
sorry

end NUMINAMATH_CALUDE_method_two_more_swims_at_300_method_one_cheaper_for_40_plus_swims_l522_52211


namespace NUMINAMATH_CALUDE_abc_sum_eq_three_l522_52243

theorem abc_sum_eq_three (a b c : ℕ+) 
  (h1 : c = b^2)
  (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) :
  a + b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_eq_three_l522_52243


namespace NUMINAMATH_CALUDE_snow_leopard_lineup_l522_52268

/-- The number of ways to arrange 9 distinct objects in a row, 
    where 3 specific objects must be placed at the ends and middle -/
def arrangement_count : ℕ := 4320

/-- The number of ways to arrange 3 objects in 3 specific positions -/
def short_leopard_arrangements : ℕ := 6

/-- The number of ways to arrange the remaining 6 objects -/
def remaining_leopard_arrangements : ℕ := 720

theorem snow_leopard_lineup : 
  arrangement_count = short_leopard_arrangements * remaining_leopard_arrangements :=
sorry

end NUMINAMATH_CALUDE_snow_leopard_lineup_l522_52268


namespace NUMINAMATH_CALUDE_estimate_population_size_l522_52270

theorem estimate_population_size (sample1 : ℕ) (sample2 : ℕ) (overlap : ℕ) (total : ℕ) : 
  sample1 = 80 → sample2 = 100 → overlap = 20 → 
  (sample1 : ℝ) / total * ((sample2 : ℝ) / total) = (overlap : ℝ) / total → 
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_estimate_population_size_l522_52270


namespace NUMINAMATH_CALUDE_rectangle_area_l522_52238

/-- Given a rectangle where the length is five times the width and the perimeter is 180 cm,
    prove that its area is 1125 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 5 * w
  2 * l + 2 * w = 180 → l * w = 1125 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l522_52238


namespace NUMINAMATH_CALUDE_inequality_proof_l522_52298

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 1) :
  a^x + b^y + c^z ≥ (4*a*b*c*x*y*z) / (x + y + z - 3)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l522_52298


namespace NUMINAMATH_CALUDE_fifth_month_sale_proof_l522_52214

/-- Calculates the sale in the fifth month given the sales of other months and the average -/
def fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) : ℕ :=
  6 * avg - (m1 + m2 + m3 + m4 + m6)

/-- Proves that the sale in the fifth month is 3562 given the specified conditions -/
theorem fifth_month_sale_proof :
  fifth_month_sale 3435 3927 3855 4230 1991 3500 = 3562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_proof_l522_52214


namespace NUMINAMATH_CALUDE_boat_upstream_time_l522_52257

/-- Proves that the time taken by a boat to cover a distance upstream is 1.5 hours,
    given the conditions of the problem. -/
theorem boat_upstream_time (distance : ℝ) (stream_speed : ℝ) (boat_speed : ℝ) : 
  stream_speed = 3 →
  boat_speed = 15 →
  distance = (boat_speed + stream_speed) * 1 →
  (distance / (boat_speed - stream_speed)) = 1.5 := by
sorry

end NUMINAMATH_CALUDE_boat_upstream_time_l522_52257


namespace NUMINAMATH_CALUDE_calculate_savings_l522_52251

/-- Given a person's income and expenditure ratio, and their income, calculate their savings. -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) 
    (h1 : income_ratio = 8)
    (h2 : expenditure_ratio = 7)
    (h3 : income = 40000) :
  income - (expenditure_ratio * income / income_ratio) = 5000 := by
  sorry

#check calculate_savings

end NUMINAMATH_CALUDE_calculate_savings_l522_52251


namespace NUMINAMATH_CALUDE_total_rabbits_l522_52227

theorem total_rabbits (white_rabbits black_rabbits : ℕ) 
  (hw : white_rabbits = 15) 
  (hb : black_rabbits = 37) : 
  white_rabbits + black_rabbits = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_rabbits_l522_52227


namespace NUMINAMATH_CALUDE_cheolsu_number_problem_l522_52207

theorem cheolsu_number_problem (x : ℚ) : 
  x + (-5/12) - (-5/2) = 1/3 → x = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_cheolsu_number_problem_l522_52207


namespace NUMINAMATH_CALUDE_students_doing_hula_hoops_l522_52262

theorem students_doing_hula_hoops 
  (jumping_rope : ℕ) 
  (hula_hoop_ratio : ℕ) 
  (h1 : jumping_rope = 7)
  (h2 : hula_hoop_ratio = 5) :
  jumping_rope * hula_hoop_ratio = 35 := by
  sorry

end NUMINAMATH_CALUDE_students_doing_hula_hoops_l522_52262


namespace NUMINAMATH_CALUDE_sin_cos_sum_21_39_l522_52284

theorem sin_cos_sum_21_39 : 
  Real.sin (21 * π / 180) * Real.cos (39 * π / 180) + 
  Real.cos (21 * π / 180) * Real.sin (39 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_21_39_l522_52284


namespace NUMINAMATH_CALUDE_bug_flower_consumption_l522_52204

theorem bug_flower_consumption (num_bugs : ℝ) (flowers_per_bug : ℝ) : 
  num_bugs = 2.0 → flowers_per_bug = 1.5 → num_bugs * flowers_per_bug = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_bug_flower_consumption_l522_52204


namespace NUMINAMATH_CALUDE_problem_solution_l522_52266

theorem problem_solution :
  (∃ m_max : ℝ, 
    (∀ m : ℝ, (∀ x : ℝ, |x + 3| + |x + m| ≥ 2 * m) → m ≤ m_max) ∧
    (∀ x : ℝ, |x + 3| + |x + m_max| ≥ 2 * m_max) ∧
    m_max = 1) ∧
  (∀ a b c : ℝ, 
    a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    2 * a^2 + 3 * b^2 + 4 * c^2 ≥ 12/13 ∧
    (2 * a^2 + 3 * b^2 + 4 * c^2 = 12/13 ↔ a = 6/13 ∧ b = 4/13 ∧ c = 3/13)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l522_52266


namespace NUMINAMATH_CALUDE_divisible_by_nine_l522_52236

theorem divisible_by_nine (k : ℕ+) : 9 ∣ (3 * (2 + 7^(k : ℕ))) := by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l522_52236


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l522_52226

theorem rectangular_box_volume (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : ∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 4 * k ∧ c = 5 * k) :
  a * b * c = 320 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l522_52226


namespace NUMINAMATH_CALUDE_triangle_properties_l522_52223

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.sin t.B - t.b * Real.cos t.A = t.b)
  (h2 : t.b + t.c = 4) :
  t.A = π / 3 ∧ 
  (∃ (min_a : ℝ), min_a = 2 ∧ 
    (∀ a', t.a = a' → a' ≥ min_a) ∧
    (t.a = min_a → Real.sqrt 3 / 2 * t.b * t.c = Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l522_52223


namespace NUMINAMATH_CALUDE_flower_pots_on_path_l522_52232

/-- Calculates the number of flower pots on a path -/
def flowerPots (pathLength : ℕ) (interval : ℕ) : ℕ :=
  pathLength / interval + 1

/-- Theorem: On a 15-meter path with flower pots every 3 meters, there are 6 flower pots -/
theorem flower_pots_on_path : flowerPots 15 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_on_path_l522_52232


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_l522_52249

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem tangent_slope_at_2 :
  (deriv f) 2 = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_l522_52249


namespace NUMINAMATH_CALUDE_cube_sum_over_product_equals_three_l522_52208

theorem cube_sum_over_product_equals_three
  (p q r : ℝ)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_sum : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_equals_three_l522_52208


namespace NUMINAMATH_CALUDE_forty_sheep_eat_forty_bags_l522_52285

/-- The number of bags of grass eaten by a group of sheep -/
def bags_eaten (num_sheep : ℕ) (num_days : ℕ) : ℕ :=
  num_sheep * (num_days / 40)

/-- Theorem: 40 sheep eat 40 bags of grass in 40 days -/
theorem forty_sheep_eat_forty_bags :
  bags_eaten 40 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_sheep_eat_forty_bags_l522_52285


namespace NUMINAMATH_CALUDE_number_machine_input_l522_52230

/-- A number machine that adds 15 and then subtracts 6 -/
def number_machine (x : ℤ) : ℤ := x + 15 - 6

/-- Theorem stating that if the number machine outputs 77, the input must have been 68 -/
theorem number_machine_input (x : ℤ) : number_machine x = 77 → x = 68 := by
  sorry

end NUMINAMATH_CALUDE_number_machine_input_l522_52230


namespace NUMINAMATH_CALUDE_remainder_problem_l522_52229

theorem remainder_problem (N : ℤ) : N % 899 = 63 → N % 29 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l522_52229


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l522_52231

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l522_52231


namespace NUMINAMATH_CALUDE_fraction_equality_l522_52219

theorem fraction_equality : (5 * 3 + 4) / 7 = 19 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l522_52219


namespace NUMINAMATH_CALUDE_min_value_reciprocal_l522_52237

theorem min_value_reciprocal (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + x * y + 2 * y = 30 → 1 / (a * b) ≤ 1 / (x * y)) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + x * y + 2 * y = 30 ∧ 1 / (x * y) = 1 / 18) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_l522_52237


namespace NUMINAMATH_CALUDE_andrews_grapes_l522_52292

/-- The amount of grapes Andrew purchased -/
def grapes : ℕ := sorry

/-- The price of grapes per kg -/
def grape_price : ℕ := 98

/-- The amount of mangoes Andrew purchased in kg -/
def mangoes : ℕ := 7

/-- The price of mangoes per kg -/
def mango_price : ℕ := 50

/-- The total amount Andrew paid -/
def total_paid : ℕ := 1428

theorem andrews_grapes : 
  grapes * grape_price + mangoes * mango_price = total_paid ∧ grapes = 11 := by sorry

end NUMINAMATH_CALUDE_andrews_grapes_l522_52292


namespace NUMINAMATH_CALUDE_triangle_cos_C_eq_neg_one_fourth_l522_52203

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C opposite to these sides respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The law of sines for a triangle -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The law of cosines for a triangle -/
axiom law_of_cosines (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

theorem triangle_cos_C_eq_neg_one_fourth (t : Triangle) 
  (ha : t.a = 2)
  (hc : t.c = 4)
  (h_sin : 3 * Real.sin t.A = 2 * Real.sin t.B) :
  Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_C_eq_neg_one_fourth_l522_52203


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l522_52218

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l522_52218


namespace NUMINAMATH_CALUDE_hyperbola_properties_l522_52240

/-- A hyperbola with equation y²/2 - x²/4 = 1 -/
def hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 / 4 = 1

/-- The reference hyperbola with equation x²/2 - y² = 1 -/
def reference_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

theorem hyperbola_properties :
  (∃ (x y : ℝ), hyperbola x y ∧ x = 2 ∧ y = -2) ∧
  (∀ (x y : ℝ), ∃ (k : ℝ), hyperbola x y ↔ reference_hyperbola (x * k) (y * k)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l522_52240


namespace NUMINAMATH_CALUDE_f_properties_l522_52255

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is an odd function and monotonically increasing
theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l522_52255


namespace NUMINAMATH_CALUDE_vowel_writing_count_l522_52272

theorem vowel_writing_count (num_vowels : ℕ) (total_alphabets : ℕ) : 
  num_vowels = 5 → 
  total_alphabets = 10 → 
  ∃ (times_written : ℕ), times_written * num_vowels = total_alphabets ∧ times_written = 2 :=
by sorry

end NUMINAMATH_CALUDE_vowel_writing_count_l522_52272


namespace NUMINAMATH_CALUDE_smallest_number_l522_52252

theorem smallest_number (a b c d e : ℝ) (ha : a = 1.4) (hb : b = 1.2) (hc : c = 2.0) (hd : d = 1.5) (he : e = 2.1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d ∧ b ≤ e := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l522_52252


namespace NUMINAMATH_CALUDE_tv_and_radio_clients_l522_52247

def total_clients : ℕ := 180
def tv_clients : ℕ := 115
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine : ℕ := 85
def radio_and_magazine : ℕ := 95
def all_three : ℕ := 80

theorem tv_and_radio_clients : 
  total_clients = tv_clients + radio_clients + magazine_clients - tv_and_magazine - radio_and_magazine - (tv_clients + radio_clients - total_clients) + all_three := by
  sorry

end NUMINAMATH_CALUDE_tv_and_radio_clients_l522_52247


namespace NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l522_52234

/-- 
A regular polygon where the sum of interior angles is 4 times the sum of exterior angles.
-/
structure RegularPolygon where
  n : ℕ  -- number of sides
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  h1 : sum_interior_angles = (n - 2) * 180
  h2 : sum_exterior_angles = 360
  h3 : sum_interior_angles = 4 * sum_exterior_angles

/-- The number of sides of the regular polygon is 10. -/
theorem number_of_sides (p : RegularPolygon) : p.n = 10 := by
  sorry

/-- The measure of each interior angle of the regular polygon is 144°. -/
theorem interior_angle_measure (p : RegularPolygon) : 
  (p.n - 2) * 180 / p.n = 144 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l522_52234


namespace NUMINAMATH_CALUDE_art_project_marker_distribution_l522_52258

theorem art_project_marker_distribution (total_students : ℕ) (total_boxes : ℕ) (markers_per_box : ℕ)
  (group1_students : ℕ) (group1_markers : ℕ) (group2_students : ℕ) (group2_markers : ℕ) :
  total_students = 30 →
  total_boxes = 22 →
  markers_per_box = 5 →
  group1_students = 10 →
  group1_markers = 2 →
  group2_students = 15 →
  group2_markers = 4 →
  (total_students - group1_students - group2_students) > 0 →
  (total_boxes * markers_per_box - group1_students * group1_markers - group2_students * group2_markers) /
    (total_students - group1_students - group2_students) = 6 :=
by sorry

end NUMINAMATH_CALUDE_art_project_marker_distribution_l522_52258


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l522_52212

theorem trivia_team_tryouts (not_picked : ℕ) (num_groups : ℕ) (students_per_group : ℕ) : 
  not_picked = 9 → num_groups = 3 → students_per_group = 9 → 
  not_picked + num_groups * students_per_group = 36 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l522_52212


namespace NUMINAMATH_CALUDE_rotated_rectangle_area_fraction_l522_52288

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a rectangle on a 2D grid -/
structure Rectangle where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a rectangle given its vertices -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry

/-- Calculates the area of a square grid -/
def gridArea (size : ℤ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem rotated_rectangle_area_fraction :
  let grid_size : ℤ := 6
  let r : Rectangle := {
    v1 := { x := 2, y := 2 },
    v2 := { x := 4, y := 4 },
    v3 := { x := 2, y := 4 },
    v4 := { x := 4, y := 6 }
  }
  rectangleArea r / gridArea grid_size = Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rotated_rectangle_area_fraction_l522_52288


namespace NUMINAMATH_CALUDE_paul_candy_count_l522_52205

theorem paul_candy_count :
  ∀ (chocolate_boxes caramel_boxes pieces_per_box : ℕ),
    chocolate_boxes = 6 →
    caramel_boxes = 4 →
    pieces_per_box = 9 →
    chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_paul_candy_count_l522_52205


namespace NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l522_52287

theorem quadratic_coefficient_bounds (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hroots : b^2 - 4*a*c ≥ 0) : 
  (max a (max b c) ≥ 4/9 * (a + b + c)) ∧ 
  (min a (min b c) ≤ 1/4 * (a + b + c)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l522_52287


namespace NUMINAMATH_CALUDE_cat_weight_l522_52244

/-- Given a cat and a dog with specific weight relationships, prove the cat's weight -/
theorem cat_weight (cat_weight dog_weight : ℝ) : 
  dog_weight = cat_weight + 6 →
  cat_weight = dog_weight / 3 →
  cat_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_cat_weight_l522_52244


namespace NUMINAMATH_CALUDE_three_digit_powers_of_two_l522_52248

theorem three_digit_powers_of_two (n : ℕ) : 
  (∃ k, 100 ≤ 2^k ∧ 2^k ≤ 999) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_two_l522_52248


namespace NUMINAMATH_CALUDE_batsman_average_is_35_l522_52221

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  total_runs : ℕ
  last_inning_runs : ℕ
  average_increase : ℕ

/-- Calculates the new average of a batsman after their latest inning -/
def new_average (b : Batsman) : ℚ :=
  (b.total_runs + b.last_inning_runs) / b.innings

/-- Theorem stating that under given conditions, the batsman's new average is 35 -/
theorem batsman_average_is_35 (b : Batsman) 
    (h1 : b.innings = 17)
    (h2 : b.last_inning_runs = 83)
    (h3 : b.average_increase = 3)
    (h4 : new_average b = (new_average b - b.average_increase) + b.average_increase) :
    new_average b = 35 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_is_35_l522_52221


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l522_52280

-- Define the triangles and their side lengths
structure Triangle :=
  (a b c : ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

-- State the theorem
theorem similar_triangles_side_length 
  (PQR STU : Triangle) 
  (h_similar : similar PQR STU) 
  (h_PQ : PQR.a = 7) 
  (h_QR : PQR.b = 10) 
  (h_ST : STU.a = 4.9) : 
  STU.b = 7 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l522_52280


namespace NUMINAMATH_CALUDE_one_and_one_third_of_number_is_48_l522_52271

theorem one_and_one_third_of_number_is_48 :
  ∃ x : ℚ, (4 / 3) * x = 48 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_of_number_is_48_l522_52271


namespace NUMINAMATH_CALUDE_gcd_lcm_1729_867_l522_52241

theorem gcd_lcm_1729_867 :
  (Nat.gcd 1729 867 = 1) ∧ (Nat.lcm 1729 867 = 1499003) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_1729_867_l522_52241


namespace NUMINAMATH_CALUDE_fifth_group_sample_l522_52265

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  num_groups : ℕ
  group_size : ℕ
  first_sample : ℕ

/-- Calculates the sample number for a given group in a systematic sampling scenario -/
def sample_number (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_sample + (group - 1) * s.group_size

/-- Theorem: In the given systematic sampling scenario, the sample number in the fifth group is 43 -/
theorem fifth_group_sample (s : SystematicSampling) 
  (h1 : s.population = 60)
  (h2 : s.num_groups = 6)
  (h3 : s.group_size = s.population / s.num_groups)
  (h4 : s.first_sample = 3) :
  sample_number s 5 = 43 := by
  sorry


end NUMINAMATH_CALUDE_fifth_group_sample_l522_52265


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l522_52276

/-- A geometric sequence with real number terms -/
def GeometricSequence := ℕ → ℝ

/-- Sum of the first n terms of a geometric sequence -/
def SumN (a : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_sum (a : GeometricSequence) :
  SumN a 10 = 10 →
  SumN a 30 = 70 →
  SumN a 40 = 150 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l522_52276


namespace NUMINAMATH_CALUDE_interval_length_theorem_l522_52217

theorem interval_length_theorem (a b : ℝ) : 
  (∃ x : ℝ, a ≤ 2*x + 3 ∧ 2*x + 3 ≤ b) ∧ 
  ((b - 3) / 2 - (a - 3) / 2 = 10) → 
  b - a = 20 := by
sorry

end NUMINAMATH_CALUDE_interval_length_theorem_l522_52217


namespace NUMINAMATH_CALUDE_L₂_equations_l522_52201

noncomputable section

-- Define the line L₁
def L₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | 6 * p.1 - p.2 + 6 = 0}

-- Define points P and Q
def P : ℝ × ℝ := (-1, 0)
def Q : ℝ × ℝ := (0, 6)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a general line L₂ passing through (1,0)
def L₂ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 - m}

-- Define point R as the intersection of L₂ and y-axis
def R (m : ℝ) : ℝ × ℝ := (0, -m)

-- Define point S as the intersection of L₁ and L₂
def S (m : ℝ) : ℝ × ℝ := ((-m - 6) / (6 - m), (-12 * m) / (6 - m))

-- Define the area of a triangle given three points
def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2))

-- State the theorem
theorem L₂_equations : 
  ∀ m : ℝ, (triangleArea O P Q = 6 * triangleArea Q (R m) (S m)) → 
  (m = -3 ∨ m = -10) :=
sorry

end NUMINAMATH_CALUDE_L₂_equations_l522_52201


namespace NUMINAMATH_CALUDE_division_result_l522_52233

theorem division_result : (2014 : ℕ) / (2 * 2 + 2 * 3 + 3 * 3) = 106 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l522_52233


namespace NUMINAMATH_CALUDE_solution_replacement_l522_52273

theorem solution_replacement (initial_volume : ℝ) (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) 
  (h1 : initial_volume = 100)
  (h2 : initial_concentration = 0.4)
  (h3 : replacement_concentration = 0.25)
  (h4 : final_concentration = 0.35) :
  ∃ (replaced_volume : ℝ), 
    replaced_volume / initial_volume = 1 / 3 ∧
    initial_volume * initial_concentration - replaced_volume * initial_concentration + 
    replaced_volume * replacement_concentration = 
    initial_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_solution_replacement_l522_52273


namespace NUMINAMATH_CALUDE_original_number_proof_l522_52297

theorem original_number_proof (x : ℝ) : x * 1.1 = 660 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l522_52297


namespace NUMINAMATH_CALUDE_student_tickets_sold_l522_52279

/-- Proves the number of student tickets sold given total tickets, total money, and ticket prices -/
theorem student_tickets_sold 
  (total_tickets : ℕ) 
  (total_money : ℕ) 
  (student_price : ℕ) 
  (nonstudent_price : ℕ) 
  (h1 : total_tickets = 821)
  (h2 : total_money = 1933)
  (h3 : student_price = 2)
  (h4 : nonstudent_price = 3) :
  ∃ (student_tickets : ℕ) (nonstudent_tickets : ℕ),
    student_tickets + nonstudent_tickets = total_tickets ∧
    student_tickets * student_price + nonstudent_tickets * nonstudent_price = total_money ∧
    student_tickets = 530 := by
  sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l522_52279


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l522_52213

noncomputable def f (x : ℝ) := Real.log (x^2 + 2*x - 3) / Real.log (1/2)

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio (-3)) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l522_52213


namespace NUMINAMATH_CALUDE_shortest_path_length_l522_52289

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a circle -/
structure Circle where
  radius : ℝ

/-- Represents a path on a triangle and circle -/
def ShortestPath (t : EquilateralTriangle) (c : Circle) : ℝ := sorry

/-- The theorem stating the length of the shortest path -/
theorem shortest_path_length 
  (t : EquilateralTriangle) 
  (c : Circle) 
  (h1 : t.sideLength = 2) 
  (h2 : c.radius = 1/2) : 
  ShortestPath t c = Real.sqrt (28/3) - 1 := by sorry

end NUMINAMATH_CALUDE_shortest_path_length_l522_52289


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_three_twentieths_decimal_l522_52250

theorem fraction_to_decimal :
  (3 : ℚ) / 20 = (15 : ℚ) / 100 := by sorry

theorem decimal_representation :
  (15 : ℚ) / 100 = 0.15 := by sorry

theorem three_twentieths_decimal :
  (3 : ℚ) / 20 = 0.15 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_three_twentieths_decimal_l522_52250


namespace NUMINAMATH_CALUDE_bus_cost_proof_l522_52260

-- Define the cost of a bus ride
def bus_cost : ℝ := 3.75

-- Define the cost of a train ride
def train_cost : ℝ := bus_cost + 2.35

-- Theorem stating the conditions and the result to be proved
theorem bus_cost_proof :
  (train_cost = bus_cost + 2.35) ∧
  (train_cost + bus_cost = 9.85) →
  bus_cost = 3.75 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_cost_proof_l522_52260


namespace NUMINAMATH_CALUDE_sequence_min_value_and_ratio_l522_52267

/-- Given a positive integer m ≥ 3, an arithmetic sequence {a_n} with positive terms,
    and a geometric sequence {b_n} with positive terms, such that:
    1. The first term of {a_n} equals the common ratio of {b_n}
    2. The first term of {b_n} equals the common difference of {a_n}
    3. a_m = b_m
    This theorem proves the minimum value of a_m and the ratio of a_1 to b_1 when a_m is minimum. -/
theorem sequence_min_value_and_ratio (m : ℕ) (a b : ℝ → ℝ) (h_m : m ≥ 3) 
  (h_a_pos : ∀ n, a n > 0) (h_b_pos : ∀ n, b n > 0)
  (h_a_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_b_geom : ∀ n, b (n + 1) / b n = b 2 / b 1)
  (h_first_term : a 1 = b 2 / b 1)
  (h_common_diff : b 1 = a 2 - a 1)
  (h_m_equal : a m = b m) :
  ∃ (min_am : ℝ) (ratio : ℝ),
    min_am = ((m^m : ℝ) / ((m - 1 : ℝ)^(m - 2)))^(1 / (m - 1 : ℝ)) ∧
    ratio = (m - 1 : ℝ)^2 ∧
    a m ≥ min_am ∧
    (a m = min_am → a 1 / b 1 = ratio) := by
  sorry

end NUMINAMATH_CALUDE_sequence_min_value_and_ratio_l522_52267


namespace NUMINAMATH_CALUDE_smallest_class_size_l522_52225

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    scores i = 100 ∧ scores j = 100 ∧ scores k = 100) →
  (∀ i : Fin n, scores i ≥ 70) →
  (∀ i : Fin n, scores i ≤ 100) →
  (Finset.sum Finset.univ scores / n = 85) →
  n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l522_52225


namespace NUMINAMATH_CALUDE_kids_waiting_swings_is_three_l522_52263

/-- The number of kids waiting for the swings -/
def kids_waiting_swings : ℕ := sorry

/-- The number of kids waiting for the slide -/
def kids_waiting_slide : ℕ := 2 * kids_waiting_swings

/-- The wait time for the swings in seconds -/
def wait_time_swings : ℕ := 120 * kids_waiting_swings

/-- The wait time for the slide in seconds -/
def wait_time_slide : ℕ := 15 * kids_waiting_slide

/-- The difference between the longer and shorter wait times -/
def wait_time_difference : ℕ := 270

theorem kids_waiting_swings_is_three :
  kids_waiting_swings = 3 ∧
  kids_waiting_slide = 2 * kids_waiting_swings ∧
  wait_time_swings = 120 * kids_waiting_swings ∧
  wait_time_slide = 15 * kids_waiting_slide ∧
  wait_time_swings - wait_time_slide = wait_time_difference :=
by sorry

end NUMINAMATH_CALUDE_kids_waiting_swings_is_three_l522_52263


namespace NUMINAMATH_CALUDE_total_yells_is_sixty_l522_52210

/-- Represents the number of times Missy yells at her dogs -/
structure DogYells where
  obedient : ℕ
  stubborn : ℕ

/-- The relationship between yells at obedient and stubborn dogs -/
def stubborn_to_obedient_ratio : ℕ := 4

/-- The number of times Missy yells at her obedient dog -/
def obedient_yells : ℕ := 12

/-- Calculates the total number of yells based on the given conditions -/
def total_yells (yells : DogYells) : ℕ :=
  yells.obedient + yells.stubborn

/-- Theorem stating that the total number of yells is 60 -/
theorem total_yells_is_sixty :
  ∃ (yells : DogYells),
    yells.obedient = obedient_yells ∧
    yells.stubborn = stubborn_to_obedient_ratio * obedient_yells ∧
    total_yells yells = 60 := by
  sorry


end NUMINAMATH_CALUDE_total_yells_is_sixty_l522_52210


namespace NUMINAMATH_CALUDE_vasya_driving_distance_l522_52215

theorem vasya_driving_distance 
  (total_distance : ℝ) 
  (anton_distance vasya_distance sasha_distance dima_distance : ℝ) 
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance)
  : vasya_distance = (2 / 5) * total_distance :=
by sorry

end NUMINAMATH_CALUDE_vasya_driving_distance_l522_52215


namespace NUMINAMATH_CALUDE_simple_interest_rate_approx_l522_52278

/-- The rate of simple interest given principal, amount, and time -/
def simple_interest_rate (principal amount : ℕ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that the rate of simple interest is approximately 3.53% -/
theorem simple_interest_rate_approx :
  let rate := simple_interest_rate 12000 17500 13
  ∃ ε > 0, abs (rate - 353/100) < ε ∧ ε < 1/100 :=
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_approx_l522_52278


namespace NUMINAMATH_CALUDE_apple_distribution_l522_52220

theorem apple_distribution (boxes : Nat) (apples_per_box : Nat) (rotten_apples : Nat) (people : Nat) :
  boxes = 7 →
  apples_per_box = 9 →
  rotten_apples = 7 →
  people = 8 →
  (boxes * apples_per_box - rotten_apples) / people = 7 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l522_52220


namespace NUMINAMATH_CALUDE_complex_second_quadrant_x_range_l522_52295

theorem complex_second_quadrant_x_range (x : ℝ) :
  let z : ℂ := (x + Complex.I) / (3 - Complex.I)
  (z.re < 0 ∧ z.im > 0) → (-3 < x ∧ x < 1/3) :=
by sorry

end NUMINAMATH_CALUDE_complex_second_quadrant_x_range_l522_52295


namespace NUMINAMATH_CALUDE_first_item_is_five_l522_52200

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_items : ℕ
  sample_size : ℕ
  item_16 : ℕ

/-- The first item in a systematic sampling scheme -/
def first_item (s : SystematicSampling) : ℕ :=
  s.item_16 - (16 - 1) * (s.total_items / s.sample_size)

/-- Theorem: In the given systematic sampling scheme, the first item is 5 -/
theorem first_item_is_five :
  let s : SystematicSampling := ⟨160, 20, 125⟩
  first_item s = 5 := by sorry

end NUMINAMATH_CALUDE_first_item_is_five_l522_52200


namespace NUMINAMATH_CALUDE_max_k_value_l522_52222

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 8*x + 15 = 0 ∧ 
   y = k*x - 2 ∧ 
   ∃ cx cy : ℝ, cy = k*cx - 2 ∧ 
   (cx - x)^2 + (cy - y)^2 ≤ 1) → 
  k ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l522_52222


namespace NUMINAMATH_CALUDE_solution_verification_l522_52259

-- Define the differential equation
def differential_equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x * (deriv y x) = y x - 1

-- Define the first function
def f₁ (x : ℝ) : ℝ := 3 * x + 1

-- Define the second function (C is a real constant)
def f₂ (C : ℝ) (x : ℝ) : ℝ := C * x + 1

-- Theorem statement
theorem solution_verification :
  (∀ x, x ≠ 0 → differential_equation x f₁) ∧
  (∀ C, ∀ x, x ≠ 0 → differential_equation x (f₂ C)) :=
sorry

end NUMINAMATH_CALUDE_solution_verification_l522_52259


namespace NUMINAMATH_CALUDE_factorial_14_mod_17_l522_52224

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_14_mod_17 : 
  factorial 14 % 17 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_factorial_14_mod_17_l522_52224


namespace NUMINAMATH_CALUDE_division_problem_l522_52293

theorem division_problem (A : ℕ) (h : 34 = A * 6 + 4) : A = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l522_52293


namespace NUMINAMATH_CALUDE_x_plus_inv_x_eight_l522_52299

theorem x_plus_inv_x_eight (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_inv_x_eight_l522_52299


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l522_52256

theorem max_value_of_fraction (k : ℝ) (h : k > 0) :
  (3 * k^3 + 3 * k) / ((3/2 * k^2 + 14) * (14 * k^2 + 3/2)) ≤ Real.sqrt 21 / 175 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l522_52256


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l522_52206

/-- Given points A and B on the parabola y = -x^2 forming an equilateral triangle with the origin,
    prove that their x-coordinates are ±√3 and the side length is 2√3. -/
theorem equilateral_triangle_on_parabola :
  ∀ (a : ℝ),
  let A : ℝ × ℝ := (a, -a^2)
  let B : ℝ × ℝ := (-a, -a^2)
  let O : ℝ × ℝ := (0, 0)
  -- Distance between two points (x₁, y₁) and (x₂, y₂)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- Condition for equilateral triangle
  (dist A O = dist B O ∧ dist A O = dist A B) →
  (a = Real.sqrt 3 ∧ dist A O = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l522_52206


namespace NUMINAMATH_CALUDE_eight_and_half_minutes_in_seconds_l522_52242

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we're converting -/
def minutes : ℚ := 8.5

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℚ) : ℚ := m * seconds_per_minute

theorem eight_and_half_minutes_in_seconds :
  minutes_to_seconds minutes = 510 := by
  sorry

end NUMINAMATH_CALUDE_eight_and_half_minutes_in_seconds_l522_52242


namespace NUMINAMATH_CALUDE_wilmas_garden_rows_l522_52228

/-- The number of rows in Wilma's garden --/
def garden_rows : ℕ :=
  let yellow_flowers : ℕ := 12
  let green_flowers : ℕ := 2 * yellow_flowers
  let red_flowers : ℕ := 42
  let total_flowers : ℕ := yellow_flowers + green_flowers + red_flowers
  let flowers_per_row : ℕ := 13
  total_flowers / flowers_per_row

/-- Theorem stating that the number of rows in Wilma's garden is 6 --/
theorem wilmas_garden_rows :
  garden_rows = 6 := by
  sorry

end NUMINAMATH_CALUDE_wilmas_garden_rows_l522_52228


namespace NUMINAMATH_CALUDE_vans_needed_l522_52282

def van_capacity : ℕ := 5
def num_students : ℕ := 12
def num_adults : ℕ := 3

theorem vans_needed : 
  (num_students + num_adults + van_capacity - 1) / van_capacity = 3 := by
sorry

end NUMINAMATH_CALUDE_vans_needed_l522_52282


namespace NUMINAMATH_CALUDE_optimal_price_for_target_profit_l522_52286

-- Define the cost to produce the souvenir
def production_cost : ℝ := 30

-- Define the lower and upper bounds of the selling price
def min_price : ℝ := production_cost
def max_price : ℝ := 54

-- Define the base price and corresponding daily sales
def base_price : ℝ := 40
def base_sales : ℝ := 80

-- Define the rate of change in sales per yuan increase in price
def sales_change_rate : ℝ := -2

-- Define the target daily profit
def target_profit : ℝ := 1200

-- Define the function for daily sales based on price
def daily_sales (price : ℝ) : ℝ :=
  base_sales + sales_change_rate * (price - base_price)

-- Define the function for daily profit based on price
def daily_profit (price : ℝ) : ℝ :=
  (price - production_cost) * daily_sales price

-- Theorem statement
theorem optimal_price_for_target_profit :
  ∃ (price : ℝ), min_price ≤ price ∧ price ≤ max_price ∧ daily_profit price = target_profit ∧ price = 50 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_for_target_profit_l522_52286


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l522_52245

theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 9 * y^2 = 36 ∧ y = m * x + 3) → 
  m ≤ -Real.sqrt 5 / 3 ∨ m ≥ Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l522_52245


namespace NUMINAMATH_CALUDE_sin_2theta_value_l522_52294

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ ^ 2) ^ n = 3) : 
  Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l522_52294


namespace NUMINAMATH_CALUDE_quadrilateral_is_rhombus_l522_52264

theorem quadrilateral_is_rhombus (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = a*b + b*c + c*d + d*a) : 
  a = b ∧ b = c ∧ c = d := by
  sorry

-- The theorem states that if the given condition is true,
-- then all sides of the quadrilateral are equal,
-- which is the definition of a rhombus.

end NUMINAMATH_CALUDE_quadrilateral_is_rhombus_l522_52264


namespace NUMINAMATH_CALUDE_octagon_perimeter_is_six_l522_52246

/-- A pentagon formed by removing a right-angled isosceles triangle from a unit square -/
structure Pentagon where
  /-- The side length of the original square -/
  squareSide : ℝ
  /-- The length of the leg of the removed right-angled isosceles triangle -/
  triangleLeg : ℝ
  /-- Assertion that the square is a unit square -/
  squareIsUnit : squareSide = 1
  /-- Assertion that the removed triangle is right-angled isosceles with leg length equal to the square side -/
  triangleIsRightIsosceles : triangleLeg = squareSide

/-- An octagon formed by fitting together two congruent pentagons -/
structure Octagon where
  /-- The first pentagon used to form the octagon -/
  pentagon1 : Pentagon
  /-- The second pentagon used to form the octagon -/
  pentagon2 : Pentagon
  /-- Assertion that the two pentagons are congruent -/
  pentagonsAreCongruent : pentagon1 = pentagon2

/-- The perimeter of the octagon -/
def octagonPerimeter (o : Octagon) : ℝ :=
  -- Definition of perimeter calculation goes here
  sorry

/-- Theorem: The perimeter of the octagon is 6 -/
theorem octagon_perimeter_is_six (o : Octagon) : octagonPerimeter o = 6 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_is_six_l522_52246


namespace NUMINAMATH_CALUDE_solution_difference_l522_52202

theorem solution_difference (r s : ℝ) : 
  r ≠ s ∧ 
  (r - 5) * (r + 5) = 25 * r - 125 ∧
  (s - 5) * (s + 5) = 25 * s - 125 ∧
  r > s →
  r - s = 15 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l522_52202


namespace NUMINAMATH_CALUDE_functional_equation_solution_l522_52296

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the only function satisfying the functional equation is f(x) = 1 - x²/2 -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → ∀ x : ℝ, f x = 1 - x^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l522_52296


namespace NUMINAMATH_CALUDE_inscribed_square_area_bound_l522_52275

-- Define an acute triangle
def AcuteTriangle (A B C : Point) : Prop := sorry

-- Define a square
def Square (M N P Q : Point) : Prop := sorry

-- Define a point being on a line segment
def PointOnSegment (P A B : Point) : Prop := sorry

-- Define the area of a polygon
def Area (polygon : Set Point) : ℝ := sorry

theorem inscribed_square_area_bound 
  (A B C M N P Q : Point) 
  (h_acute : AcuteTriangle A B C)
  (h_square : Square M N P Q)
  (h_inscribed : PointOnSegment M B C ∧ PointOnSegment N B C ∧ 
                 PointOnSegment P A C ∧ PointOnSegment Q A B) :
  Area {M, N, P, Q} ≤ (1/2) * Area {A, B, C} := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_bound_l522_52275


namespace NUMINAMATH_CALUDE_powderman_distance_powderman_runs_185_yards_l522_52235

/-- The distance in yards that a powderman runs when he hears a blast, given specific conditions -/
theorem powderman_distance (fuse_time reaction_time : ℝ) (run_speed : ℝ) (sound_speed : ℝ) : ℝ :=
  let blast_time := fuse_time
  let powderman_speed_ft_per_sec := run_speed * 3 -- Convert yards/sec to feet/sec
  let time_of_hearing := (sound_speed * blast_time + powderman_speed_ft_per_sec * reaction_time) / (sound_speed - powderman_speed_ft_per_sec)
  let distance_ft := powderman_speed_ft_per_sec * (time_of_hearing - reaction_time)
  let distance_yd := distance_ft / 3
  distance_yd

/-- The powderman runs 185 yards before hearing the blast under the given conditions -/
theorem powderman_runs_185_yards : 
  powderman_distance 20 2 10 1100 = 185 := by
  sorry


end NUMINAMATH_CALUDE_powderman_distance_powderman_runs_185_yards_l522_52235


namespace NUMINAMATH_CALUDE_inequality_proof_l522_52239

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l522_52239


namespace NUMINAMATH_CALUDE_distance_between_points_l522_52269

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-4, 2)
  let p2 : ℝ × ℝ := (3, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l522_52269


namespace NUMINAMATH_CALUDE_equation_solutions_l522_52281

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 49 = 0 ↔ x = 7/2 ∨ x = -7/2) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l522_52281


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l522_52291

-- Define the parabola
def Parabola := ℝ → ℝ

-- Define the properties of the parabola
axiom parabola_increasing (f : Parabola) : ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂
axiom parabola_decreasing (f : Parabola) : ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂

-- Define the points on the parabola
def A (f : Parabola) := f (-2)
def B (f : Parabola) := f 1
def C (f : Parabola) := f 3

-- State the theorem
theorem parabola_point_ordering (f : Parabola) : B f < C f ∧ C f < A f := by sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l522_52291


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l522_52274

theorem sqrt_sum_equality : Real.sqrt 8 + Real.sqrt 18 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l522_52274
