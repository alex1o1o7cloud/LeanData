import Mathlib

namespace NUMINAMATH_CALUDE_rational_cosine_values_l4011_401120

theorem rational_cosine_values : 
  {k : ℚ | 0 ≤ k ∧ k ≤ 1/2 ∧ ∃ (q : ℚ), Real.cos (k * Real.pi) = q} = {0, 1/2, 1/3} := by sorry

end NUMINAMATH_CALUDE_rational_cosine_values_l4011_401120


namespace NUMINAMATH_CALUDE_mark_vaccine_waiting_time_l4011_401199

/-- Calculates the total waiting time in minutes for Mark's vaccine appointments and effectiveness periods -/
def total_waiting_time : ℕ :=
  let first_vaccine_wait := 4
  let second_vaccine_wait := 20
  let secondary_first_dose_wait := 30 + 10
  let secondary_second_dose_wait := 14 + 3
  let effectiveness_wait := 21
  let total_days := first_vaccine_wait + second_vaccine_wait + secondary_first_dose_wait + 
                    secondary_second_dose_wait + effectiveness_wait
  total_days * 24 * 60

theorem mark_vaccine_waiting_time :
  total_waiting_time = 146880 := by
  sorry

end NUMINAMATH_CALUDE_mark_vaccine_waiting_time_l4011_401199


namespace NUMINAMATH_CALUDE_find_y_value_l4011_401181

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l4011_401181


namespace NUMINAMATH_CALUDE_bus_speed_is_40_l4011_401159

/-- Represents the scenario of a bus and cyclist traveling between points A, B, C, and D. -/
structure TravelScenario where
  distance_AB : ℝ
  time_to_C : ℝ
  distance_CD : ℝ
  bus_speed : ℝ
  cyclist_speed : ℝ

/-- The travel scenario satisfies the given conditions. -/
def satisfies_conditions (s : TravelScenario) : Prop :=
  s.distance_AB = 4 ∧
  s.time_to_C = 1/6 ∧
  s.distance_CD = 2/3 ∧
  s.bus_speed > 0 ∧
  s.cyclist_speed > 0 ∧
  s.bus_speed > s.cyclist_speed

/-- The theorem stating that under the given conditions, the bus speed is 40 km/h. -/
theorem bus_speed_is_40 (s : TravelScenario) (h : satisfies_conditions s) : 
  s.bus_speed = 40 := by
  sorry

#check bus_speed_is_40

end NUMINAMATH_CALUDE_bus_speed_is_40_l4011_401159


namespace NUMINAMATH_CALUDE_total_energy_consumption_l4011_401143

/-- Calculate total electric energy consumption for given appliances over 30 days -/
theorem total_energy_consumption
  (fan_power : Real) (fan_hours : Real)
  (computer_power : Real) (computer_hours : Real)
  (ac_power : Real) (ac_hours : Real)
  (h : fan_power = 75 ∧ fan_hours = 8 ∧
       computer_power = 100 ∧ computer_hours = 5 ∧
       ac_power = 1500 ∧ ac_hours = 3) :
  (fan_power / 1000 * fan_hours +
   computer_power / 1000 * computer_hours +
   ac_power / 1000 * ac_hours) * 30 = 168 := by
sorry

end NUMINAMATH_CALUDE_total_energy_consumption_l4011_401143


namespace NUMINAMATH_CALUDE_job_completion_proof_l4011_401137

/-- The number of days it takes for A to complete the job alone -/
def days_A : ℝ := 10

/-- The number of days A and B work together -/
def days_together : ℝ := 4

/-- The fraction of the job completed after A and B work together -/
def fraction_completed : ℝ := 0.6

/-- The number of days it takes for B to complete the job alone -/
def days_B : ℝ := 20

theorem job_completion_proof :
  (days_together * (1 / days_A + 1 / days_B) = fraction_completed) ∧
  (days_B = 20) := by sorry

end NUMINAMATH_CALUDE_job_completion_proof_l4011_401137


namespace NUMINAMATH_CALUDE_coefficient_x3y7_value_l4011_401186

/-- The coefficient of x^3 * y^7 in the expansion of (x + 1/x - y)^10 -/
def coefficient_x3y7 : ℤ :=
  let n : ℕ := 10
  let k : ℕ := 7
  let m : ℕ := 3
  (-1)^k * (n.choose k) * (m.choose 0)

theorem coefficient_x3y7_value : coefficient_x3y7 = -120 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y7_value_l4011_401186


namespace NUMINAMATH_CALUDE_max_value_fraction_sum_l4011_401156

theorem max_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_sum_l4011_401156


namespace NUMINAMATH_CALUDE_twice_x_minus_one_negative_l4011_401188

theorem twice_x_minus_one_negative (x : ℝ) : (2 * x - 1 < 0) ↔ (∃ y, y = 2 * x - 1 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_twice_x_minus_one_negative_l4011_401188


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l4011_401179

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l4011_401179


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l4011_401165

/-- Check if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Convert a number from one base to another -/
def convertBase (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ k : ℕ, k > 30 → 
    (isPalindrome k 2 ∧ isPalindrome k 6) → 
    k ≥ 55 ∧ 
    isPalindrome 55 2 ∧ 
    isPalindrome 55 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l4011_401165


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l4011_401138

theorem small_cube_edge_length 
  (initial_volume : ℝ) 
  (remaining_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (h1 : initial_volume = 1000) 
  (h2 : remaining_volume = 488) 
  (h3 : num_small_cubes = 8) :
  ∃ (edge_length : ℝ), 
    edge_length = 4 ∧ 
    initial_volume - num_small_cubes * edge_length ^ 3 = remaining_volume :=
by sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l4011_401138


namespace NUMINAMATH_CALUDE_ball_color_probability_l4011_401130

def num_balls : ℕ := 8
def num_colors : ℕ := 2

theorem ball_color_probability :
  let p : ℚ := 1 / 2  -- probability of each color
  let n : ℕ := num_balls
  let k : ℕ := n / 2  -- number of balls of each color
  (n.choose k) * p^n = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_ball_color_probability_l4011_401130


namespace NUMINAMATH_CALUDE_common_difference_is_neg_four_general_term_l4011_401153

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  a_1 : a 1 = 23
  d : ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  a_6_positive : a 6 > 0
  a_7_negative : a 7 < 0

/-- The common difference of the arithmetic sequence is -4 -/
theorem common_difference_is_neg_four (seq : ArithmeticSequence) : seq.d = -4 := by
  sorry

/-- The general term of the arithmetic sequence is -4n + 27 -/
theorem general_term (seq : ArithmeticSequence) (n : ℕ) : seq.a n = -4 * n + 27 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_neg_four_general_term_l4011_401153


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l4011_401101

theorem hyperbola_real_axis_length
  (p : ℝ)
  (a b : ℝ)
  (h_p_pos : p > 0)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_directrix_tangent : 3 + p / 2 = 15)
  (h_asymptote : b / a = Real.sqrt 3)
  (h_focus : a^2 + b^2 = 144) :
  2 * a = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l4011_401101


namespace NUMINAMATH_CALUDE_nathans_blanket_temp_l4011_401136

def initial_temp : ℝ := 50
def type_a_effect : ℝ := 2
def type_b_effect : ℝ := 3
def total_type_a : ℕ := 8
def used_type_a : ℕ := total_type_a / 2
def total_type_b : ℕ := 6

theorem nathans_blanket_temp :
  initial_temp + (used_type_a : ℝ) * type_a_effect + (total_type_b : ℝ) * type_b_effect = 76 := by
  sorry

end NUMINAMATH_CALUDE_nathans_blanket_temp_l4011_401136


namespace NUMINAMATH_CALUDE_roberto_chicken_price_l4011_401152

/-- Represents the scenario of Roberto's chicken and egg expenses --/
structure ChickenEggScenario where
  num_chickens : ℕ
  weekly_feed_cost : ℚ
  eggs_per_chicken_per_week : ℕ
  previous_weekly_egg_cost : ℚ
  break_even_weeks : ℕ

/-- Calculates the price per chicken that makes raising chickens cheaper than buying eggs after a given number of weeks --/
def price_per_chicken (scenario : ChickenEggScenario) : ℚ :=
  (scenario.previous_weekly_egg_cost * scenario.break_even_weeks - scenario.weekly_feed_cost * scenario.break_even_weeks) / scenario.num_chickens

/-- The theorem states that given Roberto's specific scenario, the price per chicken is $20.25 --/
theorem roberto_chicken_price : 
  let scenario : ChickenEggScenario := {
    num_chickens := 4,
    weekly_feed_cost := 1,
    eggs_per_chicken_per_week := 3,
    previous_weekly_egg_cost := 2,
    break_even_weeks := 81
  }
  price_per_chicken scenario = 81/4 := by sorry

end NUMINAMATH_CALUDE_roberto_chicken_price_l4011_401152


namespace NUMINAMATH_CALUDE_root_sum_theorem_l4011_401149

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * (x^2 - 2*x) + 2*x + 3

-- Define the condition for m1 and m2
def condition (m : ℝ) : Prop :=
  ∃ (a b : ℝ), quadratic_equation m a = 0 ∧ quadratic_equation m b = 0 ∧ a/b + b/a = 3/2

-- Theorem statement
theorem root_sum_theorem (m1 m2 : ℝ) :
  condition m1 ∧ condition m2 → m1/m2 + m2/m1 = 833/64 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l4011_401149


namespace NUMINAMATH_CALUDE_book_sale_revenue_l4011_401161

theorem book_sale_revenue (total_books : ℕ) (sold_fraction : ℚ) (price_per_book : ℚ) (remaining_books : ℕ) : 
  sold_fraction = 2/3 →
  price_per_book = 2 →
  remaining_books = 36 →
  (1 - sold_fraction) * total_books = remaining_books →
  sold_fraction * total_books * price_per_book = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l4011_401161


namespace NUMINAMATH_CALUDE_u_closed_form_l4011_401180

def u : ℕ → ℤ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 5 * u (n + 1) - 6 * u n

theorem u_closed_form (n : ℕ) : u n = 2 * 3^n - 2^n := by
  sorry

end NUMINAMATH_CALUDE_u_closed_form_l4011_401180


namespace NUMINAMATH_CALUDE_pencil_pen_difference_l4011_401151

/-- Given a ratio of pens to pencils and the number of pencils, 
    calculate the difference between pencils and pens. -/
theorem pencil_pen_difference 
  (ratio_pens ratio_pencils num_pencils : ℕ) 
  (h_ratio : ratio_pens < ratio_pencils)
  (h_pencils : num_pencils = 42)
  (h_prop : ratio_pens * num_pencils = ratio_pencils * (num_pencils - 7)) :
  num_pencils - (num_pencils - 7) = 7 := by
  sorry

#check pencil_pen_difference 5 6 42

end NUMINAMATH_CALUDE_pencil_pen_difference_l4011_401151


namespace NUMINAMATH_CALUDE_marble_probability_l4011_401124

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) : 
  total = 84 →
  p_white = 1/4 →
  p_green = 1/7 →
  (total : ℚ) * (1 - p_white - p_green) / total = 17/28 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l4011_401124


namespace NUMINAMATH_CALUDE_power_multiplication_l4011_401135

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l4011_401135


namespace NUMINAMATH_CALUDE_triangle_medians_and_area_sum_l4011_401164

theorem triangle_medians_and_area_sum (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let medians_sum := 3 / 4 * (a^2 + b^2 + c^2)
  medians_sum + area^2 = 4033.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_medians_and_area_sum_l4011_401164


namespace NUMINAMATH_CALUDE_value_multiplied_with_b_l4011_401128

theorem value_multiplied_with_b (a b x : ℚ) : 
  a / b = 6 / 5 → 
  (5 * a + x * b) / (5 * a - x * b) = 5 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_value_multiplied_with_b_l4011_401128


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l4011_401144

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l4011_401144


namespace NUMINAMATH_CALUDE_share_difference_l4011_401106

/-- Represents the share of money for each person -/
structure Share where
  amount : ℕ

/-- Represents the distribution of money -/
structure Distribution where
  a : Share
  b : Share
  c : Share
  d : Share

/-- The proposition that a distribution follows the given proportion -/
def follows_proportion (dist : Distribution) : Prop :=
  6 * dist.b.amount = 3 * dist.a.amount ∧
  5 * dist.b.amount = 3 * dist.c.amount ∧
  4 * dist.b.amount = 3 * dist.d.amount

/-- The theorem to be proved -/
theorem share_difference (dist : Distribution) 
  (h1 : follows_proportion dist) 
  (h2 : dist.b.amount = 3000) : 
  dist.c.amount - dist.d.amount = 1000 := by
  sorry


end NUMINAMATH_CALUDE_share_difference_l4011_401106


namespace NUMINAMATH_CALUDE_apple_trees_count_l4011_401157

theorem apple_trees_count (total_trees orange_trees : ℕ) 
  (h1 : total_trees = 74)
  (h2 : orange_trees = 27) :
  total_trees - orange_trees = 47 := by
  sorry

end NUMINAMATH_CALUDE_apple_trees_count_l4011_401157


namespace NUMINAMATH_CALUDE_linear_function_property_l4011_401105

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) (h_linear : LinearFunction g) 
    (h_diff : g 4 - g 1 = 9) : g 10 - g 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l4011_401105


namespace NUMINAMATH_CALUDE_tan_problem_l4011_401115

noncomputable def α : ℝ := Real.arctan 3

theorem tan_problem (h : Real.tan (π - α) = -3) :
  (Real.tan α = 3) ∧
  ((Real.sin (π - α) - Real.cos (π + α) - Real.sin (2*π - α) + Real.cos (-α)) /
   (Real.sin (π/2 - α) + Real.cos (3*π/2 - α)) = -4) :=
by sorry

end NUMINAMATH_CALUDE_tan_problem_l4011_401115


namespace NUMINAMATH_CALUDE_distinct_circular_arrangements_l4011_401185

/-- The number of distinct circular arrangements of girls and boys -/
def circularArrangements (girls boys : ℕ) : ℕ :=
  (Nat.factorial 16 * Nat.factorial 25) / Nat.factorial 9

/-- Theorem stating the number of distinct circular arrangements -/
theorem distinct_circular_arrangements :
  circularArrangements 8 25 = (Nat.factorial 16 * Nat.factorial 25) / Nat.factorial 9 :=
by sorry

end NUMINAMATH_CALUDE_distinct_circular_arrangements_l4011_401185


namespace NUMINAMATH_CALUDE_min_tan_product_l4011_401116

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and satisfying bsinC + csinB = 4asinBsinC, the minimum value of tanAtanBtanC is (12 + 7√3) / 3 -/
theorem min_tan_product (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C →
  (∀ A' B' C' : ℝ,
    0 < A' ∧ A' < π/2 →
    0 < B' ∧ B' < π/2 →
    0 < C' ∧ C' < π/2 →
    A' + B' + C' = π →
    Real.tan A' * Real.tan B' * Real.tan C' ≥ (12 + 7 * Real.sqrt 3) / 3) ∧
  (∃ A' B' C' : ℝ,
    0 < A' ∧ A' < π/2 ∧
    0 < B' ∧ B' < π/2 ∧
    0 < C' ∧ C' < π/2 ∧
    A' + B' + C' = π ∧
    Real.tan A' * Real.tan B' * Real.tan C' = (12 + 7 * Real.sqrt 3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_tan_product_l4011_401116


namespace NUMINAMATH_CALUDE_expand_product_l4011_401118

theorem expand_product (x : ℝ) : (x + 3) * (x - 4) * (x + 1) = x^3 - 13*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4011_401118


namespace NUMINAMATH_CALUDE_larger_number_proof_l4011_401176

/-- Given two positive integers with HCF 23 and LCM factors 11 and 12, the larger is 276 -/
theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a.val b.val = 23) → 
  (∃ (k : ℕ+), Nat.lcm a.val b.val = 23 * 11 * 12 * k.val) → 
  max a b = 276 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4011_401176


namespace NUMINAMATH_CALUDE_games_missed_l4011_401129

theorem games_missed (planned_this_month planned_last_month attended : ℕ) 
  (h1 : planned_this_month = 11)
  (h2 : planned_last_month = 17)
  (h3 : attended = 12) :
  planned_this_month + planned_last_month - attended = 16 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l4011_401129


namespace NUMINAMATH_CALUDE_equation_solution_l4011_401121

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ * (x₁ + 2) = 3 * x₁ + 6) ∧ 
  (x₂ * (x₂ + 2) = 3 * x₂ + 6) ∧ 
  x₁ = -2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x * (x + 2) = 3 * x + 6 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4011_401121


namespace NUMINAMATH_CALUDE_value_of_Y_l4011_401172

theorem value_of_Y : ∀ P Q Y : ℚ,
  P = 6036 / 2 →
  Q = P / 4 →
  Y = P - 3 * Q →
  Y = 754.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_Y_l4011_401172


namespace NUMINAMATH_CALUDE_b_plus_c_equals_seven_l4011_401131

theorem b_plus_c_equals_seven (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : c + d = 5) 
  (h3 : a + d = 2) : 
  b + c = 7 := by sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_seven_l4011_401131


namespace NUMINAMATH_CALUDE_election_votes_calculation_l4011_401168

theorem election_votes_calculation (total_votes : ℕ) : 
  (80 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 380800 →
  total_votes = 560000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l4011_401168


namespace NUMINAMATH_CALUDE_triangle_area_l4011_401162

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 40) :
  (1 / 2) * a * b = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4011_401162


namespace NUMINAMATH_CALUDE_train_length_proof_l4011_401183

/-- Proves that given a train moving at 55 km/hr and a man moving at 7 km/hr in the opposite direction,
    if it takes 10.45077684107852 seconds for the train to pass the man, then the length of the train is 180 meters. -/
theorem train_length_proof (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 55 →
  man_speed = 7 →
  passing_time = 10.45077684107852 →
  (train_speed + man_speed) * (5 / 18) * passing_time = 180 := by
sorry

end NUMINAMATH_CALUDE_train_length_proof_l4011_401183


namespace NUMINAMATH_CALUDE_triangle_side_length_l4011_401154

noncomputable section

/-- Given a triangle ABC with BC = 1, if sin(A/2) * cos(B/2) = sin(B/2) * cos(A/2), then AC = sin(A) / sin(C) -/
theorem triangle_side_length (A B C : Real) (BC : Real) (h1 : BC = 1) 
  (h2 : Real.sin (A / 2) * Real.cos (B / 2) = Real.sin (B / 2) * Real.cos (A / 2)) :
  ∃ (AC : Real), AC = Real.sin A / Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4011_401154


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_general_formula_l4011_401122

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * sequence_a n / (sequence_a n + 2)

def sequence_b (n : ℕ) : ℚ := 1 / sequence_a n

theorem arithmetic_sequence_and_general_formula :
  (∀ n : ℕ, ∃ d : ℚ, sequence_b (n + 1) - sequence_b n = d) ∧
  (∀ n : ℕ, sequence_a n = 2 / (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_general_formula_l4011_401122


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l4011_401107

/-- An arithmetic progression with a non-zero common difference -/
def ArithmeticProgression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The condition that b_n is also an arithmetic progression -/
def BnIsArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d' : ℝ, d' ≠ 0 ∧ ∀ n, a (n + 1) * Real.cos (a (n + 1)) = a n * Real.cos (a n) + d'

/-- The given equation holds for all n -/
def EquationHolds (a : ℕ → ℝ) : Prop :=
  ∀ n, Real.sin (2 * a n) + Real.cos (a (n + 1)) = 0

theorem arithmetic_progression_theorem (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticProgression a d →
  BnIsArithmeticProgression a →
  EquationHolds a →
  (∃ m k : ℤ, k ≠ 0 ∧ 
    ((a 1 = -π / 6 + 2 * π * ↑m ∧ d = 2 * π * ↑k) ∨
     (a 1 = -5 * π / 6 + 2 * π * ↑m ∧ d = 2 * π * ↑k))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l4011_401107


namespace NUMINAMATH_CALUDE_first_pipe_rate_correct_l4011_401193

/-- The rate at which the first pipe pumps water (in gallons per hour) -/
def first_pipe_rate : ℝ := 48

/-- The rate at which the second pipe pumps water (in gallons per hour) -/
def second_pipe_rate : ℝ := 192

/-- The capacity of the well in gallons -/
def well_capacity : ℝ := 1200

/-- The time it takes to fill the well in hours -/
def fill_time : ℝ := 5

theorem first_pipe_rate_correct : 
  first_pipe_rate * fill_time + second_pipe_rate * fill_time = well_capacity := by
  sorry

end NUMINAMATH_CALUDE_first_pipe_rate_correct_l4011_401193


namespace NUMINAMATH_CALUDE_sum_of_first_10_terms_l4011_401104

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2 - 4*n + 1

-- Theorem statement
theorem sum_of_first_10_terms : S 10 = 61 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_10_terms_l4011_401104


namespace NUMINAMATH_CALUDE_find_a_l4011_401140

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5) ∧
  (∀ x : ℝ, x > 3 → (x^2 - 4*x + a) + |x - 3| > 5) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_find_a_l4011_401140


namespace NUMINAMATH_CALUDE_original_rectangle_area_l4011_401158

theorem original_rectangle_area (new_area : ℝ) (h1 : new_area = 32) : ∃ original_area : ℝ,
  (original_area * 4 = new_area) ∧ original_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l4011_401158


namespace NUMINAMATH_CALUDE_walnut_trees_in_park_l4011_401150

theorem walnut_trees_in_park (current : ℕ) (planted : ℕ) (total : ℕ) :
  current + planted = total →
  planted = 55 →
  total = 77 →
  current = 22 := by
sorry

end NUMINAMATH_CALUDE_walnut_trees_in_park_l4011_401150


namespace NUMINAMATH_CALUDE_collinear_iff_sqrt_two_l4011_401160

def a (k : ℝ) : ℝ × ℝ := (k, 2)
def b (k : ℝ) : ℝ × ℝ := (1, k)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t • w.1, t • w.2)

theorem collinear_iff_sqrt_two (k : ℝ) :
  collinear (a k) (b k) ↔ k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_iff_sqrt_two_l4011_401160


namespace NUMINAMATH_CALUDE_total_balls_bought_l4011_401173

/-- Represents the total amount of money Mr. Li had --/
def total_money : ℚ := 1

/-- The cost of a plastic ball --/
def plastic_ball_cost : ℚ := 1 / 60

/-- The cost of a glass ball --/
def glass_ball_cost : ℚ := 1 / 36

/-- The cost of a wooden ball --/
def wooden_ball_cost : ℚ := 1 / 45

/-- The number of plastic balls Mr. Li bought --/
def plastic_balls_bought : ℕ := 10

/-- The number of glass balls Mr. Li bought --/
def glass_balls_bought : ℕ := 10

theorem total_balls_bought : ℕ := by
  -- The total number of balls Mr. Li bought is 45
  sorry

end NUMINAMATH_CALUDE_total_balls_bought_l4011_401173


namespace NUMINAMATH_CALUDE_pet_store_puppies_l4011_401170

/-- The number of puppies sold -/
def puppies_sold : ℕ := 3

/-- The number of cages used -/
def cages_used : ℕ := 3

/-- The number of puppies in each cage -/
def puppies_per_cage : ℕ := 5

/-- The initial number of puppies in the pet store -/
def initial_puppies : ℕ := puppies_sold + cages_used * puppies_per_cage

theorem pet_store_puppies : initial_puppies = 18 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l4011_401170


namespace NUMINAMATH_CALUDE_farm_chickens_l4011_401114

/-- Represents the number of chickens on a farm -/
def num_chickens (total_legs total_animals : ℕ) : ℕ :=
  total_animals - (total_legs - 2 * total_animals) / 2

/-- Theorem stating that given the conditions of the farm, there are 5 chickens -/
theorem farm_chickens : num_chickens 38 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_farm_chickens_l4011_401114


namespace NUMINAMATH_CALUDE_rhombus_area_calculation_l4011_401119

/-- Represents a rhombus -/
structure Rhombus where
  side_length : ℝ
  area : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  ABCD : Rhombus
  BAFC : Rhombus
  AF_parallel_BD : Prop

/-- Main theorem -/
theorem rhombus_area_calculation (setup : ProblemSetup) 
  (h1 : setup.ABCD.side_length = 13)
  (h2 : setup.BAFC.area = 65)
  : setup.ABCD.area = 120 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_calculation_l4011_401119


namespace NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l4011_401167

theorem abc_divides_sum_power_seven (a b c : ℕ+) 
  (hab : a ∣ b^2) (hbc : b ∣ c^2) (hca : c ∣ a^2) : 
  (a * b * c) ∣ (a + b + c)^7 := by
  sorry

end NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l4011_401167


namespace NUMINAMATH_CALUDE_third_shot_probability_l4011_401174

-- Define the probability of hitting the target in one shot
def hit_probability : ℝ := 0.9

-- Define the number of shots
def num_shots : ℕ := 4

-- Define the event of hitting the target on the nth shot
def hit_on_nth_shot (n : ℕ) : ℝ := hit_probability

-- Theorem statement
theorem third_shot_probability :
  hit_on_nth_shot 3 = hit_probability :=
by sorry

end NUMINAMATH_CALUDE_third_shot_probability_l4011_401174


namespace NUMINAMATH_CALUDE_sum_of_cubes_l4011_401139

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l4011_401139


namespace NUMINAMATH_CALUDE_gcd_of_128_144_480_l4011_401184

theorem gcd_of_128_144_480 : Nat.gcd 128 (Nat.gcd 144 480) = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_128_144_480_l4011_401184


namespace NUMINAMATH_CALUDE_product_in_unit_interval_sufficient_not_necessary_l4011_401182

theorem product_in_unit_interval_sufficient_not_necessary (a b : ℝ) :
  (((0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)) → (0 ≤ a * b ∧ a * b ≤ 1)) ∧
  ¬(((0 ≤ a * b ∧ a * b ≤ 1) → ((0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)))) :=
by sorry

end NUMINAMATH_CALUDE_product_in_unit_interval_sufficient_not_necessary_l4011_401182


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l4011_401197

theorem sock_selection_theorem : 
  (Finset.univ.filter (fun x : Finset (Fin 8) => x.card = 4)).card = 70 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l4011_401197


namespace NUMINAMATH_CALUDE_mathematician_paths_l4011_401109

/-- Represents the number of rows in the diagram --/
def num_rows : ℕ := 13

/-- Represents whether the diagram is symmetric --/
def is_symmetric : Prop := true

/-- Represents that each move can be either down-left or down-right --/
def two_move_options : Prop := true

/-- The number of paths spelling "MATHEMATICIAN" in the diagram --/
def num_paths : ℕ := 2^num_rows - 1

theorem mathematician_paths :
  is_symmetric ∧ two_move_options → num_paths = 2^num_rows - 1 := by
  sorry

end NUMINAMATH_CALUDE_mathematician_paths_l4011_401109


namespace NUMINAMATH_CALUDE_cinnamon_distribution_exists_l4011_401196

/-- Represents the number of cinnamon swirls eaten by each person -/
structure CinnamonDistribution where
  jane : ℕ
  siblings : Fin 2 → ℕ
  cousins : Fin 5 → ℕ

/-- Theorem stating the existence of a valid cinnamon swirl distribution -/
theorem cinnamon_distribution_exists : ∃ (d : CinnamonDistribution), 
  -- Each person eats a different number of pieces
  (∀ (i j : Fin 2), i ≠ j → d.siblings i ≠ d.siblings j) ∧ 
  (∀ (i j : Fin 5), i ≠ j → d.cousins i ≠ d.cousins j) ∧
  (∀ (i : Fin 2) (j : Fin 5), d.siblings i ≠ d.cousins j) ∧
  (∀ (i : Fin 2), d.jane ≠ d.siblings i) ∧
  (∀ (j : Fin 5), d.jane ≠ d.cousins j) ∧
  -- Jane eats 1 fewer piece than her youngest sibling
  (∃ (i : Fin 2), d.jane + 1 = d.siblings i ∧ ∀ (j : Fin 2), d.siblings j ≥ d.siblings i) ∧
  -- Jane's youngest sibling eats 2 pieces more than one of her cousins
  (∃ (i : Fin 2) (j : Fin 5), d.siblings i = d.cousins j + 2 ∧ ∀ (k : Fin 2), d.siblings k ≥ d.siblings i) ∧
  -- The sum of all pieces eaten equals 50
  d.jane + (Finset.sum (Finset.univ : Finset (Fin 2)) d.siblings) + (Finset.sum (Finset.univ : Finset (Fin 5)) d.cousins) = 50 :=
sorry

end NUMINAMATH_CALUDE_cinnamon_distribution_exists_l4011_401196


namespace NUMINAMATH_CALUDE_cary_earns_five_per_lawn_l4011_401133

/-- The amount earned per lawn mowed --/
def amount_per_lawn (cost_of_shoes amount_saved lawns_per_weekend num_weekends : ℚ) : ℚ :=
  (cost_of_shoes - amount_saved) / (lawns_per_weekend * num_weekends)

/-- Theorem: Cary earns $5 per lawn mowed --/
theorem cary_earns_five_per_lawn :
  amount_per_lawn 120 30 3 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cary_earns_five_per_lawn_l4011_401133


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l4011_401100

theorem dennis_teaching_years 
  (total_years : ℕ) 
  (virginia_adrienne_diff : ℕ) 
  (dennis_virginia_diff : ℕ) 
  (h1 : total_years = 75)
  (h2 : virginia_adrienne_diff = 9)
  (h3 : dennis_virginia_diff = 9) :
  ∃ (adrienne virginia dennis : ℕ),
    adrienne + virginia + dennis = total_years ∧
    virginia = adrienne + virginia_adrienne_diff ∧
    dennis = virginia + dennis_virginia_diff ∧
    dennis = 34 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l4011_401100


namespace NUMINAMATH_CALUDE_keyboard_warrior_disapproval_l4011_401148

theorem keyboard_warrior_disapproval 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (sample_approving : ℕ) 
  (h1 : total_population = 9600) 
  (h2 : sample_size = 50) 
  (h3 : sample_approving = 14) :
  ⌊(total_population : ℚ) * ((sample_size - sample_approving) : ℚ) / (sample_size : ℚ)⌋ = 6912 := by
  sorry

#check keyboard_warrior_disapproval

end NUMINAMATH_CALUDE_keyboard_warrior_disapproval_l4011_401148


namespace NUMINAMATH_CALUDE_ellipse_intersection_dot_product_range_l4011_401102

def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem ellipse_intersection_dot_product_range :
  ∀ k : ℝ, k ≠ 0 →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    -4 ≤ dot_product x₁ y₁ x₂ y₂ ∧ dot_product x₁ y₁ x₂ y₂ < 13/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_dot_product_range_l4011_401102


namespace NUMINAMATH_CALUDE_cube_pyramid_sum_l4011_401126

/-- A solid figure formed by constructing a pyramid on one face of a cube -/
structure CubePyramid where
  cube_faces : ℕ := 6
  cube_edges : ℕ := 12
  cube_vertices : ℕ := 8
  pyramid_new_faces : ℕ := 4
  pyramid_new_edges : ℕ := 4
  pyramid_new_vertex : ℕ := 1

/-- The total number of exterior faces in the CubePyramid -/
def total_faces (cp : CubePyramid) : ℕ := cp.cube_faces - 1 + cp.pyramid_new_faces

/-- The total number of edges in the CubePyramid -/
def total_edges (cp : CubePyramid) : ℕ := cp.cube_edges + cp.pyramid_new_edges

/-- The total number of vertices in the CubePyramid -/
def total_vertices (cp : CubePyramid) : ℕ := cp.cube_vertices + cp.pyramid_new_vertex

theorem cube_pyramid_sum (cp : CubePyramid) : 
  total_faces cp + total_edges cp + total_vertices cp = 34 := by
  sorry

end NUMINAMATH_CALUDE_cube_pyramid_sum_l4011_401126


namespace NUMINAMATH_CALUDE_regular_triangle_rotation_l4011_401127

/-- The minimum angle of rotation (in degrees) for a regular triangle to coincide with itself. -/
def min_rotation_angle_regular_triangle : ℝ := 120

/-- Theorem stating that the minimum angle of rotation for a regular triangle to coincide with itself is 120 degrees. -/
theorem regular_triangle_rotation :
  min_rotation_angle_regular_triangle = 120 := by sorry

end NUMINAMATH_CALUDE_regular_triangle_rotation_l4011_401127


namespace NUMINAMATH_CALUDE_gcd_g_x_is_20_l4011_401163

def g (x : ℤ) : ℤ := (3*x + 4)*(8*x + 5)*(15*x + 11)*(x + 17)

theorem gcd_g_x_is_20 (x : ℤ) (h : 34560 ∣ x) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 20 := by
sorry

end NUMINAMATH_CALUDE_gcd_g_x_is_20_l4011_401163


namespace NUMINAMATH_CALUDE_triangle_construction_exists_l4011_401145

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given points and line
variable (A B : Point)
variable (bisector : Line)

-- Define the reflection of a point over a line
def reflect (p : Point) (l : Line) : Point :=
  sorry

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  sorry

-- Define a function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

-- Define a function to calculate the distance between two points
def distance (p q : Point) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_construction_exists :
  ∃ C : Point,
    point_on_line C bisector ∧
    distance A C = distance (reflect A bisector) C ∧
    ¬ collinear C A B :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_exists_l4011_401145


namespace NUMINAMATH_CALUDE_fruits_per_slice_l4011_401178

/-- Represents the number of fruits per dozen -/
def dozenSize : ℕ := 12

/-- Represents the number of dozens of Granny Smith apples -/
def grannySmithDozens : ℕ := 4

/-- Represents the number of dozens of Fuji apples -/
def fujiDozens : ℕ := 2

/-- Represents the number of dozens of Bartlett pears -/
def bartlettDozens : ℕ := 3

/-- Represents the number of Granny Smith apple pies -/
def grannySmithPies : ℕ := 4

/-- Represents the number of slices per Granny Smith apple pie -/
def grannySmithSlices : ℕ := 6

/-- Represents the number of Fuji apple pies -/
def fujiPies : ℕ := 3

/-- Represents the number of slices per Fuji apple pie -/
def fujiSlices : ℕ := 8

/-- Represents the number of pear tarts -/
def pearTarts : ℕ := 2

/-- Represents the number of slices per pear tart -/
def pearSlices : ℕ := 10

/-- Theorem stating the number of fruits per slice for each type of pie/tart -/
theorem fruits_per_slice :
  (grannySmithDozens * dozenSize) / (grannySmithPies * grannySmithSlices) = 2 ∧
  (fujiDozens * dozenSize) / (fujiPies * fujiSlices) = 1 ∧
  (bartlettDozens * dozenSize : ℚ) / (pearTarts * pearSlices) = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_fruits_per_slice_l4011_401178


namespace NUMINAMATH_CALUDE_fish_count_difference_l4011_401192

theorem fish_count_difference (n G S R : ℕ) : 
  n > 0 → 
  n = G + S + R → 
  n - G = (2 * n) / 3 - 1 → 
  n - R = (2 * n) / 3 + 4 → 
  S = G + 2 :=
by sorry

end NUMINAMATH_CALUDE_fish_count_difference_l4011_401192


namespace NUMINAMATH_CALUDE_investment_rate_proof_l4011_401171

def total_investment : ℝ := 17000
def investment_at_4_percent : ℝ := 12000
def total_interest : ℝ := 1380
def known_rate : ℝ := 0.04

theorem investment_rate_proof :
  let remaining_investment := total_investment - investment_at_4_percent
  let interest_at_4_percent := investment_at_4_percent * known_rate
  let remaining_interest := total_interest - interest_at_4_percent
  let unknown_rate := remaining_interest / remaining_investment
  unknown_rate = 0.18 := by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l4011_401171


namespace NUMINAMATH_CALUDE_dans_marbles_l4011_401103

/-- Represents the number of marbles Dan has -/
structure Marbles where
  violet : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (m : Marbles) : ℕ := m.violet + m.red + m.blue

/-- Theorem stating the total number of marbles Dan has -/
theorem dans_marbles (x : ℕ) : 
  let initial := Marbles.mk 64 0 0
  let fromMary := Marbles.mk 0 14 0
  let fromJohn := Marbles.mk 0 0 x
  let final := Marbles.mk (initial.violet + fromMary.violet + fromJohn.violet)
                          (initial.red + fromMary.red + fromJohn.red)
                          (initial.blue + fromMary.blue + fromJohn.blue)
  totalMarbles final = 78 + x := by
  sorry

end NUMINAMATH_CALUDE_dans_marbles_l4011_401103


namespace NUMINAMATH_CALUDE_shaded_to_white_ratio_l4011_401117

/-- A square divided into smaller squares where the vertices of inner squares 
    are at the midpoints of the sides of the outer squares -/
structure NestedSquares :=
  (side : ℝ)
  (is_positive : side > 0)

/-- The area of the shaded part in a NestedSquares structure -/
def shaded_area (s : NestedSquares) : ℝ := sorry

/-- The area of the white part in a NestedSquares structure -/
def white_area (s : NestedSquares) : ℝ := sorry

/-- Theorem stating that the ratio of shaded area to white area is 5:3 -/
theorem shaded_to_white_ratio (s : NestedSquares) : 
  shaded_area s / white_area s = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_shaded_to_white_ratio_l4011_401117


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l4011_401166

theorem oil_leak_calculation (total_leak : ℕ) (initial_leak : ℕ) (h1 : total_leak = 11687) (h2 : initial_leak = 6522) :
  total_leak - initial_leak = 5165 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l4011_401166


namespace NUMINAMATH_CALUDE_certain_number_problem_l4011_401169

theorem certain_number_problem (x : ℝ) : 45 * 7 = 0.35 * x → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4011_401169


namespace NUMINAMATH_CALUDE_cloth_selling_price_l4011_401132

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  quantity * (cost_price_per_meter + profit_per_meter)

/-- Proves that the total selling price of 85 meters of cloth with a profit of 20 Rs per meter and a cost price of 85 Rs per meter is 8925 Rs. -/
theorem cloth_selling_price :
  total_selling_price 85 20 85 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l4011_401132


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l4011_401195

theorem orange_juice_fraction : 
  let pitcher_capacity : ℚ := 600
  let pitcher1_fraction : ℚ := 1/3
  let pitcher2_fraction : ℚ := 2/5
  let orange_juice1 : ℚ := pitcher_capacity * pitcher1_fraction
  let orange_juice2 : ℚ := pitcher_capacity * pitcher2_fraction
  let total_orange_juice : ℚ := orange_juice1 + orange_juice2
  let total_mixture : ℚ := pitcher_capacity * 2
  total_orange_juice / total_mixture = 11/30 := by
sorry


end NUMINAMATH_CALUDE_orange_juice_fraction_l4011_401195


namespace NUMINAMATH_CALUDE_license_plate_theorem_l4011_401194

def letter_count : Nat := 26
def digit_count : Nat := 10
def letter_positions : Nat := 5
def digit_positions : Nat := 3

def license_plate_combinations : Nat :=
  letter_count * (Nat.choose (letter_count - 1) (letter_positions - 2)) *
  (Nat.choose letter_positions 2) * (Nat.factorial (letter_positions - 2)) *
  digit_count * (digit_count - 1) * (digit_count - 2)

theorem license_plate_theorem :
  license_plate_combinations = 2594880000 := by sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l4011_401194


namespace NUMINAMATH_CALUDE_max_rabbits_l4011_401147

theorem max_rabbits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both : ℕ) :
  long_ears = 13 →
  jump_far = 17 →
  both ≥ 3 →
  long_ears + jump_far - both ≤ N →
  N ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_rabbits_l4011_401147


namespace NUMINAMATH_CALUDE_sqrt_pattern_l4011_401155

theorem sqrt_pattern (n : ℕ) (hn : n > 0) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = (n^2 + n + 1 : ℝ) / (n * (n+1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l4011_401155


namespace NUMINAMATH_CALUDE_emilys_speed_l4011_401112

/-- Given a distance of 10 miles traveled in 2 hours, prove the speed is 5 miles per hour -/
theorem emilys_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 10)
  (h2 : time = 2)
  (h3 : speed = distance / time) :
  speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_emilys_speed_l4011_401112


namespace NUMINAMATH_CALUDE_lisa_pencils_count_l4011_401198

/-- The number of pencils Gloria has initially -/
def gloria_initial : ℕ := 2

/-- The total number of pencils after Lisa gives hers to Gloria -/
def total_pencils : ℕ := 101

/-- The number of pencils Lisa has initially -/
def lisa_initial : ℕ := total_pencils - gloria_initial

theorem lisa_pencils_count : lisa_initial = 99 := by
  sorry

end NUMINAMATH_CALUDE_lisa_pencils_count_l4011_401198


namespace NUMINAMATH_CALUDE_store_refusal_illegal_l4011_401125

/-- Represents a banknote --/
structure Banknote where
  damaged : Bool
  torn : Bool

/-- Represents the store's action --/
inductive StoreAction
  | Accept
  | Refuse

/-- Defines what constitutes legal tender in Russia --/
def is_legal_tender (b : Banknote) : Bool :=
  b.damaged && b.torn

/-- Determines if a store's action is legal based on the banknote --/
def is_legal_action (b : Banknote) (a : StoreAction) : Prop :=
  is_legal_tender b → a = StoreAction.Accept

/-- The main theorem stating that refusing a torn banknote is illegal --/
theorem store_refusal_illegal (b : Banknote) (h1 : b.damaged) (h2 : b.torn) :
  ¬(is_legal_action b StoreAction.Refuse) := by
  sorry


end NUMINAMATH_CALUDE_store_refusal_illegal_l4011_401125


namespace NUMINAMATH_CALUDE_binomial_2586_1_l4011_401141

theorem binomial_2586_1 : Nat.choose 2586 1 = 2586 := by sorry

end NUMINAMATH_CALUDE_binomial_2586_1_l4011_401141


namespace NUMINAMATH_CALUDE_log_sum_equal_one_power_mult_equal_l4011_401142

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem for the first expression
theorem log_sum_equal_one : log10 2 + log10 5 = 1 := by sorry

-- Theorem for the second expression
theorem power_mult_equal : 4 * (-100)^4 = 400000000 := by sorry

end NUMINAMATH_CALUDE_log_sum_equal_one_power_mult_equal_l4011_401142


namespace NUMINAMATH_CALUDE_cassies_nail_cutting_l4011_401123

/-- The number of nails Cassie needs to cut -/
def total_nails (num_dogs : ℕ) (num_parrots : ℕ) (dog_feet : ℕ) (dog_nails_per_foot : ℕ) 
                (parrot_legs : ℕ) (parrot_claws_per_leg : ℕ) (extra_claw : ℕ) : ℕ :=
  num_dogs * dog_feet * dog_nails_per_foot + 
  (num_parrots - 1) * parrot_legs * parrot_claws_per_leg + 
  (parrot_legs * parrot_claws_per_leg + extra_claw)

/-- Theorem stating the total number of nails Cassie needs to cut -/
theorem cassies_nail_cutting : 
  total_nails 4 8 4 4 2 3 1 = 113 := by
  sorry

end NUMINAMATH_CALUDE_cassies_nail_cutting_l4011_401123


namespace NUMINAMATH_CALUDE_greatest_odd_factors_below_100_l4011_401108

/-- A number has an odd number of positive factors if and only if it is a perfect square. -/
def has_odd_factors (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The greatest whole number less than 100 that has an odd number of positive factors is 81. -/
theorem greatest_odd_factors_below_100 : 
  (∀ m : ℕ, m < 100 → has_odd_factors m → m ≤ 81) ∧ has_odd_factors 81 ∧ 81 < 100 := by
  sorry

end NUMINAMATH_CALUDE_greatest_odd_factors_below_100_l4011_401108


namespace NUMINAMATH_CALUDE_probability_ella_zoe_same_team_l4011_401190

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The card number chosen by Ella -/
def b : ℕ := 11

/-- The probability that Ella and Zoe are on the same team -/
def p (b : ℕ) : ℚ :=
  let remaining_cards := deck_size - 2
  let total_combinations := remaining_cards.choose 2
  let lower_team_combinations := (b - 1).choose 2
  let higher_team_combinations := (deck_size - b - 11).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / total_combinations

theorem probability_ella_zoe_same_team :
  p b = 857 / 1225 :=
sorry

end NUMINAMATH_CALUDE_probability_ella_zoe_same_team_l4011_401190


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l4011_401175

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) -- a is the sequence
  (S : ℕ → ℝ) -- S is the sum function
  (h1 : ∀ n, S n = 3^n - 1) -- Given condition
  (h2 : ∀ n, S n = S (n-1) + a n) -- Property of sum of sequences
  : ∀ n, a n = 2 * 3^(n-1) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l4011_401175


namespace NUMINAMATH_CALUDE_cube_root_two_solves_equation_l4011_401146

theorem cube_root_two_solves_equation :
  let x : ℝ := Real.rpow 2 (1/3)
  (x + 1)^3 = 1 / (x - 1) ∧ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_two_solves_equation_l4011_401146


namespace NUMINAMATH_CALUDE_sprint_distance_l4011_401110

def sprint_problem (speed : ℝ) (time : ℝ) : Prop :=
  speed = 6 ∧ time = 4 → speed * time = 24

theorem sprint_distance : sprint_problem 6 4 := by
  sorry

end NUMINAMATH_CALUDE_sprint_distance_l4011_401110


namespace NUMINAMATH_CALUDE_care_package_weight_l4011_401187

theorem care_package_weight (initial_weight : ℝ) (brownies_factor : ℝ) (additional_jelly_beans : ℝ) (gummy_worms_factor : ℝ) :
  initial_weight = 2 ∧
  brownies_factor = 3 ∧
  additional_jelly_beans = 2 ∧
  gummy_worms_factor = 2 →
  (((initial_weight * brownies_factor + additional_jelly_beans) * gummy_worms_factor) : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_care_package_weight_l4011_401187


namespace NUMINAMATH_CALUDE_jordan_oreos_l4011_401113

theorem jordan_oreos (jordan : ℕ) (james : ℕ) : 
  james = 2 * jordan + 3 → 
  jordan + james = 36 → 
  jordan = 11 := by
sorry

end NUMINAMATH_CALUDE_jordan_oreos_l4011_401113


namespace NUMINAMATH_CALUDE_derivative_value_implies_coefficient_l4011_401191

theorem derivative_value_implies_coefficient (f' : ℝ → ℝ) (a : ℝ) :
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_derivative_value_implies_coefficient_l4011_401191


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l4011_401134

theorem sarahs_bowling_score (jessica greg sarah : ℕ) 
  (h1 : sarah = greg + 50)
  (h2 : greg = 2 * jessica)
  (h3 : (sarah + greg + jessica) / 3 = 110) :
  sarah = 162 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l4011_401134


namespace NUMINAMATH_CALUDE_food_additives_percentage_l4011_401189

/-- Represents the budget allocation for a research category -/
structure BudgetAllocation where
  percentage : ℝ
  degrees : ℝ

/-- Represents the total budget and its allocations -/
structure Budget where
  total_degrees : ℝ
  microphotonics : BudgetAllocation
  home_electronics : BudgetAllocation
  genetically_modified_microorganisms : BudgetAllocation
  industrial_lubricants : BudgetAllocation
  basic_astrophysics : BudgetAllocation
  food_additives : BudgetAllocation

/-- The Megatech Corporation's research and development budget -/
def megatech_budget : Budget := {
  total_degrees := 360
  microphotonics := { percentage := 14, degrees := 0 }
  home_electronics := { percentage := 24, degrees := 0 }
  genetically_modified_microorganisms := { percentage := 19, degrees := 0 }
  industrial_lubricants := { percentage := 8, degrees := 0 }
  basic_astrophysics := { percentage := 0, degrees := 72 }
  food_additives := { percentage := 0, degrees := 0 }
}

/-- Theorem: The percentage of the budget allocated to food additives is 15% -/
theorem food_additives_percentage : megatech_budget.food_additives.percentage = 15 := by
  sorry


end NUMINAMATH_CALUDE_food_additives_percentage_l4011_401189


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l4011_401111

theorem solve_cubic_equation (y : ℝ) :
  5 * y^(1/3) + 3 * (y / y^(2/3)) = 10 - y^(1/3) ↔ y = (10/9)^3 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l4011_401111


namespace NUMINAMATH_CALUDE_range_of_a_l4011_401177

/-- A function f is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The logarithm function with base a -/
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- Proposition p: log_a x is monotonically increasing for x > 0 -/
def p (a : ℝ) : Prop :=
  MonotonicallyIncreasing (fun x => log_base a x)

/-- Proposition q: x^2 + ax + 1 > 0 for all real x -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + 1 > 0

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → a ∈ Set.Ioc (-2) 1 ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4011_401177
