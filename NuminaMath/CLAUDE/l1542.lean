import Mathlib

namespace tromino_tiling_l1542_154279

/-- An L-shaped tromino covers exactly 3 unit squares. -/
def Tromino : ℕ := 3

/-- Represents whether an m×n grid can be tiled with L-shaped trominoes. -/
def can_tile (m n : ℕ) : Prop := 6 ∣ (m * n)

/-- 
Theorem: An m×n grid can be tiled with L-shaped trominoes if and only if 6 divides mn.
-/
theorem tromino_tiling (m n : ℕ) : can_tile m n ↔ 6 ∣ (m * n) := by sorry

end tromino_tiling_l1542_154279


namespace distance_BC_l1542_154284

/-- Represents a point on the route --/
structure Point :=
  (position : ℝ)

/-- Represents the route with points A, B, and C --/
structure Route :=
  (A B C : Point)
  (speed : ℝ)
  (time : ℝ)
  (AC_distance : ℝ)

/-- The theorem statement --/
theorem distance_BC (route : Route) : 
  route.A.position = 0 ∧ 
  route.speed = 50 ∧ 
  route.time = 20 ∧ 
  route.AC_distance = 600 →
  route.C.position - route.B.position = 400 :=
sorry

end distance_BC_l1542_154284


namespace min_value_of_c_l1542_154219

theorem min_value_of_c (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 →
  c = b + 1 →
  d = c + 1 →
  e = d + 1 →
  ∃ n : ℕ, a + b + c + d + e = n^3 →
  ∃ m : ℕ, b + c + d = m^2 →
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
    b' = a' + 1 ∧ c' = b' + 1 ∧ d' = c' + 1 ∧ e' = d' + 1 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3 ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2) →
  c' ≥ c →
  c = 675 :=
by sorry

end min_value_of_c_l1542_154219


namespace min_abs_ab_for_perpendicular_lines_l1542_154231

/-- Given two perpendicular lines l₁ and l₂, prove that the minimum value of |ab| is 2 -/
theorem min_abs_ab_for_perpendicular_lines (a b : ℝ) : 
  (∀ x y : ℝ, a^2 * x + y + 2 = 0 → b * x - (a^2 + 1) * y - 1 = 0 → 
   (a^2 * 1) * (b / (a^2 + 1)) = -1) →
  ∃ (min : ℝ), min = 2 ∧ ∀ a' b' : ℝ, 
    (∀ x y : ℝ, (a')^2 * x + y + 2 = 0 → b' * x - ((a')^2 + 1) * y - 1 = 0 → 
     ((a')^2 * 1) * (b' / ((a')^2 + 1)) = -1) →
    |a' * b'| ≥ min :=
by sorry

end min_abs_ab_for_perpendicular_lines_l1542_154231


namespace son_age_l1542_154230

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end son_age_l1542_154230


namespace sine_identity_and_not_monotonicity_l1542_154218

theorem sine_identity_and_not_monotonicity : 
  (∀ x : ℝ, Real.sin (π - x) = Real.sin x) ∧ 
  ¬(∀ α β : ℝ, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 → α > β → Real.sin α > Real.sin β) := by
  sorry

end sine_identity_and_not_monotonicity_l1542_154218


namespace sum_eight_fib_not_fib_l1542_154234

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the sum of eight consecutive Fibonacci numbers
def sum_eight_fib (k : ℕ) : ℕ :=
  (fib (k + 1)) + (fib (k + 2)) + (fib (k + 3)) + (fib (k + 4)) +
  (fib (k + 5)) + (fib (k + 6)) + (fib (k + 7)) + (fib (k + 8))

-- Theorem statement
theorem sum_eight_fib_not_fib (k : ℕ) :
  (sum_eight_fib k > fib (k + 9)) ∧ (sum_eight_fib k < fib (k + 10)) :=
by sorry

end sum_eight_fib_not_fib_l1542_154234


namespace dog_park_ratio_l1542_154277

theorem dog_park_ratio (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_ear_dogs : ℕ) :
  spotted_dogs = total_dogs / 2 →
  spotted_dogs = 15 →
  pointy_ear_dogs = 6 →
  (pointy_ear_dogs : ℚ) / total_dogs = 1 / 5 := by
sorry

end dog_park_ratio_l1542_154277


namespace no_solution_system_l1542_154254

/-- Proves that the system of equations 3x - 4y = 10 and 6x - 8y = 12 has no solution -/
theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 10) ∧ (6 * x - 8 * y = 12) := by
  sorry

end no_solution_system_l1542_154254


namespace zeros_not_adjacent_probability_l1542_154209

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with four ones in a row -/
theorem zeros_not_adjacent_probability :
  (Nat.choose (total_elements - 1) num_zeros) / (Nat.choose total_elements num_zeros) = 2 / 3 := by
  sorry

end zeros_not_adjacent_probability_l1542_154209


namespace bert_stamp_ratio_l1542_154280

def stamps_before (total_after purchase : ℕ) : ℕ := total_after - purchase

def ratio_simplify (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem bert_stamp_ratio : 
  let purchase := 300
  let total_after := 450
  let before := stamps_before total_after purchase
  ratio_simplify before purchase = (1, 2) := by
sorry

end bert_stamp_ratio_l1542_154280


namespace eighth_minus_seventh_difference_l1542_154286

/-- The number of tiles in the nth square of the sequence -/
def tiles (n : ℕ) : ℕ := n^2 + 2*n

/-- The difference in tiles between the 8th and 7th squares -/
def tile_difference : ℕ := tiles 8 - tiles 7

theorem eighth_minus_seventh_difference :
  tile_difference = 17 := by sorry

end eighth_minus_seventh_difference_l1542_154286


namespace prob_different_colors_is_148_225_l1542_154210

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def total_chips : ℕ := blue_chips + red_chips + yellow_chips

def prob_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem prob_different_colors_is_148_225 :
  prob_different_colors = 148 / 225 :=
sorry

end prob_different_colors_is_148_225_l1542_154210


namespace find_x_value_l1542_154255

theorem find_x_value (x y : ℝ) (h1 : (12 : ℝ)^3 * 6^2 / x = y) (h2 : y = 144) : x = 432 := by
  sorry

end find_x_value_l1542_154255


namespace max_value_of_fraction_sum_l1542_154261

theorem max_value_of_fraction_sum (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 2) :
  (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 1 ∧
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 2 ∧
    (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) = 1 :=
by sorry

end max_value_of_fraction_sum_l1542_154261


namespace coffee_mix_proof_l1542_154205

/-- The price of Colombian coffee beans in dollars per pound -/
def colombian_price : ℝ := 5.50

/-- The price of Peruvian coffee beans in dollars per pound -/
def peruvian_price : ℝ := 4.25

/-- The total weight of the mix in pounds -/
def total_weight : ℝ := 40

/-- The desired price of the mix in dollars per pound -/
def mix_price : ℝ := 4.60

/-- The amount of Colombian coffee beans in the mix -/
def colombian_amount : ℝ := 11.2

theorem coffee_mix_proof :
  colombian_amount * colombian_price + (total_weight - colombian_amount) * peruvian_price = 
  mix_price * total_weight :=
sorry

end coffee_mix_proof_l1542_154205


namespace cone_base_circumference_l1542_154257

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = (1/3) * π * r^2 * h →
  V = 27 * π →
  h = 9 →
  2 * π * r = 6 * π :=
by sorry

end cone_base_circumference_l1542_154257


namespace unit_digit_14_power_100_l1542_154212

theorem unit_digit_14_power_100 : (14^100) % 10 = 6 := by
  sorry

end unit_digit_14_power_100_l1542_154212


namespace square_area_from_diagonal_and_perimeter_l1542_154229

/-- Given a square with diagonal 2x and perimeter 16x, prove its area is 16x² -/
theorem square_area_from_diagonal_and_perimeter (x : ℝ) :
  let diagonal := 2 * x
  let perimeter := 16 * x
  let side := perimeter / 4
  let area := side ^ 2
  diagonal ^ 2 = 2 * side ^ 2 ∧ perimeter = 4 * side → area = 16 * x ^ 2 := by
  sorry

end square_area_from_diagonal_and_perimeter_l1542_154229


namespace special_rhombus_perimeter_l1542_154276

/-- A rhombus with integer side lengths where the area equals the perimeter -/
structure SpecialRhombus where
  side_length : ℕ
  area_eq_perimeter : (side_length ^ 2 * Real.sin (π / 6)) = (4 * side_length)

/-- The perimeter of a SpecialRhombus is 32 -/
theorem special_rhombus_perimeter (r : SpecialRhombus) : 4 * r.side_length = 32 := by
  sorry

#check special_rhombus_perimeter

end special_rhombus_perimeter_l1542_154276


namespace tournament_games_l1542_154296

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInTournament (n : ℕ) : ℕ := n - 1

/-- The number of teams in the tournament -/
def numTeams : ℕ := 20

theorem tournament_games :
  gamesInTournament numTeams = 19 := by sorry

end tournament_games_l1542_154296


namespace unique_solution_system_l1542_154297

/-- The system of equations has exactly one real solution -/
theorem unique_solution_system :
  ∃! (x y z : ℝ), x + y = 2 ∧ x * y - z^2 = 1 := by sorry

end unique_solution_system_l1542_154297


namespace product_of_positive_reals_l1542_154225

theorem product_of_positive_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 24 * (3 ^ (1/4)))
  (h2 : x * z = 42 * (3 ^ (1/4)))
  (h3 : y * z = 21 * (3 ^ (1/4))) :
  x * y * z = Real.sqrt 63504 := by
sorry

end product_of_positive_reals_l1542_154225


namespace slower_speed_fraction_l1542_154227

/-- Given that a person arrives at a bus stop 9 minutes later than normal when walking
    at a certain fraction of their usual speed, and it takes 36 minutes to walk to the
    bus stop at their usual speed, prove that the fraction of the usual speed they
    were walking at is 4/5. -/
theorem slower_speed_fraction (usual_time : ℕ) (delay : ℕ) (usual_time_eq : usual_time = 36) (delay_eq : delay = 9) :
  (usual_time : ℚ) / (usual_time + delay : ℚ) = 4 / 5 := by
  sorry

end slower_speed_fraction_l1542_154227


namespace least_number_divisible_l1542_154245

def numbers : List ℕ := [52, 84, 114, 133, 221, 379]

def result : ℕ := 1097897218492

theorem least_number_divisible (n : ℕ) : n = result ↔ 
  (∀ m ∈ numbers, (n + 20) % m = 0) ∧ 
  (∀ k < n, ∃ m ∈ numbers, (k + 20) % m ≠ 0) := by
  sorry

end least_number_divisible_l1542_154245


namespace quadratic_inequality_range_l1542_154207

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by sorry

end quadratic_inequality_range_l1542_154207


namespace tommy_initial_balloons_l1542_154236

/-- The number of balloons Tommy's mom gave him -/
def balloons_from_mom : ℕ := 34

/-- The total number of balloons Tommy had after receiving more from his mom -/
def total_balloons : ℕ := 60

/-- The number of balloons Tommy had to start with -/
def initial_balloons : ℕ := total_balloons - balloons_from_mom

theorem tommy_initial_balloons : initial_balloons = 26 := by
  sorry

end tommy_initial_balloons_l1542_154236


namespace blanch_dinner_slices_l1542_154299

/-- Calculates the number of pizza slices eaten for dinner given the initial number of slices and consumption throughout the day. -/
def pizza_slices_for_dinner (initial_slices breakfast_slices lunch_slices snack_slices remaining_slices : ℕ) : ℕ :=
  initial_slices - (breakfast_slices + lunch_slices + snack_slices + remaining_slices)

/-- Proves that Blanch ate 5 slices of pizza for dinner given the conditions of the problem. -/
theorem blanch_dinner_slices :
  pizza_slices_for_dinner 15 4 2 2 2 = 5 := by
  sorry

#eval pizza_slices_for_dinner 15 4 2 2 2

end blanch_dinner_slices_l1542_154299


namespace total_distance_is_12_17_l1542_154268

def walking_time : ℚ := 30 / 60
def walking_rate : ℚ := 3
def running_time : ℚ := 20 / 60
def running_rate : ℚ := 8
def cycling_time : ℚ := 40 / 60
def cycling_rate : ℚ := 12

def total_distance : ℚ :=
  walking_time * walking_rate +
  running_time * running_rate +
  cycling_time * cycling_rate

theorem total_distance_is_12_17 :
  total_distance = 12.17 := by sorry

end total_distance_is_12_17_l1542_154268


namespace similar_right_triangles_l1542_154214

theorem similar_right_triangles (y : ℝ) : 
  -- First triangle with legs 15 and 12
  let a₁ : ℝ := 15
  let b₁ : ℝ := 12
  -- Second triangle with legs y and 9
  let a₂ : ℝ := y
  let b₂ : ℝ := 9
  -- Triangles are similar (corresponding sides are proportional)
  a₁ / a₂ = b₁ / b₂ →
  -- The value of y is 11.25
  y = 11.25 := by
sorry

end similar_right_triangles_l1542_154214


namespace estimation_theorem_l1542_154221

-- Define a function to estimate multiplication
def estimate_mult (a b : ℕ) : ℕ :=
  let a' := (a + 5) / 10 * 10  -- Round to nearest ten
  a' * b

-- Define a function to estimate division
def estimate_div (a b : ℕ) : ℕ :=
  let a' := (a + 50) / 100 * 100  -- Round to nearest hundred
  a' / b

-- State the theorem
theorem estimation_theorem :
  estimate_mult 47 20 = 1000 ∧ estimate_div 744 6 = 120 := by
  sorry

end estimation_theorem_l1542_154221


namespace find_other_number_l1542_154258

theorem find_other_number (x y : ℤ) (h1 : 2*x + 3*y = 100) (h2 : x = 28 ∨ y = 28) : x = 8 ∨ y = 8 := by
  sorry

end find_other_number_l1542_154258


namespace rainfall_ratio_rainfall_ratio_is_three_to_two_l1542_154211

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    calculate the ratio of rainfall in the second week to the first week. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) :
  total = 25 →
  second_week = 15 →
  second_week / (total - second_week) = 3 / 2 := by
  sorry

/-- The ratio of rainfall in the second week to the first week is 3:2. -/
theorem rainfall_ratio_is_three_to_two :
  ∃ (total : ℝ) (second_week : ℝ),
    total = 25 ∧
    second_week = 15 ∧
    second_week / (total - second_week) = 3 / 2 := by
  sorry

end rainfall_ratio_rainfall_ratio_is_three_to_two_l1542_154211


namespace solution_set_inequality_l1542_154281

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | -1 < x < 2},
    prove that the solution set of a(x^2 + 1) + b(x - 1) + c > 2ax is {x | 0 < x < 3} -/
theorem solution_set_inequality (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x, a*(x^2 + 1) + b*(x - 1) + c > 2*a*x ↔ 0 < x ∧ x < 3) := by
  sorry

end solution_set_inequality_l1542_154281


namespace clock_notes_in_week_total_notes_in_week_l1542_154213

/-- Represents the ringing pattern for a single hour -/
structure HourPattern where
  quarter_past : Nat
  half_past : Nat
  quarter_to : Nat
  on_hour : Nat → Nat

/-- Represents the ringing pattern for a 12-hour period (day or night) -/
structure PeriodPattern where
  pattern : HourPattern
  on_hour_even : Nat → Nat
  on_hour_odd : Nat → Nat

def day_pattern : PeriodPattern :=
  { pattern := 
    { quarter_past := 2
      half_past := 4
      quarter_to := 6
      on_hour := λ h => 8
    }
    on_hour_even := λ h => h
    on_hour_odd := λ h => h / 2
  }

def night_pattern : PeriodPattern :=
  { pattern := 
    { quarter_past := 3
      half_past := 5
      quarter_to := 7
      on_hour := λ h => 9
    }
    on_hour_even := λ h => h / 2
    on_hour_odd := λ h => h
  }

def count_notes_for_period (pattern : PeriodPattern) : Nat :=
  12 * (pattern.pattern.quarter_past + pattern.pattern.half_past + pattern.pattern.quarter_to) +
  (pattern.pattern.on_hour 6 + pattern.on_hour_even 6 +
   pattern.pattern.on_hour 8 + pattern.on_hour_even 8 +
   pattern.pattern.on_hour 10 + pattern.on_hour_even 10 +
   pattern.pattern.on_hour 12 + pattern.on_hour_even 12 +
   pattern.pattern.on_hour 2 + pattern.on_hour_even 2 +
   pattern.pattern.on_hour 4 + pattern.on_hour_even 4 +
   pattern.pattern.on_hour 7 + pattern.on_hour_odd 7 +
   pattern.pattern.on_hour 9 + pattern.on_hour_odd 9 +
   pattern.pattern.on_hour 11 + pattern.on_hour_odd 11 +
   pattern.pattern.on_hour 1 + pattern.on_hour_odd 1 +
   pattern.pattern.on_hour 3 + pattern.on_hour_odd 3 +
   pattern.pattern.on_hour 5 + pattern.on_hour_odd 5)

theorem clock_notes_in_week :
  count_notes_for_period day_pattern + count_notes_for_period night_pattern = 471 ∧
  471 * 7 = 3297 := by sorry

theorem total_notes_in_week : (count_notes_for_period day_pattern + count_notes_for_period night_pattern) * 7 = 3297 := by sorry

end clock_notes_in_week_total_notes_in_week_l1542_154213


namespace smaug_hoard_theorem_l1542_154200

/-- Calculates the total value of Smaug's hoard in copper coins -/
def smaug_hoard_value : ℕ :=
  let gold_coins : ℕ := 100
  let silver_coins : ℕ := 60
  let copper_coins : ℕ := 33
  let silver_to_copper : ℕ := 8
  let gold_to_silver : ℕ := 3
  
  let gold_value : ℕ := gold_coins * gold_to_silver * silver_to_copper
  let silver_value : ℕ := silver_coins * silver_to_copper
  let total_value : ℕ := gold_value + silver_value + copper_coins
  
  total_value

theorem smaug_hoard_theorem : smaug_hoard_value = 2913 := by
  sorry

end smaug_hoard_theorem_l1542_154200


namespace fraction_simplification_l1542_154290

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) = (5 * Real.sqrt 3) / 36 := by
  sorry

end fraction_simplification_l1542_154290


namespace chris_birthday_savings_l1542_154272

/-- Chris's birthday savings problem -/
theorem chris_birthday_savings 
  (grandmother : ℕ) 
  (aunt_uncle : ℕ) 
  (parents : ℕ) 
  (total_now : ℕ) 
  (h1 : grandmother = 25)
  (h2 : aunt_uncle = 20)
  (h3 : parents = 75)
  (h4 : total_now = 279) :
  total_now - (grandmother + aunt_uncle + parents) = 159 := by
sorry

end chris_birthday_savings_l1542_154272


namespace neil_cookies_fraction_l1542_154228

theorem neil_cookies_fraction (total : ℕ) (remaining : ℕ) (h1 : total = 20) (h2 : remaining = 12) :
  (total - remaining : ℚ) / total = 2 / 5 := by
  sorry

end neil_cookies_fraction_l1542_154228


namespace arithmetic_sequence_problem_l1542_154260

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 3 + a 13 = 20) 
  (h3 : a 2 = -2) : 
  a 15 = 24 := by
  sorry

end arithmetic_sequence_problem_l1542_154260


namespace mrs_sheridans_cats_l1542_154233

theorem mrs_sheridans_cats (initial_cats additional_cats : ℕ) :
  initial_cats = 17 →
  additional_cats = 14 →
  initial_cats + additional_cats = 31 :=
by sorry

end mrs_sheridans_cats_l1542_154233


namespace equal_probability_l1542_154274

/-- The number of black gloves in the pocket -/
def black_gloves : ℕ := 15

/-- The number of white gloves in the pocket -/
def white_gloves : ℕ := 10

/-- The total number of gloves in the pocket -/
def total_gloves : ℕ := black_gloves + white_gloves

/-- The number of ways to choose 2 gloves from n gloves -/
def choose (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of drawing two gloves of the same color -/
def prob_same_color : ℚ :=
  (choose black_gloves + choose white_gloves) / choose total_gloves

/-- The probability of drawing two gloves of different colors -/
def prob_diff_color : ℚ :=
  (black_gloves * white_gloves) / choose total_gloves

theorem equal_probability : prob_same_color = prob_diff_color := by
  sorry

end equal_probability_l1542_154274


namespace valid_x_values_l1542_154251

-- Define the property for x
def is_valid_x (x : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    x ^ 2 = 2525000000 + a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f * 1 + 89 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10

-- State the theorem
theorem valid_x_values : 
  ∀ x : ℕ, is_valid_x x ↔ (x = 502567 ∨ x = 502583) :=
sorry

end valid_x_values_l1542_154251


namespace square_less_than_triple_l1542_154244

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end square_less_than_triple_l1542_154244


namespace arithmetic_operations_correctness_l1542_154264

theorem arithmetic_operations_correctness :
  ((-2 : ℤ) + 8 ≠ 10) ∧
  ((-1 : ℤ) - 3 = -4) ∧
  ((-2 : ℤ) * 2 ≠ 4) ∧
  ((-8 : ℚ) / (-1) ≠ -1/8) := by
  sorry

end arithmetic_operations_correctness_l1542_154264


namespace not_all_greater_than_one_l1542_154263

theorem not_all_greater_than_one (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬(a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) := by
  sorry

end not_all_greater_than_one_l1542_154263


namespace sum_seven_consecutive_integers_l1542_154241

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
sorry

end sum_seven_consecutive_integers_l1542_154241


namespace dale_toast_count_l1542_154262

/-- The cost of breakfast for Dale and Andrew -/
def breakfast_cost (toast_price egg_price : ℕ) (dale_toast : ℕ) : Prop :=
  toast_price * dale_toast + 2 * egg_price + toast_price + 2 * egg_price = 15

/-- Theorem stating that Dale had 2 slices of toast -/
theorem dale_toast_count : breakfast_cost 1 3 2 := by sorry

end dale_toast_count_l1542_154262


namespace middle_bead_value_is_92_l1542_154220

/-- Represents a string of beads with specific properties -/
structure BeadString where
  total_beads : Nat
  middle_bead_index : Nat
  price_diff_left : Nat
  price_diff_right : Nat
  total_value : Nat

/-- Calculates the value of the middle bead in a BeadString -/
def middle_bead_value (bs : BeadString) : Nat :=
  sorry

/-- Theorem stating the value of the middle bead in the specific BeadString -/
theorem middle_bead_value_is_92 :
  let bs : BeadString := {
    total_beads := 31,
    middle_bead_index := 15,
    price_diff_left := 3,
    price_diff_right := 4,
    total_value := 2012
  }
  middle_bead_value bs = 92 := by sorry

end middle_bead_value_is_92_l1542_154220


namespace cost_per_pound_mixed_feed_l1542_154266

/-- Calculates the cost per pound of mixed dog feed --/
theorem cost_per_pound_mixed_feed 
  (total_weight : ℝ) 
  (cheap_price : ℝ) 
  (expensive_price : ℝ) 
  (cheap_amount : ℝ) 
  (h1 : total_weight = 35) 
  (h2 : cheap_price = 0.18) 
  (h3 : expensive_price = 0.53) 
  (h4 : cheap_amount = 17) :
  (cheap_amount * cheap_price + (total_weight - cheap_amount) * expensive_price) / total_weight = 0.36 := by
sorry


end cost_per_pound_mixed_feed_l1542_154266


namespace rectangle_area_breadth_ratio_l1542_154250

/-- Proves that for a rectangle with breadth 10 meters and length 10 meters greater than its breadth,
    the ratio of its area to its breadth is 20:1. -/
theorem rectangle_area_breadth_ratio :
  ∀ (breadth length area : ℝ),
    breadth = 10 →
    length = breadth + 10 →
    area = length * breadth →
    area / breadth = 20 := by
  sorry

end rectangle_area_breadth_ratio_l1542_154250


namespace intersection_complement_equals_interval_l1542_154288

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

-- State the theorem
theorem intersection_complement_equals_interval :
  A ∩ (Set.univ \ B) = Set.Icc 0 1 := by sorry

end intersection_complement_equals_interval_l1542_154288


namespace cubic_quadratic_comparison_l1542_154201

theorem cubic_quadratic_comparison (n : ℝ) :
  (n > -1 → n^3 + 1 > n^2 + n) ∧ (n < -1 → n^3 + 1 < n^2 + n) := by
  sorry

end cubic_quadratic_comparison_l1542_154201


namespace difference_sum_of_powers_of_three_l1542_154217

def T : Finset ℕ := Finset.range 11

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (fun i => S.sum (fun j => if i > j then 3^i - 3^j else 0))

theorem difference_sum_of_powers_of_three :
  difference_sum T = 783492 := by
  sorry

end difference_sum_of_powers_of_three_l1542_154217


namespace ball_bounce_ratio_l1542_154206

theorem ball_bounce_ratio (h₀ : ℝ) (h₅ : ℝ) (r : ℝ) :
  h₀ = 96 →
  h₅ = 3 →
  h₅ = h₀ * r^5 →
  r = Real.sqrt 2 / 4 := by
sorry

end ball_bounce_ratio_l1542_154206


namespace cube_root_27_times_fourth_root_81_times_sqrt_9_l1542_154242

theorem cube_root_27_times_fourth_root_81_times_sqrt_9 :
  ∃ (a b c : ℝ), a^3 = 27 ∧ b^4 = 81 ∧ c^2 = 9 → a * b * c = 27 := by
  sorry

end cube_root_27_times_fourth_root_81_times_sqrt_9_l1542_154242


namespace a_5_equals_one_l1542_154223

/-- A geometric sequence with positive terms and common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem a_5_equals_one
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end a_5_equals_one_l1542_154223


namespace composition_is_rotation_l1542_154275

-- Define a rotation
def Rotation (center : Point) (angle : ℝ) : Point → Point :=
  sorry

-- Define the composition of two rotations
def ComposeRotations (A B : Point) (α β : ℝ) : Point → Point :=
  Rotation B β ∘ Rotation A α

-- Theorem statement
theorem composition_is_rotation (A B : Point) (α β : ℝ) 
  (h1 : A ≠ B) 
  (h2 : ¬ (∃ k : ℤ, α + β = 2 * π * k)) :
  ∃ (O : Point) (γ : ℝ), ComposeRotations A B α β = Rotation O γ ∧ γ = α + β :=
sorry

end composition_is_rotation_l1542_154275


namespace toy_cost_price_l1542_154246

theorem toy_cost_price (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) :
  total_selling_price = 18900 →
  num_toys_sold = 18 →
  num_toys_gain = 3 →
  ∃ (cost_price : ℕ),
    cost_price * num_toys_sold + cost_price * num_toys_gain = total_selling_price ∧
    cost_price = 900 := by
  sorry

end toy_cost_price_l1542_154246


namespace cos_equality_integer_l1542_154282

theorem cos_equality_integer (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.cos (↑n * π / 180) = Real.cos (430 * π / 180) →
  n = 70 ∨ n = -70 := by
sorry

end cos_equality_integer_l1542_154282


namespace equation_represents_ellipse_l1542_154295

/-- The equation x^2 + 2y^2 - 6x - 8y + 9 = 0 represents an ellipse -/
theorem equation_represents_ellipse :
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), x^2 + 2*y^2 - 6*x - 8*y + 9 = 0 ↔
      ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
by
  sorry


end equation_represents_ellipse_l1542_154295


namespace distance_between_centers_is_sqrt_5_l1542_154256

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle with sides 6, 8, and 10
def rightTriangle : Triangle := { a := 6, b := 8, c := 10 }

-- Define the distance between centers of inscribed and circumscribed circles
def distanceBetweenCenters (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem distance_between_centers_is_sqrt_5 :
  distanceBetweenCenters rightTriangle = Real.sqrt 5 := by sorry

end distance_between_centers_is_sqrt_5_l1542_154256


namespace fraction_sum_proof_l1542_154293

theorem fraction_sum_proof : 
  let a : ℚ := 12 / 15
  let b : ℚ := 7 / 9
  let c : ℚ := 1 + 1 / 6
  let sum : ℚ := a + b + c
  sum = 247 / 90 ∧ (∀ n d : ℕ, n ≠ 0 → d ≠ 0 → (n : ℚ) / d = sum → n ≥ 247 ∧ d ≥ 90) :=
by sorry

end fraction_sum_proof_l1542_154293


namespace car_speed_second_hour_car_speed_second_hour_value_l1542_154252

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 100)
  (h2 : average_speed = 90) : 
  (speed_first_hour + (2 * average_speed - speed_first_hour)) / 2 = average_speed := by
  sorry

/-- The speed of the car in the second hour is 80 km/h. -/
theorem car_speed_second_hour_value : 
  ∃ (speed_second_hour : ℝ), 
    speed_second_hour = 80 ∧ 
    (100 + speed_second_hour) / 2 = 90 := by
  sorry

end car_speed_second_hour_car_speed_second_hour_value_l1542_154252


namespace base5_123_to_base10_l1542_154283

/-- Converts a base-5 number represented as a list of digits to its base-10 equivalent -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Theorem: The base-10 representation of the base-5 number 123 is 38 -/
theorem base5_123_to_base10 :
  base5ToBase10 [3, 2, 1] = 38 := by
  sorry

#eval base5ToBase10 [3, 2, 1]

end base5_123_to_base10_l1542_154283


namespace special_function_at_three_l1542_154271

/-- An increasing function satisfying a specific functional equation -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (f x - 2^x) = 3)

/-- The value of the special function at 3 is 9 -/
theorem special_function_at_three 
  (f : ℝ → ℝ) (hf : SpecialFunction f) : f 3 = 9 := by
  sorry

end special_function_at_three_l1542_154271


namespace book_club_unique_books_l1542_154203

theorem book_club_unique_books (tony dean breanna piper asher : ℕ)
  (tony_dean breanna_piper_asher dean_piper_tony asher_breanna_tony all_five : ℕ)
  (h_tony : tony = 23)
  (h_dean : dean = 20)
  (h_breanna : breanna = 30)
  (h_piper : piper = 26)
  (h_asher : asher = 25)
  (h_tony_dean : tony_dean = 5)
  (h_breanna_piper_asher : breanna_piper_asher = 6)
  (h_dean_piper_tony : dean_piper_tony = 4)
  (h_asher_breanna_tony : asher_breanna_tony = 3)
  (h_all_five : all_five = 2) :
  tony + dean + breanna + piper + asher -
  ((tony_dean - all_five) + (breanna_piper_asher - all_five) +
   (dean_piper_tony - all_five) + (asher_breanna_tony - all_five) + all_five) = 112 :=
by sorry

end book_club_unique_books_l1542_154203


namespace min_cos_for_valid_sqrt_l1542_154292

theorem min_cos_for_valid_sqrt (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 * Real.cos x - 1)) ↔ Real.cos x ≥ (1/2 : ℝ) :=
sorry

end min_cos_for_valid_sqrt_l1542_154292


namespace classroom_ratio_l1542_154232

theorem classroom_ratio :
  ∀ (boys girls : ℕ),
  boys + girls = 36 →
  boys = girls + 6 →
  (boys : ℚ) / girls = 7 / 5 :=
by
  sorry

end classroom_ratio_l1542_154232


namespace A_has_min_l1542_154270

/-- The function f_{a,b} from R^2 to R^2 -/
def f (a b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (a - b * y - x^2, x)

/-- The n-th iteration of f_{a,b} -/
def f_iter (a b : ℝ) : ℕ → (ℝ × ℝ → ℝ × ℝ)
  | 0 => id
  | n + 1 => f a b ∘ f_iter a b n

/-- The set of periodic points of f_{a,b} -/
def per (a b : ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ n : ℕ+, f_iter a b n P = P}

/-- The set A_b -/
def A (b : ℝ) : Set ℝ :=
  {a | per a b ≠ ∅}

/-- The theorem stating that A_b has a minimum equal to -(b+1)^2/4 -/
theorem A_has_min (b : ℝ) : 
  ∃ min : ℝ, IsGLB (A b) min ∧ min = -(b + 1)^2 / 4 := by
  sorry

end A_has_min_l1542_154270


namespace triangle_side_length_l1542_154289

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  a = Real.sqrt 31 →
  b = 6 →
  (c = 1 ∨ c = 5) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) :=
by sorry

end triangle_side_length_l1542_154289


namespace another_two_digit_prime_digit_number_l1542_154226

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem another_two_digit_prime_digit_number : 
  ∃ n : ℕ, is_two_digit n ∧ 
           is_prime (n / 10) ∧ 
           is_prime (n % 10) ∧ 
           n ≠ 23 :=
sorry

end another_two_digit_prime_digit_number_l1542_154226


namespace distance_between_locations_l1542_154224

/-- The distance between two locations A and B given two cars meeting conditions --/
theorem distance_between_locations (speed_B : ℝ) (h1 : speed_B > 0) : 
  let speed_A := 1.2 * speed_B
  let midpoint_to_meeting := 8
  let time := 2 * midpoint_to_meeting / (speed_A - speed_B)
  (speed_A + speed_B) * time = 176 := by
  sorry

end distance_between_locations_l1542_154224


namespace quadratic_equation_solution_l1542_154215

theorem quadratic_equation_solution :
  let x₁ : ℝ := (1 + Real.sqrt 3) / 2
  let x₂ : ℝ := (1 - Real.sqrt 3) / 2
  2 * x₁^2 - 2 * x₁ - 1 = 0 ∧ 2 * x₂^2 - 2 * x₂ - 1 = 0 := by
  sorry

end quadratic_equation_solution_l1542_154215


namespace factorization_a5_minus_a3b2_l1542_154273

theorem factorization_a5_minus_a3b2 (a b : ℝ) : 
  a^5 - a^3 * b^2 = a^3 * (a + b) * (a - b) := by sorry

end factorization_a5_minus_a3b2_l1542_154273


namespace proposition_variants_l1542_154253

theorem proposition_variants (a b : ℝ) :
  (((a - 2 > b - 2) → (a > b)) ∧
   ((a ≤ b) → (a - 2 ≤ b - 2)) ∧
   ((a - 2 ≤ b - 2) → (a ≤ b)) ∧
   ¬((a > b) → (a - 2 ≤ b - 2))) := by
  sorry

end proposition_variants_l1542_154253


namespace celine_change_l1542_154269

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine buys -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine buys -/
def smartphones_bought : ℕ := 4

/-- The amount of money Celine has in dollars -/
def money_available : ℕ := 3000

/-- The change Celine receives after her purchase -/
def change : ℕ := money_available - (laptop_price * laptops_bought + smartphone_price * smartphones_bought)

theorem celine_change : change = 200 := by
  sorry

end celine_change_l1542_154269


namespace exam_marks_problem_l1542_154237

/-- Examination marks problem -/
theorem exam_marks_problem (full_marks : ℕ) (a_marks b_marks c_marks d_marks : ℕ) :
  full_marks = 500 →
  a_marks = (9 : ℕ) * b_marks / 10 →
  c_marks = (4 : ℕ) * d_marks / 5 →
  a_marks = 360 →
  d_marks = (4 : ℕ) * full_marks / 5 →
  b_marks - c_marks = c_marks / 4 :=
by sorry

end exam_marks_problem_l1542_154237


namespace chess_tournament_games_l1542_154294

theorem chess_tournament_games (n : ℕ) (h : n = 50) : 
  (n * (n - 1)) / 2 = 1225 :=
sorry

end chess_tournament_games_l1542_154294


namespace quadratic_one_solution_l1542_154278

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ n = 8 := by sorry

end quadratic_one_solution_l1542_154278


namespace train_length_calculation_l1542_154247

/-- Calculates the length of a train given its speed, the time it takes to pass a platform, and the length of the platform. -/
theorem train_length_calculation (train_speed : Real) (platform_pass_time : Real) (platform_length : Real) :
  train_speed = 60 →
  platform_pass_time = 23.998080153587715 →
  platform_length = 260 →
  let train_speed_mps := train_speed * 1000 / 3600
  let total_distance := train_speed_mps * platform_pass_time
  let train_length := total_distance - platform_length
  train_length = 139.968003071754 := by
  sorry

end train_length_calculation_l1542_154247


namespace mary_needs_four_cups_l1542_154291

/-- The number of cups of flour Mary needs to add to her cake -/
def additional_flour (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

/-- Proof that Mary needs to add 4 more cups of flour -/
theorem mary_needs_four_cups : additional_flour 10 6 = 4 := by
  sorry

end mary_needs_four_cups_l1542_154291


namespace letter_writing_is_permutation_problem_l1542_154243

/-- A function that represents the number of letters written when n people write to each other once -/
def letters_written (n : ℕ) : ℕ := n * (n - 1)

/-- A function that represents whether a scenario is a permutation problem -/
def is_permutation_problem (scenario : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, n > 1 ∧ scenario n ≠ scenario (n - 1)

theorem letter_writing_is_permutation_problem :
  is_permutation_problem letters_written :=
sorry


end letter_writing_is_permutation_problem_l1542_154243


namespace equation_solver_l1542_154208

theorem equation_solver (a b x y : ℝ) (h1 : x^2 / y + y^2 / x = a) (h2 : x / y + y / x = b) :
  (x = (a * (b + 2 + Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2)) ∧
   y = (a * (b + 2 - Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2))) ∨
  (x = (a * (b + 2 - Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2)) ∧
   y = (a * (b + 2 + Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2))) :=
by sorry

end equation_solver_l1542_154208


namespace smallest_invertible_domain_l1542_154238

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the property of being invertible on [c, ∞)
def is_invertible_on_range (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y, x ≥ c → y ≥ c → f x = f y → x = y

-- Theorem statement
theorem smallest_invertible_domain : 
  (∀ c < 3, ¬(is_invertible_on_range f c)) ∧ 
  (is_invertible_on_range f 3) :=
sorry

end smallest_invertible_domain_l1542_154238


namespace percentage_not_sold_approx_l1542_154216

def initial_stock : ℕ := 1100
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_not_sold_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_not_sold - 63.45| < ε :=
sorry

end percentage_not_sold_approx_l1542_154216


namespace gcf_90_108_l1542_154298

theorem gcf_90_108 : Nat.gcd 90 108 = 18 := by
  sorry

end gcf_90_108_l1542_154298


namespace total_pawns_left_l1542_154239

/-- The number of pawns each player starts with in a chess game -/
def initial_pawns : ℕ := 8

/-- The number of pawns Kennedy has lost -/
def kennedy_lost : ℕ := 4

/-- The number of pawns Riley has lost -/
def riley_lost : ℕ := 1

/-- Theorem: The total number of pawns left in the game is 11 -/
theorem total_pawns_left : 
  (initial_pawns - kennedy_lost) + (initial_pawns - riley_lost) = 11 := by
  sorry

end total_pawns_left_l1542_154239


namespace max_profit_l1542_154259

def factory_price_A : ℕ := 10
def factory_price_B : ℕ := 18
def selling_price_A : ℕ := 12
def selling_price_B : ℕ := 22
def total_vehicles : ℕ := 130

def profit_function (x : ℕ) : ℤ :=
  -2 * x + 520

def is_valid_purchase (x : ℕ) : Prop :=
  x ≤ total_vehicles ∧ (total_vehicles - x) ≤ 2 * x

theorem max_profit :
  ∃ (x : ℕ), is_valid_purchase x ∧
    ∀ (y : ℕ), is_valid_purchase y → profit_function x ≥ profit_function y ∧
    profit_function x = 432 :=
  sorry

end max_profit_l1542_154259


namespace total_raisins_l1542_154248

theorem total_raisins (yellow_raisins black_raisins : ℝ) 
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4) :
  yellow_raisins + black_raisins = 0.7 := by
  sorry

end total_raisins_l1542_154248


namespace tom_reading_speed_l1542_154249

/-- Given that Tom reads 10 hours over 5 days, reads the same amount every day,
    and reads 700 pages in 7 days, prove that he can read 50 pages per hour. -/
theorem tom_reading_speed :
  ∀ (total_hours : ℕ) (days : ℕ) (total_pages : ℕ) (week_days : ℕ),
    total_hours = 10 →
    days = 5 →
    total_pages = 700 →
    week_days = 7 →
    (total_hours / days) * week_days ≠ 0 →
    total_pages / ((total_hours / days) * week_days) = 50 := by
  sorry

end tom_reading_speed_l1542_154249


namespace min_value_theorem_l1542_154235

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9*x + 1/x^6 ≥ 10 ∧ (9*x + 1/x^6 = 10 ↔ x = 1) := by
  sorry

end min_value_theorem_l1542_154235


namespace existence_of_special_set_l1542_154204

theorem existence_of_special_set (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) := by
  sorry

end existence_of_special_set_l1542_154204


namespace bookstore_shipment_l1542_154267

theorem bookstore_shipment (displayed_percentage : ℚ) (storeroom_books : ℕ) : 
  displayed_percentage = 30 / 100 →
  storeroom_books = 210 →
  ∃ total_books : ℕ, 
    (1 - displayed_percentage) * total_books = storeroom_books ∧
    total_books = 300 :=
by sorry

end bookstore_shipment_l1542_154267


namespace circle_C_properties_l1542_154285

-- Define the circles and points
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2
def point_P : ℝ × ℝ := (1, 1)
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the vector dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the theorem
theorem circle_C_properties
  (r : ℝ)
  (h_r : r > 0)
  (h_symmetry : ∀ x y, circle_C x y ↔ 
    ∃ x' y', circle_M r x' y' ∧ symmetry_line ((x + x')/2) ((y + y')/2))
  (h_P_on_C : circle_C point_P.1 point_P.2)
  (h_complementary_slopes : ∀ A B : ℝ × ℝ, 
    circle_C A.1 A.2 → circle_C B.1 B.2 → 
    (A.2 - point_P.2) * (B.2 - point_P.2) = -(A.1 - point_P.1) * (B.1 - point_P.1)) :
  (∀ x y, circle_C x y ↔ x^2 + y^2 = 2) ∧
  (∀ Q : ℝ × ℝ, point_Q Q.1 Q.2 → 
    dot_product (Q.1 - point_P.1, Q.2 - point_P.2) (Q.1 + 2, Q.2 + 2) ≥ -4) ∧
  (∀ A B : ℝ × ℝ, circle_C A.1 A.2 → circle_C B.1 B.2 → A ≠ B →
    (A.2 - B.2) * point_P.1 = (A.1 - B.1) * point_P.2) :=
sorry

end circle_C_properties_l1542_154285


namespace number_of_boys_l1542_154265

/-- The number of boys in a school, given the number of girls and the difference between boys and girls. -/
theorem number_of_boys (girls : ℕ) (difference : ℕ) : girls = 1225 → difference = 1750 → girls + difference = 2975 := by
  sorry

end number_of_boys_l1542_154265


namespace min_value_of_2x_plus_y_l1542_154222

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2 / a + 1 / b = 1 → 2 * x + y ≤ 2 * a + b ∧ 2 * x + y = 9 :=
by sorry

end min_value_of_2x_plus_y_l1542_154222


namespace polar_coordinate_equivalence_l1542_154240

/-- Given a point in polar coordinates (-5, 5π/6), prove that it is equivalent to (5, 11π/6) in standard polar coordinate representation. -/
theorem polar_coordinate_equivalence :
  let given_point : ℝ × ℝ := (-5, 5 * Real.pi / 6)
  let standard_point : ℝ × ℝ := (5, 11 * Real.pi / 6)
  (∀ (r θ : ℝ), r > 0 → 0 ≤ θ → θ < 2 * Real.pi →
    (r * (Real.cos θ), r * (Real.sin θ)) =
    (given_point.1 * (Real.cos given_point.2), given_point.1 * (Real.sin given_point.2))) →
  (standard_point.1 * (Real.cos standard_point.2), standard_point.1 * (Real.sin standard_point.2)) =
  (given_point.1 * (Real.cos given_point.2), given_point.1 * (Real.sin given_point.2)) :=
by sorry


end polar_coordinate_equivalence_l1542_154240


namespace third_group_men_count_l1542_154202

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℕ := sorry

theorem third_group_men_count : x = 5 := by
  have h1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate := by sorry
  have h2 : x * man_rate + 2 * woman_rate = (6/7) * (3 * man_rate + 8 * woman_rate) := by sorry
  sorry

end third_group_men_count_l1542_154202


namespace wire_ratio_proof_l1542_154287

/-- Given a wire of length 70 cm cut into two pieces, where the shorter piece is 27.999999999999993 cm long,
    prove that the ratio of the shorter piece to the longer piece is 2:3. -/
theorem wire_ratio_proof (total_length : ℝ) (shorter_piece : ℝ) (longer_piece : ℝ) :
  total_length = 70 →
  shorter_piece = 27.999999999999993 →
  longer_piece = total_length - shorter_piece →
  (shorter_piece / longer_piece) = 2 / 3 := by
  sorry

end wire_ratio_proof_l1542_154287
