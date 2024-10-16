import Mathlib

namespace NUMINAMATH_CALUDE_no_rain_time_l1563_156359

theorem no_rain_time (total_time rain_time : ℕ) (h1 : total_time = 8) (h2 : rain_time = 2) :
  total_time - rain_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_time_l1563_156359


namespace NUMINAMATH_CALUDE_two_percent_as_decimal_l1563_156304

/-- Expresses a percentage as a decimal fraction -/
def percent_to_decimal (p : ℚ) : ℚ := p / 100

/-- Proves that 2% expressed as a decimal fraction is equal to 0.02 -/
theorem two_percent_as_decimal : percent_to_decimal 2 = 0.02 := by sorry

end NUMINAMATH_CALUDE_two_percent_as_decimal_l1563_156304


namespace NUMINAMATH_CALUDE_probability_of_2500_is_6_125_l1563_156367

/-- The number of outcomes on the wheel -/
def num_outcomes : ℕ := 5

/-- The number of spins -/
def num_spins : ℕ := 3

/-- The number of ways to achieve the desired sum -/
def num_successful_combinations : ℕ := 6

/-- The total number of possible outcomes after three spins -/
def total_possibilities : ℕ := num_outcomes ^ num_spins

/-- The probability of earning exactly $2500 in three spins -/
def probability_of_2500 : ℚ := num_successful_combinations / total_possibilities

theorem probability_of_2500_is_6_125 : probability_of_2500 = 6 / 125 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_2500_is_6_125_l1563_156367


namespace NUMINAMATH_CALUDE_general_term_formula_l1563_156357

-- Define the sequence
def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else if n = 2 then 8/3
  else if n = 3 then 15/4
  else if n = 4 then 24/5
  else if n = 5 then 35/6
  else if n = 6 then 48/7
  else (n^2 + 2*n) / (n + 1)

-- State the theorem
theorem general_term_formula (n : ℕ) (h : n > 0) :
  a n = (n^2 + 2*n) / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l1563_156357


namespace NUMINAMATH_CALUDE_probability_of_two_boys_l1563_156387

theorem probability_of_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = 12) 
  (h2 : boys = 8) 
  (h3 : girls = 4) 
  (h4 : total = boys + girls) : 
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2) = 14/33 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_two_boys_l1563_156387


namespace NUMINAMATH_CALUDE_solve_for_n_l1563_156345

/-- Given an equation x + 1315 + n - 1569 = 11901 where x = 88320,
    prove that the value of n is -75165. -/
theorem solve_for_n (x n : ℤ) (h1 : x + 1315 + n - 1569 = 11901) (h2 : x = 88320) :
  n = -75165 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l1563_156345


namespace NUMINAMATH_CALUDE_average_equation_solution_l1563_156393

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 5) + (8*x + 3) + (3*x + 8)) = 5*x - 10 → x = 23 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1563_156393


namespace NUMINAMATH_CALUDE_max_pages_for_25_dollars_l1563_156362

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The available amount in dollars -/
def available_dollars : ℕ := 25

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculate the maximum number of full pages that can be copied -/
def max_pages (dollars : ℕ) (cost : ℕ) : ℕ :=
  (dollars_to_cents dollars) / cost

/-- Theorem: Given $25 and a cost of 3 cents per page, the maximum number of full pages that can be copied is 833 -/
theorem max_pages_for_25_dollars :
  max_pages available_dollars cost_per_page = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_25_dollars_l1563_156362


namespace NUMINAMATH_CALUDE_community_average_age_l1563_156397

/-- Given a community with a ratio of women to men of 13:10, where the average age of women
    is 36 years and the average age of men is 31 years, prove that the average age of the
    community is 33 19/23 years. -/
theorem community_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 13 / 10)
  (h_women_age : avg_age_women = 36)
  (h_men_age : avg_age_men = 31) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 33 + 19 / 23 :=
by sorry

end NUMINAMATH_CALUDE_community_average_age_l1563_156397


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1563_156341

theorem sum_of_decimals : 
  (5.76 : ℝ) + (4.29 : ℝ) = (10.05 : ℝ) := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1563_156341


namespace NUMINAMATH_CALUDE_bacteria_growth_7_hours_l1563_156322

/-- Calculates the number of bacteria after a given number of hours, 
    given an initial count and doubling time. -/
def bacteria_growth (initial_count : ℕ) (hours : ℕ) : ℕ :=
  initial_count * 2^hours

/-- Theorem stating that after 7 hours, starting with 10 bacteria, 
    the population will be 1280 bacteria. -/
theorem bacteria_growth_7_hours : 
  bacteria_growth 10 7 = 1280 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_7_hours_l1563_156322


namespace NUMINAMATH_CALUDE_at_least_one_equals_a_l1563_156377

theorem at_least_one_equals_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end NUMINAMATH_CALUDE_at_least_one_equals_a_l1563_156377


namespace NUMINAMATH_CALUDE_angel_envelopes_l1563_156354

/-- The number of large envelopes Angel used --/
def large_envelopes : ℕ := 11

/-- The number of medium envelopes Angel used --/
def medium_envelopes : ℕ := 2 * large_envelopes

/-- The number of letters in small envelopes --/
def small_letters : ℕ := 20

/-- The number of letters in each medium envelope --/
def letters_per_medium : ℕ := 3

/-- The number of letters in each large envelope --/
def letters_per_large : ℕ := 5

/-- The total number of letters --/
def total_letters : ℕ := 150

theorem angel_envelopes :
  small_letters +
  medium_envelopes * letters_per_medium +
  large_envelopes * letters_per_large = total_letters :=
by sorry

end NUMINAMATH_CALUDE_angel_envelopes_l1563_156354


namespace NUMINAMATH_CALUDE_carpet_area_needed_l1563_156351

-- Define the room dimensions in feet
def room_length : ℝ := 18
def room_width : ℝ := 12

-- Define the conversion factor from feet to yards
def feet_per_yard : ℝ := 3

-- Define the area already covered in square yards
def area_covered : ℝ := 4

-- Theorem to prove
theorem carpet_area_needed : 
  let length_yards := room_length / feet_per_yard
  let width_yards := room_width / feet_per_yard
  let total_area := length_yards * width_yards
  total_area - area_covered = 20 := by sorry

end NUMINAMATH_CALUDE_carpet_area_needed_l1563_156351


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l1563_156355

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (4*x + 14) ∣ (6*x^3 + 19*x^2 + c*x + 70)) → c = 13 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l1563_156355


namespace NUMINAMATH_CALUDE_speed_calculation_l1563_156332

theorem speed_calculation (distance : ℝ) (early_time : ℝ) (speed_reduction : ℝ) : 
  distance = 40 ∧ early_time = 4/60 ∧ speed_reduction = 5 →
  ∃ (v : ℝ), v > 0 ∧ 
    (distance / v = distance / (v - speed_reduction) - early_time) ↔ 
    v = 60 := by sorry

end NUMINAMATH_CALUDE_speed_calculation_l1563_156332


namespace NUMINAMATH_CALUDE_toy_problem_solution_l1563_156364

/-- Represents the toy purchasing and pricing problem -/
structure ToyProblem where
  total_toys : ℕ
  total_cost : ℕ
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  original_price_A : ℕ
  original_daily_sales : ℕ
  sales_increase_rate : ℕ
  desired_daily_profit : ℕ

/-- The solution to the toy problem -/
structure ToySolution where
  num_A : ℕ
  num_B : ℕ
  new_price_A : ℕ

/-- Theorem stating the correct solution for the given problem -/
theorem toy_problem_solution (p : ToyProblem) 
  (h1 : p.total_toys = 50)
  (h2 : p.total_cost = 1320)
  (h3 : p.purchase_price_A = 28)
  (h4 : p.purchase_price_B = 24)
  (h5 : p.original_price_A = 40)
  (h6 : p.original_daily_sales = 8)
  (h7 : p.sales_increase_rate = 1)
  (h8 : p.desired_daily_profit = 96) :
  ∃ (s : ToySolution), 
    s.num_A = 30 ∧ 
    s.num_B = 20 ∧ 
    s.new_price_A = 36 ∧
    s.num_A + s.num_B = p.total_toys ∧
    s.num_A * p.purchase_price_A + s.num_B * p.purchase_price_B = p.total_cost ∧
    (s.new_price_A - p.purchase_price_A) * (p.original_daily_sales + (p.original_price_A - s.new_price_A) * p.sales_increase_rate) = p.desired_daily_profit :=
by
  sorry


end NUMINAMATH_CALUDE_toy_problem_solution_l1563_156364


namespace NUMINAMATH_CALUDE_bus_capacity_l1563_156328

theorem bus_capacity (left_seats : ℕ) (right_seats : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : 
  left_seats = 15 →
  right_seats = left_seats - 3 →
  people_per_seat = 3 →
  back_seat_capacity = 7 →
  (left_seats + right_seats) * people_per_seat + back_seat_capacity = 88 :=
by
  sorry

#check bus_capacity

end NUMINAMATH_CALUDE_bus_capacity_l1563_156328


namespace NUMINAMATH_CALUDE_subtraction_result_l1563_156343

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_valid_arrangement (a b c d e f g h i j : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ is_valid_digit e ∧
  is_valid_digit f ∧ is_valid_digit g ∧ is_valid_digit h ∧ is_valid_digit i ∧ is_valid_digit j ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem subtraction_result 
  (a b c d e f g h i j : ℕ) 
  (h1 : is_valid_arrangement a b c d e f g h i j)
  (h2 : a = 6)
  (h3 : b = 1) :
  61000 + c * 1000 + d * 100 + e * 10 + f - (g * 10000 + h * 1000 + i * 100 + j * 10 + a) = 59387 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l1563_156343


namespace NUMINAMATH_CALUDE_urn_gold_coin_percentage_l1563_156372

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  silverCoinPercentage : ℝ
  goldCoinPercentage : ℝ
  bronzeCoinPercentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def goldCoinPercentage (u : UrnComposition) : ℝ :=
  (1 - u.beadPercentage) * u.goldCoinPercentage

/-- The theorem states that given the specified urn composition,
    the percentage of gold coins in the urn is 35% --/
theorem urn_gold_coin_percentage :
  ∀ (u : UrnComposition),
    u.beadPercentage = 0.3 ∧
    u.silverCoinPercentage = 0.25 * (1 - u.beadPercentage) ∧
    u.goldCoinPercentage = 0.5 * (1 - u.beadPercentage) ∧
    u.bronzeCoinPercentage = (1 - u.beadPercentage) * (1 - 0.25 - 0.5) →
    goldCoinPercentage u = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_urn_gold_coin_percentage_l1563_156372


namespace NUMINAMATH_CALUDE_unique_modular_solution_l1563_156380

theorem unique_modular_solution : 
  ∀ n : ℤ, (10 ≤ n ∧ n ≤ 20) ∧ (n ≡ 7882 [ZMOD 7]) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l1563_156380


namespace NUMINAMATH_CALUDE_not_divisible_by_4_8_16_32_l1563_156321

def x : ℕ := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬(∃ k : ℕ, x = 4 * k) ∧ 
  ¬(∃ k : ℕ, x = 8 * k) ∧ 
  ¬(∃ k : ℕ, x = 16 * k) ∧ 
  ¬(∃ k : ℕ, x = 32 * k) :=
by sorry

end NUMINAMATH_CALUDE_not_divisible_by_4_8_16_32_l1563_156321


namespace NUMINAMATH_CALUDE_inequality_chain_l1563_156395

theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 : ℝ) / ((1 / a) + (1 / b)) < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l1563_156395


namespace NUMINAMATH_CALUDE_projection_matrix_values_l1563_156347

/-- A projection matrix Q satisfies Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The given matrix form -/
def projection_matrix (x y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![x, 12/25],
    ![y, 13/25]]

/-- Theorem stating that the projection matrix has x = 0 and y = 12/25 -/
theorem projection_matrix_values :
  ∀ x y : ℚ, is_projection_matrix (projection_matrix x y) → x = 0 ∧ y = 12/25 := by
  sorry


end NUMINAMATH_CALUDE_projection_matrix_values_l1563_156347


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l1563_156333

def R : ℝ := 10
def H : ℝ := 5

theorem cylinder_volume_increase (x : ℝ) : 
  π * (R + 2*x)^2 * H = π * R^2 * (H + 3*x) → x = 5 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l1563_156333


namespace NUMINAMATH_CALUDE_point_sum_on_reciprocal_function_l1563_156368

theorem point_sum_on_reciprocal_function (p q : ℝ → ℝ) (h1 : p 4 = 8) (h2 : ∀ x, q x = 1 / p x) :
  4 + q 4 = 33 / 8 := by
  sorry

end NUMINAMATH_CALUDE_point_sum_on_reciprocal_function_l1563_156368


namespace NUMINAMATH_CALUDE_arccos_of_neg_one_equals_pi_l1563_156394

theorem arccos_of_neg_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_of_neg_one_equals_pi_l1563_156394


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1563_156392

/-- The asymptote of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
    (|x + c| - |x - c|) / (2 * c) = 1/3) →
  (∃ (k : ℝ), k = 2 * Real.sqrt 2 ∧ 
    ∀ (x y : ℝ), y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1563_156392


namespace NUMINAMATH_CALUDE_matrix_equality_l1563_156396

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![2, 1; 4, 3]) : 
  B * A = !![2, 1; 4, 3] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l1563_156396


namespace NUMINAMATH_CALUDE_remaining_money_l1563_156391

def initial_amount : ℕ := 100
def roast_cost : ℕ := 17
def vegetable_cost : ℕ := 11

theorem remaining_money :
  initial_amount - (roast_cost + vegetable_cost) = 72 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l1563_156391


namespace NUMINAMATH_CALUDE_integer_average_l1563_156334

theorem integer_average (k m r s t : ℕ) : 
  0 < k ∧ k < m ∧ m < r ∧ r < s ∧ s < t ∧ 
  t = 40 ∧ 
  r ≤ 23 ∧ 
  ∀ (k' m' r' s' t' : ℕ), 
    (0 < k' ∧ k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < t' ∧ t' = 40) → r' ≤ r →
  (k + m + r + s + t) / 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_integer_average_l1563_156334


namespace NUMINAMATH_CALUDE_students_in_general_hall_l1563_156317

theorem students_in_general_hall (general : ℕ) (biology : ℕ) (math : ℕ) : 
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 →
  general = 30 := by
sorry

end NUMINAMATH_CALUDE_students_in_general_hall_l1563_156317


namespace NUMINAMATH_CALUDE_change_per_bill_l1563_156363

/-- Proves that the value of each bill given as change is $5 -/
theorem change_per_bill (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (num_bills : ℕ) :
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  num_bills = 2 →
  (payment - num_games * cost_per_game) / num_bills = 5 := by
  sorry

end NUMINAMATH_CALUDE_change_per_bill_l1563_156363


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1563_156300

theorem binomial_expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 3^2 = 54 → n = 4 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1563_156300


namespace NUMINAMATH_CALUDE_point_distance_from_y_axis_l1563_156346

theorem point_distance_from_y_axis (a : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (a - 3, 2 * a) ∧ |a - 3| = 2) → 
  (a = 5 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_point_distance_from_y_axis_l1563_156346


namespace NUMINAMATH_CALUDE_changfei_class_problem_l1563_156301

theorem changfei_class_problem (m n : ℕ+) 
  (h : m.val * (m.val - 1) + m.val * n.val + n.val = 51) : 
  m.val + n.val = 9 := by
sorry

end NUMINAMATH_CALUDE_changfei_class_problem_l1563_156301


namespace NUMINAMATH_CALUDE_min_journey_cost_l1563_156390

-- Define the cities and distances
def XY : ℝ := 3500
def XZ : ℝ := 4000

-- Define the cost functions
def train_cost (distance : ℝ) : ℝ := 0.20 * distance
def taxi_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the theorem
theorem min_journey_cost :
  let YZ : ℝ := Real.sqrt (XZ^2 - XY^2)
  let XY_cost : ℝ := min (train_cost XY) (taxi_cost XY)
  let YZ_cost : ℝ := min (train_cost YZ) (taxi_cost YZ)
  let ZX_cost : ℝ := min (train_cost XZ) (taxi_cost XZ)
  XY_cost + YZ_cost + ZX_cost = 1812.30 := by sorry

end NUMINAMATH_CALUDE_min_journey_cost_l1563_156390


namespace NUMINAMATH_CALUDE_lisa_photos_last_weekend_l1563_156305

/-- Calculates the number of photos Lisa took last weekend based on given conditions --/
def photos_last_weekend (animal_photos : ℕ) (flower_multiplier : ℕ) (scenery_difference : ℕ) (weekend_difference : ℕ) : ℕ :=
  let flower_photos := animal_photos * flower_multiplier
  let scenery_photos := flower_photos - scenery_difference
  let total_this_weekend := animal_photos + flower_photos + scenery_photos
  total_this_weekend - weekend_difference

/-- Theorem stating that Lisa took 45 photos last weekend given the conditions --/
theorem lisa_photos_last_weekend :
  photos_last_weekend 10 3 10 15 = 45 := by
  sorry

#eval photos_last_weekend 10 3 10 15

end NUMINAMATH_CALUDE_lisa_photos_last_weekend_l1563_156305


namespace NUMINAMATH_CALUDE_specific_garden_area_l1563_156303

/-- Represents a circular garden with a path through it. -/
structure GardenWithPath where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the remaining area of the garden not covered by the path. -/
def remainingArea (g : GardenWithPath) : ℝ :=
  sorry

/-- Theorem stating the remaining area for a specific garden configuration. -/
theorem specific_garden_area :
  let g : GardenWithPath := { diameter := 14, pathWidth := 4 }
  remainingArea g = 29 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_specific_garden_area_l1563_156303


namespace NUMINAMATH_CALUDE_complex_magnitude_for_specific_quadratic_l1563_156353

theorem complex_magnitude_for_specific_quadratic : 
  ∀ z : ℂ, z^2 - 6*z + 20 = 0 → Complex.abs z = Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_for_specific_quadratic_l1563_156353


namespace NUMINAMATH_CALUDE_second_catch_up_race_result_l1563_156382

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the state of the race -/
structure RaceState where
  runner1 : Runner
  runner2 : Runner
  laps_completed : ℝ

/-- The race setup with initial conditions -/
def initial_race : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 1 },
    laps_completed := 0.5 }

/-- The race state after the second runner doubles their speed -/
def race_after_speed_up (r : RaceState) : RaceState :=
  { runner1 := r.runner1,
    runner2 := { speed := 2 * r.runner2.speed },
    laps_completed := r.laps_completed }

/-- Theorem stating that the first runner will catch up again at 2.5 laps -/
theorem second_catch_up (r : RaceState) :
  let r' := race_after_speed_up r
  r'.runner1.speed > r'.runner2.speed →
  r.runner1.speed = 3 * r.runner2.speed →
  r.laps_completed = 0.5 →
  ∃ t : ℝ, t > 0 ∧ r'.runner1.speed * t = (2.5 - r.laps_completed + 1) * r'.runner2.speed * t :=
by
  sorry

/-- Main theorem combining all conditions and results -/
theorem race_result :
  let r := initial_race
  let r' := race_after_speed_up r
  r'.runner1.speed > r'.runner2.speed ∧
  r.runner1.speed = 3 * r.runner2.speed ∧
  r.laps_completed = 0.5 ∧
  (∃ t : ℝ, t > 0 ∧ r'.runner1.speed * t = (2.5 - r.laps_completed + 1) * r'.runner2.speed * t) :=
by
  sorry

end NUMINAMATH_CALUDE_second_catch_up_race_result_l1563_156382


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1563_156371

theorem min_value_of_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y + 2 * x + y = 4 → 
  x + y ≥ 2 * Real.sqrt 6 - 3 ∧ 
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 4 ∧ x + y = 2 * Real.sqrt 6 - 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1563_156371


namespace NUMINAMATH_CALUDE_existence_of_prime_q_l1563_156335

theorem existence_of_prime_q (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, n > 0 → ¬(q ∣ (n^p - p)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_prime_q_l1563_156335


namespace NUMINAMATH_CALUDE_oscar_coco_difference_l1563_156330

/-- The number of strides Coco takes between consecutive poles -/
def coco_strides : ℕ := 22

/-- The number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 6

/-- The number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 11

/-- The number of poles -/
def num_poles : ℕ := 31

/-- The total distance in feet between the first and last pole -/
def total_distance : ℕ := 7920

/-- The length of Coco's stride in feet -/
def coco_stride_length : ℚ := total_distance / (coco_strides * (num_poles - 1))

/-- The length of Oscar's leap in feet -/
def oscar_leap_length : ℚ := total_distance / (oscar_leaps * (num_poles - 1))

theorem oscar_coco_difference :
  oscar_leap_length - coco_stride_length = 32 := by sorry

end NUMINAMATH_CALUDE_oscar_coco_difference_l1563_156330


namespace NUMINAMATH_CALUDE_breath_holding_increase_l1563_156370

theorem breath_holding_increase (initial_time : ℝ) (final_time : ℝ) : 
  initial_time = 10 →
  final_time = 60 →
  let first_week := initial_time * 2
  let second_week := first_week * 2
  (final_time - second_week) / second_week * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_breath_holding_increase_l1563_156370


namespace NUMINAMATH_CALUDE_theresas_work_hours_l1563_156314

theorem theresas_work_hours : 
  let weekly_hours : List ℕ := [10, 13, 9, 14, 8, 0]
  let total_weeks : ℕ := 7
  let required_average : ℕ := 12
  let final_week_hours : ℕ := 30
  (List.sum weekly_hours + final_week_hours) / total_weeks = required_average :=
by
  sorry

end NUMINAMATH_CALUDE_theresas_work_hours_l1563_156314


namespace NUMINAMATH_CALUDE_polynomial_coefficient_e_l1563_156378

/-- Polynomial Q(x) = 3x^3 + dx^2 + ex + f -/
def Q (d e f : ℝ) (x : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem polynomial_coefficient_e (d e f : ℝ) :
  (Q d e f 0 = 9) →
  (3 + d + e + f = -(f / 3)) →
  (e = -15 - 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_e_l1563_156378


namespace NUMINAMATH_CALUDE_ferris_wheel_line_l1563_156329

theorem ferris_wheel_line (capacity : ℕ) (not_riding : ℕ) (total : ℕ) : 
  capacity = 56 → not_riding = 36 → total = capacity + not_riding → total = 92 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_line_l1563_156329


namespace NUMINAMATH_CALUDE_range_of_a_max_value_of_z_l1563_156340

-- Define the variables
variable (a b z : ℝ)

-- Define the conditions
def condition1 : Prop := 2 * a + b = 9
def condition2 : Prop := |9 - b| + |a| < 3
def condition3 : Prop := a > 0 ∧ b > 0
def condition4 : Prop := z = a^2 * b

-- Theorem for part (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : 
  -1 < a ∧ a < 1 := by sorry

-- Theorem for part (ii)
theorem max_value_of_z (h1 : condition1 a b) (h3 : condition3 a b) (h4 : condition4 a b z) : 
  z ≤ 27 := by sorry

end NUMINAMATH_CALUDE_range_of_a_max_value_of_z_l1563_156340


namespace NUMINAMATH_CALUDE_firewood_sacks_l1563_156320

theorem firewood_sacks (total_wood : ℕ) (wood_per_sack : ℕ) (h1 : total_wood = 80) (h2 : wood_per_sack = 20) :
  total_wood / wood_per_sack = 4 :=
by sorry

end NUMINAMATH_CALUDE_firewood_sacks_l1563_156320


namespace NUMINAMATH_CALUDE_toothpick_grid_30_15_l1563_156309

/-- Represents a rectangular grid made of toothpicks -/
structure ToothpickGrid where
  height : ℕ  -- Number of toothpicks in height
  width : ℕ   -- Number of toothpicks in width

/-- Calculates the total number of toothpicks in a grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  (grid.height + 1) * grid.width + (grid.width + 1) * grid.height

/-- Theorem: A 30x15 toothpick grid uses 945 toothpicks -/
theorem toothpick_grid_30_15 :
  totalToothpicks { height := 30, width := 15 } = 945 := by
  sorry


end NUMINAMATH_CALUDE_toothpick_grid_30_15_l1563_156309


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1563_156352

theorem range_of_a_for_always_positive_quadratic :
  {a : ℝ | ∀ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) > 0} = {a : ℝ | -1 < a ∧ a < 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1563_156352


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1563_156307

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 3 ∧ b > 3 ∧
    n = a + 3 ∧ n = 3 * b + 1 ∧
    (∀ (m : ℕ) (c d : ℕ), 
      c > 3 ∧ d > 3 ∧ m = c + 3 ∧ m = 3 * d + 1 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1563_156307


namespace NUMINAMATH_CALUDE_range_of_x_l1563_156349

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem range_of_x (h1 : ∀ x ∈ [-1, 1], Monotone f) 
  (h2 : ∀ x, f (x - 1) < f (1 - 3*x)) :
  ∃ S : Set ℝ, S = {x | 0 ≤ x ∧ x < 1/2} ∧ 
  (∀ x, x ∈ S ↔ (x - 1 ∈ [-1, 1] ∧ 1 - 3*x ∈ [-1, 1] ∧ f (x - 1) < f (1 - 3*x))) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1563_156349


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1563_156356

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = (3/2) * x ∨ y = -(3/2) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(3/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1563_156356


namespace NUMINAMATH_CALUDE_five_people_arrangement_l1563_156360

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n-1 people in a line -/
def arrangements_without_youngest (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of valid arrangements for n people where the youngest cannot be first or last -/
def validArrangements (n : ℕ) : ℕ :=
  totalArrangements n - 2 * arrangements_without_youngest n

theorem five_people_arrangement :
  validArrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_l1563_156360


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l1563_156338

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ k ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l1563_156338


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1563_156383

theorem polynomial_sum_of_coefficients 
  (f : ℂ → ℂ) 
  (a b c d : ℝ) :
  (∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d) →
  f (2*I) = 0 →
  f (2 + I) = 0 →
  a + b + c + d = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1563_156383


namespace NUMINAMATH_CALUDE_chairs_per_row_l1563_156369

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (h1 : total_chairs = 432) (h2 : num_rows = 27) :
  total_chairs / num_rows = 16 := by
  sorry

end NUMINAMATH_CALUDE_chairs_per_row_l1563_156369


namespace NUMINAMATH_CALUDE_johns_remaining_money_l1563_156358

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_cost := 4 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that John's remaining money is 50 - 15p dollars -/
theorem johns_remaining_money (p : ℝ) : money_left p = 50 - 15 * p := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l1563_156358


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l1563_156337

theorem smallest_integer_in_set (m : ℤ) : 
  (m + 3 < 3*m - 5) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l1563_156337


namespace NUMINAMATH_CALUDE_andy_restrung_seven_racquets_l1563_156374

/-- Calculates the number of racquets Andy restrung during his shift -/
def racquets_restrung (hourly_rate : ℤ) (restring_rate : ℤ) (grommet_rate : ℤ) (stencil_rate : ℤ)
                      (hours_worked : ℤ) (grommets_changed : ℤ) (stencils_painted : ℤ)
                      (total_earnings : ℤ) : ℤ :=
  let hourly_earnings := hourly_rate * hours_worked
  let grommet_earnings := grommet_rate * grommets_changed
  let stencil_earnings := stencil_rate * stencils_painted
  let restring_earnings := total_earnings - hourly_earnings - grommet_earnings - stencil_earnings
  restring_earnings / restring_rate

theorem andy_restrung_seven_racquets :
  racquets_restrung 9 15 10 1 8 2 5 202 = 7 := by
  sorry


end NUMINAMATH_CALUDE_andy_restrung_seven_racquets_l1563_156374


namespace NUMINAMATH_CALUDE_tank_capacity_l1563_156331

theorem tank_capacity : 
  ∀ (C : ℝ),
    (C / 6 + C / 12 = (2.5 * 60 + 1.5 * 60) * 8) →
    C = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1563_156331


namespace NUMINAMATH_CALUDE_combined_experience_is_68_l1563_156313

/-- Represents the years of experience for each person -/
structure Experience where
  james : ℕ
  john : ℕ
  mike : ℕ

/-- Calculates the combined experience of all three people -/
def combinedExperience (e : Experience) : ℕ :=
  e.james + e.john + e.mike

/-- Theorem stating the combined experience of James, John, and Mike -/
theorem combined_experience_is_68 (e : Experience) 
    (h1 : e.james = 20)
    (h2 : e.john = 2 * (e.james - 8) + 8)
    (h3 : e.mike = e.john - 16) : 
  combinedExperience e = 68 := by
  sorry

#check combined_experience_is_68

end NUMINAMATH_CALUDE_combined_experience_is_68_l1563_156313


namespace NUMINAMATH_CALUDE_range_of_a_l1563_156308

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1563_156308


namespace NUMINAMATH_CALUDE_remaining_amount_correct_l1563_156310

/-- Calculates the remaining amount in Will's original currency after shopping --/
def remaining_amount (initial_amount conversion_fee exchange_rate sweater_price tshirt_price
                      shoes_price hat_price socks_price shoe_refund_rate discount_rate
                      sales_tax_rate : ℚ) : ℚ :=
  let amount_after_fee := initial_amount - conversion_fee
  let local_currency_amount := amount_after_fee * exchange_rate
  let total_cost := sweater_price + tshirt_price + shoes_price + hat_price + socks_price
  let refund := shoes_price * shoe_refund_rate
  let cost_after_refund := total_cost - refund
  let discountable_items := sweater_price + tshirt_price + hat_price + socks_price
  let discount := discountable_items * discount_rate
  let cost_after_discount := cost_after_refund - discount
  let sales_tax := cost_after_discount * sales_tax_rate
  let final_cost := cost_after_discount + sales_tax
  let remaining_local := local_currency_amount - final_cost
  remaining_local / exchange_rate

/-- Theorem stating that the remaining amount is correct --/
theorem remaining_amount_correct :
  remaining_amount 74 2 (3/2) (27/2) (33/2) 45 (15/2) 6 (17/20) (1/10) (1/20) = (3987/100) := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_correct_l1563_156310


namespace NUMINAMATH_CALUDE_sandwich_fraction_l1563_156311

theorem sandwich_fraction (total : ℚ) (ticket : ℚ) (book : ℚ) (leftover : ℚ) 
  (h1 : total = 180)
  (h2 : ticket = 1/6)
  (h3 : book = 1/2)
  (h4 : leftover = 24) :
  ∃ (sandwich : ℚ), 
    sandwich * total + ticket * total + book * total = total - leftover ∧ 
    sandwich = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_fraction_l1563_156311


namespace NUMINAMATH_CALUDE_total_gift_wrapping_combinations_l1563_156399

/-- The number of different gift wrapping combinations -/
def gift_wrapping_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_card : ℕ) (gift_tag : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_card * gift_tag

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem total_gift_wrapping_combinations :
  gift_wrapping_combinations 10 5 6 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_gift_wrapping_combinations_l1563_156399


namespace NUMINAMATH_CALUDE_abc_inequality_l1563_156327

theorem abc_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1563_156327


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1563_156316

def z : ℂ := Complex.I * (-2 - Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1563_156316


namespace NUMINAMATH_CALUDE_tangent_ratio_problem_l1563_156365

theorem tangent_ratio_problem (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_problem_l1563_156365


namespace NUMINAMATH_CALUDE_triangle_properties_l1563_156385

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (t.c^2 + t.a*t.b = t.c*(t.a*Real.cos t.B - t.b*Real.cos t.A) + 2*t.b^2 → t.C = π/3) ∧
  (t.C = π/3 ∧ t.c = 2*Real.sqrt 3 → -2*Real.sqrt 3 < 4*Real.sin t.B - t.a ∧ 4*Real.sin t.B - t.a < 2*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1563_156385


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1563_156319

-- Define the length of the line of soldiers
def line_length : ℝ := 1

-- Define the distance each soldier marches
def soldier_distance : ℝ := 15

-- Define the speed ratio between the car and soldiers
def speed_ratio : ℝ := 2

-- Theorem statement
theorem car_distance_theorem :
  let car_distance := soldier_distance * speed_ratio * line_length
  car_distance = 30 := by sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1563_156319


namespace NUMINAMATH_CALUDE_lcm_36_98_l1563_156336

theorem lcm_36_98 : Nat.lcm 36 98 = 1764 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_98_l1563_156336


namespace NUMINAMATH_CALUDE_unique_pair_existence_l1563_156366

theorem unique_pair_existence :
  ∃! (c d : ℝ), 0 < c ∧ c < d ∧ d < π / 2 ∧
    Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_existence_l1563_156366


namespace NUMINAMATH_CALUDE_evaluate_expression_l1563_156381

theorem evaluate_expression : (16^24) / (64^8) = 16^8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1563_156381


namespace NUMINAMATH_CALUDE_discount_clinic_savings_l1563_156386

theorem discount_clinic_savings (normal_cost : ℚ) (discount_percentage : ℚ) (discount_visits : ℕ) : 
  normal_cost = 200 →
  discount_percentage = 70 →
  discount_visits = 2 →
  normal_cost - (discount_visits * (normal_cost * (1 - discount_percentage / 100))) = 80 := by
sorry

end NUMINAMATH_CALUDE_discount_clinic_savings_l1563_156386


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1563_156384

theorem geometric_sequence_seventh_term 
  (a₁ : ℝ) 
  (a₃ : ℝ) 
  (h₁ : a₁ = 3) 
  (h₃ : a₃ = 1/9) : 
  let r := (a₃ / a₁) ^ (1/2)
  a₁ * r^6 = Real.sqrt 3 / 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1563_156384


namespace NUMINAMATH_CALUDE_both_knaves_lied_yesterday_on_friday_l1563_156388

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the knaves
inductive Knave : Type
  | Hearts | Diamonds

-- Define the truth-telling function for each knave
def tells_truth (k : Knave) (d : Day) : Prop :=
  match k with
  | Knave.Hearts => d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Thursday
  | Knave.Diamonds => d = Day.Friday ∨ d = Day.Saturday ∨ d = Day.Sunday ∨ d = Day.Monday

-- Define the function to check if a knave lied yesterday
def lied_yesterday (k : Knave) (d : Day) : Prop :=
  ¬(tells_truth k (match d with
    | Day.Monday => Day.Sunday
    | Day.Tuesday => Day.Monday
    | Day.Wednesday => Day.Tuesday
    | Day.Thursday => Day.Wednesday
    | Day.Friday => Day.Thursday
    | Day.Saturday => Day.Friday
    | Day.Sunday => Day.Saturday))

-- Theorem: The only day when both knaves can truthfully say "Yesterday I told lies" is Friday
theorem both_knaves_lied_yesterday_on_friday :
  ∀ d : Day, (tells_truth Knave.Hearts d ∧ lied_yesterday Knave.Hearts d ∧
              tells_truth Knave.Diamonds d ∧ lied_yesterday Knave.Diamonds d) 
             ↔ d = Day.Friday :=
sorry

end NUMINAMATH_CALUDE_both_knaves_lied_yesterday_on_friday_l1563_156388


namespace NUMINAMATH_CALUDE_expression_factorization_l1563_156315

theorem expression_factorization (x y z : ℤ) :
  x^2 - (y + z)^2 + 2*x + y - z = (x - y - z) * (x + 2*y + 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1563_156315


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l1563_156302

theorem sum_of_fourth_powers_of_roots (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) → 
  (q^3 - 2*q^2 + 3*q - 4 = 0) → 
  (r^3 - 2*r^2 + 3*r - 4 = 0) → 
  p^4 + q^4 + r^4 = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l1563_156302


namespace NUMINAMATH_CALUDE_ascendant_functions_composition_inequality_l1563_156312

/-- A function is ascendant if it is monotonically increasing -/
def Ascendant (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem ascendant_functions_composition_inequality
  (f g φ : ℝ → ℝ)
  (hf : Ascendant f)
  (hg : Ascendant g)
  (hφ : Ascendant φ)
  (h : ∀ x, f x ≤ g x ∧ g x ≤ φ x) :
  ∀ x, f (f x) ≤ g (g x) ∧ g (g x) ≤ φ (φ x) := by
  sorry

end NUMINAMATH_CALUDE_ascendant_functions_composition_inequality_l1563_156312


namespace NUMINAMATH_CALUDE_equation_solution_l1563_156326

theorem equation_solution : ∃ x : ℝ, x + 1 - 2 * (x - 1) = 1 - 3 * x ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1563_156326


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1563_156325

theorem cubic_equation_roots : 
  {x : ℝ | x^9 + (9/8)*x^6 + (27/64)*x^3 - x + 219/512 = 0} = 
  {1/2, (-1 - Real.sqrt 13)/4, (-1 + Real.sqrt 13)/4} := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1563_156325


namespace NUMINAMATH_CALUDE_comparison_inequality_l1563_156376

theorem comparison_inequality : ∀ x : ℝ, (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequality_l1563_156376


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1563_156318

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1563_156318


namespace NUMINAMATH_CALUDE_sam_and_david_licks_l1563_156389

/-- The number of licks it takes for Dan to reach the center of a lollipop -/
def dan_licks : ℕ := 58

/-- The number of licks it takes for Michael to reach the center of a lollipop -/
def michael_licks : ℕ := 63

/-- The number of licks it takes for Lance to reach the center of a lollipop -/
def lance_licks : ℕ := 39

/-- The total number of people -/
def total_people : ℕ := 5

/-- The average number of licks for all people -/
def average_licks : ℕ := 60

/-- The theorem stating that Sam and David together take 140 licks to reach the center of a lollipop -/
theorem sam_and_david_licks : 
  total_people * average_licks - (dan_licks + michael_licks + lance_licks) = 140 := by
  sorry

end NUMINAMATH_CALUDE_sam_and_david_licks_l1563_156389


namespace NUMINAMATH_CALUDE_karen_start_time_l1563_156306

/-- Proves that Karen starts 4 minutes late in the car race --/
theorem karen_start_time (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_win_margin : ℝ) 
  (h1 : karen_speed = 60)
  (h2 : tom_speed = 45)
  (h3 : tom_distance = 24)
  (h4 : karen_win_margin = 4) :
  (tom_distance / tom_speed - (tom_distance + karen_win_margin) / karen_speed) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_karen_start_time_l1563_156306


namespace NUMINAMATH_CALUDE_distance_between_cities_l1563_156350

/-- The distance between two cities A and B, where two trains traveling towards each other meet. -/
theorem distance_between_cities (v1 v2 t1 t2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 75) (h3 : t1 = 4) (h4 : t2 = 3) : v1 * t1 + v2 * t2 = 465 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1563_156350


namespace NUMINAMATH_CALUDE_tinas_weekly_income_l1563_156375

/-- Calculates Tina's weekly income based on her work schedule and pay rates. -/
def calculate_weekly_income (hourly_wage : ℚ) (regular_hours : ℚ) (weekday_hours : ℚ) (weekend_hours : ℚ) : ℚ :=
  let overtime_rate := hourly_wage + hourly_wage / 2
  let double_overtime_rate := hourly_wage * 2
  let weekday_pay := (
    hourly_wage * regular_hours + 
    overtime_rate * (weekday_hours - regular_hours)
  ) * 5
  let weekend_pay := (
    hourly_wage * regular_hours + 
    overtime_rate * (regular_hours - regular_hours) +
    double_overtime_rate * (weekend_hours - regular_hours - (regular_hours - regular_hours))
  ) * 2
  weekday_pay + weekend_pay

/-- Theorem stating that Tina's weekly income is $1530.00 given her work schedule and pay rates. -/
theorem tinas_weekly_income :
  calculate_weekly_income 18 8 10 12 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tinas_weekly_income_l1563_156375


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1563_156323

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_complement_equality :
  N ∩ (Set.univ \ M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1563_156323


namespace NUMINAMATH_CALUDE_triangle_probability_l1563_156342

theorem triangle_probability (total_figures : ℕ) (triangle_count : ℕ) 
  (h1 : total_figures = 8) (h2 : triangle_count = 3) :
  (triangle_count : ℚ) / total_figures = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_probability_l1563_156342


namespace NUMINAMATH_CALUDE_petyas_sum_l1563_156344

theorem petyas_sum (n k : ℕ) : 
  (∀ i ∈ Finset.range (k + 1), Even (n + 2 * i)) →
  ((k + 1) * (n + k) = 30 * (n + 2 * k)) →
  ((k + 1) * (n + k) = 90 * n) →
  n = 44 ∧ k = 44 :=
by sorry

end NUMINAMATH_CALUDE_petyas_sum_l1563_156344


namespace NUMINAMATH_CALUDE_peters_pond_depth_l1563_156398

theorem peters_pond_depth :
  ∀ (mark_depth peter_depth : ℝ),
    mark_depth = 3 * peter_depth + 4 →
    mark_depth = 19 →
    peter_depth = 5 := by
  sorry

end NUMINAMATH_CALUDE_peters_pond_depth_l1563_156398


namespace NUMINAMATH_CALUDE_total_students_l1563_156361

theorem total_students (total : ℕ) 
  (h1 : (60 : ℝ) / 100 * total = (total : ℝ) - (40 : ℝ) / 100 * total)
  (h2 : (1 : ℝ) / 3 * ((40 : ℝ) / 100 * total) = (40 : ℝ) / 100 * total - 40) :
  total = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l1563_156361


namespace NUMINAMATH_CALUDE_max_discarded_grapes_l1563_156339

theorem max_discarded_grapes (n : ℕ) : ∃ (q : ℕ), n = 8 * q + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_discarded_grapes_l1563_156339


namespace NUMINAMATH_CALUDE_picnic_gender_difference_l1563_156324

theorem picnic_gender_difference (total : ℕ) (men : ℕ) (adult_child_diff : ℕ) 
  (h_total : total = 240)
  (h_men : men = 90)
  (h_adult_child : adult_child_diff = 40) : 
  ∃ (women children : ℕ), 
    men + women + children = total ∧ 
    men + women = children + adult_child_diff ∧ 
    men - women = 40 := by
sorry

end NUMINAMATH_CALUDE_picnic_gender_difference_l1563_156324


namespace NUMINAMATH_CALUDE_initial_sets_count_l1563_156373

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The length of each set of initials -/
def set_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through J -/
def num_initial_sets : ℕ := num_letters ^ set_length

theorem initial_sets_count : num_initial_sets = 10000 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l1563_156373


namespace NUMINAMATH_CALUDE_spinner_probability_l1563_156348

theorem spinner_probability : ∀ (p_A p_B p_C p_D p_E : ℝ),
  p_A = 1/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 3/20 :=
by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l1563_156348


namespace NUMINAMATH_CALUDE_four_stamps_cost_l1563_156379

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34/100

/-- The cost of n stamps in dollars -/
def n_stamps_cost (n : ℕ) : ℚ := n * stamp_cost

theorem four_stamps_cost :
  n_stamps_cost 4 = 136/100 :=
by sorry

end NUMINAMATH_CALUDE_four_stamps_cost_l1563_156379
