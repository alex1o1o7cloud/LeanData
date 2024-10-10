import Mathlib

namespace division_problem_l238_23821

theorem division_problem (n : ℕ) : 
  n % 23 = 19 ∧ n / 23 = 17 → (10 * n) / 23 + (10 * n) % 23 = 184 :=
by
  sorry

end division_problem_l238_23821


namespace unique_digit_arrangement_l238_23866

-- Define the type for positions
inductive Position : Type
  | A | B | C | D | E | F

-- Define the function type for digit assignments
def DigitAssignment := Position → Fin 6

-- Define the condition that all digits are used exactly once
def allDigitsUsedOnce (assignment : DigitAssignment) : Prop :=
  ∀ d : Fin 6, ∃! p : Position, assignment p = d

-- Define the sum conditions
def sumConditions (assignment : DigitAssignment) : Prop :=
  (assignment Position.A).val + (assignment Position.D).val + (assignment Position.E).val = 15 ∧
  7 + (assignment Position.C).val + (assignment Position.E).val = 15 ∧
  9 + (assignment Position.C).val + (assignment Position.A).val = 15 ∧
  (assignment Position.A).val + 8 + (assignment Position.F).val = 15 ∧
  7 + (assignment Position.D).val + (assignment Position.F).val = 15 ∧
  9 + (assignment Position.D).val + (assignment Position.B).val = 15 ∧
  (assignment Position.B).val + (assignment Position.C).val + (assignment Position.F).val = 15

-- Define the correct assignment
def correctAssignment : DigitAssignment :=
  λ p => match p with
  | Position.A => 3  -- 4 - 1 (Fin 6 is 0-based)
  | Position.B => 0  -- 1 - 1
  | Position.C => 1  -- 2 - 1
  | Position.D => 4  -- 5 - 1
  | Position.E => 5  -- 6 - 1
  | Position.F => 2  -- 3 - 1

-- Theorem statement
theorem unique_digit_arrangement :
  ∀ assignment : DigitAssignment,
    allDigitsUsedOnce assignment ∧ sumConditions assignment →
    assignment = correctAssignment :=
sorry

end unique_digit_arrangement_l238_23866


namespace nested_sqrt_value_l238_23882

theorem nested_sqrt_value : ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (2 - x) → x = 1 := by
  sorry

end nested_sqrt_value_l238_23882


namespace money_puzzle_l238_23818

theorem money_puzzle (x : ℝ) : x = 800 ↔ 4 * x - 2000 = 2000 - x := by sorry

end money_puzzle_l238_23818


namespace conference_handshakes_theorem_l238_23823

/-- Represents the number of handshakes in a conference with specific group interactions -/
def conference_handshakes (total : ℕ) (group_a : ℕ) (group_b : ℕ) (group_c : ℕ) 
  (known_per_b : ℕ) : ℕ :=
  let handshakes_ab := group_b * (group_a - known_per_b)
  let handshakes_bc := group_b * group_c
  let handshakes_c := group_c * (group_c - 1) / 2
  let handshakes_ac := group_a * group_c
  handshakes_ab + handshakes_bc + handshakes_c + handshakes_ac

/-- Theorem stating that the number of handshakes in the given conference scenario is 535 -/
theorem conference_handshakes_theorem :
  conference_handshakes 50 30 15 5 10 = 535 := by
  sorry

end conference_handshakes_theorem_l238_23823


namespace perimeter_is_twenty_l238_23848

/-- The perimeter of a six-sided figure with specified side lengths -/
def perimeter_of_figure (h1 h2 v1 v2 v3 v4 : ℕ) : ℕ :=
  h1 + h2 + v1 + v2 + v3 + v4

/-- Theorem: The perimeter of the given figure is 20 units -/
theorem perimeter_is_twenty :
  ∃ (h1 h2 v1 v2 v3 v4 : ℕ),
    h1 + h2 = 5 ∧
    v1 = 2 ∧ v2 = 3 ∧ v3 = 3 ∧ v4 = 2 ∧
    perimeter_of_figure h1 h2 v1 v2 v3 v4 = 20 :=
by
  sorry


end perimeter_is_twenty_l238_23848


namespace magazine_page_height_l238_23877

/-- Given advertising costs and dimensions, calculate the height of a magazine page -/
theorem magazine_page_height 
  (cost_per_sq_inch : ℝ) 
  (ad_fraction : ℝ) 
  (page_width : ℝ) 
  (total_cost : ℝ) 
  (h : cost_per_sq_inch = 8)
  (h₁ : ad_fraction = 1/2)
  (h₂ : page_width = 12)
  (h₃ : total_cost = 432) :
  ∃ (page_height : ℝ), 
    page_height = 9 ∧ 
    ad_fraction * page_height * page_width * cost_per_sq_inch = total_cost := by
  sorry

end magazine_page_height_l238_23877


namespace g_approaches_neg_inf_pos_g_approaches_neg_inf_neg_l238_23889

/-- The function g(x) = -3x^4 + 50x^2 - 1 -/
def g (x : ℝ) : ℝ := -3 * x^4 + 50 * x^2 - 1

/-- Theorem stating that g(x) approaches negative infinity as x approaches positive infinity -/
theorem g_approaches_neg_inf_pos (ε : ℝ) : ∃ M : ℝ, ∀ x : ℝ, x > M → g x < ε :=
sorry

/-- Theorem stating that g(x) approaches negative infinity as x approaches negative infinity -/
theorem g_approaches_neg_inf_neg (ε : ℝ) : ∃ M : ℝ, ∀ x : ℝ, x < -M → g x < ε :=
sorry

end g_approaches_neg_inf_pos_g_approaches_neg_inf_neg_l238_23889


namespace crayon_cost_l238_23837

theorem crayon_cost (total_students : ℕ) (buyers : ℕ) (crayons_per_student : ℕ) (crayon_cost : ℕ) :
  total_students = 50 →
  buyers > total_students / 2 →
  buyers * crayons_per_student * crayon_cost = 1998 →
  crayon_cost > crayons_per_student →
  crayon_cost = 37 :=
by sorry

end crayon_cost_l238_23837


namespace base3_to_base10_conversion_l238_23845

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number --/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end base3_to_base10_conversion_l238_23845


namespace tan_225_degrees_l238_23878

theorem tan_225_degrees : Real.tan (225 * π / 180) = 1 := by
  sorry

end tan_225_degrees_l238_23878


namespace common_intersection_theorem_l238_23874

/-- The common intersection point of a family of lines -/
def common_intersection_point : ℝ × ℝ := (-1, 1)

/-- The equation of the family of lines -/
def line_equation (a b d x y : ℝ) : Prop :=
  (b + d) * x + b * y = b + 2 * d

theorem common_intersection_theorem :
  ∀ (a b d : ℝ), line_equation a b d (common_intersection_point.1) (common_intersection_point.2) ∧
  (∀ (x y : ℝ), (∀ a b d : ℝ, line_equation a b d x y) → (x, y) = common_intersection_point) :=
sorry

end common_intersection_theorem_l238_23874


namespace stratified_sampling_probability_l238_23819

def total_balls : ℕ := 40
def red_balls : ℕ := 16
def blue_balls : ℕ := 12
def white_balls : ℕ := 8
def yellow_balls : ℕ := 4
def sample_size : ℕ := 10

def stratified_sample_red : ℕ := 4
def stratified_sample_blue : ℕ := 3
def stratified_sample_white : ℕ := 2
def stratified_sample_yellow : ℕ := 1

theorem stratified_sampling_probability :
  (Nat.choose yellow_balls stratified_sample_yellow *
   Nat.choose white_balls stratified_sample_white *
   Nat.choose blue_balls stratified_sample_blue *
   Nat.choose red_balls stratified_sample_red) /
  Nat.choose total_balls sample_size =
  (Nat.choose 4 1 * Nat.choose 8 2 * Nat.choose 12 3 * Nat.choose 16 4) /
  Nat.choose 40 10 := by sorry

end stratified_sampling_probability_l238_23819


namespace fruit_packing_lcm_l238_23811

theorem fruit_packing_lcm : Nat.lcm 18 (Nat.lcm 9 (Nat.lcm 12 6)) = 36 := by
  sorry

end fruit_packing_lcm_l238_23811


namespace survivor_same_tribe_quit_probability_l238_23824

/-- The probability of all three quitters being from the same tribe in a Survivor-like scenario -/
theorem survivor_same_tribe_quit_probability :
  let total_people : ℕ := 20
  let tribe_size : ℕ := 10
  let num_quitters : ℕ := 3
  let total_combinations := Nat.choose total_people num_quitters
  let same_tribe_combinations := 2 * Nat.choose tribe_size num_quitters
  (same_tribe_combinations : ℚ) / total_combinations = 4 / 19 := by
  sorry

end survivor_same_tribe_quit_probability_l238_23824


namespace jug_pouring_l238_23850

/-- Represents the state of two jugs after pouring from two equal full jugs -/
structure JugState where
  x_capacity : ℚ
  y_capacity : ℚ
  x_filled : ℚ
  y_filled : ℚ
  h_x_filled : x_filled = 1/4 * x_capacity
  h_y_filled : y_filled = 2/3 * y_capacity
  h_equal_initial : x_filled + y_filled = x_capacity

/-- The fraction of jug X that contains water after filling jug Y -/
def final_x_fraction (state : JugState) : ℚ :=
  1/8

theorem jug_pouring (state : JugState) :
  final_x_fraction state = 1/8 := by
  sorry


end jug_pouring_l238_23850


namespace same_theme_probability_l238_23854

/-- The probability of two students choosing the same theme out of two options -/
theorem same_theme_probability (themes : Nat) (students : Nat) : 
  themes = 2 → students = 2 → (themes^students / 2) / themes^students = 1/2 := by
  sorry

end same_theme_probability_l238_23854


namespace remainder_11_pow_2023_mod_33_l238_23829

theorem remainder_11_pow_2023_mod_33 : 11^2023 % 33 = 11 := by
  sorry

end remainder_11_pow_2023_mod_33_l238_23829


namespace equation_solution_difference_l238_23871

theorem equation_solution_difference : ∃ (r₁ r₂ : ℝ),
  r₁ ≠ r₂ ∧
  (r₁ + 5 ≠ 0 ∧ r₂ + 5 ≠ 0) ∧
  ((r₁^2 - 5*r₁ - 24) / (r₁ + 5) = 3*r₁ + 8) ∧
  ((r₂^2 - 5*r₂ - 24) / (r₂ + 5) = 3*r₂ + 8) ∧
  |r₁ - r₂| = 4 :=
by sorry

end equation_solution_difference_l238_23871


namespace fraction_decimal_comparison_l238_23838

theorem fraction_decimal_comparison : (1 : ℚ) / 4 - 0.250000025 = 1 / (4 * 10^7) := by sorry

end fraction_decimal_comparison_l238_23838


namespace percentage_of_120_to_80_l238_23812

theorem percentage_of_120_to_80 : 
  (120 : ℝ) / 80 * 100 = 150 := by sorry

end percentage_of_120_to_80_l238_23812


namespace faster_train_speed_l238_23803

/-- Proves that the speed of the faster train is 180 km/h given the problem conditions --/
theorem faster_train_speed
  (train1_length : ℝ)
  (train2_length : ℝ)
  (initial_distance : ℝ)
  (slower_train_speed : ℝ)
  (time_to_meet : ℝ)
  (h1 : train1_length = 100)
  (h2 : train2_length = 200)
  (h3 : initial_distance = 450)
  (h4 : slower_train_speed = 90)
  (h5 : time_to_meet = 9.99920006399488)
  : ∃ (faster_train_speed : ℝ), faster_train_speed = 180 := by
  sorry

#check faster_train_speed

end faster_train_speed_l238_23803


namespace coin_array_digit_sum_l238_23804

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The total number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem coin_array_digit_sum :
  ∃ (N : ℕ), triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end coin_array_digit_sum_l238_23804


namespace congcong_carbon_emissions_l238_23867

/-- Carbon dioxide emissions calculation for household tap water -/
def carbon_emissions (water_usage : ℝ) : ℝ := water_usage * 0.91

/-- Congcong's water usage in a certain month (in tons) -/
def congcong_water_usage : ℝ := 6

/-- Theorem stating the carbon dioxide emissions from Congcong's tap water for a certain month -/
theorem congcong_carbon_emissions :
  carbon_emissions congcong_water_usage = 5.46 := by
  sorry

end congcong_carbon_emissions_l238_23867


namespace puppy_feeding_schedule_l238_23841

-- Define the feeding schedule and amounts
def total_days : ℕ := 28 -- 4 weeks
def today_food : ℚ := 1/2
def last_two_weeks_daily : ℚ := 1
def total_food : ℚ := 25

-- Define the unknown amount for the first two weeks
def first_two_weeks_per_meal : ℚ := 1/4

-- Theorem statement
theorem puppy_feeding_schedule :
  let first_two_weeks_total := 14 * 3 * first_two_weeks_per_meal
  let last_two_weeks_total := 14 * last_two_weeks_daily
  today_food + first_two_weeks_total + last_two_weeks_total = total_food :=
by sorry

end puppy_feeding_schedule_l238_23841


namespace new_energy_vehicles_analysis_l238_23813

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 149.24 * x - 33.64

-- Define the stock data for years 2017 to 2021
def stock_data : List (ℕ × ℝ) := [
  (1, 153.4),
  (2, 260.8),
  (3, 380.2),
  (4, 492.0),
  (5, 784.0)
]

-- Theorem statement
theorem new_energy_vehicles_analysis :
  -- 1. Predicted stock for 2023 exceeds 1000 million vehicles
  (regression_eq 7 > 1000) ∧
  -- 2. Stock shows increasing trend from 2017 to 2021
  (∀ i j, i < j → i ∈ List.map Prod.fst stock_data → j ∈ List.map Prod.fst stock_data →
    (stock_data.find? (λ p => p.fst = i)).map Prod.snd < (stock_data.find? (λ p => p.fst = j)).map Prod.snd) ∧
  -- 3. Residual for 2021 is 71.44
  (((stock_data.find? (λ p => p.fst = 5)).map Prod.snd).getD 0 - regression_eq 5 = 71.44) := by
  sorry

end new_energy_vehicles_analysis_l238_23813


namespace parabola_point_order_l238_23827

/-- A parabola with equation y = -(x-2)^2 + k -/
structure Parabola where
  k : ℝ

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = -(p.x - 2)^2 + para.k

theorem parabola_point_order (para : Parabola) 
  (A B C : Point)
  (hA : A.x = -2) (hB : B.x = -1) (hC : C.x = 3)
  (liesA : lies_on A para) (liesB : lies_on B para) (liesC : lies_on C para) :
  A.y < B.y ∧ B.y < C.y := by
  sorry

end parabola_point_order_l238_23827


namespace parallel_vectors_sum_l238_23887

/-- Two vectors in R² are parallel if their components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end parallel_vectors_sum_l238_23887


namespace garden_length_perimeter_ratio_l238_23862

/-- Proves that for a rectangular garden with length 24 feet and width 18 feet, 
    the ratio of its length to its perimeter is 2:7. -/
theorem garden_length_perimeter_ratio :
  let length : ℕ := 24
  let width : ℕ := 18
  let perimeter : ℕ := 2 * (length + width)
  (length : ℚ) / perimeter = 2 / 7 := by
  sorry

end garden_length_perimeter_ratio_l238_23862


namespace largest_band_size_l238_23857

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- The total number of band members --/
def totalMembers (f : BandFormation) : ℕ := f.rows * f.membersPerRow

/-- Conditions for the band formations --/
def validFormations (original new : BandFormation) (total : ℕ) : Prop :=
  total < 100 ∧
  totalMembers original + 4 = total ∧
  totalMembers new = total ∧
  new.membersPerRow = original.membersPerRow + 2 ∧
  new.rows + 3 = original.rows

/-- The theorem stating that the largest possible number of band members is 88 --/
theorem largest_band_size :
  ∀ original new : BandFormation,
  ∀ total : ℕ,
  validFormations original new total →
  total ≤ 88 :=
sorry

end largest_band_size_l238_23857


namespace triangle_inequality_sum_l238_23859

/-- Given a triangle ABC with side lengths a, b, c, and a point P in the plane of the triangle
    with distances PA = p, PB = q, PC = r, prove that pq/ab + qr/bc + rp/ac ≥ 1 -/
theorem triangle_inequality_sum (a b c p q r : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0) :
  p * q / (a * b) + q * r / (b * c) + r * p / (a * c) ≥ 1 := by
  sorry

end triangle_inequality_sum_l238_23859


namespace joan_initial_money_l238_23801

/-- The amount of money Joan had initially -/
def initial_money : ℕ := 60

/-- The cost of one container of hummus -/
def hummus_cost : ℕ := 5

/-- The number of hummus containers Joan buys -/
def hummus_quantity : ℕ := 2

/-- The cost of chicken -/
def chicken_cost : ℕ := 20

/-- The cost of bacon -/
def bacon_cost : ℕ := 10

/-- The cost of vegetables -/
def vegetable_cost : ℕ := 10

/-- The cost of one apple -/
def apple_cost : ℕ := 2

/-- The number of apples Joan can buy with remaining money -/
def apple_quantity : ℕ := 5

theorem joan_initial_money :
  initial_money = 
    hummus_cost * hummus_quantity + 
    chicken_cost + 
    bacon_cost + 
    vegetable_cost + 
    apple_cost * apple_quantity := by
  sorry

end joan_initial_money_l238_23801


namespace smallest_value_in_range_l238_23896

theorem smallest_value_in_range (x : ℝ) (h1 : -1 < x) (h2 : x < 0) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt (x^2)) :=
by sorry

end smallest_value_in_range_l238_23896


namespace unique_positive_solution_l238_23833

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^12 + 8*x^11 + 15*x^10 + 2023*x^9 - 1500*x^8

-- State the theorem
theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

end unique_positive_solution_l238_23833


namespace ellipse_foci_distance_l238_23820

/-- Distance between foci of an ellipse -/
theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 8 := by sorry

end ellipse_foci_distance_l238_23820


namespace max_sum_consecutive_integers_with_product_constraint_l238_23872

theorem max_sum_consecutive_integers_with_product_constraint : 
  ∀ n : ℕ, n * (n + 1) < 500 → n + (n + 1) ≤ 43 :=
by
  sorry

end max_sum_consecutive_integers_with_product_constraint_l238_23872


namespace watch_ahead_by_16_minutes_l238_23852

/-- Represents the time gain of a watch in minutes per hour -/
def time_gain : ℕ := 4

/-- Represents the start time in minutes after midnight -/
def start_time : ℕ := 10 * 60

/-- Represents the event time in minutes after midnight -/
def event_time : ℕ := 14 * 60

/-- Calculates the actual time passed given the time shown on the watch -/
def actual_time (watch_time : ℕ) : ℕ :=
  (watch_time * 60) / (60 + time_gain)

/-- Theorem stating that the watch shows 16 minutes ahead of the actual time -/
theorem watch_ahead_by_16_minutes :
  actual_time (event_time - start_time) = event_time - start_time - 16 := by
  sorry


end watch_ahead_by_16_minutes_l238_23852


namespace triangle_side_length_l238_23814

theorem triangle_side_length (A B C : Real) (angleB : Real) (sideAB sideAC : Real) :
  angleB = π / 4 →
  sideAB = 100 →
  sideAC = 100 →
  (∃! bc : Real, bc = sideAB * Real.sqrt 2) :=
by
  sorry

end triangle_side_length_l238_23814


namespace ratio_of_numbers_l238_23832

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : a / b = 9 / 5 := by
  sorry

end ratio_of_numbers_l238_23832


namespace lunch_combo_count_l238_23810

/-- Represents the number of options for each food category --/
structure FoodOptions where
  lettuce : Nat
  tomatoes : Nat
  olives : Nat
  bread : Nat
  fruit : Nat
  soup : Nat

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- Calculates the total number of lunch combo options --/
def lunchComboOptions (options : FoodOptions) : Nat :=
  let remainingItems := options.olives + options.bread + options.fruit
  let remainingChoices := choose remainingItems 3
  options.lettuce * options.tomatoes * remainingChoices * options.soup

/-- Theorem stating the number of lunch combo options --/
theorem lunch_combo_count (options : FoodOptions) 
  (h1 : options.lettuce = 4)
  (h2 : options.tomatoes = 5)
  (h3 : options.olives = 6)
  (h4 : options.bread = 3)
  (h5 : options.fruit = 4)
  (h6 : options.soup = 3) :
  lunchComboOptions options = 17160 := by
  sorry

#eval lunchComboOptions { lettuce := 4, tomatoes := 5, olives := 6, bread := 3, fruit := 4, soup := 3 }

end lunch_combo_count_l238_23810


namespace equation_solution_l238_23815

theorem equation_solution : 
  {x : ℝ | Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4} = {2, -2} := by
  sorry

end equation_solution_l238_23815


namespace inequality_proof_l238_23843

theorem inequality_proof (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 2) (habc : a * b * c = 1) :
  (a / (b + c)^(1/n : ℝ)) + (b / (c + a)^(1/n : ℝ)) + (c / (a + b)^(1/n : ℝ)) ≥ 3 / (2^(1/n : ℝ)) := by
  sorry

end inequality_proof_l238_23843


namespace nonagon_diagonals_octagon_diagonals_decagon_diagonals_l238_23883

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a nonagon is 27 -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by sorry

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals 8 = 20 := by sorry

/-- Theorem: The number of diagonals in a decagon is 35 -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by sorry

end nonagon_diagonals_octagon_diagonals_decagon_diagonals_l238_23883


namespace overtime_hours_calculation_l238_23831

-- Define the hourly rates and total hours
def ordinary_rate : ℚ := 60 / 100
def overtime_rate : ℚ := 90 / 100
def total_hours : ℕ := 50

-- Define the total pay in dollars
def total_pay : ℚ := 3240 / 100

-- Theorem statement
theorem overtime_hours_calculation :
  ∃ (ordinary_hours overtime_hours : ℕ),
    ordinary_hours + overtime_hours = total_hours ∧
    ordinary_rate * ordinary_hours + overtime_rate * overtime_hours = total_pay ∧
    overtime_hours = 8 := by
  sorry

end overtime_hours_calculation_l238_23831


namespace equation_solution_l238_23856

theorem equation_solution : ∃ x : ℝ, 300 * x + (12 + 4) * (1 / 8) = 602 ∧ x = 2 := by
  sorry

end equation_solution_l238_23856


namespace non_defective_engines_count_l238_23855

def total_engines (num_batches : ℕ) (engines_per_batch : ℕ) : ℕ :=
  num_batches * engines_per_batch

def non_defective_fraction : ℚ := 3/4

theorem non_defective_engines_count 
  (num_batches : ℕ) 
  (engines_per_batch : ℕ) 
  (h1 : num_batches = 5) 
  (h2 : engines_per_batch = 80) :
  ↑(total_engines num_batches engines_per_batch) * non_defective_fraction = 300 := by
  sorry

end non_defective_engines_count_l238_23855


namespace ring_worth_proof_l238_23830

theorem ring_worth_proof (total_worth car_cost : ℕ) (h1 : total_worth = 14000) (h2 : car_cost = 2000) :
  ∃ (ring_cost : ℕ), 
    ring_cost + car_cost + 2 * ring_cost = total_worth ∧ 
    ring_cost = 4000 := by
  sorry

end ring_worth_proof_l238_23830


namespace ellipse_theorem_l238_23809

/-- The equation of the given ellipse -/
def given_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

/-- The equation of the new ellipse -/
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- The point through which the new ellipse passes -/
def point : ℝ × ℝ := (3, -2)

theorem ellipse_theorem :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), given_ellipse x y → 
      ∃ (f1 f2 : ℝ × ℝ), (f1.1 = c ∧ f1.2 = 0) ∧ (f2.1 = -c ∧ f2.2 = 0) ∧
        (x - f1.1)^2 + y^2 + (x - f2.1)^2 + y^2 = 
        ((x - f1.1)^2 + y^2)^(1/2) + ((x - f2.1)^2 + y^2)^(1/2)) ∧
    (∀ (x y : ℝ), new_ellipse x y →
      ∃ (f1 f2 : ℝ × ℝ), (f1.1 = c ∧ f1.2 = 0) ∧ (f2.1 = -c ∧ f2.2 = 0) ∧
        (x - f1.1)^2 + y^2 + (x - f2.1)^2 + y^2 = 
        ((x - f1.1)^2 + y^2)^(1/2) + ((x - f2.1)^2 + y^2)^(1/2))) ∧
  new_ellipse point.1 point.2 :=
by sorry


end ellipse_theorem_l238_23809


namespace wedge_volume_l238_23849

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h r : ℝ) (h1 : d = 20) (h2 : h = d) (h3 : r = d / 2) : 
  ∃ (m : ℕ), (1 / 3) * π * r^2 * h = m * π ∧ m = 667 := by
  sorry

end wedge_volume_l238_23849


namespace fourth_power_sum_l238_23879

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_sq_eq : a^2 + b^2 + c^2 = 6)
  (sum_cube_eq : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 := by
  sorry

end fourth_power_sum_l238_23879


namespace ceiling_neg_sqrt_64_over_9_l238_23898

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64 / 9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_64_over_9_l238_23898


namespace mess_expenditure_original_mess_expenditure_l238_23876

/-- Calculates the original daily expenditure of a mess given initial conditions. -/
theorem mess_expenditure (initial_students : ℕ) (new_students : ℕ) (expense_increase : ℕ) (avg_decrease : ℕ) : ℕ :=
  let total_students : ℕ := initial_students + new_students
  let original_expenditure : ℕ := initial_students * (total_students * expense_increase) / (total_students * avg_decrease)
  original_expenditure

/-- Proves that the original daily expenditure of the mess was 420 given the specified conditions. -/
theorem original_mess_expenditure :
  mess_expenditure 35 7 42 1 = 420 := by
  sorry

end mess_expenditure_original_mess_expenditure_l238_23876


namespace geometric_decreasing_condition_l238_23826

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

theorem geometric_decreasing_condition (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) (h_pos : a 1 > 0) :
  (is_decreasing_sequence a → a 1 > a 2) ∧
  ¬(a 1 > a 2 → is_decreasing_sequence a) :=
sorry

end geometric_decreasing_condition_l238_23826


namespace distinct_values_count_l238_23895

-- Define a type for expressions
inductive Expr
  | Num : ℕ → Expr
  | Pow : Expr → Expr → Expr

-- Define a function to evaluate expressions
def eval : Expr → ℕ
  | Expr.Num n => n
  | Expr.Pow a b => (eval a) ^ (eval b)

-- Define the base expression
def baseExpr : Expr :=
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3)))

-- Define all possible parenthesizations
def parenthesizations : List Expr := [
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3))),
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Num 3)),
  Expr.Pow (Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Num 3)) (Expr.Num 3),
  Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3))) (Expr.Num 3),
  Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Pow (Expr.Num 3) (Expr.Num 3))
]

-- Theorem: The number of distinct values is 3
theorem distinct_values_count :
  (parenthesizations.map eval).toFinset.card = 3 := by sorry

end distinct_values_count_l238_23895


namespace alice_original_seat_l238_23822

/-- Represents the possible seats in the lecture hall -/
inductive Seat
  | one
  | two
  | three
  | four
  | five
  | six
  | seven

/-- Represents the movement of a person -/
inductive Movement
  | left : Nat → Movement
  | right : Nat → Movement
  | stay : Movement
  | switch : Movement

/-- Represents a person and their movement -/
structure Person where
  name : String
  movement : Movement

/-- The state of the seating arrangement -/
structure SeatingArrangement where
  seats : Vector Person 7
  aliceOriginalSeat : Seat
  aliceFinalSeat : Seat

def isEndSeat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.seven

/-- The theorem to prove -/
theorem alice_original_seat
  (arrangement : SeatingArrangement)
  (beth_moves_right : arrangement.seats[1].movement = Movement.right 1)
  (carla_moves_left : arrangement.seats[2].movement = Movement.left 2)
  (dana_elly_switch : arrangement.seats[3].movement = Movement.switch ∧
                      arrangement.seats[4].movement = Movement.switch)
  (fiona_moves_left : arrangement.seats[5].movement = Movement.left 1)
  (grace_stays : arrangement.seats[6].movement = Movement.stay)
  (alice_ends_in_end_seat : isEndSeat arrangement.aliceFinalSeat) :
  arrangement.aliceOriginalSeat = Seat.five := by
  sorry

end alice_original_seat_l238_23822


namespace simple_interest_growth_factor_l238_23868

/-- The growth factor for simple interest -/
def growth_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

/-- Theorem: The growth factor for a 5% simple interest rate over 20 years is 2 -/
theorem simple_interest_growth_factor : 
  growth_factor (5 / 100) 20 = 2 := by
  sorry

end simple_interest_growth_factor_l238_23868


namespace fixed_point_of_log_function_l238_23890

-- Define the logarithm function for any base a > 0 and a ≠ 1
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f(x) = log_a(x+2) + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 2) + 1

-- State the theorem
theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = 1 := by sorry

end fixed_point_of_log_function_l238_23890


namespace triangle_area_l238_23836

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (∀ x y z, (x = a ∧ y = b ∧ z = c) → 
    (x = 2 * (y * Real.sin C) * (z * Real.sin B) / (Real.sin A)) ∧
    (y^2 + z^2 - x^2 = 8)) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 / 3) :=
by sorry

end triangle_area_l238_23836


namespace negation_is_returning_transformation_l238_23863

theorem negation_is_returning_transformation (a : ℝ) : -(-a) = a := by
  sorry

end negation_is_returning_transformation_l238_23863


namespace negative_division_result_l238_23899

theorem negative_division_result : (-150) / (-25) = 6 := by
  sorry

end negative_division_result_l238_23899


namespace linear_inequality_m_value_l238_23805

theorem linear_inequality_m_value (m : ℝ) : 
  (∀ x, ∃ a b, (m - 2) * x^(|m - 1|) - 3 > 6 ↔ a * x + b > 0) → 
  (m - 2 ≠ 0) → 
  m = 0 := by
sorry

end linear_inequality_m_value_l238_23805


namespace alice_lost_second_game_l238_23873

/-- Represents a participant in the arm-wrestling contest -/
inductive Participant
  | Alice
  | Belle
  | Cathy

/-- Represents the state of a participant in a game -/
inductive GameState
  | Playing
  | Resting

/-- Represents the result of a game for a participant -/
inductive GameResult
  | Win
  | Lose

/-- The total number of games played -/
def totalGames : Nat := 21

/-- The number of times each participant played -/
def timesPlayed (p : Participant) : Nat :=
  match p with
  | Participant.Alice => 10
  | Participant.Belle => 15
  | Participant.Cathy => 17

/-- The state of a participant in a specific game -/
def participantState (p : Participant) (gameNumber : Nat) : GameState := sorry

/-- The result of a game for a participant -/
def gameResult (p : Participant) (gameNumber : Nat) : Option GameResult := sorry

theorem alice_lost_second_game :
  gameResult Participant.Alice 2 = some GameResult.Lose := by sorry

end alice_lost_second_game_l238_23873


namespace gcd_of_powers_of_101_l238_23800

theorem gcd_of_powers_of_101 (h : Prime 101) :
  Nat.gcd (101^5 + 1) (101^5 + 101^3 + 101 + 1) = 1 := by
  sorry

end gcd_of_powers_of_101_l238_23800


namespace distance_proof_l238_23853

/-- The distance between two locations A and B, where two buses meet under specific conditions --/
def distance_between_locations : ℝ :=
  let first_meeting_distance : ℝ := 85
  let second_meeting_distance : ℝ := 65
  3 * first_meeting_distance - second_meeting_distance

theorem distance_proof :
  let first_meeting_distance : ℝ := 85
  let second_meeting_distance : ℝ := 65
  let total_distance := distance_between_locations
  (∃ (speed_A speed_B : ℝ), speed_A > 0 ∧ speed_B > 0 ∧
    first_meeting_distance / speed_A = (total_distance - first_meeting_distance) / speed_B ∧
    (total_distance - first_meeting_distance + second_meeting_distance) / speed_A + 0.5 =
    (first_meeting_distance + (total_distance - second_meeting_distance)) / speed_B + 0.5) →
  total_distance = 190 := by
  sorry

end distance_proof_l238_23853


namespace jesse_book_reading_l238_23891

theorem jesse_book_reading (total_pages : ℕ) (pages_read : ℕ) (pages_left : ℕ) : 
  pages_left = 166 → 
  pages_read = total_pages / 3 → 
  pages_left = 2 * total_pages / 3 → 
  pages_read = 83 := by
sorry

end jesse_book_reading_l238_23891


namespace solve_for_m_l238_23884

theorem solve_for_m : ∃ m : ℝ, 
  (∀ x : ℝ, x > 2 ↔ x - 3*m + 1 > 0) → m = 1 := by
  sorry

end solve_for_m_l238_23884


namespace fixed_point_exponential_function_l238_23828

theorem fixed_point_exponential_function 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 1
  f 1 = 2 := by
sorry

end fixed_point_exponential_function_l238_23828


namespace number_equality_l238_23802

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (16/216) * (1/x)) : x = 1/3 := by
  sorry

end number_equality_l238_23802


namespace complement_of_union_l238_23844

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 4}
def N : Set ℕ := {2, 4}

theorem complement_of_union (U M N : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hM : M = {0, 4}) (hN : N = {2, 4}) :
  U \ (M ∪ N) = {1, 3} := by
  sorry

end complement_of_union_l238_23844


namespace square_IJKL_side_length_l238_23870

-- Define the side lengths of squares ABCD and EFGH
def side_ABCD : ℝ := 3
def side_EFGH : ℝ := 9

-- Define the right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the arrangement of triangles
def triangle_arrangement (t : RightTriangle) : Prop :=
  t.leg1 - t.leg2 = side_ABCD ∧ t.leg1 + t.leg2 = side_EFGH

-- Theorem statement
theorem square_IJKL_side_length 
  (t : RightTriangle) 
  (h : triangle_arrangement t) : 
  t.hypotenuse = 3 * Real.sqrt 5 := by
  sorry

end square_IJKL_side_length_l238_23870


namespace gcd_lcm_sum_bound_gcd_lcm_sum_equality_condition_l238_23817

theorem gcd_lcm_sum_bound (a b : ℕ) (h1 : a * b > 2) :
  let d := Nat.gcd a b
  let l := Nat.lcm a b
  (∃ k : ℕ, d + l = k * (a + b)) →
  (d + l) / (a + b) ≤ (a + b) / 4 := by
sorry

theorem gcd_lcm_sum_equality_condition (a b : ℕ) (h1 : a * b > 2) :
  let d := Nat.gcd a b
  let l := Nat.lcm a b
  (∃ k : ℕ, d + l = k * (a + b)) →
  ((d + l) / (a + b) = (a + b) / 4 ↔
   ∃ (x y : ℕ), a = d * x ∧ b = d * y ∧ x = y + 2) := by
sorry

end gcd_lcm_sum_bound_gcd_lcm_sum_equality_condition_l238_23817


namespace derivative_f_at_pi_third_l238_23897

open Real

theorem derivative_f_at_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = x + Real.sin x) :
  deriv f (π / 3) = 3 / 2 := by
  sorry

end derivative_f_at_pi_third_l238_23897


namespace tan_plus_four_sin_twenty_degrees_l238_23869

theorem tan_plus_four_sin_twenty_degrees :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_plus_four_sin_twenty_degrees_l238_23869


namespace students_per_class_is_twenty_l238_23808

/-- Represents a school with teachers, a principal, classes, and students. -/
structure School where
  teachers : ℕ
  principal : ℕ
  classes : ℕ
  total_people : ℕ
  students_per_class : ℕ

/-- Theorem stating that in a school with given parameters, there are 20 students in each class. -/
theorem students_per_class_is_twenty (school : School)
  (h1 : school.teachers = 48)
  (h2 : school.principal = 1)
  (h3 : school.classes = 15)
  (h4 : school.total_people = 349)
  (h5 : school.total_people = school.teachers + school.principal + school.classes * school.students_per_class) :
  school.students_per_class = 20 := by
  sorry

end students_per_class_is_twenty_l238_23808


namespace division_remainder_problem_l238_23851

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 172 →
  divisor = 17 →
  quotient = 10 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end division_remainder_problem_l238_23851


namespace sarahs_lemonade_profit_l238_23875

/-- Calculates the total profit for Sarah's lemonade stand --/
theorem sarahs_lemonade_profit
  (total_days : ℕ)
  (hot_days : ℕ)
  (cups_per_day : ℕ)
  (cost_per_cup : ℚ)
  (hot_day_price : ℚ)
  (hot_day_markup : ℚ)
  (h1 : total_days = 10)
  (h2 : hot_days = 3)
  (h3 : cups_per_day = 32)
  (h4 : cost_per_cup = 3/4)
  (h5 : hot_day_price = 1.6351744186046513)
  (h6 : hot_day_markup = 5/4) :
  ∃ (profit : ℚ), profit = 210.2265116279069 :=
by sorry

end sarahs_lemonade_profit_l238_23875


namespace square_property_contradiction_l238_23858

theorem square_property_contradiction (property : ℝ → ℝ) 
  (h_prop : ∀ x : ℝ, property x = (x^2) * property 1) : 
  ¬ (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b = 5 * a ∧ property b = 5 * property a) :=
sorry

end square_property_contradiction_l238_23858


namespace function_passes_through_point_l238_23881

/-- The function f(x) = 3x + 1 passes through the point (2,7) -/
theorem function_passes_through_point :
  let f : ℝ → ℝ := λ x ↦ 3 * x + 1
  f 2 = 7 := by sorry

end function_passes_through_point_l238_23881


namespace total_kids_played_tag_l238_23893

def monday : ℕ := 12
def tuesday : ℕ := 7
def wednesday : ℕ := 15
def thursday : ℕ := 10
def friday : ℕ := 18

theorem total_kids_played_tag : monday + tuesday + wednesday + thursday + friday = 62 := by
  sorry

end total_kids_played_tag_l238_23893


namespace runner_solution_l238_23806

def runner_problem (t : ℕ) : Prop :=
  let first_runner := 2
  let second_runner := 4
  let third_runner := t
  let meeting_time := 44
  (meeting_time % first_runner = 0) ∧
  (meeting_time % second_runner = 0) ∧
  (meeting_time % third_runner = 0) ∧
  (first_runner < third_runner) ∧
  (second_runner < third_runner) ∧
  (∀ t' < meeting_time, t' % first_runner = 0 → t' % second_runner = 0 → t' % third_runner ≠ 0)

theorem runner_solution : runner_problem 11 := by
  sorry

end runner_solution_l238_23806


namespace vector_relations_l238_23840

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-3, 1)

theorem vector_relations (k : ℝ) :
  (((k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) → k = 3/2) ∧
  ((∃ t : ℝ, (k * a.1 + b.1, k * a.2 + b.2) = t • (a.1 - 3 * b.1, a.2 - 3 * b.2)) → k = -1/3) :=
by sorry

end vector_relations_l238_23840


namespace lunch_cost_theorem_l238_23807

/-- Calculates the total cost of lunches for a field trip --/
def total_lunch_cost (total_people : ℕ) (extra_lunches : ℕ) (vegetarian : ℕ) (gluten_free : ℕ) 
  (nut_free : ℕ) (halal : ℕ) (veg_and_gf : ℕ) (regular_cost : ℕ) (special_cost : ℕ) 
  (veg_gf_cost : ℕ) : ℕ :=
  let total_lunches := total_people + extra_lunches
  let regular_lunches := total_lunches - (vegetarian + gluten_free + nut_free + halal - veg_and_gf)
  let regular_total := regular_lunches * regular_cost
  let vegetarian_total := (vegetarian - veg_and_gf) * special_cost
  let gluten_free_total := gluten_free * special_cost
  let nut_free_total := nut_free * special_cost
  let halal_total := halal * special_cost
  let veg_gf_total := veg_and_gf * veg_gf_cost
  regular_total + vegetarian_total + gluten_free_total + nut_free_total + halal_total + veg_gf_total

theorem lunch_cost_theorem :
  total_lunch_cost 41 3 10 5 3 4 2 7 8 9 = 346 := by
  sorry

#eval total_lunch_cost 41 3 10 5 3 4 2 7 8 9

end lunch_cost_theorem_l238_23807


namespace f_monotone_decreasing_on_interval_l238_23861

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem f_monotone_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≥ f x₂ := by
  sorry

end f_monotone_decreasing_on_interval_l238_23861


namespace water_level_accurate_l238_23880

/-- Represents the water level function for a reservoir -/
def water_level (x : ℝ) : ℝ := 6 + 0.3 * x

/-- Theorem stating that the water level function accurately describes the reservoir's water level -/
theorem water_level_accurate (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  water_level x = 6 + 0.3 * x ∧ water_level x ≥ 6 ∧ water_level x ≤ 7.5 := by
  sorry

end water_level_accurate_l238_23880


namespace equivalent_division_l238_23885

theorem equivalent_division (x : ℝ) : (x / (3/9)) * (2/15) = x / 2.5 := by
  sorry

end equivalent_division_l238_23885


namespace arithmetic_equality_l238_23847

theorem arithmetic_equality : 3 * 12 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end arithmetic_equality_l238_23847


namespace x_value_l238_23835

theorem x_value (x y z : ℝ) (h1 : x = y) (h2 : x = 2*z) (h3 : x*y*z = 256) : x = 8 := by
  sorry

end x_value_l238_23835


namespace tangency_point_satisfies_equations_unique_tangency_point_l238_23834

/-- The point of tangency for two parabolas -/
def point_of_tangency : ℝ × ℝ := (-7, -24)

/-- The first parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32

/-- The second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations :
  parabola1 point_of_tangency.1 point_of_tangency.2 ∧
  parabola2 point_of_tangency.1 point_of_tangency.2 := by sorry

/-- Theorem stating that the point_of_tangency is the unique point satisfying both equations -/
theorem unique_tangency_point :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency := by sorry

end tangency_point_satisfies_equations_unique_tangency_point_l238_23834


namespace jordan_fourth_period_shots_l238_23894

/-- Given Jordan's shot-blocking performance in a hockey game, prove the number of shots blocked in the fourth period. -/
theorem jordan_fourth_period_shots 
  (first_period : ℕ) 
  (second_period : ℕ)
  (third_period : ℕ)
  (total_shots : ℕ)
  (h1 : first_period = 4)
  (h2 : second_period = 2 * first_period)
  (h3 : third_period = second_period - 3)
  (h4 : total_shots = 21)
  : total_shots - (first_period + second_period + third_period) = 4 := by
  sorry

end jordan_fourth_period_shots_l238_23894


namespace bruce_lost_eggs_main_theorem_l238_23842

/-- Proof that Bruce lost 70 eggs -/
theorem bruce_lost_eggs : ℕ → ℕ → ℕ → Prop :=
  fun initial_eggs remaining_eggs lost_eggs =>
    initial_eggs = 75 →
    remaining_eggs = 5 →
    lost_eggs = initial_eggs - remaining_eggs →
    lost_eggs = 70

/-- Main theorem statement -/
theorem main_theorem : ∃ lost_eggs : ℕ, bruce_lost_eggs 75 5 lost_eggs := by
  sorry

end bruce_lost_eggs_main_theorem_l238_23842


namespace rectangle_dimensions_l238_23892

theorem rectangle_dimensions (w l : ℕ) : 
  l = w + 5 →
  2 * l + 2 * w = 34 →
  w = 6 ∧ l = 11 := by
sorry

end rectangle_dimensions_l238_23892


namespace annas_money_l238_23825

theorem annas_money (initial_amount : ℚ) : 
  (initial_amount * (1 - 3/8) * (1 - 1/5) = 36) → initial_amount = 72 := by
  sorry

end annas_money_l238_23825


namespace alberto_bjorn_distance_difference_l238_23886

-- Define the speeds and time
def alberto_speed : ℝ := 12
def bjorn_speed : ℝ := 9
def time : ℝ := 6

-- Theorem statement
theorem alberto_bjorn_distance_difference :
  alberto_speed * time - bjorn_speed * time = 18 := by
  sorry

end alberto_bjorn_distance_difference_l238_23886


namespace vector_addition_l238_23846

theorem vector_addition (A B C : ℝ × ℝ) : 
  (B.1 - A.1, B.2 - A.2) = (0, 1) →
  (C.1 - B.1, C.2 - B.2) = (1, 0) →
  (C.1 - A.1, C.2 - A.2) = (1, 1) := by
sorry

end vector_addition_l238_23846


namespace equation_solution_l238_23860

theorem equation_solution :
  let x : ℚ := 32
  let n : ℚ := -5/6
  35 - (23 - (15 - x)) = 12 * n / (1 / 2) := by sorry

end equation_solution_l238_23860


namespace complex_sum_theorem_l238_23888

theorem complex_sum_theorem (a b c : ℂ) 
  (h1 : a^2 + a*b + b^2 = 1)
  (h2 : b^2 + b*c + c^2 = -1)
  (h3 : c^2 + c*a + a^2 = Complex.I) :
  a*b + b*c + c*a = Complex.I ∨ a*b + b*c + c*a = -Complex.I := by
  sorry

end complex_sum_theorem_l238_23888


namespace metallic_sheet_length_is_48_l238_23839

/-- Represents the dimensions and properties of a metallic sheet and the box made from it. -/
structure MetallicSheet where
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Calculates the length of the original metallic sheet given its properties. -/
def calculateLength (sheet : MetallicSheet) : ℝ :=
  sorry

/-- Theorem stating that for a sheet with width 36m, cut size 8m, and resulting box volume 5120m³,
    the original length is 48m. -/
theorem metallic_sheet_length_is_48 :
  let sheet : MetallicSheet := ⟨36, 8, 5120⟩
  calculateLength sheet = 48 := by
  sorry

end metallic_sheet_length_is_48_l238_23839


namespace hiking_problem_l238_23816

/-- Proves that the number of people in each van is 5, given the conditions of the hiking problem --/
theorem hiking_problem (num_cars num_taxis num_vans : ℕ) 
                       (people_per_car people_per_taxi total_people : ℕ) 
                       (h1 : num_cars = 3)
                       (h2 : num_taxis = 6)
                       (h3 : num_vans = 2)
                       (h4 : people_per_car = 4)
                       (h5 : people_per_taxi = 6)
                       (h6 : total_people = 58)
                       (h7 : total_people = num_cars * people_per_car + 
                                            num_taxis * people_per_taxi + 
                                            num_vans * (total_people - num_cars * people_per_car - num_taxis * people_per_taxi) / num_vans) : 
  (total_people - num_cars * people_per_car - num_taxis * people_per_taxi) / num_vans = 5 := by
  sorry

end hiking_problem_l238_23816


namespace min_value_of_expression_l238_23864

theorem min_value_of_expression (x y : ℝ) : (x*y - 1)^3 + (x + y)^3 ≥ -1 := by
  sorry

end min_value_of_expression_l238_23864


namespace tangent_sum_identity_l238_23865

theorem tangent_sum_identity (α β γ : ℝ) (h : α + β + γ = Real.pi / 2) :
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end tangent_sum_identity_l238_23865
