import Mathlib

namespace temperature_at_14_minutes_l188_18841

/-- Represents the temperature change over time -/
structure TemperatureChange where
  initialTemp : ℝ
  rate : ℝ

/-- Calculates the temperature at a given time -/
def temperature (tc : TemperatureChange) (t : ℝ) : ℝ :=
  tc.initialTemp + tc.rate * t

/-- Theorem: The temperature at 14 minutes is 52°C given the conditions -/
theorem temperature_at_14_minutes (tc : TemperatureChange) 
    (h1 : tc.initialTemp = 10)
    (h2 : tc.rate = 3) : 
    temperature tc 14 = 52 := by
  sorry

#eval temperature { initialTemp := 10, rate := 3 } 14

end temperature_at_14_minutes_l188_18841


namespace shirt_count_l188_18871

theorem shirt_count (total : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : total = 420) 
  (h2 : blue = 85) 
  (h3 : green = 157) : 
  total - (blue + green) = 178 := by
  sorry

end shirt_count_l188_18871


namespace statues_painted_l188_18853

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/8 ∧ paint_per_statue = 1/8 → total_paint / paint_per_statue = 7 := by
  sorry

end statues_painted_l188_18853


namespace paint_usage_l188_18895

theorem paint_usage (total_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) : 
  total_paint = 360 →
  first_week_fraction = 1/6 →
  total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
by sorry

end paint_usage_l188_18895


namespace mean_home_runs_l188_18891

def player_count : ℕ := 6 + 4 + 3 + 1

def total_home_runs : ℕ := 6 * 6 + 7 * 4 + 8 * 3 + 10 * 1

theorem mean_home_runs : (total_home_runs : ℚ) / player_count = 7 := by
  sorry

end mean_home_runs_l188_18891


namespace expression_simplification_l188_18801

theorem expression_simplification (x y : ℝ) (h : (x + 2)^3 - (y - 2)^3 ≠ 0) :
  ((x + 2)^3 + (y + x)^3) / ((x + 2)^3 - (y - 2)^3) = (2*x + y + 2) / (x - y + 4) := by
  sorry

end expression_simplification_l188_18801


namespace restaurant_tables_difference_l188_18830

theorem restaurant_tables_difference (total_tables : ℕ) (total_capacity : ℕ) 
  (new_table_capacity : ℕ) (original_table_capacity : ℕ) :
  total_tables = 40 →
  total_capacity = 212 →
  new_table_capacity = 6 →
  original_table_capacity = 4 →
  ∃ (new_tables original_tables : ℕ),
    new_tables + original_tables = total_tables ∧
    new_table_capacity * new_tables + original_table_capacity * original_tables = total_capacity ∧
    new_tables - original_tables = 12 :=
by sorry

end restaurant_tables_difference_l188_18830


namespace income_b_is_7200_l188_18851

/-- Represents the monthly income and expenditure of two individuals -/
structure MonthlyFinances where
  income_ratio : Rat × Rat
  expenditure_ratio : Rat × Rat
  savings_a : ℕ
  savings_b : ℕ

/-- Calculates the monthly income of the second individual given the financial data -/
def calculate_income_b (finances : MonthlyFinances) : ℕ :=
  sorry

/-- Theorem stating that given the specific financial data, the income of b is 7200 -/
theorem income_b_is_7200 (finances : MonthlyFinances) 
  (h1 : finances.income_ratio = (5, 6))
  (h2 : finances.expenditure_ratio = (3, 4))
  (h3 : finances.savings_a = 1800)
  (h4 : finances.savings_b = 1600) :
  calculate_income_b finances = 7200 := by
  sorry

end income_b_is_7200_l188_18851


namespace simple_interest_from_sum_and_true_discount_l188_18881

/-- Simple interest calculation given sum and true discount -/
theorem simple_interest_from_sum_and_true_discount
  (sum : ℝ) (true_discount : ℝ) (h1 : sum = 947.1428571428571)
  (h2 : true_discount = 78) :
  sum - (sum - true_discount) = true_discount :=
by sorry

end simple_interest_from_sum_and_true_discount_l188_18881


namespace distance_satisfies_conditions_l188_18879

/-- The distance from the village to the post-office in kilometers. -/
def D : ℝ := 20

/-- The speed of the man traveling to the post-office in km/h. -/
def speed_to_postoffice : ℝ := 25

/-- The speed of the man walking back to the village in km/h. -/
def speed_to_village : ℝ := 4

/-- The total time for the round trip in hours. -/
def total_time : ℝ := 5.8

/-- Theorem stating that the distance D satisfies the given conditions. -/
theorem distance_satisfies_conditions : 
  D / speed_to_postoffice + D / speed_to_village = total_time :=
sorry

end distance_satisfies_conditions_l188_18879


namespace perfect_fourth_power_in_range_l188_18870

theorem perfect_fourth_power_in_range : ∃! K : ℤ,
  (K > 0) ∧
  (∃ Z : ℤ, 1000 < Z ∧ Z < 2000 ∧ Z = K * K^3) ∧
  (∃ n : ℤ, K^4 = n^4) :=
by sorry

end perfect_fourth_power_in_range_l188_18870


namespace greatest_two_digit_with_digit_product_nine_l188_18817

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_nine :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 9 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 9 → m ≤ n :=
sorry

end greatest_two_digit_with_digit_product_nine_l188_18817


namespace sum_of_one_third_and_two_thirds_equals_one_l188_18835

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  (n : ℚ) / 9

theorem sum_of_one_third_and_two_thirds_equals_one :
  RepeatingDecimal 3 + RepeatingDecimal 6 = 1 := by
  sorry

end sum_of_one_third_and_two_thirds_equals_one_l188_18835


namespace A_when_one_is_element_B_is_zero_and_neg_one_third_l188_18868

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x - 3 = 0}

-- Theorem 1: If 1 ∈ A, then A = {1, -3}
theorem A_when_one_is_element (a : ℝ) : 1 ∈ A a → A a = {1, -3} := by sorry

-- Define the set B
def B : Set ℝ := {a : ℝ | ∃! x, x ∈ A a}

-- Theorem 2: B = {0, -1/3}
theorem B_is_zero_and_neg_one_third : B = {0, -1/3} := by sorry

end A_when_one_is_element_B_is_zero_and_neg_one_third_l188_18868


namespace percentage_subtraction_l188_18800

theorem percentage_subtraction (total : ℝ) (difference : ℝ) : 
  total = 8000 → 
  difference = 796 → 
  ∃ (P : ℝ), (1/10 * total) - (P/100 * total) = difference ∧ P = 5 := by
sorry

end percentage_subtraction_l188_18800


namespace proportional_function_ratio_l188_18893

theorem proportional_function_ratio (k a b : ℝ) : 
  k ≠ 0 →
  b ≠ 0 →
  3 = k * 1 →
  b = k * a →
  a / b = 1 / 3 := by
sorry

end proportional_function_ratio_l188_18893


namespace ratio_of_arithmetic_sums_l188_18829

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem ratio_of_arithmetic_sums : 
  let n₁ := (60 - 4) / 4 + 1
  let n₂ := (72 - 6) / 6 + 1
  (arithmetic_sum 4 4 n₁) / (arithmetic_sum 6 6 n₂) = 40 / 39 := by
  sorry

end ratio_of_arithmetic_sums_l188_18829


namespace dot_product_range_l188_18855

-- Define the unit circle
def unit_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 1

-- Define point A
def A : ℝ × ℝ := (-2, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define vector AO
def vector_AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)

-- Define vector AP
def vector_AP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - A.1, P.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_range :
  ∀ P : ℝ × ℝ, unit_circle P →
    2 ≤ dot_product vector_AO (vector_AP P) ∧
    dot_product vector_AO (vector_AP P) ≤ 6 :=
by sorry

end dot_product_range_l188_18855


namespace sufficient_condition_quadratic_inequality_l188_18832

theorem sufficient_condition_quadratic_inequality (m : ℝ) :
  (m ≥ 2) →
  (∀ x : ℝ, x^2 - 2*x + m ≥ 0) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - 2*x + m ≥ 0) → m ≥ 2) :=
by sorry

end sufficient_condition_quadratic_inequality_l188_18832


namespace three_digit_number_difference_l188_18805

/-- Represents a three-digit number with digits h, t, u from left to right -/
structure ThreeDigitNumber where
  h : ℕ
  t : ℕ
  u : ℕ
  h_lt_10 : h < 10
  t_lt_10 : t < 10
  u_lt_10 : u < 10
  h_gt_u : h > u

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.h + 10 * n.t + n.u

/-- The reversed value of a three-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.u + 10 * n.t + n.h

theorem three_digit_number_difference (n : ThreeDigitNumber) :
  n.value - n.reversed_value = 4 → n.h = 9 ∧ n.u = 5 := by
  sorry

end three_digit_number_difference_l188_18805


namespace bomb_guaranteed_four_of_a_kind_guaranteed_l188_18802

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (ranks : ℕ)

/-- Represents the minimum number of cards to draw to ensure a "bomb" -/
def min_cards_for_bomb (d : Deck) : ℕ := d.ranks * (d.suits - 1) + 1

/-- Theorem: Drawing 40 cards from a standard deck guarantees a "bomb" -/
theorem bomb_guaranteed (d : Deck) 
  (h1 : d.total_cards = 52) 
  (h2 : d.suits = 4) 
  (h3 : d.ranks = 13) : 
  min_cards_for_bomb d = 40 := by
sorry

/-- Corollary: Drawing 40 cards guarantees at least four cards of the same rank -/
theorem four_of_a_kind_guaranteed (d : Deck) 
  (h1 : d.total_cards = 52) 
  (h2 : d.suits = 4) 
  (h3 : d.ranks = 13) : 
  ∃ (n : ℕ), n ≤ 40 ∧ (∀ (m : ℕ), m ≥ n → ∃ (r : ℕ), r ≤ d.ranks ∧ 4 ≤ m - (d.ranks - 1) * 3) := by
sorry

end bomb_guaranteed_four_of_a_kind_guaranteed_l188_18802


namespace prec_2011_130_l188_18884

-- Define the new operation ⪯
def prec (a b : ℕ) : ℕ := b * 10 + a * 2

-- Theorem to prove
theorem prec_2011_130 : prec 2011 130 = 5322 := by
  sorry

end prec_2011_130_l188_18884


namespace limit_sin_x_over_x_sin_x_over_x_squeeze_cos_continuous_limit_sin_x_over_x_equals_one_l188_18811

open Real
open Topology
open Filter

theorem limit_sin_x_over_x : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |sin x / x - 1| < ε :=
by
  sorry

theorem sin_x_over_x_squeeze (x : ℝ) (h : x ≠ 0) (h' : |x| < π/2) :
  cos x < sin x / x ∧ sin x / x < 1 :=
by
  sorry

theorem cos_continuous : Continuous cos :=
by
  sorry

theorem limit_sin_x_over_x_equals_one :
  Tendsto (λ x => sin x / x) (𝓝[≠] 0) (𝓝 1) :=
by
  sorry

end limit_sin_x_over_x_sin_x_over_x_squeeze_cos_continuous_limit_sin_x_over_x_equals_one_l188_18811


namespace soda_price_theorem_l188_18886

/-- Calculates the price of a given number of soda cans with a discount applied to full cases. -/
def discounted_soda_price (regular_price : ℚ) (discount_percent : ℚ) (case_size : ℕ) (num_cans : ℕ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  let full_cases := num_cans / case_size
  let remaining_cans := num_cans % case_size
  full_cases * (case_size : ℚ) * discounted_price + (remaining_cans : ℚ) * discounted_price

/-- The price of 75 cans of soda purchased in 24-can cases with a 10% discount is $10.125. -/
theorem soda_price_theorem :
  discounted_soda_price (15/100) (1/10) 24 75 = 10125/1000 := by
  sorry

end soda_price_theorem_l188_18886


namespace provision_duration_l188_18859

theorem provision_duration 
  (initial_soldiers : ℕ) 
  (initial_consumption : ℚ) 
  (new_soldiers : ℕ) 
  (new_consumption : ℚ) 
  (new_duration : ℕ) 
  (h1 : initial_soldiers = 1200)
  (h2 : initial_consumption = 3)
  (h3 : new_soldiers = 1728)
  (h4 : new_consumption = 5/2)
  (h5 : new_duration = 25) : 
  ∃ (initial_duration : ℕ), 
    initial_duration = 30 ∧ 
    (initial_soldiers : ℚ) * initial_consumption * initial_duration = 
    (new_soldiers : ℚ) * new_consumption * new_duration :=
by sorry

end provision_duration_l188_18859


namespace pizza_and_burgers_cost_l188_18819

/-- The cost of a burger in dollars -/
def burger_cost : ℕ := 9

/-- The cost of a pizza in dollars -/
def pizza_cost : ℕ := 2 * burger_cost

/-- The total cost of one pizza and three burgers in dollars -/
def total_cost : ℕ := pizza_cost + 3 * burger_cost

theorem pizza_and_burgers_cost : total_cost = 45 := by
  sorry

end pizza_and_burgers_cost_l188_18819


namespace group_size_solve_group_size_l188_18842

/-- The number of persons in the group -/
def n : ℕ := sorry

/-- The age of the replaced person -/
def replaced_age : ℕ := 45

/-- The age of the new person -/
def new_age : ℕ := 15

/-- The decrease in average age -/
def avg_decrease : ℕ := 3

theorem group_size :
  (n * replaced_age - (replaced_age - new_age)) = (n * (replaced_age - avg_decrease)) :=
sorry

theorem solve_group_size : n = 10 :=
sorry

end group_size_solve_group_size_l188_18842


namespace second_day_speed_l188_18863

/-- Represents the speed and duration of travel for a day -/
structure DayTravel where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (travel : DayTravel) : ℝ := travel.speed * travel.duration

/-- Proves that the speed on the second day of the trip was 6 miles per hour -/
theorem second_day_speed (
  total_distance : ℝ)
  (day1 : DayTravel)
  (day3 : DayTravel)
  (day2_duration1 : ℝ)
  (day2_duration2 : ℝ)
  (h1 : total_distance = 115)
  (h2 : day1.speed = 5 ∧ day1.duration = 7)
  (h3 : day3.speed = 7 ∧ day3.duration = 5)
  (h4 : day2_duration1 = 6)
  (h5 : day2_duration2 = 3)
  : ∃ (day2_speed : ℝ), 
    total_distance = distance day1 + distance day3 + day2_speed * day2_duration1 + (day2_speed / 2) * day2_duration2 ∧ 
    day2_speed = 6 := by
  sorry

end second_day_speed_l188_18863


namespace rectangle_area_l188_18840

theorem rectangle_area (square_area : ℝ) (rectangle_length_multiplier : ℝ) : 
  square_area = 36 → 
  rectangle_length_multiplier = 3 → 
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_multiplier * rectangle_width
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l188_18840


namespace polynomial_product_no_x4_x3_terms_l188_18831

theorem polynomial_product_no_x4_x3_terms :
  let P (x : ℝ) := 2 * x^3 - 5 * x^2 + 7 * x - 8
  let Q (x : ℝ) := a * x^2 + b * x + 11
  (∀ x, (P x) * (Q x) = 8 * x^5 - 17 * x^2 - 3 * x - 88) →
  a = 4 ∧ b = 10 := by
sorry

end polynomial_product_no_x4_x3_terms_l188_18831


namespace boot_pairing_l188_18834

theorem boot_pairing (total_boots : ℕ) (left_boots right_boots : ℕ) (size_count : ℕ) :
  total_boots = 600 →
  left_boots = 300 →
  right_boots = 300 →
  size_count = 3 →
  total_boots = left_boots + right_boots →
  ∃ (valid_pairs : ℕ), valid_pairs ≥ 100 ∧ 
    ∃ (size_41 size_42 size_43 : ℕ),
      size_41 + size_42 + size_43 = total_boots ∧
      size_41 = size_42 ∧ size_42 = size_43 ∧
      (∀ (size : ℕ), size ∈ [size_41, size_42, size_43] → 
        ∃ (left_count right_count : ℕ), 
          left_count + right_count = size ∧
          left_count ≤ left_boots ∧
          right_count ≤ right_boots) :=
by sorry


end boot_pairing_l188_18834


namespace both_pass_through_origin_l188_18804

/-- Parabola passing through (0,1) -/
def passes_through_origin (f : ℝ → ℝ) : Prop :=
  f 0 = 1

/-- First parabola -/
def f₁ (x : ℝ) : ℝ := -x^2 + 1

/-- Second parabola -/
def f₂ (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: Both parabolas pass through (0,1) -/
theorem both_pass_through_origin :
  passes_through_origin f₁ ∧ passes_through_origin f₂ := by
  sorry

end both_pass_through_origin_l188_18804


namespace a_range_l188_18862

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + x^2 - a*x

theorem a_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) →
  (∀ x ∈ Set.Ioc 0 1, f a x ≤ 1/2 * (3*x^2 + 1/x^2 - 6*x)) →
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
by sorry

end a_range_l188_18862


namespace pool_capacity_l188_18823

theorem pool_capacity (C : ℝ) 
  (h1 : 0.45 * C + 300 = 0.75 * C) : C = 1000 := by
  sorry

end pool_capacity_l188_18823


namespace not_divisible_by_product_l188_18838

theorem not_divisible_by_product (a₁ a₂ b₁ b₂ : ℕ) 
  (h1 : 1 < b₁) (h2 : b₁ < a₁) (h3 : 1 < b₂) (h4 : b₂ < a₂) 
  (h5 : b₁ ∣ a₁) (h6 : b₂ ∣ a₂) : 
  ¬(a₁ * a₂ ∣ a₁ * b₁ + a₂ * b₂ - 1) := by
  sorry

end not_divisible_by_product_l188_18838


namespace system_solution_ratio_l188_18864

/-- Given a system of linear equations with a parameter k, 
    prove that for a specific value of k, the ratio yz/x^2 is constant --/
theorem system_solution_ratio (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  let k : ℝ := 55 / 26
  x + 2 * k * y + 4 * z = 0 ∧
  4 * x + 2 * k * y - 3 * z = 0 ∧
  3 * x + 5 * y - 4 * z = 0 →
  ∃ (c : ℝ), y * z / (x^2) = c :=
by sorry

end system_solution_ratio_l188_18864


namespace opposite_number_theorem_l188_18822

theorem opposite_number_theorem (a : ℝ) : (-(-a) = -2) → a = 2 := by
  sorry

end opposite_number_theorem_l188_18822


namespace tooth_extraction_cost_l188_18844

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def total_fillings : ℕ := 2
def total_bill_factor : ℕ := 5

theorem tooth_extraction_cost :
  let total_bill := filling_cost * total_bill_factor
  let cleaning_and_fillings_cost := cleaning_cost + (filling_cost * total_fillings)
  total_bill - cleaning_and_fillings_cost = 290 :=
by sorry

end tooth_extraction_cost_l188_18844


namespace fish_tank_balls_count_total_balls_in_tank_l188_18885

theorem fish_tank_balls_count : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_goldfish, num_platyfish, red_balls_per_goldfish, white_balls_per_platyfish =>
    num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish

theorem total_balls_in_tank : fish_tank_balls_count 3 10 10 5 = 80 := by
  sorry

end fish_tank_balls_count_total_balls_in_tank_l188_18885


namespace greatest_mean_Y_Z_l188_18815

-- Define the piles of rocks
variable (X Y Z : Set ℝ)

-- Define the mean weight functions
variable (mean : Set ℝ → ℝ)

-- Define the conditions
variable (h1 : mean X = 30)
variable (h2 : mean Y = 70)
variable (h3 : mean (X ∪ Y) = 50)
variable (h4 : mean (X ∪ Z) = 40)

-- Define the function to calculate the mean of Y and Z
def mean_Y_Z : ℝ := mean (Y ∪ Z)

-- Theorem statement
theorem greatest_mean_Y_Z : 
  ∀ n : ℕ, mean_Y_Z ≤ 70 ∧ (mean_Y_Z > 69 → mean_Y_Z = 70) :=
sorry

end greatest_mean_Y_Z_l188_18815


namespace rectangle_dimension_relationship_l188_18825

/-- Given a rectangle with perimeter 20m, prove that the relationship between its length y and width x is y = -x + 10 -/
theorem rectangle_dimension_relationship (x y : ℝ) : 
  (2 * (x + y) = 20) → (y = -x + 10) := by
  sorry

end rectangle_dimension_relationship_l188_18825


namespace geometric_arithmetic_sequences_l188_18826

/-- Geometric sequence with a₁ = 2 and a₄ = 16 -/
def geometric_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2 * (2 ^ (n - 1))

/-- Arithmetic sequence with b₃ = a₃ and b₅ = a₅ -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  12 * n - 28

/-- Sum of first n terms of the arithmetic sequence -/
def arithmetic_sum (n : ℕ) : ℝ :=
  6 * n^2 - 22 * n

theorem geometric_arithmetic_sequences :
  (∀ n, geometric_sequence n = 2^n) ∧
  (∀ n, arithmetic_sequence n = 12 * n - 28) ∧
  (∀ n, arithmetic_sum n = 6 * n^2 - 22 * n) ∧
  geometric_sequence 1 = 2 ∧
  geometric_sequence 4 = 16 ∧
  arithmetic_sequence 3 = geometric_sequence 3 ∧
  arithmetic_sequence 5 = geometric_sequence 5 :=
by sorry

end geometric_arithmetic_sequences_l188_18826


namespace coloring_books_per_shelf_l188_18892

theorem coloring_books_per_shelf 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (num_shelves : ℕ) 
  (h1 : initial_stock = 120)
  (h2 : books_sold = 39)
  (h3 : num_shelves = 9)
  (h4 : num_shelves > 0) :
  (initial_stock - books_sold) / num_shelves = 9 := by
  sorry

end coloring_books_per_shelf_l188_18892


namespace girls_not_attending_college_percentage_l188_18866

theorem girls_not_attending_college_percentage
  (total_boys : ℕ)
  (total_girls : ℕ)
  (boys_not_attending_percentage : ℚ)
  (total_attending_percentage : ℚ)
  (h1 : total_boys = 300)
  (h2 : total_girls = 240)
  (h3 : boys_not_attending_percentage = 30 / 100)
  (h4 : total_attending_percentage = 70 / 100)
  : (↑(total_girls - (total_boys + total_girls) * total_attending_percentage + total_boys * boys_not_attending_percentage) / total_girls : ℚ) = 30 / 100 := by
  sorry

end girls_not_attending_college_percentage_l188_18866


namespace sum_of_functions_l188_18828

theorem sum_of_functions (x : ℝ) (hx : x ≠ 2) :
  let f : ℝ → ℝ := λ x => x^2 - 1/(x-2)
  let g : ℝ → ℝ := λ x => 1/(x-2) + 1
  f x + g x = x^2 + 1 :=
by sorry

end sum_of_functions_l188_18828


namespace cookies_eaten_vs_given_l188_18837

theorem cookies_eaten_vs_given (initial_cookies : ℕ) (eaten_cookies : ℕ) (given_cookies : ℕ) 
  (h1 : initial_cookies = 17) 
  (h2 : eaten_cookies = 14) 
  (h3 : given_cookies = 13) :
  eaten_cookies - given_cookies = 1 := by
  sorry

end cookies_eaten_vs_given_l188_18837


namespace acute_angle_range_l188_18852

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop := dot_product v w > 0

theorem acute_angle_range (x : ℝ) :
  is_acute_angle a (b x) ↔ x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi 2 := by sorry

end acute_angle_range_l188_18852


namespace expansion_coefficient_l188_18889

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of binomial coefficients for a given n -/
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

/-- The sum of all coefficients in the expansion of (x + 3/√x)^n when x = 1 -/
def sum_all_coefficients (n : ℕ) : ℕ := 4^n

/-- The coefficient of x^3 in the expansion of (x + 3/√x)^n -/
def coefficient_x3 (n : ℕ) : ℕ := binomial n 2 * 3^2

theorem expansion_coefficient :
  ∃ n : ℕ,
    sum_all_coefficients n / sum_binomial_coefficients n = 64 ∧
    coefficient_x3 n = 135 := by sorry

end expansion_coefficient_l188_18889


namespace closest_integer_to_cube_root_150_l188_18827

theorem closest_integer_to_cube_root_150 : 
  ∀ n : ℤ, |n - (150 : ℝ)^(1/3)| ≥ |5 - (150 : ℝ)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_150_l188_18827


namespace cubic_three_zeros_l188_18887

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem cubic_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ 
  a < -3 :=
sorry

end cubic_three_zeros_l188_18887


namespace equation_solutions_l188_18883

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (2 + Real.sqrt 7) / 3 ∧ x₂ = (2 - Real.sqrt 7) / 3 ∧
    3 * x₁^2 - 1 = 4 * x₁ ∧ 3 * x₂^2 - 1 = 4 * x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4 ∧ x₂ = 1 ∧
    (x₁ + 4)^2 = 5 * (x₁ + 4) ∧ (x₂ + 4)^2 = 5 * (x₂ + 4)) :=
by sorry

end equation_solutions_l188_18883


namespace corn_donation_l188_18872

def total_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

theorem corn_donation :
  let total_ears := total_bushels * ears_per_bushel
  let given_away_ears := total_ears - remaining_ears
  given_away_ears / ears_per_bushel = 24 :=
by sorry

end corn_donation_l188_18872


namespace sqrt_sum_fractions_l188_18869

theorem sqrt_sum_fractions : Real.sqrt (1 / 4 + 1 / 25) = Real.sqrt 29 / 10 := by
  sorry

end sqrt_sum_fractions_l188_18869


namespace f_of_2_equals_1_l188_18877

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^3 - (x - 1) + 1

-- State the theorem
theorem f_of_2_equals_1 : f 2 = 1 := by
  sorry

end f_of_2_equals_1_l188_18877


namespace arithmetic_calculation_l188_18878

theorem arithmetic_calculation : 8 / 4 + 5 * 2^2 - (3 + 7) = 12 := by
  sorry

end arithmetic_calculation_l188_18878


namespace difference_of_squares_division_l188_18824

theorem difference_of_squares_division : (121^2 - 112^2) / 9 = 233 := by sorry

end difference_of_squares_division_l188_18824


namespace blocks_with_one_face_painted_10_2_l188_18850

/-- Represents a cube made of smaller blocks -/
structure BlockCube where
  largeSideLength : ℕ
  smallSideLength : ℕ
  
/-- Calculates the number of blocks with only one face painted -/
def BlockCube.blocksWithOneFacePainted (cube : BlockCube) : ℕ :=
  let blocksPerEdge := cube.largeSideLength / cube.smallSideLength
  let surfaceBlocks := 6 * blocksPerEdge * blocksPerEdge
  let edgeBlocks := 12 * blocksPerEdge - 24
  surfaceBlocks - edgeBlocks - 8

theorem blocks_with_one_face_painted_10_2 :
  (BlockCube.blocksWithOneFacePainted { largeSideLength := 10, smallSideLength := 2 }) = 54 := by
  sorry

end blocks_with_one_face_painted_10_2_l188_18850


namespace inequality_solution_set_l188_18820

theorem inequality_solution_set (a : ℝ) : 
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 2 ↔ |a * x + 2| < 6) → a = -4 := by
  sorry

end inequality_solution_set_l188_18820


namespace star_3_7_equals_16_l188_18880

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem star_3_7_equals_16 : star 3 7 = 16 := by
  sorry

end star_3_7_equals_16_l188_18880


namespace optimal_price_and_units_l188_18839

-- Define the problem parameters
def initial_cost : ℝ := 40
def initial_price : ℝ := 50
def initial_units : ℝ := 500
def price_range_low : ℝ := 50
def price_range_high : ℝ := 70
def target_profit : ℝ := 8000

-- Define the price-demand relationship
def units_sold (price : ℝ) : ℝ :=
  initial_units - 10 * (price - initial_price)

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - initial_cost) * units_sold price

-- State the theorem
theorem optimal_price_and_units :
  ∃ (price : ℝ) (units : ℝ),
    price_range_low ≤ price ∧
    price ≤ price_range_high ∧
    units = units_sold price ∧
    profit price = target_profit ∧
    price = 60 ∧
    units = 400 := by
  sorry

end optimal_price_and_units_l188_18839


namespace mod7_mul_table_mod10_mul_2_mod10_mul_5_mod9_mul_3_l188_18809

-- Define the modular multiplication function
def modMul (a b m : Nat) : Nat :=
  (a * b) % m

-- Theorem for modulo 7 multiplication table
theorem mod7_mul_table (a b : Fin 7) : 
  modMul a b 7 = 
    match a, b with
    | 0, _ => 0
    | _, 0 => 0
    | 1, x => x
    | x, 1 => x
    | 2, 2 => 4
    | 2, 3 => 6
    | 2, 4 => 1
    | 2, 5 => 3
    | 2, 6 => 5
    | 3, 2 => 6
    | 3, 3 => 2
    | 3, 4 => 5
    | 3, 5 => 1
    | 3, 6 => 4
    | 4, 2 => 1
    | 4, 3 => 5
    | 4, 4 => 2
    | 4, 5 => 6
    | 4, 6 => 3
    | 5, 2 => 3
    | 5, 3 => 1
    | 5, 4 => 6
    | 5, 5 => 4
    | 5, 6 => 2
    | 6, 2 => 5
    | 6, 3 => 4
    | 6, 4 => 3
    | 6, 5 => 2
    | 6, 6 => 1
    | _, _ => 0  -- This case should never be reached
  := by sorry

-- Theorem for modulo 10 multiplication by 2
theorem mod10_mul_2 (a : Fin 10) : 
  modMul 2 a 10 = 
    match a with
    | 0 => 0
    | 1 => 2
    | 2 => 4
    | 3 => 6
    | 4 => 8
    | 5 => 0
    | 6 => 2
    | 7 => 4
    | 8 => 6
    | 9 => 8
  := by sorry

-- Theorem for modulo 10 multiplication by 5
theorem mod10_mul_5 (a : Fin 10) : 
  modMul 5 a 10 = 
    match a with
    | 0 => 0
    | 1 => 5
    | 2 => 0
    | 3 => 5
    | 4 => 0
    | 5 => 5
    | 6 => 0
    | 7 => 5
    | 8 => 0
    | 9 => 5
  := by sorry

-- Theorem for modulo 9 multiplication by 3
theorem mod9_mul_3 (a : Fin 9) : 
  modMul 3 a 9 = 
    match a with
    | 0 => 0
    | 1 => 3
    | 2 => 6
    | 3 => 0
    | 4 => 3
    | 5 => 6
    | 6 => 0
    | 7 => 3
    | 8 => 6
  := by sorry

end mod7_mul_table_mod10_mul_2_mod10_mul_5_mod9_mul_3_l188_18809


namespace round_trip_completion_l188_18894

/-- Represents a round trip with equal outbound and inbound journeys -/
structure RoundTrip where
  total_distance : ℝ
  outbound_distance : ℝ
  inbound_distance : ℝ
  equal_journeys : outbound_distance = inbound_distance
  total_is_sum : total_distance = outbound_distance + inbound_distance

/-- Theorem stating that completing the outbound journey and 20% of the inbound journey
    results in completing 60% of the total trip -/
theorem round_trip_completion (trip : RoundTrip) :
  trip.outbound_distance + 0.2 * trip.inbound_distance = 0.6 * trip.total_distance := by
  sorry

end round_trip_completion_l188_18894


namespace gcd_2100_2091_l188_18882

theorem gcd_2100_2091 : Nat.gcd (2^2100 - 1) (2^2091 - 1) = 2^9 - 1 := by sorry

end gcd_2100_2091_l188_18882


namespace lemon_permutations_l188_18856

theorem lemon_permutations :
  (Finset.range 5).card.factorial = 120 := by
  sorry

end lemon_permutations_l188_18856


namespace find_a_l188_18847

def U (a : ℝ) : Set ℝ := {3, 7, a^2 - 2*a - 3}

def A (a : ℝ) : Set ℝ := {7, |a - 7|}

theorem find_a : ∃ a : ℝ, (U a \ A a = {5}) ∧ (A a ⊆ U a) := by
  sorry

end find_a_l188_18847


namespace regression_increase_l188_18833

/-- Linear regression equation for annual food expenditure with respect to annual income -/
def regression_equation (x : ℝ) : ℝ := 0.254 * x + 0.321

/-- Theorem stating that the increase in the regression equation's output for a 1 unit increase in input is 0.254 -/
theorem regression_increase : ∀ x : ℝ, regression_equation (x + 1) - regression_equation x = 0.254 := by
  sorry

end regression_increase_l188_18833


namespace negation_of_implication_l188_18860

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end negation_of_implication_l188_18860


namespace cone_lateral_surface_area_l188_18816

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- base radius
  let h : ℝ := 4  -- height
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  let S : ℝ := π * r * l  -- lateral surface area formula
  S = 15 * π :=
by sorry

end cone_lateral_surface_area_l188_18816


namespace intersection_condition_distance_product_condition_l188_18846

-- Define the curve C in Cartesian coordinates
def C (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Define the line l
def l (m t : ℝ) : ℝ × ℝ := (m + 3*t, 4*t)

-- Define the intersection condition
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ C (l m t₁).1 (l m t₁).2 ∧ C (l m t₂).1 (l m t₂).2

-- Define the distance product condition
def distance_product_is_one (m : ℝ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ C (l m t₁).1 (l m t₁).2 ∧ C (l m t₂).1 (l m t₂).2 ∧
    (m^2 + (3*t₁)^2 + (4*t₁)^2) * (m^2 + (3*t₂)^2 + (4*t₂)^2) = 1

-- State the theorems
theorem intersection_condition (m : ℝ) :
  intersects_at_two_points m ↔ -1/4 < m ∧ m < 9/4 :=
sorry

theorem distance_product_condition :
  ∃ m, distance_product_is_one m ∧ m = 1 :=
sorry

end intersection_condition_distance_product_condition_l188_18846


namespace inclination_angle_theorem_l188_18812

-- Define the line equation
def line_equation (x y α : ℝ) : Prop := x * Real.cos α + Real.sqrt 3 * y + 2 = 0

-- Define the range of cos α
def cos_α_range (α : ℝ) : Prop := -1 ≤ Real.cos α ∧ Real.cos α ≤ 1

-- Define the range of θ
def θ_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < Real.pi

-- Define the inclination angle range
def inclination_angle_range (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

-- Theorem statement
theorem inclination_angle_theorem (x y α θ : ℝ) :
  line_equation x y α → cos_α_range α → θ_range θ →
  inclination_angle_range θ := by sorry

end inclination_angle_theorem_l188_18812


namespace nine_possible_scores_l188_18857

/-- The number of baskets scored by the player -/
def total_baskets : ℕ := 8

/-- The possible point values for each basket -/
inductive BasketValue : Type
| one : BasketValue
| three : BasketValue

/-- A function to calculate the total score given a list of basket values -/
def total_score (baskets : List BasketValue) : ℕ :=
  baskets.foldl (fun acc b => acc + match b with
    | BasketValue.one => 1
    | BasketValue.three => 3) 0

/-- The theorem to be proved -/
theorem nine_possible_scores :
  ∃! (scores : Finset ℕ), 
    (∀ (score : ℕ), score ∈ scores ↔ 
      ∃ (baskets : List BasketValue), 
        baskets.length = total_baskets ∧ total_score baskets = score) ∧
    scores.card = 9 := by sorry

end nine_possible_scores_l188_18857


namespace hyperbola_equation_l188_18806

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (k : ℝ), k * a = b ∧ k * 2 = Real.sqrt 3) →  -- asymptote condition
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = Real.sqrt 7) →  -- focus and directrix condition
  a^2 = 4 ∧ b^2 = 3 :=
by sorry

end hyperbola_equation_l188_18806


namespace min_value_expression_l188_18848

theorem min_value_expression (x : ℝ) :
  x ≥ 0 →
  (1 + x^2) / (1 + x) ≥ -2 + 2 * Real.sqrt 2 ∧
  ∃ y : ℝ, y ≥ 0 ∧ (1 + y^2) / (1 + y) = -2 + 2 * Real.sqrt 2 :=
by sorry

end min_value_expression_l188_18848


namespace sarah_cupcake_count_l188_18814

def is_valid_cupcake_count (c : ℕ) : Prop :=
  ∃ (k : ℕ), 
    c + k = 6 ∧ 
    (90 * c + 40 * k) % 100 = 0

theorem sarah_cupcake_count :
  ∀ c : ℕ, is_valid_cupcake_count c → c = 4 ∨ c = 6 := by
  sorry

end sarah_cupcake_count_l188_18814


namespace smallest_S_value_l188_18899

/-- Represents a standard 6-sided die -/
def Die := Fin 6

/-- The number of dice rolled -/
def n : ℕ := 342

/-- The sum we're comparing against -/
def target_sum : ℕ := 2052

/-- Function to calculate the probability of obtaining a specific sum -/
noncomputable def prob_of_sum (sum : ℕ) : ℝ := sorry

/-- The smallest sum S that has the same probability as the target sum -/
def S : ℕ := 342

theorem smallest_S_value :
  (prob_of_sum target_sum > 0) ∧ 
  (∀ s : ℕ, s < S → prob_of_sum s ≠ prob_of_sum target_sum) ∧
  (prob_of_sum S = prob_of_sum target_sum) := by sorry

end smallest_S_value_l188_18899


namespace line_through_point_intersecting_circle_l188_18876

/-- A line passing through a point and intersecting a circle -/
theorem line_through_point_intersecting_circle 
  (M : ℝ × ℝ) 
  (A B : ℝ × ℝ) 
  (h_M : M = (1, 0))
  (h_circle : ∀ P : ℝ × ℝ, P ∈ {P | P.1^2 + P.2^2 = 5} ↔ A ∈ {P | P.1^2 + P.2^2 = 5} ∧ B ∈ {P | P.1^2 + P.2^2 = 5})
  (h_first_quadrant : A.1 > 0 ∧ A.2 > 0)
  (h_vector_relation : B - M = 2 • (A - M))
  : ∃ (m c : ℝ), m = 1 ∧ c = -1 ∧ ∀ P : ℝ × ℝ, P ∈ {P | P.1 = m * P.2 + c} ↔ (A ∈ {P | P.1 = m * P.2 + c} ∧ B ∈ {P | P.1 = m * P.2 + c} ∧ M ∈ {P | P.1 = m * P.2 + c}) :=
sorry

end line_through_point_intersecting_circle_l188_18876


namespace peter_notebooks_l188_18836

def green_notebooks : ℕ := 2
def black_notebooks : ℕ := 1
def pink_notebooks : ℕ := 1

def total_notebooks : ℕ := green_notebooks + black_notebooks + pink_notebooks

theorem peter_notebooks : total_notebooks = 4 := by sorry

end peter_notebooks_l188_18836


namespace second_hand_revolution_time_l188_18807

/-- The time in seconds for a second hand to complete one revolution -/
def revolution_time_seconds : ℕ := 60

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The time in minutes for a second hand to complete one revolution -/
def revolution_time_minutes : ℚ := revolution_time_seconds / seconds_per_minute

theorem second_hand_revolution_time :
  revolution_time_seconds = 60 ∧ revolution_time_minutes = 1 := by sorry

end second_hand_revolution_time_l188_18807


namespace probability_of_selection_for_six_choose_two_l188_18854

/-- The probability of choosing a specific person as a representative -/
def probability_of_selection (n : ℕ) (k : ℕ) : ℚ :=
  (n - 1).choose (k - 1) / n.choose k

/-- The problem statement -/
theorem probability_of_selection_for_six_choose_two :
  probability_of_selection 6 2 = 1 / 3 := by
  sorry

end probability_of_selection_for_six_choose_two_l188_18854


namespace triangle_angle_sine_relation_l188_18867

theorem triangle_angle_sine_relation (A B : Real) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  A > B ↔ Real.sin A > Real.sin B := by sorry

end triangle_angle_sine_relation_l188_18867


namespace min_domain_for_inverse_l188_18845

-- Define the function g
def g (x : ℝ) : ℝ := (x - 3)^2 + 4

-- State the theorem
theorem min_domain_for_inverse :
  ∃ (d : ℝ), d = 3 ∧ 
  (∀ (d' : ℝ), (∀ (x y : ℝ), x ≥ d' ∧ y ≥ d' ∧ x ≠ y → g x ≠ g y) → d' ≥ d) ∧
  (∀ (x y : ℝ), x ≥ d ∧ y ≥ d ∧ x ≠ y → g x ≠ g y) :=
sorry

end min_domain_for_inverse_l188_18845


namespace shopping_cost_calculation_l188_18875

/-- Calculates the total cost of a shopping trip, including discounts and sales tax -/
theorem shopping_cost_calculation 
  (tshirt_price sweater_price jacket_price : ℚ)
  (jacket_discount sales_tax : ℚ)
  (tshirt_quantity sweater_quantity jacket_quantity : ℕ)
  (h1 : tshirt_price = 8)
  (h2 : sweater_price = 18)
  (h3 : jacket_price = 80)
  (h4 : jacket_discount = 1/10)
  (h5 : sales_tax = 1/20)
  (h6 : tshirt_quantity = 6)
  (h7 : sweater_quantity = 4)
  (h8 : jacket_quantity = 5) :
  let tshirt_cost := tshirt_quantity * tshirt_price
  let sweater_cost := sweater_quantity * sweater_price
  let jacket_cost := jacket_quantity * jacket_price * (1 - jacket_discount)
  let subtotal := tshirt_cost + sweater_cost + jacket_cost
  let total := subtotal * (1 + sales_tax)
  total = 504 := by sorry

end shopping_cost_calculation_l188_18875


namespace lemon_orange_drink_scaling_l188_18861

/-- Represents the recipe for lemon-orange drink -/
structure DrinkRecipe where
  gallons : ℚ
  lemons : ℚ
  oranges : ℚ

/-- Calculates the number of fruits needed for a given number of gallons -/
def scaledRecipe (base : DrinkRecipe) (newGallons : ℚ) : DrinkRecipe :=
  { gallons := newGallons,
    lemons := (base.lemons / base.gallons) * newGallons,
    oranges := (base.oranges / base.gallons) * newGallons }

theorem lemon_orange_drink_scaling :
  let baseRecipe : DrinkRecipe := { gallons := 40, lemons := 30, oranges := 20 }
  let scaledRecipe := scaledRecipe baseRecipe 100
  scaledRecipe.lemons = 75 ∧ scaledRecipe.oranges = 50 := by sorry

end lemon_orange_drink_scaling_l188_18861


namespace basketball_time_calculation_l188_18810

def football_time : ℕ := 60
def total_time_hours : ℕ := 2

theorem basketball_time_calculation :
  football_time + (total_time_hours * 60 - football_time) = 60 := by
  sorry

end basketball_time_calculation_l188_18810


namespace medians_form_right_triangle_l188_18873

/-- Given a triangle ABC with sides a, b, c and corresponding medians m_a, m_b, m_c,
    if m_a ⊥ m_b, then m_a^2 + m_b^2 = m_c^2 -/
theorem medians_form_right_triangle (a b c m_a m_b m_c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_medians : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_perp : m_a * m_b = 0) : 
  m_a^2 + m_b^2 = m_c^2 := by
sorry

end medians_form_right_triangle_l188_18873


namespace arithmetic_operations_l188_18843

theorem arithmetic_operations : 
  (-3 : ℤ) + 2 = -1 ∧ (-3 : ℤ) * 2 = -6 := by sorry

end arithmetic_operations_l188_18843


namespace polynomial_evaluation_l188_18874

theorem polynomial_evaluation : 7^4 - 4 * 7^3 + 6 * 7^2 - 5 * 7 + 3 = 1553 := by
  sorry

end polynomial_evaluation_l188_18874


namespace marigold_sale_problem_l188_18808

theorem marigold_sale_problem (day1 day2 day3 total : ℕ) : 
  day1 = 14 →
  day3 = 2 * day2 →
  total = day1 + day2 + day3 →
  total = 89 →
  day2 = 25 := by
sorry

end marigold_sale_problem_l188_18808


namespace max_profit_theorem_l188_18803

def profit_A (x : ℕ) : ℚ := 5.06 * x - 0.15 * x^2
def profit_B (x : ℕ) : ℚ := 2 * x

theorem max_profit_theorem :
  ∃ (x : ℕ), x ≤ 15 ∧ 
  (∀ (y : ℕ), y ≤ 15 → 
    profit_A x + profit_B (15 - x) ≥ profit_A y + profit_B (15 - y)) ∧
  profit_A x + profit_B (15 - x) = 45.6 :=
sorry

end max_profit_theorem_l188_18803


namespace graduation_ceremony_arrangements_l188_18865

/-- The number of events in the graduation ceremony program -/
def total_events : ℕ := 6

/-- The number of positions event A can be placed in -/
def a_positions : ℕ := 3

/-- The number of events that must be scheduled together -/
def together_events : ℕ := 2

/-- The number of possible arrangements for the graduation ceremony program -/
def possible_arrangements : ℕ := 120

/-- Theorem stating that the number of possible arrangements is correct -/
theorem graduation_ceremony_arrangements :
  (total_events = 6) →
  (a_positions = 3) →
  (together_events = 2) →
  (possible_arrangements = 120) := by
  sorry

end graduation_ceremony_arrangements_l188_18865


namespace f_decreasing_inequality_solution_set_l188_18890

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_prop1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_prop2 : ∀ (x : ℝ), 0 < x → x < 1 → f x > 0
axiom f_prop3 : f (1/2) = 1

-- Theorem 1: f is decreasing on its domain
theorem f_decreasing : ∀ (x₁ x₂ : ℝ), 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Theorem 2: Solution set of the inequality
theorem inequality_solution_set : 
  {x : ℝ | f (x - 3) > f (1/x) - 2} = Set.Ioo 3 4 := by
  sorry

end

end f_decreasing_inequality_solution_set_l188_18890


namespace f_increasing_iff_a_range_f_inequality_when_a_zero_l188_18849

noncomputable section

def f (a x : ℝ) : ℝ := (x - a) * Real.log x - x

theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) ↔ a ∈ Set.Iic (-1 / Real.exp 1) :=
sorry

theorem f_inequality_when_a_zero (x : ℝ) (hx : x > 0) :
  f 0 x ≥ x * (Real.exp (-x) - 1) - 2 / Real.exp 1 :=
sorry

end f_increasing_iff_a_range_f_inequality_when_a_zero_l188_18849


namespace range_of_a_l188_18897

theorem range_of_a (a : ℝ) : (∀ x > 0, x^2 + a*x + 1 ≥ 0) → a ≥ -2 := by
  sorry

end range_of_a_l188_18897


namespace simplify_expression_l188_18888

theorem simplify_expression (a b : ℝ) : 
  (50*a + 130*b) + (21*a + 64*b) - (30*a + 115*b) - 2*(10*a - 25*b) = 21*a + 129*b := by
  sorry

end simplify_expression_l188_18888


namespace original_population_l188_18858

def population_change (p : ℕ) : ℝ :=
  0.85 * (p + 1500 : ℝ) - p

theorem original_population : 
  ∃ p : ℕ, population_change p = -50 ∧ p = 8833 := by
  sorry

end original_population_l188_18858


namespace marcos_strawberries_weight_l188_18813

/-- Given the total weight of strawberries collected by Marco and his dad,
    the weight of strawberries lost by Marco's dad, and the weight of
    Marco's dad's remaining strawberries, prove that Marco's strawberries
    weigh 12 pounds. -/
theorem marcos_strawberries_weight
  (total_weight : ℕ)
  (dads_lost_weight : ℕ)
  (dads_remaining_weight : ℕ)
  (h1 : total_weight = 36)
  (h2 : dads_lost_weight = 8)
  (h3 : dads_remaining_weight = 16) :
  total_weight - (dads_remaining_weight + dads_lost_weight) = 12 :=
by sorry

end marcos_strawberries_weight_l188_18813


namespace ninas_running_drill_l188_18821

/-- Nina's running drill problem -/
theorem ninas_running_drill 
  (initial_run : ℝ) 
  (total_distance : ℝ) 
  (h1 : initial_run = 0.08333333333333333)
  (h2 : total_distance = 0.8333333333333334) :
  total_distance - 2 * initial_run = 0.6666666666666667 := by
  sorry

end ninas_running_drill_l188_18821


namespace quadratic_roots_property_l188_18898

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + 3*x₁ - 1 = 0) → 
  (x₂^2 + 3*x₂ - 1 = 0) → 
  (x₁^2 - 3*x₂ + 1 = 11) := by sorry

end quadratic_roots_property_l188_18898


namespace intersection_product_l188_18818

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4*x

def C₂ (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 3 * Real.sqrt 3 = 0

-- Define point A
def A : ℝ × ℝ := (3, 0)

-- Define the intersection points P and Q
def isIntersection (p : ℝ × ℝ) : Prop :=
  C₁ p.1 p.2 ∧ C₂ p.1 p.2

-- State the theorem
theorem intersection_product :
  ∃ (P Q : ℝ × ℝ), isIntersection P ∧ isIntersection Q ∧ P ≠ Q ∧
    (P.1 - A.1)^2 + (P.2 - A.2)^2 * ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) = 3^2 :=
sorry

end intersection_product_l188_18818


namespace right_triangle_7_24_25_l188_18896

theorem right_triangle_7_24_25 (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  a^2 + b^2 = c^2 :=
by sorry

end right_triangle_7_24_25_l188_18896
