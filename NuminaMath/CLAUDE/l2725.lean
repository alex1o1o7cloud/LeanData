import Mathlib

namespace NUMINAMATH_CALUDE_lemon_permutations_l2725_272541

theorem lemon_permutations :
  (Finset.range 5).card.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_lemon_permutations_l2725_272541


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2725_272513

theorem inequality_solution_set (a : ℝ) : 
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 2 ↔ |a * x + 2| < 6) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2725_272513


namespace NUMINAMATH_CALUDE_probability_of_selection_for_six_choose_two_l2725_272581

/-- The probability of choosing a specific person as a representative -/
def probability_of_selection (n : ℕ) (k : ℕ) : ℚ :=
  (n - 1).choose (k - 1) / n.choose k

/-- The problem statement -/
theorem probability_of_selection_for_six_choose_two :
  probability_of_selection 6 2 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_for_six_choose_two_l2725_272581


namespace NUMINAMATH_CALUDE_distance_satisfies_conditions_l2725_272576

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

end NUMINAMATH_CALUDE_distance_satisfies_conditions_l2725_272576


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2725_272544

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x : ℚ, 3 * x^2 - 7 * x + m = 0) → m = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2725_272544


namespace NUMINAMATH_CALUDE_square_less_than_triple_l2725_272583

theorem square_less_than_triple (n : ℤ) : n^2 < 3*n ↔ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l2725_272583


namespace NUMINAMATH_CALUDE_largest_number_l2725_272508

def hcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem largest_number (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (hcf_cond : hcf a b c = 23)
  (lcm_cond : lcm a b c = 23 * 13 * 19 * 17) :
  max a (max b c) = 437 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2725_272508


namespace NUMINAMATH_CALUDE_cubic_three_zeros_l2725_272570

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem cubic_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ 
  a < -3 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_l2725_272570


namespace NUMINAMATH_CALUDE_acute_angle_range_l2725_272557

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop := dot_product v w > 0

theorem acute_angle_range (x : ℝ) :
  is_acute_angle a (b x) ↔ x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_acute_angle_range_l2725_272557


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_nine_l2725_272587

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_nine :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 9 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 9 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_nine_l2725_272587


namespace NUMINAMATH_CALUDE_negation_of_implication_l2725_272540

-- Define a triangle type
structure Triangle where
  -- Add any necessary fields here
  mk :: -- Constructor

-- Define properties for triangles
def isEquilateral (t : Triangle) : Prop := sorry
def interiorAnglesEqual (t : Triangle) : Prop := sorry

-- State the theorem
theorem negation_of_implication :
  (¬(∀ t : Triangle, isEquilateral t → interiorAnglesEqual t)) ↔
  (∀ t : Triangle, ¬isEquilateral t → ¬interiorAnglesEqual t) :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2725_272540


namespace NUMINAMATH_CALUDE_temperature_at_14_minutes_l2725_272553

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

end NUMINAMATH_CALUDE_temperature_at_14_minutes_l2725_272553


namespace NUMINAMATH_CALUDE_sufficient_condition_quadratic_inequality_l2725_272548

theorem sufficient_condition_quadratic_inequality (m : ℝ) :
  (m ≥ 2) →
  (∀ x : ℝ, x^2 - 2*x + m ≥ 0) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - 2*x + m ≥ 0) → m ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_quadratic_inequality_l2725_272548


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2725_272585

-- Define the condition p
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- Define the condition q
def q (x a : ℝ) : Prop := x ≤ a

-- State the theorem
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2725_272585


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2725_272519

theorem sqrt_sum_fractions : Real.sqrt (1 / 4 + 1 / 25) = Real.sqrt 29 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2725_272519


namespace NUMINAMATH_CALUDE_task_completion_time_l2725_272594

/-- Given that m men can complete a task in d days, 
    prove that m + r² men will complete the same task in md / (m + r²) days -/
theorem task_completion_time 
  (m d r : ℕ) -- m, d, and r are natural numbers
  (m_pos : 0 < m) -- m is positive
  (d_pos : 0 < d) -- d is positive
  (total_work : ℕ := m * d) -- total work in man-days
  : (↑total_work : ℚ) / (m + r^2 : ℚ) = (↑m * ↑d : ℚ) / (↑m + ↑r^2 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_task_completion_time_l2725_272594


namespace NUMINAMATH_CALUDE_distance_city_A_to_B_distance_city_A_to_B_value_l2725_272536

/-- Proves that the distance between city A and city B is 450 km given the problem conditions -/
theorem distance_city_A_to_B : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (time_eddy : ℝ) (time_freddy : ℝ) (speed_ratio : ℝ) (known_distance : ℝ) =>
    time_eddy = 3 ∧ 
    time_freddy = 4 ∧ 
    speed_ratio = 2 ∧ 
    known_distance = 300 →
    ∃ (distance_AB distance_AC : ℝ),
      distance_AB / time_eddy = speed_ratio * (distance_AC / time_freddy) ∧
      (distance_AB = known_distance ∨ distance_AC = known_distance) ∧
      distance_AB = 450

theorem distance_city_A_to_B_value : distance_city_A_to_B 3 4 2 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_city_A_to_B_distance_city_A_to_B_value_l2725_272536


namespace NUMINAMATH_CALUDE_percentage_subtraction_l2725_272564

theorem percentage_subtraction (total : ℝ) (difference : ℝ) : 
  total = 8000 → 
  difference = 796 → 
  ∃ (P : ℝ), (1/10 * total) - (P/100 * total) = difference ∧ P = 5 := by
sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l2725_272564


namespace NUMINAMATH_CALUDE_m_range_for_inequality_l2725_272521

theorem m_range_for_inequality (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m - m^2) * 4^x + 2^x + 1 > 0) → 
  -2 < m ∧ m < 3 := by
sorry

end NUMINAMATH_CALUDE_m_range_for_inequality_l2725_272521


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_range_f_inequality_when_a_zero_l2725_272500

noncomputable section

def f (a x : ℝ) : ℝ := (x - a) * Real.log x - x

theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) ↔ a ∈ Set.Iic (-1 / Real.exp 1) :=
sorry

theorem f_inequality_when_a_zero (x : ℝ) (hx : x > 0) :
  f 0 x ≥ x * (Real.exp (-x) - 1) - 2 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_range_f_inequality_when_a_zero_l2725_272500


namespace NUMINAMATH_CALUDE_min_domain_for_inverse_l2725_272590

-- Define the function g
def g (x : ℝ) : ℝ := (x - 3)^2 + 4

-- State the theorem
theorem min_domain_for_inverse :
  ∃ (d : ℝ), d = 3 ∧ 
  (∀ (d' : ℝ), (∀ (x y : ℝ), x ≥ d' ∧ y ≥ d' ∧ x ≠ y → g x ≠ g y) → d' ≥ d) ∧
  (∀ (x y : ℝ), x ≥ d ∧ y ≥ d ∧ x ≠ y → g x ≠ g y) :=
sorry

end NUMINAMATH_CALUDE_min_domain_for_inverse_l2725_272590


namespace NUMINAMATH_CALUDE_both_pass_through_origin_l2725_272556

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

end NUMINAMATH_CALUDE_both_pass_through_origin_l2725_272556


namespace NUMINAMATH_CALUDE_sum_of_functions_l2725_272525

theorem sum_of_functions (x : ℝ) (hx : x ≠ 2) :
  let f : ℝ → ℝ := λ x => x^2 - 1/(x-2)
  let g : ℝ → ℝ := λ x => 1/(x-2) + 1
  f x + g x = x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_functions_l2725_272525


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2725_272539

/-- The volume of a sphere circumscribing a rectangular solid with dimensions 1, 2, and 3 -/
theorem sphere_volume_circumscribing_rectangular_solid :
  let l : Real := 1  -- length
  let w : Real := 2  -- width
  let h : Real := 3  -- height
  let diagonal := Real.sqrt (l^2 + w^2 + h^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = (7 * Real.sqrt 14 / 3) * Real.pi := by
sorry


end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2725_272539


namespace NUMINAMATH_CALUDE_star_3_7_equals_16_l2725_272505

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem star_3_7_equals_16 : star 3 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_3_7_equals_16_l2725_272505


namespace NUMINAMATH_CALUDE_ninas_running_drill_l2725_272514

/-- Nina's running drill problem -/
theorem ninas_running_drill 
  (initial_run : ℝ) 
  (total_distance : ℝ) 
  (h1 : initial_run = 0.08333333333333333)
  (h2 : total_distance = 0.8333333333333334) :
  total_distance - 2 * initial_run = 0.6666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_ninas_running_drill_l2725_272514


namespace NUMINAMATH_CALUDE_min_value_expression_l2725_272517

theorem min_value_expression (x : ℝ) :
  x ≥ 0 →
  (1 + x^2) / (1 + x) ≥ -2 + 2 * Real.sqrt 2 ∧
  ∃ y : ℝ, y ≥ 0 ∧ (1 + y^2) / (1 + y) = -2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2725_272517


namespace NUMINAMATH_CALUDE_pool_capacity_l2725_272579

theorem pool_capacity (C : ℝ) 
  (h1 : 0.45 * C + 300 = 0.75 * C) : C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l2725_272579


namespace NUMINAMATH_CALUDE_basketball_time_calculation_l2725_272552

def football_time : ℕ := 60
def total_time_hours : ℕ := 2

theorem basketball_time_calculation :
  football_time + (total_time_hours * 60 - football_time) = 60 := by
  sorry

end NUMINAMATH_CALUDE_basketball_time_calculation_l2725_272552


namespace NUMINAMATH_CALUDE_three_digit_number_difference_l2725_272520

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

end NUMINAMATH_CALUDE_three_digit_number_difference_l2725_272520


namespace NUMINAMATH_CALUDE_lemon_orange_drink_scaling_l2725_272550

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

end NUMINAMATH_CALUDE_lemon_orange_drink_scaling_l2725_272550


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2725_272561

-- Define the circles and angles
def circle_small : Real → Real → Real := sorry
def circle_large : Real → Real → Real := sorry
def circle_sum : Real → Real → Real := sorry

def angle_small : Real := 60
def angle_large : Real := 48
def angle_sum : Real := 108

-- Define the radii
def radius_small : Real := sorry
def radius_large : Real := sorry
def radius_sum : Real := radius_small + radius_large

-- Define arc lengths
def arc_length (circle : Real → Real → Real) (angle : Real) : Real := sorry

-- State the theorem
theorem circle_area_ratio :
  let arc_small := arc_length circle_small angle_small
  let arc_large := arc_length circle_large angle_large
  let arc_sum := arc_length circle_sum angle_sum
  arc_small = arc_large ∧
  arc_sum = arc_small + arc_large →
  (circle_small radius_small 0) / (circle_large radius_large 0) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2725_272561


namespace NUMINAMATH_CALUDE_cookies_eaten_vs_given_l2725_272598

theorem cookies_eaten_vs_given (initial_cookies : ℕ) (eaten_cookies : ℕ) (given_cookies : ℕ) 
  (h1 : initial_cookies = 17) 
  (h2 : eaten_cookies = 14) 
  (h3 : given_cookies = 13) :
  eaten_cookies - given_cookies = 1 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_vs_given_l2725_272598


namespace NUMINAMATH_CALUDE_marigold_sale_problem_l2725_272504

theorem marigold_sale_problem (day1 day2 day3 total : ℕ) : 
  day1 = 14 →
  day3 = 2 * day2 →
  total = day1 + day2 + day3 →
  total = 89 →
  day2 = 25 := by
sorry

end NUMINAMATH_CALUDE_marigold_sale_problem_l2725_272504


namespace NUMINAMATH_CALUDE_restaurant_tables_difference_l2725_272574

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

end NUMINAMATH_CALUDE_restaurant_tables_difference_l2725_272574


namespace NUMINAMATH_CALUDE_a_range_l2725_272526

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + x^2 - a*x

theorem a_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) →
  (∀ x ∈ Set.Ioc 0 1, f a x ≤ 1/2 * (3*x^2 + 1/x^2 - 6*x)) →
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_a_range_l2725_272526


namespace NUMINAMATH_CALUDE_income_b_is_7200_l2725_272502

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

end NUMINAMATH_CALUDE_income_b_is_7200_l2725_272502


namespace NUMINAMATH_CALUDE_touching_values_are_zero_and_neg_four_l2725_272586

/-- Two linear functions with parallel, non-vertical graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c
  not_vertical : ∃ (a : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + (f 0)

/-- Condition that (f x)^2 touches 4(g x) -/
def touches_squared_to_scaled (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = 4 * (p.g x)

/-- Values of A for which (g x)^2 touches A(f x) -/
def touching_values (p : ParallelLinearFunctions) : Set ℝ :=
  {A | ∃! x, (p.g x)^2 = A * (p.f x)}

/-- Main theorem -/
theorem touching_values_are_zero_and_neg_four 
    (p : ParallelLinearFunctions) 
    (h : touches_squared_to_scaled p) : 
    touching_values p = {0, -4} := by
  sorry


end NUMINAMATH_CALUDE_touching_values_are_zero_and_neg_four_l2725_272586


namespace NUMINAMATH_CALUDE_simple_interest_from_sum_and_true_discount_l2725_272545

/-- Simple interest calculation given sum and true discount -/
theorem simple_interest_from_sum_and_true_discount
  (sum : ℝ) (true_discount : ℝ) (h1 : sum = 947.1428571428571)
  (h2 : true_discount = 78) :
  sum - (sum - true_discount) = true_discount :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_from_sum_and_true_discount_l2725_272545


namespace NUMINAMATH_CALUDE_shirt_count_l2725_272523

theorem shirt_count (total : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : total = 420) 
  (h2 : blue = 85) 
  (h3 : green = 157) : 
  total - (blue + green) = 178 := by
  sorry

end NUMINAMATH_CALUDE_shirt_count_l2725_272523


namespace NUMINAMATH_CALUDE_fish_tank_balls_count_total_balls_in_tank_l2725_272568

theorem fish_tank_balls_count : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_goldfish, num_platyfish, red_balls_per_goldfish, white_balls_per_platyfish =>
    num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish

theorem total_balls_in_tank : fish_tank_balls_count 3 10 10 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_balls_count_total_balls_in_tank_l2725_272568


namespace NUMINAMATH_CALUDE_intersection_condition_distance_product_condition_l2725_272591

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

end NUMINAMATH_CALUDE_intersection_condition_distance_product_condition_l2725_272591


namespace NUMINAMATH_CALUDE_find_a_l2725_272516

def U (a : ℝ) : Set ℝ := {3, 7, a^2 - 2*a - 3}

def A (a : ℝ) : Set ℝ := {7, |a - 7|}

theorem find_a : ∃ a : ℝ, (U a \ A a = {5}) ∧ (A a ⊆ U a) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2725_272516


namespace NUMINAMATH_CALUDE_pizza_and_burgers_cost_l2725_272589

/-- The cost of a burger in dollars -/
def burger_cost : ℕ := 9

/-- The cost of a pizza in dollars -/
def pizza_cost : ℕ := 2 * burger_cost

/-- The total cost of one pizza and three burgers in dollars -/
def total_cost : ℕ := pizza_cost + 3 * burger_cost

theorem pizza_and_burgers_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_pizza_and_burgers_cost_l2725_272589


namespace NUMINAMATH_CALUDE_A_when_one_is_element_B_is_zero_and_neg_one_third_l2725_272518

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x - 3 = 0}

-- Theorem 1: If 1 ∈ A, then A = {1, -3}
theorem A_when_one_is_element (a : ℝ) : 1 ∈ A a → A a = {1, -3} := by sorry

-- Define the set B
def B : Set ℝ := {a : ℝ | ∃! x, x ∈ A a}

-- Theorem 2: B = {0, -1/3}
theorem B_is_zero_and_neg_one_third : B = {0, -1/3} := by sorry

end NUMINAMATH_CALUDE_A_when_one_is_element_B_is_zero_and_neg_one_third_l2725_272518


namespace NUMINAMATH_CALUDE_dot_product_range_l2725_272582

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

end NUMINAMATH_CALUDE_dot_product_range_l2725_272582


namespace NUMINAMATH_CALUDE_not_divisible_by_product_l2725_272506

theorem not_divisible_by_product (a₁ a₂ b₁ b₂ : ℕ) 
  (h1 : 1 < b₁) (h2 : b₁ < a₁) (h3 : 1 < b₂) (h4 : b₂ < a₂) 
  (h5 : b₁ ∣ a₁) (h6 : b₂ ∣ a₂) : 
  ¬(a₁ * a₂ ∣ a₁ * b₁ + a₂ * b₂ - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_product_l2725_272506


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequences_l2725_272596

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

end NUMINAMATH_CALUDE_geometric_arithmetic_sequences_l2725_272596


namespace NUMINAMATH_CALUDE_statues_painted_l2725_272558

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/8 ∧ paint_per_statue = 1/8 → total_paint / paint_per_statue = 7 := by
  sorry

end NUMINAMATH_CALUDE_statues_painted_l2725_272558


namespace NUMINAMATH_CALUDE_peter_notebooks_l2725_272597

def green_notebooks : ℕ := 2
def black_notebooks : ℕ := 1
def pink_notebooks : ℕ := 1

def total_notebooks : ℕ := green_notebooks + black_notebooks + pink_notebooks

theorem peter_notebooks : total_notebooks = 4 := by sorry

end NUMINAMATH_CALUDE_peter_notebooks_l2725_272597


namespace NUMINAMATH_CALUDE_bakery_doughnuts_given_away_l2725_272560

/-- Given a bakery scenario, prove the number of doughnuts given away -/
theorem bakery_doughnuts_given_away
  (total_doughnuts : ℕ)
  (doughnuts_per_box : ℕ)
  (boxes_sold : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : doughnuts_per_box = 10)
  (h3 : boxes_sold = 27)
  : (total_doughnuts - boxes_sold * doughnuts_per_box) = 30 :=
by sorry

end NUMINAMATH_CALUDE_bakery_doughnuts_given_away_l2725_272560


namespace NUMINAMATH_CALUDE_boot_pairing_l2725_272577

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


end NUMINAMATH_CALUDE_boot_pairing_l2725_272577


namespace NUMINAMATH_CALUDE_bomb_guaranteed_four_of_a_kind_guaranteed_l2725_272554

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

end NUMINAMATH_CALUDE_bomb_guaranteed_four_of_a_kind_guaranteed_l2725_272554


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2725_272575

theorem arithmetic_calculation : 8 / 4 + 5 * 2^2 - (3 + 7) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2725_272575


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l2725_272531

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

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l2725_272531


namespace NUMINAMATH_CALUDE_mod7_mul_table_mod10_mul_2_mod10_mul_5_mod9_mul_3_l2725_272551

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

end NUMINAMATH_CALUDE_mod7_mul_table_mod10_mul_2_mod10_mul_5_mod9_mul_3_l2725_272551


namespace NUMINAMATH_CALUDE_limit_sin_x_over_x_sin_x_over_x_squeeze_cos_continuous_limit_sin_x_over_x_equals_one_l2725_272529

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

end NUMINAMATH_CALUDE_limit_sin_x_over_x_sin_x_over_x_squeeze_cos_continuous_limit_sin_x_over_x_equals_one_l2725_272529


namespace NUMINAMATH_CALUDE_inclination_angle_theorem_l2725_272530

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

end NUMINAMATH_CALUDE_inclination_angle_theorem_l2725_272530


namespace NUMINAMATH_CALUDE_equation_solutions_l2725_272532

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (2 + Real.sqrt 7) / 3 ∧ x₂ = (2 - Real.sqrt 7) / 3 ∧
    3 * x₁^2 - 1 = 4 * x₁ ∧ 3 * x₂^2 - 1 = 4 * x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4 ∧ x₂ = 1 ∧
    (x₁ + 4)^2 = 5 * (x₁ + 4) ∧ (x₂ + 4)^2 = 5 * (x₂ + 4)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2725_272532


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2725_272537

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- base radius
  let h : ℝ := 4  -- height
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  let S : ℝ := π * r * l  -- lateral surface area formula
  S = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2725_272537


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2725_272559

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (k : ℝ), k * a = b ∧ k * 2 = Real.sqrt 3) →  -- asymptote condition
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = Real.sqrt 7) →  -- focus and directrix condition
  a^2 = 4 ∧ b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2725_272559


namespace NUMINAMATH_CALUDE_tooth_extraction_cost_l2725_272563

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def total_fillings : ℕ := 2
def total_bill_factor : ℕ := 5

theorem tooth_extraction_cost :
  let total_bill := filling_cost * total_bill_factor
  let cleaning_and_fillings_cost := cleaning_cost + (filling_cost * total_fillings)
  total_bill - cleaning_and_fillings_cost = 290 :=
by sorry

end NUMINAMATH_CALUDE_tooth_extraction_cost_l2725_272563


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_l2725_272509

/-- The total amount Jason spent on clothing -/
def total_spent : ℚ := 19.02

/-- The amount Jason spent on shorts -/
def shorts_cost : ℚ := 14.28

/-- The amount Jason spent on a jacket -/
def jacket_cost : ℚ := 4.74

/-- Theorem stating that the total amount spent is the sum of the costs of shorts and jacket -/
theorem total_spent_equals_sum : total_spent = shorts_cost + jacket_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_l2725_272509


namespace NUMINAMATH_CALUDE_simplify_expression_l2725_272572

theorem simplify_expression (a b : ℝ) : 
  (50*a + 130*b) + (21*a + 64*b) - (30*a + 115*b) - 2*(10*a - 25*b) = 21*a + 129*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2725_272572


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2725_272547

/-- Given a system of linear equations with a parameter k, 
    prove that for a specific value of k, the ratio yz/x^2 is constant --/
theorem system_solution_ratio (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  let k : ℝ := 55 / 26
  x + 2 * k * y + 4 * z = 0 ∧
  4 * x + 2 * k * y - 3 * z = 0 ∧
  3 * x + 5 * y - 4 * z = 0 →
  ∃ (c : ℝ), y * z / (x^2) = c :=
by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2725_272547


namespace NUMINAMATH_CALUDE_intersection_product_l2725_272588

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

end NUMINAMATH_CALUDE_intersection_product_l2725_272588


namespace NUMINAMATH_CALUDE_girls_not_attending_college_percentage_l2725_272599

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

end NUMINAMATH_CALUDE_girls_not_attending_college_percentage_l2725_272599


namespace NUMINAMATH_CALUDE_rectangle_area_l2725_272538

theorem rectangle_area (square_area : ℝ) (rectangle_length_multiplier : ℝ) : 
  square_area = 36 → 
  rectangle_length_multiplier = 3 → 
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_multiplier * rectangle_width
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2725_272538


namespace NUMINAMATH_CALUDE_provision_duration_l2725_272542

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

end NUMINAMATH_CALUDE_provision_duration_l2725_272542


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2725_272592

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_two : 2 * lg 5 + lg 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2725_272592


namespace NUMINAMATH_CALUDE_expression_simplification_l2725_272565

theorem expression_simplification (x y : ℝ) (h : (x + 2)^3 - (y - 2)^3 ≠ 0) :
  ((x + 2)^3 + (y + x)^3) / ((x + 2)^3 - (y - 2)^3) = (2*x + y + 2) / (x - y + 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2725_272565


namespace NUMINAMATH_CALUDE_soda_price_theorem_l2725_272569

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

end NUMINAMATH_CALUDE_soda_price_theorem_l2725_272569


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l2725_272580

theorem difference_of_squares_division : (121^2 - 112^2) / 9 = 233 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l2725_272580


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_for_51234_div_9_least_addition_is_3_l2725_272528

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_for_51234_div_9 :
  ∃ (x : ℕ), x < 9 ∧ (51234 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (51234 + y) % 9 ≠ 0 :=
by
  apply least_addition_for_divisibility 51234 9
  norm_num

theorem least_addition_is_3 :
  ∃! (x : ℕ), x < 9 ∧ (51234 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (51234 + y) % 9 ≠ 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_for_51234_div_9_least_addition_is_3_l2725_272528


namespace NUMINAMATH_CALUDE_optimal_price_and_units_l2725_272507

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

end NUMINAMATH_CALUDE_optimal_price_and_units_l2725_272507


namespace NUMINAMATH_CALUDE_john_needs_29_planks_l2725_272535

/-- The number of large planks John uses for the house wall. -/
def large_planks : ℕ := 12

/-- The number of small planks John uses for the house wall. -/
def small_planks : ℕ := 17

/-- The total number of planks John needs for the house wall. -/
def total_planks : ℕ := large_planks + small_planks

/-- Theorem stating that the total number of planks John needs is 29. -/
theorem john_needs_29_planks : total_planks = 29 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_29_planks_l2725_272535


namespace NUMINAMATH_CALUDE_ratio_of_arithmetic_sums_l2725_272573

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem ratio_of_arithmetic_sums : 
  let n₁ := (60 - 4) / 4 + 1
  let n₂ := (72 - 6) / 6 + 1
  (arithmetic_sum 4 4 n₁) / (arithmetic_sum 6 6 n₂) = 40 / 39 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_arithmetic_sums_l2725_272573


namespace NUMINAMATH_CALUDE_blocks_with_one_face_painted_10_2_l2725_272501

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

end NUMINAMATH_CALUDE_blocks_with_one_face_painted_10_2_l2725_272501


namespace NUMINAMATH_CALUDE_sum_of_one_third_and_two_thirds_equals_one_l2725_272578

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  (n : ℚ) / 9

theorem sum_of_one_third_and_two_thirds_equals_one :
  RepeatingDecimal 3 + RepeatingDecimal 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_one_third_and_two_thirds_equals_one_l2725_272578


namespace NUMINAMATH_CALUDE_rectangle_dimension_relationship_l2725_272595

/-- Given a rectangle with perimeter 20m, prove that the relationship between its length y and width x is y = -x + 10 -/
theorem rectangle_dimension_relationship (x y : ℝ) : 
  (2 * (x + y) = 20) → (y = -x + 10) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_relationship_l2725_272595


namespace NUMINAMATH_CALUDE_f_decreasing_inequality_solution_set_l2725_272511

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

end NUMINAMATH_CALUDE_f_decreasing_inequality_solution_set_l2725_272511


namespace NUMINAMATH_CALUDE_prec_2011_130_l2725_272567

-- Define the new operation ⪯
def prec (a b : ℕ) : ℕ := b * 10 + a * 2

-- Theorem to prove
theorem prec_2011_130 : prec 2011 130 = 5322 := by
  sorry

end NUMINAMATH_CALUDE_prec_2011_130_l2725_272567


namespace NUMINAMATH_CALUDE_graduation_ceremony_arrangements_l2725_272566

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

end NUMINAMATH_CALUDE_graduation_ceremony_arrangements_l2725_272566


namespace NUMINAMATH_CALUDE_rob_travel_time_l2725_272522

/-- The time it takes Rob to get to the national park -/
def rob_time : ℝ := 1

/-- The time it takes Mark to get to the national park -/
def mark_time : ℝ := 3 * rob_time

/-- The head start time Mark has -/
def head_start : ℝ := 2

theorem rob_travel_time : 
  head_start + rob_time = mark_time ∧ rob_time = 1 := by sorry

end NUMINAMATH_CALUDE_rob_travel_time_l2725_272522


namespace NUMINAMATH_CALUDE_gcd_2100_2091_l2725_272546

theorem gcd_2100_2091 : Nat.gcd (2^2100 - 1) (2^2091 - 1) = 2^9 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_2100_2091_l2725_272546


namespace NUMINAMATH_CALUDE_second_day_speed_l2725_272527

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

end NUMINAMATH_CALUDE_second_day_speed_l2725_272527


namespace NUMINAMATH_CALUDE_regression_increase_l2725_272549

/-- Linear regression equation for annual food expenditure with respect to annual income -/
def regression_equation (x : ℝ) : ℝ := 0.254 * x + 0.321

/-- Theorem stating that the increase in the regression equation's output for a 1 unit increase in input is 0.254 -/
theorem regression_increase : ∀ x : ℝ, regression_equation (x + 1) - regression_equation x = 0.254 := by
  sorry

end NUMINAMATH_CALUDE_regression_increase_l2725_272549


namespace NUMINAMATH_CALUDE_series_term_equals_original_term_l2725_272593

/-- The n-th term of the series -4+7-4+7-4+7-... -/
def seriesTerm (n : ℕ) : ℝ :=
  1.5 + 5.5 * (-1)^n

/-- The original series terms -/
def originalTerm (n : ℕ) : ℝ :=
  if n % 2 = 1 then -4 else 7

theorem series_term_equals_original_term (n : ℕ) :
  seriesTerm n = originalTerm n := by
  sorry

#check series_term_equals_original_term

end NUMINAMATH_CALUDE_series_term_equals_original_term_l2725_272593


namespace NUMINAMATH_CALUDE_perfect_fourth_power_in_range_l2725_272512

theorem perfect_fourth_power_in_range : ∃! K : ℤ,
  (K > 0) ∧
  (∃ Z : ℤ, 1000 < Z ∧ Z < 2000 ∧ Z = K * K^3) ∧
  (∃ n : ℤ, K^4 = n^4) :=
by sorry

end NUMINAMATH_CALUDE_perfect_fourth_power_in_range_l2725_272512


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2725_272555

def profit_A (x : ℕ) : ℚ := 5.06 * x - 0.15 * x^2
def profit_B (x : ℕ) : ℚ := 2 * x

theorem max_profit_theorem :
  ∃ (x : ℕ), x ≤ 15 ∧ 
  (∀ (y : ℕ), y ≤ 15 → 
    profit_A x + profit_B (15 - x) ≥ profit_A y + profit_B (15 - y)) ∧
  profit_A x + profit_B (15 - x) = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l2725_272555


namespace NUMINAMATH_CALUDE_group_size_solve_group_size_l2725_272571

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

end NUMINAMATH_CALUDE_group_size_solve_group_size_l2725_272571


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l2725_272515

theorem opposite_number_theorem (a : ℝ) : (-(-a) = -2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l2725_272515


namespace NUMINAMATH_CALUDE_greatest_mean_Y_Z_l2725_272534

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

end NUMINAMATH_CALUDE_greatest_mean_Y_Z_l2725_272534


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2725_272562

theorem arithmetic_operations : 
  (-3 : ℤ) + 2 = -1 ∧ (-3 : ℤ) * 2 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2725_272562


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2725_272524

theorem closest_integer_to_cube_root_150 : 
  ∀ n : ℤ, |n - (150 : ℝ)^(1/3)| ≥ |5 - (150 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2725_272524


namespace NUMINAMATH_CALUDE_sarah_cupcake_count_l2725_272533

def is_valid_cupcake_count (c : ℕ) : Prop :=
  ∃ (k : ℕ), 
    c + k = 6 ∧ 
    (90 * c + 40 * k) % 100 = 0

theorem sarah_cupcake_count :
  ∀ c : ℕ, is_valid_cupcake_count c → c = 4 ∨ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sarah_cupcake_count_l2725_272533


namespace NUMINAMATH_CALUDE_brads_balloons_l2725_272584

/-- Brad's balloon count problem -/
theorem brads_balloons (red : ℕ) (green : ℕ) 
  (h1 : red = 8) 
  (h2 : green = 9) : 
  red + green = 17 := by
  sorry

end NUMINAMATH_CALUDE_brads_balloons_l2725_272584


namespace NUMINAMATH_CALUDE_second_hand_revolution_time_l2725_272503

/-- The time in seconds for a second hand to complete one revolution -/
def revolution_time_seconds : ℕ := 60

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The time in minutes for a second hand to complete one revolution -/
def revolution_time_minutes : ℚ := revolution_time_seconds / seconds_per_minute

theorem second_hand_revolution_time :
  revolution_time_seconds = 60 ∧ revolution_time_minutes = 1 := by sorry

end NUMINAMATH_CALUDE_second_hand_revolution_time_l2725_272503


namespace NUMINAMATH_CALUDE_negation_of_implication_l2725_272543

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2725_272543


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2725_272510

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

end NUMINAMATH_CALUDE_expansion_coefficient_l2725_272510
