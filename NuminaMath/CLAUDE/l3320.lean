import Mathlib

namespace NUMINAMATH_CALUDE_power_of_2016_expression_evaluation_l3320_332033

-- Part 1
theorem power_of_2016 (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^(m+4*n) = 324) : 
  2016^n = 2016 := by sorry

-- Part 2
theorem expression_evaluation (a : ℝ) (h : a = 5) : 
  (a+2)*(a-2) + a*(1-a) = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_2016_expression_evaluation_l3320_332033


namespace NUMINAMATH_CALUDE_problem_statement_l3320_332076

theorem problem_statement : 
  (-1)^2023 + (8 : ℝ)^(1/3) - 2 * (1/4 : ℝ)^(1/2) + |Real.sqrt 3 - 2| = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3320_332076


namespace NUMINAMATH_CALUDE_visited_neither_country_l3320_332023

theorem visited_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 250 →
  iceland = 125 →
  norway = 95 →
  both = 80 →
  total - ((iceland + norway) - both) = 110 :=
by sorry

end NUMINAMATH_CALUDE_visited_neither_country_l3320_332023


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3320_332046

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 2 * x + y → x + y ≥ 2 * Real.sqrt 2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3320_332046


namespace NUMINAMATH_CALUDE_mean_temperature_is_85_point_6_l3320_332088

def temperatures : List ℝ := [85, 84, 85, 83, 82, 84, 86, 88, 90, 89]

theorem mean_temperature_is_85_point_6 :
  (List.sum temperatures) / (List.length temperatures) = 85.6 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_85_point_6_l3320_332088


namespace NUMINAMATH_CALUDE_projectile_speed_calculation_l3320_332099

/-- 
Given two projectiles launched simultaneously 1455 km apart, with one traveling at 500 km/h,
prove that the speed of the other projectile is 470 km/h if they meet after 90 minutes.
-/
theorem projectile_speed_calculation (distance : ℝ) (time : ℝ) (speed2 : ℝ) (speed1 : ℝ) : 
  distance = 1455 → 
  time = 1.5 → 
  speed2 = 500 → 
  speed1 = 470 → 
  distance = (speed1 + speed2) * time :=
by sorry

end NUMINAMATH_CALUDE_projectile_speed_calculation_l3320_332099


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3320_332060

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≤ 999 ∧ n ≥ 100 ∧ 17 ∣ n → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3320_332060


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_one_l3320_332034

-- Define the property of being a meaningful square root
def is_meaningful_sqrt (x : ℝ) : Prop := 1 - x ≥ 0

-- State the theorem
theorem sqrt_meaningful_iff_leq_one :
  ∀ x : ℝ, is_meaningful_sqrt x ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_one_l3320_332034


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3320_332072

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y + 1) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3320_332072


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3320_332022

/-- Calculates the overall profit percentage for a retailer selling three items -/
theorem retailer_profit_percentage
  (radio_cost watch_cost phone_cost : ℚ)
  (radio_overhead watch_overhead phone_overhead : ℚ)
  (radio_sp watch_sp phone_sp : ℚ)
  (h_radio_cost : radio_cost = 225)
  (h_watch_cost : watch_cost = 425)
  (h_phone_cost : phone_cost = 650)
  (h_radio_overhead : radio_overhead = 15)
  (h_watch_overhead : watch_overhead = 20)
  (h_phone_overhead : phone_overhead = 30)
  (h_radio_sp : radio_sp = 300)
  (h_watch_sp : watch_sp = 525)
  (h_phone_sp : phone_sp = 800) :
  let total_cp := radio_cost + watch_cost + phone_cost + radio_overhead + watch_overhead + phone_overhead
  let total_sp := radio_sp + watch_sp + phone_sp
  let profit := total_sp - total_cp
  let profit_percentage := (profit / total_cp) * 100
  ∃ ε > 0, |profit_percentage - 19.05| < ε :=
by sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3320_332022


namespace NUMINAMATH_CALUDE_smallest_product_l3320_332094

def digits : List Nat := [7, 8, 9, 10]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 63990 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l3320_332094


namespace NUMINAMATH_CALUDE_orange_tree_problem_l3320_332025

theorem orange_tree_problem (total_trees : ℕ) (tree_a_percent : ℚ) (tree_b_percent : ℚ)
  (tree_b_oranges : ℕ) (tree_b_good_ratio : ℚ) (tree_a_good_ratio : ℚ) (total_good_oranges : ℕ) :
  tree_a_percent = 1/2 →
  tree_b_percent = 1/2 →
  tree_b_oranges = 15 →
  tree_b_good_ratio = 1/3 →
  tree_a_good_ratio = 3/5 →
  total_trees = 10 →
  total_good_oranges = 55 →
  ∃ (tree_a_oranges : ℕ), 
    (tree_a_percent * total_trees : ℚ) * (tree_a_oranges : ℚ) * tree_a_good_ratio +
    (tree_b_percent * total_trees : ℚ) * (tree_b_oranges : ℚ) * tree_b_good_ratio =
    total_good_oranges ∧
    tree_a_oranges = 10 :=
by sorry

end NUMINAMATH_CALUDE_orange_tree_problem_l3320_332025


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l3320_332057

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- Probability of exactly k out of n packages being delivered to their correct houses -/
def probability_correct_delivery (n k : ℕ) : ℚ :=
  (n.choose k * (k.factorial : ℚ) * ((n - k).factorial : ℚ)) / (n.factorial : ℚ)

/-- Theorem stating the probability of exactly 3 out of 5 packages being delivered correctly -/
theorem correct_delivery_probability :
  probability_correct_delivery n k = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_correct_delivery_probability_l3320_332057


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3320_332083

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3320_332083


namespace NUMINAMATH_CALUDE_a_minus_b_value_l3320_332086

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem a_minus_b_value (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  a - b = 29/4 :=
by sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l3320_332086


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l3320_332006

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples * 40 / 100 = 560) → initial_apples = 1400 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l3320_332006


namespace NUMINAMATH_CALUDE_raffle_ticket_cost_l3320_332017

theorem raffle_ticket_cost (x : ℚ) : 
  (25 * x + 30 + 20 = 100) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_cost_l3320_332017


namespace NUMINAMATH_CALUDE_shooting_competition_probabilities_l3320_332028

-- Define the probabilities for A and B hitting different rings
def prob_A_8 : ℝ := 0.6
def prob_A_9 : ℝ := 0.3
def prob_A_10 : ℝ := 0.1
def prob_B_8 : ℝ := 0.4
def prob_B_9 : ℝ := 0.4
def prob_B_10 : ℝ := 0.2

-- Define the probability that A hits more rings than B in a single round
def prob_A_beats_B : ℝ := prob_A_9 * prob_B_8 + prob_A_10 * prob_B_8 + prob_A_10 * prob_B_9

-- Define the probability that A hits more rings than B in at least two out of three independent rounds
def prob_A_beats_B_twice_or_more : ℝ :=
  3 * prob_A_beats_B^2 * (1 - prob_A_beats_B) + prob_A_beats_B^3

theorem shooting_competition_probabilities :
  prob_A_beats_B = 0.2 ∧ prob_A_beats_B_twice_or_more = 0.104 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_probabilities_l3320_332028


namespace NUMINAMATH_CALUDE_financial_audit_equation_l3320_332042

theorem financial_audit_equation (p v : ℂ) : 
  (7 * p - v = 23000) → (v = 50 + 250 * Complex.I) → 
  (p = 3292.857 + 35.714 * Complex.I) := by
sorry

end NUMINAMATH_CALUDE_financial_audit_equation_l3320_332042


namespace NUMINAMATH_CALUDE_triangle_roots_condition_l3320_332070

/-- Given a cubic polynomial x^3 - ux^2 + vx - w with roots a, b, and c forming a triangle, 
    prove that uv > 2w -/
theorem triangle_roots_condition (u v w a b c : ℝ) : 
  (∀ x, x^3 - u*x^2 + v*x - w = (x - a)*(x - b)*(x - c)) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  uv > 2*w :=
sorry

end NUMINAMATH_CALUDE_triangle_roots_condition_l3320_332070


namespace NUMINAMATH_CALUDE_initial_balance_is_20_l3320_332005

def football_club_balance (initial_balance : ℝ) : Prop :=
  let players_sold := 2
  let price_per_sold_player := 10
  let players_bought := 4
  let price_per_bought_player := 15
  let final_balance := 60
  
  initial_balance + players_sold * price_per_sold_player - 
  players_bought * price_per_bought_player = final_balance

theorem initial_balance_is_20 : 
  ∃ (initial_balance : ℝ), football_club_balance initial_balance ∧ initial_balance = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_balance_is_20_l3320_332005


namespace NUMINAMATH_CALUDE_student_marks_l3320_332047

theorem student_marks (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 20 →
  M + P = 20 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_l3320_332047


namespace NUMINAMATH_CALUDE_product_of_cosines_l3320_332098

theorem product_of_cosines : 
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) * 
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l3320_332098


namespace NUMINAMATH_CALUDE_table_stool_equation_correctness_l3320_332014

/-- Represents a scenario with tables and stools -/
structure TableStoolScenario where
  numTables : ℕ
  numStools : ℕ
  totalItems : ℕ
  totalLegs : ℕ
  h_totalItems : numTables + numStools = totalItems
  h_totalLegs : 4 * numTables + 3 * numStools = totalLegs

/-- The correct system of equations for the given scenario -/
def correctSystem (x y : ℕ) : Prop :=
  x + y = 12 ∧ 4 * x + 3 * y = 40

theorem table_stool_equation_correctness :
  ∀ (scenario : TableStoolScenario),
    scenario.totalItems = 12 →
    scenario.totalLegs = 40 →
    correctSystem scenario.numTables scenario.numStools :=
by sorry

end NUMINAMATH_CALUDE_table_stool_equation_correctness_l3320_332014


namespace NUMINAMATH_CALUDE_coin_counting_machine_result_l3320_332030

def coin_value (coin_type : String) : ℚ :=
  match coin_type with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

def total_value (quarters dimes nickels pennies : ℕ) : ℚ :=
  quarters * coin_value "quarter" +
  dimes * coin_value "dime" +
  nickels * coin_value "nickel" +
  pennies * coin_value "penny"

def fee_percentage : ℚ := 10 / 100

theorem coin_counting_machine_result 
  (quarters dimes nickels pennies : ℕ) : 
  quarters = 76 → dimes = 85 → nickels = 20 → pennies = 150 →
  (total_value quarters dimes nickels pennies) * (1 - fee_percentage) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_counting_machine_result_l3320_332030


namespace NUMINAMATH_CALUDE_pr_cr_relation_l3320_332079

theorem pr_cr_relation (p c : ℝ) :
  (6 * p * 4 = 360) → (p = 15 ∧ 6 * c * 4 = 24 * c) := by
  sorry

end NUMINAMATH_CALUDE_pr_cr_relation_l3320_332079


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l3320_332058

/-- The volume of a 60% salt solution needed to mix with 1 liter of pure water to create a 20% salt solution -/
def salt_solution_volume : ℝ := 0.5

/-- The concentration of salt in the original solution -/
def original_concentration : ℝ := 0.6

/-- The concentration of salt in the final mixture -/
def final_concentration : ℝ := 0.2

/-- The volume of pure water added -/
def pure_water_volume : ℝ := 1

theorem salt_solution_mixture :
  salt_solution_volume * original_concentration = 
  (pure_water_volume + salt_solution_volume) * final_concentration :=
sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l3320_332058


namespace NUMINAMATH_CALUDE_complexity_theorem_l3320_332012

-- Define complexity of a positive integer
def complexity (n : ℕ) : ℕ := sorry

-- Define the property for part (a)
def property_a (n : ℕ) : Prop :=
  ∀ m : ℕ, n ≤ m → m ≤ 2*n → complexity m ≤ complexity n

-- Define the property for part (b)
def property_b (n : ℕ) : Prop :=
  ∀ m : ℕ, n < m → m < 2*n → complexity m < complexity n

theorem complexity_theorem :
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n = 2^k → property_a n) ∧
  (¬ ∃ n : ℕ, n > 1 ∧ property_b n) := by sorry

end NUMINAMATH_CALUDE_complexity_theorem_l3320_332012


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3320_332055

def fourth_quadrant (z : ℂ) : Prop := 
  Complex.re z > 0 ∧ Complex.im z < 0

theorem point_in_fourth_quadrant : 
  fourth_quadrant ((2 - Complex.I) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3320_332055


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l3320_332043

theorem triangle_tangent_product (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →  -- Sum of angles in a triangle
  (a > 0) → (b > 0) → (c > 0) →  -- Positive side lengths
  (a / (2 * Real.sin (A / 2)) = b / (2 * Real.sin (B / 2))) →  -- Sine law
  (b / (2 * Real.sin (B / 2)) = c / (2 * Real.sin (C / 2))) →  -- Sine law
  (a + c = 2 * b) →  -- Given condition
  (Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l3320_332043


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l3320_332002

theorem easter_egg_hunt (kevin bonnie cheryl george : ℕ) 
  (h1 : kevin = 5)
  (h2 : bonnie = 13)
  (h3 : cheryl = 56)
  (h4 : cheryl = kevin + bonnie + george + 29) :
  george = 9 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l3320_332002


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_attained_l3320_332003

theorem min_value_of_expression (x : ℝ) : 
  (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 ≥ 2008 :=
by sorry

theorem equality_attained : 
  ∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 = 2008 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_attained_l3320_332003


namespace NUMINAMATH_CALUDE_eggs_leftover_l3320_332013

def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25
def carton_size : ℕ := 10

theorem eggs_leftover :
  (david_eggs + emma_eggs + fiona_eggs) % carton_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_leftover_l3320_332013


namespace NUMINAMATH_CALUDE_discount_difference_l3320_332021

def bill_amount : ℝ := 12000
def single_discount : ℝ := 0.42
def first_successive_discount : ℝ := 0.35
def second_successive_discount : ℝ := 0.05

def single_discounted_amount : ℝ := bill_amount * (1 - single_discount)
def successive_discounted_amount : ℝ := bill_amount * (1 - first_successive_discount) * (1 - second_successive_discount)

theorem discount_difference :
  successive_discounted_amount - single_discounted_amount = 450 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l3320_332021


namespace NUMINAMATH_CALUDE_cost_per_song_l3320_332019

/-- Calculates the cost per song given monthly music purchase, average song length, and annual expenditure -/
theorem cost_per_song 
  (monthly_hours : ℝ) 
  (song_length_minutes : ℝ) 
  (annual_cost : ℝ) 
  (h1 : monthly_hours = 20)
  (h2 : song_length_minutes = 3)
  (h3 : annual_cost = 2400) : 
  annual_cost / (monthly_hours * 12 * 60 / song_length_minutes) = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_song_l3320_332019


namespace NUMINAMATH_CALUDE_managers_salary_l3320_332052

/-- Given an organization with employees and their salaries, this theorem proves
    the salary of an additional member that would increase the average by a specific amount. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 20 →
  avg_salary = 1700 →
  avg_increase = 100 →
  (num_employees * avg_salary + 3800) / (num_employees + 1) = avg_salary + avg_increase := by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l3320_332052


namespace NUMINAMATH_CALUDE_final_water_level_l3320_332062

/-- The final water level in a system of two connected cylindrical vessels -/
theorem final_water_level 
  (h : ℝ) -- Initial height of both liquids
  (ρ_water : ℝ) -- Density of water
  (ρ_oil : ℝ) -- Density of oil
  (h_pos : h > 0)
  (ρ_water_pos : ρ_water > 0)
  (ρ_oil_pos : ρ_oil > 0)
  (h_val : h = 40)
  (ρ_water_val : ρ_water = 1000)
  (ρ_oil_val : ρ_oil = 700) :
  ∃ (h_water : ℝ), h_water = 280 / 17 ∧ 
    ρ_water * h_water = ρ_oil * (h - h_water) ∧
    h_water > 0 ∧ h_water < h :=
by
  sorry


end NUMINAMATH_CALUDE_final_water_level_l3320_332062


namespace NUMINAMATH_CALUDE_square_in_S_l3320_332040

def S : Set ℕ := {n | ∃ a b c d e f : ℕ, 
  (n - 1 = a^2 + b^2) ∧ 
  (n = c^2 + d^2) ∧ 
  (n + 1 = e^2 + f^2) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0)}

theorem square_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S := by
  sorry

end NUMINAMATH_CALUDE_square_in_S_l3320_332040


namespace NUMINAMATH_CALUDE_equation_solution_l3320_332041

theorem equation_solution : 
  ∃ x : ℝ, (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + x + (0.9 : ℝ)^2 = 0.2999999999999999 ∧ 
  x = -1.73175 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3320_332041


namespace NUMINAMATH_CALUDE_range_of_f_l3320_332069

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -18 ≤ y ∧ y ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3320_332069


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l3320_332091

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a population with subgroups -/
structure Population where
  total : Nat
  subgroups : List Nat
  sample_size : Nat

/-- Represents a simple population without subgroups -/
structure SimplePopulation where
  total : Nat
  sample_size : Nat

def student_population : Population :=
  { total := 1200
  , subgroups := [400, 600, 200]
  , sample_size := 120 }

def parent_population : SimplePopulation :=
  { total := 10
  , sample_size := 3 }

/-- Determines the best sampling method for a given population -/
def best_sampling_method (pop : Population) : SamplingMethod :=
  sorry

/-- Determines the best sampling method for a simple population -/
def best_simple_sampling_method (pop : SimplePopulation) : SamplingMethod :=
  sorry

theorem correct_sampling_methods :
  (best_sampling_method student_population = SamplingMethod.Stratified) ∧
  (best_simple_sampling_method parent_population = SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l3320_332091


namespace NUMINAMATH_CALUDE_stratified_sampling_major_c_l3320_332039

/-- Represents the number of students to be sampled from a major -/
def sampleSize (totalStudents : ℕ) (sampleTotal : ℕ) (majorStudents : ℕ) : ℕ :=
  (sampleTotal * majorStudents) / totalStudents

/-- Proves that the number of students to be drawn from major C is 40 -/
theorem stratified_sampling_major_c :
  let totalStudents : ℕ := 1200
  let sampleTotal : ℕ := 120
  let majorAStudents : ℕ := 380
  let majorBStudents : ℕ := 420
  let majorCStudents : ℕ := totalStudents - majorAStudents - majorBStudents
  sampleSize totalStudents sampleTotal majorCStudents = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_major_c_l3320_332039


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3320_332063

/-- Given vectors a and b in R², prove that |a - b| = 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3320_332063


namespace NUMINAMATH_CALUDE_rain_probability_l3320_332093

theorem rain_probability (M T N : ℝ) 
  (hM : M = 0.6)  -- 60% of counties received rain on Monday
  (hT : T = 0.55) -- 55% of counties received rain on Tuesday
  (hN : N = 0.25) -- 25% of counties received no rain on either day
  : M + T - N - 1 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3320_332093


namespace NUMINAMATH_CALUDE_library_books_checkout_l3320_332075

theorem library_books_checkout (total : ℕ) (ratio_nf : ℕ) (ratio_f : ℕ) (h1 : total = 52) (h2 : ratio_nf = 7) (h3 : ratio_f = 6) : 
  (total * ratio_f) / (ratio_nf + ratio_f) = 24 := by
  sorry

end NUMINAMATH_CALUDE_library_books_checkout_l3320_332075


namespace NUMINAMATH_CALUDE_meat_distribution_l3320_332081

/-- Proves the correct distribution of meat between two pots -/
theorem meat_distribution (pot1 pot2 total_meat : ℕ) 
  (h1 : pot1 = 645)
  (h2 : pot2 = 237)
  (h3 : total_meat = 1000) :
  ∃ (meat1 meat2 : ℕ),
    meat1 + meat2 = total_meat ∧
    pot1 + meat1 = pot2 + meat2 ∧
    meat1 = 296 ∧
    meat2 = 704 := by
  sorry

end NUMINAMATH_CALUDE_meat_distribution_l3320_332081


namespace NUMINAMATH_CALUDE_inverse_x_equals_three_l3320_332084

theorem inverse_x_equals_three (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + 1/27 = x*y) : 1/x = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_x_equals_three_l3320_332084


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3320_332008

theorem smallest_dual_base_representation : ∃ n : ℕ, ∃ a b : ℕ, 
  (a > 2 ∧ b > 2) ∧ 
  (1 * a + 3 = n) ∧ 
  (3 * b + 1 = n) ∧
  (∀ m : ℕ, ∀ c d : ℕ, 
    (c > 2 ∧ d > 2) → 
    (1 * c + 3 = m) → 
    (3 * d + 1 = m) → 
    m ≥ n) ∧
  n = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3320_332008


namespace NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l3320_332071

-- Define the lines l1 and l2 as functions of x and y
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + y + 3 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (3 * m - 2) * x + m * y + 2 = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, l1 m x y ↔ l2 m x y

-- Theorem statement
theorem m_eq_one_sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → parallel m) ∧ 
  (∃ m : ℝ, m ≠ 1 ∧ parallel m) :=
sorry

end NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l3320_332071


namespace NUMINAMATH_CALUDE_existence_of_integers_l3320_332020

theorem existence_of_integers (m : ℕ) (hm : m > 0) :
  ∃ (a b : ℤ),
    (abs a ≤ m) ∧
    (abs b ≤ m) ∧
    (0 < a + b * Real.sqrt 2) ∧
    (a + b * Real.sqrt 2 ≤ (1 + Real.sqrt 2) / (m + 2)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l3320_332020


namespace NUMINAMATH_CALUDE_distance_AB_when_parallel_coordinates_C_when_perpendicular_l3320_332036

-- Define the points A, B, C in the Cartesian coordinate system
def A (a : ℝ) : ℝ × ℝ := (-2, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a - 1, 4)
def C (b : ℝ) : ℝ × ℝ := (b - 2, b)

-- Define the condition that AB is parallel to x-axis
def AB_parallel_x (a : ℝ) : Prop := (A a).2 = (B a).2

-- Define the condition that CD is perpendicular to x-axis
def CD_perpendicular_x (b : ℝ) : Prop := (C b).1 = b - 2

-- Define the condition that CD = 1
def CD_length_1 (b : ℝ) : Prop := (C b).2 - 0 = 1 ∨ (C b).2 - 0 = -1

-- Theorem for part 1
theorem distance_AB_when_parallel (a : ℝ) :
  AB_parallel_x a → (B a).1 - (A a).1 = 4 :=
sorry

-- Theorem for part 2
theorem coordinates_C_when_perpendicular (b : ℝ) :
  CD_perpendicular_x b ∧ CD_length_1 b →
  C b = (-1, 1) ∨ C b = (-3, -1) :=
sorry

end NUMINAMATH_CALUDE_distance_AB_when_parallel_coordinates_C_when_perpendicular_l3320_332036


namespace NUMINAMATH_CALUDE_jerry_logs_count_l3320_332035

/-- The number of logs produced by a pine tree -/
def logsPerPine : ℕ := 80

/-- The number of logs produced by a maple tree -/
def logsPerMaple : ℕ := 60

/-- The number of logs produced by a walnut tree -/
def logsPerWalnut : ℕ := 100

/-- The number of pine trees Jerry cuts -/
def pineTreesCut : ℕ := 8

/-- The number of maple trees Jerry cuts -/
def mapleTreesCut : ℕ := 3

/-- The number of walnut trees Jerry cuts -/
def walnutTreesCut : ℕ := 4

/-- The total number of logs Jerry gets -/
def totalLogs : ℕ := logsPerPine * pineTreesCut + logsPerMaple * mapleTreesCut + logsPerWalnut * walnutTreesCut

theorem jerry_logs_count : totalLogs = 1220 := by sorry

end NUMINAMATH_CALUDE_jerry_logs_count_l3320_332035


namespace NUMINAMATH_CALUDE_triangle_problem_l3320_332000

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Condition 1: Vectors are parallel
  (2 * Real.sin (t.A / 2) * (2 * Real.cos (t.A / 4)^2 - 1) = Real.sqrt 3 * Real.cos t.A) →
  -- Condition 2: a = √7
  (t.a = Real.sqrt 7) →
  -- Condition 3: Area of triangle ABC is 3√3/2
  (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) →
  -- Conclusion 1: A = π/3
  (t.A = Real.pi / 3) ∧
  -- Conclusion 2: b + c = 5
  (t.b + t.c = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3320_332000


namespace NUMINAMATH_CALUDE_amount_lent_to_C_l3320_332045

/-- The amount of money A lent to B in rupees -/
def amount_B : ℝ := 5000

/-- The duration of B's loan in years -/
def duration_B : ℝ := 2

/-- The duration of C's loan in years -/
def duration_C : ℝ := 4

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.07000000000000001

/-- The total interest received from both B and C in rupees -/
def total_interest : ℝ := 1540

/-- The amount of money A lent to C in rupees -/
def amount_C : ℝ := 3000

/-- Theorem stating that given the conditions, A lent 3000 rupees to C -/
theorem amount_lent_to_C : 
  amount_B * interest_rate * duration_B + 
  amount_C * interest_rate * duration_C = total_interest :=
by sorry

end NUMINAMATH_CALUDE_amount_lent_to_C_l3320_332045


namespace NUMINAMATH_CALUDE_min_value_theorem_l3320_332038

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (z : ℝ), (y/x) + (3/y) ≥ z → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3320_332038


namespace NUMINAMATH_CALUDE_stream_speed_l3320_332097

/-- The speed of a stream given boat travel times and distances -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 84) (h2 : upstream_distance = 48) 
  (h3 : time = 2) : ∃ s : ℝ, s = 9 ∧ 
  ∃ b : ℝ, downstream_distance = (b + s) * time ∧ 
           upstream_distance = (b - s) * time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l3320_332097


namespace NUMINAMATH_CALUDE_brick_width_is_10cm_l3320_332010

/-- Proves that the width of each brick is 10 centimeters, given the courtyard dimensions,
    brick length, and total number of bricks. -/
theorem brick_width_is_10cm 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (brick_length : ℝ) 
  (total_bricks : ℕ) 
  (h1 : courtyard_length = 18) 
  (h2 : courtyard_width = 16) 
  (h3 : brick_length = 0.2) 
  (h4 : total_bricks = 14400) : 
  ∃ (brick_width : ℝ), brick_width = 0.1 ∧ 
    courtyard_length * courtyard_width * 100 * 100 = 
    brick_length * brick_width * total_bricks * 10000 :=
by sorry

end NUMINAMATH_CALUDE_brick_width_is_10cm_l3320_332010


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3320_332026

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₈ = 12, then a₅ = 6. -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_sum : a 2 + a 8 = 12) : a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3320_332026


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l3320_332009

theorem recreation_spending_comparison (wages_last_week : ℝ) : 
  let recreation_last_week := 0.60 * wages_last_week
  let wages_this_week := 0.90 * wages_last_week
  let recreation_this_week := 0.70 * wages_this_week
  recreation_this_week / recreation_last_week = 1.05 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l3320_332009


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3320_332073

theorem rectangle_area_proof : ∃ (x y : ℚ), 
  (x - (7/2)) * (y + (3/2)) = x * y ∧ 
  (x + (7/2)) * (y - (5/2)) = x * y ∧ 
  x * y = 20/7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l3320_332073


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l3320_332061

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_weekly_hours schedule)

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 8
  , tue_thu_hours := 6
  , weekly_earnings := 504 }

/-- Theorem stating that Sheila's hourly wage is $14 --/
theorem sheila_hourly_wage :
  hourly_wage sheila_schedule = 14 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_wage_l3320_332061


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3320_332007

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  length : ℝ
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating the width of the metallic sheet given the conditions -/
theorem metallic_sheet_width (sheet : MetallicSheet)
  (h1 : sheet.length = 50)
  (h2 : sheet.cutSize = 8)
  (h3 : sheet.boxVolume = 5440)
  (h4 : sheet.boxVolume = (sheet.length - 2 * sheet.cutSize) * (sheet.width - 2 * sheet.cutSize) * sheet.cutSize) :
  sheet.width = 36 := by
  sorry


end NUMINAMATH_CALUDE_metallic_sheet_width_l3320_332007


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l3320_332095

/-- Proves that a boat traveling 11 km downstream in one hour with a still water speed of 8 km/h
    will travel 5 km upstream in one hour. -/
theorem boat_upstream_distance
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (h1 : boat_speed = 8)
  (h2 : downstream_distance = 11) :
  boat_speed - (downstream_distance - boat_speed) = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_upstream_distance_l3320_332095


namespace NUMINAMATH_CALUDE_max_sides_diagonal_polygon_13gon_l3320_332051

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a convex n-gon -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon formed by diagonals of a larger polygon -/
structure DiagonalPolygon (n : ℕ) where
  sides : ℕ
  sides_le : sides ≤ n

/-- Theorem: In a convex 13-gon with all diagonals drawn, 
    the maximum number of sides of any polygon formed by these diagonals is 13 -/
theorem max_sides_diagonal_polygon_13gon :
  ∀ (p : ConvexPolygon 13) (d : DiagonalPolygon 13),
    d.sides ≤ 13 ∧ ∃ (d' : DiagonalPolygon 13), d'.sides = 13 :=
sorry

end NUMINAMATH_CALUDE_max_sides_diagonal_polygon_13gon_l3320_332051


namespace NUMINAMATH_CALUDE_smallest_base_sum_l3320_332065

theorem smallest_base_sum : ∃ (c d : ℕ), 
  c ≠ d ∧ 
  c > 9 ∧ 
  d > 9 ∧ 
  8 * c + 9 = 9 * d + 8 ∧ 
  c + d = 19 ∧ 
  (∀ (c' d' : ℕ), c' ≠ d' → c' > 9 → d' > 9 → 8 * c' + 9 = 9 * d' + 8 → c' + d' ≥ 19) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_sum_l3320_332065


namespace NUMINAMATH_CALUDE_equal_roots_implies_value_l3320_332024

/-- If x^2 + 2kx + k^2 + k + 3 = 0 has two equal real roots with respect to x,
    then k^2 + k + 3 = 9 -/
theorem equal_roots_implies_value (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*k*x + k^2 + k + 3 = 0 ∧
   ∀ y : ℝ, y^2 + 2*k*y + k^2 + k + 3 = 0 → y = x) →
  k^2 + k + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_implies_value_l3320_332024


namespace NUMINAMATH_CALUDE_parabola_directrix_l3320_332078

/-- Given a parabola with equation y = ax² and directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (2 : ℝ) = -1 / (4 * a) →    -- Directrix equation (in standard form)
  a = -1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3320_332078


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_30_and_18_l3320_332027

theorem arithmetic_mean_of_30_and_18 : (30 + 18) / 2 = 24 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_30_and_18_l3320_332027


namespace NUMINAMATH_CALUDE_mary_warmth_duration_l3320_332044

/-- The number of sticks of wood produced by chopping up furniture -/
def sticksFromFurniture (chairs tables stools : ℕ) : ℕ :=
  6 * chairs + 9 * tables + 2 * stools

/-- The number of hours Mary can keep warm given a certain amount of wood -/
def hoursWarm (totalSticks burningRate : ℕ) : ℕ :=
  totalSticks / burningRate

/-- Theorem: Mary can keep warm for 34 hours with the wood from 18 chairs, 6 tables, and 4 stools -/
theorem mary_warmth_duration :
  let totalSticks := sticksFromFurniture 18 6 4
  let burningRate := 5
  hoursWarm totalSticks burningRate = 34 := by
  sorry

end NUMINAMATH_CALUDE_mary_warmth_duration_l3320_332044


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3320_332090

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + Complex.I) / (1 - Complex.I)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3320_332090


namespace NUMINAMATH_CALUDE_joshua_crates_count_l3320_332085

def bottles_per_crate : ℕ := 12
def total_bottles : ℕ := 130
def unpacked_bottles : ℕ := 10

theorem joshua_crates_count :
  (total_bottles - unpacked_bottles) / bottles_per_crate = 10 := by
  sorry

end NUMINAMATH_CALUDE_joshua_crates_count_l3320_332085


namespace NUMINAMATH_CALUDE_email_problem_l3320_332077

theorem email_problem (x : ℚ) : 
  x + x/2 + x/4 + x/8 = 30 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_email_problem_l3320_332077


namespace NUMINAMATH_CALUDE_small_cubes_count_l3320_332089

/-- Given a cube with edge length 9 cm cut into smaller cubes with edge length 3 cm,
    the number of small cubes obtained is 27. -/
theorem small_cubes_count (large_edge : ℕ) (small_edge : ℕ) : 
  large_edge = 9 → small_edge = 3 → (large_edge / small_edge) ^ 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_small_cubes_count_l3320_332089


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l3320_332053

/-- Represents the number of varieties in a category -/
structure Category where
  varieties : ℕ

/-- Represents the total population of varieties -/
def total_population (categories : List Category) : ℕ :=
  categories.map (·.varieties) |> List.sum

/-- Calculates the number of items in a stratified sample for a given category -/
def stratified_sample_size (category : Category) (total_pop : ℕ) (sample_size : ℕ) : ℕ :=
  (category.varieties * sample_size) / total_pop

/-- Theorem: The sum of vegetable oils and fruits/vegetables in a stratified sample is 6 -/
theorem stratified_sample_sum (vegetable_oils fruits_vegetables : Category)
    (h1 : vegetable_oils.varieties = 10)
    (h2 : fruits_vegetables.varieties = 20)
    (h3 : total_population [vegetable_oils, fruits_vegetables] = 30)
    (h4 : total_population [Category.mk 40, vegetable_oils, Category.mk 30, fruits_vegetables] = 100) :
    stratified_sample_size vegetable_oils 100 20 + stratified_sample_size fruits_vegetables 100 20 = 6 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_sum_l3320_332053


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3320_332096

/-- An isosceles triangle with sides of length 8 and 3 has a perimeter of 19 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 8 ∧ b = 8 ∧ c = 3 →
  a + b > c ∧ b + c > a ∧ a + c > b →
  a + b + c = 19 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3320_332096


namespace NUMINAMATH_CALUDE_max_expected_value_l3320_332056

/-- The probability of winning when there are n red balls and 5 white balls -/
def probability (n : ℕ) : ℚ :=
  (10 * n) / ((n + 5) * (n + 4))

/-- The expected value of the game when there are n red balls -/
def expected_value (n : ℕ) : ℚ :=
  2 * probability n - 1

/-- Theorem stating that the expected value is maximized when n is 4 or 5 -/
theorem max_expected_value :
  ∀ n : ℕ, n > 0 → (expected_value n ≤ expected_value 4 ∧ expected_value n ≤ expected_value 5) :=
by sorry

end NUMINAMATH_CALUDE_max_expected_value_l3320_332056


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l3320_332049

theorem pure_imaginary_solutions_of_polynomial :
  let p (x : ℂ) := x^4 - 4*x^3 + 10*x^2 - 40*x - 100
  ∀ x : ℂ, (∃ a : ℝ, x = Complex.I * a) ∧ p x = 0 ↔ x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l3320_332049


namespace NUMINAMATH_CALUDE_age_problem_l3320_332074

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 3 * d →
  a + b + c + d = 87 →
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3320_332074


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3320_332066

theorem area_between_concentric_circles (R : ℝ) (chord_length : ℝ) : 
  R = 13 → 
  chord_length = 24 → 
  (π * R^2) - (π * (R^2 - (chord_length/2)^2)) = 144 * π :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3320_332066


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l3320_332054

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 - 4*x + y^2 = 0) :
  ∃ (max : ℝ), (∀ (x' y' : ℝ), x'^2 - 4*x' + y'^2 = 0 → x' - 2*y' ≤ max) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀^2 - 4*x₀ + y₀^2 = 0 ∧ x₀ - 2*y₀ = max) ∧ 
  max = 2*Real.sqrt 5 + 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l3320_332054


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l3320_332015

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20]) 
  (h2 : 4 * x ≡ 12 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l3320_332015


namespace NUMINAMATH_CALUDE_sphere_radius_and_area_l3320_332032

/-- A sphere with a chord creating a hollow on its surface -/
structure SphereWithHollow where
  radius : ℝ
  hollowDiameter : ℝ
  hollowDepth : ℝ

/-- The theorem about the sphere's radius and surface area given the hollow dimensions -/
theorem sphere_radius_and_area (s : SphereWithHollow) 
  (h1 : s.hollowDiameter = 12)
  (h2 : s.hollowDepth = 2) :
  s.radius = 10 ∧ 4 * Real.pi * s.radius^2 = 400 * Real.pi := by
  sorry

#check sphere_radius_and_area

end NUMINAMATH_CALUDE_sphere_radius_and_area_l3320_332032


namespace NUMINAMATH_CALUDE_integral_p_equals_one_l3320_332092

noncomputable def p (α : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else α * Real.exp (-α * x)

theorem integral_p_equals_one (α : ℝ) (h : α > 0) :
  ∫ (x : ℝ), p α x = 1 := by sorry

end NUMINAMATH_CALUDE_integral_p_equals_one_l3320_332092


namespace NUMINAMATH_CALUDE_at_least_two_satisfying_functions_l3320_332001

/-- A function satisfying the given equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + (f y)^2) = x + y^2

/-- Theorem stating that there are at least two distinct functions satisfying the equation -/
theorem at_least_two_satisfying_functions :
  ∃ f g : ℝ → ℝ, f ≠ g ∧ SatisfyingFunction f ∧ SatisfyingFunction g :=
sorry

end NUMINAMATH_CALUDE_at_least_two_satisfying_functions_l3320_332001


namespace NUMINAMATH_CALUDE_power_difference_seven_l3320_332050

theorem power_difference_seven (n k : ℕ) : 2^n - 5^k = 7 ↔ n = 5 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_seven_l3320_332050


namespace NUMINAMATH_CALUDE_percentage_problem_l3320_332064

/-- Prove that the percentage is 50% given the conditions -/
theorem percentage_problem (x : ℝ) (a : ℝ) : 
  (x / 100) * a = 95 → a = 190 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3320_332064


namespace NUMINAMATH_CALUDE_department_store_discount_rate_l3320_332029

/-- Represents the discount rate calculation for a department store purchase --/
theorem department_store_discount_rate : 
  -- Define the prices of items
  let shoe_price : ℚ := 74
  let sock_price : ℚ := 2
  let bag_price : ℚ := 42
  let sock_quantity : ℕ := 2
  
  -- Calculate total price before discount
  let total_before_discount : ℚ := shoe_price + sock_price * sock_quantity + bag_price
  
  -- Define the threshold for discount application
  let discount_threshold : ℚ := 100
  
  -- Define the amount paid by Jaco
  let amount_paid : ℚ := 118
  
  -- Calculate the discount amount
  let discount_amount : ℚ := total_before_discount - amount_paid
  
  -- Calculate the amount subject to discount
  let amount_subject_to_discount : ℚ := total_before_discount - discount_threshold
  
  -- Calculate the discount rate
  let discount_rate : ℚ := discount_amount / amount_subject_to_discount * 100
  
  discount_rate = 10 := by sorry

end NUMINAMATH_CALUDE_department_store_discount_rate_l3320_332029


namespace NUMINAMATH_CALUDE_distributive_law_example_l3320_332048

theorem distributive_law_example :
  (7 + 125) * 8 = 7 * 8 + 125 * 8 := by sorry

end NUMINAMATH_CALUDE_distributive_law_example_l3320_332048


namespace NUMINAMATH_CALUDE_binomial_congruence_l3320_332037

theorem binomial_congruence (p m n : ℕ) (hp : Prime p) (h_mn : m ≥ n) :
  (Nat.choose (p * m) (p * n)) ≡ (Nat.choose m n) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_congruence_l3320_332037


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3320_332080

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the base-2 logarithm
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3320_332080


namespace NUMINAMATH_CALUDE_c_share_of_rent_l3320_332011

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given usage -/
def calculateShare (u : Usage) (totalRent : ℕ) (totalUsage : ℕ) : ℚ :=
  (u.oxen * u.months : ℚ) / totalUsage * totalRent

/-- The main theorem stating C's share of the rent -/
theorem c_share_of_rent :
  let a := Usage.mk 10 7
  let b := Usage.mk 12 5
  let c := Usage.mk 15 3
  let totalRent := 210
  let totalUsage := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  calculateShare c totalRent totalUsage = 54 := by
  sorry


end NUMINAMATH_CALUDE_c_share_of_rent_l3320_332011


namespace NUMINAMATH_CALUDE_pants_to_shirts_ratio_l3320_332067

/-- Proves that the ratio of pants to shirts is 1/2 given the problem conditions --/
theorem pants_to_shirts_ratio :
  ∀ (num_pants : ℕ),
  (10 * 6 + num_pants * 8 = 100) →
  (num_pants : ℚ) / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pants_to_shirts_ratio_l3320_332067


namespace NUMINAMATH_CALUDE_pentagon_rectangle_intersection_angle_l3320_332059

-- Define the structure of our problem
structure PentagonWithRectangles where
  -- Regular pentagon
  pentagon_angle : ℝ
  -- Right angles from rectangles
  right_angle1 : ℝ
  right_angle2 : ℝ
  -- Reflex angle
  reflex_angle : ℝ
  -- The angle we're solving for
  x : ℝ

-- Define our theorem
theorem pentagon_rectangle_intersection_angle 
  (p : PentagonWithRectangles) 
  (h1 : p.pentagon_angle = 108)
  (h2 : p.right_angle1 = 90)
  (h3 : p.right_angle2 = 90)
  (h4 : p.reflex_angle = 198)
  (h5 : p.pentagon_angle + p.right_angle1 + p.right_angle2 + p.reflex_angle + p.x = 540) :
  p.x = 54 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_intersection_angle_l3320_332059


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l3320_332004

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l3320_332004


namespace NUMINAMATH_CALUDE_polynomial_equality_l3320_332082

theorem polynomial_equality (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*x - 3 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3320_332082


namespace NUMINAMATH_CALUDE_even_increasing_ordering_l3320_332068

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f is increasing on (0, +∞) if x < y implies f(x) < f(y) for all x, y > 0 -/
def IncreasingOnPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

theorem even_increasing_ordering (f : ℝ → ℝ) 
  (h_even : IsEven f) (h_incr : IncreasingOnPositive f) :
  f 3 < f (-Real.pi) ∧ f (-Real.pi) < f (-4) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_ordering_l3320_332068


namespace NUMINAMATH_CALUDE_farm_ratio_after_transaction_l3320_332031

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the ratio of horses to cows -/
structure Ratio where
  horses : ℕ
  cows : ℕ

def initial_ratio : Ratio := { horses := 5, cows := 1 }

def transaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

theorem farm_ratio_after_transaction (farm : FarmAnimals) :
  farm.horses = 5 * farm.cows →
  (transaction farm).horses = (transaction farm).cows + 50 →
  ∃ (k : ℕ), k > 0 ∧ (transaction farm).horses = 17 * k ∧ (transaction farm).cows = 7 * k :=
sorry

end NUMINAMATH_CALUDE_farm_ratio_after_transaction_l3320_332031


namespace NUMINAMATH_CALUDE_no_integer_solution_l3320_332087

theorem no_integer_solution : ¬ ∃ (m n : ℤ), m^3 = 4*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3320_332087


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3320_332016

theorem complex_expression_equality : 
  let x := (3 + 3/8)^(2/3) - (5 + 4/9)^(1/2) + 0.008^(2/3) / 0.02^(1/2) * 0.32^(1/2)
  x / 0.0625^(1/4) = 23/150 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3320_332016


namespace NUMINAMATH_CALUDE_representation_of_1917_l3320_332018

theorem representation_of_1917 : ∃ (a b c : ℤ), 1917 = a^2 - b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_representation_of_1917_l3320_332018
