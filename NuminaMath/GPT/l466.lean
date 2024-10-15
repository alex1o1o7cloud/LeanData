import Mathlib

namespace NUMINAMATH_GPT_factorial_div_eq_l466_46638
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_factorial_div_eq_l466_46638


namespace NUMINAMATH_GPT_Mia_and_dad_time_to_organize_toys_l466_46657

theorem Mia_and_dad_time_to_organize_toys :
  let total_toys := 60
  let dad_add_rate := 6
  let mia_remove_rate := 4
  let net_gain_per_cycle := dad_add_rate - mia_remove_rate
  let seconds_per_cycle := 30
  let total_needed_cycles := (total_toys - 2) / net_gain_per_cycle -- 58 toys by the end of repeated cycles, 2 is to ensure dad's last placement
  let last_cycle_time := seconds_per_cycle
  let total_time_seconds := total_needed_cycles * seconds_per_cycle + last_cycle_time
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 15 :=
by
  sorry

end NUMINAMATH_GPT_Mia_and_dad_time_to_organize_toys_l466_46657


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l466_46628

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l466_46628


namespace NUMINAMATH_GPT_expected_value_of_girls_left_of_boys_l466_46602

def num_girls_to_left_of_all_boys (boys girls : ℕ) : ℚ :=
  (boys + girls : ℚ) / (boys + 1)

theorem expected_value_of_girls_left_of_boys :
  num_girls_to_left_of_all_boys 10 7 = 7 / 11 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_girls_left_of_boys_l466_46602


namespace NUMINAMATH_GPT_circle_area_l466_46689

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end NUMINAMATH_GPT_circle_area_l466_46689


namespace NUMINAMATH_GPT_alpha_in_third_quadrant_l466_46646

theorem alpha_in_third_quadrant (α : ℝ)
 (h₁ : Real.tan (α - 3 * Real.pi) > 0)
 (h₂ : Real.sin (-α + Real.pi) < 0) :
 (0 < α % (2 * Real.pi) ∧ α % (2 * Real.pi) < Real.pi) := 
sorry

end NUMINAMATH_GPT_alpha_in_third_quadrant_l466_46646


namespace NUMINAMATH_GPT_quotient_division_l466_46607

/-- Definition of the condition that when 14 is divided by 3, the remainder is 2 --/
def division_property : Prop :=
  14 = 3 * (14 / 3) + 2

/-- Statement for finding the quotient when 14 is divided by 3 --/
theorem quotient_division (A : ℕ) (h : 14 = 3 * A + 2) : A = 4 :=
by
  have rem_2 := division_property
  sorry

end NUMINAMATH_GPT_quotient_division_l466_46607


namespace NUMINAMATH_GPT_bob_start_time_l466_46624

-- Define constants for the problem conditions
def yolandaRate : ℝ := 3 -- Yolanda's walking rate in miles per hour
def bobRate : ℝ := 4 -- Bob's walking rate in miles per hour
def distanceXY : ℝ := 10 -- Distance between point X and Y in miles
def bobDistanceWhenMet : ℝ := 4 -- Distance Bob had walked when they met in miles

-- Define the theorem statement
theorem bob_start_time : 
  ∃ T : ℝ, (yolandaRate * T + bobDistanceWhenMet = distanceXY) →
  (T = 2) →
  ∃ tB : ℝ, T - tB = 1 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_bob_start_time_l466_46624


namespace NUMINAMATH_GPT_smallest_divisible_by_15_11_12_l466_46643

theorem smallest_divisible_by_15_11_12 : ∃ n : ℕ, (n > 0) ∧ (15 ∣ n) ∧ (11 ∣ n) ∧ (12 ∣ n) ∧ (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (11 ∣ m) ∧ (12 ∣ m) → n ≤ m) ∧ n = 660 :=
by
  sorry

end NUMINAMATH_GPT_smallest_divisible_by_15_11_12_l466_46643


namespace NUMINAMATH_GPT_Karlee_initial_grapes_l466_46694

theorem Karlee_initial_grapes (G S Remaining_Fruits : ℕ)
  (h1 : S = (3 * G) / 5)
  (h2 : Remaining_Fruits = 96)
  (h3 : Remaining_Fruits = (3 * G) / 5 + (9 * G) / 25) :
  G = 100 := by
  -- add proof here
  sorry

end NUMINAMATH_GPT_Karlee_initial_grapes_l466_46694


namespace NUMINAMATH_GPT_pentagon_PT_value_l466_46664

-- Given conditions
def length_QR := 3
def length_RS := 3
def length_ST := 3
def angle_T := 90
def angle_P := 120
def angle_Q := 120
def angle_R := 120

-- The target statement to prove
theorem pentagon_PT_value (a b : ℝ) (h : PT = a + 3 * Real.sqrt b) : a + b = 6 :=
sorry

end NUMINAMATH_GPT_pentagon_PT_value_l466_46664


namespace NUMINAMATH_GPT_apples_in_each_box_l466_46616

theorem apples_in_each_box (x : ℕ) :
  (5 * x - (60 * 5)) = (2 * x) -> x = 100 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_each_box_l466_46616


namespace NUMINAMATH_GPT_pool_half_capacity_at_6_hours_l466_46660

noncomputable def double_volume_every_hour (t : ℕ) : ℕ := 2 ^ t

theorem pool_half_capacity_at_6_hours (V : ℕ) (h : ∀ t : ℕ, V = double_volume_every_hour 8) : double_volume_every_hour 6 = V / 2 := by
  sorry

end NUMINAMATH_GPT_pool_half_capacity_at_6_hours_l466_46660


namespace NUMINAMATH_GPT_largest_a_for_integer_solution_l466_46619

theorem largest_a_for_integer_solution :
  ∃ a : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a * x + 3 * y = 1) ∧ (∀ a' : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a' * x + 3 * y = 1) → a' ≤ a) ∧ a = 1 :=
sorry

end NUMINAMATH_GPT_largest_a_for_integer_solution_l466_46619


namespace NUMINAMATH_GPT_gcd_paving_courtyard_l466_46685

theorem gcd_paving_courtyard :
  Nat.gcd 378 595 = 7 :=
by
  sorry

end NUMINAMATH_GPT_gcd_paving_courtyard_l466_46685


namespace NUMINAMATH_GPT_simplify_expression_l466_46641

variable (a b c : ℝ) 

theorem simplify_expression (h1 : a ≠ 4) (h2 : b ≠ 5) (h3 : c ≠ 6) :
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l466_46641


namespace NUMINAMATH_GPT_tan_add_l466_46676

open Real

-- Define positive acute angles
def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

-- Theorem: Tangent addition formula
theorem tan_add (α β : ℝ) (hα : acute_angle α) (hβ : acute_angle β) :
  tan (α + β) = (tan α + tan β) / (1 - tan α * tan β) :=
  sorry

end NUMINAMATH_GPT_tan_add_l466_46676


namespace NUMINAMATH_GPT_water_level_balance_l466_46666

noncomputable def exponential_decay (a n t : ℝ) : ℝ := a * Real.exp (n * t)

theorem water_level_balance
  (a : ℝ)
  (n : ℝ)
  (m : ℝ)
  (h5 : exponential_decay a n 5 = a / 2)
  (h8 : exponential_decay a n m = a / 8) :
  m = 10 := by
  sorry

end NUMINAMATH_GPT_water_level_balance_l466_46666


namespace NUMINAMATH_GPT_problem_1_l466_46667

theorem problem_1 :
  (-7/4) - (19/3) - 9/4 + 10/3 = -7 := by
  sorry

end NUMINAMATH_GPT_problem_1_l466_46667


namespace NUMINAMATH_GPT_train_passes_man_in_4_4_seconds_l466_46613

noncomputable def train_speed_kmph : ℝ := 84
noncomputable def man_speed_kmph : ℝ := 6
noncomputable def train_length_m : ℝ := 110

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def man_speed_mps : ℝ :=
  kmph_to_mps man_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps + man_speed_mps

noncomputable def passing_time : ℝ :=
  train_length_m / relative_speed_mps

theorem train_passes_man_in_4_4_seconds :
  passing_time = 4.4 :=
by
  sorry -- Proof not required, skipping the proof logic

end NUMINAMATH_GPT_train_passes_man_in_4_4_seconds_l466_46613


namespace NUMINAMATH_GPT_max_perimeter_of_triangle_l466_46649

theorem max_perimeter_of_triangle (x : ℕ) 
  (h1 : 3 < x) 
  (h2 : x < 15) 
  (h3 : 7 + 8 > x) 
  (h4 : 7 + x > 8) 
  (h5 : 8 + x > 7) :
  x = 14 ∧ 7 + 8 + x = 29 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_perimeter_of_triangle_l466_46649


namespace NUMINAMATH_GPT_expansion_l466_46663

variable (x : ℝ)

noncomputable def expr : ℝ := (3 / 4) * (8 / (x^2) + 5 * x - 6)

theorem expansion :
  expr x = (6 / (x^2)) + (15 * x / 4) - 4.5 :=
by
  sorry

end NUMINAMATH_GPT_expansion_l466_46663


namespace NUMINAMATH_GPT_daisies_bought_l466_46618

theorem daisies_bought (cost_per_flower roses total_cost : ℕ) 
  (h1 : cost_per_flower = 3) 
  (h2 : roses = 8) 
  (h3 : total_cost = 30) : 
  (total_cost - (roses * cost_per_flower)) / cost_per_flower = 2 :=
by
  sorry

end NUMINAMATH_GPT_daisies_bought_l466_46618


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l466_46639

theorem arithmetic_geometric_sequence :
  ∀ (a₁ a₂ b₂ : ℝ),
    -- Conditions for arithmetic sequence: -1, a₁, a₂, 8
    2 * a₁ = -1 + a₂ ∧
    2 * a₂ = a₁ + 8 →
    -- Conditions for geometric sequence: -1, b₁, b₂, b₃, -4
    (∃ (b₁ b₃ : ℝ), b₁^2 = b₂ ∧ b₁ != 0 ∧ -4 * b₁^4 = b₂ → -1 * b₁ = b₃) →
    -- Goal: Calculate and prove the value
    (a₁ * a₂ / b₂) = -5 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l466_46639


namespace NUMINAMATH_GPT_marie_daily_rent_l466_46669

noncomputable def daily_revenue (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ) : ℝ :=
  bread_loaves * bread_price + cakes * cake_price

noncomputable def total_profit (daily_revenue : ℝ) (days : ℕ) (cash_register_cost : ℝ) : ℝ :=
  cash_register_cost

noncomputable def daily_profit (total_profit : ℝ) (days : ℕ) : ℝ :=
  total_profit / days

noncomputable def daily_profit_after_electricity (daily_profit : ℝ) (electricity_cost : ℝ) : ℝ :=
  daily_profit - electricity_cost

noncomputable def daily_rent (daily_revenue : ℝ) (daily_profit_after_electricity : ℝ) : ℝ :=
  daily_revenue - daily_profit_after_electricity

theorem marie_daily_rent
  (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ)
  (days : ℕ) (cash_register_cost : ℝ) (electricity_cost : ℝ) :
  bread_loaves = 40 → bread_price = 2 → cakes = 6 → cake_price = 12 →
  days = 8 → cash_register_cost = 1040 → electricity_cost = 2 →
  daily_rent (daily_revenue bread_loaves bread_price cakes cake_price)
             (daily_profit_after_electricity (daily_profit (total_profit (daily_revenue bread_loaves bread_price cakes cake_price) days cash_register_cost) days) electricity_cost) = 24 :=
by
  intros h0 h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_marie_daily_rent_l466_46669


namespace NUMINAMATH_GPT_wire_cut_l466_46627

theorem wire_cut (total_length : ℝ) (ratio : ℝ) (shorter longer : ℝ) (h_total : total_length = 21) (h_ratio : ratio = 2/5)
  (h_shorter : longer = (5/2) * shorter) (h_sum : total_length = shorter + longer) : shorter = 6 := 
by
  -- total_length = 21, ratio = 2/5, longer = (5/2) * shorter, total_length = shorter + longer, prove shorter = 6
  sorry

end NUMINAMATH_GPT_wire_cut_l466_46627


namespace NUMINAMATH_GPT_no_such_continuous_function_exists_l466_46673

theorem no_such_continuous_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (Continuous f) ∧ ∀ x : ℝ, ((∃ q : ℚ, f x = q) ↔ ∀ q' : ℚ, f (x + 1) ≠ q') :=
sorry

end NUMINAMATH_GPT_no_such_continuous_function_exists_l466_46673


namespace NUMINAMATH_GPT_inequality_x_y_z_l466_46611

-- Definitions for the variables
variables {x y z : ℝ} 
variable {n : ℕ}

-- Positive numbers and summation condition
axiom h1 : 0 < x ∧ 0 < y ∧ 0 < z
axiom h2 : x + y + z = 1

-- The theorem to be proven
theorem inequality_x_y_z (h1 : 0 < x ∧ 0 < y ∧ 0 < z) (h2 : x + y + z = 1) (hn : n > 0) : 
  x^n + y^n + z^n ≥ (1 : ℝ) / (3:ℝ)^(n-1) :=
sorry

end NUMINAMATH_GPT_inequality_x_y_z_l466_46611


namespace NUMINAMATH_GPT_shopkeeper_oranges_l466_46698

theorem shopkeeper_oranges (O : ℕ) 
  (bananas : ℕ) 
  (percent_rotten_oranges : ℕ) 
  (percent_rotten_bananas : ℕ) 
  (percent_good_condition : ℚ) 
  (h1 : bananas = 400) 
  (h2 : percent_rotten_oranges = 15) 
  (h3 : percent_rotten_bananas = 6) 
  (h4 : percent_good_condition = 88.6) : 
  O = 600 :=
by
  -- This proof needs to be filled in.
  sorry

end NUMINAMATH_GPT_shopkeeper_oranges_l466_46698


namespace NUMINAMATH_GPT_abcd_zero_l466_46671

theorem abcd_zero (a b c d : ℝ) (h1 : a + b + c + d = 0) (h2 : ab + ac + bc + bd + ad + cd = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end NUMINAMATH_GPT_abcd_zero_l466_46671


namespace NUMINAMATH_GPT_find_x_plus_one_over_x_l466_46622

variable (x : ℝ)

theorem find_x_plus_one_over_x
  (h1 : x^3 + (1/x)^3 = 110)
  (h2 : (x + 1/x)^2 - 2*x - 2*(1/x) = 38) :
  x + 1/x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_plus_one_over_x_l466_46622


namespace NUMINAMATH_GPT_average_salary_all_workers_l466_46645

-- Definitions based on the conditions
def num_technicians : ℕ := 7
def num_other_workers : ℕ := 7
def avg_salary_technicians : ℕ := 12000
def avg_salary_other_workers : ℕ := 8000
def total_workers : ℕ := 14

-- Total salary calculations based on the conditions
def total_salary_technicians : ℕ := num_technicians * avg_salary_technicians
def total_salary_other_workers : ℕ := num_other_workers * avg_salary_other_workers
def total_salary_all_workers : ℕ := total_salary_technicians + total_salary_other_workers

-- The statement to be proved
theorem average_salary_all_workers : total_salary_all_workers / total_workers = 10000 :=
by
  -- proof will be added here
  sorry

end NUMINAMATH_GPT_average_salary_all_workers_l466_46645


namespace NUMINAMATH_GPT_sqrt_mul_neg_eq_l466_46629

theorem sqrt_mul_neg_eq : - (Real.sqrt 2) * (Real.sqrt 7) = - (Real.sqrt 14) := sorry

end NUMINAMATH_GPT_sqrt_mul_neg_eq_l466_46629


namespace NUMINAMATH_GPT_total_weight_collected_l466_46699

def GinaCollectedBags : ℕ := 8
def NeighborhoodFactor : ℕ := 120
def WeightPerBag : ℕ := 6

theorem total_weight_collected :
  (GinaCollectedBags * NeighborhoodFactor + GinaCollectedBags) * WeightPerBag = 5808 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_collected_l466_46699


namespace NUMINAMATH_GPT_term_in_census_is_population_l466_46609

def term_for_entire_set_of_objects : String :=
  "population"

theorem term_in_census_is_population :
  term_for_entire_set_of_objects = "population" :=
sorry

end NUMINAMATH_GPT_term_in_census_is_population_l466_46609


namespace NUMINAMATH_GPT_solve_x_l466_46662

theorem solve_x (x : ℝ) (h : 9 - 4 / x = 7 + 8 / x) : x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_solve_x_l466_46662


namespace NUMINAMATH_GPT_probability_all_same_color_l466_46623

theorem probability_all_same_color :
  let red_plates := 7
  let blue_plates := 5
  let total_plates := red_plates + blue_plates
  let total_combinations := Nat.choose total_plates 3
  let red_combinations := Nat.choose red_plates 3
  let blue_combinations := Nat.choose blue_plates 3
  let favorable_combinations := red_combinations + blue_combinations
  let probability := (favorable_combinations : ℚ) / total_combinations
  probability = 9 / 44 :=
by 
  sorry

end NUMINAMATH_GPT_probability_all_same_color_l466_46623


namespace NUMINAMATH_GPT_complement_union_l466_46686

open Set

namespace ProofFormalization

/-- Declaration of the universal set U, and sets A and B -/
def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

def complement {α : Type*} (s t : Set α) : Set α := t \ s

/-- Theorem statement that proves the complement of A ∪ B with respect to U is {5} -/
theorem complement_union :
  complement (A ∪ B) U = {5} :=
by
  sorry

end ProofFormalization

end NUMINAMATH_GPT_complement_union_l466_46686


namespace NUMINAMATH_GPT_cost_of_paving_l466_46690

noncomputable def length : Float := 5.5
noncomputable def width : Float := 3.75
noncomputable def cost_per_sq_meter : Float := 600

theorem cost_of_paving :
  (length * width * cost_per_sq_meter) = 12375 := by
  sorry

end NUMINAMATH_GPT_cost_of_paving_l466_46690


namespace NUMINAMATH_GPT_solve_for_x_l466_46621

theorem solve_for_x (x : ℝ) (h : 9 / (1 + 4 / x) = 1) : x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l466_46621


namespace NUMINAMATH_GPT_eval_expression_l466_46633

-- Define the given expression
def given_expression : ℤ := -( (16 / 2) * 12 - 75 + 4 * (2 * 5) + 25 )

-- State the desired result in a theorem
theorem eval_expression : given_expression = -86 := by
  -- Skipping the proof as per instructions
  sorry

end NUMINAMATH_GPT_eval_expression_l466_46633


namespace NUMINAMATH_GPT_deepak_age_l466_46642

variable (A D : ℕ)

theorem deepak_age (h1 : A / D = 2 / 3) (h2 : A + 5 = 25) : D = 30 :=
sorry

end NUMINAMATH_GPT_deepak_age_l466_46642


namespace NUMINAMATH_GPT_sum_of_first_6033_terms_l466_46640

noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms
  (a r : ℝ)  
  (h1 : geometric_series_sum a r 2011 = 200)
  (h2 : geometric_series_sum a r 4022 = 380) :
  geometric_series_sum a r 6033 = 542 := 
sorry

end NUMINAMATH_GPT_sum_of_first_6033_terms_l466_46640


namespace NUMINAMATH_GPT_carl_city_mileage_l466_46656

noncomputable def city_mileage (miles_city mpg_highway cost_per_gallon total_cost miles_highway : ℝ) : ℝ :=
  let total_gallons := total_cost / cost_per_gallon
  let gallons_highway := miles_highway / mpg_highway
  let gallons_city := total_gallons - gallons_highway
  miles_city / gallons_city

theorem carl_city_mileage :
  city_mileage 60 40 3 42 200 = 20 / 3 := by
  sorry

end NUMINAMATH_GPT_carl_city_mileage_l466_46656


namespace NUMINAMATH_GPT_problem_l466_46668

def otimes (x y : ℝ) : ℝ := x^3 + 5 * x * y - y

theorem problem (a : ℝ) : 
  otimes a (otimes a a) = 5 * a^4 + 24 * a^3 - 10 * a^2 + a :=
by
  sorry

end NUMINAMATH_GPT_problem_l466_46668


namespace NUMINAMATH_GPT_bobs_total_profit_l466_46670

theorem bobs_total_profit :
  let cost_parent_dog := 250
  let num_parent_dogs := 2
  let num_puppies := 6
  let cost_food_vaccinations := 500
  let cost_advertising := 150
  let selling_price_parent_dog := 200
  let selling_price_puppy := 350
  let total_cost_parent_dogs := num_parent_dogs * cost_parent_dog
  let total_cost_puppies := cost_food_vaccinations + cost_advertising
  let total_revenue_puppies := num_puppies * selling_price_puppy
  let total_revenue_parent_dogs := num_parent_dogs * selling_price_parent_dog
  let total_revenue := total_revenue_puppies + total_revenue_parent_dogs
  let total_cost := total_cost_parent_dogs + total_cost_puppies
  let total_profit := total_revenue - total_cost
  total_profit = 1350 :=
by
  sorry

end NUMINAMATH_GPT_bobs_total_profit_l466_46670


namespace NUMINAMATH_GPT_max_length_segment_l466_46608

theorem max_length_segment (p b : ℝ) (h : b = p / 2) : (b * (p - b)) / p = p / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_length_segment_l466_46608


namespace NUMINAMATH_GPT_arithmetic_sequence_2023rd_term_l466_46604

theorem arithmetic_sequence_2023rd_term 
  (p q : ℤ)
  (h1 : 3 * p - q + 9 = 9)
  (h2 : 3 * (3 * p - q + 9) - q + 9 = 3 * p + q) :
  p + (2023 - 1) * (3 * p - q + 9) = 18189 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_2023rd_term_l466_46604


namespace NUMINAMATH_GPT_gilbert_parsley_count_l466_46632

variable (basil mint parsley : ℕ)
variable (initial_basil : ℕ := 3)
variable (extra_basil : ℕ := 1)
variable (initial_mint : ℕ := 2)
variable (herb_total : ℕ := 5)

def initial_parsley := herb_total - (initial_basil + extra_basil)

theorem gilbert_parsley_count : initial_parsley = 1 := by
  -- basil = initial_basil + extra_basil
  -- mint = 0 (since all mint plants eaten)
  -- herb_total = basil + parsley
  -- 5 = 4 + parsley
  -- parsley = 1
  sorry

end NUMINAMATH_GPT_gilbert_parsley_count_l466_46632


namespace NUMINAMATH_GPT_total_cost_proof_l466_46659

-- Define the cost of items
def cost_of_1kg_of_mango (M : ℚ) : Prop := sorry
def cost_of_1kg_of_rice (R : ℚ) : Prop := sorry
def cost_of_1kg_of_flour (F : ℚ) : Prop := F = 23

-- Condition 1: cost of some kg of mangos is equal to the cost of 24 kg of rice
def condition1 (M R : ℚ) (x : ℚ) : Prop := M * x = R * 24

-- Condition 2: cost of 6 kg of flour equals to the cost of 2 kg of rice
def condition2 (R : ℚ) : Prop := 23 * 6 = R * 2

-- Final proof problem
theorem total_cost_proof (M R F : ℚ) (x : ℚ) 
  (h1: condition1 M R x) 
  (h2: condition2 R) 
  (h3: cost_of_1kg_of_flour F) :
  4 * (69 * 24 / x) + 3 * R + 5 * 23 = 1978 :=
sorry

end NUMINAMATH_GPT_total_cost_proof_l466_46659


namespace NUMINAMATH_GPT_count_multiples_of_12_l466_46631

theorem count_multiples_of_12 (a b : ℤ) (h1 : 15 < a) (h2 : b < 205) (h3 : ∃ k : ℤ, a = 12 * k) (h4 : ∃ k : ℤ, b = 12 * k) : 
  ∃ n : ℕ, n = 16 := 
by 
  sorry

end NUMINAMATH_GPT_count_multiples_of_12_l466_46631


namespace NUMINAMATH_GPT_simple_annual_interest_rate_l466_46658

noncomputable def monthly_interest_payment : ℝ := 216
noncomputable def principal_amount : ℝ := 28800
noncomputable def number_of_months_in_a_year : ℕ := 12

theorem simple_annual_interest_rate :
  ((monthly_interest_payment * number_of_months_in_a_year) / principal_amount) * 100 = 9 := by
sorry

end NUMINAMATH_GPT_simple_annual_interest_rate_l466_46658


namespace NUMINAMATH_GPT_regular_price_of_each_shirt_l466_46678

theorem regular_price_of_each_shirt (P : ℝ) :
    let total_shirts := 20
    let sale_price_per_shirt := 0.8 * P
    let tax_rate := 0.10
    let total_paid := 264
    let total_price := total_shirts * sale_price_per_shirt * (1 + tax_rate)
    total_price = total_paid → P = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_regular_price_of_each_shirt_l466_46678


namespace NUMINAMATH_GPT_problem_l466_46655

theorem problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_l466_46655


namespace NUMINAMATH_GPT_possible_value_of_sum_l466_46681

theorem possible_value_of_sum (p q r : ℝ) (h₀ : q = p * (4 - p)) (h₁ : r = q * (4 - q)) (h₂ : p = r * (4 - r)) 
  (h₃ : p ≠ q ∧ p ≠ r ∧ q ≠ r) : p + q + r = 6 :=
sorry

end NUMINAMATH_GPT_possible_value_of_sum_l466_46681


namespace NUMINAMATH_GPT_initial_dog_cat_ratio_l466_46626

theorem initial_dog_cat_ratio (C : ℕ) :
  75 / (C + 20) = 15 / 11 →
  (75 / C) = 15 / 7 :=
by
  sorry

end NUMINAMATH_GPT_initial_dog_cat_ratio_l466_46626


namespace NUMINAMATH_GPT_circle_diameter_tangents_l466_46648

open Real

theorem circle_diameter_tangents {x y : ℝ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) :
  ∃ d : ℝ, d = sqrt (x * y) :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_tangents_l466_46648


namespace NUMINAMATH_GPT_minimum_overlap_l466_46600

variable (U : Finset ℕ) -- This is the set of all people surveyed
variable (B V : Finset ℕ) -- These are the sets of people who like Beethoven and Vivaldi respectively.

-- Given conditions:
axiom h_total : U.card = 120
axiom h_B : B.card = 95
axiom h_V : V.card = 80
axiom h_subset_B : B ⊆ U
axiom h_subset_V : V ⊆ U

-- Question to prove:
theorem minimum_overlap : (B ∩ V).card = 95 + 80 - 120 := by
  sorry

end NUMINAMATH_GPT_minimum_overlap_l466_46600


namespace NUMINAMATH_GPT_functional_linear_solution_l466_46612

variable (f : ℝ → ℝ)

theorem functional_linear_solution (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_GPT_functional_linear_solution_l466_46612


namespace NUMINAMATH_GPT_new_sign_cost_l466_46661

theorem new_sign_cost 
  (p_s : ℕ) (p_c : ℕ) (n : ℕ) (h_ps : p_s = 30) (h_pc : p_c = 26) (h_n : n = 10) : 
  (p_s - p_c) * n / 2 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_new_sign_cost_l466_46661


namespace NUMINAMATH_GPT_simplify_fraction_l466_46634

theorem simplify_fraction (b : ℕ) (hb : b = 5) : (15 * b^4) / (90 * b^3 * b) = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l466_46634


namespace NUMINAMATH_GPT_car_owners_without_motorcycles_l466_46683

theorem car_owners_without_motorcycles 
    (total_adults : ℕ) 
    (car_owners : ℕ) 
    (motorcycle_owners : ℕ) 
    (total_with_vehicles : total_adults = 500) 
    (total_car_owners : car_owners = 480) 
    (total_motorcycle_owners : motorcycle_owners = 120) : 
    car_owners - (car_owners + motorcycle_owners - total_adults) = 380 := 
by
    sorry

end NUMINAMATH_GPT_car_owners_without_motorcycles_l466_46683


namespace NUMINAMATH_GPT_fly_travel_distance_l466_46620

theorem fly_travel_distance
  (carA_speed : ℕ)
  (carB_speed : ℕ)
  (initial_distance : ℕ)
  (fly_speed : ℕ)
  (relative_speed : ℕ := carB_speed - carA_speed)
  (catchup_time : ℚ := initial_distance / relative_speed)
  (fly_travel : ℚ := fly_speed * catchup_time) :
  carA_speed = 20 → carB_speed = 30 → initial_distance = 1 → fly_speed = 40 → fly_travel = 4 :=
by
  sorry

end NUMINAMATH_GPT_fly_travel_distance_l466_46620


namespace NUMINAMATH_GPT_odd_function_a_minus_b_l466_46636

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_a_minus_b
  (a b : ℝ)
  (h : is_odd_function (λ x => 2 * x ^ 3 + a * x ^ 2 + b - 1)) :
  a - b = -1 :=
sorry

end NUMINAMATH_GPT_odd_function_a_minus_b_l466_46636


namespace NUMINAMATH_GPT_angela_problems_l466_46601

theorem angela_problems (M J S K A : ℕ) :
  M = 3 →
  J = (M * M - 5) + ((M * M - 5) / 3) →
  S = 50 / 10 →
  K = (J + S) / 2 →
  A = 50 - (M + J + S + K) →
  A = 32 :=
by
  intros hM hJ hS hK hA
  sorry

end NUMINAMATH_GPT_angela_problems_l466_46601


namespace NUMINAMATH_GPT_proof_problem_l466_46647

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 1

-- Define the circle C with center (h, k) and radius r
def circle_eq (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Define condition of line that intersects the circle C at points A and B
def line_eq (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Condition: OA ⊥ OB
def perpendicular_cond (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem stating the proof problem
theorem proof_problem :
  (∃ (h k r : ℝ),
    circle_eq h k r 3 1 ∧
    circle_eq h k r 5 0 ∧
    circle_eq h k r 1 0 ∧
    h = 3 ∧ k = 1 ∧ r = 3) ∧
    (∃ (a : ℝ),
      (∀ (x1 y1 x2 y2 : ℝ),
        line_eq a x1 y1 ∧
        circle_eq 3 1 3 x1 y1 ∧
        line_eq a x2 y2 ∧
        circle_eq 3 1 3 x2 y2 → 
        perpendicular_cond x1 y1 x2 y2) →
      a = -1) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l466_46647


namespace NUMINAMATH_GPT_cone_height_l466_46637

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end NUMINAMATH_GPT_cone_height_l466_46637


namespace NUMINAMATH_GPT_non_zero_real_x_solution_l466_46679

noncomputable section

variables {x : ℝ} (hx : x ≠ 0)

theorem non_zero_real_x_solution 
  (h : (3 * x)^5 = (9 * x)^4) : 
  x = 27 := by
  sorry

end NUMINAMATH_GPT_non_zero_real_x_solution_l466_46679


namespace NUMINAMATH_GPT_weights_are_equal_l466_46697

variable {n : ℕ}
variables {a : Fin (2 * n + 1) → ℝ}

def weights_condition
    (a : Fin (2 * n + 1) → ℝ) : Prop :=
  ∀ i : Fin (2 * n + 1), ∃ (A B : Finset (Fin (2 * n + 1))),
    A.card = n ∧ B.card = n ∧ A ∩ B = ∅ ∧
    A ∪ B = Finset.univ.erase i ∧
    (A.sum a = B.sum a)

theorem weights_are_equal
    (h : weights_condition a) :
  ∃ k : ℝ, ∀ i : Fin (2 * n + 1), a i = k :=
  sorry

end NUMINAMATH_GPT_weights_are_equal_l466_46697


namespace NUMINAMATH_GPT_cost_per_ton_ice_correct_l466_46693

variables {a p n s : ℝ}

-- Define the cost per ton of ice received by enterprise A
noncomputable def cost_per_ton_ice_received (a p n s : ℝ) : ℝ :=
  (2.5 * a + p * s) * 1000 / (2000 - n * s)

-- The statement of the theorem
theorem cost_per_ton_ice_correct :
  ∀ a p n s : ℝ,
  2000 - n * s ≠ 0 →
  cost_per_ton_ice_received a p n s = (2.5 * a + p * s) * 1000 / (2000 - n * s) := by
  intros a p n s h
  unfold cost_per_ton_ice_received
  sorry

end NUMINAMATH_GPT_cost_per_ton_ice_correct_l466_46693


namespace NUMINAMATH_GPT_round_to_nearest_whole_l466_46630

theorem round_to_nearest_whole (x : ℝ) (hx : x = 7643.498201) : Int.floor (x + 0.5) = 7643 := 
by
  -- To prove
  sorry

end NUMINAMATH_GPT_round_to_nearest_whole_l466_46630


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l466_46688

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 2 ∧ b = 5 ∨ a = 5 ∧ b = 2):
  ∃ c : ℕ, (c = a ∨ c = b) ∧ 2 * c + (if c = a then b else a) = 12 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l466_46688


namespace NUMINAMATH_GPT_initial_extra_planks_l466_46696

-- Definitions corresponding to the conditions
def charlie_planks : Nat := 10
def father_planks : Nat := 10
def total_planks : Nat := 35

-- The proof problem statement
theorem initial_extra_planks : total_planks - (charlie_planks + father_planks) = 15 := by
  sorry

end NUMINAMATH_GPT_initial_extra_planks_l466_46696


namespace NUMINAMATH_GPT_total_days_needed_l466_46614

-- Define the conditions
def project1_questions : ℕ := 518
def project2_questions : ℕ := 476
def questions_per_day : ℕ := 142

-- Define the statement to prove
theorem total_days_needed :
  (project1_questions + project2_questions) / questions_per_day = 7 := by
  sorry

end NUMINAMATH_GPT_total_days_needed_l466_46614


namespace NUMINAMATH_GPT_range_of_a_l466_46606

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * Real.sin x

theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂, (1 + 1 / x₁) * (a - 2 * Real.cos x₂) = -1) →
  -2 ≤ a ∧ a ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l466_46606


namespace NUMINAMATH_GPT_purple_marble_probability_l466_46672

theorem purple_marble_probability (blue green : ℝ) (p : ℝ) 
  (h_blue : blue = 0.25)
  (h_green : green = 0.4)
  (h_sum : blue + green + p = 1) : p = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_purple_marble_probability_l466_46672


namespace NUMINAMATH_GPT_regular_octagon_diagonal_l466_46684

variable {a b c : ℝ}

-- Define a function to check for a regular octagon where a, b, c are respective side, shortest diagonal, and longest diagonal
def is_regular_octagon (a b c : ℝ) : Prop :=
  -- Here, we assume the standard geometric properties of a regular octagon.
  -- In a real formalization, we might model the octagon directly.

  -- longest diagonal c of a regular octagon (spans 4 sides)
  c = 2 * a

theorem regular_octagon_diagonal (a b c : ℝ) (h : is_regular_octagon a b c) : c = 2 * a :=
by
  exact h

end NUMINAMATH_GPT_regular_octagon_diagonal_l466_46684


namespace NUMINAMATH_GPT_negation_equiv_l466_46695

theorem negation_equiv (x : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) := 
by 
  sorry

end NUMINAMATH_GPT_negation_equiv_l466_46695


namespace NUMINAMATH_GPT_fruit_basket_ratio_l466_46617

theorem fruit_basket_ratio (total_fruits : ℕ) (oranges : ℕ) (apples : ℕ) (h1 : total_fruits = 40) (h2 : oranges = 10) (h3 : apples = total_fruits - oranges) :
  (apples / oranges) = 3 := by
  sorry

end NUMINAMATH_GPT_fruit_basket_ratio_l466_46617


namespace NUMINAMATH_GPT_find_second_number_l466_46652

theorem find_second_number
  (a : ℝ) (b : ℝ)
  (h : a = 1280)
  (h_percent : 0.25 * a = 0.20 * b + 190) :
  b = 650 :=
sorry

end NUMINAMATH_GPT_find_second_number_l466_46652


namespace NUMINAMATH_GPT_average_age_constant_l466_46615

theorem average_age_constant 
  (average_age_3_years_ago : ℕ) 
  (number_of_members_3_years_ago : ℕ) 
  (baby_age_today : ℕ) 
  (number_of_members_today : ℕ) 
  (H1 : average_age_3_years_ago = 17) 
  (H2 : number_of_members_3_years_ago = 5) 
  (H3 : baby_age_today = 2) 
  (H4 : number_of_members_today = 6) : 
  average_age_3_years_ago = (average_age_3_years_ago * number_of_members_3_years_ago + baby_age_today + 3 * number_of_members_3_years_ago) / number_of_members_today := 
by sorry

end NUMINAMATH_GPT_average_age_constant_l466_46615


namespace NUMINAMATH_GPT_combine_quadratic_radicals_l466_46680

theorem combine_quadratic_radicals (x : ℝ) (h : 3 * x + 5 = 2 * x + 7) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_combine_quadratic_radicals_l466_46680


namespace NUMINAMATH_GPT_mother_kept_one_third_l466_46651

-- Define the problem conditions
def total_sweets : ℕ := 27
def eldest_sweets : ℕ := 8
def youngest_sweets : ℕ := eldest_sweets / 2
def second_sweets : ℕ := 6
def total_children_sweets : ℕ := eldest_sweets + youngest_sweets + second_sweets
def sweets_mother_kept : ℕ := total_sweets - total_children_sweets
def fraction_mother_kept : ℚ := sweets_mother_kept / total_sweets

-- Prove the fraction of sweets the mother kept
theorem mother_kept_one_third : fraction_mother_kept = 1 / 3 := 
  by
    sorry

end NUMINAMATH_GPT_mother_kept_one_third_l466_46651


namespace NUMINAMATH_GPT_boat_travel_time_downstream_l466_46650

-- Define the given conditions and statement to prove
theorem boat_travel_time_downstream (B : ℝ) (C : ℝ) (Us : ℝ) (Ds : ℝ) :
  (C = B / 4) ∧ (Us = B - C) ∧ (Ds = B + C) ∧ (Us = 3) ∧ (15 / Us = 5) ∧ (15 / Ds = 3) :=
by
  -- Provide the proof here; currently using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_boat_travel_time_downstream_l466_46650


namespace NUMINAMATH_GPT_common_area_of_rectangle_and_circle_l466_46674

theorem common_area_of_rectangle_and_circle :
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  ∃ (common_area : ℝ), common_area = 9 * Real.pi :=
by
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  have common_area := 9 * Real.pi
  use common_area
  sorry

end NUMINAMATH_GPT_common_area_of_rectangle_and_circle_l466_46674


namespace NUMINAMATH_GPT_min_max_ab_bc_cd_de_l466_46691

theorem min_max_ab_bc_cd_de (a b c d e : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e) (h_sum : a + b + c + d + e = 2018) : 
  ∃ a b c d e, 
  a > 0 ∧ 
  b > 0 ∧ 
  c > 0 ∧ 
  d > 0 ∧ 
  e > 0 ∧ 
  a + b + c + d + e = 2018 ∧ 
  ∀ M, M = max (max (max (a + b) (b + c)) (max (c + d) (d + e))) ↔ M = 673  :=
sorry

end NUMINAMATH_GPT_min_max_ab_bc_cd_de_l466_46691


namespace NUMINAMATH_GPT_equation_circle_iff_a_equals_neg_one_l466_46687

theorem equation_circle_iff_a_equals_neg_one :
  (∀ x y : ℝ, ∃ k : ℝ, a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = k * (x^2 + y^2)) ↔ 
  a = -1 :=
by sorry

end NUMINAMATH_GPT_equation_circle_iff_a_equals_neg_one_l466_46687


namespace NUMINAMATH_GPT_shekar_average_is_81_9_l466_46644

def shekar_average_marks (marks : List ℕ) : ℚ :=
  (marks.sum : ℚ) / marks.length

theorem shekar_average_is_81_9 :
  shekar_average_marks [92, 78, 85, 67, 89, 74, 81, 95, 70, 88] = 81.9 :=
by
  sorry

end NUMINAMATH_GPT_shekar_average_is_81_9_l466_46644


namespace NUMINAMATH_GPT_find_b_l466_46677

theorem find_b (b : ℚ) (h : b * (-3) - (b - 1) * 5 = b - 3) : b = 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l466_46677


namespace NUMINAMATH_GPT_trigonometric_equation_solution_l466_46675

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) :
  5.14 * (Real.sin (3 * x)) + Real.sin (5 * x) = 2 * (Real.cos (2 * x)) ^ 2 - 2 * (Real.sin (3 * x)) ^ 2 →
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨ (∃ k : ℤ, x = (π / 18) * (4 * k + 1)) :=
  by
  intro h
  sorry

end NUMINAMATH_GPT_trigonometric_equation_solution_l466_46675


namespace NUMINAMATH_GPT_value_of_expression_l466_46603

theorem value_of_expression :
  (3 * (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) + 2) = 4373 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l466_46603


namespace NUMINAMATH_GPT_total_chips_eaten_l466_46653

theorem total_chips_eaten (dinner_chips after_dinner_chips : ℕ) (h1 : dinner_chips = 1) (h2 : after_dinner_chips = 2 * dinner_chips) : dinner_chips + after_dinner_chips = 3 := by
  sorry

end NUMINAMATH_GPT_total_chips_eaten_l466_46653


namespace NUMINAMATH_GPT_units_digit_first_four_composite_is_eight_l466_46635

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end NUMINAMATH_GPT_units_digit_first_four_composite_is_eight_l466_46635


namespace NUMINAMATH_GPT_third_candle_remaining_fraction_l466_46654

theorem third_candle_remaining_fraction (t : ℝ) 
  (h1 : 0 < t)
  (second_candle_fraction_remaining : ℝ := 2/5)
  (third_candle_fraction_remaining : ℝ := 3/7)
  (second_candle_burned_fraction : ℝ := 3/5)
  (third_candle_burned_fraction : ℝ := 4/7)
  (second_candle_burn_rate : ℝ := 3 / (5 * t))
  (third_candle_burn_rate : ℝ := 4 / (7 * t))
  (remaining_burn_time_second : ℝ := (2 * t) / 3)
  (third_candle_burned_in_remaining_time : ℝ := (2 * t * 4) / (3 * 7 * t))
  (common_denominator_third : ℝ := 21)
  (converted_third_candle_fraction_remaining : ℝ := 9 / 21)
  (third_candle_fraction_subtracted : ℝ := 8 / 21) :
  (converted_third_candle_fraction_remaining - third_candle_fraction_subtracted) = 1 / 21 := by
  sorry

end NUMINAMATH_GPT_third_candle_remaining_fraction_l466_46654


namespace NUMINAMATH_GPT_derivative_of_constant_function_l466_46682

-- Define the constant function
def f (x : ℝ) : ℝ := 0

-- State the theorem
theorem derivative_of_constant_function : deriv f 0 = 0 := by
  -- Proof will go here, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_derivative_of_constant_function_l466_46682


namespace NUMINAMATH_GPT_range_of_m_l466_46665

theorem range_of_m (m : ℝ) :
  (∀ x : ℕ, (x = 1 ∨ x = 2 ∨ x = 3) → (3 * x - m ≤ 0)) ↔ 9 ≤ m ∧ m < 12 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l466_46665


namespace NUMINAMATH_GPT_complex_problem_l466_46605

theorem complex_problem 
  (a : ℝ) 
  (ha : a^2 - 9 = 0) :
  (a + (Complex.I ^ 19)) / (1 + Complex.I) = 1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_GPT_complex_problem_l466_46605


namespace NUMINAMATH_GPT_chimps_seen_l466_46610

-- Given conditions
def lions := 8
def lion_legs := 4
def lizards := 5
def lizard_legs := 4
def tarantulas := 125
def tarantula_legs := 8
def goal_legs := 1100

-- Required to be proved
def chimp_legs := 4

theorem chimps_seen : (goal_legs - ((lions * lion_legs) + (lizards * lizard_legs) + (tarantulas * tarantula_legs))) / chimp_legs = 25 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_chimps_seen_l466_46610


namespace NUMINAMATH_GPT_sequence_fill_l466_46692

theorem sequence_fill (x2 x3 x4 x5 x6 x7: ℕ) : 
  (20 + x2 + x3 = 100) ∧ 
  (x2 + x3 + x4 = 100) ∧ 
  (x3 + x4 + x5 = 100) ∧ 
  (x4 + x5 + x6 = 100) ∧ 
  (x5 + x6 + 16 = 100) →
  [20, x2, x3, x4, x5, x6, 16] = [20, 16, 64, 20, 16, 64, 20, 16] :=
by
  sorry

end NUMINAMATH_GPT_sequence_fill_l466_46692


namespace NUMINAMATH_GPT_sum_xyz_l466_46625

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_xyz :
  (∀ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 →
  x + y + z = 932) := 
by
  sorry

end NUMINAMATH_GPT_sum_xyz_l466_46625
