import Mathlib

namespace NUMINAMATH_GPT_chosen_number_l317_31757

theorem chosen_number (x : ℝ) (h1 : x / 9 - 100 = 10) : x = 990 :=
  sorry

end NUMINAMATH_GPT_chosen_number_l317_31757


namespace NUMINAMATH_GPT_jeff_cats_count_l317_31797

theorem jeff_cats_count :
  let initial_cats := 20
  let found_monday := 2 + 3
  let found_tuesday := 1 + 2
  let adopted_wednesday := 4 * 2
  let adopted_thursday := 3
  let found_friday := 3
  initial_cats + found_monday + found_tuesday - adopted_wednesday - adopted_thursday + found_friday = 20 := by
  sorry

end NUMINAMATH_GPT_jeff_cats_count_l317_31797


namespace NUMINAMATH_GPT_avg_growth_rate_selling_price_reduction_l317_31762

open Real

-- Define the conditions for the first question
def sales_volume_aug : ℝ := 50000
def sales_volume_oct : ℝ := 72000

-- Define the conditions for the second question
def cost_price_per_unit : ℝ := 40
def initial_selling_price_per_unit : ℝ := 80
def initial_sales_volume_per_day : ℝ := 20
def additional_units_per_half_dollar_decrease : ℝ := 4
def desired_daily_profit : ℝ := 1400

-- First proof: monthly average growth rate
theorem avg_growth_rate (x : ℝ) :
  sales_volume_aug * (1 + x)^2 = sales_volume_oct → x = 0.2 :=
by {
  sorry
}

-- Second proof: reduction in selling price for daily profit
theorem selling_price_reduction (y : ℝ) :
  (initial_selling_price_per_unit - y - cost_price_per_unit) * (initial_sales_volume_per_day + additional_units_per_half_dollar_decrease * y / 0.5) = desired_daily_profit → y = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_avg_growth_rate_selling_price_reduction_l317_31762


namespace NUMINAMATH_GPT_average_salary_of_laborers_l317_31782

-- Define the main statement as a theorem
theorem average_salary_of_laborers 
  (total_workers : ℕ)
  (total_salary_all : ℕ)
  (supervisors : ℕ)
  (supervisor_salary : ℕ)
  (laborers : ℕ)
  (expected_laborer_salary : ℝ) :
  total_workers = 48 → 
  total_salary_all = 60000 →
  supervisors = 6 →
  supervisor_salary = 2450 →
  laborers = 42 →
  expected_laborer_salary = 1078.57 :=
sorry

end NUMINAMATH_GPT_average_salary_of_laborers_l317_31782


namespace NUMINAMATH_GPT_production_line_B_units_l317_31766

theorem production_line_B_units {x y z : ℕ} (h1 : x + y + z = 24000) (h2 : 2 * y = x + z) : y = 8000 :=
sorry

end NUMINAMATH_GPT_production_line_B_units_l317_31766


namespace NUMINAMATH_GPT_find_number_l317_31773

theorem find_number (x : ℝ) (h : 5020 - (1004 / x) = 4970) : x = 20.08 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l317_31773


namespace NUMINAMATH_GPT_slips_with_3_count_l317_31729

def number_of_slips_with_3 (x : ℕ) : Prop :=
  let total_slips := 15
  let expected_value := 4.6
  let prob_3 := (x : ℚ) / total_slips
  let prob_8 := (total_slips - x : ℚ) / total_slips
  let E := prob_3 * 3 + prob_8 * 8
  E = expected_value

theorem slips_with_3_count : ∃ x : ℕ, number_of_slips_with_3 x ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_slips_with_3_count_l317_31729


namespace NUMINAMATH_GPT_quadratic_inequality_ab_l317_31702

theorem quadratic_inequality_ab (a b : ℝ) :
  (∀ x : ℝ, (x > -1 ∧ x < 1 / 3) → a * x^2 + b * x + 1 > 0) →
  a * b = 6 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_ab_l317_31702


namespace NUMINAMATH_GPT_average_speed_is_37_5_l317_31712

-- Define the conditions
def distance_local : ℕ := 60
def speed_local : ℕ := 30
def distance_gravel : ℕ := 10
def speed_gravel : ℕ := 20
def distance_highway : ℕ := 105
def speed_highway : ℕ := 60
def traffic_delay : ℚ := 15 / 60
def obstruction_delay : ℚ := 10 / 60

-- Define the total distance
def total_distance : ℕ := distance_local + distance_gravel + distance_highway

-- Define the total time
def total_time : ℚ :=
  (distance_local / speed_local) +
  (distance_gravel / speed_gravel) +
  (distance_highway / speed_highway) +
  traffic_delay +
  obstruction_delay

-- Define the average speed as distance divided by time
def average_speed : ℚ := total_distance / total_time

theorem average_speed_is_37_5 :
  average_speed = 37.5 := by sorry

end NUMINAMATH_GPT_average_speed_is_37_5_l317_31712


namespace NUMINAMATH_GPT_sum_of_g_of_nine_values_l317_31735

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (y : ℝ) : ℝ := 3 * y - 4

theorem sum_of_g_of_nine_values : (g 9) = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_g_of_nine_values_l317_31735


namespace NUMINAMATH_GPT_geometric_sequence_a3_l317_31745

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = a 1 * q ^ 3) (h2 : a 2 = a 1 * q) (h3 : a 5 = a 1 * q ^ 4) 
    (h4 : a 4 - a 2 = 6) (h5 : a 5 - a 1 = 15) : a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l317_31745


namespace NUMINAMATH_GPT_jugglers_count_l317_31724

-- Define the conditions
def num_balls_each_juggler := 6
def total_balls := 2268

-- Define the theorem to prove the number of jugglers
theorem jugglers_count : (total_balls / num_balls_each_juggler) = 378 :=
by
  sorry

end NUMINAMATH_GPT_jugglers_count_l317_31724


namespace NUMINAMATH_GPT_max_min_cos_sin_product_l317_31732

theorem max_min_cos_sin_product (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (maximum minimum : ℝ), maximum = (2 + Real.sqrt 3) / 8 ∧ minimum = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_max_min_cos_sin_product_l317_31732


namespace NUMINAMATH_GPT_group_8_extracted_number_is_72_l317_31723

-- Definitions related to the problem setup
def individ_to_group (n : ℕ) : ℕ := n / 10 + 1
def unit_digit (n : ℕ) : ℕ := n % 10
def extraction_rule (k m : ℕ) : ℕ := (k + m - 1) % 10

-- Given condition: total individuals split into sequential groups and m = 5
def total_individuals : ℕ := 100
def total_groups : ℕ := 10
def m : ℕ := 5
def k_8 : ℕ := 8

-- The final theorem statement
theorem group_8_extracted_number_is_72 : ∃ n : ℕ, individ_to_group n = k_8 ∧ unit_digit n = extraction_rule k_8 m := by
  sorry

end NUMINAMATH_GPT_group_8_extracted_number_is_72_l317_31723


namespace NUMINAMATH_GPT_cycle_selling_price_l317_31747

noncomputable def selling_price (cost_price : ℝ) (gain_percent : ℝ) : ℝ :=
  let gain_amount := (gain_percent / 100) * cost_price
  cost_price + gain_amount

theorem cycle_selling_price :
  selling_price 450 15.56 = 520.02 :=
by
  sorry

end NUMINAMATH_GPT_cycle_selling_price_l317_31747


namespace NUMINAMATH_GPT_train_length_180_l317_31714

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_seconds

theorem train_length_180 :
  train_length 6 108 = 180 :=
sorry

end NUMINAMATH_GPT_train_length_180_l317_31714


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l317_31739

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 1 = 0 → x^3 - x = 0) ∧ ¬ (x^3 - x = 0 → x^2 - 1 = 0) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l317_31739


namespace NUMINAMATH_GPT_trig_expression_value_l317_31742

open Real

theorem trig_expression_value : 
  (2 * cos (10 * (π / 180)) - sin (20 * (π / 180))) / cos (20 * (π / 180)) = sqrt 3 :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_trig_expression_value_l317_31742


namespace NUMINAMATH_GPT_quadratic_interval_inequality_l317_31781

theorem quadratic_interval_inequality (a b c : ℝ) :
  (∀ x : ℝ, -1 / 2 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  a < 0 ∧ c > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_interval_inequality_l317_31781


namespace NUMINAMATH_GPT_calc_exp_l317_31795

open Real

theorem calc_exp (x y : ℝ) : 
  (-(1/3) * (x^2) * y) ^ 3 = -(x^6 * y^3) / 27 := 
  sorry

end NUMINAMATH_GPT_calc_exp_l317_31795


namespace NUMINAMATH_GPT_prism_diagonal_length_l317_31748

theorem prism_diagonal_length (x y z : ℝ) (h1 : 4 * x + 4 * y + 4 * z = 24) (h2 : 2 * x * y + 2 * x * z + 2 * y * z = 11) : Real.sqrt (x^2 + y^2 + z^2) = 5 :=
  by
  sorry

end NUMINAMATH_GPT_prism_diagonal_length_l317_31748


namespace NUMINAMATH_GPT_total_number_of_people_l317_31759

-- Conditions
def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698

-- Theorem stating the total number of people is 803 given the conditions
theorem total_number_of_people : 
  number_of_parents + number_of_pupils = 803 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_people_l317_31759


namespace NUMINAMATH_GPT_total_yellow_balloons_l317_31706

theorem total_yellow_balloons (n_tom : ℕ) (n_sara : ℕ) (h_tom : n_tom = 9) (h_sara : n_sara = 8) : n_tom + n_sara = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_yellow_balloons_l317_31706


namespace NUMINAMATH_GPT_remaining_gift_card_value_correct_l317_31738

def initial_best_buy := 5
def initial_target := 3
def initial_walmart := 7
def initial_amazon := 2

def value_best_buy := 500
def value_target := 250
def value_walmart := 100
def value_amazon := 1000

def sent_best_buy := 1
def sent_walmart := 2
def sent_amazon := 1

def remaining_dollars : Nat :=
  (initial_best_buy - sent_best_buy) * value_best_buy +
  initial_target * value_target +
  (initial_walmart - sent_walmart) * value_walmart +
  (initial_amazon - sent_amazon) * value_amazon

theorem remaining_gift_card_value_correct : remaining_dollars = 4250 :=
  sorry

end NUMINAMATH_GPT_remaining_gift_card_value_correct_l317_31738


namespace NUMINAMATH_GPT_find_certain_number_l317_31799

theorem find_certain_number (x : ℕ) (h : (55 * x) % 8 = 7) : x = 1 := 
sorry

end NUMINAMATH_GPT_find_certain_number_l317_31799


namespace NUMINAMATH_GPT_men_handshakes_l317_31769

theorem men_handshakes (n : ℕ) (h : n * (n - 1) / 2 = 435) : n = 30 :=
sorry

end NUMINAMATH_GPT_men_handshakes_l317_31769


namespace NUMINAMATH_GPT_students_not_taking_either_l317_31767

-- Definitions of the conditions
def total_students : ℕ := 28
def students_taking_french : ℕ := 5
def students_taking_spanish : ℕ := 10
def students_taking_both : ℕ := 4

-- Theorem stating the mathematical problem
theorem students_not_taking_either :
  total_students - (students_taking_french + students_taking_spanish + students_taking_both) = 9 :=
sorry

end NUMINAMATH_GPT_students_not_taking_either_l317_31767


namespace NUMINAMATH_GPT_max_value_min_value_l317_31719

noncomputable def y (x : ℝ) : ℝ := 2 * Real.sin (3 * x + (Real.pi / 3))

theorem max_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 + Real.pi / 18) ↔ y x = 2 :=
sorry

theorem min_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 - 5 * Real.pi / 18) ↔ y x = -2 :=
sorry

end NUMINAMATH_GPT_max_value_min_value_l317_31719


namespace NUMINAMATH_GPT_gummy_vitamins_cost_l317_31721

def bottle_discounted_price (P D_s : ℝ) : ℝ :=
  P * (1 - D_s)

def normal_purchase_discounted_price (discounted_price D_n : ℝ) : ℝ :=
  discounted_price * (1 - D_n)

def bulk_purchase_discounted_price (discounted_price D_b : ℝ) : ℝ :=
  discounted_price * (1 - D_b)

def total_cost (normal_bottles bulk_bottles normal_price bulk_price : ℝ) : ℝ :=
  (normal_bottles * normal_price) + (bulk_bottles * bulk_price)

def apply_coupons (total_cost N_c C : ℝ) : ℝ :=
  total_cost - (N_c * C)

theorem gummy_vitamins_cost 
  (P N_c C D_s D_n D_b : ℝ) 
  (normal_bottles bulk_bottles : ℕ) :
  bottle_discounted_price P D_s = 12.45 → 
  normal_purchase_discounted_price 12.45 D_n = 11.33 → 
  bulk_purchase_discounted_price 12.45 D_b = 11.83 → 
  total_cost 4 3 11.33 11.83 = 80.81 → 
  apply_coupons 80.81 N_c C = 70.81 :=
sorry

end NUMINAMATH_GPT_gummy_vitamins_cost_l317_31721


namespace NUMINAMATH_GPT_total_ladybugs_l317_31770

theorem total_ladybugs (ladybugs_with_spots ladybugs_without_spots : ℕ) 
  (h1 : ladybugs_with_spots = 12170) 
  (h2 : ladybugs_without_spots = 54912) : 
  ladybugs_with_spots + ladybugs_without_spots = 67082 := 
by
  sorry

end NUMINAMATH_GPT_total_ladybugs_l317_31770


namespace NUMINAMATH_GPT_solve_system_of_equations_l317_31703

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + x * y = 15) (h2 : x^2 + x * y = 10) :
  (x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l317_31703


namespace NUMINAMATH_GPT_sum_G_correct_l317_31749

def G (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 + 1 else n^2

def sum_G (a b : ℕ) : ℕ :=
  List.sum (List.map G (List.range' a (b - a + 1)))

theorem sum_G_correct :
  sum_G 2 2007 = 8546520 := by
  sorry

end NUMINAMATH_GPT_sum_G_correct_l317_31749


namespace NUMINAMATH_GPT_difference_quotient_correct_l317_31761

theorem difference_quotient_correct (a b : ℝ) :
  abs (3 * a - b) / abs (a + 2 * b) = abs (3 * a - b) / abs (a + 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_difference_quotient_correct_l317_31761


namespace NUMINAMATH_GPT_max_log_expression_l317_31763

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem max_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x > y) :
  log_base x (x^2 / y^3) + log_base y (y^2 / x^3) = -2 :=
by
  sorry

end NUMINAMATH_GPT_max_log_expression_l317_31763


namespace NUMINAMATH_GPT_vector_calculation_l317_31700

def vector_a : ℝ × ℝ := (1, -1)
def vector_b : ℝ × ℝ := (-1, 2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

theorem vector_calculation :
  (dot_product (vector_add (scalar_mult 2 vector_a) vector_b) vector_a) = 1 :=
by
  sorry

end NUMINAMATH_GPT_vector_calculation_l317_31700


namespace NUMINAMATH_GPT_smallest_n_divisibility_problem_l317_31720

theorem smallest_n_divisibility_problem :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → ¬(n^2 + n) % k = 0)) ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisibility_problem_l317_31720


namespace NUMINAMATH_GPT_proof_solution_l317_31731

noncomputable def proof_problem (x : ℝ) : Prop :=
  (⌈2 * x⌉₊ : ℝ) - (⌊2 * x⌋₊ : ℝ) = 0 → (⌈2 * x⌉₊ : ℝ) - 2 * x = 0

theorem proof_solution (x : ℝ) : proof_problem x :=
by
  sorry

end NUMINAMATH_GPT_proof_solution_l317_31731


namespace NUMINAMATH_GPT_age_difference_l317_31726

-- Define the present age of the son.
def S : ℕ := 22

-- Define the present age of the man.
variable (M : ℕ)

-- Given condition: In two years, the man's age will be twice the age of his son.
axiom condition : M + 2 = 2 * (S + 2)

-- Prove that the difference in present ages of the man and his son is 24 years.
theorem age_difference : M - S = 24 :=
by 
  -- We will fill in the proof here
  sorry

end NUMINAMATH_GPT_age_difference_l317_31726


namespace NUMINAMATH_GPT_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l317_31704

variables {m : ℝ}

-- (1) For z to be a real number
theorem real_number_condition : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) :=
by sorry

-- (2) For z to be an imaginary number
theorem imaginary_number_condition : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) :=
by sorry

-- (3) For z to be a purely imaginary number
theorem pure_imaginary_number_condition : (m^2 - 5 * m + 6 = 0 ∧ m^2 - 3 * m ≠ 0) ↔ (m = 2) :=
by sorry

end NUMINAMATH_GPT_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l317_31704


namespace NUMINAMATH_GPT_middle_rungs_widths_l317_31728

theorem middle_rungs_widths (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 33 ∧ a 12 = 110 ∧ (∀ n, a (n + 1) = a n + 7) →
  (a 2 = 40 ∧ a 3 = 47 ∧ a 4 = 54 ∧ a 5 = 61 ∧
   a 6 = 68 ∧ a 7 = 75 ∧ a 8 = 82 ∧ a 9 = 89 ∧
   a 10 = 96 ∧ a 11 = 103) :=
by
  sorry

end NUMINAMATH_GPT_middle_rungs_widths_l317_31728


namespace NUMINAMATH_GPT_problem_1_problem_2_l317_31798

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * |x| - 2

theorem problem_1 : {x : ℝ | f x > 3} = {x : ℝ | x < -1 ∨ x > 5} :=
sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m ≤ 1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l317_31798


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l317_31771

theorem molecular_weight_of_one_mole (molecular_weight_3_moles : ℕ) (h : molecular_weight_3_moles = 222) : (molecular_weight_3_moles / 3) = 74 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l317_31771


namespace NUMINAMATH_GPT_smallest_three_digit_integer_l317_31701

theorem smallest_three_digit_integer (n : ℕ) : 
  100 ≤ n ∧ n < 1000 ∧ ¬ (n - 1 ∣ (n!)) ↔ n = 1004 := 
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_integer_l317_31701


namespace NUMINAMATH_GPT_extreme_points_of_f_range_of_a_l317_31778

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ -1 then Real.log (x + 1) + a * (x^2 - x) 
  else 0

theorem extreme_points_of_f (a : ℝ) :
  (a < 0 → ∃ x, f a x = 0) ∧
  (0 ≤ a ∧ a ≤ 8/9 → ∃! x, f a x = 0) ∧
  (a > 8/9 → ∃ x₁ x₂, x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_extreme_points_of_f_range_of_a_l317_31778


namespace NUMINAMATH_GPT_total_operations_in_one_hour_l317_31707

theorem total_operations_in_one_hour :
  let additions_per_second := 12000
  let multiplications_per_second := 8000
  (additions_per_second + multiplications_per_second) * 3600 = 72000000 :=
by
  sorry

end NUMINAMATH_GPT_total_operations_in_one_hour_l317_31707


namespace NUMINAMATH_GPT_greatest_integer_difference_l317_31774

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) : y - x = 3 :=
sorry

end NUMINAMATH_GPT_greatest_integer_difference_l317_31774


namespace NUMINAMATH_GPT_no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l317_31709

open Nat

theorem no_odd_prime_pn_plus_1_eq_2m (n p m : ℕ)
  (hn : n > 1) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n + 1 ≠ 2^m := by
  sorry

theorem no_odd_prime_pn_minus_1_eq_2m (n p m : ℕ)
  (hn : n > 2) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n - 1 ≠ 2^m := by
  sorry

end NUMINAMATH_GPT_no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l317_31709


namespace NUMINAMATH_GPT_annie_extracurricular_hours_l317_31777

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end NUMINAMATH_GPT_annie_extracurricular_hours_l317_31777


namespace NUMINAMATH_GPT_spadesuit_eval_l317_31786

def spadesuit (a b : ℤ) := abs (a - b)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 3 (spadesuit 8 12)) = 4 := 
by
  sorry

end NUMINAMATH_GPT_spadesuit_eval_l317_31786


namespace NUMINAMATH_GPT_largest_pot_cost_l317_31784

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ :=
  x + 5 * 0.15

theorem largest_pot_cost :
  ∃ (x : ℝ), (6 * x + 5 * 0.15 + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15) = 8.85) →
    cost_of_largest_pot x = 1.85 :=
by
  sorry

end NUMINAMATH_GPT_largest_pot_cost_l317_31784


namespace NUMINAMATH_GPT_find_d_l317_31746

theorem find_d 
  (d : ℝ)
  (d_gt_zero : d > 0)
  (line_eq : ∀ x : ℝ, (2 * x - 6 = 0) → x = 3)
  (y_intercept : ∀ y : ℝ, (2 * 0 - 6 = y) → y = -6)
  (area_condition : (1/2 * 3 * 6 = 9) → (1/2 * (d - 3) * (2 * d - 6) = 36)) :
  d = 9 :=
sorry

end NUMINAMATH_GPT_find_d_l317_31746


namespace NUMINAMATH_GPT_divisor_is_22_l317_31754

theorem divisor_is_22 (n d : ℤ) (h1 : n % d = 12) (h2 : (2 * n) % 11 = 2) : d = 22 :=
by
  sorry

end NUMINAMATH_GPT_divisor_is_22_l317_31754


namespace NUMINAMATH_GPT_no_infinite_pos_sequence_l317_31743

theorem no_infinite_pos_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) :
  ¬(∃ a : ℕ → ℝ, (∀ n : ℕ, a n > 0) ∧ (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n)) :=
sorry

end NUMINAMATH_GPT_no_infinite_pos_sequence_l317_31743


namespace NUMINAMATH_GPT_parabola_vertex_position_l317_31758

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem parabola_vertex_position (x y : ℝ) :
  (∃ a b : ℝ, f a = y ∧ g b = y ∧ a = 1 ∧ b = -1)
  → (1 > -1) ∧ (f 1 > g (-1)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_position_l317_31758


namespace NUMINAMATH_GPT_cos_squared_value_l317_31787

theorem cos_squared_value (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) : 
  Real.cos (π / 3 - x) ^ 2 = 1 / 16 := 
sorry

end NUMINAMATH_GPT_cos_squared_value_l317_31787


namespace NUMINAMATH_GPT_product_of_xyz_is_correct_l317_31713

theorem product_of_xyz_is_correct : 
  ∃ x y z : ℤ, 
    (-3 * x + 4 * y - z = 28) ∧ 
    (3 * x - 2 * y + z = 8) ∧ 
    (x + y - z = 2) ∧ 
    (x * y * z = 2898) :=
by
  sorry

end NUMINAMATH_GPT_product_of_xyz_is_correct_l317_31713


namespace NUMINAMATH_GPT_bike_travel_distance_l317_31722

-- Declaring the conditions as definitions
def speed : ℝ := 50 -- Speed in meters per second
def time : ℝ := 7 -- Time in seconds

-- Declaring the question and expected answer
def expected_distance : ℝ := 350 -- Expected distance in meters

-- The proof statement that needs to be proved
theorem bike_travel_distance : (speed * time = expected_distance) :=
by
  sorry

end NUMINAMATH_GPT_bike_travel_distance_l317_31722


namespace NUMINAMATH_GPT_find_some_value_l317_31792

theorem find_some_value (m n k : ℝ)
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + k = (n + 18) / 6 - 2 / 5) : 
  k = 3 :=
sorry

end NUMINAMATH_GPT_find_some_value_l317_31792


namespace NUMINAMATH_GPT_fraction_addition_l317_31772

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_addition_l317_31772


namespace NUMINAMATH_GPT_fourth_grade_students_l317_31751

theorem fourth_grade_students:
  (initial_students = 35) →
  (first_semester_left = 6) →
  (first_semester_joined = 4) →
  (first_semester_transfers = 2) →
  (second_semester_left = 3) →
  (second_semester_joined = 7) →
  (second_semester_transfers = 2) →
  final_students = initial_students - first_semester_left + first_semester_joined - second_semester_left + second_semester_joined :=
  sorry

end NUMINAMATH_GPT_fourth_grade_students_l317_31751


namespace NUMINAMATH_GPT_mason_hotdogs_proof_mason_ate_15_hotdogs_l317_31775

-- Define the weights of the items.
def weight_hotdog := 2 -- in ounces
def weight_burger := 5 -- in ounces
def weight_pie := 10 -- in ounces

-- Define Noah's consumption
def noah_burgers := 8

-- Define the total weight of hotdogs Mason ate
def mason_hotdogs_weight := 30

-- Calculate the number of hotdogs Mason ate
def hotdogs_mason_ate := mason_hotdogs_weight / weight_hotdog

-- Calculate the number of pies Jacob ate
def jacob_pies := noah_burgers - 3

-- Given conditions
theorem mason_hotdogs_proof :
  mason_hotdogs_weight / weight_hotdog = 3 * (noah_burgers - 3) :=
by
  sorry

-- Proving the number of hotdogs Mason ate equals 15
theorem mason_ate_15_hotdogs :
  hotdogs_mason_ate = 15 :=
by
  sorry

end NUMINAMATH_GPT_mason_hotdogs_proof_mason_ate_15_hotdogs_l317_31775


namespace NUMINAMATH_GPT_min_value_frac_l317_31783

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (c : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → c ≤ 8 / x + 2 / y) ∧ c = 18 :=
sorry

end NUMINAMATH_GPT_min_value_frac_l317_31783


namespace NUMINAMATH_GPT_functional_inequality_solution_l317_31736

theorem functional_inequality_solution {f : ℝ → ℝ} 
  (h : ∀ x y : ℝ, f (x * y) ≤ y * f (x) + f (y)) : 
  ∀ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_functional_inequality_solution_l317_31736


namespace NUMINAMATH_GPT_abs_neg_eight_l317_31715

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end NUMINAMATH_GPT_abs_neg_eight_l317_31715


namespace NUMINAMATH_GPT_seeds_germinated_percentage_l317_31794

theorem seeds_germinated_percentage 
  (n1 n2 : ℕ) 
  (p1 p2 : ℝ) 
  (h1 : n1 = 300)
  (h2 : n2 = 200)
  (h3 : p1 = 0.15)
  (h4 : p2 = 0.35) : 
  ( ( p1 * n1 + p2 * n2 ) / ( n1 + n2 ) ) * 100 = 23 :=
by
  -- Mathematical proof goes here.
  sorry

end NUMINAMATH_GPT_seeds_germinated_percentage_l317_31794


namespace NUMINAMATH_GPT_diameter_of_large_circle_l317_31744

-- Given conditions
def small_radius : ℝ := 3
def num_small_circles : ℕ := 6

-- Problem statement: Prove the diameter of the large circle
theorem diameter_of_large_circle (r : ℝ) (n : ℕ) (h_radius : r = small_radius) (h_num : n = num_small_circles) :
  ∃ (R : ℝ), R = 9 * 2 := 
sorry

end NUMINAMATH_GPT_diameter_of_large_circle_l317_31744


namespace NUMINAMATH_GPT_Scruffy_weight_l317_31711

variable {Muffy Puffy Scruffy : ℝ}

def Puffy_weight_condition (Muffy Puffy : ℝ) : Prop := Puffy = Muffy + 5
def Scruffy_weight_condition (Muffy Scruffy : ℝ) : Prop := Scruffy = Muffy + 3
def Combined_weight_condition (Muffy Puffy : ℝ) : Prop := Muffy + Puffy = 23

theorem Scruffy_weight (h1 : Puffy_weight_condition Muffy Puffy) (h2 : Scruffy_weight_condition Muffy Scruffy) (h3 : Combined_weight_condition Muffy Puffy) : Scruffy = 12 := by
  sorry

end NUMINAMATH_GPT_Scruffy_weight_l317_31711


namespace NUMINAMATH_GPT_average_disk_space_per_hour_l317_31730

theorem average_disk_space_per_hour :
  let days : ℕ := 15
  let total_mb : ℕ := 20000
  let hours_per_day : ℕ := 24
  let total_hours := days * hours_per_day
  total_mb / total_hours = 56 :=
by
  let days := 15
  let total_mb := 20000
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  have h : total_mb / total_hours = 56 := sorry
  exact h

end NUMINAMATH_GPT_average_disk_space_per_hour_l317_31730


namespace NUMINAMATH_GPT_min_value_of_quadratic_expression_l317_31750

theorem min_value_of_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, x^2 + 6 * x + 3 ≤ y) ∧ x^2 + 6 * x + 3 = -6 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_expression_l317_31750


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l317_31760

-- Problem 1
theorem problem1 : (2 / 19) * (8 / 25) + (17 / 25) / (19 / 2) = 2 / 19 := 
by sorry

-- Problem 2
theorem problem2 : (1 / 4) * 125 * (1 / 25) * 8 = 10 := 
by sorry

-- Problem 3
theorem problem3 : ((1 / 3) + (1 / 4)) / ((1 / 2) - (1 / 3)) = 7 / 2 := 
by sorry

-- Problem 4
theorem problem4 : ((1 / 6) + (1 / 8)) * 24 * (1 / 9) = 7 / 9 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l317_31760


namespace NUMINAMATH_GPT_weight_problem_l317_31727

theorem weight_problem (w1 w2 w3 : ℝ) (h1 : w1 + w2 + w3 = 100)
  (h2 : w1 + 2 * w2 + w3 = 101) (h3 : w1 + w2 + 2 * w3 = 102) : 
  w1 ≥ 90 ∨ w2 ≥ 90 ∨ w3 ≥ 90 :=
by
  sorry

end NUMINAMATH_GPT_weight_problem_l317_31727


namespace NUMINAMATH_GPT_labor_cost_calculation_l317_31741

def num_men : Nat := 5
def num_women : Nat := 8
def num_boys : Nat := 10

def base_wage_man : Nat := 100
def base_wage_woman : Nat := 80
def base_wage_boy : Nat := 50

def efficiency_man_woman_ratio : Nat := 2
def efficiency_man_boy_ratio : Nat := 3

def overtime_rate_multiplier : Nat := 3 / 2 -- 1.5 as a ratio
def holiday_rate_multiplier : Nat := 2

def num_men_working_overtime : Nat := 3
def hours_worked_overtime : Nat := 10
def regular_workday_hours : Nat := 8

def is_holiday : Bool := true

theorem labor_cost_calculation : 
  (num_men * base_wage_man * holiday_rate_multiplier
    + num_women * base_wage_woman * holiday_rate_multiplier
    + num_boys * base_wage_boy * holiday_rate_multiplier
    + num_men_working_overtime * (hours_worked_overtime - regular_workday_hours) * (base_wage_man * overtime_rate_multiplier)) 
  = 4180 :=
by
  sorry

end NUMINAMATH_GPT_labor_cost_calculation_l317_31741


namespace NUMINAMATH_GPT_employee_age_when_hired_l317_31725

theorem employee_age_when_hired
    (hire_year retire_year : ℕ)
    (rule_of_70 : ∀ A Y, A + Y = 70)
    (years_worked : ∀ hire_year retire_year, retire_year - hire_year = 19)
    (hire_year_eqn : hire_year = 1987)
    (retire_year_eqn : retire_year = 2006) :
  ∃ A : ℕ, A = 51 :=
by
  have Y := 19
  have A := 70 - Y
  use A
  sorry

end NUMINAMATH_GPT_employee_age_when_hired_l317_31725


namespace NUMINAMATH_GPT_time_both_pipes_opened_l317_31776

def fill_rate_p := 1 / 10
def fill_rate_q := 1 / 15
def total_fill_rate := fill_rate_p + fill_rate_q -- Combined fill rate of both pipes

def remaining_fill_rate := 10 * fill_rate_q -- Fill rate of pipe q in 10 minutes

theorem time_both_pipes_opened (t : ℝ) :
  (t / 6) + (2 / 3) = 1 → t = 2 :=
by
  sorry

end NUMINAMATH_GPT_time_both_pipes_opened_l317_31776


namespace NUMINAMATH_GPT_integer_solution_for_equation_l317_31733

theorem integer_solution_for_equation :
  ∃ (M : ℤ), 14^2 * 35^2 = 10^2 * (M - 10)^2 ∧ M = 59 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_for_equation_l317_31733


namespace NUMINAMATH_GPT_distance_between_A_and_B_is_750_l317_31717

def original_speed := 150 -- derived from the solution

def distance (S D : ℝ) :=
  (D / S) - (D / ((5 / 4) * S)) = 1 ∧
  ((D - 150) / S) - ((5 * (D - 150)) / (6 * S)) = 2 / 3

theorem distance_between_A_and_B_is_750 :
  ∃ D : ℝ, distance original_speed D ∧ D = 750 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_is_750_l317_31717


namespace NUMINAMATH_GPT_remaining_cube_edge_length_l317_31755

theorem remaining_cube_edge_length (a b : ℕ) (h : a^3 = 98 + b^3) : b = 3 :=
sorry

end NUMINAMATH_GPT_remaining_cube_edge_length_l317_31755


namespace NUMINAMATH_GPT_complement_of_M_in_U_l317_31790

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x > 0}
def complement_U_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_of_M_in_U : (U \ M) = complement_U_M :=
by sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l317_31790


namespace NUMINAMATH_GPT_most_cost_effective_way_cost_is_860_l317_31752

-- Definitions based on the problem conditions
def adult_cost := 150
def child_cost := 60
def group_cost_per_person := 100
def group_min_size := 5

-- Number of adults and children
def num_adults := 4
def num_children := 7

-- Calculate the total cost for the most cost-effective way
noncomputable def most_cost_effective_way_cost :=
  let group_tickets_count := 5  -- 4 adults + 1 child
  let remaining_children := num_children - 1
  group_tickets_count * group_cost_per_person + remaining_children * child_cost

-- Theorem to state the cost for the most cost-effective way
theorem most_cost_effective_way_cost_is_860 : most_cost_effective_way_cost = 860 := by
  sorry

end NUMINAMATH_GPT_most_cost_effective_way_cost_is_860_l317_31752


namespace NUMINAMATH_GPT_probability_difference_l317_31716

-- Definitions for probabilities
def P_plane : ℚ := 7 / 10
def P_train : ℚ := 3 / 10
def P_on_time_plane : ℚ := 8 / 10
def P_on_time_train : ℚ := 9 / 10

-- Events definitions
def P_arrive_on_time : ℚ := (7 / 10) * (8 / 10) + (3 / 10) * (9 / 10)
def P_plane_and_on_time : ℚ := (7 / 10) * (8 / 10)
def P_train_and_on_time : ℚ := (3 / 10) * (9 / 10)
def P_conditional_plane_given_on_time : ℚ := P_plane_and_on_time / P_arrive_on_time
def P_conditional_train_given_on_time : ℚ := P_train_and_on_time / P_arrive_on_time

theorem probability_difference :
  P_conditional_plane_given_on_time - P_conditional_train_given_on_time = 29 / 83 :=
by sorry

end NUMINAMATH_GPT_probability_difference_l317_31716


namespace NUMINAMATH_GPT_simplified_fraction_sum_l317_31791

theorem simplified_fraction_sum (n d : ℕ) (h_n : n = 144) (h_d : d = 256) : (9 + 16 = 25) := by
  have h1 : n = 2^4 * 3^2 := by sorry
  have h2 : d = 2^8 := by sorry
  have h3 : (n / gcd n d) = 9 := by sorry
  have h4 : (d / gcd n d) = 16 := by sorry
  exact rfl

end NUMINAMATH_GPT_simplified_fraction_sum_l317_31791


namespace NUMINAMATH_GPT_hayley_stickers_l317_31785

theorem hayley_stickers (S F x : ℕ) (hS : S = 72) (hF : F = 9) (hx : x = S / F) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_hayley_stickers_l317_31785


namespace NUMINAMATH_GPT_range_of_a_l317_31740

theorem range_of_a (M : Set ℝ) (a : ℝ) :
  (M = {x | x^2 - 4 * x + 4 * a < 0}) →
  ¬(2 ∈ M) →
  (1 ≤ a) :=
by
  -- Given assumptions
  intros hM h2_notin_M
  -- Convert h2_notin_M to an inequality and prove the desired result
  sorry

end NUMINAMATH_GPT_range_of_a_l317_31740


namespace NUMINAMATH_GPT_age_of_teacher_l317_31796

theorem age_of_teacher (S : ℕ) (T : Real) (n : ℕ) (average_student_age : Real) (new_average_age : Real) : 
  average_student_age = 14 → 
  new_average_age = 14.66 → 
  n = 45 → 
  S = average_student_age * n → 
  T = 44.7 :=
by
  sorry

end NUMINAMATH_GPT_age_of_teacher_l317_31796


namespace NUMINAMATH_GPT_total_chocolates_l317_31737

-- Definitions based on conditions
def chocolates_per_bag := 156
def number_of_bags := 20

-- Statement to prove
theorem total_chocolates : chocolates_per_bag * number_of_bags = 3120 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_total_chocolates_l317_31737


namespace NUMINAMATH_GPT_intersection_eq_l317_31764

def M : Set ℝ := {x | ∃ y, y = Real.log (2 - x) / Real.log 3}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_intersection_eq_l317_31764


namespace NUMINAMATH_GPT_three_digit_number_second_digit_l317_31718

theorem three_digit_number_second_digit (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (100 * a + 10 * b + c) - (a + b + c) = 261 → b = 7 :=
by sorry

end NUMINAMATH_GPT_three_digit_number_second_digit_l317_31718


namespace NUMINAMATH_GPT_expand_product_l317_31705

theorem expand_product (y : ℝ) : 4 * (y - 3) * (y + 2) = 4 * y^2 - 4 * y - 24 := 
by
  sorry

end NUMINAMATH_GPT_expand_product_l317_31705


namespace NUMINAMATH_GPT_evaluate_expression_l317_31788

theorem evaluate_expression : (64^(1 / 6) * 16^(1 / 4) * 8^(1 / 3) = 8) :=
by
  -- sorry added to skip the proof
  sorry

end NUMINAMATH_GPT_evaluate_expression_l317_31788


namespace NUMINAMATH_GPT_area_of_rectangle_EFGH_l317_31780

theorem area_of_rectangle_EFGH :
  ∀ (a b c : ℕ), 
    a = 7 → 
    b = 3 * a → 
    c = 2 * a → 
    (area : ℕ) = b * c → 
    area = 294 := 
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_EFGH_l317_31780


namespace NUMINAMATH_GPT_circle_radius_seven_l317_31756

theorem circle_radius_seven (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) ↔ (k = -3) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_seven_l317_31756


namespace NUMINAMATH_GPT_number_of_new_students_l317_31734

theorem number_of_new_students (initial_students left_students final_students new_students : ℕ) 
  (h_initial : initial_students = 4) 
  (h_left : left_students = 3) 
  (h_final : final_students = 43) : 
  new_students = final_students - (initial_students - left_students) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_new_students_l317_31734


namespace NUMINAMATH_GPT_candy_last_days_l317_31789

theorem candy_last_days (candy_neighbors candy_sister candy_per_day : ℕ)
  (h1 : candy_neighbors = 5)
  (h2 : candy_sister = 13)
  (h3 : candy_per_day = 9):
  (candy_neighbors + candy_sister) / candy_per_day = 2 :=
by
  sorry

end NUMINAMATH_GPT_candy_last_days_l317_31789


namespace NUMINAMATH_GPT_remainder_of_num_five_element_subsets_with_two_consecutive_l317_31710

-- Define the set and the problem
noncomputable def num_five_element_subsets_with_two_consecutive (n : ℕ) : ℕ := 
  Nat.choose 14 5 - Nat.choose 10 5

-- Main Lean statement: prove the final condition
theorem remainder_of_num_five_element_subsets_with_two_consecutive :
  (num_five_element_subsets_with_two_consecutive 14) % 1000 = 750 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_of_num_five_element_subsets_with_two_consecutive_l317_31710


namespace NUMINAMATH_GPT_remaining_balance_on_phone_card_l317_31768

theorem remaining_balance_on_phone_card (original_balance : ℝ) (cost_per_minute : ℝ) (call_duration : ℕ) :
  original_balance = 30 → cost_per_minute = 0.16 → call_duration = 22 →
  original_balance - (cost_per_minute * call_duration) = 26.48 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remaining_balance_on_phone_card_l317_31768


namespace NUMINAMATH_GPT_curve_not_parabola_l317_31765

theorem curve_not_parabola (k : ℝ) : ¬(∃ (a b c d e f : ℝ), k * x^2 + y^2 = a * x^2 + b * x * y + c * y^2 + d * x + e * y + f ∧ b^2 = 4*a*c ∧ (a = 0 ∨ c = 0)) := sorry

end NUMINAMATH_GPT_curve_not_parabola_l317_31765


namespace NUMINAMATH_GPT_infinite_common_divisor_l317_31708

theorem infinite_common_divisor (n : ℕ) : ∃ᶠ n in at_top, Nat.gcd (2 * n - 3) (3 * n - 2) > 1 := 
sorry

end NUMINAMATH_GPT_infinite_common_divisor_l317_31708


namespace NUMINAMATH_GPT_find_constants_PQR_l317_31779

theorem find_constants_PQR :
  ∃ P Q R : ℝ, 
    (6 * x + 2) / ((x - 4) * (x - 2) ^ 3) = P / (x - 4) + Q / (x - 2) + R / (x - 2) ^ 3 :=
by
  use 13 / 4
  use -6.5
  use -7
  sorry

end NUMINAMATH_GPT_find_constants_PQR_l317_31779


namespace NUMINAMATH_GPT_calculate_power_of_fractions_l317_31753

-- Defining the fractions
def a : ℚ := 5 / 6
def b : ℚ := 3 / 5

-- The main statement to prove the given question
theorem calculate_power_of_fractions : a^3 + b^3 = (21457 : ℚ) / 27000 := by 
  sorry

end NUMINAMATH_GPT_calculate_power_of_fractions_l317_31753


namespace NUMINAMATH_GPT_solve_f_g_f_3_l317_31793

def f (x : ℤ) : ℤ := 2 * x + 4

def g (x : ℤ) : ℤ := 5 * x + 2

theorem solve_f_g_f_3 :
  f (g (f 3)) = 108 := by
  sorry

end NUMINAMATH_GPT_solve_f_g_f_3_l317_31793
