import Mathlib

namespace NUMINAMATH_GPT_proof_problem_l1898_189805

theorem proof_problem (a b : ℤ) (h1 : ∃ k, a = 5 * k) (h2 : ∃ m, b = 10 * m) :
  (∃ n, b = 5 * n) ∧ (∃ p, a - b = 5 * p) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1898_189805


namespace NUMINAMATH_GPT_jenna_eel_length_l1898_189882

theorem jenna_eel_length (j b : ℕ) (h1 : b = 3 * j) (h2 : b + j = 64) : j = 16 := by 
  sorry

end NUMINAMATH_GPT_jenna_eel_length_l1898_189882


namespace NUMINAMATH_GPT_total_students_l1898_189866

theorem total_students (N : ℕ) (num_provincial : ℕ) (sample_provincial : ℕ) 
(sample_experimental : ℕ) (sample_regular : ℕ) (sample_sino_canadian : ℕ) 
(ratio : ℕ) 
(h1 : num_provincial = 96) 
(h2 : sample_provincial = 12) 
(h3 : sample_experimental = 21) 
(h4 : sample_regular = 25) 
(h5 : sample_sino_canadian = 43) 
(h6 : ratio = num_provincial / sample_provincial) 
(h7 : ratio = 8) 
: N = ratio * (sample_provincial + sample_experimental + sample_regular + sample_sino_canadian) := 
by 
  sorry

end NUMINAMATH_GPT_total_students_l1898_189866


namespace NUMINAMATH_GPT_longest_side_enclosure_l1898_189830

variable (l w : ℝ)

theorem longest_side_enclosure (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 :=
sorry

end NUMINAMATH_GPT_longest_side_enclosure_l1898_189830


namespace NUMINAMATH_GPT_expand_expression_l1898_189810

variable {x y z : ℝ}

theorem expand_expression :
  (2 * x + 5) * (3 * y + 15 + 4 * z) = 6 * x * y + 30 * x + 8 * x * z + 15 * y + 20 * z + 75 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1898_189810


namespace NUMINAMATH_GPT_fraction_arithmetic_l1898_189844

theorem fraction_arithmetic : ((3 / 5 : ℚ) + (4 / 15)) * (2 / 3) = 26 / 45 := 
by
  sorry

end NUMINAMATH_GPT_fraction_arithmetic_l1898_189844


namespace NUMINAMATH_GPT_tourists_walking_speed_l1898_189875

-- Define the conditions
def tourists_start_time := 3 + 10 / 60 -- 3:10 A.M.
def bus_pickup_time := 5 -- 5:00 A.M.
def bus_speed := 60 -- 60 km/h
def early_arrival := 20 / 60 -- 20 minutes earlier

-- This is the Lean 4 theorem statement
theorem tourists_walking_speed : 
  (bus_speed * (10 / 60) / (100 / 60)) = 6 := 
by
  sorry

end NUMINAMATH_GPT_tourists_walking_speed_l1898_189875


namespace NUMINAMATH_GPT_cos_double_angle_zero_l1898_189842

theorem cos_double_angle_zero (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = Real.cos (Real.pi / 6 + α)) : Real.cos (2 * α) = 0 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_zero_l1898_189842


namespace NUMINAMATH_GPT_factory_A_higher_output_l1898_189851

theorem factory_A_higher_output (a x : ℝ) (a_pos : a > 0) (x_pos : x > 0) 
  (h_eq_march : 1 + 2 * a = (1 + x) ^ 2) : 
  1 + a > 1 + x :=
by
  sorry

end NUMINAMATH_GPT_factory_A_higher_output_l1898_189851


namespace NUMINAMATH_GPT_line_equation_sum_l1898_189863

theorem line_equation_sum (m b x y : ℝ) (hx : x = 4) (hy : y = 2) (hm : m = -5) (hline : y = m * x + b) : m + b = 17 := by
  sorry

end NUMINAMATH_GPT_line_equation_sum_l1898_189863


namespace NUMINAMATH_GPT_tan_beta_eq_minus_one_seventh_l1898_189889

theorem tan_beta_eq_minus_one_seventh {α β : ℝ} 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := 
by
  sorry

end NUMINAMATH_GPT_tan_beta_eq_minus_one_seventh_l1898_189889


namespace NUMINAMATH_GPT_square_of_neg_three_l1898_189828

theorem square_of_neg_three : (-3 : ℤ)^2 = 9 := by
  sorry

end NUMINAMATH_GPT_square_of_neg_three_l1898_189828


namespace NUMINAMATH_GPT_age_ratio_l1898_189865

theorem age_ratio (x : ℕ) (h : (5 * x - 4) = (3 * x + 4)) :
    (5 * x + 4) / (3 * x - 4) = 3 :=
by sorry

end NUMINAMATH_GPT_age_ratio_l1898_189865


namespace NUMINAMATH_GPT_sum_difference_arithmetic_sequences_l1898_189845

open Nat

def arithmetic_sequence_sum (a d n : Nat) : Nat :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference_arithmetic_sequences :
  arithmetic_sequence_sum 2101 1 123 - arithmetic_sequence_sum 401 1 123 = 209100 := by
  sorry

end NUMINAMATH_GPT_sum_difference_arithmetic_sequences_l1898_189845


namespace NUMINAMATH_GPT_alcohol_quantity_l1898_189832

theorem alcohol_quantity (A W : ℕ) (h1 : 4 * W = 3 * A) (h2 : 4 * (W + 8) = 5 * A) : A = 16 := 
by
  sorry

end NUMINAMATH_GPT_alcohol_quantity_l1898_189832


namespace NUMINAMATH_GPT_pipe_A_time_to_fill_l1898_189895

theorem pipe_A_time_to_fill (T_B : ℝ) (T_combined : ℝ) (T_A : ℝ): 
  T_B = 75 → T_combined = 30 → 
  (1 / T_B + 1 / T_A = 1 / T_combined) → T_A = 50 :=
by
  -- Placeholder proof
  intro h1 h2 h3
  have h4 : T_B = 75 := h1
  have h5 : T_combined = 30 := h2
  have h6 : 1 / T_B + 1 / T_A = 1 / T_combined := h3
  sorry

end NUMINAMATH_GPT_pipe_A_time_to_fill_l1898_189895


namespace NUMINAMATH_GPT_symmetric_points_on_parabola_l1898_189837

theorem symmetric_points_on_parabola (x1 x2 y1 y2 m : ℝ)
  (h1: y1 = 2 * x1 ^ 2)
  (h2: y2 = 2 * x2 ^ 2)
  (h3: x1 * x2 = -1 / 2)
  (h4: y2 - y1 = 2 * (x2 ^ 2 - x1 ^ 2))
  (h5: (x1 + x2) / 2 = -1 / 4)
  (h6: (y1 + y2) / 2 = (x1 + x2) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_on_parabola_l1898_189837


namespace NUMINAMATH_GPT_minimize_F_l1898_189883

theorem minimize_F : ∃ x1 x2 x3 x4 x5 : ℝ, 
  (-2 * x1 + x2 + x3 = 2) ∧ 
  (x1 - 2 * x2 + x4 = 2) ∧ 
  (x1 + x2 + x5 = 5) ∧ 
  (x1 ≥ 0) ∧ 
  (x2 ≥ 0) ∧ 
  (x2 - x1 = -3) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimize_F_l1898_189883


namespace NUMINAMATH_GPT_longest_tape_length_l1898_189809

theorem longest_tape_length (a b c : ℕ) (h1 : a = 600) (h2 : b = 500) (h3 : c = 1200) : Nat.gcd (Nat.gcd a b) c = 100 :=
by
  sorry

end NUMINAMATH_GPT_longest_tape_length_l1898_189809


namespace NUMINAMATH_GPT_card_game_final_amounts_l1898_189857

theorem card_game_final_amounts
  (T : ℝ)
  (aldo_initial_ratio : ℝ := 7)
  (bernardo_initial_ratio : ℝ := 6)
  (carlos_initial_ratio : ℝ := 5)
  (aldo_final_ratio : ℝ := 6)
  (bernardo_final_ratio : ℝ := 5)
  (carlos_final_ratio : ℝ := 4)
  (aldo_won : ℝ := 1200) :
  aldo_won = (1 / 90) * T →
  T = 108000 →
  (36 / 90) * T = 43200 ∧ (30 / 90) * T = 36000 ∧ (24 / 90) * T = 28800 := sorry

end NUMINAMATH_GPT_card_game_final_amounts_l1898_189857


namespace NUMINAMATH_GPT_subtract_eq_l1898_189869

theorem subtract_eq (x y : ℝ) (h1 : 4 * x - 3 * y = 2) (h2 : 4 * x + y = 10) : 4 * y = 8 :=
by
  sorry

end NUMINAMATH_GPT_subtract_eq_l1898_189869


namespace NUMINAMATH_GPT_eight_digit_descending_numbers_count_l1898_189821

theorem eight_digit_descending_numbers_count : (Nat.choose 10 2) = 45 :=
by
  sorry

end NUMINAMATH_GPT_eight_digit_descending_numbers_count_l1898_189821


namespace NUMINAMATH_GPT_minimum_a_l1898_189886

def f (x a : ℝ) : ℝ := x^2 - 2*x - abs (x-1-a) - abs (x-2) + 4

theorem minimum_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = -2 :=
sorry

end NUMINAMATH_GPT_minimum_a_l1898_189886


namespace NUMINAMATH_GPT_scooter_price_l1898_189897

theorem scooter_price (total_cost: ℝ) (h: 0.20 * total_cost = 240): total_cost = 1200 :=
by
  sorry

end NUMINAMATH_GPT_scooter_price_l1898_189897


namespace NUMINAMATH_GPT_intersection_P_Q_l1898_189840

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem intersection_P_Q :
  P ∩ Q = {-1, 0, 1} :=
sorry

end NUMINAMATH_GPT_intersection_P_Q_l1898_189840


namespace NUMINAMATH_GPT_van_speed_maintain_l1898_189891

theorem van_speed_maintain 
  (D : ℕ) (T T_new : ℝ) 
  (initial_distance : D = 435) 
  (initial_time : T = 5) 
  (new_time : T_new = T / 2) : 
  D / T_new = 174 := 
by 
  sorry

end NUMINAMATH_GPT_van_speed_maintain_l1898_189891


namespace NUMINAMATH_GPT_students_more_than_rabbits_l1898_189831

/- Define constants for the problem. -/
def students_per_class : ℕ := 20
def rabbits_per_class : ℕ := 3
def num_classes : ℕ := 5

/- Define total counts based on given conditions. -/
def total_students : ℕ := students_per_class * num_classes
def total_rabbits : ℕ := rabbits_per_class * num_classes

/- The theorem we need to prove: The difference between total students and total rabbits is 85. -/
theorem students_more_than_rabbits : total_students - total_rabbits = 85 := by
  sorry

end NUMINAMATH_GPT_students_more_than_rabbits_l1898_189831


namespace NUMINAMATH_GPT_work_completion_l1898_189807

theorem work_completion 
  (x_work_days : ℕ) 
  (y_work_days : ℕ) 
  (y_worked_days : ℕ) 
  (x_rate := 1 / (x_work_days : ℚ)) 
  (y_rate := 1 / (y_work_days : ℚ)) 
  (work_remaining := 1 - y_rate * y_worked_days) 
  (remaining_work_days := work_remaining / x_rate) : 
  x_work_days = 18 → 
  y_work_days = 15 → 
  y_worked_days = 5 → 
  remaining_work_days = 12 := 
by
  intros
  sorry

end NUMINAMATH_GPT_work_completion_l1898_189807


namespace NUMINAMATH_GPT_regular_2020_gon_isosceles_probability_l1898_189898

theorem regular_2020_gon_isosceles_probability :
  let n := 2020
  let totalTriangles := (n * (n - 1) * (n - 2)) / 6
  let isoscelesTriangles := n * ((n - 2) / 2)
  let probability := isoscelesTriangles * 6 / totalTriangles
  let (a, b) := (1, 673)
  100 * a + b = 773 := by
    sorry

end NUMINAMATH_GPT_regular_2020_gon_isosceles_probability_l1898_189898


namespace NUMINAMATH_GPT_cube_edge_length_l1898_189870

theorem cube_edge_length (a : ℝ) (h : 6 * a^2 = 24) : a = 2 :=
by sorry

end NUMINAMATH_GPT_cube_edge_length_l1898_189870


namespace NUMINAMATH_GPT_attendees_proportion_l1898_189839

def attendees (t k : ℕ) := k / t

theorem attendees_proportion (n t new_t : ℕ) (h1 : n * t = 15000) (h2 : t = 50) (h3 : new_t = 75) : attendees new_t 15000 = 200 :=
by
  -- Proof omitted, main goal is to assert equivalency
  sorry

end NUMINAMATH_GPT_attendees_proportion_l1898_189839


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l1898_189801

open Real

theorem sum_of_reciprocals_of_squares {a b c : ℝ} (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = -7) (h3 : a * b * c = -2) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 73 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l1898_189801


namespace NUMINAMATH_GPT_find_rate_of_interest_l1898_189825

-- Define the problem conditions
def principal_B : ℝ := 4000
def principal_C : ℝ := 2000
def time_B : ℝ := 2
def time_C : ℝ := 4
def total_interest : ℝ := 2200

-- Define the unknown rate of interest per annum
noncomputable def rate_of_interest (R : ℝ) : Prop :=
  let interest_B := (principal_B * R * time_B) / 100
  let interest_C := (principal_C * R * time_C) / 100
  interest_B + interest_C = total_interest

-- Statement to prove that the rate of interest is 13.75%
theorem find_rate_of_interest : rate_of_interest 13.75 := by
  sorry

end NUMINAMATH_GPT_find_rate_of_interest_l1898_189825


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1898_189827

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
a / b

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a * b = 50) (h3 : a > b) :
  ratio_of_numbers a b = 4 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1898_189827


namespace NUMINAMATH_GPT_remainder_division_l1898_189833
-- Import the necessary library

-- Define the number and the divisor
def number : ℕ := 2345678901
def divisor : ℕ := 101

-- State the theorem
theorem remainder_division : number % divisor = 23 :=
by sorry

end NUMINAMATH_GPT_remainder_division_l1898_189833


namespace NUMINAMATH_GPT_total_prime_ending_starting_numerals_l1898_189820

def single_digit_primes : List ℕ := [2, 3, 5, 7]
def number_of_possible_digits := 10

def count_3digit_numerals : ℕ :=
  4 * number_of_possible_digits * 4

def count_4digit_numerals : ℕ :=
  4 * number_of_possible_digits * number_of_possible_digits * 4

theorem total_prime_ending_starting_numerals : 
  count_3digit_numerals + count_4digit_numerals = 1760 := by
sorry

end NUMINAMATH_GPT_total_prime_ending_starting_numerals_l1898_189820


namespace NUMINAMATH_GPT_product_of_integers_l1898_189823

theorem product_of_integers :
  ∃ (A B C : ℤ), A + B + C = 33 ∧ C = 3 * B ∧ A = C - 23 ∧ A * B * C = 192 :=
by
  sorry

end NUMINAMATH_GPT_product_of_integers_l1898_189823


namespace NUMINAMATH_GPT_faster_pump_rate_ratio_l1898_189835

theorem faster_pump_rate_ratio (S F : ℝ) 
  (h1 : S + F = 1/5) 
  (h2 : S = 1/12.5) : F / S = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_faster_pump_rate_ratio_l1898_189835


namespace NUMINAMATH_GPT_smallest_odd_number_divisible_by_3_l1898_189846

theorem smallest_odd_number_divisible_by_3 : ∃ n : ℕ, n = 3 ∧ ∀ m : ℕ, (m % 2 = 1 ∧ m % 3 = 0) → m ≥ n := 
by
  sorry

end NUMINAMATH_GPT_smallest_odd_number_divisible_by_3_l1898_189846


namespace NUMINAMATH_GPT_jane_oldest_child_age_l1898_189885

-- Define the conditions
def jane_start_age : ℕ := 20
def jane_current_age : ℕ := 32
def stopped_babysitting_years_ago : ℕ := 10
def baby_sat_condition (jane_age child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- Define the proof problem
theorem jane_oldest_child_age :
  (∃ age_stopped child_age,
    stopped_babysitting_years_ago = jane_current_age - age_stopped ∧
    baby_sat_condition age_stopped child_age ∧
    (32 - stopped_babysitting_years_ago = 22) ∧ -- Jane's age when she stopped baby-sitting
    child_age = 22 / 2 ∧ -- Oldest child she could have baby-sat at age 22
    child_age + stopped_babysitting_years_ago = 21) --  current age of the oldest person for whom Jane could have baby-sat
:= sorry

end NUMINAMATH_GPT_jane_oldest_child_age_l1898_189885


namespace NUMINAMATH_GPT_polynomial_evaluation_l1898_189812

theorem polynomial_evaluation 
  (x : ℝ) 
  (h1 : x^2 - 3 * x - 10 = 0) 
  (h2 : x > 0) : 
  (x^4 - 3 * x^3 + 2 * x^2 + 5 * x - 7) = 318 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1898_189812


namespace NUMINAMATH_GPT_number_of_black_cats_l1898_189815

-- Definitions of the conditions.
def white_cats : Nat := 2
def gray_cats : Nat := 3
def total_cats : Nat := 15

-- The theorem we want to prove.
theorem number_of_black_cats : ∃ B : Nat, B = total_cats - (white_cats + gray_cats) ∧ B = 10 := by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_number_of_black_cats_l1898_189815


namespace NUMINAMATH_GPT_find_ab_bc_value_l1898_189861

theorem find_ab_bc_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 := by
sorry

end NUMINAMATH_GPT_find_ab_bc_value_l1898_189861


namespace NUMINAMATH_GPT_find_numbers_l1898_189822

theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a * b) = Real.sqrt 5) ∧ 
  (2 * a * b / (a + b) = 5 / 3) → 
  (a = 5 ∧ b = 1) ∨ (a = 1 ∧ b = 5) := 
sorry

end NUMINAMATH_GPT_find_numbers_l1898_189822


namespace NUMINAMATH_GPT_possible_measures_A_l1898_189841

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end NUMINAMATH_GPT_possible_measures_A_l1898_189841


namespace NUMINAMATH_GPT_greatest_whole_number_satisfying_inequalities_l1898_189804

theorem greatest_whole_number_satisfying_inequalities :
  ∃ x : ℕ, 3 * (x : ℤ) - 5 < 1 - x ∧ 2 * (x : ℤ) + 4 ≤ 8 ∧ ∀ y : ℕ, y > x → ¬ (3 * (y : ℤ) - 5 < 1 - y ∧ 2 * (y : ℤ) + 4 ≤ 8) :=
sorry

end NUMINAMATH_GPT_greatest_whole_number_satisfying_inequalities_l1898_189804


namespace NUMINAMATH_GPT_neg_abs_value_eq_neg_three_l1898_189859

theorem neg_abs_value_eq_neg_three : -|-3| = -3 := 
by sorry

end NUMINAMATH_GPT_neg_abs_value_eq_neg_three_l1898_189859


namespace NUMINAMATH_GPT_max_product_of_xy_l1898_189829

open Real

theorem max_product_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) :
  x * y ≤ 1 / 16 := 
sorry

end NUMINAMATH_GPT_max_product_of_xy_l1898_189829


namespace NUMINAMATH_GPT_inverse_sum_l1898_189893

def f (x : ℝ) : ℝ := x * |x|

theorem inverse_sum (h1 : ∃ x : ℝ, f x = 9) (h2 : ∃ x : ℝ, f x = -81) :
  ∃ a b: ℝ, f a = 9 ∧ f b = -81 ∧ a + b = -6 :=
by
  sorry

end NUMINAMATH_GPT_inverse_sum_l1898_189893


namespace NUMINAMATH_GPT_kim_saplings_left_l1898_189876

def sprouted_pits (total_pits num_sprouted_pits: ℕ) (percent_sprouted: ℝ) : Prop :=
  percent_sprouted * total_pits = num_sprouted_pits

def sold_saplings (total_saplings saplings_sold saplings_left: ℕ) : Prop :=
  total_saplings - saplings_sold = saplings_left

theorem kim_saplings_left
  (total_pits : ℕ) (num_sprouted_pits : ℕ) (percent_sprouted : ℝ)
  (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  percent_sprouted = 0.25 →
  saplings_sold = 6 →
  sprouted_pits total_pits num_sprouted_pits percent_sprouted →
  sold_saplings num_sprouted_pits saplings_sold saplings_left →
  saplings_left = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_kim_saplings_left_l1898_189876


namespace NUMINAMATH_GPT_macey_needs_to_save_three_more_weeks_l1898_189803

def cost_of_shirt : ℝ := 3.0
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

theorem macey_needs_to_save_three_more_weeks :
  ∃ W : ℝ, W * saving_per_week = cost_of_shirt - amount_saved ∧ W = 3 := by
  sorry

end NUMINAMATH_GPT_macey_needs_to_save_three_more_weeks_l1898_189803


namespace NUMINAMATH_GPT_division_remainder_l1898_189881

theorem division_remainder :
  let p := fun x : ℝ => 5 * x^4 - 9 * x^3 + 3 * x^2 - 7 * x - 30
  let q := 3 * x - 9
  p 3 % q = 138 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1898_189881


namespace NUMINAMATH_GPT_smallest_value_of_a_minus_b_l1898_189879

theorem smallest_value_of_a_minus_b (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_a_minus_b_l1898_189879


namespace NUMINAMATH_GPT_shopper_saved_percentage_l1898_189848

theorem shopper_saved_percentage (amount_paid : ℝ) (amount_saved : ℝ) (original_price : ℝ)
  (h1 : amount_paid = 45) (h2 : amount_saved = 5) (h3 : original_price = amount_paid + amount_saved) :
  (amount_saved / original_price) * 100 = 10 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_shopper_saved_percentage_l1898_189848


namespace NUMINAMATH_GPT_student_calculation_no_error_l1898_189899

theorem student_calculation_no_error :
  let correct_result : ℚ := (7 * 4) / (5 / 3)
  let student_result : ℚ := (7 * 4) * (3 / 5)
  correct_result = student_result → 0 = 0 := 
by
  intros correct_result student_result h
  sorry

end NUMINAMATH_GPT_student_calculation_no_error_l1898_189899


namespace NUMINAMATH_GPT_length_of_bridge_l1898_189811

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_hr : ℝ)
  (time_sec : ℝ)
  (h_train_length : length_of_train = 155)
  (h_train_speed : speed_km_hr = 45)
  (h_time : time_sec = 30) :
  ∃ (length_of_bridge : ℝ),
    length_of_bridge = 220 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1898_189811


namespace NUMINAMATH_GPT_scott_earnings_l1898_189896

theorem scott_earnings
  (price_smoothie : ℝ)
  (price_cake : ℝ)
  (cups_sold : ℝ)
  (cakes_sold : ℝ)
  (earnings_smoothies : ℝ := cups_sold * price_smoothie)
  (earnings_cakes : ℝ := cakes_sold * price_cake) :
  price_smoothie = 3 → price_cake = 2 → cups_sold = 40 → cakes_sold = 18 → 
  (earnings_smoothies + earnings_cakes) = 156 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_scott_earnings_l1898_189896


namespace NUMINAMATH_GPT_fruits_in_good_condition_percentage_l1898_189836

theorem fruits_in_good_condition_percentage (total_oranges total_bananas rotten_oranges_percentage rotten_bananas_percentage : ℝ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percentage = 0.15) 
  (h4 : rotten_bananas_percentage = 0.08) : 
  (1 - ((rotten_oranges_percentage * total_oranges + rotten_bananas_percentage * total_bananas) / (total_oranges + total_bananas))) * 100 = 87.8 :=
by 
  sorry

end NUMINAMATH_GPT_fruits_in_good_condition_percentage_l1898_189836


namespace NUMINAMATH_GPT_dice_product_divisibility_probability_l1898_189802

theorem dice_product_divisibility_probability :
  let p := 1 - ((5 / 18)^6 : ℚ)
  p = (33996599 / 34012224 : ℚ) :=
by
  -- This is the condition where the probability p is computed as the complementary probability.
  sorry

end NUMINAMATH_GPT_dice_product_divisibility_probability_l1898_189802


namespace NUMINAMATH_GPT_at_least_3_defective_correct_l1898_189806

/-- Number of products in batch -/
def total_products : ℕ := 50

/-- Number of defective products -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

/-- Number of ways to draw at least 3 defective products out of 5 -/
def num_ways_at_least_3_defective : ℕ :=
  (Nat.choose defective_products 4) * (Nat.choose (total_products - defective_products) 1) +
  (Nat.choose defective_products 3) * (Nat.choose (total_products - defective_products) 2)

theorem at_least_3_defective_correct : num_ways_at_least_3_defective = 4186 := by
  sorry

end NUMINAMATH_GPT_at_least_3_defective_correct_l1898_189806


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1898_189816

theorem geometric_sequence_seventh_term (a r : ℝ) 
    (h1 : a * r^3 = 8) 
    (h2 : a * r^9 = 2) : 
    a * r^6 = 1 := 
by 
    sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1898_189816


namespace NUMINAMATH_GPT_team_combinations_l1898_189892

/-- 
The math club at Walnutridge High School has five girls and seven boys. 
How many different teams, comprising two girls and two boys, can be formed 
if one boy on each team must also be designated as the team leader?
-/
theorem team_combinations (girls boys : ℕ) (h_girls : girls = 5) (h_boys : boys = 7) :
  ∃ n, n = 420 :=
by
  sorry

end NUMINAMATH_GPT_team_combinations_l1898_189892


namespace NUMINAMATH_GPT_trajectory_of_Q_is_parabola_l1898_189858

/--
Given a point P (x, y) moves on a unit circle centered at the origin,
prove that the trajectory of point Q (u, v) defined by u = x + y and v = xy 
satisfies u^2 - 2v = 1 and is thus a parabola.
-/
theorem trajectory_of_Q_is_parabola 
  (x y u v : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : u = x + y) 
  (h3 : v = x * y) :
  u^2 - 2 * v = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_Q_is_parabola_l1898_189858


namespace NUMINAMATH_GPT_hundredth_ring_square_count_l1898_189855

-- Conditions
def center_rectangle : ℤ × ℤ := (1, 2)
def first_ring_square_count : ℕ := 10
def square_count_nth_ring (n : ℕ) : ℕ := 8 * n + 2

-- Problem Statement
theorem hundredth_ring_square_count : square_count_nth_ring 100 = 802 := 
  sorry

end NUMINAMATH_GPT_hundredth_ring_square_count_l1898_189855


namespace NUMINAMATH_GPT_order_of_values_l1898_189800

noncomputable def a : ℝ := Real.log 2 / 2
noncomputable def b : ℝ := Real.log 3 / 3
noncomputable def c : ℝ := Real.log Real.pi / Real.pi
noncomputable def d : ℝ := Real.log 2.72 / 2.72
noncomputable def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_values : a < f ∧ f < c ∧ c < b ∧ b < d :=
by
  sorry

end NUMINAMATH_GPT_order_of_values_l1898_189800


namespace NUMINAMATH_GPT_sin_A_in_right_triangle_l1898_189852

theorem sin_A_in_right_triangle (B C A : Real) (hBC: B + C = π / 2) 
(h_sinB: Real.sin B = 3 / 5) (h_sinC: Real.sin C = 4 / 5) : 
Real.sin A = 1 := 
by 
  sorry

end NUMINAMATH_GPT_sin_A_in_right_triangle_l1898_189852


namespace NUMINAMATH_GPT_crayons_total_l1898_189853

def crayons_per_child := 6
def number_of_children := 12
def total_crayons := 72

theorem crayons_total :
  crayons_per_child * number_of_children = total_crayons := by
  sorry

end NUMINAMATH_GPT_crayons_total_l1898_189853


namespace NUMINAMATH_GPT_star_value_l1898_189854

-- Define the operation a star b
def star (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

-- We want to prove that 5 star 3 = 4
theorem star_value : star 5 3 = 4 := by
  sorry

end NUMINAMATH_GPT_star_value_l1898_189854


namespace NUMINAMATH_GPT_greatest_integer_less_than_or_equal_to_frac_l1898_189871

theorem greatest_integer_less_than_or_equal_to_frac (a b c d : ℝ)
  (ha : a = 4^100) (hb : b = 3^100) (hc : c = 4^95) (hd : d = 3^95) :
  ⌊(a + b) / (c + d)⌋ = 1023 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_less_than_or_equal_to_frac_l1898_189871


namespace NUMINAMATH_GPT_tan_alpha_plus_405_deg_l1898_189849

theorem tan_alpha_plus_405_deg (α : ℝ) (h : Real.tan (180 - α) = -4 / 3) : Real.tan (α + 405) = -7 := 
sorry

end NUMINAMATH_GPT_tan_alpha_plus_405_deg_l1898_189849


namespace NUMINAMATH_GPT_cookies_per_bag_l1898_189819

theorem cookies_per_bag (b T : ℕ) (h1 : b = 37) (h2 : T = 703) : (T / b) = 19 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_cookies_per_bag_l1898_189819


namespace NUMINAMATH_GPT_selling_price_ratio_l1898_189888

theorem selling_price_ratio (C : ℝ) (hC : C > 0) :
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  S₂ / S₁ = 21 / 8 :=
by
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  sorry

end NUMINAMATH_GPT_selling_price_ratio_l1898_189888


namespace NUMINAMATH_GPT_min_value_I_is_3_l1898_189887

noncomputable def min_value_I (a b c x y : ℝ) : ℝ :=
  1 / (2 * a^3 * x + b^3 * y^2) + 1 / (2 * b^3 * x + c^3 * y^2) + 1 / (2 * c^3 * x + a^3 * y^2)

theorem min_value_I_is_3 {a b c x y : ℝ} (h1 : a^6 + b^6 + c^6 = 3) (h2 : (x + 1)^2 + y^2 ≤ 2) :
  3 ≤ min_value_I a b c x y :=
sorry

end NUMINAMATH_GPT_min_value_I_is_3_l1898_189887


namespace NUMINAMATH_GPT_compare_fractions_l1898_189867

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end NUMINAMATH_GPT_compare_fractions_l1898_189867


namespace NUMINAMATH_GPT_problem_3_problem_4_l1898_189847

open Classical

section
  variable {x₁ x₂ : ℝ}
  theorem problem_3 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) : (Real.log (x₁ * x₂) = Real.log x₁ + Real.log x₂) :=
  by
    sorry

  theorem problem_4 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hlt : x₁ < x₂) : ((Real.log x₁ - Real.log x₂) / (x₁ - x₂) > 0) :=
  by
    sorry
end

end NUMINAMATH_GPT_problem_3_problem_4_l1898_189847


namespace NUMINAMATH_GPT_sugar_snap_peas_l1898_189894

theorem sugar_snap_peas (P : ℕ) (h1 : P / 7 = 72 / 9) : P = 56 := 
sorry

end NUMINAMATH_GPT_sugar_snap_peas_l1898_189894


namespace NUMINAMATH_GPT_fraction_red_knights_magical_l1898_189877

theorem fraction_red_knights_magical (total_knights red_knights blue_knights magical_knights : ℕ)
  (fraction_red fraction_magical : ℚ)
  (frac_red_mag : ℚ) :
  (red_knights = total_knights * fraction_red) →
  (fraction_red = 3 / 8) →
  (magical_knights = total_knights * fraction_magical) →
  (fraction_magical = 1 / 4) →
  (frac_red_mag * red_knights + (frac_red_mag / 3) * blue_knights = magical_knights) →
  (frac_red_mag = 3 / 7) :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_fraction_red_knights_magical_l1898_189877


namespace NUMINAMATH_GPT_cards_distribution_l1898_189873

theorem cards_distribution (total_cards people : ℕ) (h1 : total_cards = 48) (h2 : people = 7) :
  (people - (total_cards % people)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_cards_distribution_l1898_189873


namespace NUMINAMATH_GPT_john_newspaper_percentage_less_l1898_189862

theorem john_newspaper_percentage_less
  (total_newspapers : ℕ)
  (selling_price : ℝ)
  (percentage_sold : ℝ)
  (profit : ℝ)
  (total_cost : ℝ)
  (cost_per_newspaper : ℝ)
  (percentage_less : ℝ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : percentage_sold = 0.80)
  (h4 : profit = 550)
  (h5 : total_cost = 800 - profit)
  (h6 : cost_per_newspaper = total_cost / total_newspapers)
  (h7 : percentage_less = ((selling_price - cost_per_newspaper) / selling_price) * 100) :
  percentage_less = 75 :=
by
  sorry

end NUMINAMATH_GPT_john_newspaper_percentage_less_l1898_189862


namespace NUMINAMATH_GPT_units_digit_of_k_squared_plus_2_to_the_k_l1898_189818

def k : ℕ := 2021^2 + 2^2021 + 3

theorem units_digit_of_k_squared_plus_2_to_the_k :
    (k^2 + 2^k) % 10 = 0 :=
by
    sorry

end NUMINAMATH_GPT_units_digit_of_k_squared_plus_2_to_the_k_l1898_189818


namespace NUMINAMATH_GPT_multiples_of_7_are_128_l1898_189814

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end NUMINAMATH_GPT_multiples_of_7_are_128_l1898_189814


namespace NUMINAMATH_GPT_largest_integer_solution_of_abs_eq_and_inequality_l1898_189860

theorem largest_integer_solution_of_abs_eq_and_inequality : 
  ∃ x : ℤ, |x - 3| = 15 ∧ x ≤ 20 ∧ (∀ y : ℤ, |y - 3| = 15 ∧ y ≤ 20 → y ≤ x) :=
sorry

end NUMINAMATH_GPT_largest_integer_solution_of_abs_eq_and_inequality_l1898_189860


namespace NUMINAMATH_GPT_minimum_value_of_f_l1898_189890

def f (x : ℝ) : ℝ := |x - 4| + |x + 6| + |x - 5|

theorem minimum_value_of_f :
  ∃ x : ℝ, (x = -6 ∧ f (-6) = 1) ∧ ∀ y : ℝ, f y ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1898_189890


namespace NUMINAMATH_GPT_func_increasing_l1898_189826

noncomputable def func (x : ℝ) : ℝ :=
  x^3 + x + 1

theorem func_increasing : ∀ x : ℝ, deriv func x > 0 := by
  sorry

end NUMINAMATH_GPT_func_increasing_l1898_189826


namespace NUMINAMATH_GPT_convex_quad_sum_greater_diff_l1898_189850

theorem convex_quad_sum_greater_diff (α β γ δ : ℝ) 
    (h_sum : α + β + γ + δ = 360) 
    (h_convex : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180) :
    ∀ (x y z w : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) → (y = α ∨ y = β ∨ y = γ ∨ y = δ) → 
                     (z = α ∨ z = β ∨ z = γ ∨ z = δ) → (w = α ∨ w = β ∨ w = γ ∨ w = δ) 
                     → x + y > |z - w| := 
by
  sorry

end NUMINAMATH_GPT_convex_quad_sum_greater_diff_l1898_189850


namespace NUMINAMATH_GPT_abc_divisibility_l1898_189817

theorem abc_divisibility (a b c : ℕ) (h₁ : a ∣ (b * c - 1)) (h₂ : b ∣ (c * a - 1)) (h₃ : c ∣ (a * b - 1)) : 
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 1 ∧ b = 1 ∧ ∃ n : ℕ, n ≥ 1 ∧ c = n) :=
by
  sorry

end NUMINAMATH_GPT_abc_divisibility_l1898_189817


namespace NUMINAMATH_GPT_total_animals_l1898_189824

-- Define the number of pigs and giraffes
def num_pigs : ℕ := 7
def num_giraffes : ℕ := 6

-- Theorem stating the total number of giraffes and pigs
theorem total_animals : num_pigs + num_giraffes = 13 :=
by sorry

end NUMINAMATH_GPT_total_animals_l1898_189824


namespace NUMINAMATH_GPT_why_build_offices_l1898_189864

structure Company where
  name : String
  hasSkillfulEmployees : Prop
  uniqueComfortableWorkEnvironment : Prop
  integratedWorkLeisureSpaces : Prop
  reducedEmployeeStress : Prop
  flexibleWorkSchedules : Prop
  increasesProfit : Prop

theorem why_build_offices (goog_fb : Company)
  (h1 : goog_fb.hasSkillfulEmployees)
  (h2 : goog_fb.uniqueComfortableWorkEnvironment)
  (h3 : goog_fb.integratedWorkLeisureSpaces)
  (h4 : goog_fb.reducedEmployeeStress)
  (h5 : goog_fb.flexibleWorkSchedules) :
  goog_fb.increasesProfit := 
sorry

end NUMINAMATH_GPT_why_build_offices_l1898_189864


namespace NUMINAMATH_GPT_john_has_48_l1898_189856

variable (Ali Nada John : ℕ)

theorem john_has_48 
  (h1 : Ali + Nada + John = 67)
  (h2 : Ali = Nada - 5)
  (h3 : John = 4 * Nada) : 
  John = 48 := 
by 
  sorry

end NUMINAMATH_GPT_john_has_48_l1898_189856


namespace NUMINAMATH_GPT_maximize_NPM_l1898_189880

theorem maximize_NPM :
  ∃ (M N P : ℕ), 
    (∀ M, M < 10 → (11 * M * M) = N * 100 + P * 10 + M) →
    N * 100 + P * 10 + M = 396 :=
by
  sorry

end NUMINAMATH_GPT_maximize_NPM_l1898_189880


namespace NUMINAMATH_GPT_speed_of_car_B_is_correct_l1898_189843

def carB_speed : ℕ := 
  let speedA := 50 -- Car A's speed in km/hr
  let timeA := 6 -- Car A's travel time in hours
  let ratio := 3 -- The ratio of distances between Car A and Car B
  let distanceA := speedA * timeA -- Calculate Car A's distance
  let timeB := 1 -- Car B's travel time in hours
  let distanceB := distanceA / ratio -- Calculate Car B's distance
  distanceB / timeB -- Calculate Car B's speed

theorem speed_of_car_B_is_correct : carB_speed = 100 := by
  sorry

end NUMINAMATH_GPT_speed_of_car_B_is_correct_l1898_189843


namespace NUMINAMATH_GPT_system_real_solutions_l1898_189813

theorem system_real_solutions (a b c : ℝ) :
  (∃ x : ℝ, 
    a * x^2 + b * x + c = 0 ∧ 
    b * x^2 + c * x + a = 0 ∧ 
    c * x^2 + a * x + b = 0) ↔ 
  a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_system_real_solutions_l1898_189813


namespace NUMINAMATH_GPT_union_P_complement_Q_l1898_189868

-- Define sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_RQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the main theorem
theorem union_P_complement_Q : (P ∪ C_RQ) = {x | -2 < x ∧ x ≤ 3} := 
by
  sorry

end NUMINAMATH_GPT_union_P_complement_Q_l1898_189868


namespace NUMINAMATH_GPT_projectile_reaches_24m_at_12_7_seconds_l1898_189884

theorem projectile_reaches_24m_at_12_7_seconds :
  ∃ t : ℝ, (y = -4.9 * t^2 + 25 * t) ∧ y = 24 ∧ t = 12 / 7 :=
by
  use 12 / 7
  sorry

end NUMINAMATH_GPT_projectile_reaches_24m_at_12_7_seconds_l1898_189884


namespace NUMINAMATH_GPT_optometrist_sales_l1898_189872

noncomputable def total_pairs_optometrist_sold (H S : ℕ) (total_sales: ℝ) : Prop :=
  (S = H + 7) ∧ 
  (total_sales = 0.9 * (95 * ↑H + 175 * ↑S)) ∧ 
  (total_sales = 2469)

theorem optometrist_sales :
  ∃ H S : ℕ, total_pairs_optometrist_sold H S 2469 ∧ H + S = 17 :=
by 
  sorry

end NUMINAMATH_GPT_optometrist_sales_l1898_189872


namespace NUMINAMATH_GPT_shpuntik_can_form_triangle_l1898_189874

-- Define lengths of the sticks before swap
variables {a b c d e f : ℝ}

-- Conditions before the swap
-- Both sets of sticks can form a triangle
-- The lengths of Vintik's sticks are a, b, c
-- The lengths of Shpuntik's sticks are d, e, f
axiom triangle_ineq_vintik : a + b > c ∧ b + c > a ∧ c + a > b
axiom triangle_ineq_shpuntik : d + e > f ∧ e + f > d ∧ f + d > e
axiom sum_lengths_vintik : a + b + c = 1
axiom sum_lengths_shpuntik : d + e + f = 1

-- Define lengths of the sticks after swap
-- x1, x2, x3 are Vintik's new sticks; y1, y2, y3 are Shpuntik's new sticks
variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Neznaika's swap
axiom swap_stick_vintik : x1 = a ∧ x2 = b ∧ x3 = f ∨ x1 = a ∧ x2 = d ∧ x3 = c ∨ x1 = e ∧ x2 = b ∧ x3 = c
axiom swap_stick_shpuntik : y1 = d ∧ y2 = e ∧ y3 = c ∨ y1 = e ∧ y2 = b ∧ y3 = f ∨ y1 = a ∧ y2 = b ∧ y3 = f 

-- Total length after the swap remains unchanged
axiom sum_lengths_after_swap : x1 + x2 + x3 + y1 + y2 + y3 = 2

-- Vintik cannot form a triangle with the current lengths
axiom no_triangle_vintik : x1 >= x2 + x3

-- Prove that Shpuntik can still form a triangle
theorem shpuntik_can_form_triangle : y1 + y2 > y3 ∧ y2 + y3 > y1 ∧ y3 + y1 > y2 := sorry

end NUMINAMATH_GPT_shpuntik_can_form_triangle_l1898_189874


namespace NUMINAMATH_GPT_total_weight_of_towels_is_40_lbs_l1898_189808

def number_of_towels_Mary := 24
def factor_Mary_Frances := 4
def weight_Frances_towels_oz := 128
def pounds_per_ounce := 1 / 16

def number_of_towels_Frances := number_of_towels_Mary / factor_Mary_Frances

def total_number_of_towels := number_of_towels_Mary + number_of_towels_Frances
def weight_per_towel_oz := weight_Frances_towels_oz / number_of_towels_Frances

def total_weight_oz := total_number_of_towels * weight_per_towel_oz
def total_weight_lbs := total_weight_oz * pounds_per_ounce

theorem total_weight_of_towels_is_40_lbs :
  total_weight_lbs = 40 :=
sorry

end NUMINAMATH_GPT_total_weight_of_towels_is_40_lbs_l1898_189808


namespace NUMINAMATH_GPT_range_of_a_l1898_189834

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 + (a+1)*x + a < 0) → a ∈ Set.Iio (-2 / 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1898_189834


namespace NUMINAMATH_GPT_cost_prices_max_units_B_possible_scenarios_l1898_189838

-- Part 1: Prove cost prices of Product A and B
theorem cost_prices (x : ℝ) (A B : ℝ) 
  (h₁ : B = x ∧ A = x - 2) 
  (h₂ : 80 / A = 100 / B) 
  : B = 10 ∧ A = 8 :=
by 
  sorry

-- Part 2: Prove maximum units of product B that can be purchased
theorem max_units_B (y : ℕ) 
  (h₁ : ∀ y : ℕ, 3 * y - 5 + y ≤ 95) 
  : y ≤ 25 :=
by 
  sorry

-- Part 3: Prove possible scenarios for purchasing products A and B
theorem possible_scenarios (y : ℕ) 
  (h₁ : y > 23 * 9/17 ∧ y ≤ 25) 
  : y = 24 ∨ y = 25 :=
by 
  sorry

end NUMINAMATH_GPT_cost_prices_max_units_B_possible_scenarios_l1898_189838


namespace NUMINAMATH_GPT_unique_N_l1898_189878

-- Given conditions and question in the problem
variable (N : Matrix (Fin 2) (Fin 2) ℝ)

-- Problem statement: prove that the matrix defined below is the only matrix satisfying the given condition
theorem unique_N 
  (h : ∀ (w : Fin 2 → ℝ), N.mulVec w = -7 • w) 
  : N = ![![-7, 0], ![0, -7]] := 
sorry

end NUMINAMATH_GPT_unique_N_l1898_189878
