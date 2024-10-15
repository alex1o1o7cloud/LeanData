import Mathlib

namespace NUMINAMATH_GPT_solve_arithmetic_sequence_l518_51898

theorem solve_arithmetic_sequence :
  ∀ (x : ℝ), x > 0 ∧ x^2 = (2^2 + 5^2) / 2 → x = Real.sqrt (29 / 2) :=
by
  intro x
  intro hx
  sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_l518_51898


namespace NUMINAMATH_GPT_geometric_sequence_sum_twenty_terms_l518_51864

noncomputable def geom_seq_sum : ℕ → ℕ → ℕ := λ a r =>
  if r = 1 then a * (1 + 20 - 1) else a * ((1 - r^20) / (1 - r))

theorem geometric_sequence_sum_twenty_terms (a₁ q : ℕ) (h1 : a₁ * (q + 2) = 4) (h2 : (a₃:ℕ) * (q ^ 4) = (a₁ : ℕ) * (q ^ 4)) :
  geom_seq_sum a₁ q = 2^20 - 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_twenty_terms_l518_51864


namespace NUMINAMATH_GPT_maximize_fraction_l518_51873

theorem maximize_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
sorry

end NUMINAMATH_GPT_maximize_fraction_l518_51873


namespace NUMINAMATH_GPT_problem_statement_l518_51807
noncomputable def a : ℕ := 10
noncomputable def b : ℕ := a^3

theorem problem_statement (a b : ℕ) (a_pos : 0 < a) (b_eq : b = a^3)
    (log_ab : Real.logb a (b : ℝ) = 3) (b_minus_a : b = a + 891) :
    a + b = 1010 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l518_51807


namespace NUMINAMATH_GPT_calc_f_xh_min_f_x_l518_51849

def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

theorem calc_f_xh_min_f_x (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := 
by
  sorry

end NUMINAMATH_GPT_calc_f_xh_min_f_x_l518_51849


namespace NUMINAMATH_GPT_sample_capacity_l518_51824

theorem sample_capacity 
  (n : ℕ) 
  (model_A : ℕ) 
  (model_B model_C : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ)
  (r_A : ratio_A = 2)
  (r_B : ratio_B = 3)
  (r_C : ratio_C = 5)
  (total_production_ratio : ratio_A + ratio_B + ratio_C = 10)
  (items_model_A : model_A = 15)
  (proportion : (model_A : ℚ) / (ratio_A : ℚ) = (n : ℚ) / 10) :
  n = 75 :=
by sorry

end NUMINAMATH_GPT_sample_capacity_l518_51824


namespace NUMINAMATH_GPT_find_a_l518_51883

open Set

theorem find_a :
  ∀ (A B : Set ℕ) (a : ℕ),
    A = {1, 2, 3} →
    B = {2, a} →
    A ∪ B = {0, 1, 2, 3} →
    a = 0 :=
by
  intros A B a hA hB hUnion
  rw [hA, hB] at hUnion
  sorry

end NUMINAMATH_GPT_find_a_l518_51883


namespace NUMINAMATH_GPT_find_sum_of_vars_l518_51830

-- Definitions of the quadratic polynomials
def quadratic1 (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 11
def quadratic2 (y : ℝ) : ℝ := y^2 - 10 * y + 29
def quadratic3 (z : ℝ) : ℝ := 3 * z^2 - 18 * z + 32

-- Theorem statement
theorem find_sum_of_vars (x y z : ℝ) :
  quadratic1 x * quadratic2 y * quadratic3 z ≤ 60 → x + y - z = 0 :=
by 
-- here we would complete the proof steps
sorry

end NUMINAMATH_GPT_find_sum_of_vars_l518_51830


namespace NUMINAMATH_GPT_max_range_of_temps_l518_51811

noncomputable def max_temp_range (T1 T2 T3 T4 T5 : ℝ) : ℝ := 
  max (max (max (max T1 T2) T3) T4) T5 - min (min (min (min T1 T2) T3) T4) T5

theorem max_range_of_temps :
  ∀ (T1 T2 T3 T4 T5 : ℝ), 
  (T1 + T2 + T3 + T4 + T5) / 5 = 60 →
  T1 = 40 →
  (max_temp_range T1 T2 T3 T4 T5) = 100 :=
by
  intros T1 T2 T3 T4 T5 Havg Hlowest
  sorry

end NUMINAMATH_GPT_max_range_of_temps_l518_51811


namespace NUMINAMATH_GPT_find_M_value_l518_51894

def distinct_positive_integers (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_M_value (C y M A : ℕ) 
  (h1 : distinct_positive_integers C y M A) 
  (h2 : C + y + 2 * M + A = 11) : M = 1 :=
sorry

end NUMINAMATH_GPT_find_M_value_l518_51894


namespace NUMINAMATH_GPT_ceil_square_of_neg_seven_fourths_l518_51810

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end NUMINAMATH_GPT_ceil_square_of_neg_seven_fourths_l518_51810


namespace NUMINAMATH_GPT_no_solution_for_steers_and_cows_purchase_l518_51891

theorem no_solution_for_steers_and_cows_purchase :
  ¬ ∃ (s c : ℕ), 30 * s + 32 * c = 1200 ∧ c > s :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_steers_and_cows_purchase_l518_51891


namespace NUMINAMATH_GPT_rhombus_side_length_l518_51809

theorem rhombus_side_length (a b s K : ℝ)
  (h1 : b = 3 * a)
  (h2 : K = (1 / 2) * a * b)
  (h3 : s ^ 2 = (a / 2) ^ 2 + (3 * a / 2) ^ 2) :
  s = Real.sqrt (5 * K / 3) :=
by
  sorry

end NUMINAMATH_GPT_rhombus_side_length_l518_51809


namespace NUMINAMATH_GPT_bricks_needed_per_square_meter_l518_51817

theorem bricks_needed_per_square_meter 
  (num_rooms : ℕ) (room_length room_breadth : ℕ) (total_bricks : ℕ)
  (h1 : num_rooms = 5)
  (h2 : room_length = 4)
  (h3 : room_breadth = 5)
  (h4 : total_bricks = 340) : 
  (total_bricks / (room_length * room_breadth)) = 17 := 
by
  sorry

end NUMINAMATH_GPT_bricks_needed_per_square_meter_l518_51817


namespace NUMINAMATH_GPT_difference_of_extreme_valid_numbers_l518_51837

theorem difference_of_extreme_valid_numbers :
  ∃ (largest smallest : ℕ),
    (largest = 222210 ∧ smallest = 100002) ∧ 
    (largest % 3 = 0 ∧ smallest % 3 = 0) ∧ 
    (largest ≥ 100000 ∧ largest < 1000000) ∧
    (smallest ≥ 100000 ∧ smallest < 1000000) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10])) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10])) ∧ 
    (∀ d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10], d ∈ [0, 1, 2]) ∧
    (∀ d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10], d ∈ [0, 1, 2]) ∧
    (largest - smallest = 122208) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_extreme_valid_numbers_l518_51837


namespace NUMINAMATH_GPT_mason_internet_speed_l518_51813

-- Definitions based on the conditions
def total_data : ℕ := 880
def downloaded_data : ℕ := 310
def remaining_time : ℕ := 190

-- Statement: The speed of Mason's Internet connection after it slows down
theorem mason_internet_speed :
  (total_data - downloaded_data) / remaining_time = 3 :=
by
  sorry

end NUMINAMATH_GPT_mason_internet_speed_l518_51813


namespace NUMINAMATH_GPT_length_of_d_in_proportion_l518_51843

variable (a b c d : ℝ)

theorem length_of_d_in_proportion
  (h1 : a = 3) 
  (h2 : b = 2)
  (h3 : c = 6)
  (h_prop : a / b = c / d) : 
  d = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_d_in_proportion_l518_51843


namespace NUMINAMATH_GPT_sum_of_positive_ks_l518_51829

theorem sum_of_positive_ks :
  ∃ (S : ℤ), S = 39 ∧ ∀ k : ℤ, 
  (∃ α β : ℤ, α * β = 18 ∧ α + β = k) →
  (k > 0 → S = 19 + 11 + 9) := sorry

end NUMINAMATH_GPT_sum_of_positive_ks_l518_51829


namespace NUMINAMATH_GPT_calc_difference_of_squares_l518_51893

theorem calc_difference_of_squares :
  625^2 - 375^2 = 250000 :=
by sorry

end NUMINAMATH_GPT_calc_difference_of_squares_l518_51893


namespace NUMINAMATH_GPT_molecular_weight_CCl4_l518_51868

theorem molecular_weight_CCl4 (MW_7moles_CCl4 : ℝ) (h : MW_7moles_CCl4 = 1064) : 
  MW_7moles_CCl4 / 7 = 152 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_CCl4_l518_51868


namespace NUMINAMATH_GPT_maximize_area_CDFE_l518_51879

-- Given the side lengths of the rectangle
def AB : ℝ := 2
def AD : ℝ := 1

-- Definitions for points E and F
def AE (x : ℝ) : ℝ := x
def AF (x : ℝ) : ℝ := x

-- The formula for the area of quadrilateral CDFE
def area_CDFE (x : ℝ) : ℝ := 
  0.5 * x * (3 - 2 * x)

theorem maximize_area_CDFE : 
  ∃ x : ℝ, x = 3 / 4 ∧ area_CDFE x = 9 / 16 :=
by 
  sorry

end NUMINAMATH_GPT_maximize_area_CDFE_l518_51879


namespace NUMINAMATH_GPT_only_zero_solution_l518_51872

theorem only_zero_solution (a b c n : ℤ) (h_gcd : Int.gcd (Int.gcd (Int.gcd a b) c) n = 1)
  (h_eq : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
sorry

end NUMINAMATH_GPT_only_zero_solution_l518_51872


namespace NUMINAMATH_GPT_correct_average_calculation_l518_51895

theorem correct_average_calculation (n : ℕ) (incorrect_avg correct_num wrong_num : ℕ) (incorrect_avg_eq : incorrect_avg = 21) (n_eq : n = 10) (correct_num_eq : correct_num = 36) (wrong_num_eq : wrong_num = 26) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 22 := by
  sorry

end NUMINAMATH_GPT_correct_average_calculation_l518_51895


namespace NUMINAMATH_GPT_min_absolute_sum_value_l518_51839

def absolute_sum (x : ℝ) : ℝ :=
  abs (x + 3) + abs (x + 6) + abs (x + 7)

theorem min_absolute_sum_value : ∃ x, absolute_sum x = 4 :=
sorry

end NUMINAMATH_GPT_min_absolute_sum_value_l518_51839


namespace NUMINAMATH_GPT_largest_base5_number_conversion_l518_51880

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_largest_base5_number_conversion_l518_51880


namespace NUMINAMATH_GPT_square_difference_l518_51888

variable (n : ℕ)

theorem square_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 :=
sorry

end NUMINAMATH_GPT_square_difference_l518_51888


namespace NUMINAMATH_GPT_regular_polygon_angle_not_divisible_by_five_l518_51823

theorem regular_polygon_angle_not_divisible_by_five :
  ∃ (n_values : Finset ℕ), n_values.card = 5 ∧
    ∀ n ∈ n_values, 3 ≤ n ∧ n ≤ 15 ∧
      ¬ (∃ k : ℕ, (180 * (n - 2)) / n = 5 * k) := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_angle_not_divisible_by_five_l518_51823


namespace NUMINAMATH_GPT_remaining_sweets_in_packet_l518_51841

theorem remaining_sweets_in_packet 
  (C : ℕ) (S : ℕ) (P : ℕ) (R : ℕ) (L : ℕ)
  (HC : C = 30) (HS : S = 100) (HP : P = 60) (HR : R = 25) (HL : L = 150) 
  : (C - (2 * C / 5) - ((C - P / 4) / 3)) 
  + (S - (S / 4)) 
  + (P - (3 * P / 5)) 
  + ((max 0 (R - (3 * R / 2))))
  + (L - (3 * (S / 4) / 2)) = 232 :=
by
  sorry

end NUMINAMATH_GPT_remaining_sweets_in_packet_l518_51841


namespace NUMINAMATH_GPT_circle_eq1_circle_eq2_l518_51803

-- Problem 1: Circle with center M(-5, 3) and passing through point A(-8, -1)
theorem circle_eq1 : ∀ (x y : ℝ), (x + 5) ^ 2 + (y - 3) ^ 2 = 25 :=
by
  sorry

-- Problem 2: Circle passing through three points A(-2, 4), B(-1, 3), C(2, 6)
theorem circle_eq2 : ∀ (x y : ℝ), x ^ 2 + (y - 5) ^ 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_circle_eq1_circle_eq2_l518_51803


namespace NUMINAMATH_GPT_convert_20202_3_l518_51866

def ternary_to_decimal (a4 a3 a2 a1 a0 : ℕ) : ℕ :=
  a4 * 3^4 + a3 * 3^3 + a2 * 3^2 + a1 * 3^1 + a0 * 3^0

theorem convert_20202_3 : ternary_to_decimal 2 0 2 0 2 = 182 :=
  sorry

end NUMINAMATH_GPT_convert_20202_3_l518_51866


namespace NUMINAMATH_GPT_coordinate_sum_l518_51889

theorem coordinate_sum (f : ℝ → ℝ) (x y : ℝ) (h₁ : f 9 = 7) (h₂ : 3 * y = f (3 * x) / 3 + 3) (h₃ : x = 3) : 
  x + y = 43 / 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_coordinate_sum_l518_51889


namespace NUMINAMATH_GPT_expression_value_l518_51877

def a : ℤ := 5
def b : ℤ := -3
def c : ℕ := 2

theorem expression_value : (3 * c) / (a + b) + c = 5 := by
  sorry

end NUMINAMATH_GPT_expression_value_l518_51877


namespace NUMINAMATH_GPT_direct_proportion_function_l518_51815

-- Definitions of the given functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 4

-- Direct proportion function definition
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∀ x, f 0 = 0 ∧ (f x) / x = f 1 / 1

-- Prove that fC (x) is the only direct proportion function among the given options
theorem direct_proportion_function :
  is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_function_l518_51815


namespace NUMINAMATH_GPT_triangle_angles_l518_51820

theorem triangle_angles (A B C : ℝ) 
  (h1 : B = 4 * A)
  (h2 : C - B = 27)
  (h3 : A + B + C = 180) : 
  A = 17 ∧ B = 68 ∧ C = 95 :=
by {
  -- Sorry will be replaced once the actual proof is provided
  sorry 
}

end NUMINAMATH_GPT_triangle_angles_l518_51820


namespace NUMINAMATH_GPT_sum_of_solutions_l518_51874

theorem sum_of_solutions (x : ℝ) (h : x + (25 / x) = 10) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l518_51874


namespace NUMINAMATH_GPT_part1_part2_part3_l518_51865

noncomputable def f (x a : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem part1 (a : ℝ) (x : ℝ) (h : a = -1) : 
  (f x a = 1) ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ (a ≥ 1 / 3) :=
sorry

-- Part 3
theorem part3 (a : ℝ) (h1 : a < 1) (h2 : ∀ x : ℝ, f x a ≥ 2 * x - 3) : 
  -3 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l518_51865


namespace NUMINAMATH_GPT_relay_race_time_l518_51816

-- Define the time it takes for each runner.
def Rhonda_time : ℕ := 24
def Sally_time : ℕ := Rhonda_time + 2
def Diane_time : ℕ := Rhonda_time - 3

-- Define the total time for the relay race.
def total_relay_time : ℕ := Rhonda_time + Sally_time + Diane_time

-- State the theorem we want to prove: the total relay time is 71 seconds.
theorem relay_race_time : total_relay_time = 71 := 
by 
  -- The following "sorry" indicates a step where the proof would be completed.
  sorry

end NUMINAMATH_GPT_relay_race_time_l518_51816


namespace NUMINAMATH_GPT_coeff_x3_in_x_mul_1_add_x_pow_6_l518_51870

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem coeff_x3_in_x_mul_1_add_x_pow_6 :
  ∀ x : ℕ, (∃ c : ℕ, c * x^3 = x * (1 + x)^6 ∧ c = 15) :=
by
  sorry

end NUMINAMATH_GPT_coeff_x3_in_x_mul_1_add_x_pow_6_l518_51870


namespace NUMINAMATH_GPT_speed_downstream_l518_51850

variables (V_m V_s V_u V_d : ℕ)
variables (h1 : V_u = 12)
variables (h2 : V_m = 25)
variables (h3 : V_u = V_m - V_s)

theorem speed_downstream (h1 : V_u = 12) (h2 : V_m = 25) (h3 : V_u = V_m - V_s) :
  V_d = V_m + V_s :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_speed_downstream_l518_51850


namespace NUMINAMATH_GPT_sequence_count_l518_51860

theorem sequence_count :
  ∃ n : ℕ, 
  (∀ a : Fin 101 → ℤ, 
    a 1 = 0 ∧ 
    a 100 = 475 ∧ 
    (∀ k : ℕ, 1 ≤ k ∧ k < 100 → |a (k + 1) - a k| = 5) → 
    n = 4851) := 
sorry

end NUMINAMATH_GPT_sequence_count_l518_51860


namespace NUMINAMATH_GPT_eliot_account_balance_l518_51847

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A - 1.15 * E = 22) : 
  E = 146.67 :=
by
  sorry

end NUMINAMATH_GPT_eliot_account_balance_l518_51847


namespace NUMINAMATH_GPT_partial_fraction_product_l518_51887

theorem partial_fraction_product :
  ∃ (A B C : ℚ), 
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 → 
    (x^2 - 4) / (x^3 + x^2 - 11 * x - 13) = A / (x - 1) + B / (x + 3) + C / (x - 4)) ∧
  A * B * C = 5 / 196 :=
sorry

end NUMINAMATH_GPT_partial_fraction_product_l518_51887


namespace NUMINAMATH_GPT_total_morning_afternoon_emails_l518_51882

-- Define the conditions
def morning_emails : ℕ := 5
def afternoon_emails : ℕ := 8
def evening_emails : ℕ := 72

-- State the proof problem
theorem total_morning_afternoon_emails : 
  morning_emails + afternoon_emails = 13 := by
  sorry

end NUMINAMATH_GPT_total_morning_afternoon_emails_l518_51882


namespace NUMINAMATH_GPT_greatest_int_with_gcd_five_l518_51892

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end NUMINAMATH_GPT_greatest_int_with_gcd_five_l518_51892


namespace NUMINAMATH_GPT_range_of_a_l518_51800

theorem range_of_a (a : Real) : 
  (∀ x y : Real, (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0 → x < 0 ∧ y > 0)) ↔ (a > 2) := 
sorry

end NUMINAMATH_GPT_range_of_a_l518_51800


namespace NUMINAMATH_GPT_unique_solution_l518_51867

-- Define the system of equations
def system_of_equations (m x y : ℝ) := 
  (m + 1) * x - y - 3 * m = 0 ∧ 4 * x + (m - 1) * y + 7 = 0

-- Define the determinant condition
def determinant_nonzero (m : ℝ) := m^2 + 3 ≠ 0

-- Theorem to prove there is exactly one solution
theorem unique_solution (m x y : ℝ) : 
  determinant_nonzero m → ∃! (x y : ℝ), system_of_equations m x y :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l518_51867


namespace NUMINAMATH_GPT_max_profit_at_max_price_l518_51827

-- Definitions based on the given problem's conditions
def cost_price : ℝ := 30
def profit_margin : ℝ := 0.5
def max_price : ℝ := cost_price * (1 + profit_margin)
def min_price : ℝ := 35
def base_sales : ℝ := 350
def sales_decrease_per_price_increase : ℝ := 50
def price_increase_step : ℝ := 5

-- Profit function based on the conditions
def profit (x : ℝ) : ℝ := (-10 * x^2 + 1000 * x - 21000)

-- Maximum profit and corresponding price
theorem max_profit_at_max_price :
  ∀ x, min_price ≤ x ∧ x ≤ max_price →
  profit x ≤ profit max_price ∧ profit max_price = 3750 :=
by sorry

end NUMINAMATH_GPT_max_profit_at_max_price_l518_51827


namespace NUMINAMATH_GPT_find_x_l518_51804

-- Definitions for the vectors and their relationships
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := (a.1 + 2 * (b x).1, a.2 + 2 * (b x).2)
def v (x : ℝ) : ℝ × ℝ := (2 * a.1 - (b x).1, 2 * a.2 - (b x).2)

-- Given condition that u is parallel to v
def u_parallel_v (x : ℝ) : Prop := u x = v x

-- Prove that the value of x is 1/2
theorem find_x : ∃ x : ℝ, u_parallel_v x ∧ x = 1 / 2 := 
sorry

end NUMINAMATH_GPT_find_x_l518_51804


namespace NUMINAMATH_GPT_repeated_mul_eq_pow_l518_51881

-- Define the repeated multiplication of 2, n times
def repeated_mul (n : ℕ) : ℕ :=
  (List.replicate n 2).prod

-- State the theorem to prove
theorem repeated_mul_eq_pow (n : ℕ) : repeated_mul n = 2 ^ n :=
by
  sorry

end NUMINAMATH_GPT_repeated_mul_eq_pow_l518_51881


namespace NUMINAMATH_GPT_triangle_angle_C_triangle_max_area_l518_51808

noncomputable def cos (θ : Real) : Real := sorry
noncomputable def sin (θ : Real) : Real := sorry

theorem triangle_angle_C (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) : C = (2 * Real.pi) / 3 :=
sorry

theorem triangle_max_area (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) (hc : c = 6)
  (hC : C = (2 * Real.pi) / 3) : 
  ∃ (S : Real), S = 3 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_triangle_angle_C_triangle_max_area_l518_51808


namespace NUMINAMATH_GPT_total_cans_l518_51801

theorem total_cans (total_oil : ℕ) (oil_in_8_liter_cans : ℕ) (number_of_8_liter_cans : ℕ) (remaining_oil : ℕ) 
(oil_per_15_liter_can : ℕ) (number_of_15_liter_cans : ℕ) :
  total_oil = 290 ∧ oil_in_8_liter_cans = 8 ∧ number_of_8_liter_cans = 10 ∧ oil_per_15_liter_can = 15 ∧
  remaining_oil = total_oil - (number_of_8_liter_cans * oil_in_8_liter_cans) ∧
  number_of_15_liter_cans = remaining_oil / oil_per_15_liter_can →
  (number_of_8_liter_cans + number_of_15_liter_cans) = 24 := sorry

end NUMINAMATH_GPT_total_cans_l518_51801


namespace NUMINAMATH_GPT_barrel_capacity_l518_51842

theorem barrel_capacity (x y : ℝ) (h1 : y = 45 / (3/5)) (h2 : 0.6*x = y*3/5) (h3 : 0.4*x = 18) : 
  y = 75 :=
by
  sorry

end NUMINAMATH_GPT_barrel_capacity_l518_51842


namespace NUMINAMATH_GPT_zeroes_y_minus_a_l518_51818

open Real

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then |2 ^ x - 1| else 3 / (x - 1)

theorem zeroes_y_minus_a (a : ℝ) : (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) → (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_zeroes_y_minus_a_l518_51818


namespace NUMINAMATH_GPT_sum_of_cubes_eq_three_l518_51819

theorem sum_of_cubes_eq_three (k : ℤ) : 
  (1 + 6 * k^3)^3 + (1 - 6 * k^3)^3 + (-6 * k^2)^3 + 1^3 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_three_l518_51819


namespace NUMINAMATH_GPT_yule_log_surface_area_increase_l518_51896

noncomputable def yuleLogIncreaseSurfaceArea : ℝ := 
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initialSurfaceArea := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let sliceHeight := h / n
  let sliceSurfaceArea := 2 * Real.pi * r * sliceHeight + 2 * Real.pi * r^2
  let totalSlicesSurfaceArea := n * sliceSurfaceArea
  let increaseSurfaceArea := totalSlicesSurfaceArea - initialSurfaceArea
  increaseSurfaceArea

theorem yule_log_surface_area_increase : yuleLogIncreaseSurfaceArea = 100 * Real.pi := by
  sorry

end NUMINAMATH_GPT_yule_log_surface_area_increase_l518_51896


namespace NUMINAMATH_GPT_larry_initial_money_l518_51806

theorem larry_initial_money
  (M : ℝ)
  (spent_maintenance : ℝ := 0.04 * M)
  (saved_for_emergencies : ℝ := 0.30 * M)
  (snack_cost : ℝ := 5)
  (souvenir_cost : ℝ := 25)
  (lunch_cost : ℝ := 12)
  (loan_cost : ℝ := 10)
  (remaining_money : ℝ := 368)
  (total_spent : ℝ := snack_cost + souvenir_cost + lunch_cost + loan_cost) :
  M - spent_maintenance - saved_for_emergencies - total_spent = remaining_money →
  M = 636.36 :=
by
  sorry

end NUMINAMATH_GPT_larry_initial_money_l518_51806


namespace NUMINAMATH_GPT_Manny_lasagna_pieces_l518_51876

-- Define variables and conditions
variable (M : ℕ) -- Manny's desired number of pieces
variable (A : ℕ := 0) -- Aaron's pieces
variable (K : ℕ := 2 * M) -- Kai's pieces
variable (R : ℕ := M / 2) -- Raphael's pieces
variable (L : ℕ := 2 + R) -- Lisa's pieces

-- Prove that Manny wants 1 piece of lasagna
theorem Manny_lasagna_pieces (M : ℕ) (A : ℕ := 0) (K : ℕ := 2 * M) (R : ℕ := M / 2) (L : ℕ := 2 + R) 
  (h : M + A + K + R + L = 6) : M = 1 :=
by
  sorry

end NUMINAMATH_GPT_Manny_lasagna_pieces_l518_51876


namespace NUMINAMATH_GPT_maria_dozen_flowers_l518_51828

theorem maria_dozen_flowers (x : ℕ) (h : 12 * x + 2 * x = 42) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_maria_dozen_flowers_l518_51828


namespace NUMINAMATH_GPT_find_m_l518_51886

variable (m : ℝ)

def vector_oa : ℝ × ℝ := (-1, 2)
def vector_ob : ℝ × ℝ := (3, m)

def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_m
  (h : orthogonal (vector_oa) (vector_ob m)) :
  m = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_find_m_l518_51886


namespace NUMINAMATH_GPT_lion_weight_l518_51897

theorem lion_weight :
  ∃ (L : ℝ), 
    (∃ (T P : ℝ), 
      L + T + P = 106.6 ∧ 
      P = T - 7.7 ∧ 
      T = L - 4.8) ∧ 
    L = 41.3 :=
by
  sorry

end NUMINAMATH_GPT_lion_weight_l518_51897


namespace NUMINAMATH_GPT_train_passes_man_in_approx_18_seconds_l518_51863

noncomputable def train_length : ℝ := 300 -- meters
noncomputable def train_speed : ℝ := 68 -- km/h
noncomputable def man_speed : ℝ := 8 -- km/h
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := kmh_to_mps (train_speed - man_speed)
noncomputable def time_to_pass_man : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_18_seconds :
  abs (time_to_pass_man - 18) < 1 :=
by
  sorry

end NUMINAMATH_GPT_train_passes_man_in_approx_18_seconds_l518_51863


namespace NUMINAMATH_GPT_find_f_six_l518_51856

noncomputable def f : ℕ → ℤ := sorry

axiom f_one_eq_one : f 1 = 1
axiom f_add (x y : ℕ) : f (x + y) = f x + f y + 8 * x * y - 2
axiom f_seven_eq_163 : f 7 = 163

theorem find_f_six : f 6 = 116 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_f_six_l518_51856


namespace NUMINAMATH_GPT_water_depth_correct_l518_51878

noncomputable def water_depth (ron_height : ℝ) (dean_shorter_by : ℝ) : ℝ :=
  let dean_height := ron_height - dean_shorter_by
  2.5 * dean_height + 3

theorem water_depth_correct :
  water_depth 14.2 8.3 = 17.75 :=
by
  let ron_height := 14.2
  let dean_shorter_by := 8.3
  let dean_height := ron_height - dean_shorter_by
  let depth := 2.5 * dean_height + 3
  simp [water_depth, dean_height, depth]
  sorry

end NUMINAMATH_GPT_water_depth_correct_l518_51878


namespace NUMINAMATH_GPT_equal_sunday_tuesday_count_l518_51851

theorem equal_sunday_tuesday_count (h : ∀ (d : ℕ), d < 7 → d ≠ 0 → d ≠ 1 → d ≠ 2 → d ≠ 3) :
  ∃! d, d = 4 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_equal_sunday_tuesday_count_l518_51851


namespace NUMINAMATH_GPT_smallest_n_modulo_l518_51884

theorem smallest_n_modulo (
  n : ℕ
) (h1 : 17 * n ≡ 5678 [MOD 11]) : n = 4 :=
by sorry

end NUMINAMATH_GPT_smallest_n_modulo_l518_51884


namespace NUMINAMATH_GPT_solve_equation_l518_51861

theorem solve_equation : ∀ x : ℝ, (x + 2) / 4 - 1 = (2 * x + 1) / 3 → x = -2 :=
by
  intro x
  intro h
  sorry  

end NUMINAMATH_GPT_solve_equation_l518_51861


namespace NUMINAMATH_GPT_find_M_l518_51834

theorem find_M (a b M : ℝ) (h : (a + 2 * b)^2 = (a - 2 * b)^2 + M) : M = 8 * a * b :=
by sorry

end NUMINAMATH_GPT_find_M_l518_51834


namespace NUMINAMATH_GPT_tile_arrangement_probability_l518_51835

theorem tile_arrangement_probability :
  let X := 4  -- Number of tiles marked X
  let O := 2  -- Number of tiles marked O
  let total := 6  -- Total number of tiles
  let arrangement := [true, true, false, true, false, true]  -- XXOXOX represented as [X, X, O, X, O, X]
  (↑(X / total) * ↑((X - 1) / (total - 1)) * ↑((O / (total - 2))) * ↑((X - 2) / (total - 3)) * ↑((O - 1) / (total - 4)) * 1 : ℚ) = 1 / 15 :=
sorry

end NUMINAMATH_GPT_tile_arrangement_probability_l518_51835


namespace NUMINAMATH_GPT_find_value_of_expression_l518_51869

theorem find_value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_value_of_expression_l518_51869


namespace NUMINAMATH_GPT_least_sum_exponents_of_1000_l518_51862

def sum_least_exponents (n : ℕ) : ℕ :=
  if n = 1000 then 38 else 0 -- Since we only care about the case for 1000.

theorem least_sum_exponents_of_1000 :
  sum_least_exponents 1000 = 38 := by
  sorry

end NUMINAMATH_GPT_least_sum_exponents_of_1000_l518_51862


namespace NUMINAMATH_GPT_total_files_deleted_l518_51833

theorem total_files_deleted 
  (initial_files : ℕ) (initial_apps : ℕ)
  (deleted_files1 : ℕ) (deleted_apps1 : ℕ)
  (added_files1 : ℕ) (added_apps1 : ℕ)
  (deleted_files2 : ℕ) (deleted_apps2 : ℕ)
  (added_files2 : ℕ) (added_apps2 : ℕ)
  (final_files : ℕ) (final_apps : ℕ)
  (h_initial_files : initial_files = 24)
  (h_initial_apps : initial_apps = 13)
  (h_deleted_files1 : deleted_files1 = 5)
  (h_deleted_apps1 : deleted_apps1 = 3)
  (h_added_files1 : added_files1 = 7)
  (h_added_apps1 : added_apps1 = 4)
  (h_deleted_files2 : deleted_files2 = 10)
  (h_deleted_apps2 : deleted_apps2 = 4)
  (h_added_files2 : added_files2 = 5)
  (h_added_apps2 : added_apps2 = 7)
  (h_final_files : final_files = 21)
  (h_final_apps : final_apps = 17) :
  (deleted_files1 + deleted_files2 = 15) := 
by
  sorry

end NUMINAMATH_GPT_total_files_deleted_l518_51833


namespace NUMINAMATH_GPT_closed_fishing_season_purpose_sustainable_l518_51846

-- Defining the options for the purpose of the closed fishing season
inductive FishingPurpose
| sustainable_development : FishingPurpose
| inspect_fishing_vessels : FishingPurpose
| prevent_red_tides : FishingPurpose
| zoning_management : FishingPurpose

-- Defining rational utilization of resources involving fishing seasons
def rational_utilization (closed_fishing_season: Bool) : FishingPurpose := 
  if closed_fishing_season then FishingPurpose.sustainable_development 
  else FishingPurpose.inspect_fishing_vessels -- fallback for contradiction; shouldn't be used

-- The theorem we want to prove
theorem closed_fishing_season_purpose_sustainable :
  rational_utilization true = FishingPurpose.sustainable_development :=
sorry

end NUMINAMATH_GPT_closed_fishing_season_purpose_sustainable_l518_51846


namespace NUMINAMATH_GPT_eve_spending_l518_51852

-- Definitions of the conditions
def cost_mitt : ℝ := 14.00
def cost_apron : ℝ := 16.00
def cost_utensils : ℝ := 10.00
def cost_knife : ℝ := 2 * cost_utensils -- Twice the amount of the utensils
def discount_rate : ℝ := 0.25
def num_nieces : ℝ := 3

-- Total cost before the discount for one kit
def total_cost_one_kit : ℝ :=
  cost_mitt + cost_apron + cost_utensils + cost_knife

-- Discount for one kit
def discount_one_kit : ℝ := 
  total_cost_one_kit * discount_rate

-- Discounted price for one kit
def discounted_cost_one_kit : ℝ :=
  total_cost_one_kit - discount_one_kit

-- Total cost for all kits
def total_cost_all_kits : ℝ :=
  num_nieces * discounted_cost_one_kit

-- The theorem statement
theorem eve_spending : total_cost_all_kits = 135.00 :=
by sorry

end NUMINAMATH_GPT_eve_spending_l518_51852


namespace NUMINAMATH_GPT_arithmetic_sequence_index_l518_51875

theorem arithmetic_sequence_index {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3) (h₃ : a n = 2014) : n = 672 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_index_l518_51875


namespace NUMINAMATH_GPT_f_14_52_eq_364_l518_51822

def f : ℕ → ℕ → ℕ := sorry  -- Placeholder definition

axiom f_xx (x : ℕ) : f x x = x
axiom f_sym (x y : ℕ) : f x y = f y x
axiom f_rec (x y : ℕ) (h : x + y > 0) : (x + y) * f x y = y * f x (x + y)

theorem f_14_52_eq_364 : f 14 52 = 364 := 
by {
  sorry  -- Placeholder for the proof steps
}

end NUMINAMATH_GPT_f_14_52_eq_364_l518_51822


namespace NUMINAMATH_GPT_no_polynomial_deg_ge_3_satisfies_conditions_l518_51840

theorem no_polynomial_deg_ge_3_satisfies_conditions :
  ¬ ∃ f : Polynomial ℝ, f.degree ≥ 3 ∧ f.eval (x^2) = (f.eval x)^2 ∧ f.coeff 2 = 0 :=
sorry

end NUMINAMATH_GPT_no_polynomial_deg_ge_3_satisfies_conditions_l518_51840


namespace NUMINAMATH_GPT_max_area_of_house_l518_51890

-- Definitions for conditions
def height_of_plates : ℝ := 2.5
def price_per_meter_colored : ℝ := 450
def price_per_meter_composite : ℝ := 200
def roof_cost_per_sqm : ℝ := 200
def cost_limit : ℝ := 32000

-- Definitions for the variables
variables (x y : ℝ) (P S : ℝ)

-- Definition for the material cost P
def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

-- Maximum area S and corresponding x
theorem max_area_of_house (x y : ℝ) (h : material_cost x y ≤ cost_limit) :
  S = 100 ∧ x = 20 / 3 :=
sorry

end NUMINAMATH_GPT_max_area_of_house_l518_51890


namespace NUMINAMATH_GPT_gcd_51457_37958_l518_51832

theorem gcd_51457_37958 : Nat.gcd 51457 37958 = 1 := 
  sorry

end NUMINAMATH_GPT_gcd_51457_37958_l518_51832


namespace NUMINAMATH_GPT_new_average_weight_l518_51826

-- noncomputable theory can be enabled if necessary for real number calculations.
-- noncomputable theory

def original_players : Nat := 7
def original_avg_weight : Real := 103
def new_players : Nat := 2
def weight_first_new_player : Real := 110
def weight_second_new_player : Real := 60

theorem new_average_weight :
  let original_total_weight : Real := original_players * original_avg_weight
  let total_weight : Real := original_total_weight + weight_first_new_player + weight_second_new_player
  let total_players : Nat := original_players + new_players
  total_weight / total_players = 99 := by
  sorry

end NUMINAMATH_GPT_new_average_weight_l518_51826


namespace NUMINAMATH_GPT_kolya_or_leva_l518_51871

theorem kolya_or_leva (k l : ℝ) (hkl : k > 0) (hll : l > 0) : 
  (k > l → ∃ a b c : ℝ, a = l + (2 / 3) * (k - l) ∧ b = (1 / 6) * (k - l) ∧ c = (1 / 6) * (k - l) ∧ a > b + c + l ∧ ¬(a < b + c + a)) ∨ 
  (k ≤ l → ∃ k1 k2 k3 : ℝ, k1 ≥ k2 ∧ k2 ≥ k3 ∧ k = k1 + k2 + k3 ∧ ∃ a' b' c' : ℝ, a' = k1 ∧ b' = (l - k1) / 2 ∧ c' = (l - k1) / 2 ∧ a' + a' > k2 ∧ b' + b' > k3) :=
by sorry

end NUMINAMATH_GPT_kolya_or_leva_l518_51871


namespace NUMINAMATH_GPT_max_expression_value_l518_51821

noncomputable def A : ℝ := 15682 + (1 / 3579)
noncomputable def B : ℝ := 15682 - (1 / 3579)
noncomputable def C : ℝ := 15682 * (1 / 3579)
noncomputable def D : ℝ := 15682 / (1 / 3579)
noncomputable def E : ℝ := 15682.3579

theorem max_expression_value :
  D = 56109138 ∧ D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end NUMINAMATH_GPT_max_expression_value_l518_51821


namespace NUMINAMATH_GPT_original_amount_water_l518_51857

theorem original_amount_water (O : ℝ) (h1 : (0.75 = 0.05 * O)) : O = 15 :=
by sorry

end NUMINAMATH_GPT_original_amount_water_l518_51857


namespace NUMINAMATH_GPT_polynomial_real_root_condition_l518_51812

theorem polynomial_real_root_condition (b : ℝ) :
    (∃ x : ℝ, x^4 + b * x^3 + x^2 + b * x - 1 = 0) ↔ (b ≥ 1 / 2) :=
by sorry

end NUMINAMATH_GPT_polynomial_real_root_condition_l518_51812


namespace NUMINAMATH_GPT_cheyenne_clay_pots_l518_51825

theorem cheyenne_clay_pots (P : ℕ) (cracked_ratio sold_ratio : ℝ) (total_revenue price_per_pot : ℝ) 
    (P_sold : ℕ) :
  cracked_ratio = (2 / 5) →
  sold_ratio = (3 / 5) →
  total_revenue = 1920 →
  price_per_pot = 40 →
  P_sold = 48 →
  (sold_ratio * P = P_sold) →
  P = 80 :=
by
  sorry

end NUMINAMATH_GPT_cheyenne_clay_pots_l518_51825


namespace NUMINAMATH_GPT_walkway_area_correct_l518_51858

/-- Definitions as per problem conditions --/
def bed_length : ℕ := 8
def bed_width : ℕ := 3
def walkway_bed_width : ℕ := 2
def walkway_row_width : ℕ := 1
def num_beds_in_row : ℕ := 3
def num_rows : ℕ := 4

/-- Total dimensions including walkways --/
def total_width := num_beds_in_row * bed_length + (num_beds_in_row + 1) * walkway_bed_width
def total_height := num_rows * bed_width + (num_rows + 1) * walkway_row_width

/-- Total areas --/
def total_area := total_width * total_height
def bed_area := bed_length * bed_width
def total_bed_area := num_beds_in_row * num_rows * bed_area
def walkway_area := total_area - total_bed_area

theorem walkway_area_correct : walkway_area = 256 := by
  /- Import necessary libraries and skip the proof -/
  sorry

end NUMINAMATH_GPT_walkway_area_correct_l518_51858


namespace NUMINAMATH_GPT_altitude_circumradius_relation_l518_51848

variable (a b c R ha : ℝ)
-- Assume S is the area of the triangle
variable (S : ℝ)
-- conditions
axiom area_circumradius : S = (a * b * c) / (4 * R)
axiom area_altitude : S = (a * ha) / 2

-- Prove the equivalence
theorem altitude_circumradius_relation 
  (area_circumradius : S = (a * b * c) / (4 * R)) 
  (area_altitude : S = (a * ha) / 2) : 
  ha = (b * c) / (2 * R) :=
sorry

end NUMINAMATH_GPT_altitude_circumradius_relation_l518_51848


namespace NUMINAMATH_GPT_complement_intersection_l518_51836

def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | x < 2 }
def CR (S : Set ℝ) : Set ℝ := { x | x ∉ S }

theorem complement_intersection :
  CR (M ∩ N) = { x | x < 1 } ∪ { x | x ≥ 2 } := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l518_51836


namespace NUMINAMATH_GPT_simplify_and_multiply_l518_51831

theorem simplify_and_multiply :
  let a := 3
  let b := 17
  let d1 := 504
  let d2 := 72
  let m := 5
  let n := 7
  let fraction1 := a / d1
  let fraction2 := b / d2
  ((fraction1 - (b * n / (d2 * n))) * (m / n)) = (-145 / 882) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_multiply_l518_51831


namespace NUMINAMATH_GPT_TableCostEquals_l518_51844

-- Define the given conditions and final result
def total_spent : ℕ := 56
def num_chairs : ℕ := 2
def chair_cost : ℕ := 11
def table_cost : ℕ := 34

-- State the assertion to be proved
theorem TableCostEquals :
  table_cost = total_spent - (num_chairs * chair_cost) := 
by 
  sorry

end NUMINAMATH_GPT_TableCostEquals_l518_51844


namespace NUMINAMATH_GPT_batsman_inning_problem_l518_51885

-- Define the problem in Lean 4
theorem batsman_inning_problem (n R : ℕ) (h1 : R = 55 * n) (h2 : R + 110 = 60 * (n + 1)) : n + 1 = 11 := 
  sorry

end NUMINAMATH_GPT_batsman_inning_problem_l518_51885


namespace NUMINAMATH_GPT_correct_average_l518_51802

-- let's define the numbers as a list
def numbers : List ℕ := [1200, 1300, 1510, 1520, 1530, 1200]

-- the condition given in the problem: the stated average is 1380
def stated_average : ℕ := 1380

-- given the correct calculation of average, let's write the theorem statement
theorem correct_average : (numbers.foldr (· + ·) 0) / numbers.length = 1460 :=
by
  -- we would prove it here
  sorry

end NUMINAMATH_GPT_correct_average_l518_51802


namespace NUMINAMATH_GPT_factorization_correct_l518_51899

noncomputable def factor_expression (y : ℝ) : ℝ :=
  3 * y * (y - 5) + 4 * (y - 5)

theorem factorization_correct (y : ℝ) : factor_expression y = (3 * y + 4) * (y - 5) :=
by sorry

end NUMINAMATH_GPT_factorization_correct_l518_51899


namespace NUMINAMATH_GPT_original_stone_count_145_l518_51853

theorem original_stone_count_145 : 
  ∃ (n : ℕ), (n ≡ 1 [MOD 18]) ∧ (n = 145) :=
by
  sorry

end NUMINAMATH_GPT_original_stone_count_145_l518_51853


namespace NUMINAMATH_GPT_donuts_per_student_l518_51855

theorem donuts_per_student 
    (dozens_of_donuts : ℕ)
    (students_in_class : ℕ)
    (percentage_likes_donuts : ℕ)
    (students_who_like_donuts : ℕ)
    (total_donuts : ℕ)
    (donuts_per_student : ℕ) :
    dozens_of_donuts = 4 →
    students_in_class = 30 →
    percentage_likes_donuts = 80 →
    students_who_like_donuts = (percentage_likes_donuts * students_in_class) / 100 →
    total_donuts = dozens_of_donuts * 12 →
    donuts_per_student = total_donuts / students_who_like_donuts →
    donuts_per_student = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_donuts_per_student_l518_51855


namespace NUMINAMATH_GPT_geometric_common_ratio_l518_51854

theorem geometric_common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 5 * d)^2 = a1 * (a1 + 20 * d)) : 
  (a1 + 5 * d) / a1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_common_ratio_l518_51854


namespace NUMINAMATH_GPT_angles_around_point_sum_l518_51814

theorem angles_around_point_sum 
  (x y : ℝ)
  (h1 : 130 + x + y = 360)
  (h2 : y = x + 30) :
  x = 100 ∧ y = 130 :=
by
  sorry

end NUMINAMATH_GPT_angles_around_point_sum_l518_51814


namespace NUMINAMATH_GPT_digit_at_position_2020_l518_51838

def sequence_digit (n : Nat) : Nat :=
  -- Function to return the nth digit of the sequence formed by concatenating the integers from 1 to 1000
  sorry

theorem digit_at_position_2020 : sequence_digit 2020 = 7 :=
  sorry

end NUMINAMATH_GPT_digit_at_position_2020_l518_51838


namespace NUMINAMATH_GPT_john_spent_on_sweets_l518_51859

theorem john_spent_on_sweets (initial_amount : ℝ) (amount_given_per_friend : ℝ) (friends : ℕ) (amount_left : ℝ) (total_spent_on_sweets : ℝ) :
  initial_amount = 20.10 →
  amount_given_per_friend = 1.00 →
  friends = 2 →
  amount_left = 17.05 →
  total_spent_on_sweets = initial_amount - (amount_given_per_friend * friends) - amount_left →
  total_spent_on_sweets = 1.05 :=
by
  intros h_initial h_given h_friends h_left h_spent
  sorry

end NUMINAMATH_GPT_john_spent_on_sweets_l518_51859


namespace NUMINAMATH_GPT_prime_has_property_p_l518_51845

theorem prime_has_property_p (n : ℕ) (hn : Prime n) (a : ℕ) (h : n ∣ a^n - 1) : n^2 ∣ a^n - 1 := by
  sorry

end NUMINAMATH_GPT_prime_has_property_p_l518_51845


namespace NUMINAMATH_GPT_digit_product_inequality_l518_51805

noncomputable def digit_count_in_n (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).count d

theorem digit_product_inequality (n : ℕ) (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ)
  (h1 : a1 = digit_count_in_n n 1)
  (h2 : a2 = digit_count_in_n n 2)
  (h3 : a3 = digit_count_in_n n 3)
  (h4 : a4 = digit_count_in_n n 4)
  (h5 : a5 = digit_count_in_n n 5)
  (h6 : a6 = digit_count_in_n n 6)
  (h7 : a7 = digit_count_in_n n 7)
  (h8 : a8 = digit_count_in_n n 8)
  (h9 : a9 = digit_count_in_n n 9)
  : 2^a1 * 3^a2 * 4^a3 * 5^a4 * 6^a5 * 7^a6 * 8^a7 * 9^a8 * 10^a9 ≤ n + 1 :=
  sorry

end NUMINAMATH_GPT_digit_product_inequality_l518_51805
