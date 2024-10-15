import Mathlib

namespace NUMINAMATH_GPT_quadratic_equation_unique_solution_l1138_113825

theorem quadratic_equation_unique_solution (p : ℚ) :
  (∃ x : ℚ, 3 * x^2 - 7 * x + p = 0) ∧ 
  ∀ y : ℚ, 3 * y^2 -7 * y + p ≠ 0 → ∀ z : ℚ, 3 * z^2 - 7 * z + p = 0 → y = z ↔ 
  p = 49 / 12 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_unique_solution_l1138_113825


namespace NUMINAMATH_GPT_eq_one_solution_in_interval_l1138_113814

theorem eq_one_solution_in_interval (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (2 * a * x^2 - x - 1 = 0) ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 1 ∧ y ≠ x → (2 * a * y^2 - y - 1 ≠ 0))) → (1 < a) :=
by
  sorry

end NUMINAMATH_GPT_eq_one_solution_in_interval_l1138_113814


namespace NUMINAMATH_GPT_find_parallelepiped_dimensions_l1138_113843

theorem find_parallelepiped_dimensions :
  ∃ (x y z : ℕ),
    (x * y * z = 2 * (x * y + y * z + z * x)) ∧
    (x = 6 ∧ y = 6 ∧ z = 6 ∨
     x = 5 ∧ y = 5 ∧ z = 10 ∨
     x = 4 ∧ y = 8 ∧ z = 8 ∨
     x = 3 ∧ y = 12 ∧ z = 12 ∨
     x = 3 ∧ y = 7 ∧ z = 42 ∨
     x = 3 ∧ y = 8 ∧ z = 24 ∨
     x = 3 ∧ y = 9 ∧ z = 18 ∨
     x = 3 ∧ y = 10 ∧ z = 15 ∨
     x = 4 ∧ y = 5 ∧ z = 20 ∨
     x = 4 ∧ y = 6 ∧ z = 12) :=
by
  sorry

end NUMINAMATH_GPT_find_parallelepiped_dimensions_l1138_113843


namespace NUMINAMATH_GPT_total_sample_size_l1138_113803

theorem total_sample_size
    (undergrad_count : ℕ) (masters_count : ℕ) (doctoral_count : ℕ)
    (total_students : ℕ) (sample_size_doctoral : ℕ) (proportion_sample : ℕ)
    (n : ℕ)
    (H1 : undergrad_count = 12000)
    (H2 : masters_count = 1000)
    (H3 : doctoral_count = 200)
    (H4 : total_students = undergrad_count + masters_count + doctoral_count)
    (H5 : sample_size_doctoral = 20)
    (H6 : proportion_sample = sample_size_doctoral / doctoral_count)
    (H7 : n = proportion_sample * total_students) :
  n = 1320 := 
sorry

end NUMINAMATH_GPT_total_sample_size_l1138_113803


namespace NUMINAMATH_GPT_complex_quadrant_l1138_113884

open Complex

theorem complex_quadrant :
  let z := (1 - I) * (3 + I)
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l1138_113884


namespace NUMINAMATH_GPT_composite_has_at_least_three_divisors_l1138_113846

def is_composite (n : ℕ) : Prop := ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_divisors (n : ℕ) (h : is_composite n) : ∃ a b c, a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c :=
sorry

end NUMINAMATH_GPT_composite_has_at_least_three_divisors_l1138_113846


namespace NUMINAMATH_GPT_find_perimeter_correct_l1138_113860

noncomputable def find_perimeter (L W : ℝ) (x : ℝ) :=
  L * W = (L + 6) * (W - 2) ∧
  L * W = (L - 12) * (W + 6) ∧
  x = 2 * (L + W)

theorem find_perimeter_correct : ∀ (L W : ℝ), L * W = (L + 6) * (W - 2) → 
                                      L * W = (L - 12) * (W + 6) → 
                                      2 * (L + W) = 132 :=
sorry

end NUMINAMATH_GPT_find_perimeter_correct_l1138_113860


namespace NUMINAMATH_GPT_total_amount_spent_l1138_113853

theorem total_amount_spent (avg_price_goat : ℕ) (num_goats : ℕ) (avg_price_cow : ℕ) (num_cows : ℕ) (total_spent : ℕ) 
  (h1 : avg_price_goat = 70) (h2 : num_goats = 10) (h3 : avg_price_cow = 400) (h4 : num_cows = 2) :
  total_spent = 1500 :=
by
  have cost_goats := avg_price_goat * num_goats
  have cost_cows := avg_price_cow * num_cows
  have total := cost_goats + cost_cows
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1138_113853


namespace NUMINAMATH_GPT_length_of_integer_eq_24_l1138_113824

theorem length_of_integer_eq_24 (k : ℕ) (h1 : k > 1) (h2 : ∃ (p1 p2 p3 p4 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ k = p1 * p2 * p3 * p4) : k = 24 := by
  sorry

end NUMINAMATH_GPT_length_of_integer_eq_24_l1138_113824


namespace NUMINAMATH_GPT_sum_two_primes_eq_91_prod_is_178_l1138_113849

theorem sum_two_primes_eq_91_prod_is_178
  (p1 p2 : ℕ) 
  (hp1 : p1.Prime) 
  (hp2 : p2.Prime) 
  (h_sum : p1 + p2 = 91) :
  p1 * p2 = 178 := 
sorry

end NUMINAMATH_GPT_sum_two_primes_eq_91_prod_is_178_l1138_113849


namespace NUMINAMATH_GPT_average_salary_l1138_113802

theorem average_salary (T_salary : ℕ) (R_salary : ℕ) (total_salary : ℕ) (T_count : ℕ) (R_count : ℕ) (total_count : ℕ) :
    T_salary = 12000 * T_count →
    R_salary = 6000 * R_count →
    total_salary = T_salary + R_salary →
    T_count = 6 →
    R_count = total_count - T_count →
    total_count = 18 →
    (total_salary / total_count) = 8000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_salary_l1138_113802


namespace NUMINAMATH_GPT_prove_sums_l1138_113864

-- Given conditions
def condition1 (a b : ℤ) : Prop := ∀ x : ℝ, (x + a) * (x + b) = x^2 + 9 * x + 14
def condition2 (b c : ℤ) : Prop := ∀ x : ℝ, (x + b) * (x - c) = x^2 + 7 * x - 30

-- We need to prove that a + b + c = 15
theorem prove_sums (a b c : ℤ) (h1: condition1 a b) (h2: condition2 b c) : a + b + c = 15 := 
sorry

end NUMINAMATH_GPT_prove_sums_l1138_113864


namespace NUMINAMATH_GPT_find_min_values_l1138_113871

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 - 2 * x * y + 6 * y^2 - 14 * x - 6 * y + 72

theorem find_min_values :
  (∀x y : ℝ, f x y ≥ f (15 / 2) (1 / 2)) ∧ f (15 / 2) (1 / 2) = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_find_min_values_l1138_113871


namespace NUMINAMATH_GPT_gcd_324_243_135_l1138_113819

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 :=
by
  sorry

end NUMINAMATH_GPT_gcd_324_243_135_l1138_113819


namespace NUMINAMATH_GPT_multiplicative_inverse_l1138_113830

def A : ℕ := 123456
def B : ℕ := 162738
def N : ℕ := 503339
def modulo : ℕ := 1000000

theorem multiplicative_inverse :
  (A * B * N) % modulo = 1 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_l1138_113830


namespace NUMINAMATH_GPT_range_of_k_l1138_113863

theorem range_of_k {x y k : ℝ} :
  (∀ x y, 2 * x - y ≤ 1 ∧ x + y ≥ 2 ∧ y - x ≤ 2) →
  (z = k * x + 2 * y) →
  (∀ (x y : ℝ), z = k * x + 2 * y → (x = 1) ∧ (y = 1)) →
  -4 < k ∧ k < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1138_113863


namespace NUMINAMATH_GPT_find_possible_values_of_a_l1138_113812

theorem find_possible_values_of_a (a b c : ℝ) (h1 : a * b + a + b = c) (h2 : b * c + b + c = a) (h3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_possible_values_of_a_l1138_113812


namespace NUMINAMATH_GPT_action_figure_value_l1138_113855

theorem action_figure_value (
    V1 V2 V3 V4 : ℝ
) : 5 * 15 = 75 ∧ 
    V1 - 5 + V2 - 5 + V3 - 5 + V4 - 5 + (20 - 5) = 55 ∧
    V1 + V2 + V3 + V4 + 20 = 80 → 
    ∀ i, i = 15 := by
    sorry

end NUMINAMATH_GPT_action_figure_value_l1138_113855


namespace NUMINAMATH_GPT_rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l1138_113878

variables (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0)

def S1 := (x + 5) * (y + 5)
def S2 := (x - 2) * (y - 2)
def perimeter := 2 * (x + y)

theorem rectangle_perimeter (h : S1 - S2 = 196) :
  perimeter = 50 :=
sorry

theorem difference_multiple_of_7 (h : S1 - S2 = 196) :
  ∃ k : ℕ, S1 - S2 = 7 * k :=
sorry

theorem area_seamless_combination (h : S1 - S2 = 196) :
  S1 - x * y = (x + 5) * (y + 5) - x * y ∧ x = y + 5 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l1138_113878


namespace NUMINAMATH_GPT_defective_probability_bayesian_probabilities_l1138_113829

noncomputable def output_proportion_A : ℝ := 0.25
noncomputable def output_proportion_B : ℝ := 0.35
noncomputable def output_proportion_C : ℝ := 0.40

noncomputable def defect_rate_A : ℝ := 0.05
noncomputable def defect_rate_B : ℝ := 0.04
noncomputable def defect_rate_C : ℝ := 0.02

noncomputable def probability_defective : ℝ :=
  output_proportion_A * defect_rate_A +
  output_proportion_B * defect_rate_B +
  output_proportion_C * defect_rate_C 

theorem defective_probability :
  probability_defective = 0.0345 := 
  by sorry

noncomputable def P_A_given_defective : ℝ :=
  (output_proportion_A * defect_rate_A) / probability_defective

noncomputable def P_B_given_defective : ℝ :=
  (output_proportion_B * defect_rate_B) / probability_defective

noncomputable def P_C_given_defective : ℝ :=
  (output_proportion_C * defect_rate_C) / probability_defective

theorem bayesian_probabilities :
  P_A_given_defective = 25 / 69 ∧
  P_B_given_defective = 28 / 69 ∧
  P_C_given_defective = 16 / 69 :=
  by sorry

end NUMINAMATH_GPT_defective_probability_bayesian_probabilities_l1138_113829


namespace NUMINAMATH_GPT_sum_of_roots_l1138_113838

theorem sum_of_roots (a β : ℝ) 
  (h1 : a^2 - 2 * a = 1) 
  (h2 : β^2 - 2 * β - 1 = 0) 
  (hne : a ≠ β) 
  : a + β = 2 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_l1138_113838


namespace NUMINAMATH_GPT_quadrilateral_choices_l1138_113889

theorem quadrilateral_choices :
  let available_rods : List ℕ := (List.range' 1 41).diff [5, 12, 20]
  let valid_rods := available_rods.filter (λ x => 4 ≤ x ∧ x ≤ 36)
  valid_rods.length = 30 := sorry

end NUMINAMATH_GPT_quadrilateral_choices_l1138_113889


namespace NUMINAMATH_GPT_correct_factorization_l1138_113847

theorem correct_factorization :
  (x^2 - 2 * x + 1 = (x - 1)^2) ∧ 
  (¬ (x^2 - 4 * y^2 = (x + y) * (x - 4 * y))) ∧ 
  (¬ ((x + 4) * (x - 4) = x^2 - 16)) ∧ 
  (¬ (x^2 - 8 * x + 9 = (x - 4)^2 - 7)) :=
by
  sorry

end NUMINAMATH_GPT_correct_factorization_l1138_113847


namespace NUMINAMATH_GPT_find_BF_pqsum_l1138_113813

noncomputable def square_side_length : ℝ := 900
noncomputable def EF_length : ℝ := 400
noncomputable def m_angle_EOF : ℝ := 45
noncomputable def center_mid_to_side : ℝ := square_side_length / 2

theorem find_BF_pqsum :
  let G_mid : ℝ := center_mid_to_side
  let x : ℝ := G_mid - (2 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let y : ℝ := (1 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let BF := G_mid - y
  BF = 250 + 50 * Real.sqrt 7 ->
  250 + 50 + 7 = 307 := sorry

end NUMINAMATH_GPT_find_BF_pqsum_l1138_113813


namespace NUMINAMATH_GPT_remainder_when_4_pow_2023_div_17_l1138_113826

theorem remainder_when_4_pow_2023_div_17 :
  ∀ (x : ℕ), (x = 4) → x^2 ≡ 16 [MOD 17] → x^2023 ≡ 13 [MOD 17] := by
  intros x hx h
  sorry

end NUMINAMATH_GPT_remainder_when_4_pow_2023_div_17_l1138_113826


namespace NUMINAMATH_GPT_total_calories_consumed_l1138_113805

def caramel_cookies := 10
def caramel_calories := 18

def chocolate_chip_cookies := 8
def chocolate_chip_calories := 22

def peanut_butter_cookies := 7
def peanut_butter_calories := 24

def selected_caramel_cookies := 5
def selected_chocolate_chip_cookies := 3
def selected_peanut_butter_cookies := 2

theorem total_calories_consumed : 
  (selected_caramel_cookies * caramel_calories) + 
  (selected_chocolate_chip_cookies * chocolate_chip_calories) + 
  (selected_peanut_butter_cookies * peanut_butter_calories) = 204 := 
by
  sorry

end NUMINAMATH_GPT_total_calories_consumed_l1138_113805


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_circle_l1138_113809

theorem sufficient_but_not_necessary_condition_circle {a : ℝ} (h : a = 1) :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 → (∀ a, a < 2 → (x - 1)^2 + (y + 1)^2 = 2 - a) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_circle_l1138_113809


namespace NUMINAMATH_GPT_negation_of_one_odd_l1138_113811

-- Given a, b, c are natural numbers
def exactly_one_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 1)

def not_exactly_one_odd (a b c : ℕ) : Prop :=
  ¬ exactly_one_odd a b c

def at_least_two_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1) ∨
  (a % 2 = 1 ∧ c % 2 = 1) ∨
  (b % 2 = 1 ∧ c % 2 = 1)

def all_even (a b c : ℕ) : Prop :=
  (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0)

theorem negation_of_one_odd (a b c : ℕ) : ¬ exactly_one_odd a b c ↔ all_even a b c ∨ at_least_two_odd a b c := by
  sorry

end NUMINAMATH_GPT_negation_of_one_odd_l1138_113811


namespace NUMINAMATH_GPT_total_number_of_items_in_base10_l1138_113862

theorem total_number_of_items_in_base10 : 
  let clay_tablets := (2 * 5^0 + 3 * 5^1 + 4 * 5^2 + 1 * 5^3)
  let bronze_sculptures := (1 * 5^0 + 4 * 5^1 + 0 * 5^2 + 2 * 5^3)
  let stone_carvings := (2 * 5^0 + 3 * 5^1 + 2 * 5^2)
  let total_items := clay_tablets + bronze_sculptures + stone_carvings
  total_items = 580 := by
  sorry

end NUMINAMATH_GPT_total_number_of_items_in_base10_l1138_113862


namespace NUMINAMATH_GPT_smallest_special_integer_l1138_113807

noncomputable def is_special (N : ℕ) : Prop :=
  N > 1 ∧ 
  (N % 8 = 1) ∧ 
  (2 * 8 ^ Nat.log N (8) / 2 > N / 8 ^ Nat.log N (8)) ∧ 
  (N % 9 = 1) ∧ 
  (2 * 9 ^ Nat.log N (9) / 2 > N / 9 ^ Nat.log N (9))

theorem smallest_special_integer : ∃ (N : ℕ), is_special N ∧ N = 793 :=
by 
  use 793
  sorry

end NUMINAMATH_GPT_smallest_special_integer_l1138_113807


namespace NUMINAMATH_GPT_domain_function_1_domain_function_2_domain_function_3_l1138_113854

-- Define the conditions and the required domain equivalence in Lean 4
-- Problem (1)
theorem domain_function_1 (x : ℝ): x + 2 ≠ 0 ∧ x + 5 ≥ 0 ↔ x ≥ -5 ∧ x ≠ -2 := 
sorry

-- Problem (2)
theorem domain_function_2 (x : ℝ): x^2 - 4 ≥ 0 ∧ 4 - x^2 ≥ 0 ∧ x^2 - 9 ≠ 0 ↔ (x = 2 ∨ x = -2) :=
sorry

-- Problem (3)
theorem domain_function_3 (x : ℝ): x - 5 ≥ 0 ∧ |x| ≠ 7 ↔ x ≥ 5 ∧ x ≠ 7 :=
sorry

end NUMINAMATH_GPT_domain_function_1_domain_function_2_domain_function_3_l1138_113854


namespace NUMINAMATH_GPT_firefighters_time_to_extinguish_fire_l1138_113800

theorem firefighters_time_to_extinguish_fire (gallons_per_minute_per_hose : ℕ) (total_gallons : ℕ) (number_of_firefighters : ℕ)
  (H1 : gallons_per_minute_per_hose = 20)
  (H2 : total_gallons = 4000)
  (H3 : number_of_firefighters = 5): 
  (total_gallons / (gallons_per_minute_per_hose * number_of_firefighters)) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_firefighters_time_to_extinguish_fire_l1138_113800


namespace NUMINAMATH_GPT_monitor_height_l1138_113844

theorem monitor_height (width circumference : ℕ) (h_width : width = 12) (h_circumference : circumference = 38) :
  2 * (width + 7) = circumference :=
by
  sorry

end NUMINAMATH_GPT_monitor_height_l1138_113844


namespace NUMINAMATH_GPT_average_speed_l1138_113841

theorem average_speed
    (distance1 distance2 : ℕ)
    (time1 time2 : ℕ)
    (h1 : distance1 = 100)
    (h2 : distance2 = 80)
    (h3 : time1 = 1)
    (h4 : time2 = 1) :
    (distance1 + distance2) / (time1 + time2) = 90 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l1138_113841


namespace NUMINAMATH_GPT_twenty_million_in_scientific_notation_l1138_113833

/-- Prove that 20 million in scientific notation is 2 * 10^7 --/
theorem twenty_million_in_scientific_notation : 20000000 = 2 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_twenty_million_in_scientific_notation_l1138_113833


namespace NUMINAMATH_GPT_big_container_capacity_l1138_113801

theorem big_container_capacity (C : ℝ)
    (h1 : 0.75 * C - 0.40 * C = 14) : C = 40 :=
  sorry

end NUMINAMATH_GPT_big_container_capacity_l1138_113801


namespace NUMINAMATH_GPT_min_supreme_supervisors_l1138_113804

-- Definitions
def num_employees : ℕ := 50000
def supervisors (e : ℕ) : ℕ := 7 - e

-- Theorem statement
theorem min_supreme_supervisors (k : ℕ) (num_employees_le_reached : ∀ n : ℕ, 50000 ≤ n) : 
  k ≥ 28 := 
sorry

end NUMINAMATH_GPT_min_supreme_supervisors_l1138_113804


namespace NUMINAMATH_GPT_anthony_more_shoes_than_jim_l1138_113893

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem anthony_more_shoes_than_jim : (anthony_shoes - jim_shoes) = 2 :=
by
  sorry

end NUMINAMATH_GPT_anthony_more_shoes_than_jim_l1138_113893


namespace NUMINAMATH_GPT_sonika_initial_deposit_l1138_113850

variable (P R : ℝ)

theorem sonika_initial_deposit :
  (P + (P * R * 3) / 100 = 9200) → (P + (P * (R + 2.5) * 3) / 100 = 9800) → P = 8000 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sonika_initial_deposit_l1138_113850


namespace NUMINAMATH_GPT_seq_1000_eq_2098_l1138_113861

-- Define the sequence a_n
def seq (n : ℕ) : ℤ := sorry

-- Initial conditions
axiom a1 : seq 1 = 100
axiom a2 : seq 2 = 101

-- Recurrence relation condition
axiom recurrence_relation : ∀ n : ℕ, 1 ≤ n → seq n + seq (n+1) + seq (n+2) = 2 * ↑n + 3

-- Main theorem to prove
theorem seq_1000_eq_2098 : seq 1000 = 2098 :=
by {
  sorry
}

end NUMINAMATH_GPT_seq_1000_eq_2098_l1138_113861


namespace NUMINAMATH_GPT_point_in_third_quadrant_l1138_113820

theorem point_in_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : -m^2 < 0 ∧ -n < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l1138_113820


namespace NUMINAMATH_GPT_team_a_completion_rate_l1138_113891

theorem team_a_completion_rate :
  ∃ x : ℝ, (9000 / x - 9000 / (1.5 * x) = 15) ∧ x = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_team_a_completion_rate_l1138_113891


namespace NUMINAMATH_GPT_simple_interest_principal_l1138_113842

theorem simple_interest_principal
  (P_CI : ℝ)
  (r_CI t_CI : ℝ)
  (CI : ℝ)
  (P_SI : ℝ)
  (r_SI t_SI SI : ℝ)
  (h_compound_interest : (CI = P_CI * (1 + r_CI / 100)^t_CI - P_CI))
  (h_simple_interest : SI = (1 / 2) * CI)
  (h_SI_formula : SI = P_SI * r_SI * t_SI / 100) :
  P_SI = 1750 :=
by
  have P_CI := 4000
  have r_CI := 10
  have t_CI := 2
  have r_SI := 8
  have t_SI := 3
  have CI := 840
  have SI := 420
  sorry

end NUMINAMATH_GPT_simple_interest_principal_l1138_113842


namespace NUMINAMATH_GPT_inequality_solution_l1138_113858

theorem inequality_solution (x : ℝ) : (5 < x ∧ x ≤ 6) ↔ (x-3)/(x-5) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1138_113858


namespace NUMINAMATH_GPT_inequality_1_inequality_2_inequality_3_inequality_4_l1138_113866

-- Definition for the first problem
theorem inequality_1 (x : ℝ) : |2 * x - 1| < 15 ↔ (-7 < x ∧ x < 8) := by
  sorry
  
-- Definition for the second problem
theorem inequality_2 (x : ℝ) : x^2 + 6 * x - 16 < 0 ↔ (-8 < x ∧ x < 2) := by
  sorry

-- Definition for the third problem
theorem inequality_3 (x : ℝ) : |2 * x + 1| > 13 ↔ (x < -7 ∨ x > 6) := by
  sorry

-- Definition for the fourth problem
theorem inequality_4 (x : ℝ) : x^2 - 2 * x > 0 ↔ (x < 0 ∨ x > 2) := by
  sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_inequality_3_inequality_4_l1138_113866


namespace NUMINAMATH_GPT_rich_knight_l1138_113873

-- Definitions for the problem
inductive Status
| knight  -- Always tells the truth
| knave   -- Always lies

def tells_truth (s : Status) : Prop := 
  s = Status.knight

def lies (s : Status) : Prop := 
  s = Status.knave

def not_poor (s : Status) : Prop := 
  s = Status.knight ∨ s = Status.knave -- Knights can either be poor or wealthy

def wealthy (s : Status) : Prop :=
  s = Status.knight

-- Statement to be proven
theorem rich_knight (s : Status) (h_truth : tells_truth s) (h_not_poor : not_poor s) : wealthy s :=
by
  sorry

end NUMINAMATH_GPT_rich_knight_l1138_113873


namespace NUMINAMATH_GPT_find_k_l1138_113899

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

theorem find_k (k : ℝ) (h : f 5 - g 5 k = 24) : k = -16.36 := by
  sorry

end NUMINAMATH_GPT_find_k_l1138_113899


namespace NUMINAMATH_GPT_alyssa_games_next_year_l1138_113857

/-- Alyssa went to 11 games this year -/
def games_this_year : ℕ := 11

/-- Alyssa went to 13 games last year -/
def games_last_year : ℕ := 13

/-- Alyssa will go to a total of 39 games -/
def total_games : ℕ := 39

/-- Alyssa plans to go to 15 games next year -/
theorem alyssa_games_next_year : 
  games_this_year + games_last_year <= total_games ∧
  total_games - (games_this_year + games_last_year) = 15 := by {
  sorry
}

end NUMINAMATH_GPT_alyssa_games_next_year_l1138_113857


namespace NUMINAMATH_GPT_circle_diameter_l1138_113839

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_l1138_113839


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1138_113897

open Real

def ellipse_foci_x_axis (m : ℝ) : Prop :=
  ∃ a b c e,
    a = sqrt m ∧
    b = sqrt 6 ∧
    c = sqrt (m - 6) ∧
    e = c / a ∧
    e = 1 / 2

theorem ellipse_eccentricity (m : ℝ) (h : ellipse_foci_x_axis m) :
  m = 8 := by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1138_113897


namespace NUMINAMATH_GPT_fish_distribution_l1138_113856

theorem fish_distribution 
  (fish_caught : ℕ)
  (eyes_per_fish : ℕ := 2)
  (total_eyes : ℕ := 24)
  (people : ℕ := 3)
  (eyes_eaten_by_dog : ℕ := 2)
  (eyes_eaten_by_oomyapeck : ℕ := 22)
  (oomyapeck_total_eyes : eyes_eaten_by_oomyapeck + eyes_eaten_by_dog = total_eyes)
  (fish_per_person := fish_caught / people)
  (fish_eyes_relation : total_eyes = eyes_per_fish * fish_caught) :
  fish_per_person = 4 := by
  sorry

end NUMINAMATH_GPT_fish_distribution_l1138_113856


namespace NUMINAMATH_GPT_acute_angle_sine_diff_l1138_113831

theorem acute_angle_sine_diff (α β : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : Real.sin α = (Real.sqrt 5) / 5) (h₃ : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 :=
sorry

end NUMINAMATH_GPT_acute_angle_sine_diff_l1138_113831


namespace NUMINAMATH_GPT_intersection_solution_l1138_113823

-- Define lines
def line1 (x : ℝ) : ℝ := -x + 4
def line2 (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

-- Define system of equations
def system1 (x y : ℝ) : Prop := x + y = 4
def system2 (x y m : ℝ) : Prop := 2 * x - y + m = 0

-- Proof statement
theorem intersection_solution (m : ℝ) (n : ℝ) :
  (system1 3 n) ∧ (system2 3 n m) ∧ (line1 3 = n) ∧ (line2 3 m = n) →
  (3, n) = (3, 1) :=
  by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_intersection_solution_l1138_113823


namespace NUMINAMATH_GPT_license_plate_increase_l1138_113845

def old_license_plates : ℕ := 26 * (10^5)

def new_license_plates : ℕ := 26^2 * (10^4)

theorem license_plate_increase :
  (new_license_plates / old_license_plates : ℝ) = 2.6 := by
  sorry

end NUMINAMATH_GPT_license_plate_increase_l1138_113845


namespace NUMINAMATH_GPT_desired_antifreeze_pct_in_colder_climates_l1138_113898

-- Definitions for initial conditions
def initial_antifreeze_pct : ℝ := 0.10
def radiator_volume : ℝ := 4
def drained_volume : ℝ := 2.2857
def replacement_antifreeze_pct : ℝ := 0.80

-- Proof goal: Desired percentage of antifreeze in the mixture is 50%
theorem desired_antifreeze_pct_in_colder_climates :
  (drained_volume * replacement_antifreeze_pct + (radiator_volume - drained_volume) * initial_antifreeze_pct) / radiator_volume = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_desired_antifreeze_pct_in_colder_climates_l1138_113898


namespace NUMINAMATH_GPT_alpha_beta_diff_l1138_113880

theorem alpha_beta_diff 
  (α β : ℝ)
  (h1 : α + β = 17)
  (h2 : α * β = 70) : |α - β| = 3 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_diff_l1138_113880


namespace NUMINAMATH_GPT_four_cells_different_colors_l1138_113865

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end NUMINAMATH_GPT_four_cells_different_colors_l1138_113865


namespace NUMINAMATH_GPT_arrangement_count_is_43200_l1138_113816

noncomputable def number_of_arrangements : Nat :=
  let number_of_boys := 6
  let number_of_girls := 3
  let boys_arrangements := Nat.factorial number_of_boys
  let spaces := number_of_boys - 1
  let girls_arrangements := Nat.factorial (spaces) / Nat.factorial (spaces - number_of_girls)
  boys_arrangements * girls_arrangements

theorem arrangement_count_is_43200 :
  number_of_arrangements = 43200 := by
  sorry

end NUMINAMATH_GPT_arrangement_count_is_43200_l1138_113816


namespace NUMINAMATH_GPT_minimum_value_Q_l1138_113885

theorem minimum_value_Q (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 47 := 
  sorry

end NUMINAMATH_GPT_minimum_value_Q_l1138_113885


namespace NUMINAMATH_GPT_c_is_perfect_square_or_not_even_c_cannot_be_even_l1138_113883

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem c_is_perfect_square_or_not_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_odd : c % 2 = 1) : is_perfect_square c :=
sorry

theorem c_cannot_be_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_even : c % 2 = 0) : false :=
sorry

end NUMINAMATH_GPT_c_is_perfect_square_or_not_even_c_cannot_be_even_l1138_113883


namespace NUMINAMATH_GPT_actual_travel_time_l1138_113821

noncomputable def distance : ℕ := 360
noncomputable def scheduled_time : ℕ := 9
noncomputable def speed_increase : ℕ := 5

theorem actual_travel_time (d : ℕ) (t_sched : ℕ) (Δv : ℕ) : 
  (d = distance) ∧ (t_sched = scheduled_time) ∧ (Δv = speed_increase) → 
  t_sched + Δv = 8 :=
by
  sorry

end NUMINAMATH_GPT_actual_travel_time_l1138_113821


namespace NUMINAMATH_GPT_max_value_of_function_l1138_113869

noncomputable def function_y (x : ℝ) : ℝ := x + Real.sin x

theorem max_value_of_function : 
  ∀ (a b : ℝ), a = 0 → b = Real.pi → 
  (∀ x : ℝ, x ∈ Set.Icc a b → x + Real.sin x ≤ Real.pi) :=
by
  intros a b ha hb x hx
  sorry

end NUMINAMATH_GPT_max_value_of_function_l1138_113869


namespace NUMINAMATH_GPT_general_term_sequence_l1138_113868

/--
Given the sequence a : ℕ → ℝ such that a 0 = 1/2,
a 1 = 1/4,
a 2 = -1/8,
a 3 = 1/16,
and we observe that
a n = (-(1/2))^n,
prove that this formula holds for all n : ℕ.
-/
theorem general_term_sequence (a : ℕ → ℝ) :
  (∀ n, a n = (-(1/2))^n) :=
sorry

end NUMINAMATH_GPT_general_term_sequence_l1138_113868


namespace NUMINAMATH_GPT_edges_parallel_to_axes_l1138_113870

theorem edges_parallel_to_axes (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ)
  (hx : x1 = 0 ∨ y1 = 0 ∨ z1 = 0)
  (hy : x2 = x1 + 1 ∨ y2 = y1 + 1 ∨ z2 = z1 + 1)
  (hz : x3 = x1 + 1 ∨ y3 = y1 + 1 ∨ z3 = z1 + 1)
  (hv : x4*y4*z4 = 2011) :
  (x2-x1 ∣ 2011) ∧ (y2-y1 ∣ 2011) ∧ (z2-z1 ∣ 2011) := 
sorry

end NUMINAMATH_GPT_edges_parallel_to_axes_l1138_113870


namespace NUMINAMATH_GPT_statement1_statement2_statement3_statement4_correctness_A_l1138_113818

variables {a b : Line} {α β γ : Plane}

def perpendicular (a : Line) (α : Plane) : Prop := sorry
def parallel (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- Statement ①: If a ⊥ α and b ⊥ α, then a ∥ b
theorem statement1 (h1 : perpendicular a α) (h2 : perpendicular b α) : parallel a b := sorry

-- Statement ②: If a ⊥ α, b ⊥ β, and a ∥ b, then α ∥ β
theorem statement2 (h1 : perpendicular a α) (h2 : perpendicular b β) (h3 : parallel a b) : parallel_planes α β := sorry

-- Statement ③: If γ ⊥ α and γ ⊥ β, then α ∥ β
theorem statement3 (h1 : perpendicular γ α) (h2 : perpendicular γ β) : parallel_planes α β := sorry

-- Statement ④: If a ⊥ α and α ⊥ β, then a ∥ β
theorem statement4 (h1 : perpendicular a α) (h2 : parallel_planes α β) : parallel a b := sorry

-- The correct choice is A: Statements ① and ② are correct
theorem correctness_A : statement1_correct ∧ statement2_correct := sorry

end NUMINAMATH_GPT_statement1_statement2_statement3_statement4_correctness_A_l1138_113818


namespace NUMINAMATH_GPT_final_price_l1138_113832

def initial_price : ℝ := 200
def discount_morning : ℝ := 0.40
def increase_noon : ℝ := 0.25
def discount_afternoon : ℝ := 0.20

theorem final_price : 
  let price_after_morning := initial_price * (1 - discount_morning)
  let price_after_noon := price_after_morning * (1 + increase_noon)
  let final_price := price_after_noon * (1 - discount_afternoon)
  final_price = 120 := 
by
  sorry

end NUMINAMATH_GPT_final_price_l1138_113832


namespace NUMINAMATH_GPT_magnitude_of_angle_A_range_of_b_plus_c_l1138_113852

--- Definitions for the conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition a / (sqrt 3 * cos A) = c / sin C
axiom condition1 : a / (Real.sqrt 3 * Real.cos A) = c / Real.sin C

-- Given a = 6
axiom condition2 : a = 6

-- Conditions for sides b and c being positive
axiom condition3 : b > 0
axiom condition4 : c > 0
-- Condition for triangle inequality
axiom condition5 : b + c > a

-- Part (I) Find the magnitude of angle A
theorem magnitude_of_angle_A : A = Real.pi / 3 :=
by
  sorry

-- Part (II) Determine the range of values for b + c given a = 6
theorem range_of_b_plus_c : 6 < b + c ∧ b + c ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_angle_A_range_of_b_plus_c_l1138_113852


namespace NUMINAMATH_GPT_sum_of_remainders_l1138_113887

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1138_113887


namespace NUMINAMATH_GPT_missing_digit_is_4_l1138_113877

theorem missing_digit_is_4 (x : ℕ) (hx : 7385 = 7380 + x + 5)
  (hdiv : (7 + 3 + 8 + x + 5) % 9 = 0) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_missing_digit_is_4_l1138_113877


namespace NUMINAMATH_GPT_min_trips_calculation_l1138_113881

noncomputable def min_trips (total_weight : ℝ) (truck_capacity : ℝ) : ℕ :=
  ⌈total_weight / truck_capacity⌉₊

theorem min_trips_calculation : min_trips 18.5 3.9 = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_min_trips_calculation_l1138_113881


namespace NUMINAMATH_GPT_log_expression_value_l1138_113859

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_expression_value :
  log_base 3 32 * log_base 4 9 - log_base 2 (3/4) + log_base 2 6 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_log_expression_value_l1138_113859


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1138_113890

/-- The quadratic equation x^2 + 2x - 3 = 0 has two distinct real roots. -/
theorem quadratic_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ ^ 2 + 2 * x₁ - 3 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 3 = 0) := by
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1138_113890


namespace NUMINAMATH_GPT_sought_line_eq_l1138_113872

-- Definitions used in the conditions
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def line_perpendicular (x y : ℝ) : Prop := x + y = 0
def center_of_circle : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem sought_line_eq (x y : ℝ) :
  (circle_eq x y ∧ line_perpendicular x y ∧ (x, y) = center_of_circle) →
  (x + y + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sought_line_eq_l1138_113872


namespace NUMINAMATH_GPT_original_price_of_RAM_l1138_113867

variables (P : ℝ)

-- Conditions extracted from the problem statement
def priceAfterFire (P : ℝ) : ℝ := 1.30 * P
def priceAfterDecrease (P : ℝ) : ℝ := 1.04 * P

-- The given current price
axiom current_price : priceAfterDecrease P = 52

-- Theorem to prove the original price P
theorem original_price_of_RAM : P = 50 :=
sorry

end NUMINAMATH_GPT_original_price_of_RAM_l1138_113867


namespace NUMINAMATH_GPT_sector_central_angle_l1138_113840

theorem sector_central_angle 
  (R : ℝ) (P : ℝ) (θ : ℝ) (π : ℝ) (L : ℝ)
  (h1 : P = 83) 
  (h2 : R = 14)
  (h3 : P = 2 * R + L)
  (h4 : L = θ * R)
  (degree_conversion : θ * (180 / π) = 225) : 
  θ * (180 / π) = 225 :=
by sorry

end NUMINAMATH_GPT_sector_central_angle_l1138_113840


namespace NUMINAMATH_GPT_solve_for_x_l1138_113876

-- Define the operation *
def op (a b : ℝ) : ℝ := 2 * a - b

-- The theorem statement
theorem solve_for_x :
  (∃ x : ℝ, op x (op 1 3) = 2) ∧ (∀ x, op x -1 = 2)
  → x = 1/2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1138_113876


namespace NUMINAMATH_GPT_tan_alpha_sin_double_angle_l1138_113875

theorem tan_alpha_sin_double_angle (α : ℝ) (h : Real.tan α = 3/4) : Real.sin (2 * α) = 24/25 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_sin_double_angle_l1138_113875


namespace NUMINAMATH_GPT_hyperbola_sufficient_asymptotes_l1138_113815

open Real

def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptotes_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

theorem hyperbola_sufficient_asymptotes (a b x y : ℝ) :
  (hyperbola_eq a b x y) → (asymptotes_eq a b x y) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_sufficient_asymptotes_l1138_113815


namespace NUMINAMATH_GPT_sum_of_fractions_l1138_113888

theorem sum_of_fractions : 
  (7 / 8 + 3 / 4) = (13 / 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1138_113888


namespace NUMINAMATH_GPT_probability_same_suit_JQKA_l1138_113894

theorem probability_same_suit_JQKA  : 
  let deck_size := 52 
  let prob_J := 4 / deck_size
  let prob_Q_given_J := 1 / (deck_size - 1) 
  let prob_K_given_JQ := 1 / (deck_size - 2)
  let prob_A_given_JQK := 1 / (deck_size - 3)
  prob_J * prob_Q_given_J * prob_K_given_JQ * prob_A_given_JQK = 1 / 1624350 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_suit_JQKA_l1138_113894


namespace NUMINAMATH_GPT_sin_neg_three_pi_over_four_l1138_113892

theorem sin_neg_three_pi_over_four : Real.sin (-3 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_neg_three_pi_over_four_l1138_113892


namespace NUMINAMATH_GPT_Mrs_Fredricksons_chickens_l1138_113827

theorem Mrs_Fredricksons_chickens (C : ℕ) (h1 : 1/4 * C + 1/4 * (3/4 * C) = 35) : C = 80 :=
by
  sorry

end NUMINAMATH_GPT_Mrs_Fredricksons_chickens_l1138_113827


namespace NUMINAMATH_GPT_problem1_problem2_l1138_113817

-- Proof problem (1)
theorem problem1 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 1 < x ∧ x < 2} ∧ m = 1 →
  (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := 
by 
  sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 2 * m - 1 < x ∧ x < m + 1} →
  (B ⊆ A ↔ (m ≥ 2 ∨ (-1 ≤ m ∧ m < 2))) := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1138_113817


namespace NUMINAMATH_GPT_sum_of_integers_ending_in_2_between_100_and_500_l1138_113882

theorem sum_of_integers_ending_in_2_between_100_and_500 :
  let s : List ℤ := List.range' 102 400 10
  let sum_of_s := s.sum
  sum_of_s = 11880 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_ending_in_2_between_100_and_500_l1138_113882


namespace NUMINAMATH_GPT_initial_distance_between_fred_and_sam_l1138_113895

-- Define the conditions as parameters
variables (initial_distance : ℝ)
          (fred_speed sam_speed meeting_distance : ℝ)
          (h_fred_speed : fred_speed = 5)
          (h_sam_speed : sam_speed = 5)
          (h_meeting_distance : meeting_distance = 25)

-- State the theorem
theorem initial_distance_between_fred_and_sam :
  initial_distance = meeting_distance + meeting_distance :=
by
  -- Inline proof structure (sorry means the proof is omitted here)
  sorry

end NUMINAMATH_GPT_initial_distance_between_fred_and_sam_l1138_113895


namespace NUMINAMATH_GPT_karl_total_income_correct_l1138_113835

noncomputable def price_of_tshirt : ℝ := 5
noncomputable def price_of_pants : ℝ := 4
noncomputable def price_of_skirt : ℝ := 6
noncomputable def price_of_refurbished_tshirt : ℝ := price_of_tshirt / 2

noncomputable def discount_for_skirts (n : ℕ) : ℝ := (n / 2) * 2 * price_of_skirt * 0.10
noncomputable def discount_for_tshirts (n : ℕ) : ℝ := (n / 5) * 5 * price_of_tshirt * 0.20
noncomputable def discount_for_pants (n : ℕ) : ℝ := 0 -- accounted for in quantity

noncomputable def sales_tax (amount : ℝ) : ℝ := amount * 0.08

noncomputable def total_income : ℝ := 
  let tshirt_income := 8 * price_of_tshirt + 7 * price_of_refurbished_tshirt - discount_for_tshirts 15
  let pants_income := 6 * price_of_pants - discount_for_pants 6
  let skirts_income := 12 * price_of_skirt - discount_for_skirts 12
  let income_before_tax := tshirt_income + pants_income + skirts_income
  income_before_tax + sales_tax income_before_tax

theorem karl_total_income_correct : total_income = 141.80 :=
by
  sorry

end NUMINAMATH_GPT_karl_total_income_correct_l1138_113835


namespace NUMINAMATH_GPT_ratio_naomi_to_katherine_l1138_113896

theorem ratio_naomi_to_katherine 
  (katherine_time : ℕ) 
  (naomi_total_time : ℕ) 
  (websites_naomi : ℕ)
  (hk : katherine_time = 20)
  (hn : naomi_total_time = 750)
  (wn : websites_naomi = 30) : 
  naomi_total_time / websites_naomi / katherine_time = 5 / 4 := 
by sorry

end NUMINAMATH_GPT_ratio_naomi_to_katherine_l1138_113896


namespace NUMINAMATH_GPT_prob_less_than_9_l1138_113810

def prob_10 : ℝ := 0.24
def prob_9 : ℝ := 0.28
def prob_8 : ℝ := 0.19

theorem prob_less_than_9 : prob_10 + prob_9 + prob_8 < 1 → 1 - prob_10 - prob_9 = 0.48 := 
by {
  sorry
}

end NUMINAMATH_GPT_prob_less_than_9_l1138_113810


namespace NUMINAMATH_GPT_radius_area_tripled_l1138_113848

theorem radius_area_tripled (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = (n * (Real.sqrt 3 - 1)) / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_radius_area_tripled_l1138_113848


namespace NUMINAMATH_GPT_present_age_of_younger_l1138_113834

-- Definition based on conditions
variable (y e : ℕ)
variable (h1 : e = y + 20)
variable (h2 : e - 8 = 5 * (y - 8))

-- Statement to be proven
theorem present_age_of_younger (y e: ℕ) (h1: e = y + 20) (h2: e - 8 = 5 * (y - 8)) : y = 13 := 
by 
  sorry

end NUMINAMATH_GPT_present_age_of_younger_l1138_113834


namespace NUMINAMATH_GPT_range_of_a_product_greater_than_one_l1138_113879

namespace ProofProblem

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + x^2 - a * x + 2

variables {x1 x2 a : ℝ}

-- Conditions
axiom f_has_two_distinct_zeros : f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2

-- Goal 1: Prove the range of a
theorem range_of_a : a ∈ Set.Ioi 3 := sorry  -- Formal expression for (3, +∞) in Lean

-- Goal 2: Prove x1 * x2 > 1 given that a is in the correct range
theorem product_greater_than_one (ha : a ∈ Set.Ioi 3) : x1 * x2 > 1 := sorry

end ProofProblem

end NUMINAMATH_GPT_range_of_a_product_greater_than_one_l1138_113879


namespace NUMINAMATH_GPT_find_x_y_sum_l1138_113808

theorem find_x_y_sum :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (∃ (a b : ℕ), 360 * x = a^2 ∧ 360 * y = b^4) ∧ x + y = 2260 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_y_sum_l1138_113808


namespace NUMINAMATH_GPT_range_of_b_min_value_a_add_b_min_value_ab_l1138_113836

theorem range_of_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : b > 1 := sorry

theorem min_value_a_add_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a + b ≥ 8 := sorry

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a * b ≥ 16 := sorry

end NUMINAMATH_GPT_range_of_b_min_value_a_add_b_min_value_ab_l1138_113836


namespace NUMINAMATH_GPT_veridux_male_associates_l1138_113822

theorem veridux_male_associates (total_employees female_employees total_managers female_managers : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : female_managers = 40) :
  total_employees - female_employees = 160 :=
by
  sorry

end NUMINAMATH_GPT_veridux_male_associates_l1138_113822


namespace NUMINAMATH_GPT_subtraction_of_decimals_l1138_113806

theorem subtraction_of_decimals : 7.42 - 2.09 = 5.33 := 
by
  sorry

end NUMINAMATH_GPT_subtraction_of_decimals_l1138_113806


namespace NUMINAMATH_GPT_holes_in_compartment_l1138_113837

theorem holes_in_compartment :
  ∀ (rect : Type) (holes : ℕ) (compartments : ℕ),
  compartments = 9 →
  holes = 20 →
  (∃ (compartment : rect ) (n : ℕ), n ≥ 3) :=
by
  intros rect holes compartments h_compartments h_holes
  sorry

end NUMINAMATH_GPT_holes_in_compartment_l1138_113837


namespace NUMINAMATH_GPT_tank_fraction_l1138_113828

theorem tank_fraction (x : ℚ) : 
  let tank1_capacity := 7000
  let tank2_capacity := 5000
  let tank3_capacity := 3000
  let tank2_fraction := 4 / 5
  let tank3_fraction := 1 / 2
  let total_water := 10850
  tank1_capacity * x + tank2_capacity * tank2_fraction + tank3_capacity * tank3_fraction = total_water → 
  x = 107 / 140 := 
by {
  sorry
}

end NUMINAMATH_GPT_tank_fraction_l1138_113828


namespace NUMINAMATH_GPT_gcd_4536_13440_216_l1138_113886

def gcd_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_4536_13440_216 : gcd_of_three_numbers 4536 13440 216 = 216 :=
by
  sorry

end NUMINAMATH_GPT_gcd_4536_13440_216_l1138_113886


namespace NUMINAMATH_GPT_rectangle_side_length_l1138_113851

theorem rectangle_side_length (x : ℝ) (h1 : 0 < x) (h2 : 2 * (x + 6) = 40) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_length_l1138_113851


namespace NUMINAMATH_GPT_female_cows_percentage_l1138_113874

theorem female_cows_percentage (TotalCows PregnantFemaleCows : Nat) (PregnantPercentage : ℚ)
    (h1 : TotalCows = 44)
    (h2 : PregnantFemaleCows = 11)
    (h3 : PregnantPercentage = 0.50) :
    (PregnantFemaleCows / PregnantPercentage / TotalCows) * 100 = 50 := 
sorry

end NUMINAMATH_GPT_female_cows_percentage_l1138_113874
