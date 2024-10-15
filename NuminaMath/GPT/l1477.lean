import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_l1477_147754

theorem arithmetic_sequence_first_term :
  ∃ a₁ a₂ d : ℤ, a₂ = -5 ∧ d = 3 ∧ a₂ = a₁ + d ∧ a₁ = -8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_l1477_147754


namespace NUMINAMATH_GPT_square_of_third_side_l1477_147730

theorem square_of_third_side (a b : ℕ) (h1 : a = 4) (h2 : b = 5) 
    (h_right_triangle : (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2)) : 
    (c = 9) ∨ (c = 41) :=
sorry

end NUMINAMATH_GPT_square_of_third_side_l1477_147730


namespace NUMINAMATH_GPT_intersection_M_N_l1477_147733

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1477_147733


namespace NUMINAMATH_GPT_solve_for_x_l1477_147718

theorem solve_for_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1477_147718


namespace NUMINAMATH_GPT_find_remainder_l1477_147716

theorem find_remainder :
  ∀ (D d q r : ℕ), 
    D = 18972 → 
    d = 526 → 
    q = 36 → 
    D = d * q + r → 
    r = 36 :=
by 
  intros D d q r hD hd hq hEq
  sorry

end NUMINAMATH_GPT_find_remainder_l1477_147716


namespace NUMINAMATH_GPT_sin_double_angle_of_tan_pi_sub_alpha_eq_two_l1477_147751

theorem sin_double_angle_of_tan_pi_sub_alpha_eq_two 
  (α : Real) 
  (h : Real.tan (Real.pi - α) = 2) : 
  Real.sin (2 * α) = -4 / 5 := 
  by sorry

end NUMINAMATH_GPT_sin_double_angle_of_tan_pi_sub_alpha_eq_two_l1477_147751


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l1477_147727

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a > 4 :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l1477_147727


namespace NUMINAMATH_GPT_price_of_lemonade_l1477_147788

def costOfIngredients : ℝ := 20
def numberOfCups : ℕ := 50
def desiredProfit : ℝ := 80

theorem price_of_lemonade (price_per_cup : ℝ) :
  (costOfIngredients + desiredProfit) / numberOfCups = price_per_cup → price_per_cup = 2 :=
by
  sorry

end NUMINAMATH_GPT_price_of_lemonade_l1477_147788


namespace NUMINAMATH_GPT_polynomial_value_at_neg2_l1477_147740

def polynomial (x : ℝ) : ℝ :=
  x^6 - 5 * x^5 + 6 * x^4 + x^2 + 0.3 * x + 2

theorem polynomial_value_at_neg2 :
  polynomial (-2) = 325.4 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_neg2_l1477_147740


namespace NUMINAMATH_GPT_maximilian_wealth_greater_than_national_wealth_l1477_147764

theorem maximilian_wealth_greater_than_national_wealth (x y z : ℝ) (h1 : 2 * x > z) (h2 : y < z) :
    x > (2 * x + y) - (x + z) :=
by
  sorry

end NUMINAMATH_GPT_maximilian_wealth_greater_than_national_wealth_l1477_147764


namespace NUMINAMATH_GPT_value_of_x_l1477_147728

-- Define variables and conditions
def consecutive (x y z : ℤ) : Prop := x = z + 2 ∧ y = z + 1

-- Main proposition
theorem value_of_x (x y z : ℤ) (h1 : consecutive x y z) (h2 : z = 2) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1477_147728


namespace NUMINAMATH_GPT_cos_8_degree_l1477_147739

theorem cos_8_degree (m : ℝ) (h : Real.sin (74 * Real.pi / 180) = m) :
  Real.cos (8 * Real.pi / 180) = Real.sqrt ((1 + m) / 2) :=
sorry

end NUMINAMATH_GPT_cos_8_degree_l1477_147739


namespace NUMINAMATH_GPT_mowing_lawn_time_l1477_147793

theorem mowing_lawn_time (pay_mow : ℝ) (rate_hour : ℝ) (time_plant : ℝ) (charge_flowers : ℝ) :
  pay_mow = 15 → rate_hour = 20 → time_plant = 2 → charge_flowers = 45 → 
  (charge_flowers + pay_mow) / rate_hour - time_plant = 1 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- This is an outline, so the actual proof steps are omitted
  sorry

end NUMINAMATH_GPT_mowing_lawn_time_l1477_147793


namespace NUMINAMATH_GPT_change_in_expression_is_correct_l1477_147724

def change_in_expression (x a : ℝ) : ℝ :=
  if increases : true then (x + a)^2 - 3 - (x^2 - 3)
  else (x - a)^2 - 3 - (x^2 - 3)

theorem change_in_expression_is_correct (x a : ℝ) :
  a > 0 → change_in_expression x a = 2 * a * x + a^2 ∨ change_in_expression x a = -(2 * a * x) + a^2 :=
by
  sorry

end NUMINAMATH_GPT_change_in_expression_is_correct_l1477_147724


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1477_147747

-- Part (a) Lean Statement
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ k : ℝ, k = 2 * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (b) Lean Statement
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ q : ℝ, q = 1 - p ∧ ∃ r : ℝ, r = 2 * p / (2 * p + (1 - p) ^ 2)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (c) Lean Statement
theorem part_c (N : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ S : ℝ, S = N * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1477_147747


namespace NUMINAMATH_GPT_area_of_gray_region_l1477_147769

def center_C : ℝ × ℝ := (4, 6)
def radius_C : ℝ := 6
def center_D : ℝ × ℝ := (14, 6)
def radius_D : ℝ := 6

theorem area_of_gray_region :
  let area_of_rectangle := (14 - 4) * 6
  let quarter_circle_area := (π * 6 ^ 2) / 4
  let area_to_subtract := 2 * quarter_circle_area
  area_of_rectangle - area_to_subtract = 60 - 18 * π := 
by {
  sorry
}

end NUMINAMATH_GPT_area_of_gray_region_l1477_147769


namespace NUMINAMATH_GPT_length_LL1_l1477_147765

theorem length_LL1 (XZ : ℝ) (XY : ℝ) (YZ : ℝ) (X1Y : ℝ) (X1Z : ℝ) (LM : ℝ) (LN : ℝ) (MN : ℝ) (L1N : ℝ) (LL1 : ℝ) : 
  XZ = 13 → XY = 5 → 
  YZ = Real.sqrt (XZ^2 - XY^2) → 
  X1Y = 60 / 17 → 
  X1Z = 84 / 17 → 
  LM = X1Z → LN = X1Y → 
  MN = Real.sqrt (LM^2 - LN^2) → 
  (∀ k, L1N = 5 * k ∧ (7 * k + 5 * k) = MN → LL1 = 5 * k) →
  LL1 = 20 / 17 :=
by sorry

end NUMINAMATH_GPT_length_LL1_l1477_147765


namespace NUMINAMATH_GPT_xiao_ming_error_step_l1477_147715

theorem xiao_ming_error_step (x : ℝ) :
  (1 / (x + 1) = (2 * x) / (3 * x + 3) - 1) → 
  3 = 2 * x - (3 * x + 3) → 
  (3 = 2 * x - 3 * x + 3) ↔ false := by
  sorry

end NUMINAMATH_GPT_xiao_ming_error_step_l1477_147715


namespace NUMINAMATH_GPT_sum_even_integers_602_to_700_l1477_147720

-- Definitions based on the conditions and the problem statement
def sum_first_50_even_integers := 2550
def n_even_602_700 := 50
def first_term_602_to_700 := 602
def last_term_602_to_700 := 700

-- Theorem statement
theorem sum_even_integers_602_to_700 : 
  sum_first_50_even_integers = 2550 → 
  n_even_602_700 = 50 →
  (n_even_602_700 / 2) * (first_term_602_to_700 + last_term_602_to_700) = 32550 :=
by
  sorry

end NUMINAMATH_GPT_sum_even_integers_602_to_700_l1477_147720


namespace NUMINAMATH_GPT_overall_profit_refrigerator_mobile_phone_l1477_147753

theorem overall_profit_refrigerator_mobile_phone
  (purchase_price_refrigerator : ℕ)
  (purchase_price_mobile_phone : ℕ)
  (loss_percentage_refrigerator : ℕ)
  (profit_percentage_mobile_phone : ℕ)
  (selling_price_refrigerator : ℕ)
  (selling_price_mobile_phone : ℕ)
  (total_cost_price : ℕ)
  (total_selling_price : ℕ)
  (overall_profit : ℕ) :
  purchase_price_refrigerator = 15000 →
  purchase_price_mobile_phone = 8000 →
  loss_percentage_refrigerator = 4 →
  profit_percentage_mobile_phone = 10 →
  selling_price_refrigerator = purchase_price_refrigerator - (purchase_price_refrigerator * loss_percentage_refrigerator / 100) →
  selling_price_mobile_phone = purchase_price_mobile_phone + (purchase_price_mobile_phone * profit_percentage_mobile_phone / 100) →
  total_cost_price = purchase_price_refrigerator + purchase_price_mobile_phone →
  total_selling_price = selling_price_refrigerator + selling_price_mobile_phone →
  overall_profit = total_selling_price - total_cost_price →
  overall_profit = 200 :=
  by sorry

end NUMINAMATH_GPT_overall_profit_refrigerator_mobile_phone_l1477_147753


namespace NUMINAMATH_GPT_undefined_values_of_expression_l1477_147777

theorem undefined_values_of_expression (a : ℝ) :
  a^2 - 9 = 0 ↔ a = -3 ∨ a = 3 := 
sorry

end NUMINAMATH_GPT_undefined_values_of_expression_l1477_147777


namespace NUMINAMATH_GPT_solution_set_system_of_inequalities_l1477_147791

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_system_of_inequalities_l1477_147791


namespace NUMINAMATH_GPT_steven_has_19_peaches_l1477_147702

-- Conditions
def jill_peaches : ℕ := 6
def steven_peaches : ℕ := jill_peaches + 13

-- Statement to prove
theorem steven_has_19_peaches : steven_peaches = 19 :=
by {
    -- Proof steps would go here
    sorry
}

end NUMINAMATH_GPT_steven_has_19_peaches_l1477_147702


namespace NUMINAMATH_GPT_part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l1477_147755

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end NUMINAMATH_GPT_part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l1477_147755


namespace NUMINAMATH_GPT_at_most_one_existence_l1477_147785

theorem at_most_one_existence
  (p : ℕ) (hp : Nat.Prime p)
  (A B : Finset (Fin p))
  (h_non_empty_A : A.Nonempty) (h_non_empty_B : B.Nonempty)
  (h_union : A ∪ B = Finset.univ) (h_disjoint : A ∩ B = ∅) :
  ∃! a : Fin p, ¬ (∃ x y : Fin p, (x ∈ A ∧ y ∈ B ∧ x + y = a) ∨ (x + y = a + p)) :=
sorry

end NUMINAMATH_GPT_at_most_one_existence_l1477_147785


namespace NUMINAMATH_GPT_largest_n_exists_l1477_147783

theorem largest_n_exists (n x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6 → 
  n ≤ 8 :=
sorry

end NUMINAMATH_GPT_largest_n_exists_l1477_147783


namespace NUMINAMATH_GPT_ordered_triples_count_l1477_147732

theorem ordered_triples_count :
  {n : ℕ // n = 4} :=
sorry

end NUMINAMATH_GPT_ordered_triples_count_l1477_147732


namespace NUMINAMATH_GPT_calculate_expr_l1477_147701

theorem calculate_expr : (2023^0 + (-1/3) = 2/3) := by
  sorry

end NUMINAMATH_GPT_calculate_expr_l1477_147701


namespace NUMINAMATH_GPT_probability_factor_120_lt_8_l1477_147797

theorem probability_factor_120_lt_8 :
  let n := 120
  let total_factors := 16
  let favorable_factors := 6
  (6 / 16 : ℚ) = 3 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_probability_factor_120_lt_8_l1477_147797


namespace NUMINAMATH_GPT_exists_100_distinct_sums_l1477_147734

theorem exists_100_distinct_sums : ∃ (a : Fin 100 → ℕ), (∀ i j k l : Fin 100, i ≠ j → k ≠ l → (i, j) ≠ (k, l) → a i + a j ≠ a k + a l) ∧ (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 25000) :=
by
  sorry

end NUMINAMATH_GPT_exists_100_distinct_sums_l1477_147734


namespace NUMINAMATH_GPT_oldest_child_age_l1477_147774

theorem oldest_child_age (x : ℕ) (h_avg : (5 + 7 + 10 + x) / 4 = 8) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_oldest_child_age_l1477_147774


namespace NUMINAMATH_GPT_valid_three_digit_card_numbers_count_l1477_147798

def card_numbers : List (ℕ × ℕ) := [(0, 1), (2, 3), (4, 5), (7, 8)]

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 -- Ensures it's three digits

def three_digit_numbers : List ℕ := 
  [201, 210, 102, 120, 301, 310, 103, 130, 401, 410, 104, 140,
   501, 510, 105, 150, 601, 610, 106, 160, 701, 710, 107, 170,
   801, 810, 108, 180, 213, 231, 312, 321, 413, 431, 512, 521,
   613, 631, 714, 741, 813, 831, 214, 241, 315, 351, 415, 451,
   514, 541, 615, 651, 716, 761, 815, 851, 217, 271, 317, 371,
   417, 471, 517, 571, 617, 671, 717, 771, 817, 871, 217, 271,
   321, 371, 421, 471, 521, 571, 621, 671, 721, 771, 821, 871]

def count_valid_three_digit_numbers : ℕ :=
  three_digit_numbers.length

theorem valid_three_digit_card_numbers_count :
    count_valid_three_digit_numbers = 168 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_valid_three_digit_card_numbers_count_l1477_147798


namespace NUMINAMATH_GPT_all_integers_equal_l1477_147707

theorem all_integers_equal (k : ℕ) (a : Fin (2 * k + 1) → ℤ)
(h : ∀ b : Fin (2 * k + 1) → ℤ,
  (∀ i : Fin (2 * k + 1), b i = (a ((i : ℕ) % (2 * k + 1)) + a ((i + 1) % (2 * k + 1))) / 2) →
  ∀ i : Fin (2 * k + 1), ↑(b i) % 2 = 0) :
∀ i j : Fin (2 * k + 1), a i = a j :=
by
  sorry

end NUMINAMATH_GPT_all_integers_equal_l1477_147707


namespace NUMINAMATH_GPT_probability_divisible_by_5_l1477_147709

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end NUMINAMATH_GPT_probability_divisible_by_5_l1477_147709


namespace NUMINAMATH_GPT_xy_yz_zx_nonzero_l1477_147719

theorem xy_yz_zx_nonzero (x y z : ℝ)
  (h1 : 1 / |x^2 + 2 * y * z| + 1 / |y^2 + 2 * z * x| > 1 / |z^2 + 2 * x * y|)
  (h2 : 1 / |y^2 + 2 * z * x| + 1 / |z^2 + 2 * x * y| > 1 / |x^2 + 2 * y * z|)
  (h3 : 1 / |z^2 + 2 * x * y| + 1 / |x^2 + 2 * y * z| > 1 / |y^2 + 2 * z * x|) :
  x * y + y * z + z * x ≠ 0 := by
  sorry

end NUMINAMATH_GPT_xy_yz_zx_nonzero_l1477_147719


namespace NUMINAMATH_GPT_decreasing_function_range_l1477_147781

theorem decreasing_function_range {a : ℝ} (h1 : ∀ x y : ℝ, x < y → (1 - 2 * a)^x > (1 - 2 * a)^y) : 
    0 < a ∧ a < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l1477_147781


namespace NUMINAMATH_GPT_binomial_coefficient_12_4_l1477_147726

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

theorem binomial_coefficient_12_4 : binomial_coefficient 12 4 = 495 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_12_4_l1477_147726


namespace NUMINAMATH_GPT_average_movers_l1477_147799

noncomputable def average_people_per_hour (total_people : ℕ) (total_hours : ℕ) : ℝ :=
  total_people / total_hours

theorem average_movers :
  average_people_per_hour 5000 168 = 29.76 :=
by
  sorry

end NUMINAMATH_GPT_average_movers_l1477_147799


namespace NUMINAMATH_GPT_closest_multiple_of_17_to_2502_is_2499_l1477_147756

def isNearestMultipleOf17 (m n : ℤ) : Prop :=
  ∃ k : ℤ, 17 * k = n ∧ abs (m - n) ≤ abs (m - 17 * (k + 1)) ∧ abs (m - n) ≤ abs (m - 17 * (k - 1))

theorem closest_multiple_of_17_to_2502_is_2499 :
  isNearestMultipleOf17 2502 2499 :=
sorry

end NUMINAMATH_GPT_closest_multiple_of_17_to_2502_is_2499_l1477_147756


namespace NUMINAMATH_GPT_product_of_integers_l1477_147749

theorem product_of_integers (x y : ℕ) (h_gcd : Nat.gcd x y = 10) (h_lcm : Nat.lcm x y = 60) : x * y = 600 := by
  sorry

end NUMINAMATH_GPT_product_of_integers_l1477_147749


namespace NUMINAMATH_GPT_find_t_l1477_147750

theorem find_t (t : ℝ) :
  let P := (t - 5, -2)
  let Q := (-3, t + 4)
  let M := ((t - 8) / 2, (t + 2) / 2)
  (dist M P) ^ 2 = t^2 / 3 →
  t = -12 + 2 * Real.sqrt 21 ∨ t = -12 - 2 * Real.sqrt 21 := sorry

end NUMINAMATH_GPT_find_t_l1477_147750


namespace NUMINAMATH_GPT_units_digit_7_pow_6_pow_5_l1477_147761

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_6_pow_5_l1477_147761


namespace NUMINAMATH_GPT_solution_of_valve_problem_l1477_147743

noncomputable def valve_filling_problem : Prop :=
  ∃ (x y z : ℝ), 
    (x + y + z = 1 / 2) ∧    -- Condition when all three valves are open
    (x + z = 1 / 3) ∧        -- Condition when valves X and Z are open
    (y + z = 1 / 4) ∧        -- Condition when valves Y and Z are open
    (1 / (x + y) = 2.4)      -- Required condition for valves X and Y

theorem solution_of_valve_problem : valve_filling_problem :=
sorry

end NUMINAMATH_GPT_solution_of_valve_problem_l1477_147743


namespace NUMINAMATH_GPT_three_digit_even_sum_12_l1477_147784

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end NUMINAMATH_GPT_three_digit_even_sum_12_l1477_147784


namespace NUMINAMATH_GPT_line_equation_cartesian_circle_equation_cartesian_l1477_147714

theorem line_equation_cartesian (t : ℝ) (x y : ℝ) : 
  (x = 3 - (Real.sqrt 2 / 2) * t ∧ y = Real.sqrt 5 + (Real.sqrt 2 / 2) * t) -> 
  y = -2 * x + 6 + Real.sqrt 5 :=
sorry

theorem circle_equation_cartesian (ρ θ x y : ℝ) : 
  (ρ = 2 * Real.sqrt 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) -> 
  x^2 = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_cartesian_circle_equation_cartesian_l1477_147714


namespace NUMINAMATH_GPT_average_weight_increase_l1477_147789

theorem average_weight_increase (A : ℝ) (hA : 8 * A + 20 = (80 : ℝ) + (8 * (A - (60 - 80) / 8))) :
  ((8 * A + 20) / 8) - A = (2.5 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l1477_147789


namespace NUMINAMATH_GPT_closest_point_to_line_l1477_147768

theorem closest_point_to_line {x y : ℝ} (h : y = 2 * x - 4) :
  ∃ (closest_x closest_y : ℝ),
    closest_x = 9 / 5 ∧ closest_y = -2 / 5 ∧ closest_y = 2 * closest_x - 4 ∧
    ∀ (x' y' : ℝ), y' = 2 * x' - 4 → (closest_x - 3)^2 + (closest_y + 1)^2 ≤ (x' - 3)^2 + (y' + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_closest_point_to_line_l1477_147768


namespace NUMINAMATH_GPT_distinct_solutions_abs_eq_l1477_147723

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 3| = |x + 5|) → x = -1 :=
by
  sorry

end NUMINAMATH_GPT_distinct_solutions_abs_eq_l1477_147723


namespace NUMINAMATH_GPT_solve_container_capacity_l1477_147767

noncomputable def container_capacity (C : ℝ) :=
  (0.75 * C - 0.35 * C = 48)

theorem solve_container_capacity : ∃ C : ℝ, container_capacity C ∧ C = 120 :=
by
  use 120
  constructor
  {
    -- Proof that 0.75 * 120 - 0.35 * 120 = 48
    sorry
  }
  -- Proof that C = 120
  sorry

end NUMINAMATH_GPT_solve_container_capacity_l1477_147767


namespace NUMINAMATH_GPT_find_x_correct_l1477_147787

theorem find_x_correct (x : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * 180 = 360 * x - 480)
  (h2 : (x + 4) + x + (x - 4) = 3 * x)
  (h3 : 100 = (360 * x - 480) / (3 * x)) : 
  x = 8 := 
sorry

end NUMINAMATH_GPT_find_x_correct_l1477_147787


namespace NUMINAMATH_GPT_CannotDetermineDraculaStatus_l1477_147729

variable (Transylvanian_is_human : Prop)
variable (Dracula_is_alive : Prop)
variable (Statement : Transylvanian_is_human → Dracula_is_alive)

theorem CannotDetermineDraculaStatus : ¬ (∃ (H : Prop), H = Dracula_is_alive) :=
by
  sorry

end NUMINAMATH_GPT_CannotDetermineDraculaStatus_l1477_147729


namespace NUMINAMATH_GPT_new_prism_volume_l1477_147744

-- Define the original volume
def original_volume : ℝ := 12

-- Define the dimensions modification factors
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 3

-- Define the volume of the new prism
def new_volume := (length_factor * width_factor * height_factor) * original_volume

-- State the theorem to prove
theorem new_prism_volume : new_volume = 144 := 
by sorry

end NUMINAMATH_GPT_new_prism_volume_l1477_147744


namespace NUMINAMATH_GPT_john_initial_bench_weight_l1477_147779

variable (B : ℕ)

theorem john_initial_bench_weight (B : ℕ) (HNewTotal : 1490 = 490 + B + 600) : B = 400 :=
by
  sorry

end NUMINAMATH_GPT_john_initial_bench_weight_l1477_147779


namespace NUMINAMATH_GPT_pencil_distribution_l1477_147717

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) : 
  total_pencils / max_students = 10 :=
by
  sorry

end NUMINAMATH_GPT_pencil_distribution_l1477_147717


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_l1477_147738

theorem arithmetic_sequence_formula :
  ∀ (a : ℕ → ℕ), (a 1 = 2) → (∀ n, a (n + 1) = a n + 2) → ∀ n, a n = 2 * n :=
by
  intro a
  intro h1
  intro hdiff
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_l1477_147738


namespace NUMINAMATH_GPT_hexagon_perimeter_l1477_147771

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (perimeter : ℕ) 
  (h1 : num_sides = 6)
  (h2 : side_length = 7)
  (h3 : perimeter = side_length * num_sides) : perimeter = 42 := by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l1477_147771


namespace NUMINAMATH_GPT_blankets_first_day_l1477_147722

-- Definition of the conditions
def num_people := 15
def blankets_day_three := 22
def total_blankets := 142

-- The problem statement
theorem blankets_first_day (B : ℕ) : 
  (num_people * B) + (3 * (num_people * B)) + blankets_day_three = total_blankets → 
  B = 2 :=
by sorry

end NUMINAMATH_GPT_blankets_first_day_l1477_147722


namespace NUMINAMATH_GPT_sum_of_squares_not_divisible_by_13_l1477_147748

theorem sum_of_squares_not_divisible_by_13
  (x y z : ℤ)
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_xz : Int.gcd x z = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_sum : (x + y + z) % 13 = 0)
  (h_prod : (x * y * z) % 13 = 0) :
  (x^2 + y^2 + z^2) % 13 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_not_divisible_by_13_l1477_147748


namespace NUMINAMATH_GPT_rhombus_area_l1477_147752

theorem rhombus_area (side d1 d2 : ℝ) (h_side : side = 25) (h_d1 : d1 = 30) (h_diag : d2 = 40) :
  (d1 * d2) / 2 = 600 :=
by
  rw [h_d1, h_diag]
  norm_num

end NUMINAMATH_GPT_rhombus_area_l1477_147752


namespace NUMINAMATH_GPT_tanker_fill_rate_l1477_147772

theorem tanker_fill_rate :
  let barrels_per_min := 2
  let liters_per_barrel := 159
  let cubic_meters_per_liter := 0.001
  let minutes_per_hour := 60
  let liters_per_min := barrels_per_min * liters_per_barrel
  let liters_per_hour := liters_per_min * minutes_per_hour
  let cubic_meters_per_hour := liters_per_hour * cubic_meters_per_liter
  cubic_meters_per_hour = 19.08 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_tanker_fill_rate_l1477_147772


namespace NUMINAMATH_GPT_value_of_x_l1477_147759

theorem value_of_x (x : ℝ) : 
  (x ≤ 0 → x^2 + 1 = 5 → x = -2) ∧ 
  (0 < x → -2 * x = 5 → false) := 
sorry

end NUMINAMATH_GPT_value_of_x_l1477_147759


namespace NUMINAMATH_GPT_tan_pi_seven_product_eq_sqrt_seven_l1477_147763

theorem tan_pi_seven_product_eq_sqrt_seven :
  (Real.tan (Real.pi / 7)) * (Real.tan (2 * Real.pi / 7)) * (Real.tan (3 * Real.pi / 7)) = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_seven_product_eq_sqrt_seven_l1477_147763


namespace NUMINAMATH_GPT_cuts_for_20_pentagons_l1477_147795

theorem cuts_for_20_pentagons (K : ℕ) : 20 * 540 + (K - 19) * 180 ≤ 360 * K + 540 ↔ K ≥ 38 :=
by
  sorry

end NUMINAMATH_GPT_cuts_for_20_pentagons_l1477_147795


namespace NUMINAMATH_GPT_cos_five_theta_l1477_147736

theorem cos_five_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (5 * θ) = (125 * Real.sqrt 15 - 749) / 1024 := 
  sorry

end NUMINAMATH_GPT_cos_five_theta_l1477_147736


namespace NUMINAMATH_GPT_mack_writing_time_tuesday_l1477_147731

variable (minutes_per_page_mon : ℕ := 30)
variable (time_mon : ℕ := 60)
variable (pages_wed : ℕ := 5)
variable (total_pages : ℕ := 10)
variable (minutes_per_page_tue : ℕ := 15)

theorem mack_writing_time_tuesday :
  (time_mon / minutes_per_page_mon) + pages_wed + (3 * minutes_per_page_tue / minutes_per_page_tue) = total_pages →
  (3 * minutes_per_page_tue) = 45 := by
  intros h
  sorry

end NUMINAMATH_GPT_mack_writing_time_tuesday_l1477_147731


namespace NUMINAMATH_GPT_butterfat_mixture_l1477_147778

theorem butterfat_mixture (x : ℝ) :
  (0.10 * x + 0.30 * 8 = 0.20 * (x + 8)) → x = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_butterfat_mixture_l1477_147778


namespace NUMINAMATH_GPT_evaluate_expression_l1477_147725

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 9) : 
  2 * x^(y / 2 : ℕ) + 5 * y^(x / 2 : ℕ) = 1429 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1477_147725


namespace NUMINAMATH_GPT_eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l1477_147776

noncomputable def relative_speed_moon_sun := (17/16 : ℝ) - (1/12 : ℝ)
noncomputable def initial_distance := (47/10 : ℝ)
noncomputable def time_coincide := initial_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_time_coincide : 
  (time_coincide - 12 : ℝ) = (2 + 1/60 : ℝ) :=
sorry

noncomputable def start_distance := (37/10 : ℝ)
noncomputable def time_start := start_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_start_time : 
  (time_start - 12 : ℝ) = (1 + 59/60 : ℝ) :=
sorry

noncomputable def end_distance := (57/10 : ℝ)
noncomputable def time_end := end_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_end_time : 
  (time_end - 12 : ℝ) = (3 + 2/60 : ℝ) :=
sorry

end NUMINAMATH_GPT_eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l1477_147776


namespace NUMINAMATH_GPT_age_of_B_is_23_l1477_147735

-- Definitions of conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 87)
variable (h2 : A + C = 64)

-- Statement of the problem
theorem age_of_B_is_23 : B = 23 :=
by { sorry }

end NUMINAMATH_GPT_age_of_B_is_23_l1477_147735


namespace NUMINAMATH_GPT_function_d_has_no_boundary_point_l1477_147708

def is_boundary_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∃ x₁ < x₀, f x₁ = 0) ∧ (∃ x₂ > x₀, f x₂ = 0)

def f_a (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x - 2
def f_b (x : ℝ) : ℝ := abs (x^2 - 3)
def f_c (x : ℝ) : ℝ := 1 - abs (x - 2)
def f_d (x : ℝ) : ℝ := x^3 + x

theorem function_d_has_no_boundary_point :
  ¬ ∃ x₀ : ℝ, is_boundary_point f_d x₀ :=
sorry

end NUMINAMATH_GPT_function_d_has_no_boundary_point_l1477_147708


namespace NUMINAMATH_GPT_max_value_g_l1477_147704

def g : ℕ → ℤ
| n => if n < 5 then n + 10 else g (n - 3)

theorem max_value_g : ∃ x, (∀ n : ℕ, g n ≤ x) ∧ (∃ y, g y = x) ∧ x = 14 := 
by
  sorry

end NUMINAMATH_GPT_max_value_g_l1477_147704


namespace NUMINAMATH_GPT_minimum_value_of_f_range_of_x_l1477_147703

noncomputable def f (x : ℝ) := |2*x + 1| + |2*x - 1|

-- Problem 1
theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 2 :=
by
  intro x
  sorry

-- Problem 2
theorem range_of_x (a b : ℝ) (h : |2*a + b| + |a| - (1/2) * |a + b| * f x ≥ 0) : 
  - (1/2) ≤ x ∧ x ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_range_of_x_l1477_147703


namespace NUMINAMATH_GPT_expansion_of_product_l1477_147700

theorem expansion_of_product (x : ℝ) :
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := 
by
  sorry

end NUMINAMATH_GPT_expansion_of_product_l1477_147700


namespace NUMINAMATH_GPT_tony_average_time_to_store_l1477_147705

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_tony_average_time_to_store_l1477_147705


namespace NUMINAMATH_GPT_find_numerical_value_l1477_147780

-- Define the conditions
variables {x y z : ℝ}
axiom h1 : 3 * x - 4 * y - 2 * z = 0
axiom h2 : x + 4 * y - 20 * z = 0
axiom h3 : z ≠ 0

-- State the goal
theorem find_numerical_value : (x^2 + 4 * x * y) / (y^2 + z^2) = 2.933 :=
by
  sorry

end NUMINAMATH_GPT_find_numerical_value_l1477_147780


namespace NUMINAMATH_GPT_area_of_right_triangle_l1477_147706

-- Define the conditions
def hypotenuse : ℝ := 9
def angle : ℝ := 30

-- Define the Lean statement for the proof problem
theorem area_of_right_triangle : 
  ∃ (area : ℝ), area = 10.125 * Real.sqrt 3 ∧
  ∃ (shorter_leg : ℝ) (longer_leg : ℝ),
    shorter_leg = hypotenuse / 2 ∧
    longer_leg = shorter_leg * Real.sqrt 3 ∧
    area = (shorter_leg * longer_leg) / 2 :=
by {
  -- The proof would go here, but we only need to state the problem for this task.
  sorry
}

end NUMINAMATH_GPT_area_of_right_triangle_l1477_147706


namespace NUMINAMATH_GPT_solve_for_x_l1477_147711

variable (x y z a b w : ℝ)
variable (angle_DEB : ℝ)

def angle_sum_D (x y z angle_DEB : ℝ) : Prop := x + y + z + angle_DEB = 360
def angle_sum_E (a b w angle_DEB : ℝ) : Prop := a + b + w + angle_DEB = 360

theorem solve_for_x 
  (h1 : angle_sum_D x y z angle_DEB) 
  (h2 : angle_sum_E a b w angle_DEB) : 
  x = a + b + w - y - z :=
by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_solve_for_x_l1477_147711


namespace NUMINAMATH_GPT_total_fruit_count_l1477_147760

-- Define the number of oranges
def oranges : ℕ := 6

-- Define the number of apples based on the number of oranges
def apples : ℕ := oranges - 2

-- Define the number of bananas based on the number of apples
def bananas : ℕ := 3 * apples

-- Define the number of peaches based on the number of bananas
def peaches : ℕ := bananas / 2

-- Define the total number of fruits in the basket
def total_fruits : ℕ := oranges + apples + bananas + peaches

-- Prove that the total number of pieces of fruit in the basket is 28
theorem total_fruit_count : total_fruits = 28 := by
  sorry

end NUMINAMATH_GPT_total_fruit_count_l1477_147760


namespace NUMINAMATH_GPT_xy_yz_zx_over_x2_y2_z2_l1477_147790

theorem xy_yz_zx_over_x2_y2_z2 (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h_sum : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end NUMINAMATH_GPT_xy_yz_zx_over_x2_y2_z2_l1477_147790


namespace NUMINAMATH_GPT_no_non_integer_point_exists_l1477_147710

variable (b0 b1 b2 b3 b4 b5 u v : ℝ)

def q (x y : ℝ) : ℝ := b0 + b1 * x + b2 * y + b3 * x^2 + b4 * x * y + b5 * y^2

theorem no_non_integer_point_exists
    (h₀ : q b0 b1 b2 b3 b4 b5 0 0 = 0)
    (h₁ : q b0 b1 b2 b3 b4 b5 1 0 = 0)
    (h₂ : q b0 b1 b2 b3 b4 b5 (-1) 0 = 0)
    (h₃ : q b0 b1 b2 b3 b4 b5 0 1 = 0)
    (h₄ : q b0 b1 b2 b3 b4 b5 0 (-1) = 0)
    (h₅ : q b0 b1 b2 b3 b4 b5 1 1 = 0) :
  ∀ u v : ℝ, (¬ ∃ (n m : ℤ), u = n ∧ v = m) → q b0 b1 b2 b3 b4 b5 u v ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_non_integer_point_exists_l1477_147710


namespace NUMINAMATH_GPT_a_and_b_together_complete_in_10_days_l1477_147737

noncomputable def a_works_twice_as_fast_as_b (a b : ℝ) : Prop :=
  a = 2 * b

noncomputable def b_can_complete_work_in_30_days (b : ℝ) : Prop :=
  b = 1/30

theorem a_and_b_together_complete_in_10_days (a b : ℝ) 
  (h₁ : a_works_twice_as_fast_as_b a b)
  (h₂ : b_can_complete_work_in_30_days b) : 
  (1 / (a + b)) = 10 := 
sorry

end NUMINAMATH_GPT_a_and_b_together_complete_in_10_days_l1477_147737


namespace NUMINAMATH_GPT_abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l1477_147746

theorem abs_x_minus_one_sufficient_but_not_necessary_for_quadratic (x : ℝ) :
  (|x - 1| < 2) → (x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l1477_147746


namespace NUMINAMATH_GPT_sequence_pattern_l1477_147794

theorem sequence_pattern (a b c d e f : ℕ) 
  (h1 : a + b = 12)
  (h2 : 8 + 9 = 16)
  (h3 : 5 + 6 = 10)
  (h4 : 7 + 8 = 14)
  (h5 : 3 + 3 = 5) : 
  ∀ x, ∃ y, x + y = 2 * x := by
  intros x
  use 0
  sorry

end NUMINAMATH_GPT_sequence_pattern_l1477_147794


namespace NUMINAMATH_GPT_directrix_of_parabola_l1477_147713

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1477_147713


namespace NUMINAMATH_GPT_albert_horses_l1477_147712

variable {H C : ℝ}

theorem albert_horses :
  (2000 * H + 9 * C = 13400) ∧ (200 * H + 0.20 * 9 * C = 1880) ∧ (∀ x : ℝ, x = 2000) → H = 4 := 
by
  sorry

end NUMINAMATH_GPT_albert_horses_l1477_147712


namespace NUMINAMATH_GPT_percentage_increase_from_boys_to_total_l1477_147766

def DamesSchoolBoys : ℕ := 2000
def DamesSchoolGirls : ℕ := 5000
def TotalAttendance : ℕ := DamesSchoolBoys + DamesSchoolGirls
def PercentageIncrease (initial final : ℕ) : ℚ := ((final - initial) / initial) * 100

theorem percentage_increase_from_boys_to_total :
  PercentageIncrease DamesSchoolBoys TotalAttendance = 250 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_from_boys_to_total_l1477_147766


namespace NUMINAMATH_GPT_cost_price_of_table_l1477_147792

theorem cost_price_of_table 
  (SP : ℝ) 
  (CP : ℝ) 
  (h1 : SP = 1.24 * CP) 
  (h2 : SP = 8215) :
  CP = 6625 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_table_l1477_147792


namespace NUMINAMATH_GPT_range_of_m_l1477_147775

theorem range_of_m (m x : ℝ) : 
  (2 / (x - 3) + (x + m) / (3 - x) = 2) 
  ∧ (x ≥ 0) →
  (m ≤ 8 ∧ m ≠ -1) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1477_147775


namespace NUMINAMATH_GPT_AD_mutually_exclusive_not_complementary_l1477_147745

-- Define the sets representing the outcomes of the events
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {2, 4, 6}
def D : Set ℕ := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set ℕ) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set ℕ) : Prop := X ∪ Y = {1, 2, 3, 4, 5, 6}

-- The statement to prove that events A and D are mutually exclusive but not complementary
theorem AD_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬ complementary A D :=
by
  sorry

end NUMINAMATH_GPT_AD_mutually_exclusive_not_complementary_l1477_147745


namespace NUMINAMATH_GPT_measure_of_angle_A_l1477_147742

theorem measure_of_angle_A {A B C : ℝ} (hC : C = 2 * B) (hB : B = 21) :
  A = 180 - B - C := 
by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_l1477_147742


namespace NUMINAMATH_GPT_machine_working_time_l1477_147762

theorem machine_working_time (total_shirts_made : ℕ) (shirts_per_minute : ℕ)
  (h1 : total_shirts_made = 196) (h2 : shirts_per_minute = 7) :
  (total_shirts_made / shirts_per_minute = 28) :=
by
  sorry

end NUMINAMATH_GPT_machine_working_time_l1477_147762


namespace NUMINAMATH_GPT_product_of_roots_l1477_147782

noncomputable def f : ℝ → ℝ := sorry

theorem product_of_roots :
  (∀ x : ℝ, 4 * f (3 - x) - f x = 3 * x ^ 2 - 4 * x - 3) →
  (∃ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5) :=
sorry

end NUMINAMATH_GPT_product_of_roots_l1477_147782


namespace NUMINAMATH_GPT_shaded_region_area_l1477_147786

theorem shaded_region_area (ABCD: Type) (D B: Type) (AD CD: ℝ) 
  (h1: (AD = 5)) (h2: (CD = 12)):
  let radiusD := Real.sqrt (AD^2 + CD^2)
  let quarter_circle_area := Real.pi * radiusD^2 / 4
  let radiusC := CD / 2
  let semicircle_area := Real.pi * radiusC^2 / 2
  quarter_circle_area - semicircle_area = 97 * Real.pi / 4 :=
by sorry

end NUMINAMATH_GPT_shaded_region_area_l1477_147786


namespace NUMINAMATH_GPT_unique_non_zero_in_rows_and_cols_l1477_147770

variable (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)

theorem unique_non_zero_in_rows_and_cols
  (non_neg_A : ∀ i j, 0 ≤ A i j)
  (non_sing_A : Invertible A)
  (non_neg_A_inv : ∀ i j, 0 ≤ (A⁻¹) i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end NUMINAMATH_GPT_unique_non_zero_in_rows_and_cols_l1477_147770


namespace NUMINAMATH_GPT_pieces_given_by_brother_l1477_147757

-- Given conditions
def original_pieces : ℕ := 18
def total_pieces_now : ℕ := 62

-- The statement to prove
theorem pieces_given_by_brother : total_pieces_now - original_pieces = 44 := by
  -- Starting with the given conditions
  unfold original_pieces total_pieces_now
  -- Place to insert the proof
  sorry

end NUMINAMATH_GPT_pieces_given_by_brother_l1477_147757


namespace NUMINAMATH_GPT_child_grandmother_ratio_l1477_147758

def grandmother_weight (G D C : ℝ) : Prop :=
  G + D + C = 160

def daughter_child_weight (D C : ℝ) : Prop :=
  D + C = 60

def daughter_weight (D : ℝ) : Prop :=
  D = 40

theorem child_grandmother_ratio (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_child_weight D C) (h3 : daughter_weight D) :
  C / G = 1 / 5 :=
sorry

end NUMINAMATH_GPT_child_grandmother_ratio_l1477_147758


namespace NUMINAMATH_GPT_triangle_inequality_l1477_147721

variables {α β γ a b c : ℝ}
variable {n : ℕ}

theorem triangle_inequality (h_sum_angles : α + β + γ = Real.pi) (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.pi / 3) ^ n ≤ (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) ∧ 
  (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) < (Real.pi ^ n / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1477_147721


namespace NUMINAMATH_GPT_tied_in_runs_l1477_147796

def aaron_runs : List ℕ := [4, 8, 15, 7, 4, 12, 11, 5]
def bonds_runs : List ℕ := [3, 5, 18, 9, 12, 14, 9, 0]

def total_runs (runs : List ℕ) : ℕ := runs.foldl (· + ·) 0

theorem tied_in_runs : total_runs aaron_runs = total_runs bonds_runs := by
  sorry

end NUMINAMATH_GPT_tied_in_runs_l1477_147796


namespace NUMINAMATH_GPT_total_cost_is_correct_l1477_147773

noncomputable def total_cost : ℝ :=
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let elephant_ear_cost := 7.00
  let purple_fountain_grass_cost := 6.00
  let pots := 6
  let sales_tax := 0.07
  let cost_one_pot := palm_fern_cost 
                   + 4 * creeping_jenny_cost 
                   + 4 * geranium_cost 
                   + 2 * elephant_ear_cost 
                   + 3 * purple_fountain_grass_cost
  let total_pots_cost := pots * cost_one_pot
  let tax := total_pots_cost * sales_tax
  total_pots_cost + tax

theorem total_cost_is_correct : total_cost = 494.34 :=
by
  -- This is where the proof would go, but we are adding sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1477_147773


namespace NUMINAMATH_GPT_largest_whole_number_l1477_147741

theorem largest_whole_number :
  ∃ x : ℕ, 9 * x - 8 < 130 ∧ (∀ y : ℕ, 9 * y - 8 < 130 → y ≤ x) ∧ x = 15 :=
sorry

end NUMINAMATH_GPT_largest_whole_number_l1477_147741
