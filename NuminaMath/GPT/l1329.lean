import Mathlib

namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1329_132993

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 / (x + 2) + x - 2) / ((x^2 - 2*x + 1) / (x + 2))

theorem simplify_and_evaluate_expression (x : ℝ) (hx : |x| = 2) (h_ne : x ≠ -2) :
  given_expression x = 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1329_132993


namespace NUMINAMATH_GPT_binary_difference_l1329_132969

theorem binary_difference (n : ℕ) (b_2 : List ℕ) (x y : ℕ) (h1 : n = 157)
  (h2 : b_2 = [1, 0, 0, 1, 1, 1, 0, 1])
  (hx : x = b_2.count 0)
  (hy : y = b_2.count 1) : y - x = 2 := by
  sorry

end NUMINAMATH_GPT_binary_difference_l1329_132969


namespace NUMINAMATH_GPT_a5_b5_sum_l1329_132920

-- Definitions of arithmetic sequences
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable
def a : ℕ → ℝ := sorry -- defining the arithmetic sequences
noncomputable
def b : ℕ → ℝ := sorry

-- Common differences for the sequences
noncomputable
def d_a : ℝ := sorry
noncomputable
def d_b : ℝ := sorry

-- Conditions given in the problem
axiom a1_b1_sum : a 1 + b 1 = 7
axiom a3_b3_sum : a 3 + b 3 = 21
axiom a_is_arithmetic : arithmetic_seq a d_a
axiom b_is_arithmetic : arithmetic_seq b d_b

-- Theorem to be proved
theorem a5_b5_sum : a 5 + b 5 = 35 := 
by sorry

end NUMINAMATH_GPT_a5_b5_sum_l1329_132920


namespace NUMINAMATH_GPT_factor_value_l1329_132925

theorem factor_value 
  (m : ℝ) 
  (h : ∀ x : ℝ, x + 5 = 0 → (x^2 - m * x - 40) = 0) : 
  m = 3 := 
sorry

end NUMINAMATH_GPT_factor_value_l1329_132925


namespace NUMINAMATH_GPT_min_value_l1329_132901

noncomputable def min_res (a b c : ℝ) : ℝ := 
  if h : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
  then (1 / a + 2 / b + 3 / c) 
  else 0

theorem min_value (a b c : ℝ) : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
    → min_res a b c = 6 := 
sorry

end NUMINAMATH_GPT_min_value_l1329_132901


namespace NUMINAMATH_GPT_find_b_for_inf_solutions_l1329_132904

theorem find_b_for_inf_solutions (x : ℝ) (b : ℝ) : 5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_for_inf_solutions_l1329_132904


namespace NUMINAMATH_GPT_probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l1329_132966

def probability_of_excellence_A : ℚ := 2/5
def probability_of_excellence_B1 : ℚ := 1/4
def probability_of_excellence_B2 : ℚ := 2/5
def probability_of_excellence_B3 (n : ℚ) : ℚ := n

def one_excellence_A : ℚ := 3 * (2/5) * (3/5)^2
def one_excellence_B (n : ℚ) : ℚ := 
    (probability_of_excellence_B1 * (3/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (2/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (3/5) * n)

theorem probability_one_excellence_A : one_excellence_A = 54/125 := sorry

theorem probability_one_excellence_B (n : ℚ) (hn : n = 1/3) : one_excellence_B n = 9/20 := sorry

def expected_excellence_A : ℚ := 3 * (2/5)
def expected_excellence_B (n : ℚ) : ℚ := (13/20) + n

theorem range_n_for_A (n : ℚ) (hn1 : 0 < n) (hn2 : n < 11/20): 
    expected_excellence_A > expected_excellence_B n := sorry

end NUMINAMATH_GPT_probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l1329_132966


namespace NUMINAMATH_GPT_water_remainder_l1329_132956

theorem water_remainder (n : ℕ) (f : ℕ → ℚ) (h_init : f 1 = 1) 
  (h_recursive : ∀ k, k ≥ 2 → f k = f (k - 1) * (k^2 - 1) / k^2) :
  f 7 = 1 / 50 := 
sorry

end NUMINAMATH_GPT_water_remainder_l1329_132956


namespace NUMINAMATH_GPT_calculate_product_l1329_132939

theorem calculate_product (a : ℝ) : 2 * a * (3 * a) = 6 * a^2 := by
  -- This will skip the proof, denoted by 'sorry'
  sorry

end NUMINAMATH_GPT_calculate_product_l1329_132939


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_intercepts_l1329_132981

theorem area_of_triangle_formed_by_intercepts :
  let f (x : ℝ) := (x - 4)^2 * (x + 3)
  let x_intercepts := [-3, 4]
  let y_intercept := 48
  let base := 7
  let height := 48
  let area := (1 / 2 : ℝ) * base * height
  area = 168 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_formed_by_intercepts_l1329_132981


namespace NUMINAMATH_GPT_total_recovery_time_l1329_132934

theorem total_recovery_time 
  (lions: ℕ := 3) (rhinos: ℕ := 2) (time_per_animal: ℕ := 2) :
  (lions + rhinos) * time_per_animal = 10 := by
  sorry

end NUMINAMATH_GPT_total_recovery_time_l1329_132934


namespace NUMINAMATH_GPT_train_speed_l1329_132963

theorem train_speed (L V : ℝ) (h1 : L = V * 20) (h2 : L + 300.024 = V * 50) : V = 10.0008 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1329_132963


namespace NUMINAMATH_GPT_rectangle_side_length_l1329_132932

theorem rectangle_side_length (a b c d : ℕ) 
  (h₁ : a = 3) 
  (h₂ : b = 6) 
  (h₃ : a / c = 3 / 4) : 
  c = 4 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_side_length_l1329_132932


namespace NUMINAMATH_GPT_find_m_l1329_132972

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

theorem find_m (h : (1 + 3, m - 2) = (4, m - 2) ∧ (4 * 3 + (m - 2) * (-2) = 0)) : m = 8 := by
  sorry

end NUMINAMATH_GPT_find_m_l1329_132972


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_function_l1329_132958

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_quadratic_function 
  (f : ℝ → ℝ)
  (n : ℕ)
  (h1 : f n = 6)
  (h2 : f (n + 1) = 5)
  (h3 : f (n + 2) = 5)
  (hf : ∃ a b c : ℝ, f = quadratic_function a b c) :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_function_l1329_132958


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1329_132987

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) (S5 : S 5 = 30) (S10 : S 10 = 110) : S 15 = 240 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1329_132987


namespace NUMINAMATH_GPT_smallest_k_correct_l1329_132994

noncomputable def smallest_k (n m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) : ℕ :=
    6

theorem smallest_k_correct (n : ℕ) (m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) :
  64 ^ smallest_k n m hn hm + 32 ^ m > 4 ^ (16 + n) :=
sorry

end NUMINAMATH_GPT_smallest_k_correct_l1329_132994


namespace NUMINAMATH_GPT_part_1_part_2_l1329_132978

-- Part (Ⅰ)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 + (m - 2) * x + 1

theorem part_1 (m : ℝ) : (∀ x : ℝ, ¬ f x m < 0) ↔ (-2 ≤ m ∧ m ≤ 6) :=
by sorry

-- Part (Ⅱ)
theorem part_2 (m : ℝ) (h_even : ∀ ⦃x : ℝ⦄, f x m = f (-x) m) :
  (m = 2) → 
  ((∀ x : ℝ, x ≤ 0 → f x 2 ≥ f 0 2) ∧ (∀ x : ℝ, x ≥ 0 → f x 2 ≥ f 0 2)) :=
by sorry

end NUMINAMATH_GPT_part_1_part_2_l1329_132978


namespace NUMINAMATH_GPT_find_b_value_l1329_132927

theorem find_b_value (b : ℚ) (x : ℚ) (h1 : 3 * x + 9 = 0) (h2 : b * x + 15 = 5) : b = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l1329_132927


namespace NUMINAMATH_GPT_num_double_yolk_eggs_l1329_132973

noncomputable def double_yolk_eggs (total_eggs total_yolks : ℕ) (double_yolk_contrib : ℕ) : ℕ :=
(total_yolks - total_eggs + double_yolk_contrib) / double_yolk_contrib

theorem num_double_yolk_eggs (total_eggs total_yolks double_yolk_contrib expected : ℕ)
    (h1 : total_eggs = 12)
    (h2 : total_yolks = 17)
    (h3 : double_yolk_contrib = 2)
    (h4 : expected = 5) :
  double_yolk_eggs total_eggs total_yolks double_yolk_contrib = expected :=
by
  rw [h1, h2, h3, h4]
  dsimp [double_yolk_eggs]
  norm_num
  sorry

end NUMINAMATH_GPT_num_double_yolk_eggs_l1329_132973


namespace NUMINAMATH_GPT_possible_values_of_N_l1329_132914

theorem possible_values_of_N (N : ℤ) (h : N^2 - N = 12) : N = 4 ∨ N = -3 :=
sorry

end NUMINAMATH_GPT_possible_values_of_N_l1329_132914


namespace NUMINAMATH_GPT_rectangle_area_l1329_132933

theorem rectangle_area (w d : ℝ) 
  (h1 : d = (w^2 + (3 * w)^2) ^ (1/2))
  (h2 : ∃ A : ℝ, A = w * 3 * w) :
  ∃ A : ℝ, A = 3 * (d^2 / 10) := 
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_area_l1329_132933


namespace NUMINAMATH_GPT_total_pennies_l1329_132931

theorem total_pennies (rachelle gretchen rocky max taylor : ℕ) (h_r : rachelle = 720) (h_g : gretchen = rachelle / 2)
  (h_ro : rocky = gretchen / 3) (h_m : max = rocky * 4) (h_t : taylor = max / 5) :
  rachelle + gretchen + rocky + max + taylor = 1776 := 
by
  sorry

end NUMINAMATH_GPT_total_pennies_l1329_132931


namespace NUMINAMATH_GPT_werewolf_is_A_l1329_132948

def is_liar (x : ℕ) : Prop := sorry
def is_knight (x : ℕ) : Prop := sorry
def is_werewolf (x : ℕ) : Prop := sorry

axiom A : ℕ
axiom B : ℕ
axiom C : ℕ

-- Conditions from the problem
axiom A_statement : is_liar A ∨ is_liar B
axiom B_statement : is_werewolf C
axiom exactly_one_werewolf : 
  (is_werewolf A ∧ ¬ is_werewolf B ∧ ¬ is_werewolf C) ∨
  (is_werewolf B ∧ ¬ is_werewolf A ∧ ¬ is_werewolf C) ∨
  (is_werewolf C ∧ ¬ is_werewolf A ∧ ¬ is_werewolf B)
axiom werewolf_is_knight : ∀ x : ℕ, is_werewolf x → is_knight x

-- Prove the conclusion
theorem werewolf_is_A : 
  is_werewolf A ∧ is_knight A :=
sorry

end NUMINAMATH_GPT_werewolf_is_A_l1329_132948


namespace NUMINAMATH_GPT_marshmallow_per_smore_l1329_132991

theorem marshmallow_per_smore (graham_crackers : ℕ) (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) 
                               (graham_crackers_per_smore : ℕ) :
  graham_crackers = 48 ∧ initial_marshmallows = 6 ∧ additional_marshmallows = 18 ∧ graham_crackers_per_smore = 2 →
  (initial_marshmallows + additional_marshmallows) / (graham_crackers / graham_crackers_per_smore) = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_marshmallow_per_smore_l1329_132991


namespace NUMINAMATH_GPT_nine_digit_numbers_divisible_by_eleven_l1329_132954

theorem nine_digit_numbers_divisible_by_eleven :
  ∃ (n : ℕ), n = 31680 ∧
    ∃ (num : ℕ), num < 10^9 ∧ num ≥ 10^8 ∧
      (∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ i : ℕ, i ≤ 8 ∧ (num / 10^i) % 10 = d) ∧
      (num % 11 = 0) := 
sorry

end NUMINAMATH_GPT_nine_digit_numbers_divisible_by_eleven_l1329_132954


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1329_132946

variable {x y : ℝ}
variable (hx : x + y = 3 * x * y + 2)

theorem sum_of_reciprocals : (1 / x) + (1 / y) = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1329_132946


namespace NUMINAMATH_GPT_sin_240_deg_l1329_132985

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_240_deg_l1329_132985


namespace NUMINAMATH_GPT_sum_of_three_consecutive_integers_product_990_l1329_132942

theorem sum_of_three_consecutive_integers_product_990 
  (a b c : ℕ) 
  (h1 : b = a + 1)
  (h2 : c = b + 1)
  (h3 : a * b * c = 990) :
  a + b + c = 30 :=
sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_integers_product_990_l1329_132942


namespace NUMINAMATH_GPT_find_x_satisfying_sinx_plus_cosx_eq_one_l1329_132926

theorem find_x_satisfying_sinx_plus_cosx_eq_one :
  ∀ x, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0) := by
  sorry

end NUMINAMATH_GPT_find_x_satisfying_sinx_plus_cosx_eq_one_l1329_132926


namespace NUMINAMATH_GPT_velociraptor_catch_time_l1329_132953

/-- You encounter a velociraptor while out for a stroll. You run to the northeast at 10 m/s 
    with a 3-second head start. The velociraptor runs at 15√2 m/s but only runs either north or east at any given time. 
    Prove that the time until the velociraptor catches you is 6 seconds. -/
theorem velociraptor_catch_time (v_yours : ℝ) (t_head_start : ℝ) (v_velociraptor : ℝ)
  (v_eff : ℝ) (speed_advantage : ℝ) (headstart_distance : ℝ) :
  v_yours = 10 → t_head_start = 3 → v_velociraptor = 15 * Real.sqrt 2 →
  v_eff = 15 → speed_advantage = v_eff - v_yours → headstart_distance = v_yours * t_head_start →
  (headstart_distance / speed_advantage) = 6 :=
by
  sorry

end NUMINAMATH_GPT_velociraptor_catch_time_l1329_132953


namespace NUMINAMATH_GPT_total_present_ages_l1329_132918

variable (P Q P' Q' : ℕ)

-- Condition 1: 6 years ago, \( p \) was half of \( q \) in age.
axiom cond1 : P = Q / 2

-- Condition 2: The ratio of their present ages is 3:4.
axiom cond2 : (P + 6) * 4 = (Q + 6) * 3

-- We need to prove: the total of their present ages is 21
theorem total_present_ages : P' + Q' = 21 :=
by
  -- We already have the variables and axioms in the context, so we just need to state the goal
  sorry

end NUMINAMATH_GPT_total_present_ages_l1329_132918


namespace NUMINAMATH_GPT_sum_of_three_positives_eq_2002_l1329_132999

theorem sum_of_three_positives_eq_2002 : 
  ∃ (n : ℕ), n = 334000 ∧ (∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ (A B C : ℕ), f A B C ↔ (0 < A ∧ A ≤ B ∧ B ≤ C ∧ A + B + C = 2002))) := by
  sorry

end NUMINAMATH_GPT_sum_of_three_positives_eq_2002_l1329_132999


namespace NUMINAMATH_GPT_integer_solutions_determinant_l1329_132915

theorem integer_solutions_determinant (a b c d : ℤ)
    (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
sorry

end NUMINAMATH_GPT_integer_solutions_determinant_l1329_132915


namespace NUMINAMATH_GPT_point_P_lies_on_x_axis_l1329_132976

noncomputable def point_on_x_axis (x : ℝ) : Prop :=
  (0 = (0 : ℝ)) -- This is a placeholder definition stating explicitly that point lies on the x-axis

theorem point_P_lies_on_x_axis (x : ℝ) : point_on_x_axis x :=
by
  sorry

end NUMINAMATH_GPT_point_P_lies_on_x_axis_l1329_132976


namespace NUMINAMATH_GPT_pencil_cost_is_11_l1329_132951

-- Define the initial and remaining amounts
def initial_amount : ℤ := 15
def remaining_amount : ℤ := 4

-- Define the cost of the pencil
def cost_of_pencil : ℤ := initial_amount - remaining_amount

-- The statement we need to prove
theorem pencil_cost_is_11 : cost_of_pencil = 11 :=
by
  sorry

end NUMINAMATH_GPT_pencil_cost_is_11_l1329_132951


namespace NUMINAMATH_GPT_cost_price_of_table_l1329_132982

theorem cost_price_of_table (C S : ℝ) (h1 : S = 1.25 * C) (h2 : S = 4800) : C = 3840 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_of_table_l1329_132982


namespace NUMINAMATH_GPT_per_can_price_difference_cents_l1329_132949

   theorem per_can_price_difference_cents :
     let bulk_warehouse_price_per_case := 12.0
     let bulk_warehouse_cans_per_case := 48
     let bulk_warehouse_discount := 0.10
     let local_store_price_per_case := 6.0
     let local_store_cans_per_case := 12
     let local_store_promotion_factor := 1.5 -- represents the effect of the promotion (3 cases for the price of 2.5 cases)
     let bulk_warehouse_price_per_can := (bulk_warehouse_price_per_case * (1 - bulk_warehouse_discount)) / bulk_warehouse_cans_per_case
     let local_store_price_per_can := (local_store_price_per_case * local_store_promotion_factor) / (local_store_cans_per_case * 3)
     let price_difference_cents := (local_store_price_per_can - bulk_warehouse_price_per_can) * 100
     price_difference_cents = 19.17 :=
   by
     sorry
   
end NUMINAMATH_GPT_per_can_price_difference_cents_l1329_132949


namespace NUMINAMATH_GPT_fraction_subtraction_l1329_132967

theorem fraction_subtraction : (3 + 5 + 7) / (2 + 4 + 6) - (2 - 4 + 6) / (3 - 5 + 7) = 9 / 20 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1329_132967


namespace NUMINAMATH_GPT_area_change_l1329_132916

theorem area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := 1.2 * L
  let B' := 0.8 * B
  let A := L * B
  let A' := L' * B'
  A' = 0.96 * A :=
by
  sorry

end NUMINAMATH_GPT_area_change_l1329_132916


namespace NUMINAMATH_GPT_frank_columns_l1329_132964

theorem frank_columns (people : ℕ) (brownies_per_person : ℕ) (rows : ℕ)
  (h1 : people = 6) (h2 : brownies_per_person = 3) (h3 : rows = 3) : 
  (people * brownies_per_person) / rows = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_frank_columns_l1329_132964


namespace NUMINAMATH_GPT_product_of_p_r_s_l1329_132917

theorem product_of_p_r_s
  (p r s : ℕ)
  (h1 : 3^p + 3^4 = 90)
  (h2 : 2^r + 44 = 76)
  (h3 : 5^3 + 6^s = 1421) :
  p * r * s = 40 := 
sorry

end NUMINAMATH_GPT_product_of_p_r_s_l1329_132917


namespace NUMINAMATH_GPT_arrangement_count_l1329_132919

def number_of_arrangements (slots total_geometry total_number_theory : ℕ) : ℕ :=
  Nat.choose slots total_geometry

theorem arrangement_count :
  number_of_arrangements 8 5 3 = 56 := 
by
  sorry

end NUMINAMATH_GPT_arrangement_count_l1329_132919


namespace NUMINAMATH_GPT_smallest_interior_angle_l1329_132907

open Real

theorem smallest_interior_angle (A B C : ℝ) (hA : 0 < A ∧ A < π)
    (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
    (h_sum_angles : A + B + C = π)
    (h_ratio : sin A / sin B = 2 / sqrt 6 ∧ sin A / sin C = 2 / (sqrt 3 + 1)) :
    min A (min B C) = π / 4 := 
  by sorry

end NUMINAMATH_GPT_smallest_interior_angle_l1329_132907


namespace NUMINAMATH_GPT_polynomial_simplification_l1329_132979

theorem polynomial_simplification (x : ℤ) :
  (5 * x ^ 12 + 8 * x ^ 11 + 10 * x ^ 9) + (3 * x ^ 13 + 2 * x ^ 12 + x ^ 11 + 6 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9) =
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 16 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l1329_132979


namespace NUMINAMATH_GPT_total_amount_paid_l1329_132908

def apples_kg := 8
def apples_rate := 70
def mangoes_kg := 9
def mangoes_rate := 65
def oranges_kg := 5
def oranges_rate := 50
def bananas_kg := 3
def bananas_rate := 30

def total_amount := (apples_kg * apples_rate) + (mangoes_kg * mangoes_rate) + (oranges_kg * oranges_rate) + (bananas_kg * bananas_rate)

theorem total_amount_paid : total_amount = 1485 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1329_132908


namespace NUMINAMATH_GPT_number_of_articles_l1329_132941

theorem number_of_articles (C S : ℝ) (N : ℝ) 
    (h1 : N * C = 40 * S) 
    (h2 : (S - C) / C * 100 = 49.999999999999986) : 
    N = 60 :=
sorry

end NUMINAMATH_GPT_number_of_articles_l1329_132941


namespace NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1329_132930

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1329_132930


namespace NUMINAMATH_GPT_sum_of_cubes_of_three_consecutive_integers_l1329_132929

theorem sum_of_cubes_of_three_consecutive_integers (a : ℕ) (h : (a * a) + (a + 1) * (a + 1) + (a + 2) * (a + 2) = 2450) : a * a * a + (a + 1) * (a + 1) * (a + 1) + (a + 2) * (a + 2) * (a + 2) = 73341 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_three_consecutive_integers_l1329_132929


namespace NUMINAMATH_GPT_number_of_female_athletes_l1329_132959

theorem number_of_female_athletes (male_athletes female_athletes male_selected female_selected : ℕ)
  (h1 : male_athletes = 56)
  (h2 : female_athletes = 42)
  (h3 : male_selected = 8)
  (ratio : male_athletes / female_athletes = 4 / 3)
  (stratified_sampling : female_selected = (3 / 4) * male_selected)
  : female_selected = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_female_athletes_l1329_132959


namespace NUMINAMATH_GPT_continuous_stripe_probability_is_3_16_l1329_132911

-- Define the stripe orientation enumeration
inductive StripeOrientation
| diagonal
| straight

-- Define the face enumeration
inductive Face
| front
| back
| left
| right
| top
| bottom

-- Total number of stripe combinations (2^6 for each face having 2 orientations)
def total_combinations : ℕ := 2^6

-- Number of combinations for continuous stripes along length, width, and height
def length_combinations : ℕ := 2^2 -- 4 combinations
def width_combinations : ℕ := 2^2  -- 4 combinations
def height_combinations : ℕ := 2^2 -- 4 combinations

-- Total number of continuous stripe combinations across all dimensions
def total_continuous_stripe_combinations : ℕ :=
  length_combinations + width_combinations + height_combinations

-- Probability calculation
def continuous_stripe_probability : ℚ :=
  total_continuous_stripe_combinations / total_combinations

-- Final theorem statement
theorem continuous_stripe_probability_is_3_16 :
  continuous_stripe_probability = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_continuous_stripe_probability_is_3_16_l1329_132911


namespace NUMINAMATH_GPT_find_root_and_coefficient_l1329_132952

theorem find_root_and_coefficient (m: ℝ) (x: ℝ) (h₁: x ^ 2 - m * x - 6 = 0) (h₂: x = 3) :
  (x = 3 ∧ -2 = -6 / 3 ∨ m = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_root_and_coefficient_l1329_132952


namespace NUMINAMATH_GPT_find_a_l1329_132910

theorem find_a (a : ℝ) (M : Set ℝ) (N : Set ℝ) : 
  M = {1, 3} → N = {1 - a, 3} → (M ∪ N) = {1, 2, 3} → a = -1 :=
by
  intros hM hN hUnion
  sorry

end NUMINAMATH_GPT_find_a_l1329_132910


namespace NUMINAMATH_GPT_find_solutions_l1329_132962

theorem find_solutions (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48 →
  (x = 1.2 ∨ x = -81.2) :=
by sorry

end NUMINAMATH_GPT_find_solutions_l1329_132962


namespace NUMINAMATH_GPT_time_spent_per_egg_in_seconds_l1329_132935

-- Definitions based on the conditions in the problem
def minutes_per_roll : ℕ := 30
def number_of_rolls : ℕ := 7
def total_cleaning_time : ℕ := 225
def number_of_eggs : ℕ := 60

-- Problem statement
theorem time_spent_per_egg_in_seconds :
  (total_cleaning_time - number_of_rolls * minutes_per_roll) * 60 / number_of_eggs = 15 := by
  sorry

end NUMINAMATH_GPT_time_spent_per_egg_in_seconds_l1329_132935


namespace NUMINAMATH_GPT_determine_n_l1329_132957

theorem determine_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26) : n = 25 :=
by
  sorry

end NUMINAMATH_GPT_determine_n_l1329_132957


namespace NUMINAMATH_GPT_find_range_of_a_l1329_132986

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}

noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem find_range_of_a : {a : ℝ | set_B a ⊆ set_A} = {a : ℝ | a < -1} ∪ {1} :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l1329_132986


namespace NUMINAMATH_GPT_base_d_digit_difference_l1329_132923

theorem base_d_digit_difference (A C d : ℕ) (h1 : d > 8)
  (h2 : d * A + C + (d * C + C) = 2 * d^2 + 3 * d + 2) :
  (A - C = d + 1) :=
sorry

end NUMINAMATH_GPT_base_d_digit_difference_l1329_132923


namespace NUMINAMATH_GPT_ball_hits_ground_at_t_l1329_132988

noncomputable def ball_height (t : ℝ) : ℝ := -6 * t^2 - 10 * t + 56

theorem ball_hits_ground_at_t :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 7 / 3 := by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_at_t_l1329_132988


namespace NUMINAMATH_GPT_Z_is_all_positive_integers_l1329_132903

theorem Z_is_all_positive_integers (Z : Set ℕ) (h_nonempty : Z.Nonempty)
(h1 : ∀ x ∈ Z, 4 * x ∈ Z)
(h2 : ∀ x ∈ Z, (Nat.sqrt x) ∈ Z) : 
Z = { n : ℕ | n > 0 } :=
sorry

end NUMINAMATH_GPT_Z_is_all_positive_integers_l1329_132903


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1329_132913

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

theorem intersection_of_A_and_B : A ∩ B = {z | 2 ≤ z ∧ z < 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1329_132913


namespace NUMINAMATH_GPT_max_P_l1329_132909

noncomputable def P (a b : ℝ) : ℝ :=
  (a^2 + 6*b + 1) / (a^2 + a)

theorem max_P (a b x1 x2 x3 : ℝ) (h1 : a = x1 + x2 + x3) (h2 : a = x1 * x2 * x3) (h3 : ab = x1 * x2 + x2 * x3 + x3 * x1) 
    (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
    P a b ≤ (9 + Real.sqrt 3) / 9 := 
sorry

end NUMINAMATH_GPT_max_P_l1329_132909


namespace NUMINAMATH_GPT_percentage_of_girls_taking_lunch_l1329_132971

theorem percentage_of_girls_taking_lunch 
  (total_students : ℕ)
  (boys_ratio girls_ratio : ℕ)
  (boys_to_girls_ratio : boys_ratio + girls_ratio = 10)
  (boys : ℕ)
  (girls : ℕ)
  (boys_calc : boys = (boys_ratio * total_students) / 10)
  (girls_calc : girls = (girls_ratio * total_students) / 10)
  (boys_lunch_percentage : ℕ)
  (boys_lunch : ℕ)
  (boys_lunch_calc : boys_lunch = (boys_lunch_percentage * boys) / 100)
  (total_lunch_percentage : ℕ)
  (total_lunch : ℕ)
  (total_lunch_calc : total_lunch = (total_lunch_percentage * total_students) / 100)
  (girls_lunch : ℕ)
  (girls_lunch_calc : girls_lunch = total_lunch - boys_lunch) :
  ((girls_lunch * 100) / girls) = 40 :=
by 
  -- The proof can be filled in here
  sorry

end NUMINAMATH_GPT_percentage_of_girls_taking_lunch_l1329_132971


namespace NUMINAMATH_GPT_fraction_of_girls_l1329_132943

theorem fraction_of_girls (G T B : ℕ) (Fraction : ℚ)
  (h1 : Fraction * G = (1/3 : ℚ) * T)
  (h2 : (B : ℚ) / G = 1/2) :
  Fraction = 1/2 := by
  sorry

end NUMINAMATH_GPT_fraction_of_girls_l1329_132943


namespace NUMINAMATH_GPT_min_value_expression_l1329_132900

/--
  Prove that the minimum value of the expression (xy - 2)^2 + (x + y - 1)^2 
  for real numbers x and y is 2.
--/
theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, (a * b - 2)^2 + (a + b - 1)^2 ≥ (x * y - 2)^2 + (x + y - 1)^2 ) ∧ 
  (x * y - 2)^2 + (x + y - 1)^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1329_132900


namespace NUMINAMATH_GPT_correct_barometric_pressure_l1329_132983

noncomputable def true_barometric_pressure (p1 p2 v1 v2 T1 T2 observed_pressure_final observed_pressure_initial : ℝ) : ℝ :=
  let combined_gas_law : ℝ := (p1 * v1 * T2) / (v2 * T1)
  observed_pressure_final + combined_gas_law

theorem correct_barometric_pressure :
  true_barometric_pressure 58 56 143 155 288 303 692 704 = 748 :=
by
  sorry

end NUMINAMATH_GPT_correct_barometric_pressure_l1329_132983


namespace NUMINAMATH_GPT_period_is_3_years_l1329_132905

def gain_of_B_per_annum (principal : ℕ) (rate_A rate_B : ℚ) : ℚ := 
  (rate_B - rate_A) * principal

def period (principal : ℕ) (rate_A rate_B : ℚ) (total_gain : ℚ) : ℚ := 
  total_gain / gain_of_B_per_annum principal rate_A rate_B

theorem period_is_3_years :
  period 1500 (10 / 100) (11.5 / 100) 67.5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_period_is_3_years_l1329_132905


namespace NUMINAMATH_GPT_complex_product_eq_50i_l1329_132960

open Complex

theorem complex_product_eq_50i : 
  let Q := (4 : ℂ) + 3 * I
  let E := (2 * I : ℂ)
  let D := (4 : ℂ) - 3 * I
  Q * E * D = 50 * I :=
by
  -- Complex numbers and multiplication are handled here
  sorry

end NUMINAMATH_GPT_complex_product_eq_50i_l1329_132960


namespace NUMINAMATH_GPT_c_minus_a_is_10_l1329_132955

variable (a b c d k : ℝ)

theorem c_minus_a_is_10 (h1 : a + b = 90)
                        (h2 : b + c = 100)
                        (h3 : a + c + d = 180)
                        (h4 : a^2 + b^2 + c^2 + d^2 = k) :
  c - a = 10 :=
by sorry

end NUMINAMATH_GPT_c_minus_a_is_10_l1329_132955


namespace NUMINAMATH_GPT_part1_part2_l1329_132950

open Complex

noncomputable def z1 : ℂ := 1 - 2 * I
noncomputable def z2 : ℂ := 4 + 3 * I

theorem part1 : z1 * z2 = 10 - 5 * I := by
  sorry

noncomputable def z : ℂ := -Real.sqrt 2 - Real.sqrt 2 * I

theorem part2 (h_abs_z : abs z = 2)
              (h_img_eq_real : z.im = (3 * z1 - z2).re)
              (h_quadrant : z.re < 0 ∧ z.im < 0) : z = -Real.sqrt 2 - Real.sqrt 2 * I := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1329_132950


namespace NUMINAMATH_GPT_largest_sum_digits_24_hour_watch_l1329_132924

theorem largest_sum_digits_24_hour_watch : 
  (∃ h m : ℕ, 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ 
              (h / 10 + h % 10 + m / 10 + m % 10 = 24)) :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_digits_24_hour_watch_l1329_132924


namespace NUMINAMATH_GPT_alyssa_plums_correct_l1329_132944

def total_plums : ℕ := 27
def jason_plums : ℕ := 10
def alyssa_plums : ℕ := 17

theorem alyssa_plums_correct : alyssa_plums = total_plums - jason_plums := by
  sorry

end NUMINAMATH_GPT_alyssa_plums_correct_l1329_132944


namespace NUMINAMATH_GPT_graduation_ceremony_l1329_132995

theorem graduation_ceremony (teachers administrators graduates chairs : ℕ) 
  (h1 : teachers = 20) 
  (h2 : administrators = teachers / 2) 
  (h3 : graduates = 50) 
  (h4 : chairs = 180) :
  (chairs - (teachers + administrators + graduates)) / graduates = 2 :=
by 
  sorry

end NUMINAMATH_GPT_graduation_ceremony_l1329_132995


namespace NUMINAMATH_GPT_alex_jellybeans_l1329_132922

theorem alex_jellybeans (x : ℕ) : x = 254 → x ≥ 150 ∧ x % 15 = 14 ∧ x % 17 = 16 :=
by
  sorry

end NUMINAMATH_GPT_alex_jellybeans_l1329_132922


namespace NUMINAMATH_GPT_find_b_if_continuous_l1329_132998

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 5 * x^2 + 4 else b * x + 1

theorem find_b_if_continuous (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 23 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_if_continuous_l1329_132998


namespace NUMINAMATH_GPT_max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l1329_132928

noncomputable def f (x : ℝ) := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_f_gt_sqrt2 : (∃ x : ℝ, f x > Real.sqrt 2) :=
sorry

theorem f_is_periodic : ∀ x : ℝ, f (x - 2 * Real.pi) = f x :=
sorry

theorem f_pi_shift_pos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0 :=
sorry

end NUMINAMATH_GPT_max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l1329_132928


namespace NUMINAMATH_GPT_charge_for_each_additional_fifth_mile_l1329_132921

theorem charge_for_each_additional_fifth_mile
  (initial_charge : ℝ)
  (total_charge : ℝ)
  (distance_in_miles : ℕ)
  (distance_per_increment : ℝ)
  (x : ℝ) :
  initial_charge = 2.10 →
  total_charge = 17.70 →
  distance_in_miles = 8 →
  distance_per_increment = 1/5 →
  (total_charge - initial_charge) / ((distance_in_miles / distance_per_increment) - 1) = x →
  x = 0.40 :=
by
  intros h_initial_charge h_total_charge h_distance_in_miles h_distance_per_increment h_eq
  sorry

end NUMINAMATH_GPT_charge_for_each_additional_fifth_mile_l1329_132921


namespace NUMINAMATH_GPT_find_initial_money_l1329_132940

-- Definitions of the conditions
def basketball_card_cost : ℕ := 3
def baseball_card_cost : ℕ := 4
def basketball_packs : ℕ := 2
def baseball_decks : ℕ := 5
def change_received : ℕ := 24

-- Total cost calculation
def total_cost : ℕ := (basketball_card_cost * basketball_packs) + (baseball_card_cost * baseball_decks)

-- Initial money calculation
def initial_money : ℕ := total_cost + change_received

-- Proof statement
theorem find_initial_money : initial_money = 50 := 
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_find_initial_money_l1329_132940


namespace NUMINAMATH_GPT_interest_rate_per_annum_is_four_l1329_132984

-- Definitions
def P : ℕ := 300
def t : ℕ := 8
def I : ℤ := P - 204

-- Interest formula
def simple_interest (P : ℕ) (r : ℕ) (t : ℕ) : ℤ := P * r * t / 100

-- Statement to prove
theorem interest_rate_per_annum_is_four :
  ∃ r : ℕ, I = simple_interest P r t ∧ r = 4 :=
by sorry

end NUMINAMATH_GPT_interest_rate_per_annum_is_four_l1329_132984


namespace NUMINAMATH_GPT_solve_for_x_l1329_132937

theorem solve_for_x (x : ℚ) 
  (h : (1/3 : ℚ) + 1/x = (7/9 : ℚ) + 1) : 
  x = 9/13 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1329_132937


namespace NUMINAMATH_GPT_rate_per_kg_for_apples_l1329_132977

theorem rate_per_kg_for_apples (A : ℝ) :
  (8 * A + 9 * 45 = 965) → (A = 70) :=
by
  sorry

end NUMINAMATH_GPT_rate_per_kg_for_apples_l1329_132977


namespace NUMINAMATH_GPT_can_reach_4_white_l1329_132912

/-
We define the possible states and operations on the urn as described.
-/

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def operation1 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation2 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation3 (u : Urn) : Urn :=
  { white := u.white - 1, black := u.black - 1 }

def operation4 (u : Urn) : Urn :=
  { white := u.white - 2, black := u.black + 1 }

theorem can_reach_4_white : ∃ (u : Urn), u.white = 4 ∧ u.black > 0 :=
  sorry

end NUMINAMATH_GPT_can_reach_4_white_l1329_132912


namespace NUMINAMATH_GPT_factorization_6x2_minus_24x_plus_18_l1329_132906

theorem factorization_6x2_minus_24x_plus_18 :
    ∀ x : ℝ, 6 * x^2 - 24 * x + 18 = 6 * (x - 1) * (x - 3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorization_6x2_minus_24x_plus_18_l1329_132906


namespace NUMINAMATH_GPT_minimal_range_of_sample_l1329_132965

theorem minimal_range_of_sample (x1 x2 x3 x4 x5 : ℝ) 
  (mean_condition : (x1 + x2 + x3 + x4 + x5) / 5 = 6) 
  (median_condition : x3 = 10) 
  (sample_order : x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5) : 
  (x5 - x1) = 10 :=
sorry

end NUMINAMATH_GPT_minimal_range_of_sample_l1329_132965


namespace NUMINAMATH_GPT_number_of_scenarios_l1329_132990

theorem number_of_scenarios :
  ∃ (count : ℕ), count = 42244 ∧
  (∃ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
    x1 % 7 = 0 ∧ x2 % 7 = 0 ∧ x3 % 7 = 0 ∧ x4 % 7 = 0 ∧
    x5 % 13 = 0 ∧ x6 % 13 = 0 ∧ x7 % 13 = 0 ∧
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 270) :=
sorry

end NUMINAMATH_GPT_number_of_scenarios_l1329_132990


namespace NUMINAMATH_GPT_angle_A_is_120_degrees_l1329_132936

theorem angle_A_is_120_degrees
  (b c l_a : ℝ)
  (h : (1 / b) + (1 / c) = 1 / l_a) :
  ∃ A : ℝ, A = 120 :=
by
  sorry

end NUMINAMATH_GPT_angle_A_is_120_degrees_l1329_132936


namespace NUMINAMATH_GPT_probability_of_drawing_K_is_2_over_27_l1329_132992

-- Define the total number of cards in a standard deck of 54 cards
def total_cards : ℕ := 54

-- Define the number of "K" cards in the standard deck
def num_K_cards : ℕ := 4

-- Define the probability function for drawing a "K"
def probability_drawing_K (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- Prove that the probability of drawing a "K" is 2/27
theorem probability_of_drawing_K_is_2_over_27 :
  probability_drawing_K total_cards num_K_cards = 2 / 27 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_K_is_2_over_27_l1329_132992


namespace NUMINAMATH_GPT_area_of_right_angled_isosceles_triangle_l1329_132997

-- Definitions
variables {x y : ℝ}
def is_right_angled_isosceles (x y : ℝ) : Prop := y^2 = 2 * x^2
def sum_of_square_areas (x y : ℝ) : Prop := x^2 + x^2 + y^2 = 72

-- Theorem
theorem area_of_right_angled_isosceles_triangle (x y : ℝ) 
  (h1 : is_right_angled_isosceles x y) 
  (h2 : sum_of_square_areas x y) : 
  1/2 * x^2 = 9 :=
sorry

end NUMINAMATH_GPT_area_of_right_angled_isosceles_triangle_l1329_132997


namespace NUMINAMATH_GPT_tip_is_24_l1329_132989

-- Definitions based on conditions
def women's_haircut_cost : ℕ := 48
def children's_haircut_cost : ℕ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℚ := 0.20

-- Calculating total cost and tip amount
def total_cost : ℕ := women's_haircut_cost + (number_of_children * children's_haircut_cost)
def tip_amount : ℚ := tip_percentage * total_cost

-- Lean theorem statement based on the problem
theorem tip_is_24 : tip_amount = 24 := by
  sorry

end NUMINAMATH_GPT_tip_is_24_l1329_132989


namespace NUMINAMATH_GPT_rectangle_bounds_product_l1329_132945

theorem rectangle_bounds_product (b : ℝ) :
  (∃ b, y = 3 ∧ y = 7 ∧ x = -1 ∧ (x = b) 
   → (b = 3 ∨ b = -5) 
    ∧ (3 * -5 = -15)) :=
sorry

end NUMINAMATH_GPT_rectangle_bounds_product_l1329_132945


namespace NUMINAMATH_GPT_floor_tiling_l1329_132980

theorem floor_tiling (n : ℕ) (x : ℕ) (h1 : 6 * x = n^2) : 6 ∣ n := sorry

end NUMINAMATH_GPT_floor_tiling_l1329_132980


namespace NUMINAMATH_GPT_radius_circumcircle_l1329_132970

variables (R1 R2 R3 : ℝ)
variables (d : ℝ)
variables (R : ℝ)

noncomputable def sum_radii := R1 + R2 = 11
noncomputable def distance_centers := d = 5 * Real.sqrt 17
noncomputable def radius_third_sphere := R3 = 8
noncomputable def touching := R1 + R2 + 2 * R3 = d

theorem radius_circumcircle :
  R = 5 * Real.sqrt 17 / 2 :=
  by
  -- Use conditions here if necessary
  sorry

end NUMINAMATH_GPT_radius_circumcircle_l1329_132970


namespace NUMINAMATH_GPT_how_many_green_towels_l1329_132968

-- Define the conditions
def initial_white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34
def towels_left_after_giving : ℕ := 22

-- Define the statement to prove
theorem how_many_green_towels (G : ℕ) (initial_white : ℕ) (given : ℕ) (left_after : ℕ) :
  initial_white = initial_white_towels →
  given = towels_given_to_mother →
  left_after = towels_left_after_giving →
  (G + initial_white) - given = left_after →
  G = 35 :=
by
  intros
  sorry

end NUMINAMATH_GPT_how_many_green_towels_l1329_132968


namespace NUMINAMATH_GPT_range_of_m_l1329_132961

theorem range_of_m (p q : Prop) (m : ℝ) (h₀ : ∀ x : ℝ, p ↔ (x^2 - 8 * x - 20 ≤ 0)) 
  (h₁ : ∀ x : ℝ, q ↔ (x^2 - 2 * x + 1 - m^2 ≤ 0)) (hm : m > 0) 
  (hsuff : (∃ x : ℝ, x > 10 ∨ x < -2) → (∃ x : ℝ, x < 1 - m ∨ x > 1 + m)) :
  0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1329_132961


namespace NUMINAMATH_GPT_emily_furniture_assembly_time_l1329_132996

-- Definitions based on conditions
def chairs := 4
def tables := 2
def time_per_piece := 8

-- Proof statement
theorem emily_furniture_assembly_time : (chairs + tables) * time_per_piece = 48 :=
by
  sorry

end NUMINAMATH_GPT_emily_furniture_assembly_time_l1329_132996


namespace NUMINAMATH_GPT_brendan_total_wins_l1329_132947

-- Define the number of matches won in each round
def matches_won_first_round : ℕ := 6
def matches_won_second_round : ℕ := 4
def matches_won_third_round : ℕ := 3
def matches_won_final_round : ℕ := 5

-- Define the total number of matches won
def total_matches_won : ℕ := 
  matches_won_first_round + matches_won_second_round + matches_won_third_round + matches_won_final_round

-- State the theorem that needs to be proven
theorem brendan_total_wins : total_matches_won = 18 := by
  sorry

end NUMINAMATH_GPT_brendan_total_wins_l1329_132947


namespace NUMINAMATH_GPT_locus_of_point_R_l1329_132975

theorem locus_of_point_R :
  ∀ (P Q O F R : ℝ × ℝ)
    (hP_on_parabola : ∃ x1 y1, P = (x1, y1) ∧ y1^2 = 2 * x1)
    (h_directrix : Q.1 = -1 / 2)
    (hQ : ∃ x1 y1, Q = (x1, y1) ∧ P = (x1, y1))
    (hO : O = (0, 0))
    (hF : F = (1 / 2, 0))
    (h_intersection : ∃ x y, 
      R = (x, y) ∧
      ∃ x1 y1,
      P = (x1, y1) ∧ 
      y1^2 = 2 * x1 ∧
      ∃ (m_OP : ℝ), 
        m_OP = y1 / x1 ∧ 
        y = m_OP * x ∧
      ∃ (m_FQ : ℝ), 
        m_FQ = -y1 ∧
        y = m_FQ * x + y1 * (1 + 3 / 2)),
  R.2^2 = -2 * R.1^2 + R.1 :=
by sorry

end NUMINAMATH_GPT_locus_of_point_R_l1329_132975


namespace NUMINAMATH_GPT_sum_a2_to_a5_eq_zero_l1329_132938

theorem sum_a2_to_a5_eq_zero 
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : ∀ x : ℝ, x * (1 - 2 * x)^4 = a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_2 + a_3 + a_4 + a_5 = 0 :=
sorry

end NUMINAMATH_GPT_sum_a2_to_a5_eq_zero_l1329_132938


namespace NUMINAMATH_GPT_norm_squared_sum_l1329_132974

variables (p q : ℝ × ℝ)
def n : ℝ × ℝ := (4, -2)
variables (h_midpoint : n = ((p.1 + q.1) / 2, (p.2 + q.2) / 2))
variables (h_dot_product : p.1 * q.1 + p.2 * q.2 = 12)

theorem norm_squared_sum : (p.1 ^ 2 + p.2 ^ 2) + (q.1 ^ 2 + q.2 ^ 2) = 56 :=
by
  sorry

end NUMINAMATH_GPT_norm_squared_sum_l1329_132974


namespace NUMINAMATH_GPT_possible_values_of_AD_l1329_132902

-- Define the conditions as variables
variables {A B C D : ℝ}
variables {AB BC CD : ℝ}

-- Assume the given conditions
def conditions (A B C D : ℝ) (AB BC CD : ℝ) : Prop :=
  AB = 1 ∧ BC = 2 ∧ CD = 4

-- Define the proof goal: proving the possible values of AD
theorem possible_values_of_AD (h : conditions A B C D AB BC CD) :
  ∃ AD, AD = 1 ∨ AD = 3 ∨ AD = 5 ∨ AD = 7 :=
sorry

end NUMINAMATH_GPT_possible_values_of_AD_l1329_132902
