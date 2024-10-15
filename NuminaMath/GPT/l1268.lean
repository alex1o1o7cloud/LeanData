import Mathlib

namespace NUMINAMATH_GPT_vojta_correct_sum_l1268_126878

theorem vojta_correct_sum (S A B C : ℕ)
  (h1 : S + (10 * B + C) = 2224)
  (h2 : S + (10 * A + B) = 2198)
  (h3 : S + (10 * A + C) = 2204)
  (A_digit : 0 ≤ A ∧ A < 10)
  (B_digit : 0 ≤ B ∧ B < 10)
  (C_digit : 0 ≤ C ∧ C < 10) :
  S + 100 * A + 10 * B + C = 2324 := 
sorry

end NUMINAMATH_GPT_vojta_correct_sum_l1268_126878


namespace NUMINAMATH_GPT_simplify_expr_l1268_126810

theorem simplify_expr (x : ℝ) : 
  2 * x * (4 * x ^ 3 - 3 * x + 1) - 7 * (x ^ 3 - x ^ 2 + 3 * x - 4) = 
  8 * x ^ 4 - 7 * x ^ 3 + x ^ 2 - 19 * x + 28 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1268_126810


namespace NUMINAMATH_GPT_problem_prove_ω_and_delta_l1268_126830

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem problem_prove_ω_and_delta (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) 
    (h_sym_axis : ∀ x, f ω φ x = f ω φ (-(x + π))) 
    (h_center_sym : ∃ c : ℝ, (c = π / 2) ∧ (f ω φ c = 0)) 
    (h_monotone_increasing : ∀ x, -π ≤ x ∧ x ≤ -π / 2 → f ω φ x < f ω φ (x + 1)) :
    (ω = 1 / 3) ∧ (∀ δ : ℝ, (∀ x : ℝ, f ω φ (x + δ) = f ω φ (-x + δ)) → ∃ k : ℤ, δ = 2 * π + 3 * k * π) :=
by
  sorry

end NUMINAMATH_GPT_problem_prove_ω_and_delta_l1268_126830


namespace NUMINAMATH_GPT_simplify_fraction_l1268_126806

variable {x y : ℝ}

theorem simplify_fraction (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1268_126806


namespace NUMINAMATH_GPT_remainder_of_1999_pow_81_mod_7_eq_1_l1268_126847

/-- 
  Prove the remainder R when 1999^81 is divided by 7 is equal to 1.
  Conditions:
  - number: 1999
  - divisor: 7
-/
theorem remainder_of_1999_pow_81_mod_7_eq_1 : (1999 ^ 81) % 7 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_1999_pow_81_mod_7_eq_1_l1268_126847


namespace NUMINAMATH_GPT_infinite_x_differs_from_two_kth_powers_l1268_126849

theorem infinite_x_differs_from_two_kth_powers (k : ℕ) (h : k > 1) : 
  ∃ (f : ℕ → ℕ), (∀ n, f n = (2^(n+1))^k - (2^n)^k) ∧ (∀ n, ∀ a b : ℕ, ¬ f n = a^k + b^k) :=
sorry

end NUMINAMATH_GPT_infinite_x_differs_from_two_kth_powers_l1268_126849


namespace NUMINAMATH_GPT_max_tan_alpha_l1268_126826

theorem max_tan_alpha (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h : Real.tan (α + β) = 9 * Real.tan β) : Real.tan α ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_tan_alpha_l1268_126826


namespace NUMINAMATH_GPT_percentage_deposit_l1268_126850

theorem percentage_deposit (deposited : ℝ) (initial_amount : ℝ) (amount_deposited : ℝ) (P : ℝ) 
  (h1 : deposited = 750) 
  (h2 : initial_amount = 50000)
  (h3 : amount_deposited = 0.20 * (P / 100) * (0.25 * initial_amount))
  (h4 : amount_deposited = deposited) : 
  P = 30 := 
sorry

end NUMINAMATH_GPT_percentage_deposit_l1268_126850


namespace NUMINAMATH_GPT_snake_length_difference_l1268_126874

theorem snake_length_difference :
  ∀ (jake_len penny_len : ℕ), 
    jake_len > penny_len →
    jake_len + penny_len = 70 →
    jake_len = 41 →
    jake_len - penny_len = 12 :=
by
  intros jake_len penny_len h1 h2 h3
  sorry

end NUMINAMATH_GPT_snake_length_difference_l1268_126874


namespace NUMINAMATH_GPT_polynomial_not_factorizable_l1268_126890

theorem polynomial_not_factorizable
  (n m : ℕ)
  (hnm : n > m)
  (hm1 : m > 1)
  (hn_odd : n % 2 = 1)
  (hm_odd : m % 2 = 1) :
  ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧ (x^n + x^m + x + 1 = g * h) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_not_factorizable_l1268_126890


namespace NUMINAMATH_GPT_squared_expression_l1268_126859

variable (x : ℝ)

theorem squared_expression (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_GPT_squared_expression_l1268_126859


namespace NUMINAMATH_GPT_find_a_of_exponential_inverse_l1268_126829

theorem find_a_of_exponential_inverse (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x, a^x = 9 ↔ x = 2) : a = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_a_of_exponential_inverse_l1268_126829


namespace NUMINAMATH_GPT_complement_set_unique_l1268_126836

-- Define the universal set U
def U : Set ℕ := {1,2,3,4,5,6,7,8}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := {1,3}

-- The set B that we need to prove
def B : Set ℕ := {2,4,5,6,7,8}

-- State that B is the set with the given complement in U
theorem complement_set_unique (U : Set ℕ) (complement_B : Set ℕ) :
    (U \ complement_B = {2,4,5,6,7,8}) :=
by
    -- We need to prove B is the set {2,4,5,6,7,8}
    sorry

end NUMINAMATH_GPT_complement_set_unique_l1268_126836


namespace NUMINAMATH_GPT_last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l1268_126853

theorem last_number_of_nth_row (n : ℕ) : 
    let last_number := 2^n - 1
    last_number = 2^n - 1 := 
sorry

theorem sum_of_numbers_in_nth_row (n : ℕ) :
    let sum := (3 * 2^(n-3)) - 2^(n-2)
    sum = (3 * 2^(n-3)) - 2^(n-2) :=
sorry

theorem position_of_2008 : 
    let position := 985
    position = 985 :=
sorry

end NUMINAMATH_GPT_last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l1268_126853


namespace NUMINAMATH_GPT_total_toothpicks_needed_l1268_126804

theorem total_toothpicks_needed (length width : ℕ) (hl : length = 50) (hw : width = 40) : 
  (length + 1) * width + (width + 1) * length = 4090 := 
by
  -- proof omitted, replace this line with actual proof
  sorry

end NUMINAMATH_GPT_total_toothpicks_needed_l1268_126804


namespace NUMINAMATH_GPT_percentage_of_men_l1268_126898

variable (M : ℝ)

theorem percentage_of_men (h1 : 0.20 * M + 0.40 * (1 - M) = 0.33) : 
  M = 0.35 :=
sorry

end NUMINAMATH_GPT_percentage_of_men_l1268_126898


namespace NUMINAMATH_GPT_find_y_l1268_126838

noncomputable def a := (3/5) * 2500
noncomputable def b := (2/7) * ((5/8) * 4000 + (1/4) * 3600 - (11/20) * 7200)
noncomputable def c (y : ℚ) := (3/10) * y
def result (a b c : ℚ) := a * b / c

theorem find_y : ∃ y : ℚ, result a b (c y) = 25000 ∧ y = -4/21 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l1268_126838


namespace NUMINAMATH_GPT_exists_a_satisfying_f_l1268_126802

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 1 else x - 1

theorem exists_a_satisfying_f (a : ℝ) : 
  f (a + 1) = f a ↔ (a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end NUMINAMATH_GPT_exists_a_satisfying_f_l1268_126802


namespace NUMINAMATH_GPT_simplify_fraction_l1268_126837

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1268_126837


namespace NUMINAMATH_GPT_sequence_general_term_l1268_126857

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2) 
    (h_a₁ : S 1 = 1) (h_an : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) : 
  ∀ n, a n = 2 * n - 1 := 
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1268_126857


namespace NUMINAMATH_GPT_find_base_l1268_126880

theorem find_base (b : ℕ) (h : (3 * b + 2) ^ 2 = b ^ 3 + b + 4) : b = 8 :=
sorry

end NUMINAMATH_GPT_find_base_l1268_126880


namespace NUMINAMATH_GPT_juniors_involved_in_sports_l1268_126812

theorem juniors_involved_in_sports 
    (total_students : ℕ) (percentage_juniors : ℝ) (percentage_sports : ℝ) 
    (H1 : total_students = 500) 
    (H2 : percentage_juniors = 0.40) 
    (H3 : percentage_sports = 0.70) : 
    total_students * percentage_juniors * percentage_sports = 140 := 
by
  sorry

end NUMINAMATH_GPT_juniors_involved_in_sports_l1268_126812


namespace NUMINAMATH_GPT_inverse_B_squared_l1268_126876

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

def B_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -3, 2],
    ![  1, -1 ]]

theorem inverse_B_squared :
  B⁻¹ = B_inv →
  (B^2)⁻¹ = B_inv * B_inv :=
by sorry

end NUMINAMATH_GPT_inverse_B_squared_l1268_126876


namespace NUMINAMATH_GPT_find_ordered_pairs_l1268_126872

theorem find_ordered_pairs (a b : ℕ) (h1 : 2 * a + 1 ∣ 3 * b - 1) (h2 : 2 * b + 1 ∣ 3 * a - 1) : 
  (a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12) :=
by {
  sorry -- proof omitted
}

end NUMINAMATH_GPT_find_ordered_pairs_l1268_126872


namespace NUMINAMATH_GPT_abc_inequality_l1268_126883

theorem abc_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end NUMINAMATH_GPT_abc_inequality_l1268_126883


namespace NUMINAMATH_GPT_range_of_squared_sum_l1268_126819

theorem range_of_squared_sum (x y : ℝ) (h : x^2 + 1 / y^2 = 2) : ∃ z, z = x^2 + y^2 ∧ z ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_squared_sum_l1268_126819


namespace NUMINAMATH_GPT_convex_hexagon_possibilities_l1268_126895

noncomputable def hexagon_side_lengths : List ℕ := [1, 2, 3, 4, 5, 6]

theorem convex_hexagon_possibilities : 
  ∃ (hexagons : List (List ℕ)), 
    (∀ h ∈ hexagons, 
      (h.length = 6) ∧ 
      (∀ a ∈ h, a ∈ hexagon_side_lengths)) ∧ 
      (hexagons.length = 3) := 
sorry

end NUMINAMATH_GPT_convex_hexagon_possibilities_l1268_126895


namespace NUMINAMATH_GPT_problem_proof_l1268_126833

def delta (a b : ℕ) : ℕ := a^2 + b

theorem problem_proof :
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  delta u v = 5^88 + 7^18 :=
by
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  have h1: delta x y = 44 := by sorry
  have h2: delta z w = 18 := by sorry
  have hu: u = 5^44 := by sorry
  have hv: v = 7^18 := by sorry
  have hdelta: delta u v = 5^88 + 7^18 := by sorry
  exact hdelta

end NUMINAMATH_GPT_problem_proof_l1268_126833


namespace NUMINAMATH_GPT_lcm_of_8_9_5_10_l1268_126867

theorem lcm_of_8_9_5_10 : Nat.lcm (Nat.lcm 8 9) (Nat.lcm 5 10) = 360 := by
  sorry

end NUMINAMATH_GPT_lcm_of_8_9_5_10_l1268_126867


namespace NUMINAMATH_GPT_fraction_d_can_be_zero_l1268_126869

theorem fraction_d_can_be_zero :
  ∃ x : ℝ, (x + 1) / (x - 1) = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_d_can_be_zero_l1268_126869


namespace NUMINAMATH_GPT_percentage_decrease_l1268_126866

theorem percentage_decrease (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : x = 0.65 * z) : 
  ((z - y) / z) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l1268_126866


namespace NUMINAMATH_GPT_original_number_l1268_126835

theorem original_number (x y : ℕ) (h1 : x + y = 859560) (h2 : y = 859560 % 456) : x = 859376 ∧ 456 ∣ x :=
by
  sorry

end NUMINAMATH_GPT_original_number_l1268_126835


namespace NUMINAMATH_GPT_find_max_marks_l1268_126864

theorem find_max_marks (M : ℝ) (h1 : 0.60 * M = 80 + 100) : M = 300 := 
by
  sorry

end NUMINAMATH_GPT_find_max_marks_l1268_126864


namespace NUMINAMATH_GPT_bill_due_in_9_months_l1268_126892

-- Define the conditions
def true_discount : ℝ := 240
def face_value : ℝ := 2240
def interest_rate : ℝ := 0.16

-- Define the present value calculated from the true discount and face value
def present_value := face_value - true_discount

-- Define the time in months required to match the conditions
noncomputable def time_in_months : ℝ := 12 * ((face_value / present_value - 1) / interest_rate)

-- State the theorem that the bill is due in 9 months
theorem bill_due_in_9_months : time_in_months = 9 :=
by
  sorry

end NUMINAMATH_GPT_bill_due_in_9_months_l1268_126892


namespace NUMINAMATH_GPT_find_x_l1268_126808

-- Define vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 5)
def c (x : ℝ) : ℝ × ℝ := (3, x)

-- Dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Compute 8a - b
def sum_vec : ℝ × ℝ :=
  (8 * a.1 - b.1, 8 * a.2 - b.2)

-- Prove that x = 4 given condition
theorem find_x (x : ℝ) (h : dot_product sum_vec (c x) = 30) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1268_126808


namespace NUMINAMATH_GPT_general_formula_l1268_126843

theorem general_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end NUMINAMATH_GPT_general_formula_l1268_126843


namespace NUMINAMATH_GPT_initial_percentage_proof_l1268_126888

noncomputable def initialPercentageAntifreeze (P : ℝ) : Prop :=
  let initial_fluid : ℝ := 4
  let drained_fluid : ℝ := 2.2857
  let added_antifreeze_fluid : ℝ := 2.2857 * 0.8
  let final_percentage : ℝ := 0.5
  let final_fluid : ℝ := 4
  
  let initial_antifreeze : ℝ := initial_fluid * P
  let drained_antifreeze : ℝ := drained_fluid * P
  let total_antifreeze_after_replacement : ℝ := initial_antifreeze - drained_antifreeze + added_antifreeze_fluid
  
  total_antifreeze_after_replacement = final_fluid * final_percentage

-- Prove that the initial percentage is 0.1
theorem initial_percentage_proof : initialPercentageAntifreeze 0.1 :=
by
  dsimp [initialPercentageAntifreeze]
  simp
  exact sorry

end NUMINAMATH_GPT_initial_percentage_proof_l1268_126888


namespace NUMINAMATH_GPT_new_computer_lasts_l1268_126893

theorem new_computer_lasts (x : ℕ) 
  (h1 : 600 = 400 + 200)
  (h2 : ∀ y : ℕ, (2 * 200 = 400) → (2 * 3 = 6) → y = 6)
  (h3 : 200 = 600 - 400) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_new_computer_lasts_l1268_126893


namespace NUMINAMATH_GPT_goods_purchase_solutions_l1268_126844

theorem goods_purchase_solutions (a : ℕ) (h1 : 0 < a ∧ a ≤ 45) :
  ∃ x : ℝ, 45 - 20 * (x - 1) = a * x :=
by sorry

end NUMINAMATH_GPT_goods_purchase_solutions_l1268_126844


namespace NUMINAMATH_GPT_wizard_elixir_combinations_l1268_126884

def roots : ℕ := 4
def minerals : ℕ := 5
def incompatible_pairs : ℕ := 3
def total_combinations : ℕ := roots * minerals
def valid_combinations : ℕ := total_combinations - incompatible_pairs

theorem wizard_elixir_combinations : valid_combinations = 17 := by
  sorry

end NUMINAMATH_GPT_wizard_elixir_combinations_l1268_126884


namespace NUMINAMATH_GPT_find_k_l1268_126809

theorem find_k
  (t k : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : t = 20) :
  k = 68 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l1268_126809


namespace NUMINAMATH_GPT_currency_conversion_l1268_126894

variable (a : ℚ)

theorem currency_conversion
  (h1 : (0.5 / 100) * a = 75 / 100) -- 0.5% of 'a' = 75 paise
  (rate_usd : ℚ := 0.012)          -- Conversion rate (USD/INR)
  (rate_eur : ℚ := 0.010)          -- Conversion rate (EUR/INR)
  (rate_gbp : ℚ := 0.009)          -- Conversion rate (GBP/INR)
  (paise_to_rupees : ℚ := 1 / 100) -- 1 Rupee = 100 paise
  : (a * paise_to_rupees * rate_usd = 1.8) ∧
    (a * paise_to_rupees * rate_eur = 1.5) ∧
    (a * paise_to_rupees * rate_gbp = 1.35) :=
by
  sorry

end NUMINAMATH_GPT_currency_conversion_l1268_126894


namespace NUMINAMATH_GPT_range_of_first_term_l1268_126873

-- Define the arithmetic sequence and its common difference.
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the first n terms of the sequence.
def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Prove the range of the first term a1 given the conditions.
theorem range_of_first_term (a d : ℤ) (S : ℕ → ℤ) (h1 : d = -2)
  (h2 : ∀ n, S n = sum_of_first_n_terms a d n)
  (h3 : S 7 = S 7)
  (h4 : ∀ n, n ≠ 7 → S n < S 7) :
  12 < a ∧ a < 14 :=
by
  sorry

end NUMINAMATH_GPT_range_of_first_term_l1268_126873


namespace NUMINAMATH_GPT_area_of_OBEC_is_25_l1268_126851

noncomputable def area_OBEC : ℝ :=
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let E := (5, 5)
  let O := (0, 0)
  let area_triangle (P Q R : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))
  area_triangle O B E - area_triangle O E C

theorem area_of_OBEC_is_25 :
  area_OBEC = 25 := 
by
  sorry

end NUMINAMATH_GPT_area_of_OBEC_is_25_l1268_126851


namespace NUMINAMATH_GPT_parabola_symmetric_points_l1268_126886

theorem parabola_symmetric_points (a : ℝ) (x1 y1 x2 y2 m : ℝ) 
  (h_parabola : ∀ x, y = a * x^2)
  (h_a_pos : a > 0)
  (h_focus_directrix : 1 / (2 * a) = 1 / 4)
  (h_symmetric : y1 = a * x1^2 ∧ y2 = a * x2^2 ∧ ∃ m, y1 = m + (x1 - m))
  (h_product : x1 * x2 = -1 / 2) :
  m = 3 / 2 := 
sorry

end NUMINAMATH_GPT_parabola_symmetric_points_l1268_126886


namespace NUMINAMATH_GPT_half_angle_in_second_quadrant_l1268_126891

theorem half_angle_in_second_quadrant 
  {θ : ℝ} (k : ℤ)
  (hθ_quadrant4 : 2 * k * Real.pi + (3 / 2) * Real.pi ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi)
  (hcos : abs (Real.cos (θ / 2)) = - Real.cos (θ / 2)) : 
  ∃ m : ℤ, (m * Real.pi + (Real.pi / 2) ≤ θ / 2 ∧ θ / 2 ≤ m * Real.pi + Real.pi) :=
sorry

end NUMINAMATH_GPT_half_angle_in_second_quadrant_l1268_126891


namespace NUMINAMATH_GPT_speed_of_stream_l1268_126861

theorem speed_of_stream (v : ℝ) :
  (∀ s : ℝ, s = 3 → (3 + v) / (3 - v) = 2) → v = 1 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1268_126861


namespace NUMINAMATH_GPT_triangle_area_l1268_126871

theorem triangle_area (a b c : ℕ) (h : a = 12) (i : b = 16) (j : c = 20) (hc : c * c = a * a + b * b) :
  ∃ (area : ℕ), area = 96 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1268_126871


namespace NUMINAMATH_GPT_ne_of_P_l1268_126854

-- Define the initial proposition P
def P : Prop := ∀ m : ℝ, (0 ≤ m → 4^m ≥ 4 * m)

-- Define the negation of P
def not_P : Prop := ∃ m : ℝ, (0 ≤ m ∧ 4^m < 4 * m)

-- The theorem we need to prove
theorem ne_of_P : ¬P ↔ not_P :=
by
  sorry

end NUMINAMATH_GPT_ne_of_P_l1268_126854


namespace NUMINAMATH_GPT_truthful_dwarfs_count_l1268_126865

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end NUMINAMATH_GPT_truthful_dwarfs_count_l1268_126865


namespace NUMINAMATH_GPT_find_tangency_segments_equal_l1268_126897

-- Conditions of the problem as a theorem statement
theorem find_tangency_segments_equal (AB BC CD DA : ℝ) (x y : ℝ)
    (h1 : AB = 80)
    (h2 : BC = 140)
    (h3 : CD = 100)
    (h4 : DA = 120)
    (h5 : x + y = CD)
    (tangency_property : |x - y| = 0) :
  |x - y| = 0 :=
sorry

end NUMINAMATH_GPT_find_tangency_segments_equal_l1268_126897


namespace NUMINAMATH_GPT_find_smallest_angle_b1_l1268_126811

-- Definitions and conditions
def smallest_angle_in_sector (b1 e : ℕ) (k : ℕ := 5) : Prop :=
  2 * b1 + (k - 1) * k * e = 360 ∧ b1 + 2 * e = 36

theorem find_smallest_angle_b1 (b1 e : ℕ) : smallest_angle_in_sector b1 e → b1 = 30 :=
  sorry

end NUMINAMATH_GPT_find_smallest_angle_b1_l1268_126811


namespace NUMINAMATH_GPT_ramu_profit_percent_is_21_64_l1268_126858

-- Define the costs and selling price as constants
def cost_of_car : ℕ := 42000
def cost_of_repairs : ℕ := 13000
def selling_price : ℕ := 66900

-- Define the total cost and profit
def total_cost : ℕ := cost_of_car + cost_of_repairs
def profit : ℕ := selling_price - total_cost

-- Define the profit percent formula
def profit_percent : ℚ := ((profit : ℚ) / (total_cost : ℚ)) * 100

-- State the theorem we want to prove
theorem ramu_profit_percent_is_21_64 : profit_percent = 21.64 := by
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_is_21_64_l1268_126858


namespace NUMINAMATH_GPT_find_y_l1268_126841

theorem find_y (y : ℕ) (hy1 : y % 9 = 0) (hy2 : y^2 > 200) (hy3 : y < 30) : y = 18 :=
sorry

end NUMINAMATH_GPT_find_y_l1268_126841


namespace NUMINAMATH_GPT_quadratic_points_range_l1268_126803

theorem quadratic_points_range (a : ℝ) (y1 y2 y3 y4 : ℝ) :
  (∀ (x : ℝ), 
    (x = -4 → y1 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = -3 → y2 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 0 → y3 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 2 → y4 = a * x^2 + 4 * a * x - 6)) →
  (∃! (y : ℝ), y > 0 ∧ (y = y1 ∨ y = y2 ∨ y = y3 ∨ y = y4)) →
  (a < -2 ∨ a > 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_points_range_l1268_126803


namespace NUMINAMATH_GPT_pool_capacity_l1268_126827

theorem pool_capacity
  (pump_removes : ∀ (x : ℝ), x > 0 → (2 / 3) * x / 7.5 = (4 / 15) * x)
  (working_time : 0.15 * 60 = 9)
  (remaining_water : ∀ (x : ℝ), x > 0 → x - (0.8 * x) = 25) :
  ∃ x : ℝ, x = 125 :=
by
  sorry

end NUMINAMATH_GPT_pool_capacity_l1268_126827


namespace NUMINAMATH_GPT_jamie_minimum_4th_quarter_score_l1268_126828

-- Define the conditions for Jamie's scores and the average requirement
def qualifying_score := 85
def first_quarter_score := 80
def second_quarter_score := 85
def third_quarter_score := 78

-- The function to determine the required score in the 4th quarter
def minimum_score_for_quarter (N : ℕ) := first_quarter_score + second_quarter_score + third_quarter_score + N ≥ 4 * qualifying_score

-- The main statement to be proved
theorem jamie_minimum_4th_quarter_score (N : ℕ) : minimum_score_for_quarter N ↔ N ≥ 97 :=
by
  sorry

end NUMINAMATH_GPT_jamie_minimum_4th_quarter_score_l1268_126828


namespace NUMINAMATH_GPT_ratio_of_fixing_times_is_two_l1268_126831

noncomputable def time_per_shirt : ℝ := 1.5
noncomputable def number_of_shirts : ℕ := 10
noncomputable def number_of_pants : ℕ := 12
noncomputable def hourly_rate : ℝ := 30
noncomputable def total_cost : ℝ := 1530

theorem ratio_of_fixing_times_is_two :
  let total_hours := total_cost / hourly_rate
  let shirt_hours := number_of_shirts * time_per_shirt
  let pant_hours := total_hours - shirt_hours
  let time_per_pant := pant_hours / number_of_pants
  (time_per_pant / time_per_shirt) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_fixing_times_is_two_l1268_126831


namespace NUMINAMATH_GPT_lines_intersecting_sum_a_b_l1268_126881

theorem lines_intersecting_sum_a_b 
  (a b : ℝ) 
  (hx : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x = 3 * y + a)
  (hy : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ y = 3 * x + b)
  : a + b = -10 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersecting_sum_a_b_l1268_126881


namespace NUMINAMATH_GPT_pipe_cistern_problem_l1268_126813

theorem pipe_cistern_problem:
  ∀ (rate_p rate_q : ℝ),
    rate_p = 1 / 10 →
    rate_q = 1 / 15 →
    ∀ (filled_in_4_minutes : ℝ),
      filled_in_4_minutes = 4 * (rate_p + rate_q) →
      ∀ (remaining : ℝ),
        remaining = 1 - filled_in_4_minutes →
        ∀ (time_to_fill : ℝ),
          time_to_fill = remaining / rate_q →
          time_to_fill = 5 :=
by
  intros rate_p rate_q Hp Hq filled_in_4_minutes H4 remaining Hr time_to_fill Ht
  sorry

end NUMINAMATH_GPT_pipe_cistern_problem_l1268_126813


namespace NUMINAMATH_GPT_xyz_sum_l1268_126825

theorem xyz_sum (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  sorry

end NUMINAMATH_GPT_xyz_sum_l1268_126825


namespace NUMINAMATH_GPT_minimum_seedlings_needed_l1268_126845

theorem minimum_seedlings_needed (n : ℕ) (h1 : 75 ≤ n) (h2 : n ≤ 80) (H : 1200 * 100 / n = 1500) : n = 80 :=
sorry

end NUMINAMATH_GPT_minimum_seedlings_needed_l1268_126845


namespace NUMINAMATH_GPT_range_of_k_l1268_126801

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x

theorem range_of_k :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ k) → k ≥ Real.exp 1 - 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1268_126801


namespace NUMINAMATH_GPT_total_tickets_sold_l1268_126814

theorem total_tickets_sold (x y : ℕ) (h1 : 12 * x + 8 * y = 3320) (h2 : y = x + 240) : 
  x + y = 380 :=
by -- proof
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l1268_126814


namespace NUMINAMATH_GPT_number_of_ways_to_cut_pipe_l1268_126832

theorem number_of_ways_to_cut_pipe : 
  (∃ (x y: ℕ), 2 * x + 3 * y = 15) ∧ 
  (∃! (x y: ℕ), 2 * x + 3 * y = 15) :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_cut_pipe_l1268_126832


namespace NUMINAMATH_GPT_min_value_problem_l1268_126805

theorem min_value_problem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 3 * b = 1) :
    (1 / a) + (3 / b) ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_problem_l1268_126805


namespace NUMINAMATH_GPT_h_oplus_h_op_h_equals_h_l1268_126882

def op (x y : ℝ) : ℝ := x^3 - y

theorem h_oplus_h_op_h_equals_h (h : ℝ) : op h (op h h) = h := by
  sorry

end NUMINAMATH_GPT_h_oplus_h_op_h_equals_h_l1268_126882


namespace NUMINAMATH_GPT_stream_speed_is_2_l1268_126856

variable (v : ℝ) -- Let v be the speed of the stream in km/h

-- Condition 1: Man's swimming speed in still water
def swimming_speed_still : ℝ := 6

-- Condition 2: It takes him twice as long to swim upstream as downstream
def condition : Prop := (swimming_speed_still + v) / (swimming_speed_still - v) = 2

theorem stream_speed_is_2 : condition v → v = 2 := by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_stream_speed_is_2_l1268_126856


namespace NUMINAMATH_GPT_units_digit_base8_l1268_126863

theorem units_digit_base8 (a b : ℕ) (h_a : a = 505) (h_b : b = 71) : 
  ((a * b) % 8) = 7 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_base8_l1268_126863


namespace NUMINAMATH_GPT_electronic_items_stock_l1268_126815

-- Define the base statements
def all_in_stock (S : Type) (p : S → Prop) : Prop := ∀ x, p x
def some_not_in_stock (S : Type) (p : S → Prop) : Prop := ∃ x, ¬ p x

-- Define the main theorem statement
theorem electronic_items_stock (S : Type) (p : S → Prop) :
  ¬ all_in_stock S p → some_not_in_stock S p :=
by
  intros
  sorry

end NUMINAMATH_GPT_electronic_items_stock_l1268_126815


namespace NUMINAMATH_GPT_prime_bounds_l1268_126816

noncomputable def is_prime (p : ℕ) : Prop := 2 ≤ p ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem prime_bounds (n : ℕ) (h1 : 2 ≤ n) 
  (h2 : ∀ k, 0 ≤ k → k ≤ Nat.sqrt (n / 3) → is_prime (k^2 + k + n)) : 
  ∀ k, 0 ≤ k → k ≤ n - 2 → is_prime (k^2 + k + n) :=
by
  sorry

end NUMINAMATH_GPT_prime_bounds_l1268_126816


namespace NUMINAMATH_GPT_age_difference_l1268_126855

theorem age_difference 
  (a b : ℕ) 
  (h1 : 0 ≤ a ∧ a < 10) 
  (h2 : 0 ≤ b ∧ b < 10) 
  (h3 : 10 * a + b + 5 = 3 * (10 * b + a + 5)) : 
  (10 * a + b) - (10 * b + a) = 63 := 
by
  sorry

end NUMINAMATH_GPT_age_difference_l1268_126855


namespace NUMINAMATH_GPT_picnic_problem_l1268_126852

variables (M W C A : ℕ)

theorem picnic_problem
  (H1 : M + W + C = 200)
  (H2 : A = C + 20)
  (H3 : M = 65)
  (H4 : A = M + W) :
  M - W = 20 :=
by sorry

end NUMINAMATH_GPT_picnic_problem_l1268_126852


namespace NUMINAMATH_GPT_inequality_solution_set_l1268_126840

theorem inequality_solution_set :
  {x : ℝ | (3 - x) * (1 + x) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1268_126840


namespace NUMINAMATH_GPT_combined_stickers_l1268_126820

def initial_stickers_june : ℕ := 76
def initial_stickers_bonnie : ℕ := 63
def birthday_stickers : ℕ := 25

theorem combined_stickers : 
  (initial_stickers_june + birthday_stickers) + (initial_stickers_bonnie + birthday_stickers) = 189 := 
by
  sorry

end NUMINAMATH_GPT_combined_stickers_l1268_126820


namespace NUMINAMATH_GPT_lcm_5_6_8_9_l1268_126817

theorem lcm_5_6_8_9 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := by
  sorry

end NUMINAMATH_GPT_lcm_5_6_8_9_l1268_126817


namespace NUMINAMATH_GPT_abs_inequality_solution_l1268_126824

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1268_126824


namespace NUMINAMATH_GPT_ratio_of_money_spent_l1268_126899

theorem ratio_of_money_spent (h : ∀(a b c : ℕ), a + b + c = 75) : 
  (25 / 75 = 1 / 3) ∧ 
  (40 / 75 = 4 / 3) ∧ 
  (10 / 75 = 2 / 15) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_money_spent_l1268_126899


namespace NUMINAMATH_GPT_bottle_and_beverage_weight_l1268_126807

theorem bottle_and_beverage_weight 
  (B : ℝ)  -- Weight of the bottle in kilograms
  (x : ℝ)  -- Original weight of the beverage in kilograms
  (h1 : B + 2 * x = 5)  -- Condition: double the beverage weight total
  (h2 : B + 4 * x = 9)  -- Condition: quadruple the beverage weight total
: x = 2 ∧ B = 1 := 
by
  sorry

end NUMINAMATH_GPT_bottle_and_beverage_weight_l1268_126807


namespace NUMINAMATH_GPT_quadratic_grid_fourth_column_l1268_126860

theorem quadratic_grid_fourth_column 
  (grid : ℕ → ℕ → ℝ)
  (row_quadratic : ∀ i : ℕ, (∃ a b c : ℝ, ∀ n : ℕ, grid i n = a * n^2 + b * n + c))
  (col_quadratic : ∀ j : ℕ, j ≤ 3 → (∃ a b c : ℝ, ∀ n : ℕ, grid n j = a * n^2 + b * n + c)) :
  ∃ a b c : ℝ, ∀ n : ℕ, grid n 4 = a * n^2 + b * n + c := 
sorry

end NUMINAMATH_GPT_quadratic_grid_fourth_column_l1268_126860


namespace NUMINAMATH_GPT_factorize_expression_l1268_126889

theorem factorize_expression (m n : ℤ) : 
  4 * m^2 * n - 4 * n^3 = 4 * n * (m + n) * (m - n) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1268_126889


namespace NUMINAMATH_GPT_remainder_12345678901_mod_101_l1268_126846

theorem remainder_12345678901_mod_101 : 12345678901 % 101 = 24 :=
by
  sorry

end NUMINAMATH_GPT_remainder_12345678901_mod_101_l1268_126846


namespace NUMINAMATH_GPT_Carol_optimal_choice_l1268_126839

noncomputable def Alice_choices := Set.Icc 0 (1 : ℝ)
noncomputable def Bob_choices := Set.Icc (1 / 3) (3 / 4 : ℝ)

theorem Carol_optimal_choice : 
  ∀ (c : ℝ), c ∈ Set.Icc 0 1 → 
  (∃! c, c = 7 / 12) := 
sorry

end NUMINAMATH_GPT_Carol_optimal_choice_l1268_126839


namespace NUMINAMATH_GPT_sugar_theft_problem_l1268_126868

-- Define the statements by Gercoginya and the Cook
def gercoginya_statement := "The cook did not steal the sugar"
def cook_statement := "The sugar was stolen by Gercoginya"

-- Define the thief and truth/lie conditions
def thief_lies (x: String) : Prop := x = "The cook stole the sugar"
def other_truth_or_lie (x y: String) : Prop := x = "The sugar was stolen by Gercoginya" ∨ x = "The sugar was not stolen by Gercoginya"

-- The main proof problem to be solved
theorem sugar_theft_problem : 
  ∃ thief : String, 
    (thief = "cook" ∧ thief_lies gercoginya_statement ∧ other_truth_or_lie cook_statement gercoginya_statement) ∨ 
    (thief = "gercoginya" ∧ thief_lies cook_statement ∧ other_truth_or_lie gercoginya_statement cook_statement) :=
sorry

end NUMINAMATH_GPT_sugar_theft_problem_l1268_126868


namespace NUMINAMATH_GPT_tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l1268_126879

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

/-- Problem 1 -/
theorem tangent_line_at_neg_ln_2 :
  let x := -Real.log 2
  let y := f x
  ∃ k b : ℝ, (y - b) = k * (x - (-Real.log 2)) ∧ k = (Real.exp x - 1) ∧ b = Real.log 2 + 1/2 :=
sorry

/-- Problem 2 -/
theorem range_of_a_inequality :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x > a * x) ↔ a ∈ Set.Iio (Real.exp 1 - 1) :=
sorry

/-- Problem 3 -/
theorem range_of_a_zero_point :
  ∀ a : ℝ, (∃! x : ℝ, f x - a * x = 0) ↔ a ∈ (Set.Iio (-1) ∪ Set.Ioi (Real.exp 1 - 1)) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l1268_126879


namespace NUMINAMATH_GPT_polygon_interior_angles_l1268_126822

theorem polygon_interior_angles {n : ℕ} (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_angles_l1268_126822


namespace NUMINAMATH_GPT_triangle_not_always_obtuse_l1268_126862

def is_acute_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90

theorem triangle_not_always_obtuse : ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ is_acute_triangle A B C :=
by
  -- Exact proof here.
  sorry

end NUMINAMATH_GPT_triangle_not_always_obtuse_l1268_126862


namespace NUMINAMATH_GPT_function_g_l1268_126875

theorem function_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ t, (20 * t - 14) = 2 * (g t) - 40) → (g t = 10 * t + 13) :=
by
  intro h
  have h1 : 20 * t - 14 = 2 * (g t) - 40 := h t
  sorry

end NUMINAMATH_GPT_function_g_l1268_126875


namespace NUMINAMATH_GPT_polynomial_value_l1268_126848

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by sorry

end NUMINAMATH_GPT_polynomial_value_l1268_126848


namespace NUMINAMATH_GPT_find_N_l1268_126823

theorem find_N (N p q : ℝ) 
  (h1 : N / p = 4) 
  (h2 : N / q = 18) 
  (h3 : p - q = 0.5833333333333334) :
  N = 3 := 
sorry

end NUMINAMATH_GPT_find_N_l1268_126823


namespace NUMINAMATH_GPT_completing_the_square_l1268_126842

theorem completing_the_square {x : ℝ} : x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := 
sorry

end NUMINAMATH_GPT_completing_the_square_l1268_126842


namespace NUMINAMATH_GPT_age_group_caloric_allowance_l1268_126877

theorem age_group_caloric_allowance
  (average_daily_allowance : ℕ)
  (daily_reduction : ℕ)
  (reduced_weekly_allowance : ℕ)
  (week_days : ℕ)
  (h1 : daily_reduction = 500)
  (h2 : week_days = 7)
  (h3 : reduced_weekly_allowance = 10500)
  (h4 : reduced_weekly_allowance = (average_daily_allowance - daily_reduction) * week_days) :
  average_daily_allowance = 2000 :=
sorry

end NUMINAMATH_GPT_age_group_caloric_allowance_l1268_126877


namespace NUMINAMATH_GPT_A_walking_speed_l1268_126821

-- Definition for the conditions
def A_speed (v : ℝ) : Prop := 
  ∃ (t : ℝ), 120 = 20 * (t - 6) ∧ 120 = v * t

-- The main theorem to prove the question
theorem A_walking_speed : ∀ (v : ℝ), A_speed v → v = 10 :=
by
  intros v h
  sorry

end NUMINAMATH_GPT_A_walking_speed_l1268_126821


namespace NUMINAMATH_GPT_negation_of_proposition_l1268_126887

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x + x^2 > 0) → (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1268_126887


namespace NUMINAMATH_GPT_terminating_decimals_count_l1268_126896

noncomputable def int_counts_terminating_decimals : ℕ :=
  let n_limit := 500
  let denominator := 2100
  Nat.floor (n_limit / 21)

theorem terminating_decimals_count :
  int_counts_terminating_decimals = 23 :=
by
  /- Proof will be here eventually -/
  sorry

end NUMINAMATH_GPT_terminating_decimals_count_l1268_126896


namespace NUMINAMATH_GPT_unique_acute_triangulation_l1268_126800

-- Definitions for the proof problem
def is_convex (polygon : Type) : Prop := sorry
def is_acute_triangle (triangle : Type) : Prop := sorry
def is_triangulation (polygon : Type) (triangulation : List Type) : Prop := sorry
def is_acute_triangulation (polygon : Type) (triangulation : List Type) : Prop :=
  is_triangulation polygon triangulation ∧ ∀ triangle ∈ triangulation, is_acute_triangle triangle

-- Proposition to be proved
theorem unique_acute_triangulation (n : ℕ) (polygon : Type) 
  (h₁ : is_convex polygon) (h₂ : n ≥ 3) :
  ∃! triangulation : List Type, is_acute_triangulation polygon triangulation := 
sorry

end NUMINAMATH_GPT_unique_acute_triangulation_l1268_126800


namespace NUMINAMATH_GPT_total_animals_hunted_l1268_126834

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_animals_hunted_l1268_126834


namespace NUMINAMATH_GPT_intersection_point_polar_coords_l1268_126870

open Real

def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

def curve_C2 (t x y : ℝ) : Prop :=
  (x = 2 - t) ∧ (y = t)

theorem intersection_point_polar_coords :
  ∃ (ρ θ : ℝ), (ρ = sqrt 2) ∧ (θ = π / 4) ∧
  ∃ (x y t : ℝ), curve_C2 t x y ∧ curve_C1 x y ∧
  (ρ = sqrt (x^2 + y^2)) ∧ (tan θ = y / x) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_polar_coords_l1268_126870


namespace NUMINAMATH_GPT_theo_eggs_needed_l1268_126818

def customers_first_hour : ℕ := 5
def customers_second_hour : ℕ := 7
def customers_third_hour : ℕ := 3
def customers_fourth_hour : ℕ := 8
def eggs_per_3_egg_omelette : ℕ := 3
def eggs_per_4_egg_omelette : ℕ := 4

theorem theo_eggs_needed :
  (customers_first_hour * eggs_per_3_egg_omelette) +
  (customers_second_hour * eggs_per_4_egg_omelette) +
  (customers_third_hour * eggs_per_3_egg_omelette) +
  (customers_fourth_hour * eggs_per_4_egg_omelette) = 84 := by
  sorry

end NUMINAMATH_GPT_theo_eggs_needed_l1268_126818


namespace NUMINAMATH_GPT_turtle_population_2002_l1268_126885

theorem turtle_population_2002 (k : ℝ) (y : ℝ)
  (h1 : 58 + k * 92 = y)
  (h2 : 179 - 92 = k * y) 
  : y = 123 :=
by
  sorry

end NUMINAMATH_GPT_turtle_population_2002_l1268_126885
