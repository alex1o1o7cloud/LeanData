import Mathlib

namespace NUMINAMATH_GPT_greatest_unexpressible_sum_l147_14719

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem greatest_unexpressible_sum : 
  ∀ (n : ℕ), (∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n) → n ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_greatest_unexpressible_sum_l147_14719


namespace NUMINAMATH_GPT_length_of_train_l147_14742

-- Define the conditions
def bridge_length : ℕ := 200
def train_crossing_time : ℕ := 60
def train_speed : ℕ := 5

-- Define the total distance traveled by the train while crossing the bridge
def total_distance : ℕ := train_speed * train_crossing_time

-- The problem is to show the length of the train
theorem length_of_train :
  total_distance - bridge_length = 100 :=
by sorry

end NUMINAMATH_GPT_length_of_train_l147_14742


namespace NUMINAMATH_GPT_geometric_series_sum_l147_14776

def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  sum_geometric_series (1/4) (1/4) 7 = 4/3 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l147_14776


namespace NUMINAMATH_GPT_terry_age_proof_l147_14749

theorem terry_age_proof
  (nora_age : ℕ)
  (h1 : nora_age = 10)
  (terry_age_in_10_years : ℕ)
  (h2 : terry_age_in_10_years = 4 * nora_age)
  (nora_age_in_5_years : ℕ)
  (h3 : nora_age_in_5_years = nora_age + 5)
  (sam_age_in_5_years : ℕ)
  (h4 : sam_age_in_5_years = 2 * nora_age_in_5_years)
  (sam_current_age : ℕ)
  (h5 : sam_current_age = sam_age_in_5_years - 5)
  (terry_current_age : ℕ)
  (h6 : sam_current_age = terry_current_age + 6) :
  terry_current_age = 19 :=
by
  sorry

end NUMINAMATH_GPT_terry_age_proof_l147_14749


namespace NUMINAMATH_GPT_find_prime_pairs_l147_14755

def is_solution_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p ∣ 5^q + 1) ∧ (q ∣ 5^p + 1)

theorem find_prime_pairs :
  {pq : ℕ × ℕ | is_solution_pair pq.1 pq.2} =
  { (2, 13), (13, 2), (3, 7), (7, 3) } :=
by
  sorry

end NUMINAMATH_GPT_find_prime_pairs_l147_14755


namespace NUMINAMATH_GPT_find_A_l147_14710

-- Define the polynomial and the partial fraction decomposition equation
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_A (A B C : ℝ) (h : ∀ x : ℝ, 1 / polynomial x = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) : 
  A = 1 / 16 :=
sorry

end NUMINAMATH_GPT_find_A_l147_14710


namespace NUMINAMATH_GPT_problem_solution_l147_14785

theorem problem_solution
  (N1 N2 : ℤ)
  (h : ∀ x : ℝ, 50 * x - 42 ≠ 0 → x ≠ 2 → x ≠ 3 → 
    (50 * x - 42) / (x ^ 2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) : 
  N1 * N2 = -6264 :=
sorry

end NUMINAMATH_GPT_problem_solution_l147_14785


namespace NUMINAMATH_GPT_quadratic_root_ratio_l147_14700

theorem quadratic_root_ratio {m p q : ℝ} (h₁ : m ≠ 0) (h₂ : p ≠ 0) (h₃ : q ≠ 0)
  (h₄ : ∀ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) →
    (∃ t₁ t₂ : ℝ, t₁ = 3 * s₁ ∧ t₂ = 3 * s₂ ∧ (t₁ + t₂ = -m ∧ t₁ * t₂ = p))) :
  p / q = 27 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_ratio_l147_14700


namespace NUMINAMATH_GPT_distribution_of_K_l147_14765

theorem distribution_of_K (x y z : ℕ) 
  (h_total : x + y + z = 370)
  (h_diff : y + z - x = 50)
  (h_prop : x * z = y^2) :
  x = 160 ∧ y = 120 ∧ z = 90 := by
  sorry

end NUMINAMATH_GPT_distribution_of_K_l147_14765


namespace NUMINAMATH_GPT_pow_neg_cubed_squared_l147_14756

variable (a : ℝ)

theorem pow_neg_cubed_squared : 
  (-a^3)^2 = a^6 := 
by 
  sorry

end NUMINAMATH_GPT_pow_neg_cubed_squared_l147_14756


namespace NUMINAMATH_GPT_find_minimum_value_2a_plus_b_l147_14726

theorem find_minimum_value_2a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_re_z : (3 * a * b + 2) = 4) : 2 * a + b = (4 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_find_minimum_value_2a_plus_b_l147_14726


namespace NUMINAMATH_GPT_trigonometric_expression_simplification_l147_14747

theorem trigonometric_expression_simplification (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := 
sorry

end NUMINAMATH_GPT_trigonometric_expression_simplification_l147_14747


namespace NUMINAMATH_GPT_cost_of_3000_pencils_l147_14716

theorem cost_of_3000_pencils (pencils_per_box : ℕ) (cost_per_box : ℝ) (pencils_needed : ℕ) (unit_cost : ℝ): 
  pencils_per_box = 120 → cost_per_box = 36 → pencils_needed = 3000 → unit_cost = 0.30 →
  (pencils_needed * unit_cost = (3000 : ℝ) * 0.30) :=
by
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_cost_of_3000_pencils_l147_14716


namespace NUMINAMATH_GPT_range_of_real_number_l147_14761

theorem range_of_real_number (a : ℝ) : (a > 0) ∧ (a - 1 > 0) → a > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_real_number_l147_14761


namespace NUMINAMATH_GPT_A_div_B_l147_14729

noncomputable def A : ℝ := 
  ∑' n, if n % 2 = 0 ∧ n % 4 ≠ 0 then 1 / (n:ℝ)^2 else 0

noncomputable def B : ℝ := 
  ∑' n, if n % 4 = 0 then (-1)^(n / 4 + 1) * 1 / (n:ℝ)^2 else 0

theorem A_div_B : A / B = 17 := by
  sorry

end NUMINAMATH_GPT_A_div_B_l147_14729


namespace NUMINAMATH_GPT_units_digit_of_2_pow_20_minus_1_l147_14797

theorem units_digit_of_2_pow_20_minus_1 : (2^20 - 1) % 10 = 5 := 
  sorry

end NUMINAMATH_GPT_units_digit_of_2_pow_20_minus_1_l147_14797


namespace NUMINAMATH_GPT_train_length_is_correct_l147_14784

noncomputable def length_of_train (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_m_s := train_speed * (1000 / 3600)
  let total_distance := speed_m_s * time_to_cross
  total_distance - bridge_length

theorem train_length_is_correct :
  length_of_train 36 24.198064154867613 132 = 109.98064154867613 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l147_14784


namespace NUMINAMATH_GPT_complement_union_eq_l147_14783

universe u

-- Definitions based on conditions in a)
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

-- The goal to prove based on c)
theorem complement_union_eq :
  (U \ (M ∪ N)) = {5, 6} := 
by sorry

end NUMINAMATH_GPT_complement_union_eq_l147_14783


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l147_14739

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l147_14739


namespace NUMINAMATH_GPT_BANANA_arrangements_l147_14718

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end NUMINAMATH_GPT_BANANA_arrangements_l147_14718


namespace NUMINAMATH_GPT_trig_identity_l147_14706

noncomputable def sin_deg (x : ℝ) := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) := Real.cos (x * Real.pi / 180)
noncomputable def tan_deg (x : ℝ) := Real.tan (x * Real.pi / 180)

theorem trig_identity :
  (2 * sin_deg 50 + sin_deg 10 * (1 + Real.sqrt 3 * tan_deg 10) * Real.sqrt 2 * (sin_deg 80)^2) = Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l147_14706


namespace NUMINAMATH_GPT_martha_found_blocks_l147_14796

variable (initial_blocks final_blocks found_blocks : ℕ)

theorem martha_found_blocks 
    (h_initial : initial_blocks = 4) 
    (h_final : final_blocks = 84) 
    (h_found : found_blocks = final_blocks - initial_blocks) : 
    found_blocks = 80 := by
  sorry

end NUMINAMATH_GPT_martha_found_blocks_l147_14796


namespace NUMINAMATH_GPT_largest_consecutive_sum_to_35_l147_14758

theorem largest_consecutive_sum_to_35 (n : ℕ) (h : ∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 35) : n ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_consecutive_sum_to_35_l147_14758


namespace NUMINAMATH_GPT_blisters_on_rest_of_body_l147_14728

theorem blisters_on_rest_of_body (blisters_per_arm total_blisters : ℕ) (h1 : blisters_per_arm = 60) (h2 : total_blisters = 200) : 
  total_blisters - 2 * blisters_per_arm = 80 :=
by {
  -- The proof can be written here
  sorry
}

end NUMINAMATH_GPT_blisters_on_rest_of_body_l147_14728


namespace NUMINAMATH_GPT_base7_digit_sum_l147_14767

theorem base7_digit_sum (A B C : ℕ) (hA : 1 ≤ A ∧ A < 7) (hB : 1 ≤ B ∧ B < 7) 
  (hC : 1 ≤ C ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eq : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) : 
  B + C = 6 := 
sorry

end NUMINAMATH_GPT_base7_digit_sum_l147_14767


namespace NUMINAMATH_GPT_proof_problem_l147_14768

-- Given conditions: 
variables (a b c d : ℝ)
axiom condition : (2 * a + b) / (b + 2 * c) = (c + 3 * d) / (4 * d + a)

-- Proof problem statement:
theorem proof_problem : (a = c ∨ 3 * a + 4 * b + 5 * c + 6 * d = 0 ∨ (a = c ∧ 3 * a + 4 * b + 5 * c + 6 * d = 0)) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l147_14768


namespace NUMINAMATH_GPT_dogs_in_kennel_l147_14757

theorem dogs_in_kennel (C D : ℕ) (h1 : C = D - 8) (h2 : C * 4 = 3 * D) : D = 32 :=
sorry

end NUMINAMATH_GPT_dogs_in_kennel_l147_14757


namespace NUMINAMATH_GPT_alcohol_percentage_solution_x_l147_14723

theorem alcohol_percentage_solution_x :
  ∃ (P : ℝ), 
  (∀ (vol_x vol_y : ℝ), vol_x = 50 → vol_y = 150 →
    ∀ (percent_y percent_new : ℝ), percent_y = 30 → percent_new = 25 →
      ((P / 100) * vol_x + (percent_y / 100) * vol_y) / (vol_x + vol_y) = percent_new) → P = 10 :=
by
  -- Given conditions
  let vol_x := 50
  let vol_y := 150
  let percent_y := 30
  let percent_new := 25

  -- The proof body should be here
  sorry

end NUMINAMATH_GPT_alcohol_percentage_solution_x_l147_14723


namespace NUMINAMATH_GPT_eight_b_value_l147_14745

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 8 * b = 16 :=
by
  sorry

end NUMINAMATH_GPT_eight_b_value_l147_14745


namespace NUMINAMATH_GPT_max_value_expr_l147_14775

theorem max_value_expr (a b c d : ℝ) (ha : -12.5 ≤ a ∧ a ≤ 12.5) (hb : -12.5 ≤ b ∧ b ≤ 12.5) (hc : -12.5 ≤ c ∧ c ≤ 12.5) (hd : -12.5 ≤ d ∧ d ≤ 12.5) :
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 650 :=
sorry

end NUMINAMATH_GPT_max_value_expr_l147_14775


namespace NUMINAMATH_GPT_stream_speed_zero_l147_14792

theorem stream_speed_zero (v_c v_s : ℝ)
  (h1 : v_c - v_s - 2 = 9)
  (h2 : v_c + v_s + 1 = 12) :
  v_s = 0 := 
sorry

end NUMINAMATH_GPT_stream_speed_zero_l147_14792


namespace NUMINAMATH_GPT_equal_savings_l147_14702

theorem equal_savings (U B UE BE US BS : ℕ) (h1 : U / B = 8 / 7) 
                      (h2 : U = 16000) (h3 : UE / BE = 7 / 6) (h4 : US = BS) :
                      US = 2000 ∧ BS = 2000 :=
by
  sorry

end NUMINAMATH_GPT_equal_savings_l147_14702


namespace NUMINAMATH_GPT_min_value_of_f_l147_14734

noncomputable def f (x : ℝ) : ℝ := 2 + 3 * x + 4 / (x - 1)

theorem min_value_of_f :
  (∀ x : ℝ, x > 1 → f x ≥ (5 + 4 * Real.sqrt 3)) ∧
  (f (1 + 2 * Real.sqrt 3 / 3) = 5 + 4 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l147_14734


namespace NUMINAMATH_GPT_average_speed_of_train_l147_14772

theorem average_speed_of_train (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 125) (h2 : d2 = 270) (h3 : t1 = 2.5) (h4 : t2 = 3) :
  (d1 + d2) / (t1 + t2) = 71.82 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l147_14772


namespace NUMINAMATH_GPT_min_value_frac_l147_14714

theorem min_value_frac (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ (∀ y, 0 < y ∧ y < 1 → (a * a / y + b * b / (1 - y)) ≥ (a + b) * (a + b)) ∧ 
       a * a / x + b * b / (1 - x) = (a + b) * (a + b) := 
by {
  sorry
}

end NUMINAMATH_GPT_min_value_frac_l147_14714


namespace NUMINAMATH_GPT_value_of_m_making_365m_divisible_by_12_l147_14732

theorem value_of_m_making_365m_divisible_by_12
  (m : ℕ)
  (h1 : (3650 + m) % 3 = 0)
  (h2 : (50 + m) % 4 = 0) :
  m = 0 :=
sorry

end NUMINAMATH_GPT_value_of_m_making_365m_divisible_by_12_l147_14732


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l147_14743

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l147_14743


namespace NUMINAMATH_GPT_arithmetic_mean_odd_primes_lt_30_l147_14753

theorem arithmetic_mean_odd_primes_lt_30 : 
  (3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29) / 9 = 14 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_odd_primes_lt_30_l147_14753


namespace NUMINAMATH_GPT_factorization_m_minus_n_l147_14703

theorem factorization_m_minus_n :
  ∃ (m n : ℤ), (6 * (x:ℝ)^2 - 5 * x - 6 = (6 * x + m) * (x + n)) ∧ (m - n = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_factorization_m_minus_n_l147_14703


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_correct_l147_14764

/-- Convert repeating decimals to fractions -/
def rep_dec_1 : ℚ := 1 / 9
def rep_dec_2 : ℚ := 2 / 9
def rep_dec_3 : ℚ := 1 / 3
def rep_dec_4 : ℚ := 4 / 9
def rep_dec_5 : ℚ := 5 / 9
def rep_dec_6 : ℚ := 2 / 3
def rep_dec_7 : ℚ := 7 / 9
def rep_dec_8 : ℚ := 8 / 9

/-- Define the terms in the sum -/
def term_1 : ℚ := 8 + rep_dec_1
def term_2 : ℚ := 7 + 1 + rep_dec_2
def term_3 : ℚ := 6 + 2 + rep_dec_3
def term_4 : ℚ := 5 + 3 + rep_dec_4
def term_5 : ℚ := 4 + 4 + rep_dec_5
def term_6 : ℚ := 3 + 5 + rep_dec_6
def term_7 : ℚ := 2 + 6 + rep_dec_7
def term_8 : ℚ := 1 + 7 + rep_dec_8

/-- Define the sum of the terms -/
def total_sum : ℚ := term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

/-- Proof problem statement -/
theorem sum_of_repeating_decimals_correct : total_sum = 39.2 := 
sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_correct_l147_14764


namespace NUMINAMATH_GPT_peytons_children_l147_14707

theorem peytons_children (C : ℕ) (juice_per_week : ℕ) (weeks_in_school_year : ℕ) (total_juice_boxes : ℕ) 
  (h1 : juice_per_week = 5) 
  (h2 : weeks_in_school_year = 25) 
  (h3 : total_juice_boxes = 375)
  (h4 : C * (juice_per_week * weeks_in_school_year) = total_juice_boxes) 
  : C = 3 :=
sorry

end NUMINAMATH_GPT_peytons_children_l147_14707


namespace NUMINAMATH_GPT_quadratic_roots_identity_l147_14717

theorem quadratic_roots_identity (α β : ℝ) (hαβ : α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0) : 
  α^2 + α*β - 3*α = 0 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_identity_l147_14717


namespace NUMINAMATH_GPT_sheep_ratio_l147_14750

theorem sheep_ratio (S : ℕ) (h1 : 400 - S = 2 * 150) :
  S / 400 = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sheep_ratio_l147_14750


namespace NUMINAMATH_GPT_negation_of_p_l147_14759

variable (f : ℝ → ℝ)

theorem negation_of_p :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔ (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l147_14759


namespace NUMINAMATH_GPT_roots_squared_sum_l147_14725

theorem roots_squared_sum {x y : ℝ} (hx : 3 * x^2 - 7 * x + 5 = 0) (hy : 3 * y^2 - 7 * y + 5 = 0) (hxy : x ≠ y) :
  x^2 + y^2 = 19 / 9 :=
sorry

end NUMINAMATH_GPT_roots_squared_sum_l147_14725


namespace NUMINAMATH_GPT_larger_number_is_30_l147_14711

-- Formalizing the conditions
variables (x y : ℝ)

-- Define the conditions given in the problem
def sum_condition : Prop := x + y = 40
def ratio_condition : Prop := x / y = 3

-- Formalize the problem statement
theorem larger_number_is_30 (h1 : sum_condition x y) (h2 : ratio_condition x y) : x = 30 :=
sorry

end NUMINAMATH_GPT_larger_number_is_30_l147_14711


namespace NUMINAMATH_GPT_calculate_tax_l147_14762

noncomputable def cadastral_value : ℝ := 3000000 -- 3 million rubles
noncomputable def tax_rate : ℝ := 0.001        -- 0.1% converted to decimal
noncomputable def tax : ℝ := cadastral_value * tax_rate -- Tax formula

theorem calculate_tax : tax = 3000 := by
  sorry

end NUMINAMATH_GPT_calculate_tax_l147_14762


namespace NUMINAMATH_GPT_direction_vector_b_l147_14763

theorem direction_vector_b (b : ℝ) 
  (P Q : ℝ × ℝ) (hP : P = (-3, 1)) (hQ : Q = (1, 5))
  (hdir : 3 - (-3) = 3 ∧ 5 - 1 = b) : b = 3 := by
  sorry

end NUMINAMATH_GPT_direction_vector_b_l147_14763


namespace NUMINAMATH_GPT_quotient_change_l147_14705

variables {a b : ℝ} (h : a / b = 0.78)

theorem quotient_change (a b : ℝ) (h : a / b = 0.78) : (10 * a) / (b / 10) = 78 :=
by
  sorry

end NUMINAMATH_GPT_quotient_change_l147_14705


namespace NUMINAMATH_GPT_Alex_sandwich_count_l147_14788

theorem Alex_sandwich_count :
  let meats := 10
  let cheeses := 9
  let sandwiches := meats * (cheeses.choose 2)
  sandwiches = 360 :=
by
  -- Here start your proof
  sorry

end NUMINAMATH_GPT_Alex_sandwich_count_l147_14788


namespace NUMINAMATH_GPT_part1_part2_part3_l147_14744

variable (a b c : ℝ) (f : ℝ → ℝ)
-- Defining the polynomial function f
def polynomial (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem part1 (h0 : polynomial a b 6 0 = 6) : c = 6 :=
by sorry

theorem part2 (h1 : polynomial a b (-2) 0 = -2) (h2 : polynomial a b (-2) 1 = 5) : polynomial a b (-2) (-1) = -9 :=
by sorry

theorem part3 (h3 : polynomial a b 3 5 + polynomial a b 3 (-5) = 6) (h4 : polynomial a b 3 2 = 8) : polynomial a b 3 (-2) = -2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l147_14744


namespace NUMINAMATH_GPT_ship_speeds_l147_14708

theorem ship_speeds (x : ℝ) 
  (h1 : (2 * x) ^ 2 + (2 * (x + 3)) ^ 2 = 174 ^ 2) :
  x = 60 ∧ x + 3 = 63 :=
by
  sorry

end NUMINAMATH_GPT_ship_speeds_l147_14708


namespace NUMINAMATH_GPT_equation_solution_l147_14798

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end NUMINAMATH_GPT_equation_solution_l147_14798


namespace NUMINAMATH_GPT_base9_to_decimal_unique_solution_l147_14791

theorem base9_to_decimal_unique_solution :
  ∃ m : ℕ, 1 * 9^4 + 6 * 9^3 + m * 9^2 + 2 * 9^1 + 7 = 11203 ∧ m = 3 :=
by
  sorry

end NUMINAMATH_GPT_base9_to_decimal_unique_solution_l147_14791


namespace NUMINAMATH_GPT_timesToFillBottlePerWeek_l147_14724

noncomputable def waterConsumptionPerDay : ℕ := 4 * 5
noncomputable def waterConsumptionPerWeek : ℕ := 7 * waterConsumptionPerDay
noncomputable def bottleCapacity : ℕ := 35

theorem timesToFillBottlePerWeek : 
  waterConsumptionPerWeek / bottleCapacity = 4 := 
by
  sorry

end NUMINAMATH_GPT_timesToFillBottlePerWeek_l147_14724


namespace NUMINAMATH_GPT_inversely_proportional_percentage_change_l147_14722

variable {x y k : ℝ}
variable (a b : ℝ)

/-- Given that x and y are positive numbers and inversely proportional,
if x increases by a% and y decreases by b%, then b = 100a / (100 + a) -/
theorem inversely_proportional_percentage_change
  (hx : 0 < x) (hy : 0 < y) (hinv : y = k / x)
  (ha : 0 < a) (hb : 0 < b)
  (hchange : ((1 + a / 100) * x) * ((1 - b / 100) * y) = k) :
  b = 100 * a / (100 + a) :=
sorry

end NUMINAMATH_GPT_inversely_proportional_percentage_change_l147_14722


namespace NUMINAMATH_GPT_train_passes_platform_in_39_2_seconds_l147_14782

def length_of_train : ℝ := 360
def speed_in_kmh : ℝ := 45
def length_of_platform : ℝ := 130

noncomputable def speed_in_mps : ℝ := speed_in_kmh * 1000 / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_in_mps

theorem train_passes_platform_in_39_2_seconds :
  time_to_pass_platform = 39.2 := by
  sorry

end NUMINAMATH_GPT_train_passes_platform_in_39_2_seconds_l147_14782


namespace NUMINAMATH_GPT_positive_integer_satisfies_condition_l147_14754

def num_satisfying_pos_integers : ℕ :=
  1

theorem positive_integer_satisfies_condition :
  ∃ (n : ℕ), 16 - 4 * n > 10 ∧ n = num_satisfying_pos_integers := by
  sorry

end NUMINAMATH_GPT_positive_integer_satisfies_condition_l147_14754


namespace NUMINAMATH_GPT_cricketer_total_matches_l147_14778

theorem cricketer_total_matches (n : ℕ)
  (avg_total : ℝ) (avg_first_6 : ℝ) (avg_last_4 : ℝ)
  (total_runs_eq : 6 * avg_first_6 + 4 * avg_last_4 = n * avg_total) :
  avg_total = 38.9 ∧ avg_first_6 = 42 ∧ avg_last_4 = 34.25 → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_cricketer_total_matches_l147_14778


namespace NUMINAMATH_GPT_decagon_adjacent_probability_l147_14712

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end NUMINAMATH_GPT_decagon_adjacent_probability_l147_14712


namespace NUMINAMATH_GPT_radius_of_O2016_l147_14786

-- Define the centers and radii of circles
variable (a : ℝ) (n : ℕ) (r : ℕ → ℝ)

-- Conditions
-- Radius of the first circle
def initial_radius := r 1 = 1 / (2 * a)
-- Sequence of the radius difference based on solution step
def radius_recursive := ∀ n > 1, r (n + 1) - r n = 1 / a

-- The final statement to be proven
theorem radius_of_O2016 (h1 : initial_radius a r) (h2 : radius_recursive a r) :
  r 2016 = 4031 / (2 * a) := 
by sorry

end NUMINAMATH_GPT_radius_of_O2016_l147_14786


namespace NUMINAMATH_GPT_partA_l147_14794

theorem partA (a b : ℝ) : (a - b) ^ 2 ≥ 0 → (a^2 + b^2) / 2 ≥ a * b := 
by
  intro h
  sorry

end NUMINAMATH_GPT_partA_l147_14794


namespace NUMINAMATH_GPT_tangent_parabola_line_l147_14701

theorem tangent_parabola_line (a : ℝ) :
  (∃ x : ℝ, ax^2 + 1 = x ∧ ∀ y : ℝ, (y = ax^2 + 1 → y = x)) ↔ a = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_parabola_line_l147_14701


namespace NUMINAMATH_GPT_determine_base_l147_14766

theorem determine_base (r : ℕ) (a b x : ℕ) (h₁ : r ≤ 100) 
  (h₂ : x = a * r + a) (h₃ : a < r) (h₄ : a > 0) 
  (h₅ : x^2 = b * r^3 + b) : r = 2 ∨ r = 23 :=
by
  sorry

end NUMINAMATH_GPT_determine_base_l147_14766


namespace NUMINAMATH_GPT_isosceles_right_triangle_inscribed_circle_l147_14799

theorem isosceles_right_triangle_inscribed_circle
  (h r x : ℝ)
  (h_def : h = 2 * r)
  (r_def : r = Real.sqrt 2 / 4)
  (x_def : x = h - r) :
  x = Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_inscribed_circle_l147_14799


namespace NUMINAMATH_GPT_last_colored_cell_is_51_50_l147_14704

def last_spiral_cell (width height : ℕ) : ℕ × ℕ :=
  -- Assuming an external or pre-defined process to calculate the last cell for a spiral pattern
  sorry 

theorem last_colored_cell_is_51_50 :
  last_spiral_cell 200 100 = (51, 50) :=
sorry

end NUMINAMATH_GPT_last_colored_cell_is_51_50_l147_14704


namespace NUMINAMATH_GPT_selling_price_correct_l147_14751

-- Define the parameters
def stamp_duty_rate : ℝ := 0.002
def commission_rate : ℝ := 0.0035
def bought_shares : ℝ := 3000
def buying_price_per_share : ℝ := 12
def profit : ℝ := 5967

-- Define the selling price per share
noncomputable def selling_price_per_share (x : ℝ) : ℝ :=
  bought_shares * x - bought_shares * buying_price_per_share -
  bought_shares * x * (stamp_duty_rate + commission_rate) - 
  bought_shares * buying_price_per_share * (stamp_duty_rate + commission_rate)

-- The target selling price per share
def target_selling_price_per_share : ℝ := 14.14

-- Statement of the problem
theorem selling_price_correct (x : ℝ) : selling_price_per_share x = profit → x = target_selling_price_per_share := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l147_14751


namespace NUMINAMATH_GPT_total_oil_leaked_correct_l147_14735

-- Definitions of given conditions.
def initial_leak_A : ℕ := 6522
def leak_rate_A : ℕ := 257
def time_A : ℕ := 20

def initial_leak_B : ℕ := 3894
def leak_rate_B : ℕ := 182
def time_B : ℕ := 15

def initial_leak_C : ℕ := 1421
def leak_rate_C : ℕ := 97
def time_C : ℕ := 12

-- Total additional leaks calculation.
def additional_leak (rate time : ℕ) : ℕ := rate * time
def additional_leak_A : ℕ := additional_leak leak_rate_A time_A
def additional_leak_B : ℕ := additional_leak leak_rate_B time_B
def additional_leak_C : ℕ := additional_leak leak_rate_C time_C

-- Total leaks from each pipe.
def total_leak_A : ℕ := initial_leak_A + additional_leak_A
def total_leak_B : ℕ := initial_leak_B + additional_leak_B
def total_leak_C : ℕ := initial_leak_C + additional_leak_C

-- Total oil leaked.
def total_oil_leaked : ℕ := total_leak_A + total_leak_B + total_leak_C

-- The proof problem statement.
theorem total_oil_leaked_correct : total_oil_leaked = 20871 := by
  sorry

end NUMINAMATH_GPT_total_oil_leaked_correct_l147_14735


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l147_14737

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Define the equations for the asymptotes
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

-- State the theorem
theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola_eq x y → (asymptote_pos x y ∨ asymptote_neg x y) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l147_14737


namespace NUMINAMATH_GPT_multiplication_problem_division_problem_l147_14746

theorem multiplication_problem :
  125 * 76 * 4 * 8 * 25 = 7600000 :=
sorry

theorem division_problem :
  (6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741 :=
sorry

end NUMINAMATH_GPT_multiplication_problem_division_problem_l147_14746


namespace NUMINAMATH_GPT_relationship_among_abc_l147_14720

noncomputable def a : ℝ := 36^(1/5)
noncomputable def b : ℝ := 3^(4/3)
noncomputable def c : ℝ := 9^(2/5)

theorem relationship_among_abc (a_def : a = 36^(1/5)) 
                              (b_def : b = 3^(4/3)) 
                              (c_def : c = 9^(2/5)) : a < c ∧ c < b :=
by
  rw [a_def, b_def, c_def]
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l147_14720


namespace NUMINAMATH_GPT_problem1_l147_14771

theorem problem1 : 2 * (-5) + 2^3 - 3 + (1/2 : ℚ) = -15 / 2 := 
by
  sorry

end NUMINAMATH_GPT_problem1_l147_14771


namespace NUMINAMATH_GPT_expand_expression_l147_14760

variable {R : Type*} [CommRing R]
variable (x y : R)

theorem expand_expression : 
  ((10 * x - 6 * y + 9) * 3 * y) = (30 * x * y - 18 * y * y + 27 * y) :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l147_14760


namespace NUMINAMATH_GPT_no_real_x_condition_l147_14727

theorem no_real_x_condition (x : ℝ) : 
(∃ a b : ℕ, 4 * x^5 - 7 = a^2 ∧ 4 * x^13 - 7 = b^2) → false := 
by {
  sorry
}

end NUMINAMATH_GPT_no_real_x_condition_l147_14727


namespace NUMINAMATH_GPT_petya_vasya_cubic_roots_diff_2014_l147_14733

theorem petya_vasya_cubic_roots_diff_2014 :
  ∀ (p q r : ℚ), ∃ (x1 x2 x3 : ℚ), x1 ≠ 0 ∧ (x1 - x2 = 2014 ∨ x1 - x3 = 2014 ∨ x2 - x3 = 2014) :=
sorry

end NUMINAMATH_GPT_petya_vasya_cubic_roots_diff_2014_l147_14733


namespace NUMINAMATH_GPT_triangle_solution_l147_14730

noncomputable def solve_triangle (a : ℝ) (α : ℝ) (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let s := 75
  let b := 41
  let c := 58
  let β := 43 + 36 / 60 + 10 / 3600
  let γ := 77 + 19 / 60 + 11 / 3600
  ((b, c), (β, γ))

theorem triangle_solution :
  solve_triangle 51 (59 + 4 / 60 + 39 / 3600) 1020 = ((41, 58), (43 + 36 / 60 + 10 / 3600, 77 + 19 / 60 + 11 / 3600)) :=
sorry  

end NUMINAMATH_GPT_triangle_solution_l147_14730


namespace NUMINAMATH_GPT_find_other_parallel_side_l147_14709

variable (a b h : ℝ) (Area : ℝ)

-- Conditions
axiom h_pos : h = 13
axiom a_val : a = 18
axiom area_val : Area = 247
axiom area_formula : Area = (1 / 2) * (a + b) * h

-- Theorem (to be proved by someone else)
theorem find_other_parallel_side (a b h : ℝ) 
  (h_pos : h = 13) 
  (a_val : a = 18) 
  (area_val : Area = 247) 
  (area_formula : Area = (1 / 2) * (a + b) * h) : 
  b = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_other_parallel_side_l147_14709


namespace NUMINAMATH_GPT_stationery_shop_costs_l147_14741

theorem stationery_shop_costs (p n : ℝ) 
  (h1 : 9 * p + 6 * n = 3.21)
  (h2 : 8 * p + 5 * n = 2.84) :
  12 * p + 9 * n = 4.32 :=
sorry

end NUMINAMATH_GPT_stationery_shop_costs_l147_14741


namespace NUMINAMATH_GPT_barry_should_pay_l147_14790

def original_price : ℝ := 80
def discount_rate : ℝ := 0.15

theorem barry_should_pay:
  original_price * (1 - discount_rate) = 68 := 
by 
  -- Original price: 80
  -- Discount rate: 0.15
  -- Question: Final price after discount
  sorry

end NUMINAMATH_GPT_barry_should_pay_l147_14790


namespace NUMINAMATH_GPT_product_consecutive_two_digits_l147_14752

theorem product_consecutive_two_digits (a b c : ℕ) : 
  ¬(∃ n : ℕ, (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2)) :=
by
  sorry

end NUMINAMATH_GPT_product_consecutive_two_digits_l147_14752


namespace NUMINAMATH_GPT_intersection_of_sets_l147_14769

def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {1, 3, 5}

theorem intersection_of_sets : A ∩ B = {1, 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l147_14769


namespace NUMINAMATH_GPT_bobby_books_count_l147_14773

variable (KristiBooks BobbyBooks : ℕ)

theorem bobby_books_count (h1 : KristiBooks = 78) (h2 : BobbyBooks = KristiBooks + 64) : BobbyBooks = 142 :=
by
  sorry

end NUMINAMATH_GPT_bobby_books_count_l147_14773


namespace NUMINAMATH_GPT_multiple_of_pumpkins_l147_14781

theorem multiple_of_pumpkins (M S : ℕ) (hM : M = 14) (hS : S = 54) (h : S = x * M + 12) : x = 3 := sorry

end NUMINAMATH_GPT_multiple_of_pumpkins_l147_14781


namespace NUMINAMATH_GPT_unique_function_l147_14793

noncomputable def find_function (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 → b > 0 → a + b > 2019 → a + f b ∣ a^2 + b * f a

theorem unique_function (r : ℕ) (f : ℕ → ℕ) :
  find_function f → (∀ x : ℕ, f x = r * x) :=
sorry

end NUMINAMATH_GPT_unique_function_l147_14793


namespace NUMINAMATH_GPT_c_alone_finishes_in_60_days_l147_14780

-- Definitions for rates of work
variables (A B C : ℝ)

-- The conditions given in the problem
-- A and B together can finish the job in 15 days
def condition1 : Prop := A + B = 1 / 15
-- A, B, and C together can finish the job in 12 days
def condition2 : Prop := A + B + C = 1 / 12

-- The statement to prove: C alone can finish the job in 60 days
theorem c_alone_finishes_in_60_days 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) : 
  (1 / C) = 60 :=
by
  sorry

end NUMINAMATH_GPT_c_alone_finishes_in_60_days_l147_14780


namespace NUMINAMATH_GPT_find_f_prime_one_l147_14777

noncomputable def f (f'_1 : ℝ) (x : ℝ) := f'_1 * x^3 - 2 * x^2 + 3

theorem find_f_prime_one (f'_1 : ℝ) 
  (h_derivative : ∀ x : ℝ, deriv (f f'_1) x = 3 * f'_1 * x^2 - 4 * x)
  (h_value_at_1 : deriv (f f'_1) 1 = f'_1) :
  f'_1 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_f_prime_one_l147_14777


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l147_14795

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin (2 * x)

theorem number_of_zeros_of_f : (∃ l : List ℝ, (∀ x ∈ l, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧ l.length = 4) := 
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l147_14795


namespace NUMINAMATH_GPT_compound_interest_semiannual_l147_14787

theorem compound_interest_semiannual
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (initial_amount : P = 900)
  (interest_rate : r = 0.10)
  (compounding_periods : n = 2)
  (time_period : t = 1) :
  P * (1 + r / n) ^ (n * t) = 992.25 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_semiannual_l147_14787


namespace NUMINAMATH_GPT_tabby_swimming_speed_l147_14770

theorem tabby_swimming_speed :
  ∃ (S : ℝ), S = 4.125 ∧ (∀ (D : ℝ), 6 = (2 * D) / ((D / S) + (D / 11))) :=
by {
 sorry
}

end NUMINAMATH_GPT_tabby_swimming_speed_l147_14770


namespace NUMINAMATH_GPT_sugar_water_inequality_triangle_inequality_l147_14748

-- Condition for question (1)
variable (x y m : ℝ)
variable (hx : x > 0) (hy : y > 0) (hxy : x > y) (hm : m > 0)

-- Proof problem for question (1)
theorem sugar_water_inequality : y / x < (y + m) / (x + m) :=
sorry

-- Condition for question (2)
variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (hab : b + c > a) (hac : a + c > b) (hbc : a + b > c)

-- Proof problem for question (2)
theorem triangle_inequality : 
  a / (b + c) + b / (a + c) + c / (a + b) < 2 :=
sorry

end NUMINAMATH_GPT_sugar_water_inequality_triangle_inequality_l147_14748


namespace NUMINAMATH_GPT_Lisa_flight_time_l147_14774

theorem Lisa_flight_time :
  let distance := 500
  let speed := 45
  (distance : ℝ) / (speed : ℝ) = 500 / 45 := by
  sorry

end NUMINAMATH_GPT_Lisa_flight_time_l147_14774


namespace NUMINAMATH_GPT_largest_circle_diameter_l147_14721

theorem largest_circle_diameter
  (A : ℝ) (hA : A = 180)
  (w l : ℝ) (hw : l = 3 * w)
  (hA2 : w * l = A) :
  ∃ d : ℝ, d = 16 * Real.sqrt 15 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_largest_circle_diameter_l147_14721


namespace NUMINAMATH_GPT_problems_per_page_is_eight_l147_14713

noncomputable def totalProblems := 60
noncomputable def finishedProblems := 20
noncomputable def totalPages := 5
noncomputable def problemsLeft := totalProblems - finishedProblems
noncomputable def problemsPerPage := problemsLeft / totalPages

theorem problems_per_page_is_eight :
  problemsPerPage = 8 :=
by
  sorry

end NUMINAMATH_GPT_problems_per_page_is_eight_l147_14713


namespace NUMINAMATH_GPT_radius_increase_50_percent_l147_14740

theorem radius_increase_50_percent 
  (r : ℝ)
  (h1 : 1.5 * r = r + r * 0.5) : 
  (3 * Real.pi * r = 2 * Real.pi * r + (2 * Real.pi * r * 0.5)) ∧
  (2.25 * Real.pi * r^2 = Real.pi * r^2 + (Real.pi * r^2 * 1.25)) := 
sorry

end NUMINAMATH_GPT_radius_increase_50_percent_l147_14740


namespace NUMINAMATH_GPT_total_notes_l147_14789

theorem total_notes (total_money : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) (fivehundred_value : ℕ) (fivehundred_notes : ℕ) :
  total_money = 10350 →
  fifty_notes = 57 →
  fifty_value = 50 →
  fivehundred_value = 500 →
  57 * 50 + fivehundred_notes * 500 = 10350 →
  fifty_notes + fivehundred_notes = 72 :=
by
  intros h_total_money h_fifty_notes h_fifty_value h_fivehundred_value h_equation
  sorry

end NUMINAMATH_GPT_total_notes_l147_14789


namespace NUMINAMATH_GPT_mnpq_product_l147_14738

noncomputable def prove_mnpq_product (a b x y : ℝ) : Prop :=
  ∃ (m n p q : ℤ), (a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4 ∧
                    m * n * p * q = 4

theorem mnpq_product (a b x y : ℝ) (h : a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) :
  prove_mnpq_product a b x y :=
sorry

end NUMINAMATH_GPT_mnpq_product_l147_14738


namespace NUMINAMATH_GPT_list_price_is_40_l147_14779

theorem list_price_is_40 (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to indicate we're skipping the proof.
  sorry

end NUMINAMATH_GPT_list_price_is_40_l147_14779


namespace NUMINAMATH_GPT_population_net_increase_l147_14736

theorem population_net_increase
  (birth_rate : ℕ) (death_rate : ℕ) (T : ℕ)
  (h1 : birth_rate = 7) (h2 : death_rate = 3) (h3 : T = 86400) :
  (birth_rate - death_rate) * (T / 2) = 172800 :=
by
  sorry

end NUMINAMATH_GPT_population_net_increase_l147_14736


namespace NUMINAMATH_GPT_factor_expression_l147_14731

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2 * x + 1) * (2 * x - 1) :=
sorry

end NUMINAMATH_GPT_factor_expression_l147_14731


namespace NUMINAMATH_GPT_relation_between_y_l147_14715

/-- Definition of the points on the parabola y = -(x-3)^2 - 4 --/
def pointA (y₁ : ℝ) : Prop := y₁ = -(1/4 - 3)^2 - 4
def pointB (y₂ : ℝ) : Prop := y₂ = -(1 - 3)^2 - 4
def pointC (y₃ : ℝ) : Prop := y₃ = -(4 - 3)^2 - 4 

/-- Relationship between y₁, y₂, y₃ for given points on the quadratic function --/
theorem relation_between_y (y₁ y₂ y₃ : ℝ) 
  (hA : pointA y₁)
  (hB : pointB y₂)
  (hC : pointC y₃) : 
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_GPT_relation_between_y_l147_14715
