import Mathlib

namespace NUMINAMATH_GPT_find_gear_p_rpm_l1703_170341

def gear_p_rpm (r : ℕ) (gear_p_revs : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) : Prop :=
  r = gear_p_revs * 2

theorem find_gear_p_rpm (r : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) :
  gear_q_rpm = 40 ∧ time_seconds = 30 ∧ extra_revs_q_over_p = 15 ∧ gear_p_revs = 10 / 2 →
  r = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_gear_p_rpm_l1703_170341


namespace NUMINAMATH_GPT_find_c_l1703_170310

-- Define the function
def f (c x : ℝ) : ℝ := x^4 - 8 * x^2 + c

-- Condition: The function has a minimum value of -14 on the interval [-1, 3]
def condition (c : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1 : ℝ) 3, ∀ y ∈ Set.Icc (-1 : ℝ) 3, f c x ≤ f c y ∧ f c x = -14

-- The theorem to be proved
theorem find_c : ∃ c : ℝ, condition c ∧ c = 2 :=
sorry

end NUMINAMATH_GPT_find_c_l1703_170310


namespace NUMINAMATH_GPT_quadratic_coefficients_l1703_170358

theorem quadratic_coefficients :
  ∀ (a b c : ℤ), (2 * a * a - b * a - 5 = 0) → (a = 2 ∧ b = -1) :=
by
  intros a b c H
  sorry

end NUMINAMATH_GPT_quadratic_coefficients_l1703_170358


namespace NUMINAMATH_GPT_minimum_value_A_l1703_170376

theorem minimum_value_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3) ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_A_l1703_170376


namespace NUMINAMATH_GPT_exists_composite_arith_sequence_pairwise_coprime_l1703_170309

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem exists_composite_arith_sequence_pairwise_coprime (n : ℕ) : 
  ∃ seq : Fin n → ℕ, (∀ i, ∃ k, seq i = factorial n + k) ∧ 
  (∀ i j, i ≠ j → gcd (seq i) (seq j) = 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_composite_arith_sequence_pairwise_coprime_l1703_170309


namespace NUMINAMATH_GPT_total_teams_l1703_170343

theorem total_teams (m n : ℕ) (hmn : m > n) : 
  (m - n) + 1 = m - n + 1 := 
by sorry

end NUMINAMATH_GPT_total_teams_l1703_170343


namespace NUMINAMATH_GPT_fleas_after_treatment_l1703_170382

theorem fleas_after_treatment
  (F : ℕ)  -- F is the number of fleas the dog has left after the treatments
  (half_fleas : ℕ → ℕ)  -- Function representing halving fleas
  (initial_fleas := F + 210)  -- Initial number of fleas before treatment
  (half_fleas_def : ∀ n, half_fleas n = n / 2)  -- Definition of half_fleas function
  (condition : F = (half_fleas (half_fleas (half_fleas (half_fleas initial_fleas)))))  -- Condition given in the problem
  :
  F = 14 := 
  sorry

end NUMINAMATH_GPT_fleas_after_treatment_l1703_170382


namespace NUMINAMATH_GPT_johns_speed_l1703_170327

def time1 : ℕ := 2
def time2 : ℕ := 3
def total_distance : ℕ := 225

def total_time : ℕ := time1 + time2

theorem johns_speed :
  (total_distance : ℝ) / (total_time : ℝ) = 45 :=
sorry

end NUMINAMATH_GPT_johns_speed_l1703_170327


namespace NUMINAMATH_GPT_compute_expression_l1703_170383

theorem compute_expression :
  (75 * 1313 - 25 * 1313 + 50 * 1313 = 131300) :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1703_170383


namespace NUMINAMATH_GPT_circle_equation_tangent_l1703_170325

theorem circle_equation_tangent (h : ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = 25)) :
    ∃ c : ℝ × ℝ, c = (1, 2) ∧ ∃ r : ℝ, r = 5 ∧ ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2) := 
by
    sorry

end NUMINAMATH_GPT_circle_equation_tangent_l1703_170325


namespace NUMINAMATH_GPT_f_of_13_eq_223_l1703_170304

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_of_13_eq_223 : f 13 = 223 := 
by sorry

end NUMINAMATH_GPT_f_of_13_eq_223_l1703_170304


namespace NUMINAMATH_GPT_roses_cut_l1703_170334

def r_before := 13
def r_after := 14

theorem roses_cut : r_after - r_before = 1 := by
  sorry

end NUMINAMATH_GPT_roses_cut_l1703_170334


namespace NUMINAMATH_GPT_desired_percentage_total_annual_income_l1703_170313

variable (investment1 : ℝ)
variable (investment2 : ℝ)
variable (rate1 : ℝ)
variable (rate2 : ℝ)

theorem desired_percentage_total_annual_income (h1 : investment1 = 2000)
  (h2 : rate1 = 0.05)
  (h3 : investment2 = 1000-1e-13)
  (h4 : rate2 = 0.08):
  ((investment1 * rate1 + investment2 * rate2) / (investment1 + investment2) * 100) = 6 := by
  sorry

end NUMINAMATH_GPT_desired_percentage_total_annual_income_l1703_170313


namespace NUMINAMATH_GPT_geometric_series_expr_l1703_170301

theorem geometric_series_expr :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4)))))))))) + 100 = 5592504 := 
sorry

end NUMINAMATH_GPT_geometric_series_expr_l1703_170301


namespace NUMINAMATH_GPT_cakes_difference_l1703_170359

theorem cakes_difference (cakes_made : ℕ) (cakes_sold : ℕ) (cakes_bought : ℕ) 
  (h1 : cakes_made = 648) (h2 : cakes_sold = 467) (h3 : cakes_bought = 193) :
  (cakes_sold - cakes_bought = 274) :=
by
  sorry

end NUMINAMATH_GPT_cakes_difference_l1703_170359


namespace NUMINAMATH_GPT_components_le_20_components_le_n_squared_div_4_l1703_170311

-- Question part b: 8x8 grid, can the number of components be more than 20
theorem components_le_20 {c : ℕ} (h1 : c = 64 / 4) : c ≤ 20 := by
  sorry

-- Question part c: n x n grid, can the number of components be more than n^2 / 4
theorem components_le_n_squared_div_4 (n : ℕ) (h2 : n > 8) {c : ℕ} (h3 : c = n^2 / 4) : 
  c ≤ n^2 / 4 := by
  sorry

end NUMINAMATH_GPT_components_le_20_components_le_n_squared_div_4_l1703_170311


namespace NUMINAMATH_GPT_value_of_expression_l1703_170339

variable {a : Nat → Int}

def arithmetic_sequence (a : Nat → Int) : Prop :=
  ∀ n m : Nat, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_expression
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 :=
  sorry

end NUMINAMATH_GPT_value_of_expression_l1703_170339


namespace NUMINAMATH_GPT_seq_100_eq_11_div_12_l1703_170374

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1 / 3
  else if n ≥ 3 then (2 - seq (n - 1)) / (3 * seq (n - 2) + 1)
  else 0 -- This line handles the case n < 1, but shouldn't ever be used in practice.

theorem seq_100_eq_11_div_12 : seq 100 = 11 / 12 :=
  sorry

end NUMINAMATH_GPT_seq_100_eq_11_div_12_l1703_170374


namespace NUMINAMATH_GPT_one_third_pow_3_eq_3_pow_nineteen_l1703_170350

theorem one_third_pow_3_eq_3_pow_nineteen (y : ℤ) (h : (1 / 3 : ℝ) * (3 ^ 20) = 3 ^ y) : y = 19 :=
by
  sorry

end NUMINAMATH_GPT_one_third_pow_3_eq_3_pow_nineteen_l1703_170350


namespace NUMINAMATH_GPT_minimum_value_of_a2b_l1703_170386

noncomputable def minimum_value (a b : ℝ) := a + 2 * b

theorem minimum_value_of_a2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / (2 * a + b) + 1 / (b + 1) = 1) :
  minimum_value a b = (2 * Real.sqrt 3 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a2b_l1703_170386


namespace NUMINAMATH_GPT_simple_interest_rate_l1703_170326

theorem simple_interest_rate (P T A R : ℝ) (hT : T = 15) (hA : A = 4 * P)
  (hA_simple_interest : A = P + (P * R * T / 100)) : R = 20 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1703_170326


namespace NUMINAMATH_GPT_geometric_progression_sixth_term_proof_l1703_170365

noncomputable def geometric_progression_sixth_term (b₁ b₅ : ℝ) (q : ℝ) := b₅ * q
noncomputable def find_q (b₁ b₅ : ℝ) := (b₅ / b₁)^(1/4)

theorem geometric_progression_sixth_term_proof (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) : 
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = - Real.sqrt 3) ∧ geometric_progression_sixth_term b₁ b₅ q = 27 ∨ geometric_progression_sixth_term b₁ b₅ q = -27 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_sixth_term_proof_l1703_170365


namespace NUMINAMATH_GPT_sum_abc_eq_8_l1703_170312

theorem sum_abc_eq_8 (a b c : ℝ) 
  (h : (a - 5) ^ 2 + (b - 6) ^ 2 + (c - 7) ^ 2 - 2 * (a - 5) * (b - 6) = 0) : 
  a + b + c = 8 := 
sorry

end NUMINAMATH_GPT_sum_abc_eq_8_l1703_170312


namespace NUMINAMATH_GPT_black_stones_count_l1703_170385

theorem black_stones_count (T W B : ℕ) (hT : T = 48) (hW1 : 4 * W = 37 * 2 + 26) (hB : B = T - W) : B = 23 :=
by
  sorry

end NUMINAMATH_GPT_black_stones_count_l1703_170385


namespace NUMINAMATH_GPT_math_problem_solution_l1703_170340

theorem math_problem_solution (a b n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_eq : a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
sorry

end NUMINAMATH_GPT_math_problem_solution_l1703_170340


namespace NUMINAMATH_GPT_multiple_of_every_positive_integer_is_zero_l1703_170329

theorem multiple_of_every_positive_integer_is_zero :
  ∀ (n : ℤ), (∀ (m : ℕ), ∃ (k : ℤ), n = k * (m : ℤ)) → n = 0 := 
by
  sorry

end NUMINAMATH_GPT_multiple_of_every_positive_integer_is_zero_l1703_170329


namespace NUMINAMATH_GPT_S6_equals_63_l1703_170397

variable {S : ℕ → ℕ}

-- Define conditions
axiom S_n_geometric_sequence (a : ℕ → ℕ) (n : ℕ) : n ≥ 1 → S n = (a 0) * ((a 1)^(n) -1) / (a 1 - 1)
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- State theorem
theorem S6_equals_63 : S 6 = 63 := by
  sorry

end NUMINAMATH_GPT_S6_equals_63_l1703_170397


namespace NUMINAMATH_GPT_blue_candies_count_l1703_170389

theorem blue_candies_count (total_pieces red_pieces : Nat) (h1 : total_pieces = 3409) (h2 : red_pieces = 145) : total_pieces - red_pieces = 3264 := 
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_blue_candies_count_l1703_170389


namespace NUMINAMATH_GPT_find_y_in_terms_of_abc_l1703_170390

theorem find_y_in_terms_of_abc 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (h1 : xy / (x - y) = a)
  (h2 : xz / (x - z) = b)
  (h3 : yz / (y - z) = c) :
  y = bcx / ((b + c) * x - bc) := 
sorry

end NUMINAMATH_GPT_find_y_in_terms_of_abc_l1703_170390


namespace NUMINAMATH_GPT_students_taking_both_chorus_and_band_l1703_170375

theorem students_taking_both_chorus_and_band (total_students : ℕ) 
                                             (chorus_students : ℕ)
                                             (band_students : ℕ)
                                             (not_enrolled_students : ℕ) : 
                                             total_students = 50 ∧
                                             chorus_students = 18 ∧
                                             band_students = 26 ∧
                                             not_enrolled_students = 8 →
                                             ∃ (both_chorus_and_band : ℕ), both_chorus_and_band = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_students_taking_both_chorus_and_band_l1703_170375


namespace NUMINAMATH_GPT_finishing_order_l1703_170369

-- Definitions of conditions
def athletes := ["Grisha", "Sasha", "Lena"]

def overtakes : (String → ℕ) := 
  fun athlete =>
    if athlete = "Grisha" then 10
    else if athlete = "Sasha" then 4
    else if athlete = "Lena" then 6
    else 0

-- All three were never at the same point at the same time
def never_same_point_at_same_time : Prop := True -- Simplified for translation purpose

-- The main theorem stating the finishing order given the provided conditions
theorem finishing_order :
  never_same_point_at_same_time →
  (overtakes "Grisha" = 10) →
  (overtakes "Sasha" = 4) →
  (overtakes "Lena" = 6) →
  athletes = ["Grisha", "Sasha", "Lena"] :=
  by
    intro h1 h2 h3 h4
    exact sorry -- The proof is not required, just ensuring the statement is complete.


end NUMINAMATH_GPT_finishing_order_l1703_170369


namespace NUMINAMATH_GPT_find_a_l1703_170337

theorem find_a (a b : ℝ) (h₀ : b = 4) (h₁ : (4, b) ∈ {p | p.snd = 0.75 * p.fst + 1}) 
  (h₂ : (a, 5) ∈ {p | p.snd = 0.75 * p.fst + 1}) (h₃ : (a, b+1) ∈ {p | p.snd = 0.75 * p.fst + 1}) : 
  a = 5.33 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_l1703_170337


namespace NUMINAMATH_GPT_solve_P_Q_l1703_170333

theorem solve_P_Q :
  ∃ P Q : ℝ, (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 6) + Q / (x * (x - 5)) = (x^2 - 3*x + 15) / (x * (x + 6) * (x - 5)))) ∧
    P = 1 ∧ Q = 5/2 :=
by
  sorry

end NUMINAMATH_GPT_solve_P_Q_l1703_170333


namespace NUMINAMATH_GPT_evaluate_expression_l1703_170306

-- Define the expression and the expected result
def expression := -(14 / 2 * 9 - 60 + 3 * 9)
def expectedResult := -30

-- The theorem that states the equivalence
theorem evaluate_expression : expression = expectedResult := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1703_170306


namespace NUMINAMATH_GPT_factorization_correct_l1703_170391

noncomputable def factor_polynomial (x : ℝ) : ℝ := 4 * x^3 - 4 * x^2 + x

theorem factorization_correct (x : ℝ) : 
  factor_polynomial x = x * (2 * x - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1703_170391


namespace NUMINAMATH_GPT_extreme_value_of_f_range_of_values_for_a_l1703_170352

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem extreme_value_of_f :
  ∃ x_min : ℝ, f x_min = 1 :=
sorry

theorem range_of_values_for_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ (x^3) / 6 + a) → a ≤ 1 :=
sorry

end NUMINAMATH_GPT_extreme_value_of_f_range_of_values_for_a_l1703_170352


namespace NUMINAMATH_GPT_modulus_sum_l1703_170303

def z1 : ℂ := 3 - 5 * Complex.I
def z2 : ℂ := 3 + 5 * Complex.I

theorem modulus_sum : Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := 
by 
  sorry

end NUMINAMATH_GPT_modulus_sum_l1703_170303


namespace NUMINAMATH_GPT_park_area_l1703_170387

theorem park_area (l w : ℝ) (h1 : l + w = 40) (h2 : l = 3 * w) : l * w = 300 :=
by
  sorry

end NUMINAMATH_GPT_park_area_l1703_170387


namespace NUMINAMATH_GPT_mark_more_hours_than_kate_l1703_170320

theorem mark_more_hours_than_kate {K : ℕ} (h1 : K + 2 * K + 6 * K = 117) :
  6 * K - K = 65 :=
by
  sorry

end NUMINAMATH_GPT_mark_more_hours_than_kate_l1703_170320


namespace NUMINAMATH_GPT_system_solution_l1703_170381

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 3) : x - y = 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_system_solution_l1703_170381


namespace NUMINAMATH_GPT_ratio_yx_l1703_170315

variable (c x y : ℝ)

theorem ratio_yx (h1: x = 0.80 * c) (h2: y = 1.25 * c) : y / x = 25 / 16 := by
  -- Proof to be written here
  sorry

end NUMINAMATH_GPT_ratio_yx_l1703_170315


namespace NUMINAMATH_GPT_frenchwoman_present_l1703_170357

theorem frenchwoman_present
    (M_F M_R W_R : ℝ)
    (condition_1 : M_F > M_R + W_R)
    (condition_2 : W_R > M_F + M_R) 
    : false :=
by
  -- We would assume the opposite of what we know to lead to a contradiction here.
  -- This is a placeholder to indicate the proof should lead to a contradiction.
  sorry

end NUMINAMATH_GPT_frenchwoman_present_l1703_170357


namespace NUMINAMATH_GPT_chuck_bicycle_trip_l1703_170366

theorem chuck_bicycle_trip (D : ℝ) (h1 : D / 16 + D / 24 = 3) : D = 28.80 :=
by
  sorry

end NUMINAMATH_GPT_chuck_bicycle_trip_l1703_170366


namespace NUMINAMATH_GPT_find_a_b_l1703_170346

theorem find_a_b (a b : ℤ) (h : ∀ x : ℤ, (x - 2) * (x + 3) = x^2 + a * x + b) : a = 1 ∧ b = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l1703_170346


namespace NUMINAMATH_GPT_domain_of_function_l1703_170342

noncomputable def function_defined (x : ℝ) : Prop :=
  (x > 1) ∧ (x ≠ 2)

theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, y = (1 / (Real.sqrt (x - 1))) + (1 / (x - 2))) ↔ function_defined x :=
by sorry

end NUMINAMATH_GPT_domain_of_function_l1703_170342


namespace NUMINAMATH_GPT_find_large_number_l1703_170378

theorem find_large_number (L S : ℤ)
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := 
sorry

end NUMINAMATH_GPT_find_large_number_l1703_170378


namespace NUMINAMATH_GPT_point_not_in_third_quadrant_l1703_170363

theorem point_not_in_third_quadrant (x y : ℝ) (h : y = -x + 1) : ¬(x < 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_not_in_third_quadrant_l1703_170363


namespace NUMINAMATH_GPT_range_of_a_l1703_170394

-- Define sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Mathematical statement to be proven
theorem range_of_a (a : ℝ) : (∃ x, x ∈ set_A ∧ x ∈ set_B a) → a ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1703_170394


namespace NUMINAMATH_GPT_find_f_neg1_l1703_170392

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1 else -2^(-x) + 2*x + 1

theorem find_f_neg1 : f (-1) = -3 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_find_f_neg1_l1703_170392


namespace NUMINAMATH_GPT_value_of_nested_f_l1703_170351

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_nested_f : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end NUMINAMATH_GPT_value_of_nested_f_l1703_170351


namespace NUMINAMATH_GPT_avg_daily_production_n_l1703_170371

theorem avg_daily_production_n (n : ℕ) (h₁ : 50 * n + 110 = 55 * (n + 1)) : n = 11 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_avg_daily_production_n_l1703_170371


namespace NUMINAMATH_GPT_larry_substitution_l1703_170302

theorem larry_substitution (a b c d e : ℤ)
  (ha : a = 1)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 4)
  (h_ignored : a - b - c - d + e = a - (b - (c - (d + e)))) :
  e = 3 :=
by
  sorry

end NUMINAMATH_GPT_larry_substitution_l1703_170302


namespace NUMINAMATH_GPT_det_B_l1703_170319

open Matrix

-- Define matrix B
def B (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 2], ![-3, y]]

-- Define the condition B + 2 * B⁻¹ = 0
def condition (x y : ℝ) : Prop :=
  let Binv := (1 / (x * y + 6)) • ![![y, -2], ![3, x]]
  B x y + 2 • Binv = 0

-- Prove that if the condition holds, then det B = 2
theorem det_B (x y : ℝ) (h : condition x y) : det (B x y) = 2 :=
  sorry

end NUMINAMATH_GPT_det_B_l1703_170319


namespace NUMINAMATH_GPT_second_job_pay_rate_l1703_170308

-- Definitions of the conditions
def h1 : ℕ := 3 -- hours for the first job
def r1 : ℕ := 7 -- rate for the first job
def h2 : ℕ := 2 -- hours for the second job
def h3 : ℕ := 4 -- hours for the third job
def r3 : ℕ := 12 -- rate for the third job
def d : ℕ := 5   -- number of days
def T : ℕ := 445 -- total earnings

-- The proof statement
theorem second_job_pay_rate (x : ℕ) : 
  d * (h1 * r1 + 2 * x + h3 * r3) = T ↔ x = 10 := 
by 
  -- Implement the necessary proof steps here
  sorry

end NUMINAMATH_GPT_second_job_pay_rate_l1703_170308


namespace NUMINAMATH_GPT_necessary_condition_transitivity_l1703_170399

theorem necessary_condition_transitivity (A B C : Prop) 
  (hAB : A → B) (hBC : B → C) : A → C := 
by
  intro ha
  apply hBC
  apply hAB
  exact ha

-- sorry


end NUMINAMATH_GPT_necessary_condition_transitivity_l1703_170399


namespace NUMINAMATH_GPT_measure_angle_B_triangle_area_correct_l1703_170300

noncomputable def triangle_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) → B = Real.pi / 3

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area1 := (3 + Real.sqrt 3)
  let area2 := Real.sqrt 3
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  let sinA1 := (Real.sqrt 2 / 2)
  let sinA2 := (Real.sqrt 6 - Real.sqrt 2) / 4
  let S1 := (1 / 2) * b * c * sinA1
  let S2 := (1 / 2) * b * c * sinA2
  S1 = area1 ∨ S2 = area2

theorem measure_angle_B :
  ∀ (a b c A B C : ℝ),
    triangle_angle_B a b c A B C := sorry

theorem triangle_area_correct :
  ∀ (a b c A B C : ℝ),
    triangle_area a b c A B C := sorry

end NUMINAMATH_GPT_measure_angle_B_triangle_area_correct_l1703_170300


namespace NUMINAMATH_GPT_annual_interest_rate_l1703_170384

theorem annual_interest_rate (initial_amount final_amount : ℝ) 
  (h_initial : initial_amount = 90) 
  (h_final : final_amount = 99) : 
  ((final_amount - initial_amount) / initial_amount) * 100 = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_annual_interest_rate_l1703_170384


namespace NUMINAMATH_GPT_alpha_beta_squared_l1703_170353

section
variables (α β : ℝ)
-- Given conditions
def is_root (a b : ℝ) : Prop :=
  a + b = 2 ∧ a * b = -1 ∧ (∀ x : ℝ, x^2 - 2 * x - 1 = 0 → x = a ∨ x = b)

-- The theorem to prove
theorem alpha_beta_squared (h: is_root α β) : α^2 + β^2 = 6 :=
sorry
end

end NUMINAMATH_GPT_alpha_beta_squared_l1703_170353


namespace NUMINAMATH_GPT_inequality_proof_l1703_170395

theorem inequality_proof (a b c : ℝ) (h : a * c^2 > b * c^2) (hc2 : c^2 > 0) : a > b :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1703_170395


namespace NUMINAMATH_GPT_multiple_of_a_power_l1703_170361

theorem multiple_of_a_power (a n m : ℕ) (h : a^n ∣ m) : a^(n+1) ∣ (a+1)^m - 1 := 
sorry

end NUMINAMATH_GPT_multiple_of_a_power_l1703_170361


namespace NUMINAMATH_GPT_only_positive_integer_a_squared_plus_2a_is_perfect_square_l1703_170364

/-- Prove that the only positive integer \( a \) for which \( a^2 + 2a \) is a perfect square is \( a = 0 \). -/
theorem only_positive_integer_a_squared_plus_2a_is_perfect_square :
  ∀ (a : ℕ), (∃ (k : ℕ), a^2 + 2*a = k^2) → a = 0 :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_only_positive_integer_a_squared_plus_2a_is_perfect_square_l1703_170364


namespace NUMINAMATH_GPT_cricketer_average_after_22nd_inning_l1703_170377

theorem cricketer_average_after_22nd_inning (A : ℚ) 
  (h1 : 21 * A + 134 = (A + 3.5) * 22)
  (h2 : 57 = A) :
  A + 3.5 = 60.5 :=
by
  exact sorry

end NUMINAMATH_GPT_cricketer_average_after_22nd_inning_l1703_170377


namespace NUMINAMATH_GPT_sum_of_first_n_natural_numbers_single_digit_l1703_170317

theorem sum_of_first_n_natural_numbers_single_digit (n : ℕ) :
  (∃ a : ℕ, a ≤ 9 ∧ (a ≠ 0) ∧ 37 * (3 * a) = n * (n + 1) / 2) ↔ (n = 36) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_n_natural_numbers_single_digit_l1703_170317


namespace NUMINAMATH_GPT_five_diff_numbers_difference_l1703_170370

theorem five_diff_numbers_difference (S : Finset ℕ) (hS_size : S.card = 5) 
    (hS_range : ∀ x ∈ S, x ≤ 10) : 
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a - b = c - d ∧ a - b ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_five_diff_numbers_difference_l1703_170370


namespace NUMINAMATH_GPT_hash_value_is_minus_15_l1703_170321

def hash (a b c : ℝ) : ℝ := b^2 - 3 * a * c

theorem hash_value_is_minus_15 : hash 2 3 4 = -15 :=
by
  sorry

end NUMINAMATH_GPT_hash_value_is_minus_15_l1703_170321


namespace NUMINAMATH_GPT_balloons_remaining_intact_l1703_170388

def initial_balloons : ℕ := 200
def blown_up_after_half_hour (n : ℕ) : ℕ := n / 5
def remaining_balloons_after_half_hour (n : ℕ) : ℕ := n - blown_up_after_half_hour n

def percentage_of_remaining_balloons_blow_up (remaining : ℕ) : ℕ := remaining * 30 / 100
def remaining_balloons_after_one_hour (remaining : ℕ) : ℕ := remaining - percentage_of_remaining_balloons_blow_up remaining

def durable_balloons (remaining : ℕ) : ℕ := remaining * 10 / 100
def non_durable_balloons (remaining : ℕ) (durable : ℕ) : ℕ := remaining - durable

def twice_non_durable (non_durable : ℕ) : ℕ := non_durable * 2

theorem balloons_remaining_intact : 
  (remaining_balloons_after_half_hour initial_balloons) - 
  (percentage_of_remaining_balloons_blow_up 
    (remaining_balloons_after_half_hour initial_balloons)) - 
  (twice_non_durable 
    (non_durable_balloons 
      (remaining_balloons_after_one_hour 
        (remaining_balloons_after_half_hour initial_balloons)) 
      (durable_balloons 
        (remaining_balloons_after_one_hour 
          (remaining_balloons_after_half_hour initial_balloons))))) = 
  0 := 
by
  sorry

end NUMINAMATH_GPT_balloons_remaining_intact_l1703_170388


namespace NUMINAMATH_GPT_solve_for_b_l1703_170331

theorem solve_for_b (a b : ℚ) 
  (h1 : 8 * a + 3 * b = -1) 
  (h2 : a = b - 3 ) : 
  5 * b = 115 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_b_l1703_170331


namespace NUMINAMATH_GPT_solve_for_m_l1703_170372

theorem solve_for_m (x m : ℝ) (h1 : 2 * 1 - m = -3) : m = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l1703_170372


namespace NUMINAMATH_GPT_xyz_inequality_l1703_170314

theorem xyz_inequality (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + 
  (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
sorry

end NUMINAMATH_GPT_xyz_inequality_l1703_170314


namespace NUMINAMATH_GPT_product_of_roots_cubic_l1703_170347

theorem product_of_roots_cubic:
  (∀ x : ℝ, x^3 - 15 * x^2 + 60 * x - 45 = 0 → x = r_1 ∨ x = r_2 ∨ x = r_3) →
  r_1 * r_2 * r_3 = 45 :=
by
  intro h
  -- the proof should be filled in here
  sorry

end NUMINAMATH_GPT_product_of_roots_cubic_l1703_170347


namespace NUMINAMATH_GPT_tobee_points_l1703_170380

theorem tobee_points (T J S : ℕ) (h1 : J = T + 6) (h2 : S = 2 * (T + 3) - 2) (h3 : T + J + S = 26) : T = 4 := 
by
  sorry

end NUMINAMATH_GPT_tobee_points_l1703_170380


namespace NUMINAMATH_GPT_cube_identity_l1703_170336

theorem cube_identity (a : ℝ) (h : (a + 1/a) ^ 2 = 3) : a^3 + 1/a^3 = 0 := 
by
  sorry

end NUMINAMATH_GPT_cube_identity_l1703_170336


namespace NUMINAMATH_GPT_triangle_area_ratio_l1703_170344

noncomputable def vector_sum_property (OA OB OC : ℝ × ℝ × ℝ) : Prop :=
  OA + (2 : ℝ) • OB + (3 : ℝ) • OC = (0 : ℝ × ℝ × ℝ)

noncomputable def area_ratio (S_ABC S_AOC : ℝ) : Prop :=
  S_ABC / S_AOC = 3

theorem triangle_area_ratio
    (OA OB OC : ℝ × ℝ × ℝ)
    (S_ABC S_AOC : ℝ)
    (h1 : vector_sum_property OA OB OC)
    (h2 : S_ABC = 3 * S_AOC) :
  area_ratio S_ABC S_AOC :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1703_170344


namespace NUMINAMATH_GPT_exist_positive_m_l1703_170368

theorem exist_positive_m {n p q : ℕ} (hn_pos : 0 < n) (hp_prime : Prime p) (hq_prime : Prime q) 
  (h1 : pq ∣ n ^ p + 2) (h2 : n + 2 ∣ n ^ p + q ^ p) : ∃ m : ℕ, q ∣ 4 ^ m * n + 2 := 
sorry

end NUMINAMATH_GPT_exist_positive_m_l1703_170368


namespace NUMINAMATH_GPT_smallest_p_l1703_170362

theorem smallest_p (p q : ℕ) (h1 : p + q = 2005) (h2 : (5:ℚ)/8 < p / q) (h3 : p / q < (7:ℚ)/8) : p = 772 :=
sorry

end NUMINAMATH_GPT_smallest_p_l1703_170362


namespace NUMINAMATH_GPT_part1_part2_l1703_170335

open Real

noncomputable def f (x a : ℝ) : ℝ := x * exp (a * x) + x * cos x + 1

theorem part1 (x : ℝ) (hx : 0 ≤ x) : cos x ≥ 1 - (1 / 2) * x^2 := 
sorry

theorem part2 (a x : ℝ) (ha : 1 ≤ a) (hx : 0 ≤ x) : f x a ≥ (1 + sin x)^2 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1703_170335


namespace NUMINAMATH_GPT_opposite_numbers_pow_sum_zero_l1703_170393

theorem opposite_numbers_pow_sum_zero (a b : ℝ) (h : a + b = 0) : a^5 + b^5 = 0 :=
by sorry

end NUMINAMATH_GPT_opposite_numbers_pow_sum_zero_l1703_170393


namespace NUMINAMATH_GPT_largest_of_four_consecutive_even_numbers_l1703_170356

-- Conditions
def sum_of_four_consecutive_even_numbers (x : ℤ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) = 92

-- Proof statement
theorem largest_of_four_consecutive_even_numbers (x : ℤ) 
  (h : sum_of_four_consecutive_even_numbers x) : x + 6 = 26 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_four_consecutive_even_numbers_l1703_170356


namespace NUMINAMATH_GPT_opposite_of_2023_l1703_170348

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l1703_170348


namespace NUMINAMATH_GPT_extremum_of_function_l1703_170354

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem extremum_of_function :
  (∀ x, f x ≥ -Real.exp 1) ∧ (f 1 = -Real.exp 1) ∧ (∀ M, ∃ x, f x > M) :=
by
  sorry

end NUMINAMATH_GPT_extremum_of_function_l1703_170354


namespace NUMINAMATH_GPT_find_x_for_parallel_vectors_l1703_170328

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (1, x)
def b (x : ℝ) : vector := (2, 2 - x)

def are_parallel (v w : vector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, are_parallel (a x) (b x) → x = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_parallel_vectors_l1703_170328


namespace NUMINAMATH_GPT_maximum_area_of_rectangle_with_given_perimeter_l1703_170324

theorem maximum_area_of_rectangle_with_given_perimeter {x y : ℕ} (h₁ : 2 * x + 2 * y = 160) : 
  (∃ x y : ℕ, 2 * x + 2 * y = 160 ∧ x * y = 1600) := 
sorry

end NUMINAMATH_GPT_maximum_area_of_rectangle_with_given_perimeter_l1703_170324


namespace NUMINAMATH_GPT_rates_of_interest_l1703_170349

theorem rates_of_interest (P_B P_C T_B T_C SI_B SI_C : ℝ) (R_B R_C : ℝ)
  (hB1 : P_B = 5000) (hB2: T_B = 5) (hB3: SI_B = 2200)
  (hC1 : P_C = 3000) (hC2 : T_C = 7) (hC3 : SI_C = 2730)
  (simple_interest : ∀ {P R T SI : ℝ}, SI = (P * R * T) / 100)
  : R_B = 8.8 ∧ R_C = 13 := by
  sorry

end NUMINAMATH_GPT_rates_of_interest_l1703_170349


namespace NUMINAMATH_GPT_consecutive_integers_sum_l1703_170373

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l1703_170373


namespace NUMINAMATH_GPT_min_y_value_l1703_170322

noncomputable def min_value_y : ℝ :=
  18 - 2 * Real.sqrt 106

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20 * x + 36 * y) : 
  y >= 18 - 2 * Real.sqrt 106 :=
sorry

end NUMINAMATH_GPT_min_y_value_l1703_170322


namespace NUMINAMATH_GPT_minimum_dot_product_l1703_170345

noncomputable def min_AE_dot_AF : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60 -- this is 60 degrees, which should be converted to radians if we need to use it
  sorry

theorem minimum_dot_product :
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60
  ∃ (E F : ℝ), (min_AE_dot_AF = 29 / 18) :=
    sorry

end NUMINAMATH_GPT_minimum_dot_product_l1703_170345


namespace NUMINAMATH_GPT_intersection_point_of_circle_and_line_l1703_170379

noncomputable def circle_parametric (α : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos α, 2 * Real.sin α)
noncomputable def line_polar (rho θ : ℝ) : Prop := rho * Real.sin θ = 2

theorem intersection_point_of_circle_and_line :
  ∃ (α : ℝ) (rho θ : ℝ), circle_parametric α = (1, 2) ∧ line_polar rho θ := sorry

end NUMINAMATH_GPT_intersection_point_of_circle_and_line_l1703_170379


namespace NUMINAMATH_GPT_series_sum_equals_one_sixth_l1703_170307

noncomputable def series_sum : ℝ :=
  ∑' n, 2^n / (7^(2^n) + 1)

theorem series_sum_equals_one_sixth : series_sum = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_equals_one_sixth_l1703_170307


namespace NUMINAMATH_GPT_vicente_meat_purchase_l1703_170396

theorem vicente_meat_purchase :
  ∃ (meat_lbs : ℕ),
  (∃ (rice_kgs cost_rice_per_kg cost_meat_per_lb total_spent : ℕ),
    rice_kgs = 5 ∧
    cost_rice_per_kg = 2 ∧
    cost_meat_per_lb = 5 ∧
    total_spent = 25 ∧
    total_spent - (rice_kgs * cost_rice_per_kg) = meat_lbs * cost_meat_per_lb) ∧
  meat_lbs = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_vicente_meat_purchase_l1703_170396


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l1703_170305

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.cos α < 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * Real.pi + β ∧ β ∈ Set.Ioo (0 : ℝ) Real.pi :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l1703_170305


namespace NUMINAMATH_GPT_consecutive_numbers_N_l1703_170323

theorem consecutive_numbers_N (N : ℕ) (h : ∀ k, 0 < k → k < 15 → N + k < 81) : N = 66 :=
sorry

end NUMINAMATH_GPT_consecutive_numbers_N_l1703_170323


namespace NUMINAMATH_GPT_min_sum_of_angles_l1703_170338

theorem min_sum_of_angles (A B C : ℝ) (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin B + Real.sin C ≤ 1) : 
  min (A + B) (min (B + C) (C + A)) < 30 := 
sorry

end NUMINAMATH_GPT_min_sum_of_angles_l1703_170338


namespace NUMINAMATH_GPT_original_plan_trees_per_day_l1703_170355

theorem original_plan_trees_per_day (x : ℕ) :
  (∃ x, (960 / x - 960 / (2 * x) = 4)) → x = 120 := 
sorry

end NUMINAMATH_GPT_original_plan_trees_per_day_l1703_170355


namespace NUMINAMATH_GPT_maximum_m_l1703_170330

theorem maximum_m (a b c : ℝ)
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b + c = 10)
  (h₅ : a * b + b * c + c * a = 25) :
  ∃ m, (m = min (a * b) (min (b * c) (c * a)) ∧ m = 25 / 9) :=
sorry

end NUMINAMATH_GPT_maximum_m_l1703_170330


namespace NUMINAMATH_GPT_polygon_sides_l1703_170398

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1703_170398


namespace NUMINAMATH_GPT_geometric_sequence_l1703_170316

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence
def geom_seq (a₁ r : α) (n : ℕ) : α := a₁ * r^(n-1)

theorem geometric_sequence :
  ∀ (a₁ : α), a₁ > 0 → geom_seq a₁ 2 3 * geom_seq a₁ 2 11 = 16 → geom_seq a₁ 2 5 = 1 :=
by
  intros a₁ h_pos h_eq
  sorry

end NUMINAMATH_GPT_geometric_sequence_l1703_170316


namespace NUMINAMATH_GPT_candy_bar_sales_l1703_170332

def max_sales : ℕ := 24
def seth_sales (max_sales : ℕ) : ℕ := 3 * max_sales + 6
def emma_sales (seth_sales : ℕ) : ℕ := seth_sales / 2 + 5
def total_sales (seth_sales emma_sales : ℕ) : ℕ := seth_sales + emma_sales

theorem candy_bar_sales : total_sales (seth_sales max_sales) (emma_sales (seth_sales max_sales)) = 122 := by
  sorry

end NUMINAMATH_GPT_candy_bar_sales_l1703_170332


namespace NUMINAMATH_GPT_geometric_mean_eq_6_l1703_170318

theorem geometric_mean_eq_6 (b c : ℝ) (hb : b = 3) (hc : c = 12) :
  (b * c) ^ (1/2 : ℝ) = 6 := 
by
  sorry

end NUMINAMATH_GPT_geometric_mean_eq_6_l1703_170318


namespace NUMINAMATH_GPT_simple_interest_rate_l1703_170360

theorem simple_interest_rate (P A T : ℝ) (R : ℝ) (hP : P = 750) (hA : A = 900) (hT : T = 5) :
    (A - P) = (P * R * T) / 100 → R = 4 := by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1703_170360


namespace NUMINAMATH_GPT_lcm_problem_l1703_170367

theorem lcm_problem :
  ∃ k_values : Finset ℕ, (∀ k ∈ k_values, (60^10 : ℕ) = Nat.lcm (Nat.lcm (10^10) (12^12)) k) ∧ k_values.card = 121 :=
by
  sorry

end NUMINAMATH_GPT_lcm_problem_l1703_170367
