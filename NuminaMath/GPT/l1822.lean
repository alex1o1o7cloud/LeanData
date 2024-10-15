import Mathlib

namespace NUMINAMATH_GPT_minimum_15_equal_differences_l1822_182213

-- Definition of distinct integers a_i
def distinct_sequence (a : Fin 100 → ℕ) : Prop :=
  ∀ i j : Fin 100, i < j → a i < a j

-- Definition of the differences d_i
def differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, d i = a ⟨i + 1, Nat.lt_of_lt_of_le (Nat.succ_lt_succ i.2) (by norm_num)⟩ - a i

-- Main theorem statement
theorem minimum_15_equal_differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) :
  (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 400) →
  distinct_sequence a →
  differences a d →
  ∃ t : Finset ℕ, t.card ≥ 15 ∧ ∀ x : ℕ, x ∈ t → (∃ i j : Fin 99, i ≠ j ∧ d i = x ∧ d j = x) :=
sorry

end NUMINAMATH_GPT_minimum_15_equal_differences_l1822_182213


namespace NUMINAMATH_GPT_probability_exactly_five_shots_expected_shots_to_hit_all_l1822_182272

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end NUMINAMATH_GPT_probability_exactly_five_shots_expected_shots_to_hit_all_l1822_182272


namespace NUMINAMATH_GPT_total_meals_per_week_l1822_182256

-- Definitions of the conditions
def meals_per_day_r1 : ℕ := 20
def meals_per_day_r2 : ℕ := 40
def meals_per_day_r3 : ℕ := 50
def days_per_week : ℕ := 7

-- The proof goal
theorem total_meals_per_week : 
  (meals_per_day_r1 * days_per_week) + 
  (meals_per_day_r2 * days_per_week) + 
  (meals_per_day_r3 * days_per_week) = 770 :=
by
  sorry

end NUMINAMATH_GPT_total_meals_per_week_l1822_182256


namespace NUMINAMATH_GPT_starting_number_of_range_divisible_by_11_l1822_182287

theorem starting_number_of_range_divisible_by_11 (a : ℕ) : 
  a ≤ 79 ∧ (a + 22 = 77) ∧ ((a + 11) + 11 = 77) → a = 55 := 
by
  sorry

end NUMINAMATH_GPT_starting_number_of_range_divisible_by_11_l1822_182287


namespace NUMINAMATH_GPT_negation_of_universal_statement_l1822_182244

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by 
  -- Proof steps would be added here
  sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l1822_182244


namespace NUMINAMATH_GPT_no_real_roots_smallest_m_l1822_182282

theorem no_real_roots_smallest_m :
  ∃ m : ℕ, m = 4 ∧
  ∀ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 7 = 0 → ¬ ∃ x₀ : ℝ, 
  (3 * m - 2) * x₀^2 - 15 * x₀ + 7 = 0 ∧ 281 - 84 * m < 0 := sorry

end NUMINAMATH_GPT_no_real_roots_smallest_m_l1822_182282


namespace NUMINAMATH_GPT_bathing_suits_per_model_l1822_182248

def models : ℕ := 6
def evening_wear_sets_per_model : ℕ := 3
def time_per_trip_minutes : ℕ := 2
def total_show_time_minutes : ℕ := 60

theorem bathing_suits_per_model : (total_show_time_minutes - (models * evening_wear_sets_per_model * time_per_trip_minutes)) / (time_per_trip_minutes * models) = 2 :=
by
  sorry

end NUMINAMATH_GPT_bathing_suits_per_model_l1822_182248


namespace NUMINAMATH_GPT_polynomial_equality_l1822_182219

theorem polynomial_equality :
  (3 * x + 1) ^ 4 = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e →
  a - b + c - d + e = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_equality_l1822_182219


namespace NUMINAMATH_GPT_rectangle_perimeter_is_28_l1822_182223

-- Define the variables and conditions
variables (h w : ℝ)

-- Problem conditions
def rectangle_area (h w : ℝ) : Prop := h * w = 40
def width_greater_than_twice_height (h w : ℝ) : Prop := w > 2 * h
def parallelogram_area (h w : ℝ) : Prop := h * (w - h) = 24

-- The theorem stating the perimeter of the rectangle given the conditions
theorem rectangle_perimeter_is_28 (h w : ℝ) 
  (H1 : rectangle_area h w) 
  (H2 : width_greater_than_twice_height h w) 
  (H3 : parallelogram_area h w) :
  2 * h + 2 * w = 28 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_is_28_l1822_182223


namespace NUMINAMATH_GPT_domain_inequality_l1822_182293

theorem domain_inequality (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ (m ≥ 1/3) :=
by
  sorry

end NUMINAMATH_GPT_domain_inequality_l1822_182293


namespace NUMINAMATH_GPT_james_initial_amount_l1822_182294

noncomputable def initial_amount (total_amount_invested_per_week: ℕ) 
                                (number_of_weeks_in_year: ℕ) 
                                (windfall_factor: ℚ) 
                                (amount_after_windfall: ℕ) : ℚ :=
  let total_investment := total_amount_invested_per_week * number_of_weeks_in_year
  let amount_without_windfall := (amount_after_windfall : ℚ) / (1 + windfall_factor)
  amount_without_windfall - total_investment

theorem james_initial_amount:
  initial_amount 2000 52 0.5 885000 = 250000 := sorry

end NUMINAMATH_GPT_james_initial_amount_l1822_182294


namespace NUMINAMATH_GPT_sequence_eventually_periodic_modulo_l1822_182225

noncomputable def a_n (n : ℕ) : ℕ :=
  n ^ n + (n - 1) ^ (n + 1)

theorem sequence_eventually_periodic_modulo (m : ℕ) (hm : m > 0) : ∃ K s : ℕ, ∀ k : ℕ, (K ≤ k → a_n (k) % m = a_n (k + s) % m) :=
sorry

end NUMINAMATH_GPT_sequence_eventually_periodic_modulo_l1822_182225


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1822_182215

theorem solution_set_of_inequality (x : ℝ) : 
  (3*x^2 - 4*x + 7 > 0) → (1 - 2*x) / (3*x^2 - 4*x + 7) ≥ 0 ↔ x ≤ 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1822_182215


namespace NUMINAMATH_GPT_area_triangle_BQW_l1822_182226

theorem area_triangle_BQW (ABCD : Rectangle) (AZ WC : ℝ) (AB : ℝ)
    (area_trapezoid_ZWCD : ℝ) :
    AZ = WC ∧ AZ = 6 ∧ AB = 12 ∧ area_trapezoid_ZWCD = 120 →
    (1/2) * ((120) - (1/2) * 6 * 12) = 42 :=
by
  intros
  sorry

end NUMINAMATH_GPT_area_triangle_BQW_l1822_182226


namespace NUMINAMATH_GPT_cost_of_purchase_l1822_182228

theorem cost_of_purchase :
  (5 * 3) + (8 * 2) = 31 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_purchase_l1822_182228


namespace NUMINAMATH_GPT_problem1_problem2_l1822_182278

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : cos B - 2 * cos A = (2 * a - b) * cos C / c)
variable (h2 : a = 2 * b)

theorem problem1 : a / b = 2 :=
by sorry

theorem problem2 (h3 : A > π / 2) (h4 : c = 3) : 0 < b ∧ b < 3 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1822_182278


namespace NUMINAMATH_GPT_evaluate_fraction_eq_10_pow_10_l1822_182229

noncomputable def evaluate_fraction (a b c : ℕ) : ℕ :=
  (a ^ 20) / ((a * b) ^ 10)

theorem evaluate_fraction_eq_10_pow_10 :
  evaluate_fraction 30 3 10 = 10 ^ 10 :=
by
  -- We define what is given and manipulate it directly to form a proof outline.
  sorry

end NUMINAMATH_GPT_evaluate_fraction_eq_10_pow_10_l1822_182229


namespace NUMINAMATH_GPT_quadratic_roots_l1822_182231

theorem quadratic_roots:
  ∀ x : ℝ, x^2 - 1 = 0 ↔ (x = -1 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1822_182231


namespace NUMINAMATH_GPT_greatest_possible_remainder_l1822_182235

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 9 ∧ x % 9 = r ∧ r = 8 :=
by
  use 8
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_greatest_possible_remainder_l1822_182235


namespace NUMINAMATH_GPT_students_participated_in_both_l1822_182233

theorem students_participated_in_both (total_students volleyball track field no_participation both: ℕ) 
  (h1 : total_students = 45) 
  (h2 : volleyball = 12) 
  (h3 : track = 20) 
  (h4 : no_participation = 19) 
  (h5 : both = volleyball + track - (total_students - no_participation)) 
  : both = 6 :=
by
  sorry

end NUMINAMATH_GPT_students_participated_in_both_l1822_182233


namespace NUMINAMATH_GPT_part1_part2_l1822_182206

-- Define the operation * on integers
def op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Prove that 2 * 3 = 7 given the defined operation
theorem part1 : op 2 3 = 7 := 
sorry

-- Prove that (-2) * (op 2 (-3)) = 1 given the defined operation
theorem part2 : op (-2) (op 2 (-3)) = 1 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1822_182206


namespace NUMINAMATH_GPT_find_borrowed_interest_rate_l1822_182237

theorem find_borrowed_interest_rate :
  ∀ (principal : ℝ) (time : ℝ) (lend_rate : ℝ) (gain_per_year : ℝ) (borrow_rate : ℝ),
  principal = 5000 →
  time = 1 → -- Considering per year
  lend_rate = 0.06 →
  gain_per_year = 100 →
  (principal * lend_rate - gain_per_year = principal * borrow_rate * time) →
  borrow_rate * 100 = 4 :=
by
  intros principal time lend_rate gain_per_year borrow_rate h_principal h_time h_lend_rate h_gain h_equation
  rw [h_principal, h_time, h_lend_rate] at h_equation
  have h_borrow_rate := h_equation
  sorry

end NUMINAMATH_GPT_find_borrowed_interest_rate_l1822_182237


namespace NUMINAMATH_GPT_cos_seventh_eq_sum_of_cos_l1822_182249

theorem cos_seventh_eq_sum_of_cos:
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
  (∀ θ : ℝ, (Real.cos θ) ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) ∧
  (b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 + b₆ ^ 2 + b₇ ^ 2 = 1555 / 4096) :=
sorry

end NUMINAMATH_GPT_cos_seventh_eq_sum_of_cos_l1822_182249


namespace NUMINAMATH_GPT_codger_feet_l1822_182241

theorem codger_feet (F : ℕ) (h1 : 6 = 2 * (5 - 1) * F) : F = 3 := by
  sorry

end NUMINAMATH_GPT_codger_feet_l1822_182241


namespace NUMINAMATH_GPT_largest_n_l1822_182238

def canBeFactored (A B : ℤ) : Bool :=
  A * B = 54

theorem largest_n (n : ℤ) (h : ∃ (A B : ℤ), canBeFactored A B ∧ 3 * B + A = n) :
  n = 163 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_l1822_182238


namespace NUMINAMATH_GPT_Hari_investment_contribution_l1822_182280

noncomputable def Praveen_investment : ℕ := 3780
noncomputable def Praveen_time : ℕ := 12
noncomputable def Hari_time : ℕ := 7
noncomputable def profit_ratio : ℚ := 2 / 3

theorem Hari_investment_contribution :
  ∃ H : ℕ, (Praveen_investment * Praveen_time) / (H * Hari_time) = (2 : ℕ) / 3 ∧ H = 9720 :=
by
  sorry

end NUMINAMATH_GPT_Hari_investment_contribution_l1822_182280


namespace NUMINAMATH_GPT_gcd_840_1764_l1822_182212

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 :=
by
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l1822_182212


namespace NUMINAMATH_GPT_part2_l1822_182262

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part2 (x y : ℝ) (h₁ : |x - y - 1| ≤ 1 / 3) (h₂ : |2 * y + 1| ≤ 1 / 6) :
  f x < 1 := 
by
  sorry

end NUMINAMATH_GPT_part2_l1822_182262


namespace NUMINAMATH_GPT_find_speed_of_A_l1822_182220

noncomputable def speed_of_A_is_7_5 (a : ℝ) : Prop :=
  -- Conditions
  ∃ (b : ℝ), b = a + 5 ∧ 
  (60 / a = 100 / b) → 
  -- Conclusion
  a = 7.5

-- Statement in Lean 4
theorem find_speed_of_A (a : ℝ) (h : speed_of_A_is_7_5 a) : a = 7.5 :=
  sorry

end NUMINAMATH_GPT_find_speed_of_A_l1822_182220


namespace NUMINAMATH_GPT_fraction_never_reducible_by_11_l1822_182297

theorem fraction_never_reducible_by_11 :
  ∀ (n : ℕ), Nat.gcd (1 + n) (3 + 7 * n) ≠ 11 := by
  sorry

end NUMINAMATH_GPT_fraction_never_reducible_by_11_l1822_182297


namespace NUMINAMATH_GPT_probability_of_suitcase_at_60th_position_expected_waiting_time_l1822_182210

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end NUMINAMATH_GPT_probability_of_suitcase_at_60th_position_expected_waiting_time_l1822_182210


namespace NUMINAMATH_GPT_problem_statement_l1822_182273

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1822_182273


namespace NUMINAMATH_GPT_power_six_sum_l1822_182289

theorem power_six_sum (x : ℝ) (h : x + 1 / x = 3) : x^6 + 1 / x^6 = 322 := 
by 
  sorry

end NUMINAMATH_GPT_power_six_sum_l1822_182289


namespace NUMINAMATH_GPT_abc_inequality_l1822_182281

theorem abc_inequality (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (h : a * b + a * c + b * c = a + b + c) : 
  a + b + c + 1 ≥ 4 * a * b * c :=
by 
  sorry

end NUMINAMATH_GPT_abc_inequality_l1822_182281


namespace NUMINAMATH_GPT_find_integers_with_sum_and_gcd_l1822_182247

theorem find_integers_with_sum_and_gcd {a b : ℕ} (h_sum : a + b = 104055) (h_gcd : Nat.gcd a b = 6937) :
  (a = 6937 ∧ b = 79118) ∨ (a = 13874 ∧ b = 90181) ∨ (a = 27748 ∧ b = 76307) ∨ (a = 48559 ∧ b = 55496) :=
sorry

end NUMINAMATH_GPT_find_integers_with_sum_and_gcd_l1822_182247


namespace NUMINAMATH_GPT_AM_GM_HY_order_l1822_182239

noncomputable def AM (a b c : ℝ) : ℝ := (a + b + c) / 3
noncomputable def GM (a b c : ℝ) : ℝ := (a * b * c)^(1/3)
noncomputable def HY (a b c : ℝ) : ℝ := 2 * a * b * c / (a * b + b * c + c * a)

theorem AM_GM_HY_order (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  AM a b c > GM a b c ∧ GM a b c > HY a b c := by
  sorry

end NUMINAMATH_GPT_AM_GM_HY_order_l1822_182239


namespace NUMINAMATH_GPT_num_pos_divisors_36_l1822_182257

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end NUMINAMATH_GPT_num_pos_divisors_36_l1822_182257


namespace NUMINAMATH_GPT_ranking_possibilities_l1822_182224

theorem ranking_possibilities (A B C D E : Type) : 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1 → n ≠ last)) →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1)) →
  ∃ (positions : Finset (List ℕ)),
    positions.card = 54 :=
by
  sorry

end NUMINAMATH_GPT_ranking_possibilities_l1822_182224


namespace NUMINAMATH_GPT_part1_part2_part3_l1822_182253

-- Define conditions
variables (n : ℕ) (h₁ : 5 ≤ n)

-- Problem part (1): Define p_n and prove its value
def p_n (n : ℕ) := (10 * n) / ((n + 5) * (n + 4))

-- Problem part (2): Define EX and prove its value for n = 5
def EX : ℚ := 5 / 3

-- Problem part (3): Prove n = 20 maximizes P
def P (n : ℕ) := 3 * ((p_n n) ^ 3 - 2 * (p_n n) ^ 2 + (p_n n))
def n_max := 20

-- Making the proof skeletons for clarity, filling in later
theorem part1 : p_n n = 10 * n / ((n + 5) * (n + 4)) :=
sorry

theorem part2 (h₂ : n = 5) : EX = 5 / 3 :=
sorry

theorem part3 : n_max = 20 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1822_182253


namespace NUMINAMATH_GPT_polynomial_division_quotient_l1822_182234

theorem polynomial_division_quotient :
  ∀ (x : ℝ), (x^5 - 21*x^3 + 8*x^2 - 17*x + 12) / (x - 3) = (x^4 + 3*x^3 - 12*x^2 - 28*x - 101) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_division_quotient_l1822_182234


namespace NUMINAMATH_GPT_smallest_difference_l1822_182203

noncomputable def triangle_lengths (DE EF FD : ℕ) : Prop :=
  DE < EF ∧ EF ≤ FD ∧ DE + EF + FD = 3010 ∧ DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference :
  ∃ (DE EF FD : ℕ), triangle_lengths DE EF FD ∧ EF - DE = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_difference_l1822_182203


namespace NUMINAMATH_GPT_solve_system_of_equations_l1822_182217

theorem solve_system_of_equations :
    ∃ x y : ℚ, 4 * x - 3 * y = 2 ∧ 6 * x + 5 * y = 1 ∧ x = 13 / 38 ∧ y = -4 / 19 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1822_182217


namespace NUMINAMATH_GPT_find_fixed_monthly_fee_l1822_182275

noncomputable def fixed_monthly_fee (f h : ℝ) (february_bill march_bill : ℝ) : Prop :=
  (f + h = february_bill) ∧ (f + 3 * h = march_bill)

theorem find_fixed_monthly_fee (h : ℝ):
  fixed_monthly_fee 13.44 h 20.72 35.28 :=
by 
  sorry

end NUMINAMATH_GPT_find_fixed_monthly_fee_l1822_182275


namespace NUMINAMATH_GPT_length_BC_l1822_182216

noncomputable def center (O : Type) : Prop := sorry   -- Center of the circle.

noncomputable def diameter (AD : Type) : Prop := sorry   -- AD is a diameter.

noncomputable def chord (ABC : Type) : Prop := sorry   -- ABC is a chord.

noncomputable def radius_equal (BO : ℝ) : Prop := BO = 8   -- BO = 8.

noncomputable def angle_ABO (α : ℝ) : Prop := α = 45   -- ∠ABO = 45°.

noncomputable def arc_CD (β : ℝ) : Prop := β = 90   -- Arc CD subtended by ∠AOD = 90°.

theorem length_BC (O AD ABC : Type) (BO : ℝ) (α β γ : ℝ)
  (h1 : center O)
  (h2 : diameter AD)
  (h3 : chord ABC)
  (h4 : radius_equal BO)
  (h5 : angle_ABO α)
  (h6 : arc_CD β)
  : γ = 8 := 
sorry

end NUMINAMATH_GPT_length_BC_l1822_182216


namespace NUMINAMATH_GPT_exponential_function_condition_l1822_182204

theorem exponential_function_condition (a : ℝ) (x : ℝ) 
  (h1 : a^2 - 5 * a + 5 = 1) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) : 
  a = 4 := 
sorry

end NUMINAMATH_GPT_exponential_function_condition_l1822_182204


namespace NUMINAMATH_GPT_area_of_win_sector_l1822_182222

theorem area_of_win_sector (r : ℝ) (p : ℝ) (A : ℝ) (h_1 : r = 10) (h_2 : p = 1 / 4) (h_3 : A = π * r^2) : 
  (p * A) = 25 * π := 
by
  sorry

end NUMINAMATH_GPT_area_of_win_sector_l1822_182222


namespace NUMINAMATH_GPT_final_stamp_collection_l1822_182259

section StampCollection

structure Collection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)
  (vehicles : ℕ)
  (famous_people : ℕ)

def initial_collections : Collection := {
  nature := 10, architecture := 15, animals := 12, vehicles := 6, famous_people := 4
}

-- define transactions as functions that take a collection and return a modified collection
def transaction1 (c : Collection) : Collection :=
  { c with nature := c.nature + 4, architecture := c.architecture + 5, animals := c.animals + 5, vehicles := c.vehicles + 2, famous_people := c.famous_people + 1 }

def transaction2 (c : Collection) : Collection := 
  { c with nature := c.nature + 2, animals := c.animals - 1 }

def transaction3 (c : Collection) : Collection := 
  { c with animals := c.animals - 5, architecture := c.architecture + 3 }

def transaction4 (c : Collection) : Collection :=
  { c with animals := c.animals - 4, nature := c.nature + 7 }

def transaction7 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles - 2, nature := c.nature + 5 }

def transaction8 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles + 3, famous_people := c.famous_people - 3 }

def final_collection (c : Collection) : Collection :=
  transaction8 (transaction7 (transaction4 (transaction3 (transaction2 (transaction1 c)))))

theorem final_stamp_collection :
  final_collection initial_collections = { nature := 28, architecture := 23, animals := 7, vehicles := 9, famous_people := 2 } :=
by
  -- skip the proof
  sorry

end StampCollection

end NUMINAMATH_GPT_final_stamp_collection_l1822_182259


namespace NUMINAMATH_GPT_inequality_product_geq_two_power_n_equality_condition_l1822_182246

open Real BigOperators

noncomputable def is_solution (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i ∧ a i = 1

theorem inequality_product_geq_two_power_n (a : ℕ → ℝ) (n : ℕ)
  (h1 : ( ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i))
  (h2 : ∑ i in Finset.range n, a (i + 1) = n) :
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) ≥ 2 ^ n :=
sorry

theorem equality_condition (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (h2 : ∑ i in Finset.range n, a (i + 1) = n):
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) = 2 ^ n ↔ is_solution a n :=
sorry

end NUMINAMATH_GPT_inequality_product_geq_two_power_n_equality_condition_l1822_182246


namespace NUMINAMATH_GPT_range_of_a_l1822_182263

variable (a : ℝ)

def p := ∀ x : ℝ, x^2 + a ≥ 0
def q := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (h : p a ∧ q a) : 0 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1822_182263


namespace NUMINAMATH_GPT_shaded_area_of_rotated_square_is_four_thirds_l1822_182277

noncomputable def common_shaded_area_of_rotated_square (β : ℝ) (h1 : 0 < β) (h2 : β < π / 2) (h_cos_beta : Real.cos β = 3 / 5) : ℝ :=
  let side_length := 2
  let area := side_length * side_length / 3 * 2
  area

theorem shaded_area_of_rotated_square_is_four_thirds
  (β : ℝ)
  (h1 : 0 < β)
  (h2 : β < π / 2)
  (h_cos_beta : Real.cos β = 3 / 5) :
  common_shaded_area_of_rotated_square β h1 h2 h_cos_beta = 4 / 3 :=
sorry

end NUMINAMATH_GPT_shaded_area_of_rotated_square_is_four_thirds_l1822_182277


namespace NUMINAMATH_GPT_min_edge_disjoint_cycles_l1822_182279

noncomputable def minEdgesForDisjointCycles (n : ℕ) (h : n ≥ 6) : ℕ := 3 * (n - 2)

theorem min_edge_disjoint_cycles (n : ℕ) (h : n ≥ 6) : minEdgesForDisjointCycles n h = 3 * (n - 2) := 
by
  sorry

end NUMINAMATH_GPT_min_edge_disjoint_cycles_l1822_182279


namespace NUMINAMATH_GPT_max_airlines_in_country_l1822_182242

-- Definition of the problem parameters
variable (N k : ℕ) 

-- Definition of the problem conditions
variable (hN_pos : 0 < N)
variable (hk_pos : 0 < k)
variable (hN_ge_k : k ≤ N)

-- Definition of the function calculating the maximum number of air routes
def max_air_routes (N k : ℕ) : ℕ :=
  Nat.choose N 2 - Nat.choose k 2

-- Theorem stating the maximum number of airlines given the conditions
theorem max_airlines_in_country (N k : ℕ) (hN_pos : 0 < N) (hk_pos : 0 < k) (hN_ge_k : k ≤ N) :
  max_air_routes N k = Nat.choose N 2 - Nat.choose k 2 :=
by sorry

end NUMINAMATH_GPT_max_airlines_in_country_l1822_182242


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l1822_182292

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + 3) 
  (h_geom : (a 1 + 6) ^ 2 = a 1 * (a 1 + 9)) : 
  a 2 = -9 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l1822_182292


namespace NUMINAMATH_GPT_remaining_storage_space_l1822_182251

/-- Given that 1 GB = 1024 MB, a hard drive with 300 GB of total storage,
and 300000 MB of used storage, prove that the remaining storage space is 7200 MB. -/
theorem remaining_storage_space (total_gb : ℕ) (mb_per_gb : ℕ) (used_mb : ℕ) :
  total_gb = 300 → mb_per_gb = 1024 → used_mb = 300000 →
  (total_gb * mb_per_gb - used_mb) = 7200 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_remaining_storage_space_l1822_182251


namespace NUMINAMATH_GPT_find_other_parallel_side_l1822_182296

theorem find_other_parallel_side 
  (a b d : ℝ) 
  (area : ℝ) 
  (h_area : area = 285) 
  (h_a : a = 20) 
  (h_d : d = 15)
  : (∃ x : ℝ, area = 1/2 * (a + x) * d ∧ x = 18) :=
by
  sorry

end NUMINAMATH_GPT_find_other_parallel_side_l1822_182296


namespace NUMINAMATH_GPT_marbles_problem_a_marbles_problem_b_l1822_182250

-- Define the problem as Lean statements.

-- Part (a): m = 2004, n = 2006
theorem marbles_problem_a (m n : ℕ) (h_m : m = 2004) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) := 
sorry

-- Part (b): m = 2005, n = 2006
theorem marbles_problem_b (m n : ℕ) (h_m : m = 2005) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) → false := 
sorry

end NUMINAMATH_GPT_marbles_problem_a_marbles_problem_b_l1822_182250


namespace NUMINAMATH_GPT_trajectory_equation_find_m_l1822_182243

-- Define points A and B.
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for P:
def P_condition (P : ℝ × ℝ) : Prop :=
  let PA_len := Real.sqrt ((P.1 - 1)^2 + P.2^2)
  let AB_len := Real.sqrt ((1 - (-1))^2 + (0 - 0)^2)
  let PB_dot_AB := (P.1 + 1) * (-2)
  PA_len * AB_len = PB_dot_AB

-- Problem (1): The trajectory equation
theorem trajectory_equation (P : ℝ × ℝ) (hP : P_condition P) : P.2^2 = 4 * P.1 :=
sorry

-- Define orthogonality condition
def orthogonal (M N : ℝ × ℝ) : Prop := 
  let OM := M
  let ON := N
  OM.1 * ON.1 + OM.2 * ON.2 = 0

-- Problem (2): Finding the value of m
theorem find_m (m : ℝ) (hm1 : m ≠ 0) (hm2 : m < 1) 
  (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (M N : ℝ × ℝ) (hM : M.2 = M.1 + m) (hN : N.2 = N.1 + m)
  (hMN : orthogonal M N) : m = -4 :=
sorry

end NUMINAMATH_GPT_trajectory_equation_find_m_l1822_182243


namespace NUMINAMATH_GPT_cubic_polynomial_evaluation_l1822_182255

theorem cubic_polynomial_evaluation (Q : ℚ → ℚ) (m : ℚ)
  (hQ0 : Q 0 = 2 * m) 
  (hQ1 : Q 1 = 5 * m) 
  (hQm1 : Q (-1) = 0) : 
  Q 2 + Q (-2) = 8 * m := 
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_evaluation_l1822_182255


namespace NUMINAMATH_GPT_more_bags_found_l1822_182211

def bags_Monday : ℕ := 7
def bags_nextDay : ℕ := 12

theorem more_bags_found : bags_nextDay - bags_Monday = 5 := by
  -- Proof Skipped
  sorry

end NUMINAMATH_GPT_more_bags_found_l1822_182211


namespace NUMINAMATH_GPT_initial_pennies_l1822_182221

theorem initial_pennies (P : ℕ)
  (h1 : P - (P / 2 + 1) = P / 2 - 1)
  (h2 : (P / 2 - 1) - (P / 4 + 1 / 2) = P / 4 - 3 / 2)
  (h3 : (P / 4 - 3 / 2) - (P / 8 + 3 / 4) = P / 8 - 9 / 4)
  (h4 : P / 8 - 9 / 4 = 1)
  : P = 26 := 
by
  sorry

end NUMINAMATH_GPT_initial_pennies_l1822_182221


namespace NUMINAMATH_GPT_find_S_l1822_182227

theorem find_S (a b : ℝ) (R : ℝ) (S : ℝ)
  (h1 : a + b = R) 
  (h2 : a^2 + b^2 = 12)
  (h3 : R = 2)
  (h4 : S = a^3 + b^3) : S = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_S_l1822_182227


namespace NUMINAMATH_GPT_mother_picked_38_carrots_l1822_182261

theorem mother_picked_38_carrots
  (haley_carrots : ℕ)
  (good_carrots : ℕ)
  (bad_carrots : ℕ)
  (total_carrots_picked : ℕ)
  (mother_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : good_carrots = 64)
  (h3 : bad_carrots = 13)
  (h4 : total_carrots_picked = good_carrots + bad_carrots)
  (h5 : total_carrots_picked = haley_carrots + mother_carrots) :
  mother_carrots = 38 :=
by
  sorry

end NUMINAMATH_GPT_mother_picked_38_carrots_l1822_182261


namespace NUMINAMATH_GPT_original_number_l1822_182258

theorem original_number (x : ℝ) (h1 : 74 * x = 19732) : x = 267 := by
  sorry

end NUMINAMATH_GPT_original_number_l1822_182258


namespace NUMINAMATH_GPT_product_of_primes_sum_ten_l1822_182295

theorem product_of_primes_sum_ten :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ Prime p1 ∧ Prime p2 ∧ p1 + p2 = 10 ∧ p1 * p2 = 21 := 
by
  sorry

end NUMINAMATH_GPT_product_of_primes_sum_ten_l1822_182295


namespace NUMINAMATH_GPT_total_value_of_gold_l1822_182288

theorem total_value_of_gold (legacy_bars : ℕ) (aleena_bars : ℕ) (bar_value : ℕ) (total_gold_value : ℕ) 
  (h1 : legacy_bars = 12) 
  (h2 : aleena_bars = legacy_bars - 4)
  (h3 : bar_value = 3500) : 
  total_gold_value = (legacy_bars + aleena_bars) * bar_value := 
by 
  sorry

end NUMINAMATH_GPT_total_value_of_gold_l1822_182288


namespace NUMINAMATH_GPT_investor_difference_l1822_182267

/-
Scheme A yields 30% of the capital within a year.
Scheme B yields 50% of the capital within a year.
Investor invested $300 in scheme A.
Investor invested $200 in scheme B.
We need to prove that the difference in total money between scheme A and scheme B after a year is $90.
-/

def schemeA_yield_rate : ℝ := 0.30
def schemeB_yield_rate : ℝ := 0.50
def schemeA_investment : ℝ := 300
def schemeB_investment : ℝ := 200

def total_after_year (investment : ℝ) (yield_rate : ℝ) : ℝ :=
  investment * (1 + yield_rate)

theorem investor_difference :
  total_after_year schemeA_investment schemeA_yield_rate - total_after_year schemeB_investment schemeB_yield_rate = 90 := by
  sorry

end NUMINAMATH_GPT_investor_difference_l1822_182267


namespace NUMINAMATH_GPT_corn_increase_factor_l1822_182214

noncomputable def field_area : ℝ := 1

-- Let x be the remaining part of the field
variable (x : ℝ)

-- First condition: if the remaining part is fully planted with millet
-- Millet will occupy half of the field
axiom condition1 : (field_area - x) + x = field_area / 2

-- Second condition: if the remaining part x is equally divided between oats and corn
-- Oats will occupy half of the field
axiom condition2 : (field_area - x) + 0.5 * x = field_area / 2

-- Prove the factor by which the amount of corn increases
theorem corn_increase_factor : (0.5 * x + x) / (0.5 * x / 2) = 3 :=
by
  sorry

end NUMINAMATH_GPT_corn_increase_factor_l1822_182214


namespace NUMINAMATH_GPT_wendi_owns_rabbits_l1822_182270

/-- Wendi's plot of land is 200 feet by 900 feet. -/
def area_land_in_feet : ℕ := 200 * 900

/-- One rabbit can eat enough grass to clear ten square yards of lawn area per day. -/
def rabbit_clear_per_day : ℕ := 10

/-- It would take 20 days for all of Wendi's rabbits to clear all the grass off of her grassland property. -/
def days_to_clear : ℕ := 20

/-- Convert feet to yards (3 feet in a yard). -/
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

/-- Calculate the total area of the land in square yards. -/
def area_land_in_yards : ℕ := (feet_to_yards 200) * (feet_to_yards 900)

theorem wendi_owns_rabbits (total_area : ℕ := area_land_in_yards)
                            (clear_area_per_rabbit : ℕ := rabbit_clear_per_day * days_to_clear) :
  total_area / clear_area_per_rabbit = 100 := 
sorry

end NUMINAMATH_GPT_wendi_owns_rabbits_l1822_182270


namespace NUMINAMATH_GPT_rope_lengths_l1822_182276

theorem rope_lengths (joey_len chad_len mandy_len : ℝ) (h1 : joey_len = 56) 
  (h2 : 8 / 3 = joey_len / chad_len) (h3 : 5 / 2 = chad_len / mandy_len) : 
  chad_len = 21 ∧ mandy_len = 8.4 :=
by
  sorry

end NUMINAMATH_GPT_rope_lengths_l1822_182276


namespace NUMINAMATH_GPT_apple_tree_total_apples_l1822_182245

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end NUMINAMATH_GPT_apple_tree_total_apples_l1822_182245


namespace NUMINAMATH_GPT_find_tricias_age_l1822_182236

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end NUMINAMATH_GPT_find_tricias_age_l1822_182236


namespace NUMINAMATH_GPT_project_work_time_ratio_l1822_182205

theorem project_work_time_ratio (A B C : ℕ) (h_ratio : A = x ∧ B = 2 * x ∧ C = 3 * x) (h_total : A + B + C = 120) : 
  (C - A = 40) :=
by
  sorry

end NUMINAMATH_GPT_project_work_time_ratio_l1822_182205


namespace NUMINAMATH_GPT_trip_time_difference_l1822_182232

-- Definitions of the given conditions
def speed_AB := 160 -- speed from A to B in km/h
def speed_BA := 120 -- speed from B to A in km/h
def distance_AB := 480 -- distance between A and B in km

-- Calculation of the time for each trip
def time_AB := distance_AB / speed_AB
def time_BA := distance_AB / speed_BA

-- The statement we need to prove
theorem trip_time_difference :
  (time_BA - time_AB) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trip_time_difference_l1822_182232


namespace NUMINAMATH_GPT_larger_of_two_numbers_l1822_182254

theorem larger_of_two_numbers (hcf : ℕ) (f1 : ℕ) (f2 : ℕ) 
(h_hcf : hcf = 10) 
(h_f1 : f1 = 11) 
(h_f2 : f2 = 15) 
: max (hcf * f1) (hcf * f2) = 150 :=
by
  have lcm := hcf * f1 * f2
  have num1 := hcf * f1
  have num2 := hcf * f2
  sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l1822_182254


namespace NUMINAMATH_GPT_distance_big_rock_correct_l1822_182252

noncomputable def rower_in_still_water := 7 -- km/h
noncomputable def river_flow := 2 -- km/h
noncomputable def total_trip_time := 1 -- hour

def distance_to_big_rock (D : ℝ) :=
  (D / (rower_in_still_water - river_flow)) + (D / (rower_in_still_water + river_flow)) = total_trip_time

theorem distance_big_rock_correct {D : ℝ} (h : distance_to_big_rock D) : D = 45 / 14 :=
sorry

end NUMINAMATH_GPT_distance_big_rock_correct_l1822_182252


namespace NUMINAMATH_GPT_opposite_of_2023_l1822_182285

def opposite (n x : ℤ) := n + x = 0 

theorem opposite_of_2023 : ∃ x : ℤ, opposite 2023 x ∧ x = -2023 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l1822_182285


namespace NUMINAMATH_GPT_octagon_reflected_arcs_area_l1822_182202

theorem octagon_reflected_arcs_area :
  let s := 2
  let θ := 45
  let r := 2 / Real.sqrt (2 - Real.sqrt (2))
  let sector_area := θ / 360 * Real.pi * r^2
  let total_arc_area := 8 * sector_area
  let circle_area := Real.pi * r^2
  let bounded_region_area := 8 * (circle_area - 2 * Real.sqrt (2) * 1 / 2)
  bounded_region_area = (16 * Real.sqrt 2 / 3 - Real.pi)
:= sorry

end NUMINAMATH_GPT_octagon_reflected_arcs_area_l1822_182202


namespace NUMINAMATH_GPT_probability_same_color_dice_l1822_182284

theorem probability_same_color_dice :
  let total_sides := 12
  let red_sides := 3
  let green_sides := 4
  let blue_sides := 2
  let yellow_sides := 3
  let prob_red := (red_sides / total_sides) ^ 2
  let prob_green := (green_sides / total_sides) ^ 2
  let prob_blue := (blue_sides / total_sides) ^ 2
  let prob_yellow := (yellow_sides / total_sides) ^ 2
  prob_red + prob_green + prob_blue + prob_yellow = 19 / 72 := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_probability_same_color_dice_l1822_182284


namespace NUMINAMATH_GPT_nancy_antacids_l1822_182299

theorem nancy_antacids :
  ∀ (x : ℕ),
  (3 * 3 + x * 2 + 1 * 2) * 4 = 60 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_nancy_antacids_l1822_182299


namespace NUMINAMATH_GPT_range_of_m_l1822_182274

noncomputable def f (x : ℝ) := |x - 3| - 2
noncomputable def g (x : ℝ) := -|x + 1| + 4

theorem range_of_m (m : ℝ) : (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1822_182274


namespace NUMINAMATH_GPT_final_price_is_correct_l1822_182201

-- Define the original price and the discount rate
variable (a : ℝ)

-- The final price of the product after two 10% discounts
def final_price_after_discounts (a : ℝ) : ℝ :=
  a * (0.9 ^ 2)

-- Theorem stating the final price after two consecutive 10% discounts
theorem final_price_is_correct (a : ℝ) :
  final_price_after_discounts a = a * (0.9 ^ 2) :=
by sorry

end NUMINAMATH_GPT_final_price_is_correct_l1822_182201


namespace NUMINAMATH_GPT_area_of_quadrilateral_l1822_182264

theorem area_of_quadrilateral (A B C : ℝ) (h1 : A + B = C) (h2 : A = 16) (h3 : B = 16) :
  (C - A - B) / 2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l1822_182264


namespace NUMINAMATH_GPT_min_value_of_mn_squared_l1822_182200

theorem min_value_of_mn_squared 
  (a b c : ℝ) 
  (h : a^2 + b^2 = c^2) 
  (m n : ℝ) 
  (h_point : a * m + b * n + 2 * c = 0) : 
  m^2 + n^2 = 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_mn_squared_l1822_182200


namespace NUMINAMATH_GPT_correctness_of_statements_l1822_182268

theorem correctness_of_statements :
  (statement1 ∧ statement4 ∧ statement5) :=
by sorry

end NUMINAMATH_GPT_correctness_of_statements_l1822_182268


namespace NUMINAMATH_GPT_total_splash_width_l1822_182240

def pebbles : ℚ := 1/5
def rocks : ℚ := 2/5
def boulders : ℚ := 7/5
def mini_boulders : ℚ := 4/5
def large_pebbles : ℚ := 3/5

def num_pebbles : ℚ := 10
def num_rocks : ℚ := 5
def num_boulders : ℚ := 4
def num_mini_boulders : ℚ := 3
def num_large_pebbles : ℚ := 7

theorem total_splash_width : 
  num_pebbles * pebbles + 
  num_rocks * rocks + 
  num_boulders * boulders + 
  num_mini_boulders * mini_boulders + 
  num_large_pebbles * large_pebbles = 16.2 := by
  sorry

end NUMINAMATH_GPT_total_splash_width_l1822_182240


namespace NUMINAMATH_GPT_simplify_expression_l1822_182286

theorem simplify_expression :
  ((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4) = 125 / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1822_182286


namespace NUMINAMATH_GPT_first_group_number_l1822_182209

variable (x : ℕ)

def number_of_first_group :=
  x = 6

theorem first_group_number (H1 : ∀ k : ℕ, k = 8 * 15 + x)
                          (H2 : k = 126) : 
                          number_of_first_group x :=
by
  sorry

end NUMINAMATH_GPT_first_group_number_l1822_182209


namespace NUMINAMATH_GPT_system_solution_l1822_182207

theorem system_solution (x : Fin 1995 → ℤ) :
  (∀ i : (Fin 1995),
    x (i + 1) ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) →
  (∀ n : (Fin 1995),
    (x n = 0 ∧ x (n + 1) = 1 ∧ x (n + 2) = -1) ∨
    (x n = 0 ∧ x (n + 1) = -1 ∧ x (n + 2) = 1)) :=
by sorry

end NUMINAMATH_GPT_system_solution_l1822_182207


namespace NUMINAMATH_GPT_necessary_conditions_l1822_182271

theorem necessary_conditions (a b c d e : ℝ) (h : (a + b + e) / (b + c) = (c + d + e) / (d + a)) :
  a = c ∨ a + b + c + d + e = 0 :=
by
  sorry

end NUMINAMATH_GPT_necessary_conditions_l1822_182271


namespace NUMINAMATH_GPT_largest_possible_b_b_eq_4_of_largest_l1822_182208

theorem largest_possible_b (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) : b ≤ 4 := by
  sorry

theorem b_eq_4_of_largest (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) (hb : b = 4) : True := by
  sorry

end NUMINAMATH_GPT_largest_possible_b_b_eq_4_of_largest_l1822_182208


namespace NUMINAMATH_GPT_valentines_proof_l1822_182218

-- Definitions of the conditions in the problem
def original_valentines : ℝ := 58.5
def remaining_valentines : ℝ := 16.25
def valentines_given : ℝ := 42.25

-- The statement that we need to prove
theorem valentines_proof : original_valentines - remaining_valentines = valentines_given := by
  sorry

end NUMINAMATH_GPT_valentines_proof_l1822_182218


namespace NUMINAMATH_GPT_associate_professor_charts_l1822_182266

theorem associate_professor_charts (A B C : ℕ) : 
  A + B = 8 → 
  2 * A + B = 10 → 
  C * A + 2 * B = 14 → 
  C = 1 := 
by 
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_associate_professor_charts_l1822_182266


namespace NUMINAMATH_GPT_contrapositive_example_l1822_182230

theorem contrapositive_example (α : ℝ) : (α = Real.pi / 3 → Real.cos α = 1 / 2) → (Real.cos α ≠ 1 / 2 → α ≠ Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l1822_182230


namespace NUMINAMATH_GPT_sequence_sum_129_l1822_182291

/-- 
  In an increasing sequence of four positive integers where the first three terms form an arithmetic
  progression and the last three terms form a geometric progression, and where the first and fourth
  terms differ by 30, the sum of the four terms is 129.
-/
theorem sequence_sum_129 :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a < a + d) ∧ (a + d < a + 2 * d) ∧ 
    (a + 2 * d < a + 30) ∧ 30 = (a + 30) - a ∧ 
    (a + d) * (a + 30) = (a + 2 * d) ^ 2 ∧ 
    a + (a + d) + (a + 2 * d) + (a + 30) = 129 :=
sorry

end NUMINAMATH_GPT_sequence_sum_129_l1822_182291


namespace NUMINAMATH_GPT_range_of_a_l1822_182298

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 1 < a ∧ a < 2 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_range_of_a_l1822_182298


namespace NUMINAMATH_GPT_Denise_age_l1822_182260

-- Define the ages of Amanda, Carlos, Beth, and Denise
variables (A C B D : ℕ)

-- State the given conditions
def condition1 := A = C - 4
def condition2 := C = B + 5
def condition3 := D = B + 2
def condition4 := A = 16

-- The theorem to prove
theorem Denise_age (A C B D : ℕ) (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 D B) (h4 : condition4 A) : D = 17 :=
by
  sorry

end NUMINAMATH_GPT_Denise_age_l1822_182260


namespace NUMINAMATH_GPT_no_real_solution_l1822_182269

theorem no_real_solution (x : ℝ) : ¬ ∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -|x| := 
sorry

end NUMINAMATH_GPT_no_real_solution_l1822_182269


namespace NUMINAMATH_GPT_problem_proof_l1822_182265

noncomputable def triangle_expression (a b c : ℝ) (A B C : ℝ) : ℝ :=
  b^2 * (Real.cos (C / 2))^2 + c^2 * (Real.cos (B / 2))^2 + 
  2 * b * c * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2)

theorem problem_proof (a b c A B C : ℝ) (h1 : a + b + c = 16) : 
  triangle_expression a b c A B C = 64 := 
sorry

end NUMINAMATH_GPT_problem_proof_l1822_182265


namespace NUMINAMATH_GPT_vinegar_solution_concentration_l1822_182290

theorem vinegar_solution_concentration
  (original_volume : ℝ) (water_volume : ℝ)
  (original_concentration : ℝ)
  (h1 : original_volume = 12)
  (h2 : water_volume = 50)
  (h3 : original_concentration = 36.166666666666664) :
  original_concentration / 100 * original_volume / (original_volume + water_volume) = 0.07 :=
by
  sorry

end NUMINAMATH_GPT_vinegar_solution_concentration_l1822_182290


namespace NUMINAMATH_GPT_number_of_pairs_of_positive_integers_l1822_182283

theorem number_of_pairs_of_positive_integers 
    {m n : ℕ} (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m > n) (h_diff : m^2 - n^2 = 144) : 
    ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧ (∀ p ∈ pairs, p.1 > p.2 ∧ p.1^2 - p.2^2 = 144) :=
sorry

end NUMINAMATH_GPT_number_of_pairs_of_positive_integers_l1822_182283
