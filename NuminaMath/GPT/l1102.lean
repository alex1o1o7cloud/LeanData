import Mathlib

namespace NUMINAMATH_GPT_second_printer_cost_l1102_110250

theorem second_printer_cost (p1_cost : ℕ) (num_units : ℕ) (total_spent : ℕ) (x : ℕ) 
  (h1 : p1_cost = 375) 
  (h2 : num_units = 7) 
  (h3 : total_spent = p1_cost * num_units) 
  (h4 : total_spent = x * num_units) : 
  x = 375 := 
sorry

end NUMINAMATH_GPT_second_printer_cost_l1102_110250


namespace NUMINAMATH_GPT_least_N_l1102_110211

theorem least_N :
  ∃ N : ℕ, 
    (N % 2 = 1) ∧ 
    (N % 3 = 2) ∧ 
    (N % 5 = 3) ∧ 
    (N % 7 = 4) ∧ 
    (∀ M : ℕ, 
      (M % 2 = 1) ∧ 
      (M % 3 = 2) ∧ 
      (M % 5 = 3) ∧ 
      (M % 7 = 4) → 
      N ≤ M) :=
  sorry

end NUMINAMATH_GPT_least_N_l1102_110211


namespace NUMINAMATH_GPT_quadratic_real_roots_condition_l1102_110287

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m-1) * x₁^2 - 4 * x₁ + 1 = 0 ∧ (m-1) * x₂^2 - 4 * x₂ + 1 = 0) ↔ (m < 5 ∧ m ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_condition_l1102_110287


namespace NUMINAMATH_GPT_total_distance_crawled_l1102_110292

theorem total_distance_crawled :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let pos4 := 0
  abs (pos2 - pos1) + abs (pos3 - pos2) + abs (pos4 - pos3) = 29 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_crawled_l1102_110292


namespace NUMINAMATH_GPT_firm_partners_l1102_110225

theorem firm_partners
  (P A : ℕ)
  (h1 : P / A = 2 / 63)
  (h2 : P / (A + 35) = 1 / 34) :
  P = 14 :=
by
  sorry

end NUMINAMATH_GPT_firm_partners_l1102_110225


namespace NUMINAMATH_GPT_kennedy_lost_pawns_l1102_110288

-- Definitions based on conditions
def initial_pawns_per_player := 8
def total_pawns := 2 * initial_pawns_per_player -- Total pawns in the game initially
def pawns_lost_by_Riley := 1 -- Riley lost 1 pawn
def pawns_remaining := 11 -- 11 pawns left in the game

-- Translations of conditions to Lean
theorem kennedy_lost_pawns : 
  initial_pawns_per_player - (pawns_remaining - (initial_pawns_per_player - pawns_lost_by_Riley)) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_kennedy_lost_pawns_l1102_110288


namespace NUMINAMATH_GPT_gcd_triples_l1102_110294

theorem gcd_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  gcd a 20 = b ∧ gcd b 15 = c ∧ gcd a c = 5 ↔
  ∃ t : ℕ, t > 0 ∧ 
    ((a = 20 * t ∧ b = 20 ∧ c = 5) ∨ 
     (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨ 
     (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by
  sorry

end NUMINAMATH_GPT_gcd_triples_l1102_110294


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l1102_110220

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period_of_f_is_pi : 
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ ε > 0, ε < Real.pi → ∃ x, f (x + ε) ≠ f x) :=
by
  sorry

theorem f_at_pi_over_2_not_sqrt_3_over_2 : f (Real.pi / 2) ≠ Real.sqrt 3 / 2 :=
by
  sorry

theorem max_value_of_f_on_interval : 
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l1102_110220


namespace NUMINAMATH_GPT_constant_term_binomial_l1102_110232

theorem constant_term_binomial (n : ℕ) (h : n = 5) : ∃ (r : ℕ), r = 6 ∧ (Nat.choose (2 * n) r) = 210 := by
  sorry

end NUMINAMATH_GPT_constant_term_binomial_l1102_110232


namespace NUMINAMATH_GPT_sum_geometric_series_l1102_110242

noncomputable def S_n (n : ℕ) : ℝ :=
  3 - 3 * ((2 / 3)^n)

theorem sum_geometric_series (a : ℝ) (r : ℝ) (n : ℕ) (h_a : a = 1) (h_r : r = 2 / 3) :
  S_n n = a * (1 - r^n) / (1 - r) :=
by
  sorry

end NUMINAMATH_GPT_sum_geometric_series_l1102_110242


namespace NUMINAMATH_GPT_quadruple_perimeter_l1102_110284

-- Define the rectangle's original and expanded dimensions and perimeters
def original_perimeter (a b : ℝ) := 2 * (a + b)
def new_perimeter (a b : ℝ) := 2 * ((4 * a) + (4 * b))

-- Statement to be proved
theorem quadruple_perimeter (a b : ℝ) : new_perimeter a b = 4 * original_perimeter a b :=
  sorry

end NUMINAMATH_GPT_quadruple_perimeter_l1102_110284


namespace NUMINAMATH_GPT_jack_morning_emails_l1102_110273

-- Define the conditions as constants
def totalEmails : ℕ := 10
def emailsAfternoon : ℕ := 3
def emailsEvening : ℕ := 1

-- Problem statement to prove emails in the morning
def emailsMorning : ℕ := totalEmails - (emailsAfternoon + emailsEvening)

-- The theorem to prove
theorem jack_morning_emails : emailsMorning = 6 := by
  sorry

end NUMINAMATH_GPT_jack_morning_emails_l1102_110273


namespace NUMINAMATH_GPT_min_max_x_l1102_110277

-- Definitions for the initial conditions and surveys
def students : ℕ := 100
def like_math_initial : ℕ := 50
def dislike_math_initial : ℕ := 50
def like_math_final : ℕ := 60
def dislike_math_final : ℕ := 40

-- Variables for the students' responses
variables (a b c d : ℕ)

-- Conditions based on the problem statement
def initial_survey : Prop := a + d = like_math_initial ∧ b + c = dislike_math_initial
def final_survey : Prop := a + c = like_math_final ∧ b + d = dislike_math_final

-- Definition of x as the number of students who changed their answer
def x : ℕ := c + d

-- Prove the minimum and maximum value of x with given conditions
theorem min_max_x (a b c d : ℕ) 
  (initial_cond : initial_survey a b c d)
  (final_cond : final_survey a b c d)
  : 10 ≤ (x c d) ∧ (x c d) ≤ 90 :=
by
  -- This is where the proof would go, but we'll simply state sorry for now.
  sorry

end NUMINAMATH_GPT_min_max_x_l1102_110277


namespace NUMINAMATH_GPT_emily_chairs_count_l1102_110285

theorem emily_chairs_count 
  (C : ℕ) 
  (T : ℕ) 
  (time_per_furniture : ℕ)
  (total_time : ℕ) 
  (hT : T = 2) 
  (h_time : time_per_furniture = 8) 
  (h_total : 8 * C + 8 * T = 48) : 
  C = 4 := by
    sorry

end NUMINAMATH_GPT_emily_chairs_count_l1102_110285


namespace NUMINAMATH_GPT_cost_price_of_article_l1102_110286

-- Define the conditions and goal as a Lean 4 statement
theorem cost_price_of_article (M C : ℝ) (h1 : 0.95 * M = 75) (h2 : 1.25 * C = 75) : 
  C = 60 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l1102_110286


namespace NUMINAMATH_GPT_minimum_xy_l1102_110223

noncomputable def f (x y : ℝ) := 2 * x + y + 6

theorem minimum_xy (x y : ℝ) (h : 0 < x ∧ 0 < y) (h1 : f x y = x * y) : x * y = 18 :=
by
  sorry

end NUMINAMATH_GPT_minimum_xy_l1102_110223


namespace NUMINAMATH_GPT_A_doubles_after_6_months_l1102_110293

variable (x : ℕ)

def A_investment_share (x : ℕ) := (3000 * x) + (6000 * (12 - x))
def B_investment_share := 4500 * 12

theorem A_doubles_after_6_months (h : A_investment_share x = B_investment_share) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_A_doubles_after_6_months_l1102_110293


namespace NUMINAMATH_GPT_find_k_l1102_110233

theorem find_k (k : ℝ) (α β : ℝ) 
  (h1 : α + β = -k) 
  (h2 : α * β = 12) 
  (h3 : α + 7 + β + 7 = k) : 
  k = -7 :=
sorry

end NUMINAMATH_GPT_find_k_l1102_110233


namespace NUMINAMATH_GPT_tan_double_alpha_l1102_110291

theorem tan_double_alpha (α : ℝ) (h : ∀ x : ℝ, (3 * Real.sin x + Real.cos x) ≤ (3 * Real.sin α + Real.cos α)) :
  Real.tan (2 * α) = -3 / 4 :=
sorry

end NUMINAMATH_GPT_tan_double_alpha_l1102_110291


namespace NUMINAMATH_GPT_like_terms_exponents_l1102_110268

theorem like_terms_exponents (m n : ℕ) (x y : ℝ) (h : 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) : m = 4 ∧ n = 3 :=
by 
  sorry

end NUMINAMATH_GPT_like_terms_exponents_l1102_110268


namespace NUMINAMATH_GPT_lisa_flight_time_l1102_110214

noncomputable def distance : ℝ := 519.5
noncomputable def speed : ℝ := 54.75
noncomputable def time : ℝ := 9.49

theorem lisa_flight_time : distance / speed = time :=
by
  sorry

end NUMINAMATH_GPT_lisa_flight_time_l1102_110214


namespace NUMINAMATH_GPT_marked_elements_duplicate_l1102_110221

open Nat

def table : Matrix (Fin 4) (Fin 10) ℕ := ![
  ![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
  ![9, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
  ![8, 9, 0, 1, 2, 3, 4, 5, 6, 7], 
  ![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
]

theorem marked_elements_duplicate 
  (marked : Fin 4 → Fin 10) 
  (h_marked_unique_row : ∀ i1 i2, i1 ≠ i2 → marked i1 ≠ marked i2)
  (h_marked_unique_col : ∀ j, ∃ i, marked i = j) :
  ∃ i1 i2, i1 ≠ i2 ∧ table i1 (marked i1) = table i2 (marked i2) := sorry

end NUMINAMATH_GPT_marked_elements_duplicate_l1102_110221


namespace NUMINAMATH_GPT_equal_serving_weight_l1102_110283

theorem equal_serving_weight (total_weight : ℝ) (num_family_members : ℕ)
  (h1 : total_weight = 13) (h2 : num_family_members = 5) :
  total_weight / num_family_members = 2.6 :=
by
  sorry

end NUMINAMATH_GPT_equal_serving_weight_l1102_110283


namespace NUMINAMATH_GPT_min_value_of_f_l1102_110281

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 16) + Real.sqrt ((x + 1)^2 + 9))

theorem min_value_of_f :
  ∃ (x : ℝ), f x = 5 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_min_value_of_f_l1102_110281


namespace NUMINAMATH_GPT_smallest_b_l1102_110279

theorem smallest_b (a b : ℕ) (pos_a : 0 < a) (pos_b : 0 < b)
    (h1 : a - b = 4)
    (h2 : gcd ((a^3 + b^3) / (a + b)) (a * b) = 4) : b = 2 :=
sorry

end NUMINAMATH_GPT_smallest_b_l1102_110279


namespace NUMINAMATH_GPT_cashier_amount_l1102_110266

def amount_to_cashier (discount : ℝ) (shorts_count : ℕ) (shorts_price : ℕ) (shirts_count : ℕ) (shirts_price : ℕ) : ℝ :=
  let total_cost := (shorts_count * shorts_price) + (shirts_count * shirts_price)
  let discount_amount := discount * total_cost
  total_cost - discount_amount

theorem cashier_amount : amount_to_cashier 0.1 3 15 5 17 = 117 := 
by
  sorry

end NUMINAMATH_GPT_cashier_amount_l1102_110266


namespace NUMINAMATH_GPT_class_8_1_total_score_l1102_110299

noncomputable def total_score (spirit neatness standard_of_movements : ℝ) 
(weights_spirit weights_neatness weights_standard : ℝ) : ℝ :=
  (spirit * weights_spirit + neatness * weights_neatness + standard_of_movements * weights_standard) / 
  (weights_spirit + weights_neatness + weights_standard)

theorem class_8_1_total_score :
  total_score 8 9 10 2 3 5 = 9.3 :=
by
  sorry

end NUMINAMATH_GPT_class_8_1_total_score_l1102_110299


namespace NUMINAMATH_GPT_min_value_x_plus_inv_x_l1102_110237

open Real

theorem min_value_x_plus_inv_x (x : ℝ) (hx : 0 < x) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_inv_x_l1102_110237


namespace NUMINAMATH_GPT_successful_multiplications_in_one_hour_l1102_110213

variable (multiplications_per_second : ℕ)
variable (error_rate_percentage : ℕ)

theorem successful_multiplications_in_one_hour
  (h1 : multiplications_per_second = 15000)
  (h2 : error_rate_percentage = 5)
  : (multiplications_per_second * 3600 * (100 - error_rate_percentage) / 100) 
    + (multiplications_per_second * 3600 * error_rate_percentage / 100) = 54000000 := by
  sorry

end NUMINAMATH_GPT_successful_multiplications_in_one_hour_l1102_110213


namespace NUMINAMATH_GPT_exists_non_prime_form_l1102_110241

theorem exists_non_prime_form (n : ℕ) : ∃ n : ℕ, ¬Nat.Prime (n^2 + n + 41) :=
sorry

end NUMINAMATH_GPT_exists_non_prime_form_l1102_110241


namespace NUMINAMATH_GPT_factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l1102_110208

theorem factorization_A (x y : ℝ) : x^2 - 2 * x * y = x * (x - 2 * y) :=
  by sorry

theorem factorization_B (x y : ℝ) : x^2 - 25 * y^2 = (x - 5 * y) * (x + 5 * y) :=
  by sorry

theorem factorization_C (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 :=
  by sorry

theorem factorization_D_incorrect (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) :=
  by sorry

theorem factorization_D_correct (x : ℝ) : x^2 + x - 2 = (x + 2) * (x - 1) :=
  by sorry

end NUMINAMATH_GPT_factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l1102_110208


namespace NUMINAMATH_GPT_parabola_chords_reciprocal_sum_l1102_110222

theorem parabola_chords_reciprocal_sum (x y : ℝ) (AB CD : ℝ) (p : ℝ) :
  (y = (4 : ℝ) * x) ∧ (AB ≠ 0) ∧ (CD ≠ 0) ∧
  (p = (2 : ℝ)) ∧
  (|AB| = (2 * p / (Real.sin (Real.pi / 4))^2)) ∧ 
  (|CD| = (2 * p / (Real.cos (Real.pi / 4))^2)) →
  (1 / |AB| + 1 / |CD| = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_chords_reciprocal_sum_l1102_110222


namespace NUMINAMATH_GPT_sequence_general_term_l1102_110264

theorem sequence_general_term {a : ℕ → ℝ} (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 4 * a n - 3) :
  a n = (4/3)^(n-1) :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l1102_110264


namespace NUMINAMATH_GPT_minimum_questions_needed_to_determine_birthday_l1102_110280

def min_questions_to_determine_birthday : Nat := 9

theorem minimum_questions_needed_to_determine_birthday : min_questions_to_determine_birthday = 9 :=
sorry

end NUMINAMATH_GPT_minimum_questions_needed_to_determine_birthday_l1102_110280


namespace NUMINAMATH_GPT_inequality_range_l1102_110298

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem inequality_range (a b x: ℝ) (h : a ≠ 0) :
  (|a + b| + |a - b|) ≥ |a| * f x → 1 ≤ x ∧ x ≤ 2 :=
by
  intro h1
  unfold f at h1
  sorry

end NUMINAMATH_GPT_inequality_range_l1102_110298


namespace NUMINAMATH_GPT_sec_120_eq_neg_2_l1102_110230

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_120_eq_neg_2 : sec 120 = -2 := by
  sorry

end NUMINAMATH_GPT_sec_120_eq_neg_2_l1102_110230


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_l1102_110270

theorem arithmetic_sequence_formula (x : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = x - 1) (h2 : a 2 = x + 1) (h3 : a 3 = 2 * x + 3) :
  ∃ c d : ℤ, (∀ n : ℕ, a n = c + d * (n - 1)) ∧ ∀ n : ℕ, a n = 2 * n - 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_formula_l1102_110270


namespace NUMINAMATH_GPT_proof_least_sum_l1102_110216

noncomputable def least_sum (m n : ℕ) (h1 : Nat.gcd (m + n) 330 = 1) 
                           (h2 : n^n ∣ m^m) (h3 : ¬(n ∣ m)) : ℕ :=
  m + n

theorem proof_least_sum :
  ∃ m n : ℕ, Nat.gcd (m + n) 330 = 1 ∧ n^n ∣ m^m ∧ ¬(n ∣ m) ∧ m + n = 390 :=
by
  sorry

end NUMINAMATH_GPT_proof_least_sum_l1102_110216


namespace NUMINAMATH_GPT_perimeter_of_equilateral_triangle_l1102_110246

theorem perimeter_of_equilateral_triangle (a : ℕ) (h1 : a = 12) (h2 : ∀ sides, sides = 3) : 
  3 * a = 36 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_equilateral_triangle_l1102_110246


namespace NUMINAMATH_GPT_simplest_square_root_l1102_110219

noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def inv_sqrt2 : ℝ := 1 / Real.sqrt 2
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt_inv2 : ℝ := Real.sqrt (1 / 2)

theorem simplest_square_root : sqrt2 = Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_simplest_square_root_l1102_110219


namespace NUMINAMATH_GPT_arccos_proof_l1102_110239

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end NUMINAMATH_GPT_arccos_proof_l1102_110239


namespace NUMINAMATH_GPT_dot_product_calculation_l1102_110210

def vec_a : ℝ × ℝ := (1, 0)
def vec_b : ℝ × ℝ := (2, 3)
def vec_s : ℝ × ℝ := (2 * vec_a.1 - vec_b.1, 2 * vec_a.2 - vec_b.2)
def vec_t : ℝ × ℝ := (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_calculation :
  dot_product vec_s vec_t = -9 := by
  sorry

end NUMINAMATH_GPT_dot_product_calculation_l1102_110210


namespace NUMINAMATH_GPT_total_budget_l1102_110249

-- Define the conditions for the problem
def fiscal_months : ℕ := 12
def total_spent_at_six_months : ℕ := 6580
def over_budget_at_six_months : ℕ := 280

-- Calculate the total budget for the project
theorem total_budget (budget : ℕ) 
  (h : 6 * (total_spent_at_six_months - over_budget_at_six_months) * 2 = budget) 
  : budget = 12600 := 
  by
    -- Proof will be here
    sorry

end NUMINAMATH_GPT_total_budget_l1102_110249


namespace NUMINAMATH_GPT_tan_add_tan_105_eq_l1102_110262

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end NUMINAMATH_GPT_tan_add_tan_105_eq_l1102_110262


namespace NUMINAMATH_GPT_seashells_found_l1102_110201

theorem seashells_found (C B : ℤ) (h1 : 9 * B = 7 * C) (h2 : B = C - 12) : C = 54 :=
by
  sorry

end NUMINAMATH_GPT_seashells_found_l1102_110201


namespace NUMINAMATH_GPT_sequence_value_a8_b8_l1102_110206

theorem sequence_value_a8_b8
(a b : ℝ) 
(h1 : a + b = 1) 
(h2 : a^2 + b^2 = 3) 
(h3 : a^3 + b^3 = 4) 
(h4 : a^4 + b^4 = 7) 
(h5 : a^5 + b^5 = 11) 
(h6 : a^6 + b^6 = 18) : 
a^8 + b^8 = 47 :=
sorry

end NUMINAMATH_GPT_sequence_value_a8_b8_l1102_110206


namespace NUMINAMATH_GPT_unique_solution_condition_l1102_110261

theorem unique_solution_condition {a b : ℝ} : (∃ x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l1102_110261


namespace NUMINAMATH_GPT_maximize_side_area_of_cylinder_l1102_110240

noncomputable def radius_of_cylinder (x : ℝ) : ℝ :=
  (6 - x) / 3

noncomputable def side_area_of_cylinder (x : ℝ) : ℝ :=
  2 * Real.pi * (radius_of_cylinder x) * x

theorem maximize_side_area_of_cylinder :
  ∃ x : ℝ, (0 < x ∧ x < 6) ∧ (∀ y : ℝ, (0 < y ∧ y < 6) → (side_area_of_cylinder y ≤ side_area_of_cylinder x)) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_maximize_side_area_of_cylinder_l1102_110240


namespace NUMINAMATH_GPT_problem_solved_probability_l1102_110231

theorem problem_solved_probability :
  let PA := 1 / 2
  let PB := 1 / 3
  let PC := 1 / 4
  1 - ((1 - PA) * (1 - PB) * (1 - PC)) = 3 / 4 := 
sorry

end NUMINAMATH_GPT_problem_solved_probability_l1102_110231


namespace NUMINAMATH_GPT_pat_more_hours_than_jane_l1102_110205

theorem pat_more_hours_than_jane (H P K M J : ℝ) 
  (h_total : H = P + K + M + J)
  (h_pat : P = 2 * K)
  (h_mark : M = (1/3) * P)
  (h_jane : J = (1/2) * M)
  (H290 : H = 290) :
  P - J = 120.83 := 
by
  sorry

end NUMINAMATH_GPT_pat_more_hours_than_jane_l1102_110205


namespace NUMINAMATH_GPT_age_30_years_from_now_l1102_110209

variables (ElderSonAge : ℕ) (DeclanAgeDiff : ℕ) (YoungerSonAgeDiff : ℕ) (ThirdSiblingAgeDiff : ℕ)

-- Given conditions
def elder_son_age : ℕ := 40
def declan_age : ℕ := elder_son_age + 25
def younger_son_age : ℕ := elder_son_age - 10
def third_sibling_age : ℕ := younger_son_age - 5

-- To prove the ages 30 years from now
def younger_son_age_30_years_from_now : ℕ := younger_son_age + 30
def third_sibling_age_30_years_from_now : ℕ := third_sibling_age + 30

-- The proof statement
theorem age_30_years_from_now : 
  younger_son_age_30_years_from_now = 60 ∧ 
  third_sibling_age_30_years_from_now = 55 :=
by
  sorry

end NUMINAMATH_GPT_age_30_years_from_now_l1102_110209


namespace NUMINAMATH_GPT_ratio_arithmetic_sequences_l1102_110257

variable (a : ℕ → ℕ) (b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h : ∀ n : ℕ, S n / T n = (3 * n - 1) / (2 * n + 3))

theorem ratio_arithmetic_sequences :
  a 7 / b 7 = 38 / 29 :=
sorry

end NUMINAMATH_GPT_ratio_arithmetic_sequences_l1102_110257


namespace NUMINAMATH_GPT_sum_of_numbers_with_lcm_and_ratio_l1102_110244

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ) (h_lcm : Nat.lcm a b = 60) (h_ratio : a = 2 * b / 3) : a + b = 50 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_with_lcm_and_ratio_l1102_110244


namespace NUMINAMATH_GPT_expand_expression_l1102_110282

theorem expand_expression : 
  (5 * x^2 + 2 * x - 3) * (3 * x^3 - x^2) = 15 * x^5 + x^4 - 11 * x^3 + 3 * x^2 := 
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1102_110282


namespace NUMINAMATH_GPT_measure_A_l1102_110227

noncomputable def angle_A (C B A : ℝ) : Prop :=
  C = 3 / 2 * B ∧ B = 30 ∧ A = 180 - B - C

theorem measure_A (A B C : ℝ) (h : angle_A C B A) : A = 105 :=
by
  -- Extract conditions from h
  obtain ⟨h1, h2, h3⟩ := h
  
  -- Use the conditions to prove the thesis
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_measure_A_l1102_110227


namespace NUMINAMATH_GPT_at_least_one_not_less_than_100_l1102_110259

-- Defining the original propositions
def p : Prop := ∀ (A_score : ℕ), A_score ≥ 100
def q : Prop := ∀ (B_score : ℕ), B_score < 100

-- Assertion to be proved in Lean
theorem at_least_one_not_less_than_100 (h1 : p) (h2 : q) : p ∨ ¬q := 
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_100_l1102_110259


namespace NUMINAMATH_GPT_solution_to_inequality_system_l1102_110248

theorem solution_to_inequality_system (x : ℝ) :
  (x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_inequality_system_l1102_110248


namespace NUMINAMATH_GPT_bob_second_week_hours_l1102_110258

theorem bob_second_week_hours (total_earnings : ℕ) (total_hours_first_week : ℕ) (regular_hours_pay : ℕ) 
  (overtime_hours_pay : ℕ) (regular_hours_max : ℕ) (total_hours_overtime_first_week : ℕ) 
  (earnings_first_week : ℕ) (earnings_second_week : ℕ) : 
  total_earnings = 472 →
  total_hours_first_week = 44 →
  regular_hours_pay = 5 →
  overtime_hours_pay = 6 →
  regular_hours_max = 40 →
  total_hours_overtime_first_week = total_hours_first_week - regular_hours_max →
  earnings_first_week = regular_hours_max * regular_hours_pay + 
                          total_hours_overtime_first_week * overtime_hours_pay →
  earnings_second_week = total_earnings - earnings_first_week → 
  ∃ h, earnings_second_week = h * regular_hours_pay ∨ 
  earnings_second_week = (regular_hours_max * regular_hours_pay + (h - regular_hours_max) * overtime_hours_pay) ∧ 
  h = 48 :=
by 
  intros 
  sorry 

end NUMINAMATH_GPT_bob_second_week_hours_l1102_110258


namespace NUMINAMATH_GPT_max_min_of_f_on_interval_l1102_110238

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 4 + 4 * x ^ 3 + 34

theorem max_min_of_f_on_interval :
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 50) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, 33 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 33) :=
by
  sorry

end NUMINAMATH_GPT_max_min_of_f_on_interval_l1102_110238


namespace NUMINAMATH_GPT_find_p_value_l1102_110207

open Set

/-- Given the parabola C: y^2 = 2px with p > 0, point A(0, sqrt(3)),
    and point B on the parabola such that AB is perpendicular to AF,
    and |BF| = 4. Determine the value of p. -/
theorem find_p_value (p : ℝ) (h : p > 0) :
  ∃ p, p = 2 ∨ p = 6 :=
sorry

end NUMINAMATH_GPT_find_p_value_l1102_110207


namespace NUMINAMATH_GPT_find_minimum_value_of_f_l1102_110226

def f (x : ℝ) : ℝ := (x ^ 2 + 4 * x + 5) * (x ^ 2 + 4 * x + 2) + 2 * x ^ 2 + 8 * x + 1

theorem find_minimum_value_of_f : ∃ x : ℝ, f x = -9 :=
by
  sorry

end NUMINAMATH_GPT_find_minimum_value_of_f_l1102_110226


namespace NUMINAMATH_GPT_side_salad_cost_l1102_110215

theorem side_salad_cost (T S : ℝ)
  (h1 : T + S + 4 + 2 = 2 * T) 
  (h2 : (T + S + 4 + 2) + T = 24) : S = 2 :=
by
  sorry

end NUMINAMATH_GPT_side_salad_cost_l1102_110215


namespace NUMINAMATH_GPT_min_value_of_a2_b2_l1102_110224

noncomputable def f (x a b : ℝ) := Real.exp x + a * x + b

theorem min_value_of_a2_b2 {a b : ℝ} (h : ∃ t ∈ Set.Icc (1 : ℝ) (3 : ℝ), f t a b = 0) :
  a^2 + b^2 ≥ (Real.exp 1)^2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a2_b2_l1102_110224


namespace NUMINAMATH_GPT_legacy_earnings_l1102_110278

theorem legacy_earnings 
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (earnings_per_hour : ℕ)
  (total_floors : floors = 4)
  (total_rooms_per_floor : rooms_per_floor = 10)
  (time_per_room : hours_per_room = 6)
  (rate_per_hour : earnings_per_hour = 15) :
  floors * rooms_per_floor * hours_per_room * earnings_per_hour = 3600 := 
by
  sorry

end NUMINAMATH_GPT_legacy_earnings_l1102_110278


namespace NUMINAMATH_GPT_parallel_lines_find_m_l1102_110267

theorem parallel_lines_find_m :
  (∀ (m : ℝ), ∀ (x y : ℝ), (2 * x + (m + 1) * y + 4 = 0) ∧ (m * x + 3 * y - 2 = 0) → (m = -3 ∨ m = 2)) := 
sorry

end NUMINAMATH_GPT_parallel_lines_find_m_l1102_110267


namespace NUMINAMATH_GPT_solve_equation_l1102_110274

theorem solve_equation (x y : ℝ) : 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1102_110274


namespace NUMINAMATH_GPT_slope_of_line_l1102_110245

theorem slope_of_line : ∀ (x y : ℝ), (6 * x + 10 * y = 30) → (y = -((3 / 5) * x) + 3) :=
by
  -- Proof needs to be filled out
  sorry

end NUMINAMATH_GPT_slope_of_line_l1102_110245


namespace NUMINAMATH_GPT_adam_has_more_apples_l1102_110251

-- Define the number of apples Jackie has
def JackiesApples : Nat := 9

-- Define the number of apples Adam has
def AdamsApples : Nat := 14

-- Statement of the problem: Prove that Adam has 5 more apples than Jackie
theorem adam_has_more_apples :
  AdamsApples - JackiesApples = 5 :=
by
  sorry

end NUMINAMATH_GPT_adam_has_more_apples_l1102_110251


namespace NUMINAMATH_GPT_parametric_to_ordinary_eq_l1102_110217

-- Define the parametric equations and the domain of the parameter t
def parametric_eqns (t : ℝ) : ℝ × ℝ := (t + 1, 3 - t^2)

-- Define the target equation to be proved
def target_eqn (x y : ℝ) : Prop := y = -x^2 + 2*x + 2

-- Prove that, given the parametric equations, the target ordinary equation holds
theorem parametric_to_ordinary_eq :
  ∃ (t : ℝ) (x y : ℝ), parametric_eqns t = (x, y) ∧ target_eqn x y :=
by
  sorry

end NUMINAMATH_GPT_parametric_to_ordinary_eq_l1102_110217


namespace NUMINAMATH_GPT_probability_win_l1102_110247

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_probability_win_l1102_110247


namespace NUMINAMATH_GPT_how_many_months_to_buy_tv_l1102_110253

-- Definitions based on given conditions
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500

def total_expenses := food_expenses + utilities_expenses + other_expenses
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000
def monthly_savings := monthly_income - total_expenses

-- Theorem statement based on the problem
theorem how_many_months_to_buy_tv 
    (H_income : monthly_income = 30000)
    (H_food : food_expenses = 15000)
    (H_utilities : utilities_expenses = 5000)
    (H_other : other_expenses = 2500)
    (H_savings : current_savings = 10000)
    (H_tv_cost : tv_cost = 25000)
    : (tv_cost - current_savings) / monthly_savings = 2 :=
by
  sorry

end NUMINAMATH_GPT_how_many_months_to_buy_tv_l1102_110253


namespace NUMINAMATH_GPT_remainder_of_452867_div_9_l1102_110297

theorem remainder_of_452867_div_9 : (452867 % 9) = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_of_452867_div_9_l1102_110297


namespace NUMINAMATH_GPT_trapezoid_AD_BC_ratio_l1102_110289

variables {A B C D M N K : Type} {AD BC CM MD NA CN : ℝ}

-- Definition of the trapezoid and the ratio conditions
def is_trapezoid (A B C D : Type) : Prop := sorry -- Assume existence of a trapezoid for lean to accept the statement
def ratio_CM_MD (CM MD : ℝ) : Prop := CM / MD = 4 / 3
def ratio_NA_CN (NA CN : ℝ) : Prop := NA / CN = 4 / 3

-- Proof statement for the given problem
theorem trapezoid_AD_BC_ratio 
  (h_trapezoid: is_trapezoid A B C D)
  (h_CM_MD: ratio_CM_MD CM MD)
  (h_NA_CN: ratio_NA_CN NA CN) :
  AD / BC = 7 / 12 :=
sorry

end NUMINAMATH_GPT_trapezoid_AD_BC_ratio_l1102_110289


namespace NUMINAMATH_GPT_find_f_10_l1102_110254

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : f x = f (1 / x) * Real.log x + 10

theorem find_f_10 : f 10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_f_10_l1102_110254


namespace NUMINAMATH_GPT_exists_solution_l1102_110204

noncomputable def smallest_c0 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) : ℕ :=
  a * b - a - b + 1

theorem exists_solution (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) :
  ∃ c0, (c0 = smallest_c0 a b ha hb h) ∧ ∀ c : ℕ, c ≥ c0 → ∃ x y : ℕ, a * x + b * y = c :=
sorry

end NUMINAMATH_GPT_exists_solution_l1102_110204


namespace NUMINAMATH_GPT_totalMilkConsumption_l1102_110265

-- Conditions
def regularMilk (week: ℕ) : ℝ := 0.5
def soyMilk (week: ℕ) : ℝ := 0.1

-- Theorem statement
theorem totalMilkConsumption : regularMilk 1 + soyMilk 1 = 0.6 := 
by 
  sorry

end NUMINAMATH_GPT_totalMilkConsumption_l1102_110265


namespace NUMINAMATH_GPT_polygon_sides_l1102_110260

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_l1102_110260


namespace NUMINAMATH_GPT_quadrilateral_area_l1102_110228

theorem quadrilateral_area 
  (p : ℝ) (hp : p > 0)
  (P : ℝ × ℝ) (hP : P = (1, 1 / 4))
  (focus : ℝ × ℝ) (hfocus : focus = (0, 1))
  (directrix : ℝ → Prop) (hdirectrix : ∀ y, directrix y ↔ y = 1)
  (F : ℝ × ℝ) (hF : F = (0, 1))
  (M : ℝ × ℝ) (hM : M = (0, 1))
  (Q : ℝ × ℝ) 
  (PQ : ℝ)
  (area : ℝ) 
  (harea : area = 13 / 8) :
  ∃ (PQMF : ℝ), PQMF = 13 / 8 :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_l1102_110228


namespace NUMINAMATH_GPT_remainder_when_divided_by_15_l1102_110236

theorem remainder_when_divided_by_15 (N : ℕ) (k : ℤ) (h1 : N = 60 * k + 49) : (N % 15) = 4 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_15_l1102_110236


namespace NUMINAMATH_GPT_retail_profit_percent_l1102_110212

variable (CP : ℝ) (MP : ℝ) (SP : ℝ)
variable (h_marked : MP = CP + 0.60 * CP)
variable (h_discount : SP = MP - 0.25 * MP)

theorem retail_profit_percent : CP = 100 → MP = CP + 0.60 * CP → SP = MP - 0.25 * MP → 
       (SP - CP) / CP * 100 = 20 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_retail_profit_percent_l1102_110212


namespace NUMINAMATH_GPT_number_of_children_riding_tricycles_l1102_110234

-- Definitions
def bicycles_wheels := 2
def tricycles_wheels := 3

def adults := 6
def total_wheels := 57

-- Problem statement
theorem number_of_children_riding_tricycles (c : ℕ) (H : 12 + 3 * c = total_wheels) : c = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_riding_tricycles_l1102_110234


namespace NUMINAMATH_GPT_smallest_fourth_number_l1102_110295

theorem smallest_fourth_number :
  ∃ (a b : ℕ), 145 + 10 * a + b = 4 * (28 + a + b) ∧ 10 * a + b = 35 :=
by
  sorry

end NUMINAMATH_GPT_smallest_fourth_number_l1102_110295


namespace NUMINAMATH_GPT_simplify_expression_l1102_110272

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ( ((a ^ (4 / 3 / 5)) ^ (3 / 2)) / ((a ^ (4 / 1 / 5)) ^ 3) ) /
  ( ((a * (a ^ (2 / 3) * b ^ (1 / 3))) ^ (1 / 2)) ^ 4) * 
  (a ^ (1 / 4) * b ^ (1 / 8)) ^ 6 = 1 / ((a ^ (2 / 12)) * (b ^ (1 / 12))) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1102_110272


namespace NUMINAMATH_GPT_total_blue_marbles_l1102_110276

theorem total_blue_marbles (red_Jenny blue_Jenny red_Mary blue_Mary red_Anie blue_Anie : ℕ)
  (h1: red_Jenny = 30)
  (h2: blue_Jenny = 25)
  (h3: red_Mary = 2 * red_Jenny)
  (h4: blue_Mary = blue_Anie / 2)
  (h5: red_Anie = red_Mary + 20)
  (h6: blue_Anie = 2 * blue_Jenny) :
  blue_Mary + blue_Jenny + blue_Anie = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_blue_marbles_l1102_110276


namespace NUMINAMATH_GPT_giraffes_difference_l1102_110255

theorem giraffes_difference :
  ∃ n : ℕ, (300 = 3 * n) ∧ (300 - n = 200) :=
by
  sorry

end NUMINAMATH_GPT_giraffes_difference_l1102_110255


namespace NUMINAMATH_GPT_concentric_circles_ratio_l1102_110202

theorem concentric_circles_ratio
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : π * b^2 - π * a^2 = 4 * (π * a^2)) :
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_concentric_circles_ratio_l1102_110202


namespace NUMINAMATH_GPT_inequality_has_no_solution_l1102_110235

theorem inequality_has_no_solution (x : ℝ) : -x^2 + 2*x - 2 > 0 → false :=
by
  sorry

end NUMINAMATH_GPT_inequality_has_no_solution_l1102_110235


namespace NUMINAMATH_GPT_sequence_periodic_l1102_110290

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodic (a : ℕ → ℝ) (m_0 : ℕ) (h : sequence_condition a) :
  ∀ m ≥ m_0, a (m + 9) = a m := 
sorry

end NUMINAMATH_GPT_sequence_periodic_l1102_110290


namespace NUMINAMATH_GPT_sum_of_two_numbers_is_147_l1102_110271

theorem sum_of_two_numbers_is_147 (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) :
  A + B = 147 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_is_147_l1102_110271


namespace NUMINAMATH_GPT_find_first_discount_l1102_110263

theorem find_first_discount (price_initial : ℝ) (price_final : ℝ) (discount_additional : ℝ) (x : ℝ) :
  price_initial = 350 → price_final = 266 → discount_additional = 5 →
  price_initial * (1 - x / 100) * (1 - discount_additional / 100) = price_final →
  x = 20 :=
by
  intros h1 h2 h3 h4
  -- skippable in proofs, just holds the place
  sorry

end NUMINAMATH_GPT_find_first_discount_l1102_110263


namespace NUMINAMATH_GPT_jason_initial_pears_l1102_110218

-- Define the initial number of pears Jason picked.
variable (P : ℕ)

-- Conditions translated to Lean:
-- Jason gave Keith 47 pears and received 12 from Mike, leaving him with 11 pears.
variable (h1 : P - 47 + 12 = 11)

-- The theorem stating the problem:
theorem jason_initial_pears : P = 46 :=
by
  sorry

end NUMINAMATH_GPT_jason_initial_pears_l1102_110218


namespace NUMINAMATH_GPT_problem_statement_l1102_110229

-- Define rational number representations for points A, B, and C
def a : ℚ := (-4)^2 - 8

-- Define that B and C are opposites
def are_opposites (b c : ℚ) : Prop := b = -c

-- Define the distance condition
def distance_is_three (a c : ℚ) : Prop := |c - a| = 3

-- Main theorem statement
theorem problem_statement :
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -74) ∨
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -86) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1102_110229


namespace NUMINAMATH_GPT_product_xyz_l1102_110256

noncomputable def x : ℚ := 97 / 12
noncomputable def n : ℚ := 8 * x
noncomputable def y : ℚ := n + 7
noncomputable def z : ℚ := n - 11

theorem product_xyz 
  (h1: x + y + z = 190)
  (h2: n = 8 * x)
  (h3: n = y - 7)
  (h4: n = z + 11) : 
  x * y * z = (97 * 215 * 161) / 108 := 
by 
  sorry

end NUMINAMATH_GPT_product_xyz_l1102_110256


namespace NUMINAMATH_GPT_product_of_solutions_l1102_110275

theorem product_of_solutions : (∃ x : ℝ, |x| = 3*(|x| - 2)) → (x = 3 ∨ x = -3) → 3 * -3 = -9 :=
by sorry

end NUMINAMATH_GPT_product_of_solutions_l1102_110275


namespace NUMINAMATH_GPT_coffee_cost_per_week_l1102_110296

def num_people: ℕ := 4
def cups_per_person_per_day: ℕ := 2
def ounces_per_cup: ℝ := 0.5
def cost_per_ounce: ℝ := 1.25

theorem coffee_cost_per_week : 
  (num_people * cups_per_person_per_day * ounces_per_cup * 7 * cost_per_ounce) = 35 :=
by
  sorry

end NUMINAMATH_GPT_coffee_cost_per_week_l1102_110296


namespace NUMINAMATH_GPT_radius_of_shorter_cone_l1102_110269

theorem radius_of_shorter_cone {h : ℝ} (h_ne_zero : h ≠ 0) :
  ∀ r : ℝ, ∀ V_taller V_shorter : ℝ,
   (V_taller = (1/3) * π * (5 ^ 2) * (4 * h)) →
   (V_shorter = (1/3) * π * (r ^ 2) * h) →
   V_taller = V_shorter →
   r = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_radius_of_shorter_cone_l1102_110269


namespace NUMINAMATH_GPT_correct_calculation_is_d_l1102_110252

theorem correct_calculation_is_d :
  (-7) + (-7) ≠ 0 ∧
  ((-1 / 10) - (1 / 10)) ≠ 0 ∧
  (0 + (-101)) ≠ 101 ∧
  (1 / 3 + -1 / 2 = -1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_is_d_l1102_110252


namespace NUMINAMATH_GPT_consecutive_ints_prod_square_l1102_110243

theorem consecutive_ints_prod_square (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
sorry

end NUMINAMATH_GPT_consecutive_ints_prod_square_l1102_110243


namespace NUMINAMATH_GPT_trigonometric_identity_l1102_110200

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin α ^ 2 + 2 * Real.cos α ^ 2 = 6 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1102_110200


namespace NUMINAMATH_GPT_value_of_r_minus_p_l1102_110203

-- Define the arithmetic mean conditions
def arithmetic_mean1 (p q : ℝ) : Prop :=
  (p + q) / 2 = 10

def arithmetic_mean2 (q r : ℝ) : Prop :=
  (q + r) / 2 = 27

-- Prove that r - p = 34 based on the conditions
theorem value_of_r_minus_p (p q r : ℝ)
  (h1 : arithmetic_mean1 p q)
  (h2 : arithmetic_mean2 q r) :
  r - p = 34 :=
by
  sorry

end NUMINAMATH_GPT_value_of_r_minus_p_l1102_110203
