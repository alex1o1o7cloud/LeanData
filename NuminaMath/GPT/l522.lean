import Mathlib

namespace NUMINAMATH_GPT_explicit_formula_for_f_l522_52201

theorem explicit_formula_for_f (f : ℕ → ℕ) (h₀ : f 0 = 0)
  (h₁ : ∀ (n : ℕ), n % 6 = 0 ∨ n % 6 = 1 → f (n + 1) = f n + 3)
  (h₂ : ∀ (n : ℕ), n % 6 = 2 ∨ n % 6 = 5 → f (n + 1) = f n + 1)
  (h₃ : ∀ (n : ℕ), n % 6 = 3 ∨ n % 6 = 4 → f (n + 1) = f n + 2)
  (n : ℕ) : f (6 * n) = 12 * n :=
by
  sorry

end NUMINAMATH_GPT_explicit_formula_for_f_l522_52201


namespace NUMINAMATH_GPT_value_of_x_plus_y_l522_52285

-- Define the sum of integers from 50 to 60
def sum_integers_50_to_60 : ℤ := List.sum (List.range' 50 (60 - 50 + 1))

-- Calculate the number of even integers from 50 to 60
def count_even_integers_50_to_60 : ℤ := List.length (List.filter (λ n => n % 2 = 0) (List.range' 50 (60 - 50 + 1)))

-- Define x and y based on the given conditions
def x : ℤ := sum_integers_50_to_60
def y : ℤ := count_even_integers_50_to_60

-- The main theorem to prove
theorem value_of_x_plus_y : x + y = 611 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l522_52285


namespace NUMINAMATH_GPT_interval_of_a_l522_52260

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then Real.exp x + x^2 else Real.exp (-x) + x^2

theorem interval_of_a (a : ℝ) :
  f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_interval_of_a_l522_52260


namespace NUMINAMATH_GPT_num_five_ruble_coins_l522_52289

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end NUMINAMATH_GPT_num_five_ruble_coins_l522_52289


namespace NUMINAMATH_GPT_XiaoMing_reading_problem_l522_52275

theorem XiaoMing_reading_problem :
  ∀ (total_pages days first_days first_rate remaining_rate : ℕ),
    total_pages = 72 →
    days = 10 →
    first_days = 2 →
    first_rate = 5 →
    (first_days * first_rate) + ((days - first_days) * remaining_rate) ≥ total_pages →
    remaining_rate ≥ 8 :=
by
  intros total_pages days first_days first_rate remaining_rate
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_XiaoMing_reading_problem_l522_52275


namespace NUMINAMATH_GPT_cost_price_of_ball_l522_52239

theorem cost_price_of_ball (x : ℕ) (h : 13 * x = 720 + 5 * x) : x = 90 :=
by sorry

end NUMINAMATH_GPT_cost_price_of_ball_l522_52239


namespace NUMINAMATH_GPT_range_of_a_l522_52238

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ :=
  (m * x + n) / (x ^ 2 + 1)

example (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1) : 
  m = 2 ∧ n = 0 :=
sorry

theorem range_of_a (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1)
  (h_m : m = 2) (h_n : n = 0) {a : ℝ} : f (a-1) m n + f (a^2-1) m n < 0 ↔ 0 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l522_52238


namespace NUMINAMATH_GPT_star_proof_l522_52292

def star (a b : ℕ) : ℕ := 3 + b ^ a

theorem star_proof : star (star 2 1) 4 = 259 :=
by
  sorry

end NUMINAMATH_GPT_star_proof_l522_52292


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l522_52207

open Real

noncomputable def p (x : ℝ) : Prop := |x| < 3
noncomputable def q (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
by
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l522_52207


namespace NUMINAMATH_GPT_total_fruits_picked_l522_52277

variable (L M P B : Nat)

theorem total_fruits_picked (hL : L = 25) (hM : M = 32) (hP : P = 12) (hB : B = 18) : L + M + P = 69 :=
by
  sorry

end NUMINAMATH_GPT_total_fruits_picked_l522_52277


namespace NUMINAMATH_GPT_bugs_max_contacts_l522_52226

theorem bugs_max_contacts :
  ∃ a b : ℕ, (a + b = 2016) ∧ (a * b = 1008^2) :=
by
  sorry

end NUMINAMATH_GPT_bugs_max_contacts_l522_52226


namespace NUMINAMATH_GPT_simplify_expression_l522_52290

theorem simplify_expression (w x : ℤ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 20 * x + 24 = 45 * w + 20 * x + 24 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l522_52290


namespace NUMINAMATH_GPT_max_f_max_g_pow_f_l522_52230

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x^2 + 7 * x + 14)
noncomputable def g (x : ℝ) : ℝ := (x^2 - 5 * x + 10) / (x^2 + 5 * x + 20)

theorem max_f : ∀ x : ℝ, f x ≤ 2 := by
  intro x
  sorry

theorem max_g_pow_f : ∀ x : ℝ, g x ^ f x ≤ 9 := by
  intro x
  sorry

end NUMINAMATH_GPT_max_f_max_g_pow_f_l522_52230


namespace NUMINAMATH_GPT_exists_infinite_triples_a_no_triples_b_l522_52259

-- Question (a)
theorem exists_infinite_triples_a : ∀ k : ℕ, ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2 - 1) :=
by {
  sorry
}

-- Question (b)
theorem no_triples_b : ¬ ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_infinite_triples_a_no_triples_b_l522_52259


namespace NUMINAMATH_GPT_plan_A_fee_eq_nine_l522_52297

theorem plan_A_fee_eq_nine :
  ∃ F : ℝ, (0.25 * 60 + F = 0.40 * 60) ∧ (F = 9) :=
by
  sorry

end NUMINAMATH_GPT_plan_A_fee_eq_nine_l522_52297


namespace NUMINAMATH_GPT_correct_average_marks_l522_52278

-- Define all the given conditions
def average_marks : ℕ := 92
def number_of_students : ℕ := 25
def wrong_mark : ℕ := 75
def correct_mark : ℕ := 30

-- Define variables for total marks calculations
def total_marks_with_wrong : ℕ := average_marks * number_of_students
def total_marks_with_correct : ℕ := total_marks_with_wrong - wrong_mark + correct_mark

-- Goal: Prove that the correct average marks is 90.2
theorem correct_average_marks :
  (total_marks_with_correct : ℝ) / (number_of_students : ℝ) = 90.2 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_marks_l522_52278


namespace NUMINAMATH_GPT_friends_behind_Yuna_l522_52208

def total_friends : ℕ := 6
def friends_in_front_of_Yuna : ℕ := 2

theorem friends_behind_Yuna : total_friends - friends_in_front_of_Yuna = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_friends_behind_Yuna_l522_52208


namespace NUMINAMATH_GPT_triangle_side_a_value_l522_52279

noncomputable def a_value (A B c : ℝ) : ℝ :=
  30 * Real.sqrt 2 - 10 * Real.sqrt 6

theorem triangle_side_a_value
  (A B : ℝ) (c : ℝ)
  (hA : A = 60)
  (hB : B = 45)
  (hc : c = 20) :
  a_value A B c = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_a_value_l522_52279


namespace NUMINAMATH_GPT_addition_value_l522_52210

def certain_number : ℝ := 5.46 - 3.97

theorem addition_value : 5.46 + certain_number = 6.95 := 
  by 
    -- The proof would go here, but is replaced with sorry.
    sorry

end NUMINAMATH_GPT_addition_value_l522_52210


namespace NUMINAMATH_GPT_sum_YNRB_l522_52202

theorem sum_YNRB :
  ∃ (R Y B N : ℕ),
    (RY = 10 * R + Y) ∧
    (BY = 10 * B + Y) ∧
    (111 * N = (10 * R + Y) * (10 * B + Y)) →
    (Y + N + R + B = 21) :=
sorry

end NUMINAMATH_GPT_sum_YNRB_l522_52202


namespace NUMINAMATH_GPT_steel_more_by_l522_52240

variable {S T C k : ℝ}
variable (k_greater_than_zero : k > 0)
variable (copper_weight : C = 90)
variable (S_twice_T : S = 2 * T)
variable (S_minus_C : S = C + k)
variable (total_eq : 20 * S + 20 * T + 20 * C = 5100)

theorem steel_more_by (k): k = 20 := by
  sorry

end NUMINAMATH_GPT_steel_more_by_l522_52240


namespace NUMINAMATH_GPT_gcd_possible_values_count_l522_52243

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end NUMINAMATH_GPT_gcd_possible_values_count_l522_52243


namespace NUMINAMATH_GPT_sum_of_coefficients_is_7_l522_52291

noncomputable def v (n : ℕ) : ℕ := sorry

theorem sum_of_coefficients_is_7 : 
  (∀ n : ℕ, v (n + 1) - v n = 3 * n + 2) → (v 1 = 7) → (∃ a b c : ℝ, (a * n^2 + b * n + c = v n) ∧ (a + b + c = 7)) := 
by
  intros H1 H2
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_7_l522_52291


namespace NUMINAMATH_GPT_range_of_a_l522_52298

noncomputable def inequality_always_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (a : ℝ) : inequality_always_holds a ↔ 0 ≤ a ∧ a < 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l522_52298


namespace NUMINAMATH_GPT_base_7_digits_956_l522_52262

theorem base_7_digits_956 : ∃ n : ℕ, ∀ k : ℕ, 956 < 7^k → n = k ∧ 956 ≥ 7^(k-1) := sorry

end NUMINAMATH_GPT_base_7_digits_956_l522_52262


namespace NUMINAMATH_GPT_total_travel_time_is_19_hours_l522_52233

-- Define the distances and speeds as constants
def distance_WA_ID := 640
def speed_WA_ID := 80
def distance_ID_NV := 550
def speed_ID_NV := 50

-- Define the times based on the given distances and speeds
def time_WA_ID := distance_WA_ID / speed_WA_ID
def time_ID_NV := distance_ID_NV / speed_ID_NV

-- Define the total time
def total_time := time_WA_ID + time_ID_NV

-- Prove that the total travel time is 19 hours
theorem total_travel_time_is_19_hours : total_time = 19 := by
  sorry

end NUMINAMATH_GPT_total_travel_time_is_19_hours_l522_52233


namespace NUMINAMATH_GPT_eval_expression_l522_52237

def base8_to_base10 (n : Nat) : Nat :=
  2 * 8^2 + 4 * 8^1 + 5 * 8^0

def base4_to_base10 (n : Nat) : Nat :=
  1 * 4^1 + 5 * 4^0

def base5_to_base10 (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 2 * 5^0

def base6_to_base10 (n : Nat) : Nat :=
  3 * 6^1 + 2 * 6^0

theorem eval_expression : 
  base8_to_base10 245 / base4_to_base10 15 - base5_to_base10 232 / base6_to_base10 32 = 15 :=
by sorry

end NUMINAMATH_GPT_eval_expression_l522_52237


namespace NUMINAMATH_GPT_largest_divisor_expression_l522_52293

theorem largest_divisor_expression (y : ℤ) (h : y % 2 = 1) : 
  4320 ∣ (15 * y + 3) * (15 * y + 9) * (10 * y + 10) :=
sorry  

end NUMINAMATH_GPT_largest_divisor_expression_l522_52293


namespace NUMINAMATH_GPT_trapezoid_QR_length_l522_52216

variable (PQ RS Area Alt QR : ℝ)
variable (h1 : Area = 216)
variable (h2 : Alt = 9)
variable (h3 : PQ = 12)
variable (h4 : RS = 20)
variable (h5 : QR = 11)

theorem trapezoid_QR_length : 
  (∃ (PQ RS Area Alt QR : ℝ), 
    Area = 216 ∧
    Alt = 9 ∧
    PQ = 12 ∧
    RS = 20) → QR = 11 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_QR_length_l522_52216


namespace NUMINAMATH_GPT_merchant_articles_l522_52287

theorem merchant_articles 
   (CP SP : ℝ)
   (N : ℝ)
   (h1 : SP = 1.25 * CP)
   (h2 : N * CP = 16 * SP) : 
   N = 20 := by
   sorry

end NUMINAMATH_GPT_merchant_articles_l522_52287


namespace NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_plane_l522_52252

variables (m n l : Line) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop := sorry
def parallel (m n : Line) : Prop := sorry

theorem lines_parallel_if_perpendicular_to_plane
  (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_plane_l522_52252


namespace NUMINAMATH_GPT_michael_twice_jacob_in_11_years_l522_52281

-- Definitions
def jacob_age_4_years := 5
def jacob_current_age := jacob_age_4_years - 4
def michael_current_age := jacob_current_age + 12

-- Theorem to prove
theorem michael_twice_jacob_in_11_years :
  ∀ (x : ℕ), jacob_current_age + x = 1 →
    michael_current_age + x = 13 →
    michael_current_age + (11 : ℕ) = 2 * (jacob_current_age + (11 : ℕ)) :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_michael_twice_jacob_in_11_years_l522_52281


namespace NUMINAMATH_GPT_Sam_dimes_remaining_l522_52212

-- Define the initial and borrowed dimes
def initial_dimes_count : Nat := 8
def borrowed_dimes_count : Nat := 4

-- State the theorem
theorem Sam_dimes_remaining : (initial_dimes_count - borrowed_dimes_count) = 4 := by
  sorry

end NUMINAMATH_GPT_Sam_dimes_remaining_l522_52212


namespace NUMINAMATH_GPT_acute_triangle_area_relation_l522_52264

open Real

variables (A B C R : ℝ)
variables (acute_triangle : Prop)
variables (S p_star : ℝ)

-- Conditions
axiom acute_triangle_condition : acute_triangle
axiom area_formula : S = (R^2 / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))
axiom semiperimeter_formula : p_star = (R / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))

-- Theorem to prove
theorem acute_triangle_area_relation (h : acute_triangle) : S = p_star * R := 
by {
  sorry 
}

end NUMINAMATH_GPT_acute_triangle_area_relation_l522_52264


namespace NUMINAMATH_GPT_set_theory_problem_l522_52217

def U : Set ℤ := {x ∈ Set.univ | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_theory_problem : 
  (A ∩ B = {4}) ∧ 
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧ 
  (U \ (A ∪ C) = {6, 8, 10}) ∧ 
  ((U \ A) ∩ (U \ B) = {3}) := 
by 
  sorry

end NUMINAMATH_GPT_set_theory_problem_l522_52217


namespace NUMINAMATH_GPT_MiaShots_l522_52265

theorem MiaShots (shots_game1_to_5 : ℕ) (total_shots_game1_to_5 : ℕ) (initial_avg : ℕ → ℕ → Prop)
  (shots_game6 : ℕ) (new_avg_shots : ℕ → ℕ → Prop) (total_shots : ℕ) (new_avg : ℕ): 
  shots_game1_to_5 = 20 →
  total_shots_game1_to_5 = 50 →
  initial_avg shots_game1_to_5 total_shots_game1_to_5 →
  shots_game6 = 15 →
  new_avg_shots 29 65 →
  total_shots = total_shots_game1_to_5 + shots_game6 →
  new_avg = 45 →
  (∃ shots_made_game6 : ℕ, shots_made_game6 = 29 - shots_game1_to_5 ∧ shots_made_game6 = 9) :=
by
  sorry

end NUMINAMATH_GPT_MiaShots_l522_52265


namespace NUMINAMATH_GPT_range_of_a_l522_52218

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l522_52218


namespace NUMINAMATH_GPT_probability_five_digit_palindrome_div_by_11_l522_52203

noncomputable
def five_digit_palindrome_div_by_11_probability : ℚ :=
  let total_palindromes := 900
  let valid_palindromes := 80
  valid_palindromes / total_palindromes

theorem probability_five_digit_palindrome_div_by_11 :
  five_digit_palindrome_div_by_11_probability = 2 / 25 := by
  sorry

end NUMINAMATH_GPT_probability_five_digit_palindrome_div_by_11_l522_52203


namespace NUMINAMATH_GPT_sequence_u5_eq_27_l522_52276

theorem sequence_u5_eq_27 (u : ℕ → ℝ) 
  (h_recurrence : ∀ n, u (n + 2) = 3 * u (n + 1) - 2 * u n)
  (h_u3 : u 3 = 15)
  (h_u6 : u 6 = 43) :
  u 5 = 27 :=
  sorry

end NUMINAMATH_GPT_sequence_u5_eq_27_l522_52276


namespace NUMINAMATH_GPT_maximum_m_value_l522_52221

theorem maximum_m_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∃ m, m = 4 ∧ (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (1 / a + 1 / b) ≥ m) :=
sorry

end NUMINAMATH_GPT_maximum_m_value_l522_52221


namespace NUMINAMATH_GPT_impossible_result_l522_52219

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬ (∃ f1 f_1 : ℤ, f1 = a * Real.sin 1 + b + c ∧ f_1 = -a * Real.sin 1 - b + c ∧ (f1 = 1 ∧ f_1 = 2)) :=
by
  sorry

end NUMINAMATH_GPT_impossible_result_l522_52219


namespace NUMINAMATH_GPT_division_result_l522_52283

def m : ℕ := 16 ^ 2024

theorem division_result : m / 8 = 8 * 16 ^ 2020 :=
by
  -- sorry for the actual proof
  sorry

end NUMINAMATH_GPT_division_result_l522_52283


namespace NUMINAMATH_GPT_geometric_sequence_a4_l522_52231

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ (n : ℕ), a (n + 1) = a n * r

def a_3a_5_is_64 (a : ℕ → ℝ) : Prop :=
  a 3 * a 5 = 64

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h1 : is_geometric_sequence a) (h2 : a_3a_5_is_64 a) : a 4 = 8 ∨ a 4 = -8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l522_52231


namespace NUMINAMATH_GPT_movie_ticket_cost_l522_52225

/--
Movie tickets cost a certain amount on a Monday, twice as much on a Wednesday, and five times as much as on Monday on a Saturday. If Glenn goes to the movie theater on Wednesday and Saturday, he spends $35. Prove that the cost of a movie ticket on a Monday is $5.
-/
theorem movie_ticket_cost (M : ℕ) 
  (wednesday_cost : 2 * M = 2 * M)
  (saturday_cost : 5 * M = 5 * M) 
  (total_cost : 2 * M + 5 * M = 35) : 
  M = 5 := 
sorry

end NUMINAMATH_GPT_movie_ticket_cost_l522_52225


namespace NUMINAMATH_GPT_proof_statements_l522_52242

theorem proof_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧       -- corresponding to A
  ¬((∃ m : ℕ, 190 = 19 * m) ∧  ¬(∃ k : ℕ, 57 = 19 * k)) ∧  -- corresponding to B
  ¬((∃ p : ℕ, 90 = 30 * p) ∨ (∃ q : ℕ, 65 = 30 * q)) ∧     -- corresponding to C
  ¬((∃ r : ℕ, 33 = 11 * r) ∧ ¬(∃ s : ℕ, 55 = 11 * s)) ∧    -- corresponding to D
  (∃ t : ℕ, 162 = 9 * t) :=                                 -- corresponding to E
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_proof_statements_l522_52242


namespace NUMINAMATH_GPT_second_number_is_90_l522_52234

theorem second_number_is_90 (a b c : ℕ) 
  (h1 : a + b + c = 330) 
  (h2 : a = 2 * b) 
  (h3 : c = (1 / 3) * a) : 
  b = 90 := 
by
  sorry

end NUMINAMATH_GPT_second_number_is_90_l522_52234


namespace NUMINAMATH_GPT_simplify_expression_l522_52229

theorem simplify_expression : 2 - 2 / (2 + Real.sqrt 5) + 2 / (2 - Real.sqrt 5) = 2 + 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l522_52229


namespace NUMINAMATH_GPT_tangent_line_at_0_2_is_correct_l522_52204

noncomputable def curve (x : ℝ) : ℝ := Real.exp (-2 * x) + 1

def tangent_line_at_0_2 (x : ℝ) : ℝ := -2 * x + 2

theorem tangent_line_at_0_2_is_correct :
  tangent_line_at_0_2 = fun x => -2 * x + 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_tangent_line_at_0_2_is_correct_l522_52204


namespace NUMINAMATH_GPT_second_ball_red_probability_l522_52280

-- Definitions based on given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4
def first_ball_is_red : Prop := true

-- The probability that the second ball drawn is red given the first ball drawn is red
def prob_second_red_given_first_red : ℚ :=
  (red_balls - 1) / (total_balls - 1)

theorem second_ball_red_probability :
  first_ball_is_red → prob_second_red_given_first_red = 5 / 9 :=
by
  intro _
  -- proof goes here
  sorry

end NUMINAMATH_GPT_second_ball_red_probability_l522_52280


namespace NUMINAMATH_GPT_num_valid_functions_l522_52244

theorem num_valid_functions :
  ∃! (f : ℤ → ℝ), 
  (f 1 = 1) ∧ 
  (∀ (m n : ℤ), f m ^ 2 - f n ^ 2 = f (m + n) * f (m - n)) ∧ 
  (∀ n : ℤ, f n = f (n + 2013)) :=
sorry

end NUMINAMATH_GPT_num_valid_functions_l522_52244


namespace NUMINAMATH_GPT_total_payment_l522_52272

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end NUMINAMATH_GPT_total_payment_l522_52272


namespace NUMINAMATH_GPT_laborer_monthly_income_l522_52286

theorem laborer_monthly_income :
  (∃ (I D : ℤ),
    6 * I + D = 540 ∧
    4 * I - D = 270) →
  (∃ I : ℤ,
    I = 81) :=
by
  sorry

end NUMINAMATH_GPT_laborer_monthly_income_l522_52286


namespace NUMINAMATH_GPT_arithmetic_series_sum_l522_52220

theorem arithmetic_series_sum :
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  sum = 418 := by {
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  have h₁ : n = 11 := by sorry
  have h₂ : sum = 418 := by sorry
  exact h₂
}

end NUMINAMATH_GPT_arithmetic_series_sum_l522_52220


namespace NUMINAMATH_GPT_minimum_value_problem_l522_52263

theorem minimum_value_problem (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) 
  (hxyz : x + y + z + w = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (x + w) + 1 / (y + z) + 1 / (y + w) + 1 / (z + w)) ≥ 18 := 
sorry

end NUMINAMATH_GPT_minimum_value_problem_l522_52263


namespace NUMINAMATH_GPT_perpendicular_condition_sufficient_but_not_necessary_l522_52282

theorem perpendicular_condition_sufficient_but_not_necessary (a : ℝ) :
  (a = -2) → ((∀ x y : ℝ, ax + (a + 1) * y + 1 = 0 → x + a * y + 2 = 0 ∧ (∃ t : ℝ, t ≠ 0 ∧ x = -t / (a + 1) ∧ y = (t / a))) →
  ¬ (a = -2) ∨ (a + 1 ≠ 0 ∧ ∃ k1 k2 : ℝ, k1 * k2 = -1 ∧ k1 = -a / (a + 1) ∧ k2 = -1 / a)) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_condition_sufficient_but_not_necessary_l522_52282


namespace NUMINAMATH_GPT_fifth_observation_l522_52266

theorem fifth_observation (O1 O2 O3 O4 O5 O6 O7 O8 O9 : ℝ)
  (h1 : O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 = 72)
  (h2 : O1 + O2 + O3 + O4 + O5 = 50)
  (h3 : O5 + O6 + O7 + O8 + O9 = 40) :
  O5 = 18 := 
  sorry

end NUMINAMATH_GPT_fifth_observation_l522_52266


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l522_52246

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 > 4) → (x > 2 ∨ x < -2) ∧ ¬((x^2 > 4) ↔ (x > 2)) :=
by
  intros h
  have h1 : x > 2 ∨ x < -2 := by sorry
  have h2 : ¬((x^2 > 4) ↔ (x > 2)) := by sorry
  exact And.intro h1 h2

end NUMINAMATH_GPT_necessary_but_not_sufficient_l522_52246


namespace NUMINAMATH_GPT_union_of_sets_l522_52247

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2}) (hB : B = {2, 3}) : A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l522_52247


namespace NUMINAMATH_GPT_evaluate_expression_l522_52241

theorem evaluate_expression : 6 - 5 * (10 - (2 + 3)^2) * 2 = 306 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l522_52241


namespace NUMINAMATH_GPT_baskets_containing_neither_l522_52254

-- Definitions representing the conditions
def total_baskets : ℕ := 15
def baskets_with_apples : ℕ := 10
def baskets_with_oranges : ℕ := 8
def baskets_with_both : ℕ := 5

-- Theorem statement to prove the number of baskets containing neither apples nor oranges
theorem baskets_containing_neither : total_baskets - (baskets_with_apples + baskets_with_oranges - baskets_with_both) = 2 :=
by
  sorry

end NUMINAMATH_GPT_baskets_containing_neither_l522_52254


namespace NUMINAMATH_GPT_largest_whole_number_m_satisfies_inequality_l522_52257

theorem largest_whole_number_m_satisfies_inequality :
  ∃ m : ℕ, (1 / 4 + m / 6 : ℚ) < 3 / 2 ∧ ∀ n : ℕ, (1 / 4 + n / 6 : ℚ) < 3 / 2 → n ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_whole_number_m_satisfies_inequality_l522_52257


namespace NUMINAMATH_GPT_parameterization_of_line_l522_52256

theorem parameterization_of_line : 
  ∀ t : ℝ, ∃ f : ℝ → ℝ, (f t, 20 * t - 14) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), y = 2 * x - 40 ∧ p = (x, y) } ∧ f t = 10 * t + 13 :=
by
  sorry

end NUMINAMATH_GPT_parameterization_of_line_l522_52256


namespace NUMINAMATH_GPT_problem_f_neg2_l522_52224

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 2007 + b * x + 1

theorem problem_f_neg2 (a b : ℝ) (h : f a b 2 = 2) : f a b (-2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_f_neg2_l522_52224


namespace NUMINAMATH_GPT_six_div_one_minus_three_div_ten_equals_twenty_four_l522_52294

theorem six_div_one_minus_three_div_ten_equals_twenty_four :
  (6 : ℤ) / (1 - (3 : ℤ) / (10 : ℤ)) = 24 := 
by
  sorry

end NUMINAMATH_GPT_six_div_one_minus_three_div_ten_equals_twenty_four_l522_52294


namespace NUMINAMATH_GPT_remaining_number_l522_52274

theorem remaining_number (S : Finset ℕ) (hS : S = Finset.range 51) :
  ∃ n ∈ S, n % 2 = 0 := 
sorry

end NUMINAMATH_GPT_remaining_number_l522_52274


namespace NUMINAMATH_GPT_angle_B_equal_pi_div_3_l522_52215

-- Define the conditions and the statement to be proved
theorem angle_B_equal_pi_div_3 (A B C : ℝ) 
  (h₁ : Real.sin A / Real.sin B = 5 / 7)
  (h₂ : Real.sin B / Real.sin C = 7 / 8) : 
  B = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_angle_B_equal_pi_div_3_l522_52215


namespace NUMINAMATH_GPT_electronics_sale_negation_l522_52284

variables (E : Type) (storeElectronics : E → Prop) (onSale : E → Prop)

theorem electronics_sale_negation
  (H : ¬ ∀ e, storeElectronics e → onSale e) :
  (∃ e, storeElectronics e ∧ ¬ onSale e) ∧ ¬ ∀ e, storeElectronics e → onSale e :=
by
  -- Proving that at least one electronic is not on sale follows directly from the negation of the universal statement
  sorry

end NUMINAMATH_GPT_electronics_sale_negation_l522_52284


namespace NUMINAMATH_GPT_mary_biking_time_l522_52269

-- Define the conditions and the task
def total_time_away := 570 -- in minutes
def time_in_classes := 7 * 45 -- in minutes
def lunch_time := 40 -- in minutes
def additional_activities := 105 -- in minutes
def time_in_school_activities := time_in_classes + lunch_time + additional_activities

-- Define the total biking time based on given conditions
theorem mary_biking_time : 
  total_time_away - time_in_school_activities = 110 :=
by 
-- sorry is used to skip the proof step.
  sorry

end NUMINAMATH_GPT_mary_biking_time_l522_52269


namespace NUMINAMATH_GPT_van_distance_l522_52253

theorem van_distance (D : ℝ) (t_initial t_new : ℝ) (speed_new : ℝ) 
  (h1 : t_initial = 6) 
  (h2 : t_new = (3 / 2) * t_initial) 
  (h3 : speed_new = 30) 
  (h4 : D = speed_new * t_new) : 
  D = 270 :=
by
  sorry

end NUMINAMATH_GPT_van_distance_l522_52253


namespace NUMINAMATH_GPT_tap_fills_tank_without_leakage_in_12_hours_l522_52271

theorem tap_fills_tank_without_leakage_in_12_hours 
  (R_t R_l : ℝ)
  (h1 : (R_t - R_l) * 18 = 1)
  (h2 : R_l * 36 = 1) :
  1 / R_t = 12 := 
by
  sorry

end NUMINAMATH_GPT_tap_fills_tank_without_leakage_in_12_hours_l522_52271


namespace NUMINAMATH_GPT_first_discount_percentage_l522_52205

theorem first_discount_percentage (x : ℝ) 
  (h₁ : ∀ (p : ℝ), p = 70) 
  (h₂ : ∀ (d₁ d₂ : ℝ), d₁ = x / 100 ∧ d₂ = 0.01999999999999997 )
  (h₃ : ∀ (final_price : ℝ), final_price = 61.74):
  x = 10 := 
by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l522_52205


namespace NUMINAMATH_GPT_repeating_decimal_product_l522_52236

theorem repeating_decimal_product (x : ℚ) (h : x = 4 / 9) : x * 9 = 4 := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_product_l522_52236


namespace NUMINAMATH_GPT_find_unknown_rate_l522_52227

theorem find_unknown_rate :
    let n := 7 -- total number of blankets
    let avg_price := 150 -- average price of the blankets
    let total_price := n * avg_price
    let cost1 := 3 * 100
    let cost2 := 2 * 150
    let remaining := total_price - (cost1 + cost2)
    remaining / 2 = 225 :=
by sorry

end NUMINAMATH_GPT_find_unknown_rate_l522_52227


namespace NUMINAMATH_GPT_paths_inequality_l522_52288
open Nat

-- Definitions
def m : ℕ := sorry -- m represents the number of rows.
def n : ℕ := sorry -- n represents the number of columns.
def N : ℕ := sorry -- N is the number of ways to color the grid such that there is a path composed of black cells from the left edge to the right edge.
def M : ℕ := sorry -- M is the number of ways to color the grid such that there are two non-intersecting paths composed of black cells from the left edge to the right edge.

-- Theorem statement
theorem paths_inequality : (N ^ 2) ≥ 2 ^ (m * n) * M := 
by
  sorry

end NUMINAMATH_GPT_paths_inequality_l522_52288


namespace NUMINAMATH_GPT_kyle_practice_time_l522_52261

-- Definitions for the conditions
def weightlifting_time : ℕ := 20  -- in minutes
def running_time : ℕ := 2 * weightlifting_time  -- twice the weightlifting time
def total_running_and_weightlifting_time : ℕ := weightlifting_time + running_time  -- total time for running and weightlifting
def shooting_time : ℕ := total_running_and_weightlifting_time  -- because it's half the practice time

-- Total daily practice time, in minutes
def total_practice_time_minutes : ℕ := shooting_time + total_running_and_weightlifting_time

-- Total daily practice time, in hours
def total_practice_time_hours : ℕ := total_practice_time_minutes / 60

-- Theorem stating that Kyle practices for 2 hours every day given the conditions
theorem kyle_practice_time : total_practice_time_hours = 2 := by
  sorry

end NUMINAMATH_GPT_kyle_practice_time_l522_52261


namespace NUMINAMATH_GPT_bahs_for_1000_yahs_l522_52248

-- Definitions based on given conditions
def bahs_to_rahs_ratio (b r : ℕ) := 15 * b = 24 * r
def rahs_to_yahs_ratio (r y : ℕ) := 9 * r = 15 * y

-- Main statement to prove
theorem bahs_for_1000_yahs (b r y : ℕ) (h1 : bahs_to_rahs_ratio b r) (h2 : rahs_to_yahs_ratio r y) :
  1000 * y = 375 * b :=
by
  sorry

end NUMINAMATH_GPT_bahs_for_1000_yahs_l522_52248


namespace NUMINAMATH_GPT_verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l522_52222

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Definition for conversions from base 60 to base 10
def from_base_60 (d1 d0 : ℕ) : ℕ :=
  d1 * 60 + d0

-- Proof statements
theorem verify_21_base_60 : from_base_60 2 1 = 121 ∧ is_perfect_square 121 :=
by {
  sorry
}

theorem verify_1_base_60 : from_base_60 0 1 = 1 ∧ is_perfect_square 1 :=
by {
  sorry
}

theorem verify_2_base_60_not_square : from_base_60 0 2 = 2 ∧ ¬ is_perfect_square 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l522_52222


namespace NUMINAMATH_GPT_circle_equation_l522_52228

theorem circle_equation 
  (P : ℝ × ℝ)
  (h1 : ∀ a : ℝ, (1 - a) * 2 + (P.snd) + 2 * a - 1 = 0)
  (h2 : P = (2, -1)) :
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end NUMINAMATH_GPT_circle_equation_l522_52228


namespace NUMINAMATH_GPT_zachary_more_crunches_than_pushups_l522_52200

def zachary_pushups : ℕ := 46
def zachary_crunches : ℕ := 58
def zachary_crunches_more_than_pushups : ℕ := zachary_crunches - zachary_pushups

theorem zachary_more_crunches_than_pushups : zachary_crunches_more_than_pushups = 12 := by
  sorry

end NUMINAMATH_GPT_zachary_more_crunches_than_pushups_l522_52200


namespace NUMINAMATH_GPT_total_toads_l522_52211

def pond_toads : ℕ := 12
def outside_toads : ℕ := 6

theorem total_toads : pond_toads + outside_toads = 18 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_toads_l522_52211


namespace NUMINAMATH_GPT_percentage_of_first_to_second_l522_52296

theorem percentage_of_first_to_second (X : ℝ) (h1 : first = (7/100) * X) (h2 : second = (14/100) * X) : (first / second) * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_first_to_second_l522_52296


namespace NUMINAMATH_GPT_greatest_divisor_6215_7373_l522_52223

theorem greatest_divisor_6215_7373 : 
  Nat.gcd (6215 - 23) (7373 - 29) = 144 := by
  sorry

end NUMINAMATH_GPT_greatest_divisor_6215_7373_l522_52223


namespace NUMINAMATH_GPT_parallel_vectors_l522_52206

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (P : a = (1, m) ∧ b = (m, 2) ∧ (a.1 / m = b.1 / 2)) :
  m = -Real.sqrt 2 ∨ m = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l522_52206


namespace NUMINAMATH_GPT_original_cost_l522_52267

theorem original_cost (A : ℝ) (discount : ℝ) (sale_price : ℝ) (original_price : ℝ) (h1 : discount = 0.30) (h2 : sale_price = 35) (h3 : sale_price = (1 - discount) * original_price) : 
  original_price = 50 := by
  sorry

end NUMINAMATH_GPT_original_cost_l522_52267


namespace NUMINAMATH_GPT_sequence_sum_l522_52249

theorem sequence_sum (a : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, 0 < a n)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 2) = 1 + 1 / a n)
  (h₃ : a 2014 = a 2016) :
  a 13 + a 2016 = 21 / 13 + (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_sequence_sum_l522_52249


namespace NUMINAMATH_GPT_problem_l522_52273

theorem problem (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a + b + c)) : (a + b) * (b + c) * (a + c) = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_l522_52273


namespace NUMINAMATH_GPT_boxes_per_week_l522_52299

-- Define the given conditions
def cost_per_box : ℝ := 3.00
def weeks_in_year : ℝ := 52
def total_spent_per_year : ℝ := 312

-- The question we want to prove:
theorem boxes_per_week:
  (total_spent_per_year = cost_per_box * weeks_in_year * (total_spent_per_year / (weeks_in_year * cost_per_box))) → 
  (total_spent_per_year / (weeks_in_year * cost_per_box)) = 2 := sorry

end NUMINAMATH_GPT_boxes_per_week_l522_52299


namespace NUMINAMATH_GPT_find_radius_of_stationary_tank_l522_52232

theorem find_radius_of_stationary_tank
  (h_stationary : Real) (r_truck : Real) (h_truck : Real) (drop : Real) (V_truck : Real)
  (ht1 : h_stationary = 25)
  (ht2 : r_truck = 4)
  (ht3 : h_truck = 10)
  (ht4 : drop = 0.016)
  (ht5 : V_truck = π * r_truck ^ 2 * h_truck) :
  ∃ R : Real, π * R ^ 2 * drop = V_truck ∧ R = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_of_stationary_tank_l522_52232


namespace NUMINAMATH_GPT_xiaoGong_walking_speed_l522_52214

-- Defining the parameters for the problem
def distance : ℕ := 1200
def daChengExtraSpeedPerMinute : ℕ := 20
def timeUntilMeetingForDaCheng : ℕ := 12
def timeUntilMeetingForXiaoGong : ℕ := 6 + timeUntilMeetingForDaCheng

-- The main statement to prove Xiao Gong's speed
theorem xiaoGong_walking_speed : ∃ v : ℕ, 12 * (v + daChengExtraSpeedPerMinute) + 18 * v = distance ∧ v = 32 :=
by
  sorry

end NUMINAMATH_GPT_xiaoGong_walking_speed_l522_52214


namespace NUMINAMATH_GPT_find_principal_l522_52250

theorem find_principal
  (R : ℝ) (T : ℕ) (interest_less_than_principal : ℝ) : 
  R = 0.05 → 
  T = 10 → 
  interest_less_than_principal = 3100 → 
  ∃ P : ℝ, P - ((P * R * T): ℝ) = P - interest_less_than_principal ∧ P = 6200 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l522_52250


namespace NUMINAMATH_GPT_sam_won_total_matches_l522_52255

/-- Sam's first 100 matches and he won 50% of them -/
def first_100_matches : ℕ := 100

/-- Sam won 50% of his first 100 matches -/
def win_rate_first : ℕ := 50

/-- Sam's next 100 matches and he won 60% of them -/
def next_100_matches : ℕ := 100

/-- Sam won 60% of his next 100 matches -/
def win_rate_next : ℕ := 60

/-- The total number of matches Sam won -/
def total_matches_won (first_100_matches: ℕ) (win_rate_first: ℕ) (next_100_matches: ℕ) (win_rate_next: ℕ) : ℕ :=
  (first_100_matches * win_rate_first) / 100 + (next_100_matches * win_rate_next) / 100

theorem sam_won_total_matches :
  total_matches_won first_100_matches win_rate_first next_100_matches win_rate_next = 110 :=
by
  sorry

end NUMINAMATH_GPT_sam_won_total_matches_l522_52255


namespace NUMINAMATH_GPT_min_value_ineq_l522_52268

noncomputable def function_y (a : ℝ) (x : ℝ) : ℝ := a^(1-x)

theorem min_value_ineq (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_ineq_l522_52268


namespace NUMINAMATH_GPT_algebraic_expression_constant_l522_52270

theorem algebraic_expression_constant (x : ℝ) : x * (x - 6) - (3 - x) ^ 2 = -9 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_constant_l522_52270


namespace NUMINAMATH_GPT_div_by_90_l522_52245

def N : ℤ := 19^92 - 91^29

theorem div_by_90 : ∃ k : ℤ, N = 90 * k := 
sorry

end NUMINAMATH_GPT_div_by_90_l522_52245


namespace NUMINAMATH_GPT_gcd_lcm_sum_l522_52295

theorem gcd_lcm_sum :
  gcd 42 70 + lcm 15 45 = 59 :=
by sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l522_52295


namespace NUMINAMATH_GPT_base_8_not_divisible_by_five_l522_52251

def base_b_subtraction_not_divisible_by_five (b : ℕ) : Prop :=
  let num1 := 3 * b^3 + 1 * b^2 + 0 * b + 2
  let num2 := 3 * b^2 + 0 * b + 2
  let diff := num1 - num2
  ¬ (diff % 5 = 0)

theorem base_8_not_divisible_by_five : base_b_subtraction_not_divisible_by_five 8 := 
by
  sorry

end NUMINAMATH_GPT_base_8_not_divisible_by_five_l522_52251


namespace NUMINAMATH_GPT_maximum_value_l522_52235

theorem maximum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  (a / (a + 1) + b / (b + 2) ≤ (5 - 2 * Real.sqrt 2) / 4) :=
sorry

end NUMINAMATH_GPT_maximum_value_l522_52235


namespace NUMINAMATH_GPT_back_seat_people_l522_52213

-- Define the problem conditions

def leftSideSeats : ℕ := 15
def seatDifference : ℕ := 3
def peoplePerSeat : ℕ := 3
def totalBusCapacity : ℕ := 88

-- Define the formula for calculating the people at the back seat
def peopleAtBackSeat := 
  totalBusCapacity - ((leftSideSeats * peoplePerSeat) + ((leftSideSeats - seatDifference) * peoplePerSeat))

-- The statement we need to prove
theorem back_seat_people : peopleAtBackSeat = 7 :=
by
  sorry

end NUMINAMATH_GPT_back_seat_people_l522_52213


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l522_52209

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x - 1

theorem problem_part_1 (m n : ℝ) :
  (∀ x, f x m < 0 ↔ -2 < x ∧ x < n) → m = 5 / 2 ∧ n = 1 / 2 :=
sorry

theorem problem_part_2 (m : ℝ) :
  (∀ x, m ≤ x ∧ x ≤ m + 1 → f x m < 0) → m ∈ Set.Ioo (-Real.sqrt (2) / 2) 0 :=
sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l522_52209


namespace NUMINAMATH_GPT_crease_points_ellipse_l522_52258

theorem crease_points_ellipse (R a : ℝ) (x y : ℝ) (h1 : 0 < R) (h2 : 0 < a) (h3 : a < R) : 
  (x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) ≥ 1 :=
by
  -- Omitted detailed proof steps
  sorry

end NUMINAMATH_GPT_crease_points_ellipse_l522_52258
