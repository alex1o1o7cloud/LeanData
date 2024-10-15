import Mathlib

namespace NUMINAMATH_GPT_sufficiency_but_not_necessity_l2107_210761

theorem sufficiency_but_not_necessity (a b : ℝ) :
  (a = 0 → a * b = 0) ∧ (a * b = 0 → a = 0) → False :=
by
   -- Proof is skipped
   sorry

end NUMINAMATH_GPT_sufficiency_but_not_necessity_l2107_210761


namespace NUMINAMATH_GPT_highest_prob_of_red_card_l2107_210726

theorem highest_prob_of_red_card :
  let deck_size := 52
  let num_aces := 4
  let num_hearts := 13
  let num_kings := 4
  let num_reds := 26
  -- Event probabilities
  let prob_ace := num_aces / deck_size
  let prob_heart := num_hearts / deck_size
  let prob_king := num_kings / deck_size
  let prob_red := num_reds / deck_size
  prob_red > prob_heart ∧ prob_heart > prob_ace ∧ prob_ace = prob_king :=
sorry

end NUMINAMATH_GPT_highest_prob_of_red_card_l2107_210726


namespace NUMINAMATH_GPT_at_least_one_le_one_l2107_210791

theorem at_least_one_le_one (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end NUMINAMATH_GPT_at_least_one_le_one_l2107_210791


namespace NUMINAMATH_GPT_nonnegative_integer_solutions_l2107_210701

theorem nonnegative_integer_solutions (x y : ℕ) :
  3 * x^2 + 2 * 9^y = x * (4^(y+1) - 1) ↔ (x, y) ∈ [(2, 1), (3, 1), (3, 2), (18, 2)] :=
by sorry

end NUMINAMATH_GPT_nonnegative_integer_solutions_l2107_210701


namespace NUMINAMATH_GPT_inequality_sqrt_l2107_210772

theorem inequality_sqrt (m n : ℕ) (h : m < n) : 
  (m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_sqrt_l2107_210772


namespace NUMINAMATH_GPT_prime_power_of_n_l2107_210719

theorem prime_power_of_n (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end NUMINAMATH_GPT_prime_power_of_n_l2107_210719


namespace NUMINAMATH_GPT_students_taking_music_l2107_210732

theorem students_taking_music
  (total_students : Nat)
  (students_taking_art : Nat)
  (students_taking_both : Nat)
  (students_taking_neither : Nat)
  (total_eq : total_students = 500)
  (art_eq : students_taking_art = 20)
  (both_eq : students_taking_both = 10)
  (neither_eq : students_taking_neither = 440) :
  ∃ M : Nat, M = 50 := by
  sorry

end NUMINAMATH_GPT_students_taking_music_l2107_210732


namespace NUMINAMATH_GPT_smallest_area_2020th_square_l2107_210764

theorem smallest_area_2020th_square (n : ℕ) :
  (∃ n : ℕ, n^2 > 2019 ∧ ∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1) →
  (∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1 ∧ A = 6) :=
sorry

end NUMINAMATH_GPT_smallest_area_2020th_square_l2107_210764


namespace NUMINAMATH_GPT_fraction_zero_implies_x_zero_l2107_210710

theorem fraction_zero_implies_x_zero (x : ℝ) (h : (x^2 - x) / (x - 1) = 0) (h₁ : x ≠ 1) : x = 0 := by
  sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_zero_l2107_210710


namespace NUMINAMATH_GPT_smallest_positive_debt_resolves_l2107_210736

theorem smallest_positive_debt_resolves :
  ∃ (c t : ℤ), (240 * c + 180 * t = 60) ∧ (60 > 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_debt_resolves_l2107_210736


namespace NUMINAMATH_GPT_inequality_m_2n_l2107_210725

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

lemma max_f : ∃ x : ℝ, f x = 2 :=
sorry

theorem inequality_m_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 1/(2*n) = 2) : m + 2*n ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_m_2n_l2107_210725


namespace NUMINAMATH_GPT_sum_twice_father_age_plus_son_age_l2107_210743

/-- 
  Given:
  1. Twice the son's age plus the father's age equals 70.
  2. Father's age is 40.
  3. Son's age is 15.

  Prove:
  The sum when twice the father's age is added to the son's age is 95.
-/
theorem sum_twice_father_age_plus_son_age :
  ∀ (father_age son_age : ℕ), 
    2 * son_age + father_age = 70 → 
    father_age = 40 → 
    son_age = 15 → 
    2 * father_age + son_age = 95 := by
  intros
  sorry

end NUMINAMATH_GPT_sum_twice_father_age_plus_son_age_l2107_210743


namespace NUMINAMATH_GPT_proof_equivalence_l2107_210716

variables {a b c d e f : Prop}

theorem proof_equivalence (h₁ : (a ≥ b) → (c > d)) 
                        (h₂ : (c > d) → (a ≥ b)) 
                        (h₃ : (a < b) ↔ (e ≤ f)) :
  (c ≤ d) ↔ (e ≤ f) :=
sorry

end NUMINAMATH_GPT_proof_equivalence_l2107_210716


namespace NUMINAMATH_GPT_cubic_sum_identity_l2107_210718

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + a * c + b * c = -6) (h3 : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_identity_l2107_210718


namespace NUMINAMATH_GPT_length_of_XY_l2107_210754

-- Defining the points on the circle
variables (A B C D P Q X Y : Type*)
-- Lengths given in the problem
variables (AB_len CD_len AP_len CQ_len PQ_len : ℕ)
-- Points and lengths conditions
variables (h1 : AB_len = 11) (h2 : CD_len = 19)
variables (h3 : AP_len = 6) (h4 : CQ_len = 7)
variables (h5 : PQ_len = 27)

-- Assuming the Power of a Point theorem applied to P and Q
variables (PX_len PY_len QX_len QY_len : ℕ)
variables (h6 : PX_len = 1) (h7 : QY_len = 3)
variables (h8 : PX_len + PQ_len + QY_len = XY_len)

-- The final length of XY is to be found
def XY_len : ℕ := PX_len + PQ_len + QY_len

-- The goal is to show XY = 31
theorem length_of_XY : XY_len = 31 :=
  by
    sorry

end NUMINAMATH_GPT_length_of_XY_l2107_210754


namespace NUMINAMATH_GPT_linear_equation_a_zero_l2107_210740

theorem linear_equation_a_zero (a : ℝ) : 
  ((a - 2) * x ^ (abs (a - 1)) + 3 = 9) ∧ (abs (a - 1) = 1) → a = 0 := by
  sorry

end NUMINAMATH_GPT_linear_equation_a_zero_l2107_210740


namespace NUMINAMATH_GPT_sum_first_50_arithmetic_sequence_l2107_210749

theorem sum_first_50_arithmetic_sequence : 
  let a : ℕ := 2
  let d : ℕ := 4
  let n : ℕ := 50
  let a_n (n : ℕ) : ℕ := a + (n - 1) * d
  let S_n (n : ℕ) : ℕ := n / 2 * (2 * a + (n - 1) * d)
  S_n n = 5000 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_50_arithmetic_sequence_l2107_210749


namespace NUMINAMATH_GPT_remaining_balance_is_correct_l2107_210773

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end NUMINAMATH_GPT_remaining_balance_is_correct_l2107_210773


namespace NUMINAMATH_GPT_malingerers_exposed_l2107_210739

theorem malingerers_exposed (a b c : Nat) (ha : a > b) (hc : c = b + 9) :
  let aabbb := 10000 * a + 1000 * a + 100 * b + 10 * b + b
  let abccc := 10000 * a + 1000 * b + 100 * c + 10 * c + c
  (aabbb - 1 = abccc) -> abccc = 10999 :=
by
  sorry

end NUMINAMATH_GPT_malingerers_exposed_l2107_210739


namespace NUMINAMATH_GPT_marbles_per_pack_l2107_210702

theorem marbles_per_pack (total_marbles : ℕ) (leo_packs manny_packs neil_packs total_packs : ℕ) 
(h1 : total_marbles = 400) 
(h2 : leo_packs = 25) 
(h3 : manny_packs = total_packs / 4) 
(h4 : neil_packs = total_packs / 8) 
(h5 : leo_packs + manny_packs + neil_packs = total_packs) : 
total_marbles / total_packs = 10 := 
by sorry

end NUMINAMATH_GPT_marbles_per_pack_l2107_210702


namespace NUMINAMATH_GPT_pradeep_pass_percentage_l2107_210708

-- Define the given data as constants
def score : ℕ := 185
def shortfall : ℕ := 25
def maxMarks : ℕ := 840

-- Calculate the passing mark
def passingMark : ℕ := score + shortfall

-- Calculate the percentage needed to pass
def passPercentage (passingMark : ℕ) (maxMarks : ℕ) : ℕ :=
  (passingMark * 100) / maxMarks

-- Statement of the theorem that we aim to prove
theorem pradeep_pass_percentage (score shortfall maxMarks : ℕ)
  (h_score : score = 185) (h_shortfall : shortfall = 25) (h_maxMarks : maxMarks = 840) :
  passPercentage (score + shortfall) maxMarks = 25 :=
by
  -- This is where the proof would go
  sorry

-- Example of calling the function to ensure definitions are correct
#eval passPercentage (score + shortfall) maxMarks -- Should output 25

end NUMINAMATH_GPT_pradeep_pass_percentage_l2107_210708


namespace NUMINAMATH_GPT_no_solution_part_a_no_solution_part_b_l2107_210790

theorem no_solution_part_a 
  (x y z : ℕ) :
  ¬(x^2 + y^2 + z^2 = 2 * x * y * z) := 
sorry

theorem no_solution_part_b 
  (x y z u : ℕ) :
  ¬(x^2 + y^2 + z^2 + u^2 = 2 * x * y * z * u) := 
sorry

end NUMINAMATH_GPT_no_solution_part_a_no_solution_part_b_l2107_210790


namespace NUMINAMATH_GPT_fruit_problem_l2107_210787

theorem fruit_problem :
  let apples_initial := 7
  let oranges_initial := 8
  let mangoes_initial := 15
  let grapes_initial := 12
  let strawberries_initial := 5
  let apples_taken := 3
  let oranges_taken := 4
  let mangoes_taken := 4
  let grapes_taken := 7
  let strawberries_taken := 3
  let apples_remaining := apples_initial - apples_taken
  let oranges_remaining := oranges_initial - oranges_taken
  let mangoes_remaining := mangoes_initial - mangoes_taken
  let grapes_remaining := grapes_initial - grapes_taken
  let strawberries_remaining := strawberries_initial - strawberries_taken
  let total_remaining := apples_remaining + oranges_remaining + mangoes_remaining + grapes_remaining + strawberries_remaining
  let total_taken := apples_taken + oranges_taken + mangoes_taken + grapes_taken + strawberries_taken
  total_remaining = 26 ∧ total_taken = 21 := by
    sorry

end NUMINAMATH_GPT_fruit_problem_l2107_210787


namespace NUMINAMATH_GPT_parking_space_length_l2107_210724

theorem parking_space_length {L W : ℕ} 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 126) : 
  L = 9 := 
sorry

end NUMINAMATH_GPT_parking_space_length_l2107_210724


namespace NUMINAMATH_GPT_selling_price_l2107_210713

theorem selling_price 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (h_cost : cost_price = 192) 
  (h_profit : profit_percentage = 0.25) : 
  ∃ selling_price : ℝ, selling_price = cost_price * (1 + profit_percentage) := 
by {
  sorry
}

end NUMINAMATH_GPT_selling_price_l2107_210713


namespace NUMINAMATH_GPT_floor_T_equals_150_l2107_210798

variable {p q r s : ℝ}

theorem floor_T_equals_150
  (hpq_sum_of_squares : p^2 + q^2 = 2500)
  (hrs_sum_of_squares : r^2 + s^2 = 2500)
  (hpq_product : p * q = 1225)
  (hrs_product : r * s = 1225)
  (hp_plus_s : p + s = 75) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 150 :=
by
  sorry

end NUMINAMATH_GPT_floor_T_equals_150_l2107_210798


namespace NUMINAMATH_GPT_felix_trees_chopped_l2107_210792

-- Given conditions
def cost_per_sharpening : ℕ := 8
def total_spent : ℕ := 48
def trees_per_sharpening : ℕ := 25

-- Lean statement of the problem
theorem felix_trees_chopped (h : total_spent / cost_per_sharpening * trees_per_sharpening >= 150) : True :=
by {
  -- This is just a placeholder for the proof.
  sorry
}

end NUMINAMATH_GPT_felix_trees_chopped_l2107_210792


namespace NUMINAMATH_GPT_total_amount_received_l2107_210756

theorem total_amount_received (h1 : 12 = 12)
                              (h2 : 10 = 10)
                              (h3 : 8 = 8)
                              (h4 : 14 = 14)
                              (rate : 15 = 15) :
  (3 * (12 + 10 + 8 + 14) * 15) = 1980 :=
by sorry

end NUMINAMATH_GPT_total_amount_received_l2107_210756


namespace NUMINAMATH_GPT_total_pages_in_book_l2107_210747

theorem total_pages_in_book :
  ∃ x : ℝ, (x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20)
           - (1/2 * ((x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20))) + 25) = 120) ∧
           x = 552 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l2107_210747


namespace NUMINAMATH_GPT_polikarp_make_first_box_empty_l2107_210746

theorem polikarp_make_first_box_empty (n : ℕ) (h : n ≤ 30) : ∃ (x y : ℕ), x + y ≤ 10 ∧ ∀ k : ℕ, k ≤ x → k + k * y = n :=
by
  sorry

end NUMINAMATH_GPT_polikarp_make_first_box_empty_l2107_210746


namespace NUMINAMATH_GPT_haley_total_expenditure_l2107_210759

-- Definition of conditions
def ticket_cost : ℕ := 4
def tickets_bought_for_self_and_friends : ℕ := 3
def tickets_bought_for_others : ℕ := 5
def total_tickets : ℕ := tickets_bought_for_self_and_friends + tickets_bought_for_others

-- Proof statement
theorem haley_total_expenditure : total_tickets * ticket_cost = 32 := by
  sorry

end NUMINAMATH_GPT_haley_total_expenditure_l2107_210759


namespace NUMINAMATH_GPT_find_number_l2107_210753

theorem find_number (x : ℝ) : 35 + 3 * x^2 = 89 ↔ x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_find_number_l2107_210753


namespace NUMINAMATH_GPT_spadesuit_eval_l2107_210705

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 2 3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_eval_l2107_210705


namespace NUMINAMATH_GPT_line_intersects_parabola_at_9_units_apart_l2107_210758

theorem line_intersects_parabola_at_9_units_apart :
  ∃ m b, (∃ (k1 k2 : ℝ), 
              (y1 = k1^2 + 6*k1 - 4) ∧ 
              (y2 = k2^2 + 6*k2 - 4) ∧ 
              (y1 = m*k1 + b) ∧ 
              (y2 = m*k2 + b) ∧ 
              |y1 - y2| = 9) ∧ 
          (0 ≠ b) ∧ 
          ((1 : ℝ) = 2*m + b) ∧ 
          (m = 4 ∧ b = -7)
:= sorry

end NUMINAMATH_GPT_line_intersects_parabola_at_9_units_apart_l2107_210758


namespace NUMINAMATH_GPT_find_intersecting_lines_l2107_210715

theorem find_intersecting_lines (x y : ℝ) : 
  (2 * x - y)^2 - (x + 3 * y)^2 = 0 ↔ x = 4 * y ∨ x = - (2 / 3) * y :=
by
  sorry

end NUMINAMATH_GPT_find_intersecting_lines_l2107_210715


namespace NUMINAMATH_GPT_total_insects_eaten_l2107_210770

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end NUMINAMATH_GPT_total_insects_eaten_l2107_210770


namespace NUMINAMATH_GPT_nigel_gave_away_l2107_210752

theorem nigel_gave_away :
  ∀ (original : ℕ) (gift_from_mother : ℕ) (final : ℕ) (money_given_away : ℕ),
    original = 45 →
    gift_from_mother = 80 →
    final = 2 * original + 10 →
    final = original - money_given_away + gift_from_mother →
    money_given_away = 25 :=
by
  intros original gift_from_mother final money_given_away
  sorry

end NUMINAMATH_GPT_nigel_gave_away_l2107_210752


namespace NUMINAMATH_GPT_product_of_roots_eq_neg30_l2107_210795

theorem product_of_roots_eq_neg30 (x : ℝ) (h : (x + 3) * (x - 4) = 18) : 
  (∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = -30) :=
sorry

end NUMINAMATH_GPT_product_of_roots_eq_neg30_l2107_210795


namespace NUMINAMATH_GPT_quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l2107_210774

theorem quadratic_equation_root_conditions
  (k : ℝ)
  (h_discriminant : 4 * k - 3 > 0)
  (h_sum_product : ∀ (x1 x2 : ℝ),
    x1 + x2 = -(2 * k + 1) ∧ 
    x1 * x2 = k^2 + 1 →
    x1 + x2 + 2 * (x1 * x2) = 1) :
  k = 1 :=
by
  sorry

theorem quadratic_equation_distinct_real_roots
  (k : ℝ) :
  (∃ (x1 x2 : ℝ),
    x1 ≠ x2 ∧
    x1^2 + (2 * k + 1) * x1 + (k^2 + 1) = 0 ∧
    x2^2 + (2 * k + 1) * x2 + (k^2 + 1) = 0) ↔
  k > 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l2107_210774


namespace NUMINAMATH_GPT_fraction_to_decimal_l2107_210796

theorem fraction_to_decimal :
  (17 : ℚ) / (2^2 * 5^4) = 0.0068 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2107_210796


namespace NUMINAMATH_GPT_number_of_six_digit_palindromes_l2107_210733

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end NUMINAMATH_GPT_number_of_six_digit_palindromes_l2107_210733


namespace NUMINAMATH_GPT_trapezoid_area_division_l2107_210737

theorem trapezoid_area_division (AD BC MN : ℝ) (h₁ : AD = 4) (h₂ : BC = 3)
  (h₃ : MN > 0) (area_ratio : ∃ (S_ABMD S_MBCN : ℝ), MN/BC = (S_ABMD + S_MBCN)/(S_ABMD) ∧ (S_ABMD/S_MBCN = 2/5)) :
  MN = Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_division_l2107_210737


namespace NUMINAMATH_GPT_logarithmic_expression_evaluation_l2107_210745

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression_evaluation : 
  log_base_10 (5 / 2) + 2 * log_base_10 2 - (1/2)⁻¹ = -1 := 
by 
  sorry

end NUMINAMATH_GPT_logarithmic_expression_evaluation_l2107_210745


namespace NUMINAMATH_GPT_fourth_leg_length_l2107_210783

theorem fourth_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) :
  (∃ x : ℕ, x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ (a + x = b + c ∨ b + x = a + c ∨ c + x = a + b) ∧ (x = 7 ∨ x = 11)) :=
by sorry

end NUMINAMATH_GPT_fourth_leg_length_l2107_210783


namespace NUMINAMATH_GPT_total_worth_of_stock_l2107_210704

noncomputable def shop_equation (X : ℝ) : Prop :=
  0.04 * X - 0.02 * X = 400

theorem total_worth_of_stock :
  ∃ (X : ℝ), shop_equation X ∧ X = 20000 :=
by
  use 20000
  have h : shop_equation 20000 := by
    unfold shop_equation
    norm_num
  exact ⟨h, rfl⟩

end NUMINAMATH_GPT_total_worth_of_stock_l2107_210704


namespace NUMINAMATH_GPT_sampling_is_stratified_l2107_210755

-- Given Conditions
def number_of_male_students := 500
def number_of_female_students := 400
def sampled_male_students := 25
def sampled_female_students := 20

-- Definition of stratified sampling according to the problem context
def is_stratified_sampling (N_M F_M R_M R_F : ℕ) : Prop :=
  (R_M > 0 ∧ R_F > 0 ∧ R_M < N_M ∧ R_F < N_M ∧ N_M > 0 ∧ N_M > 0)

-- Proving that the sampling method is stratified sampling
theorem sampling_is_stratified : 
  is_stratified_sampling number_of_male_students number_of_female_students sampled_male_students sampled_female_students = true :=
by
  sorry

end NUMINAMATH_GPT_sampling_is_stratified_l2107_210755


namespace NUMINAMATH_GPT_sum_first_five_special_l2107_210757

def is_special (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

theorem sum_first_five_special :
  let special_numbers := [36, 100, 196, 484, 676]
  (∀ n ∈ special_numbers, is_special n) →
  special_numbers.sum = 1492 := by {
  sorry
}

end NUMINAMATH_GPT_sum_first_five_special_l2107_210757


namespace NUMINAMATH_GPT_value_of_expression_l2107_210721

-- Define the conditions
def x := -2
def y := 1
def z := 1
def w := 3

-- The main theorem statement
theorem value_of_expression : 
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = - (1 / 3) * Real.sin 2 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2107_210721


namespace NUMINAMATH_GPT_dolphins_to_be_trained_next_month_l2107_210744

theorem dolphins_to_be_trained_next_month :
  ∀ (total_dolphins fully_trained remaining trained_next_month : ℕ),
    total_dolphins = 20 →
    fully_trained = (1 / 4 : ℚ) * total_dolphins →
    remaining = total_dolphins - fully_trained →
    (2 / 3 : ℚ) * remaining = 10 →
    trained_next_month = remaining - 10 →
    trained_next_month = 5 :=
by
  intros total_dolphins fully_trained remaining trained_next_month
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_dolphins_to_be_trained_next_month_l2107_210744


namespace NUMINAMATH_GPT_subjects_difference_marius_monica_l2107_210766

-- Definitions of given conditions.
def Monica_subjects : ℕ := 10
def Total_subjects : ℕ := 41
def Millie_offset : ℕ := 3

-- Theorem to prove the question == answer given conditions
theorem subjects_difference_marius_monica : 
  ∃ (M : ℕ), (M + (M + Millie_offset) + Monica_subjects = Total_subjects) ∧ (M - Monica_subjects = 4) := 
by
  sorry

end NUMINAMATH_GPT_subjects_difference_marius_monica_l2107_210766


namespace NUMINAMATH_GPT_count_president_vp_secretary_l2107_210712

theorem count_president_vp_secretary (total_members boys girls : ℕ) (total_members_eq : total_members = 30) 
(boys_eq : boys = 18) (girls_eq : girls = 12) :
  ∃ (ways : ℕ), 
  ways = (boys * girls * (boys - 1) + girls * boys * (girls - 1)) ∧
  ways = 6048 :=
by
  sorry

end NUMINAMATH_GPT_count_president_vp_secretary_l2107_210712


namespace NUMINAMATH_GPT_kite_area_overlap_l2107_210731

theorem kite_area_overlap (beta : Real) (h_beta : beta ≠ 0 ∧ beta ≠ π) : 
  ∃ (A : Real), A = 1 / Real.sin beta := by
  sorry

end NUMINAMATH_GPT_kite_area_overlap_l2107_210731


namespace NUMINAMATH_GPT_pump_filling_time_without_leak_l2107_210769

theorem pump_filling_time_without_leak (P : ℝ) (h1 : 1 / P - 1 / 14 = 3 / 7) : P = 2 :=
sorry

end NUMINAMATH_GPT_pump_filling_time_without_leak_l2107_210769


namespace NUMINAMATH_GPT_apples_given_by_nathan_l2107_210767

theorem apples_given_by_nathan (initial_apples : ℕ) (total_apples : ℕ) (given_by_nathan : ℕ) :
  initial_apples = 6 → total_apples = 12 → given_by_nathan = (total_apples - initial_apples) → given_by_nathan = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_apples_given_by_nathan_l2107_210767


namespace NUMINAMATH_GPT_cindy_correct_operation_l2107_210742

-- Let's define the conditions and proof statement in Lean 4.

variable (x : ℝ)
axiom incorrect_operation : (x - 7) / 5 = 25

theorem cindy_correct_operation :
  (x - 5) / 7 = 18 + 1 / 7 :=
sorry

end NUMINAMATH_GPT_cindy_correct_operation_l2107_210742


namespace NUMINAMATH_GPT_min_colors_needed_correct_l2107_210776

-- Define the 5x5 grid as a type
def Grid : Type := Fin 5 × Fin 5

-- Define a coloring as a function from Grid to a given number of colors
def Coloring (colors : Type) : Type := Grid → colors

-- Define the property where in any row, column, or diagonal, no three consecutive cells have the same color
def valid_coloring (colors : Type) (C : Coloring colors) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 3, ( C (i, j) ≠ C (i, j + 1) ∧ C (i, j + 1) ≠ C (i, j + 2) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 5, ( C (i, j) ≠ C (i + 1, j) ∧ C (i + 1, j) ≠ C (i + 2, j) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 3, ( C (i, j) ≠ C (i + 1, j + 1) ∧ C (i + 1, j + 1) ≠ C (i + 2, j + 2) )

-- Define the minimum number of colors required
def min_colors_needed : Nat := 5

-- Prove the statement
theorem min_colors_needed_correct : ∃ C : Coloring (Fin min_colors_needed), valid_coloring (Fin min_colors_needed) C :=
sorry

end NUMINAMATH_GPT_min_colors_needed_correct_l2107_210776


namespace NUMINAMATH_GPT_rhombus_other_diagonal_length_l2107_210782

theorem rhombus_other_diagonal_length (area_square : ℝ) (side_length_square : ℝ) (d1_rhombus : ℝ) (d2_expected: ℝ) 
  (h1 : area_square = side_length_square^2) 
  (h2 : side_length_square = 8) 
  (h3 : d1_rhombus = 16) 
  (h4 : (d1_rhombus * d2_expected) / 2 = area_square) :
  d2_expected = 8 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_other_diagonal_length_l2107_210782


namespace NUMINAMATH_GPT_value_of_expression_l2107_210723

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) :
    11 / 7 + (2 * q - p) / (2 * q + p) = 2 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l2107_210723


namespace NUMINAMATH_GPT_value_range_of_quadratic_function_l2107_210709

def quadratic_function (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_quadratic_function :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → -1 < quadratic_function x ∧ quadratic_function x ≤ 3) :=
sorry

end NUMINAMATH_GPT_value_range_of_quadratic_function_l2107_210709


namespace NUMINAMATH_GPT_magnitude_of_z_l2107_210784

namespace ComplexNumberProof

open Complex

noncomputable def z (b : ℝ) : ℂ := (3 - b * Complex.I) / Complex.I

theorem magnitude_of_z (b : ℝ) (h : (z b).re = (z b).im) : Complex.abs (z b) = 3 * Real.sqrt 2 :=
by
  sorry

end ComplexNumberProof

end NUMINAMATH_GPT_magnitude_of_z_l2107_210784


namespace NUMINAMATH_GPT_lines_through_origin_l2107_210786

-- Define that a, b, c are in geometric progression
def geo_prog (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the property of the line passing through the common point (0, 0)
def passes_through_origin (a b c : ℝ) : Prop :=
  ∀ x y, (a * x + b * y = c) → (x = 0 ∧ y = 0)

theorem lines_through_origin (a b c : ℝ) (h : geo_prog a b c) : passes_through_origin a b c :=
by
  sorry

end NUMINAMATH_GPT_lines_through_origin_l2107_210786


namespace NUMINAMATH_GPT_gretchen_fewest_trips_l2107_210779

def fewestTrips (total_objects : ℕ) (max_carry : ℕ) : ℕ :=
  (total_objects + max_carry - 1) / max_carry

theorem gretchen_fewest_trips : fewestTrips 17 3 = 6 := 
  sorry

end NUMINAMATH_GPT_gretchen_fewest_trips_l2107_210779


namespace NUMINAMATH_GPT_find_fraction_value_l2107_210778

variable {x y : ℂ}

theorem find_fraction_value
    (h1 : (x^2 + y^2) / (x + y) = 4)
    (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
    (x^6 + y^6) / (x^5 + y^5) = 4 := by
  sorry

end NUMINAMATH_GPT_find_fraction_value_l2107_210778


namespace NUMINAMATH_GPT_problem_statement_l2107_210711

theorem problem_statement (a b : ℝ) (h : a^2 > b^2) : a > b → a > 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2107_210711


namespace NUMINAMATH_GPT_rhombus_diagonals_not_always_equal_l2107_210730

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end NUMINAMATH_GPT_rhombus_diagonals_not_always_equal_l2107_210730


namespace NUMINAMATH_GPT_max_value_of_f_l2107_210789

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 18 * x^2 + 27 * x

theorem max_value_of_f (x : ℝ) (h : 0 ≤ x) : ∃ M, M = 12 ∧ ∀ y, 0 ≤ y → f y ≤ M :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l2107_210789


namespace NUMINAMATH_GPT_largest_five_digit_product_l2107_210741

theorem largest_five_digit_product
  (digs : List ℕ)
  (h_digit_count : digs.length = 5)
  (h_product : (digs.foldr (· * ·) 1) = 9 * 8 * 7 * 6 * 5) :
  (digs.foldr (λ a b => if a > b then 10 * a + b else 10 * b + a) 0) = 98765 :=
sorry

end NUMINAMATH_GPT_largest_five_digit_product_l2107_210741


namespace NUMINAMATH_GPT_percentage_increase_of_y_over_x_l2107_210700

variable (x y : ℝ) (h : x > 0 ∧ y > 0) 

theorem percentage_increase_of_y_over_x
  (h_ratio : (x / 8) = (y / 7)) :
  ((y - x) / x) * 100 = 12.5 := 
sorry

end NUMINAMATH_GPT_percentage_increase_of_y_over_x_l2107_210700


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2107_210768

theorem quadratic_inequality_solution :
  {x : ℝ | 3 * x^2 + 5 * x < 8} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2107_210768


namespace NUMINAMATH_GPT_tractors_planting_rate_l2107_210714

theorem tractors_planting_rate (total_acres : ℕ) (total_days : ℕ) 
    (tractors_first_team : ℕ) (days_first_team : ℕ)
    (tractors_second_team : ℕ) (days_second_team : ℕ)
    (total_tractor_days : ℕ) :
    total_acres = 1700 →
    total_days = 5 →
    tractors_first_team = 2 →
    days_first_team = 2 →
    tractors_second_team = 7 →
    days_second_team = 3 →
    total_tractor_days = (tractors_first_team * days_first_team) + (tractors_second_team * days_second_team) →
    total_acres / total_tractor_days = 68 :=
by
  -- proof can be filled in later
  intros
  sorry

end NUMINAMATH_GPT_tractors_planting_rate_l2107_210714


namespace NUMINAMATH_GPT_number_of_boys_l2107_210794

-- Definitions of the conditions
def total_students : ℕ := 30
def ratio_girls_parts : ℕ := 1
def ratio_boys_parts : ℕ := 2
def total_parts : ℕ := ratio_girls_parts + ratio_boys_parts

-- Statement of the problem
theorem number_of_boys :
  ∃ (boys : ℕ), boys = (total_students / total_parts) * ratio_boys_parts ∧ boys = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l2107_210794


namespace NUMINAMATH_GPT_function_machine_output_is_17_l2107_210780

def functionMachineOutput (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 <= 22 then step1 + 10 else step1 - 7

theorem function_machine_output_is_17 : functionMachineOutput 8 = 17 := by
  sorry

end NUMINAMATH_GPT_function_machine_output_is_17_l2107_210780


namespace NUMINAMATH_GPT_triangle_side_a_l2107_210717

theorem triangle_side_a (c b : ℝ) (B : ℝ) (h₁ : c = 2) (h₂ : b = 6) (h₃ : B = 120) : a = 2 :=
by sorry

end NUMINAMATH_GPT_triangle_side_a_l2107_210717


namespace NUMINAMATH_GPT_rubies_in_treasure_l2107_210750

theorem rubies_in_treasure (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) : 
  total_gems - diamonds = 5110 := by
  sorry

end NUMINAMATH_GPT_rubies_in_treasure_l2107_210750


namespace NUMINAMATH_GPT_constant_term_in_modified_equation_l2107_210727

theorem constant_term_in_modified_equation :
  ∃ (c : ℝ), ∀ (q : ℝ), (3 * (3 * 5 - 3) - 3 + c = 132) → c = 99 := 
by
  sorry

end NUMINAMATH_GPT_constant_term_in_modified_equation_l2107_210727


namespace NUMINAMATH_GPT_advertising_time_l2107_210775

-- Define the conditions
def total_duration : ℕ := 30
def national_news : ℕ := 12
def international_news : ℕ := 5
def sports : ℕ := 5
def weather_forecasts : ℕ := 2

-- Calculate total content time
def total_content_time : ℕ := national_news + international_news + sports + weather_forecasts

-- Define the proof problem
theorem advertising_time (h : total_duration - total_content_time = 6) : (total_duration - total_content_time) = 6 :=
by
sorry

end NUMINAMATH_GPT_advertising_time_l2107_210775


namespace NUMINAMATH_GPT_find_unknown_number_l2107_210763

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l2107_210763


namespace NUMINAMATH_GPT_profit_15000_l2107_210793

theorem profit_15000
  (P : ℝ)
  (invest_mary : ℝ := 550)
  (invest_mike : ℝ := 450)
  (total_invest := invest_mary + invest_mike)
  (share_ratio_mary := invest_mary / total_invest)
  (share_ratio_mike := invest_mike / total_invest)
  (effort_share := P / 6)
  (invest_share_mary := share_ratio_mary * (2 * P / 3))
  (invest_share_mike := share_ratio_mike * (2 * P / 3))
  (mary_total := effort_share + invest_share_mary)
  (mike_total := effort_share + invest_share_mike)
  (condition : mary_total - mike_total = 1000) :
  P = 15000 :=  
sorry

end NUMINAMATH_GPT_profit_15000_l2107_210793


namespace NUMINAMATH_GPT_number_of_people_and_price_l2107_210762

theorem number_of_people_and_price 
  (x y : ℤ) 
  (h1 : 8 * x - y = 3) 
  (h2 : y - 7 * x = 4) : 
  x = 7 ∧ y = 53 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_and_price_l2107_210762


namespace NUMINAMATH_GPT_solve_system_l2107_210751

theorem solve_system (x y z : ℤ) 
  (h1 : x + 3 * y = 20)
  (h2 : x + y + z = 25)
  (h3 : x - z = 5) : 
  x = 14 ∧ y = 2 ∧ z = 9 := 
  sorry

end NUMINAMATH_GPT_solve_system_l2107_210751


namespace NUMINAMATH_GPT_solve_y_l2107_210748

theorem solve_y (y : ℝ) (h1 : y > 0) (h2 : (y - 6) / 16 = 6 / (y - 16)) : y = 22 :=
by
  sorry

end NUMINAMATH_GPT_solve_y_l2107_210748


namespace NUMINAMATH_GPT_odds_against_C_l2107_210788

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C (pA pB pC : ℚ) (hA : pA = 1 / 3) (hB : pB = 1 / 5) (hC : pC = 7 / 15) :
  odds_against_winning pC = 8 / 7 :=
by
  -- Definitions based on the conditions provided in a)
  have h1 : odds_against_winning (1/3) = 2 := by sorry
  have h2 : odds_against_winning (1/5) = 4 := by sorry

  -- Odds against C
  have h3 : 1 - (pA + pB) = pC := by sorry
  have h4 : pA + pB = 8 / 15 := by sorry

  -- Show that odds against C winning is 8/7
  have h5 : odds_against_winning pC = 8 / 7 := by sorry
  exact h5

end NUMINAMATH_GPT_odds_against_C_l2107_210788


namespace NUMINAMATH_GPT_investment_time_ratio_l2107_210781

theorem investment_time_ratio (x t : ℕ) (h_inv : 7 * x = t * 5) (h_prof_ratio : 7 / 10 = 70 / (5 * t)) : 
  t = 20 := sorry

end NUMINAMATH_GPT_investment_time_ratio_l2107_210781


namespace NUMINAMATH_GPT_parcels_division_l2107_210734

theorem parcels_division (x y n : ℕ) (h : 5 + 2 * x + 3 * y = 4 * n) (hn : n = x + y) :
    n = 3 ∨ n = 4 ∨ n = 5 := 
sorry

end NUMINAMATH_GPT_parcels_division_l2107_210734


namespace NUMINAMATH_GPT_tetrahedron_formable_l2107_210703

theorem tetrahedron_formable (x : ℝ) (hx_pos : 0 < x) (hx_bound : x < (Real.sqrt 6 + Real.sqrt 2) / 2) :
  true := 
sorry

end NUMINAMATH_GPT_tetrahedron_formable_l2107_210703


namespace NUMINAMATH_GPT_multiple_of_9_l2107_210728

theorem multiple_of_9 (x : ℕ) (hx1 : ∃ k : ℕ, x = 9 * k) (hx2 : x^2 > 80) (hx3 : x < 30) : x = 9 ∨ x = 18 ∨ x = 27 :=
sorry

end NUMINAMATH_GPT_multiple_of_9_l2107_210728


namespace NUMINAMATH_GPT_radical_multiplication_l2107_210706

noncomputable def root4 (x : ℝ) : ℝ := x ^ (1/4)
noncomputable def root3 (x : ℝ) : ℝ := x ^ (1/3)
noncomputable def root2 (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_multiplication : root4 256 * root3 8 * root2 16 = 32 := by
  sorry

end NUMINAMATH_GPT_radical_multiplication_l2107_210706


namespace NUMINAMATH_GPT_equivalent_problem_l2107_210720

theorem equivalent_problem (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n < 29) (h₃ : 2 * n % 29 = 1) :
  (3^n % 29)^3 - 3 % 29 = 3 :=
sorry

end NUMINAMATH_GPT_equivalent_problem_l2107_210720


namespace NUMINAMATH_GPT_train_is_late_l2107_210785

theorem train_is_late (S : ℝ) (T : ℝ) (T' : ℝ) (h1 : T = 2) (h2 : T' = T * 5 / 4) :
  (T' - T) * 60 = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_is_late_l2107_210785


namespace NUMINAMATH_GPT_total_volume_of_five_cubes_l2107_210771

-- Definition for volume of a cube function
def volume_of_cube (edge_length : ℝ) : ℝ :=
  edge_length ^ 3

-- Conditions
def edge_length : ℝ := 5
def number_of_cubes : ℝ := 5

-- Proof statement
theorem total_volume_of_five_cubes : 
  volume_of_cube edge_length * number_of_cubes = 625 := 
by
  sorry

end NUMINAMATH_GPT_total_volume_of_five_cubes_l2107_210771


namespace NUMINAMATH_GPT_option_B_correct_option_C_correct_l2107_210735

-- Define the permutation coefficient
def A (n m : ℕ) : ℕ := n * (n-1) * (n-2) * (n-m+1)

-- Prove the equation for option B
theorem option_B_correct (n m : ℕ) : A (n+1) (m+1) - A n m = n^2 * A (n-1) (m-1) :=
by
  sorry

-- Prove the equation for option C
theorem option_C_correct (n m : ℕ) : A n m = n * A (n-1) (m-1) :=
by
  sorry

end NUMINAMATH_GPT_option_B_correct_option_C_correct_l2107_210735


namespace NUMINAMATH_GPT_largest_amount_received_back_l2107_210765

theorem largest_amount_received_back 
  (x y x_lost y_lost : ℕ) 
  (h1 : 20 * x + 100 * y = 3000) 
  (h2 : x_lost + y_lost = 16) 
  (h3 : x_lost = y_lost + 2 ∨ x_lost = y_lost - 2) 
  : (3000 - (20 * x_lost + 100 * y_lost) = 2120) :=
sorry

end NUMINAMATH_GPT_largest_amount_received_back_l2107_210765


namespace NUMINAMATH_GPT_total_stamps_l2107_210738

-- Definitions based on conditions
def kylies_stamps : ℕ := 34
def nellys_stamps : ℕ := kylies_stamps + 44

-- Statement of the proof problem
theorem total_stamps : kylies_stamps + nellys_stamps = 112 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_stamps_l2107_210738


namespace NUMINAMATH_GPT_correct_option_C_l2107_210707

variable {a : ℝ} (x : ℝ) (b : ℝ)

theorem correct_option_C : 
  (a^8 / a^2 = a^6) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_option_C_l2107_210707


namespace NUMINAMATH_GPT_find_value_of_x_squared_plus_y_squared_l2107_210777

theorem find_value_of_x_squared_plus_y_squared (x y : ℝ) (h : (x^2 + y^2 + 1)^2 - 4 = 0) : x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_x_squared_plus_y_squared_l2107_210777


namespace NUMINAMATH_GPT_inscribed_circle_radius_l2107_210729

noncomputable def radius_inscribed_circle (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ) :=
  if (r1 = 2 ∧ r2 = 6) ∧ ((O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) then
    2 * (Real.sqrt 3 - 1)
  else
    0

theorem inscribed_circle_radius (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ)
  (h1 : r1 = 2) (h2 : r2 = 6)
  (h3 : (O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) :
  radius_inscribed_circle O1 O2 D r1 r2 = 2 * (Real.sqrt 3 - 1) :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l2107_210729


namespace NUMINAMATH_GPT_all_elements_rational_l2107_210797

open Set

def finite_set_in_interval (n : ℕ) : Set ℝ :=
  {x | ∃ i, i ∈ Finset.range (n + 1) ∧ (x = 0 ∨ x = 1 ∨ 0 < x ∧ x < 1)}

def unique_distance_condition (S : Set ℝ) : Prop :=
  ∀ d, d ≠ 1 → ∃ x_i x_j x_k x_l, x_i ∈ S ∧ x_j ∈ S ∧ x_k ∈ S ∧ x_l ∈ S ∧ 
        abs (x_i - x_j) = d ∧ abs (x_k - x_l) = d ∧ (x_i = x_k → x_j ≠ x_l)

theorem all_elements_rational
  (n : ℕ)
  (S : Set ℝ)
  (hS1 : ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1)
  (hS2 : 0 ∈ S)
  (hS3 : 1 ∈ S)
  (hS4 : unique_distance_condition S) :
  ∀ x ∈ S, ∃ q : ℚ, (x : ℝ) = q := 
sorry

end NUMINAMATH_GPT_all_elements_rational_l2107_210797


namespace NUMINAMATH_GPT_total_number_of_animals_is_650_l2107_210760

def snake_count : Nat := 100
def arctic_fox_count : Nat := 80
def leopard_count : Nat := 20
def bee_eater_count : Nat := 10 * leopard_count
def cheetah_count : Nat := snake_count / 2
def alligator_count : Nat := 2 * (arctic_fox_count + leopard_count)

def total_animal_count : Nat :=
  snake_count + arctic_fox_count + leopard_count + bee_eater_count + cheetah_count + alligator_count

theorem total_number_of_animals_is_650 :
  total_animal_count = 650 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_animals_is_650_l2107_210760


namespace NUMINAMATH_GPT_parametric_equations_l2107_210722

variables (t : ℝ)
def x_velocity : ℝ := 9
def y_velocity : ℝ := 12
def init_x : ℝ := 1
def init_y : ℝ := 1

theorem parametric_equations :
  (x = init_x + x_velocity * t) ∧ (y = init_y + y_velocity * t) :=
sorry

end NUMINAMATH_GPT_parametric_equations_l2107_210722


namespace NUMINAMATH_GPT_number_of_houses_built_l2107_210799

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558
def houses_built : ℕ := current_houses - original_houses

theorem number_of_houses_built :
  houses_built = 97741 := by
  sorry

end NUMINAMATH_GPT_number_of_houses_built_l2107_210799
