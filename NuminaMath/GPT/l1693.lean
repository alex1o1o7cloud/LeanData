import Mathlib

namespace NUMINAMATH_GPT_tan_sub_pi_over_4_l1693_169360

variables (α : ℝ)
axiom tan_alpha : Real.tan α = 1 / 6

theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = -5 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_sub_pi_over_4_l1693_169360


namespace NUMINAMATH_GPT_general_term_of_sequence_l1693_169304

theorem general_term_of_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, a (n + 1) = (n^2 * (a n)^2 + 5) / ((n^2 - 1) * a (n - 1))) :
  ∀ n : ℕ, a n = 
    if n = 0 then 0 else
    (1 / n) * ( (63 - 13 * Real.sqrt 21) / 42 * ((5 + Real.sqrt 21) / 2) ^ n + 
                (63 + 13 * Real.sqrt 21) / 42 * ((5 - Real.sqrt 21) / 2) ^ n) :=
by
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1693_169304


namespace NUMINAMATH_GPT_james_puzzle_completion_time_l1693_169311

theorem james_puzzle_completion_time :
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10
  total_minutes = 400 :=
by
  -- Definitions based on conditions
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10

  -- Using sorry to skip proof
  sorry

end NUMINAMATH_GPT_james_puzzle_completion_time_l1693_169311


namespace NUMINAMATH_GPT_burglary_charge_sentence_l1693_169337

theorem burglary_charge_sentence (B : ℕ) 
  (arson_counts : ℕ := 3) 
  (arson_sentence : ℕ := 36)
  (burglary_charges : ℕ := 2)
  (petty_larceny_factor : ℕ := 6)
  (total_jail_time : ℕ := 216) :
  arson_counts * arson_sentence + burglary_charges * B + (burglary_charges * petty_larceny_factor) * (B / 3) = total_jail_time → B = 18 := 
by
  sorry

end NUMINAMATH_GPT_burglary_charge_sentence_l1693_169337


namespace NUMINAMATH_GPT_divisor_of_product_of_four_consecutive_integers_l1693_169354

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end NUMINAMATH_GPT_divisor_of_product_of_four_consecutive_integers_l1693_169354


namespace NUMINAMATH_GPT_bobbit_worm_fish_count_l1693_169350

theorem bobbit_worm_fish_count 
  (initial_fish : ℕ)
  (fish_eaten_per_day : ℕ)
  (days_before_adding_fish : ℕ)
  (additional_fish : ℕ)
  (days_after_adding_fish : ℕ) :
  days_before_adding_fish = 14 →
  days_after_adding_fish = 7 →
  fish_eaten_per_day = 2 →
  initial_fish = 60 →
  additional_fish = 8 →
  (initial_fish - days_before_adding_fish * fish_eaten_per_day + additional_fish - days_after_adding_fish * fish_eaten_per_day) = 26 :=
by
  intros 
  -- sorry proof goes here
  sorry

end NUMINAMATH_GPT_bobbit_worm_fish_count_l1693_169350


namespace NUMINAMATH_GPT_gcd_840_1764_l1693_169366

-- Define the numbers according to the conditions
def a : ℕ := 1764
def b : ℕ := 840

-- The goal is to prove that the GCD of a and b is 84
theorem gcd_840_1764 : Nat.gcd a b = 84 := 
by
  -- The proof steps would normally go here
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l1693_169366


namespace NUMINAMATH_GPT_plus_one_eq_next_plus_l1693_169356

theorem plus_one_eq_next_plus (m : ℕ) (h : m > 1) : (m^2 + m) + 1 = ((m + 1)^2 + (m + 1)) := by
  sorry

end NUMINAMATH_GPT_plus_one_eq_next_plus_l1693_169356


namespace NUMINAMATH_GPT_PP1_length_l1693_169359

open Real

theorem PP1_length (AB AC : ℝ) (h₁ : AB = 5) (h₂ : AC = 3)
  (h₃ : ∃ γ : ℝ, γ = 90)  -- a right angle at A
  (BC : ℝ) (h₄ : BC = sqrt (AB^2 - AC^2))
  (A1B : ℝ) (A1C : ℝ) (h₅ : BC = A1B + A1C)
  (h₆ : A1B / A1C = AB / AC)
  (PQ : ℝ) (h₇ : PQ = A1B)
  (PR : ℝ) (h₈ : PR = A1C)
  (PP1 : ℝ) :
  PP1 = (3 * sqrt 5) / 4 :=
sorry

end NUMINAMATH_GPT_PP1_length_l1693_169359


namespace NUMINAMATH_GPT_problem_condition_l1693_169395

variable {f : ℝ → ℝ}

theorem problem_condition (h_diff : Differentiable ℝ f) (h_ineq : ∀ x : ℝ, f x < iteratedDeriv 2 f x) : 
  e^2019 * f (-2019) < f 0 ∧ f 2019 > e^2019 * f 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_condition_l1693_169395


namespace NUMINAMATH_GPT_kris_age_l1693_169364

theorem kris_age (kris_age herbert_age : ℕ) (h1 : herbert_age + 1 = 15) (h2 : herbert_age + 10 = kris_age) : kris_age = 24 :=
by
  sorry

end NUMINAMATH_GPT_kris_age_l1693_169364


namespace NUMINAMATH_GPT_number_of_cases_in_top_level_l1693_169399

-- Definitions for the total number of soda cases
def pyramid_cases (n : ℕ) : ℕ :=
  n^2 + (n + 1)^2 + (n + 2)^2 + (n + 3)^2

-- Theorem statement: proving the number of cases in the top level
theorem number_of_cases_in_top_level (n : ℕ) (h : pyramid_cases n = 30) : n = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_cases_in_top_level_l1693_169399


namespace NUMINAMATH_GPT_find_x1_l1693_169332

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1-x1)^2 + 2*(x1-x2)^2 + (x2-x3)^2 + x3^2 = 1/2) :
  x1 = (3*Real.sqrt 2 - 3)/7 :=
by
  sorry

end NUMINAMATH_GPT_find_x1_l1693_169332


namespace NUMINAMATH_GPT_find_number_l1693_169340

theorem find_number {x : ℝ} (h : (1/3) * x = 130.00000000000003) : x = 390 := 
sorry

end NUMINAMATH_GPT_find_number_l1693_169340


namespace NUMINAMATH_GPT_sam_bought_17_mystery_books_l1693_169353

def adventure_books := 13
def used_books := 15
def new_books := 15
def total_books := used_books + new_books
def mystery_books := total_books - adventure_books

theorem sam_bought_17_mystery_books : mystery_books = 17 := by
  sorry

end NUMINAMATH_GPT_sam_bought_17_mystery_books_l1693_169353


namespace NUMINAMATH_GPT_find_certain_amount_l1693_169382

theorem find_certain_amount :
  ∀ (A : ℝ), (160 * 8 * 12.5 / 100 = A * 8 * 4 / 100) → 
            (A = 500) :=
  by
    intros A h
    sorry

end NUMINAMATH_GPT_find_certain_amount_l1693_169382


namespace NUMINAMATH_GPT_average_of_D_E_F_l1693_169363

theorem average_of_D_E_F (D E F : ℝ) 
  (h1 : 2003 * F - 4006 * D = 8012) 
  (h2 : 2003 * E + 6009 * D = 10010) : 
  (D + E + F) / 3 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_average_of_D_E_F_l1693_169363


namespace NUMINAMATH_GPT_eq_iff_solution_l1693_169338

theorem eq_iff_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^y + y^x = x^x + y^y ↔ x = y :=
by sorry

end NUMINAMATH_GPT_eq_iff_solution_l1693_169338


namespace NUMINAMATH_GPT_evaluate_fraction_l1693_169361

theorem evaluate_fraction (a b : ℕ) (h₁ : a = 250) (h₂ : b = 240) :
  1800^2 / (a^2 - b^2) = 660 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l1693_169361


namespace NUMINAMATH_GPT_interest_rate_for_4000_investment_l1693_169385

theorem interest_rate_for_4000_investment
      (total_money : ℝ := 9000)
      (invested_at_9_percent : ℝ := 5000)
      (total_interest : ℝ := 770)
      (invested_at_unknown_rate : ℝ := 4000) :
  ∃ r : ℝ, invested_at_unknown_rate * r = total_interest - (invested_at_9_percent * 0.09) ∧ r = 0.08 :=
by {
  -- Proof is not required based on instruction, so we use sorry.
  sorry
}

end NUMINAMATH_GPT_interest_rate_for_4000_investment_l1693_169385


namespace NUMINAMATH_GPT_P_plus_Q_l1693_169325

theorem P_plus_Q (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 4 → (P / (x - 4) + Q * (x + 2) = (-4 * x^2 + 16 * x + 30) / (x - 4))) : P + Q = 42 :=
sorry

end NUMINAMATH_GPT_P_plus_Q_l1693_169325


namespace NUMINAMATH_GPT_product_of_two_numbers_l1693_169335

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : x - y = 16) : 
  x * y = 836 := 
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1693_169335


namespace NUMINAMATH_GPT_max_pancake_pieces_3_cuts_l1693_169324

open Nat

def P : ℕ → ℕ
| 0 => 1
| n => n * (n + 1) / 2 + 1

theorem max_pancake_pieces_3_cuts : P 3 = 7 := by
  have h0: P 0 = 1 := by rfl
  have h1: P 1 = 2 := by rfl
  have h2: P 2 = 4 := by rfl
  show P 3 = 7
  calc
    P 3 = 3 * (3 + 1) / 2 + 1 := by rfl
    _ = 3 * 4 / 2 + 1 := by rfl
    _ = 6 + 1 := by norm_num
    _ = 7 := by norm_num

end NUMINAMATH_GPT_max_pancake_pieces_3_cuts_l1693_169324


namespace NUMINAMATH_GPT_original_proposition_true_converse_proposition_false_l1693_169369

theorem original_proposition_true (a b : ℝ) : 
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := 
sorry

theorem converse_proposition_false : 
  ¬ (∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end NUMINAMATH_GPT_original_proposition_true_converse_proposition_false_l1693_169369


namespace NUMINAMATH_GPT_expected_value_range_of_p_l1693_169308

theorem expected_value_range_of_p (p : ℝ) (X : ℕ → ℝ) :
  (∀ n, (n = 1 → X n = p) ∧ 
        (n = 2 → X n = p * (1 - p)) ∧ 
        (n = 3 → X n = (1 - p) ^ 2)) →
  (p^2 - 3 * p + 3 > 1.75) → 
  0 < p ∧ p < 0.5 := by
  intros hprob hexp
  -- Proof would be filled in here
  sorry

end NUMINAMATH_GPT_expected_value_range_of_p_l1693_169308


namespace NUMINAMATH_GPT_probability_of_event_B_given_A_l1693_169368

-- Definition of events and probability
noncomputable def prob_event_B_given_A : ℝ :=
  let total_outcomes := 36
  let outcomes_A := 30
  let outcomes_B_given_A := 10
  outcomes_B_given_A / outcomes_A

-- Theorem statement
theorem probability_of_event_B_given_A : prob_event_B_given_A = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_of_event_B_given_A_l1693_169368


namespace NUMINAMATH_GPT_multiplication_of_negative_and_positive_l1693_169352

theorem multiplication_of_negative_and_positive :
  (-3) * 5 = -15 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_of_negative_and_positive_l1693_169352


namespace NUMINAMATH_GPT_square_feet_per_acre_l1693_169387

-- Define the conditions
def rent_per_acre_per_month : ℝ := 60
def total_rent_per_month : ℝ := 600
def length_of_plot : ℝ := 360
def width_of_plot : ℝ := 1210

-- Translate the problem to a Lean theorem
theorem square_feet_per_acre :
  (length_of_plot * width_of_plot) / (total_rent_per_month / rent_per_acre_per_month) = 43560 :=
by {
  -- skipping the proof steps
  sorry
}

end NUMINAMATH_GPT_square_feet_per_acre_l1693_169387


namespace NUMINAMATH_GPT_value_range_f_at_4_l1693_169329

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_range_f_at_4 (f : ℝ → ℝ)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 1 ≤ f (1) ∧ f (1) ≤ 3)
  (h3 : 2 ≤ f (2) ∧ f (2) ≤ 4)
  (h4 : -1 ≤ f (3) ∧ f (3) ≤ 1) :
  -21.75 ≤ f 4 ∧ f 4 ≤ 1 :=
sorry

end NUMINAMATH_GPT_value_range_f_at_4_l1693_169329


namespace NUMINAMATH_GPT_division_remainder_l1693_169327

theorem division_remainder (n : ℕ) (h : n = 8 * 8 + 0) : n % 5 = 4 := by
  sorry

end NUMINAMATH_GPT_division_remainder_l1693_169327


namespace NUMINAMATH_GPT_john_max_questions_correct_l1693_169309

variable (c w b : ℕ)

theorem john_max_questions_correct (H1 : c + w + b = 20) (H2 : 5 * c - 2 * w = 48) : c ≤ 12 := sorry

end NUMINAMATH_GPT_john_max_questions_correct_l1693_169309


namespace NUMINAMATH_GPT_solve_pos_int_a_l1693_169336

theorem solve_pos_int_a :
  ∀ a : ℕ, (0 < a) →
  (∀ n : ℕ, (n ≥ 5) → ((2^n - n^2) ∣ (a^n - n^a))) →
  (a = 2 ∨ a = 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_pos_int_a_l1693_169336


namespace NUMINAMATH_GPT_math_problem_l1693_169358
-- Import the entire mathlib library for necessary mathematical definitions and notations

-- Define the conditions and the statement to prove
theorem math_problem (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 :=
by 
  -- place a sorry as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_math_problem_l1693_169358


namespace NUMINAMATH_GPT_max_s_value_l1693_169305

theorem max_s_value (p q r s : ℝ) (h1 : p + q + r + s = 10) (h2 : (p * q) + (p * r) + (p * s) + (q * r) + (q * s) + (r * s) = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end NUMINAMATH_GPT_max_s_value_l1693_169305


namespace NUMINAMATH_GPT_crickets_total_l1693_169347

noncomputable def initial_amount : ℝ := 7.5
noncomputable def additional_amount : ℝ := 11.25
noncomputable def total_amount : ℝ := 18.75

theorem crickets_total : initial_amount + additional_amount = total_amount :=
by
  sorry

end NUMINAMATH_GPT_crickets_total_l1693_169347


namespace NUMINAMATH_GPT_quadratic_root_condition_l1693_169390

theorem quadratic_root_condition (m n : ℝ) (h : m * (-1)^2 - n * (-1) - 2023 = 0) :
  m + n = 2023 :=
sorry

end NUMINAMATH_GPT_quadratic_root_condition_l1693_169390


namespace NUMINAMATH_GPT_isosceles_right_triangle_contains_probability_l1693_169386

noncomputable def isosceles_right_triangle_probability : ℝ :=
  let leg_length := 2
  let triangle_area := (leg_length * leg_length) / 2
  let distance_radius := 1
  let quarter_circle_area := (Real.pi * (distance_radius * distance_radius)) / 4
  quarter_circle_area / triangle_area

theorem isosceles_right_triangle_contains_probability :
  isosceles_right_triangle_probability = (Real.pi / 8) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_contains_probability_l1693_169386


namespace NUMINAMATH_GPT_find_intersection_complement_find_value_m_l1693_169373

-- (1) Problem Statement
theorem find_intersection_complement (A : Set ℝ) (B : Set ℝ) (x : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - 3 < 0}) →
  (x ∈ A ∩ (Bᶜ : Set ℝ)) ↔ (x = -1 ∨ 3 ≤ x ∧ x ≤ 5) :=
by
  sorry

-- (2) Problem Statement
theorem find_value_m (A B : Set ℝ) (m : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - m < 0}) →
  (A ∩ B = {x | -1 ≤ x ∧ x < 4}) →
  m = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_complement_find_value_m_l1693_169373


namespace NUMINAMATH_GPT_arithmetic_sequence_a13_l1693_169398

variable (a1 d : ℤ)

theorem arithmetic_sequence_a13 (h : a1 + 2 * d + a1 + 8 * d + a1 + 26 * d = 12) : a1 + 12 * d = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a13_l1693_169398


namespace NUMINAMATH_GPT_find_a_l1693_169365

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 5) (h3 : c = 3) : a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l1693_169365


namespace NUMINAMATH_GPT_ratio_result_l1693_169316

theorem ratio_result (p q r s : ℚ) 
(h1 : p / q = 2) 
(h2 : q / r = 4 / 5) 
(h3 : r / s = 3) : 
  s / p = 5 / 24 :=
sorry

end NUMINAMATH_GPT_ratio_result_l1693_169316


namespace NUMINAMATH_GPT_range_of_a_l1693_169372

noncomputable def p (x: ℝ) : Prop := |4 * x - 1| ≤ 1
noncomputable def q (x a: ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a: ℝ) :
  (¬ (∀ x, p x) → (¬ (∀ x, q x a))) ∧ (¬ (¬ (∀ x, p x) → (¬ (∀ x, q x a))))
  ↔ (-1 / 2 ≤ a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1693_169372


namespace NUMINAMATH_GPT_find_other_number_l1693_169396

def a : ℝ := 0.5
def d : ℝ := 0.16666666666666669
def b : ℝ := 0.3333333333333333

theorem find_other_number : a - d = b := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1693_169396


namespace NUMINAMATH_GPT_seats_per_bus_l1693_169328

-- Conditions
def total_students : ℕ := 180
def total_buses : ℕ := 3

-- Theorem Statement
theorem seats_per_bus : (total_students / total_buses) = 60 := 
by 
  sorry

end NUMINAMATH_GPT_seats_per_bus_l1693_169328


namespace NUMINAMATH_GPT_transformed_quadratic_equation_l1693_169334

theorem transformed_quadratic_equation (u v: ℝ) :
  (u + v = -5 / 2) ∧ (u * v = 3 / 2) ↔ (∃ y : ℝ, y^2 - y + 6 = 0) := sorry

end NUMINAMATH_GPT_transformed_quadratic_equation_l1693_169334


namespace NUMINAMATH_GPT_product_of_roots_proof_l1693_169313

noncomputable def product_of_roots : ℚ :=
  let leading_coeff_poly1 := 3
  let leading_coeff_poly2 := 4
  let constant_term_poly1 := -15
  let constant_term_poly2 := 9
  let a := leading_coeff_poly1 * leading_coeff_poly2
  let b := constant_term_poly1 * constant_term_poly2
  (b : ℚ) / a

theorem product_of_roots_proof :
  product_of_roots = -45/4 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_proof_l1693_169313


namespace NUMINAMATH_GPT_f_11_f_2021_eq_neg_one_l1693_169388

def f (n : ℕ) : ℚ := sorry

axiom recurrence_relation (n : ℕ) : f (n + 3) = (f n - 1) / (f n + 1)
axiom f1_ne_zero : f 1 ≠ 0
axiom f1_ne_one : f 1 ≠ 1
axiom f1_ne_neg_one : f 1 ≠ -1

theorem f_11_f_2021_eq_neg_one : f 11 * f 2021 = -1 := 
by
  sorry

end NUMINAMATH_GPT_f_11_f_2021_eq_neg_one_l1693_169388


namespace NUMINAMATH_GPT_two_beta_plus_alpha_eq_pi_div_two_l1693_169323

theorem two_beta_plus_alpha_eq_pi_div_two
  (α β : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (hβ1 : 0 < β) (hβ2 : β < π / 2)
  (h : Real.tan α + Real.tan β = 1 / Real.cos α) :
  2 * β + α = π / 2 :=
sorry

end NUMINAMATH_GPT_two_beta_plus_alpha_eq_pi_div_two_l1693_169323


namespace NUMINAMATH_GPT_pairs_satisfaction_l1693_169346

-- Definitions for the conditions given
def condition1 (x y : ℝ) : Prop := y = (x + 2)^2
def condition2 (x y : ℝ) : Prop := x * y + 2 * y = 2

-- The statement that we need to prove
theorem pairs_satisfaction : 
  (∃ x y : ℝ, condition1 x y ∧ condition2 x y) ∧ 
  (∃ x1 x2 : ℂ, x^2 + -2*x + 1 = 0 ∧ ¬∃ (y : ℝ), y = (x1 + 2)^2 ∨ y = (x2 + 2)^2) :=
by
  sorry

end NUMINAMATH_GPT_pairs_satisfaction_l1693_169346


namespace NUMINAMATH_GPT_M_intersect_N_l1693_169310

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def intersection (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∈ N}

theorem M_intersect_N :
  intersection M N = {x | 1 ≤ x ∧ x < 2} := 
sorry

end NUMINAMATH_GPT_M_intersect_N_l1693_169310


namespace NUMINAMATH_GPT_pure_gala_trees_l1693_169370

variable (T F G : ℝ)

theorem pure_gala_trees (h1 : F + 0.1 * T = 170) (h2 : F = 0.75 * T): G = T - F -> G = 50 :=
by
  sorry

end NUMINAMATH_GPT_pure_gala_trees_l1693_169370


namespace NUMINAMATH_GPT_fraction_pizza_covered_by_pepperoni_l1693_169300

theorem fraction_pizza_covered_by_pepperoni :
  (∀ (r_pizz : ℝ) (n_pepp : ℕ) (d_pepp : ℝ),
      r_pizz = 8 ∧ n_pepp = 32 ∧ d_pepp = 2 →
      (n_pepp * π * (d_pepp / 2)^2) / (π * r_pizz^2) = 1 / 2) :=
sorry

end NUMINAMATH_GPT_fraction_pizza_covered_by_pepperoni_l1693_169300


namespace NUMINAMATH_GPT_amy_lily_tie_l1693_169355

noncomputable def tie_probability : ℚ :=
    let amy_win := (2 / 5 : ℚ)
    let lily_win := (1 / 4 : ℚ)
    let total_win := amy_win + lily_win
    1 - total_win

theorem amy_lily_tie (h1 : (2 / 5 : ℚ) = 2 / 5) 
                     (h2 : (1 / 4 : ℚ) = 1 / 4)
                     (h3 : (2 / 5 : ℚ) ≥ 2 * (1 / 4 : ℚ) ∨ (1 / 4 : ℚ) ≥ 2 * (2 / 5 : ℚ)) :
    tie_probability = 7 / 20 :=
by
  sorry

end NUMINAMATH_GPT_amy_lily_tie_l1693_169355


namespace NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l1693_169374

theorem arithmetic_sequence_seventh_term
  (a d : ℝ)
  (h_sum : 4 * a + 6 * d = 20)
  (h_fifth : a + 4 * d = 8) :
  a + 6 * d = 10.4 :=
by
  sorry -- proof to be provided

end NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l1693_169374


namespace NUMINAMATH_GPT_triple_sum_of_45_point_2_and_one_fourth_l1693_169303

theorem triple_sum_of_45_point_2_and_one_fourth : 
  (3 * (45.2 + 0.25)) = 136.35 :=
by
  sorry

end NUMINAMATH_GPT_triple_sum_of_45_point_2_and_one_fourth_l1693_169303


namespace NUMINAMATH_GPT_lucky_numbers_count_l1693_169342

def isLuckyNumber (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3 = 6) && (100 ≤ n) && (n < 1000)

def countLuckyNumbers : ℕ :=
  (List.range' 100 900).filter isLuckyNumber |>.length

theorem lucky_numbers_count : countLuckyNumbers = 21 := 
  sorry

end NUMINAMATH_GPT_lucky_numbers_count_l1693_169342


namespace NUMINAMATH_GPT_hair_cut_first_day_l1693_169330

theorem hair_cut_first_day 
  (total_hair_cut : ℝ) 
  (hair_cut_second_day : ℝ) 
  (h_total : total_hair_cut = 0.875) 
  (h_second : hair_cut_second_day = 0.5) : 
  total_hair_cut - hair_cut_second_day = 0.375 := 
  by
  simp [h_total, h_second]
  sorry

end NUMINAMATH_GPT_hair_cut_first_day_l1693_169330


namespace NUMINAMATH_GPT_rhombus_side_length_l1693_169341

-- Define the rhombus properties and the problem conditions
variables (p q x : ℝ)

-- State the problem as a theorem in Lean 4
theorem rhombus_side_length (h : x^2 = p * q) : x = Real.sqrt (p * q) :=
sorry

end NUMINAMATH_GPT_rhombus_side_length_l1693_169341


namespace NUMINAMATH_GPT_combination_permutation_value_l1693_169314

theorem combination_permutation_value (n : ℕ) (h : (n * (n - 1)) = 42) : (Nat.factorial n) / (Nat.factorial 3 * Nat.factorial (n - 3)) = 35 := 
by
  sorry

end NUMINAMATH_GPT_combination_permutation_value_l1693_169314


namespace NUMINAMATH_GPT_algorithm_characteristics_l1693_169319

theorem algorithm_characteristics (finiteness : Prop) (definiteness : Prop) (output_capability : Prop) (unique : Prop) 
  (h1 : finiteness = true) 
  (h2 : definiteness = true) 
  (h3 : output_capability = true) 
  (h4 : unique = false) : 
  incorrect_statement = unique := 
by
  sorry

end NUMINAMATH_GPT_algorithm_characteristics_l1693_169319


namespace NUMINAMATH_GPT_total_books_sum_l1693_169378

-- Given conditions
def Joan_books := 10
def Tom_books := 38
def Lisa_books := 27
def Steve_books := 45
def Kim_books := 14
def Alex_books := 48

-- Define the total number of books
def total_books := Joan_books + Tom_books + Lisa_books + Steve_books + Kim_books + Alex_books

-- Proof statement
theorem total_books_sum : total_books = 182 := by
  sorry

end NUMINAMATH_GPT_total_books_sum_l1693_169378


namespace NUMINAMATH_GPT_sequence_periodic_l1693_169318

theorem sequence_periodic (a : ℕ → ℝ) (h1 : a 1 = 0) (h2 : ∀ n, a n + a (n + 1) = 2) : a 2011 = 0 := by
  sorry

end NUMINAMATH_GPT_sequence_periodic_l1693_169318


namespace NUMINAMATH_GPT_find_general_term_l1693_169344

theorem find_general_term (S a : ℕ → ℤ) (n : ℕ) (h_sum : S n = 2 * a n + 1) : a n = -2 * n - 1 := sorry

end NUMINAMATH_GPT_find_general_term_l1693_169344


namespace NUMINAMATH_GPT_max_equal_product_l1693_169306

theorem max_equal_product (a b c d e f : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 20) (h4 : d = 30) (h5 : e = 40) (h6 : f = 60) :
  ∃ S, (a * b * c * d * e * f) * 450 = S^3 ∧ S = 18000 := 
by
  sorry

end NUMINAMATH_GPT_max_equal_product_l1693_169306


namespace NUMINAMATH_GPT_sum_of_midpoint_coords_l1693_169371

theorem sum_of_midpoint_coords :
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym = 11 :=
by
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  sorry

end NUMINAMATH_GPT_sum_of_midpoint_coords_l1693_169371


namespace NUMINAMATH_GPT_smallest_q_exists_l1693_169393

theorem smallest_q_exists (p q : ℕ) (h : 0 < q) (h_eq : (p : ℚ) / q = 123456789 / 100000000000) :
  q = 10989019 :=
sorry

end NUMINAMATH_GPT_smallest_q_exists_l1693_169393


namespace NUMINAMATH_GPT_problem_solution_l1693_169315
open Real

theorem problem_solution (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  a * (1 - b) ≤ 1 / 4 ∨ b * (1 - c) ≤ 1 / 4 ∨ c * (1 - a) ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1693_169315


namespace NUMINAMATH_GPT_x_values_l1693_169362

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_x_values_l1693_169362


namespace NUMINAMATH_GPT_number_of_cows_in_farm_l1693_169357

-- Definitions relating to the conditions
def total_bags_consumed := 20
def bags_per_cow := 1
def days := 20

-- Question and proof of the answer
theorem number_of_cows_in_farm : (total_bags_consumed / bags_per_cow) = 20 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_cows_in_farm_l1693_169357


namespace NUMINAMATH_GPT_find_f_x_minus_1_l1693_169397

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem find_f_x_minus_1 (x : ℝ) : f (x - 1) = 2 * x - 3 := by
  sorry

end NUMINAMATH_GPT_find_f_x_minus_1_l1693_169397


namespace NUMINAMATH_GPT_total_money_l1693_169381

namespace MoneyProof

variables (B J T : ℕ)

-- Given conditions
def condition_beth : Prop := B + 35 = 105
def condition_jan : Prop := J - 10 = B
def condition_tom : Prop := T = 3 * (J - 10)

-- Proof that the total money is $360
theorem total_money (h1 : condition_beth B) (h2 : condition_jan B J) (h3 : condition_tom J T) :
  B + J + T = 360 :=
by
  sorry

end MoneyProof

end NUMINAMATH_GPT_total_money_l1693_169381


namespace NUMINAMATH_GPT_good_games_count_l1693_169380

theorem good_games_count :
  ∀ (g1 g2 b : ℕ), g1 = 50 → g2 = 27 → b = 74 → g1 + g2 - b = 3 := by
  intros g1 g2 b hg1 hg2 hb
  sorry

end NUMINAMATH_GPT_good_games_count_l1693_169380


namespace NUMINAMATH_GPT_probability_of_3_tails_in_8_flips_l1693_169389

open ProbabilityTheory

/-- The probability of getting exactly 3 tails out of 8 flips of an unfair coin, where the probability of tails is 4/5 and the probability of heads is 1/5, is 3584/390625. -/
theorem probability_of_3_tails_in_8_flips :
  let p_heads := 1 / 5
  let p_tails := 4 / 5
  let n_trials := 8
  let k_successes := 3
  let binomial_coefficient := Nat.choose n_trials k_successes
  let probability := binomial_coefficient * (p_tails ^ k_successes) * (p_heads ^ (n_trials - k_successes))
  probability = (3584 : ℚ) / 390625 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_3_tails_in_8_flips_l1693_169389


namespace NUMINAMATH_GPT_compute_sum_of_products_of_coefficients_l1693_169343

theorem compute_sum_of_products_of_coefficients (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : ∀ x : ℝ, (x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) =
    (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4)) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_compute_sum_of_products_of_coefficients_l1693_169343


namespace NUMINAMATH_GPT_roof_length_width_diff_l1693_169351

theorem roof_length_width_diff (w l : ℕ) (h1 : l = 4 * w) (h2 : 784 = l * w) : l - w = 42 := by
  sorry

end NUMINAMATH_GPT_roof_length_width_diff_l1693_169351


namespace NUMINAMATH_GPT_plot_area_in_acres_l1693_169392

theorem plot_area_in_acres :
  let scale_cm_to_miles : ℝ := 3
  let base1_cm : ℝ := 20
  let base2_cm : ℝ := 25
  let height_cm : ℝ := 15
  let miles_to_acres : ℝ := 640
  let area_trapezoid_cm2 := (1 / 2) * (base1_cm + base2_cm) * height_cm
  let area_trapezoid_miles2 := area_trapezoid_cm2 * (scale_cm_to_miles ^ 2)
  let area_trapezoid_acres := area_trapezoid_miles2 * miles_to_acres
  area_trapezoid_acres = 1944000 := by
    sorry

end NUMINAMATH_GPT_plot_area_in_acres_l1693_169392


namespace NUMINAMATH_GPT_function_unique_l1693_169339

open Function

-- Define the domain and codomain
def NatPos : Type := {n : ℕ // n > 0}

-- Define the function f from positive integers to positive integers
noncomputable def f : NatPos → NatPos := sorry

-- Provide the main theorem
theorem function_unique (f : NatPos → NatPos) :
  (∀ (m n : NatPos), (m.val ^ 2 + (f n).val) ∣ ((m.val * (f m).val) + n.val)) →
  (∀ n : NatPos, f n = n) :=
by
  sorry

end NUMINAMATH_GPT_function_unique_l1693_169339


namespace NUMINAMATH_GPT_quadratic_sum_roots_l1693_169348

theorem quadratic_sum_roots {a b : ℝ}
  (h1 : ∀ x, x^2 - a * x + b < 0 ↔ -1 < x ∧ x < 3) :
  a + b = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_sum_roots_l1693_169348


namespace NUMINAMATH_GPT_average_cost_per_pencil_proof_l1693_169384

noncomputable def average_cost_per_pencil (pencils_qty: ℕ) (price: ℝ) (discount_percent: ℝ) (shipping_cost: ℝ) : ℝ :=
  let discounted_price := price * (1 - discount_percent / 100)
  let total_cost := discounted_price + shipping_cost
  let cost_in_cents := total_cost * 100
  cost_in_cents / pencils_qty

theorem average_cost_per_pencil_proof :
  average_cost_per_pencil 300 29.85 10 7.50 = 11 :=
by
  sorry

end NUMINAMATH_GPT_average_cost_per_pencil_proof_l1693_169384


namespace NUMINAMATH_GPT_base_five_equals_base_b_l1693_169322

theorem base_five_equals_base_b : ∃ (b : ℕ), b > 0 ∧ (2 * 5^1 + 4 * 5^0) = (1 * b^2 + 0 * b^1 + 1 * b^0) := by
  sorry

end NUMINAMATH_GPT_base_five_equals_base_b_l1693_169322


namespace NUMINAMATH_GPT_pages_left_l1693_169379

-- Define the conditions
def initial_books := 10
def pages_per_book := 100
def books_lost := 2

-- The total pages Phil had initially
def initial_pages := initial_books * pages_per_book

-- The number of books left after losing some during the move
def books_left := initial_books - books_lost

-- Prove the number of pages worth of books Phil has left
theorem pages_left : books_left * pages_per_book = 800 := by
  sorry

end NUMINAMATH_GPT_pages_left_l1693_169379


namespace NUMINAMATH_GPT_pipeline_problem_l1693_169375

theorem pipeline_problem 
  (length_pipeline : ℕ) 
  (extra_meters : ℕ) 
  (days_saved : ℕ) 
  (x : ℕ)
  (h1 : length_pipeline = 4000) 
  (h2 : extra_meters = 10) 
  (h3 : days_saved = 20) 
  (h4 : (4000:ℕ) / (x - extra_meters) - (4000:ℕ) / x = days_saved) :
  x = 4000 / ((4000 / (x - extra_meters) + 20)) + extra_meters :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_pipeline_problem_l1693_169375


namespace NUMINAMATH_GPT_sqrt_sum_eq_six_l1693_169307

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end NUMINAMATH_GPT_sqrt_sum_eq_six_l1693_169307


namespace NUMINAMATH_GPT_at_least_one_div_by_5_l1693_169394

-- Define natural numbers and divisibility by 5
def is_div_by_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- Proposition: If a, b are natural numbers and ab is divisible by 5, then at least one of a or b must be divisible by 5.
theorem at_least_one_div_by_5 (a b : ℕ) (h_ab : is_div_by_5 (a * b)) : is_div_by_5 a ∨ is_div_by_5 b :=
  by
    sorry

end NUMINAMATH_GPT_at_least_one_div_by_5_l1693_169394


namespace NUMINAMATH_GPT_yarn_total_length_l1693_169383

/-- The green yarn is 156 cm long, the red yarn is 8 cm more than three times the green yarn,
    prove that the total length of the two pieces of yarn is 632 cm. --/
theorem yarn_total_length : 
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  green_yarn + red_yarn = 632 :=
by
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  sorry

end NUMINAMATH_GPT_yarn_total_length_l1693_169383


namespace NUMINAMATH_GPT_does_not_represent_right_triangle_l1693_169367

/-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively. Given:
  - a:b:c = 6:8:10
  - ∠A:∠B:∠C = 1:1:3
  - a^2 + c^2 = b^2
  - ∠A + ∠B = ∠C

Prove that the condition ∠A:∠B:∠C = 1:1:3 does not represent a right triangle ABC. -/
theorem does_not_represent_right_triangle
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a / b = 6 / 8 ∧ b / c = 8 / 10)
  (h2 : A / B = 1 / 1 ∧ B / C = 1 / 3)
  (h3 : a^2 + c^2 = b^2)
  (h4 : A + B = C) :
  ¬ (B = 90) :=
sorry

end NUMINAMATH_GPT_does_not_represent_right_triangle_l1693_169367


namespace NUMINAMATH_GPT_blipblish_modulo_l1693_169345

-- Definitions from the conditions
inductive Letter
| B | I | L

def is_consonant (c : Letter) : Bool :=
  match c with
  | Letter.B | Letter.L => true
  | _ => false

def is_vowel (v : Letter) : Bool :=
  match v with
  | Letter.I => true
  | _ => false

def is_valid_blipblish_word (word : List Letter) : Bool :=
  -- Check if between any two I's there at least three consonants
  let rec check (lst : List Letter) (cnt : Nat) (during_vowels : Bool) : Bool :=
    match lst with
    | [] => true
    | Letter.I :: xs =>
        if during_vowels then cnt >= 3 && check xs 0 false
        else check xs 0 true
    | x :: xs =>
        if is_consonant x then check xs (cnt + 1) during_vowels
        else check xs cnt during_vowels
  check word 0 false

def number_of_valid_words (n : Nat) : Nat :=
  -- Placeholder function to compute the number of valid Blipblish words of length n
  sorry

-- Statement of the proof problem
theorem blipblish_modulo : number_of_valid_words 12 % 1000 = 312 :=
by sorry

end NUMINAMATH_GPT_blipblish_modulo_l1693_169345


namespace NUMINAMATH_GPT_days_to_clear_messages_l1693_169333

theorem days_to_clear_messages 
  (initial_messages : ℕ)
  (messages_read_per_day : ℕ)
  (new_messages_per_day : ℕ) 
  (net_messages_cleared_per_day : ℕ)
  (d : ℕ) :
  initial_messages = 98 →
  messages_read_per_day = 20 →
  new_messages_per_day = 6 →
  net_messages_cleared_per_day = messages_read_per_day - new_messages_per_day →
  d = initial_messages / net_messages_cleared_per_day →
  d = 7 :=
by
  intros h_initial h_read h_new h_net h_days
  sorry

end NUMINAMATH_GPT_days_to_clear_messages_l1693_169333


namespace NUMINAMATH_GPT_frac_left_handed_l1693_169376

variable (x : ℕ)

def red_participants := 10 * x
def blue_participants := 5 * x
def total_participants := red_participants x + blue_participants x

def left_handed_red := (1 / 3 : ℚ) * red_participants x
def left_handed_blue := (2 / 3 : ℚ) * blue_participants x
def total_left_handed := left_handed_red x + left_handed_blue x

theorem frac_left_handed :
  total_left_handed x / total_participants x = (4 / 9 : ℚ) := by
  sorry

end NUMINAMATH_GPT_frac_left_handed_l1693_169376


namespace NUMINAMATH_GPT_total_worth_of_produce_is_630_l1693_169302

def bundles_of_asparagus : ℕ := 60
def price_per_bundle_asparagus : ℝ := 3.00

def boxes_of_grapes : ℕ := 40
def price_per_box_grapes : ℝ := 2.50

def num_apples : ℕ := 700
def price_per_apple : ℝ := 0.50

def total_worth : ℝ :=
  bundles_of_asparagus * price_per_bundle_asparagus +
  boxes_of_grapes * price_per_box_grapes +
  num_apples * price_per_apple

theorem total_worth_of_produce_is_630 : 
  total_worth = 630 := by
  sorry

end NUMINAMATH_GPT_total_worth_of_produce_is_630_l1693_169302


namespace NUMINAMATH_GPT_intersection_of_set_M_with_complement_of_set_N_l1693_169321

theorem intersection_of_set_M_with_complement_of_set_N (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 4, 5}) (hN : N = {1, 3}) : M ∩ (U \ N) = {4, 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_set_M_with_complement_of_set_N_l1693_169321


namespace NUMINAMATH_GPT_shooter_prob_l1693_169331

variable (hit_prob : ℝ)
variable (miss_prob : ℝ := 1 - hit_prob)
variable (p1 : hit_prob = 0.85)
variable (independent_shots : true)

theorem shooter_prob :
  miss_prob * miss_prob * hit_prob = 0.019125 :=
by
  rw [p1]
  sorry

end NUMINAMATH_GPT_shooter_prob_l1693_169331


namespace NUMINAMATH_GPT_number_of_pieces_from_rod_l1693_169326

theorem number_of_pieces_from_rod (rod_length_m : ℕ) (piece_length_cm : ℕ) (meter_to_cm : ℕ) 
  (h1 : rod_length_m = 34) (h2 : piece_length_cm = 85) (h3 : meter_to_cm = 100) : 
  rod_length_m * meter_to_cm / piece_length_cm = 40 := by
  sorry

end NUMINAMATH_GPT_number_of_pieces_from_rod_l1693_169326


namespace NUMINAMATH_GPT_problem1_problem2_solution_l1693_169349

noncomputable def trig_expr : ℝ :=
  3 * Real.tan (30 * Real.pi / 180) - (Real.tan (45 * Real.pi / 180))^2 + 2 * Real.sin (60 * Real.pi / 180)

theorem problem1 : trig_expr = 2 * Real.sqrt 3 - 1 :=
by
  -- Proof omitted
  sorry

noncomputable def quad_eq (x : ℝ) : Prop := 
  (3*x - 1) * (x + 2) = 11*x - 4

theorem problem2_solution (x : ℝ) : quad_eq x ↔ (x = (3 + Real.sqrt 3) / 3 ∨ x = (3 - Real.sqrt 3) / 3) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem1_problem2_solution_l1693_169349


namespace NUMINAMATH_GPT_find_a_l1693_169301

noncomputable def coefficient_of_x3_in_expansion (a : ℝ) : ℝ :=
  6 * a^2 - 15 * a + 20 

theorem find_a (a : ℝ) (h : coefficient_of_x3_in_expansion a = 56) : a = 6 ∨ a = -1 :=
  sorry

end NUMINAMATH_GPT_find_a_l1693_169301


namespace NUMINAMATH_GPT_compute_expression_l1693_169391

theorem compute_expression :
  ( (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) )
  /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) )
  = 221 := 
by sorry

end NUMINAMATH_GPT_compute_expression_l1693_169391


namespace NUMINAMATH_GPT_complete_square_eq_l1693_169377

theorem complete_square_eq (x : ℝ) :
  x^2 - 8 * x + 15 = 0 →
  (x - 4)^2 = 1 :=
by sorry

end NUMINAMATH_GPT_complete_square_eq_l1693_169377


namespace NUMINAMATH_GPT_solution_l1693_169320

theorem solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 := 
by 
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_solution_l1693_169320


namespace NUMINAMATH_GPT_repeating_decimal_sum_num_denom_l1693_169312

noncomputable def repeating_decimal_to_fraction (n d : ℕ) (rep : ℚ) : ℚ :=
(rep * (10^d) - rep) / ((10^d) - 1)

theorem repeating_decimal_sum_num_denom
  (x : ℚ)
  (h1 : x = repeating_decimal_to_fraction 45 2 0.45)
  (h2 : repeating_decimal_to_fraction 45 2 0.45 = 5/11) : 
  (5 + 11) = 16 :=
by 
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_num_denom_l1693_169312


namespace NUMINAMATH_GPT_Guido_costs_42840_l1693_169317

def LightningMcQueenCost : ℝ := 140000
def MaterCost : ℝ := 0.1 * LightningMcQueenCost
def SallyCostBeforeModifications : ℝ := 3 * MaterCost
def SallyCostAfterModifications : ℝ := SallyCostBeforeModifications + 0.2 * SallyCostBeforeModifications
def GuidoCost : ℝ := SallyCostAfterModifications - 0.15 * SallyCostAfterModifications

theorem Guido_costs_42840 :
  GuidoCost = 42840 :=
sorry

end NUMINAMATH_GPT_Guido_costs_42840_l1693_169317
