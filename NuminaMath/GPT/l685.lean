import Mathlib

namespace NUMINAMATH_GPT_find_t_l685_68578

theorem find_t:
  (∃ t, (∀ (x y: ℝ), (x = 2 ∧ y = 8) ∨ (x = 4 ∧ y = 14) ∨ (x = 6 ∧ y = 20) → 
                (∀ (m b: ℝ), y = m * x + b) ∧ 
                (∀ (m b: ℝ), y = 3 * x + b ∧ b = 2 ∧ (t = 3 * 50 + 2) ∧ t = 152))) := by
  sorry

end NUMINAMATH_GPT_find_t_l685_68578


namespace NUMINAMATH_GPT_number_of_five_digit_numbers_l685_68592

def count_five_identical_digits: Nat := 9
def count_two_different_digits: Nat := 1215
def count_three_different_digits: Nat := 6480
def count_four_different_digits: Nat := 22680
def count_five_different_digits: Nat := 27216

theorem number_of_five_digit_numbers :
  count_five_identical_digits + count_two_different_digits +
  count_three_different_digits + count_four_different_digits +
  count_five_different_digits = 57600 :=
by
  sorry

end NUMINAMATH_GPT_number_of_five_digit_numbers_l685_68592


namespace NUMINAMATH_GPT_little_john_spent_on_sweets_l685_68508

theorem little_john_spent_on_sweets
  (initial_amount : ℝ)
  (amount_per_friend : ℝ)
  (friends_count : ℕ)
  (amount_left : ℝ)
  (spent_on_sweets : ℝ) :
  initial_amount = 10.50 →
  amount_per_friend = 2.20 →
  friends_count = 2 →
  amount_left = 3.85 →
  spent_on_sweets = initial_amount - (amount_per_friend * friends_count) - amount_left →
  spent_on_sweets = 2.25 :=
by
  intros h_initial h_per_friend h_friends_count h_left h_spent
  sorry

end NUMINAMATH_GPT_little_john_spent_on_sweets_l685_68508


namespace NUMINAMATH_GPT_equilateral_triangle_area_l685_68575

theorem equilateral_triangle_area (h : Real) (h_eq : h = Real.sqrt 12):
  ∃ A : Real, A = 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_l685_68575


namespace NUMINAMATH_GPT_find_expression_value_l685_68580

variable (a b : ℝ)

theorem find_expression_value (h : a - 2 * b = 7) : 6 - 2 * a + 4 * b = -8 := by
  sorry

end NUMINAMATH_GPT_find_expression_value_l685_68580


namespace NUMINAMATH_GPT_purchase_price_of_jacket_l685_68542

theorem purchase_price_of_jacket (S P : ℝ) (h1 : S = P + 0.30 * S)
                                (SP : ℝ) (h2 : SP = 0.80 * S)
                                (h3 : 8 = SP - P) :
                                P = 56 := by
  sorry

end NUMINAMATH_GPT_purchase_price_of_jacket_l685_68542


namespace NUMINAMATH_GPT_percentage_of_children_allowed_to_draw_l685_68526

def total_jelly_beans := 100
def total_children := 40
def remaining_jelly_beans := 36
def jelly_beans_per_child := 2

theorem percentage_of_children_allowed_to_draw :
  ((total_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child : ℕ) * 100 / total_children = 80 := by
  sorry

end NUMINAMATH_GPT_percentage_of_children_allowed_to_draw_l685_68526


namespace NUMINAMATH_GPT_average_marks_l685_68543

variable (M P C B : ℕ)

theorem average_marks (h1 : M + P = 20) (h2 : C = P + 20) 
  (h3 : B = 2 * M) (h4 : M ≤ 100) (h5 : P ≤ 100) (h6 : C ≤ 100) (h7 : B ≤ 100) :
  (M + C) / 2 = 20 := by
  sorry

end NUMINAMATH_GPT_average_marks_l685_68543


namespace NUMINAMATH_GPT_toothbrushes_difference_l685_68545

theorem toothbrushes_difference
  (total : ℕ)
  (jan : ℕ)
  (feb : ℕ)
  (mar : ℕ)
  (apr_may_sum : total = jan + feb + mar + 164)
  (apr_may_half : 164 / 2 = 82)
  (busy_month_given : feb = 67)
  (slow_month_given : mar = 46) :
  feb - mar = 21 :=
by
  sorry

end NUMINAMATH_GPT_toothbrushes_difference_l685_68545


namespace NUMINAMATH_GPT_total_lucky_stars_l685_68524

theorem total_lucky_stars : 
  (∃ n : ℕ, 10 * n + 6 = 116 ∧ 4 * 8 + (n - 4) * 12 = 116) → 
  116 = 116 := 
by
  intro h
  obtain ⟨n, h1, h2⟩ := h
  sorry

end NUMINAMATH_GPT_total_lucky_stars_l685_68524


namespace NUMINAMATH_GPT_range_of_a_l685_68576

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else x^2 + x -- Note: Using the specific definition matches the problem constraints clearly.

theorem range_of_a (a : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_ineq : f a + f (-a) < 4) : -1 < a ∧ a < 1 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l685_68576


namespace NUMINAMATH_GPT_cashier_total_value_l685_68539

theorem cashier_total_value (total_bills : ℕ) (ten_bills : ℕ) (twenty_bills : ℕ)
  (h1 : total_bills = 30) (h2 : ten_bills = 27) (h3 : twenty_bills = 3) :
  (10 * ten_bills + 20 * twenty_bills) = 330 :=
by
  sorry

end NUMINAMATH_GPT_cashier_total_value_l685_68539


namespace NUMINAMATH_GPT_fewest_erasers_l685_68513

theorem fewest_erasers :
  ∀ (JK JM SJ : ℕ), 
  (JK = 6) →
  (JM = JK + 4) →
  (SJ = JM - 3) →
  (JK ≤ JM ∧ JK ≤ SJ) :=
by
  intros JK JM SJ hJK hJM hSJ
  sorry

end NUMINAMATH_GPT_fewest_erasers_l685_68513


namespace NUMINAMATH_GPT_total_books_received_l685_68583

theorem total_books_received (initial_books additional_books total_books: ℕ)
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  (initial_books + additional_books = 77) := by
  sorry

end NUMINAMATH_GPT_total_books_received_l685_68583


namespace NUMINAMATH_GPT_simplify_expression_l685_68567

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  (a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2)) = 1 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l685_68567


namespace NUMINAMATH_GPT_count100DigitEvenNumbers_is_correct_l685_68587

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end NUMINAMATH_GPT_count100DigitEvenNumbers_is_correct_l685_68587


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l685_68540

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 9) (h2 : a * r^6 = 1) : a * r^4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l685_68540


namespace NUMINAMATH_GPT_convert_500_to_base2_l685_68518

theorem convert_500_to_base2 :
  let n_base10 : ℕ := 500
  let n_base8 : ℕ := 7 * 64 + 6 * 8 + 4
  let n_base2 : ℕ := 1 * 256 + 1 * 128 + 1 * 64 + 1 * 32 + 1 * 16 + 0 * 8 + 1 * 4 + 0 * 2 + 0
  n_base10 = 500 ∧ n_base8 = 500 ∧ n_base2 = n_base8 :=
by
  sorry

end NUMINAMATH_GPT_convert_500_to_base2_l685_68518


namespace NUMINAMATH_GPT_gervais_avg_mileage_l685_68572
variable (x : ℤ)

def gervais_daily_mileage : Prop := ∃ (x : ℤ), (3 * x = 1250 - 305) ∧ x = 315

theorem gervais_avg_mileage : gervais_daily_mileage :=
by
  sorry

end NUMINAMATH_GPT_gervais_avg_mileage_l685_68572


namespace NUMINAMATH_GPT_sequence_bound_equivalent_problem_l685_68514

variable {n : ℕ}
variable {a : Fin (n+2) → ℝ}

theorem sequence_bound_equivalent_problem (h1 : a 0 = 0) (h2 : a (n + 1) = 0) 
  (h3 : ∀ k : Fin n, |a (k.val - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : Fin (n+2), |a k| ≤ k * (n + 1 - k) / 2 := 
by
  sorry

end NUMINAMATH_GPT_sequence_bound_equivalent_problem_l685_68514


namespace NUMINAMATH_GPT_book_pages_l685_68509

theorem book_pages (books sheets pages_per_sheet pages_per_book : ℕ)
  (hbooks : books = 2)
  (hpages_per_sheet : pages_per_sheet = 8)
  (hsheets : sheets = 150)
  (htotal_pages : pages_per_sheet * sheets = 1200)
  (hpages_per_book : pages_per_book = 1200 / books) :
  pages_per_book = 600 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_book_pages_l685_68509


namespace NUMINAMATH_GPT_unique_partition_no_primes_l685_68573

open Set

def C_oplus_C (C : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ C ∧ y ∈ C ∧ x ≠ y ∧ z = x + y}

def is_partition (A B : Set ℕ) : Prop :=
  (A ∪ B = univ) ∧ (A ∩ B = ∅)

theorem unique_partition_no_primes (A B : Set ℕ) :
  (is_partition A B) ∧ (∀ x ∈ C_oplus_C A, ¬Nat.Prime x) ∧ (∀ x ∈ C_oplus_C B, ¬Nat.Prime x) ↔ 
    (A = { n | n % 2 = 1 }) ∧ (B = { n | n % 2 = 0 }) :=
sorry

end NUMINAMATH_GPT_unique_partition_no_primes_l685_68573


namespace NUMINAMATH_GPT_eval_expression_l685_68595

theorem eval_expression (a : ℕ) (h : a = 2) : a^3 * a^6 = 512 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l685_68595


namespace NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l685_68516

theorem ratio_of_spinsters_to_cats :
  (∀ S C : ℕ, (S : ℚ) / (C : ℚ) = 2 / 9) ↔
  (∃ S C : ℕ, S = 18 ∧ C = S + 63 ∧ (S : ℚ) / (C : ℚ) = 2 / 9) :=
sorry

end NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l685_68516


namespace NUMINAMATH_GPT_find_positive_X_l685_68535

variable (X : ℝ) (Y : ℝ)

def hash_rel (X Y : ℝ) : ℝ :=
  X^2 + Y^2

theorem find_positive_X :
  hash_rel X 7 = 250 → X = Real.sqrt 201 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_X_l685_68535


namespace NUMINAMATH_GPT_caterpillars_and_leaves_l685_68525

def initial_caterpillars : Nat := 14
def caterpillars_after_storm : Nat := initial_caterpillars - 3
def hatched_eggs : Nat := 6
def caterpillars_after_hatching : Nat := caterpillars_after_storm + hatched_eggs
def leaves_eaten_by_babies : Nat := 18
def caterpillars_after_cocooning : Nat := caterpillars_after_hatching - 9
def moth_caterpillars : Nat := caterpillars_after_cocooning / 2
def butterfly_caterpillars : Nat := caterpillars_after_cocooning - moth_caterpillars
def leaves_eaten_per_moth_per_day : Nat := 4
def days_in_week : Nat := 7
def total_leaves_eaten_by_moths : Nat := moth_caterpillars * leaves_eaten_per_moth_per_day * days_in_week
def total_leaves_eaten_by_babies_and_moths : Nat := leaves_eaten_by_babies + total_leaves_eaten_by_moths

theorem caterpillars_and_leaves :
  (caterpillars_after_cocooning = 8) ∧ (total_leaves_eaten_by_babies_and_moths = 130) :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_caterpillars_and_leaves_l685_68525


namespace NUMINAMATH_GPT_sum_first_99_terms_l685_68596

def geom_sum (n : ℕ) : ℕ := (2^n) - 1

def seq_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum geom_sum

theorem sum_first_99_terms :
  seq_sum 99 = 2^100 - 101 := by
  sorry

end NUMINAMATH_GPT_sum_first_99_terms_l685_68596


namespace NUMINAMATH_GPT_Tim_soda_cans_l685_68566

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end NUMINAMATH_GPT_Tim_soda_cans_l685_68566


namespace NUMINAMATH_GPT_curve_equation_l685_68553

theorem curve_equation
  (a b : ℝ)
  (h1 : a * 0 ^ 2 + b * (5 / 3) ^ 2 = 2)
  (h2 : a * 1 ^ 2 + b * 1 ^ 2 = 2) :
  (16 / 25) * x^2 + (9 / 25) * y^2 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_curve_equation_l685_68553


namespace NUMINAMATH_GPT_find_larger_integer_l685_68552

noncomputable def larger_integer (a b : ℤ) := max a b

theorem find_larger_integer (a b : ℕ) 
  (h1 : a/b = 7/3) 
  (h2 : a * b = 294): 
  larger_integer a b = 7 * Real.sqrt 14 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_larger_integer_l685_68552


namespace NUMINAMATH_GPT_tallest_boy_is_Vladimir_l685_68548

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end NUMINAMATH_GPT_tallest_boy_is_Vladimir_l685_68548


namespace NUMINAMATH_GPT_investment_interest_rate_calculation_l685_68541

theorem investment_interest_rate_calculation :
  let initial_investment : ℝ := 15000
  let first_year_rate : ℝ := 0.08
  let first_year_investment : ℝ := initial_investment * (1 + first_year_rate)
  let second_year_investment : ℝ := 17160
  ∃ (s : ℝ), (first_year_investment * (1 + s / 100) = second_year_investment) → s = 6 :=
by
  sorry

end NUMINAMATH_GPT_investment_interest_rate_calculation_l685_68541


namespace NUMINAMATH_GPT_w1_relation_w2_relation_maximize_total_profit_l685_68559

def w1 (x : ℕ) : ℤ := 200 * x - 10000

def w2 (x : ℕ) : ℤ := -(x ^ 2) + 1000 * x - 50000

def total_sales_vol (x y : ℕ) : Prop := x + y = 1000

def max_profit_volumes (x y : ℕ) : Prop :=
  total_sales_vol x y ∧ x = 600 ∧ y = 400

theorem w1_relation (x : ℕ) :
  w1 x = 200 * x - 10000 := 
sorry

theorem w2_relation (x : ℕ) :
  w2 x = -(x ^ 2) + 1000 * x - 50000 := 
sorry

theorem maximize_total_profit (x y : ℕ) :
  total_sales_vol x y → max_profit_volumes x y := 
sorry

end NUMINAMATH_GPT_w1_relation_w2_relation_maximize_total_profit_l685_68559


namespace NUMINAMATH_GPT_julia_error_approx_97_percent_l685_68585

theorem julia_error_approx_97_percent (x : ℝ) : 
  abs ((6 * x - x / 6) / (6 * x) * 100 - 97) < 1 :=
by 
  sorry

end NUMINAMATH_GPT_julia_error_approx_97_percent_l685_68585


namespace NUMINAMATH_GPT_twenty_five_percent_greater_l685_68504

theorem twenty_five_percent_greater (x : ℕ) (h : x = (88 + (88 * 25) / 100)) : x = 110 :=
sorry

end NUMINAMATH_GPT_twenty_five_percent_greater_l685_68504


namespace NUMINAMATH_GPT_inverse_of_3_mod_185_l685_68534

theorem inverse_of_3_mod_185 : ∃ x : ℕ, 0 ≤ x ∧ x < 185 ∧ 3 * x ≡ 1 [MOD 185] :=
by
  use 62
  sorry

end NUMINAMATH_GPT_inverse_of_3_mod_185_l685_68534


namespace NUMINAMATH_GPT_arrange_in_ascending_order_l685_68537

theorem arrange_in_ascending_order (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end NUMINAMATH_GPT_arrange_in_ascending_order_l685_68537


namespace NUMINAMATH_GPT_find_a_and_b_l685_68544

noncomputable def a_and_b (x y : ℝ) (a b : ℝ) : Prop :=
  a = Real.sqrt x + Real.sqrt y ∧ b = Real.sqrt (x + 2) + Real.sqrt (y + 2) ∧
  ∃ n : ℤ, a = n ∧ b = n + 2

theorem find_a_and_b (x y : ℝ) (a b : ℝ)
  (h₁ : 0 ≤ x)
  (h₂ : 0 ≤ y)
  (h₃ : a_and_b x y a b)
  (h₄ : ∃ n : ℤ, a = n ∧ b = n + 2) :
  a = 1 ∧ b = 3 := by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l685_68544


namespace NUMINAMATH_GPT_balls_color_equality_l685_68593

theorem balls_color_equality (r g b: ℕ) (h1: r + g + b = 20) (h2: b ≥ 7) (h3: r ≥ 4) (h4: b = 2 * g) : 
  r = b ∨ r = g :=
by
  sorry

end NUMINAMATH_GPT_balls_color_equality_l685_68593


namespace NUMINAMATH_GPT_proof_inequalities_l685_68531

theorem proof_inequalities (A B C D E : ℝ) (p q r s t : ℝ)
  (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D < E)
  (h5 : p = B - A) (h6 : q = C - A) (h7 : r = D - A)
  (h8 : s = E - B) (h9 : t = E - D)
  (ineq1 : p + 2 * s > r + t)
  (ineq2 : r + t > p)
  (ineq3 : r + t > s) :
  (p < r / 2) ∧ (s < t + p / 2) :=
by 
  sorry

end NUMINAMATH_GPT_proof_inequalities_l685_68531


namespace NUMINAMATH_GPT_evaluate_f_at_4_l685_68590

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem evaluate_f_at_4 : f 4 = 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_4_l685_68590


namespace NUMINAMATH_GPT_lassis_from_mangoes_l685_68501

-- Define the given ratio
def lassis_per_mango := 15 / 3

-- Define the number of mangoes
def mangoes := 15

-- Define the expected number of lassis
def expected_lassis := 75

-- Prove that with 15 mangoes, 75 lassis can be made given the ratio
theorem lassis_from_mangoes (h : lassis_per_mango = 5) : mangoes * lassis_per_mango = expected_lassis :=
by
  sorry

end NUMINAMATH_GPT_lassis_from_mangoes_l685_68501


namespace NUMINAMATH_GPT_rectangle_width_decreased_by_33_percent_l685_68511

theorem rectangle_width_decreased_by_33_percent
  (L W A : ℝ)
  (hA : A = L * W)
  (newL : ℝ)
  (h_newL : newL = 1.5 * L)
  (W' : ℝ)
  (h_area_unchanged : newL * W' = A) : 
  (1 - W' / W) * 100 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_decreased_by_33_percent_l685_68511


namespace NUMINAMATH_GPT_area_ACD_l685_68522

def base_ABD : ℝ := 8
def height_ABD : ℝ := 4
def base_ABC : ℝ := 4
def height_ABC : ℝ := 4

theorem area_ACD : (1/2 * base_ABD * height_ABD) - (1/2 * base_ABC * height_ABC) = 8 := by
  sorry

end NUMINAMATH_GPT_area_ACD_l685_68522


namespace NUMINAMATH_GPT_five_letter_word_combinations_l685_68536

open Nat

theorem five_letter_word_combinations :
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  total_combinations = 456976 := 
by
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  show total_combinations = 456976
  sorry

end NUMINAMATH_GPT_five_letter_word_combinations_l685_68536


namespace NUMINAMATH_GPT_area_of_moving_point_l685_68503

theorem area_of_moving_point (a b : ℝ) :
  (∀ (x y : ℝ), abs x ≤ 1 ∧ abs y ≤ 1 → a * x - 2 * b * y ≤ 2) →
  ∃ (A : ℝ), A = 8 := sorry

end NUMINAMATH_GPT_area_of_moving_point_l685_68503


namespace NUMINAMATH_GPT_power_function_value_l685_68591

theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h₁ : f x = x ^ α) (h₂ : f (1 / 2) = 4) : f 8 = 1 / 64 := by
  sorry

end NUMINAMATH_GPT_power_function_value_l685_68591


namespace NUMINAMATH_GPT_ratio_third_to_first_l685_68564

theorem ratio_third_to_first (F S T : ℕ) (h1 : F = 33) (h2 : S = 4 * F) (h3 : (F + S + T) / 3 = 77) :
  T / F = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_third_to_first_l685_68564


namespace NUMINAMATH_GPT_find_a_for_arithmetic_progression_roots_l685_68538

theorem find_a_for_arithmetic_progression_roots (x a : ℝ) : 
  (∀ (x : ℝ), x^4 - a*x^2 + 1 = 0) → 
  (∃ (t1 t2 : ℝ), t1 > 0 ∧ t2 > 0 ∧ (t2 = 9*t1) ∧ (t1 + t2 = a) ∧ (t1 * t2 = 1)) → 
  (a = 10/3) := 
  by 
    intros h1 h2
    sorry

end NUMINAMATH_GPT_find_a_for_arithmetic_progression_roots_l685_68538


namespace NUMINAMATH_GPT_burger_cost_is_350_l685_68560

noncomputable def cost_of_each_burger (tip steak_cost steak_quantity ice_cream_cost ice_cream_quantity money_left: ℝ) : ℝ :=
(tip - money_left - (steak_cost * steak_quantity + ice_cream_cost * ice_cream_quantity)) / 2

theorem burger_cost_is_350 :
  cost_of_each_burger 99 24 2 2 3 38 = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_burger_cost_is_350_l685_68560


namespace NUMINAMATH_GPT_spade_problem_l685_68550

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 2 (spade 3 (spade 1 4)) = -46652 := 
by sorry

end NUMINAMATH_GPT_spade_problem_l685_68550


namespace NUMINAMATH_GPT_notebook_cost_correct_l685_68551

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end NUMINAMATH_GPT_notebook_cost_correct_l685_68551


namespace NUMINAMATH_GPT_least_value_xy_l685_68549

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end NUMINAMATH_GPT_least_value_xy_l685_68549


namespace NUMINAMATH_GPT_difference_in_dimes_l685_68563

theorem difference_in_dimes : 
  ∀ (a b c : ℕ), (a + b + c = 100) → (5 * a + 10 * b + 25 * c = 835) → 
  (∀ b_max b_min, (b_max = 67) ∧ (b_min = 3) → (b_max - b_min = 64)) :=
by
  intros a b c h1 h2 b_max b_min h_bounds
  sorry

end NUMINAMATH_GPT_difference_in_dimes_l685_68563


namespace NUMINAMATH_GPT_treasure_coins_problem_l685_68532

theorem treasure_coins_problem (N m n t k s u : ℤ) 
  (h1 : N = (2/3) * (2/3) * (2/3) * (m - 1) - (2/3) - (2^2 / 3^2))
  (h2 : N = 3 * n)
  (h3 : 8 * (m - 1) - 30 = 81 * k)
  (h4 : m - 1 = 3 * t)
  (h5 : 8 * t - 27 * k = 10)
  (h6 : m = 3 * t + 1)
  (h7 : k = 2 * s)
  (h8 : 4 * t - 27 * s = 5)
  (h9 : t = 8 + 27 * u)
  (h10 : s = 1 + 4 * u)
  (h11 : 110 ≤ 81 * u + 25)
  (h12 : 81 * u + 25 ≤ 200) :
  m = 187 :=
sorry

end NUMINAMATH_GPT_treasure_coins_problem_l685_68532


namespace NUMINAMATH_GPT_trig_identity_solution_l685_68515

theorem trig_identity_solution (α : ℝ) (h : Real.tan α = -1 / 2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_solution_l685_68515


namespace NUMINAMATH_GPT_sum_of_products_leq_one_third_l685_68599

theorem sum_of_products_leq_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 :=
sorry

end NUMINAMATH_GPT_sum_of_products_leq_one_third_l685_68599


namespace NUMINAMATH_GPT_find_second_number_l685_68512

theorem find_second_number (A B : ℝ) (h1 : A = 3200) (h2 : 0.10 * A = 0.20 * B + 190) : B = 650 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l685_68512


namespace NUMINAMATH_GPT_tens_digit_of_smallest_even_five_digit_number_l685_68579

def smallest_even_five_digit_number (digits : List ℕ) : ℕ :=
if h : 0 ∈ digits ∧ 3 ∈ digits ∧ 5 ∈ digits ∧ 6 ∈ digits ∧ 8 ∈ digits then
  35086
else
  0  -- this is just a placeholder to make the function total

theorem tens_digit_of_smallest_even_five_digit_number : 
  ∀ digits : List ℕ, 
    0 ∈ digits ∧ 
    3 ∈ digits ∧ 
    5 ∈ digits ∧ 
    6 ∈ digits ∧ 
    8 ∈ digits ∧ 
    digits.length = 5 → 
    (smallest_even_five_digit_number digits) / 10 % 10 = 8 :=
by
  intros digits h
  sorry

end NUMINAMATH_GPT_tens_digit_of_smallest_even_five_digit_number_l685_68579


namespace NUMINAMATH_GPT_regression_analysis_correct_l685_68502

-- Definition of the regression analysis context
def regression_analysis_variation (forecast_var : Type) (explanatory_var residual_var : Type) : Prop :=
  forecast_var = explanatory_var ∧ forecast_var = residual_var

-- The theorem to prove
theorem regression_analysis_correct :
  ∀ (forecast_var explanatory_var residual_var : Type),
  regression_analysis_variation forecast_var explanatory_var residual_var →
  (forecast_var = explanatory_var ∧ forecast_var = residual_var) :=
by
  intro forecast_var explanatory_var residual_var h
  exact h

end NUMINAMATH_GPT_regression_analysis_correct_l685_68502


namespace NUMINAMATH_GPT_ratio_arms_martians_to_aliens_l685_68506

def arms_of_aliens : ℕ := 3
def legs_of_aliens : ℕ := 8
def legs_of_martians := legs_of_aliens / 2

def limbs_of_5_aliens := 5 * (arms_of_aliens + legs_of_aliens)
def limbs_of_5_martians (arms_of_martians : ℕ) := 5 * (arms_of_martians + legs_of_martians)

theorem ratio_arms_martians_to_aliens (A_m : ℕ) (h1 : limbs_of_5_aliens = limbs_of_5_martians A_m + 5) :
  (A_m : ℚ) / arms_of_aliens = 2 :=
sorry

end NUMINAMATH_GPT_ratio_arms_martians_to_aliens_l685_68506


namespace NUMINAMATH_GPT_cards_per_set_is_13_l685_68581

-- Definitions based on the conditions
def total_cards : ℕ := 365
def sets_to_brother : ℕ := 8
def sets_to_sister : ℕ := 5
def sets_to_friend : ℕ := 2
def total_sets_given : ℕ := sets_to_brother + sets_to_sister + sets_to_friend
def total_cards_given : ℕ := 195

-- The problem to prove
theorem cards_per_set_is_13 : total_cards_given / total_sets_given = 13 :=
  by
  -- Here we would provide the proof, but for now, we use sorry
  sorry

end NUMINAMATH_GPT_cards_per_set_is_13_l685_68581


namespace NUMINAMATH_GPT_mile_time_sum_is_11_l685_68577

def mile_time_sum (Tina_time Tony_time Tom_time : ℕ) : ℕ :=
  Tina_time + Tony_time + Tom_time

theorem mile_time_sum_is_11 :
  ∃ (Tina_time Tony_time Tom_time : ℕ),
  (Tina_time = 6 ∧ Tony_time = Tina_time / 2 ∧ Tom_time = Tina_time / 3) →
  mile_time_sum Tina_time Tony_time Tom_time = 11 :=
by
  sorry

end NUMINAMATH_GPT_mile_time_sum_is_11_l685_68577


namespace NUMINAMATH_GPT_evie_shells_l685_68557

theorem evie_shells (shells_per_day : ℕ) (days : ℕ) (gifted_shells : ℕ) 
  (h1 : shells_per_day = 10) 
  (h2 : days = 6)
  (h3 : gifted_shells = 2) : 
  shells_per_day * days - gifted_shells = 58 := 
by
  sorry

end NUMINAMATH_GPT_evie_shells_l685_68557


namespace NUMINAMATH_GPT_find_interest_rate_l685_68554

-- Definitions from the conditions
def principal : ℕ := 1050
def time_period : ℕ := 6
def interest : ℕ := 378  -- Interest calculated as Rs. 1050 - Rs. 672

-- Correct Answer
def interest_rate : ℕ := 6

-- Lean 4 statement of the proof problem
theorem find_interest_rate (P : ℕ) (t : ℕ) (I : ℕ) 
    (hP : P = principal) (ht : t = time_period) (hI : I = interest) : 
    (I * 100) / (P * t) = interest_rate :=
by {
    sorry
}

end NUMINAMATH_GPT_find_interest_rate_l685_68554


namespace NUMINAMATH_GPT_solve_equation_l685_68584

theorem solve_equation : ∀ x : ℝ, x * (x + 2) = 3 * x + 6 ↔ (x = -2 ∨ x = 3) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l685_68584


namespace NUMINAMATH_GPT_functional_ineq_solution_l685_68507

theorem functional_ineq_solution (n : ℕ) (h : n > 0) :
  (∀ x : ℝ, n = 1 → (x^n + (1 - x)^n ≤ 1)) ∧
  (∀ x : ℝ, n > 1 → ((x < 0 ∨ x > 1) → (x^n + (1 - x)^n > 1))) :=
by
  intros
  sorry

end NUMINAMATH_GPT_functional_ineq_solution_l685_68507


namespace NUMINAMATH_GPT_maximum_members_in_dance_troupe_l685_68589

theorem maximum_members_in_dance_troupe (m : ℕ) (h1 : 25 * m % 31 = 7) (h2 : 25 * m < 1300) : 25 * m = 875 :=
by {
  sorry
}

end NUMINAMATH_GPT_maximum_members_in_dance_troupe_l685_68589


namespace NUMINAMATH_GPT_sid_initial_money_l685_68574

variable (M : ℝ)
variable (spent_on_accessories : ℝ := 12)
variable (spent_on_snacks : ℝ := 8)
variable (remaining_money_condition : ℝ := (M / 2) + 4)

theorem sid_initial_money : (M = 48) → (remaining_money_condition = M - (spent_on_accessories + spent_on_snacks)) :=
by
  sorry

end NUMINAMATH_GPT_sid_initial_money_l685_68574


namespace NUMINAMATH_GPT_area_of_region_l685_68561

theorem area_of_region : 
  (∃ (A : ℝ), A = 12 ∧ ∀ (x y : ℝ), |x| + |y| + |x - 2| ≤ 4 → 
    (0 ≤ y ∧ y ≤ 6 - 2*x ∧ x ≥ 2) ∨
    (0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ x ∧ x < 2) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ -1 ≤ x ∧ x < 0) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ x < -1)) :=
sorry

end NUMINAMATH_GPT_area_of_region_l685_68561


namespace NUMINAMATH_GPT_nat_pairs_satisfy_conditions_l685_68568

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end NUMINAMATH_GPT_nat_pairs_satisfy_conditions_l685_68568


namespace NUMINAMATH_GPT_ratio_of_third_layer_to_second_l685_68500

theorem ratio_of_third_layer_to_second (s1 s2 s3 : ℕ) (h1 : s1 = 2) (h2 : s2 = 2 * s1) (h3 : s3 = 12) : s3 / s2 = 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_third_layer_to_second_l685_68500


namespace NUMINAMATH_GPT_binom_9_5_eq_126_l685_68533

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end NUMINAMATH_GPT_binom_9_5_eq_126_l685_68533


namespace NUMINAMATH_GPT_perimeter_pentagon_l685_68527

noncomputable def AB : ℝ := 1
noncomputable def BC : ℝ := Real.sqrt 2
noncomputable def CD : ℝ := Real.sqrt 3
noncomputable def DE : ℝ := 2

noncomputable def AC : ℝ := Real.sqrt (AB^2 + BC^2)
noncomputable def AD : ℝ := Real.sqrt (AC^2 + CD^2)
noncomputable def AE : ℝ := Real.sqrt (AD^2 + DE^2)

theorem perimeter_pentagon (ABCDE : List ℝ) (H : ABCDE = [AB, BC, CD, DE, AE]) :
  List.sum ABCDE = 3 + Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 10 :=
by
  sorry -- Proof skipped as instructed

end NUMINAMATH_GPT_perimeter_pentagon_l685_68527


namespace NUMINAMATH_GPT_inv_113_mod_114_l685_68530

theorem inv_113_mod_114 :
  (113 * 113) % 114 = 1 % 114 :=
by
  sorry

end NUMINAMATH_GPT_inv_113_mod_114_l685_68530


namespace NUMINAMATH_GPT_find_number_l685_68588

theorem find_number 
  (m : ℤ)
  (h13 : m % 13 = 12)
  (h12 : m % 12 = 11)
  (h11 : m % 11 = 10)
  (h10 : m % 10 = 9)
  (h9 : m % 9 = 8)
  (h8 : m % 8 = 7)
  (h7 : m % 7 = 6)
  (h6 : m % 6 = 5)
  (h5 : m % 5 = 4)
  (h4 : m % 4 = 3)
  (h3 : m % 3 = 2) :
  m = 360359 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l685_68588


namespace NUMINAMATH_GPT_number_of_swaps_independent_l685_68598

theorem number_of_swaps_independent (n : ℕ) (hn : n = 20) (p : Fin n → Fin n) :
    (∀ i, p i ≠ i → ∃ j, p j ≠ j ∧ p (p j) = j) →
    ∃ s : List (Fin n × Fin n), List.length s ≤ n ∧
    (∀ σ : List (Fin n × Fin n), (∀ (i j : Fin n), (i, j) ∈ σ → p i ≠ i → ∃ p', σ = (i, p') :: (p', j) :: σ) →
     List.length σ = List.length s) :=
  sorry

end NUMINAMATH_GPT_number_of_swaps_independent_l685_68598


namespace NUMINAMATH_GPT_total_hike_time_l685_68562

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_total_hike_time_l685_68562


namespace NUMINAMATH_GPT_moles_of_HCl_needed_l685_68547

-- Define the reaction and corresponding stoichiometry
def reaction_relates (NaHSO3 HCl NaCl H2O SO2 : ℕ) : Prop :=
  NaHSO3 = HCl ∧ HCl = NaCl ∧ NaCl = H2O ∧ H2O = SO2

-- Given condition: one mole of each reactant produces one mole of each product
axiom reaction_stoichiometry : reaction_relates 1 1 1 1 1

-- Prove that 2 moles of NaHSO3 reacting with 2 moles of HCl forms 2 moles of NaCl
theorem moles_of_HCl_needed :
  ∀ (NaHSO3 HCl NaCl : ℕ), reaction_relates NaHSO3 HCl NaCl NaCl NaCl → NaCl = 2 → HCl = 2 :=
by
  intros NaHSO3 HCl NaCl h_eq h_NaCl
  sorry

end NUMINAMATH_GPT_moles_of_HCl_needed_l685_68547


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l685_68558

theorem geometric_sequence_fourth_term (a₁ a₂ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 1/3) :
    ∃ a₄ : ℝ, a₄ = 1/243 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l685_68558


namespace NUMINAMATH_GPT_map_length_represents_75_km_l685_68565
-- First, we broaden the import to bring in all the necessary libraries.

-- Define the conditions given in the problem.
def cm_to_km_ratio (cm : ℕ) (km : ℕ) : ℕ := km / cm

def map_represents (length_cm : ℕ) (length_km : ℕ) : Prop :=
  length_km = length_cm * cm_to_km_ratio 15 45

-- Rewrite the problem statement as a theorem in Lean 4.
theorem map_length_represents_75_km : map_represents 25 75 :=
by
  sorry

end NUMINAMATH_GPT_map_length_represents_75_km_l685_68565


namespace NUMINAMATH_GPT_chocolate_bar_pieces_l685_68569

theorem chocolate_bar_pieces (X : ℕ) (h1 : X / 2 + X / 4 + 15 = X) : X = 60 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bar_pieces_l685_68569


namespace NUMINAMATH_GPT_find_DF_l685_68571

-- Conditions
variables {A B C D E F : Type}
variables {BC EF AC DF : ℝ}
variable (h_similar : similar_triangles A B C D E F)
variable (h_BC : BC = 6)
variable (h_EF : EF = 4)
variable (h_AC : AC = 9)

-- Question: Prove DF = 6 given the above conditions
theorem find_DF : DF = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_DF_l685_68571


namespace NUMINAMATH_GPT_unit_digit_is_nine_l685_68523

theorem unit_digit_is_nine (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : a ≠ 0) (h4 : a + b + a * b = 10 * a + b) : b = 9 := 
by 
  sorry

end NUMINAMATH_GPT_unit_digit_is_nine_l685_68523


namespace NUMINAMATH_GPT_lawn_width_l685_68529

variable (W : ℝ)
variable (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875)
variable (h₂ : 5625 = 3 * 1875)

theorem lawn_width (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875) (h₂ : 5625 = 3 * 1875) : 
  W = 60 := 
sorry

end NUMINAMATH_GPT_lawn_width_l685_68529


namespace NUMINAMATH_GPT_find_y_l685_68521

noncomputable def G (a b c d : ℝ) : ℝ := a ^ b + c ^ d

theorem find_y (h : G 3 y 2 5 = 100) : y = Real.log 68 / Real.log 3 := 
by
  have hG : G 3 y 2 5 = 3 ^ y + 2 ^ 5 := rfl
  sorry

end NUMINAMATH_GPT_find_y_l685_68521


namespace NUMINAMATH_GPT_inverse_proportion_point_passes_through_l685_68597

theorem inverse_proportion_point_passes_through
  (m : ℝ) (h1 : (4, 6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst})
  : (-4, -6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst} :=
sorry

end NUMINAMATH_GPT_inverse_proportion_point_passes_through_l685_68597


namespace NUMINAMATH_GPT_no_real_solution_l685_68570

theorem no_real_solution (x y : ℝ) (h: y = 3 * x - 1) : ¬ (4 * y ^ 2 + y + 3 = 3 * (8 * x ^ 2 + 3 * y + 1)) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_l685_68570


namespace NUMINAMATH_GPT_veronica_pitting_time_is_2_hours_l685_68510

def veronica_cherries_pitting_time (pounds : ℕ) (cherries_per_pound : ℕ) (minutes_per_20_cherries : ℕ) :=
  let cherries := pounds * cherries_per_pound
  let sets := cherries / 20
  let total_minutes := sets * minutes_per_20_cherries
  total_minutes / 60

theorem veronica_pitting_time_is_2_hours : 
  veronica_cherries_pitting_time 3 80 10 = 2 :=
  by
    sorry

end NUMINAMATH_GPT_veronica_pitting_time_is_2_hours_l685_68510


namespace NUMINAMATH_GPT_deepak_age_l685_68586

variable (R D : ℕ)

theorem deepak_age (h1 : R / D = 4 / 3) (h2 : R + 6 = 26) : D = 15 :=
sorry

end NUMINAMATH_GPT_deepak_age_l685_68586


namespace NUMINAMATH_GPT_possible_m_value_l685_68546

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3 - (1/2)*x - 1
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^3 + m / x

theorem possible_m_value :
  ∃ m : ℝ, (m = (1/2) - (1/Real.exp 1)) ∧
    (∀ x1 x2 : ℝ, 
      (f x1 = g (-x1) m) →
      (f x2 = g (-x2) m) →
      x1 ≠ 0 ∧ x2 ≠ 0 ∧
      m = x1 * Real.exp x1 - (1/2) * x1^2 - x1 ∧
      m = x2 * Real.exp x2 - (1/2) * x2^2 - x2) :=
by
  sorry

end NUMINAMATH_GPT_possible_m_value_l685_68546


namespace NUMINAMATH_GPT_jonas_tshirts_count_l685_68517

def pairs_to_individuals (pairs : Nat) : Nat := pairs * 2

variable (num_pairs_socks : Nat := 20)
variable (num_pairs_shoes : Nat := 5)
variable (num_pairs_pants : Nat := 10)
variable (num_additional_pairs_socks : Nat := 35)

def total_individual_items_without_tshirts : Nat :=
  pairs_to_individuals num_pairs_socks +
  pairs_to_individuals num_pairs_shoes +
  pairs_to_individuals num_pairs_pants

def total_individual_items_desired : Nat :=
  total_individual_items_without_tshirts +
  pairs_to_individuals num_additional_pairs_socks

def tshirts_jonas_needs : Nat :=
  total_individual_items_desired - total_individual_items_without_tshirts

theorem jonas_tshirts_count : tshirts_jonas_needs = 70 := by
  sorry

end NUMINAMATH_GPT_jonas_tshirts_count_l685_68517


namespace NUMINAMATH_GPT_probability_of_roll_6_after_E_l685_68519

/- Darryl has a six-sided die with faces 1, 2, 3, 4, 5, 6.
   The die is weighted so that one face comes up with probability 1/2,
   and the other five faces have equal probability.
   Darryl does not know which side is weighted, but each face is equally likely to be the weighted one.
   Darryl rolls the die 5 times and gets a 1, 2, 3, 4, and 5 in some unspecified order. -/

def probability_of_next_roll_getting_6 : ℚ :=
  let p_weighted := (1 / 2 : ℚ)
  let p_unweighted := (1 / 10 : ℚ)
  let p_w6_given_E := (1 / 26 : ℚ)
  let p_not_w6_given_E := (25 / 26 : ℚ)
  p_w6_given_E * p_weighted + p_not_w6_given_E * p_unweighted

theorem probability_of_roll_6_after_E : probability_of_next_roll_getting_6 = 3 / 26 := sorry

end NUMINAMATH_GPT_probability_of_roll_6_after_E_l685_68519


namespace NUMINAMATH_GPT_complement_intersection_subset_condition_l685_68520

-- Definition of sets A, B, and C
def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }
def C (a : ℝ) := { x : ℝ | x < a }

-- Proof problem 1 statement
theorem complement_intersection :
  ( { x : ℝ | x < 3 ∨ x ≥ 7 } ∩ { x : ℝ | 2 < x ∧ x < 10 } ) = { x : ℝ | 2 < x ∧ x < 3 ∨ 7 ≤ x ∧ x < 10 } :=
by
  sorry

-- Proof problem 2 statement
theorem subset_condition (a : ℝ) :
  ( { x : ℝ | 3 ≤ x ∧ x < 7 } ⊆ { x : ℝ | x < a } ) → (a ≥ 7) :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_subset_condition_l685_68520


namespace NUMINAMATH_GPT_passing_time_for_platform_l685_68582

def train_length : ℕ := 1100
def time_to_cross_tree : ℕ := 110
def platform_length : ℕ := 700
def speed := train_length / time_to_cross_tree
def combined_length := train_length + platform_length

theorem passing_time_for_platform : 
  let speed := train_length / time_to_cross_tree
  let combined_length := train_length + platform_length
  combined_length / speed = 180 :=
by
  sorry

end NUMINAMATH_GPT_passing_time_for_platform_l685_68582


namespace NUMINAMATH_GPT_problem1_problem2_l685_68555

noncomputable def interval1 (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}
noncomputable def interval2 : Set ℝ := {x | x < -1 ∨ x > 3}

theorem problem1 (a : ℝ) : (interval1 a ∩ interval2 = interval1 a) ↔ a ∈ {x | x ≤ -2} ∪ {x | 1 ≤ x} := by sorry

theorem problem2 (a : ℝ) : (interval1 a ∩ interval2 ≠ ∅) ↔ a < -1 / 2 := by sorry

end NUMINAMATH_GPT_problem1_problem2_l685_68555


namespace NUMINAMATH_GPT_seunghyeon_pizza_diff_l685_68528

theorem seunghyeon_pizza_diff (S Y : ℕ) (h : S - 2 = Y + 7) : S - Y = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_seunghyeon_pizza_diff_l685_68528


namespace NUMINAMATH_GPT_trapezoid_area_l685_68594

theorem trapezoid_area (EF GH h : ℕ) (hEF : EF = 60) (hGH : GH = 30) (hh : h = 15) : 
  (EF + GH) * h / 2 = 675 := by 
  sorry

end NUMINAMATH_GPT_trapezoid_area_l685_68594


namespace NUMINAMATH_GPT_find_angle_D_l685_68505

variables (A B C D angle : ℝ)

-- Assumptions based on the problem statement
axiom sum_A_B : A + B = 140
axiom C_eq_D : C = D

-- The claim we aim to prove
theorem find_angle_D (h₁ : A + B = 140) (h₂: C = D): D = 20 :=
by {
    sorry 
}

end NUMINAMATH_GPT_find_angle_D_l685_68505


namespace NUMINAMATH_GPT_added_number_is_five_l685_68556

variable (n x : ℤ)

theorem added_number_is_five (h1 : n % 25 = 4) (h2 : (n + x) % 5 = 4) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_added_number_is_five_l685_68556
