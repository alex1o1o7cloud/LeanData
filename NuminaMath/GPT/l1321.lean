import Mathlib

namespace NUMINAMATH_GPT_find_f_pi_over_4_l1321_132191

variable (f : ℝ → ℝ)
variable (h : ∀ x, f x = f (Real.pi / 4) * Real.cos x + Real.sin x)

theorem find_f_pi_over_4 : f (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_find_f_pi_over_4_l1321_132191


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1321_132112

noncomputable def simplified_expression (x : ℝ) : ℝ :=
  ((1 / (x - 1)) + (1 / (x + 1))) / (x^2 / (3 * x^2 - 3))

theorem simplify_and_evaluate : simplified_expression (Real.sqrt 2) = 3 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1321_132112


namespace NUMINAMATH_GPT_number_of_initials_is_10000_l1321_132150

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end NUMINAMATH_GPT_number_of_initials_is_10000_l1321_132150


namespace NUMINAMATH_GPT_greatest_k_for_200k_divides_100_factorial_l1321_132172

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_k_for_200k_divides_100_factorial :
  let x := factorial 100
  let k_max := 12
  ∃ k : ℕ, y = 200 ^ k ∧ y ∣ x ∧ k = k_max :=
sorry

end NUMINAMATH_GPT_greatest_k_for_200k_divides_100_factorial_l1321_132172


namespace NUMINAMATH_GPT_original_combined_price_l1321_132164

theorem original_combined_price (C S : ℝ)
  (hC_new : (C + 0.25 * C) = 12.5)
  (hS_new : (S + 0.50 * S) = 13.5) :
  (C + S) = 19 := by
  -- sorry makes sure to skip the proof
  sorry

end NUMINAMATH_GPT_original_combined_price_l1321_132164


namespace NUMINAMATH_GPT_one_minus_repeating_decimal_three_equals_two_thirds_l1321_132114

-- Define the repeating decimal as a fraction
def repeating_decimal_three : ℚ := 1 / 3

-- Prove the desired equality
theorem one_minus_repeating_decimal_three_equals_two_thirds :
  1 - repeating_decimal_three = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_one_minus_repeating_decimal_three_equals_two_thirds_l1321_132114


namespace NUMINAMATH_GPT_least_positive_integer_solution_l1321_132132

theorem least_positive_integer_solution :
  ∃ x : ℕ, (x + 7391) % 12 = 167 % 12 ∧ x = 8 :=
by 
  sorry

end NUMINAMATH_GPT_least_positive_integer_solution_l1321_132132


namespace NUMINAMATH_GPT_horner_method_complexity_l1321_132146

variable {α : Type*} [Field α]

/-- Evaluating a polynomial of degree n using Horner's method requires exactly n multiplications
    and n additions, and 0 exponentiations.  -/
theorem horner_method_complexity (n : ℕ) (a : Fin (n + 1) → α) (x₀ : α) :
  ∃ (muls adds exps : ℕ), 
    (muls = n) ∧ (adds = n) ∧ (exps = 0) :=
by
  sorry

end NUMINAMATH_GPT_horner_method_complexity_l1321_132146


namespace NUMINAMATH_GPT_correct_card_ordering_l1321_132137

structure CardOrder where
  left : String
  middle : String
  right : String

def is_right_of (a b : String) : Prop := (a = "club" ∧ (b = "heart" ∨ b = "diamond")) ∨ (a = "8" ∧ b = "4")

def is_left_of (a b : String) : Prop := a = "5" ∧ b = "heart"

def correct_order : CardOrder :=
  { left := "5 of diamonds", middle := "4 of hearts", right := "8 of clubs" }

theorem correct_card_ordering : 
  ∀ order : CardOrder, 
  is_right_of order.right order.middle ∧ is_right_of order.right order.left ∧ is_left_of order.left order.middle 
  → order = correct_order := 
by
  intro order
  intro h
  sorry

end NUMINAMATH_GPT_correct_card_ordering_l1321_132137


namespace NUMINAMATH_GPT_problem_statement_l1321_132151

variable (g : ℝ)

-- Definition of the operation
def my_op (g y : ℝ) : ℝ := g^2 + 2 * y

-- The statement we want to prove
theorem problem_statement : my_op g (my_op g g) = g^4 + 4 * g^3 + 6 * g^2 + 4 * g :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1321_132151


namespace NUMINAMATH_GPT_sum_first_15_odd_integers_l1321_132135

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_15_odd_integers_l1321_132135


namespace NUMINAMATH_GPT_people_per_column_in_second_arrangement_l1321_132167
-- Import the necessary libraries

-- Define the conditions as given in the problem
def number_of_people_first_arrangement : ℕ := 30 * 16
def number_of_columns_second_arrangement : ℕ := 8

-- Define the problem statement with proof
theorem people_per_column_in_second_arrangement :
  (number_of_people_first_arrangement / number_of_columns_second_arrangement) = 60 :=
by
  -- Skip the proof here
  sorry

end NUMINAMATH_GPT_people_per_column_in_second_arrangement_l1321_132167


namespace NUMINAMATH_GPT_least_multiple_of_13_gt_450_l1321_132179

theorem least_multiple_of_13_gt_450 : ∃ (n : ℕ), (455 = 13 * n) ∧ 455 > 450 ∧ ∀ m : ℕ, (13 * m > 450) → 455 ≤ 13 * m :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_of_13_gt_450_l1321_132179


namespace NUMINAMATH_GPT_seq_sum_terms_l1321_132190

def S (n : ℕ) : ℕ := 3^n - 2

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * 3^(n - 1)

theorem seq_sum_terms (n : ℕ) : 
  a n = if n = 1 then 1 else 2 * 3^(n-1) :=
sorry

end NUMINAMATH_GPT_seq_sum_terms_l1321_132190


namespace NUMINAMATH_GPT_factorization_a_squared_minus_3a_l1321_132176

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3 * a = a * (a - 3) := 
by 
  sorry

end NUMINAMATH_GPT_factorization_a_squared_minus_3a_l1321_132176


namespace NUMINAMATH_GPT_translation_equivalence_l1321_132178

def f₁ (x : ℝ) : ℝ := 4 * (x + 3)^2 - 4
def f₂ (x : ℝ) : ℝ := 4 * (x - 3)^2 + 4

theorem translation_equivalence :
  (∀ x : ℝ, f₁ (x + 6) = 4 * (x + 9)^2 + 4) ∧
  (∀ x : ℝ, f₁ x  - 8 = 4 * (x + 3)^2 - 4) :=
by sorry

end NUMINAMATH_GPT_translation_equivalence_l1321_132178


namespace NUMINAMATH_GPT_eval_expression_correct_l1321_132149

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end NUMINAMATH_GPT_eval_expression_correct_l1321_132149


namespace NUMINAMATH_GPT_bella_total_roses_l1321_132171

-- Define the constants and conditions
def dozen := 12
def roses_from_parents := 2 * dozen
def friends := 10
def roses_per_friend := 2
def total_roses := roses_from_parents + (roses_per_friend * friends)

-- Prove that the total number of roses Bella received is 44
theorem bella_total_roses : total_roses = 44 := 
by
  sorry

end NUMINAMATH_GPT_bella_total_roses_l1321_132171


namespace NUMINAMATH_GPT_exist_indices_for_sequences_l1321_132101

open Nat

theorem exist_indices_for_sequences 
  (a b c : ℕ → ℕ) : 
  ∃ p q, p ≠ q ∧ p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
  sorry

end NUMINAMATH_GPT_exist_indices_for_sequences_l1321_132101


namespace NUMINAMATH_GPT_right_triangle_medians_right_triangle_l1321_132195

theorem right_triangle_medians_right_triangle (a b c s_a s_b s_c : ℝ)
  (hyp_a_lt_b : a < b) (hyp_b_lt_c : b < c)
  (h_c_hypotenuse : c = Real.sqrt (a^2 + b^2))
  (h_sa : s_a^2 = b^2 + (a / 2)^2)
  (h_sb : s_b^2 = a^2 + (b / 2)^2)
  (h_sc : s_c^2 = (a^2 + b^2) / 4) :
  b = a * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_medians_right_triangle_l1321_132195


namespace NUMINAMATH_GPT_subtracting_is_adding_opposite_l1321_132117

theorem subtracting_is_adding_opposite (a b : ℚ) : a - b = a + (-b) :=
by sorry

end NUMINAMATH_GPT_subtracting_is_adding_opposite_l1321_132117


namespace NUMINAMATH_GPT_fencing_cost_approx_122_52_l1321_132110

noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d

noncomputable def fencing_cost (d rate : ℝ) : ℝ := circumference d * rate

theorem fencing_cost_approx_122_52 :
  let d := 26
  let rate := 1.50
  abs (fencing_cost d rate - 122.52) < 1 :=
by
  let d : ℝ := 26
  let rate : ℝ := 1.50
  let cost := fencing_cost d rate
  sorry

end NUMINAMATH_GPT_fencing_cost_approx_122_52_l1321_132110


namespace NUMINAMATH_GPT_sum_of_letters_l1321_132116

def A : ℕ := 0
def B : ℕ := 1
def C : ℕ := 2
def M : ℕ := 12

theorem sum_of_letters :
  A + B + M + C = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_letters_l1321_132116


namespace NUMINAMATH_GPT_lottery_ticket_random_event_l1321_132109

-- Define the type of possible outcomes of buying a lottery ticket
inductive LotteryOutcome
| Win
| Lose

-- Define the random event condition
def is_random_event (outcome: LotteryOutcome) : Prop :=
  match outcome with
  | LotteryOutcome.Win => True
  | LotteryOutcome.Lose => True

-- The theorem to prove that buying 1 lottery ticket and winning is a random event
theorem lottery_ticket_random_event : is_random_event LotteryOutcome.Win :=
by
  sorry

end NUMINAMATH_GPT_lottery_ticket_random_event_l1321_132109


namespace NUMINAMATH_GPT_compare_negative_fractions_l1321_132136

theorem compare_negative_fractions :
  (-5 : ℝ) / 6 < (-4 : ℝ) / 5 :=
sorry

end NUMINAMATH_GPT_compare_negative_fractions_l1321_132136


namespace NUMINAMATH_GPT_product_of_two_numbers_ratio_l1321_132188

theorem product_of_two_numbers_ratio {x y : ℝ}
  (h1 : x + y = (5/3) * (x - y))
  (h2 : x * y = 5 * (x - y)) :
  x * y = 56.25 := sorry

end NUMINAMATH_GPT_product_of_two_numbers_ratio_l1321_132188


namespace NUMINAMATH_GPT_haleys_current_height_l1321_132161

-- Define the conditions
def growth_rate : ℕ := 3
def years : ℕ := 10
def future_height : ℕ := 50

-- Define the proof problem
theorem haleys_current_height : (future_height - growth_rate * years) = 20 :=
by {
  -- This is where the actual proof would go
  sorry
}

end NUMINAMATH_GPT_haleys_current_height_l1321_132161


namespace NUMINAMATH_GPT_quadratic_inequality_l1321_132186

-- Define the quadratic function and conditions
variables {a b c x0 y1 y2 y3 : ℝ}
variables (A : (a * x0^2 + b * x0 + c = 0))
variables (B : (a * (-2)^2 + b * (-2) + c = 0))
variables (C : (a + b + c) * (4 * a + 2 * b + c) < 0)
variables (D : a > 0)
variables (E1 : y1 = a * (-1)^2 + b * (-1) + c)
variables (E2 : y2 = a * (- (sqrt 2) / 2)^2 + b * (- (sqrt 2) / 2) + c)
variables (E3 : y3 = a * 1^2 + b * 1 + c)

-- Prove that y3 > y1 > y2
theorem quadratic_inequality : y3 > y1 ∧ y1 > y2 := by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l1321_132186


namespace NUMINAMATH_GPT_sally_bread_consumption_l1321_132185

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end NUMINAMATH_GPT_sally_bread_consumption_l1321_132185


namespace NUMINAMATH_GPT_store_profit_in_february_l1321_132183

variable (C : ℝ)

def initialSellingPrice := C * 1.20
def secondSellingPrice := initialSellingPrice C * 1.25
def finalSellingPrice := secondSellingPrice C * 0.88

theorem store_profit_in_february
  (initialSellingPrice_eq : initialSellingPrice C = C * 1.20)
  (secondSellingPrice_eq : secondSellingPrice C = initialSellingPrice C * 1.25)
  (finalSellingPrice_eq : finalSellingPrice C = secondSellingPrice C * 0.88)
  : finalSellingPrice C - C = 0.32 * C :=
sorry

end NUMINAMATH_GPT_store_profit_in_february_l1321_132183


namespace NUMINAMATH_GPT_find_three_digit_number_l1321_132111

theorem find_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
    (x - 6) % 7 = 0 ∧
    (x - 7) % 8 = 0 ∧
    (x - 8) % 9 = 0 ∧
    x = 503 :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l1321_132111


namespace NUMINAMATH_GPT_total_value_of_treats_l1321_132175

def hotel_cost_per_night : ℕ := 4000
def number_of_nights : ℕ := 2
def car_cost : ℕ := 30000
def house_multiplier : ℕ := 4

theorem total_value_of_treats : 
  (number_of_nights * hotel_cost_per_night) + car_cost + (house_multiplier * car_cost) = 158000 := 
by
  sorry

end NUMINAMATH_GPT_total_value_of_treats_l1321_132175


namespace NUMINAMATH_GPT_sum_S_15_22_31_l1321_132174

-- Define the sequence \{a_n\} with the sum of the first n terms S_n
def S : ℕ → ℤ
| 0 => 0
| n + 1 => S n + (-1: ℤ)^n * (4 * (n + 1) - 3)

-- The statement to prove: S_{15} + S_{22} - S_{31} = -76
theorem sum_S_15_22_31 : S 15 + S 22 - S 31 = -76 :=
sorry

end NUMINAMATH_GPT_sum_S_15_22_31_l1321_132174


namespace NUMINAMATH_GPT_most_suitable_for_comprehensive_survey_l1321_132131

-- Definitions of the survey options
inductive SurveyOption
| A
| B
| C
| D

-- Condition definitions based on the problem statement
def comprehensive_survey (option : SurveyOption) : Prop :=
  option = SurveyOption.B

-- The theorem stating that the most suitable survey is option B
theorem most_suitable_for_comprehensive_survey : ∀ (option : SurveyOption), comprehensive_survey option ↔ option = SurveyOption.B :=
by
  intro option
  sorry

end NUMINAMATH_GPT_most_suitable_for_comprehensive_survey_l1321_132131


namespace NUMINAMATH_GPT_simplify_polynomial_l1321_132169

theorem simplify_polynomial (x : ℝ) : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2) = (-x^2 + 23 * x - 3) := 
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1321_132169


namespace NUMINAMATH_GPT_sum_of_corners_is_164_l1321_132129

section CheckerboardSum

-- Define the total number of elements in the 9x9 grid
def num_elements := 81

-- Define the positions of the corners
def top_left : ℕ := 1
def top_right : ℕ := 9
def bottom_left : ℕ := 73
def bottom_right : ℕ := 81

-- Define the sum of the corners
def corner_sum : ℕ := top_left + top_right + bottom_left + bottom_right

-- State the theorem
theorem sum_of_corners_is_164 : corner_sum = 164 :=
by
  exact sorry

end CheckerboardSum

end NUMINAMATH_GPT_sum_of_corners_is_164_l1321_132129


namespace NUMINAMATH_GPT_average_price_per_book_l1321_132182

theorem average_price_per_book
  (amount_spent_first_shop : ℕ)
  (amount_spent_second_shop : ℕ)
  (books_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_amount_spent : ℕ := amount_spent_first_shop + amount_spent_second_shop)
  (total_books_bought : ℕ := books_first_shop + books_second_shop)
  (average_price : ℕ := total_amount_spent / total_books_bought) :
  amount_spent_first_shop = 520 → amount_spent_second_shop = 248 →
  books_first_shop = 42 → books_second_shop = 22 →
  average_price = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_price_per_book_l1321_132182


namespace NUMINAMATH_GPT_max_value_ln_x_plus_x_l1321_132141

theorem max_value_ln_x_plus_x (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ Real.exp 1) : 
  ∃ y, y = Real.log x + x ∧ y ≤ Real.log (Real.exp 1) + Real.exp 1 :=
sorry

end NUMINAMATH_GPT_max_value_ln_x_plus_x_l1321_132141


namespace NUMINAMATH_GPT_find_number_l1321_132145

variable (number : ℤ)

theorem find_number (h : number - 44 = 15) : number = 59 := 
sorry

end NUMINAMATH_GPT_find_number_l1321_132145


namespace NUMINAMATH_GPT_b2009_value_l1321_132194

noncomputable def b (n : ℕ) : ℝ := sorry

axiom b_recursion (n : ℕ) (hn : 2 ≤ n) : b n = b (n - 1) * b (n + 1)

axiom b1_value : b 1 = 2 + Real.sqrt 3
axiom b1776_value : b 1776 = 10 + Real.sqrt 3

theorem b2009_value : b 2009 = -4 + 8 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_b2009_value_l1321_132194


namespace NUMINAMATH_GPT_line_intersects_ellipse_max_chord_length_l1321_132160

theorem line_intersects_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1)) ↔ 
  (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 3 * Real.sqrt 2) := 
by sorry

theorem max_chord_length : 
  (∃ m : ℝ, (m = 0) ∧ 
    (∀ x y x1 y1 : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) ∧ 
     (y1 = (3/2 : ℝ) * x1 + m) ∧ (x1^2 / 4 + y1^2 / 9 = 1) ∧ 
     (x ≠ x1 ∨ y ≠ y1) → 
     (Real.sqrt (13 / 9) * Real.sqrt (18 - m^2) = Real.sqrt 26))) := 
by sorry

end NUMINAMATH_GPT_line_intersects_ellipse_max_chord_length_l1321_132160


namespace NUMINAMATH_GPT_students_only_biology_students_biology_or_chemistry_but_not_both_l1321_132102

def students_enrolled_in_both : ℕ := 15
def total_biology_students : ℕ := 35
def students_only_chemistry : ℕ := 18

theorem students_only_biology (h₀ : students_enrolled_in_both ≤ total_biology_students) :
  total_biology_students - students_enrolled_in_both = 20 := by
  sorry

theorem students_biology_or_chemistry_but_not_both :
  total_biology_students - students_enrolled_in_both + students_only_chemistry = 38 := by
  sorry

end NUMINAMATH_GPT_students_only_biology_students_biology_or_chemistry_but_not_both_l1321_132102


namespace NUMINAMATH_GPT_price_reduction_equation_l1321_132128

theorem price_reduction_equation (x : ℝ) : 25 * (1 - x)^2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_equation_l1321_132128


namespace NUMINAMATH_GPT_arithmetic_sequence_equality_l1321_132181

theorem arithmetic_sequence_equality {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (a20 : a ≠ c) (a2012 : b ≠ c) 
(h₄ : ∀ (i : ℕ), ∃ d : ℝ, a_n = a + i * d) : 
  1992 * a * c - 1811 * b * c - 181 * a * b = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_equality_l1321_132181


namespace NUMINAMATH_GPT_jack_pays_back_l1321_132166

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end NUMINAMATH_GPT_jack_pays_back_l1321_132166


namespace NUMINAMATH_GPT_expand_polynomials_l1321_132163

def p (z : ℤ) := 3 * z^3 + 4 * z^2 - 2 * z + 1
def q (z : ℤ) := 2 * z^2 - 3 * z + 5
def r (z : ℤ) := 10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5

theorem expand_polynomials (z : ℤ) : (p z) * (q z) = r z :=
by sorry

end NUMINAMATH_GPT_expand_polynomials_l1321_132163


namespace NUMINAMATH_GPT_reach_any_natural_number_l1321_132134

theorem reach_any_natural_number (n : ℕ) : ∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = 3 * f k + 1 ∨ f (k + 1) = f k / 2) ∧ (∃ m, f m = n) := by
  sorry

end NUMINAMATH_GPT_reach_any_natural_number_l1321_132134


namespace NUMINAMATH_GPT_molecular_weight_3_moles_l1321_132105

theorem molecular_weight_3_moles
  (C_weight : ℝ)
  (H_weight : ℝ)
  (N_weight : ℝ)
  (O_weight : ℝ)
  (Molecular_formula : ℕ → ℕ → ℕ → ℕ → Prop)
  (molecular_weight : ℝ)
  (moles : ℝ) :
  C_weight = 12.01 →
  H_weight = 1.008 →
  N_weight = 14.01 →
  O_weight = 16.00 →
  Molecular_formula 13 9 5 7 →
  molecular_weight = 156.13 + 9.072 + 70.05 + 112.00 →
  moles = 3 →
  3 * molecular_weight = 1041.756 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_3_moles_l1321_132105


namespace NUMINAMATH_GPT_age_difference_l1321_132153

-- Defining the necessary variables and their types
variables (A B : ℕ)

-- Given conditions: 
axiom B_current_age : B = 38
axiom future_age_relationship : A + 10 = 2 * (B - 10)

-- Proof goal statement
theorem age_difference : A - B = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1321_132153


namespace NUMINAMATH_GPT_value_of_v_over_u_l1321_132113

variable (u v : ℝ) 

theorem value_of_v_over_u (h : u - v = (u + v) / 2) : v / u = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_v_over_u_l1321_132113


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l1321_132118

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence
variable {d : ℝ} -- Define the common difference
variable {a1 : ℝ} -- Define the first term

-- Suppose the sum of the first 17 terms equals 306
axiom h1 : S 17 = 306
-- Suppose the sum of the first n terms of an arithmetic sequence formula
axiom sum_formula : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
-- Suppose the relation between the first term, common difference and sum of the first 17 terms
axiom relation : a1 + 8 * d = 18 

theorem arithmetic_sequence_property : a 7 - (a 3) / 3 = 12 := 
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l1321_132118


namespace NUMINAMATH_GPT_winning_jackpot_is_event_l1321_132143

-- Definitions based on the conditions
def has_conditions (experiment : String) : Prop :=
  experiment = "A" ∨ experiment = "B" ∨ experiment = "C" ∨ experiment = "D"

def has_outcomes (experiment : String) : Prop :=
  experiment = "D"

def is_event (experiment : String) : Prop :=
  has_conditions experiment ∧ has_outcomes experiment

-- Statement to prove
theorem winning_jackpot_is_event : is_event "D" :=
by
  -- Trivial step to show that D meets both conditions and outcomes
  exact sorry

end NUMINAMATH_GPT_winning_jackpot_is_event_l1321_132143


namespace NUMINAMATH_GPT_max_chords_intersecting_line_l1321_132139

theorem max_chords_intersecting_line (A : Fin 2017 → Type) :
  ∃ k : ℕ, (k ≤ 2016 ∧ ∃ m : ℕ, (m = k * (2016 - k) + 2016) ∧ m = 1018080) :=
sorry

end NUMINAMATH_GPT_max_chords_intersecting_line_l1321_132139


namespace NUMINAMATH_GPT_intersection_A_B_l1321_132148

section
  def A : Set ℤ := {-2, 0, 1}
  def B : Set ℤ := {x | x^2 > 1}
  theorem intersection_A_B : A ∩ B = {-2} := 
  by
    sorry
end

end NUMINAMATH_GPT_intersection_A_B_l1321_132148


namespace NUMINAMATH_GPT_maxAdditionalTiles_l1321_132187

-- Board definition
structure Board where
  width : Nat
  height : Nat
  cells : List (Nat × Nat) -- List of cells occupied by tiles

def initialBoard : Board := 
  ⟨10, 9, [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2),
            (6,1), (6,2), (7,1), (7,2)]⟩

-- Function to count cells occupied
def occupiedCells (b : Board) : Nat :=
  b.cells.length

-- Function to calculate total cells in a board
def totalCells (b : Board) : Nat :=
  b.width * b.height

-- Function to calculate additional 2x1 tiles that can be placed
def additionalTiles (board : Board) : Nat :=
  (totalCells board - occupiedCells board) / 2

theorem maxAdditionalTiles : additionalTiles initialBoard = 36 := by
  sorry

end NUMINAMATH_GPT_maxAdditionalTiles_l1321_132187


namespace NUMINAMATH_GPT_prob_at_least_one_palindrome_correct_l1321_132130

-- Define a function to represent the probability calculation.
def probability_at_least_one_palindrome : ℚ :=
  let prob_digit_palindrome : ℚ := 1 / 100
  let prob_letter_palindrome : ℚ := 1 / 676
  let prob_both_palindromes : ℚ := (1 / 100) * (1 / 676)
  (prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes)

-- The theorem we are stating based on the given problem and solution:
theorem prob_at_least_one_palindrome_correct : probability_at_least_one_palindrome = 427 / 2704 :=
by
  -- We assume this step for now as we are just stating the theorem
  sorry

end NUMINAMATH_GPT_prob_at_least_one_palindrome_correct_l1321_132130


namespace NUMINAMATH_GPT_second_number_is_255_l1321_132198

theorem second_number_is_255 (x : ℝ) (n : ℝ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
  (h2 : (128 + n + 511 + 1023 + x) / 5 = 423) : 
  n = 255 :=
sorry

end NUMINAMATH_GPT_second_number_is_255_l1321_132198


namespace NUMINAMATH_GPT_cannot_finish_third_l1321_132120

-- Define the racers
inductive Racer
| P | Q | R | S | T | U
open Racer

-- Define the conditions
def beats (a b : Racer) : Prop := sorry  -- placeholder for strict order
def ties (a b : Racer) : Prop := sorry   -- placeholder for tie condition
def position (r : Racer) (p : Fin (6)) : Prop := sorry  -- placeholder for position in the race

theorem cannot_finish_third :
  (beats P Q) ∧
  (ties P R) ∧
  (beats Q S) ∧
  ∃ p₁ p₂ p₃, position P p₁ ∧ position T p₂ ∧ position Q p₃ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧
  ∃ p₄ p₅, position U p₄ ∧ position S p₅ ∧ p₄ < p₅ →
  ¬ position P (3 : Fin (6)) ∧ ¬ position U (3 : Fin (6)) ∧ ¬ position S (3 : Fin (6)) :=
by sorry   -- Proof is omitted

end NUMINAMATH_GPT_cannot_finish_third_l1321_132120


namespace NUMINAMATH_GPT_exists_x_gg_eq_3_l1321_132126

noncomputable def g (x : ℝ) : ℝ :=
if x < -3 then -0.5 * x^2 + 3
else if x < 2 then 1
else 0.5 * x^2 - 1.5 * x + 3

theorem exists_x_gg_eq_3 : ∃ x : ℝ, x = -5 ∨ x = 5 ∧ g (g x) = 3 :=
by
  sorry

end NUMINAMATH_GPT_exists_x_gg_eq_3_l1321_132126


namespace NUMINAMATH_GPT_translation_correct_l1321_132127

-- Define the first line l1
def l1 (x : ℝ) : ℝ := 2 * x - 2

-- Define the second line l2
def l2 (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem translation_correct :
  ∀ x : ℝ, l2 x = l1 x + 2 :=
by
  intro x
  unfold l1 l2
  sorry

end NUMINAMATH_GPT_translation_correct_l1321_132127


namespace NUMINAMATH_GPT_find_sum_12_terms_of_sequence_l1321_132168

variable {a : ℕ → ℕ}

def geometric_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

def is_periodic_sequence (a : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n : ℕ, a n = a (n + period)

noncomputable def given_sequence : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => (given_sequence n * given_sequence (n + 1) / 4) -- This should ensure periodic sequence of period 3 given a common product of 8 and simplifying the product equation.

theorem find_sum_12_terms_of_sequence :
  geometric_sequence given_sequence 8 ∧ given_sequence 0 = 1 ∧ given_sequence 1 = 2 →
  (Finset.range 12).sum given_sequence = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_12_terms_of_sequence_l1321_132168


namespace NUMINAMATH_GPT_perpendicular_lines_l1321_132133

theorem perpendicular_lines (a : ℝ) : 
  (∀ (x y : ℝ), (1 - 2 * a) * x - 2 * y + 3 = 0 → 3 * x + y + 2 * a = 0) → 
  a = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1321_132133


namespace NUMINAMATH_GPT_total_workers_is_28_l1321_132192

noncomputable def avg_salary_total : ℝ := 750
noncomputable def num_type_a : ℕ := 5
noncomputable def avg_salary_type_a : ℝ := 900
noncomputable def num_type_b : ℕ := 4
noncomputable def avg_salary_type_b : ℝ := 800
noncomputable def avg_salary_type_c : ℝ := 700

theorem total_workers_is_28 :
  ∃ (W : ℕ) (C : ℕ),
  W * avg_salary_total = num_type_a * avg_salary_type_a + num_type_b * avg_salary_type_b + C * avg_salary_type_c ∧
  W = num_type_a + num_type_b + C ∧
  W = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_workers_is_28_l1321_132192


namespace NUMINAMATH_GPT_number_of_true_statements_l1321_132124

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem number_of_true_statements (n : ℕ) :
  let s1 := reciprocal 4 + reciprocal 8 ≠ reciprocal 12
  let s2 := reciprocal 9 - reciprocal 3 ≠ reciprocal 6
  let s3 := reciprocal 5 * reciprocal 10 = reciprocal 50
  let s4 := reciprocal 16 / reciprocal 4 = reciprocal 4
  (cond s1 1 0) + (cond s2 1 0) + (cond s3 1 0) + (cond s4 1 0) = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_true_statements_l1321_132124


namespace NUMINAMATH_GPT_count_integers_divisible_by_2_3_5_7_l1321_132155

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end NUMINAMATH_GPT_count_integers_divisible_by_2_3_5_7_l1321_132155


namespace NUMINAMATH_GPT_best_k_k_l1321_132193

theorem best_k_k' (v w x y z : ℝ) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  1 < (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) ∧ 
  (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) < 4 :=
sorry

end NUMINAMATH_GPT_best_k_k_l1321_132193


namespace NUMINAMATH_GPT_eleven_billion_in_scientific_notation_l1321_132138

namespace ScientificNotation

def Yi : ℝ := 10 ^ 8

theorem eleven_billion_in_scientific_notation : (11 * (10 : ℝ) ^ 9) = (1.1 * (10 : ℝ) ^ 10) :=
by 
  sorry

end ScientificNotation

end NUMINAMATH_GPT_eleven_billion_in_scientific_notation_l1321_132138


namespace NUMINAMATH_GPT_problem_statement_l1321_132154

-- Definitions
def MagnitudeEqual : Prop := (2.4 : ℝ) = (2.40 : ℝ)
def CountUnit2_4 : Prop := (0.1 : ℝ) = 2.4 / 24
def CountUnit2_40 : Prop := (0.01 : ℝ) = 2.40 / 240

-- Theorem statement
theorem problem_statement : MagnitudeEqual ∧ CountUnit2_4 ∧ CountUnit2_40 → True := by
  intros
  sorry

end NUMINAMATH_GPT_problem_statement_l1321_132154


namespace NUMINAMATH_GPT_bugs_eat_total_flowers_l1321_132121

theorem bugs_eat_total_flowers :
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  total = 17 :=
by
  -- Applying given values to compute the total flowers eaten
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  
  -- Verify the total is 17
  have h_total : total = 17 := 
    by
    sorry

  -- Proving the final result
  exact h_total

end NUMINAMATH_GPT_bugs_eat_total_flowers_l1321_132121


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l1321_132173

variable (x y p : ℝ)

-- Conditions
def condition1 := x = 1 + 3^p
def condition2 := y = 1 + 3^(-p)

-- The theorem to be proven
theorem express_y_in_terms_of_x (h1 : condition1 x p) (h2 : condition2 y p) : y = x / (x - 1) :=
sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l1321_132173


namespace NUMINAMATH_GPT_Jason_total_money_l1321_132147

theorem Jason_total_money :
  let quarter_value := 0.25
  let dime_value := 0.10
  let nickel_value := 0.05
  let initial_quarters := 49
  let initial_dimes := 32
  let initial_nickels := 18
  let additional_quarters := 25
  let additional_dimes := 15
  let additional_nickels := 10
  let initial_money := initial_quarters * quarter_value + initial_dimes * dime_value + initial_nickels * nickel_value
  let additional_money := additional_quarters * quarter_value + additional_dimes * dime_value + additional_nickels * nickel_value
  initial_money + additional_money = 24.60 :=
by
  sorry

end NUMINAMATH_GPT_Jason_total_money_l1321_132147


namespace NUMINAMATH_GPT_simplified_identity_l1321_132140

theorem simplified_identity :
  (12 : ℚ) * ( (1/3 : ℚ) + (1/4) + (1/6) + (1/12) )⁻¹ = 72 / 5 :=
  sorry

end NUMINAMATH_GPT_simplified_identity_l1321_132140


namespace NUMINAMATH_GPT_measure_angle_ADC_l1321_132125

variable (A B C D : Type)
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Definitions for the angles
variable (angle_ABC angle_BCD angle_ADC : ℝ)

-- Conditions for the problem
axiom Angle_ABC_is_4_times_Angle_BCD : angle_ABC = 4 * angle_BCD
axiom Angle_BCD_ADC_sum_to_180 : angle_BCD + angle_ADC = 180

-- The theorem that we want to prove
theorem measure_angle_ADC (Angle_ABC_is_4_times_Angle_BCD: angle_ABC = 4 * angle_BCD)
    (Angle_BCD_ADC_sum_to_180: angle_BCD + angle_ADC = 180) : 
    angle_ADC = 144 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_ADC_l1321_132125


namespace NUMINAMATH_GPT_no_quadruples_sum_2013_l1321_132165

theorem no_quadruples_sum_2013 :
  ¬ ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c + d = 2013 ∧
  2013 % a = 0 ∧ 2013 % b = 0 ∧ 2013 % c = 0 ∧ 2013 % d = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_quadruples_sum_2013_l1321_132165


namespace NUMINAMATH_GPT_distinct_socks_pairs_l1321_132170

theorem distinct_socks_pairs (n : ℕ) (h : n = 9) : (Nat.choose n 2) = 36 := by
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_distinct_socks_pairs_l1321_132170


namespace NUMINAMATH_GPT_min_distance_between_parallel_lines_l1321_132156

theorem min_distance_between_parallel_lines
  (m c_1 c_2 : ℝ)
  (h_parallel : ∀ x : ℝ, m * x + c_1 = m * x + c_2 → false) :
  ∃ D : ℝ, D = (|c_2 - c_1|) / (Real.sqrt (1 + m^2)) :=
by
  sorry

end NUMINAMATH_GPT_min_distance_between_parallel_lines_l1321_132156


namespace NUMINAMATH_GPT_proof_problem_l1321_132122

-- Let P, Q, R be points on a circle of radius s
-- Given: PQ = PR, PQ > s, and minor arc QR is 2s
-- Prove: PQ / QR = sin(1)

noncomputable def point_on_circle (s : ℝ) : ℝ → ℝ × ℝ := sorry
def radius {s : ℝ} (P Q : ℝ × ℝ ) : Prop := dist P Q = s

theorem proof_problem (s : ℝ) (P Q R : ℝ × ℝ)
  (hPQ : dist P Q = dist P R)
  (hPQ_gt_s : dist P Q > s)
  (hQR_arc_len : 1 = s) :
  dist P Q / (2 * s) = Real.sin 1 := 
sorry

end NUMINAMATH_GPT_proof_problem_l1321_132122


namespace NUMINAMATH_GPT_sin_half_angle_l1321_132189

theorem sin_half_angle
  (theta : ℝ)
  (h1 : Real.sin theta = 3 / 5)
  (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  Real.sin (theta / 2) = - (3 * Real.sqrt 10 / 10) :=
by
  sorry

end NUMINAMATH_GPT_sin_half_angle_l1321_132189


namespace NUMINAMATH_GPT_chairs_to_exclude_l1321_132199

theorem chairs_to_exclude (chairs : ℕ) (h : chairs = 765) : 
  ∃ n, n^2 ≤ chairs ∧ chairs - n^2 = 36 := 
by 
  sorry

end NUMINAMATH_GPT_chairs_to_exclude_l1321_132199


namespace NUMINAMATH_GPT_find_k4_l1321_132152

theorem find_k4
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : ∃ r : ℝ, a_n 2^2 = a_n 1 * a_n 6)
  (h4 : a_n 1 = a_n k_1)
  (h5 : a_n 2 = a_n k_2)
  (h6 : a_n 6 = a_n k_3)
  (h_k1 : k_1 = 1)
  (h_k2 : k_2 = 2)
  (h_k3 : k_3 = 6) 
  : ∃ k_4 : ℕ, k_4 = 22 := sorry

end NUMINAMATH_GPT_find_k4_l1321_132152


namespace NUMINAMATH_GPT_find_n_l1321_132144

theorem find_n (a1 a2 : ℕ) (s2 s1 : ℕ) (n : ℕ) :
    a1 = 12 →
    a2 = 3 →
    s2 = 3 * s1 →
    ∃ n : ℕ, a1 / (1 - a2/a1) = 16 ∧
             a1 / (1 - (a2 + n) / a1) = s2 →
             n = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_n_l1321_132144


namespace NUMINAMATH_GPT_speed_of_train_b_l1321_132107

-- Defining the known data
def train_a_speed := 60 -- km/h
def train_a_time_after_meeting := 9 -- hours
def train_b_time_after_meeting := 4 -- hours

-- Statement we want to prove
theorem speed_of_train_b : ∃ (V_b : ℝ), V_b = 135 :=
by
  -- Sorry placeholder, as the proof is not required
  sorry

end NUMINAMATH_GPT_speed_of_train_b_l1321_132107


namespace NUMINAMATH_GPT_greatest_int_less_than_neg_19_div_3_l1321_132162

theorem greatest_int_less_than_neg_19_div_3 : ∃ n : ℤ, n = -7 ∧ n < (-19 / 3 : ℚ) ∧ (-19 / 3 : ℚ) < n + 1 := 
by
  sorry

end NUMINAMATH_GPT_greatest_int_less_than_neg_19_div_3_l1321_132162


namespace NUMINAMATH_GPT_find_integer_pairs_l1321_132108

noncomputable def satisfies_equation (x y : ℤ) :=
  12 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 28 * (x + y)

theorem find_integer_pairs (m n : ℤ) :
  satisfies_equation (3 * m - 4 * n) (4 * n) :=
sorry

end NUMINAMATH_GPT_find_integer_pairs_l1321_132108


namespace NUMINAMATH_GPT_additional_increment_charge_cents_l1321_132157

-- Conditions as definitions
def first_increment_charge_cents : ℝ := 3.10
def total_charge_8_minutes_cents : ℝ := 18.70
def total_minutes : ℝ := 8
def increments_per_minute : ℝ := 5
def total_increments : ℝ := total_minutes * increments_per_minute
def remaining_increments : ℝ := total_increments - 1
def remaining_charge_cents : ℝ := total_charge_8_minutes_cents - first_increment_charge_cents

-- Proof problem: What is the charge for each additional 1/5 of a minute?
theorem additional_increment_charge_cents : remaining_charge_cents / remaining_increments = 0.40 := by
  sorry

end NUMINAMATH_GPT_additional_increment_charge_cents_l1321_132157


namespace NUMINAMATH_GPT_even_sum_probability_l1321_132196

theorem even_sum_probability :
  let wheel1 := (2/6, 3/6, 1/6)   -- (probability of even, odd, zero) for the first wheel
  let wheel2 := (2/4, 2/4)        -- (probability of even, odd) for the second wheel
  let both_even := (1/3) * (1/2)  -- probability of both numbers being even
  let both_odd := (1/2) * (1/2)   -- probability of both numbers being odd
  let zero_and_even := (1/6) * (1/2)  -- probability of one number being zero and the other even
  let total_probability := both_even + both_odd + zero_and_even
  total_probability = 1/2 := by sorry

end NUMINAMATH_GPT_even_sum_probability_l1321_132196


namespace NUMINAMATH_GPT_graph_paper_problem_l1321_132180

theorem graph_paper_problem :
  let line_eq := ∀ x y : ℝ, 7 * x + 268 * y = 1876
  ∃ (n : ℕ), 
  (∀ x y : ℕ, 0 < x ∧ x ≤ 268 ∧ 0 < y ∧ y ≤ 7 ∧ (7 * (x:ℝ) + 268 * (y:ℝ)) < 1876) →
  n = 801 :=
by
  sorry

end NUMINAMATH_GPT_graph_paper_problem_l1321_132180


namespace NUMINAMATH_GPT_tank_capacity_l1321_132177

theorem tank_capacity
  (x : ℝ) -- define x as the full capacity of the tank in gallons
  (h1 : (5/6) * x - (2/3) * x = 15) -- first condition
  (h2 : (2/3) * x = y) -- second condition, though not actually needed
  : x = 90 := 
by sorry

end NUMINAMATH_GPT_tank_capacity_l1321_132177


namespace NUMINAMATH_GPT_ellipse_standard_equation_parabola_standard_equation_l1321_132159

-- Ellipse with major axis length 10 and eccentricity 4/5
theorem ellipse_standard_equation (a c b : ℝ) (h₀ : a = 5) (h₁ : c = 4) (h₂ : b = 3) :
  (x^2 / a^2) + (y^2 / b^2) = 1 := by sorry

-- Parabola with vertex at the origin and directrix y = 2
theorem parabola_standard_equation (p : ℝ) (h₀ : p = 4) :
  x^2 = -8 * y := by sorry

end NUMINAMATH_GPT_ellipse_standard_equation_parabola_standard_equation_l1321_132159


namespace NUMINAMATH_GPT_heesu_has_greatest_sum_l1321_132119

def sum_cards (cards : List Int) : Int :=
  cards.foldl (· + ·) 0

theorem heesu_has_greatest_sum :
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sum_cards heesu_cards > sum_cards sora_cards ∧ sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sorry

end NUMINAMATH_GPT_heesu_has_greatest_sum_l1321_132119


namespace NUMINAMATH_GPT_entire_show_length_l1321_132184

def first_segment (S T : ℕ) : ℕ := 2 * (S + T)
def second_segment (T : ℕ) : ℕ := 2 * T
def third_segment : ℕ := 10

theorem entire_show_length : 
  first_segment (second_segment third_segment) third_segment + 
  second_segment third_segment + 
  third_segment = 90 :=
by
  sorry

end NUMINAMATH_GPT_entire_show_length_l1321_132184


namespace NUMINAMATH_GPT_simplify_fraction_l1321_132106

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1321_132106


namespace NUMINAMATH_GPT_other_endpoint_of_diameter_l1321_132100

-- Define the basic data
def center : ℝ × ℝ := (5, 2)
def endpoint1 : ℝ × ℝ := (0, -3)
def endpoint2 : ℝ × ℝ := (10, 7)

-- State the final properties to be proved
theorem other_endpoint_of_diameter :
  ∃ (e2 : ℝ × ℝ), e2 = endpoint2 ∧
    dist center endpoint2 = dist endpoint1 center :=
sorry

end NUMINAMATH_GPT_other_endpoint_of_diameter_l1321_132100


namespace NUMINAMATH_GPT_find_Q_l1321_132158

theorem find_Q (m n Q p : ℝ) (h1 : m = 6 * n + 5)
    (h2 : p = 0.3333333333333333)
    (h3 : m + Q = 6 * (n + p) + 5) : Q = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_Q_l1321_132158


namespace NUMINAMATH_GPT_sum_zero_l1321_132123

noncomputable def f : ℝ → ℝ := sorry

theorem sum_zero :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f (x + 5) = f x) →
  f (1 / 3) = 1 →
  f (16 / 3) + f (29 / 3) + f 12 + f (-7) = 0 :=
by
  intros hodd hperiod hvalue
  sorry

end NUMINAMATH_GPT_sum_zero_l1321_132123


namespace NUMINAMATH_GPT_value_of_expression_l1321_132103

theorem value_of_expression :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l1321_132103


namespace NUMINAMATH_GPT_only_zero_function_satisfies_inequality_l1321_132142

noncomputable def f (x : ℝ) : ℝ := sorry

theorem only_zero_function_satisfies_inequality (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y ≥ (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2) →
  ∀ x : ℝ, 0 < x → f x = 0 :=
sorry

end NUMINAMATH_GPT_only_zero_function_satisfies_inequality_l1321_132142


namespace NUMINAMATH_GPT_forgotten_angle_measure_l1321_132104

theorem forgotten_angle_measure 
  (total_sum : ℕ) 
  (measured_sum : ℕ) 
  (sides : ℕ) 
  (n_minus_2 : ℕ)
  (polygon_has_18_sides : sides = 18)
  (interior_angle_sum : total_sum = n_minus_2 * 180)
  (n_minus : n_minus_2 = (sides - 2))
  (measured : measured_sum = 2754) :
  ∃ forgotten_angle, forgotten_angle = total_sum - measured_sum ∧ forgotten_angle = 126 :=
by
  sorry

end NUMINAMATH_GPT_forgotten_angle_measure_l1321_132104


namespace NUMINAMATH_GPT_academic_integers_l1321_132115

def is_academic (n : ℕ) (h : n ≥ 2) : Prop :=
  ∃ (S P : Finset ℕ), (S ∩ P = ∅) ∧ (S ∪ P = Finset.range (n + 1)) ∧ (S.sum id = P.prod id)

theorem academic_integers :
  { n | ∃ h : n ≥ 2, is_academic n h } = { n | n = 3 ∨ n ≥ 5 } :=
by
  sorry

end NUMINAMATH_GPT_academic_integers_l1321_132115


namespace NUMINAMATH_GPT_total_population_l1321_132197

variable (b g t : ℕ)

-- Conditions: 
axiom boys_to_girls (h1 : b = 4 * g) : Prop
axiom girls_to_teachers (h2 : g = 8 * t) : Prop

theorem total_population (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * b / 32 :=
sorry

end NUMINAMATH_GPT_total_population_l1321_132197
