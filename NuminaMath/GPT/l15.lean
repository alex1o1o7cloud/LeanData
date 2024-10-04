import Mathlib

namespace binom_10_3_l15_15690

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15690


namespace possible_number_of_friends_l15_15869

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l15_15869


namespace power_multiplication_l15_15686

theorem power_multiplication (a : ℝ) (b : ℝ) (m : ℕ) (n : ℕ) (h1 : a = 0.25) (h2 : b = 4) (h3 : m = 2023) (h4 : n = 2024) : 
  a^m * b^n = 4 := 
by 
  sorry

end power_multiplication_l15_15686


namespace binom_10_3_l15_15688

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15688


namespace compute_binomial_10_3_eq_120_l15_15709

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15709


namespace cost_per_book_l15_15649

theorem cost_per_book
  (books_sold_each_time : ℕ)
  (people_bought : ℕ)
  (income_per_book : ℕ)
  (profit : ℕ)
  (total_income : ℕ := books_sold_each_time * people_bought * income_per_book)
  (total_cost : ℕ := total_income - profit)
  (total_books : ℕ := books_sold_each_time * people_bought)
  (cost_per_book : ℕ := total_cost / total_books) :
  books_sold_each_time = 2 ->
  people_bought = 4 ->
  income_per_book = 20 ->
  profit = 120 ->
  cost_per_book = 5 :=
  by intros; sorry

end cost_per_book_l15_15649


namespace square_side_length_l15_15418

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l15_15418


namespace sequence_relation_l15_15922

theorem sequence_relation
  (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h1 : ∀ n, b (n + 1) * a n + b n * a (n + 1) = (-2)^n + 1)
  (h2 : ∀ n, b n = (3 + (-1 : ℚ)^(n-1)) / 2)
  (h3 : a 1 = 2) :
  ∀ n, a (2 * n) = (1 - 4^n) / 2 :=
by
  intro n
  sorry

end sequence_relation_l15_15922


namespace convince_jury_l15_15109

-- Define predicates for being a criminal, normal man, guilty, or a knight
def Criminal : Prop := sorry
def NormalMan : Prop := sorry
def Guilty : Prop := sorry
def Knight : Prop := sorry

-- Define your status
variable (you : Prop)

-- Assumptions as per given conditions
axiom criminal_not_normal_man : Criminal → ¬NormalMan
axiom you_not_guilty : ¬Guilty
axiom you_not_knight : ¬Knight

-- The statement to prove
theorem convince_jury : ¬Guilty ∧ ¬Knight := by
  exact And.intro you_not_guilty you_not_knight

end convince_jury_l15_15109


namespace problem_statement_l15_15951

open Finset BigOperators

-- Define the problem with given conditions
noncomputable def num_functions_satisfying_conditions : ℕ :=
  let A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  Finset.card ((A.product A).filter (λ p, ∃ c ∈ A, (∀ x ∈ A, f(f(x)) = c) ∧ ∃ y ∈ A, f(y) ≠ y))

-- Define the target value based on the solution
def expected_remainder : ℕ := 992

-- Statement to prove
theorem problem_statement :
  (num_functions_satisfying_conditions % 1000) = expected_remainder :=
  by sorry

end problem_statement_l15_15951


namespace find_digits_l15_15943

theorem find_digits (x y z : ℕ) (hx : x ≤ 9) (hy : y ≤ 9) (hz : z ≤ 9)
    (h_eq : (10*x+5) * (300 + 10*y + z) = 7850) : x = 2 ∧ y = 1 ∧ z = 4 :=
by {
  sorry
}

end find_digits_l15_15943


namespace bankers_gain_l15_15311

-- Definitions of given conditions
def present_worth : ℝ := 600
def rate_of_interest : ℝ := 0.10
def time_period : ℕ := 2

-- Statement of the problem to be proved: The banker's gain is 126
theorem bankers_gain 
  (PW : ℝ := present_worth) 
  (r : ℝ := rate_of_interest) 
  (n : ℕ := time_period) :
  let A := PW * (1 + r) ^ n in 
  let BG := A - PW in 
  BG = 126 := 
by 
  sorry

end bankers_gain_l15_15311


namespace sequence_formula_l15_15286

theorem sequence_formula (a : ℕ → ℕ) (c : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = a n + c * n) 
(h₃ : a 1 ≠ a 2) (h₄ : a 2 * a 2 = a 1 * a 3) : c = 2 ∧ ∀ n, a n = n^2 - n + 2 :=
by
  sorry

end sequence_formula_l15_15286


namespace anton_thought_number_l15_15038

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l15_15038


namespace anton_thought_number_is_729_l15_15036

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l15_15036


namespace not_possible_155_cents_five_coins_l15_15441

/-- It is not possible to achieve a total value of 155 cents using exactly five coins 
    from a piggy bank containing only pennies (1 cent), nickels (5 cents), 
    quarters (25 cents), and half-dollars (50 cents). -/
theorem not_possible_155_cents_five_coins (n_pennies n_nickels n_quarters n_half_dollars : ℕ) 
    (h : n_pennies + n_nickels + n_quarters + n_half_dollars = 5) : 
    n_pennies * 1 + n_nickels * 5 + n_quarters * 25 + n_half_dollars * 50 ≠ 155 := 
sorry

end not_possible_155_cents_five_coins_l15_15441


namespace tom_took_out_beads_l15_15324

-- Definitions of the conditions
def green_beads : Nat := 1
def brown_beads : Nat := 2
def red_beads : Nat := 3
def beads_left_in_container : Nat := 4

-- Total initial beads
def total_beads : Nat := green_beads + brown_beads + red_beads

-- The Lean problem statement to prove
theorem tom_took_out_beads : (total_beads - beads_left_in_container) = 2 :=
by
  sorry

end tom_took_out_beads_l15_15324


namespace alice_savings_l15_15613

variable (B : ℝ)

def savings (B : ℝ) : ℝ :=
  let first_month := 10
  let second_month := first_month + 30 + B
  let third_month := first_month + 30 + 30
  first_month + second_month + third_month

theorem alice_savings (B : ℝ) : savings B = 120 + B :=
by
  sorry

end alice_savings_l15_15613


namespace largest_multiple_of_9_less_than_100_l15_15166

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l15_15166


namespace max_value_f1_l15_15752

-- Definitions for the conditions
def f (x a b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b

-- Lean theorem statements
theorem max_value_f1 (a b : ℝ) (h : a + 2 * b = 4) :
  f 0 a b = 4 → f 1 a b ≤ 7 :=
sorry

end max_value_f1_l15_15752


namespace arithmetic_sequence_a2_value_l15_15266

theorem arithmetic_sequence_a2_value 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) = a n + 3)
  (h2 : S n = n * (a 1 + a n) / 2)
  (hS13 : S 13 = 156) :
  a 2 = -3 := 
    sorry

end arithmetic_sequence_a2_value_l15_15266


namespace company_l15_15791

-- Define conditions
def initial_outlay : ℝ := 10000

def material_cost_per_set_first_300 : ℝ := 20
def material_cost_per_set_beyond_300 : ℝ := 15

def exchange_rate : ℝ := 1.1

def import_tax_rate : ℝ := 0.10

def sales_price_per_set_first_400 : ℝ := 50
def sales_price_per_set_beyond_400 : ℝ := 45

def export_tax_threshold : ℕ := 500
def export_tax_rate : ℝ := 0.05

def production_and_sales : ℕ := 800

-- Helper functions for the problem
def material_cost_first_300_sets : ℝ :=
  300 * material_cost_per_set_first_300 * exchange_rate

def material_cost_next_500_sets : ℝ :=
  (production_and_sales - 300) * material_cost_per_set_beyond_300 * exchange_rate

def total_material_cost : ℝ :=
  material_cost_first_300_sets + material_cost_next_500_sets

def import_tax : ℝ := total_material_cost * import_tax_rate

def total_manufacturing_cost : ℝ :=
  initial_outlay + total_material_cost + import_tax

def sales_revenue_first_400_sets : ℝ :=
  400 * sales_price_per_set_first_400

def sales_revenue_next_400_sets : ℝ :=
  (production_and_sales - 400) * sales_price_per_set_beyond_400

def total_sales_revenue_before_export_tax : ℝ :=
  sales_revenue_first_400_sets + sales_revenue_next_400_sets

def sales_revenue_beyond_threshold : ℝ :=
  (production_and_sales - export_tax_threshold) * sales_price_per_set_beyond_400

def export_tax : ℝ := sales_revenue_beyond_threshold * export_tax_rate

def total_sales_revenue_after_export_tax : ℝ :=
  total_sales_revenue_before_export_tax - export_tax

def profit : ℝ :=
  total_sales_revenue_after_export_tax - total_manufacturing_cost

-- Lean 4 statement for the proof problem
theorem company's_profit_is_10990 :
  profit = 10990 := by
  sorry

end company_l15_15791


namespace book_arrangement_l15_15423

theorem book_arrangement : (Nat.choose 7 3 = 35) :=
by
  sorry

end book_arrangement_l15_15423


namespace peanut_butter_revenue_l15_15779

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l15_15779


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15222

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15222


namespace abs_eq_sum_solutions_l15_15021

theorem abs_eq_sum_solutions (x : ℝ) : (|3*x - 2| + |3*x + 1| = 3) ↔ 
  (x = -1 / 3 ∨ (-1 / 3 < x ∧ x <= 2 / 3)) :=
by
  sorry

end abs_eq_sum_solutions_l15_15021


namespace number_of_friends_l15_15874

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l15_15874


namespace simplify_fraction_l15_15019

theorem simplify_fraction : 
  (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end simplify_fraction_l15_15019


namespace carl_additional_gift_bags_l15_15057

theorem carl_additional_gift_bags (definite_visitors additional_visitors extravagant_bags average_bags total_bags_needed : ℕ) :
  definite_visitors = 50 →
  additional_visitors = 40 →
  extravagant_bags = 10 →
  average_bags = 20 →
  total_bags_needed = 90 →
  (total_bags_needed - (extravagant_bags + average_bags)) = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end carl_additional_gift_bags_l15_15057


namespace number_of_friends_l15_15873

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l15_15873


namespace arithmetic_sequence_sum_l15_15942

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_sum (h1 : a 2 + a 3 = 2) (h2 : a 4 + a 5 = 6) : a 5 + a 6 = 8 :=
sorry

end arithmetic_sequence_sum_l15_15942


namespace peanut_butter_revenue_l15_15780

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l15_15780


namespace edward_made_in_summer_l15_15563

def edward_made_in_spring := 2
def cost_of_supplies := 5
def money_left_over := 24

theorem edward_made_in_summer : edward_made_in_spring + x - cost_of_supplies = money_left_over → x = 27 :=
by
  intros h
  sorry

end edward_made_in_summer_l15_15563


namespace maximize_x4y3_l15_15770

theorem maximize_x4y3 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = 40) : 
    (x, y) = (160 / 7, 120 / 7) ↔ x ^ 4 * y ^ 3 ≤ (160 / 7) ^ 4 * (120 / 7) ^ 3 := 
sorry

end maximize_x4y3_l15_15770


namespace point_outside_circle_l15_15599

theorem point_outside_circle {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a * x + b * y = 1) : a^2 + b^2 > 1 :=
by sorry

end point_outside_circle_l15_15599


namespace side_length_of_square_l15_15395

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l15_15395


namespace Seth_gave_to_his_mother_l15_15134

variable (x : ℕ)

-- Define the conditions as per the problem statement
def initial_boxes := 9
def remaining_boxes_after_giving_to_mother := initial_boxes - x
def remaining_boxes_after_giving_half := remaining_boxes_after_giving_to_mother / 2

-- Specify the final condition
def final_boxes := 4

-- Form the main theorem
theorem Seth_gave_to_his_mother :
  final_boxes = remaining_boxes_after_giving_to_mother / 2 →
  initial_boxes - x = 8 :=
by sorry

end Seth_gave_to_his_mother_l15_15134


namespace anton_thought_number_is_729_l15_15035

-- Define the condition that a number matches another number in exactly one digit place.
def matches_in_one_digit_place (x y : ℕ) : Prop :=
  let x_h := x / 100,
      x_t := (x / 10) % 10,
      x_u := x % 10,
      y_h := y / 100,
      y_t := (y / 10) % 10,
      y_u := y % 10 in
  ((x_h = y_h ∧ x_t ≠ y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t = y_t ∧ x_u ≠ y_u) ∨
   (x_h ≠ y_h ∧ x_t ≠ y_t ∧ x_u = y_u))

-- Main theorem stating that the thought number is 729 given the conditions.
theorem anton_thought_number_is_729 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ matches_in_one_digit_place n 109 
    ∧ matches_in_one_digit_place n 704 
    ∧ matches_in_one_digit_place n 124 ∧ n = 729 := 
by
  sorry

end anton_thought_number_is_729_l15_15035


namespace largest_multiple_of_9_less_than_100_l15_15171

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l15_15171


namespace erased_number_is_100_l15_15156

theorem erased_number_is_100! :
  ∃ k : ℕ, k = 100 ∧
    ∀ (f : ℕ → ℕ), f = Nat.factorial →
    let N := ∏ i in Finset.range 200, f (i + 1) in
    let remaining_product := N / f k in
    Nat.is_square (remaining_product) :=
begin
  sorry
end

end erased_number_is_100_l15_15156


namespace find_second_number_l15_15528

theorem find_second_number
  (a : ℝ) (b : ℝ)
  (h : a = 1280)
  (h_percent : 0.25 * a = 0.20 * b + 190) :
  b = 650 :=
sorry

end find_second_number_l15_15528


namespace smallest_three_digit_multiple_of_13_l15_15829

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l15_15829


namespace min_value_expr_l15_15093

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 :=
sorry

end min_value_expr_l15_15093


namespace find_d_l15_15500

noncomputable def d_value (a b c : ℝ) := (2 * a + 2 * b + 2 * c - (3 / 4)^2) / 3

theorem find_d (a b c d : ℝ) (h : 2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + (2 * a + 2 * b + 2 * c - 3 * d)^(1/2)) : 
  d = 23 / 48 :=
sorry

end find_d_l15_15500


namespace total_weight_CaBr2_l15_15837

-- Definitions derived from conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_Br : ℝ := 79.904
def mol_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br
def moles_CaBr2 : ℝ := 4

-- Theorem statement based on the problem and correct answer
theorem total_weight_CaBr2 : moles_CaBr2 * mol_weight_CaBr2 = 799.552 :=
by
  -- Prove the theorem step-by-step
  -- substitute the definition of mol_weight_CaBr2
  -- show lhs = rhs
  sorry

end total_weight_CaBr2_l15_15837


namespace quadratic_roots_distinct_and_m_value_l15_15926

theorem quadratic_roots_distinct_and_m_value (m : ℝ) (α β : ℝ) (h_equation : ∀ x, x^2 - 2 * x - 3 * m^2 = 0 → (Root_of(x) = α ∨ Root_of(x) = β)) 
(h_alpha_beta : α + 2 * β = 5) :
  (2^2 - 4 * 1 * -3 * m^2 > 0) ∧ (m^2 = 1) :=
by
  have h_discriminant : 4 + 12 * m^2 > 0 := by sorry
  have h_root_sum : α + β = 2 := by sorry
  have h_root_product : α * β = -3 * m^2 := by sorry
  have h_quad_solved : (β = 3) ∧ (α = -1) := by sorry
  have h_m2 : m^2 = 1 := by sorry
  exact ⟨h_discriminant,h_m2⟩

end quadratic_roots_distinct_and_m_value_l15_15926


namespace valid_sequences_l15_15291

-- Define the transformation function for a ten-digit number
noncomputable def transform (n : ℕ) : ℕ := sorry

-- Given sequences
def seq1 := 1101111111
def seq2 := 1201201020
def seq3 := 1021021020
def seq4 := 0112102011

-- The proof problem statement
theorem valid_sequences :
  (transform 1101111111 = seq1) ∧
  (transform 1021021020 = seq3) ∧
  (transform 0112102011 = seq4) :=
sorry

end valid_sequences_l15_15291


namespace gcd_g105_g106_l15_15481

def g (x : ℕ) : ℕ := x^2 - x + 2502

theorem gcd_g105_g106 : gcd (g 105) (g 106) = 2 := by
  sorry

end gcd_g105_g106_l15_15481


namespace total_profit_is_correct_l15_15890

-- Definitions of the investments
def A_initial_investment : ℝ := 12000
def B_investment : ℝ := 16000
def C_investment : ℝ := 20000
def D_investment : ℝ := 24000
def E_investment : ℝ := 18000
def C_profit_share : ℝ := 36000

-- Definitions of the time periods (in months)
def time_6_months : ℝ := 6
def time_12_months : ℝ := 12

-- Calculations of investment-months for each person
def A_investment_months : ℝ := A_initial_investment * time_6_months
def B_investment_months : ℝ := B_investment * time_12_months
def C_investment_months : ℝ := C_investment * time_12_months
def D_investment_months : ℝ := D_investment * time_12_months
def E_investment_months : ℝ := E_investment * time_6_months

-- Calculation of total investment-months
def total_investment_months : ℝ :=
  A_investment_months + B_investment_months + C_investment_months +
  D_investment_months + E_investment_months

-- The main theorem stating the total profit calculation
theorem total_profit_is_correct :
  ∃ TP : ℝ, (C_profit_share / C_investment_months) = (TP / total_investment_months) ∧ TP = 135000 :=
by
  sorry

end total_profit_is_correct_l15_15890


namespace combination_10_3_l15_15696

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15696


namespace lychees_remaining_l15_15784
-- Definitions of the given conditions
def initial_lychees : ℕ := 500
def sold_lychees : ℕ := initial_lychees / 2
def home_lychees : ℕ := initial_lychees - sold_lychees
def eaten_lychees : ℕ := (3 * home_lychees) / 5

-- Statement to prove
theorem lychees_remaining : home_lychees - eaten_lychees = 100 := by
  sorry

end lychees_remaining_l15_15784


namespace greatest_integer_gcd_l15_15336

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l15_15336


namespace club_truncator_more_wins_than_losses_l15_15429

noncomputable def clubTruncatorWinsProbability : ℚ :=
  let total_matches := 8
  let prob := 1/3
  -- The combinatorial calculations for the balanced outcomes
  let balanced_outcomes := 70 + 560 + 420 + 28 + 1
  let total_outcomes := 3^total_matches
  let prob_balanced := balanced_outcomes / total_outcomes
  let prob_more_wins_or_more_losses := 1 - prob_balanced
  (prob_more_wins_or_more_losses / 2)

theorem club_truncator_more_wins_than_losses : 
  clubTruncatorWinsProbability = 2741 / 6561 := 
by 
  sorry

#check club_truncator_more_wins_than_losses

end club_truncator_more_wins_than_losses_l15_15429


namespace coordinates_of_A_after_move_l15_15451

noncomputable def moved_coordinates (a : ℝ) : ℝ × ℝ :=
  let x := 2 * a - 9 + 5
  let y := 1 - 2 * a
  (x, y)

theorem coordinates_of_A_after_move (a : ℝ) (h : moved_coordinates a = (0, 1 - 2 * a)) :
  moved_coordinates 2 = (-5, -3) :=
by
  -- Proof omitted
  sorry

end coordinates_of_A_after_move_l15_15451


namespace greatest_int_with_gcd_five_l15_15331

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l15_15331


namespace largest_integer_lt_100_with_rem_4_div_7_l15_15210

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l15_15210


namespace find_largest_integer_l15_15216

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l15_15216


namespace min_fraction_value_l15_15530

-- Define the conditions: geometric sequence, specific term relationship, product of terms

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

def specific_term_relationship (a : ℕ → ℝ) : Prop :=
  a 3 = a 2 + 2 * a 1

def product_of_terms (a : ℕ → ℝ) (m n : ℕ) : Prop :=
  a m * a n = 64 * (a 1)^2

def min_value_fraction (m n : ℕ) : Prop :=
  1 / m + 9 / n = 2

theorem min_fraction_value (a : ℕ → ℝ) (m n : ℕ)
  (h1 : geometric_sequence a)
  (h2 : specific_term_relationship a)
  (h3 : product_of_terms a m n)
  : min_value_fraction m n := by
  sorry

end min_fraction_value_l15_15530


namespace positive_difference_eq_30_l15_15995

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l15_15995


namespace series_eq_inv_sqrt_sin_l15_15437

noncomputable def S (x : ℝ) := x + ∑' n : ℕ, (2 * (n+1)).factorial / ((n+1).factorial * (n+1).factorial * (2 * (n+1) + 1)) * x^(2*(n+1) + 1)

theorem series_eq_inv_sqrt_sin :
  ∀ x ∈ Icc (-1 : ℝ) 1,
  S x = (1 - x^2)⁻¹ / 2 * arcsin x := 
by
  intro x hx
  sorry

end series_eq_inv_sqrt_sin_l15_15437


namespace gcd_90_450_l15_15916

theorem gcd_90_450 : Int.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l15_15916


namespace total_price_all_art_l15_15474

-- Define the conditions
def total_price_first_three_pieces : ℕ := 45000
def price_next_piece := (total_price_first_three_pieces / 3) * 3 / 2 

-- Statement to prove
theorem total_price_all_art : total_price_first_three_pieces + price_next_piece = 67500 :=
by
  sorry -- Proof is omitted

end total_price_all_art_l15_15474


namespace cost_per_lb_of_mixture_l15_15545

def millet_weight : ℝ := 100
def millet_cost_per_lb : ℝ := 0.60
def sunflower_weight : ℝ := 25
def sunflower_cost_per_lb : ℝ := 1.10

theorem cost_per_lb_of_mixture :
  let millet_weight := 100
  let millet_cost_per_lb := 0.60
  let sunflower_weight := 25
  let sunflower_cost_per_lb := 1.10
  let millet_total_cost := millet_weight * millet_cost_per_lb
  let sunflower_total_cost := sunflower_weight * sunflower_cost_per_lb
  let total_cost := millet_total_cost + sunflower_total_cost
  let total_weight := millet_weight + sunflower_weight
  (total_cost / total_weight) = 0.70 :=
by
  sorry

end cost_per_lb_of_mixture_l15_15545


namespace probability_of_rain_on_at_least_one_day_is_correct_l15_15570

def rain_on_friday_probability : ℝ := 0.30
def rain_on_saturday_probability : ℝ := 0.45
def rain_on_sunday_probability : ℝ := 0.50

def rain_on_at_least_one_day_probability : ℝ := 1 - (1 - rain_on_friday_probability) * (1 - rain_on_saturday_probability) * (1 - rain_on_sunday_probability)

theorem probability_of_rain_on_at_least_one_day_is_correct :
  rain_on_at_least_one_day_probability = 0.8075 := by
sorry

end probability_of_rain_on_at_least_one_day_is_correct_l15_15570


namespace total_cost_for_seeds_l15_15864

theorem total_cost_for_seeds :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  in total = 18.00 :=
by
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  have h1 : total = 18.00,
  {
    sorry
  }
  exact h1

end total_cost_for_seeds_l15_15864


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15356

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15356


namespace total_students_l15_15152

-- Definitions from the conditions
def ratio_boys_to_girls (B G : ℕ) : Prop := B / G = 1 / 2
def girls_count := 60

-- The main statement to prove
theorem total_students (B G : ℕ) (h1 : ratio_boys_to_girls B G) (h2 : G = girls_count) : B + G = 90 := sorry

end total_students_l15_15152


namespace pastries_count_l15_15551

def C : ℕ := 19
def P : ℕ := C + 112

theorem pastries_count : P = 131 := by
  -- P = 19 + 112
  -- P = 131
  sorry

end pastries_count_l15_15551


namespace rectangle_area_change_l15_15464

theorem rectangle_area_change 
  (L B : ℝ) 
  (A : ℝ := L * B) 
  (L' : ℝ := 1.30 * L) 
  (B' : ℝ := 0.75 * B) 
  (A' : ℝ := L' * B') : 
  A' / A = 0.975 := 
by sorry

end rectangle_area_change_l15_15464


namespace sin_theta_plus_45_l15_15583

-- Statement of the problem in Lean 4

theorem sin_theta_plus_45 (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) (sin_θ_eq : Real.sin θ = 3 / 5) :
  Real.sin (θ + π / 4) = 7 * Real.sqrt 2 / 10 :=
sorry

end sin_theta_plus_45_l15_15583


namespace total_goals_during_match_l15_15200

theorem total_goals_during_match (
  A1_points_first_half : ℕ := 8,
  B_points_first_half : ℕ := A1_points_first_half / 2,
  B_points_second_half : ℕ := A1_points_first_half,
  A2_points_second_half : ℕ := B_points_second_half - 2
) : (A1_points_first_half + A2_points_second_half + B_points_first_half + B_points_second_half = 26) := by
  sorry

end total_goals_during_match_l15_15200


namespace square_side_length_l15_15410

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l15_15410


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l15_15344

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l15_15344


namespace Anton_thought_of_729_l15_15049

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l15_15049


namespace is_not_innovative_54_l15_15280

def is_innovative (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < b ∧ b < a ∧ n = a^2 - b^2

theorem is_not_innovative_54 : ¬ is_innovative 54 :=
sorry

end is_not_innovative_54_l15_15280


namespace factor_polynomial_l15_15204

theorem factor_polynomial (x : ℝ) :
  3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) :=
by
  sorry

end factor_polynomial_l15_15204


namespace largest_multiple_of_9_less_than_100_l15_15169

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l15_15169


namespace maximum_value_l15_15118

theorem maximum_value (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 :=
sorry

end maximum_value_l15_15118


namespace sin_double_angle_identity_l15_15573

theorem sin_double_angle_identity (α : ℝ) (h : Real.cos α = 1 / 4) : 
  Real.sin (π / 2 - 2 * α) = -7 / 8 :=
by 
  sorry

end sin_double_angle_identity_l15_15573


namespace side_length_of_square_l15_15385

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l15_15385


namespace avg_length_one_third_wires_l15_15816

theorem avg_length_one_third_wires (x : ℝ) (L1 L2 L3 L4 L5 L6 : ℝ) 
  (h_total_wires : L1 + L2 + L3 + L4 + L5 + L6 = 6 * 80) 
  (h_avg_other_wires : (L3 + L4 + L5 + L6) / 4 = 85) 
  (h_avg_all_wires : (L1 + L2 + L3 + L4 + L5 + L6) / 6 = 80) :
  (L1 + L2) / 2 = 70 :=
by
  sorry

end avg_length_one_third_wires_l15_15816


namespace scenario1_scenario2_scenario3_l15_15761

noncomputable def scenario1_possible_situations : Nat :=
  12

noncomputable def scenario2_possible_situations : Nat :=
  144

noncomputable def scenario3_possible_situations : Nat :=
  50

theorem scenario1 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) (not_consecutive : Prop) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 5 ∧ remaining_hits = 2 ∧ not_consecutive → 
  scenario1_possible_situations = 12 := by
  sorry

theorem scenario2 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 4 ∧ remaining_hits = 3 → 
  scenario2_possible_situations = 144 := by
  sorry

theorem scenario3 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 6 ∧ consecutive_hits = 4 ∧ remaining_hits = 2 → 
  scenario3_possible_situations = 50 := by
  sorry

end scenario1_scenario2_scenario3_l15_15761


namespace number_in_sequence_l15_15547

theorem number_in_sequence : ∃ n : ℕ, n * (n + 2) = 99 :=
by
  sorry

end number_in_sequence_l15_15547


namespace smallest_number_divisible_conditions_l15_15850

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l15_15850


namespace possible_number_of_friends_l15_15881

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l15_15881


namespace side_length_of_square_l15_15381

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l15_15381


namespace quadratic_two_distinct_real_roots_l15_15582

theorem quadratic_two_distinct_real_roots 
    (a b c : ℝ)
    (h1 : a > 0)
    (h2 : c < 0) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + bx + c = 0 := 
sorry

end quadratic_two_distinct_real_roots_l15_15582


namespace possible_number_of_friends_l15_15883

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l15_15883


namespace AM_GM_proof_equality_condition_l15_15614

variable (a b : ℝ)
variable (ha : 0 < a) (hb : 0 < b)

theorem AM_GM_proof : (a + b)^3 / (a^2 * b) ≥ 27 / 4 :=
sorry

theorem equality_condition : (a + b)^3 / (a^2 * b) = 27 / 4 ↔ a = 2 * b :=
sorry

end AM_GM_proof_equality_condition_l15_15614


namespace square_side_length_l15_15415

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l15_15415


namespace sum_of_squares_l15_15792

theorem sum_of_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := 
by
  sorry

end sum_of_squares_l15_15792


namespace matt_peanut_revenue_l15_15776

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l15_15776


namespace combination_10_3_l15_15695

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15695


namespace functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l15_15526

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

def profit_maximized (x : ℝ) : Prop :=
  daily_sales_profit x = -5 * (80 - x)^2 + 4500

def sufficient_profit_range (x : ℝ) : Prop :=
  daily_sales_profit x >= 4000 ∧ (x - 50) * (500 - 5 * x) <= 7000

theorem functional_relationship (x : ℝ) : daily_sales_profit x = -5 * x^2 + 800 * x - 27500 :=
  sorry

theorem profit_maximized_at (x : ℝ) : profit_maximized x → x = 80 ∧ daily_sales_profit x = 4500 :=
  sorry

theorem sufficient_profit_range_verified (x : ℝ) : sufficient_profit_range x → 82 ≤ x ∧ x ≤ 90 :=
  sorry

end functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l15_15526


namespace total_copies_produced_l15_15991

theorem total_copies_produced
  (rate_A : ℕ)
  (rate_B : ℕ)
  (rate_C : ℕ)
  (time_A : ℕ)
  (time_B : ℕ)
  (time_C : ℕ)
  (total_time : ℕ)
  (ha : rate_A = 10)
  (hb : rate_B = 10)
  (hc : rate_C = 10)
  (hA_time : time_A = 15)
  (hB_time : time_B = 20)
  (hC_time : time_C = 25)
  (h_total_time : total_time = 30) :
  rate_A * time_A + rate_B * time_B + rate_C * time_C = 600 :=
by 
  -- Machine A: 10 copies per minute * 15 minutes = 150 copies
  -- Machine B: 10 copies per minute * 20 minutes = 200 copies
  -- Machine C: 10 copies per minute * 25 minutes = 250 copies
  -- Hence, the total number of copies = 150 + 200 + 250 = 600
  sorry

end total_copies_produced_l15_15991


namespace largest_integer_lt_100_with_rem_4_div_7_l15_15208

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l15_15208


namespace coins_remainder_l15_15852

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l15_15852


namespace probability_gpa_at_least_3_is_2_over_9_l15_15961

def gpa_points (grade : ℕ) : ℕ :=
  match grade with
  | 4 => 4 -- A
  | 3 => 3 -- B
  | 2 => 2 -- C
  | 1 => 1 -- D
  | _ => 0 -- otherwise

def probability_of_GPA_at_least_3 : ℚ :=
  let points_physics := gpa_points 4
  let points_chemistry := gpa_points 4
  let points_biology := gpa_points 3
  let total_known_points := points_physics + points_chemistry + points_biology
  let required_points := 18 - total_known_points -- 18 points needed in total for a GPA of at least 3.0
  -- Probabilities in Mathematics:
  let prob_math_A := 1 / 9
  let prob_math_B := 4 / 9
  let prob_math_C :=  4 / 9
  -- Probabilities in Sociology:
  let prob_soc_A := 1 / 3
  let prob_soc_B := 1 / 3
  let prob_soc_C := 1 / 3
  -- Calculate the total probability of achieving at least 7 points from Mathematics and Sociology
  let prob_case_1 := prob_math_A * prob_soc_A -- Both A in Mathematics and Sociology
  let prob_case_2 := prob_math_A * prob_soc_B -- A in Mathematics and B in Sociology
  let prob_case_3 := prob_math_B * prob_soc_A -- B in Mathematics and A in Sociology
  prob_case_1 + prob_case_2 + prob_case_3 -- Total Probability

theorem probability_gpa_at_least_3_is_2_over_9 : probability_of_GPA_at_least_3 = 2 / 9 :=
by sorry

end probability_gpa_at_least_3_is_2_over_9_l15_15961


namespace math_problem_l15_15845

theorem math_problem :
  let a := 481 * 7
  let b := 426 * 5
  ((a + b) ^ 3 - 4 * a * b) = 166021128033 := 
by
  let a := 481 * 7
  let b := 426 * 5
  sorry

end math_problem_l15_15845


namespace work_completion_l15_15012

variable (A B : Type)

/-- A can do half of the work in 70 days and B can do one third of the work in 35 days.
Together, A and B can complete the work in 60 days. -/
theorem work_completion (hA : (1 : ℚ) / 2 / 70 = (1 : ℚ) / a) 
                      (hB : (1 : ℚ) / 3 / 35 = (1 : ℚ) / b) :
                      (1 / 140 + 1 / 105) = 1 / 60 :=
  sorry

end work_completion_l15_15012


namespace possible_values_2n_plus_m_l15_15982

theorem possible_values_2n_plus_m :
  ∀ (n m : ℤ), 3 * n - m < 5 → n + m > 26 → 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
by sorry

end possible_values_2n_plus_m_l15_15982


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l15_15341

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l15_15341


namespace ratio_circle_to_triangle_area_l15_15671

theorem ratio_circle_to_triangle_area 
  (h d : ℝ) 
  (h_pos : 0 < h) 
  (d_pos : 0 < d) 
  (R : ℝ) 
  (R_def : R = h / 2) :
  (π * R^2) / (1/2 * h * d) = (π * h) / (2 * d) :=
by sorry

end ratio_circle_to_triangle_area_l15_15671


namespace skirt_more_than_pants_l15_15030

def amount_cut_off_skirt : ℝ := 0.75
def amount_cut_off_pants : ℝ := 0.5

theorem skirt_more_than_pants : 
  amount_cut_off_skirt - amount_cut_off_pants = 0.25 := 
by
  sorry

end skirt_more_than_pants_l15_15030


namespace A_lt_B_l15_15274

variable (x y : ℝ)

def A (x y : ℝ) : ℝ := - y^2 + 4 * x - 3
def B (x y : ℝ) : ℝ := x^2 + 2 * x + 2 * y

theorem A_lt_B (x y : ℝ) : A x y < B x y := 
by
  sorry

end A_lt_B_l15_15274


namespace no_adjacent_black_balls_l15_15574

theorem no_adjacent_black_balls (m n : ℕ) (h : m > n) : 
  (m + 1).choose n = (m + 1).factorial / (n.factorial * (m + 1 - n).factorial) := by
  sorry

end no_adjacent_black_balls_l15_15574


namespace solve_for_x_l15_15074

theorem solve_for_x (x y : ℕ) (h1 : x / y = 10 / 4) (h2 : y = 18) : x = 45 :=
sorry

end solve_for_x_l15_15074


namespace side_length_of_square_l15_15397

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l15_15397


namespace fraction_order_l15_15516

theorem fraction_order :
  (25 / 19 : ℚ) < (21 / 16 : ℚ) ∧ (21 / 16 : ℚ) < (23 / 17 : ℚ) := by
  sorry

end fraction_order_l15_15516


namespace sara_initial_black_marbles_l15_15133

-- Define the given conditions
def red_marbles (sara_has : Nat) : Prop := sara_has = 122
def black_marbles_taken_by_fred (fred_took : Nat) : Prop := fred_took = 233
def black_marbles_now (sara_has_now : Nat) : Prop := sara_has_now = 559

-- The proof problem statement
theorem sara_initial_black_marbles
  (sara_has_red : ∀ n : Nat, red_marbles n)
  (fred_took_marbles : ∀ f : Nat, black_marbles_taken_by_fred f)
  (sara_has_now_black : ∀ b : Nat, black_marbles_now b) :
  ∃ b, b = 559 + 233 :=
by
  sorry

end sara_initial_black_marbles_l15_15133


namespace triple_sum_equals_seven_l15_15989

theorem triple_sum_equals_seven {k m n : ℕ} (hk : 0 < k) (hm : 0 < m) (hn : 0 < n)
  (hcoprime : Nat.gcd k m = 1 ∧ Nat.gcd k n = 1 ∧ Nat.gcd m n = 1)
  (hlog : k * Real.log 5 / Real.log 400 + m * Real.log 2 / Real.log 400 = n) :
  k + m + n = 7 := by
  sorry

end triple_sum_equals_seven_l15_15989


namespace largest_int_less_than_100_by_7_l15_15234

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l15_15234


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15237

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15237


namespace increase_productivity_RnD_l15_15053

theorem increase_productivity_RnD :
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  RnD_t / ΔAPL_t2 = 3260 :=
by
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  have h : RnD_t / ΔAPL_t2 = 3260 := sorry
  exact h

end increase_productivity_RnD_l15_15053


namespace largest_integer_lt_100_with_rem_4_div_7_l15_15211

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l15_15211


namespace Carter_card_number_l15_15622

-- Definitions based on conditions
def Marcus_cards : ℕ := 210
def difference : ℕ := 58

-- Definition to infer the number of Carter's baseball cards
def Carter_cards : ℕ := Marcus_cards - difference

-- Theorem statement asserting the number of baseball cards Carter has
theorem Carter_card_number : Carter_cards = 152 := by
  sorry

end Carter_card_number_l15_15622


namespace arithmetic_seq_common_diff_l15_15105

theorem arithmetic_seq_common_diff (a b : ℕ) (d : ℕ) (a1 a2 a8 a9 : ℕ) 
  (h1 : a1 + a8 = 10)
  (h2 : a2 + a9 = 18)
  (h3 : a2 = a1 + d)
  (h4 : a8 = a1 + 7 * d)
  (h5 : a9 = a1 + 8 * d)
  : d = 4 :=
by
  sorry

end arithmetic_seq_common_diff_l15_15105


namespace xiaohong_home_to_school_distance_l15_15485

noncomputable def driving_distance : ℝ := 1000
noncomputable def total_travel_time : ℝ := 22.5
noncomputable def walking_speed : ℝ := 80
noncomputable def biking_time : ℝ := 40
noncomputable def biking_speed_offset : ℝ := 800

theorem xiaohong_home_to_school_distance (d : ℝ) (v_d : ℝ) :
    let t_w := (d - driving_distance) / walking_speed
    let t_d := driving_distance / v_d
    let v_b := v_d - biking_speed_offset
    (t_d + t_w = total_travel_time)
    → (d / v_b = biking_time)
    → d = 2720 :=
by
  sorry

end xiaohong_home_to_school_distance_l15_15485


namespace friends_game_l15_15876

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l15_15876


namespace R_and_D_expense_corresponding_to_productivity_increase_l15_15055

/-- Given values for R&D expenses and increase in average labor productivity -/
def R_and_D_t : ℝ := 2640.92
def Delta_APL_t_plus_2 : ℝ := 0.81

/-- Statement to be proved: the R&D expense in million rubles corresponding 
    to an increase in average labor productivity by 1 million rubles per person -/
theorem R_and_D_expense_corresponding_to_productivity_increase : 
  R_and_D_t / Delta_APL_t_plus_2 = 3260 := 
by
  sorry

end R_and_D_expense_corresponding_to_productivity_increase_l15_15055


namespace cosine_sum_of_angles_l15_15448

theorem cosine_sum_of_angles (α β : ℝ) 
  (hα : Complex.exp (Complex.I * α) = (4 / 5) + (3 / 5) * Complex.I)
  (hβ : Complex.exp (Complex.I * β) = (-5 / 13) + (12 / 13) * Complex.I) :
  Real.cos (α + β) = -7 / 13 :=
by
  sorry

end cosine_sum_of_angles_l15_15448


namespace solve_for_m_l15_15315

namespace ProofProblem

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem solve_for_m (m : ℝ) : 3 * f 3 m = g 3 m → m = 0 := by
  sorry

end ProofProblem

end solve_for_m_l15_15315


namespace combination_10_3_l15_15701

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15701


namespace shen_winning_probability_sum_l15_15972

/-!
# Shen Winning Probability

Prove that the sum of the numerator and the denominator, m + n, 
of the simplified fraction representing Shen's winning probability is 184.
-/

theorem shen_winning_probability_sum :
  let m := 67
  let n := 117
  m + n = 184 :=
by sorry

end shen_winning_probability_sum_l15_15972


namespace six_sin6_cos6_l15_15094

theorem six_sin6_cos6 (A : ℝ) (h : Real.cos (2 * A) = - Real.sqrt 5 / 3) : 
  6 * Real.sin (A) ^ 6 + 6 * Real.cos (A) ^ 6 = 4 := 
sorry

end six_sin6_cos6_l15_15094


namespace number_of_friends_l15_15872

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l15_15872


namespace count_valid_pairs_l15_15931

theorem count_valid_pairs (i j: ℤ) (h : 0 ≤ i ∧ i < j ∧ j ≤ 49) : 
  set.count (λ (i j : ℤ),  6^j - 6^i % 210 = 0 ∧ 0 ≤ i ∧ i < j ∧ j ≤ 49) = 600 := 
sorry

end count_valid_pairs_l15_15931


namespace book_price_percentage_change_l15_15889

theorem book_price_percentage_change (P : ℝ) (x : ℝ) (h : P * (1 - (x / 100) ^ 2) = 0.90 * P) : x = 32 := by
sorry

end book_price_percentage_change_l15_15889


namespace probability_ace_then_king_l15_15668

-- Definitions of the conditions
def custom_deck := 65
def extra_spades := 14
def total_aces := 4
def total_kings := 4

-- Probability calculations
noncomputable def P_ace_first : ℚ := total_aces / custom_deck
noncomputable def P_king_second : ℚ := total_kings / (custom_deck - 1)

theorem probability_ace_then_king :
  (P_ace_first * P_king_second) = 1 / 260 :=
by
  sorry

end probability_ace_then_king_l15_15668


namespace regular_polygon_perimeter_l15_15539

theorem regular_polygon_perimeter
  (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : side_length = 8)
  (h2 : exterior_angle = 90)
  (h3 : n = 360 / exterior_angle) :
  n * side_length = 32 := by
  sorry

end regular_polygon_perimeter_l15_15539


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15238

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15238


namespace ratio_of_first_term_to_common_difference_l15_15062

theorem ratio_of_first_term_to_common_difference 
  (a d : ℤ) 
  (h : 15 * a + 105 * d = 3 * (10 * a + 45 * d)) :
  a = -2 * d :=
by 
  sorry

end ratio_of_first_term_to_common_difference_l15_15062


namespace min_megabytes_for_plan_Y_more_economical_l15_15892

theorem min_megabytes_for_plan_Y_more_economical :
  ∃ (m : ℕ), 2500 + 10 * m < 15 * m ∧ m = 501 :=
by
  sorry

end min_megabytes_for_plan_Y_more_economical_l15_15892


namespace necessary_and_sufficient_condition_l15_15078

universe u

variables {Point : Type u} 
variables (Plane : Type u) (Line : Type u)
variables (α β : Plane) (l : Line)
variables (P Q : Point)
variables (is_perpendicular : Plane → Plane → Prop)
variables (is_on_plane : Point → Plane → Prop)
variables (is_on_line : Point → Line → Prop)
variables (PQ_perpendicular_to_l : Prop) 
variables (PQ_perpendicular_to_β : Prop)
variables (line_in_plane : Line → Plane → Prop)

-- Given conditions
axiom plane_perpendicular : is_perpendicular α β
axiom plane_intersection : ∀ (α β : Plane), is_perpendicular α β → ∃ l : Line, line_in_plane l β
axiom point_on_plane_alpha : is_on_plane P α
axiom point_on_line : is_on_line Q l

-- Problem statement
theorem necessary_and_sufficient_condition :
  (PQ_perpendicular_to_l ↔ PQ_perpendicular_to_β) :=
sorry

end necessary_and_sufficient_condition_l15_15078


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15360

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15360


namespace sum_of_a_b_l15_15090

theorem sum_of_a_b (a b : ℝ) (h₁ : a^3 - 3 * a^2 + 5 * a = 1) (h₂ : b^3 - 3 * b^2 + 5 * b = 5) : a + b = 2 :=
sorry

end sum_of_a_b_l15_15090


namespace anton_thought_of_729_l15_15043

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l15_15043


namespace find_a_from_log_condition_l15_15455

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_a_from_log_condition (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
  (h₂ : f a 9 = 2) : a = 3 :=
by
  sorry

end find_a_from_log_condition_l15_15455


namespace total_balls_in_bag_l15_15104

theorem total_balls_in_bag (x : ℕ) (H : 3/(4 + x) = x/(4 + x)) : 3 + 1 + x = 7 :=
by
  -- We would provide the proof here, but it's not required as per the instructions.
  sorry

end total_balls_in_bag_l15_15104


namespace s_l15_15183

def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def total_money_spent : ℝ := 15.0

theorem s'mores_per_scout :
  (total_money_spent / cost_per_chocolate_bar * sections_per_chocolate_bar) / scouts = 2 :=
by
  sorry

end s_l15_15183


namespace proof_problem1_proof_problem2_proof_problem3_proof_problem4_l15_15023

noncomputable def problem1 : Prop := 
  2500 * (1/10000) = 0.25

noncomputable def problem2 : Prop := 
  20 * (1/100) = 0.2

noncomputable def problem3 : Prop := 
  45 * (1/60) = 3/4

noncomputable def problem4 : Prop := 
  1250 * (1/10000) = 0.125

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

theorem proof_problem3 : problem3 := by
  sorry

theorem proof_problem4 : problem4 := by
  sorry

end proof_problem1_proof_problem2_proof_problem3_proof_problem4_l15_15023


namespace inequality_proof_l15_15294

variables {x y z : ℝ}

theorem inequality_proof 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x) 
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) : 
  4 * x + y ≥ 4 * z :=
sorry

end inequality_proof_l15_15294


namespace siblings_count_l15_15111

noncomputable def Masud_siblings (M : ℕ) : Prop :=
  (4 * M - 60 = (3 * M) / 4 + 135) → M = 60

theorem siblings_count (M : ℕ) : Masud_siblings M :=
  by
  sorry

end siblings_count_l15_15111


namespace wheels_on_each_other_axle_l15_15987

def truck_toll_wheels (t : ℝ) (x : ℝ) (w : ℕ) : Prop :=
  t = 1.50 + 1.50 * (x - 2) ∧ (w = 18) ∧ (∀ y : ℕ, y = 18 - 2 - 4 *(x - 5) / 4)

theorem wheels_on_each_other_axle :
  ∀ t x w, truck_toll_wheels t x w → w = 18 ∧ x = 5 → (18 - 2) / 4 = 4 :=
by
  intros t x w h₁ h₂
  have h₃ : t = 6 := sorry
  have h₄ : x = 4 := sorry
  have h₅ : w = 18 := sorry
  have h₆ : (18 - 2) / 4 = 4 := sorry
  exact h₆

end wheels_on_each_other_axle_l15_15987


namespace applicants_less_4_years_no_degree_l15_15628

theorem applicants_less_4_years_no_degree
    (total_applicants : ℕ)
    (A : ℕ) 
    (B : ℕ)
    (C : ℕ)
    (D : ℕ)
    (h_total : total_applicants = 30)
    (h_A : A = 10)
    (h_B : B = 18)
    (h_C : C = 9)
    (h_D : total_applicants - (A - C + B - C + C) = D) :
  D = 11 :=
by
  sorry

end applicants_less_4_years_no_degree_l15_15628


namespace largest_int_less_than_100_by_7_l15_15236

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l15_15236


namespace min_sum_first_n_terms_l15_15140

variable {a₁ d c : ℝ} (n : ℕ)

noncomputable def sum_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_first_n_terms (h₁ : ∀ x, 1/3 ≤ x ∧ x ≤ 4/5 → a₁ * x^2 + (d/2 - a₁) * x + c ≥ 0)
                              (h₂ : a₁ = -15/4 * d)
                              (h₃ : d > 0) :
                              ∃ n : ℕ, n > 0 ∧ sum_first_n_terms a₁ d n ≤ sum_first_n_terms a₁ d 4 :=
by
  use 4
  sorry

end min_sum_first_n_terms_l15_15140


namespace area_of_rectangular_field_l15_15465

-- Define the conditions
def length (b : ℕ) : ℕ := b + 30
def perimeter (b : ℕ) (l : ℕ) : ℕ := 2 * (b + l)

-- Define the main theorem to prove
theorem area_of_rectangular_field (b : ℕ) (l : ℕ) (h1 : l = length b) (h2 : perimeter b l = 540) : 
  l * b = 18000 := by
  -- Placeholder for the proof
  sorry

end area_of_rectangular_field_l15_15465


namespace intersection_A_B_l15_15793

noncomputable def set_A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
noncomputable def set_B : Set ℝ := { x | 3 ≤ x }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 3 ≤ x ∧ x < 4 } := 
sorry

end intersection_A_B_l15_15793


namespace friends_number_options_l15_15884

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l15_15884


namespace find_largest_integer_l15_15217

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l15_15217


namespace area_of_square_on_PS_l15_15823

-- Given parameters as conditions in the form of hypotheses
variables (PQ QR RS PS PR : ℝ)

-- Hypotheses based on problem conditions
def hypothesis1 : PQ^2 = 25 := sorry
def hypothesis2 : QR^2 = 49 := sorry
def hypothesis3 : RS^2 = 64 := sorry
def hypothesis4 : PR^2 = PQ^2 + QR^2 := sorry
def hypothesis5 : PS^2 = PR^2 - RS^2 := sorry

-- The main theorem we need to prove
theorem area_of_square_on_PS :
  PS^2 = 10 := 
by {
  sorry
}

end area_of_square_on_PS_l15_15823


namespace determine_b_l15_15648

theorem determine_b (b : ℚ) (x y : ℚ) (h1 : x = -3) (h2 : y = 4) (h3 : 2 * b * x + (b + 2) * y = b + 6) :
  b = 2 / 3 := 
sorry

end determine_b_l15_15648


namespace problem1_problem2_l15_15663

-- Problem (I)
theorem problem1 (a b : ℝ) (h : a ≥ b ∧ b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := sorry

-- Problem (II)
theorem problem2 (a b c x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : a^2 + b^2 + c^2 = 10) 
  (h2 : x^2 + y^2 + z^2 = 40) 
  (h3 : a * x + b * y + c * z = 20) : 
  (a + b + c) / (x + y + z) = 1 / 2 := sorry

end problem1_problem2_l15_15663


namespace geometric_reasoning_l15_15754

-- Definitions of relationships between geometric objects
inductive GeometricObject
  | Line
  | Plane

open GeometricObject

def perpendicular (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be perpendicular
  | Line, Plane => True   -- Lines can be perpendicular to planes
  | Plane, Line => True   -- Planes can be perpendicular to lines
  | Line, Line => True    -- Lines can be perpendicular to lines (though normally in a 3D space specific context)

def parallel (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be parallel
  | Line, Plane => True   -- Lines can be parallel to planes under certain interpretation
  | Plane, Line => True
  | Line, Line => True    -- Lines can be parallel

axiom x : GeometricObject
axiom y : GeometricObject
axiom z : GeometricObject

-- Main theorem statement
theorem geometric_reasoning (hx : perpendicular x y) (hy : parallel y z) 
  : ¬ (perpendicular x z) → (x = Plane ∧ y = Plane ∧ z = Line) :=
  sorry

end geometric_reasoning_l15_15754


namespace range_of_a_l15_15590

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 9 ^ x - 2 * 3 ^ x + a - 3 > 0) → a > 4 :=
by
  sorry

end range_of_a_l15_15590


namespace largest_integer_less_than_100_leaving_remainder_4_l15_15250

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l15_15250


namespace solution1_solution2_l15_15553

open Complex

noncomputable def problem1 : Prop := 
  ((3 - I) / (1 + I)) ^ 2 = -3 - 4 * I

noncomputable def problem2 (z : ℂ) : Prop := 
  z = 1 + I → (2 / z - z = -2 * I)

theorem solution1 : problem1 := 
  by sorry

theorem solution2 : problem2 (1 + I) :=
  by sorry

end solution1_solution2_l15_15553


namespace function_properties_l15_15300

noncomputable def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

theorem function_properties 
  (b c : ℝ) :
  ((c = 0 → (∀ x : ℝ, f (-x) b 0 = -f x b 0)) ∧
   (b = 0 → (∀ x₁ x₂ : ℝ, (x₁ ≤ x₂ → f x₁ 0 c ≤ f x₂ 0 c))) ∧
   (∃ (c : ℝ), ∀ (x : ℝ), f (x + c) b c = f (x - c) b c) ∧
   (¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0))) := 
by
  sorry

end function_properties_l15_15300


namespace range_j_l15_15119

def h (x : ℝ) : ℝ := 2 * x + 3

def j (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_j : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 61 ≤ j x ∧ j x ≤ 93) := 
by 
  sorry

end range_j_l15_15119


namespace even_function_a_value_monotonicity_on_neg_infinity_l15_15267

noncomputable def f (x a : ℝ) : ℝ := ((x + 1) * (x + a)) / (x^2)

-- (1) Proving f(x) is even implies a = -1
theorem even_function_a_value (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = -1 :=
by
  sorry

-- (2) Proving monotonicity on (-∞, 0) for f(x) with a = -1
theorem monotonicity_on_neg_infinity (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ < 0) :
  (f x₁ (-1) > f x₂ (-1)) :=
by
  sorry

end even_function_a_value_monotonicity_on_neg_infinity_l15_15267


namespace volume_of_prism_is_429_l15_15818

theorem volume_of_prism_is_429 (x y z : ℝ) (h1 : x * y = 56) (h2 : y * z = 57) (h3 : z * x = 58) : 
  x * y * z = 429 :=
by
  sorry

end volume_of_prism_is_429_l15_15818


namespace not_convex_f4_l15_15453

-- Definition of functions
def f1 (x : ℝ) := sin x + cos x
def f2 (x : ℝ) := log (1-x)
def f3 (x : ℝ) := -x^3 + 2*x - 1
def f4 (x : ℝ) := x * exp x

-- Definition of convex function on domain D
def is_convex_on (f : ℝ → ℝ) (D : set ℝ) :=
  ∀ x ∈ D, deriv (deriv f x) < 0

-- Definitions of the specific domain
def D : set ℝ := Ioo 0 (π / 2)

-- Statement to prove
theorem not_convex_f4 : ¬is_convex_on f4 D := by
  sorry

end not_convex_f4_l15_15453


namespace solve_equation_l15_15309

theorem solve_equation (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end solve_equation_l15_15309


namespace red_bordered_area_l15_15673

theorem red_bordered_area (r₁ r₂ : ℝ) (A₁ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 6) (h₃ : A₁ = 37) :
  ∃ A₂ : ℝ, A₂ = 83.25 :=
by {
  -- We use the fact that the area relationships are proportional to the square of radii.
  let k := (r₂ / r₁)^2,
  use k * A₁,
  have h_k : k = (6/4)^2, by simp [h₁, h₂],
  simp [h₃, h_k],
  -- Provided that A₂ indeed equals the calculated value.
  norm_num,
  sorry
}

end red_bordered_area_l15_15673


namespace side_length_of_square_l15_15403

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l15_15403


namespace greatest_integer_gcd_30_is_125_l15_15350

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l15_15350


namespace problem_I_problem_II_problem_III_l15_15592

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln (a * x + 1) + x^3 - x^2 - a * x

theorem problem_I (a : ℝ) : (∃ x : ℝ, x = 2 / 3 ∧ deriv (λ x, f x a) x = 0) ↔ a = 0 :=
sorry

theorem problem_II (a : ℝ) : (∀ x : ℝ, 1 ≤ x → deriv (λ x, f x a) x ≥ 0) ↔ 0 < a ∧ a ≤ (1 + Real.sqrt 5) / 2 :=
sorry

noncomputable def f_transformed (x : ℝ) : ℝ := ln (x + 1) + x^3 - x^2 + x

theorem problem_III (b : ℝ) : (∃ x : ℝ, 0 < x ∧ f_transformed (1 - x) - (1 - x)^3 = b / x) ↔ b ∈ Iic 0 :=
sorry

end problem_I_problem_II_problem_III_l15_15592


namespace largest_n_l15_15255

def canBeFactored (A B : ℤ) : Bool :=
  A * B = 54

theorem largest_n (n : ℤ) (h : ∃ (A B : ℤ), canBeFactored A B ∧ 3 * B + A = n) :
  n = 163 :=
by
  sorry

end largest_n_l15_15255


namespace find_a_l15_15755

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem find_a (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = -1 ∨ a = (1 / 3) :=
sorry

end find_a_l15_15755


namespace quadratic_roots_r6_s6_l15_15479

theorem quadratic_roots_r6_s6 (r s : ℝ) (h1 : r + s = 3 * Real.sqrt 2) (h2 : r * s = 4) : r^6 + s^6 = 648 := by
  sorry

end quadratic_roots_r6_s6_l15_15479


namespace side_length_of_square_l15_15380

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l15_15380


namespace average_of_11_numbers_l15_15803

theorem average_of_11_numbers (a b c d e f g h i j k : ℝ)
  (h_first_6_avg : (a + b + c + d + e + f) / 6 = 98)
  (h_last_6_avg : (f + g + h + i + j + k) / 6 = 65)
  (h_6th_number : f = 318) :
  ((a + b + c + d + e + f + g + h + i + j + k) / 11) = 60 :=
by
  sorry

end average_of_11_numbers_l15_15803


namespace quadratic_polynomial_value_bound_l15_15969

theorem quadratic_polynomial_value_bound (a b : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |(x^2 + a * x + b)| ≥ 1/2 :=
by
  sorry

end quadratic_polynomial_value_bound_l15_15969


namespace probability_all_genuine_l15_15738

-- Definitions used directly in the conditions
def p_coins {C: Type} (n m: ℕ) [fintype C] [decidable_eq C] (is_genuine: C → Prop) (genuine_weight: C → ℝ) (counterfeit_weight: C → ℝ → Prop) : ℝ :=
  (15/18 : ℝ) * (14/17) * (13/16) * (12/15) * (11/14) * (10/13)

theorem probability_all_genuine (C: Type) [fintype C] [decidable_eq C] 
  (genuine: set C) (counterfeit: set C) (genuine_weight: C → ℝ) (counterfeit_weight: C → ℝ → Prop) :
  ∀ (n: ℕ) (m: ℕ) (is_genuine: C → Prop),
  (card genuine = 15) →
  (card counterfeit = 3) →
  (∀ c ∈ genuine, ∀ cw ∈ counterfeit_weight c, genuine_weight c ≠ cw) →
  let pB := p_coins n m is_genuine genuine_weight counterfeit_weight in
  pB = 55/204 →
  (55/204) / (55/204) = 1 :=
by
  sorry

end probability_all_genuine_l15_15738


namespace angle_relationship_l15_15510

-- Define the angles and the relationship
def larger_angle : ℝ := 99
def smaller_angle : ℝ := 81

-- State the problem as a theorem
theorem angle_relationship : larger_angle - smaller_angle = 18 := 
by
  -- The proof would be here
  sorry

end angle_relationship_l15_15510


namespace side_length_of_square_l15_15379

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l15_15379


namespace tangent_line_at_perpendicular_l15_15086

noncomputable theory

-- Define the given function
def f (x a : ℝ) : ℝ := (x + a) * Real.exp x

-- Define the slope of the line perpendicular to x + y + 1 = 0
def perp_slope : ℝ := 1

-- Define tangent point coordinates
def tangent_point : ℝ × ℝ := (0, 0)

-- Define the tangent line equation
def tangent_line_eq (x : ℝ) : ℝ := x

theorem tangent_line_at_perpendicular {a : ℝ} 
  (h_deriv : ∀ x, deriv (λ x, f x a) x = (x + a + 1) * Real.exp x)
  (h_perpendicular : perp_slope = 1)
  (h_tangent : tangent_point = (0, 0)) :
  tangent_line_eq 0 = 0 :=
begin
  sorry
end

end tangent_line_at_perpendicular_l15_15086


namespace area_triangle_ABC_l15_15669

noncomputable def area_of_triangle_ABC : ℝ :=
  let base_AB : ℝ := 6 - 0
  let height_AB : ℝ := 2 - 0
  let base_BC : ℝ := 6 - 3
  let height_BC : ℝ := 8 - 0
  let base_CA : ℝ := 3 - 0
  let height_CA : ℝ := 8 - 2
  let area_ratio : ℝ := 1 / 2
  let area_I' : ℝ := area_ratio * base_AB * height_AB
  let area_II' : ℝ := area_ratio * 8 * 6
  let area_III' : ℝ := area_ratio * 8 * 3
  let total_small_triangles : ℝ := area_I' + area_II' + area_III'
  let total_area_rectangle : ℝ := 6 * 8
  total_area_rectangle - total_small_triangles

theorem area_triangle_ABC : area_of_triangle_ABC = 6 := 
by
  sorry

end area_triangle_ABC_l15_15669


namespace gcd_90_450_l15_15913

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l15_15913


namespace triangle_is_isosceles_l15_15287

-- Given condition as an assumption in Lean
def sides_opposite_to_angles (a b c A B C : ℝ) (triangle : Prop) :=
  a = 2 * b * real.cos C

-- Conclusion that needs to be proved
theorem triangle_is_isosceles
  {a b c A B C : ℝ}
  (h1 : sides_opposite_to_angles a b c A B C (triangle a b c A B C)) :
  (∃ t : triangle, is_isosceles t) :=
sorry

end triangle_is_isosceles_l15_15287


namespace lydia_current_age_l15_15110

def years_for_apple_tree_to_bear_fruit : ℕ := 7
def lydia_age_when_planted_tree : ℕ := 4
def lydia_age_when_eats_apple : ℕ := 11

theorem lydia_current_age 
  (h : lydia_age_when_eats_apple - lydia_age_when_planted_tree = years_for_apple_tree_to_bear_fruit) :
  lydia_age_when_eats_apple = 11 := 
by
  sorry

end lydia_current_age_l15_15110


namespace range_of_a_l15_15936

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x - 12| < 6 → False) → (a ≤ 6 ∨ a ≥ 18) :=
by 
  intro h
  sorry

end range_of_a_l15_15936


namespace greatest_integer_gcd_l15_15337

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l15_15337


namespace expression_c_is_negative_l15_15499

noncomputable def A : ℝ := -4.2
noncomputable def B : ℝ := 2.3
noncomputable def C : ℝ := -0.5
noncomputable def D : ℝ := 3.4
noncomputable def E : ℝ := -1.8

theorem expression_c_is_negative : D / B * C < 0 := 
by
  -- proof goes here
  sorry

end expression_c_is_negative_l15_15499


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15224

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15224


namespace problem1_problem2_l15_15787

-- Definition for the first problem: determine the number of arrangements when no box is empty and ball 3 is in box B
def arrangements_with_ball3_in_B_and_no_empty_box : ℕ :=
  12

theorem problem1 : arrangements_with_ball3_in_B_and_no_empty_box = 12 :=
  by
    sorry

-- Definition for the second problem: determine the number of arrangements when ball 1 is not in box A and ball 2 is not in box B
def arrangements_with_ball1_not_in_A_and_ball2_not_in_B : ℕ :=
  36

theorem problem2 : arrangements_with_ball1_not_in_A_and_ball2_not_in_B = 36 :=
  by
    sorry

end problem1_problem2_l15_15787


namespace square_side_length_l15_15416

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l15_15416


namespace neg_p_true_l15_15457

theorem neg_p_true :
  ∀ (x : ℝ), -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 :=
by
  sorry

end neg_p_true_l15_15457


namespace solve_for_a_l15_15580

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x >= 0 then 4 ^ x else 2 ^ (a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 := 
by {
  sorry
}

end solve_for_a_l15_15580


namespace sum_base7_l15_15546

def base7_to_base10 (n : ℕ) : ℕ := 
  -- Function to convert base 7 to base 10 (implementation not shown)
  sorry

def base10_to_base7 (n : ℕ) : ℕ :=
  -- Function to convert base 10 to base 7 (implementation not shown)
  sorry

theorem sum_base7 (a b : ℕ) (ha : a = base7_to_base10 12) (hb : b = base7_to_base10 245) :
  base10_to_base7 (a + b) = 260 :=
sorry

end sum_base7_l15_15546


namespace min_sum_xy_l15_15076

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y + x * y = 3) : x + y ≥ 2 :=
by
  sorry

end min_sum_xy_l15_15076


namespace carterHas152Cards_l15_15624

-- Define the number of baseball cards Marcus has.
def marcusCards : Nat := 210

-- Define the number of baseball cards Carter has.
def carterCards : Nat := marcusCards - 58

-- Theorem to prove Carter's baseball cards total 152 given the conditions.
theorem carterHas152Cards (h1 : marcusCards = 210) (h2 : marcusCards = carterCards + 58) : carterCards = 152 :=
by
  -- Proof omitted for this exercise
  sorry

end carterHas152Cards_l15_15624


namespace possible_values_x_l15_15543

theorem possible_values_x : 
  let x := Nat.gcd 112 168 
  ∃ d : Finset ℕ, d.card = 8 ∧ ∀ y ∈ d, y ∣ 112 ∧ y ∣ 168 := 
by
  let x := Nat.gcd 112 168
  have : x = 56 := by norm_num
  use Finset.filter (fun n => 56 % n = 0) (Finset.range 57)
  sorry

end possible_values_x_l15_15543


namespace total_price_all_art_l15_15473

-- Define the conditions
def total_price_first_three_pieces : ℕ := 45000
def price_next_piece := (total_price_first_three_pieces / 3) * 3 / 2 

-- Statement to prove
theorem total_price_all_art : total_price_first_three_pieces + price_next_piece = 67500 :=
by
  sorry -- Proof is omitted

end total_price_all_art_l15_15473


namespace num_values_f100_eq_0_l15_15115

def f0 (x : ℝ) : ℝ := x + |x - 100| - |x + 100|

def fn : ℕ → ℝ → ℝ
| 0, x   => f0 x
| (n+1), x => |fn n x| - 1

theorem num_values_f100_eq_0 : ∃ (xs : Finset ℝ), ∀ x ∈ xs, fn 100 x = 0 ∧ xs.card = 301 :=
by
  sorry

end num_values_f100_eq_0_l15_15115


namespace right_triangle_m_c_l15_15433

theorem right_triangle_m_c (a b c : ℝ) (m_c : ℝ) 
  (h : (1 / a) + (1 / b) = 3 / c) : 
  m_c = (c * (1 + Real.sqrt 10)) / 9 :=
sorry

end right_triangle_m_c_l15_15433


namespace range_of_a_l15_15598

noncomputable def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ∧ 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a = -1 := 
by
  sorry

end range_of_a_l15_15598


namespace discount_percent_l15_15494

theorem discount_percent
  (MP CP SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (gainPercent : ℝ)
  (h2 : gainPercent = 54.54545454545454 / 100)
  (h3 : (SP - CP) / CP = gainPercent)
  : ((MP - SP) / MP) * 100 = 15 := by
  sorry

end discount_percent_l15_15494


namespace solution_problem_l15_15265

noncomputable def problem :=
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 →
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c)

theorem solution_problem : problem :=
  sorry

end solution_problem_l15_15265


namespace gcd_90_450_l15_15915

theorem gcd_90_450 : Int.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l15_15915


namespace count_semiprimes_expressed_as_x_cubed_minus_1_l15_15541

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_semiprime (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p * q = n

theorem count_semiprimes_expressed_as_x_cubed_minus_1 :
  (∃ S : Finset ℕ, 
    S.card = 4 ∧ 
    ∀ n ∈ S, n < 2018 ∧ 
    ∃ x : ℕ, x > 0 ∧ x^3 - 1 = n ∧ is_semiprime n) :=
sorry

end count_semiprimes_expressed_as_x_cubed_minus_1_l15_15541


namespace simplify_fractions_l15_15642

theorem simplify_fractions :
  (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 :=
by
  sorry

end simplify_fractions_l15_15642


namespace math_problem_l15_15279

noncomputable def condition1 (a b : ℤ) : Prop :=
  |2 + a| + |b - 3| = 0

noncomputable def condition2 (c d : ℝ) : Prop :=
  1 / c = -d

noncomputable def condition3 (e : ℤ) : Prop :=
  e = -5

theorem math_problem (a b e : ℤ) (c d : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 c d) 
  (h3 : condition3 e) : 
  -a^b + 1 / c - e + d = 13 :=
by
  sorry

end math_problem_l15_15279


namespace square_side_length_l15_15420

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l15_15420


namespace Brian_Frodo_ratio_l15_15052

-- Definitions from the conditions
def Lily_tennis_balls : Int := 3
def Frodo_tennis_balls : Int := Lily_tennis_balls + 8
def Brian_tennis_balls : Int := 22

-- The proof statement
theorem Brian_Frodo_ratio :
  Brian_tennis_balls / Frodo_tennis_balls = 2 := by
  sorry

end Brian_Frodo_ratio_l15_15052


namespace balloon_difference_l15_15519

theorem balloon_difference 
  (your_balloons : ℕ := 7) 
  (friend_balloons : ℕ := 5) : 
  your_balloons - friend_balloons = 2 := 
by 
  sorry

end balloon_difference_l15_15519


namespace days_to_learn_all_vowels_l15_15260

-- Defining the number of vowels
def number_of_vowels : Nat := 5

-- Defining the days Charles takes to learn one alphabet
def days_per_vowel : Nat := 7

-- Prove that Charles needs 35 days to learn all the vowels
theorem days_to_learn_all_vowels : number_of_vowels * days_per_vowel = 35 := by
  sorry

end days_to_learn_all_vowels_l15_15260


namespace greatest_integer_gcd_30_is_125_l15_15347

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l15_15347


namespace positive_difference_eq_30_l15_15996

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l15_15996


namespace binom_10_3_l15_15689

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15689


namespace binomial_coefficient_10_3_l15_15722

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15722


namespace temp_fri_l15_15139

-- Define the temperatures on Monday, Tuesday, Wednesday, Thursday, and Friday
variables (M T W Th F : ℝ)

-- Define the conditions as given in the problem
axiom avg_mon_thurs : (M + T + W + Th) / 4 = 48
axiom avg_tues_fri : (T + W + Th + F) / 4 = 46
axiom temp_mon : M = 39

-- The theorem to prove that the temperature on Friday is 31 degrees
theorem temp_fri : F = 31 :=
by
  -- placeholder for proof
  sorry

end temp_fri_l15_15139


namespace complex_magnitude_difference_eq_one_l15_15930

noncomputable def magnitude (z : Complex) : ℝ := Complex.abs z

/-- Lean 4 statement of the problem -/
theorem complex_magnitude_difference_eq_one (z₁ z₂ : Complex) (h₁ : magnitude z₁ = 1) (h₂ : magnitude z₂ = 1) (h₃ : magnitude (z₁ + z₂) = Real.sqrt 3) : magnitude (z₁ - z₂) = 1 := 
sorry

end complex_magnitude_difference_eq_one_l15_15930


namespace minimum_value_frac_abc_l15_15769

variable (a b c : ℝ)

theorem minimum_value_frac_abc
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + 2 * c = 2) :
  (a + b) / (a * b * c) ≥ 8 :=
sorry

end minimum_value_frac_abc_l15_15769


namespace value_of_a_b_squared_l15_15262

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a - b = Real.sqrt 2
axiom h2 : a * b = 4

theorem value_of_a_b_squared : (a + b)^2 = 18 := by
   sorry

end value_of_a_b_squared_l15_15262


namespace specified_time_eq_l15_15765

def distance : ℕ := 900
def ts (x : ℕ) : ℕ := x + 1
def tf (x : ℕ) : ℕ := x - 3

theorem specified_time_eq (x : ℕ) (h1 : x > 3) : 
  (distance / tf x) = 2 * (distance / ts x) :=
sorry

end specified_time_eq_l15_15765


namespace total_sum_step_l15_15654

-- Defining the conditions
def step_1_sum : ℕ := 2

-- Define the inductive process
def total_sum_labels (n : ℕ) : ℕ :=
  if n = 1 then step_1_sum
  else 2 * 3^(n - 1)

-- The theorem to prove
theorem total_sum_step (n : ℕ) : 
  total_sum_labels n = 2 * 3^(n - 1) :=
by
  sorry

end total_sum_step_l15_15654


namespace sufficient_condition_for_A_l15_15089

variables {A B C : Prop}

theorem sufficient_condition_for_A (h1 : A ↔ B) (h2 : C → B) : C → A :=
sorry

end sufficient_condition_for_A_l15_15089


namespace root_of_equation_imp_expression_eq_one_l15_15075

variable (m : ℝ)

theorem root_of_equation_imp_expression_eq_one
  (h : m^2 - m - 1 = 0) : m^2 - m = 1 :=
  sorry

end root_of_equation_imp_expression_eq_one_l15_15075


namespace simplify_power_of_power_l15_15973

theorem simplify_power_of_power (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end simplify_power_of_power_l15_15973


namespace greatest_integer_with_gcd_l15_15352

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l15_15352


namespace quadratic_two_distinct_real_roots_l15_15581

theorem quadratic_two_distinct_real_roots 
    (a b c : ℝ)
    (h1 : a > 0)
    (h2 : c < 0) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + bx + c = 0 := 
sorry

end quadratic_two_distinct_real_roots_l15_15581


namespace binomial_coefficient_10_3_l15_15712

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15712


namespace min_x_y_l15_15527

theorem min_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_x_y_l15_15527


namespace pink_highlighters_count_l15_15601

-- Definitions for the problem's conditions
def total_highlighters : Nat := 11
def yellow_highlighters : Nat := 2
def blue_highlighters : Nat := 5
def non_pink_highlighters : Nat := yellow_highlighters + blue_highlighters

-- Statement of the problem as a theorem
theorem pink_highlighters_count : total_highlighters - non_pink_highlighters = 4 :=
by
  sorry

end pink_highlighters_count_l15_15601


namespace oxen_grazing_months_l15_15178

theorem oxen_grazing_months (a_oxen : ℕ) (a_months : ℕ) (b_oxen : ℕ) (c_oxen : ℕ) (c_months : ℕ) (total_rent : ℝ) (c_share_rent : ℝ) (x : ℕ) :
  a_oxen = 10 →
  a_months = 7 →
  b_oxen = 12 →
  c_oxen = 15 →
  c_months = 3 →
  total_rent = 245 →
  c_share_rent = 63 →
  (c_oxen * c_months) / ((a_oxen * a_months) + (b_oxen * x) + (c_oxen * c_months)) = c_share_rent / total_rent →
  x = 5 :=
sorry

end oxen_grazing_months_l15_15178


namespace weight_of_five_bowling_balls_l15_15627

theorem weight_of_five_bowling_balls (b c : ℕ) (hb : 9 * b = 4 * c) (hc : c = 36) : 5 * b = 80 := by
  sorry

end weight_of_five_bowling_balls_l15_15627


namespace total_cost_for_seeds_l15_15865

theorem total_cost_for_seeds :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  in total = 18.00 :=
by
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  have h1 : total = 18.00,
  {
    sorry
  }
  exact h1

end total_cost_for_seeds_l15_15865


namespace elsa_final_marbles_l15_15065

def initial_marbles : ℕ := 40
def marbles_lost_at_breakfast : ℕ := 3
def marbles_given_to_susie : ℕ := 5
def marbles_bought_by_mom : ℕ := 12
def twice_marbles_given_back : ℕ := 2 * marbles_given_to_susie

theorem elsa_final_marbles :
    initial_marbles
    - marbles_lost_at_breakfast
    - marbles_given_to_susie
    + marbles_bought_by_mom
    + twice_marbles_given_back = 54 := 
by
    sorry

end elsa_final_marbles_l15_15065


namespace total_bike_clamps_given_away_l15_15665

-- Definitions for conditions
def bike_clamps_per_bike := 2
def bikes_sold_morning := 19
def bikes_sold_afternoon := 27

-- Theorem statement to be proven
theorem total_bike_clamps_given_away :
  bike_clamps_per_bike * bikes_sold_morning +
  bike_clamps_per_bike * bikes_sold_afternoon = 92 :=
by
  sorry -- Proof is to be filled in later

end total_bike_clamps_given_away_l15_15665


namespace side_length_of_square_l15_15383

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l15_15383


namespace symmetric_line_x_axis_l15_15646

theorem symmetric_line_x_axis (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = 2 * x + 1) → (∀ x, -y x = 2 * x + 1) → y x = -2 * x -1 :=
by
  intro h1 h2
  sorry

end symmetric_line_x_axis_l15_15646


namespace direction_vectors_of_line_l15_15978

theorem direction_vectors_of_line : 
  ∃ v : ℝ × ℝ, (3 * v.1 - 4 * v.2 = 0) ∧ (v = (1, 3/4) ∨ v = (4, 3)) :=
by
  sorry

end direction_vectors_of_line_l15_15978


namespace calc_g_f_3_l15_15957

def f (x : ℕ) : ℕ := x^3 + 3

def g (x : ℕ) : ℕ := 2 * x^2 + 3 * x + 2

theorem calc_g_f_3 : g (f 3) = 1892 := by
  sorry

end calc_g_f_3_l15_15957


namespace log_base_2_of_7_l15_15446

variable (m n : ℝ)

theorem log_base_2_of_7 (h1 : Real.log 5 = m) (h2 : Real.log 7 = n) : Real.logb 2 7 = n / (1 - m) :=
by
  sorry

end log_base_2_of_7_l15_15446


namespace smallest_n_with_314_in_decimal_l15_15977

theorem smallest_n_with_314_in_decimal {m n : ℕ} (h_rel_prime : Nat.gcd m n = 1) (h_m_lt_n : m < n) 
  (h_contains_314 : ∃ k : ℕ, (10^k * m) % n == 314) : n = 315 :=
sorry

end smallest_n_with_314_in_decimal_l15_15977


namespace max_saturdays_l15_15108

theorem max_saturdays (days_in_month : ℕ) (month : string) (is_leap_year : Prop) (start_day : ℕ) : 
  (days_in_month = 29 → is_leap_year → start_day = 6 → true) ∧ -- February in a leap year starts on Saturday
  (days_in_month = 30 → (start_day = 5 ∨ start_day = 6) → true) ∧ -- 30-day months start on Friday or Saturday
  (days_in_month = 31 → (start_day = 4 ∨ start_day = 5 ∨ start_day = 6) → true) ∧ -- 31-day months start on Thursday, Friday, or Saturday
  (31 ≤ days_in_month ∧ days_in_month ≤ 28 → false) → -- Other case should be false
  ∃ n : ℕ, n = 5 := -- Maximum number of Saturdays is 5
sorry

end max_saturdays_l15_15108


namespace binomial_coefficient_10_3_l15_15711

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15711


namespace elois_made_3_loaves_on_Monday_l15_15564

theorem elois_made_3_loaves_on_Monday
    (bananas_per_loaf : ℕ)
    (twice_as_many : ℕ)
    (total_bananas : ℕ) 
    (h1 : bananas_per_loaf = 4) 
    (h2 : twice_as_many = 2) 
    (h3 : total_bananas = 36)
  : ∃ L : ℕ, (4 * L + 8 * L = 36) ∧ L = 3 :=
sorry

end elois_made_3_loaves_on_Monday_l15_15564


namespace parallelLines_perpendicularLines_l15_15369

-- Problem A: Parallel lines
theorem parallelLines (a : ℝ) : 
  (∀x y : ℝ, y = -x + 2 * a → y = (a^2 - 2) * x + 2 → -1 = a^2 - 2) → 
  a = -1 := 
sorry

-- Problem B: Perpendicular lines
theorem perpendicularLines (a : ℝ) : 
  (∀x y : ℝ, y = (2 * a - 1) * x + 3 → y = 4 * x - 3 → (2 * a - 1) * 4 = -1) →
  a = 3 / 8 := 
sorry

end parallelLines_perpendicularLines_l15_15369


namespace greatest_integer_with_gcd_l15_15355

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l15_15355


namespace square_side_length_l15_15390

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l15_15390


namespace inequlity_proof_l15_15606

theorem inequlity_proof (a b : ℝ) : a^2 + a * b + b^2 ≥ 3 * (a + b - 1) := 
  sorry

end inequlity_proof_l15_15606


namespace real_root_exists_l15_15264

theorem real_root_exists (a b c : ℝ) :
  (∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b - c) * x + (c - a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c - a) * x + (a - b) = 0) :=
by {
  sorry
}

end real_root_exists_l15_15264


namespace train_length_l15_15544

theorem train_length (speed_kmph : ℕ) (time_seconds : ℕ) (length_meters : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_seconds = 14)
  (h3 : length_meters = speed_kmph * 1000 * time_seconds / 3600)
  : length_meters = 280 := by
  sorry

end train_length_l15_15544


namespace anton_thought_number_l15_15040

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l15_15040


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15219

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15219


namespace greatest_int_with_gcd_five_l15_15334

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l15_15334


namespace original_ratio_of_flour_to_baking_soda_l15_15107

-- Define the conditions
def sugar_to_flour_ratio_5_to_5 (sugar flour : ℕ) : Prop :=
  sugar = 2400 ∧ sugar = flour

def baking_soda_mass_condition (flour : ℕ) (baking_soda : ℕ) : Prop :=
  flour = 2400 ∧ (∃ b : ℕ, baking_soda = b ∧ flour / (b + 60) = 8)

-- The theorem statement we need to prove
theorem original_ratio_of_flour_to_baking_soda :
  ∃ flour baking_soda : ℕ,
  sugar_to_flour_ratio_5_to_5 2400 flour ∧
  baking_soda_mass_condition flour baking_soda →
  flour / baking_soda = 10 :=
by
  sorry

end original_ratio_of_flour_to_baking_soda_l15_15107


namespace largest_integer_less_than_100_leaving_remainder_4_l15_15251

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l15_15251


namespace square_side_length_l15_15388

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l15_15388


namespace factorization_from_left_to_right_l15_15893

theorem factorization_from_left_to_right (a x y b : ℝ) :
  (a * (a + 1) = a^2 + a ∨
   a^2 + 3 * a - 1 = a * (a + 3) + 1 ∨
   x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) ∨
   (a - b)^3 = -(b - a)^3) →
  (x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)) := sorry

end factorization_from_left_to_right_l15_15893


namespace john_average_speed_l15_15469

theorem john_average_speed :
  let distance_uphill := 2 -- distance in km
  let distance_downhill := 2 -- distance in km
  let time_uphill := 45 / 60 -- time in hours (45 minutes)
  let time_downhill := 15 / 60 -- time in hours (15 minutes)
  let total_distance := distance_uphill + distance_downhill -- total distance in km
  let total_time := time_uphill + time_downhill -- total time in hours
  total_distance / total_time = 4 := by
  sorry

end john_average_speed_l15_15469


namespace intersection_is_correct_l15_15271

def setA := {x : ℝ | 3 * x - x^2 > 0}
def setB := {x : ℝ | x ≤ 1}

theorem intersection_is_correct : 
  setA ∩ setB = {x | 0 < x ∧ x ≤ 1} :=
sorry

end intersection_is_correct_l15_15271


namespace three_digit_problem_l15_15908

theorem three_digit_problem :
  ∃ (M Γ U : ℕ), 
    M ≠ Γ ∧ M ≠ U ∧ Γ ≠ U ∧
    M ≤ 9 ∧ Γ ≤ 9 ∧ U ≤ 9 ∧
    100 * M + 10 * Γ + U = (M + Γ + U) * (M + Γ + U - 2) ∧
    100 * M + 10 * Γ + U = 195 :=
by
  sorry

end three_digit_problem_l15_15908


namespace find_a_and_b_min_value_expression_l15_15082

universe u

-- Part (1): Prove the values of a and b
theorem find_a_and_b :
    (∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
    a = 1 ∧ b = 2 :=
sorry

-- Part (2): Given a = 1 and b = 2 prove the minimum value of 2x + y + 3
theorem min_value_expression :
    (1 / (x + 1) + 2 / (y + 1) = 1) →
    (x > 0) →
    (y > 0) →
    ∀ x y : ℝ, 2 * x + y + 3 ≥ 8 :=
sorry

end find_a_and_b_min_value_expression_l15_15082


namespace george_and_hannah_received_A_grades_l15_15736

-- Define students as propositions
variables (Elena Fred George Hannah : Prop)

-- Define the conditions
def condition1 : Prop := Elena → Fred
def condition2 : Prop := Fred → George
def condition3 : Prop := George → Hannah
def condition4 : Prop := ∃ A1 A2 : Prop, A1 ∧ A2 ∧ (A1 ≠ A2) ∧ (A1 = George ∨ A1 = Hannah) ∧ (A2 = George ∨ A2 = Hannah)

-- The theorem to be proven: George and Hannah received A grades
theorem george_and_hannah_received_A_grades :
  condition1 Elena Fred →
  condition2 Fred George →
  condition3 George Hannah →
  condition4 George Hannah :=
by
  sorry

end george_and_hannah_received_A_grades_l15_15736


namespace largest_possible_integer_in_list_l15_15186

theorem largest_possible_integer_in_list :
  ∃ (a b c d e : ℕ), 
  (a = 6) ∧ 
  (b = 6) ∧ 
  (c = 7) ∧ 
  (∀ x, x ≠ a ∨ x ≠ b ∨ x ≠ c → x ≠ 6) ∧ 
  (d > 7) ∧ 
  (12 = (a + b + c + d + e) / 5) ∧ 
  (max a (max b (max c (max d e))) = 33) := by
  sorry

end largest_possible_integer_in_list_l15_15186


namespace group_friends_opponents_l15_15878

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l15_15878


namespace problem_statement_l15_15010

variables {A B x y a : ℝ}

theorem problem_statement (h1 : 1/A = 1 - (1 - x) / y)
                          (h2 : 1/B = 1 - y / (1 - x))
                          (h3 : x = (1 - a) / (1 - 1/a))
                          (h4 : y = 1 - 1/x)
                          (h5 : a ≠ 1) (h6 : a ≠ -1) : 
                          A + B = 1 :=
sorry

end problem_statement_l15_15010


namespace determine_k_value_l15_15064

theorem determine_k_value (x y z k : ℝ) 
  (h1 : 5 / (x + y) = k / (x - z))
  (h2 : k / (x - z) = 9 / (z + y)) :
  k = 14 :=
sorry

end determine_k_value_l15_15064


namespace binom_10_3_l15_15692

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15692


namespace comb_10_3_eq_120_l15_15729

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15729


namespace stone_solution_l15_15180

noncomputable def stone_problem : Prop :=
  ∃ y : ℕ, (∃ x z : ℕ, x + y + z = 100 ∧ x + 10 * y + 50 * z = 500) ∧
    ∀ y1 y2 : ℕ, (∃ x1 z1 : ℕ, x1 + y1 + z1 = 100 ∧ x1 + 10 * y1 + 50 * z1 = 500) ∧
                (∃ x2 z2 : ℕ, x2 + y2 + z2 = 100 ∧ x2 + 10 * y2 + 50 * z2 = 500) →
                y1 = y2

theorem stone_solution : stone_problem :=
sorry

end stone_solution_l15_15180


namespace initial_amount_l15_15375

theorem initial_amount (M : ℝ) (h1 : M * 2 - 50 > 0) (h2 : (M * 2 - 50) * 2 - 60 > 0) 
(h3 : ((M * 2 - 50) * 2 - 60) * 2 - 70 > 0) 
(h4 : (((M * 2 - 50) * 2 - 60) * 2 - 70) * 2 - 80 = 0) : M = 53.75 := 
sorry

end initial_amount_l15_15375


namespace lines_parallel_l15_15984

theorem lines_parallel :
  ∀ (x y : ℝ), (x - y + 2 = 0) ∧ (x - y + 1 = 0) → False :=
by
  intros x y h
  sorry

end lines_parallel_l15_15984


namespace problem1_problem2_problem3_problem4_l15_15664

theorem problem1 (α : ℝ) (h₁ : Real.sin α > 0) (h₂ : Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π/2 } := sorry

theorem problem2 (α : ℝ) (h₁ : Real.tan α * Real.sin α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > π ∧ x < 3 * π / 2) } := sorry

theorem problem3 (α : ℝ) (h₁ : Real.sin α * Real.cos α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > 3 * π / 2 ∧ x < 2 * π) } := sorry

theorem problem4 (α : ℝ) (h₁ : Real.cos α * Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π ∨ x > π ∧ x < 3 * π / 2 } := sorry

end problem1_problem2_problem3_problem4_l15_15664


namespace tank_capacity_l15_15144

theorem tank_capacity (one_third_full : ℚ) (added_water : ℚ) (capacity : ℚ) 
  (h1 : one_third_full = 1 / 3) 
  (h2 : 2 * one_third_full * capacity = 16) 
  (h3 : added_water = 16) 
  : capacity = 24 := 
by
  sorry

end tank_capacity_l15_15144


namespace hh3_value_l15_15456

noncomputable def h (x : ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x - 1

theorem hh3_value : h (h 3) = 3406935 := by
  sorry

end hh3_value_l15_15456


namespace original_avg_age_is_fifty_l15_15364

-- Definitions based on conditions
variable (N : ℕ) -- original number of students
variable (A : ℕ) -- original average age
variable (new_students : ℕ) -- number of new students
variable (new_avg_age : ℕ) -- average age of new students
variable (decreased_avg_age : ℕ) -- new average age after new students join

-- Conditions given in the problem
def original_avg_age_condition : Prop := A = 50
def new_students_condition : Prop := new_students = 12
def avg_age_new_students_condition : Prop := new_avg_age = 32
def decreased_avg_age_condition : Prop := decreased_avg_age = 46

-- Final Mathematical Equivalent Proof Problem
theorem original_avg_age_is_fifty
  (h1 : original_avg_age_condition A)
  (h2 : new_students_condition new_students)
  (h3 : avg_age_new_students_condition new_avg_age)
  (h4 : decreased_avg_age_condition decreased_avg_age) :
  A = 50 :=
by sorry

end original_avg_age_is_fifty_l15_15364


namespace sum_of_cubes_mod_7_l15_15744

theorem sum_of_cubes_mod_7 :
  (∑ k in Finset.range 150, (k + 1) ^ 3) % 7 = 1 := 
sorry

end sum_of_cubes_mod_7_l15_15744


namespace compound_analysis_l15_15666

noncomputable def molecular_weight : ℝ := 18
noncomputable def atomic_weight_nitrogen : ℝ := 14.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.01

theorem compound_analysis :
  ∃ (n : ℕ) (element : String), element = "hydrogen" ∧ n = 4 ∧
  (∃ remaining_weight : ℝ, remaining_weight = molecular_weight - atomic_weight_nitrogen ∧
   ∃ k, remaining_weight / atomic_weight_hydrogen = k ∧ k = n) :=
by
  sorry

end compound_analysis_l15_15666


namespace total_art_cost_l15_15471

-- Definitions based on the conditions
def total_price_first_3_pieces (price_per_piece : ℤ) : ℤ :=
  price_per_piece * 3

def price_increase (price_per_piece : ℤ) : ℤ :=
  price_per_piece / 2

def total_price_all_arts (price_per_piece next_piece_price : ℤ) : ℤ :=
  (total_price_first_3_pieces price_per_piece) + next_piece_price

-- The proof problem statement
theorem total_art_cost : 
  ∀ (price_per_piece : ℤ),
  total_price_first_3_pieces price_per_piece = 45000 →
  next_piece_price = price_per_piece + price_increase price_per_piece →
  total_price_all_arts price_per_piece next_piece_price = 67500 :=
  by
    intros price_per_piece h1 h2
    sorry

end total_art_cost_l15_15471


namespace bird_migration_difference_correct_l15_15361

def bird_migration_difference : ℕ := 54

/--
There are 250 bird families consisting of 3 different bird species, each with varying migration patterns.

Species A: 100 bird families; 35% fly to Africa, 65% fly to Asia
Species B: 120 bird families; 50% fly to Africa, 50% fly to Asia
Species C: 30 bird families; 10% fly to Africa, 90% fly to Asia

Prove that the difference in the number of bird families migrating to Asia and Africa is 54.
-/
theorem bird_migration_difference_correct (A_Africa_percent : ℕ := 35) (A_Asia_percent : ℕ := 65)
  (B_Africa_percent : ℕ := 50) (B_Asia_percent : ℕ := 50)
  (C_Africa_percent : ℕ := 10) (C_Asia_percent : ℕ := 90)
  (A_count : ℕ := 100) (B_count : ℕ := 120) (C_count : ℕ := 30) :
    bird_migration_difference = 
      (A_count * A_Asia_percent / 100 + B_count * B_Asia_percent / 100 + C_count * C_Asia_percent / 100) - 
      (A_count * A_Africa_percent / 100 + B_count * B_Africa_percent / 100 + C_count * C_Africa_percent / 100) :=
by sorry

end bird_migration_difference_correct_l15_15361


namespace sugar_at_home_l15_15125

-- Definitions based on conditions
def bags_of_sugar := 2
def cups_per_bag := 6
def cups_for_batter_per_12_cupcakes := 1
def cups_for_frosting_per_12_cupcakes := 2
def dozens_of_cupcakes := 5

-- Calculation of total sugar needed and bought, in terms of definitions
def total_cupcakes := dozens_of_cupcakes * 12
def total_sugar_needed_for_batter := (total_cupcakes / 12) * cups_for_batter_per_12_cupcakes
def total_sugar_needed_for_frosting := dozens_of_cupcakes * cups_for_frosting_per_12_cupcakes
def total_sugar_needed := total_sugar_needed_for_batter + total_sugar_needed_for_frosting
def total_sugar_bought := bags_of_sugar * cups_per_bag

-- The statement to be proven in Lean
theorem sugar_at_home : total_sugar_needed - total_sugar_bought = 3 := by
  sorry

end sugar_at_home_l15_15125


namespace inequality_x4_y4_z2_l15_15846

theorem inequality_x4_y4_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^4 + y^4 + z^2 ≥  xyz * 8^(1/2) :=
  sorry

end inequality_x4_y4_z2_l15_15846


namespace inequality_solution_l15_15557

theorem inequality_solution (x : ℝ) (h : 4 ≤ |x + 2| ∧ |x + 2| ≤ 8) :
  (-10 : ℝ) ≤ x ∧ x ≤ -6 ∨ (2 : ℝ) ≤ x ∧ x ≤ 6 :=
sorry

end inequality_solution_l15_15557


namespace marbles_end_of_day_l15_15067

theorem marbles_end_of_day :
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54 :=
by
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  show initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54
  sorry

end marbles_end_of_day_l15_15067


namespace garden_plant_count_l15_15374

theorem garden_plant_count :
  let rows := 52
  let columns := 15
  rows * columns = 780 := 
by
  sorry

end garden_plant_count_l15_15374


namespace length_ab_is_constant_l15_15576

noncomputable def length_AB_constant (p : ℝ) (hp : p > 0) : Prop :=
  let parabola := { P : ℝ × ℝ | P.1 ^ 2 = 2 * p * P.2 }
  let line := { P : ℝ × ℝ | P.2 = P.1 + p / 2 }
  (∃ A B : ℝ × ℝ, A ∈ parabola ∧ B ∈ parabola ∧ A ∈ line ∧ B ∈ line ∧ 
    dist A B = 4 * p)

theorem length_ab_is_constant (p : ℝ) (hp : p > 0) : length_AB_constant p hp :=
by {
  sorry
}

end length_ab_is_constant_l15_15576


namespace side_length_of_square_l15_15400

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l15_15400


namespace product_of_five_consecutive_integers_not_square_l15_15634

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  let P := n * (n + 1) * (n + 2) * (n + 3) * (n + 4)
  ∀ k : ℕ, P ≠ k^2 := 
sorry

end product_of_five_consecutive_integers_not_square_l15_15634


namespace part1_part2_l15_15937

variables (A B C : ℝ)
variables (a b c : ℝ) -- sides of the triangle opposite to angles A, B, and C respectively

-- Part (I): Prove that c / a = 2 given b(cos A - 2 * cos C) = (2 * c - a) * cos B
theorem part1 (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B) : c / a = 2 :=
sorry

-- Part (II): Prove that b = 2 given the results from part (I) and additional conditions
theorem part2 (h1 : c / a = 2) (h2 : Real.cos B = 1 / 4) (h3 : a + b + c = 5) : b = 2 :=
sorry

end part1_part2_l15_15937


namespace coins_remainder_divide_by_nine_remainder_l15_15854

def smallest_n (n : ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_remainder (n : ℕ) (h : smallest_n n) : (∃ m : ℕ, n = 54) :=
  sorry

theorem divide_by_nine_remainder (n : ℕ) (h : smallest_n n) (h_smallest: coins_remainder n h) : n % 9 = 0 :=
  sorry

end coins_remainder_divide_by_nine_remainder_l15_15854


namespace sum_of_coefficients_l15_15759

theorem sum_of_coefficients (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ) :
  (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 →
  b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 729 :=
by
  sorry

end sum_of_coefficients_l15_15759


namespace maximal_roads_l15_15181

open Finset

-- Define the main proof problem, encapsulating the conditions and conclusion
theorem maximal_roads (N : ℕ) (hN : N ≥ 1) :
  ∃ (d : ℕ), d = Nat.choose N 3 ∧
  (∀ (f : Fin N.succ → Set (Fin N.succ)),
    (∀ (i : Fin N.succ), f i ⊆ (Fin N.succ).erase i) →
    (∀ i j, i ≠ j → (f i ∩ f j).card ≤ 1) →
    (∀ S : Finset (Fin N.succ), S.card < N → 
      ∃ i ∈ S, ∀ j ∈ S, i ≠ j → f i ∪ f j ≠ univ) →
    (∑ i, (f i).card = d)) :=
begin
  sorry
end

end maximal_roads_l15_15181


namespace problem_statement_l15_15595

variable (X Y : ℝ)

theorem problem_statement
  (h1 : 0.18 * X = 0.54 * 1200)
  (h2 : X = 4 * Y) :
  X = 3600 ∧ Y = 900 := by
  sorry

end problem_statement_l15_15595


namespace largest_integer_lt_100_with_rem_4_div_7_l15_15207

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l15_15207


namespace slope_angle_135_l15_15559

theorem slope_angle_135 (x y : ℝ) : 
  (∃ (m b : ℝ), 3 * x + 3 * y + 1 = 0 ∧ y = m * x + b ∧ m = -1) ↔ 
  (∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ Real.tan α = -1 ∧ α = 135) :=
sorry

end slope_angle_135_l15_15559


namespace max_integer_k_l15_15270

-- Definitions of the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2

-- Definition of the inequality condition
theorem max_integer_k (k : ℝ) : 
  (∀ x : ℝ, x > 2 → k * (x - 2) < x * f x + 2 * g' x + 3) ↔
  k ≤ 5 :=
sorry

end max_integer_k_l15_15270


namespace ring_toss_total_amount_l15_15986

-- Defining the amounts made in the two periods
def amount_first_period : Nat := 382
def amount_second_period : Nat := 374

-- The total amount made
def total_amount : Nat := amount_first_period + amount_second_period

-- Statement that the total amount calculated is equal to the given answer
theorem ring_toss_total_amount :
  total_amount = 756 := by
  sorry

end ring_toss_total_amount_l15_15986


namespace greatest_int_with_gcd_five_l15_15335

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l15_15335


namespace derivative_at_one_third_l15_15301

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

theorem derivative_at_one_third : (deriv f (1 / 3) = -3) := by
  sorry

end derivative_at_one_third_l15_15301


namespace trajectory_eq_l15_15940

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation for point C given the parameters s and t
def C (s t : ℝ) : ℝ × ℝ := (s * 2 + t * -1, s * 1 + t * -2)

-- Prove the equation of the trajectory of C given s + t = 1
theorem trajectory_eq (s t : ℝ) (h : s + t = 1) : ∃ x y : ℝ, C s t = (x, y) ∧ x - y - 1 = 0 := by
  -- The proof will be added here
  sorry

end trajectory_eq_l15_15940


namespace larger_fraction_of_two_l15_15000

theorem larger_fraction_of_two (x y : ℚ) (h1 : x + y = 7/8) (h2 : x * y = 1/4) : max x y = 1/2 :=
sorry

end larger_fraction_of_two_l15_15000


namespace part_a_part_b_l15_15117

def A (n : ℕ) : Set ℕ := { p | ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ p ∣ a + b ∧ p^2 ∣ a^n + b^n ∧ Nat.gcd p (a + b) = 1 }

noncomputable def f (n : ℕ) : ℕ := @Set.finite_toFinset ℕ _ (A n) sorry Finset.card

theorem part_a (n : ℕ) : Set.Finite (A n) ↔ n ≠ 2 := sorry

theorem part_b (m k : ℕ) (hmo : m % 2 = 1) (hko : k % 2 = 1) (d : ℕ) (hd : Nat.gcd m k = d) :
  f d ≤ f k + f m - f (k * m) ∧ f k + f m - f (k * m) ≤ 2 * f d := sorry

end part_a_part_b_l15_15117


namespace solve_for_a_l15_15558

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem solve_for_a (a : ℤ) : star a 3 = 18 → a = 15 := by
  intro h₁
  sorry

end solve_for_a_l15_15558


namespace combination_10_3_l15_15698

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15698


namespace half_radius_y_l15_15523

theorem half_radius_y (r_x r_y : ℝ) (hx : 2 * Real.pi * r_x = 12 * Real.pi) (harea : Real.pi * r_x ^ 2 = Real.pi * r_y ^ 2) : r_y / 2 = 3 := by
  sorry

end half_radius_y_l15_15523


namespace binomial_coefficient_10_3_l15_15716

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15716


namespace real_inequality_l15_15130

theorem real_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + a * c + b * c := by
  sorry

end real_inequality_l15_15130


namespace tangent_lines_to_circle_l15_15323

-- Conditions
def regions_not_enclosed := 68
def num_lines := 30 - 4

-- Theorem statement
theorem tangent_lines_to_circle (h: regions_not_enclosed = 68) : num_lines = 26 :=
by {
  sorry
}

end tangent_lines_to_circle_l15_15323


namespace least_three_digit_multiple_13_l15_15834

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l15_15834


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15240

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15240


namespace calculate_x_one_minus_f_l15_15297

noncomputable def x := (2 + Real.sqrt 3) ^ 500
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem calculate_x_one_minus_f : x * (1 - f) = 1 := by
  sorry

end calculate_x_one_minus_f_l15_15297


namespace a6_equals_8_l15_15285

-- Defining Sn as given in the condition
def S (n : ℕ) : ℤ :=
  if n = 0 then 0
  else n^2 - 3*n

-- Defining a_n in terms of the differences stated in the solution
def a (n : ℕ) : ℤ := S n - S (n-1)

-- The problem statement to prove
theorem a6_equals_8 : a 6 = 8 :=
by
  sorry

end a6_equals_8_l15_15285


namespace sum_cubes_mod_7_l15_15741

theorem sum_cubes_mod_7 :
  (∑ i in Finset.range 151, i ^ 3) % 7 = 0 := by
  sorry

end sum_cubes_mod_7_l15_15741


namespace value_of_2a_plus_b_l15_15142

theorem value_of_2a_plus_b : ∀ (a b : ℝ), (∀ x : ℝ, x^2 - 4*x + 7 = 19 → (x = a ∨ x = b)) → a ≥ b → 2 * a + b = 10 :=
by
  intros a b h_sol h_order
  sorry

end value_of_2a_plus_b_l15_15142


namespace complex_div_eq_l15_15368

theorem complex_div_eq (z1 z2 : ℂ) (h1 : z1 = 3 - i) (h2 : z2 = 2 + i) :
  z1 / z2 = 1 - i := by
  sorry

end complex_div_eq_l15_15368


namespace f_monotonic_m_range_l15_15591

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x - 2 * x

theorem f_monotonic {x : ℝ} (h : x ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
  Monotone f :=
sorry

theorem m_range {x : ℝ} (h : x ∈ Set.Ioo 0 (Real.pi / 2)) {m : ℝ} (hm : f x ≥ m * x^2) :
  m ≤ 0 :=
sorry

end f_monotonic_m_range_l15_15591


namespace possible_number_of_friends_l15_15882

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l15_15882


namespace largest_multiple_of_9_less_than_100_l15_15159

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l15_15159


namespace compute_binomial_10_3_eq_120_l15_15704

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15704


namespace woody_savings_l15_15362

-- Definitions from conditions
def console_cost : Int := 282
def weekly_allowance : Int := 24
def saving_weeks : Int := 10

-- Theorem to prove that the amount Woody already has is $42
theorem woody_savings :
  (console_cost - (weekly_allowance * saving_weeks)) = 42 := 
by
  sorry

end woody_savings_l15_15362


namespace least_three_digit_multiple_of_13_l15_15832

-- Define what it means to be a multiple of 13
def is_multiple_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13 * k

-- Define the range of three-digit numbers
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define our main theorem
theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), is_three_digit_number n ∧ is_multiple_of_13 n ∧
  ∀ (m : ℕ), is_three_digit_number m ∧ is_multiple_of_13 m → n ≤ m :=
begin
  -- We state the theorem without proof for simplicity
  sorry
end

end least_three_digit_multiple_of_13_l15_15832


namespace average_rate_second_drive_l15_15034

theorem average_rate_second_drive 
 (distance : ℕ) (total_time : ℕ) (d1 d2 d3 : ℕ)
 (t1 t2 t3 : ℕ) (r1 r2 r3 : ℕ)
 (h_distance : d1 = d2 ∧ d2 = d3 ∧ d1 + d2 + d3 = distance)
 (h_total_time : t1 + t2 + t3 = total_time)
 (h_drive_1 : r1 = 4 ∧ t1 = d1 / r1)
 (h_drive_2 : r3 = 6 ∧ t3 = d3 / r3)
 (h_distance_total : distance = 180)
 (h_total_time_val : total_time = 37)
  : r2 = 5 := 
by sorry

end average_rate_second_drive_l15_15034


namespace volume_range_of_rectangular_solid_l15_15083

theorem volume_range_of_rectangular_solid
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 48)
  (h2 : 4 * (a + b + c) = 36) :
  (16 : ℝ) ≤ a * b * c ∧ a * b * c ≤ 20 :=
by sorry

end volume_range_of_rectangular_solid_l15_15083


namespace ratio_of_x_to_y_l15_15838

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) : 
  x / y = 37 / 16 :=
by
  sorry

end ratio_of_x_to_y_l15_15838


namespace LiFangOutfitChoices_l15_15772

variable (shirts skirts dresses : Nat) 

theorem LiFangOutfitChoices (h_shirts : shirts = 4) (h_skirts : skirts = 3) (h_dresses : dresses = 2) :
  shirts * skirts + dresses = 14 :=
by 
  -- Given the conditions and the calculations, the expected result follows.
  sorry

end LiFangOutfitChoices_l15_15772


namespace bob_before_1230_conditional_prob_l15_15427

open ProbabilityTheory

noncomputable def prob_bob_before_1230_alice_after_bob :
  ℝ := sorry

theorem bob_before_1230_conditional_prob :
  prob_bob_before_1230_alice_after_bob = 1 / 4 :=
sorry

end bob_before_1230_conditional_prob_l15_15427


namespace graveling_cost_is_3900_l15_15013

noncomputable def cost_of_graveling_roads 
  (length : ℕ) (breadth : ℕ) (width_road : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_length := length * width_road
  let area_road_breadth := (breadth - width_road) * width_road
  let total_area := area_road_length + area_road_breadth
  total_area * cost_per_sq_m

theorem graveling_cost_is_3900 :
  cost_of_graveling_roads 80 60 10 3 = 3900 := 
by 
  unfold cost_of_graveling_roads
  sorry

end graveling_cost_is_3900_l15_15013


namespace binomial_coefficient_10_3_l15_15726

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15726


namespace largest_integer_lt_100_with_rem_4_div_7_l15_15212

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l15_15212


namespace group_of_friends_l15_15868

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l15_15868


namespace sum_of_solutions_eq_neg4_l15_15950

theorem sum_of_solutions_eq_neg4 :
  ∃ (n : ℕ) (solutions : Fin n → ℝ × ℝ),
    (∀ i, ∃ (x y : ℝ), solutions i = (x, y) ∧ abs (x - 3) = abs (y - 9) ∧ abs (x - 9) = 2 * abs (y - 3)) ∧
    (Finset.univ.sum (fun i => (solutions i).1 + (solutions i).2) = -4) :=
sorry

end sum_of_solutions_eq_neg4_l15_15950


namespace arithmetic_sequence_a5_l15_15611

-- Define the concept of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- The problem's conditions
def a₁ : ℕ := 2
def d : ℕ := 3

-- The proof problem
theorem arithmetic_sequence_a5 : arithmetic_sequence a₁ d 5 = 14 := by
  sorry

end arithmetic_sequence_a5_l15_15611


namespace stock_price_at_end_of_second_year_l15_15903

def stock_price_first_year (initial_price : ℝ) : ℝ :=
  initial_price * 2

def stock_price_second_year (price_after_first_year : ℝ) : ℝ :=
  price_after_first_year * 0.75

theorem stock_price_at_end_of_second_year : 
  (stock_price_second_year (stock_price_first_year 100) = 150) :=
by
  sorry

end stock_price_at_end_of_second_year_l15_15903


namespace square_side_length_l15_15419

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l15_15419


namespace ceil_y_squared_possibilities_l15_15461

theorem ceil_y_squared_possibilities (y : ℝ) (h : ⌈y⌉ = 15) : 
  ∃ n : ℕ, (n = 29) ∧ (∀ z : ℕ, ⌈y^2⌉ = z → (197 ≤ z ∧ z ≤ 225)) :=
by
  sorry

end ceil_y_squared_possibilities_l15_15461


namespace side_length_of_square_l15_15405

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l15_15405


namespace vector_CB_correct_l15_15275

-- Define the vectors AB and AC
def AB : ℝ × ℝ := (2, 3)
def AC : ℝ × ℝ := (-1, 2)

-- Define the vector CB as the difference of AB and AC
def CB (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Prove that CB = (3, 1) given AB and AC
theorem vector_CB_correct : CB AB AC = (3, 1) :=
by
  sorry

end vector_CB_correct_l15_15275


namespace cassy_initial_jars_l15_15194

theorem cassy_initial_jars (boxes1 jars1 boxes2 jars2 leftover: ℕ) (h1: boxes1 = 10) (h2: jars1 = 12) (h3: boxes2 = 30) (h4: jars2 = 10) (h5: leftover = 80) : 
  boxes1 * jars1 + boxes2 * jars2 + leftover = 500 := 
by 
  sorry

end cassy_initial_jars_l15_15194


namespace final_retail_price_l15_15670

theorem final_retail_price (wholesale_price markup_percentage discount_percentage desired_profit_percentage : ℝ)
  (h_wholesale : wholesale_price = 90)
  (h_markup : markup_percentage = 1)
  (h_discount : discount_percentage = 0.2)
  (h_desired_profit : desired_profit_percentage = 0.6) :
  let initial_retail_price := wholesale_price + (wholesale_price * markup_percentage)
  let discount_amount := initial_retail_price * discount_percentage
  let final_retail_price := initial_retail_price - discount_amount
  final_retail_price = 144 ∧ final_retail_price = wholesale_price + (wholesale_price * desired_profit_percentage) := by
 sorry

end final_retail_price_l15_15670


namespace smallest_base_10_integer_l15_15514

-- Given conditions
def is_valid_base (a b : ℕ) : Prop := a > 2 ∧ b > 2

def base_10_equivalence (a b n : ℕ) : Prop := (2 * a + 1 = n) ∧ (b + 2 = n)

-- The smallest base-10 integer represented as 21_a and 12_b
theorem smallest_base_10_integer :
  ∃ (a b n : ℕ), is_valid_base a b ∧ base_10_equivalence a b n ∧ n = 7 :=
by
  sorry

end smallest_base_10_integer_l15_15514


namespace domain_of_tan_function_l15_15979

theorem domain_of_tan_function :
  (∀ x : ℝ, ∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2 ↔ x ≠ (k * π) / 2 + 3 * π / 8) :=
sorry

end domain_of_tan_function_l15_15979


namespace Math_Proof_Problem_l15_15430

noncomputable def problem : ℝ := (1005^3) / (1003 * 1004) - (1003^3) / (1004 * 1005)

theorem Math_Proof_Problem : ⌊ problem ⌋ = 8 :=
by
  sorry

end Math_Proof_Problem_l15_15430


namespace wall_thickness_is_correct_l15_15459

-- Define the dimensions of the brick.
def brick_length : ℝ := 80
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of required bricks.
def num_bricks : ℝ := 2000

-- Define the dimensions of the wall.
def wall_length : ℝ := 800
def wall_height : ℝ := 600

-- The volume of one brick.
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- The volume of the wall.
def wall_volume (T : ℝ) : ℝ := wall_length * wall_height * T

-- The thickness of the wall to be proved.
theorem wall_thickness_is_correct (T_wall : ℝ) (h : num_bricks * brick_volume = wall_volume T_wall) : 
  T_wall = 22.5 :=
sorry

end wall_thickness_is_correct_l15_15459


namespace find_x_if_perpendicular_l15_15594

-- Definitions based on the conditions provided
structure Vector2 := (x : ℚ) (y : ℚ)

def a : Vector2 := ⟨2, 3⟩
def b (x : ℚ) : Vector2 := ⟨x, 4⟩

def dot_product (v1 v2 : Vector2) : ℚ := v1.x * v2.x + v1.y * v2.y

theorem find_x_if_perpendicular :
  ∀ x : ℚ, dot_product a (Vector2.mk (a.x - (b x).x) (a.y - (b x).y)) = 0 → x = 1/2 :=
by
  intro x
  intro h
  sorry

end find_x_if_perpendicular_l15_15594


namespace radishes_times_carrots_l15_15814

theorem radishes_times_carrots (cucumbers radishes carrots : ℕ) 
  (h1 : cucumbers = 15) 
  (h2 : radishes = 3 * cucumbers) 
  (h3 : carrots = 9) : 
  radishes / carrots = 5 :=
by
  sorry

end radishes_times_carrots_l15_15814


namespace side_length_of_square_l15_15398

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l15_15398


namespace compare_P_Q_l15_15953

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (1 - x)^(2*n - 1)
noncomputable def Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n - 1)*x + (n - 1)*(2*n - 1)*x^2

theorem compare_P_Q :
  ∀ (n : ℕ) (x : ℝ), n > 0 →
  ((n = 1 → P n x = Q n x) ∧
   (n = 2 → ((x = 0 → P n x = Q n x) ∧ (x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x))) ∧
   (n ≥ 3 → ((x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x)))) :=
by
  intros
  sorry

end compare_P_Q_l15_15953


namespace combination_10_3_l15_15697

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15697


namespace binomial_coefficient_10_3_l15_15718

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15718


namespace binom_10_3_l15_15694

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15694


namespace desserts_brought_by_mom_l15_15782

-- Definitions for the number of each type of dessert
def num_coconut := 1
def num_meringues := 2
def num_caramel := 7

-- Conditions from the problem as definitions
def total_desserts := num_coconut + num_meringues + num_caramel = 10
def fewer_coconut_than_meringues := num_coconut < num_meringues
def most_caramel := num_caramel > num_meringues
def josef_jakub_condition := (num_coconut + num_meringues + num_caramel) - (4 * 2) = 1

-- We need to prove the answer based on these conditions
theorem desserts_brought_by_mom :
  total_desserts ∧ fewer_coconut_than_meringues ∧ most_caramel ∧ josef_jakub_condition → 
  num_coconut = 1 ∧ num_meringues = 2 ∧ num_caramel = 7 :=
by sorry

end desserts_brought_by_mom_l15_15782


namespace comb_10_3_eq_120_l15_15733

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15733


namespace largest_integer_less_than_100_with_remainder_4_l15_15230

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l15_15230


namespace new_average_income_l15_15138

/-!
# Average Monthly Income Problem

## Problem Statement
Given:
1. The average monthly income of a family of 4 earning members was Rs. 735.
2. One of the earning members died, and the average income changed.
3. The income of the deceased member was Rs. 1170.

Prove that the new average monthly income of the family is Rs. 590.
-/

theorem new_average_income (avg_income : ℝ) (num_members : ℕ) (income_deceased : ℝ) (new_num_members : ℕ) 
  (h1 : avg_income = 735) 
  (h2 : num_members = 4) 
  (h3 : income_deceased = 1170) 
  (h4 : new_num_members = 3) : 
  (num_members * avg_income - income_deceased) / new_num_members = 590 := 
by 
  sorry

end new_average_income_l15_15138


namespace students_taking_history_but_not_statistics_l15_15103

theorem students_taking_history_but_not_statistics (H S U : ℕ) (total_students : ℕ) 
  (H_val : H = 36) (S_val : S = 30) (U_val : U = 59) (total_students_val : total_students = 90) :
  H - (H + S - U) = 29 := 
by
  sorry

end students_taking_history_but_not_statistics_l15_15103


namespace greatest_divisor_condition_l15_15567

-- Define conditions
def leaves_remainder (a b k : ℕ) : Prop := ∃ q : ℕ, a = b * q + k

-- Define the greatest common divisor property
def gcd_of (a b k: ℕ) (g : ℕ) : Prop :=
  leaves_remainder a k g ∧ leaves_remainder b k g ∧ ∀ d : ℕ, (leaves_remainder a k d ∧ leaves_remainder b k d) → d ≤ g

theorem greatest_divisor_condition 
  (N : ℕ) (h1 : leaves_remainder 1657 N 6) (h2 : leaves_remainder 2037 N 5) :
  N = 127 :=
sorry

end greatest_divisor_condition_l15_15567


namespace coins_remainder_divide_by_nine_remainder_l15_15855

def smallest_n (n : ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_remainder (n : ℕ) (h : smallest_n n) : (∃ m : ℕ, n = 54) :=
  sorry

theorem divide_by_nine_remainder (n : ℕ) (h : smallest_n n) (h_smallest: coins_remainder n h) : n % 9 = 0 :=
  sorry

end coins_remainder_divide_by_nine_remainder_l15_15855


namespace least_three_digit_multiple_13_l15_15833

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l15_15833


namespace product_of_five_consecutive_integers_not_perfect_square_l15_15633

theorem product_of_five_consecutive_integers_not_perfect_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = k * k :=
by {
  sorry
}

end product_of_five_consecutive_integers_not_perfect_square_l15_15633


namespace option_two_not_binomial_l15_15182

section
  variables {n : ℕ} {p : ℝ}

  -- Conditions for the distributions
  def computer_virus_distribution (X : ℕ → Prop) : Prop :=
    ∃ (n : ℕ), ∀ (X : ℕ), X = binomial n 0.65

  def first_hit_distribution (X : ℕ → Prop) : Prop :=
    ∀ (X : ℕ), X = geometric p

  def target_hits_distribution (X : ℕ → Prop) : Prop :=
    ∃ (n : ℕ), ∀ (X : ℕ), X = binomial n p

  def refueling_cars_distribution (X : ℕ → Prop) : Prop :=
    ∃ (k : ℕ), k = 50 ∧ ∀ (X : ℕ), X = binomial 50 0.6

  theorem option_two_not_binomial :
    (∃ (X : ℕ → Prop), computer_virus_distribution X) ∧
    (∃ (X : ℕ → Prop), first_hit_distribution X) ∧ 
    (∃ (X : ℕ → Prop), target_hits_distribution X) ∧ 
    (∃ (X : ℕ → Prop), refueling_cars_distribution X) →
    ∀ X, first_hit_distribution X → ¬ (∃ (X : ℕ → Prop), binomial X)
  :=
  sorry
end

end option_two_not_binomial_l15_15182


namespace number_of_valid_subsets_l15_15758

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def subset_is_prime (s : Finset ℕ) : Prop := ∀ n ∈ s, is_prime n

noncomputable def prime_subsets :=
  ((Finset.powerset (Finset.filter is_prime (Finset.range 10))).filter (λ s, 0 < s.card))

theorem number_of_valid_subsets : 
  prime_subsets.filter (λ s, (Finset.sum s id) > 10).card = 4 :=  
by 
  sorry

end number_of_valid_subsets_l15_15758


namespace calculate_dividend_l15_15659

def divisor : ℕ := 21
def quotient : ℕ := 14
def remainder : ℕ := 7
def expected_dividend : ℕ := 301

theorem calculate_dividend : (divisor * quotient + remainder = expected_dividend) := 
by
  sorry

end calculate_dividend_l15_15659


namespace problem_l15_15522

-- Define the conditions
variables (x y : ℝ)
axiom h1 : 2 * x + y = 7
axiom h2 : x + 2 * y = 5

-- Statement of the problem
theorem problem : (2 * x * y) / 3 = 2 :=
by 
  -- Proof is omitted, but you should replace 'sorry' by the actual proof
  sorry

end problem_l15_15522


namespace photos_to_cover_poster_l15_15458

/-
We are given a poster of dimensions 3 feet by 5 feet, and photos of dimensions 3 inches by 5 inches.
We need to prove that the number of such photos required to cover the poster is 144.
-/

-- Convert feet to inches
def feet_to_inches(feet : ℕ) : ℕ := 12 * feet

-- Dimensions of the poster in inches
def poster_height_in_inches := feet_to_inches 3
def poster_width_in_inches := feet_to_inches 5

-- Area of the poster
def poster_area : ℕ := poster_height_in_inches * poster_width_in_inches

-- Dimensions and area of one photo in inches
def photo_height := 3
def photo_width := 5
def photo_area : ℕ := photo_height * photo_width

-- Number of photos required to cover the poster
def number_of_photos : ℕ := poster_area / photo_area

-- Theorem stating the required number of photos is 144
theorem photos_to_cover_poster : number_of_photos = 144 := by
  -- Proof is omitted
  sorry

end photos_to_cover_poster_l15_15458


namespace sixth_graders_count_l15_15001

theorem sixth_graders_count (total_students seventh_graders_percentage sixth_graders_percentage : ℝ)
                            (seventh_graders_count : ℕ)
                            (h1 : seventh_graders_percentage = 0.32)
                            (h2 : seventh_graders_count = 64)
                            (h3 : sixth_graders_percentage = 0.38)
                            (h4 : seventh_graders_count = seventh_graders_percentage * total_students) :
                            sixth_graders_percentage * total_students = 76 := by
  sorry

end sixth_graders_count_l15_15001


namespace movie_production_cost_l15_15026

-- Definitions based on the conditions
def opening_revenue : ℝ := 120 -- in million dollars
def total_revenue : ℝ := 3.5 * opening_revenue -- movie made during its entire run
def kept_revenue : ℝ := 0.60 * total_revenue -- production company keeps 60% of total revenue
def profit : ℝ := 192 -- in million dollars

-- Theorem stating the cost to produce the movie
theorem movie_production_cost : 
  (kept_revenue - 60) = profit :=
by
  sorry

end movie_production_cost_l15_15026


namespace geometric_sequence_formula_and_sum_l15_15749

theorem geometric_sequence_formula_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, a (n+1) = 2 * a n) 
  (h_arith : a 1 = 2 ∧ 2 * (a 3 + 1) = a 1 + a 4)
  (h_b : ∀ n, b n = Nat.log2 (a n)) :
  (∀ n, a n = 2 ^ n) ∧ (S n = (n * (n + 1)) / 2) := 
by 
  sorry

end geometric_sequence_formula_and_sum_l15_15749


namespace find_interest_rate_l15_15187

noncomputable def interest_rate_solution : ℝ :=
  let P := 800
  let A := 1760
  let t := 4
  let n := 1
  (A / P) ^ (1 / (n * t)) - 1

theorem find_interest_rate : interest_rate_solution = 0.1892 := 
by
  sorry

end find_interest_rate_l15_15187


namespace product_of_five_consecutive_integers_not_square_l15_15635

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  let P := n * (n + 1) * (n + 2) * (n + 3) * (n + 4)
  ∀ k : ℕ, P ≠ k^2 := 
sorry

end product_of_five_consecutive_integers_not_square_l15_15635


namespace combination_10_3_l15_15702

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15702


namespace definitely_incorrect_conclusions_l15_15589

theorem definitely_incorrect_conclusions (a b c : ℝ) (x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : a * x2^2 + b * x2 + c = 0)
  (h3 : x1 > 0) 
  (h4 : x2 > 0) 
  (h5 : x1 + x2 = -b / a) 
  (h6 : x1 * x2 = c / a) : 
  (a > 0 ∧ b > 0 ∧ c > 0) = false ∧ 
  (a < 0 ∧ b < 0 ∧ c < 0) = false ∧ 
  (a > 0 ∧ b < 0 ∧ c < 0) = true ∧ 
  (a < 0 ∧ b > 0 ∧ c > 0) = true :=
sorry

end definitely_incorrect_conclusions_l15_15589


namespace cost_equation_l15_15860

variables (x y z : ℝ)

theorem cost_equation (h1 : 2 * x + y + 3 * z = 24) (h2 : 3 * x + 4 * y + 2 * z = 36) : x + y + z = 12 := by
  -- proof steps would go here, but are omitted as per instruction
  sorry

end cost_equation_l15_15860


namespace three_digit_problem_l15_15909

theorem three_digit_problem :
  ∃ (M Γ U : ℕ), 
    M ≠ Γ ∧ M ≠ U ∧ Γ ≠ U ∧
    M ≤ 9 ∧ Γ ≤ 9 ∧ U ≤ 9 ∧
    100 * M + 10 * Γ + U = (M + Γ + U) * (M + Γ + U - 2) ∧
    100 * M + 10 * Γ + U = 195 :=
by
  sorry

end three_digit_problem_l15_15909


namespace radius_of_circle_of_roots_l15_15190

theorem radius_of_circle_of_roots (z : ℂ)
  (h : (z + 2)^6 = 64 * z^6) :
  ∃ r : ℝ, r = 4 / 3 ∧ ∀ z, (z + 2)^6 = 64 * z^6 →
  abs (z + 2) = (4 / 3 : ℝ) * abs z :=
by
  sorry

end radius_of_circle_of_roots_l15_15190


namespace remainder_when_subtracted_l15_15844

theorem remainder_when_subtracted (s t : ℕ) (hs : s % 6 = 2) (ht : t % 6 = 3) (h : s > t) : (s - t) % 6 = 5 :=
by
  sorry -- Proof not required

end remainder_when_subtracted_l15_15844


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l15_15343

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l15_15343


namespace gcd_90_450_l15_15914

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l15_15914


namespace eval_power_expression_l15_15069

theorem eval_power_expression : (3^3)^2 / 3^2 = 81 := by
  sorry -- Proof omitted as instructed

end eval_power_expression_l15_15069


namespace ratio_of_x_y_l15_15617

theorem ratio_of_x_y (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 4) (h₃ : ∃ a b : ℤ, x = a * y / b ) (h₄ : x + y = 10) :
  x / y = -2 := sorry

end ratio_of_x_y_l15_15617


namespace number_of_herrings_l15_15904

theorem number_of_herrings (total_fishes pikes sturgeons herrings : ℕ)
  (h1 : total_fishes = 145)
  (h2 : pikes = 30)
  (h3 : sturgeons = 40)
  (h4 : total_fishes = pikes + sturgeons + herrings) :
  herrings = 75 :=
by
  sorry

end number_of_herrings_l15_15904


namespace calculate_F_5_f_6_l15_15616

def f (a : ℤ) : ℤ := a + 3

def F (a b : ℤ) : ℤ := b^3 - 2 * a

theorem calculate_F_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end calculate_F_5_f_6_l15_15616


namespace gcd_324_243_135_l15_15008

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_324_243_135_l15_15008


namespace set_list_method_l15_15203

theorem set_list_method : 
  {x : ℝ | x^2 - 2 * x + 1 = 0} = {1} :=
sorry

end set_list_method_l15_15203


namespace compute_binomial_10_3_eq_120_l15_15703

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15703


namespace correct_input_statement_l15_15513

-- Definitions based on the conditions
def input_format_A : Prop := sorry
def input_format_B : Prop := sorry
def input_format_C : Prop := sorry
def output_format_D : Prop := sorry

-- The main statement we need to prove
theorem correct_input_statement : input_format_A ∧ ¬ input_format_B ∧ ¬ input_format_C ∧ ¬ output_format_D := 
by sorry

end correct_input_statement_l15_15513


namespace Matt_income_from_plantation_l15_15774

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l15_15774


namespace max_value_of_a_l15_15750

variable {a : ℝ}

theorem max_value_of_a (h : a > 0) : 
  (∀ x : ℝ, x > 0 → (2 * x^2 - a * x + a > 0)) ↔ a ≤ 8 := 
sorry

end max_value_of_a_l15_15750


namespace weaving_problem_solution_l15_15525

noncomputable def daily_increase :=
  let a1 := 5
  let n := 30
  let sum_total := 390
  let d := (sum_total - a1 * n) * 2 / (n * (n - 1))
  d

theorem weaving_problem_solution :
  daily_increase = 16 / 29 :=
by
  sorry

end weaving_problem_solution_l15_15525


namespace evaluate_expression_l15_15795

theorem evaluate_expression : (2 * (-1) + 3) * (2 * (-1) - 3) - ((-1) - 1) * ((-1) + 5) = 3 := by
  sorry

end evaluate_expression_l15_15795


namespace anton_thought_number_l15_15041

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l15_15041


namespace zoo_initial_animals_l15_15028

theorem zoo_initial_animals (X : ℕ) :
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 :=
by
  intro h
  sorry

end zoo_initial_animals_l15_15028


namespace hyeyoung_walked_correct_l15_15983

/-- The length of the promenade near Hyeyoung's house is 6 kilometers (km). -/
def promenade_length : ℕ := 6

/-- Hyeyoung walked from the starting point to the halfway point of the trail. -/
def hyeyoung_walked : ℕ := promenade_length / 2

/-- The distance Hyeyoung walked is 3 kilometers (km). -/
theorem hyeyoung_walked_correct : hyeyoung_walked = 3 := by
  sorry

end hyeyoung_walked_correct_l15_15983


namespace rick_books_total_l15_15971

theorem rick_books_total 
  (N : ℕ)
  (h : N / 16 = 25) : 
  N = 400 := 
  sorry

end rick_books_total_l15_15971


namespace find_a_minus_b_l15_15504

theorem find_a_minus_b (a b c d : ℤ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 :=
sorry

end find_a_minus_b_l15_15504


namespace p_q_relation_n_le_2_p_q_relation_n_ge_3_l15_15952

open Real -- for ℝ
open Nat -- for ℕ

definition P (n : ℕ) (x : ℝ) : ℝ := (1-x)^(2*n-1)
definition Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n-1) * x + (n-1) * (2*n-1) * x^2

theorem p_q_relation_n_le_2 (n : ℕ+) (x : ℝ)
  (h_n_le_2 : n.val <= 2) : 
  if n.val = 1 then P n x = Q n x 
  else if x = 0 then P n x = Q n x
  else if x > 0 then P n x < Q n x
  else P n x > Q n x :=
sorry

theorem p_q_relation_n_ge_3 (n : ℕ+) (x : ℝ)
  (h_n_ge_3 : n.val >= 3) :
  if x = 0 then P n x = Q n x
  else if x > 0 then P n x < Q n x
  else P n x > Q n x :=
sorry

end p_q_relation_n_le_2_p_q_relation_n_ge_3_l15_15952


namespace work_days_l15_15843

theorem work_days (hp : ℝ) (hq : ℝ) (fraction_left : ℝ) (d : ℝ) :
  hp = 1 / 20 → hq = 1 / 10 → fraction_left = 0.7 → (3 / 20) * d = (1 - fraction_left) → d = 2 :=
  by
  intros hp_def hq_def fraction_def work_eq
  sorry

end work_days_l15_15843


namespace marketing_survey_l15_15532

theorem marketing_survey
  (H_neither : Nat := 80)
  (H_only_A : Nat := 60)
  (H_ratio_Both_to_Only_B : Nat := 3)
  (H_both : Nat := 25) :
  H_neither + H_only_A + (H_ratio_Both_to_Only_B * H_both) + H_both = 240 := 
sorry

end marketing_survey_l15_15532


namespace polynomial_value_at_five_l15_15652

def f (x : ℤ) : ℤ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem polynomial_value_at_five : f 5 = 2677 := by
  -- The proof goes here.
  sorry

end polynomial_value_at_five_l15_15652


namespace side_length_of_square_l15_15394

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l15_15394


namespace arithmetic_seq_sum_l15_15467

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℤ → ℤ) 
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) 
  (h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 110) : 
  S 15 = 330 := 
by
  sorry

end arithmetic_seq_sum_l15_15467


namespace range_of_a_l15_15751

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 < a ∧ a ≤ 3 :=
by {
  sorry
}

end range_of_a_l15_15751


namespace exceeds_threshold_at_8_l15_15895

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def exceeds_threshold (n : ℕ) : Prop :=
  geometric_sum 2 2 n ≥ 500

theorem exceeds_threshold_at_8 :
  ∀ n < 8, ¬exceeds_threshold n ∧ exceeds_threshold 8 :=
by
  sorry

end exceeds_threshold_at_8_l15_15895


namespace percentage_of_l15_15179

theorem percentage_of (part whole : ℕ) (h_part : part = 120) (h_whole : whole = 80) : 
  ((part : ℚ) / (whole : ℚ)) * 100 = 150 := 
by
  sorry

end percentage_of_l15_15179


namespace neighbors_receive_mangoes_l15_15963

-- Definitions of the conditions
def harvested_mangoes : ℕ := 560
def sold_mangoes : ℕ := harvested_mangoes / 2
def given_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Calculation of mangoes left
def mangoes_left : ℕ := harvested_mangoes - sold_mangoes - given_to_family

-- The statement we want to prove
theorem neighbors_receive_mangoes : mangoes_left / num_neighbors = 19 := by
  sorry

end neighbors_receive_mangoes_l15_15963


namespace symmetric_points_power_l15_15941

variables (m n : ℝ)

def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_power 
  (h : symmetric_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2023 = -1 :=
by 
  sorry

end symmetric_points_power_l15_15941


namespace sqrt_identity_l15_15585

def condition1 (α : ℝ) : Prop := 
  ∃ P : ℝ × ℝ, P = (Real.sin 2, Real.cos 2) ∧ Real.sin α = Real.cos 2

def condition2 (P : ℝ × ℝ) : Prop := 
  P.1 ^ 2 + P.2 ^ 2 = 1

theorem sqrt_identity (α : ℝ) (P : ℝ × ℝ) 
  (h₁ : condition1 α) (h₂ : condition2 P) : 
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by 
  sorry

end sqrt_identity_l15_15585


namespace trigonometric_identity_l15_15933

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := 
by sorry

end trigonometric_identity_l15_15933


namespace largest_int_less_than_100_by_7_l15_15233

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l15_15233


namespace polygon_sides_l15_15365

theorem polygon_sides (n : ℕ) (f : ℕ) (h1 : f = n * (n - 3) / 2) (h2 : 2 * n = f) : n = 7 :=
  by
  sorry

end polygon_sides_l15_15365


namespace group_friends_opponents_l15_15880

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l15_15880


namespace sqrt_eq_solutions_l15_15206

theorem sqrt_eq_solutions (x : ℝ) : 
  (Real.sqrt ((2 + Real.sqrt 5) ^ x) + Real.sqrt ((2 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end sqrt_eq_solutions_l15_15206


namespace probability_union_A_B_l15_15993

open Probability

-- Definitions of events A and B
def event_A (ω : ω) : Prop := coin_flip ω = Heads
def event_B (ω : ω) : Prop := die_roll ω = 3

-- Given conditions
axiom fair_coin : Probability(coin_flip = Heads) = 1 / 2
axiom fair_die : Probability(die_roll = 3) = 1 / 6
axiom independent_A_B : independent event_A event_B

-- Target proof statement
theorem probability_union_A_B : 
  Probability(event_A ∪ event_B) = 7 / 12 :=
sorry -- Proof is not required

end probability_union_A_B_l15_15993


namespace quad_function_one_zero_l15_15100

theorem quad_function_one_zero (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 6 * x + 1 = 0 ∧ (∀ x1 x2 : ℝ, m * x1^2 - 6 * x1 + 1 = 0 ∧ m * x2^2 - 6 * x2 + 1 = 0 → x1 = x2)) ↔ (m = 0 ∨ m = 9) :=
by
  sorry

end quad_function_one_zero_l15_15100


namespace shaded_area_of_hexagon_with_semicircles_l15_15281

theorem shaded_area_of_hexagon_with_semicircles :
  let s := 3
  let r := 3 / 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let semicircle_area := 3 * (1/2 * Real.pi * r^2)
  let shaded_area := hexagon_area - semicircle_area
  shaded_area = 13.5 * Real.sqrt 3 - 27 * Real.pi / 8 :=
by
  sorry

end shaded_area_of_hexagon_with_semicircles_l15_15281


namespace largest_multiple_of_9_less_than_100_l15_15163

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l15_15163


namespace Matt_income_from_plantation_l15_15773

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l15_15773


namespace algebraic_expression_evaluation_l15_15988

theorem algebraic_expression_evaluation (x y : ℤ) (h1 : x = -2) (h2 : y = -4) : 2 * x^2 - y + 3 = 15 :=
by
  rw [h1, h2]
  sorry

end algebraic_expression_evaluation_l15_15988


namespace binomial_coefficient_10_3_l15_15720

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15720


namespace problem_statement_l15_15268

noncomputable def f (x : ℝ) (A : ℝ) (ϕ : ℝ) : ℝ := A * Real.cos (2 * x + ϕ)

theorem problem_statement {A ϕ : ℝ} (hA : A > 0) (hϕ : |ϕ| < π / 2)
  (h1 : f (-π / 4) A ϕ = 2 * Real.sqrt 2)
  (h2 : f 0 A ϕ = 2 * Real.sqrt 6)
  (h3 : f (π / 12) A ϕ = 2 * Real.sqrt 2)
  (h4 : f (π / 4) A ϕ = -2 * Real.sqrt 2)
  (h5 : f (π / 3) A ϕ = -2 * Real.sqrt 6) :
  ϕ = π / 6 ∧ f (5 * π / 12) A ϕ = -4 * Real.sqrt 2 := 
sorry

end problem_statement_l15_15268


namespace remaining_books_l15_15006

def initial_books : Nat := 500
def num_people_donating : Nat := 10
def books_per_person : Nat := 8
def borrowed_books : Nat := 220

theorem remaining_books :
  (initial_books + num_people_donating * books_per_person - borrowed_books) = 360 := 
by 
  -- This will contain the mathematical proof
  sorry

end remaining_books_l15_15006


namespace sum_numerator_denominator_q_l15_15746

noncomputable def probability_q : ℚ := 32 / 70

theorem sum_numerator_denominator_q :
  (probability_q.num + probability_q.denom) = 51 :=
by
  -- Definitions and assumptions (corresponding to conditions)
  let a := {a_1, a_2, a_3, a_4 | a_1, a_2, a_3, a_4 ∈ (finset.range 1000).val}
  let b := {b_1, b_2, b_3, b_4 | b_1, b_2, b_3, b_4 ∈ (finset.range 1000).val \ a}
  -- Further definitions can be added as necessary, following the problem conditions

  -- Sorry is used to avoid proving manually
  sorry

end sum_numerator_denominator_q_l15_15746


namespace greatest_integer_gcd_l15_15340

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l15_15340


namespace division_rounded_nearest_hundredth_l15_15684

theorem division_rounded_nearest_hundredth :
  Float.round (285 * 387 / (981^2) * 100) / 100 = 0.11 :=
by
  sorry

end division_rounded_nearest_hundredth_l15_15684


namespace time_taken_by_alex_l15_15426

-- Define the conditions
def distance_per_lap : ℝ := 500 -- distance per lap in meters
def distance_first_part : ℝ := 150 -- first part of the distance in meters
def speed_first_part : ℝ := 3 -- speed for the first part in meters per second
def distance_second_part : ℝ := 350 -- remaining part of the distance in meters
def speed_second_part : ℝ := 4 -- speed for the remaining part in meters per second
def num_laps : ℝ := 4 -- number of laps run by Alex

-- Target time, expressed in seconds
def target_time : ℝ := 550 -- 9 minutes and 10 seconds is 550 seconds

-- Prove that given the conditions, the total time Alex takes to run 4 laps is 550 seconds
theorem time_taken_by_alex :
  (distance_first_part / speed_first_part + distance_second_part / speed_second_part) * num_laps = target_time :=
by
  sorry

end time_taken_by_alex_l15_15426


namespace original_height_in_feet_l15_15949

-- Define the current height in inches
def current_height_in_inches : ℚ := 180

-- Define the percentage increase in height
def percentage_increase : ℚ := 0.5

-- Define the conversion factor from inches to feet
def inches_to_feet : ℚ := 12

-- Define the initial height in inches
def initial_height_in_inches : ℚ := current_height_in_inches / (1 + percentage_increase)

-- Prove that the original height in feet was 10 feet
theorem original_height_in_feet : initial_height_in_inches / inches_to_feet = 10 :=
by
  -- Placeholder for the full proof
  sorry

end original_height_in_feet_l15_15949


namespace coins_remainder_l15_15858

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l15_15858


namespace largest_integer_lt_100_with_rem_4_div_7_l15_15209

theorem largest_integer_lt_100_with_rem_4_div_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 → m % 7 = 4 → m ≤ n := 
by
  sorry

end largest_integer_lt_100_with_rem_4_div_7_l15_15209


namespace subset_0_in_X_l15_15466

def X : Set ℝ := {x | x > -1}

theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l15_15466


namespace largest_multiple_of_9_less_than_100_l15_15168

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l15_15168


namespace current_speed_l15_15800

theorem current_speed (c : ℝ) :
  (∀ d1 t1 u v, d1 = 20 ∧ t1 = 2 ∧ u = 6 ∧ v = c → d1 = t1 * (u + v))
  ∧ (∀ d2 t2 u w, d2 = 4 ∧ t2 = 2 ∧ u = 6 ∧ w = c → d2 = t2 * (u - w)) 
  → c = 4 :=
by 
  intros
  sorry

end current_speed_l15_15800


namespace binomial_coefficient_10_3_l15_15719

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15719


namespace triangle_area_relation_l15_15888

theorem triangle_area_relation :
  let A := (1 / 2) * 5 * 5
  let B := (1 / 2) * 12 * 12
  let C := (1 / 2) * 13 * 13
  A + B = C :=
by
  sorry

end triangle_area_relation_l15_15888


namespace fraction_equiv_ratio_equiv_percentage_equiv_l15_15018

-- Define the problem's components and conditions.
def frac_1 : ℚ := 3 / 5
def frac_2 (a b : ℚ) : Prop := 3 / 5 = a / b
def ratio_1 (a b : ℚ) : Prop := 10 / a = b / 100
def percentage_1 (a b : ℚ) : Prop := (a / b) * 100 = 60

-- Problem statement 1: Fraction equality
theorem fraction_equiv : frac_2 12 20 := 
by sorry

-- Problem statement 2: Ratio equality
theorem ratio_equiv : ratio_1 (50 / 3) 60 := 
by sorry

-- Problem statement 3: Percentage equality
theorem percentage_equiv : percentage_1 60 100 := 
by sorry

end fraction_equiv_ratio_equiv_percentage_equiv_l15_15018


namespace find_r_l15_15120

-- Declaring the roots of the first polynomial
variables (a b m : ℝ)
-- Declaring the roots of the second polynomial
variables (p r : ℝ)

-- Assumptions based on the given conditions
def roots_of_first_eq : Prop :=
  a + b = m ∧ a * b = 3

def roots_of_second_eq : Prop :=
  ∃ (p : ℝ), (a^2 + 1/b) * (b^2 + 1/a) = r

-- The desired theorem
theorem find_r 
  (h1 : roots_of_first_eq a b m)
  (h2 : (a^2 + 1/b) * (b^2 + 1/a) = r) :
  r = 46/3 := by sorry

end find_r_l15_15120


namespace Carla_more_miles_than_Daniel_after_5_hours_l15_15141

theorem Carla_more_miles_than_Daniel_after_5_hours (Carla_distance : ℝ) (Daniel_distance : ℝ) (h_Carla : Carla_distance = 100) (h_Daniel : Daniel_distance = 75) : 
  Carla_distance - Daniel_distance = 25 := 
by
  sorry

end Carla_more_miles_than_Daniel_after_5_hours_l15_15141


namespace train_length_l15_15512

noncomputable def length_of_each_train : ℝ :=
  let speed_faster_train_km_per_hr := 46
  let speed_slower_train_km_per_hr := 36
  let relative_speed_km_per_hr := speed_faster_train_km_per_hr - speed_slower_train_km_per_hr
  let relative_speed_m_per_s := (relative_speed_km_per_hr * 1000) / 3600
  let time_s := 54
  let distance_m := relative_speed_m_per_s * time_s
  distance_m / 2

theorem train_length : length_of_each_train = 75 := by
  sorry

end train_length_l15_15512


namespace tank_capacity_l15_15145

theorem tank_capacity (one_third_full : ℚ) (added_water : ℚ) (capacity : ℚ) 
  (h1 : one_third_full = 1 / 3) 
  (h2 : 2 * one_third_full * capacity = 16) 
  (h3 : added_water = 16) 
  : capacity = 24 := 
by
  sorry

end tank_capacity_l15_15145


namespace problem_xy_minimized_problem_x_y_minimized_l15_15917

open Real

theorem problem_xy_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 16 ∧ y = 2 ∧ x * y = 32 := 
sorry

theorem problem_x_y_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 8 + 2 * sqrt 2 ∧ y = 1 + sqrt 2 ∧ x + y = 9 + 4 * sqrt 2 := 
sorry

end problem_xy_minimized_problem_x_y_minimized_l15_15917


namespace principal_amount_l15_15496

theorem principal_amount
  (P : ℝ)
  (r : ℝ := 0.05)
  (t : ℝ := 2)
  (H : P * (1 + r)^t - P - P * r * t = 17) :
  P = 6800 :=
by sorry

end principal_amount_l15_15496


namespace fraction_not_simplifiable_l15_15306

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_not_simplifiable_l15_15306


namespace quadratic_eq_coeff_m_l15_15497

theorem quadratic_eq_coeff_m (m : ℤ) : 
  (|m| = 2 ∧ m + 2 ≠ 0) → m = 2 := 
by
  intro h
  sorry

end quadratic_eq_coeff_m_l15_15497


namespace net_profit_from_plant_sales_l15_15552

noncomputable def calculate_net_profit : ℝ :=
  let cost_basil := 2.00
  let cost_mint := 3.00
  let cost_zinnia := 7.00
  let cost_soil := 15.00
  let total_cost := cost_basil + cost_mint + cost_zinnia + cost_soil
  let basil_germinated := 20 * 0.80
  let mint_germinated := 15 * 0.75
  let zinnia_germinated := 10 * 0.70
  let revenue_healthy_basil := 12 * 5.00
  let revenue_small_basil := 8 * 3.00
  let revenue_healthy_mint := 10 * 6.00
  let revenue_small_mint := 4 * 4.00
  let revenue_healthy_zinnia := 5 * 10.00
  let revenue_small_zinnia := 2 * 7.00
  let total_revenue := revenue_healthy_basil + revenue_small_basil + revenue_healthy_mint + revenue_small_mint + revenue_healthy_zinnia + revenue_small_zinnia
  total_revenue - total_cost

theorem net_profit_from_plant_sales : calculate_net_profit = 197.00 := by
  sorry

end net_profit_from_plant_sales_l15_15552


namespace find_y_eq_1_div_5_l15_15460

theorem find_y_eq_1_div_5 (b : ℝ) (y : ℝ) (h1 : b > 2) (h2 : y > 0) (h3 : (3 * y)^(Real.log 3 / Real.log b) - (5 * y)^(Real.log 5 / Real.log b) = 0) :
  y = 1 / 5 :=
by
  sorry

end find_y_eq_1_div_5_l15_15460


namespace degree_of_d_l15_15377

theorem degree_of_d (f d q r : Polynomial ℝ) (f_deg : f.degree = 17)
  (q_deg : q.degree = 10) (r_deg : r.degree = 4) 
  (remainder : r = Polynomial.C 5 * X^4 - Polynomial.C 3 * X^3 + Polynomial.C 2 * X^2 - X + 15)
  (div_relation : f = d * q + r) (r_deg_lt_d_deg : r.degree < d.degree) :
  d.degree = 7 :=
sorry

end degree_of_d_l15_15377


namespace partitions_distinct_parts_eq_odd_parts_l15_15308

def num_partitions_into_distinct_parts (n : ℕ) : ℕ := sorry
def num_partitions_into_odd_parts (n : ℕ) : ℕ := sorry

theorem partitions_distinct_parts_eq_odd_parts (n : ℕ) :
  num_partitions_into_distinct_parts n = num_partitions_into_odd_parts n :=
  sorry

end partitions_distinct_parts_eq_odd_parts_l15_15308


namespace rational_number_theorem_l15_15277

theorem rational_number_theorem (x y : ℚ) 
  (h1 : |(x + 2017 : ℚ)| + (y - 2017) ^ 2 = 0) : 
  (x / y) ^ 2017 = -1 := 
by
  sorry

end rational_number_theorem_l15_15277


namespace quadruplets_satisfy_l15_15569

-- Define the condition in the problem
def equation (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

-- State the theorem
theorem quadruplets_satisfy (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  equation x y z w ↔ (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) :=
by
  sorry

end quadruplets_satisfy_l15_15569


namespace geometric_seq_ratio_l15_15586

theorem geometric_seq_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a (n+1) = q * a n)
  (h2 : 0 < q)                    -- ensuring positivity
  (h3 : 3 * a 0 + 2 * q * a 0 = q^2 * a 0)  -- condition from problem
  : ∀ n, (a (n+3) + a (n+2)) / (a (n+1) + a n) = 9 :=
by
  sorry

end geometric_seq_ratio_l15_15586


namespace side_length_of_square_l15_15406

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l15_15406


namespace range_of_p_l15_15091

-- Conditions: p is a prime number and the roots of the quadratic equation are integers 
def p_is_prime (p : ℕ) : Prop := Nat.Prime p

def roots_are_integers (p : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x * y = -204 * p ∧ (x + y) = p

-- Main statement: Prove the range of p
theorem range_of_p (p : ℕ) (hp : p_is_prime p) (hr : roots_are_integers p) : 11 < p ∧ p ≤ 21 :=
  sorry

end range_of_p_l15_15091


namespace percentage_of_part_l15_15022

theorem percentage_of_part (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 50) : (Part / Whole) * 100 = 240 := 
by
  sorry

end percentage_of_part_l15_15022


namespace binary_representation_of_38_l15_15059

theorem binary_representation_of_38 : ∃ binary : ℕ, binary = 0b100110 ∧ binary = 38 :=
by
  sorry

end binary_representation_of_38_l15_15059


namespace m_minus_n_eq_2_l15_15150

theorem m_minus_n_eq_2 (m n : ℕ) (h1 : ∃ x : ℕ, m = 101 * x) (h2 : ∃ y : ℕ, n = 63 * y) (h3 : m + n = 2018) : m - n = 2 :=
sorry

end m_minus_n_eq_2_l15_15150


namespace geometric_sequence_problem_l15_15919

variable (a : ℕ → ℝ)
variable (r : ℝ) (hpos : ∀ n, 0 < a n)

theorem geometric_sequence_problem
  (hgeom : ∀ n, a (n+1) = a n * r)
  (h_eq : a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) :
  a 2 + a 6 = 2 :=
sorry

end geometric_sequence_problem_l15_15919


namespace monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l15_15058

noncomputable def f (x : ℝ) := Real.exp x - (1 / 2) * x^2 - x - 1
noncomputable def f' (x : ℝ) := Real.exp x - x - 1
noncomputable def f'' (x : ℝ) := Real.exp x - 1
noncomputable def g (x : ℝ) := -f (-x)

-- Proof of (I)
theorem monotonic_intervals_and_extreme_values_of_f' :
  f' 0 = 0 ∧ (∀ x < 0, f'' x < 0 ∧ f' x > f' 0) ∧ (∀ x > 0, f'' x > 0 ∧ f' x > f' 0) := 
sorry

-- Proof of (II)
theorem f_g_inequality (x : ℝ) (hx : x > 0) : f x > g x :=
sorry

-- Proof of (III)
theorem sum_of_x1_x2 (x1 x2 : ℝ) (h : f x1 + f x2 = 0) (hne : x1 ≠ x2) : x1 + x2 < 0 := 
sorry

end monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l15_15058


namespace tan_beta_value_l15_15261

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = -3 / 4) (h2 : Real.tan (α + β) = 1) : Real.tan β = 7 :=
sorry

end tan_beta_value_l15_15261


namespace Travis_annual_cereal_cost_l15_15820

def cost_of_box_A : ℚ := 2.50
def cost_of_box_B : ℚ := 3.50
def cost_of_box_C : ℚ := 4.00
def cost_of_box_D : ℚ := 5.25
def cost_of_box_E : ℚ := 6.00

def quantity_of_box_A : ℚ := 1
def quantity_of_box_B : ℚ := 0.5
def quantity_of_box_C : ℚ := 0.25
def quantity_of_box_D : ℚ := 0.75
def quantity_of_box_E : ℚ := 1.5

def cost_week1 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week2 : ℚ :=
  let subtotal := 
    cost_of_box_A * quantity_of_box_A +
    cost_of_box_B * quantity_of_box_B +
    cost_of_box_C * quantity_of_box_C +
    cost_of_box_D * quantity_of_box_D +
    cost_of_box_E * quantity_of_box_E
  subtotal * 0.8

def cost_week3 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  0 +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week4 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  let discounted_box_E := cost_of_box_E * quantity_of_box_E * 0.85
  cost_of_box_A * quantity_of_box_A +
  discounted_box_E
  
def monthly_cost : ℚ :=
  cost_week1 + cost_week2 + cost_week3 + cost_week4

def annual_cost : ℚ :=
  monthly_cost * 12

theorem Travis_annual_cereal_cost :
  annual_cost = 792.24 := by
  sorry

end Travis_annual_cereal_cost_l15_15820


namespace polynomial_factorization_l15_15367

theorem polynomial_factorization (x y : ℝ) : -(2 * x - y) * (2 * x + y) = -4 * x ^ 2 + y ^ 2 :=
by sorry

end polynomial_factorization_l15_15367


namespace translate_B_to_origin_l15_15609

structure Point where
  x : ℝ
  y : ℝ

def translate_right (p : Point) (d : ℕ) : Point := 
  { x := p.x + d, y := p.y }

theorem translate_B_to_origin :
  ∀ (A B : Point) (d : ℕ),
  A = { x := -4, y := 0 } →
  B = { x := 0, y := 2 } →
  (translate_right A d).x = 0 →
  translate_right B d = { x := 4, y := 2 } :=
by
  intros A B d hA hB hA'
  sorry

end translate_B_to_origin_l15_15609


namespace f_minimum_positive_period_and_max_value_l15_15148

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) + (1 + (Real.tan x)^2) * (Real.cos x)^2

theorem f_minimum_positive_period_and_max_value :
  (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) ∧ (∃ M, ∀ x : ℝ, f x ≤ M ∧ M = 3 / 2) := by
  sorry

end f_minimum_positive_period_and_max_value_l15_15148


namespace moles_of_SO2_formed_l15_15439

variable (n_NaHSO3 n_HCl n_SO2 : ℕ)

/--
The reaction between sodium bisulfite (NaHSO3) and hydrochloric acid (HCl) is:
NaHSO3 + HCl → NaCl + H2O + SO2
Given 2 moles of NaHSO3 and 2 moles of HCl, prove that the number of moles of SO2 formed is 2.
-/
theorem moles_of_SO2_formed :
  (n_NaHSO3 = 2) →
  (n_HCl = 2) →
  (∀ (n : ℕ), (n_NaHSO3 = n) → (n_HCl = n) → (n_SO2 = n)) →
  n_SO2 = 2 :=
by 
  intros hNaHSO3 hHCl hReaction
  exact hReaction 2 hNaHSO3 hHCl

end moles_of_SO2_formed_l15_15439


namespace increase_productivity_RnD_l15_15054

theorem increase_productivity_RnD :
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  RnD_t / ΔAPL_t2 = 3260 :=
by
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  have h : RnD_t / ΔAPL_t2 = 3260 := sorry
  exact h

end increase_productivity_RnD_l15_15054


namespace number_of_times_difference_fits_is_20_l15_15789

-- Definitions for Ralph's pictures
def ralph_wild_animals := 75
def ralph_landscapes := 36
def ralph_family_events := 45
def ralph_cars := 20
def ralph_total_pictures := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars

-- Definitions for Derrick's pictures
def derrick_wild_animals := 95
def derrick_landscapes := 42
def derrick_family_events := 55
def derrick_cars := 25
def derrick_airplanes := 10
def derrick_total_pictures := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Combined total number of pictures
def combined_total_pictures := ralph_total_pictures + derrick_total_pictures

-- Difference in wild animals pictures
def difference_wild_animals := derrick_wild_animals - ralph_wild_animals

-- Number of times the difference fits into the combined total (rounded down)
def times_difference_fits := combined_total_pictures / difference_wild_animals

-- Statement of the problem
theorem number_of_times_difference_fits_is_20 : times_difference_fits = 20 := by
  -- The proof will be written here
  sorry

end number_of_times_difference_fits_is_20_l15_15789


namespace group_of_friends_l15_15866

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l15_15866


namespace binomial_coefficient_10_3_l15_15721

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15721


namespace w_share_l15_15421

theorem w_share (k : ℝ) (w x y z : ℝ) (h1 : w = k) (h2 : x = 6 * k) (h3 : y = 2 * k) (h4 : z = 4 * k) (h5 : x - y = 1500):
  w = 375 := by
  /- Lean code to show w = 375 -/
  sorry

end w_share_l15_15421


namespace remainder_div_1442_l15_15146

theorem remainder_div_1442 (x k l r : ℤ) (h1 : 1816 = k * x + 6) (h2 : 1442 = l * x + r) (h3 : x = Int.gcd 1810 374) : r = 0 := by
  sorry

end remainder_div_1442_l15_15146


namespace circumcircle_circumference_thm_triangle_perimeter_thm_l15_15954

-- Definition and theorem for the circumference of the circumcircle
def circumcircle_circumference (a b c R : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * R = c / (Real.sqrt (1 - cosC^2)) 
  ∧ 2 * R * Real.pi = 3 * Real.pi

theorem circumcircle_circumference_thm (a b c R : ℝ) (cosC : ℝ) :
  circumcircle_circumference a b c R cosC → 2 * R * Real.pi = 3 * Real.pi :=
by
  intro h;
  sorry

-- Definition and theorem for the perimeter of the triangle
def triangle_perimeter (a b c : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * a = 3 * b ∧ (a + b + c) = 5 + Real.sqrt 5

theorem triangle_perimeter_thm (a b c : ℝ) (cosC : ℝ) :
  triangle_perimeter a b c cosC → (a + b + c) = 5 + Real.sqrt 5 :=
by
  intro h;
  sorry

end circumcircle_circumference_thm_triangle_perimeter_thm_l15_15954


namespace find_polynomial_l15_15618

-- Define the polynomial conditions
structure CubicPolynomial :=
  (P : ℝ → ℝ)
  (P0 : ℝ)
  (P1 : ℝ)
  (P2 : ℝ)
  (P3 : ℝ)
  (cubic_eq : ∀ x, P x = P0 + P1 * x + P2 * x^2 + P3 * x^3)

theorem find_polynomial (P : CubicPolynomial) (h_neg1 : P.P (-1) = 2) (h0 : P.P 0 = 3) (h1 : P.P 1 = 1) (h2 : P.P 2 = 15) :
  ∀ x, P.P x = 3 + x - 2 * x^2 - x^3 :=
sorry

end find_polynomial_l15_15618


namespace find_number_l15_15847

theorem find_number (x : ℝ) (h : 5 * 1.6 - (2 * 1.4) / x = 4) : x = 0.7 :=
by
  sorry

end find_number_l15_15847


namespace hyperbola_sufficient_asymptotes_l15_15806

open Real

def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptotes_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

theorem hyperbola_sufficient_asymptotes (a b x y : ℝ) :
  (hyperbola_eq a b x y) → (asymptotes_eq a b x y) :=
by
  sorry

end hyperbola_sufficient_asymptotes_l15_15806


namespace largest_multiple_of_9_less_than_100_l15_15161

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l15_15161


namespace true_propositions_in_reverse_neg_neg_reverse_l15_15809

theorem true_propositions_in_reverse_neg_neg_reverse (a b : ℕ) : 
  (¬ (a ≠ 0 → a * b ≠ 0) ∧ ∃ (a : ℕ), (a = 0 ∧ a * b ≠ 0) ∨ (a ≠ 0 ∧ a * b = 0) ∧ ¬ (¬ ∃ (a : ℕ), a ≠ 0 ∧ a * b ≠ 0 ∧ ¬ ∃ (a : ℕ), a = 0 ∧ a * b = 0)) ∧ (0 = 1) :=
by {
  sorry
}

end true_propositions_in_reverse_neg_neg_reverse_l15_15809


namespace bubble_bath_amount_l15_15289

noncomputable def total_bubble_bath_needed 
  (couple_rooms : ℕ) (single_rooms : ℕ) (people_per_couple_room : ℕ) (people_per_single_room : ℕ) (ml_per_bath : ℕ) : ℕ :=
  couple_rooms * people_per_couple_room * ml_per_bath + single_rooms * people_per_single_room * ml_per_bath

theorem bubble_bath_amount :
  total_bubble_bath_needed 13 14 2 1 10 = 400 := by 
  sorry

end bubble_bath_amount_l15_15289


namespace part_a_part_b_l15_15293

def g (n : ℕ) : ℕ := (n.digits 10).prod

theorem part_a : ∀ n : ℕ, g n ≤ n :=
by
  -- Proof omitted
  sorry

theorem part_b : {n : ℕ | n^2 - 12*n + 36 = g n} = {4, 9} :=
by
  -- Proof omitted
  sorry

end part_a_part_b_l15_15293


namespace parametric_line_eq_l15_15495

-- Define the parameterized functions for x and y 
def parametric_x (t : ℝ) : ℝ := 3 * t + 7
def parametric_y (t : ℝ) : ℝ := 5 * t - 8

-- Define the equation of the line (here it's a relation that relates x and y)
def line_equation (x y : ℝ) : Prop := 
  y = (5 / 3) * x - (59 / 3)

theorem parametric_line_eq : 
  ∃ t : ℝ, line_equation (parametric_x t) (parametric_y t) := 
by
  -- Proof goes here
  sorry

end parametric_line_eq_l15_15495


namespace side_length_of_square_l15_15393

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l15_15393


namespace max_value_a_l15_15588

theorem max_value_a (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + y = 1) : 
  ∃ a, a = 16 ∧ (∀ x y, (x > 0 → y > 0 → x + y = 1 → a ≤ (1/x) + (9/y))) :=
by 
  use 16
  sorry

end max_value_a_l15_15588


namespace sum_of_coordinates_reflection_l15_15128

theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C := (3, y)
  let D := (3, -y)
  (C.1 + C.2 + D.1 + D.2) = 6 :=
by
  let C := (3, y)
  let D := (3, -y)
  have h : C.1 + C.2 + D.1 + D.2 = 6 := sorry
  exact h

end sum_of_coordinates_reflection_l15_15128


namespace ratio_of_money_earned_l15_15482

variable (L T J : ℕ) 

theorem ratio_of_money_earned 
  (total_earned : L + T + J = 60)
  (lisa_earning : L = 30)
  (lisa_tommy_diff : L = T + 15) : 
  T / L = 1 / 2 := 
by
  sorry

end ratio_of_money_earned_l15_15482


namespace cars_served_from_4pm_to_6pm_l15_15051

theorem cars_served_from_4pm_to_6pm : 
  let cars_per_15_min_peak := 12
  let cars_per_15_min_offpeak := 8 
  let blocks_in_an_hour := 4 
  let total_peak_hour := cars_per_15_min_peak * blocks_in_an_hour 
  let total_offpeak_hour := cars_per_15_min_offpeak * blocks_in_an_hour 
  total_peak_hour + total_offpeak_hour = 80 := 
by 
  sorry 

end cars_served_from_4pm_to_6pm_l15_15051


namespace number_of_sunflowers_l15_15768

noncomputable def cost_per_red_rose : ℝ := 1.5
noncomputable def cost_per_sunflower : ℝ := 3
noncomputable def total_cost : ℝ := 45
noncomputable def cost_of_red_roses : ℝ := 24 * cost_per_red_rose
noncomputable def money_left_for_sunflowers : ℝ := total_cost - cost_of_red_roses

theorem number_of_sunflowers :
  (money_left_for_sunflowers / cost_per_sunflower) = 3 :=
by
  sorry

end number_of_sunflowers_l15_15768


namespace xy_value_l15_15956

structure Point (R : Type) := (x : R) (y : R)

def A : Point ℝ := ⟨2, 7⟩ 
def C : Point ℝ := ⟨4, 3⟩ 

def is_midpoint (A B C : Point ℝ) : Prop :=
  (C.x = (A.x + B.x) / 2) ∧ (C.y = (A.y + B.y) / 2)

theorem xy_value (x y : ℝ) (B : Point ℝ := ⟨x, y⟩) (H : is_midpoint A B C) :
  x * y = -6 := 
sorry

end xy_value_l15_15956


namespace melted_ice_cream_depth_l15_15677

noncomputable def radius_sphere : ℝ := 3
noncomputable def radius_cylinder : ℝ := 10
noncomputable def height_cylinder : ℝ := 36 / 100

theorem melted_ice_cream_depth :
  (4 / 3) * Real.pi * radius_sphere^3 = Real.pi * radius_cylinder^2 * height_cylinder :=
by
  sorry

end melted_ice_cream_depth_l15_15677


namespace least_three_digit_multiple_of_13_l15_15831

-- Define what it means to be a multiple of 13
def is_multiple_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13 * k

-- Define the range of three-digit numbers
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define our main theorem
theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), is_three_digit_number n ∧ is_multiple_of_13 n ∧
  ∀ (m : ℕ), is_three_digit_number m ∧ is_multiple_of_13 m → n ≤ m :=
begin
  -- We state the theorem without proof for simplicity
  sorry
end

end least_three_digit_multiple_of_13_l15_15831


namespace fraction_of_seniors_study_japanese_l15_15681

variable (J S : ℝ)
variable (fraction_seniors fraction_juniors : ℝ)
variable (total_fraction_study_japanese : ℝ)

theorem fraction_of_seniors_study_japanese 
  (h1 : S = 2 * J)
  (h2 : fraction_juniors = 3 / 4)
  (h3 : total_fraction_study_japanese = 1 / 3) :
  fraction_seniors = 1 / 8 :=
by
  -- Here goes the proof.
  sorry

end fraction_of_seniors_study_japanese_l15_15681


namespace trisha_spent_on_eggs_l15_15327

def totalSpent (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ) : ℕ :=
  initialAmount - (meat + chicken + veggies + dogFood + amountLeft)

theorem trisha_spent_on_eggs :
  ∀ (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ),
    meat = 17 →
    chicken = 22 →
    veggies = 43 →
    dogFood = 45 →
    amountLeft = 35 →
    initialAmount = 167 →
    totalSpent meat chicken veggies eggs dogFood amountLeft initialAmount = 5 :=
by
  intros meat chicken veggies eggs dogFood amountLeft initialAmount
  sorry

end trisha_spent_on_eggs_l15_15327


namespace find_Y_l15_15762

-- Definition of the problem.
def arithmetic_sequence (a d n : ℕ) : ℕ := a + d * (n - 1)

-- Conditions provided in the problem.
-- Conditions of the first row
def first_row (a₁ a₄ : ℕ) : Prop :=
  a₁ = 4 ∧ a₄ = 16

-- Conditions of the last row
def last_row (a₁' a₄' : ℕ) : Prop :=
  a₁' = 10 ∧ a₄' = 40

-- Value of Y (the second element of the second row from the second column)
def center_top_element (Y : ℕ) : Prop :=
  Y = 12

-- The theorem to prove.
theorem find_Y (a₁ a₄ a₁' a₄' Y : ℕ) (h1 : first_row a₁ a₄) (h2 : last_row a₁' a₄') (h3 : center_top_element Y) : Y = 12 := 
by 
  sorry -- proof to be provided.

end find_Y_l15_15762


namespace sum_of_series_equals_one_half_l15_15197

theorem sum_of_series_equals_one_half : 
  (∑' k : ℕ, (1 / ((2 * k + 1) * (2 * k + 3)))) = 1 / 2 :=
sorry

end sum_of_series_equals_one_half_l15_15197


namespace eval_five_over_two_l15_15314

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 2 else Real.log (x - 1) / Real.log 2

theorem eval_five_over_two : f (5 / 2) = -1 := by
  sorry

end eval_five_over_two_l15_15314


namespace minimum_value_sine_shift_l15_15143

theorem minimum_value_sine_shift :
  ∀ (f : ℝ → ℝ) (φ : ℝ), (∀ x, f x = Real.sin (2 * x + φ)) → |φ| < Real.pi / 2 →
  (∀ x, f (x + Real.pi / 6) = f (-x)) →
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = - Real.sqrt 3 / 2 :=
by
  sorry

end minimum_value_sine_shift_l15_15143


namespace part1_part2_l15_15927

-- Define the quadratic equation and its discriminant
def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

-- Define the conditions
def quadratic_equation (m : ℝ) : ℝ :=
  quadratic_discriminant 1 (-2) (-3 * m^2)

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem part1 (m : ℝ) : 
  quadratic_equation m > 0 :=
by
  sorry

-- Part 2: Find the value of m given the roots satisfy the equation α + 2β = 5
theorem part2 (α β m : ℝ) (h1 : α + β = 2) (h2 : α + 2 * β = 5) : 
  m = 1 ∨ m = -1 :=
by
  sorry


end part1_part2_l15_15927


namespace P_inter_M_l15_15020

def set_P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def set_M : Set ℝ := {x | x^2 ≤ 9}

theorem P_inter_M :
  set_P ∩ set_M = {x | 0 ≤ x ∧ x < 3} := sorry

end P_inter_M_l15_15020


namespace money_per_percentage_point_l15_15967

theorem money_per_percentage_point
  (plates : ℕ) (total_states : ℕ) (total_amount : ℤ)
  (h_plates : plates = 40) (h_total_states : total_states = 50) (h_total_amount : total_amount = 160) :
  total_amount / (plates * 100 / total_states) = 2 :=
by
  -- Omitted steps of the proof
  sorry

end money_per_percentage_point_l15_15967


namespace find_a_plus_b_l15_15269

noncomputable def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1 / 3
noncomputable def f_prime (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a^2

theorem find_a_plus_b 
  (a b : ℝ)
  (h_deriv : f_prime a b 1 = 0)
  (h_extreme : f a b 1 = 0) :
  a + b = -7 / 9 := 
sorry

end find_a_plus_b_l15_15269


namespace binomial_coefficient_10_3_l15_15717

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15717


namespace balance_after_6_months_l15_15894

noncomputable def final_balance : ℝ :=
  let balance_m1 := 5000 * (1 + 0.04 / 12)
  let balance_m2 := (balance_m1 + 1000) * (1 + 0.042 / 12)
  let balance_m3 := balance_m2 * (1 + 0.038 / 12)
  let balance_m4 := (balance_m3 - 1500) * (1 + 0.05 / 12)
  let balance_m5 := (balance_m4 + 750) * (1 + 0.052 / 12)
  let balance_m6 := (balance_m5 - 1000) * (1 + 0.045 / 12)
  balance_m6

theorem balance_after_6_months : final_balance = 4371.51 := sorry

end balance_after_6_months_l15_15894


namespace balance_of_diamondsuits_and_bullets_l15_15072

variable (a b c : ℕ)

theorem balance_of_diamondsuits_and_bullets 
  (h1 : 4 * a + 2 * b = 12 * c)
  (h2 : a = b + 3 * c) :
  3 * b = 6 * c := 
sorry

end balance_of_diamondsuits_and_bullets_l15_15072


namespace perpendicular_lines_l15_15317

theorem perpendicular_lines (m : ℝ) :
  (m+2)*(m-1) + m*(m-4) = 0 ↔ m = 2 ∨ m = -1/2 :=
by 
  sorry

end perpendicular_lines_l15_15317


namespace smallest_number_divisible_conditions_l15_15849

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l15_15849


namespace power_equiv_l15_15326

theorem power_equiv (x_0 : ℝ) (h : x_0 ^ 11 + x_0 ^ 7 + x_0 ^ 3 = 1) : x_0 ^ 4 + x_0 ^ 3 - 1 = x_0 ^ 15 :=
by
  -- the proof goes here
  sorry

end power_equiv_l15_15326


namespace mrs_franklin_initial_valentines_l15_15484

theorem mrs_franklin_initial_valentines (v g l : ℕ) (h1 : g = 42) (h2 : l = 16) (h3 : v = g + l) : v = 58 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end mrs_franklin_initial_valentines_l15_15484


namespace possible_number_of_friends_l15_15871

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l15_15871


namespace side_length_of_square_l15_15404

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l15_15404


namespace geologists_probability_l15_15284

theorem geologists_probability
  (n roads : ℕ) (speed_per_hour : ℕ) 
  (angle_between_neighbors : ℕ)
  (distance_limit : ℝ) : 
  n = 6 ∧ speed_per_hour = 4 ∧ angle_between_neighbors = 60 ∧ distance_limit = 6 → 
  prob_distance_at_least_6_km = 0.5 :=
by
  sorry

noncomputable def prob_distance_at_least_6_km : ℝ := 0.5  -- Placeholder definition

end geologists_probability_l15_15284


namespace batsman_average_30_matches_l15_15492

theorem batsman_average_30_matches (avg_20_matches : ℕ -> ℚ) (avg_10_matches : ℕ -> ℚ)
  (h1 : avg_20_matches 20 = 40)
  (h2 : avg_10_matches 10 = 20)
  : (20 * (avg_20_matches 20) + 10 * (avg_10_matches 10)) / 30 = 33.33 := by
  sorry

end batsman_average_30_matches_l15_15492


namespace total_pamphlets_correct_l15_15626

-- Define the individual printing rates and hours
def Mike_pre_break_rate := 600
def Mike_pre_break_hours := 9
def Mike_post_break_rate := Mike_pre_break_rate / 3
def Mike_post_break_hours := 2

def Leo_pre_break_rate := 2 * Mike_pre_break_rate
def Leo_pre_break_hours := Mike_pre_break_hours / 3
def Leo_post_first_break_rate := Leo_pre_break_rate / 2
def Leo_post_second_break_rate := Leo_post_first_break_rate / 2

def Sally_pre_break_rate := 3 * Mike_pre_break_rate
def Sally_pre_break_hours := Mike_post_break_hours / 2
def Sally_post_break_rate := Leo_post_first_break_rate
def Sally_post_break_hours := 1

-- Calculate the total number of pamphlets printed by each person
def Mike_pamphlets := 
  (Mike_pre_break_rate * Mike_pre_break_hours) + (Mike_post_break_rate * Mike_post_break_hours)

def Leo_pamphlets := 
  (Leo_pre_break_rate * 1) + (Leo_post_first_break_rate * 1) + (Leo_post_second_break_rate * 1)

def Sally_pamphlets := 
  (Sally_pre_break_rate * Sally_pre_break_hours) + (Sally_post_break_rate * Sally_post_break_hours)

-- Calculate the total number of pamphlets printed by all three
def total_pamphlets := Mike_pamphlets + Leo_pamphlets + Sally_pamphlets

theorem total_pamphlets_correct : total_pamphlets = 10700 := by
  sorry

end total_pamphlets_correct_l15_15626


namespace binom_10_3_l15_15693

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15693


namespace friends_game_l15_15875

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l15_15875


namespace sum_of_given_numbers_l15_15812

theorem sum_of_given_numbers : 30 + 80000 + 700 + 60 = 80790 :=
  by
    sorry

end sum_of_given_numbers_l15_15812


namespace cost_per_pound_correct_l15_15638

noncomputable def cost_per_pound_of_coffee (initial_amount spent_amount pounds_of_coffee : ℕ) : ℚ :=
  (initial_amount - spent_amount) / pounds_of_coffee

theorem cost_per_pound_correct :
  let initial_amount := 70
  let amount_left    := 35.68
  let pounds_of_coffee := 4
  (initial_amount - amount_left) / pounds_of_coffee = 8.58 := 
by
  sorry

end cost_per_pound_correct_l15_15638


namespace jesse_gave_pencils_l15_15112

theorem jesse_gave_pencils (initial_pencils : ℕ) (final_pencils : ℕ) (pencils_given : ℕ) :
  initial_pencils = 78 → final_pencils = 34 → pencils_given = initial_pencils - final_pencils → pencils_given = 44 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jesse_gave_pencils_l15_15112


namespace bakery_made_muffins_l15_15804

-- Definitions based on conditions
def muffins_per_box : ℕ := 5
def available_boxes : ℕ := 10
def additional_boxes_needed : ℕ := 9

-- Theorem statement
theorem bakery_made_muffins :
  (available_boxes * muffins_per_box) + (additional_boxes_needed * muffins_per_box) = 95 := 
by
  sorry

end bakery_made_muffins_l15_15804


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l15_15345

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l15_15345


namespace ellipse_standard_equation_l15_15320

theorem ellipse_standard_equation :
  ∀ (a b c : ℝ), a = 9 → c = 6 → b = Real.sqrt (a^2 - c^2) →
  (b ≠ 0 ∧ a ≠ 0 → (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)) :=
by
  sorry

end ellipse_standard_equation_l15_15320


namespace coleFenceCostCorrect_l15_15898

noncomputable def coleFenceCost : ℕ := 455

def woodenFenceCost : ℕ := 15 * 6
def woodenFenceNeighborContribution : ℕ := woodenFenceCost / 3
def coleWoodenFenceCost : ℕ := woodenFenceCost - woodenFenceNeighborContribution

def metalFenceCost : ℕ := 15 * 8
def coleMetalFenceCost : ℕ := metalFenceCost

def hedgeCost : ℕ := 30 * 10
def hedgeNeighborContribution : ℕ := hedgeCost / 2
def coleHedgeCost : ℕ := hedgeCost - hedgeNeighborContribution

def installationFee : ℕ := 75
def soilPreparationFee : ℕ := 50

def totalCost : ℕ := coleWoodenFenceCost + coleMetalFenceCost + coleHedgeCost + installationFee + soilPreparationFee

theorem coleFenceCostCorrect : totalCost = coleFenceCost := by
  -- Skipping the proof steps with sorry
  sorry

end coleFenceCostCorrect_l15_15898


namespace share_of_A_eq_70_l15_15639

theorem share_of_A_eq_70 (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 595) : A = 70 :=
sorry

end share_of_A_eq_70_l15_15639


namespace greatest_integer_gcd_30_is_125_l15_15348

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l15_15348


namespace john_average_speed_l15_15468

variable {minutes_uphill : ℝ} (h1 : minutes_uphill = 45)
variable {distance_uphill : ℝ} (h2 : distance_uphill = 2)
variable {minutes_downhill : ℝ} (h3 : minutes_downhill = 15)
variable {distance_downhill : ℝ} (h4 : distance_downhill = 2)

theorem john_average_speed : 
  let total_distance := distance_uphill + distance_downhill in
  let total_time := minutes_uphill + minutes_downhill in
  total_distance / (total_time / 60) = 4 :=
by
  sorry

end john_average_speed_l15_15468


namespace max_value_of_f_l15_15014

theorem max_value_of_f :
  ∀ (x : ℝ), -5 ≤ x ∧ x ≤ 13 → ∃ (y : ℝ), y = x - 5 ∧ y ≤ 8 ∧ y >= -10 ∧ 
  (∀ (z : ℝ), z = (x - 5) → z ≤ 8) := 
by
  sorry

end max_value_of_f_l15_15014


namespace Alice_min_speed_l15_15312

theorem Alice_min_speed
  (distance : Real := 120)
  (bob_speed : Real := 40)
  (alice_delay : Real := 0.5)
  (alice_min_speed : Real := distance / (distance / bob_speed - alice_delay)) :
  alice_min_speed = 48 := 
by
  sorry

end Alice_min_speed_l15_15312


namespace R_and_D_expense_corresponding_to_productivity_increase_l15_15056

/-- Given values for R&D expenses and increase in average labor productivity -/
def R_and_D_t : ℝ := 2640.92
def Delta_APL_t_plus_2 : ℝ := 0.81

/-- Statement to be proved: the R&D expense in million rubles corresponding 
    to an increase in average labor productivity by 1 million rubles per person -/
theorem R_and_D_expense_corresponding_to_productivity_increase : 
  R_and_D_t / Delta_APL_t_plus_2 = 3260 := 
by
  sorry

end R_and_D_expense_corresponding_to_productivity_increase_l15_15056


namespace average_speed_over_ride_l15_15127

theorem average_speed_over_ride :
  let speed1 := 12 -- speed in km/h
  let time1 := 5 / 60 -- time in hours
  
  let speed2 := 15 -- speed in km/h
  let time2 := 10 / 60 -- time in hours
  
  let speed3 := 18 -- speed in km/h
  let time3 := 15 / 60 -- time in hours
  
  let distance1 := speed1 * time1 -- distance for the first segment
  let distance2 := speed2 * time2 -- distance for the second segment
  let distance3 := speed3 * time3 -- distance for the third segment
  
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1 + time2 + time3
  let avg_speed := total_distance / total_time
  
  avg_speed = 16 :=
by
  sorry

end average_speed_over_ride_l15_15127


namespace EM_parallel_AC_l15_15450

-- Define the points A, B, C, D, E, and M
variables (A B C D E M : Type) 

-- Define the conditions described in the problem
variables {x y : Real}

-- Given that ABCD is an isosceles trapezoid with AB parallel to CD and AB > CD
variable (isosceles_trapezoid : Prop)

-- E is the foot of the perpendicular from D to AB
variable (foot_perpendicular : Prop)

-- M is the midpoint of BD
variable (midpoint : Prop)

-- We need to prove that EM is parallel to AC
theorem EM_parallel_AC (h1 : isosceles_trapezoid) (h2 : foot_perpendicular) (h3 : midpoint) : Prop := sorry

end EM_parallel_AC_l15_15450


namespace arc_lengths_l15_15975

-- Definitions for the given conditions
def circumference : ℝ := 80  -- Circumference of the circle

-- Angles in degrees
def angle_AOM : ℝ := 45
def angle_MOB : ℝ := 90

-- Radius of the circle using the formula C = 2 * π * r
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

-- Calculate the arc lengths using the angles
noncomputable def arc_length_AM : ℝ := (angle_AOM / 360) * circumference
noncomputable def arc_length_MB : ℝ := (angle_MOB / 360) * circumference

-- The theorem stating the required lengths
theorem arc_lengths (h : circumference = 80 ∧ angle_AOM = 45 ∧ angle_MOB = 90) :
  arc_length_AM = 10 ∧ arc_length_MB = 20 :=
by
  sorry

end arc_lengths_l15_15975


namespace group_of_friends_l15_15867

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l15_15867


namespace complex_modulus_square_l15_15298

open Complex

theorem complex_modulus_square (z : ℂ) (h : z^2 + abs z ^ 2 = 7 + 6 * I) : abs z ^ 2 = 85 / 14 :=
sorry

end complex_modulus_square_l15_15298


namespace complex_number_identity_l15_15976

open Complex

theorem complex_number_identity : (2 - I : ℂ) / (1 + 2 * I) = -I := 
by 
  sorry

end complex_number_identity_l15_15976


namespace regular_polygon_perimeter_l15_15540

theorem regular_polygon_perimeter
  (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : side_length = 8)
  (h2 : exterior_angle = 90)
  (h3 : n = 360 / exterior_angle) :
  n * side_length = 32 := by
  sorry

end regular_polygon_perimeter_l15_15540


namespace smallest_three_digit_multiple_of_13_l15_15830

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l15_15830


namespace evaluate_expression_l15_15202

noncomputable def x : ℚ := 4 / 7
noncomputable def y : ℚ := 6 / 8

theorem evaluate_expression : (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end evaluate_expression_l15_15202


namespace cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l15_15556

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

theorem sin_135_eq_sqrt2_div_2 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by sorry

end cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l15_15556


namespace oxygen_atoms_in_compound_l15_15667

-- Define given conditions as parameters in the problem.
def number_of_oxygen_atoms (molecular_weight : ℕ) (weight_Al : ℕ) (weight_H : ℕ) (weight_O : ℕ) (atoms_Al : ℕ) (atoms_H : ℕ) (weight : ℕ) : ℕ := 
  (weight - (atoms_Al * weight_Al + atoms_H * weight_H)) / weight_O

-- Define the actual problem using the defined conditions.
theorem oxygen_atoms_in_compound
  (molecular_weight : ℕ := 78) 
  (weight_Al : ℕ := 27) 
  (weight_H : ℕ := 1) 
  (weight_O : ℕ := 16) 
  (atoms_Al : ℕ := 1) 
  (atoms_H : ℕ := 3) : 
  number_of_oxygen_atoms molecular_weight weight_Al weight_H weight_O atoms_Al atoms_H molecular_weight = 3 := 
sorry

end oxygen_atoms_in_compound_l15_15667


namespace Carol_saves_9_per_week_l15_15897

variable (C : ℤ)

def Carol_savings (weeks : ℤ) : ℤ :=
  60 + weeks * C

def Mike_savings (weeks : ℤ) : ℤ :=
  90 + weeks * 3

theorem Carol_saves_9_per_week (h : Carol_savings C 5 = Mike_savings 5) : C = 9 :=
by
  dsimp [Carol_savings, Mike_savings] at h
  sorry

end Carol_saves_9_per_week_l15_15897


namespace greatest_integer_with_gcd_l15_15351

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l15_15351


namespace geometric_seq_increasing_condition_l15_15771

theorem geometric_seq_increasing_condition (q : ℝ) (a : ℕ → ℝ): 
  (∀ n : ℕ, a (n + 1) = q * a n) → (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m) ∧ ¬ (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m))) :=
sorry

end geometric_seq_increasing_condition_l15_15771


namespace find_distance_to_school_l15_15947

variable (v d : ℝ)
variable (h_rush_hour : d = v * (1 / 2))
variable (h_no_traffic : d = (v + 20) * (1 / 4))

theorem find_distance_to_school (h_rush_hour : d = v * (1 / 2)) (h_no_traffic : d = (v + 20) * (1 / 4)) : d = 10 := by
  sorry

end find_distance_to_school_l15_15947


namespace apex_angle_of_quadrilateral_pyramid_l15_15565

theorem apex_angle_of_quadrilateral_pyramid :
  ∃ (α : ℝ), α = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end apex_angle_of_quadrilateral_pyramid_l15_15565


namespace coeff_x2_in_PQ_is_correct_l15_15566

variable (c : ℝ)

def P (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2 - 3 * x + 1
def Q (x : ℝ) : ℝ := 3 * x^3 + c * x^2 - 8 * x - 5

def coeff_x2 (x : ℝ) : ℝ := -20 - 2 * c

theorem coeff_x2_in_PQ_is_correct :
  (4 : ℝ) * (-5) + (-3) * c + c = -20 - 2 * c := by
  sorry

end coeff_x2_in_PQ_is_correct_l15_15566


namespace increasing_function_condition_l15_15807

variable {x : ℝ} {a : ℝ}

theorem increasing_function_condition (h : 0 < a) :
  (∀ x ≥ 1, deriv (λ x => x^3 - a * x) x ≥ 0) ↔ (0 < a ∧ a ≤ 3) :=
by
  sorry

end increasing_function_condition_l15_15807


namespace problem_1_problem_2_l15_15122

-- Define the sets M and N as conditions and include a > 0 condition.
def M (a : ℝ) : Set ℝ := {x : ℝ | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x : ℝ | 4 * x ^ 2 - 4 * x - 3 < 0}

-- Problem 1: Prove that a = 2 given the set conditions.
theorem problem_1 (a : ℝ) (h_pos : a > 0) :
  M a ∪ N = {x : ℝ | -2 ≤ x ∧ x < 3 / 2} → a = 2 :=
sorry

-- Problem 2: Prove the range of a is 0 < a ≤ 1 / 2 given the set conditions.
theorem problem_2 (a : ℝ) (h_pos : a > 0) :
  N ∪ (compl (M a)) = Set.univ → 0 < a ∧ a ≤ 1 / 2 :=
sorry

end problem_1_problem_2_l15_15122


namespace basic_cable_cost_l15_15290

variable (B M S : ℝ)

def CostOfMovieChannels (B : ℝ) : ℝ := B + 12
def CostOfSportsChannels (M : ℝ) : ℝ := M - 3

theorem basic_cable_cost :
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  B + M + S = 36 → B = 5 :=
by
  intro h
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  sorry

end basic_cable_cost_l15_15290


namespace percentage_defective_meters_l15_15548

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (percentage : ℚ) :
  total_meters = 2500 →
  defective_meters = 2 →
  percentage = (defective_meters / total_meters) * 100 →
  percentage = 0.08 := 
sorry

end percentage_defective_meters_l15_15548


namespace side_length_of_square_l15_15384

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l15_15384


namespace tina_sequence_erasure_l15_15992

open Nat

def initial_sequence : List ℕ :=
  List.repeat [1, 2, 3, 4, 5, 6] 2500 >>= id

def erase_every_nth {α : Type} (lst : List α) (n : ℕ) : List α :=
  lst.enum.filter (λ ⟨idx, _⟩, (idx + 1) % n ≠ 0).map Prod.snd

def final_sequence : List ℕ :=
  erase_every_nth (erase_every_nth (erase_every_nth initial_sequence 4) 5) 6

def positions := [3018, 3019, 3020]  -- zero-indexed

def sum_positions (lst : List ℕ) (pos : List ℕ) : ℕ :=
  pos.map (λ i, lst.nth_le i sorry).sum

theorem tina_sequence_erasure :
  sum_positions final_sequence positions = 5 := 
sorry

end tina_sequence_erasure_l15_15992


namespace younger_brother_height_l15_15645

theorem younger_brother_height
  (O Y : ℕ)
  (h1 : O - Y = 12)
  (h2 : O + Y = 308) :
  Y = 148 :=
by
  sorry

end younger_brother_height_l15_15645


namespace ball_draw_probability_red_is_one_ninth_l15_15005

theorem ball_draw_probability_red_is_one_ninth :
  let A_red := 4
  let A_white := 2
  let B_red := 1
  let B_white := 5
  let P_red_A := A_red / (A_red + A_white)
  let P_red_B := B_red / (B_red + B_white)
  P_red_A * P_red_B = 1 / 9 := by
    -- Proof here
    sorry

end ball_draw_probability_red_is_one_ninth_l15_15005


namespace total_jellybeans_l15_15550

def nephews := 3
def nieces := 2
def jellybeans_per_child := 14
def children := nephews + nieces

theorem total_jellybeans : children * jellybeans_per_child = 70 := by
  sorry

end total_jellybeans_l15_15550


namespace least_three_digit_multiple_of_13_l15_15835

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, n ≥ 100 ∧ n % 13 = 0 ∧ ∀ m : ℕ, m ≥ 100 → m % 13 = 0 → n ≤ m :=
begin
  use 104,
  split,
  { exact nat.le_of_eq rfl },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero rfl) },
  { intros m hm hmod,
    have h8 : 8 * 13 = 104 := rfl,
    rw ←h8,
    exact nat.le_mul_of_pos_left (by norm_num) },
end

end least_three_digit_multiple_of_13_l15_15835


namespace annabelle_savings_l15_15896

noncomputable def weeklyAllowance : ℕ := 30
noncomputable def junkFoodFraction : ℚ := 1 / 3
noncomputable def sweetsCost : ℕ := 8

theorem annabelle_savings :
  let junkFoodCost := weeklyAllowance * junkFoodFraction
  let totalSpent := junkFoodCost + sweetsCost
  let savings := weeklyAllowance - totalSpent
  savings = 12 := 
by
  sorry

end annabelle_savings_l15_15896


namespace probability_two_green_balls_picked_l15_15184

theorem probability_two_green_balls_picked :
  let total_balls := 12 in
  let green_balls := 5 in
  let yellow_balls := 3 in
  let blue_balls := 4 in
  let chosen_balls := 4 in
  let combinations (n k : ℕ) := Nat.choose n k in
  let total_ways := combinations total_balls chosen_balls in
  let ways_to_pick_two_green := combinations green_balls 2 in
  let ways_to_pick_two_remaining := combinations (total_balls - green_balls) (chosen_balls - 2) in
  let successful_outcomes := ways_to_pick_two_green * ways_to_pick_two_remaining in
  let probability := successful_outcomes / total_ways in
  probability = (42 : ℚ) / 99 :=  
by
  sorry

end probability_two_green_balls_picked_l15_15184


namespace part_1_part_2_part_3_l15_15299

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2
noncomputable def g (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := a * Real.log x
noncomputable def F (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x * g x a h
noncomputable def G (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x - g x a h + (a - 1) * x 

theorem part_1 (a : ℝ) (h : 0 < a) :
  ∃(x : ℝ), x = -(a / (4 * Real.exp 1)) :=
sorry

theorem part_2 (a : ℝ) (h1 : 0 < a) : 
  (∃ x1 x2, (1/e) < x1 ∧ x1 < e ∧ (1/e) < x2 ∧ x2 < e ∧ G x1 a h1 = 0 ∧ G x2 a h1 = 0) 
    ↔ (a > (2 * Real.exp 1 - 1) / (2 * (Real.exp 1)^2 + 2 * Real.exp 1) ∧ a < 1/2) :=
sorry

theorem part_3 : 
  ∀ {x : ℝ}, 0 < x → Real.log x + (3 / (4 * x^2)) - (1 / Real.exp x) > 0 :=
sorry

end part_1_part_2_part_3_l15_15299


namespace alcohol_percentage_after_adding_water_l15_15011

variables (initial_volume : ℕ) (initial_percentage : ℕ) (added_volume : ℕ)
def initial_alcohol_volume := initial_volume * initial_percentage / 100
def final_volume := initial_volume + added_volume
def final_percentage := initial_alcohol_volume * 100 / final_volume

theorem alcohol_percentage_after_adding_water :
  initial_volume = 15 →
  initial_percentage = 20 →
  added_volume = 5 →
  final_percentage = 15 := by
sorry

end alcohol_percentage_after_adding_water_l15_15011


namespace matt_peanut_revenue_l15_15778

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l15_15778


namespace binomial_coefficient_10_3_l15_15725

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15725


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15358

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15358


namespace multiplication_result_l15_15657

theorem multiplication_result :
  3^2 * 5^2 * 7 * 11^2 = 190575 :=
by sorry

end multiplication_result_l15_15657


namespace solution_set_eq_l15_15502

theorem solution_set_eq : { x : ℝ | |x| * (x - 2) ≥ 0 } = { x : ℝ | x ≥ 2 ∨ x = 0 } := by
  sorry

end solution_set_eq_l15_15502


namespace distinct_values_l15_15084

-- Define the expressions as terms in Lean
def expr1 : ℕ := 3 ^ (3 ^ 3)
def expr2 : ℕ := (3 ^ 3) ^ 3

-- State the theorem that these terms yield exactly two distinct values
theorem distinct_values : (expr1 ≠ expr2) ∧ ((expr1 = 3^27) ∨ (expr1 = 19683)) ∧ ((expr2 = 3^27) ∨ (expr2 = 19683)) := 
  sorry

end distinct_values_l15_15084


namespace dictionary_prices_and_max_A_l15_15185

-- Definitions for the problem
def price_A := 70
def price_B := 50

-- Conditions from the problem
def condition1 := (price_A + 2 * price_B = 170)
def condition2 := (2 * price_A + 3 * price_B = 290)

-- The proof problem statement
theorem dictionary_prices_and_max_A (h1 : price_A + 2 * price_B = 170) (h2 : 2 * price_A + 3 * price_B = 290) :
  price_A = 70 ∧ price_B = 50 ∧ (∀ (x y : ℕ), x + y = 30 → 70 * x + 50 * y ≤ 1600 → x ≤ 5) :=
by
  sorry

end dictionary_prices_and_max_A_l15_15185


namespace side_length_of_square_l15_15396

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l15_15396


namespace students_enrolled_for_german_l15_15604

theorem students_enrolled_for_german 
  (total_students : ℕ)
  (both_english_german : ℕ)
  (only_english : ℕ)
  (at_least_one_subject : total_students = 32 ∧ both_english_german = 12 ∧ only_english = 10) :
  ∃ G : ℕ, G = 22 :=
by
  -- Lean proof steps will go here.
  sorry

end students_enrolled_for_german_l15_15604


namespace consumer_installment_credit_l15_15520

theorem consumer_installment_credit : 
  ∃ C : ℝ, 
    (0.43 * C = 200) ∧ 
    (C = 465.116) :=
by
  sorry

end consumer_installment_credit_l15_15520


namespace divisibility_criterion_l15_15071

theorem divisibility_criterion (n : ℕ) : 
  (20^n - 13^n - 7^n) % 309 = 0 ↔ 
  ∃ k : ℕ, n = 1 + 6 * k ∨ n = 5 + 6 * k := 
  sorry

end divisibility_criterion_l15_15071


namespace infinite_points_of_one_color_l15_15905

theorem infinite_points_of_one_color (colors : ℤ → Prop) (red blue : ℤ → Prop)
  (h_colors : ∀ n : ℤ, colors n → (red n ∨ blue n))
  (h_red_blue : ∀ n : ℤ, red n → ¬ blue n)
  (h_blue_red : ∀ n : ℤ, blue n → ¬ red n) :
  ∃ c : ℤ → Prop, (∀ k : ℕ, ∃ infinitely_many p : ℤ, c p ∧ p % k = 0) :=
by
  sorry

end infinite_points_of_one_color_l15_15905


namespace pizza_slices_with_both_toppings_l15_15370

theorem pizza_slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices n : ℕ) 
    (h1 : total_slices = 14) 
    (h2 : pepperoni_slices = 8) 
    (h3 : mushroom_slices = 12) 
    (h4 : ∀ s, s = pepperoni_slices + mushroom_slices - n ∧ s = total_slices := by sorry) :
    n = 6 :=
sorry

end pizza_slices_with_both_toppings_l15_15370


namespace largest_multiple_of_9_less_than_100_l15_15167

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n < 100 ∧ n % 9 = 0 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 0 → m ≤ n :=
by
  use 99
  split
  · exact dec_trivial
  split 
  · exact dec_trivial
  intro m hm
  cases hm 
  cases hm_right 
  have h : m ≤ 99 / 1 := by norm_cast; simp only [Nat.le_div_iff_mul_le dec_trivial, mul_one, div_one]
  exact h
  sorry -- Complete the proof

end largest_multiple_of_9_less_than_100_l15_15167


namespace D_is_painting_l15_15102

def A_activity (act : String) : Prop := 
  act ≠ "walking" ∧ act ≠ "playing basketball"

def B_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "running"

def C_activity_implies_A_activity (C_act A_act : String) : Prop :=
  C_act = "walking" → A_act = "dancing"

def D_activity (act : String) : Prop :=
  act ≠ "playing basketball" ∧ act ≠ "running"

def C_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "playing basketball"

theorem D_is_painting :
  (∃ a b c d : String,
    A_activity a ∧
    B_activity b ∧
    C_activity_implies_A_activity c a ∧
    D_activity d ∧
    C_activity c) →
  ∃ d : String, d = "painting" :=
by
  intros h
  sorry

end D_is_painting_l15_15102


namespace trig_expression_value_l15_15507

open Real

theorem trig_expression_value : 
  (2 * cos (10 * (π / 180)) - sin (20 * (π / 180))) / cos (20 * (π / 180)) = sqrt 3 :=
by
  -- Proof should go here
  sorry

end trig_expression_value_l15_15507


namespace Q_eq_sum_of_binom_l15_15572

open Nat

def Q (n k : ℕ) : ℕ :=
(coef k (expand (x + x^2 + x^3 + 1) ^ n))

theorem Q_eq_sum_of_binom 
  (n k : ℕ) :
  Q n k = ∑ j in range (n + 1), binom n j * binom n (k - 2 * j) :=
by
  sorry

end Q_eq_sum_of_binom_l15_15572


namespace find_largest_integer_l15_15213

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l15_15213


namespace varphi_le_one_varphi_l15_15454

noncomputable def f (a x : ℝ) := -a * Real.log x

-- Definition of the minimum value function φ for a > 0
noncomputable def varphi (a : ℝ) := -a * Real.log a

theorem varphi_le_one (a : ℝ) (h : 0 < a) : varphi a ≤ 1 := 
by sorry

theorem varphi'_le (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
    (1 - Real.log a) ≤ (1 - Real.log b) := 
by sorry

end varphi_le_one_varphi_l15_15454


namespace solution1_solution2_l15_15305

section
variable {α : Type} [Fintype α] (s : Finset α)

-- Problem 1: The number of ways to select 2 males and 2 females from 4 males and 5 females.
def problem1 : Prop :=
  (s.filter (λ x, x ∈ finset.range 4)).card = 2 ∧
  (s.filter (λ x, x ∈ finset.range 5)).card = 2 →
  s.card = 60

-- Problem 2: The number of ways to select at least 1 male and 1 female,
-- and male student A and female student B cannot be selected together, is 99.
def problem2 (A B : α) : Prop :=
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 3 ∧ (s.filter (λ x, x ∈ finset.range 4)).card = k ∧ (s.filter (λ x, x ∈ finset.range 5)).card = (4 - k)) ∧
  ¬(A ∈ s ∧ B ∈ s) →
  s.card = 99

end

-- Assertions to the assumptions and results.
theorem solution1 : problem1 := sorry
theorem solution2 {α : Type} [Fintype α] (A B : α) : problem2 A B := sorry

end solution1_solution2_l15_15305


namespace age_of_student_who_left_l15_15493

/-- 
The average student age of a class with 30 students is 10 years.
After one student leaves and the teacher (who is 41 years old) is included,
the new average age is 11 years. Prove that the student who left is 11 years old.
-/
theorem age_of_student_who_left (x : ℕ) (h1 : (30 * 10) = 300)
    (h2 : (300 - x + 41) / 30 = 11) : x = 11 :=
by 
  -- This is where the proof would go
  sorry

end age_of_student_who_left_l15_15493


namespace base_of_power_expr_l15_15944

-- Defining the power expression as a condition
def power_expr : ℤ := (-4 : ℤ) ^ 3

-- The Lean statement for the proof problem
theorem base_of_power_expr : ∃ b : ℤ, (power_expr = b ^ 3) ∧ (b = -4) := 
sorry

end base_of_power_expr_l15_15944


namespace transformed_roots_equation_l15_15088

theorem transformed_roots_equation (α β : ℂ) (h1 : 3 * α^2 + 2 * α + 1 = 0) (h2 : 3 * β^2 + 2 * β + 1 = 0) :
  ∃ (y : ℂ), (y - (3 * α + 2)) * (y - (3 * β + 2)) = y^2 + 4 := 
sorry

end transformed_roots_equation_l15_15088


namespace product_of_five_consecutive_integers_not_perfect_square_l15_15632

theorem product_of_five_consecutive_integers_not_perfect_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = k * k :=
by {
  sorry
}

end product_of_five_consecutive_integers_not_perfect_square_l15_15632


namespace jelly_beans_correct_l15_15132

-- Define the constants and conditions
def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def total_amount : ℕ := sandra_savings + mother_gift + father_gift

def candy_cost : ℕ := 5 / 10 -- == 0.5
def jelly_bean_cost : ℕ := 2 / 10 -- == 0.2

def candies_bought : ℕ := 14
def money_spent_on_candies : ℕ := candies_bought * candy_cost

def remaining_money : ℕ := total_amount - money_spent_on_candies
def money_left : ℕ := 11

-- Prove the number of jelly beans bought is 20
def number_of_jelly_beans : ℕ :=
  (remaining_money - money_left) / jelly_bean_cost

theorem jelly_beans_correct : number_of_jelly_beans = 20 :=
sorry

end jelly_beans_correct_l15_15132


namespace complex_multiplication_l15_15192

theorem complex_multiplication : ∀ (i : ℂ), i^2 = -1 → i * (2 + 3 * i) = (-3 : ℂ) + 2 * i :=
by
  intros i hi
  sorry

end complex_multiplication_l15_15192


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15241

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15241


namespace cars_in_section_H_l15_15785

theorem cars_in_section_H
  (rows_G : ℕ) (cars_per_row_G : ℕ) (rows_H : ℕ)
  (cars_per_minute : ℕ) (minutes_spent : ℕ)  
  (total_cars_walked_past : ℕ) :
  rows_G = 15 →
  cars_per_row_G = 10 →
  rows_H = 20 →
  cars_per_minute = 11 →
  minutes_spent = 30 →
  total_cars_walked_past = (rows_G * cars_per_row_G) + ((cars_per_minute * minutes_spent) - (rows_G * cars_per_row_G)) →
  (total_cars_walked_past - (rows_G * cars_per_row_G)) / rows_H = 9 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end cars_in_section_H_l15_15785


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15357

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15357


namespace probability_between_lines_l15_15432

def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

theorem probability_between_lines 
  (h1 : ∀ x > 0, line_l x ≥ 0) 
  (h2 : ∀ x > 0, line_m x ≥ 0) 
  (h3 : ∀ x > 0, line_l x < line_m x ∨ line_m x ≤ 0) : 
  (1 / 16 : ℝ) * 100 = 0.16 :=
by
  sorry

end probability_between_lines_l15_15432


namespace positive_difference_solutions_abs_eq_30_l15_15997

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l15_15997


namespace acute_angle_coincidence_l15_15817

theorem acute_angle_coincidence (α : ℝ) (k : ℤ) :
  0 < α ∧ α < 180 ∧ 9 * α = k * 360 + α → α = 45 ∨ α = 90 ∨ α = 135 :=
by
  sorry

end acute_angle_coincidence_l15_15817


namespace product_of_two_numbers_l15_15511

-- Define HCF (Highest Common Factor) and LCM (Least Common Multiple) conditions
def hcf_of_two_numbers (a b : ℕ) : ℕ := 11
def lcm_of_two_numbers (a b : ℕ) : ℕ := 181

-- The theorem to prove
theorem product_of_two_numbers (a b : ℕ) 
  (h1 : hcf_of_two_numbers a b = 11)
  (h2 : lcm_of_two_numbers a b = 181) : 
  a * b = 1991 :=
by 
  -- This is where we would put the proof, but we can use sorry for now
  sorry

end product_of_two_numbers_l15_15511


namespace binomial_coefficient_10_3_l15_15713

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15713


namespace find_N_l15_15173

theorem find_N (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 :=
by
  intros h
  -- Sorry to skip the proof.
  sorry

end find_N_l15_15173


namespace central_angle_proof_l15_15964

noncomputable def central_angle (l r : ℝ) : ℝ :=
  l / r

theorem central_angle_proof :
  central_angle 300 100 = 3 :=
by
  -- The statement of the theorem aligns with the given problem conditions and the expected answer.
  sorry

end central_angle_proof_l15_15964


namespace soccer_game_goals_l15_15201

theorem soccer_game_goals (A1_first_half A2_first_half B1_first_half B2_first_half : ℕ) 
  (h1 : A1_first_half = 8)
  (h2 : B1_first_half = A1_first_half / 2)
  (h3 : B2_first_half = A1_first_half)
  (h4 : A2_first_half = B2_first_half - 2) : 
  A1_first_half + A2_first_half + B1_first_half + B2_first_half = 26 :=
by
  -- The proof is not needed, so we use sorry to skip it.
  sorry

end soccer_game_goals_l15_15201


namespace infinite_area_sum_ratio_l15_15501

theorem infinite_area_sum_ratio (T t : ℝ) (p q : ℝ) (h_ratio : T / t = 3 / 2) :
    let series_ratio_triangles := (p + q)^2 / (3 * p * q)
    let series_ratio_quadrilaterals := (p + q)^2 / (2 * p * q)
    (T * series_ratio_triangles) / (t * series_ratio_quadrilaterals) = 1 :=
by
  -- Proof steps go here
  sorry

end infinite_area_sum_ratio_l15_15501


namespace product_consecutive_two_digits_l15_15959

theorem product_consecutive_two_digits (a b c : ℕ) : 
  ¬(∃ n : ℕ, (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2)) :=
by
  sorry

end product_consecutive_two_digits_l15_15959


namespace initial_books_correct_l15_15966

def sold_books : ℕ := 78
def left_books : ℕ := 37
def initial_books : ℕ := sold_books + left_books

theorem initial_books_correct : initial_books = 115 := by
  sorry

end initial_books_correct_l15_15966


namespace zero_if_sum_of_squares_eq_zero_l15_15970

theorem zero_if_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_if_sum_of_squares_eq_zero_l15_15970


namespace xyz_value_l15_15079

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
                x * y * z = 6 := by
  sorry

end xyz_value_l15_15079


namespace largest_multiple_of_9_less_than_100_l15_15170

theorem largest_multiple_of_9_less_than_100 : ∃ x : ℕ, 9 * x < 100 ∧ ∀ y : ℕ, 9 * y < 100 → y ≤ x :=
by
  exists 11
  split
  · linarith
  · intro y h
    have : 9 * y < 100 := h
    calc
      y ≤ floor (11.11) : by linarith

end largest_multiple_of_9_less_than_100_l15_15170


namespace highest_number_on_dice_l15_15372

theorem highest_number_on_dice (n : ℕ) (h1 : 0 < n)
  (h2 : ∃ p : ℝ, p = 0.1111111111111111) 
  (h3 : 1 / 9 = 4 / (n * n)) 
  : n = 6 :=
sorry

end highest_number_on_dice_l15_15372


namespace square_side_length_l15_15409

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l15_15409


namespace percentage_profits_to_revenues_l15_15603

theorem percentage_profits_to_revenues (R P : ℝ) 
  (h1 : R > 0) 
  (h2 : P > 0)
  (h3 : 0.12 * R = 1.2 * P) 
  : P / R = 0.1 :=
by
  sorry

end percentage_profits_to_revenues_l15_15603


namespace train_speed_equivalent_l15_15328

def length_train1 : ℝ := 180
def length_train2 : ℝ := 160
def speed_train1 : ℝ := 60 
def crossing_time_sec : ℝ := 12.239020878329734

noncomputable def speed_train2 (length1 length2 speed1 time : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hr := time / 3600
  let relative_speed := total_length_km / time_hr
  relative_speed - speed1

theorem train_speed_equivalent :
  speed_train2 length_train1 length_train2 speed_train1 crossing_time_sec = 40 :=
by
  simp [length_train1, length_train2, speed_train1, crossing_time_sec, speed_train2]
  sorry

end train_speed_equivalent_l15_15328


namespace geometric_series_sum_150_terms_l15_15981

theorem geometric_series_sum_150_terms (a : ℕ) (r : ℝ)
  (h₁ : a = 250)
  (h₂ : (a - a * r ^ 50) / (1 - r) = 625)
  (h₃ : (a - a * r ^ 100) / (1 - r) = 1225) :
  (a - a * r ^ 150) / (1 - r) = 1801 := by
  sorry

end geometric_series_sum_150_terms_l15_15981


namespace largest_integer_less_than_100_with_remainder_4_l15_15228

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l15_15228


namespace total_amount_received_l15_15371

theorem total_amount_received
  (total_books : ℕ := 500)
  (novels_price : ℕ := 8)
  (biographies_price : ℕ := 12)
  (science_books_price : ℕ := 10)
  (novels_discount : ℚ := 0.25)
  (biographies_discount : ℚ := 0.30)
  (science_books_discount : ℚ := 0.20)
  (sales_tax : ℚ := 0.05)
  (remaining_novels : ℕ := 60)
  (remaining_biographies : ℕ := 65)
  (remaining_science_books : ℕ := 50)
  (novel_ratio_sold : ℚ := 3/5)
  (biography_ratio_sold : ℚ := 2/3)
  (science_book_ratio_sold : ℚ := 7/10)
  (original_novels : ℕ := 150)
  (original_biographies : ℕ := 195)
  (original_science_books : ℕ := 167) -- Rounded from 166.67
  (sold_novels : ℕ := 90)
  (sold_biographies : ℕ := 130)
  (sold_science_books : ℕ := 117)
  (total_revenue_before_discount : ℚ := (90 * 8 + 130 * 12 + 117 * 10))
  (total_revenue_after_discount : ℚ := (720 * (1 - 0.25) + 1560 * (1 - 0.30) + 1170 * (1 - 0.20)))
  (total_revenue_after_tax : ℚ := (2568 * 1.05)) :
  total_revenue_after_tax = 2696.4 :=
by
  sorry

end total_amount_received_l15_15371


namespace largest_integer_less_than_100_leaving_remainder_4_l15_15252

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l15_15252


namespace LCM_GCD_even_nonnegative_l15_15307

theorem LCM_GCD_even_nonnegative (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  : ∃ (n : ℕ), (n = Nat.lcm a b + Nat.gcd a b - a - b) ∧ (n % 2 = 0) ∧ (0 ≤ n) := 
sorry

end LCM_GCD_even_nonnegative_l15_15307


namespace diameter_of_circle_l15_15822

theorem diameter_of_circle {a b c d e f D : ℕ} 
  (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) (h4 : d = 33) (h5 : e = 56) (h6 : f = 65)
  (h_right_triangle1 : a^2 + b^2 = c^2)
  (h_right_triangle2 : d^2 + e^2 = f^2)
  (h_inscribed_triangles : true) -- This represents that both triangles are inscribed in the circle.
: D = 65 :=
sorry

end diameter_of_circle_l15_15822


namespace binomial_coefficient_10_3_l15_15715

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15715


namespace find_x_for_condition_l15_15521

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x_for_condition :
  (2 * f 1 - 16 = f (1 - 6)) :=
by
  sorry

end find_x_for_condition_l15_15521


namespace books_not_sold_l15_15024

theorem books_not_sold (X : ℕ) (H1 : (2/3 : ℝ) * X * 4 = 288) : (1 / 3 : ℝ) * X = 36 :=
by
  -- Proof goes here
  sorry

end books_not_sold_l15_15024


namespace phase_shift_of_cosine_transformation_l15_15257

theorem phase_shift_of_cosine_transformation :
  ∀ (A B C : ℝ), 
  (∀ x : ℝ, y = A * cos (B * x + C)) →
  A = 3 → B = 3 → C = -π / 4 →
  (∃ φ : ℝ, φ = -C / B ∧ φ = π / 12) :=
by
  intros A B C h y_eq_cos A_eq B_eq C_eq
  sorry

end phase_shift_of_cosine_transformation_l15_15257


namespace Matt_income_from_plantation_l15_15775

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l15_15775


namespace Robert_photo_count_l15_15960

theorem Robert_photo_count (k : ℕ) (hLisa : ∃ n : ℕ, k = 8 * n) : k = 24 - 16 → k = 24 :=
by
  intro h
  sorry

end Robert_photo_count_l15_15960


namespace side_length_of_square_l15_15402

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l15_15402


namespace functional_expression_y_l15_15080

theorem functional_expression_y (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x, y + 2 = k * x) 
  (h2 : y = 7) 
  (h3 : x = 3) : 
  y = 3 * x - 2 := 
by 
  sorry

end functional_expression_y_l15_15080


namespace series_sum_l15_15685

theorem series_sum :
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  -- We define the sequence in list form for clarity
  (s.sum = 29) :=
by
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  sorry

end series_sum_l15_15685


namespace g_value_at_2_l15_15313

theorem g_value_at_2 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2 - 2) : g 2 = 11 / 28 :=
sorry

end g_value_at_2_l15_15313


namespace distance_from_P_to_origin_l15_15610

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_P_to_origin :
  distance (-1) 2 0 0 = Real.sqrt 5 :=
by
  sorry

end distance_from_P_to_origin_l15_15610


namespace least_positive_three_digit_multiple_of_13_is_104_l15_15827

theorem least_positive_three_digit_multiple_of_13_is_104 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n = 104 :=
by
  existsi 104
  split
  · show 100 ≤ 104
    exact le_refl 104
  split
  · show 104 < 1000
    exact dec_trivial
  split
  · show 104 % 13 = 0
    exact dec_trivial
  · show 104 = 104
    exact rfl

end least_positive_three_digit_multiple_of_13_is_104_l15_15827


namespace binom_10_3_l15_15691

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15691


namespace prove_composite_k_l15_15077

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := ∃ p q, p > 1 ∧ q > 1 ∧ n = p * q

def problem_statement (a b c d : ℕ) (h : a * b = c * d) : Prop :=
  is_composite (a^1984 + b^1984 + c^1984 + d^1984)

-- The theorem to prove
theorem prove_composite_k (a b c d : ℕ) (h : a * b = c * d) : 
  problem_statement a b c d h := sorry

end prove_composite_k_l15_15077


namespace fraction_sum_is_five_l15_15901

noncomputable def solve_fraction_sum (x y z : ℝ) : Prop :=
  (x + 1/y = 5) ∧ (y + 1/z = 2) ∧ (z + 1/x = 3) ∧ 0 < x ∧ 0 < y ∧ 0 < z → 
  (x / y + y / z + z / x = 5)
    
theorem fraction_sum_is_five (x y z : ℝ) : solve_fraction_sum x y z :=
  sorry

end fraction_sum_is_five_l15_15901


namespace general_term_formula_l15_15647

-- Define the given sequence as a function
def seq (n : ℕ) : ℤ :=
  match n with
  | 0 => 3
  | n + 1 => if (n % 2 = 0) then 4 * (n + 1) - 1 else -(4 * (n + 1) - 1)

-- Define the proposed general term formula
def a_n (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4 * n - 1)

-- State the theorem that general term of the sequence equals the proposed formula
theorem general_term_formula : ∀ n : ℕ, seq n = a_n n := 
by
  sorry

end general_term_formula_l15_15647


namespace volume_of_max_area_rect_prism_l15_15174

noncomputable def side_length_of_square_base (P: ℕ) : ℕ := P / 4

noncomputable def area_of_square_base (side: ℕ) : ℕ := side * side

noncomputable def volume_of_rectangular_prism (base_area: ℕ) (height: ℕ) : ℕ := base_area * height

theorem volume_of_max_area_rect_prism
  (P : ℕ) (hP : P = 32) 
  (H : ℕ) (hH : H = 9) 
  : volume_of_rectangular_prism (area_of_square_base (side_length_of_square_base P)) H = 576 := 
by
  sorry

end volume_of_max_area_rect_prism_l15_15174


namespace sqrt_mixed_number_l15_15737

theorem sqrt_mixed_number :
  (Real.sqrt (8 + 9/16)) = (Real.sqrt 137) / 4 :=
by
  sorry

end sqrt_mixed_number_l15_15737


namespace number_of_cases_for_Ds_hearts_l15_15561

theorem number_of_cases_for_Ds_hearts (hA : 5 ≤ 13) (hB : 4 ≤ 13) (dist : 52 % 4 = 0) : 
  ∃ n, n = 5 ∧ 0 ≤ n ∧ n ≤ 13 := sorry

end number_of_cases_for_Ds_hearts_l15_15561


namespace relationship_between_k_and_a_l15_15147

theorem relationship_between_k_and_a (a k : ℝ) (h_a : 0 < a ∧ a < 1) :
  (k^2 + 1) * a^2 ≥ 1 :=
sorry

end relationship_between_k_and_a_l15_15147


namespace divisibility_of_product_l15_15129

theorem divisibility_of_product (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a * b) % 5 = 0) :
  a % 5 = 0 ∨ b % 5 = 0 :=
sorry

end divisibility_of_product_l15_15129


namespace side_length_of_square_l15_15399

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l15_15399


namespace square_side_length_l15_15411

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l15_15411


namespace first_nonzero_digit_fraction_one_over_197_l15_15330

theorem first_nonzero_digit_fraction_one_over_197 : 
  ∃ d : ℤ, d ≠ 0 ∧ (1 / 197 : ℚ) * 10 ^ (find (λ n, exists_digit (10^n * (1 / 197) % 10)) = d) = 5 :=
sorry

end first_nonzero_digit_fraction_one_over_197_l15_15330


namespace range_of_x_l15_15442

theorem range_of_x (m : ℝ) (x : ℝ) (h : 0 < m ∧ m ≤ 5) : 
  (x^2 + (2 * m - 1) * x > 4 * x + 2 * m - 4) ↔ (x < -6 ∨ x > 4) := 
sorry

end range_of_x_l15_15442


namespace percentage_decrease_is_20_l15_15318

-- Define the original and new prices in Rs.
def original_price : ℕ := 775
def new_price : ℕ := 620

-- Define the decrease in price
def decrease_in_price : ℕ := original_price - new_price

-- Define the formula to calculate the percentage decrease
def percentage_decrease (orig_price new_price : ℕ) : ℕ :=
  (decrease_in_price * 100) / orig_price

-- Prove that the percentage decrease is 20%
theorem percentage_decrease_is_20 :
  percentage_decrease original_price new_price = 20 :=
by
  sorry

end percentage_decrease_is_20_l15_15318


namespace geom_sequence_arith_ratio_l15_15939

variable (a : ℕ → ℝ) (q : ℝ)
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_arith : 3 * a 0 + 2 * a 1 = 2 * (1/2) * a 2)

theorem geom_sequence_arith_ratio (ha : 3 * a 0 + 2 * a 1 = a 2) :
    (a 8 + a 9) / (a 6 + a 7) = 9 := sorry

end geom_sequence_arith_ratio_l15_15939


namespace sum_E_eq_1000_2n_minus_1_2_999n_l15_15295

open Finset

noncomputable def S (n : ℕ) : Finset (Fin (n+1) → Finset (Fin 1000)) := univ

def E {n : ℕ} (a : Fin (n+1) → Finset (Fin 1000)) : ℕ := (Finset.univ.bUnion a).card

theorem sum_E_eq_1000_2n_minus_1_2_999n (n : ℕ) :
  ∑ a in S n, E a = 1000 * (2^n - 1) * 2^(999 * n) := sorry

end sum_E_eq_1000_2n_minus_1_2_999n_l15_15295


namespace square_side_length_l15_15412

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l15_15412


namespace stack_logs_total_l15_15542

   theorem stack_logs_total (a l d : ℤ) (n : ℕ) (top_logs : ℕ) (h1 : a = 15) (h2 : l = 5) (h3 : d = -2) (h4 : n = ((l - a) / d).natAbs + 1) (h5 : top_logs = 5) : (n / 2 : ℤ) * (a + l) = 60 :=
   by
   sorry
   
end stack_logs_total_l15_15542


namespace football_game_cost_l15_15650

theorem football_game_cost :
  ∀ (total_spent strategy_game_cost batman_game_cost football_game_cost : ℝ),
  total_spent = 35.52 →
  strategy_game_cost = 9.46 →
  batman_game_cost = 12.04 →
  total_spent - strategy_game_cost - batman_game_cost = football_game_cost →
  football_game_cost = 13.02 :=
by
  intros total_spent strategy_game_cost batman_game_cost football_game_cost h1 h2 h3 h4
  have : football_game_cost = 13.02 := sorry
  exact this

end football_game_cost_l15_15650


namespace fraction_irreducible_l15_15135

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducible_l15_15135


namespace find_matrix_N_l15_15910

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![47/9, -16/9], 
    ![-10/3, 14/3]]

def v₁ : Fin 2 → ℚ := ![4, 1]
def v₂ : Fin 2 → ℚ := ![1, -2]

def w₁ : Fin 2 → ℚ := ![12, 10]
def w₂ : Fin 2 → ℚ := ![7, -8]

theorem find_matrix_N :
  (N ⬝ v₁ = w₁) ∧ (N ⬝ v₂ = w₂) :=
by
  sorry

end find_matrix_N_l15_15910


namespace power_equality_l15_15932

theorem power_equality (p : ℕ) : 16^10 = 4^p → p = 20 :=
by
  intro h
  -- proof goes here
  sorry

end power_equality_l15_15932


namespace sum_of_rel_prime_greater_than_one_l15_15629

theorem sum_of_rel_prime_greater_than_one (a : ℕ) (h : a > 6) : 
  ∃ b c : ℕ, a = b + c ∧ b > 1 ∧ c > 1 ∧ Nat.gcd b c = 1 :=
sorry

end sum_of_rel_prime_greater_than_one_l15_15629


namespace OHaraTriple_example_l15_15325

def OHaraTriple (a b x : ℕ) : Prop :=
  (Nat.sqrt a + Nat.sqrt b = x)

theorem OHaraTriple_example : OHaraTriple 49 64 15 :=
by
  sorry

end OHaraTriple_example_l15_15325


namespace girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l15_15508

namespace PhotoArrangement

/-- There are 4 boys and 3 girls. -/
def boys : ℕ := 4
def girls : ℕ := 3

/-- Number of ways to arrange given conditions -/
def arrangementsWithGirlsAtEnds : ℕ := 720
def arrangementsWithNoGirlsNextToEachOther : ℕ := 1440
def arrangementsWithGirlAtoRightOfGirlB : ℕ := 2520

-- Problem 1: If there are girls at both ends
theorem girls_at_ends (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlsAtEnds := by
  sorry

-- Problem 2: If no two girls are standing next to each other
theorem no_girls_next_to_each_other (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithNoGirlsNextToEachOther := by
  sorry

-- Problem 3: If girl A must be to the right of girl B
theorem girl_A_right_of_girl_B (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlAtoRightOfGirlB := by
  sorry

end PhotoArrangement

end girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l15_15508


namespace coins_remainder_l15_15851

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l15_15851


namespace bankers_gain_correct_l15_15310

def PW : ℝ := 600
def R : ℝ := 0.10
def n : ℕ := 2

def A : ℝ := PW * (1 + R)^n
def BG : ℝ := A - PW

theorem bankers_gain_correct : BG = 126 :=
by
  sorry

end bankers_gain_correct_l15_15310


namespace integral_identity_proof_l15_15683

noncomputable def integral_identity : Prop :=
  ∫ x in (0 : Real)..(Real.pi / 2), (Real.cos (Real.cos x))^2 + (Real.sin (Real.sin x))^2 = Real.pi / 2

theorem integral_identity_proof : integral_identity :=
sorry

end integral_identity_proof_l15_15683


namespace tax_percentage_excess_l15_15938

/--
In Country X, each citizen is taxed an amount equal to 15 percent of the first $40,000 of income,
plus a certain percentage of all income in excess of $40,000. A citizen of Country X is taxed a total of $8,000
and her income is $50,000.

Prove that the percentage of the tax on the income in excess of $40,000 is 20%.
-/
theorem tax_percentage_excess (total_tax : ℝ) (first_income : ℝ) (additional_income : ℝ) (income : ℝ) (tax_first_part : ℝ) (tax_rate_first_part : ℝ) (tax_rate_excess : ℝ) (tax_excess : ℝ) :
  total_tax = 8000 →
  first_income = 40000 →
  additional_income = 10000 →
  income = first_income + additional_income →
  tax_rate_first_part = 0.15 →
  tax_first_part = tax_rate_first_part * first_income →
  tax_excess = total_tax - tax_first_part →
  tax_rate_excess * additional_income = tax_excess →
  tax_rate_excess = 0.20 :=
by
  intro h_total_tax h_first_income h_additional_income h_income h_tax_rate_first_part h_tax_first_part h_tax_excess h_tax_equation
  sorry

end tax_percentage_excess_l15_15938


namespace sandy_net_amount_spent_l15_15641

def amount_spent_shorts : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def amount_received_return : ℝ := 7.43

theorem sandy_net_amount_spent :
  amount_spent_shorts + amount_spent_shirt - amount_received_return = 18.70 :=
by
  sorry

end sandy_net_amount_spent_l15_15641


namespace Tanya_bought_9_apples_l15_15643

def original_fruit_count : ℕ := 18
def remaining_fruit_count : ℕ := 9
def pears_count : ℕ := 6
def pineapples_count : ℕ := 2
def plums_basket_count : ℕ := 1

theorem Tanya_bought_9_apples : 
  remaining_fruit_count * 2 = original_fruit_count →
  original_fruit_count - (pears_count + pineapples_count + plums_basket_count) = 9 :=
by
  intros h1
  sorry

end Tanya_bought_9_apples_l15_15643


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15221

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15221


namespace polynomial_product_l15_15366

theorem polynomial_product (a b c : ℝ) :
  a * (b - c) ^ 3 + b * (c - a) ^ 3 + c * (a - b) ^ 3 = (a - b) * (b - c) * (c - a) * (a + b + c) :=
by sorry

end polynomial_product_l15_15366


namespace total_cost_correct_l15_15863

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end total_cost_correct_l15_15863


namespace parabola_intersection_l15_15811

theorem parabola_intersection :
  let y := fun x : ℝ => x^2 - 2*x - 3 in
  (∀ x : ℝ, y x = 0 ↔ x = 3 ∨ x = -1) ∧ y (-1) = 0 :=
by
  sorry

end parabola_intersection_l15_15811


namespace find_largest_integer_l15_15214

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l15_15214


namespace necessary_but_not_sufficient_l15_15063

-- Define the geometric mean condition between 2 and 8
def is_geometric_mean (m : ℝ) := m = 4 ∨ m = -4

-- Prove that m = 4 is a necessary but not sufficient condition for is_geometric_mean
theorem necessary_but_not_sufficient (m : ℝ) :
  (is_geometric_mean m) ↔ (m = 4) :=
sorry

end necessary_but_not_sufficient_l15_15063


namespace complement_of_angle_l15_15073

variable (α : ℝ)

axiom given_angle : α = 63 + 21 / 60

theorem complement_of_angle :
  90 - α = 26 + 39 / 60 :=
by
  sorry

end complement_of_angle_l15_15073


namespace square_side_length_l15_15387

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l15_15387


namespace fraction_equiv_l15_15596

theorem fraction_equiv (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 :=
by
  sorry

end fraction_equiv_l15_15596


namespace damage_in_usd_correct_l15_15029

def exchange_rate := (125 : ℚ) / 100
def damage_CAD := 45000000
def damage_USD := damage_CAD / exchange_rate

theorem damage_in_usd_correct (CAD_to_USD : exchange_rate = (125 : ℚ) / 100) (damage_in_cad : damage_CAD = 45000000) : 
  damage_USD = 36000000 :=
by
  sorry

end damage_in_usd_correct_l15_15029


namespace curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l15_15753

-- Definitions for the conditions
def curve_C_polar (ρ θ : ℝ) := ρ = 4 * Real.sin θ
def line_l_parametric (x y t : ℝ) := 
  x = (Real.sqrt 3 / 2) * t ∧ 
  y = 1 + (1 / 2) * t

-- Theorem statements
theorem curve_C_cartesian_eq : ∀ x y : ℝ,
  (∃ (ρ θ : ℝ), curve_C_polar ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

theorem line_l_general_eq : ∀ x y t : ℝ,
  line_l_parametric x y t →
  x - (Real.sqrt 3) * y + Real.sqrt 3 = 0 :=
by sorry

theorem max_area_triangle_PAB : ∀ (P A B : ℝ × ℝ),
  (∃ (θ : ℝ), P = ⟨2 * Real.cos θ, 2 + 2 * Real.sin θ⟩ ∧
   (∃ t : ℝ, line_l_parametric A.1 A.2 t) ∧
   (∃ t' : ℝ, line_l_parametric B.1 B.2 t') ∧
   A ≠ B) →
  (1/2) * Real.sqrt 13 * (2 + Real.sqrt 3 / 2) = (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
by sorry

end curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l15_15753


namespace largest_integer_remainder_condition_l15_15244

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l15_15244


namespace base8_perfect_square_b_zero_l15_15099

-- Define the base 8 representation and the perfect square condition
def base8_to_decimal (a b : ℕ) : ℕ := 512 * a + 64 + 8 * b + 4

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating that if the number in base 8 is a perfect square, then b = 0
theorem base8_perfect_square_b_zero (a b : ℕ) (h₀ : a ≠ 0) 
  (h₁ : is_perfect_square (base8_to_decimal a b)) : b = 0 :=
sorry

end base8_perfect_square_b_zero_l15_15099


namespace count_valid_n_le_30_l15_15571

theorem count_valid_n_le_30 :
  ∀ n : ℕ, (0 < n ∧ n ≤ 30) → (n! * 2) % (n * (n + 1)) = 0 := by
  sorry

end count_valid_n_le_30_l15_15571


namespace length_PR_in_triangle_l15_15945

/-- In any triangle PQR, given:
  PQ = 7, QR = 10, median PS = 5,
  the length of PR must be sqrt(149). -/
theorem length_PR_in_triangle (PQ QR PS : ℝ) (PQ_eq : PQ = 7) (QR_eq : QR = 10) (PS_eq : PS = 5) : 
  ∃ (PR : ℝ), PR = Real.sqrt 149 := 
sorry

end length_PR_in_triangle_l15_15945


namespace rachel_makes_money_l15_15017

theorem rachel_makes_money (cost_per_bar total_bars remaining_bars : ℕ) (h_cost : cost_per_bar = 2) (h_total : total_bars = 13) (h_remaining : remaining_bars = 4) :
  cost_per_bar * (total_bars - remaining_bars) = 18 :=
by 
  sorry

end rachel_makes_money_l15_15017


namespace area_on_larger_sphere_l15_15672

-- Define the radii of the spheres
def r_small : ℝ := 1
def r_in : ℝ := 4
def r_out : ℝ := 6

-- Given the area on the smaller sphere
def A_small_sphere_area : ℝ := 37

-- Statement: Find the area on the larger sphere
theorem area_on_larger_sphere :
  (A_small_sphere_area * (r_out / r_in) ^ 2 = 83.25) := by
  sorry

end area_on_larger_sphere_l15_15672


namespace friends_game_l15_15877

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l15_15877


namespace votes_difference_l15_15764

theorem votes_difference (T : ℕ) (V_a : ℕ) (V_f : ℕ) 
  (h1 : T = 330) (h2 : V_a = 40 * T / 100) (h3 : V_f = T - V_a) : V_f - V_a = 66 :=
by
  sorry

end votes_difference_l15_15764


namespace Greenwood_High_School_chemistry_students_l15_15428

theorem Greenwood_High_School_chemistry_students 
    (U : Finset ℕ) (B C P : Finset ℕ) 
    (hU_card : U.card = 20) 
    (hB_subset_U : B ⊆ U) 
    (hC_subset_U : C ⊆ U)
    (hP_subset_U : P ⊆ U)
    (hB_card : B.card = 10) 
    (hB_C_card : (B ∩ C).card = 4) 
    (hB_C_P_card : (B ∩ C ∩ P).card = 3) 
    (hAll_atleast_one : ∀ x ∈ U, x ∈ B ∨ x ∈ C ∨ x ∈ P) :
    C.card = 6 := 
by 
  sorry

end Greenwood_High_School_chemistry_students_l15_15428


namespace xy_sufficient_but_not_necessary_l15_15597

theorem xy_sufficient_but_not_necessary (x y : ℝ) : (x > 0 ∧ y > 0) → (xy > 0) ∧ ¬(xy > 0 → (x > 0 ∧ y > 0)) :=
by
  intros h
  sorry

end xy_sufficient_but_not_necessary_l15_15597


namespace total_art_cost_l15_15472

-- Definitions based on the conditions
def total_price_first_3_pieces (price_per_piece : ℤ) : ℤ :=
  price_per_piece * 3

def price_increase (price_per_piece : ℤ) : ℤ :=
  price_per_piece / 2

def total_price_all_arts (price_per_piece next_piece_price : ℤ) : ℤ :=
  (total_price_first_3_pieces price_per_piece) + next_piece_price

-- The proof problem statement
theorem total_art_cost : 
  ∀ (price_per_piece : ℤ),
  total_price_first_3_pieces price_per_piece = 45000 →
  next_piece_price = price_per_piece + price_increase price_per_piece →
  total_price_all_arts price_per_piece next_piece_price = 67500 :=
  by
    intros price_per_piece h1 h2
    sorry

end total_art_cost_l15_15472


namespace ratio_of_Y_share_l15_15378

theorem ratio_of_Y_share (total_profit share_diff X_share Y_share : ℝ) 
(h1 : total_profit = 700) (h2 : share_diff = 140) 
(h3 : X_share + Y_share = 700) (h4 : X_share - Y_share = 140) : 
Y_share / total_profit = 2 / 5 :=
sorry

end ratio_of_Y_share_l15_15378


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15239

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15239


namespace compute_binomial_10_3_eq_120_l15_15706

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15706


namespace mass_percentage_of_C_in_benzene_l15_15568

theorem mass_percentage_of_C_in_benzene :
  let C_molar_mass := 12.01 -- g/mol
  let H_molar_mass := 1.008 -- g/mol
  let benzene_C_atoms := 6
  let benzene_H_atoms := 6
  let C_total_mass := benzene_C_atoms * C_molar_mass
  let H_total_mass := benzene_H_atoms * H_molar_mass
  let benzene_total_mass := C_total_mass + H_total_mass
  let mass_percentage_C := (C_total_mass / benzene_total_mass) * 100
  (mass_percentage_C = 92.26) :=
by
  sorry

end mass_percentage_of_C_in_benzene_l15_15568


namespace probability_300_feet_or_less_l15_15032

noncomputable def calculate_probability : ℚ :=
  let gates := 16
  let distance := 75
  let max_distance := 300
  let initial_choices := gates
  let final_choices := gates - 1 -- because the final choice cannot be the same as the initial one
  let total_choices := initial_choices * final_choices
  let valid_choices :=
    (2 * 4 + 2 * 5 + 2 * 6 + 2 * 7 + 8 * 8) -- the total valid assignments as calculated in the solution
  (valid_choices : ℚ) / total_choices

theorem probability_300_feet_or_less : calculate_probability = 9 / 20 := 
by 
  sorry

end probability_300_feet_or_less_l15_15032


namespace number_of_y_axis_returns_l15_15136

-- Definitions based on conditions
noncomputable def unit_length : ℝ := 0.5
noncomputable def diagonal_length : ℝ := Real.sqrt 2 * unit_length
noncomputable def pen_length_cm : ℝ := 8000 * 100 -- converting meters to cm
noncomputable def circle_length (n : ℕ) : ℝ := ((3 + Real.sqrt 2) * n ^ 2 + 2 * n) * unit_length

-- The main theorem
theorem number_of_y_axis_returns : ∃ n : ℕ, circle_length n ≤ pen_length_cm ∧ circle_length (n+1) > pen_length_cm :=
sorry

end number_of_y_axis_returns_l15_15136


namespace smallest_consecutive_integers_product_l15_15745

theorem smallest_consecutive_integers_product (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 5040) : 
  n = 7 :=
sorry

end smallest_consecutive_integers_product_l15_15745


namespace correct_addition_by_changing_digit_l15_15489

theorem correct_addition_by_changing_digit :
  ∃ (d : ℕ), (d < 10) ∧ (d = 4) ∧
  (374 + (500 + d) + 286 = 1229 - 50) :=
by
  sorry

end correct_addition_by_changing_digit_l15_15489


namespace average_price_per_book_l15_15637

def books_from_shop1 := 42
def price_from_shop1 := 520
def books_from_shop2 := 22
def price_from_shop2 := 248

def total_books := books_from_shop1 + books_from_shop2
def total_price := price_from_shop1 + price_from_shop2
def average_price := total_price / total_books

theorem average_price_per_book : average_price = 12 := by
  sorry

end average_price_per_book_l15_15637


namespace find_value_of_m_l15_15924

/-- Given the parabola y = 4x^2 + 4x + 5 and the line y = 8mx + 8m intersect at exactly one point,
    prove the value of m^{36} + 1155 / m^{12} is 39236. -/
theorem find_value_of_m (m : ℝ) (h: ∃ x, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m ∧
  ∀ x₁ x₂, 4 * x₁^2 + 4 * x₁ + 5 = 8 * m * x₁ + 8 * m →
  4 * x₂^2 + 4 * x₂ + 5 = 8 * m * x₂ + 8 * m → x₁ = x₂) :
  m^36 + 1155 / m^12 = 39236 := 
sorry

end find_value_of_m_l15_15924


namespace add_and_round_58_29_l15_15799

def add_and_round_to_nearest_ten (a b : ℕ) : ℕ :=
  let sum := a + b
  let rounded_sum := if sum % 10 < 5 then sum - (sum % 10) else sum + (10 - sum % 10)
  rounded_sum

theorem add_and_round_58_29 : add_and_round_to_nearest_ten 58 29 = 90 := by
  sorry

end add_and_round_58_29_l15_15799


namespace greatest_integer_gcd_30_is_125_l15_15346

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l15_15346


namespace multiple_of_four_l15_15815

theorem multiple_of_four (n : ℕ) (h1 : ∃ k : ℕ, 12 + 4 * k = n) (h2 : 21 = (n - 12) / 4 + 1) : n = 96 := 
sorry

end multiple_of_four_l15_15815


namespace arithmetic_sequence_fifth_term_l15_15283

theorem arithmetic_sequence_fifth_term :
  ∀ (a₁ d n : ℕ), a₁ = 3 → d = 4 → n = 5 → a₁ + (n - 1) * d = 19 :=
by
  intros a₁ d n ha₁ hd hn
  sorry

end arithmetic_sequence_fifth_term_l15_15283


namespace find_eighth_number_l15_15974

-- Define the given problem with the conditions
noncomputable def sum_of_sixteen_numbers := 16 * 55
noncomputable def sum_of_first_eight_numbers := 8 * 60
noncomputable def sum_of_last_eight_numbers := 8 * 45
noncomputable def sum_of_last_nine_numbers := 9 * 50
noncomputable def sum_of_first_ten_numbers := 10 * 62

-- Define what we want to prove
theorem find_eighth_number :
  (exists (x : ℕ), x = 90) →
  sum_of_first_eight_numbers = 480 →
  sum_of_last_eight_numbers = 360 →
  sum_of_last_nine_numbers = 450 →
  sum_of_first_ten_numbers = 620 →
  sum_of_sixteen_numbers = 880 →
  x = 90 :=
by sorry

end find_eighth_number_l15_15974


namespace Carter_card_number_l15_15621

-- Definitions based on conditions
def Marcus_cards : ℕ := 210
def difference : ℕ := 58

-- Definition to infer the number of Carter's baseball cards
def Carter_cards : ℕ := Marcus_cards - difference

-- Theorem statement asserting the number of baseball cards Carter has
theorem Carter_card_number : Carter_cards = 152 := by
  sorry

end Carter_card_number_l15_15621


namespace find_three_digit_number_l15_15907

def is_valid_three_digit_number (M G U : ℕ) : Prop :=
  M ≠ G ∧ G ≠ U ∧ M ≠ U ∧ 
  0 ≤ M ∧ M ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧
  100 * M + 10 * G + U = (M + G + U) * (M + G + U - 2)

theorem find_three_digit_number : ∃ (M G U : ℕ), 
  is_valid_three_digit_number M G U ∧
  100 * M + 10 * G + U = 195 :=
by
  sorry

end find_three_digit_number_l15_15907


namespace largest_integer_less_than_100_leaving_remainder_4_l15_15249

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l15_15249


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15223

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15223


namespace students_taking_statistics_l15_15607

-- Definitions based on conditions
def total_students := 89
def history_students := 36
def history_or_statistics := 59
def history_not_statistics := 27

-- The proof problem
theorem students_taking_statistics : ∃ S : ℕ, S = 32 ∧
  ((history_students - history_not_statistics) + S - (history_students - history_not_statistics)) = history_or_statistics :=
by
  use 32
  sorry

end students_taking_statistics_l15_15607


namespace binomial_coefficient_10_3_l15_15723

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15723


namespace percentage_passed_both_l15_15015

-- Define the percentages of failures
def percentage_failed_hindi : ℕ := 34
def percentage_failed_english : ℕ := 44
def percentage_failed_both : ℕ := 22

-- Statement to prove
theorem percentage_passed_both : 
  (100 - (percentage_failed_hindi + percentage_failed_english - percentage_failed_both)) = 44 := by
  sorry

end percentage_passed_both_l15_15015


namespace find_smaller_number_l15_15321

-- Define the conditions
def condition1 (x y : ℤ) : Prop := x + y = 30
def condition2 (x y : ℤ) : Prop := x - y = 10

-- Define the theorem to prove the smaller number is 10
theorem find_smaller_number (x y : ℤ) (h1 : condition1 x y) (h2 : condition2 x y) : y = 10 := 
sorry

end find_smaller_number_l15_15321


namespace marbles_end_of_day_l15_15068

theorem marbles_end_of_day :
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54 :=
by
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  show initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54
  sorry

end marbles_end_of_day_l15_15068


namespace divisibility_polynomial_l15_15636

variables {a m x n : ℕ}

theorem divisibility_polynomial (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) :=
by
  sorry

end divisibility_polynomial_l15_15636


namespace dana_total_earnings_l15_15061

-- Define the constants for Dana's hourly rate and hours worked each day
def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

-- Define the total earnings calculation function
def total_earnings (rate : ℝ) (hours1 hours2 hours3 : ℝ) : ℝ :=
  rate * hours1 + rate * hours2 + rate * hours3

-- The main statement
theorem dana_total_earnings : total_earnings hourly_rate friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end dana_total_earnings_l15_15061


namespace convert_38_to_binary_l15_15060

theorem convert_38_to_binary :
  let decimal_to_binary (n : ℕ) : list ℕ :=
    if n = 0 then []
    else (n % 2) :: decimal_to_binary (n / 2)
  decimal_to_binary 38.reverse = [1, 0, 0, 1, 1, 0] :=
by
  sorry

end convert_38_to_binary_l15_15060


namespace dollars_tina_l15_15968

open Real

theorem dollars_tina (P Q R S T : ℤ)
  (h1 : abs (P - Q) = 21)
  (h2 : abs (Q - R) = 9)
  (h3 : abs (R - S) = 7)
  (h4 : abs (S - T) = 6)
  (h5 : abs (T - P) = 13)
  (h6 : P + Q + R + S + T = 86) :
  T = 16 :=
sorry

end dollars_tina_l15_15968


namespace largest_possible_p_l15_15536

theorem largest_possible_p (m n p : ℕ) (h1 : m > 2) (h2 : n > 2) (h3 : p > 2) (h4 : gcd m n = 1) (h5 : gcd n p = 1) (h6 : gcd m p = 1)
  (h7 : (1/m : ℚ) + (1/n : ℚ) + (1/p : ℚ) = 1/2) : p ≤ 42 :=
by sorry

end largest_possible_p_l15_15536


namespace ram_shyam_weight_ratio_l15_15155

theorem ram_shyam_weight_ratio
    (R S : ℝ)
    (h1 : 1.10 * R + 1.22 * S = 82.8)
    (h2 : R + S = 72) :
    R / S = 7 / 5 :=
by sorry

end ram_shyam_weight_ratio_l15_15155


namespace total_plums_picked_l15_15962

-- Conditions
def Melanie_plums : ℕ := 4
def Dan_plums : ℕ := 9
def Sally_plums : ℕ := 3

-- Proof statement
theorem total_plums_picked : Melanie_plums + Dan_plums + Sally_plums = 16 := by
  sorry

end total_plums_picked_l15_15962


namespace largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15242

theorem largest_int_lt_100_with_remainder_4_when_div_by_7 : 
  ∃ n : ℤ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℤ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
begin
  use 95,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm,
    cases hm with hm1 hm2,
    have k_m_geq : m = 7 * ((m - 4) / 7) + 4 := by ring,
    have H : ∃ k : ℤ, m = 7 * k + 4 := ⟨(m - 4) / 7, k_m_geq⟩,
    obtain ⟨k, Hk⟩ := H,
    have : 7 * k + 4 < 100 := by { rw Hk at hm1, exact hm1 },
    replace := int.lt_ceil.mp (by linarith [1]),
    linarith,
  },
  sorry -- Additional proof required to complete the theorem
end

end largest_int_lt_100_with_remainder_4_when_div_by_7_l15_15242


namespace greatest_integer_with_gcd_l15_15353

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l15_15353


namespace sixth_graders_l15_15003

theorem sixth_graders (total_students sixth_graders seventh_graders : ℕ)
    (h1 : seventh_graders = 64)
    (h2 : 32 * total_students = 64 * 100)
    (h3 : sixth_graders * 100 = 38 * total_students) :
    sixth_graders = 76 := by
  sorry

end sixth_graders_l15_15003


namespace solution_set_of_inequality_l15_15259

theorem solution_set_of_inequality (x : ℝ) : (2 * x + 3) * (4 - x) > 0 ↔ -3 / 2 < x ∧ x < 4 :=
by
  sorry

end solution_set_of_inequality_l15_15259


namespace calculator_to_protractors_l15_15682

def calculator_to_rulers (c: ℕ) : ℕ := 100 * c
def rulers_to_compasses (r: ℕ) : ℕ := (r * 30) / 10
def compasses_to_protractors (p: ℕ) : ℕ := (p * 50) / 25

theorem calculator_to_protractors (c: ℕ) : compasses_to_protractors (rulers_to_compasses (calculator_to_rulers c)) = 600 * c :=
by
  sorry

end calculator_to_protractors_l15_15682


namespace find_somus_age_l15_15016

def somus_current_age (S F : ℕ) := S = F / 3
def somus_age_7_years_ago (S F : ℕ) := (S - 7) = (F - 7) / 5

theorem find_somus_age (S F : ℕ) 
  (h1 : somus_current_age S F) 
  (h2 : somus_age_7_years_ago S F) : S = 14 :=
sorry

end find_somus_age_l15_15016


namespace positive_difference_solutions_abs_eq_30_l15_15998

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l15_15998


namespace compute_binomial_10_3_eq_120_l15_15707

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15707


namespace distinct_real_roots_k_root_condition_k_l15_15925

-- Part (1) condition: The quadratic equation has two distinct real roots
theorem distinct_real_roots_k (k : ℝ) : (∃ x : ℝ, x^2 + 2*x + k = 0) ∧ (∀ x y : ℝ, x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0 → x ≠ y) → k < 1 := 
sorry

-- Part (2) condition: m is a root and satisfies m^2 + 2m = 2
theorem root_condition_k (m k : ℝ) : m^2 + 2*m = 2 → m^2 + 2*m + k = 0 → k = -2 := 
sorry

end distinct_real_roots_k_root_condition_k_l15_15925


namespace problem_equivalent_statement_l15_15612

-- Define the operations provided in the problem
inductive Operation
| add
| sub
| mul
| div

open Operation

-- Represents the given equation with the specified operation
def applyOperation (op : Operation) (a b : ℕ) : ℕ :=
  match op with
  | add => a + b
  | sub => a - b
  | mul => a * b
  | div => a / b

theorem problem_equivalent_statement : 
  (∀ (op : Operation), applyOperation op 8 2 - 5 + 7 - (3^2 - 4) ≠ 6) → (¬ ∃ op : Operation, applyOperation op 8 2 = 9) := 
by
  sorry

end problem_equivalent_statement_l15_15612


namespace greatest_integer_with_gcd_l15_15354

theorem greatest_integer_with_gcd (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  -- The proof would go here
  sorry

example : ∃ n < 150, Nat.gcd n 30 = 5 ∧ ∀ m < 150, Nat.gcd m 30 = 5 → m ≤ 145 :=
by
  use 145
  split
  · exact Nat.lt_succ_self 149
  split
  · simp [Nat.gcd_comm]
  · intros m m_lt m_gcd
    exact greatest_integer_with_gcd m m_lt m_gcd

end greatest_integer_with_gcd_l15_15354


namespace largest_integer_less_than_100_with_remainder_4_l15_15227

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l15_15227


namespace temp_pot_C_to_F_l15_15158

-- Definitions
def boiling_point_C : ℕ := 100
def boiling_point_F : ℕ := 212
def melting_point_C : ℕ := 0
def melting_point_F : ℕ := 32
def temp_pot_C : ℕ := 55
def celsius_to_fahrenheit (c : ℕ) : ℕ := (c * 9 / 5) + 32

-- Theorem to be proved
theorem temp_pot_C_to_F : celsius_to_fahrenheit temp_pot_C = 131 := by
  sorry

end temp_pot_C_to_F_l15_15158


namespace find_f_2008_l15_15958

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero : f 0 = 2008

axiom f_inequality1 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality2 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 : f 2008 = 2^2008 + 2007 :=
sorry

end find_f_2008_l15_15958


namespace distance_to_Rock_Mist_Mountains_l15_15031

theorem distance_to_Rock_Mist_Mountains (d_Sky_Falls : ℕ) (multiplier : ℕ) (d_Rock_Mist : ℕ) :
  d_Sky_Falls = 8 → multiplier = 50 → d_Rock_Mist = d_Sky_Falls * multiplier → d_Rock_Mist = 400 :=
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end distance_to_Rock_Mist_Mountains_l15_15031


namespace exists_n_for_perfect_square_l15_15794

theorem exists_n_for_perfect_square (k : ℕ) (hk_pos : k > 0) :
  ∃ n : ℕ, n > 0 ∧ ∃ a : ℕ, a^2 = n * 2^k - 7 :=
by
  sorry

end exists_n_for_perfect_square_l15_15794


namespace smallest_number_divisible_conditions_l15_15848

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l15_15848


namespace similar_triangles_ratios_l15_15631

-- Define the context
variables {a b c a' b' c' : ℂ}

-- Define the statement of the problem
theorem similar_triangles_ratios (h_sim : ∃ z : ℂ, z ≠ 0 ∧ b - a = z * (b' - a') ∧ c - a = z * (c' - a')) :
  (b - a) / (c - a) = (b' - a') / (c' - a') :=
sorry

end similar_triangles_ratios_l15_15631


namespace compute_binomial_10_3_eq_120_l15_15710

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15710


namespace anton_thought_number_l15_15039

def is_match_in_one_digit_place (a b : Nat) : Prop :=
  let a_digits := [(a / 100) % 10, (a / 10) % 10, a % 10]
  let b_digits := [(b / 100) % 10, (b / 10) % 10, b % 10]
  (a_digits.zip b_digits).count (λ (x : Nat × Nat), x.fst = x.snd) = 1

theorem anton_thought_number : ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧
  is_match_in_one_digit_place n 109 ∧
  is_match_in_one_digit_place n 704 ∧
  is_match_in_one_digit_place n 124 ∧
  n = 729 :=
by
  sorry

end anton_thought_number_l15_15039


namespace no_form3000001_is_perfect_square_l15_15788

theorem no_form3000001_is_perfect_square (n : ℕ) : 
  ∀ k : ℤ, (3 * 10^n + 1 ≠ k^2) :=
by
  sorry

end no_form3000001_is_perfect_square_l15_15788


namespace original_quadrilateral_area_l15_15577

theorem original_quadrilateral_area :
  let deg45 := (Real.pi / 4)
  let h := 1 * Real.sin deg45
  let base_bottom := 1 + 2 * h
  let area_perspective := 0.5 * (1 + base_bottom) * h
  let area_original := area_perspective * (2 * Real.sqrt 2)
  area_original = 2 + Real.sqrt 2 := by
  sorry

end original_quadrilateral_area_l15_15577


namespace crayons_received_l15_15486

theorem crayons_received (crayons_left : ℕ) (crayons_lost_given_away : ℕ) (lost_twice_given : ∃ (G L : ℕ), L = 2 * G ∧ L + G = crayons_lost_given_away) :
  crayons_left = 2560 →
  crayons_lost_given_away = 9750 →
  ∃ (total_crayons_received : ℕ), total_crayons_received = 12310 :=
by
  intros h1 h2
  obtain ⟨G, L, hL, h_sum⟩ := lost_twice_given
  sorry -- Proof goes here

end crayons_received_l15_15486


namespace sequence_bk_bl_sum_l15_15476

theorem sequence_bk_bl_sum (b : ℕ → ℕ) (m : ℕ) 
  (h_pairwise_distinct : ∀ i j, i ≠ j → b i ≠ b j)
  (h_b0 : b 0 = 0)
  (h_b_lt_2n : ∀ n, 0 < n → b n < 2 * n) :
  ∃ k ℓ : ℕ, b k + b ℓ = m := 
  sorry

end sequence_bk_bl_sum_l15_15476


namespace largest_multiple_of_9_less_than_100_l15_15164

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ (∀ k, k < 100 → k % 9 = 0 → k ≤ n) :=
by {
  use 99,
  split,
  { exact lt_of_le_of_ne (nat.le_of_dvd (by norm_num) (by norm_num)) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by use 11) },
  { intros k hk hkm, apply nat.mul_le_mul_left 9 (le_of_lt $ nat.div_lt_self hk $ by norm_num), exact le_of_lt_succ (nat.succ_le_of_lt (nat.div_lt_self hk $ by norm_num)) }
  sorry
}

end largest_multiple_of_9_less_than_100_l15_15164


namespace possible_number_of_friends_l15_15870

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l15_15870


namespace comb_10_3_eq_120_l15_15731

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15731


namespace inequality_of_sum_l15_15630

theorem inequality_of_sum 
  (a : ℕ → ℝ)
  (h : ∀ n m, 0 ≤ n → n < m → a n < a m) :
  (0 < a 1 ->
  0 < a 2 ->
  0 < a 3 ->
  0 < a 4 ->
  0 < a 5 ->
  0 < a 6 ->
  0 < a 7 ->
  0 < a 8 ->
  0 < a 9 ->
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / (a 3 + a 6 + a 9) < 3) :=
by
  intros
  sorry

end inequality_of_sum_l15_15630


namespace expression_value_l15_15934

theorem expression_value (x y : ℝ) (h : x + y = -1) : 
  x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l15_15934


namespace find_common_ratio_l15_15263

theorem find_common_ratio (a_1 q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS1 : S 1 = a_1)
  (hS2 : S 2 = a_1 * (1 + q))
  (hS3 : S 3 = a_1 * (1 + q + q^2))
  (ha2 : a 2 = a_1 * q)
  (ha3 : a 3 = a_1 * q^2)
  (hcond : 2 * (S 1 + 2 * a 2) = S 3 + a 3 + S 2 + a 2) :
  q = -1/2 :=
by
  sorry

end find_common_ratio_l15_15263


namespace quotient_is_seven_l15_15965

def dividend : ℕ := 22
def divisor : ℕ := 3
def remainder : ℕ := 1

theorem quotient_is_seven : ∃ quotient : ℕ, dividend = (divisor * quotient) + remainder ∧ quotient = 7 := by
  sorry

end quotient_is_seven_l15_15965


namespace train_crossing_time_l15_15424

/-- Given the conditions that a moving train requires 10 seconds to pass a pole,
    its speed is 36 km/h, and the length of a stationary train is 300 meters,
    prove that the moving train takes 40 seconds to cross the stationary train. -/
theorem train_crossing_time (t_pole : ℕ)
  (v_kmh : ℕ)
  (length_stationary : ℕ) :
  t_pole = 10 →
  v_kmh = 36 →
  length_stationary = 300 →
  ∃ t_cross : ℕ, t_cross = 40 :=
by
  intros h1 h2 h3
  sorry

end train_crossing_time_l15_15424


namespace spring_bud_cup_eq_289_l15_15798

theorem spring_bud_cup_eq_289 (x : ℕ) (h : x + x = 578) : x = 289 :=
sorry

end spring_bud_cup_eq_289_l15_15798


namespace find_k_l15_15477

theorem find_k (Z K : ℤ) (h1 : 2000 < Z) (h2 : Z < 3000) (h3 : K > 1) (h4 : Z = K * K^2) (h5 : ∃ n : ℤ, n^3 = Z) : K = 13 :=
by
-- Solution omitted
sorry

end find_k_l15_15477


namespace part_I_solution_part_II_solution_l15_15923

-- Definition of the function f(x)
def f (x a : ℝ) := |x - a| + |2 * x - 1|

-- Part (I) when a = 1, find the solution set for f(x) ≤ 2
theorem part_I_solution (x : ℝ) : f x 1 ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 :=
by sorry

-- Part (II) if the solution set for f(x) ≤ |2x + 1| contains [1/2, 1], find the range of a
theorem part_II_solution (a : ℝ) :
  (∀ x : ℝ, 1 / 2 ≤ x ∧ x ≤ 1 → f x a ≤ |2 * x + 1|) → -1 ≤ a ∧ a ≤ 5 / 2 :=
by sorry

end part_I_solution_part_II_solution_l15_15923


namespace green_or_blue_marble_probability_l15_15009

theorem green_or_blue_marble_probability :
  (4 + 3 : ℝ) / (4 + 3 + 8) = 0.4667 := by
  sorry

end green_or_blue_marble_probability_l15_15009


namespace total_marks_more_than_physics_l15_15506

-- Definitions of variables for marks in different subjects
variables (P C M : ℕ)

-- Conditions provided in the problem
def total_marks_condition (P : ℕ) (C : ℕ) (M : ℕ) : Prop := P + C + M > P
def average_chemistry_math_marks (C : ℕ) (M : ℕ) : Prop := (C + M) / 2 = 55

-- The main proof statement: Proving the difference in total marks and physics marks
theorem total_marks_more_than_physics 
    (h1 : total_marks_condition P C M)
    (h2 : average_chemistry_math_marks C M) :
  (P + C + M) - P = 110 := 
sorry

end total_marks_more_than_physics_l15_15506


namespace binom_10_3_l15_15687

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l15_15687


namespace meaning_of_probability_l15_15602

-- Definitions

def probability_of_winning (p : ℚ) : Prop :=
  p = 1 / 4

-- Theorem statement
theorem meaning_of_probability :
  probability_of_winning (1 / 4) →
  ∀ n : ℕ, (n ≠ 0) → (n / 4 * 4) = n :=
by
  -- Placeholder proof
  sorry

end meaning_of_probability_l15_15602


namespace circles_tangent_internally_l15_15131

theorem circles_tangent_internally 
  (x y : ℝ) 
  (h : x^4 - 16 * x^2 + 2 * x^2 * y^2 - 16 * y^2 + y^4 = 4 * x^3 + 4 * x * y^2 - 64 * x) :
  ∃ c₁ c₂ : ℝ × ℝ, 
    (c₁ = (0, 0)) ∧ (c₂ = (2, 0)) ∧ 
    ((x - c₁.1)^2 + (y - c₁.2)^2 = 16) ∧ 
    ((x - c₂.1)^2 + (y - c₂.2)^2 = 4) ∧
    dist c₁ c₂ = 2 := 
sorry

end circles_tangent_internally_l15_15131


namespace find_all_good_sets_l15_15735

def is_good_set (A : Finset ℕ) : Prop :=
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) = 1) ∧
  (∀ (b c : ℕ), b ∈ A → c ∈ A → b ≠ c → ∃ (a : ℕ), a ∈ A ∧ a ≠ b ∧ a ≠ c ∧ (b * c) % a = 0)

theorem find_all_good_sets : ∀ (A : Finset ℕ), is_good_set A ↔ 
  (A = {a, b, a * b} ∧ Nat.gcd a b = 1) ∨ 
  ∃ (p q r : ℕ), Nat.gcd p q = 1 ∧ Nat.gcd q r = 1 ∧ Nat.gcd r p = 1 ∧ A = {p * q, q * r, r * p} :=
by
  sorry

end find_all_good_sets_l15_15735


namespace find_k_values_l15_15121

open Set

def A : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def B (k : ℝ) : Set ℝ := {x | x^2 - (k + 1) * x + k = 0}

theorem find_k_values (k : ℝ) : (A ∩ B k = B k) ↔ k ∈ ({1, -3} : Set ℝ) := by
  sorry

end find_k_values_l15_15121


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15359

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l15_15359


namespace amount_paid_for_peaches_l15_15640

def total_spent := 23.86
def cherries_spent := 11.54
def peaches_spent := 12.32

theorem amount_paid_for_peaches :
  total_spent - cherries_spent = peaches_spent :=
sorry

end amount_paid_for_peaches_l15_15640


namespace smallestC_l15_15620

def isValidFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧
  f 1 = 1 ∧
  (∀ x y, 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1 → f x + f y ≤ f (x + y))

theorem smallestC (f : ℝ → ℝ) (h : isValidFunction f) : ∃ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ c * x) ∧
  (∀ d, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ d * x) → 2 ≤ d) :=
sorry

end smallestC_l15_15620


namespace largest_multiple_of_9_less_than_100_l15_15172

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l15_15172


namespace number_of_even_permutations_l15_15740

variable {α : Type*} [DecidableEq α]

def is_even_permutation (σ : equiv.perm α) : Prop :=
  equiv.perm.sign σ = 1

def problem (a : Fin 6 → ℕ) : Prop :=
  (∀ i : Fin 6, a i ∈ Finset.univ.map ⟨Nat.succ, Nat.succ_injective⟩.to_fun) ∧
  ∏ i, (a i + i + 1) / 3 > 120

theorem number_of_even_permutations
  : ∃ (a : Finset (Fin 6 → ℕ)), a.card = 360 ∧ ∀ f ∈ a, problem f :=
sorry

end number_of_even_permutations_l15_15740


namespace sum_of_edge_lengths_of_truncated_octahedron_prism_l15_15188

-- Define the vertices, edge length, and the assumption of the prism being a truncated octahedron
def prism_vertices : ℕ := 24
def edge_length : ℕ := 5
def truncated_octahedron_edges : ℕ := 36

-- The Lean statement to prove the sum of edge lengths
theorem sum_of_edge_lengths_of_truncated_octahedron_prism :
  prism_vertices = 24 ∧ edge_length = 5 ∧ truncated_octahedron_edges = 36 →
  truncated_octahedron_edges * edge_length = 180 :=
by
  sorry

end sum_of_edge_lengths_of_truncated_octahedron_prism_l15_15188


namespace min_value_inverse_sum_l15_15747

variable {x y : ℝ}

theorem min_value_inverse_sum (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : (1/x + 1/y) ≥ 1 :=
  sorry

end min_value_inverse_sum_l15_15747


namespace victor_percentage_of_marks_l15_15653

theorem victor_percentage_of_marks (marks_obtained max_marks : ℝ) (percentage : ℝ) 
  (h_marks_obtained : marks_obtained = 368) 
  (h_max_marks : max_marks = 400) 
  (h_percentage : percentage = (marks_obtained / max_marks) * 100) : 
  percentage = 92 := by
sorry

end victor_percentage_of_marks_l15_15653


namespace no_odd_tens_digit_in_square_l15_15443

theorem no_odd_tens_digit_in_square (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n > 0) (h₃ : n < 100) : 
  (n * n / 10) % 10 % 2 = 0 := 
sorry

end no_odd_tens_digit_in_square_l15_15443


namespace picnic_basket_cost_l15_15813

theorem picnic_basket_cost :
  let sandwich_cost := 5
  let fruit_salad_cost := 3
  let soda_cost := 2
  let snack_bag_cost := 4
  let num_people := 4
  let num_sodas_per_person := 2
  let num_snack_bags := 3
  (num_people * sandwich_cost) + (num_people * fruit_salad_cost) + (num_people * num_sodas_per_person * soda_cost) + (num_snack_bags * snack_bag_cost) = 60 :=
by
  sorry

end picnic_basket_cost_l15_15813


namespace value_of_x_is_two_l15_15322

theorem value_of_x_is_two (x : ℝ) (h : x + x^3 = 10) : x = 2 :=
sorry

end value_of_x_is_two_l15_15322


namespace range_of_a_l15_15928

theorem range_of_a (a : ℝ) (h : a > 0) :
  let A := {x : ℝ | x^2 + 2 * x - 8 > 0}
  let B := {x : ℝ | x^2 - 2 * a * x + 4 ≤ 0}
  (∃! x : ℤ, (x : ℝ) ∈ A ∩ B) → (13 / 6 ≤ a ∧ a < 5 / 2) :=
by
  sorry

end range_of_a_l15_15928


namespace Mirella_read_purple_books_l15_15562

theorem Mirella_read_purple_books (P : ℕ) 
  (pages_per_purple_book : ℕ := 230)
  (pages_per_orange_book : ℕ := 510)
  (orange_books_read : ℕ := 4)
  (extra_orange_pages : ℕ := 890)
  (total_orange_pages : ℕ := orange_books_read * pages_per_orange_book)
  (total_purple_pages : ℕ := P * pages_per_purple_book)
  (condition : total_orange_pages - total_purple_pages = extra_orange_pages) :
  P = 5 := 
by 
  sorry

end Mirella_read_purple_books_l15_15562


namespace greatest_int_with_gcd_five_l15_15332

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l15_15332


namespace projectile_height_reaches_45_at_t_0_5_l15_15805

noncomputable def quadratic (a b c : ℝ) : ℝ → ℝ :=
  λ t => a * t^2 + b * t + c

theorem projectile_height_reaches_45_at_t_0_5 :
  ∃ t : ℝ, quadratic (-16) 98.5 (-45) t = 45 ∧ 0 ≤ t ∧ t = 0.5 :=
by
  sorry

end projectile_height_reaches_45_at_t_0_5_l15_15805


namespace proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l15_15098

theorem proportion_false_if_x_is_0_75 (x : ℚ) (h1 : x = 0.75) : ¬ (x / 2 = 2 / 6) :=
by sorry

theorem correct_value_of_x_in_proportion (x : ℚ) (h1 : x / 2 = 2 / 6) : x = 2 / 3 :=
by sorry

end proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l15_15098


namespace largest_integer_less_than_100_with_remainder_4_l15_15225

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l15_15225


namespace john_paid_more_l15_15470

-- Define the required variables
def original_price : ℝ := 84.00000000000009
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

-- Define John and Jane's payments
def discounted_price : ℝ := original_price * (1 - discount_rate)
def johns_tip : ℝ := tip_rate * original_price
def johns_total_payment : ℝ := original_price + johns_tip
def janes_tip : ℝ := tip_rate * discounted_price
def janes_total_payment : ℝ := discounted_price + janes_tip

-- Calculate the difference
def payment_difference : ℝ := johns_total_payment - janes_total_payment

-- Statement to prove the payment difference equals $9.66
theorem john_paid_more : payment_difference = 9.66 := by
  sorry

end john_paid_more_l15_15470


namespace investor_share_price_l15_15861

theorem investor_share_price (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) (price_per_share : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 40 →
  roi = 0.25 →
  ((dividend_rate * face_value) / price_per_share) = roi →
  price_per_share = 20 :=
by 
  intros h1 h2 h3 h4
  sorry

end investor_share_price_l15_15861


namespace find_angle_l15_15584

theorem find_angle (x : ℝ) (h : 90 - x = 2 * x + 15) : x = 25 :=
by
  sorry

end find_angle_l15_15584


namespace regular_polygon_sides_l15_15463

theorem regular_polygon_sides (n : ℕ) (h : 0 < n) (h_angle : (n - 2) * 180 = 144 * n) :
  n = 10 :=
sorry

end regular_polygon_sides_l15_15463


namespace sixth_graders_count_l15_15002

theorem sixth_graders_count (total_students seventh_graders_percentage sixth_graders_percentage : ℝ)
                            (seventh_graders_count : ℕ)
                            (h1 : seventh_graders_percentage = 0.32)
                            (h2 : seventh_graders_count = 64)
                            (h3 : sixth_graders_percentage = 0.38)
                            (h4 : seventh_graders_count = seventh_graders_percentage * total_students) :
                            sixth_graders_percentage * total_students = 76 := by
  sorry

end sixth_graders_count_l15_15002


namespace solve_fraction_zero_l15_15154

theorem solve_fraction_zero (x : ℕ) (h : x ≠ 0) (h_eq : (x - 1) / x = 0) : x = 1 := by 
  sorry

end solve_fraction_zero_l15_15154


namespace minimum_games_for_80_percent_l15_15801

theorem minimum_games_for_80_percent :
  ∃ N : ℕ, ( ∀ N' : ℕ, (1 + N') / (5 + N') * 100 < 80 → N < N') ∧ (1 + N) / (5 + N) * 100 ≥ 80 :=
sorry

end minimum_games_for_80_percent_l15_15801


namespace melted_ice_cream_depth_l15_15676

noncomputable def radius_sphere : ℝ := 3
noncomputable def radius_cylinder : ℝ := 10
noncomputable def height_cylinder : ℝ := 36 / 100

theorem melted_ice_cream_depth :
  (4 / 3) * Real.pi * radius_sphere^3 = Real.pi * radius_cylinder^2 * height_cylinder :=
by
  sorry

end melted_ice_cream_depth_l15_15676


namespace smallest_three_digit_multiple_of_eleven_l15_15824

theorem smallest_three_digit_multiple_of_eleven : ∃ n, n = 110 ∧ 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n := by
  sorry

end smallest_three_digit_multiple_of_eleven_l15_15824


namespace factorize_expression_l15_15070

theorem factorize_expression (x : ℝ) : x^3 - 2 * x^2 + x = x * (x - 1)^2 :=
by sorry

end factorize_expression_l15_15070


namespace carterHas152Cards_l15_15623

-- Define the number of baseball cards Marcus has.
def marcusCards : Nat := 210

-- Define the number of baseball cards Carter has.
def carterCards : Nat := marcusCards - 58

-- Theorem to prove Carter's baseball cards total 152 given the conditions.
theorem carterHas152Cards (h1 : marcusCards = 210) (h2 : marcusCards = carterCards + 58) : carterCards = 152 :=
by
  -- Proof omitted for this exercise
  sorry

end carterHas152Cards_l15_15623


namespace evaluate_series_l15_15176

theorem evaluate_series : 1 + (1 / 2) + (1 / 4) + (1 / 8) = 15 / 8 := by
  sorry

end evaluate_series_l15_15176


namespace largest_integer_remainder_condition_l15_15243

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l15_15243


namespace heights_inequality_l15_15296

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h : a ≤ b ∧ b ≤ c) : 
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) := 
sorry

end heights_inequality_l15_15296


namespace largest_multiple_of_9_less_than_100_l15_15162

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l15_15162


namespace sally_picked_3_plums_l15_15625

theorem sally_picked_3_plums (melanie_picked : ℕ) (dan_picked : ℕ) (total_picked : ℕ) 
    (h1 : melanie_picked = 4) (h2 : dan_picked = 9) (h3 : total_picked = 16) : 
    total_picked - (melanie_picked + dan_picked) = 3 := 
by 
  -- proof steps go here
  sorry

end sally_picked_3_plums_l15_15625


namespace complex_div_i_l15_15766

open Complex

theorem complex_div_i (z : ℂ) (hz : z = -2 - i) : z / i = -1 + 2 * i :=
by
  sorry

end complex_div_i_l15_15766


namespace math_problem_l15_15276

theorem math_problem (a b c d x : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |x| = 2) :
  x^4 + c * d * x^2 - a - b = 20 :=
sorry

end math_problem_l15_15276


namespace largest_int_less_than_100_by_7_l15_15235

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l15_15235


namespace square_side_length_l15_15417

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l15_15417


namespace ceil_sq_values_count_l15_15462

theorem ceil_sq_values_count (y : ℝ) (hy : ⌈y⌉ = 15) : 
  (finset.range (⌈15^2⌉ - ⌈14^2⌉ + 1)).card = 29 :=
by
  let lower_bound := 14
  let upper_bound := 15
  have h1 : lower_bound < y := sorry
  have h2 : y ≤ upper_bound := sorry
  have sq_lower_bound : lower_bound^2 < y^2 := sorry
  have sq_upper_bound : y^2 ≤ upper_bound^2 := sorry
  let vals := (finset.range (⌈upper_bound^2⌉ - ⌈lower_bound^2⌉ + 1)).val
  have h3 : sq_lower_bound ≥ 196 := sorry
  have h4 : sq_upper_bound ≤ 225 := sorry
  rw finset.card
  exact sorry

end ceil_sq_values_count_l15_15462


namespace find_largest_integer_l15_15215

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l15_15215


namespace largest_integer_remainder_condition_l15_15247

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l15_15247


namespace makeup_exam_probability_l15_15763

theorem makeup_exam_probability (total_students : ℕ) (students_in_makeup_exam : ℕ)
  (h1 : total_students = 42) (h2 : students_in_makeup_exam = 3) :
  (students_in_makeup_exam : ℚ) / total_students = 1 / 14 := by
  sorry

end makeup_exam_probability_l15_15763


namespace largest_integer_less_than_100_with_remainder_4_l15_15229

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l15_15229


namespace anton_thought_number_l15_15037

def matches_one_place (n guessed : ℕ) : Prop :=
  let digits (x : ℕ) := (x / 100, (x % 100) / 10, x % 10)
  in (digits n).1 = (digits guessed).1 ∨ (digits n).2 = (digits guessed).2 ∨ (digits n).3 = (digits guessed).3

theorem anton_thought_number : 
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    matches_one_place n 109 ∧ 
    matches_one_place n 704 ∧ 
    matches_one_place n 124 ∧ 
    ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ matches_one_place m 109 ∧ matches_one_place m 704 ∧ matches_one_place m 124) → m = n :=
  ∃ n = 729 ∧ sorry

end anton_thought_number_l15_15037


namespace find_other_intersection_point_l15_15810

-- Definitions
def parabola_eq (x : ℝ) : ℝ := x^2 - 2 * x - 3
def intersection_point1 : Prop := parabola_eq (-1) = 0
def intersection_point2 : Prop := parabola_eq 3 = 0

-- Proof problem
theorem find_other_intersection_point :
  intersection_point1 → intersection_point2 := by
  sorry

end find_other_intersection_point_l15_15810


namespace initial_ratio_of_milk_water_l15_15533

theorem initial_ratio_of_milk_water (M W : ℝ) (H1 : M + W = 85) (H2 : M / (W + 5) = 3) : M / W = 27 / 7 :=
by sorry

end initial_ratio_of_milk_water_l15_15533


namespace tangent_curve_line_l15_15087

/-- Given the line y = x + 1 and the curve y = ln(x + a) are tangent, prove that the value of a is 2. -/
theorem tangent_curve_line (a : ℝ) :
  (∃ x₀ y₀, y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 / (x₀ + a) = 1)) → a = 2 :=
by
  sorry

end tangent_curve_line_l15_15087


namespace cos_triple_angle_l15_15095

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l15_15095


namespace combined_pumps_fill_time_l15_15363

theorem combined_pumps_fill_time (small_pump_time large_pump_time : ℝ) (h1 : small_pump_time = 4) (h2 : large_pump_time = 1/2) : 
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  (1 / combined_rate) = 4 / 9 :=
by
  -- Definitions of rates
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  
  -- Using placeholder for the proof.
  sorry

end combined_pumps_fill_time_l15_15363


namespace square_side_length_l15_15391

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l15_15391


namespace factor_x_minus_1_l15_15619

theorem factor_x_minus_1 (P Q R S : Polynomial ℂ) : 
  (P.eval 1 = 0) → 
  (P.eval (x^5) + x * Q.eval (x^5) + x^2 * R.eval (x^5) 
  = (x^4 + x^3 + x^2 + x + 1) * S.eval (x)) :=
sorry

end factor_x_minus_1_l15_15619


namespace peanut_butter_revenue_l15_15781

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l15_15781


namespace smallest_n_for_polygon_cutting_l15_15153

theorem smallest_n_for_polygon_cutting : 
  ∃ n : ℕ, (∃ k : ℕ, n - 2 = k * 31) ∧ (∃ k' : ℕ, n - 2 = k' * 65) ∧ n = 2017 :=
sorry

end smallest_n_for_polygon_cutting_l15_15153


namespace solve_log_eq_l15_15841

theorem solve_log_eq (x : ℝ) (hx : x > 0) 
  (h : 4^(Real.log x / Real.log 9 * 2) + Real.log 3 / (1/2 * Real.log 3) = 
       0.2 * (4^(2 + Real.log x / Real.log 9) - 4^(Real.log x / Real.log 9))) :
  x = 1 ∨ x = 3 :=
by sorry

end solve_log_eq_l15_15841


namespace smallest_possible_difference_after_101_years_l15_15900

theorem smallest_possible_difference_after_101_years {D E : ℤ} 
  (init_dollar : D = 6) 
  (init_euro : E = 7)
  (transformations : ∀ D E : ℤ, 
    (D', E') = (D + E, 2 * D + 1) ∨ (D', E') = (D + E, 2 * D - 1) ∨ 
    (D', E') = (D + E, 2 * E + 1) ∨ (D', E') = (D + E, 2 * E - 1)) :
  ∃ n_diff : ℤ, 101 = 2 * n_diff ∧ n_diff = 2 :=
sorry

end smallest_possible_difference_after_101_years_l15_15900


namespace largest_integer_less_than_100_leaving_remainder_4_l15_15253

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l15_15253


namespace intersect_circle_line_l15_15101

theorem intersect_circle_line (k m : ℝ) : 
  (∃ (x y : ℝ), y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 :=
by
  -- This statement follows from the conditions given in the problem
  -- You can use implicit for pure documentation
  -- We include a sorry here to skip the proof
  sorry

end intersect_circle_line_l15_15101


namespace solutions_to_equation_l15_15797

variable (x : ℝ)

def original_eq : Prop :=
  (3 * x - 9) / (x^2 - 6 * x + 8) = (x + 1) / (x - 2)

theorem solutions_to_equation : (original_eq 1 ∧ original_eq 5) :=
by
  sorry

end solutions_to_equation_l15_15797


namespace gcm_less_than_90_l15_15656

theorem gcm_less_than_90 (a b : ℕ) (h1 : a = 8) (h2 : b = 12) : 
  ∃ x : ℕ, x < 90 ∧ ∀ y : ℕ, y < 90 → (a ∣ y) ∧ (b ∣ y) → y ≤ x → x = 72 :=
sorry

end gcm_less_than_90_l15_15656


namespace Anton_thought_number_is_729_l15_15045

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l15_15045


namespace factor_between_l15_15449

theorem factor_between (n a b : ℕ) (h1 : 10 < n) 
(h2 : n = a * a + b) 
(h3 : a ∣ n) 
(h4 : b ∣ n) 
(h5 : a ≠ b) 
(h6 : 1 < a) 
(h7 : 1 < b) : 
    ∃ m : ℕ, b = m * a ∧ 1 < m ∧ a < a + m ∧ a + m < b  :=
by
  -- proof to be filled in
  sorry

end factor_between_l15_15449


namespace domain_of_function_l15_15912

theorem domain_of_function :
  {x : ℝ | (x + 1 ≥ 0) ∧ (2 - x ≠ 0)} = {x : ℝ | -1 ≤ x ∧ x ≠ 2} :=
by {
  sorry
}

end domain_of_function_l15_15912


namespace findLineEquation_l15_15452

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to represent the hyperbola condition
def isOnHyperbola (pt : Point) : Prop :=
  pt.x ^ 2 - 4 * pt.y ^ 2 = 4

-- Define midpoint condition for points A and B
def isMidpoint (P A B : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- Define points
def P : Point := ⟨8, 1⟩
def A : Point := sorry
def B : Point := sorry

-- Statement to prove
theorem findLineEquation :
  isOnHyperbola A ∧ isOnHyperbola B ∧ isMidpoint P A B →
  ∃ m b, (∀ pt : Point, pt.y = m * pt.x + b ↔ pt.x = 8 ∧ pt.y = 1) ∧ (m = 2) ∧ (b = -15) :=
by
  sorry

end findLineEquation_l15_15452


namespace square_side_length_l15_15414

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l15_15414


namespace compute_binomial_10_3_eq_120_l15_15708

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15708


namespace new_quadratic_equation_has_square_roots_l15_15518

theorem new_quadratic_equation_has_square_roots (p q : ℝ) (x : ℝ) :
  (x^2 + px + q = 0 → ∃ x1 x2 : ℝ, x^2 - (p^2 - 2 * q) * x + q^2 = 0 ∧ (x1^2 = x ∨ x2^2 = x)) :=
by sorry

end new_quadratic_equation_has_square_roots_l15_15518


namespace anton_thought_number_l15_15048

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l15_15048


namespace anton_thought_of_729_l15_15044

-- Definitions from the problem conditions
def guessed_numbers : List Nat := [109, 704, 124]

def matches_in_one_place (secret guess : Nat) : Prop :=
  let s := secret.digits 10
  let g := guess.digits 10
  if s.length = g.length then
    (s.zip g).count (λ (si, gi) => si = gi) = 1
  else
    False

noncomputable def anton_thought_number := 729

-- Lean statement to confirm that the number Anton thought of is 729
theorem anton_thought_of_729 : (∀ guess ∈ guessed_numbers, matches_in_one_place anton_thought_number guess) :=
  sorry

end anton_thought_of_729_l15_15044


namespace abc_system_proof_l15_15790

theorem abc_system_proof (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 + a = b^2) (h5 : b^2 + b = c^2) (h6 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 :=
by
  sorry

end abc_system_proof_l15_15790


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15220

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_7 :
  ∃ x : ℤ, x < 100 ∧ x % 7 = 4 ∧ (∀ y : ℤ, y < 100 ∧ y % 7 = 4 → y ≤ x) :=
begin
  use 95,
  split,
  { -- Proof that 95 < 100
    exact dec_trivial
  },
  split,
  { -- Proof that 95 % 7 = 4
    exact dec_trivial
  },
  { -- Proof that 95 is the largest such integer
    intros y hy,
    have h : 7 * (y / 7) + 4 ≤ 95, 
    { linarith [hy] },
    exact h
  }
end

end largest_integer_less_than_100_with_remainder_4_when_divided_by_7_l15_15220


namespace ice_cream_melting_l15_15674

theorem ice_cream_melting :
  ∀ (r1 r2 : ℝ) (h : ℝ),
    r1 = 3 ∧ r2 = 10 →
    4 / 3 * π * r1^3 = π * r2^2 * h →
    h = 9 / 25 :=
by intros r1 r2 h hcond voldist
   sorry

end ice_cream_melting_l15_15674


namespace expected_value_of_unfair_die_l15_15891

-- Define the probabilities for each face of the die.
def prob_face (n : ℕ) : ℚ :=
  if n = 8 then 5/14 else 1/14

-- Define the expected value of a roll of this die.
def expected_value : ℚ :=
  (1 / 14) * 1 + (1 / 14) * 2 + (1 / 14) * 3 + (1 / 14) * 4 + (1 / 14) * 5 + (1 / 14) * 6 + (1 / 14) * 7 + (5 / 14) * 8

-- The statement to prove: the expected value of a roll of this die is 4.857.
theorem expected_value_of_unfair_die : expected_value = 4.857 := by
  sorry

end expected_value_of_unfair_die_l15_15891


namespace square_side_length_l15_15407

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l15_15407


namespace sum_of_cubes_mod_7_l15_15743

theorem sum_of_cubes_mod_7 :
  (∑ k in Finset.range 150, (k + 1) ^ 3) % 7 = 1 := 
sorry

end sum_of_cubes_mod_7_l15_15743


namespace isosceles_triangle_l15_15288

noncomputable def triangle_is_isosceles (A B C a b c : ℝ) (h_triangle : a = 2 * b * Real.cos C) : Prop :=
  ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)

theorem isosceles_triangle
  (A B C a b c : ℝ)
  (h_sides : a = 2 * b * Real.cos C)
  (h_triangle : ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)) :
  B = C :=
sorry

end isosceles_triangle_l15_15288


namespace simplify_expression_l15_15436

theorem simplify_expression :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) *
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 :=
by 
  sorry

end simplify_expression_l15_15436


namespace katya_sequences_l15_15292

def is_valid_digit_replacement (original : ℕ) (left_neigh : Option ℕ) (right_neigh : Option ℕ) : ℕ :=
  (if left_neigh.isSome ∧ left_neigh.get < original then 1 else 0) +
  (if right_neigh.isSome ∧ right_neigh.get < original then 1 else 0)

def valid_transformation_sequence (seq : List ℕ) : Bool :=
  seq.length = 10 ∧
  ∀ i, i < 10 -> 
    let left_neigh := if i > 0 then some (seq.get ⟨i - 1, by linarith⟩) else none;
    let right_neigh := if i < 9 then some (seq.get ⟨i + 1, by linarith⟩) else none;
    seq.get ⟨i, by linarith⟩ = is_valid_digit_replacement(i, left_neigh, right_neigh)

theorem katya_sequences :
  valid_transformation_sequence [1, 1, 0, 1, 1, 1, 1, 1, 1, 1] ∧
  ¬valid_transformation_sequence [1, 2, 0, 1, 2, 0, 1, 0, 2, 0] ∧
  valid_transformation_sequence [1, 0, 2, 1, 0, 2, 1, 0, 2, 0] ∧
  valid_transformation_sequence [0, 1, 1, 2, 1, 0, 2, 0, 1, 1] := 
by {
  sorry
}

end katya_sequences_l15_15292


namespace original_price_of_painting_l15_15033

theorem original_price_of_painting (purchase_price : ℝ) (fraction : ℝ) (original_price : ℝ) :
  purchase_price = 200 → fraction = 1/4 → purchase_price = original_price * fraction → original_price = 800 :=
by
  intros h1 h2 h3
  -- proof steps here
  sorry

end original_price_of_painting_l15_15033


namespace least_positive_three_digit_multiple_of_13_is_104_l15_15828

theorem least_positive_three_digit_multiple_of_13_is_104 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n = 104 :=
by
  existsi 104
  split
  · show 100 ≤ 104
    exact le_refl 104
  split
  · show 104 < 1000
    exact dec_trivial
  split
  · show 104 % 13 = 0
    exact dec_trivial
  · show 104 = 104
    exact rfl

end least_positive_three_digit_multiple_of_13_is_104_l15_15828


namespace maximize_net_income_l15_15678

-- Define the conditions of the problem
def bicycles := 50
def management_cost := 115

def rental_income (x : ℕ) : ℕ :=
if x ≤ 6 then bicycles * x
else (bicycles - 3 * (x - 6)) * x

def net_income (x : ℕ) : ℤ :=
rental_income x - management_cost

-- Define the domain of the function
def domain (x : ℕ) : Prop := 3 ≤ x ∧ x ≤ 20

-- Define the piecewise function for y = f(x)
def f (x : ℕ) : ℤ :=
if 3 ≤ x ∧ x ≤ 6 then 50 * x - 115
else if 6 < x ∧ x ≤ 20 then -3 * x * x + 68 * x - 115
else 0  -- Out of domain

-- The theorem that we need to prove
theorem maximize_net_income :
  (∀ x, domain x → net_income x = f x) ∧
  (∃ x, domain x ∧ (∀ y, domain y → net_income y ≤ net_income x) ∧ x = 11) :=
by
  sorry

end maximize_net_income_l15_15678


namespace square_side_length_l15_15408

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l15_15408


namespace tank_filling_time_l15_15524

theorem tank_filling_time
  (T : ℕ) (Rₐ R_b R_c : ℕ) (C : ℕ)
  (hRₐ : Rₐ = 40) (hR_b : R_b = 30) (hR_c : R_c = 20) (hC : C = 950)
  (h_cycle : T = 1 + 1 + 1) : 
  T * (C / (Rₐ + R_b - R_c)) - 1 = 56 :=
by
  sorry

end tank_filling_time_l15_15524


namespace greatest_integer_gcd_30_is_125_l15_15349

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l15_15349


namespace proof_probability_at_least_one_makes_both_shots_l15_15587

-- Define the shooting percentages for Player A and Player B
def shooting_percentage_A : ℝ := 0.4
def shooting_percentage_B : ℝ := 0.5

-- Define the probability that Player A makes both shots
def prob_A_makes_both_shots : ℝ := shooting_percentage_A * shooting_percentage_A

-- Define the probability that Player B makes both shots
def prob_B_makes_both_shots : ℝ := shooting_percentage_B * shooting_percentage_B

-- Define the probability that neither makes both shots
def prob_neither_makes_both_shots : ℝ := (1 - prob_A_makes_both_shots) * (1 - prob_B_makes_both_shots)

-- Define the probability that at least one of them makes both shots
def prob_at_least_one_makes_both_shots : ℝ := 1 - prob_neither_makes_both_shots

-- Prove that the probability that at least one of them makes both shots is 0.37
theorem proof_probability_at_least_one_makes_both_shots :
  prob_at_least_one_makes_both_shots = 0.37 :=
sorry

end proof_probability_at_least_one_makes_both_shots_l15_15587


namespace sym_diff_A_B_l15_15578

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- Definition of the symmetric difference
def sym_diff (A B : Set ℕ) : Set ℕ := {x | (x ∈ A ∨ x ∈ B) ∧ x ∉ (A ∩ B)}

theorem sym_diff_A_B : sym_diff A B = {0, 3} := 
by 
  sorry

end sym_diff_A_B_l15_15578


namespace largest_integer_remainder_condition_l15_15246

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l15_15246


namespace apples_fraction_of_pears_l15_15177

variables (A O P : ℕ)

-- Conditions
def oranges_condition := O = 3 * A
def pears_condition := P = 4 * O

-- Statement we need to prove
theorem apples_fraction_of_pears (A O P : ℕ) (h1 : O = 3 * A) (h2 : P = 4 * O) : (A : ℚ) / P = 1 / 12 :=
by
  sorry

end apples_fraction_of_pears_l15_15177


namespace galya_overtakes_sasha_l15_15304

variable {L : ℝ} -- Length of the track
variable (Sasha_uphill_speed : ℝ := 8)
variable (Sasha_downhill_speed : ℝ := 24)
variable (Galya_uphill_speed : ℝ := 16)
variable (Galya_downhill_speed : ℝ := 18)

noncomputable def average_speed (uphill_speed: ℝ) (downhill_speed: ℝ) : ℝ :=
  1 / ((1 / (4 * uphill_speed)) + (3 / (4 * downhill_speed)))

noncomputable def time_for_one_lap (L: ℝ) (speed: ℝ) : ℝ :=
  L / speed

theorem galya_overtakes_sasha 
  (L_pos : 0 < L) :
  let v_Sasha := average_speed Sasha_uphill_speed Sasha_downhill_speed
  let v_Galya := average_speed Galya_uphill_speed Galya_downhill_speed
  let t_Sasha := time_for_one_lap L v_Sasha
  let t_Galya := time_for_one_lap L v_Galya
  (L * 11 / v_Galya) < (L * 10 / v_Sasha) :=
by
  sorry

end galya_overtakes_sasha_l15_15304


namespace abc_equivalence_l15_15116

theorem abc_equivalence (n : ℕ) (k : ℤ) (a b c : ℤ)
  (hn : 0 < n) (hk : k % 2 = 1)
  (h : a^n + k * b = b^n + k * c ∧ b^n + k * c = c^n + k * a) :
  a = b ∧ b = c := 
sorry

end abc_equivalence_l15_15116


namespace lychees_remaining_l15_15783

theorem lychees_remaining 
  (initial_lychees : ℕ) 
  (sold_fraction taken_fraction : ℕ → ℕ) 
  (initial_lychees = 500) 
  (sold_fraction = λ x, x / 2) 
  (taken_fraction = λ x, x * 2 / 5) 
  : initial_lychees - sold_fraction initial_lychees - taken_fraction (initial_lychees - sold_fraction initial_lychees) = 100 := 
by 
  sorry

end lychees_remaining_l15_15783


namespace compute_expression_l15_15195

theorem compute_expression :
    (3 + 5)^2 + (3^2 + 5^2 + 3 * 5) = 113 := 
by sorry

end compute_expression_l15_15195


namespace comb_10_3_eq_120_l15_15734

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15734


namespace time_between_rings_is_288_minutes_l15_15096

def intervals_between_rings (total_rings : ℕ) (total_minutes : ℕ) : ℕ := 
  let intervals := total_rings - 1
  total_minutes / intervals

theorem time_between_rings_is_288_minutes (total_minutes_in_day total_rings : ℕ) 
  (h1 : total_minutes_in_day = 1440) (h2 : total_rings = 6) : 
  intervals_between_rings total_rings total_minutes_in_day = 288 := 
by 
  sorry

end time_between_rings_is_288_minutes_l15_15096


namespace sufficient_but_not_necessary_condition_l15_15760

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x = 0 → (x^2 - 2 * x = 0)) ∧ (∃ y : ℝ, y ≠ 0 ∧ y ^ 2 - 2 * y = 0) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l15_15760


namespace min_value_frac_sqrt_l15_15444

theorem min_value_frac_sqrt (x : ℝ) (h : x > 1) : 
  (x + 10) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 11 :=
sorry

end min_value_frac_sqrt_l15_15444


namespace solve_puzzle_l15_15756

theorem solve_puzzle
  (EH OY AY OH : ℕ)
  (h1 : EH = 4 * OY)
  (h2 : AY = 4 * OH) :
  EH + OY + AY + OH = 150 :=
sorry

end solve_puzzle_l15_15756


namespace least_k_for_divisibility_l15_15097

theorem least_k_for_divisibility (k : ℕ) : (k ^ 4) % 1260 = 0 ↔ k ≥ 210 :=
sorry

end least_k_for_divisibility_l15_15097


namespace phase_shift_of_cosine_l15_15258

theorem phase_shift_of_cosine (a b c : ℝ) (h : c = -π / 4 ∧ b = 3) :
  (-c / b) = π / 12 :=
by
  sorry

end phase_shift_of_cosine_l15_15258


namespace equation_has_one_solution_l15_15199

theorem equation_has_one_solution : ∀ x : ℝ, x - 6 / (x - 2) = 4 - 6 / (x - 2) ↔ x = 4 :=
by {
  -- proof goes here
  sorry
}

end equation_has_one_solution_l15_15199


namespace problem_solution_l15_15593

noncomputable def rectangular_equation_of_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * y = 0

noncomputable def general_equation_of_l (x y : ℝ) : Prop :=
  x - sqrt 3 * y + sqrt 3 = 0

noncomputable def max_area_of_triangle_PAB (A B P : ℝ × ℝ) : ℝ :=
  (4 * sqrt 13 + sqrt 39) / 4

theorem problem_solution {θ t : ℝ} (x y : ℝ) :
  ( rectangular_equation_of_C x y ) ∧
  ( general_equation_of_l x y ) ∧
  ( ∃ A B P, A ≠ B ∧ A ≠ P ∧ B ≠ P ∧ max_area_of_triangle_PAB A B P = (4 * sqrt 13 + sqrt 39) / 4 ) :=
begin
  sorry
end

end problem_solution_l15_15593


namespace cups_per_serving_l15_15535

theorem cups_per_serving (total_cups servings : ℝ) (h1 : total_cups = 36) (h2 : servings = 18.0) :
  total_cups / servings = 2 :=
by 
  sorry

end cups_per_serving_l15_15535


namespace solve_sin_equation_l15_15488

theorem solve_sin_equation :
  ∀ x : ℝ, 0 < x ∧ x < 90 → 
  (sin 9 * sin 21 * sin (102 + x) = sin 30 * sin 42 * sin x) → 
  x = 9 :=
by
  intros x h_cond h_eq
  sorry

end solve_sin_equation_l15_15488


namespace tailor_time_l15_15422

theorem tailor_time (x : ℝ) 
  (t_shirt : ℝ := x) 
  (t_pants : ℝ := 2 * x) 
  (t_jacket : ℝ := 3 * x) 
  (h_capacity : 2 * t_shirt + 3 * t_pants + 4 * t_jacket = 10) : 
  14 * t_shirt + 10 * t_pants + 2 * t_jacket = 20 :=
by
  sorry

end tailor_time_l15_15422


namespace solve_for_n_l15_15796

theorem solve_for_n (n : ℕ) : 
  9^n * 9^n * 9^(2*n) = 81^4 → n = 2 :=
by
  sorry

end solve_for_n_l15_15796


namespace comb_10_3_eq_120_l15_15728

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15728


namespace homework_duration_equation_l15_15651

-- Given conditions
def initial_duration : ℝ := 120
def final_duration : ℝ := 60
variable (x : ℝ)

-- The goal is to prove that the appropriate equation holds
theorem homework_duration_equation : initial_duration * (1 - x)^2 = final_duration := 
sorry

end homework_duration_equation_l15_15651


namespace value_of_x_l15_15748

variable (x y z a b c : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h1 : x * y / (x + y) = a)
variable (h2 : x * z / (x + z) = b)
variable (h3 : y * z / (y + z) = c)

theorem value_of_x : x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end value_of_x_l15_15748


namespace length_of_LM_l15_15644

-- Definitions of the conditions
variable (P Q R L M : Type)
variable (b : Real) (PR_area : Real) (PR_base : Real)
variable (PR_base_eq : PR_base = 15)
variable (crease_parallel : Parallel L M)
variable (projected_area_fraction : Real)
variable (projected_area_fraction_eq : projected_area_fraction = 0.25 * PR_area)

-- Theorem statement to prove the length of LM
theorem length_of_LM : ∀ (LM_length : Real), (LM_length = 7.5) :=
sorry

end length_of_LM_l15_15644


namespace sin2θ_over_1pluscos2θ_eq_sqrt3_l15_15579

theorem sin2θ_over_1pluscos2θ_eq_sqrt3 {θ : ℝ} (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 :=
sorry

end sin2θ_over_1pluscos2θ_eq_sqrt3_l15_15579


namespace polygon_perimeter_l15_15537

theorem polygon_perimeter (side_length : ℝ) (ext_angle_deg : ℝ) (n : ℕ) (h1 : side_length = 8) 
  (h2 : ext_angle_deg = 90) (h3 : ext_angle_deg = 360 / n) : 
  4 * side_length = 32 := 
  by 
    sorry

end polygon_perimeter_l15_15537


namespace sum_mod_17_l15_15193

theorem sum_mod_17 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 17 = 2 :=
by
  sorry

end sum_mod_17_l15_15193


namespace anton_thought_number_l15_15047

theorem anton_thought_number (n : ℕ) : 
  (∃ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ,
    d1 = 1 ∧ d2 = 0 ∧ d3 = 9 ∧ 
    d4 = 7 ∧ d5 = 0 ∧ d6 = 4 ∧ 
    d7 = 1 ∧ d8 = 2 ∧ d9 = 4 ∧ 
    (n = d1*100 + d2*10 + d3 ∨ n = d4*100 + d5*10 + d6 ∨ n = d7*100 + d8*10 + d9) ∧
    (n ≥ 100 ∧ n < 1000) ∧
    (∃ h t u : ℕ, n = h * 100 + t * 10 + u ∧ 
      ((h = 1 ∧ t ≠ 0 ∧ u ≠ 9) ∨ (h ≠ 1 ∧ t = 0 ∧ u ≠ 4) ∨ (h ≠ 7 ∧ t ≠ 1 ∧ u = 4))) → 
  n = 729 :=
by sorry

end anton_thought_number_l15_15047


namespace line_slope_l15_15189

theorem line_slope (x1 y1 x2 y2 : ℝ) (h1 : x1 = 0) (h2 : y1 = 100) (h3 : x2 = 50) (h4 : y2 = 300) :
  (y2 - y1) / (x2 - x1) = 4 :=
by sorry

end line_slope_l15_15189


namespace average_cd_e_l15_15498

theorem average_cd_e (c d e : ℝ) (h : (4 + 6 + 9 + c + d + e) / 6 = 20) : 
    (c + d + e) / 3 = 101 / 3 :=
by
  sorry

end average_cd_e_l15_15498


namespace dave_earnings_l15_15196

def total_games : Nat := 10
def non_working_games : Nat := 2
def price_per_game : Nat := 4
def working_games : Nat := total_games - non_working_games
def money_earned : Nat := working_games * price_per_game

theorem dave_earnings : money_earned = 32 := by
  sorry

end dave_earnings_l15_15196


namespace find_triples_l15_15198

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≤ y) (hyz : y ≤ z) 
  (h_eq : x * y + y * z + z * x - x * y * z = 2) : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 2 ∧ y = 3 ∧ z = 4) := 
by 
  sorry

end find_triples_l15_15198


namespace rectangle_horizontal_length_l15_15149

theorem rectangle_horizontal_length (s v : ℕ) (h : ℕ) 
  (hs : s = 80) (hv : v = 100) 
  (eq_perimeters : 4 * s = 2 * (v + h)) : h = 60 :=
by
  sorry

end rectangle_horizontal_length_l15_15149


namespace greatest_integer_gcd_l15_15338

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l15_15338


namespace combination_10_3_l15_15699

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15699


namespace total_cost_correct_l15_15862

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end total_cost_correct_l15_15862


namespace matrix_power_2023_correct_l15_15555

noncomputable def matrix_power_2023 : Matrix (Fin 2) (Fin 2) ℤ :=
  let A := !![1, 0; 2, 1]  -- Define the matrix
  A^2023

theorem matrix_power_2023_correct :
  matrix_power_2023 = !![1, 0; 4046, 1] := by
  sorry

end matrix_power_2023_correct_l15_15555


namespace gcd_lcm_product_eq_abc_l15_15272

theorem gcd_lcm_product_eq_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  let D := Nat.gcd (Nat.gcd a b) c
  let m := Nat.lcm (Nat.lcm a b) c
  D * m = a * b * c :=
by
  sorry

end gcd_lcm_product_eq_abc_l15_15272


namespace geom_seq_min_value_proof_l15_15605

noncomputable def geom_seq_min_value : ℝ := 3 / 2

theorem geom_seq_min_value_proof (a : ℕ → ℝ) (a1 : ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  a 2017 = a 2016 + 2 * a 2015 →
  a m * a n = 16 * a1^2 →
  (4 / m + 1 / n) = geom_seq_min_value :=
by {
  sorry
}

end geom_seq_min_value_proof_l15_15605


namespace sum_cubes_mod_7_l15_15742

theorem sum_cubes_mod_7 :
  (∑ i in Finset.range 151, i ^ 3) % 7 = 0 := by
  sorry

end sum_cubes_mod_7_l15_15742


namespace faulty_clock_correct_display_fraction_l15_15373

-- Defining the faulty display clock and the fraction calculation proof
theorem faulty_clock_correct_display_fraction : 
  let hours := 12
  let correct_hours := 10
  let hours_fraction := (correct_hours : ℚ) / hours 
  let minutes_per_hour := 60
  let incorrect_minutes := 16
  let correct_minutes := minutes_per_hour - incorrect_minutes
  let minutes_fraction := (correct_minutes : ℚ) / minutes_per_hour
  hours_fraction * minutes_fraction = 11 / 18 :=
by
  let hours := 12
  let correct_hours := 10
  let correct_hours_fraction := (correct_hours : ℚ) / hours
  let minutes_per_hour := 60
  let incorrect_minutes := 16
  let correct_minutes := minutes_per_hour - incorrect_minutes
  let correct_minutes_fraction := (correct_minutes : ℚ) / minutes_per_hour
  calc
    (correct_hours_fraction * correct_minutes_fraction) 
      = (10 / 12) * (44 / 60) : by sorry
    ... = 11 / 18 : by sorry

end faulty_clock_correct_display_fraction_l15_15373


namespace positive_numbers_with_cube_root_lt_10_l15_15273

def cube_root_lt_10 (n : ℕ) : Prop :=
  (↑n : ℝ)^(1 / 3 : ℝ) < 10

theorem positive_numbers_with_cube_root_lt_10 : 
  ∃ (count : ℕ), (count = 999) ∧ ∀ n : ℕ, (1 ≤ n ∧ n ≤ 999) → cube_root_lt_10 n :=
by
  sorry

end positive_numbers_with_cube_root_lt_10_l15_15273


namespace greatest_integer_gcd_l15_15339

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l15_15339


namespace mowing_lawn_time_l15_15303

theorem mowing_lawn_time (mary_time tom_time tom_solo_work : ℝ) 
  (mary_rate tom_rate : ℝ)
  (combined_rate remaining_lawn total_time : ℝ) :
  mary_time = 3 → 
  tom_time = 6 → 
  tom_solo_work = 3 → 
  mary_rate = 1 / mary_time → 
  tom_rate = 1 / tom_time → 
  combined_rate = mary_rate + tom_rate →
  remaining_lawn = 1 - (tom_solo_work * tom_rate) →
  total_time = tom_solo_work + (remaining_lawn / combined_rate) →
  total_time = 4 :=
by sorry

end mowing_lawn_time_l15_15303


namespace diamond_property_C_l15_15431

-- Define the binary operation diamond
def diamond (a b : ℕ) : ℕ := a ^ (2 * b)

theorem diamond_property_C (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) : 
  (diamond a b) ^ n = diamond a (b * n) :=
by
  sorry

end diamond_property_C_l15_15431


namespace binomial_coefficient_10_3_l15_15724

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l15_15724


namespace least_three_digit_multiple_of_13_l15_15836

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, n ≥ 100 ∧ n % 13 = 0 ∧ ∀ m : ℕ, m ≥ 100 → m % 13 = 0 → n ≤ m :=
begin
  use 104,
  split,
  { exact nat.le_of_eq rfl },
  split,
  { exact nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero rfl) },
  { intros m hm hmod,
    have h8 : 8 * 13 = 104 := rfl,
    rw ←h8,
    exact nat.le_mul_of_pos_left (by norm_num) },
end

end least_three_digit_multiple_of_13_l15_15836


namespace conic_section_is_hyperbola_l15_15435

theorem conic_section_is_hyperbola (x y : ℝ) :
  (x - 3)^2 = (3 * y + 4)^2 - 75 → 
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 :=
sorry

end conic_section_is_hyperbola_l15_15435


namespace Karls_Total_Travel_Distance_l15_15475

theorem Karls_Total_Travel_Distance :
  let consumption_rate := 35
  let full_tank_gallons := 14
  let initial_miles := 350
  let added_gallons := 8
  let remaining_gallons := 7
  let net_gallons_consumed := (full_tank_gallons + added_gallons - remaining_gallons)
  let total_distance := net_gallons_consumed * consumption_rate
  total_distance = 525 := 
by 
  sorry

end Karls_Total_Travel_Distance_l15_15475


namespace greatest_int_with_gcd_five_l15_15333

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l15_15333


namespace total_tickets_sold_l15_15802

theorem total_tickets_sold 
  (A D : ℕ) 
  (cost_adv cost_door : ℝ) 
  (revenue : ℝ)
  (door_tickets_sold total_tickets : ℕ) 
  (h1 : cost_adv = 14.50) 
  (h2 : cost_door = 22.00)
  (h3 : revenue = 16640) 
  (h4 : door_tickets_sold = 672) : 
  (total_tickets = 800) :=
by
  sorry

end total_tickets_sold_l15_15802


namespace largest_integer_remainder_condition_l15_15245

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l15_15245


namespace anton_thought_number_l15_15042

def matches_exactly_one_digit (a b : ℕ) : Prop :=
  let digits_a := [a / 100 % 10, a / 10 % 10, a % 10]
  let digits_b := [b / 100 % 10, b / 10 % 10, b % 10]
  (digits_a.zip digits_b).count (λ (pair : ℕ × ℕ) => pair.1 = pair.2) = 1

theorem anton_thought_number {n : ℕ} :
  n = 729 →
  matches_exactly_one_digit n 109 →
  matches_exactly_one_digit n 704 →
  matches_exactly_one_digit n 124 :=
by
  intros h1 h2 h3
  rw h1 at *
  exact ⟨h2, h3⟩
  sorry

end anton_thought_number_l15_15042


namespace triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l15_15487

/-- Points \(P, Q, R, S\) are distinct, collinear, and ordered on a line with line segment lengths \( a, b, c \)
    such that \(a = PQ\), \(b = PR\), \(c = PS\). After rotating \(PQ\) and \(RS\) to make \( P \) and \( S \) coincide
    and form a triangle with a positive area, we must show:
    \(I. a < \frac{c}{3}\) must be satisfied in accordance to the triangle inequality revelations -/
theorem triangle_inequality_necessary_conditions (a b c : ℝ)
  (h_abc1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : b > c - b ∧ c > a ∧ c > b - a) :
  a < c / 3 :=
sorry

theorem triangle_inequality_sufficient_conditions (a b c : ℝ)
  (h_abc2 : b ≥ c / 3 ∧ a < c ∧ 2 * b ≤ c) :
  ¬ b < c / 3 :=
sorry

end triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l15_15487


namespace largest_int_less_than_100_by_7_l15_15232

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l15_15232


namespace max_sections_with_five_lines_l15_15106

def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  n * (n + 1) / 2 + 1

theorem max_sections_with_five_lines : sections 5 = 16 := by
  sorry

end max_sections_with_five_lines_l15_15106


namespace box_height_correct_l15_15529

noncomputable def box_height : ℕ :=
  8

theorem box_height_correct (box_width box_length block_height block_width block_length : ℕ) (num_blocks : ℕ) :
  box_width = 10 ∧
  box_length = 12 ∧
  block_height = 3 ∧
  block_width = 2 ∧
  block_length = 4 ∧
  num_blocks = 40 →
  (num_blocks * block_height * block_width * block_length) /
  (box_width * box_length) = box_height :=
  by
  sorry

end box_height_correct_l15_15529


namespace apples_total_l15_15124

theorem apples_total (lexie_apples : ℕ) (tom_apples : ℕ) (h1 : lexie_apples = 12) (h2 : tom_apples = 2 * lexie_apples) : lexie_apples + tom_apples = 36 :=
by
  sorry

end apples_total_l15_15124


namespace polynomial_value_at_3_l15_15480

theorem polynomial_value_at_3 :
  ∃ (P : ℕ → ℚ), 
    (∀ (x : ℕ), P x = b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + b_4 * x^4 + b_5 * x^5 + b_6 * x^6) ∧ 
    (∀ (i : ℕ), i ≤ 6 → 0 ≤ b_i ∧ b_i < 5) ∧ 
    P (Nat.sqrt 5) = 35 + 26 * Nat.sqrt 5 -> 
    P 3 = 437 := 
by
  simp
  sorry

end polynomial_value_at_3_l15_15480


namespace binomial_coefficient_10_3_l15_15714

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l15_15714


namespace annual_pension_l15_15027

theorem annual_pension (c d r s x k : ℝ) (hc : c ≠ 0) (hd : d ≠ c)
  (h1 : k * (x + c) ^ (3 / 2) = k * x ^ (3 / 2) + r)
  (h2 : k * (x + d) ^ (3 / 2) = k * x ^ (3 / 2) + s) :
  k * x ^ (3 / 2) = 4 * r^2 / (9 * c^2) :=
by
  sorry

end annual_pension_l15_15027


namespace group_friends_opponents_l15_15879

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l15_15879


namespace complement_of_A_with_respect_to_U_l15_15929

open Set

-- Definitions
def U : Set ℤ := {-1, 1, 3}
def A : Set ℤ := {-1}

-- Theorem statement
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {1, 3} :=
by
  sorry

end complement_of_A_with_respect_to_U_l15_15929


namespace compute_binomial_10_3_eq_120_l15_15705

-- Define the factorial function to be used in the binomial coefficient
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define binomial coefficient using the factorial function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement we want to prove
theorem compute_binomial_10_3_eq_120 : binomial 10 3 = 120 := 
by
  -- Here we skip the proof with sorry
  sorry

end compute_binomial_10_3_eq_120_l15_15705


namespace initial_quantity_of_milk_in_A_l15_15680

theorem initial_quantity_of_milk_in_A (A : ℝ) 
  (h1: ∃ C B: ℝ, B = 0.375 * A ∧ C = 0.625 * A) 
  (h2: ∃ M: ℝ, M = 0.375 * A + 154 ∧ M = 0.625 * A - 154) 
  : A = 1232 :=
by
  -- you can use sorry to skip the proof
  sorry

end initial_quantity_of_milk_in_A_l15_15680


namespace side_length_of_square_l15_15401

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l15_15401


namespace find_largest_integer_l15_15218

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l15_15218


namespace largest_integer_less_than_100_leaving_remainder_4_l15_15254

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l15_15254


namespace range_of_a_l15_15662

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -(x^2 + 2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2 * x) →
  f (2 - a^2) > f a ↔ -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l15_15662


namespace janet_needs_9_dog_collars_l15_15946

variable (D : ℕ)

theorem janet_needs_9_dog_collars (h1 : ∀ d : ℕ, d = 18)
  (h2 : ∀ c : ℕ, c = 10)
  (h3 : (18 * D) + (3 * 10) = 192) :
  D = 9 :=
by
  sorry

end janet_needs_9_dog_collars_l15_15946


namespace fish_in_pond_l15_15757

noncomputable def number_of_fish (marked_first: ℕ) (marked_second: ℕ) (catch_first: ℕ) (catch_second: ℕ) : ℕ :=
  (marked_first * catch_second) / marked_second

theorem fish_in_pond (h1 : marked_first = 30) (h2 : marked_second = 2) (h3 : catch_first = 30) (h4 : catch_second = 40) :
  number_of_fish marked_first marked_second catch_first catch_second = 600 :=
by
  rw [h1, h2, h3, h4]
  sorry

end fish_in_pond_l15_15757


namespace comb_10_3_eq_120_l15_15730

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15730


namespace volume_of_regular_triangular_pyramid_l15_15440

noncomputable def pyramid_volume (R φ : ℝ) : ℝ :=
  (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)

theorem volume_of_regular_triangular_pyramid (R φ : ℝ) 
  (cond1 : R > 0)
  (cond2: 0 < φ ∧ φ < π) :
  ∃ V, V = pyramid_volume R φ := by
    use (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)
    sorry

end volume_of_regular_triangular_pyramid_l15_15440


namespace number_of_coins_l15_15842

-- Define the conditions
def equal_number_of_coins (x : ℝ) :=
  ∃ n : ℝ, n = x

-- Define the total value condition
def total_value (x : ℝ) :=
  x + 0.50 * x + 0.25 * x = 70

-- The theorem to be proved
theorem number_of_coins (x : ℝ) (h1 : equal_number_of_coins x) (h2 : total_value x) : x = 40 :=
by sorry

end number_of_coins_l15_15842


namespace relationship_P_Q_l15_15955

theorem relationship_P_Q (x : ℝ) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.exp x + Real.exp (-x)) 
  (hQ : Q = (Real.sin x + Real.cos x) ^ 2) : 
  P ≥ Q := 
sorry

end relationship_P_Q_l15_15955


namespace distance_A_B_l15_15319

theorem distance_A_B 
  (perimeter_small_square : ℝ)
  (area_large_square : ℝ)
  (h1 : perimeter_small_square = 8)
  (h2 : area_large_square = 64) :
  let side_small_square := perimeter_small_square / 4
  let side_large_square := Real.sqrt area_large_square
  let horizontal_distance := side_small_square + side_large_square
  let vertical_distance := side_large_square - side_small_square
  let distance_AB := Real.sqrt (horizontal_distance^2 + vertical_distance^2)
  distance_AB = 11.7 :=
  by sorry

end distance_A_B_l15_15319


namespace cole_drive_time_correct_l15_15554

noncomputable def cole_drive_time : ℕ :=
  let distance_to_work := 45 -- derived from the given problem   
  let speed_to_work := 30
  let time_to_work := distance_to_work / speed_to_work -- in hours
  (time_to_work * 60 : ℕ) -- converting hours to minutes

theorem cole_drive_time_correct
  (speed_to_work speed_return: ℕ)
  (total_time: ℕ)
  (H1: speed_to_work = 30)
  (H2: speed_return = 90)
  (H3: total_time = 2):
  cole_drive_time = 90 := by
  -- Proof omitted
  sorry

end cole_drive_time_correct_l15_15554


namespace find_x_values_l15_15438

theorem find_x_values (x : ℝ) (h : x ≠ 5) : x + 36 / (x - 5) = -12 ↔ x = -8 ∨ x = 3 :=
by sorry

end find_x_values_l15_15438


namespace distinct_license_plates_l15_15531

theorem distinct_license_plates :
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  total = 122504000 :=
by
  -- Definitions from the conditions
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  -- Calculation
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  -- Assertion
  have h : total = 122504000 := sorry
  exact h

end distinct_license_plates_l15_15531


namespace greatest_common_divisor_is_40_l15_15821

def distance_to_boston : ℕ := 840
def distance_to_atlanta : ℕ := 440

theorem greatest_common_divisor_is_40 :
  Nat.gcd distance_to_boston distance_to_atlanta = 40 :=
by
  -- The theorem statement as described is correct
  -- Proof is omitted as per instructions
  sorry

end greatest_common_divisor_is_40_l15_15821


namespace coins_remainder_divide_by_nine_remainder_l15_15856

def smallest_n (n : ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_remainder (n : ℕ) (h : smallest_n n) : (∃ m : ℕ, n = 54) :=
  sorry

theorem divide_by_nine_remainder (n : ℕ) (h : smallest_n n) (h_smallest: coins_remainder n h) : n % 9 = 0 :=
  sorry

end coins_remainder_divide_by_nine_remainder_l15_15856


namespace pentagon_area_l15_15887

theorem pentagon_area (a b c d e : ℤ) (O : 31 * 25 = 775) (H : 12^2 + 5^2 = 13^2) 
  (rect_side_lengths : (a, b, c, d, e) = (13, 19, 20, 25, 31)) :
  775 - 1/2 * 12 * 5 = 745 := 
by
  sorry

end pentagon_area_l15_15887


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l15_15342

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l15_15342


namespace total_cost_pants_and_belt_l15_15505

theorem total_cost_pants_and_belt (P B : ℝ) 
  (hP : P = 34.0) 
  (hCondition : P = B - 2.93) : 
  P + B = 70.93 :=
by
  -- Placeholder for proof
  sorry

end total_cost_pants_and_belt_l15_15505


namespace no_equal_numbers_from_19_and_98_l15_15517

theorem no_equal_numbers_from_19_and_98 :
  ¬ (∃ s : ℕ, ∃ (a b : ℕ → ℕ), 
       (a 0 = 19) ∧ (b 0 = 98) ∧
       (∀ k, a (k + 1) = a k * a k ∨ a (k + 1) = a k + 1) ∧
       (∀ k, b (k + 1) = b k * b k ∨ b (k + 1) = b k + 1) ∧
       a s = b s) :=
sorry

end no_equal_numbers_from_19_and_98_l15_15517


namespace largest_integer_remainder_condition_l15_15248

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l15_15248


namespace sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l15_15515

theorem sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3 :
  let largestThreeDigitMultipleOf4 := 996
  let smallestFourDigitMultipleOf3 := 1002
  largestThreeDigitMultipleOf4 + smallestFourDigitMultipleOf3 = 1998 :=
by
  sorry

end sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l15_15515


namespace ice_cream_melting_l15_15675

theorem ice_cream_melting :
  ∀ (r1 r2 : ℝ) (h : ℝ),
    r1 = 3 ∧ r2 = 10 →
    4 / 3 * π * r1^3 = π * r2^2 * h →
    h = 9 / 25 :=
by intros r1 r2 h hcond voldist
   sorry

end ice_cream_melting_l15_15675


namespace moles_of_water_used_l15_15256

-- Define the balanced chemical equation's molar ratios
def balanced_reaction (Li3N_moles : ℕ) (H2O_moles : ℕ) (LiOH_moles : ℕ) (NH3_moles : ℕ) : Prop :=
  Li3N_moles = 1 ∧ H2O_moles = 3 ∧ LiOH_moles = 3 ∧ NH3_moles = 1

-- Given 1 mole of lithium nitride and 3 moles of lithium hydroxide produced, 
-- prove that 3 moles of water were used.
theorem moles_of_water_used (Li3N_moles : ℕ) (LiOH_moles : ℕ) (H2O_moles : ℕ) :
  Li3N_moles = 1 → LiOH_moles = 3 → H2O_moles = 3 :=
by
  intros h1 h2
  sorry

end moles_of_water_used_l15_15256


namespace find_k_check_divisibility_l15_15560

-- Define the polynomial f(x) as 2x^3 - 8x^2 + kx - 10
def f (x k : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + k * x - 10

-- Define the polynomial g(x) as 2x^3 - 8x^2 + 13x - 10 after finding k = 13
def g (x : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + 13 * x - 10

-- The first proof problem: Finding k
theorem find_k : (f 2 k = 0) → k = 13 := 
sorry

-- The second proof problem: Checking divisibility by 2x^2 - 1
theorem check_divisibility : ¬ (∃ h : ℝ → ℝ, g x = (2 * x^2 - 1) * h x) := 
sorry

end find_k_check_divisibility_l15_15560


namespace minimum_detectors_required_l15_15509

/-- There is a cube with each face divided into 4 identical square cells, making a total of 24 cells.
Oleg wants to mark 8 cells with invisible ink such that no two marked cells share a side.
Rustem wants to place detectors in the cells so that all marked cells can be identified. -/
def minimum_detectors_to_identify_all_marked_cells (total_cells: ℕ) (marked_cells: ℕ) 
  (cells_per_face: ℕ) (faces: ℕ) : ℕ :=
  if total_cells = faces * cells_per_face ∧ marked_cells = 8 then 16 else 0

theorem minimum_detectors_required :
  minimum_detectors_to_identify_all_marked_cells 24 8 4 6 = 16 :=
by
  sorry

end minimum_detectors_required_l15_15509


namespace ordering_of_powers_l15_15329

theorem ordering_of_powers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by sorry

end ordering_of_powers_l15_15329


namespace sum_of_squares_of_products_eq_factorial_l15_15899

open Nat

-- Definitions for sets and conditions
def validSets (n : ℕ) : List (List ℕ) :=
  List.filter (λ s, ∀ i ∈ s, ∀ j ∈ s, i ≠ j + 1 ∧ i ≠ j - 1) (List.powerset (List.range (n + 1)))

def productOfSet (s : List ℕ) : ℕ := s.foldr (*) 1

def sumOfSquaresOfProducts (n : ℕ) : ℕ :=
  (validSets n).foldr (λ s acc, acc + (productOfSet s)^2) 0

-- The theorem statement
theorem sum_of_squares_of_products_eq_factorial (n : ℕ) : sumOfSquaresOfProducts n = (nat.factorial (n + 1)) - 1 := by
  sorry

end sum_of_squares_of_products_eq_factorial_l15_15899


namespace range_a_l15_15920

theorem range_a (x a : ℝ) (h1 : x^2 - 8 * x - 33 > 0) (h2 : |x - 1| > a) (h3 : a > 0) :
  0 < a ∧ a ≤ 4 :=
by
  sorry

end range_a_l15_15920


namespace smallest_n_for_congruence_l15_15434

theorem smallest_n_for_congruence : ∃ n : ℕ, 0 < n ∧ 7^n % 5 = n^4 % 5 ∧ (∀ m : ℕ, 0 < m ∧ 7^m % 5 = m^4 % 5 → n ≤ m) ∧ n = 4 :=
by
  sorry

end smallest_n_for_congruence_l15_15434


namespace side_length_of_square_l15_15382

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l15_15382


namespace comb_10_3_eq_120_l15_15727

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15727


namespace julie_reads_tomorrow_l15_15113

theorem julie_reads_tomorrow :
  let total_pages := 120
  let pages_read_yesterday := 12
  let pages_read_today := 2 * pages_read_yesterday
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  remaining_pages / 2 = 42 :=
by
  sorry

end julie_reads_tomorrow_l15_15113


namespace coins_remainder_l15_15853

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l15_15853


namespace friends_number_options_l15_15885

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l15_15885


namespace solve_for_a_minus_b_l15_15575

theorem solve_for_a_minus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 7) (h3 : |a + b| = a + b) : a - b = -2 := 
sorry

end solve_for_a_minus_b_l15_15575


namespace min_transport_cost_l15_15902

-- Definitions based on conditions
def total_washing_machines : ℕ := 100
def typeA_max_count : ℕ := 4
def typeB_max_count : ℕ := 8
def typeA_cost : ℕ := 400
def typeA_capacity : ℕ := 20
def typeB_cost : ℕ := 300
def typeB_capacity : ℕ := 10

-- Minimum transportation cost calculation
def min_transportation_cost : ℕ :=
  let typeA_trucks_used := min typeA_max_count (total_washing_machines / typeA_capacity)
  let remaining_washing_machines := total_washing_machines - typeA_trucks_used * typeA_capacity
  let typeB_trucks_used := min typeB_max_count (remaining_washing_machines / typeB_capacity)
  typeA_trucks_used * typeA_cost + typeB_trucks_used * typeB_cost

-- Lean 4 statement to prove the minimum transportation cost
theorem min_transport_cost : min_transportation_cost = 2200 := by
  sorry

end min_transport_cost_l15_15902


namespace problem_l15_15935

theorem problem (a b c : ℤ) (h1 : 0 < c) (h2 : c < 90) (h3 : Real.sqrt (9 - 8 * Real.sin (50 * Real.pi / 180)) = a + b * Real.sin (c * Real.pi / 180)) : 
  (a + b) / c = 1 / 2 :=
by
  sorry

end problem_l15_15935


namespace total_weight_of_oranges_l15_15990

theorem total_weight_of_oranges :
  let capacity1 := 80
  let capacity2 := 50
  let capacity3 := 60
  let filled1 := 3 / 4
  let filled2 := 3 / 5
  let filled3 := 2 / 3
  let weight_per_orange1 := 0.25
  let weight_per_orange2 := 0.30
  let weight_per_orange3 := 0.40
  let num_oranges1 := capacity1 * filled1
  let num_oranges2 := capacity2 * filled2
  let num_oranges3 := capacity3 * filled3
  let total_weight1 := num_oranges1 * weight_per_orange1
  let total_weight2 := num_oranges2 * weight_per_orange2
  let total_weight3 := num_oranges3 * weight_per_orange3
  total_weight1 + total_weight2 + total_weight3 = 40 := by
  sorry

end total_weight_of_oranges_l15_15990


namespace square_side_length_l15_15389

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l15_15389


namespace y_days_worked_l15_15661

theorem y_days_worked 
  ( W : ℝ )
  ( x_rate : ℝ := W / 21 )
  ( y_rate : ℝ := W / 15 )
  ( d : ℝ )
  ( y_work_done : ℝ := d * y_rate )
  ( x_work_done_after_y_leaves : ℝ := 14 * x_rate )
  ( total_work_done : y_work_done + x_work_done_after_y_leaves = W ) :
  d = 5 := 
sorry

end y_days_worked_l15_15661


namespace least_positive_three_digit_multiple_of_11_l15_15825

theorem least_positive_three_digit_multiple_of_11 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n ∧ n = 110 :=
by {
  use 110,
  split,
  { exact Nat.le_refl _ },
  split,
  { norm_num },
  split,
  { use 10,
    norm_num },
  { refl },
  sorry
}

end least_positive_three_digit_multiple_of_11_l15_15825


namespace solution_set_inequality_l15_15081

noncomputable def f : ℝ → ℝ := sorry

variable {f : ℝ → ℝ}
variable (hf_diff : Differentiable ℝ f)
variable (hf_ineq : ∀ x, f x > deriv f x)
variable (hf_zero : f 0 = 2)

theorem solution_set_inequality : {x : ℝ | f x < 2 * Real.exp x} = {x | 0 < x} :=
by
  sorry

end solution_set_inequality_l15_15081


namespace combination_10_3_l15_15700

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l15_15700


namespace friends_number_options_l15_15886

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l15_15886


namespace lexie_and_tom_apple_picking_l15_15123

theorem lexie_and_tom_apple_picking :
    ∀ (lexie_apples : ℕ),
    lexie_apples = 12 →
    (let tom_apples := 2 * lexie_apples in
     let total_apples := lexie_apples + tom_apples in
     total_apples = 36) :=
by
  intros lexie_apples h_lexie_apples
  let tom_apples := 2 * lexie_apples
  let total_apples := lexie_apples + tom_apples
  rw h_lexie_apples at *
  simp at *
  exact sorry

end lexie_and_tom_apple_picking_l15_15123


namespace positive_difference_eq_30_l15_15994

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l15_15994


namespace profit_percentage_l15_15658

theorem profit_percentage (C S : ℝ) (hC : C = 60) (hS : S = 75) : ((S - C) / C) * 100 = 25 :=
by
  sorry

end profit_percentage_l15_15658


namespace factorization_of_polynomial_l15_15840

theorem factorization_of_polynomial :
  (x : ℤ) → x^10 + x^5 + 1 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) :=
by
  sorry

end factorization_of_polynomial_l15_15840


namespace square_side_length_l15_15413

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l15_15413


namespace pipe_A_time_to_fill_l15_15157

theorem pipe_A_time_to_fill (T_B : ℝ) (T_combined : ℝ) (T_A : ℝ): 
  T_B = 75 → T_combined = 30 → 
  (1 / T_B + 1 / T_A = 1 / T_combined) → T_A = 50 :=
by
  -- Placeholder proof
  intro h1 h2 h3
  have h4 : T_B = 75 := h1
  have h5 : T_combined = 30 := h2
  have h6 : 1 / T_B + 1 / T_A = 1 / T_combined := h3
  sorry

end pipe_A_time_to_fill_l15_15157


namespace largest_integer_less_than_100_with_remainder_4_l15_15226

theorem largest_integer_less_than_100_with_remainder_4 (k n : ℤ) (h1 : k = 7 * n + 4) (h2 : k < 100) : k ≤ 95 :=
sorry

end largest_integer_less_than_100_with_remainder_4_l15_15226


namespace quadruple_equation_solution_count_l15_15921

theorem quadruple_equation_solution_count (
    a b c d : ℕ
) (h_pos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_order: a < b ∧ b < c ∧ c < d) 
  (h_equation: 2 * a + 2 * b + 2 * c + 2 * d = d^2 - c^2 + b^2 - a^2) : 
  num_correct_statements = 2 :=
sorry

end quadruple_equation_solution_count_l15_15921


namespace coins_remainder_l15_15859

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l15_15859


namespace students_count_l15_15490

noncomputable def num_students (N T : ℕ) : Prop :=
  T = 72 * N ∧ (T - 200) / (N - 5) = 92

theorem students_count (N T : ℕ) : num_students N T → N = 13 :=
by
  sorry

end students_count_l15_15490


namespace gross_profit_percentage_is_correct_l15_15445

def selling_price : ℝ := 28
def wholesale_cost : ℝ := 24.56
def gross_profit : ℝ := selling_price - wholesale_cost

-- Define the expected profit percentage as a constant value.
def expected_profit_percentage : ℝ := 14.01

theorem gross_profit_percentage_is_correct :
  ((gross_profit / wholesale_cost) * 100) = expected_profit_percentage :=
by
  -- Placeholder for proof
  sorry

end gross_profit_percentage_is_correct_l15_15445


namespace equation_conditions_l15_15980

theorem equation_conditions (m n : ℤ) (h1 : m ≠ 1) (h2 : n = 1) :
  ∃ x : ℤ, (m - 1) * x = 3 ↔ m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end equation_conditions_l15_15980


namespace find_actual_price_of_good_l15_15425

theorem find_actual_price_of_good (P : ℝ) (price_after_discounts : P * 0.93 * 0.90 * 0.85 * 0.75 = 6600) :
  P = 11118.75 :=
by
  sorry

end find_actual_price_of_good_l15_15425


namespace unknown_diagonal_length_l15_15137

noncomputable def rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem unknown_diagonal_length
  (area : ℝ) (d2 : ℝ) (h_area : area = 150)
  (h_d2 : d2 = 30) :
  rhombus_diagonal_length area d2 = 10 :=
  by
  rw [h_area, h_d2]
  -- Here, the essential proof would go
  -- Since solving would require computation,
  -- which we are omitting, we use:
  sorry

end unknown_diagonal_length_l15_15137


namespace Anton_thought_number_is_729_l15_15046

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end Anton_thought_number_is_729_l15_15046


namespace positive_difference_solutions_abs_eq_30_l15_15999

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l15_15999


namespace number_of_cans_on_third_day_l15_15175

-- Definition of an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

theorem number_of_cans_on_third_day :
  (arithmetic_sequence 4 5 2 = 9) →   -- on the second day, he found 9 cans
  (arithmetic_sequence 4 5 7 = 34) →  -- on the seventh day, he found 34 cans
  (arithmetic_sequence 4 5 3 = 14) :=  -- therefore, on the third day, he found 14 cans
by
  intros h1 h2
  sorry

end number_of_cans_on_third_day_l15_15175


namespace ethel_subtracts_l15_15819

theorem ethel_subtracts (h : 50^2 = 2500) : 2500 - 99 = 49^2 :=
by
  sorry

end ethel_subtracts_l15_15819


namespace largest_multiple_of_9_less_than_100_l15_15160

theorem largest_multiple_of_9_less_than_100 : ∃ n, n < 100 ∧ n % 9 = 0 ∧ (∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 99,
  split,
  -- Prove the number is less than 100
  { linarith, },
  split,
  -- Prove the number is a multiple of 9
  { norm_num, },
  -- Prove it is the largest
  { intros m hlt hdiv,
    linarith [hlt, hdiv],
    norm_num at hdiv,
    cases hdiv,
    exact hdiv_left, },
end

end largest_multiple_of_9_less_than_100_l15_15160


namespace simple_interest_is_correct_l15_15151

-- Define the principal amount, rate of interest, and time
def P : ℕ := 400
def R : ℚ := 22.5
def T : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P : ℕ) (R : ℚ) (T : ℕ) : ℚ :=
  (P * R * T) / 100

-- The statement we need to prove
theorem simple_interest_is_correct : simple_interest P R T = 90 :=
by
  sorry

end simple_interest_is_correct_l15_15151


namespace average_of_remaining_two_l15_15491

theorem average_of_remaining_two
  (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
  (h_avg_2_1 : (a + b) / 2 = 4.2)
  (h_avg_2_2 : (c + d) / 2 = 3.85) : 
  ((e + f) / 2) = 3.8 :=
by
  sorry

end average_of_remaining_two_l15_15491


namespace vera_first_place_l15_15549

noncomputable def placement (anna vera katya natasha : ℕ) : Prop :=
  (anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)

theorem vera_first_place :
  ∃ (anna vera katya natasha : ℕ),
    (placement anna vera katya natasha) ∧ 
    (vera = 1) ∧ 
    (1 ≠ 4) → 
    ((anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)) ∧ 
    (1 = 1) ∧ 
    (∃ i j k l : ℕ, (i ≠ 1 ∧ i ≠ 4) ∧ (j = 1) ∧ (k ≠ 1) ∧ (l = 4)) ∧ 
    (vera = 1) :=
sorry

end vera_first_place_l15_15549


namespace square_side_length_l15_15392

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l15_15392


namespace inequality_solution_l15_15278

theorem inequality_solution (x : ℝ) : 
  x^2 - 9 * x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := 
by
  sorry

end inequality_solution_l15_15278


namespace coins_remainder_l15_15857

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l15_15857


namespace elsa_final_marbles_l15_15066

def initial_marbles : ℕ := 40
def marbles_lost_at_breakfast : ℕ := 3
def marbles_given_to_susie : ℕ := 5
def marbles_bought_by_mom : ℕ := 12
def twice_marbles_given_back : ℕ := 2 * marbles_given_to_susie

theorem elsa_final_marbles :
    initial_marbles
    - marbles_lost_at_breakfast
    - marbles_given_to_susie
    + marbles_bought_by_mom
    + twice_marbles_given_back = 54 := 
by
    sorry

end elsa_final_marbles_l15_15066


namespace matt_peanut_revenue_l15_15777

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l15_15777


namespace cos_two_pi_over_three_plus_two_alpha_l15_15092

theorem cos_two_pi_over_three_plus_two_alpha 
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := 
by
  sorry

end cos_two_pi_over_three_plus_two_alpha_l15_15092


namespace grade_assignment_ways_l15_15534

theorem grade_assignment_ways (n_students : ℕ) (n_grades : ℕ) (h_students : n_students = 12) (h_grades : n_grades = 4) :
  (n_grades ^ n_students) = 16777216 := by
  rw [h_students, h_grades]
  rfl

end grade_assignment_ways_l15_15534


namespace rectangular_solid_surface_area_l15_15503

theorem rectangular_solid_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 14) 
  (h2 : a^2 + b^2 + c^2 = 121) : 
  2 * (a * b + b * c + a * c) = 75 := 
by
  sorry

end rectangular_solid_surface_area_l15_15503


namespace find_divisor_l15_15786

theorem find_divisor (d : ℕ) (H1 : 199 = d * 11 + 1) : d = 18 := 
sorry

end find_divisor_l15_15786


namespace james_music_BPM_l15_15767

theorem james_music_BPM 
  (hours_per_day : ℕ)
  (beats_per_week : ℕ)
  (days_per_week : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_per_day : ℕ)
  (total_minutes_per_week : ℕ)
  (BPM : ℕ)
  (h1 : hours_per_day = 2)
  (h2 : beats_per_week = 168000)
  (h3 : days_per_week = 7)
  (h4 : minutes_per_hour = 60)
  (h5 : minutes_per_day = hours_per_day * minutes_per_hour)
  (h6 : total_minutes_per_week = minutes_per_day * days_per_week)
  (h7 : BPM = beats_per_week / total_minutes_per_week)
  : BPM = 200 :=
sorry

end james_music_BPM_l15_15767


namespace solve_real_triples_l15_15739

theorem solve_real_triples (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b) ∧
   b * (c^2 + a) = a * (a + b * c) ∧
   c * (a^2 + b) = b * (b + c * a)) ↔ 
  (∃ (x : ℝ), (a = x) ∧ (b = x) ∧ (c = x)) ∨ 
  (b = 0 ∧ c = 0) :=
sorry

end solve_real_triples_l15_15739


namespace largest_multiple_of_9_less_than_100_l15_15165

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n < 100 ∧ n % 9 = 0 ∧ ∀ m, m < 100 ∧ m % 9 = 0 → m ≤ n :=
begin
  use 99,
  split,
  { norm_num }, -- 99 < 100
  split,
  { norm_num }, -- 99 % 9 = 0
  { intros m hm,
    obtain ⟨k, rfl⟩ := nat.exists_eq_mul_right_of_dvd (dvd_of_mod_eq_zero hm.2),
    rw [mul_comm, nat.mul_lt_mul_iff_left (by norm_num : 0 < 9)],
    norm_num,
    exact nat.succ_le_iff.mpr (le_of_lt hm.1), }
end

end largest_multiple_of_9_less_than_100_l15_15165


namespace ages_correct_l15_15376

variables (Son Daughter Wife Man Father : ℕ)

theorem ages_correct :
  (Man = Son + 20) ∧
  (Man = Daughter + 15) ∧
  (Man + 2 = 2 * (Son + 2)) ∧
  (Man + 2 = 3 * (Daughter + 2)) ∧
  (Wife = Man - 5) ∧
  (Wife + 6 = 2 * (Daughter + 6)) ∧
  (Father = Man + 32) →
  (Son = 7 ∧ Daughter = 12 ∧ Wife = 22 ∧ Man = 27 ∧ Father = 59) :=
by
  intros h
  sorry

end ages_correct_l15_15376


namespace carpet_dimensions_l15_15679

-- Define the problem parameters
def width_a : ℕ := 50
def width_b : ℕ := 38

-- The dimensions x and y are integral numbers of feet
variables (x y : ℕ)

-- The same length L for both rooms that touches all four walls
noncomputable def length (x y : ℕ) : ℚ := (22 * (x^2 + y^2)) / (x * y)

-- The final theorem to be proven
theorem carpet_dimensions (x y : ℕ) (h : (x^2 + y^2) * 1056 = (x * y) * 48 * (length x y)) : (x = 50) ∧ (y = 25) :=
by
  sorry -- Proof is omitted

end carpet_dimensions_l15_15679


namespace equation_of_trisection_line_l15_15302

/-- Let P be the point (1, 2) and let A and B be the points (2, 3) and (-3, 0), respectively. 
    One of the lines through point P and a trisection point of the line segment joining A and B has 
    the equation 3x + 7y = 17. -/
theorem equation_of_trisection_line :
  let P : ℝ × ℝ := (1, 2)
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (-3, 0)
  -- Definition of the trisection points
  let T1 : ℝ × ℝ := ((2 + (-3 - 2) / 3) / 1, (3 + (0 - 3) / 3) / 1) -- First trisection point
  let T2 : ℝ × ℝ := ((2 + 2 * (-3 - 2) / 3) / 1, (3 + 2 * (0 - 3) / 3) / 1) -- Second trisection point
  -- Equation of the line through P and T2 is 3x + 7y = 17
  3 * (P.1 + P.2) + 7 * (P.2 + T2.2) = 17 :=
sorry

end equation_of_trisection_line_l15_15302


namespace Anton_thought_of_729_l15_15050

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l15_15050


namespace find_angle_l15_15660

def complementary (x : ℝ) := 90 - x
def supplementary (x : ℝ) := 180 - x

theorem find_angle (x : ℝ) (h : supplementary x = 3 * complementary x) : x = 45 :=
by 
  sorry

end find_angle_l15_15660


namespace ab_value_l15_15808

theorem ab_value (a b : ℝ) (h1 : b^2 - a^2 = 4) (h2 : a^2 + b^2 = 25) : abs (a * b) = Real.sqrt (609 / 4) := 
sorry

end ab_value_l15_15808


namespace find_positive_integers_l15_15205

theorem find_positive_integers (a b c : ℕ) (ha : a ≥ b) (hb : b ≥ c) :
  (∃ n₁ : ℕ, a^2 + 3 * b = n₁^2) ∧ 
  (∃ n₂ : ℕ, b^2 + 3 * c = n₂^2) ∧ 
  (∃ n₃ : ℕ, c^2 + 3 * a = n₃^2) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 37 ∧ b = 25 ∧ c = 17) :=
by
  sorry

end find_positive_integers_l15_15205


namespace largest_possible_b_l15_15615

theorem largest_possible_b (b : ℚ) (h : (3 * b + 7) * (b - 2) = 9 * b) : b ≤ 2 :=
sorry

end largest_possible_b_l15_15615


namespace area_of_quadrilateral_l15_15911

-- Definitions of the given conditions
def diagonal_length : ℝ := 40
def offset1 : ℝ := 11
def offset2 : ℝ := 9

-- The area of the quadrilateral
def quadrilateral_area : ℝ := 400

-- Proof statement
theorem area_of_quadrilateral :
  (1/2 * diagonal_length * offset1 + 1/2 * diagonal_length * offset2) = quadrilateral_area :=
by sorry

end area_of_quadrilateral_l15_15911


namespace fraction_division_addition_l15_15655

theorem fraction_division_addition :
  (3 / 7 / 4) + (2 / 7) = 11 / 28 := by
  sorry

end fraction_division_addition_l15_15655


namespace find_three_digit_number_l15_15906

def is_valid_three_digit_number (M G U : ℕ) : Prop :=
  M ≠ G ∧ G ≠ U ∧ M ≠ U ∧ 
  0 ≤ M ∧ M ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧
  100 * M + 10 * G + U = (M + G + U) * (M + G + U - 2)

theorem find_three_digit_number : ∃ (M G U : ℕ), 
  is_valid_three_digit_number M G U ∧
  100 * M + 10 * G + U = 195 :=
by
  sorry

end find_three_digit_number_l15_15906


namespace largest_int_less_than_100_by_7_l15_15231

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l15_15231


namespace math_proof_problem_l15_15191

noncomputable def mixed_number_eval : ℚ :=
  65 * ((4 + 1 / 3) + (3 + 1 / 2)) / ((2 + 1 / 5) - (1 + 2 / 3))

theorem math_proof_problem :
  mixed_number_eval = 954 + 33 / 48 := 
sorry

end math_proof_problem_l15_15191


namespace sixth_graders_l15_15004

theorem sixth_graders (total_students sixth_graders seventh_graders : ℕ)
    (h1 : seventh_graders = 64)
    (h2 : 32 * total_students = 64 * 100)
    (h3 : sixth_graders * 100 = 38 * total_students) :
    sixth_graders = 76 := by
  sorry

end sixth_graders_l15_15004


namespace least_positive_three_digit_multiple_of_11_l15_15826

theorem least_positive_three_digit_multiple_of_11 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 11 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 11 ∣ m → n ≤ m) :=
begin
  use 110,
  split,
  exact (100 ≤ 110),
  split,
  exact (110 ≤ 999),
  split,
  exact (11 ∣ 110),
  intros m Hm,
  cases Hm with Hm100 HmGCD,
  cases HmGCD with Hm999 HmDiv,
  sorry,
end

end least_positive_three_digit_multiple_of_11_l15_15826


namespace estate_value_l15_15483

theorem estate_value (x : ℕ) (E : ℕ) (cook_share : ℕ := 500) 
  (daughter_share : ℕ := 4 * x) (son_share : ℕ := 3 * x) 
  (wife_share : ℕ := 6 * x) (estate_eqn : E = 14 * x) : 
  2 * (daughter_share + son_share) = E ∧ wife_share = 2 * son_share ∧ E = 13 * x + cook_share → 
  E = 7000 :=
by
  sorry

end estate_value_l15_15483


namespace julie_read_pages_tomorrow_l15_15114

-- Definitions based on conditions
def total_pages : ℕ := 120
def pages_read_yesterday : ℕ := 12
def pages_read_today : ℕ := 2 * pages_read_yesterday

-- Problem statement to prove Julie should read 42 pages tomorrow
theorem julie_read_pages_tomorrow :
  let total_read := pages_read_yesterday + pages_read_today in
  let remaining_pages := total_pages - total_read in
  let pages_tomorrow := remaining_pages / 2 in
  pages_tomorrow = 42 :=
by
  sorry

end julie_read_pages_tomorrow_l15_15114


namespace solve_geometric_sequence_product_l15_15608

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r

theorem solve_geometric_sequence_product (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h_a35 : a 3 * a 5 = 4) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 :=
sorry

end solve_geometric_sequence_product_l15_15608


namespace gcd_324_243_135_l15_15007

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_324_243_135_l15_15007


namespace pct_three_petals_is_75_l15_15948

-- Given Values
def total_clovers : Nat := 200
def pct_two_petals : Nat := 24
def pct_four_petals : Nat := 1

-- Statement: Prove that the percentage of clovers with three petals is 75%
theorem pct_three_petals_is_75 :
  (100 - pct_two_petals - pct_four_petals) = 75 := by
  sorry

end pct_three_petals_is_75_l15_15948


namespace correct_operation_l15_15839

noncomputable def valid_operation (n : ℕ) (a b : ℕ) (c d : ℤ) (x : ℚ) : Prop :=
  match n with
  | 0 => (x ^ a / x ^ b = x ^ (a - b))
  | 1 => (x ^ a * x ^ b = x ^ (a + b))
  | 2 => (c * x ^ a + d * x ^ a = (c + d) * x ^ a)
  | 3 => ((c * x ^ a) ^ b = c ^ b * x ^ (a * b))
  | _ => False

theorem correct_operation (x : ℚ) : valid_operation 1 2 3 0 0 x :=
by sorry

end correct_operation_l15_15839


namespace right_triangle_hypotenuse_l15_15282

theorem right_triangle_hypotenuse (x : ℝ) (h : x^2 = 3^2 + 5^2) : x = Real.sqrt 34 :=
by sorry

end right_triangle_hypotenuse_l15_15282


namespace function_characterization_l15_15478

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization :
  (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) →
  (∀ x : ℝ, 0 ≤ x → f x = if x < 2 then 2 / (2 - x) else 0) := sorry

end function_characterization_l15_15478


namespace coloring_possible_l15_15126

-- Define what it means for a graph to be planar and bipartite
def planar_graph (G : Type) : Prop := sorry
def bipartite_graph (G : Type) : Prop := sorry

-- The planar graph G results after subdivision without introducing new intersections
def subdivided_graph (G : Type) : Type := sorry

-- Main theorem to prove
theorem coloring_possible (G : Type) (h1 : planar_graph G) : 
  bipartite_graph (subdivided_graph G) :=
sorry

end coloring_possible_l15_15126


namespace polygon_perimeter_l15_15538

theorem polygon_perimeter (side_length : ℝ) (ext_angle_deg : ℝ) (n : ℕ) (h1 : side_length = 8) 
  (h2 : ext_angle_deg = 90) (h3 : ext_angle_deg = 360 / n) : 
  4 * side_length = 32 := 
  by 
    sorry

end polygon_perimeter_l15_15538


namespace comb_10_3_eq_120_l15_15732

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l15_15732


namespace water_filter_capacity_l15_15316

theorem water_filter_capacity (x : ℝ) (h : 0.30 * x = 36) : x = 120 :=
sorry

end water_filter_capacity_l15_15316


namespace pos_numbers_equal_l15_15600

theorem pos_numbers_equal (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eq : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end pos_numbers_equal_l15_15600


namespace odd_function_increasing_l15_15918

variables {f : ℝ → ℝ}

/-- Let f be an odd function defined on (-∞, 0) ∪ (0, ∞). 
If ∀ y z ∈ (0, ∞), y ≠ z → (f y - f z) / (y - z) > 0, then f(-3) > f(-5). -/
theorem odd_function_increasing {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ y z : ℝ, y > 0 → z > 0 → y ≠ z → (f y - f z) / (y - z) > 0) :
  f (-3) > f (-5) :=
sorry

end odd_function_increasing_l15_15918


namespace find_two_digit_number_l15_15985

theorem find_two_digit_number (N : ℕ) (a b c : ℕ) 
  (h_end_digits : N % 1000 = c + 10 * b + 100 * a)
  (hN2_end_digits : N^2 % 1000 = c + 10 * b + 100 * a)
  (h_nonzero : a ≠ 0) :
  10 * a + b = 24 := 
by
  sorry

end find_two_digit_number_l15_15985


namespace square_side_length_l15_15386

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l15_15386


namespace increasing_function_iff_l15_15085

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then (-x^2 + 2 * a * x - 3) else (4 - a) * x + 1

theorem increasing_function_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 1 ≤ a ∧ a ≤ 3 :=
by 
  -- The proof would go here. For now, we use sorry to skip it.
  sorry

end increasing_function_iff_l15_15085


namespace sin_double_angle_l15_15447

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 4 / 5) : Real.sin (2 * x) = -7 / 25 := 
by 
  sorry

end sin_double_angle_l15_15447


namespace max_cigarettes_with_staggered_packing_l15_15025

theorem max_cigarettes_with_staggered_packing :
  ∃ n : ℕ, n > 160 ∧ n = 176 :=
by
  let diameter := 2
  let rows_initial := 8
  let cols_initial := 20
  let total_initial := rows_initial * cols_initial
  have h1 : total_initial = 160 := by norm_num
  let alternative_packing_capacity := 176
  have h2 : alternative_packing_capacity > total_initial := by norm_num
  use alternative_packing_capacity
  exact ⟨h2, rfl⟩

end max_cigarettes_with_staggered_packing_l15_15025
