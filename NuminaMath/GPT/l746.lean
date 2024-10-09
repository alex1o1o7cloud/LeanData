import Mathlib

namespace original_mixture_percentage_l746_74639

variables (a w : ℝ)

-- Conditions given
def condition1 : Prop := a / (a + w + 2) = 0.3
def condition2 : Prop := (a + 2) / (a + w + 4) = 0.4

theorem original_mixture_percentage (h1 : condition1 a w) (h2 : condition2 a w) : (a / (a + w)) * 100 = 36 :=
by
sorry

end original_mixture_percentage_l746_74639


namespace negation_of_forall_l746_74669

theorem negation_of_forall (h : ¬ ∀ x > 0, Real.exp x > x + 1) : ∃ x > 0, Real.exp x < x + 1 :=
sorry

end negation_of_forall_l746_74669


namespace find_values_of_a_l746_74692

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

-- The theorem we want to prove
theorem find_values_of_a (a : ℝ) : (A ∪ B a = A) ↔ (a = -6 ∨ a = 0 ∨ a = 3) :=
by
  sorry

end find_values_of_a_l746_74692


namespace alice_has_ball_after_two_turns_l746_74698

noncomputable def probability_alice_has_ball_twice_turns : ℚ :=
  let P_AB_A : ℚ := 1/2 * 1/3
  let P_ABC_A : ℚ := 1/2 * 1/3 * 1/2
  let P_AA : ℚ := 1/2 * 1/2
  P_AB_A + P_ABC_A + P_AA

theorem alice_has_ball_after_two_turns :
  probability_alice_has_ball_twice_turns = 1/2 := 
by
  sorry

end alice_has_ball_after_two_turns_l746_74698


namespace martha_cards_l746_74667

theorem martha_cards (start_cards : ℕ) : start_cards + 76 = 79 → start_cards = 3 :=
by
  sorry

end martha_cards_l746_74667


namespace find_p_q_l746_74661

variable (R : Set ℝ)

def A (p : ℝ) : Set ℝ := {x | x^2 + p * x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5 * x + q = 0}

theorem find_p_q 
  (h : (R \ (A p)) ∩ (B q) = {2}) : p + q = -1 :=
by
  sorry

end find_p_q_l746_74661


namespace eval_expr_at_3_l746_74658

theorem eval_expr_at_3 : (3^2 - 5 * 3 + 6) / (3 - 2) = 0 := by
  sorry

end eval_expr_at_3_l746_74658


namespace rectangle_area_error_percentage_l746_74638

theorem rectangle_area_error_percentage 
  (L W : ℝ)
  (measured_length : ℝ := L * 1.16)
  (measured_width : ℝ := W * 0.95)
  (actual_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width) :
  ((measured_area - actual_area) / actual_area) * 100 = 10.2 := 
by
  sorry

end rectangle_area_error_percentage_l746_74638


namespace max_saved_houses_l746_74683

theorem max_saved_houses (n c : ℕ) (h₁ : 1 ≤ c ∧ c ≤ n / 2) : 
  ∃ k, k = n^2 + c^2 - n * c - c :=
by
  sorry

end max_saved_houses_l746_74683


namespace necessary_but_not_sufficient_l746_74622

theorem necessary_but_not_sufficient (x y : ℝ) :
  (x = 0) → (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0) :=
by sorry

end necessary_but_not_sufficient_l746_74622


namespace simon_number_of_legos_l746_74648

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l746_74648


namespace weight_ratios_l746_74618

theorem weight_ratios {x y z k : ℝ} (h1 : x + y = k * z) (h2 : y + z = k * x) (h3 : z + x = k * y) : x = y ∧ y = z :=
by 
  -- Proof to be filled in later
  sorry

end weight_ratios_l746_74618


namespace find_divisor_l746_74668

def remainder : Nat := 1
def quotient : Nat := 54
def dividend : Nat := 217

theorem find_divisor : ∃ divisor : Nat, (dividend = divisor * quotient + remainder) ∧ divisor = 4 :=
by
  sorry

end find_divisor_l746_74668


namespace find_circle_eq_find_range_of_dot_product_l746_74630

open Real
open Set

-- Define the problem conditions
def line_eq (x y : ℝ) : Prop := x - sqrt 3 * y = 4
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P inside the circle and condition that |PA|, |PO|, |PB| form a geometric sequence
def geometric_sequence_condition (x y : ℝ) : Prop :=
  sqrt ((x + 2)^2 + y^2) * sqrt ((x - 2)^2 + y^2) = x^2 + y^2

-- Prove the equation of the circle
theorem find_circle_eq :
  (∃ (r : ℝ), ∀ (x y : ℝ), line_eq x y → r = 2) → circle_eq x y :=
by
  -- skipping the proof
  sorry

-- Prove the range of values for the dot product
theorem find_range_of_dot_product :
  (∀ (x y : ℝ), circle_eq x y ∧ geometric_sequence_condition x y) →
  -2 < (x^2 - 1 * y^2 - 1) → (x^2 - 4 + y^2) < 0 :=
by
  -- skipping the proof
  sorry

end find_circle_eq_find_range_of_dot_product_l746_74630


namespace comparison_of_prices_l746_74609

theorem comparison_of_prices:
  ∀ (x y : ℝ), (6 * x + 3 * y > 24) → (4 * x + 5 * y < 22) → (2 * x > 3 * y) :=
by
  intros x y h1 h2
  sorry

end comparison_of_prices_l746_74609


namespace julie_initial_savings_l746_74693

-- Definition of the simple interest condition
def simple_interest_condition (P : ℝ) : Prop :=
  575 = P * 0.04 * 5

-- Definition of the compound interest condition
def compound_interest_condition (P : ℝ) : Prop :=
  635 = P * ((1 + 0.05) ^ 5 - 1)

-- The final proof problem
theorem julie_initial_savings (P : ℝ) :
  simple_interest_condition P →
  compound_interest_condition P →
  2 * P = 5750 :=
by sorry

end julie_initial_savings_l746_74693


namespace cs_competition_hits_l746_74620

theorem cs_competition_hits :
  (∃ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1)
  ∧ (∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1 → (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)) :=
by
  sorry

end cs_competition_hits_l746_74620


namespace problem1_problem2_l746_74675

-- Definitions of the sets A, B, C
def A (a : ℝ) : Set ℝ := { x | x^2 - a*x + a^2 - 12 = 0 }
def B : Set ℝ := { x | x^2 - 2*x - 8 = 0 }
def C (m : ℝ) : Set ℝ := { x | m*x + 1 = 0 }

-- Problem 1: If A = B, then a = 2
theorem problem1 (a : ℝ) (h : A a = B) : a = 2 := sorry

-- Problem 2: If B ∪ C m = B, then m ∈ {-1/4, 0, 1/2}
theorem problem2 (m : ℝ) (h : B ∪ C m = B) : m = -1/4 ∨ m = 0 ∨ m = 1/2 := sorry

end problem1_problem2_l746_74675


namespace color_column_l746_74607

theorem color_column (n : ℕ) (color : ℕ) (board : ℕ → ℕ → ℕ) 
  (h_colors : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2)
  (h_block : ∀ i j, (∀ k l : ℕ, k < n → l < n → ∃ c, ∀ a b : ℕ, k + a * n < n → l + b * n < n → board (i + k + a * n) (j + l + b * n) = c))
  (h_row : ∃ r, ∀ k, k < n → ∃ c, 1 ≤ c ∧ c ≤ n ∧ board r k = c) :
  ∃ c, (∀ j, 1 ≤ board c j ∧ board c j ≤ n) :=
sorry

end color_column_l746_74607


namespace shells_put_back_l746_74614

def shells_picked_up : ℝ := 324.0
def shells_left : ℝ := 32.0

theorem shells_put_back : shells_picked_up - shells_left = 292 := by
  sorry

end shells_put_back_l746_74614


namespace units_digit_17_pow_31_l746_74657

theorem units_digit_17_pow_31 : (17 ^ 31) % 10 = 3 := by
  sorry

end units_digit_17_pow_31_l746_74657


namespace total_price_correct_l746_74601

-- Define the initial price, reduction, and the number of boxes
def initial_price : ℝ := 104
def price_reduction : ℝ := 24
def number_of_boxes : ℕ := 20

-- Define the new price as initial price minus the reduction
def new_price := initial_price - price_reduction

-- Define the total price as the new price times the number of boxes
def total_price := (number_of_boxes : ℝ) * new_price

-- The goal is to prove the total price equals 1600
theorem total_price_correct : total_price = 1600 := by
  sorry

end total_price_correct_l746_74601


namespace selection_competition_l746_74682

variables (p q r : Prop)

theorem selection_competition 
  (h1 : p ∨ q) 
  (h2 : ¬ (p ∧ q)) 
  (h3 : ¬ q ∧ r) : p ∧ ¬ q ∧ r :=
by
  sorry

end selection_competition_l746_74682


namespace kendall_nickels_l746_74655

def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

theorem kendall_nickels (q d : ℕ) (total : ℝ) (hq : q = 10) (hd : d = 12) (htotal : total = 4) : 
  ∃ n : ℕ, value_of_nickels n = total - (value_of_quarters q + value_of_dimes d) ∧ n = 6 :=
by
  sorry

end kendall_nickels_l746_74655


namespace n_value_condition_l746_74605

theorem n_value_condition (n : ℤ) : 
  (3 * (n ^ 2 + n) + 7) % 5 = 0 ↔ n % 5 = 2 := sorry

end n_value_condition_l746_74605


namespace union_of_A_and_B_l746_74681

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def union_AB := {x : ℝ | 1 < x ∧ x ≤ 8}

theorem union_of_A_and_B : A ∪ B = union_AB :=
sorry

end union_of_A_and_B_l746_74681


namespace ratio_shorter_to_longer_l746_74691

-- Define the total length and the length of the shorter piece
def total_length : ℕ := 90
def shorter_length : ℕ := 20

-- Define the length of the longer piece
def longer_length : ℕ := total_length - shorter_length

-- Define the ratio of shorter piece to longer piece
def ratio := shorter_length / longer_length

-- The target statement to prove
theorem ratio_shorter_to_longer : ratio = 2 / 7 := by
  sorry

end ratio_shorter_to_longer_l746_74691


namespace cost_price_computer_table_l746_74662

theorem cost_price_computer_table :
  ∃ CP : ℝ, CP * 1.25 = 5600 ∧ CP = 4480 :=
by
  sorry

end cost_price_computer_table_l746_74662


namespace max_divisor_of_five_consecutive_integers_l746_74690

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l746_74690


namespace hcl_reaction_l746_74680

theorem hcl_reaction
  (stoichiometry : ∀ (HCl NaHCO3 H2O CO2 NaCl : ℕ), HCl = NaHCO3 ∧ H2O = NaHCO3 ∧ CO2 = NaHCO3 ∧ NaCl = NaHCO3)
  (naHCO3_moles : ℕ)
  (reaction_moles : naHCO3_moles = 3) :
  ∃ (HCl_moles : ℕ), HCl_moles = naHCO3_moles :=
by
  sorry

end hcl_reaction_l746_74680


namespace ratio_of_speeds_l746_74678

theorem ratio_of_speeds (v_A v_B : ℝ) (h1 : 500 / v_A = 400 / v_B) : v_A / v_B = 5 / 4 :=
by
  sorry

end ratio_of_speeds_l746_74678


namespace find_monthly_salary_l746_74606

variable (S : ℝ)

theorem find_monthly_salary
  (h1 : 0.20 * S - 0.20 * (0.20 * S) = 220) :
  S = 1375 :=
by
  -- Proof goes here
  sorry

end find_monthly_salary_l746_74606


namespace second_shift_production_l746_74656

-- Question: Prove that the number of cars produced by the second shift is 1,100 given the conditions
-- Conditions:
-- 1. P_day = 4 * P_second
-- 2. P_day + P_second = 5,500

theorem second_shift_production (P_day P_second : ℕ) (h1 : P_day = 4 * P_second) (h2 : P_day + P_second = 5500) :
  P_second = 1100 := by
  sorry

end second_shift_production_l746_74656


namespace igors_number_l746_74613

-- Define the initial lineup of players
def initialLineup : List ℕ := [9, 7, 11, 10, 6, 8, 5, 4, 1]

-- Define the condition for a player running to the locker room
def runsToLockRoom (n : ℕ) (left : Option ℕ) (right : Option ℕ) : Prop :=
  match left, right with
  | some l, some r => n < l ∨ n < r
  | some l, none   => n < l
  | none, some r   => n < r
  | none, none     => False

-- Define the process of players running to the locker room iteratively
def runProcess : List ℕ → List ℕ := 
  sorry   -- Implementation of the run process is skipped

-- Define the remaining players after repeated commands until 3 players are left
def remainingPlayers (lineup : List ℕ) : List ℕ :=
  sorry  -- Implementation to find the remaining players is skipped

-- Statement of the theorem
theorem igors_number (afterIgorRanOff : List ℕ := remainingPlayers initialLineup)
  (finalLineup : List ℕ := [9, 11, 10]) :
  ∃ n, n ∈ initialLineup ∧ ¬(n ∈ finalLineup) ∧ afterIgorRanOff.length = 3 → n = 5 :=
  sorry

end igors_number_l746_74613


namespace three_digit_number_count_correct_l746_74689

noncomputable
def count_three_digit_numbers (digits : List ℕ) : ℕ :=
  if h : digits.length = 5 then
    (5 * 4 * 3 : ℕ)
  else
    0

theorem three_digit_number_count_correct :
  count_three_digit_numbers [1, 3, 5, 7, 9] = 60 :=
by
  unfold count_three_digit_numbers
  simp only [List.length, if_pos]
  rfl

end three_digit_number_count_correct_l746_74689


namespace democrats_ratio_l746_74677

noncomputable def F : ℕ := 240
noncomputable def M : ℕ := 480
noncomputable def D_F : ℕ := 120
noncomputable def D_M : ℕ := 120

theorem democrats_ratio (total_participants : ℕ := 720)
  (h1 : F + M = total_participants)
  (h2 : D_F = 120)
  (h3 : D_F = 1/2 * F)
  (h4 : D_M = 1/4 * M)
  (h5 : D_F + D_M = 240)
  (h6 : F + M = 720) : (D_F + D_M) / total_participants = 1 / 3 :=
by
  sorry

end democrats_ratio_l746_74677


namespace coloring_satisfies_conditions_l746_74608

/-- Define what it means for a point to be a lattice point -/
def is_lattice_point (x y : ℤ) : Prop := true

/-- Define the coloring function based on coordinates -/
def color (x y : ℤ) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 1) ∨   -- white
  (x % 2 = 1 ∧ y % 2 = 0) ∨   -- black
  (x % 2 = 0)                 -- red (both (even even) and (even odd) are included)

/-- Proving the method of coloring lattice points satisfies the given conditions -/
theorem coloring_satisfies_conditions :
  (∀ x y : ℤ, is_lattice_point x y → 
    color x y ∧ 
    ∃ (A B C : ℤ × ℤ), 
      (is_lattice_point A.fst A.snd ∧ 
       is_lattice_point B.fst B.snd ∧ 
       is_lattice_point C.fst C.snd ∧ 
       color A.fst A.snd ∧ 
       color B.fst B.snd ∧ 
       color C.fst C.snd ∧
       ∃ D : ℤ × ℤ, 
         (is_lattice_point D.fst D.snd ∧ 
          color D.fst D.snd ∧ 
          D.fst = A.fst + C.fst - B.fst ∧ 
          D.snd = A.snd + C.snd - B.snd))) :=
sorry

end coloring_satisfies_conditions_l746_74608


namespace car_speed_is_48_l746_74624

theorem car_speed_is_48 {v : ℝ} : (3600 / v = 75) → v = 48 := 
by {
  sorry
}

end car_speed_is_48_l746_74624


namespace waiter_tables_l746_74673

theorem waiter_tables (total_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (number_of_tables : ℕ) 
  (h1 : total_customers = 22)
  (h2 : customers_left = 14)
  (h3 : people_per_table = 4)
  (h4 : remaining_customers = total_customers - customers_left)
  (h5 : number_of_tables = remaining_customers / people_per_table) :
  number_of_tables = 2 :=
by
  sorry

end waiter_tables_l746_74673


namespace zero_point_six_one_eight_method_l746_74631

theorem zero_point_six_one_eight_method (a b : ℝ) (h : a = 2 ∧ b = 4) : 
  ∃ x₁ x₂, x₁ = a + 0.618 * (b - a) ∧ x₂ = a + b - x₁ ∧ (x₁ = 3.236 ∨ x₂ = 2.764) := by
  sorry

end zero_point_six_one_eight_method_l746_74631


namespace percent_of_70_is_56_l746_74645

theorem percent_of_70_is_56 : (70 / 125) * 100 = 56 := by
  sorry

end percent_of_70_is_56_l746_74645


namespace quadratic_function_min_value_in_interval_l746_74632

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 6 * x + 10

theorem quadratic_function_min_value_in_interval :
  ∀ (x : ℝ), 2 ≤ x ∧ x < 5 → (∃ min_val : ℝ, min_val = 1) ∧ (∀ upper_bound : ℝ, ∃ x0 : ℝ, x0 < 5 ∧ quadratic_function x0 > upper_bound) := 
by
  sorry

end quadratic_function_min_value_in_interval_l746_74632


namespace problem_statement_l746_74623

-- Definition of sum of digits function
def S (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Definition of the function f₁
def f₁ (k : ℕ) : ℕ :=
  (S k) ^ 2

-- Definition of the function fₙ₊₁
def f : ℕ → ℕ → ℕ
| 0, k => k
| (n+1), k => f₁ (f n k)

-- Theorem stating the proof problem
theorem problem_statement : f 2005 (2 ^ 2006) = 169 :=
  sorry

end problem_statement_l746_74623


namespace remainder_is_162_l746_74615

def polynomial (x : ℝ) : ℝ := 2 * x^4 - x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_is_162 : polynomial 3 = 162 :=
by 
  sorry

end remainder_is_162_l746_74615


namespace yoongi_has_fewest_apples_l746_74629

noncomputable def yoongi_apples : ℕ := 4
noncomputable def yuna_apples : ℕ := 5
noncomputable def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples : yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples := by
  sorry

end yoongi_has_fewest_apples_l746_74629


namespace math_lovers_l746_74663

/-- The proof problem: 
Given 1256 students in total and the difference of 408 between students who like math and others,
prove that the number of students who like math is 424, given that students who like math are fewer than 500.
--/
theorem math_lovers (M O : ℕ) (h1 : M + O = 1256) (h2: O - M = 408) (h3 : M < 500) : M = 424 :=
by
  sorry

end math_lovers_l746_74663


namespace cards_ratio_l746_74654

theorem cards_ratio (b_c : ℕ) (m_c : ℕ) (m_l : ℕ) (m_g : ℕ) 
  (h1 : b_c = 20) 
  (h2 : m_c = b_c + 8) 
  (h3 : m_l = 14) 
  (h4 : m_g = m_c - m_l) : 
  m_g / m_c = 1 / 2 :=
by
  sorry

end cards_ratio_l746_74654


namespace base6_arithmetic_l746_74647

theorem base6_arithmetic :
  let a := 4512
  let b := 2324
  let c := 1432
  let base := 6
  let a_b10 := 4 * base^3 + 5 * base^2 + 1 * base + 2
  let b_b10 := 2 * base^3 + 3 * base^2 + 2 * base + 4
  let c_b10 := 1 * base^3 + 4 * base^2 + 3 * base + 2
  let result_b10 := a_b10 - b_b10 + c_b10
  let result_base6 := 4020
  (result_b10 / base^3) % base = 4 ∧
  (result_b10 / base^2) % base = 0 ∧
  (result_b10 / base) % base = 2 ∧
  result_b10 % base = 0 →
  result_base6 = 4020 := by
  sorry

end base6_arithmetic_l746_74647


namespace values_of_a_and_b_range_of_c_isosceles_perimeter_l746_74628

def a : ℝ := 3
def b : ℝ := 4

axiom triangle_ABC (c : ℝ) : 0 < c

noncomputable def equation_condition (a b : ℝ) : Prop :=
  |a-3| + (b-4)^2 = 0

noncomputable def is_valid_c (c : ℝ) : Prop :=
  1 < c ∧ c < 7

theorem values_of_a_and_b (h : equation_condition a b) : a = 3 ∧ b = 4 := sorry

theorem range_of_c (h : equation_condition a b) : is_valid_c c := sorry

noncomputable def isosceles_triangle (c : ℝ) : Prop :=
  c = 4 ∨ c = 3

theorem isosceles_perimeter (h : equation_condition a b) (hc : isosceles_triangle c) : (3 + 3 + 4 = 10) ∨ (4 + 4 + 3 = 11) := sorry

end values_of_a_and_b_range_of_c_isosceles_perimeter_l746_74628


namespace total_students_calculation_l746_74600

variable (x : ℕ)
variable (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ)
variable (total_students : ℕ)
variable (remaining_jelly_beans : ℕ)

-- Defining the number of boys as per the problem's conditions
def boys (x : ℕ) : ℕ := 2 * x + 3

-- Defining the jelly beans given to girls
def jelly_beans_given_to_girls (x girls_jelly_beans : ℕ) : Prop :=
  girls_jelly_beans = 2 * x * x

-- Defining the jelly beans given to boys
def jelly_beans_given_to_boys (x boys_jelly_beans : ℕ) : Prop :=
  boys_jelly_beans = 3 * (2 * x + 3) * (2 * x + 3)

-- Defining the total jelly beans given out
def total_jelly_beans_given_out (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ) : Prop :=
  total_jelly_beans = girls_jelly_beans + boys_jelly_beans

-- Defining the total number of students
def total_students_in_class (x total_students : ℕ) : Prop :=
  total_students = x + boys x

-- Proving that the total number of students is 18 under given conditions
theorem total_students_calculation (h1 : jelly_beans_given_to_girls x girls_jelly_beans)
                                   (h2 : jelly_beans_given_to_boys x boys_jelly_beans)
                                   (h3 : total_jelly_beans_given_out girls_jelly_beans boys_jelly_beans total_jelly_beans)
                                   (h4 : total_jelly_beans - remaining_jelly_beans = 642)
                                   (h5 : remaining_jelly_beans = 3) :
                                   total_students = 18 :=
by
  sorry

end total_students_calculation_l746_74600


namespace sum_two_smallest_prime_factors_l746_74666

theorem sum_two_smallest_prime_factors (n : ℕ) (h : n = 462) : 
  (2 + 3) = 5 := 
by {
  sorry
}

end sum_two_smallest_prime_factors_l746_74666


namespace num_dogs_l746_74696

-- Define the conditions
def total_animals := 11
def ducks := 6
def total_legs := 32
def legs_per_duck := 2
def legs_per_dog := 4

-- Calculate intermediate values based on conditions
def duck_legs := ducks * legs_per_duck
def remaining_legs := total_legs - duck_legs

-- The proof statement
theorem num_dogs : ∃ D : ℕ, D = remaining_legs / legs_per_dog ∧ D + ducks = total_animals :=
by
  sorry

end num_dogs_l746_74696


namespace triangle_inequality_l746_74626

theorem triangle_inequality (a b c : ℝ) (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l746_74626


namespace oz_lost_words_count_l746_74650
-- We import the necessary library.

-- Define the context.
def total_letters := 69
def forbidden_letter := 7

-- Define function to calculate lost words when a specific letter is forbidden.
def lost_words (total_letters : ℕ) (forbidden_letter : ℕ) : ℕ :=
  let one_letter_lost := 1
  let two_letter_lost := 2 * (total_letters - 1)
  one_letter_lost + two_letter_lost

-- State the theorem.
theorem oz_lost_words_count :
  lost_words total_letters forbidden_letter = 139 :=
by
  sorry

end oz_lost_words_count_l746_74650


namespace find_prime_pair_l746_74697
open Int

theorem find_prime_pair :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∃ (p : ℕ), Prime p ∧ p = a * b^2 / (a + b) ∧ (a, b) = (6, 2) := by
  sorry

end find_prime_pair_l746_74697


namespace total_boxes_packed_l746_74619

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end total_boxes_packed_l746_74619


namespace periodic_functions_exist_l746_74653

theorem periodic_functions_exist (p1 p2 : ℝ) (h1 : p1 > 0) (h2 : p2 > 0) :
    ∃ (f1 f2 : ℝ → ℝ), (∀ x, f1 (x + p1) = f1 x) ∧ (∀ x, f2 (x + p2) = f2 x) ∧ ∃ T > 0, ∀ x, (f1 - f2) (x + T) = (f1 - f2) x :=
sorry

end periodic_functions_exist_l746_74653


namespace total_cost_750_candies_l746_74649

def candy_cost (candies : ℕ) (cost_per_box : ℕ) (candies_per_box : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let boxes := candies / candies_per_box
  let total_cost := boxes * cost_per_box
  if candies > discount_threshold then
    (1 - discount_rate) * total_cost
  else
    total_cost

theorem total_cost_750_candies :
  candy_cost 750 8 30 500 0.1 = 180 :=
by sorry

end total_cost_750_candies_l746_74649


namespace range_of_m_l746_74679

noncomputable def f (x m a : ℝ) : ℝ := Real.exp (x + 1) - m * a
noncomputable def g (x a : ℝ) : ℝ := a * Real.exp x - x

theorem range_of_m (h : ∃ a : ℝ, ∀ x : ℝ, f x m a ≤ g x a) : m ≥ -1 / Real.exp 1 :=
by
  sorry

end range_of_m_l746_74679


namespace rice_mixture_ratio_l746_74651

-- Definitions for the given conditions
def cost_per_kg_rice1 : ℝ := 5
def cost_per_kg_rice2 : ℝ := 8.75
def cost_per_kg_mixture : ℝ := 7.50

-- The problem: ratio of two quantities
theorem rice_mixture_ratio (x y : ℝ) (h : cost_per_kg_rice1 * x + cost_per_kg_rice2 * y = 
                                     cost_per_kg_mixture * (x + y)) :
  y / x = 2 := 
sorry

end rice_mixture_ratio_l746_74651


namespace weight_around_59_3_l746_74670

noncomputable def weight_at_height (height: ℝ) : ℝ := 0.75 * height - 68.2

theorem weight_around_59_3 (x : ℝ) (h : x = 170) : abs (weight_at_height x - 59.3) < 1 :=
by
  sorry

end weight_around_59_3_l746_74670


namespace category_D_cost_after_discount_is_correct_l746_74652

noncomputable def total_cost : ℝ := 2500
noncomputable def percentage_A : ℝ := 0.30
noncomputable def percentage_B : ℝ := 0.25
noncomputable def percentage_C : ℝ := 0.20
noncomputable def percentage_D : ℝ := 0.25
noncomputable def discount_A : ℝ := 0.03
noncomputable def discount_B : ℝ := 0.05
noncomputable def discount_C : ℝ := 0.07
noncomputable def discount_D : ℝ := 0.10

noncomputable def cost_before_discount_D : ℝ := total_cost * percentage_D
noncomputable def discount_amount_D : ℝ := cost_before_discount_D * discount_D
noncomputable def cost_after_discount_D : ℝ := cost_before_discount_D - discount_amount_D

theorem category_D_cost_after_discount_is_correct : cost_after_discount_D = 562.5 := 
by 
  sorry

end category_D_cost_after_discount_is_correct_l746_74652


namespace arithmetic_sequence_first_term_and_common_difference_l746_74637

def a_n (n : ℕ) : ℕ := 2 * n + 5

theorem arithmetic_sequence_first_term_and_common_difference :
  a_n 1 = 7 ∧ ∀ n : ℕ, a_n (n + 1) - a_n n = 2 := by
  sorry

end arithmetic_sequence_first_term_and_common_difference_l746_74637


namespace find_skirts_l746_74659

variable (blouses : ℕ) (skirts : ℕ) (slacks : ℕ)
variable (blouses_in_hamper : ℕ) (slacks_in_hamper : ℕ) (skirts_in_hamper : ℕ)
variable (clothes_in_hamper : ℕ)

-- Given conditions
axiom h1 : blouses = 12
axiom h2 : slacks = 8
axiom h3 : blouses_in_hamper = (75 * blouses) / 100
axiom h4 : slacks_in_hamper = (25 * slacks) / 100
axiom h5 : skirts_in_hamper = 3
axiom h6 : clothes_in_hamper = blouses_in_hamper + slacks_in_hamper + skirts_in_hamper
axiom h7 : clothes_in_hamper = 11

-- Proof goal: proving the total number of skirts
theorem find_skirts : skirts_in_hamper = (50 * skirts) / 100 → skirts = 6 :=
by sorry

end find_skirts_l746_74659


namespace negation_of_p_l746_74611

open Real

def p : Prop := ∃ x : ℝ, sin x < (1 / 2) * x

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, sin x ≥ (1 / 2) * x := 
by
  sorry

end negation_of_p_l746_74611


namespace obtuse_angle_only_dihedral_planar_l746_74688

/-- Given the range of three types of angles, prove that only the dihedral angle's planar angle can be obtuse. -/
theorem obtuse_angle_only_dihedral_planar 
  (α : ℝ) (β : ℝ) (γ : ℝ) 
  (hα : 0 < α ∧ α ≤ 90)
  (hβ : 0 ≤ β ∧ β ≤ 90)
  (hγ : 0 ≤ γ ∧ γ < 180) : 
  (90 < γ ∧ (¬(90 < α)) ∧ (¬(90 < β))) :=
by 
  sorry

end obtuse_angle_only_dihedral_planar_l746_74688


namespace glenda_speed_is_8_l746_74604

noncomputable def GlendaSpeed : ℝ :=
  let AnnSpeed := 6
  let Hours := 3
  let Distance := 42
  let AnnDistance := AnnSpeed * Hours
  let GlendaDistance := Distance - AnnDistance
  GlendaDistance / Hours

theorem glenda_speed_is_8 : GlendaSpeed = 8 := by
  sorry

end glenda_speed_is_8_l746_74604


namespace log_inequality_l746_74671

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem log_inequality :
  a > b ∧ b > c :=
by
  sorry

end log_inequality_l746_74671


namespace keaton_apple_earnings_l746_74641

theorem keaton_apple_earnings
  (orange_harvest_interval : ℕ)
  (orange_income_per_harvest : ℕ)
  (total_yearly_income : ℕ)
  (orange_harvests_per_year : ℕ)
  (orange_yearly_income : ℕ)
  (apple_yearly_income : ℕ) :
  orange_harvest_interval = 2 →
  orange_income_per_harvest = 50 →
  total_yearly_income = 420 →
  orange_harvests_per_year = 12 / orange_harvest_interval →
  orange_yearly_income = orange_harvests_per_year * orange_income_per_harvest →
  apple_yearly_income = total_yearly_income - orange_yearly_income →
  apple_yearly_income = 120 :=
by
  sorry

end keaton_apple_earnings_l746_74641


namespace sin_monotonically_decreasing_l746_74635

open Real

theorem sin_monotonically_decreasing (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = sin (2 * x + π / 3)) →
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, (π / 12) ≤ x ∧ x ≤ (7 * π / 12)) →
  ∀ x y, (x < y → f y ≤ f x) := by
  sorry

end sin_monotonically_decreasing_l746_74635


namespace robe_initial_savings_l746_74695

noncomputable def initial_savings (repair_fee corner_light_cost brake_disk_cost tires_cost remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost + tires_cost

theorem robe_initial_savings :
  let R := 10
  let corner_light := 2 * R
  let brake_disk := 3 * corner_light
  let tires := corner_light + 2 * brake_disk
  let remaining := 480
  initial_savings R corner_light brake_disk tires remaining = 770 :=
by
  sorry

end robe_initial_savings_l746_74695


namespace sum_is_45_l746_74665

noncomputable def sum_of_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_is_45 {a b c : ℝ} (h1 : ∃ a b c, (a ≤ b ∧ b ≤ c) ∧ b = 10)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 25) :
  sum_of_numbers a b c = 45 := 
sorry

end sum_is_45_l746_74665


namespace find_x_when_y_64_l746_74676

theorem find_x_when_y_64 (x y k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_inv_prop : x^3 * y = k) (h_given : x = 2 ∧ y = 8 ∧ k = 64) :
  y = 64 → x = 1 :=
by
  sorry

end find_x_when_y_64_l746_74676


namespace G_is_odd_l746_74633

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_odd (F : ℝ → ℝ) (a : ℝ) (h : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x : ℝ, F (-x) = - F x) :
  ∀ x : ℝ, G F a (-x) = - G F a x :=
by 
  sorry

end G_is_odd_l746_74633


namespace a_is_perfect_square_l746_74640

theorem a_is_perfect_square {a : ℕ} (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ d ∣ n ^ 2 * a - 1) : ∃ k : ℕ, a = k ^ 2 :=
by
  sorry

end a_is_perfect_square_l746_74640


namespace hair_cut_amount_l746_74642

theorem hair_cut_amount (initial_length final_length cut_length : ℕ) (h1 : initial_length = 11) (h2 : final_length = 7) : cut_length = 4 :=
by 
  sorry

end hair_cut_amount_l746_74642


namespace inequality_proof_l746_74685

noncomputable def a := Real.log 1 / Real.log 3
noncomputable def b := Real.log 1 / Real.log (1 / 2)
noncomputable def c := (1/2)^(1/3)

theorem inequality_proof : b > c ∧ c > a := 
by 
  sorry

end inequality_proof_l746_74685


namespace num_false_statements_is_three_l746_74610

-- Definitions of the statements on the card
def s1 : Prop := ∀ (false_statements : ℕ), false_statements = 1
def s2 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 + false_statements_card2 = 2
def s3 : Prop := ∀ (false_statements : ℕ), false_statements = 3
def s4 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 = false_statements_card2

-- Main proof problem: The number of false statements on this card is 3
theorem num_false_statements_is_three 
  (h_s1 : ¬ s1)
  (h_s2 : ¬ s2)
  (h_s3 : s3)
  (h_s4 : ¬ s4) :
  ∃ (n : ℕ), n = 3 :=
by
  sorry

end num_false_statements_is_three_l746_74610


namespace percentage_spent_l746_74674

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h_initial : initial_amount = 1200) 
  (h_remaining : remaining_amount = 840) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 :=
by
  sorry

end percentage_spent_l746_74674


namespace area_enclosed_by_region_l746_74603

theorem area_enclosed_by_region :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 6*y - 3 = 0) → 
  (∃ r : ℝ, r = 4 ∧ area = (π * r^2)) :=
by
  -- Starting proof setup
  sorry

end area_enclosed_by_region_l746_74603


namespace focus_of_parabola_tangent_to_circle_directrix_l746_74660

theorem focus_of_parabola_tangent_to_circle_directrix :
  ∃ p : ℝ, p > 0 ∧
  (∃ (x y : ℝ), x ^ 2 + y ^ 2 - 6 * x - 7 = 0 ∧
  ∀ x y : ℝ, y ^ 2 = 2 * p * x → x = -p) →
  (1, 0) = (p, 0) :=
by
  sorry

end focus_of_parabola_tangent_to_circle_directrix_l746_74660


namespace associate_professor_pencils_l746_74634

theorem associate_professor_pencils
  (A B P : ℕ)
  (h1 : A + B = 7)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 11) :
  P = 2 :=
by {
  -- Variables declarations and assumptions
  -- Combine and manipulate equations to prove P = 2
  sorry
}

end associate_professor_pencils_l746_74634


namespace number_of_ways_to_divide_l746_74643

def shape_17_cells : Type := sorry -- We would define the structure of the shape here
def checkerboard_pattern : shape_17_cells → Prop := sorry -- The checkerboard pattern condition
def num_black_cells (s : shape_17_cells) : ℕ := 9 -- Number of black cells
def num_gray_cells (s : shape_17_cells) : ℕ := 8 -- Number of gray cells
def divides_into (s : shape_17_cells) (rectangles : ℕ) (squares : ℕ) : Prop := sorry -- Division condition

theorem number_of_ways_to_divide (s : shape_17_cells) (h1 : checkerboard_pattern s) (h2 : divides_into s 8 1) :
  num_black_cells s = 9 ∧ num_gray_cells s = 8 → 
  (∃ ways : ℕ, ways = 10) := 
sorry

end number_of_ways_to_divide_l746_74643


namespace enlarged_banner_height_l746_74686

-- Definitions and theorem statement
theorem enlarged_banner_height 
  (original_width : ℝ) 
  (original_height : ℝ) 
  (new_width : ℝ) 
  (scaling_factor : ℝ := new_width / original_width ) 
  (new_height : ℝ := original_height * scaling_factor) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 15): 
  new_height = 10 := 
by 
  -- The proof would go here
  sorry

end enlarged_banner_height_l746_74686


namespace base9_4318_is_base10_3176_l746_74612

def base9_to_base10 (n : Nat) : Nat :=
  let d₀ := (n % 10) * 9^0
  let d₁ := ((n / 10) % 10) * 9^1
  let d₂ := ((n / 100) % 10) * 9^2
  let d₃ := ((n / 1000) % 10) * 9^3
  d₀ + d₁ + d₂ + d₃

theorem base9_4318_is_base10_3176 :
  base9_to_base10 4318 = 3176 :=
by
  sorry

end base9_4318_is_base10_3176_l746_74612


namespace red_cars_in_lot_l746_74625

theorem red_cars_in_lot (B : ℕ) (hB : B = 90) (ratio_condition : 3 * B = 8 * R) : R = 33 :=
by
  -- Given
  have h1 : B = 90 := hB
  have h2 : 3 * B = 8 * R := ratio_condition

  -- To solve
  sorry

end red_cars_in_lot_l746_74625


namespace prove_sets_l746_74664

noncomputable def A := { y : ℝ | ∃ x : ℝ, y = 3^x }
def B := { x : ℝ | x^2 - 4 ≤ 0 }

theorem prove_sets :
  A ∪ B = { x : ℝ | x ≥ -2 } ∧ A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by {
  sorry
}

end prove_sets_l746_74664


namespace cosine_inequality_l746_74617

theorem cosine_inequality (a b c : ℝ) : ∃ x : ℝ, 
    a * Real.cos x + b * Real.cos (3 * x) + c * Real.cos (9 * x) ≥ (|a| + |b| + |c|) / 2 :=
sorry

end cosine_inequality_l746_74617


namespace total_memory_space_l746_74684

def morning_songs : Nat := 10
def afternoon_songs : Nat := 15
def night_songs : Nat := 3
def song_size : Nat := 5

theorem total_memory_space : (morning_songs + afternoon_songs + night_songs) * song_size = 140 := by
  sorry

end total_memory_space_l746_74684


namespace product_of_primes_is_582_l746_74602

-- Define the relevant primes based on the conditions.
def smallest_one_digit_prime_1 := 2
def smallest_one_digit_prime_2 := 3
def largest_two_digit_prime := 97

-- Define the product of these primes as stated in the problem.
def product_of_primes := smallest_one_digit_prime_1 * smallest_one_digit_prime_2 * largest_two_digit_prime

-- Prove that this product equals to 582.
theorem product_of_primes_is_582 : product_of_primes = 582 :=
by {
  sorry
}

end product_of_primes_is_582_l746_74602


namespace distinguishable_arrangements_l746_74627

theorem distinguishable_arrangements :
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  (Nat.factorial total) / (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow * Nat.factorial blue) = 50400 := 
by
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  sorry

end distinguishable_arrangements_l746_74627


namespace number_of_triangles_with_perimeter_nine_l746_74672

theorem number_of_triangles_with_perimeter_nine : 
  ∃ (a b c : ℕ), a + b + c = 9 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry  -- Proof steps are omitted.

end number_of_triangles_with_perimeter_nine_l746_74672


namespace inequality_proof_l746_74687

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : a > b) :
  a * c^2 ≥ b * c^2 := 
sorry

end inequality_proof_l746_74687


namespace smallest_integer_representation_l746_74621

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l746_74621


namespace inequality_holds_l746_74694

noncomputable def positive_real_numbers := { x : ℝ // 0 < x }

theorem inequality_holds (a b c : positive_real_numbers) (h : (a.val * b.val + b.val * c.val + c.val * a.val) = 1) :
    (a.val / b.val + b.val / c.val + c.val / a.val) ≥ (a.val^2 + b.val^2 + c.val^2 + 2) :=
by
  sorry

end inequality_holds_l746_74694


namespace product_xyz_l746_74644

/-- Prove that if x + 1/y = 2 and y + 1/z = 3, then xyz = 1/11. -/
theorem product_xyz {x y z : ℝ} (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 3) : x * y * z = 1 / 11 :=
sorry

end product_xyz_l746_74644


namespace solution_cos_eq_l746_74699

open Real

theorem solution_cos_eq (x : ℝ) :
  (cos x)^2 + (cos (2 * x))^2 + (cos (3 * x))^2 = 1 ↔
  (∃ k : ℤ, x = k * π / 2 + π / 4) ∨ (∃ k : ℤ, x = k * π / 3 + π / 6) :=
by sorry

end solution_cos_eq_l746_74699


namespace expand_and_simplify_l746_74636

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l746_74636


namespace largest_of_A_B_C_l746_74646

noncomputable def A : ℝ := (3003 / 3002) + (3003 / 3004)
noncomputable def B : ℝ := (3003 / 3004) + (3005 / 3004)
noncomputable def C : ℝ := (3004 / 3003) + (3004 / 3005)

theorem largest_of_A_B_C : A > B ∧ A ≥ C := by
  sorry

end largest_of_A_B_C_l746_74646


namespace unique_triple_sum_l746_74616

theorem unique_triple_sum :
  ∃ (a b c : ℕ), 
    (10 ≤ a ∧ a < 100) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (10 ≤ c ∧ c < 100) ∧ 
    (a^3 + 3 * b^3 + 9 * c^3 = 9 * a * b * c + 1) ∧ 
    (a + b + c = 9) := 
sorry

end unique_triple_sum_l746_74616
