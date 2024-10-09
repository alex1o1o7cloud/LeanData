import Mathlib

namespace simplify_expression_l1321_132155

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = 3) :
  (x + 2 * y)^2 - (x + y) * (2 * x - y) = 23 :=
by
  sorry

end simplify_expression_l1321_132155


namespace ab_value_l1321_132178

theorem ab_value (a b : ℚ) (h1 : 3 * a - 8 = 0) (h2 : b = 3) : a * b = 8 :=
by
  sorry

end ab_value_l1321_132178


namespace sum_series_eq_4_div_9_l1321_132148

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l1321_132148


namespace original_number_div_eq_l1321_132196

theorem original_number_div_eq (h : 204 / 12.75 = 16) : 2.04 / 1.6 = 1.275 :=
by sorry

end original_number_div_eq_l1321_132196


namespace sum_of_inserted_numbers_l1321_132121

theorem sum_of_inserted_numbers (x y : ℝ) (r : ℝ) 
  (h1 : 4 * r = x) 
  (h2 : 4 * r^2 = y) 
  (h3 : (2 / y) = ((1 / x) + (1 / 16))) :
  x + y = 8 :=
sorry

end sum_of_inserted_numbers_l1321_132121


namespace coo_coo_count_correct_l1321_132159

theorem coo_coo_count_correct :
  let monday_coos := 89
  let tuesday_coos := 179
  let wednesday_coos := 21
  let total_coos := monday_coos + tuesday_coos + wednesday_coos
  total_coos = 289 :=
by
  sorry

end coo_coo_count_correct_l1321_132159


namespace num_isosceles_triangles_l1321_132197

theorem num_isosceles_triangles (a b : ℕ) (h1 : 2 * a + b = 27) (h2 : a > b / 2) : 
  ∃! (n : ℕ), n = 13 :=
by 
  sorry

end num_isosceles_triangles_l1321_132197


namespace correct_operation_is_C_l1321_132104

/--
Given the following statements:
1. \( a^3 \cdot a^2 = a^6 \)
2. \( (2a^3)^3 = 6a^9 \)
3. \( -6x^5 \div 2x^3 = -3x^2 \)
4. \( (-x-2)(x-2) = x^2 - 4 \)

Prove that the correct statement is \( -6x^5 \div 2x^3 = -3x^2 \) and the other statements are incorrect.
-/
theorem correct_operation_is_C (a x : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧
  ((2 * a^3)^3 ≠ 6 * a^9) ∧
  (-6 * x^5 / (2 * x^3) = -3 * x^2) ∧
  ((-x - 2) * (x - 2) ≠ x^2 - 4) := by
  sorry

end correct_operation_is_C_l1321_132104


namespace solution_is_correct_l1321_132152

noncomputable def solve_system_of_inequalities : Prop :=
  ∃ x y : ℝ, 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧ 
    (x = -1/3) ∧ 
    (y = 2/3)

theorem solution_is_correct : solve_system_of_inequalities :=
sorry

end solution_is_correct_l1321_132152


namespace range_of_a_l1321_132181

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h_cond : ∀ (n : ℕ), n > 0 → (a_seq n = if n ≤ 4 then 2^n - 1 else -n^2 + (a - 1) * n))
  (h_max_a5 : ∀ (n : ℕ), n > 0 → a_seq n ≤ a_seq 5) :
  9 ≤ a ∧ a ≤ 12 := 
by
  sorry

end range_of_a_l1321_132181


namespace age_difference_between_brother_and_cousin_is_five_l1321_132173

variable (Lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ)

-- Conditions
axiom lexie_age_def : Lexie_age = 8
axiom grandma_age_def : grandma_age = 68
axiom lexie_brother_condition : Lexie_age = brother_age + 6
axiom lexie_sister_condition : sister_age = 2 * Lexie_age
axiom uncle_grandma_condition : uncle_age = grandma_age - 12
axiom cousin_brother_condition : cousin_age = brother_age + 5

-- Goal
theorem age_difference_between_brother_and_cousin_is_five : 
  Lexie_age = 8 → grandma_age = 68 → brother_age = Lexie_age - 6 → cousin_age = brother_age + 5 → cousin_age - brother_age = 5 :=
by sorry

end age_difference_between_brother_and_cousin_is_five_l1321_132173


namespace sandwich_and_soda_cost_l1321_132198

theorem sandwich_and_soda_cost:
  let sandwich_cost := 4
  let soda_cost := 1
  let num_sandwiches := 6
  let num_sodas := 10
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  total_cost = 34 := 
by 
  sorry

end sandwich_and_soda_cost_l1321_132198


namespace max_teams_participation_l1321_132146

theorem max_teams_participation (n : ℕ) (H : 9 * n * (n - 1) / 2 ≤ 200) : n ≤ 7 := by
  -- Proof to be filled in
  sorry

end max_teams_participation_l1321_132146


namespace repeating_decimal_to_fraction_l1321_132180

theorem repeating_decimal_to_fraction :
  (2 + (35 / 99 : ℚ)) = (233 / 99) := 
sorry

end repeating_decimal_to_fraction_l1321_132180


namespace sum_of_money_is_6000_l1321_132183

noncomputable def original_interest (P R : ℝ) := (P * R * 3) / 100
noncomputable def new_interest (P R : ℝ) := (P * (R + 2) * 3) / 100

theorem sum_of_money_is_6000 (P R : ℝ) (h : new_interest P R - original_interest P R = 360) : P = 6000 :=
by
  sorry

end sum_of_money_is_6000_l1321_132183


namespace probability_of_two_points_is_three_sevenths_l1321_132154

/-- Define the problem's conditions and statement. -/
def num_choices (n : ℕ) : ℕ :=
  match n with
  | 1 => 4  -- choose 1 option from 4
  | 2 => 6  -- choose 2 options from 4 (binomial coefficient)
  | 3 => 4  -- choose 3 options from 4 (binomial coefficient)
  | _ => 0

def total_ways : ℕ := 14  -- Total combinations of choosing 1 to 3 options from 4

def two_points_ways : ℕ := 6  -- 3 ways for 1 correct, 3 ways for 2 correct (B, C, D combinations)

def probability_two_points : ℚ :=
  (two_points_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_two_points_is_three_sevenths :
  probability_two_points = (3 / 7 : ℚ) :=
sorry

end probability_of_two_points_is_three_sevenths_l1321_132154


namespace exists_close_ratios_l1321_132176

theorem exists_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  abs ((a - b) / (c - d) - 1) < 1 / 100000 :=
sorry

end exists_close_ratios_l1321_132176


namespace paint_left_l1321_132136

-- Define the conditions
def total_paint_needed : ℕ := 333
def paint_needed_to_buy : ℕ := 176

-- State the theorem
theorem paint_left : total_paint_needed - paint_needed_to_buy = 157 := 
by 
  sorry

end paint_left_l1321_132136


namespace find_angle_A_find_range_expression_l1321_132171

-- Define the variables and conditions in a way consistent with Lean's syntax
variables {α β γ : Type}
variables (a b c : ℝ) (A B C : ℝ)

-- The mathematical conditions translated to Lean
def triangle_condition (a b c A B C : ℝ) : Prop := (b + c) / a = Real.cos B + Real.cos C

-- Statement for Proof 1: Prove that A = π/2 given the conditions
theorem find_angle_A (h : triangle_condition a b c A B C) : A = Real.pi / 2 :=
sorry

-- Statement for Proof 2: Prove the range of the given expression under the given conditions
theorem find_range_expression (h : triangle_condition a b c A B C) (hA : A = Real.pi / 2) :
  ∃ (l u : ℝ), l = Real.sqrt 3 + 2 ∧ u = Real.sqrt 3 + 3 ∧ (2 * Real.cos (B / 2) ^ 2 + 2 * Real.sqrt 3 * Real.cos (C / 2) ^ 2) ∈ Set.Ioc l u :=
sorry

end find_angle_A_find_range_expression_l1321_132171


namespace task_assignment_l1321_132132

theorem task_assignment (volunteers : ℕ) (tasks : ℕ) (selected : ℕ) (h_volunteers : volunteers = 6) (h_tasks : tasks = 4) (h_selected : selected = 4) :
  ((Nat.factorial volunteers) / (Nat.factorial (volunteers - selected))) = 360 :=
by
  rw [h_volunteers, h_selected]
  norm_num
  sorry

end task_assignment_l1321_132132


namespace molecular_weight_of_7_moles_KBrO3_l1321_132149

def potassium_atomic_weight : ℝ := 39.10
def bromine_atomic_weight : ℝ := 79.90
def oxygen_atomic_weight : ℝ := 16.00
def oxygen_atoms_in_KBrO3 : ℝ := 3

def KBrO3_molecular_weight : ℝ := 
  potassium_atomic_weight + bromine_atomic_weight + (oxygen_atomic_weight * oxygen_atoms_in_KBrO3)

def moles := 7

theorem molecular_weight_of_7_moles_KBrO3 : KBrO3_molecular_weight * moles = 1169.00 := 
by {
  -- The proof would be here, but it is omitted as instructed.
  sorry
}

end molecular_weight_of_7_moles_KBrO3_l1321_132149


namespace find_A_l1321_132167

def U : Set ℕ := {1, 2, 3, 4, 5}

def compl_U (A : Set ℕ) : Set ℕ := U \ A

theorem find_A (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (h_compl_U : compl_U A = {2, 3}) : A = {1, 4, 5} :=
by
  sorry

end find_A_l1321_132167


namespace length_of_BC_l1321_132175

theorem length_of_BC (x : ℝ) (h1 : (20 * x^2) / 3 - (400 * x) / 3 = 140) :
  ∃ (BC : ℝ), BC = 29 := 
by
  sorry

end length_of_BC_l1321_132175


namespace factor_expr_l1321_132141

variable (x : ℝ)

def expr : ℝ := (20 * x ^ 3 + 100 * x - 10) - (-5 * x ^ 3 + 5 * x - 10)

theorem factor_expr :
  expr x = 5 * x * (5 * x ^ 2 + 19) :=
by
  sorry

end factor_expr_l1321_132141


namespace remainder_polynomial_l1321_132162

theorem remainder_polynomial (x : ℤ) : (1 + x) ^ 2010 % (1 + x + x^2) = 1 := 
  sorry

end remainder_polynomial_l1321_132162


namespace maximize_profit_l1321_132140

-- Define the price of the book
variables (p : ℝ) (p_max : ℝ)
-- Define the revenue function
def R (p : ℝ) : ℝ := p * (150 - 4 * p)
-- Define the profit function accounting for fixed costs of $200
def P (p : ℝ) := R p - 200
-- Set the maximum feasible price
def max_price_condition := p_max = 30
-- Define the price that maximizes the profit
def optimal_price := 18.75

-- The theorem to be proved
theorem maximize_profit : p_max = 30 → p = 18.75 → P p = 2612.5 :=
by {
  sorry
}

end maximize_profit_l1321_132140


namespace change_in_expression_l1321_132193

variables (x b : ℝ) (hb : 0 < b)

theorem change_in_expression : (b * x)^2 - 5 - (x^2 - 5) = (b^2 - 1) * x^2 :=
by sorry

end change_in_expression_l1321_132193


namespace parallel_ne_implies_value_l1321_132100

theorem parallel_ne_implies_value 
  (x : ℝ) 
  (m : ℝ × ℝ := (2 * x, 7)) 
  (n : ℝ × ℝ := (6, x + 4)) 
  (h1 : 2 * x * (x + 4) = 42) 
  (h2 : m ≠ n) :
  x = -7 :=
by {
  sorry
}

end parallel_ne_implies_value_l1321_132100


namespace manager_salary_is_correct_l1321_132163

noncomputable def manager_salary (avg_salary_50_employees : ℝ) (increase_in_avg : ℝ) : ℝ :=
  let total_salary_50_employees := 50 * avg_salary_50_employees
  let new_avg_salary := avg_salary_50_employees + increase_in_avg
  let total_salary_51_people := 51 * new_avg_salary
  let manager_salary := total_salary_51_people - total_salary_50_employees
  manager_salary

theorem manager_salary_is_correct :
  manager_salary 2500 1500 = 79000 :=
by
  sorry

end manager_salary_is_correct_l1321_132163


namespace part1_part2_l1321_132186

open Real

noncomputable def part1_statement (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0

noncomputable def part2_statement (x : ℝ) : Prop := 
  ∀ (m : ℝ), |m| ≤ 1 → (m * x^2 - 2 * m * x - 1 < 0)

theorem part1 : part1_statement m ↔ (-1 < m ∧ m ≤ 0) :=
sorry

theorem part2 : part2_statement x ↔ ((1 - sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + sqrt 2)) :=
sorry

end part1_part2_l1321_132186


namespace line_through_center_eq_line_chord_len_eq_l1321_132169

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

noncomputable def point_P : ℝ × ℝ := (2, 2)

def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

def line_chord_len (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0 ∨ x = 2

theorem line_through_center_eq (x y : ℝ) (hC : circle_eq x y) :
  line_through_center x y :=
sorry

theorem line_chord_len_eq (x y : ℝ) (hC : circle_eq x y) (hP : x = 2 ∧ y = 2 ∧ (line_through_center x y)) :
  line_chord_len x y :=
sorry

end line_through_center_eq_line_chord_len_eq_l1321_132169


namespace smaller_group_men_l1321_132107

-- Define the main conditions of the problem
def men_work_days : ℕ := 36 * 18  -- 36 men for 18 days

-- Define the theorem we need to prove
theorem smaller_group_men (M : ℕ) (h: M * 72 = men_work_days) : M = 9 :=
by
  -- proof is not required
  sorry

end smaller_group_men_l1321_132107


namespace max_value_of_function_is_seven_l1321_132110

theorem max_value_of_function_is_seven:
  ∃ a: ℕ, (0 < a) ∧ 
  (∃ x: ℝ, (x + Real.sqrt (13 - 2 * a * x)) = 7 ∧
    ∀ y: ℝ, (y = x + Real.sqrt (13 - 2 * a * x)) → y ≤ 7) :=
sorry

end max_value_of_function_is_seven_l1321_132110


namespace intersection_P_complement_Q_l1321_132128

-- Defining the sets P and Q
def R := Set ℝ
def P : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def Q : Set ℝ := {x | Real.log x < 1}
def complement_R_Q : Set ℝ := {x | x ≤ 0 ∨ x ≥ Real.exp 1}
def intersection := {x | x ∈ P ∧ x ∈ complement_R_Q}

-- Statement of the theorem
theorem intersection_P_complement_Q : 
  intersection = {-3} :=
by
  sorry

end intersection_P_complement_Q_l1321_132128


namespace pete_miles_walked_l1321_132158

-- Define the conditions
def maxSteps := 99999
def numFlips := 50
def finalReading := 25000
def stepsPerMile := 1500

-- Proof statement that Pete walked 3350 miles
theorem pete_miles_walked : 
  (numFlips * (maxSteps + 1) + finalReading) / stepsPerMile = 3350 := 
by 
  sorry

end pete_miles_walked_l1321_132158


namespace number_of_labelings_l1321_132130

-- Define the concept of a truncated chessboard with 8 squares
structure TruncatedChessboard :=
(square_labels : Fin 8 → ℕ)
(condition : ∀ i j, i ≠ j → square_labels i ≠ square_labels j)

-- Assuming a wider adjacency matrix for "connected" (has at least one common vertex)
def connected (i j : Fin 8) : Prop := sorry

-- Define the non-consecutiveness condition
def non_consecutive (board : TruncatedChessboard) :=
  ∀ i j, connected i j → (board.square_labels i ≠ board.square_labels j + 1 ∧
                          board.square_labels i ≠ board.square_labels j - 1)

-- Theorem statement
theorem number_of_labelings : ∃ c : Fin 8 → ℕ, ∀ b : TruncatedChessboard, non_consecutive b → 
  (b.square_labels = c) := sorry

end number_of_labelings_l1321_132130


namespace smallest_b_for_undefined_inverse_mod_70_77_l1321_132103

theorem smallest_b_for_undefined_inverse_mod_70_77 (b : ℕ) :
  (∀ k, k < b → k * 1 % 70 ≠ 1 ∧ k * 1 % 77 ≠ 1) ∧ (b * 1 % 70 ≠ 1) ∧ (b * 1 % 77 ≠ 1) → b = 7 :=
by sorry

end smallest_b_for_undefined_inverse_mod_70_77_l1321_132103


namespace number_of_valid_ns_l1321_132153

theorem number_of_valid_ns :
  ∃ (n : ℝ), (n = 8 ∨ n = 1/2) ∧ ∀ n₁ n₂, (n₁ = 8 ∨ n₁ = 1/2) ∧ (n₂ = 8 ∨ n₂ = 1/2) → n₁ = n₂ :=
sorry

end number_of_valid_ns_l1321_132153


namespace only_book_A_l1321_132184

theorem only_book_A (purchasedBoth : ℕ) (purchasedOnlyB : ℕ) (purchasedA : ℕ) (purchasedB : ℕ) 
  (h1 : purchasedBoth = 500)
  (h2 : 2 * purchasedOnlyB = purchasedBoth)
  (h3 : purchasedA = 2 * purchasedB)
  (h4 : purchasedB = purchasedOnlyB + purchasedBoth) :
  purchasedA - purchasedBoth = 1000 :=
by
  sorry

end only_book_A_l1321_132184


namespace range_of_a_l1321_132143

open Set

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) :
  a ∈ Iic (-1 / 2) ∪ Ici 2 :=
by
  sorry

end range_of_a_l1321_132143


namespace true_proposition_l1321_132185

variable (p : Prop) (q : Prop)

-- Introduce the propositions as Lean variables
def prop_p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2
def prop_q : Prop := ∀ a b : ℝ, ((a > 1 ∧ b > 1) → a * b > 1) ∧ ((a * b > 1) ∧ (¬ (a > 1 ∧ b > 1)))

-- Rewrite the main goal as a Lean statement
theorem true_proposition : ¬ prop_p ∧ prop_q := 
  sorry

end true_proposition_l1321_132185


namespace largest_angle_of_triangle_ABC_l1321_132113

theorem largest_angle_of_triangle_ABC (a b c : ℝ)
  (h₁ : a + b + 2 * c = a^2) 
  (h₂ : a + b - 2 * c = -1) : 
  ∃ C : ℝ, C = 120 :=
sorry

end largest_angle_of_triangle_ABC_l1321_132113


namespace radar_coverage_correct_l1321_132145

noncomputable def radar_coverage (r : ℝ) (width : ℝ) : ℝ × ℝ :=
  let θ := Real.pi / 7
  let distance := 40 / Real.sin θ
  let area := 1440 * Real.pi / Real.tan θ
  (distance, area)

theorem radar_coverage_correct : radar_coverage 41 18 = 
  (40 / Real.sin (Real.pi / 7), 1440 * Real.pi / Real.tan (Real.pi / 7)) :=
by
  sorry

end radar_coverage_correct_l1321_132145


namespace sandy_final_position_and_distance_l1321_132168

-- Define the conditions as statements
def walked_south (distance : ℕ) := distance = 20
def turned_left_facing_east := true
def walked_east (distance : ℕ) := distance = 20
def turned_left_facing_north := true
def walked_north (distance : ℕ) := distance = 20
def turned_right_facing_east := true
def walked_east_again (distance : ℕ) := distance = 20

-- Final position computation as a proof statement
theorem sandy_final_position_and_distance :
  ∃ (d : ℕ) (dir : String), walked_south 20 → turned_left_facing_east → walked_east 20 →
  turned_left_facing_north → walked_north 20 →
  turned_right_facing_east → walked_east_again 20 ∧ d = 40 ∧ dir = "east" :=
by
  sorry

end sandy_final_position_and_distance_l1321_132168


namespace exists_X_Y_l1321_132166

theorem exists_X_Y {A n : ℤ} (h_coprime : Int.gcd A n = 1) :
  ∃ X Y : ℤ, |X| < Int.sqrt n ∧ |Y| < Int.sqrt n ∧ n ∣ (A * X - Y) :=
sorry

end exists_X_Y_l1321_132166


namespace base8_subtraction_l1321_132142

theorem base8_subtraction : (325 : Nat) - (237 : Nat) = 66 :=
by 
  sorry

end base8_subtraction_l1321_132142


namespace difference_of_profit_share_l1321_132147

theorem difference_of_profit_share (a b c : ℕ) (pa pb pc : ℕ) (profit_b : ℕ) 
  (a_capital : a = 8000) (b_capital : b = 10000) (c_capital : c = 12000) 
  (b_profit_share : profit_b = 1600)
  (investment_ratio : pa / 4 = pb / 5 ∧ pb / 5 = pc / 6) :
  pa - pc = 640 := 
sorry

end difference_of_profit_share_l1321_132147


namespace children_marbles_problem_l1321_132134

theorem children_marbles_problem (n x N : ℕ) 
  (h1 : N = n * x)
  (h2 : 1 + (N - 1) / 10 = x) :
  n = 9 ∧ x = 9 :=
by
  sorry

end children_marbles_problem_l1321_132134


namespace tangent_curve_line_l1321_132160

/-- Given the line y = x + 1 and the curve y = ln(x + a) are tangent, prove that the value of a is 2. -/
theorem tangent_curve_line (a : ℝ) :
  (∃ x₀ y₀, y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 / (x₀ + a) = 1)) → a = 2 :=
by
  sorry

end tangent_curve_line_l1321_132160


namespace general_term_sequence_l1321_132116

def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 3 ∧ a 1 = 9 ∧ ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2) - 4 * n + 2

theorem general_term_sequence (a : ℕ → ℤ) (h : seq a) : 
  ∀ n, a n = 3^n + n^2 + 3 * n + 2 :=
by
  sorry

end general_term_sequence_l1321_132116


namespace average_age_of_class_l1321_132189

theorem average_age_of_class 
  (avg_age_8 : ℕ → ℕ)
  (avg_age_6 : ℕ → ℕ)
  (age_15th : ℕ)
  (A : ℕ)
  (h1 : avg_age_8 8 = 112)
  (h2 : avg_age_6 6 = 96)
  (h3 : age_15th = 17)
  (h4 : 15 * A = (avg_age_8 8) + (avg_age_6 6) + age_15th)
  : A = 15 :=
by
  sorry

end average_age_of_class_l1321_132189


namespace exist_ordering_rectangles_l1321_132172

open Function

structure Rectangle :=
  (left_bot : ℝ × ℝ)  -- Bottom-left corner
  (right_top : ℝ × ℝ)  -- Top-right corner

def below (R1 R2 : Rectangle) : Prop :=
  ∃ g : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → y < g) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → y > g)

def to_right_of (R1 R2 : Rectangle) : Prop :=
  ∃ h : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → x > h) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → x < h)

def disjoint (R1 R2 : Rectangle) : Prop :=
  ¬ ((R1.left_bot.1 < R2.right_top.1) ∧ (R1.right_top.1 > R2.left_bot.1) ∧
     (R1.left_bot.2 < R2.right_top.2) ∧ (R1.right_top.2 > R2.left_bot.2))

theorem exist_ordering_rectangles (n : ℕ) (rectangles : Fin n → Rectangle)
  (h_disjoint : ∀ i j, i ≠ j → disjoint (rectangles i) (rectangles j)) :
  ∃ f : Fin n → Fin n, ∀ i j : Fin n, i < j → 
    (to_right_of (rectangles (f i)) (rectangles (f j)) ∨ 
    below (rectangles (f i)) (rectangles (f j))) := 
sorry

end exist_ordering_rectangles_l1321_132172


namespace divisible_sum_or_difference_l1321_132102

theorem divisible_sum_or_difference (a : Fin 52 → ℤ) :
  ∃ i j, (i ≠ j) ∧ (a i + a j) % 100 = 0 ∨ (a i - a j) % 100 = 0 :=
by
  sorry

end divisible_sum_or_difference_l1321_132102


namespace sum_of_two_numbers_l1321_132177

theorem sum_of_two_numbers (x y : ℝ) (h1 : 0.5 * x + 0.3333 * y = 11)
(h2 : max x y = y) (h3 : y = 15) : x + y = 27 :=
by
  -- Skip the proof and add sorry
  sorry

end sum_of_two_numbers_l1321_132177


namespace allie_carl_product_points_l1321_132123

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldr (λ x acc => g x + acc) 0

theorem allie_carl_product_points : (total_points allie_rolls) * (total_points carl_rolls) = 594 :=
  sorry

end allie_carl_product_points_l1321_132123


namespace simplify_fraction_l1321_132190

theorem simplify_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^(2*b) * b^a) / (b^(2*a) * a^b) = (a / b)^b := 
by sorry

end simplify_fraction_l1321_132190


namespace initial_ratio_of_milk_to_water_l1321_132191

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + 20 = 3 * W) (h2 : M + W = 40) :
  (M : ℚ) / W = 5 / 3 := by
sorry

end initial_ratio_of_milk_to_water_l1321_132191


namespace employee_pay_per_week_l1321_132125

theorem employee_pay_per_week (total_pay : ℝ) (ratio : ℝ) (pay_b : ℝ)
  (h1 : total_pay = 570)
  (h2 : ratio = 1.5)
  (h3 : total_pay = pay_b * (ratio + 1)) :
  pay_b = 228 :=
sorry

end employee_pay_per_week_l1321_132125


namespace percentage_wearing_blue_shirts_l1321_132164

theorem percentage_wearing_blue_shirts (total_students : ℕ) (red_percentage green_percentage : ℕ) 
  (other_students : ℕ) (H1 : total_students = 900) (H2 : red_percentage = 28) 
  (H3 : green_percentage = 10) (H4 : other_students = 162) : 
  (44 : ℕ) = 100 - (red_percentage + green_percentage + (other_students * 100 / total_students)) :=
by
  sorry

end percentage_wearing_blue_shirts_l1321_132164


namespace jose_initial_caps_l1321_132112

-- Definition of conditions and the problem
def jose_starting_caps : ℤ :=
  let final_caps := 9
  let caps_from_rebecca := 2
  final_caps - caps_from_rebecca

-- Lean theorem to state the required proof
theorem jose_initial_caps : jose_starting_caps = 7 := by
  -- skip proof
  sorry

end jose_initial_caps_l1321_132112


namespace maximize_revenue_l1321_132117

def revenue (p : ℝ) : ℝ := 150 * p - 4 * p^2

theorem maximize_revenue : 
  ∃ p, 0 ≤ p ∧ p ≤ 30 ∧ p = 18.75 ∧ (∀ q, 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue 18.75) :=
by
  sorry

end maximize_revenue_l1321_132117


namespace part1_solution_set_m1_part2_find_m_l1321_132161

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (m+1) * x^2 - m * x + m - 1

theorem part1_solution_set_m1 :
  { x : ℝ | f x 1 > 0 } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 0.5 } :=
by
  sorry

theorem part2_find_m :
  (∀ x : ℝ, f x m + 1 > 0 ↔ x > 1.5 ∧ x < 3) → m = -9/7 :=
by
  sorry

end part1_solution_set_m1_part2_find_m_l1321_132161


namespace range_of_a_l1321_132129

theorem range_of_a (a : ℝ) 
  (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) 
  (q : ∃ x : ℝ, x^2 - 4 * x + a ≤ 0) : 
  e ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l1321_132129


namespace polynomial_sum_of_squares_is_23456_l1321_132131

theorem polynomial_sum_of_squares_is_23456 (p q r s t u : ℤ) :
  (∀ x, 1728 * x ^ 3 + 64 = (p * x ^ 2 + q * x + r) * (s * x ^ 2 + t * x + u)) →
  p ^ 2 + q ^ 2 + r ^ 2 + s ^ 2 + t ^ 2 + u ^ 2 = 23456 :=
by
  sorry

end polynomial_sum_of_squares_is_23456_l1321_132131


namespace solution_sum_l1321_132179

theorem solution_sum (m n : ℝ) (h₀ : m ≠ 0) (h₁ : m^2 + m * n - m = 0) : m + n = 1 := 
by 
  sorry

end solution_sum_l1321_132179


namespace vector_properties_l1321_132106

/-- The vectors a, b, and c used in the problem. --/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-4, 2)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  ((∃ k : ℝ, b = k • a) ∧ (b.1 * c.1 + b.2 * c.2 = 0) ∧ (a.1*a.1 + a.2*a.2 = c.1*c.1 + c.2*c.2)) :=
  by sorry

end vector_properties_l1321_132106


namespace typist_current_salary_l1321_132144

def original_salary : ℝ := 4000.0000000000005
def increased_salary (os : ℝ) : ℝ := os + (os * 0.1)
def decreased_salary (is : ℝ) : ℝ := is - (is * 0.05)

theorem typist_current_salary : decreased_salary (increased_salary original_salary) = 4180 :=
by
  sorry

end typist_current_salary_l1321_132144


namespace total_boys_across_grades_is_692_l1321_132135

theorem total_boys_across_grades_is_692 (ga_girls gb_girls gc_girls : ℕ) (ga_boys : ℕ) :
  ga_girls = 256 →
  ga_girls = ga_boys + 52 →
  gb_girls = 360 →
  gb_boys = gb_girls - 40 →
  gc_girls = 168 →
  gc_girls = gc_boys →
  ga_boys + gb_boys + gc_boys = 692 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_boys_across_grades_is_692_l1321_132135


namespace x_pow_12_eq_one_l1321_132101

theorem x_pow_12_eq_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 :=
sorry

end x_pow_12_eq_one_l1321_132101


namespace dart_lands_in_center_hexagon_l1321_132126

noncomputable def area_regular_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

theorem dart_lands_in_center_hexagon {s : ℝ} (h : s > 0) :
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  (A_inner / A_outer) = 1 / 4 :=
by
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  sorry

end dart_lands_in_center_hexagon_l1321_132126


namespace suit_price_the_day_after_sale_l1321_132127

def originalPrice : ℕ := 300
def increaseRate : ℚ := 0.20
def couponDiscount : ℚ := 0.30
def additionalReduction : ℚ := 0.10

def increasedPrice := originalPrice * (1 + increaseRate)
def priceAfterCoupon := increasedPrice * (1 - couponDiscount)
def finalPrice := increasedPrice * (1 - additionalReduction)

theorem suit_price_the_day_after_sale 
  (op : ℕ := originalPrice) 
  (ir : ℚ := increaseRate) 
  (cd : ℚ := couponDiscount) 
  (ar : ℚ := additionalReduction) :
  finalPrice = 324 := 
sorry

end suit_price_the_day_after_sale_l1321_132127


namespace find_f_of_3_l1321_132170

theorem find_f_of_3 (a b c : ℝ) (f : ℝ → ℝ) (h1 : f 1 = 7) (h2 : f 2 = 12) (h3 : ∀ x, f x = ax + bx + c) : f 3 = 17 :=
by
  sorry

end find_f_of_3_l1321_132170


namespace product_b2_b7_l1321_132157

def is_increasing_arithmetic_sequence (bs : ℕ → ℤ) :=
  ∀ n m : ℕ, n < m → bs n < bs m

def arithmetic_sequence (bs : ℕ → ℤ) (d : ℤ) :=
  ∀ n : ℕ, bs (n + 1) - bs n = d

theorem product_b2_b7 (bs : ℕ → ℤ) (d : ℤ) (h_incr : is_increasing_arithmetic_sequence bs)
    (h_arith : arithmetic_sequence bs d)
    (h_prod : bs 4 * bs 5 = 10) :
    bs 2 * bs 7 = -224 ∨ bs 2 * bs 7 = -44 :=
by
  sorry

end product_b2_b7_l1321_132157


namespace find_number_l1321_132137

theorem find_number (x : ℤ) (h : 3 * x - 4 = 5) : x = 3 :=
sorry

end find_number_l1321_132137


namespace quadrilateral_area_l1321_132165

def diagonal : ℝ := 15
def offset1 : ℝ := 6
def offset2 : ℝ := 4

theorem quadrilateral_area :
  (1/2) * diagonal * (offset1 + offset2) = 75 :=
by 
  sorry

end quadrilateral_area_l1321_132165


namespace sum_of_ages_is_42_l1321_132182

-- Define the variables for present ages of the son (S) and the father (F)
variables (S F : ℕ)

-- Define the conditions:
-- 1. 6 years ago, the father's age was 4 times the son's age.
-- 2. After 6 years, the son's age will be 18 years.

def son_age_condition := S + 6 = 18
def father_age_6_years_ago_condition := F - 6 = 4 * (S - 6)

-- Theorem statement to prove:
theorem sum_of_ages_is_42 (S F : ℕ)
  (h1 : son_age_condition S)
  (h2 : father_age_6_years_ago_condition F S) :
  S + F = 42 :=
sorry

end sum_of_ages_is_42_l1321_132182


namespace am_gm_equality_l1321_132195

theorem am_gm_equality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end am_gm_equality_l1321_132195


namespace area_percent_difference_l1321_132114

theorem area_percent_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  percent_difference = 4 := 
by
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  sorry

end area_percent_difference_l1321_132114


namespace area_ratio_proof_l1321_132138

open Real

noncomputable def area_ratio (FE AF DE CD ABCE : ℝ) :=
  (AF = 3 * FE) ∧ (CD = 3 * DE) ∧ (ABCE = 16 * FE^2) →
  (10 * FE^2 / ABCE = (5 / 8))

theorem area_ratio_proof (FE AF DE CD ABCE : ℝ) :
  AF = 3 * FE → CD = 3 * DE → ABCE = 16 * FE^2 →
  10 * FE^2 / ABCE = 5 / 8 :=
by
  intro hAF hCD hABCE
  sorry

end area_ratio_proof_l1321_132138


namespace age_of_vanya_and_kolya_l1321_132109

theorem age_of_vanya_and_kolya (P V K : ℕ) (hP : P = 10)
  (hV : V = P - 1) (hK : K = P - 5 + 1) : V = 9 ∧ K = 6 :=
by
  sorry

end age_of_vanya_and_kolya_l1321_132109


namespace divisor_five_l1321_132150

theorem divisor_five {D : ℝ} (h : 95 / D + 23 = 42) : D = 5 := by
  sorry

end divisor_five_l1321_132150


namespace intersection_of_M_and_N_l1321_132111

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0} := 
by sorry

end intersection_of_M_and_N_l1321_132111


namespace num_integers_n_with_properties_l1321_132156

theorem num_integers_n_with_properties :
  ∃ (N : Finset ℕ), N.card = 50 ∧
  ∀ n ∈ N, n < 150 ∧
    ∃ (m : ℕ), (∃ k, n = 2*k + 1 ∧ m = k*(k+1)) ∧ ¬ (3 ∣ m) :=
sorry

end num_integers_n_with_properties_l1321_132156


namespace find_value_of_c_l1321_132105

-- Mathematical proof problem in Lean 4 statement
theorem find_value_of_c (a b c d : ℝ)
  (h1 : a + c = 900)
  (h2 : b + c = 1100)
  (h3 : a + d = 700)
  (h4 : a + b + c + d = 2000) : 
  c = 200 :=
sorry

end find_value_of_c_l1321_132105


namespace problem_1_problem_2_l1321_132199

-- Problem 1:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is {x | x < -3 or x > -1}, prove k = -1/2
theorem problem_1 {k : ℝ} :
  (∀ x : ℝ, (kx^2 - 2*x + 3*k < 0 ↔ x < -3 ∨ x > -1)) → k = -1/2 :=
sorry

-- Problem 2:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is ∅, prove 0 < k ≤ sqrt(3) / 3
theorem problem_2 {k : ℝ} :
  (∀ x : ℝ, ¬ (kx^2 - 2*x + 3*k < 0)) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end problem_1_problem_2_l1321_132199


namespace total_cost_production_l1321_132108

variable (FC MC : ℕ) (n : ℕ)

theorem total_cost_production : FC = 12000 → MC = 200 → n = 20 → (FC + MC * n = 16000) :=
by
  intro hFC hMC hn
  sorry

end total_cost_production_l1321_132108


namespace solve_inequality_l1321_132174

theorem solve_inequality : {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {(-1 : ℝ) / 3} :=
by
  sorry

end solve_inequality_l1321_132174


namespace factor_expression_l1321_132187

theorem factor_expression (m n x y : ℝ) :
  m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end factor_expression_l1321_132187


namespace power_eval_l1321_132151

theorem power_eval : (9^6 * 3^4) / (27^5) = 3 := by
  sorry

end power_eval_l1321_132151


namespace wang_payment_correct_l1321_132119

noncomputable def first_trip_payment (x : ℝ) : ℝ := 0.9 * x
noncomputable def second_trip_payment (y : ℝ) : ℝ := 300 * 0.9 + (y - 300) * 0.8

theorem wang_payment_correct (x y: ℝ) 
  (cond1: 0.1 * x = 19)
  (cond2: (x + y) - (0.9 * x + ((y - 300) * 0.8 + 300 * 0.9)) = 67) :
  first_trip_payment x = 171 ∧ second_trip_payment y = 342 := 
by
  sorry

end wang_payment_correct_l1321_132119


namespace mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l1321_132194

-- Definitions
def mad_hatter_clock_rate := 5 / 4
def march_hare_clock_rate := 5 / 6
def time_at_dormouse_clock := 5 -- 5:00 PM

-- Real time calculation based on clock rates
def real_time (clock_rate : ℚ) (clock_time : ℚ) : ℚ := clock_time * (1 / clock_rate)

-- Mad Hatter's and March Hare's arrival times in real time
def mad_hatter_real_time := real_time mad_hatter_clock_rate time_at_dormouse_clock
def march_hare_real_time := real_time march_hare_clock_rate time_at_dormouse_clock

-- Theorems to be proved
theorem mad_hatter_waiting_time : mad_hatter_real_time = 4 := sorry
theorem march_hare_waiting_time : march_hare_real_time = 6 := sorry

-- Main theorem
theorem waiting_time : march_hare_real_time - mad_hatter_real_time = 2 := sorry

end mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l1321_132194


namespace mushroom_drying_l1321_132118

theorem mushroom_drying (M M' : ℝ) (m1 m2 : ℝ) :
  M = 100 ∧ m1 = 0.01 * M ∧ m2 = 0.02 * M' ∧ m1 = 1 → M' = 50 :=
by
  sorry

end mushroom_drying_l1321_132118


namespace length_of_track_l1321_132192

-- Conditions as definitions
def Janet_runs (m : Nat) := m = 120
def Leah_distance_after_first_meeting (x : Nat) (m : Nat) := m = (x / 2 - 120 + 200)
def Janet_distance_after_first_meeting (x : Nat) (m : Nat) := m = (x - 120 + (x - (x / 2 + 80)))

-- Questions and answers combined in proof statement
theorem length_of_track (x : Nat) (hx : Janet_runs 120) (hy : Leah_distance_after_first_meeting x 280) (hz : Janet_distance_after_first_meeting x (x / 2 - 40)) :
  x = 480 :=
sorry

end length_of_track_l1321_132192


namespace num_positive_terms_arithmetic_seq_l1321_132139

theorem num_positive_terms_arithmetic_seq :
  (∃ k : ℕ+, (∀ n : ℕ, n ≤ k → (90 - 2 * n) > 0)) → (k = 44) :=
sorry

end num_positive_terms_arithmetic_seq_l1321_132139


namespace minimum_value_of_f_l1321_132133

noncomputable def f (x a : ℝ) := (1/3) * x^3 + (a-1) * x^2 - 4 * a * x + a

theorem minimum_value_of_f (a : ℝ) (h : a < -1) :
  (if -3/2 < a then ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f (-2*a) a
   else ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f 3 a) :=
sorry

end minimum_value_of_f_l1321_132133


namespace negation_of_universal_proposition_l1321_132122

variable (p : Prop)
variable (x : ℝ)

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l1321_132122


namespace counts_duel_with_marquises_l1321_132124

theorem counts_duel_with_marquises (x y z k : ℕ) (h1 : 3 * x = 2 * y) (h2 : 6 * y = 3 * z)
    (h3 : ∀ c : ℕ, c = x → ∃ m : ℕ, m = k) : k = 6 :=
by
  sorry

end counts_duel_with_marquises_l1321_132124


namespace simplify_fraction_l1321_132115

theorem simplify_fraction :
  5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l1321_132115


namespace total_bus_capacity_l1321_132188

def left_seats : ℕ := 15
def right_seats : ℕ := left_seats - 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 8

theorem total_bus_capacity :
  (left_seats + right_seats) * people_per_seat + back_seat_capacity = 89 := by
  sorry

end total_bus_capacity_l1321_132188


namespace number_of_oranges_l1321_132120

theorem number_of_oranges (B T O : ℕ) (h₁ : B + T = 178) (h₂ : B + T + O = 273) : O = 95 :=
by
  -- Begin proof here
  sorry

end number_of_oranges_l1321_132120
