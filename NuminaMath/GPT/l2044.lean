import Mathlib

namespace find_g_neg_three_l2044_204495

namespace ProofProblem

def g (d e f x : ℝ) : ℝ := d * x^5 + e * x^3 + f * x + 6

theorem find_g_neg_three (d e f : ℝ) (h : g d e f 3 = -9) : g d e f (-3) = 21 := by
  sorry

end ProofProblem

end find_g_neg_three_l2044_204495


namespace bids_per_person_l2044_204401

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l2044_204401


namespace similar_polygons_area_sum_l2044_204435

theorem similar_polygons_area_sum (a b c k : ℝ) (t' t'' T : ℝ)
    (h₁ : t' = k * a^2)
    (h₂ : t'' = k * b^2)
    (h₃ : T = t' + t''):
    c^2 = a^2 + b^2 := 
by 
  sorry

end similar_polygons_area_sum_l2044_204435


namespace part_a_part_b_l2044_204413

noncomputable def f (g n : ℕ) : ℕ := g^n + 1

theorem part_a (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → f g n ∣ f g (3*n) ∧ f g n ∣ f g (5*n) ∧ f g n ∣ f g (7*n) :=
sorry

theorem part_b (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → ∀ k : ℕ, 1 ≤ k → gcd (f g n) (f g (2*k*n)) = 1 :=
sorry

end part_a_part_b_l2044_204413


namespace GMAT_scores_ratio_l2044_204420

variables (u v w : ℝ)

theorem GMAT_scores_ratio
  (h1 : u - w = (u + v + w) / 3)
  (h2 : u - v = 2 * (v - w))
  : v / u = 4 / 7 :=
sorry

end GMAT_scores_ratio_l2044_204420


namespace more_boys_than_girls_l2044_204438

theorem more_boys_than_girls : 
  let girls := 28.0
  let boys := 35.0
  boys - girls = 7.0 :=
by
  sorry

end more_boys_than_girls_l2044_204438


namespace ninja_star_ratio_l2044_204499

-- Define variables for the conditions
variables (Eric_stars Chad_stars Jeff_stars Total_stars : ℕ) (Jeff_bought : ℕ)

/-- Given the following conditions:
1. Eric has 4 ninja throwing stars.
2. Jeff now has 6 throwing stars.
3. Jeff bought 2 ninja stars from Chad.
4. Altogether, they have 16 ninja throwing stars.

We want to prove that the ratio of the number of ninja throwing stars Chad has to the number Eric has is 2:1. --/
theorem ninja_star_ratio
  (h1 : Eric_stars = 4)
  (h2 : Jeff_stars = 6)
  (h3 : Jeff_bought = 2)
  (h4 : Total_stars = 16)
  (h5 : Eric_stars + Jeff_stars - Jeff_bought + Chad_stars = Total_stars) :
  Chad_stars / Eric_stars = 2 :=
by
  sorry

end ninja_star_ratio_l2044_204499


namespace find_angle_QPR_l2044_204487

-- Define the angles and line segment
variables (R S Q T P : Type) 
variables (line_RT : R ≠ S)
variables (x : ℝ) 
variables (angle_PTQ : ℝ := 62)
variables (angle_RPS : ℝ := 34)

-- Hypothesis that PQ = PT, making triangle PQT isosceles
axiom eq_PQ_PT : ℝ

-- Conditions
axiom lie_on_RT : ∀ {R S Q T : Type}, R ≠ S 
axiom angle_PTQ_eq : angle_PTQ = 62
axiom angle_RPS_eq : angle_RPS = 34

-- Hypothesis that defines the problem structure
theorem find_angle_QPR : x = 11 := by
sorry

end find_angle_QPR_l2044_204487


namespace perimeter_of_figure_is_correct_l2044_204424

-- Define the conditions as Lean variables and constants
def area_of_figure : ℝ := 144
def number_of_squares : ℕ := 4

-- Define the question as a theorem to be proven in Lean
theorem perimeter_of_figure_is_correct :
  let area_of_square := area_of_figure / number_of_squares
  let side_length := Real.sqrt area_of_square
  let perimeter := 9 * side_length
  perimeter = 54 :=
by
  intro area_of_square
  intro side_length
  intro perimeter
  sorry

end perimeter_of_figure_is_correct_l2044_204424


namespace smallest_lcm_of_4digit_gcd_5_l2044_204464

theorem smallest_lcm_of_4digit_gcd_5 :
  ∃ (m n : ℕ), (1000 ≤ m ∧ m < 10000) ∧ (1000 ≤ n ∧ n < 10000) ∧ 
               m.gcd n = 5 ∧ m.lcm n = 203010 :=
by sorry

end smallest_lcm_of_4digit_gcd_5_l2044_204464


namespace money_distribution_l2044_204488

theorem money_distribution (A B C : ℝ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 340) : 
  C = 40 := 
sorry

end money_distribution_l2044_204488


namespace geometric_seq_a4_a7_l2044_204421

variable {a : ℕ → ℝ}

def is_geometric (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_seq_a4_a7
  (h_geom : is_geometric a)
  (h_roots : ∃ a_1 a_10 : ℝ, (a 1 = a_1 ∧ a 10 = a_10) ∧ (2 * a_1 ^ 2 + 5 * a_1 + 1 = 0) ∧ (2 * a_10 ^ 2 + 5 * a_10 + 1 = 0)):
  a 4 * a 7 = 1 / 2 :=
by
  sorry

end geometric_seq_a4_a7_l2044_204421


namespace savings_proof_l2044_204402

variable (income expenditure savings : ℕ)

def ratio_income_expenditure (i e : ℕ) := i / 10 = e / 7

theorem savings_proof (h : ratio_income_expenditure income expenditure) (hincome : income = 10000) :
  savings = income - expenditure → savings = 3000 :=
by
  sorry

end savings_proof_l2044_204402


namespace max_value_of_expression_eq_two_l2044_204481

noncomputable def max_value_of_expression (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) : ℝ :=
  (a^2 + b^2 + c^2) / c^2

theorem max_value_of_expression_eq_two (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) :
  max_value_of_expression a b c h_right_triangle h_a = 2 := by
  sorry

end max_value_of_expression_eq_two_l2044_204481


namespace trader_sold_95_pens_l2044_204455

theorem trader_sold_95_pens
  (C : ℝ)   -- cost price of one pen
  (N : ℝ)   -- number of pens sold
  (h1 : 19 * C = 0.20 * N * C):  -- condition: profit from selling N pens is equal to the cost of 19 pens, with 20% gain percentage
  N = 95 := by
-- You would place the proof here.
  sorry

end trader_sold_95_pens_l2044_204455


namespace rate_of_interest_per_annum_l2044_204418

def simple_interest (P T R : ℕ) : ℕ :=
  (P * T * R) / 100

theorem rate_of_interest_per_annum :
  let P_B := 5000
  let T_B := 2
  let P_C := 3000
  let T_C := 4
  let total_interest := 1980
  ∃ R : ℕ, 
      simple_interest P_B T_B R + simple_interest P_C T_C R = total_interest ∧
      R = 9 :=
by
  sorry

end rate_of_interest_per_annum_l2044_204418


namespace math_problem_l2044_204412

theorem math_problem
  (a b c : ℚ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c))
    / (2 / (a * b) - 2 * a * b / c)
    / (101 / c)
  ) = -1 / 202 := 
sorry

end math_problem_l2044_204412


namespace simplify_expression_evaluate_expression_with_values_l2044_204467

-- Problem 1: Simplify the expression to -xy
theorem simplify_expression (x y : ℤ) : 
  3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = - x * y :=
  sorry

-- Problem 2: Evaluate the expression with given values
theorem evaluate_expression_with_values (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 :=
  sorry

end simplify_expression_evaluate_expression_with_values_l2044_204467


namespace arithmetic_sequence_twentieth_term_l2044_204472

theorem arithmetic_sequence_twentieth_term
  (a1 : ℤ) (a13 : ℤ) (a20 : ℤ) (d : ℤ)
  (h1 : a1 = 3)
  (h2 : a13 = 27)
  (h3 : a13 = a1 + 12 * d)
  (h4 : a20 = a1 + 19 * d) : 
  a20 = 41 :=
by
  --  We assume a20 and prove it equals 41 instead of solving it in steps
  sorry

end arithmetic_sequence_twentieth_term_l2044_204472


namespace reading_order_l2044_204452

theorem reading_order (a b c d : ℝ) 
  (h1 : a + c = b + d) 
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by sorry

end reading_order_l2044_204452


namespace digit_after_decimal_l2044_204480

theorem digit_after_decimal (n : ℕ) : 
  (Nat.floor (10 * (Real.sqrt (n^2 + n) - Nat.floor (Real.sqrt (n^2 + n))))) = 4 :=
by
  sorry

end digit_after_decimal_l2044_204480


namespace find_t_l2044_204430

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the perpendicular condition and solve for t
theorem find_t (t : ℝ) : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 → t = -2 :=
by
  sorry

end find_t_l2044_204430


namespace greatest_possible_int_diff_l2044_204427

theorem greatest_possible_int_diff (x a y b : ℝ) 
    (hx : 3 < x ∧ x < 4) 
    (ha : 4 < a ∧ a < x) 
    (hy : 6 < y ∧ y < 8) 
    (hb : 8 < b ∧ b < y) 
    (h_ineq : a^2 + b^2 > x^2 + y^2) : 
    abs (⌊x⌋ - ⌈y⌉) = 2 :=
sorry

end greatest_possible_int_diff_l2044_204427


namespace train_speed_is_42_point_3_km_per_h_l2044_204493

-- Definitions for the conditions.
def train_length : ℝ := 150
def bridge_length : ℝ := 320
def crossing_time : ℝ := 40
def meter_per_sec_to_km_per_hour : ℝ := 3.6
def total_distance : ℝ := train_length + bridge_length

-- The theorem we want to prove
theorem train_speed_is_42_point_3_km_per_h : 
    (total_distance / crossing_time) * meter_per_sec_to_km_per_hour = 42.3 :=
by 
    -- Proof omitted
    sorry

end train_speed_is_42_point_3_km_per_h_l2044_204493


namespace number_of_solutions_decrease_l2044_204477

-- Define the conditions and the main theorem
theorem number_of_solutions_decrease (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∀ x y : ℝ, x^2 - x^2 = 0 ∧ (x - a)^2 + x^2 = 1) →
  a = 1 ∨ a = -1 := 
sorry

end number_of_solutions_decrease_l2044_204477


namespace base_comparison_l2044_204448

theorem base_comparison : (1 * 6^1 + 2 * 6^0) > (1 * 2^2 + 0 * 2^1 + 1 * 2^0) := by
  sorry

end base_comparison_l2044_204448


namespace remainder_eq_27_l2044_204445

def p (x : ℝ) : ℝ := x^4 + 2 * x^2 + 3
def a : ℝ := -2
def remainder := p (-2)
theorem remainder_eq_27 : remainder = 27 :=
by
  sorry

end remainder_eq_27_l2044_204445


namespace new_car_travel_distance_l2044_204474

theorem new_car_travel_distance
  (old_distance : ℝ)
  (new_distance : ℝ)
  (h1 : old_distance = 150)
  (h2 : new_distance = 1.30 * old_distance) : 
  new_distance = 195 := 
by 
  /- include required assumptions and skip the proof. -/
  sorry

end new_car_travel_distance_l2044_204474


namespace mil_equals_one_fortieth_mm_l2044_204442

-- The condition that one mil is equal to one thousandth of an inch
def mil_in_inch := 1 / 1000

-- The condition that an inch is about 2.5 cm
def inch_in_mm := 25

-- The problem statement in Lean 4 form
theorem mil_equals_one_fortieth_mm : (mil_in_inch * inch_in_mm = 1 / 40) :=
by
  sorry

end mil_equals_one_fortieth_mm_l2044_204442


namespace median_score_interval_l2044_204407

def intervals : List (Nat × Nat × Nat) :=
  [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]

def total_students : Nat := 100

def median_interval : Nat × Nat :=
  (70, 74)

theorem median_score_interval :
  ∃ l u n, intervals = [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]
  ∧ total_students = 100
  ∧ median_interval = (70, 74)
  ∧ ((l, u, n) ∈ intervals ∧ l ≤ 50 ∧ 50 ≤ u) :=
by
  sorry

end median_score_interval_l2044_204407


namespace band_formation_l2044_204422

theorem band_formation (r x m : ℕ) (h1 : r * x + 3 = m) (h2 : (r - 1) * (x + 2) = m) (h3 : m < 100) : m = 69 :=
by
  sorry

end band_formation_l2044_204422


namespace cos_A_value_compare_angles_l2044_204473

variable (A B C : ℝ) (a b c : ℝ)

-- Given conditions
variable (h1 : a = 3) (h2 : b = 2 * Real.sqrt 6) (h3 : B = 2 * A)

-- Problem (I) statement
theorem cos_A_value (hcosA : Real.cos A = Real.sqrt 6 / 3) : 
  Real.cos A = Real.sqrt 6 / 3 :=
by 
  sorry

-- Problem (II) statement
theorem compare_angles (hcosA : Real.cos A = Real.sqrt 6 / 3) (hcosC : Real.cos C = Real.sqrt 6 / 9) :
  B < C :=
by
  sorry

end cos_A_value_compare_angles_l2044_204473


namespace total_cans_l2044_204446

def bag1 := 5
def bag2 := 7
def bag3 := 12
def bag4 := 4
def bag5 := 8
def bag6 := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end total_cans_l2044_204446


namespace increasing_function_condition_l2044_204494

variable {x : ℝ} {a : ℝ}

theorem increasing_function_condition (h : 0 < a) :
  (∀ x ≥ 1, deriv (λ x => x^3 - a * x) x ≥ 0) ↔ (0 < a ∧ a ≤ 3) :=
by
  sorry

end increasing_function_condition_l2044_204494


namespace arithmetic_sequence_n_2005_l2044_204475

/-- Define an arithmetic sequence with first term a₁ = 1 and common difference d = 3. -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + (n - 1) * 3

/-- Statement of the proof problem. -/
theorem arithmetic_sequence_n_2005 : 
  ∃ n : ℕ, arithmetic_sequence n = 2005 ∧ n = 669 := 
sorry

end arithmetic_sequence_n_2005_l2044_204475


namespace negation_exists_l2044_204436

-- Definitions used in the conditions
def prop1 (x : ℝ) : Prop := x^2 ≥ 1
def neg_prop1 : Prop := ∃ x : ℝ, x^2 < 1

-- Statement to be proved
theorem negation_exists (h : ∀ x : ℝ, prop1 x) : neg_prop1 :=
by
  sorry

end negation_exists_l2044_204436


namespace total_ideal_matching_sets_l2044_204406

-- Definitions based on the provided problem statement
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def is_ideal_matching_set (A B : Set ℕ) : Prop := A ∩ B = {1, 3, 5}

-- Theorem statement for the total number of ideal matching sets
theorem total_ideal_matching_sets : ∃ n, n = 27 ∧ ∀ (A B : Set ℕ), A ⊆ U ∧ B ⊆ U ∧ is_ideal_matching_set A B → n = 27 := 
sorry

end total_ideal_matching_sets_l2044_204406


namespace complement_A_in_U_l2044_204450

open Set

variable {𝕜 : Type*} [LinearOrderedField 𝕜]

def A (x : 𝕜) : Prop := |x - (1 : 𝕜)| > 2
def U : Set 𝕜 := univ

theorem complement_A_in_U : (U \ {x : 𝕜 | A x}) = {x : 𝕜 | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_in_U_l2044_204450


namespace team_card_sending_l2044_204437

theorem team_card_sending (x : ℕ) (h : x * (x - 1) = 56) : x * (x - 1) = 56 := 
by 
  sorry

end team_card_sending_l2044_204437


namespace johns_current_income_l2044_204441

theorem johns_current_income
  (prev_income : ℝ := 1000000)
  (prev_tax_rate : ℝ := 0.20)
  (new_tax_rate : ℝ := 0.30)
  (extra_taxes_paid : ℝ := 250000) :
  ∃ (X : ℝ), 0.30 * X - 0.20 * prev_income = extra_taxes_paid ∧ X = 1500000 :=
by
  use 1500000
  -- Proof would come here
  sorry

end johns_current_income_l2044_204441


namespace find_function_α_l2044_204415

theorem find_function_α (α : ℝ) (hα : 0 < α) 
  (f : ℕ+ → ℝ) (h : ∀ k m : ℕ+, α * m ≤ k ∧ k < (α + 1) * m → f (k + m) = f k + f m) :
  ∃ b : ℝ, ∀ n : ℕ+, f n = b * n :=
sorry

end find_function_α_l2044_204415


namespace meet_at_starting_point_second_time_in_minutes_l2044_204447

theorem meet_at_starting_point_second_time_in_minutes :
  let racing_magic_time := 60 -- in seconds
  let charging_bull_time := 3600 / 40 -- in seconds
  let lcm_time := Nat.lcm racing_magic_time charging_bull_time -- LCM of the round times in seconds
  let answer := lcm_time / 60 -- convert seconds to minutes
  answer = 3 :=
by
  sorry

end meet_at_starting_point_second_time_in_minutes_l2044_204447


namespace exists_unique_solution_l2044_204426

theorem exists_unique_solution : ∀ a b : ℝ, 2 * (a ^ 2 + 1) * (b ^ 2 + 1) = (a + 1) * (b + 1) * (a * b + 1) ↔ (a, b) = (1, 1) := by
  sorry

end exists_unique_solution_l2044_204426


namespace line_through_circle_center_slope_one_eq_l2044_204476

theorem line_through_circle_center_slope_one_eq (x y : ℝ) :
  (∃ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 ∧ y = 2) →
  (∃ m : ℝ, m = 1 ∧ (x + 1) = m * (y - 2)) →
  (x - y + 3 = 0) :=
sorry

end line_through_circle_center_slope_one_eq_l2044_204476


namespace circle_area_l2044_204491

-- Definition of the given circle equation
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0

-- Prove the area of the circle defined by circle_eq (x y) is 25/4 * π
theorem circle_area (x y : ℝ) (h : circle_eq x y) : ∃ r : ℝ, r = 5 / 2 ∧ π * r^2 = 25 / 4 * π :=
by
  sorry

end circle_area_l2044_204491


namespace tangent_condition_l2044_204483

def curve1 (x y : ℝ) : Prop := y = x ^ 3 + 2
def curve2 (x y m : ℝ) : Prop := y^2 - m * x = 1

theorem tangent_condition (m : ℝ) (h : ∃ x y : ℝ, curve1 x y ∧ curve2 x y m) :
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end tangent_condition_l2044_204483


namespace compute_expression_l2044_204459

theorem compute_expression : (7^2 - 2 * 5 + 2^3) = 47 :=
by
  sorry

end compute_expression_l2044_204459


namespace who_plays_chess_l2044_204434

def person_plays_chess (A B C : Prop) : Prop := 
  (A ∧ ¬ B ∧ ¬ C) ∨ (¬ A ∧ B ∧ ¬ C) ∨ (¬ A ∧ ¬ B ∧ C)

axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom one_statement_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Definition translating the statements made by A, B, and C
def A_plays := true
def B_not_plays := true
def A_not_plays := ¬ A_plays

-- Axiom stating that only one of A's, B's, or C's statements are true
axiom only_one_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Prove that B is the one who knows how to play Chinese chess
theorem who_plays_chess : B_plays :=
by
  -- Insert proof steps here
  sorry

end who_plays_chess_l2044_204434


namespace ellipse_equation_and_m_value_l2044_204492

variable {a b : ℝ}
variable (e : ℝ) (F : ℝ × ℝ) (h1 : e = Real.sqrt 2 / 2) (h2 : F = (1, 0))

theorem ellipse_equation_and_m_value (h3 : a > b) (h4 : b > 0) 
  (h5 : (x y : ℝ) → (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 → (x - 1) ^ 2 + y ^ 2 = 1) :
  (a = Real.sqrt 2 ∧ b = 1) ∧
  (∀ m : ℝ, (y = x + m) → 
  ((∃ A B : ℝ × ℝ, A = (x₁, x₁ + m) ∧ B = (x₂, x₂ + m) ∧
  (x₁ ^ 2) / 2 + (x₁ + m) ^ 2 = 1 ∧ (x₂ ^ 2) / 2 + (x₂ + m) ^ 2 = 1 ∧
  x₁ * x₂ + (x₁ + m) * (x₂ + m) = -1) ↔ m = Real.sqrt 3 / 3 ∨ m = - Real.sqrt 3 / 3))
  :=
sorry

end ellipse_equation_and_m_value_l2044_204492


namespace inequality_ab_l2044_204410

theorem inequality_ab (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end inequality_ab_l2044_204410


namespace percentage_of_part_of_whole_l2044_204478

theorem percentage_of_part_of_whole :
  let part := 375.2
  let whole := 12546.8
  (part / whole) * 100 = 2.99 :=
by
  sorry

end percentage_of_part_of_whole_l2044_204478


namespace hypotenuse_length_l2044_204496

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 := 
by
  sorry

end hypotenuse_length_l2044_204496


namespace chess_tournament_participants_l2044_204470

theorem chess_tournament_participants (n : ℕ) 
  (h : (n * (n - 1)) / 2 = 15) : n = 6 :=
sorry

end chess_tournament_participants_l2044_204470


namespace train_length_l2044_204486

namespace TrainProblem

def speed_kmh : ℤ := 60
def time_sec : ℤ := 18
def speed_ms : ℚ := (speed_kmh : ℚ) * (1000 / 1) * (1 / 3600)
def length_meter := speed_ms * (time_sec : ℚ)

theorem train_length :
  length_meter = 300.06 := by
  sorry

end TrainProblem

end train_length_l2044_204486


namespace trigonometric_identity_l2044_204479

open Real

theorem trigonometric_identity (α : ℝ) (hα : sin (2 * π - α) = 4 / 5) (hα_range : 3 * π / 2 < α ∧ α < 2 * π) : 
  (sin α + cos α) / (sin α - cos α) = 1 / 7 := 
by
  sorry

end trigonometric_identity_l2044_204479


namespace mary_total_baseball_cards_l2044_204405

noncomputable def mary_initial_baseball_cards : ℕ := 18
noncomputable def torn_baseball_cards : ℕ := 8
noncomputable def fred_given_baseball_cards : ℕ := 26
noncomputable def mary_bought_baseball_cards : ℕ := 40

theorem mary_total_baseball_cards :
  mary_initial_baseball_cards - torn_baseball_cards + fred_given_baseball_cards + mary_bought_baseball_cards = 76 :=
by
  sorry

end mary_total_baseball_cards_l2044_204405


namespace correct_tourism_model_l2044_204416

noncomputable def tourism_model (x : ℕ) : ℝ :=
  80 * (Real.cos ((Real.pi / 6) * x + (2 * Real.pi / 3))) + 120

theorem correct_tourism_model :
  (∀ n : ℕ, tourism_model (n + 12) = tourism_model n) ∧
  (tourism_model 8 - tourism_model 2 = 160) ∧
  (tourism_model 2 = 40) :=
by
  sorry

end correct_tourism_model_l2044_204416


namespace store_profit_l2044_204468

variables (m n : ℝ)

def total_profit (m n : ℝ) : ℝ :=
  110 * m - 50 * n

theorem store_profit (m n : ℝ) : total_profit m n = 110 * m - 50 * n :=
  by
  -- sorry indicates that the proof is skipped
  sorry

end store_profit_l2044_204468


namespace day50_previous_year_is_Wednesday_l2044_204489

-- Given conditions
variable (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)

-- Provided conditions stating specific days are Fridays
def day250_is_Friday : Prop := dayOfWeek 250 N = 5
def day150_is_Friday_next_year : Prop := dayOfWeek 150 (N+1) = 5

-- Proving the day of week for the 50th day of year N-1
def day50_previous_year : Prop := dayOfWeek 50 (N-1) = 3

-- Main theorem tying it together
theorem day50_previous_year_is_Wednesday (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)
  (h1 : day250_is_Friday N dayOfWeek)
  (h2 : day150_is_Friday_next_year N dayOfWeek) :
  day50_previous_year N dayOfWeek :=
sorry -- Placeholder for actual proof

end day50_previous_year_is_Wednesday_l2044_204489


namespace remaining_money_after_expenditures_l2044_204440

def initial_amount : ℝ := 200.50
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20

theorem remaining_money_after_expenditures :
  ((initial_amount - spent_on_sweets) - 2 * given_to_each_friend) = 114.85 :=
by
  sorry

end remaining_money_after_expenditures_l2044_204440


namespace find_a_l2044_204465

theorem find_a (a : ℝ) (h : 1 / Real.log 5 / Real.log a + 1 / Real.log 6 / Real.log a + 1 / Real.log 10 / Real.log a = 1) : a = 300 :=
sorry

end find_a_l2044_204465


namespace parametric_curve_intersects_itself_l2044_204497

-- Given parametric equations
def param_x (t : ℝ) : ℝ := t^2 + 3
def param_y (t : ℝ) : ℝ := t^3 - 6 * t + 4

-- Existential statement for self-intersection
theorem parametric_curve_intersects_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ param_x t1 = param_x t2 ∧ param_y t1 = param_y t2 ∧ param_x t1 = 9 ∧ param_y t1 = 4 :=
sorry

end parametric_curve_intersects_itself_l2044_204497


namespace more_girls_than_boys_l2044_204482

theorem more_girls_than_boys (total_kids girls boys : ℕ) (h1 : total_kids = 34) (h2 : girls = 28) (h3 : total_kids = girls + boys) : girls - boys = 22 :=
by
  -- Proof placeholder
  sorry

end more_girls_than_boys_l2044_204482


namespace numeral_is_1_11_l2044_204457

-- Define the numeral question and condition
def place_value_difference (a b : ℝ) : Prop :=
  10 * b - b = 99.99

-- Now we define the problem statement in Lean
theorem numeral_is_1_11 (a b : ℝ) (h : place_value_difference a b) : 
  a = 100 ∧ b = 11.11 ∧ (a - b = 99.99) :=
  sorry

end numeral_is_1_11_l2044_204457


namespace geometric_series_smallest_b_l2044_204460

theorem geometric_series_smallest_b (a b c : ℝ) (h_geometric : a * c = b^2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 216) : b = 6 :=
sorry

end geometric_series_smallest_b_l2044_204460


namespace problem_l2044_204431

theorem problem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) 
  (h1 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := 
by
  sorry

end problem_l2044_204431


namespace carlos_jogged_distance_l2044_204469

def carlos_speed := 4 -- Carlos's speed in miles per hour
def jogging_time := 2 -- Time in hours

theorem carlos_jogged_distance : carlos_speed * jogging_time = 8 :=
by
  sorry

end carlos_jogged_distance_l2044_204469


namespace problem_solution_l2044_204443

/-- Define proposition p: ∀α∈ℝ, sin(π-α) ≠ -sin(α) -/
def p := ∀ α : ℝ, Real.sin (Real.pi - α) ≠ -Real.sin α

/-- Define proposition q: ∃x∈[0,+∞), sin(x) > x -/
def q := ∃ x : ℝ, 0 ≤ x ∧ Real.sin x > x

/-- Prove that ¬p ∨ q is a true proposition -/
theorem problem_solution : ¬p ∨ q :=
by
  sorry

end problem_solution_l2044_204443


namespace parabola_translation_shift_downwards_l2044_204454

theorem parabola_translation_shift_downwards :
  ∀ (x y : ℝ), (y = x^2 - 5) ↔ ((∃ (k : ℝ), k = -5 ∧ y = x^2 + k)) :=
by
  sorry

end parabola_translation_shift_downwards_l2044_204454


namespace initial_value_l2044_204484

theorem initial_value (x k : ℤ) (h : x + 335 = k * 456) : x = 121 := sorry

end initial_value_l2044_204484


namespace distance_from_center_to_line_l2044_204466

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l2044_204466


namespace count_four_digit_numbers_l2044_204461

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l2044_204461


namespace mario_total_flowers_l2044_204453

-- Define the number of flowers on the first plant
def F1 : ℕ := 2

-- Define the number of flowers on the second plant as twice the first
def F2 : ℕ := 2 * F1

-- Define the number of flowers on the third plant as four times the second
def F3 : ℕ := 4 * F2

-- Prove that total number of flowers is 22
theorem mario_total_flowers : F1 + F2 + F3 = 22 := by
  -- Proof is to be filled here
  sorry

end mario_total_flowers_l2044_204453


namespace primes_sum_eq_2001_l2044_204400

/-- If a and b are prime numbers such that a^2 + b = 2003, then a + b = 2001. -/
theorem primes_sum_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) :
    a + b = 2001 := 
  sorry

end primes_sum_eq_2001_l2044_204400


namespace minimal_pieces_required_for_cubes_l2044_204417

theorem minimal_pieces_required_for_cubes 
  (e₁ e₂ n₁ n₂ n₃ : ℕ)
  (h₁ : e₁ = 14)
  (h₂ : e₂ = 10)
  (h₃ : n₁ = 13)
  (h₄ : n₂ = 11)
  (h₅ : n₃ = 6)
  (disassembly_possible : ∀ {x y z : ℕ}, x^3 + y^3 = z^3 → n₁^3 + n₂^3 + n₃^3 = 14^3 + 10^3)
  (cutting_constraints : ∀ d : ℕ, (d > 0) → (d ≤ e₁ ∨ d ≤ e₂) → (d ≤ n₁ ∨ d ≤ n₂ ∨ d ≤ n₃) → (d ≤ 6))
  : ∃ minimal_pieces : ℕ, minimal_pieces = 11 := 
sorry

end minimal_pieces_required_for_cubes_l2044_204417


namespace trigonometric_identity_l2044_204432

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end trigonometric_identity_l2044_204432


namespace part1_area_quadrilateral_part2_maximized_line_equation_l2044_204433

noncomputable def area_MA_NB (α : ℝ) : ℝ :=
  (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)

theorem part1_area_quadrilateral (α : ℝ) :
  area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2) :=
by sorry

theorem part2_maximized_line_equation :
  ∃ α : ℝ, area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)
    ∧ (Real.tan α = -1 / 2) ∧ (∀ x : ℝ, x = -1 / 2 * y + Real.sqrt 5 / 2) :=
by sorry

end part1_area_quadrilateral_part2_maximized_line_equation_l2044_204433


namespace median_product_sum_l2044_204449

-- Let's define the lengths of medians and distances from a point P to these medians
variables {s1 s2 s3 d1 d2 d3 : ℝ}

-- Define the conditions
def is_median_lengths (s1 s2 s3 : ℝ) : Prop := 
  ∃ (A B C : ℝ × ℝ), -- vertices of the triangle
    (s1 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2) ∧
    (s2 = ((C.1 - B.1)^2 + (C.2 - B.2)^2) / 2) ∧
    (s3 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 2)

def distances_to_medians (d1 d2 d3 : ℝ) : Prop :=
  ∃ (P A B C : ℝ × ℝ), -- point P and vertices of the triangle
    (d1 = dist P ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
    (d2 = dist P ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) ∧
    (d3 = dist P ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

-- The theorem which we need to prove
theorem median_product_sum (h_medians : is_median_lengths s1 s2 s3) 
  (h_distances : distances_to_medians d1 d2 d3) :
  s1 * d1 + s2 * d2 + s3 * d3 = 0 := sorry

end median_product_sum_l2044_204449


namespace smallest_integer_in_set_l2044_204403

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 157) (h2 : greatest = 169) :
  ∃ (smallest : ℤ), smallest = 145 :=
by
  -- Setup the conditions
  have set_cons_odd : True := trivial
  -- Known facts
  have h_median : median = 157 := by exact h1
  have h_greatest : greatest = 169 := by exact h2
  -- We must prove
  existsi 145
  sorry

end smallest_integer_in_set_l2044_204403


namespace inscribed_circle_radius_right_triangle_l2044_204451

theorem inscribed_circle_radius_right_triangle : 
  ∀ (DE EF DF : ℝ), 
    DE = 6 →
    EF = 8 →
    DF = 10 →
    ∃ (r : ℝ), r = 2 :=
by
  intros DE EF DF hDE hEF hDF
  sorry

end inscribed_circle_radius_right_triangle_l2044_204451


namespace algebra_problem_l2044_204471

theorem algebra_problem (a b c d x : ℝ) (h1 : a = -b) (h2 : c * d = 1) (h3 : |x| = 3) : 
  (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end algebra_problem_l2044_204471


namespace max_tension_of_pendulum_l2044_204490

theorem max_tension_of_pendulum 
  (m g L θ₀ : ℝ) 
  (h₀ : θ₀ < π / 2) 
  (T₀ : ℝ) 
  (no_air_resistance : true) 
  (no_friction : true) : 
  ∃ T_max, T_max = m * g * (3 - 2 * Real.cos θ₀) := 
by 
  sorry

end max_tension_of_pendulum_l2044_204490


namespace angle_measure_triple_complement_l2044_204462

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l2044_204462


namespace Ramesh_paid_l2044_204425

theorem Ramesh_paid (P : ℝ) (h1 : 1.10 * P = 21725) : 0.80 * P + 125 + 250 = 16175 :=
by
  sorry

end Ramesh_paid_l2044_204425


namespace modulus_of_z_is_five_l2044_204404

def z : Complex := 3 + 4 * Complex.I

theorem modulus_of_z_is_five : Complex.abs z = 5 := by
  sorry

end modulus_of_z_is_five_l2044_204404


namespace solve_quadratic_complete_square_l2044_204429

theorem solve_quadratic_complete_square :
  ∃ b c : ℤ, (∀ x : ℝ, (x + b)^2 = c ↔ x^2 + 6 * x - 9 = 0) ∧ b + c = 21 := by
  sorry

end solve_quadratic_complete_square_l2044_204429


namespace fraction_of_income_from_tips_l2044_204439

theorem fraction_of_income_from_tips 
  (salary tips : ℝ)
  (h1 : tips = (7/4) * salary) 
  (total_income : ℝ)
  (h2 : total_income = salary + tips) :
  (tips / total_income) = (7 / 11) :=
by
  sorry

end fraction_of_income_from_tips_l2044_204439


namespace find_base_of_numeral_system_l2044_204485

def base_of_numeral_system (x : ℕ) : Prop :=
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2

theorem find_base_of_numeral_system :
  ∃ x : ℕ, base_of_numeral_system x ∧ x = 7 := sorry

end find_base_of_numeral_system_l2044_204485


namespace find_pqr_l2044_204444

variable (p q r : ℚ)

theorem find_pqr (h1 : ∃ a : ℚ, ∀ x : ℚ, (p = a) ∧ (q = -2 * a * 3) ∧ (r = a * 3 * 3 + 7) ∧ (r = 10 + 7)) :
  p + q + r = 8 + 1/3 := by
  sorry

end find_pqr_l2044_204444


namespace probability_multiple_of_3_or_4_l2044_204409

theorem probability_multiple_of_3_or_4 : ((15 : ℚ) / 30) = (1 / 2) := by
  sorry

end probability_multiple_of_3_or_4_l2044_204409


namespace ratio_expression_value_l2044_204423

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l2044_204423


namespace math_problem_l2044_204411

theorem math_problem
  (a b c : ℝ)
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 :=
sorry

end math_problem_l2044_204411


namespace area_of_red_region_on_larger_sphere_l2044_204414

/-- 
A smooth ball with a radius of 1 cm was dipped in red paint and placed between two 
absolutely smooth concentric spheres with radii of 4 cm and 6 cm, respectively
(the ball is outside the smaller sphere but inside the larger sphere).
As the ball moves and touches both spheres, it leaves a red mark. 
After traveling a closed path, a region outlined in red with an area of 37 square centimeters is formed on the smaller sphere. 
Find the area of the region outlined in red on the larger sphere. 
The answer should be 55.5 square centimeters.
-/
theorem area_of_red_region_on_larger_sphere
  (r1 r2 r3 : ℝ)
  (A_small : ℝ)
  (h_red_small_sphere : 37 = 2 * π * r2 * (A_small / (2 * π * r2)))
  (h_red_large_sphere : 55.5 = 2 * π * r3 * (A_small / (2 * π * r2))) :
  ∃ A_large : ℝ, A_large = 55.5 :=
by
  -- Definitions and conditions
  let r1 := 1  -- radius of small ball (1 cm)
  let r2 := 4  -- radius of smaller sphere (4 cm)
  let r3 := 6  -- radius of larger sphere (6 cm)

  -- Given: A small red area is 37 cm^2 on the smaller sphere.
  let A_small := 37

  -- Proof of the relationship of the spherical caps
  sorry

end area_of_red_region_on_larger_sphere_l2044_204414


namespace original_number_abc_l2044_204408

theorem original_number_abc (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 528)
  (N : ℕ)
  (h1 : N + (100 * a + 10 * b + c) = 222 * (a + b + c))
  (hN : N = 2670) :
  100 * a + 10 * b + c = 528 := by
  sorry

end original_number_abc_l2044_204408


namespace triangle_side_ratio_l2044_204428

theorem triangle_side_ratio
  (α β γ : Real)
  (a b c p q r : Real)
  (h1 : (Real.tan α) / (Real.tan β) = p / q)
  (h2 : (Real.tan β) / (Real.tan γ) = q / r)
  (h3 : (Real.tan γ) / (Real.tan α) = r / p) :
  a^2 / b^2 / c^2 = (1/q + 1/r) / (1/r + 1/p) / (1/p + 1/q) := 
sorry

end triangle_side_ratio_l2044_204428


namespace calculate_total_cost_l2044_204463

def num_chicken_nuggets := 100
def num_per_box := 20
def cost_per_box := 4

theorem calculate_total_cost :
  (num_chicken_nuggets / num_per_box) * cost_per_box = 20 := by
  sorry

end calculate_total_cost_l2044_204463


namespace triangle_perimeter_from_medians_l2044_204419

theorem triangle_perimeter_from_medians (m1 m2 m3 : ℕ) (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) :
  ∃ (p : ℕ), p = 26 :=
by sorry

end triangle_perimeter_from_medians_l2044_204419


namespace final_length_of_movie_l2044_204498

theorem final_length_of_movie :
  let original_length := 3600 -- original movie length in seconds
  let cut_1 := 3 * 60 -- first scene cut in seconds
  let cut_2 := (5 * 60) + 30 -- second scene cut in seconds
  let cut_3 := (2 * 60) + 15 -- third scene cut in seconds
  let total_cut := cut_1 + cut_2 + cut_3 -- total cut time in seconds
  let final_length_seconds := original_length - total_cut -- final length in seconds
  final_length_seconds = 2955 ∧ final_length_seconds / 60 = 49 ∧ final_length_seconds % 60 = 15
:= by
  sorry

end final_length_of_movie_l2044_204498


namespace ratio_a_to_c_l2044_204458

theorem ratio_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 1 / 3) :
  a / c = 15 / 8 :=
by {
  sorry
}

end ratio_a_to_c_l2044_204458


namespace simplify_fraction_l2044_204456

theorem simplify_fraction :
  (4 / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27)) = (Real.sqrt 3 / 12) := 
by
  -- Proof goes here
  sorry

end simplify_fraction_l2044_204456
