import Mathlib

namespace target_avg_weekly_income_l411_41198

-- Define the weekly incomes for the past 5 weeks
def past_incomes : List ℤ := [406, 413, 420, 436, 395]

-- Define the average income over the next 2 weeks
def avg_income_next_two_weeks : ℤ := 365

-- Define the target average weekly income over the 7-week period
theorem target_avg_weekly_income : 
  ((past_incomes.sum + 2 * avg_income_next_two_weeks) / 7 = 400) :=
sorry

end target_avg_weekly_income_l411_41198


namespace min_value_of_m_l411_41102

open Real

-- Definitions from the conditions
def condition1 (m : ℝ) : Prop :=
  m > 0

def condition2 (m : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → 2 * exp (2 * m * x) - (log x) / m ≥ 0

-- The theorem statement for the minimum value of m
theorem min_value_of_m (m : ℝ) : condition1 m → condition2 m → m ≥ 1 / (2 * exp 1) := 
sorry

end min_value_of_m_l411_41102


namespace units_produced_today_eq_90_l411_41190

-- Define the average production and number of past days
def average_past_production (n : ℕ) (past_avg : ℕ) : ℕ :=
  n * past_avg

def average_total_production (n : ℕ) (current_avg : ℕ) : ℕ :=
  (n + 1) * current_avg

def units_produced_today (n : ℕ) (past_avg : ℕ) (current_avg : ℕ) : ℕ :=
  average_total_production n current_avg - average_past_production n past_avg

-- Given conditions
def n := 5
def past_avg := 60
def current_avg := 65

-- Statement to prove
theorem units_produced_today_eq_90 : units_produced_today n past_avg current_avg = 90 :=
by
  -- Declare which parts need proving
  sorry

end units_produced_today_eq_90_l411_41190


namespace necessary_but_not_sufficient_converse_implies_l411_41174

theorem necessary_but_not_sufficient (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * (Real.log x) ^ 2 < 1) → (x * Real.log x < 1) :=
sorry

theorem converse_implies (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * Real.log x < 1) → (x * (Real.log x) ^ 2 < 1) :=
sorry

end necessary_but_not_sufficient_converse_implies_l411_41174


namespace tan_identity_l411_41115

variable (α β : ℝ)

theorem tan_identity (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : Real.sin (2 * α) = 2 * Real.sin (2 * β)) : 
  Real.tan (α + β) = 3 * Real.tan (α - β) := 
by 
  sorry

end tan_identity_l411_41115


namespace intersection_of_A_and_B_l411_41127

open Set

theorem intersection_of_A_and_B (A B : Set ℕ) (hA : A = {1, 2, 4}) (hB : B = {2, 4, 6}) : A ∩ B = {2, 4} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  sorry

end intersection_of_A_and_B_l411_41127


namespace simplify_expression_l411_41195

variable (a : ℝ)

theorem simplify_expression (h₁ : a ≠ -3) (h₂ : a ≠ 1) :
  (1 - 4/(a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) :=
sorry

end simplify_expression_l411_41195


namespace complement_intersection_l411_41131

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

-- Define the complements of A and B with respect to U
def complement_U (s : Set ℕ) := { x ∈ U | x ∉ s }

-- The theorem to prove the intersection of the complements
theorem complement_intersection :
  (complement_U A) ∩ (complement_U B) = {7, 9} :=
sorry

end complement_intersection_l411_41131


namespace cards_probability_ratio_l411_41185

theorem cards_probability_ratio :
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  q / p = 44 :=
by
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  have : q / p = 44 := sorry
  exact this

end cards_probability_ratio_l411_41185


namespace third_sec_second_chap_more_than_first_sec_third_chap_l411_41120

-- Define the page lengths for each section in each chapter
def first_chapter : List ℕ := [20, 10, 30]
def second_chapter : List ℕ := [5, 12, 8, 22]
def third_chapter : List ℕ := [7, 11]

-- Define the specific sections of interest
def third_section_second_chapter := second_chapter[2]  -- 8
def first_section_third_chapter := third_chapter[0]   -- 7

-- The theorem we want to prove
theorem third_sec_second_chap_more_than_first_sec_third_chap :
  third_section_second_chapter - first_section_third_chapter = 1 :=
by
  -- Sorry is used here to skip the proof.
  sorry

end third_sec_second_chap_more_than_first_sec_third_chap_l411_41120


namespace min_sum_reciprocals_of_roots_l411_41194

theorem min_sum_reciprocals_of_roots (k : ℝ) 
  (h_roots_positive : ∀ x : ℝ, (x^2 - k * x + k + 3 = 0) → 0 < x) :
  (k ≥ 6) → 
  ∀ x1 x2 : ℝ, (x1*x2 = k + 3) ∧ (x1 + x2 = k) ∧ (x1 > 0) ∧ (x2 > 0) → 
  (1 / x1 + 1 / x2) = 2 / 3 :=
by 
  -- proof steps go here
  sorry

end min_sum_reciprocals_of_roots_l411_41194


namespace probability_300_feet_or_less_l411_41150

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

end probability_300_feet_or_less_l411_41150


namespace stratified_sampling_employees_over_50_l411_41165

theorem stratified_sampling_employees_over_50 :
  let total_employees := 500
  let employees_under_35 := 125
  let employees_35_to_50 := 280
  let employees_over_50 := 95
  let total_samples := 100
  (employees_over_50 / total_employees * total_samples) = 19 := by
  sorry

end stratified_sampling_employees_over_50_l411_41165


namespace average_of_xyz_l411_41171

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 :=
sorry

end average_of_xyz_l411_41171


namespace isosceles_right_triangle_area_l411_41104

theorem isosceles_right_triangle_area (a b : ℝ) (h₁ : a = b) (h₂ : a + b = 20) : 
  (1 / 2) * a * b = 50 := 
by 
  sorry

end isosceles_right_triangle_area_l411_41104


namespace programs_produce_same_result_l411_41179

-- Define Program A's computation
def programA_sum : ℕ := (List.range (1000 + 1)).sum -- Sum of numbers from 0 to 1000

-- Define Program B's computation
def programB_sum : ℕ := (List.range (1000 + 1)).reverse.sum -- Sum of numbers from 1000 down to 0

theorem programs_produce_same_result : programA_sum = programB_sum :=
  sorry

end programs_produce_same_result_l411_41179


namespace mixed_fractions_calculation_l411_41151

theorem mixed_fractions_calculation :
  2017 + (2016 / 2017) / (2019 + (1 / 2016)) + (1 / 2017) = 1 :=
by
  sorry

end mixed_fractions_calculation_l411_41151


namespace number_of_bottom_row_bricks_l411_41170

theorem number_of_bottom_row_bricks :
  ∃ (x : ℕ), (x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) ∧ x = 22 :=
by 
  sorry

end number_of_bottom_row_bricks_l411_41170


namespace team_won_five_games_l411_41137
-- Import the entire Mathlib library

-- Number of games played (given as a constant)
def numberOfGamesPlayed : ℕ := 10

-- Number of losses definition based on the ratio condition
def numberOfLosses : ℕ := numberOfGamesPlayed / 2

-- The number of wins is defined as the total games played minus the number of losses
def numberOfWins : ℕ := numberOfGamesPlayed - numberOfLosses

-- Proof statement: The number of wins is 5
theorem team_won_five_games :
  numberOfWins = 5 := by
  sorry

end team_won_five_games_l411_41137


namespace geom_seq_fraction_l411_41143

theorem geom_seq_fraction (a_1 a_2 a_3 a_4 a_5 q : ℝ)
  (h1 : q > 0)
  (h2 : a_2 = q * a_1)
  (h3 : a_3 = q^2 * a_1)
  (h4 : a_4 = q^3 * a_1)
  (h5 : a_5 = q^4 * a_1)
  (h_arith : a_2 - (1/2) * a_3 = (1/2) * a_3 - a_1) :
  (a_3 + a_4) / (a_4 + a_5) = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end geom_seq_fraction_l411_41143


namespace range_of_a_for_empty_solution_set_l411_41111

theorem range_of_a_for_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, ¬ (|x - 3| + |x - 4| < a)) ↔ a ≤ 1 := 
sorry

end range_of_a_for_empty_solution_set_l411_41111


namespace solution_set_absolute_value_l411_41126

theorem solution_set_absolute_value (x : ℝ) : 
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  -- Proof goes here
  sorry

end solution_set_absolute_value_l411_41126


namespace greatest_ab_sum_l411_41191

theorem greatest_ab_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) :
  a + b = Real.sqrt 220 ∨ a + b = -Real.sqrt 220 :=
sorry

end greatest_ab_sum_l411_41191


namespace children_count_l411_41175

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l411_41175


namespace polynomial_factors_l411_41167

theorem polynomial_factors (h k : ℤ)
  (h1 : 3 * (-2)^4 - 2 * h * (-2)^2 + h * (-2) + k = 0)
  (h2 : 3 * 1^4 - 2 * h * 1^2 + h * 1 + k = 0)
  (h3 : 3 * (-3)^4 - 2 * h * (-3)^2 + h * (-3) + k = 0) :
  |3 * h - 2 * k| = 11 :=
by
  sorry

end polynomial_factors_l411_41167


namespace numCounterexamplesCorrect_l411_41135

-- Define a function to calculate the sum of digits of a number
def digitSum (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Predicate to check if a number is prime
def isPrime (n : Nat) : Prop := 
  Nat.Prime n

-- Set definition where the sum of digits must be 5 and all digits are non-zero
def validSet (n : Nat) : Prop :=
  digitSum n = 5 ∧ ∀ d ∈ n.digits 10, d ≠ 0

-- Define the number of counterexamples
def numCounterexamples : Nat := 6

-- The final theorem stating the number of counterexamples
theorem numCounterexamplesCorrect :
  (∃ ns : Finset Nat, 
    (∀ n ∈ ns, validSet n) ∧ 
    (∀ n ∈ ns, ¬ isPrime n) ∧ 
    ns.card = numCounterexamples) :=
sorry

end numCounterexamplesCorrect_l411_41135


namespace compute_expression_l411_41128

theorem compute_expression :
  6 * (2 / 3)^4 - 1 / 6 = 55 / 54 :=
by
  sorry

end compute_expression_l411_41128


namespace damaged_books_count_l411_41103

variables (o d : ℕ)

theorem damaged_books_count (h1 : o + d = 69) (h2 : o = 6 * d - 8) : d = 11 := 
by 
  sorry

end damaged_books_count_l411_41103


namespace ratio_t_q_l411_41148

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end ratio_t_q_l411_41148


namespace find_a_l411_41130

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : deriv (f a) (-1) = 4) : 
  a = 10 / 3 :=
sorry

end find_a_l411_41130


namespace line_l_statements_correct_l411_41112

theorem line_l_statements_correct
  (A B C : ℝ)
  (hAB : ¬(A = 0 ∧ B = 0)) :
  ( (2 * A + B + C = 0 → ∀ x y, A * (x - 2) + B * (y - 1) = 0 ↔ A * x + B * y + C = 0 ) ∧
    ((A ≠ 0 ∧ B ≠ 0) → ∃ x, A * x + C = 0 ∧ ∃ y, B * y + C = 0) ∧
    (A = 0 ∧ B ≠ 0 ∧ C ≠ 0 → ∀ y, B * y + C = 0 ↔ y = -C / B) ∧
    (A ≠ 0 ∧ B^2 + C^2 = 0 → ∀ x, A * x = 0 ↔ x = 0) ) :=
by
  sorry

end line_l_statements_correct_l411_41112


namespace fifteenth_number_in_base_8_l411_41164

theorem fifteenth_number_in_base_8 : (15 : ℕ) = 1 * 8 + 7 := 
sorry

end fifteenth_number_in_base_8_l411_41164


namespace polynomial_divisible_by_24_l411_41119

theorem polynomial_divisible_by_24 (n : ℤ) : 24 ∣ (n^4 + 6 * n^3 + 11 * n^2 + 6 * n) :=
sorry

end polynomial_divisible_by_24_l411_41119


namespace union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l411_41162

open Set

def setA : Set ℝ := {x | -4 < x ∧ x < 2}
def setB : Set ℝ := {x | x < -5 ∨ x > 1}
def setComplementB : Set ℝ := {x | -5 ≤ x ∧ x ≤ 1}

theorem union_of_A_and_B : setA ∪ setB = {x | x < -5 ∨ x > -4} := by
  sorry

theorem intersection_of_A_and_complementB : setA ∩ setComplementB = {x | -4 < x ∧ x ≤ 1} := by
  sorry

noncomputable def setC (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

theorem range_of_m (m : ℝ) (h : setB ∩ (setC m) = ∅) : -4 ≤ m ∧ m ≤ 0 := by
  sorry

end union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l411_41162


namespace solve_quadratic_expr_l411_41125

theorem solve_quadratic_expr (x : ℝ) (h : 2 * x^2 - 5 = 11) : 
  4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2 ∨ 4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2 := 
by 
  sorry

end solve_quadratic_expr_l411_41125


namespace roots_of_equation_l411_41109

theorem roots_of_equation (a x : ℝ) : x * (x + 5)^2 * (a - x) = 0 ↔ (x = 0 ∨ x = -5 ∨ x = a) :=
by
  sorry

end roots_of_equation_l411_41109


namespace average_age_of_large_family_is_correct_l411_41169

def average_age_of_family 
  (num_grandparents : ℕ) (avg_age_grandparents : ℕ) 
  (num_parents : ℕ) (avg_age_parents : ℕ) 
  (num_children : ℕ) (avg_age_children : ℕ) 
  (num_siblings : ℕ) (avg_age_siblings : ℕ)
  (num_cousins : ℕ) (avg_age_cousins : ℕ)
  (num_aunts : ℕ) (avg_age_aunts : ℕ) : ℕ := 
  let total_age := num_grandparents * avg_age_grandparents + 
                   num_parents * avg_age_parents + 
                   num_children * avg_age_children + 
                   num_siblings * avg_age_siblings + 
                   num_cousins * avg_age_cousins + 
                   num_aunts * avg_age_aunts
  let total_family_members := num_grandparents + num_parents + num_children + num_siblings + num_cousins + num_aunts
  (total_age : ℕ) / total_family_members

theorem average_age_of_large_family_is_correct :
  average_age_of_family 4 67 3 41 5 8 2 35 3 22 2 45 = 35 := 
by 
  sorry

end average_age_of_large_family_is_correct_l411_41169


namespace eval_expression_l411_41160

theorem eval_expression : (20 - 16) * (12 + 8) / 4 = 20 := 
by 
  sorry

end eval_expression_l411_41160


namespace sum_of_roots_of_quadratic_l411_41181

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : 2 * (X^2) - 8 * X + 6 = 0) : 
  (-b / a) = 4 :=
sorry

end sum_of_roots_of_quadratic_l411_41181


namespace function_in_second_quadrant_l411_41123

theorem function_in_second_quadrant (k : ℝ) : (∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → (k / x₁ < k / x₂)) → (∀ x : ℝ, x < 0 → (k > 0)) :=
sorry

end function_in_second_quadrant_l411_41123


namespace prime_sum_divisible_l411_41117

theorem prime_sum_divisible (p : Fin 2021 → ℕ) (prime : ∀ i, Nat.Prime (p i))
  (h : 6060 ∣ Finset.univ.sum (fun i => (p i)^4)) : 4 ≤ Finset.card (Finset.univ.filter (fun i => p i < 2021)) :=
sorry

end prime_sum_divisible_l411_41117


namespace expression_eval_l411_41129

theorem expression_eval :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
by
  sorry

end expression_eval_l411_41129


namespace potatoes_leftover_l411_41121

-- Define the necessary conditions
def fries_per_potato : ℕ := 25
def total_potatoes : ℕ := 15
def fries_needed : ℕ := 200

-- Prove the goal
theorem potatoes_leftover : total_potatoes - (fries_needed / fries_per_potato) = 7 :=
sorry

end potatoes_leftover_l411_41121


namespace fish_remaining_when_discovered_l411_41144

def start_fish := 60
def fish_eaten_per_day := 2
def days_two_weeks := 2 * 7
def fish_added_after_two_weeks := 8
def days_one_week := 7

def fish_after_two_weeks (start: ℕ) (eaten_per_day: ℕ) (days: ℕ) (added: ℕ): ℕ :=
  start - eaten_per_day * days + added

def fish_after_three_weeks (fish_after_two_weeks: ℕ) (eaten_per_day: ℕ) (days: ℕ): ℕ :=
  fish_after_two_weeks - eaten_per_day * days

theorem fish_remaining_when_discovered :
  (fish_after_three_weeks (fish_after_two_weeks start_fish fish_eaten_per_day days_two_weeks fish_added_after_two_weeks) fish_eaten_per_day days_one_week) = 26 := 
by {
  sorry
}

end fish_remaining_when_discovered_l411_41144


namespace conic_sections_l411_41188

theorem conic_sections (x y : ℝ) (h : y^4 - 6 * x^4 = 3 * y^2 - 2) :
  (∃ a b : ℝ, y^2 = a + b * x^2) ∨ (∃ c d : ℝ, y^2 = c - d * x^2) :=
sorry

end conic_sections_l411_41188


namespace volunteers_distribution_l411_41199

theorem volunteers_distribution:
  let num_volunteers := 5
  let group_distribution := (2, 2, 1)
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end volunteers_distribution_l411_41199


namespace B_completes_work_in_n_days_l411_41178

-- Define the conditions
def can_complete_work_A_in_d_days (d : ℕ) : Prop := d = 15
def fraction_of_work_left_after_working_together (t : ℕ) (fraction : ℝ) : Prop :=
  t = 5 ∧ fraction = 0.41666666666666663

-- Define the theorem to be proven
theorem B_completes_work_in_n_days (d t : ℕ) (fraction : ℝ) (x : ℕ) 
  (hA : can_complete_work_A_in_d_days d) 
  (hB : fraction_of_work_left_after_working_together t fraction) : x = 20 :=
sorry

end B_completes_work_in_n_days_l411_41178


namespace range_of_a_l411_41149

theorem range_of_a (a : ℝ) (h_decreasing : ∀ x y : ℝ, x < y → (a-1)^x > (a-1)^y) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l411_41149


namespace sum_of_x_coordinates_where_g_eq_2_5_l411_41113

def g1 (x : ℝ) : ℝ := 3 * x + 6
def g2 (x : ℝ) : ℝ := -x + 2
def g3 (x : ℝ) : ℝ := 2 * x - 2
def g4 (x : ℝ) : ℝ := -2 * x + 8

def is_within (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

theorem sum_of_x_coordinates_where_g_eq_2_5 :
     (∀ x, g1 x = 2.5 → (is_within x (-4) (-2) → false)) ∧
     (∀ x, g2 x = 2.5 → (is_within x (-2) (0) → x = -0.5)) ∧
     (∀ x, g3 x = 2.5 → (is_within x 0 3 → x = 2.25)) ∧
     (∀ x, g4 x = 2.5 → (is_within x 3 5 → x = 2.75)) →
     (-0.5 + 2.25 + 2.75 = 4.5) :=
by { sorry }

end sum_of_x_coordinates_where_g_eq_2_5_l411_41113


namespace books_difference_l411_41186

theorem books_difference (maddie_books luisa_books amy_books total_books : ℕ) 
  (h1 : maddie_books = 15) 
  (h2 : luisa_books = 18) 
  (h3 : amy_books = 6) 
  (h4 : total_books = amy_books + luisa_books) :
  total_books - maddie_books = 9 := 
sorry

end books_difference_l411_41186


namespace shirts_count_l411_41140

theorem shirts_count (S : ℕ) (hours_per_shirt hours_per_pant cost_per_hour total_pants total_cost : ℝ) :
  hours_per_shirt = 1.5 →
  hours_per_pant = 3 →
  cost_per_hour = 30 →
  total_pants = 12 →
  total_cost = 1530 →
  45 * S + 1080 = total_cost →
  S = 10 :=
by
  intros hps hpp cph tp tc cost_eq
  sorry

end shirts_count_l411_41140


namespace total_marbles_l411_41106

theorem total_marbles (y b g : ℝ) (h1 : y = 1.4 * b) (h2 : g = 1.75 * y) :
  b + y + g = 3.4643 * y :=
sorry

end total_marbles_l411_41106


namespace time_difference_l411_41156

/-
Malcolm's speed: 5 minutes per mile
Joshua's speed: 7 minutes per mile
Race length: 12 miles
Question: Prove that the time difference between Joshua crossing the finish line after Malcolm is 24 minutes
-/
noncomputable def time_taken (speed: ℕ) (distance: ℕ) : ℕ :=
  speed * distance

theorem time_difference :
  let malcolm_speed := 5
  let joshua_speed := 7
  let race_length := 12
  let malcolm_time := time_taken malcolm_speed race_length
  let joshua_time := time_taken joshua_speed race_length
  malcolm_time < joshua_time →
  joshua_time - malcolm_time = 24 :=
by
  intros malcolm_speed joshua_speed race_length malcolm_time joshua_time malcolm_time_lt_joshua_time
  sorry

end time_difference_l411_41156


namespace set_operations_l411_41138

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6})
variable (hA : A = {2, 4, 5})
variable (hB : B = {1, 2, 5})

theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) :=
by
  sorry

end set_operations_l411_41138


namespace min_value_exp_l411_41142

theorem min_value_exp (x y : ℝ) (h : x + 2 * y = 4) : ∃ z : ℝ, (2^x + 4^y = z) ∧ (∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ z) :=
sorry

end min_value_exp_l411_41142


namespace share_of_c_l411_41114

variable (a b c : ℝ)

theorem share_of_c (h1 : a + b + c = 427) (h2 : 3 * a = 7 * c) (h3 : 4 * b = 7 * c) : c = 84 :=
  by
  sorry

end share_of_c_l411_41114


namespace angle_E_measure_l411_41184

-- Definition of degrees for each angle in the quadrilateral
def angle_measure (E F G H : ℝ) : Prop :=
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360

-- Prove the measure of angle E
theorem angle_E_measure (E F G H : ℝ) (h : angle_measure E F G H) : E = 360 * (4 / 7) :=
by {
  sorry
}

end angle_E_measure_l411_41184


namespace minimum_value_expression_l411_41147

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l411_41147


namespace find_b_l411_41192

theorem find_b (b : ℝ) (x : ℝ) (hx : x^2 + b * x - 45 = 0) (h_root : x = -5) : b = -4 :=
by
  sorry

end find_b_l411_41192


namespace minimum_value_of_fm_plus_fp_l411_41172

def f (x a : ℝ) : ℝ := -x^3 + a * x^2 - 4

def f_prime (x a : ℝ) : ℝ := -3 * x^2 + 2 * a * x

theorem minimum_value_of_fm_plus_fp (a : ℝ) (h_extremum : f_prime 2 a = 0) (m n : ℝ) 
  (hm : -1 ≤ m ∧ m ≤ 1) (hn : -1 ≤ n ∧ n ≤ 1) : 
  f m a + f_prime n a = -13 := 
by
  -- steps of the proof would go here
  sorry

end minimum_value_of_fm_plus_fp_l411_41172


namespace owen_turtles_l411_41187

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end owen_turtles_l411_41187


namespace crosswalk_red_light_wait_l411_41105

theorem crosswalk_red_light_wait :
  let red_light_duration := 40
  let wait_time_requirement := 15
  let favorable_duration := red_light_duration - wait_time_requirement
  (favorable_duration : ℝ) / red_light_duration = (5 : ℝ) / 8 :=
by
  sorry

end crosswalk_red_light_wait_l411_41105


namespace minimum_value_of_f_l411_41139

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 :=
by
  sorry

end minimum_value_of_f_l411_41139


namespace largest_divisor_of_n4_minus_n2_l411_41134

theorem largest_divisor_of_n4_minus_n2 :
  ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_of_n4_minus_n2_l411_41134


namespace initial_incorrect_average_l411_41122

theorem initial_incorrect_average :
  let avg_correct := 24
  let incorrect_insertion := 26
  let correct_insertion := 76
  let n := 10  
  let correct_sum := avg_correct * n
  let incorrect_sum := correct_sum - correct_insertion + incorrect_insertion   
  avg_correct * n - correct_insertion + incorrect_insertion = incorrect_sum →
  incorrect_sum / n = 19 :=
by 
  sorry

end initial_incorrect_average_l411_41122


namespace original_height_in_feet_l411_41163

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

end original_height_in_feet_l411_41163


namespace problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l411_41176

variable (a b : ℝ)

theorem problem_statement_part1 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a + 2 / b) ≥ 9 := sorry

theorem problem_statement_part2 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (2 ^ a + 4 ^ b) ≥ 2 * Real.sqrt 2 := sorry

theorem problem_statement_part3 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a * b) ≤ (1 / 8) := sorry

theorem problem_statement_part4 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2) ≥ (1 / 5) := sorry

end problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l411_41176


namespace find_y_l411_41110

theorem find_y (x : ℝ) (y : ℝ) (h : (3 + y)^5 = (1 + 3 * y)^4) (hx : x = 1.5) : y = 1.5 :=
by
  -- Proof steps go here
  sorry

end find_y_l411_41110


namespace car_speed_conversion_l411_41116

theorem car_speed_conversion :
  let speed_mps := 10 -- speed of the car in meters per second
  let conversion_factor := 3.6 -- conversion factor from m/s to km/h
  let speed_kmph := speed_mps * conversion_factor -- speed of the car in kilometers per hour
  speed_kmph = 36 := 
by
  sorry

end car_speed_conversion_l411_41116


namespace jenny_sold_boxes_l411_41161

-- Given conditions as definitions
def cases : ℕ := 3
def boxes_per_case : ℕ := 8

-- Mathematically equivalent proof problem
theorem jenny_sold_boxes : cases * boxes_per_case = 24 := by
  sorry

end jenny_sold_boxes_l411_41161


namespace max_length_PC_l411_41118

-- Define the circle C and its properties
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

-- The equilateral triangle condition and what we need to prove
theorem max_length_PC :
  (∃ (P A B : ℝ × ℝ), 
    (Circle A.1 A.2) ∧
    (Circle B.1 B.2) ∧
    (Circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2)) ∧
    (A ≠ B) ∧
    (∃ r : ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧ 
               (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2 ∧ 
               (P.1 - B.1)^2 + (P.2 - B.2)^2 = r^2)) → 
  (∀ (P : ℝ × ℝ), 
     ∃ (max_val : ℝ), max_val = 4 ∧
     (¬(∃ (Q : ℝ × ℝ), (Circle P.1 P.2) ∧ ((Q.1 - 0)^2 + (Q.2 - 1)^2 > max_val^2))))
:= 
sorry

end max_length_PC_l411_41118


namespace total_combined_area_l411_41107

-- Definition of the problem conditions
def base_parallelogram : ℝ := 20
def height_parallelogram : ℝ := 4
def base_triangle : ℝ := 20
def height_triangle : ℝ := 2

-- Given the conditions, we want to prove:
theorem total_combined_area :
  (base_parallelogram * height_parallelogram) + (0.5 * base_triangle * height_triangle) = 100 :=
by
  sorry  -- proof goes here

end total_combined_area_l411_41107


namespace quadratic_equation_only_option_B_l411_41177

theorem quadratic_equation_only_option_B (a b c : ℝ) (x : ℝ):
  (a ≠ 0 → (a * x^2 + b * x + c = 0)) ∧              -- Option A
  (3 * (x + 1)^2 = 2 * (x - 2) ↔ 3 * x^2 + 4 * x + 7 = 0) ∧  -- Option B
  (1 / x^2 + 1 = x^2 + 1 → False) ∧         -- Option C
  (1 / x^2 + 1 / x - 2 = 0 → False) →       -- Option D
  -- Option B is the only quadratic equation.
  (3 * (x + 1)^2 = 2 * (x - 2)) :=
sorry

end quadratic_equation_only_option_B_l411_41177


namespace abs_gt_not_implies_gt_l411_41108

noncomputable def abs_gt_implies_gt (a b : ℝ) : Prop :=
  |a| > |b| → a > b

theorem abs_gt_not_implies_gt (a b : ℝ) :
  ¬ abs_gt_implies_gt a b :=
sorry

end abs_gt_not_implies_gt_l411_41108


namespace total_food_per_day_l411_41193

theorem total_food_per_day 
  (first_soldiers : ℕ)
  (second_soldiers : ℕ)
  (food_first_side_per_soldier : ℕ)
  (food_second_side_per_soldier : ℕ) :
  first_soldiers = 4000 →
  second_soldiers = first_soldiers - 500 →
  food_first_side_per_soldier = 10 →
  food_second_side_per_soldier = food_first_side_per_soldier - 2 →
  (first_soldiers * food_first_side_per_soldier + second_soldiers * food_second_side_per_soldier = 68000) :=
by
  intros h1 h2 h3 h4
  sorry

end total_food_per_day_l411_41193


namespace area_ratio_of_circles_l411_41180

-- Define the circles and lengths of arcs
variables {R_C R_D : ℝ} (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D))

-- Theorem proving the ratio of the areas
theorem area_ratio_of_circles (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 := sorry

end area_ratio_of_circles_l411_41180


namespace multiple_of_15_bounds_and_difference_l411_41197

theorem multiple_of_15_bounds_and_difference :
  ∃ (n : ℕ), 15 * n ≤ 2016 ∧ 2016 < 15 * (n + 1) ∧ (15 * (n + 1) - 2016) = 9 :=
by
  sorry

end multiple_of_15_bounds_and_difference_l411_41197


namespace find_actual_weights_l411_41183

noncomputable def melon_weight : ℝ := 4.5
noncomputable def watermelon_weight : ℝ := 3.5
noncomputable def scale_error : ℝ := 0.5

def weight_bounds (actual_weight measured_weight error_margin : ℝ) :=
  (measured_weight - error_margin ≤ actual_weight) ∧ (actual_weight ≤ measured_weight + error_margin)

theorem find_actual_weights (x y : ℝ) 
  (melon_measured : x = 4)
  (watermelon_measured : y = 3)
  (combined_measured : x + y = 8.5)
  (hx : weight_bounds melon_weight x scale_error)
  (hy : weight_bounds watermelon_weight y scale_error)
  (h_combined : weight_bounds (melon_weight + watermelon_weight) (x + y) (2 * scale_error)) :
  x = melon_weight ∧ y = watermelon_weight := 
sorry

end find_actual_weights_l411_41183


namespace convert_to_base7_l411_41141

theorem convert_to_base7 : 3589 = 1 * 7^4 + 3 * 7^3 + 3 * 7^2 + 1 * 7^1 + 5 * 7^0 :=
by
  sorry

end convert_to_base7_l411_41141


namespace problem_part1_problem_part2_l411_41152

variable (A B : Set ℝ)
def C_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem problem_part1 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  C_R (A ∩ B) = { x : ℝ | x < 3 ∨ x ≥ 6 } :=
by
  intros hA hB
  sorry

theorem problem_part2 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  (C_R B) ∪ A = { x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by
  intros hA hB
  sorry

end problem_part1_problem_part2_l411_41152


namespace masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l411_41145

/-- 
Part (a): Define the problem statement where, given the iterative process on a number,
it stabilizes at 17.
-/
theorem masha_final_number_stabilizes (x y : ℕ) (n : ℕ) (h_stable : ∀ x y, 10 * x + y = 3 * x + 2 * y) :
  n = 17 :=
by
  sorry

/--
Part (b): Define the problem statement to find the smallest 2015-digit number ending with the
digits 09 that eventually stabilizes to 17.
-/
theorem masha_smallest_initial_number_ends_with_09 :
  ∃ (n : ℕ), n ≥ 10^2014 ∧ n % 100 = 9 ∧ (∃ k : ℕ, 10^2014 + k = n ∧ (10 * ((n - k) / 10) + (n % 10)) = 17) :=
by
  sorry

end masha_final_number_stabilizes_masha_smallest_initial_number_ends_with_09_l411_41145


namespace parallelogram_midpoints_XY_square_l411_41153

theorem parallelogram_midpoints_XY_square (A B C D X Y : ℝ)
  (AB CD : ℝ) (BC DA : ℝ) (angle_D : ℝ)
  (mid_X : X = (B + C) / 2) (mid_Y : Y = (D + A) / 2)
  (h1: AB = 10) (h2: BC = 17) (h3: CD = 10) (h4 : angle_D = 60) :
  (XY ^ 2 = 219 / 4) :=
by
  sorry

end parallelogram_midpoints_XY_square_l411_41153


namespace haley_money_difference_l411_41159

def initial_amount : ℕ := 2
def chores : ℕ := 5
def birthday : ℕ := 10
def neighbor : ℕ := 7
def candy : ℕ := 3
def lost : ℕ := 2

theorem haley_money_difference : (initial_amount + chores + birthday + neighbor - candy - lost) - initial_amount = 17 := by
  sorry

end haley_money_difference_l411_41159


namespace iter_f_eq_l411_41154

namespace IteratedFunction

def f (n : ℕ) (x : ℕ) : ℕ :=
  if 2 * x <= n then
    2 * x
  else
    2 * n - 2 * x + 1

def iter_f (n m : ℕ) (x : ℕ) : ℕ :=
  (Nat.iterate (f n) m) x

variables (n m : ℕ) (S : Fin n.succ → Fin n.succ)

theorem iter_f_eq (h : iter_f n m 1 = 1) (k : Fin n.succ) :
  iter_f n m k = k := by
  sorry

end IteratedFunction

end iter_f_eq_l411_41154


namespace jamie_school_distance_l411_41100

theorem jamie_school_distance
  (v : ℝ) -- usual speed in miles per hour
  (d : ℝ) -- distance to school in miles
  (h1 : (20 : ℝ) / 60 = 1 / 3) -- usual time to school in hours
  (h2 : (10 : ℝ) / 60 = 1 / 6) -- lighter traffic time in hours
  (h3 : d = v * (1 / 3)) -- distance equation for usual traffic
  (h4 : d = (v + 15) * (1 / 6)) -- distance equation for lighter traffic
  : d = 5 := by
  sorry

end jamie_school_distance_l411_41100


namespace rational_expr_evaluation_l411_41132

theorem rational_expr_evaluation (a b c : ℚ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) (h2 : a + b + c = a * b * c) :
  (a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a) = -3 :=
by
  sorry

end rational_expr_evaluation_l411_41132


namespace calculate_molecular_weight_l411_41157

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def num_atoms_C := 3
def num_atoms_H := 6
def num_atoms_O := 1

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  (nC * wC) + (nH * wH) + (nO * wO)

theorem calculate_molecular_weight :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 58.078 :=
by
  sorry

end calculate_molecular_weight_l411_41157


namespace original_number_input_0_2_l411_41133

theorem original_number_input_0_2 (x : ℝ) (hx : x ≠ 0) (h : (1 / (1 / x - 1) - 1 = -0.75)) : x = 0.2 := 
sorry

end original_number_input_0_2_l411_41133


namespace plane_equation_proof_l411_41155

-- Define the parametric representation of the plane
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - t, 1 + 2 * s, 4 - s + 3 * t)

-- Define the plane equation form
def plane_equation (x y z : ℝ) (A B C D : ℤ) : Prop :=
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0

-- Define the normal vector derived from the cross product
def normal_vector : ℝ × ℝ × ℝ := (6, -5, 2)

-- Define the initial point used to calculate D
def initial_point : ℝ × ℝ × ℝ := (2, 1, 4)

-- Proposition to prove the equation of the plane
theorem plane_equation_proof :
  ∃ (A B C D : ℤ), A = 6 ∧ B = -5 ∧ C = 2 ∧ D = -15 ∧
    ∀ x y z : ℝ, plane_equation x y z A B C D ↔
      ∃ s t : ℝ, plane_parametric s t = (x, y, z) :=
by
  sorry

end plane_equation_proof_l411_41155


namespace max_value_a_l411_41158

theorem max_value_a (a b c d : ℝ) 
  (h1 : a ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : b ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : c ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h4 : d ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h5 : Real.sin a + Real.sin b + Real.sin c + Real.sin d = 1)
  (h6 : Real.cos (2 * a) + Real.cos (2 * b) + Real.cos (2 * c) + Real.cos (2 * d) ≥ 10 / 3) : 
  a ≤ Real.arcsin (1 / 2) := 
sorry

end max_value_a_l411_41158


namespace pow137_mod8_l411_41189

theorem pow137_mod8 : (5 ^ 137) % 8 = 5 := by
  -- Use the provided conditions
  have h1: 5 % 8 = 5 := by norm_num
  have h2: (5 ^ 2) % 8 = 1 := by norm_num
  sorry

end pow137_mod8_l411_41189


namespace ab_value_is_3360_l411_41166

noncomputable def find_ab (a b : ℤ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧
  (∃ r s : ℤ, 
    (x : ℤ) → 
      (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s)) ∧ 
      (2 * r + s = -a) ∧ 
      (r^2 + 2 * r * s = b) ∧ 
      (r^2 * s = -16 * a))

theorem ab_value_is_3360 (a b : ℤ) (h : find_ab a b) : |a * b| = 3360 :=
sorry

end ab_value_is_3360_l411_41166


namespace find_friends_l411_41136

-- Definitions
def shells_Jillian : Nat := 29
def shells_Savannah : Nat := 17
def shells_Clayton : Nat := 8
def shells_per_friend : Nat := 27

-- Main statement
theorem find_friends :
  (shells_Jillian + shells_Savannah + shells_Clayton) / shells_per_friend = 2 :=
by
  sorry

end find_friends_l411_41136


namespace least_three_digit_multiple_of_3_4_5_l411_41196

def is_multiple_of (a b : ℕ) : Prop := b % a = 0

theorem least_three_digit_multiple_of_3_4_5 : 
  ∃ n : ℕ, is_multiple_of 3 n ∧ is_multiple_of 4 n ∧ is_multiple_of 5 n ∧ 100 ≤ n ∧ n < 1000 ∧ (∀ m : ℕ, is_multiple_of 3 m ∧ is_multiple_of 4 m ∧ is_multiple_of 5 m ∧ 100 ≤ m ∧ m < 1000 → n ≤ m) ∧ n = 120 :=
by
  sorry

end least_three_digit_multiple_of_3_4_5_l411_41196


namespace problem_statement_l411_41124

def are_collinear (A B C : Point) : Prop := sorry -- Definition for collinearity should be expanded.
def area (A B C : Point) : ℝ := sorry -- Definition for area must be provided.

theorem problem_statement :
  ∀ n : ℕ, (n > 3) →
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ¬ are_collinear (A i) (A j) (A k)) ∧
    (∀ i j k : Fin n, area (A i) (A j) (A k) = r i + r j + r k)) →
  n = 4 :=
by sorry

end problem_statement_l411_41124


namespace ted_age_l411_41146

theorem ted_age (t s : ℝ) 
  (h1 : t = 3 * s - 20) 
  (h2: t + s = 70) : 
  t = 47.5 := 
by
  sorry

end ted_age_l411_41146


namespace mr_thompson_third_score_is_78_l411_41101

theorem mr_thompson_third_score_is_78 :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
                   (a = 58 ∧ b = 65 ∧ c = 70 ∧ d = 78) ∧ 
                   (a + b + c + d) % 4 = 3 ∧ 
                   (∀ i j k, (a + i + j + k) % 4 = 0) ∧ -- This checks that average is integer
                   c = 78 := sorry

end mr_thompson_third_score_is_78_l411_41101


namespace part1_solution_set_l411_41182

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part1_solution_set : {x : ℝ | f x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end part1_solution_set_l411_41182


namespace total_water_in_containers_l411_41173

/-
We have four containers. The first three contain water, while the fourth is empty. 
The second container holds twice as much water as the first, and the third holds twice as much water as the second. 
We transfer half of the water from the first container, one-third of the water from the second container, 
and one-quarter of the water from the third container into the fourth container. 
Now, there are 26 liters of water in the fourth container. Prove that initially, 
there were 84 liters of water in total in the first three containers.
-/

theorem total_water_in_containers (x : ℕ) (h1 : x / 2 + 2 * x / 3 + x = 26) : x + 2 * x + 4 * x = 84 := 
sorry

end total_water_in_containers_l411_41173


namespace original_number_of_students_l411_41168

theorem original_number_of_students (x : ℕ)
  (h1: 40 * x / x = 40)
  (h2: 12 * 34 = 408)
  (h3: (40 * x + 408) / (x + 12) = 36) : x = 6 :=
by
  sorry

end original_number_of_students_l411_41168
