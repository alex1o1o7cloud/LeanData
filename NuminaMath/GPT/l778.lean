import Mathlib

namespace NUMINAMATH_GPT_total_flowers_in_vase_l778_77877

-- Conditions as definitions
def num_roses : ℕ := 5
def num_lilies : ℕ := 2

-- Theorem statement
theorem total_flowers_in_vase : num_roses + num_lilies = 7 :=
by
  sorry

end NUMINAMATH_GPT_total_flowers_in_vase_l778_77877


namespace NUMINAMATH_GPT_quincy_sold_more_than_jake_l778_77895

theorem quincy_sold_more_than_jake :
  ∀ (T Jake : ℕ), Jake = 2 * T + 15 → 4000 = 100 * (T + Jake) → 4000 - Jake = 3969 :=
by
  intros T Jake hJake hQuincy
  sorry

end NUMINAMATH_GPT_quincy_sold_more_than_jake_l778_77895


namespace NUMINAMATH_GPT_WR_eq_35_l778_77880

theorem WR_eq_35 (PQ ZY SX : ℝ) (hPQ : PQ = 30) (hZY : ZY = 15) (hSX : SX = 10) :
    let WS := ZY - SX
    let SR := PQ
    let WR := WS + SR
    WR = 35 := by
  sorry

end NUMINAMATH_GPT_WR_eq_35_l778_77880


namespace NUMINAMATH_GPT_contrapositive_proposition_l778_77820

theorem contrapositive_proposition {a b : ℝ} :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) → (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
sorry

end NUMINAMATH_GPT_contrapositive_proposition_l778_77820


namespace NUMINAMATH_GPT_mary_lambs_count_l778_77818

def initial_lambs : Nat := 6
def baby_lambs : Nat := 2 * 2
def traded_lambs : Nat := 3
def extra_lambs : Nat := 7

theorem mary_lambs_count : initial_lambs + baby_lambs - traded_lambs + extra_lambs = 14 := by
  sorry

end NUMINAMATH_GPT_mary_lambs_count_l778_77818


namespace NUMINAMATH_GPT_quadratic_factorization_l778_77806

theorem quadratic_factorization (C D : ℤ) (h : (15 * y^2 - 74 * y + 48) = (C * y - 16) * (D * y - 3)) :
  C * D + C = 20 :=
sorry

end NUMINAMATH_GPT_quadratic_factorization_l778_77806


namespace NUMINAMATH_GPT_find_certain_number_l778_77829

theorem find_certain_number (x : ℝ) (h : 34 = (4/5) * x + 14) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l778_77829


namespace NUMINAMATH_GPT_union_of_sets_l778_77852

def M : Set ℝ := {x | x^2 + 2 * x = 0}

def N : Set ℝ := {x | x^2 - 2 * x = 0}

theorem union_of_sets : M ∪ N = {x | x = -2 ∨ x = 0 ∨ x = 2} := sorry

end NUMINAMATH_GPT_union_of_sets_l778_77852


namespace NUMINAMATH_GPT_plywood_perimeter_difference_l778_77861

theorem plywood_perimeter_difference :
  let l := 10
  let w := 6
  let n := 6
  ∃ p_max p_min, 
    (l * w) % n = 0 ∧
    (p_max = 24) ∧
    (p_min = 12.66) ∧
    p_max - p_min = 11.34 := 
by
  sorry

end NUMINAMATH_GPT_plywood_perimeter_difference_l778_77861


namespace NUMINAMATH_GPT_squared_expression_l778_77866

variable {x y : ℝ}

theorem squared_expression (x y : ℝ) : (-3 * x^2 * y)^2 = 9 * x^4 * y^2 :=
  by
  sorry

end NUMINAMATH_GPT_squared_expression_l778_77866


namespace NUMINAMATH_GPT_student_ticket_cost_l778_77803

theorem student_ticket_cost :
  ∀ (S : ℤ),
  (525 - 388) * S + 388 * 6 = 2876 → S = 4 :=
by
  sorry

end NUMINAMATH_GPT_student_ticket_cost_l778_77803


namespace NUMINAMATH_GPT_cos_double_angle_l778_77805

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l778_77805


namespace NUMINAMATH_GPT_ratio_is_correct_l778_77889

-- Define the constants
def total_students : ℕ := 47
def current_students : ℕ := 6 * 3
def girls_bathroom : ℕ := 3
def new_groups : ℕ := 2 * 4
def foreign_exchange_students : ℕ := 3 * 3

-- The total number of missing students
def missing_students : ℕ := girls_bathroom + new_groups + foreign_exchange_students

-- The number of students who went to the canteen
def students_canteen : ℕ := total_students - current_students - missing_students

-- The ratio of students who went to the canteen to girls who went to the bathroom
def canteen_to_bathroom_ratio : ℕ × ℕ := (students_canteen, girls_bathroom)

theorem ratio_is_correct : canteen_to_bathroom_ratio = (3, 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_is_correct_l778_77889


namespace NUMINAMATH_GPT_smallest_arithmetic_geometric_seq_sum_l778_77857

variable (A B C D : ℕ)

noncomputable def arithmetic_seq (A B C : ℕ) (d : ℕ) : Prop :=
  B - A = d ∧ C - B = d

noncomputable def geometric_seq (B C D : ℕ) : Prop :=
  C = (5 / 3) * B ∧ D = (25 / 9) * B

theorem smallest_arithmetic_geometric_seq_sum :
  ∃ A B C D : ℕ, 
    arithmetic_seq A B C 12 ∧ 
    geometric_seq B C D ∧ 
    (A + B + C + D = 104) :=
sorry

end NUMINAMATH_GPT_smallest_arithmetic_geometric_seq_sum_l778_77857


namespace NUMINAMATH_GPT_correct_calculation_l778_77838

theorem correct_calculation (a : ℝ) :
  2 * a^4 * 3 * a^5 = 6 * a^9 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l778_77838


namespace NUMINAMATH_GPT_sum_of_all_possible_values_l778_77811

theorem sum_of_all_possible_values (x y : ℝ) (h : x * y - x^2 - y^2 = 4) :
  (x - 2) * (y - 2) = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_l778_77811


namespace NUMINAMATH_GPT_nontrivial_solution_fraction_l778_77828

theorem nontrivial_solution_fraction (x y z : ℚ)
  (h₁ : x - 6 * y + 3 * z = 0)
  (h₂ : 3 * x - 6 * y - 2 * z = 0)
  (h₃ : x + 6 * y - 5 * z = 0)
  (hne : x ≠ 0) :
  (y * z) / (x^2) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_nontrivial_solution_fraction_l778_77828


namespace NUMINAMATH_GPT_total_students_standing_committee_ways_different_grade_pairs_ways_l778_77862

-- Given conditions
def freshmen : ℕ := 5
def sophomores : ℕ := 6
def juniors : ℕ := 4

-- Proofs (statements only, no proofs provided)
theorem total_students : freshmen + sophomores + juniors = 15 :=
by sorry

theorem standing_committee_ways : freshmen * sophomores * juniors = 120 :=
by sorry

theorem different_grade_pairs_ways :
  freshmen * sophomores + sophomores * juniors + juniors * freshmen = 74 :=
by sorry

end NUMINAMATH_GPT_total_students_standing_committee_ways_different_grade_pairs_ways_l778_77862


namespace NUMINAMATH_GPT_equal_charge_at_250_l778_77844

/-- Define the monthly fee for Plan A --/
def planA_fee (x : ℕ) : ℝ :=
  0.4 * x + 50

/-- Define the monthly fee for Plan B --/
def planB_fee (x : ℕ) : ℝ :=
  0.6 * x

/-- Prove that the charges for Plan A and Plan B are equal when the call duration is 250 minutes --/
theorem equal_charge_at_250 : planA_fee 250 = planB_fee 250 :=
by
  sorry

end NUMINAMATH_GPT_equal_charge_at_250_l778_77844


namespace NUMINAMATH_GPT_passenger_capacity_passenger_capacity_at_5_max_profit_l778_77846

section SubwayProject

-- Define the time interval t and the passenger capacity function p(t)
def p (t : ℕ) : ℕ :=
  if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2
  else if 10 ≤ t ∧ t ≤ 20 then 500
  else 0

-- Define the net profit function Q(t)
def Q (t : ℕ) : ℚ :=
  if 2 ≤ t ∧ t < 10 then (8 * p t - 2656) / t - 60
  else if 10 ≤ t ∧ t ≤ 20 then (1344 : ℚ) / t - 60
  else 0

-- Statement 1: Prove the correct expression for p(t) and its value at t = 5
theorem passenger_capacity (t : ℕ) (ht1 : 2 ≤ t) (ht2 : t ≤ 20) :
  (p t = if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2 else 500) :=
sorry

theorem passenger_capacity_at_5 : p 5 = 450 :=
sorry

-- Statement 2: Prove the time interval t and the maximum value of Q(t)
theorem max_profit : ∃ t : ℕ, 2 ≤ t ∧ t ≤ 10 ∧ Q t = 132 ∧ (∀ u : ℕ, 2 ≤ u ∧ u ≤ 10 → Q u ≤ Q t) :=
sorry

end SubwayProject

end NUMINAMATH_GPT_passenger_capacity_passenger_capacity_at_5_max_profit_l778_77846


namespace NUMINAMATH_GPT_probability_two_dice_sum_seven_l778_77832

theorem probability_two_dice_sum_seven (z : ℕ) (w : ℚ) (h : z = 2) : w = 1 / 6 :=
by sorry

end NUMINAMATH_GPT_probability_two_dice_sum_seven_l778_77832


namespace NUMINAMATH_GPT_minimum_value_of_E_l778_77886

theorem minimum_value_of_E (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 12) : |E| = 11 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_E_l778_77886


namespace NUMINAMATH_GPT_sum_of_remainders_l778_77875

theorem sum_of_remainders (n : ℤ) (h₁ : n % 12 = 5) (h₂ : n % 3 = 2) (h₃ : n % 4 = 1) : 2 + 1 = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l778_77875


namespace NUMINAMATH_GPT_total_money_shared_l778_77833

theorem total_money_shared (A B C : ℕ) (rA rB rC : ℕ) (bens_share : ℕ) 
  (h_ratio : rA = 2 ∧ rB = 3 ∧ rC = 8)
  (h_ben : B = bens_share)
  (h_bensShareGiven : bens_share = 60) : 
  (rA * (bens_share / rB)) + bens_share + (rC * (bens_share / rB)) = 260 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_money_shared_l778_77833


namespace NUMINAMATH_GPT_average_production_is_correct_l778_77887

noncomputable def average_tv_production_last_5_days
  (daily_production : ℕ)
  (ill_workers : List ℕ)
  (decrease_rate : ℕ) : ℚ :=
  let productivity_decrease (n : ℕ) : ℚ := (1 - (decrease_rate * n) / 100 : ℚ) * daily_production
  let total_production := (ill_workers.map productivity_decrease).sum
  total_production / ill_workers.length

theorem average_production_is_correct :
  average_tv_production_last_5_days 50 [3, 5, 2, 4, 3] 2 = 46.6 :=
by
  -- proof needed here
  sorry

end NUMINAMATH_GPT_average_production_is_correct_l778_77887


namespace NUMINAMATH_GPT_determine_a_l778_77830

theorem determine_a (a : ℕ) (p1 p2 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : 2 * p1 * p2 = a) (h4 : p1 + p2 = 15) : 
  a = 52 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l778_77830


namespace NUMINAMATH_GPT_total_cost_is_46_8_l778_77848

def price_pork : ℝ := 6
def price_chicken : ℝ := price_pork - 2
def price_beef : ℝ := price_chicken + 4
def price_lamb : ℝ := price_pork + 3

def quantity_chicken : ℝ := 3.5
def quantity_pork : ℝ := 1.2
def quantity_beef : ℝ := 2.3
def quantity_lamb : ℝ := 0.8

def total_cost : ℝ :=
    (quantity_chicken * price_chicken) +
    (quantity_pork * price_pork) +
    (quantity_beef * price_beef) +
    (quantity_lamb * price_lamb)

theorem total_cost_is_46_8 : total_cost = 46.8 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_46_8_l778_77848


namespace NUMINAMATH_GPT_spadesuit_value_l778_77808

def spadesuit (a b : ℤ) : ℤ :=
  |a^2 - b^2|

theorem spadesuit_value :
  spadesuit 3 (spadesuit 5 2) = 432 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_value_l778_77808


namespace NUMINAMATH_GPT_appropriate_word_count_l778_77840

-- Define the conditions of the problem
def min_minutes := 40
def max_minutes := 55
def words_per_minute := 120

-- Define the bounds for the number of words
def min_words := min_minutes * words_per_minute
def max_words := max_minutes * words_per_minute

-- Define the appropriate number of words
def appropriate_words (words : ℕ) : Prop :=
  words >= min_words ∧ words <= max_words

-- The specific numbers to test
def words1 := 5000
def words2 := 6200

-- The main proof statement
theorem appropriate_word_count : 
  appropriate_words words1 ∧ appropriate_words words2 :=
by
  -- We do not need to provide the proof steps, just state the theorem
  sorry

end NUMINAMATH_GPT_appropriate_word_count_l778_77840


namespace NUMINAMATH_GPT_child_sold_apples_correct_l778_77870

-- Definitions based on conditions
def initial_apples (children : ℕ) (apples_per_child : ℕ) : ℕ := children * apples_per_child
def eaten_apples (children_eating : ℕ) (apples_eaten_per_child : ℕ) : ℕ := children_eating * apples_eaten_per_child
def remaining_apples (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten
def sold_apples (remaining : ℕ) (final : ℕ) : ℕ := remaining - final

-- Given conditions
variable (children : ℕ := 5)
variable (apples_per_child : ℕ := 15)
variable (children_eating : ℕ := 2)
variable (apples_eaten_per_child : ℕ := 4)
variable (final_apples : ℕ := 60)

-- Theorem statement
theorem child_sold_apples_correct :
  sold_apples (remaining_apples (initial_apples children apples_per_child) (eaten_apples children_eating apples_eaten_per_child)) final_apples = 7 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_child_sold_apples_correct_l778_77870


namespace NUMINAMATH_GPT_rectangle_area_l778_77874

theorem rectangle_area (p q : ℝ) (x : ℝ) (h1 : x^2 + (2 * x)^2 = (p + q)^2) : 
    2 * x^2 = (2 * (p + q)^2) / 5 := 
sorry

end NUMINAMATH_GPT_rectangle_area_l778_77874


namespace NUMINAMATH_GPT_find_divisor_l778_77899

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 34) / y = 2) : y = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l778_77899


namespace NUMINAMATH_GPT_probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l778_77859

-- Considering a die with 6 faces
def die_faces := 6

-- Total number of possible outcomes when rolling 3 dice
def total_outcomes := die_faces^3

-- 1. Probability of having exactly one die showing a 6 when rolling 3 dice
def prob_exactly_one_six : ℚ :=
  have favorable_outcomes := 3 * 5^2 -- 3 ways to choose which die shows 6, and 25 ways for others to not show 6
  favorable_outcomes / total_outcomes

-- Proof statement
theorem probability_exactly_one_six : prob_exactly_one_six = 25/72 := by 
  sorry

-- 2. Probability of having at least one die showing a 6 when rolling 3 dice
def prob_at_least_one_six : ℚ :=
  have no_six_outcomes := 5^3
  (total_outcomes - no_six_outcomes) / total_outcomes

-- Proof statement
theorem probability_at_least_one_six : prob_at_least_one_six = 91/216 := by 
  sorry

-- 3. Probability of having at most one die showing a 6 when rolling 3 dice
def prob_at_most_one_six : ℚ :=
  have no_six_probability := 125 / total_outcomes
  have one_six_probability := 75 / total_outcomes
  no_six_probability + one_six_probability

-- Proof statement
theorem probability_at_most_one_six : prob_at_most_one_six = 25/27 := by 
  sorry

end NUMINAMATH_GPT_probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l778_77859


namespace NUMINAMATH_GPT_tetrahedron_volume_correct_l778_77858

noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2))

theorem tetrahedron_volume_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 = c^2) :
  tetrahedron_volume a b c = (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2)) :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_correct_l778_77858


namespace NUMINAMATH_GPT_cost_of_each_candy_bar_l778_77890

-- Definitions of the conditions
def initial_amount : ℕ := 20
def final_amount : ℕ := 12
def number_of_candy_bars : ℕ := 4

-- Statement of the proof problem: prove the cost of each candy bar
theorem cost_of_each_candy_bar :
  (initial_amount - final_amount) / number_of_candy_bars = 2 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_candy_bar_l778_77890


namespace NUMINAMATH_GPT_equation_1_solution_equation_2_solution_l778_77885

theorem equation_1_solution (x : ℝ) :
  6 * (x - 2 / 3) - (x + 7) = 11 → x = 22 / 5 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

theorem equation_2_solution (x : ℝ) :
  (2 * x - 1) / 3 = (2 * x + 1) / 6 - 2 → x = -9 / 2 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

end NUMINAMATH_GPT_equation_1_solution_equation_2_solution_l778_77885


namespace NUMINAMATH_GPT_a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l778_77854

theorem a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4 (a : ℝ) :
  (a < 2 → a^2 < 4) ∧ (a^2 < 4 → a < 2) :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l778_77854


namespace NUMINAMATH_GPT_solve_inequality_system_l778_77816

theorem solve_inequality_system (x : ℝ) (h1 : 2 * x + 1 < 5) (h2 : 2 - x ≤ 1) : 1 ≤ x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l778_77816


namespace NUMINAMATH_GPT_min_possible_value_box_l778_77853

theorem min_possible_value_box :
  ∃ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15 ∧ a^2 + b^2 = 61) ∧
  ∀ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15) → (a^2 + b^2 ≥ 61) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_possible_value_box_l778_77853


namespace NUMINAMATH_GPT_not_possible_d_count_l778_77831

open Real

theorem not_possible_d_count (t s d : ℝ) (h1 : 3 * t - 4 * s = 1989) (h2 : t - s = d) (h3 : 4 * s > 0) :
  ∃ k : ℕ, k = 663 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ k → d ≠ n :=
by
  sorry

end NUMINAMATH_GPT_not_possible_d_count_l778_77831


namespace NUMINAMATH_GPT_seating_arrangement_l778_77813

theorem seating_arrangement :
  let total_arrangements := Nat.factorial 8
  let adjacent_arrangements := Nat.factorial 7 * 2
  total_arrangements - adjacent_arrangements = 30240 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l778_77813


namespace NUMINAMATH_GPT_max_x_plus_y_l778_77897

theorem max_x_plus_y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 9) (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7 / 3 :=
sorry

end NUMINAMATH_GPT_max_x_plus_y_l778_77897


namespace NUMINAMATH_GPT_new_students_correct_l778_77845

variable 
  (students_start_year : Nat)
  (students_left : Nat)
  (students_end_year : Nat)

def new_students (students_start_year students_left students_end_year : Nat) : Nat :=
  students_end_year - (students_start_year - students_left)

theorem new_students_correct :
  ∀ (students_start_year students_left students_end_year : Nat),
  students_start_year = 10 →
  students_left = 4 →
  students_end_year = 48 →
  new_students students_start_year students_left students_end_year = 42 :=
by
  intros students_start_year students_left students_end_year h1 h2 h3
  rw [h1, h2, h3]
  unfold new_students
  norm_num

end NUMINAMATH_GPT_new_students_correct_l778_77845


namespace NUMINAMATH_GPT_tony_gas_expense_in_4_weeks_l778_77856

theorem tony_gas_expense_in_4_weeks :
  let miles_per_gallon := 25
  let miles_per_round_trip_per_day := 50
  let travel_days_per_week := 5
  let tank_capacity_in_gallons := 10
  let cost_per_gallon := 2
  let weeks := 4
  let total_miles_per_week := miles_per_round_trip_per_day * travel_days_per_week
  let total_miles := total_miles_per_week * weeks
  let miles_per_tank := miles_per_gallon * tank_capacity_in_gallons
  let fill_ups_needed := total_miles / miles_per_tank
  let total_gallons_needed := fill_ups_needed * tank_capacity_in_gallons
  let total_cost := total_gallons_needed * cost_per_gallon
  total_cost = 80 :=
by
  sorry

end NUMINAMATH_GPT_tony_gas_expense_in_4_weeks_l778_77856


namespace NUMINAMATH_GPT_terminal_side_angle_is_in_fourth_quadrant_l778_77876

variable (α : ℝ)
variable (tan_alpha cos_alpha : ℝ)

-- Given conditions
def in_second_quadrant := tan_alpha < 0 ∧ cos_alpha > 0

-- Conclusion to prove
theorem terminal_side_angle_is_in_fourth_quadrant 
  (h : in_second_quadrant tan_alpha cos_alpha) : 
  -- Here we model the "fourth quadrant" in a proof-statement context:
  true := sorry

end NUMINAMATH_GPT_terminal_side_angle_is_in_fourth_quadrant_l778_77876


namespace NUMINAMATH_GPT_number_line_problem_l778_77819

theorem number_line_problem (x : ℤ) (h : x + 7 - 4 = 0) : x = -3 :=
by
  -- The proof is omitted as only the statement is required.
  sorry

end NUMINAMATH_GPT_number_line_problem_l778_77819


namespace NUMINAMATH_GPT_gus_buys_2_dozen_l778_77835

-- Definitions from conditions
def dozens_to_golf_balls (d : ℕ) : ℕ := d * 12
def total_golf_balls : ℕ := 132
def golf_balls_per_dozen : ℕ := 12
def dan_buys : ℕ := 5
def chris_buys_golf_balls : ℕ := 48

-- The number of dozens Gus buys
noncomputable def gus_buys (total_dozens dan_dozens chris_dozens : ℕ) : ℕ := total_dozens - dan_dozens - chris_dozens

theorem gus_buys_2_dozen : gus_buys (total_golf_balls / golf_balls_per_dozen) dan_buys (chris_buys_golf_balls / golf_balls_per_dozen) = 2 := by
  sorry

end NUMINAMATH_GPT_gus_buys_2_dozen_l778_77835


namespace NUMINAMATH_GPT_inequality_solution_set_l778_77837

theorem inequality_solution_set :
  { x : ℝ | -3 < x ∧ x < 2 } = { x : ℝ | abs (x - 1) + abs (x + 2) < 5 } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l778_77837


namespace NUMINAMATH_GPT_fraction_distance_traveled_by_bus_l778_77872

theorem fraction_distance_traveled_by_bus (D : ℝ) (hD : D = 105.00000000000003)
    (distance_by_foot : ℝ) (h_foot : distance_by_foot = (1 / 5) * D)
    (distance_by_car : ℝ) (h_car : distance_by_car = 14) :
    (D - (distance_by_foot + distance_by_car)) / D = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_distance_traveled_by_bus_l778_77872


namespace NUMINAMATH_GPT_triangle_perimeter_l778_77812

theorem triangle_perimeter (side1 side2 side3 : ℕ) (h1 : side1 = 40) (h2 : side2 = 50) (h3 : side3 = 70) : 
  side1 + side2 + side3 = 160 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l778_77812


namespace NUMINAMATH_GPT_rotation_result_l778_77802

def initial_vector : ℝ × ℝ × ℝ := (3, -1, 1)

def rotate_180_z (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match v with
  | (x, y, z) => (-x, -y, z)

theorem rotation_result :
  rotate_180_z initial_vector = (-3, 1, 1) :=
by
  sorry

end NUMINAMATH_GPT_rotation_result_l778_77802


namespace NUMINAMATH_GPT_max_mark_is_600_l778_77814

-- Define the conditions
def forty_percent (M : ℝ) : ℝ := 0.40 * M
def student_score : ℝ := 175
def additional_marks_needed : ℝ := 65

-- The goal is to prove that the maximum mark is 600
theorem max_mark_is_600 (M : ℝ) :
  forty_percent M = student_score + additional_marks_needed → M = 600 := 
by 
  sorry

end NUMINAMATH_GPT_max_mark_is_600_l778_77814


namespace NUMINAMATH_GPT_brother_growth_is_one_l778_77826

-- Define measurements related to Stacy's height.
def Stacy_previous_height : ℕ := 50
def Stacy_current_height : ℕ := 57

-- Define the condition that Stacy's growth is 6 inches more than her brother's growth.
def Stacy_growth := Stacy_current_height - Stacy_previous_height
def Brother_growth := Stacy_growth - 6

-- Prove that Stacy's brother grew 1 inch.
theorem brother_growth_is_one : Brother_growth = 1 :=
by
  sorry

end NUMINAMATH_GPT_brother_growth_is_one_l778_77826


namespace NUMINAMATH_GPT_find_circle_radius_l778_77896

noncomputable def circle_radius (x y : ℝ) : ℝ :=
  (x - 1) ^ 2 + (y + 2) ^ 2

theorem find_circle_radius :
  (∀ x y : ℝ, 25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 → circle_radius x y = 0) → radius = 0 :=
sorry

end NUMINAMATH_GPT_find_circle_radius_l778_77896


namespace NUMINAMATH_GPT_x_cubed_inverse_cubed_l778_77801

theorem x_cubed_inverse_cubed (x : ℝ) (hx : x + 1/x = 3) : x^3 + 1/x^3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_x_cubed_inverse_cubed_l778_77801


namespace NUMINAMATH_GPT_unqualified_weight_l778_77834

theorem unqualified_weight (w : ℝ) (upper_limit lower_limit : ℝ) 
  (h1 : upper_limit = 10.1) 
  (h2 : lower_limit = 9.9) 
  (h3 : w = 9.09 ∨ w = 9.99 ∨ w = 10.01 ∨ w = 10.09) :
  ¬ (9.09 ≥ lower_limit ∧ 9.09 ≤ upper_limit) :=
by
  sorry

end NUMINAMATH_GPT_unqualified_weight_l778_77834


namespace NUMINAMATH_GPT_taxi_faster_than_truck_l778_77865

noncomputable def truck_speed : ℝ := 2.1 / 1
noncomputable def taxi_speed : ℝ := 10.5 / 4

theorem taxi_faster_than_truck :
  taxi_speed / truck_speed = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_taxi_faster_than_truck_l778_77865


namespace NUMINAMATH_GPT_find_a_in_subset_l778_77864

theorem find_a_in_subset 
  (A : Set ℝ)
  (B : Set ℝ)
  (hA : A = { x | x^2 ≠ 1 })
  (hB : ∃ a : ℝ, B = { x | a * x = 1 })
  (h_subset : B ⊆ A) : 
  ∃ a : ℝ, a = 0 ∨ a = 1 ∨ a = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_in_subset_l778_77864


namespace NUMINAMATH_GPT_men_absent_is_5_l778_77843

-- Define the given conditions
def original_number_of_men : ℕ := 30
def planned_days : ℕ := 10
def actual_days : ℕ := 12

-- Prove the number of men absent (x) is 5, under given conditions
theorem men_absent_is_5 : ∃ x : ℕ, 30 * planned_days = (original_number_of_men - x) * actual_days ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_men_absent_is_5_l778_77843


namespace NUMINAMATH_GPT_sequence_polynomial_l778_77860

theorem sequence_polynomial (f : ℕ → ℤ) :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ↔ (∀ n, f n = n^3 + 2 * n^2 + n + 3) :=
by
  sorry

end NUMINAMATH_GPT_sequence_polynomial_l778_77860


namespace NUMINAMATH_GPT_car_a_speed_l778_77863

theorem car_a_speed (d_A d_B v_B t v_A : ℝ)
  (h1 : d_A = 10)
  (h2 : v_B = 50)
  (h3 : t = 2.25)
  (h4 : d_A + 8 - d_B = v_A * t)
  (h5 : d_B = v_B * t) :
  v_A = 58 :=
by
  -- Work on the proof here
  sorry

end NUMINAMATH_GPT_car_a_speed_l778_77863


namespace NUMINAMATH_GPT_total_alphabets_written_l778_77867

-- Define the number of vowels and the number of times each is written
def num_vowels : ℕ := 5
def repetitions : ℕ := 4

-- The theorem stating the total number of alphabets written on the board
theorem total_alphabets_written : num_vowels * repetitions = 20 := by
  sorry

end NUMINAMATH_GPT_total_alphabets_written_l778_77867


namespace NUMINAMATH_GPT_det_condition_l778_77892

theorem det_condition (a b c d : ℤ) 
    (h_exists : ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n) :
    |a * d - b * c| = 1 :=
sorry

end NUMINAMATH_GPT_det_condition_l778_77892


namespace NUMINAMATH_GPT_original_price_of_petrol_l778_77868

theorem original_price_of_petrol (P : ℝ) (h : 0.9 * P * 190 / (0.9 * P) = 190 / P + 5) : P = 4.22 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_original_price_of_petrol_l778_77868


namespace NUMINAMATH_GPT_simplify_fraction_l778_77807

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end NUMINAMATH_GPT_simplify_fraction_l778_77807


namespace NUMINAMATH_GPT_eventB_is_not_random_l778_77815

def eventA := "The sun rises in the east and it rains in the west"
def eventB := "It's not cold when it snows but cold when it melts"
def eventC := "It rains continuously during the Qingming festival"
def eventD := "It's sunny every day when the plums turn yellow"

def is_random_event (event : String) : Prop :=
  event = eventA ∨ event = eventC ∨ event = eventD

theorem eventB_is_not_random : ¬ is_random_event eventB :=
by
  unfold is_random_event
  sorry

end NUMINAMATH_GPT_eventB_is_not_random_l778_77815


namespace NUMINAMATH_GPT_player_weekly_earnings_l778_77879

structure Performance :=
  (points assists rebounds steals : ℕ)

def base_pay (avg_points : ℕ) : ℕ :=
  if avg_points >= 30 then 10000 else 8000

def assists_bonus (total_assists : ℕ) : ℕ :=
  if total_assists >= 20 then 5000
  else if total_assists >= 10 then 3000
  else 1000

def rebounds_bonus (total_rebounds : ℕ) : ℕ :=
  if total_rebounds >= 40 then 5000
  else if total_rebounds >= 20 then 3000
  else 1000

def steals_bonus (total_steals : ℕ) : ℕ :=
  if total_steals >= 15 then 5000
  else if total_steals >= 5 then 3000
  else 1000

def total_payment (performances : List Performance) : ℕ :=
  let total_points := performances.foldl (λ acc p => acc + p.points) 0
  let total_assists := performances.foldl (λ acc p => acc + p.assists) 0
  let total_rebounds := performances.foldl (λ acc p => acc + p.rebounds) 0
  let total_steals := performances.foldl (λ acc p => acc + p.steals) 0
  let avg_points := total_points / performances.length
  base_pay avg_points + assists_bonus total_assists + rebounds_bonus total_rebounds + steals_bonus total_steals
  
theorem player_weekly_earnings :
  let performances := [
    Performance.mk 30 5 7 3,
    Performance.mk 28 6 5 2,
    Performance.mk 32 4 9 1,
    Performance.mk 34 3 11 2,
    Performance.mk 26 2 8 3
  ]
  total_payment performances = 23000 := by 
    sorry

end NUMINAMATH_GPT_player_weekly_earnings_l778_77879


namespace NUMINAMATH_GPT_HCF_of_numbers_l778_77817

theorem HCF_of_numbers (a b : ℕ) (h₁ : a * b = 84942) (h₂ : Nat.lcm a b = 2574) : Nat.gcd a b = 33 :=
by
  sorry

end NUMINAMATH_GPT_HCF_of_numbers_l778_77817


namespace NUMINAMATH_GPT_number_being_divided_l778_77825

theorem number_being_divided (divisor quotient remainder number : ℕ) 
  (h_divisor : divisor = 3) 
  (h_quotient : quotient = 7) 
  (h_remainder : remainder = 1)
  (h_number : number = divisor * quotient + remainder) : 
  number = 22 :=
by
  rw [h_divisor, h_quotient, h_remainder] at h_number
  exact h_number

end NUMINAMATH_GPT_number_being_divided_l778_77825


namespace NUMINAMATH_GPT_paris_total_study_hours_semester_l778_77881

-- Definitions
def weeks_in_semester := 15
def weekday_study_hours_per_day := 3
def weekdays_per_week := 5
def saturday_study_hours := 4
def sunday_study_hours := 5

-- Theorem statement
theorem paris_total_study_hours_semester :
  weeks_in_semester * (weekday_study_hours_per_day * weekdays_per_week + saturday_study_hours + sunday_study_hours) = 360 := 
sorry

end NUMINAMATH_GPT_paris_total_study_hours_semester_l778_77881


namespace NUMINAMATH_GPT_sum_of_variables_l778_77804

theorem sum_of_variables (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) (ha : a = 2 * Real.sqrt 6) (hb : b = 3 * Real.sqrt 6) (hc : c = 6 * Real.sqrt 6) : 
  a + b + c = 11 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_variables_l778_77804


namespace NUMINAMATH_GPT_abs_sum_inequality_l778_77869

theorem abs_sum_inequality (x : ℝ) : (|x - 2| + |x + 3| < 7) ↔ (-6 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_abs_sum_inequality_l778_77869


namespace NUMINAMATH_GPT_find_room_height_l778_77849

theorem find_room_height (l b d : ℕ) (h : ℕ) (hl : l = 12) (hb : b = 8) (hd : d = 17) :
  d = Int.sqrt (l^2 + b^2 + h^2) → h = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_room_height_l778_77849


namespace NUMINAMATH_GPT_brokerage_percentage_l778_77821

theorem brokerage_percentage
  (cash_realized : ℝ)
  (cash_before_brokerage : ℝ)
  (h₁ : cash_realized = 109.25)
  (h₂ : cash_before_brokerage = 109) :
  ((cash_realized - cash_before_brokerage) / cash_before_brokerage) * 100 = 0.23 := 
by
  sorry

end NUMINAMATH_GPT_brokerage_percentage_l778_77821


namespace NUMINAMATH_GPT_smallest_base_10_integer_l778_77800

-- Given conditions
def is_valid_base (a b : ℕ) : Prop := a > 2 ∧ b > 2

def base_10_equivalence (a b n : ℕ) : Prop := (2 * a + 1 = n) ∧ (b + 2 = n)

-- The smallest base-10 integer represented as 21_a and 12_b
theorem smallest_base_10_integer :
  ∃ (a b n : ℕ), is_valid_base a b ∧ base_10_equivalence a b n ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_base_10_integer_l778_77800


namespace NUMINAMATH_GPT_quadratic_nonneg_iff_l778_77842

variable {a b c : ℝ}

theorem quadratic_nonneg_iff :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4 * a * c ≤ 0) :=
by sorry

end NUMINAMATH_GPT_quadratic_nonneg_iff_l778_77842


namespace NUMINAMATH_GPT_eugene_pencils_after_giving_l778_77851

-- Define Eugene's initial number of pencils and the number of pencils given away.
def initial_pencils : ℝ := 51.0
def pencils_given : ℝ := 6.0

-- State the theorem that should be proved.
theorem eugene_pencils_after_giving : initial_pencils - pencils_given = 45.0 :=
by
  -- We would normally provide the proof steps here, but as per instructions, we'll use "sorry" to skip it.
  sorry

end NUMINAMATH_GPT_eugene_pencils_after_giving_l778_77851


namespace NUMINAMATH_GPT_find_mother_age_l778_77823

-- Definitions for the given conditions
def serena_age_now := 9
def years_in_future := 6
def serena_age_future := serena_age_now + years_in_future
def mother_age_future (M : ℕ) := 3 * serena_age_future

-- The main statement to prove
theorem find_mother_age (M : ℕ) (h1 : M = mother_age_future M - years_in_future) : M = 39 :=
by
  sorry

end NUMINAMATH_GPT_find_mother_age_l778_77823


namespace NUMINAMATH_GPT_find_4a_add_c_find_2a_sub_2b_sub_c_l778_77893

variables {R : Type*} [CommRing R]

theorem find_4a_add_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  4 * a + c = 12 :=
sorry

theorem find_2a_sub_2b_sub_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  2 * a - 2 * b - c = 14 :=
sorry

end NUMINAMATH_GPT_find_4a_add_c_find_2a_sub_2b_sub_c_l778_77893


namespace NUMINAMATH_GPT_problem_statement_l778_77884

theorem problem_statement (M N : ℕ) 
  (hM : M = 2020 / 5) 
  (hN : N = 2020 / 20) : 10 * M / N = 40 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l778_77884


namespace NUMINAMATH_GPT_purchase_price_of_first_commodity_l778_77888

-- Define the conditions
variable (price_first price_second : ℝ)
variable (h1 : price_first - price_second = 127)
variable (h2 : price_first + price_second = 827)

-- Prove the purchase price of the first commodity is $477
theorem purchase_price_of_first_commodity : price_first = 477 :=
by
  sorry

end NUMINAMATH_GPT_purchase_price_of_first_commodity_l778_77888


namespace NUMINAMATH_GPT_k_value_if_divisible_l778_77841

theorem k_value_if_divisible :
  ∀ k : ℤ, (x^2 + k * x - 3) % (x - 1) = 0 → k = 2 :=
by
  intro k
  sorry

end NUMINAMATH_GPT_k_value_if_divisible_l778_77841


namespace NUMINAMATH_GPT_largest_square_side_length_l778_77822

theorem largest_square_side_length (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) : 
  ∃ x : ℝ, x = (a * b) / (a + b) := 
sorry

end NUMINAMATH_GPT_largest_square_side_length_l778_77822


namespace NUMINAMATH_GPT_JulieCompletesInOneHour_l778_77873

-- Define conditions
def JuliePeelsIn : ℕ := 10
def TedPeelsIn : ℕ := 8
def TimeTogether : ℕ := 4

-- Define their respective rates
def JulieRate : ℚ := 1 / JuliePeelsIn
def TedRate : ℚ := 1 / TedPeelsIn

-- Define the task completion in 4 hours together
def TaskCompletedTogether : ℚ := (JulieRate * TimeTogether) + (TedRate * TimeTogether)

-- Define remaining task after working together
def RemainingTask : ℚ := 1 - TaskCompletedTogether

-- Define time for Julie to complete the remaining task
def TimeForJulieToComplete : ℚ := RemainingTask / JulieRate

-- The theorem statement
theorem JulieCompletesInOneHour :
  TimeForJulieToComplete = 1 := by
  sorry

end NUMINAMATH_GPT_JulieCompletesInOneHour_l778_77873


namespace NUMINAMATH_GPT_car_rental_daily_rate_l778_77871

theorem car_rental_daily_rate (x : ℝ) : 
  (x + 0.18 * 48 = 18.95 + 0.16 * 48) -> 
  x = 17.99 :=
by 
  sorry

end NUMINAMATH_GPT_car_rental_daily_rate_l778_77871


namespace NUMINAMATH_GPT_symmetric_line_eq_x_axis_l778_77878

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * (-y) + 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_x_axis_l778_77878


namespace NUMINAMATH_GPT_second_largest_is_D_l778_77883

noncomputable def A := 3 * 3
noncomputable def C := 4 * A
noncomputable def B := C - 15
noncomputable def D := A + 19

theorem second_largest_is_D : 
    ∀ (A B C D : ℕ), 
      A = 9 → 
      B = 21 →
      C = 36 →
      D = 28 →
      D = 28 :=
by
  intros A B C D hA hB hC hD
  have h1 : A = 9 := by assumption
  have h2 : B = 21 := by assumption
  have h3 : C = 36 := by assumption
  have h4 : D = 28 := by assumption
  exact h4

end NUMINAMATH_GPT_second_largest_is_D_l778_77883


namespace NUMINAMATH_GPT_haley_initial_trees_l778_77824

theorem haley_initial_trees (dead_trees trees_left initial_trees : ℕ) 
    (h_dead: dead_trees = 2)
    (h_left: trees_left = 10)
    (h_initial: initial_trees = trees_left + dead_trees) : 
    initial_trees = 12 := 
by sorry

end NUMINAMATH_GPT_haley_initial_trees_l778_77824


namespace NUMINAMATH_GPT_isosceles_triangle_interior_angles_l778_77894

theorem isosceles_triangle_interior_angles (a b c : ℝ) 
  (h1 : b = c) (h2 : a + b + c = 180) (exterior : a + 40 = 180 ∨ b + 40 = 140) :
  (a = 40 ∧ b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40 ∧ c = 40) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_interior_angles_l778_77894


namespace NUMINAMATH_GPT_nonagon_arithmetic_mean_property_l778_77809

def is_equilateral_triangle (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  (j = (i + 3) % 9) ∧ (k = (i + 6) % 9)

def is_arithmetic_mean (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  A j = (A i + A k) / 2

theorem nonagon_arithmetic_mean_property :
  ∀ (A : Fin 9 → ℤ),
    (∀ i, A i = 2016 + i) →
    (∀ i j k : Fin 9, is_equilateral_triangle A i j k → is_arithmetic_mean A i j k) :=
by
  intros
  sorry

end NUMINAMATH_GPT_nonagon_arithmetic_mean_property_l778_77809


namespace NUMINAMATH_GPT_find_number_type_l778_77827

-- Definitions of the problem conditions
def consecutive (a b c d : ℤ) : Prop := (b = a + 2) ∧ (c = a + 4) ∧ (d = a + 6)
def sum_is_52 (a b c d : ℤ) : Prop := a + b + c + d = 52
def third_number_is_14 (c : ℤ) : Prop := c = 14

-- The proof problem statement
theorem find_number_type (a b c d : ℤ) 
                         (h1 : consecutive a b c d) 
                         (h2 : sum_is_52 a b c d) 
                         (h3 : third_number_is_14 c) :
  (∃ (k : ℤ), a = 2 * k ∧ b = 2 * k + 2 ∧ c = 2 * k + 4 ∧ d = 2 * k + 6) 
  := sorry

end NUMINAMATH_GPT_find_number_type_l778_77827


namespace NUMINAMATH_GPT_crossing_time_l778_77836

-- Define the conditions
def walking_speed_kmh : Float := 10
def bridge_length_m : Float := 1666.6666666666665

-- Convert the man's walking speed to meters per minute
def walking_speed_mpm : Float := walking_speed_kmh * (1000 / 60)

-- State the theorem we want to prove
theorem crossing_time 
  (ws_kmh : Float := walking_speed_kmh)
  (bl_m : Float := bridge_length_m)
  (ws_mpm : Float := walking_speed_mpm) :
  bl_m / ws_mpm = 10 :=
by
  sorry

end NUMINAMATH_GPT_crossing_time_l778_77836


namespace NUMINAMATH_GPT_ratio_spaghetti_pizza_l778_77855

/-- Define the number of students who participated in the survey and their preferences --/
def students_surveyed : ℕ := 800
def lasagna_pref : ℕ := 150
def manicotti_pref : ℕ := 120
def ravioli_pref : ℕ := 180
def spaghetti_pref : ℕ := 200
def pizza_pref : ℕ := 150

/-- Prove the ratio of students who preferred spaghetti to those who preferred pizza is 4/3 --/
theorem ratio_spaghetti_pizza : (200 / 150 : ℚ) = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_spaghetti_pizza_l778_77855


namespace NUMINAMATH_GPT_bobby_finishes_candies_in_weeks_l778_77898

def total_candies (packets: Nat) (candies_per_packet: Nat) : Nat := packets * candies_per_packet

def candies_eaten_per_week (candies_per_day_mon_fri: Nat) (days_mon_fri: Nat) (candies_per_day_weekend: Nat) (days_weekend: Nat) : Nat :=
  (candies_per_day_mon_fri * days_mon_fri) + (candies_per_day_weekend * days_weekend)

theorem bobby_finishes_candies_in_weeks :
  let packets := 2
  let candies_per_packet := 18
  let candies_per_day_mon_fri := 2
  let days_mon_fri := 5
  let candies_per_day_weekend := 1
  let days_weekend := 2

  total_candies packets candies_per_packet / candies_eaten_per_week candies_per_day_mon_fri days_mon_fri candies_per_day_weekend days_weekend = 3 :=
by
  sorry

end NUMINAMATH_GPT_bobby_finishes_candies_in_weeks_l778_77898


namespace NUMINAMATH_GPT_parking_lot_vehicle_spaces_l778_77839

theorem parking_lot_vehicle_spaces
  (total_spaces : ℕ)
  (spaces_per_caravan : ℕ)
  (num_caravans : ℕ)
  (remaining_spaces : ℕ) :
  total_spaces = 30 →
  spaces_per_caravan = 2 →
  num_caravans = 3 →
  remaining_spaces = total_spaces - (spaces_per_caravan * num_caravans) →
  remaining_spaces = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_parking_lot_vehicle_spaces_l778_77839


namespace NUMINAMATH_GPT_fundamental_disagreement_l778_77850

-- Definitions based on conditions
def represents_materialism (s : String) : Prop :=
  s = "Without scenery, where does emotion come from?"

def represents_idealism (s : String) : Prop :=
  s = "Without emotion, where does scenery come from?"

-- Theorem statement
theorem fundamental_disagreement :
  ∀ (s1 s2 : String),
  (represents_materialism s1 ∧ represents_idealism s2) →
  (∃ disagreement : String,
    disagreement = "Acknowledging whether the essence of the world is material or consciousness") :=
by
  intros s1 s2 h
  existsi "Acknowledging whether the essence of the world is material or consciousness"
  sorry

end NUMINAMATH_GPT_fundamental_disagreement_l778_77850


namespace NUMINAMATH_GPT_coles_average_speed_l778_77810

theorem coles_average_speed (t_work : ℝ) (t_round : ℝ) (s_return : ℝ) (t_return : ℝ) (d : ℝ) (t_work_min : ℕ) :
  t_work_min = 72 ∧ t_round = 2 ∧ s_return = 90 ∧ 
  t_work = t_work_min / 60 ∧ t_return = t_round - t_work ∧ d = s_return * t_return →
  d / t_work = 60 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_coles_average_speed_l778_77810


namespace NUMINAMATH_GPT_roster_method_A_l778_77847

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem roster_method_A :
  A = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_roster_method_A_l778_77847


namespace NUMINAMATH_GPT_john_small_planks_l778_77882

theorem john_small_planks (L S : ℕ) (h1 : L = 12) (h2 : L + S = 29) : S = 17 :=
by {
  sorry
}

end NUMINAMATH_GPT_john_small_planks_l778_77882


namespace NUMINAMATH_GPT_cube_volume_of_surface_area_l778_77891

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end NUMINAMATH_GPT_cube_volume_of_surface_area_l778_77891
