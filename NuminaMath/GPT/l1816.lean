import Mathlib

namespace NUMINAMATH_GPT_total_fruits_in_30_days_l1816_181699

-- Define the number of oranges Sophie receives each day
def sophie_daily_oranges : ℕ := 20

-- Define the number of grapes Hannah receives each day
def hannah_daily_grapes : ℕ := 40

-- Define the number of days
def number_of_days : ℕ := 30

-- Calculate the total number of fruits received by Sophie and Hannah in 30 days
theorem total_fruits_in_30_days :
  (sophie_daily_oranges * number_of_days) + (hannah_daily_grapes * number_of_days) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_total_fruits_in_30_days_l1816_181699


namespace NUMINAMATH_GPT_value_2_std_dev_less_than_mean_l1816_181682

-- Define the mean and standard deviation as constants
def mean : ℝ := 14.5
def std_dev : ℝ := 1.5

-- State the theorem (problem)
theorem value_2_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.5 := by
  sorry

end NUMINAMATH_GPT_value_2_std_dev_less_than_mean_l1816_181682


namespace NUMINAMATH_GPT_area_of_fig_eq_2_l1816_181654

noncomputable def area_of_fig : ℝ :=
  - ∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem area_of_fig_eq_2 : area_of_fig = 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_fig_eq_2_l1816_181654


namespace NUMINAMATH_GPT_round_robin_highest_score_l1816_181652

theorem round_robin_highest_score
  (n : ℕ) (hn : n = 16)
  (teams : Fin n → ℕ)
  (games_played : Fin n → Fin n → ℕ)
  (draws : Fin n → Fin n → ℕ)
  (win_points : ℕ := 2)
  (draw_points : ℕ := 1)
  (total_games : ℕ := (n * (n - 1)) / 2) :
  ¬ (∃ max_score : ℕ, ∀ i : Fin n, teams i ≤ max_score ∧ max_score < 16) :=
by sorry

end NUMINAMATH_GPT_round_robin_highest_score_l1816_181652


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1816_181615

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 + a 4 = 15)
  (h2 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2)
  (h_positive : ∀ n, 0 < a n) :
  a 10 = 19 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1816_181615


namespace NUMINAMATH_GPT_sum_of_factors_30_l1816_181627

def sum_of_factors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ x => n % x = 0) |>.sum

theorem sum_of_factors_30 : sum_of_factors 30 = 72 := by
  sorry

end NUMINAMATH_GPT_sum_of_factors_30_l1816_181627


namespace NUMINAMATH_GPT_intersection_A_B_l1816_181620

open Set

noncomputable def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

noncomputable def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_A_B_l1816_181620


namespace NUMINAMATH_GPT_optimal_position_station_l1816_181633

-- Definitions for the conditions
def num_buildings := 5
def building_workers (k : ℕ) : ℕ := if k ≤ 5 then k else 0
def distance_between_buildings := 50

-- Function to calculate the total walking distance
noncomputable def total_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

-- Theorem statement
theorem optimal_position_station :
  ∃ x : ℝ, (∀ y : ℝ, total_distance x ≤ total_distance y) ∧ x = 150 :=
by
  sorry

end NUMINAMATH_GPT_optimal_position_station_l1816_181633


namespace NUMINAMATH_GPT_frog_climb_time_l1816_181666

-- Definitions related to the problem
def well_depth : ℕ := 12
def climb_per_cycle : ℕ := 3
def slip_per_cycle : ℕ := 1
def effective_climb_per_cycle : ℕ := climb_per_cycle - slip_per_cycle

-- Time taken for each activity
def time_to_climb : ℕ := 10 -- given as t
def time_to_slip : ℕ := time_to_climb / 3
def total_time_per_cycle : ℕ := time_to_climb + time_to_slip

-- Condition specifying the observed frog position at a certain time
def observed_time : ℕ := 17 -- minutes since 8:00
def observed_position : ℕ := 9 -- meters climbed since it's 3 meters from the top of the well (well_depth - 3)

-- The main theorem stating the total time taken to climb to the top of the well
theorem frog_climb_time : 
  ∃ (k : ℕ), k * effective_climb_per_cycle + climb_per_cycle = well_depth ∧ k * total_time_per_cycle + time_to_climb = 22 := 
sorry

end NUMINAMATH_GPT_frog_climb_time_l1816_181666


namespace NUMINAMATH_GPT_least_repeating_block_of_8_over_11_l1816_181687

theorem least_repeating_block_of_8_over_11 : (∃ n : ℕ, (∀ m : ℕ, m < n → ¬(∃ a b : ℤ, (10^m - 1) * (8 * 10^n - b * 11 * 10^(n - t)) = a * 11 * 10^(m - t))) ∧ n ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_least_repeating_block_of_8_over_11_l1816_181687


namespace NUMINAMATH_GPT_num_males_in_group_l1816_181637

-- Definitions based on the given conditions
def num_females (f : ℕ) : Prop := f = 16
def num_males_choose_malt (m_malt : ℕ) : Prop := m_malt = 6
def num_females_choose_malt (f_malt : ℕ) : Prop := f_malt = 8
def num_choose_malt (m_malt f_malt n_malt : ℕ) : Prop := n_malt = m_malt + f_malt
def num_choose_coke (c : ℕ) (n_malt : ℕ) : Prop := n_malt = 2 * c
def total_cheerleaders (t : ℕ) (n_malt c : ℕ) : Prop := t = n_malt + c
def num_males (m f t : ℕ) : Prop := m = t - f

theorem num_males_in_group
  (f m_malt f_malt n_malt c t m : ℕ)
  (hf : num_females f)
  (hmm : num_males_choose_malt m_malt)
  (hfm : num_females_choose_malt f_malt)
  (hmalt : num_choose_malt m_malt f_malt n_malt)
  (hc : num_choose_coke c n_malt)
  (ht : total_cheerleaders t n_malt c)
  (hm : num_males m f t) :
  m = 5 := 
sorry

end NUMINAMATH_GPT_num_males_in_group_l1816_181637


namespace NUMINAMATH_GPT_find_capacity_l1816_181696

noncomputable def pool_capacity (V1 V2 q : ℝ) : Prop :=
  V1 = q / 120 ∧ V2 = V1 + 50 ∧ V1 + V2 = q / 48

theorem find_capacity (q : ℝ) : ∃ V1 V2, pool_capacity V1 V2 q → q = 12000 :=
by 
  sorry

end NUMINAMATH_GPT_find_capacity_l1816_181696


namespace NUMINAMATH_GPT_shells_total_l1816_181695

variable (x y : ℝ)

theorem shells_total (h1 : y = x + (x + 32)) : y = 2 * x + 32 :=
sorry

end NUMINAMATH_GPT_shells_total_l1816_181695


namespace NUMINAMATH_GPT_katie_miles_l1816_181638

theorem katie_miles (x : ℕ) (h1 : ∀ y, y = 3 * x → y ≤ 240) (h2 : x + 3 * x = 240) : x = 60 :=
sorry

end NUMINAMATH_GPT_katie_miles_l1816_181638


namespace NUMINAMATH_GPT_ratio_nine_years_ago_correct_l1816_181630

-- Conditions
def C : ℕ := 24
def G : ℕ := C / 2

-- Question and expected answer
def ratio_nine_years_ago : ℕ := (C - 9) / (G - 9)

theorem ratio_nine_years_ago_correct : ratio_nine_years_ago = 5 := by
  sorry

end NUMINAMATH_GPT_ratio_nine_years_ago_correct_l1816_181630


namespace NUMINAMATH_GPT_sequence_sum_a5_a6_l1816_181626

-- Given sequence partial sum definition
def partial_sum (n : ℕ) : ℕ := n^3

-- Definition of sequence term a_n
def a (n : ℕ) : ℕ := partial_sum n - partial_sum (n - 1)

-- Main theorem to prove a_5 + a_6 = 152
theorem sequence_sum_a5_a6 : a 5 + a 6 = 152 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_a5_a6_l1816_181626


namespace NUMINAMATH_GPT_PQRS_product_l1816_181686

theorem PQRS_product :
  let P := (Real.sqrt 2012 + Real.sqrt 2013)
  let Q := (-Real.sqrt 2012 - Real.sqrt 2013)
  let R := (Real.sqrt 2012 - Real.sqrt 2013)
  let S := (Real.sqrt 2013 - Real.sqrt 2012)
  P * Q * R * S = 1 :=
by
  sorry

end NUMINAMATH_GPT_PQRS_product_l1816_181686


namespace NUMINAMATH_GPT_abs_neg_three_l1816_181683

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1816_181683


namespace NUMINAMATH_GPT_find_a_plus_b_l1816_181608

theorem find_a_plus_b 
  (a b : ℝ)
  (f : ℝ → ℝ) 
  (f_def : ∀ x, f x = x^3 + 3 * x^2 + 6 * x + 14)
  (cond_a : f a = 1) 
  (cond_b : f b = 19) :
  a + b = -2 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1816_181608


namespace NUMINAMATH_GPT_or_false_iff_not_p_l1816_181649

theorem or_false_iff_not_p (p q : Prop) : (p ∨ q → false) ↔ ¬p :=
by sorry

end NUMINAMATH_GPT_or_false_iff_not_p_l1816_181649


namespace NUMINAMATH_GPT_complement_intersection_l1816_181658

open Set -- Open the Set namespace

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2})
variable (B : Set ℝ := {x | x ≤ -1 ∨ x > 2})

theorem complement_intersection :
  (U \ B) ∩ A = {x | x = 0 ∨ x = 1 ∨ x = 2} :=
by
  sorry -- Proof not required as per the instructions

end NUMINAMATH_GPT_complement_intersection_l1816_181658


namespace NUMINAMATH_GPT_number_of_red_notes_each_row_l1816_181640

-- Definitions for the conditions
variable (R : ℕ) -- Number of red notes in each row
variable (total_notes : ℕ := 100) -- Total number of notes

-- Derived quantities
def total_red_notes := 5 * R
def total_blue_notes := 2 * total_red_notes + 10

-- Statement of the theorem
theorem number_of_red_notes_each_row 
  (h : total_red_notes + total_blue_notes = total_notes) : 
  R = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_notes_each_row_l1816_181640


namespace NUMINAMATH_GPT_tricycles_count_l1816_181602

theorem tricycles_count (B T : ℕ) (hB : B = 50) (hW : 2 * B + 3 * T = 160) : T = 20 :=
by
  sorry

end NUMINAMATH_GPT_tricycles_count_l1816_181602


namespace NUMINAMATH_GPT_Jacqueline_gave_Jane_l1816_181614

def total_fruits (plums guavas apples : ℕ) : ℕ :=
  plums + guavas + apples

def fruits_given_to_Jane (initial left : ℕ) : ℕ :=
  initial - left

theorem Jacqueline_gave_Jane :
  let plums := 16
  let guavas := 18
  let apples := 21
  let left := 15
  let initial := total_fruits plums guavas apples
  fruits_given_to_Jane initial left = 40 :=
by
  sorry

end NUMINAMATH_GPT_Jacqueline_gave_Jane_l1816_181614


namespace NUMINAMATH_GPT_complex_number_arithmetic_l1816_181678

theorem complex_number_arithmetic (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end NUMINAMATH_GPT_complex_number_arithmetic_l1816_181678


namespace NUMINAMATH_GPT_retirement_year_2020_l1816_181685

-- Given conditions
def femaleRetirementAge := 55
def initialRetirementYear (birthYear : ℕ) := birthYear + femaleRetirementAge
def delayedRetirementYear (baseYear additionalYears : ℕ) := baseYear + additionalYears

def postponementStep := 3
def delayStartYear := 2018
def retirementAgeIn2045 := 65
def retirementYear (birthYear : ℕ) : ℕ :=
  let originalRetirementYear := initialRetirementYear birthYear
  let delayYears := ((originalRetirementYear - delayStartYear) / postponementStep) + 1
  delayedRetirementYear originalRetirementYear delayYears

-- Main theorem to prove
theorem retirement_year_2020 : retirementYear 1964 = 2020 := sorry

end NUMINAMATH_GPT_retirement_year_2020_l1816_181685


namespace NUMINAMATH_GPT_red_suit_top_card_probability_l1816_181632

theorem red_suit_top_card_probability :
  let num_cards := 104
  let num_red_suits := 4
  let cards_per_suit := 26
  let num_red_cards := num_red_suits * cards_per_suit
  let top_card_is_red_probability := num_red_cards / num_cards
  top_card_is_red_probability = 1 := by
  sorry

end NUMINAMATH_GPT_red_suit_top_card_probability_l1816_181632


namespace NUMINAMATH_GPT_vampires_after_two_nights_l1816_181662

-- Define the initial conditions and calculations
def initial_vampires : ℕ := 2
def transformation_rate : ℕ := 5
def first_night_vampires : ℕ := initial_vampires * transformation_rate + initial_vampires
def second_night_vampires : ℕ := first_night_vampires * transformation_rate + first_night_vampires

-- Prove that the number of vampires after two nights is 72
theorem vampires_after_two_nights : second_night_vampires = 72 :=
by sorry

end NUMINAMATH_GPT_vampires_after_two_nights_l1816_181662


namespace NUMINAMATH_GPT_transmission_time_l1816_181639

theorem transmission_time :
  let regular_blocks := 70
  let large_blocks := 30
  let chunks_per_regular_block := 800
  let chunks_per_large_block := 1600
  let channel_rate := 200
  let total_chunks := (regular_blocks * chunks_per_regular_block) + (large_blocks * chunks_per_large_block)
  let total_time_seconds := total_chunks / channel_rate
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 8.67 := 
by 
  sorry

end NUMINAMATH_GPT_transmission_time_l1816_181639


namespace NUMINAMATH_GPT_range_of_m_l1816_181688

open Real Set

variable (x m : ℝ)

def p (x : ℝ) := (x + 1) * (x - 1) ≤ 0
def q (x m : ℝ) := (x + 1) * (x - (3 * m - 1)) ≤ 0 ∧ m > 0

theorem range_of_m (hpsuffq : ∀ x, p x → q x m) (hqnotsuffp : ∃ x, q x m ∧ ¬ p x) : m > 2 / 3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1816_181688


namespace NUMINAMATH_GPT_rohan_house_rent_percentage_l1816_181645

variable (salary savings food entertainment conveyance : ℕ)
variable (spend_on_house : ℚ)

-- Given conditions
axiom h1 : salary = 5000
axiom h2 : savings = 1000
axiom h3 : food = 40
axiom h4 : entertainment = 10
axiom h5 : conveyance = 10

-- Define savings percentage
def savings_percentage (salary savings : ℕ) : ℚ := (savings : ℚ) / salary * 100

-- Define percentage equation
def total_percentage (food entertainment conveyance spend_on_house savings_percentage : ℚ) : ℚ :=
  food + spend_on_house + entertainment + conveyance + savings_percentage

-- Prove that house rent percentage is 20%
theorem rohan_house_rent_percentage : 
  food = 40 → entertainment = 10 → conveyance = 10 → salary = 5000 → savings = 1000 → 
  total_percentage 40 10 10 spend_on_house (savings_percentage 5000 1000) = 100 →
  spend_on_house = 20 := by
  intros
  sorry

end NUMINAMATH_GPT_rohan_house_rent_percentage_l1816_181645


namespace NUMINAMATH_GPT_distance_from_origin_l1816_181661

noncomputable def point_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin (x y : ℝ) (h₁ : abs y = 15) (h₂ : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h₃ : x > 2) :
  point_distance x y = Real.sqrt (334 + 4 * Real.sqrt 105) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_origin_l1816_181661


namespace NUMINAMATH_GPT_octagon_side_length_eq_l1816_181691

theorem octagon_side_length_eq (AB BC : ℝ) (AE FB s : ℝ) :
  AE = FB → AE < 5 → AB = 10 → BC = 12 →
  s = -11 + Real.sqrt 242 →
  EF = (10.5 - (Real.sqrt 242) / 2) :=
by
  -- Identified parameters and included all conditions from step a)
  intros h1 h2 h3 h4 h5
  -- statement of the theorem to be proven
  let EF := (10.5 - (Real.sqrt 242) / 2)
  sorry  -- placeholder for proof

end NUMINAMATH_GPT_octagon_side_length_eq_l1816_181691


namespace NUMINAMATH_GPT_bricks_of_other_types_l1816_181689

theorem bricks_of_other_types (A B total other: ℕ) (hA: A = 40) (hB: B = A / 2) (hTotal: total = 150) (hSum: total = A + B + other): 
  other = 90 :=
by sorry

end NUMINAMATH_GPT_bricks_of_other_types_l1816_181689


namespace NUMINAMATH_GPT_vec_subtraction_l1816_181642

-- Definitions
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Condition: a is parallel to b
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Main theorem
theorem vec_subtraction (m : ℝ) (h : are_parallel a (b m)) :
  2 • a - b m = (4, -8) :=
sorry

end NUMINAMATH_GPT_vec_subtraction_l1816_181642


namespace NUMINAMATH_GPT_snow_probability_l1816_181668

theorem snow_probability :
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  1 - p_no_snow_week = 29 / 32 :=
by
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  have p_no_snow_week_eq : p_no_snow_week = 3 / 32 := sorry
  have p_snow_at_least_once_week : 1 - p_no_snow_week = 29 / 32 := sorry
  exact p_snow_at_least_once_week

end NUMINAMATH_GPT_snow_probability_l1816_181668


namespace NUMINAMATH_GPT_math_problem_l1816_181698

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => if a n < 2 * n then a n + 1 else a n

theorem math_problem (n : ℕ) (hn : n > 0) (ha_inc : ∀ m, m > 0 → a m < a (m + 1)) 
  (ha_rec : ∀ m, m > 0 → a (m + 1) ≤ 2 * m) : 
  ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ n = a p - a q := sorry

end NUMINAMATH_GPT_math_problem_l1816_181698


namespace NUMINAMATH_GPT_roots_quadratic_expression_l1816_181690

theorem roots_quadratic_expression (α β : ℝ) (hα : α^2 - 3 * α - 2 = 0) (hβ : β^2 - 3 * β - 2 = 0) :
    7 * α^4 + 10 * β^3 = 544 := 
sorry

end NUMINAMATH_GPT_roots_quadratic_expression_l1816_181690


namespace NUMINAMATH_GPT_part1_subsets_m_0_part2_range_m_l1816_181653

namespace MathProof

variables {α : Type*} {m : ℝ}

def A := {x : ℝ | x^2 + 5 * x - 6 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 3 = 0}
def subsets (A : Set ℝ) := {s : Set ℝ | s ⊆ A}

theorem part1_subsets_m_0 :
  subsets (A ∪ B 0) = {∅, {-6}, {1}, {-3}, {-6,1}, {-6,-3}, {1,-3}, {-6,1,-3}} :=
sorry

theorem part2_range_m (h : ∀ x, x ∈ B m → x ∈ A) : m ≤ -2 :=
sorry

end MathProof

end NUMINAMATH_GPT_part1_subsets_m_0_part2_range_m_l1816_181653


namespace NUMINAMATH_GPT_only_exprC_cannot_be_calculated_with_square_of_binomial_l1816_181604

-- Definitions of our expressions using their variables
def exprA (a b : ℝ) := (a + b) * (a - b)
def exprB (x : ℝ) := (-x + 1) * (-x - 1)
def exprC (y : ℝ) := (y + 1) * (-y - 1)
def exprD (m : ℝ) := (m - 1) * (-1 - m)

-- Statement that only exprC cannot be calculated using the square of a binomial formula
theorem only_exprC_cannot_be_calculated_with_square_of_binomial :
  (∀ a b : ℝ, ∃ (u v : ℝ), exprA a b = u^2 - v^2) ∧
  (∀ x : ℝ, ∃ (u v : ℝ), exprB x = u^2 - v^2) ∧
  (forall m : ℝ, ∃ (u v : ℝ), exprD m = u^2 - v^2) 
  ∧ (∀ v : ℝ, ¬ ∃ (u : ℝ), exprC v = u^2 ∨ (exprC v = - (u^2))) := sorry

end NUMINAMATH_GPT_only_exprC_cannot_be_calculated_with_square_of_binomial_l1816_181604


namespace NUMINAMATH_GPT_work_done_by_gas_l1816_181692

theorem work_done_by_gas (n : ℕ) (R T0 Pa : ℝ) (V0 : ℝ) (W : ℝ) :
  -- Conditions
  n = 1 ∧
  R = 8.314 ∧
  T0 = 320 ∧
  Pa * V0 = n * R * T0 ∧
  -- Question Statement and Correct Answer
  W = Pa * V0 / 2 →
  W = 665 :=
by sorry

end NUMINAMATH_GPT_work_done_by_gas_l1816_181692


namespace NUMINAMATH_GPT_original_price_proof_l1816_181679

noncomputable def original_price (profit selling_price : ℝ) : ℝ :=
  (profit / 0.20)

theorem original_price_proof (P : ℝ) : 
  original_price 600 (P + 600) = 3000 :=
by
  sorry

end NUMINAMATH_GPT_original_price_proof_l1816_181679


namespace NUMINAMATH_GPT_integer_range_2014_l1816_181672

theorem integer_range_2014 : 1000 < 2014 ∧ 2014 < 10000 := by
  sorry

end NUMINAMATH_GPT_integer_range_2014_l1816_181672


namespace NUMINAMATH_GPT_jellybean_proof_l1816_181669

def number_vanilla_jellybeans : ℕ := 120

def number_grape_jellybeans (V : ℕ) : ℕ := 5 * V + 50

def number_strawberry_jellybeans (V : ℕ) : ℕ := (2 * V) / 3

def total_number_jellybeans (V G S : ℕ) : ℕ := V + G + S

def cost_per_vanilla_jellybean : ℚ := 0.05

def cost_per_grape_jellybean : ℚ := 0.08

def cost_per_strawberry_jellybean : ℚ := 0.07

def total_cost_jellybeans (V G S : ℕ) : ℚ := 
  (cost_per_vanilla_jellybean * V) + 
  (cost_per_grape_jellybean * G) + 
  (cost_per_strawberry_jellybean * S)

theorem jellybean_proof :
  ∃ (V G S : ℕ), 
    V = number_vanilla_jellybeans ∧
    G = number_grape_jellybeans V ∧
    S = number_strawberry_jellybeans V ∧
    total_number_jellybeans V G S = 850 ∧
    total_cost_jellybeans V G S = 63.60 :=
by
  sorry

end NUMINAMATH_GPT_jellybean_proof_l1816_181669


namespace NUMINAMATH_GPT_cos_neg_79_pi_over_6_l1816_181603

theorem cos_neg_79_pi_over_6 : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_neg_79_pi_over_6_l1816_181603


namespace NUMINAMATH_GPT_intersection_eq_l1816_181628

namespace SetIntersection

open Set

-- Definitions of sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Prove the intersection of A and B is {1, 2}
theorem intersection_eq : A ∩ B = {1, 2} :=
by
  sorry

end SetIntersection

end NUMINAMATH_GPT_intersection_eq_l1816_181628


namespace NUMINAMATH_GPT_first_number_less_than_twice_second_l1816_181636

theorem first_number_less_than_twice_second (x y z : ℕ) : 
  x + y = 50 ∧ y = 19 ∧ x = 2 * y - z → z = 7 :=
by sorry

end NUMINAMATH_GPT_first_number_less_than_twice_second_l1816_181636


namespace NUMINAMATH_GPT_product_sum_l1816_181616

theorem product_sum (y x z: ℕ) 
  (h1: 2014 + y = 2015 + x) 
  (h2: 2015 + x = 2016 + z) 
  (h3: y * x * z = 504): 
  y * x + x * z = 128 := 
by 
  sorry

end NUMINAMATH_GPT_product_sum_l1816_181616


namespace NUMINAMATH_GPT_smallest_composite_no_prime_factors_lt_20_l1816_181606

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n : ℕ, (n > 1 ∧ ¬ Prime n ∧ (∀ p : ℕ, Prime p → p < 20 → p ∣ n → False)) ∧ n = 529 :=
by
  sorry

end NUMINAMATH_GPT_smallest_composite_no_prime_factors_lt_20_l1816_181606


namespace NUMINAMATH_GPT_min_value_l1816_181629

variable {α : Type*} [LinearOrderedField α]

-- Define a geometric sequence with strictly positive terms
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ (q : α), q > 0 ∧ ∀ n, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → α) (S : ℕ → α)
variables (h_geom : is_geometric_sequence a)
variables (h_pos : ∀ n, a n > 0)
variables (h_a23 : a 2 * a 6 = 4) (h_a3 : a 3 = 1)

-- Sum of the first n terms of a geometric sequence
def sum_first_n (a : ℕ → α) (n : ℕ) : α :=
  if n = 0 then 0
  else a 0 * ((1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0)))

-- Statement of the theorem
theorem min_value (a : ℕ → α) (S : ℕ → α) 
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a23 : a 2 * a 6 = 4)
  (h_a3 : a 3 = 1)
  (h_Sn : ∀ n, S n = sum_first_n a n) :
  ∃ n, n = 3 ∧ (S n + 9 / 4) ^ 2 / (2 * a n) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_l1816_181629


namespace NUMINAMATH_GPT_max_value_l1816_181643

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) := ∀ {a b}, a < b → f a < f b

theorem max_value (f : ℝ → ℝ) (x y : ℝ)
  (h_odd : is_odd f)
  (h_increasing : is_increasing f)
  (h_eq : f (x^2 - 2 * x) + f y = 0) :
  2 * x + y ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_l1816_181643


namespace NUMINAMATH_GPT_fraction_equivalence_l1816_181607

theorem fraction_equivalence (a b : ℝ) (h : ((1 / a) + (1 / b)) / ((1 / a) - (1 / b)) = 2020) : (a + b) / (a - b) = 2020 :=
sorry

end NUMINAMATH_GPT_fraction_equivalence_l1816_181607


namespace NUMINAMATH_GPT_polar_to_rectangular_l1816_181667

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 6) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (3, -3 * Real.sqrt 3) :=
by
  -- Definitions and assertions from the conditions
  have cos_theta : Real.cos (5 * Real.pi / 3) = 1 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted
  have sin_theta : Real.sin (5 * Real.pi / 3) = - Real.sqrt 3 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted

  -- Proof that the converted coordinates match the expected result
  rw [hr, hθ, cos_theta, sin_theta]
  simp
  -- Detailed proof steps to verify (6 * (1 / 2), 6 * (- Real.sqrt 3 / 2)) = (3, -3 * Real.sqrt 3) omitted
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l1816_181667


namespace NUMINAMATH_GPT_correct_option_D_l1816_181631

theorem correct_option_D (a : ℝ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end NUMINAMATH_GPT_correct_option_D_l1816_181631


namespace NUMINAMATH_GPT_ratio_of_areas_l1816_181693

theorem ratio_of_areas (r : ℝ) (h1 : r > 0) : 
  let OX := r / 3
  let area_OP := π * r ^ 2
  let area_OX := π * (OX) ^ 2
  (area_OX / area_OP) = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1816_181693


namespace NUMINAMATH_GPT_sum_of_terms_l1816_181671

def geometric_sequence (a b c d : ℝ) :=
  ∃ q : ℝ, a = b / q ∧ c = b * q ∧ d = c * q

def symmetric_sequence_of_length_7 (s : Fin 8 → ℝ) :=
  ∀ i : Fin 8, s i = s (Fin.mk (7 - i) sorry)

def sequence_conditions (s : Fin 8 → ℝ) :=
  symmetric_sequence_of_length_7 s ∧
  geometric_sequence (s ⟨1,sorry⟩) (s ⟨2,sorry⟩) (s ⟨3,sorry⟩) (s ⟨4,sorry⟩) ∧
  s ⟨1,sorry⟩ = 2 ∧
  s ⟨3,sorry⟩ = 8

theorem sum_of_terms (s : Fin 8 → ℝ) (h : sequence_conditions s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = 44 ∨
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = -4 :=
sorry

end NUMINAMATH_GPT_sum_of_terms_l1816_181671


namespace NUMINAMATH_GPT_geometric_figure_perimeter_l1816_181677

theorem geometric_figure_perimeter (A : ℝ) (n : ℝ) (area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  A = 216 ∧ n = 6 ∧ area = A / n ∧ side_length = Real.sqrt area ∧ perimeter = 2 * (3 * side_length + 2 * side_length) + 2 * side_length →
  perimeter = 72 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_figure_perimeter_l1816_181677


namespace NUMINAMATH_GPT_inequality_xyz_l1816_181680

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * ((x ^ 3 + y ^ 3 + z ^ 3) ^ (1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_xyz_l1816_181680


namespace NUMINAMATH_GPT_calculation_result_l1816_181670

theorem calculation_result:
  5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l1816_181670


namespace NUMINAMATH_GPT_election_ratio_l1816_181697

theorem election_ratio (X Y : ℝ) 
  (h : 0.74 * X + 0.5000000000000002 * Y = 0.66 * (X + Y)) : 
  X / Y = 2 :=
by sorry

end NUMINAMATH_GPT_election_ratio_l1816_181697


namespace NUMINAMATH_GPT_inradius_inequality_l1816_181613

theorem inradius_inequality
  (r r_A r_B r_C : ℝ) 
  (h_inscribed_circle: r > 0) 
  (h_tangent_circles_A: r_A > 0) 
  (h_tangent_circles_B: r_B > 0) 
  (h_tangent_circles_C: r_C > 0)
  : r ≤ r_A + r_B + r_C :=
  sorry

end NUMINAMATH_GPT_inradius_inequality_l1816_181613


namespace NUMINAMATH_GPT_smallest_possible_area_of_2020th_square_l1816_181655

theorem smallest_possible_area_of_2020th_square :
  ∃ A : ℕ, (∃ n : ℕ, n * n = 2019 + A) ∧ A ≠ 1 ∧
  ∀ A' : ℕ, A' > 0 ∧ (∃ n : ℕ, n * n = 2019 + A') ∧ A' ≠ 1 → A ≤ A' :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_area_of_2020th_square_l1816_181655


namespace NUMINAMATH_GPT_simplify_expression_l1816_181619

theorem simplify_expression (a : ℝ) (h : a ≠ -1) : a - 1 + 1 / (a + 1) = a^2 / (a + 1) :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l1816_181619


namespace NUMINAMATH_GPT_bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l1816_181664

variables {a b c : ℝ}
-- Given conditions from Vieta's formulas for the polynomial x^3 - 20x^2 + 22
axiom vieta1 : a + b + c = 20
axiom vieta2 : a * b + b * c + c * a = 0
axiom vieta3 : a * b * c = -22

theorem bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3 (a b c : ℝ)
  (h1 : a + b + c = 20)
  (h2 : a * b + b * c + c * a = 0)
  (h3 : a * b * c = -22) :
  (b * c / a^2) + (a * c / b^2) + (a * b / c^2) = 3 := 
  sorry

end NUMINAMATH_GPT_bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l1816_181664


namespace NUMINAMATH_GPT_log_ride_cost_l1816_181634

noncomputable def cost_of_log_ride (ferris_wheel : ℕ) (roller_coaster : ℕ) (initial_tickets : ℕ) (additional_tickets : ℕ) : ℕ :=
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  total_needed - total_known

theorem log_ride_cost :
  cost_of_log_ride 6 5 2 16 = 7 :=
by
  -- specify the values for ferris_wheel, roller_coaster, initial_tickets, additional_tickets
  let ferris_wheel := 6
  let roller_coaster := 5
  let initial_tickets := 2
  let additional_tickets := 16
  -- calculate the cost of the log ride
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  let log_ride := total_needed - total_known
  -- assert that the cost of the log ride is 7
  have : log_ride = 7 := by
    -- use arithmetic to justify the answer
    sorry
  exact this

end NUMINAMATH_GPT_log_ride_cost_l1816_181634


namespace NUMINAMATH_GPT_aerith_seat_l1816_181601

-- Let the seats be numbered 1 through 8
-- Assigned seats for Aerith, Bob, Chebyshev, Descartes, Euler, Fermat, Gauss, and Hilbert
variables (a b c d e f g h : ℕ)

-- Define the conditions described in the problem
axiom Bob_assigned : b = 1
axiom Chebyshev_assigned : c = g + 2
axiom Descartes_assigned : d = f - 1
axiom Euler_assigned : e = h - 4
axiom Fermat_assigned : f = d + 5
axiom Gauss_assigned : g = e + 1
axiom Hilbert_assigned : h = a - 3

-- Provide the proof statement to find whose seat Aerith sits
theorem aerith_seat : a = c := sorry

end NUMINAMATH_GPT_aerith_seat_l1816_181601


namespace NUMINAMATH_GPT_derivative_at_zero_l1816_181617

def f (x : ℝ) : ℝ := x^3

theorem derivative_at_zero : deriv f 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l1816_181617


namespace NUMINAMATH_GPT_gcd_45_81_63_l1816_181641

theorem gcd_45_81_63 : Nat.gcd 45 (Nat.gcd 81 63) = 9 := 
sorry

end NUMINAMATH_GPT_gcd_45_81_63_l1816_181641


namespace NUMINAMATH_GPT_can_construct_length_one_l1816_181609

noncomputable def possible_to_construct_length_one_by_folding (n : ℕ) : Prop :=
  ∃ k ≤ 10, ∃ (segment_constructed : ℝ), segment_constructed = 1

theorem can_construct_length_one : possible_to_construct_length_one_by_folding 2016 :=
by sorry

end NUMINAMATH_GPT_can_construct_length_one_l1816_181609


namespace NUMINAMATH_GPT_no_unhappy_days_l1816_181600

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_GPT_no_unhappy_days_l1816_181600


namespace NUMINAMATH_GPT_rose_bushes_planted_l1816_181684

-- Define the conditions as variables
variable (current_bushes planted_bushes total_bushes : Nat)
variable (h1 : current_bushes = 2) (h2 : total_bushes = 6)
variable (h3 : total_bushes = current_bushes + planted_bushes)

theorem rose_bushes_planted : planted_bushes = 4 := by
  sorry

end NUMINAMATH_GPT_rose_bushes_planted_l1816_181684


namespace NUMINAMATH_GPT_find_mn_solutions_l1816_181651

theorem find_mn_solutions :
  ∀ (m n : ℤ), m^5 - n^5 = 16 * m * n →
  (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_mn_solutions_l1816_181651


namespace NUMINAMATH_GPT_find_value_of_m_l1816_181647

noncomputable def m : ℤ := -2

theorem find_value_of_m (m : ℤ) :
  (m-2) ≠ 0 ∧ (m^2 - 3 = 1) → m = -2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_value_of_m_l1816_181647


namespace NUMINAMATH_GPT_unique_triplet_exists_l1816_181674

theorem unique_triplet_exists (a b p : ℕ) (hp : Nat.Prime p) : 
  (a + b)^p = p^a + p^b → (a = 1 ∧ b = 1 ∧ p = 2) :=
by sorry

end NUMINAMATH_GPT_unique_triplet_exists_l1816_181674


namespace NUMINAMATH_GPT_solve_for_x_l1816_181648

noncomputable def vec (x y : ℝ) : ℝ × ℝ := (x, y)

theorem solve_for_x (x : ℝ) :
  let a := vec 1 2
  let b := vec x 1
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 - 2 * b.1, 2 * a.2 - 2 * b.2)
  (u.1 * v.2 = u.2 * v.1) → x = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1816_181648


namespace NUMINAMATH_GPT_joe_avg_speed_l1816_181644

noncomputable def total_distance : ℝ :=
  420 + 250 + 120 + 65

noncomputable def total_time : ℝ :=
  (420 / 60) + (250 / 50) + (120 / 40) + (65 / 70)

noncomputable def avg_speed : ℝ :=
  total_distance / total_time

theorem joe_avg_speed : avg_speed = 53.67 := by
  sorry

end NUMINAMATH_GPT_joe_avg_speed_l1816_181644


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l1816_181624

theorem geometric_progression_common_ratio :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l1816_181624


namespace NUMINAMATH_GPT_candle_length_sum_l1816_181673

theorem candle_length_sum (l s : ℕ) (x : ℤ) 
  (h1 : l = s + 32)
  (h2 : s = (5 * x)) 
  (h3 : l = (7 * (3 * x))) :
  l + s = 52 := 
sorry

end NUMINAMATH_GPT_candle_length_sum_l1816_181673


namespace NUMINAMATH_GPT_greatest_possible_median_l1816_181650

theorem greatest_possible_median {k m r s t : ℕ} 
  (h_mean : (k + m + r + s + t) / 5 = 18) 
  (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) 
  (h_t : t = 40) :
  r = 23 := sorry

end NUMINAMATH_GPT_greatest_possible_median_l1816_181650


namespace NUMINAMATH_GPT_size_of_angle_C_l1816_181656

theorem size_of_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 5) 
  (h2 : b + c = 2 * a) 
  (h3 : 3 * Real.sin A = 5 * Real.sin B) : 
  C = 2 * Real.pi / 3 := 
sorry

end NUMINAMATH_GPT_size_of_angle_C_l1816_181656


namespace NUMINAMATH_GPT_distance_AC_100_l1816_181675

theorem distance_AC_100 (d_AB : ℝ) (t1 : ℝ) (t2 : ℝ) (AC : ℝ) (CB : ℝ) :
  d_AB = 150 ∧ t1 = 3 ∧ t2 = 12 ∧ d_AB = AC + CB ∧ AC / 3 = CB / 12 → AC = 100 := 
by
  sorry

end NUMINAMATH_GPT_distance_AC_100_l1816_181675


namespace NUMINAMATH_GPT_johns_weekly_earnings_increase_l1816_181657

def combined_percentage_increase (initial final : ℕ) : ℕ :=
  ((final - initial) * 100) / initial

theorem johns_weekly_earnings_increase :
  combined_percentage_increase 40 60 = 50 :=
by
  sorry

end NUMINAMATH_GPT_johns_weekly_earnings_increase_l1816_181657


namespace NUMINAMATH_GPT_find_n_l1816_181623

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n % 9 = 4897 % 9 ∧ n = 1 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_find_n_l1816_181623


namespace NUMINAMATH_GPT_a_equals_5_l1816_181605

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9
def f' (x : ℝ) (a : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem a_equals_5 (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ f' x a = 0) → a = 5 := 
by
  sorry

end NUMINAMATH_GPT_a_equals_5_l1816_181605


namespace NUMINAMATH_GPT_intersection_M_N_l1816_181681

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Prove the intersection of M and N equals (1, 2)
theorem intersection_M_N :
  ∀ x, x ∈ M ∩ N ↔ 1 < x ∧ x < 2 :=
by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1816_181681


namespace NUMINAMATH_GPT_curve_left_of_line_l1816_181621

theorem curve_left_of_line (x y : ℝ) : x^3 + 2*y^2 = 8 → x ≤ 2 := 
sorry

end NUMINAMATH_GPT_curve_left_of_line_l1816_181621


namespace NUMINAMATH_GPT_probability_odd_product_l1816_181646

theorem probability_odd_product :
  let box1 := [1, 2, 3, 4]
  let box2 := [1, 2, 3, 4]
  let total_outcomes := 4 * 4
  let favorable_outcomes := [(1,1), (1,3), (3,1), (3,3)]
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_probability_odd_product_l1816_181646


namespace NUMINAMATH_GPT_perimeter_of_staircase_region_l1816_181694

-- Definitions according to the conditions.
def staircase_region.all_right_angles : Prop := True -- Given condition that all angles are right angles.
def staircase_region.side_length : ℕ := 1 -- Given condition that the side length of each congruent side is 1 foot.
def staircase_region.total_area : ℕ := 120 -- Given condition that the total area of the region is 120 square feet.
def num_sides : ℕ := 12 -- Number of congruent sides.

-- The question is to prove that the perimeter of the region is 36 feet.
theorem perimeter_of_staircase_region : 
  (num_sides * staircase_region.side_length + 
    15 + -- length added to complete the larger rectangle assuming x = 15
    9   -- length added to complete the larger rectangle assuming y = 9
  ) = 36 := 
by
  -- Given and facts are already logically considered to prove (conditions and right angles are trivial)
  sorry

end NUMINAMATH_GPT_perimeter_of_staircase_region_l1816_181694


namespace NUMINAMATH_GPT_fg_2_eq_9_l1816_181663

def f (x: ℝ) := x^2
def g (x: ℝ) := -4 * x + 5

theorem fg_2_eq_9 : f (g 2) = 9 :=
by
  sorry

end NUMINAMATH_GPT_fg_2_eq_9_l1816_181663


namespace NUMINAMATH_GPT_ellipse_foci_distance_l1816_181660

-- Definitions based on the problem conditions
def ellipse_eq (x y : ℝ) :=
  Real.sqrt (((x - 4)^2) + ((y - 5)^2)) + Real.sqrt (((x + 6)^2) + ((y + 9)^2)) = 22

def focus1 : (ℝ × ℝ) := (4, -5)
def focus2 : (ℝ × ℝ) := (-6, 9)

-- Statement of the problem
noncomputable def distance_between_foci : ℝ :=
  Real.sqrt (((focus1.1 + 6)^2) + ((focus1.2 - 9)^2))

-- Proof statement
theorem ellipse_foci_distance : distance_between_foci = 2 * Real.sqrt 74 := by
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l1816_181660


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1816_181659

theorem quadratic_inequality_solution :
  (∀ x : ℝ, x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3) → -9 * x^2 + 6 * x + 1 < 0) ∧
  (∀ x : ℝ, -9 * x^2 + 6 * x + 1 < 0 → x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1816_181659


namespace NUMINAMATH_GPT_points_on_curve_is_parabola_l1816_181610

theorem points_on_curve_is_parabola (X Y : ℝ) (h : Real.sqrt X + Real.sqrt Y = 1) :
  ∃ a b c : ℝ, Y = a * X^2 + b * X + c :=
sorry

end NUMINAMATH_GPT_points_on_curve_is_parabola_l1816_181610


namespace NUMINAMATH_GPT_drivers_distance_difference_l1816_181611

noncomputable def total_distance_driven (initial_distance : ℕ) (speed_A : ℕ) (speed_B : ℕ) (start_delay : ℕ) : ℕ := sorry

theorem drivers_distance_difference
  (initial_distance : ℕ)
  (speed_A : ℕ)
  (speed_B : ℕ)
  (start_delay : ℕ)
  (correct_difference : ℕ)
  (h_initial : initial_distance = 1025)
  (h_speed_A : speed_A = 90)
  (h_speed_B : speed_B = 80)
  (h_start_delay : start_delay = 1)
  (h_correct_difference : correct_difference = 145) :
  total_distance_driven initial_distance speed_A speed_B start_delay = correct_difference :=
sorry

end NUMINAMATH_GPT_drivers_distance_difference_l1816_181611


namespace NUMINAMATH_GPT_carrots_weight_l1816_181625

-- Let the weight of the carrots be denoted by C (in kg).
variables (C : ℕ)

-- Conditions:
-- The merchant installed 13 kg of zucchini and 8 kg of broccoli.
-- He sold only half of the total, which amounted to 18 kg, so the total weight was 36 kg.
def conditions := (C + 13 + 8 = 36)

-- Prove that the weight of the carrots installed is 15 kg.
theorem carrots_weight (H : C + 13 + 8 = 36) : C = 15 :=
by {
  sorry -- proof to be filled in
}

end NUMINAMATH_GPT_carrots_weight_l1816_181625


namespace NUMINAMATH_GPT_triangle_area_l1816_181635

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) (h4 : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 336 := 
by 
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_triangle_area_l1816_181635


namespace NUMINAMATH_GPT_tickets_difference_is_cost_l1816_181665

def tickets_won : ℝ := 48.5
def yoyo_cost : ℝ := 11.7
def tickets_left (w : ℝ) (c : ℝ) : ℝ := w - c
def difference (w : ℝ) (l : ℝ) : ℝ := w - l

theorem tickets_difference_is_cost :
  difference tickets_won (tickets_left tickets_won yoyo_cost) = yoyo_cost :=
by
  -- Proof will be written here
  sorry

end NUMINAMATH_GPT_tickets_difference_is_cost_l1816_181665


namespace NUMINAMATH_GPT_bug_total_distance_l1816_181612

/-- 
A bug starts at position 3 on a number line. It crawls to -4, then to 7, and finally to 1.
The total distance the bug crawls is 24 units.
-/
theorem bug_total_distance : 
  let start := 3
  let first_stop := -4
  let second_stop := 7
  let final_position := 1
  let distance := abs (first_stop - start) + abs (second_stop - first_stop) + abs (final_position - second_stop)
  distance = 24 := 
by
  sorry

end NUMINAMATH_GPT_bug_total_distance_l1816_181612


namespace NUMINAMATH_GPT_degree_le_of_lt_eventually_l1816_181622

open Polynomial

theorem degree_le_of_lt_eventually {P Q : Polynomial ℝ} (h_exists : ∃ N : ℝ, ∀ x : ℝ, x > N → P.eval x < Q.eval x) :
  P.degree ≤ Q.degree :=
sorry

end NUMINAMATH_GPT_degree_le_of_lt_eventually_l1816_181622


namespace NUMINAMATH_GPT_division_result_l1816_181676

theorem division_result (a b : ℕ) (ha : a = 7) (hb : b = 3) :
    ((a^3 + b^3) / (a^2 - a * b + b^2) = 10) := 
by
  sorry

end NUMINAMATH_GPT_division_result_l1816_181676


namespace NUMINAMATH_GPT_hyperbola_asymptote_ratio_l1816_181618

theorem hyperbola_asymptote_ratio
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1))
  (h₃ : ∀ m n: ℝ, m * n = -1 → ∃ θ: ℝ, θ = 90* (π / 180)): 
  a / b = 1 := 
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_ratio_l1816_181618
