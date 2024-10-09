import Mathlib

namespace total_calories_consumed_l1661_166154

def caramel_cookies := 10
def caramel_calories := 18

def chocolate_chip_cookies := 8
def chocolate_chip_calories := 22

def peanut_butter_cookies := 7
def peanut_butter_calories := 24

def selected_caramel_cookies := 5
def selected_chocolate_chip_cookies := 3
def selected_peanut_butter_cookies := 2

theorem total_calories_consumed : 
  (selected_caramel_cookies * caramel_calories) + 
  (selected_chocolate_chip_cookies * chocolate_chip_calories) + 
  (selected_peanut_butter_cookies * peanut_butter_calories) = 204 := 
by
  sorry

end total_calories_consumed_l1661_166154


namespace machines_finish_together_in_2_hours_l1661_166176

def machineA_time := 4
def machineB_time := 12
def machineC_time := 6

def machineA_rate := 1 / machineA_time
def machineB_rate := 1 / machineB_time
def machineC_rate := 1 / machineC_time

def combined_rate := machineA_rate + machineB_rate + machineC_rate
def total_time := 1 / combined_rate

-- We want to prove that the total_time for machines A, B, and C to finish the job together is 2 hours.
theorem machines_finish_together_in_2_hours : total_time = 2 := by
  sorry

end machines_finish_together_in_2_hours_l1661_166176


namespace gift_card_remaining_l1661_166101

theorem gift_card_remaining (initial_amount : ℕ) (half_monday : ℕ) (quarter_tuesday : ℕ) : 
  initial_amount = 200 → 
  half_monday = initial_amount / 2 →
  quarter_tuesday = (initial_amount - half_monday) / 4 →
  initial_amount - half_monday - quarter_tuesday = 75 :=
by
  intros h_init h_half h_quarter
  rw [h_init, h_half, h_quarter]
  sorry

end gift_card_remaining_l1661_166101


namespace Drew_age_is_12_l1661_166148

def Sam_age_current : ℕ := 46
def Sam_age_in_five_years : ℕ := Sam_age_current + 5

def Drew_age_now (D : ℕ) : Prop :=
  Sam_age_in_five_years = 3 * (D + 5)

theorem Drew_age_is_12 (D : ℕ) (h : Drew_age_now D) : D = 12 :=
by
  sorry

end Drew_age_is_12_l1661_166148


namespace arc_length_of_circle_l1661_166191

theorem arc_length_of_circle (r : ℝ) (alpha : ℝ) (h_r : r = 10) (h_alpha : alpha = (2 * Real.pi) / 6) : 
  (alpha * r) = (10 * Real.pi) / 3 :=
by
  rw [h_r, h_alpha]
  sorry

end arc_length_of_circle_l1661_166191


namespace find_delta_l1661_166131

theorem find_delta (p q Δ : ℕ) (h₁ : Δ + q = 73) (h₂ : 2 * (Δ + q) + p = 172) (h₃ : p = 26) : Δ = 12 :=
by
  sorry

end find_delta_l1661_166131


namespace susie_rhode_island_reds_l1661_166138

variable (R G B_R B_G : ℕ)

def susie_golden_comets := G = 6
def britney_rir := B_R = 2 * R
def britney_golden_comets := B_G = G / 2
def britney_more_chickens := B_R + B_G = R + G + 8

theorem susie_rhode_island_reds
  (h1 : susie_golden_comets G)
  (h2 : britney_rir R B_R)
  (h3 : britney_golden_comets G B_G)
  (h4 : britney_more_chickens R G B_R B_G) :
  R = 11 :=
by
  sorry

end susie_rhode_island_reds_l1661_166138


namespace locus_of_point_is_circle_l1661_166156

theorem locus_of_point_is_circle (x y : ℝ) 
  (h : 10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3 * x - 4 * y|) : 
  ∃ (c : ℝ) (r : ℝ), ∀ (x y : ℝ), (x - c)^2 + (y - c)^2 = r^2 := 
sorry

end locus_of_point_is_circle_l1661_166156


namespace problem_proof_equality_cases_l1661_166163

theorem problem_proof (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (x * y - 10) ^ 2 ≥ 64 := sorry

theorem equality_cases (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10) ^ 2 = 64 ↔ ((x,y) = (1, 2) ∨ (x,y) = (-3, -6)) := sorry

end problem_proof_equality_cases_l1661_166163


namespace solve_eqn_l1661_166146

theorem solve_eqn {x : ℝ} : x^4 + (3 - x)^4 = 130 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end solve_eqn_l1661_166146


namespace complex_number_condition_l1661_166112

theorem complex_number_condition (z : ℂ) (h : z^2 + z + 1 = 0) :
  2 * z^96 + 3 * z^97 + 4 * z^98 + 5 * z^99 + 6 * z^100 = 3 + 5 * z := 
by 
  sorry

end complex_number_condition_l1661_166112


namespace tiling_ratio_l1661_166129

theorem tiling_ratio (n a b : ℕ) (ha : a ≠ 0) (H : b = a * 2^(n/2)) :
  b / a = 2^(n/2) :=
  by
  sorry

end tiling_ratio_l1661_166129


namespace S_is_line_l1661_166178

open Complex

noncomputable def S : Set ℂ := { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ 3 * y + 4 * x = 0 }

theorem S_is_line :
  ∃ (m b : ℝ), S = { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ x = m * y + b } :=
sorry

end S_is_line_l1661_166178


namespace lincoln_high_school_students_l1661_166102

theorem lincoln_high_school_students (total students_in_either_or_both_clubs students_in_photography students_in_science : ℕ)
  (h1 : total = 300)
  (h2 : students_in_photography = 120)
  (h3 : students_in_science = 140)
  (h4 : students_in_either_or_both_clubs = 220):
  ∃ x, x = 40 ∧ (students_in_photography + students_in_science - students_in_either_or_both_clubs = x) := 
by
  use 40
  sorry

end lincoln_high_school_students_l1661_166102


namespace minimize_M_l1661_166104

noncomputable def M (x y : ℝ) : ℝ := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9

theorem minimize_M : ∃ x y, M x y = 5 ∧ x = -3 ∧ y = -2 :=
by
  sorry

end minimize_M_l1661_166104


namespace problem1_problem2_l1661_166193

-- Proof problem (1)
theorem problem1 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 1 < x ∧ x < 2} ∧ m = 1 →
  (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := 
by 
  sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} ∧ B = {x : ℝ | 2 * m - 1 < x ∧ x < m + 1} →
  (B ⊆ A ↔ (m ≥ 2 ∨ (-1 ≤ m ∧ m < 2))) := 
by 
  sorry

end problem1_problem2_l1661_166193


namespace geometric_sum_six_l1661_166113

theorem geometric_sum_six (a r : ℚ) (n : ℕ) 
  (hn₁ : a = 1/4) 
  (hn₂ : r = 1/2) 
  (hS: a * (1 - r^n) / (1 - r) = 63/128) : 
  n = 6 :=
by
  -- Statement to be Proven
  rw [hn₁, hn₂] at hS
  sorry

end geometric_sum_six_l1661_166113


namespace kite_area_eq_twenty_l1661_166114

theorem kite_area_eq_twenty :
  let base := 10
  let height := 2
  let area_of_triangle := (1 / 2 : ℝ) * base * height
  let total_area := 2 * area_of_triangle
  total_area = 20 :=
by
  sorry

end kite_area_eq_twenty_l1661_166114


namespace first_player_wins_l1661_166168

theorem first_player_wins :
  ∀ {table : Type} {coin : Type} 
  (can_place : table → coin → Prop) -- function defining if a coin can be placed on the table
  (not_overlap : ∀ (t : table) (c1 c2 : coin), (can_place t c1 ∧ can_place t c2) → c1 ≠ c2) -- coins do not overlap
  (first_move_center : table → coin) -- first player places the coin at the center
  (mirror_move : table → coin → coin), -- function to place a coin symmetrically
  (∃ strategy : (table → Prop) → (coin → Prop),
    (∀ (t : table) (p : table → Prop), p t → strategy p (mirror_move t (first_move_center t))) ∧ 
    (∀ (t : table) (p : table → Prop), strategy p (first_move_center t) → p t)) := sorry

end first_player_wins_l1661_166168


namespace petals_in_garden_l1661_166124

def lilies_count : ℕ := 8
def tulips_count : ℕ := 5
def petals_per_lily : ℕ := 6
def petals_per_tulip : ℕ := 3

def total_petals : ℕ := lilies_count * petals_per_lily + tulips_count * petals_per_tulip

theorem petals_in_garden : total_petals = 63 := by
  sorry

end petals_in_garden_l1661_166124


namespace g_odd_l1661_166126

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_odd {x₁ x₂ : ℝ} 
  (h₁ : |f x₁ + f x₂| ≥ |g x₁ + g x₂|)
  (hf_odd : ∀ x, f x = -f (-x)) : ∀ x, g x = -g (-x) :=
by
  -- The proof would go here, but it's omitted for the purpose of this translation.
  sorry

end g_odd_l1661_166126


namespace decimal_between_0_996_and_0_998_ne_0_997_l1661_166177

theorem decimal_between_0_996_and_0_998_ne_0_997 :
  ∃ x : ℝ, 0.996 < x ∧ x < 0.998 ∧ x ≠ 0.997 :=
by
  sorry

end decimal_between_0_996_and_0_998_ne_0_997_l1661_166177


namespace best_discount_option_l1661_166107

-- Define the original price
def original_price : ℝ := 100

-- Define the discount functions for each option
def option_A : ℝ := original_price * (1 - 0.20)
def option_B : ℝ := (original_price * (1 - 0.10)) * (1 - 0.10)
def option_C : ℝ := (original_price * (1 - 0.15)) * (1 - 0.05)
def option_D : ℝ := (original_price * (1 - 0.05)) * (1 - 0.15)

-- Define the theorem stating that option A gives the best price
theorem best_discount_option : option_A ≤ option_B ∧ option_A ≤ option_C ∧ option_A ≤ option_D :=
by {
  sorry
}

end best_discount_option_l1661_166107


namespace arrangement_count_is_43200_l1661_166190

noncomputable def number_of_arrangements : Nat :=
  let number_of_boys := 6
  let number_of_girls := 3
  let boys_arrangements := Nat.factorial number_of_boys
  let spaces := number_of_boys - 1
  let girls_arrangements := Nat.factorial (spaces) / Nat.factorial (spaces - number_of_girls)
  boys_arrangements * girls_arrangements

theorem arrangement_count_is_43200 :
  number_of_arrangements = 43200 := by
  sorry

end arrangement_count_is_43200_l1661_166190


namespace find_s_base_10_l1661_166100

-- Defining the conditions of the problem
def s_in_base_b_equals_42 (b : ℕ) : Prop :=
  let factor_1 := b + 3
  let factor_2 := b + 4
  let factor_3 := b + 5
  let produced_number := factor_1 * factor_2 * factor_3
  produced_number = 2 * b^3 + 3 * b^2 + 2 * b + 5

-- The proof problem as a Lean 4 statement
theorem find_s_base_10 :
  (∃ b : ℕ, s_in_base_b_equals_42 b) →
  13 + 14 + 15 = 42 :=
sorry

end find_s_base_10_l1661_166100


namespace measure_of_theta_l1661_166175

theorem measure_of_theta 
  (ACB FEG DCE DEC : ℝ)
  (h1 : ACB = 10)
  (h2 : FEG = 26)
  (h3 : DCE = 14)
  (h4 : DEC = 33) : θ = 11 :=
by
  sorry

end measure_of_theta_l1661_166175


namespace sufficient_but_not_necessary_condition_circle_l1661_166132

theorem sufficient_but_not_necessary_condition_circle {a : ℝ} (h : a = 1) :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 → (∀ a, a < 2 → (x - 1)^2 + (y + 1)^2 = 2 - a) :=
by
  sorry

end sufficient_but_not_necessary_condition_circle_l1661_166132


namespace intersection_solution_l1661_166169

-- Define lines
def line1 (x : ℝ) : ℝ := -x + 4
def line2 (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

-- Define system of equations
def system1 (x y : ℝ) : Prop := x + y = 4
def system2 (x y m : ℝ) : Prop := 2 * x - y + m = 0

-- Proof statement
theorem intersection_solution (m : ℝ) (n : ℝ) :
  (system1 3 n) ∧ (system2 3 n m) ∧ (line1 3 = n) ∧ (line2 3 m = n) →
  (3, n) = (3, 1) :=
  by 
  -- The proof would go here
  sorry

end intersection_solution_l1661_166169


namespace complex_expression_value_l1661_166116

theorem complex_expression_value :
  (i^3 * (1 + i)^2 = 2) :=
by
  sorry

end complex_expression_value_l1661_166116


namespace tank_fraction_after_adding_water_l1661_166147

noncomputable def fraction_of_tank_full 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  : ℚ :=
(initial_fraction * total_capacity + additional_water) / total_capacity

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  (h_initial : initial_fraction = 3 / 4) 
  (h_addition : additional_water = 4) 
  (h_capacity : total_capacity = 32) 
: fraction_of_tank_full initial_fraction additional_water total_capacity = 7 / 8 :=
by
  sorry

end tank_fraction_after_adding_water_l1661_166147


namespace negation_of_one_odd_l1661_166192

-- Given a, b, c are natural numbers
def exactly_one_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 1)

def not_exactly_one_odd (a b c : ℕ) : Prop :=
  ¬ exactly_one_odd a b c

def at_least_two_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1) ∨
  (a % 2 = 1 ∧ c % 2 = 1) ∨
  (b % 2 = 1 ∧ c % 2 = 1)

def all_even (a b c : ℕ) : Prop :=
  (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0)

theorem negation_of_one_odd (a b c : ℕ) : ¬ exactly_one_odd a b c ↔ all_even a b c ∨ at_least_two_odd a b c := by
  sorry

end negation_of_one_odd_l1661_166192


namespace statement1_statement2_statement3_statement4_correctness_A_l1661_166194

variables {a b : Line} {α β γ : Plane}

def perpendicular (a : Line) (α : Plane) : Prop := sorry
def parallel (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- Statement ①: If a ⊥ α and b ⊥ α, then a ∥ b
theorem statement1 (h1 : perpendicular a α) (h2 : perpendicular b α) : parallel a b := sorry

-- Statement ②: If a ⊥ α, b ⊥ β, and a ∥ b, then α ∥ β
theorem statement2 (h1 : perpendicular a α) (h2 : perpendicular b β) (h3 : parallel a b) : parallel_planes α β := sorry

-- Statement ③: If γ ⊥ α and γ ⊥ β, then α ∥ β
theorem statement3 (h1 : perpendicular γ α) (h2 : perpendicular γ β) : parallel_planes α β := sorry

-- Statement ④: If a ⊥ α and α ⊥ β, then a ∥ β
theorem statement4 (h1 : perpendicular a α) (h2 : parallel_planes α β) : parallel a b := sorry

-- The correct choice is A: Statements ① and ② are correct
theorem correctness_A : statement1_correct ∧ statement2_correct := sorry

end statement1_statement2_statement3_statement4_correctness_A_l1661_166194


namespace quadratic_equation_unique_solution_l1661_166198

theorem quadratic_equation_unique_solution (p : ℚ) :
  (∃ x : ℚ, 3 * x^2 - 7 * x + p = 0) ∧ 
  ∀ y : ℚ, 3 * y^2 -7 * y + p ≠ 0 → ∀ z : ℚ, 3 * z^2 - 7 * z + p = 0 → y = z ↔ 
  p = 49 / 12 :=
by
  sorry

end quadratic_equation_unique_solution_l1661_166198


namespace equation_satisfied_by_r_l1661_166152

theorem equation_satisfied_by_r {x y z r : ℝ} (h1: x ≠ y) (h2: y ≠ z) (h3: z ≠ x) 
    (h4: x ≠ 0) (h5: y ≠ 0) (h6: z ≠ 0) 
    (h7: ∃ (r: ℝ), x * (y - z) = (y * (z - x)) / r ∧ y * (z - x) = (z * (y - x)) / r ∧ z * (y - x) = (x * (y - z)) * r) 
    : r^2 - r + 1 = 0 := 
sorry

end equation_satisfied_by_r_l1661_166152


namespace eq_one_solution_in_interval_l1661_166166

theorem eq_one_solution_in_interval (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (2 * a * x^2 - x - 1 = 0) ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 1 ∧ y ≠ x → (2 * a * y^2 - y - 1 ≠ 0))) → (1 < a) :=
by
  sorry

end eq_one_solution_in_interval_l1661_166166


namespace range_of_a_l1661_166185

-- Define the conditions and what we want to prove
theorem range_of_a (a : ℝ) (x : ℝ) 
    (h1 : ∀ x, |x - 1| + |x + 1| ≥ 3 * a)
    (h2 : ∀ x, (2 * a - 1) ^ x ≤ 1 → (2 * a - 1) < 1 ∧ (2 * a - 1) > 0) :
    (1 / 2 < a ∧ a ≤ 2 / 3) :=
by
  sorry -- Here will be the proof

end range_of_a_l1661_166185


namespace seventh_degree_solution_l1661_166127

theorem seventh_degree_solution (a b x : ℝ) :
  (x^7 - 7 * a * x^5 + 14 * a^2 * x^3 - 7 * a^3 * x = b) ↔
  ∃ α β : ℝ, α + β = x ∧ α * β = a ∧ α^7 + β^7 = b :=
by
  sorry

end seventh_degree_solution_l1661_166127


namespace luke_points_per_round_l1661_166144

-- Definitions for conditions
def total_points : ℤ := 84
def rounds : ℤ := 2
def points_per_round (total_points rounds : ℤ) : ℤ := total_points / rounds

-- Statement of the problem
theorem luke_points_per_round : points_per_round total_points rounds = 42 := 
by 
  sorry

end luke_points_per_round_l1661_166144


namespace total_flowers_eaten_l1661_166161

theorem total_flowers_eaten :
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = 16.5 :=
by
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  sorry

end total_flowers_eaten_l1661_166161


namespace average_salary_l1661_166174

theorem average_salary (T_salary : ℕ) (R_salary : ℕ) (total_salary : ℕ) (T_count : ℕ) (R_count : ℕ) (total_count : ℕ) :
    T_salary = 12000 * T_count →
    R_salary = 6000 * R_count →
    total_salary = T_salary + R_salary →
    T_count = 6 →
    R_count = total_count - T_count →
    total_count = 18 →
    (total_salary / total_count) = 8000 :=
by
  intros
  sorry

end average_salary_l1661_166174


namespace kneading_time_is_correct_l1661_166120

def total_time := 280
def rising_time_per_session := 120
def number_of_rising_sessions := 2
def baking_time := 30

def total_rising_time := rising_time_per_session * number_of_rising_sessions
def total_non_kneading_time := total_rising_time + baking_time
def kneading_time := total_time - total_non_kneading_time

theorem kneading_time_is_correct : kneading_time = 10 := by
  have h1 : total_rising_time = 240 := by
    sorry
  have h2 : total_non_kneading_time = 270 := by
    sorry
  have h3 : kneading_time = 10 := by
    sorry
  exact h3

end kneading_time_is_correct_l1661_166120


namespace actual_travel_time_l1661_166188

noncomputable def distance : ℕ := 360
noncomputable def scheduled_time : ℕ := 9
noncomputable def speed_increase : ℕ := 5

theorem actual_travel_time (d : ℕ) (t_sched : ℕ) (Δv : ℕ) : 
  (d = distance) ∧ (t_sched = scheduled_time) ∧ (Δv = speed_increase) → 
  t_sched + Δv = 8 :=
by
  sorry

end actual_travel_time_l1661_166188


namespace find_BF_pqsum_l1661_166143

noncomputable def square_side_length : ℝ := 900
noncomputable def EF_length : ℝ := 400
noncomputable def m_angle_EOF : ℝ := 45
noncomputable def center_mid_to_side : ℝ := square_side_length / 2

theorem find_BF_pqsum :
  let G_mid : ℝ := center_mid_to_side
  let x : ℝ := G_mid - (2 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let y : ℝ := (1 / 3 * EF_length) -- Approximation, actual calculation involves solving quadratic 
  let BF := G_mid - y
  BF = 250 + 50 * Real.sqrt 7 ->
  250 + 50 + 7 = 307 := sorry

end find_BF_pqsum_l1661_166143


namespace min_supreme_supervisors_l1661_166153

-- Definitions
def num_employees : ℕ := 50000
def supervisors (e : ℕ) : ℕ := 7 - e

-- Theorem statement
theorem min_supreme_supervisors (k : ℕ) (num_employees_le_reached : ∀ n : ℕ, 50000 ≤ n) : 
  k ≥ 28 := 
sorry

end min_supreme_supervisors_l1661_166153


namespace exponents_to_99_l1661_166181

theorem exponents_to_99 :
  (1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99) :=
sorry

end exponents_to_99_l1661_166181


namespace ellipse_properties_l1661_166122

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2) / 4 + y^2 = 1

noncomputable def trajectory_equation_midpoint (x y : ℝ) : Prop :=
  ((2 * x - 1)^2) / 4 + (2 * y - 1 / 2)^2 = 1

theorem ellipse_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y) ∧
  (∀ x y : ℝ, trajectory_equation_midpoint x y) :=
by
  sorry

end ellipse_properties_l1661_166122


namespace smallest_n_contains_digit9_and_terminating_decimal_l1661_166111

-- Define the condition that a number contains the digit 9
def contains_digit_9 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

-- Define the condition that a number is of the form 2^a * 5^b
def is_form_of_2a_5b (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2 ^ a * 5 ^ b

-- Define the main theorem
theorem smallest_n_contains_digit9_and_terminating_decimal : 
  ∃ (n : ℕ), contains_digit_9 n ∧ is_form_of_2a_5b n ∧ (∀ m, (contains_digit_9 m ∧ is_form_of_2a_5b m) → n ≤ m) ∧ n = 12500 :=
  sorry

end smallest_n_contains_digit9_and_terminating_decimal_l1661_166111


namespace scooter_price_and_installment_l1661_166137

variable {P : ℝ} -- price of the scooter
variable {m : ℝ} -- monthly installment

theorem scooter_price_and_installment (h1 : 0.2 * P = 240) (h2 : (0.8 * P) = 12 * m) : 
  P = 1200 ∧ m = 80 := by
  sorry

end scooter_price_and_installment_l1661_166137


namespace people_visited_neither_l1661_166121

-- Definitions based on conditions
def total_people : ℕ := 60
def visited_iceland : ℕ := 35
def visited_norway : ℕ := 23
def visited_both : ℕ := 31

-- Theorem statement
theorem people_visited_neither :
  total_people - (visited_iceland + visited_norway - visited_both) = 33 :=
by sorry

end people_visited_neither_l1661_166121


namespace cistern_fill_time_l1661_166105

theorem cistern_fill_time (F E : ℝ) (hF : F = 1 / 7) (hE : E = 1 / 9) : (1 / (F - E)) = 31.5 :=
by
  sorry

end cistern_fill_time_l1661_166105


namespace express_scientific_notation_l1661_166150

theorem express_scientific_notation : (152300 : ℝ) = 1.523 * 10^5 := 
by
  sorry

end express_scientific_notation_l1661_166150


namespace find_x_y_sum_l1661_166160

theorem find_x_y_sum :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (∃ (a b : ℕ), 360 * x = a^2 ∧ 360 * y = b^4) ∧ x + y = 2260 :=
by {
  sorry
}

end find_x_y_sum_l1661_166160


namespace edge_of_new_cube_l1661_166110

theorem edge_of_new_cube (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) :
  ∃ d : ℝ, d^3 = a^3 + b^3 + c^3 ∧ d = 12 :=
by
  sorry

end edge_of_new_cube_l1661_166110


namespace males_watch_tvxy_l1661_166142

-- Defining the conditions
def total_watch := 160
def females_watch := 75
def males_dont_watch := 83
def total_dont_watch := 120

-- Proving that the number of males who watch TVXY equals 85
theorem males_watch_tvxy : (total_watch - females_watch) = 85 :=
by sorry

end males_watch_tvxy_l1661_166142


namespace find_unknown_number_l1661_166172

theorem find_unknown_number (x : ℕ) (h₁ : (20 + 40 + 60) / 3 = 5 + (10 + 50 + x) / 3) : x = 45 :=
by sorry

end find_unknown_number_l1661_166172


namespace smallest_special_integer_l1661_166139

noncomputable def is_special (N : ℕ) : Prop :=
  N > 1 ∧ 
  (N % 8 = 1) ∧ 
  (2 * 8 ^ Nat.log N (8) / 2 > N / 8 ^ Nat.log N (8)) ∧ 
  (N % 9 = 1) ∧ 
  (2 * 9 ^ Nat.log N (9) / 2 > N / 9 ^ Nat.log N (9))

theorem smallest_special_integer : ∃ (N : ℕ), is_special N ∧ N = 793 :=
by 
  use 793
  sorry

end smallest_special_integer_l1661_166139


namespace remainder_when_4_pow_2023_div_17_l1661_166196

theorem remainder_when_4_pow_2023_div_17 :
  ∀ (x : ℕ), (x = 4) → x^2 ≡ 16 [MOD 17] → x^2023 ≡ 13 [MOD 17] := by
  intros x hx h
  sorry

end remainder_when_4_pow_2023_div_17_l1661_166196


namespace value_of_a_12_l1661_166184

variable {a : ℕ → ℝ} (h1 : a 6 + a 10 = 20) (h2 : a 4 = 2)

theorem value_of_a_12 : a 12 = 18 :=
by
  sorry

end value_of_a_12_l1661_166184


namespace isabella_babysits_afternoons_per_week_l1661_166125

-- Defining the conditions of Isabella's babysitting job
def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 5
def days_per_week (weeks : ℕ) (total_earnings : ℕ) : ℕ := total_earnings / (weeks * (hourly_rate * hours_per_day))

-- Total earnings after 7 weeks
def total_earnings : ℕ := 1050
def weeks : ℕ := 7

-- State the theorem
theorem isabella_babysits_afternoons_per_week :
  days_per_week weeks total_earnings = 6 :=
by
  sorry

end isabella_babysits_afternoons_per_week_l1661_166125


namespace tank_fraction_l1661_166199

theorem tank_fraction (x : ℚ) : 
  let tank1_capacity := 7000
  let tank2_capacity := 5000
  let tank3_capacity := 3000
  let tank2_fraction := 4 / 5
  let tank3_fraction := 1 / 2
  let total_water := 10850
  tank1_capacity * x + tank2_capacity * tank2_fraction + tank3_capacity * tank3_fraction = total_water → 
  x = 107 / 140 := 
by {
  sorry
}

end tank_fraction_l1661_166199


namespace find_divisor_of_x_l1661_166109

theorem find_divisor_of_x (x : ℕ) (q p : ℕ) (h1 : x % n = 5) (h2 : 4 * x % n = 2) : n = 9 :=
by
  sorry

end find_divisor_of_x_l1661_166109


namespace find_possible_values_of_a_l1661_166171

theorem find_possible_values_of_a (a b c : ℝ) (h1 : a * b + a + b = c) (h2 : b * c + b + c = a) (h3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 :=
by
  sorry

end find_possible_values_of_a_l1661_166171


namespace find_initial_passengers_l1661_166195

def initial_passengers_found (P : ℕ) : Prop :=
  let after_first_station := (2 / 3 : ℚ) * P + 280
  let after_second_station := (1 / 2 : ℚ) * after_first_station + 12
  after_second_station = 242

theorem find_initial_passengers :
  ∃ P : ℕ, initial_passengers_found P ∧ P = 270 :=
by
  sorry

end find_initial_passengers_l1661_166195


namespace find_k_l1661_166118

noncomputable def sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  3 * 2^n + k

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a n * a (n + 2) = (a (n + 1))^2

theorem find_k
  (a : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a n = sequence_sum (n + 1) k - sequence_sum n k)
  (h2 : geometric_sequence a) :
  k = -3 :=
  by sorry

end find_k_l1661_166118


namespace solution_set_I_range_of_a_II_l1661_166141

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2*x - 1|

theorem solution_set_I (x : ℝ) (a : ℝ) (h : a = 2) :
  f x a ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem range_of_a_II (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end solution_set_I_range_of_a_II_l1661_166141


namespace lowest_test_score_dropped_is_35_l1661_166170

theorem lowest_test_score_dropped_is_35 
  (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : min A (min B (min C D)) = D)
  (h3 : (A + B + C) / 3 = 55) : 
  D = 35 := by
  sorry

end lowest_test_score_dropped_is_35_l1661_166170


namespace family_has_11_eggs_l1661_166136

def initialEggs : ℕ := 10
def eggsUsed : ℕ := 5
def chickens : ℕ := 2
def eggsPerChicken : ℕ := 3

theorem family_has_11_eggs :
  (initialEggs - eggsUsed) + (chickens * eggsPerChicken) = 11 := by
  sorry

end family_has_11_eggs_l1661_166136


namespace min_value_l1661_166106

open Real

noncomputable def y1 (x1 : ℝ) : ℝ := x1 * log x1
noncomputable def y2 (x2 : ℝ) : ℝ := x2 - 3

theorem min_value :
  ∃ (x1 x2 : ℝ), (x1 - x2)^2 + (y1 x1 - y2 x2)^2 = 2 :=
by
  sorry

end min_value_l1661_166106


namespace hyperbola_sufficient_asymptotes_l1661_166155

open Real

def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptotes_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

theorem hyperbola_sufficient_asymptotes (a b x y : ℝ) :
  (hyperbola_eq a b x y) → (asymptotes_eq a b x y) :=
by
  sorry

end hyperbola_sufficient_asymptotes_l1661_166155


namespace averagePrice_is_20_l1661_166182

-- Define the conditions
def books1 : Nat := 32
def cost1 : Nat := 1500

def books2 : Nat := 60
def cost2 : Nat := 340

-- Define the total books and total cost
def totalBooks : Nat := books1 + books2
def totalCost : Nat := cost1 + cost2

-- Define the average price calculation
def averagePrice : Nat := totalCost / totalBooks

-- The statement to prove
theorem averagePrice_is_20 : averagePrice = 20 := by
  -- Sorry is used here as a placeholder for the actual proof.
  sorry

end averagePrice_is_20_l1661_166182


namespace negation_ln_eq_x_minus_1_l1661_166183

theorem negation_ln_eq_x_minus_1 :
  ¬(∃ x : ℝ, 0 < x ∧ Real.log x = x - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by 
  sorry

end negation_ln_eq_x_minus_1_l1661_166183


namespace maximum_rectangle_area_l1661_166135

variable (x y : ℝ)

def area (x y : ℝ) : ℝ :=
  x * y

def similarity_condition (x y : ℝ) : Prop :=
  (11 - x) / (y - 6) = 2

theorem maximum_rectangle_area :
  ∃ (x y : ℝ), similarity_condition x y ∧ area x y = 66 :=  by
  sorry

end maximum_rectangle_area_l1661_166135


namespace sum_of_angles_l1661_166103

namespace BridgeProblem

def is_isosceles (A B C : Type) (AB AC : ℝ) : Prop := AB = AC

def angle_bac (A B C : Type) : ℝ := 15

def angle_edf (D E F : Type) : ℝ := 45

theorem sum_of_angles (A B C D E F : Type) 
  (h_isosceles_ABC : is_isosceles A B C 1 1)
  (h_isosceles_DEF : is_isosceles D E F 1 1)
  (h_angle_BAC : angle_bac A B C = 15)
  (h_angle_EDF : angle_edf D E F = 45) :
  true := 
by 
  sorry

end BridgeProblem

end sum_of_angles_l1661_166103


namespace rubber_bands_per_large_ball_l1661_166165

open Nat

theorem rubber_bands_per_large_ball :
  let total_rubber_bands := 5000
  let small_bands := 50
  let small_balls := 22
  let large_balls := 13
  let used_bands := small_balls * small_bands
  let remaining_bands := total_rubber_bands - used_bands
  let large_bands := remaining_bands / large_balls
  large_bands = 300 :=
by
  sorry

end rubber_bands_per_large_ball_l1661_166165


namespace largest_consecutive_odd_integer_sum_l1661_166149

theorem largest_consecutive_odd_integer_sum
  (x : Real)
  (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = -378.5) :
  x + 8 = -79.7 + 8 :=
by
  sorry

end largest_consecutive_odd_integer_sum_l1661_166149


namespace range_of_m_l1661_166123

theorem range_of_m (x m : ℝ) (h1 : -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2)
                   (h2 : x^2 - 2*x + 1 - m^2 ≤ 0)
                   (h3 : m > 0)
                   (h4 : (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
                   (h5 : ¬((x < 1 - m ∨ x > 1 + m) → (x < -2 ∨ x > 10))) :
                   m ≤ 3 :=
by
  sorry

end range_of_m_l1661_166123


namespace solve_equations_l1661_166128

theorem solve_equations :
  (∃ x : ℝ, (x + 2) ^ 3 + 1 = 0 ∧ x = -3) ∧
  (∃ x : ℝ, ((3 * x - 2) ^ 2 = 64 ∧ (x = 10/3 ∨ x = -2))) :=
by {
  -- Prove the existence of solutions for both problems
  sorry
}

end solve_equations_l1661_166128


namespace prob_less_than_9_l1661_166133

def prob_10 : ℝ := 0.24
def prob_9 : ℝ := 0.28
def prob_8 : ℝ := 0.19

theorem prob_less_than_9 : prob_10 + prob_9 + prob_8 < 1 → 1 - prob_10 - prob_9 = 0.48 := 
by {
  sorry
}

end prob_less_than_9_l1661_166133


namespace correct_total_cost_correct_remaining_donuts_l1661_166119

-- Conditions
def budget : ℝ := 50
def cost_per_box : ℝ := 12
def discount_percentage : ℝ := 0.10
def number_of_boxes_bought : ℕ := 4
def donuts_per_box : ℕ := 12
def boxes_given_away : ℕ := 1
def additional_donuts_given_away : ℕ := 6

-- Calculations based on conditions
def total_cost_before_discount : ℝ := number_of_boxes_bought * cost_per_box
def discount_amount : ℝ := discount_percentage * total_cost_before_discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

def total_donuts : ℕ := number_of_boxes_bought * donuts_per_box
def total_donuts_given_away : ℕ := (boxes_given_away * donuts_per_box) + additional_donuts_given_away
def remaining_donuts : ℕ := total_donuts - total_donuts_given_away

-- Theorems to prove
theorem correct_total_cost : total_cost_after_discount = 43.20 := by
  -- proof here
  sorry

theorem correct_remaining_donuts : remaining_donuts = 30 := by
  -- proof here
  sorry

end correct_total_cost_correct_remaining_donuts_l1661_166119


namespace total_sample_size_l1661_166164

theorem total_sample_size
    (undergrad_count : ℕ) (masters_count : ℕ) (doctoral_count : ℕ)
    (total_students : ℕ) (sample_size_doctoral : ℕ) (proportion_sample : ℕ)
    (n : ℕ)
    (H1 : undergrad_count = 12000)
    (H2 : masters_count = 1000)
    (H3 : doctoral_count = 200)
    (H4 : total_students = undergrad_count + masters_count + doctoral_count)
    (H5 : sample_size_doctoral = 20)
    (H6 : proportion_sample = sample_size_doctoral / doctoral_count)
    (H7 : n = proportion_sample * total_students) :
  n = 1320 := 
sorry

end total_sample_size_l1661_166164


namespace xy_squared_l1661_166115

theorem xy_squared (x y : ℚ) (h1 : x + y = 9 / 20) (h2 : x - y = 1 / 20) :
  x^2 - y^2 = 9 / 400 :=
by
  sorry

end xy_squared_l1661_166115


namespace Shekar_science_marks_l1661_166151

theorem Shekar_science_marks (S : ℕ) : 
  let math_marks := 76
  let social_studies_marks := 82
  let english_marks := 67
  let biology_marks := 75
  let average_marks := 73
  let num_subjects := 5
  ((math_marks + S + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks) → S = 65 :=
by
  sorry

end Shekar_science_marks_l1661_166151


namespace valid_three_digit_numbers_count_l1661_166134

noncomputable def count_valid_numbers : ℕ :=
  let valid_first_digits := [2, 4, 6, 8].length
  let valid_other_digits := [0, 2, 4, 6, 8].length
  let total_even_digit_3_digit_numbers := valid_first_digits * valid_other_digits * valid_other_digits
  let no_4_or_8_first_digits := [2, 6].length
  let no_4_or_8_other_digits := [0, 2, 6].length
  let numbers_without_4_or_8 := no_4_or_8_first_digits * no_4_or_8_other_digits * no_4_or_8_other_digits
  let numbers_with_4_or_8 := total_even_digit_3_digit_numbers - numbers_without_4_or_8
  let valid_even_sum_count := 50  -- Assumed from the manual checking
  valid_even_sum_count

theorem valid_three_digit_numbers_count :
  count_valid_numbers = 50 :=
by
  sorry

end valid_three_digit_numbers_count_l1661_166134


namespace gcd_324_243_135_l1661_166180

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_324_243_135_l1661_166180


namespace point_in_third_quadrant_l1661_166157

theorem point_in_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : -m^2 < 0 ∧ -n < 0 :=
by
  sorry

end point_in_third_quadrant_l1661_166157


namespace number_of_companies_l1661_166162

theorem number_of_companies (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by
  sorry

end number_of_companies_l1661_166162


namespace big_container_capacity_l1661_166187

theorem big_container_capacity (C : ℝ)
    (h1 : 0.75 * C - 0.40 * C = 14) : C = 40 :=
  sorry

end big_container_capacity_l1661_166187


namespace infinitely_many_m_l1661_166117

theorem infinitely_many_m (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ m in Filter.atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end infinitely_many_m_l1661_166117


namespace subtraction_of_decimals_l1661_166167

theorem subtraction_of_decimals : 7.42 - 2.09 = 5.33 := 
by
  sorry

end subtraction_of_decimals_l1661_166167


namespace firefighters_time_to_extinguish_fire_l1661_166186

theorem firefighters_time_to_extinguish_fire (gallons_per_minute_per_hose : ℕ) (total_gallons : ℕ) (number_of_firefighters : ℕ)
  (H1 : gallons_per_minute_per_hose = 20)
  (H2 : total_gallons = 4000)
  (H3 : number_of_firefighters = 5): 
  (total_gallons / (gallons_per_minute_per_hose * number_of_firefighters)) = 40 := 
by 
  sorry

end firefighters_time_to_extinguish_fire_l1661_166186


namespace circle_properties_l1661_166179

noncomputable def circle_eq (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0
noncomputable def line_eq (x y : ℝ) := x + 2*y - 4 = 0
noncomputable def perpendicular (x1 y1 x2 y2 : ℝ) := 
  (x1 * x2 + y1 * y2 = 0)

theorem circle_properties (m : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, circle_eq x y m) →
  (∀ x, line_eq x (y1 + y2)) →
  perpendicular (4 - 2*y1) y1 (4 - 2*y2) y2 →
  m = 8 / 5 ∧ 
  (∀ x y, (x^2 + y^2 - (8 / 5) * x - (16 / 5) * y = 0) ↔ 
           (x - (4 - 2*(16/5))) * (x - (4 - 2*(16/5))) + (y - (16/5)) * (y - (16/5)) = 5 - (8/5)) :=
sorry

end circle_properties_l1661_166179


namespace enchilada_taco_cost_l1661_166173

theorem enchilada_taco_cost (e t : ℝ) 
  (h1 : 3 * e + 4 * t = 3.50) 
  (h2 : 4 * e + 3 * t = 3.90) : 
  4 * e + 5 * t = 4.56 := 
sorry

end enchilada_taco_cost_l1661_166173


namespace length_of_integer_eq_24_l1661_166158

theorem length_of_integer_eq_24 (k : ℕ) (h1 : k > 1) (h2 : ∃ (p1 p2 p3 p4 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ k = p1 * p2 * p3 * p4) : k = 24 := by
  sorry

end length_of_integer_eq_24_l1661_166158


namespace Mrs_Fredricksons_chickens_l1661_166197

theorem Mrs_Fredricksons_chickens (C : ℕ) (h1 : 1/4 * C + 1/4 * (3/4 * C) = 35) : C = 80 :=
by
  sorry

end Mrs_Fredricksons_chickens_l1661_166197


namespace veridux_male_associates_l1661_166189

theorem veridux_male_associates (total_employees female_employees total_managers female_managers : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : female_managers = 40) :
  total_employees - female_employees = 160 :=
by
  sorry

end veridux_male_associates_l1661_166189


namespace total_wheels_l1661_166145

def num_wheels_in_garage : Nat :=
  let cars := 2 * 4
  let lawnmower := 4
  let bicycles := 3 * 2
  let tricycle := 3
  let unicycle := 1
  let skateboard := 4
  let wheelbarrow := 1
  let wagon := 4
  let dolly := 2
  let shopping_cart := 4
  let scooter := 2
  cars + lawnmower + bicycles + tricycle + unicycle + skateboard + wheelbarrow + wagon + dolly + shopping_cart + scooter

theorem total_wheels : num_wheels_in_garage = 39 := by
  sorry

end total_wheels_l1661_166145


namespace remainder_of_x_plus_3uy_l1661_166159

-- Given conditions
variables (x y u v : ℕ)
variable (Hdiv : x = u * y + v)
variable (H0_le_v : 0 ≤ v)
variable (Hv_lt_y : v < y)

-- Statement to prove
theorem remainder_of_x_plus_3uy (x y u v : ℕ) (Hdiv : x = u * y + v) (H0_le_v : 0 ≤ v) (Hv_lt_y : v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end remainder_of_x_plus_3uy_l1661_166159


namespace collinear_vectors_l1661_166108

theorem collinear_vectors (m : ℝ) (h_collinear : 1 * m - (-2) * (-3) = 0) : m = 6 :=
by
  sorry

end collinear_vectors_l1661_166108


namespace base4_base7_digit_difference_l1661_166130

def num_digits_base (n b : ℕ) : ℕ :=
  if b > 1 then Nat.log b n + 1 else 0

theorem base4_base7_digit_difference :
  let n := 1573
  num_digits_base n 4 - num_digits_base n 7 = 2 := by
  sorry

end base4_base7_digit_difference_l1661_166130


namespace number_of_paths_3x3_l1661_166140

-- Definition of the problem conditions
def grid_moves (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- Lean statement for the proof problem
theorem number_of_paths_3x3 : grid_moves 3 3 = 20 := by
  sorry

end number_of_paths_3x3_l1661_166140
