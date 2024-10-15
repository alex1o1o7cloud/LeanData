import Mathlib

namespace NUMINAMATH_GPT_prob_no_infection_correct_prob_one_infection_correct_l384_38480

-- Probability that no chicken is infected
def prob_no_infection (p_not_infected : ℚ) (n : ℕ) : ℚ := p_not_infected^n

-- Given
def p_not_infected : ℚ := 4 / 5
def n : ℕ := 5

-- Expected answer for no chicken infected
def expected_prob_no_infection : ℚ := 1024 / 3125

-- Lean statement
theorem prob_no_infection_correct : 
  prob_no_infection p_not_infected n = expected_prob_no_infection := by
  sorry

-- Probability that exactly one chicken is infected
def prob_one_infection (p_infected : ℚ) (p_not_infected : ℚ) (n : ℕ) : ℚ := 
  (n * p_not_infected^(n-1) * p_infected)

-- Given
def p_infected : ℚ := 1 / 5

-- Expected answer for exactly one chicken infected
def expected_prob_one_infection : ℚ := 256 / 625

-- Lean statement
theorem prob_one_infection_correct : 
  prob_one_infection p_infected p_not_infected n = expected_prob_one_infection := by
  sorry

end NUMINAMATH_GPT_prob_no_infection_correct_prob_one_infection_correct_l384_38480


namespace NUMINAMATH_GPT_average_goals_l384_38472

def num_goals_3 := 3
def num_players_3 := 2
def num_goals_4 := 4
def num_players_4 := 3
def num_goals_5 := 5
def num_players_5 := 1
def num_goals_6 := 6
def num_players_6 := 1

def total_goals := (num_goals_3 * num_players_3) + (num_goals_4 * num_players_4) + (num_goals_5 * num_players_5) + (num_goals_6 * num_players_6)
def total_players := num_players_3 + num_players_4 + num_players_5 + num_players_6

theorem average_goals :
  (total_goals / total_players : ℚ) = 29 / 7 :=
sorry

end NUMINAMATH_GPT_average_goals_l384_38472


namespace NUMINAMATH_GPT_length_of_bridge_is_255_l384_38425

noncomputable def bridge_length (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ) : ℕ :=
  let train_speed_mps := train_speed_kph * 1000 / (60 * 60)
  let total_distance := train_speed_mps * cross_time_sec
  total_distance - train_length

theorem length_of_bridge_is_255 :
  ∀ (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ), 
    train_length = 120 →
    train_speed_kph = 45 →
    cross_time_sec = 30 →
    bridge_length train_length train_speed_kph cross_time_sec = 255 :=
by
  intros train_length train_speed_kph cross_time_sec htl htsk hcts
  simp [bridge_length]
  rw [htl, htsk, hcts]
  norm_num
  sorry

end NUMINAMATH_GPT_length_of_bridge_is_255_l384_38425


namespace NUMINAMATH_GPT_ten_integers_disjoint_subsets_same_sum_l384_38492

theorem ten_integers_disjoint_subsets_same_sum (S : Finset ℕ) (h : S.card = 10) (h_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ≠ B ∧ A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end NUMINAMATH_GPT_ten_integers_disjoint_subsets_same_sum_l384_38492


namespace NUMINAMATH_GPT_steve_average_speed_l384_38477

/-
Problem Statement:
Prove that the average speed of Steve's travel for the entire journey is 55 mph given the following conditions:
1. Steve's first part of journey: 5 hours at 40 mph.
2. Steve's second part of journey: 3 hours at 80 mph.
-/

theorem steve_average_speed :
  let time1 := 5 -- hours
  let speed1 := 40 -- mph
  let time2 := 3 -- hours
  let speed2 := 80 -- mph
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 55 := by
  sorry

end NUMINAMATH_GPT_steve_average_speed_l384_38477


namespace NUMINAMATH_GPT_remainder_when_3n_plus_2_squared_divided_by_11_l384_38453

theorem remainder_when_3n_plus_2_squared_divided_by_11 (n : ℕ) (h : n % 7 = 5) : ((3 * n + 2)^2) % 11 = 3 :=
  sorry

end NUMINAMATH_GPT_remainder_when_3n_plus_2_squared_divided_by_11_l384_38453


namespace NUMINAMATH_GPT_boss_salary_percentage_increase_l384_38495

theorem boss_salary_percentage_increase (W B : ℝ) (h : W = 0.2 * B) : ((B / W - 1) * 100) = 400 := by
sorry

end NUMINAMATH_GPT_boss_salary_percentage_increase_l384_38495


namespace NUMINAMATH_GPT_find_side_b_l384_38438

variable {a b c : ℝ} -- sides of the triangle
variable {A B C : ℝ} -- angles of the triangle
variable {area : ℝ}

axiom sides_form_arithmetic_sequence : 2 * b = a + c
axiom angle_B_is_60_degrees : B = Real.pi / 3
axiom area_is_3sqrt3 : area = 3 * Real.sqrt 3
axiom area_formula : area = 1 / 2 * a * c * Real.sin (B)

theorem find_side_b : b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_side_b_l384_38438


namespace NUMINAMATH_GPT_smallest_possible_n_l384_38424

theorem smallest_possible_n
  (n : ℕ)
  (d : ℕ)
  (h_d_pos : d > 0)
  (h_profit : 10 * n - 30 = 100)
  (h_cost_multiple : ∃ k, d = 2 * n * k) :
  n = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_possible_n_l384_38424


namespace NUMINAMATH_GPT_odd_n_cube_plus_one_not_square_l384_38496

theorem odd_n_cube_plus_one_not_square (n : ℤ) (h : n % 2 = 1) : ¬ ∃ (x : ℤ), x^2 = n^3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_odd_n_cube_plus_one_not_square_l384_38496


namespace NUMINAMATH_GPT_initial_candies_l384_38483

-- Define initial variables and conditions
variable (x : ℕ)
variable (remaining_candies_after_first_day : ℕ)
variable (remaining_candies_after_second_day : ℕ)

-- Conditions as per given problem
def condition1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
def condition2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
def final_condition : remaining_candies_after_second_day = 10 := sorry

-- Goal: Prove that initially, Liam had 52 candies
theorem initial_candies : x = 52 := by
  have h1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
  have h2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
  have h3 : remaining_candies_after_second_day = 10 := sorry
    
  -- Combine conditions to solve for x
  sorry

end NUMINAMATH_GPT_initial_candies_l384_38483


namespace NUMINAMATH_GPT_members_do_not_play_either_l384_38459

noncomputable def total_members := 30
noncomputable def badminton_players := 16
noncomputable def tennis_players := 19
noncomputable def both_players := 7

theorem members_do_not_play_either : 
  (total_members - (badminton_players + tennis_players - both_players)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_members_do_not_play_either_l384_38459


namespace NUMINAMATH_GPT_fixed_point_difference_l384_38423

noncomputable def func (a x : ℝ) : ℝ := a^x + Real.log a

theorem fixed_point_difference (a m n : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (func a 0 = n) ∧ (y = func a x → (x = m) ∧ (y = n)) → (m - n = -2) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_fixed_point_difference_l384_38423


namespace NUMINAMATH_GPT_sign_up_ways_l384_38467

theorem sign_up_ways : (3 ^ 4) = 81 :=
by
  sorry

end NUMINAMATH_GPT_sign_up_ways_l384_38467


namespace NUMINAMATH_GPT_calculate_total_interest_l384_38409

theorem calculate_total_interest :
  let total_money := 9000
  let invested_at_8_percent := 4000
  let invested_at_9_percent := total_money - invested_at_8_percent
  let interest_rate_8 := 0.08
  let interest_rate_9 := 0.09
  let interest_from_8_percent := invested_at_8_percent * interest_rate_8
  let interest_from_9_percent := invested_at_9_percent * interest_rate_9
  let total_interest := interest_from_8_percent + interest_from_9_percent
  total_interest = 770 :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_interest_l384_38409


namespace NUMINAMATH_GPT_nth_equation_l384_38401

open Nat

theorem nth_equation (n : ℕ) (hn : 0 < n) :
  (n + 1)/((n + 1) * (n + 1) - 1) - (1/(n * (n + 1) * (n + 2))) = 1/(n + 1) := 
by
  sorry

end NUMINAMATH_GPT_nth_equation_l384_38401


namespace NUMINAMATH_GPT_at_most_n_zeros_l384_38431

-- Definitions of conditions
variables {α : Type*} [Inhabited α]

/-- Define the structure of the sheet of numbers with the given properties -/
structure sheet :=
(n : ℕ)
(val : ℕ → ℤ)

-- Assuming infinite sheet and the properties
variable (s : sheet)

-- Predicate for a row having only positive integers
def all_positive (r : ℕ → ℤ) : Prop := ∀ i, r i > 0

-- Define the initial row R which has all positive integers
variable {R : ℕ → ℤ}

-- Statement that each element in the row below is sum of element above and to the left
def below_sum (r R : ℕ → ℤ) (n : ℕ) : Prop := ∀ i, r i = R i + (if i = 0 then 0 else R (i - 1))

-- Variable for the row n below R
variable {Rn : ℕ → ℤ}

-- Main theorem statement
theorem at_most_n_zeros (n : ℕ) (hr : all_positive R) (hs : below_sum R Rn n) : 
  ∃ k ≤ n, Rn k = 0 ∨ Rn k > 0 := sorry

end NUMINAMATH_GPT_at_most_n_zeros_l384_38431


namespace NUMINAMATH_GPT_joann_lollipop_wednesday_l384_38494

variable (a : ℕ) (d : ℕ) (n : ℕ)

def joann_lollipop_count (a d n : ℕ) : ℕ :=
  a + d * n

theorem joann_lollipop_wednesday :
  let a := 4
  let d := 3
  let total_days := 7
  let target_total := 133
  ∀ (monday tuesday wednesday thursday friday saturday sunday : ℕ),
    monday = a ∧
    tuesday = a + d ∧
    wednesday = a + 2 * d ∧
    thursday = a + 3 * d ∧
    friday = a + 4 * d ∧
    saturday = a + 5 * d ∧
    sunday = a + 6 * d ∧
    (monday + tuesday + wednesday + thursday + friday + saturday + sunday = target_total) →
    wednesday = 10 :=
by
  sorry

end NUMINAMATH_GPT_joann_lollipop_wednesday_l384_38494


namespace NUMINAMATH_GPT_integer_solutions_for_exponential_equation_l384_38445

theorem integer_solutions_for_exponential_equation :
  ∃ (a b c : ℕ), 
  2 ^ a * 3 ^ b + 9 = c ^ 2 ∧ 
  (a = 4 ∧ b = 0 ∧ c = 5) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 3 ∧ c = 21) ∨ 
  (a = 3 ∧ b = 3 ∧ c = 15) ∨ 
  (a = 4 ∧ b = 5 ∧ c = 51) :=
by {
  -- This is where the proof would go.
  sorry
}

end NUMINAMATH_GPT_integer_solutions_for_exponential_equation_l384_38445


namespace NUMINAMATH_GPT_smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l384_38449

theorem smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450 :
  ∃ n : ℕ, (n - 10) % 12 = 0 ∧
           (n - 10) % 16 = 0 ∧
           (n - 10) % 18 = 0 ∧
           (n - 10) % 21 = 0 ∧
           (n - 10) % 28 = 0 ∧
           (n - 10) % 35 = 0 ∧
           (n - 10) % 40 = 0 ∧
           (n - 10) % 45 = 0 ∧
           (n - 10) % 55 = 0 ∧
           n = 55450 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l384_38449


namespace NUMINAMATH_GPT_oil_ratio_l384_38464

theorem oil_ratio (x : ℝ) (initial_small_tank : ℝ) (initial_large_tank : ℝ) (total_capacity_large : ℝ)
  (half_capacity_large : ℝ) (additional_needed : ℝ) :
  initial_small_tank = 4000 ∧ initial_large_tank = 3000 ∧ total_capacity_large = 20000 ∧
  half_capacity_large = total_capacity_large / 2 ∧ additional_needed = 4000 ∧
  (initial_large_tank + x + additional_needed = half_capacity_large) →
  x / initial_small_tank = 3 / 4 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end NUMINAMATH_GPT_oil_ratio_l384_38464


namespace NUMINAMATH_GPT_trig_identity_l384_38429

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem trig_identity (θ : ℝ) : sin (θ + 75 * Real.pi / 180) + cos (θ + 45 * Real.pi / 180) - Real.sqrt 3 * cos (θ + 15 * Real.pi / 180) = 0 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l384_38429


namespace NUMINAMATH_GPT_cyclic_inequality_l384_38460

theorem cyclic_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := 
by
  sorry

end NUMINAMATH_GPT_cyclic_inequality_l384_38460


namespace NUMINAMATH_GPT_advantageous_bank_l384_38455

variable (C : ℝ) (p n : ℝ)

noncomputable def semiAnnualCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (2 * 100)) ^ n

noncomputable def monthlyCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (12 * 100)) ^ (6 * n)

theorem advantageous_bank (p n : ℝ) :
  monthlyCompounding p n - semiAnnualCompounding p n > 0 := sorry

#check advantageous_bank

end NUMINAMATH_GPT_advantageous_bank_l384_38455


namespace NUMINAMATH_GPT_frustum_smaller_cone_height_l384_38463

theorem frustum_smaller_cone_height (H frustum_height radius1 radius2 : ℝ) 
  (h : ℝ) (h_eq : h = 30 - 18) : 
  radius1 = 6 → radius2 = 10 → frustum_height = 18 → H = 30 → h = 12 := 
by
  intros
  sorry

end NUMINAMATH_GPT_frustum_smaller_cone_height_l384_38463


namespace NUMINAMATH_GPT_certain_number_l384_38471

theorem certain_number (x : ℝ) : 
  0.55 * x = (4/5 : ℝ) * 25 + 2 → 
  x = 40 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l384_38471


namespace NUMINAMATH_GPT_seventh_root_binomial_expansion_l384_38402

theorem seventh_root_binomial_expansion : 
  (∃ (n : ℕ), n = 137858491849 ∧ (∃ (k : ℕ), n = (10 + 1) ^ k)) →
  (∃ a, a = 11 ∧ 11 ^ 7 = 137858491849) := 
by {
  sorry 
}

end NUMINAMATH_GPT_seventh_root_binomial_expansion_l384_38402


namespace NUMINAMATH_GPT_hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l384_38419

noncomputable def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

noncomputable def sum_first_n_odd_positive_integers (n : ℕ) : ℕ :=
  n * n

theorem hundredth_odd_integer_is_199 : nth_odd_positive_integer 100 = 199 :=
  by
  sorry

theorem sum_of_first_100_odd_integers_is_10000 : sum_first_n_odd_positive_integers 100 = 10000 :=
  by
  sorry

end NUMINAMATH_GPT_hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l384_38419


namespace NUMINAMATH_GPT_average_probable_weight_l384_38488

theorem average_probable_weight (weight : ℝ) (h1 : 61 < weight) (h2 : weight ≤ 64) : 
  (61 + 64) / 2 = 62.5 := 
by
  sorry

end NUMINAMATH_GPT_average_probable_weight_l384_38488


namespace NUMINAMATH_GPT_person_B_work_days_l384_38416

-- Let a be the work rate for person A, and b be the work rate for person B.
-- a completes the work in 20 days
-- b completes the work in x days
-- When working together, a and b complete 0.375 of the work in 5 days


theorem person_B_work_days (x : ℝ) :
  ((5 : ℝ) * ((1 / 20) + 1 / x) = 0.375) -> x = 40 := 
by 
  sorry

end NUMINAMATH_GPT_person_B_work_days_l384_38416


namespace NUMINAMATH_GPT_sail_time_difference_l384_38478

theorem sail_time_difference (distance : ℕ) (v_big : ℕ) (v_small : ℕ) (t_big t_small : ℕ)
  (h_distance : distance = 200)
  (h_v_big : v_big = 50)
  (h_v_small : v_small = 20)
  (h_t_big : t_big = distance / v_big)
  (h_t_small : t_small = distance / v_small)
  : t_small - t_big = 6 := by
  sorry

end NUMINAMATH_GPT_sail_time_difference_l384_38478


namespace NUMINAMATH_GPT_largest_n_divisible_l384_38420

theorem largest_n_divisible (n : ℕ) (h : (n : ℤ) > 0) : 
  (n^3 + 105) % (n + 12) = 0 ↔ n = 93 :=
sorry

end NUMINAMATH_GPT_largest_n_divisible_l384_38420


namespace NUMINAMATH_GPT_metal_waste_l384_38410

theorem metal_waste (l b : ℝ) (h : l > b) : l * b - (b^2 / 2) = 
  (l * b - (π * (b / 2)^2)) + (π * (b / 2)^2 - (b^2 / 2)) := by
  sorry

end NUMINAMATH_GPT_metal_waste_l384_38410


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l384_38493

theorem line_passes_through_fixed_point 
  (m : ℝ) : ∃ x y : ℝ, y = m * x + (2 * m + 1) ∧ (x, y) = (-2, 1) :=
by
  use (-2), (1)
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l384_38493


namespace NUMINAMATH_GPT_set_difference_lt3_gt0_1_leq_x_leq_2_l384_38434

def A := {x : ℝ | |x| < 3}
def B := {x : ℝ | x^2 - 3 * x + 2 > 0}

theorem set_difference_lt3_gt0_1_leq_x_leq_2 : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_GPT_set_difference_lt3_gt0_1_leq_x_leq_2_l384_38434


namespace NUMINAMATH_GPT_halfway_fraction_eq_l384_38461

-- Define the fractions
def one_seventh := 1 / 7
def one_fourth := 1 / 4

-- Define the common denominators
def common_denom_1 := 4 / 28
def common_denom_2 := 7 / 28

-- Define the addition of the common denominators
def addition := common_denom_1 + common_denom_2

-- Define the average of the fractions
noncomputable def average := addition / 2

-- State the theorem
theorem halfway_fraction_eq : average = 11 / 56 :=
by
  -- Provide the steps which will be skipped here
  sorry

end NUMINAMATH_GPT_halfway_fraction_eq_l384_38461


namespace NUMINAMATH_GPT_tip_percentage_l384_38447

theorem tip_percentage (T : ℝ) 
  (total_cost meal_cost sales_tax : ℝ)
  (h1 : meal_cost = 61.48)
  (h2 : sales_tax = 0.07 * meal_cost)
  (h3 : total_cost = meal_cost + sales_tax + T * meal_cost)
  (h4 : total_cost ≤ 75) :
  T ≤ 0.1499 :=
by
  -- main proof goes here
  sorry

end NUMINAMATH_GPT_tip_percentage_l384_38447


namespace NUMINAMATH_GPT_pure_imaginary_z1_over_z2_l384_38474

theorem pure_imaginary_z1_over_z2 (b : Real) : 
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  (Complex.re ((z1 / z2) : Complex)) = 0 → b = -3 / 2 :=
by
  intros
  -- Conditions
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  -- Assuming that the real part of (z1 / z2) is zero
  have h : Complex.re (z1 / z2) = 0 := ‹_›
  -- Require to prove that b = -3 / 2
  sorry

end NUMINAMATH_GPT_pure_imaginary_z1_over_z2_l384_38474


namespace NUMINAMATH_GPT_f_value_at_2_9_l384_38439

-- Define the function f with its properties as conditions
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the domain of f
axiom f_domain : ∀ x, 0 ≤ x ∧ x ≤ 1

-- Condition (i)
axiom f_0_eq : f 0 = 0

-- Condition (ii)
axiom f_monotone : ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- Condition (iii)
axiom f_symmetry : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (1 - x) = 3/4 - f x / 2

-- Condition (iv)
axiom f_scale : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x / 3) = f x / 3

-- Proof goal
theorem f_value_at_2_9 : f (2/9) = 5/24 := by
  sorry

end NUMINAMATH_GPT_f_value_at_2_9_l384_38439


namespace NUMINAMATH_GPT_daps_to_dips_l384_38426

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end NUMINAMATH_GPT_daps_to_dips_l384_38426


namespace NUMINAMATH_GPT_students_surveyed_l384_38484

theorem students_surveyed (S : ℕ)
  (h1 : (2/3 : ℝ) * 6 + (1/3 : ℝ) * 4 = 16/3)
  (h2 : S * (16/3 : ℝ) = 320) :
  S = 60 :=
sorry

end NUMINAMATH_GPT_students_surveyed_l384_38484


namespace NUMINAMATH_GPT_curve_statements_incorrect_l384_38448

theorem curve_statements_incorrect (t : ℝ) :
  (1 < t ∧ t < 3 → ¬ ∀ x y : ℝ, (x^2 / (3 - t) + y^2 / (t - 1) = 1 → x^2 + y^2 ≠ 1)) ∧
  ((3 - t) * (t - 1) < 0 → ¬ t < 1) :=
by
  sorry

end NUMINAMATH_GPT_curve_statements_incorrect_l384_38448


namespace NUMINAMATH_GPT_book_prices_purchasing_plans_l384_38468

theorem book_prices (x y : ℕ) (h1 : 20 * x + 40 * y = 1600) (h2 : 20 * x = 30 * y + 200) : x = 40 ∧ y = 20 :=
by
  sorry

theorem purchasing_plans (m : ℕ) (h3 : 2 * m + 20 ≥ 70) (h4 : 40 * m + 20 * (m + 20) ≤ 2000) :
  (m = 25 ∧ m + 20 = 45) ∨ (m = 26 ∧ m + 20 = 46) :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_book_prices_purchasing_plans_l384_38468


namespace NUMINAMATH_GPT_painted_cubes_even_faces_l384_38430

theorem painted_cubes_even_faces :
  let L := 6 -- length of the block
  let W := 2 -- width of the block
  let H := 2 -- height of the block
  let total_cubes := 24 -- the block is cut into 24 1-inch cubes
  let cubes_even_faces := 12 -- the number of 1-inch cubes with even number of blue faces
  -- each cube has a total of 6 faces,
  -- we need to count how many cubes have an even number of painted faces.
  L * W * H = total_cubes → 
  cubes_even_faces = 12 := sorry

end NUMINAMATH_GPT_painted_cubes_even_faces_l384_38430


namespace NUMINAMATH_GPT_price_for_3years_service_l384_38443

def full_price : ℝ := 85
def discount_price_1year (price : ℝ) : ℝ := price - (0.20 * price)
def discount_price_3years (price : ℝ) : ℝ := price - (0.25 * price)

theorem price_for_3years_service : discount_price_3years (discount_price_1year full_price) = 51 := 
by 
  sorry

end NUMINAMATH_GPT_price_for_3years_service_l384_38443


namespace NUMINAMATH_GPT_number_of_cats_l384_38427

variable (C D : ℕ)

-- Conditions
def condition1 : Prop := C = 15 * D / 7
def condition2 : Prop := C = 15 * (D + 12) / 11

-- Proof problem
theorem number_of_cats (h1 : condition1 C D) (h2 : condition2 C D) : C = 45 := sorry

end NUMINAMATH_GPT_number_of_cats_l384_38427


namespace NUMINAMATH_GPT_solve_equations_l384_38479

theorem solve_equations :
  (∀ x : ℝ, x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2) ∧ 
  (∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 ↔ x = 1/2) :=
by sorry

end NUMINAMATH_GPT_solve_equations_l384_38479


namespace NUMINAMATH_GPT_three_digit_powers_of_two_l384_38465

theorem three_digit_powers_of_two : 
  ∃ (N : ℕ), N = 3 ∧ ∀ (n : ℕ), (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_powers_of_two_l384_38465


namespace NUMINAMATH_GPT_height_of_model_l384_38422

noncomputable def original_monument_height : ℝ := 100
noncomputable def original_monument_radius : ℝ := 20
noncomputable def original_monument_volume : ℝ := 125600
noncomputable def model_volume : ℝ := 1.256

theorem height_of_model : original_monument_height / (original_monument_volume / model_volume)^(1/3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_height_of_model_l384_38422


namespace NUMINAMATH_GPT_mars_moon_cost_share_l384_38497

theorem mars_moon_cost_share :
  let total_cost := 40 * 10^9 -- total cost in dollars
  let num_people := 200 * 10^6 -- number of people sharing the cost
  (total_cost / num_people) = 200 := by
  sorry

end NUMINAMATH_GPT_mars_moon_cost_share_l384_38497


namespace NUMINAMATH_GPT_unique_solution_xp_eq_1_l384_38400

theorem unique_solution_xp_eq_1 (x p q : ℕ) (h1 : x ≥ 2) (h2 : p ≥ 2) (h3 : q ≥ 2):
  ((x + 1)^p - x^q = 1) ↔ (x = 2 ∧ p = 2 ∧ q = 3) :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_xp_eq_1_l384_38400


namespace NUMINAMATH_GPT_min_crossing_time_proof_l384_38411

def min_crossing_time (times : List ℕ) : ℕ :=
  -- Function to compute the minimum crossing time. Note: Actual implementation skipped.
sorry

theorem min_crossing_time_proof
  (times : List ℕ)
  (h_times : times = [2, 4, 8, 16]) :
  min_crossing_time times = 30 :=
sorry

end NUMINAMATH_GPT_min_crossing_time_proof_l384_38411


namespace NUMINAMATH_GPT_kitchen_supplies_sharon_wants_l384_38451

theorem kitchen_supplies_sharon_wants (P : ℕ) (plates_angela cutlery_angela pots_sharon plates_sharon cutlery_sharon : ℕ) 
  (h1 : plates_angela = 3 * P + 6) 
  (h2 : cutlery_angela = (3 * P + 6) / 2) 
  (h3 : pots_sharon = P / 2) 
  (h4 : plates_sharon = 3 * (3 * P + 6) - 20) 
  (h5 : cutlery_sharon = 2 * (3 * P + 6) / 2) 
  (h_total : pots_sharon + plates_sharon + cutlery_sharon = 254) : 
  P = 20 :=
sorry

end NUMINAMATH_GPT_kitchen_supplies_sharon_wants_l384_38451


namespace NUMINAMATH_GPT_scrabble_champions_l384_38458

theorem scrabble_champions :
  let total_champions := 10
  let men_percentage := 0.40
  let men_champions := total_champions * men_percentage
  let bearded_percentage := 0.40
  let non_bearded_percentage := 0.60

  let bearded_men_champions := men_champions * bearded_percentage
  let non_bearded_men_champions := men_champions * non_bearded_percentage

  let bearded_bald_percentage := 0.60
  let bearded_with_hair_percentage := 0.40
  let non_bearded_bald_percentage := 0.30
  let non_bearded_with_hair_percentage := 0.70

  (bearded_men_champions * bearded_bald_percentage).round = 2 ∧
  (bearded_men_champions * bearded_with_hair_percentage).round = 1 ∧
  (non_bearded_men_champions * non_bearded_bald_percentage).round = 2 ∧
  (non_bearded_men_champions * non_bearded_with_hair_percentage).round = 4 :=
by 
sorry

end NUMINAMATH_GPT_scrabble_champions_l384_38458


namespace NUMINAMATH_GPT_rice_weight_per_container_in_grams_l384_38469

-- Define the initial problem conditions
def total_weight_pounds : ℚ := 35 / 6
def number_of_containers : ℕ := 5
def pound_to_grams : ℚ := 453.592

-- Define the expected answer
def expected_answer : ℚ := 529.1907

-- The statement to prove
theorem rice_weight_per_container_in_grams :
  (total_weight_pounds / number_of_containers) * pound_to_grams = expected_answer :=
by
  sorry

end NUMINAMATH_GPT_rice_weight_per_container_in_grams_l384_38469


namespace NUMINAMATH_GPT_complete_square_l384_38428

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end NUMINAMATH_GPT_complete_square_l384_38428


namespace NUMINAMATH_GPT_find_joe_age_l384_38435

noncomputable def billy_age (joe_age : ℕ) : ℕ := 3 * joe_age
noncomputable def emily_age (billy_age joe_age : ℕ) : ℕ := (billy_age + joe_age) / 2

theorem find_joe_age (joe_age : ℕ) 
    (h1 : billy_age joe_age = 3 * joe_age)
    (h2 : emily_age (billy_age joe_age) joe_age = (billy_age joe_age + joe_age) / 2)
    (h3 : billy_age joe_age + joe_age + emily_age (billy_age joe_age) joe_age = 90) : 
    joe_age = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_joe_age_l384_38435


namespace NUMINAMATH_GPT_num_of_nickels_is_two_l384_38482

theorem num_of_nickels_is_two (d n : ℕ) 
    (h1 : 10 * d + 5 * n = 70) 
    (h2 : d + n = 8) : 
    n = 2 := 
by 
    sorry

end NUMINAMATH_GPT_num_of_nickels_is_two_l384_38482


namespace NUMINAMATH_GPT_solve_inequality_for_a_l384_38404

theorem solve_inequality_for_a (a : ℝ) :
  (∀ x : ℝ, abs (x^2 + 3 * a * x + 4 * a) ≤ 3 → x = -3 * a / 2)
  ↔ (a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13) :=
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_for_a_l384_38404


namespace NUMINAMATH_GPT_main_l384_38498

-- Definition for part (a)
def part_a : Prop :=
  ∀ (a b : ℕ), a = 300 ∧ b = 200 → 3^b > 2^a

-- Definition for part (b)
def part_b : Prop :=
  ∀ (c d : ℕ), c = 40 ∧ d = 28 → 3^d > 2^c

-- Definition for part (c)
def part_c : Prop :=
  ∀ (e f : ℕ), e = 44 ∧ f = 53 → 4^f > 5^e

-- Main conjecture proving all parts
theorem main : part_a ∧ part_b ∧ part_c :=
by
  sorry

end NUMINAMATH_GPT_main_l384_38498


namespace NUMINAMATH_GPT_find_initial_principal_amount_l384_38470

noncomputable def compound_interest (initial_principal : ℝ) : ℝ :=
  let year1 := initial_principal * 1.09
  let year2 := (year1 + 500) * 1.10
  let year3 := (year2 - 300) * 1.08
  let year4 := year3 * 1.08
  let year5 := year4 * 1.09
  year5

theorem find_initial_principal_amount :
  ∃ (P : ℝ), (|compound_interest P - 1120| < 0.01) :=
sorry

end NUMINAMATH_GPT_find_initial_principal_amount_l384_38470


namespace NUMINAMATH_GPT_t_shirt_cost_l384_38418

theorem t_shirt_cost (n_tshirts : ℕ) (total_cost : ℝ) (cost_per_tshirt : ℝ)
  (h1 : n_tshirts = 25)
  (h2 : total_cost = 248) :
  cost_per_tshirt = 9.92 :=
by
  sorry

end NUMINAMATH_GPT_t_shirt_cost_l384_38418


namespace NUMINAMATH_GPT_allison_greater_probability_l384_38407

-- Definitions and conditions for the problem
def faceRollAllison : Nat := 6
def facesBrian : List Nat := [1, 3, 3, 5, 5, 6]
def facesNoah : List Nat := [4, 4, 4, 4, 5, 5]

-- Function to calculate probability
def probability_less_than (faces : List Nat) (value : Nat) : ℚ :=
  (faces.filter (fun x => x < value)).length / faces.length

-- Main theorem statement
theorem allison_greater_probability :
  probability_less_than facesBrian 6 * probability_less_than facesNoah 6 = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_allison_greater_probability_l384_38407


namespace NUMINAMATH_GPT_curious_number_is_digit_swap_divisor_l384_38412

theorem curious_number_is_digit_swap_divisor (a b : ℕ) (hab : a ≠ 0 ∧ b ≠ 0) :
  (10 * a + b) ∣ (10 * b + a) → (10 * a + b) = 11 ∨ (10 * a + b) = 22 ∨ (10 * a + b) = 33 ∨ 
  (10 * a + b) = 44 ∨ (10 * a + b) = 55 ∨ (10 * a + b) = 66 ∨ 
  (10 * a + b) = 77 ∨ (10 * a + b) = 88 ∨ (10 * a + b) = 99 :=
by
  sorry

end NUMINAMATH_GPT_curious_number_is_digit_swap_divisor_l384_38412


namespace NUMINAMATH_GPT_gain_percentage_is_15_l384_38491

-- Initial conditions
def CP_A : ℤ := 100
def CP_B : ℤ := 200
def CP_C : ℤ := 300
def SP_A : ℤ := 110
def SP_B : ℤ := 250
def SP_C : ℤ := 330

-- Definitions for total values
def Total_CP : ℤ := CP_A + CP_B + CP_C
def Total_SP : ℤ := SP_A + SP_B + SP_C
def Overall_gain : ℤ := Total_SP - Total_CP
def Gain_percentage : ℚ := (Overall_gain * 100) / Total_CP

-- Theorem to prove the overall gain percentage
theorem gain_percentage_is_15 :
  Gain_percentage = 15 := 
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_gain_percentage_is_15_l384_38491


namespace NUMINAMATH_GPT_no_int_solutions_l384_38444

open Nat

theorem no_int_solutions (p1 p2 α n : ℕ)
  (hp1_prime : p1.Prime)
  (hp2_prime : p2.Prime)
  (hp1_odd : p1 % 2 = 1)
  (hp2_odd : p2 % 2 = 1)
  (hα_pos : 0 < α)
  (hn_pos : 0 < n)
  (hα_gt1 : 1 < α)
  (hn_gt1 : 1 < n) :
  ¬(let lhs := ((p2 - 1) / 2) ^ p1 + ((p2 + 1) / 2) ^ p1
    lhs = α ^ n) :=
sorry

end NUMINAMATH_GPT_no_int_solutions_l384_38444


namespace NUMINAMATH_GPT_simplify_expression_l384_38462

variable {a : ℝ} (h1 : a ≠ -3) (h2 : a ≠ 3) (h3 : a ≠ 2) (h4 : 2 * a + 6 ≠ 0)

theorem simplify_expression : (1 / (a + 3) + 1 / (a ^ 2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l384_38462


namespace NUMINAMATH_GPT_math_study_time_l384_38457

-- Conditions
def science_time : ℕ := 25
def total_time : ℕ := 60

-- Theorem statement
theorem math_study_time :
  total_time - science_time = 35 := by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_math_study_time_l384_38457


namespace NUMINAMATH_GPT_find_g_inv_f_neg7_l384_38433

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_def : ∀ x, f_inv (g x) = 5 * x + 3

theorem find_g_inv_f_neg7 : g_inv (f (-7)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_g_inv_f_neg7_l384_38433


namespace NUMINAMATH_GPT_codys_grandmother_age_l384_38485

theorem codys_grandmother_age
  (cody_age : ℕ)
  (grandmother_multiplier : ℕ)
  (h_cody_age : cody_age = 14)
  (h_grandmother_multiplier : grandmother_multiplier = 6) :
  (cody_age * grandmother_multiplier = 84) :=
by
  sorry

end NUMINAMATH_GPT_codys_grandmother_age_l384_38485


namespace NUMINAMATH_GPT_scientific_notation_correct_l384_38490

-- Define the input number
def input_number : ℕ := 858000000

-- Define the expected scientific notation result
def scientific_notation (n : ℕ) : ℝ := 8.58 * 10^8

-- The theorem states that the input number in scientific notation is indeed 8.58 * 10^8
theorem scientific_notation_correct :
  scientific_notation input_number = 8.58 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_correct_l384_38490


namespace NUMINAMATH_GPT_conner_collected_on_day_two_l384_38454

variable (s0 : ℕ) (c0 : ℕ) (s1 : ℕ) (c1 : ℕ) (c2 : ℕ) (s3 : ℕ) (c3 : ℕ) (total_sydney : ℕ) (total_conner : ℕ)

theorem conner_collected_on_day_two :
  s0 = 837 ∧ c0 = 723 ∧ 
  s1 = 4 ∧ c1 = 8 * s1 ∧
  s3 = 2 * c1 ∧ c3 = 27 ∧
  total_sydney = s0 + s1 + s3 ∧
  total_conner = c0 + c1 + c2 + c3 ∧
  total_conner >= total_sydney
  → c2 = 123 :=
by
  sorry

end NUMINAMATH_GPT_conner_collected_on_day_two_l384_38454


namespace NUMINAMATH_GPT_monotonically_increasing_range_of_a_l384_38476

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_range_of_a_l384_38476


namespace NUMINAMATH_GPT_mary_candies_l384_38446

-- The conditions
def bob_candies : Nat := 10
def sue_candies : Nat := 20
def john_candies : Nat := 5
def sam_candies : Nat := 10
def total_candies : Nat := 50

-- The theorem to prove
theorem mary_candies :
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies) = 5 := by
  -- Here is where the proof would go; currently using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_mary_candies_l384_38446


namespace NUMINAMATH_GPT_rectangle_ratio_l384_38487

theorem rectangle_ratio (s : ℝ) (w h : ℝ) (h_cond : h = 3 * s) (w_cond : w = 2 * s) :
  h / w = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l384_38487


namespace NUMINAMATH_GPT_greatest_value_b_l384_38413

-- Define the polynomial and the inequality condition
def polynomial (b : ℝ) : ℝ := -b^2 + 8*b - 12
#check polynomial
-- State the main theorem with the given condition and the result
theorem greatest_value_b (b : ℝ) : -b^2 + 8*b - 12 ≥ 0 → b ≤ 6 :=
sorry

end NUMINAMATH_GPT_greatest_value_b_l384_38413


namespace NUMINAMATH_GPT_bryan_total_after_discount_l384_38473

theorem bryan_total_after_discount 
  (n : ℕ) (p : ℝ) (d : ℝ) (h_n : n = 8) (h_p : p = 1785) (h_d : d = 0.12) :
  (n * p - (n * p * d) = 12566.4) :=
by
  sorry

end NUMINAMATH_GPT_bryan_total_after_discount_l384_38473


namespace NUMINAMATH_GPT_slope_range_l384_38440

theorem slope_range (a : ℝ) (ha : a ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  ∃ k : ℝ, k = Real.tan a ∧ k ∈ Set.Ici 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_slope_range_l384_38440


namespace NUMINAMATH_GPT_manufacturing_percentage_l384_38405

theorem manufacturing_percentage (a b : ℕ) (h1 : a = 108) (h2 : b = 360) : (a / b : ℚ) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_manufacturing_percentage_l384_38405


namespace NUMINAMATH_GPT_min_workers_for_profit_l384_38499

theorem min_workers_for_profit
    (maintenance_fees : ℝ)
    (worker_hourly_wage : ℝ)
    (widgets_per_hour : ℝ)
    (widget_price : ℝ)
    (work_hours : ℝ)
    (n : ℕ)
    (h_maintenance : maintenance_fees = 470)
    (h_wage : worker_hourly_wage = 10)
    (h_production : widgets_per_hour = 6)
    (h_price : widget_price = 3.5)
    (h_hours : work_hours = 8) :
  470 + 80 * n < 168 * n → n ≥ 6 := 
by
  sorry

end NUMINAMATH_GPT_min_workers_for_profit_l384_38499


namespace NUMINAMATH_GPT_sasha_work_fraction_l384_38481

theorem sasha_work_fraction :
  let sasha_first := 1 / 3
  let sasha_second := 1 / 5
  let sasha_third := 1 / 15
  let total_sasha_contribution := sasha_first + sasha_second + sasha_third
  let fraction_per_car := total_sasha_contribution / 3
  fraction_per_car = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sasha_work_fraction_l384_38481


namespace NUMINAMATH_GPT_perfect_square_mod_3_l384_38442

theorem perfect_square_mod_3 (n : ℤ) : n^2 % 3 = 0 ∨ n^2 % 3 = 1 :=
sorry

end NUMINAMATH_GPT_perfect_square_mod_3_l384_38442


namespace NUMINAMATH_GPT_no_discount_profit_percentage_l384_38437

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 4 / 100  -- 4%
noncomputable def profit_percentage_with_discount : ℝ := 20 / 100  -- 20%

theorem no_discount_profit_percentage : 
  (1 + profit_percentage_with_discount) * cost_price / (1 - discount_percentage) / cost_price - 1 = 0.25 := by
  sorry

end NUMINAMATH_GPT_no_discount_profit_percentage_l384_38437


namespace NUMINAMATH_GPT_count_integers_abs_inequality_l384_38441

theorem count_integers_abs_inequality : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℤ, |(x: ℝ) - 3| ≤ 7.2 ↔ x ∈ {i : ℤ | -4 ≤ i ∧ i ≤ 10} := 
by 
  sorry

end NUMINAMATH_GPT_count_integers_abs_inequality_l384_38441


namespace NUMINAMATH_GPT_johns_gas_usage_per_week_l384_38432

def mpg : ℕ := 30
def miles_to_work_each_way : ℕ := 20
def days_per_week_to_work : ℕ := 5
def leisure_miles_per_week : ℕ := 40

theorem johns_gas_usage_per_week : 
  (2 * miles_to_work_each_way * days_per_week_to_work + leisure_miles_per_week) / mpg = 8 :=
by
  sorry

end NUMINAMATH_GPT_johns_gas_usage_per_week_l384_38432


namespace NUMINAMATH_GPT_line_slope_intercept_l384_38486

theorem line_slope_intercept (a b: ℝ) (h₁: ∀ x y, (x, y) = (2, 3) ∨ (x, y) = (10, 19) → y = a * x + b)
  (h₂: (a * 6 + b) = 11) : a - b = 3 :=
by
  sorry

end NUMINAMATH_GPT_line_slope_intercept_l384_38486


namespace NUMINAMATH_GPT_space_between_trees_l384_38450

theorem space_between_trees (tree_count : ℕ) (tree_space : ℕ) (road_length : ℕ)
  (h1 : tree_space = 1) (h2 : tree_count = 13) (h3 : road_length = 157) :
  (road_length - tree_count * tree_space) / (tree_count - 1) = 12 := by
  sorry

end NUMINAMATH_GPT_space_between_trees_l384_38450


namespace NUMINAMATH_GPT_candies_per_packet_l384_38436

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end NUMINAMATH_GPT_candies_per_packet_l384_38436


namespace NUMINAMATH_GPT_f_inequality_l384_38452

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a > 2 * Real.log a + 3 / 2 :=
sorry

end NUMINAMATH_GPT_f_inequality_l384_38452


namespace NUMINAMATH_GPT_intersection_PQ_eq_23_l384_38456

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | 2 < x}

theorem intersection_PQ_eq_23 : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_PQ_eq_23_l384_38456


namespace NUMINAMATH_GPT_calc_man_dividend_l384_38466

noncomputable def calc_dividend (investment : ℝ) (face_value : ℝ) (premium : ℝ) (dividend_percent : ℝ) : ℝ :=
  let cost_per_share := face_value * (1 + premium / 100)
  let number_of_shares := investment / cost_per_share
  let dividend_per_share := dividend_percent / 100 * face_value
  let total_dividend := dividend_per_share * number_of_shares
  total_dividend

theorem calc_man_dividend :
  calc_dividend 14400 100 20 5 = 600 :=
by
  sorry

end NUMINAMATH_GPT_calc_man_dividend_l384_38466


namespace NUMINAMATH_GPT_sufficient_condition_for_gt_l384_38475

theorem sufficient_condition_for_gt (a : ℝ) : (∀ x : ℝ, x > a → x > 1) → (∃ x : ℝ, x > 1 ∧ x ≤ a) → a > 1 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_gt_l384_38475


namespace NUMINAMATH_GPT_current_failing_rate_l384_38489

def failing_student_rate := 28

def is_failing_student_rate (V : Prop) (n : ℕ) (rate : ℕ) : Prop :=
  (V ∧ rate = 24 ∧ n = 25) ∨ (¬V ∧ rate = 25 ∧ n - 1 = 24)

theorem current_failing_rate (V : Prop) (n : ℕ) (rate : ℕ) :
  is_failing_student_rate V n rate → rate = failing_student_rate :=
by
  sorry

end NUMINAMATH_GPT_current_failing_rate_l384_38489


namespace NUMINAMATH_GPT_rudy_first_run_rate_l384_38414

def first_run_rate (R : ℝ) : Prop :=
  let time_first_run := 5 * R
  let time_second_run := 4 * 9.5
  let total_time := time_first_run + time_second_run
  total_time = 88

theorem rudy_first_run_rate : first_run_rate 10 :=
by
  unfold first_run_rate
  simp
  sorry

end NUMINAMATH_GPT_rudy_first_run_rate_l384_38414


namespace NUMINAMATH_GPT_point_C_correct_l384_38417

-- Definitions of point A and B
def A : ℝ × ℝ := (4, -4)
def B : ℝ × ℝ := (18, 6)

-- Coordinate of C obtained from the conditions of the problem
def C : ℝ × ℝ := (25, 11)

-- Proof statement
theorem point_C_correct :
  ∃ C : ℝ × ℝ, (∃ (BC : ℝ × ℝ), BC = (1/2) • (B.1 - A.1, B.2 - A.2) ∧ C = (B.1 + BC.1, B.2 + BC.2)) ∧ C = (25, 11) :=
by
  sorry

end NUMINAMATH_GPT_point_C_correct_l384_38417


namespace NUMINAMATH_GPT_not_possible_127_points_l384_38403

theorem not_possible_127_points (n_correct n_unanswered n_incorrect : ℕ) :
  n_correct + n_unanswered + n_incorrect = 25 →
  127 ≠ 5 * n_correct + 2 * n_unanswered - n_incorrect :=
by
  intro h_total
  sorry

end NUMINAMATH_GPT_not_possible_127_points_l384_38403


namespace NUMINAMATH_GPT_least_number_divisible_by_6_has_remainder_4_is_40_l384_38421

-- Define the least number N which leaves a remainder of 4 when divided by 6
theorem least_number_divisible_by_6_has_remainder_4_is_40 :
  ∃ (N : ℕ), (∀ (k : ℕ), N = 6 * k + 4) ∧ N = 40 := by
  sorry

end NUMINAMATH_GPT_least_number_divisible_by_6_has_remainder_4_is_40_l384_38421


namespace NUMINAMATH_GPT_simultaneous_equations_solution_l384_38408

-- Definition of the two equations
def eq1 (m x y : ℝ) : Prop := y = m * x + 5
def eq2 (m x y : ℝ) : Prop := y = (3 * m - 2) * x + 6

-- Lean theorem statement to check if the equations have a solution
theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ ∃ x y : ℝ, eq1 m x y ∧ eq2 m x y := 
sorry

end NUMINAMATH_GPT_simultaneous_equations_solution_l384_38408


namespace NUMINAMATH_GPT_john_initial_money_l384_38406

variable (X S : ℕ)
variable (L : ℕ := 500)
variable (cond1 : L = S - 600)
variable (cond2 : X = S + L)

theorem john_initial_money : X = 1600 :=
by
  sorry

end NUMINAMATH_GPT_john_initial_money_l384_38406


namespace NUMINAMATH_GPT_rowing_distance_l384_38415

theorem rowing_distance
  (rowing_speed_in_still_water : ℝ)
  (velocity_of_current : ℝ)
  (total_time : ℝ)
  (H1 : rowing_speed_in_still_water = 5)
  (H2 : velocity_of_current = 1)
  (H3 : total_time = 1) :
  ∃ (D : ℝ), D = 2.4 := 
sorry

end NUMINAMATH_GPT_rowing_distance_l384_38415
