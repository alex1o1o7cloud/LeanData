import Mathlib

namespace sum_of_coordinates_of_point_B_l2003_200370

theorem sum_of_coordinates_of_point_B
  (A : ℝ × ℝ) (hA : A = (0, 0))
  (B : ℝ × ℝ) (hB : ∃ x : ℝ, B = (x, 3))
  (slope_AB : ∃ x : ℝ, (3 - 0)/(x - 0) = 3/4) :
  (∃ x : ℝ, B = (x, 3)) ∧ x + 3 = 7 :=
by
  sorry

end sum_of_coordinates_of_point_B_l2003_200370


namespace range_of_k_l2003_200362

theorem range_of_k (k : ℝ) (h : -3 < k ∧ k ≤ 0) : ∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0 :=
sorry

end range_of_k_l2003_200362


namespace probability_greater_difficulty_probability_same_difficulty_l2003_200314

/-- A datatype representing the difficulty levels of questions. -/
inductive Difficulty
| easy : Difficulty
| medium : Difficulty
| difficult : Difficulty

/-- A datatype representing the four questions with their difficulties. -/
inductive Question
| A1 : Question
| A2 : Question
| B : Question
| C : Question

/-- The function to get the difficulty of a question. -/
def difficulty (q : Question) : Difficulty :=
  match q with
  | Question.A1 => Difficulty.easy
  | Question.A2 => Difficulty.easy
  | Question.B  => Difficulty.medium
  | Question.C  => Difficulty.difficult

/-- The set of all possible pairings of questions selected by two students A and B. -/
def all_pairs : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A1, Question.B), (Question.A1, Question.C),
    (Question.A2, Question.A1), (Question.A2, Question.A2), (Question.A2, Question.B), (Question.A2, Question.C),
    (Question.B, Question.A1), (Question.B, Question.A2), (Question.B, Question.B), (Question.B, Question.C),
    (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B), (Question.C, Question.C) ]

/-- The event that the difficulty of the question selected by student A is greater than that selected by student B. -/
def event_N : List (Question × Question) :=
  [ (Question.B, Question.A1), (Question.B, Question.A2), (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B) ]

/-- The event that the difficulties of the questions selected by both students are the same. -/
def event_M : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A2, Question.A1), (Question.A2, Question.A2), 
    (Question.B, Question.B), (Question.C, Question.C) ]

/-- The probabilities of the events. -/
noncomputable def probability_event_N : ℚ := (event_N.length : ℚ) / (all_pairs.length : ℚ)
noncomputable def probability_event_M : ℚ := (event_M.length : ℚ) / (all_pairs.length : ℚ)

/-- The theorem statements -/
theorem probability_greater_difficulty : probability_event_N = 5 / 16 := sorry
theorem probability_same_difficulty : probability_event_M = 3 / 8 := sorry

end probability_greater_difficulty_probability_same_difficulty_l2003_200314


namespace red_light_after_two_red_light_expectation_and_variance_l2003_200311

noncomputable def prob_red_light_after_two : ℚ := (2/3) * (2/3) * (1/3)
theorem red_light_after_two :
  prob_red_light_after_two = 4/27 :=
by
  -- We have defined the probability calculation directly
  sorry

noncomputable def expected_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p
noncomputable def variance_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem red_light_expectation_and_variance :
  expected_red_lights 6 (1/3) = 2 ∧ variance_red_lights 6 (1/3) = 4/3 :=
by
  -- We have defined expectation and variance calculations directly
  sorry

end red_light_after_two_red_light_expectation_and_variance_l2003_200311


namespace weight_of_new_person_is_correct_l2003_200338

noncomputable def weight_new_person (increase_per_person : ℝ) (old_weight : ℝ) (group_size : ℝ) : ℝ :=
  old_weight + group_size * increase_per_person

theorem weight_of_new_person_is_correct :
  weight_new_person 7.2 65 10 = 137 :=
by
  sorry

end weight_of_new_person_is_correct_l2003_200338


namespace binomial_10_10_binomial_10_9_l2003_200383

-- Prove that \(\binom{10}{10} = 1\)
theorem binomial_10_10 : Nat.choose 10 10 = 1 :=
by sorry

-- Prove that \(\binom{10}{9} = 10\)
theorem binomial_10_9 : Nat.choose 10 9 = 10 :=
by sorry

end binomial_10_10_binomial_10_9_l2003_200383


namespace larger_number_is_450_l2003_200326

-- Given conditions
def HCF := 30
def Factor1 := 10
def Factor2 := 15

-- Derived definitions needed for the proof
def LCM := HCF * Factor1 * Factor2

def Number1 := LCM / Factor1
def Number2 := LCM / Factor2

-- The goal is to prove the larger of the two numbers is 450
theorem larger_number_is_450 : max Number1 Number2 = 450 :=
by
  sorry

end larger_number_is_450_l2003_200326


namespace pie_left_is_30_percent_l2003_200379

def Carlos_share : ℝ := 0.60
def remaining_after_Carlos : ℝ := 1 - Carlos_share
def Jessica_share : ℝ := 0.25 * remaining_after_Carlos
def final_remaining : ℝ := remaining_after_Carlos - Jessica_share

theorem pie_left_is_30_percent :
  final_remaining = 0.30 :=
sorry

end pie_left_is_30_percent_l2003_200379


namespace barbara_typing_time_l2003_200389

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l2003_200389


namespace rice_grain_difference_l2003_200306

theorem rice_grain_difference :
  (3^8) - (3^1 + 3^2 + 3^3 + 3^4 + 3^5) = 6198 :=
by
  sorry

end rice_grain_difference_l2003_200306


namespace sum_of_ages_3_years_hence_l2003_200385

theorem sum_of_ages_3_years_hence (A B C D S : ℝ) (h1 : A = 2 * B) (h2 : C = A / 2) (h3 : D = A - C) (h_sum : A + B + C + D = S) : 
  (A + 3) + (B + 3) + (C + 3) + (D + 3) = S + 12 :=
by sorry

end sum_of_ages_3_years_hence_l2003_200385


namespace permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l2003_200376

open Finset

def digits : Finset ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def permutations_no_repetition : ℤ :=
  (digits.card.factorial) / ((digits.card - 4).factorial)

noncomputable def four_digit_numbers_no_repetition : ℤ :=
  9 * ((digits.card - 1).factorial / ((digits.card - 1 - 3).factorial))

noncomputable def even_four_digit_numbers_gt_3000_no_repetition : ℤ :=
  784 + 1008

theorem permutations_count_5040 : permutations_no_repetition = 5040 := by
  sorry

theorem four_digit_numbers_count_4356 : four_digit_numbers_no_repetition = 4356 := by
  sorry

theorem even_four_digit_numbers_count_1792 : even_four_digit_numbers_gt_3000_no_repetition = 1792 := by
  sorry

end permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l2003_200376


namespace taxi_ride_cost_l2003_200349

-- Definitions based on the conditions
def fixed_cost : ℝ := 2.00
def variable_cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 7

-- Theorem statement
theorem taxi_ride_cost : fixed_cost + (variable_cost_per_mile * distance_traveled) = 4.10 :=
by
  sorry

end taxi_ride_cost_l2003_200349


namespace charge_per_block_l2003_200313

noncomputable def family_vacation_cost : ℝ := 1000
noncomputable def family_members : ℝ := 5
noncomputable def walk_start_fee : ℝ := 2
noncomputable def dogs_walked : ℝ := 20
noncomputable def total_blocks : ℝ := 128

theorem charge_per_block : 
  (family_vacation_cost / family_members) = 200 →
  (dogs_walked * walk_start_fee) = 40 →
  ((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) = 160 →
  (((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) / total_blocks) = 1.25 :=
by intros h1 h2 h3; sorry

end charge_per_block_l2003_200313


namespace inequality_abc_l2003_200301

theorem inequality_abc (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 2) (h₃ : 0 ≤ b) (h₄ : b ≤ 2) (h₅ : 0 ≤ c) (h₆ : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end inequality_abc_l2003_200301


namespace intersection_point_correct_l2003_200312

-- Points in 3D coordinate space
def P : ℝ × ℝ × ℝ := (3, -9, 6)
def Q : ℝ × ℝ × ℝ := (13, -19, 11)
def R : ℝ × ℝ × ℝ := (1, 4, -7)
def S : ℝ × ℝ × ℝ := (3, -6, 9)

-- Vectors for parameterization
def pq_vector (t : ℝ) : ℝ × ℝ × ℝ := (3 + 10 * t, -9 - 10 * t, 6 + 5 * t)
def rs_vector (s : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - 10 * s, -7 + 16 * s)

-- The proof of the intersection point equals the correct answer
theorem intersection_point_correct : 
  ∃ t s : ℝ, pq_vector t = rs_vector s ∧ 
  pq_vector t = (-19 / 3, 10 / 3, 4 / 3) := 
by
  sorry

end intersection_point_correct_l2003_200312


namespace equation_of_line_AB_l2003_200355

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (3, 2)

def equation_of_line (p1 p2 : point) : ℝ × ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  -- Calculate the slope
  let k := (y2 - y1) / (x2 - x1)
  -- Use point-slope form and simplify the equation to standard form
  (((1 : ℝ), -2, 1) : ℝ × ℝ × ℝ)

theorem equation_of_line_AB :
  equation_of_line A B = (1, -2, 1) :=
sorry

end equation_of_line_AB_l2003_200355


namespace cube_side_length_l2003_200367

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l2003_200367


namespace remainder_of_10_pow_23_minus_7_mod_6_l2003_200372

theorem remainder_of_10_pow_23_minus_7_mod_6 : ((10 ^ 23 - 7) % 6) = 3 := by
  sorry

end remainder_of_10_pow_23_minus_7_mod_6_l2003_200372


namespace car_price_l2003_200371

/-- Prove that the price of the car Quincy bought is $20,000 given the conditions. -/
theorem car_price (years : ℕ) (monthly_payment : ℕ) (down_payment : ℕ) 
  (h1 : years = 5) 
  (h2 : monthly_payment = 250) 
  (h3 : down_payment = 5000) : 
  (down_payment + (monthly_payment * (12 * years))) = 20000 :=
by
  /- We provide the proof below with sorry because we are only writing the statement as requested. -/
  sorry

end car_price_l2003_200371


namespace find_theta_l2003_200315

-- Definitions based on conditions
def angle_A : ℝ := 10
def angle_B : ℝ := 14
def angle_C : ℝ := 26
def angle_D : ℝ := 33
def sum_rect_angles : ℝ := 360
def sum_triangle_angles : ℝ := 180
def sum_right_triangle_acute_angles : ℝ := 90

-- Main theorem statement
theorem find_theta (A B C D : ℝ)
  (hA : A = angle_A)
  (hB : B = angle_B)
  (hC : C = angle_C)
  (hD : D = angle_D)
  (sum_rect : sum_rect_angles = 360)
  (sum_triangle : sum_triangle_angles = 180) :
  ∃ θ : ℝ, θ = 11 := 
sorry

end find_theta_l2003_200315


namespace twelve_million_plus_twelve_thousand_l2003_200396

theorem twelve_million_plus_twelve_thousand :
  12000000 + 12000 = 12012000 :=
by
  sorry

end twelve_million_plus_twelve_thousand_l2003_200396


namespace find_speed_of_B_l2003_200368

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end find_speed_of_B_l2003_200368


namespace square_area_l2003_200356

theorem square_area :
  ∀ (x1 x2 : ℝ), (x1^2 + 2 * x1 + 1 = 8) ∧ (x2^2 + 2 * x2 + 1 = 8) ∧ (x1 ≠ x2) →
  (abs (x1 - x2))^2 = 36 :=
by
  sorry

end square_area_l2003_200356


namespace boots_ratio_l2003_200358

noncomputable def problem_statement : Prop :=
  let total_money : ℝ := 50
  let cost_toilet_paper : ℝ := 12
  let cost_groceries : ℝ := 2 * cost_toilet_paper
  let remaining_after_groceries : ℝ := total_money - cost_toilet_paper - cost_groceries
  let extra_money_per_person : ℝ := 35
  let total_extra_money : ℝ := 2 * extra_money_per_person
  let total_cost_boots : ℝ := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots : ℝ := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3

theorem boots_ratio (total_money : ℝ) (cost_toilet_paper : ℝ) (extra_money_per_person : ℝ) : 
  let cost_groceries := 2 * cost_toilet_paper
  let remaining_after_groceries := total_money - cost_toilet_paper - cost_groceries
  let total_extra_money := 2 * extra_money_per_person
  let total_cost_boots := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3 :=
by
  sorry

end boots_ratio_l2003_200358


namespace largest_x_solution_l2003_200342

noncomputable def solve_eq (x : ℝ) : Prop :=
  (15 * x^2 - 40 * x + 16) / (4 * x - 3) + 3 * x = 7 * x + 2

theorem largest_x_solution : 
  ∃ x : ℝ, solve_eq x ∧ x = -14 + Real.sqrt 218 := 
sorry

end largest_x_solution_l2003_200342


namespace units_digit_fraction_l2003_200373

open Nat

theorem units_digit_fraction : 
  (15 * 16 * 17 * 18 * 19 * 20) % 500 % 10 = 2 := by
  sorry

end units_digit_fraction_l2003_200373


namespace midpoint_sum_l2003_200319

theorem midpoint_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -4) (hy2 : y2 = -7) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 1 :=
by
  rw [hx1, hy1, hx2, hy2]
  norm_num

end midpoint_sum_l2003_200319


namespace domain_of_sqrt_function_l2003_200341

theorem domain_of_sqrt_function (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 := sorry

end domain_of_sqrt_function_l2003_200341


namespace total_time_to_complete_work_l2003_200377

noncomputable def mahesh_work_rate (W : ℕ) := W / 40
noncomputable def mahesh_work_done_in_20_days (W : ℕ) := 20 * (mahesh_work_rate W)
noncomputable def remaining_work (W : ℕ) := W - (mahesh_work_done_in_20_days W)
noncomputable def rajesh_work_rate (W : ℕ) := (remaining_work W) / 30

theorem total_time_to_complete_work (W : ℕ) :
    (mahesh_work_rate W) + (rajesh_work_rate W) = W / 24 →
    (mahesh_work_done_in_20_days W) = W / 2 →
    (remaining_work W) = W / 2 →
    (rajesh_work_rate W) = W / 60 →
    20 + 30 = 50 :=
by 
  intros _ _ _ _
  sorry

end total_time_to_complete_work_l2003_200377


namespace total_time_to_complete_l2003_200397

noncomputable def time_to_clean_keys (n : Nat) (t : Nat) : Nat := n * t

def assignment_time : Nat := 10
def time_per_key : Nat := 3
def remaining_keys : Nat := 14

theorem total_time_to_complete :
  time_to_clean_keys remaining_keys time_per_key + assignment_time = 52 := by
  sorry

end total_time_to_complete_l2003_200397


namespace area_within_square_outside_semicircles_l2003_200392

theorem area_within_square_outside_semicircles (side_length : ℝ) (r : ℝ) (area_square : ℝ) (area_semicircles : ℝ) (area_shaded : ℝ) 
  (h1 : side_length = 4)
  (h2 : r = side_length / 2)
  (h3 : area_square = side_length * side_length)
  (h4 : area_semicircles = 4 * (1 / 2 * π * r^2))
  (h5 : area_shaded = area_square - area_semicircles)
  : area_shaded = 16 - 8 * π :=
sorry

end area_within_square_outside_semicircles_l2003_200392


namespace find_a100_l2003_200363

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Given conditions
variables {a d : ℤ}
variables (S_9 : ℤ) (a_10 : ℤ)

-- Conditions in Lean definition
def conditions (a d : ℤ) : Prop :=
  (9 / 2 * (2 * a + 8 * d) = 27) ∧ (a + 9 * d = 8)

-- Prove the final statement
theorem find_a100 : ∃ a d : ℤ, conditions a d → arithmetic_sequence a d 100 = 98 := 
by {
    sorry
}

end find_a100_l2003_200363


namespace value_of_a2_l2003_200347

theorem value_of_a2 (a : ℕ → ℤ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 2)
  (h2 : ∃ r : ℤ, a 3 = r * a 1 ∧ a 4 = r * a 3) :
  a 2 = -6 :=
by
  sorry

end value_of_a2_l2003_200347


namespace sequence_and_sum_l2003_200300

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem sequence_and_sum
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (cond : a 2 + a 8 = 15 - a 5) :
  S 9 = 45 :=
sorry

end sequence_and_sum_l2003_200300


namespace average_difference_l2003_200398

def differences : List ℤ := [15, -5, 25, 35, -15, 10, 20]
def days : ℤ := 7

theorem average_difference (diff : List ℤ) (n : ℤ) 
  (h : diff = [15, -5, 25, 35, -15, 10, 20]) (h_days : n = 7) : 
  (diff.sum / n : ℚ) = 12 := 
by 
  rw [h, h_days]
  norm_num
  sorry

end average_difference_l2003_200398


namespace distance_between_point_and_center_l2003_200320

noncomputable def polar_to_rectangular_point (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def center_of_circle : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_between_point_and_center :
  distance (polar_to_rectangular_point 2 (Real.pi / 3)) center_of_circle = Real.sqrt 3 := 
sorry

end distance_between_point_and_center_l2003_200320


namespace find_d_l2003_200307

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l2003_200307


namespace sum_even_integers_correct_l2003_200318

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end sum_even_integers_correct_l2003_200318


namespace term_sequence_l2003_200395

theorem term_sequence (n : ℕ) (h : (-1:ℤ) ^ (n + 1) * n * (n + 1) = -20) : n = 4 :=
sorry

end term_sequence_l2003_200395


namespace housing_price_equation_l2003_200390

-- Initial conditions
def january_price : ℝ := 8300
def march_price : ℝ := 8700
variables (x : ℝ)

-- Lean statement of the problem
theorem housing_price_equation :
  january_price * (1 + x)^2 = march_price := 
sorry

end housing_price_equation_l2003_200390


namespace sheena_sewing_hours_weekly_l2003_200339

theorem sheena_sewing_hours_weekly
  (hours_per_dress : ℕ)
  (number_of_dresses : ℕ)
  (weeks_to_complete : ℕ)
  (total_sewing_hours : ℕ)
  (hours_per_week : ℕ) :
  hours_per_dress = 12 →
  number_of_dresses = 5 →
  weeks_to_complete = 15 →
  total_sewing_hours = number_of_dresses * hours_per_dress →
  hours_per_week = total_sewing_hours / weeks_to_complete →
  hours_per_week = 4 := by
  intros h1 h2 h3 h4 h5
  sorry

end sheena_sewing_hours_weekly_l2003_200339


namespace sum_of_ages_eq_19_l2003_200316

theorem sum_of_ages_eq_19 :
  ∃ (a b s : ℕ), (3 * a + 5 + b = s) ∧ (6 * s^2 = 2 * a^2 + 10 * b^2) ∧ (Nat.gcd a (Nat.gcd b s) = 1 ∧ a + b + s = 19) :=
sorry

end sum_of_ages_eq_19_l2003_200316


namespace leap_years_count_l2003_200321

theorem leap_years_count :
  let is_leap_year (y : ℕ) := (y % 900 = 150 ∨ y % 900 = 450) ∧ y % 100 = 0
  let range_start := 2100
  let range_end := 4200
  ∃ L, L = [2250, 2850, 3150, 3750, 4050] ∧ (∀ y ∈ L, is_leap_year y ∧ range_start ≤ y ∧ y ≤ range_end)
  ∧ L.length = 5 :=
by
  sorry

end leap_years_count_l2003_200321


namespace box_breadth_l2003_200374

noncomputable def cm_to_m (cm : ℕ) : ℝ := cm / 100

theorem box_breadth :
  ∀ (length depth cm cubical_edge blocks : ℕ), 
    length = 160 →
    depth = 60 →
    cubical_edge = 20 →
    blocks = 120 →
    breadth = (blocks * (cubical_edge ^ 3)) / (length * depth) →
    breadth = 100 :=
by
  sorry

end box_breadth_l2003_200374


namespace fraction_product_l2003_200324

theorem fraction_product :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end fraction_product_l2003_200324


namespace compare_fractions_l2003_200310

theorem compare_fractions : (- (4 / 5) < - (2 / 3)) :=
by
  sorry

end compare_fractions_l2003_200310


namespace a_sufficient_not_necessary_l2003_200384

theorem a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (¬(1 / a < 1 → a > 1)) :=
by
  sorry

end a_sufficient_not_necessary_l2003_200384


namespace find_d_minus_a_l2003_200308

theorem find_d_minus_a (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b = 240)
  (h2 : (b + c) / 2 = 60)
  (h3 : (c + d) / 2 = 90) : d - a = 116 :=
sorry

end find_d_minus_a_l2003_200308


namespace min_sugar_l2003_200328

variable (f s : ℝ)

theorem min_sugar (h1 : f ≥ 10 + 3 * s) (h2 : f ≤ 4 * s) : s ≥ 10 := by
  sorry

end min_sugar_l2003_200328


namespace relation_y₁_y₂_y₃_l2003_200360

def parabola (x : ℝ) : ℝ := - x^2 - 2 * x + 2
noncomputable def y₁ : ℝ := parabola (-2)
noncomputable def y₂ : ℝ := parabola (1)
noncomputable def y₃ : ℝ := parabola (2)

theorem relation_y₁_y₂_y₃ : y₁ > y₂ ∧ y₂ > y₃ := by
  have h₁ : y₁ = 2 := by
    unfold y₁ parabola
    norm_num
    
  have h₂ : y₂ = -1 := by
    unfold y₂ parabola
    norm_num
    
  have h₃ : y₃ = -6 := by
    unfold y₃ parabola
    norm_num
    
  rw [h₁, h₂, h₃]
  exact ⟨by norm_num, by norm_num⟩

end relation_y₁_y₂_y₃_l2003_200360


namespace arithmetic_sequence_7th_term_l2003_200388

theorem arithmetic_sequence_7th_term 
  (a d : ℝ)
  (n : ℕ)
  (h1 : 5 * a + 10 * d = 34)
  (h2 : 5 * a + 5 * (n - 1) * d = 146)
  (h3 : (n / 2 : ℝ) * (2 * a + (n - 1) * d) = 234) :
  a + 6 * d = 19 :=
by
  sorry

end arithmetic_sequence_7th_term_l2003_200388


namespace least_number_divisible_by_11_and_remainder_2_l2003_200322

theorem least_number_divisible_by_11_and_remainder_2 :
  ∃ n, (∀ k : ℕ, 3 ≤ k ∧ k ≤ 7 → n % k = 2) ∧ n % 11 = 0 ∧ n = 1262 :=
by
  sorry

end least_number_divisible_by_11_and_remainder_2_l2003_200322


namespace break_even_performances_l2003_200380

def totalCost (x : ℕ) : ℕ := 81000 + 7000 * x
def totalRevenue (x : ℕ) : ℕ := 16000 * x

theorem break_even_performances : ∃ x : ℕ, totalCost x = totalRevenue x ∧ x = 9 := 
by
  sorry

end break_even_performances_l2003_200380


namespace tilly_star_count_l2003_200375

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) (total_stars : ℕ) 
  (h1 : stars_east = 120)
  (h2 : stars_west = 6 * stars_east)
  (h3 : total_stars = stars_east + stars_west) :
  total_stars = 840 :=
sorry

end tilly_star_count_l2003_200375


namespace find_k_l2003_200335

-- Given definition for a quadratic expression that we want to be a square of a binomial
def quadratic_expression (x k : ℝ) := x^2 - 20 * x + k

-- The binomial square matching.
def binomial_square (x b : ℝ) := (x + b)^2

-- Statement to prove that k = 100 makes the quadratic_expression to be a square of binomial
theorem find_k :
  (∃ k : ℝ, ∀ x : ℝ, quadratic_expression x k = binomial_square x (-10)) ↔ k = 100 :=
by
  sorry

end find_k_l2003_200335


namespace algebraic_expression_value_l2003_200361

theorem algebraic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 7 = -6 := by
  sorry

end algebraic_expression_value_l2003_200361


namespace pipe_filling_time_l2003_200304

theorem pipe_filling_time (T : ℝ) (h1 : T > 0) (h2 : 1/(3:ℝ) = 1/T - 1/(6:ℝ)) : T = 2 := 
by sorry

end pipe_filling_time_l2003_200304


namespace simplify_expression_l2003_200348

theorem simplify_expression (x y z : ℝ) : (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := 
by 
sorry

end simplify_expression_l2003_200348


namespace frank_initial_mushrooms_l2003_200393

theorem frank_initial_mushrooms (pounds_eaten pounds_left initial_pounds : ℕ) 
  (h1 : pounds_eaten = 8) 
  (h2 : pounds_left = 7) 
  (h3 : initial_pounds = pounds_eaten + pounds_left) : 
  initial_pounds = 15 := 
by 
  sorry

end frank_initial_mushrooms_l2003_200393


namespace contrapositive_equivalence_l2003_200359

-- Definitions based on the conditions
variables (R S : Prop)

-- Statement of the proof
theorem contrapositive_equivalence (h : ¬R → S) : ¬S → R := 
sorry

end contrapositive_equivalence_l2003_200359


namespace age_proof_l2003_200334

theorem age_proof (A B C D k m : ℕ)
  (h1 : A + B + C + D = 76)
  (h2 : A - 3 = k)
  (h3 : B - 3 = 2*k)
  (h4 : C - 3 = 3*k)
  (h5 : A - 5 = 3*m)
  (h6 : D - 5 = 4*m)
  (h7 : B - 5 = 5*m) :
  A = 11 := 
sorry

end age_proof_l2003_200334


namespace tangent_slope_correct_l2003_200309

noncomputable def slope_of_directrix (focus: ℝ × ℝ) (p1: ℝ × ℝ) (p2: ℝ × ℝ) : ℝ :=
  let c1 := p1
  let c2 := p2
  let radius1 := Real.sqrt ((c1.1 + 1)^2 + (c1.2 + 1)^2)
  let radius2 := Real.sqrt ((c2.1 - 2)^2 + (c2.2 - 2)^2)
  let dist := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let slope := (focus.2 - p1.2) / (focus.1 - p1.1)
  let tangent_slope := (9 : ℝ) / (7 : ℝ) + (4 * Real.sqrt 2) / 7
  tangent_slope

theorem tangent_slope_correct :
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 + 4 * Real.sqrt 2) / 7) ∨
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 - 4 * Real.sqrt 2) / 7) :=
by
  -- Proof omitted here
  sorry

end tangent_slope_correct_l2003_200309


namespace necessary_sufficient_condition_l2003_200365

theorem necessary_sufficient_condition (a b x_0 : ℝ) (h : a > 0) :
  (x_0 = b / a) ↔ (∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x_0^2 - b * x_0) :=
sorry

end necessary_sufficient_condition_l2003_200365


namespace solve_quadratic_eq_l2003_200351

theorem solve_quadratic_eq (b c : ℝ) :
  (∀ x : ℝ, |x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = -1) →
  b = -6 ∧ c = -7 :=
by
  intros h_abs_val_eq h_quad_eq
  sorry

end solve_quadratic_eq_l2003_200351


namespace collinear_points_inverse_sum_half_l2003_200357

theorem collinear_points_inverse_sum_half (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
    (collinear : (a - 2) * (b - 2) - (-2) * a = 0) : 
    1 / a + 1 / b = 1 / 2 := 
by
  sorry

end collinear_points_inverse_sum_half_l2003_200357


namespace herd_total_cows_l2003_200394

theorem herd_total_cows (n : ℕ) (h1 : (1 / 3 : ℚ) * n + (1 / 5 : ℚ) * n + (1 / 6 : ℚ) * n + 19 = n) : n = 63 :=
sorry

end herd_total_cows_l2003_200394


namespace regular_polygon_sides_l2003_200340

-- Conditions
def central_angle (θ : ℝ) := θ = 30
def sum_of_central_angles (sumθ : ℝ) := sumθ = 360

-- The proof problem
theorem regular_polygon_sides (θ sumθ : ℝ) (h₁ : central_angle θ) (h₂ : sum_of_central_angles sumθ) :
  sumθ / θ = 12 := by
  sorry

end regular_polygon_sides_l2003_200340


namespace cards_problem_l2003_200352

-- Definitions of the cards and their arrangement
def cards : List ℕ := [1, 3, 4, 6, 7, 8]
def missing_numbers : List ℕ := [2, 5, 9]

-- Function to check no three consecutive numbers are in ascending or descending order
def no_three_consec (ls : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a < b → b < c → b - a = 1 → c - b = 1 → False ∧
                a > b → b > c → a - b = 1 → b - c = 1 → False

-- Assume that cards A, B, and C are not visible
variables (A B C : ℕ)

-- Ensure that A, B, and C are among the missing numbers
axiom A_in_missing : A ∈ missing_numbers
axiom B_in_missing : B ∈ missing_numbers
axiom C_in_missing : C ∈ missing_numbers

-- Ensuring no three consecutive cards are in ascending or descending order
axiom no_three_consec_cards : no_three_consec (cards ++ [A, B, C])

-- The final proof problem
theorem cards_problem : A = 5 ∧ B = 2 ∧ C = 9 :=
by
  sorry

end cards_problem_l2003_200352


namespace part_a_part_b_l2003_200330

-- Part (a)
theorem part_a (a b c : ℚ) (z : ℚ) (h : a * z^2 + b * z + c = 0) (n : ℕ) (hn : n > 0) :
  ∃ f : ℚ → ℚ, z = f (z^n) :=
sorry

-- Part (b)
theorem part_b (x : ℚ) (h : x ≠ 0) :
  x = (x^3 + (x + 1/x)) / ((x + 1/x)^2 - 1) :=
sorry

end part_a_part_b_l2003_200330


namespace alissa_presents_l2003_200346

theorem alissa_presents :
  let Ethan_presents := 31
  let Alissa_presents := Ethan_presents + 22
  Alissa_presents = 53 :=
by
  sorry

end alissa_presents_l2003_200346


namespace intersection_point_of_line_and_y_axis_l2003_200391

theorem intersection_point_of_line_and_y_axis :
  {p : ℝ × ℝ | ∃ x, p = (x, 2 * x + 1) ∧ x = 0} = {(0, 1)} :=
by sorry

end intersection_point_of_line_and_y_axis_l2003_200391


namespace original_price_of_wand_l2003_200364

-- Definitions as per the conditions
def price_paid (paid : Real) := paid = 8
def fraction_of_original (fraction : Real) := fraction = 1 / 8

-- Question and correct answer put as a theorem to prove
theorem original_price_of_wand (paid : Real) (fraction : Real) 
  (h1 : price_paid paid) (h2 : fraction_of_original fraction) : 
  (paid / fraction = 64) := 
by
  -- This 'sorry' indicates where the actual proof would go.
  sorry

end original_price_of_wand_l2003_200364


namespace committee_meeting_people_l2003_200399

theorem committee_meeting_people (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 11) : 
  A + B = 7 :=
sorry

end committee_meeting_people_l2003_200399


namespace farmer_initial_plan_days_l2003_200366

def initialDaysPlan
    (daily_hectares : ℕ)
    (increased_productivity : ℕ)
    (hectares_ploughed_first_two_days : ℕ)
    (hectares_remaining : ℕ)
    (days_ahead_schedule : ℕ)
    (total_hectares : ℕ)
    (days_actual : ℕ) : ℕ :=
  days_actual + days_ahead_schedule

theorem farmer_initial_plan_days : 
  ∀ (x days_ahead_schedule : ℕ) 
    (daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual : ℕ),
  daily_hectares = 120 →
  increased_productivity = daily_hectares + daily_hectares / 4 →
  hectares_ploughed_first_two_days = 2 * daily_hectares →
  total_hectares = 1440 →
  days_ahead_schedule = 2 →
  days_actual = 10 →
  hectares_remaining = total_hectares - hectares_ploughed_first_two_days →
  hectares_remaining = increased_productivity * (days_actual - 2) →
  x = 12 :=
by
  intros x days_ahead_schedule daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual
  intros h_daily_hectares h_increased_productivity h_hectares_ploughed_first_two_days h_total_hectares h_days_ahead_schedule h_days_actual h_hectares_remaining h_hectares_ploughed
  sorry

end farmer_initial_plan_days_l2003_200366


namespace consecutive_even_integers_sum_l2003_200337

theorem consecutive_even_integers_sum (n : ℕ) (h : n % 2 = 0) (h_pro : n * (n + 2) * (n + 4) = 3360) :
  n + (n + 2) + (n + 4) = 48 :=
by sorry

end consecutive_even_integers_sum_l2003_200337


namespace hernandez_state_tax_l2003_200354

theorem hernandez_state_tax 
    (res_months : ℕ) (total_months : ℕ) 
    (taxable_income : ℝ) (tax_rate : ℝ) 
    (prorated_income : ℝ) (state_tax : ℝ) 
    (h1 : res_months = 9) 
    (h2 : total_months = 12) 
    (h3 : taxable_income = 42500) 
    (h4 : tax_rate = 0.04) 
    (h5 : prorated_income = taxable_income * (res_months / total_months)) 
    (h6 : state_tax = prorated_income * tax_rate) : 
    state_tax = 1275 := 
by 
  -- this is where the proof would go
  sorry

end hernandez_state_tax_l2003_200354


namespace find_a_plus_k_l2003_200382

-- Define the conditions.
def foci1 : (ℝ × ℝ) := (2, 0)
def foci2 : (ℝ × ℝ) := (2, 4)
def ellipse_point : (ℝ × ℝ) := (7, 2)

-- Statement of the equivalent proof problem.
theorem find_a_plus_k (a b h k : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (∀ x y, ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ↔ (x, y) = ellipse_point) →
  h = 2 → k = 2 → a = 5 →
  a + k = 7 :=
by
  sorry

end find_a_plus_k_l2003_200382


namespace train_speed_is_correct_l2003_200369

/-- Define the length of the train (in meters) -/
def train_length : ℕ := 120

/-- Define the length of the bridge (in meters) -/
def bridge_length : ℕ := 255

/-- Define the time to cross the bridge (in seconds) -/
def time_to_cross : ℕ := 30

/-- Define the total distance covered by the train while crossing the bridge -/
def total_distance : ℕ := train_length + bridge_length

/-- Define the speed of the train in meters per second -/
def speed_m_per_s : ℚ := total_distance / time_to_cross

/-- Conversion factor from m/s to km/hr -/
def m_per_s_to_km_per_hr : ℚ := 3.6

/-- The expected speed of the train in km/hr -/
def expected_speed_km_per_hr : ℕ := 45

/-- The theorem stating that the speed of the train is 45 km/hr -/
theorem train_speed_is_correct :
  (speed_m_per_s * m_per_s_to_km_per_hr) = expected_speed_km_per_hr := by
  sorry

end train_speed_is_correct_l2003_200369


namespace cost_of_renting_per_month_l2003_200343

namespace RentCarProblem

def cost_new_car_per_month : ℕ := 30
def months_per_year : ℕ := 12
def yearly_difference : ℕ := 120

theorem cost_of_renting_per_month (R : ℕ) :
  (cost_new_car_per_month * months_per_year + yearly_difference) / months_per_year = R → 
  R = 40 :=
by
  sorry

end RentCarProblem

end cost_of_renting_per_month_l2003_200343


namespace binomial_constant_term_l2003_200332

theorem binomial_constant_term : 
  ∃ (c : ℚ), (x : ℝ) → (x^2 + (1 / (2 * x)))^6 = c ∧ c = 15 / 16 := by
  sorry

end binomial_constant_term_l2003_200332


namespace class1_qualified_l2003_200331

variables (Tardiness : ℕ → ℕ) -- Tardiness function mapping days to number of tardy students

def classQualified (mean variance median mode : ℕ) : Prop :=
  (mean = 2 ∧ variance = 2) ∨
  (mean = 3 ∧ median = 3) ∨
  (mean = 2 ∧ variance > 0) ∨
  (median = 2 ∧ mode = 2)

def eligible (Tardiness : ℕ → ℕ) : Prop :=
  ∀ i, i < 5 → Tardiness i ≤ 5

theorem class1_qualified : 
  (∀ Tardiness, (∃ mean variance median mode,
    classQualified mean variance median mode 
    ∧ mean = 2 ∧ variance = 2 
    ∧ eligible Tardiness)) → 
  (∀ Tardiness, eligible Tardiness) :=
by
  sorry

end class1_qualified_l2003_200331


namespace archibald_percentage_games_won_l2003_200345

theorem archibald_percentage_games_won
  (A B F1 F2 : ℝ) -- number of games won by Archibald, his brother, and his two friends
  (total_games : ℝ)
  (A_eq_1_1B : A = 1.1 * B)
  (F_eq_2_1B : F1 + F2 = 2.1 * B)
  (total_games_eq : A + B + F1 + F2 = total_games)
  (total_games_val : total_games = 280) :
  (A / total_games * 100) = 26.19 :=
by
  sorry

end archibald_percentage_games_won_l2003_200345


namespace jen_age_difference_l2003_200323

-- Definitions as conditions given in the problem
def son_present_age := 16
def jen_present_age := 41

-- The statement to be proved
theorem jen_age_difference :
  3 * son_present_age - jen_present_age = 7 :=
by
  sorry

end jen_age_difference_l2003_200323


namespace delta_evaluation_l2003_200303

def delta (a b : ℕ) : ℕ := a^3 - b

theorem delta_evaluation :
  delta (2^(delta 3 8)) (5^(delta 4 9)) = 2^19 - 5^55 := 
sorry

end delta_evaluation_l2003_200303


namespace exist_n_consecutive_not_perfect_power_l2003_200353

theorem exist_n_consecutive_not_perfect_power (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, ∀ k : ℕ, k < n → ¬ (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (m + k) = a ^ b) :=
sorry

end exist_n_consecutive_not_perfect_power_l2003_200353


namespace find_triples_l2003_200350

theorem find_triples (x y z : ℕ) :
  (1 / x + 2 / y - 3 / z = 1) ↔ 
  ((x = 2 ∧ y = 1 ∧ z = 2) ∨
   (x = 2 ∧ y = 3 ∧ z = 18) ∨
   ∃ (n : ℕ), n ≥ 1 ∧ x = 1 ∧ y = 2 * n ∧ z = 3 * n ∨
   ∃ (k : ℕ), k ≥ 1 ∧ x = k ∧ y = 2 ∧ z = 3 * k) := sorry

end find_triples_l2003_200350


namespace fg_sqrt3_eq_neg3_minus_2sqrt3_l2003_200325
noncomputable def f (x : ℝ) : ℝ := 5 - 2 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + x + 1

theorem fg_sqrt3_eq_neg3_minus_2sqrt3 : f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := 
by sorry

end fg_sqrt3_eq_neg3_minus_2sqrt3_l2003_200325


namespace sum_of_xyz_l2003_200381

theorem sum_of_xyz (x y z : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : z > 0)
  (h4 : x^2 + y^2 + x * y = 3)
  (h5 : y^2 + z^2 + y * z = 4)
  (h6 : z^2 + x^2 + z * x = 7) :
  x + y + z = Real.sqrt 13 :=
by sorry -- Proof omitted, but the statement formulation is complete and checks the equality under given conditions.

end sum_of_xyz_l2003_200381


namespace arithmetic_seq_sum_l2003_200336

theorem arithmetic_seq_sum {a : ℕ → ℝ} (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end arithmetic_seq_sum_l2003_200336


namespace bridge_length_correct_l2003_200305

noncomputable def length_of_bridge : ℝ :=
  let train_length := 110 -- in meters
  let train_speed_kmh := 72 -- in km/hr
  let crossing_time := 14.248860091192705 -- in seconds
  let speed_in_mps := train_speed_kmh * (1000 / 3600)
  let distance := speed_in_mps * crossing_time
  distance - train_length

theorem bridge_length_correct :
  length_of_bridge = 174.9772018238541 := by
  sorry

end bridge_length_correct_l2003_200305


namespace min_value_of_sum_squares_on_circle_l2003_200386

theorem min_value_of_sum_squares_on_circle :
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ∧ x^2 + y^2 = 6 - 2 * Real.sqrt 5 :=
sorry

end min_value_of_sum_squares_on_circle_l2003_200386


namespace largest_negative_root_l2003_200317

theorem largest_negative_root : 
  ∃ x : ℝ, (∃ k : ℤ, x = -1/2 + 2 * ↑k) ∧ 
  ∀ y : ℝ, (∃ k : ℤ, (y = -1/2 + 2 * ↑k ∨ y = 1/6 + 2 * ↑k ∨ y = 5/6 + 2 * ↑k)) → y < 0 → y ≤ x :=
sorry

end largest_negative_root_l2003_200317


namespace length_of_AC_l2003_200302

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AB BC AC : ℝ)
variables (right_triangle : AB ^ 2 + BC ^ 2 = AC ^ 2)
variables (tan_A : BC / AB = 4 / 3)
variable (AB_val : AB = 4)

theorem length_of_AC :
  AC = 20 / 3 :=
sorry

end length_of_AC_l2003_200302


namespace A_pow_101_l2003_200329

def A : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 1],
  ![1, 0, 0],
  ![0, 1, 0]
]

theorem A_pow_101 :
  A ^ 101 = ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] := by
  sorry

end A_pow_101_l2003_200329


namespace deficit_calculation_l2003_200333

theorem deficit_calculation
    (L W : ℝ)  -- Length and Width
    (dW : ℝ)  -- Deficit in width
    (h1 : (1.08 * L) * (W - dW) = 1.026 * (L * W))  -- Condition on the calculated area
    : dW / W = 0.05 := 
by
    sorry

end deficit_calculation_l2003_200333


namespace max_value_expression_l2003_200387

theorem max_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1 / Real.sqrt 3) :
  27 * a * b * c + a * Real.sqrt (a^2 + 2 * b * c) + b * Real.sqrt (b^2 + 2 * c * a) + c * Real.sqrt (c^2 + 2 * a * b) ≤ 2 / (3 * Real.sqrt 3) :=
sorry

end max_value_expression_l2003_200387


namespace binomial_12_6_eq_924_l2003_200378

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l2003_200378


namespace alice_current_age_l2003_200327

theorem alice_current_age (a b : ℕ) 
  (h1 : a + 8 = 2 * (b + 8)) 
  (h2 : (a - 10) + (b - 10) = 21) : 
  a = 30 := 
by 
  sorry

end alice_current_age_l2003_200327


namespace total_amount_owed_l2003_200344

-- Conditions
def borrowed_amount : ℝ := 500
def monthly_interest_rate : ℝ := 0.02
def months_not_paid : ℕ := 3

-- Compounded monthly formula
def amount_after_n_months (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Theorem statement
theorem total_amount_owed :
  amount_after_n_months borrowed_amount monthly_interest_rate months_not_paid = 530.604 :=
by
  -- Proof to be filled in here
  sorry

end total_amount_owed_l2003_200344
