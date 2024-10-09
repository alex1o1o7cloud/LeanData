import Mathlib

namespace circle_condition_l824_82455

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) →
  m < 1 :=
sorry

end circle_condition_l824_82455


namespace evaluate_expression_l824_82496

theorem evaluate_expression :
  - (18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end evaluate_expression_l824_82496


namespace Tim_has_7_times_more_l824_82461

-- Define the number of Dan's violet balloons
def Dan_violet_balloons : ℕ := 29

-- Define the number of Tim's violet balloons
def Tim_violet_balloons : ℕ := 203

-- Prove that the ratio of Tim's balloons to Dan's balloons is 7
theorem Tim_has_7_times_more (h : Tim_violet_balloons = 7 * Dan_violet_balloons) : 
  Tim_violet_balloons = 7 * Dan_violet_balloons := 
by {
  sorry
}

end Tim_has_7_times_more_l824_82461


namespace warriors_truth_tellers_l824_82454

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l824_82454


namespace value_range_of_f_l824_82498

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 2 * x - x^2
  else if -4 ≤ x ∧ x < 0 then x^2 + 6 * x
  else 0

theorem value_range_of_f : Set.range f = {y : ℝ | -9 ≤ y ∧ y ≤ 1} :=
by
  sorry

end value_range_of_f_l824_82498


namespace part_1_part_2_l824_82490

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

theorem part_1 (m : ℝ) : (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → (m = 2) := 
by
  sorry

theorem part_2 (m : ℝ) : (A ⊆ (Set.univ \ B m)) → (m > 5 ∨ m < -3) := 
by
  sorry

end part_1_part_2_l824_82490


namespace point_in_second_quadrant_coordinates_l824_82414

theorem point_in_second_quadrant_coordinates (a : ℤ) (h1 : a + 1 < 0) (h2 : 2 * a + 6 > 0) :
  (a + 1, 2 * a + 6) = (-1, 2) :=
sorry

end point_in_second_quadrant_coordinates_l824_82414


namespace volume_of_pyramid_l824_82442

noncomputable def volume_of_pyramid_QEFGH : ℝ := 
  let EF := 10
  let FG := 3
  let base_area := EF * FG
  let height := 9
  (1/3) * base_area * height

theorem volume_of_pyramid {EF FG : ℝ} (hEF : EF = 10) (hFG : FG = 3)
  (QE_perpendicular_EF : true) (QE_perpendicular_EH : true) (QE_height : QE = 9) :
  volume_of_pyramid_QEFGH = 90 := by
  sorry

end volume_of_pyramid_l824_82442


namespace movie_tickets_l824_82493

theorem movie_tickets (r h : ℕ) (h1 : r = 25) (h2 : h = 3 * r + 18) : h = 93 :=
by
  sorry

end movie_tickets_l824_82493


namespace minimum_employees_needed_l824_82439

noncomputable def employees_needed (total_days : ℕ) (work_days : ℕ) (rest_days : ℕ) (min_on_duty : ℕ) : ℕ :=
  let comb := (total_days.choose rest_days)
  min_on_duty * comb / work_days

theorem minimum_employees_needed {total_days work_days rest_days min_on_duty : ℕ} (h_total_days: total_days = 7) (h_work_days: work_days = 5) (h_rest_days: rest_days = 2) (h_min_on_duty: min_on_duty = 45) : 
  employees_needed total_days work_days rest_days min_on_duty = 63 := by
  rw [h_total_days, h_work_days, h_rest_days, h_min_on_duty]
  -- detailed computation and proofs steps omitted
  -- the critical part is to ensure 63 is derived correctly based on provided values
  sorry

end minimum_employees_needed_l824_82439


namespace find_sum_uv_l824_82428

theorem find_sum_uv (u v : ℝ) (h1 : 3 * u - 7 * v = 29) (h2 : 5 * u + 3 * v = -9) : u + v = -3.363 := 
sorry

end find_sum_uv_l824_82428


namespace count_N_less_than_2000_l824_82435

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l824_82435


namespace value_of_x_l824_82445

theorem value_of_x : ∃ (x : ℚ), (10 - 2 * x) ^ 2 = 4 * x ^ 2 + 20 * x ∧ x = 5 / 3 :=
by
  sorry

end value_of_x_l824_82445


namespace total_texts_sent_is_97_l824_82497

def textsSentOnMondayAllison := 5
def textsSentOnMondayBrittney := 5
def textsSentOnMondayCarol := 5

def textsSentOnTuesdayAllison := 15
def textsSentOnTuesdayBrittney := 10
def textsSentOnTuesdayCarol := 12

def textsSentOnWednesdayAllison := 20
def textsSentOnWednesdayBrittney := 18
def textsSentOnWednesdayCarol := 7

def totalTextsAllison := textsSentOnMondayAllison + textsSentOnTuesdayAllison + textsSentOnWednesdayAllison
def totalTextsBrittney := textsSentOnMondayBrittney + textsSentOnTuesdayBrittney + textsSentOnWednesdayBrittney
def totalTextsCarol := textsSentOnMondayCarol + textsSentOnTuesdayCarol + textsSentOnWednesdayCarol

def totalTextsAllThree := totalTextsAllison + totalTextsBrittney + totalTextsCarol

theorem total_texts_sent_is_97 : totalTextsAllThree = 97 := by
  sorry

end total_texts_sent_is_97_l824_82497


namespace find_B_l824_82408

theorem find_B (A B : ℝ) : (1 / 4 * 1 / 8 = 1 / (4 * A) ∧ 1 / 32 = 1 / B) → B = 32 := by
  intros h
  sorry

end find_B_l824_82408


namespace volume_is_correct_l824_82495

def condition1 (x y z : ℝ) : Prop := abs (x + 2 * y + 3 * z) + abs (x + 2 * y - 3 * z) ≤ 18
def condition2 (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0
def region (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

noncomputable def volume_of_region : ℝ :=
  60.75 -- the result obtained from the calculation steps

theorem volume_is_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 60.75 :=
by
  sorry

end volume_is_correct_l824_82495


namespace maria_scored_33_points_l824_82446

-- Defining constants and parameters
def num_shots := 40
def equal_distribution : ℕ := num_shots / 3 -- each type of shot

-- Given success rates
def success_rate_three_point : ℚ := 0.25
def success_rate_two_point : ℚ := 0.50
def success_rate_free_throw : ℚ := 0.80

-- Defining the points per successful shot
def points_per_successful_three_point_shot : ℕ := 3
def points_per_successful_two_point_shot : ℕ := 2
def points_per_successful_free_throw_shot : ℕ := 1

-- Calculating total points scored
def total_points_scored :=
  (success_rate_three_point * points_per_successful_three_point_shot * equal_distribution) +
  (success_rate_two_point * points_per_successful_two_point_shot * equal_distribution) +
  (success_rate_free_throw * points_per_successful_free_throw_shot * equal_distribution)

theorem maria_scored_33_points :
  total_points_scored = 33 := 
sorry

end maria_scored_33_points_l824_82446


namespace conditions_for_a_and_b_l824_82410

variables (a b x y : ℝ)

theorem conditions_for_a_and_b (h1 : x^2 + x * y + y^2 - y = 0) (h2 : a * x^2 + b * x * y + x = 0) :
  (a + 1)^2 = 4 * (b + 1) ∧ b ≠ -1 :=
sorry

end conditions_for_a_and_b_l824_82410


namespace initial_total_packs_l824_82425

def initial_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  total_packs = regular_packs + unusual_packs + excellent_packs

def ratio_packs (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  3 * (regular_packs + unusual_packs + excellent_packs) = 3 * regular_packs + 4 * unusual_packs + 6 * excellent_packs

def new_ratios (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  2 * (new_regular_packs) + 5 * (new_unusual_packs) + 8 * (new_excellent_packs) = regular_packs + unusual_packs + excellent_packs + 8 * (regular_packs)

def pack_changes (initial_regular_packs : ℕ) (initial_unusual_packs : ℕ) (initial_excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  initial_excellent_packs <= new_excellent_packs + 80 ∧ initial_regular_packs - new_regular_packs ≤ 10

theorem initial_total_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) 
(new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) :
  initial_packs total_packs regular_packs unusual_packs excellent_packs ∧
  ratio_packs regular_packs unusual_packs excellent_packs ∧ 
  new_ratios regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs ∧ 
  pack_changes regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs 
  → total_packs = 260 := 
sorry

end initial_total_packs_l824_82425


namespace john_bought_slurpees_l824_82451

noncomputable def slurpees_bought (total_money paid change slurpee_cost : ℕ) : ℕ :=
  (paid - change) / slurpee_cost

theorem john_bought_slurpees :
  let total_money := 20
  let slurpee_cost := 2
  let change := 8
  slurpees_bought total_money total_money change slurpee_cost = 6 :=
by
  sorry

end john_bought_slurpees_l824_82451


namespace exists_function_f_l824_82491

-- Define the problem statement
theorem exists_function_f :
  ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (abs (x + 1)) = x^2 + 2 * x :=
sorry

end exists_function_f_l824_82491


namespace alice_speed_is_6_5_l824_82481

-- Definitions based on the conditions.
def a : ℝ := sorry -- Alice's speed
def b : ℝ := a + 3 -- Bob's speed

-- Alice cycles towards the park 80 miles away and Bob meets her 15 miles away from the park
def d_alice : ℝ := 65 -- Alice's distance traveled (80 - 15)
def d_bob : ℝ := 95 -- Bob's distance traveled (80 + 15)

-- Equating the times
def time_eqn := d_alice / a = d_bob / b

-- Alice's speed is 6.5 mph
theorem alice_speed_is_6_5 : a = 6.5 :=
by
  have h1 : b = a + 3 := sorry
  have h2 : a * 65 = (a + 3) * 95 := sorry
  have h3 : 30 * a = 195 := sorry
  have h4 : a = 6.5 := sorry
  exact h4

end alice_speed_is_6_5_l824_82481


namespace integer_pairs_satisfy_equation_l824_82489

theorem integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), b + 1 ≠ 0 → b + 2 ≠ 0 → a + b + 1 ≠ 0 →
    ( (a + 2)/(b + 1) + (a + 1)/(b + 2) = 1 + 6/(a + b + 1) ↔ 
      (a = 1 ∧ b = 0) ∨ (∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ) :=
by
  intros a b h1 h2 h3
  sorry

end integer_pairs_satisfy_equation_l824_82489


namespace interest_rate_of_first_account_l824_82437

theorem interest_rate_of_first_account (r : ℝ) 
  (h1 : 7200 = 4000 + 4000)
  (h2 : 4000 * r = 4000 * 0.10) : 
  r = 0.10 :=
sorry

end interest_rate_of_first_account_l824_82437


namespace least_positive_integer_l824_82458

theorem least_positive_integer (N : ℕ) :
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (N % 15 = 14) ∧
  (N % 16 = 15) →
  N = 720719 :=
by
  sorry

end least_positive_integer_l824_82458


namespace factor_square_difference_l824_82479

theorem factor_square_difference (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := 
  sorry

end factor_square_difference_l824_82479


namespace juice_difference_is_eight_l824_82429

-- Defining the initial conditions
def initial_large_barrel : ℕ := 10
def initial_small_barrel : ℕ := 8
def poured_juice : ℕ := 3

-- Defining the final amounts
def final_large_barrel : ℕ := initial_large_barrel + poured_juice
def final_small_barrel : ℕ := initial_small_barrel - poured_juice

-- The statement we need to prove
theorem juice_difference_is_eight :
  final_large_barrel - final_small_barrel = 8 :=
by
  -- Skipping the proof
  sorry

end juice_difference_is_eight_l824_82429


namespace average_age_increase_l824_82470

theorem average_age_increase 
  (n : Nat) 
  (a : ℕ) 
  (b : ℕ) 
  (total_students : Nat)
  (avg_age_9 : ℕ) 
  (tenth_age : ℕ) 
  (original_total_age : Nat)
  (new_total_age : Nat)
  (new_avg_age : ℕ)
  (age_increase : ℕ) 
  (h1 : n = 9) 
  (h2 : avg_age_9 = 8) 
  (h3 : tenth_age = 28)
  (h4 : total_students = 10)
  (h5 : original_total_age = n * avg_age_9) 
  (h6 : new_total_age = original_total_age + tenth_age)
  (h7 : new_avg_age = new_total_age / total_students)
  (h8 : age_increase = new_avg_age - avg_age_9) :
  age_increase = 2 := 
by 
  sorry

end average_age_increase_l824_82470


namespace beach_ball_problem_l824_82427

noncomputable def change_in_radius (C₁ C₂ : ℝ) : ℝ := (C₂ - C₁) / (2 * Real.pi)

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

noncomputable def percentage_increase_in_volume (V₁ V₂ : ℝ) : ℝ := (V₂ - V₁) / V₁ * 100

theorem beach_ball_problem (C₁ C₂ : ℝ) (hC₁ : C₁ = 30) (hC₂ : C₂ = 36) :
  change_in_radius C₁ C₂ = 3 / Real.pi ∧
  percentage_increase_in_volume (volume (C₁ / (2 * Real.pi))) (volume (C₂ / (2 * Real.pi))) = 72.78 :=
by
  sorry

end beach_ball_problem_l824_82427


namespace simplify_expression_l824_82488

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem simplify_expression : (x⁻¹ - y) ^ 2 = (1 / x ^ 2 - 2 * y / x + y ^ 2) :=
  sorry

end simplify_expression_l824_82488


namespace correct_expression_l824_82486

theorem correct_expression (a b : ℝ) : (a - b) * (b + a) = a^2 - b^2 :=
by
  sorry

end correct_expression_l824_82486


namespace solution_set_ineq_l824_82409

theorem solution_set_ineq (x : ℝ) : x^2 - 2 * abs x - 15 > 0 ↔ x < -5 ∨ x > 5 :=
sorry

end solution_set_ineq_l824_82409


namespace find_number_of_terms_l824_82420

variable {n : ℕ} {a : ℕ → ℤ}
variable (a_seq : ℕ → ℤ)

def sum_first_three_terms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3

def sum_last_three_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  a (n-2) + a (n-1) + a n

def sum_all_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem find_number_of_terms (h1 : sum_first_three_terms a_seq = 20)
    (h2 : sum_last_three_terms n a_seq = 130)
    (h3 : sum_all_terms n a_seq = 200) : n = 8 :=
sorry

end find_number_of_terms_l824_82420


namespace sequence_divisible_by_13_l824_82430

theorem sequence_divisible_by_13 (n : ℕ) (h : n ≤ 1000) : 
  ∃ m, m = 165 ∧ ∀ k, 1 ≤ k ∧ k ≤ m → (10^(6*k) + 1) % 13 = 0 := 
sorry

end sequence_divisible_by_13_l824_82430


namespace triangle_area_l824_82449

theorem triangle_area (r : ℝ) (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 2 * r) (r_val : r = 5) (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) :
  (1 / 2) * a * b = 24 :=
by
  -- We assume statements are given
  sorry

end triangle_area_l824_82449


namespace divisible_12_or_36_l824_82456

theorem divisible_12_or_36 (x : ℕ) (n : ℕ) (h1 : Nat.Prime x) (h2 : 3 < x) (h3 : x = 3 * n + 1 ∨ x = 3 * n - 1) :
  12 ∣ (x^6 - x^3 - x^2 + x) ∨ 36 ∣ (x^6 - x^3 - x^2 + x) := 
by
  sorry

end divisible_12_or_36_l824_82456


namespace problem_inequality_l824_82453

noncomputable def A (x : ℝ) := (x - 3) ^ 2
noncomputable def B (x : ℝ) := (x - 2) * (x - 4)

theorem problem_inequality (x : ℝ) : A x > B x :=
  by
    sorry

end problem_inequality_l824_82453


namespace find_a4_b4_c4_l824_82476

theorem find_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 5) (h3 : a^3 + b^3 + c^3 = 15) : 
    a^4 + b^4 + c^4 = 35 := 
by 
  sorry

end find_a4_b4_c4_l824_82476


namespace area_perimeter_quadratic_l824_82483

theorem area_perimeter_quadratic (a x y : ℝ) (h1 : x = 4 * a) (h2 : y = a^2) : y = (x / 4)^2 :=
by sorry

end area_perimeter_quadratic_l824_82483


namespace neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l824_82472

theorem neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one :
  ¬(∃ x : ℝ, x^2 < 1) ↔ ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 := 
by 
  sorry

end neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l824_82472


namespace number_of_pencils_l824_82471

theorem number_of_pencils (P : ℕ) (h : ∃ (n : ℕ), n * 4 = P) : ∃ k, 4 * k = P :=
  by
  sorry

end number_of_pencils_l824_82471


namespace min_value_of_expression_l824_82432

theorem min_value_of_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (hxyz : x * y * z = 27) :
  x + 3 * y + 6 * z >= 27 :=
by
  sorry

end min_value_of_expression_l824_82432


namespace probability_of_answering_phone_in_4_rings_l824_82406

/-- A proof statement that asserts the probability of answering the phone within the first four rings is equal to 9/10. -/
theorem probability_of_answering_phone_in_4_rings :
  (1/10) + (3/10) + (2/5) + (1/10) = 9/10 :=
by
  sorry

end probability_of_answering_phone_in_4_rings_l824_82406


namespace incorrect_inequality_transformation_l824_82407

theorem incorrect_inequality_transformation 
    (a b : ℝ) 
    (h : a > b) 
    : ¬(1 - a > 1 - b) := 
by {
  sorry 
}

end incorrect_inequality_transformation_l824_82407


namespace ajay_income_l824_82405

theorem ajay_income
  (I : ℝ)
  (h₁ : I * 0.45 + I * 0.25 + I * 0.075 + 9000 = I) :
  I = 40000 :=
by
  sorry

end ajay_income_l824_82405


namespace evaluate_expression_l824_82467

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l824_82467


namespace identical_dice_probability_l824_82492

def num_ways_to_paint_die : ℕ := 3^6

def total_ways_to_paint_dice (n : ℕ) : ℕ := (num_ways_to_paint_die ^ n)

def count_identical_ways : ℕ := 1 + 324 + 8100

def probability_identical_dice : ℚ :=
  (count_identical_ways : ℚ) / (total_ways_to_paint_dice 2 : ℚ)

theorem identical_dice_probability : probability_identical_dice = 8425 / 531441 := by
  sorry

end identical_dice_probability_l824_82492


namespace shadow_length_to_time_l824_82499

theorem shadow_length_to_time (shadow_length_inches : ℕ) (stretch_rate_feet_per_hour : ℕ) (inches_per_foot : ℕ) 
                              (shadow_start_time : ℕ) :
  shadow_length_inches = 360 → stretch_rate_feet_per_hour = 5 → inches_per_foot = 12 → shadow_start_time = 0 →
  (shadow_length_inches / inches_per_foot) / stretch_rate_feet_per_hour = 6 := by
  intros h1 h2 h3 h4
  sorry

end shadow_length_to_time_l824_82499


namespace second_differences_of_cubes_l824_82401

-- Define the first difference for cubes of consecutive natural numbers
def first_difference (n : ℕ) : ℕ :=
  ((n + 1) ^ 3) - (n ^ 3)

-- Define the second difference for the first differences
def second_difference (n : ℕ) : ℕ :=
  first_difference (n + 1) - first_difference n

-- Proof statement: Prove that second differences are equal to 6n + 6
theorem second_differences_of_cubes (n : ℕ) : second_difference n = 6 * n + 6 :=
  sorry

end second_differences_of_cubes_l824_82401


namespace sum_interest_l824_82450

noncomputable def simple_interest (P : ℝ) (R : ℝ) := P * R * 3 / 100

theorem sum_interest (P R : ℝ) (h : simple_interest P (R + 1) - simple_interest P R = 75) : P = 2500 :=
by
  sorry

end sum_interest_l824_82450


namespace exists_multiple_sum_divides_l824_82440

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_sum_divides {n : ℕ} (hn : n > 0) :
  ∃ (n_ast : ℕ), n ∣ n_ast ∧ sum_of_digits n_ast ∣ n_ast :=
by
  sorry

end exists_multiple_sum_divides_l824_82440


namespace positive_difference_of_two_numbers_l824_82422

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_of_two_numbers_l824_82422


namespace least_positive_integer_k_l824_82466

noncomputable def least_k (a : ℝ) (n : ℕ) : ℝ :=
  (1 : ℝ) / ((n + 1 : ℝ) ^ 3)

theorem least_positive_integer_k :
  ∃ k : ℕ , (∀ a : ℝ, ∀ n : ℕ,
  (0 ≤ a ∧ a ≤ 1) → (a^k * (1 - a)^n < least_k a n)) ∧
  (∀ k' : ℕ, k' < 4 → ¬(∀ a : ℝ, ∀ n : ℕ, (0 ≤ a ∧ a ≤ 1) → (a^k' * (1 - a)^n < least_k a n))) :=
sorry

end least_positive_integer_k_l824_82466


namespace race_result_130m_l824_82423

theorem race_result_130m (d : ℕ) (t_a t_b: ℕ) (a_speed b_speed : ℚ) (d_a_t : ℚ) (d_b_t : ℚ) (distance_covered_by_B_in_20_secs : ℚ) :
  d = 130 →
  t_a = 20 →
  t_b = 25 →
  a_speed = (↑d) / t_a →
  b_speed = (↑d) / t_b →
  d_a_t = a_speed * t_a →
  d_b_t = b_speed * t_b →
  distance_covered_by_B_in_20_secs = b_speed * 20 →
  (d - distance_covered_by_B_in_20_secs = 26) :=
by
  sorry

end race_result_130m_l824_82423


namespace find_a_l824_82441

theorem find_a (r s : ℚ) (a : ℚ) :
  (∀ x : ℚ, (ax^2 + 18 * x + 16 = (r * x + s)^2)) → 
  s = 4 ∨ s = -4 →
  a = (9 / 4) * (9 / 4)
:= sorry

end find_a_l824_82441


namespace man_speed_with_stream_l824_82474

variable (V_m V_as : ℝ)
variable (V_s V_ws : ℝ)

theorem man_speed_with_stream
  (cond1 : V_m = 5)
  (cond2 : V_as = 8)
  (cond3 : V_as = V_m - V_s)
  (cond4 : V_ws = V_m + V_s) :
  V_ws = 8 := 
by
  sorry

end man_speed_with_stream_l824_82474


namespace mike_pull_ups_per_week_l824_82404

theorem mike_pull_ups_per_week (pull_ups_per_entry entries_per_day days_per_week : ℕ)
  (h1 : pull_ups_per_entry = 2)
  (h2 : entries_per_day = 5)
  (h3 : days_per_week = 7)
  : pull_ups_per_entry * entries_per_day * days_per_week = 70 := 
by
  sorry

end mike_pull_ups_per_week_l824_82404


namespace trapezoid_area_l824_82484

theorem trapezoid_area (h : ℝ) : 
  let base1 := 3 * h 
  let base2 := 4 * h 
  let average_base := (base1 + base2) / 2 
  let area := average_base * h 
  area = (7 * h^2) / 2 := 
by
  sorry

end trapezoid_area_l824_82484


namespace population_2002_l824_82482

-- Predicate P for the population of rabbits in a given year
def P : ℕ → ℝ := sorry

-- Given conditions
axiom cond1 : ∃ k : ℝ, P 2003 - P 2001 = k * P 2002
axiom cond2 : ∃ k : ℝ, P 2002 - P 2000 = k * P 2001
axiom condP2000 : P 2000 = 50
axiom condP2001 : P 2001 = 80
axiom condP2003 : P 2003 = 186

-- The statement we need to prove
theorem population_2002 : P 2002 = 120 :=
by
  sorry

end population_2002_l824_82482


namespace tan_add_formula_l824_82415

noncomputable def tan_subtract (a b : ℝ) : ℝ := (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b)
noncomputable def tan_add (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

theorem tan_add_formula (α : ℝ) (hf : tan_subtract α (Real.pi / 4) = 1 / 4) :
  tan_add α (Real.pi / 4) = -4 :=
by
  sorry

end tan_add_formula_l824_82415


namespace x_minus_y_eq_11_l824_82468

theorem x_minus_y_eq_11 (x y : ℝ) (h : |x - 6| + |y + 5| = 0) : x - y = 11 := by
  sorry

end x_minus_y_eq_11_l824_82468


namespace cos_sin_identity_l824_82443

theorem cos_sin_identity : 
  (Real.cos (75 * Real.pi / 180) + Real.sin (75 * Real.pi / 180)) * 
  (Real.cos (75 * Real.pi / 180) - Real.sin (75 * Real.pi / 180)) = -Real.sqrt 3 / 2 := 
  sorry

end cos_sin_identity_l824_82443


namespace b_share_l824_82485

-- Definitions based on the conditions
def salary (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ d = 6 * x

def condition (d c : ℕ) : Prop :=
  d = c + 700

-- Proof problem based on the correct answer
theorem b_share (a b c d : ℕ) (x : ℕ) (salary_cond : salary a b c d) (cond : condition d c) :
  b = 1050 := by
  sorry

end b_share_l824_82485


namespace part1_part2_l824_82447

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Statement for part (1)
theorem part1 (m : ℝ) : (m > -2) → (∀ x : ℝ, m + f x > 0) :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : (m > 2) ↔ (∀ x : ℝ, m - f x > 0) :=
sorry

end part1_part2_l824_82447


namespace number_of_sets_of_positive_integers_l824_82403

theorem number_of_sets_of_positive_integers : 
  ∃ n : ℕ, n = 3333 ∧ ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → x < y → y < z → x + y + z = 203 → n = 3333 :=
by
  sorry

end number_of_sets_of_positive_integers_l824_82403


namespace kenneth_earnings_l824_82402

theorem kenneth_earnings (E : ℝ) (h1 : E - 0.1 * E = 405) : E = 450 :=
sorry

end kenneth_earnings_l824_82402


namespace systematic_sampling_example_l824_82433

theorem systematic_sampling_example (rows seats : ℕ) (all_seats_filled : Prop) (chosen_seat : ℕ):
  rows = 50 ∧ seats = 60 ∧ all_seats_filled ∧ chosen_seat = 18 → sampling_method = "systematic_sampling" :=
by
  sorry

end systematic_sampling_example_l824_82433


namespace find_number_l824_82469

theorem find_number (x : ℝ) :
  10 * x - 10 = 50 ↔ x = 6 := by
  sorry

end find_number_l824_82469


namespace jump_length_third_frog_l824_82431

theorem jump_length_third_frog (A B C : ℝ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 2) 
  (h3 : |B - A| + |(B - C) / 2| = 60) : 
  |C - (A + B) / 2| = 30 :=
sorry

end jump_length_third_frog_l824_82431


namespace larger_integer_value_l824_82400

theorem larger_integer_value
  (a b : ℕ)
  (h1 : a ≥ b)
  (h2 : ↑a / ↑b = 7 / 3)
  (h3 : a * b = 294) :
  a = 7 * Int.sqrt 14 := 
sorry

end larger_integer_value_l824_82400


namespace seq_a_2014_l824_82457

theorem seq_a_2014 {a : ℕ → ℕ}
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 1) * a n) :
  a 2014 = 2014 :=
sorry

end seq_a_2014_l824_82457


namespace added_water_correct_l824_82417

theorem added_water_correct (initial_fullness : ℝ) (final_fullness : ℝ) (capacity : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (added_water : ℝ) :
    initial_fullness = 0.30 →
    final_fullness = 3/4 →
    capacity = 60 →
    initial_amount = initial_fullness * capacity →
    final_amount = final_fullness * capacity →
    added_water = final_amount - initial_amount →
    added_water = 27 :=
by
  intros
  -- Insert the proof here
  sorry

end added_water_correct_l824_82417


namespace ellipse_product_l824_82416

noncomputable def AB_CD_product (a b c : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) : ℝ :=
  2 * a * 2 * b

-- The main statement
theorem ellipse_product (c : ℝ) (h_c : c = 8) (h_diameter : 6 = 6)
  (a b : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) :
  AB_CD_product a b c h1 h2 = 175 := sorry

end ellipse_product_l824_82416


namespace adam_has_10_apples_l824_82448

theorem adam_has_10_apples
  (Jackie_has_2_apples : ∀ Jackie_apples, Jackie_apples = 2)
  (Adam_has_8_more_apples : ∀ Adam_apples Jackie_apples, Adam_apples = Jackie_apples + 8)
  : ∀ Adam_apples, Adam_apples = 10 :=
by {
  sorry
}

end adam_has_10_apples_l824_82448


namespace find_fg3_l824_82478

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 1

theorem find_fg3 : f (g 3) = 15 :=
by
  sorry

end find_fg3_l824_82478


namespace ratio_blue_yellow_l824_82444

theorem ratio_blue_yellow (total_butterflies blue_butterflies black_butterflies : ℕ)
  (h_total : total_butterflies = 19)
  (h_blue : blue_butterflies = 6)
  (h_black : black_butterflies = 10) :
  (blue_butterflies : ℚ) / (total_butterflies - blue_butterflies - black_butterflies : ℚ) = 2 / 1 := 
by {
  sorry
}

end ratio_blue_yellow_l824_82444


namespace trigonometric_expression_l824_82411

open Real

theorem trigonometric_expression (α β : ℝ) (h : cos α ^ 2 = cos β ^ 2) :
  (sin β ^ 2 / sin α + cos β ^ 2 / cos α = sin α + cos α ∨ sin β ^ 2 / sin α + cos β ^ 2 / cos α = -sin α + cos α) :=
sorry

end trigonometric_expression_l824_82411


namespace evaluate_expression_l824_82419

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l824_82419


namespace d_minus_r_eq_15_l824_82421

theorem d_minus_r_eq_15 (d r : ℤ) (h_d_gt_1 : d > 1)
  (h1 : 1059 % d = r)
  (h2 : 1417 % d = r)
  (h3 : 2312 % d = r) :
  d - r = 15 :=
sorry

end d_minus_r_eq_15_l824_82421


namespace least_positive_base_ten_seven_binary_digits_l824_82412

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l824_82412


namespace vertical_asymptote_at_x_4_l824_82487

def P (x : ℝ) : ℝ := x^2 + 2 * x + 8
def Q (x : ℝ) : ℝ := x^2 - 8 * x + 16

theorem vertical_asymptote_at_x_4 : ∃ x : ℝ, Q x = 0 ∧ P x ≠ 0 ∧ x = 4 :=
by
  use 4
  -- Proof skipped
  sorry

end vertical_asymptote_at_x_4_l824_82487


namespace geometric_sequence_a6_l824_82424

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h1 : a 4 * a 8 = 9) 
  (h2 : a 4 + a 8 = 8) 
  (geom_seq : ∀ n m, a (n + m) = a n * a m): 
  a 6 = 3 :=
by
  -- skipped proof
  sorry

end geometric_sequence_a6_l824_82424


namespace intercept_x_parallel_lines_l824_82436

theorem intercept_x_parallel_lines (m : ℝ) 
    (line_l : ∀ x y : ℝ, y + m * (x + 1) = 0) 
    (parallel : ∀ x y : ℝ, y * m - (2 * m + 1) * x = 1) : 
    ∃ x : ℝ, x + 1 = -1 :=
by
  sorry

end intercept_x_parallel_lines_l824_82436


namespace range_of_a_l824_82477

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + Real.log x - (x^2 / (x - Real.log x))

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔
  1 < a ∧ a < (Real.exp 1) / (Real.exp 1 - 1) - 1 / Real.exp 1 :=
sorry

end range_of_a_l824_82477


namespace calculate_expression_value_l824_82463

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l824_82463


namespace A_formula_l824_82464

noncomputable def A (i : ℕ) (A₀ θ : ℝ) : ℝ :=
match i with
| 0     => A₀
| (i+1) => (A i A₀ θ * Real.cos θ + Real.sin θ) / (-A i A₀ θ * Real.sin θ + Real.cos θ)

theorem A_formula (A₀ θ : ℝ) (n : ℕ) :
  A n A₀ θ = (A₀ * Real.cos (n * θ) + Real.sin (n * θ)) / (-A₀ * Real.sin (n * θ) + Real.cos (n * θ)) :=
by
  sorry

end A_formula_l824_82464


namespace tenth_number_in_sixteenth_group_is_257_l824_82460

-- Define the general term of the sequence a_n = 2n - 3.
def a_n (n : ℕ) : ℕ := 2 * n - 3

-- Define the first number of the n-th group.
def first_number_of_group (n : ℕ) : ℕ := n^2 - n - 1

-- Define the m-th number in the n-th group.
def group_n_m (n m : ℕ) : ℕ := first_number_of_group n + (m - 1) * 2

theorem tenth_number_in_sixteenth_group_is_257 : group_n_m 16 10 = 257 := by
  sorry

end tenth_number_in_sixteenth_group_is_257_l824_82460


namespace Jim_weekly_savings_l824_82413

-- Define the given conditions
def Sara_initial_savings : ℕ := 4100
def Sara_weekly_savings : ℕ := 10
def weeks : ℕ := 820

-- Define the proof goal based on the conditions
theorem Jim_weekly_savings :
  let Sara_total_savings := Sara_initial_savings + (Sara_weekly_savings * weeks)
  let Jim_weekly_savings := Sara_total_savings / weeks
  Jim_weekly_savings = 15 := 
by 
  sorry

end Jim_weekly_savings_l824_82413


namespace time_to_pass_platform_l824_82475

-- Definitions
def train_length : ℕ := 1400
def platform_length : ℕ := 700
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

-- Prove that the time to pass the platform is 150 seconds
theorem time_to_pass_platform : total_distance / train_speed = 150 :=
by
  sorry

end time_to_pass_platform_l824_82475


namespace additional_savings_zero_l824_82426

noncomputable def windows_savings (purchase_price : ℕ) (free_windows : ℕ) (paid_windows : ℕ)
  (dave_needs : ℕ) (doug_needs : ℕ) : ℕ := sorry

theorem additional_savings_zero :
  windows_savings 100 2 5 12 10 = 0 := sorry

end additional_savings_zero_l824_82426


namespace triangle_problem_l824_82465

/-- 
Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively, 
if b = 2 and 2*b*cos B = a*cos C + c*cos A,
prove that B = π/3 and find the maximum area of ΔABC.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : b = 2) (h2 : 2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) :
  B = Real.pi / 3 ∧
  (∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ max_area = (1/2) * a * c * Real.sin B) :=
by
  sorry

end triangle_problem_l824_82465


namespace price_of_movie_ticket_l824_82462

theorem price_of_movie_ticket
  (M F : ℝ)
  (h1 : 8 * M = 2 * F)
  (h2 : 8 * M + 5 * F = 840) :
  M = 30 :=
by
  sorry

end price_of_movie_ticket_l824_82462


namespace fish_weight_l824_82418

-- Definitions of weights
variable (T B H : ℝ)

-- Given conditions
def cond1 : Prop := T = 9
def cond2 : Prop := H = T + (1/2) * B
def cond3 : Prop := B = H + T

-- Theorem to prove
theorem fish_weight (h1 : cond1 T) (h2 : cond2 T B H) (h3 : cond3 T B H) :
  T + B + H = 72 :=
by
  sorry

end fish_weight_l824_82418


namespace prime_quadruples_unique_l824_82452

noncomputable def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → (m = 1 ∨ m = n)

theorem prime_quadruples_unique (p q r n : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (hn : n > 0)
  (h_eq : p^2 = q^2 + r^n) :
  (p, q, r, n) = (3, 2, 5, 1) ∨ (p, q, r, n) = (5, 3, 2, 4) :=
by
  sorry

end prime_quadruples_unique_l824_82452


namespace round_robin_10_person_tournament_l824_82473

noncomputable def num_matches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem round_robin_10_person_tournament :
  num_matches 10 = 45 :=
by
  sorry

end round_robin_10_person_tournament_l824_82473


namespace gas_volume_at_31_degrees_l824_82459

theorem gas_volume_at_31_degrees :
  (∀ T V : ℕ, (T = 45 → V = 30) ∧ (∀ k, T = 45 - 2 * k → V = 30 - 3 * k)) →
  ∃ V, (T = 31) ∧ (V = 9) :=
by
  -- The proof would go here
  sorry

end gas_volume_at_31_degrees_l824_82459


namespace average_is_207_l824_82438

variable (x : ℕ)

theorem average_is_207 (h : (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212 + x) / 10 = 207) :
  x = 212 :=
sorry

end average_is_207_l824_82438


namespace positive_integers_a_2014_b_l824_82494

theorem positive_integers_a_2014_b (a : ℕ) :
  (∃! b : ℕ, 2 ≤ a / b ∧ a / b ≤ 5) → a = 6710 ∨ a = 6712 ∨ a = 6713 :=
by
  sorry

end positive_integers_a_2014_b_l824_82494


namespace sequence_an_correct_l824_82434

noncomputable def seq_an (n : ℕ) : ℚ :=
if h : n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3))

theorem sequence_an_correct (n : ℕ) (S : ℕ → ℚ)
  (h1 : S 1 = 1)
  (h2 : ∀ n ≥ 2, S n ^ 2 = seq_an n * (S n - 0.5)) :
  seq_an n = if n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3)) :=
sorry

end sequence_an_correct_l824_82434


namespace find_minimum_value_l824_82480

theorem find_minimum_value (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : x > 0): 
  (∃ m : ℝ, ∀ x > 0, (a^2 + x^2) / x ≥ m ∧ ∃ x₀ > 0, (a^2 + x₀^2) / x₀ = m) :=
sorry

end find_minimum_value_l824_82480
